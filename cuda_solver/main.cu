#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <algorithm>

#define K 9
#define COPS 3
#define THREADS_PER_BLOCK 256
#define TT_SIZE_LOG2 26
#define TT_ENTRIES (1ULL << TT_SIZE_LOG2)

struct AdjMatrix {
    uint16_t rows[16];
};

__device__ __host__ void init_matrix(AdjMatrix* m) {
    for (int i = 0; i < 16; i++) m->rows[i] = 0;
}

__device__ __host__ void add_edge(AdjMatrix* m, int u, int v) {
    if (u >= 0 && u < 16 && v >= 0 && v < 16) {
        m->rows[u] |= (1 << v);
        m->rows[v] |= (1 << u);
    }
}

__device__ __host__ bool has_edge(const AdjMatrix* m, int u, int v) {
    return (m->rows[u] & (1 << v)) != 0;
}

__host__ int host_ffs(uint16_t x) {
    if (x == 0) return 0;
    return __builtin_ffs(x);
}

__device__ __host__ bool is_0_1_connected(const AdjMatrix* m) {
    uint16_t visited = 1;
    uint16_t frontier = 1;
    int iters = 0;
    while (frontier != 0 && iters < 16) {
        if ((visited & 2) != 0) return true;
        uint16_t next_frontier = 0;
        uint16_t f = frontier;
        while (f != 0) {
#ifdef __CUDA_ARCH__
            int bit = __ffs(f) - 1;
#else
            int bit = host_ffs(f) - 1;
#endif
            if (bit >= 0 && bit < 16) {
                next_frontier |= m->rows[bit];
            }
            f &= f - 1;
        }
        next_frontier &= ~visited;
        visited |= next_frontier;
        frontier = next_frontier;
        iters++;
    }
    return (visited & 2) != 0;
}

struct TTEntry {
    uint64_t word0;
    uint64_t word1;
};

__device__ uint64_t splitmix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ void get_hash(const AdjMatrix* blue, const AdjMatrix* red, bool is_cop, int picks_left, uint64_t* h0, uint64_t* h1) {
    uint64_t state_hash = 0;
    for(int i=0; i<K; i++) {
        state_hash ^= ((uint64_t)blue->rows[i]) << (i*2);
        state_hash ^= ((uint64_t)red->rows[i]) << (i*2 + 1);
    }
    if (is_cop) state_hash ^= 0x123456789ABCDEF0ULL;
    state_hash ^= ((uint64_t)picks_left << 48);
    
    *h0 = splitmix64(state_hash);
    *h1 = splitmix64(state_hash ^ 0x517cc1b727220a95ULL);
}

__device__ void tt_store(TTEntry* tt, uint64_t h0, uint64_t h1, int score, int flag, int depth) {
    uint64_t idx = h0 & (TT_ENTRIES - 1);
    
    uint64_t s = score + 1; // map -1,0,1 to 0,1,2
    uint64_t f = flag; // 0=exact, 1=lower, 2=upper
    uint64_t d = depth;
    
    uint64_t payload = (s & 0xFF) | ((f & 0x3) << 8) | ((d & 0x3F) << 10);
    
    uint64_t word0 = (h0 & ~0xFFFFULL) | payload;
    if (word0 == 0) word0 = 1; // prevent 0
    uint64_t word1 = h1 ^ word0;
    
    tt[idx].word0 = word0;
    tt[idx].word1 = word1;
}

__device__ bool tt_probe(TTEntry* tt, uint64_t h0, uint64_t h1, int* score, int* flag, int depth) {
    uint64_t idx = h0 & (TT_ENTRIES - 1);
    uint64_t word0 = tt[idx].word0;
    if (word0 == 0) return false;
    
    uint64_t word1 = tt[idx].word1;
    
    if ((word0 & ~0xFFFFULL) == (h0 & ~0xFFFFULL) && (word1 ^ word0) == h1) {
        int d = (word0 >> 10) & 0x3F;
        if (d >= depth) {
            *score = (int)(word0 & 0xFF) - 1;
            *flag = (word0 >> 8) & 0x3;
            return true;
        }
    }
    return false;
}

// Global metrics
__device__ unsigned long long nodes_evaluated = 0;
__device__ unsigned long long tt_hits = 0;

__device__ int device_alpha_beta(AdjMatrix blue, AdjMatrix red, int depth, int alpha, int beta, bool is_cop, int picks_left, TTEntry* tt) {
    atomicAdd(&nodes_evaluated, 1);
    
    if (is_0_1_connected(&red)) return -1;
    
    AdjMatrix available;
    init_matrix(&available);
    uint16_t mask = (1 << K) - 1;
    for(int i=0; i<K; i++) {
        available.rows[i] = mask & ~(blue.rows[i]) & ~(1 << i);
    }
    if (!is_0_1_connected(&available)) return 1; 
    
    if (depth <= 0) return 0; 
    
    uint64_t h0, h1;
    get_hash(&blue, &red, is_cop, picks_left, &h0, &h1);
    
    int tt_score, tt_flag;
    if (tt_probe(tt, h0, h1, &tt_score, &tt_flag, depth)) {
        atomicAdd(&tt_hits, 1);
        if (tt_flag == 0) return tt_score;
        if (tt_flag == 1 && tt_score > alpha) alpha = tt_score;
        if (tt_flag == 2 && tt_score < beta) beta = tt_score;
        if (alpha >= beta) return tt_score;
    }
    
    uint8_t edges_u[120];
    uint8_t edges_v[120];
    int edge_scores[120];
    int num_edges = 0;
    
    for (int u = 0; u < K; u++) {
        uint16_t avail = mask & ~blue.rows[u] & ~red.rows[u];
        avail &= ~((1 << (u + 1)) - 1); 
        while (avail != 0) {
            int v = __ffs(avail) - 1;
            if (num_edges < 120 && v >= 0 && v < 16) {
                int score = 0;
                
                // Extremely greedy edge sorting metric
                if (u <= 1 && v <= 1) {
                    score = 10000; // direct 0-1 edge is critical
                } else if (u <= 1 || v <= 1) {
                    score = 1000;
                }
                
                // Compute degree in the red graph as a proxy for "growing a component"
                int red_deg_u = __popc((unsigned int)red.rows[u]);
                int red_deg_v = __popc((unsigned int)red.rows[v]);
                score += (red_deg_u + red_deg_v) * 10;
                
                // Compute degree in the blue graph as a proxy for "already defended"
                int blue_deg_u = __popc((unsigned int)blue.rows[u]);
                int blue_deg_v = __popc((unsigned int)blue.rows[v]);
                score += (blue_deg_u + blue_deg_v) * 5;
                
                // Insertion sort
                int k = num_edges;
                while (k > 0 && edge_scores[k-1] < score) {
                    edges_u[k] = edges_u[k-1];
                    edges_v[k] = edges_v[k-1];
                    edge_scores[k] = edge_scores[k-1];
                    k--;
                }
                edges_u[k] = u;
                edges_v[k] = v;
                edge_scores[k] = score;
                num_edges++;
            }
            avail &= avail - 1;
        }
    }
    
    if (num_edges == 0) return 1; 
    
    int orig_alpha = alpha;
    int best;
    
    if (is_cop) {
        best = -999;
        for (int i = 0; i < num_edges; i++) {
            AdjMatrix next_blue = blue;
            add_edge(&next_blue, edges_u[i], edges_v[i]);
            
            int score;
            if (picks_left > 1) {
                score = device_alpha_beta(next_blue, red, depth, alpha, beta, true, picks_left - 1, tt);
            } else {
                score = device_alpha_beta(next_blue, red, depth - 1, alpha, beta, false, 0, tt);
            }
            
            if (score > best) best = score;
            if (best > alpha) alpha = best;
            if (alpha >= beta) break; 
        }
    } else {
        best = 999;
        for (int i = 0; i < num_edges; i++) {
            AdjMatrix next_red = red;
            add_edge(&next_red, edges_u[i], edges_v[i]);
            
            int score = device_alpha_beta(blue, next_red, depth - 1, alpha, beta, true, COPS, tt);
            
            if (score < best) best = score;
            if (best < beta) beta = best;
            if (alpha >= beta) break; 
        }
    }
    
    int flag = 0; 
    if (best <= orig_alpha) flag = 2; // upperbound
    else if (best >= beta) flag = 1; // lowerbound
    
    tt_store(tt, h0, h1, best, flag, depth);
    return best;
}

struct Job {
    AdjMatrix blue;
    AdjMatrix red;
    bool is_cop;
    int picks_left;
};

__global__ void evaluate_jobs_kernel(Job* jobs, int* results, int num_jobs, TTEntry* tt, int max_depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_jobs) return;
    
    Job j = jobs[idx];
    // Each root job explores with full Alpha-Beta window (-999, 999)
    results[idx] = device_alpha_beta(j.blue, j.red, max_depth, -999, 999, j.is_cop, j.picks_left, tt);
}

struct Edge { int u, v; };

std::vector<Edge> get_remaining_edges(const AdjMatrix& blue, const AdjMatrix& red) {
    std::vector<Edge> edges;
    uint16_t mask = (1 << K) - 1;
    for (int u = 0; u < K; u++) {
        uint16_t avail = mask & ~blue.rows[u] & ~red.rows[u];
        avail &= ~((1 << (u + 1)) - 1); 
        while (avail != 0) {
            int v = host_ffs(avail) - 1;
            edges.push_back({u, v});
            avail &= avail - 1;
        }
    }
    return edges;
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    printf("Initializing Optimized GPU Parallel Alpha-Beta Solver for K%d (%d Cops)\n", K, COPS);
    
    AdjMatrix root_blue, root_red;
    init_matrix(&root_blue);
    init_matrix(&root_red);
    
    std::vector<Job> jobs;
    std::vector<Edge> edges = get_remaining_edges(root_blue, root_red);
    
    // Sort edges so 0-1 threats are handled first even at the root!
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        int a_score = 0;
        if (a.u <= 1 && a.v <= 1) a_score += 10000;
        else if (a.u <= 1 || a.v <= 1) a_score += 1000;
        
        int b_score = 0;
        if (b.u <= 1 && b.v <= 1) b_score += 10000;
        else if (b.u <= 1 || b.v <= 1) b_score += 1000;
        
        return a_score > b_score;
    });
    
    // Generate all Cop 1st turn combinations
    if (COPS == 3) {
        for(int i=0; i<edges.size(); i++) {
            for(int j=i+1; j<edges.size(); j++) {
                for(int k=j+1; k<edges.size(); k++) {
                    Job job;
                    init_matrix(&job.blue);
                    init_matrix(&job.red);
                    add_edge(&job.blue, edges[i].u, edges[i].v);
                    add_edge(&job.blue, edges[j].u, edges[j].v);
                    add_edge(&job.blue, edges[k].u, edges[k].v);
                    job.is_cop = false; 
                    job.picks_left = 0;
                    jobs.push_back(job);
                }
            }
        }
    } else {
        printf("Set COPS=3 for this layout.\n");
        return 1;
    }
    
    int num_jobs = jobs.size();
    printf("CPU generated %d parallel root jobs.\n", num_jobs);

    size_t tt_bytes = TT_ENTRIES * sizeof(TTEntry);
    printf("Allocating %.2f GB for GPU Transposition Table...\n", tt_bytes / (1024.0*1024*1024));
    
    TTEntry* d_tt;
    cudaMalloc(&d_tt, tt_bytes);
    
    Job* d_jobs;
    int* d_results;
    cudaMalloc(&d_jobs, num_jobs * sizeof(Job));
    cudaMalloc(&d_results, num_jobs * sizeof(int));
    
    cudaMemcpy(d_jobs, jobs.data(), num_jobs * sizeof(Job), cudaMemcpyHostToDevice);
    
    int blocks = (num_jobs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaDeviceSetLimit(cudaLimitStackSize, 65536); 
    
    for (int depth = 1; depth <= 12; depth++) {
        printf("\n[Depth %d] Launching GPU Alpha-Beta search...\n", depth);
        if (depth == 1) cudaMemset(d_tt, 0, tt_bytes); 
        
        unsigned long long zero = 0;
        cudaMemcpyToSymbol(nodes_evaluated, &zero, sizeof(unsigned long long));
        cudaMemcpyToSymbol(tt_hits, &zero, sizeof(unsigned long long));
        
        double t1 = get_time();
        evaluate_jobs_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_jobs, d_results, num_jobs, d_tt, depth);
        cudaDeviceSynchronize();
        double t2 = get_time();
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            return 1;
        }
        
        std::vector<int> results(num_jobs);
        cudaMemcpy(results.data(), d_results, num_jobs * sizeof(int), cudaMemcpyDeviceToHost);
        
        unsigned long long h_nodes = 0, h_hits = 0;
        cudaMemcpyFromSymbol(&h_nodes, nodes_evaluated, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
        cudaMemcpyFromSymbol(&h_hits, tt_hits, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
        
        int cop_wins = 0, rob_wins = 0, draws = 0;
        for (int r : results) {
            if (r == 1) cop_wins++;
            else if (r == -1) rob_wins++;
            else draws++;
        }
        
        printf("  -> Cop forced wins: %d | Robber forced wins: %d | Unknowns: %d\n", cop_wins, rob_wins, draws);
        printf("  -> Time: %.3f seconds | Nodes: %llu | TT Hits: %llu | Throughput: %.2f M nodes/s\n", 
            t2 - t1, h_nodes, h_hits, (h_nodes / 1000000.0) / (t2 - t1));
            
        if (cop_wins > 0) {
            printf("\n*** COP WINS! Found a winning opening branch at depth %d ***\n", depth);
            break;
        }
        if (rob_wins == num_jobs) {
            printf("\n*** ROBBER WINS! Robber survives all Cop openings at depth %d ***\n", depth);
            break;
        }
    }
    
    cudaFree(d_tt);
    cudaFree(d_jobs);
    cudaFree(d_results);
    
    return 0;
}