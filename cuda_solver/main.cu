#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <sys/time.h>

#define K 9
#define COPS 3
#define GPU_SEARCH_DEPTH 5
#define CPU_EXPANSION_DEPTH 2
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

// Builtin FFS for Host
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

// GPU Transposition Table
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

__device__ void tt_store(TTEntry* tt, uint64_t h0, uint64_t h1, int score) {
    uint64_t idx = h0 & (TT_ENTRIES - 1);
    uint64_t victor_bit = (score == 1) ? 1 : 0;
    uint64_t word0 = (h0 & ~1ULL) | victor_bit;
    uint64_t word1 = h1 ^ word0;
    tt[idx].word0 = word0;
    tt[idx].word1 = word1;
}

__device__ bool tt_probe(TTEntry* tt, uint64_t h0, uint64_t h1, int* score) {
    uint64_t idx = h0 & (TT_ENTRIES - 1);
    uint64_t word0 = tt[idx].word0;
    if (word0 == 0) return false;
    uint64_t word1 = tt[idx].word1;
    if ((word0 & ~1ULL) == (h0 & ~1ULL) && (word1 ^ word0) == h1) {
        *score = (word0 & 1) ? 1 : -1;
        return true;
    }
    return false;
}

__device__ int device_minimax(AdjMatrix blue, AdjMatrix red, int depth, bool is_cop, int picks_left, TTEntry* tt) {
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
    
    int tt_score;
    if (tt_probe(tt, h0, h1, &tt_score)) {
        return tt_score;
    }
    
    uint8_t edges_u[45];
    uint8_t edges_v[45];
    int num_edges = 0;
    
    for (int u = 0; u < K; u++) {
        uint16_t avail = mask & ~blue.rows[u] & ~red.rows[u];
        avail &= ~((1 << (u + 1)) - 1); 
        while (avail != 0) {
            int v = __ffs(avail) - 1;
            if (num_edges < 45 && v >= 0 && v < 16) {
                edges_u[num_edges] = u;
                edges_v[num_edges] = v;
                num_edges++;
            }
            avail &= avail - 1;
        }
    }
    
    if (num_edges == 0) return 1; 
    
    int val;
    if (is_cop) {
        int best = -1;
        for (int i = 0; i < num_edges; i++) {
            AdjMatrix next_blue = blue;
            add_edge(&next_blue, edges_u[i], edges_v[i]);
            
            int score;
            if (picks_left > 1) {
                score = device_minimax(next_blue, red, depth, true, picks_left - 1, tt);
            } else {
                score = device_minimax(next_blue, red, depth - 1, false, 0, tt);
            }
            
            if (score > best) best = score;
            if (best == 1) break; 
        }
        val = best;
    } else {
        int best = 1;
        for (int i = 0; i < num_edges; i++) {
            AdjMatrix next_red = red;
            add_edge(&next_red, edges_u[i], edges_v[i]);
            
            int score = device_minimax(blue, next_red, depth - 1, true, COPS, tt);
            if (score < best) best = score;
            if (best == -1) break; 
        }
        val = best;
    }
    
    if (val != 0) {
        tt_store(tt, h0, h1, val);
    }
    return val;
}

// Job struct passed from CPU to GPU
struct Job {
    AdjMatrix blue;
    AdjMatrix red;
    bool is_cop;
    int picks_left;
};

__global__ void evaluate_jobs_kernel(Job* jobs, int* results, int num_jobs, TTEntry* tt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_jobs) return;
    
    Job j = jobs[idx];
    results[idx] = device_minimax(j.blue, j.red, GPU_SEARCH_DEPTH, j.is_cop, j.picks_left, tt);
}

// ---------------------------------------------------------------------------------------
// CPU Host Code (Boss Node)
// ---------------------------------------------------------------------------------------

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

// Expand the tree on the CPU down to a certain depth to generate thousands of independent branches
void cpu_expand_tree(AdjMatrix blue, AdjMatrix red, int depth, bool is_cop, int picks_left, std::vector<Job>& jobs) {
    if (is_0_1_connected(&red)) return; // Dead branch
    
    AdjMatrix available;
    init_matrix(&available);
    uint16_t mask = (1 << K) - 1;
    for(int i=0; i<K; i++) {
        available.rows[i] = mask & ~(blue.rows[i]) & ~(1 << i);
    }
    if (!is_0_1_connected(&available)) return; // Dead branch
    
    if (depth <= 0) {
        Job j;
        j.blue = blue;
        j.red = red;
        j.is_cop = is_cop;
        j.picks_left = picks_left;
        jobs.push_back(j);
        return;
    }
    
    std::vector<Edge> edges = get_remaining_edges(blue, red);
    if (edges.empty()) return;
    
    for (const auto& e : edges) {
        if (is_cop) {
            AdjMatrix nb = blue;
            add_edge(&nb, e.u, e.v);
            if (picks_left > 1) {
                cpu_expand_tree(nb, red, depth, true, picks_left - 1, jobs);
            } else {
                cpu_expand_tree(nb, red, depth - 1, false, 0, jobs);
            }
        } else {
            AdjMatrix nr = red;
            add_edge(&nr, e.u, e.v);
            cpu_expand_tree(blue, nr, depth - 1, true, COPS, jobs);
        }
    }
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    printf("Initializing CPU-GPU Hybrid Solver for K%d (%d Cops)\n", K, COPS);
    printf("CPU Expansion Depth: %d | GPU Minimax Depth: %d\n", CPU_EXPANSION_DEPTH, GPU_SEARCH_DEPTH);
    
    AdjMatrix root_blue, root_red;
    init_matrix(&root_blue);
    init_matrix(&root_red);
    
    // 1. CPU generates thousands of independent jobs
    double t0 = get_time();
    std::vector<Job> jobs;
    cpu_expand_tree(root_blue, root_red, CPU_EXPANSION_DEPTH, true, COPS, jobs);
    
    int num_jobs = jobs.size();
    printf("CPU generated %d parallel jobs in %.3f seconds.\n", num_jobs, get_time() - t0);
    
    if (num_jobs == 0) {
        printf("Game ends within CPU expansion depth!\n");
        return 0;
    }

    // 2. Allocate memory on GPU
    size_t tt_bytes = TT_ENTRIES * sizeof(TTEntry);
    printf("Allocating %.2f GB for Lockless GPU Transposition Table...\n", tt_bytes / (1024.0*1024*1024));
    
    TTEntry* d_tt;
    cudaMalloc(&d_tt, tt_bytes);
    cudaMemset(d_tt, 0, tt_bytes);
    
    Job* d_jobs;
    int* d_results;
    cudaMalloc(&d_jobs, num_jobs * sizeof(Job));
    cudaMalloc(&d_results, num_jobs * sizeof(int));
    
    cudaMemcpy(d_jobs, jobs.data(), num_jobs * sizeof(Job), cudaMemcpyHostToDevice);
    
    int blocks = (num_jobs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaDeviceSetLimit(cudaLimitStackSize, 65536);
    
    // 3. Launch the massive parallel evaluator!
    printf("Launching 5,000+ CUDA Cores to evaluate %d trees...\n", num_jobs);
    double t1 = get_time();
    
    evaluate_jobs_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_jobs, d_results, num_jobs, d_tt);
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    double t2 = get_time();
    printf("GPU Kernel finished in %.3f seconds.\n", t2 - t1);
    
    // 4. Retrieve results
    std::vector<int> results(num_jobs);
    cudaMemcpy(results.data(), d_results, num_jobs * sizeof(int), cudaMemcpyDeviceToHost);
    
    int cop_wins = 0, rob_wins = 0, draws = 0;
    for (int r : results) {
        if (r == 1) cop_wins++;
        else if (r == -1) rob_wins++;
        else draws++;
    }
    
    printf("\n=== FRONTIER RESULTS ===\n");
    printf("Cop proved forced win on %d branches.\n", cop_wins);
    printf("Robber proved forced win on %d branches.\n", rob_wins);
    printf("Search hit max depth (Draw/Unknown) on %d branches.\n", draws);
    
    cudaFree(d_tt);
    cudaFree(d_jobs);
    cudaFree(d_results);
    
    return 0;
}
