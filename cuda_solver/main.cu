#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <algorithm>

// Settings for K9 / 3 Cops
#define K 9
#define COPS 3
#define THREADS_PER_BLOCK 256

// 8 GB Transposition Table (Fits in 15GB VRAM)
#define TT_SIZE_LOG2 29
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
            int bit = __builtin_ffs(f) - 1;
#endif
            if (bit >= 0 && bit < 16) next_frontier |= m->rows[bit];
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

__device__ __host__ uint64_t splitmix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ __host__ void get_hash(const AdjMatrix* blue, const AdjMatrix* red, bool is_cop, int picks_left, uint64_t* h0, uint64_t* h1) {
    uint64_t h = 0x123456789ABCDEF0ULL;
    for(int i=0; i<K; i++) {
        h ^= splitmix64(blue->rows[i] + (h << 7));
        h ^= splitmix64(red->rows[i] + (h >> 3));
    }
    if (is_cop) h ^= 0x5555555555555555ULL;
    h ^= splitmix64(picks_left + 0xAAAAAAAABBBBBBBBULL);
    *h0 = splitmix64(h);
    *h1 = splitmix64(h ^ 0xDEADBEEFCAFEBABEULL);
}

__device__ void tt_store(TTEntry* tt, uint64_t h0, uint64_t h1, int score, int flag, int depth) {
    uint64_t idx = h0 & (TT_ENTRIES - 1);
    uint64_t payload = ((uint64_t)(score + 1) & 0xFF) | (((uint64_t)flag & 0x3) << 8) | (((uint64_t)depth & 0x3F) << 10);
    uint64_t word0 = (h0 & 0xFFFFFFFF00000000ULL) | payload;
    if (word0 == 0) word0 = 1;
    uint64_t word1 = h1 ^ word0;
    tt[idx].word0 = word0;
    tt[idx].word1 = word1;
}

__device__ bool tt_probe(TTEntry* tt, uint64_t h0, uint64_t h1, int* score, int* flag, int depth) {
    uint64_t idx = h0 & (TT_ENTRIES - 1);
    uint64_t word0 = tt[idx].word0;
    if (word0 == 0) return false;
    uint64_t word1 = tt[idx].word1;
    if ((word1 ^ word0) != h1) return false;
    if ((word0 & 0xFFFFFFFF00000000ULL) != (h0 & 0xFFFFFFFF00000000ULL)) return false;
    int d = (int)((word0 >> 10) & 0x3F);
    int s = (int)(word0 & 0xFF) - 1;
    if (s == 1 || s == -1) { *score = s; *flag = 0; return true; }
    if (d >= depth) { *score = s; *flag = (int)((word0 >> 8) & 0x3); return true; }
    return false;
}

__device__ unsigned long long nodes_evaluated = 0;

__device__ int device_alpha_beta(AdjMatrix blue, AdjMatrix red, int depth, int alpha, int beta, bool is_cop, int picks_left, TTEntry* tt) {
    atomicAdd(&nodes_evaluated, 1);
    if (is_0_1_connected(&red)) return -1;
    AdjMatrix available; init_matrix(&available);
    uint16_t mask = (1 << K) - 1;
    for(int i=0; i<K; i++) available.rows[i] = mask & ~(blue.rows[i]) & ~(1 << i);
    if (!is_0_1_connected(&available)) return 1; 
    if (depth <= 0) return 0; 

    uint64_t h0, h1; get_hash(&blue, &red, is_cop, picks_left, &h0, &h1);
    int tt_score, tt_flag;
    if (tt_probe(tt, h0, h1, &tt_score, &tt_flag, depth)) {
        if (tt_flag == 0) return tt_score;
        if (tt_flag == 1 && tt_score > alpha) alpha = tt_score;
        if (tt_flag == 2 && tt_score < beta) beta = tt_score;
        if (alpha >= beta) return tt_score;
    }

    uint8_t edges_u[120], edges_v[120]; int num_edges = 0;
    for (int u = 0; u < K; u++) {
        uint16_t avail = mask & ~blue.rows[u] & ~red.rows[u];
        avail &= ~((1 << (u + 1)) - 1); 
        while (avail != 0) {
            int v = __ffs(avail) - 1;
            if (num_edges < 120 && v >= 0) { edges_u[num_edges] = u; edges_v[num_edges] = v; num_edges++; }
            avail &= avail - 1;
        }
    }
    if (num_edges == 0) return 1; 

    int orig_alpha = alpha, orig_beta = beta, best;
    if (is_cop) {
        best = -999;
        for (int i = 0; i < num_edges; i++) {
            AdjMatrix next_blue = blue; add_edge(&next_blue, edges_u[i], edges_v[i]);
            int score = (picks_left > 1) 
                ? device_alpha_beta(next_blue, red, depth, alpha, beta, true, picks_left - 1, tt)
                : device_alpha_beta(next_blue, red, depth - 1, alpha, beta, false, 0, tt);
            if (score > best) best = score;
            if (best > alpha) alpha = best;
            if (alpha >= beta) break; 
        }
    } else {
        best = 999;
        for (int i = 0; i < num_edges; i++) {
            AdjMatrix next_red = red; add_edge(&next_red, edges_u[i], edges_v[i]);
            int score = device_alpha_beta(blue, next_red, depth - 1, alpha, beta, true, COPS, tt);
            if (score < best) best = score;
            if (best < beta) beta = best;
            if (alpha >= beta) break; 
        }
    }
    int flag = 0; 
    if (best <= orig_alpha) flag = 2; else if (best >= orig_beta) flag = 1;
    tt_store(tt, h0, h1, best, flag, depth);
    return best;
}

struct Job { AdjMatrix blue; AdjMatrix red; bool is_cop; int picks_left; };
__global__ void evaluate_jobs_kernel(Job* jobs, int* results, int num_jobs, TTEntry* tt, int max_depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_jobs) return;
    results[idx] = device_alpha_beta(jobs[idx].blue, jobs[idx].red, max_depth, -999, 999, jobs[idx].is_cop, jobs[idx].picks_left, tt);
}

struct Edge { int u, v; };
double get_time() { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec + tv.tv_usec * 1e-6; }

int main() {
    printf("Initializing n-cop GPU Hybrid Solver (K%d, %d Cops)\n", K, COPS);
    AdjMatrix rb, rr; init_matrix(&rb); init_matrix(&rr);
    std::vector<Job> jobs;
    uint16_t mask = (1 << K) - 1;
    std::vector<Edge> edges;
    for (int u = 0; u < K; u++) {
        uint16_t avail = mask & ~((1 << (u + 1)) - 1);
        while (avail != 0) { int v = __builtin_ffs(avail) - 1; edges.push_back({u, v}); avail &= avail - 1; }
    }
    // Simple 3-cop opening combinations generator
    for(int i=0; i<edges.size(); i++) {
        for(int j=i+1; j<edges.size(); j++) {
            for(int k=j+1; k<edges.size(); k++) {
                Job job; init_matrix(&job.blue); init_matrix(&job.red);
                add_edge(&job.blue, edges[i].u, edges[i].v);
                add_edge(&job.blue, edges[j].u, edges[j].v);
                add_edge(&job.blue, edges[k].u, edges[k].v);
                job.is_cop = false; job.picks_left = 0; jobs.push_back(job);
            }
        }
    }
    int num_jobs = jobs.size();
    size_t tt_bytes = TT_ENTRIES * sizeof(TTEntry);
    printf("CPU generated %d parallel opening branches.\n", num_jobs);
    printf("Allocating %.2f GB for verified GPU TT...\n", tt_bytes / (1024.0*1024*1024));
    
    TTEntry* d_tt; cudaMalloc(&d_tt, tt_bytes);
    Job* d_jobs; int* d_results;
    cudaMalloc(&d_jobs, num_jobs * sizeof(Job));
    cudaMalloc(&d_results, num_jobs * sizeof(int));
    cudaMemcpy(d_jobs, jobs.data(), num_jobs * sizeof(Job), cudaMemcpyHostToDevice);
    cudaDeviceSetLimit(cudaLimitStackSize, 65536); 
    
    for (int depth = 1; depth <= 12; depth++) {
        printf("\n[Depth %d] search...\n", depth);
        if (depth == 1) cudaMemset(d_tt, 0, tt_bytes); 
        unsigned long long zero = 0;
        cudaMemcpyToSymbol(nodes_evaluated, &zero, sizeof(unsigned long long));
        double t1 = get_time();
        evaluate_jobs_kernel<<<(num_jobs + 255)/256, 256>>>(d_jobs, d_results, num_jobs, d_tt, depth);
        cudaDeviceSynchronize();
        double t2 = get_time();
        std::vector<int> results(num_jobs);
        cudaMemcpy(results.data(), d_results, num_jobs * sizeof(int), cudaMemcpyDeviceToHost);
        unsigned long long h_nodes = 0;
        cudaMemcpyFromSymbol(&h_nodes, nodes_evaluated, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
        int cop_wins = 0, rob_wins = 0, draws = 0;
        for (int r : results) { if (r == 1) cop_wins++; else if (r == -1) rob_wins++; else draws++; }
        printf("  -> Cop: %d | Robber: %d | Unknown: %d\n", cop_wins, rob_wins, draws);
        printf("  -> Time: %.3f s | Nodes: %llu | Speed: %.2f M/s\n", t2 - t1, h_nodes, (h_nodes / 1000000.0) / (t2 - t1));
        if (cop_wins > 0) { printf("*** COP WINS ***\n"); break; }
        if (rob_wins == num_jobs) { printf("*** ROBBER WINS ***\n"); break; }
    }
    return 0;
}
