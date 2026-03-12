#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <algorithm>

// Configuration for K9 / 3 Cops
#define K 9
#define COPS 3
#define THREADS_PER_BLOCK 256
// 8 GB Transposition Table (16 bytes per entry)
#define TT_SIZE_LOG2 29
#define TT_ENTRIES (1ULL << TT_SIZE_LOG2)

struct AdjMatrix { uint16_t rows[16]; };
struct Edge { int u, v; };
struct Job { AdjMatrix blue; bool is_cop; int picks_left; };
struct TTEntry { uint64_t word0, word1; };

__device__ __host__ void init_matrix(AdjMatrix* m) { for (int i=0; i<16; i++) m->rows[i]=0; }
__device__ __host__ void add_edge(AdjMatrix* m, int u, int v) { if (u>=0 && u<16 && v>=0 && v<16) { m->rows[u]|=(1<<v); m->rows[v]|=(1<<u); } }
__device__ __host__ bool has_edge(const AdjMatrix* m, int u, int v) { return (m->rows[u] & (1<<v)) != 0; }

__device__ uint16_t get_component(const AdjMatrix* red, int start, int k) {
    uint16_t visited = 0, frontier = (1 << start);
    while (frontier) {
        visited |= frontier; uint16_t next_f = 0;
        while (frontier) {
            int b = __ffs(frontier)-1;
            next_f |= red->rows[b];
            frontier &= ~(1<<b);
        }
        frontier = next_f & ~visited;
    }
    return visited;
}

__device__ __host__ uint64_t mix(uint64_t h) {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

__device__ void tt_store(TTEntry* tt, uint64_t h0, uint64_t h1, int score, int flag, int depth) {
    uint64_t idx = h0 & (TT_ENTRIES - 1);
    uint64_t payload = ((uint64_t)(score+1)&0xFF) | (((uint64_t)flag&0x3)<<8) | (((uint64_t)depth&0x3F)<<10);
    uint64_t word0 = (h0 & 0xFFFFFFFF00000000ULL) | payload;
    if (word0 == 0) word0 = 1;
    tt[idx].word0 = word0;
    tt[idx].word1 = h1 ^ word0;
}

__device__ bool tt_probe(TTEntry* tt, uint64_t h0, uint64_t h1, int* score, int* flag, int depth) {
    uint64_t idx = h0 & (TT_ENTRIES - 1);
    uint64_t word0 = tt[idx].word0; if (word0 == 0) return false;
    if ((tt[idx].word1 ^ word0) != h1) return false;
    if ((word0 & 0xFFFFFFFF00000000ULL) != (h0 & 0xFFFFFFFF00000000ULL)) return false;
    int s = (int)(word0 & 0xFF)-1;
    if (s == 1 || s == -1) { *score = s; *flag = 0; return true; }
    if (((word0>>10)&0x3F) >= depth) { *score = s; *flag = (int)((word0>>8)&0x3); return true; }
    return false;
}

__device__ unsigned long long nodes_evaluated = 0;

__device__ int device_alpha_beta(AdjMatrix blue, AdjMatrix red, int depth, int alpha, int beta, bool is_cop, int picks_left, TTEntry* tt) {
    atomicAdd(&nodes_evaluated, 1);
    
    // 1. Component Mapping & Win Check
    uint8_t comp_map[16]; uint16_t visited = 0;
    for (int i=0; i<K; i++) {
        if (!(visited & (1<<i))) {
            uint16_t m = get_component(&red, i, K);
            visited |= m;
            while(m) { int b=__ffs(m)-1; comp_map[b]=i; m &= m-1; }
        }
    }
    if (comp_map[0] == comp_map[1]) return -1;
    
    // 2. Cop Win Check (Pruning)
    uint16_t mask = (1 << K) - 1;
    AdjMatrix av_g; init_matrix(&av_g);
    for (int i=0; i<K; i++) av_g.rows[i] = mask & ~blue.rows[i] & ~(1 << i);
    {
        uint16_t v = 1, f = 1; bool ok = false;
        while(f) {
            if(v&2){ok=true;break;}
            uint16_t nf=0, tf=f;
            while(tf){int b=__ffs(tf)-1; nf|=av_g.rows[b]; tf&=~(1<<b);}
            f=nf&~v; v|=nf;
        }
        if (!ok) return 1;
    }
    if (depth <= 0) return 0;

    // 3. Transposition Table Probe
    uint64_t h = 0x123456789ABCDEF0ULL;
    for(int i=0; i<K; i++) { h ^= mix(blue.rows[i] + (h<<7)); h ^= mix(comp_map[i] + (h>>3)); }
    if (is_cop) h ^= 0x5555555555555555ULL;
    h ^= mix(picks_left + 0xAAAAAAAABBBBBBBBULL);
    uint64_t h0 = mix(h), h1 = mix(h ^ 0xDEADBEEFCAFEBABEULL);

    int tt_s, tt_f;
    if (tt_probe(tt, h0, h1, &tt_s, &tt_f, depth)) {
        if (tt_f == 0) return tt_s;
        if (tt_f == 1 && tt_s > alpha) alpha = tt_s;
        if (tt_f == 2 && tt_s < beta) beta = tt_s;
        if (alpha >= beta) return tt_s;
    }

    // 4. Move Generation with Redundant Edge Pruning
    uint8_t edges_u[120], edges_v[120]; int n_edges = 0;
    for (int u=0; u<K; u++) {
        uint16_t av = mask & ~blue.rows[u] & ~red.rows[u] & ~((1<<(u+1))-1);
        while(av) {
            int v = __ffs(av)-1;
            if (comp_map[u] != comp_map[v]) { edges_u[n_edges]=u; edges_v[n_edges]=v; n_edges++; }
            av &= av-1;
        }
    }
    if (n_edges == 0) return 1;

    // 5. Alpha-Beta Recursion
    int orig_a = alpha, orig_b = beta, best;
    if (is_cop) {
        best = -999;
        for (int i=0; i<n_edges; i++) {
            AdjMatrix nb = blue; add_edge(&nb, edges_u[i], edges_v[i]);
            int s = (picks_left > 1) 
                ? device_alpha_beta(nb, red, depth, alpha, beta, true, picks_left-1, tt) 
                : device_alpha_beta(nb, red, depth, alpha, beta, false, 0, tt);
            if (s > best) best = s;
            if (best > alpha) alpha = best;
            if (alpha >= beta) break;
        }
    } else {
        best = 999;
        for (int i=0; i<n_edges; i++) {
            AdjMatrix nr = red; add_edge(&nr, edges_u[i], edges_v[i]);
            int s = device_alpha_beta(blue, nr, depth - 1, alpha, beta, true, COPS, tt);
            if (s < best) best = s;
            if (best < beta) beta = best;
            if (alpha >= beta) break;
        }
    }
    
    // 6. TT Store
    int flag = 0; 
    if (best <= orig_a) flag = 2; else if (best >= orig_b) flag = 1;
    tt_store(tt, h0, h1, best, flag, depth);
    return best;
}

__global__ void evaluate_jobs_kernel(Job* jobs, int* results, int num_jobs, TTEntry* tt, int max_depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_jobs) return;
    AdjMatrix red; init_matrix(&red);
    results[idx] = device_alpha_beta(jobs[idx].blue, red, max_depth, -999, 999, jobs[idx].is_cop, jobs[idx].picks_left, tt);
}

void generate_root_jobs(int u_s, int v_s, int p_l, AdjMatrix blue, std::vector<Job>& jobs) {
    if (p_l == 0) {
        Job j; j.blue = blue; j.is_cop = false; j.picks_left = 0;
        jobs.push_back(j); return;
    }
    for (int u = u_s; u < K; u++) {
        for (int v = (u == u_s ? v_s : u + 1); v < K; v++) {
            if (!has_edge(&blue, u, v)) {
                AdjMatrix next = blue; add_edge(&next, u, v);
                generate_root_jobs(u, v + 1, p_l - 1, next, jobs);
            }
        }
    }
}

int main() {
    printf("Initializing Absolute n-cop GPU Solver (K%d, %d Cops)\n", K, COPS);
    AdjMatrix rb; init_matrix(&rb);
    std::vector<Job> jobs;
    generate_root_jobs(0, 1, COPS, rb, jobs);
    
    // Initial Move Ordering: process opening moves that block node 0 or 1 first
    std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
        int a_s = (has_edge(&a.blue, 0, 1) ? 1000 : 0);
        int b_s = (has_edge(&b.blue, 0, 1) ? 1000 : 0);
        return a_s > b_s;
    });
    
    int n_jobs = jobs.size();
    size_t tt_b = TT_ENTRIES * sizeof(TTEntry);
    printf("CPU jobs: %d | TT Size: %.2f GB\n", n_jobs, tt_b / (1024.0*1024*1024));
    
    TTEntry* d_tt; cudaMalloc(&d_tt, tt_b);
    Job* d_j; int* d_r;
    cudaMalloc(&d_j, n_jobs * sizeof(Job));
    cudaMalloc(&d_r, n_jobs * sizeof(int));
    cudaMemcpy(d_j, jobs.data(), n_jobs * sizeof(Job), cudaMemcpyHostToDevice);
    
    cudaDeviceSetLimit(cudaLimitStackSize, 65536);
    
    for (int depth = 1; depth <= 12; depth++) {
        printf("\n[Depth %d] starting GPU search...\n", depth);
        if (depth == 1) cudaMemset(d_tt, 0, tt_b);
        
        unsigned long long zero = 0;
        cudaMemcpyToSymbol(nodes_evaluated, &zero, sizeof(unsigned long long));
        
        auto t1 = std::chrono::steady_clock::now();
        evaluate_jobs_kernel<<<(n_jobs + 255)/256, 256>>>(d_j, d_r, n_jobs, d_tt, depth);
        cudaDeviceSynchronize();
        auto t2 = std::chrono::steady_clock::now();
        
        double dt = std::chrono::duration<double>(t2 - t1).count();
        std::vector<int> res(n_jobs);
        cudaMemcpy(res.data(), d_r, n_jobs * sizeof(int), cudaMemcpyDeviceToHost);
        
        unsigned long long n_e = 0;
        cudaMemcpyFromSymbol(&n_e, nodes_evaluated, sizeof(unsigned long long), 0, cudaMemcpyDeviceToHost);
        
        int c_w = 0, r_w = 0, dr = 0;
        for (int r : res) {
            if (r == 1) c_w++; else if (r == -1) r_w++; else dr++;
        }
        
        printf("  -> Cop force-wins: %d | Robber force-wins: %d | Unresolved: %d\n", c_w, r_w, dr);
        printf("  -> Time: %.3f s | Nodes: %llu | Speed: %.2f M/s\n", dt, n_e, (n_e / 1000000.0) / dt);
        
        if (c_w > 0) { printf("\n*** PROOF COMPLETE: COP WINS ***\n"); break; }
        if (r_w == n_jobs) { printf("\n*** PROOF COMPLETE: ROBBER WINS ***\n"); break; }
    }
    
    cudaFree(d_tt); cudaFree(d_j); cudaFree(d_r);
    return 0;
}
