#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>

#define K 9
#define COPS 3
#define THREADS_PER_BLOCK 256

// 100 Million states = 3.2 GB per buffer (Total VRAM used ~7GB). 
// Fits perfectly into Colab's 15GB T4 GPU.
#define MAX_STATES_PER_LAYER 100000000

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

// Memory-efficient state (32 bytes per node)
struct GameState {
    AdjMatrix blue;
    AdjMatrix red;
};

// ---------------------------------------------------------------------------------------
// GPU BFS Kernel
// ---------------------------------------------------------------------------------------
__global__ void expand_layer_kernel(
    const GameState* in_states, 
    int num_in_states, 
    GameState* out_states, 
    unsigned int* out_count, 
    int* terminal_evals, 
    bool is_cop
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_in_states) return;
    
    GameState state = in_states[idx];
    
    // Check Robber win (already connected)
    if (is_0_1_connected(&state.red)) {
        terminal_evals[idx] = -1; // Robber wins
        return;
    }
    
    // Check Cop win (graph disconnected and no way to reconnect)
    AdjMatrix available;
    init_matrix(&available);
    uint16_t mask = (1 << K) - 1;
    for(int i = 0; i < K; i++) {
        available.rows[i] = mask & ~(state.blue.rows[i]) & ~(1 << i);
    }
    if (!is_0_1_connected(&available)) {
        terminal_evals[idx] = 1; // Cop wins
        return;
    }
    
    // Generate legal moves
    uint8_t edges_u[120];
    uint8_t edges_v[120];
    int num_edges = 0;
    
    for (int u = 0; u < K; u++) {
        uint16_t avail = mask & ~state.blue.rows[u] & ~state.red.rows[u];
        avail &= ~((1 << (u + 1)) - 1); // Only count u < v
        while (avail != 0) {
            int v = __ffs(avail) - 1;
            if (num_edges < 120 && v >= 0 && v < 16) {
                edges_u[num_edges] = u;
                edges_v[num_edges] = v;
                num_edges++;
            }
            avail &= avail - 1;
        }
    }
    
    if (num_edges == 0) {
        terminal_evals[idx] = 1; // Cop wins by default if board fills up
        return;
    }
    
    terminal_evals[idx] = 0; // State is active, pushing children
    
    // Since this is true BFS, one parent state generates `num_edges` children.
    // We use an atomic add on `out_count` to dynamically allocate a chunk of the output array.
    unsigned int write_idx = atomicAdd(out_count, num_edges);
    
    // If the output array is full, we must abort writing children to prevent memory corruption!
    if (write_idx + num_edges >= MAX_STATES_PER_LAYER) {
        return; 
    }
    
    for (int i = 0; i < num_edges; i++) {
        GameState child = state;
        if (is_cop) {
            add_edge(&child.blue, edges_u[i], edges_v[i]);
        } else {
            add_edge(&child.red, edges_u[i], edges_v[i]);
        }
        out_states[write_idx + i] = child;
    }
}

// ---------------------------------------------------------------------------------------
// CPU Host Code
// ---------------------------------------------------------------------------------------

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    printf("Initializing Massive n-cop GPU BFS Solver for K%d (%d Cops)\n", K, COPS);
    
    // Allocate device memory for ping-pong buffering
    size_t state_bytes = MAX_STATES_PER_LAYER * sizeof(GameState);
    printf("Allocating %.2f GB for ping-pong state buffers...\n", (2 * state_bytes) / (1024.0 * 1024.0 * 1024.0));
    
    GameState *d_states_A, *d_states_B;
    cudaError_t err1 = cudaMalloc(&d_states_A, state_bytes);
    cudaError_t err2 = cudaMalloc(&d_states_B, state_bytes);
    
    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        printf("Failed to allocate massively large states! Reduce MAX_STATES_PER_LAYER.\n");
        return 1;
    }
    
    int* d_terminals;
    cudaMalloc(&d_terminals, MAX_STATES_PER_LAYER * sizeof(int));
    
    unsigned int* d_out_count;
    cudaMalloc(&d_out_count, sizeof(unsigned int));
    
    // Setup initial state (Empty K9 board)
    GameState root;
    init_matrix(&root.blue);
    init_matrix(&root.red);
    
    cudaMemcpy(d_states_A, &root, sizeof(GameState), cudaMemcpyHostToDevice);
    
    int current_layer_size = 1;
    bool is_cop_turn = true;
    int depth = 0;
    
    // Ping pong pointers
    GameState* d_in = d_states_A;
    GameState* d_out = d_states_B;
    
    double t0 = get_time();
    
    while (current_layer_size > 0 && depth < 20) {
        printf("\n[Depth %d] Expanding %d states (Turn: %s)...\n", 
               depth, current_layer_size, is_cop_turn ? "Cop" : "Robber");
               
        // Reset output counter
        unsigned int zero = 0;
        cudaMemcpy(d_out_count, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
        
        int blocks = (current_layer_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        // Launch BFS expansion!
        expand_layer_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_in, 
            current_layer_size, 
            d_out, 
            d_out_count, 
            d_terminals, 
            is_cop_turn
        );
        cudaDeviceSynchronize();
        
        // Retrieve number of newly generated children
        unsigned int next_layer_size = 0;
        cudaMemcpy(&next_layer_size, d_out_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
        // Check for memory overflow
        if (next_layer_size >= MAX_STATES_PER_LAYER) {
            printf("CRITICAL WARNING: Next layer generated %u states, exceeding buffer capacity of %d!\n", 
                   next_layer_size, MAX_STATES_PER_LAYER);
            next_layer_size = MAX_STATES_PER_LAYER - 1; // Clamp it
        }
        
        // Retrieve terminal evaluations for the CURRENT layer
        std::vector<int> terminals(current_layer_size);
        cudaMemcpy(terminals.data(), d_terminals, current_layer_size * sizeof(int), cudaMemcpyDeviceToHost);
        
        int cop_wins = 0, rob_wins = 0, ongoing = 0;
        for (int r : terminals) {
            if (r == 1) cop_wins++;
            else if (r == -1) rob_wins++;
            else ongoing++;
        }
        printf("  -> Terminals found: Cop Wins = %d | Robber Wins = %d | Pushed to next layer: %u\n", 
               cop_wins, rob_wins, next_layer_size);
               
        // Swap buffers
        GameState* temp = d_in;
        d_in = d_out;
        d_out = temp;
        
        current_layer_size = next_layer_size;
        is_cop_turn = !is_cop_turn;
        depth++;
    }
    
    printf("\n=== BFS SEARCH COMPLETE ===\n");
    printf("Total Execution Time: %.3f seconds.\n", get_time() - t0);
    
    cudaFree(d_states_A);
    cudaFree(d_states_B);
    cudaFree(d_terminals);
    cudaFree(d_out_count);
    
    return 0;
}