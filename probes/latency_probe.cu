/**
 * Memory Latency Hierarchy Probe (Pointer Chasing)
 *
 * Measures access latency at different array sizes to identify
 * L1 cache, L2 cache, and DRAM latency in clock cycles.
 * Uses a random single-cycle permutation to defeat hardware prefetchers.
 *
 * Output format:
 *   SIZE_BYTES=<n> AVG_CYCLES=<f>
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/**
 * Single-thread pointer chasing kernel.
 * Each iteration: idx = chain[idx], creating a serial dependency chain
 * that the GPU cannot prefetch or pipeline around.
 */
__global__ void pointer_chase(const int* __restrict__ chain,
                               long long* cycles_out,
                               int* dummy_out,
                               int num_steps) {
    int idx = 0;

    // Warmup: traverse the chain to populate cache hierarchy
    for (int i = 0; i < num_steps; i++) {
        idx = chain[idx];
    }

    // Reset to start of chain
    idx = 0;

    // Timed measurement with dependency chain
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < num_steps; i++) {
        idx = chain[idx];
    }
    long long end = clock64();

    *dummy_out = idx;       // Prevent dead code elimination
    *cycles_out = end - start;
}

/**
 * Creates a random single-cycle permutation (Hamiltonian cycle).
 * This ensures every element is visited exactly once before returning
 * to the start, and the stride pattern is random to defeat prefetchers.
 */
void create_random_chain(int* chain, int n) {
    int* perm = (int*)malloc(n * sizeof(int));
    if (!perm) { fprintf(stderr, "malloc failed\n"); exit(1); }

    for (int i = 0; i < n; i++) perm[i] = i;

    // Fisher-Yates shuffle
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }

    // Build cycle: perm[0] -> perm[1] -> ... -> perm[n-1] -> perm[0]
    for (int i = 0; i < n - 1; i++) {
        chain[perm[i]] = perm[i + 1];
    }
    chain[perm[n - 1]] = perm[0];

    free(perm);
}

int main(int argc, char** argv) {
    int num_steps = 100000;
    int num_trials = 7;

    if (argc > 1) num_steps = atoi(argv[1]);
    if (argc > 2) num_trials = atoi(argv[2]);

    srand(42);

    // Array sizes in number of ints (each int = 4 bytes)
    // Covering L1 (up to ~128-192KB), L2 (up to several MB), and DRAM
    int sizes[] = {
        256,         // 1 KB
        512,         // 2 KB
        1024,        // 4 KB
        2048,        // 8 KB
        4096,        // 16 KB
        8192,        // 32 KB
        16384,       // 64 KB
        32768,       // 128 KB
        49152,       // 192 KB
        65536,       // 256 KB
        98304,       // 384 KB
        131072,      // 512 KB
        196608,      // 768 KB
        262144,      // 1 MB
        393216,      // 1.5 MB
        524288,      // 2 MB
        786432,      // 3 MB
        1048576,     // 4 MB
        1572864,     // 6 MB
        2097152,     // 8 MB
        3145728,     // 12 MB
        4194304,     // 16 MB
        8388608,     // 32 MB
        16777216,    // 64 MB
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    long long* d_cycles;
    int* d_dummy;
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_dummy, sizeof(int)));

    // GPU warmup: launch a trivial kernel
    pointer_chase<<<1, 1>>>(NULL, d_cycles, d_dummy, 0);
    cudaDeviceSynchronize();
    cudaGetLastError(); // clear any error from NULL access

    // Allocate a small dummy buffer for warmup
    int* d_warmup;
    CUDA_CHECK(cudaMalloc(&d_warmup, 1024 * sizeof(int)));
    int h_warmup[1024];
    for (int i = 0; i < 1024; i++) h_warmup[i] = (i + 1) % 1024;
    CUDA_CHECK(cudaMemcpy(d_warmup, h_warmup, 1024 * sizeof(int), cudaMemcpyHostToDevice));
    pointer_chase<<<1, 1>>>(d_warmup, d_cycles, d_dummy, 1000);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_warmup));

    printf("LATENCY_PROBE_START\n");
    printf("num_steps=%d num_trials=%d\n", num_steps, num_trials);

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        size_t bytes = (size_t)n * sizeof(int);

        // Check available memory
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        if (bytes > free_mem * 0.8) {
            printf("SIZE_BYTES=%zu SKIPPED=out_of_memory\n", bytes);
            continue;
        }

        // Create chain on host
        int* h_chain = (int*)malloc(bytes);
        if (!h_chain) {
            printf("SIZE_BYTES=%zu SKIPPED=host_malloc_failed\n", bytes);
            continue;
        }
        create_random_chain(h_chain, n);

        // Copy to device
        int* d_chain;
        CUDA_CHECK(cudaMalloc(&d_chain, bytes));
        CUDA_CHECK(cudaMemcpy(d_chain, h_chain, bytes, cudaMemcpyHostToDevice));

        // Warmup run (populates caches)
        pointer_chase<<<1, 1>>>(d_chain, d_cycles, d_dummy, num_steps);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Measurement runs - collect all trials
        double cycles_arr[32]; // max trials
        for (int t = 0; t < num_trials; t++) {
            pointer_chase<<<1, 1>>>(d_chain, d_cycles, d_dummy, num_steps);
            CUDA_CHECK(cudaDeviceSynchronize());

            long long cycles;
            CUDA_CHECK(cudaMemcpy(&cycles, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost));
            cycles_arr[t] = (double)cycles / num_steps;
        }

        // Compute median (sort and take middle value for robustness)
        for (int i = 0; i < num_trials - 1; i++) {
            for (int j = i + 1; j < num_trials; j++) {
                if (cycles_arr[j] < cycles_arr[i]) {
                    double tmp = cycles_arr[i];
                    cycles_arr[i] = cycles_arr[j];
                    cycles_arr[j] = tmp;
                }
            }
        }
        double median_cycles = cycles_arr[num_trials / 2];

        // Also compute mean excluding min/max (trimmed mean)
        double trimmed_sum = 0;
        int trimmed_count = 0;
        for (int t = 1; t < num_trials - 1; t++) {
            trimmed_sum += cycles_arr[t];
            trimmed_count++;
        }
        double trimmed_mean = (trimmed_count > 0) ? trimmed_sum / trimmed_count : median_cycles;

        printf("SIZE_BYTES=%zu AVG_CYCLES=%.2f MEDIAN_CYCLES=%.2f TRIMMED_MEAN=%.2f\n",
               bytes, cycles_arr[0], median_cycles, trimmed_mean);

        CUDA_CHECK(cudaFree(d_chain));
        free(h_chain);
    }

    printf("LATENCY_PROBE_END\n");

    CUDA_CHECK(cudaFree(d_cycles));
    CUDA_CHECK(cudaFree(d_dummy));

    return 0;
}
