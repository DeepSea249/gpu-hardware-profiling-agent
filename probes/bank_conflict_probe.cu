/**
 * Shared Memory Bank Conflict Penalty Probe
 *
 * Measures the latency difference between conflict-free and
 * N-way bank conflict shared memory access patterns.
 *
 * Bank layout: 32 banks, 4 bytes each. smem[i] maps to bank (i % 32).
 * - Stride 1: thread i reads smem[i] -> all different banks (0 conflicts)
 * - Stride 32: thread i reads smem[i*32] -> all bank 0 (32-way conflict)
 *
 * Uses volatile shared memory to prevent compiler from caching values
 * in registers and optimizing away the actual shared memory reads.
 *
 * Output format:
 *   STRIDE=<n> CYCLES_PER_ACCESS=<f>
 *   BANK_CONFLICT_PENALTY_CYCLES=<f>
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

#define SMEM_SIZE 4096

/**
 * Bank conflict measurement kernel using volatile shared memory.
 * volatile prevents the compiler from caching shared memory values
 * in registers, ensuring every iteration performs an actual smem read.
 */
__global__ void bank_conflict_kernel(long long* cycles_out, int stride,
                                      int iterations, int* dummy_out) {
    __shared__ volatile int smem[SMEM_SIZE];
    int tid = threadIdx.x;
    int lane = tid & 31;

    // Initialize shared memory
    for (int i = tid; i < SMEM_SIZE; i += blockDim.x) {
        smem[i] = (int)(i * 17 + 3);
    }
    __syncthreads();

    int idx = (lane * stride) % SMEM_SIZE;
    int val = 0;

    // Warmup
    for (int i = 0; i < 2000; i++) {
        val += smem[idx];
    }
    __syncthreads();

    // Timed measurement
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        val += smem[idx];
    }
    long long end = clock64();
    __syncthreads();

    dummy_out[tid] = val;
    if (tid == 0) {
        *cycles_out = end - start;
    }
}

/**
 * Alternative: dependency chain through shared memory.
 * Each read produces a value that modifies the next read index,
 * while staying within the same bank.
 */
__global__ void bank_conflict_dep_kernel(long long* cycles_out, int stride,
                                          int iterations, int* dummy_out) {
    __shared__ int smem[SMEM_SIZE];
    int tid = threadIdx.x;
    int lane = tid & 31;

    // Initialize: create chained values within each bank
    for (int i = tid; i < SMEM_SIZE; i += blockDim.x) {
        int bank = i % 32;
        int slot = i / 32;
        int next_slot = (slot + 7) % (SMEM_SIZE / 32);
        smem[i] = bank + next_slot * 32; // Points to another element in same bank
    }
    __syncthreads();

    int bank = (lane * stride) % 32;
    int idx = bank; // Start at first element of target bank
    int val = 0;

    // Warmup
    for (int i = 0; i < 1000; i++) {
        idx = smem[idx % SMEM_SIZE];
        val += idx;
    }
    __syncthreads();

    idx = bank;
    long long start = clock64();
    #pragma unroll 1
    for (int i = 0; i < iterations; i++) {
        idx = smem[idx % SMEM_SIZE];
    }
    long long end = clock64();
    __syncthreads();

    dummy_out[tid] = idx + val;
    if (tid == 0) {
        *cycles_out = end - start;
    }
}

int main(int argc, char** argv) {
    int iterations = 100000;
    int num_trials = 7;

    if (argc > 1) iterations = atoi(argv[1]);
    if (argc > 2) num_trials = atoi(argv[2]);

    printf("BANK_CONFLICT_PROBE_START\n");
    printf("iterations=%d num_trials=%d\n", iterations, num_trials);

    long long* d_cycles;
    int* d_dummy;
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_dummy, 1024 * sizeof(int)));

    int strides[] = {1, 2, 4, 8, 16, 32};
    int num_strides = sizeof(strides) / sizeof(strides[0]);

    double stride_latencies[16];
    double dep_latencies[16];

    printf("--- Volatile shared memory method ---\n");
    for (int s = 0; s < num_strides; s++) {
        int stride = strides[s];

        bank_conflict_kernel<<<1, 32, 0>>>(d_cycles, stride, iterations, d_dummy);
        CUDA_CHECK(cudaDeviceSynchronize());

        double trials[32];
        for (int t = 0; t < num_trials; t++) {
            bank_conflict_kernel<<<1, 32, 0>>>(d_cycles, stride, iterations, d_dummy);
            CUDA_CHECK(cudaDeviceSynchronize());

            long long cycles;
            CUDA_CHECK(cudaMemcpy(&cycles, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost));
            trials[t] = (double)cycles / iterations;
        }

        // Sort for median
        for (int i = 0; i < num_trials - 1; i++) {
            for (int j = i + 1; j < num_trials; j++) {
                if (trials[j] < trials[i]) {
                    double tmp = trials[i];
                    trials[i] = trials[j];
                    trials[j] = tmp;
                }
            }
        }
        stride_latencies[s] = trials[num_trials / 2];
        printf("STRIDE=%d CYCLES_PER_ACCESS=%.2f\n", stride, stride_latencies[s]);
    }

    printf("--- Dependency chain method ---\n");
    for (int s = 0; s < num_strides; s++) {
        int stride = strides[s];

        bank_conflict_dep_kernel<<<1, 32, 0>>>(d_cycles, stride, iterations, d_dummy);
        CUDA_CHECK(cudaDeviceSynchronize());

        double trials[32];
        for (int t = 0; t < num_trials; t++) {
            bank_conflict_dep_kernel<<<1, 32, 0>>>(d_cycles, stride, iterations, d_dummy);
            CUDA_CHECK(cudaDeviceSynchronize());

            long long cycles;
            CUDA_CHECK(cudaMemcpy(&cycles, d_cycles, sizeof(long long), cudaMemcpyDeviceToHost));
            trials[t] = (double)cycles / iterations;
        }

        for (int i = 0; i < num_trials - 1; i++) {
            for (int j = i + 1; j < num_trials; j++) {
                if (trials[j] < trials[i]) {
                    double tmp = trials[i];
                    trials[i] = trials[j];
                    trials[j] = tmp;
                }
            }
        }
        dep_latencies[s] = trials[num_trials / 2];
        printf("DEP_STRIDE=%d CYCLES_PER_ACCESS=%.2f\n", stride, dep_latencies[s]);
    }

    // Use whichever method shows a clearer penalty
    double vol_penalty = stride_latencies[num_strides - 1] - stride_latencies[0];
    double dep_penalty = dep_latencies[num_strides - 1] - dep_latencies[0];

    double penalty, no_conflict, max_conflict;
    if (vol_penalty >= dep_penalty) {
        penalty = vol_penalty;
        no_conflict = stride_latencies[0];
        max_conflict = stride_latencies[num_strides - 1];
    } else {
        penalty = dep_penalty;
        no_conflict = dep_latencies[0];
        max_conflict = dep_latencies[num_strides - 1];
    }

    printf("BANK_CONFLICT_PENALTY_CYCLES=%.2f\n", penalty);
    printf("NO_CONFLICT_CYCLES=%.2f\n", no_conflict);
    printf("MAX_CONFLICT_CYCLES=%.2f\n", max_conflict);

    CUDA_CHECK(cudaFree(d_cycles));
    CUDA_CHECK(cudaFree(d_dummy));

    printf("BANK_CONFLICT_PROBE_END\n");
    return 0;
}
