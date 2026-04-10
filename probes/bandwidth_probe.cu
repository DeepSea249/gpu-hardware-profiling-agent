/**
 * Memory Bandwidth Probe
 *
 * Measures effective peak bandwidth for:
 * 1. Global Memory (VRAM) - read and write
 * 2. Shared Memory - per-SM and aggregate
 *
 * Uses vectorized (float4) access patterns for maximum throughput.
 * Timing via CUDA events for accurate wall-clock measurement.
 *
 * Output format:
 *   GLOBAL_READ_BW_GBPS=<f> SIZE_MB=<n>
 *   GLOBAL_WRITE_BW_GBPS=<f> SIZE_MB=<n>
 *   SHMEM_BW_GBPS_PER_SM=<f> SHMEM_BW_GBPS_AGGREGATE=<f>
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
 * Global memory read bandwidth kernel.
 * Each thread reads float4 (16 bytes) in a coalesced pattern.
 * Grid-stride loop to handle arbitrary sizes.
 */
__global__ void global_read_kernel(const float4* __restrict__ data,
                                    float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int i = idx; i < n; i += stride) {
        float4 val = data[i];
        sum.x += val.x;
        sum.y += val.y;
        sum.z += val.z;
        sum.w += val.w;
    }

    // Prevent optimization - single atomic add
    if (idx == 0) {
        output[0] = sum.x + sum.y + sum.z + sum.w;
    }
}

/**
 * Global memory write bandwidth kernel.
 * Each thread writes float4 (16 bytes) in a coalesced pattern.
 */
__global__ void global_write_kernel(float4* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float4 val = make_float4(1.0f, 2.0f, 3.0f, 4.0f);

    for (int i = idx; i < n; i += stride) {
        data[i] = val;
    }
}

/**
 * Global memory copy bandwidth kernel (read+write).
 */
__global__ void global_copy_kernel(const float4* __restrict__ src,
                                    float4* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

/**
 * Shared memory bandwidth kernel.
 * Each block loads data into shared memory and performs repeated reads.
 * Bank-conflict-free access pattern for maximum throughput.
 */
__global__ void shmem_bandwidth_kernel(float* output, int iterations) {
    extern __shared__ float4 smem[];
    int tid = threadIdx.x;
    int block_threads = blockDim.x;

    // Initialize shared memory with bank-conflict-free pattern
    for (int i = tid; i < block_threads; i += block_threads) {
        smem[i] = make_float4((float)i, (float)(i+1), (float)(i+2), (float)(i+3));
    }
    __syncthreads();

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Warmup
    for (int iter = 0; iter < 100; iter++) {
        float4 v = smem[tid % block_threads];
        sum.x += v.x;
        sum.y += v.y;
    }
    __syncthreads();

    // Timed reads from shared memory
    // Each thread reads sizeof(float4)=16 bytes per iteration
    long long start = clock64();
    #pragma unroll 4
    for (int iter = 0; iter < iterations; iter++) {
        float4 v = smem[tid % block_threads];
        sum.x += v.x;
        sum.y += v.y;
        sum.z += v.z;
        sum.w += v.w;
    }
    long long end = clock64();
    __syncthreads();

    // Store results
    if (tid == 0) {
        output[blockIdx.x * 3 + 0] = sum.x + sum.y + sum.z + sum.w;
        // Pack cycle count into two floats
        long long elapsed = end - start;
        output[blockIdx.x * 3 + 1] = __int_as_float((int)(elapsed & 0xFFFFFFFF));
        output[blockIdx.x * 3 + 2] = __int_as_float((int)((elapsed >> 32) & 0xFFFFFFFF));
    }
}

/**
 * Detect number of active SMs by collecting unique SM IDs.
 */
__global__ void detect_sm_count(int* sm_ids, int* counter) {
    if (threadIdx.x == 0) {
        int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        int idx = atomicAdd(counter, 1);
        sm_ids[idx] = smid;
    }
}

int get_active_sm_count() {
    int max_blocks = 2048;
    int* d_sm_ids;
    int* d_counter;
    CUDA_CHECK(cudaMalloc(&d_sm_ids, max_blocks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    detect_sm_count<<<max_blocks, 1>>>(d_sm_ids, d_counter);
    CUDA_CHECK(cudaDeviceSynchronize());

    int count;
    CUDA_CHECK(cudaMemcpy(&count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));

    int* h_sm_ids = (int*)malloc(count * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_sm_ids, d_sm_ids, count * sizeof(int), cudaMemcpyDeviceToHost));

    // Count unique SM IDs
    int unique = 0;
    int seen[256] = {0};
    for (int i = 0; i < count; i++) {
        int id = h_sm_ids[i];
        if (id >= 0 && id < 256 && !seen[id]) {
            seen[id] = 1;
            unique++;
        }
    }

    free(h_sm_ids);
    CUDA_CHECK(cudaFree(d_sm_ids));
    CUDA_CHECK(cudaFree(d_counter));

    return unique;
}

int main(int argc, char** argv) {
    printf("BANDWIDTH_PROBE_START\n");

    // Detect active SM count
    int num_sms = get_active_sm_count();
    printf("ACTIVE_SM_COUNT=%d\n", num_sms);

    // ==================== Global Memory Bandwidth ====================
    size_t test_sizes_mb[] = {64, 128, 256, 512};
    int num_test_sizes = sizeof(test_sizes_mb) / sizeof(test_sizes_mb[0]);
    int num_trials = 10;

    double best_read_bw = 0, best_write_bw = 0, best_copy_bw = 0;

    for (int s = 0; s < num_test_sizes; s++) {
        size_t bytes = test_sizes_mb[s] * 1024ULL * 1024ULL;

        // Check available memory
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        if (bytes * 2 > free_mem * 0.8) {
            printf("GLOBAL_SKIP_SIZE_MB=%zu reason=insufficient_memory\n", test_sizes_mb[s]);
            continue;
        }

        int n = bytes / sizeof(float4);

        float4* d_data;
        float4* d_data2;
        float* d_output;
        CUDA_CHECK(cudaMalloc(&d_data, bytes));
        CUDA_CHECK(cudaMalloc(&d_data2, bytes));
        CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_data, 1, bytes));

        int block_size = 256;
        int num_blocks = (n + block_size - 1) / block_size;
        if (num_blocks > 65535) num_blocks = 65535;

        // Warmup
        global_read_kernel<<<num_blocks, block_size>>>(d_data, d_output, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // ---- Read Bandwidth ----
        CUDA_CHECK(cudaEventRecord(start));
        for (int t = 0; t < num_trials; t++) {
            global_read_kernel<<<num_blocks, block_size>>>(d_data, d_output, n);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        double read_bw = (double)bytes * num_trials / (ms / 1000.0) / 1e9;
        if (read_bw > best_read_bw) best_read_bw = read_bw;
        printf("GLOBAL_READ_BW_GBPS=%.2f SIZE_MB=%zu\n", read_bw, test_sizes_mb[s]);

        // ---- Write Bandwidth ----
        CUDA_CHECK(cudaEventRecord(start));
        for (int t = 0; t < num_trials; t++) {
            global_write_kernel<<<num_blocks, block_size>>>(d_data, n);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        double write_bw = (double)bytes * num_trials / (ms / 1000.0) / 1e9;
        if (write_bw > best_write_bw) best_write_bw = write_bw;
        printf("GLOBAL_WRITE_BW_GBPS=%.2f SIZE_MB=%zu\n", write_bw, test_sizes_mb[s]);

        // ---- Copy Bandwidth ----
        CUDA_CHECK(cudaEventRecord(start));
        for (int t = 0; t < num_trials; t++) {
            global_copy_kernel<<<num_blocks, block_size>>>(d_data, d_data2, n);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        // Copy reads + writes, so total bytes = 2 * bytes
        double copy_bw = 2.0 * (double)bytes * num_trials / (ms / 1000.0) / 1e9;
        if (copy_bw > best_copy_bw) best_copy_bw = copy_bw;
        printf("GLOBAL_COPY_BW_GBPS=%.2f SIZE_MB=%zu\n", copy_bw, test_sizes_mb[s]);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_data2));
        CUDA_CHECK(cudaFree(d_output));
    }

    printf("BEST_GLOBAL_READ_BW_GBPS=%.2f\n", best_read_bw);
    printf("BEST_GLOBAL_WRITE_BW_GBPS=%.2f\n", best_write_bw);
    printf("BEST_GLOBAL_COPY_BW_GBPS=%.2f\n", best_copy_bw);

    // ==================== Shared Memory Bandwidth ====================
    {
        int block_size = 1024;  // Max threads per block
        int shmem_per_block = block_size * sizeof(float4); // 1024 * 16 = 16KB
        int iterations = 100000;
        int blocks = num_sms;  // One block per SM

        float* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, blocks * 3 * sizeof(float)));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        shmem_bandwidth_kernel<<<blocks, block_size, shmem_per_block>>>(d_output, 1000);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed measurement
        CUDA_CHECK(cudaEventRecord(start));
        shmem_bandwidth_kernel<<<blocks, block_size, shmem_per_block>>>(d_output, iterations);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        // Per SM: each block has block_size threads, each reading 16 bytes per iteration
        double total_bytes = (double)blocks * block_size * sizeof(float4) * iterations;
        double aggregate_bw_gbps = total_bytes / (ms / 1000.0) / 1e9;
        double per_sm_bw_gbps = aggregate_bw_gbps / blocks;

        printf("SHMEM_BW_GBPS_PER_SM=%.2f\n", per_sm_bw_gbps);
        printf("SHMEM_BW_GBPS_AGGREGATE=%.2f\n", aggregate_bw_gbps);
        printf("SHMEM_TEST_BLOCKS=%d THREADS=%d ITERATIONS=%d\n", blocks, block_size, iterations);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_output));
    }

    printf("BANDWIDTH_PROBE_END\n");
    return 0;
}
