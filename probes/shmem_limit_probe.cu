/**
 * Shared Memory Limit Probe
 *
 * Determines the maximum amount of dynamic shared memory that can be
 * allocated per block. Uses binary search with kernel launch attempts.
 *
 * Supports both default and opt-in extended shared memory
 * (cudaFuncSetAttribute for architectures >= 7.0).
 *
 * Output format:
 *   MAX_SHMEM_PER_BLOCK_BYTES=<n>
 *   MAX_SHMEM_PER_BLOCK_KB=<n>
 *   DEFAULT_SHMEM_LIMIT_BYTES=<n>
 *   EXTENDED_SHMEM_LIMIT_BYTES=<n>
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
 * Simple kernel that uses dynamic shared memory.
 * Each thread writes to and reads from shared memory.
 */
__global__ void shmem_test_kernel(int* output) {
    extern __shared__ char smem[];

    int tid = threadIdx.x;
    // Write to shared memory
    smem[tid % 1024] = (char)(tid & 0xFF);
    __syncthreads();

    // Read back
    if (tid == 0) {
        output[0] = (int)smem[0];
    }
}

/**
 * Try to launch a kernel with the given shared memory size.
 * Returns true if successful, false otherwise.
 */
bool try_shmem_size(int shmem_bytes, int* d_output, bool use_extended) {
    cudaError_t err;

    if (use_extended) {
        err = cudaFuncSetAttribute(
            shmem_test_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_bytes
        );
        if (err != cudaSuccess) {
            cudaGetLastError(); // Clear error
            return false;
        }
    }

    shmem_test_kernel<<<1, 32, shmem_bytes>>>(d_output);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaGetLastError(); // Clear error
        return false;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaGetLastError(); // Clear error
        return false;
    }

    return true;
}

int main(int argc, char** argv) {
    printf("SHMEM_LIMIT_PROBE_START\n");

    int* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(int)));

    // ==================== Phase 1: Default Shared Memory Limit ====================
    // Without cudaFuncSetAttribute, test default limit
    int default_max = 0;
    {
        int lo = 1024;       // 1 KB
        int hi = 256 * 1024; // 256 KB upper bound

        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;

            if (try_shmem_size(mid, d_output, false)) {
                default_max = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
    }
    printf("DEFAULT_SHMEM_LIMIT_BYTES=%d\n", default_max);
    printf("DEFAULT_SHMEM_LIMIT_KB=%d\n", default_max / 1024);

    // ==================== Phase 2: Extended Shared Memory Limit ====================
    // With cudaFuncSetAttribute to opt-in to maximum
    int extended_max = default_max;
    {
        int lo = default_max + 1;
        int hi = 256 * 1024; // 256 KB upper bound (covers up to Hopper 228KB)

        // First check if extended shared memory is supported at all
        if (try_shmem_size(default_max + 1024, d_output, true)) {
            extended_max = default_max + 1024; // At least this much

            while (lo <= hi) {
                int mid = lo + (hi - lo) / 2;

                if (try_shmem_size(mid, d_output, true)) {
                    extended_max = mid;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
        }
    }
    printf("EXTENDED_SHMEM_LIMIT_BYTES=%d\n", extended_max);
    printf("EXTENDED_SHMEM_LIMIT_KB=%d\n", extended_max / 1024);

    // The actual maximum usable shared memory per block
    int actual_max = (extended_max > default_max) ? extended_max : default_max;
    printf("MAX_SHMEM_PER_BLOCK_BYTES=%d\n", actual_max);
    printf("MAX_SHMEM_PER_BLOCK_KB=%d\n", actual_max / 1024);

    // Also report device properties for cross-reference
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("REPORTED_SHMEM_PER_BLOCK=%zu\n", prop.sharedMemPerBlock);
    printf("REPORTED_SHMEM_PER_SM=%zu\n", prop.sharedMemPerMultiprocessor);

    CUDA_CHECK(cudaFree(d_output));

    printf("SHMEM_LIMIT_PROBE_END\n");
    return 0;
}
