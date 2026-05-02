/**
 * matmul_tensor.cu  –  Tensor-Core matrix multiplication via WMMA API
 *                       with shared-memory staging.
 *
 * Uses nvcuda::wmma with FP16 inputs / FP32 accumulation on 16×16×16 tiles.
 * Shared memory is used to cache an entire K-strip (BLOCK_K wide) of A and B
 * for all warps in the block, dramatically reducing global memory traffic
 * and keeping the HMMA pipeline saturated.
 *
 * Block:  256 threads = 8 warps arranged as 4(row) × 2(col).
 *         Each warp owns one 16×16 WMMA output tile.
 *         Block output = 64 × 32  (4×WMMA_M × 2×WMMA_N).
 *
 * Usage:  ./matmul_tensor [N]       (default N=4096, must be multiple of 16)
 *         Computes  C(FP32) = A(FP16) × B(FP16)
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// WMMA tile dimensions (hardware-fixed for FP16 → FP32)
static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// Block-level tiling
static constexpr int WARPS_Y = 4;  // warps along M
static constexpr int WARPS_X = 2;  // warps along N
static constexpr int BLOCK_M = WARPS_Y * WMMA_M;  // 64
static constexpr int BLOCK_N = WARPS_X * WMMA_N;   // 32
static constexpr int BLOCK_K = 64;                  // K-strip: 4 × WMMA_K → 4 MMAs per shmem load
static constexpr int THREADS = WARPS_Y * WARPS_X * 32;  // 256

__global__ void matmul_tensor_kernel(const half *__restrict__ A,
                                     const half *__restrict__ B,
                                     float      *__restrict__ C,
                                     int N) {
    // Shared memory tiles for one K-strip:
    //   sA[BLOCK_M][BLOCK_K]  +  sB[BLOCK_K][BLOCK_N]
    // Pad to avoid bank conflicts on FP16 loads.
    __shared__ half sA[BLOCK_M][BLOCK_K + 8];
    __shared__ half sB[BLOCK_K][BLOCK_N + 8];

    const int tid    = threadIdx.x;
    const int warpId = tid / 32;
    const int warpY  = warpId / WARPS_X;  // 0..3
    const int warpX  = warpId % WARPS_X;  // 0..1

    // Block-level origin in the output matrix
    const int blockRowBase = blockIdx.y * BLOCK_M;
    const int blockColBase = blockIdx.x * BLOCK_N;

    // Declare accumulator fragments (one per warp)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Loop over K in strips of BLOCK_K
    for (int kOuter = 0; kOuter < N; kOuter += BLOCK_K) {
        // --- Cooperative load of sA[BLOCK_M][BLOCK_K] from global A -----
        // 256 threads load 64×64 = 4096 half values → 16 halves per thread
        for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += THREADS) {
            int row = idx / BLOCK_K;
            int col = idx % BLOCK_K;
            int gRow = blockRowBase + row;
            int gCol = kOuter + col;
            sA[row][col] = (gRow < N && gCol < N) ? A[(size_t)gRow * N + gCol]
                                                   : __float2half(0.0f);
        }

        // --- Cooperative load of sB[BLOCK_K][BLOCK_N] from global B -----
        // 256 threads load 64×32 = 2048 half values → 8 per thread
        for (int idx = tid; idx < BLOCK_K * BLOCK_N; idx += THREADS) {
            int row = idx / BLOCK_N;
            int col = idx % BLOCK_N;
            int gRow = kOuter + row;
            int gCol = blockColBase + col;
            sB[row][col] = (gRow < N && gCol < N) ? B[(size_t)gRow * N + gCol]
                                                   : __float2half(0.0f);
        }

        __syncthreads();

        // --- Each warp performs BLOCK_K / WMMA_K = 4 WMMA MMAs ----------
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       half, wmma::row_major> b_frag;

        #pragma unroll
        for (int kInner = 0; kInner < BLOCK_K; kInner += WMMA_K) {
            const half *tile_a = &sA[warpY * WMMA_M][kInner];
            const half *tile_b = &sB[kInner][warpX * WMMA_N];

            wmma::load_matrix_sync(a_frag, tile_a, BLOCK_K + 8);
            wmma::load_matrix_sync(b_frag, tile_b, BLOCK_N + 8);
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        __syncthreads();
    }

    // --- Store the 16×16 accumulator tile to global C -------------------
    int outRow = blockRowBase + warpY * WMMA_M;
    int outCol = blockColBase + warpX * WMMA_N;
    if (outRow < N && outCol < N) {
        float *c_ptr = C + (size_t)outRow * N + outCol;
        wmma::store_matrix_sync(c_ptr, acc, N, wmma::mem_row_major);
    }
}

int main(int argc, char **argv) {
    int N = 4096;
    if (argc > 1) N = atoi(argv[1]);

    // N must be a multiple of 16 for WMMA
    if (N % 16 != 0) {
        N = ((N + 15) / 16) * 16;
        printf("Rounded N up to %d (must be multiple of 16)\n", N);
    }

    size_t bytes_fp16 = (size_t)N * N * sizeof(half);
    size_t bytes_fp32 = (size_t)N * N * sizeof(float);

    // Host allocations
    half  *h_A = (half  *)malloc(bytes_fp16);
    half  *h_B = (half  *)malloc(bytes_fp16);
    float *h_C = (float *)malloc(bytes_fp32);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = __float2half((float)(rand() % 100) / 100.0f);
        h_B[i] = __float2half((float)(rand() % 100) / 100.0f);
    }

    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, bytes_fp16);
    cudaMalloc(&d_B, bytes_fp16);
    cudaMalloc(&d_C, bytes_fp32);
    cudaMemcpy(d_A, h_A, bytes_fp16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_fp16, cudaMemcpyHostToDevice);

    dim3 block(THREADS);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N,
              (N + BLOCK_M - 1) / BLOCK_M);

    // Warmup
    matmul_tensor_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_tensor_kernel<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // 2*N^3 FP16 FLOPs (each MMA does multiply-add)
    double flops = 2.0 * N * N * N;
    double tflops = flops / (ms * 1e9);
    printf("MATMUL_TENSOR (WMMA FP16) N=%d  time=%.3f ms  TFLOPS=%.2f\n",
           N, ms, tflops);

    cudaMemcpy(h_C, d_C, bytes_fp32, cudaMemcpyDeviceToHost);
    printf("C[0][0] = %.4f (sanity check)\n", h_C[0]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
