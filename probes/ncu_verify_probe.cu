/**
 * ncu_verify_probe.cu  –  Lightweight verification kernels for ncu cross-check.
 *
 * Contains TWO tiny kernels designed to be profiled by ncu in < 30 s:
 *   1. A global memory READ kernel   → derives peak DRAM bandwidth from ncu
 *   2. A sustained FMA compute kernel → derives actual clock frequency from ncu
 *
 * The Agent runs:
 *     ncu --csv --launch-count 1 --metrics <list> build/ncu_verify_probe
 *   and extracts dram__throughput, sm__cycles_elapsed, gpu__time_duration to
 *   cross-verify its own micro-benchmark results.
 *
 * NOTE: stdout is just a sentinel; the real data comes from ncu CSV output.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/* Kernel 0  –  global memory read  (large vectorised read)           */
/* ------------------------------------------------------------------ */
__global__ void ncu_bw_kernel(const float4 *__restrict__ src, float *dst, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float4 accum = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int i = tid; i < n; i += stride) {
        float4 v = src[i];
        accum.x += v.x;
        accum.y += v.y;
        accum.z += v.z;
        accum.w += v.w;
    }
    if (tid == 0) dst[0] = accum.x + accum.y + accum.z + accum.w;
}

/* ------------------------------------------------------------------ */
/* Kernel 1  –  sustained FMA compute  (clock estimation from ncu)    */
/* ------------------------------------------------------------------ */
__global__ void ncu_compute_kernel(float *out, int iters) {
    float a = 1.0001f, b = 0.9999f;
    #pragma unroll 8
    for (int i = 0; i < iters; i++) {
        a = __fmaf_rn(a, b, a);
    }
    if (threadIdx.x == 0) out[0] = a;
}

int main() {
    printf("NCU_VERIFY_START\n");

    /* ----------  Kernel 0: Bandwidth  ---------- */
    const size_t N_FLOAT4 = 64 * 1024 * 1024 / sizeof(float4);  // 64 MB
    float4 *d_src;  float *d_dst;
    cudaMalloc(&d_src, N_FLOAT4 * sizeof(float4));
    cudaMalloc(&d_dst, sizeof(float));
    cudaMemset(d_src, 1, N_FLOAT4 * sizeof(float4));

    ncu_bw_kernel<<<256, 256>>>(d_src, d_dst, N_FLOAT4);
    cudaDeviceSynchronize();
    printf("NCU_VERIFY_BW_KERNEL_DONE\n");

    /* ----------  Kernel 1: Compute  ------------ */
    float *d_out;
    cudaMalloc(&d_out, sizeof(float));
    ncu_compute_kernel<<<1, 1>>>(d_out, 20000000);   // ~20M FMA → quick
    cudaDeviceSynchronize();
    printf("NCU_VERIFY_COMPUTE_KERNEL_DONE\n");

    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_out);
    printf("NCU_VERIFY_END\n");
    return 0;
}
