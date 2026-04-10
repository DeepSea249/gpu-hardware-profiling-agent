# Kernel Bottleneck Analysis: `matmul_tensor_kernel`

### 1. Kernel Classification
The kernel is classified as **compute-bound**. According to the Speed-of-Light (SOL) metrics, the SM Throughput reached **71.9%**, significantly outpacing the Memory Throughput of **49.6%**. This disparity indicates that the pipeline is primarily limited by the execution units rather than the available memory bandwidth. However, as detailed below, this "compute-bound" status is a result of inefficient execution rather than optimal arithmetic intensity.

### 2. Primary Bottleneck: Shared Memory Bank Conflicts
The critical bottleneck is **shared memory bank conflicts**, flagged as `HIGH` priority with 1,750,862 conflicts recorded.

While the kernel is theoretically compute-bound, the 71.9% SOL indicates the SMs are active but not efficiently producing results. This is confirmed by the extremely low **Tensor Core Utilization (13.9%)**. The bank conflicts are causing thread serialization within warps. When multiple threads in a warp access different addresses within the same memory bank, the hardware splits the request into separate transactions, stalling the pipeline.

In the context of a matrix multiplication kernel, this typically occurs when accessing column-major data in shared memory. For example, a naive access pattern like `val = shared_mem[threadIdx.x][threadIdx.y]` can cause conflicts if the column width is a multiple of the bank size (32).

### 3. Optimisation Recommendations
To resolve this, apply the following code modifications:

*   **Shared Memory Padding:** Modify shared memory allocations to avoid bank alignment. Instead of `__shared__ half tile[TILE_SIZE][TILE_SIZE]`, declare it as `__shared__ half tile[TILE_SIZE][TILE_SIZE + 1]`. The padding shifts the starting address of each row, ensuring that column accesses by a warp stride across different banks.
*   **Vectorized Access:** Use vectorized loads (e.g., `float2` or `uint4`) to reduce the total number of memory transactions and improve coalescing on the global loads (currently seeing excessive L1 sectors).
*   **Native Tensor Core APIs:** The low Tensor Core usage suggests manual implementation overhead. Replace manual HMMA assembly or naive FP16 loops with the **CUDA WMMA (Warp Matrix Multiply Accumulate) API** or the **CublasLt** library calls. This ensures data is laid out optimally for Tensor Core consumption (fragment loading).

### 4. Secondary Issues
*   **Excessive Global Loads:** The high count of L1 Global Load Sectors (201M) suggests non-coalesced memory access patterns. Ensure that adjacent threads read adjacent memory addresses (stride-1 access) when loading input tiles from global memory to shared memory.
*   **Occupancy:** Achieved occupancy (65.7%) is reasonable and not the primary limiter. Optimizing the bank conflicts will naturally improve throughput without requiring further occupancy tuning.