# Kernel Bottleneck Analysis: `matmul_naive_kernel`

### 1. Roofline Classification
**Verdict:** **Compute-Bound** (Pipeline Saturation).

The kernel reports an SM Throughput (Compute SOL) of **98.1%**, indicating that the Streaming Multiprocessor pipelines are fully occupied. However, this is "inefficient" saturation. While the roofline model classifies it as compute-bound due to the high SOL, the **FMA/FP32 Utilization is only 19.3%**. This disparity reveals that the compute pipelines are stalled waiting for data rather than performing arithmetic. The extremely low DRAM Throughput (1.2%) confirms this is not a bandwidth saturation issue; rather, the kernel is choked by internal serialization and memory contention inside the SM.

### 2. Primary Bottleneck
The critical bottleneck is **L1 Bank Conflicts** and **Non-Coalesced Global Memory Access**.

The profiling data shows **32.6 million bank conflicts** against roughly 1.07 billion FP32 instructions. This ratio is catastrophic for performance. In a naive matrix multiply implementation, threads in a warp accessing a column of matrix $B$ (if stored row-major) result in strided access patterns.
1.  **Uncoalesced Loads:** Adjacent threads attempt to read non-adjacent memory addresses. This scatters the memory transactions, generating excessive L1 Global Load Sectors ($1.34 \times 10^8$).
2.  **Bank Conflicts:** When these scattered reads land in shared memory (or L1 cache banks), threads within a warp request addresses residing in the same memory bank, serializing the access.

The SM is "busy" at 98.1% throughput simply because the warp schedulers are occupied managing these memory stalls and serialization delays, preventing the issuance of FMA instructions.

### 3. Actionable Optimisation Recommendations
To resolve this, the code must transition from a naive approach to a tiled approach:

*   **Shared Memory Tiling:** Refactor the kernel to load tiles of matrices $A$ and $B$ into `__shared__` memory.
    *   *Pattern:* Have threads collaboratively load a tile (e.g., $16\times16$) ensuring **coalesced global reads** (adjacent threads read adjacent addresses).
    *   *Impact:* This will reduce the excessive L1 sector count and DRAM traffic by a factor of the tile dimension (e.g., 16x).
*   **Bank Conflict Avoidance:** Pad the shared memory array to prevent threads from accessing the same bank.
    *   *Pattern:* Declare shared memory as `__shared__ float tile[BLOCK_SIZE][BLOCK_SIZE+1];`.
    *   *Impact:* This simple padding offsets the memory stride, ensuring unique bank access per thread, eliminating the 32M bank conflicts.
*   **Enable Tensor Cores:** The kernel currently utilizes **0.0%** of available Tensor Cores on this SM 8.6 device.
    *   *Recommendation:* Use `wmma::load_matrix_sync` and `wmma::mma_sync` intrinsics with FP16 inputs.

### 4. Secondary Issues
While **Occupancy** is healthy (95.2%), the lack of L2 throughput (0.0%) indicates that the data reuse strategy is non-existent. Furthermore, the reliance on `float` (FP32) limits throughput compared to the Tensor Core capabilities of the RTX 30-series architecture.