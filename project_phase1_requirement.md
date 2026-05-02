# MLSYS Course Project

With the dawn of the Agent Era, the landscape of systems engineering is undergoing a fundamental paradigm shift. Traditionally, tasks such as operator analysis and deep GPU performance profiling were reserved for a small circle of elite experts who possessed deep, vertical knowledge spanning from high-level algorithms down to the intricacies of GPU hardware architecture. In the past, manually interpreting complex hardware metrics to identify performance bottlenecks would have required immense time and specialized expertise.

Today, that barrier is dissolving. We are entering a time where a single individual, empowered by intelligent agents, can perform the deep profiling work that once required seasoned systems engineers. By designing agents that can perceive hardware behavior through raw metrics, reason through memory and compute limitations, and pinpoint the exact causes of inefficiency, we can automate the "expert" layer of performance analysis. This is the core objective of this project: to design an autonomous system capable of utilizing NVIDIA Nsight Compute (ncu) to evaluate CUDA kernels, analyze hardware execution data, and accurately diagnose GPU performance bottlenecks.

---

## Section 1: GPU Performance Analysis Guide: Identifying Bottlenecks with ncu

**NVIDIA Nsight Compute (ncu)** is an interactive kernel-level profiling tool for CUDA kernels. Unlike Nsight Systems (nsys), which observes the global timeline, `ncu` dives deep into the internal execution of each kernel to show how hardware resources are being consumed.

In this project, your Agent needs to analyze the metrics output by `ncu` to determine the bottleneck of an operator (e.g., Matrix Multiplication). Below are the core metric categories and their significance in performance tuning.

### 1.1 Core Overview Metrics

Before performing a detailed analysis, first use the **Roofline Model** to determine if the operator is **Compute-Bound** or **Memory-Bound**.

- **`sm__throughput.avg.pct_of_peak_sustained_elapsed` (Compute Utilization)**
  - **Meaning**: The utilization rate of the SM (Streaming Multiprocessor) computing units.
  - **Analysis**: If this value is high, it indicates that the GPU's compute cores are fully loaded.

- **`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` (Memory Throughput)**
  - **Meaning**: The utilization rate of the GPU's memory bandwidth.
  - **Analysis**: If this value is high, data transfer speeds are limiting performance.

- **SOL (Speed of Light)**: The "SOL Compute" and "SOL Memory" frequently mentioned in `ncu` reports represent what percentage of the hardware's theoretical peak performance the current operator has reached.

### 1.2 Memory Hierarchy Metrics

If the operator is memory-intensive, you need to identify which layer of the storage hierarchy is the bottleneck.

- **`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` (L1 Cache)**
  - **Analysis**: L1 cache hit rate and throughput. Frequent global memory accesses that miss the L1 cache significantly increase latency.

- **`l2__throughput.avg.pct_of_peak_sustained_elapsed` (L2 Cache)**
  - **Analysis**: L2 serves as the bridge between VRAM and the SM. Extremely high L2 utilization suggests massive data exchange between these layers.

- **`dram__throughput.avg.pct_of_peak_sustained_elapsed` (VRAM/DRAM)**
  - **Analysis**: External VRAM bandwidth. If this exceeds 80%, your code likely needs better data reuse (e.g., utilizing Shared Memory).

### 1.3 Compute Unit Metrics

If the operator is compute-bound, you need to identify which specific compute units are active.

- **`sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active` (Tensor Cores)**
  - **Analysis**: **Crucial!** For matrix multiplications in deep learning, if this metric is low, it means you are not effectively utilizing the Tensor Cores.

- **`sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` (FP32/FMA)**
  - **Analysis**: Utilization of traditional single-precision floating-point units.

- **`sm__sass_thread_inst_executed_op_fp32_pred_on.sum`**
  - **Analysis**: The total number of FP32 instructions actually executed.

### 1.4 Occupancy and Scheduling

Sometimes hardware utilization is low because threads are failing to "fill up" the GPU.

- **`sm__maximum_warps_per_active_cycle_pct` (Theoretical Occupancy)**
  - **Meaning**: The maximum percentage of warps that can hardware-**theoretically** run simultaneously, based on register and shared memory usage.

- **`sm__warps_active.avg.pct_of_peak_sustained_active` (Achieved Occupancy)**
  - **Meaning**: The occupancy **actually achieved** during the operator's execution.
  - **Analysis**: If Achieved Occupancy is much lower than Theoretical Occupancy, it indicates **instruction latency** issues or uneven thread block distribution.

### 1.5 Common Bottlenecks & Diagnostic Table

| Bottleneck Type        | Key Metrics                                            | Optimization Direction                                       |
| :--------------------- | :----------------------------------------------------- | :----------------------------------------------------------- |
| **VRAM Bound**         | `dram__throughput` > 70%                               | Reduce memory access; increase data reuse; use Shared Memory. |
| **Compute Bound**      | High `sm__throughput`, high `tensor_op_hmma`           | Consider algorithmic improvements or reduce precision (e.g., FP32 to FP16/BF16). |
| **Uncoalesced Access** | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld` too high  | Check memory access patterns; ensure adjacent threads access adjacent addresses. |
| **Warp Divergence**    | `sm__sass_thread_inst_executed_per_inst_executed` < 32 | Reduce `if/else` branches; ensure threads within a warp follow the same path. |
| **Bank Conflict**      | `l1tex__data_bank_conflicts_pipe_lsu.sum` > 0          | Adjust Shared Memory indexing (e.g., using Padding).         |

### 1.6 Advice for Students: How to Let the Agent Use This Data?

1. **Step 1: Get the Roofline**. Have the Agent read `sm__throughput` and `gpu__compute_memory_throughput` first.
2. **Step 2: Characterize**.
   - If Memory % > Compute %, dive into `dram` and `l1/l2` metrics.
   - If Compute % > Memory %, check if `tensor_op` is being triggered.
3. **Step 3: Look for Anomalies**. Check if `Occupancy` is too low or if `Bank Conflict` exists.
4. **Step 4: Map back to Code**. Relate these metrics to your CUDA kernel code (e.g., is the loop unrolling insufficient? Is a `__shared__` array missing?).

### How to acquire these metrics via Command Line?

```bash
# Example: Get all detailed metrics for a Matrix Multiplication kernel
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active ./your_executable
```

### 1.7 Hardware Intrinsic Profiling: Probing the GPU "DNA"

In this advanced phase of the project, your Agent is no longer just a passive reporter of existing kernel performance. It must act as a **Hardware Probe**. The goal is to "reverse-engineer" the physical characteristics and architectural limits of the underlying GPU by autonomously generating and analyzing micro-benchmarks.

#### 1.7.1 Probing Objectives (Target Metrics)

Your Agent will be given a task to identify the following hardware-intrinsic parameters:

1. **Memory Latency Hierarchy**: Measure the exact access cycles for **L1 Cache**, **L2 Cache**, and **DRAM**. This requires the Agent to generate "Pointer Chasing" kernels to bypass hardware prefetchers.
2. **Effective Peak Bandwidth**: Determine the maximum achievable throughput for **Shared Memory** and **Global Memory (VRAM)** under current conditions.
3. **L2 Cache Capacity**: Identify the "cliff" in the latency-vs-size curve to pinpoint the exact physical size of the L2 cache.
4. **Actual Boost Frequency**: Report the stable core clock frequency (MHz) while the GPU is under sustained compute load.
5. **Resource Penalties**: Quantify the latency cost of a **bank conflict** in Shared Memory compared to a conflict-free access.

#### 1.7.2 Submission & Evaluation Workflow

- **Student Submission**: You must submit your **Agent System (Infra Code)**.
- **Input**: The evaluation system will provide a `target_spec.json` containing the list of hardware metrics to be identified.
  - *Example Input*: `{"targets": ["dram_latency_cycles", "max_shmem_per_block_kb", "actual_boost_clock_mhz"]}`
- **Output**: Your Agent must output a `results.json` containing the identified numeric values.
  - *Example Output*: `{"dram_latency_cycles": 442, "max_shmem_per_block_kb": 48, ...}`
- **Ground Truth Comparison**: Your Agent's output will be compared against a high-precision ground truth generated by our server-side reference benchmarks.

#### 1.7.3 Anti-Hacking & Environment Variations

To ensure your Agent is performing real hardware analysis rather than performing a simple "table lookup" (e.g., searching for "A100 specs"), the evaluation environment will be dynamically altered:

1. **Non-Standard Frequency Locking**: The GPU core and memory clocks may be locked at **arbitrary, non-standard frequencies** (e.g., 825 MHz instead of 1410 MHz) using `nvidia-smi`. A static lookup table will provide incorrect bandwidth/GFLOPS results.
2. **Resource Masking (SM Limiting)**: The system may restrict the kernel execution to a **subset of SMs** or limit the available memory per block via CUDA environment variables.
3. **Instruction Set Restrictions**: Standard API calls like `cudaGetDeviceProperties` may be intercepted or report virtualized/misleading data.

**Recommendation**: Your Agent should adopt a **Multi-Strategy Fusion** approach—combining low-level micro-benchmarking (writing small C++/CUDA probes), binary execution, and `ncu` metric analysis to cross-verify its findings.

#### 1.7.4 LLM-Based Evaluation and Scoring

To ensure a holistic assessment that covers both numerical precision and engineering reasoning, this project employs an **LLM-as-a-Judge** framework for grading.

**The Grading Process:** Upon submission, the evaluation system will feed the following three components into a high-capability Large Language Model (e.g., Gemini 3.1 Pro):

1. **Student Agent Output**: The final `results.json` containing the identified hardware metrics and the reasoning/logs provided by your Agent.
2. **Ground Truth Data**: The exact hardware parameters measured by our reference benchmarks under the specific environment (including any active frequency locks or resource masking).
3. **Experimental Evidence**: Summaries of the `ncu` traces and micro-benchmarks generated during the Agent's execution.

**Scoring Rubric (100 Points Total):**

- **Numerical Alignment (70 Points)**: The LLM will compare your Agent's reported values against the Ground Truth.
  - **Full Marks**: Values are within an acceptable engineering tolerance (e.g., ±5% for latency, ±2% for bandwidth).
  - **Partial Marks**: Values follow the correct trend but show calibration errors.
  - **Zero Marks**: Values match standard "online spec tables" but fail to reflect the **actual throttled/masked environment** provided during the test.

- **Engineering Reasoning & Methodology (30 Points)**: The LLM will evaluate the "how" and "why" behind your Agent's conclusions.
  - **Inference Quality**: Did the Agent correctly identify that the GPU was locked at a non-standard frequency? Did it notice SM masking?
  - **Micro-benchmark Validity**: Did the Agent generate appropriate CUDA kernels (e.g., pointer-chasing for latency) or did it rely on potentially restricted API calls?
  - **Cross-Verification**: Did the Agent perform multiple trials or different types of probes to confirm a suspicious metric?