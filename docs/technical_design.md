# Technical Design Document — GPU Hardware Intrinsic Profiling Agent

**Version**: 1.0  
**Date**: April 2026  
**Platform**: NVIDIA GeForce RTX 3090 (SM 8.6, 82 SMs, 24 GB GDDR6X)  
**Runtime**: CUDA 12.4, Python 3.10, GLM-5 via DashScope API

---

## 1. System Overview

This system is an autonomous GPU profiling agent that accomplishes two complementary objectives:

1. **Hardware Intrinsic Probing** (Section 1.7): Reverse-engineers the GPU's physical characteristics (memory latency hierarchy, peak bandwidth, clock frequency, SM count, shared memory limits, bank conflict penalty) using custom CUDA micro-benchmarks.

2. **Kernel Bottleneck Analysis** (Sections 1.1–1.6): Profiles user-supplied CUDA kernels via NVIDIA Nsight Compute (ncu) and performs a structured 4-step diagnosis — Roofline classification → Subsystem deep-dive → Anomaly scan → Source-code mapping — culminating in an LLM-synthesised report.

The entire workflow is driven by a single `target_spec.json` file. No metric names or binary paths are hardcoded.

---

## 2. Architecture

### 2.1 Component Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │               target_spec.json              │
                    │  { "targets": [...], "run": "binary" }      │
                    └────────────────────┬────────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────────┐
                    │               agent.py (Orchestrator)        │
                    │  - Loads spec, dispatches Phase 1 & Phase 2 │
                    │  - Writes results.json + output files        │
                    └──────┬───────────────────┬──────────────────┘
                           │                   │
              ┌────────────▼────────┐   ┌──────▼───────────────┐
              │  Phase 1            │   │  Phase 2             │
              │  HardwareProber     │   │  KernelAnalyzer      │
              │  (hardware_prober)  │   │  (kernel_analyzer)   │
              └──┬──────┬──────┬───┘   └──┬──────┬────────────┘
                 │      │      │          │      │
        ┌────────▼┐ ┌───▼───┐ ┌▼────┐  ┌─▼──┐ ┌─▼──────────┐
        │ Probe   │ │ NCU   │ │Utils│  │NCU │ │LLM Client  │
        │ Manager │ │Profiler│ │     │  │Prof│ │(DashScope) │
        └────┬────┘ └───────┘ └─────┘  └────┘ └────────────┘
             │
    ┌────────▼───────────────────────────┐
    │  6 CUDA Micro-Benchmark Probes     │
    │  latency / bandwidth / clock /     │
    │  shmem_limit / bank_conflict / ncu │
    └────────────────────────────────────┘
```

### 2.2 Module Responsibilities

| Module | Lines | Responsibility |
|--------|------:|---------------|
| `agent.py` | 349 | CLI parsing, spec loading, Phase 1/2 dispatch, results serialisation |
| `hardware_prober.py` | 1,187 | Probe orchestration, output parsing, metric extraction, cross-verification, LLM-based semantic target resolution |
| `kernel_analyzer.py` | 591 | 4-step kernel analysis, source-file discovery, code-pattern scanning, LLM report generation |
| `ncu_profiler.py` | 475 | ncu invocation, CSV parsing (with fallback), roofline/memory/compute/occupancy analysis, bottleneck identification |
| `reasoning.py` | 363 | Step/anomaly/cross-verification logging, LLM-authored `_reasoning` + `_methodology` narratives |
| `llm_client.py` | 161 | OpenAI-compatible client for DashScope (GLM-5), streaming, retry with tenacity |
| `probe_manager.py` | 166 | nvcc auto-detection, architecture-fallback compilation, binary execution |
| `utils.py` | 120 | nvidia-smi queries, `median()`, `trimmed_mean()`, CUDA env checks |

### 2.3 Data Flow

```
target_spec.json
       │
       ▼
┌─── Phase 1 ────────────────────────────────────────────────┐
│  1. Resolve unknown targets → LLM semantic batch resolver  │
│  2. Plan needed probes (deduplicate)                       │
│  3. Compile CUDA probes via nvcc                           │
│  4. Execute probes (clock first, then latency/bw/shmem)   │
│  5. Parse stdout → structured data points                  │
│  6. Extract target metrics from parsed data                │
│  7. Cross-verify: probe vs probe, probe vs nvidia-smi,    │
│     probe vs ncu                                           │
│  8. Detect anomalies (freq locking, SM masking)            │
│  9. LLM generates final _reasoning + _methodology         │
└───────────────────────────────────────────────────┬────────┘
                                                    │
       ▼                                            ▼
┌─── Phase 2 ──────────────────────────┐    results.json
│  1. Roofline: sm_throughput vs       │
│     memory_throughput → classify     │
│  2. Characterise: memory → dram/l2   │
│     or compute → tensor_core/fma     │
│  3. Anomaly scan: occupancy, bank    │
│     conflicts, divergence            │
│  4. Map to Code: find .cu source,    │
│     scan for patterns (__shared__,   │
│     #pragma unroll, wmma)            │
│  5. LLM synthesis: metrics + source  │
│     + code issues → narrative        │
└──────────────────────────────────────┘
```

---

## 3. Micro-Benchmark Design

### 3.1 Latency Probe (latency_probe.cu)

**Objective**: Measure L1, L2, and DRAM access latencies; determine L2 cache capacity.

**Technique**: Single-thread pointer-chasing with random stride. A linked list is created in device memory where each element points to a pseudo-random next element. The thread follows the pointer chain and measures clock cycles per access via `clock64()`.

**Cache Hierarchy Detection**: The probe sweeps array sizes from 4 KB to 32 MB. The `_analyze_latency_curve()` method in `hardware_prober.py` identifies:
- **L1 latency**: Minimum stable latency for arrays ≤ 16 KB
- **L1→L2 transition**: First point where latency > 2× L1 latency
- **L2 latency**: Median of consecutive-stable points (< 2% rate of change)
- **L2→DRAM cliff**: First pair where rate of change > 10% after the L2 plateau
- **L2 cache size**: Array size at the cliff boundary
- **DRAM latency**: Maximum latency at largest array sizes

### 3.2 Bandwidth Probe (bandwidth_probe.cu)

**Objective**: Measure peak global memory and shared memory bandwidth.

**Technique**:
- **Global read/write**: Uses `float4` vectorised loads/stores across all SMs, with arrays from 16 MB to 512 MB. CUDA events provide wall-clock timing. Reports peak across all sizes.
- **Shared memory**: One block per SM, 1024 threads, bank-conflict-free access pattern. Aggregate and per-SM bandwidth reported.

### 3.3 Clock Probe (clock_probe.cu)

**Objective**: Measure actual GPU boost clock frequency and count active SMs.

**Technique**:
- **Clock frequency**: Runs 50M dependent FMA operations with a 100K-iteration warmup to reach stable boost. Measures both `clock64()` cycles and CUDA-event wall-clock time. Frequency = cycles / time. Median of multiple trials.
- **Active SM count**: Launches 4096 blocks, each writes its SM ID via inline PTX `asm("mov.u32 %0, %%smid;")`. Counts unique IDs on host.

### 3.4 Shared Memory Limit Probe (shmem_limit_probe.cu)

**Objective**: Determine maximum dynamic shared memory allocation per block.

**Technique**: Binary search over allocation sizes. Uses `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, size)` to opt into extended shared memory (beyond the default 48 KB limit on Ampere GPUs). Reports both default and extended limits.

### 3.5 Bank Conflict Probe (bank_conflict_probe.cu)

**Objective**: Measure the cycle penalty of shared memory bank conflicts.

**Technique**: Compares access times for different strides:
- **Stride 1**: 32 threads access 32 distinct banks (0 conflicts)
- **Stride 32**: 32 threads all access bank 0 (32-way conflict)

Penalty = conflict_cycles − no_conflict_cycles.

### 3.6 NCU Verification Probe (ncu_verify_probe.cu)

**Objective**: Lightweight kernel for ncu cross-verification of bandwidth and clock.

Provides an independent data point: ncu-measured DRAM throughput percentage against micro-benchmark bandwidth, and ncu-measured `sm__cycles_elapsed` against micro-benchmark clock frequency.

---

## 4. LLM Integration Architecture

### 4.1 Client Layer (`llm_client.py`)

- **Model**: GLM-5 via Alibaba Cloud DashScope (OpenAI-compatible endpoint)
- **Streaming**: Chunks are received iteratively; reasoning trace and answer content are separated
- **Retry**: tenacity-based, 3 attempts with exponential backoff (2–8s) for `APITimeoutError`, `APIConnectionError`, `RateLimitError`
- **Thinking mode**: Configurable via `enable_thinking` parameter. Thinking-on for deep analysis (~60s), thinking-off for fast classification (~100ms)

### 4.2 Usage Points

| Use Case | Module | Thinking | Timeout |
|----------|--------|----------|---------|
| Semantic target resolution | `hardware_prober.py` | Off | 30s |
| Anomaly analysis | `reasoning.py` | On | default |
| Engineering reasoning narrative | `reasoning.py` | On | default |
| Methodology narrative | `reasoning.py` | On | default |
| Kernel bottleneck report | `kernel_analyzer.py` | On | 120s |

### 4.3 Graceful Degradation

Every LLM call has a `fallback` string. If the API is unavailable or returns empty, the system falls back to template-based text — the agent produces complete, correct results regardless of LLM availability.

---

## 5. Cross-Verification Strategy

The agent implements a 3-tier verification pyramid:

```
            ┌──────────────┐
            │   Tier 3     │  nvidia-smi / CUDA API
            │ (sanity)     │  Device Properties, clocks, SM count
            └──────┬───────┘
            ┌──────▼───────┐
            │   Tier 2     │  NVIDIA Nsight Compute (ncu)
            │ (secondary)  │  DRAM throughput %, sm__cycles_elapsed
            └──────┬───────┘
            ┌──────▼───────┐
            │   Tier 1     │  CUDA Micro-Benchmarks
            │ (primary)    │  Direct hardware measurement
            └──────────────┘
```

**Verification pairs implemented:**

| Metric | Tier 1 (primary) | Tier 2/3 (secondary) |
|--------|-------------------|---------------------|
| Clock frequency | clock_probe (clock64/events) | ncu (sm__cycles/gpu__time), nvidia-smi (current clock) |
| SM count | clock_probe (PTX %smid) | bandwidth_probe (PTX %smid), cudaGetDeviceProperties |
| Global bandwidth | bandwidth_probe (float4 reads) | ncu-implied peak (bandwidth / DRAM_utilisation%) |
| Shared memory limit | shmem_limit_probe (binary search) | cudaGetDeviceProperties (reported value) |

Discrepancies generate anomaly entries with LLM-powered root-cause analysis.

---

## 6. Kernel Analysis Pipeline (Sections 1.1–1.6)

### 6.1 Step 1 — Roofline Classification (§1.1)

Reads two SOL (Speed of Light) percentages:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` → Compute SOL
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` → Memory SOL

Classification logic with 20% hysteresis:
- Memory SOL > 1.2 × Compute SOL → **MEMORY_BOUND**
- Compute SOL > 1.2 × Memory SOL → **COMPUTE_BOUND**
- Otherwise → **BALANCED**

### 6.2 Step 2 — Characterisation (§1.2 / §1.3)

**Memory-bound path**: Examines DRAM throughput (> 80% → VRAM-saturated), L2 throughput, L1 global load sectors.

**Compute-bound path**: Examines Tensor Core utilisation (`sm__pipe_tensor_op_hmma_cycles_active`) and FMA utilisation. Flags idle Tensor Cores when FMA is active.

### 6.3 Step 3 — Anomaly Scan (§1.4 / §1.5)

Detects seven bottleneck types with severity levels:

| Bottleneck | Trigger Condition | Severity |
|-----------|-------------------|----------|
| `VRAM_BOUND` | DRAM throughput > 70% | medium/high |
| `COMPUTE_BOUND` | SM throughput > 70% | medium/high |
| `TENSOR_CORES_UNUSED` | Tensor < 5% while FMA > 10% | medium |
| `BANK_CONFLICT` | Bank conflicts > 0 | low/medium/high |
| `WARP_DIVERGENCE` | Thread/inst ratio < 28/32 | medium/high |
| `EXCESSIVE_GLOBAL_LOADS` | L1 sectors > 50M | medium |
| `LOW_OCCUPANCY` | Occupancy gap > 20% | medium/high |

### 6.4 Step 4 — Map Back to Code (§1.6)

The analyser locates the `.cu` source file by walking up from the binary path and searching `kernels/`, `src/`, and parent directories. It then performs a heuristic pattern scan:

| Code Pattern Checked | Related Bottleneck |
|---------------------|-------------------|
| Missing `__shared__` | VRAM_BOUND / memory_bound |
| `__shared__` without tiling loop | Memory-bound with partial optimisation |
| Unpadded shared memory (no `+1`) | BANK_CONFLICT |
| No `wmma`/`mma` intrinsics | TENSOR_CORES_UNUSED |
| No `#pragma unroll` | Any non-trivial classification |
| High local variable count (> 30) | LOW_OCCUPANCY |

Both the ncu metrics and the kernel source code (line-numbered, truncated to 200 lines) are fed to the LLM for a professional ~500-word analysis report.

---

## 7. Semantic Target Resolution

To handle arbitrary metric naming conventions from evaluators, the `HardwareProber` includes a batch LLM resolver:

1. Targets not found in `METRIC_TO_PROBE` are collected.
2. A single LLM call (with `enable_thinking=False` for speed) classifies all unknowns.
3. The LLM prompt lists all 5 probes with descriptions and expects `metric -> probe_name` output.
4. Results are cached in `_semantic_cache` to avoid duplicate calls.
5. `_extract_metric()` uses the cache as a fallback lookup.

This ensures the agent handles names like `"mem_delay"` → `latency_probe`, `"gpu_frequency"` → `clock_probe`, etc.

---

## 8. Error Handling and Resilience

| Scenario | Handling |
|----------|---------|
| ncu not installed | Best-effort: Phase 2 returns error dict; Phase 1 ncu cross-verification is skipped |
| LLM API unavailable | All LLM calls have template fallbacks; results are complete but with less narrative |
| Probe compilation failure | Architecture fallback chain: native → sm_90 → sm_89 → sm_86 → ... → sm_60 |
| Probe execution timeout | 300s timeout with clear error logging |
| ncu streaming timeout | 30s (semantic), 120s (kernel analysis) timeouts on HTTP client |
| Unknown target metric | LLM semantic resolver; if still unknown, logged as warning and set to null |

---

## 9. Validated Results (RTX 3090)

| Metric | Measured Value | Validation |
|--------|---------------|-----------|
| L1 latency | 38.1 cycles | Consistent with Ampere L1 cache spec |
| L2 latency | 250.4 cycles | Within expected range for 6 MB L2 |
| DRAM latency | 515.3 cycles | Consistent with GDDR6X on GA102 |
| L2 cache size | 6,144 KB (6 MB) | Exact match with spec |
| Global bandwidth | 910.22 GB/s | ~93% of theoretical 936 GB/s peak |
| Shared memory BW | 68,655 GB/s | Aggregate across 82 SMs |
| Boost clock | 1,979.4 MHz | Within spec range (max 2,100 MHz) |
| Shared memory/block | 99 KB | Extended via `cudaFuncSetAttribute` (default: 48 KB) |
| Bank conflict penalty | 62.0 cycles | Consistent with 32-way conflict measurement |
| Active SMs | 82 | Exact match with RTX 3090 spec |
