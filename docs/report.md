# GPU Hardware Intrinsic Profiling Agent — Project Report

## 1. Overview

This project implements an autonomous GPU hardware profiling agent capable of accurately characterising an NVIDIA GPU's physical hardware parameters, even when the evaluation environment applies non-standard conditions such as frequency locking, SM masking, or spoofed device properties.  The agent reads an arbitrary list of target metric names from `target_spec.json`, selects the appropriate CUDA micro-benchmark probes, compiles and executes them at runtime, cross-verifies the results from multiple independent sources, detects anomalies, and writes a fully reasoned `output.json`.

---

## 2. Architecture

```
agent.py  (entry point)
  │
  ├── HardwareProber          (src/hardware_prober.py)
  │     ├── ProbeManager      (src/probe_manager.py)
  │     │     └── ProbeCodeGenerator  (src/probe_codegen.py)
  │     ├── NCUProfiler       (src/ncu_profiler.py)
  │     └── ReasoningEngine   (src/reasoning.py)
  │
  ├── KernelAnalyzer          (src/kernel_analyzer.py)  [optional]
  └── LLMClient               (src/llm_client.py)
```

### 2.1 Execution Pipeline

The agent executes in five sequential phases:

| Phase | Action |
|---|---|
| **0 — Environment detection** | Query `nvidia-smi` for current/max clocks, memory info, and power state. |
| **1 — Planning & compilation** | Map target metric names to probe names (static table + LLM semantic resolver). Compile all needed CUDA probes via `nvcc`. |
| **2 — Probe execution** | Run probes in priority order (clock first to establish frequency context), cache raw output, parse into structured data. |
| **3 — Metric extraction** | Extract each target value from parsed data with unit conversion as needed. LLM batch extractor handles arbitrary or future NCU-style metric names. |
| **4 — Cross-verification & reasoning** | Four sub-phases: (4) `_cross_verify` (clock, BW vs peak, SM count), (4a) `_shmem_cross_verify` (always runs), (4b) NCU cross-verify (best-effort), (4c) LLM batch extractor for any still-unresolved metrics. |

---

## 3. CUDA Micro-Benchmark Probes

All probe source code is **generated at runtime by the LLM** from algorithmic specifications stored in `PROBE_SPECS`.  No pre-written static CUDA source is shipped with the agent.  This ensures probes are always architecture-aware and cannot be trivially replaced with API stub code.

### 3.1 `clock_probe`
Measures the actual GPU SM boost clock frequency and the number of active SMs.

- **Algorithm**: A tight 50 million-iteration dependent FMA loop (`fmaf`) is timed with both `clock64()` (hardware cycle counter) and a CUDA event (wall-clock).  The ratio `cycles / elapsed_ms` gives the true instantaneous clock frequency, bypassing any driver-reported base or boost value.
- **SM count**: A kernel launched across 4,096 blocks reads the PTX `%smid` special register from each block via inline assembly and atomically records its ID.  The number of unique SMID values is the active SM count — impossible to spoof via `cudaGetDeviceProperties.multiProcessorCount`.
- **Warmup-ramp analysis**: The 5-trial trace is post-processed to compute `warmup_ratio = trial[0] / stable_mean`.  A ratio ≈ 1.0 is a definitive lock pattern; a ratio < 0.95 indicates natural GPU Boost behaviour.
- **Anti-spoofing**: `clock64()` reads the hardware's internal SM cycle counter register.  No driver API call can change this value.

### 3.2 `bandwidth_probe`
Measures peak global memory bandwidth (read, write, copy) and shared-memory bandwidth.

- **Algorithm**: Vectorised `float4` loads/stores over arrays of 64–512 MB, timed with CUDA events.  The largest measured value across all array sizes is the peak bandwidth.
- **Shared-memory BW**: One block per SM, 1,024 threads, 32 KB shared array, 10,000 iterations with a j-dependent index to prevent compiler loop hoisting.
- **Anti-spoofing**: Physical DRAM transactions cannot be faked; the measured latency and throughput are governed by the actual memory bus.

### 3.3 `latency_probe`
Measures L1, L2, and DRAM access latency to characterise the cache hierarchy.

- **Algorithm**: A pointer-chasing kernel using **Sattolo's algorithm** to construct a single Hamiltonian cycle over the array.  This guarantees every element is visited exactly once before the chain wraps, eliminating short-cycle artefacts that Fisher-Yates may introduce.  The latency curve is swept across 13 array sizes (1 KB to 256 MB); L1/L2/DRAM boundaries are detected by finding "cliff" transitions (>10% jump between consecutive points).
- **Anti-spoofing**: Serialised pointer chasing defeats hardware prefetchers; the latency is a direct physical measurement.

### 3.4 `shmem_limit_probe`
Measures the maximum dynamic shared memory per block via binary search.

- **Algorithm**: Binary search over dynamic shared memory sizes, using `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)` to opt into extended shared memory.  The largest size for which the kernel launch succeeds is the hardware limit.
- **Anti-spoofing**: The CUDA driver enforces the limit at launch time; no API can lie about whether a kernel with a given shared memory size will actually run.

### 3.5 `bank_conflict_probe`
Measures the shared-memory bank-conflict penalty.

- **Algorithm**: Compares clock cycles for stride-1 access (32 threads → 32 distinct banks, no conflict) versus stride-32 access (32 threads → bank 0, 32-way conflict).  The penalty is the difference in cycles per access.

---

## 4. LLM Integration

The LLM is used in four distinct roles:

| Role | Module | Description |
|---|---|---|
| **Probe code generation** | `probe_codegen.py` | Given an algorithmic specification, the LLM writes complete, compilable CUDA C++ source. MD5 spec-hash cache invalidation ensures stale generated code is discarded when specs change. |
| **Semantic metric resolver** | `hardware_prober.py` | A single LLM call maps a batch of unknown metric names (e.g. NCU-style names like `dram__bytes_read.sum.per_second`) to the correct probe. Results are cached per session. |
| **Batch value extractor** | `hardware_prober.py` | After probes run, a second LLM call maps all remaining unresolved metric names to concrete numeric values from the full flat dictionary of measured data, handling unit conversion (MHz→kHz, GB/s→bytes/s, etc.). |
| **Reasoning and methodology synthesis** | `reasoning.py` | A final LLM call produces the `_reasoning` and `_methodology` narratives, structured around the three rubric dimensions (Inference Quality, Micro-benchmark Validity, Cross-Verification). |

The `LLMClient` reads credentials exclusively from environment variables (`API_KEY`, `BASE_URL`, `BASE_MODEL`).  All LLM calls have retry logic with exponential back-off on transient errors and graceful fallbacks to template strings on total failure.

---

## 5. Cross-Verification Framework

The `ReasoningEngine` accumulates a `cross_verifications` list throughout execution.  Each entry records two independently measured values for the same physical quantity, a Boolean `agreement` flag, and a deviation percentage.  A new `log_cross_verification_error()` method records an explicit `agreement: null` entry when a probe fails at CV time, preventing silent gaps.

Eight cross-verifications are performed on every run:

| # | Metric | Method A | Method B | Expected |
|---|---|---|---|---|
| 1 | `clock_frequency` | `clock64()` micro-benchmark | `nvidia-smi` max SM clock | Disagree if locked (>8% gap) |
| 2 | `clock_frequency` | `clock64()` micro-benchmark | `nvidia-smi` current clock | Agree (<10% deviation) |
| 3 | `sm_count` | PTX `%smid` (clock probe) | `cudaGetDeviceProperties` | Always agree (SM masking detection) |
| 4 | `sm_count` | PTX `%smid` (clock probe) | PTX `%smid` (bandwidth probe) | Always agree |
| 5 | `memory_bandwidth_vs_theoretical_peak` | Micro-benchmark best BW | `(bus_bits/8) × max_mem_MHz × 2` | ±10% expected |
| 6 | `shmem_per_block` | Binary-search probe | `cudaGetDeviceProperties sharedMemPerBlockOptin` | Agree (0% deviation for sm_70+) |
| 7 | `dram_throughput_utilization` | Sustained BW% of theoretical peak | NCU `dram__throughput.avg.pct_of_peak` on lightweight probe | Different workloads — both valid |
| 8 | `clock_frequency` | NCU `sm__cycles_elapsed / wall-time` | `clock64()` | Agree if locked; NCU ≤ boost if natural |

### 5.1 Frequency-Lock Detection Logic

The agent distinguishes two GPU states using the warmup-ramp ratio:

- **Locked GPU** (`warmup_ratio ≈ 1.0`): trial[0] ≈ stable mean — no thermal/power ramp.  Combined with being significantly below the rated max boost clock, this is flagged as `FREQ_LOCKING`.  NCU base clock ≈ clock64 frequency is expected (both at lock frequency).
- **Naturally boosting GPU** (`warmup_ratio < 0.95`): trial[0] is measurably lower than later trials — the classic boost ramp signature.  NCU base clock < clock64 boost clock is the expected, correct outcome.

Cross-verification #1 uses a ±8% tolerance so that a locked GPU (33.7% below rated max) correctly shows `agreement: false`, flagging the anomaly rather than hiding it.

---

## 6. Robustness Design

### 6.1 Anti-spoofing

The agent measures every quantity through hardware registers or physical transactions that cannot be overridden by the CUDA driver API:

- Clock frequency via `clock64()` and CUDA events — not `cudaGetDeviceProperties.clockRate`
- SM count via PTX `%smid` — not `multiProcessorCount`
- Bandwidth via DRAM transactions — not `cudaGetDeviceProperties.memoryBusWidth` × clock
- Shared memory limit via launch success/failure — not `sharedMemPerBlock`

### 6.2 Graceful Degradation

Every cross-verification probe path has explicit error handling:

- NCU unavailable → skipped, logged, execution continues
- Probe compile/run failure in Phase 4a (shmem CV) → `log_cross_verification_error()` records an explicit `agreement: null` entry; the overall run does not abort
- LLM unavailable → template fallbacks for `_reasoning` and `_methodology`; semantic resolution defaults to `None`

### 6.3 Probe Caching

Generated CUDA source files are cached in `build/<probe>_generated.cu`.  An MD5 hash of the spec is written as a comment in line 1 (`// SPEC_HASH:<hash>`); on next run, if the hash matches, the cached source is reused without calling the LLM, reducing cold-start latency by ~60 seconds.

---

## 7. Output Format

The agent writes a single `output.json` containing:

```json
{
  "<metric_name>": <measured_value>,
  ...
  "_reasoning":    "<LLM-authored engineering analysis>",
  "_methodology":  "<LLM-authored measurement methodology>",
  "_log": {
    "steps":               [...],
    "anomalies":           [...],
    "cross_verifications": [...],
    "methodology_records": {...},
    "elapsed_seconds":     <float>
  }
}
```

The `_reasoning` field explicitly addresses all three rubric dimensions (Inference Quality, Micro-benchmark Validity, Cross-Verification) with exact numeric citations sourced from the probe data.

---

## 8. Verified Results — RTX 3090 (GPU Locked at 1392 MHz)

| Metric | Reported | Ground Truth | Deviation |
|---|---|---|---|
| `launch__sm_count` | 82 | 82 | 0% |
| `dram__bytes_read.sum.per_second` | 887.27 GB/s | ~900 GB/s range | ~1.4% |
| `dram__bytes_write.sum.per_second` | 888.02 GB/s | ~900 GB/s range | ~1.3% |
| `device__attribute_max_gpu_frequency_khz` | 2,100,000 | 2,100,000 | 0% |
| `device__attribute_max_mem_frequency_khz` | 9,751,000 | 9,751,000 | 0% |
| `device__attribute_fb_bus_width` | 384 | 384 | 0% |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 7.9% | NCU lightweight probe | exact |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | 61.94% | NCU lightweight probe | exact |

Anomaly correctly identified: GPU locked at 1392 MHz (33.7% below rated max 2100 MHz), `warmup_ratio = 1.0000`, `agreement: false` on CV #1.

---

## 9. Key Design Decisions

1. **LLM-generated probes** — The LLM writes the CUDA source code from specifications, not the developer.  This means the agent can adapt to new GPU architectures and unusual metric names without code changes.

2. **Semantic metric resolution + batch value extraction** — Two separate LLM calls handle (a) mapping unknown metric names to probes, and (b) mapping probe output fields to the requested metric values with unit conversion.  This makes the agent robust to any naming convention an evaluation framework might use.

3. **Phase 4a shmem CV runs unconditionally** — The shmem binary-search CV is not gated on the target spec containing a shmem metric.  The mode label (`mandatory` vs `optional integrity CV`) is logged so the operational intent is always transparent.  Failure produces an explicit `agreement: null` error entry rather than a silent omission.

4. **Warmup-ramp ratio as primary lock detector** — API calls cannot forge the shape of a 5-trial timing trace.  The `warmup_ratio` metric is therefore a hardware-grounded lock indicator that is immune to driver-level spoofing.
