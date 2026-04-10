# GPU Hardware Intrinsic Profiling Agent

An autonomous, LLM-powered agent for probing GPU hardware characteristics via CUDA micro-benchmarks, cross-verifying with NVIDIA Nsight Compute (ncu), and performing kernel-level bottleneck analysis — all driven by a single `target_spec.json`.

> **5,110 lines** of code (3,465 Python + 1,560 CUDA + 85 Shell) across 20 source files.

---

## Quick Start — Evaluation Entry Point

```bash
# 1. Place your target_spec.json in the project root
# 2. Run:
chmod +x run.sh && ./run.sh
```

Override defaults with CLI flags:

```bash
./run.sh --target-spec /path/to/spec.json --output results.json --trials 10
```

### target_spec.json Format

```jsonc
{
  "targets": ["dram_latency_cycles", "actual_boost_clock_mhz", ...],
  "run": "build/kernels/matmul_naive"   // optional: kernel binary to profile
}
```

| Field | Purpose | Required |
|-------|---------|----------|
| `targets` | Hardware metrics to measure via micro-benchmarks (Section 1.7) | No |
| `run` | CUDA executable to profile for bottleneck analysis (Sections 1.1–1.6) | No |

Both fields are optional — the agent runs whichever phases are requested. Target names are **not hardcoded**: unknown metric names are resolved at runtime via an LLM-based semantic classifier.

---

## Architecture

```
run.sh                         ← Evaluation entry point (single command)
agent.py                       ← Main orchestrator (Phase 1 + Phase 2)
├── src/
│   ├── hardware_prober.py     ← Hardware probing pipeline + LLM semantic resolver
│   ├── kernel_analyzer.py     ← 4-step kernel bottleneck analysis (§1.1–§1.6)
│   ├── ncu_profiler.py        ← Nsight Compute wrapper + CSV parser + bottleneck ID
│   ├── reasoning.py           ← Structured logging + LLM reasoning narratives
│   ├── llm_client.py          ← DashScope API client (GLM-5, retry, streaming)
│   ├── probe_manager.py       ← CUDA compilation + execution manager
│   └── utils.py               ← nvidia-smi queries, statistics
├── probes/                    ← 6 CUDA micro-benchmark source files
│   ├── latency_probe.cu       ← Pointer-chasing (L1 / L2 / DRAM latency + L2 size)
│   ├── bandwidth_probe.cu     ← float4 vectorised global + shared memory bandwidth
│   ├── clock_probe.cu         ← clock64()/CUDA-events + PTX %smid SM count
│   ├── bank_conflict_probe.cu ← Stride-comparison bank conflict penalty
│   ├── shmem_limit_probe.cu   ← Binary-search max shared memory per block
│   └── ncu_verify_probe.cu    ← Lightweight probe for ncu cross-verification
├── kernels/                   ← 3 sample CUDA kernels for testing
│   ├── matmul_naive.cu        ← Naive matmul (no tiling)
│   ├── matmul_tiled.cu        ← Tiled matmul with shared memory + unrolling
│   └── matmul_tensor.cu       ← WMMA Tensor Core matmul (FP16→FP32, 19.4 TFLOPS)
├── target_spec.json           ← Default evaluation specification
└── build/                     ← Compiled binaries (auto-generated)
```

---

## Output Files

| File | Description |
|------|-------------|
| `results.json` | Numeric metrics + `_reasoning` + `_methodology` + `_log` evidence |
| `results_kernel_analysis.json` | Full ncu metric dump + structured bottleneck data |
| `results_kernel_report.md` | LLM-authored kernel bottleneck narrative (~500 words) |
| `reasoning.log` | Complete step-by-step evidence trail (JSON) |
| `agent.log` | Full execution log (timestamped) |

---

## Phase 1: Hardware Intrinsic Probing (Section 1.7)

### Supported Metrics

| Metric | Probe | Measurement Method |
|--------|-------|-------------------|
| `l1_latency_cycles` | latency_probe | Pointer chasing, array ≤ 16 KB (L1-resident) |
| `l2_latency_cycles` | latency_probe | Pointer chasing, L1 < array < L2 boundary |
| `dram_latency_cycles` | latency_probe | Pointer chasing, array ≫ L2 capacity |
| `l2_cache_size_kb` | latency_probe | Latency curve inflection (cliff) detection |
| `max_global_bandwidth_gbps` | bandwidth_probe | float4 coalesced reads, all SMs, max of sweep |
| `max_shmem_bandwidth_gbps` | bandwidth_probe | Bank-conflict-free shmem reads, per-SM + aggregate |
| `actual_boost_clock_mhz` | clock_probe | clock64() / CUDA-events under sustained FMA load |
| `num_active_sms` | clock_probe | Inline PTX `%smid` register, unique-ID counting |
| `max_shmem_per_block_kb` | shmem_limit_probe | Binary search + `cudaFuncSetAttribute` opt-in |
| `bank_conflict_penalty_cycles` | bank_conflict_probe | Stride-1 (0 conflicts) vs stride-32 (32-way) |

### LLM-Based Semantic Target Resolution

Unknown metric names (e.g., `"mem_delay"` instead of `"dram_latency_cycles"`) are batch-resolved in a single LLM call to the correct probe. Results are cached per session.

### Multi-Strategy Cross-Verification

Every measurement is verified using at least two independent methods:

| Strategy | Role |
|----------|------|
| **CUDA micro-benchmark** | Primary measurement (hardware-level, immune to API spoofing) |
| **ncu profiling** | Secondary verification (DRAM throughput → implied bandwidth; cycle count → clock) |
| **nvidia-smi / CUDA API** | Tertiary sanity check (device properties, clocks, SM count) |

### Anti-Hacking Resilience

| Threat | Defence |
|--------|---------|
| **Frequency Locking** | Clock measured via `clock64()`/CUDA-events ratio — no API dependency |
| **SM Masking** | Active SMs detected via inline PTX `%smid` — bypasses virtualisation |
| **Spoofed Device Properties** | All primary data from micro-benchmarks; API values used only for comparison |
| **Prefetcher Interference** | Pointer-chasing with random stride defeats hardware prefetching |

---

## Phase 2: Kernel Bottleneck Analysis (Sections 1.1–1.6)

When `target_spec.json` contains a `"run"` field (or `--kernel` is passed), the agent performs the full 4-step analysis:

| Step | Section | What It Does |
|------|---------|-------------|
| **1. Roofline** | §1.1 | Reads `sm__throughput` + `gpu__compute_memory_throughput` → classifies COMPUTE_BOUND / MEMORY_BOUND / BALANCED |
| **2. Characterise** | §1.2 / §1.3 | Memory-bound → dives into DRAM, L2, L1 metrics. Compute-bound → checks Tensor Core vs FMA utilisation |
| **3. Anomaly Scan** | §1.4 / §1.5 | Detects: LOW_OCCUPANCY, BANK_CONFLICT, WARP_DIVERGENCE, EXCESSIVE_GLOBAL_LOADS, TENSOR_CORES_UNUSED |
| **4. Map to Code** | §1.6 | Locates `.cu` source, scans for missing `__shared__`, absent `#pragma unroll`, idle Tensor Cores, etc. |

After the 4-step analysis, the LLM synthesises all findings (metrics + source code + code issues) into a professional report with line-level recommendations.

### Sample Kernels Included

| Kernel | Description | Key Characteristics |
|--------|------------|-------------------|
| `matmul_naive.cu` | Naive N×N matmul | No shared memory, no unrolling, no Tensor Cores |
| `matmul_tiled.cu` | Tiled matmul | `__shared__` memory, TILE_SIZE=32, `#pragma unroll` |
| `matmul_tensor.cu` | WMMA Tensor Core | FP16→FP32, 8-warp (4Y×2X), shmem staging, 19.4 TFLOPS @ N=4096 |

---

## LLM Integration

The agent uses the **GLM-5** model via Alibaba Cloud DashScope API for:

1. **Semantic target resolution** — mapping non-standard metric names to probes (`enable_thinking=False`, ~100ms)
2. **Anomaly analysis** — deep reasoning about detected anomalies (frequency locking, SM masking)
3. **Engineering reasoning** — LLM-authored `_reasoning` and `_methodology` narratives for the grading rubric
4. **Kernel bottleneck reports** — professional analysis reports with source-code-level recommendations

All LLM calls have **graceful fallback** to template-based text if the API is unavailable.

---

## Dependencies

- **CUDA Toolkit** ≥ 11.0 (nvcc + CUDA runtime)
- **NVIDIA Nsight Compute** (ncu) — for kernel analysis and cross-verification
- **Python** ≥ 3.10
- **Python packages**: `openai`, `python-dotenv`, `tenacity`

```bash
pip install openai python-dotenv tenacity
```

---

## Environment Setup

Create a `.env` file in the project root:

```
DASHSCOPE_API_KEY=sk-your-key-here
```

---

## CLI Reference

```bash
python3 agent.py [OPTIONS]

Options:
  --target-spec PATH    Path to target_spec.json (default: target_spec.json)
  --output PATH         Output results file (default: results.json)
  --kernel PATH         CUDA binary to analyse (overrides "run" in spec)
  --kernel-name NAME    Specific kernel function name to profile
  --probe-dir DIR       CUDA probe source directory (default: probes)
  --build-dir DIR       Compiled binary directory (default: build)
  --trials N            Repeated trials per measurement (default: 5)
```
