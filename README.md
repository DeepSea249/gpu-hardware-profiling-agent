# GPU Hardware Intrinsic Profiling Agent

This repository now contains both course phases:

- Phase 1: GPU hardware intrinsic probing and kernel bottleneck analysis
- Phase 2: An agentic LoRA kernel optimizer that maintains `optimized_lora.cu`

An autonomous, LLM-powered agent for probing GPU hardware characteristics via CUDA micro-benchmarks, cross-verifying with NVIDIA Nsight Compute (ncu), and performing kernel-level bottleneck analysis — all driven by a single `target_spec.json`.

All CUDA micro-benchmark probe source code is **generated autonomously by the LLM** at runtime; no pre-written benchmark files are required or used.

---

## Quick Start — Evaluation Entry Point

`run.sh` now defaults to the Phase 2 submission workflow required by `project_phase2_requirement.md`. Use `--phase phase1` only when you want the original hardware-probing entrypoint.

```bash
# Phase 2 (default): generate and iteratively optimize optimized_lora.cu
chmod +x run.sh && ./run.sh
```

```bash
# Phase 1 (legacy): hardware probing + optional kernel analysis
./run.sh --phase phase1 --target-spec /path/to/target_spec.json
```

Override Phase 2 defaults with CLI flags:

```bash
./run.sh --search-rounds 6 --time-budget-minutes 25 --benchmark-sizes 3584,4096,4608
```

The Phase 2 agent writes:

- `optimized_lora.cu` — the current best single-file implementation
- `phase2_results.json` — candidate-by-candidate benchmark history
- `phase2_summary.md` — compact optimization summary

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
│   ├── probe_codegen.py       ← LLM-driven autonomous CUDA probe source generation
│   ├── probe_manager.py       ← CUDA compilation + execution manager
│   ├── reasoning.py           ← Structured logging + LLM reasoning narratives
│   ├── llm_client.py          ← OpenAI-compatible API client (retry, streaming)
│   └── utils.py               ← nvidia-smi queries, statistics
├── kernels/                   ← 3 sample CUDA kernels for kernel analysis
│   ├── matmul_naive.cu        ← Naive matmul (no tiling)
│   ├── matmul_tiled.cu        ← Tiled matmul with shared memory + unrolling
│   └── matmul_tensor.cu       ← WMMA Tensor Core matmul (FP16→FP32)
├── build/                     ← Auto-generated directory
│   └── *_generated.cu         ← LLM-authored CUDA probe sources (cached)
└── target_spec.json           ← Default evaluation specification
```

---

## Output Files

| File | Description |
|------|-------------|
| `results.json` | Numeric metrics + `_reasoning` + `_methodology` + `_log` evidence |
| `results_kernel_analysis.json` | Full ncu metric dump + structured bottleneck data |
| `results_kernel_report.md` | LLM-authored kernel bottleneck narrative |
| `reasoning.log` | Complete step-by-step evidence trail (JSON) |
| `agent.log` | Full execution log (timestamped) |

All output files are regenerated on every agent run and are excluded from version control.

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

> **Autonomous probe generation**: all CUDA source code for the probes above is written by the LLM at runtime based on design specifications. Sources are cached in `build/*_generated.cu` and reused on subsequent runs to avoid redundant API calls.

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

The agent uses any **OpenAI-compatible** LLM endpoint for:

1. **Autonomous probe generation** — the LLM writes all 6 CUDA micro-benchmark `.cu` source files from design specifications at runtime; sources are cached in `build/` and reused on subsequent runs
2. **Semantic target resolution** — mapping non-standard metric names to probes (~100ms)
3. **Anomaly analysis** — deep reasoning about detected anomalies (frequency locking, SM masking)
4. **Engineering reasoning** — LLM-authored `_reasoning` and `_methodology` narratives
5. **Kernel bottleneck reports** — professional analysis reports with source-code-level recommendations

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
# Required: API key for the LLM endpoint
API_KEY=your-api-key-here

# Optional: override the default OpenAI endpoint
# LLM_BASE_URL=https://api.openai.com/v1

# Optional: override the model name (default: gpt-4o)
# LLM_MODEL=gpt-4o
```

| Variable | Required | Default | Notes |
|----------|----------|---------|-------|
| `API_KEY` | Yes | — | API key for the LLM |
| `LLM_BASE_URL` | No | `https://api.openai.com/v1` | Any OpenAI-compatible base URL |
| `LLM_MODEL` | No | `gpt-4o` | Model name passed to the API |

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
