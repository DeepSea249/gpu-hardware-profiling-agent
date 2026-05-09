# GPU Hardware Profiling and LoRA Optimization Agent

This repository contains an MLSys course project with two related workflows:

- **Phase 2, default:** an agentic optimizer for a LoRA forward operator that continuously maintains `optimized_lora.cu`.
- **Phase 1, legacy/manual:** a GPU hardware intrinsic profiler plus an NVIDIA Nsight Compute (`ncu`) kernel bottleneck analyzer.

The important entry point is:

```bash
bash run.sh
```

By default this runs **Phase 2**. Use `--phase phase1` only when you want the original hardware profiling and kernel analysis workflow.

---

## Quick Start

### Phase 2: LoRA optimization, default submission path

```bash
bash run.sh
```

This creates or updates:

| File | Meaning |
|---|---|
| `optimized_lora.cu` | Current best single-file PyTorch CUDA extension for the LoRA operator |
| `phase2_results.json` | Full Phase 2 result object written by `agent.py` |
| `phase2_search_history.json` | Candidate history written by `LoRAOptimizationAgent` |
| `phase2_summary.md` | Human-readable optimization summary |

Useful overrides:

```bash
bash run.sh --time-budget-minutes 30 --benchmark-sizes 3584,4096,4608
```

Equivalent environment-variable overrides:

```bash
LORA_TIME_BUDGET_MINUTES=30 \
LORA_BENCHMARK_SIZES=3584,4096,4608 \
bash run.sh
```

### Phase 1: hardware profiling and ncu kernel analysis

```bash
bash run.sh --phase phase1 --target-spec target_spec.json
```

`run.sh` writes Phase 1 output to `output.json` by default. To override that path:

```bash
PHASE1_OUTPUT=results.json bash run.sh --phase phase1 --target-spec target_spec.json
```

---

## What Phase 2 Optimizes

The target operator is:

```text
Y = W @ X + A @ (B^T @ X)
```

with:

- `W`, `X`: `[d, d]` float32 CUDA tensors
- `A`, `B`: `[d, 16]` float32 CUDA tensors
- public size range: `d in [3584, 4608]`
- correctness gate: `torch.allclose(..., rtol=1e-4, atol=1e-4)`

The final measured artifact must be a single file:

```text
optimized_lora.cu
```

It exports:

```cpp
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B);
```

and binds it with `PYBIND11_MODULE`, so the official harness can call `module.forward(W, X, A, B)`.

---

## Phase 2 Agent Design

`agent.py --phase phase2` delegates to `LoRAOptimizationAgent` in `src/lora_optimizer.py`.

The optimizer always writes an initial valid baseline to `optimized_lora.cu` before benchmarking. This keeps the required output file present even if the search is interrupted.

### Stage 1: fixed template search

The current implementation evaluates an eight-candidate library:

| Candidate | Strategy |
|---|---|
| `baseline_sequential` | Single-stream `at::mm` plus `at::addmm_out` |
| `dual_stream_overlap` | Overlap `W@X` and `B^T@X` on two CUDA streams |
| `dual_stream_pretranspose` | Dual-stream path with explicit contiguous `B^T` |
| `dual_stream_mm_add` | Dual-stream path with separate low-rank `mm` plus in-place add |
| `triple_stream_prealloc` | Pre-allocated outputs with separate streams for both first-stage GEMMs |
| `sequential_prealloc` | Single-stream `mm_out` and `addmm_out` with pre-allocated tensors |
| `dual_stream_prealloc` | Dual-stream overlap plus pre-allocated outputs |
| `dual_stream_prealloc_contig` | Dual-stream pre-allocation plus contiguous `B^T` inside the auxiliary stream |

The templates intentionally use PyTorch C++ operators such as `at::mm`, `at::mm_out`, and `at::addmm_out`. This preserves the same cuBLAS-backed accumulation behavior as the Python reference and avoids correctness failures from changed FP32 accumulation order.

Each candidate is:

1. rendered as a complete `.cu` extension,
2. compiled with `torch.utils.cpp_extension.load`,
3. checked against the PyTorch reference,
4. benchmarked with CUDA events after warmup,
5. scored by geometric mean speedup across the configured benchmark sizes.

### Stage 2: optional LLM iterative improvement

If a usable LLM API key is available, Stage 2 asks the model for complete replacement `.cu` files, then compiles, validates, and benchmarks them. A candidate replaces `optimized_lora.cu` only if it passes correctness and beats the current best score.

If no API key is available, Stage 2 is skipped gracefully and the best Stage 1 template remains in `optimized_lora.cu`.

---

## Phase 1 Capabilities

Phase 1 is the original GPU hardware profiling agent. It is driven by a `target_spec.json` file:

```json
{
  "targets": [
    "l1_latency_cycles",
    "l2_latency_cycles",
    "dram_latency_cycles",
    "l2_cache_size_kb",
    "max_global_bandwidth_gbps",
    "max_shmem_bandwidth_gbps",
    "actual_boost_clock_mhz",
    "max_shmem_per_block_kb",
    "bank_conflict_penalty_cycles",
    "num_active_sms"
  ],
  "run": "build/kernels/matmul_naive"
}
```

Both fields are optional:

- `targets`: hardware metrics to measure with generated CUDA micro-benchmarks
- `run`: CUDA binary or `.cu` source to profile with `ncu`

### Hardware probes

`HardwareProber` maps requested target names to probes. Known canonical metrics use a direct lookup; unknown names are batch-resolved by the LLM semantic resolver.

| Probe | Metrics measured |
|---|---|
| `latency_probe` | L1 latency, L2 latency, DRAM latency, L2 size |
| `bandwidth_probe` | Global read/write/copy bandwidth, shared-memory bandwidth |
| `clock_probe` | Actual SM clock under load, active SM count |
| `shmem_limit_probe` | Default and opt-in dynamic shared memory per block |
| `bank_conflict_probe` | Shared-memory bank conflict penalty |
| `ncu_verify_probe` | Lightweight probe used for `ncu` cross-verification |

Probe source code is generated at runtime by `ProbeCodeGenerator` from detailed design specifications in `src/probe_codegen.py`, then cached as `build/*_generated.cu`. On compilation failure, the generator can ask the LLM to repair the CUDA source and retry.

### Kernel analysis

When `target_spec.json` contains `run`, or `--kernel` is passed directly, `KernelAnalyzer` performs:

1. roofline classification from SM and memory SOL metrics,
2. memory or compute subsystem deep-dive,
3. bottleneck scan for occupancy, bank conflicts, divergence, excess loads, and unused Tensor Cores,
4. source-code pattern mapping for missing shared memory, missing unroll pragmas, absent WMMA/MMA usage, and related issues,
5. LLM or template report generation.

Phase 1 writes one consolidated JSON output. Besides requested numeric metrics, it can include:

- `_kernel_analysis_full`
- `_kernel_analysis`
- `_kernel_report`
- `_reasoning`
- `_methodology`
- `_log`

---

## Project Layout

```text
.
|-- agent.py                         Main CLI and phase dispatcher
|-- run.sh                           Evaluation entry point
|-- target_spec.json                 Default Phase 1 target specification
|-- target_spec_ncu_test.json        Example non-canonical target names
|-- test_llm.py                      LLM client smoke test
|-- src/
|   |-- lora_optimizer.py            Phase 2 template search and LLM improvement loop
|   |-- hardware_prober.py           Phase 1 probe orchestration and metric extraction
|   |-- probe_codegen.py             LLM-generated CUDA micro-benchmark specs
|   |-- probe_manager.py             nvcc discovery, compilation, and execution
|   |-- ncu_profiler.py              Nsight Compute wrapper and CSV parser
|   |-- kernel_analyzer.py           ncu bottleneck analysis and source mapping
|   |-- reasoning.py                 Structured logs, anomalies, and LLM narratives
|   |-- llm_client.py                OpenAI-compatible streaming API client
|   `-- utils.py                     nvidia-smi and statistics helpers
|-- kernels/
|   |-- matmul_naive.cu              Naive FP32 matmul sample
|   |-- matmul_tiled.cu              Shared-memory tiled FP32 matmul sample
|   `-- matmul_tensor.cu             WMMA Tensor Core FP16->FP32 matmul sample
`-- docs/                            Design notes and reports from development
```

Generated files such as `build/`, `optimized_lora.cu`, `phase2_results.json`, `phase2_summary.md`, and `phase2_search_history.json` are ignored by git.

---

## Dependencies

Core requirements:

- Python 3.10+
- CUDA Toolkit with `nvcc`
- NVIDIA GPU and driver
- PyTorch with CUDA support for Phase 2
- NVIDIA Nsight Compute (`ncu`) for Phase 1 kernel analysis and cross-verification

Python packages used directly by this repository:

```bash
pip install openai python-dotenv tenacity
```

`run.sh` attempts to install those three packages automatically and continues if installation is already satisfied or unavailable. It does not install PyTorch; Phase 2 expects a CUDA-enabled PyTorch environment.

---

## LLM Configuration

The LLM client is OpenAI-compatible and reads `.env` from the repository root.

```bash
# Required for LLM-backed generation or improvement
API_KEY=your-api-key

# Optional
BASE_URL=https://api.openai.com/v1
BASE_MODEL=gpt-4o
```

Development fallback:

```bash
DASHSCOPE_API_KEY=your-dashscope-key
```

If `API_KEY` is absent but `DASHSCOPE_API_KEY` is present, the client auto-selects the DashScope compatible endpoint and `glm-5`.

LLM usage by workflow:

| Workflow | LLM role |
|---|---|
| Phase 2 Stage 1 | Not required; uses local templates |
| Phase 2 Stage 2 | Optional iterative code improvement |
| Phase 1 probe generation | Required unless matching generated sources already exist in `build/` |
| Phase 1 reasoning/reporting | Optional; template fallbacks are used when unavailable |
| Phase 1 semantic target resolution | Optional but useful for non-canonical target names |

---

## CLI Reference

Direct agent usage:

```bash
python3 agent.py [OPTIONS]
```

Important options:

| Option | Default | Description |
|---|---|---|
| `--phase {auto,phase1,phase2}` | `auto` | `auto` chooses Phase 1 when a spec/kernel is present, otherwise Phase 2 |
| `--target-spec PATH` | `/target/target_spec.json` | Phase 1 target spec |
| `--output PATH` | `/workspace/output.json` | Consolidated JSON output path |
| `--kernel PATH` | none | CUDA binary or `.cu` source for Phase 1 kernel analysis |
| `--kernel-name NAME` | none | Specific kernel name for `ncu` filtering |
| `--probe-dir DIR` | `probes` | Compatibility path; generated probes are cached under `build/` |
| `--build-dir DIR` | `build` | Build/cache directory |
| `--trials N` | `5` | Phase 1 repeated trials per measurement |
| `--optimized-path PATH` | `optimized_lora.cu` | Phase 2 best implementation path |
| `--summary-path PATH` | `phase2_summary.md` | Phase 2 markdown summary |
| `--search-rounds N` | `8` | Parsed for compatibility; the current code evaluates the fixed 8-template Stage 1 library subject to the time budget |
| `--time-budget-minutes N` | `20` | Phase 2 time budget |
| `--benchmark-sizes LIST` | `3584,3600,3712,3840,3968,4000,4096,4200,4352,4480,4608` | Comma-separated Phase 2 dimensions |
| `--benchmark-warmup N` | `5` | Warmup iterations per benchmark |
| `--benchmark-iters N` | `15` | Timed iterations per benchmark |
| `--safety-margin-seconds N` | `150` | Soft-deadline margin reserved inside the hard time budget |
| `--stage1-pass1-iters N` | `5` | Cheap Stage 1 pass timed iterations |
| `--stage1-pass2-iters N` | `15` | Deep Stage 1 pass timed iterations |
| `--stage1-topk N` | `3` | Candidates promoted to the deep Stage 1 pass |
| `--stage2-max-iters N` | `3` | Maximum guarded LLM improvement iterations |
| `--min-stage2-seconds N` | `360` | Minimum time left before starting a Stage 2 iteration |
| `--skip-llm` | false | Disable Phase 2 LLM improvement loop |

Examples:

```bash
# Direct Phase 2 run without LLM improvement
python3 agent.py --phase phase2 --skip-llm --output phase2_results.json

# Direct Phase 1 hardware probing
python3 agent.py --phase phase1 --target-spec target_spec.json --output output.json

# Analyze one CUDA source file with ncu
python3 agent.py --phase phase1 --kernel kernels/matmul_tiled.cu --output output.json
```

---

## Notes for Maintainers

- `run.sh` defaults to Phase 2 because that is the current course submission contract.
- `agent.py --phase auto` is different from `run.sh`: direct `auto` mode chooses Phase 1 if the target spec contains `targets`, `run`, or a `--kernel`; otherwise it chooses Phase 2.
- Phase 1 output is intentionally consolidated into a single JSON file for evaluator compatibility.
- Some historical design documents in `docs/` describe earlier development results and may not match the latest Phase 2 code path exactly; `agent.py`, `run.sh`, and `src/` are the source of truth.
