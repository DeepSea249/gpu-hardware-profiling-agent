# GPU Hardware Intrinsic Profiling Agent

An autonomous agent system for probing GPU hardware characteristics using CUDA micro-benchmarks and NVIDIA Nsight Compute (ncu) profiling, plus kernel-level bottleneck analysis (Sections 1.1-1.6).

## Evaluation Entry Point

```bash
# Place your target_spec.json in the project root, then:
chmod +x run.sh && ./run.sh

# Or with a custom spec path:
./run.sh --target-spec /path/to/target_spec.json --output results.json
```

The agent reads **everything** from `target_spec.json` — no hardcoded workflow:

```jsonc
{
  "targets": ["dram_latency_cycles", "actual_boost_clock_mhz", ...],
  "run": "path/to/executable"   // optional: kernel to profile via ncu
}
```

- **`targets`**: list of hardware metrics to identify via micro-benchmarks (Section 1.7)
- **`run`**: path to a CUDA executable to profile for bottleneck analysis (Sections 1.1-1.6)

Both fields are optional — the agent runs whichever phases are requested.

## Architecture

```
run.sh                       # ← Evaluation entry point (single command)
agent.py                     # Main orchestrator
├── src/
│   ├── hardware_prober.py   # Coordinates all hardware probing
│   ├── probe_manager.py     # Compiles & runs CUDA micro-benchmarks
│   ├── ncu_profiler.py      # Nsight Compute integration & CSV parser
│   ├── kernel_analyzer.py   # Kernel bottleneck analysis (Sections 1.1-1.6)
│   ├── reasoning.py         # Structured reasoning/logging + LLM integration
│   ├── llm_client.py        # DashScope LLM client with retry & streaming
│   └── utils.py             # nvidia-smi queries, statistics
├── probes/                  # CUDA micro-benchmark source files
│   ├── latency_probe.cu     # Pointer-chasing memory latency hierarchy
│   ├── bandwidth_probe.cu   # Global & shared memory bandwidth
│   ├── clock_probe.cu       # Clock frequency & SM count detection
│   ├── bank_conflict_probe.cu # Shared memory bank conflict penalty
│   ├── shmem_limit_probe.cu # Max shared memory per block
│   └── ncu_verify_probe.cu  # Cross-verification via ncu
├── kernels/                 # Sample CUDA kernels for testing
│   ├── matmul_naive.cu      # Naive matmul (no tiling, no Tensor Cores)
│   ├── matmul_tiled.cu      # Tiled matmul with shared memory
│   └── matmul_tensor.cu     # WMMA Tensor Core matmul (FP16→FP32)
└── build/                   # Compiled binaries (auto-generated)
```

## Output Files

| File | Description |
|------|-------------|
| `results.json` | Numeric metrics + `_reasoning` + `_methodology` + `_log` evidence |
| `results_kernel_analysis.json` | Full ncu metric dump + bottleneck details |
| `results_kernel_report.md` | LLM-authored bottleneck narrative |
| `reasoning.log` | Step-by-step evidence trail for grading |
| `agent.log` | Full execution log |

## Supported Metrics

| Metric | Probe | Method |
|--------|-------|--------|
| `l1_latency_cycles` | latency_probe | Pointer chasing, array < L1 size |
| `l2_latency_cycles` | latency_probe | Pointer chasing, L1 < array < L2 |
| `dram_latency_cycles` | latency_probe | Pointer chasing, array >> L2 |
| `l2_cache_size_kb` | latency_probe | Latency curve inflection detection |
| `max_global_bandwidth_gbps` | bandwidth_probe | Vectorized float4 coalesced reads |
| `max_shmem_bandwidth_gbps` | bandwidth_probe | Bank-conflict-free shared memory reads |
| `actual_boost_clock_mhz` | clock_probe | clock64() / CUDA-events ratio |
| `num_active_sms` | clock_probe | Inline PTX %smid register |
| `max_shmem_per_block_kb` | shmem_limit_probe | Binary search with kernel launches |
| `bank_conflict_penalty_cycles` | bank_conflict_probe | Stride comparison (1 vs 32) |

## Anti-Hacking Resilience

This agent is designed to produce accurate measurements even when the evaluation environment is modified:

1. **Frequency Locking**: Clock frequency measured directly via `clock64()`/CUDA-events ratio. Does not rely on API-reported values.
2. **SM Masking**: Active SM count detected via inline PTX `%smid` register. Cross-verified across multiple probes.
3. **Spoofed Device Properties**: All measurements use actual micro-benchmarks. `cudaGetDeviceProperties` values are only used for cross-verification, not as primary data.

## Kernel Analysis (Sections 1.1-1.6)

When `target_spec.json` contains a `"run"` field (or `--kernel` is passed), the agent performs a 4-step ncu bottleneck analysis:

1. **Roofline** (§1.1): Classifies kernel as compute-bound or memory-bound via SOL metrics
2. **Deep-dive** (§1.2/§1.3): Memory hierarchy or compute unit analysis depending on classification
3. **Anomaly scan** (§1.4/§1.5): Checks occupancy gaps, bank conflicts, warp divergence, uncoalesced access
4. **LLM synthesis**: Generates a professional narrative report with actionable optimisation recommendations

The agent does **not** hardcode which executable to profile — it reads the path from the spec file at runtime.
