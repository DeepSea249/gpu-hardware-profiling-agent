# GPU Hardware Intrinsic Profiling Agent

An autonomous agent system for probing GPU hardware characteristics using CUDA micro-benchmarks and NVIDIA Nsight Compute (ncu) profiling.

## Architecture

```
agent.py                     # Main entry point & orchestrator
├── src/
│   ├── hardware_prober.py   # Coordinates all hardware probing
│   ├── probe_manager.py     # Compiles & runs CUDA micro-benchmarks
│   ├── ncu_profiler.py      # Nsight Compute integration
│   ├── kernel_analyzer.py   # Kernel bottleneck analysis (Section 1)
│   ├── reasoning.py         # Structured reasoning/logging engine
│   └── utils.py             # nvidia-smi queries, statistics
├── probes/
│   ├── latency_probe.cu     # Pointer-chasing memory latency hierarchy
│   ├── bandwidth_probe.cu   # Global & shared memory bandwidth
│   ├── clock_probe.cu       # Clock frequency & SM count detection
│   ├── bank_conflict_probe.cu # Shared memory bank conflict penalty
│   └── shmem_limit_probe.cu # Max shared memory per block
└── build/                   # Compiled probe binaries (auto-generated)
```

## Quick Start

```bash
# Run with default target_spec.json
chmod +x run.sh && ./run.sh

# Or directly with Python
python3 agent.py --target-spec target_spec.json --output results.json
```

## Input / Output

**Input** (`target_spec.json`):
```json
{
  "targets": ["dram_latency_cycles", "max_shmem_per_block_kb", "actual_boost_clock_mhz"]
}
```

**Output** (`results.json`):
```json
{
  "dram_latency_cycles": 442,
  "max_shmem_per_block_kb": 100,
  "actual_boost_clock_mhz": 1410,
  "_reasoning": { ... },
  "_methodology": { ... }
}
```

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

## Kernel Analysis Mode

For analyzing CUDA kernel bottlenecks (Section 1):

```bash
python3 agent.py --kernel ./my_cuda_binary
```

Performs the 4-step analysis:
1. Roofline classification (compute vs. memory bound)
2. Deep-dive into relevant metric category
3. Anomaly detection (occupancy gaps, bank conflicts)
4. Recommendations mapped to optimization strategies
