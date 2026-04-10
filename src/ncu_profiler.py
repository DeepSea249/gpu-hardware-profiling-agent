"""
NCU Profiler - Nsight Compute integration for kernel analysis.

Provides:
- Run ncu on CUDA kernel binaries with specified metrics
- Robust CSV parser for ncu --csv output (handles n/a, ==PROF== lines)
- Roofline classification (compute vs memory bound)
- Memory hierarchy, compute unit, occupancy, divergence analysis
- Bottleneck identification with severity and recommendations
"""

import subprocess
import shutil
import logging
import re
import csv
import io
from typing import Optional

logger = logging.getLogger('GPUAgent.NCUProfiler')


class NCUProfiler:
    """Wraps NVIDIA Nsight Compute (ncu) for kernel profiling."""

    # Core metrics for roofline analysis (Section 1.1)
    ROOFLINE_METRICS = [
        'sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
    ]

    # Memory hierarchy metrics (Section 1.2)
    MEMORY_METRICS = [
        'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
        'l2__throughput.avg.pct_of_peak_sustained_elapsed',
        'dram__throughput.avg.pct_of_peak_sustained_elapsed',
        'l1tex__data_bank_conflicts_pipe_lsu.sum',
    ]

    # Compute unit metrics (Section 1.3)
    COMPUTE_METRICS = [
        'sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active',
        'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active',
        'sm__sass_thread_inst_executed_op_fp32_pred_on.sum',
    ]

    # Occupancy metrics (Section 1.4)
    OCCUPANCY_METRICS = [
        'sm__maximum_warps_per_active_cycle_pct',
        'sm__warps_active.avg.pct_of_peak_sustained_active',
    ]

    # Warp divergence metrics (Section 1.5)
    DIVERGENCE_METRICS = [
        'smsp__sass_thread_inst_executed_per_inst_executed.ratio',
    ]
    # Warp divergence metrics
    DIVERGENCE_METRICS = [
        'smsp__sass_thread_inst_executed_per_inst_executed.ratio',
    ]

    ALL_ANALYSIS_METRICS = (
        ROOFLINE_METRICS + MEMORY_METRICS + COMPUTE_METRICS +
        OCCUPANCY_METRICS + DIVERGENCE_METRICS
    )

    def __init__(self):
        self._ncu_path = self._find_ncu()

    def _find_ncu(self) -> str:
        """Locate the ncu binary."""
        ncu = shutil.which('ncu')
        if ncu:
            return ncu

        common_paths = [
            '/usr/local/bin/ncu',
            '/usr/local/cuda/bin/ncu',
            '/usr/local/cuda-12/bin/ncu',
            '/usr/local/cuda-12.4/bin/ncu',
            '/opt/nvidia/nsight-compute/ncu',
        ]
        for path in common_paths:
            if path and shutil.which(path):
                return path

        logger.warning("ncu not found - Nsight Compute profiling will be unavailable")
        return None

    @property
    def available(self) -> bool:
        """Check if ncu is available."""
        return self._ncu_path is not None

    def profile(self, binary: str, metrics: list = None,
                kernel_name: str = None, timeout: int = 300,
                launch_skip: int = 0, launch_count: int = 0) -> dict:
        """
        Run ncu on a binary and return parsed metrics.

        Args:
            binary: Path to CUDA binary
            metrics: List of metric names (default: all analysis metrics)
            kernel_name: Specific kernel to profile (optional)
            timeout: Timeout in seconds
            launch_skip: Number of kernel launches to skip (e.g. warmup)
            launch_count: Number of kernel launches to profile (0 = all)

        Returns:
            Dictionary with 'metrics' (name->value), 'kernel_info' (metadata)
        """
        if not self.available:
            logger.warning("ncu unavailable, skipping profiling")
            return {}

        if metrics is None:
            metrics = self.ALL_ANALYSIS_METRICS

        cmd = [
            self._ncu_path,
            '--csv',
            '--metrics', ','.join(metrics),
        ]

        if kernel_name:
            cmd.extend(['--kernel-name', kernel_name])
        if launch_skip > 0:
            cmd.extend(['--launch-skip', str(launch_skip)])
        if launch_count > 0:
            cmd.extend(['--launch-count', str(launch_count)])

        cmd.append(binary)

        logger.info(f"Running ncu: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        except subprocess.TimeoutExpired:
            logger.error(f"ncu timed out after {timeout}s")
            return {}

        if result.returncode != 0:
            logger.error(f"ncu failed (rc={result.returncode}): {result.stderr[:500]}")
            return {}

        return self._parse_csv_output(result.stdout)

    def _parse_csv_output(self, csv_text: str) -> dict:
        """
        Parse ncu --csv output into a structured result.

        ncu CSV format (one row per metric, per kernel launch):
          "ID","Process ID","Process Name","Host Name","Kernel Name",
          "Context","Stream","Block Size","Grid Size","Device","CC",
          "Section Name","Metric Name","Metric Unit","Metric Value"

        Returns dict with:
          'metrics': {metric_name: numeric_value_or_string, ...}
          'kernel_info': {kernel_name, block_size, grid_size, cc}
          'raw_rows': [list of parsed CSV dicts]
        """
        result: dict = {'metrics': {}, 'kernel_info': {}, 'raw_rows': []}

        # Filter out ==PROF== lines and program stdout
        csv_lines = []
        for line in csv_text.split('\n'):
            stripped = line.strip()
            if stripped.startswith('"'):
                csv_lines.append(stripped)

        if len(csv_lines) < 2:
            logger.warning("ncu CSV output has no data rows")
            return result

        try:
            reader = csv.DictReader(io.StringIO('\n'.join(csv_lines)))
            for row in reader:
                metric_name = row.get('Metric Name', '').strip()
                value_str = row.get('Metric Value', '').strip()
                unit = row.get('Metric Unit', '').strip()

                # Extract kernel info from first row
                if not result['kernel_info']:
                    result['kernel_info'] = {
                        'kernel_name': row.get('Kernel Name', '').strip(),
                        'block_size': row.get('Block Size', '').strip(),
                        'grid_size': row.get('Grid Size', '').strip(),
                        'cc': row.get('CC', '').strip(),
                    }

                result['raw_rows'].append({
                    'name': metric_name,
                    'value_raw': value_str,
                    'unit': unit,
                })

                if not metric_name or not value_str or value_str.lower() == 'n/a':
                    continue

                # Parse numeric value
                try:
                    clean = value_str.replace(',', '').replace('%', '')
                    result['metrics'][metric_name] = float(clean)
                except ValueError:
                    result['metrics'][metric_name] = value_str
        except Exception as e:
            logger.warning(f"CSV DictReader failed, trying field parser: {e}")
            result = self._parse_csv_field_fallback(csv_lines)

        logger.info(f"ncu parsed {len(result['metrics'])} metrics for kernel "
                     f"'{result['kernel_info'].get('kernel_name', '?')}'")
        return result

    def _parse_csv_field_fallback(self, csv_lines: list) -> dict:
        """Fallback field-level CSV parser when DictReader fails."""
        result: dict = {'metrics': {}, 'kernel_info': {}, 'raw_rows': []}
        for line in csv_lines[1:]:  # skip header
            parts = line.split('","')
            if len(parts) < 15:
                continue
            parts = [p.strip('"') for p in parts]
            metric_name = parts[12]
            value_str = parts[14]

            if not result['kernel_info']:
                result['kernel_info'] = {
                    'kernel_name': parts[4],
                    'block_size': parts[7],
                    'grid_size': parts[8],
                    'cc': parts[10],
                }

            result['raw_rows'].append({
                'name': metric_name, 'value_raw': value_str, 'unit': parts[13],
            })

            if not value_str or value_str.lower() == 'n/a':
                continue
            try:
                clean = value_str.replace(',', '').replace('%', '')
                result['metrics'][metric_name] = float(clean)
            except ValueError:
                result['metrics'][metric_name] = value_str
        return result

    def profile_with_details(self, binary: str, kernel_name: str = None,
                             timeout: int = 300) -> dict:
        """
        Run comprehensive ncu profiling with all analysis metrics.

        Profiles the *second* kernel launch (skips warmup) for more
        representative numbers.  Returns structured analysis result.
        """
        raw = self.profile(
            binary,
            self.ALL_ANALYSIS_METRICS,
            kernel_name=kernel_name,
            timeout=timeout,
            launch_skip=1,   # skip warmup launch
            launch_count=1,  # profile exactly one launch
        )

        metrics = raw.get('metrics', {})
        kernel_info = raw.get('kernel_info', {})

        if not metrics:
            return {'error': 'No metrics collected', 'metrics': {},
                    'kernel_info': kernel_info}

        analysis = {
            'metrics': metrics,
            'kernel_info': kernel_info,
            'raw_rows': raw.get('raw_rows', []),
            'roofline': self._analyze_roofline(metrics),
            'memory': self._analyze_memory(metrics),
            'compute': self._analyze_compute(metrics),
            'occupancy': self._analyze_occupancy(metrics),
            'bottlenecks': self._identify_bottlenecks(metrics),
        }

        return analysis

    @staticmethod
    def _m(metrics: dict, key: str, default: float = 0.0) -> float:
        """Safely extract a numeric metric value."""
        v = metrics.get(key, default)
        if isinstance(v, (int, float)):
            return float(v)
        return default

    def _analyze_roofline(self, metrics: dict) -> dict:
        """Determine compute vs memory bound classification (Section 1.1)."""
        compute_util = self._m(metrics,
            'sm__throughput.avg.pct_of_peak_sustained_elapsed')
        memory_util = self._m(metrics,
            'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed')

        if compute_util == 0 and memory_util == 0:
            classification = 'unknown'
        elif memory_util > compute_util * 1.2:
            classification = 'memory_bound'
        elif compute_util > memory_util * 1.2:
            classification = 'compute_bound'
        else:
            classification = 'balanced'

        return {
            'compute_utilization_pct': compute_util,
            'memory_utilization_pct': memory_util,
            'classification': classification,
        }

    def _analyze_memory(self, metrics: dict) -> dict:
        """Analyze memory hierarchy utilization (Section 1.2)."""
        l1_sectors = self._m(metrics,
            'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum')
        return {
            'l1_global_load_sectors': l1_sectors,
            'l2_throughput_pct': self._m(metrics,
                'l2__throughput.avg.pct_of_peak_sustained_elapsed'),
            'dram_throughput_pct': self._m(metrics,
                'dram__throughput.avg.pct_of_peak_sustained_elapsed'),
            'bank_conflicts': self._m(metrics,
                'l1tex__data_bank_conflicts_pipe_lsu.sum'),
        }

    def _analyze_compute(self, metrics: dict) -> dict:
        """Analyze compute unit utilization (Section 1.3)."""
        return {
            'tensor_core_utilization_pct': self._m(metrics,
                'sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active'),
            'fma_utilization_pct': self._m(metrics,
                'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active'),
            'fp32_instructions': self._m(metrics,
                'sm__sass_thread_inst_executed_op_fp32_pred_on.sum'),
        }

    def _analyze_occupancy(self, metrics: dict) -> dict:
        """Analyze warp occupancy (Section 1.4)."""
        theoretical = self._m(metrics, 'sm__maximum_warps_per_active_cycle_pct')
        achieved = self._m(metrics,
            'sm__warps_active.avg.pct_of_peak_sustained_active')
        gap = theoretical - achieved if theoretical and achieved else 0

        return {
            'theoretical_occupancy_pct': theoretical,
            'achieved_occupancy_pct': achieved,
            'occupancy_gap_pct': gap,
        }

    def _identify_bottlenecks(self, metrics: dict) -> list:
        """Identify specific bottlenecks (Section 1.5)."""
        bottlenecks = []

        # VRAM Bound: dram throughput > 70%
        dram_tp = self._m(metrics,
            'dram__throughput.avg.pct_of_peak_sustained_elapsed')
        if dram_tp > 70:
            bottlenecks.append({
                'type': 'VRAM_BOUND',
                'severity': 'high' if dram_tp > 85 else 'medium',
                'metric': 'dram__throughput.avg.pct_of_peak_sustained_elapsed',
                'value': dram_tp,
                'recommendation': (
                    'Reduce memory access volume; increase data reuse with '
                    'Shared Memory tiling; consider reduced precision (FP16/BF16).'
                ),
            })

        # Compute Bound: SM throughput > 70%
        sm_tp = self._m(metrics,
            'sm__throughput.avg.pct_of_peak_sustained_elapsed')
        if sm_tp > 70:
            bottlenecks.append({
                'type': 'COMPUTE_BOUND',
                'severity': 'high' if sm_tp > 85 else 'medium',
                'metric': 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
                'value': sm_tp,
                'recommendation': (
                    'Consider algorithmic improvements, precision reduction '
                    '(FP16/BF16), or Tensor Core utilisation via WMMA/MMA.'
                ),
            })

            # Sub-check: Tensor Cores not being used for matrix ops?
            tensor_util = self._m(metrics,
                'sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active')
            fma_util = self._m(metrics,
                'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active')
            if tensor_util < 5 and fma_util > 10:
                bottlenecks.append({
                    'type': 'TENSOR_CORES_UNUSED',
                    'severity': 'medium',
                    'metric': 'sm__pipe_tensor_op_hmma_cycles_active',
                    'value': tensor_util,
                    'recommendation': (
                        'Tensor Cores are idle while FMA units are active. For '
                        'matrix operations, use WMMA/MMA intrinsics or cuBLAS '
                        'with FP16 inputs aligned to 16/32 boundaries.'
                    ),
                })

        # Bank Conflicts
        bank_conflicts = self._m(metrics,
            'l1tex__data_bank_conflicts_pipe_lsu.sum')
        if bank_conflicts > 0:
            severity = 'high' if bank_conflicts > 1000000 else 'medium' if bank_conflicts > 10000 else 'low'
            bottlenecks.append({
                'type': 'BANK_CONFLICT',
                'severity': severity,
                'metric': 'l1tex__data_bank_conflicts_pipe_lsu.sum',
                'value': bank_conflicts,
                'recommendation': (
                    'Shared Memory bank conflicts detected. Adjust indexing '
                    'with padding (e.g. TILE[N][N+1]) or restructure access '
                    'so that threads in a warp access distinct banks.'
                ),
            })

        # Warp Divergence
        divergence_ratio = self._m(metrics,
            'smsp__sass_thread_inst_executed_per_inst_executed.ratio', 32)
        if divergence_ratio and divergence_ratio < 28:
            bottlenecks.append({
                'type': 'WARP_DIVERGENCE',
                'severity': 'high' if divergence_ratio < 20 else 'medium',
                'metric': 'smsp__sass_thread_inst_executed_per_inst_executed.ratio',
                'value': divergence_ratio,
                'recommendation': (
                    'Significant warp divergence (ratio={:.1f}/32). Reduce '
                    'if/else branching within warps; partition work so threads '
                    'in the same warp follow identical control flow.'.format(
                        divergence_ratio)
                ),
            })

        # Uncoalesced Access (high L1 sector count relative to computation)
        l1_sectors = self._m(metrics,
            'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum')
        if l1_sectors > 50_000_000:  # Heuristic for large kernels
            bottlenecks.append({
                'type': 'EXCESSIVE_GLOBAL_LOADS',
                'severity': 'medium',
                'metric': 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
                'value': l1_sectors,
                'recommendation': (
                    'Excessive L1 global load sectors ({:.0f}). Check memory '
                    'access patterns: ensure coalesced access (adjacent '
                    'threads access adjacent addresses) and consider data '
                    'prefetching or shared memory caching.'.format(l1_sectors)
                ),
            })

        # Low Occupancy
        theoretical = self._m(metrics, 'sm__maximum_warps_per_active_cycle_pct')
        achieved = self._m(metrics,
            'sm__warps_active.avg.pct_of_peak_sustained_active')
        if theoretical > 0 and achieved > 0:
            gap = theoretical - achieved
            if gap > 20:
                bottlenecks.append({
                    'type': 'LOW_OCCUPANCY',
                    'severity': 'high' if gap > 40 else 'medium',
                    'metric': 'occupancy_gap',
                    'value': gap,
                    'recommendation': (
                        'Achieved occupancy ({:.1f}%) is significantly below '
                        'theoretical ({:.1f}%). Investigate instruction '
                        'latency hiding, register pressure, or uneven block '
                        'distribution.'.format(achieved, theoretical)
                    ),
                })

        return bottlenecks
