"""
NCU Profiler - Nsight Compute integration for kernel analysis.

Provides:
- Run ncu on CUDA binaries with specified metrics
- Parse ncu CSV output into structured data
- Classify kernels as compute-bound or memory-bound
- Generate diagnostic reports
"""

import subprocess
import shutil
import logging
import re
import csv
import io

logger = logging.getLogger('GPUAgent.NCUProfiler')


class NCUProfiler:
    """Wraps NVIDIA Nsight Compute (ncu) for kernel profiling."""

    # Core metrics for roofline analysis
    ROOFLINE_METRICS = [
        'sm__throughput.avg.pct_of_peak_sustained_elapsed',
        'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
    ]

    # Memory hierarchy metrics
    MEMORY_METRICS = [
        'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
        'l2__throughput.avg.pct_of_peak_sustained_elapsed',
        'dram__throughput.avg.pct_of_peak_sustained_elapsed',
        'l1tex__data_bank_conflicts_pipe_lsu.sum',
    ]

    # Compute unit metrics
    COMPUTE_METRICS = [
        'sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active',
        'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active',
        'sm__sass_thread_inst_executed_op_fp32_pred_on.sum',
    ]

    # Occupancy metrics
    OCCUPANCY_METRICS = [
        'sm__maximum_warps_per_active_cycle_pct',
        'sm__warps_active.avg.pct_of_peak_sustained_active',
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
                kernel_name: str = None, timeout: int = 300) -> dict:
        """
        Run ncu on a binary and return parsed metrics.

        Args:
            binary: Path to CUDA binary
            metrics: List of metric names (default: all analysis metrics)
            kernel_name: Specific kernel to profile (optional)
            timeout: Timeout in seconds

        Returns:
            Dictionary of metric_name -> value
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

        cmd.append(binary)

        logger.info(f"Running ncu: {' '.join(cmd[:6])}...")

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
            logger.error(f"ncu failed: {result.stderr[:500]}")
            return {}

        return self._parse_csv_output(result.stdout)

    def _parse_csv_output(self, csv_text: str) -> dict:
        """Parse ncu CSV output into a metric dictionary."""
        metrics = {}

        # ncu CSV output has header rows then data
        lines = csv_text.strip().split('\n')
        data_lines = [l for l in lines if not l.startswith('=')]

        if not data_lines:
            return metrics

        try:
            reader = csv.DictReader(io.StringIO('\n'.join(data_lines)))
            for row in reader:
                name = row.get('Metric Name', '').strip()
                value_str = row.get('Metric Value', '').strip()

                if name and value_str:
                    try:
                        # Remove commas and percentage signs
                        clean = value_str.replace(',', '').replace('%', '')
                        metrics[name] = float(clean)
                    except ValueError:
                        metrics[name] = value_str
        except Exception as e:
            logger.warning(f"Error parsing ncu CSV: {e}")
            # Try regex-based fallback parsing
            metrics = self._parse_regex_fallback(csv_text)

        return metrics

    def _parse_regex_fallback(self, text: str) -> dict:
        """Fallback regex-based parsing for ncu output."""
        metrics = {}
        # Match patterns like: metric_name ... value
        for line in text.split('\n'):
            # Look for metric value pairs
            match = re.search(
                r'"([^"]+)"[^"]*"([^"]*)"[^"]*"([^"]*)"',
                line
            )
            if match:
                name = match.group(1).strip()
                value_str = match.group(3).strip()
                try:
                    clean = value_str.replace(',', '').replace('%', '')
                    metrics[name] = float(clean)
                except ValueError:
                    metrics[name] = value_str

        return metrics

    def profile_with_details(self, binary: str, timeout: int = 300) -> dict:
        """
        Run comprehensive ncu profiling with all analysis metrics.
        Returns structured analysis result.
        """
        metrics = self.profile(binary, self.ALL_ANALYSIS_METRICS, timeout=timeout)

        if not metrics:
            return {'error': 'No metrics collected', 'metrics': {}}

        analysis = {
            'metrics': metrics,
            'roofline': self._analyze_roofline(metrics),
            'memory': self._analyze_memory(metrics),
            'compute': self._analyze_compute(metrics),
            'occupancy': self._analyze_occupancy(metrics),
            'bottlenecks': self._identify_bottlenecks(metrics),
        }

        return analysis

    def _analyze_roofline(self, metrics: dict) -> dict:
        """Determine compute vs memory bound classification."""
        compute_util = metrics.get(
            'sm__throughput.avg.pct_of_peak_sustained_elapsed', 0
        )
        memory_util = metrics.get(
            'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed', 0
        )

        classification = 'balanced'
        if compute_util > memory_util * 1.2:
            classification = 'compute_bound'
        elif memory_util > compute_util * 1.2:
            classification = 'memory_bound'

        return {
            'compute_utilization_pct': compute_util,
            'memory_utilization_pct': memory_util,
            'classification': classification,
        }

    def _analyze_memory(self, metrics: dict) -> dict:
        """Analyze memory hierarchy utilization."""
        return {
            'l2_throughput_pct': metrics.get(
                'l2__throughput.avg.pct_of_peak_sustained_elapsed', 0
            ),
            'dram_throughput_pct': metrics.get(
                'dram__throughput.avg.pct_of_peak_sustained_elapsed', 0
            ),
            'bank_conflicts': metrics.get(
                'l1tex__data_bank_conflicts_pipe_lsu.sum', 0
            ),
        }

    def _analyze_compute(self, metrics: dict) -> dict:
        """Analyze compute unit utilization."""
        return {
            'tensor_core_utilization_pct': metrics.get(
                'sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active', 0
            ),
            'fma_utilization_pct': metrics.get(
                'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active', 0
            ),
        }

    def _analyze_occupancy(self, metrics: dict) -> dict:
        """Analyze warp occupancy."""
        theoretical = metrics.get('sm__maximum_warps_per_active_cycle_pct', 0)
        achieved = metrics.get(
            'sm__warps_active.avg.pct_of_peak_sustained_active', 0
        )

        return {
            'theoretical_occupancy_pct': theoretical,
            'achieved_occupancy_pct': achieved,
            'occupancy_gap_pct': theoretical - achieved if theoretical and achieved else 0,
        }

    def _identify_bottlenecks(self, metrics: dict) -> list:
        """Identify specific bottlenecks based on metric thresholds."""
        bottlenecks = []

        dram_tp = metrics.get(
            'dram__throughput.avg.pct_of_peak_sustained_elapsed', 0
        )
        if dram_tp and dram_tp > 70:
            bottlenecks.append({
                'type': 'VRAM_BOUND',
                'severity': 'high' if dram_tp > 85 else 'medium',
                'metric': 'dram__throughput',
                'value': dram_tp,
                'recommendation': 'Reduce memory access; increase data reuse; use Shared Memory.',
            })

        sm_tp = metrics.get(
            'sm__throughput.avg.pct_of_peak_sustained_elapsed', 0
        )
        if sm_tp and sm_tp > 70:
            bottlenecks.append({
                'type': 'COMPUTE_BOUND',
                'severity': 'high' if sm_tp > 85 else 'medium',
                'metric': 'sm__throughput',
                'value': sm_tp,
                'recommendation': 'Consider algorithmic improvements or precision reduction (FP16/BF16).',
            })

        bank_conflicts = metrics.get(
            'l1tex__data_bank_conflicts_pipe_lsu.sum', 0
        )
        if bank_conflicts and bank_conflicts > 0:
            bottlenecks.append({
                'type': 'BANK_CONFLICT',
                'severity': 'medium',
                'metric': 'l1tex__data_bank_conflicts_pipe_lsu.sum',
                'value': bank_conflicts,
                'recommendation': 'Adjust Shared Memory indexing (e.g., padding).',
            })

        divergence_ratio = metrics.get(
            'smsp__sass_thread_inst_executed_per_inst_executed.ratio', 32
        )
        if divergence_ratio and divergence_ratio < 28:
            bottlenecks.append({
                'type': 'WARP_DIVERGENCE',
                'severity': 'high' if divergence_ratio < 20 else 'medium',
                'metric': 'thread_inst_executed_per_inst_executed',
                'value': divergence_ratio,
                'recommendation': 'Reduce branching; ensure threads in a warp follow the same path.',
            })

        return bottlenecks
