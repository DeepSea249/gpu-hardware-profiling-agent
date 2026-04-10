"""
Kernel Analyzer - Analyzes CUDA kernels using ncu to diagnose bottlenecks.

Implements the Section 1 analysis flow:
1. Roofline classification (compute vs memory bound)
2. Deep-dive into the relevant metric category
3. Anomaly detection (low occupancy, bank conflicts)
4. Recommendations mapped back to code patterns
"""

import logging
from .ncu_profiler import NCUProfiler
from .reasoning import ReasoningEngine

logger = logging.getLogger('GPUAgent.KernelAnalyzer')


class KernelAnalyzer:
    """Analyzes CUDA kernel performance using ncu metrics."""

    def __init__(self, reasoning: ReasoningEngine = None):
        self.ncu = NCUProfiler()
        self.reasoning = reasoning or ReasoningEngine()

    def analyze(self, binary_path: str, kernel_name: str = None) -> dict:
        """
        Perform full bottleneck analysis on a CUDA kernel.

        Args:
            binary_path: Path to the CUDA binary
            kernel_name: Specific kernel function name (optional)

        Returns:
            Structured analysis result with bottleneck diagnosis
        """
        self.reasoning.log_step(
            'kernel_analysis',
            f'Starting kernel analysis for {binary_path}'
        )

        if not self.ncu.available:
            self.reasoning.log_step(
                'kernel_analysis',
                'WARNING: ncu not available, cannot perform kernel analysis'
            )
            return {
                'error': 'ncu not available',
                'recommendation': 'Install NVIDIA Nsight Compute for kernel profiling'
            }

        # Step 1: Get the Roofline
        self.reasoning.log_step(
            'roofline',
            'Step 1: Collecting roofline metrics (compute vs memory utilization)'
        )

        analysis = self.ncu.profile_with_details(binary_path)

        if 'error' in analysis and analysis['error']:
            return analysis

        roofline = analysis.get('roofline', {})
        classification = roofline.get('classification', 'unknown')

        self.reasoning.log_step(
            'roofline',
            f'Kernel classified as: {classification}',
            data={
                'compute_util': roofline.get('compute_utilization_pct'),
                'memory_util': roofline.get('memory_utilization_pct'),
            }
        )

        # Step 2: Characterize - dive deeper based on classification
        if classification == 'memory_bound':
            self.reasoning.log_step(
                'characterize',
                'Step 2: Memory-bound - analyzing memory hierarchy (DRAM, L2, L1)'
            )
            memory = analysis.get('memory', {})
            self.reasoning.log_step(
                'characterize',
                'Memory hierarchy analysis complete',
                data=memory
            )

            if memory.get('dram_throughput_pct', 0) > 80:
                self.reasoning.log_step(
                    'recommendation',
                    'DRAM bandwidth is saturated (>80%). '
                    'Consider: shared memory tiling, data reuse, '
                    'reduced precision (FP16/BF16).'
                )

        elif classification == 'compute_bound':
            self.reasoning.log_step(
                'characterize',
                'Step 2: Compute-bound - analyzing compute units (Tensor, FMA)'
            )
            compute = analysis.get('compute', {})
            self.reasoning.log_step(
                'characterize',
                'Compute unit analysis complete',
                data=compute
            )

            tensor_util = compute.get('tensor_core_utilization_pct', 0)
            if tensor_util is not None and tensor_util < 30:
                self.reasoning.log_step(
                    'recommendation',
                    f'Tensor Core utilization is low ({tensor_util:.1f}%). '
                    'For matrix operations, consider: using WMMA/MMA instructions, '
                    'aligning dimensions to 16/32, using FP16 inputs.'
                )

        # Step 3: Look for Anomalies
        self.reasoning.log_step(
            'anomalies',
            'Step 3: Checking for occupancy issues and bank conflicts'
        )

        occupancy = analysis.get('occupancy', {})
        gap = occupancy.get('occupancy_gap_pct', 0)
        if gap and gap > 20:
            self.reasoning.log_anomaly(
                'LOW_OCCUPANCY',
                f'Significant occupancy gap: theoretical='
                f'{occupancy.get("theoretical_occupancy_pct", 0):.1f}%, '
                f'achieved={occupancy.get("achieved_occupancy_pct", 0):.1f}%. '
                'Possible causes: instruction latency, uneven block distribution.',
                expected=occupancy.get('theoretical_occupancy_pct'),
                measured=occupancy.get('achieved_occupancy_pct'),
            )

        bottlenecks = analysis.get('bottlenecks', [])
        for bn in bottlenecks:
            self.reasoning.log_step(
                'bottleneck',
                f'Detected: {bn["type"]} (severity={bn["severity"]})',
                data={'metric': bn['metric'], 'value': bn['value']}
            )

        # Step 4: Summary & Recommendations
        analysis['summary'] = self._generate_summary(analysis)
        self.reasoning.log_step(
            'summary',
            'Step 4: Analysis complete',
            data={'summary': analysis['summary']}
        )

        return analysis

    def _generate_summary(self, analysis: dict) -> dict:
        """Generate a summary of the kernel analysis."""
        roofline = analysis.get('roofline', {})
        bottlenecks = analysis.get('bottlenecks', [])

        recommendations = []
        for bn in bottlenecks:
            recommendations.append(bn['recommendation'])

        return {
            'classification': roofline.get('classification', 'unknown'),
            'primary_bottleneck': bottlenecks[0]['type'] if bottlenecks else 'none',
            'num_issues': len(bottlenecks),
            'recommendations': recommendations,
        }
