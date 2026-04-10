"""
Kernel Analyzer - Analyzes CUDA kernels using ncu to diagnose bottlenecks.

Implements the full Section 1 analysis flow:
  Step 1: Roofline (compute vs memory bound)   – Section 1.1
  Step 2: Deep-dive into relevant metric class  – Section 1.2 / 1.3
  Step 3: Anomaly scan (occupancy, bank conflicts, divergence) – Section 1.4/1.5
  Step 4: LLM synthesis of all findings into an actionable report

Produces a structured JSON analysis AND an LLM-authored narrative
suitable for inclusion in a course submission or CI report.
"""

import json
import logging
from .ncu_profiler import NCUProfiler
from .reasoning import ReasoningEngine

logger = logging.getLogger('GPUAgent.KernelAnalyzer')

# Lazy LLM helpers (identical interface to reasoning.py)
_llm_client = None


def _get_llm():
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    try:
        from .llm_client import LLMClient
        _llm_client = LLMClient()
    except Exception:
        _llm_client = None
    return _llm_client


def _llm_call(system: str, user: str, fallback: str) -> str:
    client = _get_llm()
    if client is None:
        return fallback
    try:
        answer = client.generate_reasoning(system, user)
        if answer and answer.strip():
            return answer.strip()
        return fallback
    except Exception as exc:
        logger.warning("LLM call failed in kernel_analyzer: %s", exc)
        return fallback


class KernelAnalyzer:
    """Analyzes CUDA kernel performance using ncu metrics."""

    def __init__(self, reasoning: ReasoningEngine = None):
        self.ncu = NCUProfiler()
        self.reasoning = reasoning or ReasoningEngine()

    def analyze(self, binary_path: str, kernel_name: str = None) -> dict:
        """
        Perform full bottleneck analysis on a CUDA kernel.

        Following the Section 1.1-1.6 methodology:
          1. Collect ncu metrics (roofline + detail)
          2. Classify compute vs memory bound
          3. Deep-dive into the saturated subsystem
          4. Scan for anomalies (occupancy, bank conflicts, divergence)
          5. Synthesise findings with LLM

        Returns a structured dict with analysis, bottlenecks,
        recommendations, and an LLM-authored narrative.
        """
        self.reasoning.log_step(
            'kernel_analysis',
            f'Starting kernel bottleneck analysis for {binary_path}'
            + (f' (kernel={kernel_name})' if kernel_name else ''),
        )

        if not self.ncu.available:
            self.reasoning.log_step(
                'kernel_analysis',
                'WARNING: ncu not available – cannot perform kernel analysis'
            )
            return {
                'error': 'ncu not available',
                'recommendation': 'Install NVIDIA Nsight Compute for kernel profiling',
            }

        # ---------------------------------------------------------------- #
        # Step 1: Get the Roofline  (Section 1.1)                          #
        # ---------------------------------------------------------------- #
        self.reasoning.log_step(
            'roofline',
            'Step 1: Collecting all analysis metrics via ncu (skip warmup launch)',
        )

        analysis = self.ncu.profile_with_details(
            binary_path, kernel_name=kernel_name, timeout=300,
        )

        if 'error' in analysis and analysis.get('error'):
            self.reasoning.log_step(
                'roofline', f'ncu profiling failed: {analysis["error"]}',
            )
            return analysis

        roofline = analysis.get('roofline', {})
        metrics = analysis.get('metrics', {})
        kernel_info = analysis.get('kernel_info', {})
        classification = roofline.get('classification', 'unknown')

        self.reasoning.log_step(
            'roofline',
            f'Kernel "{kernel_info.get("kernel_name", "?")}" classified as '
            f'{classification.upper()}',
            data={
                'compute_utilization_pct': roofline.get('compute_utilization_pct'),
                'memory_utilization_pct': roofline.get('memory_utilization_pct'),
                'block_size': kernel_info.get('block_size'),
                'grid_size': kernel_info.get('grid_size'),
            },
        )

        # ---------------------------------------------------------------- #
        # Step 2: Characterise – deep-dive  (Section 1.2 / 1.3)           #
        # ---------------------------------------------------------------- #
        memory = analysis.get('memory', {})
        compute = analysis.get('compute', {})

        if classification in ('memory_bound', 'balanced'):
            self.reasoning.log_step(
                'characterise',
                'Step 2: Memory-bound path – analysing DRAM, L2, L1 hierarchy',
                data=memory,
            )
            dram_pct = memory.get('dram_throughput_pct', 0)
            if dram_pct > 80:
                self.reasoning.log_step(
                    'characterise',
                    f'DRAM bandwidth saturated at {dram_pct:.1f}% – kernel is '
                    'VRAM-limited. Shared-memory tiling or reduced precision '
                    'recommended.',
                )
            l2_pct = memory.get('l2_throughput_pct', 0)
            if l2_pct > 70:
                self.reasoning.log_step(
                    'characterise',
                    f'L2 throughput at {l2_pct:.1f}% – heavy SM ↔ VRAM traffic.',
                )

        if classification in ('compute_bound', 'balanced'):
            self.reasoning.log_step(
                'characterise',
                'Step 2: Compute-bound path – analysing Tensor / FMA units',
                data=compute,
            )
            tensor_pct = compute.get('tensor_core_utilization_pct', 0)
            fma_pct = compute.get('fma_utilization_pct', 0)
            if tensor_pct < 5 and fma_pct > 0:
                self.reasoning.log_step(
                    'characterise',
                    f'Tensor Cores idle ({tensor_pct:.1f}%) while FMA active '
                    f'({fma_pct:.1f}%). For matmul-like kernels, enabling '
                    'WMMA/MMA would dramatically improve throughput.',
                )

        # ---------------------------------------------------------------- #
        # Step 3: Anomaly scan  (Section 1.4 / 1.5)                       #
        # ---------------------------------------------------------------- #
        self.reasoning.log_step(
            'anomaly_scan',
            'Step 3: Scanning for occupancy, bank-conflict, and divergence anomalies',
        )

        occupancy = analysis.get('occupancy', {})
        gap = occupancy.get('occupancy_gap_pct', 0)
        if gap > 20:
            self.reasoning.log_anomaly(
                'LOW_OCCUPANCY',
                f'Occupancy gap: theoretical={occupancy["theoretical_occupancy_pct"]:.1f}%, '
                f'achieved={occupancy["achieved_occupancy_pct"]:.1f}% (gap={gap:.1f}%). '
                'Potential causes: high register count, shared-memory pressure, '
                'instruction latency, or unbalanced block distribution.',
                expected=occupancy.get('theoretical_occupancy_pct'),
                measured=occupancy.get('achieved_occupancy_pct'),
            )

        bottlenecks = analysis.get('bottlenecks', [])
        for bn in bottlenecks:
            self.reasoning.log_step(
                'bottleneck',
                f'Detected: {bn["type"]} (severity={bn["severity"]}, '
                f'metric={bn["metric"]}, value={bn["value"]})',
                data=bn,
            )

        # Log all raw metric values for reference
        self.reasoning.log_step(
            'raw_metrics',
            f'All {len(metrics)} ncu metrics collected',
            data={k: v for k, v in sorted(metrics.items())},
        )

        # ---------------------------------------------------------------- #
        # Step 4: LLM Synthesis  (combine all findings)                    #
        # ---------------------------------------------------------------- #
        self.reasoning.log_step(
            'llm_synthesis',
            'Step 4: Generating LLM-authored bottleneck analysis',
        )

        llm_report = self._generate_llm_report(analysis)
        analysis['llm_report'] = llm_report

        # Build structured summary
        analysis['summary'] = self._generate_summary(analysis)

        self.reasoning.log_step(
            'kernel_analysis',
            'Kernel analysis complete',
            data={
                'classification': classification,
                'num_bottlenecks': len(bottlenecks),
                'primary_bottleneck': bottlenecks[0]['type'] if bottlenecks else 'none',
            },
        )

        return analysis

    # ------------------------------------------------------------------ #
    #  LLM report generation                                              #
    # ------------------------------------------------------------------ #
    def _generate_llm_report(self, analysis: dict) -> str:
        """Ask the LLM to produce a professional bottleneck analysis."""
        metrics = analysis.get('metrics', {})
        roofline = analysis.get('roofline', {})
        memory = analysis.get('memory', {})
        compute = analysis.get('compute', {})
        occupancy = analysis.get('occupancy', {})
        bottlenecks = analysis.get('bottlenecks', [])
        kernel_info = analysis.get('kernel_info', {})

        # Build context for the LLM
        lines = [
            f"Kernel: {kernel_info.get('kernel_name', 'unknown')}",
            f"Block Size: {kernel_info.get('block_size', '?')}",
            f"Grid Size: {kernel_info.get('grid_size', '?')}",
            f"Compute Capability: {kernel_info.get('cc', '?')}",
            "",
            "=== Roofline Classification ===",
            f"  SM Throughput (Compute SOL): {roofline.get('compute_utilization_pct', 0):.1f}%",
            f"  Memory Throughput (Memory SOL): {roofline.get('memory_utilization_pct', 0):.1f}%",
            f"  Classification: {roofline.get('classification', 'unknown')}",
            "",
            "=== Memory Hierarchy ===",
            f"  DRAM Throughput: {memory.get('dram_throughput_pct', 0):.1f}%",
            f"  L2 Throughput: {memory.get('l2_throughput_pct', 0):.1f}%",
            f"  L1 Global Load Sectors: {memory.get('l1_global_load_sectors', 0):.0f}",
            f"  Bank Conflicts: {memory.get('bank_conflicts', 0):.0f}",
            "",
            "=== Compute Units ===",
            f"  Tensor Core Utilisation: {compute.get('tensor_core_utilization_pct', 0):.1f}%",
            f"  FMA/FP32 Utilisation: {compute.get('fma_utilization_pct', 0):.1f}%",
            f"  FP32 Instructions: {compute.get('fp32_instructions', 0):.0f}",
            "",
            "=== Occupancy ===",
            f"  Theoretical: {occupancy.get('theoretical_occupancy_pct', 0):.1f}%",
            f"  Achieved: {occupancy.get('achieved_occupancy_pct', 0):.1f}%",
            f"  Gap: {occupancy.get('occupancy_gap_pct', 0):.1f}%",
            "",
            f"=== Detected Bottlenecks ({len(bottlenecks)}) ===",
        ]
        for bn in bottlenecks:
            lines.append(f"  [{bn['severity'].upper()}] {bn['type']}: "
                         f"{bn['metric']}={bn['value']}")
            lines.append(f"    → {bn['recommendation']}")

        context = '\n'.join(lines)

        fallback = self._generate_template_report(analysis)

        return _llm_call(
            system=(
                "You are an expert GPU performance engineer writing a "
                "kernel bottleneck analysis report for a graduate "
                "ML-Systems course. Given the ncu profiling metrics below, "
                "produce a clear, structured analysis that:\n"
                "1. States whether the kernel is compute-bound or memory-bound "
                "   and explains why (cite SOL numbers).\n"
                "2. Identifies the primary bottleneck (VRAM saturation, "
                "   uncoalesced access, bank conflicts, low Tensor Core "
                "   usage, warp divergence, low occupancy).\n"
                "3. Provides specific, actionable optimisation recommendations "
                "   mapped back to CUDA code patterns.\n"
                "4. Notes any secondary issues.\n"
                "Use markdown formatting. ~400 words."
            ),
            user=context,
            fallback=fallback,
        )

    def _generate_template_report(self, analysis: dict) -> str:
        """Fallback template-based report when LLM is unavailable."""
        roofline = analysis.get('roofline', {})
        bottlenecks = analysis.get('bottlenecks', [])
        memory = analysis.get('memory', {})
        compute = analysis.get('compute', {})
        occupancy = analysis.get('occupancy', {})

        lines = [
            "# Kernel Bottleneck Analysis Report",
            "",
            "## Roofline Classification",
            f"- Compute SOL: {roofline.get('compute_utilization_pct', 0):.1f}%",
            f"- Memory SOL: {roofline.get('memory_utilization_pct', 0):.1f}%",
            f"- Classification: **{roofline.get('classification', 'unknown').upper()}**",
            "",
            "## Memory Hierarchy",
            f"- DRAM Throughput: {memory.get('dram_throughput_pct', 0):.1f}%",
            f"- L2 Throughput: {memory.get('l2_throughput_pct', 0):.1f}%",
            f"- Bank Conflicts: {memory.get('bank_conflicts', 0):.0f}",
            "",
            "## Compute Units",
            f"- Tensor Core: {compute.get('tensor_core_utilization_pct', 0):.1f}%",
            f"- FMA/FP32: {compute.get('fma_utilization_pct', 0):.1f}%",
            "",
            "## Occupancy",
            f"- Theoretical: {occupancy.get('theoretical_occupancy_pct', 0):.1f}%",
            f"- Achieved: {occupancy.get('achieved_occupancy_pct', 0):.1f}%",
            "",
        ]
        if bottlenecks:
            lines.append("## Bottlenecks Detected")
            for bn in bottlenecks:
                lines.append(f"- **{bn['type']}** ({bn['severity']}): "
                             f"{bn['recommendation']}")
        else:
            lines.append("## No major bottlenecks detected.")

        return '\n'.join(lines)

    def _generate_summary(self, analysis: dict) -> dict:
        """Generate a structured summary of the kernel analysis."""
        roofline = analysis.get('roofline', {})
        bottlenecks = analysis.get('bottlenecks', [])

        return {
            'classification': roofline.get('classification', 'unknown'),
            'compute_sol_pct': roofline.get('compute_utilization_pct', 0),
            'memory_sol_pct': roofline.get('memory_utilization_pct', 0),
            'primary_bottleneck': bottlenecks[0]['type'] if bottlenecks else 'none',
            'num_issues': len(bottlenecks),
            'bottleneck_types': [bn['type'] for bn in bottlenecks],
            'recommendations': [bn['recommendation'] for bn in bottlenecks],
        }
