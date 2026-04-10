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
import os
import glob
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
        _llm_client._client.timeout = 120  # 2-minute cap on streaming
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

    def analyze(self, binary_path: str, kernel_name: str = None,
                 source_path: str = None) -> dict:
        """
        Perform full bottleneck analysis on a CUDA kernel.

        Following the Section 1.1-1.6 methodology:
          1. Get the Roofline (sm__throughput vs memory throughput)
          2. Characterise (memory→dram/l1/l2, compute→tensor_op)
          3. Look for Anomalies (occupancy, bank conflicts, divergence)
          4. Map back to Code (relate metrics to CUDA kernel source)

        Args:
            binary_path: Path to the compiled CUDA binary.
            kernel_name: Optional specific kernel function name.
            source_path: Optional path to the .cu source file.
                         If omitted, attempts to locate it automatically.

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
        # Step 4: Map back to Code  (Section 1.6)                         #
        # ---------------------------------------------------------------- #
        self.reasoning.log_step(
            'code_mapping',
            'Step 4: Mapping metrics back to CUDA kernel code',
        )

        kernel_source = self._find_kernel_source(binary_path, source_path)
        if kernel_source:
            analysis['kernel_source'] = kernel_source
            code_issues = self._map_metrics_to_code(
                kernel_source, analysis
            )
            analysis['code_issues'] = code_issues
            for issue in code_issues:
                self.reasoning.log_step(
                    'code_mapping',
                    f'Code issue: {issue["issue"]} → {issue["suggestion"]}',
                    data=issue,
                )
        else:
            analysis['code_issues'] = []
            self.reasoning.log_step(
                'code_mapping',
                'Kernel source not found — code mapping skipped '
                '(LLM will provide generic recommendations)',
            )

        # ---------------------------------------------------------------- #
        # LLM Synthesis  (combine all findings incl. source code)          #
        # ---------------------------------------------------------------- #
        self.reasoning.log_step(
            'llm_synthesis',
            'Generating LLM-authored bottleneck analysis',
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
    #  Step 4 helpers: find source & map metrics to code                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _find_kernel_source(binary_path: str,
                            source_path: str = None) -> str | None:
        """Try to locate and read the .cu source file for a compiled binary.

        Search order:
          1. Explicit *source_path* argument.
          2. Same basename with .cu in well-known source directories.
          3. Glob any .cu files in nearby directories.

        Returns the source text, or None.
        """
        if source_path and os.path.isfile(source_path):
            try:
                return open(source_path).read()
            except Exception:
                pass

        base = os.path.splitext(os.path.basename(binary_path))[0]
        # Walk up from the binary to find a sibling "kernels/" dir
        binary_dir = os.path.dirname(os.path.abspath(binary_path))
        search_roots = [
            binary_dir,
            os.path.dirname(binary_dir),          # e.g. build/../
            os.path.dirname(os.path.dirname(binary_dir)),  # build/kernels/../../
        ]
        for root in search_roots:
            for candidate in [
                os.path.join(root, f'{base}.cu'),
                os.path.join(root, 'kernels', f'{base}.cu'),
                os.path.join(root, 'src', f'{base}.cu'),
            ]:
                if os.path.isfile(candidate):
                    try:
                        return open(candidate).read()
                    except Exception:
                        pass

        # Last resort: glob upward
        for root in search_roots:
            for cu in glob.glob(os.path.join(root, '**', f'{base}.cu'),
                                recursive=True):
                try:
                    return open(cu).read()
                except Exception:
                    pass

        return None

    @staticmethod
    def _map_metrics_to_code(source: str, analysis: dict) -> list[dict]:
        """Heuristic code-pattern scan that relates ncu bottlenecks to source.

        Returns a list of dicts: {issue, pattern, line_hint, suggestion}.
        """
        issues: list[dict] = []
        bottleneck_types = {
            bn['type'] for bn in analysis.get('bottlenecks', [])
        }
        classification = analysis.get('roofline', {}).get(
            'classification', '')
        memory = analysis.get('memory', {})
        compute = analysis.get('compute', {})

        has_shared = '__shared__' in source
        has_syncthreads = '__syncthreads' in source
        has_wmma = 'wmma' in source.lower() or 'mma' in source.lower()
        has_tile_loop = 'TILE' in source or 'tile' in source
        has_pragma_unroll = '#pragma unroll' in source

        # VRAM / memory-bound but no shared-memory tiling
        if ('VRAM_BOUND' in bottleneck_types
                or classification == 'memory_bound'):
            if not has_shared:
                issues.append({
                    'issue': 'Memory-bound kernel with no __shared__ memory',
                    'pattern': 'missing __shared__',
                    'line_hint': None,
                    'suggestion': (
                        'Add shared-memory tiling: load a tile of input '
                        'into __shared__ arrays, __syncthreads(), then '
                        'compute from shared memory to reduce DRAM traffic.'
                    ),
                })
            elif has_shared and not has_tile_loop:
                issues.append({
                    'issue': 'Shared memory declared but no tiling loop detected',
                    'pattern': '__shared__ without TILE',
                    'line_hint': None,
                    'suggestion': (
                        'Ensure the kernel uses a tiled loop (e.g. '
                        'for(t=0; t<K; t+=TILE_SIZE)) to stream data '
                        'through shared memory and maximise reuse.'
                    ),
                })

        # Bank conflicts detected
        bank_conflicts = memory.get('bank_conflicts', 0)
        if 'BANK_CONFLICT' in bottleneck_types or bank_conflicts > 10000:
            # Try to find shared-memory declarations and hint about padding
            if has_shared and '+1' not in source and 'PADDING' not in source:
                issues.append({
                    'issue': 'Bank conflicts with unpadded shared memory',
                    'pattern': '__shared__ without padding',
                    'line_hint': None,
                    'suggestion': (
                        'Pad shared-memory arrays: use tile[N][N+1] instead '
                        'of tile[N][N] to avoid bank conflicts when threads '
                        'in a warp access the same bank.'
                    ),
                })

        # Compute-bound but no Tensor Core usage
        tensor_pct = compute.get('tensor_core_utilization_pct', 0)
        if ('TENSOR_CORES_UNUSED' in bottleneck_types
                or (classification == 'compute_bound' and tensor_pct < 5)):
            if not has_wmma:
                issues.append({
                    'issue': 'Compute-bound with Tensor Cores idle',
                    'pattern': 'no wmma/mma intrinsics',
                    'line_hint': None,
                    'suggestion': (
                        'For matrix-multiply-like kernels, use WMMA/MMA '
                        'intrinsics (nvcuda::wmma) with FP16 inputs to '
                        'leverage Tensor Core hardware and dramatically '
                        'increase throughput.'
                    ),
                })

        # No loop unrolling
        if not has_pragma_unroll and classification in (
                'compute_bound', 'memory_bound', 'balanced'):
            issues.append({
                'issue': 'No #pragma unroll in inner loops',
                'pattern': 'missing #pragma unroll',
                'line_hint': None,
                'suggestion': (
                    'Add #pragma unroll before tight inner loops to '
                    'reduce loop overhead and enable instruction-level '
                    'parallelism.'
                ),
            })

        # Low occupancy — check for large register / shared-memory usage hints
        if 'LOW_OCCUPANCY' in bottleneck_types:
            # Count local variable declarations as a rough register-pressure proxy
            local_vars = source.count('float ') + source.count('int ')
            if local_vars > 30:
                issues.append({
                    'issue': 'Low occupancy with many local variables',
                    'pattern': 'high register pressure',
                    'line_hint': None,
                    'suggestion': (
                        'Reduce register pressure: combine variables, '
                        'use __launch_bounds__(maxThreads, minBlocks) '
                        'to give the compiler an occupancy hint, or '
                        'consider using float2/float4 vectors.'
                    ),
                })

        return issues

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

        # Include automated code-mapping findings (Step 4)
        code_issues = analysis.get('code_issues', [])
        if code_issues:
            lines.append("")
            lines.append(f"=== Code Pattern Issues ({len(code_issues)}) ===")
            for ci in code_issues:
                lines.append(f"  • {ci['issue']}")
                lines.append(f"    Fix: {ci['suggestion']}")

        # Include kernel source (truncated) so the LLM can reference
        # specific lines and patterns
        kernel_source = analysis.get('kernel_source')
        if kernel_source:
            # Limit to first 200 lines to stay within token budget
            src_lines = kernel_source.splitlines()[:200]
            numbered = '\n'.join(
                f'{i+1:4d}| {l}' for i, l in enumerate(src_lines)
            )
            lines.append("")
            lines.append("=== Kernel Source Code ===")
            lines.append(numbered)

        context = '\n'.join(lines)

        fallback = self._generate_template_report(analysis)

        return _llm_call(
            system=(
                "You are an expert GPU performance engineer writing a "
                "kernel bottleneck analysis report for a graduate "
                "ML-Systems course. Given the ncu profiling metrics AND "
                "the kernel source code below, produce a clear, structured "
                "analysis that:\n"
                "1. States whether the kernel is compute-bound or memory-bound "
                "   and explains why (cite SOL numbers).\n"
                "2. Identifies the primary bottleneck (VRAM saturation, "
                "   uncoalesced access, bank conflicts, low Tensor Core "
                "   usage, warp divergence, low occupancy).\n"
                "3. Maps bottlenecks back to *specific lines or patterns* in "
                "   the kernel source code (e.g. missing __shared__ arrays, "
                "   insufficient loop unrolling, non-coalesced access).\n"
                "4. Provides specific, actionable optimisation recommendations "
                "   that reference the code.\n"
                "5. Notes any secondary issues.\n"
                "Use markdown formatting. ~500 words."
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

        code_issues = analysis.get('code_issues', [])
        if code_issues:
            lines.append("")
            lines.append("## Code Pattern Issues")
            for ci in code_issues:
                lines.append(f"- **{ci['issue']}**: {ci['suggestion']}")

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
            'code_issues': [ci['issue'] for ci in analysis.get('code_issues', [])],
        }
