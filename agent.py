#!/usr/bin/env python3
"""
GPU Hardware Intrinsic Profiling Agent
======================================

Main entry point for the MLSys course project.

This agent autonomously probes GPU hardware characteristics using
CUDA micro-benchmarks and NVIDIA Nsight Compute (ncu) profiling.
It produces accurate measurements even under non-standard configurations
such as frequency locking, SM masking, or spoofed device properties.

Architecture:
  1. Reads target_spec.json to determine which metrics to measure
     and (optionally) which executable to profile for bottleneck analysis
  2. Compiles and runs CUDA micro-benchmark probes
  3. Parses raw output and applies statistical analysis
  4. Cross-verifies results using multiple methods (probes, ncu, nvidia-smi)
  5. Detects anomalies (frequency locking, SM masking)
  6. If "run" is specified in target_spec.json, performs kernel bottleneck
     analysis on that executable via ncu (Sections 1.1-1.6)
  7. Outputs results.json with measurements, kernel analysis, and reasoning log

Usage:
  python3 agent.py [--target-spec target_spec.json] [--output results.json]
  python3 agent.py --kernel ./my_cuda_binary  # For standalone kernel analysis
"""

import argparse
import json
import os
import sys
import logging

from src.hardware_prober import HardwareProber
from src.kernel_analyzer import KernelAnalyzer
from src.reasoning import ReasoningEngine
from src import utils

# Configure logging with both console and file output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent.log', mode='w'),
    ]
)
logger = logging.getLogger('GPUAgent')


def load_target_spec(spec_path: str) -> dict:
    """Load and validate target_spec.json.

    Returns the parsed dict which may contain:
      - "targets": list of hardware metric names to probe
      - "run":     path to a CUDA binary to profile (optional)
    """
    if not os.path.exists(spec_path):
        logger.error(f"Target spec not found: {spec_path}")
        return {}
    with open(spec_path, 'r') as f:
        return json.load(f)


def run_hardware_probing(targets: list, args, reasoning: ReasoningEngine) -> dict:
    """
    Phase 1: Hardware Intrinsic Probing

    Runs micro-benchmarks for the requested target metrics and returns
    the measured values.
    """
    logger.info("=" * 60)
    logger.info("Phase 1: Hardware Intrinsic Profiling")
    logger.info("=" * 60)

    if not targets:
        logger.warning("No hardware target metrics requested — skipping Phase 1")
        return {}

    logger.info(f"Target metrics ({len(targets)}): {targets}")
    reasoning.log_step(
        'initialization',
        f'Loaded target_spec.json with {len(targets)} metrics',
        data={'targets': targets}
    )

    # Initialize the hardware prober
    prober = HardwareProber(
        probe_dir=args.probe_dir,
        build_dir=args.build_dir,
        num_trials=args.trials,
        reasoning=reasoning,
    )

    # Run all probes and collect results
    results = prober.probe_all(targets)

    return results


def run_kernel_analysis(binary_path: str, args, reasoning: ReasoningEngine,
                        kernel_name: str = None) -> dict:
    """
    Phase 2: Kernel Performance Analysis

    Uses ncu to profile a CUDA kernel and diagnose bottlenecks
    following the Section 1.1-1.6 analysis methodology.

    The binary_path can come from:
      - the "run" field in target_spec.json   (evaluation workflow)
      - the --kernel CLI flag                 (manual / standalone)
    """
    logger.info("=" * 60)
    logger.info("Phase 2: Kernel Bottleneck Analysis")
    logger.info("=" * 60)
    logger.info(f"Binary to profile: {binary_path}")

    # If the user supplied a .cu source file, compile it first
    if binary_path.endswith('.cu'):
        from src.probe_manager import ProbeManager
        pm = ProbeManager(probe_dir=os.path.dirname(binary_path) or '.',
                          build_dir=args.build_dir)
        probe_name = os.path.splitext(os.path.basename(binary_path))[0]
        compiled = pm.compile(probe_name, source_path=binary_path)
        if not compiled:
            logger.error(f"Failed to compile {binary_path}")
            return {'error': f'compilation failed for {binary_path}'}
        binary_path = compiled

    analyzer = KernelAnalyzer(reasoning=reasoning)
    analysis = analyzer.analyze(binary_path, kernel_name=kernel_name)

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description='GPU Hardware Intrinsic Profiling Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the full evaluation workflow (reads targets + executable from spec)
  python3 agent.py --target-spec target_spec.json --output results.json

  # Analyze a specific CUDA kernel for bottlenecks (standalone)
  python3 agent.py --kernel ./matmul

  # Both: probe hardware and analyze kernel
  python3 agent.py --target-spec target_spec.json --kernel ./matmul

  # Customize probe parameters
  python3 agent.py --trials 10 --target-spec target_spec.json
        """
    )

    parser.add_argument(
        '--target-spec', type=str, default='target_spec.json',
        help='Path to target_spec.json (default: target_spec.json)'
    )
    parser.add_argument(
        '--output', type=str, default='results.json',
        help='Path to output results.json (default: results.json)'
    )
    parser.add_argument(
        '--kernel', type=str, default=None,
        help='Path to CUDA kernel binary to analyze (overrides "run" in spec)'
    )
    parser.add_argument(
        '--kernel-name', type=str, default=None,
        help='Specific kernel function name to profile (optional)'
    )
    parser.add_argument(
        '--probe-dir', type=str, default='probes',
        help='Directory containing CUDA probe source files (default: probes)'
    )
    parser.add_argument(
        '--build-dir', type=str, default='build',
        help='Directory for compiled probe binaries (default: build)'
    )
    parser.add_argument(
        '--trials', type=int, default=5,
        help='Number of repeated trials per measurement (default: 5)'
    )

    args = parser.parse_args()

    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.probe_dir):
        args.probe_dir = os.path.join(script_dir, args.probe_dir)
    if not os.path.isabs(args.build_dir):
        args.build_dir = os.path.join(script_dir, args.build_dir)
    if not os.path.isabs(args.target_spec):
        args.target_spec = os.path.join(script_dir, args.target_spec)
    if not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)

    logger.info("=" * 60)
    logger.info("GPU Hardware Intrinsic Profiling Agent")
    logger.info("=" * 60)
    logger.info(f"Probe directory: {args.probe_dir}")
    logger.info(f"Build directory: {args.build_dir}")
    logger.info(f"Target spec: {args.target_spec}")
    logger.info(f"Output: {args.output}")

    # Initialize reasoning engine
    reasoning = ReasoningEngine()

    # Detect environment
    reasoning.log_step('startup', 'Agent starting - detecting environment')
    gpu_info = utils.get_gpu_info()
    reasoning.log_step('startup', 'GPU environment detected', data=gpu_info)

    # ------------------------------------------------------------------ #
    # Load the evaluation spec — this single file drives the entire run   #
    # ------------------------------------------------------------------ #
    target_spec = {}
    if os.path.exists(args.target_spec):
        target_spec = load_target_spec(args.target_spec)
        reasoning.log_step(
            'initialization',
            f'Loaded target_spec.json: keys={list(target_spec.keys())}',
            data=target_spec,
        )

    hw_targets = target_spec.get('targets', [])
    # The "run" field tells us which executable to profile for Sections 1.1-1.6
    run_binary = target_spec.get('run', None)
    # CLI --kernel overrides the spec's "run" field
    kernel_binary = args.kernel or run_binary

    if kernel_binary:
        # Resolve relative paths against the spec-file directory first,
        # then against the script directory.
        if not os.path.isabs(kernel_binary):
            spec_dir = os.path.dirname(args.target_spec)
            candidate = os.path.join(spec_dir, kernel_binary)
            if os.path.exists(candidate):
                kernel_binary = candidate
            else:
                kernel_binary = os.path.join(script_dir, kernel_binary)

    logger.info(f"Hardware targets: {hw_targets}")
    logger.info(f"Kernel to profile: {kernel_binary or '(none)'}")

    all_results = {}

    # ------------------------------------------------------------------ #
    # Phase 1: Hardware Probing (if any hardware targets requested)       #
    # ------------------------------------------------------------------ #
    if hw_targets:
        hw_results = run_hardware_probing(hw_targets, args, reasoning)
        all_results.update(hw_results)

    # ------------------------------------------------------------------ #
    # Phase 2: Kernel Analysis (from spec "run" or CLI --kernel)          #
    # ------------------------------------------------------------------ #
    if kernel_binary:
        kernel_analysis = run_kernel_analysis(
            kernel_binary, args, reasoning,
            kernel_name=args.kernel_name,
        )
        # Save kernel analysis to a separate JSON file
        analysis_path = args.output.replace('.json', '_kernel_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(kernel_analysis, f, indent=2, default=str)
        logger.info(f"Kernel analysis saved to {analysis_path}")

        # Include summary in main results
        summary = kernel_analysis.get('summary', {})
        all_results['_kernel_analysis'] = summary

        # Save the LLM report as a markdown file
        llm_report = kernel_analysis.get('llm_report', '')
        if llm_report:
            report_path = args.output.replace('.json', '_kernel_report.md')
            with open(report_path, 'w') as f:
                f.write(llm_report)
            logger.info(f"Kernel bottleneck report saved to {report_path}")

    # Write final results
    output_data = {}

    # Add numeric results (what the evaluator checks)
    for key, value in all_results.items():
        if value is not None:
            # Round to reasonable precision
            if isinstance(value, float):
                if 'cycles' in key:
                    output_data[key] = round(value, 1)
                elif 'mhz' in key:
                    output_data[key] = round(value, 1)
                elif 'gbps' in key:
                    output_data[key] = round(value, 2)
                elif 'kb' in key or 'mb' in key:
                    output_data[key] = round(value, 1)
                else:
                    output_data[key] = round(value, 2)
            else:
                output_data[key] = value

    # Add reasoning metadata (for Engineering Reasoning scoring)
    # LLM-authored narratives go into the top-level fields the grader reads
    reasoning_text = reasoning.get_reasoning_text()
    methodology_text = reasoning.get_methodology_text()

    output_data['_reasoning'] = reasoning_text if reasoning_text else reasoning.get_summary()
    output_data['_methodology'] = methodology_text if methodology_text else reasoning.get_methodology()

    # Detailed evidence log for LLM-as-a-Judge evaluation
    summary = reasoning.get_summary()
    output_data['_log'] = {
        'steps': summary.get('steps', []),
        'anomalies': summary.get('anomalies', []),
        'cross_verifications': summary.get('cross_verifications', []),
        'methodology_records': reasoning.get_methodology(),
        'elapsed_seconds': summary.get('elapsed_seconds', 0),
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Results written to {args.output}")

    # Save detailed reasoning log
    reasoning_log_path = os.path.join(
        os.path.dirname(args.output), 'reasoning.log'
    )
    reasoning.save_log(reasoning_log_path)

    # Print human-readable report
    report = reasoning.format_report()
    print("\n" + report)

    # Print final results summary
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    for key, value in output_data.items():
        if not key.startswith('_'):
            logger.info(f"  {key}: {value}")

    logger.info("=" * 60)
    logger.info("Agent finished successfully.")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
