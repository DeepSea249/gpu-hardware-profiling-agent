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
  2. Compiles and runs CUDA micro-benchmark probes
  3. Parses raw output and applies statistical analysis
  4. Cross-verifies results using multiple methods (probes, ncu, nvidia-smi)
  5. Detects anomalies (frequency locking, SM masking)
  6. Outputs results.json with measurements and reasoning log

Usage:
  python3 agent.py [--target-spec target_spec.json] [--output results.json]
  python3 agent.py --kernel ./my_cuda_binary  # For kernel analysis
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


def run_hardware_probing(args, reasoning: ReasoningEngine) -> dict:
    """
    Phase 1: Hardware Intrinsic Probing

    Reads target_spec.json, runs micro-benchmarks, analyzes results,
    and outputs results.json.
    """
    logger.info("=" * 60)
    logger.info("Phase 1: Hardware Intrinsic Profiling")
    logger.info("=" * 60)

    # Load target specification
    if not os.path.exists(args.target_spec):
        logger.error(f"Target spec not found: {args.target_spec}")
        return {}

    with open(args.target_spec, 'r') as f:
        target_spec = json.load(f)

    targets = target_spec.get('targets', [])
    if not targets:
        logger.warning("No targets specified in target_spec.json")
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


def run_kernel_analysis(args, reasoning: ReasoningEngine) -> dict:
    """
    Phase 2: Kernel Performance Analysis

    Uses ncu to profile a CUDA kernel and diagnose bottlenecks
    following the Section 1 analysis methodology.
    """
    logger.info("=" * 60)
    logger.info("Phase 2: Kernel Bottleneck Analysis")
    logger.info("=" * 60)

    analyzer = KernelAnalyzer(reasoning=reasoning)
    analysis = analyzer.analyze(args.kernel)

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description='GPU Hardware Intrinsic Profiling Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Probe hardware metrics specified in target_spec.json
  python3 agent.py --target-spec target_spec.json --output results.json

  # Analyze a specific CUDA kernel for bottlenecks
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
        help='Path to CUDA kernel binary to analyze for bottlenecks'
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

    all_results = {}

    # Phase 1: Hardware Probing (if target_spec exists)
    if os.path.exists(args.target_spec):
        hw_results = run_hardware_probing(args, reasoning)
        all_results.update(hw_results)

    # Phase 2: Kernel Analysis (if kernel binary specified)
    if args.kernel:
        kernel_analysis = run_kernel_analysis(args, reasoning)
        # Save kernel analysis separately
        analysis_path = args.output.replace('.json', '_kernel_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(kernel_analysis, f, indent=2)
        logger.info(f"Kernel analysis saved to {analysis_path}")

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
    output_data['_reasoning'] = reasoning.get_summary()
    output_data['_methodology'] = reasoning.get_methodology()

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
