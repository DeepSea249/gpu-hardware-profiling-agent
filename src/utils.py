"""
Utility functions for the GPU profiling agent.

Provides nvidia-smi queries, statistical helpers, and
environment detection.
"""

import subprocess
import logging
import re
import os

logger = logging.getLogger('GPUAgent.Utils')


def query_nvidia_smi(field: str) -> str:
    """Query a single nvidia-smi field."""
    try:
        result = subprocess.run(
            ['nvidia-smi', f'--query-gpu={field}', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0].strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_nvidia_smi_clocks() -> dict:
    """Get current GPU clocks from nvidia-smi."""
    info = {}
    fields = {
        'clocks.current.sm': 'current_sm_clock_mhz',
        'clocks.current.memory': 'current_mem_clock_mhz',
        'clocks.max.sm': 'max_sm_clock_mhz',
        'clocks.max.memory': 'max_mem_clock_mhz',
    }
    for field, key in fields.items():
        val = query_nvidia_smi(field)
        if val:
            try:
                info[key] = float(val)
            except ValueError:
                info[key] = val
    return info


def get_gpu_info() -> dict:
    """Get comprehensive GPU information from nvidia-smi."""
    info = {}
    fields = {
        'name': 'gpu_name',
        'compute_cap': 'compute_capability',
        'memory.total': 'total_memory_mib',
        'memory.free': 'free_memory_mib',
        'clocks.current.sm': 'current_sm_clock_mhz',
        'clocks.current.memory': 'current_mem_clock_mhz',
        'clocks.max.sm': 'max_sm_clock_mhz',
        'clocks.max.memory': 'max_mem_clock_mhz',
        'temperature.gpu': 'temperature_c',
        'power.draw': 'power_draw_w',
        'power.limit': 'power_limit_w',
        'persistence_mode': 'persistence_mode',
        'memory.bus.width': 'memory_bus_width_bits',
    }

    for field, key in fields.items():
        val = query_nvidia_smi(field)
        if val:
            try:
                info[key] = float(val)
            except ValueError:
                info[key] = val

    return info


def median(values: list) -> float:
    """Compute the median of a list of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    return sorted_vals[n // 2]


def trimmed_mean(values: list, trim_fraction: float = 0.2) -> float:
    """Compute trimmed mean (remove outliers from both ends)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    trim_count = max(1, int(n * trim_fraction))
    if 2 * trim_count >= n:
        return median(values)
    trimmed = sorted_vals[trim_count : n - trim_count]
    return sum(trimmed) / len(trimmed) if trimmed else median(values)


def check_cuda_env() -> dict:
    """Check CUDA-related environment variables that might affect execution."""
    relevant_vars = [
        'CUDA_VISIBLE_DEVICES',
        'CUDA_DEVICE_ORDER',
        'CUDA_MPS_PIPE_DIRECTORY',
        'CUDA_MPS_LOG_DIRECTORY',
        'CUDA_DEVICE_MAX_CONNECTIONS',
        'CUDA_LAUNCH_BLOCKING',
    ]

    env_info = {}
    for var in relevant_vars:
        val = os.environ.get(var)
        if val is not None:
            env_info[var] = val
            logger.info(f"Environment: {var}={val}")

    return env_info
