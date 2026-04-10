"""
Probe Manager - Compiles and executes CUDA micro-benchmark probes.

Handles:
- CUDA source compilation with nvcc (auto-detect architecture)
- Binary execution with argument passing
- Output capture and error handling
- Fallback compilation strategies for different GPU architectures
"""

import os
import subprocess
import shutil
import logging
import re

logger = logging.getLogger('GPUAgent.ProbeManager')


class ProbeManager:
    """Manages compilation and execution of CUDA micro-benchmark probes."""

    # Compilation architectures to try in order of preference (newest first)
    ARCH_FALLBACKS = [
        'native',  # CUDA 11.1+
        'sm_90', 'sm_89', 'sm_86', 'sm_80', 'sm_75', 'sm_70', 'sm_61', 'sm_60'
    ]

    def __init__(self, probe_dir: str, build_dir: str):
        self.probe_dir = os.path.abspath(probe_dir)
        self.build_dir = os.path.abspath(build_dir)
        os.makedirs(self.build_dir, exist_ok=True)
        self._nvcc_path = self._find_nvcc()
        self._compiled = {}

    def _find_nvcc(self) -> str:
        """Locate the nvcc compiler."""
        # Check common CUDA installation paths first
        common_paths = [
            '/usr/local/cuda/bin/nvcc',
            '/usr/local/cuda-12.4/bin/nvcc',
            '/usr/local/cuda-12/bin/nvcc',
            '/usr/local/cuda-11/bin/nvcc',
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path

        nvcc = shutil.which('nvcc')
        if nvcc:
            return nvcc

        raise RuntimeError(
            "nvcc not found. Please install CUDA toolkit or set PATH."
        )

    def compile(self, probe_name: str, extra_flags: list = None) -> str:
        """
        Compile a CUDA probe source file.

        Args:
            probe_name: Name of the probe (without .cu extension)
            extra_flags: Additional nvcc flags

        Returns:
            Path to the compiled binary
        """
        if probe_name in self._compiled:
            binary_path = self._compiled[probe_name]
            if os.path.exists(binary_path):
                return binary_path

        src_path = os.path.join(self.probe_dir, f"{probe_name}.cu")
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Probe source not found: {src_path}")

        binary_path = os.path.join(self.build_dir, probe_name)
        flags = ['-O3', '-lineinfo']
        if extra_flags:
            flags.extend(extra_flags)

        # Try architectures in order until one succeeds
        last_error = None
        for arch in self.ARCH_FALLBACKS:
            arch_flag = f'-arch={arch}'
            cmd = [self._nvcc_path] + flags + [arch_flag, '-o', binary_path, src_path]
            logger.info(f"Compiling {probe_name}: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                logger.info(f"Successfully compiled {probe_name} with {arch_flag}")
                self._compiled[probe_name] = binary_path
                return binary_path
            else:
                last_error = result.stderr
                logger.debug(f"Compilation failed with {arch_flag}: {result.stderr[:200]}")

        raise RuntimeError(
            f"Failed to compile {probe_name} with any architecture.\n"
            f"Last error: {last_error}"
        )

    def run(self, probe_name: str, args: list = None, timeout: int = 180) -> str:
        """
        Run a compiled probe and return its stdout output.

        Args:
            probe_name: Name of the probe
            args: Command line arguments to pass
            timeout: Execution timeout in seconds

        Returns:
            stdout output as string
        """
        # Ensure it's compiled
        if probe_name not in self._compiled:
            self.compile(probe_name)

        binary_path = self._compiled[probe_name]
        cmd = [binary_path]
        if args:
            cmd.extend([str(a) for a in args])

        logger.info(f"Running probe: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Probe {probe_name} timed out after {timeout}s"
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"Probe {probe_name} failed (exit code {result.returncode}):\n"
                f"stderr: {result.stderr}\n"
                f"stdout: {result.stdout[:500]}"
            )

        logger.info(f"Probe {probe_name} completed successfully ({len(result.stdout)} bytes output)")
        return result.stdout

    def compile_and_run(self, probe_name: str, args: list = None,
                         extra_flags: list = None, timeout: int = 180) -> str:
        """Compile (if needed) and run a probe in one call."""
        self.compile(probe_name, extra_flags)
        return self.run(probe_name, args, timeout)

    def get_nvcc_version(self) -> str:
        """Get nvcc version string."""
        result = subprocess.run(
            [self._nvcc_path, '--version'],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
