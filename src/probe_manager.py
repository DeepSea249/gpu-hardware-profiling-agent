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
import platform

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
        self._ensure_cl_in_path()
        self._compiled = {}
        from .probe_codegen import ProbeCodeGenerator
        self._codegen = ProbeCodeGenerator(build_dir=self.build_dir)

    def _ensure_cl_in_path(self):
        """On Windows, ensure cl.exe (MSVC) is on PATH for nvcc host compiler."""
        if platform.system() != 'Windows':
            return
        if shutil.which('cl'):
            return  # Already available

        # Try vswhere to locate the latest MSVC toolset.
        # Use environment variables instead of hardcoded drive/path so the code
        # works regardless of Windows installation drive or locale.
        prog_x86 = os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')
        prog_pf  = os.environ.get('ProgramFiles',       r'C:\Program Files')
        vswhere_candidates = [
            os.path.join(prog_x86, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe'),
            os.path.join(prog_pf,  'Microsoft Visual Studio', 'Installer', 'vswhere.exe'),
            shutil.which('vswhere') or '',   # PATH fallback (e.g. Chocolatey install)
        ]
        vswhere = next(
            (c for c in vswhere_candidates if c and os.path.exists(c)), None
        )
        if not vswhere:
            logger.warning('vswhere.exe not found; cl.exe may not be on PATH')
            return

        try:
            result = subprocess.run(
                [vswhere, '-latest', '-property', 'installationPath'],
                capture_output=True, text=True, timeout=10
            )
            vs_path = result.stdout.strip()
        except Exception:
            logger.warning('vswhere failed to find VS installation')
            return

        if not vs_path or not os.path.isdir(vs_path):
            logger.warning('No Visual Studio installation found via vswhere')
            return

        msvc_root = os.path.join(vs_path, 'VC', 'Tools', 'MSVC')
        if not os.path.isdir(msvc_root):
            logger.warning(f'MSVC tools not found at {msvc_root}')
            return

        # Pick the latest toolset version
        versions = sorted(os.listdir(msvc_root), reverse=True)
        if not versions:
            return

        cl_dir = os.path.join(msvc_root, versions[0], 'bin', 'Hostx64', 'x64')
        if os.path.exists(os.path.join(cl_dir, 'cl.exe')):
            os.environ['PATH'] = cl_dir + os.pathsep + os.environ.get('PATH', '')
            logger.info(f'Added MSVC cl.exe to PATH: {cl_dir}')
        else:
            logger.warning(f'cl.exe not found at expected path: {cl_dir}')

    def _find_nvcc(self) -> str:
        """Locate the nvcc compiler in a cross-platform, portable way."""
        # 1. Honour CUDA_PATH / CUDA_HOME / CUDA_ROOT (set by installers on all OSes)
        for env_var in ('CUDA_PATH', 'CUDA_HOME', 'CUDA_ROOT'):
            cuda_root = os.environ.get(env_var)
            if not cuda_root:
                continue
            candidate = os.path.join(cuda_root, 'bin', 'nvcc')
            if os.path.exists(candidate):
                return candidate
            if os.path.exists(candidate + '.exe'):   # Windows
                return candidate + '.exe'

        # 2. Common Linux / macOS symlink paths
        if platform.system() != 'Windows':
            for path in (
                '/usr/local/cuda/bin/nvcc',
                '/usr/local/cuda-12/bin/nvcc',
                '/usr/local/cuda-11/bin/nvcc',
            ):
                if os.path.exists(path):
                    return path

        # 3. Windows: enumerate installed CUDA Toolkit versions
        if platform.system() == 'Windows':
            cuda_base = os.path.join(
                os.environ.get('ProgramFiles', r'C:\Program Files'),
                'NVIDIA GPU Computing Toolkit', 'CUDA',
            )
            if os.path.isdir(cuda_base):
                # Pick the newest version (sorted descending)
                for ver in sorted(os.listdir(cuda_base), reverse=True):
                    nvcc_path = os.path.join(cuda_base, ver, 'bin', 'nvcc.exe')
                    if os.path.exists(nvcc_path):
                        return nvcc_path

        # 4. Rely on PATH as last resort (works when CUDA bin dir is on PATH)
        nvcc = shutil.which('nvcc')
        if nvcc:
            return nvcc

        raise RuntimeError(
            "nvcc not found. Install the CUDA Toolkit and ensure its bin/ "
            "directory is on PATH, or set the CUDA_PATH environment variable."
        )

    def compile(self, probe_name: str, extra_flags: list = None,
                source_path: str = None) -> str:
        """
        Compile a CUDA probe.

        If source_path is provided, that file is compiled directly (no LLM).
        Otherwise the ProbeCodeGenerator autonomously generates the CUDA C++
        source from a design specification via the LLM, then compiles it.
        On compilation failure the LLM is asked to fix the errors and the
        process retries up to ProbeCodeGenerator.MAX_RETRIES times.

        Args:
            probe_name:  Name of the probe (without extension).
            extra_flags: Additional nvcc flags.
            source_path: Explicit path to a .cu file; bypasses code generation.

        Returns:
            Absolute path to the compiled binary.
        """
        if probe_name in self._compiled:
            binary_path = self._compiled[probe_name]
            if os.path.exists(binary_path):
                return binary_path

        # ── Resolve source ────────────────────────────────────────────────── #
        if source_path is not None:
            src_path = source_path
            max_attempts = 1          # External source: no LLM retry
        else:
            from .probe_codegen import ProbeCodeGenerator
            src_path = self._codegen.get_source_path(probe_name)
            max_attempts = ProbeCodeGenerator.MAX_RETRIES

        binary_path = os.path.join(self.build_dir, probe_name)
        flags = ['-O3', '-lineinfo']
        if platform.system() == 'Windows':
            flags.append('-allow-unsupported-compiler')
        if extra_flags:
            flags.extend(extra_flags)

        # ── Compile with optional LLM-powered fix-and-retry ───────────────── #
        for attempt in range(max_attempts):
            success, last_error = self._compile_with_arch_fallback(
                probe_name, src_path, binary_path, flags
            )
            if success:
                # On Windows nvcc appends .exe automatically
                if platform.system() == 'Windows' and not os.path.exists(binary_path):
                    if os.path.exists(binary_path + '.exe'):
                        binary_path = binary_path + '.exe'
                self._compiled[probe_name] = binary_path
                logger.info("Compiled '%s' successfully", probe_name)
                return binary_path

            # Compilation failed — ask the LLM to fix and retry
            if source_path is None and attempt < max_attempts - 1:
                logger.warning(
                    "Compilation attempt %d/%d failed for '%s'; "
                    "asking LLM to fix the error...",
                    attempt + 1, max_attempts, probe_name,
                )
                src_path = self._codegen.regenerate_with_error(probe_name, last_error)
            else:
                raise RuntimeError(
                    f"Failed to compile '{probe_name}' "
                    f"(attempt {attempt + 1}/{max_attempts}).\n"
                    f"Last error:\n{last_error}"
                )

        raise RuntimeError(
            f"Failed to compile '{probe_name}' after {max_attempts} attempts"
        )

    def _compile_with_arch_fallback(
        self, probe_name: str, src_path: str, binary_path: str, flags: list
    ):
        """Try to compile src_path with multiple architecture fallbacks.

        Returns (success: bool, last_error: str).
        """
        last_error = ""
        for arch in self.ARCH_FALLBACKS:
            arch_flag = f'-arch={arch}'
            cmd = [self._nvcc_path] + flags + [arch_flag, '-o', binary_path, src_path]
            logger.info("Compiling %s: %s", probe_name, ' '.join(cmd))

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                logger.info("Successfully compiled %s with %s", probe_name, arch_flag)
                return True, ""
            else:
                last_error = result.stderr
                logger.debug(
                    "Compilation failed with %s: %s", arch_flag, result.stderr[:200]
                )
        return False, last_error

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
                         extra_flags: list = None, timeout: int = 180,
                         source_path: str = None) -> str:
        """Compile (if needed) and run a probe in one call."""
        self.compile(probe_name, extra_flags, source_path=source_path)
        return self.run(probe_name, args, timeout)

    def get_nvcc_version(self) -> str:
        """Get nvcc version string."""
        result = subprocess.run(
            [self._nvcc_path, '--version'],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
