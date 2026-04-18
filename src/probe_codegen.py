"""
Probe Code Generator
====================

Autonomous CUDA C++ source generation via the LLM.

Each entry in PROBE_SPECS contains only a *description* and *algorithm*
specification — design rationale, required output format, and measurement
approach.  No CUDA source code is included in the specs.  The LLM reads
the specification and writes a complete, compilable CUDA C++ file, which
is then compiled by ProbeManager.

Flow
----
    ProbeCodeGenerator.get_source_path(name)
      └─ if build/<name>_generated.cu already on disk → reuse it
         else → call LLM with spec → strip fences → write file

    ProbeManager.compile(name)          [in probe_manager.py]
      └─ get_source_path(name)
           └─ try _compile_with_arch_fallback()
                  on failure → regenerate_with_error(name, err) → retry
"""

import hashlib
import os
import re
import logging
from typing import Optional

logger = logging.getLogger('GPUAgent.ProbeCodegen')


def _spec_hash(probe_name: str) -> str:
    """Return an 8-char hash of the probe's spec for cache-invalidation."""
    spec = PROBE_SPECS.get(probe_name, {})
    return hashlib.md5(str(spec).encode()).hexdigest()[:8]


_HASH_PREFIX = '// SPEC_HASH:'


# ── Lazy LLM client ───────────────────────────────────────────────────────────
_llm_client: Optional[object] = None


def _get_llm():
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    try:
        from .llm_client import LLMClient
        # Code generation does not need thinking mode — specs are detailed enough
        # and non-thinking is ~10× faster.
        _llm_client = LLMClient(enable_thinking=False)
        logger.info("LLMClient (codegen) initialised")
    except Exception as exc:
        logger.warning("LLMClient unavailable for code generation: %s", exc)
        _llm_client = None
    return _llm_client


# ── Utility ───────────────────────────────────────────────────────────────────

def _strip_code_fences(text: str) -> str:
    """Remove markdown ``` / ~~~ fences that LLMs sometimes wrap code in."""
    text = text.strip()
    for fence in ('```', '~~~'):
        # Opening fence with optional language tag (c, cpp, cuda, c++)
        text = re.sub(r'^' + re.escape(fence) + r'[a-zA-Z+]*\s*\n',
                      '', text, count=1)
        # Closing fence at end of text
        text = re.sub(r'\n' + re.escape(fence) + r'\s*$',
                      '', text, count=1)
    return text.strip()


# ── Design specifications ─────────────────────────────────────────────────────
# These contain the ALGORITHM and the mandatory OUTPUT FORMAT.
# There is NO CUDA source code here — the LLM generates the code.

PROBE_SPECS: dict = {

    # ---------------------------------------------------------------------- #
    "clock_probe": {
        "description": "GPU SM clock frequency and active SM count probe",
        "algorithm": r"""
=== TASK ===
Write a standalone CUDA C++ program that measures the actual GPU SM boost-clock
frequency and the number of active SMs.

=== REQUIRED OUTPUT (exact key names, one key=value per line) ===
CLOCK_PROBE_START
NUM_ACTIVE_SMS=<integer>
SM_ID_MAX=<integer> SM_ID_UNIQUE=<integer>
TRIAL=<n> CYCLES=<lld> ELAPSED_MS=<f4> CLOCK_MHZ=<f2>    (repeat 5 times)
CLOCK_MHZ=<f2>                 (final median)
REPORTED_CLOCK_KHZ=<d>
REPORTED_MEM_CLOCK_KHZ=<d>
REPORTED_SM_COUNT=<d>
REPORTED_DEVICE_NAME=<string>
REPORTED_COMPUTE_CAP=<d>.<d>
MEMORY_BUS_WIDTH_BITS=<d>
CLOCK_DEVIATION_PCT=<f2>
CLOCK_PROBE_END

=== ALGORITHM ===
Step 1 – SM count (one-off kernel):
  Allocate d_sm_ids[4096] (int) and d_counter (int, zeroed).
  Kernel sm_detect_kernel<<<4096,32>>>:
    if (threadIdx.x == 0):
      int smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
      int idx = atomicAdd(counter, 1);  sm_ids[idx] = smid.
  Sync; copy back min(count, 4096) values; scan for unique IDs.
  Print NUM_ACTIVE_SMS and SM_ID_MAX / SM_ID_UNIQUE.

Step 2 – Clock frequency measurement (5 trials, 1 block × 1 thread):
  Kernel sustained_compute_kernel(float* out, long long* cycles_out, int iters):
    float x=1.0f, a=1.0000001f, b=0.0000001f;
    warmup: 100 000 dependent fmaf ops (not timed).
    long long start = clock64();
    #pragma unroll 1
    timed loop: iters=50 000 000 dependent fmaf ops.
    long long end = clock64();
    if (threadIdx.x==0) *cycles_out = end - start;
    out[threadIdx.x] = x;   // prevent DCE
  Wrap each trial in cudaEventRecord/cudaEventSynchronize.
  clock_mhz = cycles / (elapsed_ms * 1000.0).
  After 5 trials: sort array, median = clock_measurements[num_trials/2].

Step 3 – Device properties:
  cudaGetDeviceProperties(&prop, 0); print REPORTED_* lines.
  Print MEMORY_BUS_WIDTH_BITS=prop.memoryBusWidth.
  deviation = 100.0 * (median_clock - prop.clockRate/1000.0) / (prop.clockRate/1000.0).

=== IMPLEMENTATION ===
- #define CUDA_CHECK(call) macro; exit on CUDA error with file/line.
- #include <stdio.h>, <stdlib.h>, <cuda_runtime.h>.
- argc/argv: optional argv[1]=iters, argv[2]=num_trials.
""",
    },

    # ---------------------------------------------------------------------- #
    "bandwidth_probe": {
        "description": "Global memory and shared memory peak bandwidth probe",
        "algorithm": r"""
=== TASK ===
Write a standalone CUDA C++ program that measures peak GPU memory bandwidth.

=== REQUIRED OUTPUT ===
ACTIVE_SM_COUNT=<integer>
GLOBAL_READ_BW_GBPS=<f2> SIZE_MB=<integer>   (4 lines: 64, 128, 256, 512 MB)
BEST_GLOBAL_READ_BW_GBPS=<f2>
GLOBAL_WRITE_BW_GBPS=<f2> SIZE_MB=<integer>  (4 lines)
BEST_GLOBAL_WRITE_BW_GBPS=<f2>
GLOBAL_COPY_BW_GBPS=<f2>                     (4 lines: read+write copy)
BEST_GLOBAL_COPY_BW_GBPS=<f2>
SHMEM_BW_GBPS_PER_SM=<f2>
SHMEM_BW_GBPS_AGGREGATE=<f2>

=== ALGORITHM ===
Active-SM detection: same PTX smid trick as clock_probe (smid register, atomicAdd,
4096 blocks × 32 threads) → ACTIVE_SM_COUNT.  Store num_sms for later.

Global read kernel (coalesced float4 loads):
  Each thread: stride-1 float4 reads over the full array; accumulate sum.
  Write sum.x to out[blockIdx.x] to prevent DCE.
  Launch config: num_sms*512 blocks, 256 threads.
  GB/s = (array_bytes) / (elapsed_s * 1e9).  Elapsed via cudaEventElapsedTime.

Global write kernel: stride-1 float4 stores of a constant value.
  Launch same config.

Global copy kernel: float4 src → dst (read+write counts as 2× bytes).
  GB/s = 2 * array_bytes / (elapsed_s * 1e9).

Array sizes: 64, 128, 256, 512 MB.  Allocate only the largest once and reuse.
Track maximum across sizes for BEST_* lines.

Shared-memory bandwidth kernel:
  One block per SM (gridDim.x = num_sms), 1024 threads, 32 KB shared (float[8192]).
  REPEAT_ITERS = 10000.  Use cudaEventRecord/cudaEventSynchronize for timing.

  CRITICAL — prevent compiler loop hoisting / dead-code elimination:
    (a) The index into shm MUST include the outer-loop variable j so the compiler
        cannot prove consecutive outer iterations access the same address.
        Formula: int idx = (tid * 8 + i + j) & 8191;
    (b) After the outer loop, write acc to out[] immediately (no branch guard).
  Without (a) the compiler collapses all 10000 iterations into one, inflating BW
  by 10000×.

  Kernel body:
    __shared__ float shm[8192];
    int tid = threadIdx.x;
    // Initialise
    for (int k = tid; k < 8192; k += 1024) shm[k] = (float)k;
    __syncthreads();
    float acc = 0.0f;
    for (int j = 0; j < REPEAT_ITERS; j++) {
        for (int i = 0; i < 8; i++) {
            int idx = (tid * 8 + i + j) & 8191;   // j-dependent: no hoisting
            acc += shm[idx];
        }
    }
    out[blockIdx.x * blockDim.x + tid] = acc;   // write to global (no branch)

  Bytes transferred per SM per kernel call = 1024 * 8 * 4 * REPEAT_ITERS.
  SHMEM_BW_GBPS_PER_SM = bytes_per_sm / (elapsed_s * 1e9).
  SHMEM_BW_GBPS_AGGREGATE = per_sm * num_sms.

=== IMPLEMENTATION ===
- CUDA_CHECK macro; #include <stdio.h>, <stdlib.h>, <cuda_runtime.h>.
- cudaFree all device pointers before exit.
""",
    },

    # ---------------------------------------------------------------------- #
    "latency_probe": {
        "description": "Memory hierarchy latency probe via pointer chasing",
        "algorithm": r"""
=== TASK ===
Write a standalone CUDA C++ program that measures access latency at 11 array
sizes to reveal the L1/L2/DRAM cache hierarchy.

=== REQUIRED OUTPUT (one line per size tested) ===
SIZE_BYTES=<n> AVG_CYCLES=<f2> MEDIAN_CYCLES=<f2> TRIMMED_MEAN=<f2>

=== ARRAY SIZES (bytes) ===
1024, 2048, 4096, 8192, 16384, 32768, 131072, 1048576, 8388608,
16777216, 33554432, 67108864, 268435456
(13 sizes total; the 16 MB and 32 MB points help bracket the L2→DRAM boundary)

=== SHUFFLE CONSTRUCTION (CRITICAL — Sattolo's algorithm) ===
  You MUST use Sattolo's algorithm, NOT Fisher-Yates.  Fisher-Yates can
  produce many short disjoint cycles; following one short cycle never probes
  the intended working set and yields non-monotonic, wrong latency curves.
  Sattolo guarantees the permutation is a SINGLE cycle of length N, so
  following arr[idx] visits every element before returning to 0.

  void sattolo_shuffle(uint64_t* arr, size_t n) {
      for (size_t i = 0; i < n; i++) arr[i] = (uint64_t)i;
      // i goes from n-1 DOWN to 1; j drawn from [0, i)  (NOT [0,i])
      for (size_t i = n - 1; i > 0; i--) {
          size_t j = (size_t)rand() % i;   // j < i  ← SATTOLO requirement
          uint64_t tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
      }
  }
  Call srand(42) once before all shuffles.

=== ITERATIONS (proportional to array size) ===
  CHASE_ITERS for each size must be large enough to stress the correct
  cache level.  Use this formula:
      size_t N = size_bytes / sizeof(uint64_t);
      size_t iters = N * 3;             // 3 full cycle traversals
      if (iters < 50000)  iters = 50000;    // minimum for timing accuracy
      if (iters > 5000000) iters = 5000000; // cap to keep runtime sane

  (Example: 268 MB → N=33 554 432 → iters=5 000 000 (capped) — enough to
   overflow even a 64 MB L2;  32 KB → N=4096 → iters=50 000 (floored).)

=== TIMING KERNEL — EXACT signature required ===
  __global__ void chase_kernel(const uint64_t* arr,
                                float* timing_out, uint64_t* idx_sink,
                                size_t iters)
  Body:
    uint64_t idx = 0;
    long long t0 = clock64();
    for (size_t i = 0; i < iters; i++) {
        asm volatile("" ::: "memory");   // prevent hardware prefetching
        idx = arr[idx];   // *** NO MODULO — arr is a Sattolo permutation,
                          //     all indices are in [0,N-1] by construction.
                          //     Adding % n or any arithmetic would add
                          //     hundreds of cycles of false overhead. ***
    }
    long long t1 = clock64();
    idx_sink[0] = idx;   // *** MANDATORY: write idx to global memory to
                         //     prevent the compiler from eliminating the
                         //     chase loop as dead code.  Without this every
                         //     output will be 0.00. ***
    timing_out[0] = (float)((double)(t1 - t0) / (double)iters);

  Launch: <<<1, 1>>>       (single thread — measures latency, not bandwidth)

=== PER-SIZE LOOP (5 trials) ===
  Allocate d_arr (uint64_t, size_bytes), d_timing (float, 1), d_sink (uint64_t, 1).
  cudaMemcpy h_array → d_arr; for each trial:
    cudaMemset(d_timing, 0, sizeof(float)); cudaMemset(d_sink, 0, sizeof(uint64_t));
    launch chase_kernel<<<1,1>>>(d_arr, d_timing, d_sink, iters);
    cudaDeviceSynchronize(); cudaMemcpy d_timing → h_timing (float).
  Compute: average, median (sort 5 values; result = sorted[2]),
           trimmed mean (drop sorted[0] and sorted[4]; mean of sorted[1..3]).
  Print: SIZE_BYTES=<n> AVG_CYCLES=<f.2f> MEDIAN_CYCLES=<f.2f> TRIMMED_MEAN=<f.2f>
  Free d_arr, d_timing, d_sink.

Skip sizes where size_bytes > 1 073 741 824 (1 GB).

=== IMPLEMENTATION ===
- CUDA_CHECK macro; #include <stdio.h>, <stdlib.h>, <cuda_runtime.h>.
- Use size_t for all size/count variables; cast correctly when calling kernels.
""",
    },

    # ---------------------------------------------------------------------- #
    "shmem_limit_probe": {
        "description": "Maximum dynamic shared memory per block probe (default + opt-in)",
        "algorithm": r"""
=== TASK ===
Write a standalone CUDA C++ program that determines the maximum dynamic
shared memory per block: first the default soft limit, then the opt-in
hardware maximum available on sm_70+ (Ada, Hopper, Ampere, etc.).

=== REQUIRED OUTPUT ===
SHMEM_LIMIT_PROBE_START
DEFAULT_SHMEM_LIMIT_BYTES=<n>
DEFAULT_SHMEM_LIMIT_KB=<n>
EXTENDED_SHMEM_LIMIT_BYTES=<n>
EXTENDED_SHMEM_LIMIT_KB=<n>
MAX_SHMEM_PER_BLOCK_BYTES=<n>
MAX_SHMEM_PER_BLOCK_KB=<n>
REPORTED_SHMEM_PER_BLOCK=<n>
REPORTED_SHMEM_PER_SM=<n>
REPORTED_SHMEM_PER_BLOCK_OPTIN=<n>
SHMEM_LIMIT_PROBE_END

=== ALGORITHM ===
Test kernel:
  __global__ void shmem_test_kernel(int* output):
    extern __shared__ char smem[];
    smem[threadIdx.x % 1024] = (char)(threadIdx.x & 0xFF);
    __syncthreads();
    if (threadIdx.x == 0) output[0] = (int)smem[0];

Helper bool try_shmem_size(int bytes, int* d_output, bool use_extended):
  if (use_extended):
    cudaFuncSetAttribute(shmem_test_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    If that returns error: cudaGetLastError(); return false.
  Launch shmem_test_kernel<<<1,32,bytes>>>(d_output).
  Check cudaGetLastError(). Call cudaDeviceSynchronize(). Check error.
  Clear errors with cudaGetLastError() on any failure.
  Return true only if all steps succeeded.

Phase 1 – default limit:
  Binary search lo=1024, hi=256*1024 with use_extended=false.
  Track largest bytes that succeeds → default_max.

Phase 2 – opt-in extended limit:
  If try_shmem_size(default_max + 1024, d_output, true) succeeds:
    Binary search lo=default_max+1, hi=256*1024 with use_extended=true.
    Track largest success → extended_max.
  Else: extended_max = default_max.

actual_max = max(default_max, extended_max).
Print all output lines.
cudaGetDeviceProperties(&prop, 0); print REPORTED_* including prop.sharedMemPerBlockOptin.

=== IMPLEMENTATION ===
- CUDA_CHECK macro; #include <stdio.h>, <stdlib.h>, <cuda_runtime.h>.
""",
    },

    # ---------------------------------------------------------------------- #
    "bank_conflict_probe": {
        "description": "Shared memory bank conflict penalty measurement",
        "algorithm": r"""
=== TASK ===
Write a standalone CUDA C++ program that measures the shared-memory bank
conflict overhead by comparing strided access patterns.

=== REQUIRED OUTPUT ===
STRIDE=<n> CYCLES_PER_ACCESS=<f2>   (6 lines: strides 1, 2, 4, 8, 16, 32)
NO_CONFLICT_CYCLES=<f2>
MAX_CONFLICT_CYCLES=<f2>
BANK_CONFLICT_PENALTY_CYCLES=<f2>

=== ALGORITHM ===
CALCULATION: cycles_per_access = total_cycles_for_all_threads / (ITERS * 256)
where total_cycles = clock64() delta measured by thread 0 (single block).

Kernel — EXACT implementation required:
  __global__ void bank_conflict_kernel(float* out, int stride, long long* cyc_out)

  // *** CRITICAL: declare shared memory as volatile to prevent the CUDA
  // compiler from hoisting loads out of the timing loop.  Without 'volatile'
  // all iterations access the same constant address and the compiler replaces
  // the loop with a single multiply, giving 0 cycles. ***
  __shared__ volatile int smem[1024];
  int tid = threadIdx.x;   // blockDim.x = 256

  // Initialise all 1024 entries (4 writes per thread for 256 threads)
  smem[tid]       = tid + 1;
  smem[tid + 256] = tid + 1;
  smem[tid + 512] = tid + 1;
  smem[tid + 768] = tid + 1;
  __syncthreads();

  const int ITERS = 1000000;
  int sum = 0;
  long long t0 = clock64();
  #pragma unroll 1
  for (int i = 0; i < ITERS; i++) {
      // No asm volatile needed since smem is volatile.
      sum += smem[(tid * stride) % 1024];
  }
  long long t1 = clock64();

  // *** ALL threads write to out[] to prevent DCE for threads != 0 ***
  out[tid] = (float)sum;
  // Thread 0 reports timing for the entire warp/block (all threads ran same iters)
  if (tid == 0) *cyc_out = t1 - t0;

Config: <<<1, 256>>>.
Allocate: d_out (float, 256), d_cyc (long long, 1).

Per-stride loop (strides 1, 2, 4, 8, 16, 32):
  Run 5 trials; collect total_cycles from d_cyc each trial; take median.
  cycles_per_access = median_total_cycles / (double)(ITERS * 256).
  Print STRIDE=<n> CYCLES_PER_ACCESS=<cycles_per_access:.2f>

no_conflict_cycles  = cycles_per_access for stride 1.
max_conflict_cycles = cycles_per_access for stride 32.
penalty             = max_conflict_cycles - no_conflict_cycles.

=== IMPLEMENTATION ===
- CUDA_CHECK macro; #include <stdio.h>, <stdlib.h>, <cuda_runtime.h>.
- Only one kernel function is needed; call it with different stride values.
""",
    },

    # ---------------------------------------------------------------------- #
    "ncu_verify_probe": {
        "description": "Lightweight Nsight Compute cross-verification probe",
        "algorithm": r"""
=== TASK ===
Write a minimal CUDA C++ program suitable for ncu (Nsight Compute) profiling.

=== REQUIRED OUTPUT ===
NCU_VERIFY_PROBE_READY    (printed BEFORE kernels run)

=== ALGORITHM ===
Kernel A (bandwidth test):
  __global__ void bw_kernel(const float* src, float* out, int n):
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.f;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x):
      sum += src[i];
    if (tid == 0) out[0] = sum;
  Array: 32 MB (8 388 608 floats).  Launch: 512 blocks × 256 threads.

Kernel B (compute test):
  __global__ void fma_kernel(float* out):
    float x = 1.0f;
    for (int i = 0; i < 1000000; i++):
      x = fmaf(x, 1.0000001f, 0.0000001f);
    out[blockIdx.x * blockDim.x + threadIdx.x] = x;
  Launch: 512 blocks × 256 threads.

main():
  printf("NCU_VERIFY_PROBE_READY\n");
  cudaMalloc / init src to 1.0f.
  Launch bw_kernel; cudaDeviceSynchronize.
  Launch fma_kernel; cudaDeviceSynchronize.
  cudaFree.

=== IMPLEMENTATION ===
- CUDA_CHECK macro; #include <stdio.h>, <stdlib.h>, <cuda_runtime.h>.
""",
    },
}


# ── Code Generator ────────────────────────────────────────────────────────────

class ProbeCodeGenerator:
    """
    Autonomous CUDA probe source generator.

    Uses the LLM to write a complete, compilable CUDA C++ file for each
    named probe, guided by the design specification in PROBE_SPECS.
    Generated files are cached in build_dir so they are only regenerated
    when the LLM is explicitly asked to fix a compilation error.
    """

    MAX_RETRIES = 3   # Maximum compile-and-fix iterations

    def __init__(self, build_dir: str):
        self.build_dir = build_dir
        os.makedirs(build_dir, exist_ok=True)
        self._source_cache: dict = {}   # probe_name -> abs path to .cu file

    # ── Public interface ───────────────────────────────────────────────────── #

    def get_source_path(self, probe_name: str) -> str:
        """
        Return the path to the CUDA source file for probe_name.
        Generates it via LLM if it does not already exist on disk.
        """
        cached = self._source_cache.get(probe_name)
        if cached and os.path.exists(cached):
            return cached

        src_path = os.path.join(self.build_dir, f"{probe_name}_generated.cu")
        if os.path.exists(src_path):
            expected = _spec_hash(probe_name)
            try:
                with open(src_path, 'r') as f:
                    first_line = f.readline()
                cached_hash = first_line.strip().replace(_HASH_PREFIX, '') if first_line.startswith(_HASH_PREFIX) else ''
            except Exception:
                cached_hash = ''
            if cached_hash == expected:
                logger.info("Re-using previously generated source: %s", src_path)
                self._source_cache[probe_name] = src_path
                return src_path
            else:
                logger.info(
                    "Spec changed for '%s' (hash %s→%s); regenerating source...",
                    probe_name, cached_hash or 'none', expected
                )
                os.remove(src_path)

        return self._generate_fresh(probe_name)

    def regenerate_with_error(self, probe_name: str, error_text: str) -> str:
        """
        Ask the LLM to fix the compilation errors in the previously generated
        source.  Overwrites build/<probe_name>_generated.cu and returns the
        new path.
        """
        src_path = os.path.join(self.build_dir, f"{probe_name}_generated.cu")
        previous_code = ""
        if os.path.exists(src_path):
            with open(src_path, 'r') as f:
                previous_code = f.read()

        logger.info("Asking LLM to fix compilation error in '%s'...", probe_name)
        client = _get_llm()
        if client is None:
            raise RuntimeError(
                f"LLM unavailable; cannot fix compilation error for '{probe_name}'"
            )

        system = (
            "You are an expert CUDA C++ developer. "
            "The CUDA program below failed to compile with nvcc. "
            "Fix ALL compilation errors shown. "
            "Output ONLY the corrected C++ source code — no markdown fences, "
            "no explanation, no commentary."
        )
        user = (
            f"=== CUDA source to fix ===\n{previous_code}\n\n"
            f"=== nvcc compilation errors ===\n{error_text[:3000]}\n\n"
            "Output the complete, corrected CUDA C++ source file."
        )
        fixed_code = client.generate_reasoning(system, user)
        fixed_code = _strip_code_fences(fixed_code)

        hash_comment = f"{_HASH_PREFIX}{_spec_hash(probe_name)}\n"
        with open(src_path, 'w') as f:
            f.write(hash_comment + fixed_code)
        self._source_cache[probe_name] = src_path
        logger.info("Fixed source written (%d chars): %s", len(fixed_code), src_path)
        return src_path

    # ── Private ────────────────────────────────────────────────────────────── #

    def _generate_fresh(self, probe_name: str) -> str:
        spec = PROBE_SPECS.get(probe_name)
        if spec is None:
            raise ValueError(
                f"No design specification for probe '{probe_name}'. "
                f"Available: {sorted(PROBE_SPECS)}"
            )

        logger.info("Generating CUDA source for '%s' via LLM...", probe_name)
        client = _get_llm()
        if client is None:
            raise RuntimeError(
                f"LLM unavailable; cannot generate CUDA source for '{probe_name}'"
            )

        system = (
            "You are an expert CUDA C++ developer. "
            "Generate a complete, standalone, compilable CUDA C++ source file "
            "based on the design specification provided. "
            "Pay careful attention to the REQUIRED OUTPUT section — the "
            "Python parser depends on the exact key=value format. "
            "Output ONLY the raw C++ source. "
            "Do NOT use markdown code fences (```) around the code. "
            "Do NOT add any prose before or after the code."
        )
        user = (
            f"Task: {spec['description']}\n\n"
            f"{spec['algorithm']}\n\n"
            "Generate the complete CUDA C++ source file now."
        )

        code = client.generate_reasoning(system, user)
        code = _strip_code_fences(code)

        src_path = os.path.join(self.build_dir, f"{probe_name}_generated.cu")
        hash_comment = f"{_HASH_PREFIX}{_spec_hash(probe_name)}\n"
        with open(src_path, 'w') as f:
            f.write(hash_comment + code)
        self._source_cache[probe_name] = src_path
        logger.info(
            "Generated '%s' source: %d chars → %s", probe_name, len(code), src_path
        )
        return src_path
