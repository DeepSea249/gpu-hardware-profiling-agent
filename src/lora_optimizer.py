"""
Phase 2 LoRA optimization agent.

Architecture:
  Stage 1 – Template Search
    Evaluates a fixed library of hand-written CUDA extension templates that
    explore different optimisation axes (stream overlap, pre-allocation,
    contiguous transpose, addmm vs mm+add_).  The best correct variant
    becomes the starting point for Stage 2.

  Stage 2 – LLM-driven Iterative Improvement
    Feeds the current best CUDA code and its benchmark results to an LLM,
    which proposes concrete code modifications.  Each proposal is compiled,
    validated, and benchmarked.  If it beats the current best, it replaces
    optimized_lora.cu.  The loop repeats until the time budget is exhausted.

All templates use PyTorch's at::mm / at::addmm so that the computation
follows the exact same cuBLAS code-path as the reference implementation,
guaranteeing bit-exact correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import textwrap
import time
from typing import Any, Optional

from . import utils

logger = logging.getLogger('GPUAgent.LoRAOptimizer')

_llm_client = None

DEFAULT_BENCHMARK_SIZES = [
    3584, 3600, 3712, 3840, 3968, 4000,
    4096, 4200, 4352, 4480, 4608,
]
MIN_SPEEDUP_GUARD = 0.97
ROBUST_REGRESSION_WEIGHT = 0.35
STAGE1_DEEP_TOP_K = 3
STAGE2_MAX_ITERATIONS = 3
STAGE2_ROBUST_MARGIN = 0.005
STAGE2_MIN_SPEEDUP_GAIN = 0.01
DEFAULT_SAFETY_MARGIN_SECONDS = 150
DEFAULT_MIN_STAGE2_ITERATION_SECONDS = 360
DEFAULT_STAGE1_PASS1_ITERS = 5
DEFAULT_STAGE1_PASS2_ITERS = 15
DEFAULT_STAGE1_SOFT_FRACTION = 0.65
MIN_STAGE1_PASS1_SECONDS = 90
MIN_STAGE1_PASS2_SECONDS = 240
MIN_SHAPE_AWARE_SECONDS = 240
SHAPE_AWARE_BANDS = [
    ('small', 3840),
    ('mid', 4200),
    ('large', None),
]


def _entry_metric(entry: dict[str, Any], key: str,
                  default: float = 0.0) -> float:
    value = entry.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_result_passed(entry: dict[str, Any]) -> bool:
    return bool(entry.get('passed') is True)


def _is_guard_acceptable(entry: dict[str, Any]) -> bool:
    return _entry_metric(entry, 'min_speedup') >= MIN_SPEEDUP_GUARD


def _result_name(entry: dict[str, Any]) -> str:
    return entry.get('candidate', {}).get('name', 'unknown')


def _selection_key(entry: dict[str, Any],
                   prefer_min_first: bool = False) -> tuple[float, ...]:
    min_speedup = _entry_metric(entry, 'min_speedup')
    robust = _entry_metric(entry, 'robust_score',
                           _entry_metric(entry, 'score'))
    geom = _entry_metric(entry, 'geometric_mean')
    mean = _entry_metric(entry, 'arithmetic_mean')
    if prefer_min_first:
        return (min_speedup, robust, geom, mean)
    return (robust, geom, min_speedup, mean)


def _rank_results(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passed = [e for e in entries if _is_result_passed(e)]
    if not passed:
        return []
    acceptable = [e for e in passed if _is_guard_acceptable(e)]
    rejected = [e for e in passed if not _is_guard_acceptable(e)]
    if acceptable:
        return (
            sorted(acceptable, key=_selection_key, reverse=True) +
            sorted(rejected, key=lambda e: _selection_key(e, True),
                   reverse=True)
        )
    return sorted(passed, key=lambda e: _selection_key(e, True),
                  reverse=True)


def _select_best_result(entries: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    ranked = _rank_results(entries)
    return ranked[0] if ranked else None


def _is_stage2_improvement(candidate: dict[str, Any],
                           current_best: dict[str, Any]) -> bool:
    """Guard replacement with robust-score margin and min-speedup safety."""
    if not _is_result_passed(candidate):
        return False
    if not current_best:
        return True

    cand_min = _entry_metric(candidate, 'min_speedup')
    best_min = _entry_metric(current_best, 'min_speedup')
    cand_robust = _entry_metric(candidate, 'robust_score',
                                _entry_metric(candidate, 'score'))
    best_robust = _entry_metric(current_best, 'robust_score',
                                _entry_metric(current_best, 'score'))
    cand_geom = _entry_metric(candidate, 'geometric_mean')
    best_geom = _entry_metric(current_best, 'geometric_mean')

    if best_min >= MIN_SPEEDUP_GUARD and cand_min < MIN_SPEEDUP_GUARD:
        return False
    if cand_robust >= best_robust + STAGE2_ROBUST_MARGIN:
        return True
    if cand_geom >= best_geom - 0.002 and cand_min >= best_min + STAGE2_MIN_SPEEDUP_GAIN:
        return True
    return False


def _get_llm():
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    try:
        from .llm_client import LLMClient
        _llm_client = LLMClient(enable_thinking=False)
        _llm_client._client.timeout = 120
    except Exception as exc:
        logger.warning('LLM client unavailable: %s', exc)
        _llm_client = None
    return _llm_client


# =========================================================================
# Template library (Stage 1)
# =========================================================================

_COMMON_HEADER = textwrap.dedent(r'''
// Auto-generated by Phase 2 LoRA optimization agent.
// Candidate config: __CONFIG_JSON__
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) TORCH_CHECK((expr) == cudaSuccess, \
    "CUDA error: ", cudaGetErrorString(expr))
''').strip()

_PYBIND_FOOTER = textwrap.dedent(r'''
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Optimized LoRA forward");
}
''').strip()

# ---- Template A: baseline sequential (single stream) ----
TEMPLATE_A = _COMMON_HEADER + "\n\n" + textwrap.dedent(r'''
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
  TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(W.is_contiguous() && X.is_contiguous() &&
              A.is_contiguous() && B.is_contiguous(),
              "All tensors must be contiguous");
  at::Tensor Y = at::mm(W, X);
  at::Tensor tmp = at::mm(B.t().contiguous(), X);
  at::addmm_out(Y, Y, A, tmp, 1.0, 1.0);
  return Y;
}
''').strip() + "\n\n" + _PYBIND_FOOTER + "\n"

# ---- Template B: dual-stream overlap (W@X || B^T@X) ----
TEMPLATE_B = _COMMON_HEADER + "\n\n" + textwrap.dedent(r'''
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
  TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(W.is_contiguous() && X.is_contiguous() &&
              A.is_contiguous() && B.is_contiguous(),
              "All tensors must be contiguous");
  const int device = W.get_device();
  c10::cuda::CUDAGuard guard(device);
  cudaStream_t main_stream = c10::cuda::getCurrentCUDAStream(device);
  auto aux_stream = c10::cuda::getStreamFromPool(false, device);
  cudaEvent_t evt_btx;
  CUDA_CHECK(cudaEventCreateWithFlags(&evt_btx, cudaEventDisableTiming));
  at::Tensor Y = at::mm(W, X);
  at::Tensor tmp;
  {
    c10::cuda::CUDAStreamGuard sg(aux_stream);
    tmp = at::mm(B.t().contiguous(), X);
    CUDA_CHECK(cudaEventRecord(evt_btx, aux_stream.stream()));
  }
  CUDA_CHECK(cudaStreamWaitEvent(main_stream, evt_btx, 0));
  at::addmm_out(Y, Y, A, tmp, 1.0, 1.0);
  CUDA_CHECK(cudaEventDestroy(evt_btx));
  return Y;
}
''').strip() + "\n\n" + _PYBIND_FOOTER + "\n"

# ---- Template C: pre-transpose B, dual-stream ----
TEMPLATE_C = _COMMON_HEADER + "\n\n" + textwrap.dedent(r'''
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
  TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(W.is_contiguous() && X.is_contiguous() &&
              A.is_contiguous() && B.is_contiguous(),
              "All tensors must be contiguous");
  const int device = W.get_device();
  c10::cuda::CUDAGuard guard(device);
  cudaStream_t main_stream = c10::cuda::getCurrentCUDAStream(device);
  auto aux_stream = c10::cuda::getStreamFromPool(false, device);
  cudaEvent_t evt_btx;
  CUDA_CHECK(cudaEventCreateWithFlags(&evt_btx, cudaEventDisableTiming));
  at::Tensor Bt;
  {
    c10::cuda::CUDAStreamGuard sg(aux_stream);
    Bt = B.t().contiguous();
  }
  at::Tensor Y = at::mm(W, X);
  at::Tensor tmp;
  {
    c10::cuda::CUDAStreamGuard sg(aux_stream);
    tmp = at::mm(Bt, X);
    CUDA_CHECK(cudaEventRecord(evt_btx, aux_stream.stream()));
  }
  CUDA_CHECK(cudaStreamWaitEvent(main_stream, evt_btx, 0));
  at::addmm_out(Y, Y, A, tmp, 1.0, 1.0);
  CUDA_CHECK(cudaEventDestroy(evt_btx));
  return Y;
}
''').strip() + "\n\n" + _PYBIND_FOOTER + "\n"

# ---- Template D: mm + add_ instead of addmm_out ----
TEMPLATE_D = _COMMON_HEADER + "\n\n" + textwrap.dedent(r'''
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
  TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(W.is_contiguous() && X.is_contiguous() &&
              A.is_contiguous() && B.is_contiguous(),
              "All tensors must be contiguous");
  const int device = W.get_device();
  c10::cuda::CUDAGuard guard(device);
  cudaStream_t main_stream = c10::cuda::getCurrentCUDAStream(device);
  auto aux_stream = c10::cuda::getStreamFromPool(false, device);
  cudaEvent_t evt_btx;
  CUDA_CHECK(cudaEventCreateWithFlags(&evt_btx, cudaEventDisableTiming));
  at::Tensor Y = at::mm(W, X);
  at::Tensor tmp;
  {
    c10::cuda::CUDAStreamGuard sg(aux_stream);
    tmp = at::mm(B.t().contiguous(), X);
    CUDA_CHECK(cudaEventRecord(evt_btx, aux_stream.stream()));
  }
  CUDA_CHECK(cudaStreamWaitEvent(main_stream, evt_btx, 0));
  at::Tensor lora_out = at::mm(A, tmp);
  Y.add_(lora_out);
  CUDA_CHECK(cudaEventDestroy(evt_btx));
  return Y;
}
''').strip() + "\n\n" + _PYBIND_FOOTER + "\n"

# ---- Template E: triple-stream ----
TEMPLATE_E = _COMMON_HEADER + "\n\n" + textwrap.dedent(r'''
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
  TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(W.is_contiguous() && X.is_contiguous() &&
              A.is_contiguous() && B.is_contiguous(),
              "All tensors must be contiguous");
  const int device = W.get_device();
  c10::cuda::CUDAGuard guard(device);
  cudaStream_t main_stream = c10::cuda::getCurrentCUDAStream(device);
  auto stream1 = c10::cuda::getStreamFromPool(false, device);
  auto stream2 = c10::cuda::getStreamFromPool(false, device);
  const auto d = W.size(0);
  const auto r = A.size(1);
  auto Y = at::empty({d, d}, W.options());
  auto tmp = at::empty({r, d}, W.options());
  cudaEvent_t evt_wx, evt_btx;
  CUDA_CHECK(cudaEventCreateWithFlags(&evt_wx, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&evt_btx, cudaEventDisableTiming));
  {
    c10::cuda::CUDAStreamGuard sg(stream1);
    at::mm_out(Y, W, X);
    CUDA_CHECK(cudaEventRecord(evt_wx, stream1.stream()));
  }
  {
    c10::cuda::CUDAStreamGuard sg(stream2);
    at::mm_out(tmp, B.t().contiguous(), X);
    CUDA_CHECK(cudaEventRecord(evt_btx, stream2.stream()));
  }
  CUDA_CHECK(cudaStreamWaitEvent(main_stream, evt_wx, 0));
  CUDA_CHECK(cudaStreamWaitEvent(main_stream, evt_btx, 0));
  at::addmm_out(Y, Y, A, tmp, 1.0, 1.0);
  CUDA_CHECK(cudaEventDestroy(evt_wx));
  CUDA_CHECK(cudaEventDestroy(evt_btx));
  return Y;
}
''').strip() + "\n\n" + _PYBIND_FOOTER + "\n"

# ---- Template F: single stream, pre-alloc mm_out ----
TEMPLATE_F = _COMMON_HEADER + "\n\n" + textwrap.dedent(r'''
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
  TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(W.is_contiguous() && X.is_contiguous() &&
              A.is_contiguous() && B.is_contiguous(),
              "All tensors must be contiguous");
  const auto d = W.size(0);
  const auto r = A.size(1);
  auto Y = at::empty({d, d}, W.options());
  auto tmp = at::empty({r, d}, W.options());
  at::mm_out(Y, W, X);
  at::mm_out(tmp, B.t().contiguous(), X);
  at::addmm_out(Y, Y, A, tmp, 1.0, 1.0);
  return Y;
}
''').strip() + "\n\n" + _PYBIND_FOOTER + "\n"

# ---- Template G: dual-stream + pre-alloc + mm_out ----
TEMPLATE_G = _COMMON_HEADER + "\n\n" + textwrap.dedent(r'''
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
  TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(W.is_contiguous() && X.is_contiguous() &&
              A.is_contiguous() && B.is_contiguous(),
              "All tensors must be contiguous");
  const int device = W.get_device();
  c10::cuda::CUDAGuard guard(device);
  cudaStream_t main_stream = c10::cuda::getCurrentCUDAStream(device);
  auto aux_stream = c10::cuda::getStreamFromPool(false, device);
  const auto d = W.size(0);
  const auto r = A.size(1);
  auto Y = at::empty({d, d}, W.options());
  auto tmp = at::empty({r, d}, W.options());
  cudaEvent_t evt_btx;
  CUDA_CHECK(cudaEventCreateWithFlags(&evt_btx, cudaEventDisableTiming));
  at::mm_out(Y, W, X);
  {
    c10::cuda::CUDAStreamGuard sg(aux_stream);
    at::mm_out(tmp, B.t().contiguous(), X);
    CUDA_CHECK(cudaEventRecord(evt_btx, aux_stream.stream()));
  }
  CUDA_CHECK(cudaStreamWaitEvent(main_stream, evt_btx, 0));
  at::addmm_out(Y, Y, A, tmp, 1.0, 1.0);
  CUDA_CHECK(cudaEventDestroy(evt_btx));
  return Y;
}
''').strip() + "\n\n" + _PYBIND_FOOTER + "\n"

# ---- Template H: dual-stream + contiguous Bt + mm_out ----
TEMPLATE_H = _COMMON_HEADER + "\n\n" + textwrap.dedent(r'''
torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
  TORCH_CHECK(W.is_cuda() && X.is_cuda() && A.is_cuda() && B.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(W.is_contiguous() && X.is_contiguous() &&
              A.is_contiguous() && B.is_contiguous(),
              "All tensors must be contiguous");
  const int device = W.get_device();
  c10::cuda::CUDAGuard guard(device);
  cudaStream_t main_stream = c10::cuda::getCurrentCUDAStream(device);
  auto aux_stream = c10::cuda::getStreamFromPool(false, device);
  const auto d = W.size(0);
  const auto r = A.size(1);
  auto Y = at::empty({d, d}, W.options());
  auto tmp = at::empty({r, d}, W.options());
  cudaEvent_t evt_btx;
  CUDA_CHECK(cudaEventCreateWithFlags(&evt_btx, cudaEventDisableTiming));
  at::mm_out(Y, W, X);
  {
    c10::cuda::CUDAStreamGuard sg(aux_stream);
    at::Tensor Bt = B.t().contiguous();
    at::mm_out(tmp, Bt, X);
    CUDA_CHECK(cudaEventRecord(evt_btx, aux_stream.stream()));
  }
  CUDA_CHECK(cudaStreamWaitEvent(main_stream, evt_btx, 0));
  at::addmm_out(Y, Y, A, tmp, 1.0, 1.0);
  CUDA_CHECK(cudaEventDestroy(evt_btx));
  return Y;
}
''').strip() + "\n\n" + _PYBIND_FOOTER + "\n"


TEMPLATE_LIBRARY: dict[str, str] = {
    'baseline_sequential': TEMPLATE_A,
    'dual_stream_overlap': TEMPLATE_B,
    'dual_stream_pretranspose': TEMPLATE_C,
    'dual_stream_mm_add': TEMPLATE_D,
    'triple_stream_prealloc': TEMPLATE_E,
    'sequential_prealloc': TEMPLATE_F,
    'dual_stream_prealloc': TEMPLATE_G,
    'dual_stream_prealloc_contig': TEMPLATE_H,
}


# =========================================================================
# Data classes
# =========================================================================

@dataclass(frozen=True)
class CandidateConfig:
    template_name: str
    block_size: Optional[int]
    rationale: str

    @property
    def name(self) -> str:
        if self.block_size is None:
            return self.template_name
        return f'{self.template_name}_bs{self.block_size}'

    def to_dict(self) -> dict[str, Any]:
        return {
            'name': self.name,
            'template_name': self.template_name,
            'block_size': self.block_size,
            'rationale': self.rationale,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(payload.encode('utf-8')).hexdigest()[:8]


# =========================================================================
# Code generation
# =========================================================================

class LoRACodegen:
    """Render templates and LLM-generated code."""

    @staticmethod
    def render(config: CandidateConfig) -> str:
        template = TEMPLATE_LIBRARY[config.template_name]
        return template.replace(
            '__CONFIG_JSON__',
            json.dumps(config.to_dict(), sort_keys=True),
        )

    @staticmethod
    def render_helper(config: CandidateConfig, function_name: str) -> str:
        """Render a template candidate as a static helper function."""
        code = LoRACodegen.render(config)
        start = code.find('torch::Tensor forward(')
        end = code.find('PYBIND11_MODULE')
        if start < 0 or end < 0:
            raise ValueError(f'Could not extract forward() from {config.name}')
        helper = code[start:end].strip()
        return helper.replace(
            'torch::Tensor forward(',
            f'static torch::Tensor {function_name}(',
            1,
        )

    @staticmethod
    def render_shape_aware_dispatch(intervals: list[dict[str, Any]]) -> str:
        """Render a final single-file dispatcher over safe template helpers."""
        payload = {
            'template_name': 'shape_aware_dispatch',
            'intervals': [
                {
                    'label': item['label'],
                    'upper_bound': item['upper_bound'],
                    'candidate': item['candidate'].to_dict(),
                }
                for item in intervals
            ],
        }
        header = _COMMON_HEADER.replace(
            '__CONFIG_JSON__',
            json.dumps(payload, sort_keys=True),
        )

        functions = []
        fn_by_name: dict[str, str] = {}
        for item in intervals:
            candidate = item['candidate']
            if candidate.name in fn_by_name:
                continue
            fn_name = 'forward_' + re.sub(r'[^a-zA-Z0-9_]', '_', candidate.name)
            fn_by_name[candidate.name] = fn_name
            functions.append(LoRACodegen.render_helper(candidate, fn_name))

        lines = [header, '']
        lines.extend(functions)
        lines.extend([
            '',
            'torch::Tensor forward(torch::Tensor W,',
            '                      torch::Tensor X,',
            '                      torch::Tensor A,',
            '                      torch::Tensor B) {',
            '  const auto d = W.size(0);',
        ])

        fallback_fn = None
        for item in intervals:
            fn_name = fn_by_name[item['candidate'].name]
            upper = item['upper_bound']
            fallback_fn = fn_name
            if upper is not None:
                lines.append(
                    f'  if (d <= {int(upper)}) return {fn_name}(W, X, A, B);'
                )
            else:
                lines.append(f'  return {fn_name}(W, X, A, B);')
        if intervals and intervals[-1]['upper_bound'] is not None:
            lines.append(f'  return {fallback_fn}(W, X, A, B);')
        lines.extend(['}', '', _PYBIND_FOOTER, ''])
        return '\n'.join(lines)

    @staticmethod
    def validate_llm_code_safety(code: str) -> tuple[bool, str]:
        """Reject Stage 2 code that attempts risky numerical paths."""
        lowered = code.lower()
        forbidden = [
            'cublas', 'wmma', 'mma_sync', 'nvcuda', '__global__',
            '__device__', '__half', 'cutlass', 'setallowtf32',
            'allow_tf32', 'setfloat32matmulprecision', 'khalf',
            'kbfloat16', 'bfloat16', 'cudadevicesynchronize',
            'cudamemcpy',
        ]
        for token in forbidden:
            if token in lowered:
                return False, f'forbidden token: {token}'
        if 'at::mm' not in code and 'at::addmm' not in code:
            return False, 'missing safe PyTorch mm/addmm operators'
        if '.contiguous()' not in code or ('B.t()' not in code and 'B.transpose' not in code):
            return False, 'missing B transpose followed by contiguous materialization'
        if 'PYBIND11_MODULE' not in code or 'torch::Tensor forward' not in code:
            return False, 'missing required extension interface'
        return True, ''

    @staticmethod
    def extract_cuda_from_llm(llm_response: str) -> Optional[str]:
        """Extract CUDA source code from an LLM response.

        Looks for code inside ```cpp or ```cuda fences first, then falls
        back to ```...``` fences that contain typical CUDA markers.
        """
        # Try explicit language fences first
        for lang in ('cpp', 'cuda', 'c++', 'c'):
            pattern = rf'```{lang}\s*\n(.*?)```'
            m = re.search(pattern, llm_response, re.DOTALL | re.IGNORECASE)
            if m:
                code = m.group(1).strip()
                if 'PYBIND11_MODULE' in code and 'forward' in code:
                    return code

        # Generic fence
        pattern = r'```\s*\n(.*?)```'
        for m in re.finditer(pattern, llm_response, re.DOTALL):
            code = m.group(1).strip()
            if 'PYBIND11_MODULE' in code and 'forward' in code:
                return code

        # No fences – check if the whole response looks like code
        if ('PYBIND11_MODULE' in llm_response and
                'forward' in llm_response and
                '#include' in llm_response):
            return llm_response.strip()

        return None


# =========================================================================
# Benchmark harness
# =========================================================================

class LoRABenchmarkHarness:
    """Compile, validate, and benchmark generated candidates."""

    def __init__(self, build_dir: Path, benchmark_sizes: list[int],
                 warmup: int, iters: int):
        self.build_dir = build_dir
        self.benchmark_sizes = benchmark_sizes
        self.warmup = warmup
        self.iters = iters

    @staticmethod
    def torch_available() -> bool:
        return importlib.util.find_spec('torch') is not None

    @staticmethod
    def _import_torch():
        import torch
        from torch.utils.cpp_extension import load
        return torch, load

    def evaluate(self, candidate_name: str, source_path: Path,
                 fingerprint: str = '', benchmark_sizes: Optional[list[int]] = None,
                 warmup: Optional[int] = None,
                 iters: Optional[int] = None) -> dict[str, Any]:
        """Compile and benchmark a single .cu file.

        Returns a dict with per-size speedups and robust summary metrics.
        """
        torch, load = self._import_torch()
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA not available')
        sizes = benchmark_sizes or self.benchmark_sizes
        warmup = self.warmup if warmup is None else warmup
        iters = self.iters if iters is None else iters

        module_name = f'lora_{candidate_name}_{fingerprint}'[:60]
        # sanitise for C identifier
        module_name = re.sub(r'[^a-zA-Z0-9_]', '_', module_name)
        module_build_dir = self.build_dir / module_name
        module_build_dir.mkdir(parents=True, exist_ok=True)

        compile_start = time.time()
        try:
            module = load(
                name=module_name,
                sources=[str(source_path)],
                verbose=False,
                build_directory=str(module_build_dir),
                with_cuda=True,
                extra_cflags=['-O3'],
                extra_cuda_cflags=['-O3'],
            )
        except Exception as exc:
            return {
                'passed': False, 'score': 0.0,
                'geometric_mean': 0.0, 'arithmetic_mean': 0.0,
                'min_speedup': 0.0, 'max_speedup': 0.0,
                'robust_score': 0.0,
                'compile_seconds': round(time.time() - compile_start, 3),
                'per_size': [],
                'error': f'Compilation failed: {exc}',
            }
        compile_seconds = time.time() - compile_start

        per_size = []
        for dim in sizes:
            W, X, A, B = self._make_inputs(torch, dim)
            try:
                with torch.no_grad():
                    y_student = module.forward(W, X, A, B)
                    y_ref = self._reference_impl(torch, W, X, A, B)
            except Exception as exc:
                per_size.append({
                    'd': dim, 'passed': False,
                    'error': str(exc),
                })
                del W, X, A, B
                torch.cuda.empty_cache()
                return {
                    'passed': False, 'score': 0.0,
                    'geometric_mean': 0.0, 'arithmetic_mean': 0.0,
                    'min_speedup': 0.0, 'max_speedup': 0.0,
                    'robust_score': 0.0,
                    'compile_seconds': round(compile_seconds, 3),
                    'per_size': per_size,
                }

            passed, max_abs_err, rel_l2_err = self._check_correctness(
                torch, y_student, y_ref)
            if not passed:
                per_size.append({
                    'd': dim, 'passed': False,
                    'max_abs_err': max_abs_err, 'rel_l2_err': rel_l2_err,
                })
                del W, X, A, B, y_student, y_ref
                torch.cuda.empty_cache()
                return {
                    'passed': False, 'score': 0.0,
                    'geometric_mean': 0.0, 'arithmetic_mean': 0.0,
                    'min_speedup': 0.0, 'max_speedup': 0.0,
                    'robust_score': 0.0,
                    'compile_seconds': round(compile_seconds, 3),
                    'per_size': per_size,
                }

            student_ms = self._benchmark(
                torch, lambda: module.forward(W, X, A, B), warmup, iters)
            torch_ms = self._benchmark(
                torch, lambda: self._reference_impl(torch, W, X, A, B),
                warmup, iters)
            speedup = torch_ms / student_ms if student_ms > 0 else 0.0
            per_size.append({
                'd': dim, 'passed': True,
                'max_abs_err': max_abs_err, 'rel_l2_err': rel_l2_err,
                'student_median_ms': round(student_ms, 4),
                'torch_median_ms': round(torch_ms, 4),
                'speedup': round(speedup, 6),
            })
            del W, X, A, B, y_student, y_ref
            torch.cuda.empty_cache()

        summary = self._summarize_speedups(per_size)
        return {
            'passed': True,
            'score': summary['robust_score'],
            'compile_seconds': round(compile_seconds, 3),
            'per_size': per_size,
            **summary,
        }

    def evaluate_config(self, candidate: CandidateConfig,
                        source_path: Path,
                        benchmark_sizes: Optional[list[int]] = None,
                        warmup: Optional[int] = None,
                        iters: Optional[int] = None) -> dict[str, Any]:
        result = self.evaluate(candidate.name, source_path,
                               candidate.fingerprint,
                               benchmark_sizes=benchmark_sizes,
                               warmup=warmup,
                               iters=iters)
        result['candidate'] = candidate.to_dict()
        return result

    @staticmethod
    def _make_inputs(torch, dim: int):
        torch.manual_seed(dim)
        device = 'cuda'
        W = torch.randn(dim, dim, device=device, dtype=torch.float32).contiguous()
        X = torch.randn(dim, dim, device=device, dtype=torch.float32).contiguous()
        A = torch.randn(dim, 16, device=device, dtype=torch.float32).contiguous()
        B = torch.randn(dim, 16, device=device, dtype=torch.float32).contiguous()
        return W, X, A, B

    @staticmethod
    def _reference_impl(torch, W, X, A, B):
        return W @ X + A @ (B.transpose(0, 1).contiguous() @ X)

    @staticmethod
    def _check_correctness(torch, y_student, y_ref):
        diff = (y_student - y_ref).float()
        max_abs_err = diff.abs().max().item()
        rel_l2_err = (diff.norm() / (y_ref.float().norm() + 1e-12)).item()
        passed = torch.allclose(y_student, y_ref, rtol=1e-4, atol=1e-4)
        return passed, max_abs_err, rel_l2_err

    def _benchmark(self, torch, fn, warmup: int, iters: int):
        for _ in range(warmup):
            _ = fn()
        torch.cuda.synchronize()
        timings = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = fn()
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
        timings.sort()
        return timings[len(timings) // 2]

    @classmethod
    def _summarize_speedups(cls, per_size: list[dict[str, Any]]) -> dict[str, float]:
        speedups = [
            float(entry['speedup'])
            for entry in per_size
            if entry.get('passed') and entry.get('speedup') is not None
        ]
        if not speedups:
            return {
                'geometric_mean': 0.0,
                'arithmetic_mean': 0.0,
                'min_speedup': 0.0,
                'max_speedup': 0.0,
                'robust_score': 0.0,
            }
        geometric = cls._geometric_mean(speedups)
        arithmetic = sum(speedups) / len(speedups)
        min_speedup = min(speedups)
        max_speedup = max(speedups)
        robust = geometric - ROBUST_REGRESSION_WEIGHT * max(
            0.0, 1.0 - min_speedup)
        return {
            'geometric_mean': round(geometric, 6),
            'arithmetic_mean': round(arithmetic, 6),
            'min_speedup': round(min_speedup, 6),
            'max_speedup': round(max_speedup, 6),
            'robust_score': round(robust, 6),
        }

    @staticmethod
    def _geometric_mean(values: list[float]) -> float:
        if not values:
            return 0.0
        safe = [max(v, 1e-9) for v in values]
        return math.exp(sum(math.log(v) for v in safe) / len(safe))


# =========================================================================
# LLM-driven iterative optimizer (Stage 2)
# =========================================================================

_LLM_SYSTEM_PROMPT = textwrap.dedent("""\
You are a CUDA performance engineer optimising a LoRA forward operator:
  Y = W @ X + A @ (B^T @ X)
where W,X are [d,d] float32, A,B are [d,16] float32, d in [3584,4608].
Target GPU: NVIDIA RTX 3090 (sm_86), CUDA 12.4, PyTorch 2.3.

CRITICAL CONSTRAINT — correctness:
  The evaluation harness compares your output against PyTorch's reference:
    Y_ref = W @ X + A @ (B.transpose(0,1).contiguous() @ X)
  using torch.allclose(Y, Y_ref, rtol=1e-4, atol=1e-4).
  To guarantee bit-exact match you MUST use PyTorch's at::mm / at::addmm
  for matrix multiplications (they call the same cuBLAS path as Python's @).
  DO NOT call cublasSgemm directly — it uses a different accumulation order
  and will fail the correctness check.

Your code must be a single self-contained .cu file that:
  - #include <torch/extension.h> and related headers
  - defines: torch::Tensor forward(torch::Tensor W, X, A, B)
  - ends with PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward, ...); }

Allowed optimisations (SAFE — preserve correctness):
  - Stream overlap: run W@X and B^T@X on separate CUDA streams
  - Pre-allocate output tensors with at::empty + at::mm_out / at::addmm_out
  - Make B.t() contiguous before mm to match reference accumulation order
  - Use at::addmm_out vs separate at::mm + add_
  - Simple shape-aware dispatch among safe PyTorch-op implementations
  - Avoid unnecessary allocation or stream/event overhead

FORBIDDEN (will break correctness or compilation):
  - Direct cublasSgemm calls for the large d×d GEMMs
  - Changing dtype (fp16/bf16/tf32) for the d×d multiplications
  - Approximate or reordered accumulation for W@X
  - Custom CUDA kernels, custom accumulation, WMMA, or Tensor Cores

Reply with the COMPLETE .cu file inside a ```cpp code fence.
No partial snippets — the entire file must be compilable.
""").strip()

_LLM_SYSTEM_PROMPT = textwrap.dedent("""\
You are a CUDA performance engineer optimising a LoRA forward operator:
  Y = W @ X + A @ (B^T @ X)
where W,X are [d,d] float32, A,B are [d,16] float32, d in [3584,4608].
Target GPU: NVIDIA RTX 3090 (sm_86), CUDA 12.4, PyTorch 2.3.

CRITICAL CONSTRAINTS - correctness and safety:
  The evaluation harness compares your output against PyTorch's reference:
    Y_ref = W @ X + A @ (B.transpose(0,1).contiguous() @ X)
  using torch.allclose(Y, Y_ref, rtol=1e-4, atol=1e-4).
  You MUST preserve B.t().contiguous() before B^T @ X.
  You MUST use safe PyTorch C++ ops such as at::mm, at::mm_out,
  at::addmm, and at::addmm_out for all matrix multiplications.

Your code must be a single self-contained .cu file that:
  - #include <torch/extension.h> and related headers
  - preserves the exact public interface:
      torch::Tensor forward(torch::Tensor W,
                            torch::Tensor X,
                            torch::Tensor A,
                            torch::Tensor B)
  - ends with PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward, ...); }

Allowed optimisations only:
  - Stream scheduling: overlap W@X and B^T@X on safe CUDA streams
  - Pre-allocate output tensors with at::empty + at::mm_out / at::addmm_out
  - Choose between at::addmm_out and safe mm + add_ variants
  - Simple shape-aware dispatch among safe PyTorch-op implementations
  - Avoid unnecessary allocation or stream/event overhead

FORBIDDEN:
  - raw cuBLAS calls, including cublasSgemm
  - FP16, BF16, WMMA, Tensor Cores, CUTLASS, or custom matmul kernels
  - custom CUDA accumulation kernels or __global__ kernels
  - changing global TF32 or PyTorch matmul settings
  - caching full outputs or assuming repeated benchmark inputs
  - dropping B.t().contiguous()

The candidate will be rejected unless it compiles, passes correctness on all
benchmark sizes, and improves robust score by at least 0.005, or keeps
geometric mean similar while clearly improving minimum per-size speedup.

Reply with the COMPLETE .cu file inside a ```cpp code fence.
No partial snippets; the entire file must be compilable.
""").strip()


class LLMIterativeOptimizer:
    """Stage 2: LLM-driven iterative improvement loop."""

    def __init__(self, harness: LoRABenchmarkHarness, build_dir: Path,
                 optimized_path: Path, deadline: float,
                 max_iterations: int = STAGE2_MAX_ITERATIONS,
                 min_iteration_seconds: int = DEFAULT_MIN_STAGE2_ITERATION_SECONDS):
        self.harness = harness
        self.build_dir = build_dir
        self.optimized_path = optimized_path
        self.deadline = deadline
        self.max_iterations = max_iterations
        self.min_iteration_seconds = min_iteration_seconds

    def run(self, current_best_code: str, current_best_entry: dict[str, Any],
            stage1_history: list[dict[str, Any]],
            start_time: float) -> list[dict[str, Any]]:
        """Run LLM-driven iterations. Returns iteration history."""
        client = _get_llm()
        if client is None:
            logger.warning('LLM unavailable — skipping Stage 2')
            return []

        history = []
        best_code = current_best_code
        best_entry = dict(current_best_entry)

        for iteration_index in range(self.max_iterations):
            elapsed = time.time() - start_time
            remaining = self.deadline - time.time()
            if remaining < self.min_iteration_seconds:
                skipped = self.max_iterations - iteration_index
                logger.info(
                    'Stage 2 stopped early due to time budget: '
                    'time_left=%.1fs minimum_required=%ds skipped=%d',
                    remaining, self.min_iteration_seconds, skipped)
                history.append({
                    'status': 'time_budget_stop',
                    'skipped_iterations': skipped,
                    'time_left_seconds': round(max(0.0, remaining), 3),
                    'minimum_stage2_iteration_seconds': self.min_iteration_seconds,
                })
                break

            iteration = iteration_index + 1
            logger.info('='*50)
            logger.info('Stage 2 iteration %d (%.0fs elapsed, %.0fs remaining)',
                        iteration, elapsed, remaining)
            logger.info('='*50)

            # Build prompt with current best code and benchmark results
            user_prompt = self._build_prompt(
                best_code, best_entry, stage1_history, history, iteration)

            # Ask LLM
            try:
                logger.info('Requesting LLM for improved implementation...')
                llm_response = client.generate_reasoning(
                    system_prompt=_LLM_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
            except Exception as exc:
                logger.warning('LLM call failed: %s', exc)
                history.append({
                    'iteration': iteration,
                    'status': 'llm_error',
                    'error': str(exc),
                })
                continue

            # Extract code
            new_code = LoRACodegen.extract_cuda_from_llm(llm_response)
            if new_code is None:
                logger.warning('Could not extract valid CUDA code from LLM')
                history.append({
                    'iteration': iteration,
                    'status': 'extraction_failed',
                    'llm_response_length': len(llm_response),
                })
                continue

            safe, reason = LoRACodegen.validate_llm_code_safety(new_code)
            if not safe:
                logger.warning('Stage 2 code rejected by safety filter: %s',
                               reason)
                history.append({
                    'iteration': iteration,
                    'status': 'safety_rejected',
                    'reason': reason,
                })
                continue

            remaining = self.deadline - time.time()
            if remaining < self.min_iteration_seconds:
                skipped = self.max_iterations - iteration_index
                logger.info(
                    'Stage 2 iteration %d generated code but skipped compile/'
                    'benchmark due to time budget: time_left=%.1fs '
                    'minimum_required=%ds skipped=%d',
                    iteration, remaining, self.min_iteration_seconds, skipped)
                history.append({
                    'iteration': iteration,
                    'status': 'time_budget_stop',
                    'skipped_iterations': skipped,
                    'time_left_seconds': round(max(0.0, remaining), 3),
                    'minimum_stage2_iteration_seconds': self.min_iteration_seconds,
                })
                break

            # Write, compile, benchmark
            source_path = (self.build_dir /
                           f'llm_iter{iteration}_{int(time.time())}.cu')
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(new_code, encoding='utf-8')

            fp = hashlib.md5(new_code.encode()).hexdigest()[:8]
            result = self.harness.evaluate(
                f'llm_iter{iteration}', source_path, fp)
            result['iteration'] = iteration
            result['status'] = 'benchmarked'
            result['candidate'] = {
                'name': f'llm_iter{iteration}',
                'template_name': 'llm_generated',
                'block_size': None,
                'rationale': 'LLM-proposed safe scheduling/preallocation variant',
            }

            if result.get('passed') and result.get('robust_score', 0) > 0:
                logger.info(
                    'Stage 2 iter %d: PASSED robust=%.6f geom=%.6f min=%.6f '
                    '(best robust=%.6f geom=%.6f min=%.6f)',
                    iteration,
                    result.get('robust_score', 0.0),
                    result.get('geometric_mean', 0.0),
                    result.get('min_speedup', 0.0),
                    best_entry.get('robust_score', best_entry.get('score', 0.0)),
                    best_entry.get('geometric_mean', 0.0),
                    best_entry.get('min_speedup', 0.0),
                )
                if _is_stage2_improvement(result, best_entry):
                    logger.info('*** New robust best: %s -> %s ***',
                                _result_name(best_entry), _result_name(result))
                    best_entry = result
                    best_code = new_code
                    # Write to optimized_lora.cu immediately
                    tmp = self.optimized_path.with_suffix('.cu.tmp')
                    tmp.write_text(best_code, encoding='utf-8')
                    tmp.replace(self.optimized_path)
                    result['is_new_best'] = True
                else:
                    result['is_new_best'] = False
            else:
                logger.info('Stage 2 iter %d: FAILED (passed=%s robust=%s)',
                            iteration, result.get('passed'),
                            result.get('robust_score'))
                result['is_new_best'] = False

            history.append(result)

        return history

    def _build_prompt(self, current_code: str, current_best_entry: dict[str, Any],
                      stage1_history: list, stage2_history: list,
                      iteration: int) -> str:
        lines = []
        lines.append('## Current best implementation')
        lines.append(
            f"- candidate: {_result_name(current_best_entry)}\n"
            f"- robust_score: {current_best_entry.get('robust_score', current_best_entry.get('score'))}\n"
            f"- geometric_mean: {current_best_entry.get('geometric_mean')}\n"
            f"- min_speedup: {current_best_entry.get('min_speedup')}\n"
        )
        lines.append('```cpp')
        lines.append(current_code)
        lines.append('```\n')

        # Stage 1 summary
        lines.append('## Stage 1 template search results:')
        for entry in stage1_history:
            cand = entry.get('candidate', {})
            name = cand.get('name', '?')
            score = entry.get('robust_score', entry.get('score', 0))
            geom = entry.get('geometric_mean', 0)
            min_sp = entry.get('min_speedup', 0)
            status = entry.get('status', '?')
            lines.append(
                f'  - {name}: robust={score} geom={geom} '
                f'min={min_sp} status={status}')
        lines.append('')

        # Stage 2 history
        if stage2_history:
            lines.append('## Previous LLM iterations:')
            for h in stage2_history:
                it = h.get('iteration', '?')
                sc = h.get('robust_score', h.get('score', 0))
                st = h.get('status', '?')
                best = h.get('is_new_best', False)
                lines.append(f'  - iter {it}: robust={sc} status={st}'
                             f'{" (NEW BEST)" if best else ""}')
            lines.append('')

        lines.append(f'## Task (iteration {iteration}):')
        lines.append('Analyse the current implementation and propose one '
                     'complete replacement .cu file. Improve robust score, '
                     'especially minimum per-size speedup. Focus only on:')
        lines.append('  1. Safe stream overlap strategies')
        lines.append('  2. Pre-allocation using at::empty + mm_out/addmm_out')
        lines.append('  3. at::addmm_out vs safe mm + add_ scheduling choices')
        lines.append('  4. Simple shape-aware dispatch among safe variants')
        lines.append('Do not add raw cuBLAS, FP16/BF16, WMMA/Tensor Cores, '
                     'custom kernels, custom accumulation, TF32 setting '
                     'changes, or output/input caching.')
        lines.append('')
        lines.append('Return the COMPLETE .cu file in a ```cpp code fence.')

        return '\n'.join(lines)


# =========================================================================
# Top-level orchestrator
# =========================================================================

class LoRAOptimizationAgent:
    """Two-stage Phase 2 orchestrator.

    Stage 1: fixed template search (fast baseline)
    Stage 2: LLM-driven iterative improvement (if LLM is available)
    """

    def __init__(
        self,
        project_root: str,
        build_dir: str,
        optimized_path: str,
        summary_path: str,
        benchmark_sizes: list[int],
        benchmark_warmup: int,
        benchmark_iters: int,
        search_rounds: int,
        time_budget_seconds: int,
        safety_margin_seconds: int = DEFAULT_SAFETY_MARGIN_SECONDS,
        stage2_max_iters: int = STAGE2_MAX_ITERATIONS,
        min_stage2_iteration_seconds: int = DEFAULT_MIN_STAGE2_ITERATION_SECONDS,
        stage1_pass1_iters: int = DEFAULT_STAGE1_PASS1_ITERS,
        stage1_pass2_iters: int = DEFAULT_STAGE1_PASS2_ITERS,
        stage1_topk: int = STAGE1_DEEP_TOP_K,
        use_llm: bool = True,
    ):
        self.project_root = Path(project_root)
        self.build_dir = Path(build_dir)
        self.optimized_path = Path(optimized_path)
        self.summary_path = Path(summary_path)
        self.history_path = self.project_root / 'phase2_search_history.json'
        self.benchmark_sizes = benchmark_sizes
        self.benchmark_warmup = benchmark_warmup
        self.benchmark_iters = benchmark_iters
        self.search_rounds = search_rounds
        self.time_budget_seconds = time_budget_seconds
        self.safety_margin_seconds = max(0, int(safety_margin_seconds))
        self.stage2_max_iters = max(0, int(stage2_max_iters))
        self.min_stage2_iteration_seconds = max(
            0, int(min_stage2_iteration_seconds))
        self.stage1_pass1_iters = max(1, int(stage1_pass1_iters))
        self.stage1_pass2_iters = max(1, int(stage1_pass2_iters))
        self.stage1_topk = max(1, int(stage1_topk))
        self.use_llm = use_llm
        self.harness = LoRABenchmarkHarness(
            build_dir=self.build_dir / 'phase2',
            benchmark_sizes=benchmark_sizes,
            warmup=benchmark_warmup,
            iters=benchmark_iters,
        )

    def run(self) -> dict[str, Any]:
        start = time.time()
        hard_budget_seconds = max(0, int(self.time_budget_seconds))
        effective_budget_seconds = max(
            0, hard_budget_seconds - self.safety_margin_seconds)
        deadline = start + effective_budget_seconds
        stage1_deadline = (
            start + effective_budget_seconds * DEFAULT_STAGE1_SOFT_FRACTION
            if self.use_llm else deadline
        )
        logger.info(
            'Time budget: hard=%ds safety_margin=%ds effective=%ds '
            'deadline_epoch=%.3f stage1_deadline_epoch=%.3f',
            hard_budget_seconds, self.safety_margin_seconds,
            effective_budget_seconds, deadline, stage1_deadline)

        self.optimized_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.harness.build_dir.mkdir(parents=True, exist_ok=True)

        gpu_info = utils.get_gpu_info()
        candidates = self._candidate_library()
        initial_candidate = self._baseline_candidate(candidates)
        self._write_candidate(initial_candidate)

        # Check torch availability
        torch_available = self.harness.torch_available()
        if torch_available:
            try:
                torch, _ = self.harness._import_torch()
                torch_available = bool(torch.cuda.is_available())
            except Exception as exc:
                logger.warning('Torch import failed: %s', exc)
                torch_available = False

        # =============================================================
        # Stage 1: Template search
        # =============================================================
        logger.info('=' * 60)
        logger.info('STAGE 1: Template Search (%d candidates)', len(candidates))
        logger.info('=' * 60)

        stage1_history: list[dict[str, Any]] = []
        pass1_results: list[dict[str, Any]] = []
        pass2_results: list[dict[str, Any]] = []
        shape_aware_code: Optional[str] = None

        cheap_warmup = max(1, min(2, self.benchmark_warmup))
        cheap_iters = self.stage1_pass1_iters

        for candidate in candidates:
            if not self._has_time_for_work(
                deadline, MIN_STAGE1_PASS1_SECONDS,
                f'Stage 1 pass 1 candidate {candidate.name}'):
                break
            if not self._has_time_for_work(
                stage1_deadline, MIN_STAGE1_PASS1_SECONDS,
                f'Stage 1 pass 1 candidate {candidate.name}'):
                break
            source_path = self._write_candidate(
                candidate,
                self.harness.build_dir / f'{candidate.name}_{candidate.fingerprint}.cu',
            )
            if not torch_available:
                result = {
                    'candidate': candidate.to_dict(),
                    'passed': None, 'score': None,
                    'geometric_mean': None, 'arithmetic_mean': None,
                    'min_speedup': None, 'max_speedup': None,
                    'robust_score': None,
                    'status': 'torch_unavailable',
                }
                stage1_history.append(result)
                pass1_results.append(result)
                continue

            try:
                result = self.harness.evaluate_config(
                    candidate, source_path,
                    benchmark_sizes=self.benchmark_sizes,
                    warmup=cheap_warmup,
                    iters=cheap_iters,
                )
                result['status'] = 'pass1_cheap_benchmarked'
                result['benchmark_pass'] = 1
                result['benchmark_warmup'] = cheap_warmup
                result['benchmark_iters'] = cheap_iters
                stage1_history.append(result)
                pass1_results.append(result)
                self._log_candidate_result('Stage 1 pass 1', candidate.name,
                                           result)
            except Exception as exc:
                logger.exception('Candidate %s failed in pass 1', candidate.name)
                result = self._error_result(candidate, 'pass1_error', exc)
                stage1_history.append(result)
                pass1_results.append(result)

        logger.info('Elapsed after Stage 1 pass 1: %.1fs',
                    time.time() - start)

        incumbent_entry = _select_best_result(
            [r for r in pass1_results if _is_result_passed(r)])
        if incumbent_entry:
            incumbent_code = self._code_for_result(incumbent_entry, None)
            self._write_code(incumbent_code)
            logger.info(
                'Best-so-far after Stage 1 pass 1: %s robust=%s min=%s',
                _result_name(incumbent_entry),
                incumbent_entry.get('robust_score'),
                incumbent_entry.get('min_speedup'))

        top_for_deep = self._top_candidates_for_deep_pass(
            pass1_results, candidates, start)
        if torch_available and top_for_deep:
            logger.info('STAGE 1 PASS 2: Deep benchmark for %d candidates: %s',
                        len(top_for_deep), [c.name for c in top_for_deep])

        for candidate in top_for_deep:
            if not self._has_time_for_work(
                deadline, MIN_STAGE1_PASS2_SECONDS,
                f'Stage 1 pass 2 candidate {candidate.name}'):
                break
            if not self._has_time_for_work(
                stage1_deadline, MIN_STAGE1_PASS2_SECONDS,
                f'Stage 1 pass 2 candidate {candidate.name}'):
                break
            source_path = self._write_candidate(
                candidate,
                self.harness.build_dir / f'{candidate.name}_{candidate.fingerprint}_deep.cu',
            )
            try:
                result = self.harness.evaluate_config(
                    candidate, source_path,
                    benchmark_sizes=self.benchmark_sizes,
                    warmup=self.benchmark_warmup,
                    iters=self.stage1_pass2_iters,
                )
                result['status'] = 'pass2_deep_benchmarked'
                result['benchmark_pass'] = 2
                result['benchmark_warmup'] = self.benchmark_warmup
                result['benchmark_iters'] = self.stage1_pass2_iters
                stage1_history.append(result)
                pass2_results.append(result)
                self._log_candidate_result('Stage 1 pass 2', candidate.name,
                                           result)
                if _is_result_passed(result):
                    best_now = _select_best_result(
                        [e for e in (incumbent_entry, result) if e])
                    if best_now is result or best_now == result:
                        incumbent_entry = result
                        incumbent_code = self._code_for_result(result, None)
                        self._write_code(incumbent_code)
                        logger.info(
                            'Best-so-far updated during Stage 1 pass 2: '
                            '%s robust=%s min=%s',
                            _result_name(result), result.get('robust_score'),
                            result.get('min_speedup'))
            except Exception as exc:
                logger.exception('Candidate %s failed in pass 2', candidate.name)
                result = self._error_result(candidate, 'pass2_error', exc)
                stage1_history.append(result)
                pass2_results.append(result)

        logger.info('Elapsed after Stage 1 pass 2: %.1fs',
                    time.time() - start)

        selection_pool = [r for r in pass2_results if _is_result_passed(r)]
        if not selection_pool:
            selection_pool = [r for r in pass1_results if _is_result_passed(r)]

        if (torch_available and pass2_results and
                self._has_time_for_work(deadline, MIN_SHAPE_AWARE_SECONDS,
                                        'shape-aware dispatch benchmark') and
                self._has_time_for_work(stage1_deadline,
                                        MIN_SHAPE_AWARE_SECONDS,
                                        'shape-aware dispatch benchmark')):
            shape_result, shape_aware_code = self._try_shape_aware_dispatch(
                pass2_results)
            if shape_result:
                stage1_history.append(shape_result)
                if _is_result_passed(shape_result):
                    selection_pool.append(shape_result)

        best_entry = _select_best_result(selection_pool)
        best_code: Optional[str] = None
        if best_entry:
            best_code = self._code_for_result(best_entry, shape_aware_code)
            self._write_code(best_code)
            logger.info(
                'Stage 1 selected %s robust=%.6f geom=%.6f min=%.6f',
                _result_name(best_entry),
                best_entry.get('robust_score', best_entry.get('score', 0.0)),
                best_entry.get('geometric_mean', 0.0),
                best_entry.get('min_speedup', 0.0),
            )
        else:
            best_entry = {
                'candidate': initial_candidate.to_dict(),
                'passed': None if not torch_available else False,
                'score': None if not torch_available else 0.0,
                'geometric_mean': None if not torch_available else 0.0,
                'arithmetic_mean': None if not torch_available else 0.0,
                'min_speedup': None if not torch_available else 0.0,
                'max_speedup': None if not torch_available else 0.0,
                'robust_score': None if not torch_available else 0.0,
                'status': ('heuristic_fallback' if not torch_available
                           else 'no_valid_candidate'),
            }
            best_code = LoRACodegen.render(initial_candidate)
            self._write_code(best_code)

        best_score = best_entry.get('robust_score', best_entry.get('score', 0.0)) or 0.0

        # =============================================================
        # Stage 2: LLM-driven iterative improvement
        # =============================================================
        stage2_history: list[dict[str, Any]] = []

        if (self.use_llm and torch_available and best_code and
                best_score > 0):
            logger.info('=' * 60)
            logger.info('STAGE 2: LLM Iterative Improvement')
            logger.info('Current best score: %.6f', best_score)
            logger.info('=' * 60)

            llm_optimizer = LLMIterativeOptimizer(
                harness=self.harness,
                build_dir=self.harness.build_dir,
                optimized_path=self.optimized_path,
                deadline=deadline,
                max_iterations=self.stage2_max_iters,
                min_iteration_seconds=self.min_stage2_iteration_seconds,
            )
            stage2_history = llm_optimizer.run(
                current_best_code=best_code,
                current_best_entry=best_entry,
                stage1_history=stage1_history,
                start_time=start,
            )

            # Update best_entry if Stage 2 improved
            for h in stage2_history:
                if h.get('is_new_best') and _is_stage2_improvement(h, best_entry):
                    best_entry = h
                    best_score = h.get('robust_score', h.get('score', 0.0))
        else:
            if not self.use_llm:
                logger.info('Stage 2 skipped (--skip-llm)')
            elif not torch_available:
                logger.info('Stage 2 skipped (torch unavailable)')
            elif best_score <= 0:
                logger.info('Stage 2 skipped (no valid baseline)')
            else:
                logger.info('Stage 2 skipped (no best code)')

        stage2_attempted = sum(
            1 for h in stage2_history
            if h.get('iteration') is not None and
            h.get('status') != 'time_budget_stop'
        )
        stage2_skipped_due_time = sum(
            int(h.get('skipped_iterations', 0))
            for h in stage2_history
            if h.get('status') == 'time_budget_stop'
        )
        logger.info(
            'Stage 2 iterations attempted=%d skipped_due_time=%d max=%d',
            stage2_attempted, stage2_skipped_due_time, self.stage2_max_iters)

        # =============================================================
        # Final summary
        # =============================================================
        elapsed = time.time() - start
        logger.info(
            'Final elapsed %.1fs; best=%s robust=%s geom=%s min=%s',
            elapsed, _result_name(best_entry),
            best_entry.get('robust_score', best_entry.get('score')),
            best_entry.get('geometric_mean'),
            best_entry.get('min_speedup'))
        summary = {
            'phase': 'phase2',
            'gpu_info': gpu_info,
            'torch_available': torch_available,
            'stage1_candidates': len(candidates),
            'stage2_iterations': stage2_attempted,
            'stage2_skipped_due_time': stage2_skipped_due_time,
            'benchmark_sizes': self.benchmark_sizes,
            'time_budget_seconds': self.time_budget_seconds,
            'hard_budget_seconds': hard_budget_seconds,
            'safety_margin_seconds': self.safety_margin_seconds,
            'effective_budget_seconds': effective_budget_seconds,
            'deadline_epoch': round(deadline, 3),
            'stage1_deadline_epoch': round(stage1_deadline, 3),
            'stage2_max_iterations': self.stage2_max_iters,
            'minimum_stage2_iteration_seconds': self.min_stage2_iteration_seconds,
            'stage1_pass1_iters': self.stage1_pass1_iters,
            'stage1_pass2_iters': self.stage1_pass2_iters,
            'stage1_topk': self.stage1_topk,
            'elapsed_seconds': round(elapsed, 3),
            'best_candidate': best_entry,
            'stage1_history': stage1_history,
            'stage2_history': stage2_history,
            'analysis': self._generate_analysis(
                stage1_history, stage2_history, best_entry),
            'optimized_lora_path': str(self.optimized_path),
        }

        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        self._write_summary_markdown(summary)
        return summary

    # -----------------------------------------------------------------
    # Stage 1 selection helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _error_result(candidate: CandidateConfig, status: str,
                      exc: Exception) -> dict[str, Any]:
        return {
            'candidate': candidate.to_dict(),
            'passed': False,
            'score': 0.0,
            'geometric_mean': 0.0,
            'arithmetic_mean': 0.0,
            'min_speedup': 0.0,
            'max_speedup': 0.0,
            'robust_score': 0.0,
            'status': status,
            'error': str(exc),
            'per_size': [],
        }

    @staticmethod
    def _seconds_left(deadline: float) -> float:
        return deadline - time.time()

    def _has_time_for_work(self, deadline: float, minimum_seconds: int,
                           label: str) -> bool:
        left = self._seconds_left(deadline)
        if left < minimum_seconds:
            logger.info(
                '%s skipped/stopped due to time budget: '
                'time_left=%.1fs minimum_required=%ds',
                label, left, minimum_seconds)
            return False
        return True

    @staticmethod
    def _log_candidate_result(prefix: str, name: str,
                              result: dict[str, Any]) -> None:
        logger.info(
            '%s | %s: passed=%s robust=%s geom=%s min=%s max=%s',
            prefix, name, result.get('passed'),
            result.get('robust_score'), result.get('geometric_mean'),
            result.get('min_speedup'), result.get('max_speedup'),
        )

    def _top_candidates_for_deep_pass(
        self,
        pass1_results: list[dict[str, Any]],
        candidates: list[CandidateConfig],
        start_time: float,
    ) -> list[CandidateConfig]:
        if not pass1_results:
            return []
        ranked = _rank_results(pass1_results)
        if not ranked:
            return []

        elapsed = time.time() - start_time
        top_k = self.stage1_topk
        if elapsed > self.time_budget_seconds * 0.45:
            top_k = 2
        top_k = max(1, min(top_k, len(ranked)))

        by_name = {c.name: c for c in candidates}
        selected = []
        for entry in ranked:
            name = _result_name(entry)
            candidate = by_name.get(name)
            if candidate is not None:
                selected.append(candidate)
            if len(selected) >= top_k:
                break
        return selected

    @staticmethod
    def _config_from_entry(entry: dict[str, Any]) -> Optional[CandidateConfig]:
        cand = entry.get('candidate', {})
        template_name = cand.get('template_name')
        if template_name not in TEMPLATE_LIBRARY:
            return None
        return CandidateConfig(
            template_name=template_name,
            block_size=cand.get('block_size'),
            rationale=cand.get('rationale', 'selected from benchmark'),
        )

    def _code_for_result(self, entry: dict[str, Any],
                         shape_aware_code: Optional[str]) -> str:
        template_name = entry.get('candidate', {}).get('template_name')
        if template_name == 'shape_aware_dispatch':
            if not shape_aware_code:
                raise RuntimeError('shape-aware result selected without code')
            return shape_aware_code
        config = self._config_from_entry(entry)
        if config is None:
            raise RuntimeError(f'No renderable code for {_result_name(entry)}')
        return LoRACodegen.render(config)

    def _write_code(self, code: str) -> None:
        self.optimized_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.optimized_path.with_suffix(self.optimized_path.suffix + '.tmp')
        tmp.write_text(code, encoding='utf-8')
        tmp.replace(self.optimized_path)

    def _try_shape_aware_dispatch(
        self, pass2_results: list[dict[str, Any]]
    ) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        try:
            intervals = self._build_shape_aware_intervals(pass2_results)
            if not intervals:
                logger.info('Shape-aware dispatch skipped: no valid intervals')
                return None, None
            unique = {item['candidate'].name for item in intervals}
            if len(unique) < 2:
                logger.info(
                    'Shape-aware dispatch skipped: one strategy wins all bands')
                return None, None

            code = LoRACodegen.render_shape_aware_dispatch(intervals)
            fp = hashlib.md5(code.encode()).hexdigest()[:8]
            source_path = self.harness.build_dir / f'shape_aware_{fp}.cu'
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(code, encoding='utf-8')
            result = self.harness.evaluate(
                'shape_aware_dispatch', source_path, fp,
                benchmark_sizes=self.benchmark_sizes,
                warmup=self.benchmark_warmup,
                iters=self.stage1_pass2_iters,
            )
            result['candidate'] = {
                'name': 'shape_aware_dispatch',
                'template_name': 'shape_aware_dispatch',
                'block_size': None,
                'rationale': 'Band dispatch over safe Stage 1 candidates',
                'dispatch_intervals': [
                    {
                        'label': item['label'],
                        'upper_bound': item['upper_bound'],
                        'candidate': item['candidate'].name,
                        'band_metrics': item.get('metrics', {}),
                    }
                    for item in intervals
                ],
            }
            result['status'] = 'shape_aware_benchmarked'
            result['benchmark_pass'] = 'shape_aware'
            self._log_candidate_result('Stage 1 shape-aware',
                                       'shape_aware_dispatch', result)
            return result, code
        except Exception as exc:
            logger.exception('Shape-aware dispatch generation failed')
            return {
                'candidate': {
                    'name': 'shape_aware_dispatch',
                    'template_name': 'shape_aware_dispatch',
                },
                'passed': False,
                'score': 0.0,
                'geometric_mean': 0.0,
                'arithmetic_mean': 0.0,
                'min_speedup': 0.0,
                'max_speedup': 0.0,
                'robust_score': 0.0,
                'status': 'shape_aware_error',
                'error': str(exc),
                'per_size': [],
            }, None

    def _build_shape_aware_intervals(
        self, pass2_results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        passed = [entry for entry in pass2_results if _is_result_passed(entry)]
        if len(passed) < 2:
            return []

        intervals: list[dict[str, Any]] = []
        lower_exclusive: Optional[int] = None
        for label, upper in SHAPE_AWARE_BANDS:
            band_sizes = [
                d for d in self.benchmark_sizes
                if (lower_exclusive is None or d > lower_exclusive)
                and (upper is None or d <= upper)
            ]
            lower_exclusive = upper
            if not band_sizes:
                continue

            band_entries = []
            band_set = set(band_sizes)
            for entry in passed:
                per_size = [
                    item for item in entry.get('per_size', [])
                    if item.get('d') in band_set
                ]
                if len(per_size) != len(band_sizes):
                    continue
                metrics = LoRABenchmarkHarness._summarize_speedups(per_size)
                band_entry = {
                    **entry,
                    **metrics,
                    'score': metrics['robust_score'],
                    'per_size': per_size,
                }
                band_entries.append(band_entry)

            selected = _select_best_result(band_entries)
            if selected is None:
                continue
            config = self._config_from_entry(selected)
            if config is None:
                continue

            item = {
                'label': label,
                'upper_bound': upper,
                'candidate': config,
                'metrics': {
                    'robust_score': selected.get('robust_score'),
                    'geometric_mean': selected.get('geometric_mean'),
                    'min_speedup': selected.get('min_speedup'),
                },
            }
            if intervals and intervals[-1]['candidate'].name == config.name:
                intervals[-1]['upper_bound'] = upper
            else:
                intervals.append(item)

        return intervals

    # -----------------------------------------------------------------
    # Candidate library
    # -----------------------------------------------------------------

    def _candidate_library(self) -> list[CandidateConfig]:
        return [
            CandidateConfig(
                template_name='baseline_sequential',
                block_size=None,
                rationale='Single-stream baseline: at::mm + at::addmm, '
                          'no overlap, minimal overhead.',
            ),
            CandidateConfig(
                template_name='dual_stream_overlap',
                block_size=None,
                rationale='Overlaps W@X (main) with B^T@X (aux stream). '
                          'Sync via CUDA event before addmm.',
            ),
            CandidateConfig(
                template_name='dual_stream_pretranspose',
                block_size=None,
                rationale='Dual-stream + explicit B^T contiguous copy '
                          'before mm for better memory access.',
            ),
            CandidateConfig(
                template_name='dual_stream_mm_add',
                block_size=None,
                rationale='Dual-stream + mm + add_ instead of addmm_out '
                          'for final accumulation.',
            ),
            CandidateConfig(
                template_name='triple_stream_prealloc',
                block_size=None,
                rationale='Three streams with pre-allocated outputs. '
                          'W@X and B^T@X fully parallel.',
            ),
            CandidateConfig(
                template_name='sequential_prealloc',
                block_size=None,
                rationale='Single stream + mm_out with pre-allocated '
                          'tensors to avoid allocator overhead.',
            ),
            CandidateConfig(
                template_name='dual_stream_prealloc',
                block_size=None,
                rationale='Dual-stream + pre-allocated outputs via mm_out. '
                          'Combines overlap with zero-alloc.',
            ),
            CandidateConfig(
                template_name='dual_stream_prealloc_contig',
                block_size=None,
                rationale='Dual-stream + pre-alloc + contiguous B^T. '
                          'Tests contig + mm_out combo.',
            ),
        ]

    @staticmethod
    def _baseline_candidate(candidates: list[CandidateConfig]) -> CandidateConfig:
        for c in candidates:
            if c.template_name == 'baseline_sequential':
                return c
        return candidates[0]

    # -----------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------

    def _generate_analysis(self, stage1_history: list, stage2_history: list,
                           best_entry: dict) -> str:
        fallback = self._fallback_analysis(
            stage1_history, stage2_history, best_entry)
        if not self.use_llm:
            return fallback

        client = _get_llm()
        if client is None:
            return fallback

        lines = []
        lines.append('Stage 1 results:')
        for e in stage1_history:
            c = e.get('candidate', {})
            lines.append(
                f"  - {c.get('name')}: robust={e.get('robust_score')} "
                f"geom={e.get('geometric_mean')} min={e.get('min_speedup')} "
                f"status={e.get('status')}")
        if stage2_history:
            lines.append('\nStage 2 (LLM iterations):')
            for h in stage2_history:
                lines.append(f"  - iter {h.get('iteration')}: "
                             f"robust={h.get('robust_score')} "
                             f"status={h.get('status')} "
                             f"new_best={h.get('is_new_best')}")

        try:
            return client.generate_reasoning(
                system_prompt=(
                    'Summarise a two-stage LoRA CUDA optimisation search. '
                    'Stage 1 tried fixed templates; Stage 2 used LLM-driven '
                    'iterative improvement. Explain what worked, what did not, '
                    'and where the performance ceiling comes from. '
                    '4-6 concise sentences.'
                ),
                user_prompt=(
                    f"Best: {json.dumps(best_entry, sort_keys=True, default=str)}"
                    f"\n\n{''.join(lines)}"
                ),
            ).strip()
        except Exception as exc:
            logger.warning('LLM analysis failed: %s', exc)
            return fallback

    @staticmethod
    def _fallback_analysis(s1_history, s2_history, best_entry):
        name = best_entry.get('candidate', {}).get('name', 'unknown')
        robust = best_entry.get('robust_score', best_entry.get('score'))
        geom = best_entry.get('geometric_mean')
        min_speedup = best_entry.get('min_speedup')
        s2_count = len(s2_history)
        s2_improved = sum(1 for h in s2_history if h.get('is_new_best'))
        return (
            f"Best variant: {name} (robust {robust}, "
            f"geomean {geom}, min {min_speedup}). "
            f"Stage 1 evaluated {len(s1_history)} templates. "
            f"Stage 2 ran {s2_count} LLM iterations, "
            f"{s2_improved} produced improvements. "
            f"The dense W@X GEMM dominates runtime; "
            f"low-rank (r=16) path contributes marginal overhead."
        )

    # -----------------------------------------------------------------
    # File I/O
    # -----------------------------------------------------------------

    def _write_candidate(self, candidate: CandidateConfig,
                         destination: Optional[Path] = None) -> Path:
        destination = destination or self.optimized_path
        source = LoRACodegen.render(candidate)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = destination.with_suffix(destination.suffix + '.tmp')
        tmp_path.write_text(source, encoding='utf-8')
        tmp_path.replace(destination)
        return destination

    def _write_summary_markdown(self, summary: dict[str, Any]) -> None:
        best = summary.get('best_candidate', {})
        lines = [
            '# Phase 2 LoRA Optimization Summary', '',
            f'- Elapsed: {summary.get("elapsed_seconds")}s',
            f'- Torch available: {summary.get("torch_available")}',
            f'- Benchmark sizes: {summary.get("benchmark_sizes")}',
            f'- Stage 1 candidates: {summary.get("stage1_candidates")}',
            f'- Stage 2 iterations: {summary.get("stage2_iterations")}',
            f'- Output: {summary.get("optimized_lora_path")}',
            '',
            '## Best Candidate', '',
            f'- Name: {best.get("candidate", {}).get("name")}',
            f'- Status: {best.get("status")}',
            f'- Robust score: {best.get("robust_score", best.get("score"))}',
            f'- Geometric mean: {best.get("geometric_mean")}',
            f'- Min speedup: {best.get("min_speedup")}',
            '',
            '## Stage 1: Template Search', '',
        ]
        for entry in summary.get('stage1_history', []):
            cand = entry.get('candidate', {})
            bits = []
            for s in entry.get('per_size', []):
                if s.get('passed'):
                    bits.append(f"d={s['d']}:{s.get('speedup')}x")
                else:
                    bits.append(f"d={s['d']}:FAILED")
            suffix = f" ({', '.join(bits)})" if bits else ''
            lines.append(
                f"- {cand.get('name')}: robust={entry.get('robust_score')} "
                f"geom={entry.get('geometric_mean')} "
                f"min={entry.get('min_speedup')} "
                f"status={entry.get('status')}{suffix}")

        s2 = summary.get('stage2_history', [])
        if s2:
            lines.extend(['', '## Stage 2: LLM Iterative Improvement', ''])
            for h in s2:
                it = h.get('iteration', '?')
                sc = h.get('robust_score', h.get('score', 0))
                st = h.get('status', '?')
                nb = h.get('is_new_best', False)
                bits = []
                for s in h.get('per_size', []):
                    if s.get('passed'):
                        bits.append(f"d={s['d']}:{s.get('speedup')}x")
                    else:
                        bits.append(f"d={s['d']}:FAILED")
                suffix = f" ({', '.join(bits)})" if bits else ''
                lines.append(
                    f"- Iteration {it}: robust={sc} status={st}"
                    f"{' **NEW BEST**' if nb else ''}{suffix}")

        lines.extend([
            '', '## Analysis', '',
            summary.get('analysis', ''),
        ])
        self.summary_path.write_text('\n'.join(lines) + '\n',
                                     encoding='utf-8')
