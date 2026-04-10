#!/bin/bash
###############################################################################
# GPU Hardware Intrinsic Profiling Agent — Evaluation Entry Point
#
# This is the single command the evaluation system (or teacher) needs to run.
#
# Input:  target_spec.json   (placed in the project root before running)
#         The spec contains:
#           - "targets": list of hardware metrics to identify
#           - "run":     (optional) path to an executable to profile
#
# Output: results.json       (numeric values + reasoning + kernel analysis)
#         results_kernel_analysis.json   (detailed ncu bottleneck report)
#         results_kernel_report.md       (LLM-authored narrative)
#         reasoning.log                  (full step-by-step evidence)
#         agent.log                      (execution log)
#
# Usage:
#   ./run.sh                                     # Use default target_spec.json
#   ./run.sh --target-spec /path/to/spec.json    # Use a custom spec
#   ./run.sh --output /path/to/results.json      # Custom output path
#   ./run.sh --trials 10                         # More measurement trials
###############################################################################
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================="
echo " GPU Hardware Intrinsic Profiling Agent"
echo " Entry point: run.sh"
echo "================================================================="

# ---- Environment check ----------------------------------------------------
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
if ! command -v "$NVCC" &> /dev/null && ! command -v nvcc &> /dev/null; then
    echo "WARNING: nvcc not found. Probes that need compilation may fail."
else
    NVCC_ACTUAL="$(command -v "$NVCC" 2>/dev/null || command -v nvcc)"
    echo "CUDA compiler : $("$NVCC_ACTUAL" --version 2>/dev/null | tail -1)"
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "GPU           : $GPU_NAME"

# Show the spec file that will be used
SPEC="${1:-target_spec.json}"
# Handle --target-spec as first positional or named arg
for arg in "$@"; do
    case "$arg" in
        --target-spec) shift; SPEC="$1"; shift; break ;;
    esac
done
echo "Target spec   : $SPEC"
echo ""

# ---- Pre-build sample kernels if not yet compiled --------------------------
mkdir -p build/kernels

for src in kernels/*.cu; do
    [ -f "$src" ] || continue
    name="$(basename "$src" .cu)"
    bin="build/kernels/$name"
    if [ ! -f "$bin" ] || [ "$src" -nt "$bin" ]; then
        echo "Compiling $src ..."
        NVCC_CMD="${NVCC_ACTUAL:-/usr/local/cuda/bin/nvcc}"
        "$NVCC_CMD" -O3 -arch=sm_86 -o "$bin" "$src" 2>&1 || echo "WARNING: failed to compile $src"
    fi
done

# ---- Run the agent ---------------------------------------------------------
echo ""
echo "Launching agent ..."
echo "-----------------------------------------------------------------"
python3 agent.py "$@"

echo ""
echo "================================================================="
echo " Done. Output files:"
echo "   results.json                  — numeric metrics + reasoning"
echo "   results_kernel_analysis.json  — detailed ncu bottleneck data"
echo "   results_kernel_report.md      — LLM-authored analysis report"
echo "   reasoning.log                 — step-by-step evidence trail"
echo "   agent.log                     — execution log"
echo "================================================================="
