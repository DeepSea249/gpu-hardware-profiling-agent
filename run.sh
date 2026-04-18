#!/bin/bash
###############################################################################
# GPU Hardware Intrinsic Profiling Agent — Evaluation Entry Point
#
# Evaluation environment layout (fixed):
#   /workspace/run.sh              — this script
#   /target/target_spec.json       — evaluation specification (read-only)
#   /workspace/output.json         — single consolidated output file
#
# The spec format:
#   { "targets": ["l1_latency_cycles", ...], "run": "build/kernels/matmul_naive" }
#   Both fields are optional.
###############################################################################
set -e

# Always run from /workspace so relative paths (build/, kernels/, src/) work
WORKSPACE="/workspace"
cd "$WORKSPACE"

echo "================================================================="
echo " GPU Hardware Intrinsic Profiling Agent"
echo " Entry point: /workspace/run.sh"
echo "================================================================="

# ---- Install any extra Python packages not in the base image -------------
echo "Ensuring Python dependencies are present ..."
pip3 install python-dotenv tenacity \
    -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    --default-timeout=30 \
    --quiet \
    2>&1 || true      # non-fatal if mirror is unreachable; packages may already exist

# ---- Environment check ----------------------------------------------------
NVCC_CMD=""
for candidate in nvcc /usr/local/cuda/bin/nvcc /usr/local/cuda-12/bin/nvcc; do
    if command -v "$candidate" &>/dev/null; then
        NVCC_CMD="$candidate"
        break
    fi
done

if [ -n "$NVCC_CMD" ]; then
    echo "CUDA compiler : $("$NVCC_CMD" --version 2>/dev/null | tail -1)"
else
    echo "WARNING: nvcc not found. Probe compilation will rely on LLM source cache."
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "GPU           : $GPU_NAME"
echo "Target spec   : /target/target_spec.json"
echo "Output        : /workspace/output.json"
echo ""

# ---- Pre-build sample CUDA kernels ----------------------------------------
# These are used when target_spec.json contains a "run" field pointing to
# build/kernels/<name>. Compilation is skipped if the binary already exists.
if [ -n "$NVCC_CMD" ] && ls kernels/*.cu 1>/dev/null 2>&1; then
    mkdir -p build/kernels
    for src in kernels/*.cu; do
        [ -f "$src" ] || continue
        name="$(basename "$src" .cu)"
        bin="build/kernels/$name"
        if [ ! -f "$bin" ] || [ "$src" -nt "$bin" ]; then
            echo "Compiling $src → $bin"
            # -arch=native selects the installed GPU's SM version automatically
            "$NVCC_CMD" -O3 -arch=native -o "$bin" "$src" 2>&1 \
                || echo "WARNING: failed to compile $src (non-fatal)"
        fi
    done
fi

# ---- Run the agent ---------------------------------------------------------
echo ""
echo "Launching agent ..."
echo "-----------------------------------------------------------------"

python3 agent.py \
    --target-spec /target/target_spec.json \
    --output      /workspace/output.json   \
    --build-dir   /workspace/build

echo ""
echo "================================================================="
echo " Done."
echo "   /workspace/output.json  — metrics + kernel analysis + reasoning"
echo "================================================================="
