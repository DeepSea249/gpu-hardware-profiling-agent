#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PHASE="phase2"
TARGET_SPEC="${TARGET_SPEC:-/target/target_spec.json}"
PHASE1_OUTPUT="${PHASE1_OUTPUT:-$ROOT_DIR/output.json}"
PHASE2_OUTPUT="${PHASE2_OUTPUT:-$ROOT_DIR/phase2_results.json}"
OPTIMIZED_PATH="${OPTIMIZED_PATH:-$ROOT_DIR/optimized_lora.cu}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
SEARCH_ROUNDS="${LORA_SEARCH_ROUNDS:-8}"
TIME_BUDGET_MINUTES="${LORA_TIME_BUDGET_MINUTES:-20}"
BENCHMARK_SIZES="${LORA_BENCHMARK_SIZES:-3584,4096,4608}"
BENCHMARK_WARMUP="${LORA_BENCHMARK_WARMUP:-5}"
BENCHMARK_ITERS="${LORA_BENCHMARK_ITERS:-15}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --target-spec)
            TARGET_SPEC="$2"
            shift 2
            ;;
        --output)
            PHASE2_OUTPUT="$2"
            shift 2
            ;;
        --optimized-path)
            OPTIMIZED_PATH="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --search-rounds)
            SEARCH_ROUNDS="$2"
            shift 2
            ;;
        --time-budget-minutes)
            TIME_BUDGET_MINUTES="$2"
            shift 2
            ;;
        --benchmark-sizes)
            BENCHMARK_SIZES="$2"
            shift 2
            ;;
        --benchmark-warmup)
            BENCHMARK_WARMUP="$2"
            shift 2
            ;;
        --benchmark-iters)
            BENCHMARK_ITERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "$PHASE" == "auto" ]]; then
    if [[ -f "$TARGET_SPEC" ]]; then
        PHASE="phase1"
    else
        PHASE="phase2"
    fi
fi

echo "================================================================="
echo " GPU Hardware Profiling + LoRA Optimization Agent"
echo " Workspace   : $ROOT_DIR"
echo " Selected    : $PHASE"
echo "================================================================="

echo "Ensuring Python dependencies are present ..."
python3 -m pip install openai python-dotenv tenacity \
    -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    --default-timeout=30 \
    --quiet \
    2>&1 || true

NVCC_CMD="$(command -v nvcc || true)"
if [[ -n "$NVCC_CMD" ]]; then
    echo "CUDA compiler : $($NVCC_CMD --version 2>/dev/null | tail -1)"
else
    echo "WARNING: nvcc not found on PATH"
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "GPU           : $GPU_NAME"

if [[ "$PHASE" == "phase1" ]]; then
    echo "Target spec   : $TARGET_SPEC"
    echo "Output        : $PHASE1_OUTPUT"

    if [[ -n "$NVCC_CMD" ]] && ls kernels/*.cu >/dev/null 2>&1; then
        mkdir -p "$BUILD_DIR/kernels"
        for src in kernels/*.cu; do
            [[ -f "$src" ]] || continue
            name="$(basename "$src" .cu)"
            bin="$BUILD_DIR/kernels/$name"
            if [[ ! -f "$bin" || "$src" -nt "$bin" ]]; then
                echo "Compiling $src -> $bin"
                "$NVCC_CMD" -O3 -arch=native -o "$bin" "$src" 2>&1 \
                    || echo "WARNING: failed to compile $src (non-fatal)"
            fi
        done
    fi

    python3 agent.py \
        --phase phase1 \
        --target-spec "$TARGET_SPEC" \
        --output "$PHASE1_OUTPUT" \
        --build-dir "$BUILD_DIR"

    echo "================================================================="
    echo " Done."
    echo "   $PHASE1_OUTPUT"
    echo "================================================================="
    exit 0
fi

echo "Optimized file: $OPTIMIZED_PATH"
echo "Search log    : $PHASE2_OUTPUT"

python3 agent.py \
    --phase phase2 \
    --output "$PHASE2_OUTPUT" \
    --optimized-path "$OPTIMIZED_PATH" \
    --build-dir "$BUILD_DIR" \
    --search-rounds "$SEARCH_ROUNDS" \
    --time-budget-minutes "$TIME_BUDGET_MINUTES" \
    --benchmark-sizes "$BENCHMARK_SIZES" \
    --benchmark-warmup "$BENCHMARK_WARMUP" \
    --benchmark-iters "$BENCHMARK_ITERS"

echo "================================================================="
echo " Done."
echo "   $OPTIMIZED_PATH"
echo "   $PHASE2_OUTPUT"
echo "================================================================="
