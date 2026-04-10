#!/bin/bash
# GPU Hardware Intrinsic Profiling Agent - Runner Script
#
# Usage:
#   ./run.sh                          # Run with default target_spec.json
#   ./run.sh --target-spec custom.json
#   ./run.sh --kernel ./my_binary     # Analyze a CUDA kernel
#   ./run.sh --trials 10              # More measurement trials

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo " GPU Hardware Intrinsic Profiling Agent"
echo "=================================================="

# Check prerequisites
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

echo "CUDA compiler: $(nvcc --version | tail -1)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# Create build directory
mkdir -p build

# Run the agent
python3 agent.py "$@"

echo ""
echo "Done. Check results.json and reasoning.log for output."
