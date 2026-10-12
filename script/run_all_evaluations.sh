#!/usr/bin/env bash
set -euo pipefail

echo "========================================================================"
echo "          ConvNetCpp Automated Verification & Benchmark Runner          "
echo "========================================================================"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "[Step 1/5] Building CUDA targets, benchmark runner, and evaluator..."
make clean
make -j4

echo ""
echo "[Step 2/5] Running Edge Case & Gradient Checking Test Suite..."
bin/edge_case_tests

echo ""
echo "[Step 3/5] Running Automated Evaluator Suite (22 Unit Tests)..."
bin/gpu_evaluator

echo ""
echo "[Step 4/5] Running GPU Benchmark (GTX 1060 Measured Throughput)..."
bin/gpu_benchmark

echo ""
echo "[Step 5/5] Running Anthology Task Evaluator on Reference Solution..."
python3 anthology_task/evaluator.py anthology_task/reference_solution.cu

echo ""
echo "========================================================================"
echo " [SUCCESS] All verification, evaluator, and benchmark suites passed!    "
echo "========================================================================"
