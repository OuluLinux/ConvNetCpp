#!/bin/bash

# Script to run specific ConvNetCpp unit tests
echo "Running specific ConvNetCpp unit tests..."

# Check if a test name was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <TestName>"
    echo "Example: $0 Classify2DTest"
    echo "Available tests: CharGenTest, Regression1DTest, Classify2DTest, SimpleGANTest, GANTest, RegressionPainterTest, ClassifyImagesTest, GridWorldTest, HeteroscedasticUncertaintyTest, MartingaleTest, NetworkOptimizationTest, PuckWorldTest, ReinforcedLearningTest, TemporalDifferenceTest, TrainerBenchmarkTest, WaterWorldTest, GGUFTest"
    exit 1
fi

TEST_NAME="$1"

# Create bin directory if it doesn't exist
mkdir -p bin

echo "========================================"
echo "Running $TEST_NAME..."
echo "========================================"

if [ -f "bin/$TEST_NAME" ]; then
    LD_LIBRARY_PATH=/home/sblo/Dev/ConvNetCpp/src/llama.cpp/build/bin:$LD_LIBRARY_PATH "./bin/$TEST_NAME"
    result=$?
    if [ $result -eq 0 ]; then
        echo "✓ $TEST_NAME PASSED"
    else
        echo "✗ $TEST_NAME FAILED with exit code $result"
    fi
else
    echo "✗ $TEST_NAME executable not found in bin/"
    echo "Did you build the test first? Try: ./build-tests.sh $TEST_NAME"
fi

echo "Test $TEST_NAME completed!"