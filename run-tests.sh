#!/bin/bash

# Script to run all ConvNetCpp unit tests
echo "Running ALL ConvNetCpp unit tests..."

# Create bin directory if it doesn't exist
mkdir -p bin

# Run each test executable
TESTS=("CharGenTest" "Regression1DTest" "Classify2DTest" "SimpleGANTest" "GANTest" "RegressionPainterTest" "ClassifyImagesTest" "GridWorldTest" "HeteroscedasticUncertaintyTest" "MartingaleTest" "NetworkOptimizationTest" "PuckWorldTest" "ReinforcedLearningTest" "TemporalDifferenceTest" "TrainerBenchmarkTest" "WaterWorldTest")

for test in "${TESTS[@]}"; do
    echo "========================================"
    echo "Running $test..."
    echo "========================================"
    
    if [ -f "bin/$test" ]; then
        "./bin/$test"
        result=$?
        if [ $result -eq 0 ]; then
            echo "✓ $test PASSED"
        else
            echo "✗ $test FAILED with exit code $result"
        fi
    else
        echo "✗ $test executable not found in bin/"
        echo "  Try: ./build-tests.sh $test"
    fi
    echo ""
done

echo "All tests completed!"