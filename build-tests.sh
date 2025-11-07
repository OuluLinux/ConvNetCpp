#!/bin/bash

# Build script for ConvNetCpp unit tests following U++ upptst conventions

# Check for --clean argument
CLEAN=""
if [ "$1" = "--clean" ]; then
    CLEAN="a"
    shift
fi

# Build single test if argument provided, otherwise build all
if [ $# -gt 0 ]; then
    TESTS=("$@")
    echo "Building specific test(s): ${TESTS[*]} (clean=$CLEAN)"
else
    TESTS=("CharGenTest" "Regression1DTest" "Classify2DTest" "SimpleGANTest" "GANTest" "RegressionPainterTest" "ClassifyImagesTest" "GridWorldTest" "HeteroscedasticUncertaintyTest" "MartingaleTest" "NetworkOptimizationTest" "PuckWorldTest" "ReinforcedLearningTest" "TemporalDifferenceTest" "TrainerBenchmarkTest" "WaterWorldTest")
    echo "Building all ConvNetCpp unit tests... (clean=$CLEAN)"
fi

for test in "${TESTS[@]}"; do
    echo "Building $test..."
    if [ -d "upptst/$test" ]; then
        if command -v umk >/dev/null 2>&1; then
            # Use umk to build the test - call from project root with path to .upp file
            umk upptst,$HOME/upp/uppsrc,$HOME/upp/bazaar,src $test ~/.config/u++/theide/CLANG.bm -bds${CLEAN} +CONSOLE,DEBUG_FULL "bin/${test}"
            if [ $? -eq 0 ]; then
                echo "  ✓ $test built successfully"
            else
                echo "  ✗ $test build failed"
            fi
        else
            echo "  ✗ U++ build system (umk) not found. Please ensure U++ is installed."
        fi
    else
        echo "  ✗ Directory upptst/$test not found"
    fi
done

echo "Build process completed!"
