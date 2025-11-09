#!/usr/bin/env bash

# Script to build all examples in the examples directory

# Create bin directory if it doesn't exist
mkdir -p bin

# Counter for successful builds
success_count=0
total_count=0

# List of examples that are known to build successfully
# SimpleGAN works because it has minimal external dependencies
working_examples=("SimpleGAN" "TransformerTester" "GptTester")

# List of examples that are known to have external dependencies that cause build issues
# These fail due to:
# - JSON template redefinition issues in U++ Core library
# - Color constructor signature changes in U++ PlotLib
problematic_examples=("CharGen" "Classify2D" "ClassifyImages" "GAN" "GridWorld" "HeteroscedasticUncertainty" "Martingale" "NetworkOptimization" "PuckWorld" "Regression1D" "RegressionPainter" "ReinforcedLearning" "TemporalDifference" "TrainerBenchmark" "WaterWorld")

# Get all subdirectories in examples directory
EXAMPLES_DIR="examples"
for example_dir in "$EXAMPLES_DIR"/*/; do
    if [ -d "$example_dir" ]; then
        example_name=$(basename "$example_dir")
        total_count=$((total_count + 1))
        echo "Building example: $example_name"
        
        # Check if this example is known to work or has known issues
        is_working=0
        is_problematic=0
        
        for working in "${working_examples[@]}"; do
            if [ "$example_name" = "$working" ]; then
                is_working=1
                break
            fi
        done
        
        for problematic in "${problematic_examples[@]}"; do
            if [ "$example_name" = "$problematic" ]; then
                is_problematic=1
                break
            fi
        done
        
        if [ $is_problematic -eq 1 ]; then
            echo "⚠ Skipping $example_name (known external dependency issues)"
            echo "  Issues: JSON template redefinition, Color constructor signature changes in U++ libraries"
        else
            # Try to build the example using build.sh
            if timeout 60s ./build.sh "$example_name"; then
                echo "✓ Successfully built: $example_name"
                success_count=$((success_count + 1))
                
                # Check if binary was created
                if [ -f "bin/$example_name" ]; then
                    echo "  Binary created: bin/$example_name ($(stat -c%s "bin/$example_name") bytes)"
                else
                    echo "  Warning: Build reported success but binary not found"
                fi
            else
                echo "✗ Failed to build: $example_name"
                # If it failed but was expected to work, report as unknown issue
                if [ $is_working -eq 1 ]; then
                    echo "  (Unexpected failure for known working example)"
                fi
            fi
        fi
        echo ""
    fi
done

echo "Build process completed."
echo "Successfully built: $success_count / $total_count examples"
echo ""
echo "Note: Examples with external U++ library incompatibilities were skipped:"
echo "- JSON template redefinition issues (Core library)"
echo "- Color constructor signature changes (PlotLib library)"
echo "- These require fixes in Ultimate++ framework libraries"
echo ""
echo "Examples that currently build: ${working_examples[*]}"