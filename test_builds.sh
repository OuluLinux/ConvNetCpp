#!/bin/bash

# Temporary script to test building all examples and identify which ones actually compile

# Create bin directory if it doesn't exist
mkdir -p bin

# Arrays to track results
successful_builds=()
failed_builds=()

# Get all subdirectories in examples directory
EXAMPLES_DIR="examples"
for example_dir in "$EXAMPLES_DIR"/*/; do
    if [ -d "$example_dir" ]; then
        example_name=$(basename "$example_dir")
        echo "Testing build for example: $example_name"
        
        # Try to build the example using build.sh
        if timeout 60s ./scripts/build.sh "$example_name" > "build_log_$example_name.txt" 2>&1; then
            # Check if binary was created
            if [ -f "bin/$example_name" ]; then
                echo "✓ Successfully built: $example_name"
                successful_builds+=("$example_name")
            else
                echo "✗ Build reported success but binary not found: $example_name"
                failed_builds+=("$example_name")
            fi
        else
            echo "✗ Failed to build: $example_name"
            failed_builds+=("$example_name")
        fi
        echo ""
    fi
done

echo "Build testing completed."
echo "Successful builds: ${#successful_builds[@]}"
echo "Failed builds: ${#failed_builds[@]}"
echo ""
echo "Examples that built successfully: ${successful_builds[*]}"
echo "Examples that failed to build: ${failed_builds[*]}"

# Clean up log files
for log_file in build_log_*.txt; do
    rm "$log_file"
done