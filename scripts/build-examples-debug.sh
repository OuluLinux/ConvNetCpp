#!/usr/bin/env bash

# Script to build all examples in the examples directory - with debug to see actual errors

# Check for --clean argument
CLEAN_FLAG=""
if [ "$1" = "--clean" ]; then
    CLEAN_FLAG="a"
    shift
elif [ "$1" = "--no-clean" ]; then
    CLEAN_FLAG=""
    shift
fi

# Create bin directory if it doesn't exist
mkdir -p bin

# Counter for successful builds
success_count=0
total_count=0

# List of examples that might work as console applications
console_examples=()

# List of examples that are GUI applications (require X11 environment to build)
gui_examples=("SimpleGAN" "TransformerTester" "GptTester")

# List of examples that are known to have external dependencies that cause build issues
# These fail due to:
# - JSON template redefinition issues in U++ Core library
# - Color constructor signature changes in U++ PlotLib
problematic_examples=("CharGen" "Classify2D" "ClassifyImages" "GAN" "GridWorld" "HeteroscedasticUncertainty" "Martingale" "NetworkOptimization" "PuckWorld" "Regression1D" "RegressionPainter" "ReinforcedLearning" "TemporalDifference" "TrainerBenchmark" "WaterWorld")

# Function to determine if an example is GUI based on its .upp file
is_gui_example() {
    local example_dir="$1"
    local example_name="$2"
    local upp_file="$example_dir$example_name.upp"

    if [ ! -f "$upp_file" ]; then
        return 1  # Not a GUI example if .upp doesn't exist
    fi

    # Check if the .upp file has GUI-related configurations
    if grep -q "CtrlLib\|mainconfig.*GUI\|USE_GUI\|=.*GUI" "$upp_file"; then
        return 0  # It's a GUI example
    else
        return 1  # It's not a GUI example (likely console)
    fi
}

# Get all subdirectories in examples directory
EXAMPLES_DIR="examples"
for example_dir in "$EXAMPLES_DIR"/*/; do
    if [ -d "$example_dir" ]; then
        example_name=$(basename "$example_dir")
        total_count=$((total_count + 1))
        echo "Building example: $example_name (clean=$CLEAN_FLAG)"

        # Check if this example is known to work or has known issues
        is_console=0
        is_gui=0
        is_problematic=0

        for console in "${console_examples[@]}"; do
            if [ "$example_name" = "$console" ]; then
                is_console=1
                break
            fi
        done

        for gui in "${gui_examples[@]}"; do
            if [ "$example_name" = "$gui" ]; then
                is_gui=1
                break
            fi
        done

        for problematic in "${problematic_examples[@]}"; do
            if [ "$example_name" = "$problematic" ]; then
                is_problematic=1
                break
            fi
        done

        # Now try to build ALL examples, regardless of known issues
        # Check if this is GGUFGUI and skip it as it requires special handling
        if [ "$example_name" = "GGUFGUI" ]; then
            echo "⚠ Skipping $example_name (GUI example that requires different build configuration)"
            echo "  GUI examples require separate build process"
        else
            # Use the is_gui_example function to detect if this is a GUI example
            if is_gui_example "$example_dir" "$example_name"; then
                echo "⚠ Skipping $example_name (detected as GUI example from .upp file, requires X11 environment to build)"
                echo "  GUI examples require display server for compilation, skipping in headless mode"
            else
                # Try to locate the .upp file in the example directory
                upp_file="$example_dir$(basename "$example_dir").upp"
                if [ -f "$upp_file" ]; then
                    # Use umk to build the example as console application
                    if command -v umk >/dev/null 2>&1; then
                        echo "Attempting build of $example_name as CONSOLE (even with known issues)..."
                        if timeout 120s umk examples,$HOME/upp/uppsrc,$HOME/upp/bazaar,src "$example_name" ./CLANG.bm -bds${CLEAN_FLAG} +CONSOLE,DEBUG_FULL "bin/${example_name}"; then
                            echo "✓ Successfully built: $example_name"
                            success_count=$((success_count + 1))

                            # Check if binary was created
                            if [ -f "bin/$example_name" ]; then
                                echo "  Binary created: bin/$example_name ($(stat -c%s "bin/$example_name") bytes)"
                            else
                                echo "  Warning: Build reported success but binary not found"
                            fi
                        else
                            echo "✗ Failed to build: $example_name as CONSOLE"
                            # The main issue is that many "console" examples are actually GUI examples
                            # Let's try to build it as GUI instead if it contains GUI configurations
                            echo "  (Attempting to build as GUI app instead...)"
                            if timeout 120s umk examples,$HOME/upp/uppsrc,$HOME/upp/bazaar,src "$example_name" ./CLANG.bm -bds${CLEAN_FLAG} +GUI,DEBUG_FULL "bin/${example_name}"; then
                                echo "✓ Successfully built: $example_name (as GUI app)"
                                success_count=$((success_count + 1))

                                # Check if binary was created
                                if [ -f "bin/$example_name" ]; then
                                    echo "  Binary created: bin/$example_name ($(stat -c%s "bin/$example_name") bytes)"
                                else
                                    echo "  Warning: Build reported success but binary not found"
                                fi
                            else
                                echo "✗ Also failed to build: $example_name as GUI"
                                # Report the known issues for this example
                                if [ $is_problematic -eq 1 ]; then
                                    echo "  (Failed as expected due to known external dependency issues)"
                                    echo "  Issues: JSON template redefinition, Color constructor signature changes in U++ libraries"
                                elif [ $is_console -eq 1 ]; then
                                    echo "  (Unexpected failure for known working console example)"
                                else
                                    echo "  (Failed with unknown issue)"
                                fi
                            fi
                        fi
                    else
                        echo "✗ U++ build system (umk) not found. Please ensure U++ is installed."
                        echo "✗ Failed to build: $example_name"
                        if [ $is_console -eq 1 ]; then
                            echo "  (Unexpected failure for known working console example)"
                        fi
                    fi
                else
                    echo "✗ .upp file not found: $upp_file"
                    echo "✗ Failed to build: $example_name"
                    # If it was expected to work but .upp file is missing, report as unknown issue
                    if [ $is_console -eq 1 ]; then
                        echo "  (Unexpected failure for known working console example - missing .upp file)"
                    fi
                fi
            fi
        fi
        echo ""
    fi
done

echo "Build process completed."
echo "Successfully built: $success_count / $total_count examples"
echo ""
if [ $success_count -eq 0 ]; then
    echo "Since no examples built, we'll need to fix the following issues:"
    echo "- JSON template redefinition issues (Core library)"
    echo "- Color constructor signature changes (PlotLib library)"
    echo "- These require fixes in the source code to be compatible with current U++ framework"
fi
echo ""
echo "Examples that currently build: Console - ${console_examples[*]}"
echo "GUI examples (require X11): ${gui_examples[*]}"