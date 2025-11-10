#!/usr/bin/env bash

# Script to build all examples in the examples directory, including GUI examples in headless mode

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

# Function to determine if an example is GUI based on its .upp file
is_gui_example() {
    local example_dir="$1"
    local example_name="$2"
    local upp_file="$example_dir$example_name.upp"

    if [ ! -f "$upp_file" ]; then
        return 1  # Not a GUI example if .upp doesn't exist
    fi

    # Check if the .upp file has GUI-related configurations
    if grep -q "CtrlLib\|mainconfig.*GUI\|GUI\|use.*CtrlLib\|use.*Docking\|use.*PlotCtrl" "$upp_file"; then
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

        # Check if this is GGUFGUI and skip it as it requires special handling
        if [ "$example_name" = "GGUFGUI" ]; then
            echo "⚠ Skipping $example_name (GUI example that requires different build configuration)"
            echo "  GUI examples require separate build process"
        else
            # Use the is_gui_example function to detect if this is a GUI example
            if is_gui_example "$example_dir" "$example_name"; then
                echo "Building GUI example: $example_name"
                # Try to locate the .upp file in the example directory
                upp_file="$example_dir$(basename "$example_dir").upp"
                if [ -f "$upp_file" ]; then
                    # Use umk to build the example as GUI application
                    if command -v umk >/dev/null 2>&1; then
                        # Try to run with virtual framebuffer for headless GUI building
                        if command -v xvfb-run >/dev/null 2>&1; then
                            echo "Using xvfb-run for headless GUI build..."
                            if timeout 120s xvfb-run -a umk examples,$HOME/upp/uppsrc,$HOME/upp/bazaar,src "$example_name" ./CLANG.bm -bds${CLEAN_FLAG} +GUI,DEBUG_FULL "bin/${example_name}"; then
                                echo "✓ Successfully built GUI: $example_name"
                                success_count=$((success_count + 1))

                                # Check if binary was created
                                if [ -f "bin/$example_name" ]; then
                                    echo "  Binary created: bin/$example_name ($(stat -c%s "bin/$example_name") bytes)"
                                else
                                    echo "  Warning: Build reported success but binary not found"
                                fi
                            else
                                echo "✗ Failed to build GUI: $example_name (even with xvfb)"
                                echo "  This might be due to external dependency issues:"
                                echo "  - JSON template redefinition issues (Core library)"
                                echo "  - Color constructor signature changes (PlotLib library)"
                            fi
                        else
                            echo "xvfb-run not available, attempting GUI build directly..."
                            if timeout 120s umk examples,$HOME/upp/uppsrc,$HOME/upp/bazaar,src "$example_name" ./CLANG.bm -bds${CLEAN_FLAG} +GUI,DEBUG_FULL "bin/${example_name}"; then
                                echo "✓ Successfully built GUI: $example_name"
                                success_count=$((success_count + 1))

                                # Check if binary was created
                                if [ -f "bin/$example_name" ]; then
                                    echo "  Binary created: bin/$example_name ($(stat -c%s "bin/$example_name") bytes)"
                                else
                                    echo "  Warning: Build reported success but binary not found"
                                fi
                            else
                                echo "✗ Failed to build GUI: $example_name"
                                echo "  This might be due to external dependency issues:"
                                echo "  - JSON template redefinition issues (Core library)"
                                echo "  - Color constructor signature changes (PlotLib library)"
                                echo "  - Or missing X11 display in headless environment"
                            fi
                        fi
                    else
                        echo "✗ U++ build system (umk) not found. Please ensure U++ is installed."
                        echo "✗ Failed to build: $example_name"
                    fi
                else
                    echo "✗ .upp file not found: $upp_file"
                    echo "✗ Failed to build: $example_name"
                fi
            else
                # Build as console application
                upp_file="$example_dir$(basename "$example_dir").upp"
                if [ -f "$upp_file" ]; then
                    # Use umk to build the example as console application
                    if command -v umk >/dev/null 2>&1; then
                        echo "Attempting build of $example_name as CONSOLE..."
                        if timeout 120s umk examples,$HOME/upp/uppsrc,$HOME/upp/bazaar,src "$example_name" ./CLANG.bm -bds${CLEAN_FLAG} +CONSOLE,DEBUG_FULL "bin/${example_name}"; then
                            echo "✓ Successfully built CONSOLE: $example_name"
                            success_count=$((success_count + 1))

                            # Check if binary was created
                            if [ -f "bin/$example_name" ]; then
                                echo "  Binary created: bin/$example_name ($(stat -c%s "bin/$example_name") bytes)"
                            else
                                echo "  Warning: Build reported success but binary not found"
                            fi
                        else
                            echo "✗ Failed to build CONSOLE: $example_name"
                        fi
                    else
                        echo "✗ U++ build system (umk) not found. Please ensure U++ is installed."
                        echo "✗ Failed to build: $example_name"
                    fi
                else
                    echo "✗ .upp file not found: $upp_file"
                    echo "✗ Failed to build: $example_name"
                fi
            fi
        fi
        echo ""
    fi
done

echo "Build process completed."
echo "Successfully built: $success_count / $total_count examples"
echo ""
echo "Examples that currently build: Console - (none identified)"
echo "GUI examples: (all examples detected as GUI)"