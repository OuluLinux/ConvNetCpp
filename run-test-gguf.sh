#!/bin/bash
# Script to run the GGUF test with proper library paths for llama.cpp integration

# Set the library path to include GGML libraries
export LD_LIBRARY_PATH=/home/sblo/Dev/ConvNetCpp/src/llama.cpp/build/bin:$LD_LIBRARY_PATH

echo "Running GGUF test with llama.cpp GGML library integration..."
echo "LD_LIBRARY_PATH is set to include: /home/sblo/Dev/ConvNetCpp/src/llama.cpp/build/bin"
echo ""

# Run the test
./bin/GGUFTest

result=$?
if [ $result -eq 0 ]; then
    echo ""
    echo "✓ GGUFTest PASSED - GGUF/llama.cpp integration working correctly!"
else
    echo ""
    echo "✗ GGUFTest FAILED with exit code $result"
    exit $result
fi