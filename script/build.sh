#!/usr/bin/sh
# Build script for ConvNetCpp
# Note: If you encounter Color constructor errors in PlotLib, run patch_plotlib.sh first
CLEAN_FLAG=""
TARGET=""

for arg in "$@"; do
    if [ "$arg" = "--clean" ]; then
        CLEAN_FLAG="a"
    else
        TARGET="$arg"
    fi
done

echo umk upptst,examples,tutorial,src,$HOME/ai-upp/bazaar,$HOME/ai-upp/uppsrc "$TARGET" CLANG.bm -bds${CLEAN_FLAG} +GUI,DEBUG_FULL "bin/$TARGET"

umk upptst,examples,tutorial,src,$HOME/ai-upp/bazaar,$HOME/ai-upp/uppsrc "$TARGET" CLANG.bm -bds${CLEAN_FLAG} +GUI,DEBUG_FULL "bin/$TARGET"
