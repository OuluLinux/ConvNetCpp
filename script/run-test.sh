#!/bin/bash

set -u

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MANIFEST_FILE="$ROOT_DIR/script/test-manifest.txt"

if [ ! -f "$MANIFEST_FILE" ]; then
    echo "Missing test manifest: $MANIFEST_FILE"
    exit 1
fi

manifest_field() {
    local test="$1"
    local idx="$2"
    awk -F'|' -v t="$test" -v i="$idx" '
        $0 !~ /^#/ && NF >= 2 && $1 == t { print $i; exit }
    ' "$MANIFEST_FILE"
}

manifest_has() {
    local test="$1"
    awk -F'|' -v t="$test" '
        $0 !~ /^#/ && NF >= 2 && $1 == t { found=1; exit }
        END { exit(found ? 0 : 1) }
    ' "$MANIFEST_FILE"
}

if [ $# -eq 0 ]; then
    echo "Usage: $0 <TestName>"
    echo "Available tests:"
    awk -F'|' '$0 !~ /^#/ && NF >= 2 { print "  - " $1 }' "$MANIFEST_FILE"
    exit 1
fi

TEST_NAME="$1"
echo "Running specific ConvNetCpp test..."
echo "========================================"
echo "Running $TEST_NAME..."
echo "========================================"

if ! manifest_has "$TEST_NAME"; then
    echo "✗ Test not listed in manifest: $TEST_NAME"
    exit 1
fi

kind="$(manifest_field "$TEST_NAME" 2)"
if [ "$kind" = "python" ]; then
    if [ ! -f "$ROOT_DIR/tests/$TEST_NAME.py" ]; then
        echo "✗ Python test not found: tests/$TEST_NAME.py"
        exit 1
    fi
    python3 "$ROOT_DIR/tests/$TEST_NAME.py"
    result=$?
else
    if [ ! -f "$ROOT_DIR/bin/$TEST_NAME" ]; then
        echo "✗ $TEST_NAME executable not found in bin/"
        echo "Did you build the test first? Try: ./script/build-tests.sh $TEST_NAME"
        exit 1
    fi
    result=$?
fi

if [ $result -eq 0 ]; then
    echo "✓ $TEST_NAME PASSED"
else
    echo "✗ $TEST_NAME FAILED with exit code $result"
fi
echo "Test $TEST_NAME completed!"
exit $result
