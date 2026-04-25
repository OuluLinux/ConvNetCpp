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

manifest_default_tests() {
    awk -F'|' '
        $0 !~ /^#/ && NF >= 2 { print $1 }
    ' "$MANIFEST_FILE"
}

echo "Running ConvNetCpp tests from manifest..."

mkdir -p "$ROOT_DIR/bin"

if [ $# -gt 0 ]; then
    TESTS=("$@")
else
    mapfile -t TESTS < <(manifest_default_tests)
fi

OVERALL_RESULT=0

for test in "${TESTS[@]}"; do
    if ! manifest_has "$test"; then
        echo "========================================"
        echo "Running $test..."
        echo "========================================"
        echo "✗ Test not listed in manifest: $test"
        OVERALL_RESULT=1
        echo ""
        continue
    fi

    kind="$(manifest_field "$test" 2)"
    echo "========================================"
    echo "Running $test..."
    echo "========================================"

    if [ "$kind" = "python" ]; then
        if [ -f "$ROOT_DIR/tests/$test.py" ]; then
            python3 "$ROOT_DIR/tests/$test.py"
            result=$?
            if [ $result -eq 0 ]; then
                echo "✓ $test PASSED"
            else
                echo "✗ $test FAILED with exit code $result"
                OVERALL_RESULT=$result
            fi
        else
            echo "✗ Python test not found: tests/$test.py"
            OVERALL_RESULT=1
        fi
    else
        if [ -f "$ROOT_DIR/bin/$test" ]; then
            "$ROOT_DIR/bin/$test"
            result=$?
            if [ $result -eq 0 ]; then
                echo "✓ $test PASSED"
            else
                echo "✗ $test FAILED with exit code $result"
                OVERALL_RESULT=$result
            fi
        else
            echo "✗ $test executable not found in bin/"
            echo "  Try: ./script/build-tests.sh $test"
            OVERALL_RESULT=1
        fi
    fi
    echo ""
done

if [ $OVERALL_RESULT -eq 0 ]; then
    echo "All tests completed successfully!"
else
    echo "Test run completed with failures (exit=$OVERALL_RESULT)"
fi
exit $OVERALL_RESULT
