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

build_with_script_py() {
    local pkg="$1"
    echo "  -> script/build.py $pkg"
    if [ -n "$CLEAN" ]; then
        "$ROOT_DIR/script/build.py" --clean "$pkg"
    else
        "$ROOT_DIR/script/build.py" "$pkg"
    fi
}

build_legacy_upptst() {
    local test="$1"
    if [ ! -d "$ROOT_DIR/upptst/$test" ]; then
        echo "  ✗ Directory upptst/$test not found"
        return 1
    fi
    if ! command -v umk >/dev/null 2>&1; then
        echo "  ✗ U++ build system (umk) not found. Please ensure U++ is installed."
        return 1
    fi
    umk upptst,$HOME/upp/uppsrc,$HOME/upp/bazaar,src "$test" CLANG.bm -bdsa${CLEAN} +CONSOLE,DEBUG_FULL "bin/${test}"
}

build_runtime_deps() {
    local deps="$1"
    local IFS=','
    read -r -a dep_arr <<< "$deps"
    for dep in "${dep_arr[@]}"; do
        dep="$(echo "$dep" | xargs)"
        [ -z "$dep" ] && continue
        build_with_script_py "$dep" || return 1
    done
    return 0
}

CLEAN=""
if [ "${1:-}" = "--clean" ]; then
    CLEAN="a"
    shift
fi

if [ $# -gt 0 ]; then
    TESTS=("$@")
    echo "Building specific test(s): ${TESTS[*]} (clean=$CLEAN)"
else
    mapfile -t TESTS < <(manifest_default_tests)
    echo "Building all ConvNetCpp unit tests from manifest... (clean=$CLEAN)"
fi

OVERALL_RESULT=0

for test in "${TESTS[@]}"; do
    echo "Building $test..."
    if ! manifest_has "$test"; then
        echo "  ✗ Test not listed in manifest: $test"
        OVERALL_RESULT=1
        continue
    fi

    kind="$(manifest_field "$test" 2)"
    package="$(manifest_field "$test" 3)"
    deps="$(manifest_field "$test" 4)"

    if [ "$kind" = "legacy" ]; then
        if build_legacy_upptst "$test"; then
            echo "  ✓ $test built successfully"
        else
            echo "  ✗ $test build failed"
            OVERALL_RESULT=1
        fi
    elif [ "$kind" = "package" ]; then
        [ -z "$package" ] && package="$test"
        if build_with_script_py "$package"; then
            echo "  ✓ $test built successfully"
        else
            echo "  ✗ $test build failed"
            OVERALL_RESULT=1
        fi
    elif [ "$kind" = "python" ]; then
        if [ -n "$deps" ]; then
            if build_runtime_deps "$deps"; then
                echo "  ✓ $test runtime dependencies built"
            else
                echo "  ✗ $test runtime dependency build failed"
                OVERALL_RESULT=1
            fi
        else
            echo "  ✓ $test has no build dependencies"
        fi
    else
        echo "  ✗ Unknown manifest kind '$kind' for $test"
        OVERALL_RESULT=1
    fi
done

if [ $OVERALL_RESULT -eq 0 ]; then
    echo "Build process completed successfully."
else
    echo "Build process completed with failures (exit=$OVERALL_RESULT)."
fi
exit $OVERALL_RESULT
