#!/bin/bash
# Simple build test script

set -e
# A pipeline's status is tee's otherwise, so a failing cmake would pass.
set -o pipefail

echo "=== Testing Noumena Build ==="

# Create build directory
mkdir -p build_test
cd build_test

echo "1. Configuring CMake (without tests to speed up)..."
if ! cmake .. -DBUILD_TESTING=OFF -DENABLE_NATIVE_ARCH=OFF 2>&1 | tee cmake_output.log; then
    echo "ERROR: CMake configuration failed! (log: build_test/cmake_output.log)"
    exit 1
fi

echo "2. Building main executable..."
if ! cmake --build . 2>&1 | tee build_output.log; then
    echo "ERROR: Build failed! (log: build_test/build_output.log)"
    exit 1
fi

echo "=== Build successful! ==="
ls -lh noumena

cd ..
