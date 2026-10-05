#!/bin/bash
# Simple build test script

set -e

echo "=== Testing Noumena Build ==="

# Create build directory
mkdir -p build_test
cd build_test

echo "1. Configuring CMake (without tests to speed up)..."
cmake .. -DBUILD_TESTING=OFF -DENABLE_NATIVE_ARCH=OFF 2>&1 | tee cmake_output.log

if [ $? -ne 0 ]; then
    echo "ERROR: CMake configuration failed!"
    cat cmake_output.log
    exit 1
fi

echo "2. Building main executable..."
cmake --build . 2>&1 | tee build_output.log

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed!"
    cat build_output.log
    exit 1
fi

echo "=== Build successful! ==="
ls -lh noumena

cd ..
