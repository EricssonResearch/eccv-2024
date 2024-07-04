#!/bin/bash
set -euo pipefail
build_dir=_build

# Display information
echo "Configuring build"
echo "  - Build directory: $build_dir"

# Change directory depending on whether it is debug or release mode
mkdir -p $build_dir && cd $build_dir
cmake ..

# Build (for Make on Unix equivalent to `make -j $(nproc)`)
if [[ "$OSTYPE" == "darwin"* ]]; then
  # Mac OSX does not have nproc
  cmake --build . -- -j $(sysctl -n hw.logicalcpu)
else
  cmake --build . -- -j $(nproc)
fi

