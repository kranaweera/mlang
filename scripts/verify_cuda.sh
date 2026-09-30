#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
cargo test
cargo build --release
mkdir -p benchmark-out
./target/release/mlang build examples/matmul.ml --target cuda --emit-only --emit-c benchmark-out/matmul.cu
grep -q '__global__ void ml_kernel_m2_2_m2_2' benchmark-out/matmul.cu
if command -v nvcc >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1; then
  ./target/release/mlang build examples/matmul.ml --target cuda --output benchmark-out/matmul-gpu
  result=$(./benchmark-out/matmul-gpu)
  test "$result" = '[[19, 22], [43, 50]]'
  echo "CUDA GPU execution PASS: $result"
else
  echo 'GPU runtime test SKIPPED (requires NVIDIA Linux/Windows host with nvcc and compatible GPU)'
fi
