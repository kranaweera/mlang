#!/bin/sh
set -eu
cargo test
cargo build --release
mkdir -p benchmark-out
./target/release/mlang build examples/cuda_fused_relu.ml --target cuda --emit-only --emit-c benchmark-out/fused.cu
grep -q 'ml_matmul_relu_m2_2_m2_2' benchmark-out/fused.cu
grep -q 'atexit(ml_release_m2_2_m2_2)' benchmark-out/fused.cu
./target/release/mlang build examples/cuda_fused_relu.ml --output benchmark-out/fused_cpu
actual="$(benchmark-out/fused_cpu)"
[ "$actual" = '[[0, 4], [6, 0]]' ] || { echo "CPU mismatch: $actual"; exit 1; }
echo 'CPU result and CUDA source checks PASS'
if command -v nvcc >/dev/null 2>&1 && command -v nvidia-smi >/dev/null 2>&1; then
  ./target/release/mlang build examples/cuda_fused_relu.ml --target cuda --output benchmark-out/fused_gpu
  gpu="$(benchmark-out/fused_gpu)"
  [ "$gpu" = "$actual" ] || { echo "CUDA mismatch: $gpu vs $actual"; exit 1; }
  echo 'GPU result matches CPU'
else
  echo 'CUDA execution skipped: NVIDIA GPU/nvcc not available'
fi
