# MLang v0.7.0 — CUDA Buffer Reuse and Fused ReLU

Builds on v0.6; maintains CPU functionality, model import, typed IR, and tensor operations.

## New
- Reuses CUDA device buffers **per matrix multiplication tensor signature**, rather than allocating/freeing on every call. Allocations are released on process exit via `atexit`.
- Fuses **direct** `relu(A @ B)` into the matrix multiplication CUDA kernel; the kernel applies ReLU before writing its result.
- New `examples/cuda_fused_relu.ml` and `scripts/verify_cuda_v07.sh` for generation, CPU correctness, and optional CUDA execution checks.

## Run on Mac
```sh
cargo test
cargo build --release
./scripts/verify_cuda_v07.sh
```
CUDA source generation works on macOS; NVIDIA CUDA **execution** needs compatible Linux or Windows with `nvcc`, driver, CUDA runtime and supported NVIDIA GPU.

## Run on NVIDIA host
```sh
cargo build --release
./target/release/mlang build examples/cuda_fused_relu.ml --target cuda --output fused_gpu
./fused_gpu
# [[0, 4], [6, 0]]
```

## Current limitations (important)
- Device buffers are cached per **operator shape**, not across different GPU ops or entire inference graphs. Inputs and outputs are still copied for **every** matmul call, so there is **no end-to-end GPU residency**.
- Only direct `relu(matmul)` is fused; e.g., `relu(A @ B + bias)` **is not fused**. Other tensor arithmetic and activations still run on CPU.
- `nvcc` and CUDA Runtime API are required for CUDA compilation and CUDA runtime libraries at deployment. This is not yet a minimal Driver API/fully AOT backend.
- Buffer cache is process-global, not thread-safe; this prototype is **not safe for concurrent inference calls**.
- Performance improvements are **hypotheses**, not verified gains. Compare latency, allocations and binary size on NVIDIA hardware.

The compiler is experimental and is not validated for clinical or safety-critical deployment.
