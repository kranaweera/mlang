# MLang v0.6.0 — Experimental CUDA Milestone

Lightweight machine-learning compiler: native CPU inference, small dense-network model imports, and a **first genuine CUDA GPU kernel path for matrix multiplication**.

## Mac (Apple Silicon) — CPU tests and CUDA source generation

```sh
cargo test
cargo build --release
./target/release/mlang build examples/matmul.ml --output matmul
./matmul
./target/release/mlang build examples/matmul.ml --target cuda --emit-only --emit-c matmul.cu
./scripts/verify_cuda.sh
```

`--emit-only` generates CUDA C++ code on your Mac but **does not run CUDA**. The generated `.cu` file is intended for an NVIDIA system.

## NVIDIA Linux host — compile and execute on GPU

Install NVIDIA drivers, CUDA Toolkit (`nvcc`), and a compatible GPU. Clone/copy the project onto that host, then:

```sh
cargo build --release
./target/release/mlang build examples/matmul.ml --target cuda --output matmul_gpu
./matmul_gpu
./scripts/verify_cuda.sh
```

Expected matrix output: `[[19, 22], [43, 50]]`.

`--cc /path/to/nvcc` overrides the CUDA compiler location. The CPU backend still defaults to Clang.

## Scope and limitations

- `@` matrix-matrix and matrix-vector operators generate CUDA `__global__` kernels. CUDA allocations and host/device copies occur for each operation; this is **functionally GPU-executed**, not performance optimized.
- Elementwise operations, `relu`, `sigmoid`, `dot`, print and parsing remain on CPU in this release. GPU kernel fusion, persistent device buffers, memory pooling and CPU/GPU performance wins are not claimed.
- Uses CUDA **runtime API** (`cuda_runtime.h`, `nvcc`) rather than the previously envisioned Driver-API-only deployment. Build requires the CUDA Toolkit; executing the built program requires compatible CUDA runtime libraries and NVIDIA drivers. It is not a standalone driver-only binary.
- CUDA is not available on current macOS systems. Run real CUDA tests on a compatible Linux or Windows NVIDIA machine.
- CPU builds remain unchanged; Python/PyTorch are not needed for deployment.
- CUDA compiler/toolchain was not available in the authoring environment, so GPU compilation/execution must be verified externally. The generated CUDA source is experimental.

## Commands

```sh
mlang import examples/sample_model.mmdl --output generated.ml
mlang build generated.ml --target cpu --output model_cpu
mlang build generated.ml --target cuda --output model_cuda
mlang build examples/matmul.ml --target cuda --emit-only --emit-c matmul.cu
```

## Future CUDA milestones

Persistent GPU buffers, GPU-side activations and fusion, PTX/cubin offline compilation, optional Driver API host stub, accurate GPU memory/latency benchmarks, and test matrices across compute capabilities.
