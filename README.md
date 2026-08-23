# MLang v0.2 — Milestone 2 (CPU Tensor Engine)

MLang is an experimental dependency-free Rust compiler that emits C11 and builds a small native executable with Clang. Deployed programs require no Python, PyTorch, TensorFlow, or Rust installation. macOS Apple Silicon is the primary development target.

## Setup and test

```sh
source "$HOME/.cargo/env"
cargo test
cargo build --release
./target/release/mlang build examples/matmul.ml --output matmul
./matmul
./target/release/mlang build examples/tensor_ops.ml --output tensor_ops
./tensor_ops
./target/release/mlang build examples/matvec.ml --output matvec
./matvec
ls -lh matmul tensor_ops matvec
```

Expected results:

- `matmul`: `[[19, 22], [43, 50]]`
- `tensor_ops`: `[[4, 8], [10, 4]]` and `[0.5, 0.5]`
- `matvec`: `[17, 39]`
- `linear_regression`: `18` (backward compatible with milestone 1)

## Implemented in this milestone

- Numeric `f32`, 1D `f32[N]`, and 2D `f32[R,C]` tensors.
- Nested fixed-size matrix literals, e.g. `[[1,2],[3,4]]`.
- `+`, `-`, `*`, `/` element-wise for equal-shaped tensors; scalar arithmetic remains available. No implicit broadcasting.
- `@` matrix–matrix and matrix–vector multiplication.
- `dot(vector, vector)` unchanged.
- `relu(tensor)` and `sigmoid(tensor)` for scalars, vectors or matrices.
- `print` supports scalars, vectors and matrices.
- Compile-time shape validation and ragged-literal rejection.
- Contiguous stack-allocated C structs for tensors; specialized loops are emitted only for operators actually used by a program.
- User-defined functions may accept and return fixed-size tensors.
- Zero third-party Rust crates.

Example:

```ml
fn main() {
    let A = [[1.0, 2.0], [3.0, 4.0]];
    let B = [[5.0, 6.0], [7.0, 8.0]];
    print(A @ B);
}
```

## Known limitations

This is still a tiny prototype. The element-count limit is 4096 per tensor; only rank 1–2 is supported. No dynamic dimensions, broadcasting, slicing, loading models, automatic differentiation, training, BLAS, allocation planner or GPU execution. Generated C uses simple triple-loop matrix multiplication, which is not yet optimized for large tensors. Tensor expressions are values passed via C structs and can be copied. `sigmoid` uses C math library (`-lm`). CUDA remains a deliberately unsupported planned backend (`--target cuda`).

Important: despite the original Milestone 2 ambition, **this version does not yet implement a standalone typed IR**; the AST is type-checked and emitted directly as C. A typed, device-independent IR and explicit buffer planning are required before adding CUDA. This scope keeps the initial tensor implementation small and easier to validate.

## CLI

```text
mlang build <file.ml> [--output <binary>] [--emit-c <file.c>] [--cc <compiler>] [--target cpu|cuda]
```

The existing `examples/hello.ml`, `examples/linear_regression.ml` and Milestone 1 tests remain included.

## CUDA roadmap

1. Lower type-checked operations to a device-neutral tensor IR.
2. Define a backend interface for CPU and NVIDIA GPU implementations.
3. Add PTX generation and CUDA Driver API host stubs, with explicit backend errors.
4. Validate all operators and numerical tolerances against the CPU reference.

Only the CPU backend is executable today. Do not compile untrusted source in a privileged environment; this compiler has not undergone security review or fuzzing.
