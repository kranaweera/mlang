# MLang

**A lightweight, statically typed language for machine-learning inference, compiled to small native binaries.**

MLang is a dependency-free compiler written in Rust. It type-checks tensor shapes at compile time, lowers programs to a typed intermediate representation, and emits C11 (CPU) or CUDA C++ (GPU). The result is a self-contained executable of roughly 33 KB that needs no Python, PyTorch, TensorFlow or Rust at deployment time.

```ml
fn predict(x: f32[2]) -> f32[1] {
    let w1 = [[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]];
    let b1 = [0.0, 0.0, 1.0];
    let hidden = relu(w1 @ x + b1);
    let w2 = [[0.5, 0.25, 1.0]];
    let b2 = [0.5];
    return w2 @ hidden + b2;
}
fn main() { print(predict([2.0, 3.0])); }   // [5]
```

> **Status:** experimental (v0.7.0). Built to explore how small an ML inference toolchain can be. It is not production-hardened, fuzzed, or validated for safety-critical use.

## Features

- **Fixed-shape tensors**: `f32` scalars, `f32[N]` vectors and `f32[R,C]` matrices (up to 4096 elements), with every shape checked at compile time.
- **Tensor operators**: element-wise `+ - * /`, matrix multiplication `@` (matrix–matrix and matrix–vector), `dot`, `relu`, `sigmoid`.
- **Typed IR and optimisation**: a backend-neutral IR with scalar constant folding.
- **Native CPU backend**: emits C11 and builds with Clang. It only generates helper code for the operators a program actually uses.
- **External weights**: `read_vN()` / `read_mR_C()` load validated FP32 weights from stdin at run time.
- **Model import**: converts a simple text model format (`MLANG-MODEL 1`) for dense networks into MLang source. An exporter for scikit-learn `MLPRegressor` is included.
- **Experimental CUDA backend**: GPU matrix-multiplication kernels, reusable device buffers, and a fused `relu(A @ B)` kernel.
- **Zero third-party Rust crates.**

## Requirements

- Rust toolchain via [rustup](https://rustup.rs/) (`cargo`)
- Clang. On macOS, install it with `xcode-select --install`.
- *Optional, for running on a GPU:* an NVIDIA GPU with drivers and the CUDA Toolkit (`nvcc`) on Linux or Windows. On macOS you can still generate CUDA source with `--emit-only`.
- *Optional, for development checks only:* Python 3 (verification scripts and the scikit-learn exporter).

## Quick start

```sh
git clone https://github.com/kranaweera/mlang.git
cd mlang
cargo test
cargo build --release

./target/release/mlang build examples/matmul.ml --output matmul
./matmul
# [[19, 22], [43, 50]]
```

## CLI

```text
mlang build <file.ml> [--output <binary>] [--emit-c <file>] [--cc <compiler>]
                      [--target cpu|cuda] [--emit-only]
mlang import <model.mmdl> --output <file.ml>
```

| Option | Default | Description |
|---|---|---|
| `--output` | `a.out` | Path of the native executable |
| `--emit-c` | `<output>.c` / `<output>.cu` | Where to write the generated C or CUDA source |
| `--cc` | `clang` (CPU), `nvcc` (CUDA) | C or CUDA compiler to invoke |
| `--target` | `cpu` | `cpu` or `cuda` |
| `--emit-only` | off | Generate source without compiling it |

## Language guide

```ml
// Functions take and return fixed-size tensors; main() has no params or return type.
fn layer(w: f32[2,2], x: f32[2]) -> f32[2] {
    let z = w @ x;          // matrix–vector product
    return relu(z);
}

fn main() {
    let a = [[-1.0, 2.0], [3.0, -4.0]];   // nested literal -> f32[2,2]
    let b = [[5.0, 6.0], [7.0, 8.0]];
    print(relu(a + b));                    // [[4, 8], [10, 4]]
    print(sigmoid([0.0, 0.0]));            // [0.5, 0.5]
    print(dot([1.0, 2.0], [3.0, 4.0]));    // 11
    print(layer(a, [1.0, 1.0]));
}
```

| Construct | Notes |
|---|---|
| `fn name(arg: T, ...) -> T { ... }` | Types: `f32`, `f32[N]`, `f32[R,C]` |
| `let x = expr;` | Immutable bindings; semicolons required |
| `return expr;` | Required in functions with a return type |
| `+ - * /` | Scalars, or element-wise on tensors of identical shape (no broadcasting) |
| `@` | `f32[M,K] @ f32[K,N]` → `f32[M,N]`; `f32[M,K] @ f32[K]` → `f32[M]` |
| `dot(a, b)` | Two vectors of equal length → `f32` |
| `relu(t)`, `sigmoid(t)` | Scalars, vectors or matrices |
| `read_vN()`, `read_mR_C()` | Read an `f32[N]` / `f32[R,C]` from stdin |
| `print(expr)` | Prints scalars, vectors or matrices |
| `// comment` | Line comments |

Shape errors are caught at compile time:

```text
$ mlang build examples/shape_error.ml
mlang: matrix multiplication @ shape mismatch (expected matrix @ matrix or matrix @ vector)
```

## Neural network inference

There are two versions of the same 2→3→1 ReLU network. Its weights are hand-picked for illustration, not learned:

```sh
# Weights embedded in the binary
./target/release/mlang build examples/neural_embedded.ml --output neural_embedded
./neural_embedded                                   # [5]

# Weights loaded at run time from stdin (row-major, in read_*() call order)
./target/release/mlang build examples/neural_external.ml --output neural_external
./neural_external < examples/neural_weights.txt     # [5]
```

The external loader rejects truncated, malformed and non-finite (`NaN`/`Inf`) input with exit code 2. `scripts/verify_neural.sh` compares both binaries against an independent Python reference calculation.

## Model import

`mlang import` converts a data-only `.mmdl` file into ordinary MLang source with the weights embedded:

```text
MLANG-MODEL 1
input 2
dense 3 relu
weights 1 0 0 2 -1 1
bias 0 0 1
dense 1 linear
weights 0.5 0.25 1
bias 0.5
```

```sh
./target/release/mlang import examples/sample_model.mmdl --output imported.ml
./target/release/mlang build imported.ml --output imported
./imported                                          # [1.5] (zero input)
```

- Each `dense` layer gives its output width and a `relu` or `linear` activation.
- `weights` holds `output × input` finite values in row-major `(output, input)` order.
- `bias` holds `output` values.
- The generated `main` runs the model on a zero input. Edit it to call `predict()` with real data.
- Apply any input scaling or preprocessing yourself; it isn't stored in the model file.

To export a trained scikit-learn `MLPRegressor` (with `relu` or `identity` activation) in a trusted Python environment:

```sh
python3 tools/export_sklearn_mlp.py trusted_model.joblib model.mmdl
./target/release/mlang import model.mmdl --output model.ml
./target/release/mlang build model.ml --output model_native
```

Never load untrusted joblib/pickle files, because loading them can run arbitrary code.

## CUDA backend (experimental)

```sh
# Any platform: generate CUDA source only
./target/release/mlang build examples/matmul.ml --target cuda --emit-only --emit-c matmul.cu

# NVIDIA host with nvcc: build and run on the GPU
./target/release/mlang build examples/cuda_fused_relu.ml --target cuda --output fused_gpu
./fused_gpu                                         # [[0, 4], [6, 0]]
```

- `@` compiles to CUDA `__global__` kernels through the CUDA runtime API.
- Device buffers are allocated once per matmul shape signature, reused across calls, and freed at exit.
- A direct `relu(A @ B)` is fused into the matmul kernel.
- Element-wise ops, `dot`, `sigmoid`, unfused activations and I/O still run on the CPU.

## Architecture

```text
source.ml
   │  lexer.rs         text → tokens
   │  parser.rs        tokens → AST (precedence climbing)
   │  semantic.rs      name resolution, type and shape checking
   │  ir.rs            typed, backend-neutral IR + constant folding
   ├─ backend/cpu.rs   IR → C11 ──► clang ──► native binary
   └─ backend/cuda.rs  IR → CUDA C++ ──► nvcc ──► GPU binary

model.mmdl ── model_import.rs ──► source.ml
```

```text
src/
  main.rs            CLI (build / import)
  ast.rs  lexer.rs  parser.rs  semantic.rs  ir.rs  model_import.rs
  backend/cpu.rs     C emitter (also produces CUDA host code)
  backend/cuda.rs    CUDA entry point and status
examples/            sample .ml programs, weights and a sample .mmdl model
scripts/             benchmark and verification scripts
tools/               scikit-learn exporter
tests/               end-to-end tests (build → run → compare output)
```

## Testing and verification

```sh
cargo test                      # unit tests + end-to-end tests (requires clang)
./scripts/benchmark.sh          # binary size and wall time for the examples
./scripts/verify_neural.sh      # neural examples vs. Python reference
./scripts/verify_import.sh      # model import round-trip
./scripts/verify_cuda.sh        # CUDA source checks; runs on GPU if available
./scripts/verify_cuda_v07.sh    # fused ReLU + buffer reuse checks
```

The benchmark timings include process start-up, so they aren't inference-latency microbenchmarks.

## Project history

Each milestone is tagged in the repository:

| Version | Milestone |
|---|---|
| `v0.1.0` | Scalar/vector compiler: lexer, parser, type checker, C11 backend, `dot`, CLI |
| `v0.2.0` | CPU tensor engine: matrices, `@`, element-wise ops, `relu`/`sigmoid`, shape checking |
| `v0.3.0` | Typed IR between semantic analysis and the backend; scalar constant folding |
| `v0.4.1` | Direct IR → C emission; neural network examples; validated stdin weight loading |
| `v0.5.0` | `mlang import` for `MLANG-MODEL 1` dense networks; scikit-learn exporter |
| `v0.6.0` | Experimental CUDA backend: GPU matmul kernels, `--target cuda`, `--emit-only` |
| `v0.7.0` | CUDA device-buffer reuse and fused `relu(A @ B)` kernel |

## Limitations

- Only rank-1 and rank-2 FP32 tensors with fixed shapes, up to 4096 elements each. No broadcasting, slicing or dynamic shapes.
- Inference only: no training, automatic differentiation, control flow or mutable variables.
- CPU matmul is a simple triple loop and isn't tuned for large tensors. Tensors are passed by value as C structs.
- CUDA has no end-to-end GPU residency, because inputs and outputs are copied on every matmul. Only a direct `relu(matmul)` is fused, and only matmul runs on the GPU.
- The CUDA buffer cache is process-global and not thread-safe.
- CUDA builds need the CUDA Toolkit, and the resulting binaries need the CUDA runtime and drivers. CUDA execution isn't available on macOS.
- The stdin weight format and `.mmdl` have no versioning beyond the header, and no checksums or authentication.

## Roadmap

- GPU-resident tensors across whole inference graphs and GPU-side activations
- More operator fusion (for example `relu(A @ B + bias)`)
- Offline PTX/cubin compilation and an optional CUDA Driver API host
- Broadcasting, more activations, and a richer model import format
- Accurate latency and memory benchmarks on CPU and GPU

## Security

Treat generated C/CUDA as a build artifact. Don't compile untrusted MLang programs or models in a privileged environment: the compiler hasn't been fuzzed or security audited.

## License

MIT, as declared in `Cargo.toml`.
