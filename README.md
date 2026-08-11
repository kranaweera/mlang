# MLang v0.1 — Milestone 1

A tiny, dependency-free **Rust compiler** for a small, statically typed ML-oriented language. It generates C11 source code and invokes Clang to produce a native executable. The generated program does **not** require Rust, Python, PyTorch or TensorFlow.

**Status:** first prototype. CPU only. CUDA is intentionally **not implemented** yet; passing `--target cuda` emits a clear error instead of silently compiling for CPU. The compiler is not production-hardened.

## Requirements on macOS

1. Install Rust using [rustup](https://rustup.rs/) (includes `cargo`).
2. Install Apple's Command Line Tools: `xcode-select --install` (provides `clang`).
3. Check `cargo --version` and `clang --version`.

No external Rust crates are used.

## Build

```sh
cargo build --release
cargo test
./target/release/mlang build examples/hello.ml --output hello
./hello
./target/release/mlang build examples/linear_regression.ml --output predictor --emit-c predictor.c
./predictor
```

Expected outputs: `14` and `18` respectively.

You can also run via Cargo:

```sh
cargo run -- build examples/linear_regression.ml --output predictor
```

## Supported language (v0.1)

- `fn name(args...) -> f32 { ... }` functions; `main()` with no return type
- `let variable = expression;` local bindings
- `return expression;`
- `f32` scalar and fixed-size `f32[N]` vectors as function parameters
- vector literals `[1.0, 2.0]` (maximum 4096 elements)
- arithmetic `+ - * /`, unary `-`, parentheses
- `dot(vec, vec)` and `print(scalar)`
- `//` line comments
- semicolons are required

Important limitations: this release does **not** train models, read CSV files, perform matrix multiplication, support mutable variables or conditional execution, support strings or tensor shapes beyond one dimension, or emit CUDA code. Only scalar arithmetic is allowed outside `dot()`. Printed values have up to 9 significant digits. The parser accepts a function name that is a C keyword because all generated names are prefixed with `ml_`. Operations use native `float` and may have ordinary floating-point rounding error.

## Examples

```ml
fn predict(x: f32[3]) -> f32 {
    let weights = [0.5, 0.3, 0.2];
    let bias = 1.0;
    return dot(x, weights) + bias;
}

fn main() {
    print(predict([10.0, 20.0, 30.0]));
}
```

## CLI

```text
mlang build <file.ml> [--output <binary>] [--emit-c <file.c>] [--cc <compiler>] [--target cpu|cuda]
```

Default output is `a.out`, target `cpu`, and C compiler `clang`. The generated C is preserved as `<output>.c` unless another path is given using `--emit-c`.

## Architecture

```text
src/lexer.rs        text -> tokens
src/parser.rs       tokens -> AST
src/ast.rs          AST node definitions
src/semantic.rs     name resolution and type checks
src/backend/cpu.rs  AST -> C11
src/backend/cuda.rs planned future implementation (not executable)
src/main.rs         CLI and native C compilation
```

**Next architectural improvement:** introduce a typed, device-neutral IR between semantic analysis and the backends, then add constant folding, tensor operators, shape validation and memory planning before attempting PTX/CUDA. A C host runtime for CUDA will use the NVIDIA Driver API, with GPU code linked/loaded separately. Inference-first; training and autodiff are later milestones.

## Tests

`cargo test` runs lexer/parser/semantic unit tests plus integration tests that use Clang to build and execute two sample programs and verify that a mismatched `dot` is rejected. `clang` must be on PATH.

## Security and support

Generated C should be treated as a build artifact. Do not compile untrusted MLang programs in a privileged environment; the compiler prototype has not been fuzzed or security audited.
