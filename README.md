# MLang v0.4.1 — Native inference compiler

Built on Milestone 3. CPU backend now emits C **directly from typed IR** (without IR-to-AST conversion). Includes a small 2→3→1 dense ReLU network using embedded or external weights.

## Run on macOS (Rust + Clang)

```sh
cargo test
cargo build --release
./target/release/mlang build examples/neural_embedded.ml --output neural_embedded
./neural_embedded                     # [5]
./target/release/mlang build examples/neural_external.ml --output neural_external
./neural_external < examples/neural_weights.txt  # [5]
./scripts/benchmark.sh
```

The external weight format is plain whitespace-separated FP32 decimal numbers, row-major, in `read_*()` invocation order. **The loader uses standard input**, so a weight file is supplied via redirection. Incorrect or truncated input exits with code 2; it is not yet a versioned model format.

## Limitations

* Fixed-shape rank <= 2 FP32 tensors, max 4096 elements, no broadcasting or dynamic allocation.
* `read_vN()` and `read_mR_C()` read numbers from stdin. They do not implement named-file loading or secure weights validation.
* Dense network uses existing matmul and ReLU operators, not fused kernels; no autodiff, training, CUDA, or operator fusion yet.
* Compiler targets macOS/Linux via Clang; no standalone runtime benchmark harness yet beyond process-start wall time.
* Embedded weights are handwritten numerical examples, not weights produced by a training pipeline.

CUDA remains an intentionally unavailable backend. Binary sizes need measurement on your target system.

## Size comparison

```sh
./scripts/neural_benchmark.sh
```

This reports file size and output (not inference-only latency). Compare against the 33 KB Milestone 3 baseline; process-level measurements do not represent isolated model latency.

## Milestone 4 verification (v0.4.1)

Run `./scripts/verify_neural.sh` to execute Rust tests, compile both network
variants, and compare their outputs to a separate Python standard-library
reference calculation. Python is **only used for development verification**;
the compiled inference binary does not depend on Python.

The external-weight loader now rejects truncated, malformed and non-finite
(`NaN`, `Inf`) values with exit code 2. This is still a prototype stdin
format, **not** a production-safe model file parser; extra trailing tokens
are not rejected, and no model versioning, checksums, or authentication exist.

The 2→3→1 network has **hand-chosen illustrative weights**, not weights
learned by a training algorithm. No training functionality is claimed.

A build/test run on macOS with Cargo and Clang is still required: the
packaging environment used for this release does not provide Cargo.
