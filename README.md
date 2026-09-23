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

## Milestone 5 — portable dense model import (v0.5.0)

MLang now imports a small **data-only** `.mmdl` format and emits ordinary `.ml`
source that embeds the weights. It supports stacks of dense `f32` layers with
`relu` or `linear` activations and a maximum of 4,096 elements per tensor.
This is *not* general ONNX import, arbitrary scikit-learn import, or CUDA.

```sh
cargo test
cargo build --release
./target/release/mlang import examples/sample_model.mmdl --output imported.ml
./target/release/mlang build imported.ml --output imported
./imported
./scripts/verify_import.sh
```

Expected `./imported`: `[1.5]` for the generated zero input; a second verification
probe with `[2, 3]` should produce `[5]`.

### Model interchange specification (MLANG-MODEL 1)

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

Each dense layer specifies the output width. `weights` has exactly `output * input`
finite numbers in row-major `(output, input)` order; `bias` contains `output`
finite numbers. Layers chain in file order. Generated source uses embedded weights;
no model parser or Python runtime ships in the deployed binary. The generated
`main` demonstrates inference on a *zero input*; edit it or invoke `predict()`
with your own input to do useful predictions. No preprocessing/scaling metadata
is serialized, so apply the same preprocessing as training externally.

### Exporting an existing scikit-learn MLPRegressor

In a separate, trusted training environment where scikit-learn and joblib are
installed, train an `MLPRegressor` with `activation='relu'` or `'identity'`.
Save it with `joblib.dump`, then run:

```sh
python3 tools/export_sklearn_mlp.py trusted_model.joblib model.mmdl
./target/release/mlang import model.mmdl --output model.ml
./target/release/mlang build model.ml --output model_native
```

The exporter handles **only** `MLPRegressor`, not classifiers, and writes the
matrix transpose required by MLang. Never load untrusted joblib/pickle files.
For numerical comparison, use the same unscaled/scaled input in both runtimes;
small differences are expected due to f32 arithmetic. The importer rejects
invalid widths, missing values, unsupported activations, and non-finite weights.

### Benchmarking

```sh
ls -lh model_native
/usr/bin/time -l ./model_native  # macOS
```

Executable size varies by target and model; the previously measured 33 KB was
for tiny examples, not a guarantee for imported models. CPU inference only.
