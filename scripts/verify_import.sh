#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
cargo test
cargo build --release
mkdir -p benchmark-out
./target/release/mlang import examples/sample_model.mmdl --output benchmark-out/imported.ml
./target/release/mlang build benchmark-out/imported.ml --output benchmark-out/imported
actual=$(benchmark-out/imported)
[ "$actual" = "[1.5]" ] || { echo "Unexpected result: $actual" >&2; exit 1; }
# Change input vector to test the trained network's nontrivial forward pass.
sed 's/predict(\[0.000000000, 0.000000000\])/predict([2.0, 3.0])/' benchmark-out/imported.ml > benchmark-out/imported_probe.ml
./target/release/mlang build benchmark-out/imported_probe.ml --output benchmark-out/imported_probe
probe=$(benchmark-out/imported_probe)
[ "$probe" = "[5]" ] || { echo "Unexpected probe: $probe" >&2; exit 1; }
echo "Import OK: zero=$actual, probe=$probe"
ls -lh benchmark-out/imported benchmark-out/imported_probe
