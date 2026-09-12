#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
cargo test
cargo build --release
mkdir -p benchmark-out
for name in neural_embedded neural_external; do
  ./target/release/mlang build "examples/$name.ml" --output "benchmark-out/$name"
done
expected=$(python3 scripts/reference_neural.py)
embedded=$(benchmark-out/neural_embedded)
external=$(benchmark-out/neural_external < examples/neural_weights.txt)
[ "$embedded" = "$expected" ] || { echo "Embedded mismatch: $embedded != $expected" >&2; exit 1; }
[ "$external" = "$expected" ] || { echo "External mismatch: $external != $expected" >&2; exit 1; }
printf 'Reference: %s\nEmbedded:  %s\nExternal:  %s\n' "$expected" "$embedded" "$external"
./scripts/neural_benchmark.sh
