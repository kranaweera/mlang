#!/bin/sh
set -eu
cargo build --release
mkdir -p benchmark-out
./target/release/mlang build examples/neural_embedded.ml --output benchmark-out/neural_embedded
./target/release/mlang build examples/neural_external.ml --output benchmark-out/neural_external
printf '%s\n' 'name,bytes,prediction'
for name in neural_embedded neural_external; do
  bytes=$(wc -c < "benchmark-out/$name" | tr -d ' ')
  if [ "$name" = neural_external ]; then
    result=$("benchmark-out/$name" < examples/neural_weights.txt)
  else
    result=$("benchmark-out/$name")
  fi
  printf '%s,%s,%s\n' "$name" "$bytes" "$result"
done
