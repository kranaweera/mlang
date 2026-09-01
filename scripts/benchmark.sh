#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --release
mkdir -p benchmark-out
printf 'example,bytes,run_ms\n' > benchmark-out/results.csv
for example in hello linear_regression matmul matvec tensor_ops; do
  ./target/release/mlang build "examples/${example}.ml" --output "benchmark-out/${example}" >/dev/null
  bytes=$(wc -c < "benchmark-out/${example}" | tr -d ' ')
  start=$(python3 -c 'import time;print(time.monotonic_ns())')
  "benchmark-out/${example}" >/dev/null
  stop=$(python3 -c 'import time;print(time.monotonic_ns())')
  ms=$(python3 -c "print(round(($stop-$start)/1000000,3))")
  printf '%s,%s,%s\n' "$example" "$bytes" "$ms" | tee -a benchmark-out/results.csv
done
printf '\nSingle-run wall time includes OS process startup; not a microbenchmark.\n'
