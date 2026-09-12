#!/usr/bin/env python3
"""Independent standard-library reference for MLang's example network.
Python is a development-only validation tool; deployed MLang binaries do not need it.
"""
from pathlib import Path
import math

root = Path(__file__).resolve().parents[1]
w = [float(x) for x in (root / "examples/neural_weights.txt").read_text().split()]
assert len(w) == 13, f"expected 13 weights, received {len(w)}"
x = (2.0, 3.0)
w1 = [w[i:i + 2] for i in range(0, 6, 2)]
b1 = w[6:9]
w2 = w[9:12]
b2 = w[12]
h = [max(0.0, math.fsum(a * b for a, b in zip(row, x)) + bias) for row, bias in zip(w1, b1)]
y = math.fsum(a * b for a, b in zip(w2, h)) + b2
print(f"[{y:g}]")
