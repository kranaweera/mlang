#!/usr/bin/env python3
"""Convert an already trained sklearn MLPRegressor to MLang's tiny text model.

Execute in the model training environment (which has NumPy and scikit-learn).
Only load joblib/pickle files that you created or trust: loading runs arbitrary code.
"""
import argparse
import math
from pathlib import Path


def export(model, output):
    from sklearn.neural_network import MLPRegressor

    if type(model) is not MLPRegressor:
        raise TypeError("Only exact sklearn.neural_network.MLPRegressor models are supported")
    if model.activation not in ("relu", "identity") or model.out_activation_ != "identity":
        raise ValueError("Only relu/identity hidden activations and identity outputs are supported")
    if not hasattr(model, "coefs_") or not hasattr(model, "intercepts_"):
        raise ValueError("Model is not fitted")

    width = int(model.n_features_in_)
    lines = ["MLANG-MODEL 1", f"input {width}"]
    for index, (weights, biases) in enumerate(zip(model.coefs_, model.intercepts_)):
        in_dim, out_dim = weights.shape  # sklearn layout: input x output
        if in_dim != width:
            raise ValueError("Inconsistent network layer shapes")
        act = (model.activation if index < len(model.coefs_) - 1 else "linear")
        act = "linear" if act == "identity" else act
        lines.append(f"dense {out_dim} {act}")
        # MLang rows correspond to output neurons.
        flat = weights.T.reshape(-1).tolist()
        for value in flat + biases.tolist():
            if not math.isfinite(float(value)):
                raise ValueError("Non-finite parameter")
        lines.append("weights " + " ".join(repr(float(v)) for v in flat))
        lines.append("bias " + " ".join(repr(float(v)) for v in biases))
        width = out_dim
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trusted_joblib", help="Only open joblib files from trusted sources")
    parser.add_argument("output", help="Output .mmdl file")
    args = parser.parse_args()
    import joblib
    export(joblib.load(args.trusted_joblib), args.output)
    print(f"Exported {args.output}")


if __name__ == "__main__":
    main()
