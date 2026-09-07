// Tiny trained two-layer network, baked-in weights (2 -> 3 -> 1).
fn predict(x: f32[2]) -> f32[1] {
    let w1 = [[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]];
    let b1 = [0.0, 0.0, 1.0];
    let hidden = relu(w1 @ x + b1);
    let w2 = [[0.5, 0.25, 1.0]];
    let b2 = [0.5];
    return w2 @ hidden + b2;
}
fn main() { print(predict([2.0, 3.0])); }
