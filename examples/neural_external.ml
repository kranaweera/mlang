// Identical computation; weights read from standard input, row-major.
// Usage: ./neural_external < examples/neural_weights.txt
fn predict(x: f32[2]) -> f32[1] {
    let w1 = read_m3_2();
    let b1 = read_v3();
    let w2 = read_m1_3();
    let b2 = read_v1();
    return w2 @ relu(w1 @ x + b1) + b2;
}
fn main() { print(predict([2.0, 3.0])); }
