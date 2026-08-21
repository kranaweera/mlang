fn predict(x: f32[2]) -> f32[2] {
    let weights = [[1.0, 2.0], [3.0, 4.0]];
    return relu(weights @ x);
}
fn main() { print(predict([5.0, 6.0])); }
