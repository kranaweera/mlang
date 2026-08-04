// Expected output: 18
fn predict(x: f32[3]) -> f32 {
    let weights = [0.5, 0.3, 0.2];
    let bias = 1.0;
    return dot(x, weights) + bias;
}

fn main() {
    let features = [10.0, 20.0, 30.0];
    print(predict(features));
}
