use std::process::Command;
use std::path::PathBuf;

fn build_and_run(example: &str, expected: f32) {
    let compiler = env!("CARGO_BIN_EXE_mlang");
    let temp = std::env::temp_dir().join(format!("mlang-{}-{}", example.replace('.', "_"), std::process::id()));
    std::fs::create_dir_all(&temp).unwrap();
    let binary = temp.join("program");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let built = Command::new(compiler)
        .arg("build").arg(root.join("examples").join(example))
        .arg("--output").arg(&binary)
        .output().expect("run compiler");
    assert!(built.status.success(), "{}", String::from_utf8_lossy(&built.stderr));
    let result = Command::new(&binary).output().expect("run generated binary");
    assert!(result.status.success());
    let value: f32 = String::from_utf8(result.stdout).unwrap().trim().parse().unwrap();
    assert!((value - expected).abs() < 0.001, "{value} != {expected}");
    let _ = std::fs::remove_dir_all(temp);
}

#[test] fn hello() { build_and_run("hello.ml", 14.0); }
#[test] fn linear_regression() { build_and_run("linear_regression.ml", 18.0); }

#[test]
fn invalid_type_is_rejected() {
    let compiler = env!("CARGO_BIN_EXE_mlang");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new(compiler)
        .arg("build").arg(root.join("examples/negative_test.ml"))
        .output().expect("run compiler");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("dot expects"));
}

fn build_and_check_text(example: &str, expected: &str) {
    let compiler = env!("CARGO_BIN_EXE_mlang");
    let temp = std::env::temp_dir().join(format!("mlang-tensor-{}-{}", example.replace('.', "_"), std::process::id()));
    std::fs::create_dir_all(&temp).unwrap();
    let binary = temp.join("program");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let built = Command::new(compiler).arg("build").arg(root.join("examples").join(example)).arg("--output").arg(&binary).output().unwrap();
    assert!(built.status.success(), "{}", String::from_utf8_lossy(&built.stderr));
    let result = Command::new(&binary).output().unwrap();
    assert!(result.status.success());
    assert_eq!(String::from_utf8(result.stdout).unwrap().trim(), expected);
    let _ = std::fs::remove_dir_all(temp);
}
#[test] fn matrix_multiplication() { build_and_check_text("matmul.ml", "[[19, 22], [43, 50]]"); }
#[test] fn matrix_vector() { build_and_check_text("matvec.ml", "[17, 39]"); }
#[test] fn elementwise_and_activation() { build_and_check_text("tensor_ops.ml", "[[4, 8], [10, 4]]\n[0.5, 0.5]"); }
#[test] fn incompatible_matmul_is_rejected() {
    let compiler = env!("CARGO_BIN_EXE_mlang");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let output = Command::new(compiler).arg("build").arg(root.join("examples/shape_error.ml")).output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("shape mismatch"));
}

#[test] fn scalar_constant_folding() { build_and_check_text("constant_folding.ml", "14"); }

#[test] fn embedded_neural_network() { build_and_check_text("neural_embedded.ml", "[5]"); }
#[test] fn external_neural_network() {
 use std::process::Stdio;
 use std::io::Write;
 let compiler=env!("CARGO_BIN_EXE_mlang");
 let root=PathBuf::from(env!("CARGO_MANIFEST_DIR"));
 let temp=std::env::temp_dir().join(format!("mlang-external-{}",std::process::id()));
 std::fs::create_dir_all(&temp).unwrap();let binary=temp.join("model");
 let built=Command::new(compiler).arg("build").arg(root.join("examples/neural_external.ml")).arg("--output").arg(&binary).output().unwrap();
 assert!(built.status.success(),"{}",String::from_utf8_lossy(&built.stderr));
 let mut child=Command::new(&binary).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
 child.stdin.take().unwrap().write_all(std::fs::read(root.join("examples/neural_weights.txt")).unwrap().as_slice()).unwrap();
 let output=child.wait_with_output().unwrap();assert!(output.status.success());
 assert_eq!(String::from_utf8(output.stdout).unwrap().trim(),"[5]");
 let _=std::fs::remove_dir_all(temp);
}

#[test]
fn external_neural_network_rejects_bad_weights() {
    use std::io::Write;
    use std::process::Stdio;
    let compiler = env!("CARGO_BIN_EXE_mlang");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temp = std::env::temp_dir().join(format!("mlang-invalid-weight-{}", std::process::id()));
    std::fs::create_dir_all(&temp).unwrap();
    let binary = temp.join("model");
    let built = Command::new(compiler).arg("build")
        .arg(root.join("examples/neural_external.ml"))
        .arg("--output").arg(&binary).output().unwrap();
    assert!(built.status.success(), "{}", String::from_utf8_lossy(&built.stderr));
    for bad_input in ["1.0 2.0", "NaN 0 0 0 0 0 0 0 0 0 0 0 0"] {
        let mut child = Command::new(&binary).stdin(Stdio::piped())
            .stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
        child.stdin.take().unwrap().write_all(bad_input.as_bytes()).unwrap();
        let result = child.wait_with_output().unwrap();
        assert_eq!(result.status.code(), Some(2), "{:?}", result);
        assert!(String::from_utf8_lossy(&result.stderr).contains("invalid external weight"));
    }
    let _ = std::fs::remove_dir_all(temp);
}
