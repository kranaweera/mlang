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
