mod ast;
mod backend;
mod lexer;
mod ir;
mod parser;
mod semantic;

use std::{env, fs, path::PathBuf, process::Command};

fn compile(source: &str) -> Result<String, String> {
    let tokens = lexer::lex(source)?;
    let ast = parser::Parser::new(tokens).program()?;
    let ir = ir::lower(&ast)?;
    Ok(backend::cpu::emit_c(&ir))
}

fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.is_empty() || args[0] != "build" {
        return Err("usage: mlang build <file.ml> [--output <binary>] [--emit-c <file.c>] [--cc <compiler>] [--target cpu|cuda]".into());
    }
    if args.len() < 2 { return Err("missing source file".into()); }
    let source = fs::read_to_string(&args[1]).map_err(|e| format!("cannot read {}: {e}", args[1]))?;
    let mut out_path = PathBuf::from("a.out");
    let mut emit_c = None;
    let mut cc = "clang".to_string();
    let mut target = "cpu".to_string();
    let mut i = 2;
    while i < args.len() {
        if i + 1 >= args.len() { return Err(format!("missing value for {}", args[i])); }
        match args[i].as_str() {
            "--output" => out_path = PathBuf::from(&args[i + 1]),
            "--emit-c" => emit_c = Some(PathBuf::from(&args[i + 1])),
            "--cc" => cc = args[i + 1].clone(),
            "--target" => target = args[i + 1].clone(),
            other => return Err(format!("unknown option {other}")),
        }
        i += 2;
    }
    if target == "cuda" { return Err(backend::cuda::status().into()); }
    if target != "cpu" { return Err(format!("unsupported target: {target}")); }
    let c = compile(&source)?;
    let c_path = emit_c.unwrap_or_else(|| out_path.with_extension("c"));
    fs::write(&c_path, c).map_err(|e| format!("cannot write {}: {e}", c_path.display()))?;
    let result = Command::new(&cc).arg("-std=c11").arg("-O2").arg("-Wall").arg("-Wextra")
        .arg(&c_path).arg("-o").arg(&out_path).arg("-lm").status().map_err(|e| format!("cannot execute compiler '{cc}': {e}"))?;
    if !result.success() { return Err(format!("C compiler returned {result}")); }
    println!("Built {} (C source: {})", out_path.display(), c_path.display());
    Ok(())
}
fn main() { if let Err(err) = run() { eprintln!("mlang: {err}"); std::process::exit(1); } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn valid_program() { assert!(compile("fn main() { print(2 + 3 * 4); }").is_ok()); }
    #[test] fn undefined_variable() { assert!(compile("fn main() { print(x); }").unwrap_err().contains("undefined")); }
    #[test] fn mismatched_dot() { assert!(compile("fn main() { print(dot([1,2],[3])); }").is_err()); }
    #[test] fn missing_return() { assert!(compile("fn f() -> f32 { let x = 2; } fn main() {}").is_err()); }
    #[test] fn cuda_explicitly_unavailable() { assert!(backend::cuda::status().contains("planned")); }
}
