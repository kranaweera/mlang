mod ast;
mod backend;
mod lexer;
mod parser;
mod semantic;

use std::{env, fs};

fn compile(source: &str) -> Result<String, String> {
    let tokens = lexer::lex(source)?;
    let ast = parser::Parser::new(tokens).program()?;
    semantic::check(&ast)?;
    Ok(backend::cpu::emit_c(&ast))
}

fn run() -> Result<(), String> {
    let path = env::args().nth(1).ok_or("usage: mlang <file.ml>")?;
    let source = fs::read_to_string(&path).map_err(|e| format!("cannot read {path}: {e}"))?;
    print!("{}", compile(&source)?);
    Ok(())
}
fn main() { if let Err(err) = run() { eprintln!("mlang: {err}"); std::process::exit(1); } }
