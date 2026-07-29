use crate::ast::{Expr, Program, Stmt, Type};
use std::collections::{HashMap, HashSet};

pub fn infer(expr: &Expr, vars: &HashMap<String, Type>, sigs: &HashMap<String, (Vec<Type>, Option<Type>)>) -> Result<Type, String> {
    match expr {
        Expr::Number(_) => Ok(Type::F32),
        Expr::Var(s) => vars.get(s).cloned().ok_or_else(|| format!("undefined variable: {s}")),
        Expr::Vector(items) => {
            if items.is_empty() { return Err("empty vectors are not supported".into()); }
            for x in items { if infer(x, vars, sigs)? != Type::F32 { return Err("vector elements must be scalar f32".into()); } }
            Ok(Type::Vector(items.len()))
        }
        Expr::UnaryMinus(x) => { if infer(x, vars, sigs)? != Type::F32 { return Err("unary minus expects f32".into()); } Ok(Type::F32) }
        Expr::Binary(a, _, b) => {
            if infer(a, vars, sigs)? != Type::F32 || infer(b, vars, sigs)? != Type::F32 { return Err("arithmetic operators require scalar f32".into()); }
            Ok(Type::F32)
        }
        Expr::Call(name, args) => {
            let types = args.iter().map(|e| infer(e, vars, sigs)).collect::<Result<Vec<_>, _>>()?;
            match name.as_str() {
                "dot" => match types.as_slice() { [Type::Vector(a), Type::Vector(b)] if a == b => Ok(Type::F32), _ => Err("dot expects two vectors of the same length".into()) },
                "print" => Err("print is a statement, not a value".into()),
                _ => {
                    let (params, result) = sigs.get(name).ok_or_else(|| format!("unknown function {name}"))?;
                    if &types != params { return Err(format!("argument types do not match function {name}")); }
                    result.clone().ok_or_else(|| format!("function {name} has no return value"))
                }
            }
        }
    }
}

pub fn check(program: &Program) -> Result<(), String> {
    let mut sigs = HashMap::new();
    for f in &program.functions {
        if f.name == "dot" || f.name == "print" { return Err(format!("reserved function name: {}", f.name)); }
        if sigs.insert(f.name.clone(), (f.params.iter().map(|(_, t)| t.clone()).collect(), f.result.clone())).is_some() { return Err(format!("duplicate function {}", f.name)); }
    }
    if !sigs.contains_key("main") { return Err("missing fn main()".into()); }
    let main = program.functions.iter().find(|f| f.name == "main").unwrap();
    if !main.params.is_empty() || main.result.is_some() { return Err("main must have no parameters and no return type".into()); }
    for f in &program.functions {
        let mut vars = HashMap::new(); let mut seen = HashSet::new();
        for (name, ty) in &f.params {
            if !seen.insert(name.clone()) { return Err(format!("duplicate parameter {name}")); }
            vars.insert(name.clone(), ty.clone());
        }
        let mut returned = false;
        for stmt in &f.body {
            if returned { return Err(format!("unreachable statement after return in {}", f.name)); }
            match stmt {
                Stmt::Let(name, expr) => {
                    let ty = infer(expr, &vars, &sigs)?;
                    if !seen.insert(name.clone()) { return Err(format!("duplicate variable {name}")); }
                    vars.insert(name.clone(), ty);
                }
                Stmt::Return(expr) => {
                    let ty = infer(expr, &vars, &sigs)?;
                    if Some(ty) != f.result { return Err(format!("return type mismatch in {}", f.name)); }
                    returned = true;
                }
                Stmt::Expr(Expr::Call(name, args)) if name == "print" => {
                    if args.len() != 1 || infer(&args[0], &vars, &sigs)? != Type::F32 { return Err("print requires one f32 argument".into()); }
                }
                Stmt::Expr(expr) => { infer(expr, &vars, &sigs)?; }
            }
        }
        if f.result.is_some() && !returned { return Err(format!("function {} requires a return statement", f.name)); }
    }
    Ok(())
}
