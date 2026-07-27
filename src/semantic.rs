use crate::ast::{Expr, Type};
use std::collections::HashMap;

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
