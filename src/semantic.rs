use crate::ast::{Expr, Program, Stmt, Type};
use std::collections::{HashMap, HashSet};

type Signatures = HashMap<String, (Vec<Type>, Option<Type>)>;
fn limited(ty: Type) -> Result<Type, String> {
    let count = match ty { Type::F32 => 1, Type::Vector(n) => n, Type::Matrix(r,c) => r*c };
    if count > 4096 { Err("tensor exceeds 4096 elements".into()) } else { Ok(ty) }
}
pub fn infer(expr: &Expr, vars: &HashMap<String, Type>, sigs: &Signatures) -> Result<Type, String> {
    match expr {
        Expr::Number(_) => Ok(Type::F32),
        Expr::Var(s) => vars.get(s).cloned().ok_or_else(|| format!("undefined variable: {s}")),
        Expr::Vector(xs) => {
            if xs.is_empty() { return Err("empty tensor literal is not supported".into()); }
            let first = infer(&xs[0], vars, sigs)?;
            for x in &xs[1..] { if infer(x, vars, sigs)? != first { return Err("ragged tensor literal: rows must have matching shapes".into()); } }
            limited(match first { Type::F32 => Type::Vector(xs.len()), Type::Vector(n) => Type::Matrix(xs.len(),n), Type::Matrix(_,_) => return Err("only rank-1 and rank-2 tensors are supported".into()) })
        }
        Expr::UnaryMinus(x) => infer(x, vars, sigs),
        Expr::Binary(a, op, b) => {
            let left = infer(a,vars,sigs)?; let right = infer(b,vars,sigs)?;
            if *op == '@' {
                return match (left, right) {
                    (Type::Matrix(m,k),Type::Matrix(j,n)) if k==j => limited(Type::Matrix(m,n)),
                    (Type::Matrix(m,k),Type::Vector(j)) if k==j => limited(Type::Vector(m)),
                    _ => Err("matrix multiplication @ shape mismatch (expected matrix @ matrix or matrix @ vector)".into()),
                };
            }
            if left == right { Ok(left) } else { Err(format!("tensor shape mismatch for operator '{op}': {left:?} versus {right:?}")) }
        }
        Expr::Call(name, args) => {
            let types = args.iter().map(|x| infer(x,vars,sigs)).collect::<Result<Vec<_>,_>>()?;
            match name.as_str() {
                _ if name.starts_with("read_v") || name.starts_with("read_m") => {
                    if !args.is_empty() { return Err("read_* takes no arguments".into()); }
                    let ty = if let Some(n)=name.strip_prefix("read_v") {
                        let n=n.parse::<usize>().map_err(|_| "invalid read_v dimension")?;
                        Type::Vector(n)
                    } else {
                        let dims=name.strip_prefix("read_m").unwrap().split('_').collect::<Vec<_>>();
                        if dims.len()!=2 { return Err("invalid read_m dimensions".into()); }
                        let r=dims[0].parse::<usize>().map_err(|_| "invalid read_m rows")?;
                        let c=dims[1].parse::<usize>().map_err(|_| "invalid read_m columns")?;
                        Type::Matrix(r,c)
                    };
                    let elements=match ty {Type::Vector(n)=>n,Type::Matrix(r,c)=>r.checked_mul(c).ok_or("invalid dimensions")?,_=>0};
                    if elements==0 || elements>4096 {return Err("read tensor must contain 1..4096 elements".into());}
                    Ok(ty)
                },
                "dot" => match types.as_slice() { [Type::Vector(a),Type::Vector(b)] if a==b => Ok(Type::F32), _ => Err("dot expects two vectors of the same length".into()) },
                "relu" | "sigmoid" => match types.as_slice() { [t] => Ok(t.clone()), _ => Err(format!("{name} requires one argument")) },
                "print" => Err("print is a statement, not a value".into()),
                _ => {
                    let (params,result) = sigs.get(name).ok_or_else(|| format!("unknown function {name}"))?;
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
        if ["dot","print","relu","sigmoid"].contains(&f.name.as_str()) { return Err(format!("reserved function name: {}", f.name)); }
        if sigs.insert(f.name.clone(), (f.params.iter().map(|(_,t)|t.clone()).collect(), f.result.clone())).is_some() { return Err(format!("duplicate function {}",f.name)); }
    }
    if !sigs.contains_key("main") { return Err("missing fn main()".into()); }
    let main = program.functions.iter().find(|f|f.name=="main").unwrap();
    if !main.params.is_empty() || main.result.is_some() { return Err("main must have no parameters and no return type".into()); }
    for f in &program.functions {
        let mut vars = HashMap::new(); let mut seen = HashSet::new();
        for (name,ty) in &f.params { if !seen.insert(name.clone()) { return Err(format!("duplicate parameter {name}")); } vars.insert(name.clone(),ty.clone()); }
        let mut returned = false;
        for stmt in &f.body {
            if returned { return Err(format!("unreachable statement after return in {}",f.name)); }
            match stmt {
                Stmt::Let(name,expr) => { let ty=infer(expr,&vars,&sigs)?; if !seen.insert(name.clone()) { return Err(format!("duplicate variable {name}")); } vars.insert(name.clone(),ty); }
                Stmt::Return(expr) => { if Some(infer(expr,&vars,&sigs)?)!=f.result { return Err(format!("return type mismatch in {}",f.name)); } returned=true; }
                Stmt::Expr(Expr::Call(name,args)) if name=="print" => { if args.len()!=1 { return Err("print requires one argument".into()); } infer(&args[0],&vars,&sigs)?; }
                Stmt::Expr(expr) => { infer(expr,&vars,&sigs)?; }
            }
        }
        if f.result.is_some() && !returned { return Err(format!("function {} requires a return statement",f.name)); }
    }
    Ok(())
}
