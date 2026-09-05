//! Typed, backend-neutral intermediate representation.
//! Each expression carries an inferred type. No backend-specific C/CUDA strings.
use crate::ast::{self, Type};
use crate::semantic;
use std::collections::HashMap;

type Signatures = HashMap<String, (Vec<Type>, Option<Type>)>;

#[derive(Debug, Clone, PartialEq)]
pub struct Expr { pub ty: Type, pub kind: ExprKind }
#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Constant(f32), Variable(String), Tensor(Vec<Expr>),
    Neg(Box<Expr>), Binary(Box<Expr>, char, Box<Expr>), Call(String, Vec<Expr>),
}
#[derive(Debug, Clone, PartialEq)]
pub enum Statement { Let(String, Expr), Return(Expr), Evaluate(Expr), Print(Expr) }
#[derive(Debug, Clone, PartialEq)]
pub struct Function { pub name: String, pub params: Vec<(String,Type)>, pub result: Option<Type>, pub body: Vec<Statement> }
#[derive(Debug, Clone, PartialEq)]
pub struct Module { pub functions: Vec<Function> }

fn typed(e: &ast::Expr, vars: &HashMap<String,Type>, sigs: &Signatures) -> Result<Expr,String> {
    let ty = semantic::infer(e,vars,sigs)?;
    let kind = match e {
        ast::Expr::Number(x) => ExprKind::Constant(*x),
        ast::Expr::Var(x) => ExprKind::Variable(x.clone()),
        ast::Expr::Vector(xs) => ExprKind::Tensor(xs.iter().map(|x|typed(x,vars,sigs)).collect::<Result<_,_>>()?),
        ast::Expr::UnaryMinus(x) => ExprKind::Neg(Box::new(typed(x,vars,sigs)?)),
        ast::Expr::Binary(a,op,b) => ExprKind::Binary(Box::new(typed(a,vars,sigs)?),*op,Box::new(typed(b,vars,sigs)?)),
        ast::Expr::Call(name,args) => ExprKind::Call(name.clone(),args.iter().map(|x|typed(x,vars,sigs)).collect::<Result<_,_>>()?),
    };
    Ok(Expr{ty,kind})
}
/// Only call after semantic::check; lowering also returns useful errors.
pub fn lower(program: &ast::Program) -> Result<Module,String> {
    semantic::check(program)?;
    let sigs: Signatures = program.functions.iter().map(|f| (f.name.clone(),(f.params.iter().map(|(_,t)|t.clone()).collect(),f.result.clone()))).collect();
    let mut functions=Vec::new();
    for f in &program.functions {
        let mut vars: HashMap<String,Type> = f.params.iter().cloned().collect();
        let mut body=Vec::new();
        for stmt in &f.body {
            let lowered=match stmt {
                ast::Stmt::Let(name,e) => {let value=typed(e,&vars,&sigs)?; vars.insert(name.clone(),value.ty.clone());Statement::Let(name.clone(),value)},
                ast::Stmt::Return(e) => Statement::Return(typed(e,&vars,&sigs)?),
                ast::Stmt::Expr(ast::Expr::Call(name,args)) if name=="print" => Statement::Print(typed(&args[0],&vars,&sigs)?),
                ast::Stmt::Expr(e) => Statement::Evaluate(typed(e,&vars,&sigs)?),
            };
            body.push(lowered);
        }
        functions.push(Function{name:f.name.clone(),params:f.params.clone(),result:f.result.clone(),body});
    }
    Ok(Module{functions})
}

/// Fold scalar literal arithmetic. IEEE-754 result is evaluated at compile time;
/// divide by zero is left to the runtime to retain existing behavior.
pub fn fold_expr(expr: Expr) -> Expr {
    let ty=expr.ty;
    let kind=match expr.kind {
        ExprKind::Tensor(xs)=>ExprKind::Tensor(xs.into_iter().map(fold_expr).collect()),
        ExprKind::Neg(x)=>{
            let x=fold_expr(*x);
            if ty==Type::F32 {if let ExprKind::Constant(v)=&x.kind {return Expr{ty,kind:ExprKind::Constant(-*v)};}}
            ExprKind::Neg(Box::new(x))
        },
        ExprKind::Binary(a,op,b)=>{
            let a=fold_expr(*a);let b=fold_expr(*b);
            if ty==Type::F32 {
                if let (ExprKind::Constant(x),ExprKind::Constant(y))=(&a.kind,&b.kind) {
                    if !(op=='/' && *y==0.0) {
                        let v=match op {'+'=>Some(x+y),'-'=>Some(x-y),'*'=>Some(x*y),'/'=>Some(x/y),_=>None};
                        if let Some(v)=v {return Expr{ty,kind:ExprKind::Constant(v)};}
                    }
                }
            }
            ExprKind::Binary(Box::new(a),op,Box::new(b))
        },
        ExprKind::Call(n,xs)=>ExprKind::Call(n,xs.into_iter().map(fold_expr).collect()),
        other=>other,
    };
    Expr{ty,kind}
}
pub fn optimize(module: Module) -> Module {
    Module {functions:module.functions.into_iter().map(|mut f|{
        f.body=f.body.into_iter().map(|s|match s {
            Statement::Let(n,e)=>Statement::Let(n,fold_expr(e)),
            Statement::Return(e)=>Statement::Return(fold_expr(e)),
            Statement::Evaluate(e)=>Statement::Evaluate(fold_expr(e)),
            Statement::Print(e)=>Statement::Print(fold_expr(e)),
        }).collect();f
    }).collect()}
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn folds_scalar_constants() {
        let e=Expr{ty:Type::F32,kind:ExprKind::Binary(Box::new(Expr{ty:Type::F32,kind:ExprKind::Constant(2.)}),'+',Box::new(Expr{ty:Type::F32,kind:ExprKind::Constant(3.)}))};
        assert_eq!(fold_expr(e).kind,ExprKind::Constant(5.));
    }
    #[test] fn keeps_division_by_zero() {
        let e=Expr{ty:Type::F32,kind:ExprKind::Binary(Box::new(Expr{ty:Type::F32,kind:ExprKind::Constant(2.)}),'/',Box::new(Expr{ty:Type::F32,kind:ExprKind::Constant(0.)}))};
        assert!(matches!(fold_expr(e).kind,ExprKind::Binary(..)));
    }
}
