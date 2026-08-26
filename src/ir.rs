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

/// Compatibility lowering for the existing stable C emitter; all checks and
/// optimizations have already happened in the typed IR.
fn to_ast_expr(e:&Expr)->ast::Expr {match &e.kind {
    ExprKind::Constant(v)=>ast::Expr::Number(*v),
    ExprKind::Variable(n)=>ast::Expr::Var(n.clone()),
    ExprKind::Tensor(xs)=>ast::Expr::Vector(xs.iter().map(to_ast_expr).collect()),
    ExprKind::Neg(x)=>ast::Expr::UnaryMinus(Box::new(to_ast_expr(x))),
    ExprKind::Binary(a,o,b)=>ast::Expr::Binary(Box::new(to_ast_expr(a)),*o,Box::new(to_ast_expr(b))),
    ExprKind::Call(n,xs)=>ast::Expr::Call(n.clone(),xs.iter().map(to_ast_expr).collect()),
}}
pub fn to_ast(module:&Module)->ast::Program {
    ast::Program {functions:module.functions.iter().map(|f|ast::Function {
        name:f.name.clone(),params:f.params.clone(),result:f.result.clone(),
        body:f.body.iter().map(|s|match s {
            Statement::Let(n,e)=>ast::Stmt::Let(n.clone(),to_ast_expr(e)),
            Statement::Return(e)=>ast::Stmt::Return(to_ast_expr(e)),
            Statement::Evaluate(e)=>ast::Stmt::Expr(to_ast_expr(e)),
            Statement::Print(e)=>ast::Stmt::Expr(ast::Expr::Call("print".into(),vec![to_ast_expr(e)])),
        }).collect(),
    }).collect()}
}
