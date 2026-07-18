#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type { F32, Vector(usize) }

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(f32), Var(String), Vector(Vec<Expr>),
    UnaryMinus(Box<Expr>), Binary(Box<Expr>, char, Box<Expr>),
    Call(String, Vec<Expr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt { Let(String, Expr), Return(Expr), Expr(Expr) }

#[derive(Debug, Clone, PartialEq)]
pub struct Function { pub name: String, pub params: Vec<(String, Type)>, pub result: Option<Type>, pub body: Vec<Stmt> }

#[derive(Debug, Clone, PartialEq)]
pub struct Program { pub functions: Vec<Function> }
