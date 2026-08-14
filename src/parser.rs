use crate::ast::{Expr, Function, Program, Stmt, Type};
use crate::lexer::Token;

pub struct Parser { tokens: Vec<Token>, pos: usize }
impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self { Self { tokens, pos: 0 } }
    fn peek(&self) -> &Token { &self.tokens[self.pos] }
    fn bump(&mut self) -> Token { let t = self.peek().clone(); if !matches!(t, Token::Eof) { self.pos += 1; } t }
    fn symbol(&mut self, c: char) -> bool { if self.peek() == &Token::Sym(c) { self.bump(); true } else { false } }
    fn expect(&mut self, c: char) -> Result<(), String> { if self.symbol(c) { Ok(()) } else { Err(format!("expected '{c}', found {:?}", self.peek())) } }
    fn keyword(&mut self, s: &str) -> bool { if self.peek() == &Token::Ident(s.to_string()) { self.bump(); true } else { false } }
    fn ident(&mut self) -> Result<String, String> { match self.bump() { Token::Ident(s) => Ok(s), t => Err(format!("expected identifier, found {t:?}")) } }
    fn ty(&mut self) -> Result<Type, String> {
        if self.ident()? != "f32" { return Err("only f32, f32[N] and f32[R,C] are supported".into()); }
        if self.symbol('[') {
            let n = match self.bump() { Token::Number(n) if n >= 1.0 && n <= 4096.0 && n.fract() == 0.0 => n as usize, _ => return Err("array size must be an integer from 1 to 4096".into()) };
            if self.symbol(',') {
                let m = match self.bump() { Token::Number(x) if x >= 1.0 && x <= 4096.0 && x.fract() == 0.0 => x as usize, _ => return Err("matrix dimension must be an integer from 1 to 4096".into()) };
                if n * m > 4096 { return Err("tensor exceeds 4096 elements".into()); }
                self.expect(']')?; Ok(Type::Matrix(n, m))
            } else { self.expect(']')?; Ok(Type::Vector(n)) }
        } else { Ok(Type::F32) }
    }
    pub fn program(&mut self) -> Result<Program, String> {
        let mut functions = Vec::new();
        while !matches!(self.peek(), Token::Eof) { functions.push(self.function()?); }
        if functions.is_empty() { return Err("program has no functions".into()); }
        Ok(Program { functions })
    }
    fn function(&mut self) -> Result<Function, String> {
        if !self.keyword("fn") { return Err(format!("expected 'fn', found {:?}", self.peek())); }
        let name = self.ident()?;
        self.expect('(')?; let mut params = Vec::new();
        if !self.symbol(')') { loop {
            let key = self.ident()?; self.expect(':')?; params.push((key, self.ty()?));
            if self.symbol(')') { break; } self.expect(',')?;
        }}
        let result = if matches!(self.peek(), Token::Arrow) { self.bump(); Some(self.ty()?) } else { None };
        self.expect('{')?;
        let mut body = Vec::new();
        while !self.symbol('}') {
            if matches!(self.peek(), Token::Eof) { return Err("unclosed function body".into()); }
            let statement = if self.keyword("let") {
                let name = self.ident()?; self.expect('=')?; let expr = self.expr(0)?; Stmt::Let(name, expr)
            } else if self.keyword("return") { Stmt::Return(self.expr(0)?) } else { Stmt::Expr(self.expr(0)?) };
            self.expect(';')?; body.push(statement);
        }
        Ok(Function { name, params, result, body })
    }
    fn expr(&mut self, min_prec: u8) -> Result<Expr, String> {
        let mut lhs = match self.bump() {
            Token::Number(n) => Expr::Number(n),
            Token::Sym('-') => Expr::UnaryMinus(Box::new(self.expr(5)?)),
            Token::Sym('(') => { let inner = self.expr(0)?; self.expect(')')?; inner },
            Token::Sym('[') => {
                let mut elems = Vec::new();
                if !self.symbol(']') { loop { elems.push(self.expr(0)?); if self.symbol(']') { break; } self.expect(',')?; } }
                Expr::Vector(elems)
            }
            Token::Ident(name) => {
                if self.symbol('(') {
                    let mut args = Vec::new();
                    if !self.symbol(')') { loop { args.push(self.expr(0)?); if self.symbol(')') { break; } self.expect(',')?; } }
                    Expr::Call(name, args)
                } else { Expr::Var(name) }
            }
            t => return Err(format!("unexpected token in expression: {t:?}")),
        };
        loop {
            let (op, prec) = match self.peek() { Token::Sym('+') => ('+', 1), Token::Sym('-') => ('-', 1), Token::Sym('*') => ('*', 3), Token::Sym('@') => ('@', 3), Token::Sym('/') => ('/', 3), _ => break };
            if prec < min_prec { break; }
            self.bump(); let rhs = self.expr(prec + 1)?;
            lhs = Expr::Binary(Box::new(lhs), op, Box::new(rhs));
        }
        Ok(lhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;
    #[test] fn precedence() {
        let p = Parser::new(lex("fn main(){ let x = 2 + 3 * 4; }").unwrap()).program().unwrap();
        assert!(matches!(&p.functions[0].body[0], Stmt::Let(_, Expr::Binary(_, '+', _))));
    }
    #[test] fn reject_incomplete() { assert!(Parser::new(lex("fn main() { return 1;").unwrap()).program().is_err()); }
}

#[cfg(test)]
mod tensor_tests {
    use super::*;
    use crate::lexer::lex;
    #[test] fn matrix_type_and_at() {
        let p=Parser::new(lex("fn f(x: f32[2,3]) -> f32[2,3] { return x; } fn main() { let y = [[1,2],[3,4]] @ [[1,0],[0,1]]; }").unwrap()).program().unwrap();
        assert_eq!(p.functions[0].params[0].1, Type::Matrix(2,3));
        assert!(matches!(p.functions[1].body[0],Stmt::Let(_,Expr::Binary(_, '@', _))));
    }
}
