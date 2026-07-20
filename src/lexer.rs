#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Ident(String), Number(f32), Sym(char), Arrow, Eof,
}

pub fn lex(source: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = source.chars().collect();
    let mut pos = 0; let mut out = Vec::new();
    while pos < chars.len() {
        let c = chars[pos];
        if c.is_whitespace() { pos += 1; continue; }
        if c == '/' && chars.get(pos + 1) == Some(&'/') {
            while pos < chars.len() && chars[pos] != '\n' { pos += 1; }
            continue;
        }
        if c == '-' && chars.get(pos + 1) == Some(&'>') {
            out.push(Token::Arrow); pos += 2; continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let start = pos; pos += 1;
            while pos < chars.len() && (chars[pos].is_ascii_alphanumeric() || chars[pos] == '_') { pos += 1; }
            out.push(Token::Ident(chars[start..pos].iter().collect())); continue;
        }
        if c.is_ascii_digit() || (c == '.' && chars.get(pos + 1).is_some_and(char::is_ascii_digit)) {
            let start = pos;
            while pos < chars.len() && chars[pos].is_ascii_digit() { pos += 1; }
            if chars.get(pos) == Some(&'.') {
                pos += 1;
                while pos < chars.len() && chars[pos].is_ascii_digit() { pos += 1; }
            }
            let text: String = chars[start..pos].iter().collect();
            let n = text.parse::<f32>().map_err(|_| format!("invalid number {text}"))?;
            out.push(Token::Number(n)); continue;
        }
        if "(){}[],:;=+-*/".contains(c) { out.push(Token::Sym(c)); pos += 1; continue; }
        return Err(format!("unexpected character '{c}' at character {}", pos + 1));
    }
    out.push(Token::Eof); Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn comments_and_arrow() {
        assert_eq!(lex("fn test() -> f32 { // hi\n return 2.5; }").unwrap().iter().filter(|t| **t == Token::Arrow).count(), 1);
    }
    #[test] fn invalid_character() { assert!(lex("@wow").is_err()); }
}
