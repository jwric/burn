//! A tiny expression language for tensor scratch work.
//!
//! Lines are either `name = expr` or a bare `expr`. Values are 2-D tensors or scalars; every
//! operation builds tensor operations that run on whatever device the [`Session`](crate::web)
//! holds — here, a remote compute peer. Evaluation is synchronous (it only records operations);
//! reading a result back is the caller's concern.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use burn::tensor::activation::{relu, sigmoid, tanh};
use burn::tensor::{Device, Distribution, Tensor};

/// A value produced by evaluating an expression.
#[derive(Clone)]
pub enum Value {
    Tensor(Tensor<2>),
    Scalar(f32),
}

/// Named bindings that persist across REPL lines.
pub type Env = BTreeMap<String, Value>;

#[derive(Clone, Copy, PartialEq)]
enum Tok {
    Num(f32),
    Ident(usize, usize),
    LParen,
    RParen,
    Comma,
    Op(char),
    Eq,
}

fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let bytes = src.as_bytes();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        match c {
            ' ' | '\t' | '\r' | '\n' => i += 1,
            '(' => {
                toks.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                toks.push(Tok::RParen);
                i += 1;
            }
            ',' => {
                toks.push(Tok::Comma);
                i += 1;
            }
            '=' => {
                toks.push(Tok::Eq);
                i += 1;
            }
            '+' | '-' | '*' | '/' | '@' => {
                toks.push(Tok::Op(c));
                i += 1;
            }
            _ if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < bytes.len()
                    && ((bytes[i] as char).is_ascii_digit() || bytes[i] == b'.')
                {
                    i += 1;
                }
                let num = src[start..i]
                    .parse::<f32>()
                    .map_err(|_| format!("invalid number: {}", &src[start..i]))?;
                toks.push(Tok::Num(num));
            }
            _ if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < bytes.len()
                    && ((bytes[i] as char).is_ascii_alphanumeric() || bytes[i] == b'_')
                {
                    i += 1;
                }
                toks.push(Tok::Ident(start, i));
            }
            other => return Err(format!("unexpected character: {other}")),
        }
    }
    Ok(toks)
}

enum Expr {
    Num(f32),
    Var(String),
    Call(String, Vec<Expr>),
    Bin(char, Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
}

struct Parser<'a> {
    src: &'a str,
    toks: Vec<Tok>,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn ident(&self, tok: Tok) -> String {
        match tok {
            Tok::Ident(a, b) => self.src[a..b].to_string(),
            _ => String::new(),
        }
    }

    fn peek(&self) -> Option<Tok> {
        self.toks.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).copied();
        self.pos += 1;
        t
    }

    fn expect(&mut self, tok: Tok, what: &str) -> Result<(), String> {
        if self.bump() == Some(tok) {
            Ok(())
        } else {
            Err(format!("expected {what}"))
        }
    }

    fn add(&mut self) -> Result<Expr, String> {
        let mut left = self.mul()?;
        while let Some(Tok::Op(op @ ('+' | '-'))) = self.peek() {
            self.bump();
            let right = self.mul()?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn mul(&mut self) -> Result<Expr, String> {
        let mut left = self.unary()?;
        while let Some(Tok::Op(op @ ('*' | '/' | '@'))) = self.peek() {
            self.bump();
            let right = self.unary()?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<Expr, String> {
        if let Some(Tok::Op('-')) = self.peek() {
            self.bump();
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        self.atom()
    }

    fn atom(&mut self) -> Result<Expr, String> {
        match self.bump() {
            Some(Tok::Num(n)) => Ok(Expr::Num(n)),
            Some(Tok::LParen) => {
                let inner = self.add()?;
                self.expect(Tok::RParen, "`)`")?;
                Ok(inner)
            }
            Some(tok @ Tok::Ident(..)) => {
                let name = self.ident(tok);
                if self.peek() == Some(Tok::LParen) {
                    self.bump();
                    let mut args = Vec::new();
                    if self.peek() != Some(Tok::RParen) {
                        loop {
                            args.push(self.add()?);
                            match self.peek() {
                                Some(Tok::Comma) => {
                                    self.bump();
                                }
                                _ => break,
                            }
                        }
                    }
                    self.expect(Tok::RParen, "`)`")?;
                    Ok(Expr::Call(name, args))
                } else {
                    Ok(Expr::Var(name))
                }
            }
            _ => Err("expected a value".to_string()),
        }
    }
}

/// Parse a line into either an assignment target and its expression, or a bare expression.
fn parse(src: &str) -> Result<(Option<String>, Expr), String> {
    let toks = lex(src)?;
    let mut parser = Parser { src, toks, pos: 0 };

    // `name = expr` when the line starts with `ident =`.
    if let (Some(tok @ Tok::Ident(..)), Some(Tok::Eq)) =
        (parser.toks.first().copied(), parser.toks.get(1).copied())
    {
        let name = parser.ident(tok);
        parser.pos = 2;
        let expr = parser.add()?;
        if parser.pos != parser.toks.len() {
            return Err("trailing tokens after expression".to_string());
        }
        return Ok((Some(name), expr));
    }

    let expr = parser.add()?;
    if parser.pos != parser.toks.len() {
        return Err("trailing tokens after expression".to_string());
    }
    Ok((None, expr))
}

fn dim(value: &Value) -> Result<usize, String> {
    match value {
        Value::Scalar(s) if *s >= 0.0 && s.fract() == 0.0 => Ok(*s as usize),
        _ => Err("expected a non-negative integer dimension".to_string()),
    }
}

fn as_tensor(value: Value) -> Result<Tensor<2>, String> {
    match value {
        Value::Tensor(t) => Ok(t),
        Value::Scalar(_) => Err("expected a tensor, found a scalar".to_string()),
    }
}

fn eval(expr: &Expr, device: &Device, env: &Env) -> Result<Value, String> {
    match expr {
        Expr::Num(n) => Ok(Value::Scalar(*n)),
        Expr::Var(name) => env
            .get(name)
            .cloned()
            .ok_or_else(|| format!("unknown variable: {name}")),
        Expr::Neg(inner) => match eval(inner, device, env)? {
            Value::Scalar(s) => Ok(Value::Scalar(-s)),
            Value::Tensor(t) => Ok(Value::Tensor(t.neg())),
        },
        Expr::Bin(op, l, r) => {
            let left = eval(l, device, env)?;
            let right = eval(r, device, env)?;
            apply_binary(*op, left, right)
        }
        Expr::Call(name, args) => {
            let values: Result<Vec<_>, _> =
                args.iter().map(|a| eval(a, device, env)).collect();
            apply_call(name, values?, device)
        }
    }
}

fn apply_binary(op: char, left: Value, right: Value) -> Result<Value, String> {
    use Value::{Scalar, Tensor as T};

    if op == '@' {
        return Ok(T(as_tensor(left)?.matmul(as_tensor(right)?)));
    }

    Ok(match (op, left, right) {
        ('+', Scalar(a), Scalar(b)) => Scalar(a + b),
        ('-', Scalar(a), Scalar(b)) => Scalar(a - b),
        ('*', Scalar(a), Scalar(b)) => Scalar(a * b),
        ('/', Scalar(a), Scalar(b)) => Scalar(a / b),

        ('+', T(t), Scalar(s)) | ('+', Scalar(s), T(t)) => T(t.add_scalar(s)),
        ('*', T(t), Scalar(s)) | ('*', Scalar(s), T(t)) => T(t.mul_scalar(s)),
        ('-', T(t), Scalar(s)) => T(t.sub_scalar(s)),
        ('-', Scalar(s), T(t)) => T(t.neg().add_scalar(s)),
        ('/', T(t), Scalar(s)) => T(t.div_scalar(s)),
        ('/', Scalar(s), T(t)) => T(t.powf_scalar(-1.0).mul_scalar(s)),

        ('+', T(a), T(b)) => T(a.add(b)),
        ('-', T(a), T(b)) => T(a.sub(b)),
        ('*', T(a), T(b)) => T(a.mul(b)),
        ('/', T(a), T(b)) => T(a.div(b)),

        (op, _, _) => return Err(format!("unsupported operator `{op}`")),
    })
}

fn apply_call(name: &str, args: Vec<Value>, device: &Device) -> Result<Value, String> {
    let creation = |args: &[Value]| -> Result<[usize; 2], String> {
        if args.len() != 2 {
            return Err(format!("{name} expects (rows, cols)"));
        }
        Ok([dim(&args[0])?, dim(&args[1])?])
    };

    let unary = |args: Vec<Value>| -> Result<Tensor<2>, String> {
        if args.len() != 1 {
            return Err(format!("{name} expects a single tensor"));
        }
        as_tensor(args.into_iter().next().unwrap())
    };

    Ok(match name {
        "zeros" => Value::Tensor(Tensor::zeros(creation(&args)?, device)),
        "ones" => Value::Tensor(Tensor::ones(creation(&args)?, device)),
        "rand" => Value::Tensor(Tensor::random(creation(&args)?, Distribution::Default, device)),
        "randn" => Value::Tensor(Tensor::random(
            creation(&args)?,
            Distribution::Normal(0.0, 1.0),
            device,
        )),
        "relu" => Value::Tensor(relu(unary(args)?)),
        "sigmoid" => Value::Tensor(sigmoid(unary(args)?)),
        "tanh" => Value::Tensor(tanh(unary(args)?)),
        "exp" => Value::Tensor(unary(args)?.exp()),
        "sin" => Value::Tensor(unary(args)?.sin()),
        "cos" => Value::Tensor(unary(args)?.cos()),
        "abs" => Value::Tensor(unary(args)?.abs()),
        "t" | "transpose" => Value::Tensor(unary(args)?.transpose()),
        "sum" => Value::Tensor(unary(args)?.sum().reshape([1, 1])),
        "mean" => Value::Tensor(unary(args)?.mean().reshape([1, 1])),
        other => return Err(format!("unknown function: {other}")),
    })
}

/// Evaluate one REPL line, updating `env` on assignment. Returns the value to display (the
/// right-hand side for an assignment, or the expression's value otherwise).
pub fn run_line(src: &str, device: &Device, env: &mut Env) -> Result<Value, String> {
    let (target, expr) = parse(src)?;
    let value = eval(&expr, device, env)?;
    if let Some(name) = target {
        env.insert(name, value.clone());
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(src: &str) -> f32 {
        let device = Device::default();
        match run_line(src, &device, &mut Env::new()).unwrap() {
            Value::Scalar(s) => s,
            Value::Tensor(_) => panic!("expected a scalar"),
        }
    }

    fn tensor(env: &mut Env, src: &str) -> (Vec<f32>, [usize; 2]) {
        let device = Device::default();
        match run_line(src, &device, env).unwrap() {
            Value::Tensor(t) => {
                let dims = t.dims();
                (t.into_data().to_vec().unwrap(), dims)
            }
            Value::Scalar(_) => panic!("expected a tensor"),
        }
    }

    #[test]
    fn scalar_arithmetic_respects_precedence() {
        assert_eq!(scalar("2 + 3 * 4"), 14.0);
        assert_eq!(scalar("(2 + 3) * 4"), 20.0);
        assert_eq!(scalar("-2 + 10 / 4"), 0.5);
    }

    #[test]
    fn elementwise_and_broadcast_scalar() {
        let mut env = Env::new();
        let (values, dims) = tensor(&mut env, "ones(2, 2) + 1");
        assert_eq!(dims, [2, 2]);
        assert_eq!(values, vec![2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn matmul_shapes_and_values() {
        let mut env = Env::new();
        let (values, dims) = tensor(&mut env, "ones(2, 3) @ ones(3, 2)");
        assert_eq!(dims, [2, 2]);
        assert_eq!(values, vec![3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn variables_persist_across_lines() {
        let device = Device::default();
        let mut env = Env::new();
        run_line("a = ones(2, 2)", &device, &mut env).unwrap();
        let (values, dims) = tensor(&mut env, "sum(a * 3)");
        assert_eq!(dims, [1, 1]);
        assert_eq!(values, vec![12.0]);
    }

    #[test]
    fn transpose_reorders_dimensions() {
        let mut env = Env::new();
        let (_, dims) = tensor(&mut env, "t(zeros(2, 5))");
        assert_eq!(dims, [5, 2]);
    }

    #[test]
    fn reports_unknown_names() {
        let device = Device::default();
        assert!(run_line("nope(1, 2)", &device, &mut Env::new()).is_err());
        assert!(run_line("missing + 1", &device, &mut Env::new()).is_err());
    }
}
