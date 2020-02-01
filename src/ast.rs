use super::Token;
use lexer::Stream;

use std::{error, fmt};

#[derive(Debug)]
pub struct ParserError {
    text: String
}

impl ParserError {
    pub fn from(string: impl Into<String>) -> Self {
        Self { text: string.into() }
    }
}

impl error::Error for ParserError {}
impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let txt = format!("ParserrError: {}", self.text);
        f.write_str(&txt)
    }
}

pub trait ASTNode: Sized {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError>;
    fn run_parser(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        // This is not ideal.
        // Whole token stream must be cloned to perform backtracing.
        let mut stream = input.clone();
        match Self::parse(&mut stream) {
            // Parsing sucessfull, overwrite old stream with consumed one.
            Ok(expr) => {
                *input = stream;
                Ok(expr)
            },
            // Otherwise propagate error to caller and do not change stream.
            err   => err
        }
    }
}

#[derive(Debug)]
struct SingleTokenExprAST;
impl SingleTokenExprAST {
    fn expect(expected: Token, input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        let x = match input.next() {
            Some(x) => {
                if x == expected {
                    return Ok(Self)
                }
                Some(x)
            },
            _ => None
        };
        input.revert();
        Err(ParserError::from(format!("Got '{:?}'. Expected '{:?}'", x, expected)))
    }
}

impl ASTNode for SingleTokenExprAST {
    fn parse(_: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        Err(ParserError::from("parse() on SingleTokenExprAST"))
    }
}

#[derive(Debug)]
pub struct NumExprAST {
    value: f64
}

impl ASTNode for NumExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        let next = input.next();
        if let Some(Token::NUM(val)) = next {
            Ok(Self { value: val })
        } else {
            Err(ParserError::from(format!("Expected numeric. Got {:?}", next)))
        }
    }
}

#[derive(Debug)]
pub struct IdentifierExprAST {
    name: String
}

impl ASTNode for IdentifierExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        let next = input.next();
        if let Some(Token::IDENT(val)) = next {
            Ok(Self { name: val })
        } else {
            Err(ParserError::from(format!("Expected identifier. Got {:?}", next)))
        }
    }
}

#[derive(Debug)]
pub struct CallExprAST {
    name: IdentifierExprAST,
    args: Vec<ValuelikeExprAST>
}

impl ASTNode for CallExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        let ident = IdentifierExprAST::run_parser(input)?;

        SingleTokenExprAST::expect(Token::PAREN_OP, input)?;

        let mut args = Vec::new();

        loop {
            args.push(ValuelikeExprAST::run_parser(input)?);

            match input.next() {
                Some(Token::COMMA) => {},
                Some(Token::PAREN_CL) => break,
                _ => Err(ParserError::from("Expected ',' or ')'."))?
            };
        }

        Ok(Self { name: ident, args: args })
    }
}

#[derive(Debug)]
pub enum ValuelikeExprAST {
    Numeric(NumExprAST),
    Identifier(IdentifierExprAST),
    Call(CallExprAST)
}

impl ASTNode for ValuelikeExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        if let Ok(numeric) = NumExprAST::run_parser(input) {
            Ok(Self::Numeric(numeric))
        } else if let Ok(ident) = CallExprAST::run_parser(input) {
            Ok(Self::Call(ident))
        } else if let Ok(ident) = IdentifierExprAST::run_parser(input) {
            Ok(Self::Identifier(ident))
        } else {
            let token = input.peek(1);
            Err(ParserError::from(format!("Expected valuelike. Got {:?}", token.first())))
        }
    }
}

#[derive(Debug)]
pub struct AssignmentExprAST {
    ident: IdentifierExprAST,
    value: ValuelikeExprAST
}

impl ASTNode for AssignmentExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        SingleTokenExprAST::expect(Token::LET, input)?;

        let ident = IdentifierExprAST::run_parser(input)?;

        let next = input.next();
        if let Some(Token::ASSIGNMENT) = next {
            let value = ValuelikeExprAST::run_parser(input)?;
            Ok(Self { ident: ident, value: value })
        } else {
            Err(ParserError::from(format!("Expected '=' got '{:?}'.", next)))
        }
    }
}

#[derive(Debug)]
pub enum InBlockExprAST {
    Assingment(AssignmentExprAST),
    Valuelike(ValuelikeExprAST),
    If(IfElseExprAST)
}

impl ASTNode for InBlockExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        if let Ok(asign) = AssignmentExprAST::run_parser(input) {
            Ok(Self::Assingment(asign))
        } else if let Ok(iff) = IfElseExprAST::run_parser(input) {
            Ok(Self::If(iff))
        } else if let Ok(value) = ValuelikeExprAST::run_parser(input) {
            Ok(Self::Valuelike(value))
        } else {
            let token = input.peek(1);
            Err(ParserError::from(format!("Expected valuelike. Got {:?}", token.first())))
        }
    }
}

#[derive(Debug)]
pub struct BlockExprAST {
    body: Vec<InBlockExprAST>
}

impl ASTNode for BlockExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        SingleTokenExprAST::expect(Token::BLOCK_OP, input)?;

        let mut exprs = Vec::new();
        while let Ok(expr) = InBlockExprAST::run_parser(input) {
            exprs.push(expr);
        }

        SingleTokenExprAST::expect(Token::BLOCK_CL, input)?;

        Ok(Self { body: exprs } )
    }
}

#[derive(Debug)]
pub struct IfElseExprAST {
    // Temporary
    cond: IdentifierExprAST,
    block_if: BlockExprAST,
    block_else: Option<BlockExprAST>
}

impl ASTNode for IfElseExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        SingleTokenExprAST::expect(Token::IF, input)?;

        let cond = IdentifierExprAST::run_parser(input)?;
        let block_if = BlockExprAST::run_parser(input)?;

        println!("NXT: {:?}", input.peek1());
        let block_else = if let Ok(_) = SingleTokenExprAST::expect(Token::ELSE, input) {
            Some(BlockExprAST::run_parser(input)?)
        } else {
            None
        };

        Ok(Self { cond: cond, block_if: block_if, block_else: block_else })
    }
}

#[derive(Debug)]
pub struct FuncPrototypeExprAST {
    name: IdentifierExprAST,
    args: Vec<IdentifierExprAST>
}

impl ASTNode for FuncPrototypeExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        SingleTokenExprAST::expect(Token::DEF, input)?;

        let ident = IdentifierExprAST::run_parser(input)?;

        SingleTokenExprAST::expect(Token::PAREN_OP, input)?;

        let mut args = Vec::new();
        loop {
            args.push(IdentifierExprAST::run_parser(input)?);
            match input.next() {
                Some(Token::COMMA) => {},
                Some(Token::PAREN_CL) => break,
                _ => Err(ParserError::from("Expected ',' or ')'."))?
            };
        }

        Ok(Self { name: ident, args: args })
    }
}

#[derive(Debug)]
pub struct FuncDefExprAST {
    prototype: FuncPrototypeExprAST,
    body: BlockExprAST
}

impl ASTNode for FuncDefExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        let prototype = FuncPrototypeExprAST::run_parser(input)?;
        let block = BlockExprAST::run_parser(input)?;
        Ok(Self { prototype: prototype, body: block })
    }
}

#[derive(Debug)]
pub enum OutBlockExprAST {
    Assingment(AssignmentExprAST),
    FuncDef(FuncDefExprAST)
}

impl ASTNode for OutBlockExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        if let Ok(asign) = AssignmentExprAST::run_parser(input) {
            Ok(Self::Assingment(asign))
        } else if let Ok(def) = FuncDefExprAST::run_parser(input) {
            Ok(Self::FuncDef(def))
        } else {
            let token = input.peek(1);
            let err = format!("Expected assignment or func definition. Got {:?}", token.first());
            Err(ParserError::from(err))
        }
    }
}

#[derive(Debug)]
pub struct RootExprAST {
    items: Vec<OutBlockExprAST>
}

impl ASTNode for RootExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        let mut exprs = Vec::new();
        while let Ok(expr) = OutBlockExprAST::run_parser(input) {
            exprs.push(expr)
        }
        SingleTokenExprAST::expect(Token::EOF, input)?;
        Ok(Self { items: exprs })
    }
}

pub fn mk_ast(tokens: &mut impl Stream<Token>) -> Result<RootExprAST, ParserError> {
    RootExprAST::run_parser(tokens)
}
