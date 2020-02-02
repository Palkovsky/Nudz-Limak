use super::{
    Token,
    BinOpMath,
    BinOpCmp
};
use lexer::Stream;

use std::{error, fmt};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ParserError {
    text: String
}

#[derive(Debug, Clone)]
struct SingleTokenExprAST;

#[derive(Debug, Clone)]
pub struct NumLiteralExprAST {
    value: f64
}

#[derive(Debug, Clone)]
pub struct IdentifierExprAST {
    name: String
}

#[derive(Debug, Clone)]
pub struct BinOpMathExprAST {
    lhs: ValuelikeExprAST,
    op: BinOpMath,
    rhs: ValuelikeExprAST
}

#[derive(Debug, Clone)]
enum BinOpExprAtom {
    Literal(NumLiteralExprAST),
    Identifier(IdentifierExprAST),
    Call(CallExprAST)
}

impl BinOpExprAtom {
    fn valuelike(self) -> ValuelikeExprAST {
        match self {
            Self::Literal(num)      => ValuelikeExprAST::NumericLiteral(num),
            Self::Identifier(ident) => ValuelikeExprAST::Identifier(ident),
            Self::Call(call)        => ValuelikeExprAST::Call(call)
        }
    }
}

impl ASTNode for BinOpExprAtom {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        if let Ok(num) = NumLiteralExprAST::run_parser(input) {
            Ok(Self::Literal(num))
        } else if let Ok(ident) = CallExprAST::run_parser(input) {
            Ok(Self::Call(ident))
        } else if let Ok(ident) = IdentifierExprAST::run_parser(input) {
            Ok(Self::Identifier(ident))
        } else {
            Err(ParserError::from(format!("Expected bin op atom. Got {:?}", input.peek1())))
        }
    }
}

impl BinOpMathExprAST {
    fn precedence(op: BinOpMath) -> usize {
        match op {
            BinOpMath::MUL     => 100,
            BinOpMath::DIV     => 100,
            BinOpMath::MOD     => 100,
            BinOpMath::ADD     => 90,
            BinOpMath::SUB     => 90,
            BinOpMath::BIT_AND => 80,
            BinOpMath::BIT_OR  => 70
        }
    }

    fn mk_tree(atoms: Vec<BinOpExprAtom>, ops: Vec<BinOpMath>) -> Result<Self, ParserError> {
        let min_precedence = |slice: &[BinOpMath]| {
            slice.into_iter()
                .map(|op| Self::precedence(*op))
                .fold(10000, |min, x| if min > x { x } else { min })
        };

        println!("ATOMS: {:?}", atoms);
        println!("OPS:   {:?}", ops);

        if atoms.len() == 2 && ops.len() == 1 {
            let l = atoms.first().unwrap().clone();
            let r = atoms.last().unwrap().clone();
            let op = *ops.first().unwrap();
            let expr = Self { lhs: l.valuelike(), rhs: r.valuelike(), op: op };
            return Ok(expr);
        }

        let mut split = 0;
        while split < ops.len() {
            let op = *ops.get(split).unwrap();
            let rest = &ops[(split+1)..];

            if min_precedence(rest) < Self::precedence(op) {
                split += 1;
            } else {
                let l_ops = &ops[..split];
                let r_ops = &ops[split+1..];
                let l_atoms = &atoms[..split];
                let r_atoms = &atoms[split+1..];

                let lhs = if l_ops.len() > 0 {
                    ValuelikeExprAST::BinExpression(Box::new(
                        Self::mk_tree(Vec::from(l_atoms), Vec::from(l_ops))?
                    ))
                } else {
                    atoms.first().unwrap().clone().valuelike()
                };

                let rhs = if r_ops.len() > 0 {
                    ValuelikeExprAST::BinExpression(Box::new(
                        Self::mk_tree(Vec::from(r_atoms), Vec::from(r_ops))?
                    ))
                } else {
                    atoms.last().unwrap().clone().valuelike()
                };

                let expr = Self { lhs: lhs, rhs:  rhs, op: op };
                return Ok(expr)
            }
        }

        Err(ParserError::from("Unable to create expression tree."))
    }
}

impl ASTNode for BinOpMathExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        let mut atoms = Vec::new();
        let mut ops   = Vec::new();

        loop {
            let atom = BinOpExprAtom::run_parser(input);
            atoms.push(atom?);

            match input.peek1() {
                Some(Token::BIN_OP_MATH(op)) => {
                    input.next(); // eat token
                    ops.push(op);
                },
                _ => break
            }
        }


        if ops.len() < 1 || atoms.len() < 2 || ops.len() != atoms.len() - 1 {
            Err(ParserError::from(format!("Invalid math expression. Operands: '{:?}', operations: '{:?}'", atoms, ops)))?
        }

        Self::mk_tree(atoms, ops)
    }
}

#[derive(Debug, Clone)]
pub enum ValuelikeExprAST {
    NumericLiteral(NumLiteralExprAST),
    BinExpression(Box<BinOpMathExprAST>),
    Identifier(IdentifierExprAST),
    Call(CallExprAST)
}

#[derive(Debug, Clone)]
pub struct CallExprAST {
    name: IdentifierExprAST,
    args: Vec<ValuelikeExprAST>
}

#[derive(Debug, Clone)]
pub struct AssignmentExprAST {
    ident: IdentifierExprAST,
    value: ValuelikeExprAST
}

#[derive(Debug, Clone)]
pub struct BlockExprAST {
    body: Vec<InBlockExprAST>
}

#[derive(Debug, Clone)]
pub enum InBlockExprAST {
    Assingment(AssignmentExprAST),
    Valuelike(ValuelikeExprAST),
    If(IfElseExprAST)
}

#[derive(Debug, Clone)]
pub struct IfElseExprAST {
    // Temporary
    cond: IdentifierExprAST,
    block_if: BlockExprAST,
    block_else: Option<BlockExprAST>
}

#[derive(Debug, Clone)]
pub struct FuncDefExprAST {
    prototype: FuncPrototypeExprAST,
    body: BlockExprAST
}

#[derive(Debug, Clone)]
pub enum OutBlockExprAST {
    Assingment(AssignmentExprAST),
    FuncDef(FuncDefExprAST)
}

#[derive(Debug, Clone)]
pub struct RootExprAST {
    items: Vec<OutBlockExprAST>
}

impl ParserError {
    pub fn from(string: impl Into<String>) -> Self {
        Self { text: string.into() }
    }
}

#[derive(Debug, Clone)]
pub struct FuncPrototypeExprAST {
    name: IdentifierExprAST,
    args: Vec<IdentifierExprAST>
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

impl ASTNode for NumLiteralExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        let next = input.next();
        if let Some(Token::NUM(val)) = next {
            Ok(Self { value: val })
        } else {
            Err(ParserError::from(format!("Expected numeric. Got {:?}", next)))
        }
    }
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

impl ASTNode for ValuelikeExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        if let Ok(expr) = BinOpMathExprAST::run_parser(input) {
            Ok(Self::BinExpression(Box::new(expr)))
        } else if let Ok(num) = NumLiteralExprAST::run_parser(input) {
            Ok(Self::NumericLiteral(num))
        } else if let Ok(ident) = CallExprAST::run_parser(input) {
            Ok(Self::Call(ident))
        } else if let Ok(ident) = IdentifierExprAST::run_parser(input) {
            Ok(Self::Identifier(ident))
        } else {
            Err(ParserError::from(format!("Expected valuelike. Got {:?}", input.peek1())))
        }
    }
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

impl ASTNode for InBlockExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        if let Ok(asign) = AssignmentExprAST::run_parser(input) {
            Ok(Self::Assingment(asign))
        } else if let Ok(iff) = IfElseExprAST::run_parser(input) {
            Ok(Self::If(iff))
        } else if let Ok(value) = ValuelikeExprAST::run_parser(input) {
            Ok(Self::Valuelike(value))
        } else {
            Err(ParserError::from(format!("Expected valuelike. Got {:?}", input.peek1())))
        }
    }
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

impl ASTNode for FuncDefExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        let prototype = FuncPrototypeExprAST::run_parser(input)?;
        let block = BlockExprAST::run_parser(input)?;
        Ok(Self { prototype: prototype, body: block })
    }
}

impl ASTNode for OutBlockExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        if let Ok(asign) = AssignmentExprAST::run_parser(input) {
            Ok(Self::Assingment(asign))
        } else if let Ok(def) = FuncDefExprAST::run_parser(input) {
            Ok(Self::FuncDef(def))
        } else {
            let err = format!("Expected assignment or func definition. Got {:?}", input.peek1());
            Err(ParserError::from(err))
        }
    }
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
