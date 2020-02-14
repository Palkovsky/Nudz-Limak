use super::token::{
    Token,
    BinOp,
    UnaryOp
};
use lexer::Stream;

use std::{error, fmt};

#[derive(Debug, PartialEq, Clone)]
pub struct ParserError {
    pub text: String
}

#[derive(Debug, PartialEq, Clone)]
struct SingleTokenExprAST;

#[derive(Debug, PartialEq, Clone)]
pub struct VoidTypeExprAST;

#[derive(Debug, PartialEq, Clone)]
pub struct NumLiteralExprAST {
    pub value: u64
}

#[derive(Debug, PartialEq, Clone)]
pub struct IdentifierExprAST {
    pub name: String
}

#[derive(Debug, PartialEq, Clone)]
pub enum PrimitiveTypeExprAST {
    Byte,
    Int,
    Short,
    Long,
    Void
}

#[derive(Debug, PartialEq, Clone)]
pub struct PtrTypeExprAST {
    pub pointee: Box<TypeExprAST>
}

#[derive(Debug, PartialEq, Clone)]
pub enum TypeExprAST {
    Primitive(PrimitiveTypeExprAST),
    Pointer(PtrTypeExprAST)
}

#[derive(Debug, PartialEq, Clone)]
struct TypeSpecAST {
    ty: TypeExprAST
}

#[derive(Debug, PartialEq, Clone)]
pub struct UnaryOpExprAST {
    pub op: UnaryOp,
    pub expr: ValuelikeExprAST
}

#[derive(Debug, PartialEq, Clone)]
pub enum BinOpAtomAST {
    NumericLiteral(NumLiteralExprAST),
    Variable(IdentifierExprAST),
    Paren(Box<ParenExprAST>),
    Unary(UnaryOpExprAST),
    Call(CallExprAST)
}

#[derive(Debug, PartialEq, Clone)]
pub struct BinOpExprAST {
    pub lhs: ValuelikeExprAST,
    pub op: BinOp,
    pub rhs: ValuelikeExprAST
}

#[derive(Debug, PartialEq, Clone)]
pub struct ParenExprAST {
    pub body: ValuelikeExprAST
}

#[derive(Debug, PartialEq, Clone)]
pub enum ValuelikeExprAST {
    NumericLiteral(NumLiteralExprAST),
    BinExpression(Box<BinOpExprAST>),
    UnaryExpression(Box<UnaryOpExprAST>),
    Variable(IdentifierExprAST),
    Call(CallExprAST)
}

#[derive(Debug, PartialEq, Clone)]
pub struct CallExprAST {
    pub name: IdentifierExprAST,
    pub args: Vec<ValuelikeExprAST>
}

#[derive(Debug, PartialEq, Clone)]
pub struct ReturnExprAST {
    pub ret: Option<ValuelikeExprAST>
}

#[derive(Debug, PartialEq, Clone)]
pub enum ArrayDeclarationExprAST {
    BySize(ValuelikeExprAST),
    ByElements(Vec<ValuelikeExprAST>)
}

#[derive(Debug, PartialEq, Clone)]
pub enum DeclarationRHSExprAST {
    Valuelike(ValuelikeExprAST),
    Array(ArrayDeclarationExprAST)
}

#[derive(Debug, PartialEq, Clone)]
pub struct DeclarationExprAST {
    pub ident: IdentifierExprAST,
    pub ty: TypeExprAST,
    pub value: DeclarationRHSExprAST
}

#[derive(Debug, PartialEq, Clone)]
pub struct PtrWriteExprAST {
    pub target: ValuelikeExprAST
}

#[derive(Debug, PartialEq, Clone)]
pub enum WritableExprAST {
    Variable(IdentifierExprAST),
    PtrWrite(PtrWriteExprAST)
}

#[derive(Debug, PartialEq, Clone)]
pub struct ReDeclarationExprAST {
    pub target: WritableExprAST,
    pub value: ValuelikeExprAST
}

#[derive(Debug, PartialEq, Clone)]
pub struct BlockExprAST {
    pub body: Vec<InBlockExprAST>
}

#[derive(Debug, PartialEq, Clone)]
pub enum InBlockExprAST {
    Declaration(DeclarationExprAST),
    ReDeclaration(ReDeclarationExprAST),
    Valuelike(ValuelikeExprAST),
    If(IfElseExprAST),
    While(WhileExprAST),
    Return(ReturnExprAST)
}

#[derive(Debug, PartialEq, Clone)]
pub struct IfElseExprAST {
    pub cond: ValuelikeExprAST,
    pub block_if: BlockExprAST,
    pub block_else: Option<BlockExprAST>
}

#[derive(Debug, PartialEq, Clone)]
pub struct WhileExprAST {
    pub cond: ValuelikeExprAST,
    pub body: BlockExprAST
}

#[derive(Debug, PartialEq, Clone)]
pub struct FuncDefExprAST {
    pub prototype: FuncPrototypeExprAST,
    pub body: BlockExprAST
}

#[derive(Debug, PartialEq, Clone)]
pub enum OutBlockExprAST {
    Declaration(DeclarationExprAST),
    FuncDef(FuncDefExprAST)
}

#[derive(Debug, PartialEq, Clone)]
pub struct RootExprAST {
    pub items: Vec<OutBlockExprAST>
}

impl ParserError {
    pub fn from(string: impl Into<String>) -> Self {
        Self { text: string.into() }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct FuncPrototypeExprAST {
    pub name: IdentifierExprAST,
    pub ret_type: TypeExprAST,
    pub args: Vec<(IdentifierExprAST, TypeExprAST)>
}

impl error::Error for ParserError {}
impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let txt = format!("ParserrError: {}", self.text);
        f.write_str(&txt)
    }
}

pub trait ASTNode<T=Self>: Sized {
    fn parse(input: &mut impl Stream<Token>) -> Result<T, ParserError>;
    fn run_parser(input: &mut impl Stream<Token>) -> Result<T, ParserError> {
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
            _ =>
                None
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

impl ASTNode for VoidTypeExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        let p1 = SingleTokenExprAST::expect(Token::PAREN_OP, input);
        let p2 = SingleTokenExprAST::expect(Token::PAREN_CL, input);
        if p1.is_ok() && p2.is_ok() {
            Ok(Self)
        } else {
            Err(ParserError::from("Expected ()"))
        }
    }
}


impl ASTNode for NumLiteralExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        let next = input.next();
        if let Some(Token::NUM(val)) = next {
            Ok(Self { value: val as u64 })
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

impl ASTNode for PrimitiveTypeExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        if VoidTypeExprAST::run_parser(input).is_ok() {
            return Ok(Self::Void)
        }

        match input.next() {
            Some(Token::IDENT(x)) => {
                match x.as_str() {
                    "byte"  => Ok(Self::Byte),
                    "short" => Ok(Self::Short),
                    "long"  => Ok(Self::Long),
                    "int"   => Ok(Self::Int),
                    _ => {
                        let err = format!("Expected type specification. Got {:?}", x);
                        Err(ParserError::from(err))
                    }
                }
            },
            x => {
                let err = format!("Expected type specification. Got {:?}", x);
                Err(ParserError::from(err))
            }
        }
    }
}

impl ASTNode for PtrTypeExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        SingleTokenExprAST::expect(Token::UNARY_OP(UnaryOp::DEREF), input)?;
        let pointee = TypeExprAST::run_parser(input)?;
        Ok( Self { pointee: Box::new(pointee) })
    }
}

impl ASTNode for TypeExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        if let Ok(primitive) = PrimitiveTypeExprAST::run_parser(input) {
            Ok(Self::Primitive(primitive))
        } else if let Ok(ptr) = PtrTypeExprAST::run_parser(input) {
            Ok(Self::Pointer(ptr))
        } else {
            Err(ParserError::from("Expected type."))
        }
    }
}

impl ASTNode<TypeExprAST> for TypeSpecAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<TypeExprAST, ParserError> {
        SingleTokenExprAST::expect(Token::COLON, input)?;
        TypeExprAST::run_parser(input)
    }
}

impl ASTNode for UnaryOpExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        match input.next() {
            Some(Token::UNARY_OP(op)) => {
                let expr = BinOpAtomAST::run_parser(input)?;
                Ok(Self {op: op, expr: expr.valuelike()})
            },
            Some(Token::BIN_OP(BinOp::SUB)) => {
                let expr = BinOpAtomAST::run_parser(input)?;
                Ok(Self {op: UnaryOp::MINUS, expr: expr.valuelike()})
            },
            _ => Err(ParserError::from("Expected valuelike."))
        }
    }
}

impl ASTNode for ParenExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        SingleTokenExprAST::expect(Token::PAREN_OP, input)?;
        let paren_body = ValuelikeExprAST::run_parser(input)?;
        SingleTokenExprAST::expect(Token::PAREN_CL, input)?;
        Ok(Self {body: paren_body})
    }
}

impl BinOpAtomAST {
    fn valuelike(self) -> ValuelikeExprAST {

        match self {
            Self::NumericLiteral(num) =>
                ValuelikeExprAST::NumericLiteral(num),
            Self::Variable(ident) =>
                ValuelikeExprAST::Variable(ident),
            Self::Call(call) =>
                ValuelikeExprAST::Call(call),
            Self::Paren(paren) =>
                paren.body,
            Self::Unary(unary) =>
                ValuelikeExprAST::UnaryExpression(Box::new(unary))
        }
    }
}

impl ASTNode for BinOpAtomAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        if let Ok(unary) = UnaryOpExprAST::run_parser(input) {
            Ok(Self::Unary(unary))
        } else if let Ok(paren) = ParenExprAST::run_parser(input) {
            Ok(Self::Paren(Box::new(paren)))
        } else if let Ok(num) = NumLiteralExprAST::run_parser(input) {
            Ok(Self::NumericLiteral(num))
        } else if let Ok(ident) = CallExprAST::run_parser(input) {
            Ok(Self::Call(ident))
        } else if let Ok(ident) = IdentifierExprAST::run_parser(input) {
            Ok(Self::Variable(ident))
        } else {
            Err(ParserError::from(format!("Expected bin op atom. Got {:?}", input.peek1())))
        }
    }
}

impl BinOpExprAST {
    fn precedence(op: BinOp) -> usize {
        match op {
            BinOp::MUL | BinOp::DIV | BinOp::MOD =>
                100,
            BinOp::ADD | BinOp::SUB =>
                90,
            BinOp::GT | BinOp::GTE | BinOp::LT | BinOp::LTE =>
                80,
            BinOp::EQ | BinOp::NON_EQ =>
                75,
            BinOp::BIT_AND =>
                70,
            BinOp::BIT_OR  =>
                60,
            BinOp::AND =>
                50,
            BinOp::OR =>
                40
        }
    }

    fn min_precedence(slice: &[BinOp]) -> usize {
        slice.into_iter()
            .map(|op| Self::precedence(*op))
            .fold(std::usize::MAX, |min, x| if min > x { x } else { min })
    }

    fn mk_tree(atoms: Vec<BinOpAtomAST>, ops: Vec<BinOp>) -> Result<Self, ParserError> {
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

            if Self::min_precedence(rest) < Self::precedence(op) {
                split += 1;
            } else {
                let l_ops = &ops[..split];
                let r_ops = &ops[split+1..];
                let l_atoms = &atoms[..split+1];
                let r_atoms = &atoms[split+1..];

                let lhs = if l_ops.len() > 0 {
                    ValuelikeExprAST::BinExpression(Box::new(
                        Self::mk_tree(Vec::from(l_atoms), Vec::from(l_ops))?
                    ))
                } else {
                    atoms.first().unwrap().clone().valuelike()
                };

                let rhs = if r_ops
                    .len() > 0 {
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

impl ASTNode for BinOpExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        let mut atoms = Vec::new();
        let mut ops   = Vec::new();

        loop {
            let atom = BinOpAtomAST::run_parser(input);
            atoms.push(atom?);

            match input.peek1() {
                Some(Token::BIN_OP(op)) => {
                    input.next(); // eat token
                    ops.push(op);
                },
                _ =>
                    break
            }
        }

        if ops.len() < 1 || atoms.len() < 2 || ops.len() != atoms.len() - 1 {
            Err(ParserError::from(format!("Invalid math expression. Operands: '{:?}', operations: '{:?}'", atoms, ops)))?
        }

        Self::mk_tree(atoms, ops)
    }
}

impl ASTNode for CallExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        let ident = IdentifierExprAST::run_parser(input)?;

        SingleTokenExprAST::expect(Token::PAREN_OP, input)?;

        let mut args = Vec::new();

        loop {
            // Argless function
            if SingleTokenExprAST::expect(Token::PAREN_CL, input).is_ok() {
                break;
            }

            args.push(ValuelikeExprAST::run_parser(input)?);

            match input.next() {
                Some(Token::COMMA) =>
                    {},
                Some(Token::PAREN_CL) =>
                    break,
                _ =>
                    Err(ParserError::from("Expected ',' or ')'."))?
            };
        }

        Ok(Self { name: ident, args: args })
    }
}

impl ASTNode for ReturnExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        SingleTokenExprAST::expect(Token::RETURN, input)?;

        // 'return ()'
        if VoidTypeExprAST::run_parser(input).is_ok() {
            return Ok(Self {ret: None})
        }

        let ret = if let Ok(valuelike) = ValuelikeExprAST::run_parser(input) {
            Some(valuelike)
        } else {
            None
        };

        Ok(Self { ret: ret })
    }
}

impl ValuelikeExprAST {
    // This should be more complex.
    // It should for example accept binary/unary epxpressions of literals.
    pub fn is_literal(&self) -> bool {
        match self {
            Self::NumericLiteral(_) => true,
            _ => false
        }
    }
}

impl ASTNode for ValuelikeExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        if let Ok(expr) = BinOpExprAST::run_parser(input) {
            Ok(Self::BinExpression(Box::new(expr)))
        } else if let Ok(unary) = UnaryOpExprAST::run_parser(input) {
            Ok(Self::UnaryExpression(Box::new(unary)))
        } else if let Ok(paren) = ParenExprAST::run_parser(input) {
             Ok(paren.body)
        } else if let Ok(num) = NumLiteralExprAST::run_parser(input) {
            Ok(Self::NumericLiteral(num))
        } else if let Ok(ident) = CallExprAST::run_parser(input) {
            Ok(Self::Call(ident))
        } else if let Ok(ident) = IdentifierExprAST::run_parser(input) {
            Ok(Self::Variable(ident))
        } else {
            Err(ParserError::from(format!("Expected valuelike. Got {:?}", input.peek1())))
        }
    }
}

impl ASTNode for ArrayDeclarationExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        SingleTokenExprAST::expect(Token::BRACKET_OP, input)?;

        let mut items = Vec::new();
        loop {
            items.push(ValuelikeExprAST::run_parser(input)?);

            match input.next() {
                Some(Token::COMMA) => {},
                Some(Token::BRACKET_CL) =>
                    break,
                _ =>
                    return Err(ParserError::from(format!("Expected array declaration.")))
            }
        }

        match &items[..] {
            [] => {
                Err(ParserError::from(format!("Empty array declaration.")))
            },
            [valuelike] => {
                Ok(Self::BySize(valuelike.clone()))
            },
            _ => {
                Ok(Self::ByElements(items) )
            }
        }
    }
}

impl ASTNode for DeclarationRHSExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        if let Ok(array) = ArrayDeclarationExprAST::run_parser(input) {
            Ok(Self::Array(array))
        } else if let Ok(valuelike) = ValuelikeExprAST::run_parser(input) {
            Ok(Self::Valuelike(valuelike))
        } else {
            Err(ParserError::from(format!("Expected declaration RHS.")))
        }
    }
}

impl ASTNode for DeclarationExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        SingleTokenExprAST::expect(Token::LET, input)?;

        let ident = IdentifierExprAST::run_parser(input)?;
        let ty = TypeSpecAST::run_parser(input)?;

        match input.next() {
            Some(Token::ASSIGNMENT) => {
                let value = DeclarationRHSExprAST::run_parser(input)?;
                Ok(Self { ident: ident, ty: ty, value: value })
            },
            next => {
                Err(ParserError::from(format!("Expected '=' got '{:?}'.", next)))
            }
        }
    }
}

impl ASTNode for PtrWriteExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        SingleTokenExprAST::expect(Token::UNARY_OP(UnaryOp::DEREF), input)?;
        let valuelike = if let Ok(paren) = ParenExprAST::run_parser(input) {
            paren.body
        } else if let Ok(ident) = IdentifierExprAST::run_parser(input) {
            ValuelikeExprAST::Variable(ident)
        } else {
            return Err(ParserError::from("Expected *(...)"))
        };

        Ok(Self { target: valuelike })
    }
}

impl ASTNode for WritableExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        if let Ok(ptrwrite) = PtrWriteExprAST::run_parser(input) {
            return Ok(Self::PtrWrite(ptrwrite))
        } else if let Ok(ident) = IdentifierExprAST::run_parser(input) {
            return Ok(Self::Variable(ident));
        }
        Err(ParserError::from(format!("Expected writable got '{:?}'.", input.peek1())))
    }
}

impl ASTNode for ReDeclarationExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        let writable = WritableExprAST::run_parser(input)?;

        match input.next() {
            Some(Token::ASSIGNMENT) => {
                let value = ValuelikeExprAST::run_parser(input)?;
                Ok(Self { target: writable, value: value })
            },
            next => {
                Err(ParserError::from(format!("Expected '=' got '{:?}'.", next)))
            }
        }
    }
}

impl ASTNode for InBlockExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        if let Ok(ret) = ReturnExprAST::run_parser(input) {
            Ok(Self::Return(ret))
        } else if let Ok(decl) = DeclarationExprAST::run_parser(input) {
            Ok(Self::Declaration(decl))
        } else if let Ok(redecl) = ReDeclarationExprAST::run_parser(input) {
            Ok(Self::ReDeclaration(redecl))
        } else if let Ok(iff) = IfElseExprAST::run_parser(input) {
            Ok(Self::If(iff))
        } else if let Ok(wwhile) = WhileExprAST::run_parser(input) {
            Ok(Self::While(wwhile))
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

        let cond = ValuelikeExprAST::run_parser(input)?;
        let block_if = BlockExprAST::run_parser(input)?;

        let block_else = if let Ok(_) = SingleTokenExprAST::expect(Token::ELSE, input) {
            Some(BlockExprAST::run_parser(input)?)
        } else {
            None
        };

        Ok(Self { cond: cond, block_if: block_if, block_else: block_else })
    }
}

impl ASTNode for WhileExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        SingleTokenExprAST::expect(Token::WHILE, input)?;
        let cond = ValuelikeExprAST::run_parser(input)?;
        let body = BlockExprAST::run_parser(input)?;
        Ok(Self { cond: cond, body: body })
    }
}

impl ASTNode for FuncPrototypeExprAST {
    fn parse(input: &mut impl Stream<Token>) -> Result<Self, ParserError> {
        SingleTokenExprAST::expect(Token::DEF, input)?;

        let ident = IdentifierExprAST::run_parser(input)?;

        SingleTokenExprAST::expect(Token::PAREN_OP, input)?;

        let mut args = Vec::new();
        loop {
            // Argless function
            if SingleTokenExprAST::expect(Token::PAREN_CL, input).is_ok() {
                break;
            }

            // Get argname
            let argname = IdentifierExprAST::run_parser(input)?;
            let ty = TypeSpecAST::run_parser(input)?;
            args.push((argname, ty));

            // Ignore comma, finish when closing paren hit
            match input.next() {
                Some(Token::COMMA) =>
                    {},
                Some(Token::PAREN_CL) =>
                    break,
                _ =>
                    Err(ParserError::from("Expected ',' or ')'."))?
            };
        }

        // Optional return type
        let ret_type = if SingleTokenExprAST::expect(Token::ARROW, input).is_ok() {
            TypeExprAST::run_parser(input)?
        } else {
            TypeExprAST::Primitive(PrimitiveTypeExprAST::Void)
        };

        Ok(Self { name: ident, ret_type: ret_type, args: args })
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
        if let Ok(decl) = DeclarationExprAST::run_parser(input) {
            Ok(Self::Declaration(decl))
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
