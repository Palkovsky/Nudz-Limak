extern crate llvm;

use super::ast::*;
use super::token::*;

use std::collections::HashMap;

// llvm-rs docs:
// https://tombebbington.github.io/llvm-rs/llvm/index.html
use llvm::{
    CSemiBox,
    Value,
    Function,
    Compile
};

const MOD_NAME: &'static str = "llvm-tut";

fn mk_box<T>(value: &T) -> Box<T> {
    let ptr = (value as *const T) as usize;
    unsafe { Box::from_raw(ptr as *mut T) }
}

struct Context<'r> {
    llvm_ctx: &'r llvm::Context,
    llvm_module: CSemiBox<'r, llvm::Module>,
    llvm_builder: CSemiBox<'r, llvm::Builder>,
    funcs: HashMap<String, &'r Function>,
    glob_vars: HashMap<String, &'r Value>,
    local_vars: HashMap<String, &'r Value>,
}

impl<'r> Context<'r> {
    fn new(ctx: &'r llvm::Context) -> Self {
        let module = llvm::Module::new(MOD_NAME, ctx);
        let builder = llvm::Builder::new(ctx);
        Self {
            llvm_ctx: ctx,
            llvm_module: module,
            llvm_builder: builder,
            funcs: HashMap::new(),
            glob_vars: HashMap::new(),
            local_vars: HashMap::new(),
        }
    }

    fn variable(&self, name: impl Into<String>) -> Option<&'r Value> {
        let ref key = name.into();
        let global = self.glob_vars.get(key);
        let local  = self.local_vars.get(key);

        // Local variable has higher priority
        local.or(global)
            .map(|x| *x)
    }

    fn func(&self, name: impl Into<String>) -> Option<&'r Function> {
        let ref key = name.into();
        self.funcs.get(key)
            .map(|x| *x)
    }
}

trait Codegen<'r, T> {
    fn gencode(&self, ctx: &mut Context<'r>) -> T;
}

impl<'r> Codegen<'r, Box<Value>> for NumLiteralExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> Box<Value> {
        let num = self.value.compile(ctx.llvm_ctx);
        mk_box(num)
    }
}

#[test]
fn num_literal_expr_test() {
    let ref mut llvm = llvm::Context::new();
    let ref mut ctx = Context::new(llvm);
    let expr = NumLiteralExprAST { value: 88.88 };
    println!("{:=<1$}", "", 80);
    println!("{:?} => {:?}", expr, expr.gencode(ctx));
    println!("{:=<1$}", "", 80);
    assert!(false);
}

impl<'r> Codegen<'r, Box<Value>> for ValuelikeExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> Box<Value> {
        match self {
            ValuelikeExprAST::NumericLiteral(num) =>
                num.gencode(ctx),
            ValuelikeExprAST::Variable(ident) => {
                let ref key = ident.name;
                if let Some(val) = ctx.variable(key) {
                    mk_box(val)
                } else {
                    panic!("'{}' not found.", key)
                }
            },
            ValuelikeExprAST::Call(call) => {
                let ref name = call.name.name;
                let func = {
                    let f = ctx.func(name);
                    if f.is_none() {
                        panic!(" Function '{}' is undefined.", name);
                    }
                    f
                };

                let mut args = call.args
                    .iter()
                    .map(|arg| arg.gencode(ctx))
                    .collect::<Vec<Box<Value>>>();

                // Some arg-checks
                panic!("Function calls not supported.");

            },
            ValuelikeExprAST::BinExpression(expr) => {
                panic!("")
            },
            ValuelikeExprAST::UnaryExpression(unary) => {
                unary.gencode(ctx)
            }
        }
    }
}

impl<'r> Codegen<'r, Box<Value>> for UnaryOpExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> Box<Value> {
        let value = self.expr.gencode(ctx);

        let value_op = match self.op {
            UnaryOp::NOT   => ctx.llvm_builder.build_not(value.as_ref()),
            UnaryOp::MINUS => ctx.llvm_builder.build_neg(value.as_ref())
        };

        mk_box(value_op)
    }
}

#[test]
fn unary_neg_expr_test() {
    let ref mut llvm = llvm::Context::new();
    let ref mut ctx = Context::new(llvm);
    let num = ValuelikeExprAST::NumericLiteral(NumLiteralExprAST { value: 88.88 });
    let expr = UnaryOpExprAST {
        op: UnaryOp::NOT,
        expr: num
    };
    println!("{:=<1$}", "", 80);
    println!("{:?} => {:?}", expr, expr.gencode(ctx));
    println!("{:=<1$}", "", 80);
    assert!(false);
}

impl<'r> Codegen<'r, ()> for RootExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> () {}
}

pub fn gencode(root: &mut RootExprAST) {
    let ref mut llvm = llvm::Context::new();
    let ref mut ctx = Context::new(llvm);
    root.gencode(ctx)
}
