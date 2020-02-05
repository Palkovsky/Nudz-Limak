extern crate llvm;

use super::ast::*;
use super::token::*;

use std::process::Command;
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

fn mk_slice<T>(vec: &Vec<Box<T>>) -> Vec<&T> {
    unsafe {
        let ptrs = vec
            .iter()
            .map(|item| item.as_ref() as *const T)
            .map(|ptr| ptr.as_ref().unwrap())
            .collect::<Vec<&T>>();
        ptrs
    }
}

struct Context<'r> {
    llvm_ctx: &'r llvm::Context,
    llvm_module: CSemiBox<'r, llvm::Module>,
    llvm_builder: CSemiBox<'r, llvm::Builder>,
    funcs: HashMap<String, Box<Function>>,
    glob_vars: HashMap<String, Box<Value>>,
    local_vars: HashMap<String, Box<Value>>,
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

    fn variable(&self, name: impl Into<String>) -> Option<Box<Value>> {
        let ref key = name.into();
        let global = self.glob_vars.get(key);
        let local  = self.local_vars.get(key);

        // Local variable has higher priority
        local.or(global)
            .map(|x| mk_box(x.as_ref()))
    }

    fn func(&self, name: impl Into<String>) -> Option<Box<Function>> {
        let ref key = name.into();
        self.funcs.get(key)
            .map(|x| mk_box(x.as_ref()))
    }

    fn mk_func(&mut self, name: impl AsRef<str>, ty: &llvm::Type) -> Box<Function> {
        let key = name.as_ref();
        let func = mk_box(self.llvm_module.add_function(key, ty));

        // self.llvm_builder.position_at_end(func.append("entrypoint"));

        self.funcs.insert(String::from(key), func);
        self.func(key).unwrap()
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
                    val
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
                    f.unwrap()
                };

                let ref args = call.args
                    .iter()
                    .map(|arg| arg.gencode(ctx))
                    .collect::<Vec<Box<Value>>>();
                let argslice = &mk_slice(args)[..];

                let value = ctx.llvm_builder.build_call(func.as_ref(), argslice);
                mk_box(value)

            },
            ValuelikeExprAST::BinExpression(binexpr) =>
                binexpr.gencode(ctx),
            ValuelikeExprAST::UnaryExpression(unary) =>
                unary.gencode(ctx)
        }
    }
}

impl<'r> Codegen<'r, Box<Value>> for BinOpExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> Box<Value> {
        let ref lhs = self.lhs.gencode(ctx);
        let ref rhs = self.rhs.gencode(ctx);
        let ref builder = ctx.llvm_builder;

        let res = match self.op {
            BinOp::ADD =>
                builder.build_add(lhs, rhs),
            BinOp::SUB =>
                builder.build_sub(lhs, rhs),
            BinOp::MUL =>
                builder.build_mul(lhs, rhs),
            BinOp::DIV =>
                builder.build_div(lhs, rhs),
            BinOp::MOD =>
                panic!("Modulo not implemented"),
            BinOp::BIT_AND | BinOp::AND =>
                builder.build_and(lhs, rhs),
            BinOp::BIT_OR | BinOp::OR =>
                builder.build_or(lhs, rhs),
            BinOp::LT =>
                builder.build_cmp(lhs, rhs, llvm::Predicate::LessThan),
            BinOp::LTE =>
                builder.build_cmp(lhs, rhs, llvm::Predicate::LessThanOrEqual),
            BinOp::GT =>
                builder.build_cmp(lhs, rhs, llvm::Predicate::GreaterThan),
            BinOp::GTE =>
                builder.build_cmp(lhs, rhs, llvm::Predicate::GreaterThanOrEqual),
            BinOp::EQ =>
                builder.build_cmp(lhs, rhs, llvm::Predicate::Equal),
            BinOp::NON_EQ =>
                builder.build_cmp(lhs, rhs, llvm::Predicate::NotEqual),
        };

        mk_box(res)
    }
}

#[test]
fn bin_expr_test() {
    let ref mut llvm = llvm::Context::new();
    let ref mut ctx = Context::new(llvm);

    let num1 = ValuelikeExprAST::NumericLiteral(NumLiteralExprAST { value: 88.88 });
    let num2 = ValuelikeExprAST::NumericLiteral(NumLiteralExprAST { value: 44.44 });
    let expr1 = ValuelikeExprAST::UnaryExpression(Box::new(
        UnaryOpExprAST {
            op: UnaryOp::NOT,
            expr: num1
        }
    ));

    let expr = ValuelikeExprAST::BinExpression(Box::new(
        BinOpExprAST {
            lhs: expr1,
            rhs: num2,
            op: BinOp::MUL
        }
    ));

    println!("{:=<1$}", "", 50);
    println!("{:?} => {:?}", expr, expr.gencode(ctx));
    println!("{:=<1$}", "", 50);

    assert!(false);
}

impl<'r> Codegen<'r, Box<Value>> for UnaryOpExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> Box<Value> {
        let value = self.expr.gencode(ctx);
        let ref builder = ctx.llvm_builder;

        let value_op = match self.op {
            UnaryOp::NOT =>
                builder.build_not(value.as_ref()),
            UnaryOp::MINUS =>
                builder.build_neg(value.as_ref())
        };

        mk_box(value_op)
    }
}

#[test]
fn unary_expr_test() {
    let ref mut llvm = llvm::Context::new();
    let ref mut ctx = Context::new(llvm);
    let num1 = ValuelikeExprAST::NumericLiteral(NumLiteralExprAST { value: 88.88 });
    let expr1 = UnaryOpExprAST {
        op: UnaryOp::NOT,
        expr: num1
    };
    let expr2 = UnaryOpExprAST {
        op: UnaryOp::MINUS,
        expr: ValuelikeExprAST::UnaryExpression(Box::new(expr1))
    };

    println!("{:=<1$}", "", 80);
    println!("{:?} => {:?}", expr2, expr2.gencode(ctx));
    println!("{:=<1$}", "", 80);
    assert!(false);
}

impl<'r> Codegen<'r, ()> for RootExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> () {}
}

fn module_disasm(ctx: &Context) -> String {
    let mktemp = || {
        let stdout = Command::new("mktemp")
            .output()
            .unwrap()
            .stdout;

        String::from(
            std::str::from_utf8(&stdout[..])
                .unwrap())
    };

    let temp1 = mktemp();
    let temp1 = temp1.trim();
    let temp2 = mktemp();
    let temp2 = temp2.trim();

    // Run library and dump bitcode in the temp directory
    ctx.llvm_module
        .verify()
        .unwrap();

    ctx.llvm_module
        .write_bitcode(&temp1)
        .unwrap();

    // Run llvm-dis and dump disassembly in the temp directory
    Command::new("llvm-dis").arg(temp1).arg("-o").arg(temp2)
        .status()
        .unwrap();

    // Read file with disassembly
    let cat_stdout = Command::new("cat").arg(temp2)
        .output()
        .unwrap()
        .stdout;

    let disasm = std::str::from_utf8(&cat_stdout[..])
        .unwrap()
        .to_string();

    // Cleanup temps
    Command::new("rm").arg(temp1).arg(temp2).status().unwrap();

    disasm
}

fn valuelike_disasm(binexpr: &ValuelikeExprAST) -> String {
    let ref mut llvm = llvm::Context::new();
    let ref mut ctx = Context::new(llvm);

    // Add test function
    let func = ctx.mk_func("add", llvm::Type::get::<fn(f64, f64) -> f64>(ctx.llvm_ctx));
    ctx.llvm_builder.position_at_end(func.append("entrypoint"));
    ctx.llvm_builder.build_ret(88.88f64.compile(ctx.llvm_ctx));

    // This type signature won't be possible to run.
    // Executables must have C-like signature: int main()/int main(int argc, char**argv)
    let func = mk_box(
        ctx.llvm_module.add_function("main",
                                     llvm::Type::get::<fn() -> f64>(ctx.llvm_ctx)));

    let entrypoint = func.append("entrypoint");
    ctx.llvm_builder.position_at_end(entrypoint);

    let value = binexpr.gencode(ctx);
    ctx.llvm_builder.build_ret(value.as_ref());

    module_disasm(ctx)
}

#[test]
fn simple_binexpr_codegen_test() {
    let ref mut tokens = mk_tokens("add(add(1, 2+3), (-5+2)*3/2)".to_owned()).unwrap();
    let ref binexpr = ValuelikeExprAST::run_parser(tokens).unwrap();
    println!("{}", valuelike_disasm(binexpr));
    assert!(false);
}

pub fn gencode(root: &RootExprAST) {
    let ref mut llvm = llvm::Context::new();
    let ref mut ctx = Context::new(llvm);
    root.gencode(ctx)
}
