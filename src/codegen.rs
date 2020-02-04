extern crate llvm;

use super::ast::*;
use super::token::*;

use std::collections::HashMap;

use llvm::{
    CBox,
    CSemiBox,
    Value,
    GlobalValue,
    Function
};
// llvm-rs docs:
// https://tombebbington.github.io/llvm-rs/llvm/index.html

const MOD_NAME: &'static str = "llvm-tut";

struct Context<'r> {
    llvm_ctx: &'r llvm::Context,
    llvm_module: CSemiBox<'r, llvm::Module>,
    llvm_builder: CSemiBox<'r, llvm::Builder>,
    funcs: HashMap<String, &'r Function>,
    glob_vars: HashMap<String, &'r GlobalValue>,
    local_vars: HashMap<String, &'r Value>
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
}

trait Codegen {
    fn gencode(&mut self, ctx: &mut Context) {}
}

impl Codegen for RootExprAST {
    fn gencode(&mut self, ctx: &mut Context) {}
}

pub fn gencode(root: &mut RootExprAST) {
    let ctx = llvm::Context::new();
    let mut scope = Context::new(&ctx);
    root.gencode(&mut scope)
}
