extern crate llvm;

use super::ast::*;
use super::token::*;

use std::process::Command;
use std::collections::HashMap;

// llvm-rs docs:
// https://tombebbington.github.io/llvm-rs/llvm/index.html
use llvm::{
    CSemiBox,
    BasicBlock,
    Value, GlobalValue, GlobalVariable,
    Function,
    Compile,
    Sub
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
    blocks: HashMap<String, HashMap<String, Box<BasicBlock>>>,

    ret_block: Option<String>,
    exit_block: Option<String>,
    parent_func_name: Option<String>,
    parent_block_name: Option<String>,

    glob_vars: HashMap<String, Box<Value>>,
    local_vars: HashMap<String, Box<Value>>,
    retval: Option<Box<Value>>
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
            blocks: HashMap::new(),

            ret_block: None,
            exit_block: None,
            parent_func_name: None,
            parent_block_name: None,

            glob_vars: HashMap::new(),
            local_vars: HashMap::new(),
            retval: None
        }
    }

    fn variable(&self, name: impl Into<String>) -> Option<Box<Value>> {
        let ref key = name.into();
        let global = self.llvm_module.get_global(key)
            .map(|glob| glob.to_super())
            .map(|ptr| mk_box(self.llvm_builder.build_load(ptr)));

        let local = self.local_vars.get(key)
            .map(|ptr| mk_box(self.llvm_builder.build_load(ptr)));

        // Local variable has higher priority
        local.or(global)
    }

    fn func(&mut self, name: impl Into<String>) -> Option<Box<Function>> {
        let ref varname = name.into();
        self.funcs.get_mut(varname)
            .map(|f| mk_box(f.as_ref()))
    }

    fn block(&mut self, funcname: impl Into<String>, blkname: impl Into<String>) -> Option<Box<BasicBlock>> {
        self.blocks.get(&funcname.into())
            .and_then(|blocks| blocks.get(&blkname.into()))
            .map(|block| mk_box(block.as_ref()))
    }

    fn parent_func(&mut self) -> Option<Box<Function>> {
        let parent = self.parent_func_name.clone();
        parent.and_then(|key| self.func(key))
    }

    fn parent_block(&mut self) -> Option<Box<BasicBlock>> {
        let parent_func = self.parent_func_name.clone()?;
        let parent_block = self.parent_block_name.clone()?;
        self.block(parent_func, parent_block)
    }

    fn mk_func(&mut self, name: impl AsRef<str>, ty: &llvm::Type) -> Box<Function> {
        let funcname = name.as_ref();
        let func = mk_box(self.llvm_module.add_function(funcname, ty));

        self.funcs.insert(String::from(funcname), func);
        self.blocks.insert(String::from(funcname), HashMap::new());

        self.func(funcname).unwrap()
    }

    fn mk_block(&mut self, funcname: impl Into<String>, blkname: impl Into<String>) -> Box<BasicBlock> {
        let funcname = funcname.into();
        let blkname = blkname.into();

        match self.func(funcname.clone()) {
            Some(func) => {
                let block = func.append(blkname.clone().as_str());

                let ref mut blocks = self.blocks.get_mut(&funcname)
                    .unwrap_or_else(|| panic!("mk_block: Unable to find blocks map for function '{}'.", funcname.clone()));

                if blocks.insert(blkname.clone(), mk_box(block)).is_some() {
                    panic!("Duplicated block '{}' in function '{}'.",
                           blkname.clone(), funcname.clone())
                }

                self.llvm_builder.position_at_end(block);
                mk_box(block)
            },
            None => {
                panic!("mk_block: Block '{}' creation failed. Unexistend function '{}'.",
                       funcname.clone(), blkname.clone())
            }
        }
    }

    fn mk_retval(&mut self, ty: &llvm::Type) -> Box<Value> {
        let alloca = self.llvm_builder.build_alloca(ty);
        self.retval = Some(mk_box(alloca));
        mk_box(alloca)
    }

    fn set_retval(&mut self, val: Box<Value>) -> Box<Value> {
        if let Some(ret) = &self.retval {
            self.llvm_builder.build_store(val.as_ref(), ret.as_ref());
            val
        } else {
            panic!("set_retval: Tried setting rerval without prior allocation.");
        }
    }

    fn drop_retval(&mut self) {
        self.retval = None;
    }

    fn drop_retval_llvm(&mut self) {
        if let Some(ret) = &self.retval {
            self.llvm_builder.build_free(ret.as_ref());
        } else {
            panic!("drop_retval: Tried dropping retval without prior allocation.");
        }
    }

    fn read_retval(&mut self) -> Box<Value> {
        if let Some(ret) = &self.retval {
            let value = self.llvm_builder.build_load(ret.as_ref());
            mk_box(value)
        } else {
            panic!("read_retval: Tried reading retval without prior allocation.");
        }
    }

    fn mk_local_var(&mut self, name: impl AsRef<str>, ty: &llvm::Type, def: Option<Box<Value>>) -> Box<Value> {
        let varname = name.as_ref();
        let alloca = self.llvm_builder.build_alloca(ty);

        self.local_vars.insert(varname.to_owned(), mk_box(alloca));
        if let Some(default) = def {
            self.llvm_builder.build_store(default.as_ref(), alloca);
        }

        self.variable(varname).unwrap()
    }

    fn drop_local_var(&mut self, varname: impl Into<String>) -> Option<Box<Value>> {
        let varname = varname.into();
        self.local_vars.remove(&varname)
    }

    fn drop_local_var_llvm(&mut self, varname: impl Into<String>) -> Option<Box<Value>> {
        let varname = varname.into();
        self.local_vars.get(&varname).map(|value| mk_box(value.as_ref()))
    }

    fn mk_global_var(&mut self, name: impl AsRef<str>, val: Box<Value>) -> Box<Value> {
        let varname = name.as_ref();

        let glob = self.llvm_module.add_global_variable(varname, val.as_ref());
        let glob = glob.to_super().to_super();

        self.glob_vars.insert(varname.to_owned(), mk_box(glob));
        self.variable(varname).unwrap()
    }

    fn gen_blk_ident(&self) -> String {
        // Get name of parent function
        let ref parenfunc = self.parent_func_name.clone()
            .unwrap_or_else(|| panic!("gen_blk_ident: No parent function."));

        // Count how many blocks generated for parent function
        let blocks_cnt = self.blocks.get(parenfunc)
            .unwrap_or_else(|| panic!("gen_blk_ident: No blocks for function '{}'.", parenfunc))
            .len();

        // Name of parent function and number of blocks generated will be new block identifier
        format!("{}_{}", parenfunc, blocks_cnt)
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

impl<'r> Codegen<'r, Box<BasicBlock>> for BlockExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> Box<BasicBlock> {
        // Name of parent function and number of blocks generated will be new block identifier
        let block_ident = ctx.gen_blk_ident();
        let parenfunc = ctx.parent_func_name.clone().unwrap();

        // Build block and move builder to the end
        let block = ctx.mk_block(parenfunc.clone(), block_ident.clone());
        ctx.llvm_builder.position_at_end(block.as_ref());

        // Update parent block name
        let prev_parent_block = ctx.parent_block_name.clone();
        ctx.parent_block_name = Some(block_ident.clone());

        for expr in &self.body {
            expr.gencode(ctx);
        }

        // If no return, then go back to exitblk
        if let Some(InBlockExprAST::Return(_)) = self.body.last() {
            // ...
        } else if let Some(blk) = ctx.exit_block.clone() {
            let exitblk = ctx.block(parenfunc.clone(), blk).unwrap();
            ctx.llvm_builder.build_br(exitblk.as_ref());
        }

        // Revert to previus block name
        ctx.parent_block_name = prev_parent_block;

        block
    }
}

impl<'r> Codegen<'r, ()> for InBlockExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> () {
        match self {
            InBlockExprAST::Assingment(ass) => {
                let ref name = ass.ident.name;
                let valuelike = ass.value.gencode(ctx);
                let ty = llvm::Type::get::<f64>(ctx.llvm_ctx);

                ctx.mk_local_var(name, ty, Some(valuelike));
            },

            // Ignored for now
            InBlockExprAST::Valuelike(_) => {},

            InBlockExprAST::If(iff) => {
                // Create exit block for if branches
                let funcname = ctx.parent_func_name.clone().unwrap();
                let startblk = ctx.parent_block().unwrap();

                let new_exitblk_name = ctx.gen_blk_ident();
                let exitblk = ctx.mk_block(funcname, new_exitblk_name.clone());

                // Generate first branch
                ctx.exit_block = Some(new_exitblk_name.clone());
                let b1 = iff.block_if.gencode(ctx);

                let b2 = if let Some(blk) = iff.block_else.clone() {
                    ctx.exit_block = Some(new_exitblk_name.clone());
                    blk.gencode(ctx)
                } else {
                    mk_box(exitblk.as_ref())
                };

                // Move back to start block
                ctx.llvm_builder.position_at_end(startblk.as_ref());
                // Calculate condition
                let cond = iff.cond.gencode(ctx);
                // Build branch
                ctx.llvm_builder.build_cond_br(cond.as_ref(),
                                               b1.as_ref(),
                                               Some(b2.as_ref()));
                // Move to exitblock
                ctx.llvm_builder.position_at_end(exitblk.as_ref());
            },

            InBlockExprAST::Return(ret) => {
                // Create return block for current branch.
                let parenname = ctx.parent_func_name.clone()
                    .unwrap_or_else(|| panic!("Return without parent function."));

                // Block termination
                if let Some(valuelike) = &ret.ret {
                    let ret_value = valuelike.gencode(ctx);
                    if ctx.retval.is_some() {
                        ctx.set_retval(ret_value);
                    } else {
                        panic!("retun in '{}', which is void function!", parenname);
                    }
                }

                let endblk_name = ctx.ret_block.clone().unwrap();
                let endblk = ctx.block(parenname.clone(), endblk_name).unwrap();
                ctx.llvm_builder.build_br(endblk.as_ref());
            },
        };
    }
}

impl<'r> Codegen<'r, Box<Function>> for FuncDefExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> Box<Function> {
        let ref prot = self.prototype;
        let ref body = self.body;

        let arity = prot.args.len();
        let ref funcname = prot.name.name;

        // Assume f64 args only.
        let arg_types = (0..arity)
            .map(|_| llvm::Type::get::<f64>(ctx.llvm_ctx))
            .collect::<Vec<&llvm::Type>>();

        // Either f64 or void
        let ret_type = if let Some(_) = &prot.ret_type {
            llvm::Type::get::<f64>(ctx.llvm_ctx)
        } else {
            llvm::Type::get::<()>(ctx.llvm_ctx)
        };

        let sig = llvm::FunctionType::new(ret_type, &arg_types[..]);
        let func = ctx.mk_func(funcname.clone(), sig.to_super());

        /*
         * PREAMB
         */
        let preamb_blk_name = format!("{}_preamb", funcname.clone());
        let preamb_blk = ctx.mk_block(funcname.clone(), preamb_blk_name.clone());

        // Allocate memory for arguments
        for i in 0..arity {
            let arg = &func[i];
            let ref name = prot.args.get(i).unwrap().name;
            ctx.mk_local_var(name,
                             llvm::Type::get::<f64>(ctx.llvm_ctx),
                             Some(mk_box(arg)));
        }
        // Allocate memory for return
        if !ret_type.is_void() {
            ctx.mk_retval(ret_type);
        }

        /*
         * DROP CODE
         */
        let drop_blk_name = format!("{}_drop", funcname.clone());
        let drop_blk = ctx.mk_block(funcname.clone(), drop_blk_name.clone());

        // Free memory for arguments
        ctx.llvm_builder.position_at_end(drop_blk.as_ref());
        for i in 0..arity {
            let ref name = prot.args.get(i).unwrap().name;
            if let Some(_) = ctx.local_vars.get(name) {
                ctx.drop_local_var_llvm(name);
            }
        }

        // Return value from retval
        if ctx.retval.is_some() {
            let ret_value = ctx.read_retval();
            ctx.drop_retval_llvm();
            ctx.llvm_builder.build_ret(ret_value.as_ref());
        } else {
            ctx.llvm_builder.build_ret_void();
        }

        /*
         * FUNCTION BODY
         */
        ctx.parent_func_name = Some(funcname.clone());
        ctx.ret_block = Some(drop_blk_name.clone());
        ctx.exit_block = Some(drop_blk_name.clone());
        let next_blk = body.gencode(ctx);
        ctx.ret_block = None;
        ctx.exit_block = None;
        ctx.parent_func_name = None;
        ctx.drop_retval();
        ctx.local_vars.clear();

        // Link PREAMB -> BODY
        ctx.llvm_builder.position_at_end(preamb_blk.as_ref());
        ctx.llvm_builder.build_br(next_blk.as_ref());

        ctx.llvm_builder.position_at_end(drop_blk.as_ref());

        func
    }
}

impl<'r> Codegen<'r, ()> for OutBlockExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> () {
        match self {
            OutBlockExprAST::Assingment(ass) => {
                let ref name = ass.ident.name;
                let valuelike = ass.value.gencode(ctx);

                ctx.mk_global_var(name, valuelike);
            },
            OutBlockExprAST::FuncDef(funcdef) => {
                funcdef.gencode(ctx);
            }
        };
    }
}

impl<'r> Codegen<'r, ()> for RootExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> () {
        self.items.iter().for_each(|expr| { expr.gencode(ctx); });
    }
}

fn module_disasm(ctx: &Context) -> String {
    let mktemp = || {
        let stdout = Command::new("mktemp")
            .output()
            .unwrap()
            .stdout;

        std::str::from_utf8(&stdout[..])
            .unwrap()
            .to_owned()
    };

    let temp1 = mktemp();
    let temp1 = temp1.trim();
    let temp2 = mktemp();
    let temp2 = temp2.trim();

    // Run library and dump bitcode in the temp directory
    /*
    ctx.llvm_module
        .verify()
        .unwrap();
     */
    
    ctx.llvm_module
        .write_bitcode(&temp1)
        .unwrap();

    // Run llvm-dis and dump disassembly in the temp directory
    Command::new("llvm-dis").arg(temp1).arg("-o").arg(temp2)
        .status()
        .unwrap();

    // Read disassembled file
    let cat_stdout = Command::new("cat").arg(temp2)
        .output()
        .unwrap()
        .stdout;

    // Convert vector of u8s into the string.
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
    ctx.mk_func("add",
                llvm::Type::get::<fn(f64, f64) -> f64>(ctx.llvm_ctx));
    let ref entry = ctx.mk_block("add", "entrypoint");

    ctx.llvm_builder.position_at_end(entry);
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

pub fn gencode(root: &RootExprAST) -> String {
    let ref mut llvm = llvm::Context::new();
    let ref mut ctx = Context::new(llvm);
    root.gencode(ctx);
    module_disasm(ctx)
}
