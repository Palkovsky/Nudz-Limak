extern crate llvm;

use super::ast::*;
use super::token::*;

use std::process::Command;
use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

// llvm-rs docs:
// https://tombebbington.github.io/llvm-rs/llvm/index.html
use llvm::{
    CSemiBox,
    BasicBlock,
    Value,
    Function,
    Compile,
    Sub
};

const MOD_NAME: &'static str = "llvm-tut";

type RcBox<T> = Rc<Box<T>>;
type RcRef<T> = Rc<RefCell<T>>;

/*
 * Nasty stuff
 */
fn mk_box<T>(value: &T) -> Box<T> {
    let ptr = (value as *const T) as usize;
    unsafe { Box::from_raw(ptr as *mut T) }
}

fn mk_rcbox<T>(value: &T) -> RcBox<T> {
    let boxed = mk_box(value);
    Rc::new(boxed)
}

fn mk_rcref<T>(value: T) -> RcRef<T> {
    Rc::new(RefCell::new(value))
}

fn mk_slice<T>(vec: &Vec<RcBox<T>>) -> Vec<&T> {
    unsafe {
        let ptrs = vec
            .iter()
            .map(|item| item.as_ref().as_ref() as *const T)
            .map(|ptr| ptr.as_ref().unwrap())
            .collect::<Vec<&T>>();
        ptrs
    }
}

struct FuncMeta {
    func:     RcBox<Function>,
    blocks:   HashMap<String, RcBox<BasicBlock>>,
    argnames: Vec<String>,
    argptrs:  Vec<RcBox<Value>>,
    preamb:   Option<RcBox<BasicBlock>>,
    drop:     Option<RcBox<BasicBlock>>,
    retval:   Option<RcBox<Value>>
}

impl FuncMeta {
    fn new<'r>(ctx: &mut Context<'r>, name: impl AsRef<str>, argnames: Vec<String>, ty: &llvm::FunctionType) -> Self {
        let funcname = name.as_ref();
        let func = ctx.llvm_module.add_function(funcname, ty);
        let arity = argnames.len();

        Self {
            func: mk_rcbox(func),
            blocks: HashMap::new(),
            argnames: argnames,
            argptrs: Vec::with_capacity(arity),
            preamb: None,
            drop: None,
            retval: None
        }
    }

    fn block(&self, blkname: impl AsRef<str>) -> Option<RcBox<BasicBlock>> {
        self.blocks.get(blkname.as_ref())
            .map(|r| r.clone())
    }

    fn mk_named_block(&mut self, blkname: impl Into<String>) -> RcBox<BasicBlock> {
        let blkname = blkname.into();
        let block = mk_rcbox(self.func.append(blkname.clone().as_str()));

        let insert = self.blocks.insert(blkname.clone(), block.clone());
        if insert.is_some() {
            panic!("Duplicated block '{}' in function '{}'.", blkname.clone(), self.name());
        }

        self.block(blkname).unwrap()
    }

    fn mk_preamb<'r>(&mut self, ctx: &mut Context<'r>) -> RcBox<BasicBlock> {
        let preamb_ident = format!("{}_preamb", self.name());
        let preamb_block = self.mk_named_block(preamb_ident.clone());
        ctx.llvm_builder.position_at_end(&preamb_block);

        let func = self.func.clone();
        let signature = func.get_signature();
        let arity = signature.num_params();
        let arg_types = signature.get_params();
        let ret_type = signature.get_return();

        // Allocate memory for arguments
        (0..arity)
            .zip(arg_types.into_iter())
            .zip(self.argnames.clone().iter())
            .for_each(|((i, ty), argname): ((usize, &llvm::Type), &String)| {
                let argvalue = &self.func[i];
                let alloca = ctx.mk_local_var(argname,
                                             ty,
                                             Some(mk_rcbox(argvalue)));
                self.argptrs.push(alloca.clone());
            });

        // Allocate memory for return value
        self.retval = if ret_type.is_void() {
            None
        } else {
            let ret_alloca = ctx.llvm_builder.build_alloca(ret_type);
            Some(mk_rcbox(ret_alloca))
        };

        // Update structure field
        self.preamb = Some(preamb_block);
        // Return reference to newly created block
        self.block(preamb_ident.clone()).unwrap()
    }

    fn mk_drop<'r>(&mut self, ctx: &mut Context<'r>) -> RcBox<BasicBlock> {
        let drop_ident = format!("{}_drop", self.name());
        let drop_block = self.mk_named_block(drop_ident.clone());
        ctx.llvm_builder.position_at_end(&drop_block);

        let signature = self.func.get_signature();
        let ret_type = signature.get_return();

        // Deallocate and return retval
        if ret_type.is_void() {
            ctx.llvm_builder.build_ret_void();
        } else {
            let ret_value_addr = self.retval.clone().unwrap();
            let ret_value = ctx.llvm_builder.build_load(&ret_value_addr);
            ctx.llvm_builder.build_ret(ret_value);
        }

        // Update structe field
        self.drop = Some(drop_block);
        // Return reference to newly created block
        self.block(drop_ident.clone()).unwrap()
    }

    fn set_retval<'r>(&mut self, ctx: &mut Context<'r>, value: RcBox<Value>) {
        if let Some(ptr) = &self.retval {
            ctx.llvm_builder.build_store(&value, &ptr);
        } else {
            panic!("set_retval(): Tried setting retval in function '{}' without prioir allocation.", self.name());
        }
    }

    fn name(&self) -> String {
        let name = self.func.get_name();
        String::from(name)
    }

    fn gen_blk_ident(&self) -> String {
        // Count how many blocks generated for parent function
        let mut blocks_cnt = self.blocks.len();

        // Don't take preambule and drop into account when calculating identifier.
        if self.preamb.is_some() { blocks_cnt -= 1; }
        if self.drop.is_some() { blocks_cnt -= 1; }

        // Name of parent function and number of blocks generated will be new block identifier
        format!("{}_{}", self.name(), blocks_cnt)
    }
}

struct Context<'r> {
    llvm_ctx: &'r llvm::Context,
    llvm_module: CSemiBox<'r, llvm::Module>,
    llvm_builder: CSemiBox<'r, llvm::Builder>,

    funcs: HashMap<String, RcRef<FuncMeta>>,
    parent_func: Option<RcRef<FuncMeta>>,

    exit_block: Option<RcBox<BasicBlock>>,
    parent_block: Option<RcBox<BasicBlock>>,

    glob_vars: HashMap<String, RcBox<Value>>,
    local_vars: HashMap<String, RcBox<Value>>,
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
            parent_func: None,

            exit_block: None,
            parent_block: None,

            glob_vars: HashMap::new(),
            local_vars: HashMap::new(),
        }
    }

    fn variable(&self, name: impl Into<String>) -> Option<RcBox<Value>> {
        let ref key = name.into();

        let global = self.llvm_module.get_global(key)
            .map(|glob| glob.to_super())
            .map(|ptr| mk_rcbox(self.llvm_builder.build_load(ptr)));

        let local = self.local_vars.get(key)
            .map(|ptr| mk_rcbox(self.llvm_builder.build_load(ptr)));

        // Local variable has higher priority
        local.or(global)
    }

    fn func(&mut self, name: impl Into<String>) -> Option<RcRef<FuncMeta>> {
        let ref varname = name.into();
        self.funcs.get_mut(varname)
            .map(|f| Rc::clone(f))
    }

    fn mk_func(&mut self, name: impl Into<String>, argnames: Vec<String>, ty: &llvm::FunctionType) -> RcRef<FuncMeta> {
        let funcname = name.into();
        let funcmeta = FuncMeta::new(self, funcname.clone(), argnames, ty);

        let funcmeta_rcref = mk_rcref(funcmeta);
        self.funcs.insert(funcname.clone(), funcmeta_rcref.clone());
        funcmeta_rcref
    }

    fn mk_local_var(&mut self, name: impl AsRef<str>, ty: &llvm::Type, def: Option<RcBox<Value>>) -> RcBox<Value> {
        let varname = name.as_ref();
        let alloca = self.llvm_builder.build_alloca(ty);

        self.local_vars.insert(varname.to_owned(), mk_rcbox(alloca));
        if let Some(default) = def {
            self.llvm_builder.build_store(default.as_ref(), alloca);
        }

        self.variable(varname).unwrap()
    }

    fn mk_global_var(&mut self, name: impl AsRef<str>, val: RcBox<Value>) -> RcBox<Value> {
        let varname = name.as_ref();

        let glob = self.llvm_module.add_global_variable(varname, val.as_ref());
        let glob = glob.to_super().to_super();

        self.glob_vars.insert(varname.to_owned(), mk_rcbox(glob));
        self.variable(varname).unwrap()
    }
}

trait Codegen<'r, T> {
    fn gencode(&self, ctx: &mut Context<'r>) -> T;
}

impl<'r> Codegen<'r, RcBox<Value>> for NumLiteralExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> RcBox<Value> {
        let num = self.value.compile(ctx.llvm_ctx);
        mk_rcbox(num)
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

impl<'r> Codegen<'r, RcBox<Value>> for ValuelikeExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> RcBox<Value> {
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
                let funcmeta = {
                    let f = ctx.func(name);
                    if f.is_none() {
                        panic!(" Function '{}' is undefined.", name);
                    }
                    f.unwrap().clone()
                };

                let ref args = call.args
                    .iter()
                    .map(|arg| arg.gencode(ctx))
                    .collect::<Vec<RcBox<Value>>>();

                let argslice = &mk_slice(args)[..];
                let value = ctx.llvm_builder.build_call(funcmeta.borrow().func.as_ref(), argslice);
                mk_rcbox(&value)
            },
            ValuelikeExprAST::BinExpression(binexpr) =>
                binexpr.gencode(ctx),
            ValuelikeExprAST::UnaryExpression(unary) =>
                unary.gencode(ctx)
        }
    }
}

impl<'r> Codegen<'r, RcBox<Value>> for BinOpExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> RcBox<Value> {
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

        mk_rcbox(res)
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

impl<'r> Codegen<'r, RcBox<Value>> for UnaryOpExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> RcBox<Value> {
        let value = self.expr.gencode(ctx);
        let ref builder = ctx.llvm_builder;

        let value_op = match self.op {
            UnaryOp::NOT =>
                builder.build_not(value.as_ref()),
            UnaryOp::MINUS =>
                builder.build_neg(value.as_ref())
        };

        mk_rcbox(value_op)
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

impl<'r> Codegen<'r, RcBox<BasicBlock>> for BlockExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> RcBox<BasicBlock> {
        // Get info about parent function.
        let parenfunc = ctx.parent_func.clone().unwrap();
        let parenfunc_name = parenfunc.borrow().name();

        // Save current parenblock to bring it back after exiting current block.
        let prev_parenblock = ctx.parent_block.clone();

        // Build block and move builder to the end.
        let block_ident = parenfunc.borrow().gen_blk_ident();
        let block = parenfunc.borrow_mut().mk_named_block(block_ident.clone());
        // Change parent block to newly generated one.
        ctx.parent_block = Some(Rc::clone(&block));
        ctx.llvm_builder.position_at_end(&block);

        // Generate body of current block
        for expr in &self.body {
            expr.gencode(ctx);
        }

        // Make sure block is terminated.
        match (self.body.last(), ctx.exit_block.clone()) {
            // When generated block has return at the end.
            (Some(InBlockExprAST::Return(_)), _) => {},
            // When there's no terminator, branch unconditionaly to the exit block.
            (_, Some(exitblk)) => {
                ctx.llvm_builder.build_br(exitblk.as_ref());
            },
            // When not terminated with return and no exit block specified, we got a dangling block.
            _ => {
                panic!("Unable to perform block termination for '{}' in '{}'.", block_ident, parenfunc_name)
            }
        }

        // Revert to previus parent block
        ctx.parent_block = prev_parenblock;
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
                let parenfunc = ctx.parent_func.clone()
                    .unwrap_or_else(|| panic!("If without parent function."));

                let startblk = ctx.parent_block.clone().unwrap();

                let new_exitblk_name = parenfunc.borrow().gen_blk_ident();
                let new_exitblk = parenfunc.borrow_mut().mk_named_block(new_exitblk_name.clone());

                // Generate first branch
                ctx.exit_block = Some(Rc::clone(&new_exitblk));
                let b1 = iff.block_if.gencode(ctx);

                // If there's a second branch specified, generate it.
                // Otherwise branch out to the exitblock.
                let b2 = if let Some(blk) = iff.block_else.clone() {
                    ctx.exit_block = Some(Rc::clone(&new_exitblk));
                    blk.gencode(ctx)
                } else {
                    Rc::clone(&new_exitblk)
                };

                // Move back to start block
                ctx.llvm_builder.position_at_end(&startblk);
                // Generate code for condition
                let cond = iff.cond.gencode(ctx);
                // Build conditional branching
                ctx.llvm_builder.build_cond_br(&cond, &b1, Some(&b2));
                // Move builder to the end of exitblock
                ctx.llvm_builder.position_at_end(&new_exitblk);
            },

            InBlockExprAST::Return(ret) => {
                // Create return block for current branch.
                let parenfunc = ctx.parent_func.clone()
                    .unwrap_or_else(|| panic!("Return without parent function."));
                let parenfunc_name = parenfunc.borrow().name();

                // Setting return value
                if let Some(valuelike) = &ret.ret {
                    let ret_value = valuelike.gencode(ctx);
                    parenfunc.borrow_mut().set_retval(ctx, ret_value);
                }

                // Block termination with unconditional jump to endblock.
                let endblk = parenfunc.borrow().drop.clone()
                    .unwrap_or_else(|| panic!("Return couldn't get drop block reference in func '{}'.", parenfunc_name));
                ctx.llvm_builder.build_br(&endblk);
            },
        };
    }
}

impl<'r> Codegen<'r, RcRef<FuncMeta>> for FuncDefExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> RcRef<FuncMeta> {
        let ref prot = self.prototype;
        let ref body = self.body;

        let arity = prot.args.len();
        let ref funcname = prot.name.name;

        // Extract arg names
        let arg_names = prot.args.iter()
            .map(|ident| ident.name.clone())
            .collect::<Vec<String>>();

        // Assume f64 args only
        let arg_types = (0..arity)
            .map(|_| llvm::Type::get::<f64>(ctx.llvm_ctx))
            .collect::<Vec<&llvm::Type>>();

        // Either f64 or void as return
        let ret_type = if let Some(_) = &prot.ret_type {
            llvm::Type::get::<f64>(ctx.llvm_ctx)
        } else {
            llvm::Type::get::<()>(ctx.llvm_ctx)
        };

        let sig = llvm::FunctionType::new(ret_type, &arg_types[..]);

        let funcmeta = ctx.mk_func(funcname.clone(), arg_names, sig);
        let preamb_blk = funcmeta.borrow_mut().mk_preamb(ctx);
        let drop_blk = funcmeta.borrow_mut().mk_drop(ctx);

        /*
         * FUNCTION BODY
         */
        ctx.parent_func = Some(Rc::clone(&funcmeta));
        ctx.exit_block = Some(Rc::clone(&drop_blk));
        let next_blk = body.gencode(ctx);
        ctx.exit_block = None;
        ctx.parent_func = None;
        ctx.local_vars.clear();

        // Link PREAMB -> BODY
        ctx.llvm_builder.position_at_end(preamb_blk.as_ref());
        ctx.llvm_builder.build_br(next_blk.as_ref());

        // Move to DROP end
        ctx.llvm_builder.position_at_end(drop_blk.as_ref());

        funcmeta
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
    let runcmd = |cmd: &str| -> Option<String> {
        let chunks = cmd.split(' ').collect::<Vec<&str>>();
        let name = chunks.first()?;
        let args = &chunks[1..];

        let mut cmd = Command::new(name);
        args.iter().for_each(|arg| { cmd.arg(arg); });

        let result = cmd.output().ok()?;
        let stdout = result.stdout;

        std::str::from_utf8(&stdout[..]).ok()
            .map(|as_str| String::from(as_str))
    };

    let mktemp = || {
        let temp = runcmd("mktemp").unwrap();
        temp.trim().to_owned()
    };

    let temp1 = mktemp();
    let temp2 = mktemp();

    // Run library and dump bitcode in the temp directory
    ctx.llvm_module.verify().unwrap();
    ctx.llvm_module.write_bitcode(&temp1).unwrap();

    // Run llvm-dis and dump disassembly in the temp directory
    runcmd(&format!("llvm-dis {} -o {}", temp1, temp2));

    // Read disassembled file
    let disasm = runcmd(&format!("cat {}", temp2)).unwrap();

    // Cleanup temps
    runcmd(&format!("rm {} {}", temp1, temp2));

    disasm
}

fn valuelike_disasm(binexpr: &ValuelikeExprAST) -> String {
    let ref mut llvm = llvm::Context::new();
    let ref mut ctx = Context::new(llvm);

    // Add test function
    let f64ty = llvm::Type::get::<f64>(ctx.llvm_ctx);
    let funcmeta = ctx.mk_func("add",
                               vec!["a".to_string(), "b".to_string()],
                                   llvm::FunctionType::new(f64ty, &vec![f64ty, f64ty][..]));

    let ref entry = funcmeta.borrow_mut().mk_named_block("entrypoint");

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
