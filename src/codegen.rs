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
type RcSemiBox<'r, T> = Rc<CSemiBox<'r, T>>;

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

struct FuncMeta<'r> {
    builder:  RcSemiBox<'r, llvm::Builder>,
    func:     RcBox<Function>,
    blocks:   HashMap<String, RcBox<BasicBlock>>,
    argnames: Vec<String>,
    argptrs:  Vec<RcBox<Value>>,
    preamb:   Option<RcBox<BasicBlock>>,
    drop:     Option<RcBox<BasicBlock>>,
    retval:   Option<RcBox<Value>>,
    scope:    Scope
}

impl<'r> FuncMeta<'r> {
    fn new(ctx: RcBox<llvm::Context>,
           func: &Function,
           argnames: Vec<String>) -> Self
    {
        let arity = argnames.len();

        let ctx_ptr = &**ctx as *const llvm::Context;
        let builder = llvm::Builder::new(unsafe {
            ctx_ptr.as_ref().unwrap()
        });

        Self {
            builder: Rc::new(builder),
            func: mk_rcbox(func),
            blocks: HashMap::new(),
            argnames: argnames,
            argptrs: Vec::with_capacity(arity),
            preamb: None,
            drop: None,
            retval: None,
            scope: Scope::new()
        }
    }

    fn block(&self,
             blkname: impl AsRef<str>) -> Option<RcBox<BasicBlock>> {
        self.blocks.get(blkname.as_ref())
            .map(|r| r.clone())
    }

    fn mk_block(&mut self,
                blkname: impl Into<String>) -> RcBox<BasicBlock> {
        let blkname = blkname.into();
        let block = mk_rcbox(self.func.append(&blkname.clone()));

        let insert = self.blocks.insert(blkname.clone(), block.clone());
        if insert.is_some() {
            panic!("Duplicated block '{}' in function '{}'.", blkname.clone(), self.name());
        }

        self.block(blkname).unwrap()
    }

    fn mk_local_var(&mut self,
                    name: impl AsRef<str>,
                    ty: &llvm::Type,
                    def: Option<RcBox<Value>>) -> RcBox<Value>
    {
        let builder = self.builder.clone();

        let varname = name.as_ref();
        let alloca = builder.build_alloca(ty);

        if let Some(default) = def {
            builder.build_store(default.as_ref(), alloca);
        }

        self.scope.add_local(varname, mk_rcbox(alloca));
        self.scope.local(varname).unwrap()
    }

    fn mk_preamb(&mut self) -> RcBox<BasicBlock> {
        let preamb_ident = format!("{}_preamb", self.name());
        let preamb_block = self.mk_block(preamb_ident.clone());
        self.builder.position_at_end(&preamb_block);

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
                let argvalue = mk_rcbox(self.func[i].to_super());
                let alloca = self.mk_local_var(
                    argname,
                    ty,
                    Some(argvalue)
                );
                self.argptrs.push(alloca.clone());
            });

        // Allocate memory for return value
        self.retval = if ret_type.is_void() {
            None
        } else {
            let ret_alloca = self.builder.build_alloca(ret_type);
            Some(mk_rcbox(ret_alloca))
        };

        // Update structure field
        self.preamb = Some(preamb_block);
        // Return reference to newly created block
        self.block(preamb_ident.clone()).unwrap()
    }

    fn mk_drop(&mut self) -> RcBox<BasicBlock> {
        let drop_ident = format!("{}_drop", self.name());
        let drop_block = self.mk_block(drop_ident.clone());
        let ref builder = self.builder;
        builder.position_at_end(&drop_block);

        let signature = self.func.get_signature();
        let ret_type = signature.get_return();

        // Deallocate and return retval
        if ret_type.is_void() {
            builder.build_ret_void();
        } else {
            let ret_value_addr = self.retval.clone().unwrap();
            let ret_value = builder.build_load(&ret_value_addr);
            builder.build_ret(ret_value);
        }

        // Update structe field
        self.drop = Some(drop_block);
        // Return reference to newly created block
        self.block(drop_ident.clone()).unwrap()
    }

    fn set_retval(&mut self,
                  value: RcBox<Value>) {
        if let Some(ptr) = &self.retval {
            self.builder.build_store(&value, &ptr);
        } else {
            panic!("set_retval(): Tried setting retval in function '{}' without prioir allocation.",
                   self.name());
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

struct Scope {
    stack: Vec<HashMap<String, RcBox<Value>>>,
    touched: Vec<HashMap<String, bool>>
}

impl Scope {
    fn new() -> Self {
        Self {
            stack: vec![HashMap::new()],
            touched: vec![HashMap::new()]
        }
    }

    fn level(&self) -> usize {
        self.stack.len()
    }

    fn local(&self,
             name: impl Into<String>) -> Option<RcBox<Value>>
    {
        let ref key = name.into();
        let scope = self.stack.last().unwrap();
        let local_ptr = scope.get(key)?;
        Some(local_ptr.clone())
    }

    fn add_local(&mut self,
                 name: impl Into<String>,
                 value: RcBox<Value>)
    {
        let ref key = name.into();

        let scope = self.stack.last_mut().unwrap();
        let scope_touched = self.touched.last_mut().unwrap();

        // Check if value with this name was defined in current scope.
        let is_touched = *scope_touched.get(key).unwrap_or(&false);

        // Panic when detected redefinition of non-inherited value
        let insert = scope.insert(key.clone(), value);
        if insert.is_some() && is_touched {
            panic!("add_local(): Duplicate redefinition in the same scope: '{}'.", key);
        }

        // Prevent redefinition in the future
        scope_touched.insert(key.clone(), true);
    }

    fn drop_block(&mut self) {
        if self.stack.len() == 1 {
            panic!("drop_block(): Attempted to drop base scope.");
        }

        let n = self.stack.len();
        self.stack.remove(n-1);
        self.touched.remove(n-1);
    }

    fn new_block(&mut self) {
        let prev_scope = self.stack.last().unwrap();

        let new_scope = prev_scope.clone();
        let new_touched = new_scope.iter()
            .map(|(key, _)| (key.clone(), false))
            .collect::<HashMap<String, bool>>();

        self.stack.push(new_scope);
        self.touched.push(new_touched);
    }

    fn drop_all(&mut self) {
        self.stack = vec![HashMap::new()];
        self.touched = vec![HashMap::new()];
    }
}

struct Context<'r> {
    llvm_ctx: RcBox<llvm::Context>,
    llvm_module: RcSemiBox<'r, llvm::Module>,

    parent_func: Option<RcRef<FuncMeta<'r>>>,
    exit_block: Option<RcBox<BasicBlock>>,
    parent_block: Option<RcBox<BasicBlock>>,

    funcs: HashMap<String, RcRef<FuncMeta<'r>>>,
    glob_vars: HashMap<String, RcBox<Value>>
}

impl<'r> Context<'r> {
    fn new() -> Self {
        let llvm_ctx = &*llvm::Context::new();
        let llvm_ctx_ptr = llvm_ctx as *const llvm::Context;

        let module = llvm::Module::new(MOD_NAME, unsafe {
            llvm_ctx_ptr.as_ref().unwrap()
        });
        let llvm_ctx_rcbox = mk_rcbox(unsafe {
            llvm_ctx_ptr.as_ref().unwrap()
        });

        Self {
            llvm_ctx: llvm_ctx_rcbox,
            llvm_module: Rc::new(module),

            parent_func: None,
            exit_block: None,
            parent_block: None,

            funcs: HashMap::new(),
            glob_vars: HashMap::new(),
        }
    }

    fn builder(&self) -> RcSemiBox<'r, llvm::Builder> {
        if let Some(paren) = &self.parent_func {
            let parenfunc = paren.borrow();
            return parenfunc.builder.clone();
        }
        panic!("Requested builder without parent function set.");
    }

    fn deref_ptr(&self,
                 ptr: RcBox<Value>) -> RcBox<Value>
    {
        let builder = self.builder();
        let loaded = builder.build_load(&ptr);
        mk_rcbox(loaded)
    }

    fn local_ptr(&self,
                    name: impl Into<String>) -> Option<RcBox<Value>>
    {
        let parent = self.parent_func.clone()?;
        let loc = parent.borrow().scope.local(name.into())?;
        Some(loc)
    }

    fn global_ptr(&self,
                  name: impl Into<String>) -> Option<RcBox<Value>>
    {
        let ref key = name.into();
        let ptr = self.glob_vars.get(key)?;
        Some(mk_rcbox(ptr))
    }


    fn variable(&self,
                name: impl Into<String>) -> Option<RcBox<Value>>
    {
        let name = name.into();
        self.local_ptr(name.clone())
            .or(self.global_ptr(name.clone()))
            .map(|ptr| self.deref_ptr(ptr))
    }

    fn func(&mut self,
            name: impl Into<String>) -> Option<RcRef<FuncMeta<'r>>>
    {
        let ref varname = name.into();
        self.funcs.get_mut(varname)
            .map(|f| Rc::clone(f))
    }

    fn mk_func(&mut self,
               name: impl Into<String>,
               argnames: Vec<String>,
               ty: &llvm::FunctionType) -> RcRef<FuncMeta<'r>>
    {
        let funcname = name.into();
        let func = self.llvm_module.add_function(&funcname, ty);
        let funcmeta = FuncMeta::new(self.llvm_ctx.clone(),
                                     func,
                                     argnames);

        let funcmeta_rcref = mk_rcref(funcmeta);
        self.funcs.insert(funcname.clone(), funcmeta_rcref.clone());
        funcmeta_rcref
    }

    fn mk_global_var(&mut self,
                     name: impl AsRef<str>,
                     val: RcBox<Value>) -> RcBox<Value>
    {
        let varname = name.as_ref();

        let glob = self.llvm_module.add_global_variable(varname, val.as_ref());
        let glob = glob.to_super().to_super();

        self.glob_vars.insert(varname.to_string(), mk_rcbox(glob));
        self.global_ptr(varname).unwrap()
    }
}

trait Codegen<'r, T> {
    fn gencode(&self, ctx: &mut Context<'r>) -> T;
}

impl<'r> Codegen<'r, RcBox<Value>> for NumLiteralExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> RcBox<Value> {
        let val = self.value as i64;
        let num = val.compile(&ctx.llvm_ctx);
        mk_rcbox(num)
    }
}

impl<'r> Codegen<'r, RcBox<Value>> for ValuelikeExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> RcBox<Value> {
        match self {
            ValuelikeExprAST::NumericLiteral(num) => {
                num.gencode(ctx)
            },
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
                let builder = ctx.builder();
                let value = builder
                    .build_call(funcmeta.borrow().func.as_ref(), argslice);

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
        let ref builder = ctx.builder();

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
                panic!("No modulo for now"),
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

impl<'r> Codegen<'r, RcBox<Value>> for UnaryOpExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> RcBox<Value> {
        let value = self.expr.gencode(ctx);
        let ref builder = ctx.builder();

        let value_op = match self.op {
            UnaryOp::NOT =>
                builder.build_not(value.as_ref()),
            UnaryOp::MINUS =>
                builder.build_neg(value.as_ref())
        };

        mk_rcbox(value_op)
    }
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
        let block = parenfunc.borrow_mut().mk_block(block_ident.clone());

        // Change parent block to newly generated one.
        ctx.parent_block = Some(Rc::clone(&block));
        ctx.builder().position_at_end(&block);

        // Create new scope
        parenfunc.borrow_mut()
            .scope.new_block();

        // Backup exitblock value
        let exitblock_temp = ctx.exit_block.clone();
        ctx.exit_block = None;

        // Generate body of current block
        for expr in &self.body {
            if let InBlockExprAST::Return(_) = expr {
                if *expr != *self.body.last().unwrap() {
                    panic!("Detected return in the miedle of block.");
                }
            }
            expr.gencode(ctx);
        }

        // Drop scope
        parenfunc.borrow_mut()
            .scope.drop_block();

        // Bring back previous exitblock
        ctx.exit_block = exitblock_temp;

        // Make sure block is terminated.
        match (self.body.last(), ctx.exit_block.clone()) {
            // When generated block has return at the end.
            (Some(InBlockExprAST::Return(_)), _) => {},
            // When there's no terminator, branch unconditionaly to the exit block.
            (_, Some(exitblk)) => {
                ctx.builder().build_br(exitblk.as_ref());
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
        let parenfunc = ctx.parent_func.clone()
            .unwrap_or_else(|| panic!("InBlockExpr: no parent func set."));

        match self {
            // Ignored for now
            InBlockExprAST::Valuelike(_) => {},

            InBlockExprAST::Declaration(decl) => {
                let ref name = decl.ident.name;
                let valuelike = decl.value.gencode(ctx);

                // Assume i64 for now
                let ty = llvm::Type::get::<i64>(&ctx.llvm_ctx);
                parenfunc.borrow_mut().mk_local_var(name, ty, Some(valuelike));
            },

            InBlockExprAST::ReDeclaration(redecl) => {
                let ref name = redecl.ident.name;
                let valuelike = redecl.value.gencode(ctx);
                let local_ptr = ctx.local_ptr(name)
                    .unwrap_or_else(|| panic!("Redeclaration of '{}'. No such name in current scope.", name));

                let builder = ctx.builder();
                builder.build_store(&valuelike, &local_ptr);
            },

            InBlockExprAST::If(iff) => {
                let startblk = ctx.parent_block.clone().unwrap();

                let new_exitblk_name = parenfunc.borrow().gen_blk_ident();
                let new_exitblk = parenfunc
                    .borrow_mut()
                    .mk_block(new_exitblk_name.clone());

                // Generate first branch
                ctx.exit_block = Some(Rc::clone(&new_exitblk));
                let b1 = iff.block_if.gencode(ctx);

                // If there's a second branch, generate it.
                // Otherwise branch out to the exitblock.
                let b2 = if let Some(blk) = iff.block_else.clone() {
                    ctx.exit_block = Some(Rc::clone(&new_exitblk));
                    blk.gencode(ctx)
                } else {
                    Rc::clone(&new_exitblk)
                };

                // No use for exit_block in future processing
                ctx.exit_block = None;

                // Move back to start block
                ctx.builder().position_at_end(&startblk);
                // Generate code for condition
                let cond = iff.cond.gencode(ctx);
                // Build conditional branching
                ctx.builder().build_cond_br(&cond, &b1, Some(&b2));
                // Move builder to the end of exitblock
                ctx.builder().position_at_end(&new_exitblk);
                // Exitblock will be a parent for following ones.
                ctx.parent_block = Some(new_exitblk);

                // PROBLEM: When both branches return, the exitblock is made redundant.
                // Possible solution: When gencode() for BlockExprAST uses exitblock it sets it to None
                // this way, caller can determine whether block returned or used exitblock.
            },

            InBlockExprAST::Return(ret) => {
                let parenfunc_name = parenfunc.borrow().name();

                // Setting return value
                if let Some(valuelike) = &ret.ret {
                    let ret_value = valuelike.gencode(ctx);
                    parenfunc.borrow_mut().set_retval(ret_value);
                }

                // Block termination with unconditional jump to endblock.
                let endblk = parenfunc.borrow().drop.clone()
                    .unwrap_or_else(|| panic!("Return couldn't get drop block reference in func '{}'.", parenfunc_name));
                ctx.builder().build_br(&endblk);
            },
        };
    }
}

impl <'r> Codegen<'r, Box<llvm::Type>> for TypeExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> Box<llvm::Type> {
        let ty = match self {
            Self::Int =>
                llvm::Type::get::<i64>(&ctx.llvm_ctx),
            Self::Num =>
                llvm::Type::get::<f64>(&ctx.llvm_ctx),
            Self::Void =>
                llvm::Type::get::<()>(&ctx.llvm_ctx),
        };
        mk_box(ty)
    }
}

impl<'r> Codegen<'r, RcRef<FuncMeta<'r>>> for FuncDefExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> RcRef<FuncMeta<'r>> {
        let ref prot = self.prototype;
        let ref body = self.body;

        let arity = prot.args.len();
        let ref funcname = prot.name.name;

        // Extract arg names
        let arg_names = prot.args.iter()
            .map(|ident| ident.name.clone())
            .collect::<Vec<String>>();

        // Assume i64 args only
        let llvm_ctx = ctx.llvm_ctx.clone();
        let arg_types = (0..arity)
            .map(|_| llvm::Type::get::<i64>(&llvm_ctx))
            .collect::<Vec<&llvm::Type>>();
        let ret_type = prot.ret_type.gencode(ctx);
        let sig = llvm::FunctionType::new(&ret_type, &arg_types[..]);

        let funcmeta = ctx.mk_func(funcname.clone(), arg_names, sig);
        ctx.parent_func = Some(Rc::clone(&funcmeta));

        let preamb_blk = funcmeta.borrow_mut().mk_preamb();
        let drop_blk = funcmeta.borrow_mut().mk_drop();
        ctx.exit_block = Some(Rc::clone(&drop_blk));

        /*
         * FUNCTION BODY
         */
        let next_blk = body.gencode(ctx);
        ctx.exit_block = None;

        // Link PREAMB -> BODY
        ctx.builder().position_at_end(preamb_blk.as_ref());
        ctx.builder().build_br(next_blk.as_ref());

        // Move to DROP end
        ctx.builder().position_at_end(drop_blk.as_ref());

        ctx.parent_func = None;
        funcmeta
    }
}

impl<'r> Codegen<'r, ()> for OutBlockExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> () {
        match self {
            OutBlockExprAST::Declaration(decl) => {
                // PROBLEM: Some globals might depend on functions call/other computation.
                let ref name = decl.ident.name;
                let valuelike = decl.value.gencode(ctx);

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

fn run_cmd(cmd_str: &str) -> Result<String, String> {
    let chunks = cmd_str
        .split(' ')
        .collect::<Vec<&str>>();
    let name = chunks.first().ok_or(cmd_str)?;
    let args = &chunks[1..];

    let mut cmd = Command::new(name);
    args.iter().for_each(|arg| { cmd.arg(arg); });

    let result = cmd
        .output()
        .ok().ok_or(cmd_str)?;

    // Convert stdout bytes into string
    let stdout = result.stdout;
    std::str::from_utf8(&stdout[..])
        .ok()
        .map(|as_str| as_str.to_string())
        .ok_or(cmd_str.to_string())
}

fn mk_temp() -> String {
    let temp = run_cmd("mktemp")
        .unwrap_or_else(|_| panic!("Unable to execute 'mktemp'."));
    temp.trim().to_owned()
}

fn disasm<'r>(ctx: &Context<'r>) -> String {
    let temp1 = mk_temp();
    let temp2 = mk_temp();

    // Run library and dump bitcode in the temp directory
    ctx.llvm_module.verify().unwrap();
    ctx.llvm_module.write_bitcode(&temp1).unwrap();

    // Run llvm-dis and dump disassembly in the temp directory
    run_cmd(&format!("llvm-dis {} -o {}", temp1, temp2))
        .unwrap_or_else(|err| panic!("Error executing '{}'", err));

    // Read disassembled file
    let disasm = run_cmd(&format!("cat {}", temp2))
        .unwrap_or_else(|err| panic!("Error executing '{}'", err));

    // Cleanup temps
    run_cmd(&format!("rm {} {}", temp1, temp2))
        .unwrap_or_else(|err| panic!("Error executing '{}'", err));

    disasm
}

fn execute<'r>(ctx: &Context<'r>) -> i32 {
    let temp = mk_temp();

    // Run library and dump bitcode in the temp directory
    ctx.llvm_module.verify().unwrap();
    ctx.llvm_module.write_bitcode(&temp).unwrap();

    Command::new("lli")
        .arg(&temp)
        .status()
        .unwrap_or_else(|_| panic!("Error executing 'lli {}'.", &temp))
        .code()
        .unwrap()
}

pub fn gencode(root: &RootExprAST) -> (String, impl Fn() -> i32) {
    let mut ctx = Context::new();

    root.gencode(&mut ctx);

    (disasm(&ctx), move || execute(&ctx))
}
