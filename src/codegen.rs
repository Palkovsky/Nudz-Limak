extern crate llvm;

use super::ast::*;
use super::utils::*;
use super::token::*;
use super::types::*;

use std::process::Command;
use std::collections::HashMap;

// llvm-rs docs:
// https://tombebbington.github.io/llvm-rs/llvm/index.html
use llvm::{
    BasicBlock,
    Value,
    Function,
    Compile,
    Sub
};

const MOD_NAME: &'static str = "llvm-tut";

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
            builder.build_store(&default, alloca);
        }

        self.scope.add_local(varname, mk_rcbox(alloca));
        self.scope.local(varname).unwrap()
    }

    fn mk_local_arr(&mut self,
                    name: impl AsRef<str>,
                    ty: &llvm::Type,
                    size: RcBox<Value>) -> RcBox<Value>
    {

        let builder = self.builder.clone();

        let arrname = name.as_ref();
        let arr_ptr = builder.build_array_alloca(ty, &size);
        // alloca is pointer to first element on stack
        // we need to store ptr to ptr on stack
        self.mk_local_var(arrname, arr_ptr.get_type(), Some(mk_rcbox(arr_ptr)))
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
                  value: RcBox<Value>)
    {
        if let Some(ptr) = &self.retval {
            self.builder.build_store(&value, &ptr);
        } else {
            panic!("set_retval(): Tried setting retval in function '{}' without prioir allocation.",
                   self.name());
        }
    }

    fn name(&self) -> String {
        let name = self.func.get_name().unwrap_or("");
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

    /// Pointer to parent function of currently processed ASTNode.
    parent_func: Option<RcRef<FuncMeta<'r>>>,
    /// Instructs BlockExprAST to unconditionaly branch out to this block, after generating code for it.
    exit_block: Option<RcBox<BasicBlock>>,
    /// Pointer to block in which builder currently stays in.
    parent_block: Option<RcBox<BasicBlock>>,

    funcs: HashMap<String, RcRef<FuncMeta<'r>>>,
    globals: HashMap<String, RcBox<Value>>
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
            globals: HashMap::new(),
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
        let ptr = self.globals.get(key)?;
        Some(mk_rcbox(ptr))
    }

    fn variable_ptr(&self,
                    name: impl Into<String>) -> Option<RcBox<Value>>
    {
        let ref key = name.into();
        self.local_ptr(key).or(self.global_ptr(key))
    }

    fn variable(&self,
                name: impl Into<String>) -> Option<RcBox<Value>>
    {
        let name = name.into();
        self.variable_ptr(name).map(|ptr| self.deref_ptr(ptr))
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

        self.globals.insert(varname.to_string(), mk_rcbox(glob));
        self.global_ptr(varname).unwrap()
    }
}

trait Codegen<'r, T> {
    fn gencode(&self, ctx: &mut Context<'r>) -> T;
}

/// Used for generating stub values, i.e. functions signatues without impl.
trait Stubgen<'r, T> {
    fn genstub(&self, ctx: &mut Context<'r>) -> T;
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

        // Detect if we're dealing with pointer arithmetic.
        let ptr_artih = lhs.get_type().is_pointer() || rhs.get_type().is_pointer();

        let res = match self.op {
            BinOp::ADD => {
                if ptr_artih {
                    builder.build_unsafe_gep(lhs, &[rhs])
                } else {
                    builder.build_add(lhs, rhs)
                }
            },
            BinOp::SUB => {
                let neg_rhs = builder.build_neg(rhs);
                if ptr_artih {
                    builder.build_unsafe_gep(lhs, &[neg_rhs])
                } else {
                    builder.build_sub(lhs, rhs)
                }
            },
            BinOp::MUL =>
                builder.build_mul(lhs, rhs),
            BinOp::DIV =>
                builder.build_div(lhs, rhs),
            BinOp::MOD =>
                panic!("No modulo for now"),
            BinOp::AND =>
                builder.build_and(lhs, rhs),
            BinOp::OR =>
                builder.build_or(lhs, rhs),
            BinOp::BIT_AND | BinOp::BIT_OR =>
                panic!("No bitwise AND and OR."),
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
        let ref builder = ctx.builder();

        let value_op = match self.op {
            UnaryOp::NOT => {
                let ref value = self.expr.gencode(ctx);
                builder.build_not(value)
            },
            UnaryOp::MINUS => {
                let ref value = self.expr.gencode(ctx);
                builder.build_neg(&value)
            },
            UnaryOp::REF => {
                match &self.expr {
                    ValuelikeExprAST::Variable(ident) => {
                        let ptr = ctx.variable_ptr(&ident.name);
                        let ptr = ptr
                            .unwrap_or_else(|| panic!("&{0}: No such value with identifier '{0}'.",
                                                      &ident.name));

                        // Get reference to local stack variable
                        let ptr_ty: &llvm::Type = ptr.get_type();
                        assert!(ptr_ty.is_pointer());

                        // Cast it to pointer type
                        let ptr_ty: &llvm::PointerType =
                            llvm::Sub::<llvm::Type>::from_super(ptr_ty).unwrap();

                        builder.build_ptr_to_int(&ptr,
                                                 ptr_ty)
                    },
                    x => {
                        panic!("Unable to get pointer of '{:?}'.", x)
                    }
                }
            },
            UnaryOp::DEREF => {
                let ref value = self.expr.gencode(ctx);
                let value_ty = value.get_type();
                if !value_ty.is_pointer() {
                    panic!("Tried dereferencing {:?}, which is not a pointer type.", &value);
                }

                // Cast value to pointer type
                let value_ty: &llvm::PointerType =
                    llvm::Sub::<llvm::Type>::from_super(value_ty).unwrap();

                let ptr = builder.build_int_to_ptr(value, value_ty);
                let loaded = builder.build_load(&ptr);
                loaded
            },
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
            // Function call
            InBlockExprAST::Valuelike(ValuelikeExprAST::Call(call)) => {
                ValuelikeExprAST::Call(call.clone()).gencode(ctx);
            },

            // Ignored for now
            InBlockExprAST::Valuelike(_) => {}

            InBlockExprAST::Declaration(decl) => {
                let ref name = decl.ident.name;
                let ty = decl.ty.gencode(ctx).as_llvm(ctx.llvm_ctx.clone());

                match &decl.value {
                    DeclarationRHSExprAST::Valuelike(valuelike) => {
                        let valuelike = valuelike.gencode(ctx);
                        parenfunc.borrow_mut().mk_local_var(name, &ty, Some(valuelike))
                    },

                    DeclarationRHSExprAST::Array(
                        ArrayDeclarationExprAST::BySize(sizeval)
                    ) => {
                        let size = sizeval.gencode(ctx);

                        // Arrays should have pointer type specified.
                        assert!(ty.is_pointer());
                        let ty: &llvm::PointerType =
                            llvm::Sub::<llvm::Type>::from_super(&ty).unwrap();

                        parenfunc.borrow_mut().mk_local_arr(
                            name,
                            ty.get_element(),
                            size
                        )
                    },

                    DeclarationRHSExprAST::Array(
                        ArrayDeclarationExprAST::ByElements(_)
                    ) => {
                        panic!("By element decl not supported yet.")
                    }
                };
            },

            InBlockExprAST::ReDeclaration(redecl) => {
                let valuelike = redecl.value.gencode(ctx);
                match &redecl.target {
                    // In this case it should create new local varaible.
                    WritableExprAST::Variable(ident) => {
                        let ref name = ident.name;
                        let local_ptr = ctx.local_ptr(name)
                            .unwrap_or_else(||
                                            panic!("Redeclaration of '{}'. No such name in current scope.",
                                                   name));
                        let builder = ctx.builder();
                        builder.build_store(&valuelike, &local_ptr);
                    },
                    // 1. Evaluate body of deref and make sure it's a pointer.
                    // 2. Write evaluated value into target address.
                    WritableExprAST::PtrWrite(deref) => {
                        let target_ptr = deref.target.gencode(ctx);
                        let target_ty = target_ptr.get_type();

                        if !target_ty.is_pointer() {
                            panic!("PtrWrite: {:?} was expected to be a pointer.", deref.target);
                        }

                        let builder = ctx.builder();
                        builder.build_store(&valuelike, &target_ptr);
                    }
                }
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

            InBlockExprAST::While(wwhile) => {
                // 1. Create exit block.
                // 2. Create loop body block.
                // 3. Create block for condition checking.
                //    It should branch conditionally to either exit
                //    or loop body.
                // 4. Terminate loop block with branch to condition block.
                let startblk = ctx.parent_block.clone().unwrap();

                // EXIT BLOCK
                let new_exitblk_name = parenfunc.borrow().gen_blk_ident();
                let exitblk = parenfunc
                    .borrow_mut()
                    .mk_block(new_exitblk_name.clone());

                // CONDITION BLOCK
                let new_condblock_name = parenfunc.borrow().gen_blk_ident();
                let condblk = parenfunc
                    .borrow_mut()
                    .mk_block(new_condblock_name.clone());

                // LOOP BODY BLOCK
                ctx.exit_block = Some(condblk.clone());
                let loopblk = wwhile.body.gencode(ctx);
                ctx.exit_block = None;

                // Terminate condition block with conditional branch
                ctx.builder().position_at_end(&condblk);
                let cond = wwhile.cond.gencode(ctx);
                ctx.builder().build_cond_br(&cond, &loopblk, Some(&exitblk));

                // Terminate start with branch to conditional
                ctx.builder().position_at_end(&startblk);
                ctx.builder().build_br(&condblk);

                // Exitblock will be new parent block for next ASTNodes.
                ctx.builder().position_at_end(&exitblk);
                ctx.parent_block = Some(exitblk);
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

impl<'r> Codegen<'r, RcBox<dyn Typed>> for PrimitiveTypeExprAST {
    fn gencode(&self, _: &mut Context<'r>) -> RcBox<dyn Typed> {
        let ty: Box<dyn Typed> = match self {
            Self::Byte  => Box::new(Byte),
            Self::Short => Box::new(Short),
            Self::Int   => Box::new(Int),
            Self::Long  => Box::new(Long),
            Self::Void  => Box::new(Void),
        };
        Rc::new(ty)
    }
}

impl<'r> Codegen<'r, RcBox<dyn Typed>> for PtrTypeExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> RcBox<dyn Typed> {
        let ty = self.pointee.gencode(ctx);
        let ptr_ty = Ptr::new(ty);
        Rc::new(Box::new(ptr_ty))
    }
}

impl<'r> Codegen<'r, RcBox<dyn Typed>> for TypeExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> RcBox<dyn Typed> {
        match self {
            TypeExprAST::Primitive(primitive) =>
                primitive.gencode(ctx),
            TypeExprAST::Pointer(ptr) =>
                ptr.gencode(ctx)
        }
    }
}

impl<'r> Codegen<'r, RcRef<FuncMeta<'r>>> for FuncDefExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> RcRef<FuncMeta<'r>> {
        let ref prot = self.prototype;
        let ref body = self.body;

        let ref funcname = prot.name.name;

        let funcmeta = ctx.func(funcname.clone())
            .unwrap_or_else(|| panic!("FuncDefExprAST: No FuncMeta for '{}'.", funcname));
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
            // Globals generation is handled by Stubgen
            OutBlockExprAST::Declaration(_) => {},
            OutBlockExprAST::FuncDef(funcdef) => {
                funcdef.gencode(ctx);
            }
        };
    }
}

impl<'r> Codegen<'r, ()> for RootExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> () {
        // Detect all function signatures/globals.
        self.items.iter().for_each(|expr| expr.genstub(ctx));
        // Perform actuall code generation.
        self.items.iter().for_each(|expr| { expr.gencode(ctx); });
    }
}

/// Stubgen for FuncDefExprAST will detect function signature and create entry in funcs HashMap. It WILL NOT generate function body.
impl<'r> Stubgen<'r, RcRef<FuncMeta<'r>>> for FuncDefExprAST {
    fn genstub(&self, ctx: &mut Context<'r>) -> RcRef<FuncMeta<'r>> {
        let ref prot = self.prototype;
        let ref funcname = prot.name.name;

        // Extract arg names
        let arg_names = prot.args.iter()
            .map(|(ident, _)| ident.name.clone())
            .collect::<Vec<String>>();

        let arg_types = prot.args.iter()
            .map(|(_, ty)| {
                let typed = ty.gencode(ctx);
                typed.as_llvm(ctx.llvm_ctx.clone())
            })
            .collect::<Vec<RcBox<llvm::Type>>>();

        let ret_type = prot.ret_type
            .gencode(ctx)
            .as_llvm(ctx.llvm_ctx.clone());

        let sig = llvm::FunctionType::new(&ret_type, &mk_slice(&arg_types)[..]);
        ctx.mk_func(funcname.clone(), arg_names, sig)
    }
}

/// Stubgen for OutBlockExprAST will detect globals initialized with literals + Stubgen for function definitions.
impl<'r> Stubgen<'r, ()> for OutBlockExprAST {
    fn genstub(&self, ctx: &mut Context<'r>) -> () {
        match self {
            OutBlockExprAST::Declaration(decl) => {
                let ref name = decl.ident.name;
                let ref value = decl.value;

                let is_lit = match value {
                    DeclarationRHSExprAST::Valuelike(value) =>
                        value.is_literal(),
                    _ => false
                };

                if !is_lit {
                    panic!("Value of global '{}' must be literal.", name);
                }

                if let DeclarationRHSExprAST::Valuelike(valuelike) = value {
                    let ref valuelike = valuelike.gencode(ctx);
                    ctx.mk_global_var(name, mk_rcbox(valuelike));
                }
            },
            OutBlockExprAST::FuncDef(funcdef) => {
                funcdef.genstub(ctx);
            }
        }
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
