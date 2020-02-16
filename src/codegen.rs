extern crate llvm;

use super::ast::*;
use super::utils::*;
use super::token::*;
use super::types::*;
use super::allocator::*;

use std::process::Command;
use std::collections::HashMap;

// llvm-rs docs:
// https://tombebbington.github.io/llvm-rs/llvm/index.html
use llvm::{
    BasicBlock,
    Function,
    Compile,
    Sub
};

const MOD_NAME: &'static str = "llvm-tut";

struct FuncMeta<'r> {
    ctx:       RcBox<llvm::Context>,
    builder:   RcSemiBox<'r, llvm::Builder>,
    allocator: Rc<StackAllocator<'r>>,
    func:      RcBox<Function>,
    blocks:    HashMap<String, RcBox<BasicBlock>>,

    argnames:  Vec<(LangType, String)>,
    argptrs:   Vec<Rc<TypedValue<'r>>>,
    ret_ty:    LangType,

    preamb:    Option<RcBox<BasicBlock>>,
    drop:      Option<RcBox<BasicBlock>>,
    retval:    Option<Rc<TypedValue<'r>>>,
    scope:     Scope<'r>
}

impl<'r> FuncMeta<'r> {
    fn new(ctx: RcBox<llvm::Context>,
           func: &Function,
           argnames: Vec<(LangType, String)>,
           ret_ty: LangType
    ) -> Self
    {
        let arity = argnames.len();
        let ctx_ptr = &**ctx as *const llvm::Context;
        let builder = Rc::new(llvm::Builder::new(unsafe {
            ctx_ptr.as_ref().unwrap()
        }));
        let allocator = Rc::new(StackAllocator::new(ctx.clone(), builder.clone()));

        Self {
            ctx: ctx.clone(),
            builder: builder,
            allocator: allocator,
            func: mk_rcbox(func),
            blocks: HashMap::new(),

            argnames: argnames,
            argptrs: Vec::with_capacity(arity),
            ret_ty: ret_ty,

            preamb: None,
            drop: None,
            retval: None,
            scope: Scope::new()
        }
    }

    fn block(&self,
             blkname: impl AsRef<str>) -> Option<RcBox<BasicBlock>>
    {
        self.blocks.get(blkname.as_ref()).map(|r| r.clone())
    }

    fn allocator(&self) -> Rc<StackAllocator<'r>> {
        self.allocator.clone()
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
                    typed: Rc<TypedValue<'r>>) -> Rc<TypedValue<'r>>
    {
        let varname = name.as_ref();
        let allocated = self.allocator.val(typed.ty(), Some(typed.clone()))
            .unwrap_or_else(|err| panic!(err));

        self.scope.add_local(varname, Rc::new(allocated));
        self.scope.local(varname).unwrap()
    }

    fn mk_local_arr_ofsize(&mut self,
                           name: impl AsRef<str>,
                           ty: LangType,
                           size: Rc<TypedValue<'r>>) -> Rc<TypedValue<'r>>
    {
        let varname = name.as_ref();
        let allocated = self.allocator.arr_ofsize(ty, size.clone())
            .unwrap_or_else(|err| panic!(err));

        self.mk_local_var(varname, Rc::new(allocated))
    }

    fn mk_local_arr_byelems(&mut self,
                            name: impl AsRef<str>,
                            ty: LangType,
                            elems: Vec<Rc<TypedValue<'r>>>) -> Rc<TypedValue<'r>>
    {
        let varname = name.as_ref();

        // Try to cast elems to ty
        let mut casted = Vec::with_capacity(elems.len());
        for elem in elems {
            let cast = elem.cast(ty.clone())
                .unwrap_or_else(|| panic!("Unable to cast array element into '{:?}'.", ty));
            casted.push(cast);
        }

        let allocated = self.allocator.arr_byelems(ty, casted)
            .unwrap_or_else(|err| panic!(err));

        self.mk_local_var(varname, Rc::new(allocated))
    }

    fn mk_preamb(this: RcRef<Self>) -> RcBox<BasicBlock> {
        let mut this = this.borrow_mut();

        let preamb_ident = format!("{}_preamb", this.name());
        let preamb_block = this.mk_block(preamb_ident.clone());
        this.builder.position_at_end(&preamb_block);

        let arity = this.argnames.len();

        // Allocate memory for arguments
        for (i, (ty, argname)) in (0..arity).zip(this.argnames.clone().into_iter()) {
            let argvalue = mk_rcbox(this.func[i].to_super());

            let typed = TypedValue::new(
                this.ctx.clone(), this.builder.clone(),
                argvalue, ty
            ).unwrap();

            let alloca = this.mk_local_var(argname, Rc::new(typed));
            this.argptrs.push(alloca);
        }

        // Allocate memory for return value
        let ctx = this.ctx.clone();
        let builder = this.builder.clone();
        let ret_ty = this.ret_ty.clone();
        let retval = if ret_ty.is_void() {
            None
        } else {
            let ret_alloca = mk_rcbox(this.builder.build_alloca(
                &ret_ty.as_llvm(ctx.clone())
            ));
            let typed = TypedValue::new(
                ctx, builder,
                ret_alloca, LangType::new_ptr(ret_ty)
            ).unwrap();
            Some(Rc::new(typed))
        };

        this.retval = retval;

        // Update structure field
        this.preamb = Some(preamb_block);
        // Return reference to newly created block
        this.block(preamb_ident.clone()).unwrap()
    }


    fn mk_drop(&mut self) -> RcBox<BasicBlock> {
        let drop_ident = format!("{}_drop", self.name());
        let drop_block = self.mk_block(drop_ident.clone());
        let ref builder = self.builder;
        builder.position_at_end(&drop_block);

        let signature = self.func.get_signature();
        let ret_type = signature.get_return();

        // Return retval
        if ret_type.is_void() {
            builder.build_ret_void();
        } else {
            let ret_value = self.retval.clone().unwrap();
            let ret_value = builder.build_load(&ret_value.llvm());
            builder.build_ret(ret_value);
        }

        // Update structe field
        self.drop = Some(drop_block);
        // Return reference to newly created block
        self.block(drop_ident.clone()).unwrap()
    }

    fn set_retval(&mut self,
                  value: Rc<TypedValue>)
    {
        if value.ty() != self.ret_ty {
            panic!(
                "set_retval(): Tried setting invalid type. Got: '{:?}, expected '{:?}'.",
                value.ty(), self.ret_ty
            );
        }

        if let Some(ptr) = &self.retval {
            self.builder.build_store(&value.llvm(), &ptr.llvm());
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

struct Scope<'r> {
    stack: Vec<HashMap<String, Rc<TypedValue<'r>>>>,
    touched: Vec<HashMap<String, bool>>
}

impl<'r> Scope<'r> {
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
             name: impl Into<String>) -> Option<Rc<TypedValue<'r>>>
    {
        let ref key = name.into();
        let scope = self.stack.last().unwrap();
        let local_ptr = scope.get(key)?;
        Some(local_ptr.clone())
    }

    fn add_local(&mut self,
                 name: impl Into<String>,
                 value: Rc<TypedValue<'r>>)
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
    glob_builder: RcSemiBox<'r, llvm::Builder>,

    /// Pointer to parent function of currently processed ASTNode.
    parent_func: Option<RcRef<FuncMeta<'r>>>,
    /// Instructs BlockExprAST to unconditionaly branch out to this block, after generating code for it.
    exit_block: Option<RcBox<BasicBlock>>,
    /// Pointer to block in which builder currently stays in.
    parent_block: Option<RcBox<BasicBlock>>,

    funcs: HashMap<String, RcRef<FuncMeta<'r>>>,
    globals: HashMap<String, Rc<TypedValue<'r>>>
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
        let glob_builder = llvm::Builder::new(unsafe {
            llvm_ctx_ptr.as_ref().unwrap()
        });

        Self {
            llvm_ctx: llvm_ctx_rcbox,
            llvm_module: Rc::new(module),
            glob_builder: Rc::new(glob_builder),

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
            parenfunc.builder.clone()
        } else {
            self.glob_builder.clone()
        }
    }

    fn allocator(&self) -> Rc<StackAllocator<'r>> {
        if let Some(paren) = &self.parent_func {
            let parenfunc = paren.borrow();
            return parenfunc.allocator.clone();
        }
        panic!("Requested allocator without paren function set.");
    }

    fn deref_ptr(&self,
                 ptr: Rc<TypedValue>) -> Rc<TypedValue<'r>>
    {
        let ptr_ty = ptr.ty();
        if let LangType::Ptr(pointee_ty) = &ptr_ty {
            assert!(ptr_ty.is_pointer());

            let builder = self.builder();
            let loaded = builder.build_load(&ptr.llvm());

            let typed = TypedValue::new(
                self.llvm_ctx.clone(),
                self.builder().clone(),
                mk_rcbox(loaded),
                *pointee_ty.clone()
            ).unwrap();

            Rc::new(typed)
        } else {
            panic!("Tried derefing non-pointer type: '{:?}'.", ptr_ty);
        }
    }

    fn local_ptr(&self,
                 name: impl Into<String>) -> Option<Rc<TypedValue<'r>>>
    {
        let parent = self.parent_func.clone()?;
        let loc = parent.borrow().scope.local(name.into())?;
        Some(loc)
    }

    fn global_ptr(&self,
                  name: impl Into<String>) -> Option<Rc<TypedValue<'r>>>
    {
        let ref key = name.into();
        let ptr = self.globals.get(key)?;
        Some(ptr.clone())
    }

    fn variable_ptr(&self,
                    name: impl Into<String>) -> Option<Rc<TypedValue<'r>>>
    {
        let ref key = name.into();
        self.local_ptr(key).or(self.global_ptr(key))
    }

    fn variable(&self,
                name: impl Into<String>) -> Option<Rc<TypedValue<'r>>>
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
               argnames: Vec<(LangType, String)>,
               ret_ty: LangType,
               ty: &llvm::FunctionType) -> RcRef<FuncMeta<'r>>
    {
        let funcname = name.into();
        let func = self.llvm_module.add_function(&funcname, ty);
        let funcmeta = FuncMeta::new(
            self.llvm_ctx.clone(),
            func,
            argnames,
            ret_ty
        );

        let funcmeta_rcref = mk_rcref(funcmeta);
        self.funcs.insert(funcname.clone(), funcmeta_rcref.clone());
        funcmeta_rcref
    }

    fn mk_global_var(&mut self,
                     name: impl AsRef<str>,
                     val: Rc<TypedValue<'r>>) -> Rc<TypedValue<'r>>
    {
        let varname = name.as_ref();

        let val_llvm = val.llvm();
        let glob = self.llvm_module.add_global_variable(varname, &val_llvm);
        let glob = glob.to_super().to_super();

        let typed = TypedValue::new(
            self.llvm_ctx.clone(),
            self.builder().clone(),
            mk_rcbox(glob),
            LangType::new_ptr(val.ty())
        ).unwrap();

        self.globals.insert(varname.to_string(), Rc::new(typed));
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

impl<'r> Codegen<'r, Rc<TypedValue<'r>>> for NumLiteralExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> Rc<TypedValue<'r>> {
        let val = self.value as i64;
        let num = val.compile(&ctx.llvm_ctx);
        let typed = TypedValue::new(
            ctx.llvm_ctx.clone(), ctx.builder().clone(),
            mk_rcbox(num), LangType::Long
        ).unwrap();
        Rc::new(typed)
    }
}

impl<'r> Codegen<'r, Rc<TypedValue<'r>>> for StringLiteralExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> Rc<TypedValue<'r>> {
        let elements = self.chars.clone()
            .into_iter()
            .map(|chr| chr.compile(&ctx.llvm_ctx))
            .map(|value| {
                TypedValue::new(
                    ctx.llvm_ctx.clone(), ctx.builder().clone(),
                    mk_rcbox(value), LangType::Byte
                ).unwrap()
            })
            .collect::<Vec<TypedValue>>();

        //let allocator = ;
        let allocator = ctx.allocator();
        let allocated = allocator.arr_byelems(LangType::Byte, elements)
            .unwrap_or_else(|err| panic!("'{}' when allocating string. ", err.msg));

        Rc::new(allocated)
    }
}

impl<'r> Codegen<'r, Rc<TypedValue<'r>>> for ValuelikeExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> Rc<TypedValue<'r>> {
        match self {
            ValuelikeExprAST::NumericLiteral(num) => {
                num.gencode(ctx)
            },

            ValuelikeExprAST::StringLiteral(string) => {
                string.gencode(ctx)
            },

            ValuelikeExprAST::Variable(ident) => {
                let ref key = ident.name;
                if let Some(val) = ctx.variable(key) {
                    val
                } else {
                    panic!("'{}' not found.", key)
                }
            },

            ValuelikeExprAST::Casted(cast) => {
                let value = cast.value.gencode(ctx);
                let cast_ty = cast.target_ty.gencode(ctx);
                let casted = value.clone()
                    .cast(cast_ty.clone())
                    .unwrap_or_else(|| panic!("Unable to cast '{:?}' as {:?}", &value.ty(), cast_ty));
                Rc::new(casted)
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

                let ref args_typed = call.args
                    .iter()
                    .map(|arg| arg.gencode(ctx))
                    .collect::<Vec<Rc<TypedValue<'r>>>>();
                let ref args_llvm = args_typed
                    .iter()
                    .map(|arg| arg.llvm())
                    .collect::<Vec<RcBox<llvm::Value>>>();
                let argslice = &mk_slice(args_llvm)[..];
                let builder = ctx.builder();
                let ret_value = builder
                    .build_call(funcmeta.borrow().func.as_ref(), argslice);
                let ret_ty = funcmeta.borrow().ret_ty.clone();

                Rc::new(
                    TypedValue::new(
                        ctx.llvm_ctx.clone(), ctx.builder().clone(),
                        mk_rcbox(&ret_value), ret_ty
                    ).unwrap()
                )
            },

            ValuelikeExprAST::BinExpression(binexpr) => {
                binexpr.gencode(ctx)
            },

            ValuelikeExprAST::UnaryExpression(unary) => {
                unary.gencode(ctx)
            }
        }
    }
}

impl<'r> Codegen<'r, Rc<TypedValue<'r>>> for BinOpExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> Rc<TypedValue<'r>> {
        let mut lhs = self.lhs.gencode(ctx);
        let mut rhs = self.rhs.gencode(ctx);
        let lhs_ty = lhs.ty();
        let rhs_ty = rhs.ty();

        // Detect if we're dealing with pointer arithmetic.
        let ptr_artih = lhs_ty.is_pointer() && rhs_ty.is_integer();

        // If adding two integers of different sizes, cast smaller one to bigger one.
        if lhs_ty != rhs_ty && !ptr_artih {
            if lhs_ty.is_integer() && rhs_ty.is_integer() {
                if lhs_ty.size() > rhs_ty.size() {
                    rhs = Rc::new(rhs.cast(lhs_ty.clone()).unwrap());
                } else {
                    lhs = Rc::new(lhs.cast(rhs_ty.clone()).unwrap());
                }
            } else {
                panic!("Operation '{:?}' on types '{:?}' and '{:?}'.",
                       self.op, lhs_ty, rhs_ty);
            }
        }

        let builder = ctx.builder();
        let lhs_ty = lhs.ty();
        let rhs_ty = rhs.ty();
        let ref lhs = lhs.llvm();
        let ref rhs = rhs.llvm();


        let (value, ty) = match self.op {
            BinOp::ADD => {
                let val = if ptr_artih {
                    builder.build_unsafe_gep(lhs, &[rhs])
                } else {
                    builder.build_add(lhs, rhs)
                };
                (val, lhs_ty)
            },

            BinOp::SUB => {
                let neg_rhs = builder.build_neg(rhs);
                let val = if ptr_artih {
                    builder.build_unsafe_gep(lhs, &[neg_rhs])
                } else {
                    builder.build_sub(lhs, rhs)
                };
                (val, lhs_ty)
            },

            BinOp::MUL =>
                (builder.build_mul(lhs, rhs), lhs_ty),

            BinOp::DIV =>
                (builder.build_div(lhs, rhs), lhs_ty),

            BinOp::MOD =>
                panic!("No modulo for now"),

            BinOp::AND =>
                (builder.build_and(lhs, rhs), lhs_ty),

            BinOp::OR =>
                (builder.build_or(lhs, rhs), lhs_ty),

            BinOp::BIT_AND | BinOp::BIT_OR =>
                panic!("No bitwise AND and OR."),

            BinOp::LT =>
                (builder.build_cmp(lhs, rhs, llvm::Predicate::LessThan),
                 LangType::Bool),

            BinOp::LTE =>
                (builder.build_cmp(lhs, rhs, llvm::Predicate::LessThanOrEqual),
                 LangType::Bool),

            BinOp::GT =>
                (builder.build_cmp(lhs, rhs, llvm::Predicate::GreaterThan),
                 LangType::Bool),

            BinOp::GTE =>
                (builder.build_cmp(lhs, rhs, llvm::Predicate::GreaterThanOrEqual),
                 LangType::Bool),

            BinOp::EQ =>
                (builder.build_cmp(lhs, rhs, llvm::Predicate::Equal),
                 LangType::Bool),

            BinOp::NON_EQ =>
                (builder.build_cmp(lhs, rhs, llvm::Predicate::NotEqual),
                 LangType::Bool)
        };

        Rc::new(TypedValue::new(
            ctx.llvm_ctx.clone(), ctx.builder().clone(),
            mk_rcbox(value), ty
        ).unwrap())
    }
}

impl<'r> Codegen<'r, Rc<TypedValue<'r>>> for UnaryOpExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> Rc<TypedValue<'r>> {
        let ref builder = ctx.builder();

        let (value, ty) = match self.op {
            UnaryOp::NOT => {
                let ref value = self.expr.gencode(ctx);
                (builder.build_not(&value.llvm()), value.ty())
            },
            UnaryOp::MINUS => {
                let ref value = self.expr.gencode(ctx);
                (builder.build_neg(&value.llvm()), value.ty())
            },
            UnaryOp::REF => {
                match &self.expr {
                    ValuelikeExprAST::Variable(ident) => {
                        let ptr = ctx.variable_ptr(&ident.name);
                        let ptr = ptr.unwrap_or_else(
                            || panic!("&{0}: No such value with identifier '{0}'.",
                                      &ident.name));

                        // Get reference to local stack variable
                        let ptr_ty = ptr.ty();
                        assert!(ptr_ty.is_pointer());

                        return ptr
                    },
                    x => {
                        panic!("Unable to get pointer of '{:?}'.", x)
                    }
                }
            },
            UnaryOp::DEREF => {
                let ref value = self.expr.gencode(ctx);
                if let LangType::Ptr(pointee_ty) = value.ty() {
                    let loaded = builder.build_load(&value.llvm());
                    (loaded, *pointee_ty)
                } else {
                    panic!("Tried dereferencing {:?}, which is not a pointer type.", value.llvm());
                }
            },
        };

        Rc::new(TypedValue::new(
            ctx.llvm_ctx.clone(), ctx.builder().clone(),
            mk_rcbox(value), ty
        ).unwrap())
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

            // Ignored
            InBlockExprAST::Valuelike(_) => {}

            InBlockExprAST::Declaration(decl) => {
                let ref name = decl.ident.name;
                let ty = decl.ty.gencode(ctx); //.as_llvm(ctx.llvm_ctx.clone());

                match &decl.value {
                    DeclarationRHSExprAST::Valuelike(valuelike) => {
                        let valuelike = valuelike.gencode(ctx);
                        let casted = valuelike.clone().cast(ty.clone());
                        // Try to implicitly cast value to target type
                        if let Some(casted) = casted {
                            parenfunc.borrow_mut().mk_local_var(name, Rc::new(casted))
                        } else {
                            panic!("Tried declaring '{}' as {:?}, but value is {:?}",
                                   name, ty, valuelike.ty());
                        }
                    },

                    DeclarationRHSExprAST::Array(
                        ArrayDeclarationExprAST::BySize(sizeval)
                    ) => {
                        let size = sizeval.gencode(ctx);

                        // Arrays should have pointer type specified.
                        if let LangType::Ptr(pointee_ty) = ty {
                            parenfunc
                                .borrow_mut()
                                .mk_local_arr_ofsize(name, *pointee_ty, size);
                            return;
                        }

                        panic!("Array declaration should always be a pointer type.")
                    },

                    DeclarationRHSExprAST::Array(
                        ArrayDeclarationExprAST::ByElements(elems)
                    ) => {
                        if let LangType::Ptr(pointee_ty) = ty {
                            let values = elems
                                .into_iter()
                                .map(|elem| elem.gencode(ctx))
                                .collect::<Vec<Rc<TypedValue>>>();

                            parenfunc
                                .borrow_mut()
                                .mk_local_arr_byelems(name, *pointee_ty.clone(), values);
                            return ;
                        }

                        panic!("Array declaration should always be a pointer type.")
                    }
                };
            },

            InBlockExprAST::ReDeclaration(redecl) => {
                let valuelike = redecl.value.gencode(ctx);
                match &redecl.target {
                    WritableExprAST::Variable(ident) => {
                        let ref name = ident.name;
                        let local_ptr = ctx.local_ptr(name)
                            .unwrap_or_else(|| panic!("Redeclaration of '{}'. No such name in current scope.", name));

                        if let LangType::Ptr(pointee_ty) = local_ptr.ty() {
                            let builder = ctx.builder();
                            // Try to cast valuelike to pointee_ty
                            let casted = valuelike.cast(*pointee_ty.clone())
                                .unwrap_or_else(|| panic!("Redeclaration: unable to cast to '{:?}'.", pointee_ty));
                            builder.build_store(&casted.llvm(), &local_ptr.llvm());
                        } else {
                            panic!("Redeclaration: LHS {:?} was expected to be a pointer.", valuelike.llvm());
                        }
                    },

                    WritableExprAST::PtrWrite(deref) => {
                        let target_ptr = deref.target.gencode(ctx);
                        if let LangType::Ptr(pointee_ty) = target_ptr.ty() {
                            let builder = ctx.builder();
                            // Try to cast valuelike to pointee_ty
                            let casted = valuelike.cast(*pointee_ty.clone())
                                .unwrap_or_else(|| panic!("PtrWrite: unable to cast to '{:?}'.", pointee_ty));
                            builder.build_store(&casted.llvm(), &target_ptr.llvm());
                        } else {
                            panic!("PtrWrite: {:?} was expected to be a pointer.", deref.target);
                        }
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
                ctx.builder().build_cond_br(&cond.llvm(), &b1, Some(&b2));
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
                ctx.builder().build_cond_br(&cond.llvm(), &loopblk, Some(&exitblk));

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

impl<'r> Codegen<'r, LangType> for PrimitiveTypeExprAST {
    fn gencode(&self, _: &mut Context<'r>) -> LangType {
        match self {
            Self::Byte  => LangType::Byte,
            Self::Short => LangType::Short,
            Self::Int   => LangType::Int,
            Self::Long  => LangType::Long,
            Self::Void  => LangType::Void,
        }
    }
}

impl<'r> Codegen<'r, LangType> for PtrTypeExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> LangType {
        let ty = self.pointee.gencode(ctx);
        let ptr_ty = LangType::Ptr(Box::new(ty));
        ptr_ty
    }
}

impl<'r> Codegen<'r, LangType> for TypeExprAST {
    fn gencode(&self, ctx: &mut Context<'r>) -> LangType {
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

        let preamb_blk = FuncMeta::mk_preamb(funcmeta.clone());
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
            .map(|(ident, ty)| (ty.gencode(ctx), ident.name.clone()))
            .collect::<Vec<(LangType, String)>>();

        let ret_type = prot.ret_type
            .gencode(ctx);
        let arg_types = arg_names.iter()
            .map(|(ty, _)| ty.as_llvm(ctx.llvm_ctx.clone()))
            .collect::<Vec<RcBox<llvm::Type>>>();
        let ret_ty = ret_type.as_llvm(ctx.llvm_ctx.clone());
        let sig = llvm::FunctionType::new(
            &ret_ty,
            &mk_slice(&arg_types)[..]
        );
        ctx.mk_func(funcname.clone(), arg_names, ret_type, sig)
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
                    let valuelike = valuelike.gencode(ctx);
                    let ty = decl.ty.gencode(ctx);
                    // Cast valuelike to variable type
                    let casted = valuelike.clone().cast(ty.clone())
                        .unwrap_or_else(|| panic!("Tried declaring '{}' as {:?}, but value is {:?}",
                                                  name, ty, valuelike.ty()));
                    ctx.mk_global_var(name, Rc::new(casted));
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
