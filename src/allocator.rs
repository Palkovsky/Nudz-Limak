use super::utils::*;
use super::types::*;

use llvm::Compile;

type AllocRes<'r> = Result<TypedValue<'r>, AllocErr>;

pub struct AllocErr {
    pub msg: String
}

impl AllocErr {
    fn new(msg: impl Into<String>) -> Self {
        Self { msg: msg.into() }
    }
}

macro_rules! alloc_err { ($msg: expr) => { Err(AllocErr::new($msg)) } }
macro_rules! alloc_ok { ($value: expr) => { Ok($value) }}
macro_rules! typed { ($sel: ident, $value: expr, $typ: expr) => {
    TypedValue::new($sel.ctx.clone(), $sel.builder.clone(), mk_rcbox($value), $typ)
        .map_err(|err| AllocErr::new(err))
}}

pub struct StackAllocator<'r> {
    ctx:     RcBox<llvm::Context>,
    builder: RcSemiBox<'r, llvm::Builder>
}

impl<'r> StackAllocator<'r> {
    pub fn new(ctx: RcBox<llvm::Context>,
               builder: RcSemiBox<'r, llvm::Builder>) -> Self
    {
        Self { ctx: ctx, builder: builder }
    }

    pub fn val(&self, ty: LangType, default: Option<Rc<TypedValue<'r>>>) -> AllocRes<'r> {
        // Check if types match
        if let Some(value) = &default {
            if value.ty() != ty {
                alloc_err!("Type of default doesn't match type provided.")?;
            }
        }

        // Allocate memory
        let ty_llvm = ty.as_llvm(self.ctx.clone());
        let val_ptr = self.builder.build_alloca(&ty_llvm);

        // Store default if present
        if let Some(value) = default {
            self.builder.build_store(&value.llvm(), &val_ptr);
        }

        // Return pointer
        let typed = typed!(self, val_ptr, LangType::new_ptr(ty))?;
        alloc_ok!(typed)
    }

    /// ty is type of array elements
    pub fn arr_byelems(&self, ty: LangType, elems: Vec<TypedValue<'r>>) -> AllocRes<'r> {
        let size = elems.len();
        let size_value = (size as u64).compile(&self.ctx);
        let size_typed = typed!(self, size_value, LangType::Long)?;

        // Check for type uniformity
        for elem in elems.iter() {
            if elem.ty() != ty {
                alloc_err!("Array alloc by elem failed. Elements don't have unform types.")?;
            }
        }

        let arr_ptr = self.arr_ofsize(ty, Rc::new(size_typed))?;

        for (i, elem) in (0..elems.len()).zip(elems.into_iter()) {
            let offset = (i as i64).compile(&self.ctx);
            let wrie_ptr = self.builder.build_gep(&arr_ptr.llvm(), &[offset]);
            self.builder.build_store(&elem.llvm(), wrie_ptr);
        }

        alloc_ok!(arr_ptr)
    }

    pub fn arr_ofsize(&self, ty: LangType, size: Rc<TypedValue<'r>>) -> AllocRes<'r> {
        if !size.ty().is_integer() {
            alloc_err!("size to allocate must be an integer.")?
        }
        let arr_ptr = self.builder.build_array_alloca(
            &ty.as_llvm(self.ctx.clone()),
            &size.llvm()
        );
        let typed = typed!(self, arr_ptr, LangType::new_ptr(ty))?;
        alloc_ok!(typed)
    }
}
