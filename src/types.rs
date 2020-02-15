extern crate llvm;

use super::utils::*;

pub trait Typed {
    fn as_llvm(&self, ctx: RcBox<llvm::Context>) -> RcBox<llvm::Type>;

    fn cast_as_byte<'r>(&self,
                        _: RcSemiBox<'r, llvm::Builder>) -> Option<Byte> { None }
    fn cast_as_short<'r>(&self,
                         _: RcSemiBox<'r, llvm::Builder>) -> Option<Short> { None }
    fn cast_as_int<'r>(&self,
                       _: RcSemiBox<'r, llvm::Builder>) -> Option<Int> { None }
    fn cast_as_long<'r>(&self,
                        _: RcSemiBox<'r, llvm::Builder>) -> Option<Long> { None }
    fn cast_as_ptr<'r>(&self,
                       _: RcSemiBox<'r, llvm::Builder>,
                       _: &dyn Typed) -> Option<Ptr> { None }
}

/// Implementation of Typed for primitives
pub struct Long;
impl Typed for Long {
    fn as_llvm(&self, ctx: RcBox<llvm::Context>) -> RcBox<llvm::Type> {
        mk_rcbox(llvm::Type::get::<i64>(&ctx))
    }
}

pub struct Int;
impl Typed for Int {
    fn as_llvm(&self, ctx: RcBox<llvm::Context>) -> RcBox<llvm::Type> {
        mk_rcbox(llvm::Type::get::<i32>(&ctx))
    }
}

pub struct Short;
impl Typed for Short {
    fn as_llvm(&self, ctx: RcBox<llvm::Context>) -> RcBox<llvm::Type> {
        mk_rcbox(llvm::Type::get::<i16>(&ctx))
    }
}

pub struct Byte;
impl Typed for Byte {
    fn as_llvm(&self, ctx: RcBox<llvm::Context>) -> RcBox<llvm::Type> {
        mk_rcbox(llvm::Type::get::<i8>(&ctx))
    }
}

pub struct Void;
impl Typed for Void {
    fn as_llvm(&self, ctx: RcBox<llvm::Context>) -> RcBox<llvm::Type> {
        mk_rcbox(llvm::Type::get::<()>(&ctx))
    }
}

/// Pointer type
pub struct Ptr {
    pointee: Box<dyn Typed>
}

impl Ptr {
    pub fn new(pointee: Box<dyn Typed>) -> Self {
        Self { pointee: pointee }
    }
}

impl Typed for Ptr {
    fn as_llvm(&self, ctx: RcBox<llvm::Context>) -> RcBox<llvm::Type> {
        let llvm_ty = self.pointee.as_llvm(ctx);
        mk_rcbox(llvm::PointerType::new(&llvm_ty))
    }
}

/// Container for llvm Value with Typed type.
pub struct TypedValue {
    value: RcBox<llvm::Value>,
    ty:    Box<dyn Typed>
}

impl TypedValue {
    pub fn new(ctx: RcBox<llvm::Context>,
           value: RcBox<llvm::Value>,
           ty: Box<dyn Typed>) -> Result<Self, impl Into<String>>
    {
        let value_ty = value.get_type();
        let typed_ty = ty.as_llvm(ctx);

        if *value_ty != **typed_ty {
            return Err("Value type doesn't match type specified.");
        }

        Ok(Self { value: value, ty: ty })
    }
}
