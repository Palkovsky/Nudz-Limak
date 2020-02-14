extern crate llvm;

use super::utils::*;


pub trait Typed {
    fn as_llvm(&self, ctx: RcBox<llvm::Context>) -> RcBox<llvm::Type>;
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
    pointee: RcBox<dyn Typed>
}

impl Ptr {
    pub fn new(pointee: RcBox<dyn Typed>) -> Self {
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
pub struct TypedValue<T> {
    value: RcBox<llvm::Value>,
    ty: T
}

impl<T: Typed> TypedValue<T> {
    pub fn new(ctx: RcBox<llvm::Context>,
           value: RcBox<llvm::Value>,
           ty: T) -> Result<Self, impl Into<String>>
    {
        let value_ty = value.get_type();
        let typed_ty = ty.as_llvm(ctx);

        if *value_ty != **typed_ty {
            return Err("Value type doesn't match type specified.");
        }

        Ok(Self { value: value, ty: ty })
    }

    pub fn ty(&self) -> &T {
        &self.ty
    }

    pub fn value(&self) -> RcBox<llvm::Value> {
        self.value.clone()
    }
}


/// This trait represents objects castable to T.
pub trait Cast<T: Typed> {
    fn build<'r>(&self, builder: RcSemiBox<'r, llvm::Builder>) -> T;
}

