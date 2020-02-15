extern crate llvm;

use super::utils::*;

#[derive(Debug, PartialEq, Clone)]
pub enum LangType {
    Void,
    Bool,
    Byte,
    Short,
    Int,
    Long,
    Ptr(Box<LangType>)
}

macro_rules! accessor {
    ($lhs: ident, $rhs: pat) => {
        pub fn $lhs(&self) -> bool {
            match self {
                $rhs => true,
                _    => false
            }
        }
    }
}

#[macro_export]
macro_rules! llvm_typeof {
    ($ctx: expr, $typ: ty) => {
        mk_rcbox(llvm::Type::get::<$typ>(&$ctx))
    }
}

impl LangType {
    pub fn as_llvm(&self, ctx: RcBox<llvm::Context>) -> RcBox<llvm::Type> {
        match self {
            Self::Void =>
                llvm_typeof!(ctx, ()),
            Self::Bool =>
                llvm_typeof!(ctx, bool),
            Self::Byte =>
                llvm_typeof!(ctx, i8),
            Self::Short =>
                llvm_typeof!(ctx, i16),
            Self::Int =>
                llvm_typeof!(ctx, i32),
            Self::Long =>
                llvm_typeof!(ctx, i64),
            Self::Ptr(pointee) => {
                let llvm_ty = pointee.as_llvm(ctx);
                mk_rcbox(llvm::PointerType::new(&llvm_ty))

            }
        }
    }

    pub fn new_ptr(pointee: LangType) -> LangType {
        LangType::Ptr(Box::new(pointee))
    }

    accessor!(is_void, Self::Void);
    accessor!(is_bool, Self::Bool);
    accessor!(is_byte, Self::Byte);
    accessor!(is_short, Self::Short);
    accessor!(is_int, Self::Int);
    accessor!(is_long, Self::Long);
    accessor!(is_pointer, Self::Ptr(_));

    pub fn is_integer(&self) -> bool {
        match self {
            Self::Byte | Self::Short | Self::Int | Self::Long => true,
            _ => false
        }
    }
}

/// Container for llvm Value with Typed type.
pub struct TypedValue<'r> {
    ctx: RcBox<llvm::Context>,
    builder: RcSemiBox<'r, llvm::Builder>,
    value: RcBox<llvm::Value>,
    ty:    LangType
}

impl<'r> TypedValue<'r> {
    pub fn new(ctx: RcBox<llvm::Context>,
               builder: RcSemiBox<'r, llvm::Builder>,
               value: RcBox<llvm::Value>,
               ty: LangType) -> Result<Self, String>
    {
        let value_ty = value.get_type();
        let typed_ty = ty.as_llvm(ctx.clone());

        if *value_ty != **typed_ty {
            return Err(
                format!("Value type doesn't match type specified. Expected '{:?}', actual: '{:?}'",
                        **typed_ty, *value_ty)
            );
        }

        Ok(Self { ctx: ctx, builder: builder.clone(), value: value, ty: ty })
    }

    pub fn llvm(&self) -> RcBox<llvm::Value> {
        self.value.clone()
    }

    pub fn ty(&self) -> LangType {
        self.ty.clone()
    }

    fn trunc_to(&self, ty: LangType) -> Option<TypedValue> {
        let trunced = self.builder.build_trunc(
            self.llvm().as_ref(),
            &ty.as_llvm(self.ctx.clone())
        );
        TypedValue::new(
            self.ctx.clone(),
            self.builder.clone(),
            mk_rcbox(trunced),
            LangType::Byte
        ).ok()
    }

    fn extend_to(&self, ty: LangType) -> Option<TypedValue> {
        let extended = self.builder.build_zext(
            self.llvm().as_ref(),
            &ty.as_llvm(self.ctx.clone())
        );
        TypedValue::new(
            self.ctx.clone(),
            self.builder.clone(),
            mk_rcbox(extended),
            LangType::Byte
        ).ok()
    }

    pub fn cast_as_byte(&self) -> Option<TypedValue> {
        match self.ty() {
            // Void/ptr to byte seems weird
            LangType::Void | LangType::Ptr(_) | LangType::Bool => None,
            // Return new TypedValue, no need to use builder.
            LangType::Byte =>
                TypedValue::new(
                    self.ctx.clone(),
                    self.builder.clone(),
                    self.llvm(),
                    LangType::Byte
                ).ok(),
            // Trunc to byte
            LangType::Short | LangType::Int | LangType::Long =>
                self.trunc_to(LangType::Byte)
        }
    }

    pub fn cast_as_short(&self) -> Option<TypedValue> {
        match self.ty() {
            // Void/ptr to byte seems weird
            LangType::Void | LangType::Ptr(_) | LangType::Bool =>
                None,
            // Extend to short
            LangType::Byte =>
                self.extend_to(LangType::Short),
            // Do nothing
            LangType::Short =>
                TypedValue::new(
                    self.ctx.clone(),
                    self.builder.clone(),
                    self.llvm(),
                    LangType::Short
                ).ok(),
            // Truncate
            LangType::Int | LangType::Long =>
                self.trunc_to(LangType::Short)
        }
    }

    pub fn cast_as_int(&self) -> Option<TypedValue> {
        match self.ty() {
            // Void/ptr to byte seems weird
            LangType::Void | LangType::Ptr(_) | LangType::Bool =>
                None,
            // Extend to int
            LangType::Byte | LangType::Short =>
                self.extend_to(LangType::Int),
            // Do nothing
            LangType::Int =>
                TypedValue::new(
                    self.ctx.clone(),
                    self.builder.clone(),
                    self.llvm(),
                    LangType::Int
                ).ok(),
            // Truncate to int
            LangType::Long =>
                self.trunc_to(LangType::Int)
        }
    }

    pub fn cast_as_long(&self) -> Option<TypedValue> {
        match self.ty() {
            // Void/ptr to byte seems weird
            LangType::Void | LangType::Bool =>
                None,
            // Pointers should be convertable to longs
            LangType::Ptr(_) => {
                let ptr_as_int = self.builder.build_ptr_to_int(
                    &self.llvm(),
                    &LangType::Long.as_llvm(self.ctx.clone())
                );
                TypedValue::new(
                    self.ctx.clone(),
                    self.builder.clone(),
                    mk_rcbox(ptr_as_int),
                    LangType::Long
                ).ok()
            },
            // Extend to long
            LangType::Byte | LangType::Short | LangType::Int =>
                self.extend_to(LangType::Long),
            // Do nothing
            LangType::Long =>
                TypedValue::new(
                    self.ctx.clone(),
                    self.builder.clone(),
                    self.llvm(),
                    LangType::Long
                ).ok()
        }
    }

    pub fn cast_as_ptr(&self, pointee: LangType) -> Option<TypedValue> {
        let target_ty = LangType::Ptr(Box::new(pointee));

        match self.ty() {
            LangType::Void | LangType::Bool | LangType::Byte | LangType::Short | LangType::Int =>
                None,
            // Long to ptr cast
            LangType::Long => {
                let casted = self.builder.build_int_to_ptr(
                    &self.llvm(),
                    &target_ty.as_llvm(self.ctx.clone())
                );
                TypedValue::new(
                    self.ctx.clone(),
                    self.builder.clone(),
                    mk_rcbox(casted),
                    target_ty
                ).ok()
            },
            // Ptr to ptr cast
            LangType::Ptr(_) => {
                let casted = self.builder.build_bit_cast(
                    &self.llvm(),
                    &target_ty.as_llvm(self.ctx.clone())
                );
                TypedValue::new(
                    self.ctx.clone(),
                    self.builder.clone(),
                    mk_rcbox(casted),
                    target_ty
                ).ok()
            }
        }
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_is_methods() {
        assert_eq!(LangType::Byte.is_int(), false);
        assert_eq!(LangType::Byte.is_byte(), true);

        assert_eq!(LangType::Short.is_int(), false);
        assert_eq!(LangType::Short.is_short(), true);

        assert_eq!(LangType::Long.is_short(), false);
        assert_eq!(LangType::Long.is_long(), true);

        assert_eq!(LangType::Bool.is_byte(), false);
        assert_eq!(LangType::Bool.is_bool(), true);

        assert_eq!(LangType::new_ptr(LangType::Byte).is_pointer(), true);
        assert_eq!(LangType::new_ptr(LangType::Byte).is_int(), false);
    }
}
