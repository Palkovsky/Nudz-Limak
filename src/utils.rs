pub use std::rc::Rc;
pub use std::cell::RefCell;

pub use llvm::CSemiBox;

pub type RcBox<T> = Rc<Box<T>>;
pub type RcRef<T> = Rc<RefCell<T>>;
pub type RcSemiBox<'r, T> = Rc<CSemiBox<'r, T>>;

pub fn mk_box<T>(value: &T) -> Box<T> {
    let ptr = (value as *const T) as usize;
    unsafe { Box::from_raw(ptr as *mut T) }
}

pub fn mk_rcbox<T>(value: &T) -> RcBox<T> {
    let boxed = mk_box(value);
    Rc::new(boxed)
}

pub fn mk_rcref<T>(value: T) -> RcRef<T> {
    Rc::new(RefCell::new(value))
}

pub fn mk_slice<T>(vec: &Vec<RcBox<T>>) -> Vec<&T> {
    unsafe {
        let ptrs = vec
            .iter()
            .map(|item| item.as_ref().as_ref() as *const T)
            .map(|ptr| ptr.as_ref().unwrap())
            .collect::<Vec<&T>>();
        ptrs
    }
}
