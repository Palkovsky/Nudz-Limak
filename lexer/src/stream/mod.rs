pub trait Stream<T>: Clone {
    fn next(&mut self) -> Option<T>;
    fn revert(&mut self) -> Option<T>;

    fn is_finished(&mut self) -> bool {
        let item = self.next();

        if item.is_some() {
            self.revert();
            false
        } else {
            true
        }
    }

    fn peek1(&mut self) -> Option<T> {
        let item = self.next();
        self.revert();
        item
    }

    fn peek(&mut self, n: usize) -> Vec<T> {
        let mut i = 0;
        let mut buff = Vec::with_capacity(n);

        while i < n {
            if let Some(x) = self.next() {
                buff.push(x);
                i += 1;
            } else {
                break;
            }
        }

        for _ in 0..i {
            self.revert();
        }

        buff
    }
}

#[derive(Debug, Clone)]
pub struct VecStream<T: Clone> {
    vector: Vec<T>,
    stack: Vec<T>
}

impl<T: Clone> VecStream<T> {
    pub fn from(vector: Vec<T>) -> Self {
        let n = vector.len();
        Self {
            vector: vector,
            stack: Vec::with_capacity(n)
        }
    }
}

impl<T: Clone> Stream<T> for VecStream<T> {
    fn next(&mut self) -> Option<T> {
        if self.vector.is_empty() {
            None
        } else {
            let item = self.vector.remove(0);
            self.stack.push(item.clone());
            Some(item)
        }
    }

    fn revert(&mut self) -> Option<T> {
        let item = self.stack.pop()?;
        self.vector.insert(0, item.clone());
        Some(item)
    }
}
