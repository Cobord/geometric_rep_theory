use std::cell::OnceCell;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

#[derive(Clone, Debug, Default)]
pub struct CachedOnce<T>(OnceCell<T>);

impl<T: Hash> Hash for CachedOnce<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.get().hash(state);
    }
}

impl<T: PartialEq> PartialEq for CachedOnce<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.get() == other.0.get()
    }
}

impl<T: Eq> Eq for CachedOnce<T> {}

impl<T> Deref for CachedOnce<T> {
    type Target = OnceCell<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> From<Option<T>> for CachedOnce<T> {
    fn from(opt: Option<T>) -> Self {
        let cell = OnceCell::new();
        if let Some(v) = opt {
            let _ = cell.set(v);
        }
        CachedOnce(cell)
    }
}
