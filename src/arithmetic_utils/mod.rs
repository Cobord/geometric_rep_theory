mod checked_arith;
mod dyn_matrix;

pub use checked_arith::{
    ChainMultiplyable, CheckedAdd, CheckedAddAssign, CheckedArithError, Field, Ring, SemiRing,
    rank
};
pub use dyn_matrix::{DynMatrix, ShapeMismatch};