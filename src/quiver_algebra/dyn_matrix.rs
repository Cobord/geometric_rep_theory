use std::ops::{Add, AddAssign, Mul, MulAssign, Neg};

use nalgebra::DMatrix;
use num::{One, Zero};

use crate::quiver_algebra::checked_arith::{ChainMultiplyable, CheckedAdd, CheckedAddAssign};

/// A wrapper around [`nalgebra::DMatrix<T>`] that implements the [`CheckedMul`] and [`CheckedAdd`]
/// families of traits, returning [`ShapeMismatch`] errors instead of panicking when matrix
/// dimensions are incompatible.
/// In addition it is the opposite algebra.
/// This means that the multiplication is reversed.
#[derive(Clone, PartialEq, Debug)]
pub struct DynMatrix<T: nalgebra::Scalar>(pub DMatrix<T>);

/// Error returned when two matrices have incompatible shapes for the requested operation.
#[derive(Debug, Clone)]
pub struct ShapeMismatch {
    pub lhs: (usize, usize),
    pub rhs: (usize, usize),
}

impl<T: nalgebra::Scalar + Zero> DynMatrix<T> {
    /// Construct a zero matrix of the given dimensions.
    #[must_use = "A possibly large zero filled matrix is returned"]
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        DynMatrix(DMatrix::zeros(nrows, ncols))
    }
}

impl<T: nalgebra::Scalar + Zero + One> DynMatrix<T> {
    /// Construct an n×n identity matrix.
    #[must_use = "A possibly large identity matrix is returned"]
    pub fn identity(n: usize) -> Self {
        DynMatrix(DMatrix::identity(n, n))
    }
}

// ── CheckedMul ───────────────────────────────────────────────────────────────

impl<T> ChainMultiplyable for DynMatrix<T>
where
    T: nalgebra::Scalar
        + Copy
        + Zero
        + One
        + Add<T, Output = T>
        + AddAssign<T>
        + Mul<T, Output = T>
        + MulAssign<T>,
{
    type MultiplicationError = ShapeMismatch;

    fn mul_two(self, mut then_this: Self) -> Result<Self, Self::MultiplicationError> {
        let (a_rows, a_cols) = self.0.shape();
        let (b_rows, b_cols) = then_this.0.shape();
        if b_cols == a_rows {
            then_this.0 *= self.0;
            Ok(then_this)
        } else {
            Err(ShapeMismatch {
                lhs: (a_rows, a_cols),
                rhs: (b_rows, b_cols),
            })
        }
    }

    fn chain_multiply_after(
        mut self,
        these_ops: impl IntoIterator<Item = Self>,
    ) -> Result<Self, Self::MultiplicationError> {
        for next_item in these_ops {
            self = Self::mul_two(self, next_item)?;
        }
        Ok(self)
    }
}

// ── CheckedAdd ───────────────────────────────────────────────────────────────

impl<T> CheckedAdd for DynMatrix<T>
where
    T: nalgebra::Scalar + Copy + Add<Output = T> + AddAssign,
{
    type AdditionError = ShapeMismatch;

    fn will_error(&self, rhs: &Self) -> bool {
        self.0.shape() != rhs.0.shape()
    }

    fn checked_add(self, rhs: Self) -> Result<Self, ShapeMismatch> {
        if <Self as CheckedAdd>::will_error(&self, &rhs) {
            Err(ShapeMismatch {
                lhs: self.0.shape(),
                rhs: rhs.0.shape(),
            })
        } else {
            Ok(DynMatrix(self.0 + rhs.0))
        }
    }
}

impl<T> CheckedAddAssign for DynMatrix<T>
where
    T: nalgebra::Scalar + Copy + Add<Output = T> + AddAssign,
{
    type AdditionError = ShapeMismatch;

    fn will_error(&self, rhs: &Self) -> bool {
        self.0.shape() != rhs.0.shape()
    }

    fn checked_add_assign(&mut self, rhs: Self) -> Result<(), ShapeMismatch> {
        if <Self as CheckedAddAssign>::will_error(self, &rhs) {
            Err(ShapeMismatch {
                lhs: self.0.shape(),
                rhs: rhs.0.shape(),
            })
        } else {
            self.0 += rhs.0;
            Ok(())
        }
    }
}

// ── Scalar multiplication and negation (for DGModule) ────────────────────────

impl<T> MulAssign<T> for DynMatrix<T>
where
    T: nalgebra::Scalar + Copy + Mul<Output = T> + MulAssign,
{
    fn mul_assign(&mut self, rhs: T) {
        self.0 *= rhs;
    }
}

impl<T> Neg for DynMatrix<T>
where
    T: nalgebra::Scalar + Copy + Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self {
        DynMatrix(-self.0)
    }
}
