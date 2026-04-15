use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use nonempty::NonEmpty;
use num::{One, Zero};

/// Fallible sequential composition: `mul_two(A, B)` means "do A, then B".
///
/// For matrices this corresponds to right-multiplication (i.e. `B * A` in standard notation),
/// matching the opposite-algebra convention used by `DynMatrix`.
pub trait ChainMultiplyable: Sized {
    type MultiplicationError;

    /// `self ; then_this`
    /// which means
    /// `then_this * self`
    ///
    /// # Errors
    ///
    /// If the multiplication failed.
    /// When these are matrices this would be
    /// when there was a matrix multiplication
    /// of incompatible shapes.
    fn mul_two(self, then_this: Self) -> Result<Self, Self::MultiplicationError>;

    /// Multiply all of `these_ops`
    /// For `[a1...an]`
    /// Do `a1` first and then `a2`
    /// and so on.
    /// So as multiplication
    /// this means `an * ... a1`
    ///
    /// # Errors
    ///
    /// If at any step the multiplication failed.
    /// When `ai` are matrices this would be
    /// when there was a matrix multiplication
    /// of incompatible shapes.
    fn nonempty_chain_multiply(
        mut these_ops: NonEmpty<Self>,
    ) -> Result<Self, Self::MultiplicationError> {
        if these_ops.len() == 1 {
            Ok(these_ops.head)
        } else if these_ops.len() == 2 {
            let a = these_ops.head;
            let b = these_ops.tail.pop().expect("Length is 2");
            Self::mul_two(a, b)
        } else {
            let (first, tail) = (these_ops.head, these_ops.tail);
            first.chain_multiply_after(tail)
        }
    }

    /// Multiply all of `these_ops`
    /// with `self` as `a0`
    /// For `[a1...an]`
    /// Do `self` first then `a1` and then `a2`
    /// and so on.
    /// So as multiplication
    /// this means `an * ... a1 * self`
    ///
    /// # Errors
    ///
    /// If at any step the multiplication failed.
    /// When `ai` are matrices this would be
    /// when there was a matrix multiplication
    /// of incompatible shapes.
    fn chain_multiply_after(
        self,
        these_ops: impl IntoIterator<Item = Self>,
    ) -> Result<Self, Self::MultiplicationError>;
}

impl<T> ChainMultiplyable for T
where
    T: Mul<T, Output = T> + MulAssign<T>,
{
    type MultiplicationError = ();

    fn chain_multiply_after(
        self,
        these_ops: impl IntoIterator<Item = Self>,
    ) -> Result<Self, Self::MultiplicationError> {
        let res = these_ops
            .into_iter()
            .fold(self, |acc, next_op| next_op * acc);
        Ok(res)
    }

    fn mul_two(self, mut then_this: Self) -> Result<Self, Self::MultiplicationError> {
        then_this *= self;
        Ok(then_this)
    }
}

/// Addition that can fail — e.g. when matrix dimensions do not match.
pub trait CheckedAdd {
    type AdditionError;

    /// Returns `true` if adding `rhs` to `self` would fail (e.g. shape mismatch).
    fn will_error(&self, rhs: &Self) -> bool;

    /// Add but there is a possibility for not being able to be added
    /// like matrix dimensions mismatching
    ///
    /// # Errors
    /// For some reason we could not multiply. Likely something about matrix addition.
    fn checked_add(self, rhs: Self) -> Result<Self, Self::AdditionError>
    where
        Self: Sized;
}

impl<T> CheckedAdd for T
where
    T: Add<T, Output = T>,
{
    type AdditionError = ();

    fn will_error(&self, _rhs: &Self) -> bool {
        false
    }

    fn checked_add(self, rhs: Self) -> Result<Self, Self::AdditionError>
    where
        Self: Sized,
    {
        Ok(self + rhs)
    }
}

/// In-place addition that can fail — e.g. when matrix dimensions do not match.
pub trait CheckedAddAssign {
    type AdditionError;

    /// Returns `true` if adding `rhs` to `self` in place would fail.
    fn will_error(&self, rhs: &Self) -> bool;

    /// Add but there is a possibility for not being able to be added
    /// like matrix dimensions mismatching
    ///
    /// # Errors
    /// For some reason we could not multiply. Likely something about matrix addition.
    fn checked_add_assign(&mut self, rhs: Self) -> Result<(), Self::AdditionError>;
}

impl<T> CheckedAddAssign for T
where
    T: AddAssign<T>,
{
    type AdditionError = ();

    fn will_error(&self, _rhs: &Self) -> bool {
        false
    }

    fn checked_add_assign(&mut self, rhs: Self) -> Result<(), Self::AdditionError>
    where
        Self: Sized,
    {
        *self += rhs;
        Ok(())
    }
}

/// Unified error type for operations on types that implement all three checked arithmetic traits.
#[allow(clippy::enum_variant_names)]
pub enum CheckedArithError<T>
where
    T: CheckedAdd + CheckedAddAssign + ChainMultiplyable,
{
    AddError(<T as CheckedAdd>::AdditionError),
    AddAssignError(<T as CheckedAddAssign>::AdditionError),
    ChainMulError(<T as ChainMultiplyable>::MultiplicationError),
}

impl<T> CheckedArithError<T>
where
    T: CheckedAdd + CheckedAddAssign + ChainMultiplyable,
{
    /// Wrap a `CheckedAdd` error.
    pub fn from_add(value: <T as CheckedAdd>::AdditionError) -> Self {
        Self::AddError(value)
    }

    /// Wrap a `CheckedAddAssign` error.
    pub fn from_add_assign(value: <T as CheckedAddAssign>::AdditionError) -> Self {
        Self::AddAssignError(value)
    }

    /// Wrap a `ChainMultiplyable` error.
    pub fn from_mul(value: <T as ChainMultiplyable>::MultiplicationError) -> Self {
        Self::ChainMulError(value)
    }
}

/// A commutative monoid under `+` and a monoid under `*`, with distributivity. No subtraction.
pub trait SemiRing:
    Add<Self, Output = Self>
    + AddAssign<Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Clone
    + One
    + Zero
{
}

impl<T> SemiRing for T where
    T: Add<Self, Output = Self>
        + AddAssign<Self>
        + Mul<Self, Output = Self>
        + MulAssign<Self>
        + Clone
        + One
        + Zero
{
}

/// A `SemiRing` that also supports subtraction and negation.
pub trait Ring: SemiRing + Sub<Self, Output = Self> + SubAssign<Self> + Neg<Output = Self> {}

impl<T> Ring for T where
    T: SemiRing + Sub<Self, Output = Self> + SubAssign<Self> + Neg<Output = Self>
{
}

/// A minimal field-like trait for the exact linear algebra used by the
/// Hochschild complex implementation.
///
/// This intentionally stays separate from `Ring` because Gaussian elimination
/// needs distinguished `0`, `1`, and multiplicative inverses for nonzero pivots.
pub trait Field: Ring + PartialEq {
    #[must_use = "division is hard, don't waste it"]
    fn inv(self) -> Self;
}

impl<T> Field for T
where
    T: Ring + PartialEq + Div<Self, Output = Self>,
{
    fn inv(self) -> Self {
        Self::one() / self
    }
}

#[allow(clippy::needless_range_loop)]
#[must_use = "Use the rank of each differential to determine rank of cohomology vector spaces"]
pub fn rank<Scalar>(matrix: &[Vec<Scalar>]) -> usize
where
    Scalar: Field,
{
    if matrix.is_empty() {
        return 0;
    }
    let mut mat = matrix.to_vec();
    let m = mat.len();
    let n = mat[0].len();
    let mut r = 0;
    let mut c = 0;
    while r < m && c < n {
        let pivot = (r..m).find(|&i| !mat[i][c].is_zero());
        let Some(pivot_row) = pivot else {
            c += 1;
            continue;
        };
        if pivot_row != r {
            mat.swap(r, pivot_row);
        }
        let inv = mat[r][c].clone().inv();
        for j in c..n {
            mat[r][j] *= inv.clone();
        }
        for i in 0..m {
            if i == r || mat[i][c].is_zero() {
                continue;
            }
            let lambda = mat[i][c].clone();
            for j in c..n {
                let correction = lambda.clone() * mat[r][j].clone();
                mat[i][j] -= correction;
            }
        }
        r += 1;
        c += 1;
    }
    r
}
