use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub trait CheckedMul {
    type MultiplicationError;

    fn will_error(&self, rhs: &Self) -> bool;

    /// Multiply but there is a possibility for not being able to be multiplied
    /// like matrix dimensions mismatching
    ///
    /// # Errors
    /// For some reason we could not multiply. Likely something about matrix multiplication.
    fn checked_mul(self, rhs: Self) -> Result<Self, Self::MultiplicationError>
    where
        Self: Sized;
}

impl<T> CheckedMul for T
where
    T: Mul<T, Output = T>,
{
    type MultiplicationError = ();

    fn will_error(&self, _rhs: &Self) -> bool {
        false
    }

    fn checked_mul(self, rhs: Self) -> Result<Self, Self::MultiplicationError>
    where
        Self: Sized,
    {
        Ok(self * rhs)
    }
}

pub trait CheckedMulAssign {
    type MultiplicationError;

    fn will_error(&self, rhs: &Self) -> bool;

    /// Multiply but there is a possibility for not being able to be multiplied
    /// like matrix dimensions mismatching
    ///
    /// # Errors
    /// For some reason we could not multiply. Likely something about matrix multiplication.
    fn checked_mul_assign(&mut self, rhs: Self) -> Result<(), Self::MultiplicationError>;
}

impl<T> CheckedMulAssign for T
where
    T: MulAssign<T>,
{
    type MultiplicationError = ();

    fn will_error(&self, _rhs: &Self) -> bool {
        false
    }

    fn checked_mul_assign(&mut self, rhs: Self) -> Result<(), Self::MultiplicationError>
    where
        Self: Sized,
    {
        *self *= rhs;
        Ok(())
    }
}

pub trait CheckedAdd {
    type AdditionError;

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

pub trait CheckedAddAssign {
    type AdditionError;

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

#[allow(clippy::enum_variant_names)]
pub enum CheckedArithError<T>
where
    T: CheckedAdd + CheckedAddAssign + CheckedMul + CheckedMulAssign,
{
    AddError(<T as CheckedAdd>::AdditionError),
    AddAssignError(<T as CheckedAddAssign>::AdditionError),
    MulError(<T as CheckedMul>::MultiplicationError),
    MulAssignError(<T as CheckedMulAssign>::MultiplicationError),
}

impl<T> CheckedArithError<T>
where
    T: CheckedAdd + CheckedAddAssign + CheckedMul + CheckedMulAssign,
{
    pub fn from_add(value: <T as CheckedAdd>::AdditionError) -> Self {
        Self::AddError(value)
    }

    pub fn from_add_assign(value: <T as CheckedAddAssign>::AdditionError) -> Self {
        Self::AddAssignError(value)
    }

    pub fn from_mul(value: <T as CheckedMul>::MultiplicationError) -> Self {
        Self::MulError(value)
    }

    pub fn from_mul_assign(value: <T as CheckedMulAssign>::MultiplicationError) -> Self {
        Self::MulAssignError(value)
    }
}

pub trait Ring:
    Add<Self, Output = Self>
    + AddAssign<Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Neg<Output = Self>
    + Clone
{
}

impl<T> Ring for T where
    T: Add<Self, Output = Self>
        + AddAssign<Self>
        + Mul<Self, Output = Self>
        + MulAssign<Self>
        + Sub<Self, Output = Self>
        + SubAssign<Self>
        + Neg<Output = Self>
        + Clone
{
}

/// A minimal field-like trait for the exact linear algebra used by the
/// Hochschild complex implementation.
///
/// This intentionally stays separate from `Ring` because Gaussian elimination
/// needs distinguished `0`, `1`, and multiplicative inverses for nonzero pivots.
pub trait Field: Ring + PartialEq {
    fn zero() -> Self;
    fn one() -> Self;
    #[must_use = "division is hard, don't waste it"]
    fn inv(self) -> Self;

    fn is_zero(&self) -> bool {
        self == &Self::zero()
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
