use std::ops::{Add, AddAssign, Mul};

use crate::arithmetic_utils::Ring;

/// A formal linear combination of pure tensors in `A^⊗N`.
///
/// Elements are sums of pure tensors scaled by coefficients from `Coeffs`. The
/// decomposition into summands is an implementation detail; only the total element
/// of `A^⊗N` that it represents matters to callers.
pub struct TensorPower<A, Coeffs, const N: usize>
where
    Coeffs: Ring,
{
    summands: Vec<([A; N], Coeffs)>,
}

impl<A, Coeffs, const N: usize> TensorPower<A, Coeffs, N>
where
    Coeffs: Ring,
{
    /// The zero element of `A^⊗N`: the empty linear combination.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            summands: Vec::new(),
        }
    }

    /// A single pure tensor scaled by `coeff`.
    pub fn from_pure_tensor(tensor: [A; N], coeff: Coeffs) -> Self {
        Self {
            summands: vec![(tensor, coeff)],
        }
    }

    pub(crate) fn into_summands(self) -> Vec<([A; N], Coeffs)> {
        self.summands
    }
}

impl<A, Coeffs, const N: usize> Add for TensorPower<A, Coeffs, N>
where
    Coeffs: Ring,
{
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        self.summands.extend(rhs.summands);
        self
    }
}

impl<A, Coeffs, const N: usize> AddAssign for TensorPower<A, Coeffs, N>
where
    Coeffs: Ring,
{
    fn add_assign(&mut self, rhs: Self) {
        self.summands.extend(rhs.summands);
    }
}

impl<A, Coeffs, const N: usize> Mul<Coeffs> for TensorPower<A, Coeffs, N>
where
    Coeffs: Ring,
{
    type Output = Self;
    fn mul(self, scalar: Coeffs) -> Self {
        Self {
            summands: self
                .summands
                .into_iter()
                .map(|(t, c)| (t, c * scalar.clone()))
                .collect(),
        }
    }
}
