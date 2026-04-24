use std::ops::{Add, AddAssign, Mul};

use crate::arithmetic_utils::Ring;

/// A formal linear combination of pure wedges in `∧^N A`.
///
/// Elements are sums of pure wedge products scaled by coefficients from `Coeffs`. The
/// decomposition into summands is an implementation detail; only the total element
/// of `∧^N A` that it represents matters to callers.
pub struct ExteriorPower<A, Coeffs, const N: usize>
where
    Coeffs: Ring,
{
    summands: Vec<([A; N], Coeffs)>,
}

impl<A, Coeffs, const N: usize> ExteriorPower<A, Coeffs, N>
where
    Coeffs: Ring,
{
    /// The zero element of `∧^N A`: the empty linear combination.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            summands: Vec::new(),
        }
    }

    /// A single pure wedge scaled by `coeff`.
    pub fn from_pure_wedge(wedge: [A; N], coeff: Coeffs) -> Self {
        Self {
            summands: vec![(wedge, coeff)],
        }
    }

    pub(crate) fn into_summands(self) -> Vec<([A; N], Coeffs)> {
        self.summands
    }
}

impl<A, Coeffs, const N: usize> Add for ExteriorPower<A, Coeffs, N>
where
    Coeffs: Ring,
{
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        self.summands.extend(rhs.summands);
        self
    }
}

impl<A, Coeffs, const N: usize> AddAssign for ExteriorPower<A, Coeffs, N>
where
    Coeffs: Ring,
{
    fn add_assign(&mut self, rhs: Self) {
        self.summands.extend(rhs.summands);
    }
}

impl<A, Coeffs, const N: usize> Mul<Coeffs> for ExteriorPower<A, Coeffs, N>
where
    Coeffs: Ring,
{
    type Output = Self;
    fn mul(self, scalar: Coeffs) -> Self {
        Self {
            summands: self
                .summands
                .into_iter()
                .map(|(w, c)| (w, c * scalar.clone()))
                .collect(),
        }
    }
}
