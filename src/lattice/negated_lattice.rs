use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use num::Zero;
use num::rational::Ratio;

use super::{Lattice, ShortVectorError};

/// The lattice `L(-1)`: the same underlying abelian group as `L`, with the
/// bilinear form negated. Used to build indefinite lattices out of
/// positive-definite ones, e.g. the K3 lattice
/// `U ⊕ U ⊕ U ⊕ E8(-1) ⊕ E8(-1)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NegatedLattice<L> {
    pub inner: L,
}

impl<L> NegatedLattice<L> {
    #[must_use = "The lattice point is now inside the sign-flipped lattice"]
    pub fn new(inner: L) -> Self {
        Self { inner }
    }
}

impl<L: Add<L, Output = L>> Add for NegatedLattice<L> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.inner + rhs.inner)
    }
}

impl<L: Sub<L, Output = L>> Sub for NegatedLattice<L> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.inner - rhs.inner)
    }
}

impl<L: AddAssign<L>> AddAssign for NegatedLattice<L> {
    fn add_assign(&mut self, rhs: Self) {
        self.inner += rhs.inner;
    }
}

impl<L: SubAssign<L>> SubAssign for NegatedLattice<L> {
    fn sub_assign(&mut self, rhs: Self) {
        self.inner -= rhs.inner;
    }
}

impl<L: Mul<i128, Output = L>> Mul<i128> for NegatedLattice<L> {
    type Output = Self;
    fn mul(self, rhs: i128) -> Self {
        Self::new(self.inner * rhs)
    }
}

impl<L: MulAssign<i128>> MulAssign<i128> for NegatedLattice<L> {
    fn mul_assign(&mut self, rhs: i128) {
        self.inner *= rhs;
    }
}

impl<L: Zero> Zero for NegatedLattice<L> {
    fn zero() -> Self {
        Self::new(L::zero())
    }

    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }
}

impl<const N: usize, L: Lattice<N>> Lattice<N> for NegatedLattice<L> {
    type DualLattice = NegatedLattice<L::DualLattice>;

    fn inner_product(&self, other: &Self) -> Ratio<i128> {
        -self.inner.inner_product(&other.inner)
    }

    fn is_integral() -> bool {
        L::is_integral()
    }

    fn is_even() -> bool {
        L::is_even()
    }

    // |det(-G)| = |(-1)^N det(G)| = |det(G)|, so discriminant (and hence
    // self-duality) is unaffected by negating the form.
    fn is_self_dual() -> bool {
        L::is_self_dual()
    }

    // Negating the form swaps the positive and negative eigenvalue counts.
    fn signature() -> (usize, usize, usize) {
        let (p, q, r) = L::signature();
        (q, p, r)
    }

    fn discriminant() -> Ratio<u128> {
        L::discriminant()
    }

    // If L itself is positive definite, L(-1) is negative definite (not
    // positive definite). The other direction, where L is negative
    // definite and L(-1) is positive definite, isn't computable from `L`'s
    // own `short_vectors` (which assumes *L* is the positive-definite one).
    fn short_vectors() -> Result<Vec<Self>, ShortVectorError> {
        if Self::signature() != (N, 0, 0) {
            return Err(ShortVectorError::NotPositiveDefinite);
        }
        Err(ShortVectorError::Unknown)
    }
}

#[cfg(test)]
mod tests {
    use num::rational::Ratio;

    use super::NegatedLattice;
    use crate::lattice::{Lattice, RootLatticeE8, ShortVectorError};

    #[test]
    fn negated_e8_is_even_unimodular_negative_definite() {
        type NegE8 = NegatedLattice<RootLatticeE8>;

        assert!(NegE8::is_integral());
        assert!(NegE8::is_even());
        assert!(NegE8::is_self_dual());
        assert_eq!(NegE8::signature(), (0, 8, 0));
        assert_eq!(NegE8::discriminant(), Ratio::from_integer(1));
        assert_eq!(
            NegE8::short_vectors(),
            Err(ShortVectorError::NotPositiveDefinite)
        );
    }

    #[test]
    fn inner_product_is_negated() {
        type NegE8 = NegatedLattice<RootLatticeE8>;
        let mut coeffs = [0i128; 8];
        coeffs[0] = 1;
        let a = NegE8::new(RootLatticeE8::new(coeffs));
        assert_eq!(a.lattice_norm_sq(), Ratio::from_integer(-2));
    }
}
