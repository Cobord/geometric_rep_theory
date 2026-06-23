use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use num::Zero;
use num::rational::Ratio;

use super::{Lattice, ShortVectorError};

/// An element of the hyperbolic plane lattice `U` (also written `II_{1,1}`):
/// the rank-2 lattice `Z^2` with Gram matrix `[[0, 1], [1, 0]]` in the
/// standard basis `e_1, e_2`. It is even, unimodular, and indefinite with
/// signature `(1, 1, 0)` — the basic building block of even unimodular
/// lattices such as the K3 lattice `U^3 ⊕ E8(-1)^2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct HyperbolicPlane {
    pub e1: i128,
    pub e2: i128,
}

impl HyperbolicPlane {
    #[must_use = "The coeffs are now inside the lattice point"]
    pub fn new(e1: i128, e2: i128) -> Self {
        Self { e1, e2 }
    }
}

impl Add for HyperbolicPlane {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.e1 + rhs.e1, self.e2 + rhs.e2)
    }
}

impl Sub for HyperbolicPlane {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.e1 - rhs.e1, self.e2 - rhs.e2)
    }
}

impl AddAssign for HyperbolicPlane {
    fn add_assign(&mut self, rhs: Self) {
        self.e1 += rhs.e1;
        self.e2 += rhs.e2;
    }
}

impl SubAssign for HyperbolicPlane {
    fn sub_assign(&mut self, rhs: Self) {
        self.e1 -= rhs.e1;
        self.e2 -= rhs.e2;
    }
}

impl Mul<i128> for HyperbolicPlane {
    type Output = Self;
    fn mul(self, rhs: i128) -> Self {
        Self::new(self.e1 * rhs, self.e2 * rhs)
    }
}

impl MulAssign<i128> for HyperbolicPlane {
    fn mul_assign(&mut self, rhs: i128) {
        self.e1 *= rhs;
        self.e2 *= rhs;
    }
}

impl Zero for HyperbolicPlane {
    fn zero() -> Self {
        Self::new(0, 0)
    }

    fn is_zero(&self) -> bool {
        self.e1 == 0 && self.e2 == 0
    }
}

impl Lattice<2> for HyperbolicPlane {
    // U is unimodular, so the form `v -> <v, ->` identifies U with its own
    // dual.
    type DualLattice = Self;

    fn inner_product(&self, other: &Self) -> Ratio<i128> {
        Ratio::from_integer(self.e1 * other.e2 + self.e2 * other.e1)
    }

    fn is_integral() -> bool {
        true
    }

    fn is_even() -> bool {
        true
    }

    fn is_self_dual() -> bool {
        true
    }

    fn signature() -> (usize, usize, usize) {
        (1, 1, 0)
    }

    fn discriminant() -> Ratio<u128> {
        Ratio::from_integer(1)
    }

    // Indefinite: every isotropic vector (n, 0) and (0, n) has norm 0 for
    // any nonzero integer n, so there is no finite minimal-norm set.
    fn short_vectors() -> Result<Vec<Self>, ShortVectorError> {
        Err(ShortVectorError::NotPositiveDefinite)
    }
}

#[cfg(test)]
mod tests {
    use num::Zero;
    use num::rational::Ratio;

    use super::{HyperbolicPlane, Lattice};

    #[test]
    fn basis_vectors_are_isotropic() {
        let e1 = HyperbolicPlane::new(1, 0);
        let e2 = HyperbolicPlane::new(0, 1);
        assert_eq!(e1.lattice_norm_sq(), Ratio::zero());
        assert_eq!(e2.lattice_norm_sq(), Ratio::zero());
        assert_eq!(e1.inner_product(&e2), Ratio::from_integer(1));
    }

    #[test]
    fn norm_is_always_even() {
        let v = HyperbolicPlane::new(3, -5);
        // <v, v> = 2 * e1 * e2 = -30
        assert_eq!(v.lattice_norm_sq(), Ratio::from_integer(-30));
    }

    #[test]
    fn invariants() {
        assert!(HyperbolicPlane::is_integral());
        assert!(HyperbolicPlane::is_even());
        assert!(HyperbolicPlane::is_self_dual());
        assert_eq!(HyperbolicPlane::signature(), (1, 1, 0));
        assert_eq!(HyperbolicPlane::discriminant(), Ratio::from_integer(1));
    }

    #[test]
    fn short_vectors_undefined_for_indefinite_lattice() {
        assert_eq!(
            HyperbolicPlane::short_vectors(),
            Err(super::ShortVectorError::NotPositiveDefinite)
        );
    }

    #[test]
    fn group_structure() {
        let mut v = HyperbolicPlane::new(2, 3);
        v += HyperbolicPlane::new(1, -1);
        assert_eq!(v, HyperbolicPlane::new(3, 2));
        v -= HyperbolicPlane::new(3, 2);
        assert!(v.is_zero());

        let w = HyperbolicPlane::new(1, 1) * 4;
        assert_eq!(w, HyperbolicPlane::new(4, 4));
    }
}
