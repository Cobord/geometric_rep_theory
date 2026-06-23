use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use num::Zero;
use num::rational::Ratio;

use super::{Lattice, ShortVectorError};

/// An element of the standard lattice `Z^N`, expressed as integer
/// coefficients `[i128; N]` in the standard basis `e_1, ..., e_N`, with
/// Gram matrix the identity (the usual dot product). Integral, unimodular,
/// but odd (`<e_1, e_1> = 1`) — the simplest example of an integral
/// lattice that is not even.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StandardLattice<const N: usize> {
    pub coeffs: [i128; N],
}

impl<const N: usize> StandardLattice<N> {
    #[must_use = "The coeffs are now inside the lattice point"]
    pub fn new(coeffs: [i128; N]) -> Self {
        Self { coeffs }
    }
}

impl<const N: usize> Add for StandardLattice<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] + rhs.coeffs[i]))
    }
}

impl<const N: usize> Sub for StandardLattice<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] - rhs.coeffs[i]))
    }
}

impl<const N: usize> AddAssign for StandardLattice<N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coeffs[i] += rhs.coeffs[i];
        }
    }
}

impl<const N: usize> SubAssign for StandardLattice<N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coeffs[i] -= rhs.coeffs[i];
        }
    }
}

impl<const N: usize> Mul<i128> for StandardLattice<N> {
    type Output = Self;
    fn mul(self, rhs: i128) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] * rhs))
    }
}

impl<const N: usize> MulAssign<i128> for StandardLattice<N> {
    fn mul_assign(&mut self, rhs: i128) {
        for i in 0..N {
            self.coeffs[i] *= rhs;
        }
    }
}

impl<const N: usize> Zero for StandardLattice<N> {
    fn zero() -> Self {
        Self::new([0; N])
    }

    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| *c == 0)
    }
}

impl<const N: usize> Lattice<N> for StandardLattice<N> {
    // Z^N is unimodular, so the form `v -> <v, ->` identifies it with its
    // own dual.
    type DualLattice = Self;

    fn inner_product(&self, other: &Self) -> Ratio<i128> {
        Ratio::from_integer((0..N).map(|i| self.coeffs[i] * other.coeffs[i]).sum())
    }

    fn is_integral() -> bool {
        true
    }

    fn is_even() -> bool {
        false
    }

    fn is_self_dual() -> bool {
        true
    }

    fn signature() -> (usize, usize, usize) {
        (N, 0, 0)
    }

    fn discriminant() -> Ratio<u128> {
        Ratio::from_integer(1)
    }

    // The minimal vectors are the 2N standard basis vectors +-e_i, each of
    // norm 1.
    fn short_vectors() -> Result<Vec<Self>, ShortVectorError> {
        let mut roots = Vec::with_capacity(2 * N);
        for i in 0..N {
            let mut coeffs = [0i128; N];
            coeffs[i] = 1;
            let e_i = Self::new(coeffs);
            roots.push(e_i);
            roots.push(e_i * -1);
        }
        Ok(roots)
    }
}

#[cfg(test)]
mod tests {
    use num::Zero;
    use num::rational::Ratio;

    use super::StandardLattice;
    use crate::lattice::Lattice;

    #[test]
    fn basis_vectors_have_norm_1_and_are_orthogonal() {
        let e1 = StandardLattice::<3>::new([1, 0, 0]);
        let e2 = StandardLattice::<3>::new([0, 1, 0]);
        assert_eq!(e1.lattice_norm_sq(), Ratio::from_integer(1));
        assert_eq!(e2.lattice_norm_sq(), Ratio::from_integer(1));
        assert_eq!(e1.inner_product(&e2), Ratio::from_integer(0));
    }

    #[test]
    fn invariants() {
        assert!(StandardLattice::<4>::is_integral());
        assert!(!StandardLattice::<4>::is_even());
        assert!(StandardLattice::<4>::is_self_dual());
        assert_eq!(StandardLattice::<4>::signature(), (4, 0, 0));
        assert_eq!(StandardLattice::<4>::discriminant(), Ratio::from_integer(1));
    }

    #[test]
    fn short_vectors_are_the_signed_basis_vectors() {
        let roots = StandardLattice::<4>::short_vectors().unwrap();
        assert_eq!(roots.len(), 8);
        for r in &roots {
            assert_eq!(r.lattice_norm_sq(), Ratio::from_integer(1));
        }
        assert!(roots.contains(&StandardLattice::<4>::new([1, 0, 0, 0])));
        assert!(roots.contains(&StandardLattice::<4>::new([-1, 0, 0, 0])));
    }

    #[test]
    fn group_structure() {
        let mut v = StandardLattice::<3>::new([1, 2, 3]);
        v += StandardLattice::<3>::new([-1, 0, 1]);
        assert_eq!(v, StandardLattice::<3>::new([0, 2, 4]));
        v -= StandardLattice::<3>::new([0, 2, 4]);
        assert!(v.is_zero());

        let w = StandardLattice::<3>::new([1, -1, 2]) * 3;
        assert_eq!(w, StandardLattice::<3>::new([3, -3, 6]));
    }
}
