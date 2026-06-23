use std::collections::HashSet;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use num::Zero;
use num::rational::Ratio;

use super::{Lattice, ShortVectorError};

/// The Bourbaki-labelled `E8` Dynkin diagram: the chain
/// `1-3-4-5-6-7-8` with node `2` attached to node `4` (0-indexed below).
const EDGES: [(usize, usize); 7] = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 3)];

/// An element of the `E8` root lattice, expressed as integer coefficients
/// `[i128; 8]` in the basis of simple roots `alpha_1, ..., alpha_8`
/// (Bourbaki labeling), with Gram matrix the `E8` Cartan matrix. Even and
/// unimodular (discriminant `1`) — the unique such lattice of rank 8, so
/// `DualLattice = Self`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootLatticeE8 {
    pub coeffs: [i128; 8],
}

impl RootLatticeE8 {
    #[must_use = "The coeffs are now inside the lattice point"]
    pub fn new(coeffs: [i128; 8]) -> Self {
        Self { coeffs }
    }
}

impl Add for RootLatticeE8 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] + rhs.coeffs[i]))
    }
}

impl Sub for RootLatticeE8 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] - rhs.coeffs[i]))
    }
}

impl AddAssign for RootLatticeE8 {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..8 {
            self.coeffs[i] += rhs.coeffs[i];
        }
    }
}

impl SubAssign for RootLatticeE8 {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..8 {
            self.coeffs[i] -= rhs.coeffs[i];
        }
    }
}

impl Mul<i128> for RootLatticeE8 {
    type Output = Self;
    fn mul(self, rhs: i128) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] * rhs))
    }
}

impl MulAssign<i128> for RootLatticeE8 {
    fn mul_assign(&mut self, rhs: i128) {
        for i in 0..8 {
            self.coeffs[i] *= rhs;
        }
    }
}

impl Zero for RootLatticeE8 {
    fn zero() -> Self {
        Self::new([0; 8])
    }

    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| *c == 0)
    }
}

impl Lattice<8> for RootLatticeE8 {
    // E8 is unimodular, so the form `v -> <v, ->` identifies E8 with its
    // own dual.
    type DualLattice = Self;

    fn inner_product(&self, other: &Self) -> Ratio<i128> {
        let mut total = 0i128;
        for i in 0..8 {
            total += 2 * self.coeffs[i] * other.coeffs[i];
        }
        for &(a, b) in &EDGES {
            total -= self.coeffs[a] * other.coeffs[b] + self.coeffs[b] * other.coeffs[a];
        }
        Ratio::from_integer(total)
    }

    fn is_integral() -> bool {
        true
    }

    // <x, x> = 2 * (sum_i x_i^2 - sum_edges x_a x_b), always even.
    fn is_even() -> bool {
        true
    }

    fn is_self_dual() -> bool {
        true
    }

    // The Cartan matrix of a finite-type root system is positive definite.
    fn signature() -> (usize, usize, usize) {
        (8, 0, 0)
    }

    // det(Cartan matrix of E8) = 1.
    fn discriminant() -> Ratio<u128> {
        Ratio::from_integer(1)
    }

    // The Weyl group of a simply-laced root system acts transitively on the
    // roots, so the reflection closure of the simple roots (s_i(r) = r -
    // <r, alpha_i> * alpha_i, since every alpha_i has norm 2) is the full
    // set of 240 roots.
    fn short_vectors() -> Result<Vec<Self>, ShortVectorError> {
        let simple_roots: [Self; 8] = core::array::from_fn(|i| {
            let mut c = [0i128; 8];
            c[i] = 1;
            Self::new(c)
        });

        let mut seen: HashSet<[i128; 8]> = simple_roots.iter().map(|r| r.coeffs).collect();
        let mut frontier: Vec<Self> = simple_roots.to_vec();

        while let Some(r) = frontier.pop() {
            for alpha in &simple_roots {
                let pairing = r.inner_product(alpha).to_integer();
                let reflected = r - (*alpha * pairing);
                if seen.insert(reflected.coeffs) {
                    frontier.push(reflected);
                }
            }
        }

        Ok(seen.into_iter().map(Self::new).collect())
    }
}

#[cfg(test)]
mod tests {
    use num::Zero;
    use num::rational::Ratio;

    use super::{Lattice, RootLatticeE8};

    fn alpha(i: usize) -> RootLatticeE8 {
        let mut coeffs = [0i128; 8];
        coeffs[i] = 1;
        RootLatticeE8::new(coeffs)
    }

    #[test]
    fn simple_roots_have_norm_2_and_match_bourbaki_diagram() {
        for i in 0..8 {
            assert_eq!(alpha(i).lattice_norm_sq(), Ratio::from_integer(2));
        }

        // Edges (1-indexed labels): 1-3, 3-4, 4-5, 5-6, 6-7, 7-8, 2-4.
        let edges = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 3)];
        for &(a, b) in &edges {
            assert_eq!(alpha(a).inner_product(&alpha(b)), Ratio::from_integer(-1));
        }

        // A non-adjacent pair is orthogonal.
        assert_eq!(alpha(0).inner_product(&alpha(1)), Ratio::from_integer(0));
        assert_eq!(alpha(4).inner_product(&alpha(7)), Ratio::from_integer(0));
    }

    #[test]
    fn invariants() {
        assert!(RootLatticeE8::is_integral());
        assert!(RootLatticeE8::is_even());
        assert!(RootLatticeE8::is_self_dual());
        assert_eq!(RootLatticeE8::signature(), (8, 0, 0));
        assert_eq!(RootLatticeE8::discriminant(), Ratio::from_integer(1));
    }

    #[test]
    fn has_240_roots_all_norm_2_no_duplicates() {
        let roots = RootLatticeE8::short_vectors().unwrap();
        assert_eq!(roots.len(), 240);
        for r in &roots {
            assert_eq!(r.lattice_norm_sq(), Ratio::from_integer(2));
        }
        let unique: std::collections::HashSet<_> = roots.iter().map(|r| r.coeffs).collect();
        assert_eq!(unique.len(), 240);

        // Negation-closed.
        for r in &roots {
            assert!(roots.contains(&(*r * -1)));
        }

        // The simple roots themselves are among the 240.
        for i in 0..8 {
            assert!(roots.contains(&alpha(i)));
        }
    }

    #[test]
    fn group_structure() {
        let mut v = RootLatticeE8::new([1, 2, 3, 4, 5, 6, 7, 8]);
        v += RootLatticeE8::new([-1, 0, 1, 0, -1, 0, 1, 0]);
        assert_eq!(v, RootLatticeE8::new([0, 2, 4, 4, 4, 6, 8, 8]));
        v -= RootLatticeE8::new([0, 2, 4, 4, 4, 6, 8, 8]);
        assert!(v.is_zero());

        let w = RootLatticeE8::new([1, -1, 2, 0, 0, 0, 0, 0]) * 3;
        assert_eq!(w, RootLatticeE8::new([3, -3, 6, 0, 0, 0, 0, 0]));
    }
}
