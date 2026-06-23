use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use num::Zero;
use num::rational::Ratio;

use super::{Lattice, ShortVectorError};

/// `<alpha_i*, alpha_j*> = (C^{-1})_{ij}` for the inverse of the `A_N`
/// Cartan matrix, using the closed form `min(a,b)(N+1-max(a,b))/(N+1)` for
/// the 1-indexed simple roots `a = i+1`, `b = j+1`.
#[allow(clippy::many_single_char_names)]
fn dual_cartan_entry(n: usize, i: usize, j: usize) -> Ratio<i128> {
    let a = (i + 1) as i128;
    let b = (j + 1) as i128;
    let n_plus_one = (n + 1) as i128;
    Ratio::new(a.min(b) * (n_plus_one - a.max(b)), n_plus_one)
}

/// An element of the `A_N` root lattice (rank `N`), expressed as integer
/// coefficients in the basis of simple roots `alpha_1, ..., alpha_N`, with
/// Gram matrix the `A_N` Cartan matrix. `A_N` is even but, for `N >= 1`, not
/// self-dual: its discriminant is `N + 1`. See [`DualRootLatticeA`] for the
/// dual lattice `A_N*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootLatticeA<const N: usize> {
    pub coeffs: [i128; N],
}

impl<const N: usize> RootLatticeA<N> {
    #[must_use = "The coeffs are now inside the lattice point"]
    pub fn new(coeffs: [i128; N]) -> Self {
        Self { coeffs }
    }
}

impl<const N: usize> Add for RootLatticeA<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] + rhs.coeffs[i]))
    }
}

impl<const N: usize> Sub for RootLatticeA<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] - rhs.coeffs[i]))
    }
}

impl<const N: usize> AddAssign for RootLatticeA<N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coeffs[i] += rhs.coeffs[i];
        }
    }
}

impl<const N: usize> SubAssign for RootLatticeA<N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coeffs[i] -= rhs.coeffs[i];
        }
    }
}

impl<const N: usize> Mul<i128> for RootLatticeA<N> {
    type Output = Self;
    fn mul(self, rhs: i128) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] * rhs))
    }
}

impl<const N: usize> MulAssign<i128> for RootLatticeA<N> {
    fn mul_assign(&mut self, rhs: i128) {
        for i in 0..N {
            self.coeffs[i] *= rhs;
        }
    }
}

impl<const N: usize> Zero for RootLatticeA<N> {
    fn zero() -> Self {
        Self::new([0; N])
    }

    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| *c == 0)
    }
}

impl<const N: usize> Lattice<N> for RootLatticeA<N> {
    type DualLattice = DualRootLatticeA<N>;

    fn inner_product(&self, other: &Self) -> Ratio<i128> {
        let mut total = 0i128;
        for i in 0..N {
            total += 2 * self.coeffs[i] * other.coeffs[i];
        }
        for i in 0..N.saturating_sub(1) {
            total -= self.coeffs[i] * other.coeffs[i + 1] + self.coeffs[i + 1] * other.coeffs[i];
        }
        Ratio::from_integer(total)
    }

    fn is_integral() -> bool {
        true
    }

    // <x, x> = 2 * (sum_i x_i^2 - sum_i x_i x_{i+1}), always even.
    fn is_even() -> bool {
        true
    }

    fn is_self_dual() -> bool {
        N == 0
    }

    // The Cartan matrix of a finite-type root system is positive definite.
    fn signature() -> (usize, usize, usize) {
        (N, 0, 0)
    }

    // det(Cartan matrix of A_N) = N + 1.
    fn discriminant() -> Ratio<u128> {
        Ratio::from_integer(N as u128 + 1)
    }

    // The roots are alpha_i + alpha_{i+1} + ... + alpha_j (coefficient 1 on
    // positions i..=j, else 0) and their negatives, for 0 <= i <= j < N;
    // N(N+1) roots in total, each of norm 2.
    fn short_vectors() -> Result<Vec<Self>, ShortVectorError> {
        let mut roots = Vec::with_capacity(N * (N + 1));
        for i in 0..N {
            for j in i..N {
                let mut coeffs = [0i128; N];
                coeffs[i..=j].fill(1);
                let root = Self::new(coeffs);
                roots.push(root);
                roots.push(root * -1);
            }
        }
        Ok(roots)
    }
}

/// An element of the dual lattice `A_N*`, expressed as integer coefficients
/// in the dual basis `alpha_1*, ..., alpha_N*` (where `<alpha_i*, alpha_j>
/// = delta_ij`). Its Gram matrix is the inverse Cartan matrix, with
/// `discriminant() == 1/(N+1)`; it is integral only in the trivial case
/// `N == 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DualRootLatticeA<const N: usize> {
    pub coeffs: [i128; N],
}

impl<const N: usize> DualRootLatticeA<N> {
    #[must_use = "The coeffs are now inside the lattice point"]
    pub fn new(coeffs: [i128; N]) -> Self {
        Self { coeffs }
    }
}

impl<const N: usize> Add for DualRootLatticeA<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] + rhs.coeffs[i]))
    }
}

impl<const N: usize> Sub for DualRootLatticeA<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] - rhs.coeffs[i]))
    }
}

impl<const N: usize> AddAssign for DualRootLatticeA<N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coeffs[i] += rhs.coeffs[i];
        }
    }
}

impl<const N: usize> SubAssign for DualRootLatticeA<N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coeffs[i] -= rhs.coeffs[i];
        }
    }
}

impl<const N: usize> Mul<i128> for DualRootLatticeA<N> {
    type Output = Self;
    fn mul(self, rhs: i128) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] * rhs))
    }
}

impl<const N: usize> MulAssign<i128> for DualRootLatticeA<N> {
    fn mul_assign(&mut self, rhs: i128) {
        for i in 0..N {
            self.coeffs[i] *= rhs;
        }
    }
}

impl<const N: usize> Zero for DualRootLatticeA<N> {
    fn zero() -> Self {
        Self::new([0; N])
    }

    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| *c == 0)
    }
}

impl<const N: usize> Lattice<N> for DualRootLatticeA<N> {
    type DualLattice = RootLatticeA<N>;

    fn inner_product(&self, other: &Self) -> Ratio<i128> {
        let mut total = Ratio::from_integer(0);
        for i in 0..N {
            for j in 0..N {
                total += dual_cartan_entry(N, i, j)
                    * Ratio::from_integer(self.coeffs[i] * other.coeffs[j]);
            }
        }
        total
    }

    fn is_integral() -> bool {
        N == 0
    }

    fn is_even() -> bool {
        N == 0
    }

    fn is_self_dual() -> bool {
        N == 0
    }

    // The inverse of a positive-definite matrix is positive definite.
    fn signature() -> (usize, usize, usize) {
        (N, 0, 0)
    }

    fn discriminant() -> Ratio<u128> {
        Ratio::new(1, N as u128 + 1)
    }

    // In the standard embedding A_N* = { P(z) : z in Z^{N+1} }, where P is
    // orthogonal projection onto the sum-zero hyperplane: the minimal
    // vectors are exactly +-P(e_i) for i = 1..=N+1, each of norm N/(N+1).
    // Since alpha_k* = P(e_1 + ... + e_k), this gives
    // P(e_i) = alpha_i* - alpha_{i-1}* (with alpha_0* = alpha_{N+1}* = 0).
    // Deduplicated since for N == 1 this set collapses to a single +-pair.
    fn short_vectors() -> Result<Vec<Self>, ShortVectorError> {
        if N == 0 {
            return Ok(Vec::new());
        }
        let mut vectors: Vec<Self> = Vec::with_capacity(2 * (N + 1));
        for i in 1..=(N + 1) {
            let mut coeffs = [0i128; N];
            if i <= N {
                coeffs[i - 1] += 1;
            }
            if i >= 2 {
                coeffs[i - 2] -= 1;
            }
            let v = Self::new(coeffs);
            if !vectors.contains(&v) {
                vectors.push(v);
            }
            let neg = v * -1;
            if !vectors.contains(&neg) {
                vectors.push(neg);
            }
        }
        Ok(vectors)
    }
}

#[cfg(test)]
mod tests {
    use num::Zero;
    use num::rational::Ratio;

    use super::{DualRootLatticeA, Lattice, RootLatticeA};

    #[test]
    fn a2_simple_roots_have_norm_2_and_pair_to_minus_1() {
        let a1 = RootLatticeA::<2>::new([1, 0]);
        let a2 = RootLatticeA::<2>::new([0, 1]);
        assert_eq!(a1.lattice_norm_sq(), Ratio::from_integer(2));
        assert_eq!(a2.lattice_norm_sq(), Ratio::from_integer(2));
        assert_eq!(a1.inner_product(&a2), Ratio::from_integer(-1));
    }

    #[test]
    fn a_n_invariants() {
        assert!(RootLatticeA::<3>::is_integral());
        assert!(RootLatticeA::<3>::is_even());
        assert!(!RootLatticeA::<3>::is_self_dual());
        assert_eq!(RootLatticeA::<3>::signature(), (3, 0, 0));
        assert_eq!(RootLatticeA::<3>::discriminant(), Ratio::from_integer(4));
    }

    #[test]
    fn dual_a2_has_reciprocal_discriminant() {
        assert!(!DualRootLatticeA::<2>::is_integral());
        assert_eq!(DualRootLatticeA::<2>::signature(), (2, 0, 0));
        assert_eq!(DualRootLatticeA::<2>::discriminant(), Ratio::new(1, 3));

        // alpha_1* has norm (C^-1)_{00} = min(1,1)(3-1)/3 = 2/3
        let dual_e1 = DualRootLatticeA::<2>::new([1, 0]);
        assert_eq!(dual_e1.lattice_norm_sq(), Ratio::new(2, 3));
    }

    #[test]
    fn a2_discriminant_group_quadratic_matches_dual_norm() {
        let dual_e1 = DualRootLatticeA::<2>::new([1, 0]);
        assert_eq!(
            RootLatticeA::<2>::discriminant_group_quadratic(dual_e1),
            Ratio::new(2, 3)
        );

        // alpha_1 itself, expressed in the dual basis alpha_1*, alpha_2*,
        // is the first column of the A_2 Cartan matrix:
        // alpha_1 = 2*alpha_1* - alpha_2*. Its raw norm (14/3) differs from
        // alpha_1*'s (2/3) by the even integer 4, so after normalizing into
        // [0, 2) (A_2 is even) both representatives give the same class.
        let alpha_1_in_dual_coords = DualRootLatticeA::<2>::new([2, -1]);
        let shifted = dual_e1 + alpha_1_in_dual_coords;
        assert_eq!(shifted.lattice_norm_sq(), Ratio::new(14, 3));
        assert_eq!(
            RootLatticeA::<2>::discriminant_group_quadratic(shifted),
            RootLatticeA::<2>::discriminant_group_quadratic(dual_e1)
        );
    }

    #[test]
    fn a_n_roots() {
        let roots = RootLatticeA::<3>::short_vectors().unwrap();
        assert_eq!(roots.len(), 3 * 4);
        for r in &roots {
            assert_eq!(r.lattice_norm_sq(), Ratio::from_integer(2));
        }
        assert!(roots.contains(&RootLatticeA::<3>::new([1, 0, 0])));
        assert!(roots.contains(&RootLatticeA::<3>::new([0, -1, -1])));
    }

    #[test]
    fn dual_a_n_minimal_vectors() {
        for n in 1..=4 {
            match n {
                1 => check_dual_minimal_vectors::<1>(),
                2 => check_dual_minimal_vectors::<2>(),
                3 => check_dual_minimal_vectors::<3>(),
                4 => check_dual_minimal_vectors::<4>(),
                _ => unreachable!(),
            }
        }
    }

    fn check_dual_minimal_vectors<const N: usize>() {
        let vectors = DualRootLatticeA::<N>::short_vectors().unwrap();
        let expected_len = if N == 1 { 2 } else { 2 * (N + 1) };
        assert_eq!(vectors.len(), expected_len);
        let expected_norm = Ratio::new(N as i128, N as i128 + 1);
        for v in &vectors {
            assert_eq!(v.lattice_norm_sq(), expected_norm);
        }
    }

    #[test]
    fn group_structure() {
        let mut v = RootLatticeA::<3>::new([1, 2, 3]);
        v += RootLatticeA::<3>::new([-1, 0, 1]);
        assert_eq!(v, RootLatticeA::<3>::new([0, 2, 4]));
        v -= RootLatticeA::<3>::new([0, 2, 4]);
        assert!(v.is_zero());

        let w = RootLatticeA::<3>::new([1, -1, 2]) * 3;
        assert_eq!(w, RootLatticeA::<3>::new([3, -3, 6]));
    }
}
