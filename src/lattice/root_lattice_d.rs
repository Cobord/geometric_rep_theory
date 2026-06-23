use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use num::Zero;
use num::rational::Ratio;

use super::{Lattice, ShortVectorError};

/// An element of the `D_N` root lattice (rank `N`), expressed as integer
/// coefficients in the basis of simple roots `alpha_1, ..., alpha_N`, where
/// `alpha_i = e_i - e_{i+1}` for `i < N` and `alpha_N = e_{N-1} + e_N` (the
/// "fork" root), matching the standard embedding
/// `D_N = { x in Z^N : sum x_i even }`. Even, positive-definite, with
/// discriminant `4` for `N >= 1` — so self-dual only in the trivial case
/// `N == 0`. See [`DualRootLatticeD`] for the dual lattice `D_N*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootLatticeD<const N: usize> {
    pub coeffs: [i128; N],
}

impl<const N: usize> RootLatticeD<N> {
    #[must_use = "The coeffs are now inside the lattice point"]
    pub fn new(coeffs: [i128; N]) -> Self {
        Self { coeffs }
    }
}

impl<const N: usize> Add for RootLatticeD<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] + rhs.coeffs[i]))
    }
}

impl<const N: usize> Sub for RootLatticeD<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] - rhs.coeffs[i]))
    }
}

impl<const N: usize> AddAssign for RootLatticeD<N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coeffs[i] += rhs.coeffs[i];
        }
    }
}

impl<const N: usize> SubAssign for RootLatticeD<N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coeffs[i] -= rhs.coeffs[i];
        }
    }
}

impl<const N: usize> Mul<i128> for RootLatticeD<N> {
    type Output = Self;
    fn mul(self, rhs: i128) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] * rhs))
    }
}

impl<const N: usize> MulAssign<i128> for RootLatticeD<N> {
    fn mul_assign(&mut self, rhs: i128) {
        for i in 0..N {
            self.coeffs[i] *= rhs;
        }
    }
}

impl<const N: usize> Zero for RootLatticeD<N> {
    fn zero() -> Self {
        Self::new([0; N])
    }

    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| *c == 0)
    }
}

impl<const N: usize> Lattice<N> for RootLatticeD<N> {
    type DualLattice = DualRootLatticeD<N>;

    // Diagonal contributes 2*sum x_i^2; the chain edges (k,k+1) for
    // k = 0..=N-3 and the fork edge (N-3,N-1) (for N >= 3) each contribute
    // -1, matching the D_N Cartan matrix (the Dynkin diagram is a path
    // 0-1-...-(N-2) with an extra branch from N-3 to N-1).
    fn inner_product(&self, other: &Self) -> Ratio<i128> {
        let mut total = 0i128;
        for i in 0..N {
            total += 2 * self.coeffs[i] * other.coeffs[i];
        }
        for k in 0..N.saturating_sub(2) {
            total -= self.coeffs[k] * other.coeffs[k + 1] + self.coeffs[k + 1] * other.coeffs[k];
        }
        if N >= 3 {
            let (a, b) = (N - 3, N - 1);
            total -= self.coeffs[a] * other.coeffs[b] + self.coeffs[b] * other.coeffs[a];
        }
        Ratio::from_integer(total)
    }

    fn is_integral() -> bool {
        true
    }

    // <x, x> = 2 * (sum_i x_i^2 - sum_chain x_k x_{k+1} - fork term), always even.
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

    // det(Cartan matrix of D_N) = 4 for N >= 1.
    fn discriminant() -> Ratio<u128> {
        if N == 0 {
            Ratio::from_integer(1)
        } else {
            Ratio::from_integer(4)
        }
    }

    // The roots are e_i - e_j and e_i + e_j (and their negatives) for
    // 0 <= i < j < N, in the standard embedding. Expressed in the simple
    // root coordinates used by `coeffs`:
    //   e_i - e_j = alpha_i + ... + alpha_{j-1}        (chain sum)
    //   e_i + e_j = alpha_i + ... + alpha_{j-1}
    //               + 2*(alpha_j + ... + alpha_{N-3})  [if j < N-1]
    //               + alpha_{N-2} + alpha_N             [if j < N-1]
    //   e_i + e_{N-1} = alpha_i + ... + alpha_{N-3} + alpha_N
    // (indices above are 1-indexed alphas; `coeffs` is 0-indexed).
    fn short_vectors() -> Result<Vec<Self>, ShortVectorError> {
        if N < 2 {
            return Ok(Vec::new());
        }
        let mut roots = Vec::with_capacity(2 * N * (N - 1));
        for i in 0..N {
            for j in (i + 1)..N {
                let mut minus = [0i128; N];
                minus[i..j].fill(1);
                let minus_root = Self::new(minus);
                roots.push(minus_root);
                roots.push(minus_root * -1);

                let mut plus = [0i128; N];
                if j == N - 1 {
                    plus[i..N - 2].fill(1);
                } else {
                    plus[i..j].fill(1);
                    plus[j..N - 2].fill(2);
                    plus[N - 2] = 1;
                }
                plus[N - 1] = 1;
                let plus_root = Self::new(plus);
                roots.push(plus_root);
                roots.push(plus_root * -1);
            }
        }
        Ok(roots)
    }
}

/// An element of the dual lattice `D_N*`, expressed as integer coefficients
/// in the basis `e_1, ..., e_{N-1}, v` where `v = (1/2, ..., 1/2)` is the
/// glue vector generating `D_N* = Z^N + Z*v` over the ambient `Z^N` of
/// `D_N`'s standard embedding. Discriminant `1/4` for all `N >= 1`; not
/// integral, since e.g. `<e_1, v> = 1/2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DualRootLatticeD<const N: usize> {
    pub coeffs: [i128; N],
}

impl<const N: usize> DualRootLatticeD<N> {
    #[must_use = "The coeffs are now inside the lattice point"]
    pub fn new(coeffs: [i128; N]) -> Self {
        Self { coeffs }
    }
}

impl<const N: usize> Add for DualRootLatticeD<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] + rhs.coeffs[i]))
    }
}

impl<const N: usize> Sub for DualRootLatticeD<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] - rhs.coeffs[i]))
    }
}

impl<const N: usize> AddAssign for DualRootLatticeD<N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coeffs[i] += rhs.coeffs[i];
        }
    }
}

impl<const N: usize> SubAssign for DualRootLatticeD<N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.coeffs[i] -= rhs.coeffs[i];
        }
    }
}

impl<const N: usize> Mul<i128> for DualRootLatticeD<N> {
    type Output = Self;
    fn mul(self, rhs: i128) -> Self {
        Self::new(core::array::from_fn(|i| self.coeffs[i] * rhs))
    }
}

impl<const N: usize> MulAssign<i128> for DualRootLatticeD<N> {
    fn mul_assign(&mut self, rhs: i128) {
        for i in 0..N {
            self.coeffs[i] *= rhs;
        }
    }
}

impl<const N: usize> Zero for DualRootLatticeD<N> {
    fn zero() -> Self {
        Self::new([0; N])
    }

    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|c| *c == 0)
    }
}

impl<const N: usize> Lattice<N> for DualRootLatticeD<N> {
    type DualLattice = RootLatticeD<N>;

    // Gram matrix in the basis e_1, ..., e_{N-1}, v (coeffs[N-1] is the
    // coefficient of v): <e_i, e_j> = delta_ij, <e_i, v> = 1/2, <v, v> =
    // N/4.
    fn inner_product(&self, other: &Self) -> Ratio<i128> {
        if N == 0 {
            return Ratio::zero();
        }
        let last = N - 1;
        let mut diag = 0i128;
        let mut border = 0i128;
        for k in 0..last {
            diag += self.coeffs[k] * other.coeffs[k];
            border += self.coeffs[k] * other.coeffs[last] + self.coeffs[last] * other.coeffs[k];
        }
        let last_term = self.coeffs[last] * other.coeffs[last];
        Ratio::from_integer(diag) + Ratio::new(border, 2) + Ratio::new((N as i128) * last_term, 4)
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

    // The dual of a positive-definite lattice is positive definite.
    fn signature() -> (usize, usize, usize) {
        (N, 0, 0)
    }

    // [D_N* : D_N] = disc(D_N) = 4, so disc(D_N*) = 1/4, for any N >= 1.
    fn discriminant() -> Ratio<u128> {
        if N == 0 {
            Ratio::from_integer(1)
        } else {
            Ratio::new(1, 4)
        }
    }

    fn short_vectors() -> Result<Vec<Self>, ShortVectorError> {
        Err(ShortVectorError::Unknown)
    }
}

#[cfg(test)]
mod tests {
    use num::Zero;
    use num::rational::Ratio;

    use super::{DualRootLatticeD, Lattice, RootLatticeD, ShortVectorError};

    #[test]
    fn d4_simple_roots_have_norm_2_and_central_node_is_alpha_2() {
        let a1 = RootLatticeD::<4>::new([1, 0, 0, 0]);
        let a2 = RootLatticeD::<4>::new([0, 1, 0, 0]);
        let a3 = RootLatticeD::<4>::new([0, 0, 1, 0]);
        let a4 = RootLatticeD::<4>::new([0, 0, 0, 1]);

        for a in [a1, a2, a3, a4] {
            assert_eq!(a.lattice_norm_sq(), Ratio::from_integer(2));
        }

        // alpha_2 (the triality center) connects to alpha_1, alpha_3, alpha_4.
        assert_eq!(a2.inner_product(&a1), Ratio::from_integer(-1));
        assert_eq!(a2.inner_product(&a3), Ratio::from_integer(-1));
        assert_eq!(a2.inner_product(&a4), Ratio::from_integer(-1));

        // The three leaves alpha_1, alpha_3, alpha_4 are pairwise orthogonal.
        assert_eq!(a1.inner_product(&a3), Ratio::from_integer(0));
        assert_eq!(a1.inner_product(&a4), Ratio::from_integer(0));
        assert_eq!(a3.inner_product(&a4), Ratio::from_integer(0));
    }

    #[test]
    fn d_n_invariants() {
        assert!(RootLatticeD::<5>::is_integral());
        assert!(RootLatticeD::<5>::is_even());
        assert!(!RootLatticeD::<5>::is_self_dual());
        assert_eq!(RootLatticeD::<5>::signature(), (5, 0, 0));
        assert_eq!(RootLatticeD::<5>::discriminant(), Ratio::from_integer(4));
    }

    #[test]
    fn d_n_roots() {
        for n in 2..=7 {
            match n {
                2 => check_d_n_roots::<2>(),
                3 => check_d_n_roots::<3>(),
                4 => check_d_n_roots::<4>(),
                5 => check_d_n_roots::<5>(),
                6 => check_d_n_roots::<6>(),
                7 => check_d_n_roots::<7>(),
                _ => unreachable!(),
            }
        }
    }

    fn check_d_n_roots<const N: usize>() {
        let roots = RootLatticeD::<N>::short_vectors().unwrap();
        assert_eq!(roots.len(), 2 * N * (N - 1));
        for r in &roots {
            assert_eq!(r.lattice_norm_sq(), Ratio::from_integer(2));
        }
        // No duplicates.
        for (idx, r) in roots.iter().enumerate() {
            assert!(!roots[idx + 1..].contains(r));
        }
    }

    #[test]
    fn d6_root_matches_known_coefficients() {
        // e_0 + e_1 in the embedding model, verified independently against
        // the inverse change-of-basis matrix.
        let roots = RootLatticeD::<6>::short_vectors().unwrap();
        assert!(roots.contains(&RootLatticeD::<6>::new([1, 2, 2, 2, 1, 1])));
        assert!(roots.contains(&RootLatticeD::<6>::new([-1, -2, -2, -2, -1, -1])));
        // e_0 - e_1
        assert!(roots.contains(&RootLatticeD::<6>::new([1, 0, 0, 0, 0, 0])));
    }

    #[test]
    fn dual_d4_glue_vector_structure() {
        assert!(!DualRootLatticeD::<4>::is_integral());
        assert_eq!(DualRootLatticeD::<4>::signature(), (4, 0, 0));
        assert_eq!(DualRootLatticeD::<4>::discriminant(), Ratio::new(1, 4));
        assert_eq!(
            DualRootLatticeD::<4>::short_vectors(),
            Err(ShortVectorError::Unknown)
        );

        let e1 = DualRootLatticeD::<4>::new([1, 0, 0, 0]);
        let v = DualRootLatticeD::<4>::new([0, 0, 0, 1]);
        assert_eq!(e1.lattice_norm_sq(), Ratio::from_integer(1));
        assert_eq!(v.lattice_norm_sq(), Ratio::from_integer(1)); // N/4 = 4/4
        assert_eq!(e1.inner_product(&v), Ratio::new(1, 2));
    }

    #[test]
    fn group_structure() {
        let mut v = RootLatticeD::<4>::new([1, 2, 3, 4]);
        v += RootLatticeD::<4>::new([-1, 0, 1, 0]);
        assert_eq!(v, RootLatticeD::<4>::new([0, 2, 4, 4]));
        v -= RootLatticeD::<4>::new([0, 2, 4, 4]);
        assert!(v.is_zero());

        let w = RootLatticeD::<4>::new([1, -1, 2, 0]) * 3;
        assert_eq!(w, RootLatticeD::<4>::new([3, -3, 6, 0]));
    }
}
