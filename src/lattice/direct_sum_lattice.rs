use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use num::Zero;
use num::rational::Ratio;

use super::{Lattice, ShortVectorError};

/// The orthogonal direct sum `L1 ⊕ L2` of a rank-`N` lattice `L1` and a
/// rank-`M` lattice `L2` (no cross terms in the inner product), giving a
/// rank-`N_PLUS_M` lattice — e.g. the K3 lattice is
/// `U ⊕ U ⊕ U ⊕ E8(-1) ⊕ E8(-1)`.
///
/// Stable Rust has no `N + M` in a const-generic position, so the caller
/// must additionally specify `N_PLUS_M`; [`DirectSumLattice::new`] checks
/// at compile time that it actually equals `N + M`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectSumLattice<L1, L2, const N: usize, const M: usize, const N_PLUS_M: usize> {
    pub first: L1,
    pub second: L2,
}

impl<L1: Lattice<N>, L2: Lattice<M>, const N: usize, const M: usize, const N_PLUS_M: usize>
    DirectSumLattice<L1, L2, N, M, N_PLUS_M>
{
    const RANK_CHECK: () = assert!(
        N_PLUS_M == N + M,
        "N_PLUS_M must equal N + M for DirectSumLattice"
    );

    #[must_use = "The summands are now inside the direct sum"]
    pub fn new(first: L1, second: L2) -> Self {
        let () = Self::RANK_CHECK;
        Self { first, second }
    }
}

impl<L1: Lattice<N>, L2: Lattice<M>, const N: usize, const M: usize, const N_PLUS_M: usize> Add
    for DirectSumLattice<L1, L2, N, M, N_PLUS_M>
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.first + rhs.first, self.second + rhs.second)
    }
}

impl<L1: Lattice<N>, L2: Lattice<M>, const N: usize, const M: usize, const N_PLUS_M: usize> Sub
    for DirectSumLattice<L1, L2, N, M, N_PLUS_M>
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.first - rhs.first, self.second - rhs.second)
    }
}

impl<L1: Lattice<N>, L2: Lattice<M>, const N: usize, const M: usize, const N_PLUS_M: usize>
    AddAssign for DirectSumLattice<L1, L2, N, M, N_PLUS_M>
{
    fn add_assign(&mut self, rhs: Self) {
        self.first += rhs.first;
        self.second += rhs.second;
    }
}

impl<L1: Lattice<N>, L2: Lattice<M>, const N: usize, const M: usize, const N_PLUS_M: usize>
    SubAssign for DirectSumLattice<L1, L2, N, M, N_PLUS_M>
{
    fn sub_assign(&mut self, rhs: Self) {
        self.first -= rhs.first;
        self.second -= rhs.second;
    }
}

impl<L1: Lattice<N>, L2: Lattice<M>, const N: usize, const M: usize, const N_PLUS_M: usize>
    Mul<i128> for DirectSumLattice<L1, L2, N, M, N_PLUS_M>
{
    type Output = Self;
    fn mul(self, rhs: i128) -> Self {
        Self::new(self.first * rhs, self.second * rhs)
    }
}

impl<L1: Lattice<N>, L2: Lattice<M>, const N: usize, const M: usize, const N_PLUS_M: usize>
    MulAssign<i128> for DirectSumLattice<L1, L2, N, M, N_PLUS_M>
{
    fn mul_assign(&mut self, rhs: i128) {
        self.first *= rhs;
        self.second *= rhs;
    }
}

impl<L1: Lattice<N>, L2: Lattice<M>, const N: usize, const M: usize, const N_PLUS_M: usize> Zero
    for DirectSumLattice<L1, L2, N, M, N_PLUS_M>
{
    fn zero() -> Self {
        Self::new(L1::zero(), L2::zero())
    }

    fn is_zero(&self) -> bool {
        self.first.is_zero() && self.second.is_zero()
    }
}

impl<L1: Lattice<N>, L2: Lattice<M>, const N: usize, const M: usize, const N_PLUS_M: usize>
    Lattice<N_PLUS_M> for DirectSumLattice<L1, L2, N, M, N_PLUS_M>
{
    type DualLattice = DirectSumLattice<L1::DualLattice, L2::DualLattice, N, M, N_PLUS_M>;

    fn inner_product(&self, other: &Self) -> Ratio<i128> {
        self.first.inner_product(&other.first) + self.second.inner_product(&other.second)
    }

    fn is_integral() -> bool {
        L1::is_integral() && L2::is_integral()
    }

    fn is_even() -> bool {
        L1::is_even() && L2::is_even()
    }

    // discriminant() == 1 is the general criterion (Self::is_self_dual()
    // isn't simply L1::is_self_dual() && L2::is_self_dual(): two
    // non-integral rational summands could have reciprocal discriminants
    // without either being self-dual on its own).
    fn is_self_dual() -> bool {
        Self::discriminant() == Ratio::from_integer(1)
    }

    // Eigenvalues of a block-diagonal Gram matrix are the union of the
    // blocks' eigenvalues, so the signature counts simply add.
    fn signature() -> (usize, usize, usize) {
        let (p1, q1, r1) = L1::signature();
        let (p2, q2, r2) = L2::signature();
        (p1 + p2, q1 + q2, r1 + r2)
    }

    // det of a block-diagonal matrix is the product of the blocks' dets.
    fn discriminant() -> Ratio<u128> {
        L1::discriminant() * L2::discriminant()
    }

    // A minimal vector of L1 ⊕ L2 is either a minimal vector of L1 paired
    // with 0, or 0 paired with a minimal vector of L2, whichever factor
    // has the smaller minimal norm (any vector nonzero in both factors has
    // norm >= the sum of the two minimal norms, hence is never minimal).
    fn short_vectors() -> Result<Vec<Self>, ShortVectorError> {
        let v1 = L1::short_vectors()?;
        let v2 = L2::short_vectors()?;
        let n1 = v1.first().map(Lattice::lattice_norm_sq);
        let n2 = v2.first().map(Lattice::lattice_norm_sq);

        let lift_first = |v1: Vec<L1>| -> Vec<Self> {
            v1.into_iter().map(|a| Self::new(a, L2::zero())).collect()
        };
        let lift_second = |v2: Vec<L2>| -> Vec<Self> {
            v2.into_iter().map(|b| Self::new(L1::zero(), b)).collect()
        };

        match (n1, n2) {
            (None, None) => Ok(Vec::new()),
            (Some(_), None) => Ok(lift_first(v1)),
            (None, Some(_)) => Ok(lift_second(v2)),
            (Some(norm1), Some(norm2)) =>
            {
                #[allow(clippy::comparison_chain)]
                if norm1 < norm2 {
                    Ok(lift_first(v1))
                } else if norm2 < norm1 {
                    Ok(lift_second(v2))
                } else {
                    let mut result = lift_first(v1);
                    result.extend(lift_second(v2));
                    Ok(result)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use num::Zero;
    use num::rational::Ratio;

    use super::DirectSumLattice;
    use crate::lattice::{HyperbolicPlane, Lattice, RootLatticeA, RootLatticeE8};

    #[test]
    fn u_plus_e8_is_even_unimodular_indefinite() {
        type UPlusE8 = DirectSumLattice<HyperbolicPlane, RootLatticeE8, 2, 8, 10>;

        assert!(UPlusE8::is_integral());
        assert!(UPlusE8::is_even());
        assert!(UPlusE8::is_self_dual());
        assert_eq!(UPlusE8::signature(), (9, 1, 0));
        assert_eq!(UPlusE8::discriminant(), Ratio::from_integer(1));
    }

    #[test]
    fn a1_plus_a2_signature_and_discriminant_multiply() {
        // A_1 has discriminant 2, A_2 has discriminant 3.
        type A1PlusA2 = DirectSumLattice<RootLatticeA<1>, RootLatticeA<2>, 1, 2, 3>;

        assert_eq!(A1PlusA2::signature(), (3, 0, 0));
        assert_eq!(A1PlusA2::discriminant(), Ratio::from_integer(6));
        assert!(!A1PlusA2::is_self_dual());
    }

    #[test]
    fn short_vectors_picks_the_smaller_minimal_norm_factor() {
        // A_1 has minimal norm 2 (1 root pair); A_2 also has minimal norm 2
        // (3 root pairs) -- equal minimal norms, so short_vectors should
        // include both factors' lifted roots.
        type A1PlusA2 = DirectSumLattice<RootLatticeA<1>, RootLatticeA<2>, 1, 2, 3>;
        let roots = A1PlusA2::short_vectors().unwrap();
        assert_eq!(roots.len(), 2 + 6);
        for r in &roots {
            assert_eq!(r.lattice_norm_sq(), Ratio::from_integer(2));
        }
    }

    #[test]
    fn inner_product_has_no_cross_terms() {
        type UPlusE8 = DirectSumLattice<HyperbolicPlane, RootLatticeE8, 2, 8, 10>;
        let a = UPlusE8::new(HyperbolicPlane::new(1, 2), RootLatticeE8::zero());
        let b = UPlusE8::new(
            HyperbolicPlane::zero(),
            RootLatticeE8::new([1, 0, 0, 0, 0, 0, 0, 0]),
        );
        assert_eq!(a.inner_product(&b), Ratio::from_integer(0));
    }

    #[test]
    fn group_structure() {
        type A1PlusA2 = DirectSumLattice<RootLatticeA<1>, RootLatticeA<2>, 1, 2, 3>;
        let mut v = A1PlusA2::new(RootLatticeA::<1>::new([1]), RootLatticeA::<2>::new([1, 1]));
        v += A1PlusA2::new(RootLatticeA::<1>::new([-1]), RootLatticeA::<2>::new([0, 1]));
        assert_eq!(
            v,
            A1PlusA2::new(RootLatticeA::<1>::new([0]), RootLatticeA::<2>::new([1, 2]))
        );
        v -= v;
        assert!(v.is_zero());
    }
}
