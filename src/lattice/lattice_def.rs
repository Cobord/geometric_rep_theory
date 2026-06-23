use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use num::Zero;
use num::rational::Ratio;

use crate::arithmetic_utils::kronecker_symbol;

/// Error returned by [`Lattice::short_vectors`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShortVectorError {
    /// The form is not positive definite, so there is no finite set of
    /// minimal-norm nonzero vectors (e.g. an indefinite lattice can scale
    /// an isotropic vector by any nonzero integer and stay at norm `0`).
    NotPositiveDefinite,
    /// The lattice is positive definite, but this implementation does not
    /// (yet) know how to enumerate its minimal-norm vectors.
    Unknown,
}

impl fmt::Display for ShortVectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotPositiveDefinite => write!(
                f,
                "short vectors are only well-defined for a positive-definite lattice"
            ),
            Self::Unknown => write!(
                f,
                "short vectors are not known for this lattice implementation"
            ),
        }
    }
}

impl std::error::Error for ShortVectorError {}

/// A rank-`N` lattice: a free `Z`-module isomorphic to `Z^N` as an abelian
/// group, equipped with a rational-valued symmetric bilinear form. This
/// covers not just integral lattices but also rational ones such as the
/// dual of an integral lattice, whose Gram matrix need not have integer
/// entries (e.g. `A_n*` has pairings in `(1/(n+1))Z`).
///
/// Scalar multiplication is restricted to `i128` — a lattice is not a vector
/// space, so there is no division and no action of arbitrary rationals or
/// reals.
pub trait Lattice<const N: usize>:
    Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + Mul<i128, Output = Self>
    + MulAssign<i128>
    + PartialEq
    + Zero
    + Sized
{
    /// The dual lattice `L* = Hom(L, Z)`, the lattice of `Z`-linear
    /// functionals on `L`, with its own induced lattice structure.
    type DualLattice: Lattice<N>;

    /// The symmetric bilinear form `<self, other>`.
    fn inner_product(&self, other: &Self) -> Ratio<i128>;

    /// The associated quadratic form, `<self, self>`.
    fn lattice_norm_sq(&self) -> Ratio<i128> {
        self.inner_product(self)
    }

    /// Whether `<v, w>` is an integer for every `v, w` in the lattice, i.e.
    /// whether this is an integral lattice rather than a properly rational
    /// one (such as the dual of an integral lattice).
    fn is_integral() -> bool;

    /// Whether `<v, v>` is an even integer for every `v` in the lattice
    /// (e.g. root lattices such as `E8`), as opposed to merely an integer
    /// for every `v` (in which case the lattice is odd). Implies
    /// [`is_integral`](Self::is_integral).
    fn is_even() -> bool;

    /// Whether the lattice is unimodular, i.e. equal to its own
    /// [`DualLattice`](Self::DualLattice) under the inner product
    /// (equivalently, [`discriminant`](Self::discriminant) is `1`).
    fn is_self_dual() -> bool;

    /// The signature `(p, q, r)`: the number of positive, negative, and zero
    /// eigenvalues of the Gram matrix of `inner_product`, counted with
    /// multiplicity, so `p + q + r = N` always. `r` is the dimension of the
    /// radical; the form is non-degenerate iff `r == 0`.
    fn signature() -> (usize, usize, usize);

    /// The absolute value of the determinant of the Gram matrix of
    /// `inner_product`, `|det(G)|`. This is independent of the choice of
    /// basis, since a `GL_N(Z)` change of basis scales `det(G)` by
    /// `det(A)^2 = 1`. It is `0` exactly when the form is degenerate
    /// (`r > 0` in [`signature`](Self::signature)), and need not be an
    /// integer for a non-integral rational lattice.
    fn discriminant() -> Ratio<u128>;

    /// The nonzero vectors of minimal norm, e.g. the roots of a root
    /// lattice.
    ///
    /// # Errors
    ///
    /// Returns [`ShortVectorError::NotPositiveDefinite`] if
    /// `signature() != (N, 0, 0)`, since an indefinite or degenerate form
    /// need not have a finite set of minimal-norm vectors. May return
    /// [`ShortVectorError::Unknown`] for a positive-definite lattice whose
    /// minimal vectors this implementation does not compute.
    fn short_vectors() -> Result<Vec<Self>, ShortVectorError>;

    /// The discriminant-group quadratic form, evaluated at a representative
    /// `x` of a class in the discriminant group `L*/L`.
    ///
    /// Rather than model the quotient `L*/L` itself, this takes a
    /// representative `x ∈ L*` directly: `<x, x>` for two representatives
    /// of the same class in `L*/L` differ by an integer (or by an even
    /// integer, if `L` is even). The result is normalized into that range —
    /// `[0, 1)` in general, or `[0, 2)` if [`is_even`](Self::is_even) — so
    /// equal classes always compare as equal `Ratio<i128>` values.
    fn discriminant_group_quadratic(in_disc_group: Self::DualLattice) -> Ratio<i128> {
        let raw = in_disc_group.lattice_norm_sq();
        let modulus = Ratio::from_integer(if Self::is_even() { 2 } else { 1 });
        raw - modulus * (raw / modulus).floor()
    }

    /// Sanity check from Milgram's formula: an even unimodular lattice must
    /// have signature `p - q ≡ 0 (mod 8)`. Vacuously `true` when the
    /// lattice isn't both even and self-dual, since that's the hypothesis
    /// this checks. O(1): unlike a full discriminant-form Gauss sum (which
    /// would apply to any even lattice, not just unimodular ones, but needs
    /// to enumerate all of `L*/L`), this only ever inspects `signature()`.
    #[must_use = "If this fails the implementation is broken"]
    fn satisfies_milgram() -> bool {
        if !(Self::is_even() && Self::is_self_dual()) {
            return true;
        }
        let (p, q, _) = Self::signature();
        (p as i128 - q as i128).rem_euclid(8) == 0
    }

    /// The weight of the theta series `Θ(τ) = Σ_{x ∈ L} q^{<x,x>/2}`: the
    /// exponent `N/2` in its transformation law (see
    /// [`theta_series_character`](Self::theta_series_character)), an
    /// integer iff `N` is even.
    ///
    /// `Θ` is a genuine power series in `q` only when `L` is even (every
    /// exponent `<x,x>/2` is then an integer). Otherwise it's still a
    /// perfectly good holomorphic function of `τ` — just one expanded in a
    /// smaller nome: in `q^{1/2}` if `L` is integral but odd (some
    /// exponents are half-integers), or in `q^{1/(2s)}` for a properly
    /// rational `L` whose pairings have denominator `s` (worse still). None
    /// of that is an error condition here, and doesn't require enumerating
    /// any vectors, unlike the theta series itself; it only affects which
    /// multiplier system `Θ` transforms under, which this method doesn't
    /// otherwise track.
    ///
    /// # Errors
    ///
    /// Returns [`ShortVectorError::NotPositiveDefinite`] if
    /// `signature() != (N, 0, 0)`: the sum then diverges (or is a
    /// non-holomorphic Siegel theta function rather than a modular form of
    /// weight `N/2`), since e.g. an isotropic vector contributes `q^0`
    /// infinitely often along an indefinite direction.
    fn theta_series_weight() -> Result<Ratio<i128>, ShortVectorError> {
        if Self::signature() != (N, 0, 0) {
            return Err(ShortVectorError::NotPositiveDefinite);
        }
        Ok(Ratio::new(N as i128, 2))
    }

    /// `χ(d)` in the theta series' transformation law: for a
    /// positive-definite, integral lattice of even rank `N`, and any
    /// `g = (a b; c d) ∈ Gamma_0(M)` (`M` the lattice's level — not
    /// computed here),
    ///
    /// `Θ(gτ) = Θ((aτ + b)/(cτ + d)) = χ(d) · (cτ + d)^{N/2} · Θ(τ)`,
    ///
    /// where `Θ(τ) = Σ_{x∈L} q^{<x,x>/2}` and `χ(d) = (D|d)` is the
    /// Kronecker symbol of `D = (-1)^{N/2} * disc(L)`. Returns `Ok(None)`
    /// if this doesn't apply: `N` is odd (half-integral weight needs a
    /// different multiplier theory), or `L` isn't integral (a properly
    /// rational `L` needs an even finer multiplier system, per
    /// [`theta_series_weight`](Self::theta_series_weight)). `L` being
    /// integral but odd is *not* excluded here — the formula only needs
    /// integrality, not evenness, of `L` itself.
    ///
    /// # Errors
    ///
    /// Propagates [`ShortVectorError::NotPositiveDefinite`] from
    /// [`theta_series_weight`](Self::theta_series_weight).
    fn theta_series_character(d: i128) -> Result<Option<i128>, ShortVectorError> {
        Self::theta_series_weight()?;
        if !N.is_multiple_of(2) || !Self::is_integral() {
            return Ok(None);
        }
        let sign: i128 = if (N / 2).is_multiple_of(2) { 1 } else { -1 };
        #[allow(clippy::cast_possible_wrap)]
        let discriminant = sign * Self::discriminant().to_integer() as i128;
        Ok(Some(kronecker_symbol(discriminant, d)))
    }
}

#[cfg(test)]
mod tests {
    use num::rational::Ratio;

    use super::{Lattice, ShortVectorError};
    use crate::lattice::{
        HyperbolicPlane, NegatedLattice, RootLatticeA, RootLatticeE8, StandardLattice,
    };

    #[test]
    fn e8_theta_series_is_weight_4_trivial_character() {
        // Matches the classical fact that E8's theta series is the weight-4
        // level-1 Eisenstein series E4 (trivial character, since disc = 1).
        assert_eq!(
            RootLatticeE8::theta_series_weight(),
            Ok(Ratio::from_integer(4))
        );
        // Trivial character: kronecker_symbol(1, d) = 1 for any d.
        assert_eq!(RootLatticeE8::theta_series_character(7), Ok(Some(1)));
        assert_eq!(RootLatticeE8::theta_series_character(2), Ok(Some(1)));
    }

    #[test]
    fn a2_theta_series_character_matches_q_sqrt_minus_3_splitting() {
        // A_2's theta series is twisted by the discriminant -3 character,
        // matching the discriminant of Q(sqrt(-3)).
        assert_eq!(
            RootLatticeA::<2>::theta_series_weight(),
            Ok(Ratio::from_integer(1))
        );

        // 7 = 1 (mod 3) splits in Q(sqrt(-3)); 5 = 2 (mod 3) is inert.
        assert_eq!(RootLatticeA::<2>::theta_series_character(7), Ok(Some(1)));
        assert_eq!(RootLatticeA::<2>::theta_series_character(5), Ok(Some(-1)));
    }

    #[test]
    fn integral_but_odd_lattice_still_has_a_character() {
        assert!(StandardLattice::<4>::is_integral());
        assert!(!StandardLattice::<4>::is_even());
        assert_eq!(
            StandardLattice::<4>::theta_series_weight(),
            Ok(Ratio::from_integer(2))
        );
        // sign = (-1)^(4/2) = 1, disc = 1, so the discriminant is 1 and the
        // character is trivial — Some(1), not None.
        assert_eq!(StandardLattice::<4>::theta_series_character(7), Ok(Some(1)));
    }

    #[test]
    fn odd_rank_has_no_integral_character() {
        // A_3 (rank 3, odd) is still positive definite, so the weight 3/2
        // is well-defined, but the integer-weight character doesn't apply.
        assert_eq!(
            RootLatticeA::<3>::theta_series_weight(),
            Ok(Ratio::new(3, 2))
        );
        assert_eq!(RootLatticeA::<3>::theta_series_character(7), Ok(None));
    }

    #[test]
    fn indefinite_lattice_has_divergent_theta_series() {
        assert_eq!(
            HyperbolicPlane::theta_series_weight(),
            Err(ShortVectorError::NotPositiveDefinite)
        );
        assert_eq!(
            HyperbolicPlane::theta_series_character(7),
            Err(ShortVectorError::NotPositiveDefinite)
        );
    }

    #[test]
    fn even_unimodular_lattices_satisfy_milgram() {
        assert!(HyperbolicPlane::satisfies_milgram()); // p - q = 0
        assert!(RootLatticeE8::satisfies_milgram()); // p - q = 8
        assert!(NegatedLattice::<RootLatticeE8>::satisfies_milgram()); // p - q = -8
    }

    #[test]
    fn non_unimodular_even_lattice_is_vacuously_true() {
        // A_3 is even but not self-dual, so the check doesn't apply.
        assert!(!RootLatticeA::<3>::is_self_dual());
        assert!(RootLatticeA::<3>::satisfies_milgram());
    }

    /// A deliberately wrong rank-1 lattice (claims even + self-dual with
    /// signature `(1, 0, 0)`, i.e. `p - q = 1`) to confirm
    /// `satisfies_milgram` actually detects a violation rather than always
    /// passing vacuously.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct BrokenLattice(i128);

    impl std::ops::Add for BrokenLattice {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self(self.0 + rhs.0)
        }
    }
    impl std::ops::Sub for BrokenLattice {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self(self.0 - rhs.0)
        }
    }
    impl std::ops::AddAssign for BrokenLattice {
        fn add_assign(&mut self, rhs: Self) {
            self.0 += rhs.0;
        }
    }
    impl std::ops::SubAssign for BrokenLattice {
        fn sub_assign(&mut self, rhs: Self) {
            self.0 -= rhs.0;
        }
    }
    impl std::ops::Mul<i128> for BrokenLattice {
        type Output = Self;
        fn mul(self, rhs: i128) -> Self {
            Self(self.0 * rhs)
        }
    }
    impl std::ops::MulAssign<i128> for BrokenLattice {
        fn mul_assign(&mut self, rhs: i128) {
            self.0 *= rhs;
        }
    }
    impl num::Zero for BrokenLattice {
        fn zero() -> Self {
            Self(0)
        }
        fn is_zero(&self) -> bool {
            self.0 == 0
        }
    }
    impl Lattice<1> for BrokenLattice {
        type DualLattice = Self;
        fn inner_product(&self, other: &Self) -> num::rational::Ratio<i128> {
            num::rational::Ratio::from_integer(self.0 * other.0)
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
            (1, 0, 0)
        }
        fn discriminant() -> num::rational::Ratio<u128> {
            num::rational::Ratio::from_integer(1)
        }
        fn short_vectors() -> Result<Vec<Self>, ShortVectorError> {
            Err(ShortVectorError::Unknown)
        }
    }

    #[test]
    fn detects_a_genuine_milgram_violation() {
        assert!(!BrokenLattice::satisfies_milgram());
    }
}
