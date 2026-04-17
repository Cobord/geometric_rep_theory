use std::ops::{Div, DivAssign};

use log::warn;
use num::Zero;
use num::integer::div_ceil;

#[allow(unused_imports)]
use crate::arithmetic_utils::mobius;
use crate::arithmetic_utils::{Ring, SemiRing};
use crate::plethystic::LambdaRing;
use crate::plethystic::symmetric_function::SymmetricFunction;

/// A semiring `S` with a descending filtration
/// of submonoids (under addition)
/// `S = S^{>=0} ⊇ S^{>=1} ⊇ S^{>=2} ⊇ …`
/// such that `S^{>=n} * S^{>=m} ⊆ S^{>=(n+m)}`
/// and `S^{>=n} + S^{>=m} ⊆ S^{>=(min(n,m))}`.
pub trait FilteredSemiRing: SemiRing {
    #[must_use = "A new element in the filtered semiring which is a n-th filtration of the input is returned"]
    /// For an element `x` in the semiring,
    /// returns an element `y` of the semiring such that
    /// if this were a ring, then `x-y` would be
    /// in the `S^{>=n+1}` sub Z-module.
    fn truncate_at(self, n: usize) -> Self;

    /// Returns true if the element is in the `S^{>=n+1}` submonoid.
    fn truncates_to_zero_at(&self, n: usize) -> bool;

    /// The maximal `n` such that `self ∈ S^{>=n}`.
    ///
    /// # Errors
    ///
    /// If `self` is the zero element and thus in all `S^{>=n}`
    /// In that case there is no maximal such `n`.
    #[allow(clippy::result_unit_err)]
    fn maximal_filtration(&self) -> Result<usize, ()> {
        if self.is_zero() {
            return Err(());
        }
        let mut n = 0;
        while self.truncates_to_zero_at(n) {
            n += 1;
        }
        Ok(n)
    }
}

impl<Coeffs: Ring> FilteredSemiRing for SymmetricFunction<Coeffs> {
    fn truncate_at(self, n: usize) -> Self {
        let new_self = self
            .l
            .into_iter()
            .filter(|(k, _)| k.partition.iter().copied().sum::<usize>() <= n)
            .collect();
        SymmetricFunction { l: new_self }
    }

    fn truncates_to_zero_at(&self, n: usize) -> bool {
        self.l
            .iter()
            .all(|(k, v)| k.partition.iter().copied().sum::<usize>() > n || v.is_zero())
    }

    fn maximal_filtration(&self) -> Result<usize, ()> {
        if self.is_zero() {
            return Err(());
        }
        let w = self
            .l
            .iter()
            .filter_map(|(k, v)| {
                if v.is_zero() {
                    None
                } else {
                    Some(k.partition.iter().copied().sum::<usize>())
                }
            })
            .min()
            .ok_or(())?;
        Ok(w)
    }
}

/// Suppose `f \in L^{>=1}`.
/// Then in the formula `exp(f) = \sum_{k=0}^\infty f^k/k!`
/// if we truncate the sum to `k ≤ TRUNCATION`,
/// then the error term `\sum_{k=TRUNCATION+1}^\infty f^k/k!`
/// is in `L^{>=TRUNCATION+1}`.
/// So when we only care about giving a representative in `L`
/// in the same equivalence class of `exp(f)` in the quotient `L/L^{>=TRUNCATION+1}`,
/// we can compute a truncation of `exp(f)` by omitting terms.
///
/// # Errors
///
/// Returns `Err(partial_sum)` if `f ∉ L^{>=1}`. In that case `fᵏ` retains a part
/// outside `L^{>=1}` for every `k`, so the omitted terms `fᵏ/k!` for `k > TRUNCATION`
/// also have parts outside `L^{>=TRUNCATION+1}` and `partial_sum` does not agree with
/// `exp(f)` up to `L^{>=TRUNCATION+1}` error. To recover the correct answer, write
/// `f = c + g` with commuting `c ∉ L^{>=1}` and `g ∈ L^{>=1}`, then compute
/// `exp(c) * truncated_exponential::<TRUNCATION>(g)`.
/// In order for this to be useful `f = f + 0` would not be the choice.
/// If `L = L^{=0} + L^{>=1}` as a direct sum of commutative monoids
/// and `f` is given in that decomposition, then the `exp(c)`
/// factor stays in the `L^{=0}` sub-semiring and that problem
/// is handled separately because it uses
/// different trait bounds for splitting off the `L^{=0}` part
/// and for computing the exponential of the `L^{=0}` part.
#[allow(clippy::missing_panics_doc)]
pub fn truncated_exponential<const TRUNCATION: usize, L>(mut f: L) -> Result<L, L>
where
    L: FilteredSemiRing + DivAssign<usize> + Clone,
{
    f = f.truncate_at(TRUNCATION);
    if f.is_zero() {
        return Ok(L::one());
    }
    let no_constant_term = f.truncates_to_zero_at(0);
    // no_constant_term means that f is in L^{>=1} but not 0
    // so the error term after truncation is in L^{>=TRUNCATION+1}
    if TRUNCATION == 0 {
        return if no_constant_term {
            Ok(L::one().truncate_at(0))
        } else {
            Err(L::one().truncate_at(0))
        };
    }
    if TRUNCATION == 1 {
        return if no_constant_term {
            Ok((L::one() + f).truncate_at(1))
        } else {
            Err((L::one() + f).truncate_at(1))
        };
    }
    let mut to_return = L::one() + f.clone();
    let mut fk = f.clone();
    let mut k_factorial: usize = 1;
    let where_stop = if no_constant_term {
        let f_max = f
            .maximal_filtration()
            .expect("f is not zero, so it has a maximal filtration");
        debug_assert!(
            f_max >= 1,
            "f has no constant term, so it is in L^{{>=1}}, so its maximal filtration should be at least 1"
        );
        // `f ∈ L^{>=f_max}` implies `fᵏ ∈ L^{>=k*f_max}` by the filtration axiom.
        // We drop k when `k*f_max >= TRUNCATION+1`, i.e. k >= ceil((TRUNCATION+1)/f_max).
        // So where_stop = ceil((TRUNCATION+1)/f_max) - 1.
        div_ceil(TRUNCATION + 1, f_max) - 1
    } else {
        TRUNCATION
    };
    for k in 2..=where_stop {
        k_factorial *= k;
        fk *= f.clone();
        if fk.truncates_to_zero_at(TRUNCATION) {
            break;
        }
        fk = fk.truncate_at(TRUNCATION);
        let mut contribution = fk.clone().truncate_at(TRUNCATION);
        contribution /= k_factorial;
        to_return += contribution;
    }
    if no_constant_term {
        Ok(to_return)
    } else {
        Err(to_return)
    }
}

pub trait AdamsIncreases: LambdaRing + FilteredSemiRing {
    /// For all `f \in L^{>=f_filtration}`
    /// `ψ^{psi_n} (f) ∈ L^{>=psi_bound(psi_n, f_filtration)}`
    /// Implementing this trait is also
    /// an indication that the this
    /// function is strictly increasing in the first variable
    /// as long as the second variable is positive.
    fn psi_bound(psi_n: usize, f_filtration: usize) -> usize;
}

impl<Coeffs> AdamsIncreases for SymmetricFunction<Coeffs>
where
    Coeffs: Ring + Div<usize, Output = Coeffs> + Clone,
{
    fn psi_bound(psi_n: usize, f_filtration: usize) -> usize {
        psi_n * f_filtration
    }
}

/// Computes the plethystic exponential `PLExp(f) = exp(∑_{n=1}^∞ ψⁿ(f)/n)` truncated to
/// a representative in `L` that agrees with `Exp(f)` up to `L^{>=TRUNCATION+1}` error.
///
/// Builds the argument `g = ∑_{n=1}^{...} ψⁿ(f)/n` by accumulating Adams operations,
/// then delegates to [`truncated_exponential`]. The loop has no fixed upper bound: it
/// stops as soon as `ψⁿ(f) ∈ L^{>=TRUNCATION+1}`, at which point all higher
/// `ψ^{n'}(f)` (for `n' ≥ n`) are also in `L^{>=TRUNCATION+1}` and contribute nothing
/// to the representative up to that filtration level.
///
/// # Errors
///
/// Propagates `Err` from [`truncated_exponential`] if `g ∉ L^{>=1}`.
#[allow(clippy::needless_pass_by_value, clippy::missing_panics_doc)]
pub fn plethystic_exp<const TRUNCATION: usize, L>(f: L) -> Result<L, L>
where
    L: AdamsIncreases + DivAssign<usize> + Clone,
{
    if f.is_zero() {
        return Ok(L::one());
    }
    let mut g = L::zero();
    let fmax = f
        .maximal_filtration()
        .expect("f is not zero, so it has a maximal filtration");
    if fmax == 0 {
        warn!(
            "f has a nonzero constant term, so it is in L^{{>=0}} but not in L^{{>=1}}. \
            A constant with psi^n (c) = c for all n results in a divergent c*sum 1/n, \
            so the plethystic exponential is not defined."
        );
    }
    for n in 1usize.. {
        let psi_max = L::psi_bound(n, fmax);
        if psi_max > TRUNCATION {
            break;
        }
        let psi_n = f.clone().psi(n);
        if psi_n.truncates_to_zero_at(TRUNCATION) {
            break;
        }
        let mut contribution = psi_n.truncate_at(TRUNCATION);
        contribution /= n;
        g += contribution;
    }
    truncated_exponential::<TRUNCATION, L>(g)
}

/// Computes `log(g)` as a representative in `L` agreeing with the formal logarithm
/// up to `L^{>=TRUNCATION+1}` error.
///
/// Uses the Taylor series `log(g) = Σ_{k=1}^∞ (-1)^{k+1} (g-1)^k / k`,
/// valid when `g - 1 ∈ L^{>=1}`. The input `g` is truncated first; the loop
/// terminates once `(g-1)^k ∈ L^{>=TRUNCATION+1}`, bounded by the filtration axiom.
///
/// # Errors
///
/// Returns `Err(partial_sum)` if `g - 1 ∉ L^{>=1}`. The omitted terms `(g-1)^k/k`
/// for `k > where_stop` then still contribute to `L^{>=TRUNCATION+1}` error and
/// `partial_sum` does not give the correct representative.
#[allow(clippy::missing_panics_doc)]
pub fn truncated_log<const TRUNCATION: usize, L>(g: L) -> Result<L, L>
where
    L: Ring + FilteredSemiRing + DivAssign<usize> + Clone,
{
    let mut h = g - L::one();
    h = h.truncate_at(TRUNCATION);
    if h.is_zero() {
        return Ok(L::zero());
    }
    let no_constant_term = h.truncates_to_zero_at(0);
    if TRUNCATION == 0 {
        return if no_constant_term {
            Ok(L::zero().truncate_at(0))
        } else {
            Err(L::zero().truncate_at(0))
        };
    }
    let where_stop = if no_constant_term {
        let h_max = h
            .maximal_filtration()
            .expect("h is not zero, so it has a maximal filtration");
        debug_assert!(
            h_max >= 1,
            "h has no constant term, so it is in L^{{>=1}}, so its maximal filtration should be at least 1"
        );
        div_ceil(TRUNCATION + 1, h_max) - 1
    } else {
        TRUNCATION
    };
    let mut to_return = h.clone();
    let mut hk = h.clone();
    for k in 2..=where_stop {
        hk *= h.clone();
        if hk.truncates_to_zero_at(TRUNCATION) {
            break;
        }
        hk = hk.truncate_at(TRUNCATION);
        let mut contribution = hk.clone();
        contribution /= k;
        if k % 2 == 0 {
            to_return -= contribution;
        } else {
            to_return += contribution;
        }
    }
    if no_constant_term {
        Ok(to_return)
    } else {
        Err(to_return)
    }
}

/// Computes the plethystic logarithm `PLLog(g) = Σ_{n=1}^∞ μ(n)/n · log(ψⁿ(g))`,
/// truncated to a representative in `L` agreeing with `PLLog(g)` up to
/// `L^{>=TRUNCATION+1}` error.
///
/// This is the left inverse of [`plethystic_exp`]: `PLLog(PLExp(f)) = f` when
/// `f ∈ L^{>=1}`. The loop terminates once `psi_bound(n, hmax) > TRUNCATION`,
/// at which point `ψⁿ(g-1) ∈ L^{>=TRUNCATION+1}` and `log(ψⁿ(g))` contributes nothing.
/// Terms with `μ(n) = 0` (when `n` has a repeated prime factor) are skipped.
///
/// # Errors
///
/// Returns `Err(partial_sum)` if `g - 1 ∉ L^{>=1}`, mirroring the `Err` semantics
/// of [`truncated_exponential`] and [`plethystic_exp`].
#[allow(clippy::needless_pass_by_value, clippy::missing_panics_doc)]
pub fn plethystic_log<const TRUNCATION: usize, L>(g: L) -> Result<L, L>
where
    L: AdamsIncreases + DivAssign<usize> + Clone,
{
    let h = g.clone() - L::one();
    if h.is_zero() {
        return Ok(L::zero());
    }
    let no_constant_term = h.truncates_to_zero_at(0);
    if !no_constant_term {
        warn!(
            "g - 1 has a nonzero constant term so g ∉ 1 + L^{{>=1}}. \
            PLLog is only defined for inputs in 1 + L^{{>=1}}."
        );
    }
    let hmax = h
        .maximal_filtration()
        .expect("h is not zero, so it has a maximal filtration");
    let mut result = L::zero();
    for n in 1usize.. {
        if L::psi_bound(n, hmax) > TRUNCATION {
            break;
        }
        let mu_n = mobius(n);
        if mu_n == 0 {
            continue;
        }
        let psi_n_g = g.clone().psi(n);
        // psi_n_g - 1 = psi^n(h) ∈ L^{>=psi_bound(n,hmax)} ⊆ L^{>=1}, so Ok is expected
        let log_psi_n_g = truncated_log::<TRUNCATION, L>(psi_n_g).unwrap_or_else(|partial| partial);
        let mut contribution = log_psi_n_g;
        contribution /= n;
        if mu_n > 0 {
            result += contribution;
        } else {
            result -= contribution;
        }
    }
    if no_constant_term {
        Ok(result)
    } else {
        Err(result)
    }
}

#[cfg(test)]
mod tests {
    use num::{One, Zero};
    use proptest::prelude::{prop_assert, proptest};

    use super::{FilteredSemiRing, truncated_exponential};
    use crate::plethystic::{
        LambdaRing,
        generating_series::plethystic_exp,
        test_utils::{Q, approx_eq},
    };

    impl FilteredSemiRing for Q {
        fn truncate_at(self, _n: usize) -> Self {
            self
        }

        fn truncates_to_zero_at(&self, _n: usize) -> bool {
            *self == Q(0.0)
        }
    }

    proptest! {
        #[test]
        fn test_truncated_exponential(q in -1.0..1.0) {
            {
            const TRUNCATION: usize = 5;
            let truncation_factorial = (1..=TRUNCATION).product::<usize>();
            let q = Q(q);
            let exp_q = truncated_exponential::<TRUNCATION, Q>(q.clone());
            let exp_q = if q.truncates_to_zero_at(0) {
                exp_q.expect("truncated_exponential should succeed for inputs with zero constant term")
            } else {
                exp_q.expect_err("truncated_exponential should fail for inputs with nonzero constant term")
            };
            let expected = Q(q.0.exp());
            prop_assert!((exp_q.0 - expected.0).abs() < f64::exp(1.0)/truncation_factorial as f64, "truncated_exponential({:?}) = {:?}, expected approximately {:?}", q, exp_q, expected);
            }

            {
            const TRUNCATION: usize = 9;
            let truncation_factorial = (1..=TRUNCATION).product::<usize>();
            let q = Q(q);
            let exp_q = truncated_exponential::<TRUNCATION, Q>(q.clone());
            let exp_q = if q.truncates_to_zero_at(0) {
                exp_q.expect("truncated_exponential should succeed for inputs with zero constant term")
            } else {
                exp_q.expect_err("truncated_exponential should fail for inputs with nonzero constant term")
            };
            let expected = Q(q.0.exp());
            prop_assert!((exp_q.0 - expected.0).abs() < f64::exp(1.0)/truncation_factorial as f64, "truncated_exponential({:?}) = {:?}, expected approximately {:?}", q, exp_q, expected);
            }
        }

        #[test]
        fn sym_truncated_exp(
            partition0 in proptest::collection::vec(0usize..5, 0..5), coeff0 in -1.0..1.0,
            partition1 in proptest::collection::vec(0usize..5, 0..5), coeff1 in -1.0..1.0,
        ) {
            use crate::plethystic::SymmetricFunction;
            let pn_coeff = SymmetricFunction::singleton(partition0, Q(coeff0));
            let pm_coeff = SymmetricFunction::singleton(partition1, Q(coeff1));
            let input = pn_coeff.clone() + pm_coeff.clone();
            let truncated_exp_input = truncated_exponential::<5, SymmetricFunction<Q>>(input.clone());
            let truncated_exp_input = if input.truncates_to_zero_at(0) {
                truncated_exp_input.expect("truncated_exponential should succeed for inputs with zero constant term")
            } else {
                truncated_exp_input.expect_err("truncated_exponential should fail for inputs with nonzero constant term")
            };
            let input = input.truncate_at(5);
            let mut expected = SymmetricFunction::singleton(vec![], Q(1.0))
                + input.clone()
                + (input.clone() * input.clone()) / 2
                + (input.clone() * input.clone() * input.clone()) / 6
                + (input.clone() * input.clone() * input.clone() * input.clone()) / 24
                + (input.clone() * input.clone() * input.clone() * input.clone() * input.clone()) / 120;
            expected = expected.truncate_at(5);
            assert!(approx_eq(&truncated_exp_input, &expected), "truncated_exponential({:?}) = {:?}, expected approximately {:?}", input, truncated_exp_input, expected);
        }

        #[test]
        fn sym_plethystic_exp(
            partition0 in proptest::collection::vec(0usize..5, 0..5), coeff0 in -1.0..1.0,
            partition1 in proptest::collection::vec(0usize..5, 0..5), coeff1 in -1.0..1.0,
        ) {
            use crate::plethystic::SymmetricFunction;
            let pn_coeff = SymmetricFunction::singleton(partition0, Q(coeff0));
            let pm_coeff = SymmetricFunction::singleton(partition1, Q(coeff1));
            let input = pn_coeff.clone() + pm_coeff.clone();
            if !input.truncates_to_zero_at(0) {
                // If the input has a nonzero constant term, then the plethystic exponential is not defined.
                return Ok(()); // skip this test case
            }
            let plethystic_exp_input = plethystic_exp::<5, SymmetricFunction<Q>>(input.clone()).expect("plethystic_exp should succeed for inputs with zero constant term");
            let input = input.truncate_at(5);
            let mut expected = SymmetricFunction::one();
            if !input.is_zero() {
                for n in 1..=5 {
                    let psi_n_input = input.clone().psi(n) / n;
                    let mut expected_n = SymmetricFunction::singleton(vec![], Q(1.0))
                        + psi_n_input.clone()
                        + (psi_n_input.clone() * psi_n_input.clone()) / 2
                        + (psi_n_input.clone() * psi_n_input.clone() * psi_n_input.clone()) / 6
                        + (psi_n_input.clone() * psi_n_input.clone() * psi_n_input.clone() * psi_n_input.clone()) / 24
                        + (psi_n_input.clone() * psi_n_input.clone() * psi_n_input.clone() * psi_n_input.clone() * psi_n_input.clone()) / 120;
                    expected_n = expected_n.truncate_at(5);
                    expected *= expected_n;
                    expected = expected.truncate_at(5);
                }
            }
            assert!(approx_eq(&plethystic_exp_input, &expected), "PLExp({:?}) = {:?}, expected approximately {:?}", input, plethystic_exp_input, expected);
        }

        #[test]
        fn plexp_pllog_identity(
            partition0 in proptest::collection::vec(1usize..5, 0..4), coeff0 in -0.5..0.5f64,
            partition1 in proptest::collection::vec(1usize..5, 0..4), coeff1 in -0.5..0.5f64,
        ) {
            use crate::plethystic::SymmetricFunction;
            use crate::plethystic::generating_series::plethystic_log;
            const TRUNCATION: usize = 5;
            // Build f ∈ L^{>=1}: partitions with all parts ≥ 1 guarantee no constant term
            let f = SymmetricFunction::singleton(partition0, Q(coeff0))
                + SymmetricFunction::singleton(partition1, Q(coeff1));
            if !f.truncates_to_zero_at(0) {
                return Ok(()); // skip: constant term present
            }
            let g = plethystic_exp::<TRUNCATION, SymmetricFunction<Q>>(f.clone())
                .expect("PLExp should succeed for f ∈ L^{>=1}");
            let f_recovered = plethystic_log::<TRUNCATION, SymmetricFunction<Q>>(g)
                .expect("PLLog should succeed for g = PLExp(f) ∈ 1 + L^{>=1}");
            let f_truncated = f.truncate_at(TRUNCATION);
            assert!(
                approx_eq(&f_recovered, &f_truncated),
                "PLLog(PLExp(f)) = {:?}, expected {:?}",
                f_recovered,
                f_truncated
            );
        }
    }
}
