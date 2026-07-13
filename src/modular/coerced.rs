use std::marker::PhantomData;

use num::Zero;

use super::modular_def::{ModularError, ModularForm, ModularTransformationGroup};
use crate::arithmetic_utils::Field;

/// A witness that `T1` and `T2` are "the same"
/// [`ModularTransformationGroup`] — the same group with the same multiplier
/// system — letting a [`ModularForm`] built against `T1` be reinterpreted as
/// one against `T2` via [`EquivalentTransportedMF`], e.g. so forms that
/// started out with different (but nameably equal) `TransformationGroup`
/// types can sit together in one [`SumModularForm`](super::SumModularForm).
pub enum CoerceTransformation<
    R: Field,
    T1: ModularTransformationGroup<R>,
    T2: ModularTransformationGroup<R>,
> {
    /// Asserts `T1 == T2` outright, with no check performed at construction
    /// time. Callers who want more assurance than a bare assertion should
    /// follow up with [`validate`](CoerceTransformation::validate), which
    /// empirically spot-checks the claim rather than trusting it blindly.
    FORCED(PhantomData<R>, PhantomData<T1>, PhantomData<T2>),
}

/// A handful of pseudo-random points of `H` (the upper half-plane) in `R`,
/// used by [`validate`](CoerceTransformation::validate) to check that two
/// multiplier systems agree everywhere, not just at one `tau` a caller
/// happens to have picked. Fixed-seed (deterministic across runs, since
/// `Field` has no source of randomness of its own and reproducible test
/// failures matter more here than genuine entropy); always genuinely
/// off the real axis, since `R` is always (some representation of) the
/// complex numbers (see the [`ModularTransformationGroup`] trait-level
/// docs) and a real `tau` would sit exactly on the branch cut
/// [`EtaTransformationGroup`]'s `c < 0` / `c = 0, a = -1` multiplier branch
/// resolves by checking which side of.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn sample_points<R: Field + nalgebra::ComplexField>() -> Vec<R> {
    let rational = |num: i64, den: i64| -> R {
        let magnitude = R::natural_inclusion(num.unsigned_abs() as usize)
            * R::natural_inclusion(den.unsigned_abs() as usize).inv();
        if num < 0 { -magnitude } else { magnitude }
    };
    // `sqrt` only pins down `+-i`, not which one; `R::ComplexField` gives
    // enough (a real/imaginary decomposition, and an ordering on the real
    // part) to always pick the one with positive imaginary part.
    let mut imaginary_unit = (-R::one()).sqrt();
    if imaginary_unit.clone().imaginary() < <R as nalgebra::ComplexField>::RealField::zero() {
        imaginary_unit = -imaginary_unit;
    }
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    #[allow(clippy::cast_possible_wrap)]
    let mut next_i64 = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 33) as i64
    };
    (0..8)
        .map(|_| {
            let re_num = next_i64() % 41 - 20;
            let re_den = next_i64().rem_euclid(9) + 1;
            let im_num = next_i64().rem_euclid(37) + 1; // strictly positive: stay in H
            let im_den = next_i64().rem_euclid(9) + 1;
            rational(re_num, re_den) + imaginary_unit.clone() * rational(im_num, im_den)
        })
        .collect()
}

impl<R: Field, T1: ModularTransformationGroup<R>, T2: ModularTransformationGroup<R>>
    CoerceTransformation<R, T1, T2>
{
    /// Empirically check that `T1` and `T2` really are the same group with
    /// the same multiplier system, by testing agreement on a finite list of
    /// generators — at a handful of sample points from [`sample_points`],
    /// standing in for "for all `tau`" — rather than trusting
    /// [`CoerceTransformation::FORCED`] blindly.
    ///
    /// For each generator `g1` in `gens_t1` (an element of `T1`), rebuilds
    /// the same matrix as a `T2` element via
    /// [`new`](ModularTransformationGroup::new) and checks that the two
    /// multiplier systems agree at every sample point (via `close_enough`,
    /// since `R` is typically a floating-point type with no exact equality);
    /// symmetrically, for each generator `g2` in `gens_t2`, rebuilds it as a
    /// `T1` element and checks the same thing. *If* `gens_t1` and `gens_t2`
    /// actually generate the whole group (e.g. `S` and `T` for `SL_2(Z)`),
    /// and the multiplier systems are genuine cocycles, then agreement on
    /// generators would imply agreement everywhere — but this function has
    /// no way to verify that what it was handed actually generates the
    /// group, nor that a handful of sample points really stand in for every
    /// `tau`, so this only ever amounts to a spot check, not a proof. The
    /// caller is responsible for that part.
    ///
    /// # Errors
    ///
    /// Returns [`ModularError::InequivalentTransformationGroups`] if some
    /// generator's matrix isn't even a valid element of the other group, or
    /// if the two multiplier systems disagree at some sample point.
    #[allow(clippy::similar_names)]
    pub fn validate(
        self,
        gens_t1: &[T1],
        gens_t2: &[T2],
        close_enough: fn(&R, &R) -> bool,
    ) -> Result<Self, ModularError>
    where
        R: nalgebra::ComplexField,
    {
        let taus = sample_points::<R>();
        for gen_t1 in gens_t1 {
            let gen_t2 = T2::new(gen_t1.raw_matrix())
                .map_err(|_| ModularError::InequivalentTransformationGroups)?;
            for tau in &taus {
                let mult_t1 = gen_t1.multiplier_system(tau);
                let mult_t2 = gen_t2.multiplier_system(tau);
                if !close_enough(&mult_t1, &mult_t2) {
                    return Err(ModularError::InequivalentTransformationGroups);
                }
            }
        }
        for gen_t2 in gens_t2 {
            let gen_t1 = T1::new(gen_t2.raw_matrix())
                .map_err(|_| ModularError::InequivalentTransformationGroups)?;
            for tau in &taus {
                let mult_t1 = gen_t1.multiplier_system(tau);
                let mult_t2 = gen_t2.multiplier_system(tau);
                if !close_enough(&mult_t1, &mult_t2) {
                    return Err(ModularError::InequivalentTransformationGroups);
                }
            }
        }
        Ok(self)
    }
}

/// A [`ModularForm`] wrapping some `underlying: M`, but retyped so that
/// [`ModularForm::TransformationGroup`] reads as `TG` rather than
/// `M::TransformationGroup` — [`extract_coeffs`](ModularForm::extract_coeffs)
/// and [`evaluate_at`](ModularForm::evaluate_at) both delegate straight
/// through to `underlying`, since `TG` and `M::TransformationGroup` are the
/// same group by construction (see [`coerce`](Self::coerce)), just possibly
/// different Rust types naming it.
pub struct EquivalentTransportedMF<
    const TWICE_WEIGHT: usize,
    R: Field,
    M: ModularForm<TWICE_WEIGHT, R>,
    TG: ModularTransformationGroup<R>,
> {
    #[allow(dead_code)]
    coercion: CoerceTransformation<R, M::TransformationGroup, TG>,
    underlying: M,
}

impl<
    const TWICE_WEIGHT: usize,
    R: Field,
    M: ModularForm<TWICE_WEIGHT, R>,
    TG: ModularTransformationGroup<R>,
> EquivalentTransportedMF<TWICE_WEIGHT, R, M, TG>
{
    /// Wrap `underlying` behind the retyped `TransformationGroup = TG`,
    /// using `coercion` as the (possibly unchecked, if
    /// [`CoerceTransformation::FORCED`]) proof that `TG` really is
    /// `M::TransformationGroup`.
    pub fn coerce(
        underlying: M,
        coercion: CoerceTransformation<R, M::TransformationGroup, TG>,
    ) -> Self {
        Self {
            coercion,
            underlying,
        }
    }
}

impl<
    const TWICE_WEIGHT: usize,
    R: Field,
    M: ModularForm<TWICE_WEIGHT, R>,
    TG: ModularTransformationGroup<R>,
> ModularForm<TWICE_WEIGHT, R> for EquivalentTransportedMF<TWICE_WEIGHT, R, M, TG>
{
    type TransformationGroup = TG;

    fn extract_coeffs(&self, which_coeff: usize) -> Result<R, ModularError> {
        self.underlying.extract_coeffs(which_coeff)
    }

    fn evaluate_at(&self, q: &R) -> Result<R, ModularError> {
        self.underlying.evaluate_at(q)
    }
}
