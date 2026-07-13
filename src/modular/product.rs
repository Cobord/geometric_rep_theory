use std::marker::PhantomData;

use super::coerced::{CoerceTransformation, EquivalentTransportedMF};
use super::modular_def::{ModularError, ModularForm, ModularTransformationGroup};
use crate::arithmetic_utils::Field;

/// The product `f * g` of two modular forms `f` (weight `TWICE_WEIGHT_1/2`)
/// and `g` (weight `TWICE_WEIGHT_2/2`): since
/// `(f*g)(g*tau) = v_f(g)*(c*tau+d)^{w1}*f(tau) * v_g(g)*(c*tau+d)^{w2}*g(tau)
/// = (v_f(g)*v_g(g)) * (c*tau+d)^{w1+w2} * (f*g)(tau)`, the product is
/// itself modular, of weight `(TWICE_WEIGHT_1 + TWICE_WEIGHT_2) / 2`,
/// transforming under [`CombinedTransformationGroup`] — `v_f * v_g` packaged
/// up as a single multiplier system, so `f` and `g` need not individually
/// have a trivial multiplier, nor even share the same transformation group
/// type.
///
/// Stable Rust has no `TWICE_WEIGHT_1 + TWICE_WEIGHT_2` in a const-generic
/// position, so the caller must additionally specify `TWICE_WEIGHT_SUM`;
/// [`ProductModularForm::new`] checks at compile time that it actually
/// equals `TWICE_WEIGHT_1 + TWICE_WEIGHT_2`.
#[allow(dead_code)]
pub struct ProductModularForm<
    R: Field,
    F1,
    F2,
    const TWICE_WEIGHT_1: usize,
    const TWICE_WEIGHT_2: usize,
    const TWICE_WEIGHT_SUM: usize,
> {
    /// The weight-`TWICE_WEIGHT_1/2` factor `f`.
    pub first: F1,
    /// The weight-`TWICE_WEIGHT_2/2` factor `g`.
    pub second: F2,
    r: PhantomData<R>,
}

impl<
    R: Field,
    F1: ModularForm<TWICE_WEIGHT_1, R>,
    F2: ModularForm<TWICE_WEIGHT_2, R>,
    const TWICE_WEIGHT_1: usize,
    const TWICE_WEIGHT_2: usize,
    const TWICE_WEIGHT_SUM: usize,
> ProductModularForm<R, F1, F2, TWICE_WEIGHT_1, TWICE_WEIGHT_2, TWICE_WEIGHT_SUM>
{
    const WEIGHT_CHECK: () = assert!(
        TWICE_WEIGHT_SUM == TWICE_WEIGHT_1 + TWICE_WEIGHT_2,
        "TWICE_WEIGHT_SUM must equal TWICE_WEIGHT_1 + TWICE_WEIGHT_2 for ProductModularForm"
    );

    /// Build `first * second`. Fails to compile if
    /// `TWICE_WEIGHT_SUM != TWICE_WEIGHT_1 + TWICE_WEIGHT_2` (checked via a
    /// `const` `assert!`).
    #[must_use = "The factors are now inside the product"]
    #[allow(dead_code)]
    pub fn new(first: F1, second: F2) -> Self {
        let () = Self::WEIGHT_CHECK;
        Self {
            first,
            second,
            r: PhantomData,
        }
    }
}

/// The intersection `G1 ∩ G2` of two transformation groups — [`new`](Self::new)
/// only accepts a matrix if both `G1::new` and `G2::new` accept it — with
/// their multiplier systems multiplied together: `v(g) = v_first(g) *
/// v_second(g)`. The transformation-group-level analogue of
/// [`ProductModularForm`] — e.g. combining `Sl2Z`'s trivial multiplier with
/// a genuinely nontrivial one to get the group that a product of such forms
/// transforms under.
///
/// `raw_a`..`raw_d` and `transform_q` are delegated to `first` rather than
/// `second`, but it doesn't matter which: membership in the intersection
/// means both were built from the very same `raw_matrix`, so they agree on
/// it by construction.
#[allow(dead_code)]
pub struct CombinedTransformationGroup<G1, G2> {
    /// The `G1` side of the intersection, built from the same `raw_matrix`
    /// as `second`.
    pub first: G1,
    /// The `G2` side of the intersection, built from the same `raw_matrix`
    /// as `first`.
    pub second: G2,
}

impl<R: Field, G1: ModularTransformationGroup<R>, G2: ModularTransformationGroup<R>>
    ModularTransformationGroup<R> for CombinedTransformationGroup<G1, G2>
{
    fn new(raw_matrix: [[R; 2]; 2]) -> Result<Self, ModularError> {
        let [[a, b], [c, d]] = raw_matrix;
        let first = G1::new([[a.clone(), b.clone()], [c.clone(), d.clone()]])?;
        let second = G2::new([[a, b], [c, d]])?;
        Ok(Self { first, second })
    }

    fn transform_q(&self, q: &R) -> R {
        self.first.transform_q(q)
    }

    fn multiplier_system(&self, tau: &R) -> R {
        self.first.multiplier_system(tau) * self.second.multiplier_system(tau)
    }

    fn raw_a(&self) -> R {
        self.first.raw_a()
    }

    fn raw_b(&self) -> R {
        self.first.raw_b()
    }

    fn raw_c(&self) -> R {
        self.first.raw_c()
    }

    fn raw_d(&self) -> R {
        self.first.raw_d()
    }

    fn is_trivial_multiplier_system() -> bool {
        G1::is_trivial_multiplier_system() && G2::is_trivial_multiplier_system()
    }
}

impl<
    R: Field,
    F1: ModularForm<TWICE_WEIGHT_1, R>,
    F2: ModularForm<TWICE_WEIGHT_2, R>,
    const TWICE_WEIGHT_1: usize,
    const TWICE_WEIGHT_2: usize,
    const TWICE_WEIGHT_SUM: usize,
> ModularForm<TWICE_WEIGHT_SUM, R>
    for ProductModularForm<R, F1, F2, TWICE_WEIGHT_1, TWICE_WEIGHT_2, TWICE_WEIGHT_SUM>
{
    type TransformationGroup =
        CombinedTransformationGroup<F1::TransformationGroup, F2::TransformationGroup>;

    /// The Cauchy product of the two factors' `q`-expansions:
    /// `c_n = sum_{k=0}^{n} a_k * b_{n-k}`.
    fn extract_coeffs(&self, which_coeff: usize) -> Result<R, ModularError> {
        let mut to_return = R::zero();
        for k in 0..=which_coeff {
            let a_k = self.first.extract_coeffs(k)?;
            let b_rest = self.second.extract_coeffs(which_coeff - k)?;
            to_return += a_k * b_rest;
        }
        Ok(to_return)
    }

    fn evaluate_at(&self, q: &R) -> Result<R, ModularError> {
        Ok(self.first.evaluate_at(q)? * self.second.evaluate_at(q)?)
    }
}

/// `t * t`, as a [`ProductModularForm`] — a convenience over calling
/// [`ProductModularForm::new`] directly so the caller doesn't have to clone
/// `t` by hand or spell out `FOUR_WEIGHT = TWICE_WEIGHT + TWICE_WEIGHT`.
pub fn square_modular_form<
    R: Field,
    const TWICE_WEIGHT: usize,
    const FOUR_WEIGHT: usize,
    T: ModularForm<TWICE_WEIGHT, R> + Clone,
>(
    t: T,
) -> ProductModularForm<R, T, T, TWICE_WEIGHT, TWICE_WEIGHT, FOUR_WEIGHT> {
    ProductModularForm::new(t.clone(), t)
}

/// `t * t * t`, built as `t * (t * t)` from nested [`ProductModularForm`]s —
/// a convenience over constructing those nested products by hand, taking
/// care of the `FOUR_WEIGHT`/`SIX_WEIGHT` const-generic bookkeeping.
pub fn cube_modular_form<
    R: Field,
    const TWICE_WEIGHT: usize,
    const FOUR_WEIGHT: usize,
    const SIX_WEIGHT: usize,
    T: ModularForm<TWICE_WEIGHT, R> + Clone,
>(
    t: T,
) -> impl ModularForm<SIX_WEIGHT, R> {
    ProductModularForm::new(
        t.clone(),
        ProductModularForm::<R, T, T, TWICE_WEIGHT, TWICE_WEIGHT, FOUR_WEIGHT>::new(t.clone(), t),
    )
}

/// `t * t`, coerced down to `T::TransformationGroup` itself rather than left
/// typed as `CombinedTransformationGroup<T::TransformationGroup,
/// T::TransformationGroup>`: squaring a trivial multiplier system leaves it
/// trivial (`1 * 1 = 1`), so the two groups act identically and share the
/// same (trivial) multiplier — the coercion just names that fact at the type
/// level, via [`CoerceTransformation::FORCED`], so the result can sit
/// alongside other `T::TransformationGroup`-typed forms (e.g. in a
/// [`SumModularForm`](super::SumModularForm)) without an extra manual
/// coercion step at every call site.
///
/// # Panics
///
/// Panics if `T::TransformationGroup::is_trivial_multiplier_system()` is
/// `false` — the coercion above only holds for a trivial multiplier; for a
/// genuine character, `CombinedTransformationGroup<G, G>`'s multiplier
/// `v(g)^2` need not equal `G`'s own `v(g)`, so forcing the two together
/// would be unsound.
pub fn square_modular_form_trivial<
    R: Field,
    const TWICE_WEIGHT: usize,
    const FOUR_WEIGHT: usize,
    T: ModularForm<TWICE_WEIGHT, R> + Clone,
>(
    t: T,
) -> impl ModularForm<FOUR_WEIGHT, R, TransformationGroup = T::TransformationGroup> {
    assert!(
        T::TransformationGroup::is_trivial_multiplier_system(),
        "square_modular_form_trivial requires T::TransformationGroup to have a trivial multiplier system"
    );
    let coercion = CoerceTransformation::FORCED(
        PhantomData,
        PhantomData,
        PhantomData::<T::TransformationGroup>,
    );
    EquivalentTransportedMF::coerce(square_modular_form(t), coercion)
}

/// `t * t * t`, coerced down to `T::TransformationGroup` for the same reason
/// as [`square_modular_form_trivial`]: a trivial multiplier system stays
/// trivial under repeated products (`1 * 1 * 1 = 1`), so nesting
/// `CombinedTransformationGroup`s around a trivial `G` is "the same" group as
/// `G` alone.
///
/// # Panics
///
/// Panics if `T::TransformationGroup::is_trivial_multiplier_system()` is
/// `false`, for the same reason as [`square_modular_form_trivial`].
pub fn cube_modular_form_trivial<
    R: Field,
    const TWICE_WEIGHT: usize,
    const FOUR_WEIGHT: usize,
    const SIX_WEIGHT: usize,
    T: ModularForm<TWICE_WEIGHT, R> + Clone,
>(
    t: T,
) -> impl ModularForm<SIX_WEIGHT, R, TransformationGroup = T::TransformationGroup> {
    assert!(
        T::TransformationGroup::is_trivial_multiplier_system(),
        "cube_modular_form_trivial requires T::TransformationGroup to have a trivial multiplier system"
    );
    let product = ProductModularForm::new(
        t.clone(),
        ProductModularForm::<R, T, T, TWICE_WEIGHT, TWICE_WEIGHT, FOUR_WEIGHT>::new(t.clone(), t),
    );
    let coercion = CoerceTransformation::FORCED(
        PhantomData,
        PhantomData,
        PhantomData::<T::TransformationGroup>,
    );
    EquivalentTransportedMF::coerce(product, coercion)
}
