//! Modular forms, and the matrix groups they transform under.
//!
//! A lattice's theta series (see
//! [`Lattice::theta_series_weight`](crate::lattice::Lattice::theta_series_weight)
//! and
//! [`Lattice::theta_series_character`](crate::lattice::Lattice::theta_series_character))
//! is a motivating example of a [`ModularForm`]: this module models the
//! transformation law itself, independently of any particular lattice.

use std::fmt::Debug;
use std::{fmt, marker::PhantomData};

use nalgebra::{ArrayStorage, Complex, Const, SMatrix};

use crate::arithmetic_utils::{Field, sigma_3, sigma_5};

/// Errors arising from [`ModularTransformationGroup`] and [`ModularForm`] operations.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum ModularError {
    /// A matrix entry is not, within tolerance, an integer (see the
    /// [`ModularTransformationGroup`] trait-level docs on why entries must
    /// lie in the canonical image of `Z` inside `R`).
    NonIntegerEntry,
    /// The (integer) matrix fails this group's membership condition, e.g.
    /// determinant `!= 1` for `SL_2(Z)`.
    NotInGroup,
    /// Fewer constraints were supplied than the dimension of the space being
    /// solved for.
    NotEnoughConstraints,
    /// The linear system determined by the constraints is singular, so they
    /// don't pin down a unique combination.
    SingularSystem,
    /// A solved coefficient came out non-finite (`NaN` or infinite).
    NonFiniteCoefficient,
    /// No implementation is available to compute this value.
    Unavailable,
    /// The two subgroups of SL(2,Z) are not the same as groups or do not
    /// have the same multiplier system.
    InequivalentTransformationGroups,
}

impl fmt::Display for ModularError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonIntegerEntry => write!(f, "matrix entry is not an integer"),
            Self::NotInGroup => write!(f, "matrix fails this group's membership condition"),
            Self::NotEnoughConstraints => {
                write!(
                    f,
                    "fewer constraints supplied than the dimension of the space"
                )
            }
            Self::SingularSystem => {
                write!(f, "constraints do not pin down a unique linear combination")
            }
            Self::NonFiniteCoefficient => write!(f, "a solved coefficient is not finite"),
            Self::Unavailable => write!(f, "no implementation is available to compute this value"),
            Self::InequivalentTransformationGroups => write!(
                f,
                "The two subgroups of SL(2,Z) are not the same as groups or do not have the same multiplier system."
            ),
        }
    }
}

impl std::error::Error for ModularError {}

/// A group of `2x2` matrices `g = (a b; c d)` acting on the upper half-plane
/// by Mobius transformations `tau -> (a*tau+b)/(c*tau+d)`, together with the
/// multiplier system needed to describe how a modular form of weight `w`
/// transforms under it: by `(c*tau+d)^w` alone when `w` is an integer, or
/// with the extra root-of-unity-valued factor from
/// [`multiplier_system`](Self::multiplier_system) when `w` is a half-integer
/// (see [`ModularForm`], where `w` is represented as `TWICE_WEIGHT/2` so
/// that half-integer weights stay expressible as a `usize` const generic).
///
/// `R` is how complex numbers are being parameterized here. `tau` and `q`
/// range over the upper half-plane (or the punctured disk, for `q`), so they
/// must be genuinely complex-valued; `R` is kept generic, rather than fixed
/// to one concrete representation, so that different complex-number types
/// (`Complex<f64>` vs `Complex<f32>`, say) can all be used. The matrix
/// entries `a, b, c, d` are typed as `R` too, purely so they can be combined
/// arithmetically with `tau`/`q` (e.g. via `mul_add`) — but a modular group
/// is by definition a subgroup of `GL_2(Z)`, so the entries must actually
/// lie in the canonical image of `Z` inside `R`, not range over all of `R`.
pub trait ModularTransformationGroup<R: Field>: Sized {
    /// Build a group element from its raw matrix entries `[[a, b], [c, d]]`.
    ///
    /// # Errors
    ///
    /// Returns [`ModularError::NonIntegerEntry`] if any entry of
    /// `raw_matrix` is not an integer (i.e. not in the canonical image of
    /// `Z` inside `R` — see the trait-level docs), or
    /// [`ModularError::NotInGroup`] if the integer matrix fails whatever
    /// further membership condition defines this particular group (e.g.
    /// determinant `1` for `SL_2(Z)`, or congruence conditions for a
    /// congruence subgroup).
    #[allow(dead_code)]
    fn new(raw_matrix: [[R; 2]; 2]) -> Result<Self, ModularError>;

    /// The action on the nome `q` directly, rather than via `tau` and
    /// `q = exp(2*pi*i*tau)`. Useful when working with `q`-expansions, where
    /// composing with `exp`/`log` would be both unnecessary and (for most
    /// `R`, which need not even have those operations) unavailable.
    #[allow(dead_code)]
    fn transform_q(&self, q: &R) -> R;

    /// The Mobius action `tau -> (a*tau+b)/(c*tau+d)` on the upper
    /// half-plane.
    ///
    /// Returns `None` when `c*tau+d = 0`, since `tau` is then sent to the
    /// point at infinity rather than to another value of `R`.
    #[allow(dead_code)]
    fn transform_tau(&self, tau: &R) -> Option<R> {
        let numerator = tau.clone().mul_add(self.raw_a(), self.raw_b());
        let denominator = tau.clone().mul_add(self.raw_c(), self.raw_d());
        if denominator.is_zero() {
            None
        } else {
            let den_inv = denominator.inv();
            Some(numerator * den_inv)
        }
    }

    /// The scalar multiplier `v(g)` in the transformation law
    /// `f(g*tau) = v(g) * (c*tau+d)^w * f(tau)` for a weight-`w` modular form
    /// `f` (see [`ModularForm`] for how `w`, which may be a half-integer, is
    /// represented). `1` for an honest integer-weight form under `SL_2`; a
    /// nontrivial root of unity for forms of half-integer weight (e.g. the
    /// Dedekind eta function, or the theta series of an odd-rank lattice).
    #[allow(dead_code)]
    fn multiplier_system(&self) -> R;

    fn is_trivial_multiplier_system() -> bool;

    /// The `(1, 1)` entry `a` of the underlying matrix — an integer, embedded
    /// in `R` (see the trait-level docs).
    fn raw_a(&self) -> R;

    /// The `(1, 2)` entry `b` of the underlying matrix — an integer, embedded
    /// in `R` (see the trait-level docs).
    fn raw_b(&self) -> R;

    /// The `(2, 1)` entry `c` of the underlying matrix — an integer, embedded
    /// in `R` (see the trait-level docs).
    fn raw_c(&self) -> R;

    /// The `(2, 2)` entry `d` of the underlying matrix — an integer, embedded
    /// in `R` (see the trait-level docs).
    fn raw_d(&self) -> R;

    fn raw_matrix(&self) -> [[R; 2]; 2] {
        [[self.raw_a(), self.raw_b()], [self.raw_c(), self.raw_d()]]
    }
}

pub enum CoerceTransformation<
    R: Field,
    T1: ModularTransformationGroup<R>,
    T2: ModularTransformationGroup<R>,
> {
    FORCED(PhantomData<R>, PhantomData<T1>, PhantomData<T2>),
}

impl<R: Field, T1: ModularTransformationGroup<R>, T2: ModularTransformationGroup<R>>
    CoerceTransformation<R, T1, T2>
{
    /// Empirically check that `T1` and `T2` really are the same group with
    /// the same multiplier system, by testing agreement on a finite list of
    /// generators rather than trusting [`CoerceTransformation::FORCED`]
    /// blindly.
    ///
    /// For each generator `g1` in `gens_t1` (an element of `T1`), rebuilds
    /// the same matrix as a `T2` element via
    /// [`new`](ModularTransformationGroup::new) and checks that the two
    /// multiplier systems agree (via `close_enough`, since `R` is typically
    /// a floating-point type with no exact equality); symmetrically, for
    /// each generator `g2` in `gens_t2`, rebuilds it as a `T1` element and
    /// checks the same thing. *If* `gens_t1` and `gens_t2` actually generate
    /// the whole group (e.g. `S` and `T` for `SL_2(Z)`), and the multiplier
    /// systems are genuine cocycles, then agreement on generators would
    /// imply agreement everywhere — but this function has no way to verify
    /// that what it was handed actually generates the group, so passing it
    /// too small or otherwise insufficient a list (including the empty
    /// list, which trivially passes) only ever amounts to a spot check, not
    /// a proof. The caller is responsible for that part.
    ///
    /// # Errors
    ///
    /// Returns [`ModularError::InequivalentTransformationGroups`] if some
    /// generator's matrix isn't even a valid element of the other group, or
    /// if the two multiplier systems disagree on it.
    #[allow(clippy::similar_names)]
    pub fn validate(
        self,
        gens_t1: &[T1],
        gens_t2: &[T2],
        close_enough: fn(&R, &R) -> bool,
    ) -> Result<Self, ModularError> {
        for gen_t1 in gens_t1 {
            let gen_t2 = T2::new(gen_t1.raw_matrix())
                .map_err(|_| ModularError::InequivalentTransformationGroups)?;
            let mult_t1 = gen_t1.multiplier_system();
            let mult_t2 = gen_t2.multiplier_system();
            if !close_enough(&mult_t1, &mult_t2) {
                return Err(ModularError::InequivalentTransformationGroups);
            }
        }
        for gen_t2 in gens_t2 {
            let gen_t1 = T1::new(gen_t2.raw_matrix())
                .map_err(|_| ModularError::InequivalentTransformationGroups)?;
            let mult_t1 = gen_t1.multiplier_system();
            let mult_t2 = gen_t2.multiplier_system();
            if !close_enough(&mult_t1, &mult_t2) {
                return Err(ModularError::InequivalentTransformationGroups);
            }
        }
        Ok(self)
    }
}

/// `SL_2(Z)` itself, represented over `f64` — just enough to be a concrete
/// [`ModularTransformationGroup`] for [`EisensteinE4`] (and other weight-`k`,
/// level-`1`, trivial-character forms) to transform under.
pub struct Sl2Z {
    a: i128,
    b: i128,
    c: i128,
    d: i128,
}

impl ModularTransformationGroup<f64> for Sl2Z {
    fn new(raw_matrix: [[f64; 2]; 2]) -> Result<Self, ModularError> {
        const EPSILON: f64 = 1e-9;
        let [[a, b], [c, d]] = raw_matrix;
        if [a, b, c, d]
            .into_iter()
            .any(|x| (x - x.round()).abs() > EPSILON)
        {
            return Err(ModularError::NonIntegerEntry);
        }
        #[allow(clippy::cast_possible_truncation)]
        let (a, b, c, d) = (
            a.round() as i128,
            b.round() as i128,
            c.round() as i128,
            d.round() as i128,
        );
        if a * d - b * c != 1 {
            return Err(ModularError::NotInGroup);
        }
        Ok(Self { a, b, c, d })
    }

    fn transform_q(&self, q: &f64) -> f64 {
        *q
    }

    fn multiplier_system(&self) -> f64 {
        1.0
    }

    #[allow(clippy::cast_precision_loss)]
    fn raw_a(&self) -> f64 {
        self.a as f64
    }

    #[allow(clippy::cast_precision_loss)]
    fn raw_b(&self) -> f64 {
        self.b as f64
    }

    #[allow(clippy::cast_precision_loss)]
    fn raw_c(&self) -> f64 {
        self.c as f64
    }

    #[allow(clippy::cast_precision_loss)]
    fn raw_d(&self) -> f64 {
        self.d as f64
    }

    fn is_trivial_multiplier_system() -> bool {
        true
    }
}

/// `SL_2(Z)` again, but over `Complex<f64>` with the Dedekind eta function's
/// multiplier system instead of the trivial one: eta is weight `1/2`, so it
/// needs genuinely complex `tau`/`q` (see the [`ModularTransformationGroup`]
/// trait-level docs on why `R` is generic) and a nontrivial multiplier,
/// unlike [`Sl2Z`], which this is otherwise structurally identical to.
#[allow(dead_code)]
pub struct EtaTransformationGroup {
    a: i128,
    b: i128,
    c: i128,
    d: i128,
}

impl ModularTransformationGroup<Complex<f64>> for EtaTransformationGroup {
    fn new(raw_matrix: [[Complex<f64>; 2]; 2]) -> Result<Self, ModularError> {
        const EPSILON: f64 = 1e-9;
        let [[a, b], [c, d]] = raw_matrix;
        if [a, b, c, d]
            .into_iter()
            .any(|x| x.im.abs() > EPSILON || (x.re - x.re.round()).abs() > EPSILON)
        {
            return Err(ModularError::NonIntegerEntry);
        }
        #[allow(clippy::cast_possible_truncation)]
        let (a, b, c, d) = (
            a.re.round() as i128,
            b.re.round() as i128,
            c.re.round() as i128,
            d.re.round() as i128,
        );
        if a * d - b * c != 1 {
            return Err(ModularError::NotInGroup);
        }
        Ok(Self { a, b, c, d })
    }

    fn transform_q(&self, q: &Complex<f64>) -> Complex<f64> {
        *q
    }

    /// The eta multiplier `ε(g)` in
    /// `eta(g*tau) = ε(g) * (c*tau+d)^(1/2) * eta(tau)`, a `24`th root of
    /// unity given by Dedekind sums.
    ///
    /// For `c > 0`:
    ///
    /// `ε(g) = exp(i*pi * ((a+d)/(12*c) - s(d,c) - 1/4))`
    ///
    /// where `s(h,k)` is the Dedekind sum
    ///
    /// `s(h,k) = sum_{r=1}^{k-1} ((r/k)) * ((h*r/k))`,
    ///
    /// with `((x)) = x - floor(x) - 1/2` for non-integer `x`, and `((x)) = 0`
    /// for integer `x` (the sawtooth function) — note this is *not* yet
    /// implemented anywhere in this crate (see
    /// [`crate::arithmetic_utils`]), so it would need to be added.
    ///
    /// For `c = 0`: then `ad = 1`, and `SL_2(Z)` representatives are taken
    /// with `d > 0`, forcing `a = d = 1` (i.e. `g = T^b` for the generator
    /// `T = (1 1; 0 1)`):
    ///
    /// `ε(g) = exp(i*pi*b/12)`
    ///
    /// Sanity checks once implemented, against the two generators of
    /// `SL_2(Z)`: `S = (0 -1; 1 0)` has `c = 1`, `s(0,1) = 0` (empty sum), so
    /// `ε(S)` should come out to `exp(-i*pi/4)`, matching the classical
    /// identity `eta(-1/tau) = sqrt(-i*tau) * eta(tau)`; `T = (1 1; 0 1)` is
    /// the `c = 0` case with `b = 1`, so `ε(T)` should be `exp(i*pi/12)`,
    /// matching `eta(tau+1) = exp(i*pi/12) * eta(tau)`.
    ///
    /// # Panics
    ///
    /// Not yet implemented — always panics with `todo!`.
    fn multiplier_system(&self) -> Complex<f64> {
        todo!("Dedekind eta multiplier system (Dedekind sums) not yet implemented")
    }

    #[allow(clippy::cast_precision_loss)]
    fn raw_a(&self) -> Complex<f64> {
        Complex::new(self.a as f64, 0.0)
    }

    #[allow(clippy::cast_precision_loss)]
    fn raw_b(&self) -> Complex<f64> {
        Complex::new(self.b as f64, 0.0)
    }

    #[allow(clippy::cast_precision_loss)]
    fn raw_c(&self) -> Complex<f64> {
        Complex::new(self.c as f64, 0.0)
    }

    #[allow(clippy::cast_precision_loss)]
    fn raw_d(&self) -> Complex<f64> {
        Complex::new(self.d as f64, 0.0)
    }

    fn is_trivial_multiplier_system() -> bool {
        false
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
    pub first: G1,
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

    fn multiplier_system(&self) -> R {
        self.first.multiplier_system() * self.second.multiplier_system()
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

/// A modular form of weight `TWICE_WEIGHT / 2`, transforming under
/// [`Self::TransformationGroup`].
///
/// The weight is tracked as `TWICE_WEIGHT` (twice the actual weight) rather
/// than as a general `Ratio`, for two reasons. First, only a `usize` (or
/// other structural type) can be a const generic parameter at all. Second,
/// and more importantly: every weight that actually occurs for a classical
/// modular form — e.g. theta series, see
/// [`Lattice::theta_series_weight`](crate::lattice::Lattice::theta_series_weight)
/// — is an integer or a half-integer, never something like `1/3`. A `Ratio`
/// would admit that illegal denominator and need a runtime check to reject
/// it; `TWICE_WEIGHT: usize` makes it unrepresentable in the first place,
/// since `TWICE_WEIGHT / 2` only ever has denominator `1` or `2`.
///
/// The reason denominator `2` (and no other) actually occurs is the
/// automorphy factor `(c*tau+d)^w` itself. For integer `w` it's single
/// valued, no branch choice needed. For `w` a half-integer it needs a choice
/// of square root of `c*tau+d`, and those choices can be made consistently
/// (i.e. compatibly with the group law, as a genuine one-cocycle) precisely
/// because `SL_2` has a connected double cover — the metaplectic group
/// `Mp_2` — whose elements are exactly pairs `(g, choice of sqrt(c*tau+d))`.
pub trait ModularForm<const TWICE_WEIGHT: usize, R: Field> {
    /// The group of transformations this form is modular with respect to.
    type TransformationGroup: ModularTransformationGroup<R>;

    /// The `which_coeff`-th coefficient of this form's `q`-expansion.
    ///
    /// # Errors
    ///
    /// Returns [`ModularError::Unavailable`] if the coefficient is not
    /// available (e.g. not yet computed, or out of range for a form known
    /// only to finite precision).
    fn extract_coeffs(&self, which_coeff: usize) -> Result<R, ModularError>;

    /// Evaluate the form at the nome `q`.
    ///
    /// # Errors
    ///
    /// Returns [`ModularError::Unavailable`] if the value could not be
    /// computed (e.g. the `q`-expansion is only known to finite precision
    /// and `q` is too large for that truncation to give a reliable
    /// estimate).
    fn evaluate_at(&self, q: &R) -> Result<R, ModularError>;
}

/// The weight-`4` Eisenstein series `E4 = 1 + 240 * sum_{n>=1} sigma_3(n) q^n`,
/// where `sigma_3(n)` is the sum of cubes of the divisors of `n`. The
/// prototypical example of a [`ModularForm`]: it spans the (one-dimensional)
/// space of weight-`4` level-`1` forms, and is exactly the theta series of
/// the `E8` root lattice (see
/// [`Lattice::theta_series_weight`](crate::lattice::Lattice::theta_series_weight)).
#[derive(Clone, Copy)]
pub struct EisensteinE4;

#[allow(clippy::cast_precision_loss)]
impl ModularForm<8, f64> for EisensteinE4 {
    type TransformationGroup = Sl2Z;

    fn extract_coeffs(&self, which_coeff: usize) -> Result<f64, ModularError> {
        if which_coeff == 0 {
            Ok(1.0)
        } else {
            Ok(240.0 * sigma_3(which_coeff) as f64)
        }
    }

    fn evaluate_at(&self, _q: &f64) -> Result<f64, ModularError> {
        Err(ModularError::Unavailable)
    }
}

/// The weight-`6` Eisenstein series `E6 = 1 - 504 * sum_{n>=1} sigma_5(n) q^n`,
/// where `sigma_5(n)` is the sum of fifth powers of the divisors of `n`.
/// Together with [`EisensteinE4`], `E6` generates the entire ring of
/// level-`1` modular forms — every such form is a polynomial in `E4` and
/// `E6`.
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub struct EisensteinE6;

#[allow(clippy::cast_precision_loss)]
impl ModularForm<12, f64> for EisensteinE6 {
    type TransformationGroup = Sl2Z;

    fn extract_coeffs(&self, which_coeff: usize) -> Result<f64, ModularError> {
        if which_coeff == 0 {
            Ok(1.0)
        } else {
            Ok(-504.0 * sigma_5(which_coeff) as f64)
        }
    }

    fn evaluate_at(&self, _q: &f64) -> Result<f64, ModularError> {
        Err(ModularError::Unavailable)
    }
}

/// The constant modular form `f = 1`, of weight `0`. The multiplicative
/// identity once a product combinator exists (`f * g = g` for any modular
/// form `g` sharing its transformation group), and trivially modular under
/// any group whose [`multiplier_system`](ModularTransformationGroup::multiplier_system)
/// is itself trivial: `f(g*tau) = v(g) * (c*tau+d)^0 * f(tau) = 1 = f(tau)`.
#[allow(dead_code)]
pub struct UnitModularForm;

impl ModularForm<0, f64> for UnitModularForm {
    type TransformationGroup = Sl2Z;

    fn extract_coeffs(&self, which_coeff: usize) -> Result<f64, ModularError> {
        Ok(if which_coeff == 0 { 1.0 } else { 0.0 })
    }

    fn evaluate_at(&self, _q: &f64) -> Result<f64, ModularError> {
        Ok(1.0)
    }
}

/// A formal linear combination `sum_i coeff_i * summand_i` of modular forms
/// that all share the same weight and transformation group — and so is
/// itself a modular form of that weight and group, since modular forms of
/// fixed weight and group form a vector space over `R`.
#[allow(dead_code)]
pub struct SumModularForm<
    const TWICE_WEIGHT: usize,
    R: Field,
    TRANSFORM: ModularTransformationGroup<R>,
> {
    pub(crate) summands: Vec<(
        Box<dyn ModularForm<TWICE_WEIGHT, R, TransformationGroup = TRANSFORM>>,
        R,
    )>,
}

impl<const TWICE_WEIGHT: usize, R, TRANSFORM: ModularTransformationGroup<R>>
    SumModularForm<TWICE_WEIGHT, R, TRANSFORM>
where
    R: Debug + Field + 'static,
{
    /// Construct the unique element of the `DIM_SPACE`-dimensional space
    /// spanned by `basis_of_space` that satisfies a prescribed set of linear
    /// constraints.
    ///
    /// Each entry of `constrained_coeffs_values` is one constraint, with
    /// `Result` used as an ad hoc two-case enum rather than as a
    /// success/failure signal: `Ok((idx, value))` means "the `idx`-th
    /// `q`-expansion coefficient of the result is `value`", and
    /// `Err((q, value))` means "the result evaluates to `value` at the nome
    /// `q`". The first `DIM_SPACE` constraints determine a
    /// `DIM_SPACE x DIM_SPACE` linear system for the coordinates of the
    /// result in `basis_of_space`, solved here by matrix inversion.
    ///
    /// # Errors
    ///
    /// Returns [`ModularError::NotEnoughConstraints`] if fewer than
    /// `DIM_SPACE` constraints are given, [`ModularError::SingularSystem`]
    /// if the resulting linear system is singular (the constraints don't pin
    /// down a unique combination), or [`ModularError::NonFiniteCoefficient`]
    /// if a solved coefficient is not finite.
    ///
    /// # Panics
    ///
    /// Not yet implemented — always panics with `todo!`, since the linear
    /// system is never actually populated from `constrained_coeffs_values`.
    #[allow(dead_code, unused_variables, unused_mut)]
    #[allow(clippy::type_complexity)]
    pub fn new_from_some_coeffs<const DIM_SPACE: usize>(
        basis_of_space: [Box<dyn ModularForm<TWICE_WEIGHT, R, TransformationGroup = TRANSFORM>>;
            DIM_SPACE],
        constrained_coeffs_values: &[Result<(usize, R), (R, R)>],
    ) -> Result<Self, ModularError>
    where
        R: nalgebra::ComplexField,
    {
        if constrained_coeffs_values.len() < DIM_SPACE {
            return Err(ModularError::NotEnoughConstraints);
        }
        let mut matrix = SMatrix::<R, DIM_SPACE, DIM_SPACE>::zeros();
        let mut b_col_vector = SMatrix::<R, DIM_SPACE, 1>::zeros();
        todo!("Know that this modular form is in the vector space spanned by basis_of_space and
        that certain linear maps gotten from extracting coefficients and/or evaluation values evaluate to
        _constrained_coeffs_values");
        #[allow(unreachable_code)]
        let mut out = matrix.clone();
        let succeeded =
            nalgebra::try_invert_to::<R, Const<DIM_SPACE>, ArrayStorage<R, DIM_SPACE, DIM_SPACE>>(
                matrix, &mut out,
            );
        if !succeeded {
            return Err(ModularError::SingularSystem);
        }
        let w = out * b_col_vector;
        let mut summands = Vec::with_capacity(DIM_SPACE);
        for (idx, cur_summand) in basis_of_space.into_iter().enumerate() {
            let cur_w = w[idx];
            if cur_w.is_zero() {
                continue;
            }
            if !cur_w.is_finite() {
                return Err(ModularError::NonFiniteCoefficient);
            }
            let value = (cur_summand, cur_w);
            summands.push(value);
        }
        Ok(Self { summands })
    }

    /// The `which_coeff`-th `q`-expansion coefficient of the sum: each
    /// summand's coefficient, combined with the same weights as in the sum
    /// itself.
    ///
    /// # Errors
    ///
    /// Propagates the first `Err` returned by any summand's
    /// [`ModularForm::extract_coeffs`].
    #[allow(dead_code)]
    pub fn extract_coeffs(&self, which_coeff: usize) -> Result<R, ModularError> {
        let mut to_return = R::zero();
        for summand in &self.summands {
            to_return += summand.0.extract_coeffs(which_coeff)? * summand.1.clone();
        }
        Ok(to_return)
    }

    /// Evaluate the sum at the nome `q`: each summand evaluated at `q`,
    /// combined with the same weights as in the sum itself.
    ///
    /// # Errors
    ///
    /// Propagates the first `Err` returned by any summand's
    /// [`ModularForm::evaluate_at`].
    #[allow(dead_code)]
    pub fn evaluate_at(&self, q: &R) -> Result<R, ModularError> {
        let mut to_return = R::zero();
        for summand in &self.summands {
            to_return += summand.0.evaluate_at(q)? * summand.1.clone();
        }
        Ok(to_return)
    }
}

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
    pub first: F1,
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

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use crate::lattice::EquivalentTransportedMF;

    use super::{
        EisensteinE4, EisensteinE6, ModularForm, ProductModularForm, Sl2Z, SumModularForm,
        cube_modular_form, square_modular_form,
    };

    #[allow(dead_code)]
    //#[test]
    fn e8_theta_series_is_e4_eisenstein_series() {
        // E8's theta series is the weight-4 level-1 form with constant term 1
        // (see RootLatticeE8::theta_series_weight in lattice_def.rs's tests),
        // i.e. exactly E4 with coefficient 1 — the unique element of the
        // 1-dimensional space spanned by E4 alone whose constant term is 1.
        let basis_of_space: [Box<dyn ModularForm<8, f64, TransformationGroup = Sl2Z>>; 1] =
            [Box::new(EisensteinE4)];
        let constrained_coeffs_values = [Ok((0, 1.0))];

        let theta_e8 = SumModularForm::<8, f64, Sl2Z>::new_from_some_coeffs(
            basis_of_space,
            &constrained_coeffs_values,
        )
        .expect("E4 alone spans a 1-dimensional space pinned down by its constant term");

        assert_eq!(theta_e8.summands.len(), 1);
        assert_eq!(theta_e8.summands[0].1, 1.0);
        assert_eq!(theta_e8.extract_coeffs(1), Ok(240.0));
    }

    #[test]
    fn e4_sum_coeffs() {
        // A direct sanity check of SumModularForm's accumulation logic,
        // E4 + 5*E4 should just be 6*E4.
        const COEFF1: f64 = 1.0;
        const COEFF2: f64 = 5.0;
        let sum: SumModularForm<8, f64, Sl2Z> = SumModularForm {
            summands: vec![
                (
                    Box::new(EisensteinE4)
                        as Box<dyn ModularForm<8, f64, TransformationGroup = Sl2Z>>,
                    COEFF1,
                ),
                (
                    Box::new(EisensteinE4)
                        as Box<dyn ModularForm<8, f64, TransformationGroup = Sl2Z>>,
                    COEFF2,
                ),
            ],
        };

        assert_eq!(sum.extract_coeffs(0), Ok(COEFF1 + COEFF2));
        assert_eq!(sum.extract_coeffs(1), Ok((COEFF1 + COEFF2) * 240.0));
        assert_eq!(sum.extract_coeffs(2), Ok((COEFF1 + COEFF2) * 2160.0));
    }

    #[test]
    fn e4_times_e6_is_e10_eisenstein_series() {
        // The space of weight-10 level-1 modular forms is 1-dimensional, so
        // E4*E6 must be the unique such form with constant term 1: E10 =
        // 1 - 264 * sum_{n>=1} sigma_9(n) q^n. Check the first few
        // coefficients of the Cauchy product against that closed form.
        let product = ProductModularForm::<f64, EisensteinE4, EisensteinE6, 8, 12, 20>::new(
            EisensteinE4,
            EisensteinE6,
        );

        assert_eq!(product.extract_coeffs(0), Ok(1.0));
        assert_eq!(product.extract_coeffs(1), Ok(-264.0)); // -264 * sigma_9(1) = -264
        assert_eq!(product.extract_coeffs(2), Ok(-135_432.0)); // -264 * sigma_9(2) = -264 * 513
        assert_eq!(product.extract_coeffs(3), Ok(-5_196_576.0)); // -264 * sigma_9(3) = -264 * 19684
    }

    #[test]
    fn e4_cubed_and_e6_squared() {
        let e4_cubed = cube_modular_form::<f64, 8, 16, 24, EisensteinE4>(EisensteinE4);
        let e6_squared = square_modular_form::<f64, 12, 24, EisensteinE6>(EisensteinE6);

        // E4^3 = 1 + 720q + 179280q^2 + 16954560q^3 + ...
        assert_eq!(e4_cubed.extract_coeffs(0), Ok(1.0));
        assert_eq!(e4_cubed.extract_coeffs(1), Ok(720.0));
        assert_eq!(e4_cubed.extract_coeffs(2), Ok(179_280.0));
        assert_eq!(e4_cubed.extract_coeffs(3), Ok(16_954_560.0));

        let coercion = crate::lattice::CoerceTransformation::FORCED(
            PhantomData,
            PhantomData,
            PhantomData::<Sl2Z>,
        );
        let e4_cubed_fixed = EquivalentTransportedMF::coerce(e4_cubed, coercion);
        assert_eq!(e4_cubed_fixed.extract_coeffs(0), Ok(1.0));
        assert_eq!(e4_cubed_fixed.extract_coeffs(1), Ok(720.0));
        assert_eq!(e4_cubed_fixed.extract_coeffs(2), Ok(179_280.0));
        assert_eq!(e4_cubed_fixed.extract_coeffs(3), Ok(16_954_560.0));

        // E6^2 = 1 - 1008q + 220752q^2 + 16519104q^3 + ...
        assert_eq!(e6_squared.extract_coeffs(0), Ok(1.0));
        assert_eq!(e6_squared.extract_coeffs(1), Ok(-1008.0));
        assert_eq!(e6_squared.extract_coeffs(2), Ok(220_752.0));
        assert_eq!(e6_squared.extract_coeffs(3), Ok(16_519_104.0));

        let coercion = crate::lattice::CoerceTransformation::FORCED(
            PhantomData,
            PhantomData,
            PhantomData::<Sl2Z>,
        );
        let e6_squared_fixed = EquivalentTransportedMF::coerce(e6_squared, coercion);
        assert_eq!(e6_squared_fixed.extract_coeffs(0), Ok(1.0));
        assert_eq!(e6_squared_fixed.extract_coeffs(1), Ok(-1008.0));
        assert_eq!(e6_squared_fixed.extract_coeffs(2), Ok(220_752.0));
        assert_eq!(e6_squared_fixed.extract_coeffs(3), Ok(16_519_104.0));

        // Now both have the same nameable TransformationGroup (Sl2Z), so
        // they can sit together in one SumModularForm: E4^3 - E6^2 = 1728*Delta,
        // the usual identity defining the weight-12 cusp form Delta.
        let sum: SumModularForm<24, f64, Sl2Z> = SumModularForm {
            summands: vec![
                (
                    Box::new(e4_cubed_fixed)
                        as Box<dyn ModularForm<24, f64, TransformationGroup = Sl2Z>>,
                    1.0,
                ),
                (
                    Box::new(e6_squared_fixed)
                        as Box<dyn ModularForm<24, f64, TransformationGroup = Sl2Z>>,
                    -1.0,
                ),
            ],
        };

        // 1728 * Delta = 1728 * (q - 24q^2 + 252q^3 - ...)
        assert_eq!(sum.extract_coeffs(0), Ok(0.0));
        assert_eq!(sum.extract_coeffs(1), Ok(1_728.0));
        assert_eq!(sum.extract_coeffs(2), Ok(-41_472.0));
        assert_eq!(sum.extract_coeffs(3), Ok(435_456.0));
    }
}
