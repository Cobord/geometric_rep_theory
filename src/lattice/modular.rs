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
use num::rational::Ratio;
use num::{ToPrimitive, Zero};

use crate::arithmetic_utils::{Field, dedekind_sum, euler_function_coeff, sigma_3, sigma_5};

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
    /// The two [`ModularTransformationGroup`]s are not the same group (as
    /// witnessed by their raw-matrix presentations) or do not have the same
    /// multiplier system.
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
                "The two transformation groups are not the same group or do not have the same multiplier system."
            ),
        }
    }
}

impl std::error::Error for ModularError {}

/// Some group `G`, equipped with a multiplier system — a character
/// `chi: G x H -> R` (see [`multiplier_system`](Self::multiplier_system)) —
/// and an action on the upper half-plane by Mobius transformations
/// `tau -> (a*tau+b)/(c*tau+d)`, together describing how a modular form of
/// weight `w` transforms under it, always in the one form
/// `f(g*tau) = chi(g, tau) * (c*tau+d)^w * f(tau)` — never silently dropping
/// `chi` for an integer `w`, even though `chi` may in that case be
/// identically `1` (e.g. for [`Sl2Z`]). `w` itself only ever ranges over `Z`
/// or `Z + 1/2` (see [`ModularForm`], where it's represented as
/// `TWICE_WEIGHT/2` so that half-integer weights stay expressible as a
/// `usize` const generic); `chi` takes `tau` because, for a half-integer
/// `w`, `(c*tau+d)^w` itself means the *preferred* — i.e. principal-branch —
/// square root of `c*tau+d` (raised to the integer `2*w`), and correctly
/// computing `chi` for a metaplectic-type `G` (see below) can require
/// evaluating that preferred square root at the given `tau`.
///
/// `G` is *not* required to literally be a subgroup of `SL_2(Z)` (or
/// `GL_2(Z)`) — `raw_a`..`raw_d` are only the integer matrix data this trait
/// needs in order to write down the Mobius/`q`-disk action and the
/// automorphy factor `(c*tau+d)^w`, not a claim that this data identifies
/// `G` with (a subgroup of) `GL_2(Z)`. [`Sl2Z`] is the motivating case where
/// it does, with the trivial multiplier; [`EtaTransformationGroup`] is the
/// motivating case where it doesn't: its `G` is (a piece of) the metaplectic
/// double cover `Mp_2(Z)` of `SL_2(Z)`, whose elements carry a consistent
/// choice of square root of `c*tau+d` alongside `g` — a choice `SL_2(Z)`
/// alone has no way to pin down, since `g` and `-g` induce the same Mobius
/// transformation but would need opposite square roots.
///
/// `R` is how complex numbers are being parameterized here. `tau` and `q`
/// range over the upper half-plane (or the punctured disk, for `q`), so they
/// must be genuinely complex-valued; `R` is kept generic, rather than fixed
/// to one concrete representation, so that different complex-number types
/// (`Complex<f64>` vs `Complex<f32>`, say) can all be used. The matrix
/// entries `a, b, c, d` are typed as `R` too, purely so they can be combined
/// arithmetically with `tau`/`q` (e.g. via `mul_add`) — but they must
/// actually lie in the canonical image of `Z` inside `R`, not range over all
/// of `R`, since they are integer data describing an action, not general
/// scalars.
pub trait ModularTransformationGroup<R: Field>: Sized {
    /// Build the element of `G` whose Mobius/`q`-disk action and multiplier
    /// are given by the raw matrix entries `[[a, b], [c, d]]` — i.e. `new` is
    /// handed an element of `SL_2(Z)` (or `GL_2(Z)`), but produces (and
    /// `Self` ends up storing) an element of `G`. For a metaplectic-type `G`
    /// (see the trait-level docs) that matrix has two preimages in `G` under
    /// the covering surjection `G -> SL_2(Z)`, and `new` resolves it to one
    /// of them, in whatever way the implementor chooses. There is no
    /// *section* of that surjection — no way to make this choice
    /// consistently across every matrix at once as a group homomorphism,
    /// since the cover is nontrivial — but `new` never needs one: it only
    /// ever resolves a single matrix at a time, and surjectivity alone
    /// already guarantees a preimage exists for that one point.
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

    /// The character `chi(g, tau)` in the transformation law
    /// `f(g*tau) = chi(g, tau) * (c*tau+d)^w * f(tau)` for a weight-`w`
    /// modular form `f` (see the trait-level docs for why `w` and `chi` are
    /// written this way, and [`ModularForm`] for how `w` is represented).
    #[allow(dead_code)]
    fn multiplier_system(&self, tau: &R) -> R;

    /// Whether [`multiplier_system`](Self::multiplier_system) is identically
    /// `1` on `G` (e.g. [`Sl2Z`]) rather than a genuine character (e.g.
    /// [`EtaTransformationGroup`]) — a fact about this particular `G`, used
    /// where it lets a caller skip evaluating `chi` altogether, not an
    /// assumption baked into the rest of this trait.
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

/// `SL_2(Z)` itself, represented over `f64` — just enough to be a concrete
/// [`ModularTransformationGroup`] for [`EisensteinE4`] (and other weight-`k`,
/// level-`1`, trivial-character forms) to transform under.
pub struct Sl2Z {
    a: i128,
    b: i128,
    c: i128,
    d: i128,
}

impl ModularTransformationGroup<Complex<f64>> for Sl2Z {
    #[allow(clippy::many_single_char_names)]
    fn new(raw_matrix: [[Complex<f64>; 2]; 2]) -> Result<Self, ModularError> {
        const EPSILON: f64 = 1e-9;
        let [[a, b], [c, d]] = raw_matrix;
        if [a, b, c, d].into_iter().any(|x| {
            if x.im.abs() > EPSILON {
                true
            } else {
                let x = x.re;
                (x - x.round()).abs() > EPSILON
            }
        }) {
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

    fn multiplier_system(&self, _tau: &Complex<f64>) -> Complex<f64> {
        Complex::new(1.0, 0.0)
    }

    #[allow(clippy::cast_precision_loss)]
    fn raw_a(&self) -> Complex<f64> {
        Complex {
            re: self.a as f64,
            im: 0.0,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn raw_b(&self) -> Complex<f64> {
        Complex {
            re: self.b as f64,
            im: 0.0,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn raw_c(&self) -> Complex<f64> {
        Complex {
            re: self.c as f64,
            im: 0.0,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn raw_d(&self) -> Complex<f64> {
        Complex {
            re: self.d as f64,
            im: 0.0,
        }
    }

    fn is_trivial_multiplier_system() -> bool {
        true
    }
}

/// The metaplectic double cover `Mp_2(Z)` of `SL_2(Z)`, over `Complex<f64>`,
/// carrying the Dedekind eta function's character `chi`: eta is weight
/// `1/2`, so it needs genuinely complex `tau`/`q` (see the
/// [`ModularTransformationGroup`] trait-level docs on why `R` is generic)
/// and a nontrivial `chi` — unlike [`Sl2Z`], which `Self` otherwise
/// resembles in that both store a raw integer matrix.
///
/// [`new`](Self::new) accepts every determinant-`1` integer matrix,
/// including `-g` for any accepted `g`. `g` and `-g` induce the same Mobius
/// transformation but need different values of `chi` — this is exactly why
/// `Self` is genuinely metaplectic rather than just `SL_2(Z)`, and is the
/// reason [`multiplier_system`](ModularTransformationGroup::multiplier_system)
/// takes `tau`: `chi(g, tau)` is computed directly from a Dedekind sum for
/// `c > 0` (see there), but for `c <= 0` it's instead computed from
/// `chi(-g, tau)` via the ratio of the *preferred* (principal-branch) square
/// roots of `-(c*tau+d)` and `c*tau+d` — a ratio that is providably
/// independent of `tau` (`c*tau+d` never crosses the principal branch's cut
/// as `tau` ranges over `H`, for fixed integer `c`, `d`), but is safer to
/// evaluate at the caller's own `tau` than to hard-code by hand.
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

    /// The eta character `chi(g, tau)` in
    /// `eta(g*tau) = chi(g, tau) * (c*tau+d)^(1/2) * eta(tau)`, a `24`th root
    /// of unity given by Dedekind sums.
    ///
    /// For `c > 0`:
    ///
    /// `chi(g, tau) = exp(i*pi * ((a+d)/(12*c) - s(d,c) - 1/4))`
    ///
    /// where `s(h,k)` is the Dedekind sum, computed exactly by
    /// [`dedekind_sum`](crate::arithmetic_utils::dedekind_sum) (see there for
    /// its definition) — independent of `tau`, as promised by
    /// [`ModularTransformationGroup::is_trivial_multiplier_system`] being
    /// beside the point here (`chi` is a genuine, nontrivial character, but
    /// still a character *of `g` alone*, not of `(g, tau)` jointly; `tau` is
    /// only a parameter because *resolving* it below needs one).
    ///
    /// For `c = 0`: then `ad = 1`, forcing `a = d = 1` or `a = d = -1`. If
    /// `a = d = 1` (`g = T^b` for the generator `T = (1 1; 0 1)`):
    ///
    /// `chi(g, tau) = exp(i*pi*b/12)`
    ///
    /// Sanity-checked against the two generators of `SL_2(Z)`: `S = (0 -1;
    /// 1 0)` has `c = 1`, `s(0,1) = 0` (empty sum), so `chi(S, tau)` comes
    /// out to `exp(-i*pi/4)`, matching the classical identity
    /// `eta(-1/tau) = sqrt(-i*tau) * eta(tau)`; `T = (1 1; 0 1)` is the
    /// `c = 0` case with `b = 1`, so `chi(T, tau)` is `exp(i*pi/12)`,
    /// matching `eta(tau+1) = exp(i*pi/12) * eta(tau)`.
    ///
    /// Otherwise (`c < 0`, or `c = 0` with `a = d = -1`): `-g` falls into one
    /// of the above cases, and `eta((-g)*tau) = eta(g*tau)` (same Mobius
    /// transformation), so `chi(g, tau)` is recovered from `chi(-g, tau)` by
    /// matching up the two automorphy factors,
    /// `chi(g, tau) * sqrt(c*tau+d) = chi(-g, tau) * sqrt(-c*tau-d)`, i.e.
    /// `chi(g, tau) = chi(-g, tau) * sqrt(-(c*tau+d)) / sqrt(c*tau+d)`, both
    /// square roots taken with the preferred (principal) branch.
    fn multiplier_system(&self, tau: &Complex<f64>) -> Complex<f64> {
        let Self { a, b, c, d } = *self;
        if c < 0 || (c == 0 && a == -1) {
            let flipped = Self {
                a: -a,
                b: -b,
                c: -c,
                d: -d,
            };
            let chi_flipped = flipped.multiplier_system(tau);
            #[allow(clippy::cast_precision_loss)]
            let z = Complex::new(c as f64, 0.0) * tau + Complex::new(d as f64, 0.0);
            return chi_flipped * (-z).sqrt() / z.sqrt();
        }
        if c == 0 {
            // Only a == d == 1 remains, per the branch just above.
            #[allow(clippy::cast_precision_loss)]
            let angle = std::f64::consts::PI * b as f64 / 12.0;
            return Complex::new(angle.cos(), angle.sin());
        }
        let x = Ratio::new(a + d, 12 * c) - dedekind_sum(d, c) - Ratio::new(1, 4);
        let angle = std::f64::consts::PI * x.to_f64().expect("Ratio<i128> fits in f64");
        Complex::new(angle.cos(), angle.sin())
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

    /// The `which_coeff`-th coefficient of this form's `q`-expansion — i.e.
    /// the coefficient of `q^(which_coeff + kappa)`, where `kappa` is
    /// whatever fractional shift is forced by [`Self::TransformationGroup`]'s
    /// multiplier at `T = (1 1; 0 1)` (`v(T) = exp(2*pi*i*kappa)`). `kappa`
    /// is `0` for a trivial multiplier (e.g. [`Sl2Z`], where `which_coeff`
    /// really is just the power of `q`), but nonzero for a genuinely
    /// half-integer-weight group: [`EtaTransformationGroup`]'s multiplier
    /// gives `kappa = 1/24`, so [`DedekindEta::extract_coeffs`]'s
    /// `which_coeff` indexes the coefficient of `q^(which_coeff + 1/24)`,
    /// not of `q^which_coeff` itself.
    ///
    /// This isn't a per-implementor choice: `kappa` is pinned down by
    /// `TransformationGroup` alone, so every [`ModularForm`] sharing a given
    /// `TransformationGroup` must index `which_coeff` against the *same*
    /// `kappa` — which is exactly what lets combinators like
    /// [`SumModularForm`] add up `which_coeff`-th coefficients across
    /// summands term-by-term and have the result mean the same thing.
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
impl ModularForm<8, Complex<f64>> for EisensteinE4 {
    type TransformationGroup = Sl2Z;

    fn extract_coeffs(&self, which_coeff: usize) -> Result<Complex<f64>, ModularError> {
        if which_coeff == 0 {
            Ok(Complex::new(1.0, 0.0))
        } else {
            let re_ans = 240.0 * sigma_3(which_coeff) as f64;
            Ok(Complex::new(re_ans, 0.0))
        }
    }

    fn evaluate_at(&self, _q: &Complex<f64>) -> Result<Complex<f64>, ModularError> {
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
impl ModularForm<12, Complex<f64>> for EisensteinE6 {
    type TransformationGroup = Sl2Z;

    fn extract_coeffs(&self, which_coeff: usize) -> Result<Complex<f64>, ModularError> {
        if which_coeff == 0 {
            Ok(Complex { re: 1.0, im: 0.0 })
        } else {
            let re_ans = -504.0 * sigma_5(which_coeff) as f64;
            Ok(Complex {
                re: re_ans,
                im: 0.0,
            })
        }
    }

    fn evaluate_at(&self, _q: &Complex<f64>) -> Result<Complex<f64>, ModularError> {
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

impl ModularForm<0, Complex<f64>> for UnitModularForm {
    type TransformationGroup = Sl2Z;

    fn extract_coeffs(&self, which_coeff: usize) -> Result<Complex<f64>, ModularError> {
        Ok(if which_coeff == 0 {
            Complex { re: 1.0, im: 0.0 }
        } else {
            Complex::zero()
        })
    }

    fn evaluate_at(&self, _q: &Complex<f64>) -> Result<Complex<f64>, ModularError> {
        Ok(Complex { re: 1.0, im: 0.0 })
    }
}

/// The Dedekind eta function `eta(tau) = q^{1/24} * prod_{n>=1} (1 - q^n)`,
/// `q = exp(2*pi*i*tau)` — the prototypical weight-`1/2` modular form: it
/// transforms under [`EtaTransformationGroup`] with a genuinely nontrivial
/// `24`th-root-of-unity multiplier, unlike every other [`ModularForm`] in
/// this module.
///
/// [`ModularForm::extract_coeffs`]'s `which_coeff` is, per that method's docs,
/// the coefficient of `q^(which_coeff + kappa)` for whatever `kappa` is
/// forced by [`EtaTransformationGroup`]'s multiplier at `T`; here
/// `kappa = 1/24`, so `which_coeff` indexes the coefficients of
/// `prod_{n>=1}(1 - q^n)` (see
/// [`euler_function_coeff`](crate::arithmetic_utils::euler_function_coeff))
/// directly, with the `q^{1/24}` prefactor accounted for entirely by `kappa`
/// rather than folded into any particular `which_coeff`.
#[allow(dead_code)]
#[derive(Clone, Copy)]
pub struct DedekindEta;

impl ModularForm<1, Complex<f64>> for DedekindEta {
    type TransformationGroup = EtaTransformationGroup;

    #[allow(clippy::cast_precision_loss)]
    fn extract_coeffs(&self, which_coeff: usize) -> Result<Complex<f64>, ModularError> {
        Ok(Complex::new(euler_function_coeff(which_coeff) as f64, 0.0))
    }

    fn evaluate_at(&self, _q: &Complex<f64>) -> Result<Complex<f64>, ModularError> {
        Err(ModularError::Unavailable)
    }
}

/// A formal linear combination `sum_i coeff_i * summand_i` of modular forms
/// that all share the same weight and transformation group — and so is
/// itself a modular form of that weight and group, since modular forms of
/// fixed weight and group form a vector space over `R`. Sharing a
/// `TransformationGroup` also means every summand indexes
/// [`ModularForm::extract_coeffs`]'s `which_coeff` against the same
/// fractional shift `kappa` (see that method's docs), so adding
/// `which_coeff`-th coefficients across summands, as
/// [`Self::extract_coeffs`] does, is combining like terms rather than
/// nonsense.
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
    /// # Example
    ///
    /// The weight-12 cusp form `Delta` lives in the 2-dimensional space
    /// spanned by `E4^3` and `E6^2`. Its `q`-expansion is `q - 24q^2 + ...`,
    /// i.e. it is cut out among combinations of that basis by having
    /// vanishing constant term and unit coefficient of `q`:
    /// `Delta = w_0 * E4^3 + w_1 * E6^2` where `w_0, w_1` solve
    /// `w_0 + w_1 = 0` (constant term) and `720*w_0 - 1008*w_1 = 1`
    /// (`q`-coefficient), giving `w_0 = 1/1728`, `w_1 = -1/1728` — the usual
    /// identity `Delta = (E4^3 - E6^2) / 1728`. Passing
    /// `[Ok((0, 0.0)), Ok((1, 1.0))]` as `constrained_coeffs_values` with
    /// `basis_of_space = [Box::new(e4_cubed), Box::new(e6_squared)]`
    /// reconstructs exactly this `Delta` as a `SumModularForm`.
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
        for idx in 0..DIM_SPACE {
            for jdx in 0..DIM_SPACE {
                matrix[(idx, jdx)] = match &constrained_coeffs_values[idx] {
                    Ok((which_coeff, _)) => basis_of_space[jdx].extract_coeffs(*which_coeff)?,
                    Err((q, _)) => basis_of_space[jdx].evaluate_at(q)?,
                };
            }
        }
        for idx in 0..DIM_SPACE {
            b_col_vector[(idx, 0)] = match &constrained_coeffs_values[idx] {
                Ok((_, value)) | Err((_, value)) => value.clone(),
            };
        }
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
            if w[idx].is_zero() {
                continue;
            }
            if !w[idx].is_finite() {
                return Err(ModularError::NonFiniteCoefficient);
            }
            let value = (cur_summand, w[idx].clone());
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

    use nalgebra::Complex;

    use super::{
        DedekindEta, EisensteinE4, EisensteinE6, EtaTransformationGroup, ModularForm,
        ModularTransformationGroup, ProductModularForm, Sl2Z, SumModularForm, cube_modular_form,
        square_modular_form,
    };

    #[test]
    fn e8_theta_series_is_e4_eisenstein_series() {
        // E8's theta series is the weight-4 level-1 form with constant term 1
        // (see RootLatticeE8::theta_series_weight in lattice_def.rs's tests),
        // i.e. exactly E4 with coefficient 1 — the unique element of the
        // 1-dimensional space spanned by E4 alone whose constant term is 1.
        let basis_of_space: [Box<dyn ModularForm<8, Complex<f64>, TransformationGroup = Sl2Z>>; 1] =
            [Box::new(EisensteinE4)];
        let constrained_coeffs_values = [Ok((0, Complex { re: 1.0, im: 0.0 }))];

        let theta_e8 = SumModularForm::<8, Complex<f64>, Sl2Z>::new_from_some_coeffs(
            basis_of_space,
            &constrained_coeffs_values,
        )
        .expect("E4 alone spans a 1-dimensional space pinned down by its constant term");

        assert_eq!(theta_e8.summands.len(), 1);
        assert_eq!(theta_e8.summands[0].1, Complex { re: 1.0, im: 0.0 });
        assert_eq!(
            theta_e8.extract_coeffs(1),
            Ok(Complex { re: 240.0, im: 0.0 })
        );
    }

    #[test]
    fn e4_sum_coeffs() {
        // A direct sanity check of SumModularForm's accumulation logic,
        // E4 + 5*E4 should just be 6*E4.
        const COEFF1: Complex<f64> = Complex { re: 1.0, im: 0.0 };
        const COEFF2: Complex<f64> = Complex { re: 5.0, im: 0.0 };
        let sum: SumModularForm<8, Complex<f64>, Sl2Z> = SumModularForm {
            summands: vec![
                (
                    Box::new(EisensteinE4)
                        as Box<dyn ModularForm<8, Complex<f64>, TransformationGroup = Sl2Z>>,
                    COEFF1,
                ),
                (
                    Box::new(EisensteinE4)
                        as Box<dyn ModularForm<8, Complex<f64>, TransformationGroup = Sl2Z>>,
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
        let product =
            ProductModularForm::<Complex<f64>, EisensteinE4, EisensteinE6, 8, 12, 20>::new(
                EisensteinE4,
                EisensteinE6,
            );

        assert_eq!(product.extract_coeffs(0), Ok(Complex::new(1.0, 0.0)));
        // -264 * sigma_9(1) = -264
        assert_eq!(product.extract_coeffs(1), Ok(Complex::new(-264.0, 0.0)));
        // -264 * sigma_9(2) = -264 * 513
        assert_eq!(product.extract_coeffs(2), Ok(Complex::new(-135_432.0, 0.0)));
        // -264 * sigma_9(3) = -264 * 19684
        assert_eq!(
            product.extract_coeffs(3),
            Ok(Complex::new(-5_196_576.0, 0.0))
        );
    }

    #[test]
    fn e4_cubed_and_e6_squared() {
        let e4_cubed = cube_modular_form::<Complex<f64>, 8, 16, 24, EisensteinE4>(EisensteinE4);
        let e6_squared = square_modular_form::<Complex<f64>, 12, 24, EisensteinE6>(EisensteinE6);

        // E4^3 = 1 + 720q + 179280q^2 + 16954560q^3 + 396974160q^4 + ...
        assert_eq!(e4_cubed.extract_coeffs(0), Ok(Complex::new(1.0, 0.0)));
        assert_eq!(e4_cubed.extract_coeffs(1), Ok(Complex::new(720.0, 0.0)));
        assert_eq!(e4_cubed.extract_coeffs(2), Ok(Complex::new(179_280.0, 0.0)));
        assert_eq!(
            e4_cubed.extract_coeffs(3),
            Ok(Complex::new(16_954_560.0, 0.0))
        );
        assert_eq!(
            e4_cubed.extract_coeffs(4),
            Ok(Complex::new(396_974_160.0, 0.0))
        );

        let coercion = crate::lattice::CoerceTransformation::FORCED(
            PhantomData,
            PhantomData,
            PhantomData::<Sl2Z>,
        );
        let e4_cubed_fixed = EquivalentTransportedMF::coerce(e4_cubed, coercion);
        assert_eq!(e4_cubed_fixed.extract_coeffs(0), Ok(Complex::new(1.0, 0.0)));
        assert_eq!(
            e4_cubed_fixed.extract_coeffs(1),
            Ok(Complex::new(720.0, 0.0))
        );
        assert_eq!(
            e4_cubed_fixed.extract_coeffs(2),
            Ok(Complex::new(179_280.0, 0.0))
        );
        assert_eq!(
            e4_cubed_fixed.extract_coeffs(3),
            Ok(Complex::new(16_954_560.0, 0.0))
        );
        assert_eq!(
            e4_cubed_fixed.extract_coeffs(4),
            Ok(Complex::new(396_974_160.0, 0.0))
        );

        // E6^2 = 1 - 1008q + 220752q^2 + 16519104q^3 + 399517776q^4 + ...
        assert_eq!(e6_squared.extract_coeffs(0), Ok(Complex::new(1.0, 0.0)));
        assert_eq!(e6_squared.extract_coeffs(1), Ok(Complex::new(-1008.0, 0.0)));
        assert_eq!(
            e6_squared.extract_coeffs(2),
            Ok(Complex::new(220_752.0, 0.0))
        );
        assert_eq!(
            e6_squared.extract_coeffs(3),
            Ok(Complex::new(16_519_104.0, 0.0))
        );
        assert_eq!(
            e6_squared.extract_coeffs(4),
            Ok(Complex::new(399_517_776.0, 0.0))
        );

        let coercion = crate::lattice::CoerceTransformation::FORCED(
            PhantomData,
            PhantomData,
            PhantomData::<Sl2Z>,
        );
        let e6_squared_fixed = EquivalentTransportedMF::coerce(e6_squared, coercion);
        assert_eq!(
            e6_squared_fixed.extract_coeffs(0),
            Ok(Complex::new(1.0, 0.0))
        );
        assert_eq!(
            e6_squared_fixed.extract_coeffs(1),
            Ok(Complex::new(-1008.0, 0.0))
        );
        assert_eq!(
            e6_squared_fixed.extract_coeffs(2),
            Ok(Complex::new(220_752.0, 0.0))
        );
        assert_eq!(
            e6_squared_fixed.extract_coeffs(3),
            Ok(Complex::new(16_519_104.0, 0.0))
        );
        assert_eq!(
            e6_squared_fixed.extract_coeffs(4),
            Ok(Complex::new(399_517_776.0, 0.0))
        );

        // Now both have the same nameable TransformationGroup (Sl2Z), so
        // they can sit together in one SumModularForm: E4^3 - E6^2 = 1728*Delta,
        // the usual identity defining the weight-12 cusp form Delta.
        let sum: SumModularForm<24, Complex<f64>, Sl2Z> = SumModularForm {
            summands: vec![
                (
                    Box::new(e4_cubed_fixed)
                        as Box<dyn ModularForm<24, Complex<f64>, TransformationGroup = Sl2Z>>,
                    Complex::new(1.0, 0.0),
                ),
                (
                    Box::new(e6_squared_fixed)
                        as Box<dyn ModularForm<24, Complex<f64>, TransformationGroup = Sl2Z>>,
                    Complex::new(-1.0, 0.0),
                ),
            ],
        };

        // 1728 * Delta = 1728 * (q - 24q^2 + 252q^3 - 1472q^4 + ...)
        assert_eq!(sum.extract_coeffs(0), Ok(Complex::new(0.0, 0.0)));
        assert_eq!(sum.extract_coeffs(1), Ok(Complex::new(1_728.0, 0.0)));
        assert_eq!(sum.extract_coeffs(2), Ok(Complex::new(-41_472.0, 0.0)));
        assert_eq!(sum.extract_coeffs(3), Ok(Complex::new(435_456.0, 0.0)));
        assert_eq!(sum.extract_coeffs(4), Ok(Complex::new(-2_543_616.0, 0.0)));
    }

    #[test]
    fn new_from_some_coeffs_reconstructs_delta_from_e4_cubed_and_e6_squared() {
        // The doc example on `new_from_some_coeffs`: the weight-12 cusp form
        // Delta = q - 24q^2 + 252q^3 - ... is the unique element of the
        // 2-dimensional space spanned by E4^3 and E6^2 with vanishing
        // constant term and unit q-coefficient. Solving for those two
        // constraints should recover the classic identity
        // Delta = (E4^3 - E6^2) / 1728.
        let e4_cubed = cube_modular_form::<Complex<f64>, 8, 16, 24, EisensteinE4>(EisensteinE4);
        let e6_squared = square_modular_form::<Complex<f64>, 12, 24, EisensteinE6>(EisensteinE6);

        let coercion = crate::lattice::CoerceTransformation::FORCED(
            PhantomData,
            PhantomData,
            PhantomData::<Sl2Z>,
        );
        let e4_cubed_fixed = EquivalentTransportedMF::coerce(e4_cubed, coercion);
        let coercion = crate::lattice::CoerceTransformation::FORCED(
            PhantomData,
            PhantomData,
            PhantomData::<Sl2Z>,
        );
        let e6_squared_fixed = EquivalentTransportedMF::coerce(e6_squared, coercion);

        let basis_of_space: [Box<dyn ModularForm<24, Complex<f64>, TransformationGroup = Sl2Z>>;
            2] = [Box::new(e4_cubed_fixed), Box::new(e6_squared_fixed)];
        // Constant term is 0, q-coefficient is 1.
        let constrained_coeffs_values = [
            Ok((0, Complex::new(0.0, 0.0))),
            Ok((1, Complex::new(1.0, 0.0))),
        ];

        let delta = SumModularForm::<24, Complex<f64>, Sl2Z>::new_from_some_coeffs(
            basis_of_space,
            &constrained_coeffs_values,
        )
        .expect("E4^3 and E6^2 span a 2-dimensional space pinned down by these two constraints");

        const ONE_OVER_1728: f64 = 1.0 / 1728.0;
        assert_eq!(delta.summands.len(), 2);
        assert!((delta.summands[0].1 - ONE_OVER_1728).norm() < 1e-9);
        assert!((delta.summands[1].1 + ONE_OVER_1728).norm() < 1e-9);

        // Delta = q - 24q^2 + 252q^3 - 1472q^4 + ...
        assert!((delta.extract_coeffs(0).unwrap() - 0.0).norm() < 1e-9);
        assert!((delta.extract_coeffs(1).unwrap() - 1.0).norm() < 1e-9);
        assert!((delta.extract_coeffs(2).unwrap() - (-24.0)).norm() < 1e-9);
        assert!((delta.extract_coeffs(3).unwrap() - 252.0).norm() < 1e-9);
        assert!((delta.extract_coeffs(4).unwrap() - (-1472.0)).norm() < 1e-9);
    }

    #[test]
    fn new_from_some_coeffs_scales_eta_to_hit_a_target_coefficient() {
        // The space spanned by DedekindEta alone is 1-dimensional, so pinning
        // down "the which_coeff=7 coefficient of prod(1-q^n) is 39" (7 being
        // a generalized pentagonal number, k=-2, so euler_function_coeff(7)
        // = 1) should recover w = 39, i.e. the result is exactly
        // 39 * DedekindEta.
        let basis_of_space: [Box<
            dyn ModularForm<1, Complex<f64>, TransformationGroup = EtaTransformationGroup>,
        >; 1] = [Box::new(DedekindEta)];
        let constrained_coeffs_values = [Ok((7, Complex::new(39.0, 0.0)))];

        let scaled_eta = SumModularForm::<1, Complex<f64>, EtaTransformationGroup>::new_from_some_coeffs(
            basis_of_space,
            &constrained_coeffs_values,
        )
        .expect("DedekindEta alone spans a 1-dimensional space pinned down by a nonzero coefficient");

        assert_eq!(scaled_eta.summands.len(), 1);
        assert_eq!(scaled_eta.summands[0].1, Complex::new(39.0, 0.0));
        // euler_function_coeff(7) = 1, so the target coefficient is hit exactly.
        assert_eq!(scaled_eta.extract_coeffs(7), Ok(Complex::new(39.0, 0.0)));
        // euler_function_coeff(9) = 0 (9 isn't a generalized pentagonal
        // number), so every other-indexed coefficient scales the same way:
        // still 0 regardless of the multiple.
        assert_eq!(scaled_eta.extract_coeffs(9), Ok(Complex::new(0.0, 0.0)));
    }

    fn eta_group_from_ints(a: i128, b: i128, c: i128, d: i128) -> EtaTransformationGroup {
        let to_complex = |x: i128| Complex::new(x as f64, 0.0);
        EtaTransformationGroup::new([
            [to_complex(a), to_complex(b)],
            [to_complex(c), to_complex(d)],
        ])
        .expect("a*d - b*c == 1")
    }

    /// A handful of unremarkable points in `H`, used across these tests to
    /// demonstrate that `multiplier_system`'s result doesn't actually depend
    /// on which `tau` it's given — nothing here is special about any one of
    /// them (in particular, none is `i` or another elliptic/cusp point).
    fn generic_taus() -> [Complex<f64>; 2] {
        [Complex::new(0.37, 1.62), Complex::new(-1.1, 0.83)]
    }

    #[test]
    fn eta_multiplier_at_t_generator() {
        // T = (1 1; 0 1): eta(tau+1) = exp(i*pi/12) * eta(tau).
        let t = eta_group_from_ints(1, 1, 0, 1);
        let expected = Complex::new(0.0, std::f64::consts::PI / 12.0).exp();
        for tau in generic_taus() {
            let eps_t = t.multiplier_system(&tau);
            assert!((eps_t - expected).norm() < 1e-9);
        }
    }

    #[test]
    fn eta_multiplier_at_s_generator() {
        // S = (0 -1; 1 0): eta(-1/tau) = sqrt(-i*tau) * eta(tau), so
        // eps(S) = exp(-i*pi/4).
        let s = eta_group_from_ints(0, -1, 1, 0);
        let expected = Complex::new(0.0, -std::f64::consts::PI / 4.0).exp();
        for tau in generic_taus() {
            let eps_s = s.multiplier_system(&tau);
            assert!((eps_s - expected).norm() < 1e-9);
        }
    }

    #[test]
    fn eta_multiplier_at_negated_s_generator() {
        // -S = (0 1; -1 0) induces the same Mobius transformation as S
        // (-1/tau), but is a genuinely different element of Mp_2(Z): its
        // character is chi(S,tau) * sqrt(-(c*tau+d))/sqrt(c*tau+d), which
        // works out to exp(-i*pi/4) * i = exp(i*pi/4) at every tau.
        let neg_s = eta_group_from_ints(0, 1, -1, 0);
        let expected = Complex::new(0.0, std::f64::consts::PI / 4.0).exp();
        for tau in generic_taus() {
            let eps = neg_s.multiplier_system(&tau);
            assert!((eps - expected).norm() < 1e-9);
        }
    }

    #[test]
    fn eta_multiplier_at_negated_t_generator() {
        // -T = (-1 -1; 0 -1) (g = -T^1): c = 0 with a = d = -1, the case an
        // earlier version of new() rejected. Its automorphy factor is
        // sqrt(0*tau + (-1)) = sqrt(-1) = i (principal branch), so
        // chi(-T,tau) * i = eta((-T)*tau)/eta(tau) = eta(tau+1)/eta(tau)
        // = exp(i*pi/12), giving chi(-T,tau) = exp(i*pi/12) / i
        // = exp(-i*5*pi/12) at every tau (c = 0 makes c*tau+d = d constant).
        let neg_t = eta_group_from_ints(-1, -1, 0, -1);
        let expected = Complex::new(0.0, -5.0 * std::f64::consts::PI / 12.0).exp();
        for tau in generic_taus() {
            let eps = neg_t.multiplier_system(&tau);
            assert!((eps - expected).norm() < 1e-9);
        }
    }
}
