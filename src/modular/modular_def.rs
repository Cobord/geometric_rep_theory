//! Modular forms, and the matrix groups they transform under.
//!
//! A lattice's theta series (see
//! [`Lattice::theta_series_weight`](crate::lattice::Lattice::theta_series_weight)
//! and
//! [`Lattice::theta_series_character`](crate::lattice::Lattice::theta_series_character))
//! is a motivating example of a [`ModularForm`]: this module models the
//! transformation law itself, independently of any particular lattice.

use std::fmt;
use std::fmt::Debug;

use nalgebra::Complex;
use num::Zero;

use crate::arithmetic_utils::Field;

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
