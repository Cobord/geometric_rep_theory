use nalgebra::Complex;
use num::ToPrimitive;
use num::rational::Ratio;

use crate::arithmetic_utils::{dedekind_sum, euler_function_coeff};

use super::modular_def::{ModularError, ModularForm, ModularTransformationGroup};

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
