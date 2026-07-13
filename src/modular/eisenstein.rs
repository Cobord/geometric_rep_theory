use nalgebra::Complex;

use crate::arithmetic_utils::{sigma_3, sigma_5};

use super::modular_def::{ModularError, ModularForm, Sl2Z};

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
