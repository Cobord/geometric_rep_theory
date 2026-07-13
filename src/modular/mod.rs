pub mod coerced;
pub mod eisenstein;
pub mod eta;
pub mod modular_def;
pub mod product;
pub mod sum;

pub use coerced::{CoerceTransformation, EquivalentTransportedMF};
pub use eisenstein::{EisensteinE4, EisensteinE6};
pub use eta::{DedekindEta, EtaTransformationGroup};
pub use modular_def::{
    ModularError, ModularForm, ModularTransformationGroup, Sl2Z, UnitModularForm,
};
pub use product::{
    CombinedTransformationGroup, ProductModularForm, cube_modular_form, cube_modular_form_trivial,
    square_modular_form, square_modular_form_trivial,
};
pub use sum::SumModularForm;

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use super::EquivalentTransportedMF;

    use nalgebra::Complex;

    use super::{
        DedekindEta, EisensteinE4, EisensteinE6, EtaTransformationGroup, ModularForm,
        ModularTransformationGroup, ProductModularForm, Sl2Z, SumModularForm, cube_modular_form,
        cube_modular_form_trivial, square_modular_form, square_modular_form_trivial,
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

        let coercion =
            super::CoerceTransformation::FORCED(PhantomData, PhantomData, PhantomData::<Sl2Z>);
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

        let coercion =
            super::CoerceTransformation::FORCED(PhantomData, PhantomData, PhantomData::<Sl2Z>);
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
    fn e4_cubed_minus_e6_squared_via_trivial_variants_is_1728_delta() {
        // Same identity as `e4_cubed_and_e6_squared`, but built with
        // `cube_modular_form_trivial`/`square_modular_form_trivial`: since
        // `EisensteinE4`/`EisensteinE6` both transform under `Sl2Z`, whose
        // multiplier is trivial, the cube/square land directly on
        // `TransformationGroup = Sl2Z`, with no manual
        // `EquivalentTransportedMF::coerce` step needed at the call site.
        let e4_cubed =
            cube_modular_form_trivial::<Complex<f64>, 8, 16, 24, EisensteinE4>(EisensteinE4);
        let e6_squared =
            square_modular_form_trivial::<Complex<f64>, 12, 24, EisensteinE6>(EisensteinE6);

        let sum: SumModularForm<24, Complex<f64>, Sl2Z> = SumModularForm {
            summands: vec![
                (
                    Box::new(e4_cubed)
                        as Box<dyn ModularForm<24, Complex<f64>, TransformationGroup = Sl2Z>>,
                    Complex::new(1.0, 0.0),
                ),
                (
                    Box::new(e6_squared)
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
    #[should_panic(expected = "trivial multiplier system")]
    fn square_modular_form_trivial_panics_for_nontrivial_multiplier() {
        // DedekindEta transforms under EtaTransformationGroup, whose
        // multiplier is genuinely nontrivial, so the coercion this function
        // performs would be unsound: it must panic rather than silently
        // producing a form mistyped as `TransformationGroup = EtaTransformationGroup`
        // that doesn't actually behave that way.
        let _ = square_modular_form_trivial::<Complex<f64>, 1, 2, DedekindEta>(DedekindEta);
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

        let coercion =
            super::CoerceTransformation::FORCED(PhantomData, PhantomData, PhantomData::<Sl2Z>);
        let e4_cubed_fixed = EquivalentTransportedMF::coerce(e4_cubed, coercion);
        let coercion =
            super::CoerceTransformation::FORCED(PhantomData, PhantomData, PhantomData::<Sl2Z>);
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
