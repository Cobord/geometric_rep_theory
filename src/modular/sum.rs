use nalgebra::{ArrayStorage, Const, SMatrix};
use std::fmt::Debug;

use super::modular_def::{ModularError, ModularForm, ModularTransformationGroup};
use crate::arithmetic_utils::Field;

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
