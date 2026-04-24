use super::errors::InfinityAlgebraError;
use crate::arithmetic_utils::Ring;
use crate::infinity_algebra::exterior_power::ExteriorPower;
use crate::infinity_algebra::graded_module::GradedModule;
use crate::infinity_algebra::operation_tree::OperationTree;

/// A strongly homotopy Lie algebra (L∞-algebra).
///
/// An L∞-algebra is a graded module equipped with a sequence of graded-antisymmetric
/// multilinear operations `l_n` for `n > 0` satisfying the homotopy Jacobi identities.
/// Concretely, for each `n` the operation `l_n` has arity `n` and degree `2 - n`, and
/// the failure of the Jacobi identity for `l_2` is controlled by `l_3`, and so on up
/// the tower.
///
/// The low-arity operations have standard interpretations:
/// - `l_1`: the differential
/// - `l_2`: a Lie bracket up to homotopy
/// - `l_3`: the Jacobiator, measuring the failure of the Jacobi identity for `l_2`
///
/// Implementors must provide [`l_n_one_term`] and [`l_n_one_term_owned`] for a single
/// pure wedge input. The default methods build on them:
/// - [`l_n_slice`]: dynamic-arity dispatch, used internally for tree evaluation
/// - [`l_n`]: evaluates a full linear combination at fixed arity `N`
/// - [`evaluate`]: evaluates a composition of operations described by an [`OperationTree`]
pub trait LInfinityAlgebra<Coeffs: Ring>
where
    Self: GradedModule<Coeffs>,
{
    /// Apply the `N`-ary operation `l_N` to a single pure wedge in `∧^N A`.
    /// Implementors are responsible for enforcing graded-antisymmetry.
    fn l_n_one_term_owned<const N: usize>(inputs: [Self; N]) -> Self;

    /// Apply the `N`-ary operation `l_N` to a single pure wedge in `∧^N A`.
    /// Implementors are responsible for enforcing graded-antisymmetry.
    fn l_n_one_term<const N: usize>(inputs: [&Self; N]) -> Self;

    /// The largest `n` for which `l_n` may be nonzero, or `None` if the algebra
    /// has operations of unbounded arity.
    ///
    /// When `l_n_slice` receives arity `n > 8` and `n` exceeds this bound, it
    /// returns `Ok(zero)` instead of `Err(ArityTooLarge)`.
    #[must_use = "Avoid trying to compute the higher coherences which are definitively 0"]
    fn max_nonzero_arity() -> Option<usize> {
        None
    }

    /// Dynamic-arity dispatch to [`l_n_one_term`]. Override if your algebra uses
    /// arities above 8.
    ///
    /// # Errors
    ///
    /// - [`InfinityAlgebraError::CurvedAlgebra`] if `inputs` is empty (arity 0).
    /// - [`InfinityAlgebraError::ArityTooLarge`] if `inputs.len()` exceeds the
    ///   default dispatch ceiling of 8 and does not exceed [`max_nonzero_arity`].
    fn l_n_slice(inputs: &[Self]) -> Result<Self, InfinityAlgebraError> {
        let n = inputs.len();
        if let Some(bound) = Self::max_nonzero_arity()
            && n > bound
        {
            // n > bound >= 0, so n >= 1 and inputs[0] exists
            return Ok(Self::zero(inputs[0].ctx()));
        }
        match n {
            0 => Err(InfinityAlgebraError::CurvedAlgebra),
            1 => {
                let inputs_const: [&Self; 1] = <&[Self; 1]>::try_from(inputs)
                    .expect("length checked by match arm")
                    .each_ref();
                Ok(Self::l_n_one_term(inputs_const))
            }
            2 => {
                let inputs_const: [&Self; 2] = <&[Self; 2]>::try_from(inputs)
                    .expect("length checked by match arm")
                    .each_ref();
                Ok(Self::l_n_one_term(inputs_const))
            }
            3 => {
                let inputs_const: [&Self; 3] = <&[Self; 3]>::try_from(inputs)
                    .expect("length checked by match arm")
                    .each_ref();
                Ok(Self::l_n_one_term(inputs_const))
            }
            4 => {
                let inputs_const: [&Self; 4] = <&[Self; 4]>::try_from(inputs)
                    .expect("length checked by match arm")
                    .each_ref();
                Ok(Self::l_n_one_term(inputs_const))
            }
            5 => {
                let inputs_const: [&Self; 5] = <&[Self; 5]>::try_from(inputs)
                    .expect("length checked by match arm")
                    .each_ref();
                Ok(Self::l_n_one_term(inputs_const))
            }
            6 => {
                let inputs_const: [&Self; 6] = <&[Self; 6]>::try_from(inputs)
                    .expect("length checked by match arm")
                    .each_ref();
                Ok(Self::l_n_one_term(inputs_const))
            }
            7 => {
                let inputs_const: [&Self; 7] = <&[Self; 7]>::try_from(inputs)
                    .expect("length checked by match arm")
                    .each_ref();
                Ok(Self::l_n_one_term(inputs_const))
            }
            8 => {
                let inputs_const: [&Self; 8] = <&[Self; 8]>::try_from(inputs)
                    .expect("length checked by match arm")
                    .each_ref();
                Ok(Self::l_n_one_term(inputs_const))
            }
            _ => Err(InfinityAlgebraError::ArityTooLarge(n)),
        }
    }

    /// Apply `l_N` to an element of `∧^N A`.
    ///
    /// `wedge` is a linear combination of pure wedges representing a general element
    /// of `∧^N A`. The results are summed into a single element, with `ctx` used to
    /// construct the initial zero accumulator.
    #[must_use = "returns the algebra element resulting from the linear combination; discarding it loses the computed value"]
    fn l_n<const N: usize>(ctx: Self::Ctx, wedge: ExteriorPower<Self, Coeffs, N>) -> Self {
        let mut result = Self::zero(ctx);
        for (pure_wedge, coeff) in wedge.into_summands() {
            result += Self::l_n_one_term_owned(pure_wedge) * coeff;
        }
        result
    }

    /// Evaluate the composition of L∞ operations encoded by `tree` on a general
    /// element of `∧^N A`, where `N = tree.num_leaves()`.
    ///
    /// `combination` is a linear combination of pure wedges representing a general
    /// element of `∧^N A`. The tree specifies which `l_k` to apply at each internal
    /// node and how inputs are routed to nested operations.
    ///
    /// # Errors
    ///
    /// - [`InfinityAlgebraError::CurvedAlgebra`] if any node in `tree` has arity 0.
    /// - [`InfinityAlgebraError::ArityTooLarge`] if any node exceeds the dispatch
    ///   ceiling of `l_n_slice` (8 by default).
    fn evaluate<const N: usize>(
        ctx: Self::Ctx,
        tree: &OperationTree,
        combination: ExteriorPower<Self, Coeffs, N>,
    ) -> Result<Self, InfinityAlgebraError>
    where
        Self: Clone,
    {
        assert_eq!(
            tree.num_leaves(),
            N,
            "tree has {} leaves but combination is in ∧^{}",
            tree.num_leaves(),
            N,
        );
        let mut result = Self::zero(ctx);
        for (pure_wedge, coeff) in combination.into_summands() {
            result += tree.eval_refs_l::<Self, Coeffs>(&pure_wedge)? * coeff;
        }
        Ok(result)
    }
}

impl OperationTree {
    pub(crate) fn eval_refs_l<A, Coeffs>(&self, inputs: &[A]) -> Result<A, InfinityAlgebraError>
    where
        A: LInfinityAlgebra<Coeffs> + Clone,
        Coeffs: Ring,
    {
        match self {
            Self::Leaf {
                absoulte_position: range,
            } => Ok(inputs[*range].clone()),
            Self::Node { children, .. } => {
                let mut child_outputs: Vec<A> = Vec::with_capacity(children.len());
                for child in children {
                    child_outputs.push(child.eval_refs_l(inputs)?);
                }
                A::l_n_slice(&child_outputs)
            }
        }
    }
}
