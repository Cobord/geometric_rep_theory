use super::errors::InfinityAlgebraError;
use crate::arithmetic_utils::Ring;
use crate::infinity_algebra::graded_module::GradedModule;
use crate::infinity_algebra::operation_tree::OperationTree;
use crate::infinity_algebra::tensor_power::TensorPower;

/// A strictly unital A∞-algebra over a coefficient ring `Coeffs`.
///
/// An A∞-algebra is a graded module equipped with a sequence of multilinear operations
/// `m_n` for `n ≥ 1` satisfying the A∞ associativity relations. These generalise the
/// usual associativity of a dg-algebra: `m_1` is the differential, `m_2` is the
/// multiplication (associative only up to the homotopy `m_3`), and higher `m_n` measure
/// the failure of associativity at each level.
///
/// Only uncurved algebras are supported: `m_0 = 0` (no curvature term). Passing
/// arity 0 to [`m_n_slice`] returns [`InfinityAlgebraError::CurvedAlgebra`].
///
/// Implementors must provide [`m_n_one_term`] and [`m_n_one_term_owned`] for a single
/// pure tensor input. The default methods build on them:
/// - [`m_n_slice`]: dynamic-arity dispatch, used internally for tree evaluation
/// - [`m_n`]: evaluates a full linear combination at fixed arity `N`
/// - [`evaluate`]: evaluates a composition of operations described by an [`OperationTree`]
pub trait AInfinityAlgebra<Coeffs: Ring>
where
    Self: GradedModule<Coeffs>,
{
    /// Apply the `N`-ary operation `m_N` to a single pure tensor in `A^⊗N`.
    fn m_n_one_term_owned<const N: usize>(inputs: [Self; N]) -> Self;

    /// Apply the `N`-ary operation `m_N` to a single pure tensor in `A^⊗N`.
    fn m_n_one_term<const N: usize>(inputs: [&Self; N]) -> Self;

    /// The largest `n` for which `m_n` may be nonzero, or `None` if the algebra
    /// has operations of unbounded arity.
    ///
    /// When `m_n_slice` receives arity `n > 8` and `n` exceeds this bound, it
    /// returns `Ok(zero)` instead of `Err(ArityTooLarge)`.
    #[must_use = "Avoid trying to compute the higher coherences which are definitively 0"]
    fn max_nonzero_arity() -> Option<usize> {
        None
    }

    /// Dynamic-arity dispatch to [`m_n_one_term`]. Override if your algebra uses
    /// arities above 8.
    ///
    /// # Errors
    ///
    /// - [`InfinityAlgebraError::CurvedAlgebra`] if `inputs` is empty (arity 0).
    /// - [`InfinityAlgebraError::ArityTooLarge`] if `inputs.len()` exceeds the
    ///   default dispatch ceiling of 8 and does not exceed [`max_nonzero_arity`].
    fn m_n_slice(inputs: &[Self]) -> Result<Self, InfinityAlgebraError> {
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
                Ok(Self::m_n_one_term(inputs_const))
            }
            2 => {
                let inputs_const: [&Self; 2] = <&[Self; 2]>::try_from(inputs)
                    .expect("length checked by match arm")
                    .each_ref();
                Ok(Self::m_n_one_term(inputs_const))
            }
            3 => {
                let inputs_const: [&Self; 3] = <&[Self; 3]>::try_from(inputs)
                    .expect("length checked by match arm")
                    .each_ref();
                Ok(Self::m_n_one_term(inputs_const))
            }
            4 => {
                let inputs_const: [&Self; 4] = <&[Self; 4]>::try_from(inputs)
                    .expect("length checked by match arm")
                    .each_ref();
                Ok(Self::m_n_one_term(inputs_const))
            }
            5 => {
                let inputs_const: [&Self; 5] = <&[Self; 5]>::try_from(inputs)
                    .expect("length checked by match arm")
                    .each_ref();
                Ok(Self::m_n_one_term(inputs_const))
            }
            6 => {
                let inputs_const: [&Self; 6] = <&[Self; 6]>::try_from(inputs)
                    .expect("length checked by match arm")
                    .each_ref();
                Ok(Self::m_n_one_term(inputs_const))
            }
            7 => {
                let inputs_const: [&Self; 7] = <&[Self; 7]>::try_from(inputs)
                    .expect("length checked by match arm")
                    .each_ref();
                Ok(Self::m_n_one_term(inputs_const))
            }
            8 => {
                let inputs_const: [&Self; 8] = <&[Self; 8]>::try_from(inputs)
                    .expect("length checked by match arm")
                    .each_ref();
                Ok(Self::m_n_one_term(inputs_const))
            }
            _ => Err(InfinityAlgebraError::ArityTooLarge(n)),
        }
    }

    /// Apply `m_N` to an element of `A^⊗N`.
    ///
    /// `tensor` is a linear combination of pure tensors representing a general element
    /// of `A^⊗N`. The results are summed into a single element, with `ctx` used to
    /// construct the initial zero accumulator.
    #[must_use = "returns the algebra element resulting from the linear combination; discarding it loses the computed value"]
    fn m_n<const N: usize>(ctx: Self::Ctx, tensor: TensorPower<Self, Coeffs, N>) -> Self {
        let mut result = Self::zero(ctx);
        for (pure_tensor, coeff) in tensor.into_summands() {
            result += Self::m_n_one_term_owned(pure_tensor) * coeff;
        }
        result
    }

    /// Evaluate the composition of A∞ operations encoded by `tree` on a general
    /// element of `A^⊗n`, where `n = tree.num_leaves()`.
    ///
    /// `combination` is a linear combination of pure tensors representing a general
    /// element of `A^⊗n`: each entry is a pure tensor (a slice of `n` elements,
    /// ordered left to right across the leaves) scaled by a coefficient. The tree
    /// specifies which `m_k` to apply at each internal node and how inputs are
    /// routed to nested operations.
    ///
    /// # Errors
    ///
    /// - [`InfinityAlgebraError::CurvedAlgebra`] if any node in `tree` has arity 0.
    /// - [`InfinityAlgebraError::ArityTooLarge`] if any node exceeds the dispatch
    ///   ceiling of `m_n_slice` (8 by default).
    fn evaluate<const N: usize>(
        ctx: Self::Ctx,
        tree: &OperationTree,
        combination: TensorPower<Self, Coeffs, N>,
    ) -> Result<Self, InfinityAlgebraError>
    where
        Self: Clone,
    {
        assert_eq!(
            tree.num_leaves(),
            N,
            "tree has {} leaves but combination is in A^⊗{}",
            tree.num_leaves(),
            N,
        );
        let mut result = Self::zero(ctx);
        for (pure_tensor, coeff) in combination.into_summands() {
            result += tree.eval_refs_a::<Self, Coeffs>(&pure_tensor)? * coeff;
        }
        Ok(result)
    }
}

impl OperationTree {
    pub(crate) fn eval_refs_a<A, Coeffs>(&self, inputs: &[A]) -> Result<A, InfinityAlgebraError>
    where
        A: AInfinityAlgebra<Coeffs> + Clone,
        Coeffs: Ring,
    {
        match self {
            Self::Leaf {
                absoulte_position: range,
            } => Ok(inputs[*range].clone()),
            Self::Node { children, .. } => {
                let mut child_outputs: Vec<A> = Vec::with_capacity(children.len());
                for child in children {
                    child_outputs.push(child.eval_refs_a(inputs)?);
                }
                A::m_n_slice(&child_outputs)
            }
        }
    }
}
