use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::arithmetic_utils::Ring;

/// A graded module over a coefficient ring `Coeffs`, constructed with a context `Ctx`.
///
/// A graded module is a module `M = ⊕ M_n` decomposed as a direct sum of homogeneous
/// pieces indexed by degree. The standard module operations (addition, subtraction,
/// scalar multiplication) must respect this grading.
///
/// The `Ctx` parameter carries any runtime information needed to construct elements —
/// for example, a plain algebra may use `()`, while a differential graded algebra
/// uses `Arc<F>` to share the differential across all elements of the same instance.
///
/// Implementors must provide:
/// - [`zero`]: the additive identity, built from a context
/// - [`extract_homogeneous`]: split off the degree-`n` component from a mixed element
pub trait GradedModule<Coeffs: Ring>
where
    Self: MulAssign<Coeffs>
        + Mul<Coeffs, Output = Self>
        + Add<Self, Output = Self>
        + AddAssign<Self>
        + Sub<Self, Output = Self>
        + SubAssign<Self>
        + Neg<Output = Self>
        + Sized,
{
    type Ctx;
    /// Split off the degree-`n` homogeneous component.
    ///
    /// Returns `(homogeneous_part, remainder)` where `homogeneous_part` lies entirely
    /// in degree `n` and `remainder` is `Some` if any other degrees were present,
    /// or `None` if the element was already purely degree `n`.
    fn extract_homogeneous(self, n: i64) -> (Self, Option<Self>);

    /// The additive identity of the module, constructed from `ctx`.
    fn zero(ctx: Self::Ctx) -> Self;

    /// Extract the context from an existing element, so that a zero of the same
    /// algebra instance can be constructed without passing `ctx` explicitly.
    fn ctx(&self) -> Self::Ctx;
}
