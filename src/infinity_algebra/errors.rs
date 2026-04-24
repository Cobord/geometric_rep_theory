#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InfinityAlgebraError {
    /// The arity passed to a dynamic-arity dispatch exceeds the dispatch ceiling.
    /// Override the slice dispatch method to support higher arities.
    ArityTooLarge(usize),
    /// Curvature is not allowed
    CurvedAlgebra,
}

impl std::fmt::Display for InfinityAlgebraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArityTooLarge(n) => write!(
                f,
                "arity {n} exceeds default dispatch ceiling; override the slice dispatch method"
            ),
            Self::CurvedAlgebra => write!(
                f,
                "Curvature in the sense of infinity algebra is not allowed"
            ),
        }
    }
}

impl std::error::Error for InfinityAlgebraError {}
