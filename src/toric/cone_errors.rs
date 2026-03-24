use std::fmt;

use crate::toric::cone::RationalPolyhedralCone;

/// Errors for cone operations
#[derive(Debug, PartialEq, Eq)]
pub enum ConeError {
    DimensionMismatch { expected: usize, found: usize },
    EmptyGeneratorList,
    ZeroGenerator,
    NotMinimallyPresented,
}

/// Toric fan errors
#[derive(Debug)]
pub enum ToricFanError {
    ConeError(ConeError),
    IntersectionNotFace {
        cone1: Box<RationalPolyhedralCone>,
        cone2: Box<RationalPolyhedralCone>,
    },
    GlobalConeOnlyForAffine,
}

/// Implement Display for `ConeError`
impl fmt::Display for ConeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConeError::DimensionMismatch { expected, found } => {
                write!(f, "Dimension mismatch: expected {expected}, found {found}",)
            }
            ConeError::EmptyGeneratorList => write!(f, "Cone has empty generator list"),
            ConeError::ZeroGenerator => write!(f, "Cone has a zero generator vector"),
            ConeError::NotMinimallyPresented => write!(
                f,
                "Cone is not necessarily minimally presented e.g. Cone(e1,e1+e2,e2) vs Cone(e1,e2)"
            ),
        }
    }
}

/// Implement Display for `ToricFanError`
impl fmt::Display for ToricFanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToricFanError::ConeError(e) => write!(f, "Cone creation error: {e}"),
            ToricFanError::IntersectionNotFace { cone1, cone2 } => {
                write!(
                    f,
                    "Intersection of the following two cones is not a face:\nCone 1: {cone1:?}\nCone 2: {cone2:?}",
                )
            }
            ToricFanError::GlobalConeOnlyForAffine => {
                write!(f, "Can't make the global support as a single cone")
            }
        }
    }
}

// Optional: implement std::error::Error for compatibility
impl std::error::Error for ConeError {}
impl std::error::Error for ToricFanError {}
