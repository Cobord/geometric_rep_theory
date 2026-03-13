use std::{ops::MulAssign, sync::Arc};

use crate::{
    checked_arith::{CheckedAdd, CheckedAddAssign, CheckedMul, CheckedMulAssign, Ring},
    quiver::{PathAlgebra, Quiver},
    quiver_rep::QuiverRep,
};

#[must_use]
pub struct QuiverWithRelations<VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
    Coeffs: Ring,
{
    quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
    relations: Vec<PathAlgebra<VertexLabel, EdgeLabel, Coeffs>>,
}

impl<VertexLabel, EdgeLabel, Coeffs> QuiverWithRelations<VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
    Coeffs: Ring,
{
    ///
    ///
    /// # Panics
    ///
    /// All the relations must be in the path algebra of this particular quiver
    pub fn new(
        quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
        mut relations: Vec<PathAlgebra<VertexLabel, EdgeLabel, Coeffs>>,
        is_zero: Option<fn(&Coeffs) -> bool>,
    ) -> Self {
        relations.retain(PathAlgebra::might_be_nonzero);
        for rel in &relations {
            assert!(Arc::ptr_eq(&quiver, rel.quiver()));
            assert!(rel.all_parallel() == Err(()));
        }
        if let Some(is_zero) = is_zero {
            for rel in &mut relations {
                rel.simplify(is_zero);
            }
        }
        Self { quiver, relations }
    }

    #[allow(clippy::must_use_candidate)]
    pub fn quiver(&self) -> &Arc<Quiver<VertexLabel, EdgeLabel>> {
        &self.quiver
    }

    pub fn relations(&self) -> impl Iterator<Item = &PathAlgebra<VertexLabel, EdgeLabel, Coeffs>> {
        self.relations.iter()
    }

    pub fn rep_descends<MatrixType>(
        &self,
        quiver_rep: &QuiverRep<VertexLabel, EdgeLabel, MatrixType>,
        mut matrix_is_zero: impl FnMut(&MatrixType) -> bool,
    ) -> bool
    where
        MatrixType: CheckedAdd
            + CheckedAddAssign
            + CheckedMul
            + CheckedMulAssign
            + Clone
            + MulAssign<Coeffs>,
    {
        for rel in &self.relations {
            if let Ok(mat_this_rel) = quiver_rep.mat_from_path_algebra(rel.clone()) {
                if !matrix_is_zero(&mat_this_rel) {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
}
