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
        if let Some(is_zero) = is_zero {
            for rel in &mut relations {
                rel.simplify(is_zero);
            }
        }
        relations.retain(PathAlgebra::might_be_nonzero);
        for rel in &relations {
            assert!(Arc::ptr_eq(&quiver, rel.quiver()));
            assert!(rel.all_parallel().is_ok());
        }

        Self { quiver, relations }
    }

    pub fn from_quiver_no_relations(quiver: Arc<Quiver<VertexLabel, EdgeLabel>>) -> Self {
        Self::new(quiver, vec![], None)
    }

    pub fn from_quiver_and_w(
        quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
        w_function: &PathAlgebra<VertexLabel, EdgeLabel, Coeffs>,
        is_zero: Option<fn(&Coeffs) -> bool>,
    ) -> Self {
        let mut relations = Vec::new();
        for arrow in quiver.edge_labels() {
            let mut cur_cyclic_derivative = w_function.clone();
            cur_cyclic_derivative.cyclic_derivative(arrow);
            relations.push(cur_cyclic_derivative);
        }
        Self::new(quiver, relations, is_zero)
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
            if !rel.might_be_nonzero() {
                continue;
            }
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

#[cfg(test)]
mod tests {
    use crate::quiver;

    use super::*;

    #[test]
    fn test_ginzburg() {
        use super::PathAlgebra;
        use crate::quiver::BasisElt;
        use std::sync::Arc;
        let (ginzburg_quiver, _adjoint_pairs, _self_loops) = quiver::tests::make_ginzburg_quiver();
        let ginzburg_quiver = Arc::new(ginzburg_quiver);

        let x_omega = PathAlgebra::singleton(
            ginzburg_quiver.clone(),
            BasisElt::Path(nonempty::nonempty!["Omega0".to_string()]),
            1.0,
        );
        let x_a = PathAlgebra::singleton(
            ginzburg_quiver.clone(),
            BasisElt::Path(nonempty::nonempty!["A".to_string()]),
            1.0,
        );
        let x_adag = PathAlgebra::singleton(
            ginzburg_quiver.clone(),
            BasisElt::Path(nonempty::nonempty!["ADagger".to_string()]),
            1.0,
        );

        let ginz_cubic = (x_a.clone() * x_adag.clone() - x_adag.clone() * x_a.clone()) * x_omega;

        let _ginz_with_rels = QuiverWithRelations::new(
            ginzburg_quiver.clone(),
            vec![ginz_cubic],
            Some(|x: &f64| *x == 0.0),
        );
    }
}
