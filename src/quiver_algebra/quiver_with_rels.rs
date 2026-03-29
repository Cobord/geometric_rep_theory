use std::{collections::HashMap, ops::MulAssign, sync::Arc};

use crate::quiver_algebra::{
    checked_arith::{ChainMultiplyable, CheckedAdd, CheckedAddAssign, Ring},
    path_algebra::PathAlgebra,
    quiver::{BasisElt, Quiver},
    quiver_rep::QuiverRep,
};

/// A quiver together with a generating set for a two-sided ideal, representing the quotient
/// algebra A = kQ^{op}/I^{op} (or kQ/I when `OP_ALG = false`).
#[must_use]
pub struct QuiverWithRelations<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
    Coeffs: Ring,
{
    quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
    relations: Vec<PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>>,
}

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool>
    QuiverWithRelations<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
    Coeffs: Ring,
{
    /// Construct a quiver with relations, optionally simplifying and stripping zero relations.
    ///
    /// If `is_zero` is provided, each relation is simplified (zero coefficients removed) before
    /// being stored. Relations that are provably zero after simplification are discarded.
    ///
    /// # Panics
    ///
    /// Panics if any relation does not belong to the path algebra of `quiver` (checked via
    /// pointer equality on the underlying `Arc`), or if a relation has summands with
    /// inconsistent endpoints.
    pub fn new(
        quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
        mut relations: Vec<PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>>,
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

    /// Construct the free path algebra kQ^{op} viewed as a quiver with an empty ideal.
    pub fn from_quiver_no_relations(quiver: Arc<Quiver<VertexLabel, EdgeLabel>>) -> Self {
        Self::new(quiver, vec![], None)
    }

    /// Construct the Jacobian algebra from a superpotential `W`.
    ///
    /// The relations are the cyclic partial derivatives ∂W/∂α for each arrow α in the quiver.
    /// This gives the Jacobian (or Ginzburg) algebra kQ^{op}/⟨∂W/∂α⟩.
    pub fn from_quiver_and_w(
        quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
        w_function: &PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>,
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

    /// The underlying quiver.
    #[allow(clippy::must_use_candidate)]
    pub fn quiver(&self) -> &Arc<Quiver<VertexLabel, EdgeLabel>> {
        &self.quiver
    }

    /// Iterate over the stored relations (non-zero elements of kQ^{op} declared to be zero).
    pub fn relations(
        &self,
    ) -> impl Iterator<Item = &PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>> {
        self.relations.iter()
    }

    /// Check whether a quiver representation descends to a representation of the quotient algebra.
    ///
    /// Returns `true` iff every relation acts as the zero map on the representation, i.e.
    /// `mat_from_path_algebra(rel)` is zero (as determined by `matrix_is_zero`) for every relation.
    pub fn rep_descends<MatrixType>(
        &self,
        quiver_rep: &QuiverRep<VertexLabel, EdgeLabel, MatrixType, OP_ALG>,
        mut matrix_is_zero: impl FnMut(&MatrixType) -> bool,
    ) -> bool
    where
        MatrixType: CheckedAdd + CheckedAddAssign + ChainMultiplyable + Clone + MulAssign<Coeffs>,
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

impl<VertexLabel, EdgeLabel> Quiver<VertexLabel, EdgeLabel>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
{
    /// Construct the preprojective algebra of a quiver.
    ///
    /// Doubles the quiver (adding a reverse arrow a* for each arrow a), then imposes
    /// at each vertex v the relation:
    ///
    /// ```text
    /// ∑_{a: s(a)=v} a·a*  -  ∑_{a: t(a)=v} a*·a  =  0
    /// ```
    ///
    /// Philosophically this is the degree-0 truncation of the Ginzburg DG-algebra: if one
    /// forms the Ginzburg quiver (doubled + a self-loop `ω_v` at each vertex) and the cubic
    /// superpotential `W = ∑_a [a, a*]·ω_{s(a)}`, then the relation above is exactly the
    /// cyclic derivative `∂W/∂ω_v`.  The full Ginzburg DGA (including the `ω_v`, the relations
    /// `∂W/∂a` and `∂W/∂a*`, and the DG differential) is constructed by [`Quiver::ginzburgify_and_cubic`];
    /// use that method to preserve the complete DG structure rather than truncating here.
    ///
    /// Returns the doubled quiver (as an `Arc`) and the associated `QuiverWithRelations`.
    #[allow(clippy::missing_panics_doc)]
    pub fn preprojective_algebra<Coeffs: Ring, const OP_ALG: bool>(
        self,
        dagger: impl Fn(&EdgeLabel) -> EdgeLabel,
        one_coeffs: &Coeffs,
    ) -> (
        Arc<Self>,
        QuiverWithRelations<VertexLabel, EdgeLabel, Coeffs, OP_ALG>,
    ) {
        let (doubled, adjoint_pairs) = self.double(dagger);
        let doubled_arc = Arc::new(doubled);
        let mut vertex_relations: HashMap<
            VertexLabel,
            PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>,
        > = HashMap::new();

        for (a, a_star) in &adjoint_pairs {
            let (src, tgt) = doubled_arc
                .edge_endpoint_labels(a)
                .expect("original arrow is in the doubled quiver");

            // +a·a* at vertex src = s(a)
            let aa_star = PathAlgebra::singleton(
                doubled_arc.clone(),
                BasisElt::Path(nonempty::nonempty![a.clone(), a_star.clone()]),
                one_coeffs.clone(),
            );
            vertex_relations
                .entry(src)
                .and_modify(|rel| *rel += aa_star.clone())
                .or_insert(aa_star);

            // -a*·a at vertex tgt = t(a)
            let a_star_a = PathAlgebra::singleton(
                doubled_arc.clone(),
                BasisElt::Path(nonempty::nonempty![a_star.clone(), a.clone()]),
                one_coeffs.clone(),
            );
            vertex_relations
                .entry(tgt)
                .and_modify(|rel| *rel -= a_star_a.clone())
                .or_insert(-a_star_a);
        }

        let relations: Vec<_> = vertex_relations.into_values().collect();
        let qwr = QuiverWithRelations::new(doubled_arc.clone(), relations, None);
        (doubled_arc, qwr)
    }
}

#[cfg(test)]
mod tests {
    use crate::quiver_algebra;

    use super::*;

    #[test]
    fn a2_preprojective() {
        // make_a2_quiver uses &'static str edges; map to String so dagger can produce new labels.
        let a2 =
            quiver_algebra::quiver::tests::make_a2_quiver().map_labels(|v| v, |e| e.to_string());
        let (_doubled_arc, preprojective) =
            a2.preprojective_algebra::<_, true>(|e| format!("{e}*"), &1i64);

        // The doubled A2 quiver has arrows "a" (alpha→beta) and "a*" (beta→alpha).
        // Relation at alpha: a · a*  = 0
        // Relation at beta:  -a* · a = 0  (i.e., a* · a = 0)
        // There should be exactly 2 relations (one per vertex).
        assert_eq!(preprojective.relations().count(), 2);

        // Each vertex-local relation is a single degree-2 path.
        for rel in preprojective.relations() {
            assert!(rel.is_homogeneous_of_degree(2));
        }
    }

    #[test]
    fn test_ginzburg() {
        use super::PathAlgebra;
        use crate::quiver_algebra::quiver::BasisElt;
        use std::sync::Arc;
        let (ginzburg_quiver, _adjoint_pairs, _self_loops) =
            quiver_algebra::quiver::tests::make_ginzburg_quiver();
        let ginzburg_quiver = Arc::new(ginzburg_quiver);

        let x_omega = BasisElt::Path(nonempty::nonempty!["Omega0".to_string()]);
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

        let _ginz_with_rels = QuiverWithRelations::<_, _, _, true>::new(
            ginzburg_quiver.clone(),
            vec![ginz_cubic],
            Some(|x: &f64| *x == 0.0),
        );
    }
}
