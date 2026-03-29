use std::{
    collections::HashMap,
    ops::{MulAssign, Neg},
    sync::Arc,
};

use num::Integer;

use crate::quiver_algebra::{
    DegreeLabel, HasHomologicalDegree,
    checked_arith::{ChainMultiplyable, CheckedAdd, CheckedAddAssign, Ring},
    dg_path_algebra::GradedDifferentialQuiver,
    quiver_rep::QuiverRep,
};

/// A left module over a [`GradedDifferentialQuiver`]
/// `(kQ, d_kQ)` for any Z-graded quiver Q with differential `d_kQ`.
/// But we do not store the differential here.
///
/// Consists of:
/// - `rep`: a representation of the underlying graded quiver, giving the action of each
///   arrow as a plain matrix (degree information comes from the [`DegreeLabel`] key).
/// - `vertex_differentials`: a degree-+1 endomorphism `d_M \mid_v` at each vertex, stored as a plain
///   matrix in the fixed basis as in the [`QuiverRep`]
///
/// - Compatibility with the DGA differential is done via [`DGModule::leibniz_compatible`]
/// - The representations assigned to each vertex actually being complexes is done via [`DGModule::differential_squares_zero`].
#[must_use]
pub struct DGModule<VertexLabel, EdgeLabel, MatrixType, const OP_ALG: bool>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    MatrixType: CheckedAdd + CheckedAddAssign + ChainMultiplyable + Clone,
{
    rep: QuiverRep<VertexLabel, DegreeLabel<EdgeLabel>, MatrixType, OP_ALG>,
    vertex_differentials: HashMap<VertexLabel, MatrixType>,
}

impl<VertexLabel, EdgeLabel, MatrixType, const OP_ALG: bool>
    DGModule<VertexLabel, EdgeLabel, MatrixType, OP_ALG>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    MatrixType: CheckedAdd + CheckedAddAssign + ChainMultiplyable + Clone,
{
    /// Construct a DG-module over a graded differential path algebra `(kQ, d_kQ)`, given:
    /// - a representation of the graded quiver Q
    /// - a choice of degree-+1 endomorphisms at each vertex
    ///
    /// It is the caller's responsibility to ensure that the representation and differentials are compatible
    ///
    /// # Errors
    /// If `validate` is provided, it is used to check that the differentials square to zero.
    /// If any fail, returns the labels of the vertices where δ² ≠ 0.
    /// However checking compatibility with `d_kQ` is not done here even with `validate`.
    pub fn new(
        rep: QuiverRep<VertexLabel, DegreeLabel<EdgeLabel>, MatrixType, OP_ALG>,
        vertex_differentials: HashMap<VertexLabel, MatrixType>,
        validate: Option<fn(&MatrixType) -> bool>,
    ) -> Result<Self, Vec<VertexLabel>> {
        let to_return = Self {
            rep,
            vertex_differentials,
        };
        if let Some(matrix_is_zero) = validate {
            let all_errors = to_return.differential_squares_zero(matrix_is_zero);
            if !all_errors.is_empty() {
                return Err(all_errors);
            }
        }
        Ok(to_return)
    }

    /// The underlying quiver representation (arrow action matrices).
    pub fn rep(&self) -> &QuiverRep<VertexLabel, DegreeLabel<EdgeLabel>, MatrixType, OP_ALG> {
        &self.rep
    }

    /// The degree-+1 endomorphism `d_M` at vertex `v`, or `None` if `v` is not in the module.
    pub fn vertex_differential(&self, v: &VertexLabel) -> Option<&MatrixType> {
        self.vertex_differentials.get(v)
    }

    /// Check the Leibniz compatibility condition for each arrow `a`
    ///
    /// ```text
    /// d_{M,t(a)} ∘ ρ(a)  ==  (-1)^|a| · ρ(a) ∘ d_{M,s(a)}  +  ρ(d_kQ (a))
    /// ```
    ///
    /// Returns the labels of every arrow whose equation fails.
    ///
    /// # Panics
    ///
    /// Panics if matrix multiplications or additions fail (incompatible dimensions indicate a
    /// malformed module).
    #[allow(clippy::missing_panics_doc, clippy::similar_names)]
    pub fn leibniz_compatible<Coeffs>(
        &self,
        dga: &GradedDifferentialQuiver<VertexLabel, EdgeLabel, Coeffs, OP_ALG>,
        matrix_close_enough: impl Fn(&MatrixType, &MatrixType) -> bool,
    ) -> Vec<DegreeLabel<EdgeLabel>>
    where
        Coeffs: Ring,
        MatrixType: MulAssign<Coeffs> + Neg<Output = MatrixType>,
    {
        let quiver = self.rep.quiver();
        assert!(Arc::ptr_eq(quiver, dga.quiver()));
        let mut failing = Vec::new();

        for a in quiver.edge_labels() {
            let (src, tgt) = quiver
                .edge_endpoint_labels(a)
                .expect("edge is in the quiver");
            let rho_a = self
                .rep
                .get_edge_rep(a)
                .expect("rep has a matrix for every arrow")
                .clone();
            let delta_src = self
                .vertex_differentials
                .get(&src)
                .expect("vertex_differentials has an entry for every vertex")
                .clone();
            let delta_tgt = self
                .vertex_differentials
                .get(&tgt)
                .expect("vertex_differentials has an entry for every vertex")
                .clone();

            // LHS = δ_{t(a)} ∘ ρ(a)
            let lhs = MatrixType::mul_two(rho_a.clone(), delta_tgt)
                .unwrap_or_else(|_| panic!("compatible dimensions"));

            // RHS = (-1)^|a| · ρ(a) ∘ δ_{s(a)}
            let mut rhs = MatrixType::mul_two(delta_src, rho_a)
                .unwrap_or_else(|_| panic!("compatible dimensions"));
            if a.homological_degree()
                .expect("has homological degree")
                .is_odd()
            {
                rhs = -rhs;
            }

            // RHS += ρ(d(a)) if d(a) ≠ 0
            #[allow(clippy::collapsible_if)]
            if let Some(da) = dga.apply_differential_letter(a) {
                if da.might_be_nonzero() {
                    let rho_da = self
                        .rep
                        .mat_from_path_algebra(da)
                        .unwrap_or_else(|_| panic!("compatible dimensions"));
                    rhs = rhs
                        .checked_add(rho_da)
                        .unwrap_or_else(|_| panic!("compatible dimensions"));
                }
            }

            if !matrix_close_enough(&lhs, &rhs) {
                failing.push(a.clone());
            }
        }

        failing
    }

    /// Check that `d_{M,v}² = 0` at every vertex.
    ///
    /// # Panics
    ///
    /// Panics if squaring a vertex differential fails (incompatible dimensions).
    #[allow(clippy::missing_panics_doc)]
    pub fn differential_squares_zero(
        &self,
        matrix_is_zero: impl Fn(&MatrixType) -> bool,
    ) -> Vec<VertexLabel> {
        let mut failing = Vec::new();
        for (v, delta) in &self.vertex_differentials {
            let delta_sq = MatrixType::mul_two(delta.clone(), delta.clone())
                .unwrap_or_else(|_| panic!("compatible dimensions"));
            if !matrix_is_zero(&delta_sq) {
                failing.push(v.clone());
            }
        }
        failing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quiver_algebra::{
        BasisElt, DegreeLabel, GradedDifferentialQuiver, PathAlgebra, Quiver, QuiverRep,
    };
    use nonempty::nonempty;
    use proptest::{prelude::Strategy, proptest};
    use std::sync::Arc;

    proptest! {

        /// DGA: 1 vertex "v", arrows x (degree 0), y1 (degree -1), y2 (degree -1),
        /// with d(y1) = x, d(y2) = x, d(x) = 0.
        ///
        /// Module M = k (scalar), all maps: ρ(x)=c, ρ(y1)=ρ(y2)=0, d_M=0.
        ///
        /// Leibniz for y_i:
        ///   d_M ∘ ρ(y_i) = 0  ==  (-1)^{-1} ρ(y_i) ∘ d_M  +  ρ(x)  =  0 + c
        /// Leibniz for x:
        ///   d_M ∘ ρ(x) = 0  ==  (-1)^{-1} ρ(x) ∘ d_M  +  ρ(0)  =  0 + 0
        ///
        /// So c=0 passes, c≠0 fails on y1 and y2 and passes on x
        #[test]
        fn leibniz_detects_wrong_action_on_odd_degree_arrows(
            nonzero_c in proptest::num::f64::NORMAL.prop_filter("nonzero",|z| z.abs() > 1e-10),
        ) {
            let x = DegreeLabel::new("x".to_string(), 0i64);
            let y1 = DegreeLabel::new("y1".to_string(), -1i64);
            let y2 = DegreeLabel::new("y2".to_string(), -1i64);

            let mut q: Quiver<&str, DegreeLabel<String>> = Quiver::new();
            q.add_edge("v", "v", x.clone());
            q.add_edge("v", "v", y1.clone());
            q.add_edge("v", "v", y2.clone());
            let q_arc = Arc::new(q);

            let x_elt =
                PathAlgebra::singleton(q_arc.clone(), BasisElt::Path(nonempty![x.clone()]), 1.0f64);
            let mut differential: HashMap<String, PathAlgebra<&str, DegreeLabel<String>, f64,true>> =
                HashMap::new();
            differential.insert("y1".to_string(), x_elt.clone());
            differential.insert("y2".to_string(), x_elt.clone());
            let dga = GradedDifferentialQuiver::new(q_arc.clone(), differential);

            let make_module = |c: f64| {
                let edge_reps: HashMap<_, _> =
                    [(x.clone(), c), (y1.clone(), 0.0f64), (y2.clone(), 0.0f64)]
                        .into_iter()
                        .collect();
                let vertex_reps: HashMap<_, _> = [("v", 1usize)].into_iter().collect();
                let rep = QuiverRep::new(q_arc.clone(), edge_reps, vertex_reps, |_| 1.0).unwrap();
                let vertex_diffs: HashMap<_, _> = [("v", 0.0f64)].into_iter().collect();
                DGModule::new(rep, vertex_diffs, Some(|z: &f64| *z == 0.0)).unwrap()
            };

            // c=0: all Leibniz equations satisfied
            assert!(
                make_module(0.0)
                    .leibniz_compatible(&dga, |a, b| a == b)
                    .is_empty()
            );

            // c not equal to 0: y1 and y2 fail (RHS = ρ(x) = c ≠ 0 = LHS), x passes
            let mut failing = make_module(nonzero_c).leibniz_compatible(&dga, |a, b| a == b);
            failing.sort_by_key(|e| e.name().clone());
            assert_eq!(failing, vec![y1.clone(), y2.clone()]);
        }

        #[test]
        fn leibniz_detects_wrong_action_on_odd_degree_arrows_reg(
            nonzero_c in proptest::num::f64::NORMAL.prop_filter("nonzero",|z| z.abs() > 1e-10),
        ) {
            let x = DegreeLabel::new("x".to_string(), 0i64);
            let y1 = DegreeLabel::new("y1".to_string(), -1i64);
            let y2 = DegreeLabel::new("y2".to_string(), -1i64);

            let mut q: Quiver<&str, DegreeLabel<String>> = Quiver::new();
            q.add_edge("v", "v", x.clone());
            q.add_edge("v", "v", y1.clone());
            q.add_edge("v", "v", y2.clone());
            let q_arc = Arc::new(q);

            let x_elt =
                PathAlgebra::singleton(q_arc.clone(), BasisElt::Path(nonempty![x.clone()]), 1.0f64);
            let mut differential: HashMap<String, PathAlgebra<&str, DegreeLabel<String>, f64,false>> =
                HashMap::new();
            differential.insert("y1".to_string(), x_elt.clone());
            differential.insert("y2".to_string(), x_elt.clone());
            let dga = GradedDifferentialQuiver::new(q_arc.clone(), differential);

            let make_module = |c: f64| {
                let edge_reps: HashMap<_, _> =
                    [(x.clone(), c), (y1.clone(), 0.0f64), (y2.clone(), 0.0f64)]
                        .into_iter()
                        .collect();
                let vertex_reps: HashMap<_, _> = [("v", 1usize)].into_iter().collect();
                let rep = QuiverRep::new(q_arc.clone(), edge_reps, vertex_reps, |_| 1.0).unwrap();
                let vertex_diffs: HashMap<_, _> = [("v", 0.0f64)].into_iter().collect();
                DGModule::new(rep, vertex_diffs, Some(|z: &f64| *z == 0.0)).unwrap()
            };

            // c=0: all Leibniz equations satisfied
            assert!(
                make_module(0.0)
                    .leibniz_compatible(&dga, |a, b| a == b)
                    .is_empty()
            );

            // c not equal to 0: y1 and y2 fail (RHS = ρ(x) = c ≠ 0 = LHS), x passes
            let mut failing = make_module(nonzero_c).leibniz_compatible(&dga, |a, b| a == b);
            failing.sort_by_key(|e| e.name().clone());
            assert_eq!(failing, vec![y1.clone(), y2.clone()]);
        }
    }
}
