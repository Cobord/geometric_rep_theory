use std::{
    collections::HashMap,
    ops::{MulAssign, Neg},
    sync::Arc,
};

use num::Integer;

use crate::arithmetic_utils::{
    ChainMultiplyable, CheckedAdd, CheckedAddAssign, CheckedArithError, Ring,
};
use crate::quiver_algebra::{
    DegreeLabel, HasHomologicalDegree, dg_path_algebra::GradedDifferentialQuiver,
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

    /// Apply a gauge transformation to the DG-module.
    ///
    /// The arrow maps in `rep` are updated exactly as in [`QuiverRep::gauge_transform`].
    /// Each vertex differential is conjugated:
    /// ```text
    /// d_M'_v = g_v⁻¹ · d_{M,v} · g_v = gauge[v].0 · d_{M,v} · gauge[v].1
    /// ```
    /// Vertices absent from `gauge_transformation` are left unchanged.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any matrix multiplication fails (e.g. shape mismatch).
    pub fn gauge_transform(
        &mut self,
        gauge_transformation: &HashMap<VertexLabel, (MatrixType, MatrixType)>,
    ) -> Result<(), CheckedArithError<MatrixType>> {
        self.rep.gauge_transform(gauge_transformation)?;
        for (v, delta) in &mut self.vertex_differentials {
            if let Some((g_inv, g)) = gauge_transformation.get(v) {
                *delta = g_inv
                    .clone()
                    .chain_multiply_after([delta.clone(), g.clone()])
                    .map_err(CheckedArithError::from_mul)?;
            }
        }
        Ok(())
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

    // Two vertices u (1D) and v (2D), one arrow a: u→v of degree 0.
    //
    // ρ(a) = [[1],[1]]  (2×1)
    // d_M_u = [[0]]              (1×1 zero, trivially squares to zero)
    // d_M_v = [[0,0],[1,0]]      (lower triangular: e1 degree 0, e2 degree 1,
    //                             d maps e1→e2; squares to zero)
    //
    // Gauge: g_u = [[2]], g_u⁻¹ = [[1/2]]
    //        g_v = diag(2,3), g_v⁻¹ = diag(1/2, 1/3)
    //
    // Expected (mul_two(A,B) = B·A, so chain is g·M·g⁻¹):
    //   ρ(a)' = g_v · ρ(a) · g_u⁻¹ = [[1],[3/2]]
    //   d_M_v' = g_v · d_M_v · g_v⁻¹ = [[0,0],[3/2,0]]
    //   d_M_u' = [[0]]  (unchanged)
    #[test]
    fn gauge_transform_two_vertex_arrow() {
        use crate::arithmetic_utils::DynMatrix;
        use crate::quiver_algebra::{DegreeLabel, Quiver, QuiverRep};
        use nalgebra::DMatrix;

        let a = DegreeLabel::new("a".to_string(), 0i64);
        let mut q: Quiver<&str, DegreeLabel<String>> = Quiver::new();
        q.add_edge("u", "v", a.clone());
        let q_arc = Arc::new(q);

        let rho_a = DynMatrix(DMatrix::from_vec(2, 1, vec![1.0_f64, 1.0]));
        let edge_reps = [(a.clone(), rho_a)].into_iter().collect();
        let vertex_dims = [("u", 1usize), ("v", 2usize)].into_iter().collect();
        let rep = QuiverRep::new(q_arc.clone(), edge_reps, vertex_dims, |n| {
            DynMatrix::zeros(n, n)
        })
        .unwrap();

        // d_M_v = [[0,0],[1,0]], column-major: col0=[0,1], col1=[0,0]
        let d_v = DynMatrix(DMatrix::from_vec(2, 2, vec![0.0, 1.0, 0.0, 0.0]));
        let d_u = DynMatrix(DMatrix::from_vec(1, 1, vec![0.0]));
        let vertex_diffs = [("u", d_u), ("v", d_v)].into_iter().collect();

        let is_zero = |m: &DynMatrix<f64>| m.0.iter().all(|&x| x.abs() < 1e-10);
        let mut module = DGModule::<_, _, _, true>::new(rep, vertex_diffs, Some(is_zero)).unwrap();

        // gauge["u"] = (g_u⁻¹, g_u) = ([[1/2]], [[2]])
        // gauge["v"] = (g_v⁻¹, g_v) = (diag(1/2,1/3), diag(2,3))
        let g_u_inv = DynMatrix(DMatrix::from_vec(1, 1, vec![0.5]));
        let g_u = DynMatrix(DMatrix::from_vec(1, 1, vec![2.0]));
        // g_v = diag(2,3): col0=[2,0], col1=[0,3]
        let g_v = DynMatrix(DMatrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 3.0]));
        // g_v⁻¹ = diag(1/2,1/3): col0=[1/2,0], col1=[0,1/3]
        let g_v_inv = DynMatrix(DMatrix::from_vec(2, 2, vec![0.5, 0.0, 0.0, 1.0 / 3.0]));

        let gauge = [("u", (g_u_inv, g_u)), ("v", (g_v_inv, g_v))]
            .into_iter()
            .collect();
        module
            .gauge_transform(&gauge)
            .map_err(|_| ())
            .expect("gauge transform succeeds");

        // ρ(a)' = [[1],[3/2]]
        let expected_rho = DynMatrix(DMatrix::from_vec(2, 1, vec![1.0, 1.5]));
        let actual_rho = module.rep().get_edge_rep(&a).unwrap();
        assert!(
            (actual_rho.0.clone() - &expected_rho.0).abs().max() < 1e-10,
            "ρ(a)' wrong: got {actual_rho:?}"
        );

        // d_M_u' = [[0]]  (unchanged)
        let actual_du = module.vertex_differential(&"u").unwrap();
        assert!(is_zero(actual_du), "d_M_u should stay zero");

        // d_M_v' = [[0,0],[3/2,0]], column-major: col0=[0,3/2], col1=[0,0]
        let expected_dv = DynMatrix(DMatrix::from_vec(2, 2, vec![0.0, 1.5, 0.0, 0.0]));
        let actual_dv = module.vertex_differential(&"v").unwrap();
        assert!(
            (actual_dv.0.clone() - &expected_dv.0).abs().max() < 1e-10,
            "d_M_v' wrong: got {actual_dv:?}"
        );
    }
}
