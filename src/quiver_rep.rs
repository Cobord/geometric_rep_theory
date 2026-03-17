use std::ops::MulAssign;
use std::{collections::HashMap, sync::Arc};

use nonempty::NonEmpty;

use crate::checked_arith::{
    CheckedAdd, CheckedAddAssign, CheckedArithError, CheckedMul, CheckedMulAssign, Ring,
};
use crate::quiver::{BasisElt, PathAlgebra, Quiver};

#[must_use]
pub struct QuiverRep<VertexLabel, EdgeLabel, MatrixType>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
    MatrixType: CheckedAdd + CheckedAddAssign + CheckedMul + CheckedMulAssign + Clone,
{
    quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
    edge_reps: HashMap<EdgeLabel, MatrixType>,
    vertex_reps: HashMap<VertexLabel, MatrixType>,
}

impl<VertexLabel, EdgeLabel, MatrixType> QuiverRep<VertexLabel, EdgeLabel, MatrixType>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
    MatrixType: CheckedAdd + CheckedAddAssign + CheckedMul + CheckedMulAssign + Clone,
{
    ///
    ///
    /// # Errors
    ///
    /// If we have not given the matrix associated to every edge, then this is
    /// not enough information for a quiver representation
    ///
    /// If we are doing `check_idempotency` and any of the vertex reps is not idempotent,
    /// then this is not a valid quiver rep.
    #[allow(clippy::missing_panics_doc, clippy::type_complexity)]
    pub fn new(
        quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
        mut edge_reps: HashMap<EdgeLabel, MatrixType>,
        mut vertex_reps: HashMap<VertexLabel, MatrixType>,
        check_idempotency: Option<fn(&MatrixType, &MatrixType) -> bool>,
    ) -> Result<
        Self,
        (
            HashMap<VertexLabel, MatrixType>,
            HashMap<EdgeLabel, MatrixType>,
        ),
    > {
        edge_reps.retain(|key, _| quiver.contains_edge(key));
        vertex_reps.retain(|key, _| quiver.contains_vertex(key));
        for edge_label in quiver.edge_labels() {
            if !edge_reps.contains_key(edge_label) {
                return Err((vertex_reps, edge_reps));
            }
        }
        for vertex_label in quiver.vertex_labels() {
            if !vertex_reps.contains_key(vertex_label) {
                return Err((vertex_reps, edge_reps));
            } else if let Some(same_matrix) = check_idempotency {
                let vertex_mat = vertex_reps
                    .get(vertex_label)
                    .expect("Checked above")
                    .clone();
                let mut vertex_mat_squared = vertex_mat.clone();
                #[allow(clippy::redundant_pattern_matching)]
                if let Err(_) = vertex_mat_squared.checked_mul_assign(vertex_mat.clone()) {
                    return Err((vertex_reps, edge_reps));
                }
                if !same_matrix(&vertex_mat_squared, &vertex_mat) {
                    return Err((vertex_reps, edge_reps));
                }
            }
        }
        Ok(Self {
            quiver,
            edge_reps,
            vertex_reps,
        })
    }

    pub fn set_edge_rep(&mut self, edge: &EdgeLabel, rep: MatrixType) {
        if let Some(v) = self.edge_reps.get_mut(edge) {
            *v = rep;
        }
    }

    pub fn get_edge_rep(&self, edge: &EdgeLabel) -> Option<&MatrixType> {
        self.edge_reps.get(edge)
    }

    #[allow(clippy::must_use_candidate)]
    pub fn quiver(&self) -> &Arc<Quiver<VertexLabel, EdgeLabel>> {
        &self.quiver
    }

    ///
    ///
    /// # Errors
    ///
    /// There may be matrix dimension errors if it is a product of factors
    /// for each arrow on the path
    ///
    /// # Panics
    ///
    /// Everything on the path is actually an arrow of the quiver.
    /// Or it is just a vertex and that vertex is actually a vertex of the quiver.
    pub fn mat_from_path_or_vertex(
        &self,
        path: BasisElt<VertexLabel, EdgeLabel>,
    ) -> Result<MatrixType, <MatrixType as CheckedMulAssign>::MultiplicationError> {
        match path {
            BasisElt::Path(path) => self.mat_from_path(path),
            BasisElt::Idempotent(vertex) => Ok(self
                .vertex_reps
                .get(&vertex)
                .expect("This is a vertex of the quiver")
                .clone()),
        }
    }

    ///
    ///
    /// # Errors
    ///
    /// There may be matrix dimension errors
    ///
    /// # Panics
    ///
    /// Everything on the path is actually an arrow of the quiver.
    pub fn mat_from_path(
        &self,
        path: NonEmpty<EdgeLabel>,
    ) -> Result<MatrixType, <MatrixType as CheckedMulAssign>::MultiplicationError> {
        let first = path.first();
        let mut mat_returned = self
            .edge_reps
            .get(first)
            .expect("Everything on this path was an arrow of this quiver")
            .clone();
        for cur_edge in path {
            let rhs = self
                .edge_reps
                .get(&cur_edge)
                .expect("Everything on this path was an arrow of this quiver")
                .clone();
            mat_returned.checked_mul_assign(rhs)?;
        }
        Ok(mat_returned)
    }

    /// Check whether `quiver_hom` defines an intertwiner from `self` to `other`.
    ///
    /// For each arrow α: s → t the naturality square requires
    /// `φ_t ∘ M(α) = N(α) ∘ φ_s`, i.e. `quiver_hom[t] * self[α] == other[α] * quiver_hom[s]`.
    ///
    /// # Returns
    /// - `Ok(vec![])` — `quiver_hom` is a valid intertwiner.
    /// - `Ok(edges)` — not an intertwiner; `edges` lists every arrow whose naturality square fails.
    /// - `Err(msg)` — the data is ill-formed: a vertex present in the quiver has no matrix in
    ///   `quiver_hom`, or a multiplication failed due to incompatible dimensions.
    ///
    /// # Errors
    ///
    /// - If `quiver_hom` is missing a matrix for a vertex in the quiver.
    /// - If any matrix multiplication fails due to incompatible dimensions.
    #[allow(clippy::missing_panics_doc)]
    pub fn is_intertwiner(
        &self,
        other: &Self,
        quiver_hom: &HashMap<VertexLabel, MatrixType>,
    ) -> Result<Vec<EdgeLabel>, String>
    where
        VertexLabel: std::fmt::Debug,
        MatrixType: PartialEq,
        <MatrixType as CheckedMul>::MultiplicationError: std::fmt::Debug,
    {
        for vertex_label in self.quiver.vertex_labels() {
            if !quiver_hom.contains_key(vertex_label) {
                return Err(format!(
                    "quiver_hom is missing a matrix for {vertex_label:?} in the quiver"
                ));
            }
        }

        let mut failing_edges = Vec::new();

        for edge_label in self.quiver.edge_labels() {
            let (src, tgt) = self
                .quiver
                .edge_endpoint_labels(edge_label)
                .expect("Edge is in the quiver");

            let phi_src = quiver_hom.get(&src).expect("Checked above").clone();
            let phi_tgt = quiver_hom.get(&tgt).expect("Checked above").clone();
            let m_edge = self
                .edge_reps
                .get(edge_label)
                .expect("Edge rep exists in self")
                .clone();
            let n_edge = other
                .edge_reps
                .get(edge_label)
                .expect("Edge rep exists in other")
                .clone();

            // φ_t * M(α)
            let lhs = phi_tgt
                .checked_mul(m_edge)
                .map_err(|e| format!("Matrix multiplication error on lhs: {e:?}"))?;
            // N(α) * φ_s
            let rhs = n_edge
                .checked_mul(phi_src)
                .map_err(|e| format!("Matrix multiplication error on rhs: {e:?}"))?;

            if lhs != rhs {
                failing_edges.push(edge_label.clone());
            }
        }

        Ok(failing_edges)
    }

    ///
    ///
    /// # Errors
    ///
    /// There may be matrix dimension errors
    ///
    /// # Panics
    ///
    /// The path algebra element should all have the same endpoints
    /// so that it defines an element of `End(V_s, V_t)` for consistent s and t
    /// source and target nodes.
    pub fn mat_from_path_algebra<Coeffs>(
        &self,
        path_algebra: PathAlgebra<VertexLabel, EdgeLabel, Coeffs>,
    ) -> Result<MatrixType, CheckedArithError<MatrixType>>
    where
        Coeffs: Ring,
        MatrixType: MulAssign<Coeffs>,
    {
        assert!(path_algebra.all_parallel().is_ok());
        let mut mat_returned: Option<MatrixType> = None;
        for (path, coeff) in path_algebra {
            let mut mat_now = self
                .mat_from_path_or_vertex(path)
                .map_err(CheckedArithError::from_mul_assign)?;
            mat_now *= coeff;
            if let Some(mat_returned) = &mut mat_returned {
                mat_returned
                    .checked_add_assign(mat_now)
                    .map_err(CheckedArithError::from_add_assign)?;
            } else {
                mat_returned = Some(mat_now);
            }
        }
        Ok(mat_returned.expect("It has been set now"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quiver::Quiver;
    use std::{fmt::Debug, sync::Arc};

    // Single-edge quiver: "0" --"a"--> "1"
    fn single_edge_quiver() -> Arc<Quiver<&'static str, &'static str>> {
        let mut q = Quiver::new();
        q.add_edge("0", "1", "a");
        Arc::new(q)
    }

    fn scalar_rep<T, U>(
        q: Arc<Quiver<T, U>>,
        edges: impl IntoIterator<Item = (U, f64)>,
    ) -> QuiverRep<T, U, f64>
    where
        T: Eq + std::hash::Hash + Clone + Debug,
        U: Eq + std::hash::Hash + Clone + Debug,
    {
        let edge_reps = edges.into_iter().collect();
        let vertex_reps = q.vertex_labels().map(|v| (v.clone(), 1.0)).collect();
        QuiverRep::new(q, edge_reps, vertex_reps, None).unwrap()
    }

    // ── new ──────────────────────────────────────────────────────────────────

    #[test]
    fn new_valid() {
        let q = single_edge_quiver();
        let rep = scalar_rep(q, [("a", 2.0)]);
        assert_eq!(rep.get_edge_rep(&"a"), Some(&2.0));
    }

    #[test]
    fn new_missing_edge_is_err() {
        let q = single_edge_quiver();
        let edge_reps = HashMap::new();
        let vertex_reps = [("0", 1.0_f64), ("1", 1.0_f64)].into_iter().collect();
        assert!(QuiverRep::new(q, edge_reps, vertex_reps, None).is_err());
    }

    #[test]
    fn new_missing_vertex_is_err() {
        let q = single_edge_quiver();
        let edge_reps = [("a", 2.0_f64)].into_iter().collect();
        let vertex_reps = [("0", 1.0_f64)].into_iter().collect(); // "1" absent
        assert!(QuiverRep::new(q, edge_reps, vertex_reps, None).is_err());
    }

    #[test]
    fn new_non_idempotent_vertex_is_err() {
        // 2.0 * 2.0 = 4.0 ≠ 2.0, so this is not a valid idempotent projector
        let q = single_edge_quiver();
        let edge_reps = [("a", 1.0_f64)].into_iter().collect();
        let vertex_reps = [("0", 2.0_f64), ("1", 1.0_f64)].into_iter().collect();
        assert!(QuiverRep::new(q, edge_reps, vertex_reps, Some(f64::eq)).is_err());
    }

    #[test]
    fn new_idempotent_check_passes_for_valid_projectors() {
        // Both 0.0 and 1.0 are idempotent: 0*0=0, 1*1=1
        let q = single_edge_quiver();
        let edge_reps = [("a", 1.0_f64)].into_iter().collect();
        let vertex_reps = [("0", 1.0_f64), ("1", 0.0_f64)].into_iter().collect();
        assert!(QuiverRep::new(q, edge_reps, vertex_reps, Some(f64::eq)).is_ok());
    }

    #[test]
    fn new_spurious_keys_are_filtered() {
        let q = single_edge_quiver();
        // "b" and "2" are not in the quiver and should be silently dropped
        let edge_reps = [("a", 2.0_f64), ("b", 99.0_f64)].into_iter().collect();
        let vertex_reps = [("0", 1.0_f64), ("1", 1.0_f64), ("2", 1.0_f64)]
            .into_iter()
            .collect();
        let rep = QuiverRep::new(q, edge_reps, vertex_reps, None).unwrap();
        assert_eq!(rep.get_edge_rep(&"a"), Some(&2.0));
        assert_eq!(rep.get_edge_rep(&"b"), None);
    }

    // ── set_edge_rep / get_edge_rep ──────────────────────────────────────────

    #[test]
    fn set_edge_rep_updates_value() {
        let q = single_edge_quiver();
        let mut rep = scalar_rep(q, [("a", 2.0)]);
        rep.set_edge_rep(&"a", 7.0);
        assert_eq!(rep.get_edge_rep(&"a"), Some(&7.0));
    }

    #[test]
    fn set_edge_rep_nonexistent_is_noop() {
        let q = single_edge_quiver();
        let mut rep = scalar_rep(q, [("a", 2.0)]);
        rep.set_edge_rep(&"b", 99.0); // "b" not in the rep
        assert_eq!(rep.get_edge_rep(&"b"), None);
    }

    // ── is_intertwiner ───────────────────────────────────────────────────────
    //
    // Condition for a single edge a: 0 → 1 with scalar reps:
    //   φ[1] * M(a)  ==  N(a) * φ[0]

    #[test]
    fn is_intertwiner_valid() {
        // φ_1 * M(a) = 6 * 2 = 12  ==  N(a) * φ_0 = 4 * 3 = 12  ✓
        let q = single_edge_quiver();
        let m = scalar_rep(q.clone(), [("a", 2.0)]);
        let n = scalar_rep(q, [("a", 4.0)]);
        let phi: HashMap<_, _> = [("0", 3.0_f64), ("1", 6.0_f64)].into_iter().collect();
        assert_eq!(m.is_intertwiner(&n, &phi), Ok(vec![]));
    }

    #[test]
    fn is_intertwiner_identity_on_self() {
        // φ_v = 1 is always an intertwiner from a rep to itself
        let q = single_edge_quiver();
        let m = scalar_rep(q.clone(), [("a", 5.0)]);
        let phi: HashMap<_, _> = [("0", 1.0_f64), ("1", 1.0_f64)].into_iter().collect();
        assert_eq!(m.is_intertwiner(&m, &phi), Ok(vec![]));
    }

    #[test]
    fn is_intertwiner_zero_map_to_zero_rep() {
        // φ_v = 0 always intertwines to the zero representation
        let q = single_edge_quiver();
        let m = scalar_rep(q.clone(), [("a", 5.0)]);
        let zero = scalar_rep(q, [("a", 0.0)]);
        let phi: HashMap<_, _> = [("0", 0.0_f64), ("1", 0.0_f64)].into_iter().collect();
        assert_eq!(m.is_intertwiner(&zero, &phi), Ok(vec![]));
    }

    #[test]
    fn is_intertwiner_single_failing_edge() {
        // φ_1 * M(a) = 7 * 2 = 14  ≠  N(a) * φ_0 = 4 * 3 = 12  → "a" fails
        let q = single_edge_quiver();
        let m = scalar_rep(q.clone(), [("a", 2.0)]);
        let n = scalar_rep(q, [("a", 4.0)]);
        let phi: HashMap<_, _> = [("0", 3.0_f64), ("1", 7.0_f64)].into_iter().collect();
        assert_eq!(m.is_intertwiner(&n, &phi), Ok(vec!["a"]));
    }

    #[test]
    fn is_intertwiner_multiple_failing_edges() {
        // Kronecker quiver; both squares fail
        // φ_alpha=3, φ_beta=7  (wrong)
        // edge a: 7*2=14 ≠ 4*3=12  edge b: 7*3=21 ≠ 6*3=18
        let q = Arc::new(crate::quiver::tests::make_kronecker_quiver());
        let m = scalar_rep(q.clone(), [("a", 2.0), ("b", 3.0)]);
        let n = scalar_rep(q, [("a", 4.0), ("b", 6.0)]);
        let phi: HashMap<_, _> = [("alpha", 3.0_f64), ("beta", 7.0_f64)]
            .into_iter()
            .collect();
        let mut failing = m.is_intertwiner(&n, &phi).unwrap();
        failing.sort();
        assert_eq!(failing, vec!["a", "b"]);
    }

    #[test]
    fn is_intertwiner_partial_failure_kronecker() {
        // Only edge "b" fails: φ_beta * M(b) = 6*3=18 ≠ N(b)*φ_alpha = 7*3=21
        let q = Arc::new(crate::quiver::tests::make_kronecker_quiver());
        let m = scalar_rep(q.clone(), [("a", 2.0), ("b", 3.0)]);
        let n = scalar_rep(q, [("a", 4.0), ("b", 7.0)]);
        // φ_alpha=3, φ_beta=6 is correct for edge a (6*2=12==4*3=12) but wrong for b (6*3=18≠7*3=21)
        let phi: HashMap<_, _> = [("alpha", 3.0_f64), ("beta", 6.0_f64)]
            .into_iter()
            .collect();
        assert_eq!(m.is_intertwiner(&n, &phi), Ok(vec!["b"]));
    }

    #[test]
    fn is_intertwiner_missing_vertex_is_err() {
        let q = single_edge_quiver();
        let m = scalar_rep(q.clone(), [("a", 2.0)]);
        let n = scalar_rep(q, [("a", 4.0)]);
        let phi: HashMap<_, _> = [("0", 3.0_f64)].into_iter().collect(); // "1" absent
        assert!(m.is_intertwiner(&n, &phi).is_err());
    }

    // ── rep_descends ─────────────────────────────────────────────────────────
    //
    // A rep of kQ^op descends to kQ^{op}/<a> iff M(a) = 0.
    // We use the Kronecker quiver and the ideal generated by the single arrow "a".

    #[test]
    fn rep_descends_to_quotient_when_arrow_is_zero() {
        use crate::quiver::BasisElt;
        use crate::quiver::PathAlgebra;
        use crate::quiver_with_rels::QuiverWithRelations;

        let q = Arc::new(crate::quiver::tests::make_kronecker_quiver());
        let rel_a = PathAlgebra::singleton(
            q.clone(),
            BasisElt::Path(nonempty::nonempty!["a"]),
            4.4693_f64,
        );
        let qwr = QuiverWithRelations::new(q.clone(), vec![rel_a], Some(|x: &f64| *x == 0.0));

        // M(a) = 0 — should descend
        let m = scalar_rep(q, [("a", 0.0), ("b", 3.0)]);
        assert!(qwr.rep_descends(&m, |x: &f64| *x == 0.0));

        let q = Arc::new(crate::quiver::tests::make_kronecker_quiver());
        let rel_a =
            PathAlgebra::singleton(q.clone(), BasisElt::Path(nonempty::nonempty!["a"]), 0.0_f64);
        let qwr = QuiverWithRelations::new(q.clone(), vec![rel_a], Some(|x: &f64| *x == 0.0));

        // M(a) ≠ 0 but the ideal was actually trivial being 0*a — should descend
        let m = scalar_rep(q, [("a", 2.0), ("b", 3.0)]);
        assert!(qwr.rep_descends(&m, |x: &f64| *x == 0.0));
    }

    #[test]
    fn rep_does_not_descend_when_arrow_is_nonzero() {
        use crate::quiver::BasisElt;
        use crate::quiver::PathAlgebra;
        use crate::quiver_with_rels::QuiverWithRelations;

        let q = Arc::new(crate::quiver::tests::make_kronecker_quiver());
        let rel_a = PathAlgebra::singleton(
            q.clone(),
            BasisElt::Path(nonempty::nonempty!["a"]),
            5.93049_f64,
        );
        let qwr = QuiverWithRelations::new(q.clone(), vec![rel_a], Some(|x: &f64| *x == 0.0));

        // M(a) ≠ 0 — should not descend
        let m = scalar_rep(q, [("a", 2.0), ("b", 3.0)]);
        assert!(!qwr.rep_descends(&m, |x: &f64| *x == 0.0));
    }
}
