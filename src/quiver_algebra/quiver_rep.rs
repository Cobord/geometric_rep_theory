use std::ops::MulAssign;
use std::{collections::HashMap, sync::Arc};

use nonempty::NonEmpty;

use crate::arithmetic_utils::{
    ChainMultiplyable, CheckedAdd, CheckedAddAssign, CheckedArithError, Ring,
};
use crate::quiver_algebra::path_algebra::PathAlgebra;
use crate::quiver_algebra::quiver::{BasisElt, Quiver};

/// A representation of a quiver: a matrix `MatrixType` assigned to each arrow and a dimension
/// to each vertex, subject to the composability constraints of the quiver.
#[must_use]
pub struct QuiverRep<VertexLabel, EdgeLabel, MatrixType, const OP_ALG: bool>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
    MatrixType: CheckedAdd + CheckedAddAssign + ChainMultiplyable + Clone,
{
    quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
    edge_reps: HashMap<EdgeLabel, MatrixType>,
    vertex_dims: HashMap<VertexLabel, usize>,
    identity_producer: fn(usize) -> MatrixType,
}

impl<VertexLabel, EdgeLabel, MatrixType, const OP_ALG: bool>
    QuiverRep<VertexLabel, EdgeLabel, MatrixType, OP_ALG>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
    MatrixType: CheckedAdd + CheckedAddAssign + ChainMultiplyable + Clone,
{
    /// Construct a representation of `quiver` from maps on edges and vertices.
    ///
    /// Keys in `edge_reps` or `vertex_dims` that do not belong to the quiver are silently
    /// discarded.
    ///
    /// # Errors
    ///
    /// Returns `Err((vertex_reps, edge_reps))` if any arrow in the quiver is missing a matrix
    /// entry, any vertex is missing a dimension.
    #[allow(clippy::missing_panics_doc, clippy::type_complexity)]
    pub fn new(
        quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
        mut edge_reps: HashMap<EdgeLabel, MatrixType>,
        mut vertex_dims: HashMap<VertexLabel, usize>,
        identity_producer: fn(usize) -> MatrixType,
    ) -> Result<Self, (HashMap<VertexLabel, usize>, HashMap<EdgeLabel, MatrixType>)> {
        edge_reps.retain(|key, _| quiver.contains_edge(key));
        vertex_dims.retain(|key, _| quiver.contains_vertex(key));
        for edge_label in quiver.edge_labels() {
            if !edge_reps.contains_key(edge_label) {
                return Err((vertex_dims, edge_reps));
            }
        }
        for vertex_label in quiver.vertex_labels() {
            if !vertex_dims.contains_key(vertex_label) {
                return Err((vertex_dims, edge_reps));
            }
        }
        Ok(Self {
            quiver,
            edge_reps,
            vertex_dims,
            identity_producer,
        })
    }

    /// Construct the representation of the quiver
    /// that assigns each vertex a vector space of
    /// the given dimension and each edge the zero map.
    ///
    /// # Panics
    ///
    /// If any vertex in the quiver is missing from `dim_vector`.
    pub fn new_zero_rep(
        quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
        mut dim_vector: HashMap<VertexLabel, usize>,
        mut zero_matrix: impl FnMut(usize, usize) -> MatrixType,
        id_matrix: fn(usize) -> MatrixType,
    ) -> Self {
        dim_vector.retain(|key, _| quiver.contains_vertex(key));
        let edge_reps = quiver
            .edge_labels()
            .map(|e| {
                let (src, tgt) = quiver
                    .edge_endpoint_labels(e)
                    .expect("edge is in the quiver");
                let src_dim = *dim_vector
                    .get(&src)
                    .expect("Every vertex in the quiver has a dimension");
                let tgt_dim = *dim_vector
                    .get(&tgt)
                    .expect("Every vertex in the quiver has a dimension");
                (e.clone(), zero_matrix(tgt_dim, src_dim))
            })
            .collect();

        assert_eq!(
            dim_vector.len(),
            quiver.vertex_labels().count(),
            "Every vertex in the quiver has a dimension"
        );
        Self {
            quiver,
            edge_reps,
            vertex_dims: dim_vector,
            identity_producer: id_matrix,
        }
    }

    /// Replace the matrix assigned to `edge`. Does nothing if `edge` is not in the quiver.
    pub fn set_edge_rep(&mut self, edge: &EdgeLabel, rep: MatrixType) {
        if let Some(v) = self.edge_reps.get_mut(edge) {
            *v = rep;
        }
    }

    /// Return the matrix assigned to `edge`, or `None` if `edge` is not in the quiver.
    pub fn get_edge_rep(&self, edge: &EdgeLabel) -> Option<&MatrixType> {
        self.edge_reps.get(edge)
    }

    /// The underlying quiver this representation is defined over.
    #[allow(clippy::must_use_candidate)]
    pub fn quiver(&self) -> &Arc<Quiver<VertexLabel, EdgeLabel>> {
        &self.quiver
    }

    /// Evaluate the representation on a basis element (path or idempotent).
    ///
    /// For a path `[a₁, …, aₙ]` returns `M(aₙ) * … * M(a₁)`.
    /// For an idempotent `e_v` returns
    /// the identity matrix of the appropriate dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if any matrix multiplication fails due to incompatible dimensions.
    ///
    /// # Panics
    ///
    /// Panics if any arrow on the path or the vertex is not present in the quiver.
    pub fn mat_from_path_or_vertex(
        &self,
        path: BasisElt<VertexLabel, EdgeLabel>,
    ) -> Result<MatrixType, <MatrixType as ChainMultiplyable>::MultiplicationError> {
        match path {
            BasisElt::Path(path) => self.mat_from_path(&path),
            BasisElt::Idempotent(vertex) => {
                let vertex_dim = *self
                    .vertex_dims
                    .get(&vertex)
                    .expect("This is a vertex of the quiver");
                Ok((self.identity_producer)(vertex_dim))
            }
        }
    }

    /// Evaluate the representation on a non-empty path, returning `M(aₙ) * … * M(a₁)`.
    ///
    /// # Errors
    ///
    /// Returns an error if any matrix multiplication fails due to incompatible dimensions.
    ///
    /// # Panics
    ///
    /// Panics if any arrow on the path is not present in the quiver.
    pub fn mat_from_path(
        &self,
        path: &NonEmpty<EdgeLabel>,
    ) -> Result<MatrixType, <MatrixType as ChainMultiplyable>::MultiplicationError> {
        let first = path.first();
        let mut mat_returned = self
            .edge_reps
            .get(first)
            .expect("Everything on this path was an arrow of this quiver")
            .clone();
        mat_returned = mat_returned.chain_multiply_after(path.tail().iter().map(|cur_edge| {
            self.edge_reps
                .get(cur_edge)
                .expect("Everything on this path was an arrow of this quiver")
                .clone()
        }))?;
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
        closeness: fn(&MatrixType, &MatrixType) -> bool,
    ) -> Result<Vec<EdgeLabel>, String>
    where
        VertexLabel: std::fmt::Debug,
        MatrixType: PartialEq,
        <MatrixType as ChainMultiplyable>::MultiplicationError: std::fmt::Debug,
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
            let lhs = MatrixType::mul_two(m_edge, phi_tgt)
                .map_err(|e| format!("Matrix multiplication error on lhs: {e:?}"))?;
            // N(α) * φ_s
            let rhs = MatrixType::mul_two(phi_src, n_edge)
                .map_err(|e| format!("Matrix multiplication error on rhs: {e:?}"))?;

            if !closeness(&lhs, &rhs) {
                failing_edges.push(edge_label.clone());
            }
        }

        Ok(failing_edges)
    }

    /// Evaluate the representation on an element of the path algebra, returning the corresponding
    /// linear combination of matrices: Σ cᵢ · M(pᵢ).
    ///
    /// `PathAlgebra<VertexLabel, EdgeLabel, Coeffs>` was `kQ^op`
    /// But here we are giving the left module structure for `kQ`
    /// So for this actually represents `\rho(e_tgt) \rho(S(path_algebra)) \rho(e_src)`
    /// which because each `M_v` have been given bases, this is in
    /// `MatrixType` which is only the relevant `dim(tgt)` by `dim(src)` matrix
    ///
    /// # Errors
    ///
    /// Returns an error if any matrix multiplication or addition fails due to incompatible
    /// dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the summands of `path_algebra` do not all share the same source and target
    /// vertices (i.e. the element is not in a single Peirce piece).
    pub fn mat_from_path_algebra<Coeffs>(
        &self,
        path_algebra: PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>,
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
                .map_err(CheckedArithError::from_mul)?;
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

    /// Apply a gauge transformation (change of basis at each vertex) to the representation.
    ///
    /// For each vertex `v`, `gauge_transformation[v]` is the pair `(g_v⁻¹, g_v)` — the caller
    /// must supply both the matrix and its inverse, since the code does not compute inverses.
    ///
    /// For each edge `a: src → tgt`, the edge map is updated by:
    /// ```text
    /// M_a' = g[src].0 · M_a · g[tgt].1
    ///      = g_src⁻¹ · M_a · g_tgt
    /// ```
    /// Vertices absent from the map are treated as having the identity transformation.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any matrix multiplication fails (e.g. shape mismatch).
    #[allow(clippy::missing_panics_doc)]
    pub fn gauge_transform(
        &mut self,
        gauge_transformation: &HashMap<VertexLabel, (MatrixType, MatrixType)>,
    ) -> Result<(), CheckedArithError<MatrixType>> {
        for a in self.quiver().edge_labels().cloned().collect::<Vec<_>>() {
            let old_rep = self.get_edge_rep(&a).cloned();
            let new_rep;
            let (src, tgt) = self
                .quiver()
                .edge_endpoint_labels(&a)
                .expect("This is an edge of the quiver");
            let src_transform = gauge_transformation.get(&src).map(|z| z.0.clone());
            let tgt_transform = gauge_transformation.get(&tgt).map(|z| z.1.clone());
            match (src_transform, tgt_transform) {
                (None, None) => {
                    new_rep = None;
                }
                (None, Some(tgt_transform)) => {
                    if let Some(old_rep) = old_rep {
                        new_rep = Some(
                            ChainMultiplyable::mul_two(old_rep, tgt_transform.clone())
                                .map_err(CheckedArithError::from_mul)?,
                        );
                    } else {
                        new_rep = Some(tgt_transform.clone());
                    }
                }
                (Some(src_inv), None) => {
                    let to_set;
                    if let Some(old_rep) = old_rep {
                        to_set = ChainMultiplyable::mul_two(src_inv, old_rep)
                            .map_err(CheckedArithError::from_mul)?;
                    } else {
                        to_set = src_inv;
                    }
                    new_rep = Some(to_set);
                }
                (Some(src_inv), Some(tgt_transform)) => {
                    let to_set;
                    if let Some(old_rep) = old_rep {
                        to_set = src_inv
                            .chain_multiply_after([old_rep, tgt_transform.clone()])
                            .map_err(CheckedArithError::from_mul)?;
                    } else {
                        to_set = ChainMultiplyable::mul_two(src_inv, tgt_transform.clone())
                            .map_err(CheckedArithError::from_mul)?;
                    }
                    new_rep = Some(to_set);
                }
            }
            if let Some(to_set) = new_rep {
                self.set_edge_rep(&a, to_set);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quiver_algebra::quiver::Quiver;
    use proptest::{prelude::Strategy, proptest};
    use std::{fmt::Debug, sync::Arc};

    // Single-edge quiver: "0" --"a"--> "1"
    fn single_edge_quiver() -> Arc<Quiver<&'static str, &'static str>> {
        let mut q = Quiver::new();
        q.add_edge("0", "1", "a");
        Arc::new(q)
    }

    fn scalar_rep<T, U, Scalar>(
        q: Arc<Quiver<T, U>>,
        edges: impl IntoIterator<Item = (U, Scalar)>,
    ) -> Result<QuiverRep<T, U, Scalar, true>, (HashMap<T, usize>, HashMap<U, Scalar>)>
    where
        T: Eq + std::hash::Hash + Clone + Debug,
        U: Eq + std::hash::Hash + Clone + Debug,
        Scalar: Ring + Clone,
    {
        let edge_reps = edges.into_iter().collect();
        let vertex_dims = q.vertex_labels().map(|v| (v.clone(), 1)).collect();
        QuiverRep::new(q, edge_reps, vertex_dims, |_| Scalar::one())
    }

    // ── new ──────────────────────────────────────────────────────────────────

    #[test]
    fn new_valid() {
        let q = single_edge_quiver();
        let rep = scalar_rep(q, [("a", 2.0)]).expect("Valid quiver rep");
        assert_eq!(rep.get_edge_rep(&"a"), Some(&2.0));
    }

    #[test]
    fn new_missing_edge_is_err() {
        let q = single_edge_quiver();
        let rep = scalar_rep(q, [("0", 1), ("1", 1)]);
        assert!(rep.is_err());
    }

    #[test]
    fn new_missing_vertex_is_err() {
        let q = single_edge_quiver();
        let edge_reps = [("a", 2.0_f64)].into_iter().collect();
        // "1" absent
        let vertex_dims = [("0", 1)].into_iter().collect();
        assert!(QuiverRep::<_, _, _, true>::new(q, edge_reps, vertex_dims, |_| 1.0).is_err());
    }

    #[test]
    fn new_spurious_keys_are_filtered() {
        let q = single_edge_quiver();
        // "b" and "2" are not in the quiver and should be silently dropped
        let edge_reps = [("a", 2.0_f64), ("b", 99.0_f64)].into_iter().collect();
        let vertex_dims = [("0", 1), ("1", 1), ("2", 1)].into_iter().collect();
        let rep = QuiverRep::<_, _, _, true>::new(q, edge_reps, vertex_dims, |_| 1.0).unwrap();
        assert_eq!(rep.get_edge_rep(&"a"), Some(&2.0));
        assert_eq!(rep.get_edge_rep(&"b"), None);
    }

    // ── set_edge_rep / get_edge_rep ──────────────────────────────────────────

    #[test]
    fn set_edge_rep_updates_value() {
        let q = single_edge_quiver();
        let mut rep = scalar_rep(q, [("a", 2.0)]).expect("Valid quiver rep");
        rep.set_edge_rep(&"a", 7.0);
        assert_eq!(rep.get_edge_rep(&"a"), Some(&7.0));
    }

    #[test]
    fn set_edge_rep_nonexistent_is_noop() {
        let q = single_edge_quiver();
        let mut rep = scalar_rep(q, [("a", 2.0)]).expect("Valid quiver rep");
        rep.set_edge_rep(&"b", 99.0); // "b" not in the rep
        assert_eq!(rep.get_edge_rep(&"b"), None);
        assert_eq!(rep.get_edge_rep(&"a"), Some(&2.0));
    }

    // ── is_intertwiner ───────────────────────────────────────────────────────
    //
    // Condition for a single edge a: 0 → 1 with scalar reps:
    //   φ[1] * M(a)  ==  N(a) * φ[0]

    #[test]
    fn is_intertwiner_valid() {
        // φ_1 * M(a) = 6 * 2 = 12  ==  N(a) * φ_0 = 4 * 3 = 12  ✓
        let q = single_edge_quiver();
        let m = scalar_rep(q.clone(), [("a", 2.0)]).expect("Valid quiver rep");
        let n = scalar_rep(q, [("a", 4.0)]).expect("Valid quiver rep");
        let phi: HashMap<_, _> = [("0", 3.0_f64), ("1", 6.0_f64)].into_iter().collect();
        assert_eq!(
            m.is_intertwiner(&n, &phi, |a, b| (a - b).abs() <= f64::EPSILON),
            Ok(vec![])
        );
    }

    proptest! {
        #[test]
        fn is_intertwiner_identity_on_self(
            scaling_factor in proptest::num::f64::ANY.prop_filter("big",|z| z.abs() < 100.0),
        ) {
            let q = single_edge_quiver();
            let m = scalar_rep(q.clone(), [("a", 5.0)]).expect("Valid quiver rep");
            let phi: HashMap<_, _> = [("0", scaling_factor), ("1", scaling_factor)].into_iter().collect();
            assert_eq!(m.is_intertwiner(&m, &phi, |a,b| (a-b).abs() <= f64::EPSILON), Ok(vec![]));
        }
    }

    #[test]
    fn is_intertwiner_zero_map_to_zero_rep() {
        // φ_v = 0 always intertwines to the zero representation
        let q = single_edge_quiver();
        let m = scalar_rep(q.clone(), [("a", 5.0)]).expect("Valid quiver rep");
        let zero = scalar_rep(q, [("a", 0.0)]).expect("Valid quiver rep");
        let phi: HashMap<_, _> = [("0", 0.0_f64), ("1", 0.0_f64)].into_iter().collect();
        assert_eq!(
            m.is_intertwiner(&zero, &phi, |a, b| (a - b).abs() <= f64::EPSILON),
            Ok(vec![])
        );
    }

    #[test]
    fn is_intertwiner_single_failing_edge() {
        // φ_1 * M(a) = 7 * 2 = 14  ≠  N(a) * φ_0 = 4 * 3 = 12  → "a" fails
        let q = single_edge_quiver();
        let m = scalar_rep(q.clone(), [("a", 2.0)]).expect("Valid quiver rep");
        let n = scalar_rep(q, [("a", 4.0)]).expect("Valid quiver rep");
        let phi: HashMap<_, _> = [("0", 3.0_f64), ("1", 7.0_f64)].into_iter().collect();
        assert_eq!(
            m.is_intertwiner(&n, &phi, |a, b| (a - b).abs() <= f64::EPSILON),
            Ok(vec!["a"])
        );
    }

    #[test]
    fn is_intertwiner_multiple_failing_edges() {
        // Kronecker quiver; both squares fail
        // φ_alpha=3, φ_beta=7  (wrong)
        // edge a: 7*2=14 ≠ 4*3=12  edge b: 7*3=21 ≠ 6*3=18
        let q = Arc::new(crate::quiver_algebra::quiver::tests::make_kronecker_quiver());
        let m = scalar_rep(q.clone(), [("a", 2.0), ("b", 3.0)]).expect("Valid quiver rep");
        let n = scalar_rep(q, [("a", 4.0), ("b", 6.0)]).expect("Valid quiver rep");
        let phi: HashMap<_, _> = [("alpha", 3.0_f64), ("beta", 7.0_f64)]
            .into_iter()
            .collect();
        let mut failing = m
            .is_intertwiner(&n, &phi, |a, b| (a - b).abs() <= f64::EPSILON)
            .unwrap();
        failing.sort();
        assert_eq!(failing, vec!["a", "b"]);
    }

    #[test]
    fn is_intertwiner_partial_failure_kronecker() {
        // Only edge "b" fails: φ_beta * M(b) = 6*3=18 ≠ N(b)*φ_alpha = 7*3=21
        let q = Arc::new(crate::quiver_algebra::quiver::tests::make_kronecker_quiver());
        let m = scalar_rep(q.clone(), [("a", 2.0), ("b", 3.0)]).expect("Valid quiver rep");
        let n = scalar_rep(q, [("a", 4.0), ("b", 7.0)]).expect("Valid quiver rep");
        // φ_alpha=3, φ_beta=6 is correct for edge a (6*2=12==4*3=12) but wrong for b (6*3=18≠7*3=21)
        let phi: HashMap<_, _> = [("alpha", 3.0_f64), ("beta", 6.0_f64)]
            .into_iter()
            .collect();
        assert_eq!(
            m.is_intertwiner(&n, &phi, |a, b| (a - b).abs() <= f64::EPSILON),
            Ok(vec!["b"])
        );
    }

    #[test]
    fn is_intertwiner_missing_vertex_is_err() {
        let q = single_edge_quiver();
        let m = scalar_rep(q.clone(), [("a", 2.0)]).expect("Valid quiver rep");
        let n = scalar_rep(q, [("a", 4.0)]).expect("Valid quiver rep");
        let phi: HashMap<_, _> = [("0", 3.0_f64)].into_iter().collect(); // "1" absent
        assert!(
            m.is_intertwiner(&n, &phi, |a, b| (a - b).abs() <= f64::EPSILON)
                .is_err()
        );
    }

    // ── rep_descends ─────────────────────────────────────────────────────────
    //
    // A rep of kQ^op descends to kQ^{op}/<a> iff M(a) = 0.
    // We use the Kronecker quiver and the ideal generated by the single arrow "a".

    #[test]
    fn rep_descends_to_quotient_when_arrow_is_zero() {
        use crate::quiver_algebra::path_algebra::PathAlgebra;
        use crate::quiver_algebra::quiver::BasisElt;
        use crate::quiver_algebra::quiver_with_rels::QuiverWithRelations;

        let q = Arc::new(crate::quiver_algebra::quiver::tests::make_kronecker_quiver());
        let rel_a = PathAlgebra::singleton(
            q.clone(),
            BasisElt::Path(nonempty::nonempty!["a"]),
            4.4693_f64,
        );
        let qwr = QuiverWithRelations::new(q.clone(), vec![rel_a], Some(|x: &f64| *x == 0.0));

        // M(a) = 0 — should descend
        let m = scalar_rep(q, [("a", 0.0), ("b", 3.0)]).expect("Valid quiver rep");
        assert!(qwr.rep_descends(&m, |x: &f64| *x == 0.0));

        let q = Arc::new(crate::quiver_algebra::quiver::tests::make_kronecker_quiver());
        let rel_a =
            PathAlgebra::singleton(q.clone(), BasisElt::Path(nonempty::nonempty!["a"]), 0.0_f64);
        let qwr = QuiverWithRelations::new(q.clone(), vec![rel_a], Some(|x: &f64| *x == 0.0));

        // M(a) ≠ 0 but the ideal was actually trivial being 0*a — should descend
        let m = scalar_rep(q, [("a", 2.0), ("b", 3.0)]).expect("Valid quiver rep");
        assert!(qwr.rep_descends(&m, |x: &f64| *x == 0.0));
    }

    #[test]
    fn rep_does_not_descend_when_arrow_is_nonzero() {
        use crate::quiver_algebra::path_algebra::PathAlgebra;
        use crate::quiver_algebra::quiver::BasisElt;
        use crate::quiver_algebra::quiver_with_rels::QuiverWithRelations;

        let q = Arc::new(crate::quiver_algebra::quiver::tests::make_kronecker_quiver());
        let rel_a = PathAlgebra::singleton(
            q.clone(),
            BasisElt::Path(nonempty::nonempty!["a"]),
            5.93049_f64,
        );
        let qwr = QuiverWithRelations::new(q.clone(), vec![rel_a], Some(|x: &f64| *x == 0.0));

        // M(a) ≠ 0 — should not descend
        let m = scalar_rep(q, [("a", 2.0), ("b", 3.0)]).expect("Valid quiver rep");
        assert!(!qwr.rep_descends(&m, |x: &f64| *x == 0.0));
    }

    /// Oriented triangle quiver: a → b → c → a (one arrow on each edge).
    /// Dimension vector: dim(a)=1, dim(b)=2, dim(c)=3.
    /// Zero rep: all edge matrices are zero, vertex matrices are identity.
    /// mat_from_path_algebra on any path must produce the zero matrix of the right shape.
    #[test]
    fn zero_rep_triangle_mat_from_path_algebra() {
        use crate::arithmetic_utils::DynMatrix;
        use crate::quiver_algebra::path_algebra::PathAlgebra;
        use crate::quiver_algebra::quiver::BasisElt;
        use nonempty::nonempty;

        let mut q: Quiver<&str, &str> = Quiver::new();
        q.add_edge("a", "b", "ab");
        q.add_edge("b", "c", "bc");
        q.add_edge("c", "a", "ca");
        let q = Arc::new(q);

        let dim: HashMap<&str, usize> = [("a", 1), ("b", 2), ("c", 3)].into_iter().collect();

        let rep = QuiverRep::new_zero_rep(
            q.clone(),
            dim,
            |r, c| DynMatrix::<f64>::zeros(r, c),
            |n| DynMatrix::<f64>::identity(n),
        );

        // Single arrows: each gives the zero matrix of the right shape
        for (arrow, expected_rows, expected_cols) in [("ab", 2, 1), ("bc", 3, 2), ("ca", 1, 3)] {
            let elt = PathAlgebra::<_, _, _, true>::singleton(
                q.clone(),
                BasisElt::Path(nonempty![arrow]),
                1.0_f64,
            );
            let mat = rep
                .mat_from_path_algebra(elt)
                .map_err(|_| ())
                .expect("This should not fail");
            assert_eq!(mat, DynMatrix::zeros(expected_rows, expected_cols));
        }
        let ab_mat = rep
            .mat_from_path(&nonempty!["ab"])
            .map_err(|_| ())
            .expect("This should not fail");
        let bc_mat = rep
            .mat_from_path(&nonempty!["bc"])
            .map_err(|_| ())
            .expect("This should not fail");
        let _ = DynMatrix::nonempty_chain_multiply(nonempty![ab_mat, bc_mat])
            .expect("Multiplication should work");

        // Length-2 path ab·bc: 3×1 zero
        assert_eq!(
            rep.mat_from_path(&nonempty!["ab", "bc"])
                .map_err(|_| ())
                .expect("This should not fail"),
            DynMatrix::zeros(3, 1)
        );
        let ab_bc =
            PathAlgebra::singleton(q.clone(), BasisElt::Path(nonempty!["ab", "bc"]), 1.0_f64);
        assert_eq!(
            rep.mat_from_path_algebra(ab_bc)
                .map_err(|_| ())
                .expect("This should not fail"),
            DynMatrix::zeros(3, 1)
        );

        // Full cycle ab·bc·ca: should be 1×1 zero (same source and target: "a")
        let cycle = PathAlgebra::singleton(
            q.clone(),
            BasisElt::Path(nonempty!["ab", "bc", "ca"]),
            1.0_f64,
        );
        assert_eq!(
            rep.mat_from_path_algebra(cycle)
                .map_err(|_| ())
                .expect("This should not fail"),
            DynMatrix::zeros(1, 1)
        );
    }

    /// Same oriented triangle, now with non-zero matrices set via set_edge_rep.
    ///
    /// M(ab) = [[1],[0]]        (2×1)
    /// M(bc) = [[1,0],[0,1],[1,1]]  (3×2)
    /// M(ca) = [[1,0,1]]           (1×3)
    ///
    /// In kQ^op (mat_from_path multiplies right-to-left):
    ///   [ab]       = M(ab)                        = [[1],[0]]     (2×1)
    ///   [ab,bc]    = M(bc)·M(ab)                  = [[1],[0],[1]] (3×1)
    ///   [ab,bc,ca] = M(ca)·M(bc)·M(ab)            = [[2]]         (1×1)
    #[test]
    fn nonzero_rep_triangle_mat_from_path() {
        use crate::arithmetic_utils::DynMatrix;
        use crate::quiver_algebra::path_algebra::PathAlgebra;
        use crate::quiver_algebra::quiver::BasisElt;
        use nalgebra::DMatrix;
        use nonempty::nonempty;

        let mut q: Quiver<&str, &str> = Quiver::new();
        q.add_edge("a", "b", "ab");
        q.add_edge("b", "c", "bc");
        q.add_edge("c", "a", "ca");
        let q = Arc::new(q);

        let dim: HashMap<&str, usize> = [("a", 1), ("b", 2), ("c", 3)].into_iter().collect();

        let mut rep = QuiverRep::new_zero_rep(
            q.clone(),
            dim,
            |r, c| DynMatrix::<f64>::zeros(r, c),
            |n| DynMatrix::<f64>::identity(n),
        );

        rep.set_edge_rep(&"ab", DynMatrix(DMatrix::from_row_slice(2, 1, &[1.0, 0.0])));
        rep.set_edge_rep(
            &"bc",
            DynMatrix(DMatrix::from_row_slice(
                3,
                2,
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            )),
        );
        rep.set_edge_rep(
            &"ca",
            DynMatrix(DMatrix::from_row_slice(1, 3, &[1.0, 0.0, 1.0])),
        );

        let get = |path| {
            rep.mat_from_path(&path)
                .map_err(|_| ())
                .expect("All matrices along a path can multiply in that order")
        };

        assert_eq!(
            get(nonempty!["ab"]),
            DynMatrix(DMatrix::from_row_slice(2, 1, &[1.0, 0.0]))
        );

        assert_eq!(
            get(nonempty!["ab", "bc"]),
            DynMatrix(DMatrix::from_row_slice(3, 1, &[1.0, 0.0, 1.0]))
        );

        assert_eq!(
            get(nonempty!["ab", "bc", "ca"]),
            DynMatrix(DMatrix::from_row_slice(1, 1, &[2.0]))
        );

        let elt = PathAlgebra::<_, _, _, false>::singleton(
            q.clone(),
            BasisElt::Path(nonempty!["ab", "bc"]),
            3.0_f64,
        ) * 4.493;
        assert_eq!(
            rep.mat_from_path_algebra(elt).map_err(|_| ()).expect("ok"),
            DynMatrix(DMatrix::from_row_slice(3, 1, &[3.0, 0.0, 3.0]) * 4.493)
        );
    }
}
