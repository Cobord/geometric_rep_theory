use std::{collections::HashMap, fmt::Debug, ops::MulAssign, sync::Arc};

use nalgebra::DMatrix;

use crate::{
    checked_arith::{CheckedAdd, CheckedAddAssign, CheckedMul, Ring},
    quiver::BasisElt,
    quiver_with_mon_rels::{NonMonomialIdeal, QuiverWithMonomialRelations},
    quiver_with_rels::QuiverWithRelations,
};

/// An (A, A)-bimodule over A = kQ/I, accessed through its Peirce decomposition.
///
/// The bimodule M decomposes as `M = ⊕_{v,w} e_v M e_w`.  Implementors provide
/// the linear maps for the left and right A-actions arrow-by-arrow.
///
/// **Conventions** (paths are left-to-right: source first, target last):
///
/// - [`left_action`](QuiverBimodule::left_action)`(α, w)` returns the map
///   `L_{α,w}: e_{t(α)} M e_w → e_{s(α)} M e_w`
///   induced by left-multiplying by `α: s(α)→t(α)`.
///   Left multiplication is *contravariant* in the left Peirce index.
///
/// - [`right_action`](QuiverBimodule::right_action)`(β, v)` returns the map
///   `R_{β,v}: e_v M e_{s(β)} → e_v M e_{t(β)}`
///   induced by right-multiplying by `β: s(β)→t(β)`.
///   Right multiplication is *covariant* in the right Peirce index.
pub trait QuiverBimodule<VertexLabel, EdgeLabel, Coeffs, M>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
    Coeffs: Ring,
{
    /// The algebra A = kQ/I over which this is a bimodule.
    fn algebra(&self) -> &Arc<QuiverWithRelations<VertexLabel, EdgeLabel, Coeffs>>;

    /// The map `L_{α,w}: e_{t(α)} M e_w → e_{s(α)} M e_w`.
    ///
    /// Returns `None` when `alpha` or `w` is not in the quiver.
    fn left_action(&self, alpha: &EdgeLabel, w: &VertexLabel) -> Option<M>;

    /// The map `R_{β,v}: e_v M e_{s(β)} → e_v M e_{t(β)}`.
    ///
    /// Returns `None` when `beta` or `v` is not in the quiver.
    fn right_action(&self, beta: &EdgeLabel, v: &VertexLabel) -> Option<M>;

    /// Check the three (A, A)-bimodule axioms.
    ///
    /// Returns `Ok(violations)` — empty iff all axioms hold — or `Err(msg)`
    /// when the action data is structurally ill-formed.
    ///
    /// ## Axioms
    ///
    /// 1. **Left relations**: for each relation ρ = Σᵢ cᵢ pᵢ in I and each
    ///    right-index vertex w, the map Σᵢ cᵢ L_{pᵢ,w} = 0.
    ///
    /// 2. **Right relations**: for each relation ρ and each left-index vertex v,
    ///    Σᵢ cᵢ R_{pᵢ,v} = 0  (where R_{pᵢ,v} = R_{rₙ,v} ∘ ⋯ ∘ R_{r₁,v}).
    ///
    /// 3. **Commutativity**: for every pair of arrows α: i→j and β: p→q,
    ///    L_{α,q} ∘ R_{β,j} = R_{β,i} ∘ L_{α,p}  (maps M_{j,p} → M_{i,q}).
    ///
    /// # Errors
    ///
    /// If a required matrix is missing or a multiplication fails.
    #[allow(clippy::too_many_lines)]
    fn check_bimodule_axioms<M2>(
        &self,
        promotion: impl Fn(M) -> M2,
        is_zero: impl Fn(&M2) -> bool,
    ) -> Result<Vec<BimoduleAxiomViolation<VertexLabel, EdgeLabel>>, String>
    where
        VertexLabel: std::hash::Hash + Eq + Clone + Debug,
        EdgeLabel: std::hash::Hash + Eq + Clone + Debug,
        Coeffs: Ring + Clone,
        M2: Clone + PartialEq + CheckedMul + CheckedAdd + CheckedAddAssign + MulAssign<Coeffs>,
        <M2 as CheckedMul>::MultiplicationError: Debug,
        <M2 as CheckedAdd>::AdditionError: Debug,
        <M2 as CheckedAddAssign>::AdditionError: Debug,
    {
        let mut violations = Vec::new();
        let algebra = self.algebra();
        let quiver = algebra.quiver();

        // ── Axiom 1: left relations ───────────────────────────────────────────
        // For each relation ρ and each right-index w, compute
        //   Σᵢ cᵢ · (L_{a₁,w} * L_{a₂,w} * … * L_{aₙ,w})  and check zero.
        for (rel_idx, rel) in algebra.relations().enumerate() {
            if !rel.might_be_nonzero() {
                continue;
            }
            for w in quiver.vertex_labels() {
                let mut action_sum: Option<M2> = None;

                for (basis_elt, coeff) in rel.clone() {
                    let BasisElt::Path(word) = basis_elt else {
                        continue; // idempotent terms handled by algebra structure
                    };
                    // compose: L_{a₁,w} * L_{a₂,w} * …
                    let mat = self
                        .left_action(word.first(), w)
                        .ok_or_else(|| format!("Missing L_({:?},{w:?})", word.first()))?;
                    let mut mat = promotion(mat);
                    for arrow in word.tail() {
                        let next = self
                            .left_action(arrow, w)
                            .ok_or_else(|| format!("Missing L_({arrow:?},{w:?})"))?;
                        let next = promotion(next);
                        mat = mat
                            .checked_mul(next)
                            .map_err(|e| format!("Left action composition error: {e:?}"))?;
                    }
                    mat *= coeff;
                    match &mut action_sum {
                        None => action_sum = Some(mat),
                        Some(s) => s
                            .checked_add_assign(mat)
                            .map_err(|e| format!("Left action sum error: {e:?}"))?,
                    }
                }
                #[allow(clippy::collapsible_if)]
                if let Some(ref a) = action_sum {
                    if !is_zero(a) {
                        violations.push(BimoduleAxiomViolation::LeftRelationFails {
                            relation_index: rel_idx,
                            right_vertex: w.clone(),
                        });
                    }
                }
            }
        }

        // ── Axiom 2: right relations ──────────────────────────────────────────
        // For each relation ρ and each left-index v, compute
        //   Σᵢ cᵢ · (R_{aₙ,v} * … * R_{a₁,v})  and check zero.
        // Build R_{aₙ} * … * R_{a₁} by starting from R_{a₁} and prepending each
        // subsequent factor on the left: mat ← R_{aᵢ} * mat.
        for (rel_idx, rel) in algebra.relations().enumerate() {
            if !rel.might_be_nonzero() {
                continue;
            }
            for v in quiver.vertex_labels() {
                let mut action_sum: Option<M2> = None;

                for (basis_elt, coeff) in rel.clone() {
                    let BasisElt::Path(word) = basis_elt else {
                        continue;
                    };
                    let mat = self
                        .right_action(word.first(), v)
                        .ok_or_else(|| format!("Missing R_({:?},{v:?})", word.first()))?;
                    let mut mat = promotion(mat);
                    for arrow in word.tail() {
                        let next = self
                            .right_action(arrow, v)
                            .ok_or_else(|| format!("Missing R_({arrow:?},{v:?})"))?;
                        let next = promotion(next);
                        // R_{aᵢ} ∘ (previous) — prepend on the left
                        mat = next
                            .checked_mul(mat)
                            .map_err(|e| format!("Right action composition error: {e:?}"))?;
                    }
                    mat *= coeff;
                    match &mut action_sum {
                        None => action_sum = Some(mat),
                        Some(s) => s
                            .checked_add_assign(mat)
                            .map_err(|e| format!("Right action sum error: {e:?}"))?,
                    }
                }
                #[allow(clippy::collapsible_if)]
                if let Some(ref a) = action_sum {
                    if !is_zero(a) {
                        violations.push(BimoduleAxiomViolation::RightRelationFails {
                            relation_index: rel_idx,
                            left_vertex: v.clone(),
                        });
                    }
                }
            }
        }

        // ── Axiom 3: commutativity ────────────────────────────────────────────
        // For α: src_a→tgt_a and β: src_b→tgt_b:
        //   L_{α, tgt_b} * R_{β, tgt_a}  =  R_{β, src_a} * L_{α, src_b}.
        for alpha in quiver.edge_labels() {
            let Some((src_a, tgt_a)) = quiver.edge_endpoint_labels(alpha) else {
                continue;
            };
            for beta in quiver.edge_labels() {
                let Some((src_b, tgt_b)) = quiver.edge_endpoint_labels(beta) else {
                    continue;
                };

                let l_alpha_tgtb = self
                    .left_action(alpha, &tgt_b)
                    .ok_or_else(|| format!("Missing L_({alpha:?},{tgt_b:?})"))?;
                let r_beta_tgta = self
                    .right_action(beta, &tgt_a)
                    .ok_or_else(|| format!("Missing R_({beta:?},{tgt_a:?})"))?;
                let r_beta_srca = self
                    .right_action(beta, &src_a)
                    .ok_or_else(|| format!("Missing R_({beta:?},{src_a:?})"))?;
                let l_alpha_srcb = self
                    .left_action(alpha, &src_b)
                    .ok_or_else(|| format!("Missing L_({alpha:?},{src_b:?})"))?;

                let lhs = promotion(l_alpha_tgtb)
                    .checked_mul(promotion(r_beta_tgta))
                    .map_err(|e| format!("Commutativity lhs error: {e:?}"))?;
                let rhs = promotion(r_beta_srca)
                    .checked_mul(promotion(l_alpha_srcb))
                    .map_err(|e| format!("Commutativity rhs error: {e:?}"))?;

                if lhs != rhs {
                    violations.push(BimoduleAxiomViolation::CommutativityFails {
                        left_arrow: alpha.clone(),
                        right_arrow: beta.clone(),
                    });
                }
            }
        }

        Ok(violations)
    }
}

/// A violation of one of the three (A, A)-bimodule axioms.
#[derive(Debug, Clone, PartialEq)]
pub enum BimoduleAxiomViolation<V, E> {
    /// Left multiplication by `algebra.relations()[relation_index]` does not
    /// act as zero on the Peirce component with right index `right_vertex`.
    LeftRelationFails {
        relation_index: usize,
        right_vertex: V,
    },
    /// Right multiplication by `algebra.relations()[relation_index]` does not
    /// act as zero on the Peirce component with left index `left_vertex`.
    RightRelationFails {
        relation_index: usize,
        left_vertex: V,
    },
    /// The left action of `left_arrow` and the right action of `right_arrow`
    /// do not commute on some Peirce component.
    CommutativityFails { left_arrow: E, right_arrow: E },
}

// ── Diagonal bimodule A = kQ/I ────────────────────────────────────────────

/// Error from [`DiagonalBimodule::try_new`].
#[derive(Debug, Clone, PartialEq)]
pub enum DiagonalBimoduleError {
    /// The algebra has non-monomial relations; path enumeration requires a
    /// monomial ideal.
    NonMonomialRelations(NonMonomialIdeal),
    /// Admissible-path enumeration exceeded the caller-supplied bound.
    TooManyPaths { max_paths: usize },
}

/// A = kQ/I viewed as an (A, A)-bimodule over itself (the *diagonal bimodule*).
///
/// The Peirce piece `e_v A e_w` is spanned by all admissible paths from v to w
/// (with the idempotent `e_v`, represented as the empty path, included when v = w).
///
/// The left and right action maps are precomputed as 0/1 matrices.
///
/// - `L_{α,w}`: column j picks out the path obtained by prepending α to the j-th
///   basis element of `e_{t(α)} A e_w` (1 if non-zero in A, 0 otherwise).
/// - `R_{β,v}`: column j picks out the path obtained by appending β to the j-th
///   basis element of `e_v A e_{s(β)}`.
pub struct DiagonalBimodule<VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
    Coeffs: Ring,
{
    algebra: Arc<QuiverWithRelations<VertexLabel, EdgeLabel, Coeffs>>,
    left_matrices: HashMap<(EdgeLabel, VertexLabel), DMatrix<bool>>,
    right_matrices: HashMap<(EdgeLabel, VertexLabel), DMatrix<bool>>,
    #[allow(dead_code)]
    basis_idx: HashMap<(VertexLabel, VertexLabel), HashMap<Vec<EdgeLabel>, usize>>,
}

impl<V, E, Coeffs> DiagonalBimodule<V, E, Coeffs>
where
    V: std::hash::Hash + Eq + Clone + Debug,
    E: std::hash::Hash + Eq + Clone + Debug,
    Coeffs: crate::checked_arith::Field,
{
    /// Construct the diagonal bimodule of a monomial algebra.
    ///
    /// `max_paths` bounds the total number of admissible non-idempotent paths.
    ///
    /// # Errors
    ///
    /// - [`DiagonalBimoduleError::NonMonomialRelations`] if the algebra has
    ///   non-monomial relations (path enumeration requires a monomial ideal).
    /// - [`DiagonalBimoduleError::TooManyPaths`] if enumeration exceeds
    ///   `max_paths`.
    pub fn try_new(
        algebra: Arc<QuiverWithRelations<V, E, Coeffs>>,
        max_paths: usize,
    ) -> Result<Self, DiagonalBimoduleError> {
        let mon: QuiverWithMonomialRelations<V, E> = (&*algebra)
            .try_into()
            .map_err(DiagonalBimoduleError::NonMonomialRelations)?;

        let (left_matrices, right_matrices, basis_idx) = build_action_matrices(&mon, max_paths)?;

        Ok(Self {
            algebra,
            left_matrices,
            right_matrices,
            basis_idx,
        })
    }
}

impl<V, E, Coeffs> QuiverBimodule<V, E, Coeffs, DMatrix<bool>> for DiagonalBimodule<V, E, Coeffs>
where
    V: std::hash::Hash + Eq + Clone,
    E: std::hash::Hash + Eq + Clone,
    Coeffs: Ring,
{
    fn algebra(&self) -> &Arc<QuiverWithRelations<V, E, Coeffs>> {
        &self.algebra
    }

    fn left_action(&self, alpha: &E, w: &V) -> Option<DMatrix<bool>> {
        self.left_matrices.get(&(alpha.clone(), w.clone())).cloned()
    }

    fn right_action(&self, beta: &E, v: &V) -> Option<DMatrix<bool>> {
        self.right_matrices.get(&(beta.clone(), v.clone())).cloned()
    }
}

// ── Internal path-basis enumeration and matrix construction ───────────────

#[allow(clippy::type_complexity)]
fn build_action_matrices<V, E>(
    algebra: &QuiverWithMonomialRelations<V, E>,
    max_paths: usize,
) -> Result<
    (
        HashMap<(E, V), DMatrix<bool>>,
        HashMap<(E, V), DMatrix<bool>>,
        HashMap<(V, V), HashMap<Vec<E>, usize>>,
    ),
    DiagonalBimoduleError,
>
where
    V: std::hash::Hash + Eq + Clone + Debug,
    E: std::hash::Hash + Eq + Clone + Debug,
{
    use std::collections::HashSet;

    let all_vertices: Vec<V> = algebra.vertices().collect();
    let all_edges: Vec<E> = algebra.edge_labels().collect();

    // ── Build admissible path basis grouped by (source, target) ──────────
    //
    // An empty Vec<E> represents the idempotent e_v (present only on the diagonal).
    let mut basis_by_pair: HashMap<(V, V), Vec<Vec<E>>> = HashMap::new();
    let mut basis_idx: HashMap<(V, V), HashMap<Vec<E>, usize>> = HashMap::new();

    for v in &all_vertices {
        let pair = (v.clone(), v.clone());
        basis_by_pair.entry(pair.clone()).or_default().push(vec![]);
        basis_idx.entry(pair).or_default().insert(vec![], 0);
    }

    let mut seen: HashSet<Vec<E>> = HashSet::new();
    let mut frontier: Vec<Vec<E>> = Vec::new();
    let mut total_paths = 0usize;

    for e in &all_edges {
        let word = vec![e.clone()];
        if !algebra.is_zero_path_word(&word) && seen.insert(word.clone()) {
            if let Some((src, tgt)) = algebra.path_source_target(&word) {
                let pair = (src, tgt);
                let idx_map = basis_idx.entry(pair.clone()).or_default();
                let paths = basis_by_pair.entry(pair).or_default();
                idx_map.insert(word.clone(), paths.len());
                paths.push(word.clone());
                total_paths += 1;
            }
            frontier.push(word);
        }
    }

    while let Some(path) = frontier.pop() {
        if total_paths > max_paths {
            return Err(DiagonalBimoduleError::TooManyPaths { max_paths });
        }
        for next_e in &all_edges {
            let mut new_path = path.clone();
            new_path.push(next_e.clone());
            if seen.contains(&new_path) {
                continue;
            }
            if !algebra.is_zero_path_word(&new_path) {
                seen.insert(new_path.clone());
                if let Some((src, tgt)) = algebra.path_source_target(&new_path) {
                    let pair = (src, tgt);
                    let idx_map = basis_idx.entry(pair.clone()).or_default();
                    let paths = basis_by_pair.entry(pair).or_default();
                    idx_map.insert(new_path.clone(), paths.len());
                    paths.push(new_path.clone());
                    total_paths += 1;
                }
                frontier.push(new_path);
            }
        }
    }

    // ── Build action matrices ─────────────────────────────────────────────

    let empty_basis: Vec<Vec<E>> = Vec::new();
    let mut left_matrices: HashMap<(E, V), DMatrix<bool>> = HashMap::new();
    let mut right_matrices: HashMap<(E, V), DMatrix<bool>> = HashMap::new();

    for alpha in &all_edges {
        let Some((src_a, tgt_a)) = algebra.edge_endpoint_labels(alpha) else {
            continue;
        };
        for w in &all_vertices {
            // L_{α,w}: e_{tgt_a} A e_w → e_{src_a} A e_w
            // domain  = paths from tgt_a to w
            // codomain = paths from src_a to w
            let dom_basis = basis_by_pair
                .get(&(tgt_a.clone(), w.clone()))
                .unwrap_or(&empty_basis);
            let cod_pair = (src_a.clone(), w.clone());
            let cod_idx = basis_idx.get(&cod_pair);
            let nrows = basis_by_pair.get(&cod_pair).map_or(0, Vec::len);
            let mut mat = DMatrix::<bool>::from_element(nrows, dom_basis.len(), false);

            for (j, q) in dom_basis.iter().enumerate() {
                // Prepend α: [α, q₁, …, qₖ]
                let mut new_path = vec![alpha.clone()];
                new_path.extend_from_slice(q);
                if let Some(i) = cod_idx.and_then(|m| m.get(&new_path)).copied() {
                    mat[(i, j)] = true;
                }
            }
            left_matrices.insert((alpha.clone(), w.clone()), mat);
        }
    }

    for beta in &all_edges {
        let Some((src_b, tgt_b)) = algebra.edge_endpoint_labels(beta) else {
            continue;
        };
        for v in &all_vertices {
            // R_{β,v}: e_v A e_{src_b} → e_v A e_{tgt_b}
            let dom_basis = basis_by_pair
                .get(&(v.clone(), src_b.clone()))
                .unwrap_or(&empty_basis);
            let cod_pair = (v.clone(), tgt_b.clone());
            let cod_idx = basis_idx.get(&cod_pair);
            let nrows = basis_by_pair.get(&cod_pair).map_or(0, Vec::len);
            let mut mat = DMatrix::from_element(nrows, dom_basis.len(), false);

            for (j, q) in dom_basis.iter().enumerate() {
                // Append β: [q₁, …, qₖ, β]
                let mut new_path = q.clone();
                new_path.push(beta.clone());
                if let Some(i) = cod_idx.and_then(|m| m.get(&new_path)).copied() {
                    mat[(i, j)] = true;
                }
            }
            right_matrices.insert((beta.clone(), v.clone()), mat);
        }
    }

    Ok((left_matrices, right_matrices, basis_idx))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::quiver::{BasisElt, PathAlgebra, Quiver, tests::make_kronecker_quiver};

    fn is_zero_mat<T>(m: &DMatrix<T>, is_zero_entry: fn(&T) -> bool) -> bool {
        m.iter().all(|x| is_zero_entry(x))
    }

    fn promotion_bool(m: DMatrix<bool>) -> DMatrix<f64> {
        let mut to_return = DMatrix::zeros(m.nrows(), m.ncols());
        for idx in 0..m.nrows() {
            for jdx in 0..m.ncols() {
                if m[(idx, jdx)] {
                    to_return[(idx, jdx)] = 1.0;
                }
            }
        }
        to_return
    }

    fn make_a3_with_rel() -> Arc<QuiverWithRelations<&'static str, &'static str, f64>> {
        // 0 --"a"--> 1 --"b"--> 2,  relation ab = 0
        let mut q = Quiver::new();
        q.add_edge("0", "1", "a");
        q.add_edge("1", "2", "b");
        let q = Arc::new(q);
        let rel = PathAlgebra::singleton(
            q.clone(),
            BasisElt::Path(nonempty::nonempty!["a", "b"]),
            1.0_f64,
        );
        Arc::new(QuiverWithRelations::new(
            q,
            vec![rel],
            Some(|x: &f64| *x == 0.0),
        ))
    }

    // ── Matrix dimensions reflect the Peirce decomposition ───────────────

    #[test]
    fn a2_left_action_shape() {
        let q = crate::quiver::tests::make_a2_quiver();
        let qwr = Arc::new(QuiverWithRelations::<_, _, f64>::from_quiver_no_relations(
            Arc::new(q),
        ));
        let bim = DiagonalBimodule::try_new(qwr, 100).unwrap();

        // L_{a,"1"}: e_{1} A e_{1} → e_{0} A e_{1}
        // e_{1} A e_{1} = {e₁}  (dim 1),  e_{0} A e_{1} = {a}  (dim 1)
        let mat = bim.left_action(&"a", &"beta").unwrap();
        assert_eq!((mat.nrows(), mat.ncols()), (1, 1));
        assert_eq!(mat[(0, 0)], true);

        // L_{a,"0"}: e_{1} A e_{0} → e_{0} A e_{0}
        // e_{1} A e_{0} = {}  (dim 0)
        let mat = bim.left_action(&"a", &"alpha").unwrap();
        assert_eq!((mat.nrows(), mat.ncols()), (1, 0));
    }

    #[test]
    fn a2_right_action_shape() {
        let q = crate::quiver::tests::make_a2_quiver();
        let qwr = Arc::new(QuiverWithRelations::<_, _, f64>::from_quiver_no_relations(
            Arc::new(q),
        ));
        let bim = DiagonalBimodule::try_new(qwr, 100).unwrap();

        // R_{a,"0"}: e_{0} A e_{0} → e_{0} A e_{1}
        // e_{0} A e_{0} = {e₀}  (dim 1),  e_{0} A e_{1} = {a}  (dim 1)
        let mat = bim.right_action(&"a", &"alpha").unwrap();
        assert_eq!((mat.nrows(), mat.ncols()), (1, 1));
        assert_eq!(mat[(0, 0)], true);

        // R_{a,"1"}: e_{1} A e_{0} → e_{1} A e_{1},  domain empty
        let mat = bim.right_action(&"a", &"beta").unwrap();
        assert_eq!((mat.nrows(), mat.ncols()), (1, 0));
    }

    // ── Axiom checks ─────────────────────────────────────────────────────

    #[test]
    fn a2_diagonal_satisfies_axioms() {
        let q = crate::quiver::tests::make_a2_quiver();
        let qwr = Arc::new(QuiverWithRelations::<_, _, f64>::from_quiver_no_relations(
            Arc::new(q),
        ));
        let bim = DiagonalBimodule::try_new(qwr.clone(), 100).unwrap();
        let v = bim
            .check_bimodule_axioms(promotion_bool, |mat| is_zero_mat(mat, |x| *x == 0.0))
            .unwrap();
        assert!(v.is_empty(), "{v:?}");
    }

    #[test]
    fn kronecker_diagonal_satisfies_axioms() {
        let qwr = QuiverWithRelations::<_, _, f64>::from_quiver_no_relations(Arc::new(
            make_kronecker_quiver(),
        ));
        let qwr = Arc::new(qwr);
        let bim = DiagonalBimodule::try_new(qwr, 100).unwrap();
        let v = bim
            .check_bimodule_axioms(promotion_bool, |m| is_zero_mat(&m, |x| *x == 0.0))
            .unwrap();
        assert!(v.is_empty(), "{v:?}");
    }

    #[test]
    fn a3_with_relation_satisfies_axioms() {
        let qwr = make_a3_with_rel();
        let bim = DiagonalBimodule::try_new(qwr, 100).unwrap();
        let v = bim
            .check_bimodule_axioms(promotion_bool, |m| is_zero_mat(&m, |x| *x == 0.0))
            .unwrap();
        assert!(v.is_empty(), "{v:?}");
    }

    // ── Peirce piece dimensions ───────────────────────────────────────────

    #[test]
    fn a3_with_rel_zero_peirce_piece() {
        // In A₃/(ab), e₀ A e₂ = 0 because the only path 0→2 is "ab" = 0.
        // L_{a,"2"}: e_{1} A e_{2} → e_{0} A e_{2} should have 0 rows.
        let qwr = make_a3_with_rel();
        let bim = DiagonalBimodule::try_new(qwr, 100).unwrap();
        let mat = bim.left_action(&"a", &"2").unwrap();
        assert_eq!(mat.nrows(), 0, "e_0 A e_2 should be zero");
    }

    #[test]
    fn kronecker_peirce_piece_e0_a_e1_has_dimension_two() {
        // In the Kronecker algebra, e₀ A e₁ = span{a, b}  (dim 2).
        // R_{a,"0"}: e_{0} A e_{0} → e_{0} A e_{1}  → should be 2×1.
        let qwr = QuiverWithRelations::<_, _, f64>::from_quiver_no_relations(Arc::new(
            make_kronecker_quiver(),
        ));
        let qwr = Arc::new(qwr);
        let bim = DiagonalBimodule::try_new(qwr, 100).unwrap();
        let mat = bim.right_action(&"a", &"alpha").unwrap();
        assert_eq!((mat.nrows(), mat.ncols()), (2, 1));
    }
}
