use std::{collections::HashMap, fmt::Debug, sync::Arc};

use nalgebra::DMatrix;

use crate::arithmetic_utils::{Field,Ring};
use crate::quiver_algebra::{
    quiver::BasisElt,
    quiver_with_mon_rels::{NonMonomialIdeal, QuiverWithMonomialRelations},
    quiver_with_rels::QuiverWithRelations,
};

// ── Element type ─────────────────────────────────────────────────────────────

/// An element of the Peirce piece `e_left M e_right`, expressed as coordinates
/// in the ordered basis returned by [`QuiverBimodule::peirce_basis`].
///
/// `M` is a bimodule over `kQ^{op}` so the multiplications are all happening
/// with these opposite conventions.
///
/// The coordinates are indexed so that `coords[i]` is the coefficient of the
/// `i`-th basis element as returned by `peirce_basis(left, right)`.
#[derive(Clone, Debug, PartialEq)]
pub struct PeirceElement<V, Coeffs> {
    pub left: V,
    pub right: V,
    pub coords: Vec<Coeffs>,
}

impl<V: Clone, Coeffs: Field> PeirceElement<V, Coeffs> {
    /// Zero element of `e_left M e_right`.
    pub fn zero(left: V, right: V, dim: usize) -> Self {
        Self {
            left,
            right,
            coords: vec![Coeffs::zero(); dim],
        }
    }

    /// The `i`-th standard basis vector in `e_left M e_right`.
    pub fn basis_vec(left: V, right: V, dim: usize, i: usize) -> Self {
        let mut coords = vec![Coeffs::zero(); dim];
        coords[i] = Coeffs::one();
        Self {
            left,
            right,
            coords,
        }
    }

    /// True iff all coordinates are zero.
    pub fn is_zero(&self) -> bool {
        self.coords.iter().all(Coeffs::is_zero)
    }

    /// `self += other` coordinate-wise (panics if dimensions differ).
    pub fn add_assign(&mut self, other: &Self) {
        debug_assert_eq!(self.coords.len(), other.coords.len());
        for (s, o) in self.coords.iter_mut().zip(&other.coords) {
            *s += o.clone();
        }
    }

    /// Scale every coordinate by `c`.
    pub fn scale_assign(&mut self, c: &Coeffs) {
        for coord in &mut self.coords {
            *coord *= c.clone();
        }
    }
}

// ── Trait ─────────────────────────────────────────────────────────────────────

/// An (A, A)-bimodule over A = kQ^{op}/I^{op}, accessed through its Peirce decomposition.
///
/// **Path convention**: `BasisElt::Path([a₁, a₂, …, aₙ])` means "follow a₁ first, then
/// a₂, …, then aₙ" (left-to-right concatenation).  This makes the algebra kQ^{op}: arrow
/// α: s(α) → t(α) in Q lies in `e_{s(α)} kQ^{op} e_{t(α)}`. The multiplication and
/// elements all being interpreted with opposite multiplications.
/// For paths `p ∈ e_v kQ^{op} e_w` and `q ∈ e_w kQ^{op} e_s`, the product
/// `p·q ∈ e_v kQ^{op} e_s`.
///
/// **Indexing convention**: [`peirce_basis`](QuiverBimodule::peirce_basis)`(v, w)` and
/// [`PeirceElement`] use the convention where `left` = source in Q and `right`
/// = target in Q.  This is the swap of the `kQ` vs `kQ^{opp}` Peirce indices.
///
/// Implementors must supply:
/// - the ordered basis of each Peirce piece via [`peirce_basis`],
/// - the arrow-level actions via [`left_act`] / [`right_act`].
///
/// **Actions** (described using the internal `.left`/`.right` fields):
///
/// - [`left_act`](QuiverBimodule::left_act)`(α, elt)` requires `elt.left = t(α)` and
///   returns an element with `left = s(α)`, `right` unchanged.
///
/// - [`right_act`](QuiverBimodule::right_act)`(β, elt)` requires `elt.right = s(β)` and
///   returns an element with `right = t(β)`, `left` unchanged.
///
/// The generic parameter `BasisLabel` is the type used to label individual
/// basis elements of each Peirce piece (e.g., admissible paths for the
/// diagonal bimodule).
pub trait QuiverBimodule<V, E, Coeffs, BasisLabel, const OP_ALG: bool>
where
    V: std::hash::Hash + Eq + Clone,
    E: Eq + std::hash::Hash + Clone,
    Coeffs: Ring,
{
    /// The algebra A = kQ^{op}/I^{op} over which this is a bimodule.
    fn algebra(&self) -> &Arc<QuiverWithRelations<V, E, Coeffs, OP_ALG>>;

    /// The ordered basis of the internal Peirce piece `(v, w)`
    /// If using `kQ` instead of `kQ^{op}` this would be `e_w N e_v`
    /// where `N` is "same" bimodule as this but regarded with `kQ` actions.
    ///     
    /// Empty if the piece is zero.
    ///
    /// The `i`-th entry corresponds to `coords[i]` in any [`PeirceElement`]
    /// with `left == v` and `right == w`.
    ///
    /// If either `v` or `w` are not in the quiver then it gives an empty array.
    /// There is no subspace component there.
    fn peirce_basis(&self, v: &V, w: &V) -> &[BasisLabel];

    /// Dimension of `e_v M e_w`.
    fn peirce_dim(&self, v: &V, w: &V) -> usize {
        self.peirce_basis(v, w).len()
    }

    /// Left action by arrow `α`: requires `elt.left = t(α)` (internal convention).
    ///
    /// Returns an element with `left = s(α)`, `right` unchanged.
    /// If `elt.left ≠ t(α)` the result is zero in the appropriate Peirce piece.
    ///
    /// # Panics
    ///
    /// If `alpha` is not an edge of the quiver or `elt` is not actually an element of the
    /// pruported subspace.
    fn left_act(&self, alpha: &E, elt: &PeirceElement<V, Coeffs>) -> PeirceElement<V, Coeffs>;

    /// Right action by arrow `β`: requires `elt.right = s(β)` (internal convention).
    ///
    /// Returns an element with `right = t(β)`, `left` unchanged.
    /// If `elt.right ≠ s(β)` the result is zero in the appropriate Peirce piece.
    ///
    /// # Panics
    ///
    /// If `beta` is not an edge of the quiver or `elt` is not actually an element of the
    /// pruported subspace.
    fn right_act(&self, beta: &E, elt: &PeirceElement<V, Coeffs>) -> PeirceElement<V, Coeffs>;

    /// Left action of an arbitrary algebra basis element `b` on `elt`.
    /// This transforms from `elt \in e_{t(b))} M e_{w}`
    /// via `b \in e_{s(b)} kQ^{op} e_{t(b)}` to give
    /// the result in `e_{s(b))} M e_{w}`
    /// using the left action of `kQ^{op}` that descended to the quoteint
    /// `kQ^{op}/I^{op}`.
    ///
    /// - `BasisElt::Idempotent(v)`: returns `elt` if `elt.left == v`, else zero
    ///   (internal piece `(v, elt.right)`, usual convention piece `e_{elt.right} M e_v`).
    /// - `BasisElt::Path([a₁,…,aₙ])`: path from `s(a₁)` to `t(aₙ)` in Q, lying in
    ///   usual convention `e_{t(aₙ)} kQ e_{s(a₁)}` vs `e_{s(a₁)} kQ^{op} e_{t(aₙ)}`.
    ///   Applies `left_act` starting from `aₙ`
    ///   (last arrow) down to `a₁`.  The result has `left = s(a₁)`.
    fn left_act_algebra_elt(
        &self,
        b: &BasisElt<V, E>,
        elt: &PeirceElement<V, Coeffs>,
    ) -> PeirceElement<V, Coeffs>
    where
        Coeffs: Field,
    {
        let expected_right = elt.right.clone();
        let (expected_left, _expected_middle) = self
            .algebra()
            .quiver()
            .basis_endpoints(b)
            .expect("This is a path in the quiver");

        let to_return = match b {
            BasisElt::Idempotent(v) => {
                if &elt.left == v {
                    elt.clone()
                } else {
                    PeirceElement::zero(
                        v.clone(),
                        elt.right.clone(),
                        self.peirce_dim(v, &elt.right),
                    )
                }
            }
            BasisElt::Path(word) => {
                let mut m = elt.clone();
                for arrow in word.iter().rev() {
                    m = self.left_act(arrow, &m);
                }
                m
            }
        };
        debug_assert!(to_return.left == expected_left);
        debug_assert!(to_return.right == expected_right);
        to_return
    }

    /// Right action of an arbitrary algebra basis element `b` on `elt`.
    /// This transforms from `elt \in e_w M e_{s(b))}`
    /// via `b \in e_{s(b)} kQ^{op} e_{t(b)}` to give
    /// the result in `e_w M e_{t(b))}`
    /// using the right action of `kQ^{op}` that descended to the quoteint
    /// `kQ^{op}/I^{op}`.
    ///
    /// - `BasisElt::Idempotent(v)`: returns `elt` if `elt.right == v`, else zero
    /// - `BasisElt::Path([a₁,…,aₙ])`: path from `s(a₁)` to `t(aₙ)` in Q, lying in
    ///   usual convention `e_{t(aₙ)} kQ e_{s(a₁)}` vs `e_{s(a₁)} kQ^{op} e_{t(aₙ)}`.
    ///   Applies `right_act` starting from `a₁`
    ///   (first arrow) up to `aₙ`, appending one arrow at a time.  The result has
    ///   `right = t(aₙ)`.
    fn right_act_algebra_elt(
        &self,
        b: &BasisElt<V, E>,
        elt: &PeirceElement<V, Coeffs>,
    ) -> PeirceElement<V, Coeffs>
    where
        Coeffs: Field,
    {
        let expected_left = elt.left.clone();
        let (_expected_middle, expected_right) = self
            .algebra()
            .quiver()
            .basis_endpoints(b)
            .expect("This is a path in the quiver");

        let to_return = match b {
            BasisElt::Idempotent(v) => {
                if &elt.right == v {
                    elt.clone()
                } else {
                    PeirceElement::zero(elt.left.clone(), v.clone(), self.peirce_dim(&elt.left, v))
                }
            }
            BasisElt::Path(word) => {
                let mut m = elt.clone();
                for arrow in word.iter() {
                    m = self.right_act(arrow, &m);
                }
                m
            }
        };
        debug_assert!(to_return.left == expected_left);
        debug_assert!(to_return.right == expected_right);
        to_return
    }

    /// Check the three (A, A)-bimodule axioms.
    ///
    /// Returns the list of violations (empty iff all axioms hold).
    ///
    /// ## Axioms checked
    ///
    /// 1. **Left relations**: for each relation ρ ∈ I and each right-index
    ///    vertex w, the left action of ρ is zero on every element of
    ///    `e_{t(ρ)} M e_w`.
    ///
    /// 2. **Right relations**: for each relation ρ and each left-index vertex
    ///    v, the right action of ρ is zero on every element of
    ///    `e_v M e_{s(ρ)}`.
    ///
    /// 3. **Commutativity**: for every pair of arrows α: i→j and β: p→q and
    ///    every basis element `b` of `e_j M e_p`,
    ///    `α · (b · β) = (α · b) · β`.
    fn check_bimodule_axioms(&self) -> Vec<BimoduleAxiomViolation<V, E>>
    where
        V: Debug,
        E: Debug,
        Coeffs: Field,
        BasisLabel: Clone,
    {
        let mut violations = Vec::new();
        let algebra = self.algebra();
        let quiver = algebra.quiver();

        // ── Axiom 1: left relations ───────────────────────────────────────────
        for (rel_idx, rel) in algebra.relations().enumerate() {
            let Some((src_rel, tgt_rel)) = rel.all_parallel()
                .expect("The relations are well formed. Combining r1 + r2 with different endpoints, actually means r1 and r2 should be in the generators of the ideal separately") else {
                continue;
            };
            for w in quiver.vertex_labels() {
                let dim_in = self.peirce_dim(&tgt_rel, w);
                let dim_out = self.peirce_dim(&src_rel, w);
                'basis: for i in 0..dim_in {
                    let ei = PeirceElement::basis_vec(tgt_rel.clone(), w.clone(), dim_in, i);
                    let mut sum = PeirceElement::zero(src_rel.clone(), w.clone(), dim_out);
                    for (basis_elt, coeff) in rel.iter() {
                        if coeff.is_zero() {
                            continue;
                        }
                        let mut term = self.left_act_algebra_elt(basis_elt, &ei);
                        term.scale_assign(coeff);
                        sum.add_assign(&term);
                    }
                    if !sum.is_zero() {
                        violations.push(BimoduleAxiomViolation::LeftRelationFails {
                            relation_index: rel_idx,
                            right_vertex: w.clone(),
                        });
                        break 'basis;
                    }
                }
            }
        }

        // ── Axiom 2: right relations ──────────────────────────────────────────
        for (rel_idx, rel) in algebra.relations().enumerate() {
            let Some((src_rel, tgt_rel)) = rel.all_parallel()
                .expect("The relations are well formed. Combining r1 + r2 with different endpoints, actually means r1 and r2 should be in the generators of the ideal separately") else {
                continue;
            };
            for v in quiver.vertex_labels() {
                let dim_in = self.peirce_dim(v, &src_rel);
                let dim_out = self.peirce_dim(v, &tgt_rel);
                'basis: for i in 0..dim_in {
                    let ei = PeirceElement::basis_vec(v.clone(), src_rel.clone(), dim_in, i);
                    let mut sum = PeirceElement::zero(v.clone(), tgt_rel.clone(), dim_out);
                    for (basis_elt, coeff) in rel.iter() {
                        if coeff.is_zero() {
                            continue;
                        }
                        let mut term = self.right_act_algebra_elt(basis_elt, &ei);
                        term.scale_assign(coeff);
                        sum.add_assign(&term);
                    }
                    if !sum.is_zero() {
                        violations.push(BimoduleAxiomViolation::RightRelationFails {
                            relation_index: rel_idx,
                            left_vertex: v.clone(),
                        });
                        break 'basis;
                    }
                }
            }
        }

        // ── Axiom 3: commutativity ────────────────────────────────────────────
        // For α: i→j and β: p→q, check α·(b·β) = (α·b)·β for every
        // basis element b of e_j M e_p.
        for alpha in quiver.edge_labels() {
            let Some((_src_a, tgt_a)) = quiver.edge_endpoint_labels(alpha) else {
                continue;
            };
            for beta in quiver.edge_labels() {
                let Some((src_b, _tgt_b)) = quiver.edge_endpoint_labels(beta) else {
                    continue;
                };
                let domain_basis_len = self.peirce_dim(&tgt_a, &src_b);
                'basis: for i in 0..domain_basis_len {
                    let b =
                        PeirceElement::basis_vec(tgt_a.clone(), src_b.clone(), domain_basis_len, i);
                    // LHS: α · (b · β)
                    let lhs = self.left_act(alpha, &self.right_act(beta, &b));
                    // RHS: (α · b) · β
                    let rhs = self.right_act(beta, &self.left_act(alpha, &b));
                    if lhs != rhs {
                        violations.push(BimoduleAxiomViolation::CommutativityFails {
                            left_arrow: alpha.clone(),
                            right_arrow: beta.clone(),
                        });
                        break 'basis;
                    }
                }
            }
        }

        violations
    }
}

// ── Axiom violation type ──────────────────────────────────────────────────────

/// A violation of one of the three `(kQ^{op}/I^{op}, kQ^{op}/I^{op})`-bimodule axioms.
#[derive(Debug, Clone, PartialEq)]
pub enum BimoduleAxiomViolation<V, E> {
    /// Left multiplication by `algebra.relations()[relation_index]` does not
    /// act as zero on the Peirce component with internal right index `right_vertex`.
    /// This means the left action on the piece `e_t M e_{right_vertex}` is nonzero,
    /// so the action does not descend to the quotient by this relation.
    LeftRelationFails {
        relation_index: usize,
        right_vertex: V,
    },
    /// Right multiplication by `algebra.relations()[relation_index]` does not
    /// act as zero on the Peirce component with internal left index `left_vertex`.
    /// This means the right action on the piece `e_{left_vertex} M e_s` is nonzero,
    /// so the action does not descend to the quotient by this relation.
    RightRelationFails {
        relation_index: usize,
        left_vertex: V,
    },
    /// The left action of `left_arrow` and the right action of `right_arrow`
    /// do not commute on some basis element of the relevant Peirce piece.
    CommutativityFails { left_arrow: E, right_arrow: E },
}

// ── Diagonal bimodule A = kQ^{op}/I^{op} ─────────────────────────────────────

/// Error from [`DiagonalBimodule::try_new`].
#[derive(Debug, Clone, PartialEq)]
pub enum DiagonalBimoduleError {
    /// The algebra has non-monomial relations; path enumeration requires a
    /// monomial ideal.
    NonMonomialRelations(NonMonomialIdeal),
    /// Admissible-path enumeration exceeded the caller-supplied bound.
    TooManyPaths { max_paths: usize },
}

/// `A = kQ^{op}/I^{op}` viewed as an `(A, A)`-bimodule over itself (the *diagonal bimodule*).
///
/// The piece `e_v kQ^{op} e_w` is spanned by all admissible paths from `v`
/// to `w` in Q, with the idempotent `e_v` (empty path) included when `v = w`.
/// Internally this piece is accessed as `peirce_basis(v, w)` (source first, target second).
///
/// Basis labels are `Vec<E>` (the edge-label sequence of the path in Q; empty = idempotent).
/// [`peirce_basis`](QuiverBimodule::peirce_basis) returns these sequences in the order
/// they were enumerated (BFS).
pub struct DiagonalBimodule<V, E, Coeffs, const OP_ALG: bool>
where
    V: std::hash::Hash + Eq + Clone,
    E: Eq + std::hash::Hash + Clone,
    Coeffs: Ring,
{
    algebra: Arc<QuiverWithRelations<V, E, Coeffs, OP_ALG>>,
    /// Ordered basis of each Peirce piece: `basis_by_pair[(v, w)] = [path₀, path₁, …]`.
    basis_by_pair: HashMap<(V, V), Vec<Vec<E>>>,
    /// Left action matrices: `left_matrices[(α, w)]` is `L_{α,w}`,
    /// mapping internal piece `(t(α), w)` → `(s(α), w)`.
    /// Size: `dim(s(α), w) × dim(t(α), w)`.
    left_matrices: HashMap<(E, V), DMatrix<bool>>,
    /// Right action matrices: `right_matrices[(β, v)]` is `R_{β,v}`,
    /// mapping internal piece `(v, s(β))` → `(v, t(β))`.
    /// Size: `dim(v, t(β)) × dim(v, s(β))`.
    right_matrices: HashMap<(E, V), DMatrix<bool>>,
}

impl<V, E, Coeffs, const OP_ALG: bool> DiagonalBimodule<V, E, Coeffs, OP_ALG>
where
    V: std::hash::Hash + Eq + Clone + Debug,
    E: std::hash::Hash + Eq + Clone + Debug,
    Coeffs: Field,
{
    /// Construct the diagonal bimodule of a monomial algebra.
    ///
    /// `max_paths` bounds the total number of admissible non-idempotent paths.
    ///
    /// # Errors
    ///
    /// - [`DiagonalBimoduleError::NonMonomialRelations`] if the algebra has
    ///   non-monomial relations.
    /// - [`DiagonalBimoduleError::TooManyPaths`] if enumeration exceeds
    ///   `max_paths`.
    pub fn try_new(
        algebra: Arc<QuiverWithRelations<V, E, Coeffs, OP_ALG>>,
        max_paths: usize,
    ) -> Result<Self, DiagonalBimoduleError> {
        let mon: QuiverWithMonomialRelations<V, E> = (&*algebra)
            .try_into()
            .map_err(DiagonalBimoduleError::NonMonomialRelations)?;

        let (basis_by_pair, left_matrices, right_matrices) =
            build_action_matrices(&mon, max_paths)?;

        Ok(Self {
            algebra,
            basis_by_pair,
            left_matrices,
            right_matrices,
        })
    }
}

impl<V, E, Coeffs, const OP_ALG: bool> QuiverBimodule<V, E, Coeffs, Vec<E>, OP_ALG>
    for DiagonalBimodule<V, E, Coeffs, OP_ALG>
where
    V: std::hash::Hash + Eq + Clone,
    E: std::hash::Hash + Eq + Clone,
    Coeffs: Field,
{
    fn algebra(&self) -> &Arc<QuiverWithRelations<V, E, Coeffs, OP_ALG>> {
        &self.algebra
    }

    fn peirce_basis(&self, v: &V, w: &V) -> &[Vec<E>] {
        self.basis_by_pair
            .get(&(v.clone(), w.clone()))
            .map_or(&[], Vec::as_slice)
    }

    fn left_act(&self, alpha: &E, elt: &PeirceElement<V, Coeffs>) -> PeirceElement<V, Coeffs> {
        let Some((src_a, tgt_a)) = self.algebra.quiver().edge_endpoint_labels(alpha) else {
            return PeirceElement::zero(elt.left.clone(), elt.right.clone(), 0);
        };
        let out_dim = self.peirce_dim(&src_a, &elt.right);
        if elt.left != tgt_a {
            return PeirceElement::zero(src_a, elt.right.clone(), out_dim);
        }
        let mat = &self.left_matrices[&(alpha.clone(), elt.right.clone())];
        let mut out = vec![Coeffs::zero(); out_dim];
        for j in 0..elt.coords.len() {
            if elt.coords[j].is_zero() {
                continue;
            }
            for i in 0..out_dim {
                if mat[(i, j)] {
                    out[i] += elt.coords[j].clone();
                }
            }
        }
        PeirceElement {
            left: src_a,
            right: elt.right.clone(),
            coords: out,
        }
    }

    fn right_act(&self, beta: &E, elt: &PeirceElement<V, Coeffs>) -> PeirceElement<V, Coeffs> {
        let Some((src_b, tgt_b)) = self.algebra.quiver().edge_endpoint_labels(beta) else {
            return PeirceElement::zero(elt.left.clone(), elt.right.clone(), 0);
        };
        let out_dim = self.peirce_dim(&elt.left, &tgt_b);
        if elt.right != src_b {
            return PeirceElement::zero(elt.left.clone(), tgt_b, out_dim);
        }
        let mat = &self.right_matrices[&(beta.clone(), elt.left.clone())];
        let mut out = vec![Coeffs::zero(); out_dim];
        for j in 0..elt.coords.len() {
            if elt.coords[j].is_zero() {
                continue;
            }
            for i in 0..out_dim {
                if mat[(i, j)] {
                    out[i] += elt.coords[j].clone();
                }
            }
        }
        PeirceElement {
            left: elt.left.clone(),
            right: tgt_b,
            coords: out,
        }
    }
}

// ── Internal path-basis enumeration and matrix construction ───────────────────

#[allow(clippy::type_complexity)]
fn build_action_matrices<V, E>(
    algebra: &QuiverWithMonomialRelations<V, E>,
    max_paths: usize,
) -> Result<
    (
        HashMap<(V, V), Vec<Vec<E>>>,
        HashMap<(E, V), DMatrix<bool>>,
        HashMap<(E, V), DMatrix<bool>>,
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

    // ── Build admissible path basis grouped by (source, target) ──────────────
    //
    // Empty Vec<E> represents the idempotent e_v (present only on the diagonal).
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

    // ── Build action matrices ─────────────────────────────────────────────────

    let empty_basis: Vec<Vec<E>> = Vec::new();
    let mut left_matrices: HashMap<(E, V), DMatrix<bool>> = HashMap::new();
    let mut right_matrices: HashMap<(E, V), DMatrix<bool>> = HashMap::new();

    for alpha in &all_edges {
        let Some((src_a, tgt_a)) = algebra.edge_endpoint_labels(alpha) else {
            continue;
        };
        for w in &all_vertices {
            // L_{α,w}: `kQ^{op}` perspective (tgt_a, w) → (src_a, w)  [`kQ` perspective: e_w A e_{tgt_a} → e_w A e_{src_a}]
            let dom_basis = basis_by_pair
                .get(&(tgt_a.clone(), w.clone()))
                .unwrap_or(&empty_basis);
            let cod_pair = (src_a.clone(), w.clone());
            let cod_idx = basis_idx.get(&cod_pair);
            let nrows = basis_by_pair.get(&cod_pair).map_or(0, Vec::len);
            let mut mat = DMatrix::<bool>::from_element(nrows, dom_basis.len(), false);
            for (j, q) in dom_basis.iter().enumerate() {
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
            // R_{β,v}: `kQ^{op}` perspective (v, src_b) → (v, tgt_b)  [`kQ` perspective: e_{src_b} A e_v → e_{tgt_b} A e_v]
            let dom_basis = basis_by_pair
                .get(&(v.clone(), src_b.clone()))
                .unwrap_or(&empty_basis);
            let cod_pair = (v.clone(), tgt_b.clone());
            let cod_idx = basis_idx.get(&cod_pair);
            let nrows = basis_by_pair.get(&cod_pair).map_or(0, Vec::len);
            let mut mat = DMatrix::from_element(nrows, dom_basis.len(), false);
            for (j, q) in dom_basis.iter().enumerate() {
                let mut new_path = q.clone();
                new_path.push(beta.clone());
                if let Some(i) = cod_idx.and_then(|m| m.get(&new_path)).copied() {
                    mat[(i, j)] = true;
                }
            }
            right_matrices.insert((beta.clone(), v.clone()), mat);
        }
    }

    Ok((basis_by_pair, left_matrices, right_matrices))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::quiver_algebra::path_algebra::PathAlgebra;
    use crate::quiver_algebra::quiver::{BasisElt, Quiver, tests::make_kronecker_quiver};

    fn make_a3_with_rel() -> Arc<QuiverWithRelations<&'static str, &'static str, f64, true>> {
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

    // ── peirce_basis reflects the Peirce decomposition ────────────────────────

    #[test]
    fn a2_peirce_basis_dimensions() {
        let q = crate::quiver_algebra::quiver::tests::make_a2_quiver();
        let qwr =
            Arc::new(QuiverWithRelations::<_, _, f64, true>::from_quiver_no_relations(Arc::new(q)));
        let bim = DiagonalBimodule::try_new(qwr, 100).unwrap();

        // e_alpha kQ^op e_alpha = {e_alpha}  (dim 1)
        assert_eq!(bim.peirce_basis(&"alpha", &"alpha").len(), 1);
        // e_beta kQ^op e_alpha = {a}  (dim 1)  [a: alpha→beta, so a ∈ e_{t(a)} A e_{s(a)} = e_beta A e_alpha]
        assert_eq!(bim.peirce_basis(&"alpha", &"beta").len(), 1);
        // e_alpha kQ^op e_beta = {}  (dim 0)
        assert_eq!(bim.peirce_basis(&"beta", &"alpha").len(), 0);
        // e_beta kQ^op e_beta = {e_beta}  (dim 1)
        assert_eq!(bim.peirce_basis(&"beta", &"beta").len(), 1);
    }

    #[test]
    fn a2_peirce_basis_labels() {
        let q = crate::quiver_algebra::quiver::tests::make_a2_quiver();
        let qwr =
            Arc::new(QuiverWithRelations::<_, _, f64, true>::from_quiver_no_relations(Arc::new(q)));
        let bim = DiagonalBimodule::try_new(qwr, 100).unwrap();

        // e_alpha kQ^op e_alpha basis = [[] (idempotent)]
        assert_eq!(bim.peirce_basis(&"alpha", &"alpha"), &[vec![] as Vec<&str>]);
        // e_beta kQ^op e_alpha basis = [["a"]]
        assert_eq!(bim.peirce_basis(&"alpha", &"beta"), &[vec!["a"]]);
    }

    // ── Element-level actions ─────────────────────────────────────────────────

    #[test]
    fn a2_left_act_on_basis_element() {
        let q = crate::quiver_algebra::quiver::tests::make_a2_quiver();
        let qwr =
            Arc::new(QuiverWithRelations::<_, _, f64, true>::from_quiver_no_relations(Arc::new(q)));
        let bim = DiagonalBimodule::try_new(qwr, 100).unwrap();

        // e_beta is the unique basis element of e_beta kQ^op e_beta.
        // L_a acts: a · e_beta = a (the unique basis element of e_beta kQ^op e_alpha).
        let e_beta = PeirceElement::basis_vec("beta", "beta", 1, 0);
        let result = bim.left_act(&"a", &e_beta);
        assert_eq!(result.left, "alpha");
        assert_eq!(result.right, "beta");
        assert_eq!(result.coords, vec![1.0]);
    }

    #[test]
    fn a2_right_act_on_basis_element() {
        let q = crate::quiver_algebra::quiver::tests::make_a2_quiver();
        let qwr =
            Arc::new(QuiverWithRelations::<_, _, f64, true>::from_quiver_no_relations(Arc::new(q)));
        let bim = DiagonalBimodule::try_new(qwr, 100).unwrap();

        // e_alpha is the unique basis element of e_alpha kQ^op e_alpha.
        // R_a acts: e_alpha · a = a (the unique basis element of e_beta kQ^op e_alpha).
        let e_alpha = PeirceElement::basis_vec("alpha", "alpha", 1, 0);
        let result = bim.right_act(&"a", &e_alpha);
        assert_eq!(result.left, "alpha");
        assert_eq!(result.right, "beta");
        assert_eq!(result.coords, vec![1.0]);
    }

    #[test]
    fn left_act_wrong_peirce_index_returns_zero() {
        // L_a expects elt.left == t(a) = "beta".
        // Passing elt.left == "alpha" should return zero.
        let q = crate::quiver_algebra::quiver::tests::make_a2_quiver();
        let qwr =
            Arc::new(QuiverWithRelations::<_, _, f64, true>::from_quiver_no_relations(Arc::new(q)));
        let bim = DiagonalBimodule::try_new(qwr, 100).unwrap();
        let wrong = PeirceElement::basis_vec("alpha", "beta", 1, 0);
        let result = bim.left_act(&"a", &wrong);
        assert!(result.is_zero());
    }

    // ── Axiom checks ──────────────────────────────────────────────────────────

    #[test]
    fn a2_diagonal_satisfies_axioms() {
        let q = crate::quiver_algebra::quiver::tests::make_a2_quiver();
        let qwr =
            Arc::new(QuiverWithRelations::<_, _, f64, true>::from_quiver_no_relations(Arc::new(q)));
        let bim = DiagonalBimodule::try_new(qwr, 100).unwrap();
        let v = bim.check_bimodule_axioms();
        assert!(v.is_empty(), "{v:?}");
    }

    #[test]
    fn kronecker_diagonal_satisfies_axioms() {
        let qwr = Arc::new(
            QuiverWithRelations::<_, _, f64, true>::from_quiver_no_relations(Arc::new(
                make_kronecker_quiver(),
            )),
        );
        let bim = DiagonalBimodule::try_new(qwr, 100).unwrap();
        let v = bim.check_bimodule_axioms();
        assert!(v.is_empty(), "{v:?}");
    }

    #[test]
    fn a3_with_relation_satisfies_axioms() {
        let qwr = make_a3_with_rel();
        let bim = DiagonalBimodule::try_new(qwr, 100).unwrap();
        let v = bim.check_bimodule_axioms();
        assert!(v.is_empty(), "{v:?}");
    }

    // ── Peirce piece zero-ness ────────────────────────────────────────────────

    #[test]
    fn a3_with_rel_zero_peirce_piece() {
        // In A₃/(ab), e_2 kQ^op e_0 = 0 because path "ab" (0→1→2) is zero.
        let qwr = make_a3_with_rel();
        let bim = DiagonalBimodule::try_new(qwr, 100).unwrap();
        assert_eq!(bim.peirce_dim(&"0", &"2"), 0);
    }

    #[test]
    fn kronecker_peirce_piece_dimensions() {
        let qwr = Arc::new(
            QuiverWithRelations::<_, _, f64, true>::from_quiver_no_relations(Arc::new(
                make_kronecker_quiver(),
            )),
        );
        let bim = DiagonalBimodule::try_new(qwr, 100).unwrap();
        assert_eq!(bim.peirce_dim(&"alpha", &"beta"), 2);
        assert_eq!(bim.peirce_dim(&"alpha", &"alpha"), 1);
        assert_eq!(bim.peirce_dim(&"beta", &"beta"), 1);
        assert_eq!(bim.peirce_dim(&"beta", &"alpha"), 0);
    }
}
