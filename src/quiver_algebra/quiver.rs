use nonempty::NonEmpty;
use std::collections::HashMap;

use petgraph::{
    graph::NodeIndex,
    prelude::EdgeIndex,
    stable_graph::StableGraph,
    visit::{EdgeRef, IntoEdgeReferences},
};

/// A directed multigraph with user-supplied vertex and edge labels, backed by
/// `petgraph::StableGraph`. Vertices are created implicitly when an edge is added.
#[derive(Debug, Clone)]
#[must_use]
pub struct Quiver<VertexLabel, EdgeLabel>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
{
    p: StableGraph<VertexLabel, EdgeLabel>,
    v_labels: HashMap<VertexLabel, NodeIndex>,
    e_labels: HashMap<EdgeLabel, EdgeIndex>,
    count_parallel_pairs: u8,
}

impl<VertexLabel, EdgeLabel> Default for Quiver<VertexLabel, EdgeLabel>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
{
    fn default() -> Self {
        Self {
            p: StableGraph::new(),
            v_labels: HashMap::new(),
            e_labels: HashMap::new(),
            count_parallel_pairs: 0,
        }
    }
}

impl<VertexLabel, EdgeLabel> Quiver<VertexLabel, EdgeLabel>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
{
    /// Create an empty quiver with no vertices or arrows.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a vertex with the given label. Returns the internal `NodeIndex`.
    /// Does not check for duplicate labels.
    pub fn add_vertex(&mut self, label: VertexLabel) -> NodeIndex {
        let node_index = self.p.add_node(label.clone());
        self.v_labels.insert(label, node_index);
        node_index
    }

    /// Add an arrow `from → to` with the given label. Vertices are created if absent.
    pub fn add_edge(&mut self, from: VertexLabel, to: VertexLabel, label: EdgeLabel) {
        let from_idx = self
            .v_labels
            .get(&from)
            .copied()
            .unwrap_or_else(|| self.add_vertex(from));
        let to_idx = self
            .v_labels
            .get(&to)
            .copied()
            .unwrap_or_else(|| self.add_vertex(to));
        let cur_edge = self.p.edges_connecting(from_idx, to_idx);
        if cur_edge.count() != 0 {
            self.count_parallel_pairs += 1;
        }
        let idx = self.p.add_edge(from_idx, to_idx, label.clone());
        self.e_labels.insert(label, idx);
    }

    #[must_use = "Use the count of nodes in the quiver"]
    pub fn num_vertices(&self) -> usize {
        self.p.node_count()
    }

    #[must_use = "Use the count of arrows in the quiver"]
    pub fn num_arrows(&self) -> usize {
        self.p.edge_count()
    }

    #[must_use = "This has to go into some property about finite dimensionality of an algebra"]
    pub fn is_acyclic(&self) -> bool {
        !petgraph::algo::is_cyclic_directed(&self.p)
    }

    /// Returns `true` if `vertex` is present in the quiver.
    pub fn contains_vertex(&self, vertex: &VertexLabel) -> bool {
        self.v_labels.contains_key(vertex)
    }

    /// Remove an edge from the quiver by its label. Returns `true` if the edge was present.
    pub fn remove_edge(&mut self, edge: &EdgeLabel) -> bool {
        if let Some(edge_idx) = self.e_labels.remove(edge) {
            self.p.remove_edge(edge_idx);
            true
        } else {
            false
        }
    }

    /// Returns `true` if an arrow with label `edge` is present in the quiver.
    pub fn contains_edge(&self, edge: &EdgeLabel) -> bool {
        self.e_labels.contains_key(edge)
    }

    /// Iterate over the labels of all arrows in the quiver. Order is unspecified.
    pub fn edge_labels(&self) -> impl Iterator<Item = &EdgeLabel> {
        self.e_labels.keys()
    }

    /// Maps the labels of the quiver to new types.
    #[allow(clippy::missing_panics_doc)]
    pub fn map_labels<V2, E2>(
        self,
        fv: impl Fn(VertexLabel) -> V2,
        fe: impl Fn(EdgeLabel) -> E2,
    ) -> Quiver<V2, E2>
    where
        V2: Eq + std::hash::Hash + Clone,
        E2: Eq + std::hash::Hash + Clone,
    {
        let mut out = Quiver::new();
        for v in self.p.node_weights() {
            out.add_vertex(fv(v.clone()));
        }
        for e in self.p.edge_references() {
            let src = fv(self
                .p
                .node_weight(e.source())
                .expect("Source does exist")
                .clone());
            let tgt = fv(self
                .p
                .node_weight(e.target())
                .expect("Target does exist")
                .clone());
            out.add_edge(src, tgt, fe(e.weight().clone()));
        }
        out
    }

    /// Iterate over the labels of all vertices in the quiver. Order is unspecified.
    pub fn vertex_labels(&self) -> impl Iterator<Item = &VertexLabel> {
        self.v_labels.keys()
    }

    pub(crate) fn vertex_idx(&self, v: &VertexLabel) -> Option<&NodeIndex> {
        self.v_labels.get(v)
    }

    pub(crate) fn vertex_label(&self, idx: NodeIndex) -> Option<&VertexLabel> {
        self.p.node_weight(idx)
    }

    pub(crate) fn edge_endpoints(&self, a: &EdgeLabel) -> Option<(NodeIndex, NodeIndex)> {
        self.e_labels.get(a).and_then(|a| self.p.edge_endpoints(*a))
    }

    /// Returns `(source, target)` vertex labels for arrow `a`, or `None` if `a` is not in the quiver.
    #[allow(clippy::missing_panics_doc)]
    pub fn edge_endpoint_labels(&self, a: &EdgeLabel) -> Option<(VertexLabel, VertexLabel)> {
        self.edge_endpoints(a).map(|(idx, jdx)| {
            let idx_label = self
                .p
                .node_weight(idx)
                .expect("This is a vertex of the quiver")
                .clone();
            let jdx_label = self
                .p
                .node_weight(jdx)
                .expect("This is a vertex of the quiver")
                .clone();
            (idx_label, jdx_label)
        })
    }

    fn path_source_target_indices(&self, path: &[EdgeLabel]) -> Option<(NodeIndex, NodeIndex)> {
        let first = path.first()?;
        let last = path.last()?;
        let (src, _) = self.edge_endpoints(first)?;
        let (_, tgt) = self.edge_endpoints(last)?;
        Some((src, tgt))
    }

    pub(crate) fn path_source_target_labels(
        &self,
        path: &[EdgeLabel],
    ) -> Option<(VertexLabel, VertexLabel)> {
        let (src, tgt) = self.path_source_target_indices(path)?;
        let src_label = self.p.node_weight(src)?.clone();
        let tgt_label = self.p.node_weight(tgt)?.clone();
        Some((src_label, tgt_label))
    }

    fn path_source_target_labels_nonempty(
        &self,
        path: &NonEmpty<EdgeLabel>,
    ) -> Option<(VertexLabel, VertexLabel)> {
        let (src, tgt_first) = self.edge_endpoint_labels(path.first())?;
        if path.tail().is_empty() {
            Some((src, tgt_first))
        } else {
            let (_, tgt_tail) = self.path_source_target_labels(path.tail())?;
            Some((src, tgt_tail))
        }
    }

    /// Returns `(source, target)` vertex labels for a basis element (idempotent or path),
    /// or `None` if any constituent arrow is missing from the quiver.
    pub fn basis_endpoints(
        &self,
        basis: &BasisElt<VertexLabel, EdgeLabel>,
    ) -> Option<(VertexLabel, VertexLabel)> {
        match basis {
            BasisElt::Idempotent(v) => Some((v.clone(), v.clone())),
            BasisElt::Path(word) => self.path_source_target_labels_nonempty(word),
        }
    }

    /// Returns `true` if arrow `a` can be followed by arrow `b`, i.e. `t(a) == s(b)`.
    pub fn composable(&self, a: &EdgeLabel, b: &EdgeLabel) -> bool {
        let a_ends = self.edge_endpoints(a);
        let b_ends = self.edge_endpoints(b);
        match (a_ends, b_ends) {
            (None, None | Some(_)) | (Some(_), None) => false,
            (Some((_a_src, a_tgt)), Some((b_src, _b_tgt))) => a_tgt == b_src,
        }
    }

    pub(crate) fn is_composable_path(&self, word: &[EdgeLabel]) -> bool {
        if !word.iter().all(|edge| self.contains_edge(edge)) {
            return false;
        }
        for pair in word.windows(2) {
            if !self.composable(&pair[0], &pair[1]) {
                return false;
            }
        }
        true
    }

    pub(crate) fn is_composable_arrow_word(&self, word: &NonEmpty<EdgeLabel>) -> bool {
        if !word.iter().all(|edge| self.contains_edge(edge)) {
            return false;
        }
        #[allow(clippy::collapsible_if)]
        if let Some(second) = word.tail().first() {
            if !self.composable(word.first(), second) {
                return false;
            }
        }
        for pair in word.tail().windows(2) {
            if !self.composable(&pair[0], &pair[1]) {
                return false;
            }
        }
        true
    }

    /// Multiplication of monomials in `kQ^{op}` or `kQ`
    /// If the product is `0` due to composability reasons
    /// then return `None`
    /// Otherwise the product is another monomial in
    /// the arrows and vertex idempotents.
    ///
    /// # Panics
    ///
    /// We expect the provided arrows and vertices in `l`
    /// and `r` to actually be in the quiver.
    pub fn multiply_basis<const OP_ALG: bool>(
        &self,
        l: &BasisElt<VertexLabel, EdgeLabel>,
        r: &BasisElt<VertexLabel, EdgeLabel>,
    ) -> Option<BasisElt<VertexLabel, EdgeLabel>> {
        if OP_ALG {
            self.multiply_basis_op(l, r)
        } else {
            self.multiply_basis_reg(l, r)
        }
    }

    fn multiply_basis_op(
        &self,
        l: &BasisElt<VertexLabel, EdgeLabel>,
        r: &BasisElt<VertexLabel, EdgeLabel>,
    ) -> Option<BasisElt<VertexLabel, EdgeLabel>> {
        match (l, r) {
            (BasisElt::Path(k1), BasisElt::Path(k2)) => {
                if self.composable(k1.last(), k2.first()) {
                    let mut new_k = k1.clone();
                    new_k.extend(k2.clone());
                    Some(BasisElt::Path(new_k))
                } else {
                    None
                }
            }
            (BasisElt::Path(k1), BasisElt::Idempotent(w)) => {
                let (_, should_be_w) = self
                    .edge_endpoints(k1.last())
                    .expect("This is an arrow of the quiver");
                let does_compose = *self
                    .v_labels
                    .get(w)
                    .expect("This is a vertex of the quiver")
                    == should_be_w;
                if does_compose {
                    Some(BasisElt::Path(k1.clone()))
                } else {
                    None
                }
            }
            (BasisElt::Idempotent(w), BasisElt::Path(k2)) => {
                let (should_be_w, _) = self
                    .edge_endpoints(k2.first())
                    .expect("This is an arrow of the quiver");
                let does_compose = *self
                    .v_labels
                    .get(w)
                    .expect("This is a vertex of the quiver")
                    == should_be_w;
                if does_compose {
                    Some(BasisElt::Path(k2.clone()))
                } else {
                    None
                }
            }
            (BasisElt::Idempotent(v), BasisElt::Idempotent(w)) => {
                if v == w {
                    Some(BasisElt::Idempotent(v.clone()))
                } else {
                    None
                }
            }
        }
    }

    fn multiply_basis_reg(
        &self,
        l: &BasisElt<VertexLabel, EdgeLabel>,
        r: &BasisElt<VertexLabel, EdgeLabel>,
    ) -> Option<BasisElt<VertexLabel, EdgeLabel>> {
        self.multiply_basis_op(r, l)
    }

    /// Iterate over outgoing arrows from `vertex` as `(edge_label, target_label)` pairs.
    #[allow(clippy::missing_panics_doc)]
    pub fn successors(
        &self,
        vertex: &VertexLabel,
    ) -> impl Iterator<Item = (EdgeLabel, VertexLabel)> {
        let v_idx = self
            .v_labels
            .get(vertex)
            .expect("This is a vertex of the quiver");
        self.p
            .edges_directed(*v_idx, petgraph::Direction::Outgoing)
            .map(|eref| {
                let e_label = eref.weight().clone();
                let v_label = self
                    .p
                    .node_weight(self.p.edge_endpoints(eref.id()).expect("This is an edge").1)
                    .expect("This is a vertex of the quiver")
                    .clone();
                (e_label, v_label)
            })
    }

    /// Iterate over incoming arrows to `vertex` as `(edge_label, source_label)` pairs.
    #[allow(clippy::missing_panics_doc)]
    pub fn predecessors(
        &self,
        vertex: &VertexLabel,
    ) -> impl Iterator<Item = (EdgeLabel, VertexLabel)> {
        let v_idx = self
            .v_labels
            .get(vertex)
            .expect("This is a vertex of the quiver");
        self.p
            .edges_directed(*v_idx, petgraph::Direction::Incoming)
            .map(|eref| {
                let e_label = eref.weight().clone();
                let v_label = self
                    .p
                    .node_weight(self.p.edge_endpoints(eref.id()).expect("This is an edge").0)
                    .expect("This is a vertex of the quiver")
                    .clone();
                (e_label, v_label)
            })
    }

    /// Return the doubled quiver Q̄ together with the adjoint pairs `(a, a*)`.
    ///
    /// For each arrow `a: s → t` in `self`, adds a reverse arrow `a* = dagger(a): t → s`.
    /// Returns `(Q̄, [(a, a*), …])`.
    #[allow(clippy::missing_panics_doc)]
    pub fn double(
        mut self,
        dagger: impl Fn(&EdgeLabel) -> EdgeLabel,
    ) -> (Self, Vec<(EdgeLabel, EdgeLabel)>) {
        let mut new_edges = vec![];
        let mut adjoint_pairs = Vec::with_capacity(self.edge_labels().count());
        for edge in self.edge_labels() {
            let (src, tgt) = self
                .edge_endpoint_labels(edge)
                .expect("This is an edge of the quiver");
            let dagger_edge = dagger(edge);
            new_edges.push((tgt, src, dagger_edge.clone()));
            adjoint_pairs.push((edge.clone(), dagger_edge));
        }
        for (new_edge_src, new_edge_tgt, new_edge_label) in new_edges {
            self.add_edge(new_edge_src, new_edge_tgt, new_edge_label);
        }
        (self, adjoint_pairs)
    }

    /// The heart construction on a quiver.
    ///
    /// This takes a quiver `Q` and produces a new quiver `Q^heart`
    /// which has new adjoint arrows for every arrow of `Q`
    /// and new framing arrows from every vertex of `Q` from a new vertex.
    /// If `framings_daggered` is `true` then the framing arrows also get adjoints going the other way.
    /// 
    /// So for example
    /// 
    /// If `Q` is
    /// 
    /// ```
    /// alpha -> beta -> gamma
    /// ```
    /// 
    /// then `Q^heart` (with `framings_daggered` = false) is
    /// 
    /// ```
    ///  alpha <-> beta <-> gamma
    ///   ^         ^         ^
    ///   |         |         |
    ///  alpha*    beta*    gamma*
    /// ```
    /// 
    /// When `framings_daggered` = true, `Q^heart` also has
    /// arrows `alpha -> alpha*`, `beta -> beta*`, and `gamma -> gamma*`.
    #[allow(clippy::type_complexity)]
    pub fn heartify(
        self,
        dagger: impl Fn(&EdgeLabel) -> EdgeLabel + Clone,
        framing_creation: impl Fn(&VertexLabel) -> (EdgeLabel, VertexLabel),
        framings_daggered: bool,
    ) -> (
        Self,
        Vec<(EdgeLabel, EdgeLabel)>,
        Vec<(EdgeLabel, VertexLabel, Option<EdgeLabel>)>,
    ) {
        let vertices = self.vertex_labels().cloned().collect::<Vec<_>>();
        let (mut new_self, adjoint_pairs) = self.double(dagger.clone());
        let mut new_framing_arrows = Vec::with_capacity(vertices.len());
        for vertex in vertices {
            let (new_edge_label, new_vertex) = framing_creation(&vertex);
            new_self.add_edge(new_vertex.clone(), vertex.clone(), new_edge_label.clone());
            if framings_daggered {
                let dagger_edge = dagger(&new_edge_label);
                new_self.add_edge(vertex, new_vertex.clone(), dagger_edge.clone());
                new_framing_arrows.push((new_edge_label, new_vertex, Some(dagger_edge)));
            } else {
                new_framing_arrows.push((new_edge_label, new_vertex, None));
            }
        }
        (new_self, adjoint_pairs, new_framing_arrows)
    }

    /// The Ginzburg construction on a quiver.
    /// This can be thought of as like `heartify` with `framings_daggered = false`
    /// but with all the framing vertices being
    /// identified with the original vertices, so that the framing arrows become self-loops.
    pub(crate) fn ginzburgify(
        self,
        dagger: impl Fn(&EdgeLabel) -> EdgeLabel,
        self_loop: impl Fn(&VertexLabel) -> EdgeLabel,
    ) -> (Self, Vec<(EdgeLabel, EdgeLabel)>, Vec<EdgeLabel>) {
        let vertex_labels: Vec<_> = self.vertex_labels().cloned().collect();
        let (mut new_self, adjoint_pairs) = self.double(dagger);
        let mut new_self_loops = Vec::new();
        for v in vertex_labels {
            let loop_label = self_loop(&v);
            new_self_loops.push(loop_label.clone());
            new_self.add_edge(v.clone(), v, loop_label);
        }
        (new_self, adjoint_pairs, new_self_loops)
    }
}

/// A monomial basis element of a path algebra: either a vertex idempotent or a non-empty path.
///
/// Arrows are stored **left-to-right**: `Path([a₁, …, aₙ])` means follow `a₁` first.
/// See the `OP_ALG` discussion on [`PathAlgebra`](crate::quiver_algebra::PathAlgebra) for how
/// this interacts with the two multiplication conventions.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum BasisElt<VertexLabel, EdgeLabel> {
    /// The vertex idempotent `e_v`.
    Idempotent(VertexLabel),
    /// A non-empty composable sequence of arrows.
    Path(NonEmpty<EdgeLabel>),
}

impl<VertexLabel, EdgeLabel> BasisElt<VertexLabel, EdgeLabel> {
    /// Construct a `Path` from a slice, returning `None` for an empty slice.
    pub fn create(v: &[EdgeLabel]) -> Option<Self>
    where
        EdgeLabel: Clone,
    {
        Some(Self::Path(NonEmpty::from_slice(v)?))
    }

    /// Map vertex and edge labels to new types, preserving the variant.
    pub fn map_labels<V2, E2>(
        self,
        fv: impl Fn(VertexLabel) -> V2,
        fe: impl Fn(EdgeLabel) -> E2,
    ) -> BasisElt<V2, E2>
    where
        V2: Eq + std::hash::Hash + Clone,
        E2: Eq + std::hash::Hash + Clone,
    {
        match self {
            BasisElt::Idempotent(v) => BasisElt::Idempotent(fv(v)),
            BasisElt::Path(word) => {
                let new_nonempty = NonEmpty::map(word, fe);
                BasisElt::Path(new_nonempty)
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::fmt::Debug;

    use nonempty::NonEmpty;
    use proptest::{prelude::Strategy, prop_assert, prop_oneof, proptest};

    use super::{BasisElt, Quiver};

    pub(crate) fn make_kronecker_quiver() -> Quiver<&'static str, &'static str> {
        let mut kronecker_quiver = Quiver::new();
        kronecker_quiver.add_vertex("alpha");
        kronecker_quiver.add_vertex("beta");
        kronecker_quiver.add_edge("alpha", "beta", "a");
        kronecker_quiver.add_edge("alpha", "beta", "b");
        kronecker_quiver
    }

    pub(crate) fn make_a2_quiver() -> Quiver<&'static str, &'static str> {
        let mut kronecker_quiver = Quiver::new();
        kronecker_quiver.add_vertex("alpha");
        kronecker_quiver.add_vertex("beta");
        kronecker_quiver.add_edge("alpha", "beta", "a");
        kronecker_quiver
    }

    pub(crate) fn make_ginzburg_quiver() -> (
        Quiver<&'static str, String>,
        Vec<(String, String)>,
        Vec<String>,
    ) {
        let mut ginzburg_quiver = Quiver::new();
        ginzburg_quiver.add_vertex("0");
        ginzburg_quiver.add_edge("0", "0", "A".to_string());
        ginzburg_quiver.ginzburgify(|a| format!("{}Dagger", a), |v| format!("Omega{}", v))
    }

    fn arbitrary_vertex<'a, VertexLabel, EdgeLabel>(
        q: &Quiver<VertexLabel, EdgeLabel>,
    ) -> impl Strategy<Value = VertexLabel> + 'a
    where
        VertexLabel: std::hash::Hash + Eq + Clone + Debug + 'static,
        EdgeLabel: Eq + std::hash::Hash + Clone + Debug + 'a,
    {
        let all_vs = q.vertex_labels().cloned().collect::<Vec<_>>();
        proptest::sample::select(all_vs)
    }

    fn arbitrary_edge<'a, VertexLabel, EdgeLabel>(
        q: &Quiver<VertexLabel, EdgeLabel>,
    ) -> impl Strategy<Value = EdgeLabel> + 'a
    where
        VertexLabel: std::hash::Hash + Eq + Clone + Debug + 'a,
        EdgeLabel: Eq + std::hash::Hash + Clone + Debug + 'static,
    {
        let all_vs = q.edge_labels().cloned().collect::<Vec<_>>();
        proptest::sample::select(all_vs)
    }

    fn composable_pair<'a, VertexLabel, EdgeLabel>(
        q: Quiver<VertexLabel, EdgeLabel>,
        element: impl Strategy<Value = BasisElt<VertexLabel, EdgeLabel>> + Clone + 'a,
    ) -> impl Strategy<Value = BasisElt<VertexLabel, EdgeLabel>> + 'a
    where
        VertexLabel: std::hash::Hash + Eq + Clone + Debug + 'static,
        EdgeLabel: Eq + std::hash::Hash + Clone + Debug + 'static,
    {
        (element.clone(), element)
            .prop_filter_map("composable", move |(z, w)| q.multiply_basis_op(&z, &w))
    }

    pub(crate) fn arbitrary_basis_element<'a, VertexLabel, EdgeLabel>(
        q: Quiver<VertexLabel, EdgeLabel>,
    ) -> impl Strategy<Value = BasisElt<VertexLabel, EdgeLabel>> + 'a
    where
        VertexLabel: std::hash::Hash + Eq + Clone + Debug + 'static,
        EdgeLabel: Eq + std::hash::Hash + Clone + Debug + 'static,
    {
        let small = prop_oneof![
            arbitrary_vertex(&q).prop_map(BasisElt::Idempotent),
            arbitrary_edge(&q).prop_map(|z: EdgeLabel| BasisElt::Path(NonEmpty::singleton(z)))
        ];
        small.prop_recursive(10, 10, 2, move |element| {
            prop_oneof![composable_pair(q.clone(), element.clone()), element]
        })
    }

    pub(crate) fn testing_arbitrary_quiver() -> Quiver<String, String> {
        let mut w = make_a2_quiver().map_labels(|z| z.to_string(), |z| z.to_string());
        (w, _) = w.double(|e| format!("{e}*"));
        (w, _) = w.double(|e| format!("{e}*"));
        w
    }

    proptest! {
        #[test]
        fn basis_strat(q in arbitrary_basis_element(testing_arbitrary_quiver())) {
            let a2 = testing_arbitrary_quiver();
            match q {
                BasisElt::Idempotent(v) => {
                    prop_assert!(a2.contains_vertex(&v));
                }
                BasisElt::Path(non_empty) => {
                    prop_assert!(a2.is_composable_arrow_word(&non_empty));
                }
            }
        }
    }
}
