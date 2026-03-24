use nonempty::NonEmpty;
use std::{
    collections::HashMap,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    sync::Arc,
};

use petgraph::{graph::NodeIndex, prelude::EdgeIndex, stable_graph::StableGraph, visit::EdgeRef};

use crate::quiver_algebra::checked_arith::Ring;

#[derive(Debug)]
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_vertex(&mut self, label: VertexLabel) -> NodeIndex {
        let node_index = self.p.add_node(label.clone());
        self.v_labels.insert(label, node_index);
        node_index
    }

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

    #[must_use = "This has to go into some property about finite dimensionality of an algebra"]
    pub fn is_acyclic(&self) -> bool {
        !petgraph::algo::is_cyclic_directed(&self.p)
    }

    pub fn contains_vertex(&self, vertex: &VertexLabel) -> bool {
        self.v_labels.contains_key(vertex)
    }

    pub fn contains_edge(&self, edge: &EdgeLabel) -> bool {
        self.e_labels.contains_key(edge)
    }

    pub fn edge_labels(&self) -> impl Iterator<Item = &EdgeLabel> {
        self.e_labels.keys()
    }

    pub fn vertex_labels(&self) -> impl Iterator<Item = &VertexLabel> {
        self.v_labels.keys()
    }

    fn edge_endpoints(&self, a: &EdgeLabel) -> Option<(NodeIndex, NodeIndex)> {
        self.e_labels.get(a).and_then(|a| self.p.edge_endpoints(*a))
    }

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

    pub fn basis_endpoints(
        &self,
        basis: &BasisElt<VertexLabel, EdgeLabel>,
    ) -> Option<(VertexLabel, VertexLabel)> {
        match basis {
            BasisElt::Idempotent(v) => Some((v.clone(), v.clone())),
            BasisElt::Path(word) => self.path_source_target_labels_nonempty(word),
        }
    }

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

    /// Multiplication of monomials in `kQ^{op}`
    /// If the product is `0` due to composability reasons
    /// then return `None`
    /// Otherwise the product is another monomial in
    /// the arrows and vertex idempotents.
    ///
    /// # Panics
    ///
    /// We expect the provided arrows and vertices in `l`
    /// and `r` to actually be in the quiver.
    pub fn multiply_basis(
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
    /// and new framing arrows from every vertex of `Q` to a new vertex.
    /// If `framings_daggered` is `true` then the framing arrows also get adjoints going back the other way.
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
            new_self.add_edge(vertex.clone(), new_vertex.clone(), new_edge_label.clone());
            if framings_daggered {
                let dagger_edge = dagger(&new_edge_label);
                new_self.add_edge(new_vertex.clone(), vertex, dagger_edge.clone());
                new_framing_arrows.push((new_edge_label, new_vertex, Some(dagger_edge)));
            } else {
                new_framing_arrows.push((new_edge_label, new_vertex, None));
            }
        }
        (new_self, adjoint_pairs, new_framing_arrows)
    }

    fn ginzburgify(
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

    pub fn ginzburgify_and_cubic<Coeffs: Ring>(
        self,
        dagger: impl Fn(&EdgeLabel) -> EdgeLabel,
        self_loop: impl Fn(&VertexLabel) -> EdgeLabel,
        one_coeffs: &Coeffs,
    ) -> (Arc<Self>, PathAlgebra<VertexLabel, EdgeLabel, Coeffs>) {
        let (new_self, adjoint_pairs, new_self_loops) = self.ginzburgify(dagger, self_loop);
        let new_self_arc = Arc::new(new_self);
        (
            new_self_arc.clone(),
            PathAlgebra::create_ginzburg_cubic(
                new_self_arc,
                adjoint_pairs,
                new_self_loops,
                one_coeffs,
            ),
        )
    }
}

#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum BasisElt<VertexLabel, EdgeLabel> {
    Idempotent(VertexLabel),
    Path(NonEmpty<EdgeLabel>),
}

impl<VertexLabel, EdgeLabel> BasisElt<VertexLabel, EdgeLabel> {
    pub fn create(v: &[EdgeLabel]) -> Option<Self>
    where
        EdgeLabel: Clone,
    {
        Some(Self::Path(NonEmpty::from_slice(v)?))
    }
}

#[must_use]
#[derive(Clone, Debug)]
pub struct PathAlgebra<VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
    linear_combination_paths: HashMap<BasisElt<VertexLabel, EdgeLabel>, Coeffs>,
}

impl<VertexLabel, EdgeLabel, Coeffs> PathAlgebra<VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    ///
    ///
    /// # Panics
    ///
    /// Every summand in the `kQ^{op}` must be
    /// either a `e_v` for some `v` in the quiver
    /// or `a_1 ... a_n` a composable nonempty sequence of arrows
    pub fn new(
        quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
        mut linear_combination_paths: HashMap<BasisElt<VertexLabel, EdgeLabel>, Coeffs>,
    ) -> Self {
        let len_before = linear_combination_paths.len();
        linear_combination_paths.retain(|path, _| match path {
            BasisElt::Path(path) => quiver.is_composable_arrow_word(path),
            BasisElt::Idempotent(v) => quiver.contains_vertex(v),
        });
        let len_after = linear_combination_paths.len();
        assert_eq!(len_before, len_after);
        Self {
            quiver,
            linear_combination_paths,
        }
    }

    pub fn singleton(
        quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
        linear_combination_paths: BasisElt<VertexLabel, EdgeLabel>,
        coeff: Coeffs,
    ) -> Self {
        let mut linear = HashMap::with_capacity(1);
        linear.insert(linear_combination_paths, coeff);
        Self::new(quiver, linear)
    }

    /// This is a helper for creating the Ginzburg cubic potential.
    /// It is not intended to be a general purpose function for creating elements of the path algebra.
    ///
    /// You are given `arrows_and_daggers` which are `(x,x_dagger)` pairs as `x` goes
    /// through the arrows of the original quiver and `x_dagger` is the corresponding arrow in the opposite direction.
    /// You are also given `self_loops` which are the newly inserted self-loops at each vertex of the original quiver.
    /// This insertion of extra dagger arrows and extra self loops
    /// changes the quiver from `Q` to `Q''` and the path algebra from `kQ^{op}` to `kQ''^{op}`.
    ///
    /// From this `W = \sum_{x in arrows Q} omega_{tgt(x)} x x_dagger - omega_{src(x)} x_dagger x`
    /// is constructed and returned as an element of the path algebra `kQ''^{op}'`.
    #[allow(clippy::missing_panics_doc)]
    fn create_ginzburg_cubic(
        quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
        arrows_and_daggers: Vec<(EdgeLabel, EdgeLabel)>,
        self_loops: Vec<EdgeLabel>,
        one: &Coeffs,
    ) -> Self {
        let mut places_to_self_loops = HashMap::with_capacity(self_loops.len());
        for self_loop in self_loops {
            let (src, _) = quiver
                .edge_endpoint_labels(&self_loop)
                .expect("This is an edge of the quiver");
            places_to_self_loops.insert(
                src,
                Self::singleton(
                    quiver.clone(),
                    BasisElt::Path(nonempty::nonempty![self_loop]),
                    one.clone(),
                ),
            );
        }
        let mut ginzburg_cubic = Self::zero(quiver);
        for (a, a_dagger) in arrows_and_daggers {
            let (a_src, a_tgt) = ginzburg_cubic
                .quiver()
                .edge_endpoint_labels(&a)
                .expect("This is an edge of the quiver");
            let a_part = Self::singleton(
                ginzburg_cubic.quiver().clone(),
                BasisElt::Path(nonempty::nonempty![a]),
                one.clone(),
            );
            let (a_dagger_src, a_dagger_tgt) = ginzburg_cubic
                .quiver()
                .edge_endpoint_labels(&a_dagger)
                .expect("This is an edge of the quiver");
            debug_assert!(a_dagger_src == a_tgt);
            debug_assert!(a_dagger_tgt == a_src);
            let a_dagger_part = Self::singleton(
                ginzburg_cubic.quiver().clone(),
                BasisElt::Path(nonempty::nonempty![a_dagger]),
                one.clone(),
            );
            let loop_at_a_tgt = places_to_self_loops
                .get(&a_tgt)
                .cloned()
                .unwrap_or_else(|| {
                    panic!("No self-loop found for vertex: a's target");
                });
            let loop_at_a_dagger_tgt = places_to_self_loops
                .get(&a_dagger_tgt)
                .cloned()
                .unwrap_or_else(|| {
                    panic!("No self-loop found for vertex: a_dagger's target");
                });
            ginzburg_cubic += loop_at_a_tgt * a_part.clone() * a_dagger_part.clone()
                - loop_at_a_dagger_tgt * a_dagger_part.clone() * a_part.clone();
        }
        ginzburg_cubic
    }

    pub fn zero(quiver: Arc<Quiver<VertexLabel, EdgeLabel>>) -> Self {
        Self::new(quiver, HashMap::new())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&BasisElt<VertexLabel, EdgeLabel>, &Coeffs)> {
        self.linear_combination_paths.iter()
    }

    pub fn simplify(&mut self, mut is_zero: impl FnMut(&Coeffs) -> bool) {
        self.linear_combination_paths
            .retain(|_, coeff| !is_zero(coeff));
    }

    #[allow(clippy::must_use_candidate)]
    pub fn quiver(&self) -> &Arc<Quiver<VertexLabel, EdgeLabel>> {
        &self.quiver
    }

    #[allow(clippy::must_use_candidate)]
    pub fn is_monomial(&self) -> bool {
        self.linear_combination_paths.len() == 1
    }

    #[allow(clippy::must_use_candidate)]
    pub fn is_homogeneous_of_degree(&self, degree: usize) -> bool {
        self.linear_combination_paths.keys().all(|path| match path {
            BasisElt::Path(path) => path.len() == degree,
            BasisElt::Idempotent(_) => degree == 0,
        })
    }

    #[allow(clippy::must_use_candidate)]
    pub fn is_filtered_degree(&self, degree: usize) -> bool {
        self.linear_combination_paths.keys().all(|path| match path {
            BasisElt::Path(path) => path.len() <= degree,
            BasisElt::Idempotent(_) => true,
        })
    }

    #[must_use = "Anything that is definitely zero should be filtered out in a sum"]
    pub fn might_be_nonzero(&self) -> bool {
        !self.linear_combination_paths.is_empty()
    }

    #[must_use = "What to do about elements of the path algebra where every summand is a cycle"]
    #[allow(clippy::missing_panics_doc)]
    pub fn is_cyclic(&self) -> bool {
        for path in self.linear_combination_paths.keys() {
            match path {
                BasisElt::Path(path) => {
                    let first_edge = path.first();
                    let last_edge = path.last();
                    if !self.quiver.composable(last_edge, first_edge) {
                        return false;
                    }
                }
                BasisElt::Idempotent(_) => {
                    return false;
                }
            }
        }
        true
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn split_cyclic(mut self) -> (Self, Self) {
        let quiver_arc = self.quiver.clone();
        let path_keys: Vec<_> = self.linear_combination_paths.keys().cloned().collect();
        let mut acyclic_part = Self::new(quiver_arc, HashMap::new());
        for path in path_keys {
            if let BasisElt::Path(path_specific) = path {
                let first_edge = path_specific.first();
                let last_edge = path_specific.last();
                #[allow(clippy::collapsible_if)]
                if !self.quiver.composable(last_edge, first_edge) {
                    if let Some((path, coeff)) = self
                        .linear_combination_paths
                        .remove_entry(&BasisElt::Path(path_specific))
                    {
                        acyclic_part.linear_combination_paths.insert(path, coeff);
                    } else {
                        debug_assert!(false, "This path was definitely in the linear combination");
                    }
                }
            } else if let Some((path, coeff)) = self.linear_combination_paths.remove_entry(&path) {
                acyclic_part.linear_combination_paths.insert(path, coeff);
            } else {
                debug_assert!(false, "This path was definitely in the linear combination");
            }
        }
        (self, acyclic_part)
    }

    ///
    ///
    /// # Panics
    ///
    /// TODO
    pub fn cyclic_derivative(&mut self, wrt_edge: &EdgeLabel) {
        let wrt_edge_endpoints = self
            .quiver()
            .edge_endpoint_labels(wrt_edge)
            .expect("This is an edge of the quiver");
        let mut new_linear_combination =
            HashMap::with_capacity(self.linear_combination_paths.len());
        for (k, v) in self.linear_combination_paths.drain().filter_map(|(k, v)| {
            if let BasisElt::Path(p) = k {
                Some((p, v))
            } else {
                None
            }
        }) {
            let mut positions_done = vec![];
            while let Some(idx) = k
                .iter()
                .enumerate()
                .rposition(|(idx, p)| p == wrt_edge && !positions_done.contains(&idx))
            {
                let mut k_now: Vec<_> = k.iter().cloned().collect();
                positions_done.push(idx);
                if idx + 1 < k.len() {
                    k_now.rotate_left(idx + 1);
                }
                let z = k_now.pop();
                debug_assert!(z.is_some_and(|z| z == *wrt_edge));
                if k_now.is_empty() {
                    debug_assert!(
                        wrt_edge_endpoints.0 == wrt_edge_endpoints.1,
                        "There is a summand which is a single edge in a cyclic word so that edge should have been a self loop."
                    );
                    new_linear_combination.insert(
                        BasisElt::Idempotent(wrt_edge_endpoints.0.clone()),
                        v.clone(),
                    );
                } else {
                    new_linear_combination.insert(
                        BasisElt::Path(
                            NonEmpty::from_vec(k_now).expect("Checked that it is nonempty"),
                        ),
                        v.clone(),
                    );
                }
            }
        }
        self.linear_combination_paths = new_linear_combination;
    }

    #[allow(clippy::missing_panics_doc, clippy::result_unit_err)]
    /// Assuming all the summands of this element of `kQ`
    /// have the same endpoints then return those endpoints
    /// in `Ok(Some(_,_))`
    /// If the element is the `0` of the algebra, there
    /// are no summands and so it is all parallel vacuously and
    /// we get `Ok(None)`.
    /// This means this element of the algebra is in a
    /// specific `e_i kQ e_j` summand (or all of them for `0`)
    ///
    /// # Errors
    ///
    /// If there was a pair of summands that occured in different
    /// `e_i kQ e_j` and `e_l kQ e_m` that were different in
    /// the direct sum decomposition.
    pub fn all_parallel(&self) -> Result<Option<(VertexLabel, VertexLabel)>, ()> {
        let mut expected_src_tgt = None;
        for path in self.linear_combination_paths.keys() {
            match path {
                BasisElt::Path(path) => {
                    let first_edge = path.first();
                    let last_edge = path.last();
                    let (src_now, _) = self
                        .quiver
                        .edge_endpoints(first_edge)
                        .expect("Already know that it is an arrow in the quiver");
                    let (_, tgt_now) = self
                        .quiver
                        .edge_endpoints(last_edge)
                        .expect("Already know that it is an arrow in the quiver");
                    if let Some((exp_src, exp_tgt)) = expected_src_tgt {
                        if exp_src != src_now || exp_tgt != tgt_now {
                            return Err(());
                        }
                    } else {
                        expected_src_tgt = Some((src_now, tgt_now));
                    }
                }
                BasisElt::Idempotent(just_vertex) => {
                    let src_now = *self
                        .quiver
                        .v_labels
                        .get(just_vertex)
                        .expect("This is a vertex of the quiver");
                    let tgt_now = src_now;
                    if let Some((exp_src, exp_tgt)) = expected_src_tgt {
                        if exp_src != src_now || exp_tgt != tgt_now {
                            return Err(());
                        }
                    } else {
                        expected_src_tgt = Some((src_now, tgt_now));
                    }
                }
            }
        }
        Ok(expected_src_tgt.map(|(idx, jdx)| {
            let idx_part = self
                .quiver
                .p
                .node_weight(idx)
                .expect("This is a vertex of the quiver")
                .clone();
            let jdx_part = self
                .quiver
                .p
                .node_weight(jdx)
                .expect("This is a vertex of the quiver")
                .clone();
            (idx_part, jdx_part)
        }))
    }
}

impl<VertexLabel, EdgeLabel, Coeffs> Add<Self> for PathAlgebra<VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<VertexLabel, EdgeLabel, Coeffs> AddAssign<Self> for PathAlgebra<VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    fn add_assign(&mut self, rhs: Self) {
        assert!(Arc::ptr_eq(&self.quiver, &rhs.quiver));
        for (k, v) in rhs.linear_combination_paths {
            self.linear_combination_paths
                .entry(k)
                .and_modify(|e| *e += v.clone())
                .or_insert(v);
        }
    }
}

impl<VertexLabel, EdgeLabel, Coeffs> Sub<Self> for PathAlgebra<VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<VertexLabel, EdgeLabel, Coeffs> SubAssign<Self> for PathAlgebra<VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    fn sub_assign(&mut self, rhs: Self) {
        assert!(Arc::ptr_eq(&self.quiver, &rhs.quiver));
        for (k, v) in rhs.linear_combination_paths {
            self.linear_combination_paths
                .entry(k)
                .and_modify(|e| *e -= v.clone())
                .or_insert(-v);
        }
    }
}

impl<VertexLabel, EdgeLabel, Coeffs> Mul<Self> for PathAlgebra<VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

impl<VertexLabel, EdgeLabel, Coeffs> Mul<Coeffs> for PathAlgebra<VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    type Output = Self;

    fn mul(mut self, rhs: Coeffs) -> Self::Output {
        self *= rhs;
        self
    }
}

impl<VertexLabel, EdgeLabel, Coeffs> MulAssign<Self> for PathAlgebra<VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    fn mul_assign(&mut self, rhs: Self) {
        assert!(Arc::ptr_eq(&self.quiver, &rhs.quiver));
        let mut new_path = HashMap::new();
        for (k1, v1) in &self.linear_combination_paths {
            for (k2, v2) in &rhs.linear_combination_paths {
                let new_k = self.quiver.multiply_basis(k1, k2);
                if let Some(new_k) = new_k {
                    let coeff_contrib = v1.clone() * v2.clone();
                    new_path
                        .entry(new_k)
                        .and_modify(|e| *e += coeff_contrib.clone())
                        .or_insert(coeff_contrib);
                }
            }
        }
        self.linear_combination_paths = new_path;
    }
}

impl<VertexLabel, EdgeLabel, Coeffs> MulAssign<Coeffs>
    for PathAlgebra<VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    fn mul_assign(&mut self, rhs: Coeffs) {
        self.linear_combination_paths
            .iter_mut()
            .for_each(|(_, coeff)| {
                *coeff *= rhs.clone();
            });
    }
}

impl<VertexLabel, EdgeLabel, Coeffs> Neg for PathAlgebra<VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.linear_combination_paths
            .iter_mut()
            .for_each(|(_, coeff)| {
                *coeff = -coeff.clone();
            });
        self
    }
}

impl<VertexLabel, EdgeLabel, Coeffs> IntoIterator for PathAlgebra<VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    type Item = (BasisElt<VertexLabel, EdgeLabel>, Coeffs);

    type IntoIter = std::collections::hash_map::IntoIter<BasisElt<VertexLabel, EdgeLabel>, Coeffs>;

    fn into_iter(self) -> Self::IntoIter {
        self.linear_combination_paths.into_iter()
    }
}

impl<VertexLabel, EdgeLabel, Coeffs> PartialEq for PathAlgebra<VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone + Ord,
    EdgeLabel: Eq + Clone + std::hash::Hash + Ord,
    Coeffs: Ring + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        if !Arc::ptr_eq(&self.quiver, &other.quiver) {
            return false;
        }
        let mut self_parts: Vec<_> = self.clone().into_iter().collect();
        self_parts.sort_by_key(|z| z.0.clone());
        let mut other_parts: Vec<_> = other.clone().into_iter().collect();
        other_parts.sort_by_key(|z| z.0.clone());
        self_parts == other_parts
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::Quiver;

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

    #[test]
    fn kronecker() {
        use super::{BasisElt, PathAlgebra};
        use std::sync::Arc;
        let kronecker_quiver = make_kronecker_quiver();

        assert_eq!(kronecker_quiver.num_vertices(), 2);
        assert!(kronecker_quiver.is_acyclic());
        let kronecker_quiver = Arc::new(kronecker_quiver);
        let xa = PathAlgebra::singleton(
            kronecker_quiver.clone(),
            BasisElt::Path(nonempty::nonempty!["a"]),
            1.0,
        );
        assert_eq!(xa.all_parallel(), Ok(Some(("alpha", "beta"))));
        let xb = PathAlgebra::singleton(
            kronecker_quiver.clone(),
            BasisElt::Path(nonempty::nonempty!["b"]),
            1.0,
        );
        assert_eq!(xa.all_parallel(), Ok(Some(("alpha", "beta"))));
        let comb = xa.clone() - xb.clone() * 5.0;
        assert_eq!(comb.all_parallel(), Ok(Some(("alpha", "beta"))));
        let comb2 = -xb.clone() * 5.0 + xa.clone();
        assert_eq!(comb2.all_parallel(), Ok(Some(("alpha", "beta"))));
        let mut prod = xa * xb;
        assert_eq!(prod.all_parallel(), Ok(None));
        prod *= 303.95;
        prod -= comb;
        prod = -prod;
        assert!(prod == comb2);
    }

    #[test]
    fn ginzburg() {
        use super::{BasisElt, PathAlgebra};
        use std::sync::Arc;
        let (ginzburg_quiver, adjoint_pairs, self_loops) = make_ginzburg_quiver();
        assert_eq!(ginzburg_quiver.num_vertices(), 1);
        assert!(!ginzburg_quiver.is_acyclic());
        assert_eq!(adjoint_pairs.len(), 1);
        assert_eq!(self_loops.len(), 1);
        assert_eq!(adjoint_pairs[0], ("A".to_string(), "ADagger".to_string()));
        assert_eq!(self_loops[0], "Omega0".to_string());
        let ginzburg_quiver = Arc::new(ginzburg_quiver);

        let alt_ginz_cubic = PathAlgebra::create_ginzburg_cubic(
            ginzburg_quiver.clone(),
            adjoint_pairs,
            self_loops,
            &1.0,
        );

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

        assert_eq!(x_a.all_parallel(), Ok(Some(("0", "0"))));
        assert_eq!(x_adag.all_parallel(), Ok(Some(("0", "0"))));
        assert_eq!(x_omega.all_parallel(), Ok(Some(("0", "0"))));

        let ginz_cubic = x_omega * (x_a.clone() * x_adag.clone() - x_adag.clone() * x_a.clone());
        assert!(ginz_cubic.is_cyclic());
        assert_eq!(ginz_cubic.all_parallel(), Ok(Some(("0", "0"))));
        assert_eq!(ginz_cubic, alt_ginz_cubic);

        let mut ginz_cubic_d_omega = ginz_cubic.clone();
        ginz_cubic_d_omega.cyclic_derivative(&"Omega0".to_string());
        assert_eq!(ginz_cubic_d_omega.all_parallel(), Ok(Some(("0", "0"))));

        let expected_cyclic_derivative =
            x_a.clone() * x_adag.clone() - x_adag.clone() * x_a.clone();
        assert!(ginz_cubic_d_omega == expected_cyclic_derivative);
    }
}
