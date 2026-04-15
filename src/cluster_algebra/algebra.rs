use std::collections::{HashMap, HashSet};

use crate::quiver_algebra::Quiver;
use crate::arithmetic_utils::SemiRing;

/// A cluster algebra of rank `N` with exchange graph encoded as a quiver.
///
/// The seed stores each cluster variable as a fraction `(numerator, denominator)` built from the
/// initial seed by iterated mutation.  Mutations are performed in-place and update both the quiver
/// and the seed via the exchange relation.  Vertices may be frozen to prevent mutation while still
/// participating in exchange relations.
pub struct ClusterAlgebra<const N: usize, VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
    Coeffs: SemiRing,
{
    quiver: Quiver<VertexLabel, EdgeLabel>,
    vertices_sorted: [VertexLabel; N],
    seed: [(Coeffs, Coeffs); N],
    frozen: HashSet<VertexLabel>,
    label_counter: usize,
    make_edge_label: Box<dyn Fn(usize) -> EdgeLabel>,
}

fn ring_pow<C: SemiRing>(base: C, exp: usize) -> C {
    match exp {
        0 => C::one(),
        1 => base,
        2 => base.clone() * base,
        _ => {
            let mut base = base;
            let mut exp = exp;
            let mut result = C::one();
            while exp > 0 {
                if exp & 1 == 1 {
                    result *= base.clone();
                }
                exp >>= 1;
                if exp > 0 {
                    base = base.clone() * base;
                }
            }
            result
        }
    }
}

fn pow_fraction<C: SemiRing>(base: (C, C), exp: usize) -> (C, C) {
    (ring_pow(base.0, exp), ring_pow(base.1, exp))
}

impl<const N: usize, VertexLabel, EdgeLabel, Coeffs: SemiRing>
    ClusterAlgebra<N, VertexLabel, EdgeLabel, Coeffs>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
{
    /// Create a new cluster algebra from the given quiver and seed.
    ///
    /// # Errors
    ///
    /// Returns `Err` if
    /// - `N` is zero (use `degenerate_new` instead)
    /// - the quiver vertex count does not equal `N`
    /// - the keys of seed map are not precisely the `N` vertices in the quiver
    #[allow(clippy::missing_panics_doc)]
    pub fn new(
        quiver: Quiver<VertexLabel, EdgeLabel>,
        seed_map: HashMap<VertexLabel, Coeffs>,
        make_edge_label: impl Fn(usize) -> EdgeLabel + 'static,
    ) -> Result<Self, String> {
        if N == 0 {
            return Err("N is zero; use degenerate_new for the empty cluster algebra".to_string());
        }
        if quiver.num_vertices() != N {
            return Err(format!(
                "The number of vertices in the quiver ({}) does not match the size of the seed ({})",
                quiver.num_vertices(),
                N
            ));
        }
        let mut seed = [(); N].map(|()| Coeffs::one());
        let vertex = quiver
            .vertex_labels()
            .next()
            .cloned()
            .expect("quiver is non-empty since N > 0");
        let mut vertices_sorted = [(); N].map(|()| vertex.clone());
        for (cur_idx, (vertex, coeff)) in seed_map.into_iter().enumerate() {
            assert!(
                cur_idx < N,
                "Seed map has more entries than the number of vertices in the quiver"
            );
            assert!(
                quiver.contains_vertex(&vertex),
                "Seed map contains vertex not in the quiver"
            );
            vertices_sorted[cur_idx] = vertex.clone();
            seed[cur_idx] = coeff;
        }
        let seed = seed.map(|x| (x, Coeffs::one()));
        let label_counter = quiver.edge_labels().count();
        Ok(Self {
            quiver,
            vertices_sorted,
            seed,
            frozen: HashSet::new(),
            label_counter,
            make_edge_label: Box::new(make_edge_label),
        })
    }

    /// Construct the degenerate empty cluster algebra with no vertices.
    pub fn degenerate_new(
        make_edge_label: impl Fn(usize) -> EdgeLabel + 'static,
    ) -> ClusterAlgebra<0, VertexLabel, EdgeLabel, Coeffs> {
        ClusterAlgebra {
            quiver: Quiver::new(),
            vertices_sorted: [],
            seed: [],
            frozen: HashSet::new(),
            label_counter: 0,
            make_edge_label: Box::new(make_edge_label),
        }
    }

    fn next_edge_label(&mut self) -> EdgeLabel {
        let label = (self.make_edge_label)(self.label_counter);
        self.label_counter += 1;
        if self.quiver.contains_edge(&label) {
            self.next_edge_label()
        } else {
            label
        }
    }

    /// Mark a vertex as frozen. Frozen vertices participate in exchange relations but cannot be mutated.
    /// Does nothing if the vertex is not in the quiver.
    pub fn freeze(&mut self, vertex: VertexLabel) {
        if self.quiver.contains_vertex(&vertex) {
            self.frozen.insert(vertex);
        }
    }

    /// Unfreeze a previously frozen vertex. Does nothing if the vertex is not frozen.
    pub fn unfreeze(&mut self, vertex: &VertexLabel) {
        self.frozen.remove(vertex);
    }

    /// Returns `true` if `vertex` is frozen.
    pub fn is_frozen(&self, vertex: &VertexLabel) -> bool {
        self.frozen.contains(vertex)
    }

    /// Mutate the cluster algebra at the specified vertex, modifying the quiver and the seed accordingly.
    ///
    /// # Panics
    ///
    /// Panics if the vertex is not in the cluster algebra.
    pub fn mutate(&mut self, vertex: &VertexLabel) {
        if self.frozen.contains(vertex) {
            return;
        }
        let k = self
            .vertices_sorted
            .iter()
            .position(|v| v == vertex)
            .expect("vertex not in cluster algebra");

        // Collect incoming and outgoing arrows before any modification
        let incoming: Vec<(EdgeLabel, VertexLabel)> = self.quiver.predecessors(vertex).collect();
        let outgoing: Vec<(EdgeLabel, VertexLabel)> = self.quiver.successors(vertex).collect();

        // Compute new cluster variable via the exchange relation:
        // x_k * x'_k = P_plus + P_minus
        // P_plus  = product of x_i^(count of i→k) for each source i
        // P_minus = product of x_j^(count of k→j) for each target j
        let mut incoming_counts: HashMap<VertexLabel, usize> = HashMap::new();
        for (_, src) in &incoming {
            *incoming_counts.entry(src.clone()).or_insert(0) += 1;
        }
        let mut outgoing_counts: HashMap<VertexLabel, usize> = HashMap::new();
        for (_, tgt) in &outgoing {
            *outgoing_counts.entry(tgt.clone()).or_insert(0) += 1;
        }

        let mut p_plus = (Coeffs::one(), Coeffs::one());
        for (src, count) in &incoming_counts {
            let idx = self
                .vertices_sorted
                .iter()
                .position(|v| v == src)
                .expect("vertex in cluster algebra");
            let pow = pow_fraction(self.seed[idx].clone(), *count);
            p_plus.0 *= pow.0;
            p_plus.1 *= pow.1;
        }

        let mut p_minus = (Coeffs::one(), Coeffs::one());
        for (tgt, count) in &outgoing_counts {
            let idx = self
                .vertices_sorted
                .iter()
                .position(|v| v == tgt)
                .expect("vertex in cluster algebra");
            let pow = pow_fraction(self.seed[idx].clone(), *count);
            p_minus.0 *= pow.0;
            p_minus.1 *= pow.1;
        }

        // x'_k = (P_plus + P_minus) / x_k  expressed as a fraction
        let (old_num_k, old_den_k) = self.seed[k].clone();
        let new_num = (p_plus.0.clone() * p_minus.1.clone() + p_minus.0.clone() * p_plus.1.clone())
            * old_den_k;
        let new_den = p_plus.1 * p_minus.1 * old_num_k;
        self.seed[k] = (new_num, new_den);

        // Step 1: for each path i → k → j, add a new arrow i → j
        for (_, src) in &incoming {
            for (_, tgt) in &outgoing {
                let new_label = self.next_edge_label();
                self.quiver.add_edge(src.clone(), tgt.clone(), new_label);
            }
        }

        // Step 2: remove all arrows incident to k and add their reverses
        for (edge, src) in &incoming {
            self.quiver.remove_edge(edge);
            let new_label = self.next_edge_label();
            self.quiver.add_edge(vertex.clone(), src.clone(), new_label);
        }
        for (edge, tgt) in &outgoing {
            self.quiver.remove_edge(edge);
            let new_label = self.next_edge_label();
            self.quiver.add_edge(tgt.clone(), vertex.clone(), new_label);
        }

        // Step 3: cancel 2-cycles among all vertex pairs
        let vertices: Vec<VertexLabel> = self.quiver.vertex_labels().cloned().collect();
        for i in 0..vertices.len() {
            for j in (i + 1)..vertices.len() {
                let vi = &vertices[i];
                let vj = &vertices[j];
                let ij: Vec<EdgeLabel> = self
                    .quiver
                    .successors(vi)
                    .filter(|(_, t)| t == vj)
                    .map(|(e, _)| e)
                    .collect();
                let ji: Vec<EdgeLabel> = self
                    .quiver
                    .successors(vj)
                    .filter(|(_, t)| t == vi)
                    .map(|(e, _)| e)
                    .collect();
                let cancel = ij.len().min(ji.len());
                for e in ij.into_iter().take(cancel) {
                    self.quiver.remove_edge(&e);
                }
                for e in ji.into_iter().take(cancel) {
                    self.quiver.remove_edge(&e);
                }
            }
        }
    }

    #[must_use = "This is the version of mutate for method chaining so you have to use the result"]
    pub fn mutated(mut self, vertex: &VertexLabel) -> Self {
        self.mutate(vertex);
        self
    }

    /// Give the values for the cluster variables at the specified vertices.
    /// The result is a pair of coefficients (numerator, denominator),
    /// Numerator and Denominator are subtraction free expressions built from the initial seed variables.
    ///
    /// # Panics
    ///
    /// Panics if any vertex in `vertex` is not in the cluster algebra.
    pub fn view_cluster<const M: usize>(&self, vertex: [VertexLabel; M]) -> [(Coeffs, Coeffs); M] {
        vertex.map(|v| {
            self.vertices_sorted
                .iter()
                .position(|x| *x == v)
                .map_or_else(
                    || panic!("Vertex not found in cluster"),
                    |idx| self.seed[idx].clone(),
                )
        })
    }

    /// The underlying exchange quiver.
    pub fn quiver(&self) -> &Quiver<VertexLabel, EdgeLabel> {
        &self.quiver
    }

    /// Return the index of `vertex` in the internal ordering, or `None` if not present.
    pub fn vertex_index(&self, vertex: &VertexLabel) -> Option<usize> {
        self.vertices_sorted.iter().position(|v| v == vertex)
    }

    pub fn simplify(&mut self, reduce_fractions: impl FnMut(&mut (Coeffs, Coeffs))) {
        self.seed.iter_mut().for_each(reduce_fractions);
    }
}

#[cfg(test)]
mod test {
    use crate::quiver_algebra::make_a2_quiver;

    fn a2_string_quiver() -> crate::quiver_algebra::Quiver<&'static str, String> {
        make_a2_quiver().map_labels(|v| v, |e| e.to_string())
    }

    #[test]
    fn a2_quiver() {
        use super::ClusterAlgebra;

        let mut alg = ClusterAlgebra::<2, _, _, i32>::new(
            a2_string_quiver(),
            [("alpha", 5), ("beta", 7)].into_iter().collect(),
            |n| format!("__gen_{n}"),
        )
        .expect("valid quiver and seed");
        assert_eq!(alg.view_cluster(["alpha", "beta"]), [(5, 1), (7, 1)]);
        assert_eq!(alg.view_cluster(["beta", "alpha"]), [(7, 1), (5, 1)]);
        alg.mutate(&"alpha");
        // x'_alpha = (1 + x_beta) / x_alpha = (1 + 7) / 5 = 8/5
        assert_eq!(alg.view_cluster(["alpha"]), [(8, 5)]);
        assert_eq!(alg.view_cluster(["beta"]), [(7, 1)]);
    }

    // Check that mutating the same vertex twice is the identity.
    #[test]
    fn double_mutation_is_identity() {
        use super::ClusterAlgebra;

        let mut alg = ClusterAlgebra::<2, _, _, i64>::new(
            a2_string_quiver(),
            [("alpha", 5), ("beta", 7)].into_iter().collect(),
            |n| format!("__gen_{n}"),
        )
        .expect("valid quiver and seed");

        alg.mutate(&"alpha");
        alg.mutate(&"alpha");

        // After two mutations at the same vertex, both variables should be
        // back to their initial values.  Compare via cross-multiplication to
        // avoid requiring the fractions to be reduced.
        let [(a_num, a_den)] = alg.view_cluster(["alpha"]);
        let [(b_num, b_den)] = alg.view_cluster(["beta"]);
        assert_eq!(a_num * 1, a_den * 5, "alpha should equal 5/1");
        assert_eq!(b_num * 1, b_den * 7, "beta should equal 7/1");
    }

    // A2 cluster algebras have 5-fold periodicity: alternating mutations
    // mu_alpha, mu_beta repeated cyclically return to the original cluster
    // after 5 steps with the two variables swapped.
    proptest::proptest! {
        #[test]
        fn a2_five_fold_periodicity(a in 1i64..=100, c in 1i64..=100) {
            use super::ClusterAlgebra;

            let mut alg = ClusterAlgebra::<2, _, _, i64>::new(
                a2_string_quiver(),
                [("alpha", a), ("beta", c)].into_iter().collect(),
                |n| format!("__gen_{n}"),
            )
            .expect("valid quiver and seed");

            alg.mutate(&"alpha");
            alg.mutate(&"beta");
            alg.mutate(&"alpha");
            alg.mutate(&"beta");
            alg.mutate(&"alpha");

            // After 5 alternating mutations the two cluster variables are swapped:
            // alpha takes the initial value of beta and vice-versa.
            let [(a_num, a_den)] = alg.view_cluster(["alpha"]);
            let [(b_num, b_den)] = alg.view_cluster(["beta"]);
            proptest::prop_assert_eq!(a_num, a_den * c, "alpha should equal initial beta = c/1");
            proptest::prop_assert_eq!(b_num, b_den * a, "beta should equal initial alpha = a/1");

            // Freeze both vertices and do many mutations — all should be no-ops.
            alg.freeze("alpha");
            alg.freeze("beta");
            for _ in 0..20 {
                alg.mutate(&"alpha");
                alg.mutate(&"beta");
            }
            let [(a_num2, a_den2)] = alg.view_cluster(["alpha"]);
            let [(b_num2, b_den2)] = alg.view_cluster(["beta"]);
            proptest::prop_assert_eq!(a_num2, a_num, "frozen alpha should be unchanged");
            proptest::prop_assert_eq!(a_den2, a_den, "frozen alpha denominator should be unchanged");
            proptest::prop_assert_eq!(b_num2, b_num, "frozen beta should be unchanged");
            proptest::prop_assert_eq!(b_den2, b_den, "frozen beta denominator should be unchanged");
        }
    }
}
