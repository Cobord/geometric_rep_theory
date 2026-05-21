use std::ops::{AddAssign, Mul, Neg};

use num::Zero;

use crate::arithmetic_utils::SemiRing;

use super::algebra::ClusterAlgebra;

/// A cluster algebra equipped with a compatible quadratic (log-canonical) Poisson structure.
///
/// The Poisson bracket is `{x_i, x_j} = λ_{ij} x_i x_j` where `lambda` is a skew-symmetric
/// integer matrix.  Compatibility with the cluster algebra means that `lambda` transforms
/// covariantly under mutation: when mutating at vertex k, the k-th row and column of `lambda`
/// are updated so that the bracket remains log-canonical in the new cluster.
///
/// # Mutation formula
///
/// For each j ≠ k:
/// ```text
/// λ'_{kj} = -λ_{kj} + Σ_{i→k} B_{ik} · λ_{ij}
/// ```
/// where the sum runs over all i with a net positive arrow count into k (`B_{ik} > 0`).
/// Skew-symmetry then gives `λ'_{jk} = -λ'_{kj}`.  All other entries are unchanged.
pub struct PoissonClusterAlgebra<const N: usize, VertexLabel, EdgeLabel, Coeffs, BaseRing>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
    Coeffs: SemiRing,
    // For the full poisson bracket on arbitrary pairs of `Coeffs`
    // Need `BaseRing` to be included into `Coeffs` as data not property.
    // But to just define the poisson bracket within any
    // pair `{x_i, x_j}` for cluster `{x_1,....x_N}`
    // it only needs to be in `BaseRing`.
    BaseRing: Zero
        + Neg<Output = BaseRing>
        + PartialEq
        + Clone
        + AddAssign<BaseRing>
        + Mul<i64, Output = BaseRing>,
{
    algebra: ClusterAlgebra<N, VertexLabel, EdgeLabel, Coeffs>,
    /// Skew-symmetric Poisson matrix indexed by the same vertex ordering as the algebra.
    lambda: [[BaseRing; N]; N],
}

impl<const N: usize, VertexLabel, EdgeLabel, Coeffs: SemiRing, BaseRing>
    From<ClusterAlgebra<N, VertexLabel, EdgeLabel, Coeffs>>
    for PoissonClusterAlgebra<N, VertexLabel, EdgeLabel, Coeffs, BaseRing>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
    BaseRing: Zero
        + Neg<Output = BaseRing>
        + PartialEq
        + Clone
        + AddAssign<BaseRing>
        + Mul<i64, Output = BaseRing>,
{
    fn from(algebra: ClusterAlgebra<N, VertexLabel, EdgeLabel, Coeffs>) -> Self {
        let lambda = core::array::from_fn(|_| core::array::from_fn(|_| BaseRing::zero()));
        Self { algebra, lambda }
    }
}

impl<const N: usize, VertexLabel, EdgeLabel, Coeffs: SemiRing, BaseRing>
    PoissonClusterAlgebra<N, VertexLabel, EdgeLabel, Coeffs, BaseRing>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
    BaseRing: Zero
        + Neg<Output = BaseRing>
        + PartialEq
        + Clone
        + AddAssign<BaseRing>
        + Mul<i64, Output = BaseRing>,
{
    /// Wrap an existing cluster algebra with a Poisson structure given by `lambda`.
    ///
    /// # Panics
    ///
    /// Panics if `lambda` is not skew-symmetric.
    pub fn new(
        algebra: ClusterAlgebra<N, VertexLabel, EdgeLabel, Coeffs>,
        lambda: [[BaseRing; N]; N],
    ) -> Self {
        #[allow(clippy::needless_range_loop)]
        for i in 0..N {
            for j in 0..N {
                assert!(
                    lambda[i][j] == -lambda[j][i].clone(),
                    "lambda must be skew-symmetric"
                );
            }
        }
        Self { algebra, lambda }
    }

    /// The Poisson coefficient `λ_{ij}` for the pair of vertices.
    /// Returns `None` if either vertex is not in the algebra.
    pub fn poisson_coefficient(&self, v1: &VertexLabel, v2: &VertexLabel) -> Option<&BaseRing> {
        let i = self.algebra.vertex_index(v1)?;
        let j = self.algebra.vertex_index(v2)?;
        Some(&self.lambda[i][j])
    }

    /// Set `λ_{v1,v2} = value` and `λ_{v2,v1} = -value` by vertex label.
    /// Does nothing if either vertex is not in the algebra.
    pub fn set_bracket(&mut self, v1: &VertexLabel, v2: &VertexLabel, value: BaseRing) {
        let Some(i) = self.algebra.vertex_index(v1) else {
            return;
        };
        let Some(j) = self.algebra.vertex_index(v2) else {
            return;
        };
        self.lambda[i][j] = value.clone();
        self.lambda[j][i] = -value;
    }

    /// Mark `vertex` as frozen; delegates to the inner [`ClusterAlgebra`].
    pub fn freeze(&mut self, vertex: VertexLabel) {
        self.algebra.freeze(vertex);
    }

    /// Unfreeze `vertex`; delegates to the inner [`ClusterAlgebra`].
    pub fn unfreeze(&mut self, vertex: &VertexLabel) {
        self.algebra.unfreeze(vertex);
    }

    /// Returns `true` if `vertex` is frozen; delegates to the inner [`ClusterAlgebra`].
    pub fn is_frozen(&self, vertex: &VertexLabel) -> bool {
        self.algebra.is_frozen(vertex)
    }

    /// Read off cluster variable values; delegates to the inner [`ClusterAlgebra`].
    pub fn view_cluster<const M: usize>(&self, vertex: [VertexLabel; M]) -> [(Coeffs, Coeffs); M] {
        self.algebra.view_cluster(vertex)
    }

    /// Mutate at `vertex`, updating both the cluster variables and the Poisson matrix.
    /// Does nothing if `vertex` is frozen or not in the algebra.
    pub fn mutate(&mut self, vertex: &VertexLabel) {
        if self.algebra.is_frozen(vertex) {
            return;
        }
        let Some(k) = self.algebra.vertex_index(vertex) else {
            return;
        };

        // Compute net arrow counts B[i][k] from the quiver before mutation.
        // B[i][k] = #{arrows i→k} - #{arrows k→i}
        let mut b_col = [0; N];
        for (_, src) in self.algebra.quiver().predecessors(vertex) {
            if let Some(i) = self.algebra.vertex_index(&src) {
                b_col[i] += 1;
            }
        }
        for (_, tgt) in self.algebra.quiver().successors(vertex) {
            if let Some(i) = self.algebra.vertex_index(&tgt) {
                b_col[i] -= 1;
            }
        }

        // Update lambda: λ'_{kj} = -λ_{kj} + Σ_{i: B[i][k]>0} B[i][k] · λ[i][j]
        let old_lambda = self.lambda.clone();
        #[allow(clippy::needless_range_loop)]
        for j in 0..N {
            if j == k {
                continue;
            }
            let mut new_kj = -old_lambda[k][j].clone();
            for i in 0..N {
                if b_col[i] > 0 {
                    new_kj += old_lambda[i][j].clone() * b_col[i];
                }
            }
            self.lambda[k][j] = new_kj.clone();
            self.lambda[j][k] = -new_kj;
        }

        self.algebra.mutate(vertex);
    }
}

#[cfg(test)]
mod test {
    use super::PoissonClusterAlgebra;
    use crate::cluster_algebra::ClusterAlgebra;
    use crate::quiver_algebra::make_a2_quiver;

    fn a2_string_quiver() -> crate::quiver_algebra::Quiver<&'static str, String> {
        make_a2_quiver().map_labels(|v| v, |e| e.to_string())
    }

    #[test]
    fn a2_poisson() {
        let alg = ClusterAlgebra::<2, _, _, i32>::new(
            a2_string_quiver(),
            [("alpha", 5), ("beta", 7)].into_iter().collect(),
            |n| format!("__gen_{n}"),
        )
        .expect("valid quiver and seed");

        let mut palg = PoissonClusterAlgebra::from(alg);
        palg.set_bracket(&"alpha", &"beta", 1);
        assert_eq!(
            palg.poisson_coefficient(&"alpha", &"beta").copied(),
            Some(1)
        );
        assert_eq!(
            palg.poisson_coefficient(&"beta", &"alpha").copied(),
            Some(-1)
        );

        palg.mutate(&"alpha");
        assert_eq!(palg.view_cluster(["alpha"]), [(8, 5)]);
        assert_eq!(palg.view_cluster(["beta"]), [(7, 1)]);

        // alpha had no incoming arrows, so the correction term is zero:
        // λ'_{alpha,beta} = -λ_{alpha,beta} = -1
        assert_eq!(
            palg.poisson_coefficient(&"alpha", &"beta").copied(),
            Some(-1)
        );
        assert_eq!(
            palg.poisson_coefficient(&"beta", &"alpha").copied(),
            Some(1)
        );
    }

    // A3 quiver v1→v2→v3: mutating at v2 (which has both an incoming and outgoing
    // arrow) exercises the correction term Σ B_{ik} λ_{ij} non-trivially.
    #[test]
    fn a3_poisson_nontrivial() {
        use crate::quiver_algebra::Quiver;

        let mut quiver: Quiver<&str, String> = Quiver::new();
        quiver.add_edge("v1", "v2", "a".to_string());
        quiver.add_edge("v2", "v3", "b".to_string());

        let alg = ClusterAlgebra::<3, _, _, i32>::new(
            quiver,
            [("v1", 2), ("v2", 3), ("v3", 5)].into_iter().collect(),
            |n| format!("__gen_{n}"),
        )
        .expect("valid quiver and seed");

        let mut palg = PoissonClusterAlgebra::from(alg);
        palg.set_bracket(&"v1", &"v2", 1i64);
        palg.set_bracket(&"v2", &"v3", 1);
        palg.set_bracket(&"v1", &"v3", 3);

        palg.mutate(&"v2");

        // λ'_{v1,v2}: sign flip since j=v1 has no positive b_col contribution
        assert_eq!(palg.poisson_coefficient(&"v1", &"v2").copied(), Some(-1));
        // λ'_{v2,v3} = -λ_{v2,v3} + b_col[v1]·λ_{v1,v3} = -1 + 1·3 = 2
        assert_eq!(palg.poisson_coefficient(&"v2", &"v3").copied(), Some(2));
        // λ_{v1,v3} is between two non-k vertices, unchanged
        assert_eq!(palg.poisson_coefficient(&"v1", &"v3").copied(), Some(3));
    }
}
