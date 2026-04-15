use nonempty::NonEmpty;
use std::{
    collections::HashMap,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    sync::Arc,
};

use crate::arithmetic_utils::Ring;
use crate::quiver_algebra::quiver::{BasisElt, Quiver};

#[must_use]
#[derive(Clone, Debug)]
pub struct PathAlgebra<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
    linear_combination_paths: HashMap<BasisElt<VertexLabel, EdgeLabel>, Coeffs>,
}

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool>
    PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    /// Construct an element of kQ^{op} from a map of basis elements to coefficients.
    ///
    /// # Panics
    ///
    /// Panics if any key in `linear_combination_paths` is invalid for `quiver`: idempotents must
    /// refer to vertices that exist, and paths must be composable sequences of arrows in the quiver.
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

    #[allow(clippy::missing_panics_doc)]
    pub(crate) fn map_labels<V2, E2>(
        self,
        fv: impl Fn(VertexLabel) -> V2,
        fe: impl Fn(EdgeLabel) -> E2,
    ) -> HashMap<BasisElt<V2, E2>, Coeffs>
    where
        V2: Eq + std::hash::Hash + Clone,
        E2: Eq + std::hash::Hash + Clone,
    {
        let mut to_return = HashMap::with_capacity(self.linear_combination_paths.len());
        for (k, v) in self.linear_combination_paths {
            let new_key = k.map_labels(&fv, &fe);
            to_return
                .entry(new_key)
                .and_modify(|value| {
                    *value += v.clone();
                })
                .or_insert(v);
        }
        to_return
    }

    /// Construct a scalar multiple of a single basis element.
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
    /// From this `W = \sum_{x in arrows Q} omega_{src(x)} x x_dagger - omega_{tgt(x)} x_dagger x`
    /// is constructed and returned as an element of the path algebra `kQ''^{op}'`.
    #[allow(clippy::missing_panics_doc)]
    fn create_ginzburg_cubic(
        quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
        arrows_and_daggers: Vec<(EdgeLabel, EdgeLabel)>,
        self_loops: Vec<EdgeLabel>,
    ) -> Self {
        let mut places_to_self_loops = HashMap::with_capacity(self_loops.len());
        for self_loop in self_loops {
            let (src, _) = quiver
                .edge_endpoint_labels(&self_loop)
                .expect("This is an edge of the quiver");
            places_to_self_loops.insert(src, BasisElt::Path(nonempty::nonempty![self_loop]));
        }
        let mut ginzburg_cubic = Self::zero(quiver);
        let no_arrows = arrows_and_daggers.is_empty();
        for (a, a_dagger) in arrows_and_daggers {
            let (a_src, a_tgt) = ginzburg_cubic
                .quiver()
                .edge_endpoint_labels(&a)
                .expect("This is an edge of the quiver");
            let a_part = BasisElt::Path(nonempty::nonempty![a]);
            let (a_dagger_src, a_dagger_tgt) = ginzburg_cubic
                .quiver()
                .edge_endpoint_labels(&a_dagger)
                .expect("This is an edge of the quiver");
            debug_assert!(a_dagger_src == a_tgt);
            debug_assert!(a_dagger_tgt == a_src);
            let a_dagger_part = BasisElt::Path(nonempty::nonempty![a_dagger]);
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
            if OP_ALG {
                let loop_at_a_dagger_tgt = PathAlgebra::singleton(
                    ginzburg_cubic.quiver().clone(),
                    loop_at_a_dagger_tgt,
                    Coeffs::one(),
                );
                let loop_at_a_tgt = PathAlgebra::singleton(
                    ginzburg_cubic.quiver().clone(),
                    loop_at_a_tgt,
                    Coeffs::one(),
                );
                ginzburg_cubic += loop_at_a_dagger_tgt * a_part.clone() * a_dagger_part.clone()
                    - loop_at_a_tgt * a_dagger_part.clone() * a_part.clone();
            } else {
                let loop_at_a_dagger_src = PathAlgebra::singleton(
                    ginzburg_cubic.quiver().clone(),
                    loop_at_a_tgt,
                    Coeffs::one(),
                );
                let loop_at_a_src = PathAlgebra::singleton(
                    ginzburg_cubic.quiver().clone(),
                    loop_at_a_dagger_tgt,
                    Coeffs::one(),
                );
                let mut summand1 = PathAlgebra::singleton(
                    ginzburg_cubic.quiver().clone(),
                    a_dagger_part.clone(),
                    Coeffs::one(),
                );
                let mut summand2 = PathAlgebra::singleton(
                    ginzburg_cubic.quiver().clone(),
                    a_part.clone(),
                    Coeffs::one(),
                );
                summand1 *= a_part;
                summand1 *= loop_at_a_src;
                summand2 *= a_dagger_part;
                summand2 *= loop_at_a_dagger_src;
                ginzburg_cubic += summand1 - summand2;
            }
        }
        debug_assert!(
            no_arrows || ginzburg_cubic.might_be_nonzero(),
            "Ginzburg cubic is zero for a quiver with arrows: kQ^op convention violated"
        );
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

    pub fn simplify_default(&mut self) {
        self.simplify(num::Zero::is_zero);
    }

    /// The underlying quiver this element lives in.
    #[allow(clippy::must_use_candidate)]
    pub fn quiver(&self) -> &Arc<Quiver<VertexLabel, EdgeLabel>> {
        &self.quiver
    }

    /// Returns `true` if this element has exactly one term (one basis element with nonzero coefficient).
    #[allow(clippy::must_use_candidate)]
    pub fn is_monomial(&self) -> bool {
        self.linear_combination_paths.len() == 1
    }

    /// Returns `true` if every term has path length exactly `degree` (idempotents count as degree 0).
    #[allow(clippy::must_use_candidate)]
    pub fn is_homogeneous_of_degree(&self, degree: usize) -> bool {
        self.linear_combination_paths.keys().all(|path| match path {
            BasisElt::Path(path) => path.len() == degree,
            BasisElt::Idempotent(_) => degree == 0,
        })
    }

    /// Returns `true` if every term has path length at most `degree`.
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

    /// Partition into `(cyclic_part, acyclic_part)`.
    ///
    /// A term is cyclic if its last arrow composes back into its first arrow
    /// (i.e., the path forms a closed loop). Idempotents are acyclic.
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

    /// Replace `self` with its cyclic partial derivative with respect to `wrt_edge`.
    ///
    /// This is intended to be applied to a cyclic word (superpotential W). For each occurrence
    /// of `wrt_edge = aᵢ` in a term `c · [a₁, …, aₙ]`:
    /// - If n > 1: contributes `c · [aᵢ₊₁, …, aₙ, a₁, …, aᵢ₋₁]`.
    /// - If n = 1: the summand must be a self-loop for the word to be cyclic; contributes
    ///   `c · e_{s(wrt_edge)}`, or 0 if `wrt_edge` does not appear.
    ///
    /// # Panics
    ///
    /// Panics if `wrt_edge` is not an edge of the quiver. In debug builds, also panics if a
    /// length-1 term contains `wrt_edge` but `wrt_edge` is not a self-loop (the input was not a
    /// valid cyclic word).
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
                    new_linear_combination
                        .entry(BasisElt::Idempotent(wrt_edge_endpoints.0.clone()))
                        .and_modify(|existing| *existing += v.clone())
                        .or_insert(v.clone());
                } else {
                    new_linear_combination
                        .entry(BasisElt::Path(
                            NonEmpty::from_vec(k_now).expect("Checked that it is nonempty"),
                        ))
                        .and_modify(|existing| *existing += v.clone())
                        .or_insert(v.clone());
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
                        .vertex_idx(just_vertex)
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
                .quiver()
                .vertex_label(idx)
                .expect("This is a vertex of the quiver")
                .clone();
            let jdx_part = self
                .quiver()
                .vertex_label(jdx)
                .expect("This is a vertex of the quiver")
                .clone();
            (idx_part, jdx_part)
        }))
    }
}

impl<VertexLabel, EdgeLabel, Coeffs> PathAlgebra<VertexLabel, EdgeLabel, Coeffs, true>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    /// Reinterpret this kQ^{op} element as a kQ element without changing the stored data.
    /// Only the multiplication convention changes; basis elements and coefficients are preserved.
    pub fn toggle_convention(self) -> PathAlgebra<VertexLabel, EdgeLabel, Coeffs, false> {
        PathAlgebra {
            quiver: self.quiver,
            linear_combination_paths: self.linear_combination_paths,
        }
    }
}

impl<VertexLabel, EdgeLabel, Coeffs> PathAlgebra<VertexLabel, EdgeLabel, Coeffs, false>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    /// Reinterpret this kQ element as a kQ^{op} element without changing the stored data.
    /// Only the multiplication convention changes; basis elements and coefficients are preserved.
    pub fn toggle_convention(self) -> PathAlgebra<VertexLabel, EdgeLabel, Coeffs, true> {
        PathAlgebra {
            quiver: self.quiver,
            linear_combination_paths: self.linear_combination_paths,
        }
    }
}

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool> Add<Self>
    for PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
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

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool> AddAssign<Self>
    for PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
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

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool> Sub<Self>
    for PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
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

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool> SubAssign<Self>
    for PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
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

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool> Mul<Self>
    for PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
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

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool> Mul<BasisElt<VertexLabel, EdgeLabel>>
    for PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    type Output = Self;

    fn mul(mut self, rhs: BasisElt<VertexLabel, EdgeLabel>) -> Self::Output {
        self *= rhs;
        self
    }
}

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool> Mul<Coeffs>
    for PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
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

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool> MulAssign<Self>
    for PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
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
                let new_k = self.quiver.multiply_basis::<OP_ALG>(k1, k2);
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

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool> MulAssign<BasisElt<VertexLabel, EdgeLabel>>
    for PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    fn mul_assign(&mut self, rhs: BasisElt<VertexLabel, EdgeLabel>) {
        let mut new_path = HashMap::new();
        for (k1, v1) in &self.linear_combination_paths {
            let new_k = self.quiver.multiply_basis::<OP_ALG>(k1, &rhs);
            if let Some(new_k) = new_k {
                let coeff_contrib = v1.clone();
                new_path
                    .entry(new_k)
                    .and_modify(|e| *e += coeff_contrib.clone())
                    .or_insert(coeff_contrib);
            }
        }
        self.linear_combination_paths = new_path;
    }
}

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool>
    Mul<PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>> for BasisElt<VertexLabel, EdgeLabel>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    type Output = PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>;

    fn mul(self, mut rhs: PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>) -> Self::Output {
        let mut new_path = HashMap::new();
        for (k2, v2) in &rhs.linear_combination_paths {
            let new_k = rhs.quiver.multiply_basis::<OP_ALG>(&self, k2);
            if let Some(new_k) = new_k {
                let coeff_contrib = v2.clone();
                new_path
                    .entry(new_k)
                    .and_modify(|e| *e += coeff_contrib.clone())
                    .or_insert(coeff_contrib);
            }
        }
        rhs.linear_combination_paths = new_path;
        rhs
    }
}

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool> MulAssign<Coeffs>
    for PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
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

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool> Neg
    for PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
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

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool> IntoIterator
    for PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
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

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool> PartialEq
    for PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
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

impl<VertexLabel, EdgeLabel> Quiver<VertexLabel, EdgeLabel>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
{
    #[allow(clippy::type_complexity)]
    pub fn ginzburgify_and_cubic<Coeffs: Ring, const OP_ALG: bool>(
        self,
        dagger: impl Fn(&EdgeLabel) -> EdgeLabel,
        self_loop: impl Fn(&VertexLabel) -> EdgeLabel,
    ) -> (
        Arc<Self>,
        Vec<(EdgeLabel, EdgeLabel)>,
        Vec<EdgeLabel>,
        PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>,
    ) {
        let (new_self, adjoint_pairs, new_self_loops) = self.ginzburgify(dagger, self_loop);
        let new_self_arc = Arc::new(new_self);
        (
            new_self_arc.clone(),
            adjoint_pairs.clone(),
            new_self_loops.clone(),
            PathAlgebra::create_ginzburg_cubic(new_self_arc, adjoint_pairs, new_self_loops),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::arithmetic_utils::Ring;
    use crate::quiver_algebra::{
        PathAlgebra, Quiver,
        quiver::tests::{
            arbitrary_basis_element, make_ginzburg_quiver, make_kronecker_quiver,
            testing_arbitrary_quiver,
        },
    };
    use proptest::prelude::Strategy;
    use std::fmt::Debug;

    #[test]
    fn kronecker() {
        use super::{BasisElt, PathAlgebra};
        use std::sync::Arc;
        let kronecker_quiver = make_kronecker_quiver();

        assert_eq!(kronecker_quiver.num_vertices(), 2);
        assert!(kronecker_quiver.is_acyclic());
        let kronecker_quiver = Arc::new(kronecker_quiver);
        let xa = PathAlgebra::<_, _, _, true>::singleton(
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
    fn ginzburg_true() {
        ginzburg::<true>();
    }

    #[test]
    fn ginzburg_false() {
        ginzburg::<false>();
    }

    fn ginzburg<const OP_ALG: bool>() {
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

        let alt_ginz_cubic = PathAlgebra::<_, _, _, OP_ALG>::create_ginzburg_cubic(
            ginzburg_quiver.clone(),
            adjoint_pairs,
            self_loops,
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

        let ginz_cubic;
        if OP_ALG {
            ginz_cubic = x_omega * (x_a.clone() * x_adag.clone() - x_adag.clone() * x_a.clone());
        } else {
            ginz_cubic = (-x_a.clone() * x_adag.clone() + x_adag.clone() * x_a.clone()) * x_omega;
        }
        assert!(ginz_cubic.is_cyclic());
        assert_eq!(ginz_cubic.all_parallel(), Ok(Some(("0", "0"))));
        assert_eq!(ginz_cubic, alt_ginz_cubic);

        let mut ginz_cubic_d_omega = ginz_cubic.clone();
        ginz_cubic_d_omega.cyclic_derivative(&"Omega0".to_string());
        assert_eq!(ginz_cubic_d_omega.all_parallel(), Ok(Some(("0", "0"))));

        let expected_cyclic_derivative;
        if OP_ALG {
            expected_cyclic_derivative =
                x_a.clone() * x_adag.clone() - x_adag.clone() * x_a.clone();
        } else {
            expected_cyclic_derivative =
                -x_a.clone() * x_adag.clone() + x_adag.clone() * x_a.clone();
        }
        assert!(ginz_cubic_d_omega == expected_cyclic_derivative);
    }

    #[allow(dead_code)]
    fn arbitrary_path_algebra<'a, VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool>(
        q: Quiver<VertexLabel, EdgeLabel>,
        coeff_strat: impl Strategy<Value = Coeffs> + 'a,
    ) -> impl Strategy<Value = super::PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>> + 'a
    where
        VertexLabel: std::hash::Hash + Eq + Clone + Debug + 'static,
        EdgeLabel: Eq + std::hash::Hash + Clone + Debug + 'static,
        Coeffs: Ring + Debug + 'a,
    {
        proptest::collection::hash_map(arbitrary_basis_element(q.clone()), coeff_strat, 0..6)
            .prop_map(move |hm| PathAlgebra::new(std::sync::Arc::new(q.clone()), hm))
    }

    proptest::proptest! {
        #[test]
        fn mul_op(
            q1 in arbitrary_path_algebra::<_,_,f64,true>(testing_arbitrary_quiver(), proptest::sample::select(vec![-1.0,-4.0,0.5,1.25])),
            mut q2 in arbitrary_path_algebra::<_,_,f64,true>(testing_arbitrary_quiver(), proptest::sample::select(vec![-1.0,-4.0,0.5,1.25])),
        ) {
            q2.quiver = q1.quiver().clone();
            let q1_then_q2 = q1.clone()*q2.clone();
            let q1 = q1.toggle_convention();
            let q2 = q2.toggle_convention();
            let q1_then_q2_regular = (q2 * q1).toggle_convention();
            assert_eq!(q1_then_q2, q1_then_q2_regular);
        }

        #[test]
        fn mul_reg(
            q1 in arbitrary_path_algebra::<_,_,f64,false>(testing_arbitrary_quiver(), proptest::sample::select(vec![-1.0,-4.0,0.5,1.25])),
            mut q2 in arbitrary_path_algebra::<_,_,f64,false>(testing_arbitrary_quiver(), proptest::sample::select(vec![-1.0,-4.0,0.5,1.25])),
        ) {
            q2.quiver = q1.quiver().clone();
            let q2_then_q1 = q1.clone()*q2.clone();
            let q1 = q1.toggle_convention();
            let q2 = q2.toggle_convention();
            let q2_then_q1_op = (q2 * q1).toggle_convention();
            assert_eq!(q2_then_q1, q2_then_q1_op);
        }
    }
}
