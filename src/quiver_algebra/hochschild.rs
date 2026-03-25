use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
    sync::Arc,
};

use nonempty::NonEmpty;

use crate::quiver_algebra::{
    checked_arith::Field,
    quiver::BasisElt,
    quiver_with_mon_rels::{NonMonomialIdeal, QuiverWithMonomialRelations},
    quiver_with_rels::QuiverWithRelations,
};

pub type IndexInList = usize;
pub type CohomologicalDegree = usize;
pub type CochainBasis = (CochainAnchor, IndexInList);

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Chain<EdgeLabel> {
    pub word: Vec<EdgeLabel>,
    pub rels: Vec<(usize, Vec<EdgeLabel>)>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum CochainAnchor {
    Vertex(IndexInList),
    Chain(IndexInList),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HochschildError {
    NonMonomialIdeal(NonMonomialIdeal),
    BasisEnumerationExceeded {
        max_paths: usize,
    },
    NegativeCohomologyDimension {
        degree: CohomologicalDegree,
        value: isize,
    },
}

#[must_use]
pub struct MonomialQuiverAlgebraHH<VertexLabel, EdgeLabel, Scalar>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: std::hash::Hash + Eq + Clone,
    Scalar: Field,
{
    quiver_with_mon_relations: Arc<QuiverWithMonomialRelations<VertexLabel, EdgeLabel>>,
    chain_search_max_wordlen: usize,
    basis_built: bool,
    basis: Vec<BasisElt<VertexLabel, EdgeLabel>>,
    basis_index: HashMap<BasisElt<VertexLabel, EdgeLabel>, IndexInList>,
    basis_paths_only: Vec<Vec<EdgeLabel>>,
    basis_paths_index: HashMap<Vec<EdgeLabel>, IndexInList>,
    chains_built: bool,
    ap: HashMap<CohomologicalDegree, Vec<Chain<EdgeLabel>>>,
    cochain_basis_cache: HashMap<CohomologicalDegree, Vec<CochainBasis>>,
    differential_cache: HashMap<CohomologicalDegree, Vec<Vec<Scalar>>>,
    _marker: PhantomData<Scalar>,
}

impl<VertexLabel, EdgeLabel, Scalar> MonomialQuiverAlgebraHH<VertexLabel, EdgeLabel, Scalar>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: std::hash::Hash + Eq + Clone,
    Scalar: Field,
{
    /// Create the structure for computing Hochschild cohomology of a monomial algebra.
    ///
    /// `chain_search_max_wordlen` bounds the word length used when searching for Bardzell chains;
    /// it should be at least as long as the longest non-zero path in the algebra.
    pub fn new(
        quiver_with_mon_relations: Arc<QuiverWithMonomialRelations<VertexLabel, EdgeLabel>>,
        chain_search_max_wordlen: usize,
    ) -> Self {
        Self {
            quiver_with_mon_relations,
            chain_search_max_wordlen,
            basis_built: false,
            basis: Vec::new(),
            basis_index: HashMap::new(),
            basis_paths_only: Vec::new(),
            basis_paths_index: HashMap::new(),
            chains_built: false,
            ap: HashMap::new(),
            cochain_basis_cache: HashMap::new(),
            differential_cache: HashMap::new(),
            _marker: PhantomData,
        }
    }

    /// Create the structure to compute Hochschild Cohomology
    /// of a quiver with monomial relations where the ring
    /// of coefficients is a field.
    /// In this way a general quiver with relations `I = <k*a1..an, l*b1...bm>`
    /// can be converted into a quiver with monomial relations with just
    /// `a1..an` and `b1...bm` as generators of the ideal because the nonzero coefficients
    /// were invertible.
    ///
    /// # Errors
    ///
    /// The relations must have been all monomials and not the idempotents
    pub fn try_from_quiver_with_relations<RelCoeffs>(
        algebra: &QuiverWithRelations<VertexLabel, EdgeLabel, RelCoeffs>,
        chain_search_max_wordlen: usize,
    ) -> Result<Self, HochschildError>
    where
        RelCoeffs: Field,
    {
        let quiver_with_mon_rels = algebra
            .try_into()
            .map_err(HochschildError::NonMonomialIdeal)?;
        let quiver_with_mon_rels = Arc::new(quiver_with_mon_rels);
        Ok(Self::new(quiver_with_mon_rels, chain_search_max_wordlen))
    }

    #[must_use = "The basis is returned by reference; ignoring it means the build step ran for nothing"]
    pub fn view_basis(&self) -> &[BasisElt<VertexLabel, EdgeLabel>] {
        &self.basis
    }

    /// Return the Bardzell chains in cohomological degree `degree`, or `None` if chains have not
    /// been built yet for that degree.
    pub fn chains_in_degree(&self, degree: usize) -> Option<&[Chain<EdgeLabel>]> {
        self.ap.get(&degree).map(Vec::as_slice)
    }

    /// Enumerate all admissible paths (basis elements of A = kQ^{op}/I^{op}) up to `max_paths`.
    ///
    /// Idempotents are included automatically. The result is cached; subsequent calls are no-ops.
    ///
    /// # Errors
    ///
    /// Returns `Err(HochschildError::BasisEnumerationExceeded)` if the number of admissible paths
    /// exceeds `max_paths`.
    pub fn build_basis(&mut self, max_paths: usize) -> Result<(), HochschildError> {
        if self.basis_built {
            return Ok(());
        }

        let all_edges = self
            .quiver_with_mon_relations
            .edge_labels()
            .collect::<Vec<_>>();
        let mut admissible_paths: Vec<Vec<_>> = Vec::new();
        let mut seen: HashSet<Vec<_>> = HashSet::new();
        let mut frontier: Vec<Vec<_>> = Vec::new();

        for a in &all_edges {
            let word = vec![a.clone()];
            if !self.quiver_with_mon_relations.is_zero_path_word(&word) {
                seen.insert(word.clone());
                admissible_paths.push(word.clone());
                frontier.push(word);
            }
        }

        while !frontier.is_empty() {
            let mut new_frontier = Vec::new();
            for word in frontier {
                let Some((_, last_tgt)) = self.quiver_with_mon_relations.path_source_target(&word)
                else {
                    continue;
                };
                for a in &all_edges {
                    let Some((src, _)) = self.quiver_with_mon_relations.edge_endpoint_labels(a)
                    else {
                        continue;
                    };
                    if src != last_tgt {
                        continue;
                    }
                    let mut extended = word.clone();
                    extended.push(a.clone());
                    if seen.contains(&extended) {
                        continue;
                    }
                    if !self.quiver_with_mon_relations.is_zero_path_word(&extended) {
                        if admissible_paths.len() >= max_paths {
                            return Err(HochschildError::BasisEnumerationExceeded { max_paths });
                        }
                        seen.insert(extended.clone());
                        admissible_paths.push(extended.clone());
                        new_frontier.push(extended);
                    }
                }
            }
            frontier = new_frontier;
        }

        let mut basis = self
            .quiver_with_mon_relations
            .vertices()
            .map(BasisElt::Idempotent)
            .collect::<Vec<_>>();
        basis.extend(admissible_paths.iter().filter_map(|w| BasisElt::create(w)));

        self.basis_index = basis
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, b)| (b, i))
            .collect();
        self.basis_paths_index = admissible_paths
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, w)| (w, i))
            .collect();
        self.basis_paths_only = admissible_paths;
        self.basis = basis;
        self.basis_built = true;
        Ok(())
    }

    pub fn build_chains(&mut self) {
        if self.chains_built {
            return;
        }

        self.ap.insert(0, Vec::new());
        self.ap.insert(
            1,
            self.quiver_with_mon_relations
                .edge_labels()
                .map(|a| Chain {
                    word: vec![a],
                    rels: Vec::new(),
                })
                .collect(),
        );
        self.ap.insert(
            2,
            self.quiver_with_mon_relations
                .relations()
                .cloned()
                .map(|r| Chain {
                    word: r.clone(),
                    rels: vec![(0, r)],
                })
                .collect(),
        );

        let mut degree = 2;
        loop {
            let previous = self.ap.get(&degree).cloned().unwrap_or_default();
            let mut next = Vec::new();
            for chain in &previous {
                next.extend(self.extend_chain(chain));
            }
            let mut seen = HashSet::new();
            next.retain(|chain| seen.insert(chain.clone()));
            if next.is_empty() {
                break;
            }
            self.ap.insert(degree + 1, next);
            degree += 1;
        }

        self.chains_built = true;
    }

    fn extend_chain(&self, chain: &Chain<EdgeLabel>) -> Vec<Chain<EdgeLabel>> {
        let mut out = Vec::new();
        for relation in self.quiver_with_mon_relations.relations() {
            for overlap in 1..relation.len() {
                if chain.word.len() < overlap {
                    continue;
                }
                if chain.word[chain.word.len() - overlap..] != relation[..overlap] {
                    continue;
                }
                let mut new_word = chain.word.clone();
                new_word.extend_from_slice(&relation[overlap..]);
                if new_word.len() > self.chain_search_max_wordlen {
                    continue;
                }
                let start = chain.word.len() - overlap;
                let mut rels = chain.rels.clone();
                if rels.last().is_some_and(|(s, _)| start <= *s) {
                    continue;
                }
                rels.push((start, relation.clone()));
                if Self::is_valid_chain_history(&new_word, &rels) {
                    out.push(Chain {
                        word: new_word,
                        rels,
                    });
                }
            }
        }
        out
    }

    fn is_valid_chain_history(word: &[EdgeLabel], rels: &[(usize, Vec<EdgeLabel>)]) -> bool {
        for (i, (start, relation)) in rels.iter().enumerate() {
            if *start + relation.len() > word.len() {
                return false;
            }
            if word[*start..(*start + relation.len())] != relation[..] {
                return false;
            }
            if i > 0 {
                let (previous_start, previous_relation) = &rels[i - 1];
                if previous_start >= start {
                    return false;
                }
                if *start >= *previous_start + previous_relation.len() {
                    return false;
                }
            }
        }
        true
    }

    fn find_chain_in_ap(&self, degree: usize, candidate: &Chain<EdgeLabel>) -> Option<usize> {
        self.ap
            .get(&degree)
            .and_then(|chains| chains.iter().position(|chain| chain == candidate))
    }

    #[must_use = "The subchain decomposition is the main output; discarding it means this traversal ran for nothing"]
    pub fn subchains(
        &self,
        chain: &Chain<EdgeLabel>,
        degree: usize,
    ) -> Vec<(Vec<EdgeLabel>, Chain<EdgeLabel>, Vec<EdgeLabel>)> {
        if degree <= 1 {
            return Vec::new();
        }

        let rels = chain.rels.clone();
        let mut out = Vec::new();

        let chain_from_rels = |selected: &[(usize, Vec<EdgeLabel>)]| -> Option<Chain<EdgeLabel>> {
            if selected.is_empty() {
                return None;
            }
            let start = selected[0].0;
            let end = selected
                .iter()
                .map(|(s, r)| s + r.len())
                .max()
                .unwrap_or(start);
            let subword = chain.word[start..end].to_vec();
            let shifted = selected
                .iter()
                .map(|(s, r)| (s - start, r.clone()))
                .collect::<Vec<_>>();
            let candidate = Chain {
                word: subword,
                rels: shifted,
            };
            self.find_chain_in_ap(degree - 1, &candidate)
                .map(|_| candidate)
        };

        if degree == 2 {
            for i in 0..chain.word.len() {
                out.push((
                    chain.word[..i].to_vec(),
                    Chain {
                        word: vec![chain.word[i].clone()],
                        rels: Vec::new(),
                    },
                    chain.word[(i + 1)..].to_vec(),
                ));
            }
            return out;
        }

        if degree % 2 == 1 {
            if rels.len() >= 2 {
                if let Some(c1) = chain_from_rels(&rels[1..]) {
                    let start = rels[1].0;
                    let end = rels.last().map_or(start, |(s, r)| s + r.len());
                    out.push((chain.word[..start].to_vec(), c1, chain.word[end..].to_vec()));
                }
                if let Some(c2) = chain_from_rels(&rels[..rels.len() - 1]) {
                    let start = rels[0].0;
                    let end = rels[rels.len() - 2].0 + rels[rels.len() - 2].1.len();
                    out.push((chain.word[..start].to_vec(), c2, chain.word[end..].to_vec()));
                }
            }
        } else {
            for j in 0..rels.len() {
                let mut selected = rels.clone();
                selected.remove(j);
                if selected.is_empty() {
                    continue;
                }
                if let Some(c) = chain_from_rels(&selected) {
                    let start = selected[0].0;
                    let end = selected.last().map_or(start, |(s, r)| s + r.len());
                    out.push((chain.word[..start].to_vec(), c, chain.word[end..].to_vec()));
                }
            }
        }

        let mut seen = HashSet::new();
        out.retain(|item| seen.insert(item.clone()));
        out
    }

    fn parallel_basis_outputs(
        &self,
        degree: usize,
        chain: Option<&Chain<EdgeLabel>>,
        vertex: Option<&VertexLabel>,
    ) -> Vec<usize> {
        #[allow(clippy::single_match_else)]
        let Some((src, tgt)) = (match degree {
            0 => {
                let v = vertex.expect("vertex must be given in degree 0");
                Some((v.clone(), v.clone()))
            }
            _ => {
                let ch = chain.expect("chain must be given in positive degrees");
                self.quiver_with_mon_relations.path_source_target(&ch.word)
            }
        }) else {
            return Vec::new();
        };

        let mut outs = Vec::new();
        for (i, basis) in self.basis.iter().enumerate() {
            #[allow(clippy::collapsible_if)]
            if let Some((b_src, b_tgt)) = self.quiver_with_mon_relations.basis_endpoints(basis) {
                if b_src == src && b_tgt == tgt {
                    outs.push(i);
                }
            }
        }
        outs
    }

    /// # Panics
    ///
    /// The problem of `build_basis`
    pub fn cochain_basis(&mut self, degree: usize) -> Vec<CochainBasis> {
        if let Some(cached) = self.cochain_basis_cache.get(&degree) {
            return cached.clone();
        }

        self.build_basis(100_000)
            .expect("basis enumeration exceeded the default cap inside cochain_basis");
        self.build_chains();

        let mut basis = Vec::new();
        if degree == 0 {
            for (vi, v) in self.quiver_with_mon_relations.vertices().enumerate() {
                for out_idx in self.parallel_basis_outputs(0, None, Some(&v)) {
                    basis.push((CochainAnchor::Vertex(vi), out_idx));
                }
            }
        } else if let Some(chains) = self.ap.get(&degree).cloned() {
            for (ci, chain) in chains.iter().enumerate() {
                for out_idx in self.parallel_basis_outputs(degree, Some(chain), None) {
                    basis.push((CochainAnchor::Chain(ci), out_idx));
                }
            }
        }

        self.cochain_basis_cache.insert(degree, basis.clone());
        basis
    }

    /// Return the dimension of the Hochschild cochain space in degree `degree`.
    pub fn cochain_dim(&mut self, degree: usize) -> usize {
        self.cochain_basis(degree).len()
    }

    /// Compute the matrix of the Hochschild differential d: C^{degree} → C^{degree+1}.
    ///
    /// The result is cached; subsequent calls for the same degree are free. Internally calls
    /// `build_basis` with a cap of 100,000 paths if the basis has not been built yet.
    ///
    /// # Panics
    ///
    /// Panics if `build_basis` fails because the number of admissible paths exceeds 100,000.
    #[allow(clippy::too_many_lines)]
    pub fn differential_matrix(&mut self, degree: CohomologicalDegree) -> Vec<Vec<Scalar>> {
        if let Some(answer) = self.differential_cache.get(&degree) {
            return answer.clone();
        }
        self.build_basis(100_000)
            .expect("basis enumeration exceeded the default cap inside differential_matrix");
        self.build_chains();

        let dom_basis = self.cochain_basis(degree);
        let cod_basis = self.cochain_basis(degree + 1);
        let mut matrix = vec![vec![Scalar::zero(); dom_basis.len()]; cod_basis.len()];
        if cod_basis.is_empty() {
            self.differential_cache.insert(degree, matrix.clone());
            return matrix;
        }

        if degree == 0 {
            for (col, (anchor, out_idx)) in dom_basis.iter().enumerate() {
                let CochainAnchor::Vertex(_) = *anchor else {
                    continue;
                };
                let out = self.basis[*out_idx].clone();
                for (row, (row_anchor, row_out_idx)) in cod_basis.iter().enumerate() {
                    let CochainAnchor::Chain(rci) = *row_anchor else {
                        continue;
                    };
                    let Some(ch) = self.ap.get(&1).and_then(|ap1| ap1.get(rci)) else {
                        continue;
                    };
                    let arrow_basis = BasisElt::Path(NonEmpty::singleton(ch.word[0].clone()));
                    let left = self
                        .quiver_with_mon_relations
                        .multiply_basis_elts(&arrow_basis, &out);
                    let right = self
                        .quiver_with_mon_relations
                        .multiply_basis_elts(&out, &arrow_basis);
                    let target_out = &self.basis[*row_out_idx];
                    let mut coeff = Scalar::zero();
                    if left.as_ref() == Some(target_out) {
                        coeff += Scalar::one();
                    }
                    if right.as_ref() == Some(target_out) {
                        coeff -= Scalar::one();
                    }
                    if !coeff.is_zero() {
                        matrix[row][col] += coeff;
                    }
                }
            }
            self.differential_cache.insert(degree, matrix.clone());
            return matrix;
        }

        for (col, (dom_anchor, out_idx)) in dom_basis.iter().enumerate() {
            let CochainAnchor::Chain(dci) = *dom_anchor else {
                continue;
            };
            let Some(dch) = self
                .ap
                .get(&degree)
                .and_then(|chains| chains.get(dci))
                .cloned()
            else {
                continue;
            };
            let out = self.basis[*out_idx].clone();

            for (row, (cod_anchor, row_out_idx)) in cod_basis.iter().enumerate() {
                let CochainAnchor::Chain(cci) = *cod_anchor else {
                    continue;
                };
                let Some(cch) = self
                    .ap
                    .get(&(degree + 1))
                    .and_then(|chains| chains.get(cci))
                    .cloned()
                else {
                    continue;
                };
                let terms = self.subchains(&cch, degree + 1);
                if terms.is_empty() {
                    continue;
                }
                let target_out = self.basis[*row_out_idx].clone();

                for (k, (left_word, psi, right_word)) in terms.into_iter().enumerate() {
                    if psi != dch {
                        continue;
                    }
                    let sign = if (degree + 1) % 2 == 1 && k % 2 == 1 {
                        -Scalar::one()
                    } else {
                        Scalar::one()
                    };
                    let mut value = out.clone();
                    if !left_word.is_empty() {
                        let left_word =
                            NonEmpty::from_vec(left_word).expect("Know that it is not empty");
                        let Some(next) = self
                            .quiver_with_mon_relations
                            .multiply_basis_elts(&BasisElt::Path(left_word), &value)
                        else {
                            continue;
                        };
                        value = next;
                    }
                    if !right_word.is_empty() {
                        let right_word =
                            NonEmpty::from_vec(right_word).expect("Know that it is not empty");
                        let Some(next) = self
                            .quiver_with_mon_relations
                            .multiply_basis_elts(&value, &BasisElt::Path(right_word))
                        else {
                            continue;
                        };
                        value = next;
                    }
                    if value == target_out {
                        matrix[row][col] += sign;
                    }
                }
            }
        }

        self.differential_cache.insert(degree, matrix.clone());
        matrix
    }

    pub fn maximal_possible_hh_degree(&mut self) -> Option<usize> {
        self.build_chains();
        self.ap
            .iter()
            .filter_map(|(degree, chains)| (!chains.is_empty()).then_some(*degree))
            .max()
    }

    /// Compute the Hochschild cohomology dimensions HH^d(A) for d = 0, …, `max_degree`.
    ///
    /// Builds the basis (up to `max_paths` admissible paths), constructs the Bardzell chains,
    /// and computes rank(d^d) for each degree to get dim HH^d = ker(d^d) - im(d^{d-1}).
    ///
    /// # Errors
    ///
    /// Returns an error if `build_basis` exceeds `max_paths`, or if a cohomology dimension
    /// comes out negative (which would indicate a bug in the differential).
    pub fn hochschild_dimensions(
        &mut self,
        max_degree: CohomologicalDegree,
        max_paths: usize,
    ) -> Result<HashMap<CohomologicalDegree, usize>, HochschildError> {
        self.build_basis(max_paths)?;
        self.build_chains();

        let dims_c = (0..=(max_degree + 1))
            .map(|degree| self.cochain_dim(degree))
            .collect::<Vec<_>>();
        let ranks = (0..=max_degree)
            .map(|degree| super::checked_arith::rank(&self.differential_matrix(degree)))
            .collect::<Vec<_>>();

        let mut hh = HashMap::<CohomologicalDegree, usize>::new();
        #[allow(clippy::cast_possible_wrap)]
        for d in 0..=max_degree {
            let ker = dims_c[d] as isize - ranks[d] as isize;
            let im = if d > 0 { ranks[d - 1] as isize } else { 0 };
            let dim = ker - im;
            if dim < 0 {
                return Err(HochschildError::NegativeCohomologyDimension {
                    degree: d,
                    value: dim,
                });
            }
            #[allow(clippy::cast_sign_loss)]
            hh.insert(d, dim as usize);
        }
        Ok(hh)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::quiver_algebra::{
        checked_arith::Field, quiver::Quiver, quiver_with_mon_rels::QuiverWithMonomialRelations,
    };

    use super::MonomialQuiverAlgebraHH;

    impl Field for f64 {
        fn zero() -> Self {
            0.0
        }
        fn one() -> Self {
            1.0
        }
        fn inv(self) -> Self {
            1.0 / self
        }
    }

    fn hh_dims(
        hh: &mut MonomialQuiverAlgebraHH<&'static str, &'static str, f64>,
        max_degree: usize,
    ) -> Vec<usize> {
        let dims = hh
            .hochschild_dimensions(max_degree, 10_000)
            .expect("HH computation failed");
        (0..=max_degree).map(|d| dims[&d]).collect()
    }

    fn no_rels(
        q: Quiver<&'static str, &'static str>,
    ) -> MonomialQuiverAlgebraHH<&'static str, &'static str, f64> {
        let q = Arc::new(QuiverWithMonomialRelations::new(Arc::new(q), []));
        MonomialQuiverAlgebraHH::new(q, 20)
    }

    // ── No ideal quotient ──────────────────────────────────────────────────────

    // kQ = k (single vertex, no arrows). HH^n = 0 for n > 0, HH^0 = k.
    #[test]
    fn hh_a1_no_relations() {
        let mut q = Quiver::new();
        q.add_vertex("0");
        let mut hh = no_rels(q);
        let dims = hh_dims(&mut hh, 3);
        assert_eq!(dims[0], 1, "HH^0(k) = k");
        assert_eq!(dims[1], 0);
        assert_eq!(dims[2], 0);
        assert_eq!(dims[3], 0);
    }

    // kA_2: 2 vertices, 1 arrow. Hereditary acyclic → HH^n = 0 for n >= 1.
    // Center = k (connected), so HH^0 = k.
    #[test]
    fn hh_a2_no_relations() {
        let mut q = Quiver::new();
        q.add_vertex("0");
        q.add_vertex("1");
        q.add_edge("0", "1", "a");
        let mut hh = no_rels(q);
        let dims = hh_dims(&mut hh, 3);
        assert_eq!(
            dims[0], 1,
            "HH^0(kA_2) = k, center of connected path algebra"
        );
        assert_eq!(dims[1], 0, "kA_2 hereditary + acyclic → HH^1 = 0");
        assert_eq!(dims[2], 0);
        assert_eq!(dims[3], 0);
    }

    // Kronecker quiver: 2 vertices, 2 parallel arrows a,b: alpha -> beta.
    // Hereditary so HH^n = 0 for n >= 2.
    // HH^0 = k (center, connected). HH^1 = k^3 (3 outer derivations).
    //
    // Derivation proof: D(a), D(b) ∈ span{a,b} = k^2 each, so Der = k^4.
    // InnDer = span{ ad_{e_alpha} : D(a)=a, D(b)=b } = k^1.
    // OutDer = k^4 / k^1 = k^3.
    #[test]
    fn hh_kronecker_no_relations() {
        let kq = crate::quiver_algebra::quiver::tests::make_kronecker_quiver();
        let mut hh = no_rels(kq);
        let dims = hh_dims(&mut hh, 3);
        assert_eq!(dims[0], 1, "HH^0(Kronecker) = k");
        assert_eq!(dims[1], 3, "HH^1(Kronecker) = k^3, three outer derivations");
        assert_eq!(dims[2], 0, "Kronecker algebra is hereditary, HH^2 = 0");
        assert_eq!(dims[3], 0);
    }

    // ── Monomial ideal quotients ───────────────────────────────────────────────

    // A_3/<ab>: vertices 0,1,2; arrows a: 0→1, b: 1→2; relation ab=0.
    // Basis of A: {e_0, e_1, e_2, a, b} (ab=0 kills the only length-2 path).
    //
    // Bardzell chains: degree 1 → {[a],[b]}, degree 2 → {[a,b]}, degree 3 → none
    // (the overlap test fails: word [a,b] ends in [b] ≠ [a] = start of relation).
    //
    // Cochain dims:
    //   C^0 = 3  (one idempotent per vertex)
    //   C^1 = 2  (arrow [a] || a, arrow [b] || b — one parallel path each)
    //   C^2 = 0  (relation chain [a,b] goes 0→2, but e_0·A·e_2 = 0 since ab=0)
    //
    // d^0 matrix (rows=C^1, cols=C^0):
    //         e_0  e_1  e_2
    //   [a]:  -1   +1    0
    //   [b]:   0   -1   +1
    // rank = 2, kernel = span{(1,1,1)} → HH^0 = 1.
    // HH^1 = C^1 / im(d^0) = k^2 / k^2 = 0.
    // HH^n = 0 for n >= 2 (C^2 = 0 and no higher chains).
    #[test]
    fn hh_a3_relation_ab() {
        let mut q = Quiver::new();
        q.add_vertex("0");
        q.add_vertex("1");
        q.add_vertex("2");
        q.add_edge("0", "1", "a");
        q.add_edge("1", "2", "b");
        let q = Arc::new(QuiverWithMonomialRelations::new(
            Arc::new(q),
            [vec!["a", "b"]],
        ));
        let mut hh: MonomialQuiverAlgebraHH<_, _, f64> = MonomialQuiverAlgebraHH::new(q, 20);

        // Verify cochain dimensions reflect the structure above.
        assert_eq!(hh.cochain_dim(0), 3);
        assert_eq!(hh.cochain_dim(1), 2);
        assert_eq!(hh.cochain_dim(2), 0);

        let dims = hh_dims(&mut hh, 3);
        assert_eq!(dims[0], 1, "HH^0 = k, center of connected algebra");
        assert_eq!(dims[1], 0, "d^0 surjects onto C^1, so HH^1 = 0");
        assert_eq!(dims[2], 0, "C^2 = 0 since e_0·A·e_2 = 0");
        assert_eq!(dims[3], 0);
    }
}
