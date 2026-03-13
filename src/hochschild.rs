use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
    sync::Arc,
};

use nonempty::NonEmpty;

use crate::{
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

    #[must_use = "TODO"]
    pub fn view_basis(&self) -> &[BasisElt<VertexLabel, EdgeLabel>] {
        &self.basis
    }

    pub fn chains_in_degree(&self, degree: usize) -> Option<&[Chain<EdgeLabel>]> {
        self.ap.get(&degree).map(Vec::as_slice)
    }

    ///
    ///
    /// # Errors
    ///
    /// There could be too many admissible paths beyond the limit set by `max_paths`
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

    #[must_use = "TODO"]
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

    pub fn cochain_dim(&mut self, degree: usize) -> usize {
        self.cochain_basis(degree).len()
    }

    ///
    ///
    /// # Panics
    ///
    /// The problem of `build_basis`
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

    ///
    ///
    /// # Errors
    ///
    /// There could be too many admissible paths as in the error of `build_basis`.
    /// The other problem of seeing a kernel that is smaller than the image
    /// is a problem of d^2 not being 0, which indicates the differential was wrong.
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
            .map(|degree| crate::checked_arith::rank(&self.differential_matrix(degree)))
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
