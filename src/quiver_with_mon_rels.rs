use std::{collections::HashSet, fmt::Debug, sync::Arc};

use nonempty::NonEmpty;

use crate::{
    checked_arith::Field,
    hochschild::IndexInList,
    quiver::{BasisElt, Quiver},
    quiver_with_rels::QuiverWithRelations,
};

#[must_use]
pub struct QuiverWithMonomialRelations<VertexLabel, EdgeLabel>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: std::hash::Hash + Eq + Clone,
{
    quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
    relations: Vec<Vec<EdgeLabel>>,
}

impl<VertexLabel, EdgeLabel> QuiverWithMonomialRelations<VertexLabel, EdgeLabel>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: std::hash::Hash + Eq + Clone,
{
    pub fn new(
        quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
        ideal_generators: impl IntoIterator<Item = Vec<EdgeLabel>>,
    ) -> Self {
        let relations = Self::minimal_antichain(ideal_generators);
        Self { quiver, relations }
    }

    #[allow(dead_code)]
    pub(crate) fn quiver_arc(&self) -> &Arc<Quiver<VertexLabel, EdgeLabel>> {
        &self.quiver
    }

    pub fn vertices(&self) -> impl Iterator<Item = VertexLabel> {
        self.quiver.vertex_labels().cloned()
    }

    pub fn edge_labels(&self) -> impl Iterator<Item = EdgeLabel> {
        self.quiver.edge_labels().cloned()
    }

    pub fn relations(&self) -> impl Iterator<Item = &Vec<EdgeLabel>> {
        self.relations.iter()
    }

    pub fn basis_endpoints(
        &self,
        b: &BasisElt<VertexLabel, EdgeLabel>,
    ) -> Option<(VertexLabel, VertexLabel)> {
        self.quiver.basis_endpoints(b)
    }

    fn minimal_antichain(gens: impl IntoIterator<Item = Vec<EdgeLabel>>) -> Vec<Vec<EdgeLabel>> {
        let mut dedup = Vec::new();
        let mut seen = HashSet::new();
        for g in gens {
            if seen.insert(g.clone()) {
                dedup.push(g);
            }
        }
        dedup
            .into_iter()
            .filter(|g| {
                !seen
                    .iter()
                    .any(|h| h.len() < g.len() && crate::utils::contains_subword(g, h))
            })
            .collect()
    }

    pub(crate) fn path_source_target(
        &self,
        word: &[EdgeLabel],
    ) -> Option<(VertexLabel, VertexLabel)> {
        self.quiver.path_source_target_labels(word)
    }

    pub(crate) fn edge_endpoint_labels(&self, a: &EdgeLabel) -> Option<(VertexLabel, VertexLabel)> {
        self.quiver.edge_endpoint_labels(a)
    }

    pub(crate) fn is_zero_path_word(&self, word: &[EdgeLabel]) -> bool {
        if !self.quiver.is_composable_path(word) {
            return true;
        }
        self.relations
            .iter()
            .any(|relation| crate::utils::contains_subword(word, relation))
    }

    fn is_zero_path_nonempty(&self, word: &NonEmpty<EdgeLabel>) -> bool {
        if !self.quiver.is_composable_arrow_word(word) {
            return true;
        }
        self.relations
            .iter()
            .any(|relation| crate::utils::contains_subword_v2(word, relation))
    }

    pub fn retain_if_nonzero(
        &self,
        candidate: BasisElt<VertexLabel, EdgeLabel>,
    ) -> Option<BasisElt<VertexLabel, EdgeLabel>> {
        match candidate {
            BasisElt::Idempotent(v) => Some(BasisElt::Idempotent(v)),
            BasisElt::Path(non_empty) => {
                if self.is_zero_path_nonempty(&non_empty) {
                    None
                } else {
                    Some(BasisElt::Path(non_empty))
                }
            }
        }
    }

    pub fn multiply_basis_elts(
        &self,
        left: &BasisElt<VertexLabel, EdgeLabel>,
        right: &BasisElt<VertexLabel, EdgeLabel>,
    ) -> Option<BasisElt<VertexLabel, EdgeLabel>> {
        let left_was_idempotent = matches!(left, BasisElt::Idempotent(_));
        let right_was_idempotent = matches!(right, BasisElt::Idempotent(_));
        let product_without_relations: BasisElt<_, _> = self.quiver.multiply_basis(left, right)?;
        if !left_was_idempotent && !right_was_idempotent {
            self.retain_if_nonzero(product_without_relations)
        } else {
            Some(product_without_relations)
        }
    }

    ///
    ///
    /// # Errors
    ///
    /// If the algebra kQ^op/I^op is not gentle
    /// give the reason why.
    pub fn is_gentle(&self) -> Result<(), String>
    where
        VertexLabel: Debug,
        EdgeLabel: Debug,
    {
        for vertex in self.quiver.vertex_labels() {
            let num_starting_at_vertex = self.quiver.successors(vertex).count();
            let num_ending_at_vertex = self.quiver.predecessors(vertex).count();
            if num_starting_at_vertex > 2 || num_ending_at_vertex > 2 {
                return Err(format!(
                    "Vertex {vertex:?} has more than 2 arrows starting or ending at it"
                ));
            }
        }
        for rel in &self.relations {
            if rel.len() != 2 {
                return Err(format!("Relation {rel:?} is monomial but not of length 2"));
            }
        }
        for edge_label in self.quiver.edge_labels() {
            let count_relations_starting_with_edge = self
                .relations
                .iter()
                .filter(|rel| rel.first() == Some(edge_label))
                .count();
            if count_relations_starting_with_edge > 1 {
                return Err(format!(
                    "There are too many relations of the form {edge_label:?}*? in the ideal"
                ));
            }
            let count_relations_ending_with_edge = self
                .relations
                .iter()
                .filter(|rel| rel.last() == Some(edge_label))
                .count();
            if count_relations_ending_with_edge > 1 {
                return Err(format!(
                    "There are too many relations of the form ?*{edge_label:?} in the ideal"
                ));
            }
        }
        for edge_label in self.quiver.edge_labels() {
            let Some((cur_edge_src, cur_edge_tgt)) = self.quiver.edge_endpoint_labels(edge_label)
            else {
                continue;
            };
            let mut count_edge_next_edge_not_in_ideal = 0;
            for (successor_edge, _) in self.quiver.successors(&cur_edge_tgt) {
                if !self
                    .relations
                    .iter()
                    .any(|rel| rel[0] == *edge_label && rel[1] == successor_edge)
                {
                    count_edge_next_edge_not_in_ideal += 1;
                }
            }
            if count_edge_next_edge_not_in_ideal > 1 {
                return Err(format!(
                    "There are too many composable pairs of the form {edge_label:?}*? that are not in the ideal"
                ));
            }

            let mut count_prev_edge_edge_not_in_ideal = 0;
            for (predecessor_edge, _) in self.quiver.predecessors(&cur_edge_src) {
                if !self
                    .relations
                    .iter()
                    .any(|rel| rel[0] == predecessor_edge && rel[1] == *edge_label)
                {
                    count_prev_edge_edge_not_in_ideal += 1;
                }
            }
            if count_prev_edge_edge_not_in_ideal > 1 {
                return Err(format!(
                    "There are too many composable pairs of the form ?*{edge_label:?} that are not in the ideal"
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NonMonomialIdeal {
    NonMonomialRelation { relation_index: IndexInList },
    IdempotentRelation { relation_index: IndexInList },
}

impl<VertexLabel, EdgeLabel, Coeffs> TryFrom<&QuiverWithRelations<VertexLabel, EdgeLabel, Coeffs>>
    for QuiverWithMonomialRelations<VertexLabel, EdgeLabel>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: std::hash::Hash + Eq + Clone,
    Coeffs: Field,
{
    type Error = NonMonomialIdeal;

    fn try_from(
        algebra: &QuiverWithRelations<VertexLabel, EdgeLabel, Coeffs>,
    ) -> Result<Self, Self::Error> {
        let mut monomial_relations = Vec::new();
        for (relation_index, relation) in algebra.relations().enumerate() {
            if !relation.is_monomial() {
                return Err(NonMonomialIdeal::NonMonomialRelation { relation_index });
            }
            let Some((term, _coeff)) = relation.iter().next() else {
                continue;
            };
            match term {
                BasisElt::Path(path) => monomial_relations.push(path.iter().cloned().collect()),
                BasisElt::Idempotent(_) => {
                    return Err(NonMonomialIdeal::IdempotentRelation { relation_index });
                }
            }
        }
        Ok(Self {
            quiver: algebra.quiver().clone(),
            relations: monomial_relations,
        })
    }
}
