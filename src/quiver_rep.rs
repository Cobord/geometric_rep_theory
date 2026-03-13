use std::ops::MulAssign;
use std::{collections::HashMap, sync::Arc};

use nonempty::NonEmpty;

use crate::checked_arith::{
    CheckedAdd, CheckedAddAssign, CheckedArithError, CheckedMul, CheckedMulAssign, Ring,
};
use crate::quiver::{BasisElt, PathAlgebra, Quiver};

#[must_use]
pub struct QuiverRep<VertexLabel, EdgeLabel, MatrixType>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
    MatrixType: CheckedAdd + CheckedAddAssign + CheckedMul + CheckedMulAssign + Clone,
{
    quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
    edge_reps: HashMap<EdgeLabel, MatrixType>,
    vertex_reps: HashMap<VertexLabel, MatrixType>,
}

impl<VertexLabel, EdgeLabel, MatrixType> QuiverRep<VertexLabel, EdgeLabel, MatrixType>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone,
    MatrixType: CheckedAdd + CheckedAddAssign + CheckedMul + CheckedMulAssign + Clone,
{
    ///
    ///
    /// # Errors
    ///
    /// If we have not given the matrix associated to every edge, then this is
    /// not enough information for a quiver representation
    #[allow(clippy::type_complexity)]
    pub fn new(
        quiver: Arc<Quiver<VertexLabel, EdgeLabel>>,
        mut edge_reps: HashMap<EdgeLabel, MatrixType>,
        mut vertex_reps: HashMap<VertexLabel, MatrixType>,
    ) -> Result<
        Self,
        (
            HashMap<VertexLabel, MatrixType>,
            HashMap<EdgeLabel, MatrixType>,
        ),
    > {
        edge_reps.retain(|key, _| quiver.contains_edge(key));
        vertex_reps.retain(|key, _| quiver.contains_vertex(key));
        for edge_label in quiver.edge_labels() {
            if !edge_reps.contains_key(edge_label) {
                return Err((vertex_reps, edge_reps));
            }
        }
        for vertex_label in quiver.vertex_labels() {
            if !vertex_reps.contains_key(vertex_label) {
                return Err((vertex_reps, edge_reps));
            }
        }
        Ok(Self {
            quiver,
            edge_reps,
            vertex_reps,
        })
    }

    pub fn set_edge_rep(&mut self, edge: &EdgeLabel, rep: MatrixType) {
        if let Some(v) = self.edge_reps.get_mut(edge) {
            *v = rep;
        }
    }

    pub fn get_edge_rep(&self, edge: &EdgeLabel) -> Option<&MatrixType> {
        self.edge_reps.get(edge)
    }

    #[allow(clippy::must_use_candidate)]
    pub fn quiver(&self) -> &Arc<Quiver<VertexLabel, EdgeLabel>> {
        &self.quiver
    }

    ///
    ///
    /// # Errors
    ///
    /// There may be matrix dimension errors if it is a product of factors
    /// for each arrow on the path
    ///
    /// # Panics
    ///
    /// Everything on the path is actually an arrow of the quiver.
    /// Or it is just a vertex and that vertex is actually a vertex of the quiver.
    pub fn mat_from_path_or_vertex(
        &self,
        path: BasisElt<VertexLabel, EdgeLabel>,
    ) -> Result<MatrixType, <MatrixType as CheckedMulAssign>::MultiplicationError> {
        match path {
            BasisElt::Path(path) => self.mat_from_path(path),
            BasisElt::Idempotent(vertex) => Ok(self
                .vertex_reps
                .get(&vertex)
                .expect("This is a vertex of the quiver")
                .clone()),
        }
    }

    ///
    ///
    /// # Errors
    ///
    /// There may be matrix dimension errors
    ///
    /// # Panics
    ///
    /// Everything on the path is actually an arrow of the quiver.
    pub fn mat_from_path(
        &self,
        path: NonEmpty<EdgeLabel>,
    ) -> Result<MatrixType, <MatrixType as CheckedMulAssign>::MultiplicationError> {
        let first = path.first();
        let mut mat_returned = self
            .edge_reps
            .get(first)
            .expect("Everything on this path was an arrow of this quiver")
            .clone();
        for cur_edge in path {
            let rhs = self
                .edge_reps
                .get(&cur_edge)
                .expect("Everything on this path was an arrow of this quiver")
                .clone();
            mat_returned.checked_mul_assign(rhs)?;
        }
        Ok(mat_returned)
    }

    ///
    ///
    /// # Errors
    ///
    /// There may be matrix dimension errors
    ///
    /// # Panics
    ///
    /// The path algebra element should all have the same endpoints
    /// so that it defines an element of `End(V_s, V_t)` for consistent s and t
    /// source and target nodes.
    pub fn mat_from_path_algebra<Coeffs>(
        &self,
        path_algebra: PathAlgebra<VertexLabel, EdgeLabel, Coeffs>,
    ) -> Result<MatrixType, CheckedArithError<MatrixType>>
    where
        Coeffs: Ring,
        MatrixType: MulAssign<Coeffs>,
    {
        assert!(path_algebra.all_parallel().is_ok());
        let mut mat_returned: Option<MatrixType> = None;
        for (path, coeff) in path_algebra {
            let mut mat_now = self
                .mat_from_path_or_vertex(path)
                .map_err(CheckedArithError::from_mul_assign)?;
            mat_now *= coeff;
            if let Some(mat_returned) = &mut mat_returned {
                mat_returned
                    .checked_add_assign(mat_now)
                    .map_err(CheckedArithError::from_add_assign)?;
            } else {
                mat_returned = Some(mat_now);
            }
        }
        Ok(mat_returned.expect("It has been set now"))
    }
}
