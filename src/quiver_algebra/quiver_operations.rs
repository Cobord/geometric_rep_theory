use crate::quiver_algebra::Quiver;

impl<VertexLabel, EdgeLabel> Quiver<VertexLabel, EdgeLabel>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: std::hash::Hash + Eq + Clone,
{
    #[allow(clippy::type_complexity, clippy::missing_panics_doc)]
    pub fn repeat(self, times: usize) -> Quiver<(VertexLabel, usize), (EdgeLabel, usize)> {
        if times == 0 {
            return Quiver::new();
        }
        if times == 1 {
            return self.map_labels(|v| (v, 0), |e| (e, 0));
        }
        let mut to_return = Quiver::new();
        for v in self.vertex_labels() {
            for idx in 0..times {
                to_return.add_vertex((v.clone(), idx));
            }
        }
        for e_label in self.edge_labels() {
            let (source, target) = self
                .edge_endpoint_labels(e_label)
                .expect("This is an arrow of the quiver");
            for idx in 0..times {
                to_return.add_edge(
                    (source.clone(), idx),
                    (target.clone(), idx),
                    (e_label.clone(), idx),
                );
            }
        }
        to_return
    }

    /// We cannot really construct the full translation quiver of a given
    /// quiver because of infiniteness of Z
    /// but we can construct a finite quotient of it by specifying a periodicity.
    ///
    /// Instead of the vertices being `Z \times Q_0` we have `Z/(periodicity+1)Z \times Q_0`
    /// and we have edges from `(v, idx)` to `(w, idx)` for each edge from `v` to `w` in the original quiver
    /// and also edges from `(w, idx)` to `(v, idx+1)` for each edge from `v` to `w` in the original quiver.
    /// All indices are taken mod `periodicity+1`.
    #[allow(clippy::missing_panics_doc, clippy::type_complexity)]
    pub fn translation_quiver(
        self,
        periodicity: usize,
    ) -> Quiver<(VertexLabel, usize), Result<(EdgeLabel, usize), (EdgeLabel, usize)>> {
        if periodicity == 0 {
            let to_return = self.map_labels(|v| (v, 0), |e| Ok((e, 0)));
            let (to_return, _pairs) = to_return.double(|e| {
                debug_assert!(matches!(e, Ok((_, 0))));
                Err((e.clone().map_err(|_| ()).expect("Made only Ok values").0, 0))
            });
            return to_return;
        }
        let mut to_return = Quiver::new();
        for v in self.vertex_labels() {
            for idx in 0..=periodicity {
                to_return.add_vertex((v.clone(), idx));
            }
        }
        for e_label in self.edge_labels() {
            let (source, target) = self
                .edge_endpoint_labels(e_label)
                .expect("This is an arrow of the quiver");
            for idx in 0..=periodicity {
                to_return.add_edge(
                    (source.clone(), idx),
                    (target.clone(), idx),
                    Ok((e_label.clone(), idx)),
                );
            }
            for idx in 0..=periodicity {
                let idx_next = if idx == periodicity { 0 } else { idx + 1 };
                to_return.add_edge(
                    (target.clone(), idx),
                    (source.clone(), idx_next),
                    Err((e_label.clone(), idx)),
                );
            }
        }
        to_return
    }

    /// Construct a new quiver from two quivers and a function that
    /// specifies whether to flip the first quiver's edges when adding edges corresponding
    /// to edges that are connecting different vertices in the second quiver.
    ///
    /// The vertices of the new quiver are pairs of vertices of the original quivers.
    /// The edge labels of the new quiver are either `Ok((e1_label, v2))`
    /// for edges coming from the first quiver and having fixed vertex in the second quiver
    /// Or `Err((e1_label, e2_label))` for edges coming from edges in both quivers.
    /// If `flip_fun(e2_label)` is false, then this connects
    /// `(source1, source2)` to `(target1, target2)` like in `tensor_product`.
    /// If `flip_fun(e2_label)` is true, then this connects
    /// `(target1, source2)` to `(source1, target2)`, flipping the edge from the first quiver.
    /// like the arrows connecting `idx` and `idx+1` in `translation_quiver`.
    #[allow(
        clippy::needless_pass_by_value,
        clippy::type_complexity,
        clippy::missing_panics_doc
    )]
    pub fn family<VertexLabel2, EdgeLabel2>(
        self,
        other: Quiver<VertexLabel2, EdgeLabel2>,
        mut flip_fun: impl FnMut(EdgeLabel2) -> bool,
    ) -> Quiver<
        (VertexLabel, VertexLabel2),
        Result<(EdgeLabel, VertexLabel2), (EdgeLabel, EdgeLabel2)>,
    >
    where
        VertexLabel2: std::hash::Hash + Eq + Clone,
        EdgeLabel2: std::hash::Hash + Eq + Clone,
    {
        let mut to_return = Quiver::new();
        for v1 in self.vertex_labels() {
            for v2 in other.vertex_labels() {
                to_return.add_vertex((v1.clone(), v2.clone()));
            }
        }
        for e_label in self.edge_labels() {
            let (source, target) = self
                .edge_endpoint_labels(e_label)
                .expect("This is an arrow of the quiver");
            for v2 in other.vertex_labels() {
                to_return.add_edge(
                    (source.clone(), v2.clone()),
                    (target.clone(), v2.clone()),
                    Ok((e_label.clone(), v2.clone())),
                );
            }
        }
        for e2_label in other.edge_labels() {
            let (source2, target2) = other
                .edge_endpoint_labels(e2_label)
                .expect("This is an arrow of the quiver");
            let flip = flip_fun(e2_label.clone());
            for e1_label in self.edge_labels() {
                let (source1, target1) = self
                    .edge_endpoint_labels(e1_label)
                    .expect("This is an arrow of the quiver");
                if flip {
                    to_return.add_edge(
                        (target1.clone(), source2.clone()),
                        (source1.clone(), target2.clone()),
                        Err((e1_label.clone(), e2_label.clone())),
                    );
                } else {
                    to_return.add_edge(
                        (source1.clone(), source2.clone()),
                        (target1.clone(), target2.clone()),
                        Err((e1_label.clone(), e2_label.clone())),
                    );
                }
            }
        }
        to_return
    }

    /// Take the box product of two quivers,
    /// which is the quiver whose vertices are pairs of vertices of the original quivers,
    /// and which has an edge from `(v1, v2)` to `(w1, w2)` if and only if either
    /// - there is an edge from `v1` to `w1` in the first quiver and `v2 = w2`, or
    /// - there is an edge from `v2` to `w2` in the second quiver and `v1 = w1`.
    ///
    /// The edge labels are pairs of the original edge label
    /// and the vertex label of the other quiver that is being held fixed.
    #[allow(
        clippy::missing_panics_doc,
        clippy::needless_pass_by_value,
        clippy::type_complexity
    )]
    pub fn box_product<VertexLabel2, EdgeLabel2>(
        self,
        other: Quiver<VertexLabel2, EdgeLabel2>,
    ) -> Quiver<
        (VertexLabel, VertexLabel2),
        Result<(EdgeLabel, VertexLabel2), (EdgeLabel2, VertexLabel)>,
    >
    where
        VertexLabel2: std::hash::Hash + Eq + Clone,
        EdgeLabel2: std::hash::Hash + Eq + Clone,
    {
        let mut to_return = Quiver::new();
        for v1 in self.vertex_labels() {
            for v2 in other.vertex_labels() {
                to_return.add_vertex((v1.clone(), v2.clone()));
            }
        }
        for e_label in self.edge_labels() {
            let (source, target) = self
                .edge_endpoint_labels(e_label)
                .expect("This is an arrow of the quiver");
            for v2 in other.vertex_labels() {
                to_return.add_edge(
                    (source.clone(), v2.clone()),
                    (target.clone(), v2.clone()),
                    Ok((e_label.clone(), v2.clone())),
                );
            }
        }
        for e_label in other.edge_labels() {
            let (source, target) = other
                .edge_endpoint_labels(e_label)
                .expect("This is an arrow of the quiver");
            for v1 in self.vertex_labels() {
                to_return.add_edge(
                    (v1.clone(), source.clone()),
                    (v1.clone(), target.clone()),
                    Err((e_label.clone(), v1.clone())),
                );
            }
        }
        to_return
    }

    /// Take the tensor product of two quivers,
    /// which is the quiver whose vertices are pairs of vertices of the original quivers,
    /// and which has an edge from `(v1, v2)` to `(w1, w2)` if and only if
    /// there is an edge from `v1` to `w1` in the first quiver and there is an edge from `v2` to `w2` in the second quiver.
    ///
    /// The edge labels are pairs of the original edge labels.
    #[allow(clippy::missing_panics_doc, clippy::needless_pass_by_value)]
    pub fn tensor_product<VertexLabel2, EdgeLabel2>(
        self,
        other: Quiver<VertexLabel2, EdgeLabel2>,
    ) -> Quiver<(VertexLabel, VertexLabel2), (EdgeLabel, EdgeLabel2)>
    where
        VertexLabel2: std::hash::Hash + Eq + Clone,
        EdgeLabel2: std::hash::Hash + Eq + Clone,
    {
        let mut to_return = Quiver::new();
        for v1 in self.vertex_labels() {
            for v2 in other.vertex_labels() {
                to_return.add_vertex((v1.clone(), v2.clone()));
            }
        }
        for e1_label in self.edge_labels() {
            let (source, target) = self
                .edge_endpoint_labels(e1_label)
                .expect("This is an arrow of the quiver");
            for e2_label in other.edge_labels() {
                let (source2, target2) = other
                    .edge_endpoint_labels(e2_label)
                    .expect("This is an arrow of the quiver");
                to_return.add_edge(
                    (source.clone(), source2.clone()),
                    (target.clone(), target2.clone()),
                    (e1_label.clone(), e2_label.clone()),
                );
            }
        }
        to_return
    }

    /// Take the opposite quiver, which is the quiver obtained by reversing all edges.
    /// The edge labels are left unchanged, but the source and target of each edge are swapped.
    pub fn opposite(mut self) -> Self {
        for e_label in self.edge_labels().cloned().collect::<Vec<_>>() {
            self.flip_edge(e_label);
        }
        self
    }

    #[allow(clippy::needless_pass_by_value)]
    /// Take the disjoint union of two quivers, asserting that their vertex and edge labels are disjoint.
    ///
    /// # Panics
    /// Panics if the vertex labels of the two quivers are not disjoint,
    /// or if the edge labels of the two quivers are not disjoint.
    pub fn disjoint_union(mut self, other: Self) -> Self {
        for v in other.vertex_labels() {
            assert!(
                !self.contains_vertex(v),
                "Cannot take disjoint union of quivers with overlapping vertex labels.",
            );
            self.add_vertex(v.clone());
        }
        for e_label in other.edge_labels() {
            assert!(
                !self.contains_edge(e_label),
                "Cannot take disjoint union of quivers with overlapping edge labels.",
            );
            let (source, target) = other
                .edge_endpoint_labels(e_label)
                .expect("This is an arrow of the quiver");
            self.add_edge(source, target, e_label.clone());
        }
        self
    }
}
