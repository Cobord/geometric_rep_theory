use crate::quiver_algebra::Quiver;

impl<VertexLabel, EdgeLabel> Quiver<VertexLabel, EdgeLabel>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: std::hash::Hash + Eq + Clone,
{
    /// Construct the quiver of type `A_n` with the specified vertex labels and edge labels.
    /// This is a default orientation.
    ///
    /// The edge labels should be provided in the following order
    /// `e_01, e_12, e_23, ..., e_{n-2,n-1}`
    /// The edge `e_ij` goes from vertex `vertex_names[i]` to vertex `vertex_names[j]`
    #[allow(clippy::missing_panics_doc)]
    pub fn make_a_n<const N: usize>(
        vertex_names: &[VertexLabel; N],
        mut edge_label: impl Iterator<Item = EdgeLabel>,
    ) -> Self {
        let mut to_return = Quiver::new();
        for v in vertex_names {
            to_return.add_vertex(v.clone());
        }
        for (idx, v) in vertex_names.iter().enumerate() {
            if idx + 1 < vertex_names.len() {
                to_return.add_edge(
                    v.clone(),
                    vertex_names[idx + 1].clone(),
                    edge_label
                        .next()
                        .expect("Should have provided N-1 edge labels"),
                );
            }
        }
        assert_eq!(
            edge_label.count(),
            0,
            "Should have provided N-1 edge labels"
        );
        to_return
    }

    /// Construct the quiver of type `D_n` with the specified vertex labels and edge labels.
    /// This is a default orientation.
    ///
    /// The edge labels should be provided in the following order
    /// `e_01, e_12, e_23, ..., e_{n-3,n-2}, e_{n-3,n-1}`
    /// The edge `e_ij` goes from vertex `vertex_names[i]` to vertex `vertex_names[j]`
    #[allow(clippy::missing_panics_doc)]
    pub fn make_d_n<const N: usize>(
        vertex_names: &[VertexLabel; N],
        mut edge_label: impl Iterator<Item = EdgeLabel>,
    ) -> Self {
        let mut to_return = Quiver::new();
        for v in vertex_names {
            to_return.add_vertex(v.clone());
        }
        for (idx, v) in vertex_names.iter().enumerate() {
            if idx + 3 < vertex_names.len() {
                to_return.add_edge(
                    v.clone(),
                    vertex_names[idx + 1].clone(),
                    edge_label
                        .next()
                        .expect("Should have provided N-1 edge labels"),
                );
            }
            if idx + 3 == vertex_names.len() {
                to_return.add_edge(
                    v.clone(),
                    vertex_names[idx + 1].clone(),
                    edge_label
                        .next()
                        .expect("Should have provided N-1 edge labels"),
                );
                to_return.add_edge(
                    v.clone(),
                    vertex_names[idx + 2].clone(),
                    edge_label
                        .next()
                        .expect("Should have provided N-1 edge labels"),
                );
            }
        }
        assert_eq!(
            edge_label.count(),
            0,
            "Should have provided N-1 edge labels"
        );
        to_return
    }

    /// Construct the quiver of type `B_n` or `C_n` with the specified vertex labels and edge labels.
    /// This is a default orientation.
    ///
    /// The edge labels should be provided in the following order
    /// `e_01, e_12, e_23, ..., e_{n-3,n-2}, e_{n-2,n-1}, e_{n-2,n-1}'` for type B
    /// `e_01, e_12, e_23, ..., e_{n-3,n-2}, e_{n-1,n-2}, e_{n-1,n-2}'` for type C
    /// The edge `e_ij` goes from vertex `vertex_names[i]` to vertex `vertex_names[j]`
    #[allow(clippy::missing_panics_doc)]
    pub fn make_bc_n<const N: usize>(
        type_b: bool,
        vertex_names: &[VertexLabel; N],
        mut edge_label: impl Iterator<Item = EdgeLabel>,
    ) -> Self {
        let mut to_return = Quiver::new();
        for v in vertex_names {
            to_return.add_vertex(v.clone());
        }
        for (idx, v) in vertex_names.iter().enumerate() {
            if idx + 2 < vertex_names.len() {
                to_return.add_edge(
                    v.clone(),
                    vertex_names[idx + 1].clone(),
                    edge_label
                        .next()
                        .expect("Should have provided N edge labels"),
                );
            } else if idx + 2 == vertex_names.len() {
                if type_b {
                    to_return.add_edge(
                        v.clone(),
                        vertex_names[idx + 1].clone(),
                        edge_label
                            .next()
                            .expect("Should have provided N edge labels"),
                    );
                    to_return.add_edge(
                        v.clone(),
                        vertex_names[idx + 1].clone(),
                        edge_label
                            .next()
                            .expect("Should have provided N edge labels"),
                    );
                } else {
                    to_return.add_edge(
                        vertex_names[idx + 1].clone(),
                        v.clone(),
                        edge_label
                            .next()
                            .expect("Should have provided N edge labels"),
                    );
                    to_return.add_edge(
                        vertex_names[idx + 1].clone(),
                        v.clone(),
                        edge_label
                            .next()
                            .expect("Should have provided N edge labels"),
                    );
                }
            }
        }
        assert_eq!(edge_label.count(), 0, "Should have provided N edge labels");
        to_return
    }
}

#[cfg(test)]
mod tests {
    use crate::quiver_algebra::make_a2_quiver;

    use super::*;

    #[test]
    fn a2_match() {
        let quiver = Quiver::make_a_n(&["alpha", "beta"], ["a"].into_iter());
        let make_a2_alt = make_a2_quiver();
        assert!(quiver == make_a2_alt);
    }

    #[test]
    fn make_d5() {
        let quiver = Quiver::make_d_n(
            &["v1", "v2", "v3", "v4", "v5"],
            ["e12", "e23", "e34", "e35"].into_iter(),
        );
        let mut quiver_vs = quiver
            .vertex_labels()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        quiver_vs.sort_unstable();
        assert_eq!(quiver_vs, ["v1", "v2", "v3", "v4", "v5"]);
        let mut quiver_es = quiver
            .edge_labels()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        quiver_es.sort_unstable();
        assert_eq!(quiver_es, ["e12", "e23", "e34", "e35"]);
        assert_eq!(
            quiver.edge_endpoint_labels(&"e12"),
            Some(("v1".into(), "v2".into()))
        );
        assert_eq!(
            quiver.edge_endpoint_labels(&"e23"),
            Some(("v2".into(), "v3".into()))
        );
        assert_eq!(
            quiver.edge_endpoint_labels(&"e34"),
            Some(("v3".into(), "v4".into()))
        );
        assert_eq!(
            quiver.edge_endpoint_labels(&"e35"),
            Some(("v3".into(), "v5".into()))
        );
    }
}
