use std::ops::Range;

/// A planar rooted tree with labelled leaves.
///
/// Every variant stores an **absolute** `range` into the root's input array: evaluating
/// the subtree rooted here consumes exactly `inputs[node.range()]`.
///
/// The total number of leaves equals the number of inputs the tree consumes.
///
/// # Example
/// ```
/// use geometric_rep_theory::infinity_algebra::OperationTree;
/// // a node with 3 children, the first of which has 4 children — 6 leaves total
/// let tree = OperationTree::node(vec![
///     OperationTree::node(vec![
///         OperationTree::leaf(), OperationTree::leaf(),
///         OperationTree::leaf(), OperationTree::leaf(),
///     ]),
///     OperationTree::leaf(),
///     OperationTree::leaf(),
/// ]);
/// assert_eq!(tree.num_leaves(), 6);
/// ```
pub enum OperationTree {
    Leaf {
        absoulte_position: usize,
    },
    Node {
        children: Vec<OperationTree>,
        absoulute_range: Range<usize>,
        arities_used: Vec<usize>,
        num_leaves: usize,
    },
}

impl OperationTree {
    fn shift_ranges(&mut self, offset: usize) {
        match self {
            Self::Leaf {
                absoulte_position: range,
            } => *range += offset,
            Self::Node {
                absoulute_range: range,
                children,
                ..
            } => {
                *range = (range.start + offset)..(range.end + offset);
                for child in children {
                    child.shift_ranges(offset);
                }
            }
        }
    }

    /// A leaf node representing a single input slot.
    #[must_use = "discarding a leaf node has no effect; use it as a child of another node or as the tree itself"]
    pub fn leaf() -> Self {
        Self::Leaf {
            absoulte_position: 0,
        }
    }

    /// Build an internal node with the given children.
    ///
    /// Children are assigned contiguous absolute ranges starting at 0; if this node
    /// is later embedded in a larger tree, `shift_ranges` adjusts all ranges upward.
    ///
    /// # Panics
    /// Panics if `children` is empty.
    #[must_use = "discarding a node has no effect; use it as a child of another node or as the tree itself"]
    pub fn node(children: Vec<OperationTree>) -> Self {
        assert!(
            !children.is_empty(),
            "internal node must have at least one child"
        );
        let cur_arity = children.len();
        let mut offset = 0usize;
        let mut arities_used: Vec<usize> = Vec::new();
        let mut num_leaves = 0usize;

        let shifted_children: Vec<OperationTree> = children
            .into_iter()
            .map(|mut child| {
                child.shift_ranges(offset);
                let n = child.num_leaves();
                offset += n;
                num_leaves += n;
                arities_used.extend_from_slice(child.required_arities());
                child
            })
            .collect();

        arities_used.push(cur_arity);
        arities_used.sort_unstable();
        arities_used.dedup();

        Self::Node {
            children: shifted_children,
            absoulute_range: 0..num_leaves,
            arities_used,
            num_leaves,
        }
    }

    #[must_use = "This is the number of inputs for the entire tree"]
    pub fn num_leaves(&self) -> usize {
        match self {
            Self::Leaf { .. } => 1,
            Self::Node { num_leaves, .. } => *num_leaves,
        }
    }

    #[must_use = "If we need to check that we have the information of all of the n-ary operations for every relevant n"]
    pub fn required_arities(&self) -> &[usize] {
        match self {
            Self::Leaf { .. } => &[],
            Self::Node { arities_used, .. } => arities_used,
        }
    }
}
