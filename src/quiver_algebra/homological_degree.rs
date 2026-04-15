use crate::arithmetic_utils::Ring;
use crate::quiver_algebra::{path_algebra::PathAlgebra, quiver::BasisElt};

// Standard homological degrees for the Ginzburg DG-algebra:
//   original arrows a  →  degree  0
//   adjoint arrows  a* →  degree -1
//   self-loops      ω_v →  degree -2
// The Ginzburg cubic W = ∑_a (ω_{t(a)}·a·a* − ω_{s(a)}·a*·a) is then
// homogeneous of degree 0 + (−1) + (−2) = −3.
//
// dW/da degree -3
// dW/da^* degree -2
// dW/domega degree -1

/// Implemented by types that carry an explicit integer homological degree.
pub trait HasHomologicalDegree {
    /// Returns the homological degree of the element
    ///
    /// # Errors
    /// An element of a graded algebra
    /// may be 0 so it is in all degrees
    /// or it may be a sum of homogeneous elements of different degrees, so it has no well-defined degree.
    #[allow(clippy::result_unit_err)]
    fn homological_degree(&self) -> Result<i64, ()>;
}

impl<VertexLabel, EdgeLabel> HasHomologicalDegree for BasisElt<VertexLabel, EdgeLabel>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone + HasHomologicalDegree,
{
    fn homological_degree(&self) -> Result<i64, ()> {
        match self {
            BasisElt::Idempotent(_) => Ok(0),
            BasisElt::Path(word) => word
                .iter()
                .try_fold(0i64, |acc, e| e.homological_degree().map(|d| acc + d)),
        }
    }
}

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool> HasHomologicalDegree
    for PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + std::hash::Hash + Clone + HasHomologicalDegree,
    Coeffs: Ring,
{
    /// Returns `Ok(d)` if every summand has homological degree `d`, or `Err(())` if the
    /// element is zero (no summands) or the summands have inconsistent degrees.
    fn homological_degree(&self) -> Result<i64, ()> {
        let mut degree: Option<i64> = None;
        for basis in self.iter().map(|(b, _)| b) {
            let d = basis.homological_degree()?;
            match degree {
                None => degree = Some(d),
                Some(existing) if existing == d => {}
                Some(_) => return Err(()),
            }
        }
        degree.ok_or(())
    }
}

/// Edge label carrying an explicit homological degree.
#[derive(Clone, Eq, PartialEq, Hash, Debug, PartialOrd, Ord)]
pub struct DegreeLabel<T> {
    name: T,
    degree: Option<i64>,
}

impl<'a, T> DegreeLabel<T>
where
    T: 'a,
{
    /// Create a label with an explicit homological degree.
    pub fn new(name: T, degree: i64) -> Self {
        Self {
            name,
            degree: Some(degree),
        }
    }

    /// Create a label with homological degree 0 (the default for original quiver arrows).
    pub fn new_deg_zero(name: T) -> Self {
        Self {
            name,
            degree: Some(0),
        }
    }

    /// Returns a closure suitable for use as the `dagger` argument to
    /// [`Quiver::double`], [`Quiver::ginzburgify_and_cubic`], or
    /// [`Quiver::preprojective_algebra`].
    ///
    /// The closure renames the arrow via `rename` and sets the adjoint's degree
    /// to `−deg(a) − 1`, so that `deg(a) + deg(a*) = −1` always.  This keeps
    /// the Ginzburg cubic homogeneous of degree `−3`.
    pub fn dagger(rename: fn(&T) -> T) -> impl Fn(&Self) -> Self + 'a {
        move |t| Self {
            name: rename(&t.name),
            degree: t.homological_degree().map(|z| -z - 1).ok(),
        }
    }

    /// The underlying label, without the degree.
    pub fn name(&self) -> &T {
        &self.name
    }
}

impl<T> HasHomologicalDegree for DegreeLabel<T> {
    fn homological_degree(&self) -> Result<i64, ()> {
        self.degree.ok_or(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quiver_algebra::quiver::Quiver;

    fn self_loop(v: &&str) -> DegreeLabel<String> {
        DegreeLabel {
            name: format!("omega_{v}"),
            degree: Some(-2),
        }
    }

    /// The Ginzburg cubic W = ∑_a (ω_{t(a)}·a·a* − ω_{s(a)}·a*·a) should be homogeneous
    /// of degree −3 when original arrows are degree 0, adjoints degree −1, and self-loops
    /// degree −2.  Uses a 1-vertex quiver with a self-loop so all products compose and
    /// the cubic is non-zero.
    #[test]
    fn ginzburg_cubic_degree_minus_3() {
        let mut quiver: Quiver<&str, DegreeLabel<String>> = Quiver::new();
        quiver.add_edge(
            "v",
            "v",
            DegreeLabel {
                name: "a".into(),
                degree: Some(0),
            },
        );

        let (_ginzburg_arc, _, _, cubic) = quiver.ginzburgify_and_cubic::<i64, true>(
            DegreeLabel::<String>::dagger(|e_str| format!("{}*", e_str)),
            self_loop,
        );

        assert_eq!(cubic.homological_degree(), Ok(-3));
        assert!(cubic.all_parallel() == Ok(Some(("v", "v"))));
    }

    /// The Ginzburg cubic W = ∑_a (ω_{t(a)}·a·a* − ω_{s(a)}·a*·a) should be homogeneous
    /// of degree −3 when original arrows are degree 0, adjoints degree −1, and self-loops
    /// degree −2.  Uses an A2 quiver.
    #[test]
    fn ginzburg_cubic_a2() {
        let mut quiver: Quiver<&str, DegreeLabel<String>> = Quiver::new();
        quiver.add_edge(
            "alpha",
            "beta",
            DegreeLabel {
                name: "a".into(),
                degree: Some(0),
            },
        );

        let (_ginzburg_arc, _, _, cubic) = quiver.ginzburgify_and_cubic::<i64, true>(
            DegreeLabel::<String>::dagger(|e_str| format!("{}*", e_str)),
            self_loop,
        );

        assert_eq!(cubic.homological_degree(), Ok(-3));
        assert!(cubic.all_parallel().is_err());
    }
}
