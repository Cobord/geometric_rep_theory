use std::{collections::HashMap, sync::Arc};

use nonempty::NonEmpty;
use num::Integer;

use crate::quiver_algebra::{
    BasisElt, DegreeLabel, HasHomologicalDegree, PathAlgebra, Quiver, Ring,
};

/// A differential graded path algebra `(kQ, d)`.
///
/// Wraps a graded quiver `Q` (with `DegreeLabel`-decorated edges) together with a degree-+1
/// derivation `d` satisfying `d² = 0` and the graded Leibniz rule. The differential is stored
/// as a map `arrow ↦ PathAlgebra element` for arrows that have non-trivial `d`.
pub struct GradedDifferentialQuiver<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    quiver_arc: Arc<Quiver<VertexLabel, DegreeLabel<EdgeLabel>>>,
    differential:
        HashMap<EdgeLabel, PathAlgebra<VertexLabel, DegreeLabel<EdgeLabel>, Coeffs, OP_ALG>>,
}

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool>
    GradedDifferentialQuiver<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    /// Construct a DGA from a graded quiver and a map of arrow differentials.
    ///
    /// Arrow labels not present in `quiver_arc` are silently dropped from `differential`.
    #[must_use = "You have endowed the path algebra with a differential. Not using this gets rid of that information"]
    pub fn new(
        quiver_arc: Arc<Quiver<VertexLabel, DegreeLabel<EdgeLabel>>>,
        mut differential: HashMap<
            EdgeLabel,
            PathAlgebra<VertexLabel, DegreeLabel<EdgeLabel>, Coeffs, OP_ALG>,
        >,
    ) -> Self {
        let edge_names = quiver_arc
            .edge_labels()
            .map(DegreeLabel::name)
            .collect::<Vec<_>>();
        differential.retain(|k, _| edge_names.contains(&k));
        for v in differential.values() {
            debug_assert!(
                v.all_parallel().is_ok(),
                "a = e_s a e_t so da = e_s da e_t so it better be all parallel summands or no summands"
            );
        }
        Self {
            quiver_arc,
            differential,
        }
    }

    /// The underlying graded quiver.
    #[must_use = "What do you want with the underlying quiver of it's differential graded path algebra?"]
    pub fn quiver(&self) -> &Arc<Quiver<VertexLabel, DegreeLabel<EdgeLabel>>> {
        &self.quiver_arc
    }

    pub(crate) fn apply_differential_letter(
        &self,
        letter: &DegreeLabel<EdgeLabel>,
    ) -> Option<PathAlgebra<VertexLabel, DegreeLabel<EdgeLabel>, Coeffs, OP_ALG>> {
        self.differential.get(letter.name()).cloned()
    }

    pub(crate) fn apply_differential_word(
        &self,
        word: &[DegreeLabel<EdgeLabel>],
    ) -> PathAlgebra<VertexLabel, DegreeLabel<EdgeLabel>, Coeffs, OP_ALG> {
        let mut result = PathAlgebra::zero(self.quiver_arc.clone());
        for idx in 0..word.len() {
            let removed_edge_src_tgt = self
                .quiver()
                .edge_endpoint_labels(&word[idx])
                .expect("This is an edge in the quiver");
            let before_part = BasisElt::create(&word[..idx])
                .unwrap_or_else(|| BasisElt::Idempotent(removed_edge_src_tgt.0.clone()));
            let after_part = BasisElt::create(&word[idx + 1..])
                .unwrap_or_else(|| BasisElt::Idempotent(removed_edge_src_tgt.1.clone()));
            if let Some(diff_part) = self.apply_differential_letter(&word[idx]) {
                let mut cur_contrib = diff_part;
                cur_contrib = before_part * cur_contrib;
                cur_contrib *= after_part;
                let sign: bool = word[..idx]
                    .iter()
                    .map(|e| e.homological_degree().expect("has homological degree"))
                    .sum::<i64>()
                    .is_odd();
                if sign {
                    result -= cur_contrib;
                } else {
                    result += cur_contrib;
                }
            }
        }
        result
    }

    pub(crate) fn apply_differential_path(
        &self,
        path: &NonEmpty<DegreeLabel<EdgeLabel>>,
    ) -> PathAlgebra<VertexLabel, DegreeLabel<EdgeLabel>, Coeffs, OP_ALG> {
        if path.len() == 1 {
            self.apply_differential_letter(path.first())
                .unwrap_or_else(|| PathAlgebra::zero(self.quiver_arc.clone()))
        } else {
            let tgt_of_first_edge = self
                .quiver()
                .edge_endpoint_labels(path.first())
                .expect("This is an edge in the quiver")
                .1;
            let just_rest_of_path = BasisElt::create(path.tail())
                .unwrap_or_else(|| BasisElt::Idempotent(tgt_of_first_edge.clone()));
            let sign_first = path
                .first()
                .homological_degree()
                .expect("has homological degree")
                .is_odd();
            // (-1)^|head| head * d(tail)
            let rest_contrib = if sign_first {
                -(BasisElt::Path(NonEmpty::singleton(path.first().clone()))
                    * self.apply_differential_word(&path.tail))
            } else {
                BasisElt::Path(NonEmpty::singleton(path.first().clone()))
                    * self.apply_differential_word(&path.tail)
            };
            if let Some(first_contrib) = self.apply_differential_letter(path.first()) {
                // d(head)*tail
                let mut returning = first_contrib * just_rest_of_path;
                returning += rest_contrib;
                returning
            } else {
                rest_contrib
            }
        }
    }

    /// Apply the differential `d` to an arbitrary algebra element, using the graded Leibniz rule
    /// with Koszul signs. Idempotents are sent to zero.
    pub fn apply_differential(
        &self,
        algebra_element: &PathAlgebra<VertexLabel, DegreeLabel<EdgeLabel>, Coeffs, OP_ALG>,
    ) -> PathAlgebra<VertexLabel, DegreeLabel<EdgeLabel>, Coeffs, OP_ALG> {
        debug_assert!(
            Arc::ptr_eq(algebra_element.quiver(), self.quiver()),
            "On same quiver"
        );
        let mut result = PathAlgebra::zero(self.quiver_arc.clone());
        for (basis_elt, coeff) in algebra_element.iter() {
            match basis_elt {
                crate::quiver_algebra::BasisElt::Idempotent(_) => {
                    // The differential of an idempotent is zero.
                }
                crate::quiver_algebra::BasisElt::Path(word) => {
                    let current = self.apply_differential_path(word);
                    result += current * coeff.clone();
                }
            }
        }
        result.simplify_default();
        result
    }

    /// Construct the Ginzburg DGA `Γ(Q, W)` from a quiver and superpotential.
    ///
    /// Builds the graded quiver by adding adjoint arrows `a* = label_dagger(a)` (degree −1)
    /// and self-loops `ω_v = self_loop(v)` (degree −2) at each vertex. Sets differentials
    /// `d(a*) = ∂W/∂a` and `d(ω_v) = ∂W_cubic/∂ω_v`, where `W_cubic` is the Ginzburg cubic.
    pub fn create_ginzburg_dga(
        ungraded_quiver: Quiver<VertexLabel, EdgeLabel>,
        original_potential: PathAlgebra<VertexLabel, EdgeLabel, Coeffs, OP_ALG>,
        label_dagger: fn(&EdgeLabel) -> EdgeLabel,
        self_loop: fn(&VertexLabel) -> EdgeLabel,
    ) -> Self {
        let ungraded_quiver = ungraded_quiver.map_labels(|z| z, DegreeLabel::new_deg_zero);
        let (ginzburg_arc, adjoint_pairs, self_loops, ginzburg_cubic) = ungraded_quiver
            .ginzburgify_and_cubic(DegreeLabel::dagger(label_dagger), |v| {
                DegreeLabel::new(self_loop(v), -2)
            });
        let mut differentials = HashMap::<
            EdgeLabel,
            PathAlgebra<VertexLabel, DegreeLabel<EdgeLabel>, Coeffs, OP_ALG>,
        >::with_capacity(ginzburg_arc.num_arrows());
        let original_potential_promoted = PathAlgebra::new(
            ginzburg_arc.clone(),
            original_potential.map_labels(|z| z, DegreeLabel::new_deg_zero),
        );
        for (a, a_dagger) in adjoint_pairs {
            let mut dw_da = original_potential_promoted.clone();
            dw_da.cyclic_derivative(&a);
            differentials.insert(a_dagger.name().clone(), dw_da);
        }
        for ti in self_loops {
            let mut dcubic_dt = ginzburg_cubic.clone();
            dcubic_dt.cyclic_derivative(&ti);
            differentials.insert(ti.name().clone(), dcubic_dt);
        }
        Self::new(ginzburg_arc, differentials)
    }

    /// Returns `true` if `d(algebra_element) = 0`, i.e. the element is closed in the
    /// homological sense (a cocycle).
    #[must_use = "This method checks if the given algebra element is closed (differential is zero).
    If you just want to check if it's closed, use is_definitely_closed instead to avoid the overhead of
    computing the full differential."]
    pub fn is_definitely_closed(
        &self,
        algebra_element: &PathAlgebra<VertexLabel, DegreeLabel<EdgeLabel>, Coeffs, OP_ALG>,
    ) -> bool {
        !self.apply_differential(algebra_element).might_be_nonzero()
    }
}

impl<VertexLabel, EdgeLabel, Coeffs, const OP_ALG: bool> From<Quiver<VertexLabel, EdgeLabel>>
    for GradedDifferentialQuiver<VertexLabel, EdgeLabel, Coeffs, OP_ALG>
where
    VertexLabel: std::hash::Hash + Eq + Clone,
    EdgeLabel: Eq + Clone + std::hash::Hash,
    Coeffs: Ring,
{
    fn from(value: Quiver<VertexLabel, EdgeLabel>) -> Self {
        let value_zero_graded = value.map_labels(|z| z, DegreeLabel::new_deg_zero);
        Self {
            quiver_arc: Arc::new(value_zero_graded),
            differential: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quiver_algebra::homological_degree::DegreeLabel;
    use nonempty::nonempty;

    /// Build a GradedDifferentialQuiver on a 1-vertex quiver with self-loops
    ///   x  (degree  0)
    ///   y1 (degree -1)
    ///   y2 (degree -1)
    /// and differential d(y1) = x, d(y2) = x, d(x) = 0.
    ///
    /// Checks:
    ///  1. d(y1) = x  (basic lookup)
    ///  2. d²(y1) = 0
    ///  3. d(y1·y2) = x·y2 - y1·x  (Koszul sign from deg(y1) = -1, which is odd)
    ///  4. d²(y1·y2) = 0
    #[test]
    fn d_squared_zero_with_koszul_signs() {
        let x = DegreeLabel::new("x".to_string(), 0i64);
        let y1 = DegreeLabel::new("y1".to_string(), -1i64);
        let y2 = DegreeLabel::new("y2".to_string(), -1i64);

        let mut quiver: Quiver<&str, DegreeLabel<String>> = Quiver::new();
        quiver.add_edge("v", "v", x.clone());
        quiver.add_edge("v", "v", y1.clone());
        quiver.add_edge("v", "v", y2.clone());
        let quiver_arc = Arc::new(quiver);

        let x_elt = PathAlgebra::singleton(
            quiver_arc.clone(),
            BasisElt::Path(nonempty![x.clone()]),
            1i64,
        );

        let mut differential: HashMap<String, PathAlgebra<&str, DegreeLabel<String>, i64, true>> =
            HashMap::new();
        differential.insert("y1".to_string(), x_elt.clone());
        differential.insert("y2".to_string(), x_elt.clone());

        let dga = GradedDifferentialQuiver::new(quiver_arc.clone(), differential);

        let zero = PathAlgebra::zero(quiver_arc.clone());

        // 1. d(y1) = x
        let y1_elt = PathAlgebra::singleton(
            quiver_arc.clone(),
            BasisElt::Path(nonempty![y1.clone()]),
            1i64,
        );
        assert_eq!(dga.apply_differential(&y1_elt), x_elt);

        // 2. d²(y1) = d(x) = 0
        assert_eq!(dga.apply_differential(&x_elt), zero);

        // 3. d(y1·y2) = x·y2 - y1·x
        //    Koszul: the contribution at y2's position carries sign (-1)^{deg(y1)} = (-1)^{-1} = -1.
        let y1_y2 = PathAlgebra::singleton(
            quiver_arc.clone(),
            BasisElt::Path(nonempty![y1.clone(), y2.clone()]),
            1i64,
        );
        let x_y2 = PathAlgebra::singleton(
            quiver_arc.clone(),
            BasisElt::Path(nonempty![x.clone(), y2.clone()]),
            1i64,
        );
        let y1_x = PathAlgebra::singleton(
            quiver_arc.clone(),
            BasisElt::Path(nonempty![y1.clone(), x.clone()]),
            1i64,
        );
        let expected_d_y1y2 = x_y2 - y1_x;
        assert_eq!(
            dga.apply_differential_word(&[y1.clone(), y2.clone()]),
            expected_d_y1y2
        );

        // 4. d²(y1·y2) = 0
        let d_y1y2 = dga.apply_differential(&y1_y2);
        assert_eq!(dga.apply_differential(&d_y1y2), zero);

        // 5. d^2 (x y1 y2 + 5 y1^2 - 7 y2)
        let x_elt = PathAlgebra::singleton(quiver_arc.clone(), BasisElt::Path(nonempty![x]), 1i64);
        let y1_elt =
            PathAlgebra::singleton(quiver_arc.clone(), BasisElt::Path(nonempty![y1]), 1i64);
        let y2_elt =
            PathAlgebra::singleton(quiver_arc.clone(), BasisElt::Path(nonempty![y2]), 1i64);

        let expr = x_elt.clone() * y1_elt.clone() * y2_elt.clone() + y1_elt.clone() * y1_elt * 5i64
            - y2_elt * 7i64;
        let d_expr = dga.apply_differential(&expr);
        assert_eq!(dga.apply_differential(&d_expr), zero);
    }

    #[test]
    fn leibniz_rule() {
        let x = DegreeLabel::new("x".to_string(), 0i64);
        let y1 = DegreeLabel::new("y1".to_string(), -1i64);
        let y2 = DegreeLabel::new("y2".to_string(), -1i64);

        let mut quiver: Quiver<&str, DegreeLabel<String>> = Quiver::new();
        quiver.add_edge("v", "v", x.clone());
        quiver.add_edge("v", "v", y1.clone());
        quiver.add_edge("v", "v", y2.clone());
        let quiver_arc = Arc::new(quiver);

        let x_elt = PathAlgebra::singleton(quiver_arc.clone(), BasisElt::Path(nonempty![x]), 1i64);
        let y1_elt =
            PathAlgebra::singleton(quiver_arc.clone(), BasisElt::Path(nonempty![y1]), 1i64);
        let y2_elt =
            PathAlgebra::singleton(quiver_arc.clone(), BasisElt::Path(nonempty![y2]), 1i64);

        let mut differential: HashMap<String, PathAlgebra<&str, DegreeLabel<String>, i64, true>> =
            HashMap::new();
        differential.insert("y1".to_string(), x_elt.clone());
        differential.insert("y2".to_string(), x_elt.clone());

        let dga = GradedDifferentialQuiver::new(quiver_arc.clone(), differential);

        for a in [x_elt.clone(), y1_elt.clone(), y2_elt.clone()] {
            let sign = if a == x_elt { 1 } else { -1 };
            for b in [x_elt.clone(), y1_elt.clone(), y2_elt.clone()] {
                let a_b = a.clone() * b.clone();
                let d_ab = dga.apply_differential(&a_b);
                let da_b = dga.apply_differential(&a) * b.clone();
                let a_db = a.clone() * dga.apply_differential(&b);
                assert_eq!(d_ab, da_b + a_db * sign);
            }
        }
    }

    /// One-loop Ginzburg DGA with potential W = x^7.
    ///
    /// Quiver: single vertex "v", single loop "x" (degree 0).
    /// Ginzburg quiver adds x* (degree -1) and ω (degree -2).
    ///
    /// Differentials:
    ///   d(x)  = 0          (original arrow, absent from HashMap)
    ///   d(x*) = ∂W/∂x = 7·x^6
    ///   d(ω)  = ∂W_cubic/∂ω = x·x* - x*·x
    #[test]
    fn one_loop_x7_ginzburg_dga() {
        let mut original_quiver: Quiver<&str, String> = Quiver::new();
        original_quiver.add_edge("v", "v", "x".to_string());

        // W = x^7 built against a separate Arc so the quiver can be moved into create_ginzburg_dga
        let w = PathAlgebra::<_, _, _, true>::singleton(
            Arc::new({
                let mut q: Quiver<&str, String> = Quiver::new();
                q.add_edge("v", "v", "x".to_string());
                q
            }),
            BasisElt::Path(nonempty::nonempty![
                "x".to_string(),
                "x".to_string(),
                "x".to_string(),
                "x".to_string(),
                "x".to_string(),
                "x".to_string(),
                "x".to_string()
            ]),
            1.0f64,
        );

        let dga = GradedDifferentialQuiver::create_ginzburg_dga(
            original_quiver,
            w,
            |e| format!("{}*", e),
            |v| format!("omega_{}", v),
        );

        let x_label = DegreeLabel::new_deg_zero("x".to_string());
        let xstar_label = DegreeLabel::new("x*".to_string(), -1);
        let omega_label = DegreeLabel::new("omega_v".to_string(), -2);

        // d(x) is absent (degree 0 arrow, not in differential)
        assert_eq!(dga.apply_differential_letter(&x_label), None);

        // d(x*) = 7·x^6
        let x6 = PathAlgebra::singleton(
            dga.quiver().clone(),
            BasisElt::Path(nonempty::nonempty![
                x_label.clone(),
                x_label.clone(),
                x_label.clone(),
                x_label.clone(),
                x_label.clone(),
                x_label.clone()
            ]),
            7.0f64,
        );
        assert_eq!(dga.apply_differential_letter(&xstar_label), Some(x6));

        // d(ω) = x·x* - x*·x
        let x_xstar = PathAlgebra::singleton(
            dga.quiver().clone(),
            BasisElt::Path(nonempty::nonempty![x_label.clone(), xstar_label.clone()]),
            1.0f64,
        );
        let xstar_x = PathAlgebra::singleton(
            dga.quiver().clone(),
            BasisElt::Path(nonempty::nonempty![xstar_label.clone(), x_label.clone()]),
            1.0f64,
        );
        assert_eq!(
            dga.apply_differential_letter(&omega_label),
            Some(x_xstar - xstar_x)
        );
    }

    /// d(x*·x*) in the one-loop x^7 Ginzburg DGA.
    ///
    /// By Leibniz with |x*| = -1 (odd):
    ///   d(x*·x*) = d(x*)·x* + (-1)^{|x*|} x*·d(x*)
    ///            = 7x^6·x* - 7x*·x^6
    ///
    /// This exercises apply_differential on a compound path using the Koszul sign rule.
    #[test]
    fn one_loop_x7_leibniz_on_xstar_squared() {
        let mut original_quiver: Quiver<&str, String> = Quiver::new();
        original_quiver.add_edge("v", "v", "x".to_string());

        let w = PathAlgebra::<_, _, _, true>::singleton(
            Arc::new({
                let mut q: Quiver<&str, String> = Quiver::new();
                q.add_edge("v", "v", "x".to_string());
                q
            }),
            BasisElt::Path(nonempty::nonempty![
                "x".to_string(),
                "x".to_string(),
                "x".to_string(),
                "x".to_string(),
                "x".to_string(),
                "x".to_string(),
                "x".to_string()
            ]),
            1.0f64,
        );

        let dga = GradedDifferentialQuiver::create_ginzburg_dga(
            original_quiver,
            w,
            |e| format!("{}*", e),
            |v| format!("omega_{}", v),
        );

        let x_label = DegreeLabel::new_deg_zero("x".to_string());
        let xstar_label = DegreeLabel::new("x*".to_string(), -1);

        let xstar_sq = PathAlgebra::singleton(
            dga.quiver().clone(),
            BasisElt::Path(nonempty::nonempty![
                xstar_label.clone(),
                xstar_label.clone()
            ]),
            1.0f64,
        );

        // 7x^6·x*
        let x6_xstar = PathAlgebra::singleton(
            dga.quiver().clone(),
            BasisElt::Path(nonempty::nonempty![
                x_label.clone(),
                x_label.clone(),
                x_label.clone(),
                x_label.clone(),
                x_label.clone(),
                x_label.clone(),
                xstar_label.clone()
            ]),
            7.0f64,
        );
        // 7x*·x^6
        let xstar_x6 = PathAlgebra::singleton(
            dga.quiver().clone(),
            BasisElt::Path(nonempty::nonempty![
                xstar_label.clone(),
                x_label.clone(),
                x_label.clone(),
                x_label.clone(),
                x_label.clone(),
                x_label.clone(),
                x_label.clone()
            ]),
            7.0f64,
        );

        let expected = x6_xstar - xstar_x6;
        assert_eq!(dga.apply_differential(&xstar_sq), expected);
    }

    #[test]
    fn ginzburg_test() {
        let starting_quiver =
            crate::quiver_algebra::make_a2_quiver().map_labels(|v| v, |e| e.to_string());
        let dga = GradedDifferentialQuiver::<_, _, f64, true>::create_ginzburg_dga(
            starting_quiver,
            PathAlgebra::zero(Arc::new(Quiver::new())),
            |e| format!("{}*", e),
            |v| format!("omega_{}", v),
        );
        assert!(dga.quiver().contains_vertex(&"alpha"));
        assert!(dga.quiver().contains_vertex(&"beta"));
        assert!(
            dga.quiver()
                .contains_edge(&DegreeLabel::new_deg_zero("a".to_string()))
        );
        assert!(
            dga.quiver()
                .contains_edge(&DegreeLabel::new("a*".to_string(), -1))
        );
        assert!(
            dga.quiver()
                .contains_edge(&DegreeLabel::new("omega_alpha".to_string(), -2))
        );
        assert!(
            dga.quiver()
                .contains_edge(&DegreeLabel::new("omega_beta".to_string(), -2))
        );

        let da = dga.apply_differential_letter(&DegreeLabel::new("a".to_string(), 0));
        assert_eq!(da, None);

        let zero = PathAlgebra::zero(dga.quiver().clone());
        let a = PathAlgebra::singleton(
            dga.quiver().clone(),
            BasisElt::Path(nonempty::nonempty![DegreeLabel::new_deg_zero(
                "a".to_string()
            )]),
            1.0,
        );
        let a_dag = PathAlgebra::singleton(
            dga.quiver().clone(),
            BasisElt::Path(nonempty::nonempty![DegreeLabel::new("a*".to_string(), -1)]),
            1.0,
        );

        let dadag = dga.apply_differential_letter(&DegreeLabel::new("a*".to_string(), -1));
        assert_eq!(dadag, Some(zero));

        let domega =
            dga.apply_differential_letter(&DegreeLabel::new("omega_alpha".to_string(), -2));
        let expected = a.clone() * a_dag.clone();
        assert_eq!(domega, Some(expected));

        let domega = dga.apply_differential_letter(&DegreeLabel::new("omega_beta".to_string(), -2));
        let expected = -a_dag.clone() * a.clone();
        assert_eq!(domega, Some(expected));
    }
}
