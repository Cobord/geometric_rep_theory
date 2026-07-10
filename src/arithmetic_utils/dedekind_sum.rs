use num::rational::Ratio;

/// The Dedekind sum `s(h,k) = sum_{r=1}^{k-1} ((r/k)) * ((h*r/k))`, where
/// `((x)) = x - floor(x) - 1/2` for non-integer `x`, and `((x)) = 0` for
/// integer `x` (the sawtooth function).
///
/// The classical ingredient of the Dedekind eta function's multiplier system
/// (see [`EtaTransformationGroup`](crate::lattice::EtaTransformationGroup)):
/// for `g = (a b; c d)` in `SL_2(Z)` with `c > 0`,
/// `eta(g*tau) = eps(g) * (c*tau+d)^(1/2) * eta(tau)` with
/// `eps(g) = exp(i*pi * ((a+d)/(12*c) - s(d,c) - 1/4))`.
///
/// Computed exactly over `Ratio<i128>` rather than floating point: `s(h,k)`
/// is always rational (a finite sum of products of halves of fractions with
/// denominator `k`), so there's no rounding error to accumulate over the
/// `k-1` terms of the sum in the first place.
///
/// # Panics
///
/// Panics if `k <= 0` (the sum is only defined for `k >= 1`, where `k = 1`
/// gives the empty sum `s(h,1) = 0`).
#[must_use]
pub fn dedekind_sum(h: i128, k: i128) -> Ratio<i128> {
    assert!(k > 0, "Dedekind sum s(h,k) requires k >= 1, got k = {k}");
    let half = Ratio::new(1, 2);
    let sawtooth_over_k = |numerator_mod_k: i128| {
        if numerator_mod_k == 0 {
            Ratio::new(0, 1)
        } else {
            Ratio::new(numerator_mod_k, k) - half
        }
    };
    (1..k).fold(Ratio::new(0, 1), |acc, r| {
        let first = sawtooth_over_k(r);
        let second = sawtooth_over_k((h * r).rem_euclid(k));
        acc + first * second
    })
}

#[cfg(test)]
mod tests {
    use super::dedekind_sum;
    use num::rational::Ratio;

    #[test]
    fn empty_sum_at_k_one() {
        // s(h,1) is the sum over the empty range 1..1.
        assert_eq!(dedekind_sum(0, 1), Ratio::new(0, 1));
        assert_eq!(dedekind_sum(5, 1), Ratio::new(0, 1));
    }

    #[test]
    fn s_one_two_vanishes() {
        // ((1/2)) = 0, so the single term at r=1 is 0*0.
        assert_eq!(dedekind_sum(1, 2), Ratio::new(0, 1));
    }

    #[test]
    fn s_one_k_matches_closed_form() {
        // s(1,k) = sum_{r=1}^{k-1} ((r/k))^2 = (k-1)(k-2)/(12k).
        for k in 2..20 {
            let expected = Ratio::new((k - 1) * (k - 2), 12 * k);
            assert_eq!(dedekind_sum(1, k), expected, "k = {k}");
        }
    }

    #[test]
    fn reciprocity_law() {
        // Dedekind reciprocity: for coprime h,k > 0,
        // s(h,k) + s(k,h) = -1/4 + (h/k + k/h + 1/(h*k))/12.
        for (h, k) in [(2, 3), (3, 5), (4, 7), (5, 9), (7, 12)] {
            let lhs = dedekind_sum(h, k) + dedekind_sum(k, h);
            let rhs = Ratio::new(-1, 4)
                + (Ratio::new(h, k) + Ratio::new(k, h) + Ratio::new(1, h * k)) / 12;
            assert_eq!(lhs, rhs, "h = {h}, k = {k}");
        }
    }
}
