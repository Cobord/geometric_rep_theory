use std::{
    collections::HashMap,
    ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign},
};

use num::{One, Zero};

use crate::arithmetic_utils::Ring;
use crate::plethystic::LambdaRing;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PowerSumPolynomial {
    partition: Vec<usize>,
}

impl PowerSumPolynomial {
    #[must_use = "A new power sum polynomial is returned"]
    pub fn new(partition: Vec<usize>) -> Self {
        let mut partition = partition;
        partition.sort_unstable();
        partition.reverse();
        Self { partition }
    }

    /// Plethysm action of `p_λ` on `x ∈ L` for any lambda ring `L`:
    ///   `p_λ[x] = ψ^{λ_1}(x) · ψ^{λ_2}(x) · … · ψ^{λ_k}(x)`
    /// The empty partition gives `p_∅ = 1`.
    #[must_use = "A new element in the lambda ring is returned"]
    pub fn plethysm<L: LambdaRing>(&self, x: &L) -> L {
        if self.partition.is_empty() {
            return L::one();
        }
        self.partition
            .iter()
            .map(|&k| x.clone().psi(k))
            .reduce(|acc, psi_k| acc * psi_k)
            .unwrap_or_else(L::one)
    }
}

impl Mul<Self> for PowerSumPolynomial {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut new_partition = self.partition;
        new_partition.extend(rhs.partition);
        Self::new(new_partition)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymmetricFunction<Coeffs: Ring> {
    l: HashMap<PowerSumPolynomial, Coeffs>,
}

impl<Coeffs: Ring> SymmetricFunction<Coeffs> {
    #[must_use = "A new symmetric function is returned"]
    pub fn new(l: HashMap<PowerSumPolynomial, Coeffs>) -> Self {
        Self { l }
    }

    #[must_use = "A new symmetric function is returned"]
    pub fn singleton(parts: Vec<usize>, c: Coeffs) -> Self {
        SymmetricFunction::new([(PowerSumPolynomial::new(parts), c)].into_iter().collect())
    }

    /// Coerce the coefficients to a different ring, e.g. for exact rational arithmetic.
    #[must_use = "A new symmetric function with the same p-basis but different coefficient type is returned"]
    pub fn coerce_coeffs<Coeffs2: Ring + From<Coeffs>>(self) -> SymmetricFunction<Coeffs2> {
        let l = self
            .l
            .into_iter()
            .map(|(p, c)| (p, Coeffs2::from(c)))
            .collect();
        SymmetricFunction { l }
    }

    /// Scale all coefficients by a common factor.
    pub fn scale_by<S: Clone>(&mut self, scalar: S)
    where
        Coeffs: MulAssign<S>,
    {
        for c in self.l.values_mut() {
            *c *= scalar.clone();
        }
    }

    pub fn plethysm<L: LambdaRing + MulAssign<Coeffs>>(&self, lambda: &L) -> L {
        let mut result = L::zero();
        for (p, c) in &self.l {
            let mut lambda_p = p.plethysm(lambda);
            lambda_p *= c.clone();
            result += lambda_p;
        }
        result
    }
}

impl<Coeffs: Ring> Zero for SymmetricFunction<Coeffs> {
    fn zero() -> Self {
        Self { l: HashMap::new() }
    }

    fn is_zero(&self) -> bool {
        self.l.is_empty() || self.l.values().all(Coeffs::is_zero)
    }
}

impl<Coeffs: Ring> Add<Self> for SymmetricFunction<Coeffs> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<Coeffs: Ring> AddAssign<Self> for SymmetricFunction<Coeffs> {
    fn add_assign(&mut self, rhs: Self) {
        for (k, v) in rhs.l {
            self.l.entry(k).and_modify(|e| *e += v.clone()).or_insert(v);
        }
    }
}

impl<Coeffs: Ring> Mul<Self> for SymmetricFunction<Coeffs> {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

impl<Coeffs: Ring> MulAssign<Self> for SymmetricFunction<Coeffs> {
    fn mul_assign(&mut self, rhs: Self) {
        let mut new_sympoly = HashMap::new();
        for (k1, v1) in &self.l {
            for (k2, v2) in &rhs.l {
                let new_k = k1.clone() * k2.clone();
                let coeff_contrib = v1.clone() * v2.clone();
                new_sympoly
                    .entry(new_k)
                    .and_modify(|e| *e += coeff_contrib.clone())
                    .or_insert(coeff_contrib);
            }
        }
        self.l = new_sympoly;
    }
}

impl<Coeffs: Ring> One for SymmetricFunction<Coeffs> {
    fn one() -> Self {
        let mut l = HashMap::new();
        l.insert(PowerSumPolynomial::new(vec![]), Coeffs::one());
        Self { l }
    }
}

impl<Coeffs: Ring> Neg for SymmetricFunction<Coeffs> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let l = self.l.into_iter().map(|(p, c)| (p, -c)).collect();
        Self { l }
    }
}

impl<Coeffs: Ring> Sub<Self> for SymmetricFunction<Coeffs> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

impl<Coeffs: Ring> SubAssign<Self> for SymmetricFunction<Coeffs> {
    fn sub_assign(&mut self, rhs: Self) {
        for (k, v) in rhs.l {
            self.l
                .entry(k)
                .and_modify(|e| *e -= v.clone())
                .or_insert(-v);
        }
    }
}

impl<Coeffs: Ring> From<Coeffs> for SymmetricFunction<Coeffs> {
    fn from(c: Coeffs) -> Self {
        let mut l = HashMap::new();
        l.insert(PowerSumPolynomial::new(vec![]), c);
        Self { l }
    }
}

impl<Coeffs: Ring + Div<usize, Output = Coeffs>> Div<usize> for SymmetricFunction<Coeffs> {
    type Output = Self;

    fn div(self, n: usize) -> Self::Output {
        let l = self.l.into_iter().map(|(p, c)| (p, c / n)).collect();
        Self { l }
    }
}

impl<Coeffs: Ring + Div<usize, Output = Coeffs>> LambdaRing for SymmetricFunction<Coeffs> {
    /// `ψ^n` acts directly on the p-basis: `p_λ ↦ p_{n·λ}` (multiply every part by n).
    fn psi(self, n: usize) -> Self {
        if n == 0 {
            return Self::one();
        }
        let l = self
            .l
            .into_iter()
            .map(|(p, c)| {
                let scaled = p.partition.iter().map(|&k| n * k).collect();
                (PowerSumPolynomial::new(scaled), c)
            })
            .collect();
        Self { l }
    }

    /// λ^n via Newton's identity, using the direct ψ above:
    ///   λ^n = (1/n) Σ_{k=1}^{n} (-1)^{k-1} ψ^k · λ^{n-k}
    fn lambda(self, n: usize) -> Self {
        if n == 0 {
            return Self::one();
        }
        if n == 1 {
            return self;
        }
        // lambdas[k] = λ^k(self), built iteratively from k=0 upward.
        let mut lambdas: Vec<Self> = Vec::with_capacity(n + 1);
        lambdas.push(Self::one()); // λ^0 = 1
        lambdas.push(self.clone()); // λ^1 = self
        for k in 2..=n {
            let mut val = Self::zero();
            for j in 1..=k {
                let psi_j = self.clone().psi(j);
                let lam_prev = lambdas[k - j].clone();
                if j % 2 == 1 {
                    val += psi_j * lam_prev;
                } else {
                    val -= psi_j * lam_prev;
                }
            }
            lambdas.push(val / k);
        }
        lambdas.swap_remove(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

    /// Rational coefficients backed by f64, with the two extra ops that
    /// SymmetricFunction<Z> acting on SymmetricFunction<Q> requires:
    ///   - Div<usize>  so that SymmetricFunction<Q>: LambdaRing
    ///   - MulAssign<i64> so that SymmetricFunction<Q>: MulAssign<i64>
    #[derive(Debug, Clone, PartialEq)]
    struct Q(f64);

    impl Add for Q {
        type Output = Q;
        fn add(self, r: Q) -> Q {
            Q(self.0 + r.0)
        }
    }
    impl AddAssign for Q {
        fn add_assign(&mut self, r: Q) {
            self.0 += r.0;
        }
    }
    impl Sub for Q {
        type Output = Q;
        fn sub(self, r: Q) -> Q {
            Q(self.0 - r.0)
        }
    }
    impl SubAssign for Q {
        fn sub_assign(&mut self, r: Q) {
            self.0 -= r.0;
        }
    }
    impl Mul for Q {
        type Output = Q;
        fn mul(self, r: Q) -> Q {
            Q(self.0 * r.0)
        }
    }
    impl MulAssign for Q {
        fn mul_assign(&mut self, r: Q) {
            self.0 *= r.0;
        }
    }
    impl Neg for Q {
        type Output = Q;
        fn neg(self) -> Q {
            Q(-self.0)
        }
    }
    impl num::Zero for Q {
        fn zero() -> Q {
            Q(0.0)
        }
        fn is_zero(&self) -> bool {
            self.0 == 0.0
        }
    }
    impl num::One for Q {
        fn one() -> Q {
            Q(1.0)
        }
    }
    impl Div<usize> for Q {
        type Output = Q;
        fn div(self, n: usize) -> Q {
            Q(self.0 / n as f64)
        }
    }
    impl MulAssign<i64> for Q {
        fn mul_assign(&mut self, n: i64) {
            self.0 *= n as f64;
        }
    }

    impl MulAssign<i64> for SymmetricFunction<Q> {
        fn mul_assign(&mut self, scalar: i64) {
            self.scale_by(scalar);
        }
    }
    impl MulAssign<Q> for SymmetricFunction<Q> {
        fn mul_assign(&mut self, scalar: Q) {
            self.scale_by(scalar);
        }
    }

    use proptest::prelude::*;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: &SymmetricFunction<Q>, b: &SymmetricFunction<Q>) -> bool {
        let all_keys = a.l.keys().chain(b.l.keys());
        all_keys.into_iter().all(|k| {
            let va = a.l.get(k).map_or(0.0, |q| q.0);
            let vb = b.l.get(k).map_or(0.0, |q| q.0);
            (va - vb).abs() < EPS
        })
    }

    proptest! {

        // p_n[p_m] = ψ^n(p_m) = p_{nm}
        #[test]
        fn prop_pn_on_pm(n in 1usize..=12, m in 1usize..=12) {
            let pn_otimes_1z = SymmetricFunction::singleton(vec![n], 1);
            let pm_otimes_1q = SymmetricFunction::singleton(vec![m], Q(1.0));
            let expected = SymmetricFunction::singleton(vec![n * m], Q(1.0));
            prop_assert_eq!(pn_otimes_1z.plethysm(&pm_otimes_1q), expected);
        }

        // (c · p_n)[p_m] = c · p_{nm}  — Z-linearity in the first argument
        #[test]
        fn prop_scalar_in_first_arg(c in -100i64..=100, n in 1usize..=12, m in 1usize..=12) {
            let c_pn_otimes_1z = SymmetricFunction::singleton(vec![n], c);
            let pm_otimes_1q = SymmetricFunction::singleton(vec![m], Q(1.0));
            let expected = SymmetricFunction::singleton(vec![n * m], Q(c as f64));
            prop_assert_eq!(c_pn_otimes_1z.plethysm(&pm_otimes_1q), expected);
        }

        // p_{(a,b)}[p_m] = ψ^a(p_m) · ψ^b(p_m) = p_{am} · p_{bm} = p_{(am,bm)}
        #[test]
        fn prop_product_partition_in_first_arg(a in 1usize..=8, b in 1usize..=8, m in 1usize..=8) {
            let pab_otimes_1z = SymmetricFunction::singleton(vec![a, b], 1);
            let pm_otimes_1q = SymmetricFunction::singleton(vec![m], Q(1.0));
            let expected = SymmetricFunction::singleton(vec![a * m, b * m], Q(1.0));
            prop_assert_eq!(pab_otimes_1z.plethysm(&pm_otimes_1q), expected);
        }

        // p_n[p_a + p_b] = p_{na} + p_{nb}  — Q-linearity in the second argument
        // Require a ≠ b so both terms stay distinct in the HashMap.
        #[test]
        fn prop_additive_in_second_arg(n in 1usize..=12, a in 1usize..=8, b in 1usize..=8) {
            prop_assume!(a != b);
            let pn_otimes_1z = SymmetricFunction::singleton(vec![n], 1);
            let pa_plus_pb_otimes_1q: SymmetricFunction<Q> = SymmetricFunction::new(
                [(PowerSumPolynomial::new(vec![a]), Q(1.0)), (PowerSumPolynomial::new(vec![b]), Q(1.0))].into_iter().collect(),
            );
            let expected: SymmetricFunction<Q> = SymmetricFunction::new(
                [(PowerSumPolynomial::new(vec![n * a]), Q(1.0)), (PowerSumPolynomial::new(vec![n * b]), Q(1.0))].into_iter().collect(),
            );
            prop_assert_eq!(pn_otimes_1z.plethysm(&pa_plus_pb_otimes_1q), expected);
        }
    }

    // e_5 = Σ_{λ⊢5} (-1)^{5-ℓ(λ)} / z_λ · p_λ
    fn e5() -> SymmetricFunction<Q> {
        SymmetricFunction::new(
            [
                (PowerSumPolynomial::new(vec![5]), Q(1.0 / 5.0)),
                (PowerSumPolynomial::new(vec![4, 1]), Q(-1.0 / 4.0)),
                (PowerSumPolynomial::new(vec![3, 2]), Q(-1.0 / 6.0)),
                (PowerSumPolynomial::new(vec![3, 1, 1]), Q(1.0 / 6.0)),
                (PowerSumPolynomial::new(vec![2, 2, 1]), Q(1.0 / 8.0)),
                (PowerSumPolynomial::new(vec![2, 1, 1, 1]), Q(-1.0 / 12.0)),
                (PowerSumPolynomial::new(vec![1, 1, 1, 1, 1]), Q(1.0 / 120.0)),
            ]
            .into_iter()
            .collect(),
        )
    }

    proptest! {
        // e_5 written explicitly in the p-basis, acting on p_m, equals λ^5(p_m).
        #[test]
        fn prop_e5_plethysm_matches_lambda5(m in 1usize..=6) {
            let g = SymmetricFunction::singleton(vec![m], Q(1.0));
            let via_plethysm = e5().plethysm(&g);
            let via_lambda   = g.lambda(5);
            prop_assert!(approx_eq(&via_plethysm, &via_lambda));
        }
    }
}
