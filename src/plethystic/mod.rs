mod generating_series;
mod lambda_ring;
mod power_series;
mod symmetric_function;

pub use generating_series::{
    AdamsIncreases, FilteredSemiRing, plethystic_exp, plethystic_log, truncated_exponential,
    truncated_log,
};
pub use lambda_ring::LambdaRing;
pub use power_series::PowerSeries;
pub use symmetric_function::{PowerSumPolynomial, SymmetricFunction};

#[cfg(test)]
#[rustfmt::skip]
pub(crate) mod test_utils {
    use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

    #[derive(Debug, Clone, PartialEq)]
    pub(crate) struct Q(pub(crate) f64);

    impl Add       for Q { type Output = Q; fn add(self, r: Q) -> Q { Q(self.0 + r.0) } }
    impl AddAssign for Q { fn add_assign(&mut self, r: Q) { self.0 += r.0; } }
    impl Sub       for Q { type Output = Q; fn sub(self, r: Q) -> Q { Q(self.0 - r.0) } }
    impl SubAssign for Q { fn sub_assign(&mut self, r: Q) { self.0 -= r.0; } }
    impl Mul       for Q { type Output = Q; fn mul(self, r: Q) -> Q { Q(self.0 * r.0) } }
    impl MulAssign for Q { fn mul_assign(&mut self, r: Q) { self.0 *= r.0; } }
    impl Neg       for Q { type Output = Q; fn neg(self) -> Q { Q(-self.0) } }
    impl num::Zero for Q { fn zero() -> Q { Q(0.0) } fn is_zero(&self) -> bool { self.0 == 0.0 } }
    impl num::One  for Q { fn one()  -> Q { Q(1.0) } }
    impl Div<usize> for Q { type Output = Q; fn div(self, n: usize) -> Q { Q(self.0 / n as f64) } }
    impl DivAssign<usize> for Q { fn div_assign(&mut self, n: usize) { self.0 /= n as f64; } }
    impl MulAssign<i64> for Q { fn mul_assign(&mut self, n: i64) { self.0 *= n as f64; } }

    pub(crate) const EPS: f64 = 1e-9;

    use crate::plethystic::SymmetricFunction;
    pub(crate) fn approx_eq(a: &SymmetricFunction<Q>, b: &SymmetricFunction<Q>) -> bool {
        let all_keys = a.l.keys().chain(b.l.keys());
        all_keys.into_iter().all(|k| {
            let va = a.l.get(k).map_or(0.0, |q| q.0);
            let vb = b.l.get(k).map_or(0.0, |q| q.0);
            (va - vb).abs() < EPS
        })
    }
}
