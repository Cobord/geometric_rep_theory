use std::collections::HashMap;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num::{One, Zero};

use crate::arithmetic_utils::Ring;
use crate::plethystic::LambdaRing;
use crate::plethystic::generating_series::{AdamsIncreases, FilteredSemiRing};

/// Formal power series in `N` variables over `Coeffs`,
/// stored as a map from exponent vectors to nonzero coefficients.
/// But we only ever keep finitely many terms because it will always be truncated
/// at some point. However we are still calling them `PowerSeries`
/// instead of multivariate polynomials because of the filtration context.
///
/// The filtration is by total degree: `S^{>=k}` consists of series
/// whose every monomial has total degree `≥ k`.
/// The Adams operations act by substitution: `ψ^n(f)(x_1,…,x_N) = f(x_1^n,…,x_N^n)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerSeries<Coeffs: Ring, const N: usize> {
    terms: HashMap<[usize; N], Coeffs>,
}

impl<Coeffs: Ring, const N: usize> PowerSeries<Coeffs, N> {
    #[must_use]
    pub fn new(mut terms: HashMap<[usize; N], Coeffs>) -> Self {
        terms.retain(|_, v| !v.is_zero());
        Self { terms }
    }

    /// A single monomial `c · x^alpha`.
    #[must_use]
    pub fn monomial(alpha: [usize; N], c: Coeffs) -> Self {
        Self::new(HashMap::from([(alpha, c)]))
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn scale_by<S: Clone + Zero>(&mut self, scalar: S)
    where
        Coeffs: MulAssign<S>,
    {
        if scalar.is_zero() {
            self.terms.clear();
        } else {
            for c in self.terms.values_mut() {
                *c *= scalar.clone();
            }
        }
    }

    fn total_degree(alpha: &[usize; N]) -> usize {
        alpha.iter().sum()
    }

    #[must_use = "Differentiating the power series D^d_op (self = sum_beta c_beta x^beta)"]
    #[allow(clippy::similar_names)]
    pub fn differentiate(&self, d_op: [usize; N]) -> Self {
        let mut to_return = Self::zero();
        for (x_pow, coeff) in &self.terms {
            if x_pow
                .iter()
                .zip(d_op)
                .any(|(from_x, from_d)| from_d > *from_x)
            {
                continue;
            }
            let my_power = core::array::from_fn(|idx| x_pow[idx] - d_op[idx]);
            let mut terms = HashMap::with_capacity(1);
            let mut integer_brought_down = 1;
            for idx in 0..N {
                let cur_x_power = x_pow[idx];
                let cur_d_power = d_op[idx];
                integer_brought_down *=
                    (0..cur_d_power).fold(1, |acc, d_fall| acc * (cur_x_power - d_fall));
            }
            let integer_brought_down = Coeffs::natural_inclusion(integer_brought_down);
            terms.insert(my_power, integer_brought_down * coeff.clone());
            let cur_contrib = PowerSeries { terms };
            to_return += cur_contrib;
        }
        to_return
    }
}

impl<Coeffs: Ring, const N: usize> Zero for PowerSeries<Coeffs, N> {
    fn zero() -> Self {
        Self {
            terms: HashMap::new(),
        }
    }

    fn is_zero(&self) -> bool {
        self.terms.values().all(Coeffs::is_zero)
    }
}

impl<Coeffs: Ring, const N: usize> One for PowerSeries<Coeffs, N> {
    fn one() -> Self {
        Self::new(HashMap::from([([0; N], Coeffs::one())]))
    }
}

impl<Coeffs: Ring, const N: usize> AddAssign for PowerSeries<Coeffs, N> {
    fn add_assign(&mut self, rhs: Self) {
        for (alpha, c) in rhs.terms {
            self.terms
                .entry(alpha)
                .and_modify(|e| *e += c.clone())
                .or_insert(c);
        }
        self.terms.retain(|_k, v| !v.is_zero());
    }
}

impl<Coeffs: Ring, const N: usize> Add for PowerSeries<Coeffs, N> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        self += rhs;
        self
    }
}

impl<Coeffs: Ring, const N: usize> SubAssign for PowerSeries<Coeffs, N> {
    fn sub_assign(&mut self, rhs: Self) {
        for (alpha, c) in rhs.terms {
            self.terms
                .entry(alpha)
                .and_modify(|e| *e -= c.clone())
                .or_insert(-c);
        }
        self.terms.retain(|_k, v| !v.is_zero());
    }
}

impl<Coeffs: Ring, const N: usize> Sub for PowerSeries<Coeffs, N> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        self -= rhs;
        self
    }
}

impl<Coeffs: Ring, const N: usize> Neg for PowerSeries<Coeffs, N> {
    type Output = Self;
    fn neg(self) -> Self {
        let terms = self
            .terms
            .into_iter()
            .map(|(alpha, c)| (alpha, -c))
            .collect();
        Self { terms }
    }
}

impl<Coeffs: Ring, const N: usize> MulAssign for PowerSeries<Coeffs, N> {
    fn mul_assign(&mut self, rhs: Self) {
        let mut new_terms: HashMap<[usize; N], Coeffs> = HashMap::new();
        for (alpha, ca) in &self.terms {
            for (beta, cb) in &rhs.terms {
                #[allow(clippy::suspicious_op_assign_impl)]
                let gamma = std::array::from_fn(|i| alpha[i] + beta[i]);
                let coeff = ca.clone() * cb.clone();
                #[allow(clippy::suspicious_op_assign_impl)]
                new_terms
                    .entry(gamma)
                    .and_modify(|e| *e += coeff.clone())
                    .or_insert(coeff);
            }
        }
        self.terms = new_terms;
    }
}

impl<Coeffs: Ring, const N: usize> Mul for PowerSeries<Coeffs, N> {
    type Output = Self;
    fn mul(mut self, rhs: Self) -> Self {
        self *= rhs;
        self
    }
}

impl<Coeffs: Ring + DivAssign<usize>, const N: usize> DivAssign<usize> for PowerSeries<Coeffs, N> {
    fn div_assign(&mut self, n: usize) {
        for c in self.terms.values_mut() {
            *c /= n;
        }
    }
}

impl<Coeffs: Ring + Div<usize, Output = Coeffs>, const N: usize> Div<usize>
    for PowerSeries<Coeffs, N>
{
    type Output = Self;
    fn div(self, n: usize) -> Self {
        let terms = self
            .terms
            .into_iter()
            .map(|(alpha, c)| (alpha, c / n))
            .collect();
        Self { terms }
    }
}

impl<Coeffs: Ring, const N: usize> FilteredSemiRing for PowerSeries<Coeffs, N> {
    fn truncate_at(self, n: usize) -> Self {
        let terms = self
            .terms
            .into_iter()
            .filter(|(alpha, _)| Self::total_degree(alpha) <= n)
            .collect();
        Self { terms }
    }

    fn truncates_to_zero_at(&self, n: usize) -> bool {
        self.terms
            .iter()
            .all(|(alpha, v)| Self::total_degree(alpha) > n || v.is_zero())
    }

    fn maximal_filtration(&self) -> Result<usize, ()> {
        if self.is_zero() {
            return Err(());
        }
        self.terms
            .iter()
            .filter_map(|(alpha, v)| {
                if v.is_zero() {
                    None
                } else {
                    Some(Self::total_degree(alpha))
                }
            })
            .min()
            .ok_or(())
    }
}

impl<Coeffs: Ring + Div<usize, Output = Coeffs>, const N: usize> LambdaRing
    for PowerSeries<Coeffs, N>
{
    /// `ψ^n` acts by substituting `x_i ↦ x_i^n`:
    ///   `f(x_1,…,x_N) ↦ f(x_1^n,…,x_N^n)`,
    /// i.e. every exponent vector `α` is replaced by `n·α` (componentwise).
    fn psi(self, n: usize) -> Self {
        if n == 0 {
            let total = self
                .terms
                .into_values()
                .fold(Coeffs::zero(), |acc, c| acc + c);
            let mut result = Self::one();
            result.scale_by(total);
            return result;
        }
        let terms = self
            .terms
            .into_iter()
            .map(|(alpha, c)| (alpha.map(|e| e * n), c))
            .collect();
        Self { terms }
    }

    /// `λ^n(f)` computed inductively from `ψ^1,…,ψ^n` via the inverse Newton identity:
    ///   `n·λ^n(f) = Σ_{k=1}^{n} (−1)^{k−1} ψ^k(f)·λ^{n−k}(f)`  (with `λ^0 = 1`).
    fn lambda(self, n: usize) -> Self {
        if n == 0 {
            return Self::one();
        }
        if n == 1 {
            return self;
        }
        // lambdas[k] = λ^k(self), built forward from k=0.
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

impl<Coeffs: Ring + Div<usize, Output = Coeffs>, const N: usize> AdamsIncreases
    for PowerSeries<Coeffs, N>
{
    /// `ψ^n` maps `x^α` (total degree `|α|`) to `x^{n·α}` (total degree `n·|α|`),
    /// so `f ∈ S^{>=m}` implies `ψ^n(f) ∈ S^{>=n·m}`.
    fn psi_bound(psi_n: usize, f_filtration: usize) -> usize {
        psi_n * f_filtration
    }
}

#[cfg(test)]
mod tests {
    use super::PowerSeries;
    use num::Zero;
    use std::collections::HashMap;

    // f = 3·x^9·y^5 + 2·x^2·y^1 + 5·x^4·y^0 + 7·x^5·y^9
    fn sample_series() -> PowerSeries<i64, 2> {
        PowerSeries::new(HashMap::from([
            ([9, 5], 3i64),
            ([2, 1], 2i64),
            ([4, 0], 5i64),
            ([5, 9], 7i64),
        ]))
    }

    // D_x^3 D_y^7: every term vanishes except 7·x^5·y^9.
    // Terms that vanish:
    //   3·x^9·y^5 : D_y^7 needs y-exp >= 7, but y-exp is 5 → gone
    //   2·x^2·y^1 : D_x^3 needs x-exp >= 3, but x-exp is 2 → gone
    //   5·x^4·y^0 : D_y^7 needs y-exp >= 7, but y-exp is 0 → gone
    // Surviving term 7·x^5·y^9:
    //   D_x^3(x^5) = 5·4·3·x^2 = 60·x^2
    //   D_y^7(y^9) = 9·8·7·6·5·4·3·y^2 = 181440·y^2
    //   coefficient = 7 · 60 · 181440 = 76204800
    #[test]
    fn differentiate_large_alpha_kills_terms() {
        let f = sample_series();
        let result = f.differentiate([3, 7]);
        let expected = PowerSeries::new(HashMap::from([([2, 2], 76_204_800i64)]));
        assert_eq!(result, expected);
    }

    // D_x^3 D_y^7 applied to a series where ALL terms are killed → zero.
    #[test]
    fn differentiate_all_terms_killed_gives_zero() {
        let f: PowerSeries<i64, 2> = PowerSeries::new(HashMap::from([
            ([9, 5], 3i64), // y-exp 5 < 7 → gone
            ([2, 1], 2i64), // x-exp 2 < 3 → gone
            ([4, 0], 5i64), // y-exp 0 < 7 → gone
        ]));
        let result = f.differentiate([3, 7]);
        assert!(result.is_zero());
    }

    // D_x^0 D_y^2: alpha_x = 0 acts as identity on x, only y-differentiation applies.
    // On sample_series():
    //   3·x^9·y^5 → 3·1·(5·4)·x^9·y^3 = 60·x^9·y^3
    //   2·x^2·y^1 : y-exp 1 < 2 → gone
    //   5·x^4·y^0 : y-exp 0 < 2 → gone
    //   7·x^5·y^9 → 7·1·(9·8)·x^5·y^7 = 504·x^5·y^7
    #[test]
    fn differentiate_zero_component_acts_as_identity() {
        let f = sample_series();
        let result = f.differentiate([0, 2]);
        let expected = PowerSeries::new(HashMap::from([([9, 3], 60i64), ([5, 7], 504i64)]));
        assert_eq!(result, expected);
    }
}
