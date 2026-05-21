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
