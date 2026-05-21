use std::collections::HashMap;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num::{One, Zero};

use crate::arithmetic_utils::{Ring, binom, multi_index_le};
use crate::plethystic::PowerSeries;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DifferentialOperator<R: Ring, const NUM_VARIABLES: usize> {
    terms: HashMap<[usize; NUM_VARIABLES], PowerSeries<R, NUM_VARIABLES>>,
}

impl<R: Ring, const NUM_VARIABLES: usize> DifferentialOperator<R, NUM_VARIABLES> {
    #[must_use = "The maximum order |alpha| in sum p(x1..xn) D^alpha"]
    pub fn differential_order(&self) -> usize {
        self.terms.keys().fold(0, |acc, d_operator| {
            std::cmp::max(acc, d_operator.iter().sum())
        })
    }
}

pub trait NVarDifferentiable<const NUM_VARIABLES: usize> {
    #[must_use = "differentiated by D^alpha"]
    fn differentiate(&self, d_op: [usize; NUM_VARIABLES]) -> Self;
}

pub trait LeftMul<LHS> {
    type Output;
    fn left_mul(self, lhs: LHS) -> Self::Output;
}

impl<R: Ring, const NUM_VARIABLES: usize> LeftMul<PowerSeries<R, NUM_VARIABLES>>
    for DifferentialOperator<R, NUM_VARIABLES>
{
    type Output = Self;

    fn left_mul(mut self, lhs: PowerSeries<R, NUM_VARIABLES>) -> Self::Output {
        self.terms.values_mut().for_each(|z| {
            *z *= lhs.clone();
        });
        self
    }
}

impl<R: Ring, const NUM_VARIABLES: usize> NVarDifferentiable<NUM_VARIABLES>
    for DifferentialOperator<R, NUM_VARIABLES>
{
    // D^d · (Σ_α p_α(x) D^α) = Σ_α Σ_{β ≤ d} C(d,β) · (D^β p_α)(x) · D^{d−β+α}
    fn differentiate(&self, d_op: [usize; NUM_VARIABLES]) -> Self {
        let mut result = Self::zero();
        for (alpha, p_alpha) in &self.terms {
            for beta in multi_index_le(d_op) {
                let mut d_beta_p = p_alpha.differentiate(beta);
                if d_beta_p.is_zero() {
                    continue;
                }
                let c: usize = (0..NUM_VARIABLES)
                    .map(|i| binom(d_op[i], beta[i]))
                    .product();
                d_beta_p.scale_by(R::natural_inclusion(c));
                let gamma: [usize; NUM_VARIABLES] =
                    core::array::from_fn(|i| d_op[i] - beta[i] + alpha[i]);
                result
                    .terms
                    .entry(gamma)
                    .and_modify(|e| *e += d_beta_p.clone())
                    .or_insert(d_beta_p);
            }
        }
        result.terms.retain(|_, v| !v.is_zero());
        result
    }
}

impl<R: Ring, const NUM_VARIABLES: usize> DifferentialOperator<R, NUM_VARIABLES> {
    pub fn differential_act<ActedOn>(&self, rhs: &ActedOn) -> ActedOn
    where
        ActedOn: NVarDifferentiable<NUM_VARIABLES>
            + AddAssign<ActedOn>
            + Zero
            + LeftMul<PowerSeries<R, NUM_VARIABLES>, Output = ActedOn>,
    {
        let mut to_return = ActedOn::zero();
        for (diff_op, power_series_coeff) in &self.terms {
            to_return += rhs
                .differentiate(*diff_op)
                .left_mul(power_series_coeff.clone());
        }
        to_return
    }
}

impl<R: Ring, const NUM_VARIABLES: usize> AddAssign<Self>
    for DifferentialOperator<R, NUM_VARIABLES>
{
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

impl<R: Ring, const NUM_VARIABLES: usize> Add<Self> for DifferentialOperator<R, NUM_VARIABLES> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<R: Ring, const NUM_VARIABLES: usize> Zero for DifferentialOperator<R, NUM_VARIABLES> {
    fn zero() -> Self {
        Self {
            terms: HashMap::new(),
        }
    }

    fn is_zero(&self) -> bool {
        self.terms.iter().all(|(_, coeff)| coeff.is_zero())
    }
}

impl<R: Ring, const NUM_VARIABLES: usize> MulAssign<Self>
    for DifferentialOperator<R, NUM_VARIABLES>
{
    fn mul_assign(&mut self, rhs: Self) {
        let res = self.differential_act(&rhs);
        *self = res;
    }
}

impl<R: Ring, const NUM_VARIABLES: usize> Mul<Self> for DifferentialOperator<R, NUM_VARIABLES> {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self *= rhs;
        self
    }
}

impl<R: Ring, const NUM_VARIABLES: usize> One for DifferentialOperator<R, NUM_VARIABLES> {
    fn one() -> Self {
        let mut terms = HashMap::with_capacity(1);
        terms.insert([0usize; NUM_VARIABLES], PowerSeries::one());
        Self { terms }
    }
}

impl<R: Ring, const NUM_VARIABLES: usize> Neg for DifferentialOperator<R, NUM_VARIABLES> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.terms.values_mut().for_each(|z| {
            *z = -z.clone();
        });
        self
    }
}

impl<R: Ring, const NUM_VARIABLES: usize> SubAssign<Self>
    for DifferentialOperator<R, NUM_VARIABLES>
{
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

impl<R: Ring, const NUM_VARIABLES: usize> Sub<Self> for DifferentialOperator<R, NUM_VARIABLES> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self -= rhs;
        self
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{DifferentialOperator, NVarDifferentiable};
    use crate::plethystic::PowerSeries;

    // Operator: x²y·1 + 3x·D_x + y²·D_y
    fn sample_operator() -> DifferentialOperator<i64, 2> {
        DifferentialOperator {
            terms: HashMap::from([
                ([0, 0], PowerSeries::new(HashMap::from([([2, 1], 1i64)]))), // x²y
                ([1, 0], PowerSeries::new(HashMap::from([([1, 0], 3i64)]))), // 3x
                ([0, 1], PowerSeries::new(HashMap::from([([0, 2], 1i64)]))), // y²
            ]),
        }
    }

    // D^[1,0] · (x²y + 3x·D_x + y²·D_y)
    //
    // Term x²y (α=[0,0]), β ∈ {[0,0],[1,0]}:
    //   β=[0,0]: C=1, D^[0,0](x²y)=x²y,  γ=[1,0] → x²y·D_x
    //   β=[1,0]: C=1, D^[1,0](x²y)=2xy,  γ=[0,0] → 2xy
    //
    // Term 3x·D_x (α=[1,0]), β ∈ {[0,0],[1,0]}:
    //   β=[0,0]: C=1, D^[0,0](3x)=3x,    γ=[2,0] → 3x·D_x²
    //   β=[1,0]: C=1, D^[1,0](3x)=3,     γ=[1,0] → 3·D_x
    //
    // Term y²·D_y (α=[0,1]), β ∈ {[0,0],[1,0]}:
    //   β=[0,0]: C=1, D^[0,0](y²)=y²,    γ=[1,1] → y²·D_x D_y
    //   β=[1,0]: C=1, D^[1,0](y²)=0      (x-exp of y² is 0 < 1) → vanishes
    //
    // Result: 2xy + (x²y + 3)·D_x + 3x·D_x² + y²·D_x D_y
    #[test]
    fn differentiate_leibniz_with_vanishing_beta_contribution() {
        let op = sample_operator();
        let result = op.differentiate([1, 0]);
        let expected = DifferentialOperator {
            terms: HashMap::from([
                ([0, 0], PowerSeries::new(HashMap::from([([1, 1], 2i64)]))), // 2xy
                (
                    [1, 0],
                    PowerSeries::new(HashMap::from([([2, 1], 1i64), ([0, 0], 3i64)])),
                ), // x²y + 3
                ([2, 0], PowerSeries::new(HashMap::from([([1, 0], 3i64)]))), // 3x
                ([1, 1], PowerSeries::new(HashMap::from([([0, 2], 1i64)]))), // y²
            ]),
        };
        assert_eq!(result, expected);
    }

    // D^[0,0] is the identity: every β = [0,0] only, C=1, D^[0,0](p)=p, γ=α → unchanged.
    #[test]
    fn differentiate_by_zero_alpha_is_identity() {
        let op = sample_operator();
        let result = op.differentiate([0, 0]);
        assert_eq!(result, op);
    }

    // N=1 case: D^2 · (x · D_x), α=[1], p_α=x.
    // β ∈ {[0],[1],[2]}:
    //   β=[0]: C(2,0)=1, D^0(x)=x,  γ=[2+1]=[3] → x·D^3
    //   β=[1]: C(2,1)=2, D^1(x)=1,  γ=[1+1]=[2] → 2·D^2
    //   β=[2]: C(2,2)=1, D^2(x)=0   (x-exp 1 < 2) → vanishes
    // Result: x·D^3 + 2·D^2
    #[test]
    fn differentiate_univariate_beta_too_large_vanishes() {
        let op: DifferentialOperator<i64, 1> = DifferentialOperator {
            terms: HashMap::from([
                ([1], PowerSeries::new(HashMap::from([([1], 1i64)]))), // x·D_x
            ]),
        };
        let result = op.differentiate([2]);
        let expected: DifferentialOperator<i64, 1> = DifferentialOperator {
            terms: HashMap::from([
                ([3], PowerSeries::new(HashMap::from([([1], 1i64)]))), // x·D^3
                ([2], PowerSeries::new(HashMap::from([([0], 2i64)]))), // 2·D^2
            ]),
        };
        assert_eq!(result, expected);
    }

    // Weyl algebra relation: D_x · x = x·D_x + 1.
    //
    // self = 1·D_x  (α=[1], p_α=1)
    // rhs  = x·1   (α=[0], p_α=x)
    //
    // differential_act applies self to rhs:
    //   rhs.differentiate([1]) via Leibniz on (α=[0], p_α=x):
    //     β=[0]: C=1, D^0(x)=x, γ=[1] → x·D^[1]
    //     β=[1]: C=1, D^1(x)=1, γ=[0] → 1·D^[0]
    //   = x·D_x + 1
    //   then left_mul(1) → unchanged
    //
    // Result: x·D_x + 1
    #[test]
    fn mul_weyl_algebra_relation() {
        let d_x: DifferentialOperator<i64, 1> = DifferentialOperator {
            terms: HashMap::from([
                ([1], PowerSeries::new(HashMap::from([([0], 1i64)]))), // 1·D_x
            ]),
        };
        let x_op: DifferentialOperator<i64, 1> = DifferentialOperator {
            terms: HashMap::from([
                ([0], PowerSeries::new(HashMap::from([([1], 1i64)]))), // x·1
            ]),
        };
        let result = d_x * x_op;
        let expected: DifferentialOperator<i64, 1> = DifferentialOperator {
            terms: HashMap::from([
                ([1], PowerSeries::new(HashMap::from([([1], 1i64)]))), // x·D_x
                ([0], PowerSeries::new(HashMap::from([([0], 1i64)]))), // 1
            ]),
        };
        assert_eq!(result, expected);
    }

    // (x·D_x)² = x²·D_x² + x·D_x.
    //
    // self = rhs = x·D_x  (α=[1], p_α=x)
    //
    // differential_act applies self to rhs:
    //   rhs.differentiate([1]) via Leibniz on (α=[1], p_α=x):
    //     β=[0]: C=1, D^0(x)=x, γ=[2] → x·D^2
    //     β=[1]: C=1, D^1(x)=1, γ=[1] → 1·D^1
    //   = x·D_x² + D_x
    //   then left_mul(x): x·(x·D_x² + D_x) = x²·D_x² + x·D_x
    //
    // Result: x²·D_x² + x·D_x
    #[test]
    fn mul_x_dx_squared() {
        let x_dx: DifferentialOperator<i64, 1> = DifferentialOperator {
            terms: HashMap::from([
                ([1], PowerSeries::new(HashMap::from([([1], 1i64)]))), // x·D_x
            ]),
        };
        let result = x_dx.clone() * x_dx;
        let expected: DifferentialOperator<i64, 1> = DifferentialOperator {
            terms: HashMap::from([
                ([2], PowerSeries::new(HashMap::from([([2], 1i64)]))), // x²·D_x²
                ([1], PowerSeries::new(HashMap::from([([1], 1i64)]))), // x·D_x
            ]),
        };
        assert_eq!(result, expected);
    }

    // a†a for the quantum harmonic oscillator with ℏ=1, m=2.0, ω=3.0.
    //
    // With ℏ=1: a  = α·x + β·∂,  a† = α·x - β·∂
    //   where α = √(mω/2) = √3    ≈ 1.732
    //         β = 1/√(2mω) = 1/√12 ≈ 0.289
    //         αβ = 1/2  (always, for any m,ω — cancels exactly in IEEE 754)
    //
    // Term α_idx=[0], p=αx — rhs.differentiate([0]) = rhs, then left_mul(αx):
    //   D^[1] ← αβ·x,  D^[0] ← α²·x²
    //
    // Term α_idx=[1], p=-β — rhs.differentiate([1]):
    //   on β·D^[1]: β=[0]→β·D^[2], β=[1]→0
    //   on αx·1:    β=[0]→αx·D^[1], β=[1]→α·D^[0]
    //   = β·D^[2] + αx·D^[1] + α,  then left_mul(-β):
    //   D^[2] ← -β²,  D^[1] ← -αβ·x,  D^[0] ← -αβ
    //
    // D^[1] terms: αβx - αβx = 0  (canonical commutation, exact in float)
    // Result: -(1/2mω)·D_x² + (mω/2)·x² - 1/2
    //       = -(1/12)·D_x²  +  3·x²  -  0.5
    #[test]
    fn mul_a_dag_a_harmonic_oscillator() {
        let m = 2.0_f64;
        let omega = 3.0_f64;
        let alpha = (m * omega / 2.0).sqrt(); // √3   ≈ 1.732
        let beta = 1.0 / (2.0 * m * omega).sqrt(); // 1/√12 ≈ 0.289

        let a_dag: DifferentialOperator<f64, 1> = DifferentialOperator {
            terms: HashMap::from([
                ([0], PowerSeries::new(HashMap::from([([1], alpha)]))), // αx
                ([1], PowerSeries::new(HashMap::from([([0], -beta)]))), // -β∂
            ]),
        };
        let a: DifferentialOperator<f64, 1> = DifferentialOperator {
            terms: HashMap::from([
                ([0], PowerSeries::new(HashMap::from([([1], alpha)]))), // αx
                ([1], PowerSeries::new(HashMap::from([([0], beta)]))),  // β∂
            ]),
        };
        let result = a_dag * a;
        let expected: DifferentialOperator<f64, 1> = DifferentialOperator {
            terms: HashMap::from([
                ([2], PowerSeries::new(HashMap::from([([0], -beta * beta)]))),
                (
                    [0],
                    PowerSeries::new(HashMap::from([([2], alpha * alpha), ([0], -alpha * beta)])),
                ),
            ]),
        };
        assert_eq!(result, expected);
    }
}
