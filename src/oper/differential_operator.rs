use std::collections::HashMap;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num::{One, Zero};

use crate::arithmetic_utils::Ring;
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
    fn differentiate(&self, _d_op: [usize; NUM_VARIABLES]) -> Self {
        todo!("Differentiate through d_op sum p(x) D^alpha into sum q(x) D^gamma form")
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
    #[allow(unreachable_code, unused_variables)]
    fn mul_assign(&mut self, rhs: Self) {
        panic!("This requires differential_act which requires differentiate which is todo");
        let res = self.differential_act(&rhs);
        *self = res;
    }
}

impl<R: Ring, const NUM_VARIABLES: usize> Mul<Self> for DifferentialOperator<R, NUM_VARIABLES> {
    type Output = Self;

    #[allow(unused_mut, unreachable_code, unused_variables)]
    fn mul(mut self, rhs: Self) -> Self::Output {
        panic!("This requires differential_act which requires differentiate which is todo");
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
