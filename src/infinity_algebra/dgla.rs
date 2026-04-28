use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::arithmetic_utils::Ring;
use crate::infinity_algebra::LInfinityAlgebra;
use crate::infinity_algebra::graded_module::GradedModule;

pub trait Lie {
    /// The implementor is responsible for it being bilinear
    /// and obeying antisymmetry or symmetery as appropriate
    /// as well as it's Jacobi identity
    #[must_use = "returns the algebra element resulting from taking the bracket; discarding it loses the computed value"]
    fn bracket(self, rhs: Self) -> Self;
}

/// A differential graded lie algebra wrapping an underlying lie algebra `A` with a
/// differential given by `F: Fn(A) -> A`. The `Arc<F>` is the `Ctx` for this
/// type, shared among all elements of the same algebra instance.
///
/// - `l_1`: the differential `F`
/// - `l_2`: the bracket on `A`
/// - `l_n` for `n ≥ 3`: zero
///
/// The implementor is responsible for `differential`
/// needing to obey `d^2 = 0`, going between correct degrees
/// and the Leibniz rule on homogeneous elements
#[derive(Clone, Copy)]
pub struct DGLA<A> {
    pub value: A,
    pub differential: fn(A) -> A,
}

impl<A> PartialEq for DGLA<A>
where
    A: Lie + Clone + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<A> Eq for DGLA<A> where A: Lie + Clone + Eq {}

impl<A> Add for DGLA<A>
where
    A: Lie + Clone + Add<Output = A>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            differential: self.differential,
        }
    }
}

impl<A> AddAssign for DGLA<A>
where
    A: Lie + Clone + AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl<A> Sub for DGLA<A>
where
    A: Lie + Clone + Sub<Output = A>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            differential: self.differential,
        }
    }
}

impl<A> SubAssign for DGLA<A>
where
    A: Lie + Clone + SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl<A> Neg for DGLA<A>
where
    A: Lie + Clone + Neg<Output = A>,
{
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            differential: self.differential,
        }
    }
}

impl<A, Coeffs: Ring> Mul<Coeffs> for DGLA<A>
where
    A: Lie + Clone + Mul<Coeffs, Output = A>,
{
    type Output = Self;
    fn mul(self, coeff: Coeffs) -> Self {
        Self {
            value: self.value * coeff,
            differential: self.differential,
        }
    }
}

impl<A, Coeffs: Ring> MulAssign<Coeffs> for DGLA<A>
where
    A: Lie + Clone + MulAssign<Coeffs>,
{
    fn mul_assign(&mut self, coeff: Coeffs) {
        self.value *= coeff;
    }
}

impl<A, Coeffs> GradedModule<Coeffs> for DGLA<A>
where
    A: Lie + Clone + GradedModule<Coeffs, Ctx = ()>,
    Coeffs: Ring,
{
    type Ctx = fn(A) -> A;
    fn extract_homogeneous(self, n: i64) -> (Self, Option<Self>) {
        let ctx = self.differential;
        let (homo, rest) = self.value.extract_homogeneous(n);
        let homo_dga = Self {
            value: homo,
            differential: ctx,
        };
        let rest_dga = rest.map(|v| Self {
            value: v,
            differential: ctx,
        });
        (homo_dga, rest_dga)
    }

    fn zero(ctx: Self::Ctx) -> Self {
        Self {
            value: A::zero(()),
            differential: ctx,
        }
    }

    fn ctx(&self) -> Self::Ctx {
        self.differential
    }
}

impl<A> Lie for DGLA<A>
where
    A: Lie + Clone,
{
    fn bracket(self, rhs: Self) -> Self {
        Self {
            value: self.value.bracket(rhs.value),
            differential: self.differential,
        }
    }
}

impl<A> DGLA<A>
where
    A: Lie + Clone + Eq,
{
    #[must_use = "returns the algebra element resulting from taking the differential; discarding it loses the computed value"]
    pub fn diff(self) -> Self {
        let value = (self.differential)(self.value);
        Self {
            value,
            differential: self.differential,
        }
    }

    /// Send `x` to `dx + 1/2 [x,x]`
    /// This also does not enforce that `x` is of degree `\pm 1`
    #[must_use = "This just computes the LHS of the Maurer-Cartan equation with particular x. The caller makes the decision about this being a solution and how to use solutions."]
    pub fn maurer_cartan<Coeffs: Ring>(self) -> Self
    where
        Self: std::ops::Div<Coeffs, Output = Self> + AddAssign<Self>,
    {
        let mut to_return = self.clone().diff();
        let two = Coeffs::natural_inclusion(2);
        let next = self.clone().bracket(self) / two;
        to_return += next;
        to_return
    }
}

impl<A, Coeffs> LInfinityAlgebra<Coeffs> for DGLA<A>
where
    A: Lie + Clone + GradedModule<Coeffs, Ctx = ()>,
    Coeffs: Ring,
{
    fn max_nonzero_arity() -> Option<usize> {
        Some(2)
    }

    fn l_n_one_term_owned<const N: usize>(inputs: [Self; N]) -> Self {
        let mut iter = inputs.into_iter();
        match N {
            0 => unreachable!("N=0 (curvature) is not supported"),
            1 => {
                let input = iter.next().unwrap();
                let ctx = input.differential;
                Self {
                    value: (ctx)(input.value),
                    differential: ctx,
                }
            }
            2 => {
                let lhs = iter.next().unwrap();
                let rhs = iter.next().unwrap();
                let ctx = lhs.differential;
                Self {
                    value: lhs.value.bracket(rhs.value),
                    differential: ctx,
                }
            }
            _ => {
                let first = iter.next().unwrap();
                let ctx = first.differential;
                Self {
                    value: A::zero(()),
                    differential: ctx,
                }
            }
        }
    }

    fn l_n_one_term<const N: usize>(inputs: [&Self; N]) -> Self {
        let slice: &[&Self] = &inputs;
        match slice {
            [] => unreachable!("N=0 (curvature) is not supported"),
            [input] => {
                let ctx = input.differential;
                Self {
                    value: (ctx)(input.value.clone()),
                    differential: ctx,
                }
            }
            [lhs, rhs] => {
                let ctx = lhs.differential;
                Self {
                    value: lhs.value.clone().bracket(rhs.value.clone()),
                    differential: ctx,
                }
            }
            [first, ..] => {
                let ctx = first.differential;
                Self {
                    value: A::zero(()),
                    differential: ctx,
                }
            }
        }
    }
}
