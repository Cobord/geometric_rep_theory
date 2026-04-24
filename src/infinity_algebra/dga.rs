use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::a_infinity::AInfinityAlgebra;
use crate::arithmetic_utils::Ring;
use crate::infinity_algebra::graded_module::GradedModule;

/// A differential graded algebra wrapping an underlying algebra `A` with a
/// differential given by `F: Fn(A) -> A`. The `Arc<F>` is the `Ctx` for this
/// type, shared among all elements of the same algebra instance.
///
/// - `m_1`: the differential `F`
/// - `m_2`: the multiplication on `A`
/// - `m_n` for `n ≥ 3`: zero
#[derive(Clone, Copy)]
pub struct DGA<A>
where
    A: Mul<A, Output = A>,
{
    pub value: A,
    pub differential: fn(A) -> A,
}

impl<A> PartialEq for DGA<A>
where
    A: Mul<A, Output = A> + Clone + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<A> Eq for DGA<A> where A: Mul<A, Output = A> + Clone + Eq {}

impl<A> Add for DGA<A>
where
    A: Mul<A, Output = A> + Clone + Add<Output = A>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            differential: self.differential,
        }
    }
}

impl<A> AddAssign for DGA<A>
where
    A: Mul<A, Output = A> + Clone + AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl<A> Sub for DGA<A>
where
    A: Mul<A, Output = A> + Clone + Sub<Output = A>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            differential: self.differential,
        }
    }
}

impl<A> SubAssign for DGA<A>
where
    A: Mul<A, Output = A> + Clone + SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl<A> Neg for DGA<A>
where
    A: Mul<A, Output = A> + Clone + Neg<Output = A>,
{
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            differential: self.differential,
        }
    }
}

impl<A, Coeffs: Ring> Mul<Coeffs> for DGA<A>
where
    A: Mul<A, Output = A> + Clone + Mul<Coeffs, Output = A>,
{
    type Output = Self;
    fn mul(self, coeff: Coeffs) -> Self {
        Self {
            value: self.value * coeff,
            differential: self.differential,
        }
    }
}

impl<A, Coeffs: Ring> MulAssign<Coeffs> for DGA<A>
where
    A: Mul<A, Output = A> + Clone + MulAssign<Coeffs>,
{
    fn mul_assign(&mut self, coeff: Coeffs) {
        self.value *= coeff;
    }
}

impl<A, Coeffs> GradedModule<Coeffs> for DGA<A>
where
    A: Mul<A, Output = A> + Clone + GradedModule<Coeffs, Ctx = ()>,
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

impl<A> Mul<DGA<A>> for DGA<A>
where
    A: Mul<A, Output = A> + Clone,
{
    type Output = Self;

    fn mul(self, rhs: DGA<A>) -> Self::Output {
        Self {
            value: self.value * rhs.value,
            differential: self.differential,
        }
    }
}

impl<A> DGA<A>
where
    A: Mul<A, Output = A> + Clone + Eq,
{
    #[must_use = "returns the algebra element resulting from taking the differential; discarding it loses the computed value"]
    pub fn diff(self) -> Self {
        let value = (self.differential)(self.value);
        Self {
            value,
            differential: self.differential,
        }
    }
}

impl<A, Coeffs> AInfinityAlgebra<Coeffs> for DGA<A>
where
    A: Mul<A, Output = A> + Clone + GradedModule<Coeffs, Ctx = ()>,
    Coeffs: Ring,
{
    fn max_nonzero_arity() -> Option<usize> {
        Some(2)
    }

    fn m_n_one_term_owned<const N: usize>(inputs: [Self; N]) -> Self {
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
                    value: lhs.value * rhs.value,
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

    fn m_n_one_term<const N: usize>(inputs: [&Self; N]) -> Self {
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
                    value: lhs.value.clone() * rhs.value.clone(),
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

/// `Coeffs[x,y]/(x^3,y^2)`
/// |y| = 1, |x| = 2
/// d(y) = x d(x) = 0
#[cfg(test)]
mod test_6d_example {
    use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

    use crate::{
        arithmetic_utils::Ring,
        infinity_algebra::{DGA, graded_module::GradedModule},
    };

    #[derive(PartialEq, Eq, Clone, Copy)]
    pub struct SmallGA<Coeffs: Ring>([Coeffs; 6]);

    impl<Coeffs: Ring> AddAssign for SmallGA<Coeffs> {
        fn add_assign(&mut self, rhs: Self) {
            self.0.iter_mut().zip(rhs.0).for_each(|(z, w)| {
                *z += w;
            });
        }
    }

    impl<Coeffs: Ring> Add for SmallGA<Coeffs> {
        type Output = Self;

        fn add(mut self, rhs: Self) -> Self::Output {
            self += rhs;
            self
        }
    }

    impl<Coeffs: Ring> SubAssign for SmallGA<Coeffs> {
        fn sub_assign(&mut self, rhs: Self) {
            self.0.iter_mut().zip(rhs.0).for_each(|(z, w)| {
                *z -= w;
            });
        }
    }

    impl<Coeffs: Ring> Sub for SmallGA<Coeffs> {
        type Output = Self;

        fn sub(mut self, rhs: Self) -> Self::Output {
            self -= rhs;
            self
        }
    }

    impl<Coeffs: Ring> Neg for SmallGA<Coeffs> {
        type Output = Self;

        fn neg(mut self) -> Self::Output {
            self.0.iter_mut().for_each(|z| *z = -z.clone());
            self
        }
    }

    impl<Coeffs: Ring> Mul for SmallGA<Coeffs> {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            let self_1 = self.0[0].clone();
            let self_y = self.0[1].clone();
            let self_x = self.0[2].clone();
            let self_xy = self.0[3].clone();
            let self_x2 = self.0[4].clone();
            let self_x2y = self.0[5].clone();

            let rhs_1 = rhs.0[0].clone();
            let rhs_y = rhs.0[1].clone();
            let rhs_x = rhs.0[2].clone();
            let rhs_xy = rhs.0[3].clone();
            let rhs_x2 = rhs.0[4].clone();
            let rhs_x2y = rhs.0[5].clone();

            let mut to_return = core::array::from_fn(|_| Coeffs::zero());
            to_return[0] = self_1.clone() * rhs_1.clone();
            to_return[1] = self_1.clone() * rhs_y.clone() + self_y.clone() * rhs_1.clone();
            to_return[2] = self_1.clone() * rhs_x.clone() + self_x.clone() * rhs_1.clone();
            to_return[3] = self_1.clone() * rhs_xy.clone()
                + self_xy.clone() * rhs_1.clone()
                + self_x.clone() * rhs_y.clone()
                + self_y.clone() * rhs_x.clone();
            to_return[4] = self_1.clone() * rhs_x2.clone()
                + self_x2.clone() * rhs_1.clone()
                + self_x.clone() * rhs_x.clone();
            to_return[5] = self_1 * rhs_x2y
                + self_x2y * rhs_1
                + self_x2 * rhs_y
                + self_y * rhs_x2
                + self_x * rhs_xy
                + self_xy * rhs_x;

            Self(to_return)
        }
    }

    impl<Coeffs: Ring> MulAssign<Coeffs> for SmallGA<Coeffs> {
        fn mul_assign(&mut self, rhs: Coeffs) {
            self.0.iter_mut().for_each(|z| *z *= rhs.clone());
        }
    }

    impl<Coeffs: Ring> Mul<Coeffs> for SmallGA<Coeffs> {
        type Output = Self;

        fn mul(mut self, rhs: Coeffs) -> Self::Output {
            self *= rhs;
            self
        }
    }

    impl<Coeffs: Ring> GradedModule<Coeffs> for SmallGA<Coeffs> {
        type Ctx = ();
        fn extract_homogeneous(mut self, n: i64) -> (Self, Option<Self>) {
            if n < 0 {
                (SmallGA::zero(()), Some(self))
            } else if n > 6 {
                (SmallGA::zero(()), Some(self))
            } else {
                let n = n as usize;
                let mut just_n = SmallGA::zero(());
                just_n.0[n] = self.0[n].clone();
                self.0[n] = Coeffs::zero();
                if self.0.iter().all(Coeffs::is_zero) {
                    (just_n, None)
                } else {
                    (just_n, Some(self))
                }
            }
        }

        fn zero((): ()) -> Self {
            Self(core::array::from_fn(|_| Coeffs::zero()))
        }

        fn ctx(&self) -> Self::Ctx {}
    }

    fn diff<Coeffs: Ring>(x: SmallGA<Coeffs>) -> SmallGA<Coeffs> {
        let mut to_return = core::array::from_fn(|_| Coeffs::zero());
        to_return[2] = x.0[1].clone();
        to_return[4] = x.0[3].clone();
        SmallGA(to_return)
    }

    fn small_dga_one<Coeffs: Ring>() -> DGA<SmallGA<Coeffs>> {
        let mut to_return = core::array::from_fn(|_| Coeffs::zero());
        to_return[0] = Coeffs::one();
        DGA {
            value: SmallGA(to_return),
            differential: diff,
        }
    }

    fn small_dga_y<Coeffs: Ring>() -> DGA<SmallGA<Coeffs>> {
        let mut to_return = core::array::from_fn(|_| Coeffs::zero());
        to_return[1] = Coeffs::one();
        DGA {
            value: SmallGA(to_return),
            differential: diff,
        }
    }

    fn small_dga_x<Coeffs: Ring>() -> DGA<SmallGA<Coeffs>> {
        let mut to_return = core::array::from_fn(|_| Coeffs::zero());
        to_return[2] = Coeffs::one();
        DGA {
            value: SmallGA(to_return),
            differential: diff,
        }
    }

    fn small_dga_xy<Coeffs: Ring>() -> DGA<SmallGA<Coeffs>> {
        let mut to_return = core::array::from_fn(|_| Coeffs::zero());
        to_return[3] = Coeffs::one();
        DGA {
            value: SmallGA(to_return),
            differential: diff,
        }
    }

    fn small_dga_x2<Coeffs: Ring>() -> DGA<SmallGA<Coeffs>> {
        let mut to_return = core::array::from_fn(|_| Coeffs::zero());
        to_return[4] = Coeffs::one();
        DGA {
            value: SmallGA(to_return),
            differential: diff,
        }
    }

    fn small_dga_x2y<Coeffs: Ring>() -> DGA<SmallGA<Coeffs>> {
        let mut to_return = core::array::from_fn(|_| Coeffs::zero());
        to_return[5] = Coeffs::one();
        DGA {
            value: SmallGA(to_return),
            differential: diff,
        }
    }

    #[test]
    fn zero_times() {
        let zero_dga = DGA::<SmallGA<i64>>::zero(diff);
        let one_dga = small_dga_one();
        assert!(one_dga.clone() * one_dga.clone() - one_dga.clone() == zero_dga.clone());
        assert!(one_dga.clone().diff() == zero_dga.clone());
        let y_dga = small_dga_y();
        let x_dga = small_dga_x();
        let xy_dga = small_dga_xy();
        let x2_dga = small_dga_x2();
        let x2y_dga = small_dga_x2y();
        assert!(y_dga * zero_dga == zero_dga);
        assert!(x_dga * zero_dga == zero_dga);
        assert!(xy_dga * zero_dga == zero_dga);
        assert!(x2_dga * zero_dga == zero_dga);
        assert!(x2y_dga * zero_dga == zero_dga);

        assert!(zero_dga * y_dga == zero_dga);
        assert!(zero_dga * x_dga == zero_dga);
        assert!(zero_dga * xy_dga == zero_dga);
        assert!(zero_dga * x2_dga == zero_dga);
        assert!(zero_dga * x2y_dga == zero_dga);

        assert!(x_dga * x2y_dga == zero_dga);
        assert!(x_dga * x2_dga == zero_dga);
        assert!(y_dga * y_dga == zero_dga);
        assert!(y_dga * xy_dga == zero_dga);

        assert!(y_dga.diff() * xy_dga == x2y_dga);
    }

    #[test]
    fn evaluate_two_level_tree() {
        use crate::infinity_algebra::tensor_power::TensorPower;
        use crate::infinity_algebra::{AInfinityAlgebra, OperationTree};

        // tree: m_2(leaf, m_2(leaf, leaf))
        // The inner m_2 node sits at absolute range 1..3.
        // A sub-slicing bug would pass inputs[1..3] to the inner node
        // and then try to index that sub-slice with the absolute ranges
        // 1..2 and 2..3, giving wrong values or panicking.
        let tree = OperationTree::node(vec![
            OperationTree::leaf(),
            OperationTree::node(vec![OperationTree::leaf(), OperationTree::leaf()]),
        ]);
        assert_eq!(tree.num_leaves(), 3);

        // m_2(x, m_2(y, 1)) = x * (y * 1) = x * y = xy
        let combination =
            TensorPower::from_pure_tensor([small_dga_x(), small_dga_y(), small_dga_one()], 1i64);
        let result = DGA::<SmallGA<i64>>::evaluate(diff, &tree, combination).unwrap();
        assert!(result == small_dga_xy());
    }
}
