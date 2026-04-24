use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use geometric_rep_theory::arithmetic_utils::Ring;
use geometric_rep_theory::infinity_algebra::{
    ExteriorPower, GradedModule, LInfinityAlgebra, OperationTree,
};

// R^3 (degree 0) + R (degree -1) as a graded module:
//   l_1 = 0
//   l_2(a, b) = a x b             (cross product, \wedge^2 R^3 -> R^3)
//   l_3(a, b, c) = a.(b x c) 1_R  (scalar triple product into the R summand)
#[derive(Clone, PartialEq)]
struct R3R<C: Ring> {
    vec: [C; 3],
    scalar: C,
}

impl<C: Ring> R3R<C> {
    fn from_vec(v: [C; 3]) -> Self {
        Self {
            vec: v,
            scalar: C::zero(),
        }
    }

    fn from_scalar(s: C) -> Self {
        Self {
            vec: [C::zero(), C::zero(), C::zero()],
            scalar: s,
        }
    }
}

impl<C: Ring> Add for R3R<C> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        self += rhs;
        self
    }
}

impl<C: Ring> AddAssign for R3R<C> {
    fn add_assign(&mut self, rhs: Self) {
        let Self {
            vec: [b0, b1, b2],
            scalar: bs,
        } = rhs;
        self.vec[0] += b0;
        self.vec[1] += b1;
        self.vec[2] += b2;
        self.scalar += bs;
    }
}

impl<C: Ring> Sub for R3R<C> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        self -= rhs;
        self
    }
}

impl<C: Ring> SubAssign for R3R<C> {
    fn sub_assign(&mut self, rhs: Self) {
        let Self {
            vec: [b0, b1, b2],
            scalar: bs,
        } = rhs;
        self.vec[0] -= b0;
        self.vec[1] -= b1;
        self.vec[2] -= b2;
        self.scalar -= bs;
    }
}

impl<C: Ring> Neg for R3R<C> {
    type Output = Self;
    fn neg(self) -> Self {
        let Self {
            vec: [v0, v1, v2],
            scalar,
        } = self;
        Self {
            vec: [-v0, -v1, -v2],
            scalar: -scalar,
        }
    }
}

impl<C: Ring> Mul<C> for R3R<C> {
    type Output = Self;
    fn mul(mut self, c: C) -> Self {
        self *= c;
        self
    }
}

impl<C: Ring> MulAssign<C> for R3R<C> {
    fn mul_assign(&mut self, c: C) {
        self.vec[0] *= c.clone();
        self.vec[1] *= c.clone();
        self.vec[2] *= c.clone();
        self.scalar *= c;
    }
}

impl<C: Ring> GradedModule<C> for R3R<C> {
    type Ctx = ();

    fn extract_homogeneous(self, n: i64) -> (Self, Option<Self>) {
        let R3R { vec, scalar } = self;
        match n {
            0 => (Self::from_vec(vec), Some(Self::from_scalar(scalar))),
            -1 => (Self::from_scalar(scalar), Some(Self::from_vec(vec))),
            _ => (Self::zero(()), Some(R3R { vec, scalar })),
        }
    }

    fn zero((): ()) -> Self {
        R3R {
            vec: [C::zero(), C::zero(), C::zero()],
            scalar: C::zero(),
        }
    }

    fn ctx(&self) -> () {}
}

fn cross<C: Ring>(a: &[C; 3], b: &[C; 3]) -> [C; 3] {
    [
        a[1].clone() * b[2].clone() - a[2].clone() * b[1].clone(),
        a[2].clone() * b[0].clone() - a[0].clone() * b[2].clone(),
        a[0].clone() * b[1].clone() - a[1].clone() * b[0].clone(),
    ]
}

fn dot<C: Ring>(a: &[C; 3], b: &[C; 3]) -> C {
    a[0].clone() * b[0].clone() + a[1].clone() * b[1].clone() + a[2].clone() * b[2].clone()
}

impl<C: Ring> LInfinityAlgebra<C> for R3R<C> {
    fn max_nonzero_arity() -> Option<usize> {
        Some(3)
    }

    fn l_n_one_term_owned<const N: usize>(inputs: [Self; N]) -> Self {
        let mut iter = inputs.into_iter();
        match N {
            0 | 1 => Self::zero(()),
            2 => {
                let a = iter.next().unwrap();
                let b = iter.next().unwrap();
                Self::from_vec(cross(&a.vec, &b.vec))
            }
            3 => {
                let a = iter.next().unwrap();
                let b = iter.next().unwrap();
                let c = iter.next().unwrap();
                Self::from_scalar(dot(&a.vec, &cross(&b.vec, &c.vec)))
            }
            _ => Self::zero(()),
        }
    }

    fn l_n_one_term<const N: usize>(inputs: [&Self; N]) -> Self {
        let slice: &[&Self] = &inputs;
        match slice {
            [] | [_] => Self::zero(()),
            [a, b] => Self::from_vec(cross(&a.vec, &b.vec)),
            [a, b, c] => Self::from_scalar(dot(&a.vec, &cross(&b.vec, &c.vec))),
            _ => Self::zero(()),
        }
    }
}

pub fn l_infinity_cross_product_example() {
    let e1 = R3R::<i64>::from_vec([1, 0, 0]);
    let e2 = R3R::<i64>::from_vec([0, 1, 0]);
    let e3 = R3R::<i64>::from_vec([0, 0, 1]);

    // l_2(e_1, e_2) = e_1 x e_2 = e_3
    let wedge2 = ExteriorPower::from_pure_wedge([e1.clone(), e2.clone()], 1_i64);
    let result2 = R3R::<i64>::l_n((), wedge2);
    assert!(result2 == e3.clone());
    println!("l_2(e_1, e_2) = e_3: ok");

    // l_3(e_1, e_2, e_3) = e_1 . (e_2 x e_3) . 1_R = 1_R
    let mul_a = 7;
    let mul_b = 4;
    let mul_c = -2;
    let one_r = R3R::<i64>::from_scalar(1_i64);
    let wedge3 = ExteriorPower::from_pure_wedge([e1 * mul_a, e2 * mul_b, e3 * mul_c], 1_i64);
    let result3 = R3R::<i64>::l_n((), wedge3);
    assert!(result3 == one_r * (mul_a * mul_b * mul_c));
    println!(
        "l_3(e_1*{mul_a}, e_2*{mul_b}, e_3*{mul_c}) = {}*1_R: ok",
        mul_a * mul_b * mul_c
    );
}

pub fn l_infinity_tree_example() {
    let e1 = R3R::<i64>::from_vec([1, 0, 0]);
    let e2 = R3R::<i64>::from_vec([0, 1, 0]);
    let e3 = R3R::<i64>::from_vec([0, 0, 1]);
    let r = R3R::<i64>::from_scalar(1_i64);

    // tree: l_3(leaf_0, leaf_1, l_2(leaf_2, leaf_3))
    // applied to [a, b, c, d] this gives a . (b x (c x d)) . 1_R
    let tree = OperationTree::node(vec![
        OperationTree::leaf(),
        OperationTree::leaf(),
        OperationTree::node(vec![OperationTree::leaf(), OperationTree::leaf()]),
    ]);

    // General element of \wedge^4(R3R) — a sum of pure wedges.
    // The third term mixes the R^3 and R parts by placing e1 + 5r and e2 - 3r in slots 0 and 1.
    let combination: ExteriorPower<R3R<i64>, i64, 4> =
        ExteriorPower::from_pure_wedge(
            [
                e1.clone(),
                e2.clone(),
                e1.clone(),
                e2.clone() + e1.clone() * 6,
            ],
            3_i64,
        ) + ExteriorPower::from_pure_wedge([e2.clone(), e3.clone(), e2.clone(), e3.clone()], 2_i64)
            + ExteriorPower::from_pure_wedge(
                [
                    e1.clone() + r.clone() * 5,
                    e2.clone() - r.clone() * 3,
                    e1.clone(),
                    e2.clone(),
                ],
                1_i64,
            );

    // By the vector triple product identity b x (c x d) = c(b.d) - d(b.c):
    //   a . (b x (c x d)) = (a.c)(b.d) - (a.d)(b.c)
    // The R-part of any input is invisible to l_n (it only reads .vec), so:
    //   Term 1: [e1, e2, e1, e2+6e1] * 3  -> (1)(1) - (6)(0) = 1, times 3
    //   Term 2: [e2, e3, e2, e3] * 2  -> (1)(1) - (0)(0) = 1, times 2
    //   Term 3: [e1+5r, e2-3r, e1, e2] * 1  -> similar to term 1 -> 1, times 1
    //   Total: 6 * 1_R
    let result = R3R::<i64>::evaluate((), &tree, combination).unwrap();
    assert!(result == R3R::<i64>::from_scalar(6));
    println!("l_3(a, b, l_2(c, d)) over 3-term sum in \\wedge^4(R3R) = 6 * 1_R: ok");
}

pub fn main() {
    l_infinity_cross_product_example();
    l_infinity_tree_example();
}
