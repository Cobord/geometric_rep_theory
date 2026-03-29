#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::fmt::{Debug, Display};

use crate::toric::cone::RationalPolyhedralCone;
use crate::toric::fan::ToricFan;
use crate::toric::integer_arith::{gcd, kernel_from_snf};

/// Errors returned by toric coordinate ring computations.
#[derive(PartialEq, Eq, Debug)]
pub enum CoordinateRingError {
    /// The fan has more than one cone and is not affine.
    NotAffine,
    /// The fan is affine but the global support cone could not be constructed.
    NotManifestlyAffine,
    /// The cone is not simplicial (has more rays than its dimension).
    NonSimplicialCone,
    /// The number of rays does not match the const generic `N`.
    DimensionMismatch { expected: usize, actual: usize },
    /// An internal linear algebra step (SNF computation or kernel extraction) failed.
    LinearAlgebraFailure,
    /// The cone may not be minimally presented, so the ray count is unreliable.
    PossiblyNonMinimalCone,
}

/// A monomial in `N` variables, stored as an array of non-negative integer exponents.
#[derive(Clone, PartialEq, Eq)]
pub struct Monomial<const N: usize> {
    pub exponents: [u32; N],
}

impl<const N: usize> Monomial<N> {
    fn new_vars<const N_PLUS_MORE: usize>(self) -> Monomial<N_PLUS_MORE> {
        Monomial {
            exponents: core::array::from_fn(|idx| if idx < N { self.exponents[idx] } else { 0 }),
        }
    }
}

impl<const N: usize> Display for Monomial<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;
        for (i, &exp) in self.exponents.iter().enumerate() {
            if exp != 0 {
                if !first {
                    write!(f, "*")?;
                }
                write!(f, "x{}^{}", i + 1, exp)?;
                first = false;
            }
        }
        if first {
            write!(f, "1")?;
        }
        Ok(())
    }
}

/// A binomial relation `positive - negative` in the toric ideal.
#[derive(Clone, PartialEq, Eq)]
pub struct Binomial<const N: usize> {
    pub positive: Monomial<N>,
    pub negative: Monomial<N>,
}

impl<const N: usize> Display for Binomial<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} - {}", self.positive, self.negative)
    }
}

impl<const N: usize> Binomial<N> {
    fn new_vars<const N_PLUS_MORE: usize>(self) -> Binomial<N_PLUS_MORE> {
        Binomial {
            positive: self.positive.new_vars(),
            negative: self.negative.new_vars(),
        }
    }
}

impl<const N: usize> Binomial<N> {
    fn smaller_larger(&self) -> (&Monomial<N>, &Monomial<N>) {
        let lhs_sum: u32 = self.positive.exponents.iter().sum();
        let rhs_sum: u32 = self.negative.exponents.iter().sum();
        if lhs_sum <= rhs_sum {
            (&self.positive, &self.negative)
        } else {
            (&self.negative, &self.positive)
        }
    }
}

/// A representation of elements in the coordinate ring, used to evaluate monomials.
///
/// Implementors decide how to represent a ring element concretely (e.g. as an exponent vector).
pub trait CoordinateRingRepr: Clone + Eq {
    /// Return `self` raised to `power`.
    #[must_use = "This gives the raised power representation consuming self"]
    fn raise_power(&self, power: u32) -> Self;
    /// Return the product of `self` and `other`.
    #[must_use = "This gives the multiplied representation consuming self and other"]
    fn multiply(self, other: Self) -> Self;
}

/// The default coordinate ring representation: an integer exponent vector (column of the ray matrix).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DefaultRepr(DVector<i64>);

impl DefaultRepr {
    /// Construct a `DefaultRepr` from a raw integer vector.
    #[must_use = "Did you mean to use the created DefaultRepr?"]
    pub fn from_vec(v: Vec<i64>) -> Self {
        DefaultRepr(DVector::from_vec(v))
    }
}

impl CoordinateRingRepr for DefaultRepr {
    fn raise_power(&self, power: u32) -> Self {
        DefaultRepr(self.0.map(|x| x * i64::from(power)))
    }

    fn multiply(self, other: Self) -> Self {
        DefaultRepr(self.0 + other.0)
    }
}

/// A presentation of an affine toric coordinate ring `k[x₁,…,xₙ]/I` by binomial generators.
///
/// `generators` is the list of binomial relations `p⁺ - p⁻ = 0`, and `representing_vars` gives
/// a concrete value in `Repr` for each of the `N` coordinate ring generators.
#[derive(Clone)]
pub struct CoordinateRingPresentation<const N: usize, Repr: CoordinateRingRepr> {
    pub generators: Vec<Binomial<N>>,
    representing_vars: [Repr; N],
}

impl<const N: usize, U: CoordinateRingRepr> CoordinateRingPresentation<N, U> {
    /// Convert the representation type by applying `converter` to each of the `N` representing vars.
    pub fn conversion<T>(self, converter: fn(U) -> T) -> CoordinateRingPresentation<N, T>
    where
        T: CoordinateRingRepr,
    {
        CoordinateRingPresentation::<N, T> {
            generators: self.generators,
            representing_vars: self.representing_vars.map(converter),
        }
    }

    /// Verify that every binomial generator evaluates to zero under the representation `U`.
    ///
    /// For each generator `p⁺ - p⁻`, checks that the product of `representing_vars[i]`
    /// raised to the power `e⁺ᵢ` equals the product raised to `e⁻ᵢ`, using
    /// `CoordinateRingRepr::multiply` and `raise_power`.
    ///
    /// # Errors
    ///
    /// Returns `Err(failures)` where `failures` lists every binomial that does not evaluate to
    /// zero.
    ///
    /// # Panics
    ///
    /// Panics if `N == 0` (the const generic must be at least 1 so that `reduce` has an initial
    /// element).
    pub fn check(&self) -> Result<(), Vec<Binomial<N>>> {
        let mut failures = vec![];
        for cur_binom in &self.generators {
            let lhs = cur_binom
                .positive
                .exponents
                .iter()
                .enumerate()
                .map(|(i, power)| self.representing_vars[i].clone().raise_power(*power))
                .reduce(CoordinateRingRepr::multiply)
                .expect("N is at least 1");
            let rhs = cur_binom
                .negative
                .exponents
                .iter()
                .enumerate()
                .map(|(i, power)| self.representing_vars[i].clone().raise_power(*power))
                .reduce(CoordinateRingRepr::multiply)
                .expect("N is at least 1");
            if lhs != rhs {
                failures.push(cur_binom.clone());
            }
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures)
        }
    }

    /// Embed this presentation into a larger ring with `MORE` additional variables.
    ///
    /// All binomial generators are extended to `N_PLUS_MORE` variables (new exponents are zero).
    /// If `after_or_before` is `true`, the `MORE` new variables `new_us` are appended after the
    /// existing `N` variables; if `false`, they are prepended before them.
    pub fn extend_irrelevant<const MORE: usize, const N_PLUS_MORE: usize>(
        self,
        new_us: &[U; MORE],
        after_or_before: bool,
    ) -> CoordinateRingPresentation<N_PLUS_MORE, U> {
        CoordinateRingPresentation {
            generators: self
                .generators
                .into_iter()
                .map(Binomial::new_vars)
                .collect(),
            #[allow(clippy::collapsible_else_if)]
            representing_vars: core::array::from_fn(|idx| {
                if after_or_before {
                    if idx < N {
                        self.representing_vars[idx].clone()
                    } else {
                        new_us[idx - N].clone()
                    }
                } else {
                    if idx < MORE {
                        new_us[idx].clone()
                    } else {
                        self.representing_vars[idx - MORE].clone()
                    }
                }
            }),
        }
    }

    /// Merge two presentations by taking the union of their binomial generators.
    ///
    /// When `check` is `true`, the two presentations must have identical `representing_vars`
    /// arrays; otherwise the inputs are returned unchanged as `Err((self, other))`.
    ///
    /// # Errors
    ///
    /// Returns `Err((self, other))` if `check` is `true` and the `representing_vars` differ.
    pub fn combine(mut self, other: Self, check: bool) -> Result<Self, (Self, Self)>
    where
        U: Eq,
    {
        if check && self.representing_vars != other.representing_vars {
            return Err((self, other));
        }
        self.generators.extend(other.generators);
        Ok(Self {
            generators: self.generators,
            representing_vars: self.representing_vars,
        })
    }

    /// Form the product of two coordinate ring presentations.
    ///
    /// Embeds `self` into a ring with `N + MORE` variables (appending `other`'s variables after
    /// `self`'s) and `other` symmetrically (prepending `self`'s variables), then merges the
    /// generator lists. The const generics must satisfy `N + MORE == N_PLUS_MORE`.
    ///
    /// # Errors
    ///
    /// Returns `Err((self, other))` if the two `representing_vars` arrays share any element
    /// (the variable sets must be disjoint for the product to be well-defined).
    #[allow(clippy::missing_panics_doc)]
    pub fn product<const MORE: usize, const N_PLUS_MORE: usize>(
        self,
        other: CoordinateRingPresentation<MORE, U>,
    ) -> Result<
        CoordinateRingPresentation<N_PLUS_MORE, U>,
        (Self, CoordinateRingPresentation<MORE, U>),
    >
    where
        U: Eq + Clone,
    {
        for self_rep_var in &self.representing_vars {
            if other.representing_vars.contains(self_rep_var) {
                return Err((self, other));
            }
        }
        let self_vars = self.representing_vars.clone();
        let after_self = self.extend_irrelevant(&other.representing_vars, true);
        let after_other = other.extend_irrelevant(&self_vars, false);
        Ok(after_self
            .combine(after_other, false)
            .map_err(|_| ())
            .expect("We manually made the representing vars the exact same as selfs followed by others and they were all distinct")
        )
    }

    /// Reduce each monomial in `linear_combination` modulo the binomial generators.
    ///
    /// For each monomial, repeatedly applies the substitution `p⁺ ↦ p⁻` (replacing the larger
    /// monomial of each generator with the smaller) until no further reduction is possible.
    /// Coefficients are left unchanged.
    pub fn reduce_function_helper<Scalar>(
        &self,
        mut linear_combination: Vec<(Scalar, Monomial<N>)>,
    ) -> Vec<(Scalar, Monomial<N>)> {
        for (_coeff, monom) in &mut linear_combination {
            let mut changed = true;
            while changed {
                changed = false;
                for binom in &self.generators {
                    let (smaller, larger) = binom.smaller_larger();
                    let can_reduce = monom
                        .exponents
                        .iter()
                        .zip(larger.exponents.iter())
                        .all(|(&m_exp, &l_exp)| m_exp <= l_exp);
                    if can_reduce {
                        for i in 0..N {
                            monom.exponents[i] -= larger.exponents[i];
                            monom.exponents[i] += smaller.exponents[i];
                        }
                        changed = true;
                    }
                }
            }
        }
        linear_combination
    }

    /// Reduce a linear combination of monomials modulo the ideal and convert to the `U` representation.
    ///
    /// Calls [`reduce_function_helper`](Self::reduce_function_helper) and then maps each reduced
    /// monomial to its value in `U` via `raise_power` and `multiply`.
    #[allow(clippy::missing_panics_doc)]
    pub fn reduce_function<Scalar>(
        &self,
        linear_combination: Vec<(Scalar, Monomial<N>)>,
    ) -> Vec<(Scalar, U)> {
        let reduced = self.reduce_function_helper(linear_combination);
        reduced
            .into_iter()
            .map(|(coeff, monom)| {
                let repr = monom
                    .exponents
                    .iter()
                    .enumerate()
                    .map(|(i, &power)| self.representing_vars[i].clone().raise_power(power))
                    .reduce(CoordinateRingRepr::multiply)
                    .expect("N is at least 1");
                (coeff, repr)
            })
            .collect()
    }
}

impl<const N: usize, Repr: CoordinateRingRepr> Display for CoordinateRingPresentation<N, Repr> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Coordinate Ring Presentation with {N} generators. ")?;
        writeln!(
            f,
            "Quotiented by the ideal generated by these {}:",
            self.generators.len()
        )?;
        for binom in &self.generators {
            writeln!(f, "{} - {}", binom.positive, binom.negative)?;
        }
        Ok(())
    }
}

impl<const N: usize, Repr: CoordinateRingRepr> Debug for CoordinateRingPresentation<N, Repr>
where
    Repr: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)?;
        writeln!(f, "The N generators are {:?}", self.representing_vars)?;
        Ok(())
    }
}

impl ToricFan {
    /// Compute the toric coordinate ring presentation `k[x₁,…,xₙ]/I_σ` for an affine fan.
    ///
    /// For a single-cone fan, delegates directly to the cone's
    /// [`coordinate_ring_presentation`](RationalPolyhedralCone::coordinate_ring_presentation).
    /// For a multi-cone fan the fan must be affine; the global support cone is used.
    ///
    /// # Errors
    ///
    /// Returns `Err(CoordinateRingError::NotAffine)` if there is more than one cone and the fan
    /// is not affine, `Err(CoordinateRingError::NotManifestlyAffine)` if the global support cone
    /// cannot be constructed, or any error propagated from the single-cone computation (e.g.
    /// `NonSimplicialCone` or `DimensionMismatch` if `N` is wrong).
    #[allow(clippy::missing_panics_doc)]
    pub fn compute_coordinate_ring_presentation<const N: usize>(
        &self,
        ignore_error: bool,
    ) -> Result<CoordinateRingPresentation<N, DefaultRepr>, CoordinateRingError> {
        if self.count_cones() != 1 {
            if self.is_affine(ignore_error) {
                let global_support_cone = self
                    .global_support_cone()
                    .map_err(|_| CoordinateRingError::NotManifestlyAffine)?;
                return global_support_cone.coordinate_ring_presentation::<N>(false);
            }
            return Err(CoordinateRingError::NotAffine);
        }

        self.iter_cones()
            .next()
            .expect("Already know the count is 1")
            .coordinate_ring_presentation::<N>(true)
    }
}

use nalgebra::{DMatrix, DVector};

#[allow(clippy::needless_pass_by_value)]
fn matrix_to_integer_vecvec<const N: usize>(
    mat: DMatrix<i64>,
) -> (Vec<Vec<i64>>, [DefaultRepr; N]) {
    let mut out0 = vec![vec![0i64; mat.ncols()]; mat.nrows()];
    let mut out1 = core::array::from_fn(|_| DVector::from_vec(vec![]));

    for i in 0..mat.nrows() {
        for j in 0..mat.ncols() {
            let r = mat[(i, j)];
            out0[i][j] = r;
            out1[j] = out1[j].push(r);
        }
    }
    (out0, out1.map(DefaultRepr))
}

fn kernel_to_binomials<const N: usize>(kernel: Vec<[i64; N]>) -> Vec<Binomial<N>> {
    let mut out = Vec::new();

    for mut k in kernel {
        let mut g = 0;
        for &x in &k {
            g = gcd(g, x);
        }
        if g > 1 {
            for x in &mut k {
                *x /= g;
            }
        }

        let mut pos = [0u32; N];
        let mut neg = [0u32; N];

        for (i, cur_k) in k.into_iter().enumerate() {
            if cur_k > 0 {
                pos[i] = cur_k.unsigned_abs() as u32;
            } else if cur_k < 0 {
                neg[i] = (-cur_k).unsigned_abs() as u32;
            }
        }

        if pos.iter().any(|&x| x != 0) {
            out.push(Binomial {
                positive: Monomial { exponents: pos },
                negative: Monomial { exponents: neg },
            });
        }
    }

    out
}

fn compute_binomial_presentation_from_generators_mat<const N: usize>(
    int_mat: DMatrix<i64>,
) -> Result<CoordinateRingPresentation<N, DefaultRepr>, CoordinateRingError> {
    if int_mat.ncols() != N {
        return Err(CoordinateRingError::DimensionMismatch {
            expected: N,
            actual: int_mat.ncols(),
        });
    }

    let (int_mat, columns_of_int_mat) = matrix_to_integer_vecvec::<N>(int_mat);
    let kernel = kernel_from_snf::<N>(int_mat);
    let binomials = kernel_to_binomials::<N>(kernel);

    Ok(CoordinateRingPresentation {
        generators: binomials,
        representing_vars: columns_of_int_mat,
    })
}

impl RationalPolyhedralCone {
    /// Compute the coordinate ring presentation of the affine toric variety
    ///
    /// # Errors
    /// - Returns `CoordinateRingError::DimensionMismatch` if the number of rays does not equal `N`
    /// - Returns `CoordinateRingError::LinearAlgebraFailure` if linear algebra computations fail
    pub fn coordinate_ring_presentation<const N: usize>(
        &self,
        nonminimal_care: bool,
    ) -> Result<CoordinateRingPresentation<N, DefaultRepr>, CoordinateRingError> {
        let num_rays = if nonminimal_care {
            self.num_rays()
                .map_err(|_| CoordinateRingError::PossiblyNonMinimalCone)?
        } else {
            self.view_generators().len()
        };

        if num_rays != N {
            return Err(CoordinateRingError::DimensionMismatch {
                expected: N,
                actual: num_rays,
            });
        }
        if N == 0 {
            return Ok(CoordinateRingPresentation {
                generators: vec![],
                representing_vars: core::array::from_fn(|_| DefaultRepr::from_vec(vec![])),
            });
        }

        // Build matrix with columns = ray generators
        let mat = DMatrix::<i64>::from_columns(self.view_generators());

        compute_binomial_presentation_from_generators_mat(mat)
    }
}

#[allow(clippy::missing_panics_doc)]
pub fn main_toric_ideal_example() {
    // Example usage
    let cone = RationalPolyhedralCone::c2();

    let presentation = cone
        .coordinate_ring_presentation::<2>(true)
        .expect("Valid coordinate ring presentation");

    println!("Coordinate Ring Presentation of C^2: {presentation}");

    let mut fan = ToricFan::c3();
    let presentation = fan
        .coordinate_ring_presentation::<3, DefaultRepr>(|z| z, true)
        .expect("C3 fan");

    println!("Coordinate Ring Presentation of C^3: {presentation}");

    assert!(presentation.generators.is_empty());

    let cone =
        RationalPolyhedralCone::new(vec![vec![1, 0], vec![2, 0]], Some(false), Some(1), None)
            .expect("Valid cone");

    assert_eq!(cone.num_rays(), Ok(1));
    assert_eq!(cone.view_generators().len(), 1);

    let presentation = cone
        .coordinate_ring_presentation::<2>(true)
        .expect_err("Invalid coordinate ring presentation");

    println!("Coordinate Ring Presentation of e1 only in R^2: {presentation:?}");

    let presentation = cone
        .coordinate_ring_presentation::<1>(true)
        .expect("Valid coordinate ring presentation");

    println!("Coordinate Ring Presentation of e1 only in R^2: {presentation}");
}

mod test {

    #[test]
    fn toric_fan_c2_coordinate_ring() {
        use super::DefaultRepr;
        use crate::toric::fan::ToricFan;
        let mut fan = ToricFan::c2();

        let pres = fan
            .coordinate_ring_presentation::<2, DefaultRepr>(|z| z, true)
            .expect("C^2 should be affine");
        assert!(pres.check().is_ok());

        assert!(pres.generators.is_empty(), "C^2 should have no relations");
    }

    #[test]
    fn toric_fan_c3_coordinate_ring() {
        use super::DefaultRepr;
        use crate::toric::fan::ToricFan;
        let mut fan = ToricFan::c3();

        let pres = fan
            .coordinate_ring_presentation::<3, DefaultRepr>(|z| z, true)
            .expect("C^3 should be affine");
        assert!(pres.check().is_ok());

        assert!(pres.generators.is_empty(), "C^3 should have no relations");
    }

    #[test]
    fn one_cone_but_nonsimplicial() {
        use super::CoordinateRingError;
        use crate::toric::cone::RationalPolyhedralCone;
        use crate::toric::cone_errors::ConeError;
        let cone = RationalPolyhedralCone::new(
            vec![vec![1, 1], vec![1, 0], vec![1, -1]],
            Some(false),
            Some(2),
            None,
        )
        .expect("Valid cone");

        assert_eq!(cone.num_rays(), Err(ConeError::NotMinimallyPresented));

        let presentation = cone.coordinate_ring_presentation::<3>(true);
        assert_eq!(
            presentation.expect_err("is nonminimal"),
            CoordinateRingError::PossiblyNonMinimalCone
        );

        let presentation = cone
            .coordinate_ring_presentation::<3>(false)
            .expect("Treating as if it came as a fan with two cones");
        let b = &presentation.generators[0];
        let b = (b.positive.exponents, b.negative.exponents);
        let expected_pos = [1, 0, 1];
        let expected_neg = [0, 2, 0];
        assert!(
            b == (expected_pos, expected_neg),
            "expected xy - zw, got {:?} with pres {}",
            b,
            presentation
        );
        assert!(presentation.check().is_ok());
    }

    #[test]
    fn toric_fan_conifold_coordinate_ring() {
        use super::DefaultRepr;
        use crate::toric::fan::ToricFan;
        let mut fan = ToricFan::conifold();
        let pres = fan
            .coordinate_ring_presentation::<4, DefaultRepr>(|z| z, true)
            .expect("Conifold fan should be affine");
        assert_eq!(
            pres.generators.len(),
            1,
            "Conifold should have exactly one relation"
        );
        let b = &pres.generators[0];
        let b = (b.positive.exponents, b.negative.exponents);
        let expected_pos = [1, 0, 0, 1];
        let expected_neg = [0, 1, 1, 0];
        assert!(
            b == (expected_pos, expected_neg),
            "expected xy - zw, got {:?} with pres {}",
            b,
            pres
        );
        assert!(pres.check().is_ok());

        let mut fan = ToricFan::conifold2();
        let pres = fan
            .coordinate_ring_presentation::<4, DefaultRepr>(|z| z, true)
            .expect("Conifold fan should be affine");
        assert_eq!(
            pres.generators.len(),
            1,
            "Conifold should have exactly one relation"
        );
        let b = &pres.generators[0];
        let b = (b.positive.exponents, b.negative.exponents);
        let expected_pos = [0, 1, 1, 0];
        let expected_neg = [1, 0, 0, 1];
        assert!(
            b == (expected_pos, expected_neg),
            "expected xy - zw, got {:?} with pres {}",
            b,
            pres
        );
        assert!(pres.check().is_ok());
    }
}
