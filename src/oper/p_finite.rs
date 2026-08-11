/// A sequence is P-finite if it satisfies a linear recurrence relation with polynomial coefficients.
/// This module provides a way to represent and evaluate such sequences efficiently,
/// especially when the sequence is "balanced" (i.e., all the polynomials have the same degree).
///
/// The linear recurrence relation is of the form:
/// `A_0 (n) u_n + A_1(n) u_{n-1} + \cdots + A_r(n) u_{n-r} = 0`
/// where `A_i(n)` are polynomials in `n` and `u_n` is the nth term of the sequence.
///
/// The `PFiniteSequence` struct stores the initial values of the sequence, the polynomials `A_i(n)`,
/// and some metadata to optimize evaluation.
/// The optimization involves checkpointing previously computed values of the sequence to avoid redundant calculations,
/// If `u_{1_000-1}` through `u_{1_000-r}` are checkpointed, then `u_{1_000}`
/// can be computed with only one use of the recurrence relation.
/// It does not require doing all of `u_0` through `u_{1_000-1}` again.
use std::{
    cell::OnceCell,
    collections::VecDeque,
    ops::{Add, AddAssign, DivAssign, Mul, MulAssign, Neg},
};

/// Polynomials in one variable where the underlying ring is `Values`
pub trait Polynomial<Values>
where
    Values: Add<Values, Output = Values>
        + Mul<Values, Output = Values>
        + AddAssign<Values>
        + MulAssign<Values>
        + Clone,
{
    /// Evaluate the polynomial at `x`
    fn evaluate(&self, x: Values) -> Values;
    /// The degree of the polynomial. Assumed it is not the 0 polynomial.
    fn degree(&self) -> usize;
    /// The constant polynomial `-1`
    fn neg_one() -> Self
    where
        Values: From<u8> + Neg<Output = Values>;
}

/// A default implementation of `Polynomial<Values>` just stores
/// the coefficients in a vector. Other implementations can be more
/// efficient, but for examples this allows easy testing.
#[derive(PartialEq, Eq, Clone)]
pub struct DefaultPoly<Values>(Vec<Values>);

impl<Values> Polynomial<Values> for DefaultPoly<Values>
where
    Values: Add<Values, Output = Values>
        + Mul<Values, Output = Values>
        + AddAssign<Values>
        + MulAssign<Values>
        + Clone,
{
    fn evaluate(&self, x: Values) -> Values {
        let mut answer = self.0.first().expect("Not the 0 polynomial").clone();
        let mut x_to_power = x.clone();
        for coeff_idx in self.0.iter().skip(1) {
            answer += x_to_power.clone() * coeff_idx.clone();
            x_to_power *= x.clone();
        }
        answer
    }

    fn degree(&self) -> usize {
        self.0.len() - 1
    }

    fn neg_one() -> Self
    where
        Values: From<u8> + Neg<Output = Values>,
    {
        Self(vec![-Values::from(1)])
    }
}

/// `A_0 (n) u_n + A_1(n) u_{n-1} + \cdots A_r(n) u_{n-r}`
///
/// `\vec{u}_i \equiv (u_i, u_{i+1}, \cdots u_{i+r-1})`
///
/// `AMatrix \vec{u}_{i} = \vec{u}_{i+1}`
#[must_use = "Evaluate this sequence at some points or combine it with some other sequences"]
pub struct PFiniteSequence<Values, Poly>
where
    Values: Add<Values, Output = Values>
        + Mul<Values, Output = Values>
        + AddAssign<Values>
        + MulAssign<Values>
        + Clone
        + From<u8>,
    Poly: Polynomial<Values>,
{
    initial_values: Vec<Values>,
    a_i_n: Vec<Poly>,
    is_balanced: OnceCell<bool>,
    checkpointed_values: Vec<(usize, Vec<Values>)>,
    min_gap_to_save: usize,
    starts_with_neg_one: bool,
}

impl<Values, Poly> PFiniteSequence<Values, Poly>
where
    Values: Add<Values, Output = Values>
        + Mul<Values, Output = Values>
        + AddAssign<Values>
        + MulAssign<Values>
        + Clone
        + From<u8>
        + Neg<Output = Values>,
    Poly: Polynomial<Values>,
{
    pub fn new(
        initial_values: Vec<Values>,
        a_i_n: Vec<Poly>,
        min_gap_to_save: usize,
        starts_with_neg_one: bool,
    ) -> Self {
        Self {
            initial_values,
            a_i_n,
            is_balanced: OnceCell::new(),
            checkpointed_values: Vec::new(),
            min_gap_to_save,
            starts_with_neg_one,
        }
    }

    pub fn min_gap_to_save(&self) -> usize {
        self.min_gap_to_save
    }

    /// The order of the system is the analog
    /// of the order of a linear ordinary differential equation
    pub fn order(&self) -> usize {
        self.a_i_n.len() - 1
    }

    /// The degrees of the first and the last polynomial are the same
    /// and every degree in between is less than or equal to that
    #[allow(clippy::missing_panics_doc)]
    pub fn is_balanced(&self) -> bool {
        if let Some(answer) = self.is_balanced.get() {
            *answer
        } else {
            let answer: bool = {
                let first_degree = self
                    .a_i_n
                    .first()
                    .expect("There must be at least one polynomial")
                    .degree();
                let last_degree = self
                    .a_i_n
                    .last()
                    .expect("There must be at least one polynomial")
                    .degree();
                if first_degree == last_degree {
                    self.a_i_n
                        .iter()
                        .skip(1)
                        .all(|p_r| p_r.degree() <= first_degree)
                } else {
                    false
                }
            };
            let () = self.is_balanced.set(answer).expect("It was uninitialized");
            answer
        }
    }

    pub fn evaluate_at_saved(&mut self, idx: usize) -> Values
    where
        Values: DivAssign<Values> + Neg<Output = Values>,
    {
        let (answer, to_save) = self.evaluate_at(idx);
        if let Some(to_save) = to_save {
            let insertion_point = self
                .checkpointed_values
                .binary_search_by_key(&to_save.0, |(idx, _)| *idx);
            if let Err(insertion_point) = insertion_point {
                let gap_with_before = if insertion_point > 0 {
                    to_save.0 - self.checkpointed_values[insertion_point - 1].0
                } else {
                    to_save.0 - (self.initial_values.len() - 1)
                };
                let do_insert = if insertion_point < self.checkpointed_values.len() {
                    gap_with_before > self.min_gap_to_save()
                        && self.checkpointed_values[insertion_point].0 - to_save.0
                            > self.min_gap_to_save()
                } else {
                    gap_with_before > self.min_gap_to_save()
                };
                if do_insert {
                    self.checkpointed_values
                        .insert(insertion_point, (to_save.0, to_save.1.into()));
                } else {
                    println!(
                        "Could save {idx} had {:?} which had {gap_with_before}",
                        self.checkpointed_values
                            .iter()
                            .map(|z| z.0)
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
        answer
    }

    /// Evaluate `u(idx)`. The second output is there because
    /// values can be saved as in the original docstring to avoid
    /// for example recomputing `u_{1000}` if `u_{999}` down to `u_{990}`
    /// are already known and are enough to compute `u_{1000}`.
    #[allow(clippy::missing_panics_doc)]
    pub fn evaluate_at(&self, idx: usize) -> (Values, Option<(usize, VecDeque<Values>)>)
    where
        Values: DivAssign<Values> + Neg<Output = Values>,
    {
        if idx < self.initial_values.len() {
            return (self.initial_values[idx].clone(), None);
        }
        for (known_end, known_vec) in &self.checkpointed_values {
            if idx <= *known_end && *known_end - idx < known_vec.len() {
                let idx_in_known = known_vec.len() + idx - *known_end - 1;
                return (known_vec[idx_in_known].clone(), None);
            }
        }
        let (starting_idx, mut computed_values) =
            if let Some(w) = self.checkpointed_values.iter().rposition(|z| z.0 <= idx) {
                (
                    self.checkpointed_values[w].0 + 1,
                    VecDeque::from(self.checkpointed_values[w].1.clone()),
                )
            } else {
                let all_but_last_r = self.initial_values.len() - self.order();
                let computed_values = self
                    .initial_values
                    .iter()
                    .skip(all_but_last_r)
                    .cloned()
                    .collect();
                (self.initial_values.len(), computed_values)
            };
        for cur_idx in starting_idx..=idx {
            let at_cur_idx = self.next_value(cur_idx, &computed_values);
            computed_values.push_back(at_cur_idx);
            computed_values.pop_front();
        }
        let value_at_idx = computed_values.back().expect("It is of length r").clone();
        (value_at_idx, Some((idx, computed_values)))
    }

    /// One iteration of the recurrence
    fn next_value(&self, cur_idx: usize, values_before: &VecDeque<Values>) -> Values
    where
        Values: DivAssign<Values> + Neg<Output = Values>,
    {
        let r = values_before.len();
        let mut answer: Values = 0.into();
        let cur_idx_small = u8::try_from(cur_idx).expect("Evaluating at some large natural number");
        for (how_much_before_cur_idx, u_idx) in values_before
            .iter()
            .enumerate()
            .map(|(idx, z)| (r - idx, z))
        {
            let a_i_cur_idx = self.a_i_n[how_much_before_cur_idx].evaluate(cur_idx_small.into());
            answer += a_i_cur_idx * u_idx.clone();
        }
        if self.starts_with_neg_one {
            answer
        } else {
            answer /= self.a_i_n[0].evaluate(cur_idx_small.into());
            -answer
        }
    }
}

#[cfg(test)]
mod test {

    #[test]
    fn fibonacci() {
        use super::{DefaultPoly, PFiniteSequence};
        let one = DefaultPoly(vec![1.0]);
        let neg_one = DefaultPoly(vec![-1.0]);
        let mut fib = PFiniteSequence::new(
            vec![1.0, 1.0, 2.0, 3.0, 5.0],
            vec![neg_one, one.clone(), one],
            5,
            true,
        );
        assert!(fib.is_balanced());
        assert!(fib.checkpointed_values.is_empty());
        assert_eq!(fib.evaluate_at_saved(0), 1.0);
        assert_eq!(fib.evaluate_at_saved(1), 1.0);
        assert_eq!(fib.evaluate_at_saved(2), 2.0);
        assert_eq!(fib.evaluate_at_saved(3), 3.0);
        assert_eq!(fib.evaluate_at_saved(4), 5.0);
        assert_eq!(fib.evaluate_at_saved(5), 8.0);
        assert_eq!(fib.evaluate_at_saved(6), 13.0);
        assert!(fib.checkpointed_values.is_empty());
        assert_eq!(fib.evaluate_at_saved(16), 1597.0);
        assert_eq!(fib.checkpointed_values.len(), 1);
        assert_eq!(fib.evaluate_at_saved(15), 987.0);
        assert_eq!(fib.evaluate_at_saved(16), 1597.0);
        assert_eq!(fib.evaluate_at_saved(17), 2584.0);
        assert_eq!(fib.checkpointed_values.len(), 1);
        assert_eq!(fib.evaluate_at_saved(23), 46368.0);
        assert_eq!(fib.checkpointed_values.len(), 2);
    }

    #[test]
    fn signed_permutation_matrices() {
        use super::{DefaultPoly, PFiniteSequence};
        let two = DefaultPoly(vec![2.0]);
        let neg_one = DefaultPoly(vec![-1.0]);
        let quadratic = DefaultPoly(vec![4.0, -8.0, 4.0]);
        let mut signed_permutation_matrices_count =
            PFiniteSequence::new(vec![1.0, 2.0], vec![neg_one, two, quadratic], 5, true);
        assert!(!signed_permutation_matrices_count.is_balanced());
        assert!(
            signed_permutation_matrices_count
                .checkpointed_values
                .is_empty()
        );
        let mut idx_factorial = 1;
        for idx in 1..11 {
            idx_factorial *= idx;
            let expected = (1 << idx) * idx_factorial;
            let expected = f64::try_from(expected as u32).expect("Keeping idx small enough");
            assert_eq!(
                signed_permutation_matrices_count.evaluate_at_saved(idx),
                expected
            );
        }
    }
}
