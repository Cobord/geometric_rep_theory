use nalgebra::{DMatrix, DVector};
use num::integer::gcd;

#[allow(dead_code)]
fn normalize(v: &mut [i64]) {
    let mut g = 0;
    for &x in v.iter() {
        g = gcd(g, x);
    }
    if g > 1 {
        for x in v.iter_mut() {
            *x /= g;
        }
    }
}

/// m × n integer matrix
type ZMatrix = Vec<Vec<i64>>;

/// n × n unimodular matrix (tracks column ops)
type ZSquare = Vec<Vec<i64>>;

fn identity(n: usize) -> ZSquare {
    let mut id = vec![vec![0; n]; n];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        id[i][i] = 1;
    }
    id
}

#[allow(dead_code)]
fn identity_matrix(n: usize) -> DMatrix<i64> {
    let mut mat = DMatrix::zeros(n, n);
    for idx in 0..n {
        mat[idx * n + idx] = 1;
    }
    mat
}

#[allow(clippy::needless_range_loop, clippy::many_single_char_names)]
fn smith_normal_form(a: &mut ZMatrix) -> ZSquare {
    let m = a.len();
    let n = a[0].len();

    let mut v = identity(n);

    let mut i = 0;
    let mut j = 0;

    while i < m && j < n {
        // 1. Find nonzero pivot
        let mut pivot = None;
        for r in i..m {
            for c in j..n {
                if a[r][c] != 0 {
                    pivot = Some((r, c));
                    break;
                }
            }
            if pivot.is_some() {
                break;
            }
        }

        if pivot.is_none() {
            break;
        }

        let (r, c) = pivot.expect("pivot is Some: checked above");

        // 2. Move pivot to (i, j)
        a.swap(i, r);
        for row in &mut v {
            row.swap(j, c);
        }
        for row in a.iter_mut() {
            row.swap(j, c);
        }

        // 3. Clear column j
        for r2 in 0..m {
            if r2 != i && a[r2][j] != 0 {
                let g = gcd(a[i][j], a[r2][j]);
                let s = a[i][j] / g;
                let t = a[r2][j] / g;

                for c2 in j..n {
                    a[r2][c2] = s * a[r2][c2] - t * a[i][c2];
                }
            }
        }

        // 4. Clear row i
        for c2 in 0..n {
            if c2 != j && a[i][c2] != 0 {
                let g = gcd(a[i][j], a[i][c2]);
                let s = a[i][j] / g;
                let t = a[i][c2] / g;

                for r2 in 0..m {
                    a[r2][c2] = s * a[r2][c2] - t * a[r2][j];
                }
                for r2 in 0..n {
                    v[r2][c2] = s * v[r2][c2] - t * v[r2][j];
                }
            }
        }

        if a[i][j] < 0 {
            for c2 in j..n {
                a[i][c2] = -a[i][c2];
            }
        }

        i += 1;
        j += 1;
    }

    v
}

/// Möbius function μ(n).
/// Returns 0 if n has a squared prime factor, otherwise (−1)^k where k is the
/// number of distinct prime factors of n.
#[allow(dead_code)]
pub(crate) fn mobius(n: usize) -> i8 {
    if n == 1 {
        return 1;
    }
    let mut m = n;
    let mut k: i8 = 0;
    let mut d = 2usize;
    while d * d <= m {
        if m.is_multiple_of(d) {
            k += 1;
            m /= d;
            if m.is_multiple_of(d) {
                return 0;
            }
        }
        d += 1;
    }
    if m > 1 {
        k += 1;
    }
    if k % 2 == 0 { 1 } else { -1 }
}

/// Binomial coefficient C(n, k).
#[must_use = "n choose k"]
pub fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    (1..=k).fold(1usize, |acc, i| acc * (n - k + i) / i)
}

/// The Kronecker symbol `(a|n)`, generalizing the Legendre/Jacobi symbol to
/// any integer `n` (not just odd primes/positive odd integers): `(a|0)` is
/// `1` if `a = ±1` and `0` otherwise, `(a|n)` for negative `n` folds in the
/// sign via `(a|-1)`, and even `n` folds in the supplementary symbol
/// `(a|2)` (`0` if `a` is even, `1` if `a ≡ ±1 (mod 8)`, `-1` if
/// `a ≡ ±3 (mod 8)`). Computed via the same Euclidean-style reciprocity
/// recursion as the Jacobi symbol, in `O(log(min(|a|, n)))`, without
/// factoring `n`.
#[must_use = "the Kronecker symbol (a|n)"]
pub fn kronecker_symbol(a: i128, n: i128) -> i128 {
    if n == 0 {
        return i128::from(a == 1 || a == -1);
    }
    if n == 1 {
        return 1;
    }

    let mut result = 1i128;
    let mut n = n;
    let mut a = a;

    if n < 0 {
        if a < 0 {
            result = -result;
        }
        n = -n;
    }

    let mut twos = 0u32;
    while n % 2 == 0 {
        n /= 2;
        twos += 1;
    }
    if twos > 0 {
        let symbol_two: i128 = if a % 2 == 0 {
            0
        } else if matches!(a.rem_euclid(8), 1 | 7) {
            1
        } else {
            -1
        };
        result *= symbol_two.pow(twos);
    }
    if result == 0 {
        return 0;
    }

    a = a.rem_euclid(n);
    while a != 0 {
        while a % 2 == 0 {
            a /= 2;
            if matches!(n.rem_euclid(8), 3 | 5) {
                result = -result;
            }
        }
        std::mem::swap(&mut a, &mut n);
        if a.rem_euclid(4) == 3 && n.rem_euclid(4) == 3 {
            result = -result;
        }
        a = a.rem_euclid(n);
    }
    if n == 1 { result } else { 0 }
}

/// Iterator over all multi-indices β with `0 ≤ β[i] ≤ upper[i]` for each i,
/// in lexicographic order.
pub fn multi_index_le<const N: usize>(upper: [usize; N]) -> impl Iterator<Item = [usize; N]> {
    let sizes = upper.map(|d| d + 1);
    let total: usize = sizes.iter().product();
    (0..total).map(move |mut flat| {
        let mut beta = [0usize; N];
        for i in (0..N).rev() {
            beta[i] = flat % sizes[i];
            flat /= sizes[i];
        }
        beta
    })
}

pub fn kernel_from_snf<const N: usize>(mut a: ZMatrix) -> Vec<[i64; N]> {
    let v = smith_normal_form(&mut a);

    let mut kernel = Vec::new();

    let rank = a
        .iter()
        .enumerate()
        .take_while(|(_, row)| row.iter().any(|&x| x != 0))
        .count();

    #[allow(clippy::needless_range_loop)]
    for col in rank..N {
        let mut k = [0i64; N];
        for i in 0..N {
            k[i] = v[i][col];
        }
        kernel.push(k);
    }

    kernel
}

pub(crate) fn primitive_vector(v: &[i64], sign_flippable: bool) -> Option<DVector<i64>> {
    if v.iter().all(|z| *z == 0) {
        return None;
    }
    let g = v.iter().fold(
        0i64,
        |acc, &x| if acc == 0 { x.abs() } else { gcd(acc, x.abs()) },
    );
    let mut prim: Vec<i64> = v.iter().map(|x| x / g).collect();

    if sign_flippable {
        // canonical sign: first nonzero positive
        for x in &prim {
            if *x != 0 {
                if *x < 0 {
                    for y in &mut prim {
                        *y = -*y;
                    }
                }
                break;
            }
        }
    }

    Some(DVector::from_vec(prim))
}

#[cfg(test)]
mod tests {
    use super::kronecker_symbol;

    #[test]
    fn edge_cases() {
        assert_eq!(kronecker_symbol(1, 0), 1);
        assert_eq!(kronecker_symbol(-1, 0), 1);
        assert_eq!(kronecker_symbol(2, 0), 0);
        assert_eq!(kronecker_symbol(5, 1), 1);
        assert_eq!(kronecker_symbol(0, 1), 1);
        assert_eq!(kronecker_symbol(5, -1), 1);
        assert_eq!(kronecker_symbol(-5, -1), -1);
        assert_eq!(kronecker_symbol(0, -1), 1);
    }

    #[test]
    fn supplementary_symbol_at_2() {
        assert_eq!(kronecker_symbol(1, 2), 1); // 1 mod 8 = 1
        assert_eq!(kronecker_symbol(3, 2), -1); // 3 mod 8 = 3
        assert_eq!(kronecker_symbol(2, 2), 0); // even
        assert_eq!(kronecker_symbol(5, 2), -1); // 5 mod 8 = 5
        assert_eq!(kronecker_symbol(7, 2), 1); // 7 mod 8 = 7
    }

    #[test]
    fn matches_legendre_symbol_for_odd_primes() {
        assert_eq!(kronecker_symbol(2, 7), 1); // 3^2 = 9 = 2 (mod 7)
        assert_eq!(kronecker_symbol(-3, 7), 1); // 7 = 1 (mod 3), splits
        assert_eq!(kronecker_symbol(-3, 5), -1); // 5 = 2 (mod 3), inert
        assert_eq!(kronecker_symbol(-3, 13), 1);
        assert_eq!(kronecker_symbol(-3, 11), -1);
    }

    #[test]
    fn matches_jacobi_symbol_for_odd_composite_n() {
        assert_eq!(kronecker_symbol(8, 15), 1);
        assert_eq!(kronecker_symbol(7, 15), -1);
        assert_eq!(kronecker_symbol(1001, 9907), -1);
    }

    #[test]
    fn multiplicative_in_second_argument() {
        for &a in &[-7i128, -3, -1, 0, 1, 2, 5, 11] {
            for n1 in -6i128..=6 {
                for n2 in -6i128..=6 {
                    if n1 == 0 || n2 == 0 {
                        continue;
                    }
                    let lhs = kronecker_symbol(a, n1 * n2);
                    let rhs = kronecker_symbol(a, n1) * kronecker_symbol(a, n2);
                    assert_eq!(lhs, rhs, "a={a} n1={n1} n2={n2}");
                }
            }
        }
    }
}
