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
