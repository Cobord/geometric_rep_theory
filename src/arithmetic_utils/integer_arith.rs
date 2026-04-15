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
