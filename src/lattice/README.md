# lattice

Lattices in the sense of `Z^n` equipped with extra structure: a free
`Z`-module of finite rank `N`, together with a rational-valued symmetric
bilinear form. This covers integral lattices as well as rational ones such
as the dual of an integral lattice (e.g. `A_n*` has pairings in
`(1/(n+1))Z`, not just `Z`).

## Traits

### `Lattice<const N: usize>`

A rank-`N` lattice. Requires `Add`, `Sub`, `AddAssign`, `SubAssign` (the
abelian group structure of `Z^N`), `Mul<i128>`/`MulAssign<i128>` (scalar
multiplication by integers only — a lattice is not a vector space, so there
is no division and no action of arbitrary rationals or reals), and
`PartialEq` + `Zero` (equality and the origin).

- `inner_product(&self, other: &Self) -> Ratio<i128>` — the symmetric
  bilinear form `<self, other>`.
- `lattice_norm_sq(&self) -> Ratio<i128>` — the associated quadratic form
  `<self, self>`. Defaulted in terms of `inner_product`.
- `is_integral() -> bool` — whether `<v, w>` is an integer for every `v, w`
  in the lattice, i.e. whether this is an integral lattice rather than a
  properly rational one (such as the dual of an integral lattice).
- `is_even() -> bool` — whether `<v, v>` is an even integer for every vector
  `v` in the lattice (e.g. root lattices such as `E8`), as opposed to merely
  an integer (odd lattices). Implies `is_integral()`.
- `is_self_dual() -> bool` — whether the lattice is unimodular, i.e. equal to
  its own dual (equivalently, `discriminant() == 1`).
- `signature() -> (usize, usize, usize)` — `(p, q, r)`, the number of
  positive, negative, and zero eigenvalues of the Gram matrix of
  `inner_product`, so `p + q + r = N` always; `r` is the dimension of the
  radical, and the form is non-degenerate iff `r == 0`.
- `discriminant() -> Ratio<u128>` — `|det(G)|` for the Gram matrix `G` of
  `inner_product`, basis-independent since `GL_N(Z)` changes of basis scale
  `det(G)` by `det(A)^2 = 1`. Zero exactly when the form is degenerate; need
  not be an integer for a non-integral rational lattice.
- `type DualLattice: Lattice<N>` — the dual lattice `L* = Hom(L, Z)`, the
  lattice of `Z`-linear functionals on `L`, with its own induced lattice
  structure.
- `short_vectors() -> Result<Vec<Self>, ShortVectorError>` — the nonzero
  vectors of minimal norm (e.g. the roots of a root lattice). Fails with
  `ShortVectorError::NotPositiveDefinite` when `signature() != (N, 0, 0)`,
  since an indefinite or degenerate form need not have a finite minimal-norm
  set (e.g. an isotropic vector can be scaled by any nonzero integer); may
  fail with `ShortVectorError::Unknown` for a positive-definite lattice
  whose minimal vectors aren't computed by a given implementation.
- `discriminant_group_quadratic(x: Self::DualLattice) -> Ratio<i128>` — the
  discriminant-group quadratic form, evaluated at a representative `x` of a
  class in `L*/L`. Two representatives of the same class have `<x, x>`
  differing by an integer (or even integer, if `L` is even); the default
  implementation normalizes `x.lattice_norm_sq()` into `[0, 1)` (or `[0, 2)`
  if `is_even()`), so equal classes always give equal `Ratio<i128>` values.
- `satisfies_milgram() -> bool` — sanity check that an even unimodular
  lattice has signature `p - q ≡ 0 (mod 8)`. Vacuously `true` unless the
  lattice is both even and self-dual. `O(1)`: only inspects `signature()`,
  unlike the full discriminant-form Gauss sum this is a special case of
  (which applies to any even lattice but needs to enumerate all of `L*/L`).
- `theta_series_weight() -> Result<Ratio<i128>, ShortVectorError>` — the
  weight `N/2` of the theta series `Σ_{x∈L} q^{<x,x>/2}` as a modular form.
  Fails with `NotPositiveDefinite` if `signature() != (N, 0, 0)`, since the
  sum then diverges rather than defining a holomorphic modular form.
  Doesn't enumerate any vectors, unlike the theta series itself. The sum is
  a genuine power series in `q` only when `L` is even (every exponent is an
  integer); otherwise it's still holomorphic, just expanded in a smaller
  nome — `q^{1/2}` if `L` is integral but odd, or `q^{1/(2s)}` for a
  properly rational `L` with pairing-denominator `s` (worse still). Neither
  case is an error here; it only affects the multiplier system/congruence
  subgroup the resulting weight-`N/2` form lives on, which isn't tracked.
- `theta_series_character(d: u128) -> Result<Option<i128>, ShortVectorError>`
  — `χ(d)` in the theta series' transformation law: for a
  positive-definite, integral lattice of even rank `N` and any
  `g = (a b; c d) ∈ Gamma_0(M)` (`M` the level, not computed here),
  `Θ(gτ) = Θ((aτ+b)/(cτ+d)) = χ(d) · (cτ+d)^{N/2} · Θ(τ)`, where
  `Θ(τ) = Σ_{x∈L} q^{<x,x>/2}` and `χ(d) = (D|d)` is the Kronecker symbol
  (via `arithmetic_utils::kronecker_symbol`) of `D = (-1)^{N/2} * disc(L)`.
  `Ok(None)` if `N` is odd (half-integral weight needs a different
  multiplier theory) or `L` isn't integral — `L` being integral but *odd*
  is not excluded, since the formula only needs integrality, not evenness.
  Propagates `theta_series_weight`'s `NotPositiveDefinite` otherwise. E.g.
  `E8` gives `Ok(Some(1))` for any `d` (trivial character — its theta
  series is the weight-4 level-1 Eisenstein series `E4`), and `A_2` gives
  `Ok(Some(1))` at primes splitting in `Q(√-3)` (like `7 ≡ 1 mod 3`) and
  `Ok(Some(-1))` at inert primes (like `5 ≡ 2 mod 3`).

## Structures

### `StandardLattice<const N: usize>`

An element of `Z^N` with the identity Gram matrix (the usual dot product),
stored as integer coefficients `[i128; N]`. Integral, unimodular
(discriminant `1`), but odd (`<e_1, e_1> = 1`) — the simplest example of an
integral lattice that's not even. Self-dual, so `DualLattice = Self`.
`short_vectors` returns the `2N` standard basis vectors `±e_i`, each of
norm `1`.

### `HyperbolicPlane`

The hyperbolic plane lattice `U` (also written `II_{1,1}`): rank 2, Gram
matrix `[[0, 1], [1, 0]]` in the standard basis `e_1, e_2`. Even, unimodular,
signature `(1, 1, 0)`. The basic building block of even unimodular lattices
such as the K3 lattice `U^3 ⊕ E8(-1)^2`. Self-dual, so
`DualLattice = HyperbolicPlane`. Indefinite, so `short_vectors` always
returns `Err(NotPositiveDefinite)`.

### `RootLatticeA<const N: usize>`

An element of the `A_N` root lattice, stored as integer coefficients
`[i128; N]` in the basis of simple roots `alpha_1, ..., alpha_N`. The Gram
matrix is the `A_N` Cartan matrix (`2` on the diagonal, `-1` on the
super/subdiagonal). Even, positive-definite (signature `(N, 0, 0)`), with
discriminant `N + 1` — self-dual only in the trivial case `N == 0`.
`short_vectors` returns the `N(N+1)` roots, each of norm `2`: the vectors
`alpha_i + alpha_{i+1} + ... + alpha_j` for `0 <= i <= j < N` and their
negatives.

### `DualRootLatticeA<const N: usize>`

The dual lattice `A_N*`, i.e. `RootLatticeA::<N>::DualLattice`. Stored as
integer coefficients in the dual basis `alpha_1*, ..., alpha_N*`, with Gram
matrix the inverse Cartan matrix (closed form
`min(a,b)(N+1-max(a,b))/(N+1)`). Discriminant `1/(N+1)`; not integral for
`N >= 1`, since e.g. `alpha_1*` has norm `2/(N+1)`. `short_vectors` returns
the `2(N+1)` minimal vectors of norm `N/(N+1)` (just `2` when `N == 1`,
where the two `+-` pairs coincide): in the standard embedding `A_N* = {
P(z) : z ∈ Z^{N+1} }` (`P` the projection onto the sum-zero hyperplane),
these are `±P(e_i)` for `i = 1, ..., N+1`, i.e. `±(alpha_i* - alpha_{i-1}*)`
with `alpha_0* = alpha_{N+1}* = 0`.

### `RootLatticeD<const N: usize>`

An element of the `D_N` root lattice, stored as integer coefficients
`[i128; N]` in the basis of simple roots `alpha_1, ..., alpha_N`, where
`alpha_i = e_i - e_{i+1}` for `i < N` and `alpha_N = e_{N-1} + e_N` (the
"fork" root), matching the standard embedding
`D_N = { x ∈ Z^N : sum x_i even }`. The Dynkin diagram is the path
`0-1-...-(N-2)` with an extra branch from `N-3` to `N-1`. Even,
positive-definite (signature `(N, 0, 0)`), with discriminant `4` for
`N >= 1` — self-dual only in the trivial case `N == 0`. `short_vectors`
returns the `2N(N-1)` roots `±e_i±e_j` (`0 <= i < j < N`), each of norm `2`,
translated into simple-root coordinates.

### `DualRootLatticeD<const N: usize>`

The dual lattice `D_N*`, i.e. `RootLatticeD::<N>::DualLattice`. Stored as
integer coefficients in the basis `e_1, ..., e_{N-1}, v`, where
`v = (1/2, ..., 1/2)` is the glue vector generating `D_N* = Z^N + Z*v` over
the ambient `Z^N` of `D_N`'s standard embedding (so `<e_i, e_j> = delta_ij`,
`<e_i, v> = 1/2`, `<v, v> = N/4`). Discriminant `1/4` for all `N >= 1`; not
integral, since e.g. `<e_1, v> = 1/2`. `short_vectors` currently returns
`Err(ShortVectorError::Unknown)`.

### `RootLatticeE8`

An element of the `E8` root lattice (fixed rank 8, no generic parameter),
stored as integer coefficients `[i128; 8]` in the basis of simple roots
`alpha_1, ..., alpha_8` (Bourbaki labeling), with Gram matrix the `E8`
Cartan matrix: the chain `1-3-4-5-6-7-8` with node `2` attached to node `4`.
Even and unimodular (discriminant `1`) — the unique even unimodular lattice
of rank 8, so `DualLattice = Self`. `short_vectors` returns all 240 roots,
computed via reflection closure of the simple roots: since the Weyl group
of a simply-laced root system acts transitively on roots, repeatedly
applying `s_i(r) = r - <r, alpha_i> * alpha_i` (valid since every simple
root has norm 2) to the 8 simple roots until no new vectors appear yields
the full root system — there's no simple closed form in the simple-root
coordinates used here, unlike `A_N`/`D_N`.

### `DirectSumLattice<L1, L2, const N: usize, const M: usize, const N_PLUS_M: usize>`

The orthogonal direct sum `L1 ⊕ L2` of a rank-`N` lattice and a rank-`M`
lattice (no cross terms in `inner_product`), e.g. the K3 lattice
`U ⊕ U ⊕ U ⊕ E8(-1) ⊕ E8(-1)`. Stable Rust has no `N + M` in const-generic
position, so the caller supplies `N_PLUS_M` explicitly; `DirectSumLattice::new`
checks at compile time (via a `const` `assert!`) that it actually equals
`N + M`.

- `is_integral`/`is_even` require both summands to satisfy it.
- `signature` adds component-wise (eigenvalues of a block-diagonal Gram
  matrix are the union of the blocks' eigenvalues).
- `discriminant` multiplies (det of a block-diagonal matrix is the product
  of the blocks' dets); `is_self_dual` checks `discriminant() == 1` directly
  rather than `L1::is_self_dual() && L2::is_self_dual()`, since two
  non-self-dual rational summands could still have reciprocal discriminants.
- `DualLattice = DirectSumLattice<L1::DualLattice, L2::DualLattice, N, M, N_PLUS_M>`.
- `short_vectors` returns the lifted minimal vectors of whichever summand
  has the smaller minimal norm (both, if tied) — any vector nonzero in both
  factors has norm at least the sum of the two minimal norms, so it's never
  itself minimal.

## Errors

### `ShortVectorError`

Returned by `Lattice::short_vectors`. `NotPositiveDefinite` signals that the
lattice's form is not positive definite, so the minimal-norm vectors need
not form a finite set; `Unknown` signals a positive-definite lattice whose
minimal vectors this implementation does not (yet) compute.

## Dependencies

- [`num`](https://crates.io/crates/num) — `Zero` and `Ratio<i128>` for the
  rational-valued inner product.
