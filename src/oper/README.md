# oper

Differential operators on affine n-space over a coefficient ring `R`.

An element of the Weyl algebra A_n(R) is represented as a finite sum

```
Σ_α  p_α(x) · D^α
```

where α ∈ ℕ^n is a multi-index, p_α ∈ R[x_1,…,x_n] is a polynomial
coefficient, and D^α = ∂^{α_1}/∂x_1^{α_1} · … · ∂^{α_n}/∂x_n^{α_n}.

## Structures

### `DifferentialOperator<R: Ring, const N: usize>`

Stored as a `HashMap<[usize; N], PowerSeries<R, N>>` mapping each multi-index
α to its coefficient polynomial p_α. Zero coefficients are not stored.

Implements `Ring` (via `Add`, `Sub`, `Neg`, `Mul`) and `Zero`/`One`.

**Multiplication** is operator composition via `differential_act`: applying the
left operator term-by-term to the right operator using the Leibniz rule (see
below). The identity operator is `1 · D^0`.

**Differential order** — `differential_order()` returns max |α| over all
nonzero terms, where |α| = Σ α_i is the total order.

## Traits

### `NVarDifferentiable<const N: usize>`

Any type that can be differentiated by a multi-index `D^α`:

```rust
fn differentiate(&self, d_op: [usize; N]) -> Self;
```

Implemented for `DifferentialOperator` via the Leibniz rule (see below), and
intended for any other type that supports the `differential_act` API.

### `LeftMul<LHS>`

Left-multiplication by a value of type `LHS`:

```rust
fn left_mul(self, lhs: LHS) -> Self::Output;
```

Implemented for `DifferentialOperator` with `LHS = PowerSeries<R, N>`, where it
scales every coefficient polynomial in place. Used internally by `differential_act`.

## Operator composition: the Leibniz rule

Applying `D^d` to `p_α(x) · D^α` gives, by the generalised product rule:

```
D^d · (p_α(x) D^α) = Σ_{β ≤ d}  C(d, β) · (D^β p_α)(x) · D^{d−β+α}
```

where the sum is over all multi-indices β with β_i ≤ d_i for each i, and
C(d, β) = ∏_i C(d_i, β_i) is the multinomial binomial coefficient.

Summing over all terms of the right-hand operator:

```
D^d · (Σ_α p_α D^α) = Σ_α Σ_{β ≤ d} C(d,β) · (D^β p_α) · D^{d−β+α}
```

`differential_act` implements full operator composition by applying every term
of `self` to `rhs` and accumulating:

```
(Σ_α p_α D^α) · rhs = Σ_α p_α · (rhs.differentiate(α))
```

## Example: quantum harmonic oscillator (ℏ = 1)

With mass m and frequency ω, the ladder operators are

```
a  = √(mω/2) · x  +  (1/√(2mω)) · ∂
a† = √(mω/2) · x  −  (1/√(2mω)) · ∂
```

Their product `a†a` is computed by `Mul` and gives, after the D^1 terms cancel
exactly via the canonical commutation relation [∂, x] = 1:

```
a†a = −(1/2mω) · D_x²  +  (mω/2) · x²  −  1/2
    = H/ω − 1/2
```

where H = p²/2m + mω²x²/2 is the Hamiltonian.

## Dependencies

- [`plethystic`](../plethystic/README.md) — `PowerSeries<R, N>` for coefficient polynomials
- [`arithmetic_utils`](../arithmetic_utils/README.md) — `Ring`, `binom`, `multi_index_le`
