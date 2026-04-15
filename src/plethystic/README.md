# Plethystic module

This module implements the ring of symmetric functions Λ and the plethystic
substitution action of Λ on any λ-ring.

## Structures

### `LambdaRing`

A trait extending `Ring` with the lambda operations λ^n and Adams operations ψ^n.

**Required method:** `lambda(self, n: usize) -> Self`  
**Defaulted method:** `psi(self, n: usize) -> Self`

The default `psi` is derived from `lambda` via Newton's identity iterated
bottom-up (O(n²) λ-operations). Implementors with a direct formula for ψ^n
should override it.

**Newton's identity (λ from ψ direction):**

```
n · λ^n = Σ_{k=1}^{n} (-1)^{k-1} ψ^k · λ^{n-k}
```

**Newton's identity (ψ from λ direction, used by the default `psi`):**

```
ψ^n = Σ_{k=1}^{n-1} (-1)^{k-1} λ^k · ψ^{n-k}  +  (-1)^{n-1} n · λ^n
```

Seeing that defining `ψ^n` from `λ^n` does not require anything beyond the `Ring` bound already present but the other direction requires division by `n`, we see
that `ψ^n` is given a default implementation and `λ^n` must be provided instead of vice versa.

### `PowerSumPolynomial`

A basis element p_λ of Λ, indexed by a partition λ stored in descending order.
Multiplication concatenates partitions: p_μ · p_ν = p_{μ ∪ ν}.

### `SymmetricFunction<Coeffs>`

The ring Λ_Coeffs = Λ_ℤ ⊗_ℤ Coeffs, stored as a `HashMap<PowerSumPolynomial, Coeffs>`
— i.e. a finite formal sum Σ c_λ p_λ in the power-sum basis.

Implements `Ring`, `Zero`, `One`, and (when `Coeffs: Div<usize, Output=Coeffs>`) `LambdaRing`.

## Plethystic substitution

For f ∈ Λ_Coeffs and x in a λ-ring L there is an action

```
f[x] = Σ_λ c_λ · p_λ[x]
```

where each basis element acts by Adams operations:

```
p_λ[x] = ψ^{λ_1}(x) · ψ^{λ_2}(x) · … · ψ^{λ_k}(x)
```

This is computed by `SymmetricFunction::plethysm`. The p-basis is the natural
one for plethysm because it makes the formula above a product of Adams
operations with no basis conversion needed.


## `SymmetricFunction<Coeffs>` as a λ-ring

When `Coeffs: Ring + Div<usize, Output=Coeffs>` (e.g. rational coefficients or at least an inclusion of the rationals),
`SymmetricFunction<Coeffs>` is itself a λ-ring. The Adams operation ψ^n acts
directly on the p-basis by scaling every part:

```
ψ^n(Σ c_λ p_λ) = Σ c_λ p_{n·λ}
```

where n·λ = (n·λ_1, n·λ_2, …). The lambda operations are then recovered from
this via Newton's identity (requiring division by n, hence `Div<usize>`). This is a necessity of the formula because we do not have the guarantee of divisibilty.

## Cross-coefficient action

`SymmetricFunction<A>` can act on `SymmetricFunction<B>` via `plethysm` when:

- `SymmetricFunction<B>: LambdaRing` (requires `B: Div<usize, Output=B>`)
- `SymmetricFunction<B>: MulAssign<A>` (scaling by the `A` coefficients)

The `MulAssign<A>` impl cannot be provided generically (it would conflict with
`MulAssign<Self>`), but is straightforward to implement for concrete pairs via
`scale_by`.

## Helpers

| Method | Description |
|---|---|
| `SymmetricFunction::singleton(parts, c)` | Single-term p_λ with coefficient c |
| `SymmetricFunction::coerce_coeffs::<B>()` | Coefficient-wise coercion via `B: From<A>` |
| `SymmetricFunction::scale_by(scalar)` | In-place scalar multiplication |
| `From<Coeffs> for SymmetricFunction<Coeffs>` | Embed a constant as the p_∅ = 1 term |

## Dependencies

- [`arithmetic_utils`](../arithmetic_utils/README.md) — `Ring`
- [`num`](https://crates.io/crates/num) — `Zero` and `One`
