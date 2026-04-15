# arithmetic_utils

Shared algebraic traits and matrix types used across all modules.

## Traits

### `SemiRing`

A commutative monoid under `+` and a monoid under `*`, with distributivity. No subtraction. Blanket-implemented for any type satisfying `Add + AddAssign + Mul + MulAssign + Clone + One + Zero`.

Provides `natural_inclusion(n: usize) -> Self` — the image of n ∈ ℕ in the semiring, computed via binary doubling in O(log n) additions.

### `Ring`

A `SemiRing` that also supports subtraction and negation (`Sub + SubAssign + Neg`). Blanket-implemented.

### `Field`

A `Ring + PartialEq` with a multiplicative inverse `inv(self) -> Self`. Blanket-implemented for types with `Div<Self, Output=Self>`. Used by the Gaussian elimination `rank` function.

### `CheckedAdd` / `CheckedAddAssign`

Fallible addition returning a typed error — used where dimension mismatches (e.g. matrix shapes) must be reported rather than panicked. Blanket-implemented for all `Add`/`AddAssign` types with a trivial `()` error.

### `ChainMultiplyable`

Fallible sequential composition: `mul_two(A, B)` means "do A then B", corresponding to right-multiplication (`B * A`). Supports `chain_multiply_after` for composing a sequence. Blanket-implemented for all `Mul + MulAssign` types.

## Structures

### `DynMatrix<T>`

A dynamically-sized matrix wrapping `nalgebra::DMatrix<T>`, implementing `CheckedAdd`, `CheckedAddAssign`, and `ChainMultiplyable` with dimension-mismatch errors. Multiplication follows the **opposite-algebra convention** (`mul_two(A, B)` = B * A), matching the kQ^{op} path algebra convention used in `quiver_algebra`.

## Functions

### `rank`

Gaussian elimination over a `Field` coefficient type. Returns the row rank of a `&[Vec<Scalar>]` matrix. Used by the Hochschild cohomology computation to extract Betti numbers from chain complex differentials.

## Dependencies

- [`nalgebra`](https://crates.io/crates/nalgebra) — backing store for `DynMatrix`
- [`nonempty`](https://crates.io/crates/nonempty) — `NonEmpty` in `ChainMultiplyable::nonempty_chain_multiply`
- [`num`](https://crates.io/crates/num) — `Zero`, `One`, and integer `gcd`
