# quiver_algebra

Rust library for path algebras of quivers with relations, their bimodules, and Hochschild cohomology.

## Mathematical background

A **quiver** Q is a directed multigraph with vertices V and arrows (edges) E. Each arrow α has a source s(α) and target t(α). A **path** in Q is a sequence of composable arrows.

The **path algebra** kQ^{op} has basis consisting of all paths and idempotents, with multiplication given by left-to-right concatenation: the product of path p followed by path q (when t(p) = s(q)) is the path pq. Arrow α: s(α) → t(α) lies in the Peirce piece e_{s(α)} kQ^{op} e_{t(α)}.

A **quiver with relations** is a quiver together with elements of the path algebra that are declared to be zero, giving the quotient algebra A = kQ^{op}/I^{op}.

## Path convention

`BasisElt::Path([a₁, a₂, …, aₙ])` stores arrows **left-to-right**: follow a₁ first, then a₂, …, then aₙ. This gives the algebra kQ^{op}. For paths p ∈ e_v kQ^{op} e_w and q ∈ e_w kQ^{op} e_s, the product p·q ∈ e_v kQ^{op} e_s.

## Modules

### `Quiver<V, E>`

A directed graph with user-supplied vertex labels `V` and edge labels `E`. Backed by `petgraph::StableGraph`.

```rust
let mut q = Quiver::new();
q.add_edge("0", "1", "a");
q.add_edge("1", "2", "b");
```

### `PathAlgebra<V, E, Coeffs, const OP_ALG: bool>`

A formal linear combination of paths and idempotents over the coefficient ring `Coeffs`. The `OP_ALG` const generic selects which multiplication convention `Mul` and `MulAssign` use:

- **`OP_ALG = true`** (kQ^{op}): `p * q` in Rust concatenates left-to-right — follow `p` first, then `q`. Requires the last arrow of `p` to compose with the first arrow of `q`. This is the default convention used throughout the library.
- **`OP_ALG = false`** (kQ): `p * q` in Rust concatenates right-to-left — follow `q` first, then `p`. Concretely, `Mul` swaps the arguments before concatenating, so `l * r` produces the path `[r, l]`. This matches the function-composition convention where arrows compose like maps.

Both conventions use the same `BasisElt::Path([a₁, …, aₙ])` storage (left-to-right arrow order); only the meaning of the `*` operator differs.

### `QuiverWithRelations<V, E, Coeffs, const OP_ALG: bool>`

A quiver together with a list of relations (elements of `PathAlgebra<..., OP_ALG>` set to zero). Can be constructed from a superpotential W via cyclic derivatives. `OP_ALG` must match the `PathAlgebra` convention in which the relations are written. It controls whether this is interpreted as thinking about `kQ/I` vs `kQ^op/I^op` algebras.

### `QuiverWithMonomialRelations<V, E>`

A quiver with monomial (path) relations — each relation is a single path declared zero. The ideal is stored as an antichain (no redundant generators).

### `QuiverRep<V, E, M, const OP_ALG: bool>`

A representation of a quiver: assigns a matrix `M` to each arrow (and an idempotent matrix to each vertex), consistent with the quiver structure. `OP_ALG` is a type-safety tag that constrains `mat_from_path_algebra` to only accept `PathAlgebra<..., OP_ALG>` elements — it does not change how individual arrow matrices are stored or evaluated (path evaluation always composes `M(a₁).then(M(a₂)).then(…)` left-to-right regardless of `OP_ALG`).

### `QuiverBimodule<V, E, Coeffs, BasisLabel, const OP_ALG: bool>` (trait)

A trait for (A, A)-bimodules over `A = kQ^{op}/I^{op}` or `A=kQ/I`, accessed via the Peirce decomposition. `OP_ALG` propagates to the underlying `QuiverWithRelations` returned by `algebra()`, tying the bimodule to a specific path algebra convention. Implementors provide:

- `peirce_basis(v, w)` — ordered basis of e_v N e_w (under the kQ^{op} convention; with kQ this would be e_w N e_v)
- `left_act(α, elt)` — left action of arrow α; requires `elt.left = t(α)`, returns element with `left = s(α)`
- `right_act(β, elt)` — right action of arrow β; requires `elt.right = s(β)`, returns element with `right = t(β)`

The library then builds left/right action matrices for arbitrary algebra elements (`left_act_algebra_elt`, `right_act_algebra_elt`) and can verify all three bimodule axioms (`check_bimodule_axioms`).

#### `PeirceElement<V, Coeffs>`

An element of the Peirce piece `e_{left} M e_{right}`, stored as a coordinate vector in the basis given by `peirce_basis(left, right)`.

### `DiagonalBimodule<V, E, Coeffs, const OP_ALG: bool>`

The algebra A = kQ^{op}/I^{op} or kQ/I viewed as an (A, A)-bimodule over itself. The Peirce piece e_v A e_w is spanned by admissible paths from v to w in Q (paths not containing any forbidden monomial as a subpath). Precomputes left and right action matrices for all arrows. `OP_ALG` is inherited from the `QuiverWithRelations` it wraps and must be consistent with it.

### `MonomialQuiverAlgebraHH`

Hochschild cohomology of A = kQ^{op}/I^{op} for monomial relations. Builds the Bardzell resolution and computes cohomology dimensions degree by degree.

### `DegreeLabel<T>` / `HasHomologicalDegree`

An edge label wrapper that attaches an explicit integer homological degree. Implements `HasHomologicalDegree`, a trait for retrieving the degree of edges, paths, and `PathAlgebra` elements. Standard Ginzburg degrees:

- original arrows a → degree 0
- adjoint arrows a\* → degree −1
- self-loop generators ω_v → degree −2

`DegreeLabel::dagger(rename)` produces the closure used to build adjoint arrows with degree `−deg(a) − 1`, keeping the Ginzburg cubic homogeneous of degree −3.

### `GradedDifferentialQuiver<V, E, Coeffs, const OP_ALG: bool>`

A differential graded path algebra (DGA): a graded quiver Q together with a degree-+1 derivation `d` on kQ satisfying `d² = 0` and the graded Leibniz rule. The differential is stored as a map from arrow labels to `PathAlgebra<..., OP_ALG>` elements, so `OP_ALG` determines the path multiplication convention used when applying `d` and when constructing the Ginzburg DGA.

Key methods:
- `apply_differential` — apply `d` to any algebra element, respecting Koszul signs
- `create_ginzburg_dga` — construct the Ginzburg DGA `Γ(Q, W)` from a quiver and potential W; adds adjoint arrows a\* and self-loops ω_v with differentials `d(a*) = ∂W/∂a` and `d(ω_v)` given by the cubic term

### `DGModule<V, E, MatrixType, const OP_ALG: bool>`

A left DG-module M over a `GradedDifferentialQuiver` `(kQ, d_kQ)`. Pairs a `QuiverRep<..., OP_ALG>` (arrow actions as matrices) with vertex differentials `d_M` (degree-+1 endomorphisms). `OP_ALG` propagates to both the inner `QuiverRep` and the `GradedDifferentialQuiver` passed to `leibniz_compatible`, ensuring all three are using the same path algebra convention. Provides:

- `leibniz_compatible` — checks `d_{t(a)} ∘ ρ(a) = (−1)^|a| ρ(a) ∘ d_{s(a)} + ρ(d(a))` for each arrow
- `differential_squares_zero` — checks `d_{M,v}² = 0` at every vertex

### `DynMatrix<T>`

A wrapper around `nalgebra::DMatrix<T>` implementing the `CheckedAdd`, `CheckedAddAssign`, and `ChainMultiplyable` traits with dimension-mismatch errors instead of panics. Multiplication is reversed (opposite algebra convention), matching the `OP_ALG = true` path algebra convention.

### Checked arithmetic traits

- `Ring` / `SemiRing` / `Field` — algebraic hierarchy for coefficient types
- `CheckedAdd` / `CheckedAddAssign` — fallible addition returning a shape-mismatch error
- `ChainMultiplyable` — fallible matrix multiplication: `mul_two(A, B)` computes A then B

## Dependencies

- [`petgraph`](https://crates.io/crates/petgraph) — quiver graph structure
- [`nalgebra`](https://crates.io/crates/nalgebra) — matrices for bimodule and DG-module action maps
- [`nonempty`](https://crates.io/crates/nonempty) — non-empty collections
- [`num`](https://crates.io/crates/num) — `Integer` trait for Koszul sign parity checks
- [`itertools`](https://crates.io/crates/itertools) — iterator utilities
