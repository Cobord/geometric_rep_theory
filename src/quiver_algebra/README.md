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

### `PathAlgebra<V, E, Coeffs>`

An element of kQ^{op} — a formal linear combination of paths and idempotents. Supports addition, multiplication, and simplification modulo a coefficient ring.

### `QuiverWithRelations<V, E, Coeffs>`

A quiver together with a list of relations (elements of `PathAlgebra` set to zero). Can be constructed from a superpotential W via cyclic derivatives.

### `QuiverWithMonomialRelations<V, E>`

A quiver with monomial (path) relations — each relation is a single path declared zero. The ideal is stored as an antichain (no redundant generators).

### `QuiverRep<V, E, M>`

A representation of a quiver: assigns a matrix `M` to each arrow (and an idempotent matrix to each vertex), consistent with the quiver structure.

### `QuiverBimodule` (trait)

A trait for (A, A)-bimodules over A = kQ^{op}/I^{op}, accessed via the Peirce decomposition. Implementors provide:

- `peirce_basis(v, w)` — ordered basis of e_v A e_w
- `left_act(α, elt)` — left action of arrow α; requires `elt.left = t(α)`, returns element with `left = s(α)`
- `right_act(β, elt)` — right action of arrow β; requires `elt.right = s(β)`, returns element with `right = t(β)`

The library then builds left/right action matrices for arbitrary algebra elements (`left_act_algebra_elt`, `right_act_algebra_elt`) and can verify all three bimodule axioms (`check_bimodule_axioms`).

#### `PeirceElement<V, Coeffs>`

An element of the Peirce piece e_{left} M e_{right}, stored as a coordinate vector in the basis given by `peirce_basis(left, right)`.

### `DiagonalBimodule`

The algebra A = kQ^{op}/I^{op} viewed as an (A, A)-bimodule over itself. The Peirce piece e_v A e_w is spanned by admissible paths from v to w in Q (paths not containing any forbidden monomial as a subpath). Precomputes left and right action matrices for all arrows.

### `MonomialQuiverAlgebraHH`

Hochschild cohomology of A = kQ^{op}/I^{op} for monomial relations. Builds the Bardzell resolution and computes cohomology dimensions degree by degree.

## Dependencies

- [`petgraph`](https://crates.io/crates/petgraph) — quiver graph structure
- [`nalgebra`](https://crates.io/crates/nalgebra) — matrices for bimodule action maps
- [`nonempty`](https://crates.io/crates/nonempty) — non-empty collections
- [`itertools`](https://crates.io/crates/itertools) — iterator utilities
