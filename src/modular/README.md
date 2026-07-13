# modular

Modular forms, and the matrix groups they transform under. A lattice's theta
series (see
[`Lattice::theta_series_weight`](../lattice/README.md) and
`Lattice::theta_series_character`) is the motivating example of a
`ModularForm`: this module models the transformation law itself,
independently of any particular lattice.

A weight-`w` modular form for a group `G` satisfies
`f(g*tau) = chi(g, tau) * (c*tau+d)^w * f(tau)` for every `g = (a b; c d)` in
`G`, where `chi` is `G`'s multiplier system. `w` only ever ranges over `Z` or
`Z + 1/2`; it's tracked as `TWICE_WEIGHT: usize` (twice the actual weight)
rather than as a general `Ratio`, since only a structural type can be a const
generic and every classical weight has denominator `1` or `2` — illegal
denominators like `1/3` are unrepresentable rather than rejected at runtime.
`R` is how complex numbers are parameterized (`Complex<f64>` throughout this
module, but kept generic via the `Field` trait), so that `tau`/`q` and the
integer matrix entries `a, b, c, d` (embedded in `R`, not ranging over all of
it) can be combined arithmetically.

## Traits

### `ModularTransformationGroup<R: Field>`

Some group `G`, equipped with a multiplier system (a character
`chi: G x H -> R`) and an action on the upper half-plane by Mobius
transformations. `G` is *not* required to literally be a subgroup of
`SL_2(Z)`/`GL_2(Z)` — `raw_a`..`raw_d` are only the integer matrix data
needed to write down the Mobius/`q`-disk action and the automorphy factor
`(c*tau+d)^w`, not a claim that this data identifies `G` with a subgroup of
`GL_2(Z)`. `Sl2Z` is the case where it does, with a trivial multiplier;
`EtaTransformationGroup` is the case where it doesn't: its `G` is (a piece
of) the metaplectic double cover `Mp_2(Z)` of `SL_2(Z)`, whose elements carry
a consistent choice of square root of `c*tau+d` alongside `g` — a choice
`SL_2(Z)` alone can't pin down, since `g` and `-g` induce the same Mobius
transformation but need opposite square roots.

- `new(raw_matrix: [[R; 2]; 2]) -> Result<Self, ModularError>` — builds the
  element of `G` whose action/multiplier are given by an `SL_2(Z)` (or
  `GL_2(Z)`) matrix. For a metaplectic-type `G` that matrix has two preimages
  under the covering surjection `G -> SL_2(Z)`; `new` resolves it to one of
  them, in whatever way the implementor chooses — there's no way to do this
  consistently across every matrix at once as a group homomorphism, since the
  cover is nontrivial, but `new` never needs one, only a single preimage per
  call. Fails with `ModularError::NonIntegerEntry` if an entry isn't (within
  tolerance) an integer, or `ModularError::NotInGroup` if the integer matrix
  fails this group's membership condition (e.g. determinant `1` for
  `SL_2(Z)`, or a congruence condition for a congruence subgroup).
- `transform_q(&self, q: &R) -> R` — the action on the nome `q` directly,
  rather than via `tau` and `q = exp(2*pi*i*tau)`; avoids `exp`/`log`, which
  most `R` don't even have.
- `transform_tau(&self, tau: &R) -> Option<R>` — the Mobius action
  `tau -> (a*tau+b)/(c*tau+d)`. Defaulted in terms of `raw_a`..`raw_d`.
  Returns `None` when `c*tau+d = 0`, since `tau` is then sent to the point at
  infinity rather than to another value of `R`.
- `multiplier_system(&self, tau: &R) -> R` — `chi(g, tau)`. Takes `tau`
  because, for half-integer `w`, `(c*tau+d)^w` means the *preferred*
  (principal-branch) square root of `c*tau+d` raised to `2*w`, and correctly
  computing `chi` for a metaplectic-type `G` can require evaluating that
  preferred square root at the given `tau`.
- `is_trivial_multiplier_system() -> bool` — whether `multiplier_system` is
  identically `1` on `G` (e.g. `Sl2Z`) rather than a genuine character (e.g.
  `EtaTransformationGroup`) — a fact about this particular `G`, letting a
  caller skip evaluating `chi` where it's known to be `1`.
- `raw_a`/`raw_b`/`raw_c`/`raw_d(&self) -> R` — the four matrix entries,
  embedded in `R`.
- `raw_matrix(&self) -> [[R; 2]; 2]` — the four entries assembled into a
  matrix. Defaulted in terms of `raw_a`..`raw_d`.

### `ModularForm<const TWICE_WEIGHT: usize, R: Field>`

A modular form of weight `TWICE_WEIGHT / 2`, transforming under
`Self::TransformationGroup: ModularTransformationGroup<R>`.

- `extract_coeffs(&self, which_coeff: usize) -> Result<R, ModularError>` —
  the coefficient of `q^(which_coeff + kappa)`, where `kappa` is whatever
  fractional shift is forced by `TransformationGroup`'s multiplier at
  `T = (1 1; 0 1)` (`v(T) = exp(2*pi*i*kappa)`). `kappa` is `0` for a trivial
  multiplier (e.g. `Sl2Z`, where `which_coeff` really is just the power of
  `q`) but nonzero for a genuinely half-integer-weight group
  (`EtaTransformationGroup` gives `kappa = 1/24`). Not a per-implementor
  choice: `kappa` is pinned down by `TransformationGroup` alone, so every
  `ModularForm` sharing a `TransformationGroup` indexes `which_coeff` against
  the same `kappa` — exactly what lets `SumModularForm` add up
  `which_coeff`-th coefficients across summands term-by-term. Fails with
  `ModularError::Unavailable` if the coefficient isn't available (not yet
  computed, or out of range for a form known only to finite precision).
- `evaluate_at(&self, q: &R) -> Result<R, ModularError>` — the value of the
  form at the nome `q`. Fails with `ModularError::Unavailable` under the same
  circumstances as `extract_coeffs`.

## Structures

### `Sl2Z`

`SL_2(Z)` itself, represented over `Complex<f64>` — just enough to be a
concrete `ModularTransformationGroup` for `EisensteinE4`/`EisensteinE6` (and
other weight-`k`, level-`1`, trivial-character forms) to transform under.
`new` rejects non-integer entries and determinant `!= 1`.
`is_trivial_multiplier_system` is always `true`, and `multiplier_system`
always returns `1`.

### `EtaTransformationGroup`

The metaplectic double cover `Mp_2(Z)` of `SL_2(Z)`, over `Complex<f64>`,
carrying the Dedekind eta function's character. `new` accepts every
determinant-`1` integer matrix, including `-g` for any accepted `g` — `g` and
`-g` induce the same Mobius transformation but need different `chi`, which is
exactly why this is genuinely metaplectic rather than just `SL_2(Z)`.
`multiplier_system(g, tau)` gives the `24`th-root-of-unity eta character
`chi(g, tau)`:

- For `c > 0`: `exp(i*pi * ((a+d)/(12*c) - s(d,c) - 1/4))`, where `s(h,k)` is
  the Dedekind sum (`arithmetic_utils::dedekind_sum`) — independent of `tau`.
- For `c = 0`, `a = d = 1` (`g = T^b`): `exp(i*pi*b/12)`.
- Otherwise (`c < 0`, or `c = 0` with `a = d = -1`): recovered from
  `chi(-g, tau)` (one of the above cases) via
  `chi(g, tau) = chi(-g, tau) * sqrt(-(c*tau+d)) / sqrt(c*tau+d)`, both square
  roots taken with the preferred (principal) branch.

Sanity-checked against `SL_2(Z)`'s generators: `S = (0 -1; 1 0)` gives
`chi(S, tau) = exp(-i*pi/4)`, matching `eta(-1/tau) = sqrt(-i*tau)*eta(tau)`;
`T = (1 1; 0 1)` gives `chi(T, tau) = exp(i*pi/12)`, matching
`eta(tau+1) = exp(i*pi/12)*eta(tau)`. `is_trivial_multiplier_system` is
always `false`.

### `UnitModularForm`

The constant modular form `f = 1`, of weight `0` (`TWICE_WEIGHT = 0`),
transforming under `Sl2Z`. The multiplicative identity once a product
combinator exists (`f * g = g` for any modular form `g` sharing its
transformation group), and trivially modular under any group with trivial
multiplier system, since `f(g*tau) = v(g) * (c*tau+d)^0 * f(tau) = 1 = f(tau)`.

### `EisensteinE4`

The weight-`4` Eisenstein series `E4 = 1 + 240 * sum_{n>=1} sigma_3(n) q^n`
(`sigma_3(n)` the sum of cubes of divisors of `n`, via
`arithmetic_utils::sigma_3`), transforming under `Sl2Z`. The prototypical
`ModularForm`: it spans the one-dimensional space of weight-`4` level-`1`
forms, and is exactly the theta series of the `E8` root lattice (see
[`lattice`](../lattice/README.md)'s `RootLatticeE8`). `evaluate_at` is
unimplemented (`Err(Unavailable)`) — only the `q`-expansion is computed.

### `EisensteinE6`

The weight-`6` Eisenstein series `E6 = 1 - 504 * sum_{n>=1} sigma_5(n) q^n`
(`sigma_5(n)` the sum of fifth powers of divisors, via
`arithmetic_utils::sigma_5`), transforming under `Sl2Z`. Together with
`EisensteinE4`, `E6` generates the entire ring of level-`1` modular forms —
every such form is a polynomial in `E4` and `E6`. `evaluate_at` is likewise
unimplemented.

### `DedekindEta`

The Dedekind eta function `eta(tau) = q^{1/24} * prod_{n>=1}(1 - q^n)`,
transforming under `EtaTransformationGroup` — the prototypical weight-`1/2`
form (`TWICE_WEIGHT = 1`), the only one in this module with a genuinely
nontrivial multiplier. Per `ModularForm::extract_coeffs`'s convention,
`which_coeff` indexes the coefficient of `q^(which_coeff + 1/24)`, so it
directly indexes the coefficients of `prod_{n>=1}(1 - q^n)`
(`arithmetic_utils::euler_function_coeff`), with the `q^{1/24}` prefactor
accounted for entirely by `kappa` rather than folded into `which_coeff`.
`evaluate_at` is unimplemented.

### `ProductModularForm<R, F1, F2, const TWICE_WEIGHT_1, const TWICE_WEIGHT_2, const TWICE_WEIGHT_SUM>`

The product `f * g` of two modular forms, of weight
`(TWICE_WEIGHT_1 + TWICE_WEIGHT_2) / 2`, transforming under
`CombinedTransformationGroup<F1::TransformationGroup, F2::TransformationGroup>`
— `f` and `g` need not individually have a trivial multiplier, nor even share
the same transformation group type. Stable Rust has no
`TWICE_WEIGHT_1 + TWICE_WEIGHT_2` in const-generic position, so the caller
supplies `TWICE_WEIGHT_SUM` explicitly; `ProductModularForm::new` checks at
compile time (via a `const` `assert!`) that it actually equals
`TWICE_WEIGHT_1 + TWICE_WEIGHT_2`. `extract_coeffs` is the Cauchy product of
the two factors' `q`-expansions (`c_n = sum_{k=0}^n a_k * b_{n-k}`);
`evaluate_at` multiplies the two factors' values.

### `CombinedTransformationGroup<G1, G2>`

The intersection `G1 ∩ G2` of two transformation groups — `new` only accepts
a matrix if both `G1::new` and `G2::new` accept it — with their multiplier
systems multiplied together (`v(g) = v_first(g) * v_second(g)`). The
transformation-group-level analogue of `ProductModularForm`, e.g. combining
`Sl2Z`'s trivial multiplier with a genuinely nontrivial one to get the group
a product of such forms transforms under. `raw_a`..`raw_d` and `transform_q`
are delegated to `first` (equivalently `second` — membership in the
intersection means both were built from the same `raw_matrix`, so they agree
by construction). `is_trivial_multiplier_system` is the conjunction of both
factors'.

### `square_modular_form` / `cube_modular_form`

Free functions building `ProductModularForm` instances for `t * t` and
`t * t * t` from a single `T: ModularForm<TWICE_WEIGHT, R> + Clone`, taking
care of the `TWICE_WEIGHT_SUM` bookkeeping (`FOUR_WEIGHT`, `SIX_WEIGHT`) so
the caller doesn't have to construct nested `ProductModularForm`s by hand.
Their result is typed with `TransformationGroup =
CombinedTransformationGroup<T::TransformationGroup, T::TransformationGroup>`
(possibly nested further for the cube), regardless of whether
`T::TransformationGroup`'s multiplier happens to be trivial.

### `square_modular_form_trivial` / `cube_modular_form_trivial`

Variants of `square_modular_form`/`cube_modular_form` that additionally
coerce the result down to `TransformationGroup = T::TransformationGroup`
itself, via `CoerceTransformation::FORCED` and `EquivalentTransportedMF` —
sound because a trivial multiplier system stays trivial under any number of
products (`1 * 1 = 1`), so `CombinedTransformationGroup<G, G>` (or nestings
of it) really is "the same" group as `G` alone whenever `G`'s multiplier is
trivial. Saves callers the manual `EquivalentTransportedMF::coerce` step
otherwise needed to get, say, `E4^3` and `E6^2` onto a nameably common
`Sl2Z`-typed `TransformationGroup` so they can sit in one `SumModularForm`.
Panics if `T::TransformationGroup::is_trivial_multiplier_system()` is
`false`, since the coercion would then be unsound (a genuine character's
square/cube need not equal the character itself).

### `SumModularForm<const TWICE_WEIGHT: usize, R, TRANSFORM>`

A formal linear combination `sum_i coeff_i * summand_i` of `Box<dyn
ModularForm<TWICE_WEIGHT, R, TransformationGroup = TRANSFORM>>`s, all sharing
the same weight and transformation group — and so itself a modular form of
that weight and group, since modular forms of fixed weight and group form a
vector space over `R`. Sharing a `TransformationGroup` also means every
summand indexes `which_coeff` against the same `kappa`, so adding
`which_coeff`-th coefficients across summands is combining like terms rather
than nonsense.

- `extract_coeffs`/`evaluate_at` — each summand's coefficient/value, combined
  with the same weights as in the sum.
- `new_from_some_coeffs<const DIM_SPACE: usize>(basis_of_space, constrained_coeffs_values) -> Result<Self, ModularError>`
  — constructs the unique element of the `DIM_SPACE`-dimensional space
  spanned by `basis_of_space` satisfying a prescribed set of linear
  constraints. Each constraint is `Ok((idx, value))` ("the `idx`-th
  `q`-expansion coefficient is `value`") or `Err((q, value))` ("the form
  evaluates to `value` at nome `q`"); the first `DIM_SPACE` constraints
  determine a `DIM_SPACE x DIM_SPACE` linear system for the coordinates in
  `basis_of_space`, solved by matrix inversion (via `nalgebra`). E.g. the
  weight-12 cusp form `Delta` is the unique element of the 2-dimensional
  space spanned by `E4^3` and `E6^2` with vanishing constant term and unit
  `q`-coefficient — recovering `Delta = (E4^3 - E6^2) / 1728`. Fails with
  `ModularError::NotEnoughConstraints` if fewer than `DIM_SPACE` constraints
  are given, `ModularError::SingularSystem` if the system is singular, or
  `ModularError::NonFiniteCoefficient` if a solved coefficient isn't finite.

### `CoerceTransformation<R, T1, T2>`

A witness — currently only the single variant `FORCED`, an unchecked
assertion — that `T1` and `T2` are "the same" `ModularTransformationGroup`
(same group, same multiplier system), letting a form built against `T1` be
reinterpreted as one against `T2` via `EquivalentTransportedMF`.

- `validate(self, gens_t1: &[T1], gens_t2: &[T2], close_enough) -> Result<Self, ModularError>`
  — empirically spot-checks the `FORCED` claim instead of trusting it
  blindly: for each generator of `T1`, rebuilds the same matrix as a `T2`
  element and checks the multiplier systems agree at a handful of
  pseudo-random sample points in `H` (and symmetrically for `T2`'s
  generators). This only proves agreement on the supplied generators at
  those sample points — genuine proof that they generate the whole group and
  that the multiplier systems are cocycles is the caller's responsibility.
  Fails with `ModularError::InequivalentTransformationGroups` if a
  generator's matrix isn't even valid in the other group, or if the
  multipliers disagree at some sample point.

### `EquivalentTransportedMF<const TWICE_WEIGHT: usize, R, M, TG>`

Wraps an `underlying: M` together with a `CoerceTransformation<R,
M::TransformationGroup, TG>`, and implements `ModularForm<TWICE_WEIGHT, R>`
with `TransformationGroup = TG` by delegating `extract_coeffs`/`evaluate_at`
straight to `underlying` — e.g. transporting `EisensteinE4::TransformationGroup`-typed
forms into ones nameably typed as `Sl2Z`, so they can sit in the same
`SumModularForm` as forms that were always `Sl2Z`-typed (see the `E4^3 - E6^2
= 1728*Delta` test).

## Errors

### `ModularError`

`NonIntegerEntry` (a matrix entry isn't an integer), `NotInGroup` (an integer
matrix fails a group's membership condition — determinant, a congruence
condition, or whatever else defines the group), `NotEnoughConstraints` /
`SingularSystem` (from `SumModularForm::new_from_some_coeffs`),
`NonFiniteCoefficient` (a solved coefficient came out `NaN`/infinite),
`Unavailable` (no implementation computes this value, e.g. every
`evaluate_at` in this module), and `InequivalentTransformationGroups` (from
`CoerceTransformation::validate`).

## Dependencies

- [`nalgebra`](https://crates.io/crates/nalgebra) — `Complex<f64>` as the
  concrete `R`, and `SMatrix`/matrix inversion for
  `SumModularForm::new_from_some_coeffs`.
- [`num`](https://crates.io/crates/num) — `Zero`, `ToPrimitive`, and
  `Ratio<i128>` (used inside `EtaTransformationGroup::multiplier_system`'s
  Dedekind-sum computation).
