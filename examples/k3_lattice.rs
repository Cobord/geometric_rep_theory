use geometric_rep_theory::lattice::{
    DirectSumLattice, HyperbolicPlane, Lattice, NegatedLattice, RootLatticeE8, ShortVectorError,
};

type U2 = DirectSumLattice<HyperbolicPlane, HyperbolicPlane, 2, 2, 4>;
type U3 = DirectSumLattice<U2, HyperbolicPlane, 4, 2, 6>;
type U3PlusE8Neg = DirectSumLattice<U3, NegatedLattice<RootLatticeE8>, 6, 8, 14>;

/// The K3 lattice `H^2(K3, Z) ≅ U ⊕ U ⊕ U ⊕ E8(-1) ⊕ E8(-1)`: the unique
/// even unimodular lattice of signature `(3, 19)`, rank 22. It is the
/// intersection form on the second cohomology of any complex K3 surface.
type K3Lattice = DirectSumLattice<U3PlusE8Neg, NegatedLattice<RootLatticeE8>, 14, 8, 22>;

fn main() {
    println!("K3 lattice = U ⊕ U ⊕ U ⊕ E8(-1) ⊕ E8(-1)");
    println!("  is_integral:  {}", K3Lattice::is_integral());
    println!("  is_even:      {}", K3Lattice::is_even());
    println!("  is_self_dual: {}", K3Lattice::is_self_dual());
    println!("  signature:    {:?}", K3Lattice::signature());
    println!("  discriminant: {}", K3Lattice::discriminant());
    println!("  short_vectors: {:?}", K3Lattice::short_vectors());

    assert!(K3Lattice::is_integral());
    assert!(K3Lattice::is_even());
    assert!(K3Lattice::is_self_dual());
    assert_eq!(K3Lattice::signature(), (3, 19, 0));
    assert_eq!(
        K3Lattice::discriminant(),
        num::rational::Ratio::from_integer(1)
    );
    // Indefinite (signature (3, 19, 0), not (22, 0, 0)), so there's no
    // finite set of minimal-norm vectors: e.g. any isotropic vector can be
    // scaled by an arbitrary nonzero integer while staying at norm 0.
    assert_eq!(
        K3Lattice::short_vectors(),
        Err(ShortVectorError::NotPositiveDefinite)
    );
}
