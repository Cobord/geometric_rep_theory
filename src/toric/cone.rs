use itertools::Itertools;
use nalgebra::{Complex, ComplexField, DMatrix, DVector};
use std::collections::HashSet;

use crate::toric::{
    cell_util::CachedOnce, cone_errors::ConeError, integer_arith::primitive_vector,
};

type FacetIdces = Vec<usize>;

/// A rational polyhedral cone σ = Cone(v₁, …, vₙ) ⊆ ℝᵈ spanned by integer rays.
///
/// Internally the generators are stored as primitive (GCD-normalised), lexicographically sorted
/// `DVector<i64>` column vectors. The zero cone (the origin) is represented by an empty generator
/// list rather than a zero vector.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RationalPolyhedralCone {
    /// Primitive, lexicographically sorted ray generators of the cone.
    generators: Vec<DVector<i64>>,
    /// Dimension of the ambient lattice ℤᵈ in which the cone lives.
    pub ambient_dim: usize,
    /// Dimension of the linear span of the cone (i.e. rank of the generator matrix).
    /// Unset until first needed; call [`with_spanning_dim`](Self::with_spanning_dim) to populate it.
    spanning_dim: CachedOnce<usize>,
    /// Whether the current generator list is a minimal (irredundant) ray description.
    /// `Some(true)` guarantees no ray is in the conic hull of the others.
    no_redundant_generators_as_cone: CachedOnce<bool>,
    /// For each facet, the sorted indices into `generators` of the rays on that facet.
    /// `None` means the facet structure has not been provided or is invalidated.
    facets_idces: CachedOnce<Vec<FacetIdces>>,
}

impl RationalPolyhedralCone {
    /// Construct a cone from a list of integer ray generators.
    ///
    /// Each generator is normalised to its primitive form (divided by GCD) and zero vectors are
    /// silently dropped. The resulting generator list is sorted lexicographically so that equality
    /// checks are independent of the order the rays were supplied.
    ///
    /// # Parameters
    /// - `generating_rays` – raw integer ray vectors; all must have the same length.
    /// - `no_redundant_parameterization` – caller hint: `Some(true)` asserts the supplied rays are
    ///   already an irredundant (minimal) ray description; `None`/`Some(false)` allows the
    ///   constructor to deduplicate.
    /// - `known_spanning_dim` – pre-computed spanning dimension to avoid an SVD call later.
    /// - `known_facets` – optional facet structure: each inner `Vec<usize>` lists the generator
    ///   indices of one facet. Indices are remapped to match the sorted generator order.
    ///
    /// # Errors
    /// Returns [`ConeError::EmptyGeneratorList`] if `generating_rays` is empty, or
    /// [`ConeError::DimensionMismatch`] if the rays have inconsistent lengths or the supplied
    /// `known_spanning_dim` contradicts the actual rank.
    #[allow(clippy::missing_panics_doc)]
    pub fn new(
        generating_rays: Vec<Vec<i64>>,
        no_redundant_parameterization: Option<bool>,
        known_spanning_dim: Option<usize>,
        mut known_facets: Option<Vec<FacetIdces>>,
    ) -> Result<Self, ConeError> {
        let mut changed_rays = false;
        if generating_rays.is_empty() {
            return Err(ConeError::EmptyGeneratorList);
        }

        let dim = generating_rays[0].len();
        for g in &generating_rays {
            if g.len() != dim {
                return Err(ConeError::DimensionMismatch {
                    expected: dim,
                    found: g.len(),
                });
            }
        }

        // Convert to primitive canonical vectors
        let count_rays = generating_rays.len();
        let mut primitive_gens: Vec<(usize, DVector<i64>)> = generating_rays
            .into_iter()
            .filter_map(|v| primitive_vector(&v, false))
            .enumerate()
            .collect();
        changed_rays = changed_rays || primitive_gens.len() != count_rays;

        let already_sorted = primitive_gens
            .windows(2)
            .all(|w| w[0].1.as_slice() <= w[1].1.as_slice());

        // Sort lexicographically
        primitive_gens.sort_by(|(_, a), (_, b)| a.as_slice().cmp(b.as_slice()));

        let (permutation, mut primitive_gens): (Vec<usize>, Vec<DVector<i64>>) =
            primitive_gens.into_iter().unzip();

        if !already_sorted && let Some(facets) = &mut known_facets {
            for facet in facets {
                for idx in facet.iter_mut() {
                    *idx = permutation
                        .iter()
                        .position(|&x| x == *idx)
                        .expect("idx came from the permutation so it must be present");
                }
                facet.sort_unstable();
            }
        }

        if no_redundant_parameterization == Some(false) || no_redundant_parameterization.is_none() {
            let count_rays = primitive_gens.len();
            primitive_gens.dedup();
            changed_rays = changed_rays || primitive_gens.len() != count_rays;
        }

        let (known_spanning_dim, no_redundant_parameterization) = if primitive_gens.len() == 1 {
            if let Some(z) = known_spanning_dim
                && z != 1
            {
                return Err(ConeError::DimensionMismatch {
                    expected: 1,
                    found: z,
                });
            }
            (Some(1), Some(true))
        } else {
            (known_spanning_dim, no_redundant_parameterization)
        };
        assert!(!changed_rays || known_facets.is_none());

        if !changed_rays && let Some(z) = &mut known_facets {
            z.sort();
            for facet in z.iter_mut() {
                facet.sort_unstable();
                facet.dedup();
            }
        }

        let to_return = RationalPolyhedralCone {
            generators: primitive_gens,
            ambient_dim: dim,
            spanning_dim: CachedOnce::from(known_spanning_dim),
            no_redundant_generators_as_cone: CachedOnce::from(no_redundant_parameterization),
            facets_idces: CachedOnce::from(if changed_rays { None } else { known_facets }),
        };
        Ok(to_return)
    }

    /// Iterate over the primitive ray generators in lexicographic order.
    pub fn iter_generators(&self) -> impl Iterator<Item = &DVector<i64>> {
        self.generators.iter()
    }

    /// Borrow the full slice of primitive ray generators.
    #[must_use = "Do something with the generators of the cone"]
    pub fn view_generators(&self) -> &[DVector<i64>] {
        &self.generators
    }

    /// Return the number of rays (extremal generators) of the cone.
    ///
    /// # Errors
    ///
    /// Only succeeds when the cone is known to be minimally presented
    /// otherwise returns [`ConeError::NotMinimallyPresented`].
    pub fn num_rays(&self) -> Result<usize, ConeError> {
        if self
            .no_redundant_generators_as_cone
            .get()
            .is_some_and(|x| *x)
        {
            Ok(self.generators.len())
        } else {
            Err(ConeError::NotMinimallyPresented)
        }
    }

    /// Compute and cache the spanning dimension (rank of the generator matrix) if not yet known.
    ///
    /// Uses a floating-point SVD via [`DMatrix::rank`] with tolerance `1e-9`. Returns `self`
    /// so it can be chained.
    #[allow(dead_code)]
    pub(crate) fn with_spanning_dim(&self) -> &Self {
        if self.spanning_dim.get().is_some() {
            return self;
        }
        #[allow(clippy::cast_precision_loss)]
        let gens_f64 = DMatrix::<f64>::from_columns(
            &self
                .generators
                .iter()
                .map(|gv| gv.map(|x| x as f64))
                .collect::<Vec<_>>(),
        );
        let _ = self.spanning_dim.set(gens_f64.rank(1e-9));
        self
    }

    pub fn view_spanning_dim(&self) -> Option<usize> {
        self.spanning_dim.get().copied()
    }

    /// The zero cone `{0}` inside an ambient space of dimension `dim`.
    ///
    /// This is the unique cone of spanning dimension 0 and serves as the identity element
    /// for intersections and as the minimal face of every cone.
    #[must_use = "Created a zero cone in some ambient space, did you mean to use it?"]
    pub fn zero_cone(dim: usize) -> Self {
        RationalPolyhedralCone {
            generators: vec![],
            ambient_dim: dim,
            spanning_dim: CachedOnce::from(Some(0_usize)),
            no_redundant_generators_as_cone: CachedOnce::from(Some(true)),
            facets_idces: CachedOnce::from(Some(vec![])),
        }
    }

    /// Enumerate candidate faces by iterating over all subsets of generators.
    ///
    /// If `facets_idces` is set, only subsets whose sorted index set appears as a facet are
    /// included. The zero cone is always appended at the end as the empty face.
    ///
    /// This is a combinatorial enumeration and does **not** verify that each returned cone is
    /// actually a face of `self`.
    #[allow(dead_code)]
    fn pre_facets(&self) -> impl Iterator<Item = RationalPolyhedralCone> {
        let gens = self.generators.clone();

        (1..=gens.len())
            .flat_map(move |r| {
                gens.clone()
                    .into_iter()
                    .enumerate()
                    .combinations(r)
                    .filter_map(move |subset| {
                        let (idces, subset): (Vec<usize>, Vec<DVector<i64>>) =
                            subset.into_iter().unzip();
                        if let Some(facets) = self.facets_idces.get()
                            && !facets.contains(&idces)
                        {
                            return None;
                        }
                        Some(
                            RationalPolyhedralCone::new(
                                subset.into_iter().map(|v| v.as_slice().to_vec()).collect(),
                                None,
                                None,
                                None,
                            )
                            .expect("Face cone of an already valid cone is valid"),
                        )
                    })
            })
            .chain(std::iter::once(RationalPolyhedralCone::zero_cone(
                self.ambient_dim,
            )))
    }

    /// Check whether `other` is a face of `self`.
    ///
    /// The zero cone (empty generator list) is always a face. For non-zero `other`, every
    /// generator of `other` must be contained in `self` (via [`contains_vector`](Self::contains_vector)).
    ///
    /// Note: this is a necessary condition but not sufficient in general; a proper face must also
    /// lie on a supporting hyperplane. Use with care for non-simplicial cones.
    ///
    /// # Panics
    ///
    /// Both cones must share the same ambient dimension.
    #[must_use = "Did you mean to use the result of the face check?"]
    pub fn is_face(&self, other: &RationalPolyhedralCone) -> bool {
        assert_eq!(
            self.ambient_dim, other.ambient_dim,
            "is_face requires both cones to live in the same ambient space"
        );
        if other.generators.is_empty() {
            return true;
        }
        other.generators.iter().all(|v| self.contains_vector(v))
    }

    /// Test whether the integer vector `v` lies inside the cone.
    ///
    /// Converts `v` to `f64` and delegates to [`contains_vectorf`](Self::contains_vectorf).
    #[allow(clippy::cast_precision_loss)]
    #[must_use = "Did you mean to use the result of the containment check?"]
    pub fn contains_vector(&self, v: &DVector<i64>) -> bool {
        let v_f64 = DVector::<f64>::from_iterator(self.ambient_dim, v.iter().map(|x| *x as f64));

        self.contains_vectorf(&v_f64)
    }

    /// Test whether the floating-point vector `v_f64` lies inside the cone.
    ///
    /// Solves the non-negative least-squares problem by computing the normal-equations solution
    /// `x = (GᵀG)⁻¹ Gᵀ v` and checking `x ≥ -1e-9` component-wise. Returns `false` if `GᵀG`
    /// is singular (i.e. the generators are linearly dependent in a degenerate way).
    #[must_use = "Did you mean to use the result of the containment check?"]
    pub fn contains_vectorf(&self, v_f64: &DVector<f64>) -> bool {
        #[allow(clippy::cast_precision_loss)]
        let gens_f64 = DMatrix::<f64>::from_columns(
            &self
                .generators
                .iter()
                .map(|gv| gv.map(|x| x as f64))
                .collect::<Vec<_>>(),
        );

        let gtg = &gens_f64.transpose() * &gens_f64;
        if let Some(gtg_inv) = gtg.clone().try_inverse() {
            let x = gtg_inv * gens_f64.transpose() * v_f64;
            x.iter().all(|xi| *xi >= -1e-9)
        } else {
            false
        }
    }

    /// Compute a combinatorial under-approximation of the intersection σ ∩ τ.
    ///
    /// Returns the cone generated by the rays that appear (as identical primitive vectors) in
    /// **both** `self` and `other`. This equals the true intersection when the two cones meet
    /// along a common face, but may be strictly smaller in general.
    ///
    /// # Errors
    ///
    /// Both cones must be minimally presented (`no_redundant_generators_as_cone == Some(true)`)
    /// and have the same ambient dimension; otherwise an error is returned.
    pub fn combinatorial_intersection(
        &self,
        other: &RationalPolyhedralCone,
    ) -> Result<RationalPolyhedralCone, ConeError> {
        if self.ambient_dim != other.ambient_dim {
            return Err(ConeError::DimensionMismatch {
                expected: self.ambient_dim,
                found: other.ambient_dim,
            });
        }
        if self
            .no_redundant_generators_as_cone
            .get()
            .is_none_or(|z| !z)
            || other
                .no_redundant_generators_as_cone
                .get()
                .is_none_or(|z| !z)
        {
            return Err(ConeError::NotMinimallyPresented);
        }

        let mut combined = self.generators.clone();
        combined.extend(other.generators.clone());

        let mut seen = HashSet::new();
        for from_self in &self.generators {
            seen.insert(from_self.as_slice().to_vec());
        }
        let intersection: Vec<Vec<i64>> = other
            .generators
            .iter()
            .filter(|v| seen.contains(v.as_slice()))
            .map(|v| v.as_slice().to_vec())
            .collect();
        if intersection.is_empty() {
            return Ok(Self::zero_cone(self.ambient_dim));
        }
        let no_redundant_parameterization = match (
            self.no_redundant_generators_as_cone.get(),
            other.no_redundant_generators_as_cone.get(),
        ) {
            (None, None) => None,
            (Some(a), None) | (None, Some(a)) => Some(*a),
            (Some(a), Some(b)) => Some(*a || *b),
        };
        RationalPolyhedralCone::new(intersection, no_redundant_parameterization, None, None)
    }

    /// Check whether the intersection of `self` and `other` is a face of both cones.
    ///
    /// This is the fan axiom: a collection of cones forms a fan iff every pairwise intersection
    /// is a face of each cone.
    ///
    /// **Not yet implemented** — currently panics with `todo!`.
    ///
    /// # Errors
    ///
    /// TODO
    pub fn intersection_is_face_of_both(
        &self,
        _other: &RationalPolyhedralCone,
    ) -> Result<bool, ConeError> {
        todo!("Honest intersection")
    }

    /// Cartesian product of cones: σ × τ
    #[must_use = "Constructed the product cone at some expense, either use it or avoid calling this method"]
    pub fn product_with(&self, _other: &Self) -> RationalPolyhedralCone {
        todo!("Product of cones not yet implemented")
    }

    /// Return the dimension of the associated toric variety, if it can be determined.
    ///
    /// Succeeds only when the cone is minimally presented, full-dimensional (spanning dimension
    /// equals ambient dimension), and simplicial with exactly `ambient_dim` rays — i.e. a smooth
    /// affine toric chart. Returns `Err(())` otherwise.
    ///
    /// # Errors
    ///
    /// If the cone is not minimally-presented, full-dimensional in the ambient space
    /// and simplicial.
    #[allow(clippy::result_unit_err)]
    pub fn variety_dim(&self) -> Result<usize, ()> {
        if self
            .no_redundant_generators_as_cone
            .get()
            .is_some_and(|z| *z)
            && self
                .spanning_dim
                .get()
                .is_some_and(|z| *z == self.ambient_dim)
            && self.generators.len() == self.ambient_dim
        {
            Ok(self.ambient_dim)
        } else {
            Err(())
        }
    }

    /// Check whether the cone is simplicial (number of rays equals spanning dimension).
    ///
    /// Computes and caches `spanning_dim` via [`with_spanning_dim`](Self::with_spanning_dim) if
    /// it is not already known.
    pub fn is_simplicial(&self) -> bool {
        // Simplicial requires #generators == dim(span(σ)). Since dim(span(σ)) ≤ ambient_dim,
        // having more generators than ambient_dim already exceeds any possible spanning dimension.
        if self.view_generators().len() > self.ambient_dim {
            return false;
        }
        // If spanning_dim is known we can answer immediately: simplicial iff #rays == rank.
        if let Some(&z) = self.spanning_dim.get() {
            return self.view_generators().len() == z;
        }
        // spanning_dim not yet computed — fill it in, then recurse (the second call will hit
        // the branch above).
        self.with_spanning_dim();
        self.is_simplicial()
    }

    /// Check whether the cone is smooth (regular), i.e. its rays form part of a ℤ-basis of the
    /// ambient lattice.
    ///
    /// For a full-dimensional simplicial cone with exactly `ambient_dim` rays this reduces to
    /// checking `|det(G)| = 1` (up to tolerance `1e-3`). The non-square case (fewer rays than
    /// ambient dimension) is not yet implemented and will panic.
    #[allow(clippy::missing_panics_doc)]
    pub fn is_smooth(&self) -> bool {
        let num_generators = self.view_generators().len();
        if self.ambient_dim < num_generators {
            return false;
        }
        if self.ambient_dim == self.view_generators().len() {
            self.no_redundant_generators_as_cone.get_or_init(|| true);
            self.spanning_dim.get_or_init(|| self.ambient_dim);
            let mut matrix =
                DMatrix::<nalgebra::Complex<f64>>::zeros(num_generators, self.ambient_dim);
            for (idx, row) in self.view_generators().iter().enumerate() {
                for (jdx, entry) in row.iter().enumerate() {
                    matrix[(idx, jdx)] =
                        Complex::new(f64::from(i32::try_from(*entry).expect("Fits")), 0.0);
                }
            }
            let mat_det = matrix.determinant();
            let mat_is_gln_z = (mat_det.abs() - 1.0).abs();
            return mat_is_gln_z <= 1e-3;
        }
        let _matrix = DMatrix::from_columns(self.view_generators());
        todo!(
            "Smooth, we have the matrix of all the generators we need to know if we can add more columns to fill it in to a GL(n,Z) matrix"
        );
    }

    /// Borrow the facet index structure, if available.
    ///
    /// Returns `None` when the facet data was not supplied at construction time or was
    /// invalidated by generator normalisation. Each inner slice contains the sorted generator
    /// indices of one facet.
    #[must_use = "If you have the facets, use the reference in the Some. If you don't, then match on None and fail gracefully."]
    pub fn view_facets_idces(&self) -> Option<&Vec<FacetIdces>> {
        self.facets_idces.get()
    }
}
