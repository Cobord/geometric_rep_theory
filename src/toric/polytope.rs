use std::collections::HashMap;

use itertools::Itertools;
use nalgebra::{DMatrix, DVector};
use qhull::{Qh, QhError, Vertex};

use crate::toric::{cone::RationalPolyhedralCone, cone_errors::ToricFanError, fan::ToricFan};

/// Errors for polytope operations
#[derive(Debug)]
pub enum PolytopeError {
    EmptyVertexList,
    DimensionMismatch {
        expected: usize,
        found: usize,
    },
    ZeroVertex,
    ConvexHullError(QhError<'static>),
    /// The polar dual requires the origin to be in the interior of the polytope.
    OriginNotContained,
    /// A facet's primitive outward normal is not divisible by its support value,
    /// so the dual vertex is not a lattice point.
    DualNotLattice,
}

/// A convex lattice polytope in ℝᴺ, backed by a qhull convex-hull computation.
///
/// # qhull dimensionality restriction
///
/// qhull requires the input points to be **full-dimensional** in their ambient space: the affine
/// hull of the vertices must be all of ℝᴺ (i.e. the polytope must have non-empty interior as a
/// subset of ℝᴺ). A polytope that lives in a proper affine subspace — for example, three
/// collinear points in ℝ², or a planar polygon supplied as 3-D points — will cause qhull to
/// error or produce incorrect output.
///
/// If your polytope naturally lives in a lower-dimensional space, embed it there first (using the
/// smallest `N` that makes it full-dimensional) and use [`expand_ambient_dimesion`] combined with
/// [`post_compose`] to map it into the larger ambient space afterwards.
pub struct ConvexPolytope<const N: usize> {
    qh: Qh<'static>,
    extra_zero_coords: usize,
    do_this_affine: Option<(DMatrix<i64>, DVector<i64>)>,
}

impl<const N: usize> ConvexPolytope<N> {
    /// Create a new convex polytope from integer vertices.
    ///
    /// The vertices must affinely span all of ℝᴺ (see the type-level note on the qhull
    /// dimensionality restriction). Passing in points that lie in a proper affine subspace will
    /// cause qhull to return an error.
    ///
    /// # Errors
    ///
    /// Returns `Err(PolytopeError::EmptyVertexList)` if `vertices` is empty, or
    /// `Err(PolytopeError::ConvexHullError(...))` if qhull fails (e.g. the points do not
    /// affinely span ℝᴺ).
    pub fn new(vertices: Vec<[i64; N]>) -> Result<Self, PolytopeError> {
        if vertices.is_empty() {
            return Err(PolytopeError::EmptyVertexList);
        }

        #[allow(clippy::cast_precision_loss)]
        let qh = Qh::builder()
            .compute(true)
            .build_from_iter(
                vertices
                    .into_iter()
                    .map(|v| v.into_iter().map(|x| x as f64)),
            )
            .map_err(PolytopeError::ConvexHullError)?;

        Ok(ConvexPolytope {
            qh,
            extra_zero_coords: 0,
            do_this_affine: None,
        })
    }

    /// The dimension of the ambient space ℝᴺ in which the polytope lives.
    #[must_use = "Getting ambient dimension is cheap, either use it or avoid calling this method"]
    pub const fn ambient_dim(&self) -> usize {
        N
    }

    /// Bake the stored affine map into the qhull representation.
    ///
    /// When `extra_zero_coords == 0` and a `do_this_affine` map `x ↦ A·q + b` is present,
    /// rebuilds qhull from the transformed vertices and clears the map. After this call the
    /// polytope is in its natural coordinate system with no pending transformation. `A` is
    /// invertible because callers supply it as an element of the affine group `Aff(ℝᴺ)`.
    ///
    /// Does nothing when `extra_zero_coords > 0`, because qhull operates in a space of dimension
    /// smaller than `N` and the full `N×N` matrix `A` cannot be applied directly to those points.
    #[allow(clippy::missing_panics_doc)]
    pub fn normalize(&mut self) {
        if self.extra_zero_coords > 0 {
            return;
        }
        let Some((a, b)) = self.do_this_affine.take() else {
            return;
        };

        // Collect the qhull vertices, apply A·q + b, and rebuild.
        let new_vertices: Vec<Vec<f64>> = self
            .qh
            .all_vertices()
            .filter_map(|v| v.point())
            .map(|pt| {
                #[allow(clippy::cast_possible_truncation)]
                let q = DVector::from_iterator(pt.len(), pt.iter().map(|&x| x as i64));
                let t = &a * q + &b;
                #[allow(clippy::cast_precision_loss)]
                t.iter().map(|&x| x as f64).collect()
            })
            .collect();

        // A is invertible, so the image is still full-dimensional.
        self.qh = Qh::builder()
            .compute(true)
            .build_from_iter(new_vertices)
            .expect("invertible A preserves full-dimensionality");
    }

    /// Embed the polytope into a larger ambient space ℝᴹ by appending `M - N` zero coordinates.
    ///
    /// The convex hull computed by qhull is unchanged; the extra coordinates are recorded and
    /// appended when vertices are read back out. If a `post_compose` affine map is already
    /// stored, it is extended to act as the identity on the new coordinates.
    ///
    /// # Panics
    ///
    /// `M` cannot be smaller than `N`.
    /// We can expand the ambient space not shrink it.
    pub fn expand_ambient_dimesion<const M: usize>(self) -> ConvexPolytope<M> {
        assert!(M >= N, "New ambient dimension must be at least the old one");
        if self.do_this_affine.is_none() {
            return ConvexPolytope {
                qh: self.qh,
                extra_zero_coords: M - N + self.extra_zero_coords,
                do_this_affine: None,
            };
        }
        if let Some((old_a_matrix, old_b_vector)) = self.do_this_affine {
            let mut new_a_matrix =
                old_a_matrix
                    .insert_columns(N, M - N, 0)
                    .insert_rows(N, M - N, 0);
            for i in N..M {
                new_a_matrix[(i, i)] = 1;
            }
            let new_b_vector = old_b_vector.insert_rows(N, M - N, 0);
            ConvexPolytope {
                qh: self.qh,
                extra_zero_coords: M - N + self.extra_zero_coords,
                do_this_affine: Some((new_a_matrix, new_b_vector)),
            }
        } else {
            ConvexPolytope {
                qh: self.qh,
                extra_zero_coords: M - N + self.extra_zero_coords,
                do_this_affine: None,
            }
        }
    }

    /// Compose an affine map `x ↦ A·x + v` onto the output coordinates.
    ///
    /// After this call, every vertex and normal read back from the polytope is transformed by
    /// `(A, v)`. If a map was already stored, the new map is composed on the left: the net
    /// transformation becomes `x ↦ A_now·(A_before·x + v_before) + v_now`. Both `A` and `v`
    /// must have shape compatible with `N`.
    ///
    /// # Errors
    /// Returns [`PolytopeError::DimensionMismatch`] if `A` is not `N×N` or `v` is not length `N`.
    pub fn post_compose(
        #[rustfmt::skip] &mut self,
        #[rustfmt::skip] (a_now, v_now): (DMatrix<i64>, DVector<i64>),
    ) -> Result<(), PolytopeError> {
        let (r, c) = a_now.shape();
        if r != self.ambient_dim() {
            return Err(PolytopeError::DimensionMismatch {
                expected: self.ambient_dim(),
                found: r,
            });
        }
        if c != self.ambient_dim() {
            return Err(PolytopeError::DimensionMismatch {
                expected: self.ambient_dim(),
                found: c,
            });
        }
        let (r, c) = v_now.shape();
        if r != self.ambient_dim() {
            return Err(PolytopeError::DimensionMismatch {
                expected: self.ambient_dim(),
                found: r,
            });
        }
        if c != 1 {
            return Err(PolytopeError::DimensionMismatch {
                expected: 1,
                found: c,
            });
        }
        if let Some((a_before, v_before)) = &self.do_this_affine {
            let a_after = &a_now * a_before;
            let v_after = a_now * v_before + v_now;
            self.do_this_affine = Some((a_after, v_after));
        } else {
            self.do_this_affine = Some((a_now, v_now));
        }
        Ok(())
    }

    fn vertices_and_facets(&self) -> (HashMap<usize, Vertex<'_>>, Vec<Vec<usize>>) {
        let mut all_vertices_ordered = HashMap::with_capacity(self.qh.num_vertices());
        let facets = self
            .qh
            .all_facets()
            .filter_map(|f| {
                if let Some(verts) = f.vertices() {
                    if verts.iter().any(|v| v.point().is_none()) {
                        return None;
                    }
                    let mut face_idx: Vec<usize> = vec![];
                    for v in verts.iter() {
                        let v_idx = v.index(&self.qh)?;
                        all_vertices_ordered.insert(v_idx, v);
                        face_idx.push(v_idx);
                    }
                    Some(face_idx)
                } else {
                    None
                }
            })
            .collect::<Vec<Vec<usize>>>();
        (all_vertices_ordered, facets)
    }

    /// Iterate over the vertices of the polytope in their ambient ℝᴺ coordinates.
    ///
    /// Each vertex is returned as a `Vec<i64>` of length `N` after applying any stored affine
    /// map and appending extra zero coordinates.
    pub fn all_vertices(&self) -> impl Iterator<Item = Vec<i64>> + '_ {
        self.qh.all_vertices().filter_map(|v| {
            if v.point().is_none() {
                None
            } else {
                Some(self.vertex_to_inhomogeneous(v))
            }
        })
    }

    fn vertex_to_homogeneous(&self, v: Vertex<'_>) -> Vec<i64> {
        let mut without_homogeneous = self.vertex_to_inhomogeneous(v);
        without_homogeneous.push(1);
        without_homogeneous
    }

    fn vertex_to_inhomogeneous(&self, v: Vertex<'_>) -> Vec<i64> {
        let mut ray = vec![];
        #[allow(clippy::cast_possible_truncation)]
        ray.extend(v.point().expect("Point is valid").iter().map(|&x| x as i64));
        ray.extend(vec![0; self.extra_zero_coords]);
        if let Some(do_this_affine) = &self.do_this_affine {
            let ray = DVector::from_vec(ray);
            let ray = &do_this_affine.0 * ray + &do_this_affine.1;
            ray.into_iter().copied().collect()
        } else {
            ray
        }
    }

    fn vec_to_inhomogeneous(&self, mut vec: DVector<i64>) -> DVector<i64> {
        if self.extra_zero_coords > 0 {
            vec.extend(vec![0_i64; self.extra_zero_coords]);
        }
        if let Some(do_this_affine) = &self.do_this_affine {
            &do_this_affine.0 * vec + &do_this_affine.1
        } else {
            vec
        }
    }

    /// Compute the polar dual `P° = {y ∈ ℝᴺ : x·y ≤ 1 ∀ x ∈ P}`.
    ///
    /// Calls [`normalize`] first to bake any pending affine map into the qhull data, then checks
    /// that the origin is contained in `P` (a necessary condition for `P°` to be bounded).
    ///
    /// The vertices of `P°` are in bijection with the facets of `P`: for a facet with primitive
    /// outward integer normal `m` and support value `h = m·v` (for any on-facet vertex `v`), the
    /// corresponding dual vertex is `m / h`.  Returns [`PolytopeError::DualNotLattice`] if any
    /// component of `m` is not divisible by `h` (the polytope is not reflexive over ℤ).
    ///
    /// # Errors
    /// - [`PolytopeError::OriginNotContained`] if the origin is not in `P`.
    /// - [`PolytopeError::DualNotLattice`] if a dual vertex is not an integer lattice point.
    /// - [`PolytopeError::ConvexHullError`] if qhull fails on the dual vertices.
    pub fn polar_dual(mut self) -> Result<ConvexPolytope<N>, PolytopeError> {
        self.normalize();
        if !self.contains_origin() {
            return Err(PolytopeError::OriginNotContained);
        }

        let mut dual_vertices: Vec<[i64; N]> = Vec::new();

        for facet in self.qh.facets() {
            let Some(normal) = facet.normal() else {
                continue;
            };
            let Some(vertices) = facet.vertices() else {
                continue;
            };
            let Some(any_v) = vertices.iter().next() else {
                continue;
            };
            let Some(pt) = any_v.point() else {
                continue;
            };

            // Scale unit normal to a primitive integer vector m.
            let n_dvec = DVector::from_iterator(normal.len(), normal.iter().copied());
            let (_, m) = to_primitive(n_dvec);

            // Support value h = m · v (positive since origin is in the interior).
            #[allow(clippy::cast_possible_truncation)]
            let pt_int = DVector::from_iterator(pt.len(), pt.iter().map(|&x| x as i64));
            let h = m.dot(&pt_int);

            // Dual vertex = m / h; each component must divide evenly.
            if m.iter().any(|&x| x % h != 0) {
                return Err(PolytopeError::DualNotLattice);
            }
            let w: Vec<i64> = m.iter().map(|&x| x / h).collect();
            let mut arr = [0i64; N];
            arr.copy_from_slice(&w);
            dual_vertices.push(arr);
        }

        ConvexPolytope::new(dual_vertices)
    }

    /// Embed the polytope as the "height-1 slice" of a rational polyhedral cone.
    ///
    /// Each vertex `v ∈ ℝᴺ` is mapped to the ray `(v, 1) ∈ ℝᴺ⁺¹`, producing the cone
    /// `σ = Cone{(v, 1) : v vertex of P}`. The facet structure of `P` lifts directly to
    /// the facet structure of `σ` (a facet of `σ` corresponds to a facet of `P` plus the
    /// extra coordinate). The resulting cone is minimally presented since qhull returns only
    /// extreme vertices.
    #[allow(clippy::missing_panics_doc)]
    pub fn homogeneous_lift(self) -> RationalPolyhedralCone {
        let (all_vertices_ordered, facets) = self.vertices_and_facets();

        // vertices sorted by qhull vertex ID
        // as the first element of the tuple
        let sorted_vertices: Vec<(usize, Vertex<'_>)> = all_vertices_ordered
            .into_iter()
            .sorted_by(|(a, _), (b, _)| a.cmp(b))
            .collect();

        // Build a map from qhull vertex ID to 0-based position in `rays`.
        let id_to_pos: HashMap<usize, usize> = sorted_vertices
            .iter()
            .enumerate()
            .map(|(pos, (id, _))| (*id, pos))
            .collect();

        let rays: Vec<Vec<i64>> = sorted_vertices
            .into_iter()
            .map(|(_, v)| self.vertex_to_homogeneous(v))
            .collect();

        let remapped_facets: Vec<Vec<usize>> = facets
            .into_iter()
            .map(|facet| facet.into_iter().map(|id| id_to_pos[&id]).collect())
            .collect();

        let to_return = RationalPolyhedralCone::new(rays, Some(true), None, Some(remapped_facets))
            .expect("Vertices guaranteed to be nonzero and same dim");
        to_return.with_spanning_dim();
        to_return
    }

    /// Returns `true` if the origin `0 ∈ ℝᴺ` is contained in the polytope.
    ///
    /// Returns `false` immediately when `extra_zero_coords > 0`, since the polytope is then
    /// confined to a proper affine subspace of ℝᴺ and has empty (relative) interior there.
    ///
    /// When a `do_this_affine` map `x ↦ A·q + b` is stored, the check is performed by inverting
    /// the map: the origin in ℝᴺ corresponds to `q = -A⁻¹·b` in qhull-space.  If A is singular
    /// (so no pre-image exists) the origin cannot be in the image and `false` is returned.
    ///
    /// Containment is tested via the facet half-space inequalities: for every facet with outward
    /// normal `n` and any facet vertex `v`, we require `n · (q - v) ≤ 0`.
    #[must_use]
    pub fn contains_origin(&self) -> bool {
        if self.extra_zero_coords > 0 {
            return false;
        }

        // Compute the pre-image of the origin in qhull-space.
        let query: Vec<f64> = if let Some((a, b)) = &self.do_this_affine {
            #[allow(clippy::cast_precision_loss)]
            let a_f64 = a.map(|x| x as f64);
            #[allow(clippy::cast_precision_loss)]
            let b_f64 = b.map(|x| x as f64);
            let Some(a_inv) = a_f64.try_inverse() else {
                return false;
            };
            (a_inv * (-b_f64)).iter().copied().collect()
        } else {
            vec![0.0_f64; N]
        };

        // Half-space test: for every facet, the query must lie on the inner side.
        // Using outward normal n and any on-facet vertex v: n · (q − v) ≤ 0.
        for facet in self.qh.facets() {
            let Some(normal) = facet.normal() else {
                continue;
            };
            let Some(vertices) = facet.vertices() else {
                continue;
            };
            let Some(any_vertex) = vertices.iter().next() else {
                continue;
            };
            let Some(point) = any_vertex.point() else {
                continue;
            };
            let dot: f64 = normal
                .iter()
                .zip(query.iter().zip(point.iter()))
                .map(|(n, (q, v))| n * (q - v))
                .sum();
            if dot > 1e-9 {
                return false;
            }
        }
        true
    }

    /// Construct the normal fan of the polytope as a [`ToricFan`].
    ///
    /// For each vertex `v` of `P`, the corresponding cone in the normal fan is generated by
    /// the outward-pointing facet normals of all facets of `P` that contain `v`. These normals
    /// are taken from qhull, scaled to primitive integer vectors via [`to_primitive`], and then
    /// mapped through the stored affine transformation (if any) before being added to the fan.
    ///
    /// The resulting fan lives in the ambient space ℝᴺ (after applying `do_this_affine`),
    /// not in the qhull coordinate space.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any per-vertex cone construction fails, for example if a vertex has no
    /// incident good facets (yielding an empty generator list for that cone).
    pub fn toric_fan(self) -> Result<ToricFan, ToricFanError> {
        let mut to_return = ToricFan::new(N);
        for vertex in self.qh.all_vertices() {
            if vertex.point().is_none() {
                continue;
            }
            let facets_containing_me = self.qh.facets().filter(|facet| {
                if !facet.good() {
                    return false;
                }
                if let Some(vs) = facet.vertices() {
                    vs.iter().map(|v| v.id()).contains(&vertex.id())
                } else {
                    false
                }
            });
            let mut generating_rays = Vec::new();
            for generating_pre_ray in facets_containing_me.filter_map(|f| f.normal()) {
                let dvec = DVector::from_vec(generating_pre_ray.to_vec());
                let pushing_now = self.vec_to_inhomogeneous(to_primitive(dvec).1);
                generating_rays.push(pushing_now.into_iter().copied().collect());
            }
            let cur_cone = RationalPolyhedralCone::new(generating_rays, None, None, None)
                .map_err(ToricFanError::ConeError)?;
            to_return.add_cone(cur_cone, false)?;
        }
        Ok(to_return)
    }

    /// Enumerate all lattice points strictly in the interior of the polytope.
    ///
    /// Iterates over all integer points in the bounding box of the qhull vertices and retains
    /// those that lie strictly on the interior side of every facet (dot product with outward
    /// normal strictly negative).
    pub fn interior_lattice_points(&self) -> Vec<[i64; N]> {
        let qh_dim = N - self.extra_zero_coords;

        let mut mins = vec![i64::MAX; qh_dim];
        let mut maxs = vec![i64::MIN; qh_dim];
        for v in self.qh.all_vertices() {
            let Some(pt) = v.point() else { continue };
            for (i, &x) in pt.iter().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                let lo = x.floor() as i64;
                #[allow(clippy::cast_possible_truncation)]
                let hi = x.ceil() as i64;
                if lo < mins[i] {
                    mins[i] = lo;
                }
                if hi > maxs[i] {
                    maxs[i] = hi;
                }
            }
        }

        let facets: Vec<(Vec<f64>, Vec<f64>)> = self
            .qh
            .facets()
            .filter_map(|f| {
                let normal = f.normal()?.to_vec();
                let pt = f.vertices()?.iter().next()?.point()?.to_vec();
                Some((normal, pt))
            })
            .collect();

        let ranges: Vec<std::ops::RangeInclusive<i64>> =
            (0..qh_dim).map(|i| mins[i]..=maxs[i]).collect();

        #[allow(clippy::cast_precision_loss)]
        ranges
            .into_iter()
            .multi_cartesian_product()
            .filter(|candidate| {
                facets.iter().all(|(normal, v_pt)| {
                    let dot: f64 = normal
                        .iter()
                        .zip(candidate.iter().zip(v_pt.iter()))
                        .map(|(n, (&p, v))| n * (p as f64 - v))
                        .sum();
                    dot < -1e-9
                })
            })
            .map(|candidate| {
                let transformed = self.vec_to_inhomogeneous(DVector::from_vec(candidate));
                let mut arr = [0i64; N];
                arr.copy_from_slice(transformed.as_slice());
                arr
            })
            .collect()
    }

    /// Return all lattice points lying in the relative interior of some codimension-`k` face.
    ///
    /// A point qualifies when it lies weakly inside every facet half-space and is exactly
    /// on the boundary of `k` of them (within tolerance 1e-9):
    /// - `k = 0`: strictly interior to the polytope
    /// - `k = 1`: in the relative interior of some facet
    /// - `k = dim`: vertices (for simple polytopes)
    pub fn lattice_points_of_codim(&self, k: usize) -> Vec<[i64; N]> {
        let qh_dim = N - self.extra_zero_coords;
        if k > qh_dim {
            return vec![];
        }

        // Bounding box in qhull's native coordinate space
        let mut mins = vec![i64::MAX; qh_dim];
        let mut maxs = vec![i64::MIN; qh_dim];
        for v in self.qh.all_vertices() {
            let Some(pt) = v.point() else { continue };
            for (i, &x) in pt.iter().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                let lo = x.floor() as i64;
                #[allow(clippy::cast_possible_truncation)]
                let hi = x.ceil() as i64;
                if lo < mins[i] {
                    mins[i] = lo;
                }
                if hi > maxs[i] {
                    maxs[i] = hi;
                }
            }
        }

        // Facets as (outward unit normal, a point on the facet)
        let facets: Vec<(Vec<f64>, Vec<f64>)> = self
            .qh
            .facets()
            .filter_map(|f| {
                let normal = f.normal()?.to_vec();
                let pt = f.vertices()?.iter().next()?.point()?.to_vec();
                Some((normal, pt))
            })
            .collect();

        let ranges: Vec<std::ops::RangeInclusive<i64>> =
            (0..qh_dim).map(|i| mins[i]..=maxs[i]).collect();

        #[allow(clippy::cast_precision_loss)]
        ranges
            .into_iter()
            .multi_cartesian_product()
            .filter(|candidate| {
                let dots: Vec<f64> = facets
                    .iter()
                    .map(|(normal, v_pt)| {
                        normal
                            .iter()
                            .zip(candidate.iter().zip(v_pt.iter()))
                            .map(|(n, (&p, v))| n * (p as f64 - v))
                            .sum()
                    })
                    .collect();
                // Must be weakly inside every facet
                let weakly_inside = dots.iter().all(|&d| d < 1e-9);
                // Must lie on exactly k facet hyperplanes
                let on_boundary = dots.iter().filter(|&&d| d.abs() < 1e-9).count();
                weakly_inside && on_boundary == k
            })
            .map(|candidate| {
                let transformed = self.vec_to_inhomogeneous(DVector::from_vec(candidate));
                let mut arr = [0i64; N];
                arr.copy_from_slice(transformed.as_slice());
                arr
            })
            .collect()
    }
}

impl ConvexPolytope<2> {
    /// Iterate over all 16 reflexive polygons in ℝ², each paired with its name.
    ///
    /// Each polygon is constructed lazily: the constructor for element `k` is only called
    /// when the iterator reaches position `k`.
    pub fn reflexive_2d() -> impl Iterator<Item = (String, Self)> {
        let constructors: [fn() -> (String, Self); 16] = [
            Self::polygon_01,
            Self::polygon_02,
            Self::polygon_03,
            Self::polygon_04,
            Self::polygon_05,
            Self::polygon_06,
            Self::polygon_07,
            Self::polygon_08,
            Self::polygon_09,
            Self::polygon_10,
            Self::polygon_11,
            Self::polygon_12,
            Self::polygon_13,
            Self::polygon_14,
            Self::polygon_15,
            Self::polygon_16,
        ];
        constructors.into_iter().map(|f| f())
    }

    // 3-vertex polygons
    fn polygon_01() -> (String, Self) {
        (
            "C3 / Z3 times Z3 (1,0,2) (0,1,2)".to_string(),
            ConvexPolytope::new(vec![[-1, 2], [-1, -1], [2, -1]]).expect("valid"),
        )
    }
    fn polygon_02() -> (String, Self) {
        (
            "C3 / Z4 times Z2 (1,0,3) (0,1,1)".to_string(),
            ConvexPolytope::new(vec![[1, 0], [-1, -2], [-1, 2]]).expect("valid"),
        )
    }
    fn polygon_03() -> (String, Self) {
        (
            "L_1,3,1 / Z2".to_string(),
            ConvexPolytope::new(vec![[1, 0], [1, -1], [-1, -1], [-1, 2]]).expect("valid"),
        )
    }
    fn polygon_04() -> (String, Self) {
        (
            "C / Z2 times Z2 (1,0,0,1) (0,1,1,0) PdP5".to_string(),
            ConvexPolytope::new(vec![[-1, 1], [-1, -1], [1, -1], [1, 1]]).expect("valid"),
        )
    }

    fn polygon_05() -> (String, Self) {
        (
            "PdP4b".to_string(),
            ConvexPolytope::new(vec![[1, 0], [0, -1], [-1, -1], [-1, 2]]).expect("valid"),
        )
    }
    fn polygon_06() -> (String, Self) {
        (
            "PdP4a".to_string(),
            ConvexPolytope::new(vec![[1, 0], [1, -1], [-1, -1], [-1, 1], [0, 1]]).expect("valid"),
        )
    }
    fn polygon_07() -> (String, Self) {
        (
            "C3/Z6 (1,2,3) PdP_3a".to_string(),
            ConvexPolytope::new(vec![[1, 0], [-1, -1], [-1, 2]]).expect("valid"),
        )
    }
    fn polygon_08() -> (String, Self) {
        (
            "SPP/Z2 (0,1,1,1) PdP_3c".to_string(),
            ConvexPolytope::new(vec![[1, 0], [0, -1], [-1, 0], [-1, 2]]).expect("valid"),
        )
    }
    fn polygon_09() -> (String, Self) {
        (
            "PdP_3b".to_string(),
            ConvexPolytope::new(vec![[1, 0], [0, -1], [-1, -1], [-1, 1], [0, 1]]).expect("valid"),
        )
    }
    fn polygon_10() -> (String, Self) {
        (
            "dP_3".to_string(),
            ConvexPolytope::new(vec![[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]])
                .expect("valid"),
        )
    }

    fn polygon_11() -> (String, Self) {
        (
            "PdP_2".to_string(),
            ConvexPolytope::new(vec![[1, 0], [0, -1], [-1, -1], [-1, 1]]).expect("valid"),
        )
    }
    fn polygon_12() -> (String, Self) {
        (
            "dP_2".to_string(),
            ConvexPolytope::new(vec![[1, 0], [0, -1], [-1, -1], [-1, 0], [0, 1]]).expect("valid"),
        )
    }
    fn polygon_13() -> (String, Self) {
        (
            "C3 / Z_4 (1,1,2) Y^2,2".to_string(),
            ConvexPolytope::new(vec![[1, 0], [-1, -1], [-1, 1]]).expect("valid"),
        )
    }
    fn polygon_14() -> (String, Self) {
        (
            "dP_1".to_string(),
            ConvexPolytope::new(vec![[1, 0], [-1, -1], [-1, 0], [0, 1]]).expect("valid"),
        )
    }
    fn polygon_15() -> (String, Self) {
        (
            "calC / Z2 (1,1,1,1) F0".to_string(),
            ConvexPolytope::new(vec![[1, 0], [0, -1], [-1, 0], [0, 1]]).expect("valid"),
        )
    }
    fn polygon_16() -> (String, Self) {
        (
            "C3 / Z3 (1,1,1) dP_0".to_string(),
            ConvexPolytope::new(vec![[1, 0], [-1, -1], [0, 1]]).expect("valid"),
        )
    }
}

impl ConvexPolytope<1> {
    /// The unique reflexive polytope in dimension 1: the interval `[−1, 1]`.
    ///
    /// It is self-dual: the polar dual of `[l, r]` with `l < 0 < r` is `[1/r, −1/l]`,
    /// which is a lattice polytope only when `l = −1` and `r = 1`.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn reflexive_1d() -> Self {
        ConvexPolytope::new(vec![[-1], [1]]).expect("valid")
    }
}

#[allow(clippy::needless_pass_by_value)]
fn to_primitive(v: DVector<f64>) -> (i64, DVector<i64>) {
    for mul_by in 1..=60_i32 {
        let v_mul_by = &v * (f64::from(mul_by).sqrt());
        if v_mul_by
            .iter()
            .all(|entry| (entry.round() - entry).abs() < 1e-6)
        {
            let v_rounded: Vec<i64> = unsafe {
                v_mul_by
                    .iter()
                    .map(|entry| entry.round().to_int_unchecked())
                    .collect()
            };
            return (i64::from(mul_by), DVector::from_vec(v_rounded));
        }
    }
    panic!("Couldn't make it integral {v}")
}

#[cfg(test)]
mod test {
    #[test]
    fn test_polytope_to_cone() {
        use crate::toric::cone::RationalPolyhedralCone;
        use crate::toric::polytope::ConvexPolytope;

        let poly = ConvexPolytope::new(vec![[0, 0], [1, 0], [0, 1]]).expect("valid polytope");
        let cone: RationalPolyhedralCone = poly.homogeneous_lift();

        assert_eq!(cone.ambient_dim, 3);
        assert_eq!(cone.view_generators().len(), 3);
        assert_eq!(
            cone.view_generators(),
            vec![
                nalgebra::DVector::from_vec(vec![0, 0, 1]),
                nalgebra::DVector::from_vec(vec![0, 1, 1]),
                nalgebra::DVector::from_vec(vec![1, 0, 1]),
            ]
        );
        assert_eq!(
            cone.view_facets_idces(),
            Some(&vec![vec![0, 1], vec![0, 2], vec![1, 2],])
        );
    }

    #[test]
    fn test_does_hulling() {
        use crate::toric::cone::RationalPolyhedralCone;
        use crate::toric::polytope::ConvexPolytope;

        let poly = ConvexPolytope::new(vec![[0, 0], [2, 0], [0, 2], [2, 2], [1, 1], [1, 2]])
            .expect("valid polytope");

        let cone: RationalPolyhedralCone = poly.homogeneous_lift();

        assert_eq!(cone.ambient_dim, 3);
        assert_eq!(cone.view_generators().len(), 4);
        assert_eq!(
            cone.view_generators(),
            vec![
                nalgebra::DVector::from_vec(vec![0, 0, 1]),
                nalgebra::DVector::from_vec(vec![0, 2, 1]),
                nalgebra::DVector::from_vec(vec![2, 0, 1]),
                nalgebra::DVector::from_vec(vec![2, 2, 1]),
            ]
        );
        assert_eq!(
            cone.view_facets_idces(),
            Some(&vec![vec![0, 1], vec![0, 2], vec![1, 3], vec![2, 3],])
        );
    }

    #[test]
    fn test_degenerate() {
        use crate::toric::cone::RationalPolyhedralCone;
        use crate::toric::polytope::ConvexPolytope;
        use nalgebra::{DMatrix, DVector};

        let mut poly = ConvexPolytope::new(vec![[0, 0], [2, 0], [0, 2], [2, 2], [1, 1], [1, 2]])
            .expect("valid polytope");
        let e1 = DVector::from_vec(vec![1, 1]);
        let e2 = DVector::from_vec(vec![0, 1]);
        let matrix: DMatrix<i64> = DMatrix::from_columns(&[e1, e2]);
        let translation = DVector::from_iterator(2, [5, 9]);
        let did_compose = poly.post_compose((matrix, translation));
        assert!(did_compose.is_ok());
        let poly = poly.expand_ambient_dimesion::<5>();
        let cone: RationalPolyhedralCone = poly.homogeneous_lift();

        assert_eq!(cone.ambient_dim, 6);
        assert_eq!(cone.view_generators().len(), 4);
        assert_eq!(
            cone.view_generators(),
            vec![
                nalgebra::DVector::from_vec(vec![5, 9, 0, 0, 0, 1]),
                nalgebra::DVector::from_vec(vec![5, 11, 0, 0, 0, 1]),
                nalgebra::DVector::from_vec(vec![7, 11, 0, 0, 0, 1]),
                nalgebra::DVector::from_vec(vec![7, 13, 0, 0, 0, 1]),
            ]
        );
        assert_eq!(
            cone.view_facets_idces(),
            Some(&vec![vec![0, 1], vec![0, 2], vec![1, 3], vec![2, 3],])
        );
    }

    #[test]
    fn square_fan() {
        use super::ConvexPolytope;
        use nalgebra::{DMatrix, DVector};
        let square = ConvexPolytope::new(vec![[0, 0], [2, 0], [0, 2], [2, 2], [1, 1], [1, 2]])
            .expect("valid polytope");
        let ans = square.toric_fan().expect("Did make the fan");
        assert_eq!(ans.ambient_dim, 2);
        assert_eq!(ans.count_cones(), 4);

        let mut square = ConvexPolytope::new(vec![[0, 0], [2, 0], [0, 2], [2, 2], [1, 1], [1, 2]])
            .expect("valid polytope");
        let e1 = DVector::from_vec(vec![1, 1]);
        let e2 = DVector::from_vec(vec![0, 1]);
        let matrix: DMatrix<i64> = DMatrix::from_columns(&[e1, e2]);
        let translation = DVector::from_iterator(2, [5, 9]);
        let did_compose = square.post_compose((matrix, translation));
        assert!(did_compose.is_ok());
        let ans = square.toric_fan().expect("Did make the fan");
        assert_eq!(ans.ambient_dim, 2);
        assert_eq!(ans.count_cones(), 4);
    }

    #[test]
    fn triangle_fan() {
        use super::ConvexPolytope;
        let triangle = ConvexPolytope::new(vec![[1, 0], [0, 1], [0, 0]]).expect("valid polytope");
        let ans = triangle.toric_fan().expect("Did make the fan");
        assert_eq!(ans.ambient_dim, 2);
        assert_eq!(ans.count_cones(), 3);
        assert!(!ans.is_affine(true));

        let triangle =
            ConvexPolytope::new(vec![[1, 0], [0, 1], [-1, -1], [0, 0]]).expect("valid polytope");
        let ans = triangle.toric_fan().expect("Did make the fan");
        assert_eq!(ans.ambient_dim, 2);
        assert_eq!(ans.count_cones(), 3);
        assert!(!ans.is_affine(true));

        let triangle =
            ConvexPolytope::new(vec![[1, 0], [0, 1], [-1, -1], [0, 0]]).expect("valid polytope");
        let triangle = triangle.expand_ambient_dimesion::<4>();
        let ans = triangle.toric_fan().expect("Did make the fan");
        assert_eq!(ans.ambient_dim, 4);
        assert_eq!(ans.count_cones(), 3);
        assert!(!ans.is_affine(true));
    }

    #[test]
    fn test_contains_origin() {
        use super::ConvexPolytope;
        use nalgebra::{DMatrix, DVector};

        // Triangle with origin strictly inside
        let tri = ConvexPolytope::new(vec![[1, 0], [0, 1], [-1, -1]]).expect("valid");
        assert!(tri.contains_origin());

        // Triangle with origin strictly outside
        let tri = ConvexPolytope::new(vec![[1, 0], [2, 0], [1, 2]]).expect("valid");
        assert!(!tri.contains_origin());

        // Triangle with origin on boundary (edge from [-1,0] to [1,0])
        let tri = ConvexPolytope::new(vec![[-1, 0], [1, 0], [0, 1]]).expect("valid");
        assert!(tri.contains_origin());

        // extra_zero_coords > 0 → always false
        let tri = ConvexPolytope::new(vec![[1, 0], [0, 1], [-1, -1]]).expect("valid");
        let tri4 = tri.expand_ambient_dimesion::<4>();
        assert!(!tri4.contains_origin());

        // With affine map: shift by (3, 0) so that the origin is outside
        let mut tri = ConvexPolytope::new(vec![[1, 0], [0, 1], [-1, -1]]).expect("valid");
        let id = DMatrix::identity(2, 2);
        let shift = DVector::from_vec(vec![3_i64, 0]);
        tri.post_compose((id, shift)).expect("ok");
        assert!(!tri.contains_origin());
    }

    fn sorted_vertices<const N: usize>(p: &super::ConvexPolytope<N>) -> Vec<Vec<i64>> {
        let mut verts: Vec<Vec<i64>> = p.all_vertices().collect();
        verts.sort();
        verts
    }

    #[test]
    fn polar_dual_reflexive_triangle() {
        use super::ConvexPolytope;
        // Each edge of {(1,0),(0,1),(-1,-1)} has support value 1, so dual vertex =
        // primitive outward normal.
        let tri = ConvexPolytope::new(vec![[1, 0], [0, 1], [-1, -1]]).expect("valid");
        let dual = tri.polar_dual().expect("dual exists");
        assert_eq!(
            sorted_vertices(&dual),
            vec![vec![-2, 1], vec![1, -2], vec![1, 1]]
        );
    }

    #[test]
    fn polar_dual_reflexive_triangle_involution() {
        use super::ConvexPolytope;
        // Dual of dual round-trips.
        let tri = ConvexPolytope::new(vec![[1, 0], [0, 1], [-1, -1]]).expect("valid");
        let dual_dual = tri.polar_dual().expect("ok").polar_dual().expect("ok");
        assert_eq!(
            sorted_vertices(&dual_dual),
            vec![vec![-1, -1], vec![0, 1], vec![1, 0]]
        );
    }

    #[test]
    fn polar_dual_square_is_diamond() {
        use super::ConvexPolytope;
        // Square {(±1,±1)}: four axis-aligned facets each with h=1.
        let square = ConvexPolytope::new(vec![[1, 1], [1, -1], [-1, 1], [-1, -1]]).expect("valid");
        let dual = square.polar_dual().expect("dual exists");
        assert_eq!(
            sorted_vertices(&dual),
            vec![vec![-1, 0], vec![0, -1], vec![0, 1], vec![1, 0]]
        );
    }

    #[test]
    fn polar_dual_origin_not_contained() {
        use super::{ConvexPolytope, PolytopeError};
        let outside = ConvexPolytope::new(vec![[1, 0], [2, 0], [1, 2]]).expect("valid");
        assert!(matches!(
            outside.polar_dual(),
            Err(PolytopeError::OriginNotContained)
        ));
    }

    #[test]
    fn polar_dual_octahedron_is_cube() {
        use super::ConvexPolytope;
        // Each of the 8 facets of the octahedron has primitive outward normal (±1,±1,±1) with h=1.
        let octa = ConvexPolytope::new(vec![
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ])
        .expect("valid");
        let cube = octa.polar_dual().expect("dual exists");
        assert_eq!(
            sorted_vertices(&cube),
            vec![
                vec![-1, -1, -1],
                vec![-1, -1, 1],
                vec![-1, 1, -1],
                vec![-1, 1, 1],
                vec![1, -1, -1],
                vec![1, -1, 1],
                vec![1, 1, -1],
                vec![1, 1, 1],
            ]
        );
    }

    #[test]
    fn reflexive_2d_are_reflexive() {
        use super::ConvexPolytope;
        for (name, poly) in ConvexPolytope::reflexive_2d() {
            assert!(poly.contains_origin(), "{name}: origin not in interior");
            let dual = poly
                .polar_dual()
                .unwrap_or_else(|e| panic!("{name}: polar_dual failed: {e:?}"));
            dual.polar_dual()
                .unwrap_or_else(|e| panic!("{name}: dual is not reflexive: {e:?}"));
        }
    }

    #[test]
    fn interior_lattice_points_reflexive_2d() {
        use super::ConvexPolytope;
        use nalgebra::{DMatrix, DVector};

        // polygon_01 (3-vertex triangle): origin is the sole interior lattice point
        let poly = ConvexPolytope::<2>::polygon_01().1;
        let pts = poly.interior_lattice_points();
        assert_eq!(pts, vec![[0, 0]], "polygon_01 plain");

        // polygon_04 (square): translated by (2, -3), sole interior point shifts accordingly
        let mut poly = ConvexPolytope::<2>::polygon_04().1;
        poly.post_compose((DMatrix::identity(2, 2), DVector::from_vec(vec![2i64, -3])))
            .unwrap();
        let pts = poly.interior_lattice_points();
        assert_eq!(pts, vec![[2, -3]], "polygon_04 translated");

        // polygon_10 (hexagon): expanded to 4D ambient, origin maps to (0,0,0,0)
        let poly = ConvexPolytope::<2>::polygon_10()
            .1
            .expand_ambient_dimesion::<4>();
        let pts = poly.interior_lattice_points();
        assert_eq!(pts, vec![[0, 0, 0, 0]], "polygon_10 expanded to 4D");

        // polygon_15 (diamond): expanded to 4D then translated by (3,1,0,0)
        let mut poly = ConvexPolytope::<2>::polygon_15()
            .1
            .expand_ambient_dimesion::<4>();
        poly.post_compose((
            DMatrix::identity(4, 4),
            DVector::from_vec(vec![3i64, 1, 0, 0]),
        ))
        .unwrap();
        let pts = poly.interior_lattice_points();
        assert_eq!(pts, vec![[3, 1, 0, 0]], "polygon_15 expanded+translated");

        // polygon_16 (3-vertex triangle): shear [[1,1],[0,1]] + translation (2,3)
        // maps origin → (2,3); bounding box still contains only one pre-image
        let mut poly = ConvexPolytope::<2>::polygon_16().1;
        poly.post_compose((
            DMatrix::from_row_slice(2, 2, &[1i64, 1, 0, 1]),
            DVector::from_vec(vec![2i64, 3]),
        ))
        .unwrap();
        let pts = poly.interior_lattice_points();
        assert_eq!(pts, vec![[2, 3]], "polygon_16 shear+translated");
    }

    #[test]
    fn lattice_points_all_codims_cube_in_n5() {
        use super::ConvexPolytope;
        use nalgebra::{DMatrix, DVector};

        // Start with the cube [-1,1]^3, expand to ambient ℝ⁵ (2 extra zero coords),
        // then apply the full GL(5,ℤ) matrix + translation:
        //   row 3 = e_0 + e_3  → dim 3 of output picks up the x coordinate
        //   row 4 = e_2 + e_4  → dim 4 of output picks up the z coordinate
        // Row-reducing to identity confirms det = 1.
        // Combined with translation (3,−2,1,5,−4) the map is:
        //   (x,y,z,0,0) ↦ (x+y+3, y−2, z+1, x+5, z−4)

        let cube = ConvexPolytope::new(vec![
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ])
        .expect("valid cube");

        let mut cube5 = cube.expand_ambient_dimesion::<5>();
        #[rustfmt::skip]
        let a = DMatrix::from_row_slice(5, 5, &[
            1i64, 1, 0, 0, 0,
            0,    1, 0, 0, 0,
            0,    0, 1, 0, 0,
            1,    0, 0, 1, 0,
            0,    0, 1, 0, 1,
        ]);
        cube5
            .post_compose((a, DVector::from_vec(vec![3i64, -2, 1, 5, -4])))
            .unwrap();

        // k=0: origin → (3,−2,1,5,−4)
        assert_eq!(cube5.lattice_points_of_codim(0), vec![[3, -2, 1, 5, -4]]);

        // k=1: 6 face centres
        let mut face_pts = cube5.lattice_points_of_codim(1);
        face_pts.sort();
        assert_eq!(
            face_pts,
            vec![
                [2, -3, 1, 5, -4],
                [2, -2, 1, 4, -4],
                [3, -2, 0, 5, -5],
                [3, -2, 2, 5, -3],
                [4, -2, 1, 6, -4],
                [4, -1, 1, 5, -4],
            ]
        );

        // k=2: 12 edge midpoints
        let mut edge_pts = cube5.lattice_points_of_codim(2);
        edge_pts.sort();
        assert_eq!(
            edge_pts,
            vec![
                [1, -3, 1, 4, -4],
                [2, -3, 0, 5, -5],
                [2, -3, 2, 5, -3],
                [2, -2, 0, 4, -5],
                [2, -2, 2, 4, -3],
                [3, -3, 1, 6, -4],
                [3, -1, 1, 4, -4],
                [4, -2, 0, 6, -5],
                [4, -2, 2, 6, -3],
                [4, -1, 0, 5, -5],
                [4, -1, 2, 5, -3],
                [5, -1, 1, 6, -4],
            ]
        );

        // k=3: 8 vertices
        let mut verts = cube5.lattice_points_of_codim(3);
        verts.sort();
        assert_eq!(
            verts,
            vec![
                [1, -3, 0, 4, -5],
                [1, -3, 2, 4, -3],
                [3, -3, 0, 6, -5],
                [3, -3, 2, 6, -3],
                [3, -1, 0, 4, -5],
                [3, -1, 2, 4, -3],
                [5, -1, 0, 6, -5],
                [5, -1, 2, 6, -3],
            ]
        );

        // All 27 = 3^3 integer pre-images are accounted for across all codimensions
        let total: usize = (0..=3)
            .map(|k| cube5.lattice_points_of_codim(k).len())
            .sum();
        assert_eq!(total, 27);

        // k beyond the polytope dimension yields nothing
        assert!(cube5.lattice_points_of_codim(4).is_empty());
        assert!(cube5.lattice_points_of_codim(5).is_empty());
        assert!(cube5.lattice_points_of_codim(100).is_empty());
    }

    #[test]
    fn test_to_primitive() {
        use super::to_primitive;

        let d = nalgebra::DVector::from_iterator(3, [1.0, 2.0, 3.0]);
        let expected = nalgebra::DVector::from_iterator(3, [1, 2, 3]);
        let (mul_by, d_prime) = to_primitive(d);
        assert_eq!(d_prime, expected);
        assert_eq!(mul_by, 1);

        for div_by_int in [2_i8, 3, 5] {
            let div_by = f64::from(div_by_int).sqrt();
            let d = nalgebra::DVector::from_iterator(3, [1.0 / div_by, 2.0 / div_by, 3.0 / div_by]);
            let expected = nalgebra::DVector::from_iterator(3, [1, 2, 3]);
            let (mul_by, d_prime) = to_primitive(d);
            assert_eq!(d_prime, expected);
            assert_eq!(mul_by, div_by_int.into());
        }
    }
}
