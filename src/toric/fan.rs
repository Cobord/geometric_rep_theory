use std::cell::Cell;

use itertools::Itertools;
use nalgebra::DVector;

use crate::toric::cone::RationalPolyhedralCone;
use crate::toric::cone_errors::{ConeError, ToricFanError};
use crate::toric::toric_ideal::{
    CoordinateRingError, CoordinateRingPresentation, CoordinateRingRepr, DefaultRepr,
};

/// Toric fan
pub struct ToricFan {
    cones: Vec<RationalPolyhedralCone>,
    pub ambient_dim: usize,
    variety_dim: Cell<Option<usize>>,
    cached_coordinate_ring: Option<Box<dyn std::any::Any>>, // can store generic presentation
}

impl ToricFan {
    #[must_use = "Constructing a fan is not that bad, but either use it or avoid calling this method"]
    pub fn new(ambient_dim: usize) -> Self {
        ToricFan {
            cones: Vec::new(),
            ambient_dim,
            variety_dim: Cell::new(None),
            cached_coordinate_ring: None,
        }
    }

    /// # Errors
    ///
    /// TODO
    pub fn add_cone(
        &mut self,
        cone: RationalPolyhedralCone,
        check_intersection: bool,
    ) -> Result<(), ToricFanError> {
        // Check that intersection with every existing cone is a face of both
        self.variety_dim.set(None);
        self.cached_coordinate_ring = None;
        if self.ambient_dim != cone.ambient_dim {
            return Err(ToricFanError::ConeError(ConeError::DimensionMismatch {
                expected: self.ambient_dim,
                found: cone.ambient_dim,
            }));
        }

        if check_intersection {
            for existing in &self.cones {
                if !existing
                    .intersection_is_face_of_both(&cone)
                    .map_err(ToricFanError::ConeError)?
                {
                    return Err(ToricFanError::IntersectionNotFace {
                        cone1: Box::new(existing.clone()),
                        cone2: Box::new(cone.clone()),
                    });
                }
            }
        }

        // Avoid duplicates
        if !self.cones.contains(&cone) {
            self.cones.push(cone);
        }
        if self.cones.len() == 1 {
            self.variety_dim.set(self.cones[0].variety_dim().ok());
        }

        Ok(())
    }

    pub fn iter_cones(&self) -> impl Iterator<Item = &RationalPolyhedralCone> {
        self.cones.iter()
    }

    #[must_use = "Counting cones is cheap, either use it or avoid calling this method"]
    pub fn count_cones(&self) -> usize {
        self.cones.len()
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn iter_combinatorial_intersections(
        &self,
    ) -> impl Iterator<Item = RationalPolyhedralCone> + '_ {
        self.cones.iter().tuple_combinations().map(|(c1, c2)| {
            c1.combinatorial_intersection(c2)
                .expect("By construction they all have the same dimension")
        })
    }

    /// Compute variety dimension lazily
    pub fn variety_dim(&self, ignore_error: bool) -> Option<usize> {
        if let Some(dim) = self.variety_dim.get() {
            return Some(dim);
        }
        if self.cones.len() == 1 {
            self.variety_dim.set(self.cones[0].variety_dim().ok());
            return self.variety_dim.get();
        }

        if self.is_affine(ignore_error) {
            return if let Ok(global) = self.global_support_cone() {
                let dim = global.variety_dim().ok();
                self.variety_dim.set(dim);
                dim
            } else {
                None
            };
        }

        None
    }

    #[must_use = "Checking if a fan is affine might be expensive, either use it or avoid calling this method"]
    pub fn is_affine(&self, ignore_error: bool) -> bool {
        let n = self.cones.len();

        if n == 0 {
            return true;
        }
        if n == 1 {
            return true;
        }

        // Collect all generators
        let mut generators = Vec::new();
        for cone in &self.cones {
            for g in cone.iter_generators() {
                generators.push(g.clone());
            }
        }

        // Try to find a separating functional using LP:
        // Find w such that <w, g> > 0 for all g ≠ 0
        //
        // This is equivalent to checking that
        // support ∩ (-support) = {0}

        let exists_line = generators.iter().any(|g| {
            let neg: Vec<i64> = g.iter().map(|&x| -x).collect();
            let neg_v = DVector::from(neg);
            self.cones.iter().any(|c| c.contains_vector(&neg_v))
        });

        if exists_line {
            return false;
        }

        if ignore_error {
            true
        } else {
            todo!(
                "For example, Cone(e1,e2) + Cone(e2-e1) without (e2-0.5*e1) being in the fan. This is a phenomenon of them being glued on only 0 face of both rather than a codimension 1 facet."
            );
        }
    }

    pub fn is_simplicial(&self) -> bool {
        self.cones.iter().all(RationalPolyhedralCone::is_simplicial)
    }

    pub fn is_smooth(&self) -> bool {
        self.cones.iter().all(RationalPolyhedralCone::is_smooth)
    }

    #[must_use = "Checking if a fan is complete might be expensive, either use it or avoid calling this method"]
    #[allow(clippy::missing_panics_doc)]
    pub fn complete(&self) -> bool {
        let without_correct_support = self
            .global_support_cone()
            .expect("collecting them all and just declaring the very redundant cone on all this")
            .with_spanning_dim()
            .view_spanning_dim()
            .expect("Made the spanning dimension")
            == self.ambient_dim;
        if !without_correct_support {
            return false;
        }
        todo!("Actually handle support");
    }

    #[cfg(test)]
    pub(crate) fn view_variety_dim(&self) -> Option<usize> {
        self.variety_dim.get()
    }
}

impl ToricFan {
    /// Take ownership of two fans and produce their product fan
    #[must_use = "Consumed self and other to produce a new fan at some expense, either use it or avoid calling this method"]
    #[allow(clippy::needless_pass_by_value, unreachable_code, unused_mut, unused)]
    pub fn product(mut self, other: ToricFan) -> ToricFan {
        todo!("Implement product of fans");
        let mut cones = Vec::new();

        for c1 in self.cones.drain(..) {
            for c2 in &other.cones {
                cones.push(c1.product_with(c2));
            }
        }

        ToricFan {
            cones,
            ambient_dim: self.ambient_dim + other.ambient_dim,
            variety_dim: Cell::from(None),
            cached_coordinate_ring: None,
        }
    }

    /// # Errors
    ///
    /// TODO
    pub fn global_support_cone(&self) -> Result<RationalPolyhedralCone, ToricFanError> {
        let num_cones = self.cones.len();

        if num_cones == 0 {
            return Ok(RationalPolyhedralCone::zero_cone(self.ambient_dim));
        }

        if num_cones == 1 {
            // Caller promised >1 cones, but handle safely
            return Ok(self.cones[0].clone());
        }

        // Collect all generators, deduplicated by exact equality
        let mut gens: Vec<Vec<i64>> = Vec::new();

        for cone in &self.cones {
            for g in cone.iter_generators() {
                let g_vec: Vec<i64> = g.iter().copied().collect();
                if !gens.contains(&g_vec) {
                    gens.push(g_vec);
                }
            }
        }

        // By assumption:
        // - is_affine() already succeeded
        // - the union is convex and strongly convex
        // So this cone is valid without further checks
        RationalPolyhedralCone::new(gens, Some(false), None, None)
            .map_err(|_| ToricFanError::GlobalConeOnlyForAffine)
    }

    /// # Errors
    ///
    /// TODO
    pub fn coordinate_ring_presentation<const N: usize, R: CoordinateRingRepr>(
        &mut self,
        converter: fn(DefaultRepr) -> R,
        ignore_error: bool,
    ) -> Result<CoordinateRingPresentation<N, R>, CoordinateRingError> {
        if let Some(boxed) = &self.cached_coordinate_ring {
            // downcast to concrete type
            if let Some(pres) = boxed.downcast_ref::<CoordinateRingPresentation<N, DefaultRepr>>() {
                return Ok(pres.clone().conversion(converter));
            }
        }

        let pres = self.compute_coordinate_ring_presentation::<N>(ignore_error)?;
        self.cached_coordinate_ring = Some(Box::new(pres.clone()));
        Ok(pres.conversion(converter))
    }
}

mod test {
    #[test]
    fn e1e2() {
        use super::{RationalPolyhedralCone, ToricFan};
        let c1 = RationalPolyhedralCone::new(
            vec![vec![1, 0], vec![0, 1]],
            Some(true),
            Some(2),
            Some(vec![vec![0], vec![1]]),
        )
        .expect("e1 e2 Valid cone");
        assert_eq!(c1.ambient_dim, 2);
        assert_eq!(c1.num_rays(), Ok(2));
        assert_eq!(c1.variety_dim(), Ok(2));
        assert_eq!(c1.view_spanning_dim(), Some(2));
        let c1 = c1;
        assert!(c1.is_smooth());
        let mut fan = ToricFan::new(2);
        assert_eq!(fan.ambient_dim, 2);
        assert_eq!(fan.cones.len(), 0);
        assert_eq!(fan.view_variety_dim(), None);
        assert!(fan.cached_coordinate_ring.is_none());
        fan.add_cone(c1, true)
            .expect("Only one cone added never causes issue with intersections");
        assert_eq!(fan.ambient_dim, 2);
        assert_eq!(fan.cones.len(), 1);
        assert_eq!(fan.view_variety_dim(), Some(2));
        assert!(fan.cached_coordinate_ring.is_none());
        assert!(fan.is_smooth());
    }

    #[test]
    fn e1e2_e2() {
        use super::{RationalPolyhedralCone, ToricFan};
        let c2 = RationalPolyhedralCone::new(
            vec![vec![1, 1], vec![0, 1]],
            Some(true),
            Some(2),
            Some(vec![vec![0], vec![1]]),
        )
        .expect("e1+e2 e2 Valid cone");
        let mut fan = ToricFan::new(2);
        fan.add_cone(c2, true)
            .expect("Only one cone added never causes issue with intersections");
        assert_eq!(fan.ambient_dim, 2);
        assert_eq!(fan.cones.len(), 1);
        assert_eq!(fan.view_variety_dim(), Some(2));
        assert!(fan.cached_coordinate_ring.is_none());
        assert!(fan.is_smooth());
    }

    #[test]
    fn both_above_cones() {
        use super::{RationalPolyhedralCone, ToricFan};
        let c1 = RationalPolyhedralCone::new(
            vec![vec![1, 0], vec![0, 1]],
            Some(true),
            Some(2),
            Some(vec![vec![0], vec![1]]),
        )
        .expect("e1 e2 Valid cone");
        let mut fan = ToricFan::new(2);
        fan.add_cone(c1, true)
            .expect("Only one cone added never causes issue with intersections");
        let c2 = RationalPolyhedralCone::new(
            vec![vec![1, 1], vec![0, 1]],
            Some(true),
            Some(2),
            Some(vec![vec![0], vec![1]]),
        )
        .expect("e1+e2 e2 Valid cone");
        assert!(fan.add_cone(c2, false).is_ok());
        assert_eq!(fan.ambient_dim, 2);
        assert_eq!(fan.cones.len(), 2);
        assert_eq!(fan.view_variety_dim(), None);
        assert_eq!(fan.variety_dim(true), None);
        assert!(fan.cached_coordinate_ring.is_none());
        assert!(fan.is_smooth());
    }

    #[test]
    fn nonsmooth() {
        use super::{RationalPolyhedralCone, ToricFan};
        for n in [2, 3, 4, 5] {
            let c1 = RationalPolyhedralCone::new(
                vec![vec![1, 0], vec![1, n]],
                Some(true),
                Some(2),
                Some(vec![vec![0], vec![1]]),
            )
            .expect("e1 e2 Valid cone");
            assert_eq!(c1.ambient_dim, 2);
            assert_eq!(c1.num_rays(), Ok(2));
            assert_eq!(c1.variety_dim(), Ok(2));
            assert_eq!(c1.view_spanning_dim(), Some(2));
            let c1 = c1;
            assert!(!c1.is_smooth());
            let mut fan = ToricFan::new(2);
            assert_eq!(fan.ambient_dim, 2);
            assert_eq!(fan.cones.len(), 0);
            assert_eq!(fan.view_variety_dim(), None);
            assert!(fan.cached_coordinate_ring.is_none());
            fan.add_cone(c1, true)
                .expect("Only one cone added never causes issue with intersections");
            assert_eq!(fan.ambient_dim, 2);
            assert_eq!(fan.cones.len(), 1);
            assert_eq!(fan.view_variety_dim(), Some(2));
            assert!(fan.cached_coordinate_ring.is_none());
            assert!(!fan.is_smooth());
        }
    }
}
