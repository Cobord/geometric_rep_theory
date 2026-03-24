use std::vec;

use crate::toric::cone::RationalPolyhedralCone;
use crate::toric::fan::ToricFan;

impl RationalPolyhedralCone {
    #[must_use = "C^2 cone constructed at some expense, either use it or avoid calling this method"]
    #[allow(clippy::missing_panics_doc)]
    pub fn c2() -> Self {
        RationalPolyhedralCone::new(
            vec![vec![1, 0], vec![0, 1]],
            Some(true),
            Some(2),
            Some(vec![vec![0], vec![1]]),
        )
        .expect("C^2 cone generators are valid and primitive")
    }

    #[must_use = "C^3 cone constructed at some expense, either use it or avoid calling this method"]
    #[allow(clippy::missing_panics_doc)]
    pub fn c3() -> Self {
        RationalPolyhedralCone::new(
            vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]],
            Some(true),
            Some(3),
            Some(vec![vec![0, 1], vec![1, 2], vec![0, 2]]),
        )
        .expect("C^3 cone generators are valid and primitive")
    }
}

impl ToricFan {
    /// The fan of C^2
    #[must_use = "This constructs the C^2 fan at some expense, either use it or avoid calling this method"]
    #[allow(clippy::missing_panics_doc)]
    pub fn c2() -> Self {
        let cone = RationalPolyhedralCone::c2();

        let mut fan = ToricFan::new(2);
        fan.add_cone(cone, true)
            .expect("Single cone always forms a valid fan");
        fan
    }

    /// The fan of C^3
    #[must_use = "This constructs the C^3 fan at some expense, either use it or avoid calling this method"]
    #[allow(clippy::missing_panics_doc)]
    pub fn c3() -> Self {
        let cone = RationalPolyhedralCone::c3();

        let mut fan = ToricFan::new(3);
        fan.add_cone(cone, true)
            .expect("Single cone always forms a valid fan");
        fan
    }

    /// The fan of P^2
    #[must_use = "This constructs the P^2 fan at some expense, either use it or avoid calling this method"]
    #[allow(clippy::missing_panics_doc)]
    pub fn p2() -> Self {
        let c1 = RationalPolyhedralCone::c2();

        let c2 = RationalPolyhedralCone::new(
            vec![vec![0, 1], vec![-1, -1]],
            Some(true),
            Some(2),
            Some(vec![vec![0], vec![1]]),
        )
        .expect("P^2 cone 2 is valid");

        let c3 = RationalPolyhedralCone::new(
            vec![vec![-1, -1], vec![1, 0]],
            Some(true),
            Some(2),
            Some(vec![vec![0], vec![1]]),
        )
        .expect("P^2 cone 3 is valid");

        let mut fan = ToricFan::new(2);
        fan.add_cone(c1, true).expect("P^2 fan condition holds");
        fan.add_cone(c2, false).expect("P^2 fan condition holds");
        fan.add_cone(c3, false).expect("P^2 fan condition holds");
        fan
    }

    #[must_use = "This constructs the Conifold fan at some expense, either use it or avoid calling this method"]
    #[allow(clippy::missing_panics_doc)]
    pub fn conifold() -> Self {
        let c1 = RationalPolyhedralCone::c3();

        let c2 = RationalPolyhedralCone::new(
            vec![vec![1, 0, 0], vec![0, 1, 0], vec![1, 1, -1]],
            Some(true),
            Some(3),
            Some(vec![vec![0, 1], vec![0, 2], vec![1, 2]]),
        )
        .expect("Conifold cone 2 is valid");

        let mut fan = ToricFan::new(3);
        fan.add_cone(c1, true)
            .expect("Conifold fan condition holds");
        fan.add_cone(c2, false)
            .expect("Conifold fan condition holds");
        fan
    }

    #[must_use = "This constructs the Conifold fan at some expense, either use it or avoid calling this method"]
    #[allow(clippy::missing_panics_doc)]
    pub fn conifold2() -> Self {
        let c1 = RationalPolyhedralCone::new(
            vec![vec![1, 0, 0], vec![0, 1, 0], vec![1, 0, 1], vec![0, 1, 1]],
            Some(true),
            Some(3),
            Some(vec![vec![0, 1], vec![0, 2], vec![1, 3], vec![2, 3]]),
        )
        .expect("Conifold cone 1 is valid");

        let mut fan = ToricFan::new(3);
        fan.add_cone(c1, true)
            .expect("Conifold fan condition holds");
        fan
    }

    /// A collection of common example fans
    pub fn common_cases() -> impl Iterator<Item = (String, ToricFan)> {
        [
            Self::c2,
            Self::c3,
            Self::p2,
            Self::conifold,
            Self::conifold2,
        ]
        .into_iter()
        .enumerate()
        .map(|(idx, f)| {
            let name = match idx {
                0 => "C^2",
                1 => "C^3",
                2 => "P^2",
                3 | 4 => "Conifold",
                _ => "Unknown",
            };
            (name.to_string(), f())
        })
    }
}

pub(crate) fn main_examples() {
    for (fan_name, fan) in ToricFan::common_cases() {
        println!(
            "Example Fan {fan_name}: {:?}",
            fan.iter_cones().collect::<Vec<_>>()
        );
    }
}
