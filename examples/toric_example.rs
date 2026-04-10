use geometric_rep_theory::toric::{DefaultRepr, RationalPolyhedralCone, ToricFan};

pub fn toric_fan_examples() {
    for (fan_name, fan) in ToricFan::common_cases() {
        println!(
            "Example Fan {fan_name}: {:?}",
            fan.iter_cones().collect::<Vec<_>>()
        );
    }
}

#[allow(clippy::missing_panics_doc)]
pub fn toric_ideal_example() {
    // Example usage
    let cone = RationalPolyhedralCone::c2();

    let presentation = cone
        .coordinate_ring_presentation::<2>(true)
        .expect("Valid coordinate ring presentation");

    println!("Coordinate Ring Presentation of C^2: {presentation}");

    let mut fan = ToricFan::c3();
    let presentation = fan
        .coordinate_ring_presentation::<3, DefaultRepr>(|z| z, true)
        .expect("C3 fan");

    println!("Coordinate Ring Presentation of C^3: {presentation}");

    assert!(presentation.generators.is_empty());

    let cone =
        RationalPolyhedralCone::new(vec![vec![1, 0], vec![2, 0]], Some(false), Some(1), None)
            .expect("Valid cone");

    assert_eq!(cone.num_rays(), Ok(1));
    assert_eq!(cone.view_generators().len(), 1);

    let presentation = cone
        .coordinate_ring_presentation::<2>(true)
        .expect_err("Invalid coordinate ring presentation");

    println!("Coordinate Ring Presentation of e1 only in R^2: {presentation:?}");

    let presentation = cone
        .coordinate_ring_presentation::<1>(true)
        .expect("Valid coordinate ring presentation");

    println!("Coordinate Ring Presentation of e1 only in R^2: {presentation}");
}

pub fn main() {
    toric_fan_examples();
    toric_ideal_example();
}
