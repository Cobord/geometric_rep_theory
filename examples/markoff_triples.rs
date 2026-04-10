use geometric_rep_theory::cluster_algebra::ClusterAlgebra;
use num::integer::gcd;

fn main() {
    let mut markoff_quiver = geometric_rep_theory::quiver_algebra::Quiver::new();
    let _x_vertex = markoff_quiver.add_vertex("x");
    let _y_vertex = markoff_quiver.add_vertex("y");
    let _z_vertex = markoff_quiver.add_vertex("z");
    markoff_quiver.add_edge("x", "y", "xy_1".to_string());
    markoff_quiver.add_edge("x", "y", "xy_2".to_string());
    markoff_quiver.add_edge("y", "z", "yz_1".to_string());
    markoff_quiver.add_edge("y", "z", "yz_2".to_string());
    markoff_quiver.add_edge("z", "x", "zx_1".to_string());
    markoff_quiver.add_edge("z", "x", "zx_2".to_string());
    let mut seed_map = std::collections::HashMap::with_capacity(3);
    seed_map.insert("x", 1i32);
    seed_map.insert("y", 1);
    seed_map.insert("z", 1);
    let mut markoff_cluster_algebra =
        ClusterAlgebra::<3, _, _, _>::new(markoff_quiver, seed_map, |idx| format!("x_{idx}"))
            .expect("Is valid");
    let simplifier = |(a, b): &mut (i32, i32)| {
        let g = gcd(*a, *b);
        *a /= g;
        *b /= g;
    };
    let xyz_values = markoff_cluster_algebra.view_cluster(["x", "y", "z"]);
    assert_eq!(xyz_values, [(1, 1), (1, 1), (1, 1)]);
    println!(
        "Initial cluster: {:?}",
        xyz_values.map(|(a, b)| format!("{}", a / b))
    );
    markoff_cluster_algebra.mutate(&"x");
    let xyz_values = markoff_cluster_algebra.view_cluster(["x", "y", "z"]);
    assert_eq!(xyz_values, [(2, 1), (1, 1), (1, 1)]);
    println!(
        "Mutate at x: {:?}",
        xyz_values.map(|(a, b)| format!("{}", a / b))
    );
    markoff_cluster_algebra.mutate(&"x");
    let xyz_values = markoff_cluster_algebra.view_cluster(["x", "y", "z"]);
    assert_eq!(xyz_values, [(2, 2), (1, 1), (1, 1)]);
    println!(
        "Mutate at x again back to the beginning: {:?}",
        xyz_values.map(|(a, b)| format!("{}", a / b))
    );
    markoff_cluster_algebra.mutate(&"y");
    markoff_cluster_algebra.simplify(simplifier);
    let xyz_values = markoff_cluster_algebra.view_cluster(["x", "y", "z"]);
    assert_eq!(xyz_values, [(1, 1), (2, 1), (1, 1)]);
    markoff_cluster_algebra.mutate(&"z");
    markoff_cluster_algebra.mutate(&"x");
    markoff_cluster_algebra.mutate(&"y");
    markoff_cluster_algebra.simplify(simplifier);
    let xyz_values = markoff_cluster_algebra.view_cluster(["x", "y", "z"]);
    assert_eq!(xyz_values, [(29, 1), (433, 1), (5, 1)]);
    println!(
        "Mutate at y,z,x,y in that order: {:?}",
        xyz_values.map(|(a, b)| format!("{}", a / b))
    );
}
