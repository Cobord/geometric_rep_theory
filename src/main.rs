use crate::toric::{main_examples, main_toric_ideal_example};

pub mod quiver_algebra;
pub mod toric;
mod utils;

fn main() {
    main_examples();
    main_toric_ideal_example();
}
