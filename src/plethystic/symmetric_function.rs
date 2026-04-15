use std::collections::HashMap;

use crate::arithmetic_utils::Ring;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PowerSumPolynomial {
    partition : Vec<usize>,
}

#[allow(dead_code)]
pub struct SymmetricFunction<Coeffs: Ring> {
    l : HashMap<PowerSumPolynomial, Coeffs>,
}