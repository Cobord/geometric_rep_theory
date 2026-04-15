#[allow(dead_code)]
pub trait LambdaRing : crate::arithmetic_utils::Ring {
    fn lambda(self, n: usize) -> Self;
}