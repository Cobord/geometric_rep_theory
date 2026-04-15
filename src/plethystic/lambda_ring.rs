pub trait LambdaRing: crate::arithmetic_utils::Ring {
    /// The n-th lambda operation. Must satisfy λ^1(x) = x.
    #[must_use = "A new element in the lambda ring which is lambda^n of the input is returned"]
    fn lambda(self, n: usize) -> Self;

    /// The n-th Adams (power-sum) operation ψ^n.
    ///
    /// Default implementation uses Newton's identity iteratively:
    ///   ψ^k = Σ_{j=1}^{k-1} (-1)^{j-1} λ^j · ψ^{k-j}  +  (-1)^{k-1} k·λ^k
    ///
    /// Implementors may override this with a direct formula for efficiency.
    #[must_use = "A new element in the lambda ring which is psi^n of the input is returned"]
    fn psi(self, n: usize) -> Self {
        if n == 0 {
            return Self::one();
        }
        // psi_vals[i] holds ψ^{i+1}, so index i corresponds to exponent i+1.
        let mut psi_vals: Vec<Self> = Vec::with_capacity(n);
        for k in 1..=n {
            // Newton: ψ^k = Σ_{j=1}^{k-1} (-1)^{j-1} λ^j(self) · ψ^{k-j}
            //               + (-1)^{k-1} k·λ^k(self)
            let mut val = Self::zero();
            for j in 1..k {
                let lj = self.clone().lambda(j);
                // ψ^{k-j} is at index (k-j)-1 = k-j-1
                let psi_k_minus_j = psi_vals[k - j - 1].clone();
                if j % 2 == 1 {
                    val += lj * psi_k_minus_j;
                } else {
                    val -= lj * psi_k_minus_j;
                }
            }
            // k · λ^k(self)
            let lk = self.clone().lambda(k);
            let k_lk = Self::natural_inclusion(k) * lk;
            if k % 2 == 1 {
                val += k_lk;
            } else {
                val -= k_lk;
            }
            psi_vals.push(val);
        }
        // ψ^n is at index n-1
        psi_vals.swap_remove(n - 1)
    }
}
