//use ark_bls12_381::Fr as F;
use ark_goldilocks::Goldilocks as F;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};

#[cfg(feature = "parallel")]
use once_cell::sync::OnceCell;
#[cfg(feature = "parallel")]
use rayon::ThreadPoolBuilder;

/// Initialise Rayon’s global thread pool once.
/// Call this during program start-up if you want to pin the FFT work
/// to the two vCPUs on the t4g.micro.
#[cfg(feature = "parallel")]
pub fn init_parallelism(num_threads: usize) {
    static THREAD_POOL_INIT: OnceCell<()> = OnceCell::new();

    THREAD_POOL_INIT.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .expect("initialize rayon global thread pool");
    });
}

/// No-op fallback when the `parallel` feature is disabled.
#[cfg(not(feature = "parallel"))]
pub fn init_parallelism(_num_threads: usize) {}

/// Perform IFFT in place without copying through a temporary buffer.
pub fn ifft_in_place(domain: &Radix2EvaluationDomain<F>, vals: &mut Vec<F>) {
    domain.ifft_in_place(vals);
}

/// Perform FFT in place without copying through a temporary buffer.
pub fn fft_in_place(domain: &Radix2EvaluationDomain<F>, vals: &mut Vec<F>) {
    domain.fft_in_place(vals);
}

/// Convenience helper that allocates a new Vec and returns the result.
pub fn fft(domain: &Radix2EvaluationDomain<F>, coeffs: &[F]) -> Vec<F> {
    let mut v: Vec<F> = coeffs.to_vec();
    domain.fft_in_place(&mut v);
    v
}

/// Convenience helper that allocates a new Vec and returns the result.
pub fn ifft(domain: &Radix2EvaluationDomain<F>, evals: &[F]) -> Vec<F> {
    let mut v: Vec<F> = evals.to_vec();
    domain.ifft_in_place(&mut v);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::One;

    #[test]
    fn roundtrip_fft_ifft() {
        #[cfg(feature = "parallel")]
        init_parallelism(2);

        let n = 8usize;
        let domain = Radix2EvaluationDomain::<F>::new(n).expect("domain");
        let mut coeffs = vec![F::one(); n];

        // Vec-based FFT / IFFT round-trip
        let evals = fft(&domain, &coeffs);
        let back = ifft(&domain, &evals);
        assert_eq!(coeffs, back);

        // mutate Vec in-place
        fft_in_place(&domain, &mut coeffs);
        ifft_in_place(&domain, &mut coeffs);
        assert_eq!(coeffs, vec![F::one(); n]);
    }
}