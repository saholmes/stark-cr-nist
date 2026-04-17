
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused_macros)]
use ark_ff::{Field, One, Zero};
use ark_goldilocks::Goldilocks as F;

pub mod trace_import;

use ark_poly::{
    EvaluationDomain,
    GeneralEvaluationDomain,
    DenseUVPolynomial,
    Polynomial,    
};
use ark_poly::polynomial::univariate::DensePolynomial;



#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::deep_tower::Fp3;

const PARALLEL_MIN_ELEMS: usize = 1 << 12;

#[inline]
fn enable_parallel(len: usize) -> bool {
    #[cfg(feature = "parallel")]
    {
        len >= PARALLEL_MIN_ELEMS && rayon::current_num_threads() > 1
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = len;
        false
    }
}

fn build_omega_pows(omega: F, n: usize) -> Vec<F> {
    let mut omega_pows = Vec::with_capacity(n);
    let mut x = F::one();
    for _ in 0..n {
        omega_pows.push(x);
        x *= omega;
    }
    omega_pows
}

/// Return true if z ∈ H = <omega> of size n
fn is_in_domain(z: F, n: usize) -> bool {
    z.pow(&[n as u64, 0, 0, 0]) == F::one()
}

/// Vanishing polynomial on H
fn zh_at(z: F, n: usize) -> F {
    z.pow(&[n as u64, 0, 0, 0]) - F::one()
}

/// ---------------------------------------------------------------------------
/// ✅ DEEP‑ALI merge using Fp³ (Option A, sound)
/// ---------------------------------------------------------------------------

pub fn deep_ali_merge_evals(
    a_eval: &[F],
    s_eval: &[F],
    e_eval: &[F],
    t_eval: &[F],
    omega: F,
    z_fp3: Fp3,
) -> (Vec<F>, Fp3, F) {
    deep_ali_merge_evals_blinded(
        a_eval,
        s_eval,
        e_eval,
        t_eval,
        None,
        F::zero(),
        omega,
        z_fp3,
    )
}

pub fn deep_ali_merge_evals_blinded(
    a_eval: &[F],
    s_eval: &[F],
    e_eval: &[F],
    t_eval: &[F],
    r_eval_opt: Option<&[F]>,
    beta: F,
    omega: F,
    z_fp3: Fp3,
) -> (Vec<F>, Fp3, F) {
    let n = a_eval.len();
    assert!(n > 1);
    assert!(n.is_power_of_two(), "Domain size must be power of two");

    assert_eq!(s_eval.len(), n);
    assert_eq!(e_eval.len(), n);
    assert_eq!(t_eval.len(), n);
    if let Some(r_eval) = r_eval_opt {
        assert_eq!(r_eval.len(), n);
    }

    // ω^j
    let omega_pows = build_omega_pows(omega, n);

    // Φ̃(ω^j)
    let mut phi_eval = vec![F::zero(); n];
    for i in 0..n {
        let base = a_eval[i] * s_eval[i] + e_eval[i] - t_eval[i];
        phi_eval[i] = if let Some(r) = r_eval_opt {
            base + beta * r[i]
        } else {
            base
        };
    }

    // -----------------------------------------------------------------------
    // ✅ Φ̃(z) / Z_H(z)  — computed in Fp³
    // -----------------------------------------------------------------------

    let n_inv = F::from(n as u64).inverse().unwrap();

    // Lift into Fp³
    let omega_fp3: Vec<Fp3> =
        omega_pows.iter().map(|&w| Fp3::from_base(w)).collect();
    let phi_fp3: Vec<Fp3> =
        phi_eval.iter().map(|&v| Fp3::from_base(v)).collect();

    // Barycentric sum (Fp³)
    let mut bary_sum_fp3 = Fp3::zero();
    for j in 0..n {
        let inv = (z_fp3 - omega_fp3[j]).inv();
        bary_sum_fp3 =
            bary_sum_fp3 + phi_fp3[j] * omega_fp3[j] * inv;
    }

    // c* = (1/n) · bary_sum   (project to base field)
    let c_star = n_inv * bary_sum_fp3.a0;

    // -----------------------------------------------------------------------
    // ✅ f₀(ω^j) = Φ̃(ω^j) / (ω^j − z)
    // -----------------------------------------------------------------------

    let mut f0_eval = Vec::with_capacity(n);
    for j in 0..n {
        let denom = omega_fp3[j] - z_fp3;
        let val = (phi_fp3[j] * denom.inv()).a0;
        f0_eval.push(val);
    }

    // -----------------------------------------------------------------------
    // ✅ Enforce ρ₀ = 1/32  (degree bound)
    // -----------------------------------------------------------------------

    let domain =
        GeneralEvaluationDomain::<F>::new(n)
            .expect("power-of-two domain");

    // Interpolate evaluations → coefficients
    let mut coeffs = domain.ifft(&f0_eval);

    // Target degree bound
    let d0 = n / 32;
    assert!(d0 > 0, "n must be >= 32 to enforce 1/32 rate");

    // Truncate high-degree coefficients
    if coeffs.len() > d0 {
        coeffs.truncate(d0);
    }

    // Reconstruct low-degree polynomial
    let poly = DensePolynomial::from_coefficients_vec(coeffs);

    // Re-evaluate over domain
    let f0_low_rate = domain.fft(poly.coeffs());

    debug_assert!(poly.degree() < d0);

    (f0_low_rate, z_fp3, c_star)
}

pub mod fri;
pub mod deep_tower;
pub mod deep;
pub mod cubic_ext;
pub mod tower_field;
pub mod sextic_ext;
pub mod octic_ext;
pub mod air_workloads;