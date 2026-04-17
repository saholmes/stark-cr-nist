
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
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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

// ═══════════════════════════════════════════════════════════════════
//  Polynomial division by Z_H(X) = X^m − 1
// ═══════════════════════════════════════════════════════════════════

/// Exact polynomial division of `dividend` by Z_H(X) = X^m − 1.
///
/// Given Φ̃(X) with coefficients `dividend`, computes c(X) such that
///   Φ̃(X) = c(X) · (X^m − 1)
///
/// When the execution trace satisfies the AIR constraints, Φ̃ vanishes
/// on the trace domain H = {ω^{blowup·i}}, so Z_H | Φ̃ and the
/// division is exact.  Debug builds verify the remainder is zero.
///
/// Algorithm:  from the identity  dividend[k] = c[k−m] − c[k],
/// solve top-down:  c[k−m] = dividend[k] + c[k].
fn poly_div_zh(dividend: &[F], m: usize) -> Vec<F> {
    let n = dividend.len();

    // If deg(Φ̃) < m, the quotient is the zero polynomial.
    // (Only valid when Φ̃ is itself zero.)
    if n <= m {
        #[cfg(debug_assertions)]
        for (i, &c) in dividend.iter().enumerate() {
            debug_assert!(
                c.is_zero(),
                "poly_div_zh: Φ̃ has degree < m={} but coeff[{}] is nonzero — \
                 constraints are not satisfied on the trace domain",
                m, i,
            );
        }
        return vec![F::zero()];
    }

    let q_len = n - m;
    let mut q = vec![F::zero(); q_len];

    // Top-down solve: for k = n−1, n−2, …, m
    //   c[k − m] = dividend[k] + c[k]
    // where c[k] = 0 for k ≥ q_len (beyond the quotient's degree).
    for k in (m..n).rev() {
        let qk = if k < q_len { q[k] } else { F::zero() };
        q[k - m] = dividend[k] + qk;
    }

    // ── Verify remainder is zero (debug only) ──
    //
    // For k = 0..m−1:  remainder[k] = dividend[k] + c[k]  (since c[k−m]
    // doesn't exist, i.e. k−m < 0).  Must be zero for exact division.
    #[cfg(debug_assertions)]
    {
        for k in 0..m.min(n) {
            let qk = if k < q_len { q[k] } else { F::zero() };
            let remainder = dividend[k] + qk;
            debug_assert!(
                remainder.is_zero(),
                "poly_div_zh: nonzero remainder at coeff index {} \
                 (remainder = {:?}) — constraints not satisfied on H",
                k, remainder,
            );
        }
    }

    q
}

// ═══════════════════════════════════════════════════════════════════
//  DEEP-ALI constraint quotient
// ═══════════════════════════════════════════════════════════════════
//
//  Previous code (BUGGY — removed):
//    1. Computed  f₀(ωʲ) = Φ̃(ωʲ) / (ωʲ − z)     ← Bug 1: missing eval subtraction
//    2. IFFT + truncate to degree d₀; re-FFT        ← Bug 2: nullifies FRI soundness
//    3. Returned  bary_sum_fp3.a0 / n  as c*         ← Bug 3: discards extension structure
//
//  Fixed code:
//    Computes  c(X) = Φ̃(X) / Z_H(X)  via exact polynomial division.
//    Returns evaluations of c on the FRI domain.  The DEEP quotient
//    (proximity testing) is handled inside the FRI subsystem, which
//    already correctly subtracts the claimed evaluation and divides
//    by (X − z) in the extension field.

/// Compute the constraint quotient  c(X) = Φ̃(X) / Z_H(X)  and return
/// its evaluations on the FRI domain of size `n = a_eval.len()`.
///
/// # Arguments
///
/// * `a_eval, s_eval, e_eval, t_eval` — evaluations of four trace
///   polynomials on the FRI domain (LDE of the execution trace).
///   The constraint composition is Φ̃(X) = a(X)·s(X) + e(X) − t(X).
///
/// * `omega` — primitive n-th root of unity generating the FRI domain.
///
/// * `n_trace` — size of the execution-trace domain H  (= n / blowup).
///   Z_H(X) = X^{n_trace} − 1 is the vanishing polynomial of H.
///
/// # Returns
///
/// `Vec<F>` of length `n`: evaluations of c(X) on {ω⁰, ω¹, …, ω^{n−1}}.
///
/// # Soundness
///
/// If the trace satisfies the AIR constraints, Φ̃ vanishes on H and
/// c(X) is a polynomial of degree ≤ deg(Φ̃) − n_trace.  The FRI
/// subsystem then verifies c is low-degree via its own DEEP quotient
/// at a random extension-field challenge z′ ∈ F_{p^d}, giving
/// soundness error ≤ deg(c) / |F_{p^d}|.
///
/// If the trace does NOT satisfy the constraints, Z_H ∤ Φ̃ and the
/// prover cannot produce a valid low-degree commitment, so FRI rejects.
pub fn deep_ali_merge_evals(
    a_eval: &[F],
    s_eval: &[F],
    e_eval: &[F],
    t_eval: &[F],
    omega: F,
    n_trace: usize,
) -> Vec<F> {
    deep_ali_merge_evals_blinded(
        a_eval,
        s_eval,
        e_eval,
        t_eval,
        None,
        F::zero(),
        omega,
        n_trace,
    )
}

/// Same as [`deep_ali_merge_evals`] but with an optional blinding
/// polynomial:  Φ̃(X) = a·s + e − t + β·r(X).
pub fn deep_ali_merge_evals_blinded(
    a_eval: &[F],
    s_eval: &[F],
    e_eval: &[F],
    t_eval: &[F],
    r_eval_opt: Option<&[F]>,
    beta: F,
    omega: F,
    n_trace: usize,
) -> Vec<F> {
    let n = a_eval.len();
    assert!(n > 1);
    assert!(n.is_power_of_two(), "FRI domain size must be a power of two");
    assert!(n_trace > 0, "trace domain must be nonempty");
    assert!(n_trace < n, "blowup factor must be ≥ 2");
    assert!(
        n % n_trace == 0,
        "trace domain size ({}) must divide FRI domain size ({})",
        n_trace, n,
    );

    assert_eq!(s_eval.len(), n);
    assert_eq!(e_eval.len(), n);
    assert_eq!(t_eval.len(), n);
    if let Some(r_eval) = r_eval_opt {
        assert_eq!(r_eval.len(), n);
    }

    // ── Step 1: Φ̃(ωʲ) — constraint composition evaluated pointwise ──
    //
    // On H this is zero (when constraints are satisfied); on the
    // complement of H it is generically nonzero.
    let mut phi_eval = vec![F::zero(); n];
    for i in 0..n {
        let base = a_eval[i] * s_eval[i] + e_eval[i] - t_eval[i];
        phi_eval[i] = if let Some(r) = r_eval_opt {
            base + beta * r[i]
        } else {
            base
        };
    }

    // ── Step 2: IFFT → Φ̃(X) coefficient representation ──
    let domain =
        GeneralEvaluationDomain::<F>::new(n).expect("power-of-two domain");
    let phi_coeffs = domain.ifft(&phi_eval);

    // ── Step 3: c(X) = Φ̃(X) / Z_H(X)  via polynomial long division ──
    //
    //   Z_H(X) = X^{n_trace} − 1   (vanishing polynomial of the trace domain)
    //
    //   If the constraints hold, this division is exact and
    //   deg(c) = deg(Φ̃) − n_trace  <  n − n_trace  <  n,
    //   so the rate  ρ = deg(c)/n  <  1 − (n_trace/n)  =  1 − 1/blowup.
    let c_coeffs = poly_div_zh(&phi_coeffs, n_trace);

    // ── Step 4: FFT c(X) → evaluations on the FRI domain ──
    //
    //   Since deg(c) < n, evaluating on an n-point radix-2 domain
    //   is exact (no aliasing).
    let mut padded = c_coeffs;
    padded.resize(n, F::zero());
    domain.fft(&padded)
}

pub mod fri;
pub mod deep_tower;
pub mod deep;
pub mod cubic_ext;
pub mod tower_field;
pub mod sextic_ext;
pub mod octic_ext;
pub mod air_workloads;