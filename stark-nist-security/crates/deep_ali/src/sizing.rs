// src/sizing.rs

/// Bits from r queries given eps_eff: λ = r * log2(1/(1 - eps_eff))
#[inline]
pub fn bits_from_r(eps_eff: f64, r: usize) -> f64 {
    let pe = 1.0_f64 - eps_eff.clamp(1e-12, 1.0 - 1e-12);
    (1.0 / pe).log2() * (r as f64)
}

/// Minimal r to reach target bits with single-instance soundness.
#[inline]
pub fn r_for_bits(eps_eff: f64, bits: f64) -> usize {
    let per_query_bits = (1.0 / (1.0 - eps_eff.clamp(1e-12, 1.0 - 1e-12))).log2();
    ((bits / per_query_bits).ceil() as usize).max(1)
}

/// Path A (preferred if you have per-schedule λ at r0 queries):
/// Given λ_s at r0 (e.g., r0 = 32), compute eps_eff(s) and r for target bits.
/// eps_eff = 1 - 2^(-λ_s / r0), r_128 = ceil(128 / (λ_s / r0)) = ceil(128 * r0 / λ_s).
#[inline]
pub fn eps_eff_from_lambda(lambda_bits_at_r0: f64, r0: usize) -> f64 {
    let per_query_bits = lambda_bits_at_r0 / (r0 as f64);
    let one_minus = 2f64.powf(-per_query_bits);
    1.0 - one_minus
}

#[inline]
pub fn r_for_bits_from_lambda(lambda_bits_at_r0: f64, r0: usize, bits: f64) -> usize {
    let per_query_bits = lambda_bits_at_r0 / (r0 as f64);
    ((bits / per_query_bits).ceil() as usize).max(1)
}

/// Path B (baseline-calibrated constant eps_eff across schedules in the same regime):
/// Use the paper’s baseline eps_eff (e.g., 0.96 for [16,16,8]) to size r.
#[inline]
pub fn r_for_bits_baseline(eps_eff_baseline: f64, bits: f64) -> usize {
    r_for_bits(eps_eff_baseline, bits)
}