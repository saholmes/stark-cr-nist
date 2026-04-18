// src/sizing.rs
// src/sizing.rs

use crate::ProximityGapBound;

/// Bits from r queries given per-query error: λ = r * log2(1/(1 - δ))
/// where δ is the proximity gap.
#[inline]
pub fn bits_from_r(per_query_error: f64, r: usize) -> f64 {
    let pe = per_query_error.clamp(1e-12, 1.0 - 1e-12);
    -(pe.log2()) * (r as f64)
}

/// Minimal r to reach target bits of FRI soundness.
#[inline]
pub fn r_for_bits(per_query_error: f64, bits: f64) -> usize {
    let pe = per_query_error.clamp(1e-12, 1.0 - 1e-12);
    let bits_per_query = -(pe.log2());
    if bits_per_query <= 0.0 {
        return usize::MAX;
    }
    ((bits / bits_per_query).ceil() as usize).max(1)
}

/// Compute r from the actual protocol parameters.
///
/// This is the KEY function that connects the proximity gap bound
/// choice to the number of queries.
///
/// `rho`:         code rate = polynomial_degree / domain_size.
/// `bound`:       which proximity gap lower bound to use.
/// `target_bits`: target security level (e.g. 128.0).
/// `pow_bits`:    bits contributed by proof-of-work grinding.
///
/// Returns `(r, bits_per_query, delta)`.
pub fn r_from_protocol_params(
    rho: f64,
    bound: ProximityGapBound,
    target_bits: f64,
    pow_bits: f64,
) -> (usize, f64, f64) {
    let delta = bound.delta(rho);
    let per_query_err = bound.per_query_error(rho);
    let bits_per_query = bound.bits_per_query(rho);
    let fri_target = target_bits - pow_bits;
    let r = if bits_per_query > 0.0 {
        (fri_target / bits_per_query).ceil() as usize
    } else {
        usize::MAX
    };
    (r.max(1), bits_per_query, delta)
}

/// Path A: Given λ_s at r0 queries, compute eps_eff and r for target bits.
#[inline]
pub fn eps_eff_from_lambda(lambda_bits_at_r0: f64, r0: usize) -> f64 {
    let per_query_bits = lambda_bits_at_r0 / (r0 as f64);
    let one_minus = 2f64.powf(-per_query_bits);
    1.0 - one_minus
}

#[inline]
pub fn r_for_bits_from_lambda(lambda_bits_at_r0: f64, r0: usize, bits: f64) -> usize {
    let per_query_bits = lambda_bits_at_r0 / (r0 as f64);
    if per_query_bits <= 0.0 {
        return usize::MAX;
    }
    ((bits / per_query_bits).ceil() as usize).max(1)
}

/// Path B: baseline-calibrated constant eps_eff.
#[inline]
pub fn r_for_bits_baseline(eps_eff_baseline: f64, bits: f64) -> usize {
    let per_query_error = 1.0 - eps_eff_baseline;
    r_for_bits(per_query_error, bits)
}

/// Full sizing report: given AIR + FRI parameters, compute everything.
pub struct SizingReport {
    pub proximity_bound: ProximityGapBound,
    pub rho: f64,
    pub delta: f64,
    pub bits_per_query: f64,
    pub r: usize,
    pub fri_bits: f64,
    pub ali_bits: f64,
    pub pow_bits: f64,
    pub total_bits: f64,
    pub bottleneck: String,
}

pub fn full_sizing(
    rho: f64,
    bound: ProximityGapBound,
    target_bits: f64,
    pow_bits: f64,
    ext_field_log_size: f64,
    num_constraints: usize,
    max_constraint_deg: usize,
    trace_width: usize,
) -> SizingReport {
    let (r, bits_per_query, delta) =
        r_from_protocol_params(rho, bound, target_bits, pow_bits);
    let fri_bits = bits_per_query * (r as f64);

    // ALI bits
    let numerator_log2 = (num_constraints as f64
        * max_constraint_deg as f64
        * (trace_width + 1) as f64)
        .log2();
    let ali_bits = ext_field_log_size - numerator_log2;

    let total_bits = ali_bits.min(fri_bits + pow_bits);
    let bottleneck = if ali_bits <= fri_bits + pow_bits {
        "ALI reduction".to_string()
    } else {
        "FRI queries".to_string()
    };

    SizingReport {
        proximity_bound: bound,
        rho,
        delta,
        bits_per_query,
        r,
        fri_bits,
        ali_bits,
        pow_bits,
        total_bits,
        bottleneck,
    }
}