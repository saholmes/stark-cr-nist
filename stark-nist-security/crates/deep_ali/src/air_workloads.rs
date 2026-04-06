//! air_workloads.rs
//!
//! Three AIR workloads for benchmarking DEEP-ALI + MF-FRI across
//! varying trace widths and constraint structures.
//!
//!   AIR                  | w   | constraints | degree | blowup
//!   ---------------------|-----|-------------|--------|-------
//!   Fibonacci            |  2  |     1       |   2    |   4
//!   Poseidon hash chain  | 16  |    16       |   2    |   4
//!   Register machine     |  8  |     8       |   2    |   4
//!
//! All AIRs produce genuine execution traces that satisfy their
//! transition constraints, so the composition quotient polynomial
//! is well-defined and low-degree.

use ark_ff::{Field, Zero, One, UniformRand};
use ark_goldilocks::Goldilocks as F;
use rand::{rngs::StdRng, SeedableRng};

// ═══════════════════════════════════════════════════════════════════
//  AIR type enumeration
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AirType {
    /// Fibonacci recurrence  f(i+2) = f(i+1) + f(i).
    /// w = 2 trace columns, 1 degree-2 transition constraint.
    Fibonacci,

    /// Poseidon-like hash chain with state width t = 4.
    /// S-box x^7 decomposed: sq = x², cu = x³, fo = x⁴
    ///   → sbox_out = fo · cu = x⁷  (each step is degree 2).
    /// w = 16 columns  (4 state + 4 sq + 4 cu + 4 fo).
    /// 16 degree-2 transition constraints.
    PoseidonChain,

    /// Eight-register arithmetic machine with cross-coupled
    /// bilinear (degree-2) transition constraints.
    /// w = 8 columns, 8 degree-2 transition constraints.
    RegisterMachine,
}

impl AirType {
    /// Short label for CSV / filenames.
    pub fn label(self) -> &'static str {
        match self {
            AirType::Fibonacci      => "fib_w2_d2",
            AirType::PoseidonChain  => "poseidon_w16_d2",
            AirType::RegisterMachine => "regmach_w8_d2",
        }
    }

    /// Number of trace columns.
    pub fn width(self) -> usize {
        match self {
            AirType::Fibonacci      => 2,
            AirType::PoseidonChain  => 16,
            AirType::RegisterMachine => 8,
        }
    }

    /// Maximum individual constraint degree.
    pub fn max_constraint_degree(self) -> usize {
        // All three are degree-2 after S-box decomposition.
        2
    }

    /// Number of transition constraints.
    pub fn num_constraints(self) -> usize {
        match self {
            AirType::Fibonacci      => 1,
            AirType::PoseidonChain  => 16,
            AirType::RegisterMachine => 8,
        }
    }

    /// Convenience: all defined workloads.
    pub fn all() -> &'static [AirType] {
        &[
            AirType::Fibonacci,
            AirType::PoseidonChain,
            AirType::RegisterMachine,
        ]
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Top-level dispatcher
// ═══════════════════════════════════════════════════════════════════

/// Build a raw execution trace (w columns × n_trace rows) for the
/// given AIR.  Every row genuinely satisfies the transition
/// constraints so that the composition quotient is low-degree.
pub fn build_execution_trace(air: AirType, n_trace: usize) -> Vec<Vec<F>> {
    assert!(n_trace >= 2, "trace must have at least 2 rows");
    match air {
        AirType::Fibonacci       => build_fibonacci_trace(n_trace),
        AirType::PoseidonChain   => build_poseidon_chain_trace(n_trace),
        AirType::RegisterMachine => build_register_machine_trace(n_trace),
    }
}

/// Evaluate the transition constraints for AIR type `air` given
/// the current row values `cur` and the next row values `nxt`.
/// Returns a vector of length `air.num_constraints()`.
/// On a valid trace every entry is zero.
pub fn evaluate_constraints(
    air: AirType,
    cur: &[F],
    nxt: &[F],
    // Poseidon needs round constants per row; pass row index
    row: usize,
) -> Vec<F> {
    match air {
        AirType::Fibonacci       => eval_fibonacci_constraints(cur, nxt),
        AirType::PoseidonChain   => eval_poseidon_constraints(cur, nxt, row),
        AirType::RegisterMachine => eval_register_constraints(cur, nxt),
    }
}

// ═══════════════════════════════════════════════════════════════════
//  AIR 1 — Fibonacci  (w = 2)
// ═══════════════════════════════════════════════════════════════════

fn build_fibonacci_trace(n: usize) -> Vec<Vec<F>> {
    let mut c0 = vec![F::zero(); n];
    let mut c1 = vec![F::zero(); n];
    c0[0] = F::one();
    c1[0] = F::one();
    for i in 0..n - 1 {
        // transition: c0' = c1,  c1' = c0 + c1
        let next_c0 = c1[i];
        let next_c1 = c0[i] + c1[i];
        if i + 1 < n {
            c0[i + 1] = next_c0;
            c1[i + 1] = next_c1;
        }
    }
    vec![c0, c1]
}

fn eval_fibonacci_constraints(cur: &[F], nxt: &[F]) -> Vec<F> {
    // constraint:  nxt[1] - cur[0] - cur[1] = 0
    vec![nxt[1] - cur[0] - cur[1]]
}

// ═══════════════════════════════════════════════════════════════════
//  AIR 2 — Poseidon-like hash chain  (w = 16)
// ═══════════════════════════════════════════════════════════════════
//
//  State width t = 4.
//  Columns layout:
//     [0..4)   state   s_j
//     [4..8)   sq_j  = (s_j + rc_j)²
//     [8..12)  cu_j  = sq_j · (s_j + rc_j)     = (s_j + rc_j)³
//     [12..16) fo_j  = sq_j²                    = (s_j + rc_j)⁴
//
//  sbox_out_j = fo_j · cu_j  = (s_j + rc_j)⁷
//
//  Transition constraints (all degree ≤ 2):
//    C_{4+j}:  sq_j  - (s_j + rc_j)²                     = 0
//    C_{8+j}:  cu_j  - sq_j · (s_j + rc_j)               = 0
//    C_{12+j}: fo_j  - sq_j²                              = 0
//    C_j:      s_j'  - Σ_k mds[j][k] · (fo_k · cu_k)     = 0
//
//  Round constants are derived deterministically from a fixed seed.

/// Deterministic round constants.  Cached via a simple closure;
/// the benchmark calls `build_execution_trace` which generates them
/// inline. For constraint evaluation we regenerate from the same seed.
fn poseidon_round_constant(row: usize, col: usize) -> F {
    // Fast deterministic derivation — not cryptographically strong,
    // but sufficient for a benchmark trace.
    let seed = (row as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(col as u64)
        .wrapping_mul(0x6C62_272E_07BB_0142);
    F::from(seed)
}

fn build_mds_4x4() -> [[F; 4]; 4] {
    // Cauchy matrix:  M[i][j] = 1 / (x_i + y_j)
    // with x_i = i+1, y_j = t+j+1,  t = 4.
    let mut m = [[F::zero(); 4]; 4];
    for i in 0..4u64 {
        for j in 0..4u64 {
            let denom = F::from(i + 1) + F::from(4 + j + 1);
            m[i as usize][j as usize] =
                denom.inverse().expect("Cauchy denominator is nonzero");
        }
    }
    m
}

fn build_poseidon_chain_trace(n: usize) -> Vec<Vec<F>> {
    let t = 4usize;
    let w = 4 * t; // 16
    let mut trace = vec![vec![F::zero(); n]; w];
    let mds = build_mds_4x4();

    let mut state: [F; 4] = [
        F::from(1u64),
        F::from(2u64),
        F::from(3u64),
        F::from(4u64),
    ];

    for row in 0..n {
        // ---- write state columns 0..4 ----
        for j in 0..t {
            trace[j][row] = state[j];
        }

        // ---- S-box decomposition ----
        let mut sbox_out = [F::zero(); 4];
        for j in 0..t {
            let rc = poseidon_round_constant(row, j);
            let s  = state[j] + rc;
            let sq = s * s;        // s²
            let cu = sq * s;       // s³
            let fo = sq * sq;      // s⁴
            sbox_out[j] = fo * cu; // s⁷

            trace[t     + j][row] = sq; // cols  4..8
            trace[2 * t + j][row] = cu; // cols  8..12
            trace[3 * t + j][row] = fo; // cols 12..16
        }

        // ---- MDS → next state ----
        if row + 1 < n {
            for j in 0..t {
                let mut acc = F::zero();
                for k in 0..t {
                    acc += mds[j][k] * sbox_out[k];
                }
                state[j] = acc;
            }
        }
    }

    trace
}

fn eval_poseidon_constraints(cur: &[F], nxt: &[F], row: usize) -> Vec<F> {
    let t = 4usize;
    let mds = build_mds_4x4(); // cheap for t = 4
    let mut out = vec![F::zero(); 16];

    // ---- auxiliary column constraints ----
    for j in 0..t {
        let rc = poseidon_round_constant(row, j);
        let s  = cur[j] + rc;         // state + round constant
        let sq = cur[t + j];          // sq column
        let cu = cur[2 * t + j];      // cu column
        let fo = cur[3 * t + j];      // fo column

        out[t     + j] = sq - s * s;         // sq  = s²
        out[2 * t + j] = cu - sq * s;        // cu  = s³
        out[3 * t + j] = fo - sq * sq;       // fo  = s⁴
    }

    // ---- state transition constraints ----
    for j in 0..t {
        let mut expected = F::zero();
        for k in 0..t {
            let fo = cur[3 * t + k];
            let cu = cur[2 * t + k];
            expected += mds[j][k] * fo * cu; // fo · cu = s⁷
        }
        out[j] = nxt[j] - expected;
    }

    out
}

// ═══════════════════════════════════════════════════════════════════
//  AIR 3 — Eight-register arithmetic machine  (w = 8)
// ═══════════════════════════════════════════════════════════════════
//
//  Transitions (all degree-2, bilinear cross-coupling):
//    r0' = r0 · r1 + r2
//    r1' = r1 · r2 + r3
//    r2' = r2 · r3 + r4
//    r3' = r3 · r4 + r5
//    r4' = r4 · r5 + r6
//    r5' = r5 · r6 + r7
//    r6' = r6 · r7 + r0
//    r7' = r0 · r4 + r1 · r5 + r2 · r6 + r3 · r7
//
//  The last constraint couples all 8 registers via an inner-product
//  structure, making the constraint system non-separable.

fn build_register_machine_trace(n: usize) -> Vec<Vec<F>> {
    let w = 8usize;
    let mut trace = vec![vec![F::zero(); n]; w];

    let mut r: [F; 8] = core::array::from_fn(|i| F::from((i + 1) as u64));

    for row in 0..n {
        for j in 0..w {
            trace[j][row] = r[j];
        }
        if row + 1 < n {
            let p = r; // snapshot
            r[0] = p[0] * p[1] + p[2];
            r[1] = p[1] * p[2] + p[3];
            r[2] = p[2] * p[3] + p[4];
            r[3] = p[3] * p[4] + p[5];
            r[4] = p[4] * p[5] + p[6];
            r[5] = p[5] * p[6] + p[7];
            r[6] = p[6] * p[7] + p[0];
            r[7] = p[0] * p[4] + p[1] * p[5] + p[2] * p[6] + p[3] * p[7];
        }
    }

    trace
}

fn eval_register_constraints(cur: &[F], nxt: &[F]) -> Vec<F> {
    let r = cur;
    vec![
        nxt[0] - (r[0] * r[1] + r[2]),
        nxt[1] - (r[1] * r[2] + r[3]),
        nxt[2] - (r[2] * r[3] + r[4]),
        nxt[3] - (r[3] * r[4] + r[5]),
        nxt[4] - (r[4] * r[5] + r[6]),
        nxt[5] - (r[5] * r[6] + r[7]),
        nxt[6] - (r[6] * r[7] + r[0]),
        nxt[7] - (r[0] * r[4] + r[1] * r[5] + r[2] * r[6] + r[3] * r[7]),
    ]
}

// ═══════════════════════════════════════════════════════════════════
//  Sanity check (debug builds / tests)
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn verify_trace(air: AirType, n: usize) {
        let trace = build_execution_trace(air, n);
        assert_eq!(trace.len(), air.width());
        for col in &trace {
            assert_eq!(col.len(), n);
        }
        // Check constraints on interior rows
        for row in 0..n - 1 {
            let cur: Vec<F> = trace.iter().map(|c| c[row]).collect();
            let nxt: Vec<F> = trace.iter().map(|c| c[row + 1]).collect();
            let cv = evaluate_constraints(air, &cur, &nxt, row);
            for (ci, val) in cv.iter().enumerate() {
                assert!(
                    val.is_zero(),
                    "AIR {:?}  row {}  constraint {} != 0",
                    air, row, ci
                );
            }
        }
    }

    #[test]
    fn fibonacci_trace_valid()   { verify_trace(AirType::Fibonacci, 1024); }

    #[test]
    fn poseidon_trace_valid()    { verify_trace(AirType::PoseidonChain, 1024); }

    #[test]
    fn register_trace_valid()    { verify_trace(AirType::RegisterMachine, 1024); }
}
