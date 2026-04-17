// src/trace_import.rs

use ark_goldilocks::Goldilocks as F;
use ark_ff::Zero;
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain as Domain};

/// Four evaluation vectors over the FRI domain, derived from a real
/// execution trace rather than random sampling.
///
/// Each vector contains evaluations of a polynomial with degree < n0/rate_inv
/// on an n0-point domain.  This gives FRI the same algebraic structure
/// (bounded-degree polynomials evaluated on a larger domain) as a real STARK,
/// which random vectors do NOT have — random vectors are full-rank, so they
/// don't exercise the degree-testing logic that FRI actually performs.
pub struct RealTraceInputs {
    pub a_eval: Vec<F>,
    pub s_eval: Vec<F>,
    pub e_eval: Vec<F>,
    pub t_eval: Vec<F>,
}

/// Build 4 Fibonacci trace columns over Goldilocks, interpolate, and
/// LDE-evaluate on an n0-point domain.
///
/// `n0`:       FRI domain size (power of 2)
/// `rate_inv`: blowup factor (2, 4, 8, 16…).  Polynomials have degree < n0/rate_inv.
pub fn real_trace_inputs(n0: usize, rate_inv: usize) -> RealTraceInputs {
    assert!(n0.is_power_of_two());
    assert!(rate_inv >= 2 && rate_inv.is_power_of_two());
    let trace_len = n0 / rate_inv;
    assert!(trace_len >= 2, "trace too short");

    // Four Fibonacci-like columns with different seeds.
    // Each column satisfies col[i] = col[i-1] + col[i-2],
    // so it's a valid execution trace of a degree-1 transition constraint.
    let seeds: [(u64, u64); 4] = [
        (1, 1),
        (2, 3),
        (5, 8),
        (13, 21),
    ];

    let trace_dom = Domain::<F>::new(trace_len).unwrap();
    let lde_dom   = Domain::<F>::new(n0).unwrap();

    let mut evals = Vec::with_capacity(4);

    for &(s0, s1) in &seeds {
        // 1. Build trace column
        let mut col = Vec::with_capacity(trace_len);
        col.push(F::from(s0));
        col.push(F::from(s1));
        for i in 2..trace_len {
            col.push(col[i - 1] + col[i - 2]);
        }

        // 2. Interpolate: IFFT over trace domain → coefficients
        //    Polynomial has degree trace_len - 1 = n0/rate_inv - 1
        let coeffs = trace_dom.ifft(&col);

        // 3. LDE: pad coefficients to n0 (zeros for high degrees),
        //    then FFT over the larger domain
        let mut padded = coeffs;
        padded.resize(n0, F::zero());
        evals.push(lde_dom.fft(&padded));
    }

    RealTraceInputs {
        a_eval: evals.remove(0),
        s_eval: evals.remove(0),
        e_eval: evals.remove(0),
        t_eval: evals.remove(0),
    }
}

/// Convert an arbitrary execution trace (produced by an AIR workload)
/// into `RealTraceInputs` by interpolating each column and LDE-evaluating
/// on the extended domain.
///
/// `trace_columns`: each inner Vec is one column of length `n0 / blowup`.
/// `n0`:            FRI / extended-evaluation domain size (power of 2).
/// `blowup`:        rate inverse (typically 4).
///
/// The function maps the first four columns to `a_eval … t_eval`.
/// If the trace has fewer than four columns, columns are reused with
/// wraparound (same strategy as `import_winterfell_trace`).
/// If the trace has more than four columns, the extra columns are ignored.
pub fn trace_inputs_from_air(
    trace_columns: Vec<Vec<F>>,
    n0: usize,
    blowup: usize,
) -> RealTraceInputs {
    let num_cols = trace_columns.len();
    assert!(num_cols >= 1, "need at least 1 trace column");
    assert!(n0.is_power_of_two());
    assert!(blowup >= 2 && blowup.is_power_of_two());

    let trace_len = n0 / blowup;
    assert!(trace_len >= 2, "trace too short");

    // Sanity-check that every column has the expected length
    for (i, col) in trace_columns.iter().enumerate() {
        assert_eq!(
            col.len(),
            trace_len,
            "column {} has length {} but expected {}",
            i,
            col.len(),
            trace_len
        );
    }

    let trace_dom = Domain::<F>::new(trace_len).unwrap();
    let lde_dom   = Domain::<F>::new(n0).unwrap();

    let lde = |col: &[F]| -> Vec<F> {
        let coeffs = trace_dom.ifft(col);
        let mut padded = coeffs;
        padded.resize(n0, F::zero());
        lde_dom.fft(&padded)
    };

    // Map columns to the four required vectors with wraparound
    let a_eval = lde(&trace_columns[0]);
    let s_eval = lde(&trace_columns[1 % num_cols]);
    let e_eval = lde(&trace_columns[2 % num_cols]);
    let t_eval = lde(&trace_columns[3 % num_cols]);

    RealTraceInputs { a_eval, s_eval, e_eval, t_eval }
}

/// Same as above but reads trace columns from a binary file exported
/// by Winterfell's FibSmall example (f64 = Goldilocks).
///
/// File format (produced by the export binary in Path B):
///   Header line:  "TRACE <trace_len> <num_cols> <field_bits>\n"
///   Body:         column-major, each element as u64 little-endian (8 bytes)
pub fn import_winterfell_trace(path: &str, n0: usize) -> RealTraceInputs {
    use std::io::{BufRead, BufReader, Read};
    use std::fs::File;

    let file = File::open(path).expect("cannot open trace file");
    let mut reader = BufReader::new(file);

    // Parse header
    let mut header = String::new();
    reader.read_line(&mut header).unwrap();
    let parts: Vec<&str> = header.trim().split_whitespace().collect();
    assert_eq!(parts[0], "TRACE");
    let trace_len: usize = parts[1].parse().unwrap();
    let num_cols: usize  = parts[2].parse().unwrap();
    assert!(num_cols >= 2, "need at least 2 trace columns");

    // Read columns (u64 LE → Goldilocks)
    let mut columns: Vec<Vec<F>> = Vec::with_capacity(num_cols);
    let mut buf = [0u8; 8];

    for _col in 0..num_cols {
        let mut column = Vec::with_capacity(trace_len);
        for _row in 0..trace_len {
            reader.read_exact(&mut buf).unwrap();
            let val = u64::from_le_bytes(buf);
            column.push(F::from(val));
        }
        columns.push(column);
    }

    // Interpolate and LDE, same as above
    let trace_dom = Domain::<F>::new(trace_len).unwrap();
    let lde_dom   = Domain::<F>::new(n0).unwrap();

    let lde = |col: &[F]| -> Vec<F> {
        let coeffs = trace_dom.ifft(col);
        let mut padded = coeffs;
        padded.resize(n0, F::zero());
        lde_dom.fft(&padded)
    };

    // Map columns to the four vectors.
    // With 2 trace columns we duplicate; with 4+ we use the first 4.
    let a_eval = lde(&columns[0]);
    let s_eval = lde(&columns[1 % num_cols]);
    let e_eval = lde(&columns[2 % num_cols]);
    let t_eval = lde(&columns[3 % num_cols]);

    RealTraceInputs { a_eval, s_eval, e_eval, t_eval }
}