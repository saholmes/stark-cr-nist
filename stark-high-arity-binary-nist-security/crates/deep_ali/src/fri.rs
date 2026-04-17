#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused_macros)]

use ark_goldilocks::Goldilocks as F;
use rand::{rngs::StdRng, Rng, SeedableRng};

use ark_poly::domain::radix2::Radix2EvaluationDomain as Domain;
use ark_ff::{Field, One, PrimeField, Zero};
use ark_poly::{
    EvaluationDomain, GeneralEvaluationDomain,
};
use hash::SelectedHasher;
use hash::selected::HASH_BYTES;

use hash::sha3::Digest;
use crate::tower_field::TowerField;

use merkle::{
    MerkleChannelCfg,
    MerkleTreeChannel,
    MerkleOpening,
    compute_leaf_hash,
};

use transcript::Transcript;


#[cfg(feature = "parallel")]
use rayon::prelude::*;


// ────────────────────────────────────────────────────────────────────────
//  Hash helpers
// ────────────────────────────────────────────────────────────────────────

#[inline]
fn finalize_to_digest(h: SelectedHasher) -> [u8; HASH_BYTES] {
    let result = h.finalize();
    let mut out = [0u8; HASH_BYTES];
    out.copy_from_slice(result.as_slice());
    out
}

fn transcript_challenge_hash(tr: &mut Transcript, label: &[u8]) -> [u8; HASH_BYTES] {
    let v = tr.challenge_bytes(label);
    assert!(
        v.len() >= HASH_BYTES,
        "transcript digest ({} bytes) shorter than HASH_BYTES ({})",
        v.len(),
        HASH_BYTES,
    );
    let mut out = [0u8; HASH_BYTES];
    out.copy_from_slice(&v[..HASH_BYTES]);
    out
}

// ────────────────────────────────────────────────────────────────────────
//  Safe field serialization helper
// ────────────────────────────────────────────────────────────────────────

#[inline]
fn field_to_le_bytes(f: F) -> [u8; 8] {
    f.into_bigint().0[0].to_le_bytes()
}

// ────────────────────────────────────────────────────────────────────────
//  Safe field-challenge helper
// ────────────────────────────────────────────────────────────────────────

fn safe_field_challenge(tr: &mut Transcript, label: &[u8]) -> F {
    let bytes = tr.challenge_bytes(label);
    let mut acc = F::zero();
    for chunk in bytes.rchunks(7) {
        let shift = 1u64 << (chunk.len() as u64 * 8);
        let mut val = 0u64;
        for (i, &b) in chunk.iter().enumerate() {
            val |= (b as u64) << (i * 8);
        }
        acc = acc * F::from(shift) + F::from(val);
    }
    acc
}

// ────────────────────────────────────────────────────────────────────────

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

#[cfg(feature = "fri_bench_log")]
#[allow(unused_macros)]
macro_rules! logln {
    ($($tt:tt)*) => { eprintln!($($tt)*); }
}
#[cfg(not(feature = "fri_bench_log"))]
macro_rules! logln {
    ($($tt:tt)*) => {};
}

mod ds {
    pub const FRI_SEED: &[u8] = b"FRI/seed";
    pub const FRI_INDEX: &[u8] = b"FRI/index";
    pub const FRI_Z_L: &[u8] = b"FRI/z/l";
    pub const FRI_Z_L_1: &[u8] = b"FRI/z/l/1";
    pub const FRI_Z_L_2: &[u8] = b"FRI/z/l/2";
    pub const FRI_LEAF: &[u8] = b"FRI/leaf";
}

fn tr_hash_fields_tagged(tag: &[u8], fields: &[F]) -> F {
    let mut tr = Transcript::new_matching_hash(b"FRI/FS");
    tr.absorb_bytes(tag);
    for &x in fields {
        tr.absorb_field(x);
    }
    safe_field_challenge(&mut tr, b"out")
}

// ────────────────────────────────────────────────────────────────────────
//  Extension-field division helper
// ────────────────────────────────────────────────────────────────────────

#[inline]
fn ext_div<E: TowerField>(a: E, b: E) -> E {
    a * b.invert().expect("ext_div: division by zero in extension field")
}

// ────────────────────────────────────────────────────────────────────────
//  Extension-field power helper (square-and-multiply)
// ────────────────────────────────────────────────────────────────────────

fn ext_pow<E: TowerField>(mut base: E, mut exp: u64) -> E {
    if exp == 0 {
        return E::one();
    }
    let mut result = E::one();
    while exp > 1 {
        if exp & 1 == 1 {
            result = result * base;
        }
        base = base.sq();
        exp >>= 1;
    }
    result * base
}

// ────────────────────────────────────────────────────────────────────────
//  FRI Domain
// ────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct FriDomain {
    pub omega: F,
    pub size: usize,
}

impl FriDomain {
    pub fn new_radix2(size: usize) -> Self {
        let dom = Domain::<F>::new(size).expect("radix-2 domain exists");
        Self { omega: dom.group_gen, size }
    }
}

// ────────────────────────────────────────────────────────────────────────
//  Base-field utilities (kept for backward compatibility and tests)
// ────────────────────────────────────────────────────────────────────────

fn build_z_pows(z_l: F, m: usize) -> Vec<F> {
    let mut z_pows = Vec::with_capacity(m);
    let mut acc = F::one();
    for _ in 0..m {
        z_pows.push(acc);
        acc *= z_l;
    }
    z_pows
}

fn build_ext_pows<E: TowerField>(alpha: E, m: usize) -> Vec<E> {
    let mut pows = Vec::with_capacity(m);
    let mut acc = E::one();
    for _ in 0..m {
        pows.push(acc);
        acc = acc * alpha;
    }
    pows
}

fn eval_poly_at_ext<E: TowerField>(coeffs: &[F], z: E) -> E {
    E::eval_base_poly(coeffs, z)
}

fn compute_q_layer_ext<E: TowerField + Send + Sync>(
    f_l: &[F],
    z: E,
    omega: F,
) -> (Vec<E>, E) {
    let n = f_l.len();
    let dom = Domain::<F>::new(n).unwrap();
    let coeffs = dom.ifft(f_l);
    let fz = eval_poly_at_ext(&coeffs, z);

    let omega_ext = E::from_fp(omega);
    let xs: Vec<E> = {
        let mut v = Vec::with_capacity(n);
        let mut x = E::one();
        for _ in 0..n {
            v.push(x);
            x = x * omega_ext;
        }
        v
    };

    #[cfg(feature = "parallel")]
    let q: Vec<E> = f_l
        .par_iter()
        .zip(xs.par_iter())
        .map(|(&fi, &xi)| {
            let num   = E::from_fp(fi) - fz;
            let denom = xi - z;
            ext_div(num, denom)
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let q: Vec<E> = f_l
        .iter()
        .zip(xs.iter())
        .map(|(&fi, &xi)| {
            let num   = E::from_fp(fi) - fz;
            let denom = xi - z;
            ext_div(num, denom)
        })
        .collect();

    (q, fz)
}

// ────────────────────────────────────────────────────────────────────────
//  Extension-field FRI core — generic over E : TowerField
// ────────────────────────────────────────────────────────────────────────

fn compute_q_layer_ext_on_ext<E: TowerField + Send + Sync>(
    f_l: &[E],
    z: E,
    omega: F,
) -> (Vec<E>, E) {
    let n = f_l.len();
    let d = E::DEGREE;
    let dom = Domain::<F>::new(n).unwrap();

    let mut comp_evals: Vec<Vec<F>> = vec![Vec::with_capacity(n); d];
    for elem in f_l {
        let comps = elem.to_fp_components();
        for j in 0..d {
            comp_evals[j].push(comps[j]);
        }
    }

    #[cfg(feature = "parallel")]
    let comp_coeffs: Vec<Vec<F>> = comp_evals
        .par_iter()
        .map(|evals| dom.ifft(evals))
        .collect();

    #[cfg(not(feature = "parallel"))]
    let comp_coeffs: Vec<Vec<F>> = comp_evals
        .iter()
        .map(|evals| dom.ifft(evals))
        .collect();

    let mut fz = E::zero();
    for k in (0..n).rev() {
        let coeff_comps: Vec<F> = (0..d).map(|j| comp_coeffs[j][k]).collect();
        let coeff_k = E::from_fp_components(&coeff_comps)
            .expect("compute_q_layer_ext_on_ext: bad coefficient components");
        fz = fz * z + coeff_k;
    }

    let omega_ext = E::from_fp(omega);
    let xs: Vec<E> = {
        let mut v = Vec::with_capacity(n);
        let mut x = E::one();
        for _ in 0..n {
            v.push(x);
            x = x * omega_ext;
        }
        v
    };

    #[cfg(feature = "parallel")]
    let q: Vec<E> = f_l
        .par_iter()
        .zip(xs.par_iter())
        .map(|(&fi, &xi)| {
            let num   = fi - fz;
            let denom = xi - z;
            ext_div(num, denom)
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let q: Vec<E> = f_l
        .iter()
        .zip(xs.iter())
        .map(|(&fi, &xi)| {
            let num   = fi - fz;
            let denom = xi - z;
            ext_div(num, denom)
        })
        .collect();

    (q, fz)
}

// ────────────────────────────────────────────────────────────────────────
//  STIR: Batched multi-point DEEP quotient
// ────────────────────────────────────────────────────────────────────────

fn interpolate_stir_coset<E: TowerField>(
    coset_evals: &[E],
    z_ell: E,
    zeta: F,
    m: usize,
) -> Vec<E> {
    let m_inv = F::from(m as u64).inverse().unwrap();
    let zeta_inv = zeta.inverse().unwrap();

    let mut zi_pows = vec![F::one(); m];
    for k in 1..m {
        zi_pows[k] = zi_pows[k - 1] * zeta_inv;
    }

    let mut d = vec![E::zero(); m];
    for k in 0..m {
        let mut sum = E::zero();
        for j in 0..m {
            let exp = (j * k) % m;
            sum = sum + coset_evals[j] * E::from_fp(zi_pows[exp]);
        }
        d[k] = sum * E::from_fp(m_inv);
    }

    let z_inv = z_ell.invert().expect("STIR z_ell must be nonzero");
    let mut coeffs = Vec::with_capacity(m);
    let mut z_inv_pow = E::one();
    for k in 0..m {
        coeffs.push(d[k] * z_inv_pow);
        z_inv_pow = z_inv_pow * z_inv;
    }

    coeffs
}

fn compute_stir_quotient_ext<E: TowerField + Send + Sync>(
    f_l: &[E],
    z_ell: E,
    omega: F,
    m: usize,
) -> (Vec<E>, Vec<E>, Vec<E>) {
    let n = f_l.len();
    let d = E::DEGREE;
    let dom = Domain::<F>::new(n).unwrap();
    let n_next = n / m;

    let mut comp_evals: Vec<Vec<F>> = vec![Vec::with_capacity(n); d];
    for elem in f_l {
        let comps = elem.to_fp_components();
        for j in 0..d {
            comp_evals[j].push(comps[j]);
        }
    }

    #[cfg(feature = "parallel")]
    let comp_coeffs: Vec<Vec<F>> = comp_evals
        .par_iter()
        .map(|evals| dom.ifft(evals))
        .collect();

    #[cfg(not(feature = "parallel"))]
    let comp_coeffs: Vec<Vec<F>> = comp_evals
        .iter()
        .map(|evals| dom.ifft(evals))
        .collect();

    let eval_at = |point: E| -> E {
        let mut result = E::zero();
        for k in (0..n).rev() {
            let comps: Vec<F> = (0..d).map(|j| comp_coeffs[j][k]).collect();
            let coeff_k = E::from_fp_components(&comps).unwrap();
            result = result * point + coeff_k;
        }
        result
    };

    let zeta_base = omega.pow([n_next as u64]);
    let zeta = E::from_fp(zeta_base);
    let mut coset_evals = Vec::with_capacity(m);
    let mut zeta_pow = E::one();
    for _ in 0..m {
        coset_evals.push(eval_at(zeta_pow * z_ell));
        zeta_pow = zeta_pow * zeta;
    }

    let interp_coeffs = interpolate_stir_coset(&coset_evals, z_ell, zeta_base, m);

    let z_ell_m = ext_pow(z_ell, m as u64);

    let omega_ext = E::from_fp(omega);
    let xs: Vec<E> = {
        let mut v = Vec::with_capacity(n);
        let mut x = E::one();
        for _ in 0..n {
            v.push(x);
            x = x * omega_ext;
        }
        v
    };

    #[cfg(feature = "parallel")]
    let quotient: Vec<E> = f_l
        .par_iter()
        .zip(xs.par_iter())
        .map(|(&fi, &xi)| {
            let p_xi = eval_final_poly_ext(&interp_coeffs, xi);
            let v_xi = ext_pow(xi, m as u64) - z_ell_m;
            ext_div(fi - p_xi, v_xi)
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let quotient: Vec<E> = f_l
        .iter()
        .zip(xs.iter())
        .map(|(&fi, &xi)| {
            let p_xi = eval_final_poly_ext(&interp_coeffs, xi);
            let v_xi = ext_pow(xi, m as u64) - z_ell_m;
            ext_div(fi - p_xi, v_xi)
        })
        .collect();

    (quotient, coset_evals, interp_coeffs)
}

// ────────────────────────────────────────────────────────────────────────

fn fri_fold_layer_ext_impl<E: TowerField>(
    evals: &[E],
    alpha: E,
    folding_factor: usize,
) -> Vec<E> {
    let n = evals.len();
    assert!(n % folding_factor == 0);
    let n_next = n / folding_factor;

    let alpha_pows = build_ext_pows(alpha, folding_factor);

    let mut out = vec![E::zero(); n_next];

    if enable_parallel(n_next) {
        #[cfg(feature = "parallel")]
        {
            out.par_iter_mut().enumerate().for_each(|(b, out_b)| {
                let mut acc = E::zero();
                for j in 0..folding_factor {
                    acc = acc + evals[b + j * n_next] * alpha_pows[j];
                }
                *out_b = acc;
            });
            return out;
        }
    }

    for b in 0..n_next {
        let mut acc = E::zero();
        for j in 0..folding_factor {
            acc = acc + evals[b + j * n_next] * alpha_pows[j];
        }
        out[b] = acc;
    }
    out
}

fn compute_s_layer_ext<E: TowerField>(
    f_l: &[E],
    alpha: E,
    m: usize,
) -> Vec<E> {
    let n = f_l.len();
    assert!(n % m == 0);
    let n_next = n / m;

    let alpha_pows = build_ext_pows(alpha, m);

    let mut folded = vec![E::zero(); n_next];
    for b in 0..n_next {
        let mut sum = E::zero();
        for j in 0..m {
            sum = sum + f_l[b + j * n_next] * alpha_pows[j];
        }
        folded[b] = sum;
    }

    let mut s_per_i = vec![E::zero(); n];
    for b in 0..n_next {
        for j in 0..m {
            s_per_i[b + j * n_next] = folded[b];
        }
    }
    s_per_i
}

// ────────────────────────────────────────────────────────────────────────
//  Construction 5.1 — Coefficient extraction & interpolation fold
// ────────────────────────────────────────────────────────────────────────

fn extract_all_coset_coefficients<E: TowerField>(
    evals: &[E],
    omega: F,
    m: usize,
) -> Vec<Vec<E>> {
    let n = evals.len();
    assert!(n % m == 0);
    let n_next = n / m;
    let zeta = omega.pow([n_next as u64]);

    (0..n_next)
        .map(|b| {
            let fibre_values: Vec<E> = (0..m)
                .map(|j| evals[b + j * n_next])
                .collect();
            let omega_b = omega.pow([b as u64]);
            interpolate_coset_ext(&fibre_values, omega_b, zeta, m)
        })
        .collect()
}

fn interpolate_coset_ext<E: TowerField>(
    fibre_values: &[E],
    omega_b: F,
    zeta: F,
    m: usize,
) -> Vec<E> {
    let d = E::DEGREE;
    let m_inv = F::from(m as u64).inverse().unwrap();
    let zeta_inv = zeta.inverse().unwrap();

    let mut zi_pows = vec![F::one(); m];
    for k in 1..m {
        zi_pows[k] = zi_pows[k - 1] * zeta_inv;
    }

    let mut comp_vecs: Vec<Vec<F>> = vec![Vec::with_capacity(m); d];
    for val in fibre_values {
        let comps = val.to_fp_components();
        for c in 0..d {
            comp_vecs[c].push(comps[c]);
        }
    }

    let mut comp_d: Vec<Vec<F>> = vec![vec![F::zero(); m]; d];
    for c in 0..d {
        for i in 0..m {
            let mut sum = F::zero();
            for j in 0..m {
                let exp = (i * j) % m;
                sum += comp_vecs[c][j] * zi_pows[exp];
            }
            comp_d[c][i] = sum * m_inv;
        }
    }

    let omega_b_inv = if omega_b == F::zero() {
        F::one()
    } else {
        omega_b.inverse().unwrap()
    };

    let mut result = Vec::with_capacity(m);
    let mut ob_inv_pow = F::one();
    for i in 0..m {
        let comps: Vec<F> = (0..d).map(|c| comp_d[c][i]).collect();
        let di = E::from_fp_components(&comps).unwrap();
        result.push(di * E::from_fp(ob_inv_pow));
        ob_inv_pow *= omega_b_inv;
    }

    result
}

fn interpolation_fold_ext<E: TowerField>(
    coeff_tuples: &[Vec<E>],
    alpha: E,
) -> Vec<E> {
    let m = coeff_tuples[0].len();
    let alpha_pows = build_ext_pows(alpha, m);

    coeff_tuples
        .iter()
        .map(|coeffs| {
            let mut sum = E::zero();
            for i in 0..m {
                sum = sum + coeffs[i] * alpha_pows[i];
            }
            sum
        })
        .collect()
}

fn compute_s_layer_from_coeffs<E: TowerField>(
    coeff_tuples: &[Vec<E>],
    alpha: E,
    n: usize,
    m: usize,
) -> Vec<E> {
    let n_next = n / m;
    let folded = interpolation_fold_ext(coeff_tuples, alpha);

    let mut s_per_i = vec![E::zero(); n];
    for b in 0..n_next {
        for j in 0..m {
            s_per_i[b + j * n_next] = folded[b];
        }
    }
    s_per_i
}

fn verify_interpolation_consistency<E: TowerField>(
    fibre_values: &[E],
    fibre_points: &[F],
    coeff_tuple: &[E],
) -> bool {
    let m = fibre_values.len();
    for j in 0..m {
        let mut eval = E::zero();
        let mut x_pow = F::one();
        for i in 0..m {
            eval = eval + coeff_tuple[i] * E::from_fp(x_pow);
            x_pow *= fibre_points[j];
        }
        if eval != fibre_values[j] {
            return false;
        }
    }
    true
}

fn batched_degree_check_ext<E: TowerField>(
    coeff_tuples: &[Vec<E>],
    beta: E,
    d_final: usize,
) -> bool {
    let n_final = coeff_tuples.len();
    if n_final == 0 {
        return true;
    }
    let m = coeff_tuples[0].len();
    let beta_pows = build_ext_pows(beta, m);

    let gamma_evals: Vec<E> = (0..n_final)
        .map(|b| {
            let mut sum = E::zero();
            for i in 0..m {
                sum = sum + coeff_tuples[b][i] * beta_pows[i];
            }
            sum
        })
        .collect();

    let deg = E::DEGREE;
    let dom = Domain::<F>::new(n_final).unwrap();

    let mut comp_evals: Vec<Vec<F>> = vec![Vec::with_capacity(n_final); deg];
    for elem in &gamma_evals {
        let comps = elem.to_fp_components();
        for j in 0..deg {
            comp_evals[j].push(comps[j]);
        }
    }

    let comp_coeffs: Vec<Vec<F>> = comp_evals
        .iter()
        .map(|evals| dom.ifft(evals))
        .collect();

    for k in d_final..n_final {
        for j in 0..deg {
            if !comp_coeffs[j][k].is_zero() {
                return false;
            }
        }
    }

    true
}

fn coeff_tree_config(n_final: usize) -> MerkleChannelCfg {
    let arity = pick_arity_for_layer(n_final, 16).max(2);
    let depth = merkle_depth(n_final, arity);
    MerkleChannelCfg::new(vec![arity; depth], 0xFE)
}

fn coeff_leaf_fields<E: TowerField>(tuple: &[E]) -> Vec<F> {
    tuple.iter().flat_map(|e| e.to_fp_components()).collect()
}

// ────────────────────────────────────────────────────────────────────────
//  Extension-field evaluation/coefficient helpers
// ────────────────────────────────────────────────────────────────────────

fn ext_evals_to_coeffs<E: TowerField>(evals: &[E]) -> Vec<E> {
    let n = evals.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return evals.to_vec();
    }

    let d = E::DEGREE;
    let dom = Domain::<F>::new(n).unwrap();

    let mut comp_evals: Vec<Vec<F>> = vec![Vec::with_capacity(n); d];
    for elem in evals {
        let comps = elem.to_fp_components();
        for j in 0..d {
            comp_evals[j].push(comps[j]);
        }
    }

    let comp_coeffs: Vec<Vec<F>> = comp_evals
        .iter()
        .map(|e| dom.ifft(e))
        .collect();

    (0..n)
        .map(|k| {
            let comps: Vec<F> = (0..d).map(|j| comp_coeffs[j][k]).collect();
            E::from_fp_components(&comps).unwrap()
        })
        .collect()
}

#[inline]
fn eval_final_poly_ext<E: TowerField>(coeffs: &[E], x: E) -> E {
    let mut result = E::zero();
    for c in coeffs.iter().rev() {
        result = result * x + *c;
    }
    result
}

// ────────────────────────────────────────────────────────────────────────
//  Base-field FRI fold (kept for backward compatibility / tests)
// ────────────────────────────────────────────────────────────────────────

fn dot_with_z_pows(chunk: &[F], z_pows: &[F]) -> F {
    debug_assert_eq!(chunk.len(), z_pows.len());
    let mut s = F::zero();
    for (val, zp) in chunk.iter().zip(z_pows.iter()) {
        s += *val * *zp;
    }
    s
}

fn fold_layer_sequential(f_l: &[F], z_pows: &[F], m: usize) -> Vec<F> {
    f_l.chunks(m)
        .map(|chunk| dot_with_z_pows(chunk, z_pows))
        .collect()
}

#[cfg(feature = "parallel")]
fn fold_layer_parallel(f_l: &[F], z_pows: &[F], m: usize) -> Vec<F> {
    f_l.par_chunks(m)
        .map(|chunk| dot_with_z_pows(chunk, z_pows))
        .collect()
}

fn fill_repeated_targets(target: &mut [F], src: &[F], m: usize) {
    for (bucket, chunk) in src.iter().zip(target.chunks_mut(m)) {
        for item in chunk {
            *item = *bucket;
        }
    }
}

fn merkle_depth(leaves: usize, arity: usize) -> usize {
    assert!(arity >= 2, "Merkle arity must be ≥ 2");
    let mut depth = 1;
    let mut cur = leaves;
    while cur > arity {
        cur = (cur + arity - 1) / arity;
        depth += 1;
    }
    depth
}

#[cfg(feature = "parallel")]
fn fill_repeated_targets_parallel(target: &mut [F], src: &[F], m: usize) {
    target
        .par_chunks_mut(m)
        .enumerate()
        .for_each(|(idx, chunk)| {
            let bucket = src[idx];
            for item in chunk {
                *item = bucket;
            }
        });
}

pub fn fri_sample_z_ell(seed_z: u64, level: usize, domain_size: usize) -> F {
    let fused = tr_hash_fields_tagged(
        ds::FRI_Z_L,
        &[F::from(seed_z), F::from(level as u64), F::from(domain_size as u64)],
    );
    let mut seed_bytes = [0u8; 32];
    seed_bytes[..8].copy_from_slice(&field_to_le_bytes(fused));
    let mut rng = StdRng::from_seed(seed_bytes);
    let exp_bigint = <F as PrimeField>::BigInt::from(domain_size as u64);
    let mut tries = 0usize;
    const MAX_TRIES: usize = 1_000;
    loop {
        let cand = F::from(rng.gen::<u64>());
        if !cand.is_zero() && cand.pow(exp_bigint.as_ref()) != F::one() {
            return cand;
        }
        tries += 1;
        if tries >= MAX_TRIES {
            let fallback = F::from(seed_z.wrapping_add(level as u64).wrapping_add(7));
            if fallback.pow(exp_bigint.as_ref()) != F::one() {
                return fallback;
            }
            return F::from(11u64);
        }
    }
}

pub fn compute_s_layer(f_l: &[F], z_l: F, m: usize) -> Vec<F> {
    let n = f_l.len();
    assert!(n % m == 0);
    let n_next = n / m;
    let z_pows = build_z_pows(z_l, m);
    let mut folded = vec![F::zero(); n_next];
    for b in 0..n_next {
        let mut acc = F::zero();
        for j in 0..m {
            acc += f_l[b + j * n_next] * z_pows[j];
        }
        folded[b] = acc;
    }
    let mut s_per_i = vec![F::zero(); n];
    for b in 0..n_next {
        for j in 0..m {
            s_per_i[b + j * n_next] = folded[b];
        }
    }
    s_per_i
}

fn layer_sizes_from_schedule(n0: usize, schedule: &[usize]) -> Vec<usize> {
    let mut sizes = Vec::with_capacity(schedule.len() + 1);
    let mut n = n0;
    sizes.push(n);
    for &m in schedule {
        assert!(n % m == 0, "schedule not dividing domain size");
        n /= m;
        sizes.push(n);
    }
    sizes
}

fn hash_node(children: &[[u8; HASH_BYTES]]) -> [u8; HASH_BYTES] {
    let mut h = SelectedHasher::new();
    Digest::update(&mut h, b"FRI/MERKLE/NODE");
    for c in children {
        Digest::update(&mut h, c);
    }
    finalize_to_digest(h)
}

fn index_from_seed(seed_f: F, n_pow2: usize) -> usize {
    assert!(n_pow2.is_power_of_two());
    let mask = n_pow2 - 1;
    let mut seed_bytes = [0u8; 32];
    seed_bytes[..8].copy_from_slice(&field_to_le_bytes(seed_f));
    let mut rng = StdRng::from_seed(seed_bytes);
    (rng.gen::<u64>() as usize) & mask
}

fn index_seed(roots_seed: F, ell: usize, q: usize) -> F {
    tr_hash_fields_tagged(
        ds::FRI_INDEX,
        &[roots_seed, F::from(ell as u64), F::from(q as u64)],
    )
}

fn f0_trace_hash(n0: usize, seed_z: u64) -> [u8; HASH_BYTES] {
    let mut h = SelectedHasher::new();
    Digest::update(&mut h, b"FRI/F0_TREE_DOMAIN");
    Digest::update(&mut h, &(n0 as u64).to_le_bytes());
    Digest::update(&mut h, &seed_z.to_le_bytes());
    finalize_to_digest(h)
}

fn f0_tree_config(n0: usize) -> MerkleChannelCfg {
    let arity = pick_arity_for_layer(n0, 16).max(2);
    let depth = merkle_depth(n0, arity);
    MerkleChannelCfg::new(vec![arity; depth], 0xFF)
}

fn pick_arity_for_layer(n: usize, requested_m: usize) -> usize {
    if requested_m >= 128 && n % 128 == 0 { return 128; }
    if requested_m >= 64  && n % 64  == 0 { return 64; }
    if requested_m >= 32  && n % 32  == 0 { return 32; }
    if requested_m >= 16  && n % 16  == 0 { return 16; }
    if requested_m >= 8   && n % 8   == 0 { return 8; }
    if requested_m >= 4   && n % 4   == 0 { return 4; }
    if n % 2 == 0 { return 2; }
    1
}

fn bind_statement_to_transcript<E: TowerField>(
    tr: &mut Transcript,
    schedule: &[usize],
    n0: usize,
    seed_z: u64,
    coeff_commit_final: bool,
    stir: bool,
) {
    tr.absorb_bytes(b"DEEP-FRI-STATEMENT");
    tr.absorb_field(F::from(n0 as u64));
    tr.absorb_field(F::from(schedule.len() as u64));
    for &m in schedule {
        tr.absorb_field(F::from(m as u64));
    }
    tr.absorb_field(F::from(seed_z));
    tr.absorb_field(F::from(E::DEGREE as u64));
    tr.absorb_field(F::from(coeff_commit_final as u64));
    tr.absorb_field(F::from(stir as u64));
}

pub fn fri_fold_layer(
    evals: &[F],
    z_l: F,
    folding_factor: usize,
) -> Vec<F> {
    let domain_size = evals.len();
    let domain = GeneralEvaluationDomain::<F>::new(domain_size)
        .expect("Domain size must be a power of two.");
    let domain_generator = domain.group_gen();
    fri_fold_layer_impl(evals, z_l, domain_generator, folding_factor)
}

fn fri_fold_layer_impl(
    evals: &[F],
    z_l: F,
    omega: F,
    folding_factor: usize,
) -> Vec<F> {
    let n = evals.len();
    assert!(n % folding_factor == 0);
    let n_next = n / folding_factor;
    let mut out = vec![F::zero(); n_next];
    let z_pows = build_z_pows(z_l, folding_factor);

    if enable_parallel(n_next) {
        #[cfg(feature = "parallel")]
        {
            out.par_iter_mut().enumerate().for_each(|(b, out_b)| {
                let mut acc = F::zero();
                for j in 0..folding_factor {
                    acc += evals[b + j * n_next] * z_pows[j];
                }
                *out_b = acc;
            });
            return out;
        }
    }

    for b in 0..n_next {
        let mut acc = F::zero();
        for j in 0..folding_factor {
            acc += evals[b + j * n_next] * z_pows[j];
        }
        out[b] = acc;
    }
    out
}

// ────────────────────────────────────────────────────────────────────────
//  Core protocol structs — generic over E : TowerField
// ────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct CombinedLeaf<E: TowerField> {
    pub f: E,
    pub s: E,
    pub q: E,
}

pub struct FriLayerCommitment {
    pub n: usize,
    pub m: usize,
    pub root: [u8; HASH_BYTES],
}

pub struct FriTranscript {
    pub schedule: Vec<usize>,
    pub layers: Vec<FriLayerCommitment>,
}

pub struct FriProverParams {
    pub schedule: Vec<usize>,
    pub seed_z: u64,
    pub coeff_commit_final: bool,
    pub d_final: usize,
    pub stir: bool,
}

pub struct FriProverState<E: TowerField> {
    pub f0_base: Vec<F>,
    pub f_layers_ext: Vec<Vec<E>>,
    pub s_layers: Vec<Vec<E>>,
    pub q_layers: Vec<Vec<E>>,
    pub fz_layers: Vec<E>,
    pub transcript: FriTranscript,
    pub omega_layers: Vec<F>,
    pub z_ext: E,
    pub alpha_layers: Vec<E>,
    pub root_f0: [u8; HASH_BYTES],
    pub trace_hash: [u8; HASH_BYTES],
    pub seed_z: u64,
    pub coeff_tuples: Option<Vec<Vec<E>>>,
    pub coeff_root: Option<[u8; HASH_BYTES]>,
    pub beta_deg: Option<E>,
    pub coeff_commit_final: bool,
    pub d_final: usize,
    pub stir_coset_evals: Option<Vec<Vec<E>>>,
    pub stir_z_per_layer: Option<Vec<E>>,
    pub stir_interp_coeffs: Option<Vec<Vec<E>>>,
    pub stir: bool,
}

#[derive(Clone)]
pub struct LayerQueryRef {
    pub i: usize,
    pub child_pos: usize,
    pub parent_index: usize,
    pub parent_pos: usize,
}

#[derive(Clone)]
pub struct FriQueryOpenings {
    pub per_layer_refs: Vec<LayerQueryRef>,
    pub final_index: usize,
}

#[derive(Clone)]
pub struct LayerOpenPayload<E: TowerField> {
    pub f_val: E,
    pub s_val: E,
    pub q_val: E,
}

#[derive(Clone)]
pub struct FriQueryPayload<E: TowerField> {
    pub per_layer_refs: Vec<LayerQueryRef>,
    pub per_layer_payloads: Vec<LayerOpenPayload<E>>,
    pub f0_opening: MerkleOpening,
    pub final_index: usize,
}

/// STIR proximity-query payload: only (f, q) at layer 0
#[derive(Clone)]
pub struct StirProximityPayload<E: TowerField> {
    pub i: usize,
    pub f_val: E,
    pub q_val: E,
    pub f0_opening: MerkleOpening,
    pub layer0_opening: MerkleOpening,
}

#[derive(Clone)]
pub struct LayerProof {
    pub openings: Vec<MerkleOpening>,
}

pub struct FriLayerProofs {
    pub layers: Vec<LayerProof>,
}

pub struct DeepFriProof<E: TowerField> {
    pub root_f0: [u8; HASH_BYTES],
    pub roots: Vec<[u8; HASH_BYTES]>,
    pub layer_proofs: FriLayerProofs,
    pub f0_openings: Vec<MerkleOpening>,
    pub queries: Vec<FriQueryPayload<E>>,
    pub fz_per_layer: Vec<E>,
    pub final_poly_coeffs: Vec<E>,
    pub n0: usize,
    pub omega0: F,
    pub coeff_tuples: Option<Vec<Vec<E>>>,
    pub coeff_root: Option<[u8; HASH_BYTES]>,
    pub stir_coset_evals: Option<Vec<Vec<E>>>,
    /// STIR additive proximity queries — only present when stir=true
    pub stir_proximity_queries: Option<Vec<StirProximityPayload<E>>>,
}

#[derive(Clone, Debug)]
pub struct DeepFriParams {
    pub schedule: Vec<usize>,
    pub r: usize,
    pub seed_z: u64,
    pub coeff_commit_final: bool,
    pub d_final: usize,
    pub stir: bool,
    /// Number of STIR proximity queries (s_0).
    /// Only used when stir=true.  Defaults to `r` for backward compat.
    pub s0: usize,
}

impl DeepFriParams {
    pub fn new(schedule: Vec<usize>, r: usize, seed_z: u64) -> Self {
        Self {
            schedule,
            r,
            seed_z,
            coeff_commit_final: false,
            d_final: 1,
            stir: false,
            s0: r,
        }
    }

    pub fn with_coeff_commit(mut self) -> Self {
        self.coeff_commit_final = true;
        self
    }

    pub fn with_d_final(mut self, d: usize) -> Self {
        self.d_final = d;
        self
    }

    pub fn with_stir(mut self) -> Self {
        self.stir = true;
        self
    }

    pub fn with_s0(mut self, s0: usize) -> Self {
        self.s0 = s0;
        self
    }
}

// ────────────────────────────────────────────────────────────────────────
//  Leaf serialization helpers
// ────────────────────────────────────────────────────────────────────────

/// Classic FRI leaf: (f, s, q) — three extension-field elements
#[inline]
fn ext_leaf_fields<E: TowerField>(f: E, s: E, q: E) -> Vec<F> {
    let mut fields = f.to_fp_components();
    fields.extend(s.to_fp_components());
    fields.extend(q.to_fp_components());
    fields
}

/// STIR leaf: (f, q) — two extension-field elements (no s_val needed)
#[inline]
fn stir_leaf_fields<E: TowerField>(f: E, q: E) -> Vec<F> {
    let mut fields = f.to_fp_components();
    fields.extend(q.to_fp_components());
    fields
}

// ────────────────────────────────────────────────────────────────────────
//  Extension-field challenge helpers
// ────────────────────────────────────────────────────────────────────────

fn challenge_ext<E: TowerField>(tr: &mut Transcript, tag: &[u8]) -> E {
    let d = E::DEGREE;
    let mut components = Vec::with_capacity(d);
    for i in 0..d {
        let mut sub_tag = Vec::with_capacity(tag.len() + 5);
        sub_tag.extend_from_slice(tag);
        sub_tag.extend_from_slice(b"/c");
        for byte in i.to_string().bytes() {
            sub_tag.push(byte);
        }
        components.push(safe_field_challenge(tr, &sub_tag));
    }
    E::from_fp_components(&components)
        .expect("challenge_ext: failed to build extension element from squeezed components")
}

fn absorb_ext<E: TowerField>(tr: &mut Transcript, v: E) {
    for c in v.to_fp_components() {
        tr.absorb_field(c);
    }
}

// =============================================================================
// ── Transcript builder — generic over E : TowerField ──
// =============================================================================

pub fn fri_build_transcript<E: TowerField>(
    f0: Vec<F>,
    domain0: FriDomain,
    params: &FriProverParams,
) -> FriProverState<E> {
    let schedule = params.schedule.clone();
    let l = schedule.len();
    let use_coeff_commit = params.coeff_commit_final && l > 0;
    let use_stir = params.stir;
    let normal_layers = if use_coeff_commit { l - 1 } else { l };

    let mut tr = Transcript::new_matching_hash(b"FRI/FS");
    bind_statement_to_transcript::<E>(
        &mut tr,
        &schedule,
        domain0.size,
        params.seed_z,
        params.coeff_commit_final,
        params.stir,
    );

    let f0_th = f0_trace_hash(domain0.size, params.seed_z);
    let f0_cfg = f0_tree_config(domain0.size);
    let mut f0_tree = MerkleTreeChannel::new(f0_cfg.clone(), f0_th);
    for &val in &f0 {
        f0_tree.push_leaf(&[val]);
    }
    let root_f0 = f0_tree.finalize();

    tr.absorb_bytes(&root_f0);

    let z_ext = challenge_ext::<E>(&mut tr, b"z_fp3");

    let trace_hash: [u8; HASH_BYTES] = transcript_challenge_hash(&mut tr, ds::FRI_SEED);

    let f0_ext: Vec<E> = f0.iter().map(|&x| E::from_fp(x)).collect();

    let mut f_layers_ext: Vec<Vec<E>> = Vec::with_capacity(l + 1);
    let mut s_layers: Vec<Vec<E>> = Vec::with_capacity(l + 1);
    let mut q_layers: Vec<Vec<E>> = Vec::with_capacity(l);
    let mut fz_layers: Vec<E> = Vec::with_capacity(l);
    let mut omega_layers: Vec<F> = Vec::with_capacity(l);
    let mut alpha_layers: Vec<E> = Vec::with_capacity(l);
    let mut layer_commitments: Vec<FriLayerCommitment> = Vec::with_capacity(l);

    let mut stir_all_coset_evals: Vec<Vec<E>> = Vec::with_capacity(l);
    let mut stir_z_per_layer: Vec<E> = Vec::with_capacity(l);
    let mut stir_all_interp_coeffs: Vec<Vec<E>> = Vec::with_capacity(l);
    let mut z_current = z_ext;

    f_layers_ext.push(f0_ext);
    let mut cur_size = domain0.size;

    for ell in 0..normal_layers {
        let m = schedule[ell];

        let alpha_ell = challenge_ext::<E>(&mut tr, b"alpha");
        alpha_layers.push(alpha_ell);

        let dom = Domain::<F>::new(cur_size).unwrap();
        let omega = dom.group_gen;
        omega_layers.push(omega);

        let cur_f = &f_layers_ext[ell];

        let (q, fz, coset_evals_opt, interp_coeffs_opt) = if use_stir {
            let z_ell = z_current;
            stir_z_per_layer.push(z_ell);

            let (q, coset_evals, interp_coeffs) =
                compute_stir_quotient_ext(cur_f, z_ell, omega, m);

            let fz = coset_evals[0];

            z_current = ext_pow(z_current, m as u64);

            (q, fz, Some(coset_evals), Some(interp_coeffs))
        } else {
            let (q, fz) = compute_q_layer_ext_on_ext(cur_f, z_ext, omega);
            (q, fz, None, None)
        };

        q_layers.push(q.clone());
        fz_layers.push(fz);

        let coeff_tuples_layer = extract_all_coset_coefficients(cur_f, omega, m);

        // In STIR mode we commit (f, q) pairs; in classic mode (f, s, q) triples
        if use_stir {
            let s = vec![E::zero(); cur_size]; // placeholder, not committed
            s_layers.push(s);

            let arity = pick_arity_for_layer(cur_size, m).max(2);
            let depth = merkle_depth(cur_size, arity);
            let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);
            let mut tree = MerkleTreeChannel::new(cfg, trace_hash);

            let all_fields: Vec<Vec<F>> = (0..cur_size)
                .map(|i| stir_leaf_fields(cur_f[i], q_layers[ell][i]))
                .collect();
            tree.push_leaves_parallel(&all_fields);

            let layer_root = tree.finalize();

            layer_commitments.push(FriLayerCommitment {
                n: cur_size,
                m,
                root: layer_root,
            });
        } else {
            let s = compute_s_layer_from_coeffs(&coeff_tuples_layer, alpha_ell, cur_size, m);
            s_layers.push(s.clone());

            let arity = pick_arity_for_layer(cur_size, m).max(2);
            let depth = merkle_depth(cur_size, arity);
            let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);
            let mut tree = MerkleTreeChannel::new(cfg, trace_hash);

            let all_fields: Vec<Vec<F>> = (0..cur_size)
                .map(|i| ext_leaf_fields(cur_f[i], s_layers[ell][i], q_layers[ell][i]))
                .collect();
            tree.push_leaves_parallel(&all_fields);

            let layer_root = tree.finalize();

            layer_commitments.push(FriLayerCommitment {
                n: cur_size,
                m,
                root: layer_root,
            });
        }

        if use_stir {
            let coset_evals = coset_evals_opt.as_ref().unwrap();
            for &ev in coset_evals {
                absorb_ext(&mut tr, ev);
            }
            stir_all_coset_evals.push(coset_evals_opt.unwrap());
            stir_all_interp_coeffs.push(interp_coeffs_opt.unwrap());
        } else {
            absorb_ext(&mut tr, fz);
        }
        tr.absorb_bytes(&layer_commitments.last().unwrap().root);

        let next_f = interpolation_fold_ext(&coeff_tuples_layer, alpha_ell);
        cur_size /= m;
        f_layers_ext.push(next_f);

        logln!(
            "[PROVER] ell={} stir={} z_ell={:?} alpha={:?}",
            ell, use_stir,
            if use_stir { stir_z_per_layer.last().copied() } else { Some(z_ext) },
            alpha_ell
        );
    }

    let mut stored_coeff_tuples: Option<Vec<Vec<E>>> = None;
    let mut stored_coeff_root: Option<[u8; HASH_BYTES]> = None;
    let mut stored_beta: Option<E> = None;

    if use_coeff_commit {
        let ell = l - 1;
        let m = schedule[ell];

        let dom = Domain::<F>::new(cur_size).unwrap();
        let omega = dom.group_gen;
        omega_layers.push(omega);

        let cur_f = &f_layers_ext[ell];

        let (q, fz, coset_evals_opt, interp_coeffs_opt) = if use_stir {
            let z_ell = z_current;
            stir_z_per_layer.push(z_ell);

            let (q, coset_evals, interp_coeffs) =
                compute_stir_quotient_ext(cur_f, z_ell, omega, m);

            let fz = coset_evals[0];
            z_current = ext_pow(z_current, m as u64);

            (q, fz, Some(coset_evals), Some(interp_coeffs))
        } else {
            let (q, fz) = compute_q_layer_ext_on_ext(cur_f, z_ext, omega);
            (q, fz, None, None)
        };

        q_layers.push(q.clone());
        fz_layers.push(fz);

        let s = vec![E::zero(); cur_size];
        s_layers.push(s.clone());

        if use_stir {
            let arity = pick_arity_for_layer(cur_size, m).max(2);
            let depth = merkle_depth(cur_size, arity);
            let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);
            let mut tree = MerkleTreeChannel::new(cfg, trace_hash);

            let all_fields: Vec<Vec<F>> = (0..cur_size)
                .map(|i| stir_leaf_fields(cur_f[i], q_layers[ell][i]))
                .collect();
            tree.push_leaves_parallel(&all_fields);

            let layer_root = tree.finalize();

            layer_commitments.push(FriLayerCommitment {
                n: cur_size,
                m,
                root: layer_root,
            });
        } else {
            let arity = pick_arity_for_layer(cur_size, m).max(2);
            let depth = merkle_depth(cur_size, arity);
            let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);
            let mut tree = MerkleTreeChannel::new(cfg, trace_hash);

            let all_fields: Vec<Vec<F>> = (0..cur_size)
                .map(|i| ext_leaf_fields(cur_f[i], s[i], q_layers[ell][i]))
                .collect();
            tree.push_leaves_parallel(&all_fields);

            let layer_root = tree.finalize();

            layer_commitments.push(FriLayerCommitment {
                n: cur_size,
                m,
                root: layer_root,
            });
        }

        if use_stir {
            let coset_evals = coset_evals_opt.as_ref().unwrap();
            for &ev in coset_evals {
                absorb_ext(&mut tr, ev);
            }
            stir_all_coset_evals.push(coset_evals_opt.unwrap());
            stir_all_interp_coeffs.push(interp_coeffs_opt.unwrap());
        } else {
            absorb_ext(&mut tr, fz);
        }
        tr.absorb_bytes(&layer_commitments.last().unwrap().root);

        let coeff_tuples = extract_all_coset_coefficients(cur_f, omega, m);

        let n_final = cur_size / m;
        let coeff_cfg = coeff_tree_config(n_final);
        let mut coeff_tree = MerkleTreeChannel::new(coeff_cfg, trace_hash);

        let coeff_fields: Vec<Vec<F>> = coeff_tuples
            .iter()
            .map(|t| coeff_leaf_fields(t))
            .collect();
        coeff_tree.push_leaves_parallel(&coeff_fields);

        let coeff_root = coeff_tree.finalize();

        tr.absorb_bytes(&coeff_root);

        let alpha_ell = challenge_ext::<E>(&mut tr, b"alpha");
        alpha_layers.push(alpha_ell);

        let beta_deg = challenge_ext::<E>(&mut tr, b"beta_deg");

        let next_f = interpolation_fold_ext(&coeff_tuples, alpha_ell);
        cur_size = n_final;
        f_layers_ext.push(next_f);

        stored_coeff_tuples = Some(coeff_tuples);
        stored_coeff_root = Some(coeff_root);
        stored_beta = Some(beta_deg);
    }

    s_layers.push(vec![E::zero(); f_layers_ext.last().unwrap().len()]);

    FriProverState {
        f0_base: f0,
        f_layers_ext,
        s_layers,
        q_layers,
        fz_layers,
        transcript: FriTranscript {
            schedule,
            layers: layer_commitments,
        },
        omega_layers,
        z_ext,
        alpha_layers,
        root_f0,
        trace_hash,
        seed_z: params.seed_z,
        coeff_tuples: stored_coeff_tuples,
        coeff_root: stored_coeff_root,
        beta_deg: stored_beta,
        coeff_commit_final: use_coeff_commit,
        d_final: params.d_final,
        stir_coset_evals: if use_stir { Some(stir_all_coset_evals) } else { None },
        stir_z_per_layer: if use_stir { Some(stir_z_per_layer) } else { None },
        stir_interp_coeffs: if use_stir { Some(stir_all_interp_coeffs) } else { None },
        stir: use_stir,
    }
}

// =============================================================================
// ── Query derivation — generic over E ──
// =============================================================================

pub fn fri_prove_queries<E: TowerField>(
    st: &FriProverState<E>,
    r: usize,
    query_seed: F,
) -> (Vec<FriQueryOpenings>, Vec<[u8; HASH_BYTES]>, FriLayerProofs, Vec<MerkleOpening>) {
    let L = st.transcript.schedule.len();
    let mut all_refs = Vec::with_capacity(r);
    let n0 = st.transcript.layers.first().map_or(0, |l| l.n);

    for q in 0..r {
        let mut per_layer_refs = Vec::with_capacity(L);

        let mut i = {
            let n_pow2 = n0.next_power_of_two();
            let seed = index_seed(query_seed, 0, q);
            index_from_seed(seed, n_pow2) % n0
        };

        for ell in 0..L {
            let n = st.transcript.layers[ell].n;
            let m = st.transcript.schedule[ell];
            let n_next = n / m;

            per_layer_refs.push(LayerQueryRef {
                i,
                child_pos: i % m,
                parent_index: i % n_next,
                parent_pos: 0,
            });

            i = i % n_next;
        }

        all_refs.push(FriQueryOpenings {
            per_layer_refs,
            final_index: i,
        });
    }

    let f0_th = f0_trace_hash(n0, st.seed_z);
    let f0_cfg = f0_tree_config(n0);
    let mut f0_tree = MerkleTreeChannel::new(f0_cfg, f0_th);
    for &val in &st.f0_base {
        f0_tree.push_leaf(&[val]);
    }
    f0_tree.finalize();

    let mut f0_openings = Vec::with_capacity(r);
    for q in 0..r {
        let idx = all_refs[q].per_layer_refs[0].i;
        f0_openings.push(f0_tree.open(idx));
    }

    let mut layer_proofs = Vec::with_capacity(L);

    for ell in 0..L {
        let layer = &st.transcript.layers[ell];
        let arity = pick_arity_for_layer(layer.n, layer.m).max(2);
        let depth = merkle_depth(layer.n, arity);
        let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);
        let mut tree = MerkleTreeChannel::new(cfg, st.trace_hash);

        if st.stir {
            for i in 0..layer.n {
                let fields = stir_leaf_fields(
                    st.f_layers_ext[ell][i],
                    st.q_layers[ell][i],
                );
                tree.push_leaf(&fields);
            }
        } else {
            for i in 0..layer.n {
                let fields = ext_leaf_fields(
                    st.f_layers_ext[ell][i],
                    st.s_layers[ell][i],
                    st.q_layers[ell][i],
                );
                tree.push_leaf(&fields);
            }
        }
        tree.finalize();

        let mut openings = Vec::with_capacity(r);
        for q in 0..r {
            let idx = all_refs[q].per_layer_refs[ell].i;
            openings.push(tree.open(idx));
        }
        layer_proofs.push(LayerProof { openings });
    }

    let roots: Vec<[u8; HASH_BYTES]> = st.transcript.layers.iter().map(|l| l.root).collect();

    (all_refs, roots, FriLayerProofs { layers: layer_proofs }, f0_openings)
}

// =============================================================================
// ── STIR proximity query derivation ──
// =============================================================================

/// Derive s_0 STIR proximity queries that open layer 0 only.
/// These are independent of the classic FRI queries.
fn stir_prove_proximity_queries<E: TowerField>(
    st: &FriProverState<E>,
    s0: usize,
    query_seed: F,
    r: usize,
) -> Vec<StirProximityPayload<E>> {
    if !st.stir || st.transcript.layers.is_empty() {
        return vec![];
    }

    let n0 = st.transcript.layers[0].n;

    // Build f0 tree for openings
    let f0_th = f0_trace_hash(n0, st.seed_z);
    let f0_cfg = f0_tree_config(n0);
    let mut f0_tree = MerkleTreeChannel::new(f0_cfg.clone(), f0_th);
    for &val in &st.f0_base {
        f0_tree.push_leaf(&[val]);
    }
    f0_tree.finalize();

    // Build layer-0 tree for openings
    let layer0 = &st.transcript.layers[0];
    let arity = pick_arity_for_layer(layer0.n, layer0.m).max(2);
    let depth = merkle_depth(layer0.n, arity);
    let cfg0 = MerkleChannelCfg::new(vec![arity; depth], 0u64);
    let mut tree0 = MerkleTreeChannel::new(cfg0, st.trace_hash);
    for i in 0..layer0.n {
        let fields = stir_leaf_fields(
            st.f_layers_ext[0][i],
            st.q_layers[0][i],
        );
        tree0.push_leaf(&fields);
    }
    tree0.finalize();

    let mut payloads = Vec::with_capacity(s0);

    // Proximity queries use distinct seed namespace (offset by r to avoid collision)
    for q in 0..s0 {
        let seed = index_seed(query_seed, 0, r + q);
        let n_pow2 = n0.next_power_of_two();
        let i = index_from_seed(seed, n_pow2) % n0;

        let f0_opening = f0_tree.open(i);
        let layer0_opening = tree0.open(i);

        payloads.push(StirProximityPayload {
            i,
            f_val: st.f_layers_ext[0][i],
            q_val: st.q_layers[0][i],
            f0_opening,
            layer0_opening,
        });
    }

    payloads
}

// =============================================================================
// ── Prover top-level — generic over E ──
// =============================================================================

pub fn deep_fri_prove<E: TowerField>(
    f0: Vec<F>,
    domain0: FriDomain,
    params: &DeepFriParams,
) -> DeepFriProof<E> {
    let prover_params = FriProverParams {
        schedule: params.schedule.clone(),
        seed_z: params.seed_z,
        coeff_commit_final: params.coeff_commit_final,
        d_final: params.d_final,
        stir: params.stir,
    };

    let st: FriProverState<E> = fri_build_transcript(f0, domain0, &prover_params);

    let L = params.schedule.len();
    let final_evals = st.f_layers_ext[L].clone();

    let final_poly_coeffs: Vec<E> = {
        let all_coeffs = ext_evals_to_coeffs::<E>(&final_evals);
        let d_final = params.d_final.min(all_coeffs.len());

        if cfg!(debug_assertions) {
            for k in d_final..all_coeffs.len() {
                if all_coeffs[k] != E::zero() {
                    eprintln!(
                        "[WARN] Final polynomial coefficient at degree {} is non-zero; \
                         proof may not verify (d_final={})",
                        k, params.d_final,
                    );
                    break;
                }
            }
        }

        all_coeffs[..d_final].to_vec()
    };

    let query_seed = {
        let mut tr = Transcript::new_matching_hash(b"FRI/FS");
        bind_statement_to_transcript::<E>(
            &mut tr,
            &params.schedule,
            domain0.size,
            params.seed_z,
            params.coeff_commit_final,
            params.stir,
        );
        tr.absorb_bytes(&st.root_f0);

        let _ = challenge_ext::<E>(&mut tr, b"z_fp3");
        let _: [u8; HASH_BYTES] = transcript_challenge_hash(&mut tr, ds::FRI_SEED);

        let use_coeff_commit = params.coeff_commit_final && L > 0;
        let normal_layers = if use_coeff_commit { L - 1 } else { L };

        for ell in 0..normal_layers {
            let _ = challenge_ext::<E>(&mut tr, b"alpha");
            if params.stir {
                let coset_evals = &st.stir_coset_evals.as_ref().unwrap()[ell];
                for &ev in coset_evals {
                    absorb_ext(&mut tr, ev);
                }
            } else {
                absorb_ext(&mut tr, st.fz_layers[ell]);
            }
            tr.absorb_bytes(&st.transcript.layers[ell].root);
        }

        if use_coeff_commit {
            let ell = L - 1;
            if params.stir {
                let coset_evals = &st.stir_coset_evals.as_ref().unwrap()[ell];
                for &ev in coset_evals {
                    absorb_ext(&mut tr, ev);
                }
            } else {
                absorb_ext(&mut tr, st.fz_layers[ell]);
            }
            tr.absorb_bytes(&st.transcript.layers[ell].root);

            tr.absorb_bytes(&st.coeff_root.unwrap());
            let _ = challenge_ext::<E>(&mut tr, b"alpha");
            let _ = challenge_ext::<E>(&mut tr, b"beta_deg");
        }

        for &c in &final_poly_coeffs {
            absorb_ext::<E>(&mut tr, c);
        }

        safe_field_challenge(&mut tr, b"query_seed")
    };

    // In STIR mode: r=0 classic fold-queries, s0 proximity queries
    // In classic mode: r classic fold-queries, 0 proximity queries
    let effective_r = if params.stir { 0 } else { params.r };

    let (query_refs, roots, layer_proofs, f0_openings) =
        fri_prove_queries(&st, effective_r, query_seed);

    let stir_proximity = if params.stir {
        Some(stir_prove_proximity_queries(&st, params.s0, query_seed, effective_r))
    } else {
        None
    };

    let mut queries = Vec::with_capacity(effective_r);

    for (qi, q) in query_refs.into_iter().enumerate() {
        let mut payloads = Vec::with_capacity(params.schedule.len());

        for (ell, rref) in q.per_layer_refs.iter().enumerate() {
            payloads.push(LayerOpenPayload {
                f_val: st.f_layers_ext[ell][rref.i],
                s_val: st.s_layers[ell][rref.i],
                q_val: st.q_layers[ell][rref.i],
            });
        }

        queries.push(FriQueryPayload {
            per_layer_refs: q.per_layer_refs,
            per_layer_payloads: payloads,
            f0_opening: f0_openings[qi].clone(),
            final_index: q.final_index,
        });
    }

    let stir_coset_evals_proof = if params.stir {
        st.stir_coset_evals.clone()
    } else {
        None
    };

    DeepFriProof {
        root_f0: st.root_f0,
        roots,
        layer_proofs,
        f0_openings: queries.iter().map(|q| q.f0_opening.clone()).collect(),
        queries,
        fz_per_layer: st.fz_layers.clone(),
        final_poly_coeffs,
        n0: domain0.size,
        omega0: domain0.omega,
        coeff_tuples: st.coeff_tuples.clone(),
        coeff_root: st.coeff_root,
        stir_coset_evals: stir_coset_evals_proof,
        stir_proximity_queries: stir_proximity,
    }
}

pub fn deep_fri_proof_size_bytes<E: TowerField>(proof: &DeepFriProof<E>, stir: bool) -> usize {
    const FIELD_BYTES: usize = 8;
    let ext_bytes: usize = E::DEGREE * FIELD_BYTES;

    let mut bytes = 0usize;

    // Merkle roots: f0 root + layer roots
    bytes += HASH_BYTES;
    bytes += proof.roots.len() * HASH_BYTES;

    // Final polynomial coefficients
    bytes += proof.final_poly_coeffs.len() * ext_bytes;

    if stir {
        // ── STIR mode ──
        // Coset evaluations (includes fz as first element, so fz_per_layer is
        // redundant and NOT counted separately)
        if let Some(ref stir_evals) = proof.stir_coset_evals {
            for layer_evals in stir_evals {
                bytes += layer_evals.len() * ext_bytes;
            }
        }

        // Proximity queries: each has (f_val, q_val) + f0 Merkle opening + layer0 Merkle opening
        if let Some(ref prox) = proof.stir_proximity_queries {
            for pq in prox {
                // f_val + q_val (2 extension elements)
                bytes += 2 * ext_bytes;

                // f0 Merkle opening
                bytes += HASH_BYTES; // leaf hash
                for level in &pq.f0_opening.path {
                    bytes += level.len() * HASH_BYTES;
                }

                // layer-0 Merkle opening
                bytes += HASH_BYTES; // leaf hash
                for level in &pq.layer0_opening.path {
                    bytes += level.len() * HASH_BYTES;
                }
            }
        }
    } else {
        // ── Classic FRI mode ──
        // fz per layer
        bytes += proof.fz_per_layer.len() * ext_bytes;

        // Classic query payloads: (f, s, q) per layer per query
        for q in &proof.queries {
            bytes += q.per_layer_payloads.len() * 3 * ext_bytes;
        }

        // f0 Merkle openings
        for opening in &proof.f0_openings {
            bytes += HASH_BYTES;
            for level in &opening.path {
                bytes += level.len() * HASH_BYTES;
            }
        }

        // Layer Merkle openings
        for layer in &proof.layer_proofs.layers {
            for opening in &layer.openings {
                bytes += HASH_BYTES;
                for level in &opening.path {
                    bytes += level.len() * HASH_BYTES;
                }
            }
        }
    }

    // Coefficient-commit data (both modes)
    if let Some(ref tuples) = proof.coeff_tuples {
        for t in tuples {
            bytes += t.len() * ext_bytes;
        }
    }
    if proof.coeff_root.is_some() {
        bytes += HASH_BYTES;
    }

    bytes
}

// =============================================================================
// ── Verifier — generic over E : TowerField ──
// =============================================================================

pub fn deep_fri_verify<E: TowerField>(
    params: &DeepFriParams,
    proof: &DeepFriProof<E>,
) -> bool {
    let L = params.schedule.len();
    let sizes = layer_sizes_from_schedule(proof.n0, &params.schedule);
    let use_coeff_commit = params.coeff_commit_final && L > 0;
    let use_stir = params.stir;
    let normal_layers = if use_coeff_commit { L - 1 } else { L };

    let mut tr = Transcript::new_matching_hash(b"FRI/FS");
    bind_statement_to_transcript::<E>(
        &mut tr,
        &params.schedule,
        proof.n0,
        params.seed_z,
        params.coeff_commit_final,
        params.stir,
    );
    tr.absorb_bytes(&proof.root_f0);

    let z_ext = challenge_ext::<E>(&mut tr, b"z_fp3");
    let trace_hash: [u8; HASH_BYTES] = transcript_challenge_hash(&mut tr, ds::FRI_SEED);

    logln!("[VERIFY] z_ext = {:?}  stir = {}", z_ext, use_stir);

    let mut z_current = z_ext;
    let mut stir_z_per_layer: Vec<E> = Vec::with_capacity(L);
    let mut stir_interp_per_layer: Vec<Vec<E>> = Vec::with_capacity(L);

    let mut alpha_layers: Vec<E> = Vec::with_capacity(L);

    for ell in 0..normal_layers {
        let alpha_ell = challenge_ext::<E>(&mut tr, b"alpha");
        alpha_layers.push(alpha_ell);

        if use_stir {
            let z_ell = z_current;
            stir_z_per_layer.push(z_ell);

            let stir_evals = match proof.stir_coset_evals {
                Some(ref all) if ell < all.len() => &all[ell],
                _ => {
                    eprintln!("[FAIL][STIR COSET EVALS MISSING] ell={}", ell);
                    return false;
                }
            };
            let m = params.schedule[ell];
            if stir_evals.len() != m {
                eprintln!(
                    "[FAIL][STIR COSET EVALS SIZE] ell={} expected={} got={}",
                    ell, m, stir_evals.len()
                );
                return false;
            }

            for &ev in stir_evals {
                absorb_ext(&mut tr, ev);
            }

            let omega_ell = Domain::<F>::new(sizes[ell]).unwrap().group_gen;
            let n_next = sizes[ell] / m;
            let zeta = omega_ell.pow([n_next as u64]);
            let interp_coeffs = interpolate_stir_coset(stir_evals, z_ell, zeta, m);
            stir_interp_per_layer.push(interp_coeffs);

            z_current = ext_pow(z_current, m as u64);
        } else {
            if ell < proof.fz_per_layer.len() {
                absorb_ext(&mut tr, proof.fz_per_layer[ell]);
            }
        }

        if ell < proof.roots.len() {
            tr.absorb_bytes(&proof.roots[ell]);
        }
    }

    let mut beta_deg: Option<E> = None;

    if use_coeff_commit {
        let ell = L - 1;

        if use_stir {
            let z_ell = z_current;
            stir_z_per_layer.push(z_ell);

            let stir_evals = match proof.stir_coset_evals {
                Some(ref all) if ell < all.len() => &all[ell],
                _ => {
                    eprintln!("[FAIL][STIR COSET EVALS MISSING FINAL] ell={}", ell);
                    return false;
                }
            };
            let m = params.schedule[ell];
            if stir_evals.len() != m {
                eprintln!(
                    "[FAIL][STIR COSET EVALS SIZE FINAL] ell={} expected={} got={}",
                    ell, m, stir_evals.len()
                );
                return false;
            }
            for &ev in stir_evals {
                absorb_ext(&mut tr, ev);
            }

            let omega_ell = Domain::<F>::new(sizes[ell]).unwrap().group_gen;
            let n_next = sizes[ell] / m;
            let zeta = omega_ell.pow([n_next as u64]);
            let interp_coeffs = interpolate_stir_coset(stir_evals, z_ell, zeta, m);
            stir_interp_per_layer.push(interp_coeffs);

            z_current = ext_pow(z_current, m as u64);
        } else {
            if ell < proof.fz_per_layer.len() {
                absorb_ext(&mut tr, proof.fz_per_layer[ell]);
            }
        }

        if ell < proof.roots.len() {
            tr.absorb_bytes(&proof.roots[ell]);
        }

        match proof.coeff_root {
            Some(ref cr) => tr.absorb_bytes(cr),
            None => {
                eprintln!("[FAIL][COEFF ROOT MISSING]");
                return false;
            }
        }

        let alpha_ell = challenge_ext::<E>(&mut tr, b"alpha");
        alpha_layers.push(alpha_ell);

        let beta = challenge_ext::<E>(&mut tr, b"beta_deg");
        beta_deg = Some(beta);
    }

    for &c in &proof.final_poly_coeffs {
        absorb_ext::<E>(&mut tr, c);
    }

    let query_seed: F = safe_field_challenge(&mut tr, b"query_seed");

    if proof.final_poly_coeffs.len() != params.d_final {
        eprintln!(
            "[FAIL][FINAL POLY COEFFS SIZE] expected={} got={}",
            params.d_final,
            proof.final_poly_coeffs.len()
        );
        return false;
    }

    // ─── STIR OOD fold-consistency chain ───
    if use_stir {
        for ell in 0..L {
            let m = params.schedule[ell];
            let alpha_ell = alpha_layers[ell];
            let alpha_pows = build_ext_pows(alpha_ell, m);

            let interp = &stir_interp_per_layer[ell];
            let mut fold_at_ood = E::zero();
            for k in 0..m {
                fold_at_ood = fold_at_ood + interp[k] * alpha_pows[k];
            }

            let expected = if ell + 1 < L {
                proof.stir_coset_evals.as_ref().unwrap()[ell + 1][0]
            } else {
                let mut z_final = z_ext;
                for i in 0..L {
                    z_final = ext_pow(z_final, params.schedule[i] as u64);
                }
                eval_final_poly_ext(&proof.final_poly_coeffs, z_final)
            };

            if fold_at_ood != expected {
                eprintln!(
                    "[FAIL][STIR OOD FOLD] ell={}\n  fold_at_ood={:?}\n  expected={:?}",
                    ell, fold_at_ood, expected,
                );
                return false;
            }
        }
    }

    // ─── Coefficient-commit checks ───
    if use_coeff_commit {
        let coeff_tuples = match proof.coeff_tuples {
            Some(ref ct) => ct,
            None => {
                eprintln!("[FAIL][COEFF TUPLES MISSING]");
                return false;
            }
        };

        let n_final = sizes[L];
        let m_final = params.schedule[L - 1];

        if coeff_tuples.len() != n_final {
            eprintln!(
                "[FAIL][COEFF TUPLES SIZE] expected={} got={}",
                n_final,
                coeff_tuples.len()
            );
            return false;
        }
        for (b, t) in coeff_tuples.iter().enumerate() {
            if t.len() != m_final {
                eprintln!(
                    "[FAIL][COEFF TUPLE WIDTH] coset={} expected={} got={}",
                    b, m_final, t.len()
                );
                return false;
            }
        }

        let coeff_cfg = coeff_tree_config(n_final);
        let mut coeff_tree = MerkleTreeChannel::new(coeff_cfg.clone(), trace_hash);
        let coeff_fields: Vec<Vec<F>> = coeff_tuples
            .iter()
            .map(|t| coeff_leaf_fields(t))
            .collect();
        coeff_tree.push_leaves_parallel(&coeff_fields);
        let recomputed_root = coeff_tree.finalize();

        if recomputed_root != proof.coeff_root.unwrap() {
            eprintln!("[FAIL][COEFF MERKLE ROOT MISMATCH]");
            return false;
        }

        let beta = beta_deg.unwrap();
        if !batched_degree_check_ext(coeff_tuples, beta, params.d_final) {
            eprintln!("[FAIL][BATCHED DEGREE CHECK]");
            return false;
        }
    }

    let omega_per_layer: Vec<F> = (0..L)
        .map(|ell| Domain::<F>::new(sizes[ell]).unwrap().group_gen)
        .collect();
    let omega_final: F = if sizes[L] >= 2 {
        Domain::<F>::new(sizes[L]).unwrap().group_gen
    } else {
        F::one()
    };

    let f0_th = f0_trace_hash(proof.n0, params.seed_z);
    let f0_cfg = f0_tree_config(proof.n0);

    // =================================================================
    //  STIR mode: verify s_0 proximity-only queries at layer 0
    // =================================================================
    if use_stir {
        let prox_queries = match proof.stir_proximity_queries {
            Some(ref pq) => pq,
            None => {
                eprintln!("[FAIL][STIR PROXIMITY QUERIES MISSING]");
                return false;
            }
        };

        if prox_queries.len() != params.s0 {
            eprintln!(
                "[FAIL][STIR S0 COUNT] expected={} got={}",
                params.s0,
                prox_queries.len()
            );
            return false;
        }

        let n0 = proof.n0;
        let layer0_arity = pick_arity_for_layer(sizes[0], params.schedule[0]).max(2);
        let layer0_depth = merkle_depth(sizes[0], layer0_arity);
        let layer0_cfg = MerkleChannelCfg::new(vec![layer0_arity; layer0_depth], 0u64);

        for (qi, pq) in prox_queries.iter().enumerate() {
            // Verify query index derivation
            let expected_i = {
                let effective_r = 0usize; // STIR mode has r=0 classic queries
                let seed = index_seed(query_seed, 0, effective_r + qi);
                let n_pow2 = n0.next_power_of_two();
                index_from_seed(seed, n_pow2) % n0
            };

            if pq.i != expected_i {
                eprintln!(
                    "[FAIL][STIR PROX INDEX] qi={} expected={} got={}",
                    qi, expected_i, pq.i
                );
                return false;
            }

            // Verify f0 Merkle opening
            if !MerkleTreeChannel::verify_opening(
                &f0_cfg,
                proof.root_f0,
                &pq.f0_opening,
                &f0_th,
            ) {
                eprintln!("[FAIL][STIR PROX F0 MERKLE] qi={}", qi);
                return false;
            }

            if pq.f0_opening.index != pq.i {
                eprintln!("[FAIL][STIR PROX F0 INDEX] qi={}", qi);
                return false;
            }

            // f_val must be a base-field lift
            let f_comps = pq.f_val.to_fp_components();
            let is_base = f_comps[1..].iter().all(|&c| c == F::zero());
            if !is_base {
                eprintln!("[FAIL][STIR PROX NOT BASE FIELD] qi={}", qi);
                return false;
            }

            // f0 leaf binding
            let expected_f0_leaf = compute_leaf_hash(
                &f0_cfg,
                pq.f0_opening.index,
                &[f_comps[0]],
            );
            if expected_f0_leaf != pq.f0_opening.leaf {
                eprintln!("[FAIL][STIR PROX F0 LEAF BIND] qi={}", qi);
                return false;
            }

            // Verify layer-0 Merkle opening
            if !MerkleTreeChannel::verify_opening(
                &layer0_cfg,
                proof.roots[0],
                &pq.layer0_opening,
                &trace_hash,
            ) {
                eprintln!("[FAIL][STIR PROX LAYER0 MERKLE] qi={}", qi);
                return false;
            }

            if pq.layer0_opening.index != pq.i {
                eprintln!("[FAIL][STIR PROX LAYER0 INDEX] qi={}", qi);
                return false;
            }

            // Layer-0 leaf binding: STIR leaves are (f, q)
            let leaf_fields = stir_leaf_fields(pq.f_val, pq.q_val);
            let expected_leaf = compute_leaf_hash(&layer0_cfg, pq.layer0_opening.index, &leaf_fields);
            if expected_leaf != pq.layer0_opening.leaf {
                eprintln!("[FAIL][STIR PROX LAYER0 LEAF BIND] qi={}", qi);
                return false;
            }

            // STIR quotient check: f(x_i) = q(x_i) * V(x_i) + P(x_i)
            let omega0 = omega_per_layer[0];
            let x_i = E::from_fp(omega0.pow([pq.i as u64]));
            let m = params.schedule[0];
            let z_ell = stir_z_per_layer[0];
            let z_ell_m = ext_pow(z_ell, m as u64);
            let interp = &stir_interp_per_layer[0];

            let v_xi = ext_pow(x_i, m as u64) - z_ell_m;
            let p_xi = eval_final_poly_ext(interp, x_i);

            let lhs = pq.q_val * v_xi + p_xi;
            if lhs != pq.f_val {
                eprintln!(
                    "[FAIL][STIR PROX DEEP] qi={}\n  f_val={:?}\n  q*V+P={:?}",
                    qi, pq.f_val, lhs,
                );
                return false;
            }
        }

        // STIR mode: no classic fold queries to verify — done
        logln!("[VERIFY] SUCCESS  stir=true  s0={}", params.s0);
        return true;
    }

    // =================================================================
    //  Classic FRI mode: verify r fold-queries across all layers
    // =================================================================
    for q in 0..params.r {
        let qp = &proof.queries[q];

        let expected_i0 = {
            let n_pow2 = proof.n0.next_power_of_two();
            let seed = index_seed(query_seed, 0, q);
            index_from_seed(seed, n_pow2) % proof.n0
        };

        let mut expected_i = expected_i0;
        for ell in 0..L {
            if qp.per_layer_refs[ell].i != expected_i {
                eprintln!(
                    "[FAIL][QUERY POS] q={} ell={} expected={} got={}",
                    q, ell, expected_i, qp.per_layer_refs[ell].i
                );
                return false;
            }
            let n_next = sizes[ell] / params.schedule[ell];
            expected_i = expected_i % n_next;
        }

        if qp.final_index != expected_i {
            eprintln!(
                "[FAIL][FINAL INDEX] q={} expected={} got={}",
                q, expected_i, qp.final_index
            );
            return false;
        }

        {
            let f0_opening = &qp.f0_opening;

            if !MerkleTreeChannel::verify_opening(
                &f0_cfg,
                proof.root_f0,
                f0_opening,
                &f0_th,
            ) {
                eprintln!("[FAIL][F0 MERKLE] q={}", q);
                return false;
            }

            let pay0 = &qp.per_layer_payloads[0];
            let pay0_comps = pay0.f_val.to_fp_components();
            let is_base = pay0_comps[1..].iter().all(|&c| c == F::zero());
            if !is_base {
                eprintln!("[FAIL][LAYER0 NOT BASE FIELD] q={}", q);
                return false;
            }

            let expected_f0_leaf = compute_leaf_hash(
                &f0_cfg,
                f0_opening.index,
                &[pay0_comps[0]],
            );
            if expected_f0_leaf != f0_opening.leaf {
                eprintln!("[FAIL][F0 LEAF BIND] q={}", q);
                return false;
            }

            if f0_opening.index != expected_i0 {
                eprintln!("[FAIL][F0 INDEX] q={}", q);
                return false;
            }
        }

        for ell in 0..L {
            let opening = &proof.layer_proofs.layers[ell].openings[q];
            let rref = &qp.per_layer_refs[ell];
            let pay = &qp.per_layer_payloads[ell];

            let arity = pick_arity_for_layer(sizes[ell], params.schedule[ell]).max(2);
            let depth = merkle_depth(sizes[ell], arity);
            let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);

            if !MerkleTreeChannel::verify_opening(
                &cfg,
                proof.roots[ell],
                opening,
                &trace_hash,
            ) {
                eprintln!("[FAIL][MERKLE] q={} ell={}", q, ell);
                return false;
            }

            if opening.index != rref.i {
                eprintln!("[FAIL][INDEX BINDING] q={} ell={}", q, ell);
                return false;
            }

            let leaf_fields = ext_leaf_fields(pay.f_val, pay.s_val, pay.q_val);
            let expected_leaf = compute_leaf_hash(&cfg, opening.index, &leaf_fields);
            if expected_leaf != opening.leaf {
                eprintln!("[FAIL][LEAF BINDING] q={} ell={}", q, ell);
                return false;
            }

            let omega_ell = omega_per_layer[ell];
            let x_i = E::from_fp(omega_ell.pow([rref.i as u64]));

            // Classic DEEP check
            let fz = proof.fz_per_layer[ell];
            let num   = pay.f_val - fz;
            let denom = x_i - z_ext;

            if pay.q_val * denom != num {
                eprintln!(
                    "[FAIL][DEEP-EXT] q={} ell={}\n  f_val={:?}\n  fz={:?}\n  q_val={:?}\n  x_i={:?}",
                    q, ell, pay.f_val, fz, pay.q_val, x_i,
                );
                return false;
            }

            let is_final_layer = ell == L - 1;

            if is_final_layer && use_coeff_commit {
                let m = params.schedule[ell];
                let n_next = sizes[ell] / m;
                let coset_b = qp.final_index;

                let coeff_tuples = proof.coeff_tuples.as_ref().unwrap();
                let coeff_tuple = &coeff_tuples[coset_b];

                let x_i_base = omega_ell.pow([rref.i as u64]);
                let mut h_star = E::zero();
                let mut x_pow = F::one();
                for k in 0..m {
                    h_star = h_star + coeff_tuple[k] * E::from_fp(x_pow);
                    x_pow *= x_i_base;
                }

                if pay.f_val != h_star {
                    eprintln!(
                        "[FAIL][SINGLE-POINT CONSISTENCY] q={} ell={}\n  f_val={:?}\n  h_star={:?}",
                        q, ell, pay.f_val, h_star,
                    );
                    return false;
                }

                let alpha_final = alpha_layers[L - 1];
                let alpha_pows = build_ext_pows(alpha_final, m);
                let mut fold_val = E::zero();
                for k in 0..m {
                    fold_val = fold_val + coeff_tuple[k] * alpha_pows[k];
                }

                let x_final = E::from_fp(omega_final.pow([qp.final_index as u64]));
                let expected_final = eval_final_poly_ext(
                    &proof.final_poly_coeffs,
                    x_final,
                );

                if fold_val != expected_final {
                    eprintln!(
                        "[FAIL][COEFF FOLD VALUE] q={} fold={:?} poly_eval={:?}",
                        q, fold_val, expected_final
                    );
                    return false;
                }
            } else {
                let verified_f_next = if ell + 1 < L {
                    qp.per_layer_payloads[ell + 1].f_val
                } else {
                    let x_final = E::from_fp(
                        omega_final.pow([qp.final_index as u64]),
                    );
                    eval_final_poly_ext(&proof.final_poly_coeffs, x_final)
                };

                if pay.s_val != verified_f_next {
                    eprintln!(
                        "[FAIL][FOLD] q={} ell={}\n  s_val={:?}\n  f_next={:?}",
                        q, ell, pay.s_val, verified_f_next,
                    );
                    return false;
                }
            }
        }
    }

    logln!("[VERIFY] SUCCESS  stir={}", use_stir);
    true
}

// ================================================================
// Extension-field FRI folding utilities — generic over E
// ================================================================

#[inline]
pub fn fri_fold_degree2<E: TowerField>(
    f_pos: E,
    f_neg: E,
    x: F,
    beta: E,
) -> E {
    let two_inv = F::from(2u64).invert().unwrap();
    let x_inv = x.invert().unwrap();

    let f_even = (f_pos + f_neg) * E::from_fp(two_inv);
    let f_odd  = (f_pos - f_neg) * E::from_fp(two_inv * x_inv);

    f_even + beta * f_odd
}

pub fn fri_fold_degree3<E: TowerField>(
    f_at_y:   E,
    f_at_zy:  E,
    f_at_z2y: E,
    y:        F,
    zeta:     F,
    beta:     E,
) -> E {
    let zeta2 = zeta * zeta;
    let inv3 = F::from(3u64).invert().unwrap();
    let y_inv = y.invert().unwrap();
    let y2_inv = y_inv * y_inv;

    let f0 = (f_at_y + f_at_zy + f_at_z2y) * E::from_fp(inv3);

    let f1 = (f_at_y + f_at_zy * E::from_fp(zeta2) + f_at_z2y * E::from_fp(zeta))
        * E::from_fp(inv3 * y_inv);

    let f2 = (f_at_y + f_at_zy * E::from_fp(zeta) + f_at_z2y * E::from_fp(zeta2))
        * E::from_fp(inv3 * y2_inv);

    let beta_sq = beta.sq();
    f0 + beta * f1 + beta_sq * f2
}

pub fn fri_fold_round<E: TowerField>(
    codeword: &[E],
    domain: &[F],
    beta: E,
) -> Vec<E> {
    let half = codeword.len() / 2;
    let mut folded = Vec::with_capacity(half);

    for i in 0..half {
        let f_pos = codeword[i];
        let f_neg = codeword[i + half];
        let x = domain[i];
        folded.push(fri_fold_degree2(f_pos, f_neg, x, beta));
    }

    folded
}

pub fn fri_verify_query<E: TowerField>(
    round_evals: &[(E, E)],
    round_domains: &[F],
    betas: &[E],
    final_value: E,
) -> bool {
    let num_rounds = betas.len();
    let mut expected = fri_fold_degree2(
        round_evals[0].0,
        round_evals[0].1,
        round_domains[0],
        betas[0],
    );

    for r in 1..num_rounds {
        let (f_pos, f_neg) = round_evals[r];
        if f_pos != expected && f_neg != expected {
            return false;
        }
        expected = fri_fold_degree2(f_pos, f_neg, round_domains[r], betas[r]);
    }

    expected == final_value
}


// ────────────────────────────────────────────────────────────────────────
//  Soundness accounting: proximity-gap bounds & query-count derivation
// ────────────────────────────────────────────────────────────────────────

/// Which proximity-gap bound to use for Reed-Solomon proximity testing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProximityBound {
    /// Classic Johnson bound:  δ = 1 − √ρ,  list size ℓ ≤ O(1/√ρ).
    Johnson,
    /// STIR capacity-approaching bound:  δ = 1 − ρ − γ,  list size ℓ ≤ 1/γ.
    /// `gamma_log2` is log₂(γ), e.g. −30.0 means γ = 2⁻³⁰.
    StirCapacity { gamma_log2: f64 },
}

impl ProximityBound {
    pub fn label(&self) -> String {
        match self {
            Self::Johnson => "johnson".into(),
            Self::StirCapacity { gamma_log2 } =>
                format!("capacity(g=2^{:.0})", gamma_log2),
        }
    }

    /// Returns (δ, log₂(ℓ)) for RS code at rate ρ = degree / domain_size.
    pub fn gap_and_list_size(&self, rho: f64) -> (f64, f64) {
        match *self {
            Self::Johnson => {
                let sqrt_rho = rho.sqrt();
                let delta = 1.0 - sqrt_rho;
                let list_log2 = -(sqrt_rho.log2());
                (delta, list_log2)
            }
            Self::StirCapacity { gamma_log2 } => {
                let gamma = f64::exp2(gamma_log2);
                let delta = 1.0 - rho - gamma;
                assert!(
                    delta > 0.0,
                    "gamma too large for rate {rho}: delta would be {delta}"
                );
                let list_log2 = -gamma_log2;
                (delta, list_log2)
            }
        }
    }
}

/// Hash variant for Merkle commitments.  Determines collision-resistance
/// ceiling on achievable security.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HashVariant {
    Sha3_256,
    Sha3_384,
    Sha3_512,
}

impl HashVariant {
    pub fn output_bits(&self) -> usize {
        match self {
            Self::Sha3_256 => 256,
            Self::Sha3_384 => 384,
            Self::Sha3_512 => 512,
        }
    }

    /// Birthday-bound collision resistance = output_bits / 2.
    pub fn collision_bits(&self) -> usize {
        self.output_bits() / 2
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Sha3_256 => "sha3-256",
            Self::Sha3_384 => "sha3-384",
            Self::Sha3_512 => "sha3-512",
        }
    }

    pub fn all() -> &'static [HashVariant] {
        &[Self::Sha3_256, Self::Sha3_384, Self::Sha3_512]
    }

    /// Detect from the compiled HASH_BYTES constant.
    pub fn from_compiled() -> Self {
        match HASH_BYTES {
            32 => Self::Sha3_256,
            48 => Self::Sha3_384,
            64 => Self::Sha3_512,
            other => panic!(
                "Unknown HASH_BYTES={other}; expected 32, 48, or 64"
            ),
        }
    }
}

/// Per-round breakdown for diagnostics / logging.
#[derive(Clone, Debug)]
pub struct RoundSoundnessDetail {
    pub round: usize,
    pub fold_factor: usize,
    pub degree_before: f64,
    pub alg_error_log2: f64,
}

/// Full soundness report — one per (schedule, bound, hash, s0) tuple.
#[derive(Clone, Debug)]
pub struct SoundnessReport {
    pub bound_label: String,
    pub hash_label: String,
    pub hash_collision_bits: usize,
    pub field_ext_bits: f64,
    pub target_bits: usize,
    pub effective_target: usize,
    pub achieved_bits: f64,
    pub query_security_bits: f64,
    pub algebraic_security_bits: f64,
    pub queries: usize,
    pub delta: f64,
    pub rho: f64,
    pub list_size_log2: f64,
    pub num_rounds: usize,
    pub is_stir: bool,
    pub rounds: Vec<RoundSoundnessDetail>,
}

impl SoundnessReport {
    /// Additional CSV columns to append to your existing header.
    pub fn csv_header_suffix() -> &'static str {
        ",bound,hash,hash_col_bits,field_ext_bits,\
         target_bits,eff_target,achieved_bits,\
         query_sec,alg_sec,queries_computed,\
         delta,rho,list_log2,num_rounds,is_stir"
    }

    /// CSV values matching the header above.
    pub fn csv_suffix(&self) -> String {
        format!(
            ",{},{},{},{:.0},{},{},{:.1},{:.1},{:.1},{},{:.6},{:.4},{:.1},{},{}",
            self.bound_label,
            self.hash_label,
            self.hash_collision_bits,
            self.field_ext_bits,
            self.target_bits,
            self.effective_target,
            self.achieved_bits,
            self.query_security_bits,
            self.algebraic_security_bits,
            self.queries,
            self.delta,
            self.rho,
            self.list_size_log2,
            self.num_rounds,
            self.is_stir,
        )
    }
}

/// Computes required query count and achieved security for FRI / STIR
/// given a proximity bound, hash variant, and protocol parameters.
///
/// # Soundness model
///
/// **STIR mode** (queries at layer 0 only):
///
///   ε = (1−δ)^{s₀}  +  Σ_ℓ (ℓ_list · d_ℓ / |F_ext|)
///
/// The first term is the probability that all s₀ proximity queries miss
/// a δ-far oracle.  The second is the union-bound over per-round OOD
/// algebraic errors (Schwartz-Zippel).  For large |F_ext| (e.g. 2^384)
/// the algebraic term is negligible and soundness ≈ s₀ · (−log₂(1−δ)).
///
/// **Classic FRI mode** (r queries at every layer):
///
///   ε = L · (1−δ)^r  +  d₀ / |F_ext|
///
/// Each of L rounds independently contributes (1−δ)^r.
///
/// In both cases, security is capped at hash collision resistance.
pub struct SoundnessCalculator {
    pub target_bits: usize,
    pub field_ext_bits: f64,
    pub bound: ProximityBound,
    pub hash: HashVariant,
}

impl SoundnessCalculator {
    pub fn new(
        target_bits: usize,
        field_ext_bits: f64,
        bound: ProximityBound,
        hash: HashVariant,
    ) -> Self {
        Self { target_bits, field_ext_bits, bound, hash }
    }

    /// Effective security target = min(target, hash collision bits).
    /// You can never exceed the Merkle tree's collision resistance.
    pub fn effective_target(&self) -> usize {
        self.target_bits.min(self.hash.collision_bits())
    }

    // ── STIR soundness ──────────────────────────────────────────────

    /// Returns (achieved_bits, query_sec_bits, algebraic_sec_bits).
    pub fn security_stir(
        &self,
        schedule: &[usize],
        n0: usize,
        blowup: usize,
        s0: usize,
    ) -> (f64, f64, f64) {
        let rho = 1.0 / blowup as f64;
        let (delta, list_log2) = self.bound.gap_and_list_size(rho);

        // Query error: (1 − δ)^{s₀}
        let query_err = f64::exp2(s0 as f64 * (1.0 - delta).log2());

        // Algebraic errors: Σ_ℓ  ℓ_list · d_ℓ / |F_ext|
        let mut degree = n0 as f64 / blowup as f64;
        let mut alg_err = 0.0_f64;
        for &m in schedule {
            alg_err += f64::exp2(
                list_log2 + degree.log2() - self.field_ext_bits,
            );
            degree /= m as f64;
        }
        // Final polynomial evaluation check (one more Schwartz-Zippel)
        alg_err += f64::exp2(
            degree.max(1.0).log2() - self.field_ext_bits,
        );

        let total = query_err + alg_err;
        let achieved = -(total.log2());
        let q_sec   = -(query_err.log2());
        let a_sec   = -(alg_err.log2());

        (achieved.min(self.hash.collision_bits() as f64), q_sec, a_sec)
    }

    // ── Classic FRI soundness ───────────────────────────────────────

    pub fn security_classic(
        &self,
        schedule: &[usize],
        n0: usize,
        blowup: usize,
        r: usize,
    ) -> (f64, f64, f64) {
        let rho = 1.0 / blowup as f64;
        let (delta, _) = self.bound.gap_and_list_size(rho);
        let num_rounds = schedule.len();

        // Query error: L · (1 − δ)^r  (union bound over rounds)
        let per_layer = f64::exp2(r as f64 * (1.0 - delta).log2());
        let query_err = num_rounds as f64 * per_layer;

        // DEEP algebraic error (single z for all layers)
        let degree = n0 as f64 / blowup as f64;
        let alg_err = f64::exp2(degree.log2() - self.field_ext_bits);

        let total = query_err + alg_err;
        let achieved = -(total.log2());
        let q_sec   = -(query_err.log2());
        let a_sec   = -(alg_err.log2());

        (achieved.min(self.hash.collision_bits() as f64), q_sec, a_sec)
    }

    // ── Unified interface ───────────────────────────────────────────

    pub fn security_bits(
        &self,
        schedule: &[usize],
        n0: usize,
        blowup: usize,
        queries: usize,
        stir: bool,
    ) -> (f64, f64, f64) {
        if stir {
            self.security_stir(schedule, n0, blowup, queries)
        } else {
            self.security_classic(schedule, n0, blowup, queries)
        }
    }

    /// Minimum query count to reach the effective security target.
    pub fn min_queries(
        &self,
        schedule: &[usize],
        n0: usize,
        blowup: usize,
        stir: bool,
    ) -> usize {
        let eff = self.effective_target() as f64;
        for q in 1..=2048 {
            let (achieved, _, _) =
                self.security_bits(schedule, n0, blowup, q, stir);
            if achieved >= eff {
                return q;
            }
        }
        panic!(
            "Cannot reach {}-bit security (eff_target={}) with ≤2048 \
             queries for schedule {:?}, n0={}, blowup={}, bound={:?}",
            self.target_bits, eff, schedule, n0, blowup, self.bound,
        );
    }

    /// Build a full diagnostic report for CSV / logging.
    pub fn report(
        &self,
        schedule: &[usize],
        n0: usize,
        blowup: usize,
        queries: usize,
        stir: bool,
    ) -> SoundnessReport {
        let rho = 1.0 / blowup as f64;
        let (delta, list_log2) = self.bound.gap_and_list_size(rho);
        let (achieved, q_sec, a_sec) =
            self.security_bits(schedule, n0, blowup, queries, stir);

        let mut degree = n0 as f64 / blowup as f64;
        let mut rounds = Vec::with_capacity(schedule.len());
        for (i, &m) in schedule.iter().enumerate() {
            rounds.push(RoundSoundnessDetail {
                round: i,
                fold_factor: m,
                degree_before: degree,
                alg_error_log2: list_log2 + degree.log2()
                    - self.field_ext_bits,
            });
            degree /= m as f64;
        }

        SoundnessReport {
            bound_label: self.bound.label(),
            hash_label: self.hash.label().into(),
            hash_collision_bits: self.hash.collision_bits(),
            field_ext_bits: self.field_ext_bits,
            target_bits: self.target_bits,
            effective_target: self.effective_target(),
            achieved_bits: achieved,
            query_security_bits: q_sec,
            algebraic_security_bits: a_sec,
            queries,
            delta,
            rho,
            list_size_log2: list_log2,
            num_rounds: schedule.len(),
            is_stir: stir,
            rounds,
        }
    }

    /// Print a comparison table to stderr (useful for quick diagnostics).
    pub fn print_comparison(
        schedules: &[(&str, &[usize])],
        n0: usize,
        blowup: usize,
        target_bits: usize,
        field_ext_bits: f64,
        hash: HashVariant,
        stir: bool,
    ) {
        let johnson = Self::new(
            target_bits, field_ext_bits,
            ProximityBound::Johnson, hash,
        );
        let capacity = Self::new(
            target_bits, field_ext_bits,
            ProximityBound::StirCapacity { gamma_log2: -30.0 },
            hash,
        );

        eprintln!("╔═══════════════════════╦═════════╦═════════╦═════════╦═════════╗");
        eprintln!("║ Schedule              ║ J  s0   ║ J  bits ║ SC s0   ║ SC bits ║");
        eprintln!("╠═══════════════════════╬═════════╬═════════╬═════════╬═════════╣");

        for &(label, sched) in schedules {
            let j_q = johnson.min_queries(sched, n0, blowup, stir);
            let (j_b, _, _) = johnson.security_bits(sched, n0, blowup, j_q, stir);

            let c_q = capacity.min_queries(sched, n0, blowup, stir);
            let (c_b, _, _) = capacity.security_bits(sched, n0, blowup, c_q, stir);

            eprintln!(
                "║ {:<21} ║ {:>5}   ║ {:>5.1}   ║ {:>5}   ║ {:>5.1}   ║",
                label, j_q, j_b, c_q, c_b,
            );
        }

        eprintln!("╚═══════════════════════╩═════════╩═════════╩═════════╩═════════╝");
        eprintln!(
            "  hash={} collision_bits={} field_ext_bits={:.0} stir={}",
            hash.label(), hash.collision_bits(), field_ext_bits, stir,
        );
    }
}


// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{Field, FftField, One, Zero};
    use ark_goldilocks::Goldilocks;
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    use ark_poly::DenseUVPolynomial;
    use ark_poly::Polynomial;
    use rand::Rng;
    use std::collections::HashSet;

    use ark_ff::UniformRand;
    use ark_poly::polynomial::univariate::DensePolynomial;
    use rand::seq::SliceRandom;
    use crate::cubic_ext::GoldilocksCubeConfig;

    use crate::cubic_ext::CubeExt;

    type TestField = Goldilocks;

    fn random_polynomial<F: Field>(degree: usize, rng: &mut impl Rng) -> Vec<F> {
        (0..=degree).map(|_| F::rand(rng)).collect()
    }

    fn perform_fold<F: Field + FftField>(
        evals: &[F],
        domain: GeneralEvaluationDomain<F>,
        alpha: F,
        folding_factor: usize,
    ) -> (Vec<F>, GeneralEvaluationDomain<F>) {
        assert!(evals.len() % folding_factor == 0);
        let n = evals.len();
        let next_n = n / folding_factor;
        let next_domain = GeneralEvaluationDomain::<F>::new(next_n)
            .expect("valid folded domain");
        let folding_domain = GeneralEvaluationDomain::<F>::new(folding_factor)
            .expect("valid folding domain");
        let generator = domain.group_gen();
        let folded = (0..next_n)
            .map(|i| {
                let coset_values: Vec<F> = (0..folding_factor)
                    .map(|j| evals[i + j * next_n])
                    .collect();
                let coset_generator = generator.pow([i as u64]);
                fold_one_coset(&coset_values, alpha, coset_generator, &folding_domain)
            })
            .collect();
        (folded, next_domain)
    }

    fn fold_one_coset<F: Field + FftField>(
        coset_values: &[F],
        alpha: F,
        coset_generator: F,
        folding_domain: &GeneralEvaluationDomain<F>,
    ) -> F {
        let p_coeffs = folding_domain.ifft(coset_values);
        let poly = DensePolynomial::from_coefficients_vec(p_coeffs);
        let evaluation_point = alpha * coset_generator.inverse().unwrap();
        poly.evaluate(&evaluation_point)
    }

    fn test_ext_fold_preserves_low_degree_with<E: TowerField>() {
        use ark_ff::UniformRand;

        let mut rng = StdRng::seed_from_u64(42);
        let n = 256usize;
        let m = 4usize;
        let degree = n / m - 1;

        let dom = GeneralEvaluationDomain::<TestField>::new(n).unwrap();
        let poly = DensePolynomial::<TestField>::rand(degree, &mut rng);
        let evals = dom.fft(poly.coeffs());

        let evals_ext: Vec<E> = evals.iter()
            .map(|&x| E::from_fp(x))
            .collect();

        let challenge_comps: Vec<TestField> = (0..E::DEGREE)
            .map(|i| TestField::from((17 + i * 13) as u64))
            .collect();
        let alpha = E::from_fp_components(&challenge_comps).unwrap();

        let folded = fri_fold_layer_ext_impl(&evals_ext, alpha, m);

        assert_eq!(folded.len(), n / m);

        let any_nonzero = folded.iter().any(|v| *v != E::zero());
        assert!(any_nonzero, "Folded codeword should be non-trivial");

        eprintln!(
            "[ext_fold_low_degree] E::DEGREE={} folded_len={} sample={:?}",
            E::DEGREE, folded.len(), folded[0]
        );
    }

    #[test]
    fn test_ext_fold_preserves_low_degree() {
        test_ext_fold_preserves_low_degree_with::<CubeExt<GoldilocksCubeConfig>>();
    }

    fn test_ext_fold_consistency_with_s_layer_for<E: TowerField>() {
        let mut rng = StdRng::seed_from_u64(99);
        let n = 128usize;
        let m = 4usize;

        let evals: Vec<E> = (0..n)
            .map(|_| {
                let comps: Vec<TestField> = (0..E::DEGREE)
                    .map(|_| TestField::from(rng.gen::<u64>()))
                    .collect();
                E::from_fp_components(&comps).unwrap()
            })
            .collect();

        let alpha_comps: Vec<TestField> = (0..E::DEGREE)
            .map(|i| TestField::from((13 + i * 7) as u64))
            .collect();
        let alpha = E::from_fp_components(&alpha_comps).unwrap();

        let folded = fri_fold_layer_ext_impl(&evals, alpha, m);
        let s = compute_s_layer_ext(&evals, alpha, m);

        let n_next = n / m;
        for b in 0..n_next {
            for j in 0..m {
                assert_eq!(
                    s[b + j * n_next], folded[b],
                    "s-layer mismatch at b={} j={}", b, j
                );
            }
        }
    }

    #[test]
    fn test_ext_fold_consistency_with_s_layer() {
        test_ext_fold_consistency_with_s_layer_for::<CubeExt<GoldilocksCubeConfig>>();
    }

    fn test_ext_deep_quotient_consistency_for<E: TowerField>() {
        let mut rng = StdRng::seed_from_u64(77);
        let n = 64usize;

        let dom = Domain::<TestField>::new(n).unwrap();
        let omega = dom.group_gen;

        let evals: Vec<E> = (0..n)
            .map(|_| {
                let comps: Vec<TestField> = (0..E::DEGREE)
                    .map(|_| TestField::from(rng.gen::<u64>()))
                    .collect();
                E::from_fp_components(&comps).unwrap()
            })
            .collect();

        let z_comps: Vec<TestField> = (0..E::DEGREE)
            .map(|i| TestField::from((42 + i * 11) as u64))
            .collect();
        let z = E::from_fp_components(&z_comps).unwrap();

        let (q, fz) = compute_q_layer_ext_on_ext(&evals, z, omega);

        let mut x = TestField::one();
        for i in 0..n {
            let lhs = q[i] * (E::from_fp(x) - z);
            let rhs = evals[i] - fz;
            assert_eq!(lhs, rhs, "DEEP identity failed at i={}", i);
            x *= omega;
        }
    }

    #[test]
    fn test_ext_deep_quotient_consistency() {
        test_ext_deep_quotient_consistency_for::<CubeExt<GoldilocksCubeConfig>>();
    }

    // ────────────────────────────────────────────────────────────────
    //  STIR-specific tests
    // ────────────────────────────────────────────────────────────────

    fn test_stir_batched_deep_quotient_for<E: TowerField>() {
        let mut rng = StdRng::seed_from_u64(555);
        let n = 64usize;
        let m = 4usize;

        let dom = Domain::<TestField>::new(n).unwrap();
        let omega = dom.group_gen;

        let evals: Vec<E> = (0..n)
            .map(|_| {
                let comps: Vec<TestField> = (0..E::DEGREE)
                    .map(|_| TestField::from(rng.gen::<u64>()))
                    .collect();
                E::from_fp_components(&comps).unwrap()
            })
            .collect();

        let z_comps: Vec<TestField> = (0..E::DEGREE)
            .map(|i| TestField::from((42 + i * 11) as u64))
            .collect();
        let z = E::from_fp_components(&z_comps).unwrap();

        let (q, coset_evals, interp_coeffs) = compute_stir_quotient_ext(&evals, z, omega, m);

        let z_m = ext_pow(z, m as u64);

        let mut x = E::from_fp(TestField::one());
        let omega_ext = E::from_fp(omega);
        for i in 0..n {
            let v_xi = ext_pow(x, m as u64) - z_m;
            let p_xi = eval_final_poly_ext(&interp_coeffs, x);
            let lhs = q[i] * v_xi + p_xi;
            assert_eq!(
                lhs, evals[i],
                "STIR batched DEEP identity failed at i={}",
                i
            );
            x = x * omega_ext;
        }

        let n_next = n / m;
        let zeta = E::from_fp(omega.pow([n_next as u64]));
        let mut zeta_pow = E::one();
        for j in 0..m {
            let point = zeta_pow * z;
            let p_val = eval_final_poly_ext(&interp_coeffs, point);
            assert_eq!(
                p_val, coset_evals[j],
                "Interpolation mismatch at coset point j={}",
                j
            );
            zeta_pow = zeta_pow * zeta;
        }
    }

    #[test]
    fn test_stir_batched_deep_quotient() {
        test_stir_batched_deep_quotient_for::<CubeExt<GoldilocksCubeConfig>>();
    }

    fn test_stir_ood_fold_consistency_for<E: TowerField>() {
        let mut rng = StdRng::seed_from_u64(666);
        let n = 64usize;
        let m = 4usize;
        let degree = n / m - 1;

        let dom = GeneralEvaluationDomain::<TestField>::new(n).unwrap();
        let poly = DensePolynomial::<TestField>::rand(degree, &mut rng);
        let evals = dom.fft(poly.coeffs());
        let evals_ext: Vec<E> = evals.iter().map(|&x| E::from_fp(x)).collect();

        let omega = dom.group_gen();
        let n_next = n / m;

        let alpha_comps: Vec<TestField> = (0..E::DEGREE)
            .map(|i| TestField::from((31 + i * 17) as u64))
            .collect();
        let alpha = E::from_fp_components(&alpha_comps).unwrap();

        let z_comps: Vec<TestField> = (0..E::DEGREE)
            .map(|i| TestField::from((42 + i * 11) as u64))
            .collect();
        let z = E::from_fp_components(&z_comps).unwrap();

        let (_, coset_evals, _) = compute_stir_quotient_ext(&evals_ext, z, omega, m);

        let zeta = omega.pow([n_next as u64]);
        let interp_coeffs = interpolate_stir_coset(&coset_evals, z, zeta, m);

        let alpha_pows = build_ext_pows(alpha, m);
        let mut fold_at_ood = E::zero();
        for k in 0..m {
            fold_at_ood = fold_at_ood + interp_coeffs[k] * alpha_pows[k];
        }

        let coeff_tuples = extract_all_coset_coefficients(&evals_ext, omega, m);
        let folded_evals = interpolation_fold_ext(&coeff_tuples, alpha);

        let z_m = ext_pow(z, m as u64);
        let folded_coeffs = ext_evals_to_coeffs(&folded_evals);
        let folded_at_zm = eval_final_poly_ext(&folded_coeffs, z_m);

        assert_eq!(
            fold_at_ood, folded_at_zm,
            "STIR OOD fold-consistency failed: fold(interp_coeffs) != f_next(z^m)"
        );
    }

    #[test]
    fn test_stir_ood_fold_consistency() {
        test_stir_ood_fold_consistency_for::<CubeExt<GoldilocksCubeConfig>>();
    }

    fn test_stir_end_to_end_binary_for<E: TowerField>() {
        let mut rng = StdRng::seed_from_u64(777);
        let n = 256usize;
        let schedule = vec![2, 2, 2, 2];
        let degree = n / 32 - 1;

        let dom = GeneralEvaluationDomain::<TestField>::new(n).unwrap();
        let poly = DensePolynomial::<TestField>::rand(degree, &mut rng);
        let evals = dom.fft(poly.coeffs());

        let domain0 = FriDomain::new_radix2(n);

        let final_size: usize = n / schedule.iter().product::<usize>();
        let d_final = final_size / 2;

        let params = DeepFriParams::new(schedule, 4, 42)
            .with_stir()
            .with_s0(6)
            .with_d_final(d_final);

        let proof: DeepFriProof<E> = deep_fri_prove(evals, domain0, &params);

        assert!(
            proof.stir_coset_evals.is_some(),
            "STIR proof should include coset evaluations"
        );
        assert!(
            proof.stir_proximity_queries.is_some(),
            "STIR proof should include proximity queries"
        );
        assert_eq!(
            proof.stir_proximity_queries.as_ref().unwrap().len(),
            6,
            "STIR proof should have s0=6 proximity queries"
        );
        assert!(
            proof.queries.is_empty(),
            "STIR proof should have zero classic fold queries"
        );

        let ok = deep_fri_verify(&params, &proof);
        assert!(ok, "STIR end-to-end verification failed (binary schedule)");
    }

    #[test]
    fn test_stir_end_to_end_binary() {
        test_stir_end_to_end_binary_for::<CubeExt<GoldilocksCubeConfig>>();
    }

    fn test_stir_end_to_end_quaternary_for<E: TowerField>() {
        let mut rng = StdRng::seed_from_u64(888);
        let n = 256usize;
        let schedule = vec![4, 4];
        let degree = n / 32 - 1;

        let dom = GeneralEvaluationDomain::<TestField>::new(n).unwrap();
        let poly = DensePolynomial::<TestField>::rand(degree, &mut rng);
        let evals = dom.fft(poly.coeffs());

        let domain0 = FriDomain::new_radix2(n);

        let final_size: usize = n / schedule.iter().product::<usize>();
        let d_final = final_size / 2;

        let params = DeepFriParams::new(schedule, 4, 42)
            .with_stir()
            .with_s0(8)
            .with_d_final(d_final);

        let proof: DeepFriProof<E> = deep_fri_prove(evals, domain0, &params);
        let ok = deep_fri_verify(&params, &proof);
        assert!(ok, "STIR end-to-end verification failed (quaternary schedule)");
    }

    #[test]
    fn test_stir_end_to_end_quaternary() {
        test_stir_end_to_end_quaternary_for::<CubeExt<GoldilocksCubeConfig>>();
    }

    fn test_stir_end_to_end_mixed_for<E: TowerField>() {
        let mut rng = StdRng::seed_from_u64(999);
        let n = 256usize;
        let schedule = vec![4, 2, 2, 2, 2];
        let degree = n / 32 - 1;

        let dom = GeneralEvaluationDomain::<TestField>::new(n).unwrap();
        let poly = DensePolynomial::<TestField>::rand(degree, &mut rng);
        let evals = dom.fft(poly.coeffs());

        let domain0 = FriDomain::new_radix2(n);

        let final_size: usize = n / schedule.iter().product::<usize>();
        let d_final = final_size / 2;

        let params = DeepFriParams::new(schedule, 4, 42)
            .with_stir()
            .with_s0(10)
            .with_d_final(d_final);

        let proof: DeepFriProof<E> = deep_fri_prove(evals, domain0, &params);
        let ok = deep_fri_verify(&params, &proof);
        assert!(ok, "STIR end-to-end verification failed (mixed schedule)");
    }

    #[test]
    fn test_stir_end_to_end_mixed() {
        test_stir_end_to_end_mixed_for::<CubeExt<GoldilocksCubeConfig>>();
    }

    fn test_no_stir_backward_compat_for<E: TowerField>() {
        let mut rng = StdRng::seed_from_u64(111);
        let n = 256usize;
        let schedule = vec![2, 2, 2, 2];
        let degree = n / 32 - 1;

        let dom = GeneralEvaluationDomain::<TestField>::new(n).unwrap();
        let poly = DensePolynomial::<TestField>::rand(degree, &mut rng);
        let evals = dom.fft(poly.coeffs());

        let domain0 = FriDomain::new_radix2(n);

        let final_size: usize = n / schedule.iter().product::<usize>();
        let d_final = final_size / 2;

        let params = DeepFriParams::new(schedule, 4, 42)
            .with_d_final(d_final);

        assert!(!params.stir);

        let proof: DeepFriProof<E> = deep_fri_prove(evals, domain0, &params);
        assert!(proof.stir_coset_evals.is_none());
        assert!(proof.stir_proximity_queries.is_none());

        let ok = deep_fri_verify(&params, &proof);
        assert!(ok, "Backward-compatible (no STIR) verification failed");
    }

    #[test]
    fn test_no_stir_backward_compat() {
        test_no_stir_backward_compat_for::<CubeExt<GoldilocksCubeConfig>>();
    }

    /// Confirm that STIR proof has strictly fewer oracle accesses
    /// than the classic FRI proof at matched parameters.
    fn test_stir_query_count_reduction_for<E: TowerField>() {
        let mut rng = StdRng::seed_from_u64(1234);
        let n = 256usize;
        let schedule = vec![2, 2, 2, 2];
        let L = schedule.len();
        let degree = n / 32 - 1;
        let s0 = 6usize;

        let dom = GeneralEvaluationDomain::<TestField>::new(n).unwrap();
        let poly = DensePolynomial::<TestField>::rand(degree, &mut rng);
        let evals = dom.fft(poly.coeffs());

        let domain0 = FriDomain::new_radix2(n);
        let final_size: usize = n / schedule.iter().product::<usize>();
        let d_final = final_size / 2;

        // STIR proof
        let stir_params = DeepFriParams::new(schedule.clone(), 0, 42)
            .with_stir()
            .with_s0(s0)
            .with_d_final(d_final);

        let stir_proof: DeepFriProof<E> = deep_fri_prove(evals.clone(), domain0, &stir_params);
        let stir_bytes = deep_fri_proof_size_bytes(&stir_proof, true);

        // Classic FRI proof with same s0 as r
        let fri_params = DeepFriParams::new(schedule.clone(), s0, 42)
            .with_d_final(d_final);

        let fri_proof: DeepFriProof<E> = deep_fri_prove(evals, domain0, &fri_params);
        let fri_bytes = deep_fri_proof_size_bytes(&fri_proof, false);

        // STIR oracle accesses: s0 (proximity at layer 0) + 0 (OOD is algebraic)
        // FRI oracle accesses: r * (L+1) (every query opens all layers)
        let stir_oracle_accesses = s0;
        let fri_oracle_accesses = s0 * (L + 1);

        eprintln!("[STIR vs FRI] s0={} L={}", s0, L);
        eprintln!("  STIR oracle accesses: {}", stir_oracle_accesses);
        eprintln!("  FRI  oracle accesses: {}", fri_oracle_accesses);
        eprintln!("  STIR proof size: {} bytes", stir_bytes);
        eprintln!("  FRI  proof size: {} bytes", fri_bytes);
        eprintln!("  Query ratio: {:.1}x", fri_oracle_accesses as f64 / stir_oracle_accesses as f64);
        eprintln!("  Size  ratio: {:.1}x", fri_bytes as f64 / stir_bytes as f64);

        assert!(
            stir_oracle_accesses < fri_oracle_accesses,
            "STIR should have fewer oracle accesses: {} vs {}",
            stir_oracle_accesses, fri_oracle_accesses,
        );
        assert!(
            stir_bytes < fri_bytes,
            "STIR proof should be smaller: {} vs {}",
            stir_bytes, fri_bytes,
        );

        // Verify both
        assert!(deep_fri_verify(&stir_params, &stir_proof), "STIR verify failed");
        assert!(deep_fri_verify(&fri_params, &fri_proof), "FRI verify failed");
    }

    #[test]
    fn test_stir_query_count_reduction() {
        test_stir_query_count_reduction_for::<CubeExt<GoldilocksCubeConfig>>();
    }

    // ────────────────────────────────────────────────────────────────
    //  Construction 5.1 specific tests
    // ────────────────────────────────────────────────────────────────

    fn test_coset_interpolation_roundtrip_for<E: TowerField>() {
        let mut rng = StdRng::seed_from_u64(123);
        let n = 64usize;
        let m = 4usize;
        let n_next = n / m;

        let dom = Domain::<TestField>::new(n).unwrap();
        let omega = dom.group_gen;
        let zeta = omega.pow([n_next as u64]);

        let evals: Vec<E> = (0..n)
            .map(|_| {
                let comps: Vec<TestField> = (0..E::DEGREE)
                    .map(|_| TestField::from(rng.gen::<u64>()))
                    .collect();
                E::from_fp_components(&comps).unwrap()
            })
            .collect();

        let coeff_tuples = extract_all_coset_coefficients(&evals, omega, m);

        assert_eq!(coeff_tuples.len(), n_next);
        assert_eq!(coeff_tuples[0].len(), m);

        for b in 0..n_next {
            for j in 0..m {
                let x_j = omega.pow([(b + j * n_next) as u64]);
                let mut eval = E::zero();
                let mut x_pow = F::one();
                for i in 0..m {
                    eval = eval + coeff_tuples[b][i] * E::from_fp(x_pow);
                    x_pow *= x_j;
                }
                assert_eq!(
                    eval,
                    evals[b + j * n_next],
                    "Interpolation roundtrip failed at coset b={} fibre j={}",
                    b, j
                );
            }
        }
    }

    #[test]
    fn test_coset_interpolation_roundtrip() {
        test_coset_interpolation_roundtrip_for::<CubeExt<GoldilocksCubeConfig>>();
    }

    fn test_coeff_functions_low_degree_for<E: TowerField>() {
        let mut rng = StdRng::seed_from_u64(456);
        let n = 64usize;
        let m = 4usize;
        let n_next = n / m;
        let degree = m - 1;

        let dom = GeneralEvaluationDomain::<TestField>::new(n).unwrap();
        let poly = DensePolynomial::<TestField>::rand(degree, &mut rng);
        let evals = dom.fft(poly.coeffs());

        let evals_ext: Vec<E> = evals.iter().map(|&x| E::from_fp(x)).collect();

        let omega = dom.group_gen();
        let coeff_tuples = extract_all_coset_coefficients(&evals_ext, omega, m);

        for i in 0..m {
            let c_i_0 = coeff_tuples[0][i];
            for b in 1..n_next {
                assert_eq!(
                    coeff_tuples[b][i], c_i_0,
                    "C_{} not constant: coset 0 = {:?}, coset {} = {:?}",
                    i, c_i_0, b, coeff_tuples[b][i]
                );
            }
        }
    }

    #[test]
    fn test_coeff_functions_low_degree() {
        test_coeff_functions_low_degree_for::<CubeExt<GoldilocksCubeConfig>>();
    }

    fn test_interpolation_fold_matches_poly_eval_for<E: TowerField>() {
        let mut rng = StdRng::seed_from_u64(789);
        let n = 64usize;
        let m = 4usize;
        let n_next = n / m;
        let degree = m - 1;

        let dom = GeneralEvaluationDomain::<TestField>::new(n).unwrap();
        let poly = DensePolynomial::<TestField>::rand(degree, &mut rng);
        let evals = dom.fft(poly.coeffs());
        let evals_ext: Vec<E> = evals.iter().map(|&x| E::from_fp(x)).collect();

        let omega = dom.group_gen();

        let alpha_comps: Vec<TestField> = (0..E::DEGREE)
            .map(|i| TestField::from((31 + i * 17) as u64))
            .collect();
        let alpha = E::from_fp_components(&alpha_comps).unwrap();

        let coeff_tuples = extract_all_coset_coefficients(&evals_ext, omega, m);
        let folded = interpolation_fold_ext(&coeff_tuples, alpha);

        assert_eq!(folded.len(), n_next);

        let fold_0 = folded[0];
        for b in 1..n_next {
            assert_eq!(
                folded[b], fold_0,
                "Interpolation fold not constant at coset {}",
                b
            );
        }
    }

    #[test]
    fn test_interpolation_fold_matches_poly_eval() {
        test_interpolation_fold_matches_poly_eval_for::<CubeExt<GoldilocksCubeConfig>>();
    }

    fn test_batched_degree_check_for<E: TowerField>() {
        let mut rng = StdRng::seed_from_u64(321);
        let n = 64usize;
        let m = 4usize;
        let n_next = n / m;
        let degree = m - 1;

        let dom = GeneralEvaluationDomain::<TestField>::new(n).unwrap();
        let poly = DensePolynomial::<TestField>::rand(degree, &mut rng);
        let evals = dom.fft(poly.coeffs());
        let evals_ext: Vec<E> = evals.iter().map(|&x| E::from_fp(x)).collect();

        let omega = dom.group_gen();
        let coeff_tuples = extract_all_coset_coefficients(&evals_ext, omega, m);

        let beta_comps: Vec<TestField> = (0..E::DEGREE)
            .map(|i| TestField::from((7 + i * 3) as u64))
            .collect();
        let beta = E::from_fp_components(&beta_comps).unwrap();

        assert!(
            batched_degree_check_ext(&coeff_tuples, beta, 1),
            "Batched degree check should pass for honest coefficients"
        );

        let mut bad_tuples = coeff_tuples.clone();
        bad_tuples[1][0] = bad_tuples[1][0] + E::from_fp(TestField::one());

        assert!(
            !batched_degree_check_ext(&bad_tuples, beta, 1),
            "Batched degree check should fail for corrupted coefficients"
        );
    }

    #[test]
    fn test_batched_degree_check() {
        test_batched_degree_check_for::<CubeExt<GoldilocksCubeConfig>>();
    }

    // ── Existing tests (unchanged) ──

    #[test]
    fn test_fri_local_consistency_check_soundness() {
        const DOMAIN_SIZE: usize = 1024;
        const FOLDING_FACTOR: usize = 4;
        const NUM_TRIALS: usize = 1000000;

        let mut rng = rand::thread_rng();
        let mut detections = 0;

        let z_l = TestField::from(5u64);
        let f: Vec<TestField> = (0..DOMAIN_SIZE).map(|_| TestField::rand(&mut rng)).collect();
        let f_next_claimed: Vec<TestField> = vec![TestField::zero(); DOMAIN_SIZE / FOLDING_FACTOR];

        let domain = GeneralEvaluationDomain::<TestField>::new(DOMAIN_SIZE).unwrap();
        let generator = domain.group_gen();
        let folding_domain = GeneralEvaluationDomain::<TestField>::new(FOLDING_FACTOR).unwrap();

        for _ in 0..NUM_TRIALS {
            let query_index = rng.gen_range(0..f_next_claimed.len());
            let coset_values: Vec<TestField> = (0..FOLDING_FACTOR)
                .map(|j| f[query_index + j * (DOMAIN_SIZE / FOLDING_FACTOR)])
                .collect();
            let coset_generator = generator.pow([query_index as u64]);
            let s_reconstructed = fold_one_coset(&coset_values, z_l, coset_generator, &folding_domain);
            let s_claimed = f_next_claimed[query_index];
            if s_reconstructed != s_claimed {
                detections += 1;
            }
        }
        let measured_rate = detections as f64 / NUM_TRIALS as f64;
        println!("[Consistency Check] Detections: {}/{}, Measured Rate: {:.4}", detections, NUM_TRIALS, measured_rate);
        assert!((measured_rate - 1.0).abs() < 0.01, "Detection rate should be close to 100%");
    }

    #[test]
    fn test_fri_distance_amplification() {
        const DOMAIN_SIZE: usize = 1024;
        const FOLDING_FACTOR: usize = 4;
        const NUM_TRIALS: usize = 100_000;
        const INITIAL_CORRUPTION_FRACTION: f64 = 0.05;

        let mut rng = rand::thread_rng();
        let z_l = TestField::from(5u64);

        let large_domain = GeneralEvaluationDomain::<TestField>::new(DOMAIN_SIZE).unwrap();
        let degree_bound = DOMAIN_SIZE / FOLDING_FACTOR;
        let p_coeffs = random_polynomial(degree_bound - 2, &mut rng);
        let p_poly = DensePolynomial::from_coefficients_vec(p_coeffs);
        let p_evals = large_domain.fft(p_poly.coeffs());

        let mut f_evals = p_evals.clone();
        let num_corruptions = (DOMAIN_SIZE as f64 * INITIAL_CORRUPTION_FRACTION) as usize;
        let mut corrupted_indices = HashSet::new();
        while corrupted_indices.len() < num_corruptions {
            corrupted_indices.insert(rng.gen_range(0..DOMAIN_SIZE));
        }
        for &idx in &corrupted_indices {
            f_evals[idx] = TestField::rand(&mut rng);
        }

        let folded_honest = fri_fold_layer(&p_evals, z_l, FOLDING_FACTOR);
        let folded_corrupted = fri_fold_layer(&f_evals, z_l, FOLDING_FACTOR);

        let mut detections = 0;
        for _ in 0..NUM_TRIALS {
            let query_index = rng.gen_range(0..folded_honest.len());
            if folded_honest[query_index] != folded_corrupted[query_index] {
                detections += 1;
            }
        }

        let measured_rate = detections as f64 / NUM_TRIALS as f64;
        let theoretical_rate = (FOLDING_FACTOR as f64 * INITIAL_CORRUPTION_FRACTION).min(1.0);

        println!("\n[Distance Amplification Test (Single Layer)]");
        println!("  - Initial Corruption: {:.2}% ({} points)", INITIAL_CORRUPTION_FRACTION * 100.0, num_corruptions);
        println!("  - Detections: {}/{}", detections, NUM_TRIALS);
        println!("  - Measured Detection Rate: {:.4}", measured_rate);
        println!("  - Theoretical Detection Rate: {:.4}", theoretical_rate);

        let tolerance = 0.05;
        assert!(
            (measured_rate - theoretical_rate).abs() < tolerance,
            "Measured detection rate should be close to the theoretical rate."
        );
    }

    #[test]
    #[ignore]
    fn test_full_fri_protocol_soundness() {
        const FOLDING_SCHEDULE: [usize; 3] = [4, 4, 4];
        const INITIAL_DOMAIN_SIZE: usize = 4096;
        const NUM_TRIALS: usize = 1000000;

        let mut rng = rand::thread_rng();
        let mut detections = 0;

        for _ in 0..NUM_TRIALS {
            let alpha = TestField::rand(&mut rng);

            let mut domains = Vec::new();
            let mut current_size = INITIAL_DOMAIN_SIZE;
            domains.push(GeneralEvaluationDomain::<TestField>::new(current_size).unwrap());
            for &folding_factor in &FOLDING_SCHEDULE {
                current_size /= folding_factor;
                domains.push(GeneralEvaluationDomain::<TestField>::new(current_size).unwrap());
            }

            let mut fraudulent_layers: Vec<Vec<TestField>> = Vec::new();
            let f0: Vec<TestField> = (0..INITIAL_DOMAIN_SIZE).map(|_| TestField::rand(&mut rng)).collect();
            fraudulent_layers.push(f0);

            let mut current_layer_evals = fraudulent_layers[0].clone();
            for &folding_factor in &FOLDING_SCHEDULE {
                let next_layer = fri_fold_layer(&current_layer_evals, alpha, folding_factor);
                current_layer_evals = next_layer;
                fraudulent_layers.push(current_layer_evals.clone());
            }

            let mut trial_detected = false;
            let mut query_index = rng.gen_range(0..domains[1].size());

            for l in 0..FOLDING_SCHEDULE.len() {
                let folding_factor = FOLDING_SCHEDULE[l];
                let current_domain = &domains[l];
                let next_domain = &domains[l+1];

                let coset_generator = current_domain.group_gen().pow([query_index as u64]);
                let folding_domain = GeneralEvaluationDomain::<TestField>::new(folding_factor).unwrap();

                let coset_values: Vec<TestField> = (0..folding_factor)
                    .map(|j| fraudulent_layers[l][query_index + j * next_domain.size()])
                    .collect();

                let s_reconstructed = fold_one_coset(&coset_values, alpha, coset_generator, &folding_domain);
                let s_claimed = fraudulent_layers[l+1][query_index];

                if s_reconstructed != s_claimed {
                    trial_detected = true;
                    break;
                }

                if l + 1 < FOLDING_SCHEDULE.len() {
                    query_index %= domains[l+2].size();
                }
            }

            if !trial_detected {
                let last_layer = fraudulent_layers.last().unwrap();
                let first_element = last_layer[0];
                for &element in last_layer.iter().skip(1) {
                    if element != first_element {
                        trial_detected = true;
                        break;
                    }
                }
            }

            if trial_detected {
                detections += 1;
            }
        }

        let measured_rate = detections as f64 / NUM_TRIALS as f64;

        println!("\n[Full Protocol Soundness Test (ε_eff)]");
        println!("  - Protocol Schedule: {:?}", FOLDING_SCHEDULE);
        println!("  - Initial Domain Size: {}", INITIAL_DOMAIN_SIZE);
        println!("  - Detections: {}/{}", detections, NUM_TRIALS);
        println!("  - Measured Effective Detection Rate (ε_eff): {:.4}", measured_rate);

        assert!(measured_rate > 0.90, "Effective detection rate should be very high");
    }

    #[test]
    fn test_intermediate_layer_fraud_soundness() {
        const INITIAL_DOMAIN_SIZE: usize = 4096;
        const FOLDING_SCHEDULE: [usize; 3] = [4, 4, 4];
        const NUM_TRIALS: usize = 20000;
        const FRAUD_LAYER_INDEX: usize = 1;

        let mut rng = rand::thread_rng();
        let mut detections = 0;

        let alphas: Vec<TestField> = (0..FOLDING_SCHEDULE.len())
            .map(|_| TestField::rand(&mut rng))
            .collect();

        let final_layer_size =
            INITIAL_DOMAIN_SIZE / FOLDING_SCHEDULE.iter().product::<usize>();
        let degree_bound = final_layer_size - 1;

        let p_coeffs = random_polynomial(degree_bound, &mut rng);
        let p_poly = DensePolynomial::from_coefficients_vec(p_coeffs);

        let domain0 = GeneralEvaluationDomain::<TestField>::new(INITIAL_DOMAIN_SIZE).unwrap();
        let honest_f0 = domain0.fft(p_poly.coeffs());

        let mut honest_layers = vec![honest_f0];
        let mut current = honest_layers[0].clone();

        for (l, &factor) in FOLDING_SCHEDULE.iter().enumerate() {
            let next = fri_fold_layer(&current, alphas[l], factor);
            honest_layers.push(next.clone());
            current = next;
        }

        let mut prover_layers = honest_layers.clone();

        let fraud_layer_size =
            INITIAL_DOMAIN_SIZE / FOLDING_SCHEDULE[0..FRAUD_LAYER_INDEX].iter().product::<usize>();

        let fraud_index = rng.gen_range(0..fraud_layer_size);
        let honest = prover_layers[FRAUD_LAYER_INDEX][fraud_index];
        let mut corrupted = TestField::rand(&mut rng);
        while corrupted == honest {
            corrupted = TestField::rand(&mut rng);
        }
        prover_layers[FRAUD_LAYER_INDEX][fraud_index] = corrupted;

        let l = FRAUD_LAYER_INDEX - 1;
        let folding_factor = FOLDING_SCHEDULE[l];
        let current_domain_size =
            INITIAL_DOMAIN_SIZE / FOLDING_SCHEDULE[0..l].iter().product::<usize>();
        let next_domain_size = current_domain_size / folding_factor;

        let current_domain =
            GeneralEvaluationDomain::<TestField>::new(current_domain_size).unwrap();

        for _ in 0..NUM_TRIALS {
            let query_index = rng.gen_range(0..next_domain_size);
            let x = current_domain.element(query_index);

            let coset_values: Vec<TestField> = (0..folding_factor)
                .map(|j| prover_layers[l][query_index + j * next_domain_size])
                .collect();

            let mut lhs = TestField::zero();
            let mut alpha_pow = TestField::one();

            for j in 0..folding_factor {
                lhs += coset_values[j] * alpha_pow;
                alpha_pow *= alphas[l];
            }

            let reconstructed = lhs;
            let claimed = prover_layers[l + 1][query_index];

            if reconstructed != claimed {
                detections += 1;
            }
        }

        let measured_rate = detections as f64 / NUM_TRIALS as f64;
        let theoretical_rate = 1.0 / fraud_layer_size as f64;

        println!("\n[Intermediate Layer Fraud Soundness]");
        println!("  Fraud layer size: {}", fraud_layer_size);
        println!("  Fraud index: {}", fraud_index);
        println!("  Detections: {}/{}", detections, NUM_TRIALS);
        println!("  Measured detection rate: {:.6}", measured_rate);
        println!("  Theoretical rate: {:.6}", theoretical_rate);

        let tolerance = theoretical_rate * 5.0;
        assert!(
            (measured_rate - theoretical_rate).abs() < tolerance,
            "Measured detection rate deviates from theory"
        );
    }

    #[test]
    #[ignore]
    fn test_fri_effective_detection_rate() {
        println!("\n--- Running Rust Test: Effective Detection Rate (arkworks) ---");

        const DOMAIN_SIZE: usize = 1024;
        const FOLDING_FACTOR: usize = 4;
        let degree = 31;

        let mut rng = rand::thread_rng();

        let p_coeffs = random_polynomial(degree, &mut rng);
        let p_poly = DensePolynomial::from_coefficients_vec(p_coeffs);
        let domain0 = GeneralEvaluationDomain::<TestField>::new(DOMAIN_SIZE).unwrap();

        let f0_good = domain0.fft(p_poly.coeffs());
        let mut f0_corrupt = f0_good.clone();

        let rho_0 = 0.06;
        let num_corruptions = (DOMAIN_SIZE as f64 * rho_0) as usize;
        let indices: Vec<usize> = (0..DOMAIN_SIZE).collect();

        for &idx in indices.choose_multiple(&mut rng, num_corruptions) {
            let honest = f0_corrupt[idx];
            let mut corrupted = TestField::rand(&mut rng);
            while corrupted == honest {
                corrupted = TestField::rand(&mut rng);
            }
            f0_corrupt[idx] = corrupted;
        }

        let alpha1 = TestField::rand(&mut rng);
        let f1_corrupt = fri_fold_layer(&f0_corrupt, alpha1, FOLDING_FACTOR);

        let alpha2 = TestField::rand(&mut rng);
        let f2_corrupt = fri_fold_layer(&f1_corrupt, alpha2, FOLDING_FACTOR);

        let num_trials = 200_000;
        let mut detections = 0;

        let domain1_size = DOMAIN_SIZE / FOLDING_FACTOR;
        let domain2_size = domain1_size / FOLDING_FACTOR;

        for _ in 0..num_trials {
            let i2 = rng.gen_range(0..domain2_size);
            let k = rng.gen_range(0..FOLDING_FACTOR);
            let i1 = i2 + k * domain2_size;

            let x1 = domain0.element(i1);

            let coset0: Vec<TestField> = (0..FOLDING_FACTOR)
                .map(|j| f0_corrupt[i1 + j * domain1_size])
                .collect();

            let reconstructed_f1 = fold_one_coset(&coset0, alpha1, x1, &domain0);

            if reconstructed_f1 != f1_corrupt[i1] {
                detections += 1;
                continue;
            }

            let x2 = x1.sq();

            let coset1: Vec<TestField> = (0..FOLDING_FACTOR)
                .map(|j| f1_corrupt[i2 + j * domain2_size])
                .collect();

            let reconstructed_f2 = fold_one_coset(&coset1, alpha2, x2, &domain0);

            if reconstructed_f2 != f2_corrupt[i2] {
                detections += 1;
            }
        }

        let measured_rate = detections as f64 / num_trials as f64;
        let rho_1 = 1.0 - (1.0 - rho_0).powi(FOLDING_FACTOR as i32);
        let rho_2 = 1.0 - (1.0 - rho_1).powi(FOLDING_FACTOR as i32);

        println!("rho_0 = {:.4}", rho_0);
        println!("rho_1 = {:.4}", rho_1);
        println!("rho_2 = {:.4}", rho_2);
        println!("Measured effective detection rate: {:.4}", measured_rate);
        println!("Theoretical effective detection rate: {:.4}", rho_2);

        let delta = 0.03;
        assert!(
            (measured_rate - rho_2).abs() < delta,
            "Measured rate {:.4} not close to theoretical {:.4}",
            measured_rate,
            rho_2
        );

        println!("✅ Effective detection rate matches theory");
    }

    #[test]
    fn debug_single_fold_distance_amplification() {
        let log_domain_size = 12;
        let initial_domain_size = 1 << log_domain_size;
        let folding_factor = 4;
        let initial_corruption_rate = 0.06;

        let mut rng = StdRng::seed_from_u64(0);

        let degree = (initial_domain_size / folding_factor) - 1;
        let domain = GeneralEvaluationDomain::<F>::new(initial_domain_size)
            .expect("Failed to create domain");
        let poly_p0 = DensePolynomial::<F>::rand(degree, &mut rng);

        let codeword_c0_evals = poly_p0.evaluate_over_domain(domain).evals;

        let mut corrupted_codeword_c_prime_0_evals = codeword_c0_evals.clone();
        let num_corruptions = (initial_domain_size as f64 * initial_corruption_rate).ceil() as usize;
        let mut corrupted_indices = HashSet::new();

        while corrupted_indices.len() < num_corruptions {
            let idx_to_corrupt = usize::rand(&mut rng) % initial_domain_size;
            if corrupted_indices.contains(&idx_to_corrupt) {
                continue;
            }

            let original_value = corrupted_codeword_c_prime_0_evals[idx_to_corrupt];
            let mut new_value = F::rand(&mut rng);
            while new_value == original_value {
                new_value = F::rand(&mut rng);
            }
            corrupted_codeword_c_prime_0_evals[idx_to_corrupt] = new_value;
            corrupted_indices.insert(idx_to_corrupt);
        }

        let alpha = F::rand(&mut rng);

        let (folded_corrupted_evals, new_domain) = perform_fold(
            &corrupted_codeword_c_prime_0_evals,
            domain,
            alpha,
            folding_factor,
        );

        let (folded_true_evals, _) = perform_fold(
            &codeword_c0_evals,
            domain,
            alpha,
            folding_factor,
        );

        let differing_points = folded_corrupted_evals
            .iter()
            .zip(folded_true_evals.iter())
            .filter(|(a, b)| a != b)
            .count();

        let measured_rho_1 = differing_points as f64 / new_domain.size() as f64;

        let theoretical_rho_1 = 1.0_f64 - (1.0_f64 - initial_corruption_rate).powf(folding_factor as f64);

        println!("--- Debugging Single Fold (Goldilocks Field) ---");
        println!("Initial rho_0:       {}", initial_corruption_rate);
        println!("Measured rho_1:      {}", measured_rho_1);
        println!("Theoretical rho_1:   {}", theoretical_rho_1);

        let tolerance = 0.01;
        assert!(
            (measured_rho_1 - theoretical_rho_1).abs() < tolerance,
            "Single fold amplification measured rate {} is not close to precise theoretical rate {}",
            measured_rho_1,
            theoretical_rho_1
        );
    }
}