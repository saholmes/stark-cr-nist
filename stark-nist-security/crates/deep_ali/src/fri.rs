
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused_macros)]

use ark_goldilocks::Goldilocks as F;
use ark_serialize::CanonicalSerialize;
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
    tr.challenge(b"out")
}

// ────────────────────────────────────────────────────────────────────────
//  Extension-field division helper
// ────────────────────────────────────────────────────────────────────────

#[inline]
fn ext_div<E: TowerField>(a: E, b: E) -> E {
    a * b.invert().expect("ext_div: division by zero in extension field")
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

/// Extract per-coset interpolation coefficients for an arity-m fold.
///
/// For each coset b = 0..n_next-1, the fibre is
///   { evals[b + j * n_next] : j = 0..m-1 }
/// evaluated at domain points
///   x_j = ω^{b + j * n_next}.
///
/// Returns `n_next` tuples of `m` coefficients `[C_0(b), …, C_{m-1}(b)]`
/// such that `Σ_i C_i(b) · x_j^i = evals[b + j * n_next]`.
///
/// By Lemma 1 of the paper, when `evals` is the evaluation of a
/// polynomial of degree < d on D, each `C_i` lies in `RS[D', d/m]`.
fn extract_all_coset_coefficients<E: TowerField>(
    evals: &[E],
    omega: F,
    m: usize,
) -> Vec<Vec<E>> {
    let n = evals.len();
    assert!(n % m == 0);
    let n_next = n / m;
    let zeta = omega.pow([n_next as u64]); // primitive m-th root of unity

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

/// Solve the Vandermonde system for a single coset.
///
/// Given fibre values `v_j = f(ω^b · ζ^j)` for j=0..m-1
/// (where ζ = ω^{n_next} is a primitive m-th root of unity),
/// returns coefficients `c_i` such that `Σ_i c_i · (ω^b ζ^j)^i = v_j`.
///
/// Method: `d = IDFT_m(v)` using ζ, then `c_i = d_i · (ω^b)^{-i}`.
fn interpolate_coset_ext<E: TowerField>(
    fibre_values: &[E],
    omega_b: F,
    zeta: F,
    m: usize,
) -> Vec<E> {
    let d = E::DEGREE;
    let m_inv = F::from(m as u64).inverse().unwrap();
    let zeta_inv = zeta.inverse().unwrap();

    // Precompute ζ^{-k} for k = 0..m-1
    let mut zi_pows = vec![F::one(); m];
    for k in 1..m {
        zi_pows[k] = zi_pows[k - 1] * zeta_inv;
    }

    // Component-wise IDFT: d_i = (1/m) Σ_j v_j · ζ^{-ij}
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

    // c_i = d_i · (ω^b)^{-i}
    let omega_b_inv = if omega_b == F::zero() {
        F::one()
    } else {
        omega_b.inverse().unwrap()
    };

    let mut result = Vec::with_capacity(m);
    let mut ob_inv_pow = F::one(); // (ω^b)^{-0} = 1
    for i in 0..m {
        let comps: Vec<F> = (0..d).map(|c| comp_d[c][i]).collect();
        let di = E::from_fp_components(&comps).unwrap();
        result.push(di * E::from_fp(ob_inv_pow));
        ob_inv_pow *= omega_b_inv;
    }

    result
}

/// Fold using per-coset interpolation coefficients.
///
/// Computes `P_z(α) = Σ_i C_i(z) · α^i` for each coset z.
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

/// Compute the s-layer using interpolation coefficients.
///
/// For each position in the original domain, stores the fold value
/// of the coset it belongs to (matching `compute_s_layer_ext` semantics
/// but using the interpolation-based fold formula).
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

/// Verify interpolation consistency at a single coset.
///
/// Checks that `Σ_i C_i · x_j^i == f(x_j)` for every fibre point x_j.
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

/// Batched degree check for coefficient functions.
///
/// Computes `Γ(z) = Σ_i β^i C_i(z)` for all z ∈ D_L and verifies
/// `deg(Γ) < d_final` by IFFT + high-coefficient zero-check.
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

    // Compute Γ(z_b) for each z_b
    let gamma_evals: Vec<E> = (0..n_final)
        .map(|b| {
            let mut sum = E::zero();
            for i in 0..m {
                sum = sum + coeff_tuples[b][i] * beta_pows[i];
            }
            sum
        })
        .collect();

    // Component-wise IFFT to get polynomial coefficients
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

    // Check all coefficients of degree ≥ d_final are zero
    for k in d_final..n_final {
        for j in 0..deg {
            if !comp_coeffs[j][k].is_zero() {
                return false;
            }
        }
    }

    true
}

/// Merkle tree configuration for the coefficient commitment.
fn coeff_tree_config(n_final: usize) -> MerkleChannelCfg {
    let arity = pick_arity_for_layer(n_final, 16).max(2);
    let depth = merkle_depth(n_final, arity);
    MerkleChannelCfg::new(vec![arity; depth], 0xFE)
}

/// Serialize a coefficient tuple as base-field elements for Merkle hashing.
fn coeff_leaf_fields<E: TowerField>(tuple: &[E]) -> Vec<F> {
    tuple.iter().flat_map(|e| e.to_fp_components()).collect()
}

// ────────────────────────────────────────────────────────────────────────
//  Extension-field evaluation/coefficient helpers
// ────────────────────────────────────────────────────────────────────────

/// Component-wise IFFT: convert extension-field evaluations over a
/// radix-2 domain into polynomial coefficients.
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

/// Evaluate a polynomial (given by extension-field coefficients) at an
/// extension-field point using Horner's method.  O(coeffs.len()).
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
    fused.serialize_uncompressed(&mut seed_bytes[..]).expect("serialize");
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
    seed_f.serialize_uncompressed(&mut seed_bytes[..]).unwrap();
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
) {
    tr.absorb_bytes(b"DEEP-FRI-STATEMENT");
    tr.absorb_field(F::from(n0 as u64));
    tr.absorb_field(F::from(schedule.len() as u64));
    for &m in schedule {
        tr.absorb_field(F::from(m as u64));
    }
    tr.absorb_field(F::from(seed_z));
    tr.absorb_field(F::from(E::DEGREE as u64));
    // Bind the protocol variant so transcripts are domain-separated
    tr.absorb_field(F::from(coeff_commit_final as u64));
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
}

/// Prover state after transcript construction.
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
    /// Construction 5.1: per-coset interpolation coefficient tuples.
    pub coeff_tuples: Option<Vec<Vec<E>>>,
    /// Construction 5.1: Merkle root of the coefficient tree.
    pub coeff_root: Option<[u8; HASH_BYTES]>,
    /// Construction 5.1: batched degree-check challenge.
    pub beta_deg: Option<E>,
    /// Whether Construction 5.1 is active.
    pub coeff_commit_final: bool,
    /// Degree bound for the final layer (each C_i should have degree < d_final).
    pub d_final: usize,
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
    // NOTE: No final_fibre field.  Under Construction 5.1 the
    // coefficient tuples are sent in the clear and the verifier
    // checks single-point consistency using the already-opened
    // f^{(L-1)} value (Theorem 4, revised).
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
    /// Coefficients of the final polynomial (degree < d_final).
    /// Replaces the old full `final_codeword` to keep the verifier O(r).
    pub final_poly_coeffs: Vec<E>,
    pub n0: usize,
    pub omega0: F,
    /// Construction 5.1: all per-coset coefficient tuples (sent in the clear).
    pub coeff_tuples: Option<Vec<Vec<E>>>,
    /// Construction 5.1: Merkle root of the coefficient tree.
    pub coeff_root: Option<[u8; HASH_BYTES]>,
}

#[derive(Clone, Debug)]
pub struct DeepFriParams {
    pub schedule: Vec<usize>,
    pub r: usize,
    pub seed_z: u64,
    /// Enable Construction 5.1 (coefficient-committed final round).
    pub coeff_commit_final: bool,
    /// Degree bound for the final polynomial. Default 1 (constant).
    pub d_final: usize,
}

impl DeepFriParams {
    /// Create params with the original protocol (no coefficient commitment).
    pub fn new(schedule: Vec<usize>, r: usize, seed_z: u64) -> Self {
        Self {
            schedule,
            r,
            seed_z,
            coeff_commit_final: false,
            d_final: 1,
        }
    }

    /// Enable Construction 5.1.
    pub fn with_coeff_commit(mut self) -> Self {
        self.coeff_commit_final = true;
        self
    }

    /// Set the final degree bound.
    pub fn with_d_final(mut self, d: usize) -> Self {
        self.d_final = d;
        self
    }
}

// ────────────────────────────────────────────────────────────────────────
//  Leaf serialization helpers
// ────────────────────────────────────────────────────────────────────────

#[inline]
fn ext_leaf_fields<E: TowerField>(f: E, s: E, q: E) -> Vec<F> {
    let mut fields = f.to_fp_components();
    fields.extend(s.to_fp_components());
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
        components.push(tr.challenge(&sub_tag));
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
    let normal_layers = if use_coeff_commit { l - 1 } else { l };

    // ── Phase 0: Statement binding ──
    let mut tr = Transcript::new_matching_hash(b"FRI/FS");
    bind_statement_to_transcript::<E>(
        &mut tr,
        &schedule,
        domain0.size,
        params.seed_z,
        params.coeff_commit_final,
    );

    // ── Phase 1: Commit f₀ (base-field trace) ──
    let f0_th = f0_trace_hash(domain0.size, params.seed_z);
    let f0_cfg = f0_tree_config(domain0.size);
    let mut f0_tree = MerkleTreeChannel::new(f0_cfg.clone(), f0_th);
    for &val in &f0 {
        f0_tree.push_leaf(&[val]);
    }
    let root_f0 = f0_tree.finalize();

    tr.absorb_bytes(&root_f0);

    // ── Phase 2: DEEP challenge z ∈ E ──
    let z_ext = challenge_ext::<E>(&mut tr, b"z_fp3");

    let trace_hash: [u8; HASH_BYTES] = transcript_challenge_hash(&mut tr, ds::FRI_SEED);

    // ── Phase 3: Layer-by-layer ──

    let f0_ext: Vec<E> = f0.iter().map(|&x| E::from_fp(x)).collect();

    let mut f_layers_ext: Vec<Vec<E>> = Vec::with_capacity(l + 1);
    let mut s_layers: Vec<Vec<E>> = Vec::with_capacity(l + 1);
    let mut q_layers: Vec<Vec<E>> = Vec::with_capacity(l);
    let mut fz_layers: Vec<E> = Vec::with_capacity(l);
    let mut omega_layers: Vec<F> = Vec::with_capacity(l);
    let mut alpha_layers: Vec<E> = Vec::with_capacity(l);
    let mut layer_commitments: Vec<FriLayerCommitment> = Vec::with_capacity(l);

    f_layers_ext.push(f0_ext);
    let mut cur_size = domain0.size;

    // ── Normal (intermediate) layers: 0 .. normal_layers-1 ──
    for ell in 0..normal_layers {
        let m = schedule[ell];

        let alpha_ell = challenge_ext::<E>(&mut tr, b"alpha");
        alpha_layers.push(alpha_ell);

        let dom = Domain::<F>::new(cur_size).unwrap();
        let omega = dom.group_gen;
        omega_layers.push(omega);

        let cur_f = &f_layers_ext[ell];
        let (q, fz) = compute_q_layer_ext_on_ext(cur_f, z_ext, omega);
        q_layers.push(q.clone());
        fz_layers.push(fz);

        let s = compute_s_layer_ext(cur_f, alpha_ell, m);
        s_layers.push(s.clone());

        // Merkle commitment
        let arity = pick_arity_for_layer(cur_size, m).max(2);
        let depth = merkle_depth(cur_size, arity);
        let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);
        let mut tree = MerkleTreeChannel::new(cfg, trace_hash);

        let all_fields: Vec<Vec<F>> = (0..cur_size)
            .map(|i| ext_leaf_fields(cur_f[i], s[i], q[i]))
            .collect();
        tree.push_leaves_parallel(&all_fields);

        let layer_root = tree.finalize();

        layer_commitments.push(FriLayerCommitment {
            n: cur_size,
            m,
            root: layer_root,
        });

        absorb_ext(&mut tr, fz);
        tr.absorb_bytes(&layer_root);

        let next_f = fri_fold_layer_ext_impl(cur_f, alpha_ell, m);
        cur_size /= m;
        f_layers_ext.push(next_f);

        logln!(
            "[PROVER] ell={} z_ext={:?} alpha={:?}",
            ell, z_ext, alpha_ell
        );
    }

    // ── Construction 5.1: coefficient-committed final layer ──
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

        // DEEP quotient (does not need α)
        let (q, fz) = compute_q_layer_ext_on_ext(cur_f, z_ext, omega);
        q_layers.push(q.clone());
        fz_layers.push(fz);

        // s-layer: zeros (unused under Construction 5.1, but we still
        // need to store *something* for the Merkle leaf format)
        let s = vec![E::zero(); cur_size];
        s_layers.push(s.clone());

        // Commit (f, 0, q) in the layer tree
        let arity = pick_arity_for_layer(cur_size, m).max(2);
        let depth = merkle_depth(cur_size, arity);
        let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);
        let mut tree = MerkleTreeChannel::new(cfg, trace_hash);

        let all_fields: Vec<Vec<F>> = (0..cur_size)
            .map(|i| ext_leaf_fields(cur_f[i], s[i], q[i]))
            .collect();
        tree.push_leaves_parallel(&all_fields);

        let layer_root = tree.finalize();

        layer_commitments.push(FriLayerCommitment {
            n: cur_size,
            m,
            root: layer_root,
        });

        // Absorb fz and layer root BEFORE the coefficient commitment
        absorb_ext(&mut tr, fz);
        tr.absorb_bytes(&layer_root);

        // ── Extract per-coset interpolation coefficients ──
        let coeff_tuples = extract_all_coset_coefficients(cur_f, omega, m);

        // ── Commit coefficient tree ──
        let n_final = cur_size / m;
        let coeff_cfg = coeff_tree_config(n_final);
        let mut coeff_tree = MerkleTreeChannel::new(coeff_cfg, trace_hash);

        let coeff_fields: Vec<Vec<F>> = coeff_tuples
            .iter()
            .map(|t| coeff_leaf_fields(t))
            .collect();
        coeff_tree.push_leaves_parallel(&coeff_fields);

        let coeff_root = coeff_tree.finalize();

        // Absorb coefficient root (before deriving α)
        tr.absorb_bytes(&coeff_root);

        // NOW derive the final folding challenge α_{L-1}
        let alpha_ell = challenge_ext::<E>(&mut tr, b"alpha");
        alpha_layers.push(alpha_ell);

        // Derive β for the batched degree check
        let beta_deg = challenge_ext::<E>(&mut tr, b"beta_deg");

        // Fold using interpolation coefficients
        let next_f = interpolation_fold_ext(&coeff_tuples, alpha_ell);
        cur_size = n_final;
        f_layers_ext.push(next_f);

        stored_coeff_tuples = Some(coeff_tuples);
        stored_coeff_root = Some(coeff_root);
        stored_beta = Some(beta_deg);

        logln!(
            "[PROVER][COEFF-COMMIT] ell={} coeff_root={:?}",
            ell, coeff_root
        );
    }

    // Dummy s-layer for the final codeword layer
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

    eprintln!("[DIAG] HASH_BYTES = {}", HASH_BYTES);

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

    // ── Rebuild f₀ Merkle tree ──
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

    // ── Rebuild layer Merkle trees ──
    let mut layer_proofs = Vec::with_capacity(L);

    for ell in 0..L {
        let layer = &st.transcript.layers[ell];
        let arity = pick_arity_for_layer(layer.n, layer.m).max(2);
        let depth = merkle_depth(layer.n, arity);
        let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);
        let mut tree = MerkleTreeChannel::new(cfg, st.trace_hash);

        for i in 0..layer.n {
            let fields = ext_leaf_fields(
                st.f_layers_ext[ell][i],
                st.s_layers[ell][i],
                st.q_layers[ell][i],
            );
            tree.push_leaf(&fields);
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
    };

    let st: FriProverState<E> = fri_build_transcript(f0, domain0, &prover_params);

    let L = params.schedule.len();
    let final_evals = st.f_layers_ext[L].clone();

    // ── Extract final polynomial coefficients via component-wise IFFT ──
    let final_poly_coeffs: Vec<E> = {
        let all_coeffs = ext_evals_to_coeffs::<E>(&final_evals);
        let d_final = params.d_final.min(all_coeffs.len());

        // Warn (debug only) if high coefficients are non-zero
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

    // ── Replay the transcript to derive query_seed ──
    let query_seed = {
        let mut tr = Transcript::new_matching_hash(b"FRI/FS");
        bind_statement_to_transcript::<E>(
            &mut tr,
            &params.schedule,
            domain0.size,
            params.seed_z,
            params.coeff_commit_final,
        );
        tr.absorb_bytes(&st.root_f0);

        let _ = challenge_ext::<E>(&mut tr, b"z_fp3");
        let _: [u8; HASH_BYTES] = transcript_challenge_hash(&mut tr, ds::FRI_SEED);

        let use_coeff_commit = params.coeff_commit_final && L > 0;
        let normal_layers = if use_coeff_commit { L - 1 } else { L };

        // Replay intermediate layers
        for ell in 0..normal_layers {
            let _ = challenge_ext::<E>(&mut tr, b"alpha");
            absorb_ext(&mut tr, st.fz_layers[ell]);
            tr.absorb_bytes(&st.transcript.layers[ell].root);
        }

        // Replay Construction 5.1 final layer
        if use_coeff_commit {
            let ell = L - 1;
            // fz and layer root were absorbed before coeff root
            absorb_ext(&mut tr, st.fz_layers[ell]);
            tr.absorb_bytes(&st.transcript.layers[ell].root);

            // coefficient root
            tr.absorb_bytes(&st.coeff_root.unwrap());

            // alpha (derived after coeff root)
            let _ = challenge_ext::<E>(&mut tr, b"alpha");
            // beta
            let _ = challenge_ext::<E>(&mut tr, b"beta_deg");
        }

        // Absorb only the d_final polynomial coefficients (NOT the full codeword)
        for &c in &final_poly_coeffs {
            absorb_ext::<E>(&mut tr, c);
        }

        tr.challenge(b"query_seed")
    };

    let (query_refs, roots, layer_proofs, f0_openings) =
        fri_prove_queries(&st, params.r, query_seed);

    // ── Build per-query payloads ──
    // Under Construction 5.1 the coefficient tuples are sent in the clear;
    // the verifier checks single-point consistency using the already-opened
    // f^{(L-1)} value at each query position (Theorem 4, revised).
    // No additional fibre openings are required.

    let mut queries = Vec::with_capacity(params.r);

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
    }
}

pub fn deep_fri_proof_size_bytes<E: TowerField>(proof: &DeepFriProof<E>) -> usize {
    const FIELD_BYTES: usize = 8;
    let ext_bytes: usize = E::DEGREE * FIELD_BYTES;
    const INDEX_BYTES: usize = 8;

    let mut bytes = 0usize;

    bytes += HASH_BYTES;
    bytes += proof.roots.len() * HASH_BYTES;

    bytes += proof.fz_per_layer.len() * ext_bytes;

    bytes += proof.final_poly_coeffs.len() * ext_bytes;

    for q in &proof.queries {
        bytes += q.per_layer_payloads.len() * 3 * ext_bytes;
    }

    for opening in &proof.f0_openings {
        bytes += HASH_BYTES;
        bytes += INDEX_BYTES;
        for level in &opening.path {
            bytes += level.len() * HASH_BYTES;
        }
    }

    for layer in &proof.layer_proofs.layers {
        for opening in &layer.openings {
            bytes += HASH_BYTES;
            bytes += INDEX_BYTES;
            for level in &opening.path {
                bytes += level.len() * HASH_BYTES;
            }
        }
    }

    // Coefficient tuples (Construction 5.1) — sent in the clear once
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
    let normal_layers = if use_coeff_commit { L - 1 } else { L };

    // ── Reconstruct the Fiat–Shamir transcript ──
    let mut tr = Transcript::new_matching_hash(b"FRI/FS");
    bind_statement_to_transcript::<E>(
        &mut tr,
        &params.schedule,
        proof.n0,
        params.seed_z,
        params.coeff_commit_final,
    );
    tr.absorb_bytes(&proof.root_f0);

    let z_ext = challenge_ext::<E>(&mut tr, b"z_fp3");
    let trace_hash: [u8; HASH_BYTES] = transcript_challenge_hash(&mut tr, ds::FRI_SEED);

    logln!("[VERIFY] z_ext = {:?}", z_ext);

    // ── Derive per-layer fold challenges ──
    let mut alpha_layers: Vec<E> = Vec::with_capacity(L);

    // Intermediate layers
    for ell in 0..normal_layers {
        let alpha_ell = challenge_ext::<E>(&mut tr, b"alpha");
        alpha_layers.push(alpha_ell);

        if ell < proof.fz_per_layer.len() {
            absorb_ext(&mut tr, proof.fz_per_layer[ell]);
        }
        if ell < proof.roots.len() {
            tr.absorb_bytes(&proof.roots[ell]);
        }
    }

    // Construction 5.1: final layer has different transcript ordering
    let mut beta_deg: Option<E> = None;

    if use_coeff_commit {
        let ell = L - 1;

        // Absorb fz and layer root BEFORE coeff root
        if ell < proof.fz_per_layer.len() {
            absorb_ext(&mut tr, proof.fz_per_layer[ell]);
        }
        if ell < proof.roots.len() {
            tr.absorb_bytes(&proof.roots[ell]);
        }

        // Absorb coefficient root
        match proof.coeff_root {
            Some(ref cr) => tr.absorb_bytes(cr),
            None => {
                eprintln!("[FAIL][COEFF ROOT MISSING]");
                return false;
            }
        }

        // Derive alpha AFTER coeff root
        let alpha_ell = challenge_ext::<E>(&mut tr, b"alpha");
        alpha_layers.push(alpha_ell);

        // Derive beta for degree check
        let beta = challenge_ext::<E>(&mut tr, b"beta_deg");
        beta_deg = Some(beta);
    }

    // Absorb only the d_final polynomial coefficients (NOT the full codeword)
    for &c in &proof.final_poly_coeffs {
        absorb_ext::<E>(&mut tr, c);
    }

    let query_seed: F = tr.challenge(b"query_seed");

    // ── Final polynomial coefficient count check ──
    if proof.final_poly_coeffs.len() != params.d_final {
        eprintln!(
            "[FAIL][FINAL POLY COEFFS SIZE] expected={} got={}",
            params.d_final,
            proof.final_poly_coeffs.len()
        );
        return false;
    }

    // ── Construction 5.1: coefficient commitment verification ──
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

        // Size check
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

        // Recompute Merkle root from coefficient tuples
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

        // Batched degree check
        let beta = beta_deg.unwrap();
        if !batched_degree_check_ext(coeff_tuples, beta, params.d_final) {
            eprintln!("[FAIL][BATCHED DEGREE CHECK]");
            return false;
        }
    }

    // ── Precompute per-layer domain generators (avoids repeated Domain::new) ──
    let omega_per_layer: Vec<F> = (0..L)
        .map(|ell| Domain::<F>::new(sizes[ell]).unwrap().group_gen)
        .collect();
    let omega_final: F = if L > 0 {
        Domain::<F>::new(sizes[L]).unwrap().group_gen
    } else {
        F::one()
    };

    // f₀ tree config
    let f0_th = f0_trace_hash(proof.n0, params.seed_z);
    let f0_cfg = f0_tree_config(proof.n0);

    // ── Per-query verification ──
    for q in 0..params.r {
        let qp = &proof.queries[q];

        // Re-derive query position
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

        // ── f₀ Merkle opening ──
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

        // ── Per-layer checks ──
        for ell in 0..L {
            let opening = &proof.layer_proofs.layers[ell].openings[q];
            let rref = &qp.per_layer_refs[ell];
            let pay = &qp.per_layer_payloads[ell];

            let arity = pick_arity_for_layer(sizes[ell], params.schedule[ell]).max(2);
            let depth = merkle_depth(sizes[ell], arity);
            let cfg = MerkleChannelCfg::new(vec![arity; depth], ell as u64);

            // 1. Merkle verification
            if !MerkleTreeChannel::verify_opening(
                &cfg,
                proof.roots[ell],
                opening,
                &trace_hash,
            ) {
                eprintln!("[FAIL][MERKLE] q={} ell={}", q, ell);
                return false;
            }

            // 2. Index binding
            if opening.index != rref.i {
                eprintln!("[FAIL][INDEX BINDING] q={} ell={}", q, ell);
                return false;
            }

            // 3. Leaf binding
            let leaf_fields = ext_leaf_fields(pay.f_val, pay.s_val, pay.q_val);
            let expected_leaf = compute_leaf_hash(&cfg, opening.index, &leaf_fields);
            if expected_leaf != opening.leaf {
                eprintln!("[FAIL][LEAF BINDING] q={} ell={}", q, ell);
                return false;
            }

            // 4. DEEP quotient check
            let omega_ell = omega_per_layer[ell];
            let x_i = omega_ell.pow([rref.i as u64]);

            let fz = proof.fz_per_layer[ell];
            let num   = pay.f_val - fz;
            let denom = E::from_fp(x_i) - z_ext;

            if pay.q_val * denom != num {
                eprintln!(
                    "[FAIL][DEEP-EXT] q={} ell={}\n  f_val={:?}\n  fz={:?}\n  q_val={:?}\n  x_i={:?}",
                    q, ell, pay.f_val, fz, pay.q_val, x_i,
                );
                return false;
            }

            // 5. Fold consistency
            let is_final_layer = ell == L - 1;

            if is_final_layer && use_coeff_commit {
                // ── Construction 5.1 (revised): single-point consistency ──
                //
                // The coefficient functions C_0*,...,C_{m-1}* are sent in
                // the clear and have passed the batched degree check, so
                //   h*(x) = Σ_k C_k*(x^{a_L}) · x^k
                // is a codeword of RS[D_{L-1}, d_{L-1}].
                //
                // We check f^{(L-1)}(x_i) == h*(x_i) at the single
                // already-opened query point.  Detection probability
                // is δ_{L-1} by Theorem 4 (revised).  No fibre openings
                // are required.

                let m = params.schedule[ell];
                let n_next = sizes[ell] / m;
                let coset_b = qp.final_index; // = rref.i % n_next

                let coeff_tuples = proof.coeff_tuples.as_ref().unwrap();
                let coeff_tuple = &coeff_tuples[coset_b];

                // (a) Single-point consistency: f^{(L-1)}(x_i) == Σ_k C_k(z_b) · x_i^k
                let mut h_star = E::zero();
                let mut x_pow = F::one();
                for k in 0..m {
                    h_star = h_star + coeff_tuple[k] * E::from_fp(x_pow);
                    x_pow *= x_i;
                }

                if pay.f_val != h_star {
                    eprintln!(
                        "[FAIL][SINGLE-POINT CONSISTENCY] q={} ell={}\n  f_val={:?}\n  h_star={:?}",
                        q, ell, pay.f_val, h_star,
                    );
                    return false;
                }

                // (b) Fold-value binding: Σ_k C_k(z_b) · α^k == eval(final_poly, z_b)
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
                // ── Original protocol: s-value fold consistency ──
                let verified_f_next = if ell + 1 < L {
                    qp.per_layer_payloads[ell + 1].f_val
                } else {
                    // Evaluate final polynomial at the query's domain point
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

    logln!("[VERIFY] SUCCESS");
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
    //  Construction 5.1 specific tests
    // ────────────────────────────────────────────────────────────────

    /// Verify that interpolation coefficients, when evaluated at the fibre
    /// points, reproduce the original fibre values.
    fn test_coset_interpolation_roundtrip_for<E: TowerField>() {
        let mut rng = StdRng::seed_from_u64(123);
        let n = 64usize;
        let m = 4usize;
        let n_next = n / m;

        let dom = Domain::<TestField>::new(n).unwrap();
        let omega = dom.group_gen;
        let zeta = omega.pow([n_next as u64]); // primitive m-th root

        // Random extension-field evaluations
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

        // For each coset, verify P(x_j) == evals[b + j * n_next]
        for b in 0..n_next {
            let omega_b = omega.pow([b as u64]);
            for j in 0..m {
                let x_j = omega.pow([(b + j * n_next) as u64]);
                // Evaluate P(x_j) = Σ C_i * x_j^i
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

    /// Verify that for a low-degree polynomial, the coefficient functions
    /// are themselves low-degree (Lemma 1).
    fn test_coeff_functions_low_degree_for<E: TowerField>() {
        let mut rng = StdRng::seed_from_u64(456);
        let n = 64usize;
        let m = 4usize;
        let n_next = n / m;
        let degree = m - 1; // degree < m so d_final = 1

        let dom = GeneralEvaluationDomain::<TestField>::new(n).unwrap();
        let poly = DensePolynomial::<TestField>::rand(degree, &mut rng);
        let evals = dom.fft(poly.coeffs());

        let evals_ext: Vec<E> = evals.iter().map(|&x| E::from_fp(x)).collect();

        let omega = dom.group_gen();
        let coeff_tuples = extract_all_coset_coefficients(&evals_ext, omega, m);

        // Each C_i should be constant (degree < d_final = 1)
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

    /// Verify the interpolation-based fold matches the polynomial evaluation.
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

        // The fold at each coset should equal P_z(α)
        // For a globally low-degree polynomial, all cosets yield the same P(α)
        assert_eq!(folded.len(), n_next);

        // Verify the fold is constant (since degree < m and d_final = 1)
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

    /// Verify the batched degree check passes for honest coefficients
    /// and fails for corrupted ones.
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

        // Honest: should pass with d_final = 1
        assert!(
            batched_degree_check_ext(&coeff_tuples, beta, 1),
            "Batched degree check should pass for honest coefficients"
        );

        // Corrupt one coefficient
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