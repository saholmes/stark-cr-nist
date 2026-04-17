use ark_ff::{Field, Zero};
use ark_goldilocks::Goldilocks as F;
use blake3::Hasher;

#[cfg(feature = "parallel")]
use once_cell::sync::OnceCell;
#[cfg(feature = "parallel")]
use rayon::ThreadPoolBuilder;
//use rayon::{prelude::*, ThreadPoolBuilder};

/// Poseidon permutation parameters for benchmarking and M1 scaffolding.
/// Default static width t=17 matches Merkle arity m=16 with capacity c=1.
pub const T: usize = 17;
pub const RATE: usize = 16;
pub const CAPACITY: usize = 1;
pub const RF: usize = 8;
pub const RP: usize = 64;
pub const RP_9: usize = 60;
pub const ALPHA: u64 = 5;

/// =======================
/// SAFE Goldilocks hashing
/// =======================
///
/// IMPORTANT:
/// - Does NOT use arkworks deserialization
/// - Panic-free in arkworks 0.4.x
/// - Deterministic
#[inline]
fn poseidon_fr_from_hash(tag: &str, data: &[u8]) -> F {
    let mut h = Hasher::new();
    h.update(tag.as_bytes());
    h.update(data);
    let out = h.finalize();

    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&out.as_bytes()[..8]);
    F::from(u64::from_le_bytes(bytes))
}

#[derive(Clone)]
pub struct PoseidonParams {
    pub mds: [[F; T]; T],
    pub rc_full: [[F; T]; RF],
    pub rc_partial: [F; RP],
}

#[cfg(feature = "parallel")]
pub fn init_poseidon_parallelism(num_threads: usize) {
    static POSEIDON_POOL_INIT: OnceCell<()> = OnceCell::new();
    POSEIDON_POOL_INIT.get_or_init(|| {
        let _ = ThreadPoolBuilder::new()
            .thread_name(|idx| format!("poseidon-worker-{idx}"))
            .num_threads(num_threads)
            .build_global();
    });
}

#[cfg(not(feature = "parallel"))]
pub fn init_poseidon_parallelism(_: usize) {}

#[inline]
pub fn sbox5(x: F) -> F {
    let x2 = x.square();
    let x4 = x2.square();
    x * x4
}

pub fn permute(state: &mut [F; T], params: &PoseidonParams) {
    let rf_half = RF / 2;

    for r in 0..rf_half {
        for i in 0..T {
            state[i] += params.rc_full[r][i];
            state[i] = sbox5(state[i]);
        }
        *state = mds_mul_fixed(&params.mds, state);
    }

    for r in 0..RP {
        state[0] += params.rc_partial[r];
        state[0] = sbox5(state[0]);
        *state = mds_mul_fixed(&params.mds, state);
    }

    for r in rf_half..RF {
        for i in 0..T {
            state[i] += params.rc_full[r][i];
            state[i] = sbox5(state[i]);
        }
        *state = mds_mul_fixed(&params.mds, state);
    }
}

fn mds_mul_fixed(mds: &[[F; T]; T], state: &[F; T]) -> [F; T] {
    let mut out = [F::zero(); T];
    for i in 0..T {
        for j in 0..T {
            out[i] += mds[i][j] * state[j];
        }
    }
    out
}

/// =======================
/// Parameter derivation
/// =======================

#[allow(dead_code)]
fn seed_for_t(t: usize) -> Vec<u8> {
    let mut s = Vec::new();
    s.extend_from_slice(b"POSEIDON-GOLDILOCKS-T");
    s.extend_from_slice(&(t as u64).to_le_bytes());
    s
}

fn derive_mds(seed: &[u8], t: usize) -> Vec<Vec<F>> {
    let mut m = vec![vec![F::zero(); t]; t];
    for i in 0..t {
        for j in 0..t {
            let mut data = Vec::with_capacity(seed.len() + 16);
            data.extend_from_slice(&(i as u64).to_le_bytes());
            data.extend_from_slice(&(j as u64).to_le_bytes());
            data.extend_from_slice(seed);
            m[i][j] = poseidon_fr_from_hash("POSEIDON-MDS", &data);
        }
    }
    m
}

fn derive_rc_full(seed: &[u8], rf: usize, t: usize) -> Vec<Vec<F>> {
    let mut rc = vec![vec![F::zero(); t]; rf];
    for r in 0..rf {
        for i in 0..t {
            let mut data = Vec::with_capacity(seed.len() + 16);
            data.extend_from_slice(&(r as u64).to_le_bytes());
            data.extend_from_slice(&(i as u64).to_le_bytes());
            data.extend_from_slice(seed);
            rc[r][i] = poseidon_fr_from_hash("POSEIDON-RC-FULL", &data);
        }
    }
    rc
}

fn derive_rc_partial(seed: &[u8], rp: usize) -> Vec<F> {
    let mut rc = vec![F::zero(); rp];
    for r in 0..rp {
        let mut data = Vec::with_capacity(seed.len() + 8);
        data.extend_from_slice(&(r as u64).to_le_bytes());
        data.extend_from_slice(seed);
        rc[r] = poseidon_fr_from_hash("POSEIDON-RC-PART", &data);
    }
    rc
}

/// =======================
/// Public constructor
/// =======================

pub mod params {
    use super::*;

    pub fn generate_params_t17_x5(seed: &[u8]) -> PoseidonParams {
        let mds_v = derive_mds(seed, T);
        let rc_full_v = derive_rc_full(seed, RF, T);
        let rc_partial_v = derive_rc_partial(seed, RP);

        let mut mds = [[F::zero(); T]; T];
        let mut rc_full = [[F::zero(); T]; RF];
        let mut rc_partial = [F::zero(); RP];

        for i in 0..T {
            for j in 0..T {
                mds[i][j] = mds_v[i][j];
            }
        }

        for r in 0..RF {
            for i in 0..T {
                rc_full[r][i] = rc_full_v[r][i];
            }
        }

        for r in 0..RP {
            rc_partial[r] = rc_partial_v[r];
        }

        PoseidonParams {
            mds,
            rc_full,
            rc_partial,
        }
    }
}