use ark_ff::{PrimeField, Zero};
use ark_goldilocks::Goldilocks as F;

use poseidon::{
    permute,
    params::generate_params_t17_x5,
    PoseidonParams,
    T,
};

use sha3::{Digest, Sha3_256};

/// =======================
/// Dual commitment object
/// =======================
///
/// Corresponds to the ProVerif tuple:
///   (sha3_commit, poseidon_commit, trace_hash)
///
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DualCommitment {
    pub sha_commit: [u8; 32],   // sha3_commit(encode(trace), trace_hash)
    pub poseidon_root: F,       // poseidon_commit(trace, trace_hash)
    pub trace_hash: [u8; 32],   // sha3_trace(trace)
}

/// Merkle commitment using Poseidon (t = 17, arity = 16)
pub struct MerkleCommitment {
    pub arity: usize,
    pub params: PoseidonParams,
}

impl MerkleCommitment {
    pub fn with_default_params() -> Self {
        let seed = b"POSEIDON-T17-X5-SEED";
        let params = generate_params_t17_x5(seed);
        Self {
            arity: 16,
            params,
        }
    }

    // ============================================================
    // Goldilocks field <-> bytes (canonical, injective)
    // ============================================================

    #[inline]
    fn field_to_bytes(x: &F) -> [u8; 8] {
        let limb0 = x.into_bigint().0[0];
        limb0.to_le_bytes()
    }

    // ------------------------------------------------------------
    // Row-wise encoding (Merkle leaves)
    // ------------------------------------------------------------

    fn encode_trace_rows(trace: &[Vec<F>]) -> Vec<Vec<u8>> {
        trace
            .iter()
            .map(|row| {
                let mut out = Vec::with_capacity(row.len() * 8);
                for x in row {
                    out.extend_from_slice(&Self::field_to_bytes(x));
                }
                out
            })
            .collect()
    }

    // ------------------------------------------------------------
    // Flat encoding (SHA3 binding)
    // ------------------------------------------------------------

    fn encode_trace_flat(trace: &[Vec<F>]) -> Vec<u8> {
        let mut out = Vec::new();
        for row in trace {
            for x in row {
                out.extend_from_slice(&Self::field_to_bytes(x));
            }
        }
        out
    }

    // ============================================================
    // sha3_trace : field_trace -> trace_hash
    // ============================================================

    fn sha3_trace(trace: &[Vec<F>]) -> [u8; 32] {
        let mut h = Sha3_256::new();
        h.update(b"TRACE_HASH_V1");
        h.update(&Self::encode_trace_flat(trace));
        h.finalize().into()
    }

    // ============================================================
    // sha3_commit : bit_trace x trace_hash -> bitstring
    // ============================================================

    fn sha3_commit(trace: &[Vec<F>], trace_hash: &[u8; 32]) -> [u8; 32] {
        let mut h = Sha3_256::new();
        h.update(b"TRACE_BYTES_COMMIT_V1");
        h.update(trace_hash);
        h.update(&Self::encode_trace_flat(trace));
        h.finalize().into()
    }

    // ============================================================
    // Poseidon sponge bound to trace_hash
    // ============================================================

    fn poseidon_hash_with_ds(
        inputs: &[F],
        params: &PoseidonParams,
        trace_hash: &[u8; 32],
    ) -> F {
        let mut state = [F::zero(); T];

        // ✅ Correct Goldilocks-safe domain separation:
        // Use first 64 bits of trace_hash
        let mut ds_bytes = [0u8; 8];
        ds_bytes.copy_from_slice(&trace_hash[..8]);
        state[T - 1] = F::from(u64::from_le_bytes(ds_bytes));

        for chunk in inputs.chunks(T - 1) {
            for (i, &x) in chunk.iter().enumerate() {
                state[i] += x;
            }
            permute(&mut state, params);
        }

        state[0]
    }

    // ============================================================
    // Existing API (Poseidon-only commitment)
    // ============================================================

    pub fn commit(&self, trace: &[Vec<F>]) -> F {
        let trace_hash = Self::sha3_trace(trace);
        self.commit_with_hash(trace, &trace_hash)
    }

    fn commit_with_hash(&self, trace: &[Vec<F>], trace_hash: &[u8; 32]) -> F {
        let leaves_bytes = Self::encode_trace_rows(trace);

        let mut level: Vec<F> = leaves_bytes
            .iter()
            .map(|bytes| {
                let fields: Vec<F> = bytes
                    .chunks_exact(8)
                    .map(|chunk| {
                        let mut arr = [0u8; 8];
                        arr.copy_from_slice(chunk);
                        F::from(u64::from_le_bytes(arr))
                    })
                    .collect();

                Self::poseidon_hash_with_ds(&fields, &self.params, trace_hash)
            })
            .collect();

        while level.len() > 1 {
            let mut next = Vec::new();
            for chunk in level.chunks(self.arity) {
                let parent =
                    Self::poseidon_hash_with_ds(chunk, &self.params, trace_hash);
                next.push(parent);
            }
            level = next;
        }

        level[0]
    }

    // ============================================================
    // ✅ Dual commitment (SHA3 + Poseidon)
    // ============================================================

    pub fn dual_commit(&self, trace: &[Vec<F>]) -> DualCommitment {
        let trace_hash = Self::sha3_trace(trace);
        let sha_commit = Self::sha3_commit(trace, &trace_hash);
        let poseidon_root = self.commit_with_hash(trace, &trace_hash);

        DualCommitment {
            sha_commit,
            poseidon_root,
            trace_hash,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merkle_commit_roundtrip() {
        let mc = MerkleCommitment::with_default_params();

        let trace = vec![
            vec![F::from(1u64), F::from(2u64), F::from(3u64)],
            vec![F::from(4u64), F::from(5u64), F::from(6u64)],
            vec![F::from(7u64), F::from(8u64), F::from(9u64)],
            vec![F::from(10u64), F::from(11u64), F::from(12u64)],
        ];

        let root1 = mc.commit(&trace);
        let root2 = mc.commit(&trace);

        assert_eq!(root1, root2);
    }

    #[test]
    fn dual_commit_deterministic() {
        let mc = MerkleCommitment::with_default_params();

        let trace = vec![
            vec![F::from(42u64)],
            vec![F::from(7u64)],
        ];

        let c1 = mc.dual_commit(&trace);
        let c2 = mc.dual_commit(&trace);

        assert_eq!(c1, c2);
    }

    #[test]
    fn poseidon_commit_binds_trace_hash() {
        let mc = MerkleCommitment::with_default_params();

        let t1 = vec![vec![F::from(1u64)]];
        let t2 = vec![vec![F::from(2u64)]];

        let c1 = mc.dual_commit(&t1);
        let c2 = mc.dual_commit(&t2);

        assert_ne!(c1.poseidon_root, c2.poseidon_root);
        assert_ne!(c1.trace_hash, c2.trace_hash);
    }
}