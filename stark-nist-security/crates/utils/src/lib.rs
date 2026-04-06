use ark_ff::{BigInteger, PrimeField};
use ark_goldilocks::Goldilocks as F;
use blake3::Hasher;
use serde::{Deserialize, Serialize};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// =======================
/// Field mapping utilities
/// =======================

/// Map arbitrary bytes to a field element by reducing mod p.
///
/// IMPORTANT (arkworks 0.4.x):
/// - Input MUST be exactly BigInt::NUM_LIMBS * 8 bytes
/// - `from_random_bytes` MUST NOT be used
/// - This function is panic-free
pub fn fr_from_le_bytes_mod_p(bytes: &[u8]) -> F {
    const LIMBS: usize = <F as PrimeField>::BigInt::NUM_LIMBS;
    const WIDTH: usize = LIMBS * 8;

    let mut buf = [0u8; WIDTH];
    let n = bytes.len().min(WIDTH);
    buf[..n].copy_from_slice(&bytes[..n]);

    F::from_le_bytes_mod_order(&buf)
}

/// Hash(tag || data) with BLAKE3, then map to Fr.
pub fn fr_from_hash(tag: &str, data: &[u8]) -> F {
    let mut h = Hasher::new();
    h.update(tag.as_bytes());
    h.update(data);
    let out = h.finalize();
    fr_from_le_bytes_mod_p(out.as_bytes())
}

/// Batch variant of `fr_from_hash`.
pub fn fr_from_hash_batch(tag: &str, datas: &[&[u8]]) -> Vec<F> {
    #[cfg(feature = "parallel")]
    {
        datas
            .par_iter()
            .map(|data| fr_from_hash(tag, data))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        datas.iter().map(|data| fr_from_hash(tag, data)).collect()
    }
}

/// =======================
/// Merkle-related helpers
/// =======================

/// Derive a per-node salt for Merkle hashing:
/// salt = H("MT-SALT" || level || node_idx || seed), mapped to Fr.
pub fn salt_for_node(level: usize, node_idx: usize, seed: &[u8; 32]) -> F {
    let mut h = Hasher::new();
    h.update(b"MT-SALT");
    h.update(&level.to_le_bytes());
    h.update(&node_idx.to_le_bytes());
    h.update(seed);
    let out = h.finalize();
    fr_from_le_bytes_mod_p(out.as_bytes())
}

/// Batch variant of `salt_for_node`.
pub fn salts_for_nodes(locations: &[(usize, usize)], seed: &[u8; 32]) -> Vec<F> {
    #[cfg(feature = "parallel")]
    {
        locations
            .par_iter()
            .map(|(level, node_idx)| salt_for_node(*level, *node_idx, seed))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        locations
            .iter()
            .map(|(level, node_idx)| salt_for_node(*level, *node_idx, seed))
            .collect()
    }
}

/// Domain-separation tag for Merkle hashing based on arity.
pub fn ds_tag_for_arity(arity: usize) -> F {
    fr_from_hash("MT-DS", format!("arity-{arity}").as_bytes())
}

/// Batch variant of `ds_tag_for_arity`.
pub fn ds_tags_for_arities(arities: &[usize]) -> Vec<F> {
    #[cfg(feature = "parallel")]
    {
        arities
            .par_iter()
            .map(|&arity| ds_tag_for_arity(arity))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        arities.iter().map(|&arity| ds_tag_for_arity(arity)).collect()
    }
}

/// =======================
/// Utility types
/// =======================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ByteSize(pub usize);

impl core::fmt::Display for ByteSize {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let n = self.0 as f64;
        let (value, suffix) = if n >= (1 << 30) as f64 {
            (n / (1 << 30) as f64, "GiB")
        } else if n >= (1 << 20) as f64 {
            (n / (1 << 20) as f64, "MiB")
        } else if n >= (1 << 10) as f64 {
            (n / (1 << 10) as f64, "KiB")
        } else {
            (n, "B")
        };
        write!(f, "{value:.2} {suffix}")
    }
}