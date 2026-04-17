use core::fmt::Debug;
use serde::{Serialize, Deserialize};

/// Marker trait for a fixed-size digest value.
pub trait StarkDigest:
    AsRef<[u8]>
    + AsMut<[u8]>
    + Clone
    + Copy
    + Default
    + Eq
    + PartialEq
    + Ord
    + PartialOrd
    + Debug
    + Send
    + Sync
    + Serialize
    + for<'de> Deserialize<'de>
    + 'static
{
    /// Digest length in bytes.
    const SIZE: usize;

    /// Create from a byte slice, panicking if length mismatches.
    fn from_bytes(bytes: &[u8]) -> Self;
}

/// Cryptographic hash function abstraction.
///
/// This trait is the single point of parameterization for all
/// Merkle tree, FRI, and Fiat-Shamir operations.
pub trait StarkHasher: Clone + Debug + Send + Sync + 'static {
    /// Output digest type.
    type Digest: StarkDigest;

    /// Digest size in bytes (convenience alias).
    const DIGEST_SIZE: usize;

    /// Classical collision security in bits.
    /// SHA3-256 → 128, SHA3-384 → 192, SHA3-512 → 256.
    const COLLISION_SECURITY_BITS: usize;

    /// Hash identifier byte for proof serialization.
    /// 0x01 = SHA3-256, 0x02 = SHA3-384, 0x03 = SHA3-512, 0x10 = BLAKE3.
    const HASH_ID: u8;

    /// Hash an arbitrary byte slice.
    fn hash(data: &[u8]) -> Self::Digest;

    /// Hash two digests together (Merkle interior node).
    /// Convention: left || right with a domain separator byte.
    fn merge(left: &Self::Digest, right: &Self::Digest) -> Self::Digest;

    /// Hash a sequence of byte slices without intermediate allocation.
    fn hash_many(slices: &[&[u8]]) -> Self::Digest;
}
