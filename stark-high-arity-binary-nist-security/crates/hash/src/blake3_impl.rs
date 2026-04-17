use crate::traits::StarkHasher;
use crate::digest_types::Digest32;

const MERKLE_INTERIOR_PREFIX: u8 = 0x01;

#[derive(Clone, Debug)]
pub struct Blake3Hasher;

impl StarkHasher for Blake3Hasher {
    type Digest = Digest32;
    const DIGEST_SIZE: usize = 32;
    const COLLISION_SECURITY_BITS: usize = 128;
    const HASH_ID: u8 = 0x10;

    fn hash(data: &[u8]) -> Digest32 {
        let result = blake3::hash(data);
        Digest32(*result.as_bytes())
    }

    fn merge(left: &Digest32, right: &Digest32) -> Digest32 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&[MERKLE_INTERIOR_PREFIX]);
        hasher.update(left.as_ref());
        hasher.update(right.as_ref());
        let result = hasher.finalize();
        Digest32(*result.as_bytes())
    }

    fn hash_many(slices: &[&[u8]]) -> Digest32 {
        let mut hasher = blake3::Hasher::new();
        for s in slices {
            hasher.update(s);
        }
        let result = hasher.finalize();
        Digest32(*result.as_bytes())
    }
}
