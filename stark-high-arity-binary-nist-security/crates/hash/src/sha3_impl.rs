use crate::traits::StarkHasher;
use crate::digest_types::{Digest32, Digest48, Digest64};

#[cfg(any(feature = "sha3-256", feature = "sha3-384", feature = "sha3-512"))]
use digest::Digest;

const MERKLE_INTERIOR_PREFIX: u8 = 0x01;

// ──────────────────── SHA3-256 (32-byte, 128-bit collision security) ────────────────────

#[cfg(feature = "sha3-256")]
pub struct Sha3_256Hasher(sha3::Sha3_256);

#[cfg(feature = "sha3-256")]
impl Clone for Sha3_256Hasher {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

#[cfg(feature = "sha3-256")]
impl std::fmt::Debug for Sha3_256Hasher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sha3_256Hasher").finish()
    }
}

#[cfg(feature = "sha3-256")]
impl Sha3_256Hasher {
    pub fn new() -> Self {
        Self(<sha3::Sha3_256 as Digest>::new())
    }

    pub fn update(&mut self, data: &[u8]) {
        Digest::update(&mut self.0, data);
    }

    pub fn finalize(self) -> Digest32 {
        Digest32(self.0.finalize().into())
    }
}

#[cfg(feature = "sha3-256")]
impl StarkHasher for Sha3_256Hasher {
    type Digest = Digest32;
    const DIGEST_SIZE: usize = 32;
    const COLLISION_SECURITY_BITS: usize = 128;
    const HASH_ID: u8 = 0x20;

    fn hash(data: &[u8]) -> Digest32 {
        Digest32(<sha3::Sha3_256 as Digest>::digest(data).into())
    }

    fn merge(left: &Digest32, right: &Digest32) -> Digest32 {
        let mut h = <sha3::Sha3_256 as Digest>::new();
        Digest::update(&mut h, &[MERKLE_INTERIOR_PREFIX]);
        Digest::update(&mut h, left.as_ref());
        Digest::update(&mut h, right.as_ref());
        Digest32(h.finalize().into())
    }

    fn hash_many(slices: &[&[u8]]) -> Digest32 {
        let mut h = <sha3::Sha3_256 as Digest>::new();
        for s in slices {
            Digest::update(&mut h, s);
        }
        Digest32(h.finalize().into())
    }
}

// ──────────────────── SHA3-384 (48-byte, 192-bit collision security) ────────────────────

#[cfg(feature = "sha3-384")]
pub struct Sha3_384Hasher(sha3::Sha3_384);

#[cfg(feature = "sha3-384")]
impl Clone for Sha3_384Hasher {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

#[cfg(feature = "sha3-384")]
impl std::fmt::Debug for Sha3_384Hasher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sha3_384Hasher").finish()
    }
}

#[cfg(feature = "sha3-384")]
impl Sha3_384Hasher {
    pub fn new() -> Self {
        Self(<sha3::Sha3_384 as Digest>::new())
    }

    pub fn update(&mut self, data: &[u8]) {
        Digest::update(&mut self.0, data);
    }

    pub fn finalize(self) -> Digest48 {
        Digest48(self.0.finalize().into())
    }
}

#[cfg(feature = "sha3-384")]
impl StarkHasher for Sha3_384Hasher {
    type Digest = Digest48;
    const DIGEST_SIZE: usize = 48;
    const COLLISION_SECURITY_BITS: usize = 192;
    const HASH_ID: u8 = 0x21;

    fn hash(data: &[u8]) -> Digest48 {
        Digest48(<sha3::Sha3_384 as Digest>::digest(data).into())
    }

    fn merge(left: &Digest48, right: &Digest48) -> Digest48 {
        let mut h = <sha3::Sha3_384 as Digest>::new();
        Digest::update(&mut h, &[MERKLE_INTERIOR_PREFIX]);
        Digest::update(&mut h, left.as_ref());
        Digest::update(&mut h, right.as_ref());
        Digest48(h.finalize().into())
    }

    fn hash_many(slices: &[&[u8]]) -> Digest48 {
        let mut h = <sha3::Sha3_384 as Digest>::new();
        for s in slices {
            Digest::update(&mut h, s);
        }
        Digest48(h.finalize().into())
    }
}

// ──────────────────── SHA3-512 (64-byte, 256-bit collision security) ────────────────────

#[cfg(feature = "sha3-512")]
pub struct Sha3_512Hasher(sha3::Sha3_512);

#[cfg(feature = "sha3-512")]
impl Clone for Sha3_512Hasher {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

#[cfg(feature = "sha3-512")]
impl std::fmt::Debug for Sha3_512Hasher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sha3_512Hasher").finish()
    }
}

#[cfg(feature = "sha3-512")]
impl Sha3_512Hasher {
    pub fn new() -> Self {
        Self(<sha3::Sha3_512 as Digest>::new())
    }

    pub fn update(&mut self, data: &[u8]) {
        Digest::update(&mut self.0, data);
    }

    pub fn finalize(self) -> Digest64 {
        Digest64(self.0.finalize().into())
    }
}

#[cfg(feature = "sha3-512")]
impl StarkHasher for Sha3_512Hasher {
    type Digest = Digest64;
    const DIGEST_SIZE: usize = 64;
    const COLLISION_SECURITY_BITS: usize = 256;
    const HASH_ID: u8 = 0x22;

    fn hash(data: &[u8]) -> Digest64 {
        Digest64(<sha3::Sha3_512 as Digest>::digest(data).into())
    }

    fn merge(left: &Digest64, right: &Digest64) -> Digest64 {
        let mut h = <sha3::Sha3_512 as Digest>::new();
        Digest::update(&mut h, &[MERKLE_INTERIOR_PREFIX]);
        Digest::update(&mut h, left.as_ref());
        Digest::update(&mut h, right.as_ref());
        Digest64(h.finalize().into())
    }

    fn hash_many(slices: &[&[u8]]) -> Digest64 {
        let mut h = <sha3::Sha3_512 as Digest>::new();
        for s in slices {
            Digest::update(&mut h, s);
        }
        Digest64(h.finalize().into())
    }
}