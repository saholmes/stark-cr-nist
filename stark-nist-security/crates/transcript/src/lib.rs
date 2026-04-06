//! Pluggable Fiat–Shamir transcript with support for multiple hash backends.
//!
//! Supported backends: Poseidon, SHA3-256, SHA3-384, SHA3-512, Blake3.
//! Goldilocks-safe field embedding (64-bit).

use ark_ff::{BigInteger, PrimeField, Zero};
use ark_goldilocks::Goldilocks as F;

// ────────────────────────────────────────────────────────────────────────
//  Domain separation tags
// ────────────────────────────────────────────────────────────────────────

pub mod ds {
    pub const TRANSCRIPT_INIT: &[u8] = b"FSv1-TRANSCRIPT-INIT";
    pub const ABSORB_BYTES:    &[u8] = b"FSv1-ABSORB-BYTES";
    pub const CHALLENGE:       &[u8] = b"FSv1-CHALLENGE";
}

// ────────────────────────────────────────────────────────────────────────
//  Helpers (Goldilocks-safe)
// ────────────────────────────────────────────────────────────────────────

#[inline]
fn bytes_to_field_u64(bytes: &[u8]) -> F {
    let mut le = [0u8; 8];
    let n = bytes.len().min(8);
    le[..n].copy_from_slice(&bytes[..n]);
    F::from(u64::from_le_bytes(le))
}

fn domain_tag_to_field(tag: &[u8]) -> F {
    bytes_to_field_u64(tag)
}

fn bytes_to_field_words(bytes: &[u8]) -> Vec<F> {
    bytes.chunks(8).map(bytes_to_field_u64).collect()
}

// ────────────────────────────────────────────────────────────────────────
//  Hash backend trait
// ────────────────────────────────────────────────────────────────────────

pub trait HashBackend {
    fn name(&self) -> &'static str;

    /// Native digest output length in bytes (32, 48, or 64).
    fn digest_len(&self) -> usize;

    fn absorb_bytes(&mut self, bytes: &[u8]);
    fn absorb_field(&mut self, x: F);

    /// Squeeze a single field-element challenge (64-bit for Goldilocks).
    fn challenge(&mut self, label: &[u8]) -> F;

    /// Squeeze a full challenge digest whose length matches `digest_len()`.
    ///
    /// The default implementation derives bytes from repeated field-element
    /// squeezes (8 bytes each).  Backends with native wide output (SHA3,
    /// Blake3) override this to return the raw digest.
    fn challenge_bytes(&mut self, label: &[u8]) -> Vec<u8> {
        let n = self.digest_len();
        let words = (n + 7) / 8;
        let mut out = Vec::with_capacity(n);
        for i in 0..words as u8 {
            let mut sub_label = label.to_vec();
            sub_label.push(b'/');
            sub_label.push(i);
            let f = self.challenge(&sub_label);
            let le = f.into_bigint().to_bytes_le();
            out.extend_from_slice(&le[..le.len().min(8)]);
        }
        out.truncate(n);
        out
    }
}

// ────────────────────────────────────────────────────────────────────────
//  Poseidon backend
// ────────────────────────────────────────────────────────────────────────

pub mod poseidon {
    pub use ::poseidon::*;
}

mod poseidon_backend {
    use super::*;
    use ::poseidon::{permute, PoseidonParams, RATE, T};

    pub struct PoseidonBackend {
        pub(crate) state: [F; T],
        pub(crate) pos: usize,
        pub(crate) params: PoseidonParams,
    }

    impl PoseidonBackend {
        pub fn new(params: PoseidonParams, init_label: &[u8]) -> Self {
            let mut s = Self {
                state: [F::zero(); T],
                pos: 0,
                params,
            };
            s.state[T - 1] = super::domain_tag_to_field(super::ds::TRANSCRIPT_INIT);
            s.absorb_bytes(init_label);
            s
        }

        fn absorb_field_internal(&mut self, x: F) {
            if self.pos == RATE {
                permute(&mut self.state, &self.params);
                self.pos = 0;
            }
            self.state[self.pos] += x;
            self.pos += 1;
        }

        fn squeeze(&mut self) -> F {
            permute(&mut self.state, &self.params);
            self.pos = 0;
            self.state[0]
        }
    }

    impl super::HashBackend for PoseidonBackend {
        fn name(&self) -> &'static str { "poseidon" }
        fn digest_len(&self) -> usize { 32 }

        fn absorb_bytes(&mut self, bytes: &[u8]) {
            self.absorb_field_internal(super::domain_tag_to_field(super::ds::ABSORB_BYTES));
            for w in super::bytes_to_field_words(bytes) {
                self.absorb_field_internal(w);
            }
        }

        fn absorb_field(&mut self, x: F) {
            self.absorb_field_internal(x);
        }

        fn challenge(&mut self, label: &[u8]) -> F {
            self.absorb_field_internal(super::domain_tag_to_field(super::ds::CHALLENGE));
            self.absorb_bytes(label);
            self.squeeze()
        }

        // Uses default challenge_bytes (derive from field squeezes)
    }

    pub fn default_params() -> PoseidonParams {
        ::poseidon::params::generate_params_t17_x5(b"POSEIDON-T17-X5-TRANSCRIPT")
    }

    pub(crate) use PoseidonBackend as Backend;
}

// ────────────────────────────────────────────────────────────────────────
//  SHA3 backends (256, 384, 512)
// ────────────────────────────────────────────────────────────────────────

mod sha3_backend {
    use super::*;
    use sha3::Digest;

    macro_rules! sha3_variant {
        ($struct_name:ident, $hasher:ty, $label:expr, $len:expr) => {
            #[derive(Clone)]
            pub struct $struct_name {
                h: $hasher,
            }

            impl $struct_name {
                pub fn new(init_label: &[u8]) -> Self {
                    let mut h = <$hasher>::new();
                    Digest::update(&mut h, super::ds::TRANSCRIPT_INIT);
                    Digest::update(&mut h, init_label);
                    Self { h }
                }
            }

            impl super::HashBackend for $struct_name {
                fn name(&self) -> &'static str { $label }
                fn digest_len(&self) -> usize { $len }

                fn absorb_bytes(&mut self, bytes: &[u8]) {
                    Digest::update(&mut self.h, super::ds::ABSORB_BYTES);
                    Digest::update(&mut self.h, bytes);
                }

                fn absorb_field(&mut self, x: F) {
                    let le = x.into_bigint().to_bytes_le();
                    self.absorb_bytes(&le[..8.min(le.len())]);
                }

                fn challenge(&mut self, label: &[u8]) -> F {
                    let mut h2 = self.h.clone();
                    Digest::update(&mut h2, super::ds::CHALLENGE);
                    Digest::update(&mut h2, label);
                    let out = h2.finalize();
                    super::bytes_to_field_u64(&out[..8])
                }

                fn challenge_bytes(&mut self, label: &[u8]) -> Vec<u8> {
                    let mut h2 = self.h.clone();
                    Digest::update(&mut h2, super::ds::CHALLENGE);
                    Digest::update(&mut h2, label);
                    h2.finalize().to_vec()
                }
            }
        };
    }

    sha3_variant!(Sha3_256Backend, sha3::Sha3_256, "sha3-256", 32);
    sha3_variant!(Sha3_384Backend, sha3::Sha3_384, "sha3-384", 48);
    sha3_variant!(Sha3_512Backend, sha3::Sha3_512, "sha3-512", 64);
}

// ────────────────────────────────────────────────────────────────────────
//  Blake3 backend
// ────────────────────────────────────────────────────────────────────────

mod blake3_backend {
    use super::*;

    #[derive(Clone)]
    pub struct Blake3Backend {
        h: blake3::Hasher,
    }

    impl Blake3Backend {
        pub fn new(init_label: &[u8]) -> Self {
            let mut h = blake3::Hasher::new();
            h.update(super::ds::TRANSCRIPT_INIT);
            h.update(init_label);
            Self { h }
        }
    }

    impl super::HashBackend for Blake3Backend {
        fn name(&self) -> &'static str { "blake3" }
        fn digest_len(&self) -> usize { 32 }

        fn absorb_bytes(&mut self, bytes: &[u8]) {
            self.h.update(super::ds::ABSORB_BYTES);
            self.h.update(bytes);
        }

        fn absorb_field(&mut self, x: F) {
            let le = x.into_bigint().to_bytes_le();
            self.absorb_bytes(&le[..8.min(le.len())]);
        }

        fn challenge(&mut self, label: &[u8]) -> F {
            let mut h2 = self.h.clone();
            h2.update(super::ds::CHALLENGE);
            h2.update(label);
            let out = h2.finalize();
            super::bytes_to_field_u64(out.as_bytes())
        }

        fn challenge_bytes(&mut self, label: &[u8]) -> Vec<u8> {
            let mut h2 = self.h.clone();
            h2.update(super::ds::CHALLENGE);
            h2.update(label);
            h2.finalize().as_bytes().to_vec()
        }
    }

    pub(crate) use Blake3Backend as Backend;
}

// ────────────────────────────────────────────────────────────────────────
//  Public API
// ────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsHash {
    Poseidon,
    Sha3_256,
    Sha3_384,
    Sha3_512,
    Blake3,
}

impl FsHash {
    /// Native digest output length for this hash.
    pub fn digest_len(self) -> usize {
        match self {
            FsHash::Poseidon => 32,
            FsHash::Sha3_256 => 32,
            FsHash::Sha3_384 => 48,
            FsHash::Sha3_512 => 64,
            FsHash::Blake3   => 32,
        }
    }
}

pub use poseidon_backend::default_params;

/// Fiat–Shamir transcript backed by a pluggable hash.
pub struct Transcript {
    backend: Box<dyn HashBackend>,
    hash: FsHash,
}

impl Transcript {
    /// Squeeze a 32-byte challenge digest.
    ///
    /// Panics if the backend's native digest is shorter than 32 bytes
    /// (none of the current backends are).
    #[inline]
    pub fn challenge_bytes_32(&mut self, label: &[u8]) -> [u8; 32] {
        let v = self.backend.challenge_bytes(label);
        v[..32]
            .try_into()
            .expect("backend digest shorter than 32 bytes")
    }

    /// Backward-compatible constructor: Poseidon backend.
    pub fn new(init_label: &[u8], params: poseidon::PoseidonParams) -> Self {
        Self {
            backend: Box::new(poseidon_backend::Backend::new(params, init_label)),
            hash: FsHash::Poseidon,
        }
    }

    /// SHA3-256 transcript (alias for `new_sha3_256`).
    pub fn new_sha3(init_label: &[u8]) -> Self {
        Self::new_sha3_256(init_label)
    }

    /// SHA3 transcript matching the compile-time hash feature selection.
    ///
    /// Uses `hash::HASH_BYTES` to pick the right backend:
    ///   - 32 → SHA3-256
    ///   - 48 → SHA3-384
    ///   - 64 → SHA3-512
    ///
    /// This ensures the transcript digest width always matches the
    /// Merkle tree / commitment digest width selected via Cargo features.
    pub fn new_matching_hash(init_label: &[u8]) -> Self {
        match hash::HASH_BYTES {
            48 => Self::new_sha3_384(init_label),
            64 => Self::new_sha3_512(init_label),
            _  => Self::new_sha3_256(init_label),
        }
    }

    /// SHA3-256 transcript.
    pub fn new_sha3_256(init_label: &[u8]) -> Self {
        Self {
            backend: Box::new(sha3_backend::Sha3_256Backend::new(init_label)),
            hash: FsHash::Sha3_256,
        }
    }

    /// SHA3-384 transcript.
    pub fn new_sha3_384(init_label: &[u8]) -> Self {
        Self {
            backend: Box::new(sha3_backend::Sha3_384Backend::new(init_label)),
            hash: FsHash::Sha3_384,
        }
    }

    /// SHA3-512 transcript.
    pub fn new_sha3_512(init_label: &[u8]) -> Self {
        Self {
            backend: Box::new(sha3_backend::Sha3_512Backend::new(init_label)),
            hash: FsHash::Sha3_512,
        }
    }

    /// Blake3 transcript.
    pub fn new_blake3(init_label: &[u8]) -> Self {
        Self {
            backend: Box::new(blake3_backend::Backend::new(init_label)),
            hash: FsHash::Blake3,
        }
    }

    /// Backward-compatible general constructor.
    ///
    /// `params` is only used when `hash == FsHash::Poseidon`.
    /// For all other backends it is accepted and ignored, so
    /// existing call sites that always pass `default_params()`
    /// continue to compile.
    pub fn with_backend(
        hash: FsHash,
        init_label: &[u8],
        params: poseidon::PoseidonParams,
    ) -> Self {
        match hash {
            FsHash::Poseidon => Self::new(init_label, params),
            FsHash::Sha3_256 => Self::new_sha3_256(init_label),
            FsHash::Sha3_384 => Self::new_sha3_384(init_label),
            FsHash::Sha3_512 => Self::new_sha3_512(init_label),
            FsHash::Blake3   => Self::new_blake3(init_label),
        }
    }

    /// General constructor that only requires params when needed.
    pub fn with_hash(
        hash: FsHash,
        init_label: &[u8],
        params: Option<poseidon::PoseidonParams>,
    ) -> Self {
        match hash {
            FsHash::Poseidon => {
                let p = params.expect("Poseidon backend requires PoseidonParams");
                Self::new(init_label, p)
            }
            FsHash::Sha3_256 => Self::new_sha3_256(init_label),
            FsHash::Sha3_384 => Self::new_sha3_384(init_label),
            FsHash::Sha3_512 => Self::new_sha3_512(init_label),
            FsHash::Blake3   => Self::new_blake3(init_label),
        }
    }

    /// Which hash backend is active.
    pub fn hash(&self) -> FsHash {
        self.hash
    }

    /// Native digest length of the active backend in bytes.
    pub fn digest_len(&self) -> usize {
        self.backend.digest_len()
    }

    #[inline]
    pub fn absorb_bytes(&mut self, bytes: &[u8]) {
        self.backend.absorb_bytes(bytes);
    }

    #[inline]
    pub fn absorb_field(&mut self, x: F) {
        self.backend.absorb_field(x);
    }

    /// Squeeze a field-element challenge (64-bit for Goldilocks).
    #[inline]
    pub fn challenge(&mut self, label: &[u8]) -> F {
        self.backend.challenge(label)
    }

    /// Squeeze a full challenge digest (length = `digest_len()`).
    #[inline]
    pub fn challenge_bytes(&mut self, label: &[u8]) -> Vec<u8> {
        self.backend.challenge_bytes(label)
    }
}

// ────────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poseidon_roundtrip() {
        let params = default_params();
        let mut t1 = Transcript::new(b"TEST", params.clone());
        let mut t2 = Transcript::new(b"TEST", params);

        t1.absorb_field(F::from(42u64));
        t2.absorb_field(F::from(42u64));

        assert_eq!(t1.challenge(b"c"), t2.challenge(b"c"));
        assert_eq!(t1.digest_len(), 32);
        assert_eq!(t1.hash(), FsHash::Poseidon);
    }

    #[test]
    fn sha3_256_roundtrip() {
        let mut t1 = Transcript::new_sha3_256(b"TEST");
        let mut t2 = Transcript::new_sha3_256(b"TEST");

        t1.absorb_bytes(b"hello");
        t2.absorb_bytes(b"hello");

        assert_eq!(t1.challenge(b"c"), t2.challenge(b"c"));
        assert_eq!(t1.digest_len(), 32);
        assert_eq!(t1.hash(), FsHash::Sha3_256);
    }

    #[test]
    fn sha3_384_digest_length() {
        let mut t = Transcript::new_sha3_384(b"TEST");
        t.absorb_bytes(b"data");
        let d = t.challenge_bytes(b"out");

        assert_eq!(d.len(), 48);
        assert_eq!(t.digest_len(), 48);
        assert_eq!(t.hash(), FsHash::Sha3_384);
    }

    #[test]
    fn sha3_512_digest_length() {
        let mut t = Transcript::new_sha3_512(b"TEST");
        t.absorb_bytes(b"data");
        let d = t.challenge_bytes(b"out");

        assert_eq!(d.len(), 64);
        assert_eq!(t.digest_len(), 64);
        assert_eq!(t.hash(), FsHash::Sha3_512);
    }

    #[test]
    fn blake3_roundtrip() {
        let mut t1 = Transcript::new_blake3(b"TEST");
        let mut t2 = Transcript::new_blake3(b"TEST");

        t1.absorb_field(F::from(7u64));
        t2.absorb_field(F::from(7u64));

        assert_eq!(t1.challenge(b"c"), t2.challenge(b"c"));
        assert_eq!(t1.digest_len(), 32);
        assert_eq!(t1.hash(), FsHash::Blake3);
    }

    #[test]
    fn different_backends_produce_different_challenges() {
        let backends = [
            Transcript::new_sha3_256(b"X"),
            Transcript::new_sha3_384(b"X"),
            Transcript::new_sha3_512(b"X"),
            Transcript::new_blake3(b"X"),
        ];

        let challenges: Vec<F> = backends
            .into_iter()
            .map(|mut t| {
                t.absorb_bytes(b"same-data");
                t.challenge(b"c")
            })
            .collect();

        // All pairwise distinct
        for i in 0..challenges.len() {
            for j in (i + 1)..challenges.len() {
                assert_ne!(challenges[i], challenges[j],
                    "backends {} and {} collided", i, j);
            }
        }
    }

    #[test]
    fn sha3_alias_matches_sha3_256() {
        let mut t1 = Transcript::new_sha3(b"ALIAS");
        let mut t2 = Transcript::new_sha3_256(b"ALIAS");

        t1.absorb_bytes(b"data");
        t2.absorb_bytes(b"data");

        assert_eq!(t1.challenge(b"c"), t2.challenge(b"c"));
    }

    #[test]
    fn matching_hash_picks_correct_backend() {
        let t = Transcript::new_matching_hash(b"MATCH");
        assert_eq!(t.digest_len(), hash::HASH_BYTES);
    }

    #[test]
    fn matching_hash_challenge_bytes_length() {
        let mut t = Transcript::new_matching_hash(b"MATCH");
        t.absorb_bytes(b"data");
        let d = t.challenge_bytes(b"out");
        assert_eq!(d.len(), hash::HASH_BYTES);
    }

    #[test]
    fn with_hash_no_params_for_sha3() {
        let t = Transcript::with_hash(FsHash::Sha3_384, b"T", None);
        assert_eq!(t.hash(), FsHash::Sha3_384);
        assert_eq!(t.digest_len(), 48);
    }

    #[test]
    #[should_panic(expected = "Poseidon backend requires PoseidonParams")]
    fn with_hash_panics_without_poseidon_params() {
        let _ = Transcript::with_hash(FsHash::Poseidon, b"T", None);
    }

    #[test]
    fn with_backend_backward_compat() {
        let params = default_params();
        // Old call pattern still works — params ignored for non-Poseidon
        let t = Transcript::with_backend(FsHash::Sha3_512, b"T", params);
        assert_eq!(t.hash(), FsHash::Sha3_512);
        assert_eq!(t.digest_len(), 64);
    }

    #[test]
    fn fs_hash_digest_len_matches_backend() {
        assert_eq!(FsHash::Poseidon.digest_len(), 32);
        assert_eq!(FsHash::Sha3_256.digest_len(), 32);
        assert_eq!(FsHash::Sha3_384.digest_len(), 48);
        assert_eq!(FsHash::Sha3_512.digest_len(), 64);
        assert_eq!(FsHash::Blake3.digest_len(), 32);
    }
}