//! Hash function abstraction for the STARK prover/verifier.

mod traits;
mod digest_types;

#[cfg(any(feature = "sha3-256", feature = "sha3-384", feature = "sha3-512"))]
pub mod sha3_impl;

#[cfg(feature = "blake3-hash")]
pub mod blake3_impl;

pub mod selected;

pub use traits::StarkHasher;
pub use digest_types::{Digest32, Digest48, Digest64};
pub use selected::{SelectedHasher, HASH_BYTES};
pub use sha3;
