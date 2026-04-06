//! Compile-time hash selection.
//!
//! Exactly one feature should be active.  If Cargo unifies two
//! (e.g. sha3-256 + sha3-384), the **strongest** one wins.

use sha3::Digest;   // re-export the trait so callers can do Digest::update

// ──────────── SHA3-512 (highest priority) ────────────

#[cfg(feature = "sha3-512")]
pub use sha3::Sha3_512 as SelectedHasher;

#[cfg(feature = "sha3-512")]
pub const HASH_BYTES: usize = 64;

// ──────────── SHA3-384 ────────────

#[cfg(all(feature = "sha3-384", not(feature = "sha3-512")))]
pub use sha3::Sha3_384 as SelectedHasher;

#[cfg(all(feature = "sha3-384", not(feature = "sha3-512")))]
pub const HASH_BYTES: usize = 48;

// ──────────── SHA3-256 ────────────

#[cfg(all(
    feature = "sha3-256",
    not(feature = "sha3-384"),
    not(feature = "sha3-512"),
))]
pub use sha3::Sha3_256 as SelectedHasher;

#[cfg(all(
    feature = "sha3-256",
    not(feature = "sha3-384"),
    not(feature = "sha3-512"),
))]
pub const HASH_BYTES: usize = 32;

// ──────────── BLAKE3 (lowest priority) ────────────

#[cfg(all(
    feature = "blake3-hash",
    not(feature = "sha3-256"),
    not(feature = "sha3-384"),
    not(feature = "sha3-512"),
))]
compile_error!("blake3 support requires a wrapper implementing sha3::Digest — not yet wired");

// ──────────── No hash selected ────────────

#[cfg(not(any(
    feature = "sha3-256",
    feature = "sha3-384",
    feature = "sha3-512",
    feature = "blake3-hash",
)))]
compile_error!(
    "No hash feature enabled!  Pass one of: \
     --features sha3-256 | sha3-384 | sha3-512 | blake3-hash"
);