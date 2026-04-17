//! tower_field.rs — Trait for elements in a tower of extensions over Fp.

extern crate alloc;
use alloc::vec::Vec;

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use ark_ff::{Field, One, Zero};
use ark_goldilocks::Goldilocks as Fp;

/// Every level of the tower implements this trait.
///
/// `Zero` (provides `zero()`, `is_zero()`) and `One` (provides `one()`)
/// come from the supertraits so they never collide with ark-ff's own
/// blanket impls on `Fp`.
///
/// `invert()` and `sq()` are deliberately *not* called `inverse()` /
/// `square()` to avoid ambiguity with `ark_ff::Field` when `Self = Fp`.
pub trait TowerField:
    Sized
    + Clone
    + Copy
    + fmt::Debug
    + fmt::Display
    + PartialEq
    + Eq
    + Default
    + Send
    + Sync
    + Zero
    + One
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Neg<Output = Self>
{
    /// Total extension degree over the ground field Fp.
    /// Fp → 1,  Fp2 → 2,  Fp3 → 3,  Fp4 → 4,  Fp9 → 9, …
    const DEGREE: usize;

    /// Embed a ground-field scalar.
    fn from_fp(x: Fp) -> Self;

    /// Multiplicative inverse (None for zero).
    ///
    /// Named `invert` (not `inverse`) to avoid collision with
    /// `ark_ff::Field::inverse` on the base field.
    fn invert(&self) -> Option<Self>;

    /// Squaring — override in concrete types for fewer multiplications.
    ///
    /// Named `sq` (not `square`) to avoid collision with
    /// `ark_ff::Field::square` on the base field.
    fn sq(&self) -> Self {
        *self * *self
    }

    /// Frobenius endomorphism  φ : x ↦ x^p.
    fn frobenius(&self) -> Self;

    /// Flatten to DEGREE ground-field components (for hashing / Merkle leaves).
    fn to_fp_components(&self) -> Vec<Fp>;

    /// Reconstruct from DEGREE ground-field components.
    fn from_fp_components(c: &[Fp]) -> Option<Self>;

    // ── provided methods ────────────────────────────────────────────

    /// Square-and-multiply exponentiation.
    fn pow_u64(&self, mut exp: u64) -> Self {
        if exp == 0 {
            return Self::one();
        }
        let mut base = *self;
        let mut result = Self::one();
        while exp > 0 {
            if exp & 1 == 1 {
                result = result * base;
            }
            base = base.sq();
            exp >>= 1;
        }
        result
    }

    /// Montgomery batch inversion (1 inversion + O(n) multiplications).
    fn batch_inverse(vals: &mut [Self]) {
        let n = vals.len();
        if n == 0 {
            return;
        }
        let mut prefix = Vec::with_capacity(n);
        prefix.push(vals[0]);
        for i in 1..n {
            prefix.push(prefix[i - 1] * vals[i]);
        }
        let mut inv = prefix[n - 1]
            .invert()
            .expect("batch_inverse: encountered zero element");
        for i in (1..n).rev() {
            let original = vals[i];
            vals[i] = prefix[i - 1] * inv;
            inv = inv * original;
        }
        vals[0] = inv;
    }

    /// Evaluate a ground-field polynomial at an extension-field point (Horner).
    fn eval_base_poly(coeffs: &[Fp], z: Self) -> Self {
        if coeffs.is_empty() {
            return Self::zero();
        }
        let mut acc = Self::from_fp(coeffs[coeffs.len() - 1]);
        for i in (0..coeffs.len() - 1).rev() {
            acc = acc * z + Self::from_fp(coeffs[i]);
        }
        acc
    }

    /// Byte serialization (DEGREE × 8 bytes, little-endian).
    fn to_bytes_le(&self) -> Vec<u8> {
        use ark_ff::PrimeField;
        let comps = self.to_fp_components();
        let mut out = vec![0u8; Self::DEGREE * 8];
        for (i, c) in comps.iter().enumerate() {
            let val: u64 = c.into_bigint().0[0];
            out[i * 8..(i + 1) * 8].copy_from_slice(&val.to_le_bytes());
        }
        out
    }

    /// Deserialize from bytes.
    fn from_bytes_le(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::DEGREE * 8 {
            return None;
        }
        let mut comps = Vec::with_capacity(Self::DEGREE);
        for i in 0..Self::DEGREE {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
            comps.push(Fp::from(u64::from_le_bytes(buf)));
        }
        Self::from_fp_components(&comps)
    }
}

// ── Base case: Fp is the bottom of the tower (degree 1) ─────────────
//
// `Zero` and `One` are already implemented for `Fp` by ark-ff,
// so we only need the tower-specific methods here.

impl TowerField for Fp {
    const DEGREE: usize = 1;

    #[inline]
    fn from_fp(x: Fp) -> Self {
        x
    }

    #[inline]
    fn invert(&self) -> Option<Self> {
        Field::inverse(self)
    }

    #[inline]
    fn sq(&self) -> Self {
        Field::square(self)
    }

    #[inline]
    fn frobenius(&self) -> Self {
        // x^p = x for every element of F_p
        *self
    }

    fn to_fp_components(&self) -> Vec<Fp> {
        vec![*self]
    }

    fn from_fp_components(c: &[Fp]) -> Option<Self> {
        if c.is_empty() {
            None
        } else {
            Some(c[0])
        }
    }
}