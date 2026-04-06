//! cube_ext.rs — F[X] / (X³ − W)  generic over the base field.
//!
//! Elements:  c0 + c1·α + c2·α²   where  α³ = W.

extern crate alloc;
use alloc::vec::Vec;
use std::hash::Hash;
use std::fmt::Debug;

use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use ark_goldilocks::Goldilocks as Fp;

use crate::tower_field::TowerField;

use ark_ff::{Zero, One};

impl<C: CubeExtConfig> Zero for CubeExt<C> {
    fn zero() -> Self {
        Self {
            c: [C::Base::zero(), C::Base::zero(), C::Base::zero()],
            _marker: core::marker::PhantomData,
        }
    }
    fn is_zero(&self) -> bool {
        self.c[0].is_zero() && self.c[1].is_zero() && self.c[2].is_zero()
    }
}

impl<C: CubeExtConfig> One for CubeExt<C> {
    fn one() -> Self {
        Self {
            c: [C::Base::one(), C::Base::zero(), C::Base::zero()],
            _marker: core::marker::PhantomData,
        }
    }
}

// ────────────────────────────────────────────────────────────────────
//  Configuration trait
// ────────────────────────────────────────────────────────────────────

pub trait CubeExtConfig: Clone + Copy + PartialEq + Eq + Hash + Debug + Send + Sync {
    type Base: TowerField;

    /// Cubic non-residue W such that X³ − W is irreducible over Base.
    fn nonresidue() -> Self::Base;

    /// Multiply by W (override for fast paths).
    #[inline]
    fn mul_by_nonresidue(x: Self::Base) -> Self::Base {
        Self::nonresidue() * x
    }
}

// ────────────────────────────────────────────────────────────────────
//  Goldilocks cubic extension config:  X³ − 7
// ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GoldilocksCubeConfig;

impl CubeExtConfig for GoldilocksCubeConfig {
    type Base = Fp;

    #[inline]
    fn nonresidue() -> Fp {
        Fp::from(7u64)
    }
}

/// Concrete cubic extension over Goldilocks: F_p[α]/(α³ − 7).
pub type CubicExt = CubeExt<GoldilocksCubeConfig>;

// ────────────────────────────────────────────────────────────────────
//  CubeExt<C>
// ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeExt<C: CubeExtConfig> {
    pub c: [C::Base; 3],
    _marker: PhantomData<C>,
}

impl<C: CubeExtConfig> fmt::Debug for CubeExt<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CubeExt({:?}, {:?}, {:?})", self.c[0], self.c[1], self.c[2])
    }
}

impl<C: CubeExtConfig> fmt::Display for CubeExt<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} + {}·α + {}·α²)", self.c[0], self.c[1], self.c[2])
    }
}

impl<C: CubeExtConfig> Default for CubeExt<C> {
    fn default() -> Self {
        Self::create(C::Base::zero(), C::Base::zero(), C::Base::zero())
    }
}

impl<C: CubeExtConfig> CubeExt<C> {
    #[inline]
    pub fn create(c0: C::Base, c1: C::Base, c2: C::Base) -> Self {
        Self { c: [c0, c1, c2], _marker: PhantomData }
    }

    /// Alias for `create`.
    #[inline]
    pub fn new(c0: C::Base, c1: C::Base, c2: C::Base) -> Self {
        Self::create(c0, c1, c2)
    }

    /// Embed a base-field element as (x, 0, 0).
    #[inline]
    pub fn from_base(x: C::Base) -> Self {
        Self::create(x, C::Base::zero(), C::Base::zero())
    }

    /// The element α = (0, 1, 0) satisfying α³ = W.
    #[inline]
    pub fn alpha() -> Self {
        Self::create(C::Base::zero(), C::Base::one(), C::Base::zero())
    }

    /// True if the element lies in the base field (c1 = c2 = 0).
    #[inline]
    pub fn is_base_field(&self) -> bool {
        self.c[1].is_zero() && self.c[2].is_zero()
    }

    /// Serialize to little-endian bytes (8 bytes per Fp component).
    pub fn to_bytes_le(&self) -> Vec<u8> {
        use ark_ff::PrimeField;
        let comps = self.to_fp_components();
        let mut bytes = Vec::with_capacity(comps.len() * 8);
        for fp in comps {
            bytes.extend_from_slice(&fp.into_bigint().0[0].to_le_bytes());
        }
        bytes
    }

    /// Deserialize from little-endian bytes.
    pub fn from_bytes_le_array(bytes: &[u8]) -> Self {
        let d = 3 * C::Base::DEGREE;
        assert!(bytes.len() >= d * 8, "not enough bytes for CubeExt deserialization");
        let mut fps = Vec::with_capacity(d);
        for i in 0..d {
            let start = i * 8;
            let chunk: [u8; 8] = bytes[start..start + 8].try_into().unwrap();
            let val = u64::from_le_bytes(chunk);
            fps.push(Fp::from(val));
        }
        Self::from_fp_components(&fps).expect("invalid byte representation for CubeExt")
    }

    // ── Multiplication  (schoolbook, α³ = W) ───────────────────────
    //   c0 = a0·b0 + W·(a1·b2 + a2·b1)
    //   c1 = a0·b1 + a1·b0 + W·a2·b2
    //   c2 = a0·b2 + a1·b1 + a2·b0
    #[inline]
    fn mul_impl(&self, rhs: &Self) -> Self {
        let (a0, a1, a2) = (self.c[0], self.c[1], self.c[2]);
        let (b0, b1, b2) = (rhs.c[0],  rhs.c[1],  rhs.c[2]);

        let c0 = a0 * b0 + C::mul_by_nonresidue(a1 * b2 + a2 * b1);
        let c1 = a0 * b1 + a1 * b0 + C::mul_by_nonresidue(a2 * b2);
        let c2 = a0 * b2 + a1 * b1 + a2 * b0;

        Self::create(c0, c1, c2)
    }

    // ── Norm:  N = a0³ + W·a1³ + W²·a2³ − 3W·a0·a1·a2  ∈ Base ────
    pub fn norm(&self) -> C::Base {
        let (a0, a1, a2) = (self.c[0], self.c[1], self.c[2]);
        let w  = C::nonresidue();
        let w2 = w * w;
        let three_w = (w + w) + w; // 3·W without needing From<u64> on Base

        a0 * a0 * a0
            + C::mul_by_nonresidue(a1 * a1 * a1)
            + w2 * (a2 * a2 * a2)
            - three_w * a0 * a1 * a2
    }
}

// ────────────────────────────────────────────────────────────────────
//  TowerField
// ────────────────────────────────────────────────────────────────────

impl<C: CubeExtConfig> TowerField for CubeExt<C> {
    const DEGREE: usize = 3; // overridden per-config for towers



    fn from_fp(x: Fp) -> Self {
        Self::create(C::Base::from_fp(x), C::Base::zero(), C::Base::zero())
    }


    fn invert(&self) -> Option<Self> {
        let n = self.norm();
        let n_inv = n.invert()?;
        let (a0, a1, a2) = (self.c[0], self.c[1], self.c[2]);

        let d0 = (a0 * a0 - C::mul_by_nonresidue(a1 * a2)) * n_inv;
        let d1 = (C::mul_by_nonresidue(a2 * a2) - a0 * a1) * n_inv;
        let d2 = (a1 * a1 - a0 * a2) * n_inv;

        Some(Self::create(d0, d1, d2))
    }

    fn sq(&self) -> Self {
        let (a0, a1, a2) = (self.c[0], self.c[1], self.c[2]);
        let two_a0 = a0 + a0;

        let c0 = a0 * a0 + C::mul_by_nonresidue(a1 * a2 + a1 * a2);
        let c1 = two_a0 * a1 + C::mul_by_nonresidue(a2 * a2);
        let c2 = two_a0 * a2 + a1 * a1;

        Self::create(c0, c1, c2)
    }

    fn frobenius(&self) -> Self {
        // φ(c0 + c1·α + c2·α²) = φ(c0) + φ(c1)·ω·α + φ(c2)·ω²·α²
        // where ω = W^{(p-1)/3}.
        //
        // For the base-case (Base = Fp), φ is the identity on coefficients
        // and ω is a cube root of unity in Fp.
        //
        // For higher towers, the concrete config must supply ω.
        // A default that works when Base = Fp:
        let omega = C::nonresidue().pow_u64(6_148_914_689_804_861_440u64);
        let omega2 = omega * omega;
        Self::create(
            self.c[0].frobenius(),
            self.c[1].frobenius() * omega,
            self.c[2].frobenius() * omega2,
        )
    }

    fn to_fp_components(&self) -> Vec<Fp> {
        let mut v = self.c[0].to_fp_components();
        v.extend(self.c[1].to_fp_components());
        v.extend(self.c[2].to_fp_components());
        v
    }

    fn from_fp_components(c: &[Fp]) -> Option<Self> {
        let d = C::Base::DEGREE;
        if c.len() < 3 * d { return None; }
        let c0 = C::Base::from_fp_components(&c[..d])?;
        let c1 = C::Base::from_fp_components(&c[d..2 * d])?;
        let c2 = C::Base::from_fp_components(&c[2 * d..3 * d])?;
        Some(Self::create(c0, c1, c2))
    }
}

// ────────────────────────────────────────────────────────────────────
//  Operator impls  (identical pattern to quad_ext.rs)
// ────────────────────────────────────────────────────────────────────

impl<C: CubeExtConfig> Add for CubeExt<C> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::create(
            self.c[0] + rhs.c[0],
            self.c[1] + rhs.c[1],
            self.c[2] + rhs.c[2],
        )
    }
}
impl<C: CubeExtConfig> AddAssign for CubeExt<C> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.c[0] += rhs.c[0]; self.c[1] += rhs.c[1]; self.c[2] += rhs.c[2];
    }
}
impl<C: CubeExtConfig> Sub for CubeExt<C> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::create(
            self.c[0] - rhs.c[0],
            self.c[1] - rhs.c[1],
            self.c[2] - rhs.c[2],
        )
    }
}
impl<C: CubeExtConfig> SubAssign for CubeExt<C> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.c[0] -= rhs.c[0]; self.c[1] -= rhs.c[1]; self.c[2] -= rhs.c[2];
    }
}
impl<C: CubeExtConfig> Neg for CubeExt<C> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self { Self::create(-self.c[0], -self.c[1], -self.c[2]) }
}
impl<C: CubeExtConfig> Mul for CubeExt<C> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self { self.mul_impl(&rhs) }
}
impl<C: CubeExtConfig> MulAssign for CubeExt<C> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) { *self = self.mul_impl(&rhs); }
}
impl<C: CubeExtConfig> Div for CubeExt<C> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self * rhs.invert().expect("division by zero in CubeExt")
    }
}
impl<C: CubeExtConfig> DivAssign for CubeExt<C> {
    fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
}
impl<C: CubeExtConfig> Mul<Fp> for CubeExt<C> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Fp) -> Self {
        let s = C::Base::from_fp(rhs);
        Self::create(self.c[0] * s, self.c[1] * s, self.c[2] * s)
    }
}

// ────────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{Field, MontFp};

    /// Concrete type alias — shadows the generic `CubeExt<C>` from `use super::*`
    /// so every test in this module operates over Goldilocks.
    type CubeExt = super::CubeExt<GoldilocksCubeConfig>;

    /// The nonresidue W = 7 as an Fp element.
    fn w() -> Fp {
        GoldilocksCubeConfig::nonresidue()
    }

    /// Verify that W is not a cube in F_p, i.e. W^{(p-1)/3} ≠ 1.
    fn verify_cubic_non_residue() {
        let exp = 6_148_914_689_804_861_440u64; // (p-1)/3
        let result = w().pow([exp]);
        assert_ne!(result, Fp::ONE, "W={:?} is a cube in F_p — not a valid non-residue", w());
    }

    /// Return a primitive cube root of unity ω in F_p.
    /// Since W is a cubic non-residue, W^{(p-1)/3} has multiplicative order 3.
    fn cube_root_of_unity() -> Fp {
        let exp = 6_148_914_689_804_861_440u64; // (p-1)/3
        w().pow([exp])
    }

    #[test]
    fn test_w_is_cubic_non_residue() {
        verify_cubic_non_residue();
    }

    #[test]
    fn test_cube_root_of_unity() {
        let omega = cube_root_of_unity();
        assert_ne!(omega, Fp::ONE);
        assert_eq!(omega * omega * omega, Fp::ONE);
        // ω² + ω + 1 = 0
        assert_eq!(omega * omega + omega + Fp::ONE, Fp::ZERO);
    }

    #[test]
    fn test_mul_identity() {
        let a = CubeExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        let one = CubeExt::one();
        assert_eq!(a * one, a);
        assert_eq!(one * a, a);
    }

    #[test]
    fn test_mul_zero() {
        let a = CubeExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        let zero = CubeExt::zero();
        assert_eq!(a * zero, zero);
    }

    #[test]
    fn test_add_sub_roundtrip() {
        let a = CubeExt::new(MontFp!("10"), MontFp!("20"), MontFp!("30"));
        let b = CubeExt::new(MontFp!("1"), MontFp!("2"), MontFp!("3"));
        assert_eq!((a + b) - b, a);
        assert_eq!((a - b) + b, a);
    }

    #[test]
    fn test_negation() {
        let a = CubeExt::new(MontFp!("10"), MontFp!("20"), MontFp!("30"));
        assert_eq!(a + (-a), CubeExt::zero());
    }

    #[test]
    fn test_inverse() {
        let a = CubeExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        let a_inv = a.invert().unwrap();
        let product = a * a_inv;
        assert_eq!(product, CubeExt::one());
    }

    #[test]
    fn test_inverse_of_base() {
        let a = CubeExt::from_base(MontFp!("42"));
        let a_inv = a.invert().unwrap();
        assert_eq!(a * a_inv, CubeExt::one());
        assert!(a_inv.is_base_field());
    }

    #[test]
    fn test_zero_has_no_inverse() {
        assert!(CubeExt::zero().invert().is_none());
    }

    #[test]
    fn test_norm_of_base_is_cube() {
        // N(a) = a³  when a ∈ F_p
        let a: Fp = MontFp!("5");
        let ext = CubeExt::from_base(a);
        assert_eq!(ext.norm(), a * a * a);
    }

    #[test]
    fn test_frobenius_cubed_is_identity() {
        let a = CubeExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        let a_frob3 = a.frobenius().frobenius().frobenius();
        assert_eq!(a_frob3, a, "φ³ should be the identity on F_{{p^3}}");
    }

    #[test]
    fn test_frobenius_is_ring_homomorphism() {
        let a = CubeExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        let b = CubeExt::new(MontFp!("7"), MontFp!("11"), MontFp!("13"));
        // φ(a·b) = φ(a)·φ(b)
        assert_eq!((a * b).frobenius(), a.frobenius() * b.frobenius());
        // φ(a+b) = φ(a)+φ(b)
        assert_eq!((a + b).frobenius(), a.frobenius() + b.frobenius());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let a = CubeExt::new(MontFp!("42"), MontFp!("99"), MontFp!("7"));
        let bytes = a.to_bytes_le();
        let b = CubeExt::from_bytes_le_array(&bytes);
        assert_eq!(a, b);
    }

    #[test]
    fn test_alpha_cubed_is_w() {
        let alpha = CubeExt::alpha();
        let alpha3 = alpha * alpha * alpha;
        assert_eq!(alpha3, CubeExt::from_base(w()));
    }

    #[test]
    fn test_mul_associative() {
        let a = CubeExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        let b = CubeExt::new(MontFp!("7"), MontFp!("11"), MontFp!("13"));
        let c = CubeExt::new(MontFp!("17"), MontFp!("19"), MontFp!("23"));
        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn test_mul_commutative() {
        let a = CubeExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        let b = CubeExt::new(MontFp!("7"), MontFp!("11"), MontFp!("13"));
        assert_eq!(a * b, b * a);
    }

    #[test]
    fn test_distributive() {
        let a = CubeExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        let b = CubeExt::new(MontFp!("7"), MontFp!("11"), MontFp!("13"));
        let c = CubeExt::new(MontFp!("17"), MontFp!("19"), MontFp!("23"));
        assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn test_extension_field_size() {
        // |F_{p^3}| should have ~192-bit elements (3 × 64-bit limbs)
        // This is a structural test: serialization is 24 bytes = 192 bits
        let a = CubeExt::new(MontFp!("3"), MontFp!("5"), MontFp!("2"));
        assert_eq!(a.to_bytes_le().len(), 24);
    }

    mod cubic_ext_sanity {
        use crate::cubic_ext::CubicExt as CubeExt;
        use crate::tower_field::TowerField;
        use ark_goldilocks::Goldilocks as F;

        #[test]
        fn test_no_zero_divisors() {
            let a = CubeExt::new(F::from(1u64), F::from(0u64), F::from(0u64));
            let b = CubeExt::new(F::from(0u64), F::from(1u64), F::from(0u64));
            let product = a * b;

            assert!(
                product != CubeExt::new(F::from(0u64), F::from(0u64), F::from(0u64)),
                "CubeExt has zero divisors — this is componentwise, not a field!"
            );
            assert_eq!(product, b, "1 * α should equal α");
        }

        #[test]
        fn test_alpha_squared() {
            let alpha = CubeExt::new(F::from(0u64), F::from(1u64), F::from(0u64));
            let alpha_sq = alpha * alpha;

            let expected = CubeExt::new(F::from(0u64), F::from(0u64), F::from(1u64));
            assert_eq!(alpha_sq, expected, "α² should be (0, 0, 1)");
        }

        #[test]
        fn test_alpha_cubed_equals_w() {
            let alpha = CubeExt::new(F::from(0u64), F::from(1u64), F::from(0u64));
            let alpha_cubed = alpha * alpha * alpha;

            let expected = CubeExt::from_base(F::from(7u64));
            assert_eq!(
                alpha_cubed, expected,
                "α³ should equal W (the cubic non-residue). Got {:?}",
                alpha_cubed
            );
        }

        #[test]
        fn test_inverse_exists() {
            let a = CubeExt::new(
                F::from(3u64),
                F::from(5u64),
                F::from(11u64),
            );
            let a_inv = a.invert().expect("nonzero element should have an inverse");
            let product = a * a_inv;

            let one = CubeExt::from_base(F::from(1u64));
            assert_eq!(product, one, "a * a⁻¹ should equal 1");
        }

        #[test]
        fn test_cross_terms() {
            let one_plus_alpha = CubeExt::new(
                F::from(1u64),
                F::from(1u64),
                F::from(0u64),
            );
            let sq = one_plus_alpha * one_plus_alpha;

            let expected = CubeExt::new(
                F::from(1u64),
                F::from(2u64),
                F::from(1u64),
            );
            assert_eq!(
                sq, expected,
                "(1 + α)² should be 1 + 2α + α², got {:?}",
                sq
            );
        }
    }
}