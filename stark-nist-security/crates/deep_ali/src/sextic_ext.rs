//! sextic_ext.rs — Generic quadratic extension  F[u] / (u² − β)
//!
//! Primary instantiation:  **Fp⁶ = Fp³[u] / (u² − α)**
//!
//! Tower:   Fp  ──(cubic)──▸  Fp³  ──(quadratic)──▸  Fp⁶
//!          deg 1              deg 3                   deg 6
//!
//! An element of Fp⁶ is  c₀ + c₁·u   with  c₀, c₁ ∈ Fp³  and  u² = β.
//!
//! Arithmetic:
//!   Add:        (a₀,a₁) + (b₀,b₁) = (a₀+b₀, a₁+b₁)
//!   Mul:        (a₀,a₁) · (b₀,b₁) = (a₀b₀ + a₁b₁·β,  a₀b₁ + a₁b₀)
//!   Conjugate:  conj(a₀,a₁) = (a₀, −a₁)
//!   Norm:       N(a₀+a₁u) = a₀² − a₁²·β  ∈ Fp³
//!   Inverse:    (a₀+a₁u)⁻¹ = conj / N

extern crate alloc;
use alloc::vec::Vec;
use std::hash::Hash;
use std::fmt::Debug;

use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use ark_goldilocks::Goldilocks as Fp;
use ark_ff::{Zero, One};

use crate::tower_field::TowerField;
//use crate::cubic_ext::{CubeExt, GoldilocksCubeConfig};
use crate::cubic_ext::{CubeExt, CubeExtConfig, GoldilocksCubeConfig};

// ════════════════════════════════════════════════════════════════════
//  Constants for Goldilocks   p = 2⁶⁴ − 2³² + 1
// ════════════════════════════════════════════════════════════════════

/// (p − 1) / 2 = 2⁶³ − 2³¹
const HALF_P_MINUS_1: u64 = 9_223_372_034_707_292_160;

// ════════════════════════════════════════════════════════════════════
//  QuadExtConfig — configuration trait for the quadratic extension
// ════════════════════════════════════════════════════════════════════

pub trait QuadExtConfig: Clone + Copy + PartialEq + Eq + Hash + Debug + Send + Sync {
    type Base: TowerField;

    /// Total extension degree over Fp.
    /// Must equal  2 × <Base as TowerField>::DEGREE.
    const TOTAL_DEGREE: usize;

    /// Quadratic non-residue β such that X² − β is irreducible over Base.
    fn nonresidue() -> Self::Base;

    /// Multiply by β.  Override for a fast path when β has special structure.
    #[inline]
    fn mul_by_nonresidue(x: Self::Base) -> Self::Base {
        Self::nonresidue() * x
    }
}

// ════════════════════════════════════════════════════════════════════
//  GoldilocksSexticConfig:  Fp³[u] / (u² − α)
//
//  β = α = (0, 1, 0) ∈ Fp³   where  α³ = 7.
//
//  Proof that β is a QNR in Fp³:
//    N_{Fp³/Fp}(α) = 7.  By quadratic reciprocity 7 is a QNR in Fp,
//    hence N(α) is not a square in Fp, hence α is not a square in Fp³.
// ════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GoldilocksSexticConfig;

impl QuadExtConfig for GoldilocksSexticConfig {
    type Base = CubeExt<GoldilocksCubeConfig>;
    const TOTAL_DEGREE: usize = 6;

    #[inline]
    fn nonresidue() -> CubeExt<GoldilocksCubeConfig> {
        // β = α = (0, 1, 0)  where  α³ = 7
        CubeExt::alpha()
    }

    #[inline]
    fn mul_by_nonresidue(x: CubeExt<GoldilocksCubeConfig>) -> CubeExt<GoldilocksCubeConfig> {
        // α · (c₀ + c₁·α + c₂·α²)  =  c₀·α + c₁·α² + c₂·α³
        //                            =  7·c₂ + c₀·α  + c₁·α²
        //
        // Cost: 1 Fp mul (by 7) instead of a full Fp³ mul (9 Fp muls).
        CubeExt::new(
            GoldilocksCubeConfig::mul_by_nonresidue(x.c[2]),   // 7 · c₂
            x.c[0],
            x.c[1],
        )
    }
}

/// Fp⁶ — the sextic extension over Goldilocks.
pub type SexticExt = QuadExt<GoldilocksSexticConfig>;

// ════════════════════════════════════════════════════════════════════
//  QuadExt<C> — generic element of a quadratic extension
// ════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuadExt<C: QuadExtConfig> {
    /// c[0] + c[1]·u,   where  u² = β.
    pub c: [C::Base; 2],
    _marker: PhantomData<C>,
}

// ──────────────── Display / Debug / Default ────────────────────────

impl<C: QuadExtConfig> fmt::Debug for QuadExt<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QuadExt({:?}, {:?})", self.c[0], self.c[1])
    }
}

impl<C: QuadExtConfig> fmt::Display for QuadExt<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} + {}·u)", self.c[0], self.c[1])
    }
}

impl<C: QuadExtConfig> Default for QuadExt<C> {
    fn default() -> Self {
        Self::create(C::Base::zero(), C::Base::zero())
    }
}

// ──────────────── Constructors / helpers ───────────────────────────

impl<C: QuadExtConfig> QuadExt<C> {
    #[inline]
    pub fn create(c0: C::Base, c1: C::Base) -> Self {
        Self { c: [c0, c1], _marker: PhantomData }
    }

    /// Alias for `create`.
    #[inline]
    pub fn new(c0: C::Base, c1: C::Base) -> Self {
        Self::create(c0, c1)
    }

    /// Embed a base-field element as (x, 0).
    #[inline]
    pub fn from_base(x: C::Base) -> Self {
        Self::create(x, C::Base::zero())
    }

    /// The generator  u = (0, 1)  satisfying  u² = β.
    #[inline]
    pub fn gen() -> Self {
        Self::create(C::Base::zero(), C::Base::one())
    }

    /// True if the element lies in the base field (c₁ = 0).
    #[inline]
    pub fn is_base_field(&self) -> bool {
        self.c[1].is_zero()
    }

    /// Conjugate:  (a, b) ↦ (a, −b).
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self::create(self.c[0], -self.c[1])
    }

    /// Norm down to base:  N(a + bu) = a² − b²·β  ∈ Base.
    pub fn norm(&self) -> C::Base {
        self.c[0].sq() - C::mul_by_nonresidue(self.c[1].sq())
    }

    /// Deserialize from little-endian bytes (panics on bad input).
    pub fn from_bytes_le_array(bytes: &[u8]) -> Self {
        let d = C::TOTAL_DEGREE;
        assert!(
            bytes.len() >= d * 8,
            "not enough bytes for QuadExt deserialization"
        );
        let mut fps = Vec::with_capacity(d);
        for i in 0..d {
            let start = i * 8;
            let chunk: [u8; 8] = bytes[start..start + 8].try_into().unwrap();
            fps.push(Fp::from(u64::from_le_bytes(chunk)));
        }
        Self::from_fp_components(&fps).expect("invalid byte representation for QuadExt")
    }

    // ── Multiplication  (Karatsuba, 3 base-muls) ───────────────
    //
    //   (a₀ + a₁u)(b₀ + b₁u) = (a₀b₀ + a₁b₁·β)  +  (a₀b₁ + a₁b₀)·u
    //
    //   v₀ = a₀·b₀,   v₁ = a₁·b₁
    //   c₀ = v₀ + v₁·β
    //   c₁ = (a₀+a₁)(b₀+b₁) − v₀ − v₁
    #[inline]
    fn mul_impl(&self, rhs: &Self) -> Self {
        let v0 = self.c[0] * rhs.c[0];
        let v1 = self.c[1] * rhs.c[1];
        let c0 = v0 + C::mul_by_nonresidue(v1);
        let c1 = (self.c[0] + self.c[1]) * (rhs.c[0] + rhs.c[1]) - v0 - v1;
        Self::create(c0, c1)
    }
}

// ════════════════════════════════════════════════════════════════════
//  Zero / One
// ════════════════════════════════════════════════════════════════════

impl<C: QuadExtConfig> Zero for QuadExt<C> {
    fn zero() -> Self {
        Self::create(C::Base::zero(), C::Base::zero())
    }
    fn is_zero(&self) -> bool {
        self.c[0].is_zero() && self.c[1].is_zero()
    }
}

impl<C: QuadExtConfig> One for QuadExt<C> {
    fn one() -> Self {
        Self::create(C::Base::one(), C::Base::zero())
    }
}

// ════════════════════════════════════════════════════════════════════
//  TowerField implementation — plugs directly into MF‑FRI
// ════════════════════════════════════════════════════════════════════

impl<C: QuadExtConfig> TowerField for QuadExt<C> {
    const DEGREE: usize = C::TOTAL_DEGREE;

    #[inline]
    fn from_fp(x: Fp) -> Self {
        Self::create(C::Base::from_fp(x), C::Base::zero())
    }

    fn invert(&self) -> Option<Self> {
        // (a + bu)⁻¹ = (a, −b) / N   where  N = a² − b²β  ∈ Base
        let n = self.norm();
        let n_inv = n.invert()?;
        Some(Self::create(
            self.c[0] * n_inv,
            -(self.c[1] * n_inv),
        ))
    }

    fn sq(&self) -> Self {
        // (a + bu)² = (a² + b²β) + 2ab · u
        let ab = self.c[0] * self.c[1];
        let c0 = self.c[0].sq() + C::mul_by_nonresidue(self.c[1].sq());
        let c1 = ab + ab;
        Self::create(c0, c1)
    }

    fn frobenius(&self) -> Self {
        // φ(a + bu) = φ(a) + φ(b) · γ · u
        //
        // where  γ = β^{(p−1)/2} ∈ Base.
        //
        // Derivation:  u^p = u · (u²)^{(p−1)/2} = u · β^{(p−1)/2} = γ·u
        //
        // TODO: precompute γ as a constant for hot-path performance.
        let gamma = C::nonresidue().pow_u64(HALF_P_MINUS_1);
        Self::create(
            self.c[0].frobenius(),
            self.c[1].frobenius() * gamma,
        )
    }

    fn to_fp_components(&self) -> Vec<Fp> {
        let mut v = self.c[0].to_fp_components();
        v.extend(self.c[1].to_fp_components());
        v
    }

    fn from_fp_components(c: &[Fp]) -> Option<Self> {
        let d = C::Base::DEGREE;
        if c.len() < 2 * d {
            return None;
        }
        let c0 = C::Base::from_fp_components(&c[..d])?;
        let c1 = C::Base::from_fp_components(&c[d..2 * d])?;
        Some(Self::create(c0, c1))
    }
}

// ════════════════════════════════════════════════════════════════════
//  Operator impls  (same pattern as cubic_ext.rs)
// ════════════════════════════════════════════════════════════════════

impl<C: QuadExtConfig> Add for QuadExt<C> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::create(self.c[0] + rhs.c[0], self.c[1] + rhs.c[1])
    }
}

impl<C: QuadExtConfig> AddAssign for QuadExt<C> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.c[0] += rhs.c[0];
        self.c[1] += rhs.c[1];
    }
}

impl<C: QuadExtConfig> Sub for QuadExt<C> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::create(self.c[0] - rhs.c[0], self.c[1] - rhs.c[1])
    }
}

impl<C: QuadExtConfig> SubAssign for QuadExt<C> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.c[0] -= rhs.c[0];
        self.c[1] -= rhs.c[1];
    }
}

impl<C: QuadExtConfig> Neg for QuadExt<C> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::create(-self.c[0], -self.c[1])
    }
}

impl<C: QuadExtConfig> Mul for QuadExt<C> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.mul_impl(&rhs)
    }
}

impl<C: QuadExtConfig> MulAssign for QuadExt<C> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_impl(&rhs);
    }
}

impl<C: QuadExtConfig> Div for QuadExt<C> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self * rhs.invert().expect("division by zero in QuadExt")
    }
}

impl<C: QuadExtConfig> DivAssign for QuadExt<C> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

/// Scalar multiplication by a ground-field (Fp) element.
impl<C: QuadExtConfig> Mul<Fp> for QuadExt<C> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Fp) -> Self {
        let s = C::Base::from_fp(rhs);
        Self::create(self.c[0] * s, self.c[1] * s)
    }
}

// ════════════════════════════════════════════════════════════════════
//  Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cubic_ext::{CubeExt, GoldilocksCubeConfig};
    use crate::tower_field::TowerField;
    use ark_goldilocks::Goldilocks as Fp;

    type Fp3 = CubeExt<GoldilocksCubeConfig>;
    type Fp6 = QuadExt<GoldilocksSexticConfig>;

    /// Helper to build Fp³ elements tersely.
    fn fp3(a: u64, b: u64, c: u64) -> Fp3 {
        Fp3::new(Fp::from(a), Fp::from(b), Fp::from(c))
    }

    // ── Non-residue verification ────────────────────────────────

    #[test]
    fn test_beta_is_quadratic_nonresidue_in_fp3() {
        // β = α ∈ Fp³.  We verify  β^{(p³−1)/2} = −1.
        //
        // Factor:  (p³−1)/2 = ((p−1)/2) · (p² + p + 1)
        //
        // So  β^{(p³−1)/2} = g^{p²+p+1}   where  g = β^{(p−1)/2}.
        // And  g^{p^k} = frobenius^k(g)  in Fp³.
        let beta = GoldilocksSexticConfig::nonresidue();
        let g    = beta.pow_u64(HALF_P_MINUS_1);
        let g_p  = g.frobenius();
        let g_p2 = g_p.frobenius();
        let result = g_p2 * g_p * g;             // g^{p²+p+1}
        assert_eq!(
            result,
            -Fp3::one(),
            "β = α must be a quadratic non-residue in Fp³"
        );
    }

    // ── Basic constructors ──────────────────────────────────────

    #[test]
    fn test_u_squared_is_beta() {
        let u = Fp6::gen();
        let u_sq = u.sq();
        let expected = Fp6::from_base(GoldilocksSexticConfig::nonresidue());
        assert_eq!(u_sq, expected, "u² should equal β");
    }

    #[test]
    fn test_from_base_is_base_field() {
        let x = fp3(42, 7, 13);
        let ext = Fp6::from_base(x);
        assert!(ext.is_base_field());
    }

    #[test]
    fn test_degree_is_six() {
        assert_eq!(Fp6::DEGREE, 6);
    }

    // ── Ring axioms ─────────────────────────────────────────────

    #[test]
    fn test_mul_identity() {
        let a = Fp6::new(fp3(3, 5, 2), fp3(7, 11, 13));
        let one = Fp6::one();
        assert_eq!(a * one, a);
        assert_eq!(one * a, a);
    }

    #[test]
    fn test_mul_zero() {
        let a = Fp6::new(fp3(3, 5, 2), fp3(7, 11, 13));
        assert_eq!(a * Fp6::zero(), Fp6::zero());
    }

    #[test]
    fn test_add_sub_roundtrip() {
        let a = Fp6::new(fp3(10, 20, 30), fp3(40, 50, 60));
        let b = Fp6::new(fp3(1, 2, 3), fp3(4, 5, 6));
        assert_eq!((a + b) - b, a);
        assert_eq!((a - b) + b, a);
    }

    #[test]
    fn test_negation() {
        let a = Fp6::new(fp3(10, 20, 30), fp3(40, 50, 60));
        assert_eq!(a + (-a), Fp6::zero());
    }

    #[test]
    fn test_mul_associative() {
        let a = Fp6::new(fp3(3, 5, 2), fp3(7, 11, 13));
        let b = Fp6::new(fp3(17, 19, 23), fp3(29, 31, 37));
        let c = Fp6::new(fp3(41, 43, 47), fp3(53, 59, 61));
        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn test_mul_commutative() {
        let a = Fp6::new(fp3(3, 5, 2), fp3(7, 11, 13));
        let b = Fp6::new(fp3(17, 19, 23), fp3(29, 31, 37));
        assert_eq!(a * b, b * a);
    }

    #[test]
    fn test_distributive() {
        let a = Fp6::new(fp3(3, 5, 2), fp3(7, 11, 13));
        let b = Fp6::new(fp3(17, 19, 23), fp3(29, 31, 37));
        let c = Fp6::new(fp3(41, 43, 47), fp3(53, 59, 61));
        assert_eq!(a * (b + c), a * b + a * c);
    }

    // ── Inverse ─────────────────────────────────────────────────

    #[test]
    fn test_inverse() {
        let a = Fp6::new(fp3(3, 5, 2), fp3(7, 11, 13));
        let a_inv = a.invert().unwrap();
        assert_eq!(a * a_inv, Fp6::one());
    }

    #[test]
    fn test_inverse_of_base_element() {
        let a = Fp6::from_base(fp3(42, 17, 3));
        let a_inv = a.invert().unwrap();
        assert_eq!(a * a_inv, Fp6::one());
        assert!(a_inv.is_base_field(), "inverse of base element should stay in base");
    }

    #[test]
    fn test_zero_has_no_inverse() {
        assert!(Fp6::zero().invert().is_none());
    }

    // ── Conjugation and norm ────────────────────────────────────

    #[test]
    fn test_conjugate_product_is_norm() {
        let a = Fp6::new(fp3(3, 5, 2), fp3(7, 11, 13));
        let product = a * a.conjugate();
        let expected = Fp6::from_base(a.norm());
        assert_eq!(product, expected, "a · conj(a) should equal N(a)");
    }

    #[test]
    fn test_norm_of_base_is_square() {
        // N(a + 0·u) = a² − 0·β = a²
        let x = fp3(5, 3, 7);
        let ext = Fp6::from_base(x);
        assert_eq!(ext.norm(), x.sq());
    }

    #[test]
    fn test_double_conjugate_is_identity() {
        let a = Fp6::new(fp3(3, 5, 2), fp3(7, 11, 13));
        assert_eq!(a.conjugate().conjugate(), a);
    }

    // ── Frobenius ───────────────────────────────────────────────

    #[test]
    fn test_frobenius_sixth_power_is_identity() {
        let a = Fp6::new(fp3(3, 5, 2), fp3(7, 11, 13));
        let mut x = a;
        for _ in 0..6 {
            x = x.frobenius();
        }
        assert_eq!(x, a, "φ⁶ should be the identity on Fp⁶");
    }

    #[test]
    fn test_frobenius_cubed_is_conjugation() {
        // On Fp⁶ = Fp³[u]/(u² − β) with β a QNR:
        //   u^{p³} = u · β^{(p³−1)/2} = u · (−1) = −u
        // So  φ³(a + bu) = a − bu = conj(a + bu).
        let a = Fp6::new(fp3(3, 5, 2), fp3(7, 11, 13));
        let frob3 = a.frobenius().frobenius().frobenius();
        assert_eq!(frob3, a.conjugate(), "φ³ should equal conjugation on Fp⁶");
    }

    #[test]
    fn test_frobenius_is_ring_homomorphism() {
        let a = Fp6::new(fp3(3, 5, 2), fp3(7, 11, 13));
        let b = Fp6::new(fp3(17, 19, 23), fp3(29, 31, 37));
        assert_eq!(
            (a * b).frobenius(),
            a.frobenius() * b.frobenius(),
            "Frobenius should be multiplicative"
        );
        assert_eq!(
            (a + b).frobenius(),
            a.frobenius() + b.frobenius(),
            "Frobenius should be additive"
        );
    }

    #[test]
    fn test_frobenius_fixes_ground_field() {
        let x = Fp6::from_fp(Fp::from(42u64));
        assert_eq!(x.frobenius(), x, "Frobenius should fix Fp elements");
    }

    // ── mul_by_nonresidue fast path ─────────────────────────────

    #[test]
    fn test_mul_by_nonresidue_matches_naive() {
        let x = fp3(17, 23, 31);
        let beta = GoldilocksSexticConfig::nonresidue();
        let naive = beta * x;
        let fast  = GoldilocksSexticConfig::mul_by_nonresidue(x);
        assert_eq!(fast, naive, "fast mul_by_nonresidue should match naive");
    }

    // ── Serialization ───────────────────────────────────────────

    #[test]
    fn test_serialization_roundtrip() {
        let a = Fp6::new(fp3(42, 99, 7), fp3(13, 37, 101));
        let bytes = a.to_bytes_le();
        assert_eq!(bytes.len(), 48, "Fp⁶ should serialize to 6 × 8 = 48 bytes");
        let b = Fp6::from_bytes_le_array(&bytes);
        assert_eq!(a, b, "deserialization should roundtrip");
    }

    #[test]
    fn test_fp_components_roundtrip() {
        let a = Fp6::new(fp3(42, 99, 7), fp3(13, 37, 101));
        let comps = a.to_fp_components();
        assert_eq!(comps.len(), 6);
        let b = Fp6::from_fp_components(&comps).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_element_byte_size() {
        let a = Fp6::new(fp3(1, 2, 3), fp3(4, 5, 6));
        assert_eq!(a.to_bytes_le().len(), 48);
    }

    // ── Scalar mul / division ───────────────────────────────────

    #[test]
    fn test_scalar_mul_fp() {
        let a = Fp6::new(fp3(3, 5, 2), fp3(7, 11, 13));
        let s = Fp::from(10u64);
        let result = a * s;
        let expected = Fp6::new(
            fp3(3, 5, 2) * Fp::from(10u64),
            fp3(7, 11, 13) * Fp::from(10u64),
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn test_div_roundtrip() {
        let a = Fp6::new(fp3(3, 5, 2), fp3(7, 11, 13));
        let b = Fp6::new(fp3(17, 19, 23), fp3(29, 31, 37));
        assert_eq!((a / b) * b, a, "a / b * b should equal a");
    }

    // ── Squaring vs multiplication ──────────────────────────────

    #[test]
    fn test_sq_matches_self_mul() {
        let a = Fp6::new(fp3(3, 5, 2), fp3(7, 11, 13));
        assert_eq!(a.sq(), a * a, "sq() should match a * a");
    }

    // ── pow_u64 ─────────────────────────────────────────────────

    #[test]
    fn test_pow_basic() {
        let a = Fp6::new(fp3(3, 5, 2), fp3(7, 11, 13));
        assert_eq!(a.pow_u64(0), Fp6::one());
        assert_eq!(a.pow_u64(1), a);
        assert_eq!(a.pow_u64(2), a * a);
        assert_eq!(a.pow_u64(3), a * a * a);
    }

    // ── batch_inverse ───────────────────────────────────────────

    #[test]
    fn test_batch_inverse() {
        let mut vals = vec![
            Fp6::new(fp3(3, 5, 2), fp3(7, 11, 13)),
            Fp6::new(fp3(17, 19, 23), fp3(29, 31, 37)),
            Fp6::new(fp3(41, 43, 47), fp3(53, 59, 61)),
        ];
        let originals: Vec<Fp6> = vals.clone();
        Fp6::batch_inverse(&mut vals);
        for (inv, orig) in vals.iter().zip(originals.iter()) {
            assert_eq!(*inv * *orig, Fp6::one(), "batch inverse element check");
        }
    }

    // ── eval_base_poly ──────────────────────────────────────────

    #[test]
    fn test_eval_base_poly() {
        // p(x) = 3 + 5x + 7x²  evaluated at z ∈ Fp⁶
        let coeffs = vec![Fp::from(3u64), Fp::from(5u64), Fp::from(7u64)];
        let z = Fp6::new(fp3(2, 0, 0), fp3(0, 0, 0)); // z = 2 (in base field)
        let result = Fp6::eval_base_poly(&coeffs, z);
        // p(2) = 3 + 10 + 28 = 41
        assert_eq!(result, Fp6::from_fp(Fp::from(41u64)));
    }
}
