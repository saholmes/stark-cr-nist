//! octic_ext.rs — Fp⁸ via a tower of three quadratic extensions
//!
//! Tower:   Fp  ──(quad)──▸  Fp²  ──(quad)──▸  Fp⁴  ──(quad)──▸  Fp⁸
//!          deg 1             deg 2              deg 4              deg 8
//!
//!   Fp² = Fp[i]  / (i² − 7)        w₁ = 7 ∈ Fp   (QNR, proved via QR)
//!   Fp⁴ = Fp²[j] / (j² − i)        w₂ = i ∈ Fp²  (QNR, proved via norm)
//!   Fp⁸ = Fp⁴[k] / (k² − j)        w₃ = j ∈ Fp⁴  (QNR, proved via norm)
//!
//! Each layer reuses the generic `QuadExt<C>` / `QuadExtConfig` machinery
//! from sextic_ext.rs.  The `TowerField` implementation is automatic.

extern crate alloc;

use ark_goldilocks::Goldilocks as Fp;
use ark_ff::{Zero, One, Field};

use crate::tower_field::TowerField;
use crate::sextic_ext::{QuadExt, QuadExtConfig};

// ════════════════════════════════════════════════════════════════════
//  Constants
// ════════════════════════════════════════════════════════════════════

/// (p − 1) / 2  for Goldilocks  p = 2⁶⁴ − 2³² + 1.
const HALF_P_MINUS_1: u64 = 9_223_372_034_707_292_160;

// ════════════════════════════════════════════════════════════════════
//  Layer 1:  Fp² = Fp[i] / (i² − 7)
//
//  7 is a QNR in Fp:  (7/p) = (p/7) = (6/7) = −1.
// ════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GoldilocksQuadConfig;

impl QuadExtConfig for GoldilocksQuadConfig {
    type Base = Fp;
    const TOTAL_DEGREE: usize = 2;

    #[inline]
    fn nonresidue() -> Fp {
        Fp::from(7u64)
    }

    #[inline]
    fn mul_by_nonresidue(x: Fp) -> Fp {
        // 7 · x  — single Fp multiplication
        Fp::from(7u64) * x
    }
}

/// Fp² = Fp[i] / (i² − 7).
pub type Fp2 = QuadExt<GoldilocksQuadConfig>;

// ════════════════════════════════════════════════════════════════════
//  Layer 2:  Fp⁴ = Fp²[j] / (j² − i)
//
//  i = (0,1) ∈ Fp² is a QNR:  i^{(p²−1)/2} = (−1)^{(p+1)/2} = −1
//  since (p+1)/2 is odd for Goldilocks.
// ════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GoldilocksQuarticConfig;

impl QuadExtConfig for GoldilocksQuarticConfig {
    type Base = Fp2;
    const TOTAL_DEGREE: usize = 4;

    #[inline]
    fn nonresidue() -> Fp2 {
        // i = (0, 1) ∈ Fp²
        Fp2::gen()
    }

    #[inline]
    fn mul_by_nonresidue(x: Fp2) -> Fp2 {
        // i · (a + b·i) = a·i + b·i² = b·7 + a·i = (7b, a)
        //
        // Cost: 1 Fp multiplication (by 7) + component swap.
        Fp2::create(
            x.c[1] * Fp::from(7u64),   // 7 · b
            x.c[0],                      // a
        )
    }
}

/// Fp⁴ = Fp²[j] / (j² − i).
pub type Fp4 = QuadExt<GoldilocksQuarticConfig>;

// ════════════════════════════════════════════════════════════════════
//  Layer 3:  Fp⁸ = Fp⁴[k] / (k² − j)
//
//  j = (0,1) ∈ Fp⁴ is a QNR:  j^{(p⁴−1)/2} = (−1)^{(p²+1)/2} = −1
//  since (p²+1)/2 is odd for Goldilocks.
// ════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GoldilocksOcticConfig;

impl QuadExtConfig for GoldilocksOcticConfig {
    type Base = Fp4;
    const TOTAL_DEGREE: usize = 8;

    #[inline]
    fn nonresidue() -> Fp4 {
        // j = (0, 1) ∈ Fp⁴
        Fp4::gen()
    }

    #[inline]
    fn mul_by_nonresidue(x: Fp4) -> Fp4 {
        // j · (c₀ + c₁·j) = c₀·j + c₁·j² = c₁·i + c₀·j = (i·c₁, c₀)
        //
        // i·c₁ is GoldilocksQuarticConfig::mul_by_nonresidue(c₁),
        // which itself costs 1 Fp mul (by 7) + swap.
        //
        // Total cost: 1 Fp multiplication + two component swaps.
        Fp4::create(
            GoldilocksQuarticConfig::mul_by_nonresidue(x.c[1]),   // i · c₁
            x.c[0],                                                 // c₀
        )
    }
}

/// Fp⁸ = Fp⁴[k] / (k² − j).
pub type OcticExt = QuadExt<GoldilocksOcticConfig>;

// ════════════════════════════════════════════════════════════════════
//  Convenience helpers
// ════════════════════════════════════════════════════════════════════

impl Fp2 {
    /// Serialize to 16 little-endian bytes (2 × 8).
    pub fn to_bytes_le(&self) -> Vec<u8> {
        use ark_ff::PrimeField;
        let comps = self.to_fp_components();
        let mut bytes = Vec::with_capacity(comps.len() * 8);
        for fp in comps {
            bytes.extend_from_slice(&fp.into_bigint().0[0].to_le_bytes());
        }
        bytes
    }
}

impl Fp4 {
    /// Serialize to 32 little-endian bytes (4 × 8).
    pub fn to_bytes_le(&self) -> Vec<u8> {
        use ark_ff::PrimeField;
        let comps = self.to_fp_components();
        let mut bytes = Vec::with_capacity(comps.len() * 8);
        for fp in comps {
            bytes.extend_from_slice(&fp.into_bigint().0[0].to_le_bytes());
        }
        bytes
    }
}

impl OcticExt {
    /// Serialize to 64 little-endian bytes (8 × 8).
    pub fn to_bytes_le(&self) -> Vec<u8> {
        use ark_ff::PrimeField;
        let comps = self.to_fp_components();
        let mut bytes = Vec::with_capacity(comps.len() * 8);
        for fp in comps {
            bytes.extend_from_slice(&fp.into_bigint().0[0].to_le_bytes());
        }
        bytes
    }
}

// ════════════════════════════════════════════════════════════════════
//  Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tower_field::TowerField;
    use ark_goldilocks::Goldilocks as Fp;

    // ─── Helpers ────────────────────────────────────────────────

    fn fp2(a: u64, b: u64) -> Fp2 {
        Fp2::new(Fp::from(a), Fp::from(b))
    }

    fn fp4(a: Fp2, b: Fp2) -> Fp4 {
        Fp4::new(a, b)
    }

    fn fp8(a: Fp4, b: Fp4) -> OcticExt {
        OcticExt::new(a, b)
    }

    // ════════════════════════════════════════════════════════════
    //  Non-residue verification
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_7_is_qnr_in_fp() {
        // 7^{(p-1)/2} should equal −1 in Fp.
        let result = Fp::from(7u64).pow([HALF_P_MINUS_1]);
        assert_eq!(result, -Fp::one(), "7 must be a QNR in Fp");
    }

    #[test]
    fn test_i_is_qnr_in_fp2() {
        // i^{(p²−1)/2} = −1.
        //
        // Factor: (p²−1)/2 = (p−1)(p+1)/2.
        // i^{p−1} = i^p / i = (−i)/i = −1.
        // So i^{(p²−1)/2} = (−1)^{(p+1)/2} = −1  since (p+1)/2 is odd.
        //
        // We verify computationally: apply Frobenius to i, check i^p = −i.
        let i = Fp2::gen();
        let i_frob = i.frobenius();
        assert_eq!(i_frob, -i, "Frobenius(i) should be −i");

        // Verify i^{p−1} = −1 via pow_u64
        let i_pm1 = i.pow_u64(HALF_P_MINUS_1);    // i^{(p−1)/2}
        let i_pm1_sq = i_pm1.sq();                  // i^{p−1}
        assert_eq!(i_pm1_sq, -Fp2::one(), "i^{{p−1}} should be −1 in Fp²");
    }

    #[test]
    fn test_j_is_qnr_in_fp4() {
        // j^{p²−1} = −1.
        // Frobenius² on Fp⁴ is conjugation over Fp².
        let j = Fp4::gen();
        let j_frob2 = j.frobenius().frobenius();
        assert_eq!(j_frob2, -j, "σ²(j) should be −j (conjugation over Fp²)");
    }

    // ════════════════════════════════════════════════════════════
    //  Fp² tests
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_fp2_degree() {
        assert_eq!(Fp2::DEGREE, 2);
    }

    #[test]
    fn test_fp2_i_squared_is_7() {
        let i = Fp2::gen();
        let i_sq = i.sq();
        assert_eq!(i_sq, Fp2::from_base(Fp::from(7u64)), "i² should be 7");
    }

    #[test]
    fn test_fp2_inverse() {
        let a = fp2(3, 5);
        let a_inv = a.invert().unwrap();
        assert_eq!(a * a_inv, Fp2::one());
    }

    #[test]
    fn test_fp2_frobenius_is_conjugation() {
        let a = fp2(3, 5);
        let a_frob = a.frobenius();
        assert_eq!(a_frob, a.conjugate(), "Frobenius on Fp² should be conjugation");
    }

    #[test]
    fn test_fp2_frobenius_squared_is_identity() {
        let a = fp2(42, 99);
        assert_eq!(a.frobenius().frobenius(), a, "φ² = id on Fp²");
    }

    #[test]
    fn test_fp2_mul_by_nonresidue() {
        let x = Fp::from(17u64);
        let naive = GoldilocksQuadConfig::nonresidue() * x;
        let fast = GoldilocksQuadConfig::mul_by_nonresidue(x);
        assert_eq!(fast, naive, "fast mul_by_nonresidue should match naive for Fp level");
    }

    #[test]
    fn test_fp2_serialization_roundtrip() {
        let a = fp2(42, 99);
        let bytes = a.to_bytes_le();
        assert_eq!(bytes.len(), 16);
        let b = Fp2::from_bytes_le_array(&bytes);
        assert_eq!(a, b);
    }

    // ════════════════════════════════════════════════════════════
    //  Fp⁴ tests
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_fp4_degree() {
        assert_eq!(Fp4::DEGREE, 4);
    }

    #[test]
    fn test_fp4_j_squared_is_i() {
        let j = Fp4::gen();
        let j_sq = j.sq();
        let expected = Fp4::from_base(Fp2::gen());
        assert_eq!(j_sq, expected, "j² should be i");
    }

    #[test]
    fn test_fp4_mul_identity() {
        let a = fp4(fp2(3, 5), fp2(7, 11));
        assert_eq!(a * Fp4::one(), a);
        assert_eq!(Fp4::one() * a, a);
    }

    #[test]
    fn test_fp4_mul_zero() {
        let a = fp4(fp2(3, 5), fp2(7, 11));
        assert_eq!(a * Fp4::zero(), Fp4::zero());
    }

    #[test]
    fn test_fp4_inverse() {
        let a = fp4(fp2(3, 5), fp2(7, 11));
        let a_inv = a.invert().unwrap();
        assert_eq!(a * a_inv, Fp4::one());
    }

    #[test]
    fn test_fp4_mul_associative() {
        let a = fp4(fp2(3, 5), fp2(7, 11));
        let b = fp4(fp2(13, 17), fp2(19, 23));
        let c = fp4(fp2(29, 31), fp2(37, 41));
        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn test_fp4_mul_commutative() {
        let a = fp4(fp2(3, 5), fp2(7, 11));
        let b = fp4(fp2(13, 17), fp2(19, 23));
        assert_eq!(a * b, b * a);
    }

    #[test]
    fn test_fp4_distributive() {
        let a = fp4(fp2(3, 5), fp2(7, 11));
        let b = fp4(fp2(13, 17), fp2(19, 23));
        let c = fp4(fp2(29, 31), fp2(37, 41));
        assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn test_fp4_negation() {
        let a = fp4(fp2(3, 5), fp2(7, 11));
        assert_eq!(a + (-a), Fp4::zero());
    }

    #[test]
    fn test_fp4_add_sub_roundtrip() {
        let a = fp4(fp2(3, 5), fp2(7, 11));
        let b = fp4(fp2(13, 17), fp2(19, 23));
        assert_eq!((a + b) - b, a);
        assert_eq!((a - b) + b, a);
    }

    #[test]
    fn test_fp4_sq_matches_mul() {
        let a = fp4(fp2(3, 5), fp2(7, 11));
        assert_eq!(a.sq(), a * a);
    }

    #[test]
    fn test_fp4_frobenius_fourth_is_identity() {
        let a = fp4(fp2(3, 5), fp2(7, 11));
        let mut x = a;
        for _ in 0..4 {
            x = x.frobenius();
        }
        assert_eq!(x, a, "φ⁴ = id on Fp⁴");
    }

    #[test]
    fn test_fp4_frobenius_is_ring_hom() {
        let a = fp4(fp2(3, 5), fp2(7, 11));
        let b = fp4(fp2(13, 17), fp2(19, 23));
        assert_eq!((a * b).frobenius(), a.frobenius() * b.frobenius());
        assert_eq!((a + b).frobenius(), a.frobenius() + b.frobenius());
    }

    #[test]
    fn test_fp4_mul_by_nonresidue() {
        let x = fp2(17, 23);
        let beta = GoldilocksQuarticConfig::nonresidue();
        let naive = beta * x;
        let fast = GoldilocksQuarticConfig::mul_by_nonresidue(x);
        assert_eq!(fast, naive, "fast mul_by_nonresidue should match naive for Fp2 level");
    }

    #[test]
    fn test_fp4_zero_has_no_inverse() {
        assert!(Fp4::zero().invert().is_none());
    }

    #[test]
    fn test_fp4_conjugate_product_is_norm() {
        let a = fp4(fp2(3, 5), fp2(7, 11));
        let product = a * a.conjugate();
        let expected = Fp4::from_base(a.norm());
        assert_eq!(product, expected);
    }

    #[test]
    fn test_fp4_serialization_roundtrip() {
        let a = fp4(fp2(42, 99), fp2(7, 13));
        let bytes = a.to_bytes_le();
        assert_eq!(bytes.len(), 32);
        let b = Fp4::from_bytes_le_array(&bytes);
        assert_eq!(a, b);
    }

    #[test]
    fn test_fp4_fp_components_roundtrip() {
        let a = fp4(fp2(42, 99), fp2(7, 13));
        let comps = a.to_fp_components();
        assert_eq!(comps.len(), 4);
        let b = Fp4::from_fp_components(&comps).unwrap();
        assert_eq!(a, b);
    }

    // ════════════════════════════════════════════════════════════
    //  Fp⁸ tests
    // ════════════════════════════════════════════════════════════

    #[test]
    fn test_fp8_degree() {
        assert_eq!(OcticExt::DEGREE, 8);
    }

    #[test]
    fn test_fp8_k_squared_is_j() {
        let k = OcticExt::gen();
        let k_sq = k.sq();
        let expected = OcticExt::from_base(Fp4::gen());
        assert_eq!(k_sq, expected, "k² should be j");
    }

    #[test]
    fn test_fp8_mul_identity() {
        let a = fp8(
            fp4(fp2(3, 5), fp2(7, 11)),
            fp4(fp2(13, 17), fp2(19, 23)),
        );
        assert_eq!(a * OcticExt::one(), a);
        assert_eq!(OcticExt::one() * a, a);
    }

    #[test]
    fn test_fp8_mul_zero() {
        let a = fp8(
            fp4(fp2(3, 5), fp2(7, 11)),
            fp4(fp2(13, 17), fp2(19, 23)),
        );
        assert_eq!(a * OcticExt::zero(), OcticExt::zero());
    }

    #[test]
    fn test_fp8_inverse() {
        let a = fp8(
            fp4(fp2(3, 5), fp2(7, 11)),
            fp4(fp2(13, 17), fp2(19, 23)),
        );
        let a_inv = a.invert().unwrap();
        assert_eq!(a * a_inv, OcticExt::one(), "a · a⁻¹ should be 1");
    }

    #[test]
    fn test_fp8_inverse_of_base() {
        let a = OcticExt::from_fp(Fp::from(42u64));
        let a_inv = a.invert().unwrap();
        assert_eq!(a * a_inv, OcticExt::one());
        assert!(a_inv.is_base_field());
    }

    #[test]
    fn test_fp8_zero_has_no_inverse() {
        assert!(OcticExt::zero().invert().is_none());
    }

    #[test]
    fn test_fp8_add_sub_roundtrip() {
        let a = fp8(
            fp4(fp2(10, 20), fp2(30, 40)),
            fp4(fp2(50, 60), fp2(70, 80)),
        );
        let b = fp8(
            fp4(fp2(1, 2), fp2(3, 4)),
            fp4(fp2(5, 6), fp2(7, 8)),
        );
        assert_eq!((a + b) - b, a);
        assert_eq!((a - b) + b, a);
    }

    #[test]
    fn test_fp8_negation() {
        let a = fp8(
            fp4(fp2(3, 5), fp2(7, 11)),
            fp4(fp2(13, 17), fp2(19, 23)),
        );
        assert_eq!(a + (-a), OcticExt::zero());
    }

    #[test]
    fn test_fp8_mul_associative() {
        let a = fp8(
            fp4(fp2(3, 5), fp2(7, 11)),
            fp4(fp2(13, 17), fp2(19, 23)),
        );
        let b = fp8(
            fp4(fp2(29, 31), fp2(37, 41)),
            fp4(fp2(43, 47), fp2(53, 59)),
        );
        let c = fp8(
            fp4(fp2(61, 67), fp2(71, 73)),
            fp4(fp2(79, 83), fp2(89, 97)),
        );
        assert_eq!((a * b) * c, a * (b * c));
    }

    #[test]
    fn test_fp8_mul_commutative() {
        let a = fp8(
            fp4(fp2(3, 5), fp2(7, 11)),
            fp4(fp2(13, 17), fp2(19, 23)),
        );
        let b = fp8(
            fp4(fp2(29, 31), fp2(37, 41)),
            fp4(fp2(43, 47), fp2(53, 59)),
        );
        assert_eq!(a * b, b * a);
    }

    #[test]
    fn test_fp8_distributive() {
        let a = fp8(
            fp4(fp2(3, 5), fp2(7, 11)),
            fp4(fp2(13, 17), fp2(19, 23)),
        );
        let b = fp8(
            fp4(fp2(29, 31), fp2(37, 41)),
            fp4(fp2(43, 47), fp2(53, 59)),
        );
        let c = fp8(
            fp4(fp2(61, 67), fp2(71, 73)),
            fp4(fp2(79, 83), fp2(89, 97)),
        );
        assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn test_fp8_sq_matches_mul() {
        let a = fp8(
            fp4(fp2(3, 5), fp2(7, 11)),
            fp4(fp2(13, 17), fp2(19, 23)),
        );
        assert_eq!(a.sq(), a * a);
    }

    #[test]
    fn test_fp8_div_roundtrip() {
        let a = fp8(
            fp4(fp2(3, 5), fp2(7, 11)),
            fp4(fp2(13, 17), fp2(19, 23)),
        );
        let b = fp8(
            fp4(fp2(29, 31), fp2(37, 41)),
            fp4(fp2(43, 47), fp2(53, 59)),
        );
        assert_eq!((a / b) * b, a);
    }

    #[test]
    fn test_fp8_conjugate_product_is_norm() {
        let a = fp8(
            fp4(fp2(3, 5), fp2(7, 11)),
            fp4(fp2(13, 17), fp2(19, 23)),
        );
        let product = a * a.conjugate();
        let expected = OcticExt::from_base(a.norm());
        assert_eq!(product, expected, "a · conj(a) = N(a)");
    }

    #[test]
    fn test_fp8_double_conjugate_is_identity() {
        let a = fp8(
            fp4(fp2(3, 5), fp2(7, 11)),
            fp4(fp2(13, 17), fp2(19, 23)),
        );
        assert_eq!(a.conjugate().conjugate(), a);
    }

    // ── Frobenius ───────────────────────────────────────────────

    #[test]
    fn test_fp8_frobenius_eighth_is_identity() {
        let a = fp8(
            fp4(fp2(3, 5), fp2(7, 11)),
            fp4(fp2(13, 17), fp2(19, 23)),
        );
        let mut x = a;
        for _ in 0..8 {
            x = x.frobenius();
        }
        assert_eq!(x, a, "φ⁸ = id on Fp⁸");
    }

    #[test]
    fn test_fp8_frobenius_fourth_is_conjugation() {
        // On Fp⁸ = Fp⁴[k]/(k² − j), the p⁴-Frobenius is conjugation:
        //   k^{p⁴} = k · j^{(p⁴−1)/2} = k · (−1) = −k.
        // So  φ⁴(a + bk) = a − bk = conj(a + bk).
        let a = fp8(
            fp4(fp2(3, 5), fp2(7, 11)),
            fp4(fp2(13, 17), fp2(19, 23)),
        );
        let mut x = a;
        for _ in 0..4 {
            x = x.frobenius();
        }
        assert_eq!(x, a.conjugate(), "φ⁴ should equal conjugation on Fp⁸");
    }

    #[test]
    fn test_fp8_frobenius_is_ring_hom() {
        let a = fp8(
            fp4(fp2(3, 5), fp2(7, 11)),
            fp4(fp2(13, 17), fp2(19, 23)),
        );
        let b = fp8(
            fp4(fp2(29, 31), fp2(37, 41)),
            fp4(fp2(43, 47), fp2(53, 59)),
        );
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
    fn test_fp8_frobenius_fixes_ground_field() {
        let x = OcticExt::from_fp(Fp::from(42u64));
        assert_eq!(x.frobenius(), x);
    }

    // ── mul_by_nonresidue ───────────────────────────────────────

    #[test]
    fn test_fp8_mul_by_nonresidue() {
        let x = fp4(fp2(17, 23), fp2(31, 37));
        let beta = GoldilocksOcticConfig::nonresidue();
        let naive = beta * x;
        let fast = GoldilocksOcticConfig::mul_by_nonresidue(x);
        assert_eq!(fast, naive, "fast mul_by_nonresidue should match naive for Fp4 level");
    }

    // ── No zero divisors ────────────────────────────────────────

    #[test]
    fn test_fp8_no_zero_divisors() {
        // (1 + k) · (1 − k) = 1 − k² = 1 − j  ≠ 0
        let one_plus_k = OcticExt::one() + OcticExt::gen();
        let one_minus_k = OcticExt::one() - OcticExt::gen();
        let product = one_plus_k * one_minus_k;
        assert_ne!(product, OcticExt::zero(), "Fp⁸ should have no zero divisors");

        let expected = OcticExt::one() - OcticExt::from_base(Fp4::gen());
        assert_eq!(product, expected, "(1+k)(1−k) should be 1 − j");
    }

    // ── Serialization ───────────────────────────────────────────

    #[test]
    fn test_fp8_serialization_roundtrip() {
        let a = fp8(
            fp4(fp2(42, 99), fp2(7, 13)),
            fp4(fp2(37, 101), fp2(59, 61)),
        );
        let bytes = a.to_bytes_le();
        assert_eq!(bytes.len(), 64, "Fp⁸ should serialize to 8 × 8 = 64 bytes");
        let b = OcticExt::from_bytes_le_array(&bytes);
        assert_eq!(a, b);
    }

    #[test]
    fn test_fp8_fp_components_roundtrip() {
        let a = fp8(
            fp4(fp2(42, 99), fp2(7, 13)),
            fp4(fp2(37, 101), fp2(59, 61)),
        );
        let comps = a.to_fp_components();
        assert_eq!(comps.len(), 8);
        let b = OcticExt::from_fp_components(&comps).unwrap();
        assert_eq!(a, b);
    }

    // ── pow_u64 ─────────────────────────────────────────────────

    #[test]
    fn test_fp8_pow_basic() {
        let a = fp8(
            fp4(fp2(3, 5), fp2(7, 11)),
            fp4(fp2(13, 17), fp2(19, 23)),
        );
        assert_eq!(a.pow_u64(0), OcticExt::one());
        assert_eq!(a.pow_u64(1), a);
        assert_eq!(a.pow_u64(2), a * a);
        assert_eq!(a.pow_u64(3), a * a * a);
    }

    // ── eval_base_poly ──────────────────────────────────────────

    #[test]
    fn test_fp8_eval_base_poly() {
        // p(x) = 3 + 5x + 7x²  at z = 2 (embedded in Fp⁸)
        let coeffs = vec![Fp::from(3u64), Fp::from(5u64), Fp::from(7u64)];
        let z = OcticExt::from_fp(Fp::from(2u64));
        let result = OcticExt::eval_base_poly(&coeffs, z);
        assert_eq!(result, OcticExt::from_fp(Fp::from(41u64)));
    }

    // ── batch_inverse ───────────────────────────────────────────

    #[test]
    fn test_fp8_batch_inverse() {
        let mut vals = vec![
            fp8(fp4(fp2(3, 5), fp2(7, 11)), fp4(fp2(13, 17), fp2(19, 23))),
            fp8(fp4(fp2(29, 31), fp2(37, 41)), fp4(fp2(43, 47), fp2(53, 59))),
            fp8(fp4(fp2(61, 67), fp2(71, 73)), fp4(fp2(79, 83), fp2(89, 97))),
        ];
        let originals: Vec<OcticExt> = vals.clone();
        OcticExt::batch_inverse(&mut vals);
        for (inv, orig) in vals.iter().zip(originals.iter()) {
            assert_eq!(*inv * *orig, OcticExt::one());
        }
    }

    // ── Cross-layer tower consistency ───────────────────────────

    #[test]
    fn test_tower_embedding_chain() {
        // Fp ↪ Fp² ↪ Fp⁴ ↪ Fp⁸: verify that embeddings compose correctly.
        let x = Fp::from(42u64);

        let in_fp2 = Fp2::from_fp(x);
        let in_fp4 = Fp4::from_fp(x);
        let in_fp8 = OcticExt::from_fp(x);

        // Fp² embedding via Fp⁴'s from_base
        let in_fp4_via_fp2 = Fp4::from_base(in_fp2);
        assert_eq!(in_fp4, in_fp4_via_fp2, "Fp ↪ Fp⁴ should factor through Fp²");

        // Fp⁴ embedding via Fp⁸'s from_base
        let in_fp8_via_fp4 = OcticExt::from_base(in_fp4);
        assert_eq!(in_fp8, in_fp8_via_fp4, "Fp ↪ Fp⁸ should factor through Fp⁴");
    }

    #[test]
    fn test_tower_generator_chain() {
        // k² = j,  j² = i,  i² = 7
        // So k⁴ = j² = i,  k⁸ = i² = 7.
        let k = OcticExt::gen();
        let k2 = k.sq();
        let k4 = k2.sq();
        let k8 = k4.sq();

        // k² should be j (embedded in Fp⁸)
        assert_eq!(k2, OcticExt::from_base(Fp4::gen()));

        // k⁴ should be i (embedded in Fp⁸)
        assert_eq!(k4, OcticExt::from_base(Fp4::from_base(Fp2::gen())));

        // k⁸ should be 7 (embedded in Fp⁸)
        assert_eq!(k8, OcticExt::from_fp(Fp::from(7u64)));
    }

    // ── Scalar mul by Fp ────────────────────────────────────────

    #[test]
    fn test_fp8_scalar_mul() {
        let a = fp8(
            fp4(fp2(3, 5), fp2(7, 11)),
            fp4(fp2(13, 17), fp2(19, 23)),
        );
        let s = Fp::from(10u64);
        let result = a * s;
        let expected = a * OcticExt::from_fp(s);
        assert_eq!(result, expected);
    }
}
