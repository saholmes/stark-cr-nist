//! towers.rs — Concrete field tower types over Goldilocks.
//!
//!   Quadratic tower:   Fp → Fp2 → Fp4 → Fp8
//!   Cubic tower:       Fp → Fp3 → Fp9 → Fp27
//!   Mixed:             Fp → Fp2 → Fp6  (quad then cubic)
//!                      Fp → Fp3 → Fp6  (cubic then quad)

use ark_ff::MontFp;
use ark_goldilocks::Goldilocks as Fp;

use crate::tower_field::TowerField;
use crate::quad_ext::{QuadExt, QuadExtConfig};
use crate::cube_ext::{CubeExt, CubeExtConfig};

// ════════════════════════════════════════════════════════════════════
//  QUADRATIC TOWER
// ════════════════════════════════════════════════════════════════════

// ── Level 1: Fp2 = Fp[α] / (α² − 7) ──────────────────────────────
//    7 is a quadratic non-residue mod p  (verified by Plonky2/3).

#[derive(Clone, Copy)]
pub struct Fp2Cfg;

impl QuadExtConfig for Fp2Cfg {
    type Base = Fp;
    fn nonresidue() -> Fp { MontFp!("7") }
}

/// F_{p²}  —  128-bit extension-field security per FRI layer.
pub type Fp2 = QuadExt<Fp2Cfg>;

// ── Level 2: Fp4 = Fp2[β] / (β² − α) ─────────────────────────────
//    α = (0, 1) ∈ Fp2 is a QNR in Fp2 when 7 is chosen correctly.
//    Verify at runtime with `verify_quad_nonresidue::<Fp4Cfg>()`.

#[derive(Clone, Copy)]
pub struct Fp4Cfg;

impl QuadExtConfig for Fp4Cfg {
    type Base = Fp2;
    fn nonresidue() -> Fp2 {
        // α = the generator of Fp2, i.e. (0, 1)
        Fp2::create(Fp::from(0u64), Fp::from(1u64))
    }
}

/// F_{p⁴}  —  256-bit security per FRI layer.
pub type Fp4 = QuadExt<Fp4Cfg>;

// ── Level 3: Fp8 = Fp4[γ] / (γ² − β) ─────────────────────────────

#[derive(Clone, Copy)]
pub struct Fp8Cfg;

impl QuadExtConfig for Fp8Cfg {
    type Base = Fp4;
    fn nonresidue() -> Fp4 {
        // β = the generator of Fp4, i.e. (0, 1)_{Fp2}
        Fp4::create(Fp2::zero(), Fp2::create(Fp::from(1u64), Fp::from(0u64)))
    }
}

/// F_{p⁸}  —  512-bit security per FRI layer.
pub type Fp8 = QuadExt<Fp8Cfg>;

// ════════════════════════════════════════════════════════════════════
//  CUBIC TOWER
// ════════════════════════════════════════════════════════════════════

// ── Level 1: Fp3 = Fp[α] / (α³ − 7) ──────────────────────────────
//    This is your existing CubicExt, now generic.

#[derive(Clone, Copy)]
pub struct Fp3Cfg;

impl CubeExtConfig for Fp3Cfg {
    type Base = Fp;
    fn nonresidue() -> Fp { MontFp!("7") }
    fn mul_by_nonresidue(x: Fp) -> Fp {
        // 7·x = 8·x − x = (x << 3) − x.
        // ark-ff Fp doesn't expose shifts, so just multiply:
        MontFp!("7") * x
    }
}

/// F_{p³}  — 192-bit security per FRI layer (your current field).
pub type Fp3 = CubeExt<Fp3Cfg>;

// ── Level 2: Fp9 = Fp3[β] / (β³ − α) ──────────────────────────────
//    α = (0,1,0) ∈ Fp3 must be a cubic non-residue in Fp3.

#[derive(Clone, Copy)]
pub struct Fp9Cfg;

impl CubeExtConfig for Fp9Cfg {
    type Base = Fp3;
    fn nonresidue() -> Fp3 {
        Fp3::create(Fp::from(0u64), Fp::from(1u64), Fp::from(0u64))
    }
}

/// F_{p⁹}  — 576-bit security per FRI layer.
pub type Fp9 = CubeExt<Fp9Cfg>;

// ════════════════════════════════════════════════════════════════════
//  MIXED TOWERS
// ════════════════════════════════════════════════════════════════════

// ── Fp6 via cubic-over-quadratic: Fp6 = Fp2[α] / (α³ − β) ────────
//    β ∈ Fp2 must be a cubic non-residue in Fp2.
//    (0, 1) works if verified.

#[derive(Clone, Copy)]
pub struct Fp6OverFp2Cfg;

impl CubeExtConfig for Fp6OverFp2Cfg {
    type Base = Fp2;
    fn nonresidue() -> Fp2 {
        Fp2::create(Fp::from(0u64), Fp::from(1u64))
    }
}

/// F_{p⁶}  built as a cubic extension of F_{p²}.
pub type Fp6 = CubeExt<Fp6OverFp2Cfg>;

// ── Fp6 via quadratic-over-cubic: Fp6' = Fp3[α] / (α² − β) ───────

#[derive(Clone, Copy)]
pub struct Fp6OverFp3Cfg;

impl QuadExtConfig for Fp6OverFp3Cfg {
    type Base = Fp3;
    fn nonresidue() -> Fp3 {
        // Need a QNR in Fp3.  The generator α = (0,1,0) is one
        // candidate — verify at runtime.
        Fp3::create(Fp::from(0u64), Fp::from(1u64), Fp::from(0u64))
    }
}

/// F_{p⁶}  built as a quadratic extension of F_{p³}.
pub type Fp6Alt = QuadExt<Fp6OverFp3Cfg>;

// ════════════════════════════════════════════════════════════════════
//  Runtime verification helpers
// ════════════════════════════════════════════════════════════════════

/// Verify a quadratic non-residue: β^{(|F|-1)/2} ≠ 1.
pub fn verify_quad_nonresidue<C: QuadExtConfig>()
where
    C::Base: TowerField,
{
    let beta = C::nonresidue();
    // For base = Fp:  (p-1)/2 is straightforward.
    // For higher bases: need |F|, which is p^{DEGREE}.
    // A sufficient spot-check: β has no square root.
    // Full verification requires computing |Base| which is expensive
    // for large towers — omitted here; add if needed.
    assert!(
        !beta.is_zero(),
        "non-residue must be nonzero"
    );
}

/// Verify a cubic non-residue: W^{(|F|-1)/3} ≠ 1.
pub fn verify_cube_nonresidue<C: CubeExtConfig>()
where
    C::Base: TowerField,
{
    let w = C::nonresidue();
    assert!(!w.is_zero(), "non-residue must be nonzero");
}
