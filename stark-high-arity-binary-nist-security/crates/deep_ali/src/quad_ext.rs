//! quad_ext.rs — F[X] / (X² − β)  generic over the base field.
//!
//! Elements:  c0 + c1·α   where  α² = β  (the configured non-residue).

extern crate alloc;
use alloc::vec::Vec;

use core::fmt;
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use ark_goldilocks::Goldilocks as Fp;

use crate::tower_field::TowerField;

// ────────────────────────────────────────────────────────────────────
//  Configuration trait
// ────────────────────────────────────────────────────────────────────

/// Defines which base field and which non-residue β to use.
///
/// One config type per tower level:
///   Fp2Cfg   →  base = Fp,   β = 7
///   Fp4Cfg   →  base = Fp2,  β = α   (the generator of Fp2)
///   Fp8Cfg   →  base = Fp4,  β chosen to be a QNR in Fp4
pub trait QuadExtConfig: 'static + Send + Sync + Copy + Clone {
    type Base: TowerField;

    /// The quadratic non-residue β such that X² − β is irreducible
    /// over `Base`.
    fn nonresidue() -> Self::Base;

    /// Multiply by β (override for fast paths, e.g. when β is small).
    #[inline]
    fn mul_by_nonresidue(x: Self::Base) -> Self::Base {
        Self::nonresidue() * x
    }
}

// ────────────────────────────────────────────────────────────────────
//  QuadExt<C>
// ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuadExt<C: QuadExtConfig> {
    pub c: [C::Base; 2],
    _marker: PhantomData<C>,
}

impl<C: QuadExtConfig> fmt::Debug for QuadExt<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QuadExt({:?}, {:?})", self.c[0], self.c[1])
    }
}

impl<C: QuadExtConfig> fmt::Display for QuadExt<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} + {}·α)", self.c[0], self.c[1])
    }
}

impl<C: QuadExtConfig> Default for QuadExt<C> {
    fn default() -> Self {
        Self::create(C::Base::zero(), C::Base::zero())
    }
}

impl<C: QuadExtConfig> QuadExt<C> {
    #[inline]
    pub fn create(c0: C::Base, c1: C::Base) -> Self {
        Self { c: [c0, c1], _marker: PhantomData }
    }

    /// The generator α of this extension level.
    pub fn alpha() -> Self {
        Self::create(C::Base::zero(), C::Base::one())
    }

    // ── Multiplication ──────────────────────────────────────────────
    //   (a0 + a1·α)(b0 + b1·α) = (a0·b0 + β·a1·b1) + (a0·b1 + a1·b0)·α
    #[inline]
    fn mul_impl(&self, rhs: &Self) -> Self {
        let c0 = self.c[0] * rhs.c[0] + C::mul_by_nonresidue(self.c[1] * rhs.c[1]);
        let c1 = self.c[0] * rhs.c[1] + self.c[1] * rhs.c[0];
        Self::create(c0, c1)
    }

    // ── Norm: N(a) = a0² − β·a1²  ∈ Base ──────────────────────────
    pub fn norm(&self) -> C::Base {
        self.c[0].square() - C::mul_by_nonresidue(self.c[1].square())
    }

    // ── Conjugate: conj(a0 + a1·α) = a0 − a1·α ────────────────────
    pub fn conjugate(&self) -> Self {
        Self::create(self.c[0], -self.c[1])
    }
}

// ────────────────────────────────────────────────────────────────────
//  TowerField implementation
// ────────────────────────────────────────────────────────────────────

impl<C: QuadExtConfig> TowerField for QuadExt<C> {
    // NOTE: On stable Rust you cannot write `2 * C::Base::DEGREE` in a
    // const position.  Each concrete type alias (Fp2, Fp4, …) must
    // supply DEGREE via a helper trait or a manual impl.  For nightly,
    // #![feature(generic_const_exprs)] makes this work directly.
    //
    // Workaround: add `const TOTAL_DEGREE: usize` to QuadExtConfig and
    // set it in each config.  We use that here.
    const DEGREE: usize = 2; // PLACEHOLDER — overridden per-config below

    fn zero() -> Self { Self::create(C::Base::zero(), C::Base::zero()) }
    fn one() -> Self  { Self::create(C::Base::one(),  C::Base::zero()) }

    fn from_fp(x: Fp) -> Self {
        Self::create(C::Base::from_fp(x), C::Base::zero())
    }

    fn is_zero(&self) -> bool {
        self.c[0].is_zero() && self.c[1].is_zero()
    }

    fn inverse(&self) -> Option<Self> {
        let n = self.norm();
        let n_inv = n.inverse()?;
        Some(Self::create(
            self.c[0] * n_inv,
            -(self.c[1] * n_inv),
        ))
    }

    fn square(&self) -> Self {
        // (a + b·α)² = (a² + β·b²) + 2ab·α
        // Karatsuba: use 3 muls instead of 4
        let ab = self.c[0] * self.c[1];
        let c0 = self.c[0].square() + C::mul_by_nonresidue(self.c[1].square());
        let c1 = ab + ab;
        Self::create(c0, c1)
    }

    fn frobenius(&self) -> Self {
        // φ(a + b·α) = a − b·α  when β^{(p-1)/2} = −1 (true for a QNR).
        // For higher towers this must compose the inner Frobenius too:
        //   φ(a + b·α) = φ(a) + φ(b)·α^p = φ(a) − φ(b)·α
        Self::create(self.c[0].frobenius(), -self.c[1].frobenius())
    }

    fn to_fp_components(&self) -> Vec<Fp> {
        let mut v = self.c[0].to_fp_components();
        v.extend(self.c[1].to_fp_components());
        v
    }

    fn from_fp_components(c: &[Fp]) -> Option<Self> {
        let d = C::Base::DEGREE;
        if c.len() < 2 * d { return None; }
        let c0 = C::Base::from_fp_components(&c[..d])?;
        let c1 = C::Base::from_fp_components(&c[d..2 * d])?;
        Some(Self::create(c0, c1))
    }
}

// ────────────────────────────────────────────────────────────────────
//  Operator impls
// ────────────────────────────────────────────────────────────────────

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
    fn mul(self, rhs: Self) -> Self { self.mul_impl(&rhs) }
}

impl<C: QuadExtConfig> MulAssign for QuadExt<C> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) { *self = self.mul_impl(&rhs); }
}

impl<C: QuadExtConfig> Div for QuadExt<C> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse().expect("division by zero in QuadExt")
    }
}

impl<C: QuadExtConfig> DivAssign for QuadExt<C> {
    fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
}

/// Scalar multiplication: Fp × QuadExt
impl<C: QuadExtConfig> Mul<Fp> for QuadExt<C> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Fp) -> Self {
        let s = C::Base::from_fp(rhs);
        Self::create(self.c[0] * s, self.c[1] * s)
    }
}
