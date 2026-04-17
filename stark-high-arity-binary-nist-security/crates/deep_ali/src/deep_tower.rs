//! Fp³ = F × F × F for DEEP-FRI (Option A).
//!
//! This is NOT a field extension.
//! It is a direct product ring used to amplify verifier randomness
//! and eliminate the Block–Tiwari small-field attack.
//!
//! All arithmetic is componentwise.

use ark_ff::{Field, One, Zero};
use ark_goldilocks::Goldilocks as F;
use core::ops::{Add, Sub, Mul};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp3 {
    pub a0: F,
    pub a1: F,
    pub a2: F,
}

impl Fp3 {
    /// Zero element.
    #[inline]
    pub fn zero() -> Self {
        Self {
            a0: F::zero(),
            a1: F::zero(),
            a2: F::zero(),
        }
    }

    /// One element (multiplicative identity).
    #[inline]
    pub fn one() -> Self {
        Self {
            a0: F::one(),
            a1: F::one(),
            a2: F::one(),
        }
    }

    /// Diagonal embedding of base field.
    #[inline]
    pub fn from_base(x: F) -> Self {
        Self { a0: x, a1: x, a2: x }
    }

    /// Componentwise inversion.
    #[inline]
    pub fn inv(self) -> Self {
        Self {
            a0: self.a0.inverse().unwrap(),
            a1: self.a1.inverse().unwrap(),
            a2: self.a2.inverse().unwrap(),
        }
    }
}

/* ---------- Trait impls ---------- */

impl Add for Fp3 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            a0: self.a0 + rhs.a0,
            a1: self.a1 + rhs.a1,
            a2: self.a2 + rhs.a2,
        }
    }
}

impl Sub for Fp3 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            a0: self.a0 - rhs.a0,
            a1: self.a1 - rhs.a1,
            a2: self.a2 - rhs.a2,
        }
    }
}

impl Mul for Fp3 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            a0: self.a0 * rhs.a0,
            a1: self.a1 * rhs.a1,
            a2: self.a2 * rhs.a2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn random_fp3(rng: &mut impl Rng) -> Fp3 {
        Fp3 {
            a0: F::from(rng.gen::<u64>()),
            a1: F::from(rng.gen::<u64>()),
            a2: F::from(rng.gen::<u64>()),
        }
    }

    #[test]
    fn test_add_sub_roundtrip() {
        let mut rng = StdRng::seed_from_u64(1);

        for _ in 0..1000 {
            let x = random_fp3(&mut rng);
            let y = random_fp3(&mut rng);

            let z = x.add(y).sub(y);
            assert_eq!(z, x);
        }
    }

    #[test]
    fn test_mul_identity() {
        let mut rng = StdRng::seed_from_u64(2);

        for _ in 0..1000 {
            let x = random_fp3(&mut rng);
            assert_eq!(x.mul(Fp3::one()), x);
            assert_eq!(Fp3::one().mul(x), x);
        }
    }

    #[test]
    fn test_base_field_embedding() {
        let mut rng = StdRng::seed_from_u64(3);

        for _ in 0..1000 {
            let a = F::from(rng.gen::<u64>());
            let b = F::from(rng.gen::<u64>());

            let fa = Fp3::from_base(a);
            let fb = Fp3::from_base(b);

            let prod = fa.mul(fb);

            // Direct product ring with diagonal embedding:
            // (a,a,a) * (b,b,b) = (ab, ab, ab) = from_base(a*b)
            assert_eq!(prod, Fp3::from_base(a * b));
        }
    }

    #[test]
    fn test_inverse_correctness() {
        let mut rng = StdRng::seed_from_u64(4);

        for _ in 0..1000 {
            let x = random_fp3(&mut rng);
            if x == Fp3::zero() {
                continue;
            }

            let one = x.mul(x.inv());
            assert_eq!(one, Fp3::one());
        }
    }


    #[test]
    fn test_deep_identity_equation() {
        let mut rng = StdRng::seed_from_u64(6);

        for _ in 0..100 {
            let f_i = F::from(rng.gen::<u64>());
            let f_0 = F::from(rng.gen::<u64>());
            let x   = F::from(rng.gen::<u64>());
            let z   = random_fp3(&mut rng);

            let num = Fp3::from_base(f_i - f_0);
            let denom = Fp3::from_base(x).sub(z);

            if denom == Fp3::zero() {
                continue;
            }

            let q = num.mul(denom.inv());

            assert_eq!(q.mul(denom), num);
        }
    }
}
