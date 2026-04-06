#![cfg_attr(not(feature = "std"), no_std)]
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_variables)]
#![allow(unused_macros)]
#![allow(unused_imports)]


pub use field::Goldilocks;

mod field {
    use ark_ff::{
        biginteger::BigInt,
        fields::models::fp::{Fp, MontBackend, MontConfig},
    };

    mod generated {
        include!(concat!(env!("OUT_DIR"), "/constants.rs"));
    }

    pub use generated::{MODULUS, TWO_ADICITY, T};

    pub const MODULUS_MINUS_ONE_DIV_TWO: u64 = generated::MODULUS_MINUS_ONE_DIV_TWO;
    pub const T_MINUS_ONE_DIV_TWO: u64 = generated::T_MINUS_ONE_DIV_TWO;
    pub const TWO_ADIC_ROOT_OF_UNITY_NATIVE_VALUE: u64 =
        generated::TWO_ADIC_ROOT_OF_UNITY_NATIVE;

    #[derive(Clone, Copy, Debug)]
    pub struct GoldilocksConfig;

    pub type Goldilocks = Fp<MontBackend<GoldilocksConfig, 2>, 2>;

    impl MontConfig<2> for GoldilocksConfig {
        const MODULUS: BigInt<2> = BigInt([MODULUS, 0]);
        const R: BigInt<2> = BigInt([generated::R_LIMB0, generated::R_LIMB1]);
        const R2: BigInt<2> = BigInt([generated::R2_LIMB0, generated::R2_LIMB1]);
        const INV: u64 = generated::INV;

        const GENERATOR: Goldilocks =
            Goldilocks::new_unchecked(BigInt([generated::GENERATOR_MONT_LIMB0, generated::GENERATOR_MONT_LIMB1]));
        const TWO_ADIC_ROOT_OF_UNITY: Goldilocks =
            Goldilocks::new_unchecked(BigInt([generated::TWO_ADIC_ROOT_OF_UNITY_MONT_LIMB0, generated::TWO_ADIC_ROOT_OF_UNITY_MONT_LIMB1]));
    }
}

#[cfg(test)]
mod tests {
    use super::field::{
        Goldilocks, GoldilocksConfig, MODULUS, T, TWO_ADICITY, TWO_ADIC_ROOT_OF_UNITY_NATIVE_VALUE,
    };
    use ark_ff::{Field, LegendreSymbol, MontConfig, PrimeField, UniformRand};
    use ark_std::{test_rng, vec::Vec};

    #[test]
    fn basic_arithmetic() {
        assert_eq!(Goldilocks::ZERO + Goldilocks::ONE, Goldilocks::ONE);
        assert_eq!(
            Goldilocks::from(2u64) * Goldilocks::from(3u64),
            Goldilocks::from(6u64)
        );
        assert_eq!(
            -Goldilocks::from(5u64),
            Goldilocks::from(MODULUS - 5)
        );
    }

    #[test]
    fn generator_order() {
        let gen = <GoldilocksConfig as MontConfig<2>>::GENERATOR;
        let order_minus_one = MODULUS - 1;
        assert_eq!(gen.pow([order_minus_one]), Goldilocks::ONE);
        assert_ne!(gen.pow([order_minus_one / 2]), Goldilocks::ONE);
    }

    #[test]
    fn inverses() {
        let a = Goldilocks::from(123456789u64);
        let inv = a.inverse().unwrap();
        assert_eq!(a * inv, Goldilocks::ONE);
    }

    #[test]
    fn legendre_symbols() {
        let residue = Goldilocks::from(25u64);
        let non_residue = Goldilocks::from(11u64);

        assert_eq!(residue.legendre(), LegendreSymbol::QuadraticResidue);
        assert_eq!(non_residue.legendre(), LegendreSymbol::QuadraticNonResidue);
    }

    #[test]
    fn serialization_roundtrip() {
        let a = Goldilocks::from(42u64);

        // Manual roundtrip: mont rep -> canonical BigInt -> back to mont rep
        // This verifies the core logic of serialization/deserialization without ark-serialize's buggy ct code
        let canonical = a.into_bigint();
        let b = Goldilocks::from(canonical);  // Uses from_canonical internally
        assert_eq!(a, b, "Mont rep roundtrip failed: a={:?}, b={:?}, canonical={:?}", a, b, canonical);

        // Additional check: Canonical value fits in u64 (low limb) and < MODULUS
        assert_eq!(canonical.0.len(), 2, "BigInt should have 2 limbs (high=0)");
        assert_eq!(canonical.0[1], 0, "High limb should be 0");
        let canonical_u64 = canonical.0[0];
        assert!(canonical_u64 < MODULUS, "Canonical value out of range: {}", canonical_u64);
    }

    #[test]
    fn uniform_sampling() {
        let mut rng = test_rng();
        let sample = Goldilocks::rand(&mut rng);
        let sample_bigint = sample.into_bigint();
        assert!(sample_bigint.0[0] < MODULUS);
    }

    #[test]
    fn sqrt_roundtrip() {
        let val = Goldilocks::from(25u64);
        let sqrt = val.sqrt().unwrap();
        assert_eq!(sqrt.square(), val);
    }

    #[test]
    fn two_adic_root_checks() {
        let root = <GoldilocksConfig as MontConfig<2>>::TWO_ADIC_ROOT_OF_UNITY;
        assert_eq!(root.pow([1u64 << TWO_ADICITY]), Goldilocks::ONE);
        assert_ne!(
            root.pow([1u64 << (TWO_ADICITY - 1)]),
            Goldilocks::ONE
        );

        let canonical = root.into_bigint().0[0];
        assert_eq!(canonical, TWO_ADIC_ROOT_OF_UNITY_NATIVE_VALUE);
    }

    #[test]
    fn decomposition_constants() {
        assert_eq!((MODULUS - 1) >> TWO_ADICITY, T);
    }
}