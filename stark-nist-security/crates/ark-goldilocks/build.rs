use std::{env, fs::File, io::Write, path::Path};

use num_bigint::{BigInt, BigUint, Sign};
use num_integer::Integer;
use num_traits::{Num, One, ToPrimitive};

fn main() {
    let modulus = BigUint::from_str_radix("18446744069414584321", 10).unwrap();
    let one = BigUint::one();
    let two_adicity: u32 = 32;

    // For N=2, R = 2^{128} % modulus
    let r_shift = 128u32;
    let r = (&one << r_shift) % &modulus;
    let r2 = (&r * &r) % &modulus;

    let modulus_minus_one = &modulus - &one;
    let t = &modulus_minus_one >> two_adicity;

    let generator = BigUint::from(7u32);
    let generator_mont = (&generator * &r) % &modulus;

    let root_native = generator.modpow(&t, &modulus);
    let root_mont = (&root_native * &r) % &modulus;

    let modulus_minus_one_div_two = &modulus_minus_one >> 1;
    let t_minus_one_div_two = (&t - &one) >> 1;

    // INV for -1 convention (p * INV ≡ -1 mod 2^64)
    let modulus_bigint = BigInt::from_biguint(Sign::Plus, modulus.clone());
    let m = BigInt::one() << 64;
    let egcd = modulus_bigint.extended_gcd(&m);
    assert!(egcd.gcd.is_one(), "modulus and 2^64 must be coprime");

    let modulus_inv = egcd.x.mod_floor(&m);
    let p_u64 = modulus.to_u64().expect("modulus fits in u64");
    let m128 = 1u128 << 64;
    let mask = m128 - 1;
    let candidate = modulus_inv.to_u64().expect("candidate fits in u64");
    let prod = ((p_u64 as u128).wrapping_mul(candidate as u128)) & mask;
    let target = m128 - 1;
    let inv_u64 = if prod == target {
        candidate
    } else {
        let alt = (0u64).wrapping_sub(candidate);
        let alt_prod = ((p_u64 as u128).wrapping_mul(alt as u128)) & mask;
        if alt_prod == target {
            alt
        } else {
            panic!("No INV for -1 convention");
        }
    };
    assert_eq!(((p_u64 as u128).wrapping_mul(inv_u64 as u128)) & mask, target);
    println!("cargo:warning=Selected INV for -1 convention: {}", inv_u64);

    let write_u64 = |value: &BigUint| value.to_u64().expect("value fits in u64");

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("constants.rs");
    let mut file = File::create(dest_path).unwrap();

    writeln!(file, "pub const MODULUS: u64 = {};", write_u64(&modulus)).unwrap();
    writeln!(file, "pub const R_LIMB0: u64 = {};", write_u64(&r)).unwrap();
    writeln!(file, "pub const R_LIMB1: u64 = 0;").unwrap();
    writeln!(file, "pub const R2_LIMB0: u64 = {};", write_u64(&r2)).unwrap();
    writeln!(file, "pub const R2_LIMB1: u64 = 0;").unwrap();
    writeln!(file, "pub const INV: u64 = {inv_u64};").unwrap();
    writeln!(file, "pub const TWO_ADICITY: u32 = {two_adicity};").unwrap();
    writeln!(file, "pub const T: u64 = {};", write_u64(&t)).unwrap();
    writeln!(file, "pub const GENERATOR_MONT_LIMB0: u64 = {};", write_u64(&generator_mont)).unwrap();
    writeln!(file, "pub const GENERATOR_MONT_LIMB1: u64 = 0;").unwrap();
    writeln!(file, "pub const TWO_ADIC_ROOT_OF_UNITY_NATIVE: u64 = {};", write_u64(&root_native)).unwrap();
    writeln!(file, "pub const TWO_ADIC_ROOT_OF_UNITY_MONT_LIMB0: u64 = {};", write_u64(&root_mont)).unwrap();
    writeln!(file, "pub const TWO_ADIC_ROOT_OF_UNITY_MONT_LIMB1: u64 = 0;").unwrap();
    writeln!(file, "pub const MODULUS_MINUS_ONE_DIV_TWO: u64 = {};", write_u64(&modulus_minus_one_div_two)).unwrap();
    writeln!(file, "pub const T_MINUS_ONE_DIV_TWO: u64 = {};", write_u64(&t_minus_one_div_two)).unwrap();
}