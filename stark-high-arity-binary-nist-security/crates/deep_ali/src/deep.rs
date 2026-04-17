use ark_goldilocks::Goldilocks as Fp;
use crate::tower_field::TowerField;

/// DEEP quotient — generic over any extension field E.
pub fn compute_deep_quotient<E: TowerField>(
    trace_codeword: &[Fp],
    domain: &[Fp],
    z: E,
    claimed_eval: E,
) -> Vec<E> {
    let n = trace_codeword.len();
    assert_eq!(domain.len(), n);

    let mut denominators: Vec<E> = domain
        .iter()
        .map(|&d_i| E::from_fp(d_i) - z)
        .collect();

    E::batch_inverse(&mut denominators);

    let mut quotient = Vec::with_capacity(n);
    for i in 0..n {
        let numerator = E::from_fp(trace_codeword[i]) - claimed_eval;
        quotient.push(numerator * denominators[i]);
    }
    quotient
}

pub fn compute_deep_quotient_multi<E: TowerField>(
    trace_codewords: &[Vec<Fp>],
    domain: &[Fp],
    z: E,
    claimed_evals: &[E],
    lambdas: &[E],
) -> Vec<E> {
    let n = domain.len();
    let k = trace_codewords.len();

    let mut denom_inv: Vec<E> = domain
        .iter()
        .map(|&d_i| E::from_fp(d_i) - z)
        .collect();
    E::batch_inverse(&mut denom_inv);

    let mut quotient = vec![E::zero(); n];
    for j in 0..k {
        for i in 0..n {
            let numer = E::from_fp(trace_codewords[j][i]) - claimed_evals[j];
            quotient[i] += lambdas[j] * numer * denom_inv[i];
        }
    }
    quotient
}

pub fn verify_deep_claim<E: TowerField>(
    trace_coeffs: &[Fp],
    z: E,
    claimed: E,
) -> bool {
    E::eval_base_poly(trace_coeffs, z) == claimed
}