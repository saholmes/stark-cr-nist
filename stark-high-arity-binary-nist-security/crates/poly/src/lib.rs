//! poly crate: thin helpers around ark_poly 0.5.x univariate dense polynomials.
#[allow(unused_imports)]

use ark_ff::{One, Zero};
use ark_goldilocks::Goldilocks as F;
use ark_poly::{
    univariate::DensePolynomial,
    DenseUVPolynomial, // provides constructors like from_coefficients_vec
    Polynomial,        // provides evaluation APIs
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

/// A wrapper around a DensePolynomial<F> with optional serde derives for your own types.
/// Note: Field elements should be serialized canonically via ark_serialize if needed.
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default)]
pub struct Poly {
    pub poly: DensePolynomial<F>,
}

impl Poly {
    /// Construct from coefficients in ascending order: coeffs[0] + coeffs[1] X + ...
    pub fn from_coeffs(coeffs: Vec<F>) -> Self {
        let p = DensePolynomial::from_coefficients_vec(coeffs);
        Self { poly: p }
    }

    /// Alternative: from a slice of coefficients.
    pub fn from_coeffs_slice(coeffs: &[F]) -> Self {
        let p = DensePolynomial::from_coefficients_slice(coeffs);
        Self { poly: p }
    }

    /// Return the coefficients as a slice.
    pub fn coeffs(&self) -> &[F] {
        self.poly.coeffs()
    }

    /// Evaluate the polynomial at x.
    pub fn evaluate(&self, x: &F) -> F {
        self.poly.evaluate(x)
    }

    /// Evaluate the polynomial at multiple points.
    pub fn evaluate_many(&self, points: &[F]) -> Vec<F> {
        #[cfg(feature = "parallel")]
        {
            points
                .par_iter()
                .map(|x| self.poly.evaluate(x))
                .collect()
        }

        #[cfg(not(feature = "parallel"))]
        {
            points.iter().map(|x| self.poly.evaluate(x)).collect()
        }
    }

    /// Degree as usize. In ark-poly 0.5, zero polynomial has degree 0 by convention.
    pub fn degree(&self) -> usize {
        self.poly.degree()
    }

    /// Degree as Option<usize>, returning None for the zero polynomial.
    pub fn degree_opt(&self) -> Option<usize> {
        if self.poly.is_zero() {
            None
        } else {
            Some(self.poly.degree())
        }
    }

    /// Add another polynomial (by value).
    pub fn add(&self, other: &Poly) -> Poly {
        Poly {
            poly: &self.poly + &other.poly,
        }
    }

    /// Multiply by another polynomial.
    pub fn mul(&self, other: &Poly) -> Poly {
        Poly {
            poly: &self.poly * &other.poly,
        }
    }

    /// ✅ FIXED: Scale by a field element.
    pub fn scale(&self, c: F) -> Poly {
        let mut poly = self.poly.clone();
        for coeff in poly.coeffs.iter_mut() {
            *coeff *= c;
        }
        Poly { poly }
    }

    /// Construct the zero polynomial.
    pub fn zero() -> Poly {
        Poly {
            poly: DensePolynomial::from_coefficients_vec(vec![]),
        }
    }

    /// Construct the constant polynomial c.
    pub fn constant(c: F) -> Poly {
        Poly {
            poly: DensePolynomial::from_coefficients_vec(vec![c]),
        }
    }

    /// Construct X (i.e., 0 + 1*X).
    pub fn monomial_x() -> Poly {
        Poly {
            poly: DensePolynomial::from_coefficients_vec(vec![F::zero(), F::one()]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_and_eval() {
        // p(x) = 3 + 2x + x^2
        let p = Poly::from_coeffs(vec![F::from(3u64), F::from(2u64), F::from(1u64)]);
        assert_eq!(p.degree(), 2);
        assert_eq!(p.degree_opt(), Some(2));

        let x = F::from(5u64);
        // 3 + 2*5 + 25 = 38
        let y = p.evaluate(&x);
        assert_eq!(y, F::from(38u64));
    }

    #[test]
    fn add_and_mul() {
        let p = Poly::from_coeffs(vec![F::from(1u64), F::from(1u64)]); // 1 + x
        let q = Poly::from_coeffs(vec![F::from(2u64)]); // 2

        let s = p.add(&q); // (1+x) + 2 = 3 + x
        assert_eq!(s.coeffs(), &[F::from(3u64), F::from(1u64)]);

        let m = p.mul(&q); // (1+x)*2 = 2 + 2x
        assert_eq!(m.coeffs(), &[F::from(2u64), F::from(2u64)]);
    }

    #[test]
    fn constants_and_x() {
        let z = Poly::zero();
        assert_eq!(z.degree(), 0);
        assert_eq!(z.degree_opt(), None);

        let c = Poly::constant(F::from(7u64));
        assert_eq!(c.degree(), 0);
        assert_eq!(c.evaluate(&F::from(10u64)), F::from(7u64));

        let x = Poly::monomial_x();
        assert_eq!(x.degree(), 1);
        assert_eq!(x.evaluate(&F::from(3u64)), F::from(3u64));
    }
}