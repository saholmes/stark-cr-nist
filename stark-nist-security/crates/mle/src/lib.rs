use ark_ff::Zero;
//use ark_pallas::Fr as F;
use ark_goldilocks::Goldilocks as F;

/// Multilinear polynomial represented by its evaluations on the {0,1}^n hypercube,
/// stored in lexicographic order of indices 0..2^n.
/// Indexing convention:
/// - For n variables, index bits correspond to (x_0, x_1, ..., x_{n-1})
/// - Bit i (0-based) indicates the assignment for variable i.
#[derive(Clone, Debug)]
pub struct MLE {
    /// Values at all 2^n Boolean inputs.
    values: Vec<F>,
    /// Number of variables n. Must satisfy values.len() == 2^n.
    n: usize,
}

impl MLE {
    /// Create from values; length must be a power of two.
    pub fn from_values(values: Vec<F>) -> Self {
        assert!(!values.is_empty(), "MLE: empty values");
        assert!(values.len().is_power_of_two(), "MLE: length must be power of two");
        let n = values.len().trailing_zeros() as usize;
        MLE { values, n }
    }

    /// Construct the zero polynomial of n variables.
    pub fn zero(n: usize) -> Self {
        let size = 1usize << n;
        MLE { values: vec![F::zero(); size], n }
    }

    /// Borrow the internal values slice.
    pub fn values(&self) -> &[F] {
        &self.values
    }

    /// Mutable access to the internal values (use with care).
    pub fn values_mut(&mut self) -> &mut [F] {
        &mut self.values
    }

    /// Number of variables.
    pub fn n_vars(&self) -> usize {
        self.n
    }

    /// Evaluate at a point r in F^n using the standard folding:
    /// Repeatedly combine pairs v0, v1 -> (1 - r_i) * v0 + r_i * v1.
    /// Does not mutate self; allocates a temporary buffer shrinking each round.
    pub fn eval(&self, r: &[F]) -> F {
        assert_eq!(r.len(), self.n, "MLE::eval: wrong number of variables");
        if self.n == 0 {
            return self.values[0];
        }

        // Work buffer; copy values then fold down.
        let mut buf = self.values.clone();
        let mut size = buf.len(); // = 2^k at each step

        for (i, &ri) in r.iter().enumerate() {
            debug_assert!(size % 2 == 0, "size should be even at step {}", i);
            let half = size / 2;
            let one_minus = F::from(1u64) - ri;
            for j in 0..half {
                let v0 = buf[2 * j];
                let v1 = buf[2 * j + 1];
                buf[j] = one_minus * v0 + ri * v1;
            }
            size = half;
        }
        debug_assert_eq!(size, 1);
        buf[0]
    }

    /// In-place fold along the last variable with challenge r_i:
    /// v'[j] = (1 - r_i) * v[2j] + r_i * v[2j+1], reducing dimension by 1.
    /// Returns a new MLE with n-1 variables, reusing the original allocation.
    pub fn fold_last(mut self, r_i: F) -> Self {
        if self.n == 0 {
            return self;
        }
        let one_minus = F::from(1u64) - r_i;
        let mut write = 0usize;
        // Combine adjacent pairs in-place to the front of the buffer.
        for read in (0..self.values.len()).step_by(2) {
            let v0 = self.values[read];
            let v1 = self.values[read + 1];
            self.values[write] = one_minus * v0 + r_i * v1;
            write += 1;
        }
        self.values.truncate(write);
        self.n -= 1;
        self
    }

    /// Fix variable at index var_idx to a Boolean bit (false=0, true=1),
    /// returning a new MLE with n-1 variables. This is a view-like projection,
    /// but implemented by selecting either even or odd entries per pair and
    /// compacting.
    pub fn fix(mut self, var_idx: usize, bit: bool) -> Self {
        assert!(var_idx < self.n, "var_idx out of range");
        if self.n == 0 {
            return self;
        }
        // We want to pull out all indices whose bit at var_idx equals `bit`.
        let n = self.n;
        let size = self.values.len();
        let want = if bit { 1usize } else { 0usize };

        // Bit var_idx toggles every stride = 1<<var_idx.
        let stride = 1usize << var_idx;
        let period = stride << 1;

        let mut write = 0usize;
        let mut base = 0usize;
        while base < size {
            let start = base + want * stride;
            let end = start + stride;
            for i in start..end {
                self.values[write] = self.values[i];
                write += 1;
            }
            base += period;
        }
        self.values.truncate(write);
        self.n = n - 1;
        self
    }

    /// Fold in the i-th variable with challenge r_i, combining the two sub-cubes.
    /// This is like `fold_last`, but for an arbitrary variable index; implemented by
    /// strided pairwise combination.
    pub fn fold(self, var_idx: usize, r_i: F) -> Self {
        assert!(var_idx < self.n, "var_idx out of range");
        if var_idx == self.n - 1 {
            return self.fold_last(r_i);
        }
        let mut mle = self;
        let one_minus = F::from(1u64) - r_i;

        let n = mle.n;
        let size = mle.values.len();
        let stride = 1usize << var_idx;
        let period = stride << 1;

        let mut write = 0usize;
        let mut base = 0usize;
        while base < size {
            for o in 0..stride {
                let v0 = mle.values[base + o];
                let v1 = mle.values[base + o + stride];
                mle.values[write] = one_minus * v0 + r_i * v1;
                write += 1;
            }
            base += period;
        }

        mle.values.truncate(write);
        mle.n = n - 1;
        mle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::One;

    // Build an MLE that equals sum of variables: f(x) = x0 + x1 + ... + x_{n-1}.
    fn mle_sum_vars(n: usize) -> MLE {
        let size = 1usize << n;
        let mut vals = vec![F::zero(); size];
        for idx in 0..size {
            let mut s = F::zero();
            for i in 0..n {
                if (idx >> i) & 1 == 1 {
                    s += F::one();
                }
            }
            vals[idx] = s;
        }
        MLE::from_values(vals)
    }

    #[test]
    fn eval_at_boolean_points_matches_table() {
        let n = 3;
        let mle = mle_sum_vars(n);
        for idx in 0..(1usize << n) {
            let mut r = vec![F::zero(); n];
            for i in 0..n {
                if (idx >> i) & 1 == 1 {
                    r[i] = F::one();
                }
            }
            let y = mle.eval(&r);
            assert_eq!(y, mle.values()[idx]);
        }
    }

    #[test]
    fn fold_last_matches_eval() {
        let n = 4;
        let mle = mle_sum_vars(n);
        let r = vec![F::from(2u64), F::from(3u64), F::from(5u64), F::from(7u64)];

        let direct = mle.eval(&r);

        let folded = mle.clone().fold_last(r[n - 1]);
        let direct2 = folded.eval(&r[..n - 1]);

        assert_eq!(direct, direct2);
    }

    #[test]
    fn fold_arbitrary_matches_eval() {
        let n = 4;
        let mle = mle_sum_vars(n);
        let r = vec![F::from(11u64), F::from(13u64), F::from(17u64), F::from(19u64)];

        let direct = mle.eval(&r);

        let folded = mle.clone().fold(1, r[1]);
        let r_rem = vec![r[0], r[2], r[3]];
        let direct2 = folded.eval(&r_rem);

        assert_eq!(direct, direct2);
    }

    #[test]
    fn fix_var_selects_half() {
        let n = 3;
        let mle = mle_sum_vars(n);
        let fixed = mle.clone().fix(1, true);
        assert_eq!(fixed.n_vars(), n - 1);
        for a0 in [false, true] {
            for a2 in [false, true] {
                let r = [
                    if a0 { F::from(1u64) } else { F::from(0u64) },
                    F::from(1u64),
                    if a2 { F::from(1u64) } else { F::from(0u64) },
                ];
                let r_small = [
                    if a0 { F::from(1u64) } else { F::from(0u64) },
                    if a2 { F::from(1u64) } else { F::from(0u64) },
                ];
                assert_eq!(mle.eval(&r), fixed.eval(&r_small));
            }
        }
    }
}
