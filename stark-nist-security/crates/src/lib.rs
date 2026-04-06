//! Commitment abstraction with a Merkle implementation.

use field::F;

/// Trait for vector commitments over field elements.
/// Values are F, indices are usize, and proofs/digests are scheme-specific.
pub trait CommitmentScheme {
    type Digest: Clone + core::fmt::Debug + PartialEq + Eq;
    type Proof: Clone + core::fmt::Debug;
    type Aux: Clone + core::fmt::Debug;

    /// Commit to leaves and return the commitment digest and optional auxiliary data.
    /// Aux may store the full tree or nodes needed to open later.
    fn commit(&self, leaves: &[F]) -> (Self::Digest, Self::Aux);

    /// Produce membership proofs for the given indices using the previously produced aux.
    fn open(&self, indices: &[usize], aux: &Self::Aux) -> Self::Proof;

    /// Verify membership proofs against the digest for the provided positions and values.
    fn verify(
        &self,
        root: &Self::Digest,
        indices: &[usize],
        values: &[F],
        proof: &Self::Proof,
    ) -> bool;
}

/// Concrete Merkle-based commitment scheme adapter.
///
/// It assumes the `merkle` crate exposes:
/// - MerkleTree::new(leaves: &[F]) -> Self
/// - MerkleTree::root(&self) -> Digest
/// - MerkleTree::open(indices: &[usize]) -> MerkleProof
/// - MerkleTree::verify(root: &Digest, indices: &[usize], leaves: &[F], proof: &MerkleProof) -> bool
///
/// If your API differs, adjust the adapter accordingly below.
pub struct MerkleCommitment;

impl Default for MerkleCommitment {
    fn default() -> Self {
        MerkleCommitment
    }
}

// Re-export merkle types for convenience.
pub use merkle::{Digest as MerkleDigest, MerkleProof, MerkleTree};

#[derive(Clone, Debug)]
pub struct MerkleAux {
    tree: MerkleTree,
}

impl CommitmentScheme for MerkleCommitment {
    type Digest = MerkleDigest;
    type Proof = MerkleProof;
    type Aux = MerkleAux;

    fn commit(&self, leaves: &[F]) -> (Self::Digest, Self::Aux) {
        let tree = MerkleTree::new(leaves);
        let root = tree.root();
        (root, MerkleAux { tree })
    }

    fn open(&self, indices: &[usize], aux: &Self::Aux) -> Self::Proof {
        aux.tree.open(indices)
    }

    fn verify(
        &self,
        root: &Self::Digest,
        indices: &[usize],
        values: &[F],
        proof: &Self::Proof,
    ) -> bool {
        MerkleTree::verify(root, indices, values, proof)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;

    #[test]
    fn merkle_commit_open_verify_roundtrip() {
        // Build some random leaves.
        let mut rng = ark_std::test_rng();
        let n = 32usize;
        let mut leaves = Vec::with_capacity(n);
        for _ in 0..n {
            leaves.push(F::rand(&mut rng));
        }

        let scheme = MerkleCommitment::default();
        let (root, aux) = scheme.commit(&leaves);

        let query_indices = vec![0usize, 5, 7, 16, 31];
        let proof = scheme.open(&query_indices, &aux);

        assert!(scheme.verify(&root, &query_indices, &leaves, &proof));
    }
}
