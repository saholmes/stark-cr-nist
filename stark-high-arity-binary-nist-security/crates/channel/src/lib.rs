use ark_goldilocks::Goldilocks as F;
use transcript::Transcript;
use merkle::{
    MerkleChannelCfg,
    MerkleTreeChannel,
    MerkleOpening,
};
use hash::HASH_BYTES;

// ────────────────────────────────────────────────────────────────────────
//  Fiat-Shamir channel trait
// ────────────────────────────────────────────────────────────────────────
//
//  Shared interface for prover and verifier channels.  All digest
//  inputs are &[u8] so the same channel works regardless of whether
//  the Merkle / commitment layer produces 32, 48, or 64-byte hashes.
// ────────────────────────────────────────────────────────────────────────

pub trait FiatShamirChannel {
    fn transcript_mut(&mut self) -> &mut Transcript;

    fn absorb_field(&mut self, label: &[u8], f: &F) {
        let tr = self.transcript_mut();
        tr.absorb_bytes(label);
        tr.absorb_field(*f);
    }

    /// Absorb a Merkle root or any hash digest into the transcript.
    /// Accepts any byte length — works with SHA3-256 (32 B),
    /// SHA3-384 (48 B), or SHA3-512 (64 B).
    fn absorb_root(&mut self, label: &[u8], root: &[u8]) {
        let tr = self.transcript_mut();
        tr.absorb_bytes(label);
        tr.absorb_bytes(root);
    }

    fn challenge(&mut self, label: &[u8]) -> F {
        self.transcript_mut().challenge(label)
    }
}

// ────────────────────────────────────────────────────────────────────────
//  Concrete channels
// ────────────────────────────────────────────────────────────────────────

pub struct ProverChannel {
    tr: Transcript,
}

pub struct VerifierChannel {
    tr: Transcript,
}

impl ProverChannel {
    pub fn new(tr: Transcript) -> Self {
        Self { tr }
    }
}

impl VerifierChannel {
    pub fn new(tr: Transcript) -> Self {
        Self { tr }
    }
}

impl FiatShamirChannel for ProverChannel {
    fn transcript_mut(&mut self) -> &mut Transcript {
        &mut self.tr
    }
}

impl FiatShamirChannel for VerifierChannel {
    fn transcript_mut(&mut self) -> &mut Transcript {
        &mut self.tr
    }
}

// ────────────────────────────────────────────────────────────────────────
//  Merkle channel (prover side)
// ────────────────────────────────────────────────────────────────────────

pub struct MerkleProver<'a> {
    chan: &'a mut ProverChannel,
    tree: MerkleTreeChannel,
}

impl<'a> MerkleProver<'a> {
    pub fn new(
        chan: &'a mut ProverChannel,
        cfg: MerkleChannelCfg,
        trace_hash: [u8; HASH_BYTES],
    ) -> Self {
        let tree = MerkleTreeChannel::new(cfg, trace_hash);
        Self { chan, tree }
    }

    /// Commit to a vector of field elements.
    ///
    /// Each leaf is `[value, 0, 0]` to match the current FRI
    /// base-layer convention.  A future refactor should accept
    /// a caller-provided leaf encoder instead.
    pub fn commit(&mut self, values: &[F]) -> [u8; HASH_BYTES] {
        for v in values {
            self.tree.push_leaf(&[*v, F::from(0u64), F::from(0u64)]);
        }
        let root = self.tree.finalize();
        self.chan.absorb_root(b"merkle/root", &root);
        root
    }

    pub fn open(&self, index: usize) -> MerkleOpening {
        self.tree.open(index)
    }

    pub fn challenge(&mut self, label: &[u8]) -> F {
        self.chan.challenge(label)
    }
}

// ────────────────────────────────────────────────────────────────────────
//  Merkle channel (verifier side)
// ────────────────────────────────────────────────────────────────────────

pub struct MerkleVerifier<'a> {
    chan: &'a mut VerifierChannel,
    cfg: MerkleChannelCfg,
    root: Option<[u8; HASH_BYTES]>,
}

impl<'a> MerkleVerifier<'a> {
    pub fn new(
        chan: &'a mut VerifierChannel,
        cfg: MerkleChannelCfg,
    ) -> Self {
        Self {
            chan,
            cfg,
            root: None,
        }
    }

    pub fn receive_root(&mut self, root: &[u8; HASH_BYTES]) {
        self.chan.absorb_root(b"merkle/root", root);
        self.root = Some(*root);
    }

    pub fn verify_opening(
        &self,
        opening: &MerkleOpening,
        trace_hash: &[u8; HASH_BYTES],
    ) -> bool {
        let root = match self.root {
            Some(r) => r,
            None => return false,
        };

        MerkleTreeChannel::verify_opening(
            &self.cfg,
            root,
            opening,
            trace_hash,
        )
    }

    pub fn challenge(&mut self, label: &[u8]) -> F {
        self.chan.challenge(label)
    }
}


// ────────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn merkle_channel_roundtrip() {
        let params = transcript::default_params();

        let p_tr = Transcript::new(b"CHAN-TEST", params.clone());
        let v_tr = Transcript::new(b"CHAN-TEST", params.clone());

        let mut pchan = ProverChannel::new(p_tr);
        let mut vchan = VerifierChannel::new(v_tr);

        let cfg = MerkleChannelCfg::new(vec![2, 2, 2, 2], 12345u64);
        let trace_hash = [0u8; HASH_BYTES];

        let mut rng = StdRng::seed_from_u64(42);
        let values: Vec<F> = (0..16).map(|_| F::rand(&mut rng)).collect();

        let mut prover = MerkleProver::new(&mut pchan, cfg.clone(), trace_hash);
        let root = prover.commit(&values);

        let mut verifier = MerkleVerifier::new(&mut vchan, cfg.clone());
        verifier.receive_root(&root);

        // Transcript consistency
        let alpha_p = prover.challenge(b"alpha");
        let alpha_v = verifier.challenge(b"alpha");
        assert_eq!(alpha_p, alpha_v);

        // Merkle opening
        let idx = 7usize;
        let opening = prover.open(idx);

        assert!(
            verifier.verify_opening(&opening, &trace_hash),
            "Merkle opening failed"
        );
    }

    #[test]
    fn channel_absorb_root_accepts_any_length() {
        let params = transcript::default_params();

        let tr1 = Transcript::new(b"LEN-TEST", params.clone());
        let tr2 = Transcript::new(b"LEN-TEST", params.clone());

        let mut ch1 = ProverChannel::new(tr1);
        let mut ch2 = ProverChannel::new(tr2);

        // 48-byte root (SHA3-384 sized) — passed as a slice via the trait
        let root_48 = [0xABu8; 48];
        ch1.absorb_root(b"merkle/root", &root_48);

        // 64-byte root (SHA3-512 sized) — passed as a slice via the trait
        let root_64 = [0xCDu8; 64];
        ch2.absorb_root(b"merkle/root", &root_64);

        // Both produce valid challenges (no panic)
        let _c1 = ch1.challenge(b"alpha");
        let _c2 = ch2.challenge(b"alpha");

        // Different root lengths produce different challenges
        assert_ne!(_c1, _c2);
    }

    #[test]
    fn trait_methods_match_across_roles() {
        let params = transcript::default_params();

        let p_tr = Transcript::new(b"TRAIT-TEST", params.clone());
        let v_tr = Transcript::new(b"TRAIT-TEST", params.clone());

        let mut pchan = ProverChannel::new(p_tr);
        let mut vchan = VerifierChannel::new(v_tr);

        let root = [0x42u8; HASH_BYTES];
        pchan.absorb_root(b"root", &root);
        vchan.absorb_root(b"root", &root);

        let f = F::from(999u64);
        pchan.absorb_field(b"val", &f);
        vchan.absorb_field(b"val", &f);

        // Same transcript state → same challenge
        assert_eq!(
            pchan.challenge(b"beta"),
            vchan.challenge(b"beta"),
        );
    }
}