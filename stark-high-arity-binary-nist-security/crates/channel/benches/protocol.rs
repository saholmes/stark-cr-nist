use ark_ff::UniformRand;
use ark_pallas::Fr as F;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::{rngs::StdRng, SeedableRng};

use channel::{
    Mle, MleProver, MleVerifier, MerkleChannelCfg, MerkleProver, MerkleVerifier,
    SumCheckProver, SumCheckVerifier, SumCheckMFConfig, SumCheckMFProver, SumCheckMFVerifier,
};
use transcript::Transcript;

// -------- Size helpers --------

fn field_len_bytes() -> usize { 32 } // proxy for ark-pallas::Fr canonical encoding

fn poseidon_params_size_bytes(_p: &poseidon::PoseidonParams) -> usize {
    // Matches paper setup (t=17, RF=8, RP=64). Adjust if your struct differs.
    let t = 17usize;
    let rf = 8usize;
    let rp = 64usize;
    let mds = t * t;
    let rc = rf * t + rp;
    (mds + rc) * field_len_bytes()
}

#[derive(Clone)]
struct VK {
    arity: u8,
    tree_label: u64,
    params: poseidon::PoseidonParams,
}
#[derive(Clone)]
struct PK {
    arity: u8,
    tree_label: u64,
    params: poseidon::PoseidonParams,
}

fn vk_size_bytes(vk: &VK) -> usize {
    1 + 8 + poseidon_params_size_bytes(&vk.params)
}
fn pk_size_bytes(pk: &PK) -> usize {
    1 + 8 + poseidon_params_size_bytes(&pk.params)
}

// -------- Proof size helper (Merkle) --------

fn proof_size_bytes(proof: &commitment::MerkleProof) -> usize {
    let mut total = 0usize;
    total += 1; // arity
    total += 8; // levels len (group_sizes)
    for lvl in &proof.group_sizes {
        total += 8;
        total += lvl.len();
    }
    total += 8; // siblings levels len
    for lvl in &proof.siblings {
        total += 8;
        for _s in lvl {
            total += field_len_bytes();
        }
    }
    total
}

// -------- Benches --------

fn bench_mle_commit_open(c: &mut Criterion) {
    let mut group = c.benchmark_group("protocol_mle_merkle");

    for &(k, q) in &[(12usize, 64usize), (14, 64), (16, 128)] {
        let n = 1usize << k;
        group.bench_with_input(BenchmarkId::new("commit_open", k), &k, |b, &_k| {
            b.iter_batched(
                || {
                    // Generate inputs only in setup
                    let params = transcript::default_params();
                    let tree_label = 12345u64;

                    // PK/VK
                    let vk = VK { arity: 16, tree_label, params: params.clone() };
                    let pk = PK { arity: 16, tree_label, params: params.clone() };
                    let _ = criterion::black_box(vk_size_bytes(&vk));
                    let _ = criterion::black_box(pk_size_bytes(&pk));

                    let cfg = MerkleChannelCfg::with_default_params(F::from(tree_label));

                    let mut rng = StdRng::seed_from_u64(7);
                    let table: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();

                    let step = (n / q.max(1)).max(1);
                    let indices: Vec<usize> = (0..q).map(|i| (i * step) % n).collect();

                    (params, cfg, table, indices)
                },
                |(params, cfg, table, indices)| {
                    // Do the full protocol per iteration, owning all locals
                    let p_tr = Transcript::new(b"PROTO-MLE", params.clone());
                    let v_tr = Transcript::new(b"PROTO-MLE", params.clone());
                    let mut pchan = channel::ProverChannel::new(p_tr);
                    let mut vchan = channel::VerifierChannel::new(v_tr);

                    let mut prover = MerkleProver::new(&mut pchan, cfg.clone());
                    let root = prover.commit_vector(&table);

                    let mut verifier = MerkleVerifier::new(&mut vchan, cfg.clone());
                    verifier.receive_root(&root);

                    let (values, proof) = prover.open_indices(&indices, &table);
                    let ok = verifier.verify_openings(&indices, &values, &proof);
                    assert!(ok);

                    let size = proof_size_bytes(&proof);
                    let _ = criterion::black_box(size);
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

fn bench_sumcheck_plain(c: &mut Criterion) {
    let mut group = c.benchmark_group("sumcheck_plain");

    for &k in &[12usize, 14, 16] {
        let n = 1usize << k;
        group.bench_with_input(BenchmarkId::new("prove_and_verify", k), &k, |b, &_k| {
            b.iter_batched(
                || {
                    // Inputs
                    let params = transcript::default_params();
                    let tree_label = 222u64;

                    // PK/VK
                    let vk = VK { arity: 16, tree_label, params: params.clone() };
                    let pk = PK { arity: 16, tree_label, params: params.clone() };
                    let _ = criterion::black_box(vk_size_bytes(&vk));
                    let _ = criterion::black_box(pk_size_bytes(&pk));

                    let cfg = MerkleChannelCfg::with_default_params(F::from(tree_label));

                    let mut rng = StdRng::seed_from_u64(42);
                    let table: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
                    let mle = Mle::new(table.clone());

                    (params, cfg, table, mle, k)
                },
                |(params, cfg, table, mle, k)| {
                    // Full protocol per iteration
                    let p_tr = Transcript::new(b"SUMCHECK/PLAIN", params.clone());
                    let v_tr = Transcript::new(b"SUMCHECK/PLAIN", params.clone());
                    let mut pchan = channel::ProverChannel::new(p_tr);
                    let mut vchan = channel::VerifierChannel::new(v_tr);

                    let mut mp = MerkleProver::new(&mut pchan, cfg.clone());
                    let root = mp.commit_vector(&table);
                    let mut mv = MerkleVerifier::new(&mut vchan, cfg.clone());
                    mv.receive_root(&root);

                    let mut sp = SumCheckProver::new(MleProver::new(mp, mle.clone()));
                    let mut sv = SumCheckVerifier::new(MleVerifier::new(mv, k));

                    let s = sp.send_claim();
                    sv.recv_claim(&s);

                    let mut running = s;
                    for i in 0..sp.mle_prover_mut().mle().num_vars() {
                        let (c0, c1, r_i) = sp.round(i, b"sumcheck/r");
                        let (r_i_v, s_next) = sv.round(i, running, c0, c1, b"sumcheck/r");
                        assert_eq!(r_i, r_i_v);
                        running = s_next;
                    }
                    let eval = sp.finalize_and_bind_eval();
                    sv.finalize_and_check(eval, running);
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

fn bench_sumcheck_mf(c: &mut Criterion) {
    let mut group = c.benchmark_group("sumcheck_mf");

    for &k in &[12usize, 14] {
        let n = 1usize << k;
        let qpr = 3usize;

        group.bench_with_input(BenchmarkId::new("prove_and_verify", k), &k, |b, &_k| {
            b.iter_batched(
                || {
                    let params = transcript::default_params();
                    let tree_label = 6060u64;

                    // PK/VK
                    let vk = VK { arity: 16, tree_label, params: params.clone() };
                    let pk = PK { arity: 16, tree_label, params: params.clone() };
                    let _ = criterion::black_box(vk_size_bytes(&vk));
                    let _ = criterion::black_box(pk_size_bytes(&pk));

                    let cfg = MerkleChannelCfg::with_default_params(F::from(tree_label));

                    let mut rng = StdRng::seed_from_u64(1337);
                    let table: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
                    let mle = Mle::new(table.clone());
                    let mf_cfg = channel::SumCheckMFConfig { queries_per_round: qpr };

                    (params, cfg, mle, mf_cfg)
                },
                |(params, cfg, mle, mf_cfg)| {
                    // Full protocol per iteration
                    let p_tr = Transcript::new(b"SUMCHECK/MF", params.clone());
                    let v_tr = Transcript::new(b"SUMCHECK/MF", params.clone());
                    let mut pchan = channel::ProverChannel::new(p_tr);
                    let mut vchan = channel::VerifierChannel::new(v_tr);

                    let mut sp = SumCheckMFProver::new(mf_cfg, cfg.clone(), &mut pchan, &mle);
                    let init_root = sp.current_root();

                    let mut sv = SumCheckMFVerifier::new(mf_cfg, cfg.clone(), &mut vchan, init_root, mle.num_vars());
                    sv.receive_initial_root(&init_root);

                    let s = sp.send_claim();
                    sv.recv_claim(&s);

                    let mut s_running = s;
                    let mut prev_root = init_root;
                    let mut total_proof_bytes = 0usize;

                    for i in 0..sv.rounds() {
                        let (c0, c1, r_i, next_root, openings) = sp.round(i);
                        sv.start_round(i, s_running, c0, c1);

                        let r_i_v = sv.derive_round_challenge(i);
                        assert_eq!(r_i, r_i_v);

                        sv.recv_next_root(next_root);

                        total_proof_bytes += proof_size_bytes(&openings.cur_proof);
                        total_proof_bytes += proof_size_bytes(&openings.next_proof);

                        assert!(sv.verify_fold_openings(
                            &openings.cur_indices,
                            &openings.cur_values,
                            &openings.cur_proof,
                            &openings.next_indices,
                            &openings.next_values,
                            &openings.next_proof,
                            r_i,
                            prev_root,
                            next_root
                        ));

                        s_running = sv.compute_s_next(c0, c1, r_i_v);
                        prev_root = next_root;
                    }

                    let final_eval = sp.finalize_eval();
                    sv.finalize_and_check(final_eval, s_running);

                    let _ = criterion::black_box(total_proof_bytes);
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

criterion_group!(protocol_benches, bench_mle_commit_open, bench_sumcheck_plain, bench_sumcheck_mf);
criterion_main!(protocol_benches);