use ark_ff::UniformRand;
use ark_pallas::Fr as F;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::{rngs::StdRng, SeedableRng};
use merkle::{MerkleChannelCfg, MerkleTree, MerkleProof, verify_many_ds};
use poseidon::{params::generate_params_t17_x5, dynamic_from_static_t17};

// Serialize a proof deterministically to count bytes, without requiring serde in your lib.
fn proof_size_bytes(proof: &MerkleProof) -> usize {
    // Layout:
    // - arity: u8
    // - group_sizes: for each level: len(u64) + bytes of sizes(u8 each)
    // - siblings: for each level: len(u64) + len * Fr (32 bytes if compressed via to_bytes_le padded to 32)
    let mut total = 0usize;
    total += 1; // arity
    total += 8; // number of levels for group_sizes (implicit via vec length)
    for lvl in &proof.group_sizes {
        total += 8; // len
        total += lvl.len(); // each size as 1 byte
    }
    total += 8; // number of levels for siblings
    for lvl in &proof.siblings {
        total += 8; // len
        // Field size accounting: using canonical little-endian; pad to 32 bytes.
        for _s in lvl {
            total += 32;
        }
    }
    total
}

fn bench_merkle_build_open_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_m16_t17");
    let rng_seed = 42u64;

    let params_static = generate_params_t17_x5(b"POSEIDON-T17-X5");
    let params_dyn = dynamic_from_static_t17(&params_static);

    for &n in &[1<<12, 1<<14, 1<<16] {
        // Build
        group.bench_with_input(BenchmarkId::new("build_tree", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut rng = StdRng::seed_from_u64(rng_seed);
                    (0..n).map(|_| F::rand(&mut rng)).collect::<Vec<F>>()
                },
                |leaves| {
                    let cfg = MerkleChannelCfg { arity: 16, tree_label: 777, params: params_dyn.clone() };
                    let _tree = MerkleTree::new(leaves, cfg);
                },
                BatchSize::LargeInput,
            );
        });

        // Open + verify for a fixed query size
        let q = 64usize.min(n.max(1));
        group.bench_with_input(BenchmarkId::new("open_and_verify_q", n), &n, |b, &n| {
            b.iter_batched(
                || {
                    let mut rng = StdRng::seed_from_u64(rng_seed + 1);
                    let leaves: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
                    let cfg = MerkleChannelCfg { arity: 16, tree_label: 888, params: params_dyn.clone() };
                    let tree = MerkleTree::new(leaves.clone(), cfg.clone());
                    let root = tree.root();
                    let indices: Vec<usize> = (0..q).map(|i| (i * (n / q.max(1)).max(1)) % n).collect();
                    let values: Vec<F> = indices.iter().map(|&i| leaves[i]).collect();
                    (tree, root, indices, values, cfg)
                },
                |(tree, root, indices, values, cfg)| {
                    let proof = tree.open_many(&indices);
                    let size = proof_size_bytes(&proof);
                    // record size via black_box (so it’s "used")
                    let _ = criterion::black_box(size);
                    let ok = verify_many_ds(&root, &indices, &values, &proof, cfg.tree_label, cfg.params.clone());
                    assert!(ok);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(merkle_benches, bench_merkle_build_open_verify);
criterion_main!(merkle_benches);
