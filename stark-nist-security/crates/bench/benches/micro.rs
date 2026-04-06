use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use ark_pallas::Fr as F;
use fft::fft_pow2;
use merkle::build_mary;
use poseidon::{params::generate_params_t17_x5, permute, T};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn bench_fft(c: &mut Criterion) {
    c.bench_function("fft_2048_forward", |b| {
        let mut rng = StdRng::seed_from_u64(9);
        let mut v: Vec<F> = (0..2048).map(|_| F::from(rng.gen::<u64>())).collect();
        b.iter(|| {
            let mut x = v.clone();
            fft_pow2(&mut x, false);
        })
    });

    c.bench_function("fft_2048_roundtrip", |b| {
        let mut rng = StdRng::seed_from_u64(10);
        let v: Vec<F> = (0..2048).map(|_| F::from(rng.gen::<u64>())).collect();
        b.iter_batched(
            || v.clone(),
            |mut x| {
                fft_pow2(&mut x, false);
                fft_pow2(&mut x, true);
            },
            BatchSize::SmallInput,
        )
    });
}

fn bench_poseidon(c: &mut Criterion) {
    c.bench_function("poseidon_t17_perm", |b| {
        let params = generate_params_t17_x5();
        let mut state = [F::from(0u64); T];
        for i in 0..T {
            state[i] = F::from((i as u64) + 1);
        }
        b.iter(|| {
            let mut s = state;
            permute(&mut s, &params);
        })
    });
}

fn bench_merkle(c: &mut Criterion) {
    c.bench_function("merkle_build_n2048_m16", |b| {
        let mut rng = StdRng::seed_from_u64(11);
        let leaves: Vec<F> = (0..2048).map(|_| F::from(rng.gen::<u64>())).collect();
        b.iter_batched(
            || leaves.clone(),
            |l| {
                let _tree = build_mary(&l, 16, [0u8; 32]);
            },
            BatchSize::SmallInput,
        )
    });

    c.bench_function("merkle_verify_path_n2048_m16", |b| {
        let mut rng = StdRng::seed_from_u64(12);
        let leaves: Vec<F> = (0..2048).map(|_| F::from(rng.gen::<u64>())).collect();
        let tree = build_mary(&leaves, 16, [1u8; 32]);
        let idx = 123usize;
        let leaf = leaves[idx];
        let path = merkle::open(&tree, idx);

        b.iter(|| {
            let ok = merkle::verify(&tree, leaf, idx, &path);
            assert!(ok);
        })
    });
}

criterion_group!(benches, bench_fft, bench_poseidon, bench_merkle);
criterion_main!(benches);
