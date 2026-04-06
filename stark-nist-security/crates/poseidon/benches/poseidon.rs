use ark_pallas::Fr as F;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use poseidon::{params::generate_params_t17_x5, dynamic_from_static_t17};

fn bench_permute_t17(c: &mut Criterion) {
    let params = generate_params_t17_x5(b"POSEIDON-T17-X5");
    c.bench_function("poseidon_perm_t17", |b| {
        b.iter_batched(
            || ([F::from(0u64); 17], params.clone()),
            |(mut state, p)| { poseidon::permute(&mut state, &p); },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(poseidon_benches, bench_permute_t17);
criterion_main!(poseidon_benches);
