use ark_ff::UniformRand;
use ark_pallas::Fr as F;
use criterion::{
    criterion_group, criterion_main, measurement::WallTime, BatchSize, BenchmarkGroup,
    BenchmarkId, Criterion, Throughput,
};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Duration;

// Existing APIs from your crate(s)
use channel::{build_vk_plain, prove_plain, verify_plain};

use deep_ali::fri::{
    deep_fri_prove, deep_fri_proof_size_bytes, deep_fri_verify, AliA, AliE, AliS, AliT,
    DeepAliRealBuilder, DeepFriParams, DeepFriProof,
};

// ---------------------
// CSV record
// ---------------------

#[derive(Default, Clone)]
struct CsvRow {
    label: String,
    schedule: String,
    k: usize,
    r: usize,
    proof_bytes: usize,
    prove_s: f64,           // single timed prove (seconds)
    verify_ms: f64,         // single timed verify (milliseconds)
    prove_elems_per_s: f64, // n0 / prove_s
    // deltas vs baseline
    delta_size_pct: f64,
    delta_prove_pct: f64,
    delta_verify_pct: f64,
    delta_throughput_pct: f64,
}

impl CsvRow {
    fn header() -> &'static str {
        "csv,label,k,r,schedule,proof_bytes,prove_s,verify_ms,prove_elems_per_s,delta_size_pct_vs_baseline,delta_prove_pct_vs_baseline,delta_verify_pct_vs_baseline,delta_throughput_pct_vs_baseline"
    }
    fn to_line(&self) -> String {
        format!(
            "csv,{},{},{},{},{},{:.6},{:.3},{:.6},{:.2},{:.2},{:.2},{:.2}\n",
            self.label,
            self.k,
            self.r,
            self.schedule,
            self.proof_bytes,
            self.prove_s,
            self.verify_ms,
            self.prove_elems_per_s,
            self.delta_size_pct,
            self.delta_prove_pct,
            self.delta_verify_pct,
            self.delta_throughput_pct
        )
    }
    fn print_stdout(&self) {
        print!(
            "csv,{},{},{},{},{},{:.6},{:.3},{:.6},{:.2},{:.2},{:.2},{:.2}\n",
            self.label,
            self.k,
            self.r,
            self.schedule,
            self.proof_bytes,
            self.prove_s,
            self.verify_ms,
            self.prove_elems_per_s,
            self.delta_size_pct,
            self.delta_prove_pct,
            self.delta_verify_pct,
            self.delta_throughput_pct
        );
    }
}

// ---------------------
// Schedule helpers
// ---------------------

fn schedule_str(s: &[usize]) -> String {
    format!(
        "[{}]",
        s.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

// Expand a short "motif" (e.g., [2]) into the minimal full schedule that reduces 2^k -> 1.
// Each motif entry must be a power of two ≥ 2 and must divide the current size.
fn expand_schedule_to_k(motif: &[usize], k: usize) -> Vec<usize> {
    assert!(!motif.is_empty(), "schedule motif must be non-empty");
    for &m in motif {
        assert!(m.is_power_of_two() && m >= 2, "arity must be power-of-two ≥ 2, got {}", m);
    }
    let mut expanded = Vec::new();
    let mut size = 1usize << k;
    let mut i = 0;
    while size > 1 {
        let m = motif[i % motif.len()];
        assert!(
            size % m == 0,
            "arity {} must divide current size {} (k = {}, motif = {:?})",
            m, size, k, motif
        );
        expanded.push(m);
        size /= m;
        i += 1;
    }
    expanded
}

// ---------------------
// Plain benches (unchanged behavior)
// ---------------------

fn bench_e2e_plain(c: &mut Criterion) {
    let mut g = c.benchmark_group("e2e_plain");
    g.warm_up_time(Duration::from_secs(5));
    g.measurement_time(Duration::from_secs(20));
    g.sample_size(10);

    for &k in &[12usize, 14, 16] {
        let n = 1usize << k;
        let ds = F::from(2025u64);

        let mut rng = StdRng::seed_from_u64(7);
        let witness: Vec<F> = (0..n).map(|_| F::rand(&mut rng)).collect();
        let vk = build_vk_plain(k, ds);
        let pre_proof = prove_plain(&vk, &witness);

        let vk_size = bincode::serialize(&vk).map(|b| b.len()).unwrap_or(0);
        let proof_size = bincode::serialize(&pre_proof).map(|b| b.len()).unwrap_or(0);
        eprintln!("plain k={} vk={}B proof={}B", k, vk_size, proof_size);

        g.throughput(Throughput::Elements(n as u64));

        // Prove
        g.bench_with_input(BenchmarkId::new("prove", k), &k, |b, &_k| {
            b.iter_batched(
                || (),
                |_| {
                    let proof = prove_plain(&vk, &witness);
                    criterion::black_box(proof);
                },
                BatchSize::SmallInput,
            )
        });

        // Verify
        g.bench_with_input(BenchmarkId::new("verify", k), &k, |b, &_k| {
            b.iter(|| {
                let ok = verify_plain(&vk, &pre_proof);
                assert!(ok);
            })
        });
    }
    g.finish();
}

// ---------------------
// MF-FRI benches + CSV + r sweep
// ---------------------

fn bench_e2e_mf_fri(c: &mut Criterion) {
    let mut g: BenchmarkGroup<WallTime> = c.benchmark_group("e2e_mf_fri");

    // Tuning for long benches
    g.warm_up_time(Duration::from_secs(5));
    g.measurement_time(Duration::from_secs(20));
    g.sample_size(10);

    // r sweep for soundness comparison
    let r_values: &[usize] = &[96, 112, 128];
    let seed_z: u64 = 0xDEEF_BAAD;

    // k window
    let k_lo = 11usize;
    let k_hi = 20usize;

    // Presets: interpret as motifs to be expanded per k.
    // "binary" == minimal number of layers for classic binary FRI (k layers).
    let presets: &[(&str, &[usize])] = &[
        ("binary", &[2]),
        // Add other motifs for comparison if desired:
        // ("arity-4", &[4]),
        // ("mixed-8-2", &[8, 2]),
    ];

    // Deterministic input generation
    let mut rng_seed = 1337u64;

    // Baseline is the first preset at r = 128 for each k
    let baseline_label = presets[0].0;
    let baseline_r: usize = 128;
    let mut baseline: HashMap<usize, CsvRow> = HashMap::new();

    // Prepare CSV file
    let file = File::create("binary_benchmark_data.csv")
        .expect("failed to create binary_benchmark_data.csv for writing");
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}", CsvRow::header()).expect("failed to write CSV header");
    writer.flush().ok();

    // Also print header to stdout
    println!("{}", CsvRow::header());

    for &(label, motif) in presets {
        for k in k_lo..=k_hi {
            let schedule = expand_schedule_to_k(motif, k);
            let n0 = 1usize << k;

            g.throughput(Throughput::Elements(n0 as u64));

            // Inputs (reuse across r values to isolate the effect of r)
            rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let mut rng = StdRng::seed_from_u64(rng_seed);
            let a: AliA = (0..n0).map(|_| F::rand(&mut rng)).collect();
            let s: AliS = (0..n0).map(|_| F::rand(&mut rng)).collect();
            let e: AliE = (0..n0).map(|_| F::rand(&mut rng)).collect();
            let t: AliT = (0..n0).map(|_| F::rand(&mut rng)).collect();

            for &r in r_values {
                let params = DeepFriParams {
                    schedule: schedule.clone(),
                    r,
                    seed_z,
                };
                let builder = DeepAliRealBuilder::default();

                eprintln!(
                    "mf-fri setup: label={} k={} (n0={}) schedule_len={} first5={:?} r={}",
                    label,
                    k,
                    n0,
                    schedule.len(),
                    &schedule.iter().cloned().take(5).collect::<Vec<_>>(),
                    r
                );

                // Precompute proof for verify bench and size
                eprintln!("mf-fri precompute proof…");
                let pre_proof: DeepFriProof =
                    deep_fri_prove(&builder, &a, &s, &e, &t, n0, &params);
                assert!(
                    deep_fri_verify(&params, &pre_proof),
                    "precomputed proof failed verification"
                );
                let proof_size_bytes = deep_fri_proof_size_bytes(&pre_proof);
                eprintln!(
                    "mf-fri label={} k={} r={} proof≈{}B",
                    label, k, r, proof_size_bytes
                );

                // Criterion bench: prove
                let prove_id = BenchmarkId::new(format!("prove-{}-r{}", label, r), k);
                g.bench_with_input(prove_id, &k, |b, &_k| {
                    b.iter_batched(
                        || (),
                        |_| {
                            let proof =
                                deep_fri_prove(&builder, &a, &s, &e, &t, n0, &params);
                            criterion::black_box(proof);
                        },
                        BatchSize::SmallInput,
                    )
                });

                // Criterion bench: verify
                let verify_id = BenchmarkId::new(format!("verify-{}-r{}", label, r), k);
                g.bench_with_input(verify_id, &k, |b, &_k| {
                    b.iter(|| {
                        let ok = deep_fri_verify(&params, &pre_proof);
                        assert!(ok);
                    })
                });

                // Single-shot timings to populate CSV
                let t0 = std::time::Instant::now();
                let _tmp_proof = deep_fri_prove(&builder, &a, &s, &e, &t, n0, &params);
                let prove_s = t0.elapsed().as_secs_f64();

                let t1 = std::time::Instant::now();
                let ok = deep_fri_verify(&params, &pre_proof);
                assert!(ok);
                let verify_ms = t1.elapsed().as_secs_f64() * 1e3;

                let prove_elems_per_s = (n0 as f64) / prove_s;

                let mut row = CsvRow {
                    label: format!("{}-r{}", label, r),
                    schedule: schedule_str(&schedule),
                    k,
                    r,
                    proof_bytes: proof_size_bytes,
                    prove_s,
                    verify_ms,
                    prove_elems_per_s,
                    delta_size_pct: f64::NAN,
                    delta_prove_pct: f64::NAN,
                    delta_verify_pct: f64::NAN,
                    delta_throughput_pct: f64::NAN,
                };

                // Baseline: first preset at r = baseline_r for each k
                if label == baseline_label && r == baseline_r {
                    baseline.insert(
                        k,
                        CsvRow {
                            label: row.label.clone(),
                            schedule: row.schedule.clone(),
                            k: row.k,
                            r: row.r,
                            proof_bytes: row.proof_bytes,
                            prove_s: row.prove_s,
                            verify_ms: row.verify_ms,
                            prove_elems_per_s: row.prove_elems_per_s,
                            delta_size_pct: 0.0,
                            delta_prove_pct: 0.0,
                            delta_verify_pct: 0.0,
                            delta_throughput_pct: 0.0,
                        },
                    );
                    row.delta_size_pct = 0.0;
                    row.delta_prove_pct = 0.0;
                    row.delta_verify_pct = 0.0;
                    row.delta_throughput_pct = 0.0;
                } else if let Some(base) = baseline.get(&k) {
                    row.delta_size_pct =
                        100.0 * (row.proof_bytes as f64 - base.proof_bytes as f64)
                            / (base.proof_bytes as f64);
                    row.delta_prove_pct =
                        100.0 * (row.prove_s - base.prove_s) / base.prove_s;
                    row.delta_verify_pct =
                        100.0 * (row.verify_ms - base.verify_ms) / base.verify_ms;
                    row.delta_throughput_pct = 100.0
                        * (row.prove_elems_per_s - base.prove_elems_per_s)
                        / base.prove_elems_per_s;
                } else {
                    eprintln!(
                        "warn: missing baseline for k={}, deltas set to NaN",
                        k
                    );
                }

                // Emit to stdout and CSV
                row.print_stdout();
                let line = row.to_line();
                writer
                    .write_all(line.as_bytes())
                    .expect("failed to write CSV row");
                writer.flush().ok();
            }
        }
    }

    g.finish();
}

criterion_group!(e2e, bench_e2e_plain, bench_e2e_mf_fri);
criterion_main!(e2e);