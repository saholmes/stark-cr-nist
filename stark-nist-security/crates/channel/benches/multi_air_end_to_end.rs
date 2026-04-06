use rayon;
use ark_ff::UniformRand;
use ark_goldilocks::Goldilocks as F;
use criterion::{
    criterion_group, criterion_main, measurement::WallTime,
    BenchmarkGroup, Criterion, Throughput,
};
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};

// ✅ Existing imports
use deep_ali::trace_import::real_trace_inputs;

// ✅ NEW: multi-AIR trace generation
use deep_ali::air_workloads::{AirType, build_execution_trace};
use deep_ali::trace_import::trace_inputs_from_air;

// ✅ DEEP-ALI + MF-FRI APIs
use deep_ali::{
    deep_ali_merge_evals,
    deep_tower::Fp3,
    fri::{
        deep_fri_prove,
        deep_fri_proof_size_bytes,
        deep_fri_verify,
        FriDomain,
        DeepFriParams,
    },
};

// ═══════════════════════════════════════════════════════════════════
// Extension‑field selector.  Change this ONE line to switch:
//
//   Fp³ (cubic, default):   type Ext = CubeExt<GoldilocksCubeConfig>;
//   Fp  (base field only):  type Ext = F;
//
// To add Fp⁴ you would need to implement a quartic extension struct
// and implement TowerField for it — not yet available in this crate.
// ═══════════════════════════════════════════════════════════════════
//type Ext = CubeExt<GoldilocksCubeConfig>;

//use deep_ali::sextic_ext::SexticExt;
//type Ext = SexticExt;

use deep_ali::octic_ext::OcticExt;

type Ext = OcticExt;  // Fp⁶ = Fp³[u]/(u² − α)

// ═══════════════════════════════════════════════════════════════════
//  CSV record  (extended with AIR type)
// ═══════════════════════════════════════════════════════════════════

#[derive(Default, Clone)]
struct CsvRow {
    air_type: String,
    air_width: usize,
    air_constraints: usize,
    label: String,
    schedule: String,
    k: usize,
    proof_bytes: usize,
    prove_s: f64,
    verify_ms: f64,
    prove_elems_per_s: f64,
    delta_size_pct: f64,
    delta_prove_pct: f64,
    delta_verify_pct: f64,
    delta_throughput_pct: f64,
}

impl CsvRow {
    fn header() -> &'static str {
        "csv,air_type,air_w,air_constraints,label,k,schedule,proof_bytes,prove_s,verify_ms,prove_elems_per_s,delta_size_pct,delta_prove_pct,delta_verify_pct,delta_throughput_pct"
    }
    fn to_line(&self) -> String {
        format!(
            "csv,{},{},{},{},{},{},{},{:.6},{:.3},{:.6},{:.2},{:.2},{:.2},{:.2}\n",
            self.air_type,
            self.air_width,
            self.air_constraints,
            self.label,
            self.k,
            self.schedule,
            self.proof_bytes,
            self.prove_s,
            self.verify_ms,
            self.prove_elems_per_s,
            self.delta_size_pct,
            self.delta_prove_pct,
            self.delta_verify_pct,
            self.delta_throughput_pct,
        )
    }
    fn print_stdout(&self) {
        print!("{}", self.to_line());
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Schedule helpers  (unchanged)
// ═══════════════════════════════════════════════════════════════════

fn schedule_str(s: &[usize]) -> String {
    format!(
        "[{}]",
        s.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
    )
}

fn log2_pow2(x: usize) -> usize {
    assert!(x.is_power_of_two());
    x.trailing_zeros() as usize
}

fn k_min_for_schedule(schedule: &[usize]) -> usize {
    schedule.iter().map(|&m| log2_pow2(m)).sum()
}

fn divides_chain(n0: usize, schedule: &[usize]) -> bool {
    let mut n = n0;
    for &m in schedule {
        if n % m != 0 {
            return false;
        }
        n /= m;
    }
    true
}

fn ks_for_schedule(schedule: &[usize], k_lo: usize, k_hi: usize) -> Vec<usize> {
    let k_min = k_min_for_schedule(schedule);
    (k_lo.max(k_min)..=k_hi)
        .filter(|&k| divides_chain(1usize << k, schedule))
        .collect()
}

fn normalize_fri_schedule(n0: usize, mut schedule: Vec<usize>) -> Vec<usize> {
    let mut n = n0;
    for &m in &schedule {
        assert!(n % m == 0, "schedule does not divide domain");
        n /= m;
    }
    if n > 1 {
        assert!(n.is_power_of_two(), "final layer must be power of two");
        schedule.push(n);
    }
    schedule
}

// ═══════════════════════════════════════════════════════════════════
//  Main benchmark: AIR × schedule × domain size
// ═══════════════════════════════════════════════════════════════════

fn bench_e2e_mf_fri(c: &mut Criterion) {
    eprintln!(
        "[RAYON CHECK] current_num_threads = {}",
        rayon::current_num_threads()
    );

    let mut g: BenchmarkGroup<WallTime> = c.benchmark_group("e2e_mf_fri");
    g.warm_up_time(Duration::from_secs(5));
    g.measurement_time(Duration::from_secs(20));
    g.sample_size(10);

    let r: usize = 87;
    let seed_z: u64 = 0xDEEF_BAAD;
    let k_lo = 11usize;
    let k_hi = 25usize;

    let presets: &[(&str, &[usize])] = &[
        ("2power16", &[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]),
    ];

    // ✅ Which AIR workloads to benchmark
    let air_types = AirType::all();

    let file = File::create("benchmarkdata.csv").unwrap();
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}", CsvRow::header()).unwrap();
    println!("{}", CsvRow::header());

    let mut paper_baseline: HashMap<(String, usize), CsvRow> = HashMap::new();
    let mut rng_seed = 1337u64;

    // ✅ Outer loop: AIR workloads
    for &air in air_types {
        eprintln!(
            "\n╔══════════════════════════════════════════════════╗"
        );
        eprintln!(
            "║  AIR: {:20}  w={:<3}  constraints={:<3}  ║",
            air.label(),
            air.width(),
            air.num_constraints(),
        );
        eprintln!(
            "╚══════════════════════════════════════════════════╝"
        );

        for &(label, schedule) in presets {
            let ks = ks_for_schedule(schedule, k_lo, k_hi);

            eprintln!(
                "\n[PRESET] air={} label={} schedule={} configs={}",
                air.label(),
                label,
                schedule_str(schedule),
                ks.len()
            );

            for &k in &ks {
                let run_start = Instant::now();

                let n0 = 1usize << k;
                g.throughput(Throughput::Elements(n0 as u64));

                let normalized_schedule =
                    normalize_fri_schedule(n0, schedule.to_vec());

                eprintln!(
                    "[START] air={} label={} schedule={} k={} (n={})",
                    air.label(),
                    label,
                    schedule_str(&normalized_schedule),
                    k,
                    n0
                );

                rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
                let mut rng = StdRng::seed_from_u64(rng_seed);

                // ─────────────────────────────────────────────
                // ✅ AIR-PARAMETERISED TRACE GENERATION
                // ─────────────────────────────────────────────
                let trace = match air {
                    AirType::Fibonacci => {
                        // Existing path — guaranteed compatible
                        real_trace_inputs(n0, 4)
                    }
                    _ => {
                        // New path — generalised
                            let n_trace = n0 / 4;
                            let trace_cols = build_execution_trace(air, n_trace);
                            trace_inputs_from_air(trace_cols, n0, 4)

                    }
                };

                let a_eval = trace.a_eval;
                let s_eval = trace.s_eval;
                let e_eval = trace.e_eval;
                let t_eval = trace.t_eval;

                let domain0 = FriDomain::new_radix2(n0);

                // ✅ Sample DEEP-ALI challenge z ∈ Fp³
                let z_fp3 = Fp3 {
                    a0: F::rand(&mut rng),
                    a1: F::rand(&mut rng),
                    a2: F::rand(&mut rng),
                };

                // ✅ DEEP-ALI merge
                let (f0_ali, _z_used, _c_star) =
                    deep_ali_merge_evals(
                        &a_eval,
                        &s_eval,
                        &e_eval,
                        &t_eval,
                        domain0.omega,
                        z_fp3,
                    );

                let params = DeepFriParams {
                    schedule: normalized_schedule.clone(),
                    r,
                    seed_z,
                    coeff_commit_final: true,
                    d_final: 1,
                };

                // ──────────── Prove ────────────
                let t0 = Instant::now();
                let proof = deep_fri_prove::<Ext>(f0_ali.clone(), domain0, &params);
                let prove_s = t0.elapsed().as_secs_f64();

                // ──────────── Verify ────────────
                let t1 = Instant::now();
                assert!(deep_fri_verify::<Ext>(&params, &proof));
                let verify_ms = t1.elapsed().as_secs_f64() * 1e3;

                let proof_bytes = deep_fri_proof_size_bytes::<Ext>(&proof);

                let mut row = CsvRow {
                    air_type: air.label().to_string(),
                    air_width: air.width(),
                    air_constraints: air.num_constraints(),
                    label: label.to_string(),
                    schedule: schedule_str(&normalized_schedule),
                    k,
                    proof_bytes,
                    prove_s,
                    verify_ms,
                    prove_elems_per_s: n0 as f64 / prove_s,
                    delta_size_pct: 0.0,
                    delta_prove_pct: 0.0,
                    delta_verify_pct: 0.0,
                    delta_throughput_pct: 0.0,
                };

                // ✅ Baseline keyed on (air_label, k)
                let baseline_key = (air.label().to_string(), k);
                if label == "paper" {
                    paper_baseline.insert(baseline_key, row.clone());
                } else if let Some(base) =
                    paper_baseline.get(&(air.label().to_string(), k))
                {
                    row.delta_size_pct =
                        100.0 * (row.proof_bytes as f64 - base.proof_bytes as f64)
                            / base.proof_bytes as f64;
                    row.delta_prove_pct =
                        100.0 * (row.prove_s - base.prove_s) / base.prove_s;
                    row.delta_verify_pct =
                        100.0 * (row.verify_ms - base.verify_ms) / base.verify_ms;
                    row.delta_throughput_pct =
                        100.0
                            * (row.prove_elems_per_s - base.prove_elems_per_s)
                            / base.prove_elems_per_s;
                }

                row.print_stdout();
                std::io::stdout().flush().unwrap();
                writer.write_all(row.to_line().as_bytes()).unwrap();
                writer.flush().unwrap();

                eprintln!(
                    "[DONE ] air={} label={} k={} prove={:.2}s verify={:.2}ms proof={} bytes  [ext=Fp{}]",
                    air.label(),
                    label,
                    k,
                    prove_s,
                    verify_ms,
                    proof_bytes,
                    std::mem::size_of::<Ext>() / std::mem::size_of::<F>(),
                );

                let run_secs = run_start.elapsed().as_secs_f64();
                eprintln!(
                    "[ETA ] air={} label={} schedule={} elapsed={:.2}s",
                    air.label(),
                    label,
                    schedule_str(&normalized_schedule),
                    run_secs
                );
            }
        }
    }

    g.finish();
}

criterion_group!(e2e, bench_e2e_mf_fri);
criterion_main!(e2e);
