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

use deep_ali::trace_import::real_trace_inputs;
use deep_ali::air_workloads::{AirType, build_execution_trace};
use deep_ali::trace_import::trace_inputs_from_air;

use deep_ali::{
    deep_ali_merge_evals,
    deep_tower::Fp3,
    fri::{
        deep_fri_prove,
        deep_fri_proof_size_bytes,
        deep_fri_verify,
        FriDomain,
        DeepFriParams,
        // ✅ NEW: soundness imports
        SoundnessCalculator,
        SoundnessReport,
        ProximityBound,
        HashVariant,
    },
};

use deep_ali::sextic_ext::SexticExt;
use deep_ali::tower_field::TowerField;

type Ext = SexticExt;

// ═══════════════════════════════════════════════════════════════════
//  Soundness configuration  —  replaces the old hardcoded `r = 52`
// ═══════════════════════════════════════════════════════════════════

const TARGET_SECURITY_BITS: usize = 128;
const BLOWUP: usize = 4;

/// Which proximity-gap analyses to benchmark.
/// Each entry produces a separate row in the CSV so you can compare.
fn bounds_to_test() -> Vec<(&'static str, bool, ProximityBound)> {
    vec![
        ("stir-johnson",  true,  ProximityBound::Johnson),
        ("stir-capacity", true,  ProximityBound::StirCapacity { gamma_log2: -30.0 }),
        // Uncomment to also benchmark classic FRI for comparison:
        // ("classic-johnson", false, ProximityBound::Johnson),
    ]
}

// ═══════════════════════════════════════════════════════════════════
//  CSV record  (extended with soundness columns)
// ═══════════════════════════════════════════════════════════════════

#[derive(Default, Clone)]
struct CsvRow {
    // ── AIR info ──
    air_type: String,
    air_width: usize,
    air_constraints: usize,
    // ── Schedule info ──
    label: String,
    schedule: String,
    k: usize,
    // ── Performance ──
    proof_bytes: usize,
    prove_s: f64,
    verify_ms: f64,
    prove_elems_per_s: f64,
    // ── Delta vs baseline ──
    delta_size_pct: f64,
    delta_prove_pct: f64,
    delta_verify_pct: f64,
    delta_throughput_pct: f64,
    // ── Soundness (NEW) ──
    mode_label: String,
    soundness_suffix: String,
}

impl CsvRow {
    fn header() -> String {
        format!(
            "csv,air_type,air_w,air_constraints,\
             label,k,schedule,\
             proof_bytes,prove_s,verify_ms,prove_elems_per_s,\
             delta_size_pct,delta_prove_pct,delta_verify_pct,delta_throughput_pct,\
             mode{}",
            SoundnessReport::csv_header_suffix(),
        )
    }

    fn to_line(&self) -> String {
        format!(
            "csv,{},{},{},{},{},{},{},{:.6},{:.3},{:.6},{:.2},{:.2},{:.2},{:.2},{}{}\n",
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
            self.mode_label,
            self.soundness_suffix,
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
//  Main benchmark
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

    let seed_z: u64 = 0xDEEF_BAAD;
    let k_lo = 11usize;
    let k_hi = 25usize;

    let presets: &[(&str, &[usize])] = &[
        ("2power16",       &[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]),
        ("16by16by16",     &[16,16,16]),
        ("32by16by8",      &[32,16,8]),
        ("64by32by16",     &[64,32,16]),
        ("16by16by8",      &[16,16,8]),
    ];

    let air_types = AirType::all();
    let modes = bounds_to_test();

    // ✅ Detect compiled hash → drives both soundness calc AND Merkle trees
    let hash_variant = HashVariant::from_compiled();

    // ✅ Derive extension-field size from Ext type
    let field_ext_bits = 64.0 * Ext::DEGREE as f64;

    // ✅ Print comparison table for quick reference
    eprintln!("\n── Soundness comparison (n0=2^20, blowup={}) ──", BLOWUP);
    SoundnessCalculator::print_comparison(
        presets,
        1 << 20,
        BLOWUP,
        TARGET_SECURITY_BITS,
        field_ext_bits,
        hash_variant,
        true,
    );

    let file = File::create("benchmarkdata.csv").unwrap();
    let mut writer = BufWriter::new(file);
    let hdr = CsvRow::header();
    writeln!(writer, "{}", hdr).unwrap();
    println!("{}", hdr);

    let mut paper_baseline: HashMap<(String, String, usize), CsvRow> = HashMap::new();
    let mut rng_seed = 1337u64;

    // ✅ Outer loop: proximity-bound modes
    for &(mode_label, use_stir, ref bound) in &modes {

        let calc = SoundnessCalculator::new(
            TARGET_SECURITY_BITS,
            field_ext_bits,
            *bound,
            hash_variant,
        );

        eprintln!(
            "\n╔══════════════════════════════════════════════════════════╗"
        );
        eprintln!(
            "║  MODE: {:<20}  hash={}  eff_target={}  ║",
            mode_label, hash_variant.label(), calc.effective_target(),
        );
        eprintln!(
            "╚══════════════════════════════════════════════════════════╝"
        );

        for &air in air_types {
            eprintln!(
                "\n  AIR: {:20}  w={:<3}  constraints={:<3}",
                air.label(), air.width(), air.num_constraints(),
            );

            for &(label, schedule) in presets {
                let ks = ks_for_schedule(schedule, k_lo, k_hi);

                for &k in &ks {
                    let run_start = Instant::now();
                    let n0 = 1usize << k;
                    g.throughput(Throughput::Elements(n0 as u64));

                    let normalized_schedule =
                        normalize_fri_schedule(n0, schedule.to_vec());

                    // ✅ Compute s0 / r from soundness
                    let queries = calc.min_queries(
                        &normalized_schedule, n0, BLOWUP, use_stir,
                    );
                    let report = calc.report(
                        &normalized_schedule, n0, BLOWUP, queries, use_stir,
                    );

                    eprintln!(
                        "  [START] air={} mode={} label={} k={} \
                         queries={} achieved={:.1} bits",
                        air.label(), mode_label, label, k,
                        queries, report.achieved_bits,
                    );

                    rng_seed = rng_seed
                        .wrapping_mul(1103515245)
                        .wrapping_add(12345);
                    let mut rng = StdRng::seed_from_u64(rng_seed);

                    // ── Trace generation ──
                    let trace = match air {
                        AirType::Fibonacci => real_trace_inputs(n0, BLOWUP),
                        _ => {
                            let n_trace = n0 / BLOWUP;
                            let trace_cols = build_execution_trace(air, n_trace);
                            trace_inputs_from_air(trace_cols, n0, BLOWUP)
                        }
                    };

                    let a_eval = trace.a_eval;
                    let s_eval = trace.s_eval;
                    let e_eval = trace.e_eval;
                    let t_eval = trace.t_eval;

                    let domain0 = FriDomain::new_radix2(n0);

                    let z_fp3 = Fp3 {
                        a0: F::rand(&mut rng),
                        a1: F::rand(&mut rng),
                        a2: F::rand(&mut rng),
                    };

                    let (f0_ali, _z_used, _c_star) = deep_ali_merge_evals(
                        &a_eval, &s_eval, &e_eval, &t_eval,
                        domain0.omega, z_fp3,
                    );

                    // ✅ Build params with computed query count
                    let params = DeepFriParams {
                        schedule: normalized_schedule.clone(),
                        r: if use_stir { 0 } else { queries },
                        seed_z,
                        coeff_commit_final: true,
                        d_final: 1,
                        stir: use_stir,
                        s0: if use_stir { queries } else { 0 },
                    };

                    // ── Prove ──
                    let t0 = Instant::now();
                    let proof = deep_fri_prove::<Ext>(
                        f0_ali.clone(), domain0, &params,
                    );
                    let prove_s = t0.elapsed().as_secs_f64();

                    // ── Verify ──
                    let t1 = Instant::now();
                    assert!(deep_fri_verify::<Ext>(&params, &proof));
                    let verify_ms = t1.elapsed().as_secs_f64() * 1e3;

                    let proof_bytes =
                        deep_fri_proof_size_bytes::<Ext>(&proof, params.stir);

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
                        mode_label: mode_label.to_string(),
                        soundness_suffix: report.csv_suffix(),
                    };

                    // ✅ Baseline keyed on (mode, air, k)
                    let bkey = (
                        mode_label.to_string(),
                        air.label().to_string(),
                        k,
                    );
                    if label == "2power16" {
                        paper_baseline.insert(bkey, row.clone());
                    } else if let Some(base) = paper_baseline.get(&(
                        mode_label.to_string(),
                        air.label().to_string(),
                        k,
                    )) {
                        row.delta_size_pct =
                            100.0 * (row.proof_bytes as f64
                                - base.proof_bytes as f64)
                                / base.proof_bytes as f64;
                        row.delta_prove_pct =
                            100.0 * (row.prove_s - base.prove_s) / base.prove_s;
                        row.delta_verify_pct =
                            100.0 * (row.verify_ms - base.verify_ms)
                                / base.verify_ms;
                        row.delta_throughput_pct =
                            100.0
                                * (row.prove_elems_per_s
                                    - base.prove_elems_per_s)
                                / base.prove_elems_per_s;
                    }

                    row.print_stdout();
                    std::io::stdout().flush().unwrap();
                    writer.write_all(row.to_line().as_bytes()).unwrap();
                    writer.flush().unwrap();

                    eprintln!(
                        "  [DONE ] air={} mode={} label={} k={} \
                         prove={:.2}s verify={:.2}ms proof={} B \
                         queries={} sec={:.1}b  [Fp{}]",
                        air.label(), mode_label, label, k,
                        prove_s, verify_ms, proof_bytes,
                        queries, report.achieved_bits,
                        Ext::DEGREE,
                    );

                    let run_secs = run_start.elapsed().as_secs_f64();
                    eprintln!(
                        "  [ETA  ] elapsed={:.2}s",
                        run_secs,
                    );
                }
            }
        }
    }

    g.finish();
}

criterion_group!(e2e, bench_e2e_mf_fri);
criterion_main!(e2e);