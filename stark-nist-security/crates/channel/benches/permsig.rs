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

// add to imports
use deep_ali::trace_import::real_trace_inputs;

// ✅ DEEP‑ALI + MF‑FRI APIs
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

mod perm_signature;  // or: use deep_ali::perm_signature;

use perm_signature::{keygen, sign_generate_trace, perm_trace_inputs};

// ✅ Extension‑field imports
//use deep_ali::cubic_ext::{CubeExt, GoldilocksCubeConfig};

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

use deep_ali::sextic_ext::SexticExt;

type Ext = SexticExt;  // Fp⁶ = Fp³[u]/(u² − α)

//use deep_ali::octic_ext::OcticExt;

//type Ext = OcticExt;  // Fp⁶ = Fp³[u]/(u² − α)

// ---------------------
// CSV record
// ---------------------

#[derive(Default, Clone)]
struct CsvRow {
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
        "csv,label,k,schedule,proof_bytes,prove_s,verify_ms,prove_elems_per_s,delta_size_pct_vs_paper,delta_prove_pct_vs_paper,delta_verify_pct_vs_paper,delta_throughput_pct_vs_paper"
    }
    fn to_line(&self) -> String {
        format!(
            "csv,{},{},{},{},{:.6},{:.3},{:.6},{:.2},{:.2},{:.2},{:.2}\n",
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
            self.delta_throughput_pct
        )
    }
    fn print_stdout(&self) {
        print!("{}", self.to_line());
    }
}

// ---------------------
// Schedule helpers
// ---------------------

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

// ✅ Normalize schedule so final layer size = 1
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

// ---------------------
// MF‑FRI benchmark (WITH DEEP‑ALI)
// ---------------------


fn bench_e2e_mf_fri(c: &mut Criterion) {
    eprintln!(
        "[RAYON CHECK] current_num_threads = {}",
        rayon::current_num_threads()
    );

    let mut g: BenchmarkGroup<WallTime> = c.benchmark_group("e2e_mf_fri_perm");
    g.warm_up_time(Duration::from_secs(5));
    g.measurement_time(Duration::from_secs(20));
    g.sample_size(10);

    let r: usize = 87;
    let seed_z: u64 = 0xDEEF_BAAD;
    let k_lo = 14usize;  // minimum k for n=4 permanent trace
    let k_hi = 20usize;

    // ── FRI-Perm signature parameters ──────────────────────────────
    let perm_n: usize = 4;     // matrix dimension (start small!)
    let perm_width: usize = 4; // treewidth (= n for prototype)
    // n=4, w=4: trace has 4 × 16 × 4 = 256 computation rows
    // Needs domain ≥ 256 × 4 (rate) = 1024 → k ≥ 10
    //
    // n=8, w=8:  trace has 8 × 256 × 8 = 16384 rows → k ≥ 16
    // n=8, w=4:  trace has 8 × 256 × 4 = 8192 rows  → k ≥ 15
    //            (bounded treewidth version)
    // ────────────────────────────────────────────────────────────────

    let presets: &[(&str, &[usize])] = &[
        ("2power16", &[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]),
    ];

    let file = File::create("benchmarkdata_perm.csv").unwrap();
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{}", CsvRow::header()).unwrap();
    println!("{}", CsvRow::header());

    let mut paper_baseline: HashMap<usize, CsvRow> = HashMap::new();
    let mut rng_seed = 1337u64;

    for &(label, schedule) in presets {
        let ks = ks_for_schedule(schedule, k_lo, k_hi);

        eprintln!(
            "\n[PRESET] label={} schedule={} total_k={}",
            label, schedule_str(schedule), ks.len()
        );

        for &k in &ks {
            let run_start = Instant::now();
            let n0 = 1usize << k;
            g.throughput(Throughput::Elements(n0 as u64));

            let normalized_schedule = normalize_fri_schedule(n0, schedule.to_vec());

            eprintln!(
                "[START] label={} schedule={} k={} (n={}) [PERM n={} w={}]",
                label,
                schedule_str(&normalized_schedule),
                k, n0, perm_n, perm_width,
            );

            rng_seed = rng_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let mut rng = StdRng::seed_from_u64(rng_seed);

            // ─────────────────────────────────────────────────────────
            // ✅ FRI-PERM TRACE (replaces real_trace_inputs)
            // ─────────────────────────────────────────────────────────

            let perm_inputs = perm_signature::perm_trace_inputs(
                n0, 4, perm_n, perm_width, rng_seed,
            );

            let a_eval = perm_inputs.a_eval;
            let s_eval = perm_inputs.s_eval;
            let e_eval = perm_inputs.e_eval;
            let t_eval = perm_inputs.t_eval;

            eprintln!(
                "[PERM ] permanent = {:?}, pk_commitment = {:02x?}...",
                perm_inputs.permanent,
                &perm_inputs.pk.commitment[..4],
            );

            // ─────────────────────────────────────────────────────────
            // Everything below is UNCHANGED from your original code.
            // The FRI prover doesn't care what computation the trace
            // represents — it just proves low-degree proximity.
            // ─────────────────────────────────────────────────────────

            let domain0 = FriDomain::new_radix2(n0);

            let z_fp3 = Fp3 {
                a0: F::rand(&mut rng),
                a1: F::rand(&mut rng),
                a2: F::rand(&mut rng),
            };

            let (f0_ali, _z_used, _c_star) =
                deep_ali_merge_evals(
                    &a_eval, &s_eval, &e_eval, &t_eval,
                    domain0.omega, z_fp3,
                );

            let params = DeepFriParams {
                schedule: normalized_schedule.clone(),
                r,
                seed_z,
                coeff_commit_final: true,
                d_final: 1,
            };

            // ── Prove ──
            let t0 = Instant::now();
            let proof = deep_fri_prove::<Ext>(f0_ali.clone(), domain0, &params);
            let prove_s = t0.elapsed().as_secs_f64();

            // ── Verify ──
            let t1 = Instant::now();
            assert!(deep_fri_verify::<Ext>(&params, &proof));
            let verify_ms = t1.elapsed().as_secs_f64() * 1e3;

            let proof_bytes = deep_fri_proof_size_bytes::<Ext>(&proof);

            let mut row = CsvRow {
                label: format!("{}_perm{}w{}", label, perm_n, perm_width),
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

            row.print_stdout();
            std::io::stdout().flush().unwrap();
            writer.write_all(row.to_line().as_bytes()).unwrap();
            writer.flush().unwrap();

            eprintln!(
                "[DONE ] label={} k={} prove={:.2}s verify={:.2}ms proof={} bytes  [ext=Fp{}]",
                row.label, k, prove_s, verify_ms, proof_bytes,
                std::mem::size_of::<Ext>() / std::mem::size_of::<F>(),
            );
        }
    }
    g.finish();
}

criterion_group!(e2e, bench_e2e_mf_fri);
criterion_main!(e2e);