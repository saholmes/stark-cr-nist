use deep_ali::sizing::{eps_eff_from_lambda, r_for_bits_from_lambda};

#[derive(Clone)]
struct Sched<'a> {
    name: &'a str,
    folds: &'a [usize],
    // Lambda at r0=32, in bits. Fill these from your estimator/logs.
    // For any schedule where λ_32 is unknown, you can set it to None
    // and the CLI will skip or use a conservative placeholder.
    lambda_bits_at_32: Option<f64>,
}

fn main() {
    // Target security
    let target_bits = 128.0;
    let r0 = 32usize;

    // TODO: fill lambda_bits_at_32 from your measurements/estimator per schedule.
    // Example: baseline "paper" schedule [16,16,8] has ε_eff=0.96 ⇒
    // λ ≈ 32 * log2(1/(1-0.96)) ≈ 148.6.
    let schedules: Vec<Sched> = vec![
        Sched { name: "paper",         folds: &[16, 16, 8],      lambda_bits_at_32: Some(148.6) },
        Sched { name: "mod16",         folds: &[16, 16, 16, 16], lambda_bits_at_32: None },
        Sched { name: "uni32x3",       folds: &[32, 32, 32],     lambda_bits_at_32: None },
        Sched { name: "uni64x2x8",     folds: &[64, 64, 8],      lambda_bits_at_32: None },
        Sched { name: "hi64_32_8",     folds: &[64, 32, 8],      lambda_bits_at_32: None },
        Sched { name: "hi32_32_16",    folds: &[32, 32, 16],     lambda_bits_at_32: None },
        Sched { name: "uni128",        folds: &[128],            lambda_bits_at_32: None },
        Sched { name: "uni128x2",      folds: &[128, 128],       lambda_bits_at_32: None },
        Sched { name: "hi128_64",      folds: &[128, 64],        lambda_bits_at_32: None },
        Sched { name: "hi128_32",      folds: &[128, 32],        lambda_bits_at_32: None },
        Sched { name: "hi128_16",      folds: &[128, 16],        lambda_bits_at_32: None },
        Sched { name: "hi128_64_8",    folds: &[128, 64, 8],     lambda_bits_at_32: None },
        Sched { name: "hi128_32_8",    folds: &[128, 32, 8],     lambda_bits_at_32: None },
    ];

    println!("Schedule sizing (target ≈ {:.0} bits). r0 = {}.\n", target_bits, r0);
    println!("{:<14}  {:<20}  {:>10}  {:>10}  {:>8}", "name", "folds", "eps_eff", "r_128", "λ@32");
    println!("{}", "-".repeat(70));

    for s in schedules {
        match s.lambda_bits_at_32 {
            Some(lambda_bits) => {
                let eps = eps_eff_from_lambda(lambda_bits, r0);
                let r128 = r_for_bits_from_lambda(lambda_bits, r0, target_bits);
                println!(
                    "{:<14}  {:<20}  {:>10.5}  {:>10}  {:>8.1}",
                    s.name,
                    fmt_folds(s.folds),
                    eps,
                    r128,
                    lambda_bits
                );
            }
            None => {
                // If you don’t have λ@32 yet, print a placeholder line.
                println!(
                    "{:<14}  {:<20}  {:>10}  {:>10}  {:>8}",
                    s.name,
                    fmt_folds(s.folds),
                    "—",
                    "—",
                    "—"
                );
            }
        }
    }

    println!("\nNotes:");
    println!("- Fill lambda_bits_at_32 from your estimator/logs for each schedule.");
    println!("- eps_eff = 1 - 2^(-λ/32). Minimal r for 128 bits: r_128 = ceil(128 / (λ/32)).");
    println!("- Proof size and verify time scale ≈ linearly with r in this design.");
}

fn fmt_folds(folds: &[usize]) -> String {
    let mut s = String::new();
    s.push('[');
    for (i, m) in folds.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("{}", m));
    }
    s.push(']');
    s
}