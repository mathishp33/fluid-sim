//! Banc d'essai et empreintes du solveur CPU.
//!  * `cargo test --release cpu_profile_benchmark -- --ignored --nocapture` : temps par pas et profil par partie du pas
//!    (`StepProfile`) sur quelques scénarios et tailles de grille ;
//!  * `cargo test --release cpu_state_fingerprints -- --ignored --nocapture` : empreinte de l'état complet après quelques pas, pour
//!    vérifier qu'une réécriture (parallélisme) ne change pas un bit, et que le résultat ne dépend pas du nombre de fils.
//! Variables : `MP_BENCH_SIZES=80x60,160x120` (tailles), `MP_BENCH_SCENES=0,4` (indices dans la liste du test), `MP_FP_STEPS=n`,
//! `RAYON_NUM_THREADS=n` (nombre de fils).

use super::mp_view::{build_scenario, MpScenario};
use std::time::Instant;

const SCENES: [MpScenario; 5] =
    [MpScenario::DamBreak, MpScenario::Boiling, MpScenario::PremixedFlame, MpScenario::SandColumn, MpScenario::CoalFire];

fn env_list<T>(name: &str, parse: impl Fn(&str) -> Option<T>, default: Vec<T>) -> Vec<T> {
    match std::env::var(name) {
        Ok(v) => v.split(',').filter_map(|s| parse(s.trim())).collect(),
        Err(_) => default,
    }
}

fn sizes(default: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    env_list(
        "MP_BENCH_SIZES",
        |s| {
            let (a, b) = s.split_once('x')?;
            Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?))
        },
        default,
    )
}

fn scenes(all: &[MpScenario]) -> Vec<MpScenario> {
    env_list("MP_BENCH_SCENES", |s| s.parse::<usize>().ok().and_then(|i| all.get(i).copied()), all.to_vec())
}

/// même taille physique que gpu_speed_benchmark (1.6 m de large)
fn dx_for(nx: usize) -> f64 {
    0.01 * 160.0 / nx as f64
}

#[test]
#[ignore]
fn cpu_profile_benchmark() {
    eprintln!("fils du solveur : {}", super::mp_par::threads());
    for &sc in &scenes(&SCENES) {
        for &(nx, ny) in &sizes(vec![(80, 60), (160, 120), (320, 240)]) {
            let mut s = build_scenario(sc, nx, ny, dx_for(nx), 9.81);
            for _ in 0..3 {
                s.step();
            }
            s.profile = Default::default();
            let n = if nx >= 320 { 6 } else if nx >= 160 { 12 } else { 30 };
            let t0 = Instant::now();
            for _ in 0..n {
                s.step();
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
            eprintln!(
                "{:<44} {:>9} {:>9.2} ms/pas{}",
                sc.name().chars().take(44).collect::<String>(),
                format!("{nx}×{ny}"),
                ms,
                if s.diverged { "  DIVERGÉ" } else { "" }
            );
            for (name, msp, frac) in s.profile.report() {
                if frac > 0.005 {
                    eprintln!("      {:<40} {:>9.2} ms {:>6.1} %", name, msp, 100.0 * frac);
                }
            }
        }
    }
}

/// Le nombre de fils du solveur CPU ne change pas un bit du résultat (transport, pression, chimie, grains, frottement, percolation).
#[test]
fn results_do_not_depend_on_the_number_of_cpu_threads() {
    for sc in [MpScenario::DamBreak, MpScenario::PremixedFlame, MpScenario::CoalFire] {
        let run = |t: usize| {
            super::mp_par::set_threads(t);
            let mut s = build_scenario(sc, 80, 60, dx_for(80), 9.81);
            for _ in 0..6 {
                s.step();
            }
            bincode::serialize(&s).expect("sérialisation de l'état")
        };
        let (a, b) = (run(1), run(4));
        super::mp_par::set_threads(0);
        assert!(a == b, "{} : l'état dépend du nombre de fils", sc.name());
    }
}

/// FNV-1a 64 bits
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[test]
#[ignore]
fn cpu_state_fingerprints() {
    let steps: usize = std::env::var("MP_FP_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    eprintln!("fils du solveur : {}, {steps} pas", super::mp_par::threads());
    for &sc in &scenes(&MpScenario::ALL) {
        for &(nx, ny) in &sizes(vec![(80, 60), (160, 120)]) {
            let mut s = build_scenario(sc, nx, ny, dx_for(nx), 9.81);
            for _ in 0..steps {
                s.step();
            }
            let bytes = bincode::serialize(&s).expect("sérialisation de l'état");
            eprintln!(
                "FP {:<44} {:>9} {:016x}  t = {:.6e} s, {} pas, {} refaits",
                sc.name().chars().take(44).collect::<String>(),
                format!("{nx}×{ny}"),
                fnv1a(&bytes),
                s.time,
                s.step_count,
                s.rollbacks
            );
        }
    }
}

/// Diagnostic du parallélisme (voir l'en-tête de `mp_par`) : débit d'un calcul pur (morceaux de ~0.85 ms) sur k fils, (a) en continu
/// pendant 2 s (fils système), (b) en rafales de 16 morceaux avec le groupe du solveur (attente active). Sur un portable en mode
/// équilibré, des rafales avec des fils qui s'endorment (rayon, fils système) ne vont pas plus vite qu'en séquentiel.
/// `cargo test --release parallel_throughput_diagnostic -- --ignored --nocapture`
#[test]
#[ignore]
fn parallel_throughput_diagnostic() {
    use std::hint::black_box;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let v: Vec<f64> = (0..120_000).map(|i| i as f64 + 0.5).collect();
    let work = |v: &[f64]| v.iter().map(|x| x.sqrt().ln_1p()).sum::<f64>();
    for &k in &[1usize, 2, 4, 8, 16] {
        // (a) continu
        let done = AtomicUsize::new(0);
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..k {
                s.spawn(|| {
                    while t0.elapsed().as_secs_f64() < 2.0 {
                        black_box(work(black_box(&v)));
                        done.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
        });
        let cont = done.load(Ordering::Relaxed) as f64 / t0.elapsed().as_secs_f64();
        // (b) rafales, groupe du solveur
        super::mp_par::set_threads(k);
        let t0 = Instant::now();
        let mut b = 0usize;
        while t0.elapsed().as_secs_f64() < 2.0 {
            super::mp_par::for_ranges(16, 1, |lo, hi| {
                for _ in lo..hi {
                    black_box(work(black_box(&v)));
                }
            });
            b += 1;
        }
        let burst = 16.0 * b as f64 / t0.elapsed().as_secs_f64();
        eprintln!("{k:>2} fils : continu {cont:6.0} morceaux/s, rafales (groupe du solveur) {burst:6.0} morceaux/s");
    }
    super::mp_par::set_threads(0);
}
