//! Validation du solveur GPU résident (f32) contre le solveur CPU (f64) : mêmes scènes, même temps simulé, grandeurs comparées.
//! Ces tests ont besoin d'un adaptateur wgpu (lavapipe suffit) ; sans adaptateur ils sont ignorés (message).

use super::gpu_multiphase::MultiphaseGpu;
use super::mp_view::{build_scenario, MpScenario};
use super::multiphase::MultiphaseSim;

/// grandeurs globales d'une simulation
#[derive(Debug, Clone)]
pub(crate) struct Metrics {
    pub time: f64,
    pub steps: usize,
    pub mass: Vec<f64>,
    pub area: Vec<f64>,
    /// centre de masse de chaque phase (x, y)
    pub com: Vec<(f64, f64)>,
    pub kinetic: f64,
    pub internal: f64,
    pub t_min: f64,
    pub t_max: f64,
    pub p_min: f64,
    pub p_max: f64,
    pub umax: f64,
}

pub(crate) fn metrics(s: &MultiphaseSim) -> Metrics {
    let np = s.reg.phases.len();
    let n = s.nx * s.ny;
    let mut com = vec![(0.0, 0.0); np];
    for k in 0..np {
        let (mut mx, mut my, mut m) = (0.0, 0.0, 0.0);
        for c in 0..n {
            let (x, y) = s.center(c % s.nx, c / s.nx);
            mx += s.mass[k][c] * x;
            my += s.mass[k][c] * y;
            m += s.mass[k][c];
        }
        com[k] = if m > 0.0 { (mx / m, my / m) } else { (0.0, 0.0) };
    }
    let fl: Vec<usize> = (0..n).filter(|&c| !s.is_solid(c)).collect();
    Metrics {
        time: s.time,
        steps: s.step_count,
        mass: (0..np).map(|k| s.total_mass(k)).collect(),
        area: (0..np).map(|k| s.phase_area(k)).collect(),
        com,
        kinetic: s.kinetic_energy(),
        internal: s.total_internal_energy(),
        t_min: fl.iter().map(|&c| s.temp[c]).fold(f64::MAX, f64::min),
        t_max: fl.iter().map(|&c| s.temp[c]).fold(f64::MIN, f64::max),
        p_min: fl.iter().map(|&c| s.p[c]).fold(f64::MAX, f64::min),
        p_max: fl.iter().map(|&c| s.p[c]).fold(f64::MIN, f64::max),
        umax: s.max_speed(),
    }
}

/// GPU disponible ? (lavapipe, pilote réel…)
/// au plus `GPU_TESTS_AT_ONCE` tests GPU à la fois : un processeur graphique intégré ne va pas plus vite avec davantage de calculs en
/// même temps (il partage son temps et la mémoire du système), et pendant les arrêts de la suite complète (une dizaine de tests GPU
/// en même temps que des tests CPU qui chargent tous les cœurs) Windows a noté des délais de pilote dépassés (« LiveKernelEvent 141 ») —
/// la cause principale était la fuite de mémoire de périphérique corrigée par `shared_gpu`. Jeton pris par le fil du test à son premier
/// appareil GPU, rendu à la fin du test (fin du fil).
const GPU_TESTS_AT_ONCE: usize = 2;
static GPU_SLOTS: std::sync::Mutex<usize> = std::sync::Mutex::new(0);
static GPU_SLOT_FREED: std::sync::Condvar = std::sync::Condvar::new();
struct GpuSlot;
impl Drop for GpuSlot {
    fn drop(&mut self) {
        let mut n = GPU_SLOTS.lock().unwrap_or_else(|e| e.into_inner());
        *n -= 1;
        GPU_SLOT_FREED.notify_one();
    }
}
thread_local! {
    static GPU_SLOT: std::cell::RefCell<Option<GpuSlot>> = const { std::cell::RefCell::new(None) };
}
fn take_gpu_slot() {
    GPU_SLOT.with(|slot| {
        if slot.borrow().is_some() {
            return;
        }
        let mut n = GPU_SLOTS.lock().unwrap_or_else(|e| e.into_inner());
        while *n >= GPU_TESTS_AT_ONCE {
            n = GPU_SLOT_FREED.wait(n).unwrap_or_else(|e| e.into_inner());
        }
        *n += 1;
        *slot.borrow_mut() = Some(GpuSlot);
    });
}

pub(crate) fn gpu_or_skip(sim: &MultiphaseSim) -> Option<MultiphaseGpu> {
    take_gpu_slot();
    match MultiphaseGpu::try_new(sim) {
        Ok(g) => Some(g),
        Err(e) => {
            if e.contains("compilation") || e.contains("initialisation") {
                panic!("GPU : {e}");
            }
            eprintln!("test GPU ignoré : {e}");
            None
        }
    }
}

/// avance `sim` jusqu'à t_end sur le GPU (relit l'état à la fin)
pub(crate) fn run_gpu(g: &mut MultiphaseGpu, sim: &mut MultiphaseSim, t_end: f64, max_steps: usize) {
    let mut k = 0;
    while sim.time < t_end && !sim.diverged && k < max_steps {
        g.step(sim);
        k += 1;
    }
    g.download(sim);
}


/// Accord d'UN pas (répété `steps` fois depuis l'état CPU) après `warm` pas CPU : écarts max par champ bornés. Pour les phénomènes à
/// seuil (nucléation, auto-allumage, panaches instables) la comparaison à long terme n'a pas de sens au-delà de quelques %, l'accord
/// pas à pas valide la physique implémentée.
/// `tol_m` : écart max des masses (rapporté au max de chaque phase) ; les VAPEURS d'un changement de phase ont `tol_vap` : leur production
/// est bornée par la fermeture volumique 1 − Σ m/ρ, calculée en f32 avec une erreur absolue ~1e-7 (masses stockées en f32) — quand
/// 1 − Σ m/ρ ~ 1e-5 (cellule presque pleine de liquide), c'est 1 % d'écart sur la vapeur produite dans la cellule.
pub(crate) fn one_step_agreement(name: &str, cpu: MultiphaseSim, warm: usize, steps: usize, tol_m: f64, tol_t: f64) {
    one_step_agreement_vap(name, cpu, warm, steps, tol_m, tol_t, 0.05);
}

/// `one_step_agreement` avec la tolérance des vapeurs `tol_vap` (0.05 par défaut, voir ci-dessus)
pub(crate) fn one_step_agreement_vap(name: &str, mut cpu: MultiphaseSim, warm: usize, steps: usize, tol_m: f64, tol_t: f64, tol_vap: f64) {
    for _ in 0..warm {
        cpu.step();
    }
    let Some(mut g) = gpu_or_skip(&cpu) else { return };
    for k in 0..steps {
        // chaque pas repart de l'état CPU (accord d'UN pas) : sans cela les écarts s'accumulent d'un pas à l'autre et, près de la
        // saturation (ébullition), la nucléation les amplifie d'un facteur 10 à 100 par pas — CPU contre CPU perturbé de 1e-9 compris
        let mut gs = clone_sim(&cpu);
        g.upload(&gs);
        cpu.step();
        g.step(&mut gs);
        g.download(&mut gs);
        eprintln!("== {name} : pas {k} (dt {:.3e} / {:.3e})", cpu.last_dt, gs.last_dt);
        for (n, v, at) in field_diff(&cpu, &gs) {
            if v > 0.0 {
                eprintln!("   {n:<28} {v:.3e} (case {at})");
            }
            if n.starts_with('m') {
                let k: usize = n[1..].split(' ').next().unwrap().parse().unwrap();
                let tol = if cpu.reg.changes.iter().any(|pc| pc.vapor == k) { tol_vap.max(tol_m) } else { tol_m };
                if v >= tol {
                    let c = at;
                    for (lab, s) in [("CPU", &cpu), ("GPU", &gs)] {
                        let m: Vec<String> = (0..s.mass.len()).map(|q| format!("{:.4e}", s.mass[q][c])).collect();
                        eprintln!("   {lab} case {c} : p {:.5e} T {:.3} tg {:?} bloquée {} masses {}", s.p[c], s.temp[c],
                            (0..s.ngroups()).map(|g| (s.tg[g][c] * 100.0).round() / 100.0).collect::<Vec<_>>(), s.pressure_clamped[c], m.join(" "));
                    }
                }
                assert!(v < tol, "{name} : {n} {v} (case {at})");
            } else if n == "T" {
                assert!(v < tol_t, "{name} : T {v} K (case {at})");
            }
        }
        // repart du même état (celui du CPU)
        g.upload(&cpu);
        gs = clone_sim(&cpu);
    }
}

/// copie d'une simulation (champs utiles ; MultiphaseSim n'est pas Clone)
fn clone_sim(s: &MultiphaseSim) -> MultiphaseSim {
    let mut c = MultiphaseSim::new(s.nx, s.ny, s.dx, s.reg.clone(), s.ambient_phase);
    c.gravity = s.gravity;
    c.bc = s.bc;
    c.t_ambient = s.t_ambient;
    c.mass = s.mass.clone();
    c.alpha = s.alpha.clone();
    c.e = s.e.clone();
    c.eg = s.eg.clone();
    c.tg = s.tg.clone();
    c.phase_heat_exchange = s.phase_heat_exchange;
    c.temp = s.temp.clone();
    c.p = s.p.clone();
    c.u = s.u.clone();
    c.v = s.v.clone();
    c.q_u = s.q_u.clone();
    c.q_v = s.q_v.clone();
    c.percolation = s.percolation;
    c.implicit_viscosity = s.implicit_viscosity;
    c.solid = s.solid.clone();
    c.rho_mix = s.rho_mix.clone();
    c.mu_cell = s.mu_cell.clone();
    c.k_cell = s.k_cell.clone();
    c.special = s.special.clone();
    c.thermal = s.thermal.clone();
    c.inlet = s.inlet;
    c.ambient_mix = s.ambient_mix.clone();
    c.mixtures = s.mixtures.clone();
    c.chemistry = s.chemistry;
    c.radiation = s.radiation;
    c.flame_factor = s.flame_factor.clone();
    c.heat_release = s.heat_release.clone();
    c.heat_target = s.heat_target.clone();
    c.heat_active = s.heat_active;
    c.heat_power_max = s.heat_power_max;
    c.dt_max = s.dt_max;
    c.dt_chem = s.dt_chem;
    c.dt_phase = s.dt_phase;
    // pas précédent : borne la croissance du pas (choose_dt)
    c.last_dt = s.last_dt;
    c.pressure_clamped = s.pressure_clamped.clone();
    c.boundary_flux = s.boundary_flux.clone();
    c.transferred = s.transferred.clone();
    c.reacted = s.reacted.clone();
    c.frozen = s.frozen.clone();
    c.fusion_front = s.fusion_front.clone();
    c.step_count = s.step_count;
    c.time = s.time;
    c.conservative_momentum = s.conservative_momentum;
    c.maccormack = s.maccormack;
    c.tp_mix = s.tp_mix.clone();
    c.conduction = s.conduction;
    c.phase_change = s.phase_change;
    c.surface_tension = s.surface_tension;
    c.viscous = s.viscous;
    c
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs()).max(1e-30)
}

#[test]
fn gpu_compiles_and_runs_dam_break() {
    let mk = || build_scenario(MpScenario::DamBreak, 64, 40, 0.01, 9.81);
    let mut cpu = mk();
    let mut gs = mk();
    let Some(mut g) = gpu_or_skip(&gs) else { return };
    let t_end = 0.3;
    let t0 = std::time::Instant::now();
    run_gpu(&mut g, &mut gs, t_end, 20000);
    let tg = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    cpu.run(t_end);
    let tc = t0.elapsed().as_secs_f64();
    let (a, b) = (metrics(&cpu), metrics(&gs));
    eprintln!("CPU {:.2}s {a:?}\nGPU {:.2}s {b:?}\nGPU last {:?}", tc, tg, g.last);
    assert!(!gs.diverged && gs.state_is_sane(), "GPU divergé");
    for k in 0..a.mass.len() {
        assert!(rel(a.mass[k], b.mass[k]) < 1e-3, "masse phase {k} : {} vs {}", a.mass[k], b.mass[k]);
    }
    let water = (0..cpu.reg.phases.len()).filter(|&k| !cpu.reg.phases[k].is_gas()).max_by(|&i, &j| a.mass[i].total_cmp(&a.mass[j])).unwrap();
    let d = ((a.com[water].0 - b.com[water].0).powi(2) + (a.com[water].1 - b.com[water].1).powi(2)).sqrt();
    assert!(d < 0.02 * 0.64, "centre de masse de l'eau : CPU {:?} GPU {:?}", a.com[water], b.com[water]);
    assert!(rel(a.kinetic, b.kinetic) < 0.15, "énergie cinétique {} vs {}", a.kinetic, b.kinetic);
}

/// construit deux fois la même scène, avance le CPU et le GPU jusqu'à t_end, imprime les grandeurs ; renvoie (cpu, gpu, gpu solver)
pub(crate) fn compare(name: &str, mk: &dyn Fn() -> MultiphaseSim, t_end: f64) -> Option<(MultiphaseSim, MultiphaseSim, MultiphaseGpu)> {
    let mut cpu = mk();
    let mut gs = mk();
    let mut g = gpu_or_skip(&gs)?;
    let t0 = std::time::Instant::now();
    run_gpu(&mut g, &mut gs, t_end, 50000);
    let tg = t0.elapsed().as_secs_f64();
    let t0 = std::time::Instant::now();
    cpu.run(t_end);
    let tc = t0.elapsed().as_secs_f64();
    let (a, b) = (metrics(&cpu), metrics(&gs));
    eprintln!("== {name}\n CPU {tc:.2}s {a:?}\n GPU {tg:.2}s {b:?}\n GPU dernier pas {:?}", g.last);
    assert!(!gs.diverged && gs.state_is_sane(), "{name} : GPU divergé ou état invalide");
    assert!(!cpu.diverged, "{name} : CPU divergé");
    Some((cpu, gs, g))
}

fn liquid_with_most_mass(s: &MultiphaseSim) -> usize {
    (0..s.reg.phases.len()).filter(|&k| !s.reg.phases[k].is_gas()).max_by(|&i, &j| s.total_mass(i).total_cmp(&s.total_mass(j))).unwrap()
}

fn com_dist(a: &Metrics, b: &Metrics, k: usize) -> f64 {
    ((a.com[k].0 - b.com[k].0).powi(2) + (a.com[k].1 - b.com[k].1).powi(2)).sqrt()
}

#[test]
fn gpu_matches_cpu_drop_splash_surface_tension() {
    let mk = || build_scenario(MpScenario::DropSplash, 48, 40, 0.004, 9.81);
    let Some((cpu, gs, _)) = compare("goutte", &mk, 0.12) else { return };
    let (a, b) = (metrics(&cpu), metrics(&gs));
    let w = liquid_with_most_mass(&cpu);
    assert!(rel(a.mass[w], b.mass[w]) < 1e-4);
    assert!(com_dist(&a, &b, w) < 0.01 * 0.192, "centre de masse {:?} {:?}", a.com[w], b.com[w]);
    assert!(rel(a.kinetic, b.kinetic) < 0.2, "énergie cinétique {} {}", a.kinetic, b.kinetic);
}

#[test]
fn gpu_matches_cpu_oil_water_air() {
    let mk = || build_scenario(MpScenario::OilWaterAir, 40, 40, 0.004, 9.81);
    let Some((cpu, gs, _)) = compare("huile-eau-air", &mk, 0.15) else { return };
    let (a, b) = (metrics(&cpu), metrics(&gs));
    for k in 0..a.mass.len() {
        assert!(rel(a.mass[k], b.mass[k]) < 1e-4, "masse {k}");
        if a.mass[k] > 0.0 {
            assert!(com_dist(&a, &b, k) < 0.004, "phase {k} : {:?} {:?}", a.com[k], b.com[k]);
        }
    }
}

#[test]
fn gpu_matches_cpu_boiling_on_hot_plate() {
    // eau à 99 °C sur une plaque d'acier à 700 K : ébullition nucléée sur la plaque (conduction solide → fluide, changement de phase)
    let mk = || {
        let mut s = build_scenario(MpScenario::Boiling, 32, 32, 0.003, 9.81);
        for c in 0..s.nx * s.ny {
            let t = if s.is_solid(c) { 700.0 } else if s.temp[c] > 350.0 { 372.0 } else { continue };
            if s.is_solid(c) {
                let rc = s.reg.solids[s.solid[c] as usize].rho_cp();
                s.e[c] = rc * t;
            } else {
                let e: f64 = (0..s.reg.phases.len()).map(|k| s.mass[k][c] * s.reg.phases[k].e_int(t)).sum();
                s.e[c] = e;
            }
        }
        s.refresh();
        s
    };
    // pas à pas pendant l'ébullition (accord à l'arrondi f32 près)
    // (eau : 5e-4 — la pression d'une cellule de nucléation presque pleine de liquide, 1 − Σ m/ρ ~ 1e-5 connu en f32, varie de
    // quelques kPa ; vitesses ~5 mm/s d'écart, flux d'eau ~3e-4 relatif à l'interface ; la version à une température passait à
    // 9.7e-5 pour 1e-4, et ses écarts montaient à 1e-3 – 1e-2 en deux pas depuis d'autres états de la même scène)
    one_step_agreement("ébullition", mk(), 120, 3, 5e-4, 2.0);
    // à long terme : la nucléation est un phénomène à seuil (surchauffe) ; on vérifie bilans et ordre de grandeur
    let Some((cpu, gs, _)) = compare("ébullition", &mk, 0.08) else { return };
    let (a, b) = (metrics(&cpu), metrics(&gs));
    eprintln!(" transféré CPU {:?} GPU {:?}", cpu.transferred, gs.transferred);
    let m0 = mk();
    for k in 0..a.mass.len() {
        let d = gs.total_mass(k) - m0.total_mass(k) - gs.boundary_flux[k];
        let tr: f64 = gs.reg.changes.iter().zip(&gs.transferred).map(|(pc, t)| if pc.vapor == k { *t } else if pc.liquid == k { -*t } else { 0.0 }).sum();
        assert!((d - tr).abs() < 1e-4 * m0.total_mass_all(), "bilan phase {k} : {d} vs transféré {tr}");
    }
    // plaque à 700 K sous de l'eau à 99 °C : ébullition explosive (jets d'eau par le haut ouvert, ~20 m/s) — chaotique : seuls les
    // bilans et l'existence de l'ébullition sont comparés à long terme
    assert!(cpu.transferred[0] > 0.0 && gs.transferred[0] > 0.0, "pas d'ébullition");
    assert!(b.umax < 200.0 && b.t_max < 800.0, "GPU : état non physique {:?}", b);
    let _ = a;
}

#[test]
fn gpu_matches_cpu_premixed_methane_flame() {
    // chimie (Arrhenius, CH4 + CO réversible), flamme épaissie, diffusion des espèces, rayonnement, bord ouvert
    let mk = || build_scenario(MpScenario::PremixedFlame, 64, 12, 0.001, 0.0);
    let Some((cpu, gs, g)) = compare("flamme de prémélange", &mk, 0.02) else { return };
    let fpos = |s: &MultiphaseSim| {
        // position du front : abscisse moyenne où T dépasse 1200 K, ligne du milieu
        let j = s.ny / 2;
        (0..s.nx).filter(|&i| s.temp[s.idx(i, j)] > 1200.0).map(|i| i as f64 + 0.5).fold(0.0f64, f64::max) * s.dx
    };
    eprintln!(" chaleur CPU {:.4e} GPU {:.4e} ; front CPU {:.4} GPU {:.4} ; rayonné {:.3e} {:.3e} ; F max {:.2} {:.2} ; budgets {:?}",
        cpu.chem_heat, gs.chem_heat, fpos(&cpu), fpos(&gs), cpu.radiated, gs.radiated,
        cpu.flame_factor.iter().fold(0.0f64, |a, &b| a.max(b)), gs.flame_factor.iter().fold(0.0f64, |a, &b| a.max(b)), g.last);
    assert!(cpu.chem_heat > 0.0 && gs.chem_heat > 0.0);
    // La flamme n'est pas convergée en temps au dt par défaut : le CPU seul, avec dt_max = 2e-5 / 1e-5 / 5e-6, donne une chaleur
    // de 256 / 268 / 271 au lieu de 234 et un front 3 à 7 mailles plus loin. Le GPU suit une autre suite de dt (f32 → autres décisions
    // de pas) : à long terme seul l'ordre de grandeur est comparable (Radeon : chaleur 259, front +4 mailles) ; l'accord pas à pas
    // ci-dessous valide les équations.
    assert!(rel(cpu.chem_heat, gs.chem_heat) < 0.2, "chaleur dégagée {} {}", cpu.chem_heat, gs.chem_heat);
    assert!((fpos(&cpu) - fpos(&gs)).abs() <= 8.0 * cpu.dx, "front {} {}", fpos(&cpu), fpos(&gs));
    for j in 0..cpu.reacted.len() {
        eprintln!("  réaction {j} : CPU {:.4e} GPU {:.4e}", cpu.reacted[j], gs.reacted[j]);
    }
    one_step_agreement("flamme de prémélange", mk(), 300, 2, 5e-3, 3.0);
}

#[test]
fn gpu_matches_cpu_confined_hydrogen_explosion() {
    // réactions réversibles (dissociation), fermée : la pression finale se compare à l'équilibre
    let mk = || build_scenario(MpScenario::ConfinedExplosion, 24, 24, 0.002, 0.0);
    // accord d'un pas en général à 1e-6 ; une cellule qui s'enflamme peut franchir un seuil de l'intégration chimique d'un côté de
    // l'arrondi f32 seulement (départ au pas 100 : cellule 431, 1781 / 1696 K, dt_chem 1.86e-5 / 2.26e-5, O2 à 1.1e-2 du max ; sa
    // vapeur à 4.99e-2 du max, 5.02e-2 depuis la forme adimensionnée de tau_from — d'où 6e-2 pour la vapeur ; les autres pas : 3e-6)
    one_step_agreement_vap("explosion confinée H2", mk(), 100, 3, 2e-2, 100.0, 6e-2);
    let Some((cpu, gs, g)) = compare("explosion confinée H2", &mk, 0.006) else { return };
    let (a, b) = (metrics(&cpu), metrics(&gs));
    eprintln!(" chaleur CPU {:.4e} GPU {:.4e} ; p max {:.4e} {:.4e} ; T max {:.1} {:.1} ; {:?}", cpu.chem_heat, gs.chem_heat, a.p_max, b.p_max, a.t_max, b.t_max, g.last);
    for j in 0..cpu.reacted.len() {
        eprintln!("  réaction {j} : CPU {:.4e} GPU {:.4e}", cpu.reacted[j], gs.reacted[j]);
    }
    assert!(rel(cpu.chem_heat, gs.chem_heat) < 0.05, "chaleur {} {}", cpu.chem_heat, gs.chem_heat);
    assert!(rel(a.p_max, b.p_max) < 0.05, "p max {} {}", a.p_max, b.p_max);
    // T max = point chaud local (gaz brûlés comprimés au centre) : sensible au moment exact de l'auto-allumage → comparaison large ;
    // la température moyenne (masse) est comparée finement
    assert!(rel(a.t_max, b.t_max) < 0.12, "T max {} {}", a.t_max, b.t_max);
    let tmean = |s: &MultiphaseSim| {
        let n = s.nx * s.ny;
        let m: Vec<f64> = (0..n).map(|c| (0..s.reg.phases.len()).map(|k| s.mass[k][c]).sum()).collect();
        (0..n).map(|c| m[c] * s.temp[c]).sum::<f64>() / m.iter().sum::<f64>()
    };
    eprintln!(" T moyenne CPU {:.1} GPU {:.1}", tmean(&cpu), tmean(&gs));
    assert!((tmean(&cpu) - tmean(&gs)).abs() < 30.0);
    // énergie totale (cuve fermée : seules les pertes radiatives la font varier), rapportée à la chaleur de réaction
    eprintln!(" énergie interne CPU {:.4e} GPU {:.4e} ; rayonné {:.4e} {:.4e}", a.internal, b.internal, cpu.radiated, gs.radiated);
    assert!((a.internal - b.internal).abs() < 0.01 * cpu.chem_heat, "énergie interne {} {}", a.internal, b.internal);
}

/// écart maximal champ par champ entre deux états (diagnostic)
pub(crate) fn field_diff(a: &MultiphaseSim, b: &MultiphaseSim) -> Vec<(String, f64, usize)> {
    let n = a.nx * a.ny;
    let mut out = vec![];
    let mut push = |name: String, x: &[f64], y: &[f64], scale: f64| {
        let mut m = 0.0f64;
        let mut at = 0;
        for i in 0..x.len().min(y.len()) {
            let d = (x[i] - y[i]).abs() / scale.max(1e-300);
            if d > m || d.is_nan() {
                m = d;
                at = i;
            }
        }
        out.push((name, m, at));
    };
    for k in 0..a.reg.phases.len() {
        let sc = a.mass[k].iter().fold(0.0f64, |s, &v| s.max(v.abs())).max(1e-12);
        push(format!("m{k} ({})", a.reg.phases[k].name), &a.mass[k], &b.mass[k], sc);
    }
    push("T".into(), &a.temp, &b.temp, 1.0);
    let pa: Vec<f64> = a.p.iter().map(|p| p - 101325.0).collect();
    let pb: Vec<f64> = b.p.iter().map(|p| p - 101325.0).collect();
    push("p−p0 (Pa)".into(), &pa, &pb, 1.0);
    let us = a.max_speed().max(1e-9);
    push("u".into(), &a.u, &b.u, us);
    push("v".into(), &a.v, &b.v, us);
    push("flamme".into(), &a.flame_factor, &b.flame_factor, 1.0);
    let q = a.heat_release.iter().fold(0.0f64, |s, &v| s.max(v.abs())).max(1e-12);
    push("dégagement".into(), &a.heat_release, &b.heat_release, q);
    let _ = n;
    out
}

/// un pas CPU et un pas GPU depuis le même état : écarts par champ (diagnostic, `cargo test -- --ignored --nocapture`)
#[test]
#[ignore]
fn gpu_one_step_field_diffs() {
    let sc = std::env::var("MP_SC").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(6);
    let warm = std::env::var("MP_WARM").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(20);
    let steps = std::env::var("MP_STEPS").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(3);
    let mk = || {
        let sc = MpScenario::from_index(sc);
        let (nx, ny, dx, g) = match sc {
            MpScenario::PremixedFlame => (64, 12, 0.001, 0.0),
            MpScenario::ConfinedExplosion => (24, 24, 0.002, 0.0),
            _ => (40, 32, 0.004, 9.81),
        };
        let mut s = build_scenario(sc, nx, ny, dx, g);
        for _ in 0..warm {
            s.step();
        }
        let off = |v: &str| std::env::var(v).is_ok();
        if off("MP_NOCHEM") { s.chemistry = false; }
        if off("MP_NOCOND") { s.conduction = false; }
        if off("MP_NORAD") { s.radiation = false; }
        if off("MP_NOVISC") { s.viscous = false; }
        if off("MP_NOST") { s.surface_tension = false; }
        if off("MP_NODIFF") { for p in s.reg.phases.iter_mut() { p.diffusivity = 0.0; } }
        if let Ok(k) = std::env::var("MP_KTH") { s.flame_thermo_substeps = k.parse().unwrap(); }
        s
    };
    let mut cpu = mk();
    let mut gs = mk();
    let Some(mut g) = gpu_or_skip(&gs) else { return };
    for k in 0..steps {
        let dc = cpu.step();
        let dg = g.step(&mut gs);
        g.download(&mut gs);
        eprintln!("pas {k} : dt CPU {dc:.6e} GPU {dg:.6e} ; Newton {} / {} ; PCG {} / {} ; dt_chem {:.3e} / {:.3e} ; dt_phase {:.3e} / {:.3e}",
            cpu.last_newton_iters, gs.last_newton_iters, cpu.last_pressure_iters, gs.last_pressure_iters, cpu.dt_chem, gs.dt_chem, cpu.dt_phase, gs.dt_phase);
        for (name, d, at) in field_diff(&cpu, &gs) {
            eprintln!("   {name:<28} {d:.3e} (case {at})");
        }
        if std::env::var("MP_ROWS").is_ok() {
            let k = 9.min(cpu.reg.phases.len() - 1);
            let sc = cpu.mass[k].iter().fold(0.0f64, |a, &b| a.max(b));
            for j in 0..cpu.ny {
                let row: Vec<String> = (0..cpu.nx).map(|i| { let c = cpu.idx(i, j); format!("{:+.0}", 1e4 * (gs.mass[k][c] - cpu.mass[k][c]) / sc) }).collect();
                eprintln!("   j={j:2} {}", row.join(" "));
            }
            for j in 0..cpu.ny {
                let row: Vec<String> = (0..cpu.nx).map(|i| { let c = cpu.idx(i, j); format!("{:+.1}", gs.temp[c] - cpu.temp[c]) }).collect();
                eprintln!("   T j={j:2} {}", row[..40].join(" "));
            }
        }
        // repart du même état (CPU) pour isoler l'écart d'UN pas
        g.upload(&cpu);
        gs.time = cpu.time;
        gs.step_count = cpu.step_count;
        gs.dt_chem = cpu.dt_chem;
        gs.dt_phase = cpu.dt_phase;
        let t = cpu.time;
        let _ = t;
    }
}

/// avancement des masses seul (advection PLIC) : CPU vs GPU depuis le même état (diagnostic)
#[test]
#[ignore]
fn gpu_advection_only_diffs() {
    let sc = std::env::var("MP_SC").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(6);
    let warm = std::env::var("MP_WARM").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(60);
    let mk = || {
        let sc = MpScenario::from_index(sc);
        let (nx, ny, dx, g) = match sc {
            MpScenario::PremixedFlame => (64, 12, 0.001, 0.0),
            MpScenario::ConfinedExplosion => (24, 24, 0.002, 0.0),
            _ => (40, 32, 0.004, 9.81),
        };
        let mut s = build_scenario(sc, nx, ny, dx, g);
        for _ in 0..warm {
            s.step();
        }
        s
    };
    let mut cpu = mk();
    let mut gs = mk();
    let Some(mut g) = gpu_or_skip(&gs) else { return };
    cpu.refresh();
    let before = (cpu.mass.clone(), cpu.alpha.clone(), cpu.u.clone(), cpu.v.clone());
    let dt = cpu.choose_dt();
    if std::env::var("MP_ONE").is_ok() {
        // un seul balayage (le premier du pas)
        let mut aw: Vec<Vec<f64>> = cpu.alpha.clone();
        cpu.mfx.iter_mut().for_each(|v| *v = 0.0);
        cpu.mfy.iter_mut().for_each(|v| *v = 0.0);
        cpu.m_tot_old = (0..cpu.nx * cpu.ny).map(|c| (0..cpu.mass.len()).map(|k| cpu.mass[k][c]).sum()).collect();
        let first = cpu.step_count % 2;
        eprintln!("premier balayage : direction {first}");
        cpu.plic_sweep(first, dt, &mut aw);
        g.debug_partial_step(&gs, "k_temp_update", 1);
    } else {
        cpu.advect_mass_energy(dt);
        g.debug_partial_step(&gs, "k_temp_update", 2);
    }
    g.download(&mut gs);
    eprintln!("dt {dt:.6e}");
    for (name, d, at) in field_diff(&cpu, &gs) {
        eprintln!("   {name:<28} {d:.3e} (case {at})");
    }
    let k = 9.min(cpu.reg.phases.len() - 1);
    let sc = cpu.mass[k].iter().fold(0.0f64, |a, &b| a.max(b));
    let j = cpu.ny / 2;
    let row: Vec<String> = (0..cpu.nx).map(|i| { let c = cpu.idx(i, j); format!("{:+.0}", 1e4 * (gs.mass[k][c] - cpu.mass[k][c]) / sc) }).collect();
    eprintln!("   m{k} j={j} {}", row.join(" "));
    // détail autour d'une cellule (MP_CELL)
    if let Some(c0) = std::env::var("MP_CELL").ok().and_then(|v| v.parse::<usize>().ok()) {
        let nx = cpu.nx;
        let (i0, j0) = (c0 % nx, c0 / nx);
        for (di, dj) in [(0i64, 0i64), (-1, 0), (1, 0), (0, -1), (0, 1), (0, -2), (0, 2)] {
            let (i, jj) = (i0 as i64 + di, j0 as i64 + dj);
            if i < 0 || jj < 0 || i >= nx as i64 || jj >= cpu.ny as i64 { continue; }
            let c = jj as usize * nx + i as usize;
            let np = cpu.mass.len();
            let a0: Vec<String> = (0..np).map(|q| format!("{:.6e}", before.1[q][c])).collect();
            let dm: Vec<String> = (0..np).map(|q| format!("{:+.3e}/{:+.3e}", cpu.mass[q][c] - before.0[q][c], gs.mass[q][c] - before.0[q][c])).collect();
            eprintln!("  ({i},{jj}) α {:?}
      Δm cpu/gpu {:?}", a0, dm);
        }
        let (i, jj) = (i0, j0);
        eprintln!("  u g {:.4e} d {:.4e} ; v b {:.4e} h {:.4e}", before.2[cpu.uidx(i, jj)], before.2[cpu.uidx(i + 1, jj)], before.3[cpu.vidx(i, jj)], before.3[cpu.vidx(i, jj + 1)]);
        // flux de volume par phase du dernier balayage GPU (faces gauche / droite si balayage x)
        let fl = g.debug_read_slice("fl_vol");
        let np = cpu.mass.len();
        for f in [cpu.uidx(i, jj), cpu.uidx(i + 1, jj), cpu.vidx(i, jj), cpu.vidx(i, jj + 1)] {
            let v: Vec<String> = (0..np).map(|k| format!("{:.6e}", fl[f * np + k])).collect();
            eprintln!("  face {f} : {:?}", v);
        }
        let agn = g.debug_read_slice("agn");
        let n = cpu.nx * cpu.ny;
        for dj in [1i64, 0, -1] {
            let mut row = String::new();
            for di in [-1i64, 0, 1] {
                let c = ((j0 as i64 + dj) as usize) * nx + (i0 as i64 + di) as usize;
                // eau (groupe 1) : GPU / attendu depuis α de début (somme normalisée, hors grains)
                let tot: f64 = (0..np).map(|q| before.1[q][c].clamp(0.0, 1.0)).sum();
                let ad = before.1[4][c].clamp(0.0, 1.0) / tot;
                let exp = (before.1[1][c].clamp(0.0, 1.0) / tot) / (1.0 - ad);
                row += &format!("  {:.6}/{:.6}", agn[n + c], exp);
            }
            eprintln!("  agn eau {row}");
        }
    }
}

/// scène de test des extras : cuve d'eau avec émetteur, puits, accélérateur, téléporteur, entrée de bord, solides isotherme et chauffant
fn extras_scene() -> MultiphaseSim {
    use super::mp_cells::{MpEffect, MpInlet, MpThermal};
    use super::multiphase::Bc;
    let mut s = build_scenario(MpScenario::DamBreak, 40, 30, 0.005, 9.81);
    let (nx, ny) = (s.nx, s.ny);
    let water = liquid_with_most_mass(&s);
    let steel = 0;
    let t_amb = s.t_ambient;
    // plaque isotherme chaude et chauffage au fond à droite
    for i in 26..32 {
        let c = s.idx(i, 0);
        s.solid[c] = steel as i32;
        for k in 0..s.reg.phases.len() {
            s.mass[k][c] = 0.0;
        }
        s.e[c] = s.reg.solids[steel].rho_cp() * 330.0;
        s.temp[c] = 330.0;
        s.set_thermal_cell(c, if i < 29 { MpThermal::Isothermal(360.0) } else { MpThermal::Heater(5e7) });
    }
    let e = s.idx(20, 20);
    s.set_effect(e, Some(MpEffect::Emitter { vx: 0.0, vy: -0.4, phase: water, temp: 300.0 }));
    let k = s.idx(35, 25);
    s.set_effect(k, Some(MpEffect::Sink { rate: 0.2 }));
    for j in 2..6 {
        let c = s.idx(15, j);
        s.set_effect(c, Some(MpEffect::Accelerator { ax: 30.0, ay: 0.0 }));
    }
    // bloc de 2×2 sources vers une même destination de rayon 2 (groupe : débits additionnés, fluides mélangés)
    let dst = s.idx(30, 22);
    for (i, j) in [(5, 2), (6, 2), (5, 3), (6, 3)] {
        let src = s.idx(i, j);
        s.set_effect(src, Some(MpEffect::Teleporter { dest: dst, speed: 0.3, radius: 2 }));
    }
    s.inlet[0] = Some(MpInlet { vx: 0.3, vy: 0.0, phase: s.ambient_phase, temp: t_amb + 20.0, offset: 0 });
    s.bc[0] = Bc::Wall;
    s.bc[3] = Bc::Open { p: 101325.0 };
    let _ = (nx, ny);
    s.refresh();
    s
}

#[test]
fn gpu_matches_cpu_special_cells_inlets_and_thermal_solids() {
    let Some((cpu, gs, _)) = compare("cellules spéciales", &extras_scene, 0.08) else { return };
    let (a, b) = (metrics(&cpu), metrics(&gs));
    let m0 = extras_scene();
    for k in 0..a.mass.len() {
        // bilan GPU : M(t) − M(0) = flux des bords + cellules épinglées
        let d = gs.total_mass(k) - m0.total_mass(k) - gs.boundary_flux[k];
        assert!(d.abs() < 1e-4 * m0.total_mass_all().max(1e-9), "bilan GPU phase {k} : {d}");
        eprintln!(" phase {k} : masse CPU {:.6e} GPU {:.6e} ; flux CPU {:.4e} GPU {:.4e}", a.mass[k], b.mass[k], cpu.boundary_flux[k], gs.boundary_flux[k]);
        if a.mass[k] > 1e-6 {
            assert!(rel(a.mass[k], b.mass[k]) < 0.02, "masse {k} : {} {}", a.mass[k], b.mass[k]);
        }
    }
    eprintln!(" chaleur des solides CPU {:.4e} GPU {:.4e}", cpu.energy_input, gs.energy_input);
    assert!(rel(cpu.energy_input, gs.energy_input) < 0.02, "énergie injectée {} {}", cpu.energy_input, gs.energy_input);
    assert!(rel(a.internal, b.internal) < 1e-3);
    assert!(rel(a.kinetic, b.kinetic) < 0.3, "énergie cinétique {} {}", a.kinetic, b.kinetic);
    assert_eq!(cpu.tp_mix.len(), gs.tp_mix.len());
}

#[test]
fn gpu_heats_a_liquid_to_2000_c_without_pausing() {
    // ancien bogue : montée de 2000 °C d'un coup au pinceau sur de l'eau → pause (filet de sécurité). Même scène sur les deux solveurs.
    let mk = || {
        let mut s = build_scenario(MpScenario::DamBreak, 32, 24, 0.005, 9.81);
        // eau à 95 °C : l'ébullition démarre en quelques millisecondes. Partie de 20 °C, la colonne qui s'effondre ne dépasse pas
        // 60 °C en 0.05 s et la seule vapeur vient de cavitations du déferlement, événements chaotiques (1e-12 à 1e-4 kg/m selon
        // l'arrondi) : la comparaison CPU / GPU n'avait plus de sens
        super::mp_view::reset_temperature(&mut s, 368.0);
        for j in 2..8 {
            for i in 2..6 {
                let c = s.idx(i, j);
                s.set_heat_target(c, 2273.15);
            }
        }
        s.heat_power_max = 1e10; // maximum du curseur de la fenêtre : ébullition en quelques centièmes de seconde
        s
    };
    let Some((cpu, gs, _)) = compare("chauffage 2000 °C", &mk, 0.05) else { return };
    let (a, b) = (metrics(&cpu), metrics(&gs));
    eprintln!(" énergie apportée CPU {:.4e} GPU {:.4e} ; transferts {:?} {:?} ; retours arrière {} {}", cpu.energy_input, gs.energy_input, cpu.transferred, gs.transferred, cpu.rollbacks, gs.rollbacks);
    // ébullition chaotique : sur 0.05 s, le CPU contre lui-même avec la T initiale perturbée de 1e-9 (relatif) donne une vapeur NETTE
    // (évaporée − recondensée) de 2e-7 à 5e-5 kg/m, et perturbée de 1e-6 à 1e-4 un apport d'énergie de 8.7e4 à 1.18e5 J/m (26 %) :
    // à long terme, seuls l'ordre de grandeur de l'apport, la conservation de l'énergie et l'absence de pause se comparent ; la
    // physique est validée pas à pas plus bas
    assert!(cpu.transferred[0] != 0.0, "le CPU n'a pas fait bouillir (scène de test à revoir)");
    assert!(gs.energy_input > 0.0 && gs.transferred[0] != 0.0, "le GPU n'a ni chauffé ni fait bouillir");
    assert!(!gs.diverged, "le GPU s'est mis en pause");
    assert!(rel(cpu.energy_input, gs.energy_input) < 0.35, "énergie {} {}", cpu.energy_input, gs.energy_input);
    // U − apport : même énergie initiale, conservée par chacun
    assert!(rel(a.internal - cpu.energy_input, b.internal - gs.energy_input) < 2e-3);
    // (en pleine ébullition : la fermeture f32 bruite la masse d'eau des cellules qui bouillent de ~1e-6 en général, mais une bulle
    // sous-maille peut franchir un seuil d'un côté de l'arrondi seulement : départ au pas 400, cellule 353, vapeur 0.0153 / 0.0182 kg/m³
    // à T identique (375 K), le volume de vapeur en plus chasse 3.5e-3 de l'eau max par advection ; 4.9e-4 au départ du pas 410)
    one_step_agreement("chauffage 2000 °C", mk(), 400, 3, 5e-3, 0.5);
}

#[test]
fn gpu_matches_cpu_semi_lagrangian_and_maccormack_momentum() {
    for mac in [false, true] {
        let mk = || {
            let mut s = build_scenario(MpScenario::DamBreak, 48, 32, 0.008, 9.81);
            s.conservative_momentum = false;
            s.maccormack = mac;
            s
        };
        let Some((cpu, gs, _)) = compare(if mac { "MacCormack" } else { "semi-lagrangien" }, &mk, 0.2) else { return };
        let (a, b) = (metrics(&cpu), metrics(&gs));
        let w = liquid_with_most_mass(&cpu);
        assert!(com_dist(&a, &b, w) < 0.01 * 0.384, "centre de masse {:?} {:?}", a.com[w], b.com[w]);
        assert!(rel(a.kinetic, b.kinetic) < 0.1, "énergie cinétique {} {}", a.kinetic, b.kinetic);
    }
}

#[test]
fn gpu_upload_download_roundtrip_and_render_match_cpu() {
    use super::mp_view::{render_rgba, MpDisplay, RenderOpts};
    let mut s = build_scenario(MpScenario::PremixedFlame, 48, 12, 0.001, 0.0);
    for _ in 0..30 {
        s.step();
    }
    let Some(mut g) = gpu_or_skip(&s) else { return };
    let mut back = build_scenario(MpScenario::PremixedFlame, 48, 12, 0.001, 0.0);
    g.download(&mut back);
    for (name, d, at) in field_diff(&s, &back) {
        // arrondi f32 : T ~ 2000 K → 1e-4 K ; p − p0 ~ 1e3 Pa → 1e-4 Pa ; autres champs relatifs
        let tol = if name == "T" { 1e-3 } else if name.starts_with("p") { 1e-2 } else { 1e-6 };
        assert!(d < tol, "aller-retour {name} : {d} (case {at})");
    }
    let rel_e = s.e.iter().zip(&back.e).map(|(a, b)| (a - b).abs() / a.abs().max(1e3)).fold(0.0f64, f64::max);
    assert!(rel_e < 1e-5, "énergie relue : {rel_e}");
    // rendu GPU = rendu CPU (à l'arrondi des couleurs près)
    for d in MpDisplay::ALL {
        let o = RenderOpts { display: d, t_scale: 500.0, p_scale: 0.0, show_effects: true, solids_white: false };
        let (mut a, mut b) = (vec![], vec![]);
        render_rgba(&s, o, &mut a);
        g.render(&s, &o, &mut b);
        let worst = a.iter().zip(&b).map(|(x, y)| (*x as i32 - *y as i32).abs()).max().unwrap();
        let bad = a.iter().zip(&b).filter(|(x, y)| (**x as i32 - **y as i32).abs() > 3).count();
        eprintln!(" rendu {:?} : écart max {worst}, {bad} composantes > 3", d);
        assert!(bad <= a.len() / 200, "rendu {:?} : {bad} composantes différentes", d);
    }
}

#[test]
fn gpu_rollback_and_divergence_like_cpu() {
    // vitesse limite absurde : chaque pas est jugé invalide → retours arrière (dt/4 …) puis arrêt, état d'avant conservé
    let mk = || {
        let mut s = build_scenario(MpScenario::DamBreak, 24, 16, 0.01, 9.81);
        s.speed_limit = 1e-3;
        s
    };
    let mut cpu = mk();
    let mut gs = mk();
    let Some(mut g) = gpu_or_skip(&gs) else { return };
    // quelques pas sans limite d'abord (l'eau doit se mettre en mouvement)
    cpu.speed_limit = 1000.0;
    gs.speed_limit = 1000.0;
    for _ in 0..10 {
        cpu.step();
        g.step(&mut gs);
    }
    cpu.speed_limit = 1e-3;
    gs.speed_limit = 1e-3;
    let t_before = gs.time;
    let dc = cpu.step();
    let dg = g.step(&mut gs);
    eprintln!(" CPU dt {dc} rollbacks {} diverged {} ; GPU dt {dg} rollbacks {} diverged {}", cpu.rollbacks, cpu.diverged, gs.rollbacks, gs.diverged);
    assert!(cpu.diverged && gs.diverged && dg == 0.0);
    assert_eq!(cpu.rollbacks, gs.rollbacks);
    assert_eq!(gs.time, t_before);
    g.download(&mut gs);
    assert!(gs.state_is_sane() || gs.max_speed() > 1e-3, "état restauré invalide");
}

#[test]
fn gpu_matches_cpu_evaporation_into_humid_air_dalton() {
    // eau chaude sous de l'air sec O2/N2 miscible avec la vapeur : équilibre de pression partielle (Dalton) dans les cellules mixtes et
    // échange à travers l'interface alignée sur la grille, diffusion de la vapeur dans l'air
    use super::mp_materials::MaterialLibrary;
    use super::mp_view::build_scenario_with;
    let mk = || {
        let lib = MaterialLibrary::default().with_combustion();
        let mut s = build_scenario_with(&lib, MpScenario::Sloshing, 32, 24, 0.004, 9.81).unwrap();
        let w = liquid_with_most_mass(&s);
        for c in 0..s.nx * s.ny {
            if s.alpha[w][c] > 0.5 {
                let t = 345.0;
                let e: f64 = (0..s.reg.phases.len()).map(|k| s.mass[k][c] * s.reg.phases[k].e_int(t)).sum();
                s.e[c] = e;
            }
        }
        s.refresh();
        s
    };
    let Some((cpu, gs, g)) = compare("évaporation (Dalton)", &mk, 0.05) else { return };
    eprintln!(" évaporé CPU {:?} GPU {:?} ; {:?}", cpu.transferred, gs.transferred, g.last);
    let q = cpu.reg.changes.iter().position(|pc| pc.liquid == liquid_with_most_mass(&cpu)).unwrap();
    assert!(cpu.transferred[q] > 0.0 && gs.transferred[q] > 0.0, "pas d'évaporation");
    assert!(rel(cpu.transferred[q], gs.transferred[q]) < 0.05, "évaporation {:?} {:?}", cpu.transferred, gs.transferred);
    let (a, b) = (metrics(&cpu), metrics(&gs));
    assert!(rel(a.internal, b.internal) < 1e-3);
}

#[test]
fn gpu_runs_diffusion_flame_and_vented_explosion() {
    for (sc, t_end, nx, ny, dx) in [(MpScenario::DiffusionFlame, 0.02, 32, 32, 0.002), (MpScenario::VentedExplosion, 0.01, 40, 32, 0.003)] {
        let mk = || build_scenario(sc, nx, ny, dx, 9.81);
        one_step_agreement(sc.name(), mk(), 80, 2, 5e-3, 3.0);
        let Some((cpu, gs, _)) = compare(sc.name(), &mk, t_end) else { return };
        eprintln!(" chaleur CPU {:.4e} GPU {:.4e}", cpu.chem_heat, gs.chem_heat);
        assert!(gs.chem_heat > 0.0);
        // jet / panache instables : écart à long terme de quelques dizaines de % admis
        assert!(rel(cpu.chem_heat, gs.chem_heat) < 0.25, "{} : chaleur {} {}", sc.name(), cpu.chem_heat, gs.chem_heat);
    }
}

/// diagnostic : un pas noyau par noyau (MP_SC = scénario, combustion si MP_COMB)
#[test]
#[ignore]
fn gpu_kernel_by_kernel() {
    use super::mp_materials::MaterialLibrary;
    use super::mp_view::build_scenario_with;
    let sc = MpScenario::from_index(std::env::var("MP_SC").ok().and_then(|s| s.parse().ok()).unwrap_or(2));
    let lib = if std::env::var("MP_COMB").is_ok() { MaterialLibrary::default().with_combustion() } else { MaterialLibrary::default() };
    let ev = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let mut s = build_scenario_with(&lib, sc, ev("MP_NX", 32.0) as usize, ev("MP_NY", 24.0) as usize, ev("MP_DX", 0.004), 9.81).unwrap();
    let Some(mut g) = gpu_or_skip(&s) else { return };
    let skip: usize = std::env::var("MP_SKIP").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    for k in 0..skip {
        eprintln!("pas {k}");
        g.step(&mut s);
        if std::env::var("MP_DL").is_ok() {
            g.download(&mut s);
            let bad = (0..s.nx * s.ny).filter(|&c| !s.temp[c].is_finite() || !s.e[c].is_finite()).count();
            eprintln!("   dt {:.3e} T [{:.1}, {:.1}] invalides {bad}", s.last_dt, s.temp.iter().cloned().fold(f64::MAX, f64::min), s.temp.iter().cloned().fold(f64::MIN, f64::max));
        }
    }
    g.debug_step_one_by_one(&s);
}

#[test]
fn gpu_step_batch_equals_single_steps() {
    let mk = || build_scenario(MpScenario::DropSplash, 32, 24, 0.004, 9.81);
    let mut a = mk();
    let mut b = mk();
    let Some(mut ga) = gpu_or_skip(&a) else { return };
    let Some(mut gb) = gpu_or_skip(&b) else { return };
    for _ in 0..3 {
        for _ in 0..4 {
            ga.step(&mut a);
        }
        let t = gb.step_batch(&mut b, 4);
        assert!(t > 0.0);
    }
    ga.download(&mut a);
    gb.download(&mut b);
    assert_eq!(a.step_count, b.step_count);
    assert!((a.time - b.time).abs() < 1e-9 * a.time, "temps {} {}", a.time, b.time);
    for (name, d, at) in field_diff(&a, &b) {
        assert!(d < 1e-6, "série ≠ pas à pas : {name} {d} (case {at})");
    }
}

#[test]
fn gpu_matches_cpu_miscible_liquids_diffusion() {
    // eau / alcool miscibles sous de l'air : groupe miscible d'indice 1 (emplacements de lancement propres au groupe), diffusion de Fick
    let mk = || super::mp_tests::two_liquid_box(true).0;
    one_step_agreement("liquides miscibles", mk(), 20, 2, 1e-5, 0.01);
    let Some((cpu, gs, g)) = compare("liquides miscibles", &mk, 0.5) else { return };
    let (a, b) = (metrics(&cpu), metrics(&gs));
    eprintln!(" sous-pas de diffusion GPU {}", g.last.species_sub);
    for k in 0..3 {
        assert!(rel(a.mass[k], b.mass[k]) < 1e-5, "masse {k}");
        assert!(com_dist(&a, &b, k) < 1e-4, "phase {k} : {:?} {:?}", a.com[k], b.com[k]);
    }
}

#[test]
fn gpu_survives_very_hot_oil_brushed_into_a_closed_tank() {
    // même cas que very_hot_oil_brushed_into_a_closed_tank_does_not_diverge (signalé en mode GPU), avec les séries de pas de la fenêtre
    use super::mp_tests::{closed_scenario, hot_oil_stroke};
    for (sc, t_oil) in [(MpScenario::Empty, 2000.0), (MpScenario::Boiling, 400.0)] {
        let mut s = closed_scenario(sc, 40, 30, 0.02);
        let Some(mut g) = gpu_or_skip(&s) else { return };
        let frames = 40;
        for f in 0..frames {
            if f < 26 {
                g.ensure_host(&mut s);
                hot_oil_stroke(&mut s, f, frames, t_oil);
                g.upload(&s);
            }
            let t1 = s.time + 1.0 / 60.0;
            while s.time < t1 && !s.diverged {
                let k = (((t1 - s.time) / s.last_dt.max(1e-9)).floor() as usize).clamp(1, 8);
                g.step_batch(&mut s, k);
            }
            assert!(!s.diverged, "GPU {} / huile à {t_oil} K : divergence à la trame {f}", sc.name());
        }
        g.download(&mut s);
        assert!(s.state_is_sane(), "GPU {}", sc.name());
    }
    // mêmes équations que le CPU juste après le choc thermique (huile à 2000 K au contact de l'air froid)
    let mut s = closed_scenario(MpScenario::Empty, 32, 24, 0.02);
    for f in 0..3 {
        hot_oil_stroke(&mut s, f, 40, 2000.0);
        let t1 = s.time + 1.0 / 60.0;
        while s.time < t1 {
            s.step();
        }
    }
    one_step_agreement("huile à 2000 K", s, 0, 3, 2e-2, 20.0);
}

/// diagnostic : pas GPU BRUT (sans retour arrière) jusqu'au noyau MP_STOP (défaut k_sanity_final) depuis l'état CPU du test
/// d'ébullition ; affiche les cellules invalides (T, T_g, ε, p, masses) puis les plus grands écarts au pas CPU
#[test]
#[ignore]
fn gpu_raw_step_bad_cells() {
    let mut s = build_scenario(MpScenario::Boiling, 32, 32, 0.003, 9.81);
    for c in 0..s.nx * s.ny {
        let t = if s.is_solid(c) { 700.0 } else if s.temp[c] > 350.0 { 372.0 } else { continue };
        if s.is_solid(c) {
            let rc = s.reg.solids[s.solid[c] as usize].rho_cp();
            s.e[c] = rc * t;
        } else {
            let e: f64 = (0..s.reg.phases.len()).map(|k| s.mass[k][c] * s.reg.phases[k].e_int(t)).sum();
            s.e[c] = e;
        }
    }
    s.refresh();
    let warm = std::env::var("MP_WARM").ok().and_then(|v| v.parse().ok()).unwrap_or(120usize);
    for _ in 0..warm {
        s.step();
    }
    let stop = std::env::var("MP_STOP").unwrap_or("k_sanity_final".into());
    let mut gs = clone_sim(&s);
    let Some(mut g) = gpu_or_skip(&gs) else { return };
    g.debug_partial_step(&gs, &stop, 1);
    g.download(&mut gs);
    let np = gs.mass.len();
    let mut shown = 0;
    for c in 0..gs.nx * gs.ny {
        if gs.is_solid(c) { continue; }
        let tg: Vec<f64> = (0..gs.groups.len()).map(|q| gs.tg[q][c]).collect();
        let bad = !(gs.temp[c] > 50.0 && gs.temp[c] < 5000.0) || tg.iter().any(|t| !(*t > 50.0 && *t < 5000.0)) || !gs.e[c].is_finite()
            || (0..np).any(|k| !(gs.mass[k][c] >= -1e-6));
        if bad && shown < 8 {
            shown += 1;
            let ms: Vec<String> = (0..np).map(|k| format!("{:.3e}", gs.mass[k][c])).collect();
            let tc: Vec<String> = (0..s.groups.len()).map(|q| format!("{:.1}", s.tg[q][c])).collect();
            eprintln!("({}, {}) T {:.1} (cpu avant {:.1}) T_g {:?} (cpu avant {:?}) p {:.0} m {:?}", c % gs.nx, c / gs.nx, gs.temp[c], s.temp[c], tg, tc, gs.p[c], ms);
        }
    }
    let umax = gs.u.iter().chain(gs.v.iter()).fold(0f64, |a, x| a.max(x.abs()));
    eprintln!("cellules invalides affichées {shown} ; |u|max {umax:.3}");
    let mut cpu = s;
    cpu.step();
    let um_c = cpu.u.iter().chain(cpu.v.iter()).fold(0f64, |a, x| a.max(x.abs()));
    eprintln!("CPU : dt {:.3e} |u|max {um_c:.3}", cpu.last_dt);
    let nx = gs.nx;
    let (mut dmax, mut at) = (0f64, 0usize);
    for (i, (a, b)) in gs.v.iter().zip(cpu.v.iter()).enumerate() {
        if (a - b).abs() > dmax { dmax = (a - b).abs(); at = i; }
    }
    eprintln!("v : écart max {dmax:.3} face ({}, {}) gpu {:.3} cpu {:.3}", at % nx, at / nx, gs.v[at], cpu.v[at]);
    for (i, j) in [(0i64, -1i64), (0, 0)] {
        let (ci, cj) = ((at % nx) as i64 + i, (at / nx) as i64 + j);
        if cj < 0 || cj >= gs.ny as i64 { continue; }
        let c = cj as usize * nx + ci as usize;
        let ms: Vec<String> = (0..np).map(|k| format!("{:.3e}/{:.3e}", gs.mass[k][c], cpu.mass[k][c])).collect();
        let tgs: Vec<String> = (0..gs.groups.len()).map(|q| format!("{:.2}/{:.2}", gs.tg[q][c], cpu.tg[q][c])).collect();
        eprintln!("  cellule ({ci}, {cj}) solide {} p {:.0}/{:.0} T {:.2}/{:.2} T_g {:?} m {:?}", gs.solid[c], gs.p[c], cpu.p[c], gs.temp[c], cpu.temp[c], tgs, ms);
    }
}

/// diagnostic : accord d'un pas du test de chauffage à 2000 °C après MP_WARM pas CPU (défaut 200)
#[test]
#[ignore]
fn gpu_heating_one_step_diag() {
    let mut s = build_scenario(MpScenario::DamBreak, 32, 24, 0.005, 9.81);
    super::mp_view::reset_temperature(&mut s, 368.0);
    for j in 2..8 {
        for i in 2..6 {
            let c = s.idx(i, j);
            s.set_heat_target(c, 2273.15);
        }
    }
    s.heat_power_max = 1e10;
    if std::env::var("MP_BOIL").is_ok() {
        // scène du test d'ébullition sur plaque
        s = build_scenario(MpScenario::Boiling, 32, 32, 0.003, 9.81);
        for c in 0..s.nx * s.ny {
            let t = if s.is_solid(c) { 700.0 } else if s.temp[c] > 350.0 { 372.0 } else { continue };
            if s.is_solid(c) {
                let rc = s.reg.solids[s.solid[c] as usize].rho_cp();
                s.e[c] = rc * t;
            } else {
                let e: f64 = (0..s.reg.phases.len()).map(|k| s.mass[k][c] * s.reg.phases[k].e_int(t)).sum();
                s.e[c] = e;
            }
        }
        s.refresh();
    }
    let warm = std::env::var("MP_WARM").ok().and_then(|v| v.parse().ok()).unwrap_or(200usize);
    for _ in 0..warm {
        s.step();
    }
    let e0 = s.energy_input;
    let off = |v: &str| std::env::var(v).is_ok();
    if off("MP_NOCOND") { s.conduction = false; }
    if off("MP_NOPC") { s.phase_change = false; }
    if off("MP_NOST") { s.surface_tension = false; }
    if off("MP_NOVISC") { s.viscous = false; }
    if let Some(l) = std::env::var("MP_LAMBDA").ok().and_then(|v| v.parse::<f64>().ok()) { s.phase_heat_exchange = l; }
    let mut gs = clone_sim(&s);
    let Some(mut g) = gpu_or_skip(&gs) else { return };
    if let Ok(cs) = std::env::var("MP_CELLS") {
        for c in cs.split(',').filter_map(|x| x.parse::<usize>().ok()) {
            let np = s.mass.len();
            let ms: Vec<String> = (0..np).map(|k| format!("{:.4e} a{:.4}", s.mass[k][c], s.alpha[k][c])).collect();
            let tg: Vec<String> = (0..s.tg.len()).map(|q| format!("{:.3}", s.tg[q][c])).collect();
            eprintln!("   avant : cellule {c} p {:.1} T {:.3} T_g {:?} m {:?} chi {:.3e}", s.p[c], s.temp[c], tg, ms, s.thermal_coupling(c, 1.3e-4));
        }
    }
    if let Some(stop) = std::env::var("MP_STOP").ok() {
        // pas GPU partiel jusqu'au noyau MP_STOP (1re occurrence)
        g.debug_partial_step(&gs, &stop, 1);
        g.download(&mut gs);
        if let Ok(cs) = std::env::var("MP_CELLS") {
            for c in cs.split(',').filter_map(|x| x.parse::<usize>().ok()) {
                let tg: Vec<String> = (0..gs.tg.len()).map(|q| format!("{:.3}", gs.tg[q][c])).collect();
                let eg: Vec<String> = (0..gs.eg.len()).map(|q| format!("{:.6e}", gs.eg[q][c])).collect();
                eprintln!("   GPU après {stop} : cellule {c} p {:.1} T_g {:?} E_g {:?} m_air {:.5e}", gs.p[c], tg, eg, gs.mass[0][c]);
            }
        }
        return;
    }
    for k in 0..3 {
        let ec = s.energy_input;
        let eg = gs.energy_input;
        s.step();
        g.step(&mut gs);
        g.download(&mut gs);
        eprintln!("pas {k} : dt {:.3e}/{:.3e} apport CPU {:.4e} GPU {:.4e} ; transferts {:?} {:?}", s.last_dt, gs.last_dt, s.energy_input - ec, gs.energy_input - eg, s.transferred, gs.transferred);
        for (n, v, at) in field_diff(&s, &gs) {
            if v > 0.0 {
                eprintln!("   {n:<28} {v:.3e} (case {at})");
            }
        }
        if let Ok(cs) = std::env::var("MP_CELLS") {
            for c in cs.split(',').filter_map(|x| x.parse::<usize>().ok()) {
                let np = s.mass.len();
                let ms: Vec<String> = (0..np).map(|k| format!("{:.4e}/{:.4e}", s.mass[k][c], gs.mass[k][c])).collect();
                let tg: Vec<String> = (0..s.tg.len()).map(|q| format!("{:.3}/{:.3}", s.tg[q][c], gs.tg[q][c])).collect();
                eprintln!("   cellule {c} ({}, {}) p {:.1}/{:.1} T_g {:?} m {:?}", c % s.nx, c / s.nx, s.p[c], gs.p[c], tg, ms);
            }
        }
    }
    let _ = e0;
}

/// Mesure de vitesse (à lancer sur une vraie carte, en release) :
/// `cargo test --release gpu_speed_benchmark -- --ignored --nocapture`
/// ms par pas : CPU f64 (rayon), GPU `step` (une relecture par pas), GPU `step_batch(8)` (fenêtre) ; lancements de noyaux par pas.
/// `MP_GPU_BATCH_OPS=n` change la taille max des tampons de commandes (contournement lavapipe, défaut 400).
#[test]
#[ignore]
fn gpu_speed_benchmark() {
    use std::time::Instant;
    let scenes = [MpScenario::DamBreak, MpScenario::Boiling, MpScenario::ConfinedExplosion];
    let sizes = [(80usize, 60usize), (160, 120), (320, 240)];
    let mut printed = false;
    for &sc in &scenes {
        for &(nx, ny) in &sizes {
            let dx = 0.01 * 160.0 / nx as f64;
            let mk = || build_scenario(sc, nx, ny, dx, 9.81);
            let Some(mut g) = gpu_or_skip(&mk()) else { return };
            if !printed {
                eprintln!("adaptateur : {}", g.adapter_name);
                eprintln!("{:<22} {:>9} {:>11} {:>11} {:>11} {:>9} {:>7}", "scène", "grille", "CPU ms/pas", "GPU step", "GPU ×8", "noyaux", "PCG");
                printed = true;
            }
            // CPU
            let mut c = mk();
            for _ in 0..3 { c.step(); }
            let n_cpu = if nx >= 320 { 5 } else { 15 };
            let t0 = Instant::now();
            for _ in 0..n_cpu { c.step(); }
            let cpu_ms = t0.elapsed().as_secs_f64() * 1e3 / n_cpu as f64;
            // GPU pas à pas
            let mut s = mk();
            for _ in 0..5 { g.step(&mut s); }
            let n = 40;
            let t0 = Instant::now();
            let mut disp = 0;
            for _ in 0..n { g.step(&mut s); disp += g.last.dispatches; }
            let gpu_ms = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
            let pcg = g.last.pcg;
            // GPU par séries de 8
            let mut s2 = mk();
            let mut g2 = gpu_or_skip(&s2).unwrap();
            for _ in 0..2 { g2.step_batch(&mut s2, 8); }
            let s0 = s2.step_count;
            let t0 = Instant::now();
            for _ in 0..10 { g2.step_batch(&mut s2, 8); }
            let nb = (s2.step_count - s0).max(1);
            let batch_ms = t0.elapsed().as_secs_f64() * 1e3 / nb as f64;
            eprintln!("{:<22} {:>9} {:>11.2} {:>11.2} {:>11.2} {:>9} {:>7}{}", sc.name(), format!("{nx}×{ny}"), cpu_ms, gpu_ms, batch_ms,
                disp / n, pcg, if s.diverged || s2.diverged { "  DIVERGÉ" } else { "" });
        }
    }
}

/// Diagnostic (huile à 2000 K qui atteint l'eau chaude, cuve fermée 80×60) : rejoue la trajectoire GPU jusqu'à la pause, garde l'état
/// juste avant le pas refusé, puis rejoue ce pas sur le CPU à dt, dt/4 … dt/1024 avec la sonde par étape sur la cellule la plus mauvaise.
/// `cargo test --release gpu_hot_oil_failing_step_probe -- --ignored --nocapture` (MP_DX : maille, défaut 0.01)
#[test]
#[ignore]
fn gpu_hot_oil_failing_step_probe() {
    use super::mp_step::PROBE_CELL;
    use super::mp_tests::{closed_scenario, hot_oil_stroke};
    use std::sync::atomic::Ordering;
    let dx = std::env::var("MP_DX").ok().and_then(|v| v.parse().ok()).unwrap_or(0.01);
    let mut s = closed_scenario(MpScenario::Boiling, 80, 60, dx);
    let on_cpu = std::env::var("MP_CPU").is_ok();
    let mut g = if on_cpu { None } else { match gpu_or_skip(&s) { Some(g) => Some(g), None => return } };
    let frames = 40;
    let mut start: Option<MultiphaseSim> = None;
    for f in 0..frames {
        if let Some(g) = g.as_mut() { g.ensure_host(&mut s); }
        if f < 26 {
            hot_oil_stroke(&mut s, f, frames, 2000.0);
        }
        let t1 = s.time + 1.0 / 60.0;
        if let Some(g) = g.as_mut() {
            let frame_start = clone_sim(&s);
            g.upload(&s);
            while s.time < t1 && !s.diverged {
                let k = (((t1 - s.time) / s.last_dt.max(1e-9)).floor() as usize).clamp(1, 8);
                g.step_batch(&mut s, k);
            }
            if s.diverged {
                eprintln!("pause GPU à la trame {f} ; rejoue la trame pas à pas");
                let mut r = clone_sim(&frame_start);
                let mut st = clone_sim(&frame_start);
                g.upload(&r);
                loop {
                    g.download(&mut r);
                    let prev = clone_sim(&r);
                    g.step(&mut r);
                    if r.diverged { st = prev; break; }
                    if r.time >= t1 { eprintln!("  (pas à pas : pas de pause, on prend le début de trame)"); break; }
                }
                start = Some(st);
                break;
            }
        } else {
            while s.time < t1 && !s.diverged {
                let prev = clone_sim(&s);
                s.step();
                if s.diverged {
                    eprintln!("pause CPU à la trame {f}");
                    start = Some(prev);
                }
            }
            if start.is_some() { break; }
        }
    }
    let Some(st) = start else { eprintln!("aucune pause sur 40 trames"); return };
    // cellule la plus mauvaise après un pas brut CPU
    let mut c0 = clone_sim(&st);
    let dt0 = c0.choose_dt();
    c0.step_raw_pub();
    let worst = worst_cell(&c0);
    eprintln!("dt0 {dt0:.3e} ; pire cellule ({}, {}) : {} ; état sain {} ; |u|max {:.3e} ; Courant {:.2}", worst.0 % c0.nx, worst.0 / c0.nx, worst.1,
        c0.state_is_sane(), c0.max_speed(), c0.max_speed() * dt0 / c0.dx);
    for (gr, members) in st.groups.iter().enumerate() {
        eprintln!("  g{gr} = {}", members.iter().map(|&k| st.reg.phases[k].name.as_str()).collect::<Vec<_>>().join(" + "));
    }
    PROBE_CELL.store(worst.0, Ordering::Relaxed);
    for k in 0..6 {
        let mut c = clone_sim(&st);
        let sc = 0.25f64.powi(k);
        c.dt_max = c.dt_max.min(dt0 * sc);
        c.closure_relax_scale = sc;
        eprintln!("== dt = dt0 × {sc:e}");
        let dt = c.step_raw_pub();
        let w = worst_cell(&c);
        eprintln!("   dt {dt:.3e} ; sain {} ; |u|max {:.3e} ; Courant {:.2} ; pire ({}, {}) {}", c.state_is_sane(), c.max_speed(), c.max_speed() * dt / c.dx, w.0 % c.nx, w.0 / c.nx, w.1);
    }
    PROBE_CELL.store(usize::MAX, Ordering::Relaxed);
}

/// Mesure (huile à 2000 K qui atteint l'eau chaude, cuve fermée 80×60, 40 trames) : p max, vitesse max, pas refaits, trame de pause,
/// CPU puis GPU. `cargo test --release hot_oil_on_hot_water_violence -- --ignored --nocapture`
#[test]
#[ignore]
fn hot_oil_on_hot_water_violence() {
    use super::mp_tests::{closed_scenario, hot_oil_stroke};
    let frames = 40;
    for gpu in [false, true] {
        let mut s = closed_scenario(MpScenario::Boiling, 80, 60, 0.01);
        let mut g = if gpu { match gpu_or_skip(&s) { Some(g) => Some(g), None => continue } } else { None };
        let (mut pmax, mut umax, mut pause) = (0.0f64, 0.0f64, None);
        let mut redo_gpu = 0;
        let clk = std::time::Instant::now();
        for f in 0..frames {
            if let Some(g) = g.as_mut() { g.ensure_host(&mut s); }
            if f < 26 { hot_oil_stroke(&mut s, f, frames, 2000.0); }
            if let Some(g) = g.as_mut() { g.upload(&s); }
            let t1 = s.time + 1.0 / 60.0;
            while s.time < t1 && !s.diverged {
                match g.as_mut() {
                    Some(g) => {
                        let k = (((t1 - s.time) / s.last_dt.max(1e-9)).floor() as usize).clamp(1, 8);
                        g.step_batch(&mut s, k);
                        redo_gpu += g.last.retries;
                    }
                    None => { s.step(); }
                }
            }
            if let Some(g) = g.as_mut() { g.download(&mut s); }
            pmax = pmax.max(s.p.iter().cloned().fold(0.0, f64::max));
            umax = umax.max(s.max_speed());
            if s.diverged { pause = Some(f); break; }
        }
        eprintln!("{} : p max {:.3e} Pa, |u| max {:.1} m/s, pas refaits {}, pas {}, pause {:?}, {:.1} s", if gpu { "GPU" } else { "CPU" },
            pmax, umax, if gpu { redo_gpu } else { s.rollbacks }, s.step_count, pause, clk.elapsed().as_secs_f64());
    }
}

/// cellule qui fait échouer `state_is_sane` (premier critère rencontré : T / p / masse, sinon la face la plus rapide)
fn worst_cell(s: &MultiphaseSim) -> (usize, String) {
    let n = s.nx * s.ny;
    let mut tmin = (f64::INFINITY, 0usize, String::new());
    for c in 0..n {
        if s.solid[c] >= 0 { continue; }
        if !s.temp[c].is_finite() || !s.p[c].is_finite() { return (c, format!("non fini T {} p {}", s.temp[c], s.p[c])); }
        if s.p[c] < 0.0 { return (c, format!("p < 0 : {:.3e}", s.p[c])); }
        for k in 0..s.mass.len() {
            if s.mass[k][c] < -1e-6 { return (c, format!("masse {} < 0 : {:.3e}", s.reg.phases[k].name, s.mass[k][c])); }
        }
        for g in 0..s.ngroups() {
            if s.group_mass(g, c) > 0.0 {
                let t = if s.tg[g][c].is_finite() { s.tg[g][c] } else { -1e30 };
                if t < tmin.0 { tmin = (t, c, format!("T g{g} = {t:.1}")); }
                if t > 5000.0 { return (c, format!("T g{g} = {t:.1} > 5000")); }
            }
        }
    }
    if tmin.0 < 50.0 { return (tmin.1, tmin.2); }
    // face la plus rapide
    let (mut best, mut at) = (0.0f64, 0usize);
    for j in 0..s.ny {
        for i in 0..s.nx {
            let c = s.idx(i, j);
            let v = [s.u[s.uidx(i, j)], s.u[s.uidx(i + 1, j)], s.v[s.vidx(i, j)], s.v[s.vidx(i, j + 1)]].iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            if v > best { best = v; at = c; }
        }
    }
    (at, format!("vitesse {best:.3e} m/s (T min {:.1})", tmin.0))
}

/// Diagnostic : vapeur à l'impact de la rupture de barrage (grille de la fenêtre 74×120, dx = 1 cm, CPU)
#[test]
#[ignore]
fn dam_break_vapor_cpu() {
    let mut s = build_scenario(MpScenario::DamBreak, 74, 120, 0.01, 9.81);
    let vap = s.reg.phases.iter().position(|p| p.name == "vapeur").unwrap();
    let eau = s.reg.phases.iter().position(|p| p.name == "eau").unwrap();
    let mut next = 0.05;
    let mut first = true;
    while s.time < 0.8 && !s.diverged {
        s.step();
        let mv: f64 = s.mass[vap].iter().sum::<f64>() * s.dx * s.dx;
        if mv > 0.0 && first {
            first = false;
            let c = (0..s.nx * s.ny).max_by(|&a, &b| s.mass[vap][a].partial_cmp(&s.mass[vap][b]).unwrap()).unwrap();
            eprintln!("PREMIÈRE VAPEUR t={:.4} pas {} case ({},{}) m_v {:.3e} m_eau {:.3e} α_eau {:.3} T {:.2} p {:.0} bloquée {}", s.time, s.step_count, c % s.nx, c / s.nx, s.mass[vap][c], s.mass[eau][c], s.alpha[eau][c], s.temp[c], s.p[c], s.pressure_clamped[c]);
        }
        if s.time >= next {
            let pmin = (0..s.nx * s.ny).map(|c| s.p[c]).fold(f64::MAX, f64::min);
            let tmin = (0..s.nx * s.ny).map(|c| s.temp[c]).fold(f64::MAX, f64::min);
            let nv = s.mass[vap].iter().filter(|&&m| m > 1e-9).count();
            eprintln!("t={:.3} pas {} m_v {:.3e} cellules vapeur {} p min {:.0} T min {:.2} umax {:.2} retours {}", s.time, s.step_count, mv, nv, pmin, tmin, s.max_speed(), s.rollbacks);
            next += 0.05;
        }
    }
}

/// boîte fermée de liquide (eau) contenant une suspension uniforme de sable (fraction a0, grains de diamètre d)
fn sand_suspension(nx: usize, ny: usize, dx: f64, a0: f64, d: f64) -> MultiphaseSim {
    use super::phases::{Phase, Registry};
    let mut r = Registry::new();
    let eau = r.add_phase(Phase::water());
    let mut sand = Phase::sand();
    sand.grain_diameter = d;
    let sable = r.add_phase(sand);
    let mut s = MultiphaseSim::new(nx, ny, dx, r, eau);
    s.surface_tension = false;
    let mut mix = vec![0.0; 2];
    mix[eau] = 1.0 - a0;
    mix[sable] = a0;
    s.paint_mixture(move |_, _| Some(mix.clone()));
    s.finalize_init(|_, _| 293.15);
    s
}

/// même suspension sous une couche d'air (`air_rows` rangées du haut) : récipient ouvert, plus proche des scènes réelles
fn sand_suspension_under_air(nx: usize, ny: usize, dx: f64, a0: f64, d: f64, air_rows: usize) -> MultiphaseSim {
    use super::phases::{Phase, Registry};
    let mut r = Registry::new();
    let air = r.add_phase(Phase::air());
    let eau = r.add_phase(Phase::water());
    let mut sand = Phase::sand();
    sand.grain_diameter = d;
    let sable = r.add_phase(sand);
    let mut s = MultiphaseSim::new(nx, ny, dx, r, air);
    s.surface_tension = false;
    let h = (ny - air_rows) as f64 * dx;
    s.paint_mixture(move |_, y| {
        let mut mix = vec![0.0; 3];
        if y < h {
            mix[eau] = 1.0 - a0;
            mix[sable] = a0;
        } else {
            mix[air] = 1.0;
        }
        Some(mix)
    });
    s.finalize_init(|_, _| 293.15);
    s
}

#[test]
fn gpu_matches_cpu_sand_cloud_falling_into_water() {
    // phase granulaire : transport dispersé (à travers la surface de l'eau), sédimentation, borne de pas — accord d'un pas, puis 1 s
    let mk = || build_scenario(MpScenario::Sedimentation, 40, 30, 0.01, 9.81);
    assert!(super::gpu_multiphase::gpu_unsupported_reason(&mk()).is_none());
    one_step_agreement("nuage de sable", mk(), 40, 3, 1e-4, 0.02);
    let Some((cpu, gs, _)) = compare("nuage de sable", &mk, 1.0) else { return };
    let (a, b) = (metrics(&cpu), metrics(&gs));
    let sable = cpu.reg.phases.iter().position(|p| p.is_granular()).unwrap();
    let below = |s: &MultiphaseSim| {
        let surface = 0.4 * s.ny as f64;
        (0..s.nx * s.ny).filter(|&c| ((c / s.nx) as f64) < surface).map(|c| s.mass[sable][c]).sum::<f64>() * s.dx * s.dx / s.total_mass(sable)
    };
    eprintln!(" sable sous la surface : CPU {:.3} GPU {:.3}", below(&cpu), below(&gs));
    for k in 0..a.mass.len() {
        assert!(rel(a.mass[k], b.mass[k]) < 1e-4, "masse {k} : {} vs {}", a.mass[k], b.mass[k]);
    }
    assert!(com_dist(&a, &b, sable) < 0.01, "centre de masse du sable : CPU {:?} GPU {:?}", a.com[sable], b.com[sable]);
    assert!((below(&cpu) - below(&gs)).abs() < 0.05);
}

#[test]
fn gpu_forms_the_same_packed_sand_layer_as_the_cpu() {
    // grains de 0.5 mm à 10 % dans 6 cm d'eau sous 1 cm d'air : en 3 s tout est au fond, couche tassée à α_max (miroir du test CPU).
    // (Récipient ouvert : une boîte FERMÉE entièrement liquide est instable sur le GPU f32, grains ou non — voir TODO.)
    let (ny, a0) = (70, 0.1);
    let mk = || sand_suspension_under_air(6, ny, 1e-3, a0, 5e-4, 10);
    one_step_agreement("suspension de sable", mk(), 20, 3, 1e-4, 0.01);
    let Some((cpu, gs, _)) = compare("couche tassée", &mk, 3.0) else { return };
    let sable = 2;
    let amax = gs.reg.phases[sable].alpha_max;
    let col = |s: &MultiphaseSim| (0..ny).map(|j| s.alpha[sable][s.idx(3, j)]).collect::<Vec<f64>>();
    let (cc, cg) = (col(&cpu), col(&gs));
    let show = |c: &[f64]| c[..14].iter().map(|a| (a * 1000.0).round() / 1000.0).collect::<Vec<_>>();
    eprintln!(" colonne CPU {:?}
 colonne GPU {:?}", show(&cc), show(&cg));
    let bottom: f64 = cg[..8].iter().sum::<f64>() / 8.0;
    let top = cg[14..].iter().fold(0.0f64, |a, &b| a.max(b));
    assert!(bottom > 0.95 * amax && bottom <= amax + 1e-4, "fond {bottom}");
    assert!(top < 1e-2, "grains restés en suspension : {top}");
    for k in 0..3 {
        assert!(rel(cpu.total_mass(k), gs.total_mass(k)) < 1e-5, "masse {k}");
    }
    for j in 0..ny {
        assert!((cc[j] - cg[j]).abs() < 0.02, "rangée {j} : CPU {} GPU {}", cc[j], cg[j]);
    }
}

#[test]
fn gpu_collapses_a_sand_column_like_the_cpu() {
    // frottement des grains (Drucker–Prager implicite, gradient conjugué) : accord d'un pas pendant l'effondrement, puis tas final
    use super::mp_friction::tests::{flank_angle, pile_heights, sand_column};
    let mk = || sand_column(60, 30, 0.005, 0.04, 0.08, 32.0);
    one_step_agreement("colonne de sable", mk(), 60, 3, 2e-4, 0.01);
    let Some((cpu, gs, g)) = compare("colonne de sable", &mk, 1.5) else { return };
    let (hc, hg) = (pile_heights(&cpu), pile_heights(&gs));
    let (hmc, hmg) = (hc.iter().fold(0.0f64, |a, &b| a.max(b)), hg.iter().fold(0.0f64, |a, &b| a.max(b)));
    eprintln!(" angle des flancs : CPU {:.1}° GPU {:.1}° ; hauteur CPU {hmc:.4} GPU {hmg:.4} ; itérations du frottement GPU {} (CPU {})",
        flank_angle(&cpu), flank_angle(&gs), gs.last_friction_iters, cpu.last_friction_iters);
    let _ = g;
    assert!((flank_angle(&cpu) - flank_angle(&gs)).abs() < 3.0, "angles CPU {} GPU {}", flank_angle(&cpu), flank_angle(&gs));
    assert!((hmc - hmg).abs() < 0.1 * hmc, "hauteurs CPU {hmc} GPU {hmg}");
    assert!(rel(cpu.total_mass(1), gs.total_mass(1)) < 1e-5, "masse de sable");
}

#[test]
fn gpu_keeps_sand_on_a_solid_shelf_like_the_cpu() {
    // frottement contre un obstacle solide (nœuds de paroi des solides) et chaleur de frottement
    use super::mp_friction::tests::{sand_on_shelf, sand_on_shelf_fraction};
    one_step_agreement("sable sur étagère", sand_on_shelf(), 40, 3, 2e-4, 0.01);
    let Some((cpu, gs, _)) = compare("sable sur étagère", &sand_on_shelf, 1.0) else { return };
    let (fc, fg) = (sand_on_shelf_fraction(&cpu), sand_on_shelf_fraction(&gs));
    let gsand = cpu.group_of[1];
    let tmax = |s: &MultiphaseSim| (0..s.nx * s.ny).filter(|&c| s.mass[1][c] > 1.0).map(|c| s.tg[gsand][c]).fold(0.0f64, f64::max);
    eprintln!(" sable sur l'étagère : CPU {:.3} GPU {:.3} ; T max des grains CPU {:.4} GPU {:.4} K", fc, fg, tmax(&cpu), tmax(&gs));
    assert!(fg > 0.85 && (fc - fg).abs() < 0.03, "CPU {fc} GPU {fg}");
    assert!(tmax(&gs) > 293.15 && (tmax(&cpu) - tmax(&gs)).abs() < 0.3 * (tmax(&cpu) - 293.15).max(1e-3), "échauffement CPU {} GPU {}", tmax(&cpu), tmax(&gs));
}

#[test]
fn gpu_decays_a_very_viscous_vortex_like_the_cpu() {
    // viscosité implicite (au-delà de la borne explicite) : pas de temps libres, décroissance de l'énergie cinétique
    use super::mp_friction::tests::viscous_vortex;
    let mk = || viscous_vortex(true);
    one_step_agreement("tourbillon visqueux", mk(), 5, 3, 1e-5, 0.01);
    let Some((cpu, gs, _)) = compare("tourbillon visqueux", &mk, 0.03) else { return };
    let k0 = mk().kinetic_energy();
    eprintln!(" énergie cinétique relative : CPU {:.4} GPU {:.4} ; pas CPU {} GPU {}", cpu.kinetic_energy() / k0, gs.kinetic_energy() / k0, cpu.step_count, gs.step_count);
    assert!(rel(cpu.kinetic_energy(), gs.kinetic_energy()) < 0.02);
    assert!(gs.step_count < 50, "pas GPU {}", gs.step_count);
}

#[test]
fn gpu_burns_coal_like_the_cpu() {
    // braséro : réactions de surface des braises (C + O2 → CO, CO2, limitées par la diffusion de l'oxygène), pyrolyse de la houille et
    // flamme de ses matières volatiles. Accord du PREMIER pas (pyrolyse, braises, percolation, rayonnement du lit) : dès le suivant, les
    // gaz de la houille diffusent dans la flamme d'allumage et y brûlent, et dans une maille de flamme la chimie raide intégrée en f32
    // et en f64 ne s'accorde pas pas à pas (voir le feu de bois). Puis comparaison sur la durée, en débits (le GPU s'arrête au pas qui
    // dépasse t_end)
    use super::mp_materials::{CH4, CO, CO2, HOUILLE, O2};
    let lib = super::mp_materials::MaterialLibrary::default();
    let mk = || super::mp_view::build_scenario_with(&lib, MpScenario::CoalFire, 40, 40, 0.01, 9.81).unwrap();
    one_step_agreement("feu de charbon", mk(), 0, 1, 1e-3, 0.5);
    let Some((cpu, gs, _)) = compare("feu de charbon", &mk, 1.0) else { return };
    let pos = |n: &str| cpu.reg.phases.iter().position(|p| p.name == n).unwrap();
    let (co2, co, ch4, hou, o2) = (pos(CO2), pos(CO), pos(CH4), pos(HOUILLE), pos(O2));
    let h0 = mk().total_mass(hou);
    // houille dégazée (kg/(m·s)) ; carbone passé dans le gaz (CO + CO2 + CH4, sorties comprises) et carbone brûlé en CO2 (mol/(m·s))
    let pyro = |s: &MultiphaseSim| (h0 - s.total_mass(hou)) / s.time;
    let gas = |s: &MultiphaseSim, q: usize, m: f64| (s.total_mass(q) - s.boundary_flux[q]) / m / s.time;
    let made = |s: &MultiphaseSim| gas(s, co2, 0.044009) + gas(s, co, 0.02801) + gas(s, ch4, 0.016043);
    let gg = cpu.group_of[o2];
    let tmax = |s: &MultiphaseSim| (0..s.nx * s.ny).filter(|&q| !s.is_solid(q) && s.group_mass(gg, q) > 0.0).map(|q| s.tg[gg][q]).fold(0.0f64, f64::max);
    eprintln!(" houille dégazée CPU {:.4e} GPU {:.4e} kg/(m·s) ; carbone passé dans le gaz CPU {:.4e} GPU {:.4e}, brûlé en CO2 CPU {:.4e} GPU {:.4e} mol/(m·s) ; T gaz max CPU {:.0} GPU {:.0} K",
        pyro(&cpu), pyro(&gs), made(&cpu), made(&gs), gas(&cpu, co2, 0.044009), gas(&gs, co2, 0.044009), tmax(&cpu), tmax(&gs));
    assert!(pyro(&cpu) > 0.0 && rel(pyro(&cpu), pyro(&gs)) < 0.05, "houille CPU {} GPU {}", pyro(&cpu), pyro(&gs));
    assert!(made(&cpu) > 0.0 && rel(made(&cpu), made(&gs)) < 0.1, "carbone CPU {} GPU {}", made(&cpu), made(&gs));
}

#[test]
fn gpu_pyrolyses_wood_like_the_cpu() {
    // réaction EN VOLUME (pyrolyse) à la température des grains, bornée par la conduction interne des morceaux (2 cm : τ ≈ 66 s) :
    // bois dans de l'azote à 800 K
    let mk = || super::mp_chem_tests::wood_dust(800.0, 0.05);
    one_step_agreement("pyrolyse", mk(), 2, 3, 1e-4, 1e-3);
    let Some((cpu, gs, _)) = compare("pyrolyse", &mk, 0.5) else { return };
    let pos = |n: &str| cpu.reg.phases.iter().position(|p| p.name == n).unwrap();
    let (w, ch4, ch) = (pos(super::mp_materials::WOOD), pos(super::mp_materials::CH4), pos(super::mp_materials::CHARCOAL));
    eprintln!(" bois décomposé : CPU {:.4e} GPU {:.4e} kg/m ; CH4 CPU {:.4e} GPU {:.4e} ; charbon de bois CPU {:.4e} GPU {:.4e}",
        mk().total_mass(w) - cpu.total_mass(w), mk().total_mass(w) - gs.total_mass(w), cpu.total_mass(ch4), gs.total_mass(ch4),
        cpu.total_mass(ch), gs.total_mass(ch));
    let tw = |s: &MultiphaseSim| (0..s.nx * s.ny).map(|q| s.tg[s.group_of[w]][q]).sum::<f64>() / (s.nx * s.ny) as f64;
    let tgas = |s: &MultiphaseSim| (0..s.nx * s.ny).map(|q| s.tg[s.group_of[ch4]][q]).sum::<f64>() / (s.nx * s.ny) as f64;
    eprintln!(" temps CPU {:.5} GPU {:.5} s ; pas CPU {} GPU {} ; T bois CPU {:.3} GPU {:.3} K ; T gaz CPU {:.3} GPU {:.3} K", cpu.time, gs.time, cpu.step_count, gs.step_count,
        tw(&cpu), tw(&gs), tgas(&cpu), tgas(&gs));
    // (le GPU s'arrête au pas qui dépasse t_end : comparer les débits, pas les quantités)
    let (rc, rg) = (cpu.total_mass(ch4) / cpu.time, gs.total_mass(ch4) / gs.time);
    assert!(rc > 0.0 && rel(rc, rg) < 0.01, "CH4 par seconde CPU {rc} GPU {rg}");
    assert!(rel(cpu.total_mass(ch) / cpu.time, gs.total_mass(ch) / gs.time) < 0.01);
}

#[test]
fn gpu_burns_gunpowder_like_the_cpu() {
    // combustion de la poudre noire (loi de Vieille dès 580 K, produits à la température adiabatique, borne du pas) dans un vase clos
    // d'azote : accord pas à pas, puis débit de combustion et montée en pression sur la durée
    use super::mp_materials::{FUMEE, POUDRE};
    let mk = || {
        let mut s = super::mp_chem_tests::gunpowder_dust(700.0, 10.0 / 1800.0, true);
        s.radiation = false;
        s
    };
    one_step_agreement("poudre noire", mk(), 2, 3, 1e-4, 0.05);
    let Some((cpu, gs, _)) = compare("poudre noire", &mk, 0.05) else { return };
    let pos = |n: &str| cpu.reg.phases.iter().position(|p| p.name == n).unwrap();
    let (w, f) = (pos(POUDRE), pos(FUMEE));
    let w0 = mk().total_mass(w);
    let burnt = |s: &MultiphaseSim| (w0 - s.total_mass(w)) / s.time;
    let p = |s: &MultiphaseSim| s.p.iter().sum::<f64>() / s.p.len() as f64;
    eprintln!(" poudre brûlée CPU {:.5e} GPU {:.5e} kg/(m·s) ; fumée CPU {:.4e} GPU {:.4e} ; p CPU {:.4} GPU {:.4} bar ; temps {:.4} / {:.4} s, pas {} / {}",
        burnt(&cpu), burnt(&gs), cpu.total_mass(f), gs.total_mass(f), p(&cpu) / 1e5, p(&gs) / 1e5, cpu.time, gs.time, cpu.step_count, gs.step_count);
    assert!(burnt(&cpu) > 0.0 && rel(burnt(&cpu), burnt(&gs)) < 0.01, "poudre brûlée CPU {} GPU {}", burnt(&cpu), burnt(&gs));
    assert!(gs.step_count == cpu.step_count || rel(p(&cpu), p(&gs)) < 0.05, "p CPU {} GPU {}", p(&cpu), p(&gs));
}

#[test]
fn gpu_runs_the_gunpowder_heap_like_the_cpu() {
    // tas de poudre noire : allumage au contact des gaz chauds, souffle, fumée — comparaison sur 10 ms (poudre brûlée et fumée à 15 %
    // près). Pas d'accord pas à pas : dès le premier pas, le CO et le H2 de la poudre brûlent avec l'air des pores vers 2200 K, chimie
    // raide qui ne s'accorde pas entre f32 et f64 (voir le feu de bois) ; la combustion de la poudre elle-même s'accorde à 1e-7 près
    // (vase clos, `gpu_burns_gunpowder_like_the_cpu`)
    use super::mp_materials::{FUMEE, POUDRE};
    let lib = super::mp_materials::MaterialLibrary::default();
    let mk = || super::mp_view::build_scenario_with(&lib, MpScenario::Gunpowder, 40, 30, 0.02, 9.81).unwrap();
    let Some((cpu, gs, _)) = compare("tas de poudre", &mk, 0.01) else { return };
    let pos = |n: &str| cpu.reg.phases.iter().position(|p| p.name == n).unwrap();
    let (w, f) = (pos(POUDRE), pos(FUMEE));
    let w0 = mk().total_mass(w);
    // poudre brûlée par seconde (celle qui a disparu moins celle que le souffle a fait sortir du domaine) et fumée produite
    let burnt = |s: &MultiphaseSim| (w0 - (s.total_mass(w) - s.boundary_flux[w])) / s.time;
    let smoke = |s: &MultiphaseSim| (s.total_mass(f) - s.boundary_flux[f]) / s.time;
    eprintln!(" poudre brûlée CPU {:.4e} GPU {:.4e} kg/(m·s) ; fumée CPU {:.4e} GPU {:.4e} kg/(m·s) ; temps {:.4} / {:.4} s, pas {} / {}",
        burnt(&cpu), burnt(&gs), smoke(&cpu), smoke(&gs), cpu.time, gs.time, cpu.step_count, gs.step_count);
    assert!(burnt(&cpu) > 0.0 && rel(burnt(&cpu), burnt(&gs)) < 0.15, "poudre brûlée CPU {} GPU {}", burnt(&cpu), burnt(&gs));
    assert!(rel(smoke(&cpu), smoke(&gs)) < 0.15, "fumée CPU {} GPU {}", smoke(&cpu), smoke(&gs));
}

#[test]
fn gpu_radiates_in_the_grain_bed_like_the_cpu() {
    // rayonnement du tas : entre mailles (braises sous du bois froid) et entre les populations mêlées d'une maille (miroirs de
    // conduct et de exchange_grains)
    for mixed in [false, true] {
        let name = if mixed { "braises et bois mêlés" } else { "braises sous le bois" };
        let mk = || super::mp_chem_tests::ember_bed(mixed);
        one_step_agreement(name, mk(), 2, 3, 1e-4, 1e-2);
        let Some((cpu, gs, _)) = compare(name, &mk, 0.05) else { return };
        let w = cpu.reg.phases.iter().position(|p| p.name == super::mp_materials::WOOD).unwrap();
        let gw = cpu.group_of[w];
        // échauffement moyen du bois par seconde (le GPU s'arrête au pas qui dépasse t_end : comparer des débits)
        let heat = |s: &MultiphaseSim| {
            let (mut a, mut m) = (0.0, 0.0);
            for q in 0..s.nx * s.ny {
                a += s.mass[w][q] * (s.tg[gw][q] - 293.15);
                m += s.mass[w][q];
            }
            a / m / s.time
        };
        eprintln!(" {name} : bois chauffé de {:.4} K/s (CPU), {:.4} K/s (GPU)", heat(&cpu), heat(&gs));
        assert!(heat(&cpu) > 1.0 && rel(heat(&cpu), heat(&gs)) < 0.01, "{name} : CPU {} GPU {} K/s", heat(&cpu), heat(&gs));
    }
}

#[test]
fn gpu_runs_the_wood_fire_like_the_cpu() {
    // feu de bois : pyrolyse, flamme de diffusion des gaz du bois, braises de charbon de bois, percolation. Comparaison sur la durée
    // (comme le braséro) : dans une maille de flamme, la chimie raide intégrée en f32 sur le GPU et en f64 sur le CPU ne s'accorde pas
    // pas à pas (méthane brûlé en un pas d'un côté, pas tout à fait de l'autre), mais le feu, oui
    use super::mp_materials::{CHARCOAL, CO, CO2, O2, WOOD};
    let lib = super::mp_materials::MaterialLibrary::default();
    let mk = || super::mp_view::build_scenario_with(&lib, MpScenario::WoodFire, 24, 24, 0.4 / 24.0, 9.81).unwrap();
    let Some((cpu, gs, _)) = compare("feu de bois", &mk, 0.4) else { return };
    let pos = |n: &str| cpu.reg.phases.iter().position(|p| p.name == n).unwrap();
    let (w, ch, co, co2, o2) = (pos(WOOD), pos(CHARCOAL), pos(CO), pos(CO2), pos(O2));
    let w0 = mk().total_mass(w);
    let pyro = |s: &MultiphaseSim| (w0 - s.total_mass(w)) / s.time;
    // carbone brûlé en CO2 (sorties comprises) par seconde
    let burnt = |s: &MultiphaseSim| (s.total_mass(co2) - s.boundary_flux[co2]) / 0.044009 / s.time;
    let gg = cpu.group_of[o2];
    let tmax = |s: &MultiphaseSim| (0..s.nx * s.ny).filter(|&q| !s.is_solid(q) && s.group_mass(gg, q) > 0.0).map(|q| s.tg[gg][q]).fold(0.0f64, f64::max);
    eprintln!(" bois décomposé CPU {:.4e} GPU {:.4e} kg/(m·s) ; CO2 formé CPU {:.4e} GPU {:.4e} mol/(m·s) ; T gaz max CPU {:.0} GPU {:.0} K ; CO CPU {:.3e} GPU {:.3e} ; charbon de bois CPU {:.4} GPU {:.4}",
        pyro(&cpu), pyro(&gs), burnt(&cpu), burnt(&gs), tmax(&cpu), tmax(&gs), cpu.total_mass(co), gs.total_mass(co), cpu.total_mass(ch), gs.total_mass(ch));
    assert!(rel(pyro(&cpu), pyro(&gs)) < 0.05, "pyrolyse CPU {} GPU {}", pyro(&cpu), pyro(&gs));
    assert!(rel(burnt(&cpu), burnt(&gs)) < 0.25, "CO2 formé CPU {} GPU {}", burnt(&cpu), burnt(&gs));
    assert!(tmax(&cpu) > 1700.0 && tmax(&gs) > 1700.0, "flamme CPU {} GPU {}", tmax(&cpu), tmax(&gs));
}

/// diagnostic : feu de bois sur le GPU, sur la durée (MP_WOOD_T s, défaut 60 ; une ligne toutes les MP_WOOD_DT s, défaut 2 ; scène :
/// `wood_fire_for_diag`)
#[test]
#[ignore]
fn gpu_wood_fire_long_diag() {
    let mut gs = super::mp_chem_tests::wood_fire_for_diag();
    let Some(mut g) = gpu_or_skip(&gs) else { return };
    let w = gs.reg.phases.iter().position(|p| p.name == super::mp_materials::WOOD).unwrap();
    let m_w0 = gs.total_mass(w);
    let t_end = std::env::var("MP_WOOD_T").ok().and_then(|v| v.parse().ok()).unwrap_or(60.0f64);
    let every = std::env::var("MP_WOOD_DT").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0f64);
    let t0 = std::time::Instant::now();
    let mut next = every;
    while gs.time < t_end && !gs.diverged {
        g.step_batch(&mut gs, 50);
        if gs.time >= next {
            next += every;
            g.download(&mut gs);
            super::mp_chem_tests::wood_fire_report(&gs, m_w0, t0.elapsed().as_secs_f64());
        }
    }
    eprintln!("divergé : {}", gs.diverged);
}

#[test]
fn gpu_percolates_air_through_a_coal_bed_like_the_cpu() {
    // percolation (Darcy–Ergun dans la projection) à travers une grille et un lit de charbon en morceaux d'1 cm, air soufflé à 0.3 m/s
    use super::mp_percolation::tests::bed_on_grille;
    let mk = || bed_on_grille(0.01, 0.3).0;
    one_step_agreement("lit percolé", mk(), 30, 3, 1e-4, 0.01);
    let Some((cpu, gs, _)) = compare("lit percolé", &mk, 0.6) else { return };
    let qmean = |s: &MultiphaseSim| {
        let mut a = vec![];
        for j in 8..22 {
            for i in 2..s.nx - 2 {
                a.push(s.q_v[j * s.nx + i]);
            }
        }
        a.iter().sum::<f64>() / a.len() as f64
    };
    let dp = |s: &MultiphaseSim| (2..s.nx - 2).map(|i| s.p[s.idx(i, 8)] - s.p[s.idx(i, 20)]).sum::<f64>() / (s.nx - 4) as f64;
    eprintln!(" débit dans le lit : CPU {:.4} GPU {:.4} m/s ; Δp (rangées 8 → 20) CPU {:.2} GPU {:.2} Pa", qmean(&cpu), qmean(&gs), dp(&cpu), dp(&gs));
    assert!((qmean(&gs) / 0.3 - 1.0).abs() < 0.1 && rel(qmean(&cpu), qmean(&gs)) < 0.02);
    assert!(rel(dp(&cpu), dp(&gs)) < 0.02, "Δp CPU {} GPU {}", dp(&cpu), dp(&gs));
    let c = 1;
    let moved = (0..gs.nx * gs.ny).map(|q| (gs.alpha[c][q] - cpu.alpha[c][q]).abs()).fold(0.0f64, f64::max);
    assert!(moved < 0.01, "lits différents : {moved}");
}

#[test]
fn gpu_matches_cpu_hot_powder_cooling_in_air() {
    // poudre de 0.3 mm à 5 %, à 600 K dans de l'air à 300 K, sans gravité : échange par la surface des grains (k_exch_grains).
    // Comparé à t = τ (écart divisé par e), là où l'échange est en cours, pas une fois l'équilibre atteint
    use super::phases::{Phase, Registry};
    let mk = || {
        let mut r = Registry::new();
        let air = r.add_phase(Phase::air());
        let sable = r.add_phase(Phase::sand());
        let mut s = MultiphaseSim::new(6, 6, 0.01, r, air);
        s.gravity = [0.0, 0.0];
        s.paint_mixture(move |_, _| Some(vec![0.95, 0.05]));
        s.finalize_init(|_, _| 300.0);
        let gs = s.group_of[sable];
        for c in 0..36 {
            s.set_group_temperature(c, gs, 600.0);
        }
        s.refresh();
        s
    };
    let s0 = mk();
    let (ga, gsand) = (s0.group_of[0], s0.group_of[1]);
    let c = s0.idx(3, 3);
    let g = 12.0 * s0.group_alpha(gsand, c) * s0.group_k(ga, c) / (3e-4f64).powi(2);
    let tau = 1.0 / (g * (1.0 / s0.group_cap_at(gsand, c, 600.0) + 1.0 / s0.group_cap_at(ga, c, 300.0)));
    let mk_tau = || {
        let mut s = mk();
        s.dt_max = tau / 50.0;
        s
    };
    one_step_agreement("poudre chaude", mk_tau(), 5, 3, 1e-5, 0.05);
    let Some((cpu, gs, _)) = compare("poudre chaude", &mk_tau, tau) else { return };
    let gap = |s: &MultiphaseSim| s.tg[gsand][c] - s.tg[ga][c];
    eprintln!(" τ = {tau:.3e} s ; écart grains − air à t = τ : CPU {:.3} K GPU {:.3} K (e^-1 × 300 = {:.1})", gap(&cpu), gap(&gs), 300.0 / std::f64::consts::E);
    assert!(gap(&cpu) > 60.0 && gap(&cpu) < 160.0, "écart CPU {}", gap(&cpu));
    assert!((gap(&cpu) - gap(&gs)).abs() < 0.01 * gap(&cpu), "écart CPU {} GPU {}", gap(&cpu), gap(&gs));
    assert!(rel(cpu.total_internal_energy(), gs.total_internal_energy()) < 1e-5);
}

/// diagnostic : pas GPU BRUT du nuage de sable jusqu'au noyau MP_STOP (défaut k_sanity_final), après MP_WARM pas CPU ; cellules
/// invalides et plus grands écarts de masse au CPU
#[test]
#[ignore]
fn gpu_sand_raw_step_diag() {
    let sc = std::env::var("MP_SC").ok().and_then(|v| v.parse().ok()).map(MpScenario::from_index).unwrap_or(MpScenario::Sedimentation);
    let mut s = build_scenario(sc, 40, 30, 0.01, 9.81);
    let warm = std::env::var("MP_WARM").ok().and_then(|v| v.parse().ok()).unwrap_or(40usize);
    for _ in 0..warm {
        s.step();
    }
    let stop = std::env::var("MP_STOP").unwrap_or("k_sanity_final".into());
    let occ = std::env::var("MP_OCC").ok().and_then(|v| v.parse().ok()).unwrap_or(1usize);
    let mut gs = clone_sim(&s);
    let Some(mut g) = gpu_or_skip(&gs) else { return };
    g.debug_partial_step(&gs, &stop, occ);
    g.download(&mut gs);
    let np = gs.mass.len();
    let mut shown = 0;
    for c in 0..gs.nx * gs.ny {
        let tg: Vec<f64> = (0..gs.groups.len()).map(|q| gs.tg[q][c]).collect();
        let bad = !(gs.temp[c] > 50.0 && gs.temp[c] < 5000.0) || tg.iter().any(|t| !(*t > 50.0 && *t < 5000.0)) || !gs.e[c].is_finite()
            || (0..np).any(|k| !(gs.mass[k][c] >= -1e-6)) || !gs.p[c].is_finite();
        if bad && shown < 8 {
            shown += 1;
            let ms: Vec<String> = (0..np).map(|k| format!("{:.3e}", gs.mass[k][c])).collect();
            eprintln!("({}, {}) T {:.1} T_g {:?} p {:.0} m {:?}", c % gs.nx, c / gs.nx, gs.temp[c], tg, gs.p[c], ms);
        }
    }
    let umax = gs.u.iter().chain(gs.v.iter()).fold(0f64, |a, x| a.max(x.abs()));
    let mut cpu = s;
    cpu.step();
    eprintln!("cellules invalides {shown} ; |u|max GPU {umax:.3} CPU {:.3} ; dt CPU {:.3e}", cpu.max_speed(), cpu.last_dt);
    for k in 0..np {
        let (mut dmax, mut at) = (0f64, 0usize);
        for c in 0..gs.nx * gs.ny {
            let d = (gs.mass[k][c] - cpu.mass[k][c]).abs();
            if d > dmax { dmax = d; at = c; }
        }
        eprintln!(" m{k} : écart max {dmax:.3e} en ({}, {}) gpu {:.4e} cpu {:.4e}", at % gs.nx, at / gs.nx, gs.mass[k][at], cpu.mass[k][at]);
    }
}

/// diagnostic : suspension de sable sur le GPU, grandeurs tous les MP_EVERY pas (masses, p, colonne du fond)
#[test]
#[ignore]
fn gpu_sand_layer_trace() {
    let mut s = if std::env::var("MP_AIR").is_ok() { sand_suspension_under_air(6, 70, 1e-3, 0.1, 5e-4, 10) } else { sand_suspension(6, 60, 1e-3, std::env::var("MP_A0").ok().and_then(|v| v.parse().ok()).unwrap_or(0.1), 5e-4) };
    let sable = s.reg.phases.len() - 1;
    let Some(mut g) = gpu_or_skip(&s) else { return };
    let every = std::env::var("MP_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(20usize);
    let t_end = std::env::var("MP_T").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0f64);
    let mut k = 0;
    while s.time < t_end && !s.diverged {
        g.step(&mut s);
        k += 1;
        if k % every == 0 || g.last.retries > 0 {
            g.download(&mut s);
            let col: Vec<f64> = (0..8).map(|j| (s.alpha[sable][s.idx(3, j)] * 1000.0).round() / 1000.0).collect();
            let pmax = s.p.iter().fold(0f64, |a, &b| a.max(b));
            eprintln!("{:?}", g.last);
            eprintln!("pas {k} t {:.4} dt {:.2e} essais {} m_eau {:.6e} m_sable {:.6e} pmax {:.0} umax {:.2e} col {:?}", s.time, g.last.dt, g.last.retries,
                s.total_mass(sable - 1), s.total_mass(sable), pmax, s.max_speed(), col);
        }
    }
}

/// diagnostic : accord d'un pas de la suspension après MP_WARM pas CPU, et pas GPU brut (cellules invalides)
#[test]
#[ignore]
fn gpu_sand_layer_step_diag() {
    let warm = std::env::var("MP_WARM").ok().and_then(|v| v.parse().ok()).unwrap_or(94usize);
    let mut s = sand_suspension(6, 60, 1e-3, 0.1, 5e-4);
    if std::env::var("MP_GPUWARM").is_ok() {
        // préchauffage sur le GPU (sa propre trajectoire)
        let Some(mut g0) = gpu_or_skip(&s) else { return };
        for _ in 0..warm {
            g0.step(&mut s);
        }
        g0.download(&mut s);
        eprintln!("GPU préchauffé : t {:.4} essais du dernier pas {}", s.time, g0.last.retries);
    } else {
        for _ in 0..warm {
            s.step();
        }
    }
    let mut gs = clone_sim(&s);
    let Some(mut g) = gpu_or_skip(&gs) else { return };
    let stop = std::env::var("MP_STOP").unwrap_or("k_sanity_final".into());
    let occ = std::env::var("MP_OCC").ok().and_then(|v| v.parse().ok()).unwrap_or(1usize);
    g.debug_partial_step(&gs, &stop, occ);
    g.download(&mut gs);
    let mut cpu = clone_sim(&s);
    cpu.step();
    eprintln!("dt CPU {:.3e} ; |u| GPU {:.3e} CPU {:.3e}", cpu.last_dt, gs.max_speed(), cpu.max_speed());
    for j in 0..10 {
        let c = s.idx(3, j);
        let sa: f64 = (0..2).map(|k| gs.alpha[k][c]).sum();
        eprintln!(" j {j} : α_s avant {:.5} ; GPU m {:.5e} {:.5e} p {:.0} Σα {:.6} T {:.3} | CPU m {:.5e} {:.5e} p {:.0}", s.alpha[1][c], gs.mass[0][c], gs.mass[1][c], gs.p[c], sa, gs.temp[c],
            cpu.mass[0][c], cpu.mass[1][c], cpu.p[c]);
    }
}

/// diagnostic : boîte fermée d'eau pure (sans phase granulaire) sur le GPU
#[test]
#[ignore]
fn gpu_closed_water_box_trace() {
    use super::phases::{Phase, Registry};
    let mut r = Registry::new();
    let eau = r.add_phase(Phase::water());
    let mut s = MultiphaseSim::new(6, 60, 1e-3, r, eau);
    s.surface_tension = false;
    s.paint_mixture(move |_, _| Some(vec![1.0]));
    s.finalize_init(|_, _| 293.15);
    let Some(mut g) = gpu_or_skip(&s) else { return };
    let mut k = 0;
    while s.time < 2.0 && !s.diverged && k < 3000 {
        g.step(&mut s);
        k += 1;
        if k % 50 == 0 || g.last.retries > 0 {
            g.download(&mut s);
            let pmax = s.p.iter().fold(0f64, |a, &b| a.max(b));
            eprintln!("pas {k} t {:.4} dt {:.2e} essais {} m {:.6e} pmax {:.0} umax {:.2e} {:?}", s.time, g.last.dt, g.last.retries, s.total_mass(0), pmax, s.max_speed(), g.last);
        }
    }
}

/// diagnostic : braséro sur le GPU, pas par pas jusqu'à la divergence (retours arrière, T, p, cellules invalides)
#[test]
#[ignore]
fn gpu_brasero_trace() {
    let lib = super::mp_materials::MaterialLibrary::default();
    let mut s = super::mp_view::build_scenario_with(&lib, MpScenario::CoalFire, 40, 40, 0.01, 9.81).unwrap();
    let Some(mut g) = gpu_or_skip(&s) else { return };
    let mut k = 0;
    let stop_at = std::env::var("MP_STOPAT").ok().and_then(|v| v.parse::<usize>().ok());
    while s.time < 1.0 && !s.diverged && k < 6000 {
        if Some(k) == stop_at {
            // pas brut depuis l'état GPU courant : cellules invalides, vitesse maximale
            g.download(&mut s);
            let mut probe = clone_sim(&s);
            let (refused, dt) = g.debug_raw_step(&mut probe, s.dt_max, 1.0);
            let um = probe.max_speed();
            eprintln!("pas brut {k} : refusé {refused}, dt {dt:.3e}, |u| max {um:.3} (Courant {:.2})", um * dt / probe.dx);
            let mut worst: Vec<(f64, usize)> = (0..probe.nx * probe.ny).filter(|&c| !probe.is_solid(c)).map(|c| {
                let (i, j) = (c % probe.nx, c / probe.nx);
                let uc = (probe.u[probe.uidx(i, j)].abs()).max(probe.u[probe.uidx(i + 1, j)].abs()).max(probe.v[probe.vidx(i, j)].abs()).max(probe.v[probe.vidx(i, j + 1)].abs());
                (uc, c)
            }).collect();
            worst.sort_by(|a, b| b.0.total_cmp(&a.0));
            for &(uc, c) in worst.iter().take(4) {
                let ms: Vec<String> = (0..probe.mass.len()).map(|q| format!("{:.2e}", probe.mass[q][c])).collect();
                eprintln!("  ({}, {}) |u| {uc:.3} T {:.0} p {:.0} grille {} m {:?} ; avant : T {:.0} p {:.0}", c % probe.nx, c / probe.nx, probe.temp[c], probe.p[c], probe.is_grille(c), ms, s.temp[c], s.p[c]);
            }
            for c in 0..probe.nx * probe.ny {
                if probe.is_solid(c) {
                    continue;
                }
                for gq in 0..probe.groups.len() {
                    let mg = probe.group_mass(gq, c);
                    let tq = probe.tg[gq][c];
                    if mg > 0.0 && !(tq > 50.0 && tq < 5000.0) {
                        eprintln!("  groupe {gq} invalide ({}, {}) : T_g {tq:.1}, masse {mg:.3e} (T cellule {:.1}) ; avant T_g {:.1} masse {:.3e}", c % probe.nx, c / probe.nx, probe.temp[c], s.tg[gq][c], s.group_mass(gq, c));
                    }
                }
            }
            for c in 0..probe.nx * probe.ny {
                let bad = !(probe.temp[c] > 50.0 && probe.temp[c] < 5000.0) || !probe.p[c].is_finite() || (0..probe.mass.len()).any(|q| !(probe.mass[q][c] >= -1e-6));
                if bad {
                    let ms: Vec<String> = (0..probe.mass.len()).map(|q| format!("{:.2e}", probe.mass[q][c])).collect();
                    eprintln!("  invalide ({}, {}) T {:.1} p {:.0} m {:?}", c % probe.nx, c / probe.nx, probe.temp[c], probe.p[c], ms);
                }
            }
            return;
        }
        g.step(&mut s);
        k += 1;
        if k % 100 == 0 || g.last.retries > 0 || s.diverged {
            g.download(&mut s);
            let fl: Vec<usize> = (0..s.nx * s.ny).filter(|&c| !s.is_solid(c)).collect();
            let tmax = fl.iter().map(|&c| s.temp[c]).fold(0.0f64, f64::max);
            let tmin = fl.iter().map(|&c| s.temp[c]).fold(1e9f64, f64::min);
            let (pmin, pmax) = fl.iter().fold((1e30f64, 0.0f64), |(a, b), &c| (a.min(s.p[c]), b.max(s.p[c])));
            let qmax = s.q_u.iter().chain(s.q_v.iter()).fold(0.0f64, |a, &b| a.max(b.abs()));
            eprintln!("pas {k} t {:.4} dt {:.2e} essais {} T {tmin:.0}–{tmax:.0} p {pmin:.0}–{pmax:.0} |u| {:.2} |q| {qmax:.3} ; {:?}", s.time, g.last.dt, g.last.retries, s.max_speed(), g.last);
            if s.diverged {
                for c in 0..s.nx * s.ny {
                    let bad = !(s.temp[c] > 50.0 && s.temp[c] < 5000.0) || !s.p[c].is_finite() || (0..s.mass.len()).any(|q| !(s.mass[q][c] >= -1e-6));
                    if bad {
                        eprintln!("  invalide ({}, {}) T {:.1} p {:.0}", c % s.nx, c / s.nx, s.temp[c], s.p[c]);
                    }
                }
                break;
            }
        }
    }
}

#[test]
fn gpu_freezes_and_melts_like_the_cpu() {
    // gel / fusion (méthode enthalpique, `k_fusion`) : accord d'un pas sur une eau à −10 °C qui gèle et sur une glace à +10 °C qui fond,
    // puis 5 s de croissance de la glace contre une paroi froide (masse gelée par seconde à 1 % près). Pas bornés à 2 ms des deux
    // côtés : à 20 ms, la pression de cette colonne d'eau presque incompressible est mal conditionnée en f32 (le gradient conjugué
    // plafonne, le GPU refait des pas et finit à ~2 ms, le CPU tient 20 ms)
    use super::mp_materials::GLACE;
    one_step_agreement("gel", super::mp_chem_tests::ice_water_box(0.9, 263.15, 0.0, 263.15), 0, 3, 1e-4, 1e-2);
    one_step_agreement("fusion", super::mp_chem_tests::ice_water_box(0.5, 273.15, 0.4, 283.15), 0, 3, 1e-4, 1e-2);
    let mk = || {
        let mut s = super::mp_chem_tests::freezing_column(263.15);
        s.dt_max = 2e-3;
        s
    };
    let Some((cpu, gs, _)) = compare("colonne de gel", &mk, 5.0) else { return };
    let gl = cpu.reg.phases.iter().position(|p| p.name == GLACE).unwrap();
    let rate = |s: &MultiphaseSim| s.total_mass(gl) / s.time;
    eprintln!(" glace formée CPU {:.5e} GPU {:.5e} kg/(m·s) ; bilan gelé CPU {:.5e} GPU {:.5e} kg/m ; pas {} / {}",
        rate(&cpu), rate(&gs), cpu.frozen[0], gs.frozen[0], cpu.step_count, gs.step_count);
    assert!(rate(&cpu) > 0.0 && rel(rate(&cpu), rate(&gs)) < 0.01, "glace CPU {} GPU {}", rate(&cpu), rate(&gs));
    assert!(rel(gs.frozen[0], gs.total_mass(gl)) < 1e-3, "bilan gelé du GPU {} pour {} kg/m de glace", gs.frozen[0], gs.total_mass(gl));
}

/// diagnostic : pas de temps du GPU sur la colonne de gel (20 pas)
#[test]
#[ignore]
fn gpu_freezing_column_dt_diag() {
    let mut gs = super::mp_chem_tests::freezing_column(263.15);
    let Some(mut g) = gpu_or_skip(&gs) else { return };
    let mut cpu = super::mp_chem_tests::freezing_column(263.15);
    for k in 0..20 {
        let dg = g.step(&mut gs);
        let dc = cpu.step();
        eprintln!("pas {k} : dt GPU {dg:.3e} (chimie {:.2e}, phase {:.2e}) ; CPU {dc:.3e} (chimie {:.2e}, phase {:.2e}, choisi {:.2e}) ; GPU {:?}",
            gs.dt_chem, gs.dt_phase, cpu.dt_chem, cpu.dt_phase, cpu.choose_dt(), g.last);
    }
}

#[test]
fn gpu_melts_crushed_ice_like_the_cpu() {
    // glace pilée dans l'eau tiède : les cristaux remontent en fondant ; masse fondue par seconde sur 2 s à 3 % près
    use super::mp_materials::GLACE;
    let lib = super::mp_materials::MaterialLibrary::default();
    let mk = || super::mp_view::build_scenario_with(&lib, MpScenario::IceMelting, 40, 30, 0.02, 9.81).unwrap();
    let Some((cpu, gs, _)) = compare("glace pilée", &mk, 2.0) else { return };
    let gl = cpu.reg.phases.iter().position(|p| p.name == GLACE).unwrap();
    let mi0 = mk().total_mass(gl);
    let melt = |s: &MultiphaseSim| (mi0 - s.total_mass(gl)) / s.time;
    eprintln!(" glace fondue CPU {:.5e} GPU {:.5e} kg/(m·s) ; bilan CPU {:.5e} GPU {:.5e} kg/m ; pas {} / {}",
        melt(&cpu), melt(&gs), -cpu.frozen[0], -gs.frozen[0], cpu.step_count, gs.step_count);
    assert!(melt(&cpu) > 0.0 && rel(melt(&cpu), melt(&gs)) < 0.03, "fonte CPU {} GPU {}", melt(&cpu), melt(&gs));
}

#[test]
fn gpu_supercools_water_like_the_cpu() {
    // surfusion : accord d'un pas sur une eau très pure à −8 °C sans germe (rien ne gèle) et avec des cristaux (elle gèle jusqu'à 0 °C) ;
    // front de dendrites dans un canal d'eau à −10 °C (1 s : mêmes mailles gelées à une près, glace formée à 1 % près) ; scénario « eau
    // surfondue » (2 s : glace formée à 5 % près)
    use super::mp_materials::GLACE;
    let sc = |ai: f64| {
        let mut s = super::mp_chem_tests::ice_water_box(0.9 - ai, 265.15, ai, 265.15);
        s.reg.fusions[0].supercooling = 20.0;
        s
    };
    one_step_agreement("eau surfondue sans germe", sc(0.0), 0, 3, 1e-4, 1e-2);
    one_step_agreement("eau surfondue et cristaux", sc(0.02), 0, 3, 1e-4, 1e-2);
    let mk = || super::mp_chem_tests::supercooled_channel(100, 2e-3, 263.15);
    let Some((cpu, gs, _)) = compare("front de dendrites", &mk, 1.0) else { return };
    let gl = cpu.reg.phases.iter().position(|p| p.name == GLACE).unwrap();
    let lead = |s: &MultiphaseSim| (0..s.nx).filter(|&i| s.alpha[gl][s.idx(i, 0)] > 0.05).max().unwrap_or(0);
    eprintln!(" front : maille {} (CPU) / {} (GPU) ; glace CPU {:.5e} GPU {:.5e} kg/m ; pas {} / {}",
        lead(&cpu), lead(&gs), cpu.total_mass(gl), gs.total_mass(gl), cpu.step_count, gs.step_count);
    assert!(lead(&cpu) > 30 && lead(&cpu).abs_diff(lead(&gs)) <= 1, "front CPU {} GPU {}", lead(&cpu), lead(&gs));
    assert!(rel(cpu.total_mass(gl), gs.total_mass(gl)) < 0.01, "glace CPU {} GPU {}", cpu.total_mass(gl), gs.total_mass(gl));
    let lib = super::mp_materials::MaterialLibrary::default();
    let mk = || super::mp_view::build_scenario_with(&lib, MpScenario::SupercooledWater, 40, 30, 0.02, 9.81).unwrap();
    let Some((cpu, gs, _)) = compare("eau surfondue", &mk, 2.0) else { return };
    let mi0 = mk().total_mass(gl);
    let frz = |s: &MultiphaseSim| s.total_mass(gl) - mi0;
    eprintln!(" glace formée CPU {:.5e} GPU {:.5e} kg/m ; bilan CPU {:.5e} GPU {:.5e} ; pas {} / {}",
        frz(&cpu), frz(&gs), cpu.frozen[0], gs.frozen[0], cpu.step_count, gs.step_count);
    assert!(frz(&cpu) > 0.0 && rel(frz(&cpu), frz(&gs)) < 0.05, "glace CPU {} GPU {}", frz(&cpu), frz(&gs));
}

#[test]
fn gpu_freezes_water_wax_and_lava_like_the_cpu() {
    // trois paires de fusion dans les mêmes cellules (eau ↔ glace, cire fondue ↔ cire, lave ↔ basalte) : accord d'un pas
    one_step_agreement("eau, cire et lave", super::mp_chem_tests::water_wax_lava_box(268.15, 323.15, 1323.15), 0, 3, 1e-4, 1e-2);
}

#[test]
fn gpu_holds_and_fluidizes_a_blown_bed_like_the_cpu() {
    // lit de sable de 2 mm soufflé par une grille (U_mf d'Ergun 1.08 m/s) : accord d'un pas une fois le débit établi ; à 0.9 m/s, le lit
    // reste retenu (même perte de charge à 2 % près, sable immobile) ; à 1.2 m/s, il se fluidise et monte des deux côtés
    use super::mp_percolation::fluidization_tests::{blown_sand_bed, sand_height};
    let mut warm = blown_sand_bed(12, 40, 20, 0.9).0;
    for _ in 0..40 {
        warm.step();
    }
    one_step_agreement("lit soufflé retenu", warm, 0, 3, 1e-4, 1e-2);
    let mk = || blown_sand_bed(12, 40, 20, 0.9).0;
    let y0 = sand_height(&mk(), 1);
    let Some((cpu, gs, _)) = compare("lit retenu", &mk, 0.5) else { return };
    let dp = |s: &MultiphaseSim| {
        let ps = s.grain_overburden();
        let pf = |j: usize| (2..s.nx - 2).map(|i| { let q = s.idx(i, j); s.p[q] - ps[q] }).sum::<f64>() / (s.nx - 4) as f64;
        pf(3) - pf(34)
    };
    eprintln!(" lit retenu : perte de charge CPU {:.0} GPU {:.0} Pa ; sable monté CPU {:.2} GPU {:.2} mm",
        dp(&cpu), dp(&gs), 1e3 * (sand_height(&cpu, 1) - y0), 1e3 * (sand_height(&gs, 1) - y0));
    assert!(rel(dp(&cpu), dp(&gs)) < 0.02, "perte de charge CPU {} GPU {}", dp(&cpu), dp(&gs));
    assert!((sand_height(&gs, 1) - y0).abs() < 2e-3, "lit retenu soulevé sur le GPU");
    let mk = || blown_sand_bed(12, 40, 20, 1.2).0;
    let Some((cpu, gs, _)) = compare("lit fluidisé", &mk, 0.3) else { return };
    let (dc, dg) = (sand_height(&cpu, 1) - y0, sand_height(&gs, 1) - y0);
    eprintln!(" lit fluidisé : sable monté CPU {:.1} GPU {:.1} mm", 1e3 * dc, 1e3 * dg);
    assert!(dc > 0.005 && dg > 0.005, "lit non fluidisé : CPU {dc} GPU {dg}");
}

/// diagnostic : crée, avance et détruit MP_GPU_REPEAT solveurs GPU de suite (défaut 60, barrage 160 × 120) : la mémoire du processeur
/// graphique doit être rendue à chaque fois (sinon : « Not enough memory left » ou arrêt net du processus au bout de quelques dizaines)
#[test]
#[ignore]
fn gpu_create_and_drop_many_solvers_diag() {
    let n: usize = std::env::var("MP_GPU_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
    for k in 0..n {
        let mut gs = build_scenario(MpScenario::DamBreak, 160, 120, 0.005, 9.81);
        let Some(mut g) = gpu_or_skip(&gs) else { return };
        for _ in 0..3 {
            g.step(&mut gs);
        }
        g.download(&mut gs);
        drop(g);
        if k % 10 == 0 {
            eprintln!("solveur GPU {k} créé, avancé et détruit");
        }
    }
}
