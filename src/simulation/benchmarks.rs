//! Benchmarks de référence du solveur.
//!
//!   cargo test --release                       -> tests rapides
//!   cargo test --release -- --include-ignored  -> + cavités (quelques dizaines de secondes)
//!
//! Tous les tests tournent sur CPU, sauf `gpu_matches_cpu_*` qui est ignoré si aucun GPU n'est trouvé.
//! Les tolérances sont celles mesurées lors de la validation (voir le commentaire de chaque test).

use super::fluid::*;
use super::materials::*;

fn cpu_sim(w: usize, h: usize, dx: f32, mat: FluidMaterial, g: f32) -> FluidSim {
    let mut f = FluidSim::new(w, h, dx, mat, g);
    f.set_compute_method(ComputeMethod::CPU);
    f
}

fn all_finite(f: &FluidSim) -> bool {
    f.temperature.iter().chain(&f.velocity_x).chain(&f.velocity_y).chain(&f.pressure).chain(&f.tracer)
        .all(|v| v.is_finite())
}

fn vmax_ms(f: &FluidSim) -> f32 {
    let mut m = 0.0f32;
    for y in 1..f.height - 1 {
        for x in 1..f.width - 1 {
            if f.solids[x + y * f.width] == CellType::Wall { continue; }
            let (vx, vy) = f.cell_velocity(x, y);
            m = m.max((vx * vx + vy * vy).sqrt() * f.dx);
        }
    }
    m
}

fn rel(a: f32, reference: f32) -> f32 {
    (a - reference).abs() / reference.abs()
}

// ---------------------------------------------------------------- hydrostatique

/// Un fluide au repos sous gravité doit rester immobile et P(y) = ρ·g·y.
/// (f32 à 1e5 Pa : résolution ~0,008 Pa, d'où la tolérance de 1 %.)
fn check_hydrostatic(mat: FluidMaterial, outlet: bool) {
    let mut f = cpu_sim(30, 40, 0.01, mat, 9.81);
    if outlet {
        f.top_boundary.boundary_type = BoundaryType::Outlet;
    }
    for _ in 0..300 {
        f.step(0.01, 100);
    }
    assert!(vmax_ms(&f) < 1e-6, "vitesse parasite : {} m/s", vmax_ms(&f));
    let (w, x) = (f.width, f.width / 2);
    let dp = f.pressure_pa[x + 30 * w] - f.pressure_pa[x + 10 * w];
    let expected = f.material.rho_ambient() * 9.81 * 20.0 * f.dx;
    assert!(rel(dp, expected) < 0.01, "dP = {dp} Pa, ρgΔh = {expected} Pa");
}

#[test]
fn hydrostatic_air_closed() { check_hydrostatic(FluidMaterial::air(), false); }

#[test]
fn hydrostatic_air_open_top() { check_hydrostatic(FluidMaterial::air(), true); }

#[test]
fn hydrostatic_water_closed() { check_hydrostatic(FluidMaterial::water(), false); }

// ---------------------------------------------------------------- low-Mach : conservation

fn patch(f: &mut FluidSim, x0: usize, x1: usize, y0: usize, y1: usize, t: f32) {
    let w = f.width;
    for y in y0..y1 {
        for x in x0..x1 {
            f.temperature[x + y * w] = t;
        }
    }
}

/// Boîte fermée, conduction seule : la masse reste constante (mesuré : 1e-3, indépendant de dt)
/// et P_th ne bouge pas (l'énergie interne totale est conservée, donc P·V aussi).
#[test]
fn low_mach_closed_conduction_conserves_mass_and_pressure() {
    let mut f = cpu_sim(40, 40, 0.01, FluidMaterial::air(), 0.0);
    patch(&mut f, 15, 25, 15, 25, 600.0);
    f.update_rho();
    let (m0, p0) = (f.total_mass(), f.p_th);
    for _ in 0..300 {
        f.step(0.01, 200);
    }
    assert!(all_finite(&f));
    assert!(((f.total_mass() - m0) / m0).abs() < 3e-3, "dM/M = {}", (f.total_mass() - m0) / m0);
    assert!(((f.p_th - p0) / p0).abs() < 1e-5, "dP/P = {}", (f.p_th - p0) / p0);
}

/// Boîte fermée chauffée : ΔP_th = (γ−1)·E/V (gaz parfait, isochore). Mesuré : rapport 1,000.
#[test]
fn low_mach_closed_heating_matches_ideal_gas() {
    let dt = 0.01f32;
    let mut f = cpu_sim(40, 40, 0.01, FluidMaterial::air(), 0.0);
    f.update_rho();
    let (p0, w) = (f.p_th, f.width);
    let gamma = f.material.gamma() as f64;
    let mut e_added = 0.0f64; // J par mètre de profondeur
    for _ in 0..200 {
        for y in 18..22 {
            for x in 18..22 {
                f.heat_towards(x + y * w, 900.0, 5.0);
            }
        }
        e_added += f.heat_source.iter().map(|&q| q as f64).sum::<f64>() * dt as f64 * (f.dx * f.dx) as f64;
        f.step(dt, 200);
    }
    let volume = ((f.width - 2) * (f.height - 2)) as f64 * (f.dx * f.dx) as f64;
    let expected = (gamma - 1.0) * e_added / volume;
    let ratio = (f.p_th - p0) as f64 / expected;
    assert!((ratio - 1.0).abs() < 0.01, "ΔP/ΔP_attendu = {ratio}");
    assert!(all_finite(&f));
}

// ---------------------------------------------------------------- cavité chauffée (de Vahl Davis)

struct Cavity {
    nu_mid: f32, // Nusselt moyen sur le plan médian
    u: f32,      // max |u| sur la verticale médiane · L/α
    v: f32,      // max |v| sur l'horizontale médiane · L/α
    drift: f32,  // variation relative de Nu_mid sur le dernier quart (convergence)
    f: FluidSim,
}

/// Cavité carrée, Pr ≈ 0,71 (air), parois verticales = cellules solides à T imposée
/// (ThermalCell::Isothermal ; k_solide = 100 k_fluide -> Dirichlet à mi-cellule), haut/bas adiabatiques. ΔT = 10 K.
fn heated_cavity(ra: f32, n: usize, low_mach: bool, gpu: bool, t_end: f32, dt: f32) -> Cavity {
    let mut mat = FluidMaterial::air();
    mat.solid_conductivity = 100.0;
    let alpha = mat.conductivity / (mat.rho_ambient() * mat.cp);
    let (t0, dtemp, g) = (mat.t_ref, 10.0f32, 9.81f32);
    let l = (ra * mat.viscosity * alpha * t0 / (g * dtemp)).cbrt(); // Ra = g(ΔT/T0)L³/(να)
    let dx = l / n as f32;
    let (w, h) = (n + 4, n + 2);
    let mut f = FluidSim::new(w, h, dx, mat, g);
    if !gpu {
        f.set_compute_method(ComputeMethod::CPU);
    }
    f.low_mach = low_mach;
    let (th, tc) = (t0 + dtemp / 2.0, t0 - dtemp / 2.0);
    for y in 1..h - 1 {
        f.set_cell(1, y, CellType::Wall, None);
        f.set_cell(w - 2, y, CellType::Wall, None);
        f.set_thermal(1, y, Some(ThermalCell::Isothermal(th)));
        f.set_thermal(w - 2, y, Some(ThermalCell::Isothermal(tc)));
    }
    for y in 0..h {
        for x in 0..w {
            let s = ((x as f32 - 2.0) / n as f32).clamp(0.0, 1.0);
            f.temperature[x + y * w] = th + (tc - th) * s;
        }
    }
    let nu_mid = |f: &FluidSim| -> f32 {
        let xm = 2 + n / 2 - 1; // face entre xm et xm+1
        let mut sum = 0.0;
        for y in 1..h - 1 {
            let i = xm + y * w;
            let (t1, t2) = ((f.temperature[i] - tc) / dtemp, (f.temperature[i + 1] - tc) / dtemp);
            let u_face = f.velocity_x[i] * dx; // m/s
            sum += (u_face * l / alpha) * 0.5 * (t1 + t2) - n as f32 * (t2 - t1);
        }
        sum / (h - 2) as f32
    };
    let steps = (t_end / dt) as usize;
    let mut first = 0.0;
    for s in 0..steps {
        f.step(dt, 150);
        if s == steps * 3 / 4 {
            first = nu_mid(&f);
        }
    }
    let nu = nu_mid(&f);
    let (xm, ym) = (2 + n / 2, 1 + (n + 1) / 2);
    let u = (1..h - 1).map(|y| f.cell_velocity(xm, y).0.abs()).fold(0.0, f32::max);
    let v = (2..w - 2).map(|x| f.cell_velocity(x, ym).1.abs()).fold(0.0, f32::max);
    let norm = alpha / l;
    Cavity { nu_mid: nu, u: u * dx / norm, v: v * dx / norm, drift: (first - nu).abs() / nu, f }
}

/// Ra = 1e4, N = 40 (Boussinesq). Référence de Vahl Davis : Nu = 2,243, u = 16,18, v = 19,62.
/// Mesuré : Nu 2,299 (+2,5 %), u 16,24 (+0,4 %), v 19,53 (−0,5 %). Erreur ~ ordre 1 en dx.
#[test]
#[ignore]
fn heated_cavity_ra1e4() {
    let c = heated_cavity(1e4, 40, false, false, 20.0, 0.02);
    assert!(c.drift < 1e-4, "pas stationnaire : {}", c.drift);
    assert!(rel(c.nu_mid, 2.243) < 0.05, "Nu = {}", c.nu_mid);
    assert!(rel(c.u, 16.18) < 0.03, "u = {}", c.u);
    assert!(rel(c.v, 19.62) < 0.03, "v = {}", c.v);
}

/// Ra = 1e5, N = 40. Référence : Nu = 4,519, u = 34,73, v = 68,59.
/// Mesuré : Nu 4,711 (+4,2 %), u 37,51 (+8,0 %), v 69,40 (+1,2 %).
#[test]
#[ignore]
fn heated_cavity_ra1e5() {
    let c = heated_cavity(1e5, 40, false, false, 40.0, 0.02);
    assert!(c.drift < 1e-4, "pas stationnaire : {}", c.drift);
    assert!(rel(c.nu_mid, 4.519) < 0.06, "Nu = {}", c.nu_mid);
    assert!(rel(c.u, 34.73) < 0.10, "u = {}", c.u);
    assert!(rel(c.v, 68.59) < 0.03, "v = {}", c.v);
}

/// Pour ΔT = 10 K le low-Mach doit retrouver la solution de Boussinesq (mesuré : < 1 % sur u et v).
#[test]
#[ignore]
fn low_mach_agrees_with_boussinesq_on_cavity() {
    let b = heated_cavity(1e4, 40, false, false, 20.0, 0.02);
    let l = heated_cavity(1e4, 40, true, false, 20.0, 0.02);
    assert!(rel(l.u, b.u) < 0.02, "u : {} vs {}", l.u, b.u);
    assert!(rel(l.v, b.v) < 0.02, "v : {} vs {}", l.v, b.v);
    assert!(rel(l.nu_mid, b.nu_mid) < 0.03, "Nu : {} vs {}", l.nu_mid, b.nu_mid);
}

// ---------------------------------------------------------------- cavité entraînée (Ghia 1982)

/// Re = U·L/ν = 100, eau (incompressible), couvercle = Inlet tangentiel (vitesse normale 0).
/// Renvoie (u_min sur la verticale médiane, v_max, v_min sur l'horizontale médiane), normalisés par U,
/// avec y vers le HAUT (la grille a y vers le bas).
fn lid_cavity(n: usize, re: f32) -> (f32, f32, f32) {
    let mat = FluidMaterial::water();
    let l = 0.1f32;
    let u_lid = re * mat.viscosity / l;
    let dx = l / n as f32;
    let mut f = cpu_sim(n + 2, n + 2, dx, mat, 0.0);
    f.top_boundary = Boundary {
        boundary_type: BoundaryType::Inlet, velocity_x_ms: u_lid, velocity_y_ms: 0.0,
        tracer: 0.0, offset: 0, temperature: mat.t_ref,
    };
    let dt = l / u_lid / n as f32; // ~1 cellule d'advection par pas au couvercle
    for _ in 0..(40.0 * l / u_lid / dt) as usize {
        f.step(dt, 200);
    }
    let (a, b) = (n / 2, n / 2 + 1);
    let mut u_min = 0.0f32;
    let (mut v_max, mut v_min) = (f32::MIN, f32::MAX);
    for y in 1..=n {
        u_min = u_min.min(0.5 * (f.cell_velocity(a, y).0 + f.cell_velocity(b, y).0) * dx / u_lid);
    }
    for x in 1..=n {
        let v = -0.5 * (f.cell_velocity(x, a).1 + f.cell_velocity(x, b).1) * dx / u_lid;
        v_max = v_max.max(v);
        v_min = v_min.min(v);
    }
    (u_min, v_max, v_min)
}

/// Ghia et al. (1982), Re = 100 : u_min = −0,2109, v_max = +0,1753, v_min = −0,2453.
/// Mesuré N = 32 : −0,1959 (7,1 %), +0,1666 (5,0 %), −0,2268 (7,6 %) ; N = 64 : 3,0 %, 1,3 %, 2,7 %.
#[test]
#[ignore]
fn lid_driven_cavity_re100() {
    let (u, vmax, vmin) = lid_cavity(32, 100.0);
    assert!(rel(u, -0.2109) < 0.10, "u_min = {u}");
    assert!(rel(vmax, 0.1753) < 0.10, "v_max = {vmax}");
    assert!(rel(vmin, -0.2453) < 0.10, "v_min = {vmin}");
}

// ---------------------------------------------------------------- parité GPU / CPU

/// Même cavité sur GPU et CPU : mesuré identique à 4 chiffres sur Nu, écarts ~1e-7 m/s.
#[test]
#[ignore]
fn gpu_matches_cpu_heated_cavity() {
    for low_mach in [false, true] {
        let g = heated_cavity(1e4, 24, low_mach, true, 5.0, 0.02);
        if g.f.compute_method() != ComputeMethod::GPU {
            eprintln!("pas de GPU : test ignoré");
            return;
        }
        let c = heated_cavity(1e4, 24, low_mach, false, 5.0, 0.02);
        assert!(rel(g.nu_mid, c.nu_mid) < 1e-3, "Nu : {} vs {}", g.nu_mid, c.nu_mid);
        assert!(rel(g.u, c.u) < 1e-3 && rel(g.v, c.v) < 1e-3);
        let dt_max = g.f.temperature.iter().zip(&c.f.temperature).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(dt_max < 0.05, "écart de T : {dt_max} K");
    }
}

// ---------------------------------------------------------------- solides thermiques, vitesses en m/s

/// Boîte fermée d'air + solide chauffant : l'énergie fournie (P·V·t) se retrouve dans le gaz
/// (ΔP_th·V/(γ−1)) et dans le solide (Σ ρ_s·cp_s·ΔT·dx²). Conservation à 2 % près.
#[test]
fn heater_energy_is_conserved_in_closed_box() {
    let (dt, power) = (0.01f32, 2e6f32); // W/m³
    let mut mat = FluidMaterial::air();
    mat.solid_rho = 10.0; // solide léger : une part notable de l'énergie passe dans le gaz
    let mut f = cpu_sim(40, 40, 0.01, mat, 0.0);
    let mut cells = Vec::new();
    for y in 18..22 {
        for x in 18..22 {
            f.set_cell(x, y, CellType::Wall, None);
            f.set_thermal(x, y, Some(ThermalCell::Heater(power)));
            cells.push(x + y * f.width);
        }
    }
    let (p0, t0) = (f.p_th, f.material.t_ref);
    for _ in 0..300 {
        f.step(dt, 200);
    }
    assert!(all_finite(&f));
    let dx2 = (f.dx * f.dx) as f64;
    let e_in = power as f64 * cells.len() as f64 * dx2 * 300.0 * dt as f64; // J par mètre de profondeur
    let n_gas = (0..f.solids.len()).filter(|&i| {
        let (x, y) = (i % f.width, i / f.width);
        x >= 1 && x < f.width - 1 && y >= 1 && y < f.height - 1 && f.solids[i] != CellType::Wall
    }).count();
    let volume = n_gas as f64 * dx2;
    let e_gas = (f.p_th - p0) as f64 * volume / (f.material.gamma() as f64 - 1.0);
    let e_solid: f64 = cells.iter().map(|&i| f.material.heat_capacity_solid() as f64 * (f.temperature[i] - t0) as f64 * dx2).sum();
    let balance = (e_gas + e_solid) / e_in;
    assert!((balance - 1.0).abs() < 0.02, "E_gaz + E_solide = {balance} × E_fournie (gaz {e_gas:.0} J/m, solide {e_solid:.0} J/m)");
}

/// Un solide isotherme garde sa température, et perd son effet quand on le rend au fluide.
#[test]
fn isothermal_cell_holds_temperature_and_is_cleared() {
    let mut f = cpu_sim(30, 30, 0.01, FluidMaterial::air(), 0.0);
    f.set_cell(15, 15, CellType::Wall, None);
    f.set_thermal(15, 15, Some(ThermalCell::Isothermal(500.0)));
    assert_eq!(f.thermal_count, 1);
    for _ in 0..50 {
        f.step(0.01, 50);
    }
    assert!((f.temperature[15 + 15 * f.width] - 500.0).abs() < 1e-3);
    assert!(f.temperature[16 + 15 * f.width] > f.material.t_ref + 1.0, "le voisin doit chauffer");
    f.set_cell(15, 15, CellType::Fluid, None);
    assert_eq!(f.thermal_count, 0);
    assert!(f.thermal_at(15 + 15 * f.width).is_none());
}

/// Vitesse d'Inlet en m/s : le débit à travers n'importe quelle colonne = débit d'entrée
/// (écoulement incompressible, canal à parois adhérentes), donc la vitesse moyenne vaut la vitesse imposée.
#[test]
fn inlet_velocity_is_in_meters_per_second() {
    let v_in = 0.01f32; // m/s
    let mut f = cpu_sim(60, 22, 0.01, FluidMaterial::water(), 0.0);
    f.left_boundary = Boundary {
        boundary_type: BoundaryType::Inlet, velocity_x_ms: v_in, velocity_y_ms: 0.0,
        tracer: 0.0, offset: 0, temperature: f.material.t_ref,
    };
    f.right_boundary.boundary_type = BoundaryType::Outlet;
    for _ in 0..100 {
        f.step(0.05, 150);
    }
    let rows = f.height - 2;
    for &x in &[10usize, 30, 50] {
        let mean: f32 = (1..f.height - 1).map(|y| f.velocity_x[x + y * f.width] * f.dx).sum::<f32>() / rows as f32;
        assert!(rel(mean, v_in) < 0.01, "colonne {x} : vitesse moyenne {mean} m/s (attendu {v_in})");
    }
}

/// Emitter / Accelerator / Sink (m/s, m/s²) : GPU = CPU.
#[test]
#[ignore]
fn gpu_matches_cpu_special_cells() {
    let build = |gpu: bool| {
        let mut f = FluidSim::new(40, 40, 0.01, FluidMaterial::air(), 0.0);
        if !gpu {
            f.set_compute_method(ComputeMethod::CPU);
        }
        f.right_boundary.boundary_type = BoundaryType::Outlet;
        f.set_cell(8, 20, CellType::Emitter, Some(CellEffect::Emitter { vx: 0.3, vy: 0.0, tracer: 1.0 }));
        f.set_cell(20, 10, CellType::Accelerator, Some(CellEffect::Accelerator { ax: 2.0, ay: 0.0 }));
        f.set_cell(30, 20, CellType::Sink, Some(CellEffect::Sink { rate: 0.1 }));
        f
    };
    let (mut c, mut g) = (build(false), build(true));
    if g.compute_method() != ComputeMethod::GPU {
        eprintln!("pas de GPU : test ignoré");
        return;
    }
    for _ in 0..100 {
        c.step(0.01, 100);
        g.step(0.01, 100);
    }
    let dv = c.velocity_x.iter().zip(&g.velocity_x).chain(c.velocity_y.iter().zip(&g.velocity_y))
        .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max) * c.dx;
    let vmax = c.velocity_x.iter().chain(&c.velocity_y).map(|v| v.abs()).fold(0.0, f32::max) * c.dx;
    assert!(vmax > 0.05, "scène sans écoulement : vmax = {vmax}");
    assert!(dv < 1e-3 * vmax.max(1.0), "dv = {dv} m/s (vmax {vmax})");
    let dtr = c.tracer.iter().zip(&g.tracer).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    assert!(dtr < 1e-3, "écart de tracer : {dtr}");
}