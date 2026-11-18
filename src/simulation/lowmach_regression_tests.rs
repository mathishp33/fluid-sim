use super::fluid::*;
use super::materials::*;

fn cpu_sim(w: usize, h: usize, dx: f32, mat: FluidMaterial, g: f32) -> FluidSim {
    let mut f = FluidSim::new(w, h, dx, mat, g);
    f.set_compute_method(ComputeMethod::CPU);
    f
}
fn finite(f: &FluidSim) -> bool {
    f.temperature.iter().chain(&f.velocity_x).chain(&f.velocity_y).chain(&f.pressure).chain(&f.rho).all(|v| v.is_finite())
}

#[test]
fn conduction_mass() {
    let mut f = cpu_sim(40, 40, 0.01, FluidMaterial::air(), 0.0);
    let w = f.width;
    for y in 15..25 { for x in 15..25 { f.temperature[x + y * w] = 600.0; } }
    f.update_rho();
    let (m0, p0) = (f.total_mass(), f.p_th);
    for _ in 0..300 { f.step(0.01, 200); }
    println!("conduction dM/M = {:e}  dP/P = {:e}", (f.total_mass() - m0) / m0, (f.p_th - p0) / p0);
    assert!(finite(&f));
}

#[test]
fn heat_then_cool() {
    let dt = 0.01f32;
    let mut f = cpu_sim(40, 40, 0.01, FluidMaterial::air(), 0.0);
    let w = f.width;
    f.refresh_rho();
    let m0 = f.total_mass();
    for _ in 0..200 {
        for y in 18..22 { for x in 18..22 { f.heat_towards(x + y * w, 2000.0, 5.0); } }
        f.step(dt, 100);
    }
    println!("après chauffe dM/M = {:e}", (f.total_mass() - m0) / m0);
    for _ in 0..600 {
        for y in 1..f.height - 1 { for x in 1..w - 1 { f.heat_towards(x + y * w, f.material.t_ref, 5.0); } }
        f.step(dt, 100);
    }
    println!("après refroidissement dM/M = {:e}", (f.total_mass() - m0) / m0);
    assert!(finite(&f));
}

// Pinceau clic droit : la vitesse rejoint celle du curseur (cellules/s), appliquée chaque frame
fn brush_run(heated: bool, iters: usize, dt: f32, cursor: f32, frames: usize) -> Option<usize> {
    let mut f = cpu_sim(120, 80, 0.01, FluidMaterial::air(), 9.81);
    let w = f.width;
    if heated {
        for y in 30..50 { for x in 40..60 { f.temperature[x + y * w] = 800.0; } }
    }
    f.refresh_rho();
    let m0 = f.total_mass();
    let push = 1.0 - (-20.0f32 * dt).exp();
    for frame in 0..frames {
        for y in 30..50 {
            for x in 20 + (frame % 60)..40 + (frame % 60) {
                let i = x + y * w;
                f.velocity_x[i] += (cursor - f.velocity_x[i]) * push;
                f.velocity_y[i] += (0.3 * cursor - f.velocity_y[i]) * push;
            }
        }
        f.step(dt, iters);
        if !finite(&f) { return Some(frame); }
    }
    let (mn, mx) = f.rho.iter().fold((f32::MAX, f32::MIN), |a, &r| (a.0.min(r), a.1.max(r)));
    println!("  fin : rho [{mn:.4}, {mx:.4}]  dM/M {:.2e}", (f.total_mass() - m0) / m0);
    None
}

#[test]
fn brush() {
    for &(heated, iters, dt, cursor) in &[
        (false, 30, 0.016, 200.0), (false, 30, 0.016, 1000.0), (true, 30, 0.016, 1000.0),
        (false, 30, 0.0001, 1000.0), (true, 100, 0.016, 2000.0),
    ] {
        let r = brush_run(heated, iters, dt, cursor, 200);
        println!("heated={heated} iters={iters} dt={dt} cursor={cursor} cells/s -> NaN à la frame {:?}", r);
    }
}

#[test]
fn brush_trace() {
    let (dt, cursor) = (0.016f32, 1000.0f32);
    let mut f = cpu_sim(120, 80, 0.01, FluidMaterial::air(), 9.81);
    let w = f.width;
    f.refresh_rho();
    let push = 1.0 - (-20.0f32 * dt).exp();
    for frame in 0..70 {
        for y in 30..50 { for x in 20 + (frame % 60)..40 + (frame % 60) {
            let i = x + y * w;
            f.velocity_x[i] += (cursor - f.velocity_x[i]) * push;
            f.velocity_y[i] += (0.3 * cursor - f.velocity_y[i]) * push;
        } }
        f.step(dt, 30);
        let res = (0..f.divergence.len()).map(|i| (f.divergence[i] - f.source[i]).abs()).fold(0.0f32, f32::max);
        let smax = f.source.iter().fold(0.0f32, |a, &s| a.max(s.abs()));
        let (mn, mx) = f.rho.iter().enumerate().filter(|(i, _)| f.solids[*i] == CellType::Fluid && i % w > 0 && i % w < w-1 && i / w > 0 && i / w < f.height-1)
            .fold((f32::MAX, f32::MIN), |a, (_, &r)| (a.0.min(r), a.1.max(r)));
        let (tmn, tmx) = f.temperature.iter().fold((f32::MAX, f32::MIN), |a, &t| (a.0.min(t), a.1.max(t)));
        let vmax = f.velocity_x.iter().chain(&f.velocity_y).fold(0.0f32, |a, &v| a.max(v.abs()));
        if frame % 5 == 0 || frame > 50 {
            println!("f{frame:3} |div-S| {res:9.3e} |S| {smax:8.2e} rho [{mn:8.4},{mx:8.4}] T [{tmn:8.2},{tmx:9.2}] vmax {vmax:8.1} dpdt {:.2e}", f.dpdt);
        }
        if !finite(&f) { println!("NaN"); break; }
    }
}

#[test]
fn closed_heating_ideal_gas() {
    let dt = 0.01f32;
    let mut f = cpu_sim(40, 40, 0.01, FluidMaterial::air(), 0.0);
    f.update_rho();
    let (p0, w) = (f.p_th, f.width);
    let gamma = f.material.gamma() as f64;
    let mut e_added = 0.0f64;
    for _ in 0..200 {
        for y in 18..22 { for x in 18..22 { f.heat_towards(x + y * w, 900.0, 5.0); } }
        e_added += f.heat_source.iter().map(|&q| q as f64).sum::<f64>() * dt as f64 * (f.dx * f.dx) as f64;
        f.step(dt, 200);
    }
    let volume = ((f.width - 2) * (f.height - 2)) as f64 * (f.dx * f.dx) as f64;
    let ratio = (f.p_th - p0) as f64 / ((gamma - 1.0) * e_added / volume);
    println!("ΔP/ΔP_attendu = {ratio}");
    assert!((ratio - 1.0).abs() < 0.01);
}

#[test]
fn heater_energy() {
    let (dt, power) = (0.01f32, 2e6f32);
    let mut mat = FluidMaterial::air();
    mat.solid_rho = 10.0;
    let mut f = cpu_sim(40, 40, 0.01, mat, 0.0);
    let mut cells = Vec::new();
    for y in 18..22 { for x in 18..22 {
        f.set_cell(x, y, CellType::Wall, None);
        f.set_thermal(x, y, Some(ThermalCell::Heater(power)));
        cells.push(x + y * f.width);
    } }
    let (p0, t0) = (f.p_th, f.material.t_ref);
    for _ in 0..300 { f.step(dt, 200); }
    let dx2 = (f.dx * f.dx) as f64;
    let e_in = power as f64 * cells.len() as f64 * dx2 * 300.0 * dt as f64;
    let n_gas = (0..f.solids.len()).filter(|&i| { let (x, y) = (i % f.width, i / f.width);
        x >= 1 && x < f.width - 1 && y >= 1 && y < f.height - 1 && f.solids[i] != CellType::Wall }).count();
    let e_gas = (f.p_th - p0) as f64 * n_gas as f64 * dx2 / (f.material.gamma() as f64 - 1.0);
    let e_solid: f64 = cells.iter().map(|&i| f.material.heat_capacity_solid() as f64 * (f.temperature[i] - t0) as f64 * dx2).sum();
    let balance = (e_gas + e_solid) / e_in;
    println!("bilan énergie = {balance}");
    assert!((balance - 1.0).abs() < 0.02);
}

fn cavity(low_mach: bool) -> (f32, f32) {
    let mut mat = FluidMaterial::air();
    mat.solid_conductivity = 100.0;
    let alpha = mat.conductivity / (mat.rho_ambient() * mat.cp);
    let (t0, dtemp, g, n) = (mat.t_ref, 10.0f32, 9.81f32, 40usize);
    let l = (1e4f32 * mat.viscosity * alpha * t0 / (g * dtemp)).cbrt();
    let dx = l / n as f32;
    let (w, h) = (n + 4, n + 2);
    let mut f = cpu_sim(w, h, dx, mat, g);
    f.low_mach = low_mach;
    let (th, tc) = (t0 + 5.0, t0 - 5.0);
    for y in 1..h - 1 {
        f.set_cell(1, y, CellType::Wall, None); f.set_cell(w - 2, y, CellType::Wall, None);
        f.set_thermal(1, y, Some(ThermalCell::Isothermal(th))); f.set_thermal(w - 2, y, Some(ThermalCell::Isothermal(tc)));
    }
    for y in 0..h { for x in 0..w {
        let s = ((x as f32 - 2.0) / n as f32).clamp(0.0, 1.0);
        f.temperature[x + y * w] = th + (tc - th) * s;
    } }
    for _ in 0..(20.0f32 / 0.02) as usize { f.step(0.02, 150); }
    let (xm, ym) = (2 + n / 2, 1 + (n + 1) / 2);
    let norm = alpha / l;
    let u = (1..h - 1).map(|y| f.cell_velocity(xm, y).0.abs()).fold(0.0, f32::max) * dx / norm;
    let v = (2..w - 2).map(|x| f.cell_velocity(x, ym).1.abs()).fold(0.0, f32::max) * dx / norm;
    (u, v)
}

#[test]
fn cavity_lowmach_vs_boussinesq() {
    let b = cavity(false);
    let l = cavity(true);
    println!("Boussinesq u={:.3} v={:.3} | low-Mach u={:.3} v={:.3}  (réf. de Vahl Davis u=16.18 v=19.62)", b.0, b.1, l.0, l.1);
    assert!((l.0 - b.0).abs() / b.0 < 0.02 && (l.1 - b.1).abs() / b.1 < 0.02);
}

// ---- MacCormack vs semi-lagrangienne : mêmes cavités que benchmarks.rs
struct Cav { nu: f32, u: f32, v: f32, drift: f32 }

fn heated_cavity(ra: f32, n: usize, low_mach: bool, mc: bool, t_end: f32, dt: f32) -> Cav {
    let mut mat = FluidMaterial::air();
    mat.solid_conductivity = 100.0;
    let alpha = mat.conductivity / (mat.rho_ambient() * mat.cp);
    let (t0, dtemp, g) = (mat.t_ref, 10.0f32, 9.81f32);
    let l = (ra * mat.viscosity * alpha * t0 / (g * dtemp)).cbrt();
    let dx = l / n as f32;
    let (w, h) = (n + 4, n + 2);
    let mut f = cpu_sim(w, h, dx, mat, g);
    f.low_mach = low_mach;
    f.maccormack = mc;
    f.high_order_scalars = std::env::var("UPWIND").is_err(); // UPWIND=1 : scalaires d'ordre 1
    f.variable_properties = std::env::var("CONSTPROPS").is_err(); // CONSTPROPS=1 : ν, k constants (ancien comportement)
    let (th, tc) = (t0 + dtemp / 2.0, t0 - dtemp / 2.0);
    for y in 1..h - 1 {
        f.set_cell(1, y, CellType::Wall, None);
        f.set_cell(w - 2, y, CellType::Wall, None);
        f.set_thermal(1, y, Some(ThermalCell::Isothermal(th)));
        f.set_thermal(w - 2, y, Some(ThermalCell::Isothermal(tc)));
    }
    for y in 0..h { for x in 0..w {
        let s = ((x as f32 - 2.0) / n as f32).clamp(0.0, 1.0);
        f.temperature[x + y * w] = th + (tc - th) * s;
    } }
    let nu_mid = |f: &FluidSim| -> f32 {
        let xm = 2 + n / 2 - 1;
        let mut sum = 0.0;
        for y in 1..h - 1 {
            let i = xm + y * w;
            let (t1, t2) = ((f.temperature[i] - tc) / dtemp, (f.temperature[i + 1] - tc) / dtemp);
            let u_face = f.velocity_x[i] * dx;
            sum += (u_face * l / alpha) * 0.5 * (t1 + t2) - n as f32 * (t2 - t1);
        }
        sum / (h - 2) as f32
    };
    let steps = (t_end / dt) as usize;
    let mut first = 0.0;
    for s in 0..steps {
        f.step(dt, 150);
        if s == steps * 3 / 4 { first = nu_mid(&f); }
    }
    let nu = nu_mid(&f);
    let (xm, ym) = (2 + n / 2, 1 + (n + 1) / 2);
    let u = (1..h - 1).map(|y| f.cell_velocity(xm, y).0.abs()).fold(0.0, f32::max);
    let v = (2..w - 2).map(|x| f.cell_velocity(x, ym).1.abs()).fold(0.0, f32::max);
    let norm = alpha / l;
    Cav { nu, u: u * dx / norm, v: v * dx / norm, drift: (first - nu).abs() / nu }
}

fn lid_cavity(n: usize, re: f32, mc: bool) -> (f32, f32, f32) {
    let mat = FluidMaterial::water();
    let l = 0.1f32;
    let u_lid = re * mat.viscosity / l;
    let dx = l / n as f32;
    let mut f = cpu_sim(n + 2, n + 2, dx, mat, 0.0);
    f.maccormack = mc;
    f.top_boundary = Boundary { boundary_type: BoundaryType::Inlet, velocity_x_ms: u_lid, velocity_y_ms: 0.0,
        tracer: 0.0, offset: 0, temperature: mat.t_ref };
    let dt = l / u_lid / n as f32;
    for _ in 0..(40.0 * l / u_lid / dt) as usize { f.step(dt, 200); }
    let (a, b) = (n / 2, n / 2 + 1);
    let mut u_min = 0.0f32;
    let (mut v_max, mut v_min) = (f32::MIN, f32::MAX);
    for y in 1..=n { u_min = u_min.min(0.5 * (f.cell_velocity(a, y).0 + f.cell_velocity(b, y).0) * dx / u_lid); }
    for x in 1..=n {
        let v = -0.5 * (f.cell_velocity(x, a).1 + f.cell_velocity(x, b).1) * dx / u_lid;
        v_max = v_max.max(v); v_min = v_min.min(v);
    }
    (u_min, v_max, v_min)
}

fn pct(a: f32, r: f32) -> f32 { 100.0 * (a - r) / r }

#[test]
#[ignore] // ~30 s : mesure SL vs MacCormack (affiche les écarts aux références)
fn mc_cavities() {
    for &mc in &[false, true] {
        let name = if mc { "MacCormack " } else { "semi-lagr. " };
        let c = heated_cavity(1e4, 40, false, mc, 20.0, 0.02);
        println!("{name} Ra=1e4 N=40  Nu {:.3} ({:+.1}%)  u {:.2} ({:+.1}%)  v {:.2} ({:+.1}%)  drift {:.1e}   [réf 2.243 / 16.18 / 19.62]",
                 c.nu, pct(c.nu, 2.243), c.u, pct(c.u, 16.18), c.v, pct(c.v, 19.62), c.drift);
        let c = heated_cavity(1e5, 40, false, mc, 40.0, 0.02);
        println!("{name} Ra=1e5 N=40  Nu {:.3} ({:+.1}%)  u {:.2} ({:+.1}%)  v {:.2} ({:+.1}%)  drift {:.1e}   [réf 4.519 / 34.73 / 68.59]",
                 c.nu, pct(c.nu, 4.519), c.u, pct(c.u, 34.73), c.v, pct(c.v, 68.59), c.drift);
        let c = heated_cavity(1e5, 40, true, mc, 40.0, 0.02);
        println!("{name} Ra=1e5 N=40 low-Mach  Nu {:.3} ({:+.1}%)  u {:.2} ({:+.1}%)  v {:.2} ({:+.1}%)  drift {:.1e}",
                 c.nu, pct(c.nu, 4.519), c.u, pct(c.u, 34.73), c.v, pct(c.v, 68.59), c.drift);
        for &n in &[32usize, 64] {
            let (u, vx, vn) = lid_cavity(n, 100.0, mc);
            println!("{name} Ghia Re=100 N={n}: u_min {:.4} ({:+.1}%)  v_max {:.4} ({:+.1}%)  v_min {:.4} ({:+.1}%)  [réf -0.2109 / 0.1753 / -0.2453]",
                     u, pct(u, -0.2109), vx, pct(vx, 0.1753), vn, pct(vn, -0.2453));
        }
    }
}

#[test]
#[ignore] // ~40 s : convergence en maillage de la cavité chauffée Ra=1e5
fn mc_cavity_refinement() {
    for &mc in &[false, true] {
        let name = if mc { "MacCormack " } else { "semi-lagr. " };
        for &n in &[20usize, 40, 80] {
            let dt = 0.02 * 40.0 / n as f32; // CFL ~ constant
            let c = heated_cavity(1e5, n, false, mc, 40.0, dt);
            println!("{name} Ra=1e5 N={n:3}: Nu {:.3} ({:+.1}%)  u {:.2} ({:+.1}%)  v {:.2} ({:+.1}%)  drift {:.0e}",
                     c.nu, pct(c.nu, 4.519), c.u, pct(c.u, 34.73), c.v, pct(c.v, 68.59), c.drift);
        }
    }
}

// ---- Perte d'énergie cinétique : cellule de tourbillon exacte en boîte à parois glissantes
// ψ ∝ sin(πX/L)·sin(πY/L) est une fonction propre du laplacien : J(ψ,ω)=0, l'advection ne fait rien
// physiquement. Seule la viscosité dissipe : E(t)/E0 = exp(-2·ν·k²·t), k² = 2π²/L².
// Tout écart supplémentaire est de la dissipation numérique.
fn vortex_ke(n: usize, nu: f32, mc: bool, t_end: f32, dt: f32) -> (f64, f64) {
    let mut mat = FluidMaterial::water();
    mat.viscosity = nu;
    let (l, u0) = (0.1f32, 0.1f32);
    let dx = l / n as f32;
    let (w, h) = (n + 2, n + 2);
    let mut f = cpu_sim(w, h, dx, mat, 0.0);
    f.no_slip = false;
    f.maccormack = mc;
    let pi = std::f32::consts::PI;
    let nf = n as f32;
    for y in 1..h - 1 { for x in 0..w - 1 {            // u : face droite de la cellule x, centre en y
        let (xx, yy) = (x as f32 / nf, (y as f32 - 0.5) / nf);
        f.velocity_x[x + y * w] = (u0 / dx) * (pi * xx).sin() * (pi * yy).cos();
    } }
    for y in 0..h - 1 { for x in 1..w - 1 {            // v : face basse de la cellule y, centre en x
        let (xx, yy) = ((x as f32 - 0.5) / nf, y as f32 / nf);
        f.velocity_y[x + y * w] = -(u0 / dx) * (pi * xx).cos() * (pi * yy).sin();
    } }
    let ke = |f: &FluidSim| -> f64 {
        let mut e = 0.0f64;
        for y in 1..h - 1 { for x in 1..w - 1 {
            let (vx, vy) = f.cell_velocity(x, y);
            e += (vx * vx + vy * vy) as f64;
        } }
        0.5 * e
    };
    f.step(dt, 300); // projette le champ initial
    let e0 = ke(&f);
    for _ in 0..(t_end / dt) as usize { f.step(dt, 300); }
    let k2 = 2.0 * pi * pi / (l * l);
    (ke(&f) / e0, (-2.0 * nu * k2 * t_end).exp() as f64)
}

#[test]
#[ignore]
fn vortex_energy_loss() {
    // L/U = 1 s : t_end = 5 s = 5 temps de retournement
    for &(nu, label) in &[(0.0f32, "ν = 0     "), (1e-5, "ν = 1e-5  ")] {
        for &n in &[32usize, 64] {
            let dt = 0.01 * 32.0 / n as f32; // CFL d'advection constant
            let (sl, exact) = vortex_ke(n, nu, false, 5.0, dt);
            let (mc, _) = vortex_ke(n, nu, true, 5.0, dt);
            println!("{label} N={n:2}: E/E0 après 5 s  exact {exact:.4} | semi-lagr. {sl:.4} (perte num. {:.1}%) | MacCormack {mc:.4} (perte num. {:.1}%)",
                     100.0 * (exact - sl) / exact, 100.0 * (exact - mc) / exact);
        }
    }
}


// ---- Traceur : un disque advecté par un tourbillon à parois glissantes (rotation solide + cisaillement).
// Mesure l'étalement (écart-type spatial) et les bornes : l'ordre 2 limité doit étaler beaucoup moins que
// l'upwind, sans jamais sortir de [0, 1] (TVD).
fn blob_run(hi: bool) -> (f32, f32, f32, f64) {
    let mut mat = FluidMaterial::water();
    mat.tracer_diffusion = 0.0;
    mat.viscosity = 0.0;
    let (n, l, u0) = (64usize, 0.1f32, 0.1f32);
    let dx = l / n as f32;
    let (w, h) = (n + 2, n + 2);
    let mut f = cpu_sim(w, h, dx, mat, 0.0);
    f.no_slip = false;
    f.high_order_scalars = hi;
    let pi = std::f32::consts::PI;
    let nf = n as f32;
    for y in 1..h - 1 { for x in 0..w - 1 {
        let (xx, yy) = (x as f32 / nf, (y as f32 - 0.5) / nf);
        f.velocity_x[x + y * w] = (u0 / dx) * (pi * xx).sin() * (pi * yy).cos();
    } }
    for y in 0..h - 1 { for x in 1..w - 1 {
        let (xx, yy) = ((x as f32 - 0.5) / nf, y as f32 / nf);
        f.velocity_y[x + y * w] = -(u0 / dx) * (pi * xx).cos() * (pi * yy).sin();
    } }
    let (cx, cy, r) = (0.3f32 * nf + 0.5, 0.5f32 * nf + 0.5, 6.0f32);
    for y in 1..h - 1 { for x in 1..w - 1 {
        let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
        f.tracer[x + y * w] = if d < r { 1.0 } else { 0.0 };
    } }
    f.step(0.01, 300);
    let mass = |f: &FluidSim| -> f64 { (1..h - 1).flat_map(|y| (1..w - 1).map(move |x| (x, y))).map(|(x, y)| f.tracer[x + y * w] as f64).sum() };
    let m0 = mass(&f);
    for _ in 0..500 { f.step(0.01, 100); } // 5 s ~ 0,5 tour de la cellule
    let t: Vec<f32> = (1..h - 1).flat_map(|y| (1..w - 1).map(move |x| (x, y))).map(|(x, y)| f.tracer[x + y * w]).collect();
    let (mn, mx) = t.iter().fold((f32::MAX, f32::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
    // ordre 2 du pic : fraction de cellules > 0,5 (le disque initial en compte ~ pi r^2 = 113)
    let big = t.iter().filter(|&&v| v > 0.5).count() as f32;
    (mn, mx, big, mass(&f) / m0)
}

#[test]
fn tracer_blob() {
    for &hi in &[false, true] {
        let (mn, mx, big, m) = blob_run(hi);
        assert!(mn > -1e-3 && mx < 1.001 && (m - 1.0).abs() < 1e-4, "bornes/masse du traceur");
        if hi { assert!(big > 70.0, "le MUSCL doit garder le disque net"); }
        println!("{} : min {mn:.4} max {mx:.4}  cellules>0.5 : {big}  (initial ~113)  masse finale/initiale {m:.5}",
                 if hi { "MUSCL  " } else { "upwind " });
    }
}

// ---- Chauffage visqueux : tourbillon dans une boîte fermée de gaz (low-Mach). L'énergie cinétique perdue
// doit se retrouver en chaleur : ΔP_th·V/(γ−1) = énergie interne gagnée. On compare à la perte d'énergie
// cinétique mesurée et, en glissement libre, à la décroissance exacte E0·(1 − exp(−2νk²t)).
fn heating_balance(no_slip: bool, heating: bool, t0: f32) -> (f64, f64, f64, f64) {
    let mut mat = FluidMaterial::air();
    mat.viscosity = 5e-4 * 293.15 / t0; // gaz chaud : ν = μ/ρ bien plus grande, on garde une décroissance mesurable
    mat.conductivity = 0.0; // pas de conduction : isole l'effet
    let (n, l, u0, nu) = (48usize, 0.1f32, 0.3f32, mat.viscosity); // CFL ~ 0,3 : peu de dissipation numérique
    let dx = l / n as f32;
    let (w, h) = (n + 2, n + 2);
    let mut f = cpu_sim(w, h, dx, mat, 0.0);
    f.no_slip = no_slip;
    f.viscous_heating = heating;
    f.low_mach = true;
    f.temperature.fill(t0);
    let pi = std::f32::consts::PI;
    let nf = n as f32;
    for y in 1..h - 1 { for x in 0..w - 1 {
        let (xx, yy) = (x as f32 / nf, (y as f32 - 0.5) / nf);
        f.velocity_x[x + y * w] = (u0 / dx) * (pi * xx).sin() * (pi * yy).cos();
    } }
    for y in 0..h - 1 { for x in 1..w - 1 {
        let (xx, yy) = ((x as f32 - 0.5) / nf, y as f32 / nf);
        f.velocity_y[x + y * w] = -(u0 / dx) * (pi * xx).cos() * (pi * yy).sin();
    } }
    let rho = (f.material.rho_ambient() * f.material.t_ref / t0) as f64; // ρ(t0, P_ref)
    let ke = |f: &FluidSim| -> f64 {
        let mut e = 0.0f64;
        for y in 1..h - 1 { for x in 1..w - 1 {
            let (vx, vy) = f.cell_velocity(x, y);
            e += (vx * vx + vy * vy) as f64;
        } }
        0.5 * rho * e * (dx as f64).powi(2) * (dx as f64).powi(2) // ρ·Σ½|u|²·dx²(aire)·1 m de profondeur, u en cellules/s·dx
    };
    f.step(0.001, 300);
    let (e0, p0) = (ke(&f), f.p_th_acc); // p_th_acc : p_th en f64 (l'ulp f32 de p_th vaut 0,008 Pa)
    let t_end = 1.0;
    for _ in 0..(t_end / 0.002) as usize { f.step(0.002, 100); }
    let e1 = ke(&f);
    let vol = (n as f64 * dx as f64).powi(2);
    let gamma = f.material.gamma() as f64;
    let heat = (f.p_th_acc - p0) * vol / (gamma - 1.0);
    let k2 = 2.0 * (pi as f64).powi(2) / (l as f64).powi(2);
    let exact_loss = e0 * (1.0 - (-2.0 * nu as f64 * k2 * t_end).exp());
    (e0 - e1, heat, exact_loss, e0)
}

#[test]
fn viscous_heating_balance_hot() {
    // gaz à 600 K : μ(T) de Sutherland et ρ(T) changent ν ; le bilan énergétique doit rester fermé
    for &ns in &[false, true] {
        let (lost, heat, _, e0) = heating_balance(ns, true, 600.0);
        println!("T0 = 600 K, no_slip={ns}: E0 {e0:.3e} perte {lost:.3e} chaleur {heat:.3e} ({:.1}%)", 100.0 * heat / lost);
        assert!((heat / lost - 1.0).abs() < 0.1, "bilan visqueux à chaud, no_slip={ns}");
    }
}

#[test]
fn viscous_heating_balance() {
    for &ns in &[false, true] {
        for &hv in &[false, true] {
            let (lost, heat, exact, e0) = heating_balance(ns, hv, 293.15);
            if hv {
                assert!((heat / lost - 1.0).abs() < 0.1, "no_slip={ns} : chaleur {heat:e} vs perte d'EC {lost:e}");
            } else {
                assert!(heat.abs() < 1e-3 * lost, "sans chauffage visqueux la température ne doit pas bouger");
            }
            println!("no_slip={ns:5} chauffage={hv:5}: E0 {e0:.3e} J | perte EC {lost:.3e} ({:.1}%) | chaleur rendue {heat:.3e} (= {:.1}% de la perte) | perte visqueuse exacte (glissement libre) {exact:.3e}",
                     100.0 * lost / e0, 100.0 * heat / lost);
        }
    }
}


// ---- Lois de propriétés : valeurs de la littérature (air : tables NIST/CRC ; eau : tables usuelles)
#[test]
fn property_laws() {
    let air = FluidMaterial::air();
    let close = |a: f32, b: f32, tol: f32| ((a - b) / b).abs() < tol;
    // à T_ref, μ vaut exactement ν_ref·ρ_ambient (l'UI règle ν_ref)
    assert!(close(air.mu(air.t_ref, true), air.viscosity * air.rho_ambient(), 1e-5));
    assert!(close(air.k_fluid(air.t_ref, true), air.conductivity, 1e-5));
    // air : μ(400 K) = 2,30e-5, μ(600 K) = 3,04e-5, μ(1000 K) = 4,2e-5 Pa·s ; k(400 K) = 0,0338, k(1000 K) = 0,0667 W/(m·K)
    assert!(close(air.mu(400.0, true), 2.30e-5, 0.02), "mu(400) = {}", air.mu(400.0, true));
    assert!(close(air.mu(600.0, true), 3.04e-5, 0.02), "mu(600) = {}", air.mu(600.0, true));
    assert!(close(air.mu(1000.0, true), 4.2e-5, 0.02), "mu(1000) = {}", air.mu(1000.0, true));
    assert!(close(air.k_fluid(400.0, true), 0.0338, 0.02), "k(400) = {}", air.k_fluid(400.0, true));
    assert!(close(air.k_fluid(1000.0, true), 0.0667, 0.02), "k(1000) = {}", air.k_fluid(1000.0, true));
    // ν = μ/ρ à 600 K (P_ref) : ~ 5,2e-5 m²/s (3,04e-5 / 0,586)
    let nu600 = air.mu(600.0, true) / air.density(600.0, air.p_ref);
    assert!(close(nu600, 5.2e-5, 0.03), "nu(600) = {nu600}");
    // μ constante quand les propriétés sont figées
    assert_eq!(air.mu(900.0, false), air.mu(air.t_ref, false));
    // eau : μ(40 °C) = 0,653e-3, μ(80 °C) = 0,355e-3 Pa·s
    let w = FluidMaterial::water();
    assert!(close(w.mu(313.15, true), 0.653e-3, 0.03), "mu eau 40C = {}", w.mu(313.15, true));
    assert!(close(w.mu(353.15, true), 0.355e-3, 0.04), "mu eau 80C = {}", w.mu(353.15, true));
    assert!(close(w.k_fluid(313.15, true), 0.631, 0.02), "k eau 40C = {}", w.k_fluid(313.15, true));
}

// ---- Décroissance exacte d'un tourbillon libre-glissant dans un gaz chaud : E(t)/E0 = exp(−2·ν(T)·k²·t),
// avec ν(T) = μ(T)/ρ(T). Compare propriétés variables et constantes.
fn hot_vortex_decay(t0: f32, variable: bool) -> (f64, f64) {
    let mut mat = FluidMaterial::air();
    mat.viscosity = 4.5e-4; // ν_ref grande : décroissance rapide
    mat.conductivity = 0.0;
    let (n, l, u0) = (48usize, 0.1f32, 0.2f32);
    let dx = l / n as f32;
    let (w, h) = (n + 2, n + 2);
    let mut f = cpu_sim(w, h, dx, mat, 0.0);
    f.no_slip = false;
    f.low_mach = true;
    f.variable_properties = variable;
    f.temperature.fill(t0);
    let pi = std::f32::consts::PI;
    let nf = n as f32;
    for y in 1..h - 1 { for x in 0..w - 1 {
        f.velocity_x[x + y * w] = (u0 / dx) * (pi * x as f32 / nf).sin() * (pi * (y as f32 - 0.5) / nf).cos();
    } }
    for y in 0..h - 1 { for x in 1..w - 1 {
        f.velocity_y[x + y * w] = -(u0 / dx) * (pi * (x as f32 - 0.5) / nf).cos() * (pi * y as f32 / nf).sin();
    } }
    let ke = |f: &FluidSim| -> f64 {
        let mut e = 0.0f64;
        for y in 1..h - 1 { for x in 1..w - 1 { let (a, b) = f.cell_velocity(x, y); e += (a * a + b * b) as f64; } }
        e
    };
    f.step(0.0005, 300);
    let e0 = ke(&f);
    let t_end = 0.25;
    for _ in 0..(t_end / 0.001) as usize { f.step(0.001, 100); }
    let nu = if variable {
        f.material.mu(t0, true) as f64 / f.material.density(t0, f.material.p_ref) as f64
    } else {
        f.material.viscosity as f64
    };
    let k2 = 2.0 * (pi as f64).powi(2) / (l as f64).powi(2);
    (ke(&f) / e0, (-2.0 * nu * k2 * t_end as f64).exp())
}

#[test]
fn viscosity_nu_equals_mu_over_rho() {
    for &(t0, variable) in &[(293.15f32, true), (600.0, true), (600.0, false)] {
        let (meas, exact) = hot_vortex_decay(t0, variable);
        println!("T0 = {t0} K, propriétés variables = {variable}: E/E0 mesurée {meas:.4}, exacte exp(−2νk²t) {exact:.4}");
        assert!((meas / exact - 1.0).abs() < 0.08, "T0 = {t0}, variable = {variable} : {meas} vs {exact}");
    }
}

// ---- Conduction avec k(T) : un mode cos(πx/L) de 1 K autour de T0 décroît en exp(−α(T0)·(π/L)²·t),
// α = k(T0)/(ρ(T0)·cp) (petite perturbation : conduction linéarisée).
fn conduction_decay(t0: f32, variable: bool) -> (f64, f64) {
    let mut mat = FluidMaterial::air();
    mat.viscosity = 0.0;
    let (n, l) = (40usize, 0.04f32);
    let dx = l / n as f32;
    let (w, h) = (n + 2, n + 2);
    let mut f = cpu_sim(w, h, dx, mat, 0.0);
    f.low_mach = true;
    f.variable_properties = variable;
    let pi = std::f32::consts::PI;
    for y in 0..h { for x in 0..w {
        f.temperature[x + y * w] = t0 + 1.0 * (pi * (x as f32 - 0.5) / n as f32).cos();
    } }
    let amp = |f: &FluidSim| -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for y in 1..h - 1 { for x in 1..w - 1 {
            let c = (pi * (x as f32 - 0.5) / n as f32).cos() as f64;
            num += (f.temperature[x + y * w] as f64 - t0 as f64) * c;
            den += c * c;
        } }
        num / den
    };
    f.step(0.001, 200);
    let a0 = amp(&f);
    let t_end = 1.5;
    for _ in 0..(t_end / 0.005) as usize { f.step(0.005, 100); }
    let (k, rho) = (f.material.k_fluid(t0, variable) as f64, f.material.density(t0, f.material.p_ref) as f64);
    let alpha = k / (rho * f.material.cp as f64);
    (amp(&f) / a0, (-alpha * (pi as f64 / l as f64).powi(2) * t_end as f64).exp())
}

#[test]
fn conduction_uses_k_of_t() {
    for &(t0, variable) in &[(293.15f32, true), (700.0, true), (700.0, false)] {
        let (meas, exact) = conduction_decay(t0, variable);
        println!("T0 = {t0} K, propriétés variables = {variable}: A/A0 mesurée {meas:.4}, exacte {exact:.4}");
        assert!((meas / exact - 1.0).abs() < 0.06, "T0 = {t0}, variable = {variable} : {meas} vs {exact}");
    }
}

// ---- Robustesse : gaz à 1800 K (ν = μ/ρ ~ 25× la référence) brassé violemment, grands pas de temps.
#[test]
fn hot_variable_properties_stable() {
    let mat = FluidMaterial::air();
    let (n, dx) = (60usize, 0.005f32);
    let (w, h) = (n + 2, n + 2);
    let mut f = cpu_sim(w, h, dx, mat, 9.81);
    f.low_mach = true;
    // masse des cellules intérieures seulement : les cellules fantômes du bord suivent l'EOS, pas la masse
    let mass = |f: &FluidSim| -> f64 { (1..h - 1).flat_map(|y| (1..w - 1).map(move |x| f.rho[x + y * w] as f64)).sum() };
    for y in 20..40 { for x in 20..40 { f.temperature[x + y * w] = 1800.0; } }
    f.step(0.002, 100);
    let m0 = mass(&f);
    for k in 0..300 {
        // « pinceau » : poussée violente à position tournante
        let (cx, cy) = (10 + (k * 7) % 40, 10 + (k * 13) % 40);
        for y in cy - 3..cy + 3 { for x in cx - 3..cx + 3 {
            f.velocity_x[x + y * w] += 8.0 / dx * 0.05;
            f.velocity_y[x + y * w] -= 5.0 / dx * 0.05;
        } }
        f.step(0.016, 30);
        assert!(finite(&f), "non fini au pas {k}");
    }
    let drift = (mass(&f) / m0 - 1.0).abs();
    let tmax = f.temperature.iter().cloned().fold(0.0f32, f32::max);
    println!("brassage à chaud : dérive de masse {drift:.2e}, T max {tmax:.0} K, ν_max/ν_ref = {:.1}", f.nu_max_phys / f.material.viscosity);
    assert!(drift < 5e-3, "masse : {drift}");
    assert!(tmax < 2500.0, "T max {tmax}");
}

// =====================================================================================================
// Conservation de la masse en domaine OUVERT et conditions aux bords
// =====================================================================================================

fn bnd(kind: BoundaryType, vx_ms: f32, vy_ms: f32, tracer: f32, temp: f32) -> Boundary {
    Boundary { boundary_type: kind, velocity_x_ms: vx_ms, velocity_y_ms: vy_ms, tracer, offset: 0, temperature: temp }
}

// masse (par mètre de profondeur) des cellules intérieures non solides
fn interior_mass(f: &FluidSim) -> f64 {
    let (w, h) = (f.width, f.height);
    let mut m = 0.0f64;
    for y in 1..h - 1 { for x in 1..w - 1 {
        if f.solids[x + y * w] != CellType::Wall { m += f.rho[x + y * w] as f64; }
    } }
    m * (f.dx as f64).powi(2)
}

// Canal gauche -> droite : Inlet à gauche (vitesse, T, tracer), Outlet à droite, murs haut/bas.
fn channel(t_in: f32, v_in: f32, low_mach: bool, obstacle: bool) -> FluidSim {
    let (w, h, dx) = (64usize, 26usize, 0.01f32);
    let mat = FluidMaterial::air();
    let mut f = cpu_sim(w, h, dx, mat, 0.0);
    f.low_mach = low_mach;
    f.left_boundary = bnd(BoundaryType::Inlet, v_in, 0.0, 1.0, t_in);
    f.right_boundary = bnd(BoundaryType::Outlet, 0.0, 0.0, 0.0, mat.t_ref);
    if obstacle {
        for y in 8..16 { for x in 20..24 { f.set_cell(x, y, CellType::Wall, None); } }
    }
    f
}

// Bilan de masse EXACT en domaine ouvert : M(t) − M(0) = flux net aux bords intégré par le schéma lui-même
// (`mass_boundary_flux`, mêmes flux que la mise à jour des cellules). Tout écart vient donc de ce qui modifie ρ
// À L'INTÉRIEUR du domaine sans passer par un flux (rappel ρ -> EOS, garde-fous).
fn channel_mass_error(t_in: f32, v_in: f32, obstacle: bool) -> (f64, f64, f64) {
    let mut f = channel(t_in, v_in, true, obstacle);
    // démarrage : le premier pas part d'un fluide au repos et d'un Inlet à pleine vitesse, la projection n'a pas
    // convergé (Σu_sortie << Σu_entrée) et le gaz est brièvement comprimé : on laisse l'écoulement s'établir.
    for _ in 0..150 { f.step(0.004, 200); }
    let (m0, q0) = (interior_mass(&f), f.mass_boundary_flux);
    for _ in 0..600 { f.step(0.004, 200); }
    assert!(finite(&f));
    let (dm, dq) = (interior_mass(&f) - m0, f.mass_boundary_flux - q0);
    (dm, dq, (dm - dq).abs() / m0)
}

#[test]
fn open_channel_mass_budget() {
    // Inlet ambiant / chaud / froid, avec et sans obstacle ; Outlet à droite
    for &(t_in, obstacle) in &[(293.15f32, false), (600.0, false), (600.0, true), (200.0, true), (1000.0, false)] {
        let (dm, dq, err) = channel_mass_error(t_in, 0.5, obstacle);
        println!("canal T_in={t_in:6.1} K obstacle={obstacle:5}: ΔM {dm:+.4e}  ∫flux {dq:+.4e}  |écart|/M {err:.1e}");
        assert!(err < 2e-4, "bilan de masse T_in={t_in} obstacle={obstacle} : ΔM = {dm:e} vs ∫flux = {dq:e}");
    }
}

// ---- Conditions aux bords : Inlet à gauche, Outlet à droite (régime établi)
#[test]
fn inlet_outlet_steady_state() {
    let (t_in, v_in) = (450.0f32, 0.5f32);
    let mut f = channel(t_in, v_in, true, false);
    let (w, h) = (f.width, f.height);
    for _ in 0..800 { f.step(0.004, 200); }
    assert!(finite(&f));
    // 1. vitesse normale d'entrée exacte (m/s -> cellules/s), sur toutes les lignes
    for y in 1..h - 1 {
        assert!((f.velocity_x[y * w] * f.dx - v_in).abs() < 1e-6, "ligne {y} : u_in = {}", f.velocity_x[y * w] * f.dx);
    }
    // 2. débit volumique sortant = entrant (écoulement établi à 450 K partout : plus de dilatation en transit)
    let q_in: f64 = (1..h - 1).map(|y| f.velocity_x[y * w] as f64).sum();
    let q_out: f64 = (1..h - 1).map(|y| f.velocity_x[y * w + w - 2] as f64).sum();
    println!("Σu_in {q_in:.4}  Σu_out {q_out:.4}  (rapport {:.5})", q_out / q_in);
    assert!((q_out / q_in - 1.0).abs() < 2e-3, "débit : {q_out} vs {q_in}");
    // 3. fantômes d'entrée = valeurs imposées ; intérieur rempli de gaz d'entrée (T, tracer), sans dépassement
    let mid = h / 2;
    assert_eq!(f.temperature[mid * w], t_in);
    assert_eq!(f.tracer[mid * w], 1.0);
    let (mut tmin, mut tmax, mut trmin, mut trmax) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for y in 1..h - 1 { for x in 1..w - 1 {
        let i = x + y * w;
        tmin = tmin.min(f.temperature[i]); tmax = tmax.max(f.temperature[i]);
        trmin = trmin.min(f.tracer[i]); trmax = trmax.max(f.tracer[i]);
    } }
    println!("T ∈ [{tmin:.3}, {tmax:.3}] (inlet {t_in}), tracer ∈ [{trmin:.5}, {trmax:.5}]");
    assert!(tmin > t_in - 0.5 && tmax < t_in + 0.5, "T : [{tmin}, {tmax}]");
    assert!(trmin > 0.999 && trmax < 1.0001, "tracer : [{trmin}, {trmax}]");
    // 4. P_th = P_ref en domaine ouvert, et pression dynamique bornée (pas de réflexion/pompage à la sortie)
    assert!((f.p_th - f.material.p_ref).abs() < 1.0, "p_th = {}", f.p_th);
}

// ---- Rampe `offset` : la vitesse d'entrée vaut v·min(distance au coin / offset, 1)
#[test]
fn inlet_offset_ramp() {
    let mut f = channel(293.15, 0.5, true, false);
    f.left_boundary.offset = 6;
    let (w, h) = (f.width, f.height);
    for _ in 0..100 { f.step(0.004, 150); }
    for y in 1..h - 1 {
        let d = y.min(h - 1 - y) as f32;
        let expected = 0.5 * (d / 6.0).clamp(0.0, 1.0);
        let got = f.velocity_x[y * w] * f.dx;
        assert!((got - expected).abs() < 1e-6, "ligne {y} : {got} au lieu de {expected}");
    }
}

// ---- Reflux par la sortie : un Outlet qui AVALE du fluide ramène le gaz ambiant (T_ref, tracer 0)
#[test]
fn outlet_backflow_brings_ambient() {
    let mut f = channel(293.15, -0.3, true, false); // « Inlet » gauche en aspiration : u = −0,3 m/s
    f.left_boundary = bnd(BoundaryType::Inlet, -0.3, 0.0, 1.0, 293.15);
    let (w, h) = (f.width, f.height);
    for y in 1..h - 1 { for x in 1..w - 1 { f.temperature[x + y * w] = 500.0; f.tracer[x + y * w] = 1.0; } }
    let (mut m_prev, mut q_prev) = (0.0, 0.0);
    for k in 0..3500 {
        f.step(0.004, 200);
        if k == 100 { m_prev = interior_mass(&f); q_prev = f.mass_boundary_flux; }
        assert!(finite(&f), "NaN au pas {k}");
    }
    let (mut tmin, mut tmax, mut trmax) = (f32::MAX, f32::MIN, f32::MIN);
    for y in 1..h - 1 { for x in 1..w - 1 {
        let i = x + y * w;
        tmin = tmin.min(f.temperature[i]); tmax = tmax.max(f.temperature[i]); trmax = trmax.max(f.tracer[i]);
    } }
    println!("reflux : T ∈ [{tmin:.2}, {tmax:.2}], tracer max {trmax:.5}");
    assert!(tmin > 292.0 && tmax < 293.5, "le gaz de la sortie doit remplacer le gaz chaud : T ∈ [{tmin}, {tmax}]");
    assert!(trmax < 1e-3, "tracer résiduel {trmax}");
    let (dm, dq) = (interior_mass(&f) - m_prev, f.mass_boundary_flux - q_prev);
    println!("reflux : ΔM {dm:+.4e}, ∫flux {dq:+.4e}");
    assert!((dm - dq).abs() < 2e-4 * m_prev, "bilan de masse avec reflux");
}

// ---- Boîte fermée + un Inlet : le gaz s'accumule (la compression l'échauffe), bilan de masse et d'énergie exacts
#[test]
fn closed_box_with_inlet_compresses() {
    let mat = FluidMaterial::air();
    let mut f = cpu_sim(40, 30, 0.01, mat, 0.0);
    f.low_mach = true;
    f.left_boundary = bnd(BoundaryType::Inlet, 0.05, 0.0, 0.0, mat.t_ref);
    for _ in 0..40 { f.step(0.004, 200); }
    let (m0, q0, p0) = (interior_mass(&f), f.mass_boundary_flux, f.p_th);
    for _ in 0..300 { f.step(0.004, 200); }
    assert!(finite(&f));
    let (dm, dq) = (interior_mass(&f) - m0, f.mass_boundary_flux - q0);
    // Remplir un réservoir rigide : l'enthalpie entrante cp·T_in·dm devient énergie interne, donc ΔP = γ·P0·ΔM/M0·(T_in/T0)
    // (et non P0·ΔM/M0 : la compression échauffe le gaz).
    let gamma = f.material.gamma() as f64;
    let (dp, expected_dp) = ((f.p_th - p0) as f64, gamma * p0 as f64 * dm / m0);
    println!("boîte fermée + Inlet : ΔM {dm:+.4e} ∫flux {dq:+.4e} ΔP {dp:+.3} Pa attendu {expected_dp:+.3} Pa");
    assert!((dm - dq).abs() < 1e-4 * m0.max(dq.abs()), "masse : {dm:e} vs {dq:e}");
    assert!(dm > 0.0 && ((dp - expected_dp) / expected_dp).abs() < 0.03, "ΔP_th = γ·P0·ΔM/M0 (enthalpie entrante)");
}

// ---- Symétrie : le même écoulement orienté gauche->droite, droite->gauche, haut->bas, bas->haut doit donner
// les mêmes champs, une fois les axes ramenés. Détecte toute erreur d'indice/signe dans les 4 bords.
fn oriented_run(dir: usize) -> (FluidSim, impl Fn(usize, usize) -> (usize, usize)) {
    let (n, dx) = (40usize, 0.01f32);
    let (w, h) = (n + 2, n + 2);
    let mut mat = FluidMaterial::air();
    // Re ~ 30 : écoulement laminaire STATIONNAIRE (sinon le sillage de l'obstacle est instable et amplifie
    // n'importe quel écart d'arrondi : la comparaison ne mesurerait plus que le chaos)
    mat.viscosity = 5e-4;
    let mut f = cpu_sim(w, h, dx, mat, 0.0);
    f.low_mach = true;
    let map = move |x: usize, y: usize| -> (usize, usize) {
        match dir { 0 => (x, y), 1 => (w - 1 - x, y), 2 => (y, x), _ => (y, w - 1 - x) }
    };
    let inlet = |vx: f32, vy: f32| bnd(BoundaryType::Inlet, vx, vy, 1.0, 450.0);
    let outlet = bnd(BoundaryType::Outlet, 0.0, 0.0, 0.0, mat.t_ref);
    match dir {
        0 => { f.left_boundary = inlet(0.4, 0.0); f.right_boundary = outlet; }
        1 => { f.right_boundary = inlet(-0.4, 0.0); f.left_boundary = outlet; }
        2 => { f.top_boundary = inlet(0.0, 0.4); f.bottom_boundary = outlet; }
        _ => { f.bottom_boundary = inlet(0.0, -0.4); f.top_boundary = outlet; }
    }
    for y in 12..22 { for x in 12..16 { let (a, b) = map(x, y); f.set_cell(a, b, CellType::Wall, None); } }
    let iters: usize = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
    for _ in 0..500 { f.step(0.005, iters); }
    (f, map)
}

#[test]
fn four_orientations_are_equivalent() {
    let (base, _) = oriented_run(0);
    let (w, h) = (base.width, base.height);
    let flow = |f: &FluidSim, dir: usize, x: usize, y: usize| -> (f32, f32) {
        let (vx, vy) = f.cell_velocity(x, y);
        match dir { 0 => (vx, vy), 1 => (-vx, vy), 2 => (vy, vx), _ => (-vy, vx) }
    };
    let vmax = (1..h - 1).flat_map(|y| (1..w - 1).map(move |x| (x, y))).map(|(x, y)| flow(&base, 0, x, y).0.abs()).fold(0.0f32, f32::max);
    for dir in 1..4 {
        let (f, map) = oriented_run(dir);
        assert!(finite(&f));
        let (mut dt_max, mut dtr_max, mut dv_max, mut dt_up) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for y in 1..h - 1 { for x in 1..w - 1 {
            if base.solids[x + y * w] == CellType::Wall { continue; }
            let (a, b) = map(x, y);
            let (i, j) = (x + y * w, a + b * w);
            let dti = (base.temperature[i] - f.temperature[j]).abs();
            dt_max = dt_max.max(dti);
            if x < 11 { dt_up = dt_up.max(dti); } // en amont de l'obstacle : aucune excuse physique pour un écart
            dtr_max = dtr_max.max((base.tracer[i] - f.tracer[j]).abs());
            let (p, q) = (flow(&base, 0, x, y), flow(&f, dir, a, b));
            dv_max = dv_max.max((p.0 - q.0).abs()).max((p.1 - q.1).abs());
        } }
        println!("orientation {dir} vs gauche->droite : max|ΔT| {dt_max:.3e} K (amont de l'obstacle {dt_up:.1e}), max|Δtracer| {dtr_max:.2e}, max|Δv|/v_max {:.2e}", dv_max / vmax);
        // le sillage de l'obstacle est la seule zone où des arrondis différents (ordre rouge/noir du SOR) laissent
        // un écart de quelques dixièmes de kelvin (0,3 % de ΔT) ; ailleurs les orientations coïncident à 1e-2 K
        assert!(dt_up < 1e-2, "orientation {dir} : écart de T en amont {dt_up}");
        assert!(dt_max < 1.0 && dtr_max < 1e-2 && dv_max / vmax < 5e-3, "orientation {dir} : ΔT {dt_max}, Δtracer {dtr_max}, Δv {}", dv_max / vmax);
    }
}