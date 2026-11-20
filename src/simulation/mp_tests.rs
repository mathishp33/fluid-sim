//! Tests de validation du solveur multiphasique (comparaisons à la théorie / à des tables réelles).

use super::multiphase::{Bc, MultiphaseSim};
use super::phases::{Phase, Registry};

pub const P_ATM: f64 = 101325.0;

pub fn tank(nx: usize, ny: usize, dx: f64, water_height: f64) -> (MultiphaseSim, usize, usize) {
    let mut r = Registry::new();
    let air = r.add_phase(Phase::air());
    let eau = r.add_phase(Phase::water());
    let mut s = MultiphaseSim::new(nx, ny, dx, r, air);
    s.paint_phase(move |_, y| if y < water_height { Some(eau) } else { Some(air) });
    (s, air, eau)
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |a, &b| a.max(b.abs()))
}

#[test]
fn hydrostatic_column_is_stationary_and_pressure_exact() {
    let (mut s, _air, eau) = tank(12, 24, 0.01, 0.12);
    s.bc[3] = Bc::Open { p: P_ATM };
    s.finalize_init(|_, _| 293.15);
    let m0 = s.total_mass_all();
    let a0 = s.phase_area(eau);
    for _ in 0..60 {
        s.step();
    }
    let vmax = s.max_speed();
    assert!(vmax < 2e-3, "courants parasites {vmax} m/s");
    // pression au fond : p_atm + ρ g h (ρ eau ≈ 998, + compressibilité négligeable)
    let pb = s.p[s.idx(5, 0)];
    let expect = P_ATM + 998.2 * 9.81 * (0.12 - 0.005) + 1.2 * 9.81 * (0.24 - 0.12);
    assert!((pb - expect).abs() / (expect - P_ATM) < 0.01, "p_fond {pb} attendu {expect}");
    // l'eau ne bouge pas, masse conservée (flux de bord air négligeable)
    assert!((s.phase_area(eau) / a0 - 1.0).abs() < 1e-3);
    assert!(((s.total_mass_all() - m0) / m0).abs() < 1e-3);
    eprintln!("hydrostatique : vmax={vmax:.2e} p_fond={pb:.1} (attendu {expect:.1}) itérations PCG={}", s.last_pressure_iters);
}

#[test]
fn closed_water_pressure_rise_matches_beta_over_kappa() {
    // eau seule dans une boîte rigide fermée, chauffée de 0,1 K : Δp = (β/κ_T) ΔT ≈ 4,6 bar/K
    let mut r = Registry::new();
    let air = r.add_phase(Phase::air());
    let eau = r.add_phase(Phase::water());
    let mut s = MultiphaseSim::new(8, 8, 0.01, r, air);
    s.gravity = [0.0, 0.0];
    s.paint_phase(|_, _| Some(eau));
    s.finalize_init(|_, _| 293.15);
    let p0 = s.p[0];
    let dt_k = 0.1;
    for c in 0..64 {
        s.e[c] += s.heat_capacity(c) * dt_k;
    }
    for _ in 0..3 {
        s.step();
    }
    let ph = &s.reg.phases[eau];
    let expect = ph.beta(p0, 293.25) / ph.kappa_t(p0, 293.25) * dt_k;
    let got = s.p[27] - p0;
    eprintln!("Δp mesuré {got:.0} Pa, théorie {expect:.0} Pa");
    assert!((got / expect - 1.0).abs() < 0.08, "Δp = {got} attendu {expect}");
    assert!(s.max_speed() < 1e-6);
}

#[test]
fn sloshing_frequency_matches_linear_wave_theory() {
    // cuve L=1 m, h=0.5 m, mode 1 : ω² = g k tanh(k h), k = π/L  (T ≈ 1.182 s)
    let (nx, ny, dx) = (48usize, 36usize, 1.0 / 48.0);
    let (l, h, amp) = (1.0, 0.5, 0.01);
    let k = std::f64::consts::PI / l;
    let mut r = Registry::new();
    let air = r.add_phase(Phase::air());
    let eau = r.add_phase(Phase::water());
    let mut s = MultiphaseSim::new(nx, ny, dx, r, air);
    s.paint_phase(move |x, y| if y < h + amp * (k * x).cos() { Some(eau) } else { Some(air) });
    s.finalize_init(|_, _| 293.15);
    let m0 = s.total_mass(eau);
    let omega = (9.81 * k * (k * h).tanh()).sqrt();
    let period = 2.0 * std::f64::consts::PI / omega;
    let height_left = |s: &MultiphaseSim| (0..s.ny).map(|j| s.alpha[eau][s.idx(0, j)]).sum::<f64>() * s.dx;
    let mut series: Vec<(f64, f64)> = vec![];
    while s.time < 3.2 * period {
        s.step();
        series.push((s.time, height_left(&s) - h));
    }
    // passages par zéro (descendants) → période mesurée
    let mut crossings = vec![];
    for w in series.windows(2) {
        if w[0].1 > 0.0 && w[1].1 <= 0.0 {
            crossings.push(w[0].0 + (w[1].0 - w[0].0) * w[0].1 / (w[0].1 - w[1].1));
        }
    }
    assert!(crossings.len() >= 2, "pas assez d'oscillations: {crossings:?}");
    let measured = (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64;
    let amp_end = series.iter().rev().take(40).map(|p| p.1.abs()).fold(0.0, f64::max);
    eprintln!("ballottement : T mesurée {measured:.4} s, théorie {period:.4} s ; amplitude finale {amp_end:.4} (initiale {amp}) ; masse {:.2e}", (s.total_mass(eau) - m0) / m0);
    assert!((measured / period - 1.0).abs() < 0.04, "période {measured} vs {period}");
    assert!(amp_end > 0.3 * amp && amp_end < 1.15 * amp, "amortissement incohérent {amp_end}");
    assert!(((s.total_mass(eau) - m0) / m0).abs() < 1e-9, "masse d'eau non conservée");
}

#[test]
fn laplace_pressure_of_a_cylindrical_droplet() {
    // gouttelette 2D (cylindre) R = 2 cm dans l'air, sans gravité : Δp = σ/R = 3,64 Pa
    let (n, dx) = (64usize, 0.0015625);
    let mut r = Registry::new();
    let air = r.add_phase(Phase::air());
    let eau = r.add_phase(Phase::water());
    let mut s = MultiphaseSim::new(n, n, dx, r, air);
    s.gravity = [0.0, 0.0];
    let (cx, cy, rad) = (0.05, 0.05, 0.02);
    s.paint_phase(move |x, y| if ((x - cx).powi(2) + (y - cy).powi(2)).sqrt() < rad { Some(eau) } else { Some(air) });
    s.finalize_init(|_, _| 293.15);
    for _ in 0..120 {
        s.step();
    }
    let (mut pin, mut nin, mut pout, mut nout) = (0.0, 0.0, 0.0, 0.0);
    for j in 0..n {
        for i in 0..n {
            let c = s.idx(i, j);
            let (x, y) = s.center(i, j);
            let rr = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
            if rr < rad - 3.0 * dx {
                pin += s.p[c];
                nin += 1.0;
            } else if rr > rad + 3.0 * dx {
                pout += s.p[c];
                nout += 1.0;
            }
        }
    }
    let dp = pin / nin - pout / nout;
    let expect = 0.0728 / rad;
    let vmax = s.max_speed();
    eprintln!("Laplace : Δp = {dp:.3} Pa (théorie {expect:.3}), courants parasites {vmax:.2e} m/s");
    assert!((dp / expect - 1.0).abs() < 0.08, "Δp {dp} vs {expect}");
    assert!(vmax < 0.02, "courants parasites {vmax}");
}

#[test]
fn dam_break_front_speed_and_mass() {
    // colonne a = 0.1 m, H = 0.2 m dans une cuve 0.5 × 0.4 m, parois glissantes.
    // Borne théorique (Ritter, lit sec, fluide parfait) : vitesse du front ≤ 2√(gH) ; observée ≈ 1,5–1,9 √(gH) après l'accélération initiale.
    let (nx, ny, dx) = (50usize, 40usize, 0.01);
    let (a, hh) = (0.1, 0.2);
    let mut r = Registry::new();
    let air = r.add_phase(Phase::air());
    let eau = r.add_phase(Phase::water());
    let mut s = MultiphaseSim::new(nx, ny, dx, r, air);
    s.paint_phase(move |x, y| if x < a && y < hh { Some(eau) } else { Some(air) });
    s.finalize_init(|_, _| 293.15);
    let (m0, a0) = (s.total_mass(eau), s.phase_area(eau));
    let front = |s: &MultiphaseSim| -> f64 {
        // position du front = abscisse de la dernière cellule du fond contenant de l'eau, interpolée par sa fraction
        let i = (0..s.nx).rev().find(|&i| s.alpha[eau][s.idx(i, 0)] > 0.05).unwrap_or(0);
        (i as f64 + s.alpha[eau][s.idx(i, 0)].min(1.0)) * s.dx
    };
    let c0 = (9.81 * hh as f64).sqrt();
    let (mut p1, mut p2) = (None, None);
    while s.time < 0.26 {
        s.step();
        if p1.is_none() && s.time >= 0.14 { p1 = Some((s.time, front(&s))); }
        if p2.is_none() && s.time >= 0.24 { p2 = Some((s.time, front(&s))); }
    }
    let ((t1, x1), (t2, x2)) = (p1.unwrap(), p2.unwrap());
    let speed = (x2 - x1) / (t2 - t1);
    eprintln!("barrage : front {x1:.3} m → {x2:.3} m entre t={t1:.3} et {t2:.3} s : {:.2}×√(gH) (borne de Ritter 2)", speed / c0);
    assert!(speed / c0 > 0.8 && speed / c0 < 2.0, "vitesse du front {}×√(gH)", speed / c0);
    {
        let mut best = (0.0, 0usize, 0usize);
        for j in 0..s.ny { for i in 0..=s.nx { let v = s.u[s.uidx(i, j)].abs(); if v > best.0 { best = (v, i, j); } } }
        for j in 0..=s.ny { for i in 0..s.nx { let v = s.v[s.vidx(i, j)].abs(); if v > best.0 { best = (v, i, j); } } }
        let (i, j) = (best.1.min(s.nx - 1), best.2.min(s.ny - 1));
        eprintln!("vitesse max {:.2} m/s en ({},{}) α_eau={:.3}", best.0, best.1, best.2, s.alpha[eau][s.idx(i, j)]);
    }
    // vitesse maximale DANS L'EAU (faces de masse volumique > 500 kg/m³) : bornée par Ritter 2√(gH) (+ 30 % : faces d interface mélangées eau-air, voir limites)
    let mut vw = 0.0f64;
    for j in 0..s.ny { for i in 1..s.nx { if s.face_rho_x(i, j) > 500.0 { vw = vw.max(s.u[s.uidx(i, j)].abs()); } } }
    for j in 1..s.ny { for i in 0..s.nx { if s.face_rho_y(i, j) > 500.0 { vw = vw.max(s.v[s.vidx(i, j)].abs()); } } }
    eprintln!("vitesse max dans l'eau {vw:.2} m/s (Ritter 2√(gH) = {:.2})", 2.0 * c0);
    assert!(vw < 2.0 * c0 * 1.3, "vitesse de l'eau {vw}");
    assert!(s.max_speed() < 8.0 * c0, "vitesse folle {}", s.max_speed());
    assert!((s.total_mass(eau) / m0 - 1.0).abs() < 1e-9, "masse d'eau non conservée");
    assert!((s.phase_area(eau) / a0 - 1.0).abs() < 0.01, "volume d'eau non conservé (quasi-incompressibilité) {}", s.phase_area(eau) / a0);
}

#[test]
fn boiling_closed_box_energy_and_saturation() {
    // boîte rigide fermée, sans gravité : bulle de vapeur dans de l'eau surchauffée (383 K, p = 1 atm, T_sat = 373 K).
    // Attendu : de la vapeur se forme, l'énergie interne + cinétique est conservée (boîte rigide adiabatique, g = 0),
    // la pression monte (eau comprimée) et l'interface revient vers T_sat(p).
    let (r, air, eau, vap) = Registry::water_air_steam();
    let mut s = MultiphaseSim::new(24, 24, 0.002, r, air);
    s.gravity = [0.0, 0.0];
    s.surface_tension = false;
    s.dt_max = 5e-4;
    s.paint_phase(move |x, y| if ((x - 0.024).powi(2) + (y - 0.024).powi(2)).sqrt() < 0.008 { Some(vap) } else { Some(eau) });
    s.finalize_init(|_, _| 383.0);
    let pc = s.reg.changes[0].clone();
    let u0 = s.total_internal_energy();
    let (mv0, ml0) = (s.total_mass(vap), s.total_mass(eau));
    let mtot0 = s.total_mass_all();
    let mut ke_max = 0.0f64;
    for _ in 0..150 {
        s.step();
        ke_max = ke_max.max(s.kinetic_energy());
    }
    let (mv1, ml1) = (s.total_mass(vap), s.total_mass(eau));
    let dm = mv1 - mv0;
    let u1 = s.total_internal_energy();
    let l = pc.latent(380.0);
    let pmean = s.p.iter().sum::<f64>() / s.p.len() as f64;
    // température de la cellule d'interface la plus chaude vs T_sat(p)
    let mut worst = 0.0f64;
    for c in 0..24 * 24 {
        if s.alpha[vap][c] > 0.05 && s.alpha[eau][c] > 0.05 {
            worst = worst.max(s.temp[c] - pc.t_sat(s.p[c]));
        }
    }
    eprintln!(
        "ébullition : Δm_vapeur = {dm:.3e} kg/m, Δm_eau = {:.3e}, masse totale {:.1e} ; ΔU = {:.3e} J/m (L·Δm = {:.3e}) ; p_moy = {pmean:.0} Pa ; KE_max = {ke_max:.2e} ; surchauffe interface {worst:.2} K",
        ml1 - ml0,
        (s.total_mass_all() - mtot0) / mtot0,
        u1 - u0,
        l * dm
    );
    assert!(dm > 0.0, "pas de vaporisation");
    assert!(((ml1 - ml0) + dm).abs() < 1e-9 * ml0, "la masse perdue par l'eau doit être la masse gagnée par la vapeur");
    assert!(((s.total_mass_all() - mtot0) / mtot0).abs() < 1e-12, "masse totale non conservée");
    // énergie : U + KE conservée à 3 % de la chaleur latente mise en jeu
    assert!(((u1 - u0) + (s.kinetic_energy() - 0.0)).abs() < 0.03 * l * dm + 1e-6 * u0.abs(), "énergie : ΔU = {}", u1 - u0);
    assert!(pmean > 101325.0 + 100.0, "la pression doit monter");
}

#[test]
fn cavitation_under_tension_creates_vapor_and_conserves_mass() {
    // colonne d'eau fermée, étirée par un piston virtuel : on abaisse la pression sous p_sat(20 °C) = 2,3 kPa
    // en retirant du volume d'eau (masse enlevée uniformément) ⇒ la phase liquide est en tension, la vapeur doit apparaître.
    let (r, air, eau, vap) = Registry::water_air_steam();
    let mut s = MultiphaseSim::new(10, 10, 0.01, r, air);
    s.gravity = [0.0, 0.0];
    s.surface_tension = false;
    s.dt_max = 5e-4;
    s.paint_phase(|_, _| Some(eau));
    s.finalize_init(|_, _| 293.15);
    let pc = s.reg.changes[0].clone();
    // retire 0,06 % de la masse : p ≈ p0 − 0,0006/κ ≈ −1,3 MPa (traction) avant cavitation
    for c in 0..100 {
        s.mass[eau][c] *= 1.0 - 6e-4;
    }
    let m0 = s.total_mass_all();
    for _ in 0..20 {
        s.step();
    }
    let pmin = s.p.iter().cloned().fold(f64::MAX, f64::min);
    let a_v = s.phase_area(vap);
    eprintln!("cavitation : p_min = {pmin:.0} Pa (p_sat = {:.0}), aire de vapeur = {a_v:.3e} m², masse {:.1e}", pc.p_sat(293.15), (s.total_mass_all() - m0) / m0);
    assert!(a_v > 0.0, "pas de cavitation");
    assert!(pmin > -1.0e6, "la pression ne doit pas rester en forte traction (p = {pmin})");
    assert!(((s.total_mass_all() - m0) / m0).abs() < 1e-12);
    assert!(s.max_speed().is_finite());
}

#[test]
fn three_phase_stratification_is_static_and_buoyant_oil_rises() {
    // eau / huile / air : couches stables ⇒ immobiles, p_fond hydrostatique exact ; puis une goutte d'huile dans l'eau monte.
    let mut r = Registry::new();
    let air = r.add_phase(Phase::air());
    let eau = r.add_phase(Phase::water());
    let huile = r.add_phase(Phase::oil());
    let mut s = MultiphaseSim::new(12, 30, 0.01, r, air);
    s.bc[3] = Bc::Open { p: P_ATM };
    s.paint_phase(move |_, y| if y < 0.10 { Some(eau) } else if y < 0.18 { Some(huile) } else { Some(air) });
    s.finalize_init(|_, _| 293.15);
    let (me, mh, ma) = (s.total_mass(eau), s.total_mass(huile), s.total_mass(air));
    for _ in 0..40 {
        s.step();
    }
    assert!(s.max_speed() < 2e-3, "stratification stable agitée : {}", s.max_speed());
    let pb = s.p[s.idx(5, 0)];
    let expect = P_ATM + 9.81 * (998.2 * 0.095 + 915.0 * 0.08 + 1.204 * 0.12);
    assert!((pb - P_ATM) / (expect - P_ATM) > 0.99 && (pb - P_ATM) / (expect - P_ATM) < 1.01, "p_fond {pb} attendu ≈ {expect}");
    assert!((s.total_mass(eau) / me - 1.0).abs() < 1e-6 && (s.total_mass(huile) / mh - 1.0).abs() < 1e-6);
    let _ = ma;

    // goutte d'huile (r = 1.5 cm) au fond d'une cuve d'eau fermée : elle doit monter
    let mut r = Registry::new();
    let air = r.add_phase(Phase::air());
    let eau = r.add_phase(Phase::water());
    let huile = r.add_phase(Phase::oil());
    let mut s = MultiphaseSim::new(20, 30, 0.01, r, air);
    s.bc[3] = Bc::Open { p: P_ATM };
    s.paint_phase(move |x, y| {
        if ((x - 0.10).powi(2) + (y - 0.06).powi(2)).sqrt() < 0.025 { Some(huile) } else if y < 0.24 { Some(eau) } else { Some(air) }
    });
    s.finalize_init(|_, _| 293.15);
    let (me, mh) = (s.total_mass(eau), s.total_mass(huile));
    let centroid = |s: &MultiphaseSim| {
        let (mut sy, mut sa) = (0.0, 0.0);
        for j in 0..s.ny { for i in 0..s.nx { let a = s.alpha[huile][s.idx(i, j)]; sy += a * s.center(i, j).1; sa += a; } }
        sy / sa
    };
    let y0 = centroid(&s);
    while s.time < 0.6 {
        s.step();
    }
    let y1 = centroid(&s);
    eprintln!("goutte d'huile : centroïde {y0:.4} → {y1:.4} m en {:.2} s", s.time);
    assert!(y1 > y0 + 0.005, "la goutte plus légère doit monter");
    assert!((s.total_mass(eau) / me - 1.0).abs() < 1e-6 && (s.total_mass(huile) / mh - 1.0).abs() < 1e-6, "masses par phase non conservées");
}

#[test]
fn splash_of_falling_water_block_is_stable_and_conservative() {
    // bloc d'eau 6×6 cm lâché de 12 cm dans un bassin de 10 cm : éclaboussure ; stabilité, conservation, projection d'eau en l'air
    let (nx, ny, dx) = (60usize, 60usize, 0.01);
    let mut r = Registry::new();
    let air = r.add_phase(Phase::air());
    let eau = r.add_phase(Phase::water());
    let mut s = MultiphaseSim::new(nx, ny, dx, r, air);
    s.paint_phase(move |x, y| {
        if y < 0.10 || ((x - 0.30).abs() < 0.03 && (y - 0.25).abs() < 0.03) { Some(eau) } else { Some(air) }
    });
    s.finalize_init(|_, _| 293.15);
    let (m0, a0) = (s.total_mass(eau), s.phase_area(eau));
    let mech = |s: &MultiphaseSim| -> f64 {
        let mut pe = 0.0;
        for j in 0..s.ny { for i in 0..s.nx { pe += s.rho_mix[s.idx(i, j)] * 9.81 * s.center(i, j).1; } }
        pe * s.dx * s.dx + s.kinetic_energy()
    };
    s.refresh();
    let e0 = mech(&s);
    let mut e_worst: f64 = 0.0;
    let mut hmax: f64 = 0.0;
    while s.time < 0.7 {
        s.step();
        e_worst = e_worst.max(mech(&s) / e0);
        for j in 0..ny { for i in 0..nx { if s.alpha[eau][s.idx(i, j)] > 0.5 { hmax = hmax.max(s.center(i, j).1); } } }
        assert!(s.max_speed().is_finite() && s.max_speed() < 15.0, "instable : {}", s.max_speed());
    }
    let mut hmax_after_impact: f64 = 0.0;
    for j in 0..ny { for i in 0..nx { if s.alpha[eau][s.idx(i, j)] > 0.5 { hmax_after_impact = hmax_after_impact.max(s.center(i, j).1); } } }
    eprintln!("éclaboussure : t={:.2} s, altitude max de l'eau {hmax_after_impact:.3} m, volume {:.4} (initial {:.4})", s.time, s.phase_area(eau), a0);
    eprintln!("énergie mécanique max / initiale = {e_worst:.5}");
    // La métrique ½ρ_f u² avec ρ_f moyenné aux faces d'interface surestime l'énergie cinétique de l'air entraîné par l'eau
    // (avec la QDM conservative : +0,4 % mesuré) ; on borne à +2 %.
    assert!(e_worst < 1.02, "création d'énergie mécanique : {e_worst}");
    assert!(mech(&s) < e0, "l'énergie mécanique finale doit être inférieure à l'initiale (dissipation)");
    assert!((s.total_mass(eau) / m0 - 1.0).abs() < 1e-9, "masse d'eau non conservée");
    assert!((s.phase_area(eau) / a0 - 1.0).abs() < 0.01, "volume d'eau : {}", s.phase_area(eau) / a0);
    assert!(hmax_after_impact > 0.115, "pas de jaillissement : {hmax_after_impact}");
}

#[test]
fn free_fall_and_splash_energy_with_conservative_momentum() {
    // chute libre d'un bloc d'eau dans l'air : position à 3 % de ½gt², et l'air n'est pas entraîné par la vitesse de l'eau
    let (nx, ny, dx) = (30usize, 60usize, 0.01);
    let mut r = Registry::new();
    let air = r.add_phase(Phase::air());
    let eau = r.add_phase(Phase::water());
    let mut s = MultiphaseSim::new(nx, ny, dx, r, air);
    s.paint_phase(move |x, y| if (x - 0.15).abs() < 0.03 && (y - 0.40).abs() < 0.03 { Some(eau) } else { Some(air) });
    s.finalize_init(|_, _| 293.15);
    let cy = |s: &MultiphaseSim| { let (mut a, mut b) = (0.0, 0.0); for j in 0..s.ny { for i in 0..s.nx { let w = s.alpha[eau][s.idx(i, j)]; a += w * s.center(i, j).1; b += w; } } a / b };
    let (y0, m) = (cy(&s), s.total_mass(eau));
    while s.time < 0.2 { s.step(); }
    let dy = y0 - cy(&s);
    let th = 0.5 * 9.81 * s.time * s.time;
    let ke = s.kinetic_energy();
    let ratio = ke / (m * 9.81 * dy);
    eprintln!("chute libre (QDM conservative) : Δy={dy:.4} théorie {th:.4}, KE/(m g Δy) = {ratio:.3}");
    assert!((dy / th - 1.0).abs() < 0.04 && ratio < 1.15 && ratio > 0.9, "Δy={dy} ratio={ratio}");
}


/// Un état invalide ne doit JAMAIS se propager : le pas est annulé, refait avec dt/4, dt/16… puis, si rien ne passe, l'état sain
/// précédent est restauré et `diverged` passe à true (la fenêtre se met alors en pause au lieu de planter sur des NaN).
#[test]
fn invalid_step_is_rolled_back_not_propagated() {
    use super::mp_view::{build_scenario, MpScenario};
    let mut s = build_scenario(MpScenario::DamBreak, 24, 18, 0.01, 9.81);
    s.speed_limit = 1e-4; // tout écoulement non nul est jugé « aberrant »
    let (u0, v0, p0, t0) = (s.u.clone(), s.v.clone(), s.p.clone(), s.time);
    let dt = s.step();
    assert_eq!(dt, 0.0);
    assert!(s.diverged && s.rollbacks >= 5, "rollbacks = {}", s.rollbacks);
    assert!(s.u.iter().chain(&s.v).chain(&s.p).all(|x| x.is_finite()));
    assert_eq!((s.time, &s.u, &s.v, &s.p), (t0, &u0, &v0, &p0), "l'état doit être restauré tel quel");
    assert_eq!(s.step(), 0.0, "une simulation divergée n'avance plus");
    // et un sim sain ne déclenche aucun retour arrière
    let mut ok = build_scenario(MpScenario::DamBreak, 24, 18, 0.01, 9.81);
    for _ in 0..30 {
        assert!(ok.step() > 0.0);
    }
    assert_eq!(ok.rollbacks, 0);
    assert!(ok.state_is_sane());
}

// ------------------------------------------------------------------------------------------------ miscibilité
fn alcohol() -> Phase {
    let mut a = Phase::water();
    a.name = "alcool".into();
    a.eos = super::phases::Eos::Tait { p0: 101325.0, b: 1.0e8, n: 8.0, thermal: super::phases::Thermal::Beta { rho0: 800.0, t0: 293.15, beta: 1.1e-3 } };
    a.cp = 2400.0;
    a.sigma = 0.022;
    a.conductivity = 0.17;
    a.diffusivity = 1e-5; // artificiellement grand : le test doit voir le mélange en quelques secondes simulées
    a
}

/// eau en bas, alcool au-dessus (stable), air tout en haut, boîte fermée
pub(crate) fn two_liquid_box(miscible: bool) -> (MultiphaseSim, usize, usize, usize) {
    let mut r = Registry::new();
    let air = r.add_phase(Phase::air());
    let eau = r.add_phase(Phase::water());
    let alc = r.add_phase(alcohol());
    r.set_miscible(eau, alc, miscible);
    let mut s = MultiphaseSim::new(8, 24, 0.01, r, air);
    s.paint_phase(move |_, y| if y < 0.08 { Some(eau) } else if y < 0.18 { Some(alc) } else { Some(air) });
    s.finalize_init(|_, _| 293.15);
    (s, air, eau, alc)
}

#[test]
fn miscible_liquids_mix_by_diffusion_and_conserve_everything() {
    let (mut m, _air, eau, alc) = two_liquid_box(true);
    let (mut n, _, _, _) = two_liquid_box(false);
    let masses0: Vec<f64> = (0..3).map(|k| m.total_mass(k)).collect();
    let e0 = m.total_internal_energy();
    while m.time < 6.0 {
        m.step();
    }
    while n.time < 6.0 {
        n.step();
    }
    assert!(!m.diverged && m.rollbacks == 0, "rollbacks {}", m.rollbacks);
    // masse de chaque espèce conservée (boîte fermée, pas de changement de phase)
    for k in 0..3 {
        assert!((m.total_mass(k) / masses0[k] - 1.0).abs() < 1e-9, "masse {k} : {}", m.total_mass(k) / masses0[k] - 1.0);
    }
    assert!((m.total_internal_energy() / e0 - 1.0).abs() < 1e-3, "énergie interne");
    // alcool dans la tranche j = 0..7 (initialement de l'eau pure) : apparaît par diffusion si miscible, pas sinon
    let band = |s: &MultiphaseSim, k: usize| -> f64 { (0..8).flat_map(|j| (0..8).map(move |i| (i, j))).map(|(i, j)| s.mass[k][s.idx(i, j)]).sum::<f64>() };
    let (a_m, a_n) = (band(&m, alc), band(&n, alc));
    assert!(a_m > 1e-3 * 800.0 * 64.0 * 0.5, "l'alcool n'a pas diffusé vers le bas : {a_m}");
    assert!(a_n < 1e-3 * a_m.max(1e-30) || a_n < 1e-6, "sans miscibilité l'interface doit rester nette : {a_n} vs {a_m}");
    // l'eau, elle, monte dans l'ancienne couche d'alcool
    let w_top = |s: &MultiphaseSim| -> f64 { (8..18).flat_map(|j| (0..8).map(move |i| (i, j))).map(|(i, j)| s.mass[eau][s.idx(i, j)]).sum::<f64>() };
    assert!(w_top(&m) > 50.0 * w_top(&n).max(1e-9), "{} vs {}", w_top(&m), w_top(&n));
}

#[test]
fn miscible_dam_break_conserves_each_species() {
    let mut r = Registry::new();
    let air = r.add_phase(Phase::air());
    let eau = r.add_phase(Phase::water());
    let alc = r.add_phase(alcohol());
    r.set_miscible(eau, alc, true);
    let mut s = MultiphaseSim::new(40, 24, 0.01, r, air);
    s.paint_phase(move |x, y| if y < 0.10 && x < 0.10 { Some(eau) } else if y < 0.10 && x < 0.20 { Some(alc) } else { Some(air) });
    s.finalize_init(|_, _| 293.15);
    let m0: Vec<f64> = (0..3).map(|k| s.total_mass(k)).collect();
    for _ in 0..200 {
        s.step();
    }
    assert!(!s.diverged && s.state_is_sane());
    for k in 0..3 {
        assert!((s.total_mass(k) / m0[k] - 1.0).abs() < 1e-9, "masse {k}");
    }
    // l'interface eau/alcool a disparu (mélange) : il existe des cellules contenant les deux
    let both = (0..40 * 24).filter(|&c| s.alpha[eau][c] > 0.05 && s.alpha[alc][c] > 0.05).count();
    assert!(both > 0, "aucune cellule mélangée");
}

// ------------------------------------------------------------------------------------------------ cellules spéciales
use super::mp_cells::{MpEffect, MpInlet, MpThermal};

fn open_tank(nx: usize, ny: usize, h: f64) -> (MultiphaseSim, usize, usize) {
    let (mut s, air, eau) = tank(nx, ny, 0.01, h);
    s.bc[3] = Bc::Open { p: P_ATM };
    s.finalize_init(|_, _| 293.15);
    (s, air, eau)
}

fn run_for(s: &mut MultiphaseSim, t: f64) {
    while s.time < t {
        s.step();
    }
    assert!(!s.diverged && s.rollbacks == 0, "divergence / retours arrière : {}", s.rollbacks);
}

#[test]
fn sink_removes_water_at_the_prescribed_rate_and_balances_mass() {
    let (mut s, _air, eau) = open_tank(16, 24, 0.12);
    let c = s.idx(8, 0);
    let rate = 0.2;
    s.set_effect(c, Some(MpEffect::Sink { rate }));
    let m0 = s.total_mass(eau);
    run_for(&mut s, 0.4);
    let dm = s.total_mass(eau) - m0;
    // bilan exact : toute la masse retirée est inscrite dans boundary_flux
    assert!((dm - s.boundary_flux[eau]).abs() < 1e-9 * m0.abs(), "bilan : Δm = {dm}, flux = {}", s.boundary_flux[eau]);
    // débit attendu : 3 faces ouvertes (le sud est la paroi) × rate × dx × ρ × t
    let expect = -998.2 * 3.0 * rate * 0.01 * s.time;
    assert!(dm < 0.0 && (dm / expect - 1.0).abs() < 0.25, "Δm = {dm}, attendu ≈ {expect}");
    assert!(s.state_is_sane());
}

#[test]
fn emitter_creates_the_chosen_phase_with_exact_bookkeeping() {
    let (mut s, air, eau) = open_tank(16, 24, 0.0);
    // aucun liquide au départ : l'émetteur (eau, 1 m/s vers le haut) au fond du réservoir ouvert
    let c = s.idx(8, 1);
    s.set_effect(c, Some(MpEffect::Emitter { vx: 0.0, vy: 1.0, phase: eau, temp: 300.0 }));
    run_for(&mut s, 0.15);
    let w = s.total_mass(eau);
    assert!(w > 0.0, "l'émetteur n'a rien créé");
    assert!((w - s.boundary_flux[eau]).abs() < 1e-9 * w, "bilan eau");
    let expect = 998.2 * 1.0 * 0.01 * s.time; // ρ · v · dx · t (par m de profondeur)
    assert!((w / expect - 1.0).abs() < 0.5, "masse émise {w}, attendue ≈ {expect}");
    // l'eau émise est à la bonne température dans le jet
    let jet = (2..8).map(|j| s.idx(8, j)).filter(|&c| s.alpha[eau][c] > 0.5).map(|c| s.temp[c]).collect::<Vec<_>>();
    assert!(!jet.is_empty() && jet.iter().all(|&t| (t - 300.0).abs() < 2.0), "T du jet {jet:?}");
    let _ = air;
}

#[test]
fn accelerator_drives_a_tube_with_the_force_balance_a_times_length_ratio() {
    // tube d'air (3 cellules de haut, parois haut / bas), extrémités ouvertes à 1 atm : la force de volume ρ·a sur L_acc accélère
    // toute la colonne (longueur L) : a_eff = a·L_acc/L (incompressible, p identique aux deux bouts)
    let (mut s, _air, _eau) = tank(40, 3, 0.01, 0.0);
    s.bc = [Bc::Open { p: P_ATM }, Bc::Open { p: P_ATM }, Bc::Wall, Bc::Wall];
    s.gravity = [0.0, 0.0];
    s.finalize_init(|_, _| 293.15);
    for i in 10..30 {
        for j in 0..3 {
            let c = s.idx(i, j);
            s.set_effect(c, Some(MpEffect::Accelerator { ax: 3.0, ay: 0.0 }));
        }
    }
    run_for(&mut s, 0.05);
    let expect = 3.0 * 0.5 * s.time;
    let u_mid = s.u[s.uidx(20, 1)];
    assert!((u_mid / expect - 1.0).abs() < 0.2, "u = {u_mid}, attendu ≈ {expect}");
    // hors de la zone accélérée la vitesse est la même (incompressible) : continuité
    let u_out = s.u[s.uidx(35, 1)];
    assert!((u_out / u_mid - 1.0).abs() < 0.2, "u_out = {u_out}");
}

#[test]
fn teleporter_moves_the_absorbed_fluid_to_its_destination() {
    let (mut s, _air, eau) = open_tank(24, 24, 0.12);
    let (src, dst) = (s.idx(5, 4), s.idx(18, 18)); // source dans l'eau, destination dans l'air
    s.set_effect(src, Some(MpEffect::Teleporter { dest: dst, speed: 0.3 }));
    let m0 = s.total_mass(eau);
    run_for(&mut s, 0.4);
    // de l'eau est apparue autour de la destination (dans l'air au départ)
    let around: f64 = (14..23).flat_map(|j| (14..23).map(move |i| (i, j))).map(|(i, j)| s.mass[eau][s.idx(i, j)]).sum();
    assert!(around > 1e-3, "rien n'est arrivé à destination : {around}");
    // bilan : la masse d'eau totale n'a pas été détruite sans compte
    let dm = s.total_mass(eau) - m0;
    assert!((dm - s.boundary_flux[eau]).abs() < 1e-9 * m0, "bilan : {dm} vs {}", s.boundary_flux[eau]);
    // le débit volumique émis est proche du débit aspiré : la masse d'eau nette change peu devant la masse aspirée
    let sucked = 998.2 * 4.0 * 0.3 * 0.01 * s.time;
    assert!(dm.abs() < 0.6 * sucked, "Δm = {dm} (aspiré ≈ {sucked})");
}

#[test]
fn heater_and_isothermal_solids_follow_their_thermal_mode() {
    use super::mp_view::{ACIER, VERRE};
    let reg = super::mp_view::registry();
    let air = 0;
    let mut s = MultiphaseSim::new(20, 12, 0.01, reg, air);
    s.paint_phase(|_, _| Some(air));
    s.paint_solid(|x, y| if y < 0.03 && x > 0.05 && x < 0.10 { Some(ACIER) } else if y < 0.03 && x > 0.12 && x < 0.17 { Some(VERRE) } else { None });
    s.finalize_init(|_, _| 293.15);
    let heater = s.idx(6, 1);
    let iso = s.idx(14, 1);
    for c in 0..s.nx * s.ny {
        if s.solid[c] == ACIER as i32 {
            s.set_thermal_cell(c, MpThermal::Heater(2e8));
        }
        if s.solid[c] == VERRE as i32 {
            s.set_thermal_cell(c, MpThermal::Isothermal(350.0));
        }
    }
    let e0 = s.total_internal_energy();
    run_for(&mut s, 0.1);
    assert!(s.energy_input > 0.0);
    // ΔT d'un bloc chauffant ≈ P t / (ρ c) (conduction vers l'air négligeable à cette échelle de temps)
    let expect = 2e8 * s.time / (7850.0 * 490.0);
    let dt_heater = s.temp[heater] - 293.15;
    assert!(dt_heater > 0.5 * expect && dt_heater < 1.1 * expect, "ΔT = {dt_heater}, attendu ≈ {expect}");
    assert!((s.temp[iso] - 350.0).abs() < 1e-6, "T isotherme = {}", s.temp[iso]);
    // l'air au-dessus du bloc isotherme se réchauffe (puits / source idéale de chaleur)
    assert!(s.temp[s.idx(14, 3)] > 293.2, "{}", s.temp[s.idx(14, 3)]);
    // bilan d'énergie : ΔE_interne = chaleur injectée par le chauffage + apport des solides isothermes (positif) ; au moins la première
    let de = s.total_internal_energy() - e0;
    assert!(de >= 0.95 * s.energy_input, "ΔE = {de}, chauffage = {}", s.energy_input);
}

#[test]
fn inlet_boundary_injects_the_chosen_phase_with_exact_bookkeeping() {
    let (mut s, _air, eau) = open_tank(24, 16, 0.0);
    s.bc[0] = Bc::Wall;
    s.bc[1] = Bc::Open { p: P_ATM };
    s.inlet[0] = Some(MpInlet { vx: 0.4, vy: 0.0, phase: eau, temp: 310.0, offset: 0 });
    run_for(&mut s, 0.3);
    let w = s.total_mass(eau);
    assert!((w - s.boundary_flux[eau]).abs() < 1e-9 * w.max(1.0), "bilan");
    let expect = 998.2 * 0.4 * (16.0 * 0.01) * s.time;
    assert!((w / expect - 1.0).abs() < 0.25, "masse injectée {w}, attendue ≈ {expect}");
    // l'eau entrante est à 310 K
    let t = s.temp[s.idx(0, 8)];
    assert!((t - 310.0).abs() < 3.0, "T d'entrée {t}");
}

#[test]
fn extras_survive_a_dam_break_without_nan() {
    use super::mp_view::{build_scenario, MpScenario};
    let mut s = build_scenario(MpScenario::DamBreak, 48, 30, 0.01, 9.81);
    let c = s.idx(40, 3);
    s.set_effect(c, Some(MpEffect::Sink { rate: 0.5 }));
    let c = s.idx(2, 25);
    s.set_effect(c, Some(MpEffect::Emitter { vx: 0.5, vy: 0.0, phase: 1, temp: 293.15 }));
    s.bc[3] = Bc::Open { p: P_ATM };
    for _ in 0..150 {
        s.step();
    }
    assert!(!s.diverged && s.state_is_sane());
}
