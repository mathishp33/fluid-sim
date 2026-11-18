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
    // (test de chute libre : +36 % à dx = 1 cm, +9 % à dx = 5 mm, → 0 avec le maillage) ; on borne donc à +12 %.
    assert!(e_worst < 1.12, "création d'énergie mécanique : {e_worst}");
    assert!(mech(&s) < e0, "l'énergie mécanique finale doit être inférieure à l'initiale (dissipation)");
    assert!((s.total_mass(eau) / m0 - 1.0).abs() < 1e-9, "masse d'eau non conservée");
    assert!((s.phase_area(eau) / a0 - 1.0).abs() < 0.01, "volume d'eau : {}", s.phase_area(eau) / a0);
    assert!(hmax_after_impact > 0.115, "pas de jaillissement : {hmax_after_impact}");
}
