//! Validation de la chimie, des flammes et des explosions contre des valeurs de référence (NIST-JANAF, mesures de vitesse de flamme,
//! pression de combustion isochore) et tests de conservation exacte.

use super::mp_materials::{MaterialLibrary, AIR_SEC, CH4, CO, CO2, H2, H2O_G, MIX_CH4_AIR, MIX_H2_AIR, N2, O2, RX_CH4, RX_CO, RX_H2};
use super::mp_view::{build_scenario_with, MpScenario};
use super::multiphase::{Bc, MultiphaseSim};
use super::phases::{Phase, Registry, R_GAS};

fn combustion_registry() -> (Registry, MaterialLibrary) {
    let lib = MaterialLibrary::default().with_combustion();
    let b = lib.build().expect("préréglage combustion valide");
    assert!(b.warnings.is_empty(), "avertissements inattendus : {:?}", b.warnings);
    (b.registry, lib)
}

fn idx(reg: &Registry, n: &str) -> usize {
    reg.phases.iter().position(|p| p.name == n).unwrap_or_else(|| panic!("{n} absent"))
}
fn rx<'a>(reg: &'a Registry, n: &str) -> &'a super::phases::Reaction {
    reg.reactions.iter().find(|r| r.name == n).unwrap()
}

#[test]
fn reaction_enthalpies_and_gibbs_energies_match_janaf() {
    let (reg, _) = combustion_registry();
    // pouvoir calorifique inférieur du méthane : −802.3 kJ/mol ; H2 → H2O(g) : −241.8 kJ/mol ; CO → CO2 : −283.0 kJ/mol (298 K)
    let dh = |n: &str, t: f64| rx(&reg, n).delta_h(&reg.phases, t) / 1e3;
    assert!((dh(RX_CH4, 298.15) + 802.3).abs() < 1.0, "Δh CH4 = {}", dh(RX_CH4, 298.15));
    assert!((dh(RX_H2, 298.15) + 241.83).abs() < 0.5, "Δh H2 = {}", dh(RX_H2, 298.15));
    assert!((dh(RX_CO, 298.15) + 282.99).abs() < 0.5, "Δh CO = {}", dh(RX_CO, 298.15));
    // ΔG° à 2000 K (JANAF) : H2 + ½O2 → H2O : −135.5 kJ/mol ; CO + ½O2 → CO2 : −110.3 kJ/mol. Erreur due au cp linéarisé : < 4 kJ/mol
    let dg = |n: &str, t: f64| {
        let r = rx(&reg, n);
        let lnk = r.ln_kc(&reg.phases, t);
        // ln Kc = −ΔG/RT + Δn ln(P°/RT)  ⇒  ΔG = −RT (ln Kc − Δn ln(P°/RT))
        -R_GAS * t * (lnk - r.delta_n() * (1e5 / (R_GAS * t)).ln()) / 1e3
    };
    let (g1, g2) = (dg(RX_H2, 2000.0), dg(RX_CO, 2000.0));
    assert!((g1 + 135.5).abs() < 4.0, "ΔG H2O(2000 K) = {g1}");
    assert!((g2 + 110.3).abs() < 4.0, "ΔG CO2(2000 K) = {g2}");
    // chaleur de réaction reproduite par les décalages e0 : Δu(298) = Δh(298) − Δn R T
    let r = rx(&reg, RX_H2);
    let du = r.delta_u(&reg.phases, 298.15);
    assert!((du - (r.delta_h(&reg.phases, 298.15) - r.delta_n() * R_GAS * 298.15)).abs() < 1.0);
    // latent de l'eau toujours cohérent (vapeur ancrée par la réaction, liquide recalé) : h_v − h_l = 2.257 MJ/kg à 100 °C
    let pc = &reg.changes[0];
    assert!((reg.latent_heat(pc, 373.15) - 2.257e6).abs() < 10.0, "L = {}", reg.latent_heat(pc, 373.15));
}

/// boîte fermée uniforme (aucun écoulement : chimie seule) remplie du mélange `mix` à T0
fn uniform_box(lib: &MaterialLibrary, mix: &str, t0: f64, n: usize) -> MultiphaseSim {
    let b = lib.build().unwrap();
    let a = b.mixtures.iter().find(|m| m.0 == mix).unwrap().1.clone();
    let mut s = MultiphaseSim::new(n, n, 0.01, b.registry, b.ambient);
    s.ambient_mix = b.ambient_mix.clone();
    s.mixtures = b.mixtures.clone();
    s.gravity = [0.0, 0.0];
    s.radiation = false;
    s.paint_mixture(move |_, _| Some(a.clone()));
    s.finalize_init(move |_, _| t0);
    s
}

#[test]
fn chemistry_conserves_mass_and_energy_exactly_and_reaches_equilibrium() {
    let (_, lib) = combustion_registry();
    // CH4–air préchauffé à 1500 K : combustion complète puis dissociation à l'équilibre, à volume et énergie constants
    let mut s = uniform_box(&lib, MIX_CH4_AIR, 1500.0, 3);
    let c = s.idx(1, 1);
    let np = s.reg.phases.len();
    let m0: f64 = (0..np).map(|k| s.mass[k][c]).sum();
    let e0 = s.e[c];
    let mut t_hist = vec![];
    for _ in 0..200 {
        s.react_step(1e-4);
        t_hist.push(s.temp[c]);
    }
    let m1: f64 = (0..np).map(|k| s.mass[k][c]).sum();
    assert!((m1 - m0).abs() < 1e-12 * m0, "masse : {m0} → {m1}");
    assert_eq!(s.e[c], e0, "l'énergie interne doit être inchangée par la chimie");
    let reg = &s.reg;
    let (ch4, o2, co, co2, h2, h2o, n2) = (idx(reg, CH4), idx(reg, O2), idx(reg, CO), idx(reg, CO2), idx(reg, H2), idx(reg, H2O_G), idx(reg, N2));
    let conc = |k: usize| s.mass[k][c] / reg.phases[k].molar_mass;
    assert!(conc(ch4) < 1e-6 * conc(n2), "méthane non brûlé : {}", conc(ch4));
    let t = s.temp[c];
    // combustion isochore depuis 1500 K : T ≈ 3000 K, dissociation marquée (CO, H2 présents)
    assert!(t > 2700.0 && t < 3300.0, "T = {t}");
    assert!(conc(co) > 1e-3 * conc(co2) && conc(h2) > 1e-4 * conc(h2o), "pas de dissociation");
    // équilibres : Q = Kc à 2 % près
    for (name, num, den) in [(RX_CO, vec![(co2, 1.0)], vec![(co, 1.0), (o2, 0.5)]), (RX_H2, vec![(h2o, 1.0)], vec![(h2, 1.0), (o2, 0.5)])] {
        let lnq: f64 = num.iter().map(|&(k, nu)| nu * conc(k).ln()).sum::<f64>() - den.iter().map(|&(k, nu)| nu * conc(k).ln()).sum::<f64>();
        let lnk = rx(reg, name).ln_kc(&reg.phases, t);
        assert!((lnq - lnk).abs() < 0.02, "{name} : ln Q = {lnq}, ln Kc = {lnk}");
    }
    // la chaleur dégagée comptée = baisse de l'énergie chimique
    assert!(s.chem_heat > 0.0);
}

fn flame_position(s: &MultiphaseSim, t_iso: f64) -> f64 {
    // x de l'isotherme t_iso le plus avancé (ligne médiane)
    let j = s.ny / 2;
    let mut x = 0.0;
    for i in 0..s.nx - 1 {
        let (a, b) = (s.temp[s.idx(i, j)], s.temp[s.idx(i + 1, j)]);
        if a >= t_iso && b < t_iso {
            x = (i as f64 + 0.5 + (a - t_iso) / (a - b)) * s.dx;
        }
    }
    x
}

#[test]
fn methane_air_laminar_flame_speed_and_adiabatic_temperature() {
    // tube de 40 cm (maille 2.5 mm, flamme épaissie F ≈ 43), fermé à gauche, allumé contre le fond : la flamme avance à σ·S_L
    let lib = MaterialLibrary::default().with_combustion();
    let mut s = build_scenario_with(&lib, MpScenario::PremixedFlame, 160, 4, 2.5e-3, 0.0).unwrap();
    s.radiation = false;
    let lx = s.nx as f64 * s.dx;
    let (mut t1, mut t2) = (None, None);
    while s.time < 0.4 && t2.is_none() {
        s.step();
        assert!(!s.diverged, "divergence");
        let x = flame_position(&s, 1500.0);
        if t1.is_none() && x > 0.2 * lx {
            t1 = Some(s.time);
        }
        if x > 0.7 * lx {
            t2 = Some(s.time);
        }
    }
    // température des gaz brûlés : moyenne du plateau derrière la flamme (0.35 L → 0.6 L) quand elle atteint 0.7 L (le maximum
    // ponctuel au passage du front est un transitoire de quelques dizaines de K)
    let j = s.ny / 2;
    let burnt: Vec<f64> = (0..s.nx).filter(|&i| { let x = (i as f64 + 0.5) * s.dx; x > 0.35 * lx && x < 0.6 * lx }).map(|i| s.temp[s.idx(i, j)]).collect();
    let tb_max = burnt.iter().sum::<f64>() / burnt.len() as f64;
    let (t1, t2) = (t1.expect("pas de propagation"), t2.expect("flamme trop lente"));
    // la flamme pulse avec l'acoustique du tube : vitesse MOYENNE sur 0.2 L → 0.7 L
    let u_lab = 0.5 * lx / (t2 - t1);
    // σ = ρ_u/ρ_b à pression constante : T_b/T_u × (M_u/M_b) ; M ne change pas pour CH4 + 2 O2 → CO2 + 2 H2O
    let sigma = tb_max / 293.15;
    let s_l = u_lab / sigma;
    println!("CH4–air : U_lab = {u_lab:.3} m/s, T_b = {tb_max:.0} K, σ = {sigma:.2}, S_L = {s_l:.3} m/s");
    // références : T_ad (équilibre complet, 1 atm, 298 K) = 2226 K — le modèle (dissociation CO2/H2O, sans radicaux OH, H, O) donne
    // 2230–2320 K ; S_L = 0.36–0.40 m/s mesurée (stœchiométrique, 1 atm), cible du calage du mécanisme global 0.38. Le modèle donne
    // 0.384 m/s (2231 K). Il donnait 0.47–0.49 tant que la correction de fermeture faisait osciller l'écoulement (flamme pulsée par
    // l'acoustique du tube) : bornes resserrées pour détecter ce genre de régression
    assert!(tb_max > 2150.0 && tb_max < 2340.0, "T_b = {tb_max}");
    assert!(s_l > 0.34 && s_l < 0.43, "S_L = {s_l}");
}

#[test]
fn confined_hydrogen_explosion_reaches_the_isochoric_pressure() {
    let lib = MaterialLibrary::default();
    let mut s = build_scenario_with(&lib, MpScenario::ConfinedExplosion, 48, 48, 5e-3, 0.0).unwrap();
    s.radiation = false;
    // tout le volume en mélange frais, puis un petit noyau de flamme (r = 6 mailles) : pression finale ≈ AICC
    let fresh = s.mixtures.iter().find(|m| m.0 == MIX_H2_AIR).unwrap().1.clone();
    for c in 0..s.nx * s.ny {
        s.fill_mixture(c, &fresh, 293.15);
    }
    let cells = super::mp_view::brush_cells(s.nx, s.ny, s.nx / 2, s.ny / 2, 6, true);
    super::mp_view::apply_tool(&mut s, super::mp_view::MpTool::Ignite, &cells, 2200.0, (0.0, 0.0), 0.0);
    let e0 = s.total_internal_energy() + s.kinetic_energy();
    let m0 = s.total_mass_all();
    let mut p_max: f64 = 0.0;
    while s.time < 0.04 && !s.diverged {
        s.step();
        let pm = (0..s.nx * s.ny).map(|c| s.p[c]).sum::<f64>() / (s.nx * s.ny) as f64;
        p_max = p_max.max(pm);
    }
    assert!(!s.diverged, "divergence");
    // pression de combustion adiabatique isochore (AICC) H2–air stœchiométrique depuis 1 atm, 293 K : ≈ 8.0 bar (équilibre complet ;
    // le modèle, sans radicaux OH/H/O, est un peu plus chaud)
    println!("H2–air confiné : p moyenne max = {:.2} bar", p_max / 1e5);
    assert!(p_max > 7.0e5 && p_max < 8.9e5, "p_max = {p_max}");
    // hydrogène consommé (il en reste ~5 % à l'équilibre de dissociation de H2O vers 2800 K), masse exacte, énergie totale
    // (chimique comprise) conservée à 8 % de la chaleur dégagée près (schéma en énergie INTERNE : le travail de pression discret ne
    // compense pas exactement l'énergie cinétique des oscillations acoustiques violentes de l'explosion)
    let h2 = idx(&s.reg, H2);
    assert!(s.total_mass(h2) < 0.15 * m0 * 0.0283, "H2 restant : {}", s.total_mass(h2));
    let m1 = s.total_mass_all();
    assert!((m1 - m0).abs() < 1e-9 * m0, "masse {m0} → {m1}");
    let e1 = s.total_internal_energy() + s.kinetic_energy();
    assert!((e1 - e0).abs() < 0.08 * s.chem_heat, "énergie {e0} → {e1} (chaleur dégagée {})", s.chem_heat);
}

#[test]
fn water_evaporates_into_dry_air_until_the_partial_pressure_is_saturated() {
    // préréglage combustion : tous les gaz miscibles, vapeur comprise ⇒ évaporation de Dalton (sans réaction : chimie gelée à 293 K)
    let lib = MaterialLibrary::default().with_combustion();
    let b = lib.build().unwrap();
    let mut s = MultiphaseSim::new(12, 12, 5e-3, b.registry, b.ambient);
    s.ambient_mix = b.ambient_mix.clone();
    s.radiation = false;
    let eau = idx(&s.reg, "eau");
    let vap = idx(&s.reg, H2O_G);
    s.paint_phase(move |_, y| if y < 0.02 { Some(eau) } else { None });
    s.finalize_init(|_, _| 293.15);
    let m0 = s.total_mass(eau) + s.total_mass(vap);
    for _ in 0..200 {
        s.step();
    }
    assert!(!s.diverged);
    assert!(s.total_mass(vap) > 0.0, "aucune évaporation");
    let m1 = s.total_mass(eau) + s.total_mass(vap);
    assert!((m1 - m0).abs() < 1e-9 * m0, "eau + vapeur non conservées");
    // juste au-dessus de la surface : x_v·p ≈ p_sat(T) (2.34 kPa à 20 °C ; T légèrement abaissée par l'évaporation)
    let c = s.idx(6, 4);
    let np = s.reg.phases.len();
    let ng: f64 = (0..np).filter(|&k| s.reg.phases[k].is_gas()).map(|k| s.mass[k][c] / s.reg.phases[k].molar_mass).sum();
    let pv = s.mass[vap][c] / s.reg.phases[vap].molar_mass / ng * s.p[c];
    let ps = s.reg.changes[0].p_sat(s.temp[c]);
    println!("interface : p_v = {pv:.0} Pa, p_sat(T = {:.2} K) = {ps:.0} Pa", s.temp[c]);
    assert!((pv - ps).abs() < 0.1 * ps, "p_v = {pv}, p_sat = {ps}");
    // sans miscibilité (matériaux par défaut) : pas d'évaporation à 20 °C (comportement historique)
    let mut d = super::mp_view::build_scenario(MpScenario::DamBreak, 16, 12, 5e-3, 9.81);
    for _ in 0..50 {
        d.step();
    }
    assert_eq!(d.total_mass(super::mp_view::VAPEUR), 0.0);
}

#[test]
fn species_diffusion_moves_no_net_mass_and_hydrogen_diffuses_fastest() {
    let lib = MaterialLibrary::default().with_combustion();
    let b = lib.build().unwrap();
    let mut s = MultiphaseSim::new(40, 3, 1e-3, b.registry, b.ambient);
    s.ambient_mix = b.ambient_mix.clone();
    s.gravity = [0.0, 0.0];
    let (h2, co2, n2) = (idx(&s.reg, H2), idx(&s.reg, CO2), idx(&s.reg, N2));
    // gauche : 50 % H2 + 50 % CO2 ; droite : N2
    let np = s.reg.phases.len();
    let mut left = vec![0.0; np];
    left[h2] = 0.5;
    left[co2] = 0.5;
    let mut right = vec![0.0; np];
    right[n2] = 1.0;
    s.paint_mixture(move |x, _| Some(if x < 0.02 { left.clone() } else { right.clone() }));
    s.finalize_init(|_, _| 293.15);
    s.refresh();
    let mtot: Vec<f64> = (0..s.nx * s.ny).map(|c| (0..np).map(|k| s.mass[k][c]).sum()).collect();
    let e_tot = s.total_internal_energy();
    s.diffuse_species(0.01);
    for c in 0..s.nx * s.ny {
        let m: f64 = (0..np).map(|k| s.mass[k][c]).sum();
        assert!((m - mtot[c]).abs() < 1e-12 * mtot[c], "la diffusion a transporté de la masse nette");
    }
    assert!((s.total_internal_energy() - e_tot).abs() < 1e-9 * e_tot.abs());
    // pénétration au-delà de 2.2 cm après 0.2 s : H2 (D = 7.5e-5 m²/s) ≫ CO2 (D = 1.6e-5 m²/s)
    for _ in 0..19 {
        s.diffuse_species(0.01);
    }
    let beyond = |k: usize| (22..40).map(|i| s.mass[k][s.idx(i, 1)] / s.reg.phases[k].molar_mass).sum::<f64>();
    assert!(beyond(h2) > 3.0 * beyond(co2), "H2 {} vs CO2 {}", beyond(h2), beyond(co2));
}

#[test]
fn combustion_scenarios_build_and_run_without_diverging() {
    let lib = MaterialLibrary::default();
    for sc in [MpScenario::PremixedFlame, MpScenario::ConfinedExplosion, MpScenario::VentedExplosion, MpScenario::DiffusionFlame] {
        let mut s = build_scenario_with(&lib, sc, 48, 36, 4e-3, 9.81).unwrap();
        assert!(s.ambient_mix.len() == s.reg.phases.len(), "ambiant air sec attendu");
        assert!(matches!(s.bc[0], Bc::Wall));
        for _ in 0..60 {
            s.step();
        }
        assert!(!s.diverged && s.state_is_sane(), "{} diverge", sc.name());
        assert!(s.chem_heat > 0.0, "{} : rien n'a brûlé", sc.name());
    }
    // l'ambiant de ces scénarios est bien de l'air sec
    let b = lib.with_combustion().build().unwrap();
    assert_eq!(b.registry.phases[b.ambient].name, N2);
    assert!(b.mixtures.iter().any(|m| m.0 == AIR_SEC));
    let _ = MIX_H2_AIR;
}

// ---------------------------------------------------------------- réactions de surface (poudres, charbon)

/// boîte fermée (4 × 4, sans gravité) d'air sec à T contenant une poussière de charbon uniforme (fraction a0, grains de diamètre d)
fn coal_dust(t: f64, a0: f64, d: f64) -> (MultiphaseSim, usize, usize, usize) {
    let mut lib = MaterialLibrary::default().with_coal();
    let k = lib.fluid_index(super::mp_materials::COAL).unwrap();
    lib.fluids[k].grain_diameter = d;
    let b = lib.build().expect("préréglage charbon valide");
    assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    let reg = b.registry;
    let (c, o2, co2) = (idx(&reg, super::mp_materials::COAL), idx(&reg, O2), idx(&reg, CO2));
    let mut s = MultiphaseSim::new(4, 4, 0.01, reg, b.ambient);
    s.ambient_mix = b.ambient_mix.clone();
    s.gravity = [0.0, 0.0];
    let mut m = b.ambient_mix.iter().map(|x| x * (1.0 - a0)).collect::<Vec<f64>>();
    m[c] = a0;
    s.paint_mixture(move |_, _| Some(m.clone()));
    s.finalize_init(move |_, _| t);
    s.refresh();
    (s, c, o2, co2)
}

/// vitesse initiale (mol/(m³·s)) de C + O2 → CO2 mesurée sur un appel de la chimie, et celle de la loi de surface
/// vitesse initiale de consommation du carbone (mol/(m³·s)) et rapport CO/CO2 produits, mesurés sur un appel de la chimie, et ceux de
/// la loi : C + ½O2 → CO (k_CO, Field) et C + O2 → CO2 (k_CO / R, R = 2500 e^(−6240/T)), oxygène partagé D / (K_O2 + D)
fn coal_rates(t: f64, d: f64) -> (f64, f64, f64, f64) {
    let (mut s, c, o2, co2) = coal_dust(t, 1e-5, d);
    let co = idx(&s.reg, CO);
    let q = s.idx(1, 1);
    let rx_of = |n: &str| s.reg.reactions.iter().find(|r| r.name == n).unwrap().clone();
    let (r_co, r_co2) = (rx_of(super::mp_materials::RX_COAL_CO), rx_of(super::mp_materials::RX_COAL));
    let (tc, p) = (s.tg[s.group_of[o2]][q], s.p[q]);
    let sv = 6.0 * s.alpha[c][q] / d;
    let c_o2 = s.mass[o2][q] / s.reg.phases[o2].molar_mass;
    let (k_co, k_co2) = (r_co.k_forward(tc), r_co2.k_forward(tc));
    let kd = 2.0 * s.reg.phases[o2].diffusivity_at(tc, p) / d;
    let (rk_co, rk_co2) = (sv * k_co * c_o2, sv * k_co2 * c_o2);
    let d_sup = sv * kd * c_o2;
    let lam = d_sup / (0.5 * rk_co + rk_co2 + d_sup);
    let expect = (rk_co + rk_co2) * lam;
    let (mc0, mco0, mco20) = (s.mass[c][q], s.mass[co][q], s.mass[co2][q]);
    let dt = 1e-4;
    s.react_step(dt);
    let got = (mc0 - s.mass[c][q]) / 0.012011 / dt;
    let n_co = (s.mass[co][q] - mco0) / s.reg.phases[co].molar_mass;
    let n_co2 = (s.mass[co2][q] - mco20) / s.reg.phases[co2].molar_mass;
    (got, expect, n_co / n_co2.max(1e-300), 2500.0 * (-6240.0 / tc).exp())
}

#[test]
fn coal_burns_at_the_grain_surface_kinetic_then_diffusion_limited() {
    // 1000 K : régime cinétique (k_s ≈ 0.01 m/s ≪ k_d ≈ 1 m/s), vitesse ∝ 1/d ; 1800 K : régime diffusionnel (k_s ≫ k_d), ∝ 1/d² ;
    // produits : CO/CO2 = 2500 e^(−6240/T) (Arthur 1951)
    for (t, ratio) in [(1000.0, 2.0), (1800.0, 4.0)] {
        let (g1, e1, co1, cox) = coal_rates(t, 3e-4);
        let (g2, e2, _, _) = coal_rates(t, 1.5e-4);
        eprintln!("T = {t} K : d = 0.3 mm {g1:.4e} (loi {e1:.4e}) ; d = 0.15 mm {g2:.4e} (loi {e2:.4e}) ; rapport {:.2} ; CO/CO2 {co1:.3} (loi {cox:.3})", g2 / g1);
        assert!((g1 / e1 - 1.0).abs() < 0.02 && (g2 / e2 - 1.0).abs() < 0.02, "vitesse de surface : {g1} vs {e1}, {g2} vs {e2}");
        assert!((g2 / g1 / ratio - 1.0).abs() < 0.12, "effet du diamètre à {t} K : {} (attendu ≈ {ratio})", g2 / g1);
        assert!((co1 / cox - 1.0).abs() < 0.01, "CO/CO2 {co1} (loi {cox})");
    }
    // froid : rien ne brûle
    let (g, _, _, _) = coal_rates(300.0, 3e-4);
    assert!(g < 1e-20, "le charbon ne brûle pas à 300 K : {g}");
}

#[test]
fn hot_fuel_meeting_air_burns_at_the_mixing_rate_but_premixed_and_cold_mixtures_keep_their_kinetics() {
    // couche de mélange méthane | air (boîte 12 × 3, sans gravité) : à 1200 K, ses mailles sont des mailles de flamme de DIFFUSION
    // (combustible et oxygène y arrivent de part et d'autre) — elles brûlent au moins au rythme du mélange 1/√(2Δ/g) ; à 300 K, rien
    // (mélange froid : allumage et explosions gardent leur cinétique) ; un front de PRÉMÉLANGE (gaz brûlés | méthane–air frais : le
    // combustible et l'oxygène croissent du même côté), rien non plus (vitesse de flamme inchangée)
    let lib = MaterialLibrary::default().with_combustion();
    let b = lib.build().unwrap();
    let fresh = b.mixtures.iter().find(|m| m.0 == MIX_CH4_AIR).unwrap().1.clone();
    let burnt = super::mp_species::complete_combustion(&b.registry, &fresh);
    let air = b.ambient_mix.clone();
    let ch4 = idx(&b.registry, CH4);
    let np = b.registry.phases.len();
    let layer = |t: f64, premixed: bool| -> Vec<f64> {
        let mut s = MultiphaseSim::new(12, 3, 0.01, b.registry.clone(), b.ambient);
        s.ambient_mix = air.clone();
        s.gravity = [0.0, 0.0];
        let (fresh, burnt, air) = (fresh.clone(), burnt.clone(), air.clone());
        s.paint_mixture(move |x, _| {
            // 0 à gauche, 1 à droite : rampe sur 8 cm
            let f = ((x - 0.02) / 0.08).clamp(0.0, 1.0);
            Some((0..np).map(|k| if premixed { (1.0 - f) * burnt[k] + f * fresh[k] } else { (1.0 - f) * if k == ch4 { 1.0 } else { 0.0 } + f * air[k] }).collect())
        });
        s.finalize_init(move |_, _| t);
        s.refresh();
        s.diffusion_mixing()
    };
    let rate = 1.0 / (2.0 * 0.01 / 9.81f64).sqrt();
    let hot = layer(1200.0, false);
    let lit = hot.iter().filter(|&&m| m > 0.0).count();
    assert!(lit >= 18 && hot.iter().all(|&m| m == 0.0 || (m - rate).abs() < 1e-9 * rate), "couche de diffusion chaude : {lit} mailles");
    assert!(layer(300.0, false).iter().all(|&m| m == 0.0), "mélange froid");
    assert!(layer(1200.0, true).iter().all(|&m| m == 0.0), "front de prémélange");
}

#[test]
fn a_coal_fire_keeps_burning_and_conserves_carbon() {
    // braséro, feu établi : braises (1450 K) sous une charge de houille chaude (700 K) posée au centre. Les braises restent chaudes, l'air
    // traverse la grille (tirage), la houille dégaze et ses matières volatiles brûlent en flamme, toujours allumée à la fin (accrochée par
    // les braises à nu le long des parois ; charge sur toute la largeur, elle s'éteignait en 1.3 s) ; carbone (charbon + houille + CO +
    // CO2 + CH4 + sorties) conservé
    let lib = MaterialLibrary::default();
    let mut s = build_scenario_with(&lib, MpScenario::CoalFire, 40, 40, 0.01, 9.81).unwrap();
    let k = |n: &str| idx(&s.reg, n);
    let (c, co2, o2, co, ch4, hou) = (k(super::mp_materials::COAL), k(CO2), k(O2), k(CO), k(CH4), k(super::mp_materials::HOUILLE));
    let gas_c = |s: &MultiphaseSim, q: usize| (s.total_mass(q) - s.boundary_flux[q]) / s.reg.phases[q].molar_mass;
    let gas_carbon = |s: &MultiphaseSim| gas_c(s, co2) + gas_c(s, co) + gas_c(s, ch4);
    let carbon = |s: &MultiphaseSim| s.total_mass(c) / 0.012011 + s.total_mass(hou) / s.reg.phases[hou].molar_mass + gas_carbon(s);
    let (m_c0, n_c0, g_c0, m_h0) = (s.total_mass(c), carbon(&s), gas_carbon(&s), s.total_mass(hou));
    let t_start = std::time::Instant::now();
    let mut t_hot = vec![];
    let t_end = std::env::var("MP_COAL_T").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0f64);
    // tirage naturel (l'air monte à travers la grille, percolation), moyen sur la seconde moitié ; flamme : gaz le plus chaud hors du lit
    let gj = (0..s.ny).find(|&j| (0..s.nx).any(|i| s.is_grille(s.idx(i, j)))).unwrap();
    let draft_of = |s: &MultiphaseSim| (0..s.nx).filter(|&i| s.is_grille(s.idx(i, gj))).map(|i| s.q_v[(gj + 1) * s.nx + i]).sum::<f64>() * s.dx;
    let gg = s.group_of[o2];
    let (mut d_int, mut d_t, mut t_flame) = (0.0, 0.0, 0.0f64);
    // chaleur dégagée par la flamme (hors du lit) pendant la dernière demi-seconde
    let (mut q_late, mut t_late) = (0.0, 0.0);
    while s.time < t_end && !s.diverged {
        let dt = s.step();
        if s.time > 0.5 * t_end {
            d_int += dt * draft_of(&s);
            d_t += dt;
        }
        if s.time > t_end - 0.5 {
            let q: f64 = (0..s.nx * s.ny).filter(|&q| !s.is_solid(q) && s.mass[c][q] + s.mass[hou][q] < 1.0).map(|q| s.heat_release[q].max(0.0)).sum();
            q_late += dt * q * s.dx * s.dx;
            t_late += dt;
        }
        if s.time > 0.3 {
            let n = s.nx * s.ny;
            let tf = (0..n).filter(|&q| !s.is_solid(q) && s.mass[c][q] + s.mass[hou][q] < 1.0 && s.group_mass(gg, q) > 0.0).map(|q| s.tg[gg][q]).fold(0.0f64, f64::max);
            t_flame = t_flame.max(tf);
        }
        if s.step_count % 1000 == 0 {
            let tmax = (0..s.nx * s.ny).filter(|&q| s.mass[c][q] > 1.0).map(|q| s.tg[s.group_of[c]][q]).fold(0.0f64, f64::max);
            t_hot.push((s.time, tmax));
        }
    }
    assert!(!s.diverged && s.state_is_sane());
    // carbone passé dans le gaz (CO, CO2, CH4 de la houille, sorties comprises), en kg de carbone
    let burnt = (gas_carbon(&s) - g_c0) * 0.012011;
    let tmax = t_hot.last().map_or(0.0, |x| x.1);
    let draft = d_int / d_t.max(1e-30);
    let flame_late = q_late / t_late.max(1e-30);
    eprintln!("feu de charbon : {} pas en {:.1} s ; carbone passé dans le gaz {:.3e} kg/m ({:.2} % du charbon), CO2 {:.3e} kg/m, CO {:.3e}, CH4 {:.3e}, O2 {:.3e} ; houille dégazée {:.3} % ; flamme {t_flame:.0} K, {flame_late:.3e} W/m la dernière demi-seconde ; tirage moyen {draft:.4} m²/s ; T max des braises {:?}",
        s.step_count, t_start.elapsed().as_secs_f64(), burnt, 100.0 * burnt / m_c0, s.total_mass(co2), s.total_mass(co), s.total_mass(ch4), s.total_mass(o2),
        100.0 * (1.0 - s.total_mass(hou) / m_h0.max(1e-300)), t_hot.iter().map(|(t, x)| format!("{t:.1}s:{x:.0}K")).collect::<Vec<_>>());
    assert!(burnt > 1e-5 * m_c0, "le charbon n'a pas brûlé");
    assert!(tmax > 1000.0, "le feu s'est éteint : braises à {tmax} K");
    assert!(draft > 1e-3, "pas de tirage à travers la grille : {draft}");
    // les matières volatiles de la houille (0.8 %/s de sa masse) brûlent en grande flamme au-dessus du lit
    assert!(m_h0 - s.total_mass(hou) > 0.01 * m_h0, "la houille ne dégaze pas : {} → {}", m_h0, s.total_mass(hou));
    assert!(t_flame > 1700.0, "pas de flamme au-dessus du lit : {t_flame} K");
    assert!(flame_late > 1e4, "la flamme s'est éteinte : {flame_late} W/m la dernière demi-seconde");
    assert!((carbon(&s) / n_c0 - 1.0).abs() < 1e-6, "carbone (charbon + houille + CO + CO2 + CH4 + sorties) : {} → {}", n_c0, carbon(&s));
}

/// diagnostic : braséro, débits de percolation à travers la grille et le lit, températures, oxygène (MP_COAL_T s, défaut 1)
#[test]
#[ignore]
fn coal_brasero_diag() {
    let lib = MaterialLibrary::default();
    let mut s = build_scenario_with(&lib, MpScenario::CoalFire, 40, 40, 0.01, 9.81).unwrap();
    let k = |n: &str| idx(&s.reg, n);
    let (c, o2) = (k(super::mp_materials::COAL), k(O2));
    let t_end = std::env::var("MP_COAL_T").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0f64);
    let grate_j = (0..s.ny).find(|&j| (0..s.nx).any(|i| s.is_grille(s.idx(i, j)))).unwrap();
    let column = |s: &MultiphaseSim, tag: &str| {
        let i = s.nx / 2;
        let p0 = s.p[s.idx(i, grate_j + 8)];
        eprintln!("-- {tag} : colonne centrale (j, p − p(j+8) [fluide], σ' [contact], p + σ' − p(j+8) [mélange], ρ_f, T)");
        for j in (grate_j.saturating_sub(3)..grate_j + 9).rev() {
            let q = s.idx(i, j);
            let rf: f64 = (0..s.reg.phases.len()).filter(|&k| !s.reg.phases[k].is_granular()).map(|k| s.mass[k][q]).sum::<f64>()
                / (0..s.reg.phases.len()).filter(|&k| !s.reg.phases[k].is_granular()).map(|k| s.alpha[k][q]).sum::<f64>().max(1e-12);
            eprintln!("   j {j:2} {}{:9.3} {:9.3} {:9.3} {:6.3} {:6.0}", if s.is_grille(q) { "G " } else { "  " }, s.p[q] - p0, s.sigma[q], s.p[q] + s.sigma[q] - p0, rf, s.temp[q]);
        }
    };
    column(&s, "départ");
    let mut shown = false;
    let names: Vec<String> = s.reg.reactions.iter().map(|r| r.name.clone()).collect();
    let (co, co2) = (k(CO), k(CO2));
    while s.time < t_end && !s.diverged {
        if s.step_count % 300 == 0 && s.step_count > 0 {
            let reacted: Vec<String> = names.iter().zip(&s.reacted).map(|(n, x)| format!("{}: {:.3e}", n.chars().take(18).collect::<String>(), x)).collect();
            let tgas = (0..s.nx * s.ny).filter(|&q| !s.is_solid(q) && s.mass[c][q] < 1e-3).map(|q| s.temp[q]).fold(0.0f64, f64::max);
            // CO au-dessus du lit (fraction molaire max hors du lit)
            let xco = (0..s.nx * s.ny).filter(|&q| !s.is_solid(q) && s.mass[c][q] < 1e-3).map(|q| s.alpha[co][q]).fold(0.0f64, f64::max);
            eprintln!("   fraction de CO max hors du lit {xco:.4}");
            eprintln!("   CO {:.3e} CO2 {:.3e} kg/m (sorties CO {:.3e} CO2 {:.3e}) ; T max du gaz hors du lit {tgas:.0} K ; avancements {:?}",
                s.total_mass(co), s.total_mass(co2), -s.boundary_flux[co], -s.boundary_flux[co2], reacted);
        }
        if !shown && s.step_count == 50 {
            column(&s, "pas 50");
            shown = true;
        }
        s.step();
        if s.step_count % 150 == 0 {
            let (nx, ny) = (s.nx, s.ny);
            let row_q = |j: usize| (0..nx).filter(|&i| s.is_grille(s.idx(i, grate_j))).map(|i| s.q_v[j * nx + i]).sum::<f64>();
            let tmax = (0..nx * ny).filter(|&q| !s.is_solid(q)).map(|q| s.temp[q]).fold(0.0f64, f64::max);
            let bed_o2: Vec<String> = (grate_j + 1..grate_j + 6).map(|j| {
                let q = s.idx(nx / 2, j);
                format!("{:.3}", s.alpha[o2][q] / (1.0 - s.alpha[c][q]).max(1e-9))
            }).collect();
            let tb: Vec<String> = (grate_j + 1..grate_j + 7).map(|j| format!("{:.0}", s.tg[s.group_of[c]][s.idx(nx / 2, j)])).collect();
            let top_j = (grate_j + 1..ny).find(|&j| s.alpha[c][s.idx(nx / 2, j)] < 0.3).unwrap_or(ny - 1);
            eprintln!("t {:.2} dt {:.1e} : Σq grille→lit {:.4} ; dessus du lit {:.4} m²/s ; T max {tmax:.0} K ; O2 pores {:?} ; T grains {:?} ; |u|max {:.2} ; brûlé {:.3e}",
                s.time, s.last_dt, row_q(grate_j + 1) * s.dx, row_q(top_j) * s.dx, bed_o2, tb, s.max_speed(), s.reacted.iter().sum::<f64>());
        }
    }
}


// ===================================================================================================== pyrolyse du bois

/// bois en morceaux (fraction a0) dans de l'azote à T, sans gravité
pub(crate) fn wood_dust(t: f64, a0: f64) -> MultiphaseSim {
    grain_dust(&MaterialLibrary::default().with_wood(), super::mp_materials::WOOD, t, a0)
}

/// morceaux de la phase granulaire `solid` de `lib` (fraction a0) dans de l'azote à T, sans gravité
fn grain_dust(lib: &MaterialLibrary, solid: &str, t: f64, a0: f64) -> MultiphaseSim {
    let b = lib.build().expect("préréglage valide");
    assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    let reg = b.registry;
    let (w, n2) = (idx(&reg, solid), idx(&reg, N2));
    let np = reg.phases.len();
    let mut s = MultiphaseSim::new(4, 4, 0.01, reg, b.ambient);
    s.gravity = [0.0, 0.0];
    let mut m = vec![0.0; np];
    m[n2] = 1.0 - a0;
    m[w] = a0;
    s.paint_mixture(move |_, _| Some(m.clone()));
    s.finalize_init(move |_, _| t);
    s.refresh();
    s
}

/// (vitesse de pyrolyse mesurée sur un appel de la chimie, loi k(T)·c_bois (mol/(m³·s)), charbon de bois formé / bois consommé
/// (masse), CH4 / CO formés (moles))
fn wood_rates(t: f64) -> (f64, f64, f64, f64) {
    use super::mp_materials::{CHARCOAL, RX_WOOD, WOOD};
    pyrolysis_rates(&MaterialLibrary::default().with_wood(), WOOD, CHARCOAL, RX_WOOD, t)
}

/// `wood_rates` pour la pyrolyse `rx_name` de la phase `solid` de `lib`, qui laisse le résidu `residue`
fn pyrolysis_rates(lib: &MaterialLibrary, solid: &str, residue: &str, rx_name: &str, t: f64) -> (f64, f64, f64, f64) {
    let mut s = grain_dust(lib, solid, t, 1e-3);
    let (w, ch, co, ch4) = (idx(&s.reg, solid), idx(&s.reg, residue), idx(&s.reg, CO), idx(&s.reg, CH4));
    let q = s.idx(1, 1);
    let mw = s.reg.phases[w].molar_mass;
    let r = rx(&s.reg, rx_name);
    let k = r.k_forward(s.tg[s.group_of[w]][q]);
    let tau = r.bulk_tau(&s.reg.phases);
    let expect = k / (1.0 + k * tau) * s.mass[w][q] / mw;
    let (mw0, mc0, mco0, mch40) = (s.mass[w][q], s.mass[ch][q], s.mass[co][q], s.mass[ch4][q]);
    let dt = 1e-4;
    s.pyrolysis_step(dt);
    let dmw = mw0 - s.mass[w][q];
    let got = dmw / mw / dt;
    let n_co = (s.mass[co][q] - mco0) / s.reg.phases[co].molar_mass;
    let n_ch4 = (s.mass[ch4][q] - mch40) / s.reg.phases[ch4].molar_mass;
    (got, expect, (s.mass[ch][q] - mc0) / dmw.max(1e-300), n_ch4 / n_co.max(1e-300))
}

#[test]
fn wood_pyrolysis_follows_its_rate_law_and_leaves_charcoal() {
    // réaction en volume, du premier ordre en bois : vitesse = k(T) c_bois ; 22 % du bois en charbon de bois, CH4 / CO = 0.18 / 0.30
    for t in [700.0, 800.0] {
        let (got, expect, yield_c, ch4_co) = wood_rates(t);
        eprintln!("T = {t} K : pyrolyse {got:.5e} mol/(m³·s) (loi {expect:.5e}) ; charbon de bois {:.4} de la masse ; CH4/CO {ch4_co:.4}", yield_c);
        assert!((got / expect - 1.0).abs() < 2e-3, "vitesse de pyrolyse à {t} K : {got} vs {expect}");
        assert!((yield_c - 0.44 * 0.012011 / 0.024022).abs() < 1e-4, "rendement en charbon de bois : {yield_c}");
        assert!((ch4_co - 0.6).abs() < 1e-6, "CH4/CO : {ch4_co}");
    }
    // froid : le bois ne se décompose pas
    let (got, _, _, _) = wood_rates(300.0);
    assert!(got < 1e-12, "pyrolyse à 300 K : {got}");
    // réaction en volume (pas de surface) ; morceaux de 2 cm : temps de conduction interne ~1 min, qui borne la vitesse à chaud
    let lib = MaterialLibrary::default().with_wood();
    let b = lib.build().unwrap();
    let r = b.registry.reactions.iter().find(|r| r.name == super::mp_materials::RX_WOOD).unwrap();
    assert!(r.bulk && r.surface(&b.registry.phases).is_none());
    let tau = r.bulk_tau(&b.registry.phases);
    eprintln!("temps de conduction interne d'un morceau de 2 cm : {tau:.1} s");
    assert!((tau - 65.9).abs() < 1.0, "τ = {tau}");
    let (hot, _, _, _) = wood_rates(1400.0);
    let c_w = 1e-3 * 650.0 / 0.024022;
    assert!(hot < 1.02 * c_w / tau && hot > 0.9 * c_w / tau, "à 1400 K la vitesse est bornée par la chaleur : {hot} vs {}", c_w / tau);
}

#[test]
fn houille_releases_its_volatiles_and_leaves_its_char() {
    // pyrolyse de la houille (Badzioch & Hawksley, charbons bitumineux) : du premier ordre, 0.1 s⁻¹ vers 600 K, 1 s⁻¹ vers 700 K ;
    // 35 % de matières volatiles, surtout du méthane (CH4/CO = 0.187/0.04), et 65 % de char, qui brûle ensuite comme le charbon ;
    // bornée à chaud par la conduction interne des morceaux (1.2 cm : τ ≈ 19 s)
    use super::mp_materials::{COAL, HOUILLE, RX_HOUILLE};
    let lib = MaterialLibrary::default().with_coal();
    for t in [600.0, 700.0] {
        let (got, expect, yield_c, ch4_co) = pyrolysis_rates(&lib, HOUILLE, COAL, RX_HOUILLE, t);
        eprintln!("T = {t} K : pyrolyse {got:.5e} mol/(m³·s) (loi {expect:.5e}) ; char {yield_c:.4} de la masse ; CH4/CO {ch4_co:.4}");
        assert!((got / expect - 1.0).abs() < 2e-3, "vitesse de pyrolyse à {t} K : {got} vs {expect}");
        assert!((yield_c - 0.763 * 0.012011 / 0.014097).abs() < 1e-4, "rendement en char : {yield_c}");
        assert!((ch4_co - 0.187 / 0.04).abs() < 1e-6, "CH4/CO : {ch4_co}");
    }
    let b = lib.build().unwrap();
    let r = b.registry.reactions.iter().find(|r| r.name == RX_HOUILLE).unwrap();
    assert!(r.bulk && r.surface(&b.registry.phases).is_none());
    let (k3, k6, k7) = (r.k_forward(300.0), r.k_forward(600.0), r.k_forward(700.0));
    let tau = r.bulk_tau(&b.registry.phases);
    eprintln!("k(300 K) = {k3:.2e}, k(600 K) = {k6:.3}, k(700 K) = {k7:.3} s⁻¹ ; temps de conduction interne d'un morceau de 1.2 cm : {tau:.1} s");
    assert!(k6 > 0.09 && k6 < 0.13 && k7 > 0.8 && k7 < 1.1, "k = {k6}, {k7}");
    // froide, la houille se conserve (loi extrapolée : 0.3 % par jour à 300 K)
    assert!(k3 < 1e-7, "k(300 K) = {k3}");
    assert!((tau - 19.0).abs() < 1.0, "τ = {tau}");
    // pouvoir calorifique supérieur 34 MJ/kg : CH0.8O0.08 + 1.16 O2 → CO2 + 0.4 H2O(l)
    let ph = &b.registry.phases;
    let hf = |n: &str| ph[idx(&b.registry, n)].h_formation;
    let pcs = -(hf(CO2) + 0.4 * (hf(H2O_G) - 44.0e3) - hf(HOUILLE)) / ph[idx(&b.registry, HOUILLE)].molar_mass;
    eprintln!("pouvoir calorifique supérieur de la houille : {:.2} MJ/kg", pcs / 1e6);
    assert!((pcs / 1e6 - 34.0).abs() < 0.5, "PCS {pcs}");
}

// ===================================================================================================== poudre noire

/// poudre noire en grains (fraction a0) dans de l'azote à T, sans gravité ; `only_powder` : sans les réactions du gaz (composition des
/// produits figée — vers 2500 K, la dissociation de CO2 et de H2O changerait leur nombre de moles)
pub(crate) fn gunpowder_dust(t: f64, a0: f64, only_powder: bool) -> MultiphaseSim {
    let mut lib = MaterialLibrary::default().with_gunpowder();
    if only_powder {
        lib.reactions.retain(|r| r.name == super::mp_materials::RX_POUDRE);
    }
    grain_dust(&lib, super::mp_materials::POUDRE, t, a0)
}

#[test]
fn gunpowder_burns_at_the_vieille_rate_and_releases_its_heat() {
    // poudre noire : combustion de surface sans air, r = 1.1 cm/s (p / 1 atm)^0.15 dès 580 K — des grains de 2 mm sont consommés à
    // 6 r / d = 33 s⁻¹ ; produits : 42 % de gaz (12 mol/kg), 58 % de fumée ; chaleur d'explosion 3.0 MJ/kg (Noble & Abel : 3.04, eau
    // liquide), flamme à volume constant vers 2600 K, « force » n R T_v ≈ 260 kJ/kg ; énergie exactement conservée
    use super::mp_materials::{FUMEE, POUDRE, RX_POUDRE};
    let s0 = gunpowder_dust(700.0, 1e-3, false);
    let reg = &s0.reg;
    let r = rx(reg, RX_POUDRE);
    let (w, f) = (idx(reg, POUDRE), idx(reg, FUMEE));
    assert!(r.bulk && r.burn().is_some() && r.surface(&reg.phases).is_none() && !reg.is_cell_reactive(w));
    let mw = reg.phases[w].molar_mass;
    let gas = |x: &(usize, f64)| reg.phases[x.0].is_gas();
    let m_gas: f64 = r.products.iter().filter(|x| gas(x)).map(|&(k, nu)| nu * reg.phases[k].molar_mass).sum::<f64>() / mw;
    let n_gas: f64 = r.products.iter().filter(|x| gas(x)).map(|x| x.1).sum::<f64>() / mw;
    let (ma, mb) = r.mass_balance(&reg.phases);
    let qv = -r.delta_u(&reg.phases, 298.15) / mw;
    let tv = r.adiabatic_products_temp(&reg.phases, 298.15);
    let force = n_gas * R_GAS * tv;
    eprintln!("poudre noire : gaz {:.2} % de la masse, {n_gas:.2} mol/kg ; chaleur d'explosion {:.3} MJ/kg ; flamme à volume constant {tv:.0} K ; force {:.0} kJ/kg",
        100.0 * m_gas, qv / 1e6, force / 1e3);
    assert!((ma / mb - 1.0).abs() < 1e-12, "bilan de masse : {ma} → {mb}");
    assert!((m_gas - 0.4196).abs() < 1e-3 && (n_gas - 12.01).abs() < 0.01, "gaz : {m_gas}, {n_gas} mol/kg");
    assert!((qv / 1e6 - 3.02).abs() < 0.03, "chaleur d'explosion {qv} J/kg");
    assert!(tv > 2400.0 && tv < 2800.0, "T_v = {tv}");
    assert!(force > 2.4e5 && force < 3.0e5, "force {force} J/kg");
    // vitesse dans une cellule à 700 K, sous 1 atm puis sous 10 atm (×10^0.15 = ×1.41)
    for (p, ratio) in [(101325.0, 1.0), (1013250.0, 10f64.powf(0.15))] {
        let mut s = gunpowder_dust(700.0, 1e-3, false);
        let c = s.idx(1, 1);
        s.p.iter_mut().for_each(|x| *x = p);
        let (m0, u0) = (s.mass[w][c], s.total_internal_energy());
        let dt = 1e-5;
        s.pyrolysis_step(dt);
        let k = -(s.mass[w][c] / m0).ln() / dt;
        let expect = 6.0 * 0.011 * ratio / s.reg.phases[w].grain_diameter;
        eprintln!("p = {:.1} bar : poudre consommée à {k:.4} s⁻¹ (6 r / d = {expect:.4}) ; fumée {:.3e} kg/m³ ; borne du pas {:.3e} s", p / 1e5, s.mass[f][c], s.burn_dt_bound());
        assert!((k / expect - 1.0).abs() < 1e-6, "vitesse sous {p} Pa : {k} contre {expect}");
        assert!(s.mass[f][c] > 0.0 && s.burn_dt_bound().is_finite());
        assert!((s.total_internal_energy() / u0 - 1.0).abs() < 1e-12, "énergie");
    }
    // sous la température d'allumage, rien ne brûle
    let mut s = gunpowder_dust(500.0, 1e-3, false);
    let m0 = s.total_mass(w);
    s.pyrolysis_step(1e-3);
    assert!(s.total_mass(w) == m0, "la poudre brûle à 500 K");
}

#[test]
fn gunpowder_explodes_in_a_closed_vessel_to_its_isochoric_pressure() {
    // 10 kg/m³ de poudre noire dans une boîte fermée d'azote, allumée partout à 700 K : elle brûle en ~0.1 s ; à la fin, gaz et fumée
    // sont à la température T_v qui conserve l'énergie interne, sous p_v = n_gaz R T_v / V_gaz (la boîte moins le volume de la fumée)
    use super::mp_materials::POUDRE;
    let mut s = gunpowder_dust(700.0, 10.0 / 1800.0, true);
    // (gaz optiquement mince : sous 30 bar de CO2 à 2800 K, le rayonnement viderait la boîte de sa chaleur — test de thermodynamique)
    s.radiation = false;
    let w = idx(&s.reg, POUDRE);
    let m0 = s.total_mass(w);
    let u0 = s.total_internal_energy() + s.kinetic_energy();
    while s.time < 0.3 && !s.diverged {
        s.step();
    }
    assert!(!s.diverged && s.state_is_sane());
    let burnt = 1.0 - s.total_mass(w) / m0;
    // état isochore attendu, avec les masses finales
    let np = s.reg.phases.len();
    let mass: Vec<f64> = (0..np).map(|k| s.total_mass(k)).collect();
    let (mut ce, mut sl, mut e0) = (0.0, 0.0, 0.0);
    for (k, p) in s.reg.phases.iter().enumerate() {
        ce += mass[k] * p.c_e();
        sl += mass[k] * p.cp_slope;
        e0 += mass[k] * p.e0;
    }
    let tv = super::phases::adiabatic_t(ce, sl, e0, u0);
    let vol = (s.nx * s.ny) as f64 * s.dx * s.dx;
    let v_cond: f64 = (0..np).filter(|&k| !s.reg.phases[k].is_gas()).map(|k| mass[k] / s.reg.phases[k].rho(2e6, tv)).sum();
    let n_gas: f64 = (0..np).filter(|&k| s.reg.phases[k].is_gas()).map(|k| mass[k] / s.reg.phases[k].molar_mass).sum();
    let pv = n_gas * R_GAS * tv / (vol - v_cond);
    let p_sim = s.p.iter().sum::<f64>() / s.p.len() as f64;
    let t_sim = s.temp.iter().sum::<f64>() / s.temp.len() as f64;
    eprintln!("vase clos : {} pas, poudre brûlée {:.5} % ; T {t_sim:.1} K (attendue {tv:.1}), p {:.3} bar (attendue {:.3} bar)",
        s.step_count, 100.0 * burnt, p_sim / 1e5, pv / 1e5);
    assert!(burnt > 0.999, "poudre restante : {}", 1.0 - burnt);
    assert!((t_sim - tv).abs() < 5.0, "T finale {t_sim} contre {tv}");
    assert!((p_sim / pv - 1.0).abs() < 0.01, "p finale {p_sim} contre {pv}");
    assert!(((s.total_internal_energy() + s.kinetic_energy()) / u0 - 1.0).abs() < 1e-9, "énergie");
}

#[test]
fn a_gunpowder_heap_deflagrates_into_white_smoke() {
    // tas de poudre noire allumé sur un flanc (40 × 30, 0.8 m × 0.6 m) : les gaz chauds traversent le tas et allument les grains qu'ils
    // baignent ; le tas fuse en boule de feu (gaz vers 2800 K) et laisse sa fumée, 58 % de la masse brûlée ; masse conservée (sorties
    // par le haut comprises). Une seule vitesse par cellule : les grains n'ont pas d'inertie propre, le souffle (gaz jaillissant des
    // grains tassés à plusieurs centaines de m/s) en emporte la moitié hors du domaine avant qu'ils aient brûlé
    use super::mp_materials::{FUMEE, POUDRE};
    let mut s = build_scenario_with(&MaterialLibrary::default(), MpScenario::Gunpowder, 40, 30, 0.02, 9.81).unwrap();
    let k = |n: &str| idx(&s.reg, n);
    let (w, f, n2) = (k(POUDRE), k(FUMEE), k(N2));
    let gg = s.group_of[n2];
    let np = s.reg.phases.len();
    let kept = |s: &MultiphaseSim| (0..np).map(|q| s.total_mass(q) - s.boundary_flux[q]).sum::<f64>();
    let (m0, all0) = (s.total_mass(w), kept(&s));
    let mut t_gas = 0.0f64;
    while s.time < 0.03 && !s.diverged {
        s.step();
        let tg = (0..s.nx * s.ny).filter(|&c| !s.is_solid(c) && s.group_mass(gg, c) > 0.0).map(|c| s.tg[gg][c]).fold(0.0f64, f64::max);
        t_gas = t_gas.max(tg);
    }
    assert!(!s.diverged && s.state_is_sane());
    // (poudre brûlée : celle qui a disparu, moins celle que le souffle a projetée hors du domaine par le haut ouvert)
    let burnt = m0 - (s.total_mass(w) - s.boundary_flux[w]);
    let smoke = s.total_mass(f) - s.boundary_flux[f];
    let ratio = s.reg.phases[f].molar_mass / s.reg.phases[w].molar_mass;
    eprintln!("tas de poudre : {} pas, {} refaits ; poudre brûlée {:.1} % de {m0:.3} kg/m en 30 ms (projetée dehors {:.3} kg/m) ; fumée {smoke:.4} kg/m ({:.4} de la masse brûlée, attendu {ratio:.4}) ; gaz jusqu'à {t_gas:.0} K",
        s.step_count, s.rollbacks, 100.0 * burnt / m0, -s.boundary_flux[w], smoke / burnt);
    assert!(burnt > 0.3 * m0, "le tas n'a pas brûlé : {burnt} sur {m0}");
    assert!(burnt - s.boundary_flux[w] > 0.85 * m0, "poudre restée dans le tas : {}", s.total_mass(w));
    assert!(t_gas > 2000.0, "pas de flamme : {t_gas} K");
    assert!((smoke / burnt / ratio - 1.0).abs() < 1e-6, "fumée {smoke} pour {burnt} brûlés");
    assert!((kept(&s) / all0 - 1.0).abs() < 1e-9, "masse : {all0} → {}", kept(&s));
}

/// Lit tassé de 4 × 6 mailles de 1 cm, sans gravité ni chimie, dans l'azote : braises de charbon de bois (1450 K) dans les 3 rangées du
/// bas et bois (293 K) au-dessus ; ou (`mixed`) braises et bois mêlés à parts égales dans toutes les mailles, gaz et bois à 293 K
pub(crate) fn ember_bed(mixed: bool) -> MultiphaseSim {
    use super::mp_materials::{CHARCOAL, WOOD};
    let lib = MaterialLibrary::default().with_wood();
    let b = lib.build().expect("préréglage bois valide");
    let reg = b.registry;
    let (w, ch, n2) = (idx(&reg, WOOD), idx(&reg, CHARCOAL), idx(&reg, N2));
    let np = reg.phases.len();
    let (aw, ac) = (reg.phases[w].alpha_max, reg.phases[ch].alpha_max);
    let mut s = MultiphaseSim::new(4, 6, 0.01, reg, b.ambient);
    s.gravity = [0.0, 0.0];
    s.chemistry = false;
    let mix = move |fw: f64, fc: f64| {
        let mut m = vec![0.0; np];
        m[w] = fw * aw;
        m[ch] = fc * ac;
        m[n2] = 1.0 - m[w] - m[ch];
        m
    };
    let low = |y: f64| y < 0.03;
    s.paint_mixture(move |_, y| Some(if mixed { mix(0.5, 0.5) } else if low(y) { mix(0.0, 1.0) } else { mix(1.0, 0.0) }));
    s.finalize_init(move |_, y| if !mixed && low(y) { 1450.0 } else { 293.15 });
    if mixed {
        let gc = s.group_of[ch];
        for c in 0..s.nx * s.ny {
            s.set_group_temperature(c, gc, 1450.0);
        }
    }
    s.refresh();
    s
}

#[test]
fn embers_heat_the_wood_above_them_by_radiation_through_the_bed() {
    // À haute température, les morceaux d'un tas échangent surtout par rayonnement, qui y agit comme une conduction de conductivité
    // k_rad = 4 σ ε/(2 − ε) d T_m³ (Damköhler) : 3.1 W/(m·K) entre des braises à 1450 K et du bois à 293 K (morceaux de 1.5 et 2 cm),
    // ~50× la conduction des morceaux et du gaz. La rangée de bois posée sur les braises reçoit d'abord k_rad ΔT par mètre de face
    use super::mp_granular::bed_rad_k;
    use super::mp_materials::{CHARCOAL, WOOD};
    let mut s = ember_bed(false);
    let (w, ch) = (idx(&s.reg, WOOD), idx(&s.reg, CHARCOAL));
    let gw = s.group_of[w];
    let e_w = |s: &MultiphaseSim| (0..s.nx * s.ny).map(|q| s.eg[gw][q]).sum::<f64>() * s.dx * s.dx;
    let (e0, u0) = (e_w(&s), s.total_internal_energy() + s.kinetic_energy());
    let d = 0.5 * (s.reg.phases[w].grain_diameter + s.reg.phases[ch].grain_diameter);
    let k_rad = bed_rad_k(d, 1450.0, 293.15, 1.0, 1.0);
    while s.time < 0.01 {
        s.step();
    }
    let k_eff = (e_w(&s) - e0) / (s.time * s.nx as f64 * (1450.0 - 293.15));
    eprintln!("k_rad = {k_rad:.4} W/(m·K) ; conductance mesurée entre les braises et le bois : {k_eff:.4} W/(m·K) ({} pas)", s.step_count);
    assert!(k_rad > 3.0 && k_rad < 3.2, "k_rad = {k_rad}");
    assert!(k_eff / k_rad > 0.995 && k_eff / k_rad < 1.06, "conductance {k_eff} contre k_rad {k_rad}");
    assert!(((s.total_internal_energy() + s.kinetic_energy()) / u0 - 1.0).abs() < 1e-9, "énergie");
}

#[test]
fn embers_mixed_with_wood_heat_it_by_radiation_within_the_cell() {
    // braises (1450 K) et bois (293 K) mêlés dans chaque maille. Le bois reçoit le rayonnement des braises de sa maille : h = 4 σ ε/(2 − ε)
    // T_m³ sur l'aire des morceaux d'une population tournée vers l'autre (la plus petite de 6 α_1/d_1 · α_2/α_s et 6 α_2/d_2 · α_1/α_s) ;
    // et celui des braises des mailles voisines, à travers chaque face, comme dans un tas en couches (k_rad f_braises f_bois / dx², avec
    // f = α / α_max : le quart de k_rad ici) — de même ordre à 1 cm de maille pour des morceaux de 1.5–2 cm, négligeable sur une maille
    // bien plus grande qu'eux. Le gaz, chauffé par les braises, n'en porte que quelques pour cent de plus
    use super::mp_granular::{bed_rad_k, BED_RAD, SIGMA_SB};
    use super::mp_materials::{CHARCOAL, WOOD};
    let mut s = ember_bed(true);
    let (w, ch) = (idx(&s.reg, WOOD), idx(&s.reg, CHARCOAL));
    let (gw, gc) = (s.group_of[w], s.group_of[ch]);
    let c = s.idx(1, 2);
    let (aw, ac) = (s.group_alpha(gw, c), s.group_alpha(gc, c));
    let (dw, dc) = (s.reg.phases[w].grain_diameter, s.reg.phases[ch].grain_diameter);
    let area = (6.0 * aw / dw * ac / (aw + ac)).min(6.0 * ac / dc * aw / (aw + ac));
    let (t1, t2) = (1450.0f64, 293.15f64);
    let h = 4.0 * SIGMA_SB * BED_RAD * (t1 * t1 + t2 * t2) * (t1 + t2) * 0.25;
    let g_local = s.phase_heat_exchange * h * area;
    let (fw, fc) = (aw / s.reg.phases[w].alpha_max, ac / s.reg.phases[ch].alpha_max);
    let faces = ((s.nx - 1) * s.ny + s.nx * (s.ny - 1)) as f64;
    let ncell = (s.nx * s.ny) as f64;
    let g_faces = 2.0 * faces * bed_rad_k(0.5 * (dw + dc), t1, t2, fc, fw) / (ncell * s.dx * s.dx);
    let e_w = |s: &MultiphaseSim| (0..s.nx * s.ny).map(|q| s.eg[gw][q]).sum::<f64>() / (s.nx * s.ny) as f64;
    let (e0, u0) = (e_w(&s), s.total_internal_energy() + s.kinetic_energy());
    while s.time < 0.01 {
        s.step();
    }
    let g = (e_w(&s) - e0) / (s.time * (t1 - t2));
    eprintln!("aire d'échange {area:.2} m²/m³, h = {h:.2} W/(m²·K) : conductance dans la maille {g_local:.4e}, entre mailles {g_faces:.4e} W/(m³·K), mesurée {g:.4e} ({} pas)", s.step_count);
    let ratio = g / (g_local + g_faces);
    assert!(ratio > 0.995 && ratio < 1.06, "conductance {g} contre {} + {}", g_local, g_faces);
    assert!(((s.total_internal_energy() + s.kinetic_energy()) / u0 - 1.0).abs() < 1e-9, "énergie");
}

#[test]
fn a_wood_fire_burns_its_gases_in_a_flame_and_conserves_carbon() {
    // feu établi (braises 1450 K, bûches à 750 K, flamme d'allumage) : le bois se décompose, ses gaz brûlent en flamme de diffusion
    // au-dessus du tas, bien plus chaude que les braises ; carbone (bois + charbon de bois + CO + CO2 + CH4 + sorties) conservé
    use super::mp_materials::{CHARCOAL, WOOD};
    let lib = MaterialLibrary::default();
    let mut s = build_scenario_with(&lib, MpScenario::WoodFire, 24, 24, 0.4 / 24.0, 9.81).unwrap();
    let k = |n: &str| idx(&s.reg, n);
    let (w, ch, ch4, co, co2, o2) = (k(WOOD), k(CHARCOAL), k(CH4), k(CO), k(CO2), k(O2));
    let gg = s.group_of[o2];
    let gas_c = |s: &MultiphaseSim, q: usize| (s.total_mass(q) - s.boundary_flux[q]) / s.reg.phases[q].molar_mass;
    let carbon = |s: &MultiphaseSim| {
        s.total_mass(w) / s.reg.phases[w].molar_mass + s.total_mass(ch) / 0.012011 + gas_c(s, co) + gas_c(s, co2) + gas_c(s, ch4)
    };
    let (m_w0, n_c0) = (s.total_mass(w), carbon(&s));
    let t_end = std::env::var("MP_WOOD_T").ok().and_then(|v| v.parse().ok()).unwrap_or(0.4f64);
    let mut t_flame = 0.0f64;
    while s.time < t_end && !s.diverged {
        s.step();
        if s.time > 0.15 {
            let tmax = (0..s.nx * s.ny).filter(|&q| !s.is_solid(q) && s.group_mass(gg, q) > 0.0).map(|q| s.tg[gg][q]).fold(0.0f64, f64::max);
            t_flame = t_flame.max(tmax);
        }
    }
    assert!(!s.diverged && s.state_is_sane());
    let pyro = 1.0 - s.total_mass(w) / m_w0;
    eprintln!("feu de bois : {} pas ; bois décomposé {:.3} % ; flamme {t_flame:.0} K ; CH4 {:.3e}, CO {:.3e} kg/m", s.step_count, 100.0 * pyro,
        s.total_mass(ch4), s.total_mass(co));
    assert!(pyro > 1e-3, "le bois ne se décompose pas : {pyro}");
    assert!(t_flame > 1700.0, "pas de flamme au-dessus du tas : {t_flame} K");
    assert!((carbon(&s) / n_c0 - 1.0).abs() < 1e-6, "carbone : {} → {}", n_c0, carbon(&s));
}

/// scénario du feu de bois pour les diagnostics : MP_WOOD_NX mailles de côté (défaut 40, 0.4 m) ; MP_WOOD_T0 : température initiale
/// du bois (défaut : celle du scénario) ; MP_TFM_OFF : sans épaississement des flammes
pub(crate) fn wood_fire_for_diag() -> MultiphaseSim {
    let n: usize = std::env::var("MP_WOOD_NX").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let mut s = build_scenario_with(&MaterialLibrary::default(), MpScenario::WoodFire, n, n, 0.4 / n as f64, 9.81).unwrap();
    if std::env::var("MP_TFM_OFF").is_ok() {
        s.flame_resolution = 0.0;
    }
    if let Some(t0) = std::env::var("MP_WOOD_T0").ok().and_then(|v| v.parse::<f64>().ok()) {
        let w = idx(&s.reg, super::mp_materials::WOOD);
        let gw = s.group_of[w];
        for c in 0..s.nx * s.ny {
            if s.mass[w][c] > 0.0 {
                s.set_group_temperature(c, gw, t0);
            }
        }
        s.refresh();
    }
    s
}

/// une ligne de diagnostic du feu de bois (`m_w0` : bois au départ, `wall` : durée de calcul)
pub(crate) fn wood_fire_report(s: &MultiphaseSim, m_w0: f64, wall: f64) {
    use super::mp_materials::{CHARCOAL, WOOD};
    let k = |n: &str| idx(&s.reg, n);
    let (w, ch, ch4, h2, co, co2, o2) = (k(WOOD), k(CHARCOAL), k(CH4), k(H2), k(CO), k(CO2), k(O2));
    let (gw, gg, gc) = (s.group_of[w], s.group_of[o2], s.group_of[ch]);
    let n = s.nx * s.ny;
    let tg_max = (0..n).filter(|&q| !s.is_solid(q) && s.group_mass(gg, q) > 0.0).map(|q| s.tg[gg][q]).fold(0.0f64, f64::max);
    let tw_max = (0..n).filter(|&q| s.mass[w][q] > 1e-3).map(|q| s.tg[gw][q]).fold(0.0f64, f64::max);
    let tw_mean = (0..n).map(|q| s.mass[w][q] * s.tg[gw][q]).sum::<f64>() / (0..n).map(|q| s.mass[w][q]).sum::<f64>().max(1e-300);
    let tc_max = (0..n).filter(|&q| s.mass[ch][q] > 1e-3).map(|q| s.tg[gc][q]).fold(0.0f64, f64::max);
    let bed: Vec<usize> = (0..n).filter(|&q| s.mass[w][q] + s.mass[ch][q] > 1.0).collect();
    let x_o2 = |q: usize| {
        let tot: f64 = (0..s.reg.phases.len()).filter(|&k| s.reg.phases[k].is_gas()).map(|k| s.mass[k][q] / s.reg.phases[k].molar_mass).sum();
        s.mass[o2][q] / s.reg.phases[o2].molar_mass / tot.max(1e-300)
    };
    let o2_bed = bed.iter().map(|&q| x_o2(q)).sum::<f64>() / bed.len().max(1) as f64;
    let gj = (0..s.ny).find(|&j| (0..s.nx).any(|i| s.is_grille(s.idx(i, j)))).unwrap();
    let draft: f64 = (0..s.nx).filter(|&i| s.is_grille(s.idx(i, gj))).map(|i| s.q_v[(gj + 1) * s.nx + i]).sum::<f64>() * s.dx;
    let (mut below, mut inbed, mut above, mut outside) = (0.0, 0.0, 0.0, 0.0);
    let bed_top = (0..s.ny).rev().find(|&j| (0..s.nx).any(|i| s.mass[w][s.idx(i, j)] + s.mass[ch][s.idx(i, j)] > 1.0)).unwrap_or(0);
    let (xl, xr) = ((0..s.nx).find(|&i| s.is_grille(s.idx(i, gj))).unwrap_or(0), (0..s.nx).rev().find(|&i| s.is_grille(s.idx(i, gj))).unwrap_or(0));
    for j in 0..s.ny {
        for i in 0..s.nx {
            let q = s.idx(i, j);
            let m = s.mass[ch4][q];
            if j < gj { below += m } else if i < xl || i > xr { outside += m } else if j <= bed_top { inbed += m } else { above += m }
        }
    }
    eprintln!("      CH4 sous la grille {below:.3e}, dans le tas {inbed:.3e}, au-dessus du tas {above:.3e}, hors des parois {outside:.3e} (kg/m³·mailles)");
    let made_co2 = (s.total_mass(co2) - s.boundary_flux[co2]) / s.reg.phases[co2].molar_mass;
    eprintln!(
        "t {:.2} s ({} pas, {wall:.0} s) : bois {:.4} kg/m ({:.2} %), charbon de bois {:.4}, CH4 {:.3e}, H2 {:.3e}, CO {:.3e}, CO2 formé {made_co2:.3e} mol/m ; T gaz max {tg_max:.0} K, T bois moyenne {tw_mean:.0} K (max {tw_max:.0} K), T braises max {tc_max:.0} K ; O2 moyen dans le tas {o2_bed:.3} ; tirage {draft:.4} m²/s",
        s.time, s.step_count, s.total_mass(w), 100.0 * s.total_mass(w) / m_w0, s.total_mass(ch), s.total_mass(ch4), s.total_mass(h2),
        s.total_mass(co)
    );
}

/// diagnostic : feu de bois, une ligne toutes les MP_WOOD_DT s (défaut 0.1) pendant MP_WOOD_T s (défaut 2) ; scène : `wood_fire_for_diag`
#[test]
#[ignore]
fn wood_fire_diag() {
    let mut s = wood_fire_for_diag();
    let m_w0 = s.total_mass(idx(&s.reg, super::mp_materials::WOOD));
    let t_end = std::env::var("MP_WOOD_T").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0f64);
    let every = std::env::var("MP_WOOD_DT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.1f64);
    let t0 = std::time::Instant::now();
    let mut next = 0.0;
    while s.time < t_end && !s.diverged {
        s.step();
        if s.time >= next {
            next += every;
            wood_fire_report(&s, m_w0, t0.elapsed().as_secs_f64());
        }
    }
    eprintln!("divergé : {}", s.diverged);
}

/// diagnostic : premiers pas du feu de bois (MP_WOOD_N pas, défaut 300)
#[test]
#[ignore]
fn wood_fire_first_steps() {
    let lib = MaterialLibrary::default();
    let sc = std::env::var("MP_SCENE").ok().and_then(|v| v.parse::<usize>().ok()).map_or(MpScenario::WoodFire, MpScenario::from_index);
    let mut s = build_scenario_with(&lib, sc, 40, 40, 0.01, 9.81).unwrap();
    let nmax: usize = std::env::var("MP_WOOD_N").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
    for k in 0..nmax {
        let rb = s.rollbacks;
        let dt = s.step();
        let n = s.nx * s.ny;
        let umax = s.u.iter().chain(s.v.iter()).fold(0.0f64, |a, &b| a.max(b.abs()));
        let (mut tmin, mut tmax) = (f64::INFINITY, 0.0f64);
        for g in 0..s.groups.len() {
            for q in 0..n {
                if !s.is_solid(q) && s.group_mass(g, q) > 1e-9 {
                    tmin = tmin.min(s.tg[g][q]);
                    tmax = tmax.max(s.tg[g][q]);
                }
            }
        }
        let (pmin, pmax) = s.p.iter().fold((f64::INFINITY, 0.0f64), |(a, b), &p| (a.min(p), b.max(p)));
        if k < 20 || k % 20 == 0 || s.rollbacks != rb || s.diverged {
            eprintln!("pas {k} : t {:.5} s, dt {dt:.2e} (chimie {:.2e}, phase {:.2e}, choisi {:.2e}), |u| max {umax:.2} m/s, T {tmin:.0}–{tmax:.0} K, p {pmin:.0}–{pmax:.0} Pa, refaits {}, PCG {}{}",
                s.time, s.dt_chem, s.dt_phase, s.choose_dt(), s.rollbacks, s.last_pressure_iters, if s.diverged { "  DIVERGÉ" } else { "" });
        }
        if s.diverged {
            break;
        }
    }
}

/// diagnostic : maille la plus chaude du feu de bois 24×24 (après 0.15 s) : composition, fractions, températures de groupe
#[test]
#[ignore]
fn wood_fire_hottest_cell() {
    let lib = MaterialLibrary::default();
    let mut s = build_scenario_with(&lib, MpScenario::WoodFire, 24, 24, 0.4 / 24.0, 9.81).unwrap();
    if let Some(fr) = std::env::var("MP_FR").ok().and_then(|v| v.parse().ok()) {
        s.flame_resolution = fr;
    }
    let o2 = idx(&s.reg, O2);
    let gg = s.group_of[o2];
    let mut best = (0.0f64, 0.0, String::new());
    let mut holes = 0usize;
    let mut tf: Vec<f64> = vec![];
    while s.time < 0.4 && !s.diverged {
        s.step();
        if s.time < 0.15 {
            continue;
        }
        holes += (0..s.nx * s.ny).filter(|&q| !s.is_solid(q) && s.cell_volume(q, s.p[q]) < 0.8).count();
        tf.push((0..s.nx * s.ny).filter(|&q| !s.is_solid(q) && s.group_mass(gg, q) > 0.0 && s.cell_volume(q, s.p[q]) > 0.95).map(|q| s.tg[gg][q]).fold(0.0f64, f64::max));
        for q in 0..s.nx * s.ny {
            if s.is_solid(q) || s.group_mass(gg, q) <= 0.0 || s.tg[gg][q] <= best.0 {
                continue;
            }
            let ag = s.group_alpha(gg, q);
            let comp: Vec<String> = (0..s.reg.phases.len()).filter(|&k| s.mass[k][q] > 0.0).map(|k| format!("{} {:.2e} (α {:.2e})", s.reg.phases[k].name, s.mass[k][q], s.alpha[k][q])).collect();
            let tgs: Vec<String> = (0..s.groups.len()).map(|g| format!("{:.0}", s.tg[g][q])).collect();
            best = (s.tg[gg][q], s.time, format!("maille {q} ({}, {}) : p {:.0} Pa, V − 1 = {:+.3e}, pression bloquée {}, α gaz {ag:.3e}, T groupes [{}], masse gaz {:.3e} ; {}",
                q % s.nx, q / s.nx, s.p[q], s.cell_volume(q, s.p[q]) - 1.0, s.pressure_clamped[q], tgs.join(", "), s.group_mass(gg, q), comp.join(", ")));
        }
    }
    tf.sort_by(|a, b| a.total_cmp(b));
    let med = tf.get(tf.len() / 2).copied().unwrap_or(0.0);
    eprintln!("épaississement {} : {} pas ; mailles creusées (V < 0.8) cumulées {holes} ; T de flamme (mailles pleines) médiane {med:.0} K, max {:.0} K",
        s.flame_resolution, s.step_count, tf.last().copied().unwrap_or(0.0));
    eprintln!("T max du gaz {:.0} K à t = {:.3} s : {}", best.0, best.1, best.2);
}

/// diagnostic : histoire d'une maille du feu de bois 24×24 (MP_CELL, défaut 56) entre MP_T0 et MP_T1
#[test]
#[ignore]
fn wood_fire_cell_history() {
    let lib = MaterialLibrary::default();
    let mut s = build_scenario_with(&lib, MpScenario::WoodFire, 24, 24, 0.4 / 24.0, 9.81).unwrap();
    let env = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let q = env("MP_CELL", 56.0) as usize;
    let (t0, t1) = (env("MP_T0", 0.28), env("MP_T1", 0.31));
    let gg = s.group_of[idx(&s.reg, O2)];
    let (i, j) = (q % s.nx, q / s.nx);
    while s.time < t1 && !s.diverged {
        s.step();
        if s.time >= t0 {
            let (ul, ur, vb, vt) = (s.u[s.uidx(i, j)], s.u[s.uidx(i + 1, j)], s.v[s.vidx(i, j)], s.v[s.vidx(i, j + 1)]);
            let x = |n: &str| s.mass[idx(&s.reg, n)][q] / s.group_mass(gg, q).max(1e-300);
            eprintln!("t {:.5} dt {:.2e} : V−1 {:+.3e}, T gaz {:.0} K, p {:.0}, masse gaz {:.3e}, u {ul:+.2}/{ur:+.2}, v {vb:+.2}/{vt:+.2} m/s ; chaleur {:.2e} W/m³, dt chimie {:.2e} ; Y CH4 {:.2e} CO {:.2e} H2 {:.2e} O2 {:.2e}",
                s.time, s.last_dt, s.cell_volume(q, s.p[q]) - 1.0, s.tg[gg][q], s.p[q], s.group_mass(gg, q), s.heat_release[q], s.dt_chem, x(CH4), x(CO), x(H2), x(O2));
        }
    }
}

/// diagnostic : vase clos de poudre noire, pas à pas (MP_POWDER_N pas, défaut 40)
#[test]
#[ignore]
fn gunpowder_vessel_diag() {
    use super::mp_materials::{FUMEE, POUDRE};
    let mut s = gunpowder_dust(700.0, 10.0 / 1800.0, true);
    s.radiation = std::env::var("MP_RAD").is_ok();
    let (w, f, n2) = (idx(&s.reg, POUDRE), idx(&s.reg, FUMEE), idx(&s.reg, N2));
    let u0 = s.total_internal_energy() + s.kinetic_energy();
    let nmax: usize = std::env::var("MP_POWDER_N").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let c = s.idx(1, 1);
    for k in 0..nmax {
        let dt = s.step();
        let tg: Vec<String> = (0..s.ngroups()).map(|g| format!("{:.0}", s.tg[g][c])).collect();
        eprintln!("pas {k} dt {dt:.3e} (borne {:.3e}) : poudre {:.4e} fumée {:.4e} N2 {:.4e} ; U/U0 {:.9} ; T {:.1} tg {:?} ; p {:.4} bar ; vol−1 {:+.3e} ; refaits {}",
            s.burn_dt_bound(), s.total_mass(w), s.total_mass(f), s.total_mass(n2), (s.total_internal_energy() + s.kinetic_energy()) / u0, s.temp[c], tg,
            s.p[c] / 1e5, s.cell_volume(c, s.p[c]) - 1.0, s.rollbacks);
    }
}

/// diagnostic : tas de poudre noire (MP_POWDER_NX mailles de large, défaut 80, domaine 0.8 m × 0.6 m ; MP_POWDER_T s, défaut 0.3 ;
/// une ligne toutes les MP_POWDER_DT s, défaut 0.01)
#[test]
#[ignore]
fn gunpowder_heap_diag() {
    use super::mp_materials::{FUMEE, POUDRE};
    let env = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let nx = env("MP_POWDER_NX", 80.0) as usize;
    let ny = nx * 3 / 4;
    let mut s = build_scenario_with(&MaterialLibrary::default(), MpScenario::Gunpowder, nx, ny, 0.8 / nx as f64, 9.81).unwrap();
    let (w, f, n2) = (idx(&s.reg, POUDRE), idx(&s.reg, FUMEE), idx(&s.reg, N2));
    let (gw, gg) = (s.group_of[w], s.group_of[n2]);
    let m0 = s.total_mass(w);
    let (t_end, every) = (env("MP_POWDER_T", 0.3), env("MP_POWDER_DT", 0.01));
    let t0 = std::time::Instant::now();
    let mut next = 0.0;
    while s.time < t_end && !s.diverged {
        let dt = s.step();
        if s.time >= next {
            next += every;
            let n = s.nx * s.ny;
            let x_of = |c: usize| ((c % s.nx) as f64 + 0.5) * s.dx;
            let front = (0..n).filter(|&c| s.mass[w][c] > 1.0 && s.tg[gw][c] >= 580.0).map(x_of).fold(-1.0f64, f64::max);
            let tail = (0..n).filter(|&c| s.mass[w][c] > 1.0).map(x_of).fold(f64::INFINITY, f64::min);
            let tg_max = (0..n).filter(|&c| !s.is_solid(c) && s.group_mass(gg, c) > 0.0).map(|c| s.tg[gg][c]).fold(0.0f64, f64::max);
            let pmax = s.p.iter().fold(0.0f64, |a, &b| a.max(b));
            eprintln!(
                "t {:.3} s ({} pas, {:.0} s, dt {dt:.2e}) : front de flamme x = {front:.3} m, poudre restante dès x = {tail:.3} m ({:.2} % de la poudre) ; fumée {:.3e} kg/m ; T gaz max {tg_max:.0} K ; p max {:.3} bar ; |u| max {:.1} m/s ; refaits {}",
                s.time, s.step_count, t0.elapsed().as_secs_f64(), 100.0 * s.total_mass(w) / m0, s.total_mass(f), pmax / 1e5, s.max_speed(), s.rollbacks
            );
        }
    }
    eprintln!("divergé : {}", s.diverged);
}

/// diagnostic : premier pas refusé du tas de poudre (40 × 40, 1 cm) — fait MP_POWDER_OK pas protégés (défaut 6), puis un pas
/// SANS filet de sécurité, et liste les mailles hors bornes (T de groupe, masses négatives, p) et la vitesse max
#[test]
#[ignore]
fn gunpowder_first_bad_step_probe() {
    let lib = MaterialLibrary::default();
    let mut s = build_scenario_with(&lib, MpScenario::Gunpowder, 40, 40, 0.01, 9.81).unwrap();
    let nok: usize = std::env::var("MP_POWDER_OK").ok().and_then(|v| v.parse().ok()).unwrap_or(6);
    for _ in 0..nok {
        s.step();
    }
    let dt = s.choose_dt();
    eprintln!("après {nok} pas : t {:.3e}, pas choisi {dt:.3e}, refaits {}", s.time, s.rollbacks);
    let dt = s.step_raw_pub();
    let n = s.nx * s.ny;
    eprintln!("pas brut dt {dt:.3e} : |u| max {:.1} m/s (Courant {:.2}), sain {}", s.max_speed(), s.max_speed() * dt / s.dx, s.state_is_sane());
    let names: Vec<String> = s.reg.phases.iter().map(|p| p.name.chars().take(8).collect()).collect();
    let mut shown = 0;
    for c in 0..n {
        if s.is_solid(c) {
            continue;
        }
        let mut why = vec![];
        if !(s.temp[c] >= 50.0 && s.temp[c] <= 5000.0) {
            why.push(format!("T {:.1}", s.temp[c]));
        }
        if !(s.p[c] >= 0.0) {
            why.push(format!("p {:.3e}", s.p[c]));
        }
        for g in 0..s.groups.len() {
            if s.group_mass(g, c) > 0.0 && !(s.tg[g][c] >= 50.0 && s.tg[g][c] <= 5000.0) {
                why.push(format!("T groupe {g} = {:.1} (masse {:.3e})", s.tg[g][c], s.group_mass(g, c)));
            }
        }
        for k in 0..s.mass.len() {
            if !(s.mass[k][c] >= -1e-6) {
                why.push(format!("m {} = {:.3e}", names[k], s.mass[k][c]));
            }
        }
        if !why.is_empty() && shown < 12 {
            shown += 1;
            let m: Vec<String> = (0..s.mass.len()).filter(|&k| s.mass[k][c].abs() > 1e-9).map(|k| format!("{} {:.3e}", names[k], s.mass[k][c])).collect();
            eprintln!("  maille ({}, {}) : {} — p {:.4e}, masses {}", c % s.nx, c / s.nx, why.join(", "), s.p[c], m.join(" "));
        }
    }
}

// ===================================================================================================== gel / fusion (eau ↔ glace)

/// boîte fermée (4 × 4, 1 cm, sans gravité) : eau (fraction aw) à t_w, glace (fraction ai) à t_i, le reste en air à t_w
pub(crate) fn ice_water_box(aw: f64, t_w: f64, ai: f64, t_i: f64) -> MultiphaseSim {
    let lib = MaterialLibrary::default().with_ice();
    let b = lib.build().expect("préréglage glace valide");
    assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    let reg = b.registry;
    let (eau, glace, air) = (idx(&reg, "eau"), idx(&reg, super::mp_materials::GLACE), idx(&reg, "air"));
    let np = reg.phases.len();
    let mut s = MultiphaseSim::new(4, 4, 0.01, reg, b.ambient);
    s.gravity = [0.0, 0.0];
    s.surface_tension = false;
    let mut m = vec![0.0; np];
    m[eau] = aw;
    m[glace] = ai;
    m[air] = 1.0 - aw - ai;
    s.paint_mixture(move |_, _| Some(m.clone()));
    s.finalize_init(move |_, _| t_w);
    if ai > 0.0 {
        let gi = s.group_of[glace];
        for c in 0..16 {
            s.set_group_temperature(c, gi, t_i);
        }
    }
    s.refresh();
    s
}

/// Colonne de gel sans gravité (40 × 2 mailles de 0.25 mm) : paroi d'acier à t_wall à gauche, eau à 0 °C, poche d'air au bout (elle
/// absorbe la dilatation de l'eau qui gèle)
pub(crate) fn freezing_column(t_wall: f64) -> MultiphaseSim {
    use super::mp_cells::MpThermal;
    let lib = MaterialLibrary::default().with_ice();
    let b = lib.build().expect("préréglage glace valide");
    assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    let reg = b.registry;
    let (eau, air) = (idx(&reg, "eau"), idx(&reg, "air"));
    let steel = reg.solids.iter().position(|so| so.name == "acier").expect("acier");
    let np = reg.phases.len();
    let (nx, ny, dx) = (40, 2, 2.5e-4);
    let mut s = MultiphaseSim::new(nx, ny, dx, reg, b.ambient);
    s.gravity = [0.0, 0.0];
    // (sans tension de surface : à 0.25 mm de maille, l'interface eau–air imposerait un pas capillaire de 2e-4 s)
    s.surface_tension = false;
    s.dt_max = 0.02;
    let lx = nx as f64 * dx;
    let mut w = vec![0.0; np];
    w[eau] = 1.0;
    let mut a = vec![0.0; np];
    a[air] = 1.0;
    s.paint_mixture(move |x, _| Some(if x < 0.85 * lx { w.clone() } else { a.clone() }));
    s.paint_solid(move |x, _| if x < dx { Some(steel) } else { None });
    s.finalize_init(|_, _| 273.15);
    for c in 0..nx * ny {
        if s.is_solid(c) {
            s.set_thermal_cell(c, MpThermal::Isothermal(t_wall));
        }
    }
    s.refresh();
    s
}

#[test]
fn undercooled_water_freezes_and_warm_ice_melts_back_to_the_melting_point() {
    // méthode enthalpique : l'eau à −10 °C gèle juste assez pour que sa chaleur latente la ramène à 0 °C (c ΔT / L_f = 12.55 %), la
    // glace à +10 °C fond juste assez pour revenir à 0 °C ; à −93 °C, la chaleur latente ne suffit pas (c ΔT > L_f dès −80 °C) : tout
    // gèle, et la glace reste sous 0 °C ; masse et énergie exactement conservées
    use super::mp_materials::GLACE;
    let lf = 333.55e3;
    let probe = |aw: f64, t_w: f64, ai: f64, t_i: f64| {
        let mut s = ice_water_box(aw, t_w, ai, t_i);
        let (eau, glace) = (idx(&s.reg, "eau"), idx(&s.reg, GLACE));
        let c = s.idx(1, 1);
        let (mw0, mi0, e0) = (s.mass[eau][c], s.mass[glace][c], s.total_internal_energy());
        s.fusion_step(1e-3);
        let (gw, gi) = (s.group_of[eau], s.group_of[glace]);
        let (tw, ti) = (s.tg[gw][c], s.tg[gi][c]);
        assert!(((s.total_internal_energy() - e0) / e0.abs().max(1.0)).abs() < 1e-12, "énergie");
        assert!((s.mass[eau][c] + s.mass[glace][c] - mw0 - mi0).abs() < 1e-9 * (mw0 + mi0), "masse");
        (mw0, mi0, s.mass[eau][c], s.mass[glace][c], tw, ti, s.frozen[0])
    };
    let (mw0, _, _, mi, tw, ti, frozen) = probe(0.9, 263.15, 0.0, 263.15);
    let frac = mi / mw0;
    eprintln!("eau à −10 °C : {:.4} % gelée (c ΔT / L_f = {:.4} %), eau à {tw:.6} K, glace à {ti:.6} K", 100.0 * frac, 100.0 * 4186.0 * 10.0 / lf);
    assert!((frac / (4186.0 * 10.0 / lf) - 1.0).abs() < 1e-9 && (tw - 273.15).abs() < 1e-9 && (ti - 273.15).abs() < 1e-9);
    assert!(frozen > 0.0);
    let (_, mi0, _, mi, tw, ti, frozen) = probe(0.5, 273.15, 0.4, 283.15);
    let ice = Phase::ice();
    let expect = (ice.e_int(283.15) - ice.e_int(273.15)) / lf;
    let melted = (mi0 - mi) / mi0;
    eprintln!("glace à +10 °C : {:.4} % fondue (attendu {:.4} %), glace à {ti:.6} K, eau à {tw:.6} K", 100.0 * melted, 100.0 * expect);
    assert!((melted / expect - 1.0).abs() < 1e-9 && (ti - 273.15).abs() < 1e-9 && frozen < 0.0);
    let (mw0, _, mw, mi, _, ti, _) = probe(0.9, 180.0, 0.0, 180.0);
    eprintln!("eau à −93 °C : tout gèle ({:.3e} kg/m³ d'eau restante), glace à {ti:.2} K", mw);
    assert!(mw == 0.0 && (mi - mw0).abs() < 1e-9 * mw0 && ti < 273.15 && ti > 180.0, "glace à {ti} K, eau restante {mw}");
    // sans changement de phase (case décochée) : rien ne gèle
    let mut s = ice_water_box(0.9, 263.15, 0.0, 263.15);
    s.phase_change = false;
    s.fusion_step(1e-3);
    assert!(s.total_mass(idx(&s.reg, GLACE)) == 0.0);
}

/// fonction d'erreur (Abramowitz & Stegun 7.1.26, erreur < 1.5e-7)
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    if x >= 0.0 { y } else { -y }
}

#[test]
fn ice_grows_on_a_cold_wall_like_the_stefan_solution() {
    // eau à 0 °C contre une paroi à −10 °C (sans gravité) : la glace s'épaissit en h = 2 λ √(α_s t) — problème de Stefan à une phase
    // (liquide à la température de fusion), λ e^{λ²} erf λ = St / √π, St = c_s ΔT / L_f = 0.063 → λ = 0.177 : 3 mm en 60 s
    use super::mp_materials::GLACE;
    let mut s = freezing_column(263.15);
    let gl = idx(&s.reg, GLACE);
    let ph = s.reg.phases[gl].clone();
    let (k, rho, cs) = (ph.conductivity, ph.rho(101325.0, 268.15), ph.cp_at(268.15));
    let st = cs * 10.0 / 333.55e3;
    let f = |l: f64| l * (l * l).exp() * erf(l) - st / std::f64::consts::PI.sqrt();
    let (mut a, mut b) = (1e-6, 2.0);
    for _ in 0..80 {
        let m = 0.5 * (a + b);
        if f(m) < 0.0 { a = m } else { b = m }
    }
    let lam = 0.5 * (a + b);
    let alpha = k / (rho * cs);
    let width = s.ny as f64 * s.dx;
    let t0 = std::time::Instant::now();
    let mut out = vec![];
    for t_mark in [30.0, 60.0, 120.0] {
        while s.time < t_mark && !s.diverged {
            s.step();
        }
        assert!(!s.diverged && s.state_is_sane());
        // épaisseur équivalente de glace compacte
        let h = s.total_mass(gl) / (rho * width);
        out.push((s.time, h, 2.0 * lam * (alpha * s.time).sqrt()));
    }
    eprintln!("Stefan : St = {st:.4}, λ = {lam:.4} ; {} pas en {:.1} s ; {}", s.step_count, t0.elapsed().as_secs_f64(),
        out.iter().map(|(t, h, he)| format!("t {t:.0} s : glace {:.3} mm (Stefan {:.3} mm)", 1e3 * h, 1e3 * he)).collect::<Vec<_>>().join(" ; "));
    // la croissance entre 30 et 120 s suit Stefan ; l'épaisseur garde le retard d'environ une maille pris au départ (méthode enthalpique,
    // erreur en dx sur la position du front)
    let ((_, h1, e1), (_, h3, e3)) = (out[0], out[2]);
    assert!(((h3 - h1) / (e3 - e1) - 1.0).abs() < 0.03, "croissance {} m contre {} m (Stefan)", h3 - h1, e3 - e1);
    for (t, h, he) in out {
        assert!((he - h).abs() < 1.5 * s.dx, "à {t} s : {h} m de glace contre {he} m (Stefan)");
    }
}

#[test]
fn crushed_ice_rises_and_melts_in_warm_water() {
    // nuage de glace pilée (−10 °C) au fond d'une eau à 27 °C (40 × 30, 0.8 m × 0.6 m) : les cristaux remontent vers la surface et
    // fondent (la moitié en ~5 s) ; cuve fermée : masse et énergie conservées, la masse fondue du bilan est celle de la glace disparue
    use super::mp_materials::GLACE;
    let mut s = build_scenario_with(&MaterialLibrary::default(), MpScenario::IceMelting, 40, 30, 0.02, 9.81).unwrap();
    let gl = idx(&s.reg, GLACE);
    let n = s.nx * s.ny;
    let y_ice = |s: &MultiphaseSim| {
        let m: f64 = (0..n).map(|c| s.mass[gl][c]).sum::<f64>().max(1e-300);
        (0..n).map(|c| s.mass[gl][c] * ((c / s.nx) as f64 + 0.5) * s.dx).sum::<f64>() / m
    };
    let (mi0, m0, u0, y0) = (s.total_mass(gl), s.total_mass_all(), s.total_internal_energy() + s.kinetic_energy(), y_ice(&s));
    while s.time < 5.0 && !s.diverged {
        s.step();
    }
    assert!(!s.diverged && s.state_is_sane());
    let (mi, y1) = (s.total_mass(gl), y_ice(&s));
    eprintln!("glace pilée : {} pas ; {:.1} % fondue en 5 s (bilan {:.4} kg/m, disparue {:.4}) ; centre de la glace y = {y0:.3} → {y1:.3} m ; énergie {:+.2e}",
        s.step_count, 100.0 * (1.0 - mi / mi0), -s.frozen[0], mi0 - mi, (s.total_internal_energy() + s.kinetic_energy()) / u0 - 1.0);
    assert!(mi < 0.7 * mi0, "la glace ne fond pas : {mi} sur {mi0}");
    assert!(y1 > y0 + 0.15, "la glace ne remonte pas : {y0} → {y1}");
    assert!((-s.frozen[0] / (mi0 - mi) - 1.0).abs() < 1e-9, "bilan de fusion");
    assert!((s.total_mass_all() / m0 - 1.0).abs() < 1e-9, "masse");
    assert!(((s.total_internal_energy() + s.kinetic_energy()) / u0 - 1.0).abs() < 1e-7, "énergie");
}

/// diagnostic : glace pilée dans l'eau tiède (MP_ICE_NX mailles de large, défaut 40, 0.8 m × 0.6 m ; MP_ICE_T s, défaut 10 ; une ligne
/// toutes les MP_ICE_DT s, défaut 1)
#[test]
#[ignore]
fn ice_melting_diag() {
    use super::mp_materials::GLACE;
    let env = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let nx = env("MP_ICE_NX", 40.0) as usize;
    let mut s = build_scenario_with(&MaterialLibrary::default(), MpScenario::IceMelting, nx, nx * 3 / 4, 0.8 / nx as f64, 9.81).unwrap();
    let (eau, gl) = (idx(&s.reg, "eau"), idx(&s.reg, GLACE));
    let (gw, gi) = (s.group_of[eau], s.group_of[gl]);
    let (mi0, u0) = (s.total_mass(gl), s.total_internal_energy() + s.kinetic_energy());
    let (t_end, every) = (env("MP_ICE_T", 10.0), env("MP_ICE_DT", 1.0));
    let t0 = std::time::Instant::now();
    let mut next = 0.0;
    while s.time < t_end && !s.diverged {
        let dt = s.step();
        if s.time >= next {
            next += every;
            let n = s.nx * s.ny;
            let mi = s.total_mass(gl);
            let mic: f64 = (0..n).map(|c| s.mass[gl][c]).sum::<f64>().max(1e-300);
            let yc = (0..n).map(|c| s.mass[gl][c] * ((c / s.nx) as f64 + 0.5) * s.dx).sum::<f64>() / mic;
            let tw = (0..n).map(|c| s.mass[eau][c] * s.tg[gw][c]).sum::<f64>() / (0..n).map(|c| s.mass[eau][c]).sum::<f64>();
            let ti = (0..n).map(|c| s.mass[gl][c] * s.tg[gi][c]).sum::<f64>() / mic;
            eprintln!("t {:.2} s ({} pas, {:.0} s, dt {dt:.2e}) : glace {:.2} % (centre à y = {yc:.3} m, {ti:.2} K), fondue (bilan) {:.4} kg/m ; eau {tw:.3} K ; U/U0 − 1 = {:+.2e} ; |u| max {:.3} m/s ; refaits {}",
                s.time, s.step_count, t0.elapsed().as_secs_f64(), 100.0 * mi / mi0, -s.frozen[0], (s.total_internal_energy() + s.kinetic_energy()) / u0 - 1.0, s.max_speed(), s.rollbacks);
        }
    }
    eprintln!("divergé : {}", s.diverged);
}

// ============================================================================================ surfusion : germes, front de dendrites

/// Canal d'eau très pure surfondue (préréglage `with_supercooled_water` : germination spontanée sous −20 °C), sans gravité (nx × 2
/// mailles de dx) à t_w ; germe de glace (2 % du volume) dans la première colonne ; poche d'air au bout (elle absorbe la dilatation de
/// l'eau qui gèle)
pub(crate) fn supercooled_channel(nx: usize, dx: f64, t_w: f64) -> MultiphaseSim {
    use super::mp_materials::GLACE;
    let lib = MaterialLibrary::default().with_supercooled_water();
    let b = lib.build().expect("préréglage eau surfondue valide");
    assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    let reg = b.registry;
    let (eau, glace, air) = (idx(&reg, "eau"), idx(&reg, GLACE), idx(&reg, "air"));
    let np = reg.phases.len();
    let mut s = MultiphaseSim::new(nx, 2, dx, reg, b.ambient);
    s.gravity = [0.0, 0.0];
    s.surface_tension = false;
    s.dt_max = 5e-3;
    let lx = nx as f64 * dx;
    let (mut w, mut a, mut g) = (vec![0.0; np], vec![0.0; np], vec![0.0; np]);
    w[eau] = 1.0;
    a[air] = 1.0;
    g[eau] = 0.98;
    g[glace] = 0.02;
    s.paint_mixture(move |x, _| Some(if x < dx { g.clone() } else if x < 0.9 * lx { w.clone() } else { a.clone() }));
    s.finalize_init(move |_, _| t_w);
    s.refresh();
    s
}

#[test]
fn supercooled_water_needs_a_germ_to_freeze() {
    // eau très pure à −8 °C (germination spontanée seulement sous −20 °C) : sans germe, elle reste liquide ; avec des cristaux de glace
    // (2 % de la cellule), elle gèle d'un coup jusqu'à 0 °C (c ΔT / L_f = 10.04 % de l'eau) ; des traces de glace (0.5 %, sous le seuil
    // de germe de 1 %) ne suffisent pas ; à −21 °C, elle gèle seule (26.4 %) ; sans surfusion (défaut), elle gèle dès 0 °C
    use super::mp_materials::GLACE;
    let lf = 333.55e3;
    let probe = |dtn: f64, t: f64, ai: f64| -> (f64, f64) {
        let mut s = ice_water_box(0.9 - ai, t, ai, t);
        s.reg.fusions[0].supercooling = dtn;
        let (eau, glace) = (idx(&s.reg, "eau"), idx(&s.reg, GLACE));
        let c = s.idx(1, 1);
        let (mw0, mi0, e0) = (s.mass[eau][c], s.mass[glace][c], s.total_internal_energy());
        s.fusion_step(1e-3);
        assert!(((s.total_internal_energy() - e0) / e0.abs().max(1.0)).abs() < 1e-12, "énergie");
        ((s.mass[glace][c] - mi0) / mw0, s.tg[s.group_of[eau]][c])
    };
    let expect = |dt_c: f64| 4186.0 * dt_c / lf;
    let (f, tw) = probe(20.0, 265.15, 0.0);
    eprintln!("sans germe : {:.4} % gelés, eau à {tw:.3} K", 100.0 * f);
    assert!(f == 0.0 && (tw - 265.15).abs() < 1e-9);
    let (f, tw) = probe(20.0, 265.15, 0.02);
    eprintln!("germe de 2 % : {:.4} % gelés (c ΔT / L_f = {:.4} %), eau à {tw:.6} K", 100.0 * f, 100.0 * expect(8.0));
    assert!((f / expect(8.0) - 1.0).abs() < 1e-9 && (tw - 273.15).abs() < 1e-9);
    let (f, _) = probe(20.0, 265.15, 0.005);
    assert!(f == 0.0, "des traces de glace ont fait geler l'eau : {f}");
    let (f, tw) = probe(20.0, 252.15, 0.0);
    eprintln!("à −21 °C, germination spontanée : {:.4} % gelés (attendu {:.4} %), eau à {tw:.6} K", 100.0 * f, 100.0 * expect(21.0));
    assert!((f / expect(21.0) - 1.0).abs() < 1e-9 && (tw - 273.15).abs() < 1e-9);
    let (f, _) = probe(0.0, 265.15, 0.0);
    assert!((f / expect(8.0) - 1.0).abs() < 1e-9, "sans surfusion, l'eau gèle dès 0 °C");
    // la boîte entière, pas complets : toujours liquide après 50 pas
    let mut s = ice_water_box(0.9, 265.15, 0.0, 265.15);
    s.reg.fusions[0].supercooling = 20.0;
    for _ in 0..50 {
        s.step();
    }
    assert!(s.total_mass(idx(&s.reg, GLACE)) == 0.0 && !s.diverged);
}

#[test]
fn dendrite_front_crosses_supercooled_water_at_the_measured_speed() {
    // canal de 100 mailles de 2 mm d'eau à −10 °C, germe au bout : le front de dendrites avance à v(10 K) = 1 mm/s × 10^1.9 = 7.94 cm/s
    // (mesures : 7.7 cm/s) ; la maille i gèle à t = i dx / v, au pas près (l'excédent du front passe d'une maille à la suivante) — à 2 %
    // près : l'eau qui gèle se dilate et pousse 1.1 % de sa maille, à 0 °C, dans la suivante, qui se réchauffe de 0.09 K (surfusion
    // 9.91 K) ; rien ne gèle devant le front ; derrière, l'eau est à 0 °C avec c ΔT / L_f = 12.55 % de glace
    use super::mp_materials::GLACE;
    let (nx, dx) = (100, 2e-3);
    let mut s = supercooled_channel(nx, dx, 263.15);
    let (eau, gl) = (idx(&s.reg, "eau"), idx(&s.reg, GLACE));
    let v = s.reg.fusions[0].dendrite_speed(10.0);
    let mut t_frz = vec![f64::NAN; nx];
    let (mut ahead, mut max_dt) = (0usize, 0.0f64);
    while s.time < 2.2 && !s.diverged {
        let dt = s.step();
        max_dt = max_dt.max(dt);
        // maille gelée : plus de 5 % de glace (les traces portées par l'écoulement de dilatation ne comptent pas)
        for i in 0..nx {
            if t_frz[i].is_nan() && s.alpha[gl][s.idx(i, 0)] > 0.05 {
                t_frz[i] = s.time;
            }
        }
        // aucune glace à plus d'une maille devant le front
        let lead = (0..nx).filter(|&i| !t_frz[i].is_nan()).max().unwrap_or(0);
        ahead = ahead.max(lead.saturating_sub((v * s.time / dx).floor() as usize + 1));
    }
    assert!(!s.diverged && s.state_is_sane());
    let (i0, i1) = (10, 80);
    let speed = (i1 - i0) as f64 * dx / (t_frz[i1] - t_frz[i0]);
    eprintln!("front de dendrites : {:.2} cm/s mesurés pour {:.2} cm/s (maille {i0} à {:.3} s, {i1} à {:.3} s ; attendu {:.3} et {:.3}) ; pas max {:.1} ms ; {} pas",
        100.0 * speed, 100.0 * v, t_frz[i0], t_frz[i1], i0 as f64 * dx / v, i1 as f64 * dx / v, 1e3 * max_dt, s.step_count);
    assert!(speed < v && speed > 0.975 * v, "vitesse du front {speed} m/s pour {v}");
    for &i in &[i0, i1] {
        let te = i as f64 * dx / v;
        assert!(t_frz[i] >= te - 1e-9 && t_frz[i] <= 1.025 * te + max_dt, "maille {i} gelée à {} s (attendu {te} s)", t_frz[i]);
    }
    assert!(ahead == 0, "glace devant le front ({ahead} mailles)");
    // derrière le front : neige fondante à 0 °C
    let c = s.idx(40, 0);
    let frac = s.mass[gl][c] / (s.mass[gl][c] + s.mass[eau][c]);
    let tw = s.tg[s.group_of[eau]][c];
    eprintln!("maille 40 : {:.3} % de glace (c ΔT / L_f = {:.3} %), eau à {tw:.4} K", 100.0 * frac, 100.0 * 4186.0 * 10.0 / 333.55e3);
    assert!((frac / (4186.0 * 10.0 / 333.55e3) - 1.0).abs() < 0.02 && (tw - 273.15).abs() < 0.05);
}

#[test]
fn supercooled_water_freezes_outward_from_the_germ() {
    // scénario « eau surfondue » (40 × 30, mailles de 2 cm) : sous le germe (à la surface, au milieu), le front de dendrites descend à
    // v(8 K) ≈ 5.2 cm/s ; derrière lui, de la neige fondante à 0 °C ; le long de la surface il va plus vite (la neige fondante, plus
    // légère, flotte et s'étale en courant de gravité, ~6 cm/s, en semant le gel devant elle : cristaux libres, pas de réseau de
    // dendrites soudé) ; le fond de la cuve reste surfondu ; masse et énergie conservées
    use super::mp_materials::GLACE;
    use super::phases::FUSION_SEED_ALPHA;
    let mut s = build_scenario_with(&MaterialLibrary::default(), MpScenario::SupercooledWater, 40, 30, 0.02, 9.81).unwrap();
    let (eau, gl) = (idx(&s.reg, "eau"), idx(&s.reg, GLACE));
    assert!(s.reg.fusions[0].supercooling >= 20.0);
    let v = s.reg.fusions[0].dendrite_speed(8.0);
    let (m0, u0) = (s.total_mass_all(), s.total_internal_energy() + s.kinetic_energy());
    let (mi0, mw0) = (s.total_mass(gl), s.total_mass(eau));
    let (xs, ys) = (0.5 * 40.0 * 0.02, 0.7 * 30.0 * 0.02);
    // mailles gelées d'affilée sous la surface, au milieu
    let down = |s: &MultiphaseSim| (0..21).rev().take_while(|&j| s.alpha[gl][s.idx(20, j)] >= FUSION_SEED_ALPHA).count();
    s.step();
    let d0 = down(&s);
    while s.time < 4.0 && !s.diverged {
        s.step();
    }
    assert!(!s.diverged && s.state_is_sane());
    let n = s.nx * s.ny;
    let r_front = (0..n)
        .filter(|&c| !s.is_solid(c) && s.alpha[gl][c] >= FUSION_SEED_ALPHA && s.mass[eau][c] > 1e-6)
        .map(|c| {
            let (x, y) = s.center(c % s.nx, c / s.nx);
            (x - xs).hypot(y - ys)
        })
        .fold(0.0f64, f64::max);
    let frozen = s.total_mass(gl) - mi0;
    let tb = s.tg[s.group_of[eau]][s.idx(20, 1)];
    let d1 = down(&s);
    let v_down = (d1 - d0) as f64 * s.dx / s.time;
    eprintln!("eau surfondue : {} pas ; front vers le bas {:.2} cm/s ({d0} → {d1} mailles ; v(8 K) = {:.2} cm/s), le plus loin à {:.3} m du germe (v t = {:.3} m) ; {:.2} % de l'eau gelée ; fond à {tb:.2} K ; énergie {:+.2e}",
        s.step_count, 100.0 * v_down, 100.0 * v, r_front, v * s.time, 100.0 * frozen / mw0, (s.total_internal_energy() + s.kinetic_energy()) / u0 - 1.0);
    assert!(v_down > 0.75 * v && v_down < 1.35 * v, "front vers le bas à {v_down} m/s pour {v}");
    assert!(r_front < 3.0 * v * s.time + 0.05, "front à {r_front} m pour v t = {}", v * s.time);
    assert!(frozen > 0.01 * mw0 && frozen < 0.1004 * mw0, "glace formée : {frozen} kg/m");
    assert!(tb < 266.0, "le fond a gelé : {tb} K");
    assert!((s.frozen[0] / frozen - 1.0).abs() < 1e-6, "bilan du gel");
    assert!((s.total_mass_all() / m0 - 1.0).abs() < 1e-9, "masse");
    assert!(((s.total_internal_energy() + s.kinetic_energy()) / u0 - 1.0).abs() < 1e-7, "énergie");
}

/// diagnostic : eau surfondue (MP_SC_NX mailles de large, défaut 40, 0.8 m × 0.6 m ; MP_SC_T s, défaut 6)
#[test]
#[ignore]
fn supercooled_water_diag() {
    use super::mp_materials::GLACE;
    use super::phases::FUSION_SEED_ALPHA;
    let env = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let nx = env("MP_SC_NX", 40.0) as usize;
    let dx = 0.8 / nx as f64;
    let mut s = build_scenario_with(&MaterialLibrary::default(), MpScenario::SupercooledWater, nx, nx * 3 / 4, dx, env("MP_SC_G", 9.81)).unwrap();
    let (eau, gl) = (idx(&s.reg, "eau"), idx(&s.reg, GLACE));
    if let Ok(r) = std::env::var("MP_SC_RHOI") {
        let r: f64 = r.parse().unwrap();
        if let super::phases::Eos::Tait { thermal: super::phases::Thermal::Beta { rho0, .. }, .. } = &mut s.reg.phases[gl].eos {
            *rho0 = r;
        }
        s.refresh();
    }
    if std::env::var("MP_SC_NOST").is_ok() {
        s.surface_tension = false;
    }
    let v = s.reg.fusions[0].dendrite_speed(8.0);
    let (xs, ys) = (0.4, 0.7 * (nx * 3 / 4) as f64 * dx);
    let mw0 = s.total_mass(eau);
    let mut next = 0.0;
    let t0 = std::time::Instant::now();
    let every = env("MP_SC_EVERY", 0.5);
    while s.time < env("MP_SC_T", 6.0) && !s.diverged {
        let dt = s.step();
        if std::env::var("MP_SC_WHERE").is_ok() && s.time >= next {
            // vitesse maximale : où, et dans quoi
            let (nx, ny) = (s.nx, s.ny);
            let mut best = (0.0f64, 0usize, 0usize, 'u');
            for j in 0..ny { for i in 0..=nx { let f = j * (nx + 1) + i; if s.u[f].abs() > best.0 { best = (s.u[f].abs(), i.min(nx - 1), j, 'u'); } } }
            for j in 0..=ny { for i in 0..nx { let f = j * nx + i; if s.v[f].abs() > best.0 { best = (s.v[f].abs(), i, j.min(ny - 1), 'v'); } } }
            let c = s.idx(best.1, best.2);
            let air = idx(&s.reg, "air");
            // dans l'eau (α > 0.5) : vitesse max, où, et la composition
            let mut bw = (0.0f64, 0usize, 0usize);
            for j in 0..ny { for i in 0..nx { let cc = s.idx(i, j); if s.alpha[eau][cc] > 0.5 {
                let uc = 0.5 * (s.u[j * (nx + 1) + i] + s.u[j * (nx + 1) + i + 1]); let vc = 0.5 * (s.v[j * nx + i] + s.v[(j + 1) * nx + i]);
                let sp = uc.hypot(vc); if sp > bw.0 { bw = (sp, i, j); } } } }
            let cw = s.idx(bw.1, bw.2);
            eprintln!("  eau : |u| max {:.3} m/s en ({}, {}) : eau {:.3} air {:.3} glace {:.4} ; T eau {:.2}", bw.0, bw.1, bw.2, s.alpha[eau][cw], s.alpha[air][cw], s.alpha[gl][cw], s.tg[s.group_of[eau]][cw]);
            let ws = s.max_settling_speed();
            eprintln!("  |{}| max {:.3} m/s en ({}, {}) : eau {:.3} air {:.3} glace {:.4} ; T eau {:.2} glace {:.2} ; p {:.0} Pa ; dérive max {:.3} m/s", best.3, best.0, best.1, best.2,
                s.alpha[eau][c], s.alpha[air][c], s.alpha[gl][c], s.tg[s.group_of[eau]][c], s.tg[s.group_of[gl]][c], s.p[c], ws);
        }
        if s.time >= next {
            next += every;
            let n = s.nx * s.ny;
            let (mut r_phi, mut r_ice, mut n_phi, mut n_ice) = (0.0f64, 0.0f64, 0, 0);
            for c in 0..n {
                let (x, y) = s.center(c % s.nx, c / s.nx);
                let r = (x - xs).hypot(y - ys);
                if s.fusion_front[0][c] >= 1.0 { r_phi = r_phi.max(r); n_phi += 1; }
                if s.alpha[gl][c] >= FUSION_SEED_ALPHA && s.mass[eau][c] > 1e-6 { r_ice = r_ice.max(r); n_ice += 1; }
            }
            // étendue le long de la surface (rangée sous la surface) et vers le bas sous le germ
            let js = ((ys / dx) as usize).min(s.ny) - 1;
            let isx = s.nx / 2;
            let left = (0..isx).rev().take_while(|&i| s.alpha[gl][s.idx(i, js)] >= FUSION_SEED_ALPHA).count();
            let down = (0..js).rev().take_while(|&j| s.alpha[gl][s.idx(isx, j)] >= FUSION_SEED_ALPHA).count();
            let mut tw = vec![];
            for j in [js, js / 2, 1] { tw.push(format!("{:.2}", s.tg[s.group_of[eau]][s.idx(isx / 2, j)])); }
            eprintln!("t {:.2} s ({} pas, {:.0} s, dt {dt:.2e}) : v t = {:.3} m ; front φ≥1 {:.3} m ({n_phi} mailles), glace ≥ 1 % {:.3} m ({n_ice}) ; surface ← {left} mailles, bas ↓ {down} ; gelé {:.2} % ; |u| max {:.3} m/s ; T eau à x/4 (surface, milieu, fond) {}",
                s.time, s.step_count, t0.elapsed().as_secs_f64(), v * s.time, r_phi, r_ice, 100.0 * s.frozen[0] / mw0, s.max_speed(), tw.join(" "));
        }
    }
}

/// boîte fermée (4 × 4, 1 cm, sans gravité) avec trois paires de fusion : eau (fraction 0.3) à t_w, cire fondue (0.3) à t_c, lave
/// (0.2) à t_l, air
pub(crate) fn water_wax_lava_box(t_w: f64, t_c: f64, t_l: f64) -> MultiphaseSim {
    use super::mp_materials::{CIRE_FONDUE, LAVE};
    let lib = MaterialLibrary::default().with_ice().with_wax().with_lava();
    let b = lib.build().expect("préréglages glace, cire et lave valides");
    assert!(b.warnings.is_empty(), "{:?}", b.warnings);
    let reg = b.registry;
    assert_eq!(reg.fusions.len(), 3);
    let (eau, cire, lave, air) = (idx(&reg, "eau"), idx(&reg, CIRE_FONDUE), idx(&reg, LAVE), idx(&reg, "air"));
    let np = reg.phases.len();
    let mut s = MultiphaseSim::new(4, 4, 0.01, reg, b.ambient);
    s.gravity = [0.0, 0.0];
    s.surface_tension = false;
    let mut m = vec![0.0; np];
    m[eau] = 0.3;
    m[cire] = 0.3;
    m[lave] = 0.2;
    m[air] = 0.2;
    s.paint_mixture(move |_, _| Some(m.clone()));
    s.finalize_init(move |_, _| t_w);
    let (gc, gl) = (s.group_of[cire], s.group_of[lave]);
    for c in 0..16 {
        s.set_group_temperature(c, gc, t_c);
        s.set_group_temperature(c, gl, t_l);
    }
    s.refresh();
    s
}

#[test]
fn water_wax_and_lava_freeze_at_their_own_melting_points() {
    // trois paires de fusion dans les mêmes cellules : l'eau à −5 °C gèle de c ΔT / L_f = 4186 × 5 / 333550 = 6.27 %, la cire fondue
    // à 50 °C (8 K sous son point de fusion, 58 °C) fige de 2300 × 8 / 200000 = 9.2 %, la lave à 1050 °C (50 K sous 1100 °C) de
    // 1400 × 50 / 400000 = 17.5 % ; chacune revient à SA température de fusion ; masse et énergie conservées
    use super::mp_materials::{BASALTE, CIRE, CIRE_FONDUE, GLACE, LAVE};
    let mut s = water_wax_lava_box(268.15, 323.15, 1323.15);
    let pairs = [("eau", GLACE, 4186.0 * 5.0 / 333.55e3, 273.15), (CIRE_FONDUE, CIRE, 2300.0 * 8.0 / 200.0e3, 331.15), (LAVE, BASALTE, 1400.0 * 50.0 / 400.0e3, 1373.15)];
    let c = s.idx(1, 1);
    let m0: Vec<f64> = pairs.iter().map(|&(l, _, _, _)| s.mass[idx(&s.reg, l)][c]).collect();
    let e0 = s.total_internal_energy();
    s.fusion_step(1e-3);
    for (q, &(l, sol, expect, tm)) in pairs.iter().enumerate() {
        let (kl, ks) = (idx(&s.reg, l), idx(&s.reg, sol));
        let f = s.mass[ks][c] / m0[q];
        let t = s.tg[s.group_of[kl]][c];
        eprintln!("{l} : {:.4} % figés (attendu {:.4} %), {t:.6} K (fusion à {tm} K)", 100.0 * f, 100.0 * expect);
        assert!((f / expect - 1.0).abs() < 1e-9 && (t - tm).abs() < 1e-9, "{l} : {f} figés à {t} K");
        assert!(s.frozen[q] > 0.0);
    }
    assert!(((s.total_internal_energy() - e0) / e0.abs()).abs() < 1e-12, "énergie");
}

#[test]
fn viscous_lava_resting_in_air_stays_calm() {
    // bloc de lave (50 Pa·s, 18 cm de haut) posé au fond d'une cuve d'air (40 × 30, mailles de 2 cm) : il s'affaisse (échelle
    // visqueuse ρ g h² / 3μ ≈ 6 m/s : ~4 m/s mesurés). Sans plafonner la viscosité explicite par la masse volumique des faces et des
    // nœuds, la viscosité de la lave, moyennée aux nœuds, s'appliquait à des faces d'air 600 fois plus légères : 50 à 120 m/s en
    // quelques pas, pas de 1e-4 s
    use super::mp_materials::LAVE;
    let lib = MaterialLibrary::default().with_lava();
    let b = lib.build().expect("préréglage lave valide");
    let reg = b.registry;
    let lave = idx(&reg, LAVE);
    let mut s = MultiphaseSim::new(40, 30, 0.02, reg, b.ambient);
    s.gravity = [0.0, -9.81];
    let (lx, ly) = (0.8, 0.6);
    s.paint_phase(move |x, y| if x > 0.3 * lx && x < 0.7 * lx && y < 0.3 * ly { Some(lave) } else { None });
    s.finalize_init(|_, _| 293.15);
    let gl = s.group_of[lave];
    for c in 0..s.nx * s.ny {
        if s.mass[lave][c] > 0.0 {
            s.set_group_temperature(c, gl, 1473.15);
        }
    }
    s.refresh();
    let mut umax = 0.0f64;
    while s.time < 0.3 && !s.diverged {
        s.step();
        umax = umax.max(s.max_speed());
    }
    eprintln!("lave dans l'air : {} pas, {} refaits, |u| max {umax:.3} m/s", s.step_count, s.rollbacks);
    assert!(!s.diverged && s.state_is_sane());
    assert!(umax < 10.0 && s.rollbacks == 0 && s.step_count < 1000, "|u| max {umax} m/s, {} pas refaits, {} pas", s.rollbacks, s.step_count);
}


/// diagnostic : braséro, cellule de la flamme la plus chaude au fil du temps (MP_COAL_T s, défaut 2 ; MP_DT_MAX : pas maximal)
#[test]
#[ignore]
fn coal_brasero_hot_spot_diag() {
    let lib = MaterialLibrary::default();
    let mut s = build_scenario_with(&lib, MpScenario::CoalFire, 40, 40, 0.01, 9.81).unwrap();
    let k = |n: &str| idx(&s.reg, n);
    let (c, o2, hou) = (k(super::mp_materials::COAL), k(O2), k(super::mp_materials::HOUILLE));
    let gg = s.group_of[o2];
    let t_end = std::env::var("MP_COAL_T").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0f64);
    if let Some(d) = std::env::var("MP_DT_MAX").ok().and_then(|v| v.parse::<f64>().ok()) {
        s.dt_max = d;
    }
    let mut next = 0.3;
    let mut worst = (0.0f64, 0usize, 0.0f64);
    while s.time < t_end && !s.diverged {
        s.step();
        if s.time > 0.3 {
            let n = s.nx * s.ny;
            for q in 0..n {
                if !s.is_solid(q) && s.mass[c][q] + s.mass[hou][q] < 1.0 && s.group_mass(gg, q) > 0.0 && s.tg[gg][q] > worst.0 {
                    worst = (s.tg[gg][q], q, s.time);
                }
            }
        }
        if s.time >= next {
            next += 0.1;
            let q = worst.1;
            let gas: f64 = (0..s.reg.phases.len()).filter(|&kk| s.reg.phases[kk].is_gas()).map(|kk| s.alpha[kk][q]).sum();
            eprintln!("t {:.2} dt {:.1e} : T gaz max {:.0} K en ({}, {}) à t {:.3} ; maintenant T_g {:.0} T {:.0} ; α charbon {:.2e} houille {:.2e} gaz {:.3} ; masse du groupe {:.2e} kg/m³ ; ρ_mix {:.3}",
                s.time, s.last_dt, worst.0, q % s.nx, q / s.nx, worst.2, s.tg[gg][q], s.temp[q], s.alpha[c][q], s.alpha[hou][q], gas, s.group_mass(gg, q), s.rho_mix[q]);
            worst = (0.0, 0, 0.0);
        }
    }
}


/// diagnostic : braséro, accrochage de la flamme des matières volatiles au fil du temps (MP_COAL_T s, défaut 2) : chaleur dégagée hors
/// du lit, base et centre de la flamme, gaz qui sort du lit (T, CH4, O2, vitesse), houille du haut du lit
#[test]
#[ignore]
fn coal_brasero_flame_anchor_diag() {
    let lib = MaterialLibrary::default();
    let nn = std::env::var("MP_COAL_N").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(40);
    let mut s = build_scenario_with(&lib, MpScenario::CoalFire, nn, nn, 0.4 / nn as f64, 9.81).unwrap();
    if let Some(r) = std::env::var("MP_FLAME_RES").ok().and_then(|v| v.parse::<f64>().ok()) {
        s.flame_resolution = r;
    }
    let k = |n: &str| idx(&s.reg, n);
    let (c, o2, hou, ch4) = (k(super::mp_materials::COAL), k(O2), k(super::mp_materials::HOUILLE), k(CH4));
    let gg = s.group_of[o2];
    let gh = s.group_of[hou];
    let t_end = std::env::var("MP_COAL_T").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0f64);
    let (nx, ny) = (s.nx, s.ny);
    // haut du lit (dernière rangée de grains au milieu) et rangées des parois
    let top = |s: &MultiphaseSim| (0..ny).rev().find(|&j| s.alpha[c][s.idx(nx / 2, j)] + s.alpha[hou][s.idx(nx / 2, j)] > 0.3).unwrap_or(0);
    let x_frac = |s: &MultiphaseSim, q: usize, kk: usize| {
        let n: f64 = (0..s.reg.phases.len()).filter(|&m| s.reg.phases[m].is_gas()).map(|m| s.mass[m][q].max(0.0) / s.reg.phases[m].molar_mass).sum();
        s.mass[kk][q].max(0.0) / s.reg.phases[kk].molar_mass / n.max(1e-300)
    };
    let mut next = 0.0;
    while s.time < t_end && !s.diverged {
        s.step();
        if s.time >= next {
            next += 0.05;
            let jt = top(&s);
            // chaleur dégagée hors du lit (rangées au-dessus du haut du lit) : total, maximum, base (rangée la plus basse à 10 % du max)
            let (mut qtot, mut qmax, mut qmax_at, mut yc) = (0.0f64, 0.0f64, 0usize, 0.0f64);
            for j in jt + 1..ny {
                for i in 0..nx {
                    let q = s.idx(i, j);
                    let h = s.heat_release[q].max(0.0);
                    qtot += h;
                    yc += h * j as f64;
                    if h > qmax {
                        qmax = h;
                        qmax_at = q;
                    }
                }
            }
            let base = (jt + 1..ny).find(|&j| (0..nx).any(|i| s.heat_release[s.idx(i, j)] > 0.1 * qmax)).unwrap_or(ny);
            let q_bed: f64 = (0..=jt).flat_map(|j| (0..nx).map(move |i| (i, j))).map(|(i, j)| s.heat_release[s.idx(i, j)].max(0.0)).sum();
            let a = s.idx(nx / 2, jt + 1);
            let b = s.idx(nx / 2, jt);
            // air autour du foyer : colonnes hors des parois (x < 0.3 lx ou > 0.7 lx), du haut des parois (j 20) à j 30 ; et au ras du haut
            // des parois (j 20–21) au-dessus du foyer
            let outside = |s: &MultiphaseSim, f: &dyn Fn(&MultiphaseSim, usize) -> f64| {
                let v: Vec<f64> = (20..31).flat_map(|j| (0..nx).filter(|&i| (i as f64 + 0.5) / (nx as f64) < 0.27 || (i as f64 + 0.5) / (nx as f64) > 0.73).map(move |i| (i, j)))
                    .map(|(i, j)| f(s, s.idx(i, j))).collect();
                v.iter().sum::<f64>() / v.len().max(1) as f64
            };
            let o2_out = outside(&s, &|s: &MultiphaseSim, q: usize| x_frac(s, q, o2));
            let t_out = outside(&s, &|s: &MultiphaseSim, q: usize| s.tg[gg][q]);
            let rim: Vec<String> = [11usize, 13, 20, 26, 28].iter().map(|&i| format!("{:.2}", x_frac(&s, s.idx(i, 21), o2))).collect();
            let top_in: f64 = (0..nx).map(|i| s.v[ny * nx + i].min(0.0)).sum::<f64>() * s.dx;
            let top_out: f64 = (0..nx).map(|i| s.v[ny * nx + i].max(0.0)).sum::<f64>() * s.dx;
            eprintln!("      air autour : x O2 {o2_out:.3}, T {t_out:.0} K ; x O2 à j 21 (i 11 13 20 26 28) {} ; bord du haut : entrée {top_in:.4}, sortie {top_out:.4} m²/s", rim.join(" "));
            eprintln!(
                "t {:.2} dt {:.1e} : haut du lit j {jt} ; flamme {:.2e} W/m (lit {:.2e}), max {:.2e} W/m³ en ({}, {}), base j {base}, centre j {:.1} | sortie du lit : T {:.0} K, x CH4 {:.3}, x O2 {:.3}, v {:+.3} m/s | houille du haut {:.0} K (α {:.2})",
                s.time, s.last_dt, qtot * s.dx * s.dx, q_bed * s.dx * s.dx, qmax, qmax_at % nx, qmax_at / nx, yc / qtot.max(1e-30),
                s.tg[gg][a], x_frac(&s, a, ch4), x_frac(&s, a, o2), s.v[(jt + 2) * nx + nx / 2], s.tg[gh][b], s.alpha[hou][b]
            );
        }
    }
}

/// diagnostic : chaleur dégagée au fil du temps d'un scénario de feu (MP_SC = indice du scénario, MP_NX, MP_NY, MP_DX, MP_T)
#[test]
#[ignore]
fn fire_persistence_diag() {
    let ev = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let sc = MpScenario::from_index(ev("MP_SC", 0.0) as usize);
    let lib = MaterialLibrary::default();
    let mut s = build_scenario_with(&lib, sc, ev("MP_NX", 48.0) as usize, ev("MP_NY", 36.0) as usize, ev("MP_DX", 4e-3), 9.81).unwrap();
    if let Some(r) = std::env::var("MP_FLAME_RES").ok().and_then(|v| v.parse::<f64>().ok()) {
        s.flame_resolution = r;
    }
    let t_end = ev("MP_T", 3.0);
    let mut next = 0.0;
    eprintln!("scénario {} : flame_resolution {}", sc.name(), s.flame_resolution);
    while s.time < t_end && !s.diverged {
        s.step();
        if s.time >= next {
            next += 0.1;
            let q: f64 = s.heat_release.iter().map(|h| h.max(0.0)).sum::<f64>() * s.dx * s.dx;
            let (mut qm, mut at) = (0.0f64, 0usize);
            for (c, &h) in s.heat_release.iter().enumerate() {
                if h > qm {
                    qm = h;
                    at = c;
                }
            }
            let tmax = (0..s.nx * s.ny).filter(|&c| !s.is_solid(c)).map(|c| s.temp[c]).fold(0.0f64, f64::max);
            eprintln!("t {:.2} dt {:.1e} : chaleur {q:.3e} W/m, max en ({}, {}), T max {tmax:.0} K", s.time, s.last_dt, at % s.nx, at / s.nx);
        }
    }
}
