//! Validation de la chimie, des flammes et des explosions contre des valeurs de référence (NIST-JANAF, mesures de vitesse de flamme,
//! pression de combustion isochore) et tests de conservation exacte.

use super::mp_materials::{MaterialLibrary, AIR_SEC, CH4, CO, CO2, H2, H2O_G, MIX_CH4_AIR, MIX_H2_AIR, N2, O2, RX_CH4, RX_CO, RX_H2};
use super::mp_view::{build_scenario_with, MpScenario};
use super::multiphase::{Bc, MultiphaseSim};
use super::phases::{Registry, R_GAS};

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
fn a_coal_fire_keeps_burning_and_conserves_carbon() {
    let lib = MaterialLibrary::default();
    let mut s = build_scenario_with(&lib, MpScenario::CoalFire, 40, 40, 0.01, 9.81).unwrap();
    let k = |n: &str| idx(&s.reg, n);
    let (c, co2, o2, co) = (k(super::mp_materials::COAL), k(CO2), k(O2), k(CO));
    let gas_c = |s: &MultiphaseSim, q: usize| (s.total_mass(q) - s.boundary_flux[q]) / s.reg.phases[q].molar_mass;
    let carbon = |s: &MultiphaseSim| s.total_mass(c) / 0.012011 + gas_c(s, co2) + gas_c(s, co);
    let (m_c0, n_c0) = (s.total_mass(c), carbon(&s));
    let t_start = std::time::Instant::now();
    let mut t_hot = vec![];
    let t_end = std::env::var("MP_COAL_T").ok().and_then(|v| v.parse().ok()).unwrap_or(3.0f64);
    while s.time < t_end && !s.diverged {
        s.step();
        if s.step_count % 250 == 0 {
            let tmax = (0..s.nx * s.ny).filter(|&q| s.mass[c][q] > 1.0).map(|q| s.tg[s.group_of[c]][q]).fold(0.0f64, f64::max);
            t_hot.push((s.time, tmax));
        }
    }
    assert!(!s.diverged && s.state_is_sane());
    let burnt = m_c0 - s.total_mass(c);
    let tmax = t_hot.last().map_or(0.0, |x| x.1);
    eprintln!("feu de charbon : {} pas en {:.1} s ; charbon brûlé {:.3e} kg/m ({:.2} %), CO2 {:.3e} kg/m, CO {:.3e}, O2 {:.3e} ; T max des braises {:?}",
        s.step_count, t_start.elapsed().as_secs_f64(), burnt, 100.0 * burnt / m_c0, s.total_mass(co2), s.total_mass(co), s.total_mass(o2),
        t_hot.iter().map(|(t, x)| format!("{t:.1}s:{x:.0}K")).collect::<Vec<_>>());
    assert!(burnt > 1e-5 * m_c0, "le charbon n'a pas brûlé");
    assert!(tmax > 1000.0, "le feu s'est éteint : braises à {tmax} K");
    // tirage naturel : l'air monte à travers la grille (percolation)
    let gj = (0..s.ny).find(|&j| (0..s.nx).any(|i| s.is_grille(s.idx(i, j)))).unwrap();
    let draft: f64 = (0..s.nx).filter(|&i| s.is_grille(s.idx(i, gj))).map(|i| s.q_v[(gj + 1) * s.nx + i]).sum::<f64>() * s.dx;
    eprintln!("tirage à travers la grille : {draft:.4} m²/s");
    assert!(draft > 1e-3, "pas de tirage à travers la grille : {draft}");
    assert!((carbon(&s) / n_c0 - 1.0).abs() < 1e-6, "carbone (charbon + CO + CO2 + sorties) : {} → {}", n_c0, carbon(&s));
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
        let ps = s.grain_overburden();
        let i = s.nx / 2;
        let p0 = s.p[s.idx(i, grate_j + 8)];
        eprintln!("-- {tag} : colonne centrale (j, p − p(j+8), p_s, p − p_s − p(j+8), ρ_f, T)");
        for j in (grate_j.saturating_sub(3)..grate_j + 9).rev() {
            let q = s.idx(i, j);
            let rf: f64 = (0..s.reg.phases.len()).filter(|&k| !s.reg.phases[k].is_granular()).map(|k| s.mass[k][q]).sum::<f64>()
                / (0..s.reg.phases.len()).filter(|&k| !s.reg.phases[k].is_granular()).map(|k| s.alpha[k][q]).sum::<f64>().max(1e-12);
            eprintln!("   j {j:2} {}{:9.3} {:9.3} {:9.3} {:6.3} {:6.0}", if s.is_grille(q) { "G " } else { "  " }, s.p[q] - p0, ps[q], s.p[q] - ps[q] - p0, rf, s.temp[q]);
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
