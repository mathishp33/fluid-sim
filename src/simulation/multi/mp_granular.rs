//! Phases granulaires (sable, poudre, charbon…) : sédimentation et échange thermique grains ↔ fluide. Conception : GRANULAIRE.md.
//!
//! Une phase granulaire est une phase du mélange comme les autres (masses partielles, énergie par groupe, une seule vitesse par
//! cellule), mais DISPERSÉE : pas d'interface reconstruite (voir `plic_sweep`), et une vitesse de DÉRIVE par rapport au fluide :
//!
//! * vitesse terminale d'un grain isolé, de Stokes à Newton en une formule (Ferguson & Church 2004, grains naturels C1 = 18,
//!   C2 = 0.4) : w_t = R g d² / (18 ν + √(0.75 C2 R g d³)), R = (ρ_s − ρ_f)/ρ_f — négative si le grain est plus léger que le fluide
//!   (il remonte) ;
//! * entravement de Richardson–Zaki : w = w_t (1 − α_grains)^n, n de Rowe (1987) : n = (4.7 + 0.1175 Re^0.75)/(1 + 0.0175 Re^0.75),
//!   Re = |w_t| d / ν (≈ 4.7 en régime de Stokes, ≈ 2.4 en régime de Newton) ;
//! * l'opérateur est conservatif : à travers une face, un volume de grains descend et LE MÊME VOLUME de fluide de la cellule du
//!   dessous remonte (sa composition et son énergie) ; masses, énergie et fermeture Σ m/ρ = 1 sont conservées. Il s'arrête quand la
//!   cellule du dessous est tassée (α_grains = α_max).
//!
//! Échange thermique : par la surface des grains (aire volumique 6 α_s / d, Nu = 2 de Ranz–Marshall à petit Reynolds) :
//! G = 12 α_s k_f / d² (W/(m³·K)), pour chaque fluide de la cellule au prorata de son volume ; implicite, énergie exactement conservée.
//! Le « nœud d'interface » de `exchange_heat` (interface plane à l'échelle de la maille) ne convient pas aux grains : pour de l'air
//! à 1 cm de maille il donnait une constante de temps de ~800 s, au lieu de quelques millisecondes pour une poudre fine.

use super::mp_par::{for_ranges, MIN_HEAVY};

/// constante de Stefan–Boltzmann (W/(m²·K⁴))
pub(crate) const SIGMA_SB: f64 = 5.670374e-8;
/// facteur d'échange radiatif entre grains voisins d'un tas, ε / (2 − ε) pour une émissivité ε = 0.9 (charbon, bois, cendres)
pub(crate) const BED_RAD: f64 = 0.9 / (2.0 - 0.9);

/// Rayonnement dans un tas de grains : à haute température, les grains échangent surtout par rayonnement, qui se comporte dans le tas
/// comme une conduction (Damköhler, Schotte) de conductivité k_rad = 4 σ ε_r d T³ — 8.5 W/(m·K) pour des braises de 1.5 cm à 1450 K,
/// 30× la conduction du charbon de bois. Entre deux nœuds granulaires (cellules voisines), T³ pris à la face par la linéarisation exacte
/// de T_a⁴ − T_b⁴ : T_m³ = (T_a² + T_b²)(T_a + T_b) / 4 (une braise chauffe un morceau de bois froid même s'il rayonne peu), au prorata
/// des parts tassées f = α / α_max des deux cellules. Conductance (W/(m·K)) de la face.
#[inline]
pub(crate) fn bed_rad_k(d: f64, ta: f64, tb: f64, fa: f64, fb: f64) -> f64 {
    let tm3 = (ta * ta + tb * tb) * (ta + tb) * 0.25;
    4.0 * SIGMA_SB * BED_RAD * d * tm3 * fa * fb
}
use super::multiphase::{MultiphaseSim, SyncPtr};
use super::phases::MAX_PHASES;

/// fraction de grains en dessous de laquelle une cellule ne borne pas le pas de la sédimentation : la chute explicite d'une trace est
/// bornée par le contenu de la cellule donneuse (stable), sa précision est sans objet — des traces de houille de 1e-9 arrachées au lit
/// par la flamme d'un braséro (noix de 3 cm dans un gaz à 2200 K : 90 m/s) y fixaient le pas à 5e-5 s
pub(crate) const SETTLE_TRACE: f64 = 1e-6;

impl MultiphaseSim {
    /// la simulation contient-elle au moins une phase granulaire ?
    pub fn has_granular(&self) -> bool {
        self.reg.phases.iter().any(|p| p.is_granular())
    }

    /// groupe dispersé (uniquement des phases granulaires) ?
    pub(crate) fn group_is_granular(&self, g: usize) -> bool {
        self.groups[g].iter().all(|&k| self.reg.phases[k].is_granular())
    }

    /// température du fluide au contact des grains de c : groupes non granulaires présents, moyenne pondérée par leur fraction volumique
    /// (0 sans fluide). Une poudre s'allume dans un fluide plus chaud que sa température d'allumage (voir `pyrolysis_step`)
    pub(crate) fn fluid_contact_temp(&self, c: usize) -> f64 {
        let (mut a, mut at) = (0.0, 0.0);
        for g in 0..self.groups.len() {
            if self.group_is_granular(g) || self.group_mass(g, c) <= 0.0 {
                continue;
            }
            let al = self.group_alpha(g, c);
            a += al;
            at += al * self.tg[g][c];
        }
        if a > 0.0 { at / a } else { 0.0 }
    }

    /// fraction volumique de grains (toutes phases granulaires) de la cellule c
    pub(crate) fn grain_fraction(&self, c: usize) -> f64 {
        (0..self.reg.phases.len()).filter(|&k| self.reg.phases[k].is_granular()).map(|k| self.alpha[k][c]).sum()
    }

    /// (ρ_f, μ_f, k_f, fraction volumique) du FLUIDE de la cellule (contenu non granulaire) ; None s'il n'y en a pas
    fn carrier_fluid(&self, c: usize) -> Option<(f64, f64, f64, f64)> {
        let (mut a, mut m, mut mu, mut k) = (0.0, 0.0, 0.0, 0.0);
        for (q, ph) in self.reg.phases.iter().enumerate() {
            if ph.is_granular() {
                continue;
            }
            let aq = self.alpha[q][c];
            if aq <= 0.0 {
                continue;
            }
            let t = self.phase_temp(q, c);
            a += aq;
            m += self.mass[q][c].max(0.0);
            mu += aq * ph.mu(t);
            k += aq * ph.k(t);
        }
        if a <= 1e-12 || m <= 0.0 {
            return None;
        }
        Some((m / a, mu / a, k / a, a))
    }

    /// Vitesse de chute ENTRAVÉE (m/s, > 0 dans le sens de la gravité, < 0 si le grain remonte) de la phase granulaire s dans la
    /// cellule c, et l'exposant n de Richardson–Zaki utilisé. 0 sans gravité, sans fluide ou sans grains.
    pub(crate) fn settling_velocity(&self, s: usize, c: usize) -> f64 {
        let g = (self.gravity[0].powi(2) + self.gravity[1].powi(2)).sqrt();
        if g == 0.0 || self.alpha[s][c] <= 0.0 {
            return 0.0;
        }
        let Some((rho_f, mu_f, _, _)) = self.carrier_fluid(c) else { return 0.0 };
        let ph = &self.reg.phases[s];
        let d = ph.grain_diameter;
        if !(d > 0.0) || !(mu_f > 0.0) {
            return 0.0;
        }
        let rho_s = ph.rho(self.p[c], self.phase_temp(s, c));
        let r = (rho_s - rho_f) / rho_f;
        let nu = mu_f / rho_f;
        let wt = r * g * d * d / (18.0 * nu + (0.75 * 0.4 * r.abs() * g * d * d * d).sqrt());
        let re = (wt.abs() * d / nu).powf(0.75);
        let n = (4.7 + 0.1175 * re) / (1.0 + 0.0175 * re);
        wt * (1.0 - self.grain_fraction(c)).clamp(0.0, 1.0).powf(n)
    }

    /// plus grande vitesse de chute du domaine (limite de pas : w·dt ≤ 0.5·dx), hors traces de grains (α ≤ `SETTLE_TRACE`)
    pub(crate) fn max_settling_speed(&self) -> f64 {
        let mut w = 0.0f64;
        for (s, ph) in self.reg.phases.iter().enumerate() {
            if !ph.is_granular() {
                continue;
            }
            for c in 0..self.nx * self.ny {
                if !self.is_solid(c) && self.alpha[s][c] > SETTLE_TRACE {
                    w = w.max(self.settling_velocity(s, c).abs());
                }
            }
        }
        w
    }

    /// Sédimentation explicite des phases granulaires pendant dt (voir l'en-tête du module). À appeler sur un état rafraîchi (α).
    pub(crate) fn settle_grains(&mut self, dt: f64) {
        let np = self.reg.phases.len();
        let grains: Vec<usize> = (0..np).filter(|&k| self.reg.phases[k].is_granular()).collect();
        let (gx, gy) = (self.gravity[0], self.gravity[1]);
        let gn = (gx * gx + gy * gy).sqrt();
        if grains.is_empty() || gn == 0.0 {
            return;
        }
        let (nx, ny) = (self.nx, self.ny);
        // gravité oblique : deux familles de faces, chacune limitée à la moitié des stocks
        let share = if gx != 0.0 && gy != 0.0 { 0.5 } else { 1.0 };
        let movable = |s: &Self, c: usize| !s.is_solid(c) && !s.pinned(c) && !s.is_grille(c);
        for &s in &grains {
            let gs = self.group_of[s];
            let amax = self.reg.phases[s].alpha_max;
            // transferts (donneuse de grains, receveuse, volume) évalués sur l'état d'avant l'opérateur
            let mut moves: Vec<(usize, usize, f64)> = vec![];
            for (axis, ga) in [(0usize, gx), (1usize, gy)] {
                if ga == 0.0 {
                    continue;
                }
                let proj = ga.abs() / gn;
                let (fi, fj) = if axis == 0 { (nx - 1, ny) } else { (nx, ny - 1) };
                for j in 0..fj {
                    for i in 0..fi {
                        // a : cellule « amont » au sens de la gravité (d'où tombent les grains plus lourds que le fluide), b : « aval »
                        let (c0, c1) = if axis == 0 { (self.idx(i, j), self.idx(i + 1, j)) } else { (self.idx(i, j), self.idx(i, j + 1)) };
                        let (a, b) = if ga < 0.0 { (c1, c0) } else { (c0, c1) };
                        if !movable(self, a) || !movable(self, b) {
                            continue;
                        }
                        // grains lourds : de a vers b ; grains légers (w < 0) : de b vers a (vitesse évaluée dans la donneuse)
                        let wa = self.settling_velocity(s, a);
                        let wb = self.settling_velocity(s, b);
                        let (d, r, w) = if wa > 0.0 { (a, b, wa) } else if wb < 0.0 { (b, a, -wb) } else { continue };
                        let ad = self.alpha[s][d];
                        if ad <= 1e-12 {
                            continue;
                        }
                        // un grain léger qui remonte (glace dans l'eau) ne passe pas dans un fluide où il serait lourd (l'air au-dessus de la
                        // surface, où il retomberait aussitôt) : il flotte dans la dernière maille d'eau ; un grain qui tombe de l'air dans
                        // l'eau y entre et y flotte (sans cela, ce va-et-vient à chaque pas brassait la surface et bornait le pas)
                        if wa <= 0.0 {
                            let Some((rho_fr, _, _, _)) = self.carrier_fluid(r) else { continue };
                            if self.reg.phases[s].rho(self.p[d], self.phase_temp(s, d)) >= rho_fr {
                                continue;
                            }
                        }
                        let mut vol = ad * w * proj * dt / self.dx;
                        vol = vol.min(share * ad);
                        vol = vol.min(share * (amax - self.grain_fraction(r)).max(0.0));
                        let fluid_r: f64 = (0..np).filter(|&k| !self.reg.phases[k].is_granular()).map(|k| self.alpha[k][r]).sum();
                        vol = vol.min(share * fluid_r);
                        if vol > 1e-14 {
                            moves.push((d, r, vol));
                        }
                    }
                }
            }
            if moves.is_empty() {
                continue;
            }
            let (mass0, alpha0, tg0) = (self.mass.clone(), self.alpha.clone(), self.tg.clone());
            let mut touched = vec![false; nx * ny];
            for (d, r, vol) in moves {
                // grains : masse volumique des grains de la donneuse (m/α), énergie à la température de leur groupe
                let rho_s = mass0[s][d] / alpha0[s][d];
                let dm = (vol * rho_s).min(self.mass[s][d].max(0.0));
                let de = dm * self.reg.phases[s].e_int(tg0[gs][d]);
                self.mass[s][d] -= dm;
                self.mass[s][r] += dm;
                self.eg[gs][d] -= de;
                self.eg[gs][r] += de;
                // le même volume de fluide de la receveuse remonte (composition et températures de la receveuse)
                let fluid_r: f64 = (0..np).filter(|&k| !self.reg.phases[k].is_granular()).map(|k| alpha0[k][r]).sum();
                for k in 0..np {
                    if self.reg.phases[k].is_granular() || alpha0[k][r] <= 0.0 {
                        continue;
                    }
                    let gk = self.group_of[k];
                    let dmk = (vol * mass0[k][r] / fluid_r).min(self.mass[k][r].max(0.0));
                    let dek = dmk * self.reg.phases[k].e_int(tg0[gk][r]);
                    self.mass[k][r] -= dmk;
                    self.mass[k][d] += dmk;
                    self.eg[gk][r] -= dek;
                    self.eg[gk][d] += dek;
                }
                touched[d] = true;
                touched[r] = true;
            }
            for c in 0..nx * ny {
                if touched[c] {
                    self.cell_sync(c);
                }
            }
        }
    }

    /// Échange thermique grains ↔ fluides de la cellule par la surface des grains (voir l'en-tête du module). `exchange_heat` laisse
    /// les groupes granulaires de côté ; λ = ∞ (« une seule température par cellule ») : c'est lui qui égalise tout, rien à faire ici.
    pub(crate) fn exchange_grains(&mut self, dt: f64) {
        let lam = self.phase_heat_exchange;
        if lam <= 0.0 || !lam.is_finite() {
            return;
        }
        let ng = self.groups.len();
        let gran: Vec<bool> = (0..ng).map(|g| self.group_is_granular(g)).collect();
        if !gran.iter().any(|&b| b) {
            return;
        }
        // (en parallèle : chaque cellule ne dépend que d'elle-même ; énergies mises à jour puis températures, comme `cell_sync`)
        let n = self.nx * self.ny;
        let ep = SyncPtr(self.e.as_mut_ptr());
        let tp = SyncPtr(self.temp.as_mut_ptr());
        let egp: Vec<SyncPtr<f64>> = self.eg.iter_mut().map(|v| SyncPtr(v.as_mut_ptr())).collect();
        let tgp: Vec<SyncPtr<f64>> = self.tg.iter_mut().map(|v| SyncPtr(v.as_mut_ptr())).collect();
        let this = &*self;
        let gran = &gran;
        for_ranges(n, MIN_HEAVY, |lo, hi| {
        let mut egn = [0.0f64; MAX_PHASES];
        let mut tgn = [0.0f64; MAX_PHASES];
        for c in lo..hi {
            if this.is_solid(c) {
                continue;
            }
            let a_fluid: f64 = (0..ng).filter(|&g| !gran[g]).map(|g| this.group_alpha(g, c)).sum();
            if a_fluid <= 1e-12 {
                continue;
            }
            for g in 0..ng {
                egn[g] = this.eg[g][c];
            }
            let mut changed = false;
            for s in 0..ng {
                if !gran[s] || this.group_mass(s, c) <= 0.0 {
                    continue;
                }
                let d = this.reg.phases[this.groups[s][0]].grain_diameter.max(1e-6);
                let a_s = this.group_alpha(s, c);
                for f in 0..ng {
                    if gran[f] || this.group_mass(f, c) <= 0.0 {
                        continue;
                    }
                    let (ts, tf) = (this.tg[s][c], this.tg[f][c]);
                    let g = lam * 12.0 * a_s * this.group_k(f, c) / (d * d) * this.group_alpha(f, c) / a_fluid;
                    let (cs, cf) = (this.group_cap_at(s, c, ts).max(1e-300), this.group_cap_at(f, c, tf).max(1e-300));
                    // échange implicite à deux corps : écart relaxé sans dépassement, énergie conservée exactement
                    let q = g * dt * (tf - ts) / (1.0 + g * dt * (1.0 / cs + 1.0 / cf));
                    egn[s] += q;
                    egn[f] -= q;
                    changed = true;
                }
            }
            // rayonnement entre deux populations de grains de la cellule (braises et bois mêlés) : h = 4 σ ε_r T_m³ sur l'aire des grains
            // de l'une tournée vers l'autre (la plus petite des deux estimations S_v,1 α_2 / α_s et S_v,2 α_1 / α_s), implicite à deux corps
            let a_s: f64 = (0..ng).filter(|&q| gran[q]).map(|q| this.group_alpha(q, c)).sum();
            for s1 in 0..ng {
                if !gran[s1] || this.group_mass(s1, c) <= 0.0 {
                    continue;
                }
                for s2 in s1 + 1..ng {
                    if !gran[s2] || this.group_mass(s2, c) <= 0.0 {
                        continue;
                    }
                    let (t1, t2) = (this.tg[s1][c], this.tg[s2][c]);
                    let (a1, a2) = (this.group_alpha(s1, c), this.group_alpha(s2, c));
                    let d1 = this.reg.phases[this.groups[s1][0]].grain_diameter.max(1e-6);
                    let d2 = this.reg.phases[this.groups[s2][0]].grain_diameter.max(1e-6);
                    let area = (6.0 * a1 / d1 * a2 / a_s.max(1e-30)).min(6.0 * a2 / d2 * a1 / a_s.max(1e-30));
                    let gq = lam * 4.0 * super::mp_granular::SIGMA_SB * BED_RAD * (t1 * t1 + t2 * t2) * (t1 + t2) * 0.25 * area;
                    let (c1, c2) = (this.group_cap_at(s1, c, t1).max(1e-300), this.group_cap_at(s2, c, t2).max(1e-300));
                    let q = gq * dt * (t2 - t1) / (1.0 + gq * dt * (1.0 / c1 + 1.0 / c2));
                    egn[s1] += q;
                    egn[s2] -= q;
                    changed = true;
                }
            }
            if changed {
                let e: f64 = egn[..ng].iter().fold(0.0, |a, &b| a + b);
                let (ec, tc) = this.sync_finish(c, &mut egn, &mut tgn, e);
                // SAFETY : la cellule c n'est écrite que par sa tranche (et lue avant)
                unsafe {
                    *ep.at(c) = ec;
                    *tp.at(c) = tc;
                    for g in 0..ng {
                        *egp[g].at(c) = egn[g];
                        *tgp[g].at(c) = tgn[g];
                    }
                }
            }
        }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::super::phases::{Phase, Registry};
    use super::*;

    /// boîte fermée de liquide (eau) contenant une suspension uniforme de sable (fraction a0, grains de diamètre d)
    fn suspension(nx: usize, ny: usize, dx: f64, a0: f64, d: f64) -> (MultiphaseSim, usize, usize) {
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
        (s, eau, sable)
    }

    #[test]
    fn settling_velocity_matches_stokes_newton_and_buoyant_limits() {
        // grain isolé de 20 µm dans l'eau (Re ≈ 0.007) : régime de Stokes, w = Δρ g d² / (18 μ). (À 0.1 mm, Re ≈ 1 : la formule
        // donne ~11 % de moins que Stokes, la correction de traînée attendue — Schiller–Naumann : ~13 %)
        let (s, _, sable) = suspension(4, 4, 0.01, 1e-6, 2e-5);
        let w = s.settling_velocity(sable, 5);
        let stokes = (2650.0 - 998.2) * 9.81 * 4e-10 / (18.0 * 1.002e-3);
        assert!((w / stokes - 1.0).abs() < 0.03, "w = {w}, Stokes = {stokes}");
        // grain de 3 mm dans l'air : régime de Newton, w → √(R g d / (0.75 C2)) (C2 = 0.4)
        let mut r = Registry::new();
        let air = r.add_phase(Phase::air());
        let mut sand = Phase::sand();
        sand.grain_diameter = 3e-3;
        let sable = r.add_phase(sand);
        let mut s = MultiphaseSim::new(4, 4, 0.01, r, air);
        s.paint_mixture(move |_, _| Some(vec![1.0 - 1e-6, 1e-6]));
        s.finalize_init(|_, _| 293.15);
        let w = s.settling_velocity(sable, 5);
        let rr: f64 = (2650.0 - 1.204) / 1.204;
        let newton = (rr * 9.81 * 3e-3 / (0.75 * 0.4)).sqrt();
        assert!(w > 0.8 * newton && w < newton, "w = {w}, Newton = {newton}");
        // grain plus léger que le liquide (glace dans l'eau) : il remonte
        let (mut s, _, sable) = suspension(4, 4, 0.01, 1e-6, 1e-3);
        s.reg.phases[sable].eos = super::super::phases::Eos::Tait {
            p0: 101325.0,
            b: 3e8,
            n: 7.0,
            thermal: super::super::phases::Thermal::Beta { rho0: 917.0, t0: 293.15, beta: 1.5e-4 },
        };
        assert!(s.settling_velocity(sable, 5) < 0.0);
    }

    #[test]
    fn uniform_suspension_clears_from_the_top_at_the_hindered_velocity() {
        // sable de 0.1 mm à 10 % dans 6 cm d'eau : la limite eau claire / suspension descend à w_t (1 − α)^n
        let (dx, ny, a0) = (1e-3, 60, 0.1);
        let (mut s, _, sable) = suspension(6, ny, dx, a0, 1e-4);
        let w0 = s.settling_velocity(sable, s.idx(3, ny / 2));
        let front = |s: &MultiphaseSim| {
            let j = (0..ny).rev().find(|&j| s.alpha[sable][s.idx(3, j)] > 0.5 * a0).unwrap_or(0);
            (j as f64 + 0.5) * dx
        };
        let y0 = front(&s);
        while s.time < 2.0 {
            s.step();
        }
        assert!(!s.diverged && s.rollbacks == 0);
        let drop = y0 - front(&s);
        let expect = w0 * s.time;
        eprintln!("w0 = {w0:.3e} m/s ; front descendu de {drop:.4} m (attendu {expect:.4})");
        assert!((drop / expect - 1.0).abs() < 0.25, "descente {drop}, attendue {expect}");
    }

    #[test]
    fn sediment_forms_a_packed_layer_and_conserves_mass_and_energy() {
        // grains de 0.5 mm (≈ 9 cm/s) : en 3 s tout est au fond, couche tassée d'épaisseur α0·H/α_max = 1 cm
        let (dx, ny, a0) = (1e-3, 60, 0.1);
        let (mut s, eau, sable) = suspension(6, ny, dx, a0, 5e-4);
        let (m_s0, m_w0, u0) = (s.total_mass(sable), s.total_mass(eau), s.total_internal_energy());
        while s.time < 3.0 {
            s.step();
        }
        assert!(!s.diverged && s.state_is_sane());
        let col: Vec<f64> = (0..ny).map(|j| s.alpha[sable][s.idx(3, j)]).collect();
        let amax = s.reg.phases[sable].alpha_max;
        let bottom: f64 = col[..8].iter().sum::<f64>() / 8.0;
        let top: f64 = col[14..].iter().fold(0.0f64, |a, &b| a.max(b));
        let shown: Vec<f64> = col[..14].iter().map(|a| (a * 100.0).round() / 100.0).collect();
        eprintln!("fond : α = {bottom:.3} ; max au-dessus de 1.4 cm {top:.2e} ; colonne {shown:?}");
        assert!(bottom > 0.95 * amax && bottom <= amax + 1e-6, "fond {bottom}");
        assert!(top < 1e-2, "grains restés en suspension : {top}");
        assert!((s.total_mass(sable) / m_s0 - 1.0).abs() < 1e-12, "masse de sable");
        assert!((s.total_mass(eau) / m_w0 - 1.0).abs() < 1e-12, "masse d'eau");
        let du = (s.total_internal_energy() + s.kinetic_energy() - u0) / u0.abs();
        assert!(du.abs() < 1e-6, "énergie : {du:e}");
    }

    #[test]
    fn hot_powder_relaxes_to_the_gas_temperature_through_the_grain_surface() {
        // poudre de 0.3 mm à 5 %, à 600 K dans de l'air à 300 K, sans gravité : écart divisé par e en τ = 1/(G (1/C_s + 1/C_f)),
        // G = 12 α k_f / d² (Nu = 2). Conductivité de l'air constante ici (Sutherland : l'air, porté vers 600 K, conduirait ~1.5×
        // mieux et l'échange serait plus rapide — physique, mais pas la formule à k constant)
        let mut r = Registry::new();
        let mut air_k = Phase::air();
        air_k.k_law = crate::simulation::materials::PropertyLaw::Constant;
        let air = r.add_phase(air_k);
        let sable = r.add_phase(Phase::sand());
        let mut s = MultiphaseSim::new(6, 6, 0.01, r, air);
        s.gravity = [0.0, 0.0];
        s.paint_mixture(move |_, _| Some(vec![0.95, 0.05]));
        s.finalize_init(|_, _| 300.0);
        let (ga, gs) = (s.group_of[air], s.group_of[sable]);
        for c in 0..36 {
            s.set_group_temperature(c, gs, 600.0);
        }
        s.refresh();
        let c = s.idx(3, 3);
        let g = 12.0 * s.group_alpha(gs, c) * s.group_k(ga, c) / (3e-4f64).powi(2);
        let (cs, cf) = (s.group_cap_at(gs, c, 600.0), s.group_cap_at(ga, c, 300.0));
        let tau = 1.0 / (g * (1.0 / cs + 1.0 / cf));
        let u0 = s.total_internal_energy();
        s.dt_max = tau / 50.0;
        while s.time < tau {
            s.step();
        }
        let gap = (s.tg[gs][c] - s.tg[ga][c]) / 300.0;
        let ratio = gap / (-(s.time / tau)).exp();
        eprintln!("τ = {tau:.3e} s ; écart relatif après t = {:.3e} : {gap:.4} (rapport à e^-t/τ : {ratio:.3})", s.time);
        assert!((ratio - 1.0).abs() < 0.1, "écart / e^(−t/τ) = {ratio}");
        assert!(((s.total_internal_energy() + s.kinetic_energy()) / u0 - 1.0).abs() < 1e-9, "énergie");
    }

    #[test]
    fn a_uniform_sand_water_mixture_stays_uniform_under_flow() {
        // transport sans interface : un écoulement tourbillonnaire (sans gravité) ne doit pas « trier » une suspension uniforme
        let (mut s, _, sable) = suspension(24, 24, 0.005, 0.2, 3e-4);
        s.gravity = [0.0, 0.0];
        let (nx, ny) = (s.nx, s.ny);
        let l = nx as f64 * s.dx;
        let pi = std::f64::consts::PI;
        for j in 0..ny {
            for i in 0..=nx {
                let (x, y) = (i as f64 * s.dx, (j as f64 + 0.5) * s.dx);
                let q = s.uidx(i, j);
                s.u[q] = 0.2 * (pi * x / l).sin() * (pi * y / l).cos();
            }
        }
        for j in 0..=ny {
            for i in 0..nx {
                let (x, y) = ((i as f64 + 0.5) * s.dx, j as f64 * s.dx);
                let q = s.vidx(i, j);
                s.v[q] = -0.2 * (pi * x / l).cos() * (pi * y / l).sin();
            }
        }
        while s.time < 0.2 {
            s.step();
        }
        assert!(!s.diverged);
        let (lo, hi) = (0..nx * ny).map(|c| s.alpha[sable][c]).fold((1.0f64, 0.0f64), |(a, b), x| (a.min(x), b.max(x)));
        eprintln!("α sable : {lo:.6} … {hi:.6}");
        assert!(hi - lo < 1e-3, "suspension triée par l'écoulement : {lo} … {hi}");
    }

    #[test]
    fn sand_cloud_falls_through_the_water_surface_and_settles() {
        use super::super::mp_view::{build_scenario, MpScenario};
        let mut s = build_scenario(MpScenario::Sedimentation, 40, 30, 0.01, 9.81);
        let sable = s.reg.phases.iter().position(|p| p.is_granular()).unwrap();
        let m0 = s.total_mass(sable);
        assert!(m0 > 0.0);
        while s.time < 1.5 && !s.diverged {
            s.step();
        }
        assert!(!s.diverged && s.state_is_sane());
        let surface = 0.4 * s.ny as f64;
        let below: f64 = (0..s.nx * s.ny).filter(|&c| ((c / s.nx) as f64) < surface).map(|c| s.mass[sable][c]).sum::<f64>() * s.dx * s.dx;
        eprintln!("sable sous la surface : {:.1} % ; masse {:.3e} → {:.3e}", 100.0 * below / m0, m0, s.total_mass(sable));
        assert!((s.total_mass(sable) / m0 - 1.0).abs() < 1e-9, "masse de sable");
        assert!(below > 0.9 * m0, "le sable n'est pas passé dans l'eau : {}", below / m0);
    }
}
