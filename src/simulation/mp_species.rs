//! Espèces chimiques : diffusion moléculaire, cinétique chimique, flammes épaissies et rayonnement des gaz chauds.
//!
//! * Diffusion (groupes MISCIBLES) : loi de Fick par espèce avec vitesse de correction (approximation « mélange » de Hirschfelder–Curtiss) :
//!   J_k = −ρ D_k ∇Y_k + Y_k Σ_j ρ D_j ∇Y_j, donc Σ_k J_k = 0 (la masse du mélange n'est pas transportée par la diffusion) et chaque espèce
//!   a SA diffusivité (l'hydrogène diffuse 3 à 4 fois plus vite que l'oxygène : instabilités thermo-diffusives des flammes pauvres).
//!   D_k(T, p) : gaz ∝ T^1.75/p, liquides ∝ T/μ. L'énergie interne accompagne chaque espèce : l'énergie totale est conservée exactement.
//!
//! * Cinétique (`react_step`) : loi d'Arrhenius modifiée + action de masse, réactions réversibles par l'équilibre thermodynamique
//!   (voir `phases::Reaction`). Intégration cellule par cellule à volume et énergie interne constants (découplage d'opérateurs :
//!   l'expansion est ensuite produite par la projection de pression) avec sous-pas adaptatifs — limite d'échauffement de 25 K et de
//!   consommation de 30 % par sous-pas pour les réactions irréversibles, Euler IMPLICITE (Newton) pour les réactions réversibles,
//!   raides près de l'équilibre (dissociation de CO2 / H2O). Masse exactement conservée réaction par réaction, énergie interne inchangée :
//!   la chaleur de réaction vient des enthalpies de formation portées par les décalages e0.
//!   Couplage avec l'écoulement : la chimie d'un pas ne peut pas augmenter le volume d'une cellule (à p, T) de plus de `chem_volume_limit` ;
//!   le pas de temps suivant est réduit pour suivre les dégagements de chaleur violents (allumage, explosion).
//!
//! * Flammes de prémélange (`update_flame_factor`) : modèle de flamme épaissie dynamique (TFM, Colin et al. 2000 ; Légier et al. 2000).
//!   Une flamme réelle a une épaisseur δ_L ≈ 0.3–0.5 mm, bien plus petite que la maille : on multiplie les diffusivités (thermique et des
//!   espèces) par F et on divise la vitesse de réaction par F dans la zone de flamme. La vitesse de flamme S_L ∝ √(D·ω) est conservée et
//!   la flamme est résolue sur ≈ `flame_resolution` mailles. Hors de la flamme (senseur : dégagement de chaleur, dilaté sur l'épaisseur),
//!   F = 1 : l'auto-allumage et le mélange ne sont pas modifiés.
//!
//! * Rayonnement (`radiate`) : modèle optiquement mince (TNF Workshop) q = 4σ Σ_k p_k a_P,k(T) (T⁴ − T_amb⁴), coefficients moyens de
//!   Planck de H2O et CO2. Pertes des gaz brûlés (refroidissement des fumées) ; surestime les pertes des très grandes flammes (opaques).

use super::multiphase::{temp_from_masses, MultiphaseSim};
use super::phases::{Phase, Reaction, Radiator};

/// constante de Stefan–Boltzmann (W/(m²·K⁴))
const SIGMA_SB: f64 = 5.670374419e-8;
/// plancher de concentration pour les ordres négatifs ou les exposants (o − ν) < 0 de la vitesse inverse (mol/m³)
const C_FLOOR: f64 = 1e-3;
/// nombre maximal de sous-pas chimiques par cellule et par pas
const MAX_CHEM_SUBSTEPS: usize = 4000;

/// état local d'une cellule pendant l'intégration chimique
struct CellChem<'a> {
    phases: &'a [Phase],
    rx: &'a [Reaction],
    /// masses partielles (kg/m³)
    m: Vec<f64>,
    /// énergie interne (J/m³), constante
    e: f64,
    /// facteur d'épaississement local
    f_eff: f64,
    /// rapport (masse des réactifs / masse des produits) stœchiométrique par réaction : impose la conservation exacte de la masse
    scale: Vec<f64>,
}

impl CellChem<'_> {
    fn temp(&self) -> f64 {
        temp_from_masses(self.phases, self.m.iter().copied(), self.e)
    }
    fn cap(&self, t: f64) -> f64 {
        self.m.iter().enumerate().map(|(k, &mk)| mk.max(0.0) * self.phases[k].c_e_at(t)).sum()
    }
    fn conc(&self, k: usize) -> f64 {
        self.m[k].max(0.0) / self.phases[k].molar_mass
    }
    /// vitesse NETTE de la réaction j (mol/m³/s) à la température t
    fn rate(&self, j: usize, t: f64) -> f64 {
        let r = &self.rx[j];
        let k = r.k_forward(t) / if r.flame_thickness > 0.0 { self.f_eff } else { 1.0 };
        if k <= 0.0 {
            return 0.0;
        }
        let mut modif = 1.0;
        for &(s, o) in &r.modifiers {
            let c = self.conc(s);
            modif *= if o < 0.0 { c.max(C_FLOOR).powf(o) } else if o == 0.0 { 1.0 } else { c.powf(o) };
        }
        let mut rf = k * modif;
        for (i, &(s, _)) in r.reactants.iter().enumerate() {
            let o = r.order(i);
            let c = self.conc(s);
            rf *= if o < 0.0 { c.max(C_FLOOR).powf(o) } else if o == 0.0 { if c > 0.0 { 1.0 } else { 0.0 } } else { c.powf(o) };
        }
        if !r.reversible {
            return rf;
        }
        // vitesse inverse k/Kc · Π c_prod^ν · Π c_réac^(o−ν) : l'équilibre Q = Kc est exact quels que soient les ordres
        let ln_kc = r.ln_kc(self.phases, t);
        let mut ln_rr = (k * modif).ln() - ln_kc;
        for &(s, nu) in &r.products {
            let c = self.conc(s);
            if c <= 0.0 {
                return rf;
            }
            ln_rr += nu * c.ln();
        }
        for (i, &(s, nu)) in r.reactants.iter().enumerate() {
            let d = r.order(i) - nu;
            if d != 0.0 {
                ln_rr += d * self.conc(s).max(C_FLOOR).ln();
            }
        }
        rf - ln_rr.min(700.0).exp()
    }
    /// avancement maximal (mol/m³) dans le sens direct (> 0) et inverse (< 0) permis par les masses disponibles
    fn bounds(&self, j: usize) -> (f64, f64) {
        let r = &self.rx[j];
        let fwd = r.reactants.iter().map(|&(s, nu)| self.m[s].max(0.0) / (nu * self.phases[s].molar_mass)).fold(f64::INFINITY, f64::min);
        let rev = if r.reversible {
            r.products.iter().map(|&(s, nu)| self.m[s].max(0.0) / (nu * self.phases[s].molar_mass * self.scale[j])).fold(f64::INFINITY, f64::min)
        } else {
            0.0
        };
        (-rev, fwd)
    }
    /// applique l'avancement xi (mol/m³) de la réaction j (masse totale exactement conservée)
    fn apply(&mut self, j: usize, xi: f64) {
        let r = &self.rx[j];
        for &(s, nu) in &r.reactants {
            self.m[s] = (self.m[s] - nu * self.phases[s].molar_mass * xi).max(0.0);
        }
        for &(s, nu) in &r.products {
            self.m[s] = (self.m[s] + nu * self.phases[s].molar_mass * xi * self.scale[j]).max(0.0);
        }
    }
    /// −Δu_j(T) (J/mol) : chaleur dégagée par mole d'avancement à volume constant
    fn heat(&self, j: usize, t: f64) -> f64 {
        let r = &self.rx[j];
        let s = |v: &[(usize, f64)], f: f64| v.iter().map(|&(k, nu)| nu * f * self.phases[k].molar_mass * self.phases[k].e_int(t)).sum::<f64>();
        s(&r.reactants, 1.0) - s(&r.products, self.scale[j])
    }
    /// volume occupé par les masses à (p, T) rapporté à la cellule (≈ 1 à l'équilibre mécanique)
    fn volume(&self, p: f64, t: f64) -> f64 {
        self.m.iter().enumerate().map(|(k, &mk)| mk.max(0.0) / self.phases[k].rho(p, t)).sum()
    }
}

/// Composition (fractions volumiques = molaires, gaz parfaits) après combustion COMPLÈTE d'un mélange par les réactions de flamme
/// (δ_L > 0) puis les autres réactions irréversibles, chacune jusqu'à épuisement de son réactif limitant. Sert à initialiser des noyaux
/// de gaz brûlés (allumage sans explosion parasite du noyau).
pub fn complete_combustion(reg: &super::phases::Registry, alpha: &[f64]) -> Vec<f64> {
    let mut n = alpha.to_vec();
    let mut order: Vec<usize> = (0..reg.reactions.len()).filter(|&j| reg.reactions[j].flame_thickness > 0.0).collect();
    order.extend((0..reg.reactions.len()).filter(|&j| reg.reactions[j].flame_thickness <= 0.0 && !reg.reactions[j].reversible));
    for _ in 0..3 {
        for &j in &order {
            let r = &reg.reactions[j];
            let xi = r.reactants.iter().map(|&(k, nu)| n[k].max(0.0) / nu).fold(f64::INFINITY, f64::min);
            if !(xi > 0.0) || !xi.is_finite() {
                continue;
            }
            for &(k, nu) in &r.reactants {
                n[k] = (n[k] - nu * xi).max(0.0);
            }
            for &(k, nu) in &r.products {
                n[k] += nu * xi;
            }
        }
    }
    let tot: f64 = n.iter().sum();
    n.iter().map(|x| x / tot.max(1e-300)).collect()
}

impl MultiphaseSim {
    // ================================================================ diffusion des espèces
    /// Diffusion de Fick des espèces dans les groupes miscibles (diffusivité propre à chaque espèce + vitesse de correction).
    /// Sans groupe miscible (cas par défaut) : ne fait rien.
    pub(crate) fn diffuse_species(&mut self, dt: f64) {
        let groups: Vec<Vec<usize>> = self.reg.miscible_groups().into_iter().filter(|g| g.len() > 1).collect();
        if groups.is_empty() {
            return;
        }
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let n = nx * ny;
        for g in &groups {
            if g.iter().all(|&k| !(self.reg.phases[k].diffusivity > 0.0)) {
                continue;
            }
            // diffusivités locales D_k(T, p) × F_eff (flamme épaissie)
            let dloc: Vec<Vec<f64>> = g
                .iter()
                .map(|&k| {
                    (0..n)
                        .map(|c| if self.is_solid(c) { 0.0 } else { self.reg.phases[k].diffusivity_at(self.temp[c], self.p[c]).max(0.0) * self.flame_factor[c].max(1.0) })
                        .collect()
                })
                .collect();
            let dmax = dloc.iter().flat_map(|v| v.iter()).fold(0.0f64, |a, &b| a.max(b));
            if dmax <= 0.0 {
                continue;
            }
            let nsub = ((dt * dmax / (0.2 * dx * dx)).ceil() as usize).clamp(1, 500);
            let ds = dt / nsub as f64;
            let ng = g.len();
            let mut jf = vec![0.0; ng];
            for _ in 0..nsub {
                let mg: Vec<f64> = (0..n).map(|c| if self.is_solid(c) { 0.0 } else { g.iter().map(|&k| self.mass[k][c].max(0.0)).sum() }).collect();
                let mut dm: Vec<Vec<f64>> = vec![vec![0.0; n]; ng];
                let mut de = vec![0.0; n];
                for j in 0..ny {
                    for i in 0..nx {
                        let a = j * nx + i;
                        for (di, dj) in [(1usize, 0usize), (0, 1)] {
                            let (ii, jj) = (i + di, j + dj);
                            if ii >= nx || jj >= ny {
                                continue;
                            }
                            let b = jj * nx + ii;
                            if self.is_solid(a) || self.is_solid(b) || mg[a] <= 1e-12 || mg[b] <= 1e-12 {
                                continue;
                            }
                            let mbar = 0.5 * (mg[a] + mg[b]);
                            // flux bruts −ρ D_k ∇Y_k puis correction : J_k −= Ȳ_k Σ J
                            let mut jsum = 0.0;
                            for (q, &k) in g.iter().enumerate() {
                                let (ya, yb) = (self.mass[k][a].max(0.0) / mg[a], self.mass[k][b].max(0.0) / mg[b]);
                                let dk = 2.0 * dloc[q][a] * dloc[q][b] / (dloc[q][a] + dloc[q][b]).max(1e-300);
                                jf[q] = dk * mbar * (ya - yb) / (dx * dx) * ds;
                                jsum += jf[q];
                            }
                            let mut scale = 1.0f64;
                            for (q, &k) in g.iter().enumerate() {
                                let ybar = 0.5 * (self.mass[k][a].max(0.0) / mg[a] + self.mass[k][b].max(0.0) / mg[b]);
                                jf[q] -= ybar * jsum;
                                // jamais plus que le quart de la masse de l'espèce dans la cellule donneuse (facteur commun : Σ J = 0 conservé)
                                let avail = 0.25 * if jf[q] > 0.0 { self.mass[k][a].max(0.0) } else { self.mass[k][b].max(0.0) };
                                if jf[q].abs() > avail {
                                    scale = scale.min(avail / jf[q].abs());
                                }
                            }
                            for (q, &k) in g.iter().enumerate() {
                                let flux = jf[q] * scale;
                                if flux == 0.0 {
                                    continue;
                                }
                                let t_don = if flux > 0.0 { self.temp[a] } else { self.temp[b] };
                                let ef = flux * self.reg.phases[k].e_int(t_don);
                                dm[q][a] -= flux;
                                dm[q][b] += flux;
                                de[a] -= ef;
                                de[b] += ef;
                            }
                        }
                    }
                }
                for (q, &k) in g.iter().enumerate() {
                    for c in 0..n {
                        self.mass[k][c] = (self.mass[k][c] + dm[q][c]).max(0.0);
                    }
                }
                for c in 0..n {
                    self.e[c] += de[c];
                }
                for c in 0..n {
                    if !self.is_solid(c) {
                        self.temp[c] = self.temp_from_energy(c);
                    }
                }
            }
        }
    }

    // ================================================================ flamme épaissie
    /// Facteur d'épaississement F_eff de chaque cellule pour ce pas (TFM dynamique). Senseur : puissance dégagée par les réactions « de
    /// flamme » (δ_L > 0) au-dessus de 2 % du maximum du domaine (et d'un échauffement de 50 K/s), dilatée sur `flame_resolution` + 1 mailles
    /// pour couvrir la zone de préchauffage.
    pub(crate) fn update_flame_factor(&mut self, _dt: f64) {
        let n = self.nx * self.ny;
        let flames: Vec<usize> = (0..self.reg.reactions.len()).filter(|&j| self.reg.reactions[j].flame_thickness > 0.0).collect();
        let f_max = flames
            .iter()
            .map(|&j| self.flame_resolution * self.dx / self.reg.reactions[j].flame_thickness)
            .fold(1.0f64, f64::max);
        if !self.chemistry || flames.is_empty() || f_max <= 1.0 {
            self.flame_factor.iter_mut().for_each(|f| *f = 1.0);
            return;
        }
        let rx = &self.reg.reactions;
        let phases = &self.reg.phases;
        let np = phases.len();
        let mut omega = vec![0.0f64; n];
        for c in 0..n {
            if self.is_solid(c) {
                continue;
            }
            let cc = CellChem { phases, rx, m: (0..np).map(|k| self.mass[k][c]).collect(), e: self.e[c], f_eff: 1.0, scale: vec![1.0; rx.len()] };
            let t = self.temp[c];
            let mut w = 0.0;
            for &j in &flames {
                if rx[j].reactants.iter().any(|&(s, _)| self.mass[s][c] <= 1e-12) {
                    continue;
                }
                w += (cc.rate(j, t).max(0.0) * cc.heat(j, t)).abs();
            }
            omega[c] = w;
        }
        let wmax = omega.iter().fold(0.0f64, |a, &b| a.max(b));
        let mut mask: Vec<f64> = (0..n)
            .map(|c| if wmax > 0.0 && omega[c] > 0.02 * wmax && omega[c] > 50.0 * self.heat_capacity(c) { 1.0 } else { 0.0 })
            .collect();
        // dilatation (filtre max séparable) sur r mailles
        let r = self.flame_resolution.ceil() as usize + 1;
        let (nx, ny) = (self.nx, self.ny);
        let mut tmp = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                let (a, b) = (i.saturating_sub(r), (i + r).min(nx - 1));
                tmp[j * nx + i] = (a..=b).map(|ii| mask[j * nx + ii]).fold(0.0, f64::max);
            }
        }
        for j in 0..ny {
            for i in 0..nx {
                let (a, b) = (j.saturating_sub(r), (j + r).min(ny - 1));
                mask[j * nx + i] = (a..=b).map(|jj| tmp[jj * nx + i]).fold(0.0, f64::max);
            }
        }
        for c in 0..n {
            self.flame_factor[c] = if self.is_solid(c) { 1.0 } else { 1.0 + (f_max - 1.0) * mask[c] };
        }
    }

    // ================================================================ cinétique
    /// Réactions chimiques (APRÈS conduction et diffusion, AVANT le changement de phase). Voir l'en-tête du module.
    pub(crate) fn react_step(&mut self, dt: f64) {
        if self.reg.reactions.is_empty() || !self.chemistry {
            self.dt_chem = f64::INFINITY;
            self.heat_release.iter_mut().for_each(|h| *h = 0.0);
            return;
        }
        let n = self.nx * self.ny;
        let np = self.reg.phases.len();
        let nre = self.reg.reactions.len();
        let a2 = self.dx * self.dx;
        let phases = self.reg.phases.clone();
        let rx = self.reg.reactions.clone();
        let scale: Vec<f64> = rx
            .iter()
            .map(|r| {
                let (a, b) = r.mass_balance(&phases);
                if b > 0.0 { a / b } else { 1.0 }
            })
            .collect();
        if self.reacted.len() != nre {
            self.reacted = vec![0.0; nre];
        }
        let target = 0.5 * self.chem_volume_limit;
        let mut dt_next = f64::INFINITY;
        for c in 0..n {
            self.heat_release[c] = 0.0;
            if self.is_solid(c) || self.pinned(c) {
                continue;
            }
            // tri rapide : une réaction peut-elle avoir lieu ?
            let possible = rx.iter().any(|r| {
                r.reactants.iter().all(|&(s, _)| self.mass[s][c] > 1e-12) || (r.reversible && r.products.iter().all(|&(s, _)| self.mass[s][c] > 1e-12))
            });
            if !possible {
                continue;
            }
            let mut cc = CellChem { phases: &phases, rx: &rx, m: (0..np).map(|k| self.mass[k][c]).collect(), e: self.e[c], f_eff: self.flame_factor[c].max(1.0), scale: scale.clone() };
            let t0 = cc.temp();
            // cellule « gelée » ? (aucune réaction n'avancerait de 1e-9 de sa masse disponible pendant dt)
            let active = (0..nre).any(|j| {
                let (lo, hi) = cc.bounds(j);
                let r = cc.rate(j, t0);
                (r > 0.0 && r * dt > 1e-9 * hi) || (r < 0.0 && -r * dt > 1e-9 * (-lo))
            });
            if !active {
                continue;
            }
            let p = self.p[c];
            let v0 = cc.volume(p, t0);
            let m_old = cc.m.clone();
            let mut xi_tot = vec![0.0; nre];
            let mut tl = 0.0;
            let mut nsub = 0;
            let mut stopped = false;
            let mut max_h = f64::INFINITY;
            while tl < dt * (1.0 - 1e-12) && nsub < MAX_CHEM_SUBSTEPS {
                nsub += 1;
                let t = cc.temp();
                let cap = cc.cap(t).max(1e-30);
                let r0: Vec<f64> = (0..nre).map(|j| cc.rate(j, t)).collect();
                // pas : échauffement ≤ 25 K et consommation ≤ 30 % (réactions irréversibles ; les réversibles sont implicites)
                let mut h = (dt - tl).min(max_h);
                let mut dtdt = 0.0;
                let mut cons = vec![0.0; np];
                for j in 0..nre {
                    if rx[j].reversible {
                        continue;
                    }
                    dtdt += r0[j] * cc.heat(j, t) / cap;
                    for &(s, nu) in &rx[j].reactants {
                        cons[s] += nu * phases[s].molar_mass * r0[j].max(0.0);
                    }
                }
                if dtdt.abs() > 0.0 {
                    h = h.min(25.0 / dtdt.abs());
                }
                for s in 0..np {
                    if cons[s] > 0.0 && cc.m[s] > 1e-12 {
                        h = h.min(0.3 * cc.m[s] / cons[s]);
                    }
                }
                h = h.max(dt * 1e-9);
                let m_sub = cc.m.clone();
                let xi_sub = xi_tot.clone();
                for j in 0..nre {
                    let (lo, hi) = cc.bounds(j);
                    if hi <= 0.0 && lo >= 0.0 {
                        continue;
                    }
                    let xi = if rx[j].reversible {
                        // Euler implicite : ξ = h·r(ξ), Newton avec dérivée numérique (T recalculée : rétroaction thermique comprise)
                        let base = cc.m.clone();
                        let eval = |cc: &mut CellChem, x: f64| -> f64 {
                            cc.m.copy_from_slice(&base);
                            cc.apply(j, x);
                            let tt = cc.temp();
                            cc.rate(j, tt)
                        };
                        let mut x = (h * r0[j]).clamp(0.999 * lo, 0.999 * hi);
                        for _ in 0..6 {
                            let rx_ = eval(&mut cc, x);
                            let g = x - h * rx_;
                            let d = 1e-7 * (hi - lo).abs().max(1e-12) + 1e-14;
                            let xd = if x + d < hi { x + d } else { x - d };
                            let rd = eval(&mut cc, xd);
                            let dr = (rd - rx_) / (xd - x);
                            let jac = 1.0 - h * dr;
                            let xn = if jac > 0.05 { x - g / jac } else { h * rx_ };
                            let xn = xn.clamp(0.999 * lo, 0.999 * hi);
                            let conv = (xn - x).abs() <= 1e-9 * (hi - lo).abs().max(1e-30);
                            x = xn;
                            if conv {
                                break;
                            }
                        }
                        cc.m.copy_from_slice(&base);
                        x
                    } else {
                        let tt = cc.temp();
                        (h * cc.rate(j, tt)).clamp(0.0, 0.999 * hi)
                    };
                    if xi != 0.0 {
                        cc.apply(j, xi);
                        xi_tot[j] += xi;
                    }
                }
                // sous-pas rejeté s'il échauffe de plus de 60 K (réactions réversibles implicites très exothermiques loin de l'équilibre)
                let dtemp = (cc.temp() - t).abs();
                if dtemp > 60.0 && h > dt * 1e-7 && nsub < MAX_CHEM_SUBSTEPS {
                    cc.m.copy_from_slice(&m_sub);
                    xi_tot.copy_from_slice(&xi_sub);
                    let h2 = h * (0.5 * 60.0 / dtemp).min(0.5);
                    // refait le sous-pas avec h2 : on borne la durée restante par h2 au prochain tour
                    max_h = h2;
                    continue;
                }
                max_h = f64::INFINITY;
                tl += h;
                // couplage avec l'écoulement : expansion maximale permise pendant le pas
                let t1 = cc.temp();
                if cc.volume(p, t1) - v0 > self.chem_volume_limit * v0 {
                    stopped = true;
                    break;
                }
            }
            // bilan : masses, chaleur dégagée (= baisse de l'énergie chimique Σ m e0, e inchangée), avancements
            let mut q = 0.0;
            for k in 0..np {
                q -= (cc.m[k] - m_old[k]) * phases[k].e0;
                self.mass[k][c] = cc.m[k];
            }
            self.temp[c] = cc.temp();
            self.heat_release[c] = q / dt;
            self.chem_heat += q * a2;
            for j in 0..nre {
                self.reacted[j] += xi_tot[j] * a2;
            }
            // pas de temps suivant : expansion visée `target` par pas
            let dv = (cc.volume(p, self.temp[c]) - v0).abs() / v0.max(1e-300);
            if dv > 1e-3 {
                let tau = if stopped { tl } else { dt };
                dt_next = dt_next.min(target * tau / dv);
            }
        }
        self.dt_chem = if dt_next.is_finite() { dt_next.max(1e-8) } else { f64::INFINITY };
    }

    // ================================================================ rayonnement
    /// Pertes radiatives des gaz chauds (optiquement mince) : e −= dt · 4σ κ_P (T⁴ − T_amb⁴) · α_gaz, κ_P = Σ p_k[atm] a_P,k(T).
    pub(crate) fn radiate(&mut self, dt: f64) {
        if !self.radiation {
            return;
        }
        let np = self.reg.phases.len();
        let rads: Vec<usize> = (0..np).filter(|&k| self.reg.phases[k].radiator != Radiator::None && self.reg.phases[k].is_gas()).collect();
        if rads.is_empty() {
            return;
        }
        let ta = self.t_ambient;
        let a2 = self.dx * self.dx;
        for c in 0..self.nx * self.ny {
            if self.is_solid(c) || self.pinned(c) {
                continue;
            }
            let t = self.temp[c];
            if (t - ta).abs() < 50.0 {
                continue;
            }
            let ngas: f64 = (0..np).filter(|&k| self.reg.phases[k].is_gas()).map(|k| self.mass[k][c].max(0.0) / self.reg.phases[k].molar_mass).sum();
            if ngas <= 0.0 {
                continue;
            }
            let a_gas: f64 = (0..np).filter(|&k| self.reg.phases[k].is_gas()).map(|k| self.alpha[k][c]).sum();
            let p_atm = self.p[c] / 101325.0;
            let kappa: f64 = rads
                .iter()
                .map(|&k| (self.mass[k][c].max(0.0) / self.reg.phases[k].molar_mass / ngas) * p_atm * self.reg.phases[k].radiator.planck_mean(t))
                .sum();
            let q = 4.0 * SIGMA_SB * kappa * (t.powi(4) - ta.powi(4)) * a_gas;
            // jamais plus de la moitié de l'écart à l'ambiante en un pas (stabilité)
            let lim = 0.5 * self.heat_capacity(c) * (t - ta).abs();
            let de = (q * dt).clamp(-lim, lim);
            self.e[c] -= de;
            self.radiated += de * a2;
            self.temp[c] = self.temp_from_energy(c);
        }
    }
}
