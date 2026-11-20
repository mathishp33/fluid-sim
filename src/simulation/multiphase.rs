//! Solveur multiphasique 2D (CPU, f64) : N gaz/liquides compressibles (EOS réelles) + solides, grille MAC,
//! interface PLIC conservative, pression implicite (Helmholtz, acoustique non limitante), tension de surface,
//! gravité, viscosité variable, énergie, ébullition / cavitation (Hardt–Wondra + Schnerr–Sauer).
//!
//! Conventions : j croît vers +y (HAUT) ; gravité par défaut (0, −9.81). u : (nx+1)×ny, v : nx×(ny+1).
//! Variables conservées par cellule : masses partielles m_k (kg/m³ de cellule) et énergie interne e (J/m³).
//! Les fractions volumiques α_k sont DÉRIVÉES : α_k = (m_k/ρ_k(p,T)) / Σ_j (m_j/ρ_j(p,T)).
//! Une seule pression et une seule température par cellule (équilibre mécanique et thermique des phases).

use super::mp_cells::{MpEffect, MpInlet, MpThermal, PinInfo, TpMix};
use super::phases::{PhaseKind, Registry};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bc {
    Wall,
    /// Bord ouvert à pression imposée (atmosphère). L'air/fluide entrant est la phase `ambient_phase` à `t_ambient`.
    Open { p: f64 },
}

pub struct MultiphaseSim {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub reg: Registry,
    pub gravity: [f64; 2],
    /// left, right, bottom, top
    pub bc: [Bc; 4],
    pub ambient_phase: usize,
    pub t_ambient: f64,

    pub mass: Vec<Vec<f64>>,  // [phase][cell]
    pub alpha: Vec<Vec<f64>>, // [phase][cell] (dérivé)
    pub e: Vec<f64>,          // énergie interne J/m³ (solide : ρ_s c_s T)
    pub temp: Vec<f64>,
    pub p: Vec<f64>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub solid: Vec<i32>, // −1 = fluide, sinon indice dans reg.solids
    pub rho_mix: Vec<f64>,
    pub mu_cell: Vec<f64>,
    pub k_cell: Vec<f64>,

    // numérique
    pub cfl: f64,
    /// advection de vitesse : false (défaut) = semi-lagrangien bicubique borné (dissipatif, ne crée pas d'énergie) ;
    /// true = MacCormack (moins diffusif mais CRÉE de l'énergie cinétique avec un rapport de densités eau/air de 1000 : déconseillé)
    pub maccormack: bool,
    pub cap_safety: f64, // facteur sur la limite capillaire de Brackbill
    pub dt_max: f64,
    pub surface_tension: bool,
    pub viscous: bool,
    pub conduction: bool,
    pub phase_change: bool,
    pub pressure_tol: f64,
    pub pressure_max_iter: usize,
    // changement de phase
    pub nucleation_superheat: f64, // K de surchauffe pour amorcer l'ébullition en volume
    pub nucleation_alpha: f64,     // fraction de vapeur créée à l'amorçage (germes)
    pub nuclei_density: f64,       // n0 (1/m³ de liquide) pour Schnerr–Sauer
    pub time: f64,
    pub last_dt: f64,
    pub last_pressure_iters: usize,
    /// masse (kg par m de profondeur) entrée par les bords ouverts, par phase : M_k(t) − M_k(0) = flux_k − (changements de phase)
    pub boundary_flux: Vec<f64>,
    /// masse transformée liquide→vapeur cumulée par paire (kg/m), pour les bilans
    pub transferred: Vec<f64>,
    pub(crate) step_count: usize,
    /// flux de masse total (toutes phases) à travers chaque face de cellule pendant le dernier advect_mass_energy (kg/m³ de cellule, + = vers +x / +y)
    pub(crate) mfx: Vec<f64>,
    pub(crate) mfy: Vec<f64>,
    pub(crate) m_tot_old: Vec<f64>,
    /// advection de la quantité de mouvement conservative, cohérente avec les flux de masse du VOF (défaut) ; false = semi-lagrangien
    pub conservative_momentum: bool,
    /// vitesse au-delà de laquelle un pas est jugé aberrant (m/s) — déclenche le retour arrière avec sous-pas
    pub speed_limit: f64,
    /// nombre de pas annulés puis refaits avec un dt plus petit (état invalide : NaN, T hors plage, masse négative, vitesse aberrante)
    pub rollbacks: usize,
    /// true si même le plus petit sous-pas reste invalide : l'état n'avance plus (la fenêtre se met en pause au lieu de passer en NaN)
    pub diverged: bool,

    // ---- cellules spéciales, solides thermiques, entrées (voir mp_cells.rs)
    /// effet de chaque cellule fluide (puits, émetteur, accélérateur, téléporteur)
    pub special: Vec<Option<MpEffect>>,
    /// comportement thermique de chaque cellule solide
    pub thermal: Vec<MpThermal>,
    /// entrée de bord par côté (gauche, droite, bas, haut) ; n'agit que si le bord est fermé (`Bc::Wall`)
    pub inlet: [Option<MpInlet>; 4],
    /// chaleur injectée par les solides chauffants depuis le début (J par m de profondeur)
    pub energy_input: f64,
    pub(crate) pin: Vec<Option<PinInfo>>,
    pub(crate) pf_u: Vec<f64>,
    pub(crate) pf_v: Vec<f64>,
    pub(crate) tp_mix: TpMix,

    // ---- réactions chimiques, flammes, rayonnement (voir mp_species.rs)
    /// composition du fluide ambiant (fractions VOLUMIQUES par phase, somme 1) ; vide = phase `ambient_phase` pure. Ex. air = 21 % O2 + 79 % N2
    pub ambient_mix: Vec<f64>,
    /// mélanges nommés (fractions volumiques) utilisables au pinceau (ex. « méthane–air stœchiométrique »)
    pub mixtures: Vec<(String, Vec<f64>)>,
    /// cinétique chimique active
    pub chemistry: bool,
    /// rayonnement des gaz chauds (H2O, CO2), modèle optiquement mince
    pub radiation: bool,
    /// épaisseur visée des flammes épaissies (cellules) : facteur F = flame_resolution·dx/δ_L
    pub flame_resolution: f64,
    /// variation relative de volume (à p, T de la cellule) que la chimie peut produire en un pas ; au-delà, l'intégration s'arrête et dt diminue
    pub chem_volume_limit: f64,
    /// facteur d'épaississement local F_eff (1 hors des flammes) : multiplie diffusion thermique et des espèces, divise la vitesse de réaction
    pub flame_factor: Vec<f64>,
    /// puissance chimique dégagée au dernier pas (W/m³, > 0 = exothermique)
    pub heat_release: Vec<f64>,
    /// chaleur de réaction dégagée depuis le début (J par m de profondeur) et énergie rayonnée (J/m)
    pub chem_heat: f64,
    pub radiated: f64,
    /// avancement cumulé de chaque réaction (mol par m de profondeur)
    pub reacted: Vec<f64>,
    /// pas de temps maximal imposé par la chimie au pas suivant (s)
    pub dt_chem: f64,
}

/// Température d'un mélange de masses `m` (kg/m³) d'énergie interne totale `e` (J/m³) : résout Σ m_k e_k(T) = e.
/// e_k(T) = c_k T + ½ s_k (T − 298.15)² + e0_k ⇒ trinôme du second degré en τ = T − 298.15 (forme stable sans annulation).
pub fn temp_from_masses(phases: &[super::phases::Phase], m: impl Iterator<Item = f64>, e: f64) -> f64 {
    let (mut cap, mut slope, mut off) = (0.0, 0.0, 0.0);
    for (k, mk) in m.enumerate() {
        let p = &phases[k];
        cap += mk * p.c_e();
        slope += mk * p.cp_slope;
        off += mk * p.e0;
    }
    let t0 = super::phases::T_STD;
    let rhs = e - off - cap * t0; // = C τ + ½ S τ²
    if slope.abs() <= 1e-12 * cap.abs().max(1e-300) {
        return t0 + rhs / cap.max(1e-30);
    }
    let disc = cap * cap + 2.0 * slope * rhs;
    if disc <= 0.0 {
        return t0 - cap / slope; // minimum de e(T) : énergie hors du domaine du modèle (ne se produit pas en pratique)
    }
    t0 + 2.0 * rhs / (cap + disc.sqrt()).max(1e-30)
}

impl MultiphaseSim {
    pub fn new(nx: usize, ny: usize, dx: f64, reg: Registry, ambient_phase: usize) -> Self {
        let n = nx * ny;
        let np = reg.phases.len();
        let nch = reg.changes.len();
        let nre = reg.reactions.len();
        MultiphaseSim {
            nx,
            ny,
            dx,
            reg,
            gravity: [0.0, -9.81],
            bc: [Bc::Wall; 4],
            ambient_phase,
            t_ambient: 293.15,
            mass: vec![vec![0.0; n]; np],
            alpha: vec![vec![0.0; n]; np],
            e: vec![0.0; n],
            temp: vec![293.15; n],
            p: vec![101325.0; n],
            u: vec![0.0; (nx + 1) * ny],
            v: vec![0.0; nx * (ny + 1)],
            solid: vec![-1; n],
            rho_mix: vec![1.0; n],
            mu_cell: vec![0.0; n],
            k_cell: vec![0.0; n],
            cfl: 0.4,
            maccormack: false,
            cap_safety: 0.3,
            dt_max: 1e-2,
            surface_tension: true,
            viscous: true,
            conduction: true,
            phase_change: true,
            pressure_tol: 1e-10,
            pressure_max_iter: 600,
            nucleation_superheat: 4.0,
            nucleation_alpha: 1e-3,
            nuclei_density: 1e13,
            time: 0.0,
            last_dt: 0.0,
            last_pressure_iters: 0,
            boundary_flux: vec![0.0; np],
            transferred: vec![0.0; nch],
            step_count: 0,
            mfx: vec![0.0; (nx + 1) * ny],
            mfy: vec![0.0; nx * (ny + 1)],
            m_tot_old: vec![0.0; n],
            conservative_momentum: true,
            speed_limit: 1000.0,
            rollbacks: 0,
            diverged: false,
            special: vec![None; n],
            thermal: vec![MpThermal::Passive; n],
            inlet: [None; 4],
            energy_input: 0.0,
            pin: vec![None; n],
            pf_u: vec![f64::NAN; (nx + 1) * ny],
            pf_v: vec![f64::NAN; nx * (ny + 1)],
            tp_mix: TpMix::new(),
            ambient_mix: vec![],
            mixtures: vec![],
            chemistry: true,
            radiation: nre > 0,
            flame_resolution: 6.0,
            chem_volume_limit: 0.5,
            flame_factor: vec![1.0; n],
            heat_release: vec![0.0; n],
            chem_heat: 0.0,
            radiated: 0.0,
            reacted: vec![0.0; nre],
            dt_chem: f64::INFINITY,
        }
    }

    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }
    #[inline]
    pub fn uidx(&self, i: usize, j: usize) -> usize {
        j * (self.nx + 1) + i
    }
    #[inline]
    pub fn vidx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }
    pub fn is_solid(&self, c: usize) -> bool {
        self.solid[c] >= 0
    }
    pub fn center(&self, i: usize, j: usize) -> (f64, f64) {
        ((i as f64 + 0.5) * self.dx, (j as f64 + 0.5) * self.dx)
    }

    // ---------------------------------------------------------------- initialisation
    /// Remplit les fractions volumiques par sur-échantillonnage 8×8 : `f(x,y)` donne la phase (None = laisse tel quel).
    /// À appeler sur un domaine vierge (α = 0) ou pour superposer ; puis `finalize_init`.
    pub fn paint_phase(&mut self, f: impl Fn(f64, f64) -> Option<usize>) {
        let s = 8usize;
        for j in 0..self.ny {
            for i in 0..self.nx {
                let c = self.idx(i, j);
                let mut cnt = vec![0.0; self.reg.phases.len()];
                let mut tot = 0.0;
                for b in 0..s {
                    for a in 0..s {
                        let x = (i as f64 + (a as f64 + 0.5) / s as f64) * self.dx;
                        let y = (j as f64 + (b as f64 + 0.5) / s as f64) * self.dx;
                        if let Some(k) = f(x, y) {
                            cnt[k] += 1.0;
                            tot += 1.0;
                        }
                    }
                }
                if tot > 0.0 {
                    let keep = 1.0 - tot / (s * s) as f64;
                    for k in 0..cnt.len() {
                        self.alpha[k][c] = self.alpha[k][c] * keep + cnt[k] / (s * s) as f64;
                    }
                }
            }
        }
    }
    /// Comme `paint_phase` avec un MÉLANGE : `f(x,y)` donne des fractions volumiques (somme 1) ou None.
    pub fn paint_mixture(&mut self, f: impl Fn(f64, f64) -> Option<Vec<f64>>) {
        let s = 8usize;
        let np = self.reg.phases.len();
        for j in 0..self.ny {
            for i in 0..self.nx {
                let c = self.idx(i, j);
                let mut cnt = vec![0.0; np];
                let mut tot = 0.0;
                for b in 0..s {
                    for a in 0..s {
                        let x = (i as f64 + (a as f64 + 0.5) / s as f64) * self.dx;
                        let y = (j as f64 + (b as f64 + 0.5) / s as f64) * self.dx;
                        if let Some(m) = f(x, y) {
                            for k in 0..np.min(m.len()) {
                                cnt[k] += m[k];
                            }
                            tot += 1.0;
                        }
                    }
                }
                if tot > 0.0 {
                    let keep = 1.0 - tot / (s * s) as f64;
                    for k in 0..np {
                        self.alpha[k][c] = self.alpha[k][c] * keep + cnt[k] / (s * s) as f64;
                    }
                }
            }
        }
    }
    /// Marque des cellules solides : `f(x,y)` → Some(indice solide). Les cellules solides sont étanches (u·n = 0).
    pub fn paint_solid(&mut self, f: impl Fn(f64, f64) -> Option<usize>) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                let (x, y) = self.center(i, j);
                if let Some(s) = f(x, y) {
                    let c = self.idx(i, j);
                    self.solid[c] = s as i32;
                }
            }
        }
    }

    /// Fixe T (fonction de la position), pose une pression hydrostatique (colonnes depuis le bord supérieur), calcule m_k et e.
    pub fn finalize_init(&mut self, t_fn: impl Fn(f64, f64) -> f64) {
        let (nx, ny) = (self.nx, self.ny);
        for j in 0..ny {
            for i in 0..nx {
                let c = self.idx(i, j);
                let (x, y) = self.center(i, j);
                self.temp[c] = t_fn(x, y);
                // fractions → somme 1 (la phase ambiante complète le vide)
                if !self.is_solid(c) {
                    let s: f64 = (0..self.reg.phases.len()).map(|k| self.alpha[k][c]).sum();
                    if s < 1.0 - 1e-12 {
                        let amb = self.ambient_alpha();
                        for k in 0..self.reg.phases.len() {
                            self.alpha[k][c] += (1.0 - s) * amb[k];
                        }
                    } else if s > 1.0 + 1e-12 {
                        for k in 0..self.reg.phases.len() {
                            self.alpha[k][c] /= s;
                        }
                    }
                }
            }
        }
        let p_top = match self.bc[3] {
            Bc::Open { p } => p,
            _ => 101325.0,
        };
        let g = -self.gravity[1];
        for i in 0..nx {
            let mut p_above = p_top;
            let mut rho_above = 0.0;
            for j in (0..ny).rev() {
                let c = self.idx(i, j);
                if self.is_solid(c) {
                    self.p[c] = p_top;
                    p_above = p_top;
                    rho_above = 0.0;
                    continue;
                }
                let t = self.temp[c];
                let rho_of = |sim: &Self, p: f64| -> f64 { (0..sim.reg.phases.len()).map(|k| sim.alpha[k][c] * sim.reg.phases[k].rho(p, t)).sum() };
                let mut pc = p_above;
                for _ in 0..8 {
                    let rc = rho_of(self, pc);
                    pc = if j == ny - 1 {
                        p_top + g * 0.5 * self.dx * rc
                    } else {
                        p_above + g * self.dx * 0.5 * (rho_above + rc)
                    };
                }
                self.p[c] = pc;
                p_above = pc;
                rho_above = rho_of(self, pc);
            }
        }
        for c in 0..nx * ny {
            if self.is_solid(c) {
                let s = &self.reg.solids[self.solid[c] as usize];
                self.e[c] = s.rho_cp() * self.temp[c];
                for k in 0..self.reg.phases.len() {
                    self.mass[k][c] = 0.0;
                    self.alpha[k][c] = 0.0;
                }
                continue;
            }
            let (p, t) = (self.p[c], self.temp[c]);
            let mut e = 0.0;
            for k in 0..self.reg.phases.len() {
                let m = self.alpha[k][c] * self.reg.phases[k].rho(p, t);
                self.mass[k][c] = m;
                e += m * self.reg.phases[k].e_int(t);
            }
            self.e[c] = e;
        }
        self.refresh();
    }

    // ---------------------------------------------------------------- propriétés dérivées
    /// Σ m_k c_k (J/(m³·K)) d'une cellule (solide : ρ_s c_s)
    /// ∂e/∂T à la température courante de la cellule (J/(m³·K)) (solide : ρ_s c_s)
    pub fn heat_capacity(&self, c: usize) -> f64 {
        if self.is_solid(c) {
            return self.reg.solids[self.solid[c] as usize].rho_cp();
        }
        let t = self.temp[c];
        (0..self.reg.phases.len()).map(|k| self.mass[k][c] * self.reg.phases[k].c_e_at(t)).sum()
    }
    pub fn temp_from_energy(&self, c: usize) -> f64 {
        if self.is_solid(c) {
            return self.e[c] / self.reg.solids[self.solid[c] as usize].rho_cp();
        }
        let np = self.reg.phases.len();
        temp_from_masses(&self.reg.phases, (0..np).map(|k| self.mass[k][c]), self.e[c])
    }
    /// Recalcule T (depuis e), α (depuis m, p, T), ρ_mix, μ, k de chaque cellule.
    pub fn refresh(&mut self) {
        let np = self.reg.phases.len();
        let mut vk = vec![0.0f64; np];
        for c in 0..self.nx * self.ny {
            if self.is_solid(c) {
                let s = &self.reg.solids[self.solid[c] as usize];
                self.temp[c] = self.e[c] / s.rho_cp();
                self.rho_mix[c] = s.rho;
                self.k_cell[c] = s.conductivity;
                self.mu_cell[c] = 0.0;
                continue;
            }
            let t = self.temp_from_energy(c);
            self.temp[c] = t;
            let p = self.p[c];
            let mut vol = 0.0;
            for k in 0..np {
                vk[k] = self.mass[k][c].max(0.0) / self.reg.phases[k].rho(p, t);
                vol += vk[k];
            }
            let (mut rho, mut mu, mut kc) = (0.0, 0.0, 0.0);
            for k in 0..np {
                let a = vk[k] / vol.max(1e-300);
                self.alpha[k][c] = a;
                rho += self.mass[k][c];
                mu += a * self.reg.phases[k].mu(t);
                kc += a * self.reg.phases[k].k(t);
            }
            self.rho_mix[c] = rho / vol.max(1e-300);
            self.mu_cell[c] = mu;
            self.k_cell[c] = kc;
        }
    }

    // ---------------------------------------------------------------- diagnostics
    /// Masse totale par phase (kg par m de profondeur)
    pub fn total_mass(&self, k: usize) -> f64 {
        let a = self.dx * self.dx;
        (0..self.nx * self.ny).map(|c| self.mass[k][c]).sum::<f64>() * a
    }
    pub fn total_mass_all(&self) -> f64 {
        (0..self.reg.phases.len()).map(|k| self.total_mass(k)).sum()
    }
    pub fn total_internal_energy(&self) -> f64 {
        self.e.iter().sum::<f64>() * self.dx * self.dx
    }
    /// Surface occupée par la phase k (m²)
    pub fn phase_area(&self, k: usize) -> f64 {
        (0..self.nx * self.ny).map(|c| if self.is_solid(c) { 0.0 } else { self.alpha[k][c] }).sum::<f64>() * self.dx * self.dx
    }
    pub fn kinetic_energy(&self) -> f64 {
        let mut ke = 0.0;
        for j in 0..self.ny {
            for i in 0..=self.nx {
                let rho = self.face_rho_x(i, j);
                ke += 0.5 * rho * self.u[self.uidx(i, j)].powi(2);
            }
        }
        for j in 0..=self.ny {
            for i in 0..self.nx {
                let rho = self.face_rho_y(i, j);
                ke += 0.5 * rho * self.v[self.vidx(i, j)].powi(2);
            }
        }
        ke * self.dx * self.dx
    }
    /// Masse volumique à la face verticale (i−½, j) : moyenne arithmétique des deux cellules (bord : la cellule seule)
    pub fn face_rho_x(&self, i: usize, j: usize) -> f64 {
        if i == 0 {
            self.rho_mix[self.idx(0, j)]
        } else if i == self.nx {
            self.rho_mix[self.idx(self.nx - 1, j)]
        } else {
            0.5 * (self.rho_mix[self.idx(i - 1, j)] + self.rho_mix[self.idx(i, j)])
        }
    }
    pub fn face_rho_y(&self, i: usize, j: usize) -> f64 {
        if j == 0 {
            self.rho_mix[self.idx(i, 0)]
        } else if j == self.ny {
            self.rho_mix[self.idx(i, self.ny - 1)]
        } else {
            0.5 * (self.rho_mix[self.idx(i, j - 1)] + self.rho_mix[self.idx(i, j)])
        }
    }
    pub fn max_speed(&self) -> f64 {
        let mu = self.u.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let mv = self.v.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        mu.max(mv)
    }
    /// fractions volumiques du fluide ambiant (une seule phase si `ambient_mix` est vide)
    pub fn ambient_alpha(&self) -> Vec<f64> {
        let np = self.reg.phases.len();
        if self.ambient_mix.len() == np && self.ambient_mix.iter().sum::<f64>() > 0.0 {
            let s: f64 = self.ambient_mix.iter().sum();
            self.ambient_mix.iter().map(|a| a / s).collect()
        } else {
            (0..np).map(|k| if k == self.ambient_phase { 1.0 } else { 0.0 }).collect()
        }
    }
    pub fn liquid_phases(&self) -> Vec<usize> {
        (0..self.reg.phases.len()).filter(|&k| self.reg.phases[k].kind == PhaseKind::Liquid).collect()
    }
}
