//! Température PAR GROUPE de fluides (non-équilibre thermique entre phases non miscibles).
//!
//! Un groupe = un groupe miscible du registre (`Registry::miscible_groups`) : les membres d'un groupe sont mêlés à l'échelle
//! moléculaire (air = O2 + N2, eau + alcool…) et partagent une température ; deux phases non miscibles (huile / eau / gaz) ont
//! chacune la sienne, même dans une cellule mixte. Une seule pression par cellule (équilibre mécanique).
//!
//! État : `eg[g][c]` énergie interne ABSOLUE du groupe g dans la cellule c (J/m³ de cellule, Σ_k∈g m_k e_k(T_g)), `tg[g][c]` sa
//! température. `e[c]` = Σ_g eg[g][c] et `temp[c]` (température d'un mélange de même énergie, pour l'affichage et les bilans)
//! sont recalculés par `refresh`. Les cellules solides gardent `e`, `temp` (eg = 0).
//!
//! Échange dans une cellule mixte : chaque groupe échange avec l'interface par conduction à travers sa demi-épaisseur dans la
//! cellule, G_g = λ · 2 k_g / (α_g dx²) (W/(m³·K)) ; résolution implicite exacte (énergie conservée à l'arrondi près) :
//! T_i = Σ G_g T_g/(1+a_g) / Σ G_g/(1+a_g), T_g' = (T_g + a_g T_i)/(1 + a_g), a_g = dt G_g / C_g. λ = ∞ redonne le modèle à une
//! température (équilibre instantané). Une trace (α → 0) a une constante de temps α² dx²/(2D) → 0 : elle est à la température de
//! l'interface. Sans cet échange fini, de l'huile à 2000 °C entrant dans une cellule d'air ou d'eau la portait d'un coup à ~1900 °C.

use super::multiphase::{temp_from_masses, MultiphaseSim};

/// part de capacité calorifique sous laquelle un groupe est une trace : sa température est alignée sur celle du reste de la cellule
/// (l'écart d'énergie va au groupe principal ; sans cela T = E/C d'une masse de 1e-15 kg/m³ est du bruit d'arrondi)
pub(crate) const TRACE_CAP: f64 = 1e-9;

impl MultiphaseSim {
    pub fn ngroups(&self) -> usize {
        self.groups.len()
    }
    /// température de la phase k dans la cellule c (celle de son groupe)
    #[inline]
    pub fn phase_temp(&self, k: usize, c: usize) -> f64 {
        self.tg[self.group_of[k]][c]
    }
    pub(crate) fn group_mass(&self, g: usize, c: usize) -> f64 {
        self.groups[g].iter().map(|&k| self.mass[k][c].max(0.0)).sum()
    }
    /// ∂E_g/∂T du groupe g de la cellule c à la température t (J/(m³·K))
    pub(crate) fn group_cap_at(&self, g: usize, c: usize, t: f64) -> f64 {
        self.groups[g].iter().map(|&k| self.mass[k][c].max(0.0) * self.reg.phases[k].c_e_at(t)).sum()
    }
    /// énergie qu'aurait le groupe g de c à la température t (masses courantes)
    pub(crate) fn group_energy_at(&self, g: usize, c: usize, t: f64) -> f64 {
        self.groups[g].iter().map(|&k| self.mass[k][c].max(0.0) * self.reg.phases[k].e_int(t)).sum()
    }
    /// température du groupe g de c d'après son énergie eg (NaN si le groupe est absent)
    #[allow(dead_code)]
    pub(crate) fn group_temp_calc(&self, g: usize, c: usize) -> f64 {
        if self.group_mass(g, c) <= 0.0 {
            return f64::NAN;
        }
        let ph = &self.reg.phases;
        let (mut cap, mut slope, mut off) = (0.0, 0.0, 0.0);
        for &k in &self.groups[g] {
            let m = self.mass[k][c].max(0.0);
            cap += m * ph[k].c_e();
            slope += m * ph[k].cp_slope;
            off += m * ph[k].e0;
        }
        super::multiphase::temp_from_parts(cap, slope, off, self.eg[g][c])
    }
    /// conductivité du groupe g dans c (moyenne des membres pondérée par leur volume, à T_g)
    pub(crate) fn group_k(&self, g: usize, c: usize) -> f64 {
        let (mut a, mut kk) = (0.0, 0.0);
        let t = self.tg[g][c];
        for &k in &self.groups[g] {
            let al = self.alpha[k][c];
            if al > 0.0 {
                a += al;
                kk += al * self.reg.phases[k].k(t);
            }
        }
        if a > 0.0 { kk / a } else { 0.0 }
    }
    pub(crate) fn group_alpha(&self, g: usize, c: usize) -> f64 {
        self.groups[g].iter().map(|&k| self.alpha[k][c]).sum::<f64>().min(1.0)
    }

    /// Met TOUS les groupes de la cellule fluide c à la température t (masses courantes) ; e, temp et tg cohérents.
    pub fn set_cell_temperature(&mut self, c: usize, t: f64) {
        if self.is_solid(c) {
            let s = &self.reg.solids[self.solid[c] as usize];
            self.e[c] = s.rho_cp() * t;
            self.temp[c] = t;
            for g in 0..self.groups.len() {
                self.eg[g][c] = 0.0;
                self.tg[g][c] = t;
            }
            return;
        }
        let mut e = 0.0;
        for g in 0..self.groups.len() {
            let eg = self.group_energy_at(g, c, t);
            self.eg[g][c] = eg;
            self.tg[g][c] = t;
            e += eg;
        }
        self.e[c] = e;
        self.temp[c] = t;
    }

    /// Met le seul groupe g de la cellule fluide c à la température t (les autres gardent la leur) ; e, temp, tg cohérents.
    #[allow(dead_code)]
    pub fn set_group_temperature(&mut self, c: usize, g: usize, t: f64) {
        if self.is_solid(c) || g >= self.groups.len() {
            return;
        }
        self.eg[g][c] = self.group_energy_at(g, c, t);
        self.cell_sync(c);
    }

    /// Ajoute dm (kg/m³) de la phase k à la cellule c, à la température t (énergie comprise, dans le groupe de k).
    pub(crate) fn add_mass_at(&mut self, k: usize, c: usize, dm: f64, t: f64) {
        self.mass[k][c] += dm;
        self.eg[self.group_of[k]][c] += dm * self.reg.phases[k].e_int(t);
    }

    /// Recalcule tg (groupes présents ; absents : température de la cellule), traite les traces, puis e = Σ eg et temp.
    pub(crate) fn cell_sync(&mut self, c: usize) {
        if self.is_solid(c) {
            return;
        }
        let ng = self.groups.len();
        let mut egn = [0.0f64; super::phases::MAX_PHASES];
        let mut tgn = [0.0f64; super::phases::MAX_PHASES];
        let (e, tc) = self.sync_calc(c, &mut egn, &mut tgn, false);
        self.e[c] = e;
        self.temp[c] = tc;
        for g in 0..ng {
            self.eg[g][c] = egn[g];
            self.tg[g][c] = tgn[g];
        }
    }

    /// Calcul de `cell_sync` sans écrire : énergies de groupes corrigées (traces, groupes vides), températures, (e, temp).
    /// `adopt_e` (utilisé par `refresh`) : un écart entre `e[c]` et Σ eg vient d'une modification EXTÉRIEURE de `e` (outil, test,
    /// ancien code) ; il est réparti entre les groupes au prorata de leur capacité calorifique (chauffage uniforme de la cellule).
    /// Les processus internes modifient eg puis appellent `cell_sync`, qui fait de Σ eg la référence.
    pub(crate) fn sync_calc(&self, c: usize, egn: &mut [f64], tgn: &mut [f64], adopt_e: bool) -> (f64, f64) {
        let ng = self.groups.len();
        let np = self.reg.phases.len();
        let mut e = 0.0;
        for g in 0..ng {
            egn[g] = self.eg[g][c];
            e += egn[g];
        }
        if adopt_e {
            let d = self.e[c] - e;
            if d != 0.0 && d.abs() > 1e-13 * e.abs().max(1e-300) {
                let t0 = temp_from_masses(&self.reg.phases, (0..np).map(|k| self.mass[k][c].max(0.0)), e);
                let caps: Vec<f64> = (0..ng).map(|g| self.group_cap_at(g, c, if self.group_mass(g, c) > 0.0 { self.tg[g][c] } else { t0 })).collect();
                let csum: f64 = caps.iter().sum();
                if csum > 0.0 {
                    for g in 0..ng {
                        egn[g] += d * caps[g] / csum;
                    }
                    e = self.e[c];
                }
            }
        }
        let tc = temp_from_masses(&self.reg.phases, (0..np).map(|k| self.mass[k][c].max(0.0)), e);
        if ng == 1 {
            tgn[0] = tc;
            return (e, tc);
        }
        // groupe principal (plus grande capacité) : reçoit l'énergie des groupes vides et l'écart des traces
        let mut cap = [0.0f64; super::phases::MAX_PHASES];
        let mut ctot = 0.0;
        let mut main = 0;
        for g in 0..ng {
            cap[g] = self.group_cap_at(g, c, tc);
            ctot += cap[g];
            if cap[g] > cap[main] {
                main = g;
            }
        }
        for g in 0..ng {
            if g == main {
                continue;
            }
            if self.group_mass(g, c) <= 0.0 {
                egn[main] += egn[g];
                egn[g] = 0.0;
            } else if cap[g] < TRACE_CAP * ctot {
                let target = self.group_energy_at(g, c, tc);
                egn[main] += egn[g] - target;
                egn[g] = target;
            }
        }
        for g in 0..ng {
            tgn[g] = if self.group_mass(g, c) > 0.0 {
                let ph = &self.reg.phases;
                let (mut cp, mut sl, mut off) = (0.0, 0.0, 0.0);
                for &k in &self.groups[g] {
                    let m = self.mass[k][c].max(0.0);
                    cp += m * ph[k].c_e();
                    sl += m * ph[k].cp_slope;
                    off += m * ph[k].e0;
                }
                super::multiphase::temp_from_parts(cp, sl, off, egn[g])
            } else {
                tc
            };
        }
        (e, tc)
    }

    /// Conductance d'échange G_g (W/(m³·K)) du groupe g avec l'interface de sa cellule (voir l'en-tête du module). Infinie pour une
    /// vapeur au contact de SON liquide (`vapor_locked`).
    pub(crate) fn exchange_conductance(&self, g: usize, c: usize) -> f64 {
        if self.phase_heat_exchange > 0.0 && self.vapor_locked(g, c) {
            return f64::INFINITY;
        }
        let a = self.group_alpha(g, c).max(1e-6);
        self.phase_heat_exchange * 2.0 * self.group_k(g, c) / (a * self.dx * self.dx)
    }

    /// Le groupe g est-il surtout (> 50 % en masse) la vapeur d'un liquide présent dans la cellule ? Une telle vapeur est maintenue à
    /// la température de l'interface par l'évaporation / la condensation, bien plus rapides que la conduction : on la couple
    /// parfaitement à son liquide (une bulle de vapeur se comprime et se détend presque isotherme, comme dans le modèle à une
    /// température ; sinon elle s'échaufferait adiabatiquement sans se condenser).
    pub(crate) fn vapor_locked(&self, g: usize, c: usize) -> bool {
        let mg = self.group_mass(g, c);
        if mg <= 0.0 {
            return false;
        }
        self.reg.changes.iter().any(|pc| {
            self.group_of[pc.vapor] == g && self.group_of[pc.liquid] != g && self.mass[pc.vapor][c] > 0.5 * mg && self.group_mass(self.group_of[pc.liquid], c) > 0.0
        })
    }

    /// Couplage thermique χ ∈ [0, 1] des groupes d'une cellule pendant dt : max_g a_g/(1 + a_g), a_g = dt·G_g/C_g (part de l'écart à
    /// l'interface résorbée en un pas par le groupe qui relaxe le plus vite). χ = 1 : les groupes suivent une température commune
    /// pendant le pas (bulle de vapeur dans l'eau, λ = ∞) ; χ = 0 : chacun se comprime adiabatiquement pour son compte (huile chaude
    /// et air froid). Sert à la fermeture de la projection et à la répartition du travail de pression, qui doivent être cohérentes.
    pub(crate) fn thermal_coupling(&self, c: usize, dt: f64) -> f64 {
        let ng = self.groups.len();
        if ng < 2 || self.is_solid(c) {
            return 1.0;
        }
        // les TRACES (capacité < TRACE_CAP de la cellule, alignées sur T par cell_sync) ne comptent pas : 1e-17 kg de vapeur a un
        // a_g démesuré et imposait χ = 1 à toute la cellule (fermeture, travail de pression et chaleur latente « à une température »),
        // selon que l'arrondi laisse ou non une telle trace (f64 / f32)
        let ctot: f64 = (0..ng).filter(|&g| self.group_mass(g, c) > 0.0).map(|g| self.group_cap_at(g, c, self.tg[g][c])).sum();
        let mut present = 0;
        let mut chi: f64 = 0.0;
        for g in 0..ng {
            if self.group_mass(g, c) <= 0.0 || self.group_cap_at(g, c, self.tg[g][c]) < TRACE_CAP * ctot {
                continue;
            }
            present += 1;
            if !self.phase_heat_exchange.is_finite() {
                chi = 1.0;
                continue;
            }
            let cg = self.group_cap_at(g, c, self.tg[g][c]).max(1e-300);
            let a = dt * self.exchange_conductance(g, c) / cg;
            chi = chi.max(if a.is_finite() { a / (1.0 + a) } else { 1.0 });
        }
        if present < 2 { 1.0 } else { chi }
    }

    /// Échange de chaleur entre les groupes d'une même cellule pendant dt (voir l'en-tête du module).
    pub(crate) fn exchange_heat(&mut self, dt: f64) {
        let ng = self.groups.len();
        if ng < 2 || self.phase_heat_exchange <= 0.0 {
            return;
        }
        let n = self.nx * self.ny;
        let lam = self.phase_heat_exchange;
        let mut gk = vec![0.0; ng];
        let mut ck = vec![0.0; ng];
        let mut tk = vec![0.0; ng];
        for c in 0..n {
            if self.is_solid(c) {
                continue;
            }
            let mut present = 0;
            for g in 0..ng {
                if self.group_mass(g, c) > 0.0 {
                    present += 1;
                }
            }
            if present < 2 {
                continue;
            }
            let (mut num, mut den) = (0.0, 0.0);
            let (mut ce, mut cs) = (0.0, 0.0);
            for g in 0..ng {
                ck[g] = 0.0;
                gk[g] = 0.0;
                if self.group_mass(g, c) <= 0.0 {
                    continue;
                }
                let t = self.tg[g][c];
                tk[g] = t;
                ck[g] = self.group_cap_at(g, c, t).max(1e-300);
                ce += ck[g] * t;
                cs += ck[g];
                if lam.is_finite() {
                    gk[g] = self.exchange_conductance(g, c);
                    // G/(1+a) avec a = dt G / C : → C/dt pour un couplage parfait (G = ∞)
                    let b = if gk[g].is_finite() { gk[g] / (1.0 + dt * gk[g] / ck[g]) } else { ck[g] / dt };
                    num += b * t;
                    den += b;
                }
            }
            let ti = if lam.is_finite() {
                if den <= 0.0 {
                    continue;
                }
                num / den
            } else {
                ce / cs
            };
            for g in 0..ng {
                if ck[g] <= 0.0 || self.group_mass(g, c) <= 0.0 {
                    continue;
                }
                let tn = if lam.is_finite() && gk[g].is_finite() {
                    let ag = dt * gk[g] / ck[g];
                    (tk[g] + ag * ti) / (1.0 + ag)
                } else {
                    ti
                };
                self.eg[g][c] += ck[g] * (tn - tk[g]);
                self.tg[g][c] = tn;
            }
        }
    }
}
