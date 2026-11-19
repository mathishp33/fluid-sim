//! Énergie et changement de phase.
//!  * Conduction explicite (k harmonique aux liens, sous-pas stables), fluide ↔ solide inclus.
//!  * Ébullition / condensation : régime résolu (interface) = Hardt–Wondra : la cellule d'interface est ancrée à T_sat(p) et
//!    le taux de transfert est celui qui absorbe la chaleur apportée (dm = C·(T−T_sat)/L(T)), limité par la masse disponible ;
//!    régime sous-maille (germes dans le liquide) = Schnerr–Sauer, plafonné par l'énergie disponible.
//!  * Cavitation : p < p_sat(T) ⇔ T > T_sat(p). Même machinerie (germes créés si surchauffe/dépression > seuil).
//!  * Travail de pression −p ∇·u (appliqué APRÈS la projection) : c'est lui qui, avec les décalages d'énergie e0, restitue
//!    la chaleur latente exacte L(T) (refroidissement du liquide qui bout).

use super::multiphase::MultiphaseSim;

impl MultiphaseSim {
    pub(crate) fn conduct(&mut self, dt: f64) {
        if !self.conduction {
            return;
        }
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let n = nx * ny;
        let mut amax = 0.0f64;
        for c in 0..n {
            let cap = self.heat_capacity(c);
            if cap > 0.0 {
                amax = amax.max(self.k_cell[c] / cap);
            }
        }
        if amax <= 0.0 {
            return;
        }
        let nsub = ((dt * amax / (0.2 * dx * dx)).ceil() as usize).clamp(1, 2000);
        let ds = dt / nsub as f64;
        let cap: Vec<f64> = (0..n).map(|c| self.heat_capacity(c)).collect();
        let off: Vec<f64> = (0..n)
            .map(|c| if self.is_solid(c) { 0.0 } else { (0..self.reg.phases.len()).map(|k| self.mass[k][c] * self.reg.phases[k].e0).sum() })
            .collect();
        let mut t: Vec<f64> = (0..n).map(|c| (self.e[c] - off[c]) / cap[c].max(1e-30)).collect();
        let iso: Vec<usize> = (0..n).filter(|&c| self.is_solid(c) && matches!(self.thermal[c], super::mp_cells::MpThermal::Isothermal(_))).collect();
        for _ in 0..nsub {
            let mut de = vec![0.0; n];
            for j in 0..ny {
                for i in 0..nx {
                    let c = j * nx + i;
                    for (di, dj) in [(1usize, 0usize), (0, 1)] {
                        let (ii, jj) = (i + di, j + dj);
                        if ii >= nx || jj >= ny {
                            continue;
                        }
                        let b = jj * nx + ii;
                        let (ka, kb) = (self.k_cell[c], self.k_cell[b]);
                        let kl = if ka + kb > 0.0 { 2.0 * ka * kb / (ka + kb) } else { 0.0 };
                        let q = kl * (t[b] - t[c]) / (dx * dx); // W/m³ vers c
                        de[c] += q;
                        de[b] -= q;
                    }
                }
            }
            for c in 0..n {
                self.e[c] += ds * de[c];
                t[c] += ds * de[c] / cap[c].max(1e-30);
            }
            // solides isothermes : température imposée à chaque sous-pas (ils sont des sources / puits de chaleur idéaux)
            for c in iso.iter().copied() {
                if let super::mp_cells::MpThermal::Isothermal(tc) = self.thermal[c] {
                    t[c] = tc;
                    self.e[c] = cap[c] * tc;
                }
            }
        }
        for c in 0..n {
            self.temp[c] = t[c];
        }
    }

    /// Transfert de masse liquide → vapeur de dm (kg/m³ de cellule, dm<0 : condensation). L'énergie interne est inchangée.
    fn transfer(&mut self, c: usize, liq: usize, vap: usize, dm: f64) {
        self.mass[liq][c] -= dm;
        self.mass[vap][c] += dm;
    }

    pub(crate) fn phase_change_step(&mut self, dt: f64) {
        if !self.phase_change {
            return;
        }
        let n = self.nx * self.ny;
        for (pi, pc) in self.reg.changes.clone().iter().enumerate() {
            let (liq, vap) = (pc.liquid, pc.vapor);
            let (pl, pv) = (&self.reg.phases[liq].clone(), &self.reg.phases[vap].clone());
            for c in 0..n {
                if self.is_solid(c) || self.pinned(c) {
                    continue;
                }
                let (ml, mv) = (self.mass[liq][c], self.mass[vap][c]);
                let (t, p) = (self.temp[c], self.p[c]);
                let al = self.alpha[liq][c];
                let av = self.alpha[vap][c];
                let a_other_gas: f64 = (0..self.reg.phases.len()).filter(|&k| k != liq && k != vap && self.reg.phases[k].is_gas()).map(|k| self.alpha[k][c]).sum();
                // candidats : liquide présent (ébullition/cavitation) ou vapeur + liquide (condensation)
                if ml <= 1e-12 && mv <= 1e-12 {
                    continue;
                }
                let tsat = pc.t_sat(p);
                let cap = self.heat_capacity(c);
                let latent = pc.latent(t.min(tsat.max(t)));
                let rho_v = pv.rho(p, t);
                let mut dm = 0.0;
                if t > tsat && ml > 1e-12 {
                    let dm_e = (cap * (t - tsat) / latent).min(ml);
                    let gas = av + a_other_gas;
                    let resolved = gas > 0.02 || (al < 0.98 && gas > 0.0 && a_other_gas > 0.0);
                    if resolved && al > 1e-6 {
                        // interface résolue (Hardt–Wondra) : ancrage à T_sat
                        dm = dm_e;
                    } else if av >= 1e-7 && a_other_gas < 1e-9 {
                        // germes sous-maille : Schnerr–Sauer (croissance), plafonné par l'énergie disponible
                        let r_b = (av / (1.0 - av).max(1e-9) * 3.0 / (4.0 * std::f64::consts::PI * self.nuclei_density)).cbrt();
                        let ps = pc.p_sat(t);
                        let rho_m = self.rho_mix[c];
                        let rate = rho_v * pl.rho(p, t) / rho_m * av * (1.0 - av) * 3.0 / r_b.max(1e-9) * (2.0 / 3.0 * (ps - p).max(0.0) / pl.rho(p, t)).sqrt();
                        // Pour une surchauffe SANS dépression (ébullition en volume) le taux inertiel est nul : ancrage thermique
                        let thermal = if ps - p <= 0.0 { dm_e } else { 0.0 };
                        dm = (rate * dt).max(thermal).min(dm_e);
                    } else if av < 1e-7 && a_other_gas < 1e-9 && t - tsat > self.nucleation_superheat {
                        // création de germes (nucléation) : fraction nucleation_alpha de vapeur
                        dm = (self.nucleation_alpha * rho_v).min(dm_e);
                    }
                } else if t < tsat && mv > 1e-12 && al > 1e-6 {
                    // condensation : interface résolue ancrée à T_sat ; sous-maille par Schnerr–Sauer
                    let dm_e = (cap * (tsat - t) / latent).min(mv);
                    let gas = av + a_other_gas;
                    if gas > 0.02 || a_other_gas > 0.0 {
                        dm = -dm_e;
                    } else if av > 1e-7 {
                        let r_b = (av / (1.0 - av).max(1e-9) * 3.0 / (4.0 * std::f64::consts::PI * self.nuclei_density)).cbrt();
                        let ps = pc.p_sat(t);
                        let rho_m = self.rho_mix[c];
                        let rate = rho_v * pl.rho(p, t) / rho_m * av * (1.0 - av) * 3.0 / r_b.max(1e-9) * (2.0 / 3.0 * (p - ps).max(0.0) / pl.rho(p, t)).sqrt();
                        dm = -(rate * dt).max(dm_e * 0.0).min(dm_e).min(mv);
                    }
                }
                if dm != 0.0 {
                    // couplage implicite local : on ne crée/détruit pas plus de vapeur que ce qui ramène la cellule à p = p_sat(T)
                    // (sinon un pas explicite dépasse l'équilibre : 1 kg d'eau → 1600 m³ de vapeur)
                    let ps = pc.p_sat(t);
                    let (vv, _, _) = self.cell_closure(c, ps, t);
                    let denom = 1.0 / pv.rho(ps, t) - 1.0 / pl.rho(ps, t);
                    if dm > 0.0 {
                        dm = dm.min(((1.0 - vv) / denom).max(0.0));
                    } else {
                        dm = -((-dm).min(((vv - 1.0) / denom).max(0.0)));
                    }
                }
                if dm != 0.0 {
                    self.transfer(c, liq, vap, dm);
                    self.transferred[pi] += dm * self.dx * self.dx;
                }
            }
        }
    }

    /// Travail de pression : e −= dt · p · ∇·u (u^{n+1}, p^{n+1})
    pub(crate) fn pressure_work(&mut self, dt: f64) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                let c = self.idx(i, j);
                if self.is_solid(c) || self.pinned(c) {
                    continue;
                }
                // dt·∇·u = variation relative de volume du pas : au-delà de ±50 % la cellule est quasi vide / en cavitation violente,
                // où p·∇·u n'a plus de sens physique (et divise par une capacité calorifique qui tend vers 0)
                let d = (dt * self.divergence_of(&self.u, &self.v, i, j)).clamp(-0.5, 0.5) / dt;
                self.e[c] -= dt * self.p[c].max(0.0) * d;
            }
        }
    }
}
