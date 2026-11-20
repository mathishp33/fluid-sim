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
        // flammes épaissies (TFM) : la diffusivité thermique est multipliée par F_eff dans la zone de flamme
        let kc: Vec<f64> = (0..n).map(|c| if self.is_solid(c) { self.k_cell[c] } else { self.k_cell[c] * self.flame_factor[c].max(1.0) }).collect();
        let mut amax = 0.0f64;
        for c in 0..n {
            let cap = self.heat_capacity(c);
            if cap > 0.0 {
                amax = amax.max(kc[c] / cap);
            }
        }
        if amax <= 0.0 {
            return;
        }
        let nsub = ((dt * amax / (0.2 * dx * dx)).ceil() as usize).clamp(1, 2000);
        let ds = dt / nsub as f64;
        let cap: Vec<f64> = (0..n).map(|c| self.heat_capacity(c)).collect();
        let mut t: Vec<f64> = (0..n).map(|c| self.temp_from_energy(c)).collect();
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
                        let (ka, kb) = (kc[c], kc[b]);
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
        let groups = self.reg.miscible_groups();
        for (pi, pc) in self.reg.changes.clone().iter().enumerate() {
            let (liq, vap) = (pc.liquid, pc.vapor);
            let (pl, pv) = (&self.reg.phases[liq].clone(), &self.reg.phases[vap].clone());
            // gaz miscibles avec la vapeur (air humide, fumées) : la saturation se fait à la PRESSION PARTIELLE de la vapeur (loi de Dalton)
            let others: Vec<usize> = groups
                .iter()
                .find(|g| g.contains(&vap))
                .map(|g| g.iter().copied().filter(|&k| k != vap && self.reg.phases[k].is_gas()).collect())
                .unwrap_or_default();
            for c in 0..n {
                if self.is_solid(c) || self.pinned(c) {
                    continue;
                }
                let (ml, mv) = (self.mass[liq][c], self.mass[vap][c]);
                let (t, p) = (self.temp[c], self.p[c]);
                let al = self.alpha[liq][c];
                let av = self.alpha[vap][c];
                if !others.is_empty() {
                    let n_o: f64 = others.iter().map(|&k| self.mass[k][c].max(0.0) / self.reg.phases[k].molar_mass).sum();
                    if n_o > 1e-9 && pc.p_sat(t) < 0.98 * p && (ml > 1e-12 || mv > 1e-12) {
                        if let Some(dm) = self.partial_pressure_equilibrium(c, pc, n_o, ml, mv, al) {
                            if dm != 0.0 {
                                self.transfer(c, liq, vap, dm);
                                self.transferred[pi] += dm * self.dx * self.dx;
                            }
                        }
                        continue;
                    }
                }
                let a_other_gas: f64 = (0..self.reg.phases.len()).filter(|&k| k != liq && k != vap && self.reg.phases[k].is_gas()).map(|k| self.alpha[k][c]).sum();
                // candidats : liquide présent (ébullition/cavitation) ou vapeur + liquide (condensation)
                if ml <= 1e-12 && mv <= 1e-12 {
                    continue;
                }
                let tsat = pc.t_sat(p);
                let cap = self.heat_capacity(c);
                let latent = self.reg.latent_heat(pc, t);
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
            if !others.is_empty() {
                self.surface_exchange(pi, &others);
            }
        }
    }

    /// Évaporation / condensation à travers une interface ALIGNÉE sur la grille (cellule de gaz pur voisine d'une cellule de liquide sans gaz :
    /// aucune cellule mixte, donc rien à faire pour le traitement cellule par cellule). La cellule de gaz est amenée à saturation
    /// (x_v·p = p_sat(T_surface)) ; la chaleur latente est prise au LIQUIDE (refroidissement par évaporation), la vapeur entre dans le gaz à
    /// la température du gaz ; la condensation réchauffe le liquide. Masse et énergie exactement conservées.
    pub(crate) fn surface_exchange(&mut self, pi: usize, others: &[usize]) {
        let pc = self.reg.changes[pi].clone();
        let (liq, vap) = (pc.liquid, pc.vapor);
        let (nx, ny) = (self.nx, self.ny);
        let np = self.reg.phases.len();
        let gas_ids: Vec<usize> = (0..np).filter(|&k| self.reg.phases[k].is_gas()).collect();
        for c in 0..nx * ny {
            if self.is_solid(c) || self.pinned(c) || self.alpha[liq][c] > 1e-6 {
                continue;
            }
            let n_o: f64 = others.iter().map(|&k| self.mass[k][c].max(0.0) / self.reg.phases[k].molar_mass).sum();
            if n_o <= 1e-9 {
                continue;
            }
            let (i, j) = (c % nx, c / nx);
            let nb = [(i > 0).then(|| c - 1), (i + 1 < nx).then(|| c + 1), (j > 0).then(|| c - nx), (j + 1 < ny).then(|| c + nx)];
            // voisin liquide « pur » (sans gaz) : interface entre les deux cellules
            let Some(l) = nb.into_iter().flatten().find(|&b| {
                !self.is_solid(b) && !self.pinned(b) && self.alpha[liq][b] > 0.5 && gas_ids.iter().all(|&k| self.alpha[k][b] < 1e-6)
            }) else {
                continue;
            };
            let p = self.p[c];
            let (tg, mv) = (self.temp[c], self.mass[vap][c]);
            let ml = self.mass[liq][l];
            let ev = self.reg.phases[vap].e_int(tg);
            let mvm = self.reg.phases[vap].molar_mass;
            let liq_masses: Vec<f64> = (0..np).map(|k| self.mass[k][l]).collect();
            let el = self.e[l];
            let phases = &self.reg.phases;
            // T de surface = T du liquide après l'échange (il fournit / reçoit la chaleur latente)
            let t_liq = |dm: f64| {
                let ms = liq_masses.iter().enumerate().map(|(k, &m)| if k == liq { (m - dm).max(0.0) } else { m });
                super::multiphase::temp_from_masses(phases, ms, el - ev * dm).max(150.0)
            };
            let f = |dm: f64| {
                let nv = ((mv + dm) / mvm).max(0.0);
                nv / (nv + n_o) * p - pc.p_sat(t_liq(dm))
            };
            let f0 = f(0.0);
            if f0.abs() < 1e-3 * pc.p_sat(tg).max(1.0) || pc.p_sat(t_liq(0.0)) >= 0.98 * p {
                continue;
            }
            let (lo, hi) = if f0 < 0.0 { (0.0, 0.2 * ml) } else { (-mv, 0.0) };
            let dm = if f(lo) >= 0.0 {
                lo
            } else if f(hi) <= 0.0 {
                hi
            } else {
                let (mut a, mut b) = (lo, hi);
                for _ in 0..50 {
                    let m = 0.5 * (a + b);
                    if f(m) < 0.0 {
                        a = m;
                    } else {
                        b = m;
                    }
                }
                0.5 * (a + b)
            };
            if dm == 0.0 {
                continue;
            }
            self.mass[liq][l] -= dm;
            self.mass[vap][c] += dm;
            self.e[l] -= ev * dm;
            self.e[c] += ev * dm;
            self.transferred[pi] += dm * self.dx * self.dx;
            self.temp[l] = self.temp_from_energy(l);
            self.temp[c] = self.temp_from_energy(c);
        }
    }

    /// Équilibre liquide ↔ vapeur dans un gaz mélangé (Dalton) : cherche dm (kg/m³, > 0 = évaporation) tel que la pression partielle de
    /// vapeur x_v·p égale p_sat(T') après le refroidissement / réchauffement latent (énergie interne conservée : c·ΔT = −dm·(e_v − e_l)).
    /// Évaporation : seulement s'il y a du liquide ; condensation : sur du liquide, ou en brouillard au-delà de 5 % de sursaturation.
    /// La cellule d'interface est ainsi saturée ; l'évaporation est ensuite limitée par la diffusion de la vapeur (loi de Fick).
    fn partial_pressure_equilibrium(&self, c: usize, pc: &super::phases::PhaseChange, n_o: f64, ml: f64, mv: f64, al: f64) -> Option<f64> {
        let (t, p) = (self.temp[c], self.p[c]);
        let (pl, pv) = (&self.reg.phases[pc.liquid], &self.reg.phases[pc.vapor]);
        let cap = self.heat_capacity(c).max(1e-30);
        let lat_u = (pv.e_int(t) - pl.e_int(t)).max(1.0);
        let mvm = pv.molar_mass;
        let f = |dm: f64| {
            let tn = (t - lat_u * dm / cap).max(150.0);
            let nv = ((mv + dm) / mvm).max(0.0);
            nv / (nv + n_o) * p - pc.p_sat(tn)
        };
        let f0 = f(0.0);
        let (lo, hi) = if f0 < 0.0 && ml > 1e-12 && al > 1e-6 {
            (0.0, ml)
        } else if f0 > 0.0 && mv > 1e-12 && (al > 1e-6 || f0 > 0.05 * pc.p_sat(t)) {
            (-mv, 0.0)
        } else {
            return Some(0.0);
        };
        // f croît avec dm : bracketing puis dichotomie
        let (flo, fhi) = (f(lo), f(hi));
        if flo >= 0.0 {
            return Some(lo);
        }
        if fhi <= 0.0 {
            return Some(hi);
        }
        let (mut a, mut b) = (lo, hi);
        for _ in 0..50 {
            let m = 0.5 * (a + b);
            if f(m) < 0.0 {
                a = m;
            } else {
                b = m;
            }
            if (b - a).abs() <= 1e-12 * (ml + mv).max(1e-12) {
                break;
            }
        }
        Some(0.5 * (a + b))
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
