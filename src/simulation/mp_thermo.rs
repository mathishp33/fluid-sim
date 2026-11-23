//! Énergie et changement de phase.
//!  * Conduction explicite (k harmonique aux liens, sous-pas stables), fluide ↔ solide inclus.
//!  * Ébullition / condensation : régime résolu (interface) = Hardt–Wondra : la cellule d'interface est ancrée à T_sat(p) et
//!    le taux de transfert est celui qui absorbe la chaleur apportée (dm = C·(T−T_sat)/L(T)), limité par la masse disponible ;
//!    régime sous-maille (germes dans le liquide) = Schnerr–Sauer, plafonné par l'énergie disponible.
//!  * Cavitation : p < p_sat(T) ⇔ T > T_sat(p). Même machinerie (germes créés si surchauffe/dépression > seuil).
//!  * Travail de pression −p ∇·u (appliqué APRÈS la projection) : c'est lui qui, avec les décalages d'énergie e0, restitue
//!    la chaleur latente exacte L(T) (refroidissement du liquide qui bout).

use super::mp_par::MIN_HEAVY;
use super::multiphase::{MultiphaseSim, SyncPtr};
use super::phases::MAX_PHASES;

/// température sous laquelle le travail de détente ne refroidit plus une cellule (voir `pressure_work`)
pub(crate) const T_PW_MIN: f64 = 60.0;

impl MultiphaseSim {
    /// Conduction explicite entre NŒUDS : une cellule solide, ou un groupe de fluide d'une cellule. À travers une face, le groupe g
    /// de la cellule a échange avec le groupe h de la cellule b sur la fraction de face w = α_g,a·α_h,b (Σ w = 1), avec la conductivité
    /// harmonique des deux. Cellules pures : identique à la conduction à une température. L'échange entre groupes d'une même cellule
    /// est fait ensuite (`exchange_heat`).
    pub(crate) fn conduct(&mut self, dt: f64) {
        if !self.conduction {
            return;
        }
        use super::mp_par::{for_ranges, for_rows, map_ranges, MIN_HEAVY, MIN_LIGHT};
        use super::multiphase::SyncPtr;
        use super::phases::MAX_PHASES;
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let n = nx * ny;
        let ng = self.groups.len();
        let dx2 = dx * dx;
        let harm = |a: f64, b: f64| if a + b > 0.0 { 2.0 * a * b / (a + b) } else { 0.0 };
        let ptrs = |v: &mut Vec<Vec<f64>>| -> Vec<SyncPtr<f64>> { v.iter_mut().map(|a| SyncPtr(a.as_mut_ptr())).collect() };
        // groupes granulaires : diamètre des grains et compaction maximale (rayonnement dans le tas, voir `bed_rad_k`)
        let gran: Vec<bool> = (0..ng).map(|g| self.group_is_granular(g)).collect();
        let gd: Vec<f64> = (0..ng).map(|g| if gran[g] { self.reg.phases[self.groups[g][0]].grain_diameter.max(1e-6) } else { 0.0 }).collect();
        let gmx: Vec<f64> = (0..ng).map(|g| if gran[g] { self.reg.phases[self.groups[g][0]].alpha_max.max(1e-6) } else { 1.0 }).collect();
        let (gran, gd, gmx) = (&gran, &gd, &gmx);
        // nœuds : conductivité (×F_eff des flammes épaissies), fraction, capacité, température
        let mut kg = vec![vec![0.0; n]; ng];
        let mut ag = vec![vec![0.0; n]; ng];
        let mut cg = vec![vec![0.0; n]; ng];
        let mut tgn = self.tg.clone();
        let mut ks = vec![0.0; n];
        let mut cs = vec![0.0; n];
        let mut ts = self.temp.clone();
        let amax = {
            let (kgp, agp, cgp) = (ptrs(&mut kg), ptrs(&mut ag), ptrs(&mut cg));
            let (ksp, csp) = (SyncPtr(ks.as_mut_ptr()), SyncPtr(cs.as_mut_ptr()));
            let this = &*self;
            map_ranges(n, MIN_HEAVY, |lo, hi| {
                let mut amax = 0.0f64;
                for c in lo..hi {
                    // SAFETY : la cellule c n'est écrite que par sa tranche
                    unsafe {
                        if this.is_solid(c) {
                            let (k, cap) = (this.k_cell[c], this.heat_capacity(c));
                            *ksp.at(c) = k;
                            *csp.at(c) = cap;
                            if cap > 0.0 {
                                amax = amax.max(k / cap);
                            }
                            continue;
                        }
                        let ff = this.flame_factor[c].max(1.0);
                        for g in 0..ng {
                            if this.group_mass(g, c) <= 0.0 {
                                continue;
                            }
                            let a = this.group_alpha(g, c);
                            let k = this.group_k(g, c) * ff;
                            let cap = this.group_cap_at(g, c, this.tg[g][c]);
                            *agp[g].at(c) = a;
                            *kgp[g].at(c) = k;
                            *cgp[g].at(c) = cap;
                            if cap > 0.0 && a > 0.0 {
                                amax = amax.max(k * a / cap);
                                // rayonnement dans le tas : conductance de face ≤ k_rad à la température du nœud le plus chaud, f/C ne
                                // dépend que du matériau ; marge 2 (miroir de k_cond_prep)
                                if gran[g] {
                                    let tg = this.tg[g][c];
                                    let f = (a / gmx[g]).min(1.0);
                                    amax = amax.max(2.0 * super::mp_granular::bed_rad_k(gd[g], tg, tg, f, 1.0) / cap);
                                }
                            }
                        }
                    }
                }
                amax
            })
            .into_iter()
            .fold(0.0f64, f64::max)
        };
        if amax <= 0.0 {
            return;
        }
        let nsub = ((dt * amax / (0.2 * dx2)).ceil() as usize).clamp(1, 2000);
        let ds = dt / nsub as f64;
        let iso: Vec<usize> = (0..n).filter(|&c| self.is_solid(c) && matches!(self.thermal[c], super::mp_cells::MpThermal::Isothermal(_))).collect();
        let mut des = vec![0.0; n];
        let mut deg = vec![vec![0.0; n]; ng];
        for _ in 0..nsub {
            // flux rassemblés par cellule, faces sud, ouest, est puis nord : l'ordre où l'ancienne boucle (cellules dans l'ordre, faces
            // est puis nord de chacune) les accumulait ; chaque face est calculée des deux côtés avec les mêmes opérations (a : cellule du
            // bas / de gauche, celle qui « visitait » la face) — résultat identique, en parallèle
            {
                let desp = SyncPtr(des.as_mut_ptr());
                let degp = ptrs(&mut deg);
                let (kg, ag, tgn, ks, ts) = (&kg, &ag, &tgn, &ks, &ts);
                let this = &*self;
                for_rows(ny, nx, MIN_HEAVY, |j| {
                    let mut acc = [0.0f64; MAX_PHASES];
                    for i in 0..nx {
                        let c = j * nx + i;
                        let mut dsc = 0.0;
                        acc[..ng].fill(0.0);
                        let mut face = |a: usize, b: usize| match (this.is_solid(a), this.is_solid(b)) {
                            (true, true) => {
                                let q = harm(ks[a], ks[b]) * (ts[b] - ts[a]) / dx2;
                                if c == a {
                                    dsc += q
                                } else {
                                    dsc -= q
                                }
                            }
                            (true, false) | (false, true) => {
                                let (sc, fc) = if this.is_solid(a) { (a, b) } else { (b, a) };
                                for h in 0..ng {
                                    if ag[h][fc] <= 0.0 {
                                        continue;
                                    }
                                    let q = harm(ks[sc], kg[h][fc]) * ag[h][fc] * (tgn[h][fc] - ts[sc]) / dx2; // vers le solide
                                    if c == sc {
                                        dsc += q
                                    } else {
                                        acc[h] -= q
                                    }
                                }
                            }
                            (false, false) => {
                                for g in 0..ng {
                                    if ag[g][a] <= 0.0 {
                                        continue;
                                    }
                                    for h in 0..ng {
                                        if ag[h][b] <= 0.0 {
                                            continue;
                                        }
                                        let mut kf = harm(kg[g][a], kg[h][b]) * ag[g][a] * ag[h][b];
                                        if gran[g] && gran[h] {
                                            let (fa, fb) = ((ag[g][a] / gmx[g]).min(1.0), (ag[h][b] / gmx[h]).min(1.0));
                                            kf += super::mp_granular::bed_rad_k(0.5 * (gd[g] + gd[h]), tgn[g][a], tgn[h][b], fa, fb);
                                        }
                                        let q = kf * (tgn[h][b] - tgn[g][a]) / dx2;
                                        if c == a {
                                            acc[g] += q
                                        } else {
                                            acc[h] -= q
                                        }
                                    }
                                }
                            }
                        };
                        if j > 0 {
                            face(c - nx, c);
                        }
                        if i > 0 {
                            face(c - 1, c);
                        }
                        if i + 1 < nx {
                            face(c, c + 1);
                        }
                        if j + 1 < ny {
                            face(c, c + nx);
                        }
                        // SAFETY : la cellule c n'est écrite que par sa ligne
                        unsafe {
                            *desp.at(c) = dsc;
                            for g in 0..ng {
                                *degp[g].at(c) = acc[g];
                            }
                        }
                    }
                });
            }
            {
                let ep = SyncPtr(self.e.as_mut_ptr());
                let egp = ptrs(&mut self.eg);
                let tsp = SyncPtr(ts.as_mut_ptr());
                let tgp = ptrs(&mut tgn);
                let (des, deg, cs, cg) = (&des, &deg, &cs, &cg);
                let this = &*self;
                for_ranges(n, MIN_LIGHT, |lo, hi| {
                    for c in lo..hi {
                        // SAFETY : la cellule c n'est écrite que par sa tranche
                        unsafe {
                            if this.is_solid(c) {
                                *ep.at(c) += ds * des[c];
                                *tsp.at(c) += ds * des[c] / cs[c].max(1e-30);
                                continue;
                            }
                            for g in 0..ng {
                                if deg[g][c] != 0.0 {
                                    *egp[g].at(c) += ds * deg[g][c];
                                    *tgp[g].at(c) += ds * deg[g][c] / cg[g][c].max(1e-30);
                                }
                            }
                        }
                    }
                });
            }
            // solides isothermes : température imposée à chaque sous-pas (ils sont des sources / puits de chaleur idéaux)
            for c in iso.iter().copied() {
                if let super::mp_cells::MpThermal::Isothermal(tc) = self.thermal[c] {
                    ts[c] = tc;
                    self.e[c] = cs[c] * tc;
                }
            }
        }
        for c in 0..n {
            if self.is_solid(c) {
                self.temp[c] = ts[c];
            }
        }
        self.sync_cells(|_| true);
    }

    /// Capacité calorifique qui fournit la chaleur latente d'un changement de phase du liquide `liq` dans c : celle du liquide plus
    /// χ × celle des autres groupes (χ = `thermal_coupling`).
    fn latent_cap(&self, c: usize, liq: usize, t: f64, dt: f64) -> f64 {
        let gl = self.group_of[liq];
        let cl = self.group_cap_at(gl, c, t);
        let others: f64 = (0..self.groups.len()).filter(|&g| g != gl && !self.latent_excluded(g, c)).map(|g| self.group_cap_at(g, c, self.tg[g][c])).sum();
        cl + self.thermal_coupling(c, dt) * others
    }

    /// Groupe tenu hors du partage de la chaleur latente : vapeur verrouillée sur son liquide (λ fini) — elle est maintenue à la
    /// température de l'interface par l'échange (G = ∞), elle n'absorbe pas la chaleur latente (sinon une bulle de vapeur est
    /// refroidie de quelques K à chaque pas d'ébullition, et la boîte fermée perd 14 % de l'énergie latente mise en jeu).
    fn latent_excluded(&self, g: usize, c: usize) -> bool {
        self.phase_heat_exchange.is_finite() && self.vapor_locked(g, c)
    }

    /// Transfert de masse liquide → vapeur de dm (kg/m³ de cellule, dm<0 : condensation). L'énergie totale est inchangée. La vapeur
    /// formée l'est à `t_vap` (interface), la vapeur qui se condense apporte son énergie à la température de son groupe ; la chaleur
    /// latente est partagée entre le liquide et les autres groupes EXACTEMENT comme dans `latent_cap` (liquide C_l, autres χ·C_g) :
    /// c'est avec cette capacité que dm a été estimé (équilibre, ancrage à T_sat). La faire porter au seul liquide quand χ ~ 1 (trace
    /// de liquide fortement couplée au gaz) dépassait l'équilibre : trace d'eau à 560 K après condensation, ou sous 50 K après
    /// évaporation (pas refusé, retours arrière en cascade).
    /// `transfer` sans le recalcul des températures de la cellule (fait ensuite par l'appelant, `sync_cells`)
    fn transfer_nosync(&mut self, c: usize, liq: usize, vap: usize, dm: f64, t_vap: f64, chi: f64) {
        let (gl, gv) = (self.group_of[liq], self.group_of[vap]);
        let tl = self.tg[gl][c];
        // capacités avant transfert (celles de latent_cap)
        let ng = self.groups.len();
        let mut caps = [0.0f64; super::phases::MAX_PHASES];
        let mut ctot = 0.0;
        if gl != gv && chi > 0.0 {
            for g in 0..ng {
                if g != gl && self.group_mass(g, c) > 0.0 && !self.latent_excluded(g, c) {
                    caps[g] = chi * self.group_cap_at(g, c, self.tg[g][c]);
                }
            }
            caps[gl] = self.group_cap_at(gl, c, tl);
            ctot = caps[..ng].iter().sum();
        }
        // volume créé / détruit, à p et aux températures de la masse déplacée (vapeur formée à t_vap, condensée à T_v ; brouillard :
        // liquide formé à T_v)
        {
            let p = self.p[c];
            let tv0 = self.tg[gv][c];
            let fog0 = dm < 0.0 && self.mass[liq][c] < -dm;
            let t_v = if dm > 0.0 { t_vap } else { tv0 };
            let t_l = if fog0 { tv0 } else { tl };
            let dvol = dm * (1.0 / self.reg.phases[vap].rho(p, t_v) - 1.0 / self.reg.phases[liq].rho(p, t_l));
            if dvol.is_finite() {
                self.pc_dvol[gl][c] += dvol;
            }
        }
        self.mass[liq][c] -= dm;
        self.mass[vap][c] += dm;
        if gl != gv {
            let tv = self.tg[gv][c];
            // condensation sans liquide pour recevoir la chaleur latente (brouillard) : la goutte naît à la température du gaz,
            // la chaleur latente reste au gaz
            let fog = dm < 0.0 && self.mass[liq][c] + dm < -dm;
            let ev = if dm > 0.0 {
                self.reg.phases[vap].e_int(t_vap)
            } else if fog {
                self.reg.phases[liq].e_int(tv)
            } else {
                self.reg.phases[vap].e_int(tv)
            };
            self.eg[gl][c] -= dm * ev;
            self.eg[gv][c] += dm * ev;
            if !fog && ctot > 0.0 {
                // part latente prise au (> 0) / donnée au (< 0) liquide au-delà de son énergie sensible, répartie
                let lat = dm * (ev - self.reg.phases[liq].e_int(tl));
                for g in 0..ng {
                    if g != gl && caps[g] > 0.0 {
                        let q = lat * caps[g] / ctot;
                        self.eg[g][c] -= q;
                        self.eg[gl][c] += q;
                    }
                }
            }
        }
    }

    /// Décision du changement de phase `pc` dans la cellule c (lecture seule de l'état) : (Some((dm, T de la vapeur, χ)) à transférer, borne
    /// de pas dt_phase de la cellule). Une cellule ne dépend que d'elle-même : décisions en parallèle, transferts ensuite.
    #[allow(clippy::too_many_arguments)]
    fn phase_decision(
        &self, c: usize, pc: &super::phases::PhaseChange, pl: &super::phases::Phase, pv: &super::phases::Phase, others: &[usize], dt: f64,
    ) -> (Option<(f64, f64, f64)>, f64) {
        let (liq, vap) = (pc.liquid, pc.vapor);
        let mut dt_phase = f64::INFINITY;
        if self.is_solid(c) || self.pinned(c) {
            return (None, dt_phase);
        }
        let (ml, mv) = (self.mass[liq][c], self.mass[vap][c]);
        // température du LIQUIDE (son groupe) : c'est sa surchauffe / son sous-refroidissement qui pilote le changement de phase
        let (t, p) = (self.phase_temp(liq, c), self.p[c]);
        let al = self.alpha[liq][c];
        let av = self.alpha[vap][c];
        // tri rapide (T_sat(p) se calcule par dichotomie, p_sat(T) par une exponentielle) : sans vapeur et sous la saturation,
        // ni ébullition, ni nucléation, ni condensation ne sont possibles (sauf évaporation dans un gaz mélangé : voir plus bas)
        if others.is_empty() && mv <= 1e-12 && pc.p_sat(t) <= p {
            return (None, dt_phase);
        }
        if !others.is_empty() {
            let n_o: f64 = others.iter().map(|&k| self.mass[k][c].max(0.0) / self.reg.phases[k].molar_mass).sum();
            if n_o > 1e-9 && pc.p_sat(t) < 0.98 * p && (ml > 1e-12 || mv > 1e-12) {
                if let Some(dm) = self.partial_pressure_equilibrium(c, pc, n_o, ml, mv, al, dt) {
                    if dm != 0.0 {
                        let chi = if ml > 1e-12 { self.thermal_coupling(c, dt) } else { 1.0 };
                        return (Some((dm, t, chi)), dt_phase);
                    }
                }
                return (None, dt_phase);
            }
        }
        let a_other_gas: f64 = (0..self.reg.phases.len()).filter(|&k| k != liq && k != vap && self.reg.phases[k].is_gas()).map(|k| self.alpha[k][c]).sum();
        // candidats : liquide présent (ébullition/cavitation) ou vapeur + liquide (condensation)
        if ml <= 1e-12 && mv <= 1e-12 {
            return (None, dt_phase);
        }
        let tsat = pc.t_sat(p);
        // capacité qui fournit la chaleur latente : le liquide, plus les autres groupes au prorata de leur couplage thermique χ
        // pendant le pas (χ = 1 : modèle à une température, toute la cellule) ; le transfert la prend au liquide, l'échange de
        // fin de phase thermique la répartit
        let cap = self.latent_cap(c, liq, t, dt);
        // chaleur latente pour ESTIMER la masse à changer (le transfert lui-même garde l'énergie exacte) : évaluée dans le
        // domaine de validité de p_sat (≤ 640 K) et bornée par le bas — h_v − h_l extrapolé linéairement devient NÉGATIF vers
        // 1450 K pour l'eau (liquide mêlé à de l'huile à 2000 K) et inversait le sens du transfert (masse de vapeur < 0)
        let latent = self.reg.latent_heat(pc, t.clamp(200.0, 640.0)).max(0.05 * pc.l_ref);
        let rho_v = pv.rho(p, t);
        let mut dm = 0.0;
        let mut bulk_boiling = false;
        // relaxation de l'écart à T_sat sur `phase_relaxation_time` (modèle de Lee) : instantanée pour dt ≫ τ (comportement
        // historique), ∝ dt pour dt ≪ τ — sans elle, ébullition et condensation alternent d'un pas à l'autre dans un liquide confiné
        let relax = 1.0 - (-dt / self.phase_relaxation_time.max(1e-12)).exp();
        let rate_cap = self.phase_volume_limit * dt / self.phase_relaxation_time.max(1e-12);
        if t > tsat && ml > 1e-12 {
            let dm_e = (cap * (t - tsat) / latent * relax).min(ml);
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
                // ébullition en volume (liquide chauffé) : la croissance des bulles est limitée par la chaleur disponible, pas par
                // l'inertie ; au-delà de `boil_superheat_max` le liquide ne reste plus métastable (ébullition hétérogène)
                // (surchauffe THERMIQUE, mesurée à max(p, 1 atm) : une dépression — cavitation — reste pilotée par l'inertie)
                let excess = t - pc.t_sat(p.max(101325.0)) - self.boil_superheat_max;
                let thermal = if excess > 0.0 { (cap * excess / latent * relax).min(ml) } else { 0.0 };
                bulk_boiling = excess > 0.0;
                dm = (rate * dt).max(thermal).min(dm_e);
            } else if av < 1e-7 && a_other_gas < 1e-9 && t - tsat > self.nucleation_superheat {
                // création de germes (nucléation) : fraction nucleation_alpha de vapeur
                dm = (self.nucleation_alpha * rho_v).min(dm_e);
            }
        } else if t < tsat && mv > 1e-12 && al > 1e-6 {
            // condensation : interface résolue ancrée à T_sat ; sous-maille par Schnerr–Sauer
            let dm_e = (cap * (tsat - t) / latent * relax).min(mv);
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
        // cavité discrète (DVCM) : la pression a atteint p_sat (plancher du solveur de pression) et le liquide ne remplit plus la
        // cellule : la vapeur à p_sat comble exactement le déficit de volume (cinématique, pas de limite de vitesse de croissance)
        let mut cavity = 0.0;
        if ml > 1e-12 && al > 1e-3 && dm >= 0.0 {
            let ps = pc.p_sat(t);
            // seulement dans une cellule que le solveur de pression a dû bloquer au plancher p_sat au pas précédent (traction) —
            // pas dans un liquide surchauffé ni à l'équilibre de saturation, qui n'est pas en traction
            if self.pressure_clamped[c] && p <= 1.01 * ps {
                let vv = self.cell_volume(c, ps);
                if vv < 1.0 {
                    let denom = (1.0 / pv.rho(ps, t) - 1.0 / pl.rho(ps, t)).max(1e-12);
                    cavity = ((1.0 - vv) / denom).min(ml - dm);
                    // même débit volumique maximal que la vaporisation (rate_cap ∝ dt) : remplie d'un bloc, la cavité vidait
                    // le liquide (et refroidissait la cellule sous 0 K) quel que soit dt — le filet de sécurité ne pouvait rien
                    // (huile à 2000 K au contact de l'eau chaude, cuve fermée)
                    cavity = cavity.min(rate_cap / denom);
                    let dv = cavity * denom;
                    if dv > 1e-3 {
                        dt_phase = dt_phase.min(0.5 * self.phase_volume_limit * dt / dv);
                    }
                }
            }
        }
        if dm != 0.0 {
            // couplage implicite local : on ne crée/détruit pas plus de vapeur que ce qui ramène la cellule à p = p_sat(T)
            // (sinon un pas explicite dépasse l'équilibre : 1 kg d'eau → 1600 m³ de vapeur)
            let ps = pc.p_sat(t);
            let vv = self.cell_volume(c, ps);
            let denom = 1.0 / pv.rho(ps, t) - 1.0 / pl.rho(ps, t);
            if dm > 0.0 {
                // … sauf ébullition en volume d'un liquide chauffé (surchauffe > boil_superheat_max) : la vapeur y repousse le liquide, ce que
                // fait la projection de pression ; la production est bornée à `phase_volume_limit` du volume de la cellule par pas,
                // et le pas suivant est réduit en conséquence (voir dt_phase)
                let pe = p.max(pc.p_sat(t)); // la vapeur formée est au moins à sa pression de saturation
                let dv_unit = (1.0 / pv.rho(pe, t) - 1.0 / pl.rho(pe, t)).max(1e-12);
                let by_volume = self.phase_volume_limit / dv_unit;
                let closure = ((1.0 - vv) / denom).max(0.0);
                dm = dm.min(if al > 0.98 && bulk_boiling { closure.max(by_volume) } else { closure });
                // débit volumique borné à phase_volume_limit / phase_relaxation_time (≈ 2500 /s) : sans cette borne, un liquide
                // porté bien au-delà de T_sat (eau au contact d'huile à 2000 K, même température dans la cellule) se vaporise d'un
                // bloc quel que soit dt, la vitesse produite varie en 1/dt et réduire le pas ne sert à rien
                dm = dm.min(rate_cap / dv_unit);
                let dv = dm * dv_unit;
                if dv > 1e-3 {
                    dt_phase = dt_phase.min(0.5 * self.phase_volume_limit * dt / dv);
                }
            } else {
                dm = -((-dm).min(((vv - 1.0) / denom).max(0.0)));
                // effondrement d'une bulle dans un liquide froid : au plus `phase_volume_limit` du volume de la cellule par pas
                // (sinon le liquide s'y précipite en un pas : coup de bélier de plusieurs GPa)
                let pe = p.max(pc.p_sat(t)); // la vapeur formée est au moins à sa pression de saturation
                let dv_unit = (1.0 / pv.rho(pe, t) - 1.0 / pl.rho(pe, t)).max(1e-12);
                dm = -((-dm).min(self.phase_volume_limit.min(rate_cap) / dv_unit));
                let dv = -dm * dv_unit;
                if dv > 1e-3 {
                    dt_phase = dt_phase.min(0.5 * self.phase_volume_limit * dt / dv);
                }
            }
        }
        let mut dm = dm + cavity.max(0.0);
        let chi = self.thermal_coupling(c, dt);
        if dm > 0.0 {
            // la chaleur latente prélevée ne peut pas faire tomber T de plus de la moitié de l'écart à T_PW_MIN (cavité remplie de
            // vapeur dense dans une eau très chaude : sinon T < 0). `transfer` la répartit au prorata des capacités : tous les
            // groupes qui y participent baissent d'autant — c'est le plus froid qui borne (air détendu à 130 K à côté d'eau à
            // 430 K : borné par l'eau seule, l'air passait sous 0 K)
            let gl = self.group_of[liq];
            let mut head = t - T_PW_MIN;
            if chi > 0.0 && gl != self.group_of[vap] {
                for g in 0..self.groups.len() {
                    if g != gl && self.group_mass(g, c) > 0.0 && !self.latent_excluded(g, c) {
                        head = head.min(self.tg[g][c] - T_PW_MIN);
                    }
                }
            }
            dm = dm.min(0.5 * cap * head.max(0.0) / latent);
        }
        if dm != 0.0 {
            return (Some((dm, tsat.min(t), chi)), dt_phase);
        }
        (None, dt_phase)
    }

    pub(crate) fn phase_change_step(&mut self, dt: f64) {
        let (ng, nc) = (self.groups.len().max(1), self.nx * self.ny);
        if self.pc_dvol.len() != ng || self.pc_dvol[0].len() != nc {
            self.pc_dvol = vec![vec![0.0; nc]; ng];
        } else {
            self.pc_dvol.iter_mut().for_each(|v| v.fill(0.0));
        }
        if !self.phase_change {
            self.dt_phase = f64::INFINITY;
            return;
        }
        let n = self.nx * self.ny;
        let groups = self.reg.miscible_groups();
        let mut dt_phase = f64::INFINITY;
        for (pi, pc) in self.reg.changes.clone().iter().enumerate() {
            let (liq, vap) = (pc.liquid, pc.vapor);
            let (pl, pv) = (&self.reg.phases[liq].clone(), &self.reg.phases[vap].clone());
            // gaz miscibles avec la vapeur (air humide, fumées) : la saturation se fait à la PRESSION PARTIELLE de la vapeur (loi de Dalton)
            let others: Vec<usize> = groups
                .iter()
                .find(|g| g.contains(&vap))
                .map(|g| g.iter().copied().filter(|&k| k != vap && self.reg.phases[k].is_gas()).collect())
                .unwrap_or_default();
            // décisions cellule par cellule (lecture seule, en parallèle), puis transferts dans l'ordre des cellules (sommes inchangées) ;
            // températures des cellules touchées recalculées ensuite, en parallèle
            let dec: Vec<(Option<(f64, f64, f64)>, f64)> =
                super::mp_par::collect(n, MIN_HEAVY, |c| self.phase_decision(c, pc, pl, pv, &others, dt));
            let mut touched = vec![false; n];
            for (c, &(act, dtp)) in dec.iter().enumerate() {
                dt_phase = dt_phase.min(dtp);
                if let Some((dm, tv, chi)) = act {
                    self.transfer_nosync(c, liq, vap, dm, tv, chi);
                    self.transferred[pi] += dm * self.dx * self.dx;
                    touched[c] = true;
                }
            }
            self.sync_cells(|c| touched[c]);
            if !others.is_empty() {
                self.surface_exchange(pi, &others);
            }
        }
        self.dt_phase = dt_phase.max(1e-8);
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
            // voisin liquide « pur » (gaz < 0.1 % : les traces laissées par l'advection ne doivent pas couper l'évaporation de la
            // surface) : interface entre les deux cellules
            let Some(l) = nb.into_iter().flatten().find(|&b| {
                !self.is_solid(b) && !self.pinned(b) && self.alpha[liq][b] > 0.5 && gas_ids.iter().map(|&k| self.alpha[k][b]).sum::<f64>() < 1e-3
            }) else {
                continue;
            };
            let p = self.p[c];
            let (gl, gv) = (self.group_of[liq], self.group_of[vap]);
            let (tg, mv) = (self.tg[gv][c], self.mass[vap][c]);
            let ml = self.mass[liq][l];
            let ev = self.reg.phases[vap].e_int(tg);
            let mvm = self.reg.phases[vap].molar_mass;
            // groupe du liquide dans sa cellule (seul concerné : une autre phase liquide voisine garde sa température)
            let liq_masses: Vec<f64> = (0..np).map(|k| if self.group_of[k] == gl { self.mass[k][l] } else { 0.0 }).collect();
            let el = self.eg[gl][l];
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
            self.eg[gl][l] -= ev * dm;
            self.eg[gv][c] += ev * dm;
            self.transferred[pi] += dm * self.dx * self.dx;
            self.cell_sync(l);
            self.cell_sync(c);
        }
    }

    /// Équilibre liquide ↔ vapeur dans un gaz mélangé (Dalton) : cherche dm (kg/m³, > 0 = évaporation) tel que la pression partielle de
    /// vapeur x_v·p égale p_sat(T') après le refroidissement / réchauffement latent (énergie interne conservée : c·ΔT = −dm·(e_v − e_l)).
    /// Évaporation : seulement s'il y a du liquide ; condensation : sur du liquide, ou en brouillard au-delà de 5 % de sursaturation.
    /// La cellule d'interface est ainsi saturée ; l'évaporation est ensuite limitée par la diffusion de la vapeur (loi de Fick).
    fn partial_pressure_equilibrium(&self, c: usize, pc: &super::phases::PhaseChange, n_o: f64, ml: f64, mv: f64, al: f64, dt: f64) -> Option<f64> {
        let (t, p) = (self.phase_temp(pc.liquid, c), self.p[c]);
        let (pl, pv) = (&self.reg.phases[pc.liquid], &self.reg.phases[pc.vapor]);
        // la chaleur latente vient du liquide et des groupes qui lui sont couplés ; sans liquide (brouillard) : de la cellule entière
        let cap = if ml > 1e-12 { self.latent_cap(c, pc.liquid, t, dt) } else { self.heat_capacity(c) }.max(1e-30);
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
    /// (en parallèle : chaque cellule ne dépend que d'elle-même ; énergies de groupe mises à jour puis températures, comme `cell_sync`)
    pub(crate) fn pressure_work(&mut self, dt: f64) {
        let (nx, ny) = (self.nx, self.ny);
        let ng = self.groups.len();
        let ep = SyncPtr(self.e.as_mut_ptr());
        let tp = SyncPtr(self.temp.as_mut_ptr());
        let egp: Vec<SyncPtr<f64>> = self.eg.iter_mut().map(|v| SyncPtr(v.as_mut_ptr())).collect();
        let tgp: Vec<SyncPtr<f64>> = self.tg.iter_mut().map(|v| SyncPtr(v.as_mut_ptr())).collect();
        let this = &*self;
        super::mp_par::for_rows(ny, nx, MIN_HEAVY, |j| {
            let mut egn = [0.0f64; MAX_PHASES];
            let mut tgn = [0.0f64; MAX_PHASES];
            for i in 0..nx {
                let c = this.idx(i, j);
                if this.is_solid(c) || this.pinned(c) {
                    continue;
                }
                // dt·∇·u = variation relative de volume du pas : au-delà de ±50 % la cellule est quasi vide / en cavitation violente,
                // où p·∇·u n'a plus de sens physique (et divise par une capacité calorifique qui tend vers 0)
                // (+ divergence du flux de percolation : le fluide des pores se détend ou se comprime aussi)
                let dq = if this.percolation { this.divergence_of(&this.q_u, &this.q_v, i, j) } else { 0.0 };
                let d = (dt * (this.divergence_of(&this.u, &this.v, i, j) + dq)).clamp(-0.5, 0.5) / dt;
                // au plus −50 % / +100 % de T par pas, et pas de refroidissement sous T_PW_MIN : une cellule presque vide (liquide
                // arraché, poche de gaz raréfiée par un écoulement violent) a une capacité calorifique quasi nulle et p·∇·u y donnerait des
                // températures absurdes (< 50 K → pas refusé à tout dt). L'énergie ainsi écrêtée est négligeable (capacité ~0).
                // répartition entre groupes comme dans la fermeture de la projection (C_g·dT_g/dp) : sans couplage, c'est le gaz qui se
                // détend / se comprime, pas le liquide qui l'entoure ; groupes couplés : au prorata des capacités (température commune)
                let mut tgs = [0.0f64; super::phases::MAX_PHASES];
                let mut gam = [0.0f64; super::phases::MAX_PHASES];
                let mut ks = [0.0f64; super::phases::MAX_PHASES];
                for g in 0..ng {
                    tgs[g] = this.tg[g][c];
                }
                let chi = this.thermal_coupling(c, dt);
                this.cell_closure_g(c, this.p[c], &tgs[..ng], chi, &mut gam[..ng], &mut ks[..ng]);
                let ksum: f64 = (0..ng).map(|g| ks[g].max(0.0)).sum();
                let asum: f64 = (0..ng).map(|g| this.group_alpha(g, c)).sum::<f64>().max(1e-300);
                let pc = this.p[c].max(0.0);
                let de_tot = -dt * pc * d;
                // part due au changement de phase du pas (−p·ΔV_pc) : revient au groupe du liquide qui a changé d'état, partagée comme
                // la chaleur latente (liquide C_l, autres χ·C_g, voir `transfer`) — répartie par compressibilité, le p·ΔV d'une vapeur
                // qui se condense chauffait l'air voisin (air d'une cellule air + eau porté à 5000 K, huile à 2000 K sur l'eau chaude).
                // Bornée au signe et à l'amplitude de −p∇·u dt (la projection ne réalise pas forcément tout ΔV_pc).
                let mut de_pc = [0.0f64; super::phases::MAX_PHASES];
                let mut pc_sum = 0.0;
                for g in 0..ng.min(this.pc_dvol.len()) {
                    de_pc[g] = -pc * this.pc_dvol[g][c];
                    pc_sum += de_pc[g];
                }
                let scale = if pc_sum != 0.0 && pc_sum * de_tot > 0.0 { (de_tot / pc_sum).min(1.0) } else { 0.0 };
                let de_rest = de_tot - scale * pc_sum;
                let mut de_g = [0.0f64; super::phases::MAX_PHASES];
                for g in 0..ng {
                    if this.group_mass(g, c) > 0.0 {
                        de_g[g] = de_rest * if ksum > 0.0 { ks[g].max(0.0) / ksum } else { this.group_alpha(g, c) / asum };
                    }
                }
                if scale > 0.0 {
                    for gs in 0..ng {
                        if de_pc[gs] == 0.0 {
                            continue;
                        }
                        let q = scale * de_pc[gs];
                        let mut caps = [0.0f64; super::phases::MAX_PHASES];
                        for g in 0..ng {
                            if this.group_mass(g, c) > 0.0 && (g == gs || !this.latent_excluded(g, c)) {
                                caps[g] = this.group_cap_at(g, c, tgs[g].max(1.0)) * if g == gs { 1.0 } else { chi };
                            }
                        }
                        let ctot: f64 = caps[..ng].iter().sum();
                        for g in 0..ng {
                            if ctot > 0.0 {
                                de_g[g] += q * caps[g] / ctot;
                            } else if g == gs {
                                de_g[g] += q;
                            }
                        }
                    }
                }
                for g in 0..ng {
                    egn[g] = this.eg[g][c];
                }
                for g in 0..ng {
                    if this.group_mass(g, c) <= 0.0 {
                        continue;
                    }
                    let t = tgs[g].max(1.0);
                    let cap = this.group_cap_at(g, c, t);
                    let de_min = (-0.5 * cap * t).max(cap * (T_PW_MIN - t)).min(0.0);
                    egn[g] += de_g[g].clamp(de_min, cap * t);
                }
                let e: f64 = egn[..ng].iter().fold(0.0, |a, &b| a + b);
                let (ec, tc) = this.sync_finish(c, &mut egn, &mut tgn, e);
                // SAFETY : la cellule c n'est écrite que par sa ligne
                unsafe {
                    *ep.at(c) = ec;
                    *tp.at(c) = tc;
                    for g in 0..ng {
                        *egp[g].at(c) = egn[g];
                        *tgp[g].at(c) = tgn[g];
                    }
                }
            }
        });
    }

    /// Cavités sans liquide volatil : une cellule dont la pression est bloquée au plancher alors que son contenu n'occupe plus tout
    /// son volume (liquide non volatil arraché — huile, ou cellule vidée par un écoulement violent) est complétée par du gaz ambiant
    /// à cette pression (≈ 10 Pa : ~1e-4 kg/m³, dégazage). Sans cela la cellule reste « vide » : α normalisé la fait voir pleine de
    /// liquide à l'advection, l'erreur de volume s'accumule et finit en coup de bélier. Les liquides à changement de phase sont traités
    /// par la cavité de vapeur (DVCM, voir `phase_change_step`).
    pub(crate) fn fill_cavities(&mut self) {
        let n = self.nx * self.ny;
        let amb = self.ambient_alpha();
        for c in 0..n {
            if self.is_solid(c) || self.pinned(c) || !self.pressure_clamped[c] {
                continue;
            }
            if self.phase_change && self.reg.changes.iter().any(|pc| self.alpha[pc.liquid][c] > 1e-3) {
                continue;
            }
            let p = self.p[c];
            let t = self.temp[c];
            let t = if t.is_finite() { t.clamp(150.0, 5000.0) } else { self.t_ambient };
            let deficit = 1.0 - self.cell_volume(c, p);
            if deficit <= 1e-6 {
                continue;
            }
            for k in 0..self.reg.phases.len() {
                if amb[k] > 0.0 {
                    let m = deficit * amb[k] * self.reg.phases[k].rho(p, t);
                    self.add_mass_at(k, c, m, t);
                }
            }
            self.cell_sync(c);
        }
    }
}
