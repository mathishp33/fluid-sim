//! Boucle de temps du solveur multiphasique.
//!
//! Ordre d'un pas :
//!  1. refresh (T, α, ρ, μ, k depuis m, e, p)         5. refresh (ρ de l'état advecté)
//!  2. vitesse : advection MacCormack + forces → u*   6. pression implicite → pⁿ⁺¹, uⁿ⁺¹
//!  3. (m_k, e) : advection PLIC conservative (uⁿ)    7. travail −p∇·u, refresh final
//!  4. conduction, diffusion des espèces miscibles, réactions (accroche), changement de phase (source de volume)
//! dt est limité par l'advection (CFL), la gravité, la tension de surface (Brackbill) et la viscosité — PAS par l'acoustique.

use std::time::Instant;

use super::multiphase::MultiphaseSim;

/// Diagnostic : cellule sondée par `step_raw` après chaque étape (usize::MAX = aucune). Affiche T et e massique de chaque groupe, masses,
/// p, fermeture Σ m/ρ − 1 et vitesse max des faces : repère l'étape qui fait diverger une cellule.
/// croissance maximale du pas de temps d'un pas au suivant (voir `choose_dt`). ×1.25 était trop serré : après un pas refait (dt/4),
/// le pas ne remontait pas assez vite et les retours arrière s'enchaînaient (braséro : dt jusqu'à 1.5e-6 s)
pub(crate) const DT_GROWTH: f64 = 2.0;
/// déplacement maximal de la surface d'un corps libre par pas, en mailles (voir `choose_dt`)
pub(crate) const FREE_BODY_CFL: f64 = 0.15;

pub(crate) static PROBE_CELL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

pub const P_REFRESH: usize = 0;
pub const P_FORCES: usize = 1;
pub const P_ADVECT_MASS: usize = 2;
pub const P_ADVECT_MOM: usize = 3;
pub const P_CONDUCT: usize = 4;
pub const P_SPECIES: usize = 5;
pub const P_CHEM: usize = 6;
pub const P_PHASE: usize = 7;
pub const P_PRESSURE: usize = 8;
pub const P_OTHER: usize = 9;
/// solveur GPU résident : pas complet (encodage, calcul GPU et relecture des scalaires de contrôle)
pub const P_GPU: usize = 10;
pub const PROFILE_NAMES: [&str; 11] = [
    "propriétés (refresh)",
    "forces (gravité, tension, viscosité)",
    "advection masse / énergie (PLIC)",
    "advection quantité de mouvement",
    "conduction thermique",
    "diffusion des espèces",
    "chimie + rayonnement",
    "changement de phase",
    "pression (Newton + PCG)",
    "divers (cellules spéciales, dt)",
    "GPU : pas complet (calcul + attente)",
];

/// Temps passé dans chaque partie du pas (secondes cumulées depuis la dernière remise à zéro).
#[derive(Clone, Debug, Default)]
pub struct StepProfile {
    pub secs: [f64; 11],
    pub steps: usize,
}

impl StepProfile {
    pub fn total(&self) -> f64 {
        self.secs.iter().sum()
    }
    /// (nom, ms par pas, part du total)
    pub fn report(&self) -> Vec<(&'static str, f64, f64)> {
        let tot = self.total().max(1e-30);
        let n = self.steps.max(1) as f64;
        (0..PROFILE_NAMES.len()).map(|k| (PROFILE_NAMES[k], 1e3 * self.secs[k] / n, self.secs[k] / tot)).collect()
    }
}

struct Snapshot {
    mass: Vec<Vec<f64>>, alpha: Vec<Vec<f64>>, e: Vec<f64>, temp: Vec<f64>, p: Vec<f64>, u: Vec<f64>, v: Vec<f64>,
    q_u: Vec<f64>, q_v: Vec<f64>, sigma: Vec<f64>, skel: Vec<bool>,
    eg: Vec<Vec<f64>>, tg: Vec<Vec<f64>>,
    rho_mix: Vec<f64>, mu_cell: Vec<f64>, k_cell: Vec<f64>, boundary_flux: Vec<f64>, transferred: Vec<f64>, frozen: Vec<f64>, fusion_front: Vec<Vec<f64>>,
    energy_input: f64, tp_mix: super::mp_cells::TpMix,
    flame_factor: Vec<f64>, heat_release: Vec<f64>, chem_heat: f64, radiated: f64, reacted: Vec<f64>, dt_chem: f64,
    mfx: Vec<f64>, mfy: Vec<f64>, m_tot_old: Vec<f64>,
    time: f64, last_dt: f64, last_pressure_iters: usize, step_count: usize,
    dt_phase: f64, pressure_clamped: Vec<bool>,
    /// corps rigides (vide sans corps ; leurs fractions et faces se recalculent depuis leur position) et ressorts de leurs contacts
    bodies: Vec<super::mp_bodies::RigidBody>,
    contact_springs: Vec<super::mp_contacts::ContactSpring>,
}

impl MultiphaseSim {
    pub fn choose_dt(&self) -> f64 {
        let n = self.nx * self.ny;
        let dx = self.dx;
        // corps à mouvement imposé : la vitesse de leur surface entre dans le CFL
        let umax = if self.has_moving_bodies() { self.max_speed().max(self.body_surface_speed()) } else { self.max_speed() };
        let mut dt = self.dt_max.min(self.dt_chem).min(self.dt_phase).min(self.burn_dt_bound()).min(self.front_dt_bound());
        // chauffage au pinceau : au plus ~20 K par pas (dilatation thermique progressive des liquides, ébullition suivie pas à pas)
        let gap = self.heat_gap();
        if gap > 0.0 {
            dt = dt.min((self.heat_tau * 20.0 / gap).max(1e-6));
        }
        dt = dt.min(self.cfl * dx / umax.max(1e-9));
        // corps libres : leur surface ne parcourt pas plus de 0.15 maille par pas — au-delà, l'écart entre le volume que la projection
        // anticipe (linéarisé) et celui que le corps balaie vraiment faisait osciller une cuve fermée de liquide (±26 kPa à 0.33 maille
        // par pas)
        if self.has_free_bodies() {
            let vmax = self.bodies.iter().filter(|b| b.is_free()).map(|b| b.vel[0].hypot(b.vel[1]) + b.omega.abs() * b.shape.radius()).fold(0.0, f64::max);
            dt = dt.min(FREE_BODY_CFL * dx / vmax.max(1e-9));
            // contacts (mp_contacts.rs) : pénétration bornée près d'un obstacle
            dt = dt.min(self.contact_dt_bound());
        }
        // grains : la dérive explicite ne traverse pas plus d'une demi-maille par pas ; percolation : |q| dt ≤ 0.4 du fluide amont
        let w = if self.has_granular() { self.max_settling_speed() } else { 0.0 }.max(self.max_darcy_speed());
        if w > 0.0 {
            dt = dt.min(0.5 * dx / w);
        }
        // flamme présente (facteur d'épaississement du pas précédent) : borne de pas propre aux flammes
        if self.flames_active() && self.flame_factor.iter().any(|&f| f > 1.0) {
            dt = dt.min(self.cfl_flame * dx / self.flame_speed_floor);
        }
        let g = (self.gravity[0].powi(2) + self.gravity[1].powi(2)).sqrt();
        if g > 0.0 {
            dt = dt.min(0.5 * (dx / g).sqrt());
        }
        // croissance du pas bornée (×2) : les vitesses de la dernière projection, qui servent à l'advection de ce pas, ont été calculées
        // pour corriger l'erreur de volume en `last_dt` ; appliquées sur un pas 4× plus long (après un pas court imposé par la chimie),
        // elles déplacent 4× trop : une poche de gaz qui s'enflamme d'un coup était vidée à 92 % au pas suivant et son reste porté à 3800 K
        if self.last_dt > 0.0 {
            dt = dt.min(DT_GROWTH * self.last_dt);
        }
        let (mut rmin, mut rmax, mut sig) = (f64::MAX, 0.0f64, 0.0f64);
        for c in 0..n {
            if self.is_solid(c) {
                continue;
            }
            rmin = rmin.min(self.rho_mix[c]);
            rmax = rmax.max(self.rho_mix[c]);
            // (viscosité implicite : la part au-delà de la borne explicite est résolue par `apply_implicit_viscosity`, pas de borne)
            if self.mu_cell[c] > 0.0 && !(self.viscous && self.implicit_viscosity) {
                dt = dt.min(0.2 * self.rho_mix[c] * dx * dx / self.mu_cell[c]);
            }
        }
        // corps en mouvement (mp_bodies.rs) : la viscosité implicite ne voit pas leur vitesse ; autour d'eux le pas reste sous la borne
        // explicite, la viscosité y est entièrement explicite, calculée sur la vitesse du mélange fluide + corps (sans cela, un disque
        // qui sédimente dans l'huile faisait exploser les pas plus longs que la borne : vitesses de 6000 m/s, pas refaits sans fin)
        if self.viscous && self.implicit_viscosity && self.has_moving_bodies() {
            let (nx, ny) = (self.nx, self.ny);
            for c in 0..n {
                if self.is_solid(c) || self.mu_cell[c] <= 0.0 {
                    continue;
                }
                let (i, j) = (c % nx, c / nx);
                let near = |ii: isize, jj: isize| ii >= 0 && jj >= 0 && (ii as usize) < nx && (jj as usize) < ny && self.body_frac[jj as usize * nx + ii as usize] > 0.0;
                let (ii, jj) = (i as isize, j as isize);
                if !(near(ii, jj) || near(ii - 1, jj) || near(ii + 1, jj) || near(ii, jj - 1) || near(ii, jj + 1)) {
                    continue;
                }
                let mut rn = self.rho_mix[c];
                for (ok, q) in [(i > 0, c.wrapping_sub(1)), (i + 1 < nx, c + 1), (j > 0, c.wrapping_sub(nx)), (j + 1 < ny, c + nx)] {
                    if ok && !self.is_solid(q) {
                        rn = rn.min(self.rho_mix[q]);
                    }
                }
                dt = dt.min(0.99 * super::mp_friction::VISC_EXPLICIT * rn * dx * dx / self.mu_cell[c]);
            }
        }
        // tension de surface : seulement des liquides effectivement présents (un liquide absent n'impose pas de pas capillaire)
        for (k, ph) in self.reg.phases.iter().enumerate() {
            if ph.sigma > sig && self.mass[k].iter().any(|&m| m > 1e-9) {
                sig = ph.sigma;
            }
        }
        if self.surface_tension && sig > 0.0 {
            dt = dt.min(self.cap_safety * ((rmin + rmax) * dx.powi(3) / (4.0 * std::f64::consts::PI * sig)).sqrt());
        }
        dt
    }

    /// État physiquement exploitable ? (aucun NaN/∞, T dans [50 ; 5000] K, masses ≥ 0, p ≥ 0 dans le fluide, vitesses < speed_limit)
    pub fn state_is_sane(&self) -> bool {
        let n = self.nx * self.ny;
        if !self.u.iter().chain(self.v.iter()).all(|x| x.is_finite() && x.abs() <= self.speed_limit) {
            return false;
        }
        for c in 0..n {
            if !self.temp[c].is_finite() || !self.e[c].is_finite() || !self.p[c].is_finite() {
                return false;
            }
            if self.solid[c] >= 0 {
                continue;
            }
            if self.temp[c] < 50.0 || self.temp[c] > 5000.0 || self.p[c] < 0.0 {
                return false;
            }
            // température de chaque groupe présent
            for g in 0..self.groups.len() {
                if self.group_mass(g, c) > 0.0 {
                    let t = self.tg[g][c];
                    if !t.is_finite() || t < 50.0 || t > 5000.0 || !self.eg[g][c].is_finite() {
                        return false;
                    }
                }
            }
            for k in 0..self.mass.len() {
                let m = self.mass[k][c];
                if !m.is_finite() || m < -1e-6 {
                    return false;
                }
            }
        }
        true
    }

    /// Nombre de Courant des vitesses produites par le pas ≤ `cfl_reject` ?
    fn courant_ok(&self, dt: f64) -> bool {
        self.max_speed() * dt <= self.cfl_reject * self.dx
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            mass: self.mass.clone(), alpha: self.alpha.clone(), e: self.e.clone(), temp: self.temp.clone(), p: self.p.clone(),
            eg: self.eg.clone(), tg: self.tg.clone(),
            u: self.u.clone(), v: self.v.clone(), rho_mix: self.rho_mix.clone(), mu_cell: self.mu_cell.clone(), k_cell: self.k_cell.clone(),
            q_u: self.q_u.clone(), q_v: self.q_v.clone(), sigma: self.sigma.clone(), skel: self.skel.clone(),
            boundary_flux: self.boundary_flux.clone(), transferred: self.transferred.clone(), frozen: self.frozen.clone(), fusion_front: self.fusion_front.clone(),
            energy_input: self.energy_input, tp_mix: self.tp_mix.clone(),
            flame_factor: self.flame_factor.clone(), heat_release: self.heat_release.clone(), chem_heat: self.chem_heat,
            radiated: self.radiated, reacted: self.reacted.clone(), dt_chem: self.dt_chem,
            mfx: self.mfx.clone(), mfy: self.mfy.clone(), m_tot_old: self.m_tot_old.clone(),
            time: self.time, last_dt: self.last_dt, last_pressure_iters: self.last_pressure_iters, step_count: self.step_count,
            dt_phase: self.dt_phase, pressure_clamped: self.pressure_clamped.clone(),
            bodies: self.bodies.clone(),
            contact_springs: self.contact_springs.clone(),
        }
    }

    fn restore(&mut self, s: &Snapshot) {
        self.mass.clone_from(&s.mass); self.alpha.clone_from(&s.alpha); self.e.clone_from(&s.e); self.temp.clone_from(&s.temp);
        self.eg.clone_from(&s.eg); self.tg.clone_from(&s.tg);
        self.p.clone_from(&s.p); self.u.clone_from(&s.u); self.v.clone_from(&s.v); self.rho_mix.clone_from(&s.rho_mix);
        self.q_u.clone_from(&s.q_u); self.q_v.clone_from(&s.q_v); self.sigma.clone_from(&s.sigma); self.skel.clone_from(&s.skel);
        self.mu_cell.clone_from(&s.mu_cell); self.k_cell.clone_from(&s.k_cell);
        self.boundary_flux.clone_from(&s.boundary_flux); self.transferred.clone_from(&s.transferred); self.frozen.clone_from(&s.frozen); self.fusion_front.clone_from(&s.fusion_front);
        self.energy_input = s.energy_input; self.tp_mix.clone_from(&s.tp_mix);
        self.flame_factor.clone_from(&s.flame_factor); self.heat_release.clone_from(&s.heat_release); self.chem_heat = s.chem_heat;
        self.radiated = s.radiated; self.reacted.clone_from(&s.reacted); self.dt_chem = s.dt_chem;
        self.mfx.clone_from(&s.mfx); self.mfy.clone_from(&s.mfy); self.m_tot_old.clone_from(&s.m_tot_old);
        self.time = s.time; self.last_dt = s.last_dt; self.last_pressure_iters = s.last_pressure_iters; self.step_count = s.step_count;
        self.dt_phase = s.dt_phase; self.pressure_clamped.clone_from(&s.pressure_clamped);
        if self.has_bodies() || !s.bodies.is_empty() {
            // fractions et faces : fonction exacte de la position des corps
            self.bodies.clone_from(&s.bodies);
            self.contact_springs.clone_from(&s.contact_springs);
            self.update_body_fractions();
        }
    }

    /// Un pas de temps protégé : si l'état obtenu est invalide (NaN, T aberrante, masse négative, vitesse > speed_limit) le pas est
    /// annulé et refait avec dt/4, dt/16, … (jusqu'à dt/1024). Si rien ne passe, l'état est restauré, `diverged` passe à true et
    /// 0.0 est retourné : l'appelant doit s'arrêter (au lieu de propager des NaN dans tout le domaine). Retourne dt.
    /// Sa durée nourrit le réglage automatique du nombre de fils (`mp_par`).
    pub fn step(&mut self) -> f64 {
        let t0 = Instant::now();
        let dt = self.step_guarded();
        super::mp_par::record_step(t0.elapsed().as_secs_f64());
        dt
    }

    fn step_guarded(&mut self) -> f64 {
        if self.diverged {
            return 0.0;
        }
        let snap = self.snapshot();
        let dt_max0 = self.dt_max;
        let dt = self.step_pc();
        if self.state_is_sane() && self.courant_ok(dt) {
            return dt;
        }
        let mut dt_try = dt;
        for _ in 0..5 {
            self.restore(&snap);
            dt_try *= 0.25;
            self.dt_max = dt_max0.min(dt_try);
            self.closure_relax_scale *= 0.25;
            self.rollbacks += 1;
            let dt = self.step_pc();
            if self.state_is_sane() && self.courant_ok(dt) {
                self.dt_max = dt_max0;
                self.closure_relax_scale = 1.0;
                return self.last_dt;
            }
        }
        self.restore(&snap);
        self.dt_max = dt_max0;
        self.closure_relax_scale = 1.0;
        self.diverged = true;
        0.0
    }

    /// Pas brut, en prédicteur-correcteur si `predictor_corrector` (voir le champ). La prédiction invalide (NaN, T aberrante…) est
    /// rendue telle quelle : le filet de sécurité de `step` la refait avec dt/4.
    pub(crate) fn step_pc(&mut self) -> f64 {
        if !self.use_predictor_corrector() {
            return self.step_raw();
        }
        let snap = self.snapshot();
        self.pc_predicting = true;
        let dt = self.step_raw();
        self.pc_predicting = false;
        if !self.state_is_sane() {
            return dt;
        }
        let (up, vp) = (std::mem::take(&mut self.u), std::mem::take(&mut self.v));
        let (ua, va): (Vec<f64>, Vec<f64>) = (
            snap.u.iter().zip(&up).map(|(a, b)| 0.5 * (a + b)).collect(),
            snap.v.iter().zip(&vp).map(|(a, b)| 0.5 * (a + b)).collect(),
        );
        let rollbacks = self.rollbacks;
        self.restore(&snap);
        self.rollbacks = rollbacks;
        self.adv_vel = Some((ua, va));
        self.dt_force = Some(dt);
        let dt = self.step_raw();
        self.adv_vel = None;
        self.dt_force = None;
        dt
    }

    /// Un pas de temps sans protection. Retourne dt.
    /// Un pas SANS le filet de sécurité (diagnostic, sondes).
    pub fn step_raw_pub(&mut self) -> f64 {
        self.step_raw()
    }
    pub(crate) fn step_raw(&mut self) -> f64 {
        let mut clk = Instant::now();
        let mut lap = |prof: &mut StepProfile, k: usize| {
            let now = Instant::now();
            prof.secs[k] += (now - clk).as_secs_f64();
            clk = now;
        };
        let mut prof = std::mem::take(&mut self.profile);
        self.refresh();
        lap(&mut prof, P_REFRESH);
        // cellules spéciales / entrées : faces à vitesse imposée, cellules épinglées remises à leur composition cible
        self.prepare_special();
        if self.has_extras() {
            self.apply_prescribed_faces();
            self.pin_reset();
            self.refresh();
        }
        let dt = self.dt_force.unwrap_or_else(|| self.choose_dt());
        // erreur de fermeture héritée (Σ m/ρ(p, T) − 1 avec les masses du début du pas), sous-relaxée dans l'équation de pression
        let n = self.nx * self.ny;
        let err0: Vec<f64> = super::mp_par::collect(n, super::mp_par::MIN_HEAVY, |c| {
            if self.is_solid(c) || self.pinned(c) { 0.0 } else { self.cell_volume(c, self.p[c]) - 1.0 }
        });
        self.probe("début");
        lap(&mut prof, P_OTHER);
        let (un, vn) = (self.u.clone(), self.v.clone());
        // corps en mouvement (mp_bodies.rs) : la quantité de mouvement transportée est celle du MÉLANGE fluide + corps des faces,
        // w = u + (1 − A) V_corps, régulière (rigide dans les corps, la vitesse du fluide autour) ; le débit u, nul dans les corps et
        // réduit à A V au bord, ne se transporte pas comme une vitesse : un disque qui avance avec le courant laissait le fluide qui le
        // borde à 0.8 V (frottement et pression parasites). Le débit se déduit ensuite de w avec la position des corps après leur pas.
        let mixture = if self.has_moving_bodies() { Some(self.mixture_velocity()) } else { None };
        // débit du fluide collé aux corps en mouvement (position du début du pas) : le transport des masses ne voit que le débit relatif
        // (voir `body_attached_flux`)
        let attached = self.body_attached_flux();
        // 2. prédiction de vitesse
        let (mut us, mut vs);
        if self.conservative_momentum {
            // forces évaluées sur l'état n, puis masses/énergie, puis quantité de mouvement avec les MÊMES flux de masse
            let mut fx = vec![0.0; self.u.len()];
            let mut fy = vec![0.0; self.v.len()];
            self.add_forces(dt, &mut fx, &mut fy);
            lap(&mut prof, P_FORCES);
            self.advect_mass_energy_rel(dt, attached.as_ref());
            lap(&mut prof, P_ADVECT_MASS);
            let (mut ua, mut va) = match &mixture {
                Some((wu, wv)) => {
                    let (fu, fv) = (std::mem::replace(&mut self.u, wu.clone()), std::mem::replace(&mut self.v, wv.clone()));
                    let r = self.advect_momentum();
                    self.u = fu;
                    self.v = fv;
                    r
                }
                None => self.advect_momentum(),
            };
            for (a, f) in ua.iter_mut().zip(&fx) { *a += *f; }
            for (a, f) in va.iter_mut().zip(&fy) { *a += *f; }
            us = ua;
            vs = va;
            lap(&mut prof, P_ADVECT_MOM);
        } else {
            let (mut a, mut b) = match &mixture {
                Some((wu, wv)) => {
                    let (fu, fv) = (std::mem::replace(&mut self.u, wu.clone()), std::mem::replace(&mut self.v, wv.clone()));
                    let r = self.advect_velocity(dt);
                    self.u = fu;
                    self.v = fv;
                    r
                }
                None => self.advect_velocity(dt),
            };
            lap(&mut prof, P_ADVECT_MOM);
            self.add_forces(dt, &mut a, &mut b);
            lap(&mut prof, P_FORCES);
            self.advect_mass_energy_rel(dt, attached.as_ref());
            lap(&mut prof, P_ADVECT_MASS);
            us = a;
            vs = b;
        }
        // corps à mouvement imposé ou libres (mp_bodies.rs) : ils avancent pendant que le fluide est transporté (les flux de ce pas ont
        // été calculés par la projection précédente pour faire place au volume qu'ils balaient)
        if let Some((wu, wv)) = mixture {
            self.move_bodies(self.time, self.time + dt);
            // débits déduits des vitesses du mélange avec la nouvelle position des corps : prédits (us, vs) et courants (dont la vitesse
            // du mélange sert au frottement des corps ; la divergence de la vitesse qui a transporté les masses, `un`, `vn`, est à part)
            self.mixture_to_flux(&mut us, &mut vs);
            let (mut cu, mut cv) = (wu, wv);
            self.mixture_to_flux(&mut cu, &mut cv);
            self.u = cu;
            self.v = cv;
        }
        // percolation : le fluide traverse les lits tassés et les grilles avec les flux de Darcy de la dernière projection
        self.percolate(dt);
        // sources / puits : composition absorbée par les téléporteurs, puis remise à zéro des cellules épinglées
        if self.has_extras() {
            self.absorb_teleporters();
            self.pin_reset();
        }
        // 4. conduction + changement de phase sur l'état advecté
        self.probe("advection");
        self.refresh();
        // grains : sédimentation (dérive par rapport au fluide) sur l'état advecté
        if self.has_granular() {
            self.settle_grains(dt);
            // réactions en volume des grains (pyrolyse), à la température des grains
            self.pyrolysis_step(dt);
            self.refresh();
        }
        self.apply_thermal_sources(dt);
        self.probe("refresh + sources");
        lap(&mut prof, P_REFRESH);
        self.update_flame_factor(dt);
        // flammes : conduction, diffusion et chimie intercalées en sous-pas (voir `flame_thermo_substeps`)
        // (la prédiction du prédicteur-correcteur, qui ne sert qu'à estimer u^{n+1}, se contente d'un sous-pas)
        let kth = if self.flames_active() && !self.pc_predicting { self.flame_thermo_substeps.max(1) } else { 1 };
        let h = dt / kth as f64;
        // (chimie sous-cyclée dans les seules cellules de flamme, F > 1 ; les autres — gaz brûlés à l'équilibre, l'essentiel du coût
        // de la chimie — réagissent une fois sur dt)
        use super::mp_species::ChemSel;
        for s in 0..kth {
            self.conduct(h);
            lap(&mut prof, P_CONDUCT);
            self.diffuse_species(h);
            lap(&mut prof, P_SPECIES);
            self.react_step_sel(h, if kth > 1 { ChemSel::Flame } else { ChemSel::All }, s == 0);
            lap(&mut prof, P_CHEM);
        }
        if kth > 1 {
            self.react_step_sel(dt, ChemSel::NonFlame, false);
            lap(&mut prof, P_CHEM);
        }
        self.probe("conduction + diffusion + chimie");
        self.radiate(dt);
        self.probe("rayonnement");
        lap(&mut prof, P_CHEM);
        self.phase_change_step(dt);
        self.probe("changement de phase");
        // échange entre groupes d'une cellule mixte : en dernier, juste avant la projection, qui voit ainsi l'état qu'il produit
        // (son couplage χ entre dans la fermeture et la répartition du travail de pression)
        self.exchange_heat(dt);
        self.exchange_grains(dt);
        // gel / fusion (eau ↔ glace) sur les températures de groupe après les échanges ; front de dendrites de l'eau surfondue
        self.fusion_step(dt);
        self.probe("échange entre groupes");
        lap(&mut prof, P_PHASE);
        // 5-6. pression
        self.refresh_after_thermo();
        self.probe("refresh");
        lap(&mut prof, P_REFRESH);
        // viscosité implicite (frottement des grains tassés, fluides très visqueux) sur la vitesse prédite, avant la projection
        if self.viscous {
            self.apply_implicit_viscosity(dt, &mut us, &mut vs);
            lap(&mut prof, P_FORCES);
        }
        let (mut ua, mut va) = match &self.adv_vel { Some((a, b)) => (a.clone(), b.clone()), None => (un.clone(), vn.clone()) };
        // (divergence de la vitesse qui a transporté les masses : débit relatif aux corps en mouvement)
        if let Some((eu, ev)) = &attached {
            ua.iter_mut().zip(eu).for_each(|(x, e)| *x -= *e);
            va.iter_mut().zip(ev).for_each(|(x, e)| *x -= *e);
        }
        self.last_pressure_iters = self.solve_pressure(dt, &us, &vs, &ua, &va, &err0);
        self.probe("pression");
        lap(&mut prof, P_PRESSURE);
        // 7. travail de pression et état final
        self.pressure_work(dt);
        self.probe("travail de pression");
        self.fill_cavities();
        self.probe("cavités");
        self.refresh();
        self.probe("refresh final");
        lap(&mut prof, P_REFRESH);
        prof.steps += 1;
        self.profile = prof;
        self.time += dt;
        self.last_dt = dt;
        self.step_count += 1;
        dt
    }

    fn probe(&self, stage: &str) {
        let c = PROBE_CELL.load(std::sync::atomic::Ordering::Relaxed);
        if c == usize::MAX {
            return;
        }
        let mut line = format!("  {stage:<24} p {:>11.4e} vol−1 {:>+10.3e} |u| {:>9.3e}", self.p[c], self.cell_volume(c, self.p[c]) - 1.0, self.face_speed(c));
        for g in 0..self.ngroups() {
            let m = self.group_mass(g, c);
            if m > 0.0 {
                line += &format!(" | g{g} m {m:.3e} T {:.1} e/m {:.4e}", self.tg[g][c], self.eg[g][c] / m);
            }
        }
        eprintln!("{line}");
    }
    pub(crate) fn face_speed(&self, c: usize) -> f64 {
        let (i, j) = (c % self.nx, c / self.nx);
        [self.u[self.uidx(i, j)], self.u[self.uidx(i + 1, j)], self.v[self.vidx(i, j)], self.v[self.vidx(i, j + 1)]].iter().fold(0.0f64, |a, &b| a.max(b.abs()))
    }

    /// prédicteur-correcteur demandé (option, voir le champ `predictor_corrector`)
    pub fn use_predictor_corrector(&self) -> bool {
        self.predictor_corrector
    }

    /// flammes épaissies actives (miroir de `MultiphaseGpu::flames_active`) : chimie, réaction à δ_L > 0, F_max > 1
    pub(crate) fn flames_active(&self) -> bool {
        let fmax = self
            .reg
            .reactions
            .iter()
            .filter(|r| r.flame_thickness > 0.0)
            .map(|r| self.flame_resolution * self.dx / r.flame_thickness)
            .fold(1.0f64, f64::max);
        self.chemistry && fmax > 1.0
    }

    /// advection masse / énergie avec la vitesse imposée par le correcteur (sinon la vitesse courante u^n)
    /// `advect_mass_energy_with` avec le débit RELATIF aux corps en mouvement (`attached` : débit du fluide qui leur est collé)
    fn advect_mass_energy_rel(&mut self, dt: f64, attached: Option<&(Vec<f64>, Vec<f64>)>) {
        let Some((eu, ev)) = attached else {
            self.advect_mass_energy_with(dt);
            return;
        };
        let sub = |a: &mut Vec<f64>, e: &Vec<f64>, k: f64| a.iter_mut().zip(e).for_each(|(x, y)| *x -= k * *y);
        let (u0, v0) = (self.u.clone(), self.v.clone());
        sub(&mut self.u, eu, 1.0);
        sub(&mut self.v, ev, 1.0);
        if let Some((a, b)) = &mut self.adv_vel {
            sub(a, eu, 1.0);
            sub(b, ev, 1.0);
        }
        self.advect_mass_energy_with(dt);
        self.u = u0;
        self.v = v0;
        if let Some((a, b)) = &mut self.adv_vel {
            sub(a, eu, -1.0);
            sub(b, ev, -1.0);
        }
    }

    fn advect_mass_energy_with(&mut self, dt: f64) {
        if let Some((ua, va)) = self.adv_vel.take() {
            let un = std::mem::replace(&mut self.u, ua);
            let vn = std::mem::replace(&mut self.v, va);
            self.advect_mass_energy(dt);
            let ua = std::mem::replace(&mut self.u, un);
            let va = std::mem::replace(&mut self.v, vn);
            self.adv_vel = Some((ua, va));
        } else {
            self.advect_mass_energy(dt);
        }
    }

    /// Met à jour ρ, α, μ, k à partir des masses courantes SANS recalculer T depuis e (T déjà à jour après conduction).
    fn refresh_after_thermo(&mut self) {
        self.refresh();
    }

    pub fn run(&mut self, t_end: f64) -> usize {
        let mut n = 0;
        while self.time < t_end && !self.diverged {
            self.step();
            n += 1;
        }
        n
    }
}
