//! Boucle de temps du solveur multiphasique.
//!
//! Ordre d'un pas :
//!  1. refresh (T, α, ρ, μ, k depuis m, e, p)         5. refresh (ρ de l'état advecté)
//!  2. vitesse : advection MacCormack + forces → u*   6. pression implicite → pⁿ⁺¹, uⁿ⁺¹
//!  3. (m_k, e) : advection PLIC conservative (uⁿ)    7. travail −p∇·u, refresh final
//!  4. conduction, diffusion des espèces miscibles, réactions (accroche), changement de phase (source de volume)
//! dt est limité par l'advection (CFL), la gravité, la tension de surface (Brackbill) et la viscosité — PAS par l'acoustique.

use super::multiphase::MultiphaseSim;

struct Snapshot {
    mass: Vec<Vec<f64>>, alpha: Vec<Vec<f64>>, e: Vec<f64>, temp: Vec<f64>, p: Vec<f64>, u: Vec<f64>, v: Vec<f64>,
    rho_mix: Vec<f64>, mu_cell: Vec<f64>, k_cell: Vec<f64>, boundary_flux: Vec<f64>, transferred: Vec<f64>,
    energy_input: f64, tp_mix: super::mp_cells::TpMix,
    mfx: Vec<f64>, mfy: Vec<f64>, m_tot_old: Vec<f64>,
    time: f64, last_dt: f64, last_pressure_iters: usize, step_count: usize,
}

impl MultiphaseSim {
    pub fn choose_dt(&self) -> f64 {
        let n = self.nx * self.ny;
        let dx = self.dx;
        let umax = self.max_speed();
        let mut dt = self.dt_max;
        dt = dt.min(self.cfl * dx / umax.max(1e-9));
        let g = (self.gravity[0].powi(2) + self.gravity[1].powi(2)).sqrt();
        if g > 0.0 {
            dt = dt.min(0.5 * (dx / g).sqrt());
        }
        let (mut rmin, mut rmax, mut sig) = (f64::MAX, 0.0f64, 0.0f64);
        for c in 0..n {
            if self.is_solid(c) {
                continue;
            }
            rmin = rmin.min(self.rho_mix[c]);
            rmax = rmax.max(self.rho_mix[c]);
            if self.mu_cell[c] > 0.0 {
                dt = dt.min(0.2 * self.rho_mix[c] * dx * dx / self.mu_cell[c]);
            }
        }
        for ph in &self.reg.phases {
            sig = sig.max(ph.sigma);
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
            for k in 0..self.mass.len() {
                let m = self.mass[k][c];
                if !m.is_finite() || m < -1e-6 {
                    return false;
                }
            }
        }
        true
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            mass: self.mass.clone(), alpha: self.alpha.clone(), e: self.e.clone(), temp: self.temp.clone(), p: self.p.clone(),
            u: self.u.clone(), v: self.v.clone(), rho_mix: self.rho_mix.clone(), mu_cell: self.mu_cell.clone(), k_cell: self.k_cell.clone(),
            boundary_flux: self.boundary_flux.clone(), transferred: self.transferred.clone(),
            energy_input: self.energy_input, tp_mix: self.tp_mix.clone(),
            mfx: self.mfx.clone(), mfy: self.mfy.clone(), m_tot_old: self.m_tot_old.clone(),
            time: self.time, last_dt: self.last_dt, last_pressure_iters: self.last_pressure_iters, step_count: self.step_count,
        }
    }

    fn restore(&mut self, s: &Snapshot) {
        self.mass.clone_from(&s.mass); self.alpha.clone_from(&s.alpha); self.e.clone_from(&s.e); self.temp.clone_from(&s.temp);
        self.p.clone_from(&s.p); self.u.clone_from(&s.u); self.v.clone_from(&s.v); self.rho_mix.clone_from(&s.rho_mix);
        self.mu_cell.clone_from(&s.mu_cell); self.k_cell.clone_from(&s.k_cell);
        self.boundary_flux.clone_from(&s.boundary_flux); self.transferred.clone_from(&s.transferred);
        self.energy_input = s.energy_input; self.tp_mix.clone_from(&s.tp_mix);
        self.mfx.clone_from(&s.mfx); self.mfy.clone_from(&s.mfy); self.m_tot_old.clone_from(&s.m_tot_old);
        self.time = s.time; self.last_dt = s.last_dt; self.last_pressure_iters = s.last_pressure_iters; self.step_count = s.step_count;
    }

    /// Un pas de temps protégé : si l'état obtenu est invalide (NaN, T aberrante, masse négative, vitesse > speed_limit) le pas est
    /// annulé et refait avec dt/4, dt/16, … (jusqu'à dt/1024). Si rien ne passe, l'état est restauré, `diverged` passe à true et
    /// 0.0 est retourné : l'appelant doit s'arrêter (au lieu de propager des NaN dans tout le domaine). Retourne dt.
    pub fn step(&mut self) -> f64 {
        if self.diverged {
            return 0.0;
        }
        let snap = self.snapshot();
        let dt_max0 = self.dt_max;
        let dt = self.step_raw();
        if self.state_is_sane() {
            return dt;
        }
        let mut dt_try = dt;
        for _ in 0..5 {
            self.restore(&snap);
            dt_try *= 0.25;
            self.dt_max = dt_max0.min(dt_try);
            self.rollbacks += 1;
            self.step_raw();
            if self.state_is_sane() {
                self.dt_max = dt_max0;
                return self.last_dt;
            }
        }
        self.restore(&snap);
        self.dt_max = dt_max0;
        self.diverged = true;
        0.0
    }

    /// Un pas de temps sans protection. Retourne dt.
    pub(crate) fn step_raw(&mut self) -> f64 {
        self.refresh();
        // cellules spéciales / entrées : faces à vitesse imposée, cellules épinglées remises à leur composition cible
        self.prepare_special();
        if self.has_extras() {
            self.apply_prescribed_faces();
            self.pin_reset();
            self.refresh();
        }
        let dt = self.choose_dt();
        let (un, vn) = (self.u.clone(), self.v.clone());
        // 2. prédiction de vitesse
        let (us, vs);
        if self.conservative_momentum {
            // forces évaluées sur l'état n, puis masses/énergie, puis quantité de mouvement avec les MÊMES flux de masse
            let mut fx = vec![0.0; self.u.len()];
            let mut fy = vec![0.0; self.v.len()];
            self.add_forces(dt, &mut fx, &mut fy);
            self.advect_mass_energy(dt);
            let (mut ua, mut va) = self.advect_momentum();
            for (a, f) in ua.iter_mut().zip(&fx) { *a += *f; }
            for (a, f) in va.iter_mut().zip(&fy) { *a += *f; }
            us = ua;
            vs = va;
        } else {
            let (mut a, mut b) = self.advect_velocity(dt);
            self.add_forces(dt, &mut a, &mut b);
            self.advect_mass_energy(dt);
            us = a;
            vs = b;
        }
        // sources / puits : composition absorbée par les téléporteurs, puis remise à zéro des cellules épinglées
        if self.has_extras() {
            self.absorb_teleporters();
            self.pin_reset();
        }
        // 4. conduction + changement de phase sur l'état advecté
        self.refresh();
        self.apply_thermal_sources(dt);
        self.conduct(dt);
        self.diffuse_species(dt);
        self.react_step(dt);
        self.phase_change_step(dt);
        // 5-6. pression
        self.refresh_after_thermo();
        self.last_pressure_iters = self.solve_pressure(dt, &us, &vs, &un, &vn);
        // 7. travail de pression et état final
        self.pressure_work(dt);
        self.refresh();
        self.time += dt;
        self.last_dt = dt;
        self.step_count += 1;
        dt
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
