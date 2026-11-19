//! Boucle de temps du solveur multiphasique.
//!
//! Ordre d'un pas :
//!  1. refresh (T, α, ρ, μ, k depuis m, e, p)         5. refresh (ρ de l'état advecté)
//!  2. vitesse : advection MacCormack + forces → u*   6. pression implicite → pⁿ⁺¹, uⁿ⁺¹
//!  3. (m_k, e) : advection PLIC conservative (uⁿ)    7. travail −p∇·u, refresh final
//!  4. conduction, changement de phase (source de volume)
//! dt est limité par l'advection (CFL), la gravité, la tension de surface (Brackbill) et la viscosité — PAS par l'acoustique.

use super::multiphase::MultiphaseSim;

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

    /// Un pas de temps. Retourne dt.
    pub fn step(&mut self) -> f64 {
        self.refresh();
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
        // 4. conduction + changement de phase sur l'état advecté
        self.refresh();
        self.conduct(dt);
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
        while self.time < t_end {
            self.step();
            n += 1;
        }
        n
    }
}
