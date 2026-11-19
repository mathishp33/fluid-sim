//! Espèces : diffusion de Fick entre fluides MISCIBLES d'un même groupe, et point d'accroche des réactions chimiques.
//!
//! Diffusion : pour chaque groupe miscible (≥ 2 membres), flux de la masse partielle k entre deux cellules fluides voisines a, b :
//!   J_k = D · M̄ · (Y_k[a] − Y_k[b]) / dx   avec Y_k = m_k / M_groupe, M̄ = moyenne de M_groupe aux deux cellules.
//! Σ_k J_k = 0 : la masse du groupe n'est pas transportée (pas de variation de volume au premier ordre), seules les proportions
//! changent. Conservatif cellule à cellule ; l'énergie interne accompagne chaque espèce (e_int,k à la T de la cellule donneuse) :
//! l'énergie totale est conservée exactement. Sous-pas explicites (stabilité : D·ds/dx² ≤ 0.2).
//!
//! Réactions : `react_step` est le point d'accroche (vide tant qu'aucune réaction n'est définie) ; voir `phases::Reaction`.

use super::multiphase::MultiphaseSim;

impl MultiphaseSim {
    /// Diffusion de Fick des espèces dans les groupes miscibles. Sans groupe miscible (cas par défaut) : ne fait rien.
    pub(crate) fn diffuse_species(&mut self, dt: f64) {
        let groups: Vec<Vec<usize>> = self.reg.miscible_groups().into_iter().filter(|g| g.len() > 1).collect();
        if groups.is_empty() {
            return;
        }
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        for g in &groups {
            let d = g.iter().map(|&k| self.reg.phases[k].diffusivity).sum::<f64>() / g.len() as f64;
            if !(d > 0.0) {
                continue;
            }
            let nsub = ((dt * d / (0.2 * dx * dx)).ceil() as usize).clamp(1, 500);
            let ds = dt / nsub as f64;
            for _ in 0..nsub {
                let n = nx * ny;
                // fraction massique et masse du groupe par cellule
                let mg: Vec<f64> = (0..n).map(|c| if self.is_solid(c) { 0.0 } else { g.iter().map(|&k| self.mass[k][c].max(0.0)).sum() }).collect();
                let mut dm: Vec<Vec<f64>> = vec![vec![0.0; n]; g.len()];
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
                            for (q, &k) in g.iter().enumerate() {
                                let (ya, yb) = (self.mass[k][a].max(0.0) / mg[a], self.mass[k][b].max(0.0) / mg[b]);
                                // masse transférée de a vers b pendant ds (kg/m³ de cellule)
                                let mut flux = d * mbar * (ya - yb) / (dx * dx) * ds;
                                // jamais plus que le quart de ce que la cellule donneuse contient
                                let donor = if flux > 0.0 { a } else { b };
                                flux = flux.clamp(-0.25 * self.mass[k][b].max(0.0), 0.25 * self.mass[k][a].max(0.0));
                                let _ = donor;
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
                        self.mass[k][c] += dm[q][c];
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

    /// Point d'accroche des réactions chimiques (APRÈS la conduction, AVANT le changement de phase). Non implémenté : sans effet.
    /// Quand le moteur cinétique sera écrit, il lira `self.reg.reactions` (vitesse d'Arrhenius, transfert de masse réactifs → produits
    /// en conservant la masse totale ; l'énergie interne e est inchangée car les décalages e0 portent déjà la chaleur de réaction).
    pub(crate) fn react_step(&mut self, _dt: f64) {
        if self.reg.reactions.is_empty() {
            return;
        }
        // TODO(réactions) : cinétique d'Arrhenius ; voir phases::Reaction.
    }
}
