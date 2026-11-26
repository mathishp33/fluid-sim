//! Chaleur des corps rigides (solides mobiles, étape 5 de SOLID.md) : une température par corps, de capacité M c_p — valable pour un
//! corps dont le nombre de Biot est faible (métal) —, qui échange avec le fluide des mailles que traverse son bord.

use super::mp_bodies::boundary_pieces;
use super::multiphase::MultiphaseSim;

impl MultiphaseSim {
    /// Échange de chaleur des corps avec le fluide qui les borde, sur un pas dt.
    ///
    /// À travers le morceau de bord de longueur L dans la maille c, la conductance (par mètre de profondeur) vaut L k_h / dx, k_h
    /// moyenne harmonique des conductivités du corps et du fluide : une demi-maille de chaque côté, comme entre une maille solide et sa
    /// voisine fluide (`conduct`). Elle est partagée entre les groupes de fluide de la maille au prorata de leur part du volume ouvert.
    /// Implicite (Euler arrière, corps et mailles ensemble, forme close) : la trace de fluide d'une maille presque couverte, de capacité
    /// quasi nulle, prend la température du corps sans sous-pas ni instabilité ; l'énergie que cède le corps est exactement celle que
    /// reçoit le fluide.
    pub(crate) fn exchange_body_heat(&mut self, dt: f64) {
        if !self.conduction || !(dt > 0.0) || !self.has_bodies() {
            return;
        }
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let dx2 = dx * dx;
        let ng = self.groups.len();
        let harm = |a: f64, b: f64| if a + b > 0.0 { 2.0 * a * b / (a + b) } else { 0.0 };
        let mut touched = vec![false; nx * ny];
        for k in 0..self.bodies.len() {
            let (kb, cap_b, tb, poly) = {
                let b = &self.bodies[k];
                let mat = &self.reg.solids[b.material];
                (mat.conductivity, b.density(mat.rho) * b.shape.area() * mat.cp, b.temp, b.world_vertices())
            };
            if !(cap_b > 0.0) {
                continue;
            }
            // longueur du bord dans chaque maille fluide (un morceau posé sur une ligne de la grille compte pour la maille extérieure)
            let mut len: Vec<(usize, f64)> = vec![];
            for (c, p, q) in boundary_pieces(&poly, nx, ny, dx) {
                if self.is_solid(c) {
                    continue;
                }
                let l = (q[0] - p[0]).hypot(q[1] - p[1]);
                match len.iter_mut().find(|e| e.0 == c) {
                    Some(e) => e.1 += l,
                    None => len.push((c, l)),
                }
            }
            // nœuds (maille, groupe) : conductance G vers le corps, capacité C, température
            let mut nodes: Vec<(usize, usize, f64, f64, f64)> = vec![];
            for (c, l) in len {
                let open: f64 = (0..ng).filter(|&g| self.group_mass(g, c) > 0.0).map(|g| self.group_alpha(g, c)).sum();
                if !(open > 0.0) {
                    continue;
                }
                for g in 0..ng {
                    if self.group_mass(g, c) <= 0.0 {
                        continue;
                    }
                    let t = self.tg[g][c];
                    let gc = l * (self.group_alpha(g, c) / open) * harm(kb, self.group_k(g, c)) / dx;
                    let cc = self.group_cap_at(g, c, t) * dx2;
                    if gc > 0.0 && cc > 0.0 {
                        nodes.push((c, g, gc, cc, t));
                    }
                }
            }
            if nodes.is_empty() {
                continue;
            }
            // T_b' (C_b + Σ w) = C_b T_b + Σ w T, w = G dt / (1 + G dt / C) ; chaque nœud reçoit alors w (T_b' − T)
            let (mut num, mut den) = (cap_b * tb, cap_b);
            for &(_, _, gc, cc, t) in &nodes {
                let w = dt * gc / (1.0 + gc * dt / cc);
                num += w * t;
                den += w;
            }
            let tb1 = num / den;
            for &(c, g, gc, cc, t) in &nodes {
                let w = dt * gc / (1.0 + gc * dt / cc);
                self.eg[g][c] += w * (tb1 - t) / dx2;
                touched[c] = true;
            }
            self.bodies[k].temp = tb1;
        }
        self.sync_cells(|c| touched[c]);
    }

    /// énergie thermique du corps k par mètre de profondeur, M c_p T (J/m) : bilans
    pub(crate) fn body_heat(&self, k: usize) -> f64 {
        let b = &self.bodies[k];
        let mat = &self.reg.solids[b.material];
        b.density(mat.rho) * b.shape.area() * mat.cp * b.temp
    }
}

#[cfg(test)]
mod tests {
    use crate::simulation::multi::multiphase::MultiphaseSim;

    fn scene(text: &str) -> MultiphaseSim {
        let p = crate::simulation::multi::mp_script::parse_all(text).unwrap_or_else(|e| panic!("{e:?}")).remove(0);
        crate::simulation::multi::mp_experiment::initial_state(&p, std::path::Path::new(".")).unwrap_or_else(|e| panic!("{e}")).0
    }

    /// énergie totale par mètre de profondeur : mailles (fluide et solides) et corps
    fn total_energy(s: &MultiphaseSim) -> f64 {
        let dx2 = s.dx * s.dx;
        s.e.iter().map(|e| e * dx2).sum::<f64>() + (0..s.bodies.len()).map(|k| s.body_heat(k)).sum::<f64>()
    }

    #[test]
    fn a_hot_body_cools_like_the_same_block_of_solid_cells() {
        // Sans pesanteur (conduction seule), eau à 20 °C : à gauche un corps carré d'acier de 1 cm aligné sur la grille, à droite le
        // même carré en mailles solides d'acier, tous deux à 80 °C. Le corps (une seule température) et le bloc (conduction interne,
        // k = 45 W/(m·K) : nombre de Biot ≈ 0.07) refroidissent ensemble ; l'énergie totale (eau, solide, corps) est conservée. La
        // cuve est ouverte en haut : une couche d'air y laisse l'eau se dilater sans qu'elle sorte (avec son énergie) par le bord.
        let mut s = scene(
            "départ scénario \"cuve vide\" grille 40 x 20 maille 2 mm gravité 0
             remplir rect (0, 0) (0.08, 0.036) avec \"eau\" à 20 °C
             corps rect (0.016, 0.014) (0.026, 0.024) \"acier\" à 80 °C
             solide rect (0.054, 0.014) (0.064, 0.024) \"acier\" à 80 °C
",
        );
        let block: Vec<usize> = (0..s.nx * s.ny).filter(|&c| s.is_solid(c)).collect();
        assert_eq!(block.len(), 25);
        let t0 = 293.15;
        let e0 = total_energy(&s);
        let h0 = s.body_heat(0);
        let mut worst = 0.0f64;
        for t_check in [1.0, 2.0, 4.0, 7.0, 10.0] {
            while s.time < t_check && !s.diverged {
                s.step();
            }
            assert!(!s.diverged && s.state_is_sane());
            let tb = s.bodies[0].temp - t0;
            let ts = block.iter().map(|&c| s.temp[c]).sum::<f64>() / block.len() as f64 - t0;
            worst = worst.max((tb / ts - 1.0).abs());
            eprintln!("  t {:.1} s : corps {:.3} K au-dessus de l'eau, bloc de mailles solides {:.3} K (écart {:+.2} %)", s.time, tb, ts, 100.0 * (tb / ts - 1.0));
        }
        let (tb, de, dh) = (s.bodies[0].temp - t0, total_energy(&s) - e0, s.body_heat(0) - h0);
        eprintln!(
            "corps chaud et bloc de mailles solides : écart relatif au plus {:.2} % ; corps refroidi de 60 à {tb:.2} K ; énergie totale à {:.1e} de la chaleur cédée ; {} pas",
            100.0 * worst,
            (de / dh).abs(),
            s.step_count
        );
        assert!(tb < 0.9 * 60.0, "le corps ne refroidit pas : {tb} K");
        assert!(worst < 0.05, "le corps et le bloc ne refroidissent pas ensemble : écart {worst}");
        assert!((de / dh).abs() < 1e-3, "énergie : {de} J/m pour {dh} cédés");
    }

    #[test]
    fn a_hot_steel_disc_warms_the_water_which_rises_above_it() {
        // Disque d'acier fixe à 80 °C dans l'eau à 20 °C, sous la pesanteur : il cède sa chaleur, l'eau chauffée monte en panache
        // au-dessus de lui ; l'énergie totale est conservée (couche d'air en haut de la cuve ouverte, voir l'essai précédent).
        let mut s = scene(
            "départ scénario \"cuve vide\" grille 30 x 40 maille 2 mm
             remplir rect (0, 0) (0.06, 0.07) avec \"eau\" à 20 °C
             corps disque (0.03, 0.025) rayon 1 cm \"acier\" à 80 °C
",
        );
        let (nx, ny) = (s.nx, s.ny);
        let e0 = total_energy(&s);
        let h0 = s.body_heat(0);
        let mut last = s.bodies[0].temp;
        // refroidissement mesuré entre 4 et 8 s : h = C_b |dT_b/dt| / (π D (T_b − 20 °C))
        let mut at4 = None;
        while s.time < 8.0 && !s.diverged {
            s.step();
            assert!(s.bodies[0].temp <= last + 1e-9, "le corps se réchauffe");
            last = s.bodies[0].temp;
            if at4.is_none() && s.time >= 4.0 {
                at4 = Some((s.time, s.bodies[0].temp));
            }
        }
        assert!(!s.diverged && s.state_is_sane());
        // hauteur moyenne de la chaleur reçue par l'eau (mailles hors du corps) : au-dessus du centre du disque si l'eau chauffée monte
        let (mut w, mut wy) = (0.0, 0.0);
        for c in 0..nx * ny {
            if s.is_solid(c) || s.body_frac[c] > 0.0 || s.temp[c] <= 293.15 {
                continue;
            }
            let y = ((c / nx) as f64 + 0.5) * s.dx;
            w += s.temp[c] - 293.15;
            wy += (s.temp[c] - 293.15) * y;
        }
        let y_heat = wy / w;
        let (de, dh) = (total_energy(&s) - e0, s.body_heat(0) - h0);
        let (t4, tb4) = at4.unwrap();
        let tb = s.bodies[0].temp;
        let cap = s.body_heat(0) / tb;
        let d = 0.02;
        let h = cap * (tb4 - tb) / (s.time - t4) / (std::f64::consts::PI * d * (0.5 * (tb4 + tb) - 293.15));
        eprintln!(
            "disque chaud dans l'eau : {:.2} °C après {:.1} s ; chaleur reçue par l'eau centrée à y = {y_heat:.4} m (disque en 0.025) ; h = {h:.0} W/(m²·K) de 4 à 8 s (Churchill–Chu, régime établi : ≈ 1400) ; énergie totale à {:.1e} de la chaleur cédée ; {} pas, {} refaits",
            tb - 273.15,
            s.time,
            (de / dh).abs(),
            s.step_count,
            s.rollbacks
        );
        assert!(tb < 353.15 - 5.0, "le disque ne refroidit pas : {tb} K");
        // l'eau chauffée monte : panache lent, la couche limite thermique (≈ 0.5 mm) est plus fine que la maille (2 mm)
        assert!(y_heat > 0.025 + 0.002, "pas de panache : chaleur centrée à y = {y_heat}");
        assert!((de / dh).abs() < 1e-3, "énergie : {de} J/m pour {dh} cédés");
    }
}

