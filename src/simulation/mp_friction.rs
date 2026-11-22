//! Frottement intergranulaire (étape 3 de GRANULAIRE.md) : un tas de sable tient au lieu de s'étaler comme un liquide lourd.
//!
//! Modèle de Coulomb continu (Drucker–Prager régularisé, comme dans les simulations d'effondrement de colonnes granulaires) : dans
//! les cellules TASSÉES (fraction de grains α_s au-dessus de α_c = α_max − 0.1, montée linéaire jusqu'à α_max), une viscosité
//! μ_g = tan φ · p_s / (2 √(|D|² + δ²)) — la contrainte de cisaillement vaut tan φ · p_s dès que le matériau s'écoule (|D| ≫ δ) et le
//! matériau est quasi rigide sous ce seuil. |D| = √(½ D:D) (taux de déformation, état du début du pas). Cellules concernées : tassées,
//! ou partiellement remplies posées sur une cellule tassée (surface d'un tas), au prorata α_s / α_max.
//! Pression de contact p_s : pression du mélange moins celle du fluide des pores (comme Lagrée, Staron & Popinet 2011 avec la pression
//! incompressible) — contrainte moyenne, nulle au bord libre ; le poids des grains au-dessus rendrait le sable cohésif.
//!
//! Ces viscosités sont énormes (jusqu'à ~1e6 Pa·s) : elles sont traitées IMPLICITEMENT, (ρ_f/dt)(u − u_f) = ∇·[μ_g(∇u + ∇uᵀ)],
//! par gradient conjugué préconditionné (Jacobi) — opérateur symétrique défini positif. La vitesse u_f voit déjà le gradient de la
//! pression du pas précédent (u_f = u* − dt ∇pⁿ/ρ_f, rendu ensuite) : le frottement équilibre la force NETTE (pesanteur moins poussée
//! hydrostatique), sinon un tas au repos glisserait d'un dt·g par pas. Parois du domaine : non-glissement pour la contrainte de
//! frottement (le tas tient sur le sol) ; les fluides gardent le glissement libre. μ_g est plafonnée à `FRICTION_K_MAX` ρ dx²/dt
//! (borne du nombre d'itérations ; fluage résiduel d'un tas au repos ~ 1e-3 de sa taille par seconde).

use super::multiphase::{Bc, MultiphaseSim};

/// taux de déformation de régularisation δ (1/s)
const FRICTION_DELTA: f64 = 1e-3;
/// plafond de μ_g dt / (ρ dx²)
const FRICTION_K_MAX: f64 = 1000.0;
/// tolérance relative du gradient conjugué
const FRICTION_TOL: f64 = 1e-6;

impl MultiphaseSim {
    /// au moins une phase granulaire avec frottement (et de la gravité pour peser) ?
    pub fn has_friction(&self) -> bool {
        self.viscous && self.reg.phases.iter().any(|p| p.is_granular() && p.friction_angle > 0.0) && self.gravity.iter().any(|&g| g != 0.0)
    }

    /// viscosité de frottement μ_g (Pa·s) par cellule (0 hors du sable tassé)
    pub(crate) fn friction_viscosity(&self, dt: f64) -> Vec<f64> {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let n = nx * ny;
        let np = self.reg.phases.len();
        let gn = (self.gravity[0].powi(2) + self.gravity[1].powi(2)).sqrt();
        let mut mu = vec![0.0; n];
        if gn == 0.0 {
            return mu;
        }
        // grains de la cellule : fraction, masse volumique, tan φ moyen, compaction ; fluide porteur : masse volumique
        let mut a_s = vec![0.0; n];
        let mut w = vec![0.0; n]; // poids déjaugé des grains de la cellule par unité de surface (Pa) : haut du massif au bord du domaine
        let mut tanphi = vec![0.0; n];
        let mut ramp = vec![0.0; n];
        let mut amx = vec![1.0; n];
        let mut rhof = vec![0.0; n];
        for c in 0..n {
            if self.is_solid(c) {
                continue;
            }
            let (mut as_, mut ms, mut af, mut mf, mut tp, mut am) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for k in 0..np {
                let a = self.alpha[k][c].max(0.0);
                let ph = &self.reg.phases[k];
                if ph.is_granular() {
                    as_ += a;
                    ms += self.mass[k][c].max(0.0);
                    tp += a * ph.friction_angle.to_radians().tan();
                    am += a * ph.alpha_max;
                } else {
                    af += a;
                    mf += self.mass[k][c].max(0.0);
                }
            }
            let rho_f = if af > 1e-12 { mf / af } else { 0.0 };
            rhof[c] = rho_f;
            if as_ <= 1e-9 {
                continue;
            }
            let rho_s = ms / as_;
            a_s[c] = as_;
            w[c] = (rho_s - rho_f).max(0.0) * as_ * gn * dx;
            tanphi[c] = tp / as_;
            let amax = am / as_;
            amx[c] = amax;
            let ac = amax - 0.1;
            ramp[c] = ((as_ - ac) / (amax - ac)).clamp(0.0, 1.0);
        }
        // cellules en contact : tassées (α_s ≥ α_c), ou posées sur une cellule tassée (surface d'un tas, cellule partiellement remplie) ;
        // part de la cellule occupée par le matériau granulaire tassé : α_s / α_max
        let vertical = self.gravity[1].abs() >= self.gravity[0].abs();
        let down = if vertical { self.gravity[1] < 0.0 } else { self.gravity[0] < 0.0 };
        // voisine « en dessous » (aval de la gravité, axe dominant)
        let below = |c: usize| -> Option<usize> {
            let (i, j) = (c % nx, c / nx);
            if vertical {
                if down { (j > 0).then(|| c - nx) } else { (j + 1 < ny).then(|| c + nx) }
            } else if down {
                (i > 0).then(|| c - 1)
            } else {
                (i + 1 < nx).then(|| c + 1)
            }
        };
        let mut frac = vec![0.0; n];
        for c in 0..n {
            if a_s[c] <= 1e-9 || self.is_solid(c) {
                continue;
            }
            let rest = below(c).map_or(true, |b| self.is_solid(b) || ramp[b] > 0.0);
            if ramp[c] > 0.0 || rest {
                frac[c] = (a_s[c] / amx[c]).clamp(0.0, 1.0);
            }
        }
        // pression de contact p_s = pression du mélange (pas précédent : contrainte MOYENNE, nulle au bord libre d'une falaise) moins
        // celle du fluide des pores (hydrostatique depuis le haut du massif), le long de la gravité (axe dominant). Le poids des grains
        // au-dessus (contrainte VERTICALE) rendrait le sable cohésif : une falaise verticale tiendrait dès que tan φ > 1/2.
        let mut ps = vec![0.0; n];
        let (lines, len) = if vertical { (nx, ny) } else { (ny, nx) };
        for l in 0..lines {
            let mut pf = f64::NAN; // pression du fluide des pores
            let mut prev: Option<usize> = None;
            for t in 0..len {
                let s_ = if down { len - 1 - t } else { t };
                let c = if vertical { self.idx(l, s_) } else { self.idx(s_, l) };
                if self.is_solid(c) || frac[c] <= 0.0 {
                    pf = f64::NAN;
                    prev = if self.is_solid(c) { None } else { Some(c) };
                    continue;
                }
                if pf.is_nan() {
                    // haut du massif : fluide de la cellule du dessus, ou demi-poids de la cellule au bord du domaine
                    pf = match prev { Some(q) => self.p[q] + 0.5 * (rhof[q] + rhof[c]) * gn * dx, None => self.p[c] - 0.5 * w[c] };
                } else if let Some(q) = prev {
                    pf += 0.5 * (rhof[q] + rhof[c]) * gn * dx;
                }
                ps[c] = (self.p[c] - pf).max(0.0);
                prev = Some(c);
            }
        }
        // taux de déformation |D| = √(½ D:D) au centre (vitesses du début du pas)
        let (u, v) = (&self.u, &self.v);
        let ui = |i: usize, j: usize| u[j * (nx + 1) + i];
        let vi = |i: usize, j: usize| v[j * nx + i];
        for j in 0..ny {
            for i in 0..nx {
                let c = self.idx(i, j);
                if frac[c] <= 0.0 || self.is_solid(c) {
                    continue;
                }
                let dxx = (ui(i + 1, j) - ui(i, j)) / dx;
                let dyy = (vi(i, j + 1) - vi(i, j)) / dx;
                // ∂u/∂y et ∂v/∂x centrés (décentrés au bord), moyennés sur les deux faces de la cellule
                let (jm, jp) = (j.saturating_sub(1), (j + 1).min(ny - 1));
                let (im, ip) = (i.saturating_sub(1), (i + 1).min(nx - 1));
                let dudy = 0.5 * ((ui(i, jp) - ui(i, jm)) + (ui(i + 1, jp) - ui(i + 1, jm))) / ((jp - jm).max(1) as f64 * dx);
                let dvdx = 0.5 * ((vi(ip, j) - vi(im, j)) + (vi(ip, j + 1) - vi(im, j + 1))) / ((ip - im).max(1) as f64 * dx);
                let dxy = 0.5 * (dudy + dvdx);
                let d2 = 0.5 * (dxx * dxx + dyy * dyy + 2.0 * dxy * dxy);
                let m = frac[c] * tanphi[c] * ps[c] / (2.0 * (d2 + FRICTION_DELTA * FRICTION_DELTA).sqrt());
                let cap = FRICTION_K_MAX * self.rho_mix[c] * dx * dx / dt;
                mu[c] = m.min(cap);
            }
        }
        mu
    }

    /// Frottement implicite sur la vitesse prédite (us, vs) pendant dt (voir l'en-tête du module). À appeler juste avant la projection,
    /// `self.p` étant encore la pression du pas précédent. Renvoie le nombre d'itérations du gradient conjugué.
    pub(crate) fn apply_friction(&mut self, dt: f64, us: &mut [f64], vs: &mut [f64]) -> usize {
        let mu = self.friction_viscosity(dt);
        if !mu.iter().any(|&m| m > 0.0) {
            self.last_friction_iters = 0;
            return 0;
        }
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let (nu, nv) = ((nx + 1) * ny, nx * (ny + 1));
        let dx2 = dx * dx;
        // viscosité aux nœuds (i, j) ∈ [0, nx] × [0, ny] : moyenne des cellules voisines ; nulle contre un solide (glissement libre) et
        // sur un bord ouvert ; sur une paroi du domaine, celle des deux cellules adjacentes (non-glissement)
        let mut mn = vec![0.0; (nx + 1) * (ny + 1)];
        let wall = |side: usize| matches!(self.bc[side], Bc::Wall);
        for j in 0..=ny {
            for i in 0..=nx {
                let mut cells = vec![];
                for (ci, cj) in [(i as isize - 1, j as isize - 1), (i as isize, j as isize - 1), (i as isize - 1, j as isize), (i as isize, j as isize)] {
                    if ci >= 0 && cj >= 0 && (ci as usize) < nx && (cj as usize) < ny {
                        cells.push(self.idx(ci as usize, cj as usize));
                    }
                }
                if cells.iter().any(|&c| self.is_solid(c)) {
                    continue;
                }
                let on_l = i == 0;
                let on_r = i == nx;
                let on_b = j == 0;
                let on_t = j == ny;
                if (on_l || on_r) && (on_b || on_t) {
                    continue; // coin
                }
                if (on_l && !wall(0)) || (on_r && !wall(1)) || (on_b && !wall(2)) || (on_t && !wall(3)) {
                    continue;
                }
                mn[j * (nx + 1) + i] = cells.iter().map(|&c| mu[c]).sum::<f64>() / cells.len() as f64;
            }
        }
        // inconnues : faces intérieures entre deux cellules fluides
        let mut ku = vec![false; nu];
        let mut rho_u = vec![0.0; nu];
        for j in 0..ny {
            for i in 1..nx {
                let (a, b) = (self.idx(i - 1, j), self.idx(i, j));
                if !self.is_solid(a) && !self.is_solid(b) {
                    let f = j * (nx + 1) + i;
                    ku[f] = true;
                    rho_u[f] = self.face_rho_x(i, j);
                }
            }
        }
        let mut kv = vec![false; nv];
        let mut rho_v = vec![0.0; nv];
        for j in 1..ny {
            for i in 0..nx {
                let (a, b) = (self.idx(i, j - 1), self.idx(i, j));
                if !self.is_solid(a) && !self.is_solid(b) {
                    let f = j * nx + i;
                    kv[f] = true;
                    rho_v[f] = self.face_rho_y(i, j);
                }
            }
        }
        // force de frottement F(w) = ∇·[μ_g(∇w + ∇wᵀ)] sur les faces inconnues (parois : w = 0)
        let force = |wu: &[f64], wv: &[f64], fu: &mut [f64], fv: &mut [f64]| {
            let ui = |i: usize, j: usize| wu[j * (nx + 1) + i];
            let vi = |i: usize, j: usize| wv[j * nx + i];
            let mut txx = vec![0.0; nx * ny];
            let mut tyy = vec![0.0; nx * ny];
            for j in 0..ny {
                for i in 0..nx {
                    let c = j * nx + i;
                    if mu[c] > 0.0 {
                        txx[c] = 2.0 * mu[c] * (ui(i + 1, j) - ui(i, j)) / dx;
                        tyy[c] = 2.0 * mu[c] * (vi(i, j + 1) - vi(i, j)) / dx;
                    }
                }
            }
            let mut txy = vec![0.0; (nx + 1) * (ny + 1)];
            for j in 0..=ny {
                for i in 0..=nx {
                    let m = mn[j * (nx + 1) + i];
                    if m <= 0.0 {
                        continue;
                    }
                    // vitesse tangentielle nulle à la paroi : ∂u/∂y = ±2u/dx (face à dx/2 de la paroi)
                    let dudy = if j == 0 { 2.0 * ui(i, 0) / dx } else if j == ny { -2.0 * ui(i, ny - 1) / dx } else { (ui(i, j) - ui(i, j - 1)) / dx };
                    let dvdx = if i == 0 { 2.0 * vi(0, j) / dx } else if i == nx { -2.0 * vi(nx - 1, j) / dx } else { (vi(i, j) - vi(i - 1, j)) / dx };
                    // sur une paroi horizontale, v (face de paroi) est nulle : ∂v/∂x = 0 ; idem ∂u/∂y sur une paroi verticale
                    let (dudy, dvdx) = if j == 0 || j == ny { (dudy, 0.0) } else if i == 0 || i == nx { (0.0, dvdx) } else { (dudy, dvdx) };
                    txy[j * (nx + 1) + i] = m * (dudy + dvdx);
                }
            }
            for j in 0..ny {
                for i in 1..nx {
                    let f = j * (nx + 1) + i;
                    fu[f] = if ku[f] { (txx[j * nx + i] - txx[j * nx + i - 1]) / dx + (txy[(j + 1) * (nx + 1) + i] - txy[j * (nx + 1) + i]) / dx } else { 0.0 };
                }
            }
            for j in 1..ny {
                for i in 0..nx {
                    let f = j * nx + i;
                    fv[f] = if kv[f] { (tyy[j * nx + i] - tyy[(j - 1) * nx + i]) / dx + (txy[j * (nx + 1) + i + 1] - txy[j * (nx + 1) + i]) / dx } else { 0.0 };
                }
            }
        };
        // diagonale de A = ρ/dt − F (préconditionneur de Jacobi)
        let node = |i: usize, j: usize| mn[j * (nx + 1) + i];
        let wallc = |on_wall: bool| if on_wall { 2.0 } else { 1.0 };
        let mut du = vec![1.0; nu];
        for j in 0..ny {
            for i in 1..nx {
                let f = j * (nx + 1) + i;
                if ku[f] {
                    let (a, b) = (j * nx + i - 1, j * nx + i);
                    du[f] = rho_u[f] / dt
                        + 2.0 * (mu[a] + mu[b]) / dx2
                        + (wallc(j + 1 == ny) * node(i, j + 1) + wallc(j == 0) * node(i, j)) / dx2;
                }
            }
        }
        let mut dv = vec![1.0; nv];
        for j in 1..ny {
            for i in 0..nx {
                let f = j * nx + i;
                if kv[f] {
                    let (a, b) = ((j - 1) * nx + i, j * nx + i);
                    dv[f] = rho_v[f] / dt
                        + 2.0 * (mu[a] + mu[b]) / dx2
                        + (wallc(i + 1 == nx) * node(i + 1, j) + wallc(i == 0) * node(i, j)) / dx2;
                }
            }
        }
        // vitesse qui équilibre déjà la pression du pas précédent : u_f = u* − dt ∇pⁿ / ρ_f
        let p = &self.p;
        let mut uf = us.to_vec();
        let mut vf = vs.to_vec();
        for j in 0..ny {
            for i in 1..nx {
                let f = j * (nx + 1) + i;
                if ku[f] {
                    uf[f] -= dt / rho_u[f] * (p[j * nx + i] - p[j * nx + i - 1]) / dx;
                }
            }
        }
        for j in 1..ny {
            for i in 0..nx {
                let f = j * nx + i;
                if kv[f] {
                    vf[f] -= dt / rho_v[f] * (p[j * nx + i] - p[(j - 1) * nx + i]) / dx;
                }
            }
        }
        // A δ = F(u_f), δ = 0 sur les faces fixes ; u = u_f + δ, donc u* += δ
        let (mut ru, mut rv) = (vec![0.0; nu], vec![0.0; nv]);
        force(&uf, &vf, &mut ru, &mut rv);
        let apply = |xu: &[f64], xv: &[f64], au: &mut [f64], av: &mut [f64]| {
            force(xu, xv, au, av);
            for f in 0..nu {
                au[f] = if ku[f] { rho_u[f] / dt * xu[f] - au[f] } else { 0.0 };
            }
            for f in 0..nv {
                av[f] = if kv[f] { rho_v[f] / dt * xv[f] - av[f] } else { 0.0 };
            }
        };
        let dot = |a: &[f64], b: &[f64], c: &[f64], d: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() + c.iter().zip(d).map(|(x, y)| x * y).sum::<f64>();
        let (mut xu, mut xv) = (vec![0.0; nu], vec![0.0; nv]);
        let (mut zu, mut zv): (Vec<f64>, Vec<f64>) = ((0..nu).map(|f| ru[f] / du[f]).collect(), (0..nv).map(|f| rv[f] / dv[f]).collect());
        let (mut pu, mut pv) = (zu.clone(), zv.clone());
        let (mut au, mut av) = (vec![0.0; nu], vec![0.0; nv]);
        let mut rz = dot(&ru, &zu, &rv, &zv);
        let r0 = dot(&ru, &ru, &rv, &rv).sqrt();
        let mut it = 0;
        while it < self.friction_max_iter && r0 > 0.0 {
            let rn = dot(&ru, &ru, &rv, &rv).sqrt();
            if rn <= FRICTION_TOL * r0 {
                break;
            }
            apply(&pu, &pv, &mut au, &mut av);
            let pap = dot(&pu, &au, &pv, &av);
            if pap <= 0.0 {
                break;
            }
            let al = rz / pap;
            for f in 0..nu {
                xu[f] += al * pu[f];
                ru[f] -= al * au[f];
                zu[f] = ru[f] / du[f];
            }
            for f in 0..nv {
                xv[f] += al * pv[f];
                rv[f] -= al * av[f];
                zv[f] = rv[f] / dv[f];
            }
            let rz2 = dot(&ru, &zu, &rv, &zv);
            let be = rz2 / rz.max(1e-300);
            rz = rz2;
            for f in 0..nu {
                pu[f] = zu[f] + be * pu[f];
            }
            for f in 0..nv {
                pv[f] = zv[f] + be * pv[f];
            }
            it += 1;
        }
        for f in 0..nu {
            us[f] += xu[f];
        }
        for f in 0..nv {
            vs[f] += xv[f];
        }
        self.last_friction_iters = it;
        it
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::phases::{Phase, Registry};
    use super::*;

    /// colonne de sable sec tassé (α = α_max) de largeur w et de hauteur h au milieu d'une boîte d'air ; φ = angle de frottement
    pub(crate) fn sand_column(nx: usize, ny: usize, dx: f64, w: f64, h: f64, phi: f64) -> MultiphaseSim {
        let mut r = Registry::new();
        let air = r.add_phase(Phase::air());
        let mut sand = Phase::sand();
        sand.friction_angle = phi;
        let amax = sand.alpha_max;
        let sable = r.add_phase(sand);
        let mut s = MultiphaseSim::new(nx, ny, dx, r, air);
        s.surface_tension = false;
        let lx = nx as f64 * dx;
        s.paint_mixture(move |x, y| {
            let mut m = vec![0.0; 2];
            if (x - 0.5 * lx).abs() < 0.5 * w && y < h {
                m[sable] = amax;
                m[air] = 1.0 - amax;
            } else {
                m[air] = 1.0;
            }
            Some(m)
        });
        s.finalize_init(|_, _| 293.15);
        s
    }

    /// hauteur de sable tassé de chaque colonne (Σ α_s / α_max · dx)
    pub(crate) fn pile_heights(s: &MultiphaseSim) -> Vec<f64> {
        let k = s.reg.phases.iter().position(|p| p.is_granular()).unwrap();
        let amax = s.reg.phases[k].alpha_max;
        (0..s.nx).map(|i| (0..s.ny).map(|j| s.alpha[k][s.idx(i, j)]).sum::<f64>() / amax * s.dx).collect()
    }

    /// pente moyenne (degrés) des flancs du tas : régression de h(x) là où 15 % < h/h_max < 85 %
    pub(crate) fn flank_angle(s: &MultiphaseSim) -> f64 {
        let h = pile_heights(s);
        let hm = h.iter().fold(0.0f64, |a, &b| a.max(b));
        let mut sl = vec![];
        for i in 1..h.len() - 1 {
            if h[i] > 0.15 * hm && h[i] < 0.85 * hm {
                sl.push(((h[i + 1] - h[i - 1]) / (2.0 * s.dx)).abs());
            }
        }
        if sl.is_empty() {
            return 90.0;
        }
        (sl.iter().sum::<f64>() / sl.len() as f64).atan().to_degrees()
    }

    /// tas triangulaire de sable sec tassé (hauteur h, pente θ en degrés) au milieu d'une boîte d'air
    fn sand_pile(nx: usize, ny: usize, dx: f64, h: f64, theta: f64, phi: f64) -> MultiphaseSim {
        let mut r = Registry::new();
        let air = r.add_phase(Phase::air());
        let mut sand = Phase::sand();
        sand.friction_angle = phi;
        let amax = sand.alpha_max;
        let sable = r.add_phase(sand);
        let mut s = MultiphaseSim::new(nx, ny, dx, r, air);
        s.surface_tension = false;
        let lx = nx as f64 * dx;
        let t = theta.to_radians().tan();
        s.paint_mixture(move |x, y| {
            let mut m = vec![0.0; 2];
            if y < h - (x - 0.5 * lx).abs() * t {
                m[sable] = amax;
                m[air] = 1.0 - amax;
            } else {
                m[air] = 1.0;
            }
            Some(m)
        });
        s.finalize_init(|_, _| 293.15);
        s
    }

    #[test]
    fn a_pile_below_the_angle_of_repose_stands_and_a_steeper_one_slumps() {
        // tas de 6 cm, φ = 32° : à 20° il tient (hauteurs inchangées à une maille près) ; à 50° il s'affaisse vite vers 42° puis flue
        // jusqu'à φ (≈ 35° à 3 s, 32.6° à 4 s)
        let run = |theta: f64, t_end: f64| {
            let mut s = sand_pile(80, 24, 0.005, 0.06, theta, 32.0);
            let h0 = pile_heights(&s);
            let a0 = flank_angle(&s);
            while s.time < t_end && !s.diverged {
                s.step();
            }
            assert!(!s.diverged && s.state_is_sane());
            let h1 = pile_heights(&s);
            let dh = h0.iter().zip(&h1).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            let a1 = flank_angle(&s);
            let k = 1;
            let um = (0..s.nx * s.ny).filter(|&c| s.alpha[k][c] > 0.3).map(|c| {
                let (i, j) = (c % s.nx, c / s.nx);
                (0.5 * (s.u[s.uidx(i, j)] + s.u[s.uidx(i + 1, j)])).hypot(0.5 * (s.v[s.vidx(i, j)] + s.v[s.vidx(i, j + 1)]))
            }).fold(0.0f64, f64::max);
            {
                // répartition des vitesses dans le sable : où est le maximum, médiane
                let mut v: Vec<(f64, usize)> = (0..s.nx * s.ny).filter(|&c| s.alpha[k][c] > 0.3).map(|c| {
                    let (i, j) = (c % s.nx, c / s.nx);
                    ((0.5 * (s.u[s.uidx(i, j)] + s.u[s.uidx(i + 1, j)])).hypot(0.5 * (s.v[s.vidx(i, j)] + s.v[s.vidx(i, j + 1)])), c)
                }).collect();
                v.sort_by(|a, b| a.0.total_cmp(&b.0));
                let (mx, c) = v[v.len() - 1];
                let up = c + s.nx < s.nx * s.ny && s.alpha[k][c + s.nx] > 0.3;
                eprintln!("   |u| médiane {:.2e}, 90 % {:.2e}, max {mx:.2e} en ({}, {}) α {:.2} (cellule de sable au-dessus : {up})", v[v.len() / 2].0, v[v.len() * 9 / 10].0, c % s.nx, c / s.nx, s.alpha[k][c]);
            }
            eprintln!("θ = {theta}° : angle {a0:.1}° → {a1:.1}°, écart de hauteur max {:.2} mm, |u| max dans le sable {um:.2e} m/s", dh * 1e3);
            (a1, dh, um)
        };
        let (_, dh, um) = run(20.0, 1.0);
        assert!(dh < 0.005 && um < 1e-2, "le tas à 20° doit tenir : Δh {dh}, |u| {um}");
        let (a, _, _) = run(50.0, 3.0);
        assert!(a > 26.0 && a < 37.0, "le tas à 50° doit s'affaisser vers φ = 32° : {a}°");
    }

    #[test]
    fn the_sand_column_scenario_forms_a_pile() {
        use super::super::mp_view::{build_scenario, MpScenario};
        let mut s = build_scenario(MpScenario::SandColumn, 40, 30, 0.01, 9.81);
        assert!(s.has_friction(), "le sable par défaut doit frotter");
        assert!(super::super::gpu_multiphase::gpu_unsupported_reason(&s).is_none());
        while s.time < 1.5 && !s.diverged {
            s.step();
        }
        assert!(!s.diverged && s.state_is_sane());
        let a = flank_angle(&s);
        eprintln!("scénario colonne de sable : flancs à {a:.1}° après {} pas", s.step_count);
        assert!(a > 15.0 && a < 40.0, "flancs à {a}°");
    }

    #[test]
    fn a_sand_column_collapses_into_a_pile_at_the_angle_of_repose() {
        // colonne de 4 cm × 8 cm de sable sec (φ = 32°) dans 30 × 15 cm d'air : avec frottement, un tas aux flancs proches de φ ;
        // sans frottement, une nappe presque plate
        let run = |phi: f64| {
            let mut s = sand_column(60, 30, 0.005, 0.04, 0.08, phi);
            let m0 = s.total_mass(1);
            while s.time < 1.5 && !s.diverged {
                s.step();
            }
            assert!(!s.diverged && s.state_is_sane());
            assert!((s.total_mass(1) / m0 - 1.0).abs() < 1e-9, "masse de sable");
            let h = pile_heights(&s);
            let hm = h.iter().fold(0.0f64, |a, &b| a.max(b));
            let spread = h.iter().filter(|&&x| x > 0.05 * hm).count() as f64 * s.dx;
            eprintln!("φ = {phi}° : angle des flancs {:.1}°, hauteur {:.3} m, étalement {:.3} m, {} pas, frottement {} it. (dernier pas)",
                flank_angle(&s), hm, spread, s.step_count, s.last_friction_iters);
            (flank_angle(&s), hm)
        };
        let (a0, h0) = run(0.0);
        let (a1, h1) = run(32.0);
        assert!(a0 < 10.0, "sans frottement le sable doit s'étaler : {a0}°");
        assert!(a1 > 20.0 && a1 < 40.0, "angle du tas {a1}° (φ = 32°)");
        assert!(h1 > 1.5 * h0, "le tas doit rester plus haut qu'une nappe : {h1} vs {h0}");
    }
}
