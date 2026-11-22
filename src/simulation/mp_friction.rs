//! Viscosité IMPLICITE : frottement intergranulaire (étape 3 de GRANULAIRE.md : un tas de sable tient au lieu de s'étaler comme un
//! liquide lourd) et fluides très visqueux (miel), résolus ensemble par un gradient conjugué sur les faces.
//!
//! Frottement : modèle de Coulomb continu (Drucker–Prager régularisé, comme dans les simulations d'effondrement de colonnes granulaires) :
//! dans les cellules TASSÉES (fraction de grains α_s au-dessus de α_c = α_max − 0.1), une viscosité μ_g = tan φ · p_s / (2 √(|D|² + δ²)) —
//! la contrainte de cisaillement vaut tan φ · p_s dès que le matériau s'écoule (|D| ≫ δ), le matériau est quasi rigide sous ce seuil.
//! |D| = √(½ D:D). Cellules concernées : tassées, ou partiellement remplies posées sur une cellule tassée (surface d'un tas), au prorata
//! α_s / α_max. Pression de contact p_s : pression du mélange moins celle du fluide des pores (comme Lagrée, Staron & Popinet 2011 avec
//! la pression incompressible) — contrainte moyenne, nulle au bord libre ; le poids des grains au-dessus rendrait le sable cohésif.
//! |D| est celui du pas précédent (des itérations de Picard avec |D| de la vitesse résolue ont été essayées : aucun effet mesurable,
//! coût ×3). Non-glissement contre les parois ET les obstacles solides (le sable tient sur une étagère) ; chaleur de frottement
//! Φ = τ : D rendue aux grains.
//!
//! Fluides : la part de μ au-delà de la borne explicite 0.2 ρ dx²/dt (`implicit_viscosity`) passe dans le même solveur — un miel ne
//! réduit plus le pas de temps ; glissement libre aux parois, comme la viscosité explicite. Sans fluide si visqueux, rien ne change.
//!
//! Solveur : (ρ_f/dt)(u − u_f) = ∇·[μ(∇u + ∇uᵀ)], opérateur symétrique défini positif, gradient conjugué préconditionné (Jacobi). La
//! vitesse u_f voit déjà le gradient de la pression du pas précédent (u_f = u* − dt ∇pⁿ/ρ_f, rendu ensuite) : le frottement équilibre la
//! force NETTE (pesanteur moins poussée hydrostatique), sinon un tas au repos glisserait d'un dt·g par pas. μ_g est plafonnée à
//! `FRICTION_K_MAX` ρ dx²/dt (borne du nombre d'itérations, voir sa valeur).

use super::multiphase::{Bc, MultiphaseSim};

/// taux de déformation de régularisation δ (1/s)
const FRICTION_DELTA: f64 = 1e-3;
/// plafond de μ_g dt / (ρ dx²) (fluage d'un tas au repos : ~1e-4 m/s)
pub(crate) const FRICTION_K_MAX: f64 = 1000.0;
/// tolérance relative du gradient conjugué
const FRICTION_TOL: f64 = 1e-6;
/// borne de stabilité de la viscosité explicite : dt ≤ VISC_EXPLICIT ρ dx² / μ
pub(crate) const VISC_EXPLICIT: f64 = 0.2;

/// Nœud (i, j) du maillage vu par la contrainte de cisaillement τxy : intérieur (quatre cellules fluides), paroi (deux cellules
/// bloquées alignées — solide, ou hors d'un bord fermé), ou rien (coin, bord ouvert, une ou trois cellules bloquées).
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum NodeKind {
    Interior,
    /// paroi en dessous (cellules SO et SE bloquées) : ∂u/∂y = 2 u(i, j) / dx
    Floor,
    /// paroi au-dessus (NO et NE) : ∂u/∂y = −2 u(i, j−1) / dx
    Ceiling,
    /// paroi à gauche (SO et NO) : ∂v/∂x = 2 v(i, j) / dx
    Left,
    /// paroi à droite (SE et NE) : ∂v/∂x = −2 v(i−1, j) / dx
    Right,
    None,
}

/// grandeurs du frottement indépendantes de la vitesse : part de contact, tan φ, pression de contact
pub(crate) struct FrictionGeom {
    pub frac: Vec<f64>,
    pub tanphi: Vec<f64>,
    pub ps: Vec<f64>,
}

impl MultiphaseSim {
    /// au moins une phase granulaire avec frottement (et de la gravité pour peser) ?
    pub fn has_friction(&self) -> bool {
        self.viscous && self.reg.phases.iter().any(|p| p.is_granular() && p.friction_angle > 0.0) && self.gravity.iter().any(|&g| g != 0.0)
    }

    /// cellule qui bloque les grains pour le frottement : solide, ou grille (le tas y repose, non-glissement)
    #[inline]
    pub(crate) fn fr_blocked(&self, c: usize) -> bool {
        self.is_solid(c) || self.is_grille(c)
    }

    /// plafond de la viscosité traitée explicitement dans la cellule c (le reste est implicite) ; ∞ sans viscosité implicite
    pub(crate) fn explicit_mu_cap(&self, c: usize, dt: f64) -> f64 {
        if self.implicit_viscosity && dt > 0.0 { VISC_EXPLICIT * self.rho_mix[c] * self.dx * self.dx / dt } else { f64::INFINITY }
    }

    /// genre du nœud (i, j), 0 ≤ i ≤ nx, 0 ≤ j ≤ ny
    pub(crate) fn node_kind(&self, i: usize, j: usize) -> NodeKind {
        let (nx, ny) = (self.nx as isize, self.ny as isize);
        // cellules SO, SE, NO, NE ; hors domaine : bloquée derrière une paroi, « ouverte » derrière un bord ouvert
        let mut blocked = [false; 4];
        for (q, (di, dj)) in [(-1isize, -1isize), (0, -1), (-1, 0), (0, 0)].into_iter().enumerate() {
            let (ci, cj) = (i as isize + di, j as isize + dj);
            let mut open = false;
            let out = ci < 0 || cj < 0 || ci >= nx || cj >= ny;
            if out {
                for (side, o) in [(0usize, ci < 0), (1, ci >= nx), (2, cj < 0), (3, cj >= ny)] {
                    if o && !matches!(self.bc[side], Bc::Wall) {
                        open = true;
                    }
                }
                if open {
                    return NodeKind::None;
                }
                blocked[q] = true;
            } else {
                blocked[q] = self.fr_blocked(self.idx(ci as usize, cj as usize));
            }
        }
        match blocked {
            [false, false, false, false] => NodeKind::Interior,
            [true, true, false, false] => NodeKind::Floor,
            [false, false, true, true] => NodeKind::Ceiling,
            [true, false, true, false] => NodeKind::Left,
            [false, true, false, true] => NodeKind::Right,
            _ => NodeKind::None,
        }
    }

    /// cellules (non bloquées) autour du nœud (i, j)
    fn node_cells(&self, i: usize, j: usize) -> Vec<usize> {
        let mut v = vec![];
        for (di, dj) in [(-1isize, -1isize), (0, -1), (-1, 0), (0, 0)] {
            let (ci, cj) = (i as isize + di, j as isize + dj);
            if ci >= 0 && cj >= 0 && (ci as usize) < self.nx && (cj as usize) < self.ny {
                let c = self.idx(ci as usize, cj as usize);
                if !self.fr_blocked(c) {
                    v.push(c);
                }
            }
        }
        v
    }

    /// part de contact, tan φ et pression de contact de chaque cellule (voir l'en-tête du module)
    pub(crate) fn friction_geometry(&self) -> FrictionGeom {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let n = nx * ny;
        let np = self.reg.phases.len();
        let gn = (self.gravity[0].powi(2) + self.gravity[1].powi(2)).sqrt();
        let mut g = FrictionGeom { frac: vec![0.0; n], tanphi: vec![0.0; n], ps: vec![0.0; n] };
        if gn == 0.0 || !self.has_friction() {
            return g;
        }
        let mut a_s = vec![0.0; n];
        let mut w = vec![0.0; n]; // poids déjaugé des grains de la cellule par unité de surface (Pa) : haut du massif au bord du domaine
        let mut ramp = vec![0.0; n];
        let mut amx = vec![1.0; n];
        let mut rhof = vec![0.0; n];
        for c in 0..n {
            if self.fr_blocked(c) {
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
            g.tanphi[c] = tp / as_;
            let amax = am / as_;
            amx[c] = amax;
            let ac = amax - 0.1;
            ramp[c] = ((as_ - ac) / (amax - ac)).clamp(0.0, 1.0);
        }
        // cellules en contact : tassées (α_s ≥ α_c), ou posées sur une cellule tassée (surface d'un tas, cellule partiellement remplie) ;
        // part de la cellule occupée par le matériau granulaire tassé : α_s / α_max
        let vertical = self.gravity[1].abs() >= self.gravity[0].abs();
        let down = if vertical { self.gravity[1] < 0.0 } else { self.gravity[0] < 0.0 };
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
        for c in 0..n {
            if a_s[c] <= 1e-9 || self.fr_blocked(c) {
                continue;
            }
            let rest = below(c).map_or(true, |b| self.fr_blocked(b) || ramp[b] > 0.0);
            if ramp[c] > 0.0 || rest {
                g.frac[c] = (a_s[c] / amx[c]).clamp(0.0, 1.0);
            }
        }
        // pression de contact p_s = pression du mélange (pas précédent : contrainte MOYENNE, nulle au bord libre d'une falaise) moins
        // celle du fluide des pores (hydrostatique depuis le haut du massif), le long de la gravité (axe dominant)
        let (lines, len) = if vertical { (nx, ny) } else { (ny, nx) };
        for l in 0..lines {
            let mut pf = f64::NAN;
            let mut prev: Option<usize> = None;
            for t in 0..len {
                let s_ = if down { len - 1 - t } else { t };
                let c = if vertical { self.idx(l, s_) } else { self.idx(s_, l) };
                if self.fr_blocked(c) || g.frac[c] <= 0.0 {
                    pf = f64::NAN;
                    prev = if self.fr_blocked(c) { None } else { Some(c) };
                    continue;
                }
                if pf.is_nan() {
                    pf = match prev { Some(q) => self.p[q] + 0.5 * (rhof[q] + rhof[c]) * gn * dx, None => self.p[c] - 0.5 * w[c] };
                } else if let Some(q) = prev {
                    pf += 0.5 * (rhof[q] + rhof[c]) * gn * dx;
                }
                g.ps[c] = (self.p[c] - pf).max(0.0);
                prev = Some(c);
            }
        }
        g
    }

    /// |D|² = ½ D:D au centre de la cellule (i, j) pour le champ de faces (u, v)
    fn strain_rate2(&self, u: &[f64], v: &[f64], i: usize, j: usize) -> f64 {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let ui = |i: usize, j: usize| u[j * (nx + 1) + i];
        let vi = |i: usize, j: usize| v[j * nx + i];
        let dxx = (ui(i + 1, j) - ui(i, j)) / dx;
        let dyy = (vi(i, j + 1) - vi(i, j)) / dx;
        // ∂u/∂y et ∂v/∂x centrés (décentrés au bord), moyennés sur les deux faces de la cellule
        let (jm, jp) = (j.saturating_sub(1), (j + 1).min(ny - 1));
        let (im, ip) = (i.saturating_sub(1), (i + 1).min(nx - 1));
        let dudy = 0.5 * ((ui(i, jp) - ui(i, jm)) + (ui(i + 1, jp) - ui(i + 1, jm))) / ((jp - jm).max(1) as f64 * dx);
        let dvdx = 0.5 * ((vi(ip, j) - vi(im, j)) + (vi(ip, j + 1) - vi(im, j + 1))) / ((ip - im).max(1) as f64 * dx);
        let dxy = 0.5 * (dudy + dvdx);
        0.5 * (dxx * dxx + dyy * dyy + 2.0 * dxy * dxy)
    }

    /// viscosité de frottement μ_g (Pa·s) par cellule pour le champ de vitesse (u, v) (0 hors du sable tassé)
    pub(crate) fn friction_mu(&self, g: &FrictionGeom, u: &[f64], v: &[f64], dt: f64) -> Vec<f64> {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let mut mu = vec![0.0; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let c = self.idx(i, j);
                if g.frac[c] <= 0.0 || self.fr_blocked(c) {
                    continue;
                }
                let d2 = self.strain_rate2(u, v, i, j);
                let m = g.frac[c] * g.tanphi[c] * g.ps[c] / (2.0 * (d2 + FRICTION_DELTA * FRICTION_DELTA).sqrt());
                mu[c] = m.min(FRICTION_K_MAX * self.rho_mix[c] * dx * dx / dt);
            }
        }
        mu
    }

    /// part implicite de la viscosité des fluides : μ − plafond explicite (0 sans viscosité implicite)
    pub(crate) fn implicit_fluid_mu(&self, dt: f64) -> Vec<f64> {
        let n = self.nx * self.ny;
        (0..n).map(|c| if self.fr_blocked(c) || !self.implicit_viscosity { 0.0 } else { (self.mu_cell[c] - self.explicit_mu_cap(c, dt)).max(0.0) }).collect()
    }

    /// Viscosité implicite (frottement des grains + excès de viscosité des fluides) sur la vitesse prédite (us, vs) pendant dt (voir
    /// l'en-tête du module). À appeler juste avant la projection, `self.p` étant encore la pression du pas précédent. Renvoie le nombre
    /// d'itérations du gradient conjugué.
    pub(crate) fn apply_implicit_viscosity(&mut self, dt: f64, us: &mut [f64], vs: &mut [f64]) -> usize {
        self.last_friction_iters = 0;
        if !self.viscous {
            return 0;
        }
        let friction = self.has_friction();
        let geom = self.friction_geometry();
        let mg = if friction { self.friction_mu(&geom, &self.u, &self.v, dt) } else { vec![0.0; self.nx * self.ny] };
        let mi = self.implicit_fluid_mu(dt);
        if !mg.iter().chain(mi.iter()).any(|&m| m > 0.0) {
            return 0;
        }
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let (nu, nv) = ((nx + 1) * ny, nx * (ny + 1));
        let kinds: Vec<NodeKind> = (0..(nx + 1) * (ny + 1)).map(|q| self.node_kind(q % (nx + 1), q / (nx + 1))).collect();
        let cells: Vec<Vec<usize>> = (0..(nx + 1) * (ny + 1)).map(|q| self.node_cells(q % (nx + 1), q / (nx + 1))).collect();
        // inconnues : faces intérieures entre deux cellules fluides
        let mut ku = vec![false; nu];
        let mut rho_u = vec![0.0; nu];
        for j in 0..ny {
            for i in 1..nx {
                let (a, b) = (self.idx(i - 1, j), self.idx(i, j));
                if !self.fr_blocked(a) && !self.fr_blocked(b) {
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
                if !self.fr_blocked(a) && !self.fr_blocked(b) {
                    let f = j * nx + i;
                    kv[f] = true;
                    rho_v[f] = self.face_rho_y(i, j);
                }
            }
        }
        // vitesse qui équilibre déjà la pression du pas précédent : u_f = u* − dt ∇pⁿ / ρ_f ; faces fixes (parois, solides) à 0 — u* y
        // porte encore dt·g (la projection les annule), qui fausserait le cisaillement de la couche voisine (et la chaleur de frottement)
        let mut uf = us.to_vec();
        let mut vf = vs.to_vec();
        for j in 0..ny {
            for i in 0..=nx {
                let f = j * (nx + 1) + i;
                let fixed = if i == 0 { matches!(self.bc[0], Bc::Wall) } else if i == nx { matches!(self.bc[1], Bc::Wall) } else { !ku[f] };
                let solid_side = (i > 0 && self.fr_blocked(self.idx(i - 1, j))) || (i < nx && self.fr_blocked(self.idx(i, j)));
                if fixed || solid_side {
                    uf[f] = 0.0;
                }
            }
        }
        for j in 0..=ny {
            for i in 0..nx {
                let f = j * nx + i;
                let fixed = if j == 0 { matches!(self.bc[2], Bc::Wall) } else if j == ny { matches!(self.bc[3], Bc::Wall) } else { !kv[f] };
                let solid_side = (j > 0 && self.fr_blocked(self.idx(i, j - 1))) || (j < ny && self.fr_blocked(self.idx(i, j)));
                if fixed || solid_side {
                    vf[f] = 0.0;
                }
            }
        }
        {
            let p = &self.p;
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
        }
        let (total, xu, xv) = solve_viscous(nx, ny, dx, dt, &mg, &mi, &kinds, &cells, &ku, &kv, &rho_u, &rho_v, &uf, &vf, self.friction_max_iter);
        // chaleur de frottement Φ dt (Φ = τ : D, part des grains) rendue aux groupes granulaires de la cellule
        if friction {
            let wu: Vec<f64> = (0..nu).map(|f| uf[f] + xu[f]).collect();
            let wv: Vec<f64> = (0..nv).map(|f| vf[f] + xv[f]).collect();
            let heat = friction_dissipation(nx, ny, dx, &mg, &kinds, &cells, &wu, &wv);
            self.add_granular_heat(&heat, dt);
        }
        for f in 0..nu {
            us[f] += xu[f];
        }
        for f in 0..nv {
            vs[f] += xv[f];
        }
        self.last_friction_iters = total;
        total
    }

    /// ajoute dt·Φ_c (J/m³) aux groupes granulaires de chaque cellule, au prorata de leur fraction volumique
    fn add_granular_heat(&mut self, phi: &[f64], dt: f64) {
        let np = self.reg.phases.len();
        for c in 0..self.nx * self.ny {
            if !(phi[c] > 0.0) || self.is_solid(c) {
                continue;
            }
            let a_s: f64 = (0..np).filter(|&k| self.reg.phases[k].is_granular()).map(|k| self.alpha[k][c].max(0.0)).sum();
            if a_s <= 0.0 {
                continue;
            }
            for k in 0..np {
                if self.reg.phases[k].is_granular() && self.alpha[k][c] > 0.0 {
                    let g = self.group_of[k];
                    self.eg[g][c] += dt * phi[c] * self.alpha[k][c] / a_s;
                }
            }
            self.friction_heat += dt * phi[c] * self.dx * self.dx;
            self.cell_sync(c);
        }
    }
}

/// contrainte de cisaillement au nœud q (genre `kind`) : μ_g avec non-glissement aux parois, μ_i (fluides) aux nœuds intérieurs
fn node_txy(kind: NodeKind, mgn: f64, min: f64, wu: &[f64], wv: &[f64], nx: usize, i: usize, j: usize, dx: f64) -> f64 {
    let ui = |i: usize, j: usize| wu[j * (nx + 1) + i];
    let vi = |i: usize, j: usize| wv[j * nx + i];
    match kind {
        NodeKind::Interior => (mgn + min) * ((ui(i, j) - ui(i, j - 1)) + (vi(i, j) - vi(i - 1, j))) / dx,
        NodeKind::Floor => mgn * 2.0 * ui(i, j) / dx,
        NodeKind::Ceiling => -mgn * 2.0 * ui(i, j - 1) / dx,
        NodeKind::Left => mgn * 2.0 * vi(i, j) / dx,
        NodeKind::Right => -mgn * 2.0 * vi(i - 1, j) / dx,
        NodeKind::None => 0.0,
    }
}

/// viscosités aux nœuds : (μ_g moyen des cellules non bloquées, μ_i moyen aux nœuds intérieurs)
fn node_mu(kinds: &[NodeKind], cells: &[Vec<usize>], mg: &[f64], mi: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut a = vec![0.0; kinds.len()];
    let mut b = vec![0.0; kinds.len()];
    for q in 0..kinds.len() {
        if kinds[q] == NodeKind::None || cells[q].is_empty() {
            continue;
        }
        let k = cells[q].len() as f64;
        a[q] = cells[q].iter().map(|&c| mg[c]).sum::<f64>() / k;
        if kinds[q] == NodeKind::Interior {
            b[q] = cells[q].iter().map(|&c| mi[c]).sum::<f64>() / k;
        }
    }
    (a, b)
}

/// Résout (ρ/dt) δ − F(δ) = F(u_f) (δ = 0 sur les faces fixes) par gradient conjugué préconditionné ; renvoie (itérations, δu, δv).
#[allow(clippy::too_many_arguments)]
fn solve_viscous(
    nx: usize, ny: usize, dx: f64, dt: f64, mg: &[f64], mi: &[f64], kinds: &[NodeKind], cells: &[Vec<usize>], ku: &[bool], kv: &[bool],
    rho_u: &[f64], rho_v: &[f64], uf: &[f64], vf: &[f64], max_it: usize,
) -> (usize, Vec<f64>, Vec<f64>) {
    let (nu, nv) = ((nx + 1) * ny, nx * (ny + 1));
    let dx2 = dx * dx;
    let mu: Vec<f64> = mg.iter().zip(mi).map(|(a, b)| a + b).collect();
    let (mgn, min) = node_mu(kinds, cells, mg, mi);
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
                let q = j * (nx + 1) + i;
                if mgn[q] > 0.0 || min[q] > 0.0 {
                    txy[q] = node_txy(kinds[q], mgn[q], min[q], wu, wv, nx, i, j, dx);
                }
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
    // diagonale de A = ρ/dt − F (préconditionneur de Jacobi) : contribution d'un nœud à la face qu'il cisaille
    let nodec = |q: usize, wall: NodeKind| -> f64 {
        match kinds[q] {
            NodeKind::Interior => mgn[q] + min[q],
            k if k == wall => 2.0 * mgn[q],
            _ => 0.0,
        }
    };
    let mut du = vec![1.0; nu];
    for j in 0..ny {
        for i in 1..nx {
            let f = j * (nx + 1) + i;
            if ku[f] {
                let (a, b) = (j * nx + i - 1, j * nx + i);
                du[f] = rho_u[f] / dt + 2.0 * (mu[a] + mu[b]) / dx2 + (nodec((j + 1) * (nx + 1) + i, NodeKind::Ceiling) + nodec(j * (nx + 1) + i, NodeKind::Floor)) / dx2;
            }
        }
    }
    let mut dv = vec![1.0; nv];
    for j in 1..ny {
        for i in 0..nx {
            let f = j * nx + i;
            if kv[f] {
                let (a, b) = ((j - 1) * nx + i, j * nx + i);
                dv[f] = rho_v[f] / dt + 2.0 * (mu[a] + mu[b]) / dx2 + (nodec(j * (nx + 1) + i + 1, NodeKind::Right) + nodec(j * (nx + 1) + i, NodeKind::Left)) / dx2;
            }
        }
    }
    let (mut ru, mut rv) = (vec![0.0; nu], vec![0.0; nv]);
    force(uf, vf, &mut ru, &mut rv);
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
    while it < max_it && r0 > 0.0 {
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
    (it, xu, xv)
}

/// dissipation du frottement Φ_c = τ : D (W/m³) par cellule pour la vitesse w : 2μ_g (Dxx² + Dyy²) dans la cellule, plus la part des
/// nœuds voisins (μ_n γ², poids ½ sur une paroi) répartie au prorata de μ_g des cellules qui touchent le nœud
#[allow(clippy::too_many_arguments)]
fn friction_dissipation(nx: usize, ny: usize, dx: f64, mg: &[f64], kinds: &[NodeKind], cells: &[Vec<usize>], wu: &[f64], wv: &[f64]) -> Vec<f64> {
    let ui = |i: usize, j: usize| wu[j * (nx + 1) + i];
    let vi = |i: usize, j: usize| wv[j * nx + i];
    let mut phi = vec![0.0; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            let c = j * nx + i;
            if mg[c] > 0.0 {
                let (dxx, dyy) = ((ui(i + 1, j) - ui(i, j)) / dx, (vi(i, j + 1) - vi(i, j)) / dx);
                phi[c] += 2.0 * mg[c] * (dxx * dxx + dyy * dyy);
            }
        }
    }
    let zero = vec![0.0; mg.len()];
    let (mgn, _) = node_mu(kinds, cells, mg, &zero);
    for j in 0..=ny {
        for i in 0..=nx {
            let q = j * (nx + 1) + i;
            if mgn[q] <= 0.0 {
                continue;
            }
            let t = node_txy(kinds[q], mgn[q], 0.0, wu, wv, nx, i, j, dx);
            // γ = τxy / μ ; énergie du nœud μ γ² (poids ½ sur une paroi)
            let gam = t / mgn[q];
            let wgt = if kinds[q] == NodeKind::Interior { 1.0 } else { 0.5 };
            let e = wgt * mgn[q] * gam * gam;
            let s: f64 = cells[q].iter().map(|&c| mg[c]).sum();
            if s > 0.0 {
                for &c in &cells[q] {
                    phi[c] += e * mg[c] / s;
                }
            }
        }
    }
    phi
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
        // tas de 6 cm, φ = 32° : à 20° il tient (hauteurs inchangées à une maille près, |u| < 1e-4 m/s) ; à 50° il s'effondre en
        // avalanche : 30° en 0.35 s, puis immobile
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
        assert!(dh < 0.005 && um < 1e-3, "le tas à 20° doit tenir : Δh {dh}, |u| {um}");
        let (a, _, _) = run(50.0, 1.0);
        assert!(a > 26.0 && a < 33.0, "le tas à 50° doit s'affaisser vers φ = 32° : {a}°");
    }

    /// énergie potentielle Σ m g·(hauteur) (J par m de profondeur)
    fn potential_energy(s: &MultiphaseSim) -> f64 {
        let g = (s.gravity[0].powi(2) + s.gravity[1].powi(2)).sqrt();
        (0..s.nx * s.ny).map(|c| (0..s.reg.phases.len()).map(|k| s.mass[k][c]).sum::<f64>() * g * s.center(c % s.nx, c / s.nx).1).sum::<f64>() * s.dx * s.dx
    }

    /// étagère d'acier (12 cm, x ∈ [0.09, 0.21], dessus à y = 0.065) au milieu d'une boîte d'air, colonne de sable de 4 × 6 cm posée dessus
    pub(crate) fn sand_on_shelf() -> MultiphaseSim {
        use super::super::phases::Solid;
        let mut r = Registry::new();
        let air = r.add_phase(Phase::air());
        let sand = Phase::sand();
        let amax = sand.alpha_max;
        let sable = r.add_phase(sand);
        let acier = r.add_solid(Solid::steel());
        let mut s = MultiphaseSim::new(60, 40, 0.005, r, air);
        s.surface_tension = false;
        let (x0, x1, y0, y1) = (0.09, 0.21, 0.05, 0.065);
        s.paint_solid(move |x, y| if x > x0 && x < x1 && y > y0 && y < y1 { Some(acier) } else { None });
        s.paint_mixture(move |x, y| {
            let mut m = vec![0.0; 2];
            if (x - 0.15).abs() < 0.02 && y > y1 && y < y1 + 0.06 {
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

    /// part du sable posée sur l'étagère de `sand_on_shelf`
    pub(crate) fn sand_on_shelf_fraction(s: &MultiphaseSim) -> f64 {
        let sable = 1;
        let on: f64 = (0..s.nx * s.ny)
            .filter(|&c| {
                let (x, y) = s.center(c % s.nx, c / s.nx);
                x > 0.085 && x < 0.215 && y > 0.065
            })
            .map(|c| s.mass[sable][c])
            .sum::<f64>()
            * s.dx
            * s.dx;
        on / s.total_mass(sable)
    }

    #[test]
    fn sand_stays_on_a_solid_shelf_and_friction_heats_it() {
        // non-glissement contre le solide : le tas reste sur l'étagère ; l'énergie potentielle perdue devient de la chaleur (frottement)
        let mut s = sand_on_shelf();
        let (e_pot0, u0, k0) = (potential_energy(&s), s.total_internal_energy(), s.kinetic_energy());
        while s.time < 1.5 && !s.diverged {
            s.step();
        }
        assert!(!s.diverged && s.state_is_sane());
        let on_shelf = sand_on_shelf_fraction(&s);
        let lost = e_pot0 - potential_energy(&s) - (s.kinetic_energy() - k0);
        let heat = s.friction_heat;
        eprintln!("sable resté sur l'étagère : {:.1} % ; énergie mécanique perdue {lost:.3e} J/m, chaleur de frottement {heat:.3e} J/m ({:.0} %) ; ΔU {:.3e}",
            100.0 * on_shelf, 100.0 * heat / lost, s.total_internal_energy() - u0);
        assert!(on_shelf > 0.85, "le sable a glissé de l'étagère : {on_shelf}");
        assert!(lost > 0.0 && heat > 0.3 * lost && heat < 1.2 * lost, "chaleur de frottement {heat} pour {lost} perdus");
    }

    /// tourbillon de Taylor–Green (glissement libre, 10 cm, 1 cm/s) dans un fluide de μ = 2 Pa·s (ν = 2e-3 m²/s), sans gravité
    pub(crate) fn viscous_vortex(implicit: bool) -> MultiphaseSim {
        {
            let mut r = Registry::new();
            let mut miel = Phase::water();
            miel.mu_ref = 2.0;
            miel.mu_law = crate::simulation::materials::PropertyLaw::Constant;
            let k = r.add_phase(miel);
            let mut s = MultiphaseSim::new(48, 48, 0.1 / 48.0, r, k);
            s.gravity = [0.0, 0.0];
            s.surface_tension = false;
            s.implicit_viscosity = implicit;
            s.paint_phase(move |_, _| Some(k));
            s.finalize_init(|_, _| 293.15);
            let (nx, ny, l) = (s.nx, s.ny, 0.1);
            let pi = std::f64::consts::PI;
            for j in 0..ny {
                for i in 0..=nx {
                    let (x, y) = (i as f64 * s.dx, (j as f64 + 0.5) * s.dx);
                    let q = s.uidx(i, j);
                    s.u[q] = 0.01 * (pi * x / l).sin() * (pi * y / l).cos();
                }
            }
            for j in 0..=ny {
                for i in 0..nx {
                    let (x, y) = ((i as f64 + 0.5) * s.dx, j as f64 * s.dx);
                    let q = s.vidx(i, j);
                    s.v[q] = -0.01 * (pi * x / l).cos() * (pi * y / l).sin();
                }
            }
            s.dt_max = 1e-3;
            s
        }
    }

    #[test]
    fn a_very_viscous_vortex_decays_at_the_viscous_rate_with_large_time_steps() {
        // énergie cinétique ∝ e^(−4νπ²t/L²) ; pas de temps 3× la borne explicite (viscosité implicite), écart à la loi < 5 % ; sans
        // viscosité implicite, le pas est borné
        let mk = viscous_vortex;
        let run = |mut s: MultiphaseSim| {
            let k0 = s.kinetic_energy();
            while s.time < 0.03 && !s.diverged {
                s.step();
            }
            assert!(!s.diverged);
            let nu = 2.0 / s.rho_mix[0];
            let expect = (-4.0 * nu * std::f64::consts::PI.powi(2) / 0.01 * s.time).exp();
            (s.kinetic_energy() / k0, expect, s.step_count)
        };
        let (ri, ei, ni) = run(mk(true));
        let (re, ee, ne) = run(mk(false));
        eprintln!("implicite : EC {ri:.4} (loi {ei:.4}) en {ni} pas ; explicite : EC {re:.4} (loi {ee:.4}) en {ne} pas");
        assert!((ri / ei - 1.0).abs() < 0.05, "décroissance implicite {ri} vs {ei}");
        assert!((re / ee - 1.0).abs() < 0.05, "décroissance explicite {re} vs {ee}");
        assert!(ni * 2 < ne, "la viscosité implicite doit permettre de plus grands pas : {ni} contre {ne}");
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
