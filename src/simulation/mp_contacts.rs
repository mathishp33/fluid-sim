//! Contacts des corps libres (solides mobiles, étape 4 de SOLID.md) : contre les bords du domaine, les solides fixes (mailles
//! carrées) et les autres corps.
//!
//! Contact SOUPLE, comme dans la méthode des éléments discrets : dès qu'un corps passe à moins de la peau (CONTACT_SKIN maille) d'un
//! obstacle, de profondeur δ = peau − écart, une force normale F_n = k (δ − δ0) − c v_n (ressort-amortisseur, jamais attractive) et
//! un frottement de Coulomb régularisé, |F_t| ≤ μ F_n. Chaque contact a son ressort (`ContactSpring`), réglé à son début sur la
//! vitesse d'approche : k = m ω², ω = max(√(g/δ_max), v_n/δ_max) ; la pénétration dans la peau reste de l'ordre de
//! δ_max = CONTACT_PENETRATION maille, au repos sous la pesanteur comme à l'impact. Ce ressort est gardé tant que le contact dure,
//! raidi si l'approche s'accélère (un bloc qui tombe sur un autre déjà posé), détendu quand le corps s'est posé, sans jamais rendre
//! d'énergie au corps (décalage δ0). `contact_dt_bound` garde CONTACT_STEPS pas par période. L'amortissement donne le coefficient de
//! restitution e (`damping_for_restitution`) ; le fluide amortit en plus.
//! Les forces entrent dans l'équation des corps libres avec la pesanteur (`body_couplings`) : ressort et frottement explicites,
//! amortisseur implicite.

use super::mp_bodies::{RigidBody, Shape};
use super::multiphase::MultiphaseSim;

/// pas par période de contact
pub(crate) const CONTACT_STEPS: f64 = 8.0;
/// pénétration maximale visée dans la peau, en mailles
pub(crate) const CONTACT_PENETRATION: f64 = 0.1;
/// Peau de contact, en mailles : la répulsion commence quand l'écart entre un corps et un obstacle passe sous cette épaisseur. Sans
/// elle, un bloc posé à plat laissait sous lui un film de fluide de quelques microns dont le volume, à chaque frémissement du bloc,
/// variait de plusieurs fois le sien en un pas : la détente y rendait la température négative (pas refaits, bloc relancé). Les
/// méthodes à domaine fictif font de même (zone de répulsion d'une maille environ, Glowinski et al. 1999).
pub(crate) const CONTACT_SKIN: f64 = 0.4;

/// Ce que touche un corps : un bord du domaine (0 gauche, 1 droite, 2 bas, 3 haut), les solides fixes, un autre corps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Obstacle {
    Wall(u8),
    Solid,
    Body(usize),
}

/// Contact du corps `body` avec `with` : point, normale (unitaire, dirigée VERS le corps : sens de la force sur lui), profondeur.
#[derive(Clone, Debug)]
pub(crate) struct Contact {
    pub body: usize,
    pub with: Obstacle,
    pub point: [f64; 2],
    pub normal: [f64; 2],
    pub depth: f64,
}

impl Contact {
    /// l'autre corps, si l'obstacle en est un
    pub fn other(&self) -> Option<usize> {
        match self.with {
            Obstacle::Body(l) => Some(l),
            _ => None,
        }
    }
}

/// Ressort d'un contact en cours entre `body` et `with` : force normale m ω² (δ − δ0), m masse effective au point de contact.
#[derive(Clone, Debug)]
pub(crate) struct ContactSpring {
    pub body: usize,
    pub with: Obstacle,
    /// ω² = k/m
    pub w2: f64,
    /// décalage δ0 (m) : pénétration où le ressort part de zéro (là où le contact a été vu), puis ajusté quand la raideur change
    pub offset: f64,
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

/// Distance signée de p à un polygone convexe (sens trigonométrique), positive dedans, et direction de la force qui repousserait le
/// polygone loin de p (opposée à la normale sortante au point du bord le plus proche).
fn signed_distance(poly: &[[f64; 2]], p: [f64; 2]) -> (f64, [f64; 2]) {
    let n = poly.len();
    let (mut dmin, mut outside, mut dir) = (f64::MAX, false, [0.0, 0.0]);
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        let e = sub(b, a);
        let len2 = dot(e, e);
        if len2 <= 0.0 {
            continue;
        }
        if cross(e, sub(p, a)) < 0.0 {
            outside = true;
        }
        let t = (dot(sub(p, a), e) / len2).clamp(0.0, 1.0);
        let q = sub([a[0] + t * e[0], a[1] + t * e[1]], p);
        let d = q[0].hypot(q[1]);
        if d < dmin {
            dmin = d;
            // normale sortante du côté : (e_y, −e_x) / |e| ; la force sur le polygone lui est opposée
            let len = len2.sqrt();
            dir = [-e[1] / len, e[0] / len];
        }
    }
    if outside {
        (-dmin, dir)
    } else {
        (dmin, dir)
    }
}

/// Axes séparateurs de deux polygones convexes : (recouvrement, normale unitaire de b vers a) s'ils se recouvrent ou s'ils sont à
/// moins de `skin` l'un de l'autre (recouvrement négatif : écart).
fn sat(a: &[[f64; 2]], b: &[[f64; 2]], skin: f64) -> Option<(f64, [f64; 2])> {
    let mut best: Option<(f64, [f64; 2])> = None;
    for poly in [a, b] {
        let n = poly.len();
        for i in 0..n {
            let e = sub(poly[(i + 1) % n], poly[i]);
            let len = e[0].hypot(e[1]);
            if len <= 0.0 {
                continue;
            }
            let ax = [e[1] / len, -e[0] / len];
            let (mut a0, mut a1, mut b0, mut b1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
            for p in a {
                let d = dot(*p, ax);
                a0 = a0.min(d);
                a1 = a1.max(d);
            }
            for p in b {
                let d = dot(*p, ax);
                b0 = b0.min(d);
                b1 = b1.max(d);
            }
            let ov = a1.min(b1) - a0.max(b0);
            if ov <= -skin {
                return None;
            }
            if best.map_or(true, |(d, _)| ov < d) {
                // orientée de b vers a
                let ca = 0.5 * (a0 + a1);
                let cb = 0.5 * (b0 + b1);
                let ax = if ca >= cb { ax } else { [-ax[0], -ax[1]] };
                best = Some((ov, ax));
            }
        }
    }
    best
}

/// Amortissement réduit ζ qui donne le coefficient de restitution e au schéma réel : `steps` pas par période, dans l'ordre du calcul
/// (force à la position atteinte, ressort explicite et amortisseur IMPLICITE qui n'agit qu'à l'approche, puis déplacement), le
/// ressort partant de zéro là où le contact est vu (`update_contact_springs`). Recherche par dichotomie sur le contact d'un
/// ressort-amortisseur seul (restitution décroissante en ζ, de 1 à 0), mémorisée par (e, pas par période au 1/8). Le premier
/// calibrage faisait entrer le corps d'un pas entier dans l'obstacle avant toute force : à 8 pas par période, l'énergie de cette
/// entrée seule le renvoyait à e ≈ 0.85, et ζ partait en butée (50) pour tout e < 0.85.
/// Un amortisseur explicite dépassait : au-delà de ζ ≈ 0.5 il renvoyait le corps plus vite qu'il n'arrivait (un disque qui touchait
/// le sol à 1.9 m/s repartait à 11 m/s).
pub(crate) fn damping_for_restitution(e: f64, steps: f64) -> f64 {
    static CACHE: std::sync::Mutex<Vec<(u64, u32, f64)>> = std::sync::Mutex::new(Vec::new());
    let e = e.clamp(0.0, 1.0);
    if e >= 1.0 {
        return 0.0;
    }
    let q = (steps.clamp(4.0, 256.0) * 8.0).round() as u32;
    let key = (e.to_bits(), q);
    if let Ok(c) = CACHE.lock() {
        if let Some(&(_, _, z)) = c.iter().find(|(a, b, _)| (*a, *b) == key) {
            return z;
        }
    }
    let n = q as f64 / 8.0;
    let w = 2.0 * std::f64::consts::PI / n;
    // restitution du schéma pour ζ : pénétration x (≥ 0), vitesse v (> 0 : s'enfonce), pas de temps 1
    let rest = |z: f64| -> f64 {
        let (mut x, mut v) = (0.0f64, 1.0f64);
        for _ in 0..(40.0 * n) as usize {
            let a = w * w * x;
            v = if v > 0.0 { (v - a) / (1.0 + 2.0 * z * w) } else { v - a };
            x += v;
            if x <= 0.0 {
                return (-v).max(0.0);
            }
        }
        0.0
    };
    let (mut lo, mut hi) = (0.0f64, 50.0f64);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if rest(mid) > e {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let z = 0.5 * (lo + hi);
    if let Ok(mut c) = CACHE.lock() {
        if c.len() > 4096 {
            c.clear();
        }
        c.push((key.0, key.1, z));
    }
    z
}

impl MultiphaseSim {
    /// masse du corps k (par mètre de profondeur)
    pub(crate) fn body_mass(&self, k: usize) -> f64 {
        let b = &self.bodies[k];
        b.density(self.reg.solids[b.material].rho) * b.shape.area()
    }

    /// moment d'inertie du corps k autour de son centre de masse
    pub(crate) fn body_inertia(&self, k: usize) -> f64 {
        let b = &self.bodies[k];
        b.density(self.reg.solids[b.material].rho) * b.shape.polar_moment()
    }

    /// Contacts des corps libres (vides sans corps libre). Un contact commence dès que l'écart passe sous la peau
    /// (CONTACT_SKIN maille) : sa « profondeur » est la peau moins l'écart.
    pub(crate) fn contacts(&self) -> Vec<Contact> {
        let mut out = vec![];
        if !self.has_free_bodies() {
            return out;
        }
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let (lx, ly) = (nx as f64 * dx, ny as f64 * dx);
        let skin = CONTACT_SKIN * dx;
        let polys: Vec<Vec<[f64; 2]>> = self.bodies.iter().map(|b| b.world_vertices()).collect();
        let any_solid = self.solid.iter().any(|&s| s >= 0);
        let solid_at = |ii: isize, jj: isize| ii >= 0 && jj >= 0 && (ii as usize) < nx && (jj as usize) < ny && self.is_solid(jj as usize * nx + ii as usize);
        for (k, b) in self.bodies.iter().enumerate() {
            if !b.is_free() {
                continue;
            }
            let poly = &polys[k];
            // bords du domaine : (normale vers l'intérieur, profondeur d'un sommet)
            let walls: [([f64; 2], &dyn Fn([f64; 2]) -> f64); 4] = [
                ([1.0, 0.0], &|p: [f64; 2]| skin - p[0]),
                ([-1.0, 0.0], &|p: [f64; 2]| skin + p[0] - lx),
                ([0.0, 1.0], &|p: [f64; 2]| skin - p[1]),
                ([0.0, -1.0], &|p: [f64; 2]| skin + p[1] - ly),
            ];
            for (w, (n, depth)) in walls.iter().enumerate() {
                Self::push_contact(&mut out, k, Obstacle::Wall(w as u8), *n, poly.iter().map(|p| (*p, depth(*p))));
            }
            if any_solid {
                // sommets du corps dans une maille solide (sortie par la face exposée la plus proche) ou à moins de la peau d'une
                // maille solide voisine (repoussés depuis le point le plus proche de cette maille), groupés par normale
                let mut by_normal: Vec<([f64; 2], Vec<([f64; 2], f64)>)> = vec![];
                for p in poly {
                    let (fi, fj) = ((p[0] / dx).floor(), (p[1] / dx).floor());
                    if fi < -1.0 || fj < -1.0 || fi > nx as f64 || fj > ny as f64 {
                        continue;
                    }
                    let (i, j) = (fi as isize, fj as isize);
                    let mut best: Option<(f64, [f64; 2])> = None;
                    if solid_at(i, j) {
                        let (x0, y0) = (i as f64 * dx, j as f64 * dx);
                        let open = |ii: isize, jj: isize| ii >= 0 && jj >= 0 && (ii as usize) < nx && (jj as usize) < ny && !solid_at(ii, jj);
                        for (ok, d, n) in [
                            (open(i - 1, j), p[0] - x0, [-1.0, 0.0]),
                            (open(i + 1, j), x0 + dx - p[0], [1.0, 0.0]),
                            (open(i, j - 1), p[1] - y0, [0.0, -1.0]),
                            (open(i, j + 1), y0 + dx - p[1], [0.0, 1.0]),
                        ] {
                            if ok && best.map_or(true, |(bd, _)| skin + d > bd) {
                                best = Some((skin + d, n));
                            }
                        }
                    } else {
                        for (di, dj) in [(-1isize, -1isize), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                            if !solid_at(i + di, j + dj) {
                                continue;
                            }
                            let (x0, y0) = ((i + di) as f64 * dx, (j + dj) as f64 * dx);
                            let q = [p[0].clamp(x0, x0 + dx), p[1].clamp(y0, y0 + dx)];
                            let r = sub(*p, q);
                            let d = r[0].hypot(r[1]);
                            if d > 0.0 && d < skin && best.map_or(true, |(bd, _)| skin - d > bd) {
                                best = Some((skin - d, [r[0] / d, r[1] / d]));
                            }
                        }
                    }
                    if let Some((d, n)) = best {
                        match by_normal.iter_mut().find(|(m, _)| *m == n) {
                            Some((_, v)) => v.push((*p, d)),
                            None => by_normal.push((n, vec![(*p, d)])),
                        }
                    }
                }
                for (n, pts) in by_normal {
                    Self::push_contact(&mut out, k, Obstacle::Solid, n, pts.into_iter());
                }
                // coins SAILLANTS des mailles solides (une seule des quatre mailles qui les entourent est solide) dans le corps ou à
                // moins de la peau de son bord ; un coin le long d'une surface plane n'est pas un obstacle de plus (seize ressorts en
                // parallèle sous un bloc posé à plat)
                let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
                for p in poly {
                    x0 = x0.min(p[0] - skin);
                    y0 = y0.min(p[1] - skin);
                    x1 = x1.max(p[0] + skin);
                    y1 = y1.max(p[1] + skin);
                }
                let ci = |x: f64, n: usize| ((x / dx).floor().max(0.0) as usize).min(n);
                for cy in ci(y0, ny)..=ci(y1, ny) {
                    for cx in ci(x0, nx)..=ci(x1, nx) {
                        let (ci_, cj_) = (cx as isize, cy as isize);
                        let around = [solid_at(ci_ - 1, cj_ - 1), solid_at(ci_, cj_ - 1), solid_at(ci_ - 1, cj_), solid_at(ci_, cj_)];
                        if around.iter().filter(|&&x| x).count() != 1 {
                            continue;
                        }
                        let q = [cx as f64 * dx, cy as f64 * dx];
                        let (sd, n) = signed_distance(poly, q);
                        if sd > -skin {
                            out.push(Contact { body: k, with: Obstacle::Solid, point: q, normal: n, depth: skin + sd });
                        }
                    }
                }
            }
            // autres corps (une fois par paire de corps libres)
            for (l, c) in self.bodies.iter().enumerate() {
                if l == k || (c.is_free() && l < k) {
                    continue;
                }
                let (rb, rc) = (b.shape.radius(), c.shape.radius());
                let dg = sub(b.pos, c.pos);
                let dist = dg[0].hypot(dg[1]);
                if dist >= rb + rc + skin {
                    continue;
                }
                if let (Shape::Disc { r: ra }, Shape::Disc { r: rl }) = (&b.shape, &c.shape) {
                    let depth = skin + ra + rl - dist;
                    if depth > 0.0 && dist > 0.0 {
                        let n = [dg[0] / dist, dg[1] / dist];
                        let gap = dist - ra - rl;
                        let point = [c.pos[0] + n[0] * (rl + 0.5 * gap), c.pos[1] + n[1] * (rl + 0.5 * gap)];
                        out.push(Contact { body: k, with: Obstacle::Body(l), point, normal: n, depth });
                    }
                    continue;
                }
                if let Some((ov, n)) = sat(poly, &polys[l], skin) {
                    // sommets de l'un dans l'autre ou à moins de la peau, pondérés par leur pénétration dans la peau (un bloc posé à plat
                    // sur un autre résiste ainsi au basculement)
                    let pts: Vec<([f64; 2], f64)> = poly
                        .iter()
                        .map(|p| (*p, skin + signed_distance(&polys[l], *p).0))
                        .chain(polys[l].iter().map(|p| (*p, skin + signed_distance(poly, *p).0)))
                        .filter(|(_, w)| *w > 0.0)
                        .collect();
                    let wsum: f64 = pts.iter().map(|(_, w)| w).sum();
                    let point = if pts.is_empty() {
                        [b.pos[0] - n[0] * rb, b.pos[1] - n[1] * rb]
                    } else {
                        [pts.iter().map(|(p, w)| p[0] * w).sum::<f64>() / wsum, pts.iter().map(|(p, w)| p[1] * w).sum::<f64>() / wsum]
                    };
                    out.push(Contact { body: k, with: Obstacle::Body(l), point, normal: n, depth: skin + ov });
                }
            }
        }
        out
    }

    /// Un contact par groupe de points pénétrants (même normale) : profondeur la plus grande, point pondéré par les profondeurs.
    fn push_contact(out: &mut Vec<Contact>, k: usize, with: Obstacle, normal: [f64; 2], pts: impl Iterator<Item = ([f64; 2], f64)>) {
        let (mut wsum, mut px, mut py, mut dmax) = (0.0, 0.0, 0.0, 0.0f64);
        for (p, d) in pts {
            if d > 0.0 {
                wsum += d;
                px += d * p[0];
                py += d * p[1];
                dmax = dmax.max(d);
            }
        }
        if wsum > 0.0 {
            out.push(Contact { body: k, with, point: [px / wsum, py / wsum], normal, depth: dmax });
        }
    }

    /// Forces de contact sur chaque corps (Fx, Fy, couple autour du centre de masse), pour un pas dt.
    /// Renvoie aussi, pour chaque corps, la matrice d'amortissement IMPLICITE de ses contacts (c e eᵀ, e = (n, (p − G) × n) : à
    /// ajouter à l'inertie du corps, M + dt D) ; la vitesse de l'autre corps d'une paire libre y entre explicitement.
    pub(crate) fn contact_forces(&self, dt: f64) -> (Vec<[f64; 3]>, Vec<[[f64; 3]; 3]>) {
        let mut f = vec![[0.0f64; 3]; self.bodies.len()];
        let mut damp = vec![[[0.0f64; 3]; 3]; self.bodies.len()];
        if !(dt > 0.0) {
            return (f, damp);
        }
        let (w_rest2, dmax, _) = self.contact_scales();
        for c in self.contacts() {
            let k = c.body;
            // ressort du contact, réglé à son début (`update_contact_springs`) et gardé tant qu'il dure : une raideur qui suivait le pas
            // ne rendait pas à la détente l'énergie de la compression (rebond à 0.29 pour e = 0.8)
            let (w2, offset) = match self.contact_springs.iter().find(|s| s.body == k && s.with == c.with) {
                Some(s) => (s.w2, s.offset),
                None => (w_rest2.max((self.approach_speed(&c) / dmax).powi(2)), 0.0),
            };
            let w = w2.sqrt();
            let b = &self.bodies[k];
            let free_other = c.other().filter(|&l| self.bodies[l].is_free());
            // masse effective au point de contact le long de d, rotation comprise (et des deux corps s'ils sont libres) : sans la
            // rotation, un contact en coin oscillait deux à trois fois plus vite que prévu et le ressort explicite s'emballait
            let inv = |j: usize, d: [f64; 2]| -> f64 {
                let bj = &self.bodies[j];
                let (mj, ij) = (self.body_mass(j), self.body_inertia(j));
                match bj.pivot {
                    Some(q) => {
                        let gq = sub(bj.pos, q);
                        let iq = ij + mj * dot(gq, gq);
                        cross(sub(c.point, q), d).powi(2) / iq.max(1e-300)
                    }
                    None => 1.0 / mj + cross(sub(c.point, bj.pos), d).powi(2) / ij.max(1e-300),
                }
            };
            let pair = |d: [f64; 2]| -> f64 { 1.0 / (inv(k, d) + free_other.map_or(0.0, |l| inv(l, d))) };
            let m = pair(c.normal);
            let restitution = match c.other() {
                Some(l) => 0.5 * (b.restitution + self.bodies[l].restitution),
                None => b.restitution,
            };
            let zeta = damping_for_restitution(restitution, 2.0 * std::f64::consts::PI / (w * dt));
            let (ks, mut cs) = (m * w2, 2.0 * zeta * m * w);
            if free_other.is_some() {
                // deux corps libres : chacun voit la vitesse de l'autre au pas précédent ; au-delà de c dt = m, leurs vitesses
                // s'échangeraient d'un pas à l'autre au lieu de s'égaliser
                cs = cs.min(m / dt);
            }
            let vo = c.other().map_or([0.0, 0.0], |l| self.bodies[l].velocity_at(c.point));
            let vb = b.velocity_at(c.point);
            let vr = sub(vb, vo);
            let vn = dot(vr, c.normal);
            // ressort explicite ; amortisseur implicite, à l'approche seulement (v_n < 0) : force −c (v_k − v_l)·n n sur k, l'opposée
            // sur l, la vitesse de l'autre corps explicite
            let fnrm = (ks * (c.depth - offset)).max(0.0);
            if vn < 0.0 {
                let ek = [c.normal[0], c.normal[1], cross(sub(c.point, b.pos), c.normal)];
                let von = dot(vo, c.normal);
                for r in 0..3 {
                    for q in 0..3 {
                        damp[k][r][q] += cs * ek[r] * ek[q];
                    }
                    f[k][r] += cs * von * ek[r];
                }
                if let Some(l) = free_other {
                    let bl = &self.bodies[l];
                    let el = [-c.normal[0], -c.normal[1], -cross(sub(c.point, bl.pos), c.normal)];
                    let vkn = dot(vb, c.normal);
                    for r in 0..3 {
                        for q in 0..3 {
                            damp[l][r][q] += cs * el[r] * el[q];
                        }
                        // (le signe était inversé : quand le bloc du dessous s'enfonçait, celui du dessus était relancé vers le haut)
                        f[l][r] -= cs * vkn * el[r];
                    }
                }
            }
            // frottement de Coulomb régularisé : au plus μ F_n, sinon de quoi annuler la moitié du glissement en un pas
            let vt = [vr[0] - vn * c.normal[0], vr[1] - vn * c.normal[1]];
            let st = vt[0].hypot(vt[1]);
            let mut ft = [0.0, 0.0];
            if st > 0.0 {
                let t = [vt[0] / st, vt[1] / st];
                let mt = pair(t);
                let mu = match c.other() {
                    Some(l) => 0.5 * (b.friction + self.bodies[l].friction),
                    None => b.friction,
                };
                let mag = (mu * fnrm).min(0.5 * mt / dt * st);
                ft = [-mag * t[0], -mag * t[1]];
            }
            let force = [fnrm * c.normal[0] + ft[0], fnrm * c.normal[1] + ft[1]];
            let rk = sub(c.point, b.pos);
            f[k][0] += force[0];
            f[k][1] += force[1];
            f[k][2] += cross(rk, force);
            if let Some(l) = free_other {
                let rl = sub(c.point, self.bodies[l].pos);
                f[l][0] -= force[0];
                f[l][1] -= force[1];
                f[l][2] -= cross(rl, force);
            }
        }
        (f, damp)
    }

    /// (ω² au repos g/δ_max, δ_max, vitesse d'un corps posé √(g δ_max))
    fn contact_scales(&self) -> (f64, f64, f64) {
        let g = self.gravity[0].hypot(self.gravity[1]).max(1e-3);
        let dmax = CONTACT_PENETRATION * self.dx;
        (g / dmax, dmax, (g * dmax).sqrt())
    }

    /// vitesse d'approche au point de contact (> 0 : le corps s'enfonce dans l'obstacle)
    fn approach_speed(&self, c: &Contact) -> f64 {
        let vo = c.other().map_or([0.0, 0.0], |l| self.bodies[l].velocity_at(c.point));
        (-dot(sub(self.bodies[c.body].velocity_at(c.point), vo), c.normal)).max(0.0)
    }

    /// Ressorts des contacts après un déplacement des corps : un contact qui commence prend ω = max(√(g/δ_max), v_n/δ_max) ; un
    /// contact qui dure garde le sien, raidi si l'approche l'exige (bloc posé qu'un autre vient frapper), détendu vers √(g/δ_max) une
    /// fois le corps posé (vitesse relative sous 0.2 √(g δ_max)) — une raideur d'impact gardée au repos bridait le pas sans fin. Aucun
    /// changement ne rend d'énergie au corps (décalage δ0) : un saut de raideur à pénétration donnée le relançait.
    pub(crate) fn update_contact_springs(&mut self, dt: f64) {
        let (w_rest2, dmax, quiet) = self.contact_scales();
        let contacts = self.contacts();
        // par paire (corps, obstacle) : profondeur, vitesse d'approche et vitesse relative les plus grandes
        let mut keys: Vec<(usize, Obstacle, f64, f64, f64)> = vec![];
        for c in &contacts {
            let vo = c.other().map_or([0.0, 0.0], |l| self.bodies[l].velocity_at(c.point));
            let vr = sub(self.bodies[c.body].velocity_at(c.point), vo);
            let (vn, speed) = ((-dot(vr, c.normal)).max(0.0), vr[0].hypot(vr[1]));
            match keys.iter_mut().find(|e| e.0 == c.body && e.1 == c.with) {
                Some(e) => {
                    e.2 = e.2.max(c.depth);
                    e.3 = e.3.max(vn);
                    e.4 = e.4.max(speed);
                }
                None => keys.push((c.body, c.with, c.depth, vn, speed)),
            }
        }
        let mut next = Vec::with_capacity(keys.len());
        for (body, with, depth, vn, speed) in keys {
            let need = w_rest2.max((vn / dmax).powi(2));
            let spring = match self.contact_springs.iter().find(|s| s.body == body && s.with == with) {
                // le ressort part de zéro là où le contact est vu : la pénétration du dernier pas (entrée à un instant quelconque du
                // pas) ne compte pas, sinon le rebond dépendait de cet instant ; un corps posé d'emblée dans la peau est porté tout de suite
                None => ContactSpring { body, with, w2: need, offset: depth.min(vn * dt.max(0.0)) },
                Some(s) => {
                    let w2 = if need > s.w2 {
                        need
                    } else if speed < 0.2 * quiet && s.w2 > w_rest2 {
                        (0.8 * s.w2).max(w_rest2)
                    } else {
                        s.w2
                    };
                    // Raidi : même force à la pénétration actuelle, w2 (δ − δ0) = w2' (δ − δ0') ; le ressort, plus raide, rend à la détente
                    // moins qu'il n'a reçu. Détendu : l'énergie du ressort F²/2k ne croît pas, la force baisse au moins comme √(k'/k)
                    // (δ0' ≥ 0) — garder la force en détendant la renvoyait accrue (+20 % par détente au fond d'un rebond : le bloc du
                    // dessous, lâché de 0.5 mm, remontait de 0.7 mm).
                    let load = s.w2 * (depth - s.offset).max(0.0);
                    let offset = if w2 > s.w2 {
                        depth - load / w2
                    } else if w2 < s.w2 {
                        (depth - load / (s.w2 * w2).sqrt()).max(0.0)
                    } else {
                        s.offset
                    };
                    ContactSpring { body, with, w2, offset }
                }
            };
            next.push(spring);
        }
        self.contact_springs = next;
    }

    /// Borne du pas près d'un contact : les ressorts en cours restent résolus (CONTACT_STEPS pas par période), et un corps près d'un
    /// obstacle n'y pénètre pas de plus de δ_max avant que le contact ne commence (au repos sous la pesanteur comme à l'impact, à sa
    /// vitesse ou à la vitesse relative de deux corps). Infini sans corps libre.
    pub(crate) fn contact_dt_bound(&self) -> f64 {
        if !self.has_free_bodies() {
            return f64::INFINITY;
        }
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let (lx, ly) = (nx as f64 * dx, ny as f64 * dx);
        let (w_rest2, dmax, _) = self.contact_scales();
        let n = CONTACT_STEPS / (2.0 * std::f64::consts::PI);
        let mut bound = f64::INFINITY;
        for s in &self.contact_springs {
            if s.w2 > 0.0 {
                bound = bound.min(1.0 / (n * s.w2.sqrt()));
            }
        }
        let near = |b: &RigidBody, margin: f64| -> bool {
            let r = b.shape.radius() + margin;
            if b.pos[0] - r < 0.0 || b.pos[1] - r < 0.0 || b.pos[0] + r > lx || b.pos[1] + r > ly {
                return true;
            }
            let ci = |x: f64, n: usize| ((x / dx).floor().max(0.0) as usize).min(n - 1);
            for j in ci(b.pos[1] - r, ny)..=ci(b.pos[1] + r, ny) {
                for i in ci(b.pos[0] - r, nx)..=ci(b.pos[0] + r, nx) {
                    if self.is_solid(j * nx + i) {
                        return true;
                    }
                }
            }
            false
        };
        let speed = |b: &RigidBody| b.vel[0].hypot(b.vel[1]) + b.omega.abs() * b.shape.radius();
        for (k, b) in self.bodies.iter().enumerate() {
            if !b.is_free() {
                continue;
            }
            let v = speed(b);
            let margin = (2.0 * v * self.last_dt.max(0.0)).max(dx);
            let mut vhit = if near(b, margin) { Some(v) } else { None };
            for (l, c) in self.bodies.iter().enumerate() {
                if l == k {
                    continue;
                }
                let d = sub(b.pos, c.pos);
                let vrel = (b.vel[0] - c.vel[0]).hypot(b.vel[1] - c.vel[1]) + b.omega.abs() * b.shape.radius() + c.omega.abs() * c.shape.radius();
                if d[0].hypot(d[1]) < b.shape.radius() + c.shape.radius() + (2.0 * vrel * self.last_dt.max(0.0)).max(dx) {
                    vhit = Some(vhit.unwrap_or(0.0).max(vrel));
                }
            }
            if let Some(v) = vhit {
                bound = bound.min(1.0 / (n * w_rest2.sqrt()));
                if v > 0.0 {
                    bound = bound.min(dmax / (v * n));
                }
            }
        }
        bound
    }
}
