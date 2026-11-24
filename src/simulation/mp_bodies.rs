//! Corps rigides — solides mobiles (conception et choix : SOLID.md).
//!
//! ÉTAPE 0 (préparation, aucun résultat changé) : les corps existent dans l'état (`MultiphaseSim::bodies`), leur géométrie est
//! exacte et testée, et la fraction de chaque maille qu'ils occupent (`MultiphaseSim::body_frac`) entre dans la fermeture :
//! Σ m/ρ + α_corps = 1 (`cell_volume`, `cell_closure_g`) — seulement s'il y a des corps, sans corps le calcul est inchangé au bit
//! près. Ils ne sont pas encore couplés à l'écoulement : chasser le fluide du volume qu'ils occupent (mailles entièrement couvertes),
//! masse du corps dans ρ, rigidification de la vitesse et mouvement viennent aux étapes 1 à 3 (SOLID.md § 7). Aucun scénario ni
//! outil n'en pose, et le GPU refuse les scènes qui en ont.
//!
//! * Formes convexes : disque (polygone régulier de `DISC_SIDES` côtés, de même aire que le disque : écart au cercle < 0.04 % du
//!   rayon), rectangle, polygone convexe quelconque (recentré sur son centre de masse).
//! * Fraction de maille : intersection exacte du polygone et du carré de la maille (découpage de Sutherland–Hodgman) ;
//!   Σ α dx² = aire du corps, sans marches d'escalier.
//! * Mouvement rigide : v(p) = V + ω × (p − G). Contraintes : axes bloqués un par un (x, y, rotation), pivot (rotation pure autour
//!   d'un point fixe), mouvement imposé ou corps fixe — `constrained_velocity` ramène la quantité de mouvement et le moment cinétique
//!   que l'écoulement donne au corps à un mouvement permis.

use serde::{Deserialize, Serialize};

/// côtés du polygone qui représente un disque
pub const DISC_SIDES: usize = 96;

fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

/// aire signée (positive dans le sens trigonométrique)
pub fn polygon_area(p: &[[f64; 2]]) -> f64 {
    let n = p.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        s += cross(p[i], p[(i + 1) % n]);
    }
    0.5 * s
}

/// (aire signée, centre de masse, moment polaire par unité de masse volumique autour de l'ORIGINE)
fn polygon_moments(p: &[[f64; 2]]) -> (f64, [f64; 2], f64) {
    let n = p.len();
    let (mut a, mut cx, mut cy, mut j) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        let (q, r) = (p[i], p[(i + 1) % n]);
        let w = cross(q, r);
        a += w;
        cx += (q[0] + r[0]) * w;
        cy += (q[1] + r[1]) * w;
        j += w * (q[0] * q[0] + q[0] * r[0] + r[0] * r[0] + q[1] * q[1] + q[1] * r[1] + r[1] * r[1]);
    }
    a *= 0.5;
    let c = if a != 0.0 { [cx / (6.0 * a), cy / (6.0 * a)] } else { [0.0, 0.0] };
    (a, c, j / 12.0)
}

/// Forme d'un corps, dans son repère propre (centre de masse à l'origine).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Shape {
    Disc { r: f64 },
    Rect { w: f64, h: f64 },
    /// polygone convexe, sens trigonométrique, centre de masse à l'origine (construit par `Shape::polygon`)
    Polygon { pts: Vec<[f64; 2]> },
}

impl Shape {
    /// Polygone convexe quelconque : vérifié (au moins 3 sommets, convexe, non dégénéré), remis dans le sens trigonométrique et
    /// recentré sur son centre de masse. Retourne aussi la position de ce centre dans le repère des sommets donnés.
    pub fn polygon(pts: &[[f64; 2]]) -> Result<(Shape, [f64; 2]), String> {
        if pts.len() < 3 {
            return Err("un polygone a au moins 3 sommets".into());
        }
        let mut v: Vec<[f64; 2]> = pts.to_vec();
        if polygon_area(&v) < 0.0 {
            v.reverse();
        }
        let n = v.len();
        let scale = v.iter().fold(0.0f64, |m, p| m.max(p[0].abs()).max(p[1].abs())).max(1e-300);
        for i in 0..n {
            let (a, b, c) = (v[i], v[(i + 1) % n], v[(i + 2) % n]);
            if sub(b, a) == [0.0, 0.0] {
                return Err("deux sommets confondus".into());
            }
            if cross(sub(b, a), sub(c, b)) < -1e-12 * scale * scale {
                return Err("polygone non convexe (un corps de forme quelconque se découpera en morceaux convexes, plus tard)".into());
            }
        }
        let (a, g, _) = polygon_moments(&v);
        if !(a > 1e-24 * scale * scale) {
            return Err("polygone d'aire nulle".into());
        }
        Ok((Shape::Polygon { pts: v.iter().map(|p| sub(*p, g)).collect() }, g))
    }

    /// Sommets dans le repère du corps, sens trigonométrique (disque : polygone régulier de même aire).
    pub fn local_vertices(&self) -> Vec<[f64; 2]> {
        match self {
            Shape::Disc { r } => {
                let n = DISC_SIDES as f64;
                let k = 2.0 * std::f64::consts::PI / n;
                // rayon du polygone régulier dont l'aire (n/2) R² sin(2π/n) vaut π r²
                let rr = r * (2.0 * std::f64::consts::PI / (n * k.sin())).sqrt();
                (0..DISC_SIDES).map(|i| [rr * (k * i as f64).cos(), rr * (k * i as f64).sin()]).collect()
            }
            Shape::Rect { w, h } => vec![[-0.5 * w, -0.5 * h], [0.5 * w, -0.5 * h], [0.5 * w, 0.5 * h], [-0.5 * w, 0.5 * h]],
            Shape::Polygon { pts } => pts.clone(),
        }
    }

    pub fn area(&self) -> f64 {
        match self {
            Shape::Disc { r } => std::f64::consts::PI * r * r,
            Shape::Rect { w, h } => w * h,
            Shape::Polygon { pts } => polygon_area(pts),
        }
    }

    /// Moment d'inertie polaire de la forme autour de son centre de masse, par unité de masse volumique (m⁴) : I = ρ·J.
    pub fn polar_moment(&self) -> f64 {
        match self {
            Shape::Disc { r } => 0.5 * std::f64::consts::PI * r.powi(4),
            Shape::Rect { w, h } => w * h * (w * w + h * h) / 12.0,
            Shape::Polygon { pts } => {
                let (a, c, j) = polygon_moments(pts);
                j - a * (c[0] * c[0] + c[1] * c[1])
            }
        }
    }

    /// distance maximale d'un point de la forme à son centre de masse
    pub fn radius(&self) -> f64 {
        self.local_vertices().iter().map(|p| p[0].hypot(p[1])).fold(0.0, f64::max)
    }
}

/// Degrés de liberté : true = libre.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Dof {
    pub x: bool,
    pub y: bool,
    pub rot: bool,
}

impl Dof {
    pub const FREE: Dof = Dof { x: true, y: true, rot: true };
}

/// Mode de mouvement d'un corps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Motion {
    /// couplé à l'écoulement (dans les limites de `dof` et du pivot)
    Free,
    Fixed,
    /// vitesse et rotation imposées (le corps pousse le fluide, le fluide ne le pousse pas)
    Imposed { v: [f64; 2], omega: f64 },
}

/// Corps rigide.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RigidBody {
    pub shape: Shape,
    /// matériau (indice dans `reg.solids`) : ρ, cp, k, couleur
    pub material: usize,
    /// centre de masse (m) et orientation (rad)
    pub pos: [f64; 2],
    pub angle: f64,
    /// vitesse du centre de masse (m/s) et vitesse de rotation (rad/s, sens trigonométrique)
    pub vel: [f64; 2],
    pub omega: f64,
    pub dof: Dof,
    /// point fixe autour duquel le corps tourne (charnière d'un clapet)
    pub pivot: Option<[f64; 2]>,
    pub motion: Motion,
    /// une température par corps (étape 5)
    pub temp: f64,
}

impl RigidBody {
    pub fn new(shape: Shape, material: usize, pos: [f64; 2], temp: f64) -> RigidBody {
        RigidBody { shape, material, pos, angle: 0.0, vel: [0.0, 0.0], omega: 0.0, dof: Dof::FREE, pivot: None, motion: Motion::Free, temp }
    }

    /// Sommets dans le repère du domaine.
    pub fn world_vertices(&self) -> Vec<[f64; 2]> {
        let (s, c) = self.angle.sin_cos();
        self.shape.local_vertices().iter().map(|p| [self.pos[0] + c * p[0] - s * p[1], self.pos[1] + s * p[0] + c * p[1]]).collect()
    }

    pub fn mass(&self, rho: f64) -> f64 {
        rho * self.shape.area()
    }

    /// moment d'inertie autour du centre de masse
    pub fn inertia(&self, rho: f64) -> f64 {
        rho * self.shape.polar_moment()
    }

    /// vitesse du point p du corps
    pub fn velocity_at(&self, p: [f64; 2]) -> [f64; 2] {
        let r = sub(p, self.pos);
        [self.vel[0] - self.omega * r[1], self.vel[1] + self.omega * r[0]]
    }

    /// Mouvement permis à partir de la quantité de mouvement `p` et du moment cinétique `l` (autour du centre de masse) que
    /// l'écoulement donne au corps, de masse `m` et d'inertie `i` :
    /// * libre : V = P/M, ω = L/I, composantes bloquées mises à zéro (l'énergie cinétique est diagonale en (Vx, Vy, ω) autour du
    ///   centre de masse : c'est le mouvement permis le plus proche) ;
    /// * pivot q : rotation pure autour de q, ω = L_q / I_q avec L_q = L + (G − q) × P et I_q = I + M |G − q|² (Huygens) ;
    ///   rotation bloquée sur un pivot = corps fixe ;
    /// * imposé : la loi ; fixe : rien.
    pub fn constrained_velocity(&self, m: f64, i: f64, p: [f64; 2], l: f64) -> ([f64; 2], f64) {
        match self.motion {
            Motion::Fixed => ([0.0, 0.0], 0.0),
            Motion::Imposed { v, omega } => (v, omega),
            Motion::Free => match self.pivot {
                Some(q) => {
                    let r = sub(self.pos, q);
                    let lq = l + cross(r, p);
                    let iq = i + m * (r[0] * r[0] + r[1] * r[1]);
                    let w = if self.dof.rot && iq > 0.0 { lq / iq } else { 0.0 };
                    ([-w * r[1], w * r[0]], w)
                }
                None => (
                    [if self.dof.x && m > 0.0 { p[0] / m } else { 0.0 }, if self.dof.y && m > 0.0 { p[1] / m } else { 0.0 }],
                    if self.dof.rot && i > 0.0 { l / i } else { 0.0 },
                ),
            },
        }
    }

    /// Avance la position de dt avec (vel, omega) ; autour d'un pivot, rotation exacte (la distance au pivot est gardée).
    pub fn advance(&mut self, dt: f64) {
        let a = self.omega * dt;
        match self.pivot {
            Some(q) => {
                let r = sub(self.pos, q);
                let (s, c) = a.sin_cos();
                self.pos = [q[0] + c * r[0] - s * r[1], q[1] + s * r[0] + c * r[1]];
            }
            None => {
                self.pos[0] += self.vel[0] * dt;
                self.pos[1] += self.vel[1] * dt;
            }
        }
        self.angle += a;
    }
}

/// Découpe le polygone convexe `poly` par la boîte [x0, x1] × [y0, y1] (Sutherland–Hodgman).
fn clip_to_box(poly: &[[f64; 2]], x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<[f64; 2]> {
    let mut cur: Vec<[f64; 2]> = poly.to_vec();
    // (axe, borne, garder ≥ borne ?)
    for (axis, bound, keep_above) in [(0usize, x0, true), (0, x1, false), (1, y0, true), (1, y1, false)] {
        if cur.is_empty() {
            break;
        }
        let inside = |p: &[f64; 2]| if keep_above { p[axis] >= bound } else { p[axis] <= bound };
        let mut out = Vec::with_capacity(cur.len() + 4);
        for k in 0..cur.len() {
            let (a, b) = (cur[k], cur[(k + 1) % cur.len()]);
            let (ia, ib) = (inside(&a), inside(&b));
            if ia {
                out.push(a);
            }
            if ia != ib {
                let t = (bound - a[axis]) / (b[axis] - a[axis]);
                let mut q = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
                q[axis] = bound;
                out.push(q);
            }
        }
        cur = out;
    }
    cur
}

/// Fraction de chaque maille (nx × ny, pas dx) occupée par les corps ; la somme de corps qui se chevauchent est plafonnée à 1.
pub fn body_fractions(bodies: &[RigidBody], nx: usize, ny: usize, dx: f64) -> Vec<f64> {
    let mut f = vec![0.0; nx * ny];
    let cell_area = dx * dx;
    for b in bodies {
        let poly = b.world_vertices();
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for p in &poly {
            x0 = x0.min(p[0]);
            y0 = y0.min(p[1]);
            x1 = x1.max(p[0]);
            y1 = y1.max(p[1]);
        }
        let (lx, ly) = (nx as f64 * dx, ny as f64 * dx);
        if x1 <= 0.0 || y1 <= 0.0 || x0 >= lx || y0 >= ly {
            continue;
        }
        let ci = |x: f64, n: usize| ((x / dx).floor().max(0.0) as usize).min(n - 1);
        for j in ci(y0, ny)..=ci(y1, ny) {
            for i in ci(x0, nx)..=ci(x1, nx) {
                let (cx0, cy0) = (i as f64 * dx, j as f64 * dx);
                let part = clip_to_box(&poly, cx0, cy0, cx0 + dx, cy0 + dx);
                let a = polygon_area(&part);
                if a > 0.0 {
                    f[j * nx + i] += a / cell_area;
                }
            }
        }
    }
    for a in &mut f {
        *a = a.clamp(0.0, 1.0);
    }
    f
}

impl super::multiphase::MultiphaseSim {
    /// la scène a-t-elle des corps rigides ?
    #[inline]
    pub fn has_bodies(&self) -> bool {
        !self.bodies.is_empty()
    }

    /// Recalcule `body_frac` depuis la position des corps (à appeler après en avoir ajouté, retiré ou déplacé).
    pub fn update_body_fractions(&mut self) {
        if self.bodies.is_empty() {
            if self.body_frac.iter().any(|&a| a != 0.0) {
                self.body_frac.iter_mut().for_each(|a| *a = 0.0);
            }
            return;
        }
        self.body_frac = body_fractions(&self.bodies, self.nx, self.ny, self.dx);
    }

    /// Volume occupé par les corps dans la maille c, ajouté au volume des fluides dans la fermeture (0 sans corps).
    #[inline]
    pub(crate) fn body_volume(&self, c: usize) -> f64 {
        if self.bodies.is_empty() {
            0.0
        } else {
            self.body_frac[c]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn shapes_have_their_exact_area_and_inertia() {
        let d = Shape::Disc { r: 0.05 };
        assert!((polygon_area(&d.local_vertices()) / (PI * 0.0025) - 1.0).abs() < 1e-12, "aire du polygone du disque");
        assert!((d.radius() / 0.05 - 1.0).abs() < 4e-4);
        let r = Shape::Rect { w: 0.3, h: 0.1 };
        assert!((r.area() - 0.03).abs() < 1e-15 && (r.polar_moment() - 0.03 * (0.09 + 0.01) / 12.0).abs() < 1e-15);
        // le même rectangle donné en polygone : mêmes aire et inertie
        let (p, g) = Shape::polygon(&[[0.0, 0.0], [0.3, 0.0], [0.3, 0.1], [0.0, 0.1]]).unwrap();
        assert!((g[0] - 0.15).abs() < 1e-15 && (g[1] - 0.05).abs() < 1e-15);
        assert!((p.area() - r.area()).abs() < 1e-15 && (p.polar_moment() / r.polar_moment() - 1.0).abs() < 1e-12);
        // triangle rectangle de côtés 3 : aire 4.5, centre (1, 1), J = ab(a² + b²)/36 = 4.5 ; donné à l'envers (sens horaire)
        let (t, g) = Shape::polygon(&[[0.0, 0.0], [0.0, 3.0], [3.0, 0.0]]).unwrap();
        assert!((t.area() - 4.5).abs() < 1e-12 && (g[0] - 1.0).abs() < 1e-12 && (g[1] - 1.0).abs() < 1e-12);
        assert!((t.polar_moment() - 4.5).abs() < 1e-12, "J = {}", t.polar_moment());
        // non convexe (L), dégénéré : refusés
        assert!(Shape::polygon(&[[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]]).is_err());
        assert!(Shape::polygon(&[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]).is_err());
    }

    #[test]
    fn cell_fractions_add_up_to_the_area_wherever_the_body_is_and_however_it_turns() {
        let (nx, ny, dx) = (40usize, 40usize, 0.01);
        for (shape, pos) in [
            (Shape::Disc { r: 0.05 }, [0.123, 0.217]),
            (Shape::Disc { r: 0.004 }, [0.2, 0.2]),
            (Shape::Rect { w: 0.11, h: 0.03 }, [0.2, 0.15]),
            (Shape::polygon(&[[0.0, 0.0], [0.08, 0.01], [0.05, 0.07]]).unwrap().0, [0.25, 0.25]),
        ] {
            let area = shape.area();
            for k in 0..12 {
                let mut b = RigidBody::new(shape.clone(), 0, pos, 300.0);
                b.angle = 0.37 * k as f64;
                let f = body_fractions(&[b], nx, ny, dx);
                assert!(f.iter().all(|&a| (0.0..=1.0).contains(&a)));
                let total: f64 = f.iter().sum::<f64>() * dx * dx;
                assert!((total / area - 1.0).abs() < 1e-12, "{shape:?} à {} rad : {total} pour {area}", b_angle(k));
            }
        }
        // maille entièrement couverte : 1 ; corps à moitié hors du domaine : la part intérieure
        let b = RigidBody::new(Shape::Disc { r: 0.08 }, 0, [0.2, 0.2], 300.0);
        let f = body_fractions(&[b], nx, ny, dx);
        assert!((f[20 * nx + 20] - 1.0).abs() < 1e-12);
        let half = RigidBody::new(Shape::Rect { w: 0.1, h: 0.1 }, 0, [0.0, 0.2], 300.0);
        let f = body_fractions(&[half], nx, ny, dx);
        assert!((f.iter().sum::<f64>() * dx * dx - 0.005).abs() < 1e-12);
    }

    fn b_angle(k: usize) -> f64 {
        0.37 * k as f64
    }

    #[test]
    fn a_rectangle_aligned_with_the_grid_fills_whole_cells() {
        let (nx, ny, dx) = (20usize, 20usize, 0.01);
        let b = RigidBody::new(Shape::Rect { w: 0.04, h: 0.02 }, 0, [0.10, 0.10], 300.0);
        let f = body_fractions(&[b], nx, ny, dx);
        for j in 0..ny {
            for i in 0..nx {
                let inside = (8..12).contains(&i) && (9..11).contains(&j);
                let a = f[j * nx + i];
                assert!((a - if inside { 1.0 } else { 0.0 }).abs() < 1e-12, "maille ({i}, {j}) : {a}");
            }
        }
    }

    #[test]
    fn constraints_keep_only_the_allowed_motion() {
        let mut b = RigidBody::new(Shape::Disc { r: 0.1 }, 0, [0.0, 0.0], 300.0);
        let (m, i) = (b.mass(1000.0), b.inertia(1000.0));
        let (p, l) = ([2.0 * m, -3.0 * m], 0.5 * i);
        assert_eq!(b.constrained_velocity(m, i, p, l), ([2.0, -3.0], 0.5));
        b.dof = Dof { x: false, y: true, rot: false };
        assert_eq!(b.constrained_velocity(m, i, p, l), ([0.0, -3.0], 0.0));
        // disque accroché à son bord (pivot en (r, 0)), poussé vers le bas en son centre : il tourne autour du pivot,
        // ω = r M v / (I + M r²) = v / (1.5 r), et son centre descend à 2/3 de v
        b.dof = Dof::FREE;
        b.pivot = Some([0.1, 0.0]);
        let (v, w) = b.constrained_velocity(m, i, [0.0, -m], 0.0);
        assert!((w - 1.0 / 0.15).abs() < 1e-12 && v[0].abs() < 1e-15 && (v[1] + 2.0 / 3.0).abs() < 1e-12, "{v:?} {w}");
        b.dof.rot = false;
        assert_eq!(b.constrained_velocity(m, i, [0.0, -m], 0.0), ([0.0, -0.0], 0.0));
        b.motion = Motion::Imposed { v: [0.1, 0.0], omega: 0.0 };
        assert_eq!(b.constrained_velocity(m, i, p, l), ([0.1, 0.0], 0.0));
        b.motion = Motion::Fixed;
        assert_eq!(b.constrained_velocity(m, i, p, l), ([0.0, 0.0], 0.0));
        // vitesse d'un point : V + ω × r
        let mut c = RigidBody::new(Shape::Rect { w: 1.0, h: 1.0 }, 0, [1.0, 1.0], 300.0);
        c.vel = [1.0, 0.0];
        c.omega = 2.0;
        assert_eq!(c.velocity_at([1.0, 2.0]), [-1.0, 0.0]);
    }

    #[test]
    fn a_body_in_the_scene_counts_in_the_closure_is_saved_and_keeps_the_scene_on_the_cpu() {
        use crate::simulation::gpu_multiphase::gpu_unsupported_reason;
        use crate::simulation::mp_state::{state_from_bytes, state_to_bytes};
        use crate::simulation::mp_view::{build_scenario, render_rgba, MpDisplay, MpScenario, RenderOpts};
        let mut s = build_scenario(MpScenario::Empty, 40, 30, 0.01, 9.81);
        assert!(!s.has_bodies() && gpu_unsupported_reason(&s).is_none());
        let c = s.idx(20, 15);
        let (v0, p) = (s.cell_volume(c, s.p[c]), s.p[c]);
        let mut b = RigidBody::new(Shape::Disc { r: 0.05 }, 0, [0.205, 0.155], 300.0);
        b.angle = 0.3;
        b.dof = Dof { x: false, y: true, rot: true };
        s.bodies.push(b);
        s.update_body_fractions();
        // maille du centre entièrement couverte : son volume compte en plus de celui du fluide (que l'étape 1 chassera)
        assert!((s.body_frac[c] - 1.0).abs() < 1e-12);
        assert!((s.cell_volume(c, p) - v0 - 1.0).abs() < 1e-12);
        let total: f64 = s.body_frac.iter().sum::<f64>() * s.dx * s.dx;
        assert!((total / (PI * 0.0025) - 1.0).abs() < 1e-12);
        // le GPU refuse la scène, en disant pourquoi
        assert!(gpu_unsupported_reason(&s).unwrap().contains("corps rigide"));
        // l'état complet garde les corps et leur fraction
        let bytes = state_to_bytes(&s, None, None, None).unwrap();
        let back = state_from_bytes(&bytes).unwrap().sim;
        assert_eq!(back.bodies, s.bodies);
        assert_eq!(back.body_frac, s.body_frac);
        // l'image montre le corps : la maille du centre a la couleur du matériau
        let mut rgba = vec![];
        render_rgba(&s, RenderOpts::new(MpDisplay::Phases, 100.0, 0.0), &mut rgba);
        let px = ((30 - 1 - 15) * 40 + 20) * 4;
        let col = s.reg.solids[0].color;
        for q in 0..3 {
            assert!((rgba[px + q] as f32 - col[q] * 255.0).abs() <= 1.0, "couleur du corps");
        }
        // retiré : la fermeture redevient celle du fluide seul
        s.bodies.clear();
        s.update_body_fractions();
        assert_eq!(s.cell_volume(c, p), v0);
        assert!(s.body_frac.iter().all(|&a| a == 0.0));
    }

    #[test]
    fn rotation_about_a_pivot_keeps_the_distance_and_translation_moves_the_centre() {
        let mut b = RigidBody::new(Shape::Rect { w: 0.2, h: 0.02 }, 0, [0.3, 0.2], 300.0);
        b.pivot = Some([0.2, 0.2]);
        b.omega = 1.5;
        for _ in 0..1000 {
            b.advance(1e-3);
        }
        let d = (b.pos[0] - 0.2).hypot(b.pos[1] - 0.2);
        assert!((d - 0.1).abs() < 1e-12 && (b.angle - 1.5).abs() < 1e-12);
        assert!((b.pos[0] - (0.2 + 0.1 * 1.5f64.cos())).abs() < 1e-12 && (b.pos[1] - (0.2 + 0.1 * 1.5f64.sin())).abs() < 1e-12);
        let mut c = RigidBody::new(Shape::Disc { r: 0.01 }, 0, [0.0, 0.0], 300.0);
        c.vel = [1.0, -2.0];
        c.advance(0.5);
        assert_eq!(c.pos, [0.5, -1.0]);
    }
}
