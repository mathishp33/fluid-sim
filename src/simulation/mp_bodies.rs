//! Corps rigides — solides mobiles (conception et choix : SOLID.md).
//!
//! ÉTAPES 0 ET 1 (SOLID.md § 7–8) : corps rigides FIXES, posés par la commande `corps` des protocoles ou l'outil « Corps » de la
//! fenêtre d'expérience. Sans corps, le calcul est inchangé au bit près (empreintes physiques des scénarios identiques).
//!
//! * La fraction de chaque maille occupée par les corps (`MultiphaseSim::body_frac`) entre dans la fermeture : Σ m/ρ + α_corps = 1
//!   (`cell_volume`, `cell_closure_g`).
//! * Poser un corps chasse le fluide de son volume (`add_body`) : une maille entièrement couverte n'en garde qu'une trace
//!   (`FLUID_TRACE`).
//! * Chaque face n'est ouverte au fluide qu'à 1 − α (de la plus couverte de ses deux mailles ; 0 si l'une est entièrement couverte :
//!   `u_open`, `v_open`). La projection (`face_velocities`, `build_stencil`) en tient compte, et la vitesse de la part couverte est
//!   celle du corps, nulle. La fermeture reste exacte autour des corps : pas de bruit de pression dans l'eau, très raide, où une erreur
//!   de volume de 1e-6 fausserait la pression de 2 kPa. La percolation, la conduction et la diffusion des espèces passent par la même
//!   ouverture ; sans cela, la trace de fluide d'une maille couverte, de capacité quasi nulle, rendait la conduction explicite
//!   instable.
//! * Ce qui reste (corps mobiles, chaleur du corps, grains, GPU) : SOLID.md § 7. Le GPU refuse les scènes avec des corps.
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
/// fraction de volume de fluide gardée dans une maille entièrement couverte par un corps (le solveur suppose du fluide dans toute
/// maille non solide) ; une face dont une maille est couverte au-delà de 1 − FLUID_TRACE est fermée
pub const FLUID_TRACE: f64 = 1e-6;

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
    /// Mouvement imposé : le corps pousse le fluide, le fluide ne le pousse pas. Vitesse V(t) = v + osc_v sin(2πt/period) et rotation
    /// ω(t) = omega + osc_omega sin(2πt/period) (sans oscillation si period ≤ 0) ; sur un pivot, seule la rotation compte.
    Imposed { v: [f64; 2], omega: f64, osc_v: [f64; 2], osc_omega: f64, period: f64 },
}

impl Motion {
    /// vitesse et rotation constantes
    pub fn imposed(v: [f64; 2], omega: f64) -> Motion {
        Motion::Imposed { v, omega, osc_v: [0.0, 0.0], osc_omega: 0.0, period: 0.0 }
    }
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
            Motion::Imposed { .. } => (self.vel, self.omega),
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
        self.displace(self.vel[0] * dt, self.vel[1] * dt, a);
    }

    /// translation (dx, dy) puis rotation da autour du centre de masse ; sur un pivot, rotation da autour du pivot seulement
    fn displace(&mut self, dx: f64, dy: f64, da: f64) {
        match self.pivot {
            Some(q) => {
                let r = sub(self.pos, q);
                let (s, c) = da.sin_cos();
                self.pos = [q[0] + c * r[0] - s * r[1], q[1] + s * r[0] + c * r[1]];
            }
            None => {
                self.pos[0] += dx;
                self.pos[1] += dy;
            }
        }
        self.angle += da;
    }

    /// corps dont le mouvement est imposé ?
    pub fn is_imposed(&self) -> bool {
        matches!(self.motion, Motion::Imposed { .. })
    }

    /// Vitesse du centre de masse et rotation imposées à l'instant t (sur un pivot : la vitesse du centre est ω × (G − pivot)).
    pub fn rates(&self, t: f64) -> ([f64; 2], f64) {
        match &self.motion {
            Motion::Fixed => ([0.0, 0.0], 0.0),
            Motion::Free => (self.vel, self.omega),
            Motion::Imposed { v, omega, osc_v, osc_omega, period } => {
                let s = if *period > 0.0 { (2.0 * std::f64::consts::PI * t / period).sin() } else { 0.0 };
                let w = omega + osc_omega * s;
                match self.pivot {
                    Some(q) => {
                        let r = sub(self.pos, q);
                        ([-w * r[1], w * r[0]], w)
                    }
                    None => ([v[0] + osc_v[0] * s, v[1] + osc_v[1] * s], w),
                }
            }
        }
    }

    /// Mouvement imposé de t0 à t1 : intégrale EXACTE de la loi (la partie sinusoïdale revient pile à son point de départ à chaque
    /// période, quel que soit le pas) ; vitesses mises à celles de t1.
    pub fn move_imposed(&mut self, t0: f64, t1: f64) {
        let Motion::Imposed { v, omega, osc_v, osc_omega, period } = self.motion.clone() else { return };
        let dt = t1 - t0;
        // ∫ sin(2πt/P) dt = P/(2π) (cos(2π t0/P) − cos(2π t1/P))
        let k = if period > 0.0 {
            let w = 2.0 * std::f64::consts::PI / period;
            ((w * t0).cos() - (w * t1).cos()) / w
        } else {
            0.0
        };
        self.displace(v[0] * dt + osc_v[0] * k, v[1] * dt + osc_v[1] * k, omega * dt + osc_omega * k);
        let (vv, w) = self.rates(t1);
        self.vel = vv;
        self.omega = w;
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
    body_fractions_owned(bodies, nx, ny, dx).0
}

/// Comme `body_fractions`, avec pour chaque maille l'indice du corps qui la couvre le plus (−1 : aucun).
pub fn body_fractions_owned(bodies: &[RigidBody], nx: usize, ny: usize, dx: f64) -> (Vec<f64>, Vec<i32>) {
    let mut f = vec![0.0; nx * ny];
    let mut owner = vec![-1i32; nx * ny];
    let mut best = vec![0.0f64; nx * ny];
    let cell_area = dx * dx;
    for (bi, b) in bodies.iter().enumerate() {
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
                    let c = j * nx + i;
                    f[c] += a / cell_area;
                    if a > best[c] {
                        best[c] = a;
                        owner[c] = bi as i32;
                    }
                }
            }
        }
    }
    for a in &mut f {
        *a = a.clamp(0.0, 1.0);
    }
    (f, owner)
}

impl super::multiphase::MultiphaseSim {
    /// la scène a-t-elle des corps rigides ?
    #[inline]
    pub fn has_bodies(&self) -> bool {
        !self.bodies.is_empty()
    }

    /// Recalcule `body_frac` depuis la position des corps (à appeler après en avoir ajouté, retiré ou déplacé), et, s'il y a des corps
    /// en mouvement, la vitesse normale du corps aux faces bordées par une maille couverte (celle du corps qui couvre le plus la plus
    /// couverte des deux mailles, au centre de la face).
    pub fn update_body_fractions(&mut self) {
        if self.bodies.is_empty() {
            if self.body_frac.iter().any(|&a| a != 0.0) {
                self.body_frac.iter_mut().for_each(|a| *a = 0.0);
            }
            self.body_vel_u.clear();
            self.body_vel_v.clear();
            return;
        }
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let (frac, owner) = body_fractions_owned(&self.bodies, nx, ny, dx);
        self.body_frac = frac;
        if !self.has_moving_bodies() {
            self.body_vel_u.clear();
            self.body_vel_v.clear();
            return;
        }
        let pick = |a: Option<usize>, b: Option<usize>| -> Option<usize> {
            let fa = a.map_or(0.0, |c| self.body_frac[c]);
            let fb = b.map_or(0.0, |c| self.body_frac[c]);
            let c = if fa >= fb { a } else { b }?;
            (self.body_frac[c] > 0.0 && owner[c] >= 0).then(|| owner[c] as usize)
        };
        let mut vu = vec![0.0; (nx + 1) * ny];
        for j in 0..ny {
            for i in 0..=nx {
                let (a, b) = ((i > 0).then(|| j * nx + i - 1), (i < nx).then(|| j * nx + i));
                if let Some(k) = pick(a, b) {
                    vu[j * (nx + 1) + i] = self.bodies[k].velocity_at([i as f64 * dx, (j as f64 + 0.5) * dx])[0];
                }
            }
        }
        let mut vv = vec![0.0; nx * (ny + 1)];
        for j in 0..=ny {
            for i in 0..nx {
                let (a, b) = ((j > 0).then(|| (j - 1) * nx + i), (j < ny).then(|| j * nx + i));
                if let Some(k) = pick(a, b) {
                    vv[j * nx + i] = self.bodies[k].velocity_at([(i as f64 + 0.5) * dx, j as f64 * dx])[1];
                }
            }
        }
        self.body_vel_u = vu;
        self.body_vel_v = vv;
    }

    /// Volume occupé par les corps dans la maille c, ajouté au volume des fluides dans la fermeture (0 sans corps) ; plafonné à
    /// 1 − FLUID_TRACE (la trace de fluide d'une maille entièrement couverte).
    #[inline]
    pub(crate) fn body_volume(&self, c: usize) -> f64 {
        if self.bodies.is_empty() {
            0.0
        } else {
            self.body_frac[c].min(1.0 - FLUID_TRACE)
        }
    }

    /// Ouverture d'une face entre deux mailles de fractions de corps a et b : 1 − max(a, b), 0 si l'une est entièrement couverte.
    /// Une maille couverte à 1 − φ a ainsi toutes ses faces ouvertes à φ au plus : une face ne débite jamais plus que la place de
    /// fluide de ses mailles, conductance et capacité y diminuent ensemble (les schémas explicites gardent le pas du fluide pur). Le
    /// fluide qu'un corps en mouvement balaie n'a donc pas à sortir par une face presque fermée : il est déplacé (`push_fluid`).
    #[inline]
    fn opening(a: f64, b: f64) -> f64 {
        let m = a.max(b);
        if m >= 1.0 - FLUID_TRACE {
            0.0
        } else {
            1.0 - m
        }
    }

    /// fractions de corps des deux mailles de la face u (i, j) (une face de bord : sa seule maille)
    fn u_cells(&self, i: usize, j: usize) -> (f64, f64) {
        let f = &self.body_frac;
        match (i > 0, i < self.nx) {
            (true, true) => (f[self.idx(i - 1, j)], f[self.idx(i, j)]),
            (false, true) => (f[self.idx(0, j)], 0.0),
            (true, false) => (f[self.idx(self.nx - 1, j)], 0.0),
            _ => (0.0, 0.0),
        }
    }

    fn v_cells(&self, i: usize, j: usize) -> (f64, f64) {
        let f = &self.body_frac;
        match (j > 0, j < self.ny) {
            (true, true) => (f[self.idx(i, j - 1)], f[self.idx(i, j)]),
            (false, true) => (f[self.idx(i, 0)], 0.0),
            (true, false) => (f[self.idx(i, self.ny - 1)], 0.0),
            _ => (0.0, 0.0),
        }
    }

    /// Ouverture de la face u (i, j) — entre les mailles (i − 1, j) et (i, j). 1 sans corps.
    pub(crate) fn u_open(&self, i: usize, j: usize) -> f64 {
        if self.bodies.is_empty() {
            return 1.0;
        }
        let (a, b) = self.u_cells(i, j);
        Self::opening(a, b)
    }

    /// Ouverture de la face v (i, j) — entre les mailles (i, j − 1) et (i, j). 1 sans corps.
    pub(crate) fn v_open(&self, i: usize, j: usize) -> f64 {
        if self.bodies.is_empty() {
            return 1.0;
        }
        let (a, b) = self.v_cells(i, j);
        Self::opening(a, b)
    }

    /// Vitesse normale du corps à la face u / v bordée par une maille couverte (0 sans corps ou corps fixe).
    #[inline]
    pub(crate) fn u_body_vel(&self, f: usize) -> f64 {
        self.body_vel_u.get(f).copied().unwrap_or(0.0)
    }
    #[inline]
    pub(crate) fn v_body_vel(&self, f: usize) -> f64 {
        self.body_vel_v.get(f).copied().unwrap_or(0.0)
    }

    /// Plus grande ouverture des quatre faces de la maille c (≤ 1 − α_corps de c) : borne des échanges de la maille.
    pub(crate) fn cell_open_max(&self, c: usize) -> f64 {
        let (i, j) = (c % self.nx, c / self.nx);
        self.u_open(i, j).max(self.u_open(i + 1, j)).max(self.v_open(i, j)).max(self.v_open(i, j + 1))
    }

    /// un corps au moins a-t-il un mouvement imposé ?
    pub fn has_moving_bodies(&self) -> bool {
        self.bodies.iter().any(|b| b.is_imposed())
    }

    /// Avance les corps à mouvement imposé de t0 à t1 (intégrale exacte de leur loi), recalcule fractions et faces, puis déplace le
    /// fluide qu'ils ont balayé (`push_fluid`).
    pub(crate) fn move_bodies(&mut self, t0: f64, t1: f64) {
        let old = self.body_frac.clone();
        for b in &mut self.bodies {
            b.move_imposed(t0, t1);
        }
        self.update_body_fractions();
        self.push_fluid(&old);
    }

    /// Le fluide que les corps viennent de balayer dans une maille passe à la maille voisine du côté opposé au corps (direction
    /// −∇α_corps) ; une maille qu'ils découvrent prend le sien à cette voisine. Masses et énergies sont transférées telles quelles :
    /// conservatif, et la fermeture de la maille balayée reste exacte. La projection n'a plus qu'à évacuer le petit excédent (au plus
    /// le volume balayé en un pas) d'une maille bien ouverte. Sans cela, ce fluide devait sortir par une face de plus en plus fermée à
    /// mesure que la maille se couvrait : pression qui tend vers l'infini.
    fn push_fluid(&mut self, old: &[f64]) {
        let (nx, ny) = (self.nx, self.ny);
        let cap = 1.0 - FLUID_TRACE;
        let al = |s: &Self, i: isize, j: isize, own: f64| -> f64 {
            if i < 0 || j < 0 || i >= nx as isize || j >= ny as isize { own } else { s.body_frac[j as usize * nx + i as usize] }
        };
        for c in 0..nx * ny {
            if self.is_solid(c) {
                continue;
            }
            let (a0, a1) = (old[c].min(cap), self.body_frac[c].min(cap));
            let d = a1 - a0;
            if d.abs() < 1e-12 {
                continue;
            }
            let (i, j) = ((c % nx) as isize, (c / nx) as isize);
            let own = self.body_frac[c];
            // direction qui s'éloigne du corps
            let g = [-(al(self, i + 1, j, own) - al(self, i - 1, j, own)), -(al(self, i, j + 1, own) - al(self, i, j - 1, own))];
            if g[0].abs() + g[1].abs() < 1e-12 {
                continue;
            }
            let mut best: Option<(usize, f64)> = None;
            for (di, dj) in [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
                let (ii, jj) = (i + di, j + dj);
                if ii < 0 || jj < 0 || ii >= nx as isize || jj >= ny as isize {
                    continue;
                }
                let n = jj as usize * nx + ii as usize;
                if self.is_solid(n) || self.body_frac[n] >= cap {
                    continue;
                }
                let score = g[0] * di as f64 + g[1] * dj as f64;
                if score > 0.0 && best.map_or(true, |(_, s)| score > s) {
                    best = Some((n, score));
                }
            }
            let Some((n, _)) = best else { continue };
            if d > 0.0 {
                // pousser la part balayée du fluide de c vers n
                self.transfer_fluid(c, n, (d / (1.0 - a0)).min(1.0));
            } else {
                // reprendre à n de quoi remplir la place libérée (au plus la moitié de son fluide ; la projection fait le reste)
                let phi_n = 1.0 - self.body_frac[n].min(cap);
                self.transfer_fluid(n, c, ((-d) / phi_n).min(0.5));
            }
            // travail du corps sur le fluide, p·ΔV : le déplacement n'en porte pas (la projection répartit ensuite l'excédent ou le
            // manque de n sans travail net), sans lui le gaz qu'un piston comprime ne chauffait pas comme une adiabatique
            let w = self.p[n] * d;
            self.add_body_work(n, w);
        }
        self.refresh();
    }

    /// Énergie `de` (J/m³) apportée au fluide de la maille c par le travail d'un corps, répartie entre ses groupes comme le travail
    /// de pression (`pressure_work` : au prorata de C_g dT_g/dp — c'est le gaz qui se comprime, pas le liquide qui l'entoure).
    fn add_body_work(&mut self, c: usize, de: f64) {
        use super::phases::MAX_PHASES;
        if de == 0.0 || !de.is_finite() {
            return;
        }
        let ng = self.groups.len();
        let mut t = [0.0f64; MAX_PHASES];
        for g in 0..ng {
            t[g] = self.tg[g][c];
        }
        let (mut gam, mut ks) = ([0.0f64; MAX_PHASES], [0.0f64; MAX_PHASES]);
        self.cell_closure_g(c, self.p[c], &t[..ng], 1.0, &mut gam[..ng], &mut ks[..ng]);
        let ksum: f64 = ks[..ng].iter().map(|k| k.max(0.0)).sum();
        let asum: f64 = (0..ng).filter(|&g| self.group_mass(g, c) > 0.0).map(|g| self.group_alpha(g, c)).sum::<f64>().max(1e-300);
        let mut used = 0.0;
        for g in 0..ng {
            if self.group_mass(g, c) <= 0.0 {
                continue;
            }
            let w = if ksum > 0.0 { ks[g].max(0.0) / ksum } else { self.group_alpha(g, c) / asum };
            self.eg[g][c] += de * w;
            used += de * w;
        }
        self.e[c] += used;
    }

    /// transfère la fraction q du fluide (masses, énergies) de la maille `from` à la maille `to`
    fn transfer_fluid(&mut self, from: usize, to: usize, q: f64) {
        if !(q > 0.0) {
            return;
        }
        for k in 0..self.mass.len() {
            let m = self.mass[k][from] * q;
            self.mass[k][from] -= m;
            self.mass[k][to] += m;
        }
        for g in 0..self.eg.len() {
            let e = self.eg[g][from] * q;
            self.eg[g][from] -= e;
            self.eg[g][to] += e;
        }
        let e = self.e[from] * q;
        self.e[from] -= e;
        self.e[to] += e;
    }

    /// vitesse maximale d'un point de la surface d'un corps (CFL)
    pub(crate) fn body_surface_speed(&self) -> f64 {
        self.bodies.iter().map(|b| b.vel[0].hypot(b.vel[1]) + b.omega.abs() * b.shape.radius()).fold(0.0, f64::max)
    }

    /// Pose un corps (étape 1 : corps fixes). Sa fraction de maille est calculée et le fluide qu'il recouvre est chassé : dans chaque
    /// maille, masses et énergies sont ramenées à la place restante 1 − α_corps (composition et température gardées ; une trace
    /// FLUID_TRACE reste dans les mailles entièrement couvertes). Le fluide chassé disparaît, comme quand on peint un solide.
    /// Retourne l'indice du corps.
    pub fn add_body(&mut self, b: RigidBody) -> Result<usize, String> {
        if b.material >= self.reg.solids.len() {
            return Err(format!("matériau n° {} inconnu ({} solides)", b.material, self.reg.solids.len()));
        }
        let ok = match &b.shape {
            Shape::Disc { r } => *r > 0.0 && r.is_finite(),
            Shape::Rect { w, h } => *w > 0.0 && *h > 0.0 && w.is_finite() && h.is_finite(),
            Shape::Polygon { pts } => pts.len() >= 3 && polygon_area(pts) > 0.0,
        };
        if !ok || !b.pos.iter().chain([b.angle].iter()).all(|x| x.is_finite()) {
            return Err("corps de forme ou de position invalide".into());
        }
        let mut b = b;
        if b.is_imposed() {
            // vitesses de la loi à l'instant de la pose (faces du premier pas)
            let (v, w) = b.rates(self.time);
            b.vel = v;
            b.omega = w;
        }
        let old = self.body_frac.clone();
        self.bodies.push(b);
        self.update_body_fractions();
        self.displace_fluid(&old);
        Ok(self.bodies.len() - 1)
    }

    /// Retire tous les corps ; le fluide resté dans les mailles qu'ils couvraient reprend toute la place (inverse exact de
    /// `add_body` sur un état qui n'a pas bougé, aux arrondis près).
    pub fn clear_bodies(&mut self) {
        if self.bodies.is_empty() {
            return;
        }
        let old = self.body_frac.clone();
        self.bodies.clear();
        self.body_frac.iter_mut().for_each(|a| *a = 0.0);
        self.displace_fluid(&old);
    }

    /// Ramène le fluide de chaque maille à la place que lui laissent les corps (anciennes fractions `old` → `body_frac`) :
    /// masses et énergies multipliées par (1 − α_nouveau) / (1 − α_ancien), température et composition gardées.
    fn displace_fluid(&mut self, old: &[f64]) {
        let (np, ng) = (self.reg.phases.len(), self.eg.len());
        let cap = 1.0 - FLUID_TRACE;
        for c in 0..self.nx * self.ny {
            if self.is_solid(c) {
                continue;
            }
            let (a0, a1) = (old[c].min(cap), self.body_frac[c].min(cap));
            if a0 == a1 {
                continue;
            }
            let f = (1.0 - a1) / (1.0 - a0);
            for k in 0..np {
                self.mass[k][c] *= f;
            }
            for g in 0..ng {
                self.eg[g][c] *= f;
            }
            self.e[c] *= f;
        }
        self.refresh();
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
        // imposé : la loi (vitesses du moment, tenues par `move_imposed`)
        b.motion = Motion::imposed([0.1, 0.0], 0.0);
        b.pivot = None;
        (b.vel, b.omega) = b.rates(0.0);
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
        // maille du centre entièrement couverte : son volume (plafonné pour la trace de fluide) compte en plus de celui du fluide,
        // qui n'a pas été chassé (corps ajouté à la main, sans `add_body`)
        assert!((s.body_frac[c] - 1.0).abs() < 1e-12);
        assert!((s.cell_volume(c, p) - v0 - (1.0 - FLUID_TRACE)).abs() < 1e-12);
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

    /// scène construite par un protocole (conditions initiales : départ + actions avant le premier « simuler »)
    fn scene(text: &str) -> crate::simulation::multiphase::MultiphaseSim {
        let p = crate::simulation::mp_script::parse_all(text).unwrap_or_else(|e| panic!("{e:?}")).remove(0);
        crate::simulation::mp_experiment::initial_state(&p, std::path::Path::new(".")).unwrap_or_else(|e| panic!("{e}")).0
    }

    fn run_to(s: &mut crate::simulation::multiphase::MultiphaseSim, t: f64) {
        while s.time < t && !s.diverged {
            s.step();
        }
        assert!(!s.diverged && s.state_is_sane(), "état invalide à t = {}", s.time);
    }

    #[test]
    fn placing_a_fixed_body_displaces_the_fluid_and_keeps_the_closure() {
        use crate::simulation::mp_view::{build_scenario, MpScenario};
        let mut s = build_scenario(MpScenario::DamBreak, 40, 30, 0.01, 9.81);
        let eau = s.reg.phases.iter().position(|p| p.name == "eau").unwrap();
        let (m0, mass0) = (s.total_mass(eau), s.mass.clone());
        let mut b = RigidBody::new(Shape::Disc { r: 0.03 }, 0, [0.05, 0.06], 293.15);
        b.motion = Motion::Fixed;
        s.add_body(b).unwrap();
        // fermeture tenue partout : fluide + corps = 1
        for c in 0..s.nx * s.ny {
            if !s.is_solid(c) {
                let v = s.cell_volume(c, s.p[c]);
                assert!((v - 1.0).abs() < 1e-9, "maille {c} : volume {v}");
            }
        }
        // l'eau chassée : celle du volume du disque (entièrement dans la colonne d'eau)
        let rho_w = s.reg.phases[eau].rho(s.p[s.idx(5, 5)], 293.15);
        let removed = m0 - s.total_mass(eau);
        assert!((removed / (rho_w * PI * 9e-4) - 1.0).abs() < 1e-3, "eau chassée {removed} kg/m");
        // une maille entièrement couverte garde une trace
        let c = s.idx(5, 5);
        assert!(s.body_frac[c] == 1.0 && s.mass[eau][c] > 0.0 && s.mass[eau][c] < 2e-3);
        // retirer le corps rend le fluide (aux arrondis près)
        s.clear_bodies();
        for k in 0..s.mass.len() {
            for c in 0..s.nx * s.ny {
                assert!((s.mass[k][c] - mass0[k][c]).abs() <= 1e-9 * mass0[k][c].abs().max(1e-12), "phase {k}, maille {c}");
            }
        }
    }

    const STILL: &str = "départ scénario \"cuve vide\" grille 40 x 40 maille 1 cm\n\
                         remplir rect (0, 0) (0.4, 0.3) avec \"eau\" à 20 °C\n";

    #[test]
    fn still_water_stays_still_around_fixed_bodies() {
        let mut with = scene(&format!("{STILL}corps disque (0.2, 0.15) rayon 4 cm \"acier\"\ncorps rect (0.06, 0.05) (0.12, 0.09) \"verre\" angle 30\n"));
        let mut without = scene(STILL);
        assert_eq!(with.bodies.len(), 2);
        run_to(&mut with, 0.5);
        run_to(&mut without, 0.5);
        let eau = with.reg.phases.iter().position(|p| p.name == "eau").unwrap();
        let inside: f64 = (0..with.nx * with.ny).filter(|&c| with.body_frac[c] >= 1.0 - FLUID_TRACE).map(|c| with.mass[eau][c]).fold(0.0, f64::max);
        let pmax_err = (0..with.nx * with.ny)
            .filter(|&c| !with.is_solid(c) && with.body_frac[c] < 1.0 - FLUID_TRACE)
            .map(|c| (with.p[c] - without.p[c]).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "eau au repos : vitesse max {:.3e} m/s avec corps, {:.3e} sans ; eau dans les mailles couvertes ≤ {inside:.3e} kg/m³ ; écart de pression ≤ {pmax_err:.3e} Pa ; {} pas",
            with.max_speed(),
            without.max_speed(),
            with.step_count
        );
        // mesuré : 1.8e-7 m/s, 3e-3 Pa (sans corps : 1.5e-11 m/s)
        assert!(with.max_speed() < 1e-5, "courants parasites {}", with.max_speed());
        assert!(pmax_err < 1.0, "la pression hydrostatique change autour des corps : {pmax_err} Pa");
        assert!(inside < 2e-3, "eau entrée dans un corps : {inside}");
    }

    #[test]
    fn a_dam_break_flows_around_a_fixed_body_without_entering_it() {
        // colonne d'eau à gauche (x < 0.15, y < 0.2, 29.9 kg/m) ; muret incliné posé sur le fond, puis un disque
        let mut s = scene(
            "départ scénario \"barrage\" grille 60 x 40 maille 1 cm\n\
             corps rect (0.3, 0) (0.33, 0.05) \"acier\" angle 20\n\
             corps disque (0.45, 0.04) rayon 2.5 cm \"verre\"\n",
        );
        let eau = s.reg.phases.iter().position(|p| p.name == "eau").unwrap();
        let m0 = s.total_mass(eau);
        run_to(&mut s, 0.6);
        let (nx, ny) = (s.nx, s.ny);
        let inside: f64 = (0..nx * ny).filter(|&c| s.body_frac[c] >= 1.0 - FLUID_TRACE).map(|c| s.mass[eau][c]).fold(0.0, f64::max);
        // eau passée au-delà du muret (x > 0.36 m) et du disque (x > 0.5 m)
        let beyond = |x0: usize| (0..ny).flat_map(|j| (x0..nx).map(move |i| (i, j))).map(|(i, j)| s.mass[eau][s.idx(i, j)]).sum::<f64>() * s.dx * s.dx;
        let (past_wall, past_disc) = (beyond(36), beyond(50));
        let dm = (s.total_mass(eau) / m0 - 1.0).abs();
        eprintln!(
            "barrage contre deux corps : eau dans les corps ≤ {inside:.2e} kg/m³, au-delà du muret {past_wall:.2} kg/m, du disque {past_disc:.2} kg/m (sur {m0:.1}), masse d'eau à {dm:.1e} près, vitesse max {:.2} m/s, {} pas, {} refaits",
            s.max_speed(),
            s.step_count,
            s.rollbacks
        );
        assert!(inside < 2e-3, "eau entrée dans un corps : {inside}");
        assert!(past_wall > 2.0 && past_disc > 0.5, "l'eau n'a pas franchi les obstacles ({past_wall} / {past_disc} kg/m)");
        assert!(dm < 1e-6, "masse d'eau non conservée ({dm})");
        assert!(s.max_speed() < 5.0);
    }

    #[test]
    fn a_channel_flow_goes_around_a_fixed_disc_and_not_through_it() {
        let mut s = scene(
            "départ scénario \"cuve vide\" grille 60 x 30 maille 1 cm\n\
             gravité (0, 0)\n\
             remplir partout avec \"eau\" à 20 °C\n\
             bord gauche entrée (0.2, 0) m/s à 20 °C \"eau\"\n\
             bord droite ouvert\n\
             bord haut mur\n\
             corps disque (0.25, 0.15) rayon 4 cm \"acier\"\n",
        );
        run_to(&mut s, 0.6);
        let (nx, ny) = (s.nx, s.ny);
        // faces fermées par le corps : vitesse nulle
        let mut closed = 0;
        for j in 0..ny {
            for i in 0..=nx {
                if s.u_open(i, j) == 0.0 {
                    closed += 1;
                    assert_eq!(s.u[s.uidx(i, j)], 0.0, "face u ({i}, {j}) fermée mais traversée");
                }
            }
        }
        assert!(closed > 20);
        let eau = s.reg.phases.iter().position(|p| p.name == "eau").unwrap();
        let inside: f64 = (0..nx * ny).filter(|&c| s.body_frac[c] >= 1.0 - FLUID_TRACE).map(|c| s.mass[eau][c]).fold(0.0, f64::max);
        // débit : entrée 0.2 m/s × 0.3 m ; sortie à droite ; par la section du disque (faces u en x = 0.25 m)
        let q = |i: usize| (0..ny).map(|j| s.u[s.uidx(i, j)]).sum::<f64>() * s.dx;
        let (q_in, q_mid, q_out) = (q(0), q(25), q(nx));
        // vitesse dans les passages au-dessus et au-dessous du disque (faces de la section du centre, hors corps)
        let gap: Vec<f64> = (0..ny).filter(|&j| s.u_open(25, j) == 1.0).map(|j| s.u[s.uidx(25, j)]).collect();
        let gap_mean = gap.iter().sum::<f64>() / gap.len() as f64;
        let closure = (0..nx * ny).filter(|&c| !s.is_solid(c)).map(|c| (s.cell_volume(c, s.p[c]) - 1.0).abs()).fold(0.0, f64::max);
        let qs: Vec<String> = (20..=31).map(|i| format!("{:.5}", q(i))).collect();
        eprintln!(
            "canal : débits entrée {q_in:.5} m²/s, section du disque {q_mid:.5}, sortie {q_out:.5} ; vitesse moyenne des passages {gap_mean:.3} m/s ; eau dans le corps ≤ {inside:.2e} kg/m³ ; {} faces fermées ; fermeture ≤ {closure:.1e} ; débits des sections 20 à 31 : {} ; {} pas",
            closed,
            qs.join(" "),
            s.step_count
        );
        assert!((q_in - 0.06).abs() < 1e-9, "entrée {q_in}");
        // mesuré : 0.0600 partout ; passages à 0.284 m/s (débit / section libre : 0.2 × 0.30 / 0.22 = 0.273)
        assert!((q_out / q_in - 1.0).abs() < 1e-3 && (q_mid / q_in - 1.0).abs() < 1e-3, "débits {q_in} {q_mid} {q_out}");
        assert!(gap_mean > 0.25 && gap_mean < 0.32, "l'eau n'accélère pas comme il faut autour du disque : {gap_mean}");
        assert!(inside < 2e-3, "eau entrée dans le corps : {inside}");
    }

    #[test]
    fn imposed_laws_are_integrated_exactly_whatever_the_steps() {
        // translation constante + oscillation verticale de période 0.5 s : après deux périodes, en pas irréguliers, le corps est pile à
        // (0.1, 0) — l'oscillation revient à son départ
        let mut b = RigidBody::new(Shape::Rect { w: 0.1, h: 0.02 }, 0, [0.0, 0.0], 300.0);
        b.motion = Motion::Imposed { v: [0.1, 0.0], omega: 0.0, osc_v: [0.0, 0.05], osc_omega: 0.0, period: 0.5 };
        let (mut t, mut k) = (0.0f64, 0usize);
        while t < 1.0 - 1e-15 {
            let dt = (0.013 + 0.017 * (k % 3) as f64).min(1.0 - t);
            b.move_imposed(t, t + dt);
            t += dt;
            k += 1;
        }
        assert!((b.pos[0] - 0.1).abs() < 1e-12 && b.pos[1].abs() < 1e-12, "{:?}", b.pos);
        // à un quart de période, la vitesse verticale est au maximum : déplacement ∫ = A P/(2π)
        let mut q = RigidBody::new(Shape::Disc { r: 0.01 }, 0, [0.0, 0.0], 300.0);
        q.motion = Motion::Imposed { v: [0.0, 0.0], omega: 0.0, osc_v: [0.0, 0.05], osc_omega: 0.0, period: 0.5 };
        q.move_imposed(0.0, 0.25);
        assert!((q.pos[1] - 0.05 * 0.5 / PI).abs() < 1e-12 && (q.vel[1] - 0.0).abs() < 1e-12);
        // rotation de 90 °/s sur un pivot au bout de la palette : 180° en 2 s, la palette passe de l'autre côté du pivot
        let mut c = RigidBody::new(Shape::Rect { w: 0.1, h: 0.02 }, 0, [0.05, 0.0], 300.0);
        c.pivot = Some([0.0, 0.0]);
        c.motion = Motion::imposed([0.0, 0.0], PI / 2.0);
        let mut t = 0.0;
        for k in 0..97 {
            let dt = if k < 96 { 0.02 + 0.001 * (k % 5) as f64 } else { 2.0 - t };
            if dt <= 0.0 {
                break;
            }
            c.move_imposed(t, t + dt);
            t += dt;
        }
        c.move_imposed(t, 2.0);
        assert!((c.angle - PI).abs() < 1e-12 && (c.pos[0] + 0.05).abs() < 1e-12 && c.pos[1].abs() < 1e-12, "{:?} {}", c.pos, c.angle);
        assert!((c.vel[0]).abs() < 1e-12 && (c.vel[1] + PI / 2.0 * 0.05).abs() < 1e-12, "vitesse du centre {:?}", c.vel);
    }

    #[test]
    fn a_piston_compresses_the_air_like_an_adiabat() {
        // tube de 40 × 10 cm fermé à droite, ouvert à gauche (1 atm) ; piston de 4 cm sur toute la hauteur, poussé à 5 cm/s pendant 2 s :
        // le gaz de droite passe de 34 à 24 cm. Compression lente (Mach 1.5e-4), parois adiabatiques : p V^γ et T V^(γ−1) constants.
        let mut s = scene(
            "départ scénario \"cuve vide\" grille 40 x 10 maille 1 cm\n\
             gravité (0, 0)\n\
             bord haut mur\n\
             bord gauche ouvert\n\
             corps rect (0.02, 0) (0.06, 0.1) \"acier\" vitesse (0.05, 0) m/s\n",
        );
        let air = s.ambient_phase;
        let region = |s: &crate::simulation::multiphase::MultiphaseSim, i0: usize| -> Vec<usize> {
            (0..s.ny).flat_map(|j| (i0..s.nx).map(move |i| j * 40 + i)).collect()
        };
        let mass = |s: &crate::simulation::multiphase::MultiphaseSim, cs: &[usize]| cs.iter().map(|&c| s.mass[air][c]).sum::<f64>() * s.dx * s.dx;
        let (c0, m0, p0, t0) = { let cs = region(&s, 6); (cs.clone(), mass(&s, &cs), s.p[cs[100]], s.temp[cs[100]]) };
        let _ = c0;
        run_to(&mut s, 2.0);
        assert!((s.bodies[0].pos[0] - 0.14).abs() < 1e-9, "piston en {}", s.bodies[0].pos[0]);
        let cs = region(&s, 16);
        let n = cs.len() as f64;
        let p = cs.iter().map(|&c| s.p[c]).sum::<f64>() / n;
        let t = cs.iter().map(|&c| s.temp[c]).sum::<f64>() / n;
        let pspread = cs.iter().map(|&c| (s.p[c] - p).abs()).fold(0.0, f64::max);
        let r = (0.34f64 / 0.24).ln();
        let (g_p, g_t) = ((p / p0).ln() / r, 1.0 + (t / t0).ln() / r);
        let m1 = mass(&s, &cs);
        let inside: f64 = (0..s.nx * s.ny).filter(|&c| s.body_frac[c] >= 1.0 - FLUID_TRACE).map(|c| s.mass[air][c]).fold(0.0, f64::max);
        eprintln!(
            "piston : p = {:.0} Pa (p0 {p0:.0}, ×{:.4}), T = {t:.2} K (×{:.4}) → γ par la pression {g_p:.4}, par la température {g_t:.4} ; écart de pression dans le gaz {pspread:.1} Pa ; masse du gaz comprimé {:.6e} → {:.6e} ; air dans le piston ≤ {inside:.2e} ; {} pas, {} refaits",
            p, p / p0, t / t0, m0, m1, s.step_count, s.rollbacks
        );
        assert!((g_p - 1.4).abs() < 0.02 && (g_t - 1.4).abs() < 0.02, "pas l'adiabatique : γ {g_p} / {g_t}");
        assert!(((m1 - m0) / m0).abs() < 1e-3, "fuite autour du piston : {m0} → {m1}");
        assert!(pspread < 0.01 * p, "pression non uniforme dans le gaz comprimé : {pspread} Pa");
    }

    #[test]
    fn a_disc_towed_through_still_water_pushes_it_aside_without_pressure_jumps() {
        let mut s = scene(
            "départ scénario \"cuve vide\" grille 50 x 30 maille 1 cm\n\
             remplir rect (0, 0) (0.5, 0.25) avec \"eau\" à 20 °C\n\
             corps disque (0.1, 0.12) rayon 3 cm \"acier\" vitesse (0.1, 0) m/s\n",
        );
        let eau = s.reg.phases.iter().position(|p| p.name == "eau").unwrap();
        let m0 = s.total_mass(eau);
        // sonde sous la trajectoire : sa pression ne doit pas sauter quand le disque traverse les mailles
        let probe = s.idx(25, 4);
        let (mut last, mut jump, mut pmin, mut pmax) = (f64::NAN, 0.0f64, f64::MAX, f64::MIN);
        while s.time < 2.0 && !s.diverged {
            s.step();
            let p = s.p[probe];
            if s.time > 0.1 {
                if last.is_finite() {
                    jump = jump.max((p - last).abs());
                }
                pmin = pmin.min(p);
                pmax = pmax.max(p);
            }
            last = p;
        }
        assert!(!s.diverged && s.state_is_sane());
        let inside: f64 = (0..s.nx * s.ny).filter(|&c| s.body_frac[c] >= 1.0 - FLUID_TRACE).map(|c| s.mass[eau][c]).fold(0.0, f64::max);
        let dm = (s.total_mass(eau) / m0 - 1.0).abs();
        eprintln!(
            "disque tiré dans l'eau : en x = {:.3} m ; sonde {pmin:.1} … {pmax:.1} Pa, plus grand saut d'un pas {jump:.2} Pa ; eau dans le disque ≤ {inside:.2e} ; masse d'eau à {dm:.1e} ; vitesse max {:.3} m/s ; {} pas, {} refaits",
            s.bodies[0].pos[0], s.max_speed(), s.step_count, s.rollbacks
        );
        assert!((s.bodies[0].pos[0] - (0.1 + 0.1 * s.time)).abs() < 1e-9);
        assert!(dm < 1e-6, "masse d'eau {dm}");
        assert!(inside < 5e-2, "eau restée dans le disque : {inside}");
        assert!(jump < 50.0, "saut de pression de {jump} Pa quand le disque traverse les mailles");
        assert!(s.max_speed() < 1.0);
    }

    #[test]
    fn a_paddle_spinning_on_its_pivot_stirs_the_water() {
        let mut s = scene(
            "départ scénario \"cuve vide\" grille 40 x 40 maille 1 cm\n\
             remplir rect (0, 0) (0.4, 0.35) avec \"eau\" à 20 °C\n\
             corps rect (0.12, 0.19) (0.28, 0.21) \"acier\" pivot (0.2, 0.2) rotation 90\n",
        );
        let eau = s.reg.phases.iter().position(|p| p.name == "eau").unwrap();
        let m0 = s.total_mass(eau);
        run_to(&mut s, 2.0);
        let b = &s.bodies[0];
        assert!((b.angle - 0.5 * PI * s.time).abs() < 1e-9 && (b.pos[0] - 0.2).abs() < 1e-12 && (b.pos[1] - 0.2).abs() < 1e-12, "{:?} {}", b.pos, b.angle);
        // eau mise en mouvement près des bouts de la palette (ω r = 0.11 m/s à 7 cm du pivot)
        let near: f64 = (0..s.ny)
            .flat_map(|j| (0..s.nx).map(move |i| (i, j)))
            .filter(|&(i, j)| {
                let (x, y) = ((i as f64 + 0.5) * 0.01 - 0.2, (j as f64 + 0.5) * 0.01 - 0.2);
                let r = x.hypot(y);
                r > 0.05 && r < 0.1
            })
            .map(|(i, j)| {
                let (a, bb) = crate::simulation::mp_view::cell_velocity(&s, i, j);
                a.hypot(bb)
            })
            .fold(0.0, f64::max);
        let inside: f64 = (0..s.nx * s.ny).filter(|&c| s.body_frac[c] >= 1.0 - FLUID_TRACE).map(|c| s.mass[eau][c]).fold(0.0, f64::max);
        let dm = (s.total_mass(eau) / m0 - 1.0).abs();
        eprintln!("palette sur pivot : eau près des bouts jusqu'à {near:.3} m/s ; eau dans la palette ≤ {inside:.2e} ; masse d'eau à {dm:.1e} ; {} pas, {} refaits", s.step_count, s.rollbacks);
        assert!(near > 0.02 && near < 0.5, "l'eau n'est pas brassée comme il faut : {near} m/s");
        assert!(dm < 1e-6, "masse d'eau {dm}");
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
