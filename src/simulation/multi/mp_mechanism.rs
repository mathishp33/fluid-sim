//! Mécanismes à un degré de liberté : un arbre d'angle θ dont les pièces — corps rigides menés (`Motion::Driven`) — ont une pose qui
//! découle exactement de θ. Le fluide les voit comme des corps à mouvement imposé : elles le poussent, et la force qu'il exerce sur
//! chacune est relevée à chaque projection. Pièces :
//! * bielle-manivelle (vilebrequin) : manivelle autour d'un axe fixe, bielle, piston qui glisse le long de l'axe de son cylindre
//!   (θ = 0 au point mort haut) ;
//! * roues : disque, avec ses dents ou ses pales, solidaires d'un plateau qui tourne d'un angle `phase + rapport · θ` autour de son
//!   centre — sur l'arbre (rapport 1, ou une courroie), ou qui en engrène une autre (rapport des rayons, de signe opposé, dents
//!   calées pour s'emboîter) ;
//! * pièces quelconques, dont la position et l'angle sont des formules de la rotation du mécanisme (crémaillère, came et poussoir,
//!   levier, coulisse).
//!
//! Le mécanisme est MENÉ à vitesse imposée (moteur électrique, compresseur), ou LIBRE : θ suit alors l'équation de Lagrange,
//! M(θ) θ̈ + ½ M′(θ) θ̇² = Q, où M(θ) = I_volant + Σ (m |dG/dθ|² + I (dφ/dθ)²) est l'inertie ramenée à l'arbre et
//! Q = Σ (F · dG/dθ + Γ dφ/dθ) + Γ_appliqué le couple ramené des forces du fluide (dernière projection, gardées sur le pas), des poids
//! et d'un couple appliqué en formule (frein, charge, démarreur). Intégration au point milieu, en sous-pas d'au plus 0.005 rad.

use serde::{Deserialize, Serialize};
use std::f64::consts::{PI, TAU};

use super::mp_bodies::{Motion, RigidBody, Shape};
use super::mp_fields::Formula;
use super::multiphase::MultiphaseSim;

/// variables de la formule du couple appliqué : temps (s), angle (°, de 0 à 360), tours, rotation (°/s)
pub const CRANK_VARS: [&str; 4] = ["t", "angle", "tours", "rotation"];

/// variables des formules d'une pièce : rotation du mécanisme depuis la pose de la pièce (theta en rad, angle en °, tours), et la
/// pose écrite de la pièce (x0, y0 en m, a0 en °)
pub const PIECE_VARS: [&str; 6] = ["theta", "angle", "tours", "x0", "y0", "a0"];

/// pièces d'un vilebrequin, dans l'ordre de ses `pieces` (noms « X.manivelle »… d'un vilebrequin nommé « X »)
pub const CRANK_PARTS: [&str; 3] = ["manivelle", "bielle", "piston"];

/// Géométrie d'une bielle-manivelle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CrankGeom {
    /// axe du vilebrequin (m), rayon de manivelle (m : la course vaut 2r), longueur de la bielle entre ses axes (m), direction de
    /// l'axe du cylindre depuis l'axe du vilebrequin (rad)
    pub pivot: [f64; 2],
    pub r: f64,
    pub l: f64,
    pub axis: f64,
    /// hauteur du piston le long de l'axe (l'axe de bielle est au milieu de sa face basse) ; longueur de la bielle dessinée (un peu
    /// moins que l : inclinée, elle ne mord pas sur le piston)
    pub piston_h: f64,
    pub rod_len: f64,
}

impl CrankGeom {
    /// axes de la manivelle (A) et du piston (B) pour l'angle θ
    pub fn pins(&self, th: f64) -> ([f64; 2], [f64; 2]) {
        let (e, n) = ([self.axis.cos(), self.axis.sin()], [-self.axis.sin(), self.axis.cos()]);
        let (along, across) = (self.r * th.cos(), self.r * th.sin());
        let a = [self.pivot[0] + along * e[0] + across * n[0], self.pivot[1] + along * e[1] + across * n[1]];
        let s = along + (self.l * self.l - across * across).max(0.0).sqrt();
        (a, [self.pivot[0] + s * e[0], self.pivot[1] + s * e[1]])
    }

    /// poses (centre, angle) de la manivelle, de la bielle et du piston pour l'angle θ
    pub fn poses(&self, th: f64) -> [([f64; 2], f64); 3] {
        let (a, b) = self.pins(th);
        let (o, e) = (self.pivot, [self.axis.cos(), self.axis.sin()]);
        let crank = ([(o[0] + a[0]) / 2.0, (o[1] + a[1]) / 2.0], self.axis + th);
        // bielle : de A vers B, raccourcie côté piston ; son angle par rapport à l'axe reste entre −90° et 90° (pas de saut)
        let d = [b[0] - a[0], b[1] - a[1]];
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-300);
        let u = [d[0] / len, d[1] / len];
        let rel = (e[0] * d[1] - e[1] * d[0]).atan2(e[0] * d[0] + e[1] * d[1]);
        let rod = ([a[0] + u[0] * self.rod_len / 2.0, a[1] + u[1] * self.rod_len / 2.0], self.axis + rel);
        let piston = ([b[0] + e[0] * self.piston_h / 2.0, b[1] + e[1] * self.piston_h / 2.0], self.axis - std::f64::consts::FRAC_PI_2);
        [crank, rod, piston]
    }
}

/// Comment une pièce suit l'angle θ du mécanisme.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PieceKind {
    /// manivelle (0), bielle (1) ou piston (2) de la bielle-manivelle du mécanisme
    Crank(usize),
    /// solidaire d'un plateau qui tourne autour de `center` d'un angle `phase + ratio θ` : centre du corps en `offset` dans le
    /// repère du plateau, angle propre `turn` (une roue : 0 et 0 ; une dent, une pale : leur place sur la roue)
    Frame { center: [f64; 2], ratio: f64, phase: f64, offset: [f64; 2], turn: f64 },
    /// position (m) et angle (°) en formules (`PIECE_VARS`) de la rotation depuis la pose (θ0) ; None : la valeur écrite
    Formula { x: Option<String>, y: Option<String>, angle: Option<String>, x0: f64, y0: f64, a0: f64, theta0: f64 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Piece {
    /// indice dans `bodies`
    pub body: usize,
    pub kind: PieceKind,
}

/// Roue d'un mécanisme (pour engrener les suivantes) : son angle est `phase + ratio θ`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Wheel {
    pub name: String,
    pub center: [f64; 2],
    /// rayon (primitif, pour une roue dentée)
    pub r: f64,
    pub ratio: f64,
    pub phase: f64,
    /// nombre de dents (0 : roue lisse)
    pub teeth: usize,
}

/// Dents ou pales d'une roue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rim {
    Plain,
    /// n dents trapézoïdales de module 2r/n
    Teeth(usize),
    /// n pales droites de longueur et d'épaisseur données (m), depuis le bord du disque
    Blades(usize, f64, f64),
}

/// Mécanisme à un degré de liberté.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mechanism {
    /// nom donné par le protocole (`mécanisme "X"`, `vilebrequin … nom "X"`), vide sinon
    pub name: String,
    /// angle de l'arbre (rad, non ramené à un tour ; pour un vilebrequin, depuis le point mort haut) et sa vitesse (rad/s, sens
    /// trigonométrique)
    pub theta: f64,
    pub omega: f64,
    /// vitesse imposée (rad/s) ; None : libre
    pub driven: Option<f64>,
    /// inertie ajoutée à l'arbre (volant ; kg·m par mètre de profondeur)
    pub inertia: f64,
    /// couple appliqué à l'arbre (N, par mètre de profondeur ; formule en `CRANK_VARS`)
    pub load: Option<String>,
    /// au dernier pas : couple ramené à l'arbre des forces du fluide et des poids (N), couple appliqué (N)
    pub torque: f64,
    pub applied: f64,
    /// bielle-manivelle (vilebrequin)
    pub crank: Option<CrankGeom>,
    pub pieces: Vec<Piece>,
    pub wheels: Vec<Wheel>,
}

/// formules (x, y, angle) d'une pièce, lues une fois par pas
type Compiled = Vec<[Option<Formula>; 3]>;

impl Mechanism {
    pub fn new(name: &str, theta: f64, omega: f64, driven: Option<f64>, inertia: f64, load: Option<String>) -> Mechanism {
        Mechanism {
            name: name.to_string(),
            theta,
            omega,
            driven,
            inertia,
            load,
            torque: 0.0,
            applied: 0.0,
            crank: None,
            pieces: vec![],
            wheels: vec![],
        }
    }

    /// corps des pièces, dans leur ordre
    pub fn bodies(&self) -> Vec<usize> {
        self.pieces.iter().map(|p| p.body).collect()
    }

    fn compile(&self) -> Compiled {
        let f = |s: &Option<String>| s.as_ref().and_then(|s| Formula::parse_vars(&super::mp_experiment::norm(s), &PIECE_VARS).ok());
        self.pieces
            .iter()
            .map(|p| match &p.kind {
                PieceKind::Formula { x, y, angle, .. } => [f(x), f(y), f(angle)],
                _ => [None, None, None],
            })
            .collect()
    }

    /// pose (centre, angle) de la pièce k pour l'angle θ
    fn pose(&self, k: usize, th: f64, cf: &Compiled) -> ([f64; 2], f64) {
        match &self.pieces[k].kind {
            PieceKind::Crank(i) => self.crank.as_ref().map_or(([f64::NAN; 2], f64::NAN), |c| c.poses(th)[*i]),
            PieceKind::Frame { center, ratio, phase, offset, turn } => {
                let a = phase + ratio * th;
                let (s, c) = a.sin_cos();
                ([center[0] + c * offset[0] - s * offset[1], center[1] + s * offset[0] + c * offset[1]], a + turn)
            }
            PieceKind::Formula { x, y, angle, x0, y0, a0, theta0 } => {
                let rel = th - theta0;
                let v = [rel, rel.to_degrees(), rel / TAU, *x0, *y0, a0.to_degrees()];
                let get = |i: usize, has: bool, dflt: f64| -> f64 {
                    if !has {
                        return dflt;
                    }
                    cf.get(k).and_then(|c| c[i].as_ref()).map_or(f64::NAN, |e| e.eval_vars(&v))
                };
                ([get(0, x.is_some(), *x0), get(1, y.is_some(), *y0)], if angle.is_some() { get(2, true, 0.0).to_radians() } else { *a0 })
            }
        }
    }

    pub fn poses(&self, th: f64, cf: &Compiled) -> Vec<([f64; 2], f64)> {
        (0..self.pieces.len()).map(|k| self.pose(k, th, cf)).collect()
    }

    /// dérivées des poses par rapport à θ : (dx/dθ, dy/dθ, dφ/dθ) de chaque pièce — exactes pour une pièce de roue, en différences
    /// centrées sinon
    pub fn jacobian(&self, th: f64, cf: &Compiled) -> Vec<[f64; 3]> {
        let h = 1e-6;
        (0..self.pieces.len())
            .map(|k| match &self.pieces[k].kind {
                PieceKind::Frame { ratio, phase, offset, .. } => {
                    let (s, c) = (phase + ratio * th).sin_cos();
                    [ratio * (-s * offset[0] - c * offset[1]), ratio * (c * offset[0] - s * offset[1]), *ratio]
                }
                _ => {
                    let (p, m) = (self.pose(k, th + h, cf), self.pose(k, th - h, cf));
                    [(p.0[0] - m.0[0]) / (2.0 * h), (p.0[1] - m.0[1]) / (2.0 * h), (p.1 - m.1) / (2.0 * h)]
                }
            })
            .collect()
    }

    /// inertie ramenée à l'arbre (kg·m par mètre) : volant, puis Σ m |dG/dθ|² + I (dφ/dθ)² des pièces
    pub fn inertia_at(&self, th: f64, mass: &[f64], inert: &[f64], cf: &Compiled) -> f64 {
        let j = self.jacobian(th, cf);
        self.inertia + (0..j.len()).map(|k| mass[k] * (j[k][0] * j[k][0] + j[k][1] * j[k][1]) + inert[k] * j[k][2] * j[k][2]).sum::<f64>()
    }

    /// couple ramené à l'arbre (N) des efforts `f` (Fx, Fy, couple autour du centre de masse) sur les pièces
    pub fn torque_at(&self, th: f64, f: &[[f64; 3]], cf: &Compiled) -> f64 {
        let j = self.jacobian(th, cf);
        (0..j.len()).map(|k| f[k][0] * j[k][0] + f[k][1] * j[k][1] + f[k][2] * j[k][2]).sum()
    }

    /// pose les pièces (position, angle, vitesses) pour l'état (θ, ω)
    fn place(&self, bodies: &mut [RigidBody], cf: &Compiled) {
        let (p, j) = (self.poses(self.theta, cf), self.jacobian(self.theta, cf));
        for (k, piece) in self.pieces.iter().enumerate() {
            let b = &mut bodies[piece.body];
            b.pos = p[k].0;
            b.angle = p[k].1;
            b.vel = [j[k][0] * self.omega, j[k][1] * self.omega];
            b.omega = j[k][2] * self.omega;
        }
    }

    /// poses (centre, angle) de toutes les pièces pour l'angle θ
    pub fn body_poses(&self, th: f64) -> Vec<([f64; 2], f64)> {
        self.poses(th, &self.compile())
    }

    /// angle (°) de 0 à 360 (pour un vilebrequin, depuis le point mort haut)
    pub fn angle_deg(&self) -> f64 {
        self.theta.to_degrees().rem_euclid(360.0)
    }

    /// couple appliqué (N) à l'instant t pour l'état (θ, ω)
    fn applied_at(&self, t: f64, th: f64, w: f64) -> f64 {
        let Some(f) = &self.load else { return 0.0 };
        let v = [t, th.to_degrees().rem_euclid(360.0), th / TAU, w.to_degrees()];
        Formula::parse_vars(f, &CRANK_VARS).map_or(f64::NAN, |e| e.eval_vars(&v))
    }

    /// angle actuel de la roue w
    pub fn wheel_angle(&self, w: usize) -> f64 {
        self.wheels[w].phase + self.wheels[w].ratio * self.theta
    }
}

/// Contour d'une dent en développante de cercle (angle de pression 20°) d'une roue de rayon primitif `r` et de module `m`, son axe
/// selon +x : flancs en développante du cercle de base jusqu'à la tête `ra`, radiaux sous le cercle de base jusqu'au pied `rf` ;
/// épaisseur au cercle primitif π m / 2 − `backlash` (un jeu de chaque côté de la dent d'en face). Polygone convexe.
fn involute_tooth(r: f64, m: f64, rf: f64, ra: f64, backlash: f64) -> Vec<[f64; 2]> {
    let alpha = 20f64.to_radians();
    let rb = r * alpha.cos();
    let inv = |phi: f64| phi.tan() - phi;
    let s = PI * m / 2.0 - backlash;
    // demi-épaisseur angulaire de la dent au rayon ρ
    let half = |rho: f64| s / (2.0 * r) + inv(alpha) - inv((rb / rho.max(rb)).min(1.0).acos());
    // tête : pas plus loin que là où la dent finirait en pointe
    let mut ra = ra;
    while half(ra) < 0.1 * m / ra && ra > r {
        ra -= 0.02 * m;
    }
    let lo = rf.max(rb);
    let mut radii = vec![rf];
    if rb > rf + 1e-9 * r {
        radii.push(rb);
    }
    radii.extend((1..=6).map(|k| lo + (ra - lo) * k as f64 / 6.0));
    let at = |rho: f64, sign: f64| {
        let a = sign * half(rho);
        [rho * a.cos(), rho * a.sin()]
    };
    let mut pts: Vec<[f64; 2]> = radii.iter().map(|&rho| at(rho, -1.0)).collect();
    pts.extend(radii.iter().rev().map(|&rho| at(rho, 1.0)));
    pts
}

/// même nom, sans casse ni accents (un nom vide n'en égale aucun)
fn same(a: &str, b: &str) -> bool {
    let a = super::mp_experiment::norm(a);
    !a.is_empty() && a == super::mp_experiment::norm(b)
}

impl MultiphaseSim {
    /// mécanisme nommé `name`
    pub fn mechanism_named(&self, name: &str) -> Option<usize> {
        self.mechanisms.iter().position(|m| same(&m.name, name))
    }

    /// roue nommée `name` : (mécanisme, roue)
    pub fn wheel_named(&self, name: &str) -> Option<(usize, usize)> {
        self.mechanisms.iter().enumerate().find_map(|(i, m)| m.wheels.iter().position(|w| same(&w.name, name)).map(|k| (i, k)))
    }

    /// Ajoute un mécanisme (sans pièce ; `roue` et `pièce` lui en donnent). Retourne son indice.
    pub fn add_mechanism(&mut self, m: Mechanism) -> Result<usize, String> {
        if self.mechanism_named(&m.name).is_some() {
            return Err(format!("un mécanisme s'appelle déjà « {} » (un composant posé deux fois : donnez-lui un autre nom)", m.name));
        }
        self.mechanisms.push(m);
        Ok(self.mechanisms.len() - 1)
    }

    /// corps menés `parts` (forme, pièce, nom) ajoutés au mécanisme `mi`, posés pour son angle actuel
    fn add_pieces(&mut self, mi: usize, parts: Vec<(Shape, PieceKind, String)>, material: usize, temp: f64) -> Result<(), String> {
        for (_, _, n) in &parts {
            if !n.is_empty() && self.body_named(n).is_some() {
                return Err(format!("un corps s'appelle déjà « {n} » (un composant posé deux fois : donnez-lui un autre nom)"));
            }
        }
        let mut m = self.mechanisms[mi].clone();
        let first = m.pieces.len();
        for (_, kind, _) in &parts {
            m.pieces.push(Piece { body: usize::MAX, kind: kind.clone() });
        }
        let cf = m.compile();
        for (k, (shape, _, name)) in parts.into_iter().enumerate() {
            let (pos, angle) = m.pose(first + k, m.theta, &cf);
            if !(pos[0].is_finite() && pos[1].is_finite() && angle.is_finite()) {
                return Err("pose de la pièce incalculable (formule qui ne donne pas un nombre)".into());
            }
            let mut b = RigidBody::new(shape, material, pos, temp);
            b.angle = angle;
            b.motion = Motion::Driven;
            b.name = name;
            m.pieces[first + k].body = self.add_body(b)?;
        }
        m.place(&mut self.bodies, &cf);
        for p in &m.pieces[first..] {
            self.bodies[p.body].last_vel = self.bodies[p.body].vel;
        }
        self.mechanisms[mi] = m;
        self.update_body_face_velocities();
        Ok(())
    }

    /// Pose un vilebrequin, sa bielle et son piston (corps menés, du matériau `material`, à la température `temp`) : manivelle de
    /// largeur `w` de l'axe à l'axe de bielle, bielle de largeur `w`, piston de largeur `bore`. `m` porte la géométrie (`crank`,
    /// la longueur de bielle dessinée y est calculée) et l'état de départ. Retourne l'indice du mécanisme.
    pub fn add_crank(&mut self, mut m: Mechanism, bore: f64, w: f64, material: usize, temp: f64) -> Result<usize, String> {
        let Some(mut ck) = m.crank.clone() else { return Err("vilebrequin sans géométrie".into()) };
        if !(ck.r > 0.0 && ck.l.is_finite() && ck.l > 1.05 * ck.r) {
            return Err("la bielle doit être plus longue que le rayon de manivelle (au moins 1.05 fois)".into());
        }
        if !(bore > 0.0 && ck.piston_h > 0.0 && w > 0.0) {
            return Err("alésage, hauteur du piston et largeur des pièces doivent être positifs".into());
        }
        if self.mechanism_named(&m.name).is_some() {
            return Err(format!("un vilebrequin s'appelle déjà « {} » (un composant posé deux fois : donnez-lui un autre nom)", m.name));
        }
        // (bielle inclinée d'au plus β = asin(r/l) : raccourcie pour que son coin ne morde pas sur le piston)
        let beta = (ck.r / ck.l).asin();
        ck.rod_len = (ck.l - 0.5 * w * beta.tan() - 0.05 * w).max(0.5 * ck.l);
        let shapes = [Shape::Rect { w: ck.r + w, h: w }, Shape::Rect { w: ck.rod_len, h: w }, Shape::Rect { w: bore, h: ck.piston_h }];
        m.crank = Some(ck);
        m.pieces.clear();
        let name = m.name.clone();
        let mi = self.add_mechanism(m)?;
        let parts = (0..3)
            .map(|k| (shapes[k].clone(), PieceKind::Crank(k), if name.trim().is_empty() { String::new() } else { format!("{name}.{}", CRANK_PARTS[k]) }))
            .collect();
        if let Err(e) = self.add_pieces(mi, parts, material, temp) {
            self.mechanisms.pop();
            return Err(e);
        }
        Ok(mi)
    }

    /// Roue du mécanisme `mi` : disque de rayon `r` (rayon primitif d'une roue dentée) autour de `center`, avec ses dents ou ses
    /// pales, son angle étant `angle_now` maintenant et tournant de `ratio` fois l'angle du mécanisme. Ses corps s'appellent `name`
    /// (le disque), « name.dent1 »…, « name.pale1 »… Retourne l'indice de la roue dans le mécanisme.
    #[allow(clippy::too_many_arguments)]
    pub fn add_wheel(&mut self, mi: usize, name: &str, center: [f64; 2], r: f64, ratio: f64, angle_now: f64, rim: Rim, material: usize, temp: f64) -> Result<usize, String> {
        if !(r > 0.0 && ratio.is_finite() && ratio != 0.0) {
            return Err("roue de rayon ou de rapport invalide".into());
        }
        let phase = angle_now - ratio * self.mechanisms[mi].theta;
        let frame = |offset: [f64; 2], turn: f64| PieceKind::Frame { center, ratio, phase, offset, turn };
        let sub = |what: &str, j: usize| if name.trim().is_empty() { String::new() } else { format!("{name}.{what}{j}") };
        let rot = |p: [f64; 2], a: f64| [p[0] * a.cos() - p[1] * a.sin(), p[0] * a.sin() + p[1] * a.cos()];
        let mut parts: Vec<(Shape, PieceKind, String)> = vec![];
        let teeth = match rim {
            Rim::Teeth(n) => n,
            _ => 0,
        };
        match rim {
            Rim::Plain => parts.push((Shape::Disc { r }, frame([0.0, 0.0], 0.0), name.to_string())),
            Rim::Teeth(n) => {
                if n < 6 {
                    return Err("une roue dentée a au moins 6 dents".into());
                }
                // module m = 2r/n ; dents en développante de cercle, qui engrènent sans se chevaucher ; jeux d'au moins une maille (au
                // pied, une maille et demie) : le fluide pris entre les dents s'échappe par là
                let m = 2.0 * r / n as f64;
                if m < 3.0 * self.dx {
                    return Err(format!(
                        "dents trop fines pour la maille : module {:.4} m (le diamètre divisé par le nombre de dents), il en faut au moins trois \
                         mailles ({:.4} m) — moins de dents, une roue plus grande ou une maille plus fine",
                        m,
                        3.0 * self.dx
                    ));
                }
                let (clearance, backlash) = ((0.25 * m).max(1.5 * self.dx), (0.04 * PI * m).max(self.dx));
                let root = r - m - clearance;
                parts.push((Shape::Disc { r: root }, frame([0.0, 0.0], 0.0), name.to_string()));
                let (shape, c) = Shape::polygon(&involute_tooth(r, m, root, r + m, backlash))?;
                for j in 0..n {
                    let a = TAU * j as f64 / n as f64;
                    parts.push((shape.clone(), frame(rot(c, a), a), sub("dent", j + 1)));
                }
            }
            Rim::Blades(n, len, e) => {
                if n == 0 || !(len > 0.0 && e > 0.0) {
                    return Err("pales : un nombre, une longueur et une épaisseur positifs".into());
                }
                parts.push((Shape::Disc { r }, frame([0.0, 0.0], 0.0), name.to_string()));
                // du bord du disque (une demi-épaisseur dedans) à r + longueur
                let (r0, r1) = (r - 0.5 * e, r + len);
                for j in 0..n {
                    let a = TAU * j as f64 / n as f64;
                    parts.push((Shape::Rect { w: r1 - r0, h: e }, frame(rot([(r0 + r1) / 2.0, 0.0], a), a), sub("pale", j + 1)));
                }
            }
        }
        if !name.trim().is_empty() && self.wheel_named(name).is_some() {
            return Err(format!("une roue s'appelle déjà « {name} »"));
        }
        self.add_pieces(mi, parts, material, temp)?;
        let m = &mut self.mechanisms[mi];
        m.wheels.push(Wheel { name: name.to_string(), center, r, ratio, phase, teeth });
        Ok(m.wheels.len() - 1)
    }

    /// Rapport et angle actuel d'une roue de centre `center`, de rayon `r` et de `teeth` dents (0 : lisse) qui engrène la roue
    /// `wi` du mécanisme `mi` : elle tourne en sens inverse, au rapport de leurs rayons ; une roue dentée est calée pour qu'une dent
    /// de l'une tombe dans un creux de l'autre. Erreur si elles ne se touchent pas ou si leurs dents ne vont pas ensemble.
    pub fn mesh(&self, mi: usize, wi: usize, center: [f64; 2], r: f64, teeth: usize) -> Result<(f64, f64), String> {
        let m = &self.mechanisms[mi];
        let w = &m.wheels[wi];
        let d = (center[0] - w.center[0]).hypot(center[1] - w.center[1]);
        let want = r + w.r;
        if (d - want).abs() > 0.02 * want {
            return Err(format!(
                "ces roues ne se touchent pas : leurs centres sont à {:.4} m, il faut la somme des rayons ({:.4} m)",
                d, want
            ));
        }
        if (teeth > 0) != (w.teeth > 0) {
            return Err("une roue dentée engrène une roue dentée, une roue lisse une roue lisse (roues de friction)".into());
        }
        let ratio = -w.ratio * w.r / r;
        if teeth == 0 {
            return Ok((ratio, 0.0));
        }
        let (m1, m2) = (2.0 * r / teeth as f64, 2.0 * w.r / w.teeth as f64);
        if (m1 - m2).abs() > 0.02 * m2 {
            return Err(format!(
                "dents de tailles différentes : module {:.4} m et {:.4} m (le module est le diamètre divisé par le nombre de dents ; deux roues qui engrènent ont le même)",
                m1, m2
            ));
        }
        // place des dents au point de contact, en dents : s = 0 pour une dent ; l'autre roue y met un creux (s + s' = 1/2), ce que le
        // roulement garde
        let u = [(center[0] - w.center[0]) / d, (center[1] - w.center[1]) / d];
        let s_w = (u[1].atan2(u[0]) - m.wheel_angle(wi)) * w.teeth as f64 / TAU;
        Ok((ratio, (-u[1]).atan2(-u[0]) - (0.5 - s_w) * TAU / teeth as f64))
    }

    /// Pièce quelconque du mécanisme `mi` : le corps `b` (forme, matériau, pose écrite) suit les formules `x`, `y` (m) et `angle`
    /// (°) de la rotation du mécanisme depuis maintenant (`PIECE_VARS`).
    pub fn add_formula_piece(&mut self, mi: usize, b: RigidBody, x: Option<String>, y: Option<String>, angle: Option<String>) -> Result<(), String> {
        let kind = PieceKind::Formula { x, y, angle, x0: b.pos[0], y0: b.pos[1], a0: b.angle, theta0: self.mechanisms[mi].theta };
        let (material, temp, name) = (b.material, b.temp, b.name.clone());
        self.add_pieces(mi, vec![(b.shape, kind, name)], material, temp)
    }

    /// Avance les mécanismes de t0 à t1 et pose leurs pièces : mené, à sa vitesse ; libre, par l'équation de Lagrange (point milieu,
    /// sous-pas d'au plus 0.005 rad) avec les forces du fluide de la dernière projection gardées sur le pas, les poids et le couple
    /// appliqué.
    pub(crate) fn advance_mechanisms(&mut self, t0: f64, t1: f64) {
        let dt = t1 - t0;
        for c in 0..self.mechanisms.len() {
            let mc = self.mechanisms[c].clone();
            let cf = mc.compile();
            let n = mc.pieces.len();
            let (mut mass, mut inert, mut f) = (vec![0.0; n], vec![0.0; n], vec![[0.0; 3]; n]);
            for (k, p) in mc.pieces.iter().enumerate() {
                let b = &self.bodies[p.body];
                let rho = b.density(self.reg.solids[b.material].rho);
                mass[k] = rho * b.shape.area();
                inert[k] = rho * b.shape.polar_moment();
                let fl = self.body_force.get(p.body).copied().unwrap_or([0.0; 3]);
                f[k] = [fl[0] + mass[k] * self.gravity[0], fl[1] + mass[k] * self.gravity[1], fl[2]];
            }
            let torque = mc.torque_at(mc.theta, &f, &cf);
            let (mut th, mut w) = (mc.theta, mc.omega);
            let mut applied = 0.0;
            match mc.driven {
                Some(wd) => {
                    th += wd * dt;
                    w = wd;
                    applied = mc.applied_at(t0, mc.theta, wd);
                }
                None if dt > 0.0 => {
                    let acc = |t: f64, th: f64, w: f64| -> f64 {
                        let h = 1e-5;
                        let m = mc.inertia_at(th, &mass, &inert, &cf);
                        if !(m > 0.0) {
                            // ni pièce ni volant : rien à mettre en mouvement
                            return 0.0;
                        }
                        let dm = (mc.inertia_at(th + h, &mass, &inert, &cf) - mc.inertia_at(th - h, &mass, &inert, &cf)) / (2.0 * h);
                        (mc.torque_at(th, &f, &cf) + mc.applied_at(t, th, w) - 0.5 * dm * w * w) / m.max(1e-300)
                    };
                    let n = ((w.abs().max(1.0) * dt / 0.005).ceil() as usize).clamp(1, 10_000);
                    let h = dt / n as f64;
                    for s in 0..n {
                        let t = t0 + s as f64 * h;
                        let a1 = acc(t, th, w);
                        let (thm, wm) = (th + 0.5 * h * w, w + 0.5 * h * a1);
                        let a2 = acc(t + 0.5 * h, thm, wm);
                        if s == 0 {
                            applied = mc.applied_at(t + 0.5 * h, thm, wm);
                        }
                        th += h * wm;
                        w += h * a2;
                    }
                }
                None => {}
            }
            let m = &mut self.mechanisms[c];
            m.theta = th;
            m.omega = w;
            m.torque = torque;
            m.applied = applied;
            let m = m.clone();
            m.place(&mut self.bodies, &cf);
        }
    }
}
