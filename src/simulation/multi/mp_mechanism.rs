//! Mécanisme bielle-manivelle : un vilebrequin (manivelle tournant autour d'un axe fixe), une bielle et un piston qui glisse le long
//! de l'axe de son cylindre. Les trois pièces sont des corps rigides menés (`Motion::Driven`) : leur pose découle exactement d'un seul
//! angle, celui du vilebrequin θ (0 au point mort haut, sens trigonométrique). Le fluide les voit comme des corps à mouvement imposé :
//! elles le poussent, et la force qu'il exerce sur chacune est relevée à chaque projection.
//!
//! Le vilebrequin est MENÉ à vitesse imposée (compresseur, pompe), ou LIBRE (moteur) : θ suit alors l'équation de Lagrange du
//! mécanisme à un degré de liberté, M(θ) θ̈ + ½ M′(θ) θ̇² = Q, où M(θ) = I_volant + Σ (m |dG/dθ|² + I (dφ/dθ)²) est l'inertie
//! ramenée au vilebrequin et Q = Σ (F · dG/dθ + Γ dφ/dθ) + Γ_appliqué le couple ramené des forces du fluide (dernière projection,
//! gardées sur le pas), des poids et d'un couple appliqué en formule (frein, charge, démarreur). Intégration au point milieu, en
//! sous-pas d'au plus 0.005 rad.

use serde::{Deserialize, Serialize};

use super::mp_bodies::{Motion, RigidBody, Shape};
use super::multiphase::MultiphaseSim;

/// variables de la formule du couple appliqué : temps (s), angle (°, de 0 à 360 depuis le point mort haut), tours, rotation (°/s)
pub const CRANK_VARS: [&str; 4] = ["t", "angle", "tours", "rotation"];

/// Vilebrequin, bielle et piston.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Crank {
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
    /// corps : manivelle, bielle, piston (indices dans `bodies`)
    pub bodies: [usize; 3],
    /// angle (rad, depuis le point mort haut, non ramené à un tour) et vitesse de rotation (rad/s, sens trigonométrique)
    pub theta: f64,
    pub omega: f64,
    /// vitesse imposée (rad/s) ; None : libre
    pub driven: Option<f64>,
    /// inertie ajoutée au vilebrequin (volant ; kg·m par mètre de profondeur)
    pub inertia: f64,
    /// couple appliqué au vilebrequin (N, par mètre de profondeur ; formule en `CRANK_VARS`)
    pub load: Option<String>,
    /// au dernier pas : couple ramené au vilebrequin des forces du fluide et des poids (N), couple appliqué (N)
    pub torque: f64,
    pub applied: f64,
    /// nom donné par le protocole (`vilebrequin … nom "X"`), vide sinon ; ses pièces s'appellent « X.manivelle », « X.bielle »,
    /// « X.piston »
    pub name: String,
}

/// pièces d'un vilebrequin, dans l'ordre de `Crank::bodies` (noms « X.manivelle »… d'un vilebrequin nommé « X »)
pub const CRANK_PARTS: [&str; 3] = ["manivelle", "bielle", "piston"];

impl Crank {
    /// axes de la manivelle (A) et du piston (B) pour l'angle θ
    pub fn pins(&self, th: f64) -> ([f64; 2], [f64; 2]) {
        let (e, n) = ([self.axis.cos(), self.axis.sin()], [-self.axis.sin(), self.axis.cos()]);
        let (along, across) = (self.r * th.cos(), self.r * th.sin());
        let a = [self.pivot[0] + along * e[0] + across * n[0], self.pivot[1] + along * e[1] + across * n[1]];
        let s = along + (self.l * self.l - across * across).max(0.0).sqrt();
        (a, [self.pivot[0] + s * e[0], self.pivot[1] + s * e[1]])
    }

    /// distance de l'axe du piston à l'axe du vilebrequin, le long de l'axe du cylindre : r cos θ + √(l² − r² sin² θ)
    pub fn stroke_pos(&self, th: f64) -> f64 {
        let (along, across) = (self.r * th.cos(), self.r * th.sin());
        along + (self.l * self.l - across * across).max(0.0).sqrt()
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

    /// dérivées des poses par rapport à θ : (dx/dθ, dy/dθ, dφ/dθ) de chaque pièce (différences centrées)
    pub fn jacobian(&self, th: f64) -> [[f64; 3]; 3] {
        let h = 1e-6;
        let (p, m) = (self.poses(th + h), self.poses(th - h));
        std::array::from_fn(|k| [(p[k].0[0] - m[k].0[0]) / (2.0 * h), (p[k].0[1] - m[k].0[1]) / (2.0 * h), (p[k].1 - m[k].1) / (2.0 * h)])
    }

    /// inertie ramenée au vilebrequin (kg·m par mètre) : volant, puis Σ m |dG/dθ|² + I (dφ/dθ)² des pièces
    pub fn inertia_at(&self, th: f64, mass: &[f64; 3], inert: &[f64; 3]) -> f64 {
        let j = self.jacobian(th);
        self.inertia + (0..3).map(|k| mass[k] * (j[k][0] * j[k][0] + j[k][1] * j[k][1]) + inert[k] * j[k][2] * j[k][2]).sum::<f64>()
    }

    /// couple ramené au vilebrequin (N) des efforts `f` (Fx, Fy, couple autour du centre de masse) sur les pièces
    pub fn torque_at(&self, th: f64, f: &[[f64; 3]; 3]) -> f64 {
        let j = self.jacobian(th);
        (0..3).map(|k| f[k][0] * j[k][0] + f[k][1] * j[k][1] + f[k][2] * j[k][2]).sum()
    }

    /// pose les pièces (position, angle, vitesses) pour l'état (θ, ω)
    pub fn place(&self, bodies: &mut [RigidBody]) {
        let (p, j) = (self.poses(self.theta), self.jacobian(self.theta));
        for k in 0..3 {
            let b = &mut bodies[self.bodies[k]];
            b.pos = p[k].0;
            b.angle = p[k].1;
            b.vel = [j[k][0] * self.omega, j[k][1] * self.omega];
            b.omega = j[k][2] * self.omega;
        }
    }

    /// angle (°) de 0 à 360 depuis le point mort haut
    pub fn angle_deg(&self) -> f64 {
        self.theta.to_degrees().rem_euclid(360.0)
    }

    /// couple appliqué (N) à l'instant t pour l'état (θ, ω)
    fn applied_at(&self, t: f64, th: f64, w: f64) -> f64 {
        let Some(f) = &self.load else { return 0.0 };
        let v = [t, th.to_degrees().rem_euclid(360.0), th / std::f64::consts::TAU, w.to_degrees()];
        super::mp_fields::Formula::parse_vars(f, &CRANK_VARS).map_or(f64::NAN, |e| e.eval_vars(&v))
    }
}

impl MultiphaseSim {
    /// Pose un vilebrequin, sa bielle et son piston (corps menés, du matériau `material`, à la température `temp`) : manivelle de
    /// largeur `w` de l'axe à l'axe de bielle, bielle de largeur `w`, piston de largeur `bore`. Retourne l'indice du vilebrequin.
    pub fn add_crank(&mut self, mut ck: Crank, bore: f64, w: f64, material: usize, temp: f64) -> Result<usize, String> {
        if !(ck.r > 0.0 && ck.l.is_finite() && ck.l > 1.05 * ck.r) {
            return Err("la bielle doit être plus longue que le rayon de manivelle (au moins 1.05 fois)".into());
        }
        if !(bore > 0.0 && ck.piston_h > 0.0 && w > 0.0) {
            return Err("alésage, hauteur du piston et largeur des pièces doivent être positifs".into());
        }
        // (bielle inclinée d'au plus β = asin(r/l) : raccourcie pour que son coin ne morde pas sur le piston)
        let beta = (ck.r / ck.l).asin();
        ck.rod_len = (ck.l - 0.5 * w * beta.tan() - 0.05 * w).max(0.5 * ck.l);
        if !ck.name.trim().is_empty() {
            let n = super::mp_experiment::norm(&ck.name);
            if self.cranks.iter().any(|c| super::mp_experiment::norm(&c.name) == n) {
                return Err(format!("un vilebrequin s'appelle déjà « {} » (un composant posé deux fois : donnez-lui un autre nom)", ck.name));
            }
            for part in CRANK_PARTS {
                if self.body_named(&format!("{}.{part}", ck.name)).is_some() {
                    return Err(format!("un corps s'appelle déjà « {}.{part} »", ck.name));
                }
            }
        }
        let n = self.bodies.len();
        ck.bodies = [n, n + 1, n + 2];
        let shapes = [Shape::Rect { w: ck.r + w, h: w }, Shape::Rect { w: ck.rod_len, h: w }, Shape::Rect { w: bore, h: ck.piston_h }];
        let poses = ck.poses(ck.theta);
        for k in 0..3 {
            let mut b = RigidBody::new(shapes[k].clone(), material, poses[k].0, temp);
            b.angle = poses[k].1;
            b.motion = Motion::Driven;
            if !ck.name.trim().is_empty() {
                b.name = format!("{}.{}", ck.name, CRANK_PARTS[k]);
            }
            self.add_body(b)?;
        }
        ck.place(&mut self.bodies);
        for k in ck.bodies {
            self.bodies[k].last_vel = self.bodies[k].vel;
        }
        self.cranks.push(ck);
        self.update_body_face_velocities();
        Ok(self.cranks.len() - 1)
    }

    /// Avance les vilebrequins de t0 à t1 et pose leurs pièces : mené, à sa vitesse ; libre, par l'équation de Lagrange (point milieu,
    /// sous-pas d'au plus 0.005 rad) avec les forces du fluide de la dernière projection gardées sur le pas, les poids et le couple
    /// appliqué.
    pub(crate) fn advance_cranks(&mut self, t0: f64, t1: f64) {
        let dt = t1 - t0;
        for c in 0..self.cranks.len() {
            let ck = self.cranks[c].clone();
            let (mut mass, mut inert, mut f) = ([0.0; 3], [0.0; 3], [[0.0; 3]; 3]);
            for k in 0..3 {
                let b = &self.bodies[ck.bodies[k]];
                let rho = b.density(self.reg.solids[b.material].rho);
                mass[k] = rho * b.shape.area();
                inert[k] = rho * b.shape.polar_moment();
                let fl = self.body_force.get(ck.bodies[k]).copied().unwrap_or([0.0; 3]);
                f[k] = [fl[0] + mass[k] * self.gravity[0], fl[1] + mass[k] * self.gravity[1], fl[2]];
            }
            let torque = ck.torque_at(ck.theta, &f);
            let (mut th, mut w) = (ck.theta, ck.omega);
            let mut applied = 0.0;
            match ck.driven {
                Some(wd) => {
                    th += wd * dt;
                    w = wd;
                    applied = ck.applied_at(t0, ck.theta, wd);
                }
                None if dt > 0.0 => {
                    let acc = |t: f64, th: f64, w: f64| -> f64 {
                        let h = 1e-5;
                        let m = ck.inertia_at(th, &mass, &inert);
                        let dm = (ck.inertia_at(th + h, &mass, &inert) - ck.inertia_at(th - h, &mass, &inert)) / (2.0 * h);
                        (ck.torque_at(th, &f) + ck.applied_at(t, th, w) - 0.5 * dm * w * w) / m.max(1e-300)
                    };
                    let n = ((w.abs().max(1.0) * dt / 0.005).ceil() as usize).clamp(1, 10_000);
                    let h = dt / n as f64;
                    for s in 0..n {
                        let t = t0 + s as f64 * h;
                        let a1 = acc(t, th, w);
                        let (thm, wm) = (th + 0.5 * h * w, w + 0.5 * h * a1);
                        let a2 = acc(t + 0.5 * h, thm, wm);
                        if s == 0 {
                            applied = ck.applied_at(t + 0.5 * h, thm, wm);
                        }
                        th += h * wm;
                        w += h * a2;
                    }
                }
                None => {}
            }
            let ck = &mut self.cranks[c];
            ck.theta = th;
            ck.omega = w;
            ck.torque = torque;
            ck.applied = applied;
            let ck = ck.clone();
            ck.place(&mut self.bodies);
        }
    }
}
