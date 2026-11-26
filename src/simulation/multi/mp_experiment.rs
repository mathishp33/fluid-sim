//! Expériences reproductibles (voir EXPERIENCES.md) : un PROTOCOLE — un départ, des mesures, une suite d'actions en temps simulé —
//! exécuté pas à pas, sans egui : par la fenêtre d'expérience (`window/experiment_window.rs`) ou en ligne de commande
//! (`fluid-sim experience protocole.exp`, voir `run_cli`). Le script en français est lu par `mp_script`, les champs de valeurs
//! (fichiers CSV / PNG, formules) par `mp_fields`.
//!
//! * Les actions tombent PILE à leur heure : le pas qui précède une échéance est raccourci (`dt_max`), et partagé en deux quand
//!   l'échéance tombe entre un et deux pas (pas de pas minuscule).
//! * Les mesures ne touchent jamais au pas : elles sont relevées au premier pas qui atteint leur heure (le temps réel du pas est
//!   noté). Mesurer ne change pas le résultat.
//! * Sur le CPU (f64), un protocole rejoué redonne les mêmes nombres au bit près, quel que soit le nombre de fils. Le GPU (f32) suit
//!   le même protocole, plus vite, sans cette garantie.
//! * Les zones sont en mètres (origine en bas à gauche, y vers le haut) : le même protocole tourne à n'importe quelle résolution.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::gpu_multiphase::{gpu_unsupported_reason, MultiphaseGpu};
use super::mp_cells::{MpEffect, MpInlet, MpThermal};
use super::mp_materials::MaterialLibrary;
use super::mp_state::{is_state_file, state_from_bytes, state_to_bytes};
use super::mp_view::{apply_tool_with, build_scenario_with, cell_velocity, compute_stats, MpScenario, MpTool, ToolOpts};
use super::multiphase::{Bc, MultiphaseSim};

// ------------------------------------------------------------------------------------------------ protocole

/// Zone d'action ou de mesure, en mètres. Une maille en fait partie si son CENTRE y est ; une zone trop petite pour contenir un
/// centre (disque de 1 mm sur une maille de 1 cm) prend la maille de son centre.
#[derive(Clone, Debug, PartialEq)]
pub enum Zone {
    Partout,
    Disque { c: [f64; 2], r: f64 },
    Rect { a: [f64; 2], b: [f64; 2] },
    /// la maille qui contient le point
    Point([f64; 2]),
    /// trait de pinceau : mailles à moins de `r` de la ligne brisée
    Trait { pts: Vec<[f64; 2]>, r: f64 },
}

fn dist2_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 { (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
    let (qx, qy) = (a[0] + t * dx - p[0], a[1] + t * dy - p[1]);
    qx * qx + qy * qy
}

impl Zone {
    /// Mailles (i, j) de la zone sur une grille nx × ny de pas dx.
    pub fn cells(&self, nx: usize, ny: usize, dx: f64) -> Vec<(usize, usize)> {
        let tol = 1e-9 * dx;
        let center = |i: usize, j: usize| ((i as f64 + 0.5) * dx, (j as f64 + 0.5) * dx);
        let ci = |x: f64, n: usize| ((x / dx).floor().max(0.0) as usize).min(n - 1);
        let inside = |p: [f64; 2]| p[0] >= 0.0 && p[1] >= 0.0 && p[0] <= nx as f64 * dx && p[1] <= ny as f64 * dx;
        let mut v = vec![];
        // boîte englobante (mètres) → plage de mailles
        let mut scan = |x0: f64, y0: f64, x1: f64, y1: f64, test: &dyn Fn(f64, f64) -> bool| {
            if x1 < 0.0 || y1 < 0.0 || x0 > nx as f64 * dx || y0 > ny as f64 * dx {
                return;
            }
            for j in ci(y0, ny)..=ci(y1, ny) {
                for i in ci(x0, nx)..=ci(x1, nx) {
                    let (x, y) = center(i, j);
                    if test(x, y) {
                        v.push((i, j));
                    }
                }
            }
        };
        match self {
            Zone::Partout => {
                for j in 0..ny {
                    for i in 0..nx {
                        v.push((i, j));
                    }
                }
            }
            Zone::Point(p) => {
                if inside(*p) {
                    v.push((ci(p[0], nx), ci(p[1], ny)));
                }
            }
            Zone::Rect { a, b } => {
                let (x0, x1) = (a[0].min(b[0]) - tol, a[0].max(b[0]) + tol);
                let (y0, y1) = (a[1].min(b[1]) - tol, a[1].max(b[1]) + tol);
                scan(x0, y0, x1, y1, &|x, y| x >= x0 && x <= x1 && y >= y0 && y <= y1);
            }
            Zone::Disque { c, r } => {
                let rr = (r + tol) * (r + tol);
                let c = *c;
                scan(c[0] - r, c[1] - r, c[0] + r, c[1] + r, &|x, y| (x - c[0]).powi(2) + (y - c[1]).powi(2) <= rr);
            }
            Zone::Trait { pts, r } => {
                if !pts.is_empty() {
                    let rr = (r + tol) * (r + tol);
                    let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
                    for p in pts {
                        x0 = x0.min(p[0]);
                        y0 = y0.min(p[1]);
                        x1 = x1.max(p[0]);
                        y1 = y1.max(p[1]);
                    }
                    let pts = pts.clone();
                    scan(x0 - r, y0 - r, x1 + r, y1 + r, &|x, y| {
                        if pts.len() == 1 {
                            return (x - pts[0][0]).powi(2) + (y - pts[0][1]).powi(2) <= rr;
                        }
                        pts.windows(2).any(|w| dist2_segment([x, y], w[0], w[1]) <= rr)
                    });
                }
            }
        }
        if v.is_empty() {
            if let Some(p) = self.anchor().filter(|&p| inside(p)) {
                v.push((ci(p[0], nx), ci(p[1], ny)));
            }
        }
        v
    }

    /// point de repli d'une zone qui ne contient aucun centre de maille
    fn anchor(&self) -> Option<[f64; 2]> {
        match self {
            Zone::Partout => None,
            Zone::Point(p) => Some(*p),
            Zone::Disque { c, .. } => Some(*c),
            Zone::Rect { a, b } => Some([0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]),
            Zone::Trait { pts, .. } => pts.first().copied(),
        }
    }

    /// Boîte englobante (coin bas-gauche, coin haut-droit) dans un domaine lx × ly.
    pub fn bbox(&self, lx: f64, ly: f64, dx: f64) -> ([f64; 2], [f64; 2]) {
        match self {
            Zone::Partout => ([0.0, 0.0], [lx, ly]),
            Zone::Point(p) => ([p[0] - 0.5 * dx, p[1] - 0.5 * dx], [p[0] + 0.5 * dx, p[1] + 0.5 * dx]),
            Zone::Disque { c, r } => ([c[0] - r, c[1] - r], [c[0] + r, c[1] + r]),
            Zone::Rect { a, b } => ([a[0].min(b[0]), a[1].min(b[1])], [a[0].max(b[0]), a[1].max(b[1])]),
            Zone::Trait { pts, r } => {
                let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
                for p in pts {
                    x0 = x0.min(p[0]);
                    y0 = y0.min(p[1]);
                    x1 = x1.max(p[0]);
                    y1 = y1.max(p[1]);
                }
                ([x0 - r, y0 - r], [x1 + r, y1 + r])
            }
        }
    }

    /// Texte court (libellés de colonnes, journal), même écriture que le script.
    pub fn label(&self) -> String {
        match self {
            Zone::Partout => "partout".into(),
            Zone::Point(p) => format!("en {}", fmt_point(*p)),
            Zone::Disque { c, r } => format!("disque {} rayon {}", fmt_point(*c), fmt_num(*r)),
            Zone::Rect { a, b } => format!("rect {} {}", fmt_point(*a), fmt_point(*b)),
            Zone::Trait { pts, r } => format!("trait de {} points rayon {}", pts.len(), fmt_num(*r)),
        }
    }
}

/// nombre écrit sans zéros inutiles (6 chiffres significatifs au plus après la virgule) : « 0.155 », « 2 », « 1e-05 »
pub fn fmt_num(x: f64) -> String {
    if x != 0.0 && (x.abs() < 1e-4 || x.abs() >= 1e7) {
        return format!("{x:e}");
    }
    let s = format!("{x:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0".into() } else { s.into() }
}

pub fn fmt_point(p: [f64; 2]) -> String {
    format!("({}, {})", fmt_num(p[0]), fmt_num(p[1]))
}

/// Grandeur par maille mesurée par une sonde.
#[derive(Clone, Debug, PartialEq)]
pub enum Quantity {
    T,
    P,
    Vx,
    Vy,
    Speed,
    Rho,
    /// fraction volumique d'un fluide pur
    Fraction(String),
    /// puissance chimique dégagée (W/m³)
    HeatRelease,
}

impl Quantity {
    fn name(&self) -> String {
        match self {
            Quantity::T => "T".into(),
            Quantity::P => "p".into(),
            Quantity::Vx => "vx".into(),
            Quantity::Vy => "vy".into(),
            Quantity::Speed => "vitesse".into(),
            Quantity::Rho => "rho".into(),
            Quantity::Fraction(f) => format!("fraction {f}"),
            Quantity::HeatRelease => "chaleur".into(),
        }
    }
    fn unit(&self, reduce: Reduce) -> &'static str {
        match (self, reduce) {
            (Quantity::T, _) => "K",
            (Quantity::P, _) => "Pa",
            (Quantity::Vx | Quantity::Vy | Quantity::Speed, _) => "m/s",
            (Quantity::Rho, Reduce::Sum) => "kg/m",
            (Quantity::Rho, _) => "kg/m³",
            (Quantity::Fraction(_), Reduce::Sum) => "m²",
            (Quantity::Fraction(_), _) => "-",
            (Quantity::HeatRelease, Reduce::Sum) => "W/m",
            (Quantity::HeatRelease, _) => "W/m³",
        }
    }
    /// grandeur du fluide (sans sens dans une maille solide) ?
    fn fluid_only(&self) -> bool {
        !matches!(self, Quantity::T)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reduce {
    /// valeur en un point, interpolée entre les mailles voisines (`point_value`)
    At,
    Mean,
    Max,
    Min,
    /// intégrale sur la zone (× dx², par mètre de profondeur)
    Sum,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Measure {
    /// statistiques globales (masses par fluide, énergies, T et p extrêmes, chaleur de réaction…)
    Stats,
    Probe { q: Quantity, reduce: Reduce, zone: Zone, label: Option<String> },
    /// masse d'un fluide pur dans une zone (kg par mètre de profondeur)
    Mass { fluid: String, zone: Zone, label: Option<String> },
    /// position, vitesse ou force du fluide sur un corps rigide (indice à partir de 0, dans l'ordre de pose)
    Body { index: usize, q: BodyQty, label: Option<String> },
    /// force du fluide (composante x : 0, y : 1) sur les mailles solides de la zone, par mètre de profondeur
    SolidForce { axis: usize, zone: Zone, label: Option<String> },
    /// débit volumique (m²/s) ou massique (`mass`, kg/(m·s)) d'un fluide pur, ou de tous, à travers le segment horizontal ou vertical
    /// (a, b), le long des faces des mailles ; positif vers +x (segment vertical) ou +y (horizontal)
    Flux { fluid: Option<String>, mass: bool, a: [f64; 2], b: [f64; 2], label: Option<String> },
}

/// Grandeur mesurée sur un corps rigide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BodyQty {
    X,
    Y,
    Angle,
    Vx,
    Vy,
    Rotation,
    Fx,
    Fy,
    Torque,
    Temp,
}

impl BodyQty {
    pub fn name(&self) -> &'static str {
        match self {
            BodyQty::X => "x",
            BodyQty::Y => "y",
            BodyQty::Angle => "angle",
            BodyQty::Vx => "vx",
            BodyQty::Vy => "vy",
            BodyQty::Rotation => "rotation",
            BodyQty::Fx => "fx",
            BodyQty::Fy => "fy",
            BodyQty::Torque => "couple",
            BodyQty::Temp => "température",
        }
    }
    fn unit(&self) -> &'static str {
        match self {
            BodyQty::X | BodyQty::Y => "m",
            BodyQty::Angle => "°",
            BodyQty::Vx | BodyQty::Vy => "m/s",
            BodyQty::Rotation => "°/s",
            BodyQty::Fx | BodyQty::Fy => "N/m",
            BodyQty::Torque => "N",
            BodyQty::Temp => "K",
        }
    }
    /// valeur sur le corps k (NaN s'il n'existe pas) ; forces : celles du fluide à la dernière projection (couple autour du centre de
    /// masse, par mètre de profondeur)
    fn value(&self, sim: &MultiphaseSim, k: usize) -> f64 {
        let Some(b) = sim.bodies.get(k) else { return f64::NAN };
        let f = sim.body_force.get(k).copied().unwrap_or([f64::NAN; 3]);
        match self {
            BodyQty::X => b.pos[0],
            BodyQty::Y => b.pos[1],
            BodyQty::Angle => b.angle.to_degrees(),
            BodyQty::Vx => b.vel[0],
            BodyQty::Vy => b.vel[1],
            BodyQty::Rotation => b.omega.to_degrees(),
            BodyQty::Fx => f[0],
            BodyQty::Fy => f[1],
            BodyQty::Torque => f[2],
            BodyQty::Temp => b.temp,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ThermalSpec {
    Passive,
    Isothermal(f64),
    Heater(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub enum BoundarySpec {
    Wall,
    /// ouvert à la pression p (Pa) au point le plus haut du bord
    Open(f64),
    /// paroi à vitesse imposée qui fait entrer un fluide
    Inlet { v: [f64; 2], temp: Option<f64>, fluid: Option<String> },
    /// paroi à vitesse imposée, dirigée vers l'extérieur : le fluide des mailles du bord sort (pompe, vidange)
    Outlet { v: [f64; 2] },
}

/// Grandeur d'un champ de valeurs (conditions initiales).
#[derive(Clone, Debug, PartialEq)]
pub enum FieldQty {
    T,
    Vx,
    Vy,
    /// fraction volumique d'un fluide ou d'un mélange (0 à 1) ; le reste de la maille garde sa composition
    Fraction(String),
    /// solide du matériau là où la valeur ≥ 0.5
    Solid(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum FieldSource {
    /// grille de valeurs (CSV ou PNG) étalée sur la boîte de la zone ; `top_origin` : première ligne = haut ; `range` : valeurs
    /// (a, b) des niveaux 0 et 1 d'une image
    File { path: String, top_origin: bool, range: Option<(f64, f64)> },
    /// formule en x, y (m), lx, ly (taille du domaine)
    Formula(String),
}

/// Mouvement d'un corps posé par un protocole : fixe (tout à zéro) ou imposé — vitesse et rotation constantes, plus une oscillation
/// sinusoïdale de période `period`, éventuellement autour d'un pivot (rotation seule).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct BodyMotion {
    pub v: [f64; 2],
    /// rad/s
    pub omega: f64,
    pub osc_v: [f64; 2],
    pub osc_omega: f64,
    pub period: f64,
    pub pivot: Option<[f64; 2]>,
    /// corps libre, poussé par l'écoulement : `v` et `omega` sont alors sa vitesse de départ
    pub free: bool,
    /// axes bloqués d'un corps libre (x, y, rotation)
    pub locked: [bool; 3],
    /// masse volumique propre du corps (kg/m³), à la place de celle du matériau
    pub rho: Option<f64>,
    /// contacts : coefficient de restitution et de frottement (sinon ceux par défaut)
    pub restitution: Option<f64>,
    pub friction: Option<f64>,
    /// lois en formules (variables `mp_bodies::LAW_VARS`) : translation et rotation imposées ; force (N/m) et couple (N) ajoutés à un
    /// corps libre
    pub lin_law: Option<super::mp_bodies::LinLaw>,
    pub rot_law: Option<super::mp_bodies::RotLaw>,
    pub force: Option<[String; 2]>,
    pub torque: Option<String>,
}

impl BodyMotion {
    pub fn is_fixed(&self) -> bool {
        !self.free
            && self.v == [0.0, 0.0]
            && self.omega == 0.0
            && self.osc_v == [0.0, 0.0]
            && self.osc_omega == 0.0
            && self.lin_law.is_none()
            && self.rot_law.is_none()
    }
}

/// Forme d'un corps rigide posé par un protocole (mètres, repère du domaine).
#[derive(Clone, Debug, PartialEq)]
pub enum BodySpec {
    Disc { c: [f64; 2], r: f64 },
    /// rectangle donné par deux coins (avant rotation de `angle` autour de son centre)
    Rect { a: [f64; 2], b: [f64; 2] },
    /// polygone convexe (sommets dans n'importe quel sens)
    Polygon(Vec<[f64; 2]>),
}

impl BodySpec {
    /// zone approchée (affichage sur l'image)
    pub fn zone(&self) -> Zone {
        match self {
            BodySpec::Disc { c, r } => Zone::Disque { c: *c, r: *r },
            BodySpec::Rect { a, b } => Zone::Rect { a: *a, b: *b },
            BodySpec::Polygon(p) => {
                let mut pts = p.clone();
                if let Some(&f) = p.first() {
                    pts.push(f);
                }
                Zone::Trait { pts, r: 0.0 }
            }
        }
    }

    /// corps rigide de matériau `material`, tourné de `angle` (rad) autour de son centre de masse : fixe, à mouvement imposé ou libre
    pub fn body(&self, material: usize, angle: f64, temp: f64, motion: &BodyMotion) -> Result<super::mp_bodies::RigidBody, String> {
        use super::mp_bodies::{Dof, ForceLaw, Motion, MotionLaw, RigidBody, Shape};
        let (shape, pos) = match self {
            BodySpec::Disc { c, r } => (Shape::Disc { r: *r }, *c),
            BodySpec::Rect { a, b } => {
                let (w, h) = ((b[0] - a[0]).abs(), (b[1] - a[1]).abs());
                if !(w > 0.0 && h > 0.0) {
                    return Err("rectangle plat : ses deux coins doivent différer en x et en y".into());
                }
                (Shape::Rect { w, h }, [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])])
            }
            BodySpec::Polygon(p) => Shape::polygon(p)?,
        };
        let mut b = RigidBody::new(shape, material, pos, temp);
        b.angle = angle;
        b.rho = motion.rho;
        if let Some(e) = motion.restitution {
            b.restitution = e;
        }
        if let Some(m) = motion.friction {
            b.friction = m;
        }
        if motion.free {
            b.motion = Motion::Free;
            b.dof = Dof { x: !motion.locked[0], y: !motion.locked[1], rot: !motion.locked[2] };
            b.pivot = motion.pivot;
            // vitesse de départ (ramenée aux mouvements permis par `add_body`)
            b.vel = motion.v;
            b.omega = motion.omega;
            if motion.force.is_some() || motion.torque.is_some() {
                b.force_law = Some(Box::new(ForceLaw { force: motion.force.clone(), torque: motion.torque.clone(), start: pos }));
            }
        } else if motion.is_fixed() {
            b.motion = Motion::Fixed;
        } else if motion.lin_law.is_some() || motion.rot_law.is_some() {
            let law = MotionLaw { lin: motion.lin_law.clone(), v: motion.v, rot: motion.rot_law.clone(), omega: motion.omega, start: pos };
            b.motion = Motion::Law(Box::new(law));
            b.pivot = motion.pivot;
            // vitesse de départ d'une accélération imposée
            b.vel = motion.v;
            b.omega = motion.omega;
        } else {
            let m = motion;
            b.motion = Motion::Imposed { v: m.v, omega: m.omega, osc_v: m.osc_v, osc_omega: m.osc_omega, period: m.period };
            b.pivot = m.pivot;
        }
        Ok(b)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    Fill { zone: Zone, fluid: String, temp: Option<f64> },
    /// corps rigide, fixe ou à mouvement imposé (solides mobiles : voir SOLID.md)
    /// `temp` : température du corps au départ (sinon celle du fluide ambiant)
    Body { shape: BodySpec, material: String, angle: f64, motion: BodyMotion, temp: Option<f64> },
    SetTemp { zone: Zone, temp: f64 },
    Solid { zone: Zone, material: String, thermal: ThermalSpec, temp: Option<f64> },
    Clear { zone: Zone },
    /// chauffage (cible de température du fluide) pendant `duration` (None : jusqu'à la fin)
    Heat { zone: Zone, temp: f64, duration: Option<f64> },
    /// accélération de volume pendant `duration` (None : permanente)
    Accelerate { zone: Zone, a: [f64; 2], duration: Option<f64> },
    Velocity { zone: Zone, v: [f64; 2] },
    Ignite { zone: Zone, temp: f64 },
    Sink { zone: Zone, rate: f64 },
    Emitter { zone: Zone, fluid: String, v: [f64; 2], temp: Option<f64> },
    Grille { zone: Zone, hole: f64, open: f64 },
    Boundary { side: usize, spec: BoundarySpec },
    Field { qty: FieldQty, source: FieldSource, zone: Zone, temp: Option<f64> },
    Gravity([f64; 2]),
    DtMax(f64),
    SaveState(String),
    /// image PNG du champ `display` dans le dossier des résultats (`nom.png`)
    SaveImage { name: String, display: super::mp_view::MpDisplay },
    Mark(String),
    /// met l'expérience en pause (fenêtre) ; ignorée en ligne de commande
    Pause,
    /// action répétée toutes les `period` secondes, de `start` (sinon l'instant où la ligne est atteinte) à `end` (sinon sans fin),
    /// en parallèle du déroulement : chaque fois à l'heure pile
    Periodic { period: f64, start: Option<f64>, end: Option<f64>, action: Box<Action> },
}

impl Action {
    /// action qui modifie l'état sur-le-champ (et qu'on peut donc rejouer pour construire les conditions initiales) ; une action
    /// répétée dès le départ compte pour sa première fois
    pub fn is_instant_edit(&self) -> bool {
        match self {
            Action::Periodic { start, action, .. } => start.map_or(true, |s| s <= 0.0) && action.is_instant_edit(),
            _ => !matches!(self, Action::Heat { .. } | Action::Accelerate { .. } | Action::SaveState(_) | Action::SaveImage { .. } | Action::Mark(_) | Action::Pause),
        }
    }
    /// zone de l'action (pour l'afficher sur l'image)
    pub fn zone(&self) -> Option<&Zone> {
        match self {
            Action::Fill { zone, .. }
            | Action::SetTemp { zone, .. }
            | Action::Solid { zone, .. }
            | Action::Clear { zone }
            | Action::Heat { zone, .. }
            | Action::Accelerate { zone, .. }
            | Action::Velocity { zone, .. }
            | Action::Ignite { zone, .. }
            | Action::Sink { zone, .. }
            | Action::Emitter { zone, .. }
            | Action::Grille { zone, .. }
            | Action::Field { zone, .. } => Some(zone),
            Action::Periodic { action, .. } => action.zone(),
            _ => None,
        }
    }
}

/// Comparaison d'une condition d'arrêt (`simuler jusqu'à ce que <relevé> < v`)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cmp {
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cmp {
    pub fn holds(self, x: f64, v: f64) -> bool {
        match self {
            Cmp::Lt => x < v,
            Cmp::Le => x <= v,
            Cmp::Gt => x > v,
            Cmp::Ge => x >= v,
        }
    }
    pub fn symbol(self) -> &'static str {
        match self {
            Cmp::Lt => "<",
            Cmp::Le => "≤",
            Cmp::Gt => ">",
            Cmp::Ge => "≥",
        }
    }
}

/// une comparaison d'une condition d'arrêt : relevé (comme une mesure), comparaison, seuil dans l'unité du relevé
#[derive(Clone, Debug, PartialEq)]
pub struct Clause {
    pub measure: Measure,
    pub cmp: Cmp,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    Do(Action),
    /// avance de cette durée
    Simulate(f64),
    /// avance jusqu'à ce temps simulé
    Until(f64),
    /// avance jusqu'à ce que les comparaisons `clauses` soient toutes vraies (`any` : l'une d'elles), vérifiées après chaque pas (ou
    /// chaque série de pas du GPU), au plus pendant `limit` secondes si elle est donnée
    UntilCond { clauses: Vec<Clause>, any: bool, limit: Option<f64> },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Start {
    Scenario { name: String, nx: usize, ny: usize, dx: f64, g: f64 },
    /// état complet (.mps) enregistré par une fenêtre ou une expérience
    State(String),
}

/// Protocole lu (`mp_script::parse_all`) : chaque élément porte sa ligne (à partir de 1) pour les messages et l'affichage.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Protocol {
    pub start: Option<(Start, usize)>,
    /// bibliothèque de matériaux (JSON exporté par l'éditeur) ; sinon matériaux par défaut
    pub materials: Option<String>,
    /// calcul demandé : Some(true) = GPU
    pub gpu: Option<bool>,
    /// période des mesures (s)
    pub period: Option<f64>,
    /// séries d'images PNG à intervalles réguliers : (période, affichage, ligne)
    pub images: Vec<(f64, super::mp_view::MpDisplay, usize)>,
    pub measures: Vec<(Measure, usize)>,
    pub steps: Vec<(Step, usize)>,
    /// texte du protocole (après substitution du balayage)
    pub source: String,
    /// (nom, valeur) de la variante d'un balayage
    pub variant: Option<(String, String)>,
}

impl Protocol {
    /// durée simulée prévue (fin du dernier « simuler »), pour la barre de progression
    pub fn planned_end(&self, t0: f64) -> f64 {
        let mut t = t0;
        for (s, _) in &self.steps {
            match s {
                Step::Simulate(d) => t += d,
                Step::Until(u) => t = t.max(*u),
                Step::UntilCond { limit, .. } => t += limit.unwrap_or(0.0),
                Step::Do(_) => {}
            }
        }
        t
    }
    /// indice de la première étape qui fait avancer le temps (les actions avant elle sont les conditions initiales)
    pub fn first_advance(&self) -> usize {
        self.steps.iter().position(|(s, _)| !matches!(s, Step::Do(_))).unwrap_or(self.steps.len())
    }
}

// ------------------------------------------------------------------------------------------------ noms

/// minuscules sans accents ni apostrophes (comparaison des mots-clés et des noms)
pub fn norm(s: &str) -> String {
    s.chars()
        .filter(|c| !matches!(c, '\'' | '’'))
        .flat_map(|c| c.to_lowercase())
        .map(|c| match c {
            'à' | 'â' | 'ä' | 'á' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'î' | 'ï' | 'í' => 'i',
            'ô' | 'ö' | 'ó' => 'o',
            'ù' | 'û' | 'ü' | 'ú' => 'u',
            'ç' => 'c',
            '²' => '2',
            '³' => '3',
            '–' | '—' => '-',
            o => o,
        })
        .collect()
}

/// Cherche `name` parmi `names` : égalité (sans accents ni casse), sinon début de nom unique, sinon nom qui le contient (unique).
pub fn find_name(names: &[String], name: &str, what: &str) -> Result<usize, String> {
    let n = norm(name.trim());
    let normed: Vec<String> = names.iter().map(|s| norm(s)).collect();
    if let Some(i) = normed.iter().position(|s| *s == n) {
        return Ok(i);
    }
    let starts: Vec<usize> = (0..names.len()).filter(|&i| normed[i].starts_with(&n)).collect();
    if starts.len() == 1 {
        return Ok(starts[0]);
    }
    let contains: Vec<usize> = (0..names.len()).filter(|&i| normed[i].contains(&n)).collect();
    if contains.len() == 1 {
        return Ok(contains[0]);
    }
    let list = names.iter().map(|s| format!("« {s} »")).collect::<Vec<_>>().join(", ");
    if starts.len() > 1 || contains.len() > 1 {
        let amb: Vec<usize> = if starts.len() > 1 { starts } else { contains };
        let l = amb.iter().map(|&i| format!("« {} »", names[i])).collect::<Vec<_>>().join(", ");
        return Err(format!("{what} « {name} » ambigu : {l}"));
    }
    Err(format!("{what} « {name} » inconnu ; disponibles : {list}"))
}

/// Fluide pur ou mélange nommé.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FluidRef {
    Phase(usize),
    Mixture(usize),
}

pub fn find_fluid(sim: &MultiphaseSim, name: &str) -> Result<FluidRef, String> {
    let np = sim.reg.phases.len();
    let mut names: Vec<String> = sim.reg.phases.iter().map(|p| p.name.clone()).collect();
    names.extend(sim.mixtures.iter().map(|m| m.0.clone()));
    let i = find_name(&names, name, "fluide")?;
    Ok(if i < np { FluidRef::Phase(i) } else { FluidRef::Mixture(i - np) })
}

pub fn find_phase(sim: &MultiphaseSim, name: &str) -> Result<usize, String> {
    match find_fluid(sim, name)? {
        FluidRef::Phase(k) => Ok(k),
        FluidRef::Mixture(_) => Err(format!("« {name} » est un mélange : il faut ici un fluide pur")),
    }
}

pub fn find_solid(sim: &MultiphaseSim, name: &str) -> Result<usize, String> {
    let names: Vec<String> = sim.reg.solids.iter().map(|s| s.name.clone()).collect();
    if names.is_empty() {
        return Err("aucun matériau solide défini".into());
    }
    find_name(&names, name, "matériau solide")
}

/// Scénario par son nom (ou une partie de son nom, ou son numéro dans la liste).
pub fn find_scenario(name: &str) -> Result<MpScenario, String> {
    if let Ok(i) = name.trim().parse::<usize>() {
        return MpScenario::ALL.get(i).copied().ok_or_else(|| format!("scénario n° {i} inconnu (0 à {})", MpScenario::ALL.len() - 1));
    }
    let names: Vec<String> = MpScenario::ALL.iter().map(|s| s.name().to_string()).collect();
    find_name(&names, name, "scénario").map(|i| MpScenario::ALL[i])
}

/// composition (fractions volumiques) d'un fluide ou d'un mélange
fn composition(sim: &MultiphaseSim, f: FluidRef) -> Vec<f64> {
    let np = sim.reg.phases.len();
    match f {
        FluidRef::Phase(k) => (0..np).map(|j| if j == k { 1.0 } else { 0.0 }).collect(),
        FluidRef::Mixture(m) => {
            let a = &sim.mixtures[m].1;
            let s: f64 = a.iter().sum::<f64>().max(1e-300);
            (0..np).map(|j| a.get(j).copied().unwrap_or(0.0) / s).collect()
        }
    }
}

/// Vérifie les noms (fluides, matériaux) du protocole dans le registre de la simulation : toutes les erreurs, avec leur ligne.
pub fn check_names(proto: &Protocol, sim: &MultiphaseSim) -> Vec<(usize, String)> {
    let mut errs = vec![];
    let fluid = |name: &str, line: usize, pure: bool, errs: &mut Vec<(usize, String)>| {
        let r = if pure { find_phase(sim, name).map(|_| ()) } else { find_fluid(sim, name).map(|_| ()) };
        if let Err(e) = r {
            errs.push((line, e));
        }
    };
    for (m, line) in &proto.measures {
        match m {
            Measure::Probe { q: Quantity::Fraction(f), .. } => fluid(f, *line, true, &mut errs),
            Measure::Mass { fluid: f, .. } | Measure::Flux { fluid: Some(f), .. } => fluid(f, *line, true, &mut errs),
            _ => {}
        }
    }
    for (s, line) in &proto.steps {
        let Step::Do(a) = s else { continue };
        let a = if let Action::Periodic { action, .. } = a { action.as_ref() } else { a };
        match a {
            Action::Fill { fluid: f, .. } => fluid(f, *line, false, &mut errs),
            Action::Emitter { fluid: f, .. } => fluid(f, *line, true, &mut errs),
            Action::Boundary { spec: BoundarySpec::Inlet { fluid: Some(f), .. }, .. } => fluid(f, *line, true, &mut errs),
            Action::Field { qty: FieldQty::Fraction(f), .. } => fluid(f, *line, false, &mut errs),
            Action::Solid { material, .. } | Action::Body { material, .. } | Action::Field { qty: FieldQty::Solid(material), .. } => {
                if let Err(e) = find_solid(sim, material) {
                    errs.push((*line, e));
                }
            }
            _ => {}
        }
    }
    errs
}

// ------------------------------------------------------------------------------------------------ départ et actions

/// Construit l'état de départ du protocole : (simulation, bibliothèque de matériaux, scénario).
pub fn build_start(proto: &Protocol, base: &Path) -> Result<(MultiphaseSim, Option<MaterialLibrary>, Option<usize>), String> {
    let (start, line) = proto.start.as_ref().ok_or_else(|| "le protocole doit commencer par une ligne « départ … »".to_string())?;
    let lib = match &proto.materials {
        Some(f) => {
            let p = base.join(f);
            let s = std::fs::read_to_string(&p).map_err(|e| format!("matériaux « {} » : {e}", p.display()))?;
            serde_json::from_str::<MaterialLibrary>(&s).map_err(|e| format!("matériaux « {} » illisibles : {e}", p.display()))?
        }
        None => MaterialLibrary::default(),
    };
    match start {
        Start::Scenario { name, nx, ny, dx, g } => {
            let sc = find_scenario(name).map_err(|e| format!("ligne {line} : {e}"))?;
            if *nx < 4 || *ny < 4 || *nx * *ny > 4_000_000 {
                return Err(format!("ligne {line} : grille {nx} × {ny} hors des limites (4 à 2000 × 2000)"));
            }
            let sim = build_scenario_with(&lib, sc, *nx, *ny, *dx, *g).map_err(|e| format!("ligne {line} : {e}"))?;
            Ok((sim, Some(lib), Some(sc.index())))
        }
        Start::State(f) => {
            let p = base.join(f);
            let bytes = std::fs::read(&p).map_err(|e| format!("ligne {line} : état « {} » : {e}", p.display()))?;
            if !is_state_file(&bytes) {
                return Err(format!("ligne {line} : « {} » n'est pas un état complet (fichier « Sauver l'état complet »)", p.display()));
            }
            let st = state_from_bytes(&bytes).map_err(|e| format!("ligne {line} : {e}"))?;
            Ok((st.sim, st.library, st.scenario))
        }
    }
}

fn idx_cells(sim: &MultiphaseSim, zone: &Zone) -> Vec<usize> {
    zone.cells(sim.nx, sim.ny, sim.dx).into_iter().map(|(i, j)| sim.idx(i, j)).collect()
}

fn tool_opts(temp: f64, thermal: MpThermal) -> ToolOpts {
    ToolOpts { temp, cursor: (0.0, 0.0), dt: 0.0, thermal }
}

/// Applique une action INSTANTANÉE (voir `Action::is_instant_edit`) ; retourne le nombre de mailles touchées. Les autres actions
/// (chauffage, accélération, sauvegarde, repère, pause) relèvent du déroulement (`Runner`) et sont ignorées ici.
pub fn apply_action(sim: &mut MultiphaseSim, a: &Action, base: &Path) -> Result<usize, String> {
    let t_amb = sim.t_ambient;
    let cells = |sim: &MultiphaseSim, z: &Zone| z.cells(sim.nx, sim.ny, sim.dx);
    match a {
        Action::Fill { zone, fluid, temp } => {
            let tool = match find_fluid(sim, fluid)? {
                FluidRef::Phase(k) => MpTool::Phase(k),
                FluidRef::Mixture(m) => MpTool::Mixture(m),
            };
            let cs = cells(sim, zone);
            apply_tool_with(sim, tool, &cs, &tool_opts(temp.unwrap_or(t_amb), MpThermal::Passive));
            Ok(cs.len())
        }
        Action::SetTemp { zone, temp } => {
            let cs = idx_cells(sim, zone);
            let np = sim.reg.phases.len();
            for &c in &cs {
                if sim.is_solid(c) {
                    sim.set_cell_temperature(c, *temp);
                } else {
                    let al: Vec<f64> = (0..np).map(|k| sim.alpha[k][c]).collect();
                    sim.fill_mixture(c, &al, *temp);
                }
            }
            sim.refresh();
            Ok(cs.len())
        }
        Action::Solid { zone, material, thermal, temp } => {
            let s = find_solid(sim, material)?;
            let (th, t) = match thermal {
                ThermalSpec::Passive => (MpThermal::Passive, temp.unwrap_or(t_amb)),
                ThermalSpec::Isothermal(ti) => (MpThermal::Isothermal(*ti), *ti),
                ThermalSpec::Heater(w) => (MpThermal::Heater(*w), temp.unwrap_or(t_amb)),
            };
            let cs = cells(sim, zone);
            apply_tool_with(sim, MpTool::Solid(s), &cs, &tool_opts(t, th));
            Ok(cs.len())
        }
        Action::Clear { zone } => {
            let cs = cells(sim, zone);
            apply_tool_with(sim, MpTool::ClearCell, &cs, &tool_opts(t_amb, MpThermal::Passive));
            Ok(cs.len())
        }
        Action::Body { shape, material, angle, motion, temp } => {
            let s = find_solid(sim, material)?;
            let b = shape.body(s, *angle, temp.unwrap_or(t_amb), motion)?;
            sim.add_body(b)?;
            Ok(sim.body_frac.iter().filter(|&&a| a > 0.0).count())
        }
        Action::Velocity { zone, v } => {
            let cs = cells(sim, zone);
            let (nx, ny) = (sim.nx, sim.ny);
            for &(i, j) in &cs {
                let c = sim.idx(i, j);
                if sim.is_solid(c) {
                    continue;
                }
                // faces intérieures dont l'autre maille est fluide (les bords du domaine restent tels quels)
                if i > 0 && !sim.is_solid(sim.idx(i - 1, j)) {
                    let f = sim.uidx(i, j);
                    sim.u[f] = v[0];
                }
                if i + 1 < nx && !sim.is_solid(sim.idx(i + 1, j)) {
                    let f = sim.uidx(i + 1, j);
                    sim.u[f] = v[0];
                }
                if j > 0 && !sim.is_solid(sim.idx(i, j - 1)) {
                    let f = sim.vidx(i, j);
                    sim.v[f] = v[1];
                }
                if j + 1 < ny && !sim.is_solid(sim.idx(i, j + 1)) {
                    let f = sim.vidx(i, j + 1);
                    sim.v[f] = v[1];
                }
            }
            Ok(cs.len())
        }
        Action::Ignite { zone, temp } => {
            if sim.reg.reactions.is_empty() {
                return Err("allumer : aucune réaction chimique dans ces matériaux".into());
            }
            let cs = cells(sim, zone);
            apply_tool_with(sim, MpTool::Ignite, &cs, &tool_opts(*temp, MpThermal::Passive));
            Ok(cs.len())
        }
        Action::Sink { zone, rate } => effect(sim, zone, MpEffect::Sink { rate: *rate }),
        Action::Emitter { zone, fluid, v, temp } => {
            let phase = find_phase(sim, fluid)?;
            effect(sim, zone, MpEffect::Emitter { vx: v[0], vy: v[1], phase, temp: temp.unwrap_or(t_amb) })
        }
        Action::Grille { zone, hole, open } => effect(sim, zone, MpEffect::Grille { hole: *hole, open: *open }),
        Action::Boundary { side, spec } => {
            let k = *side;
            match spec {
                BoundarySpec::Wall => {
                    sim.bc[k] = Bc::Wall;
                    sim.inlet[k] = None;
                    // une entrée ou une sortie qui se ferme : vitesse normale des faces du bord à zéro tout de suite (le transport du
                    // pas suivant, qui précède la projection, la garderait sinon un pas de plus)
                    let (nx, ny) = (sim.nx, sim.ny);
                    if k < 2 {
                        let i = if k == 0 { 0 } else { nx };
                        for j in 0..ny {
                            let f = sim.uidx(i, j);
                            sim.u[f] = 0.0;
                        }
                    } else {
                        let j = if k == 2 { 0 } else { ny };
                        for i in 0..nx {
                            let f = sim.vidx(i, j);
                            sim.v[f] = 0.0;
                        }
                    }
                }
                BoundarySpec::Open(p) => {
                    sim.bc[k] = Bc::Open { p: *p };
                    sim.inlet[k] = None;
                }
                BoundarySpec::Inlet { v, temp, fluid } => {
                    let phase = match fluid {
                        Some(f) => find_phase(sim, f)?,
                        None => sim.liquid_phases().first().copied().unwrap_or(sim.ambient_phase),
                    };
                    sim.bc[k] = Bc::Wall;
                    sim.inlet[k] = Some(MpInlet { vx: v[0], vy: v[1], phase, temp: temp.unwrap_or(t_amb), offset: 0 });
                }
                BoundarySpec::Outlet { v } => {
                    // une entrée de bord à vitesse sortante : rien n'entre, le fluide des mailles du bord sort à cette vitesse
                    sim.bc[k] = Bc::Wall;
                    sim.inlet[k] = Some(MpInlet { vx: v[0], vy: v[1], phase: sim.ambient_phase, temp: t_amb, offset: 0 });
                }
            }
            Ok(0)
        }
        Action::Field { qty, source, zone, temp } => super::mp_fields::apply_field(sim, qty, source, zone, *temp, base),
        Action::Gravity(g) => {
            sim.gravity = *g;
            Ok(0)
        }
        Action::DtMax(d) => {
            sim.dt_max = *d;
            Ok(0)
        }
        Action::Heat { .. } | Action::Accelerate { .. } | Action::SaveState(_) | Action::SaveImage { .. } | Action::Mark(_) | Action::Pause => Ok(0),
        // (conditions initiales : sa première fois ; le déroulement la répète)
        Action::Periodic { action, .. } => apply_action(sim, action, base),
    }
}

fn effect(sim: &mut MultiphaseSim, zone: &Zone, e: MpEffect) -> Result<usize, String> {
    let cs = zone.cells(sim.nx, sim.ny, sim.dx);
    apply_tool_with(sim, MpTool::Effect(e), &cs, &tool_opts(sim.t_ambient, MpThermal::Passive));
    Ok(cs.len())
}

/// Conditions initiales : le départ du protocole puis ses actions instantanées d'avant le premier « simuler » (ce que la fenêtre
/// montre en mode « Conditions initiales »). Les erreurs portent leur ligne.
pub fn initial_state(proto: &Protocol, base: &Path) -> Result<(MultiphaseSim, Option<MaterialLibrary>, Option<usize>), String> {
    let (mut sim, lib, sc) = build_start(proto, base)?;
    for (s, line) in &proto.steps[..proto.first_advance()] {
        if let Step::Do(a) = s {
            if a.is_instant_edit() {
                apply_action(&mut sim, a, base).map_err(|e| format!("ligne {line} : {e}"))?;
            }
        }
    }
    Ok((sim, lib, sc))
}

// ------------------------------------------------------------------------------------------------ mesures

enum ProbeKind {
    Q(QtyR, Reduce),
    /// valeur interpolée en un point ; `cells` : la maille qui le contient
    Point(QtyR, [f64; 2]),
    Mass(usize),
    Body(usize, BodyQty),
    /// force du fluide sur les mailles solides parmi `cells`, composante 0 (x) ou 1 (y)
    SolidForce(usize),
    /// débit à travers des faces : fluide pur (sinon tous), massique, faces de u (ligne verticale) ou de v ; par face, son indice et
    /// les mailles de part et d'autre (vers −, vers + ; usize::MAX hors du domaine) ; emplacements des faces dans `flux_acc` ; dernier
    /// relevé (t, cumul, débit)
    Flux { phase: Option<usize>, mass: bool, vertical: bool, faces: Vec<[usize; 3]>, slots: Vec<usize>, last: std::cell::Cell<Option<(f64, f64, f64)>> },
}

#[derive(Clone, Copy)]
enum QtyR {
    T,
    P,
    Vx,
    Vy,
    Speed,
    Rho,
    Frac(usize),
    Heat,
}

struct Probe {
    kind: ProbeKind,
    cells: Vec<usize>,
    fluid_only: bool,
}

/// série d'images PNG à intervalles réguliers : `0000.png`, `0001.png`… dans `images-<affichage>/` du dossier des résultats
struct ImageSeries {
    period: f64,
    display: super::mp_view::MpDisplay,
    line: usize,
    /// prochaine image (None : la série n'a pas commencé)
    next: Option<f64>,
    /// instant de la dernière image
    last: Option<f64>,
    count: usize,
}

/// nom de dossier d'un affichage (sans accent ni espace)
fn display_slug(d: super::mp_view::MpDisplay) -> &'static str {
    use super::mp_view::MpDisplay as D;
    match d {
        D::Phases => "phases",
        D::Pressure => "pression",
        D::Temperature => "temperature",
        D::Velocity => "vitesse",
        D::Density => "masse-volumique",
        D::Vapor => "vapeur",
        D::Divergence => "divergence",
        D::Flame => "flamme",
        D::HeatRelease => "chaleur",
    }
}

/// une comparaison d'une condition d'arrêt en cours : relevé, comparaison, seuil, nom et unité du relevé, dernier relevé (t, valeur)
struct CondClause {
    probe: Probe,
    cmp: Cmp,
    value: f64,
    name: String,
    unit: String,
    last: Option<(f64, f64)>,
}

/// condition d'arrêt en cours : comparaisons reliées par « et » (toutes) ou « ou » (`any`) ; `near` : franchissement prévu avant la
/// prochaine série de pas du GPU (il repasse alors au pas unique)
struct Cond {
    clauses: Vec<CondClause>,
    any: bool,
    line: usize,
    near: bool,
}

impl Cond {
    /// « corps 1 y < 0.1 m ou max T partout > 400 K »
    fn text(&self) -> String {
        let parts: Vec<String> = self.clauses.iter().map(|c| format!("{} {} {} {}", c.name, c.cmp.symbol(), fmt_num(c.value), c.unit)).collect();
        parts.join(if self.any { " ou " } else { " et " })
    }
    fn values(&self, sim: &MultiphaseSim) -> Vec<f64> {
        self.clauses.iter().map(|c| c.probe.value(sim)).collect()
    }
    fn holds(&self, xs: &[f64]) -> bool {
        let mut it = self.clauses.iter().zip(xs).map(|(c, &x)| c.cmp.holds(x, c.value));
        if self.any { it.any(|b| b) } else { it.all(|b| b) }
    }
    /// « 0.099 m ; 412 K »
    fn values_text(&self, xs: &[f64]) -> String {
        self.clauses.iter().zip(xs).map(|(c, &x)| format!("{} {}", fmt_num(x), c.unit)).collect::<Vec<_>>().join(" ; ")
    }
}

/// Force du fluide sur un ensemble de mailles solides (N par mètre de profondeur) : la pression de chaque maille fluide voisine, ramenée
/// au centre de la face commune par l'hydrostatique (p + ρ g · (x_face − x_maille)), sur la longueur de la face, poussant vers le
/// solide. Les parois glissent librement : pas de frottement. Les faces entre deux mailles solides et contre le bord du domaine ne
/// reçoivent rien (un solide collé au bord est plaqué contre lui par la pression du fluide de l'autre côté).
pub(crate) fn solid_force(sim: &MultiphaseSim, cells: &[usize]) -> [f64; 2] {
    let (nx, ny, dx) = (sim.nx, sim.ny, sim.dx);
    let (gx, gy) = (sim.gravity[0], sim.gravity[1]);
    let mut f = [0.0f64; 2];
    for &c in cells {
        if !sim.is_solid(c) {
            continue;
        }
        let (i, j) = ((c % nx) as isize, (c / nx) as isize);
        // (di, dj) : position de la maille fluide voisine ; la force sur le solide va d'elle vers lui
        for (di, dj) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
            let (ii, jj) = (i + di, j + dj);
            if ii < 0 || jj < 0 || ii >= nx as isize || jj >= ny as isize {
                continue;
            }
            let n = jj as usize * nx + ii as usize;
            if sim.is_solid(n) {
                continue;
            }
            // centre de la face moins centre de la maille fluide : −(di, dj) dx / 2
            let p = sim.p[n] + sim.rho_mix[n] * (gx * (-(di as f64)) + gy * (-(dj as f64))) * 0.5 * dx;
            f[0] -= p * dx * di as f64;
            f[1] -= p * dx * dj as f64;
        }
    }
    f
}

fn qty_value(sim: &MultiphaseSim, q: QtyR, c: usize) -> f64 {
    let (i, j) = (c % sim.nx, c / sim.nx);
    match q {
        QtyR::T => sim.temp[c],
        QtyR::P => sim.p[c],
        QtyR::Vx => cell_velocity(sim, i, j).0,
        QtyR::Vy => cell_velocity(sim, i, j).1,
        QtyR::Speed => {
            let (a, b) = cell_velocity(sim, i, j);
            a.hypot(b)
        }
        QtyR::Rho => sim.rho_mix[c],
        QtyR::Frac(k) => sim.alpha[k][c],
        QtyR::Heat => sim.heat_release.get(c).copied().unwrap_or(0.0),
    }
}

/// Valeur d'une grandeur au point p (m), interpolée (bilinéaire) entre les quatre points voisins de la grille où elle est calculée :
/// centres des mailles, milieux des faces pour les composantes de vitesse (grille décalée). Au centre d'une maille, sa valeur ; à moins
/// d'une demi-maille d'un bord, la rangée la plus proche (pas d'extrapolation). Les grandeurs du fluide écartent les mailles solides
/// voisines (poids des autres renormalisés) et n'ont pas de valeur (NaN) dans une maille solide (`cell` : la maille du point).
fn point_value(sim: &MultiphaseSim, q: QtyR, p: [f64; 2], cell: usize, fluid_only: bool) -> f64 {
    if fluid_only && sim.is_solid(cell) {
        return f64::NAN;
    }
    let (nx, ny) = (sim.nx, sim.ny);
    let (x, y) = (p[0] / sim.dx, p[1] / sim.dx);
    // grandeur aux points (i + ox, j + oy) mailles, i < ni, j < nj ; `val` : None pour un point écarté
    let interp = |ox: f64, oy: f64, ni: usize, nj: usize, val: &dyn Fn(usize, usize) -> Option<f64>| -> f64 {
        let (i0, i1, fx) = lerp_index(x - ox, ni);
        let (j0, j1, fy) = lerp_index(y - oy, nj);
        let (mut s, mut w) = (0.0, 0.0);
        for (i, j, wk) in [(i0, j0, (1.0 - fx) * (1.0 - fy)), (i1, j0, fx * (1.0 - fy)), (i0, j1, (1.0 - fx) * fy), (i1, j1, fx * fy)] {
            if wk > 0.0 {
                if let Some(v) = val(i, j) {
                    s += wk * v;
                    w += wk;
                }
            }
        }
        if w > 0.0 { s / w } else { f64::NAN }
    };
    let u = || interp(0.0, 0.5, nx + 1, ny, &|i, j| Some(sim.u[sim.uidx(i, j)]));
    let v = || interp(0.5, 0.0, nx, ny + 1, &|i, j| Some(sim.v[sim.vidx(i, j)]));
    match q {
        QtyR::Vx => u(),
        QtyR::Vy => v(),
        QtyR::Speed => u().hypot(v()),
        _ => interp(0.5, 0.5, nx, ny, &|i, j| {
            let c = sim.idx(i, j);
            if fluid_only && sim.is_solid(c) { None } else { Some(qty_value(sim, q, c)) }
        }),
    }
}

/// Débit à travers des faces, par mètre de profondeur : vitesse de chaque face × longueur dx × contenu de la maille amont (celle d'où
/// vient le fluide ; au bord du domaine, la maille intérieure), fraction volumique (débit volumique, m²/s) ou masse volumique partielle
/// (débit massique, kg/(m·s)) d'un fluide pur, ou de tous ; une maille solide ne contient rien. Signe : vers +x ou +y.
fn flux_value(sim: &MultiphaseSim, phase: Option<usize>, mass: bool, vertical: bool, faces: &[[usize; 3]]) -> f64 {
    let np = sim.reg.phases.len();
    let content = |c: usize| -> f64 {
        if sim.is_solid(c) {
            return 0.0;
        }
        let one = |k: usize| if mass { sim.mass[k][c] } else { sim.alpha[k][c] };
        match phase {
            Some(k) => one(k),
            None => (0..np).map(one).sum(),
        }
    };
    let mut q = 0.0;
    for &[f, lo, hi] in faces {
        let vel = if vertical { sim.u[f] } else { sim.v[f] };
        let up = if hi == usize::MAX || (vel > 0.0 && lo != usize::MAX) { lo } else { hi };
        if vel != 0.0 {
            q += vel * content(up);
        }
    }
    q * sim.dx
}

/// Faces d'une ligne de débit (a, b) : verticale (faces de u) ou horizontale (faces de v), sur la ligne de faces la plus proche, celles
/// dont le milieu est entre les extrémités ; par face, son indice et les mailles de part et d'autre (vers −, vers + ; usize::MAX hors du
/// domaine).
fn flux_faces(sim: &MultiphaseSim, a: [f64; 2], b: [f64; 2]) -> Result<(bool, Vec<[usize; 3]>), String> {
    let (nx, ny, dx) = (sim.nx, sim.ny, sim.dx);
    let (lx, ly) = (nx as f64 * dx, ny as f64 * dx);
    let side = |i: usize, n: usize, c: usize| if i < n { c } else { usize::MAX };
    let (vertical, faces): (bool, Vec<[usize; 3]>) = if a[0] == b[0] && a[0] >= -0.5 * dx && a[0] <= lx + 0.5 * dx {
        let i = ((a[0] / dx).round().max(0.0) as usize).min(nx);
        let faces = span(a[1], b[1], ny, dx)
            .map(|j| [sim.uidx(i, j), if i > 0 { sim.idx(i - 1, j) } else { usize::MAX }, side(i, nx, sim.idx(i.min(nx - 1), j))])
            .collect();
        (true, faces)
    } else if a[1] == b[1] && a[0] != b[0] && a[1] >= -0.5 * dx && a[1] <= ly + 0.5 * dx {
        let j = ((a[1] / dx).round().max(0.0) as usize).min(ny);
        let faces = span(a[0], b[0], nx, dx)
            .map(|i| [sim.vidx(i, j), if j > 0 { sim.idx(i, j - 1) } else { usize::MAX }, side(j, ny, sim.idx(i, j.min(ny - 1)))])
            .collect();
        (false, faces)
    } else {
        (true, vec![])
    };
    if faces.is_empty() {
        return Err(format!("la ligne {} {} est hors du domaine", fmt_point(a), fmt_point(b)));
    }
    Ok((vertical, faces))
}

/// rangées (colonnes) dont le centre est entre c1 et c2 (m) ; à défaut, celle du milieu du segment (aucune hors du domaine)
fn span(c1: f64, c2: f64, n: usize, dx: f64) -> std::ops::Range<usize> {
    let (lo, hi) = (c1.min(c2) / dx - 0.5, c1.max(c2) / dx - 0.5);
    let k0 = (lo - 1e-9).ceil().max(0.0) as usize;
    let k1 = (((hi + 1e-9).floor() + 1.0).max(0.0) as usize).min(n);
    if k0 < k1 {
        return k0..k1;
    }
    let m = 0.5 * (lo + hi) + 0.5;
    if m < 0.0 || m > n as f64 {
        return 0..0;
    }
    let k = (m.floor() as usize).min(n - 1);
    k..k + 1
}

/// points voisins d'une position s (en points de la grille, bornée à [0, n − 1]) et poids du second
fn lerp_index(s: f64, n: usize) -> (usize, usize, f64) {
    let s = s.clamp(0.0, (n - 1) as f64);
    let i0 = (s.floor() as usize).min(n - 1);
    let i1 = (i0 + 1).min(n - 1);
    (i0, i1, s - i0 as f64)
}

impl Probe {
    fn value(&self, sim: &MultiphaseSim) -> f64 {
        if let ProbeKind::Flux { phase, mass, vertical, faces, slots, last } = &self.kind {
            // débit moyen depuis le relevé précédent : ce qu'a compté le transport ; au premier relevé, l'estimation instantanée
            let np = sim.reg.phases.len();
            let off = if *mass { np } else { 0 };
            let total: f64 = slots
                .iter()
                .map(|&s| {
                    let b = 2 * s * np + off;
                    match phase {
                        Some(k) => sim.flux_acc[b + k],
                        None => sim.flux_acc[b..b + np].iter().sum(),
                    }
                })
                .sum();
            let t = sim.time;
            return match last.get() {
                Some((t0, a0, _)) if t > t0 => {
                    let q = (total - a0) / (t - t0);
                    last.set(Some((t, total, q)));
                    q
                }
                Some((_, _, q)) => q,
                None => {
                    let q = flux_value(sim, *phase, *mass, *vertical, faces);
                    last.set(Some((t, total, q)));
                    q
                }
            };
        }
        let a2 = sim.dx * sim.dx;
        match self.kind {
            ProbeKind::Flux { .. } => unreachable!(),
            ProbeKind::Mass(k) => self.cells.iter().map(|&c| sim.mass[k][c]).sum::<f64>() * a2,
            ProbeKind::SolidForce(axis) => solid_force(sim, &self.cells)[axis],
            ProbeKind::Body(k, q) => q.value(sim, k),
            ProbeKind::Point(q, p) => point_value(sim, q, p, self.cells[0], self.fluid_only),
            ProbeKind::Q(q, red) => {
                let vals = self.cells.iter().filter(|&&c| !(self.fluid_only && sim.is_solid(c))).map(|&c| qty_value(sim, q, c));
                match red {
                    Reduce::At => self.cells.first().filter(|&&c| !(self.fluid_only && sim.is_solid(c))).map_or(f64::NAN, |&c| qty_value(sim, q, c)),
                    Reduce::Mean => {
                        let (s, n) = vals.fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
                        if n > 0 { s / n as f64 } else { f64::NAN }
                    }
                    Reduce::Max => vals.fold(f64::NAN, |m, v| if m.is_nan() || v > m { v } else { m }),
                    Reduce::Min => vals.fold(f64::NAN, |m, v| if m.is_nan() || v < m { v } else { m }),
                    Reduce::Sum => vals.sum::<f64>() * a2,
                }
            }
        }
    }
}

/// colonnes des statistiques globales (même ordre que `stats_values`)
fn stats_columns(sim: &MultiphaseSim) -> Vec<String> {
    let mut v: Vec<String> = vec!["dt (s)".into(), "itérations pression".into(), "pas refaits".into()];
    for p in &sim.reg.phases {
        v.push(format!("masse {} (kg/m)", p.name));
    }
    for s in [
        "énergie cinétique (J/m)",
        "énergie interne (J/m)",
        "T min (K)",
        "T max (K)",
        "p min (Pa)",
        "p max (Pa)",
        "vitesse max (m/s)",
        "chaleur injectée (J/m)",
    ] {
        v.push(s.into());
    }
    if !sim.reg.reactions.is_empty() {
        v.push("chaleur de réaction (J/m)".into());
        v.push("puissance chimique (W/m)".into());
        v.push("énergie rayonnée (J/m)".into());
    }
    v
}

fn stats_values(sim: &MultiphaseSim) -> Vec<f64> {
    let st = compute_stats(sim);
    let mut v = vec![st.dt, st.pressure_iters as f64, st.rollbacks as f64];
    v.extend(st.phases.iter().map(|p| p.1));
    v.extend([st.kinetic, st.internal, st.t_min, st.t_max, st.p_min, st.p_max, st.speed_max, st.heat_input]);
    if !sim.reg.reactions.is_empty() {
        v.extend([st.chem_heat, st.chem_power, st.radiated]);
    }
    v
}

// ------------------------------------------------------------------------------------------------ déroulement

#[derive(Clone, Debug, PartialEq)]
pub enum RunState {
    Running,
    /// en pause (raison) ; `Runner::resume` repart
    Paused(String),
    Finished,
    Failed(String),
}

/// action à durée en cours : chauffage (cibles) ou accélération (cellules « accélérateur » posées)
/// action répétée en cours : prochaine fois, dernière permise, ligne du protocole, nombre de fois
struct Periodic {
    action: Action,
    period: f64,
    next: f64,
    end: f64,
    line: usize,
    count: usize,
}

struct Timed {
    cells: Vec<usize>,
    /// température cible (chauffage) ou accélération (ax, ay)
    value: [f64; 2],
    end: f64,
    heat: bool,
}

/// Exécution d'un protocole. `advance` fait le travail par tranches de temps réel (la fenêtre reste fluide) ; l'état courant est
/// dans `sim` (sur le GPU : relu à chaque mesure et à chaque action, sinon `sync_host`).
pub struct Runner {
    pub sim: MultiphaseSim,
    pub gpu: Option<MultiphaseGpu>,
    pub proto: Protocol,
    /// dossier du protocole : les fichiers qu'il nomme sont lus par rapport à lui
    pub base: PathBuf,
    /// dossier des résultats (None : rien n'est écrit, tout reste en mémoire)
    pub out_dir: Option<PathBuf>,
    pub library: Option<MaterialLibrary>,
    pub scenario: Option<usize>,
    pc: usize,
    target: Option<f64>,
    /// condition d'arrêt en cours (`simuler jusqu'à ce que …`)
    cond: Option<Cond>,
    timed: Vec<Timed>,
    periodic: Vec<Periodic>,
    /// journal muet (répétitions au-delà de la troisième)
    quiet: bool,
    probes: Vec<Probe>,
    stats: bool,
    /// en-têtes des colonnes des mesures (t, pas, puis les mesures)
    pub columns: Vec<String>,
    /// mesures relevées (une ligne par relevé)
    pub rows: Vec<Vec<f64>>,
    /// journal (actions, avertissements)
    pub log: Vec<String>,
    next_sample: Option<f64>,
    /// séries d'images à intervalles réguliers (`images toutes les …`)
    series: Vec<ImageSeries>,
    pub state: RunState,
    /// fin prévue (temps simulé)
    pub t_end: f64,
    pub t_start: f64,
    /// ligne en cours d'exécution
    pub line: usize,
    csv: Option<std::io::BufWriter<std::fs::File>>,
    wall: Instant,
}

const CSV_SEP: &str = ";";

impl Runner {
    /// Prépare l'exécution : construit le départ, vérifie les noms, choisit le calcul (`gpu` : demande de l'appelant, sinon celle du
    /// protocole, sinon CPU), ouvre le dossier des résultats.
    pub fn new(proto: Protocol, base: &Path, out_dir: Option<PathBuf>, gpu: Option<bool>) -> Result<Runner, String> {
        let (sim, library, scenario) = build_start(&proto, base)?;
        Self::with_start(proto, base, out_dir, gpu, sim, library, scenario)
    }

    /// Comme `new`, à partir d'un état de départ déjà construit (le même que `build_start(&proto, base)`).
    pub fn with_start(
        proto: Protocol,
        base: &Path,
        out_dir: Option<PathBuf>,
        gpu: Option<bool>,
        sim: MultiphaseSim,
        library: Option<MaterialLibrary>,
        scenario: Option<usize>,
    ) -> Result<Runner, String> {
        let errs = check_names(&proto, &sim);
        if !errs.is_empty() {
            return Err(errs.iter().map(|(l, e)| format!("ligne {l} : {e}")).collect::<Vec<_>>().join("\n"));
        }
        let mut r = Runner {
            t_end: proto.planned_end(sim.time),
            t_start: sim.time,
            sim,
            gpu: None,
            base: base.to_path_buf(),
            out_dir,
            library,
            scenario,
            pc: 0,
            target: None,
            cond: None,
            timed: vec![],
            periodic: vec![],
            quiet: false,
            probes: vec![],
            stats: false,
            columns: vec!["t (s)".into(), "pas".into()],
            rows: vec![],
            log: vec![],
            next_sample: None,
            series: vec![],
            state: RunState::Running,
            line: 0,
            csv: None,
            wall: Instant::now(),
            proto,
        };
        r.watch_flux_lines()?;
        r.compile_measures()?;
        r.log_header();
        if gpu.or(r.proto.gpu).unwrap_or(false) {
            r.enable_gpu();
        }
        if let Some(dir) = r.out_dir.clone() {
            std::fs::create_dir_all(&dir).map_err(|e| format!("dossier des résultats « {} » : {e}", dir.display()))?;
            std::fs::write(dir.join("protocole.exp"), &r.proto.source).map_err(|e| format!("copie du protocole : {e}"))?;
            if r.proto.period.is_some() {
                let f = std::fs::File::create(dir.join("mesures.csv")).map_err(|e| format!("mesures.csv : {e}"))?;
                let mut w = std::io::BufWriter::new(f);
                writeln!(w, "{}", r.columns.join(CSV_SEP)).map_err(|e| e.to_string())?;
                r.csv = Some(w);
            }
        }
        Ok(r)
    }

    /// Réserve les emplacements des faces de toutes les lignes de débit du protocole (mesures et conditions d'arrêt), avant la
    /// création du GPU qui en fixe la place.
    fn watch_flux_lines(&mut self) -> Result<(), String> {
        let mut lines = vec![];
        for (m, _) in &self.proto.measures {
            if let Measure::Flux { a, b, .. } = m {
                lines.push((*a, *b));
            }
        }
        for (s, _) in &self.proto.steps {
            if let Step::UntilCond { clauses, .. } = s {
                for c in clauses {
                    if let Measure::Flux { a, b, .. } = &c.measure {
                        lines.push((*a, *b));
                    }
                }
            }
        }
        for (a, b) in lines {
            let (vertical, faces) = flux_faces(&self.sim, a, b)?;
            let idx: Vec<usize> = faces.iter().map(|f| f[0]).collect();
            self.sim.watch_faces(vertical, &idx);
        }
        Ok(())
    }

    fn enable_gpu(&mut self) {
        if let Some(why) = gpu_unsupported_reason(&self.sim) {
            self.note(format!("GPU indisponible pour cette scène ({why}) : calcul sur le CPU"));
            return;
        }
        match MultiphaseGpu::try_new(&self.sim) {
            Ok(g) => {
                self.note(format!("calcul sur le GPU (f32) : {}", g.adapter_name));
                self.gpu = Some(g);
            }
            Err(e) => self.note(format!("GPU indisponible ({e}) : calcul sur le CPU")),
        }
    }

    fn log_header(&mut self) {
        let s = &self.sim;
        let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        self.log.push(format!("fluid-sim {} — expérience lancée (horloge Unix {secs} s)", env!("CARGO_PKG_VERSION")));
        if let Some((n, v)) = &self.proto.variant {
            self.log.push(format!("balayage : {n} = {v}"));
        }
        self.log.push(format!(
            "grille {} × {} de {} m ({} × {} m), {} fluides, {} solides, {} réactions, t initial {} s",
            s.nx,
            s.ny,
            fmt_num(s.dx),
            fmt_num(s.nx as f64 * s.dx),
            fmt_num(s.ny as f64 * s.dx),
            s.reg.phases.len(),
            s.reg.solids.len(),
            s.reg.reactions.len(),
            fmt_num(s.time)
        ));
        self.log.push(format!("calcul CPU (f64), {} fil(s) — résultats indépendants du nombre de fils", super::mp_par::threads()));
    }

    fn note(&mut self, msg: String) {
        if self.quiet {
            return;
        }
        self.log.push(format!("[t = {:.6} s] {msg}", self.sim.time));
    }

    fn compile_measures(&mut self) -> Result<(), String> {
        let ms = self.proto.measures.clone();
        for (m, line) in ms {
            match self.make_probe(&m).map_err(|e| format!("ligne {line} : {e}"))? {
                None => self.stats = true,
                Some((probe, name, unit)) => {
                    self.columns.push(format!("{name} ({unit})"));
                    self.probes.push(probe);
                }
            }
        }
        if self.stats {
            let c = stats_columns(&self.sim);
            self.columns.extend(c);
        }
        Ok(())
    }

    /// Sonde d'une mesure, son nom et son unité (en-tête de colonne « nom (unité) ») ; None pour `stats`.
    fn make_probe(&self, m: &Measure) -> Result<Option<(Probe, String, String)>, String> {
        let (nx, ny, dx) = (self.sim.nx, self.sim.ny, self.sim.dx);
        Ok(Some(match m {
            Measure::Stats => return Ok(None),
            Measure::Probe { q, reduce, zone, label } => {
                let qr = match q {
                    Quantity::T => QtyR::T,
                    Quantity::P => QtyR::P,
                    Quantity::Vx => QtyR::Vx,
                    Quantity::Vy => QtyR::Vy,
                    Quantity::Speed => QtyR::Speed,
                    Quantity::Rho => QtyR::Rho,
                    Quantity::Fraction(f) => QtyR::Frac(find_phase(&self.sim, f)?),
                    Quantity::HeatRelease => QtyR::Heat,
                };
                let cells: Vec<usize> = zone.cells(nx, ny, dx).into_iter().map(|(i, j)| self.sim.idx(i, j)).collect();
                if cells.is_empty() {
                    return Err(format!("la zone « {} » est hors du domaine", zone.label()));
                }
                let red = match reduce {
                    Reduce::At => "",
                    Reduce::Mean => "moyenne ",
                    Reduce::Max => "max ",
                    Reduce::Min => "min ",
                    Reduce::Sum => "somme ",
                };
                let name = label.clone().unwrap_or_else(|| format!("{red}{} {}", q.name(), zone.label()));
                let kind = match (reduce, zone) {
                    (Reduce::At, Zone::Point(p)) => ProbeKind::Point(qr, *p),
                    _ => ProbeKind::Q(qr, *reduce),
                };
                (Probe { kind, cells, fluid_only: q.fluid_only() }, name, q.unit(*reduce).to_string())
            }
            Measure::Mass { fluid, zone, label } => {
                let k = find_phase(&self.sim, fluid)?;
                let cells: Vec<usize> = zone.cells(nx, ny, dx).into_iter().map(|(i, j)| self.sim.idx(i, j)).collect();
                let name = label.clone().unwrap_or_else(|| format!("masse {fluid} {}", zone.label()));
                (Probe { kind: ProbeKind::Mass(k), cells, fluid_only: true }, name, "kg/m".to_string())
            }
            Measure::Body { index, q, label } => {
                let name = label.clone().unwrap_or_else(|| format!("corps {} {}", index + 1, q.name()));
                (Probe { kind: ProbeKind::Body(*index, *q), cells: vec![], fluid_only: false }, name, q.unit().to_string())
            }
            Measure::SolidForce { axis, zone, label } => {
                // toutes les mailles de la zone : les solides sont retenus au relevé (posés avant ou pendant l'expérience)
                let cells: Vec<usize> = zone.cells(nx, ny, dx).into_iter().map(|(i, j)| self.sim.idx(i, j)).collect();
                if cells.is_empty() {
                    return Err(format!("la zone « {} » est hors du domaine", zone.label()));
                }
                let name = label.clone().unwrap_or_else(|| format!("force {} sur {}", ["x", "y"][*axis], zone.label()));
                (Probe { kind: ProbeKind::SolidForce(*axis), cells, fluid_only: false }, name, "N/m".to_string())
            }
            Measure::Flux { fluid, mass, a, b, label } => {
                let phase = match fluid {
                    Some(f) => Some(find_phase(&self.sim, f)?),
                    None => None,
                };
                let (vertical, faces) = flux_faces(&self.sim, *a, *b)?;
                // emplacements des faces, réservés au départ (`watch_flux_lines`)
                let slots_of = if vertical { &self.sim.flux_slot_u } else { &self.sim.flux_slot_v };
                let slots: Vec<usize> = faces.iter().map(|f| slots_of.get(f[0]).map_or(usize::MAX, |&s| s as usize)).collect();
                if slots.iter().any(|&s| s == u32::MAX as usize || s == usize::MAX) {
                    return Err(format!("débit à travers ligne {} {} : faces non suivies (ligne ajoutée après le départ)", fmt_point(*a), fmt_point(*b)));
                }
                let what = match fluid {
                    Some(f) => format!("{f} "),
                    None => String::new(),
                };
                let name = label
                    .clone()
                    .unwrap_or_else(|| format!("débit {}{what}à travers ligne {} {}", if *mass { "massique " } else { "" }, fmt_point(*a), fmt_point(*b)));
                let unit = if *mass { "kg/(m·s)" } else { "m²/s" };
                let kind = ProbeKind::Flux { phase, mass: *mass, vertical, faces, slots, last: std::cell::Cell::new(None) };
                (Probe { kind, cells: vec![], fluid_only: true }, name, unit.to_string())
            }
        }))
    }

    /// Avance l'expérience pendant au plus `budget` de temps réel (au moins un pas ou une action). Retourne l'état.
    pub fn advance(&mut self, budget: Duration) -> &RunState {
        let t0 = Instant::now();
        let mut first = true;
        while self.state == RunState::Running && (first || t0.elapsed() < budget) {
            first = false;
            if let Some(target) = self.target {
                self.sim_step(target);
                continue;
            }
            if self.pc >= self.proto.steps.len() {
                self.finish();
                break;
            }
            let (step, line) = self.proto.steps[self.pc].clone();
            self.line = line;
            match step {
                Step::Do(_) => self.run_actions(),
                Step::Simulate(d) => {
                    self.start_sampling();
                    self.target = Some(self.sim.time + d);
                    self.pc += 1;
                }
                Step::Until(t) => {
                    self.start_sampling();
                    if t > self.sim.time {
                        self.target = Some(t);
                    } else {
                        self.note(format!("ligne {line} : « simuler jusqu'à {} s » déjà atteint", fmt_num(t)));
                    }
                    self.pc += 1;
                }
                Step::UntilCond { clauses, any, limit } => {
                    self.start_sampling();
                    self.pc += 1;
                    let mut parts = vec![];
                    for cl in &clauses {
                        match self.make_probe(&cl.measure) {
                            Ok(Some((probe, name, unit))) => parts.push(CondClause { probe, cmp: cl.cmp, value: cl.value, name, unit, last: None }),
                            Ok(None) => {
                                self.fail(format!("ligne {line} : une condition porte sur un relevé, pas sur « stats »"));
                                return &self.state;
                            }
                            Err(e) => {
                                self.fail(format!("ligne {line} : {e}"));
                                return &self.state;
                            }
                        }
                    }
                    self.sync_host();
                    let mut c = Cond { clauses: parts, any, line, near: false };
                    let xs = c.values(&self.sim);
                    let t = self.sim.time;
                    for (cl, &x) in c.clauses.iter_mut().zip(&xs) {
                        cl.last = Some((t, x));
                    }
                    if c.holds(&xs) {
                        self.note(format!("ligne {line} : {} déjà vrai ({})", c.text(), c.values_text(&xs)));
                    } else {
                        let what = match limit {
                            Some(d) => format!("au plus {} s", fmt_num(d)),
                            None => "sans limite de durée".to_string(),
                        };
                        self.note(format!("ligne {line} : simuler jusqu'à ce que {} ({what}) — {} au départ", c.text(), c.values_text(&xs)));
                        self.target = Some(limit.map_or(f64::INFINITY, |d| self.sim.time + d));
                        self.cond = Some(c);
                    }
                }
            }
        }
        &self.state
    }

    /// Reprend après une pause (ligne « pause » ou pause de la fenêtre).
    pub fn resume(&mut self) {
        if matches!(self.state, RunState::Paused(_)) {
            self.state = RunState::Running;
        }
    }

    pub fn pause(&mut self, why: &str) {
        if self.state == RunState::Running {
            self.state = RunState::Paused(why.into());
        }
    }

    /// Progression (0 à 1) d'après le temps simulé prévu.
    pub fn progress(&self) -> f64 {
        if self.state == RunState::Finished {
            return 1.0;
        }
        let span = self.t_end - self.t_start;
        if span > 0.0 { ((self.sim.time - self.t_start) / span).clamp(0.0, 1.0) } else { 0.0 }
    }

    /// État complet à jour sur l'hôte (GPU résident : relecture si des pas ont été faits).
    pub fn sync_host(&mut self) {
        if let Some(g) = &mut self.gpu {
            g.ensure_host(&mut self.sim);
        }
    }

    fn upload(&mut self) {
        if let Some(g) = &mut self.gpu {
            g.upload(&self.sim);
        }
        if self.gpu.is_some() {
            if let Some(why) = gpu_unsupported_reason(&self.sim) {
                self.gpu = None;
                self.note(format!("calcul repassé sur le CPU : {why}"));
            }
        }
    }

    /// Exécute les actions consécutives à partir de `pc` (une seule relecture / un seul envoi GPU pour le groupe).
    fn run_actions(&mut self) {
        self.sync_host();
        let (mut edited, mut heat) = (false, false);
        while self.pc < self.proto.steps.len() {
            let (step, line) = self.proto.steps[self.pc].clone();
            let Step::Do(a) = step else { break };
            self.line = line;
            self.pc += 1;
            match self.do_action(&a, line) {
                Ok((e, h)) => {
                    edited |= e;
                    heat |= h;
                }
                Err(()) => return,
            }
            if matches!(self.state, RunState::Paused(_)) {
                break;
            }
        }
        self.finish_edits(edited, heat);
    }

    /// après des actions : cibles de chauffage recalculées, état renvoyé au GPU s'il a changé
    fn finish_edits(&mut self, edited: bool, heat: bool) {
        if heat {
            self.apply_heat_targets();
        }
        if edited {
            self.upload();
        } else if heat {
            if let Some(g) = &mut self.gpu {
                g.sync_heat_targets(&self.sim);
            }
        }
    }

    /// Actions répétées dues à l'instant `now` (à `eps` près), dans l'ordre de leurs échéances : (état modifié, chauffages changés).
    fn fire_due(&mut self, now: f64, eps: f64) -> Result<(bool, bool), ()> {
        let (mut edited, mut heat) = (false, false);
        let mut fired = 0usize;
        loop {
            let due = (0..self.periodic.len())
                .filter(|&k| self.periodic[k].next <= now + eps && self.periodic[k].next <= self.periodic[k].end + eps)
                .min_by(|&a, &b| self.periodic[a].next.total_cmp(&self.periodic[b].next));
            let Some(k) = due else { break };
            fired += 1;
            if fired > 10_000 {
                self.fail(format!("ligne {} : action répétée trop souvent pour le pas de temps", self.periodic[k].line));
                return Err(());
            }
            let (action, line, count) = {
                let p = &mut self.periodic[k];
                p.count += 1;
                p.next += p.period;
                (p.action.clone(), p.line, p.count)
            };
            if count == 4 {
                self.note(format!("ligne {line} : répétitions suivantes non notées au journal"));
            }
            self.quiet = count > 3;
            let r = self.do_action(&action, line);
            self.quiet = false;
            let (e, h) = r?;
            edited |= e;
            heat |= h;
        }
        Ok((edited, heat))
    }

    /// Exécute une action à l'instant présent : (état modifié, chauffages changés) ; Err : l'expérience a échoué. `Pause` met
    /// l'expérience en pause ; une action répétée est enregistrée (et faite tout de suite si elle est due).
    fn do_action(&mut self, a: &Action, line: usize) -> Result<(bool, bool), ()> {
        let (mut edited, mut heat) = (false, false);
        let now = self.sim.time;
        {
            match a {
                Action::Pause => {
                    self.note(format!("ligne {line} : pause"));
                    self.state = RunState::Paused(format!("pause du protocole (ligne {line})"));
                }
                Action::Periodic { period, start, end, action } => {
                    let mut next = start.unwrap_or(now);
                    // (début passé : la phase est gardée, première fois à l'échéance suivante)
                    if next < now {
                        next += ((now - next) / period).ceil() * period;
                    }
                    let from = start.map_or(String::new(), |s| format!(" à partir de {} s", fmt_num(s)));
                    let until = end.map_or(String::new(), |e| format!(" jusqu'à {} s", fmt_num(e)));
                    self.note(format!("ligne {line} : toutes les {} s{from}{until} : {}", fmt_num(*period), describe(action)));
                    self.periodic.push(Periodic { action: (**action).clone(), period: *period, next, end: end.unwrap_or(f64::INFINITY), line, count: 0 });
                    let eps = 1e-12 * (1.0 + now.abs());
                    let (e, h) = self.fire_due(now, eps)?;
                    edited |= e;
                    heat |= h;
                }
                Action::Mark(t) => self.note(format!("ligne {line} : repère « {t} »")),
                Action::SaveState(name) => self.save_state(name, line),
                Action::SaveImage { name, display } => self.save_image(name, *display, line),
                Action::Heat { zone, temp, duration } => {
                    let cells = idx_cells(&self.sim, zone);
                    self.note(format!("ligne {line} : chauffer {} à {} K{} — {} mailles", zone.label(), fmt_num(*temp), dur(*duration), cells.len()));
                    self.timed.push(Timed { cells, value: [*temp, 0.0], end: duration.map_or(f64::INFINITY, |d| now + d), heat: true });
                    heat = true;
                }
                Action::Accelerate { zone, a, duration } => {
                    let mut cells = vec![];
                    let mut skipped = 0;
                    for c in idx_cells(&self.sim, zone) {
                        if self.sim.is_solid(c) || self.sim.special[c].is_some() {
                            skipped += 1;
                            continue;
                        }
                        self.sim.set_effect(c, Some(MpEffect::Accelerator { ax: a[0], ay: a[1] }));
                        cells.push(c);
                    }
                    self.sim.refresh();
                    self.note(format!(
                        "ligne {line} : accélérer {} de {} m/s²{} — {} mailles{}",
                        zone.label(),
                        fmt_point(*a),
                        dur(*duration),
                        cells.len(),
                        if skipped > 0 { format!(" ({skipped} solides ou cellules spéciales laissées)") } else { String::new() }
                    ));
                    if let Some(d) = duration {
                        self.timed.push(Timed { cells, value: *a, end: now + d, heat: false });
                    }
                    edited = true;
                }
                other => match apply_action(&mut self.sim, other, &self.base.clone()) {
                    Ok(n) => {
                        let what = describe(other);
                        let cells = if other.zone().is_some() { format!(" — {n} mailles") } else { String::new() };
                        self.note(format!("ligne {line} : {what}{cells}"));
                        edited = true;
                    }
                    Err(e) => {
                        self.fail(format!("ligne {line} : {e}"));
                        return Err(());
                    }
                },
            }
        }
        Ok((edited, heat))
    }

    /// cibles de chauffage = celles des chauffages en cours (le dernier posé l'emporte sur une maille commune)
    fn apply_heat_targets(&mut self) {
        self.sim.clear_heat_targets();
        for t in self.timed.iter().filter(|t| t.heat) {
            for &c in &t.cells {
                self.sim.set_heat_target(c, t.value[0]);
            }
        }
    }

    fn save_state(&mut self, name: &str, line: usize) {
        let Some(dir) = self.out_dir.clone() else {
            self.note(format!("ligne {line} : sauver état « {name} » — pas de dossier de résultats, rien d'écrit"));
            return;
        };
        let path = dir.join(format!("{name}.mps"));
        match state_to_bytes(&self.sim, self.library.as_ref(), self.scenario, None).and_then(|b| std::fs::write(&path, b).map_err(|e| e.to_string())) {
            Ok(()) => self.note(format!("ligne {line} : état enregistré : {}", path.display())),
            Err(e) => self.fail(format!("ligne {line} : sauver état : {e}")),
        }
    }

    /// Image PNG d'un champ (rendu de la fenêtre : `mp_view::render_rgba`, échelle ΔT 100 K, pression automatique), agrandie d'un
    /// facteur entier (au plus 16) jusqu'à ~800 pixels de côté.
    fn save_image(&mut self, name: &str, display: super::mp_view::MpDisplay, line: usize) {
        let Some(dir) = self.out_dir.clone() else {
            self.note(format!("ligne {line} : sauver image « {name} » — pas de dossier de résultats, rien d'écrit"));
            return;
        };
        let path = dir.join(format!("{name}.png"));
        match self.write_png(&path, display) {
            Ok((w, h)) => self.note(format!("ligne {line} : image {} ({w} × {h}) : {}", display.name().to_lowercase(), path.display())),
            Err(e) => self.fail(format!("ligne {line} : sauver image : {e}")),
        }
    }

    /// Écrit l'image PNG d'un champ (dimensions en retour).
    fn write_png(&mut self, path: &Path, display: super::mp_view::MpDisplay) -> Result<(usize, usize), String> {
        self.sync_host();
        let (nx, ny) = (self.sim.nx, self.sim.ny);
        let mut rgba = vec![];
        super::mp_view::render_rgba(&self.sim, super::mp_view::RenderOpts::new(display, 100.0, 0.0), &mut rgba);
        let k = (800 / nx.max(ny)).clamp(1, 16);
        let (w, h) = (nx * k, ny * k);
        let mut big = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let (s, d) = (((y / k) * nx + x / k) * 4, (y * w + x) * 4);
                big[d..d + 4].copy_from_slice(&rgba[s..s + 4]);
            }
        }
        let f = std::fs::File::create(path).map_err(|e| e.to_string())?;
        let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w as u32, h as u32);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut wr = enc.write_header().map_err(|e| e.to_string())?;
        wr.write_image_data(&big).map_err(|e| e.to_string())?;
        Ok((w, h))
    }

    fn fail(&mut self, msg: String) {
        self.quiet = false;
        self.note(format!("ERREUR : {msg}"));
        self.state = RunState::Failed(msg);
        self.close_files();
    }

    /// fin des actions à durée échues (temps courant compris)
    fn expire(&mut self, now: f64, eps: f64) {
        if !self.timed.iter().any(|t| t.end <= now + eps) {
            return;
        }
        // les cibles de chauffage ne vivent que sur l'hôte (envoyées seules) ; retirer des accélérateurs modifie l'état : relecture
        if self.timed.iter().any(|t| !t.heat && t.end <= now + eps) {
            self.sync_host();
        }
        let (mut heat, mut acc) = (false, false);
        let mut keep = vec![];
        for t in std::mem::take(&mut self.timed) {
            if t.end > now + eps {
                keep.push(t);
                continue;
            }
            if t.heat {
                heat = true;
            } else {
                for &c in &t.cells {
                    if matches!(self.sim.special[c], Some(MpEffect::Accelerator { ax, ay }) if ax == t.value[0] && ay == t.value[1]) {
                        self.sim.set_effect(c, None);
                    }
                }
                acc = true;
            }
        }
        self.timed = keep;
        let what = match (heat, acc) {
            (true, true) => "fin d'un chauffage et d'une accélération",
            (true, false) => "fin d'un chauffage",
            _ => "fin d'une accélération",
        };
        self.note(what.into());
        if heat {
            self.apply_heat_targets();
        }
        if acc {
            self.sim.refresh();
            self.upload();
        } else if let Some(g) = &mut self.gpu {
            g.sync_heat_targets(&self.sim);
        }
    }

    /// Un pas (ou une série de pas sur le GPU) vers `target`, sans dépasser la prochaine échéance.
    fn sim_step(&mut self, target: f64) {
        let now = self.sim.time;
        let pred = match self.gpu {
            Some(_) => self.sim.last_dt.max(1e-9).min(self.sim.dt_max),
            None => self.sim.choose_dt(),
        };
        // une échéance est atteinte à 1e-6 pas près (pas de pas minuscule pour un reste d'arrondi)
        let eps = (1e-12 * (1.0 + now.abs())).max(1e-6 * pred);
        self.expire(now, eps);
        // actions répétées dues, à l'heure pile (les pas s'arrêtent sur leurs échéances)
        if self.periodic.iter().any(|p| p.next <= now + eps && p.next <= p.end + eps) {
            self.sync_host();
            match self.fire_due(now, eps) {
                Ok((e, h)) => self.finish_edits(e, h),
                Err(()) => return,
            }
            if self.state != RunState::Running {
                return;
            }
        }
        if now >= target - eps {
            self.target = None;
            if let Some(c) = self.cond.take() {
                self.sync_host();
                let xs = c.values(&self.sim);
                self.note(format!("ligne {} : {} non atteint au bout du temps permis ({})", c.line, c.text(), c.values_text(&xs)));
            }
            return;
        }
        let next = self
            .timed
            .iter()
            .map(|t| t.end)
            .chain(self.periodic.iter().filter(|p| p.next <= p.end).map(|p| p.next))
            .fold(target, f64::min);
        let remaining = next - now;
        let dt_max0 = self.sim.dt_max;
        let cap = if remaining <= pred * (1.0 + 1e-9) {
            remaining
        } else if remaining < 2.0 * pred {
            0.5 * remaining
        } else {
            f64::INFINITY
        };
        match &mut self.gpu {
            Some(g) => {
                // (condition d'arrêt près de son seuil : un pas à la fois, pour s'arrêter au bon pas)
                let near = self.cond.as_ref().is_some_and(|c| c.near);
                let k = if cap.is_infinite() && !near { ((remaining / pred) * 0.5).floor().clamp(1.0, 8.0) as usize } else { 1 };
                if k > 1 {
                    // série de k pas : bornés pour que la série s'arrête avant l'échéance
                    self.sim.dt_max = dt_max0.min(remaining / (k as f64 + 1.0));
                    g.step_batch(&mut self.sim, k);
                } else {
                    self.sim.dt_max = dt_max0.min(cap);
                    g.step(&mut self.sim);
                }
            }
            None => {
                self.sim.dt_max = dt_max0.min(cap);
                self.sim.step();
            }
        }
        self.sim.dt_max = dt_max0;
        if self.sim.diverged {
            self.fail(format!("divergence numérique à t = {:.6} s (même à dt/1024, le pas donne un état invalide)", self.sim.time));
            return;
        }
        if self.cond.is_some() {
            self.sync_host();
            let (t, dt) = (self.sim.time, self.sim.last_dt);
            let xs = self.cond.as_ref().unwrap().values(&self.sim);
            if self.cond.as_ref().unwrap().holds(&xs) {
                let c = self.cond.take().unwrap();
                self.note(format!("ligne {} : {} atteint ({})", c.line, c.text(), c.values_text(&xs)));
                self.target = None;
            } else {
                // tendance des deux derniers relevés : franchissement prévu dans moins de deux séries de 8 pas ? « ou » : celui d'une
                // comparaison encore fausse ; « et » : celui de toutes les comparaisons encore fausses
                let c = self.cond.as_mut().unwrap();
                let any = c.any;
                let mut near = !any;
                for (cl, &x) in c.clauses.iter_mut().zip(&xs) {
                    if !cl.cmp.holds(x, cl.value) {
                        let close = match cl.last {
                            Some((t0, x0)) if t > t0 && x != x0 => {
                                let eta = (cl.value - x) / ((x - x0) / (t - t0));
                                eta >= 0.0 && eta < 16.0 * dt
                            }
                            _ => false,
                        };
                        if any {
                            near |= close;
                        } else {
                            near &= close;
                        }
                    }
                    cl.last = Some((t, x));
                }
                c.near = near;
            }
        }
        for k in 0..self.series.len() {
            let Some(next) = self.series[k].next else { continue };
            if self.sim.time >= next - eps {
                self.series_frame(k);
                let (t, s) = (self.sim.time, &mut self.series[k]);
                let mut n = next;
                while n <= t + eps {
                    n += s.period;
                }
                s.next = Some(n);
            }
        }
        if let Some(ns) = self.next_sample {
            if self.sim.time >= ns - eps {
                self.sample();
                let period = self.proto.period.unwrap_or(f64::INFINITY);
                let mut n = ns;
                while n <= self.sim.time + eps {
                    n += period;
                }
                self.next_sample = Some(n);
            }
        }
    }

    fn start_sampling(&mut self) {
        if self.next_sample.is_none() {
            if let Some(p) = self.proto.period {
                self.sample();
                self.next_sample = Some(self.sim.time + p);
            }
        }
        if self.series.is_empty() && !self.proto.images.is_empty() {
            if self.out_dir.is_none() {
                self.note("images à intervalles réguliers : pas de dossier de résultats, rien d'écrit".into());
            }
            self.series = self
                .proto
                .images
                .iter()
                .map(|&(period, display, line)| ImageSeries { period, display, line, next: None, last: None, count: 0 })
                .collect();
        }
        for k in 0..self.series.len() {
            if self.series[k].next.is_none() {
                self.series_frame(k);
                let s = &mut self.series[k];
                s.next = Some(self.sim.time + s.period);
            }
        }
    }

    /// image suivante de la série k (dans `images-<affichage>/` du dossier des résultats)
    fn series_frame(&mut self, k: usize) {
        let Some(dir) = self.out_dir.clone() else { return };
        let (display, count, line) = (self.series[k].display, self.series[k].count, self.series[k].line);
        let dir = dir.join(format!("images-{}", display_slug(display)));
        if count == 0 {
            if let Err(e) = std::fs::create_dir_all(&dir) {
                self.fail(format!("ligne {line} : images : {e}"));
                return;
            }
        }
        match self.write_png(&dir.join(format!("{count:04}.png")), display) {
            Ok(_) => {
                let t = self.sim.time;
                let s = &mut self.series[k];
                s.count += 1;
                s.last = Some(t);
            }
            Err(e) => self.fail(format!("ligne {line} : images : {e}")),
        }
    }

    fn sample(&mut self) {
        self.sync_host();
        let mut row = vec![self.sim.time, self.sim.steps_done() as f64];
        for p in &self.probes {
            row.push(p.value(&self.sim));
        }
        if self.stats {
            row.extend(stats_values(&self.sim));
        }
        if let Some(w) = &mut self.csv {
            let cell = |v: &f64| {
                if !v.is_finite() {
                    String::new()
                } else if v.fract() == 0.0 && v.abs() < 1e15 {
                    format!("{}", *v as i64)
                } else {
                    format!("{v:.9e}")
                }
            };
            let line = row.iter().map(cell).collect::<Vec<_>>().join(CSV_SEP);
            let _ = writeln!(w, "{line}");
        }
        self.rows.push(row);
    }

    fn finish(&mut self) {
        for k in 0..self.periodic.len() {
            let p = &self.periodic[k];
            let msg = format!("ligne {} : toutes les {} s — {} fois", p.line, fmt_num(p.period), p.count);
            self.note(msg);
        }
        if self.proto.period.is_some() {
            let last = self.rows.last().map(|r| r[0]);
            if last.map_or(true, |t| self.sim.time > t + 1e-12 * (1.0 + t.abs())) {
                self.sample();
            }
        }
        for k in 0..self.series.len() {
            let t = self.sim.time;
            if self.series[k].next.is_some() && self.series[k].last.map_or(true, |l| t > l + 1e-12 * (1.0 + l.abs())) {
                self.series_frame(k);
            }
            let s = &self.series[k];
            if self.out_dir.is_some() {
                let msg = format!(
                    "ligne {} : {} images ({}) toutes les {} s dans images-{}/",
                    s.line,
                    s.count,
                    s.display.name().to_lowercase(),
                    fmt_num(s.period),
                    display_slug(s.display)
                );
                self.note(msg);
            }
        }
        self.sync_host();
        let wall = self.wall.elapsed().as_secs_f64();
        self.note(format!("fin de l'expérience : {} pas, {:.1} s de calcul", self.sim.steps_done(), wall));
        self.state = RunState::Finished;
        self.close_files();
    }

    fn close_files(&mut self) {
        if let Some(mut w) = self.csv.take() {
            let _ = w.flush();
        }
        if let Some(dir) = &self.out_dir {
            let _ = std::fs::write(dir.join("journal.txt"), self.log.join("\n") + "\n");
        }
    }

    /// Arrête l'expérience (fenêtre) : les mesures déjà relevées et le journal sont écrits.
    pub fn stop(&mut self) {
        if matches!(self.state, RunState::Running | RunState::Paused(_)) {
            self.note("expérience arrêtée avant la fin".into());
            self.state = RunState::Failed("arrêtée".into());
            self.close_files();
        }
    }
}

fn dur(d: Option<f64>) -> String {
    d.map_or(String::new(), |d| format!(" pendant {} s", fmt_num(d)))
}

/// texte d'une action instantanée pour le journal
fn describe(a: &Action) -> String {
    match a {
        Action::Fill { zone, fluid, temp } => format!("remplir {} avec « {fluid} »{}", zone.label(), temp.map_or(String::new(), |t| format!(" à {} K", fmt_num(t)))),
        Action::SetTemp { zone, temp } => format!("température {} à {} K", zone.label(), fmt_num(*temp)),
        Action::Solid { zone, material, .. } => format!("solide « {material} » {}", zone.label()),
        Action::Clear { zone } => format!("effacer {}", zone.label()),
        Action::Body { shape, material, angle, motion, temp } => {
            let f = match shape {
                BodySpec::Disc { c, r } => format!("disque {} rayon {}", fmt_point(*c), fmt_num(*r)),
                BodySpec::Rect { a, b } => format!("rect {} {}", fmt_point(*a), fmt_point(*b)),
                BodySpec::Polygon(p) => format!("polygone de {} sommets", p.len()),
            };
            let a = if *angle != 0.0 { format!(", tourné de {}°", fmt_num(angle.to_degrees())) } else { String::new() };
            let a = match motion.rho {
                Some(r) => format!("{a}, masse volumique {} kg/m³", fmt_num(r)),
                None => a,
            };
            let a = match temp {
                Some(t) => format!("{a}, à {} K", fmt_num(*t)),
                None => a,
            };
            if motion.free {
                let m = motion;
                let mut parts = vec![];
                let axes: Vec<&str> = ["x", "y", "rotation"].iter().zip(m.locked).filter(|(_, l)| *l).map(|(n, _)| *n).collect();
                if !axes.is_empty() {
                    parts.push(format!("bloqué en {}", axes.join(", ")));
                }
                if let Some(q) = m.pivot {
                    parts.push(format!("pivot {}", fmt_point(q)));
                }
                if m.v != [0.0, 0.0] || m.omega != 0.0 {
                    parts.push(format!("vitesse de départ {} m/s, {} °/s", fmt_point(m.v), fmt_num(m.omega.to_degrees())));
                }
                if let Some(f) = &m.force {
                    parts.push(format!("force « {} », « {} » N/m", f[0], f[1]));
                }
                if let Some(c) = &m.torque {
                    parts.push(format!("couple « {c} » N"));
                }
                let p = if parts.is_empty() { String::new() } else { format!(" ({})", parts.join(", ")) };
                format!("corps libre « {material} » {f}{a}{p}")
            } else if motion.is_fixed() {
                format!("corps fixe « {material} » {f}{a}")
            } else {
                let m = motion;
                let mut parts = vec![];
                if m.v != [0.0, 0.0] {
                    parts.push(format!("vitesse {} m/s", fmt_point(m.v)));
                }
                if m.omega != 0.0 {
                    parts.push(format!("rotation {} °/s", fmt_num(m.omega.to_degrees())));
                }
                if m.osc_v != [0.0, 0.0] || m.osc_omega != 0.0 {
                    parts.push(format!("oscillation {} m/s, {} °/s sur {} s", fmt_point(m.osc_v), fmt_num(m.osc_omega.to_degrees()), fmt_num(m.period)));
                }
                use super::mp_bodies::{LinLaw, RotLaw};
                match &m.lin_law {
                    Some(LinLaw::Position(f)) => parts.push(format!("position « {} », « {} » m", f[0], f[1])),
                    Some(LinLaw::Velocity(f)) => parts.push(format!("vitesse « {} », « {} » m/s", f[0], f[1])),
                    Some(LinLaw::Acceleration(f)) => parts.push(format!("accélération « {} », « {} » m/s²", f[0], f[1])),
                    None => {}
                }
                match &m.rot_law {
                    Some(RotLaw::Angle(f)) => parts.push(format!("angle « {f} » °")),
                    Some(RotLaw::Rate(f)) => parts.push(format!("rotation « {f} » °/s")),
                    None => {}
                }
                if let Some(q) = m.pivot {
                    parts.push(format!("pivot {}", fmt_point(q)));
                }
                format!("corps « {material} » {f}{a}, mouvement imposé : {}", parts.join(", "))
            }
        }
        Action::Periodic { period, action, .. } => format!("toutes les {} s : {}", fmt_num(*period), describe(action)),
        Action::Velocity { zone, v } => format!("vitesse {} {} m/s", zone.label(), fmt_point(*v)),
        Action::Ignite { zone, temp } => format!("allumer {} à {} K", zone.label(), fmt_num(*temp)),
        Action::Sink { zone, rate } => format!("puits {} ({} m/s)", zone.label(), fmt_num(*rate)),
        Action::Emitter { zone, fluid, .. } => format!("émetteur de « {fluid} » {}", zone.label()),
        Action::Grille { zone, .. } => format!("grille {}", zone.label()),
        Action::Boundary { side, spec } => {
            let s = ["gauche", "droite", "bas", "haut"][*side];
            match spec {
                BoundarySpec::Wall => format!("bord {s} : mur"),
                BoundarySpec::Open(p) => format!("bord {s} : ouvert à {} Pa", fmt_num(*p)),
                BoundarySpec::Inlet { v, .. } => format!("bord {s} : entrée à {} m/s", fmt_point(*v)),
                BoundarySpec::Outlet { v } => format!("bord {s} : sortie à {} m/s", fmt_point(*v)),
            }
        }
        Action::Field { qty, source, zone, .. } => {
            let q = match qty {
                FieldQty::T => "T".to_string(),
                FieldQty::Vx => "vx".into(),
                FieldQty::Vy => "vy".into(),
                FieldQty::Fraction(f) => format!("fraction « {f} »"),
                FieldQty::Solid(m) => format!("solide « {m} »"),
            };
            let s = match source {
                FieldSource::File { path, .. } => format!("fichier « {path} »"),
                FieldSource::Formula(f) => format!("formule « {f} »"),
            };
            format!("champ {q} ({s}) {}", zone.label())
        }
        Action::Gravity(g) => format!("gravité {} m/s²", fmt_point(*g)),
        Action::DtMax(d) => format!("pas max {} s", fmt_num(*d)),
        _ => format!("{a:?}"),
    }
}

// ------------------------------------------------------------------------------------------------ ligne de commande

const USAGE: &str = "usage : fluid-sim experience <protocole.exp> [--gpu | --cpu] [--sortie <dossier>] [--fils <n>]\n\
  Exécute le protocole sans fenêtre ; résultats dans <dossier>/<nom du protocole>/ (par défaut : résultats/ à côté du protocole).\n\
  Format du protocole : voir docs/EXPERIENCES.md.";

/// Dossier des résultats d'un protocole : <racine>/<nom>[/<variante>].
pub fn results_dir(root: &Path, stem: &str, variant: &Option<(String, String)>) -> PathBuf {
    let mut d = root.join(stem);
    if let Some((n, v)) = variant {
        let clean: String = format!("{n}={v}").chars().map(|c| if c.is_alphanumeric() || "=.-_+".contains(c) { c } else { '_' }).collect();
        d = d.join(clean);
    }
    d
}

/// `fluid-sim experience …` : retourne le code de sortie (0 = toutes les variantes terminées).
pub fn run_cli(args: &[String]) -> i32 {
    let (mut file, mut gpu, mut out, mut threads) = (None::<PathBuf>, None, None::<PathBuf>, None::<usize>);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--gpu" => gpu = Some(true),
            "--cpu" => gpu = Some(false),
            "--sortie" | "-o" => out = it.next().map(PathBuf::from),
            "--fils" => threads = it.next().and_then(|v| v.parse().ok()),
            "-h" | "--help" | "--aide" => {
                println!("{USAGE}");
                return 0;
            }
            s if file.is_none() => file = Some(PathBuf::from(s)),
            s => {
                eprintln!("argument inattendu « {s} »\n{USAGE}");
                return 2;
            }
        }
    }
    let Some(file) = file else {
        eprintln!("{USAGE}");
        return 2;
    };
    if let Some(n) = threads {
        super::mp_par::set_auto(false);
        super::mp_par::set_threads(n);
    }
    let text = match std::fs::read_to_string(&file) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{} : {e}", file.display());
            return 1;
        }
    };
    let protos = match super::mp_script::parse_all(&text) {
        Ok(p) => p,
        Err(errs) => {
            for e in errs {
                eprintln!("{} : {e}", file.display());
            }
            return 1;
        }
    };
    let base = file.parent().map(Path::to_path_buf).unwrap_or_default();
    let root = out.unwrap_or_else(|| base.join("résultats"));
    let stem = file.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "experience".into());
    let mut code = 0;
    let n = protos.len();
    for (k, proto) in protos.into_iter().enumerate() {
        let dir = results_dir(&root, &stem, &proto.variant);
        if let Some((nm, v)) = &proto.variant {
            println!("— variante {}/{n} : {nm} = {v}", k + 1);
        }
        let mut r = match Runner::new(proto, &base, Some(dir.clone()), gpu) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{e}");
                code = 1;
                continue;
            }
        };
        for l in &r.log {
            println!("{l}");
        }
        let mut shown = r.log.len();
        let mut last_print = Instant::now();
        loop {
            let st = r.advance(Duration::from_millis(500)).clone();
            for l in &r.log[shown..] {
                println!("{l}");
            }
            shown = r.log.len();
            match st {
                RunState::Running => {}
                RunState::Paused(why) => {
                    println!("({why} : ignorée en ligne de commande)");
                    r.resume();
                }
                RunState::Finished => break,
                RunState::Failed(e) => {
                    eprintln!("échec : {e}");
                    code = 1;
                    break;
                }
            }
            if last_print.elapsed() >= Duration::from_secs(2) {
                last_print = Instant::now();
                println!("[{:5.1} %] t = {:.6} s / {:.6} s — ligne {} — {} pas", 100.0 * r.progress(), r.sim.time, r.t_end, r.line, r.sim.steps_done());
            }
        }
        println!("résultats : {}", dir.display());
    }
    code
}
