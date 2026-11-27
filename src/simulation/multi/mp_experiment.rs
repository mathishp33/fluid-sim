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

/// « 0.1 m », ou « 3 » sans unité (variable)
fn with_unit(x: f64, unit: &str) -> String {
    if unit.is_empty() { fmt_num(x) } else { format!("{} {unit}", fmt_num(x)) }
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
    /// position, vitesse ou force du fluide sur un corps rigide (par son rang dans l'ordre de pose, ou par son nom)
    Body { who: Who, q: BodyQty, label: Option<String> },
    /// force du fluide (composante x : 0, y : 1) sur les mailles solides de la zone, par mètre de profondeur
    SolidForce { axis: usize, zone: Zone, label: Option<String> },
    /// débit volumique (m²/s) ou massique (`mass`, kg/(m·s)) d'un fluide pur, ou de tous, à travers le segment horizontal ou vertical
    /// (a, b), le long des faces des mailles ; positif vers +x (segment vertical) ou +y (horizontal)
    Flux { fluid: Option<String>, mass: bool, a: [f64; 2], b: [f64; 2], label: Option<String> },
    /// temps simulé (s), surtout pour les conditions (`tant que temps < 2 s`, `quand temps > 1 s et …`)
    Time { label: Option<String> },
    /// variable du protocole (indice dans `Protocol::vars`)
    Var { var: usize, label: Option<String> },
    /// grandeur d'un vilebrequin (par son rang dans l'ordre de pose, ou par son nom)
    Crank { who: Who, q: CrankQty, label: Option<String> },
}

/// Corps ou vilebrequin d'un relevé : son rang dans l'ordre de pose (à partir de 0 ; « corps 2 » est le rang 1) ou son nom (option
/// `nom "X"` de sa ligne) — un composant désigne ainsi ses pièces sans savoir combien d'autres ont été posées avant lui.
#[derive(Clone, Debug, PartialEq)]
pub enum Who {
    Rank(usize),
    Name(String),
}

impl Who {
    /// dans le nom d'une colonne : « 2 » ou « gauche »
    pub fn label(&self) -> String {
        match self {
            Who::Rank(k) => format!("{}", k + 1),
            Who::Name(n) => n.clone(),
        }
    }
    /// indice parmi les objets posés (leurs noms, dans l'ordre de pose) ; usize::MAX s'il n'y est pas (relevé vide)
    pub fn find<'a>(&self, mut names: impl Iterator<Item = &'a str>) -> usize {
        match self {
            Who::Rank(k) => *k,
            Who::Name(n) => {
                let n = norm(n);
                names.position(|m| norm(m) == n).unwrap_or(usize::MAX)
            }
        }
    }
}

/// Force mesurée sur un corps : celle du fluide (pression et frottement), des contacts, d'une loi en formule, ou leur somme avec
/// le poids (la force nette, M a pour un corps libre)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ForcePart {
    Fluid,
    Contact,
    Law,
    Total,
}

impl ForcePart {
    fn suffix(self) -> &'static str {
        match self {
            ForcePart::Fluid => "",
            ForcePart::Contact => " contact",
            ForcePart::Law => " loi",
            ForcePart::Total => " total",
        }
    }
}

/// Grandeur mesurée sur un vilebrequin : angle depuis le point mort haut (°, 0 à 360), tours depuis le départ, rotation (°/s), couple
/// des forces du fluide et des poids ramené au vilebrequin (N, par mètre de profondeur), puissance qu'il lui donne (W/m)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CrankQty {
    Angle,
    Turns,
    Rotation,
    Torque,
    Power,
}

impl CrankQty {
    fn name(self) -> &'static str {
        match self {
            CrankQty::Angle => "angle",
            CrankQty::Turns => "tours",
            CrankQty::Rotation => "rotation",
            CrankQty::Torque => "couple",
            CrankQty::Power => "puissance",
        }
    }
    fn unit(self) -> &'static str {
        match self {
            CrankQty::Angle => "°",
            CrankQty::Turns => "tr",
            CrankQty::Rotation => "°/s",
            CrankQty::Torque => "N",
            CrankQty::Power => "W/m",
        }
    }
    fn value(self, sim: &MultiphaseSim, k: usize) -> f64 {
        let Some(c) = sim.cranks.get(k) else { return f64::NAN };
        match self {
            CrankQty::Angle => c.angle_deg(),
            CrankQty::Turns => c.theta / std::f64::consts::TAU,
            CrankQty::Rotation => c.omega.to_degrees(),
            CrankQty::Torque => c.torque,
            CrankQty::Power => c.torque * c.omega,
        }
    }
}

/// vilebrequin, bielle et piston, et peut-être leur cylindre : `vilebrequin (x, y) rayon r bielle l …`
#[derive(Clone, Debug, PartialEq)]
pub struct CrankSpec {
    pub pivot: [f64; 2],
    pub r: f64,
    pub l: f64,
    /// direction de l'axe du cylindre depuis l'axe du vilebrequin (°)
    pub axis_deg: f64,
    /// alésage (largeur du piston ; 2r par défaut) et hauteur du piston (r par défaut), m
    pub bore: Option<f64>,
    pub piston_h: Option<f64>,
    /// angle de départ depuis le point mort haut (°)
    pub angle_deg: f64,
    /// cylindre à poser : (longueur de la chambre au point mort haut, épaisseur des parois), m
    pub cylinder: Option<(Option<f64>, Option<f64>)>,
    pub material: String,
    /// libre (moteur) ; sinon mené à `omega`
    pub free: bool,
    /// rotation imposée, ou de départ (rad/s)
    pub omega: f64,
    /// inertie du volant (kg·m par mètre de profondeur) et couple appliqué (formule en t, angle, tours, rotation ; N)
    pub inertia: f64,
    pub load: Option<String>,
    pub temp: Option<f64>,
    /// nom (`nom "X"`) : relevés `vilebrequin "X" …`, pièces « X.manivelle », « X.bielle », « X.piston »
    pub name: Option<String>,
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
    Fx(ForcePart),
    Fy(ForcePart),
    Torque(ForcePart),
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
            BodyQty::Fx(_) => "fx",
            BodyQty::Fy(_) => "fy",
            BodyQty::Torque(_) => "couple",
            BodyQty::Temp => "température",
        }
    }
    fn unit(&self) -> &'static str {
        match self {
            BodyQty::X | BodyQty::Y => "m",
            BodyQty::Angle => "°",
            BodyQty::Vx | BodyQty::Vy => "m/s",
            BodyQty::Rotation => "°/s",
            BodyQty::Fx(_) | BodyQty::Fy(_) => "N/m",
            BodyQty::Torque(_) => "N",
            BodyQty::Temp => "K",
        }
    }
    /// valeur sur le corps k (NaN s'il n'existe pas) ; forces : celles de la dernière projection (couple autour du centre de masse, par
    /// mètre de profondeur)
    pub(crate) fn value(&self, sim: &MultiphaseSim, k: usize) -> f64 {
        let Some(b) = sim.bodies.get(k) else { return f64::NAN };
        let part = |p: ForcePart| -> [f64; 3] {
            let fluid = sim.body_force.get(k).copied().unwrap_or([f64::NAN; 3]);
            let contact = sim.body_contact.get(k).copied().unwrap_or([0.0; 3]);
            let law = sim.body_applied.get(k).copied().unwrap_or([0.0; 3]);
            match p {
                ForcePart::Fluid => fluid,
                ForcePart::Contact => contact,
                ForcePart::Law => law,
                ForcePart::Total => {
                    let m = b.density(sim.reg.solids[b.material].rho) * b.shape.area();
                    let w = [m * sim.gravity[0], m * sim.gravity[1], 0.0];
                    std::array::from_fn(|r| fluid[r] + contact[r] + law[r] + w[r])
                }
            }
        };
        match self {
            BodyQty::X => b.pos[0],
            BodyQty::Y => b.pos[1],
            BodyQty::Angle => b.angle.to_degrees(),
            BodyQty::Vx => b.vel[0],
            BodyQty::Vy => b.vel[1],
            BodyQty::Rotation => b.omega.to_degrees(),
            BodyQty::Fx(p) => part(*p)[0],
            BodyQty::Fy(p) => part(*p)[1],
            BodyQty::Torque(p) => part(*p)[2],
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
    /// `name` : nom du corps (`nom "X"`), pour ses relevés `corps "X" …`
    Body { shape: BodySpec, material: String, angle: f64, motion: BodyMotion, temp: Option<f64>, name: Option<String> },
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
    /// vanne à guillotine sur la zone : la part `open` (0 fermée à 1 ouverte) de ses rangées le long de son plus grand côté, depuis la
    /// gauche ou le bas, est ouverte ; le reste est fermé (faces fermées, fluide gardé)
    Valve { zone: Zone, open: f64 },
    /// pompe (compresseur) : vitesse `v` imposée sur les faces des mailles fluides de la zone
    Pump { zone: Zone, v: [f64; 2] },
    /// téléporteur : les mailles de la zone aspirent à `speed` m/s ; le fluide absorbé (composition, température) ressort avec le même
    /// débit volumique sur le disque de rayon `radius` (m, arrondi en mailles ; 0 : la seule maille) autour de `dest`
    Teleporter { zone: Zone, dest: [f64; 2], speed: f64, radius: f64 },
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
    /// variable du protocole (indice dans `Protocol::vars`) : `soit n = <valeur>`
    Let { var: usize, value: LetValue },
    /// vilebrequin, bielle, piston (et cylindre)
    Crank(CrankSpec),
    /// action aux valeurs calculées (`{…}`, formules `calcs` dans l'ordre) : ses mots `toks` sont relus avec les valeurs de l'instant
    /// chaque fois qu'elle est faite ; `check` : la même lue avec ces valeurs à 1 (noms vérifiés au départ)
    Deferred { toks: Vec<super::mp_script::Tok>, calcs: Vec<Calc>, check: Box<Action> },
    /// règle « quand <condition> : <action> », en parallèle du déroulement dès que sa ligne est atteinte : l'action est faite chaque
    /// fois que la condition devient vraie (relevée après chaque pas ; tout de suite si elle l'est déjà)
    When { clauses: Vec<Clause>, logic: Logic, action: Box<Action> },
}

impl Action {
    /// action qui modifie l'état sur-le-champ (et qu'on peut donc rejouer pour construire les conditions initiales) ; une action
    /// répétée dès le départ compte pour sa première fois
    pub fn is_instant_edit(&self) -> bool {
        match self {
            Action::Periodic { start, action, .. } => start.map_or(true, |s| s <= 0.0) && action.is_instant_edit(),
            _ => !matches!(
                self,
                Action::Heat { .. }
                    | Action::Accelerate { .. }
                    | Action::SaveState(_)
                    | Action::SaveImage { .. }
                    | Action::Mark(_)
                    | Action::Pause
                    | Action::When { .. }
                    | Action::Let { .. }
                    | Action::Deferred { .. }
            ),
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
            | Action::Valve { zone, .. }
            | Action::Pump { zone, .. }
            | Action::Teleporter { zone, .. }
            | Action::Field { zone, .. } => Some(zone),
            Action::Periodic { action, .. } | Action::When { action, .. } => action.zone(),
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

/// liaison des comparaisons d'une condition (indices dans sa liste de comparaisons) : une comparaison, toutes (« et »), l'une
/// d'elles (« ou ») ; les parenthèses du protocole font les groupes
#[derive(Clone, Debug, PartialEq)]
pub enum Logic {
    Leaf(usize),
    All(Vec<Logic>),
    Any(Vec<Logic>),
}

impl Logic {
    /// vraie, d'après la vérité de chaque comparaison
    pub fn eval(&self, truth: &[bool]) -> bool {
        match self {
            Logic::Leaf(i) => truth[*i],
            Logic::All(c) => c.iter().all(|x| x.eval(truth)),
            Logic::Any(c) => c.iter().any(|x| x.eval(truth)),
        }
    }
    /// vraie, ou bientôt vraie (`close` : comparaison encore fausse dont le franchissement est prévu)
    fn soon(&self, truth: &[bool], close: &[bool]) -> bool {
        match self {
            Logic::Leaf(i) => truth[*i] || close[*i],
            Logic::All(c) => c.iter().all(|x| x.soon(truth, close)),
            Logic::Any(c) => c.iter().any(|x| x.soon(truth, close)),
        }
    }
    /// texte, chaque groupe intérieur entre parenthèses
    fn text(&self, parts: &[String], top: bool) -> String {
        let join = |c: &[Logic], sep: &str| {
            let s = c.iter().map(|x| x.text(parts, false)).collect::<Vec<_>>().join(sep);
            if top { s } else { format!("({s})") }
        };
        match self {
            Logic::Leaf(i) => parts[*i].clone(),
            Logic::All(c) => join(c, " et "),
            Logic::Any(c) => join(c, " ou "),
        }
    }
}

/// valeur calculée d'un protocole : une formule en t (temps simulé, s) et en ses variables, gardée avec son texte
#[derive(Clone, Debug)]
pub struct Calc {
    pub text: String,
    f: super::mp_fields::Formula,
}

impl PartialEq for Calc {
    fn eq(&self, o: &Calc) -> bool {
        self.text == o.text
    }
}

impl Calc {
    /// `names` : « t », puis les variables du protocole dans l'ordre de leur première affectation (une formule lue avant que
    /// d'autres s'y ajoutent reste juste : elles viennent après)
    pub fn new(text: &str, names: &[&str]) -> Result<Calc, String> {
        Ok(Calc { text: text.to_string(), f: super::mp_fields::Formula::parse_vars(&norm(text), names)? })
    }
    /// valeur pour (t, variables…)
    pub fn eval(&self, vals: &[f64]) -> f64 {
        self.f.eval_vars(vals)
    }
}

/// seuil d'une comparaison : un nombre (dans l'unité du relevé), ou une valeur calculée à chaque relevé (variable, formule)
#[derive(Clone, Debug, PartialEq)]
pub enum Threshold {
    Num(f64),
    Calc(Calc),
}

impl Threshold {
    fn get(&self, vals: &[f64]) -> f64 {
        match self {
            Threshold::Num(v) => *v,
            Threshold::Calc(c) => c.eval(vals),
        }
    }
}

/// valeur affectée à une variable (`soit n = …`) : un nombre, un relevé (dans son unité, comme une mesure), une formule
#[derive(Clone, Debug, PartialEq)]
pub enum LetValue {
    Num(f64),
    Measure(Measure),
    Calc(Calc),
}

/// une comparaison d'une condition d'arrêt : relevé (comme une mesure), comparaison, seuil dans l'unité du relevé
#[derive(Clone, Debug, PartialEq)]
pub struct Clause {
    pub measure: Measure,
    pub cmp: Cmp,
    pub value: Threshold,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    Do(Action),
    /// avance de cette durée
    Simulate(f64),
    /// avance jusqu'à ce temps simulé
    Until(f64),
    /// avance jusqu'à ce que les comparaisons `clauses`, liées par `logic`, soient vraies, vérifiées après chaque pas (ou
    /// chaque série de pas du GPU), au plus pendant `limit` secondes si elle est donnée
    UntilCond { clauses: Vec<Clause>, logic: Logic, limit: Option<f64> },
    /// boucle « tant que <condition> [au plus d] » : la condition est relevée à chaque tour ; vraie, les étapes du corps sont faites,
    /// jusqu'au « fin » (étape `end`) qui ramène au test ; fausse (ou la durée `limit` écoulée depuis l'entrée), on passe après `end`
    While { clauses: Vec<Clause>, logic: Logic, limit: Option<f64>, end: usize },
    /// boucle « répéter n fois » : son corps, jusqu'au « fin » (étape `end`), est fait n fois ; `count` : n calculé à l'entrée
    Repeat { n: usize, count: Option<Calc>, end: usize },
    /// avance d'une durée calculée quand l'étape est atteinte (valeur × `factor`, l'unité écrite)
    SimulateCalc { calc: Calc, factor: f64 },
    /// « si <condition> » : vraie, la branche qui suit ; fausse, saut au premier « sinon » (étape `next`) ou au « fin » (`end`)
    If { clauses: Vec<Clause>, logic: Logic, next: usize, end: usize },
    /// « sinon » ou « sinon si <condition> » : atteint en suivant la branche d'avant, on passe après le « fin » (`end`) ; atteint par
    /// le saut d'une condition fausse, on entre dans sa branche (si sa propre condition est vraie ; sinon, saut à `next`)
    Else { cond: Option<(Vec<Clause>, Logic)>, next: usize, end: usize },
    /// règle « quand <condition> » en bloc : chaque fois que la condition devient vraie, les étapes jusqu'au « fin » (`end`) se
    /// déroulent en parallèle du protocole, attentes comprises ; le déroulement principal passe par-dessus
    WhenBlock { clauses: Vec<Clause>, logic: Logic, end: usize },
    /// « fin » d'un bloc (étape `start`) : retour au test d'une boucle ; sortie d'un « si » ; fin du corps d'une règle en bloc
    End { start: usize },
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
    /// texte du protocole : paramètres et balayage remplacés, composants développés (copié dans le dossier des résultats, il se
    /// relance seul)
    pub source: String,
    /// (nom, valeur) de la variante d'un balayage
    pub variant: Option<(String, String)>,
    /// variables (`soit n = …`), dans l'ordre de leur première affectation ; leurs formules les nomment avec t (temps simulé)
    pub vars: Vec<String>,
    /// lignes venues des composants (`inclure`), numérotées à la suite du texte
    pub includes: Includes,
}

/// Lignes lues dans les composants (`inclure "fichier" avec …`) : numérotées à la suite des lignes du texte du protocole, pour que
/// les étapes, les erreurs et le journal disent d'où elles viennent.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Includes {
    /// numéro de la première (le nombre de lignes du texte, plus un)
    pub first: usize,
    pub lines: Vec<IncludedLine>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IncludedLine {
    /// ligne du texte du protocole : l'« inclure » d'où elle vient
    pub line: usize,
    /// fichier du composant (tel qu'écrit dans son « inclure ») et ligne dans ce fichier ; `via` : les inclusions qui y mènent
    /// (« « composants/moteur.exp » ligne 5 → », vide pour un composant inclus par le protocole)
    pub file: String,
    pub at: usize,
    pub via: String,
    /// texte de la ligne, paramètres remplacés
    pub text: String,
}

impl Includes {
    pub fn get(&self, l: usize) -> Option<&IncludedLine> {
        if self.lines.is_empty() || l < self.first {
            return None;
        }
        self.lines.get(l - self.first)
    }
    /// « ligne 12 », ou pour une ligne d'un composant « ligne 12 (« composants/moteur.exp » ligne 30) »
    pub fn place(&self, l: usize) -> String {
        match self.get(l) {
            Some(o) => format!("ligne {} ({}« {} » ligne {})", o.line, o.via, o.file, o.at),
            None => format!("ligne {l}"),
        }
    }
    /// message (ou chacune de ses lignes) qui commence par « ligne N » : situé dans son composant si la ligne N en vient
    pub fn relabel(&self, msg: &str) -> String {
        if self.lines.is_empty() {
            return msg.to_string();
        }
        let one = |l: &str| -> String {
            if let Some(rest) = l.strip_prefix("ligne ") {
                let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(n) = digits.parse::<usize>() {
                    if self.get(n).is_some() {
                        return format!("{}{}", self.place(n), &rest[digits.len()..]);
                    }
                }
            }
            l.to_string()
        };
        msg.split('\n').map(one).collect::<Vec<_>>().join("\n")
    }
    /// lignes d'un bloc entre son ouverture (ligne l0) et son « fin » (ligne l1), dans leur fichier
    pub fn block(&self, l0: usize, l1: usize) -> String {
        match (self.get(l0), self.get(l1)) {
            (Some(a), Some(b)) if a.file == b.file && a.via == b.via => {
                format!("bloc des lignes {} à {} de « {} »", a.at + 1, b.at.saturating_sub(1).max(a.at + 1), a.file)
            }
            _ => format!("bloc des lignes {} à {}", l0 + 1, l1.saturating_sub(1).max(l0 + 1)),
        }
    }
}

impl Protocol {
    /// durée simulée prévue (fin du dernier « simuler »), pour la barre de progression
    pub fn planned_end(&self, t0: f64) -> f64 {
        self.span(0, self.steps.len(), t0)
    }
    /// fin prévue des étapes [from, to) commencées à t0 : une boucle bornée pour sa durée permise, une boucle sans borne pour un
    /// tour, « répéter n fois » pour n tours, un « si » pour sa plus longue branche ; le corps d'une règle en bloc ne compte pas (il
    /// se déroule en parallèle)
    fn span(&self, from: usize, to: usize, t0: f64) -> f64 {
        let mut t = t0;
        let mut k = from;
        while k < to {
            match &self.steps[k].0 {
                Step::Simulate(d) => t += d,
                Step::Until(u) => t = t.max(*u),
                Step::UntilCond { limit, .. } => t += limit.unwrap_or(0.0),
                Step::While { limit: Some(d), end, .. } => {
                    t += d;
                    k = *end;
                }
                Step::While { end, .. } => {
                    t = self.span(k + 1, *end, t);
                    k = *end;
                }
                Step::Repeat { n, count, end } => {
                    // (nombre de tours calculé : un tour)
                    let n = if count.is_some() { 1 } else { *n };
                    t += n as f64 * self.span(k + 1, *end, 0.0);
                    k = *end;
                }
                Step::If { next, end, .. } => {
                    let mut longest = self.span(k + 1, *next, t);
                    let mut b = *next;
                    while b < *end {
                        let Step::Else { next: nb, .. } = &self.steps[b].0 else { break };
                        longest = longest.max(self.span(b + 1, *nb, t));
                        b = *nb;
                    }
                    t = longest;
                    k = *end;
                }
                Step::WhenBlock { end, .. } => k = *end,
                Step::Do(_) | Step::Else { .. } | Step::End { .. } | Step::SimulateCalc { .. } => {}
            }
            k += 1;
        }
        t
    }
    /// conditions initiales : les actions d'avant la première étape qui fait avancer le temps ou qui décide (boucle, « si »), le
    /// corps des règles « quand » en bloc sauté ; et l'indice de cette étape
    pub fn initial_actions(&self) -> (Vec<usize>, usize) {
        let mut v = vec![];
        let mut k = 0;
        while k < self.steps.len() {
            match &self.steps[k].0 {
                Step::Do(_) => v.push(k),
                Step::WhenBlock { end, .. } => k = *end,
                _ => break,
            }
            k += 1;
        }
        (v, k)
    }
    /// indice de la première étape qui fait avancer le temps ou qui décide (les actions avant elle sont les conditions initiales)
    pub fn first_advance(&self) -> usize {
        self.initial_actions().1
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
        let a = match a {
            Action::Periodic { action, .. } | Action::When { action, .. } => action.as_ref(),
            _ => a,
        };
        let a = if let Action::Deferred { check, .. } = a { check.as_ref() } else { a };
        match a {
            Action::Fill { fluid: f, .. } => fluid(f, *line, false, &mut errs),
            Action::Emitter { fluid: f, .. } => fluid(f, *line, true, &mut errs),
            Action::Boundary { spec: BoundarySpec::Inlet { fluid: Some(f), .. }, .. } => fluid(f, *line, true, &mut errs),
            Action::Field { qty: FieldQty::Fraction(f), .. } => fluid(f, *line, false, &mut errs),
            Action::Solid { material, .. }
            | Action::Body { material, .. }
            | Action::Crank(CrankSpec { material, .. })
            | Action::Field { qty: FieldQty::Solid(material), .. } => {
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
    build_start_at(proto, base).map_err(|e| proto.includes.relabel(&e))
}

fn build_start_at(proto: &Protocol, base: &Path) -> Result<(MultiphaseSim, Option<MaterialLibrary>, Option<usize>), String> {
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
        Action::Crank(spec) => add_crank(sim, spec, base),
        Action::Body { shape, material, angle, motion, temp, name } => {
            let s = find_solid(sim, material)?;
            let mut b = shape.body(s, *angle, temp.unwrap_or(t_amb), motion)?;
            b.name = name.clone().unwrap_or_default();
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
        Action::Valve { zone, open } => {
            // le long du plus grand côté de la zone, les rangées d'indice < part ouverte × nombre de rangées sont ouvertes (mailles
            // ordinaires), les autres fermées ; une autre cellule spéciale de la zone est laissée telle quelle
            let cs = cells(sim, zone);
            if cs.is_empty() {
                return Err(format!("la zone « {} » est hors du domaine", zone.label()));
            }
            let (i0, i1) = (cs.iter().map(|c| c.0).min().unwrap(), cs.iter().map(|c| c.0).max().unwrap());
            let (j0, j1) = (cs.iter().map(|c| c.1).min().unwrap(), cs.iter().map(|c| c.1).max().unwrap());
            let along_x = i1 - i0 >= j1 - j0;
            let rows = if along_x { i1 - i0 + 1 } else { j1 - j0 + 1 };
            let n_open = (open.clamp(0.0, 1.0) * rows as f64 + 0.5).floor() as usize;
            let mut closed = 0;
            for &(i, j) in &cs {
                let c = sim.idx(i, j);
                if sim.is_solid(c) || !matches!(sim.special[c], None | Some(MpEffect::Valve)) {
                    continue;
                }
                let shut = (if along_x { i - i0 } else { j - j0 }) >= n_open;
                sim.special[c] = if shut { Some(MpEffect::Valve) } else { None };
                closed += shut as usize;
            }
            sim.refresh();
            Ok(closed)
        }
        Action::Pump { zone, v } => {
            // mailles fluides sans autre effet ; une pompe déjà posée change de vitesse, (0, 0) la retire
            let mut n = 0;
            for (i, j) in cells(sim, zone) {
                let c = sim.idx(i, j);
                if sim.is_solid(c) || !matches!(sim.special[c], None | Some(MpEffect::Pump { .. })) {
                    continue;
                }
                sim.special[c] = if *v == [0.0, 0.0] { None } else { Some(MpEffect::Pump { vx: v[0], vy: v[1] }) };
                n += 1;
            }
            sim.refresh();
            Ok(n)
        }
        Action::Teleporter { zone, dest, speed, radius } => {
            let (lx, ly) = (sim.nx as f64 * sim.dx, sim.ny as f64 * sim.dx);
            if !(dest[0] >= 0.0 && dest[0] < lx && dest[1] >= 0.0 && dest[1] < ly) {
                return Err(format!("la destination {} du téléporteur est hors du domaine", fmt_point(*dest)));
            }
            let d = sim.idx((dest[0] / sim.dx) as usize, (dest[1] / sim.dx) as usize);
            if sim.is_solid(d) {
                return Err(format!("la destination {} du téléporteur est dans un solide", fmt_point(*dest)));
            }
            // (une zone qui couvre la destination la laisse libre, comme le pinceau)
            if sim.special[d].is_some() {
                return Err(format!("la destination {} du téléporteur est sur une autre cellule spéciale", fmt_point(*dest)));
            }
            effect(sim, zone, MpEffect::Teleporter { dest: d, speed: *speed, radius: (radius / sim.dx).round() as usize })
        }
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
        Action::Heat { .. }
        | Action::Accelerate { .. }
        | Action::SaveState(_)
        | Action::SaveImage { .. }
        | Action::Mark(_)
        | Action::Pause
        | Action::When { .. }
        | Action::Let { .. }
        | Action::Deferred { .. } => Ok(0),
        // (conditions initiales : sa première fois ; le déroulement la répète)
        Action::Periodic { action, .. } => apply_action(sim, action, base),
    }
}

/// Pose un vilebrequin (`vilebrequin …`) : son cylindre s'il en faut un (parois solides le long de l'axe, culasse ; alésage d'un nombre
/// entier de mailles, l'axe du vilebrequin recalé au milieu, pour que le piston y glisse sans fuite), puis la manivelle, la bielle et
/// le piston
fn add_crank(sim: &mut MultiphaseSim, spec: &CrankSpec, base: &Path) -> Result<usize, String> {
    let mat = find_solid(sim, &spec.material)?;
    let dx = sim.dx;
    let axis = spec.axis_deg.to_radians();
    let mut bore = spec.bore.unwrap_or(2.0 * spec.r);
    let piston_h = spec.piston_h.unwrap_or(spec.r);
    let mut pivot = spec.pivot;
    if let Some((chamber, wall)) = spec.cylinder {
        let q = (spec.axis_deg / 90.0).round();
        if (spec.axis_deg - 90.0 * q).abs() > 1e-9 {
            return Err("un cylindre suit la grille : son axe fait 0, 90, 180 ou 270°".into());
        }
        let cells = (bore / dx).round().max(1.0);
        bore = cells * dx;
        // (coordonnée en travers de l'axe : x pour un axe vertical)
        let across = if (q as i64).rem_euclid(2) == 1 { 0 } else { 1 };
        pivot[across] = if (cells as i64) % 2 == 0 { (pivot[across] / dx).round() * dx } else { ((pivot[across] / dx).floor() + 0.5) * dx };
        let (c, e) = (chamber.unwrap_or(0.5 * spec.r), wall.unwrap_or(dx));
        if !(c > 0.0 && e > 0.0) {
            return Err("la chambre et la paroi du cylindre doivent être positives".into());
        }
        let (dir, nor) = ([axis.cos().round(), axis.sin().round()], [-axis.sin().round(), axis.cos().round()]);
        let world = |along: f64, side: f64| [pivot[0] + along * dir[0] + side * nor[0], pivot[1] + along * dir[1] + side * nor[1]];
        let (lo, hi, half) = (spec.l - spec.r, spec.l + spec.r + piston_h + c, bore / 2.0);
        for (a0, a1, b0, b1) in [(lo, hi + e, -half - e, -half), (lo, hi + e, half, half + e), (hi, hi + e, -half - e, half + e)] {
            let (p, q) = (world(a0, b0), world(a1, b1));
            let zone = Zone::Rect { a: [p[0].min(q[0]), p[1].min(q[1])], b: [p[0].max(q[0]), p[1].max(q[1])] };
            apply_action(sim, &Action::Solid { zone, material: spec.material.clone(), thermal: ThermalSpec::Passive, temp: spec.temp }, base)?;
        }
    }
    let ck = super::mp_mechanism::Crank {
        pivot,
        r: spec.r,
        l: spec.l,
        axis,
        piston_h,
        rod_len: 0.0,
        bodies: [0; 3],
        theta: spec.angle_deg.to_radians(),
        omega: spec.omega,
        driven: (!spec.free).then_some(spec.omega),
        inertia: spec.inertia,
        load: spec.load.clone(),
        torque: 0.0,
        applied: 0.0,
        name: spec.name.clone().unwrap_or_default(),
    };
    let w = (spec.r / 3.0).max(dx);
    let t = spec.temp.unwrap_or(sim.t_ambient);
    sim.add_crank(ck, bore, w, mat, t)?;
    Ok(3)
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
    for k in proto.initial_actions().0 {
        if let (Step::Do(a), line) = &proto.steps[k] {
            if a.is_instant_edit() {
                apply_action(&mut sim, a, base).map_err(|e| format!("{} : {e}", proto.includes.place(*line)))?;
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
    Time,
    Var(usize),
    Crank(Who, CrankQty),
    Body(Who, BodyQty),
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
        D::Vorticity => "vorticite",
        D::Flame => "flamme",
        D::HeatRelease => "chaleur",
    }
}

/// une comparaison d'une condition en cours : relevé, comparaison, seuil, nom et unité du relevé, dernier relevé (t, valeur), relevé
/// le plus éloigné du seuil depuis que la comparaison est fausse
struct CondClause {
    probe: Probe,
    cmp: Cmp,
    value: f64,
    name: String,
    unit: String,
    last: Option<(f64, f64)>,
    far: Option<(f64, f64)>,
    /// seuil écrit (nombre, ou valeur calculée à chaque relevé : `value` est sa valeur au dernier relevé)
    threshold: Threshold,
}

/// condition en cours : comparaisons liées par `logic` ; `near` : franchissement prévu avant la prochaine série de pas du GPU (il
/// repasse alors au pas unique)
struct Cond {
    clauses: Vec<CondClause>,
    logic: Logic,
    line: usize,
    near: bool,
}

impl Cond {
    /// « corps 1 y < 0.1 m ou (max T partout > 400 K et temps > « t0 + 1 » (3 s)) » — un seuil calculé avec sa dernière valeur
    fn text(&self) -> String {
        let parts: Vec<String> = self
            .clauses
            .iter()
            .map(|c| match &c.threshold {
                Threshold::Calc(k) => format!("{} {} « {} » ({})", c.name, c.cmp.symbol(), k.text, with_unit(c.value, &c.unit)),
                Threshold::Num(_) => format!("{} {} {}", c.name, c.cmp.symbol(), with_unit(c.value, &c.unit)),
            })
            .collect();
        self.logic.text(&parts, true)
    }
    /// relevés (et seuils calculés) pour l'état `sim` et les valeurs (t, variables…)
    fn values(&mut self, sim: &MultiphaseSim, vals: &[f64]) -> Vec<f64> {
        for c in &mut self.clauses {
            c.value = c.threshold.get(vals);
        }
        self.clauses.iter().map(|c| c.probe.value(sim, vals)).collect()
    }
    fn holds(&self, xs: &[f64]) -> bool {
        let truth: Vec<bool> = self.clauses.iter().zip(xs).map(|(c, &x)| c.cmp.holds(x, c.value)).collect();
        self.logic.eval(&truth)
    }
    /// relevés `xs` à l'instant t (pas dt) : les garde comme derniers relevés et prévoit, d'après la tendance des deux derniers, si la
    /// condition encore fausse va devenir vraie dans moins de deux séries de 8 pas (`near`) : chaque comparaison fausse a son
    /// franchissement prévu ou non, et la liaison dit si cela suffit (« et » : toutes celles d'un groupe ; « ou » : l'une)
    fn track(&mut self, t: f64, dt: f64, xs: &[f64]) {
        let (mut truth, mut close) = (vec![], vec![]);
        let soon = |from: Option<(f64, f64)>, x: f64, value: f64| match from {
            Some((t0, x0)) if t > t0 && x != x0 => {
                let eta = (value - x) / ((x - x0) / (t - t0));
                eta >= 0.0 && eta < 16.0 * dt
            }
            _ => false,
        };
        for (cl, &x) in self.clauses.iter_mut().zip(xs) {
            let holds = cl.cmp.holds(x, cl.value);
            if holds {
                cl.far = None;
                close.push(false);
            } else {
                if cl.far.map_or(true, |(_, xf)| (cl.value - x).abs() > (cl.value - xf).abs()) {
                    cl.far = Some((t, x));
                }
                // tendance du dernier intervalle, ou depuis le relevé le plus éloigné du seuil : une sonde qui oscille (ondes de
                // pression…) trompe la première — un compresseur qui démarre faisait sauter le seuil d'une série de 8 pas au GPU
                close.push(soon(cl.last, x, cl.value) || soon(cl.far, x, cl.value));
            }
            truth.push(holds);
            cl.last = Some((t, x));
        }
        self.near = self.logic.soon(&truth, &close);
    }
    /// « 0.099 m ; 412 K »
    fn values_text(&self, xs: &[f64]) -> String {
        self.clauses.iter().zip(xs).map(|(c, &x)| with_unit(x, &c.unit)).collect::<Vec<_>>().join(" ; ")
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
    /// valeur pour l'état `sim` ; `vals` : (t, variables du protocole…)
    fn value(&self, sim: &MultiphaseSim, vals: &[f64]) -> f64 {
        match &self.kind {
            ProbeKind::Crank(who, q) => return q.value(sim, who.find(sim.cranks.iter().map(|c| c.name.as_str()))),
            ProbeKind::Body(who, q) => return q.value(sim, who.find(sim.bodies.iter().map(|b| b.name.as_str()))),
            _ => {}
        }
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
            ProbeKind::Time => sim.time,
            ProbeKind::Var(k) => vals.get(k + 1).copied().unwrap_or(f64::NAN),
            ProbeKind::Crank(..) | ProbeKind::Body(..) => unreachable!(),
            ProbeKind::SolidForce(axis) => solid_force(sim, &self.cells)[axis],
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

/// boucle (« tant que », « répéter ») : instant d'entrée, tours faits, tours de suite sans que le temps avance, instant du dernier test
#[derive(Default)]
struct LoopState {
    since: f64,
    turns: usize,
    idle: usize,
    last: f64,
    /// tours à faire (« répéter »)
    n: usize,
}

/// suite d'étapes en cours : le déroulement principal, ou le corps d'une règle « quand » en bloc, déroulé en parallèle
#[derive(Default)]
struct Seq {
    /// prochaine étape ; la suite est finie à `end` (le « fin » de son bloc ; pour le déroulement principal, après la dernière)
    pc: usize,
    end: usize,
    /// attente : jusqu'à cet instant (borne d'une attente de condition), jusqu'à cette condition
    target: Option<f64>,
    cond: Option<Cond>,
    /// boucles (par indice de leur étape)
    loops: std::collections::HashMap<usize, LoopState>,
    /// étape atteinte par un saut : le « fin » d'une boucle vers son test, une condition fausse vers son « sinon »
    jumped: Option<usize>,
    /// corps des boucles au-delà de leur troisième tour (indices de leur « fin ») : journal muet
    mute: Vec<usize>,
    /// règle qui l'a lancée ; journal muet (au-delà de la troisième fois de cette règle)
    rule: Option<usize>,
    quiet: bool,
}

/// règle « quand » en cours : condition ; action d'une ligne, ou corps d'un bloc (étapes `start`..`end`) ; condition vraie au dernier
/// relevé ; nombre de fois ; suite de son bloc en cours ; fois sautées (la suite d'avant tournait encore)
struct WhenRule {
    cond: Cond,
    action: Option<Action>,
    body: Option<(usize, usize)>,
    on: bool,
    count: usize,
    running: bool,
    skipped: usize,
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
    /// déroulement principal (étape, attente, boucles) ; suites des règles « quand » en bloc, en parallèle
    main: Seq,
    /// valeurs des variables du protocole (0 avant leur première affectation)
    vars: Vec<f64>,
    threads: Vec<Seq>,
    timed: Vec<Timed>,
    periodic: Vec<Periodic>,
    /// règles « quand … : … »
    when: Vec<WhenRule>,
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
            return Err(errs.iter().map(|(l, e)| format!("{} : {e}", proto.includes.place(*l))).collect::<Vec<_>>().join("\n"));
        }
        let includes = proto.includes.clone();
        Self::open(proto, base, out_dir, gpu, sim, library, scenario).map_err(|e| includes.relabel(&e))
    }

    fn open(
        proto: Protocol,
        base: &Path,
        out_dir: Option<PathBuf>,
        gpu: Option<bool>,
        sim: MultiphaseSim,
        library: Option<MaterialLibrary>,
        scenario: Option<usize>,
    ) -> Result<Runner, String> {
        let mut r = Runner {
            t_end: proto.planned_end(sim.time),
            t_start: sim.time,
            sim,
            gpu: None,
            base: base.to_path_buf(),
            out_dir,
            library,
            scenario,
            main: Seq::default(),
            vars: vec![],
            threads: vec![],
            timed: vec![],
            periodic: vec![],
            when: vec![],
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
        r.main.end = r.proto.steps.len();
        r.vars = vec![0.0; r.proto.vars.len()];
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
            let clauses = match s {
                Step::UntilCond { clauses, .. }
                | Step::While { clauses, .. }
                | Step::If { clauses, .. }
                | Step::Else { cond: Some((clauses, _)), .. }
                | Step::WhenBlock { clauses, .. }
                | Step::Do(Action::When { clauses, .. }) => clauses,
                _ => continue,
            };
            for c in clauses {
                if let Measure::Flux { a, b, .. } = &c.measure {
                    lines.push((*a, *b));
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
        let msg = self.proto.includes.relabel(&msg);
        self.log.push(format!("[t = {:.6} s] {msg}", self.sim.time));
    }

    fn compile_measures(&mut self) -> Result<(), String> {
        let ms = self.proto.measures.clone();
        for (m, line) in ms {
            match self.make_probe(&m).map_err(|e| format!("ligne {line} : {e}"))? {
                None => self.stats = true,
                Some((probe, name, unit)) => {
                    self.columns.push(if unit.is_empty() { name } else { format!("{name} ({unit})") });
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
            Measure::Time { label } => {
                (Probe { kind: ProbeKind::Time, cells: vec![], fluid_only: false }, label.clone().unwrap_or_else(|| "temps".into()), "s".to_string())
            }
            Measure::Crank { who, q, label } => {
                let name = label.clone().unwrap_or_else(|| format!("vilebrequin {} {}", who.label(), q.name()));
                (Probe { kind: ProbeKind::Crank(who.clone(), *q), cells: vec![], fluid_only: false }, name, q.unit().to_string())
            }
            Measure::Var { var, label } => {
                let name = label.clone().unwrap_or_else(|| self.proto.vars.get(*var).cloned().unwrap_or_default());
                (Probe { kind: ProbeKind::Var(*var), cells: vec![], fluid_only: false }, name, String::new())
            }
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
            Measure::Body { who, q, label } => {
                let sfx = match q {
                    BodyQty::Fx(p) | BodyQty::Fy(p) | BodyQty::Torque(p) => p.suffix(),
                    _ => "",
                };
                let name = label.clone().unwrap_or_else(|| format!("corps {} {}{sfx}", who.label(), q.name()));
                (Probe { kind: ProbeKind::Body(who.clone(), *q), cells: vec![], fluid_only: false }, name, q.unit().to_string())
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
            if let Some(target) = self.main.target {
                self.sim_step(target);
                continue;
            }
            if self.main.pc >= self.main.end {
                self.finish();
                break;
            }
            let mut seq = std::mem::take(&mut self.main);
            self.exec(&mut seq, true);
            self.main = seq;
        }
        &self.state
    }

    /// Fait l'étape suivante de la suite `seq` (des actions consécutives d'un coup) : le déroulement principal (`main`) ou la suite
    /// d'une règle en bloc. Une attente pose `seq.target` (et `seq.cond`) : la suite reprend quand elle est finie.
    fn exec(&mut self, seq: &mut Seq, main: bool) {
        let outer = self.quiet;
        self.quiet = outer || seq.quiet || !seq.mute.is_empty();
        let (step, line) = self.proto.steps[seq.pc].clone();
        if main {
            self.line = line;
        }
        match step {
            Step::Do(_) => self.run_actions(seq, main),
            Step::While { clauses, logic, limit, end } => self.loop_test(seq, line, &clauses, &logic, limit, end, outer),
            Step::Repeat { n, count, end } => self.repeat_test(seq, line, n, count.as_ref(), end, outer),
            Step::SimulateCalc { calc, factor } => {
                let d = calc.eval(&self.var_values()) * factor;
                if !(d >= 0.0 && d.is_finite()) {
                    self.fail(format!("ligne {line} : durée calculée « {} » = {} s : il faut un nombre positif", calc.text, fmt_num(d)));
                } else {
                    if main {
                        self.start_sampling();
                    }
                    seq.target = Some(self.sim.time + d);
                    seq.pc += 1;
                }
            }
            Step::If { clauses, logic, next, .. } => {
                if let Some((holds, text)) = self.eval_cond(line, &clauses, &logic) {
                    self.note(format!("ligne {line} : si {text} — {}", if holds { "vrai" } else { "faux" }));
                    if holds {
                        seq.pc += 1;
                    } else {
                        seq.jumped = Some(next);
                        seq.pc = next;
                    }
                }
            }
            Step::Else { cond, next, end } => {
                if seq.jumped == Some(seq.pc) {
                    // (saut d'une condition fausse : cette branche, si sa propre condition le veut)
                    seq.jumped = None;
                    match cond {
                        None => seq.pc += 1,
                        Some((clauses, logic)) => {
                            if let Some((holds, text)) = self.eval_cond(line, &clauses, &logic) {
                                self.note(format!("ligne {line} : sinon si {text} — {}", if holds { "vrai" } else { "faux" }));
                                if holds {
                                    seq.pc += 1;
                                } else {
                                    seq.jumped = Some(next);
                                    seq.pc = next;
                                }
                            }
                        }
                    }
                } else {
                    // branche d'avant finie : après le « fin »
                    seq.pc = end + 1;
                }
            }
            Step::End { start } => {
                if matches!(self.proto.steps[start].0, Step::While { .. } | Step::Repeat { .. }) {
                    seq.jumped = Some(start);
                    seq.pc = start;
                } else {
                    seq.jumped = None;
                    seq.pc += 1;
                }
            }
            Step::WhenBlock { clauses, logic, end } => {
                // la règle est posée (une fois par ligne) ; son corps se déroule en parallèle, quand sa condition devient vraie
                let start = seq.pc;
                seq.pc = end + 1;
                if !(line > 0 && self.registered(line)) {
                    let _ = self.arm_rule(line, &clauses, &logic, None, Some((start, end)));
                }
            }
            Step::Simulate(d) => {
                if main {
                    self.start_sampling();
                }
                seq.target = Some(self.sim.time + d);
                seq.pc += 1;
            }
            Step::Until(t) => {
                if main {
                    self.start_sampling();
                }
                if t > self.sim.time {
                    seq.target = Some(t);
                } else {
                    self.note(format!("ligne {line} : « simuler jusqu'à {} s » déjà atteint", fmt_num(t)));
                }
                seq.pc += 1;
            }
            Step::UntilCond { clauses, logic, limit } => {
                if main {
                    self.start_sampling();
                }
                seq.pc += 1;
                if let Some(mut c) = self.make_cond(line, &clauses, &logic) {
                    // (pas de tendance avant deux relevés : un pas à la fois, sinon le GPU ferait d'emblée une série de 8 pas)
                    c.near = true;
                    let vals = self.var_values();
                    let xs = c.values(&self.sim, &vals);
                    if c.holds(&xs) {
                        self.note(format!("ligne {line} : {} déjà vrai ({})", c.text(), c.values_text(&xs)));
                    } else {
                        let what = match limit {
                            Some(d) => format!("au plus {} s", fmt_num(d)),
                            None => "sans limite de durée".to_string(),
                        };
                        self.note(format!("ligne {line} : simuler jusqu'à ce que {} ({what}) — {} au départ", c.text(), c.values_text(&xs)));
                        seq.target = Some(limit.map_or(f64::INFINITY, |d| self.sim.time + d));
                        seq.cond = Some(c);
                    }
                }
            }
        }
        self.quiet = outer;
    }

    /// Condition prête à être suivie (sondes, premiers relevés gardés) ; None : l'expérience a échoué (relevé impossible)
    fn make_cond(&mut self, line: usize, clauses: &[Clause], logic: &Logic) -> Option<Cond> {
        let mut parts = vec![];
        for cl in clauses {
            match self.make_probe(&cl.measure) {
                Ok(Some((probe, name, unit))) => {
                    parts.push(CondClause { probe, cmp: cl.cmp, value: f64::NAN, name, unit, last: None, far: None, threshold: cl.value.clone() })
                }
                Ok(None) => {
                    self.fail(format!("ligne {line} : une condition porte sur un relevé, pas sur « stats »"));
                    return None;
                }
                Err(e) => {
                    self.fail(format!("ligne {line} : {e}"));
                    return None;
                }
            }
        }
        self.sync_host();
        let mut c = Cond { clauses: parts, logic: logic.clone(), line, near: false };
        let vals = self.var_values();
        let xs = c.values(&self.sim, &vals);
        let t = self.sim.time;
        for (cl, &x) in c.clauses.iter_mut().zip(&xs) {
            cl.last = Some((t, x));
            cl.far = (!cl.cmp.holds(x, cl.value)).then_some((t, x));
        }
        Some(c)
    }

    /// relève une condition (boucle, « si ») : (vraie ?, « relevé op seuil (valeurs) ») ; None : l'expérience a échoué
    fn eval_cond(&mut self, line: usize, clauses: &[Clause], logic: &Logic) -> Option<(bool, String)> {
        let mut c = self.make_cond(line, clauses, logic)?;
        let vals = self.var_values();
        let xs = c.values(&self.sim, &vals);
        Some((c.holds(&xs), format!("{} ({})", c.text(), c.values_text(&xs))))
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

    /// Exécute les actions consécutives de la suite `seq` (une seule relecture / un seul envoi GPU pour le groupe).
    fn run_actions(&mut self, seq: &mut Seq, main: bool) {
        self.sync_host();
        let (mut edited, mut heat) = (false, false);
        while seq.pc < seq.end {
            let (step, line) = self.proto.steps[seq.pc].clone();
            let Step::Do(a) = step else { break };
            if main {
                self.line = line;
            }
            seq.pc += 1;
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

    /// Fait une action tout de suite, pendant l'expérience (geste de la fenêtre) ; le journal la note « geste ».
    pub fn apply_now(&mut self, a: &Action) -> Result<(), String> {
        if !matches!(self.state, RunState::Running | RunState::Paused(_)) {
            return Err("l'expérience est terminée".into());
        }
        self.sync_host();
        let n0 = self.log.len();
        let r = self.do_action(a, 0);
        for l in &mut self.log[n0..] {
            *l = l.replacen("ligne 0 : ", "geste : ", 1);
        }
        match r {
            Ok((e, h)) => {
                self.finish_edits(e, h);
                Ok(())
            }
            Err(()) => Err(match &self.state {
                RunState::Failed(m) => m.clone(),
                _ => "action refusée".into(),
            }),
        }
    }

    /// après des actions : cibles de chauffage recalculées, état renvoyé au GPU s'il a changé
    fn finish_edits(&mut self, edited: bool, heat: bool) {
        if edited || heat {
            // l'état a changé (vanne, chauffage…) : les tendances relevées avant ne disent plus rien, les conditions en cours sont
            // relevées pas à pas jusqu'aux suivantes (sans quoi le GPU vidait une cuve pendant une série de pas)
            if let Some(c) = &mut self.main.cond {
                c.near = true;
            }
            for w in &mut self.when {
                w.cond.near = true;
            }
            for c in self.threads.iter_mut().filter_map(|s| s.cond.as_mut()) {
                c.near = true;
            }
        }
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

    /// (t, variables…) : les valeurs que nomment les formules du protocole
    fn var_values(&self) -> Vec<f64> {
        std::iter::once(self.sim.time).chain(self.vars.iter().copied()).collect()
    }

    /// texte d'un repère : chaque « {…} » (formule en t et en variables, un nom seul par exemple) remplacé par sa valeur
    fn interpolate(&self, t: &str) -> String {
        let names: Vec<&str> = std::iter::once("t").chain(self.proto.vars.iter().map(|s| s.as_str())).collect();
        let vals = self.var_values();
        let (mut out, mut rest) = (String::new(), t);
        while let Some(i) = rest.find('{') {
            let Some(j) = rest[i..].find('}') else { break };
            out += &rest[..i];
            match Calc::new(rest[i + 1..i + j].trim(), &names) {
                Ok(c) => out += &fmt_num(c.eval(&vals)),
                Err(_) => out += &rest[i..i + j + 1],
            }
            rest = &rest[i + j + 1..];
        }
        out + rest
    }

    /// une règle « quand » ou une répétition de cette ligne du protocole est déjà en place (ligne repassée dans une boucle)
    fn registered(&self, line: usize) -> bool {
        self.when.iter().any(|w| w.cond.line == line) || self.periodic.iter().any(|p| p.line == line)
    }

    /// Test d'une boucle « tant que » (étape `seq.pc`) : vraie, on entre dans son corps ; fausse, ou sa durée permise écoulée depuis
    /// l'entrée, on passe après son « fin » (étape `end`). Une boucle qui tourne sans que le temps avance (corps dont les attentes
    /// sont déjà atteintes) arrête l'expérience. `outer` : journal muet hors de la suite.
    #[allow(clippy::too_many_arguments)]
    fn loop_test(&mut self, seq: &mut Seq, line: usize, clauses: &[Clause], logic: &Logic, limit: Option<f64>, end: usize, outer: bool) {
        let k = seq.pc;
        let back = seq.jumped.take() == Some(k);
        let Some((holds, text)) = self.eval_cond(line, clauses, logic) else { return };
        let now = self.sim.time;
        let eps = 1e-12 * (1.0 + now.abs());
        let st = seq.loops.entry(k).or_default();
        if back {
            st.idle = if now <= st.last + eps { st.idle + 1 } else { 0 };
            st.last = now;
        } else {
            *st = LoopState { since: now, turns: 0, idle: 0, last: now, n: 0 };
        }
        let (since, idle) = (st.since, st.idle);
        if idle > 1000 {
            return self.fail(format!(
                "ligne {line} : la boucle « tant que » tourne sans que le temps avance (ses attentes sont déjà atteintes) : il faut qu'elle simule ou attende à chaque tour"
            ));
        }
        let over = limit.is_some_and(|d| now >= since + d - eps);
        if holds && !over {
            self.enter_turn(seq, k, line, end, &format!("tant que {text}"));
        } else {
            let why = if holds { format!(" (au bout des {} s permises)", fmt_num(limit.unwrap_or(0.0))) } else { String::new() };
            self.leave_loop(seq, k, line, end, &format!("tant que {text}"), &why, outer);
        }
    }

    /// Test d'une boucle « répéter n fois » (étape `seq.pc`, « fin » à `end`) ; un nombre de tours calculé l'est à l'entrée
    fn repeat_test(&mut self, seq: &mut Seq, line: usize, n: usize, count: Option<&Calc>, end: usize, outer: bool) {
        let k = seq.pc;
        let back = seq.jumped.take() == Some(k);
        let now = self.sim.time;
        let n = if back {
            seq.loops.get(&k).map_or(n, |s| s.n)
        } else {
            let n = count.map_or(n, |c| {
                let v = c.eval(&self.var_values()).round();
                if v.is_finite() && v > 0.0 { v as usize } else { 0 }
            });
            seq.loops.insert(k, LoopState { since: now, turns: 0, idle: 0, last: now, n });
            n
        };
        let st = seq.loops.get(&k).unwrap();
        let what = format!("répéter {n} fois");
        if st.turns < n {
            self.enter_turn(seq, k, line, end, &what);
        } else {
            self.leave_loop(seq, k, line, end, &what, "", outer);
        }
    }

    /// tour suivant de la boucle de l'étape k (« fin » à `end`) : le journal note les trois premiers, puis son corps se tait
    fn enter_turn(&mut self, seq: &mut Seq, k: usize, line: usize, end: usize, what: &str) {
        let st = seq.loops.get_mut(&k).unwrap();
        st.turns += 1;
        let n = st.turns;
        if n <= 3 {
            self.note(format!("ligne {line} : {what} — tour {n}"));
        } else if n == 4 {
            self.note(format!("ligne {line} : tours suivants non notés au journal"));
            seq.mute.push(end);
        }
        seq.pc += 1;
    }

    /// sortie de la boucle de l'étape k : le journal (rouvert) note ses tours ; on passe après son « fin »
    #[allow(clippy::too_many_arguments)]
    fn leave_loop(&mut self, seq: &mut Seq, k: usize, line: usize, end: usize, what: &str, why: &str, outer: bool) {
        seq.mute.retain(|&m| m > end);
        self.quiet = outer || seq.quiet || !seq.mute.is_empty();
        let tours = match seq.loops.get(&k).map_or(0, |s| s.turns) {
            0 => "aucun tour".to_string(),
            1 => "fini après 1 tour".to_string(),
            n => format!("fini après {n} tours"),
        };
        self.note(format!("ligne {line} : {what} — {tours}{why}"));
        seq.pc = end + 1;
    }

    /// Pose une règle « quand » : action d'une ligne, ou corps d'un bloc (étapes `start`..`end`) ; faite tout de suite si sa
    /// condition est déjà vraie
    fn arm_rule(&mut self, line: usize, clauses: &[Clause], logic: &Logic, action: Option<&Action>, body: Option<(usize, usize)>) -> Result<(bool, bool), ()> {
        let Some(mut cond) = self.make_cond(line, clauses, logic) else { return Err(()) };
        let vals = self.var_values();
        let xs = cond.values(&self.sim, &vals);
        let on = cond.holds(&xs);
        // (pas de tendance avant deux relevés : un pas à la fois tant que la règle attend sa condition)
        cond.near = !on;
        let what = match (action, body) {
            (Some(Action::Let { var, value }), _) => {
                let name = self.proto.vars.get(*var).cloned().unwrap_or_default();
                describe(&Action::Let { var: *var, value: value.clone() }).replacen("variable", &format!("soit {name}"), 1)
            }
            (Some(a), _) => describe(a),
            (None, Some((a, b))) => {
                let (l0, l1) = (self.proto.steps[a].1, self.proto.steps[b].1);
                self.proto.includes.block(l0, l1)
            }
            _ => String::new(),
        };
        let now_text = if on { " : déjà vrai, faite maintenant" } else { "" };
        self.note(format!("ligne {line} : quand {} : {what} — {} au départ{now_text}", cond.text(), cond.values_text(&xs)));
        let k = self.when.len();
        self.when.push(WhenRule { cond, action: action.cloned(), body, on, count: 0, running: false, skipped: 0 });
        if on { self.fire_rule(k) } else { Ok((false, false)) }
    }

    /// Fait la règle k (sa condition vient de devenir vraie) : son action, ou le lancement de la suite de son bloc (sautée si celle
    /// d'avant tourne encore)
    fn fire_rule(&mut self, k: usize) -> Result<(bool, bool), ()> {
        let line = self.when[k].cond.line;
        if self.when[k].running {
            self.when[k].skipped += 1;
            if self.when[k].skipped == 1 {
                self.note(format!("ligne {line} : condition vraie, mais la suite d'avant n'est pas finie : cette fois est sautée (les suivantes aussi, sans note)"));
            }
            return Ok((false, false));
        }
        self.when[k].count += 1;
        let count = self.when[k].count;
        match (self.when[k].action.clone(), self.when[k].body) {
            (Some(a), _) => self.do_action(&a, line),
            (None, Some((start, end))) => {
                self.when[k].running = true;
                let seq = Seq { pc: start + 1, end, rule: Some(k), quiet: self.quiet || count > 3, ..Default::default() };
                if let Some(s) = self.run_thread(seq) {
                    self.threads.push(s);
                }
                if matches!(self.state, RunState::Failed(_)) { Err(()) } else { Ok((false, false)) }
            }
            _ => Ok((false, false)),
        }
    }

    /// Déroule la suite `seq` d'une règle en bloc jusqu'à sa prochaine attente (rendue, pour la reprendre) ou jusqu'à sa fin (None)
    fn run_thread(&mut self, mut seq: Seq) -> Option<Seq> {
        let mut n = 0usize;
        while self.state == RunState::Running && seq.target.is_none() && seq.pc < seq.end {
            n += 1;
            if n > 100_000 {
                let line = seq.rule.map_or(0, |r| self.when[r].cond.line);
                self.fail(format!("ligne {line} : la suite de la règle « quand » tourne sans fin sans que le temps avance"));
                break;
            }
            self.exec(&mut seq, false);
        }
        let done = seq.target.is_none() && seq.pc >= seq.end;
        if done || matches!(self.state, RunState::Failed(_)) {
            if let Some(r) = seq.rule {
                self.when[r].running = false;
            }
            None
        } else {
            Some(seq)
        }
    }

    /// Règles « quand » après un pas : une règle est faite si sa condition, fausse au relevé précédent, est vraie (le journal note les
    /// trois premières fois). Une pause arrête la tournée (les règles suivantes sont relevées au pas suivant).
    fn check_when(&mut self) {
        self.sync_host();
        let (t, dt) = (self.sim.time, self.sim.last_dt);
        let (mut edited, mut heat) = (false, false);
        for k in 0..self.when.len() {
            let vals = self.var_values();
            let xs = self.when[k].cond.values(&self.sim, &vals);
            let holds = self.when[k].cond.holds(&xs);
            let fire = holds && !self.when[k].on;
            {
                let w = &mut self.when[k];
                w.on = holds;
                w.cond.track(t, dt, &xs);
                if holds {
                    w.cond.near = false;
                }
            }
            if !fire {
                continue;
            }
            let line = self.when[k].cond.line;
            let count = self.when[k].count + 1;
            if !self.when[k].running {
                if count <= 3 {
                    let text = self.when[k].cond.values_text(&xs);
                    self.note(format!("ligne {line} : condition vraie ({text})"));
                } else if count == 4 {
                    self.note(format!("ligne {line} : fois suivantes non notées au journal"));
                }
            }
            self.quiet = count > 3;
            let r = self.fire_rule(k);
            self.quiet = false;
            match r {
                Ok((e, h)) => {
                    edited |= e;
                    heat |= h;
                }
                Err(()) => return,
            }
            if self.state != RunState::Running {
                break;
            }
        }
        self.finish_edits(edited, heat);
    }

    /// Suites parallèles dont l'attente est finie à l'instant `now` (instant atteint, durée permise d'une condition écoulée, ou
    /// reprise après une pause) : déroulées jusqu'à leur attente suivante
    fn run_due_threads(&mut self, now: f64, eps: f64) {
        if !self.threads.iter().any(|s| s.target.map_or(true, |t| t <= now + eps)) {
            return;
        }
        self.sync_host();
        let mut k = 0;
        while k < self.threads.len() {
            if !self.threads[k].target.map_or(true, |t| t <= now + eps) {
                k += 1;
                continue;
            }
            let mut seq = self.threads.remove(k);
            seq.target = None;
            if let Some(mut c) = seq.cond.take() {
                let vals = self.var_values();
                let xs = c.values(&self.sim, &vals);
                self.note(format!("ligne {} : {} non atteint au bout du temps permis ({})", c.line, c.text(), c.values_text(&xs)));
            }
            if let Some(s) = self.run_thread(seq) {
                self.threads.insert(k, s);
                k += 1;
            }
            if self.state != RunState::Running {
                return;
            }
        }
    }

    /// Suites parallèles qui attendent une condition, après un pas : celles dont la condition est vraie reprennent
    fn check_threads(&mut self) {
        if !self.threads.iter().any(|s| s.cond.is_some()) {
            return;
        }
        self.sync_host();
        let (t, dt) = (self.sim.time, self.sim.last_dt);
        let mut k = 0;
        while k < self.threads.len() {
            let vals = self.var_values();
            let (xs, holds) = match &mut self.threads[k].cond {
                Some(c) => {
                    let xs = c.values(&self.sim, &vals);
                    let h = c.holds(&xs);
                    (xs, h)
                }
                None => {
                    k += 1;
                    continue;
                }
            };
            if !holds {
                if let Some(c) = self.threads[k].cond.as_mut() {
                    c.track(t, dt, &xs);
                }
                k += 1;
                continue;
            }
            let mut seq = self.threads.remove(k);
            let c = seq.cond.take().unwrap();
            seq.target = None;
            let q = self.quiet;
            self.quiet = seq.quiet;
            self.note(format!("ligne {} : {} atteint ({})", c.line, c.text(), c.values_text(&xs)));
            self.quiet = q;
            if let Some(s) = self.run_thread(seq) {
                self.threads.insert(k, s);
                k += 1;
            }
            if self.state != RunState::Running {
                return;
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
                Action::Periodic { .. } | Action::When { .. } if line > 0 && self.registered(line) => {}
                Action::Periodic { period, start, end, action } => {
                    // (période infinie : action datée « à t : … », faite une fois)
                    let once = period.is_infinite();
                    let mut next = start.unwrap_or(now);
                    let mut last = end.unwrap_or(f64::INFINITY);
                    if next < now {
                        if once {
                            self.note(format!("ligne {line} : à {} s — instant déjà passé quand la ligne est atteinte : faite maintenant", fmt_num(next)));
                            (next, last) = (now, now);
                        } else {
                            // (début passé : la phase est gardée, première fois à l'échéance suivante)
                            next += ((now - next) / period).ceil() * period;
                        }
                    }
                    if once {
                        self.note(format!("ligne {line} : à {} s : {}", fmt_num(next), describe(action)));
                    } else {
                        let from = start.map_or(String::new(), |s| format!(" à partir de {} s", fmt_num(s)));
                        let until = end.map_or(String::new(), |e| format!(" jusqu'à {} s", fmt_num(e)));
                        self.note(format!("ligne {line} : toutes les {} s{from}{until} : {}", fmt_num(*period), describe(action)));
                    }
                    self.periodic.push(Periodic { action: (**action).clone(), period: *period, next, end: last, line, count: 0 });
                    let eps = 1e-12 * (1.0 + now.abs());
                    let (e, h) = self.fire_due(now, eps)?;
                    edited |= e;
                    heat |= h;
                }
                Action::When { clauses, logic, action } => {
                    let (e, h) = self.arm_rule(line, clauses, logic, Some(action), None)?;
                    edited |= e;
                    heat |= h;
                }
                Action::Mark(t) => {
                    let t = self.interpolate(t);
                    self.note(format!("ligne {line} : repère « {t} »"))
                }
                Action::Deferred { toks, calcs, .. } => {
                    // valeurs de l'instant, puis l'action relue
                    let vals = self.var_values();
                    let mut k = 0;
                    let mut concrete = Vec::with_capacity(toks.len());
                    for t in toks {
                        if let super::mp_script::Tok::Calc(_) = t {
                            let v = calcs[k].eval(&vals);
                            if !v.is_finite() {
                                self.fail(format!("ligne {line} : valeur calculée « {} » = {} : il faut un nombre", calcs[k].text, fmt_num(v)));
                                return Err(());
                            }
                            concrete.push(super::mp_script::Tok::N(v));
                            k += 1;
                        } else {
                            concrete.push(t.clone());
                        }
                    }
                    match super::mp_script::action_from_tokens(&concrete, line, &self.proto.vars) {
                        Ok(a) => {
                            let (e, h) = self.do_action(&a, line)?;
                            edited |= e;
                            heat |= h;
                        }
                        Err(e) => {
                            self.fail(format!("ligne {line} : avec ses valeurs calculées, {e}"));
                            return Err(());
                        }
                    }
                }
                Action::Let { var, value } => {
                    let v = match value {
                        LetValue::Num(x) => *x,
                        LetValue::Calc(c) => c.eval(&self.var_values()),
                        LetValue::Measure(m) => match self.make_probe(m) {
                            Ok(Some((probe, _, _))) => {
                                self.sync_host();
                                probe.value(&self.sim, &self.var_values())
                            }
                            Ok(None) => {
                                self.fail(format!("ligne {line} : « stats » n'est pas une valeur"));
                                return Err(());
                            }
                            Err(e) => {
                                self.fail(format!("ligne {line} : {e}"));
                                return Err(());
                            }
                        },
                    };
                    if let Some(x) = self.vars.get_mut(*var) {
                        *x = v;
                    }
                    let name = self.proto.vars.get(*var).cloned().unwrap_or_default();
                    self.note(format!("ligne {line} : soit {name} = {}", fmt_num(v)));
                }
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
                        let shut = if matches!(other, Action::Valve { .. }) { " fermées" } else { "" };
                        let cells = if other.zone().is_some() { format!(" — {n} mailles{shut}") } else { String::new() };
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
        let msg = self.proto.includes.relabel(&msg);
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
        // suites parallèles (règles en bloc) dont l'attente est finie, à l'heure pile
        self.run_due_threads(now, eps);
        if self.state != RunState::Running {
            return;
        }
        if now >= target - eps {
            self.main.target = None;
            if let Some(mut c) = self.main.cond.take() {
                self.sync_host();
                let vals = self.var_values();
                let xs = c.values(&self.sim, &vals);
                self.note(format!("ligne {} : {} non atteint au bout du temps permis ({})", c.line, c.text(), c.values_text(&xs)));
            }
            return;
        }
        let next = self
            .timed
            .iter()
            .map(|t| t.end)
            .chain(self.periodic.iter().filter(|p| p.next <= p.end).map(|p| p.next))
            .chain(self.threads.iter().filter_map(|s| s.target))
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
                let near = self.main.cond.as_ref().is_some_and(|c| c.near)
                    || self.when.iter().any(|w| w.cond.near)
                    || self.threads.iter().any(|s| s.cond.as_ref().is_some_and(|c| c.near));
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
        if self.main.cond.is_some() {
            self.sync_host();
            let (t, dt) = (self.sim.time, self.sim.last_dt);
            let vals = self.var_values();
            let xs = self.main.cond.as_mut().unwrap().values(&self.sim, &vals);
            if self.main.cond.as_ref().unwrap().holds(&xs) {
                let c = self.main.cond.take().unwrap();
                self.note(format!("ligne {} : {} atteint ({})", c.line, c.text(), c.values_text(&xs)));
                self.main.target = None;
            } else {
                self.main.cond.as_mut().unwrap().track(t, dt, &xs);
            }
        }
        // suites parallèles qui attendent une condition
        self.check_threads();
        if self.state != RunState::Running {
            return;
        }
        // règles « quand » : relevées après chaque pas, l'action est faite quand la condition devient vraie
        if !self.when.is_empty() {
            self.check_when();
            if self.state != RunState::Running {
                return;
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
        let vals = self.var_values();
        for p in &self.probes {
            row.push(p.value(&self.sim, &vals));
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
        for s in std::mem::take(&mut self.threads) {
            if let Some(r) = s.rule {
                let at = self.proto.steps.get(s.pc).map_or(0, |st| st.1);
                let msg = format!("ligne {} : suite de la règle « quand » arrêtée par la fin du protocole (à la ligne {at})", self.when[r].cond.line);
                self.note(msg);
            }
        }
        for k in 0..self.when.len() {
            let w = &self.when[k];
            let skipped = if w.skipped > 0 { format!(", {} fois sautée (suite d'avant en cours)", w.skipped) } else { String::new() };
            let msg = format!("ligne {} : quand {} — {} fois{skipped}", w.cond.line, w.cond.text(), w.count);
            self.note(msg);
        }
        for k in 0..self.periodic.len() {
            let p = &self.periodic[k];
            let msg = if p.period.is_infinite() {
                format!("ligne {} : à {} s — {}", p.line, fmt_num(p.end), if p.count > 0 { "faite" } else { "instant pas atteint" })
            } else {
                format!("ligne {} : toutes les {} s — {} fois", p.line, fmt_num(p.period), p.count)
            };
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
        Action::Body { shape, material, angle, motion, temp, name } => {
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
            let a = match name {
                Some(n) => format!("{a}, nommé « {n} »"),
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
        Action::Periodic { period, start, action, .. } if period.is_infinite() => format!("à {} s : {}", fmt_num(start.unwrap_or(0.0)), describe(action)),
        Action::Periodic { period, action, .. } => format!("toutes les {} s : {}", fmt_num(*period), describe(action)),
        Action::Deferred { .. } => "action aux valeurs calculées".into(),
        Action::Crank(c) => {
            let how = if c.free { "libre".to_string() } else { format!("mené à {} tr/min", fmt_num(c.omega * 60.0 / std::f64::consts::TAU)) };
            let cyl = if c.cylinder.is_some() { ", dans son cylindre" } else { "" };
            format!("vilebrequin {} rayon {} m, bielle {} m, « {} »{cyl} — {how}", fmt_point(c.pivot), fmt_num(c.r), fmt_num(c.l), c.material)
        }
        Action::Let { value, .. } => match value {
            LetValue::Num(v) => format!("variable = {}", fmt_num(*v)),
            LetValue::Calc(c) => format!("variable = « {} »", c.text),
            LetValue::Measure(_) => "variable = un relevé".into(),
        },
        Action::When { clauses, action, .. } => {
            let n = clauses.len();
            let what = if n == 1 { "une comparaison".to_string() } else { format!("{n} comparaisons") };
            format!("quand ({what}) : {}", describe(action))
        }
        Action::Velocity { zone, v } => format!("vitesse {} {} m/s", zone.label(), fmt_point(*v)),
        Action::Ignite { zone, temp } => format!("allumer {} à {} K", zone.label(), fmt_num(*temp)),
        Action::Sink { zone, rate } => format!("puits {} ({} m/s)", zone.label(), fmt_num(*rate)),
        Action::Emitter { zone, fluid, .. } => format!("émetteur de « {fluid} » {}", zone.label()),
        Action::Grille { zone, .. } => format!("grille {}", zone.label()),
        Action::Pump { zone, v } => format!("pompe {} à {} m/s", zone.label(), fmt_point(*v)),
        Action::Teleporter { zone, dest, speed, radius } => {
            format!("téléporteur {} vers {} ({} m/s, rayon {} m)", zone.label(), fmt_point(*dest), fmt_num(*speed), fmt_num(*radius))
        }
        Action::Valve { zone, open } => match *open {
            o if o <= 0.0 => format!("vanne {} fermée", zone.label()),
            o if o >= 1.0 => format!("vanne {} ouverte", zone.label()),
            o => format!("vanne {} ouverte à {} %", zone.label(), fmt_num((o * 1000.0).round() / 10.0)),
        },
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
    let protos = match super::mp_script::parse_all_in(&text, file.parent().unwrap_or(Path::new(""))) {
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
