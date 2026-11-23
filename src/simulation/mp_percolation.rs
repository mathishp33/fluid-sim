//! Percolation : un fluide traverse les PORES d'un lit de grains tassé (les grains restent en place) et les GRILLES (cellules qui arrêtent
//! les grains : le tas y repose, l'air arrive par-dessous). Sans elle, le modèle à une vitesse par cellule fait d'un lit tassé une paroi
//! étanche (le frottement le rend rigide, et le gaz des pores bouge avec les grains) : un feu de charbon ne tire pas.
//!
//! Loi d'Ergun (Darcy–Forchheimer), q = débit de fluide par unité de surface (vitesse superficielle, m/s), relatif aux grains immobiles :
//!   −∂p_f/∂n + ρ_f g_n = (μ/K) q + b |q| q,   K = d² ε³ / (150 (1 − ε)²),   b = 1.75 ρ_f (1 − ε) / (ε³ d)
//! (ε : porosité, d : diamètre des grains — ou des trous d'une grille). Pour du sable de 0.3 mm, K ≈ 1e-10 m² : il ne passe presque
//! rien (un tas de poudre fine ne respire pas : lits de grains de moins de 1 mm traités comme étanches, voir `PERC_D_MIN`) ; pour du
//! charbon en morceaux de 1 cm, K ≈ 1e-7 m² : le tirage d'un feu.
//!
//! La pression du fluide des pores est p_f = p − p_s, p_s = poids déjaugé des grains au-dessus (colonne lithostatique, sur les chaînes
//! de grains qui reposent sur un appui) : au repos, p_f est hydrostatique et rien ne coule (le poids des grains ne pousse pas le gaz des
//! pores).
//!
//! Fluidisation : la traînée du fluide qui traverse un lit posé sur un appui (grille, fond, solide) passe par les contacts des grains
//! jusqu'à l'appui, dans la limite du poids déjaugé du lit (`fluidization_force` : force rendue au mélange, la pression du mélange
//! portant déjà tout le poids) — un lit soufflé par en dessous reste immobile tant que la traînée d'Ergun est inférieure à son poids ;
//! au-delà, l'excédent le soulève : il se FLUIDISE à la vitesse minimale d'Ergun. Un lit décollé de son appui n'a plus de contacts :
//! la percolation n'y compte pas de poids lithostatique, son poids agit, il retombe si la traînée est plus faible. Le frottement voit
//! la pression de contact déchargée de la traînée (`friction_geometry`) : un lit fluidisé coule comme un liquide.
//!
//! Couplage : un flux de Darcy explicite serait instable (la pression d'un lit perméable diffuse avec D ~ K p/(μ ε) ~ 500 m²/s) ; il est
//! donc résolu DANS la projection : chaque face d'un milieu poreux ajoute la conductance C/dx² au système de pression, et la divergence de
//! q au second membre (`solve_pressure`). Le flux obtenu transporte, au pas suivant et avec la vitesse du mélange, le seul FLUIDE de la
//! cellule amont (`percolate`) ; la fermeture en tient compte (∇·(u + q)). Le travail de pression aussi.

use super::mp_cells::MpEffect;
use super::multiphase::MultiphaseSim;

/// Ergun : coefficients des termes visqueux et inertiel
const ERGUN_A: f64 = 150.0;
const ERGUN_B: f64 = 1.75;
/// diamètre minimal des grains d'un lit perméable (m). En dessous (sable, poudres), le gaz qui traverse un tas immobile est négligeable
/// (K < 1e-9 m²) et le lit est traité comme étanche : la percolation y alignerait la pression sur le poids vertical des grains, ce qui
/// raidit le frottement (contrainte moyenne surestimée — un tas de sable à 50° s'arrêtait à 42° au lieu de 30°)
const PERC_D_MIN: f64 = 1e-3;
/// part minimale du poids d'une cellule portée par la traînée de percolation pour que le lit soit retenu par son appui (faces fermées
/// au mélange, `hold_faces`) : en dessous (tirage naturel d'un braséro : 0.05 %), le lit garde sa dynamique (avalanches)
pub(crate) const HOLD_DRAG: f64 = 0.01;

/// conductances (m²/(Pa·s)) et termes moteurs (m/s) de Darcy par face : q = −C (p_b − p_a)/dx + d, positif de a (gauche / bas) vers b
pub(crate) struct DarcyFaces {
    pub cu: Vec<f64>,
    pub du: Vec<f64>,
    pub cv: Vec<f64>,
    pub dv: Vec<f64>,
}

/// milieu poreux d'une cellule : perméabilité K (m²), porosité ε, diamètre des grains ou des trous d (m), fluide (ρ, μ)
#[derive(Clone, Copy, Debug)]
struct Porous {
    k: f64,
    eps: f64,
    d: f64,
    rho: f64,
    mu: f64,
}

impl Porous {
    /// conductance K_eff/μ (m²/(Pa·s)) pour un gradient moteur |G| (Pa/m) : racine de b q² + (μ/K) q = G
    fn conductance(&self, g: f64) -> f64 {
        let a = self.mu / self.k;
        let b = ERGUN_B * self.rho * (1.0 - self.eps) / (self.eps.powi(3) * self.d);
        2.0 / (a + (a * a + 4.0 * b * g.abs()).sqrt())
    }
}

fn kozeny(d: f64, eps: f64) -> f64 {
    d * d * eps.powi(3) / (ERGUN_A * (1.0 - eps).powi(2))
}

impl MultiphaseSim {
    /// cellule grille (laisse passer le fluide, arrête les grains)
    #[inline]
    pub(crate) fn is_grille(&self, c: usize) -> bool {
        matches!(self.special[c], Some(MpEffect::Grille { .. }))
    }

    /// (ρ, μ) du fluide (phases non granulaires) de la cellule ; (0, 0) sans fluide
    fn pore_fluid(&self, c: usize) -> (f64, f64) {
        let (mut a, mut m, mut mu) = (0.0, 0.0, 0.0);
        for (k, ph) in self.reg.phases.iter().enumerate() {
            if ph.is_granular() {
                continue;
            }
            let ak = self.alpha[k][c].max(0.0);
            if ak <= 0.0 {
                continue;
            }
            a += ak;
            m += self.mass[k][c].max(0.0);
            mu += ak * ph.mu(self.phase_temp(k, c));
        }
        if a <= 1e-12 { (0.0, 0.0) } else { (m / a, mu / a) }
    }

    /// milieu poreux de la cellule : grille, ou grains (de `PERC_D_MIN` et plus) EN CONTACT — tassés, ou posés sur un appui (cellule de
    /// surface d'un lit, partiellement remplie : sans cela le frottement la rend rigide et elle scelle le lit) ; `contact` vient de
    /// `grain_contact`. None sinon (fluide libre, solide, épinglée)
    fn porous_cell_with(&self, c: usize, contact: &[bool]) -> Option<Porous> {
        if self.is_solid(c) || self.pinned(c) {
            return None;
        }
        if let Some(MpEffect::Grille { hole, open }) = self.special[c] {
            let (rho, mu) = self.pore_fluid(c);
            let (d, eps) = (hole.max(1e-5), open.clamp(0.05, 0.95));
            return (mu > 0.0).then_some(Porous { k: kozeny(d, eps), eps, d, rho, mu });
        }
        let (mut a_s, mut ad, mut am) = (0.0, 0.0, 0.0);
        for (k, ph) in self.reg.phases.iter().enumerate() {
            if ph.is_granular() {
                let a = self.alpha[k][c].max(0.0);
                a_s += a;
                ad += a * ph.grain_diameter.max(1e-6);
                am += a * ph.alpha_max;
            }
        }
        let _ = am;
        if a_s <= 1e-6 || !contact[c] || ad / a_s < PERC_D_MIN {
            return None;
        }
        let (rho, mu) = self.pore_fluid(c);
        let eps = (1.0 - a_s).clamp(0.05, 0.95);
        let d = ad / a_s;
        (mu > 0.0).then_some(Porous { k: kozeny(d, eps), eps, d, rho, mu })
    }

    /// milieu poreux de la cellule c (contact recalculé : diagnostics et tests)
    #[allow(dead_code)]
    pub(crate) fn porous_cell(&self, c: usize) -> Option<Porous> {
        let (_, contact) = self.grain_contact();
        self.porous_cell_with(c, &contact)
    }

    /// pression de contact lithostatique p_s (Pa) : poids déjaugé des grains au-dessus, le long de la gravité (axe dominant), sur les
    /// colonnes de grains en contact (chaînes de grains qui reposent sur un appui : le fond, un solide, une grille, ou des grains tassés
    /// eux-mêmes en contact)
    pub(crate) fn grain_overburden(&self) -> Vec<f64> {
        self.grain_contact().0
    }

    /// Traînée du fluide de percolation sur les grains de chaque cellule, le long de la gravité (axe dominant), comptée positive vers le
    /// haut (contre la pesanteur), par unité de surface de la cellule (Pa) : F = [(μ/K) q + b |q| q] dx, q = moyenne des flux de Darcy de
    /// la dernière projection sur les deux faces de la cellule le long de l'axe, perméabilité et fluide des grains de la cellule (Ergun,
    /// comme `Porous::conductance`). Nulle hors des lits perméables (grains de moins de `PERC_D_MIN`, grilles, cellules sans grains).
    pub(crate) fn percolation_drag(&self) -> Vec<f64> {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let n = nx * ny;
        let mut f = vec![0.0; n];
        let gn = (self.gravity[0].powi(2) + self.gravity[1].powi(2)).sqrt();
        if !self.percolation || gn == 0.0 || self.q_u.len() != (nx + 1) * ny || self.q_v.len() != nx * (ny + 1) {
            return f;
        }
        if !self.q_u.iter().chain(self.q_v.iter()).any(|&q| q != 0.0) {
            return f;
        }
        let vertical = self.gravity[1].abs() >= self.gravity[0].abs();
        let down = if vertical { self.gravity[1] < 0.0 } else { self.gravity[0] < 0.0 };
        let up = if down { 1.0 } else { -1.0 };
        for c in 0..n {
            if self.is_solid(c) || self.pinned(c) || self.is_grille(c) {
                continue;
            }
            let (mut a_s, mut ad) = (0.0, 0.0);
            for (k, ph) in self.reg.phases.iter().enumerate() {
                if ph.is_granular() {
                    let a = self.alpha[k][c].max(0.0);
                    a_s += a;
                    ad += a * ph.grain_diameter.max(1e-6);
                }
            }
            if a_s <= 1e-6 || ad / a_s < PERC_D_MIN {
                continue;
            }
            let (i, j) = (c % nx, c / nx);
            let qc = if vertical { 0.5 * (self.q_v[j * nx + i] + self.q_v[(j + 1) * nx + i]) } else { 0.5 * (self.q_u[j * (nx + 1) + i] + self.q_u[j * (nx + 1) + i + 1]) };
            if qc == 0.0 {
                continue;
            }
            let (rho, mu) = self.pore_fluid(c);
            if mu <= 0.0 {
                continue;
            }
            let (d, eps) = (ad / a_s, (1.0 - a_s).clamp(0.05, 0.95));
            let q = up * qc;
            let b = ERGUN_B * rho * (1.0 - eps) / (eps.powi(3) * d);
            f[c] = (mu / kozeny(d, eps) * q + b * q.abs() * q) * dx;
        }
        f
    }

    /// Force de décharge des contacts (N/m³, dans le sens de la gravité) : la part de la traînée de percolation (`percolation_drag`)
    /// que les contacts des grains transmettent à l'appui, cumulée depuis le haut de chaque colonne en contact et plafonnée au poids
    /// déjaugé cumulé — f = [min(D + F, L + w) − min(D, L)] / dx. La pression du mélange porte déjà tout le poids du lit (p_s
    /// lithostatique de la percolation) : sans cette force, la traînée soulevait le lit quel que soit son poids. Au-delà du poids
    /// (vitesse minimale de fluidisation), l'excédent de traînée n'est plus repris : il soulève les grains.
    pub(crate) fn fluidization_force(&self) -> Vec<f64> {
        self.fluidization_state().0
    }

    /// (force de décharge des contacts `fluidization_force`, cellules RETENUES par leur appui) : une cellule est retenue quand elle est
    /// en contact, percolée, que le fluide y porte au moins `HOLD_DRAG` de son poids et que la traînée cumulée depuis le haut de la colonne
    /// reste sous le poids déjaugé cumulé (pas fluidisée). Les faces entre deux cellules retenues, le long de la gravité, sont fermées au
    /// mélange (`hold_faces`, comme celles d'une grille : le fluide y passe par percolation) : un lit soufflé sous la vitesse minimale de
    /// fluidisation ne bouge pas — sans cela, la force de décharge, calculée sur la traînée du pas précédent, laissait passer la poussée
    /// d'un changement de débit et rien ne ramenait le lit sur son appui.
    pub(crate) fn fluidization_state(&self) -> (Vec<f64>, Vec<bool>) {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let n = nx * ny;
        let drag = self.percolation_drag();
        let mut fc = vec![0.0; n];
        let mut held = vec![false; n];
        if !drag.iter().any(|&d| d != 0.0) {
            return (fc, held);
        }
        let (_, _, contact) = self.grain_support();
        let gn = (self.gravity[0].powi(2) + self.gravity[1].powi(2)).sqrt();
        let vertical = self.gravity[1].abs() >= self.gravity[0].abs();
        let down = if vertical { self.gravity[1] < 0.0 } else { self.gravity[0] < 0.0 };
        let (lines, len) = if vertical { (nx, ny) } else { (ny, nx) };
        let at = |l: usize, s: usize| if vertical { l + s * nx } else { s + l * nx };
        for l in 0..lines {
            let (mut cd, mut cl) = (0.0f64, 0.0f64);
            // du haut vers le bas
            for tt in 0..len {
                let s = if down { len - 1 - tt } else { tt };
                let c = at(l, s);
                if !contact[c] || self.is_solid(c) || self.is_grille(c) {
                    cd = 0.0;
                    cl = 0.0;
                    continue;
                }
                let (mut a_s, mut ms) = (0.0, 0.0);
                for (k, ph) in self.reg.phases.iter().enumerate() {
                    if ph.is_granular() {
                        a_s += self.alpha[k][c].max(0.0);
                        ms += self.mass[k][c].max(0.0);
                    }
                }
                let w = if a_s > 1e-6 { (ms / a_s - self.pore_fluid(c).0).max(0.0) * a_s * gn * dx } else { 0.0 };
                fc[c] = ((cd + drag[c]).min(cl + w) - cd.min(cl)) / dx;
                held[c] = drag[c] > HOLD_DRAG * w && cd + 0.5 * drag[c] < cl + 0.5 * w;
                cd += drag[c];
                cl += w;
            }
        }
        (fc, held)
    }

    /// Faces entre deux cellules retenues par leur appui (`fluidization_state`), dans les deux directions : (faces u, faces v), à fermer
    /// au mélange — le lit retenu est un bloc poreux immobile (une face latérale ouverte laissait les grains glisser d'une colonne retenue
    /// à l'autre et le lit se tasser au-delà de son empilement maximal, le fluide chassé des pores)
    pub(crate) fn hold_faces(&self) -> Option<(Vec<bool>, Vec<bool>)> {
        if !self.percolation || !self.has_granular() {
            return None;
        }
        let (_, held) = self.fluidization_state();
        if !held.iter().any(|&h| h) {
            return None;
        }
        let (nx, ny) = (self.nx, self.ny);
        let (mut hu, mut hv) = (vec![false; (nx + 1) * ny], vec![false; nx * (ny + 1)]);
        for j in 1..ny {
            for i in 0..nx {
                hv[j * nx + i] = held[self.idx(i, j - 1)] && held[self.idx(i, j)];
            }
        }
        for j in 0..ny {
            for i in 1..nx {
                hu[j * (nx + 1) + i] = held[self.idx(i - 1, j)] && held[self.idx(i, j)];
            }
        }
        Some((hu, hv))
    }

    /// (p_s, contact) de chaque cellule (voir `grain_overburden`)
    pub(crate) fn grain_contact(&self) -> (Vec<f64>, Vec<bool>) {
        let (ps, contact, _) = self.grain_support();
        (ps, contact)
    }

    /// (p_s, contact, appui) de chaque cellule : contact = grains tassés, ou posés sur un appui (milieu poreux : la percolation les
    /// traverse) ; appui = chaîne de grains qui repose sur un appui (le fond, un solide, une grille, ou des grains tassés eux-mêmes sur
    /// un appui) : seules ces colonnes portent le poids lithostatique p_s et transmettent la traînée à l'appui — un paquet tassé décollé
    /// (lit soulevé, paquet qui tombe) est perméable, sans poids porté par des contacts
    pub(crate) fn grain_support(&self) -> (Vec<f64>, Vec<bool>, Vec<bool>) {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let n = nx * ny;
        let mut ps = vec![0.0; n];
        let mut contact_c = vec![false; n];
        let mut support_c = vec![false; n];
        let gn = (self.gravity[0].powi(2) + self.gravity[1].powi(2)).sqrt();
        if gn == 0.0 {
            // sans gravité : pas de poids ; « contact » = grains tassés
            for c in 0..n {
                if self.is_solid(c) || self.is_grille(c) {
                    continue;
                }
                let (mut a_s, mut am) = (0.0, 0.0);
                for (k, ph) in self.reg.phases.iter().enumerate() {
                    if ph.is_granular() {
                        a_s += self.alpha[k][c].max(0.0);
                        am += self.alpha[k][c].max(0.0) * ph.alpha_max;
                    }
                }
                contact_c[c] = a_s > 1e-6 && a_s >= am / a_s - 0.1;
            }
            return (ps, contact_c, support_c);
        }
        let np = self.reg.phases.len();
        // poids déjaugé w et grains tassés
        let mut w = vec![0.0; n];
        let mut packed = vec![false; n];
        let mut has = vec![false; n];
        for c in 0..n {
            if self.is_solid(c) || self.is_grille(c) {
                continue;
            }
            let (mut a_s, mut ms, mut am) = (0.0, 0.0, 0.0);
            for k in 0..np {
                if self.reg.phases[k].is_granular() {
                    let a = self.alpha[k][c].max(0.0);
                    a_s += a;
                    ms += self.mass[k][c].max(0.0);
                    am += a * self.reg.phases[k].alpha_max;
                }
            }
            if a_s <= 1e-6 {
                continue;
            }
            let (rho_f, _) = self.pore_fluid(c);
            w[c] = (ms / a_s - rho_f).max(0.0) * a_s * gn * dx;
            packed[c] = a_s >= am / a_s - 0.1;
            has[c] = true;
        }
        let vertical = self.gravity[1].abs() >= self.gravity[0].abs();
        let down = if vertical { self.gravity[1] < 0.0 } else { self.gravity[0] < 0.0 };
        let (lines, len) = if vertical { (nx, ny) } else { (ny, nx) };
        let at = |l: usize, s: usize| if vertical { l + s * nx } else { s + l * nx };
        for l in 0..lines {
            // du bas vers le haut : cellules « posées » (en contact)
            let mut contact = vec![false; len];
            for t in 0..len {
                let s = if down { t } else { len - 1 - t };
                let c = at(l, s);
                if !has[c] {
                    continue;
                }
                let support = if t == 0 {
                    true
                } else {
                    let b = at(l, if down { s - 1 } else { s + 1 });
                    self.is_solid(b) || self.is_grille(b) || (has[b] && contact[t - 1] && packed[b])
                };
                // contact = appui : chaîne de grains posée (une cellule tassée décollée de son appui — lit soulevé, paquet qui tombe —
                // n'est ni poreuse ni porteuse : la percolation y laissait le mélange se resserrer sans limite en chassant le fluide des
                // pores, α jusqu'à 1)
                contact[t] = support;
                contact_c[c] = support;
                support_c[c] = support;
            }
            // du haut vers le bas : cumul du poids sur les colonnes en contact
            let mut acc = 0.0;
            for t in (0..len).rev() {
                let s = if down { t } else { len - 1 - t };
                let c = at(l, s);
                if !has[c] || !contact[t] {
                    acc = 0.0;
                    continue;
                }
                ps[c] = acc + 0.5 * w[c];
                acc += w[c];
            }
        }
        (ps, contact_c, support_c)
    }

    /// Conductances et termes moteurs de Darcy des faces (voir l'en-tête), avec la correction d'Ergun évaluée sur la pression du pas
    /// précédent ; None s'il n'y a ni lit tassé ni grille
    pub(crate) fn darcy_faces(&self) -> Option<DarcyFaces> {
        if !self.percolation {
            return None;
        }
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let n = nx * ny;
        let (ps, contact) = self.grain_contact();
        let por: Vec<Option<Porous>> = (0..n).map(|c| self.porous_cell_with(c, &contact)).collect();
        if !por.iter().any(|p| p.is_some()) {
            return None;
        }
        let rho_f: Vec<f64> = (0..n).map(|c| if self.is_solid(c) { 0.0 } else { self.pore_fluid(c).0 }).collect();
        let face = |a: usize, b: usize, g: f64| -> (f64, f64) {
            let (pa, pb) = (por[a], por[b]);
            if (pa.is_none() && pb.is_none()) || self.is_solid(a) || self.is_solid(b) || self.pinned(a) || self.pinned(b) {
                return (0.0, 0.0);
            }
            let rf = match (rho_f[a] > 0.0, rho_f[b] > 0.0) {
                (true, true) => 0.5 * (rho_f[a] + rho_f[b]),
                (true, false) => rho_f[a],
                _ => rho_f[b],
            };
            // moteur : −(∂p/∂n − ∂p_s/∂n) + ρ_f g_n
            let drive = (ps[b] - ps[a]) + rf * g * dx;
            let gm = ((self.p[b] - self.p[a]) - drive).abs() / dx;
            let inv = |p: Option<Porous>| p.map_or(0.0, |q| 1.0 / q.conductance(gm));
            let c = 2.0 / (inv(pa) + inv(pb));
            (c, c * drive / dx)
        };
        let mut df = DarcyFaces { cu: vec![0.0; (nx + 1) * ny], du: vec![0.0; (nx + 1) * ny], cv: vec![0.0; nx * (ny + 1)], dv: vec![0.0; nx * (ny + 1)] };
        for j in 0..ny {
            for i in 1..nx {
                let f = j * (nx + 1) + i;
                (df.cu[f], df.du[f]) = face(self.idx(i - 1, j), self.idx(i, j), self.gravity[0]);
            }
        }
        for j in 1..ny {
            for i in 0..nx {
                let f = j * nx + i;
                (df.cv[f], df.dv[f]) = face(self.idx(i, j - 1), self.idx(i, j), self.gravity[1]);
            }
        }
        Some(df)
    }

    /// flux de Darcy (m/s) des faces pour la pression p
    pub(crate) fn darcy_flux(&self, df: &DarcyFaces, p: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let mut qu = vec![0.0; (nx + 1) * ny];
        let mut qv = vec![0.0; nx * (ny + 1)];
        for j in 0..ny {
            for i in 1..nx {
                let f = j * (nx + 1) + i;
                if df.cu[f] > 0.0 {
                    qu[f] = -df.cu[f] * (p[j * nx + i] - p[j * nx + i - 1]) / dx + df.du[f];
                }
            }
        }
        for j in 1..ny {
            for i in 0..nx {
                let f = j * nx + i;
                if df.cv[f] > 0.0 {
                    qv[f] = -df.cv[f] * (p[j * nx + i] - p[(j - 1) * nx + i]) / dx + df.dv[f];
                }
            }
        }
        (qu, qv)
    }

    /// part fluide (phases non granulaires) de la cellule
    fn fluid_fraction_of(&self, c: usize) -> f64 {
        (0..self.reg.phases.len()).filter(|&k| !self.reg.phases[k].is_granular()).map(|k| self.alpha[k][c].max(0.0)).sum()
    }

    /// vitesse équivalente pour la borne de pas (dt ≤ 0.5 dx / w) : w = 1.25 |q| / (part fluide de la cellule amont), donc |q| dt ≤ 0.4
    /// de son fluide par face
    pub(crate) fn max_darcy_speed(&self) -> f64 {
        let (nx, ny) = (self.nx, self.ny);
        let mut w = 0.0f64;
        let mut look = |q: f64, a: usize, b: usize| {
            if q != 0.0 {
                let d = if q > 0.0 { a } else { b };
                w = w.max(1.25 * q.abs() / self.fluid_fraction_of(d).max(0.05));
            }
        };
        for j in 0..ny {
            for i in 1..nx {
                look(self.q_u[j * (nx + 1) + i], j * nx + i - 1, j * nx + i);
            }
        }
        for j in 1..ny {
            for i in 0..nx {
                look(self.q_v[j * nx + i], (j - 1) * nx + i, j * nx + i);
            }
        }
        w
    }

    /// Transport du FLUIDE par les flux de Darcy de la dernière projection (self.q_u, self.q_v) pendant dt : à travers chaque face, le
    /// volume |q| dt/dx du fluide de la cellule amont (sa composition, ses températures) passe dans l'autre ; les grains ne bougent pas.
    /// Sortie de chaque cellule bornée à 90 % de son fluide (évaluée sur l'état d'avant l'opérateur, flux simultanés).
    pub(crate) fn percolate(&mut self, dt: f64) {
        if !self.q_u.iter().chain(self.q_v.iter()).any(|&q| q != 0.0) {
            return;
        }
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let n = nx * ny;
        let np = self.reg.phases.len();
        let fluid: Vec<bool> = self.reg.phases.iter().map(|p| !p.is_granular()).collect();
        let af: Vec<f64> = (0..n).map(|c| if self.is_solid(c) { 0.0 } else { self.fluid_fraction_of(c) }).collect();
        // (donneuse, receveuse, volume) des faces
        let mut moves: Vec<(usize, usize, f64)> = vec![];
        for j in 0..ny {
            for i in 1..nx {
                let q = self.q_u[j * (nx + 1) + i];
                if q != 0.0 {
                    let (a, b) = (j * nx + i - 1, j * nx + i);
                    let (d, r) = if q > 0.0 { (a, b) } else { (b, a) };
                    moves.push((d, r, q.abs() * dt / dx));
                }
            }
        }
        for j in 1..ny {
            for i in 0..nx {
                let q = self.q_v[j * nx + i];
                if q != 0.0 {
                    let (a, b) = ((j - 1) * nx + i, j * nx + i);
                    let (d, r) = if q > 0.0 { (a, b) } else { (b, a) };
                    moves.push((d, r, q.abs() * dt / dx));
                }
            }
        }
        let mut out = vec![0.0; n];
        for &(d, _, v) in &moves {
            out[d] += v;
        }
        let scale: Vec<f64> = (0..n).map(|c| if out[c] > 0.9 * af[c] { 0.9 * af[c] / out[c] } else { 1.0 }).collect();
        let (mass0, tg0) = (self.mass.clone(), self.tg.clone());
        let mut touched = vec![false; n];
        for (d, r, v) in moves {
            let vol = v * scale[d];
            if vol <= 0.0 || af[d] <= 1e-12 {
                continue;
            }
            for k in 0..np {
                if !fluid[k] || mass0[k][d] <= 0.0 {
                    continue;
                }
                let g = self.group_of[k];
                let dm = vol * mass0[k][d] / af[d];
                let de = dm * self.reg.phases[k].e_int(tg0[g][d]);
                self.mass[k][d] -= dm;
                self.mass[k][r] += dm;
                self.eg[g][d] -= de;
                self.eg[g][r] += de;
            }
            touched[d] = true;
            touched[r] = true;
        }
        for c in 0..n {
            if touched[c] {
                self.cell_sync(c);
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::mp_cells::{MpEffect, MpInlet};
    use super::super::multiphase::Bc;
    use super::super::phases::{Phase, Registry};
    use super::*;

    /// colonne d'air (12 × 40 mailles d'1 cm) : grille à y ∈ [4, 5) cm, lit de charbon en morceaux de `d` m tassé de 5 à 25 cm posé
    /// dessus, air soufflé par le fond à `vin` m/s (0 : fond fermé), haut ouvert ; ni chimie ni chaleur (293 K)
    pub(crate) fn bed_on_grille(d: f64, vin: f64) -> (MultiphaseSim, usize, usize) {
        let mut r = Registry::new();
        let air = r.add_phase(Phase::air());
        let mut coal = Phase::coal();
        coal.grain_diameter = d;
        let amax = coal.alpha_max;
        let c = r.add_phase(coal);
        let mut s = MultiphaseSim::new(12, 40, 0.01, r, air);
        s.surface_tension = false;
        s.paint_mixture(move |_, y| {
            let mut m = vec![0.0; 2];
            if y > 0.05 && y < 0.25 {
                m[c] = amax;
                m[air] = 1.0 - amax;
            } else {
                m[air] = 1.0;
            }
            Some(m)
        });
        s.finalize_init(|_, _| 293.15);
        for i in 0..12 {
            let q = s.idx(i, 4);
            s.set_effect(q, Some(MpEffect::Grille { hole: 0.01, open: 0.6 }));
        }
        if vin > 0.0 {
            s.inlet[2] = Some(MpInlet { vx: 0.0, vy: vin, phase: air, temp: 293.15, offset: 0 });
            s.bc[2] = Bc::Wall;
        }
        s.bc[3] = Bc::Open { p: 101325.0 };
        s.refresh();
        (s, air, c)
    }

    /// moyenne de q_v sur les faces horizontales des rangées j0..j1 (colonnes intérieures)
    fn mean_qv(s: &MultiphaseSim, j0: usize, j1: usize) -> f64 {
        let mut a = vec![];
        for j in j0..j1 {
            for i in 2..s.nx - 2 {
                a.push(s.q_v[j * s.nx + i]);
            }
        }
        a.iter().sum::<f64>() / a.len() as f64
    }

    #[test]
    fn air_blown_under_a_grate_percolates_through_a_coal_bed_at_the_ergun_pressure_drop() {
        // charbon en morceaux d'1 cm (ε = 0.4) : à 0.3 m/s, Ergun donne (μ/K) q + b q² ≈ 46 + 177 Pa/m ; le lit ne bouge pas
        let (mut s, _air, c) = super::tests::bed_on_grille(0.01, 0.3);
        let col0: Vec<f64> = (0..s.ny).map(|j| s.alpha[c][s.idx(6, j)]).collect();
        let m0 = s.total_mass(c);
        while s.time < 1.0 && !s.diverged {
            s.step();
        }
        assert!(!s.diverged && s.state_is_sane());
        let q = mean_qv(&s, 8, 22);
        // perte de charge : pression du fluide des pores p_f = p − p_s, rangées 8 et 20 (centres), moyenne sur les colonnes intérieures
        let ps = s.grain_overburden();
        let pf = |j: usize| (2..s.nx - 2).map(|i| { let q = s.idx(i, j); s.p[q] - ps[q] }).sum::<f64>() / (s.nx - 4) as f64;
        let (j1, j2) = (8, 20);
        let l = (j2 - j1) as f64 * s.dx;
        let rho = Phase::air().rho(101325.0, 293.15);
        let g = (pf(j1) - pf(j2)) / l - rho * 9.81;
        let por = s.porous_cell(s.idx(6, 14)).unwrap();
        let ergun = por.mu / por.k * q + ERGUN_B * por.rho * (1.0 - por.eps) / (por.eps.powi(3) * por.d) * q * q;
        let col1: Vec<f64> = (0..s.ny).map(|j| s.alpha[c][s.idx(6, j)]).collect();
        let moved = col0.iter().zip(&col1).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        let below: f64 = (0..4 * s.nx).map(|q| s.mass[c][q]).sum();
        eprintln!("percolation : q = {q:.4} m/s (entrée 0.3) ; perte de charge {g:.1} Pa/m (Ergun {ergun:.1}) ; grains déplacés {moved:.2e} ; sous la grille {below:.2e} ; K = {:.3e} m²", por.k);
        assert!((q / 0.3 - 1.0).abs() < 0.1, "débit à travers le lit {q}");
        assert!((g / ergun - 1.0).abs() < 0.15, "perte de charge {g} (Ergun {ergun})");
        assert!(moved < 0.02 && below < 1e-9, "le lit a bougé : {moved}, sous la grille {below}");
        assert!((s.total_mass(c) / m0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_resting_bed_does_not_leak_and_fine_powder_is_impermeable() {
        // sans soufflage : au repos, aucun écoulement de percolation (p_f hydrostatique) ; perméabilités : lumps 1 cm ≫ poudre 0.3 mm
        let (mut s, _, _) = bed_on_grille(0.01, 0.0);
        while s.time < 0.5 && !s.diverged {
            s.step();
        }
        assert!(!s.diverged);
        let qmax = s.q_u.iter().chain(s.q_v.iter()).fold(0.0f64, |a, &b| a.max(b.abs()));
        eprintln!("lit au repos : |q| max {qmax:.2e} m/s");
        assert!(qmax < 2e-3, "fuite de percolation au repos : {qmax}");
        // perméabilité des morceaux d'1 cm (Kozeny–Carman, ε = 0.4) ; la poudre de 0.3 mm (K ≈ 1e-10 m²) est traitée comme étanche
        let (lump, _, _) = bed_on_grille(0.01, 0.0);
        let (powder, _, _) = bed_on_grille(3e-4, 0.0);
        let kl = lump.porous_cell(lump.idx(6, 14)).unwrap().k;
        assert!((kl / kozeny(0.01, 0.4) - 1.0).abs() < 1e-6, "K = {kl}");
        assert!(powder.porous_cell(powder.idx(6, 14)).is_none() && kozeny(3e-4, 0.4) < 1e-9, "poudre fine perméable");
    }
}

#[cfg(test)]
pub(crate) mod fluidization_tests {
    use super::super::mp_cells::{MpEffect, MpInlet};
    use super::super::multiphase::{Bc, MultiphaseSim};
    use super::super::phases::{Phase, Registry};
    use super::{kozeny, ERGUN_B};

    /// cuve d'air (nx × ny mailles d'1 cm) : grille au rang 2, lit de sable de 2 mm (2650 kg/m³) tassé sur h mailles posé dessus, air
    /// soufflé par le fond à `vin` m/s, haut ouvert ; 293 K
    pub(crate) fn blown_sand_bed(nx: usize, ny: usize, h: usize, vin: f64) -> (MultiphaseSim, usize) {
        let mut r = Registry::new();
        let air = r.add_phase(Phase::air());
        let mut sand = Phase::sand();
        sand.grain_diameter = 2e-3;
        let amax = sand.alpha_max;
        let c = r.add_phase(sand);
        let mut s = MultiphaseSim::new(nx, ny, 0.01, r, air);
        s.surface_tension = false;
        let (y0, y1) = (0.03, 0.03 + h as f64 * 0.01);
        s.paint_mixture(move |_, y| {
            let mut m = vec![0.0; 2];
            if y > y0 && y < y1 {
                m[c] = amax;
                m[air] = 1.0 - amax;
            } else {
                m[air] = 1.0;
            }
            Some(m)
        });
        s.finalize_init(|_, _| 293.15);
        for i in 0..nx {
            let q = s.idx(i, 2);
            s.set_effect(q, Some(MpEffect::Grille { hole: 0.002, open: 0.4 }));
        }
        s.inlet[2] = Some(MpInlet { vx: 0.0, vy: vin, phase: air, temp: 293.15, offset: 0 });
        s.bc[2] = Bc::Wall;
        s.bc[3] = Bc::Open { p: 101325.0 };
        s.refresh();
        (s, c)
    }

    /// (a, b) d'Ergun du lit tassé (ε = 0.4, d = 2 mm, air) : gradient a q + b q², et poids déjaugé par unité de volume (N/m³)
    fn ergun_and_weight() -> (f64, f64, f64) {
        let air = Phase::air();
        let (rho, mu) = (air.rho(101325.0, 293.15), air.mu(293.15));
        let (d, eps) = (2e-3, 0.4);
        let a = mu / kozeny(d, eps);
        let b = ERGUN_B * rho * (1.0 - eps) / (eps.powi(3) * d);
        (a, b, (1.0 - eps) * (Phase::sand().rho(101325.0, 293.15) - rho) * 9.81)
    }

    /// vitesse minimale de fluidisation d'Ergun du lit de sable de 2 mm dans l'air : a U + b U² = poids déjaugé (1.08 m/s)
    pub(crate) fn ergun_umf() -> f64 {
        let (a, b, w) = ergun_and_weight();
        (-a + (a * a + 4.0 * b * w).sqrt()) / (2.0 * b)
    }

    /// hauteur du centre de masse du sable (m)
    pub(crate) fn sand_height(s: &MultiphaseSim, c: usize) -> f64 {
        let n = s.nx * s.ny;
        let m: f64 = (0..n).map(|q| s.mass[c][q]).sum();
        (0..n).map(|q| s.mass[c][q] * ((q / s.nx) as f64 + 0.5) * s.dx).sum::<f64>() / m.max(1e-300)
    }

    /// plus grande vitesse verticale des grains (faces voisines des cellules à plus de 10 % de grains)
    fn grain_speed(s: &MultiphaseSim, c: usize) -> f64 {
        (0..s.nx * s.ny)
            .filter(|&q| s.alpha[c][q] > 0.1)
            .map(|q| {
                let (i, j) = (q % s.nx, q / s.nx);
                s.v[j * s.nx + i].abs().max(s.v[(j + 1) * s.nx + i].abs())
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn a_bed_blown_below_minimum_fluidization_stays_put_and_unloads_its_contacts() {
        // sable de 2 mm sur une grille, 20 cm de lit, soufflé à 0.5 et 0.9 m/s (U_mf d'Ergun : 1.08 m/s) : le lit ne bouge pas, la perte
        // de charge est celle d'Ergun, et le fluide porte une part du poids — la pression de contact du frottement en est déchargée
        // (p_s = poids déjaugé au-dessus − traînée cumulée) ; sans cela, la traînée la gonflait (frottement raidi) et poussait le lit
        let (a, b, w) = ergun_and_weight();
        for vin in [0.5, 0.9] {
            let (mut s, c) = blown_sand_bed(12, 40, 20, vin);
            while s.time < 1.0 && !s.diverged {
                s.step();
            }
            assert!(!s.diverged && s.state_is_sane());
            let ps = s.grain_overburden();
            let (nx, dx) = (s.nx, s.dx);
            let pf = |j: usize| (2..nx - 2).map(|i| { let q = s.idx(i, j); s.p[q] - ps[q] }).sum::<f64>() / (nx - 4) as f64;
            let rho = Phase::air().rho(101325.0, 293.15);
            // rangées 3 (bas du lit) et 34 (air au-dessus) : traînée sur 19.5 mailles de lit (du centre de la rangée 3 au haut, y = 0.23)
            let dpf = pf(3) - pf(34) - rho * 9.81 * 31.0 * dx;
            let g = a * vin + b * vin * vin;
            let expect = g * 0.195;
            let geo = s.friction_geometry();
            // pression de contact au centre de la rangée 4 (18.5 mailles de lit au-dessus)
            let psb = (2..nx - 2).map(|i| geo.ps[s.idx(i, 4)]).sum::<f64>() / (nx - 4) as f64;
            let ps_expect = (w - g) * 0.185;
            let v = grain_speed(&s, c);
            eprintln!("U = {vin} m/s (U_mf {:.3}) : perte de charge {dpf:.0} Pa (Ergun {expect:.0}) ; contact au fond {psb:.0} Pa (poids − traînée {ps_expect:.0}, poids seul {:.0}) ; |v| grains {v:.2e}",
                ergun_umf(), w * 0.185);
            assert!((dpf / expect - 1.0).abs() < 0.05, "perte de charge {dpf} pour {expect}");
            assert!((psb / ps_expect - 1.0).abs() < 0.15, "contact {psb} pour {ps_expect}");
            assert!(v < 1e-2, "le lit bouge : {v} m/s");
        }
    }

    #[test]
    fn a_blown_bed_fluidizes_at_the_ergun_minimum_velocity() {
        // seuil : à 0.92 m/s (0.85 U_mf) le lit reste posé, à 1.2 m/s (1.11 U_mf) les contacts ne portent plus rien, il se soulève — sans
        // la décharge des contacts, un lit soufflé était poussé quelle que soit la vitesse (son poids ne s'y opposait pas). Juste sous U_mf
        // (0.93 U_mf), la surface du lit se met déjà à bouillonner : l'écoulement se concentre là où le lit est le plus lâche
        let umf = ergun_umf();
        assert!((umf - 1.08).abs() < 0.01, "U_mf {umf}");
        // (montée du centre de masse du sable en 0.5 s ; pression de contact au fond)
        let run = |vin: f64| {
            let (mut s, c) = blown_sand_bed(12, 40, 20, vin);
            let y0 = sand_height(&s, c);
            while s.time < 0.5 && !s.diverged {
                s.step();
            }
            assert!(!s.diverged && s.state_is_sane());
            let geo = s.friction_geometry();
            let psb = (2..s.nx - 2).map(|i| geo.ps[s.idx(i, 4)]).sum::<f64>() / (s.nx - 4) as f64;
            (sand_height(&s, c) - y0, psb)
        };
        let (dy_lo, ps_lo) = run(0.92);
        let (dy_hi, ps_hi) = run(1.2);
        eprintln!("U_mf {umf:.3} m/s : à 0.92 m/s le sable monte de {:.2} mm, contact {ps_lo:.0} Pa ; à 1.2 m/s de {:.1} mm, contact {ps_hi:.0} Pa",
            1e3 * dy_lo, 1e3 * dy_hi);
        assert!(dy_lo.abs() < 2e-3 && ps_lo > 50.0, "lit soulevé sous U_mf : {dy_lo} m, contact {ps_lo} Pa");
        assert!(dy_hi > 0.01, "lit immobile au-dessus de U_mf : {dy_hi} m");
    }

    #[test]
    fn a_wide_bed_blown_above_minimum_fluidization_bubbles_and_keeps_its_sand() {
        // cuve de 40 cm, lit de 15 cm, 1.4 m/s (1.3 U_mf) : le lit fluidisé gonfle et bouillonne ; le sable reste dans la cuve
        let (mut s, c) = blown_sand_bed(40, 60, 15, 1.4);
        let m0 = s.total_mass(c);
        let top = |s: &MultiphaseSim| (0..s.ny).rev().find(|&j| (0..s.nx).map(|i| s.alpha[c][s.idx(i, j)]).sum::<f64>() / s.nx as f64 > 0.05).unwrap_or(0);
        let h0 = top(&s);
        while s.time < 0.5 && !s.diverged {
            s.step();
        }
        assert!(!s.diverged && s.state_is_sane());
        let h1 = top(&s);
        eprintln!("lit fluidisé : haut du lit rangée {h0} → {h1} ; sable {:.2} % ; {} pas", 100.0 * s.total_mass(c) / m0, s.step_count);
        assert!(h1 >= h0 + 5, "le lit ne gonfle pas : {h0} → {h1}");
        assert!(s.total_mass(c) > 0.999 * m0, "sable sorti de la cuve");
    }

    /// diagnostic : lit large fluidisé (MP_FL_U m/s, défaut 1.4 ; MP_FL_T s, défaut 2)
    #[test]
    #[ignore]
    fn fluidized_bed_diag() {
        let env = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
        let (nx, ny, h) = (40, 60, 15);
        let (mut s, c) = blown_sand_bed(nx, ny, h, env("MP_FL_U", 1.4));
        let m0 = s.total_mass(c);
        let w0 = m0 / (nx as f64 * s.dx) * 9.81;
        let t0 = std::time::Instant::now();
        let mut next = 0.0;
        while s.time < env("MP_FL_T", 2.0) && !s.diverged {
            s.step();
            if s.time >= next {
                next += 0.25;
                let prof: Vec<String> = (2..ny).step_by(3).map(|j| format!("{:.2}", (0..nx).map(|i| s.alpha[c][s.idx(i, j)]).sum::<f64>() / nx as f64)).collect();
                let ps = s.grain_overburden();
                let pf = |j: usize| (0..nx).map(|i| { let q = s.idx(i, j); s.p[q] - ps[q] }).sum::<f64>() / nx as f64;
                let dpf = pf(3) - pf(ny - 2) - 1.2 * 9.81 * (ny - 5) as f64 * s.dx;
                eprintln!("t {:.2} ({} pas, {:.0} s) : sable {:.1} % ; Δp_f {dpf:.0} Pa (poids du lit {w0:.0} Pa) ; |u| max {:.2} ; α par 3 rangées {}",
                    s.time, s.step_count, t0.elapsed().as_secs_f64(), 100.0 * s.total_mass(c) / m0, s.max_speed(), prof.join(" "));
            }
        }
    }
}
