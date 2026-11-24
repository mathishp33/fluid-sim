//! Percolation : un fluide traverse les PORES d'un lit de grains posé sur un appui (les grains restent en place) et les GRILLES
//! (cellules qui arrêtent les grains : le tas y repose, l'air arrive par-dessous). Sans elle, le modèle à une vitesse par cellule fait
//! d'un lit tassé une paroi étanche (le gaz des pores bouge avec les grains) : un feu de charbon ne tire pas, l'eau ne s'infiltre pas.
//!
//! Loi d'Ergun (Darcy–Forchheimer), q = débit de fluide par unité de surface (vitesse superficielle, m/s), relatif aux grains immobiles :
//!   −∂p/∂n + ρ_f g_n = (μ/K) q + b |q| q,   K = d² ε³ / (150 (1 − ε)²),   b = 1.75 ρ_f (1 − ε) / (ε³ d)
//! (ε : porosité, d : diamètre des grains — ou des trous d'une grille). Sable de 0.3 mm : K ≈ 1e-10 m² (dans l'air, K/μ ≈ 6e-6 m²/(Pa·s) :
//! le lit respire) ; charbon en morceaux de 1 cm : K ≈ 1e-7 m² (le tirage d'un feu). Lits trop peu perméables (K/μ < `PERC_KMU_MIN` :
//! poudres fines, sable fin dans l'eau) : traités comme étanches (une seule pression).
//!
//! DEUX PRESSIONS dans un lit perméable (cellules `skel` : grains en contact, posés sur un appui, tassés) : p, pression du FLUIDE des pores (la
//! cellule est remplie de fluide à p : fermeture, thermodynamique, loi d'Ergun), et σ' ≥ 0, pression de CONTACT des grains (Terzaghi ;
//! `MultiphaseSim::sigma`). La pression du mélange, qui accélère la cellule, vaut p + σ'. Deux contraintes : volume de la cellule
//! (∇·(u + q) = compression du fluide, comme ailleurs) et SQUELETTE incompressible (∇·u = 0 : les grains ne se tassent ni ne se
//! dilatent, sinon par la sédimentation, bornée à α_max) — σ' en est le multiplicateur (`mp_pressure::solve_skeleton`), relâché à 0
//! s'il devient négatif (le lit se décolle : fluidisation). Au repos, p est hydrostatique et σ' porte le poids déjaugé des grains ;
//! soufflé par en dessous, le gradient de p porte une part du poids et σ' en est déchargé (frottement des grains sur σ',
//! `friction_geometry`) — à la vitesse minimale de fluidisation d'Ergun, σ' = 0 : le lit se soulève. Un paquet tassé décollé de son
//! appui (lit soulevé, paquet qui tombe) n'est ni poreux ni porteur : une seule pression, celle du mélange. Les cellules de surface d'un
//! lit (partiellement remplies) sont poreuses mais à une seule pression : leur fluide y est à p − p_s (contact lithostatique).
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
/// une cellule de grains NON poreuse (paquet décollé de son appui, lit étanche) d'au moins α_max − `SEAL_GAP` de grains est close au
/// fluide des pores : sans cela, l'air en sortait par la face d'une voisine poreuse et rien n'arrêtait son tassement (α_s jusqu'à 1 :
/// sable sans air, pression de 3e8 Pa sur l'effondrement sans frottement d'une colonne de sable)
const SEAL_GAP: f64 = 0.2;
/// une cellule poreuse n'est à DEUX PRESSIONS (squelette) que tassée : au moins α_max − `SKEL_GAP` de grains (comme les grains
/// « tassés » qui portent une chaîne de contact, `grain_support`). Les cellules de surface partiellement remplies restent poreuses mais
/// à une seule pression : y imposer un squelette incompressible (∇·u = 0) à 0.01 % de grains bloquait l'expansion du gaz d'une flamme
/// au-dessus d'un lit de braises (∇·u ~ 1000 /s), d'où une pression de contact de 1e7 Pa dans tout le lit et un pas de 2e-5 s
pub(crate) const SKEL_GAP: f64 = 0.1;
/// mobilité de Darcy minimale K/μ (m²/(Pa·s)) d'un lit perméable : dans l'air, grains de 0.1 mm et plus ; dans l'eau, de 0.6 mm et plus.
/// En dessous, le lit est étanche (une seule pression) : le fluide n'y passerait presque pas (0.1 mm/s d'eau dans du sable de 0.3 mm
/// sous son propre poids), et la pression de contact, partagée avec celle des pores au fil des pas (`solve_skeleton` une fois par pas),
/// y mettrait trop de pas à s'établir quand dt ρ / dx² dépasse la conductance de Darcy
pub(crate) const PERC_KMU_MIN: f64 = 5e-7;

/// conductances (m²/(Pa·s)) et termes moteurs (m/s) de Darcy par face : q = −C (p_b − p_a)/dx + d, positif de a (gauche / bas) vers b ;
/// `skel` : cellules à deux pressions (voir l'en-tête)
pub(crate) struct DarcyFaces {
    pub cu: Vec<f64>,
    pub du: Vec<f64>,
    pub cv: Vec<f64>,
    pub dv: Vec<f64>,
    pub skel: Vec<bool>,
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

    /// milieu poreux de la cellule : grille, ou grains EN CONTACT (chaîne posée sur un appui : tassés, ou cellule de surface d'un lit,
    /// partiellement remplie — sans cela elle scelle le lit) de mobilité K/μ d'au moins `PERC_KMU_MIN` ; `contact` vient de
    /// `grain_contact`. None sinon (fluide libre, solide, épinglée, lit trop peu perméable)
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
        if a_s <= 1e-6 || !contact[c] {
            return None;
        }
        let (rho, mu) = self.pore_fluid(c);
        let eps = (1.0 - a_s).clamp(0.05, 0.95);
        let d = ad / a_s;
        let k = kozeny(d, eps);
        (mu > 0.0 && k / mu >= PERC_KMU_MIN).then_some(Porous { k, eps, d, rho, mu })
    }

    /// milieu poreux de la cellule c (contact recalculé : diagnostics et tests)
    #[allow(dead_code)]
    pub(crate) fn porous_cell(&self, c: usize) -> Option<Porous> {
        let (_, contact) = self.grain_contact();
        self.porous_cell_with(c, &contact)
    }

    /// pression de contact lithostatique p_s (Pa) : poids déjaugé des grains au-dessus, le long de la gravité (axe dominant), sur les
    /// colonnes de grains en contact (chaînes de grains qui reposent sur un appui : le fond, un solide, une grille, ou des grains tassés
    /// eux-mêmes en contact) — diagnostics et tests (la projection résout σ')
    #[allow(dead_code)]
    pub(crate) fn grain_overburden(&self) -> Vec<f64> {
        self.grain_contact().0
    }

    /// (α_s, α_max) de la cellule : fraction de ses grains et compacité maximale de leur mélange (moyenne pondérée ; 1 sans grains)
    pub(crate) fn grain_packing(&self, c: usize) -> (f64, f64) {
        let (mut a_s, mut am) = (0.0, 0.0);
        for (k, ph) in self.reg.phases.iter().enumerate() {
            if ph.is_granular() {
                let a = self.alpha[k][c].max(0.0);
                a_s += a;
                am += a * ph.alpha_max;
            }
        }
        if a_s > 1e-6 { (a_s, am / a_s) } else { (a_s, 1.0) }
    }

    /// cellule de grains non poreuse et dense, close au fluide des pores (voir `SEAL_GAP`)
    fn sealed(&self, c: usize, porous: &[Option<Porous>]) -> bool {
        if porous[c].is_some() || self.is_solid(c) {
            return false;
        }
        let (a_s, amax) = self.grain_packing(c);
        a_s > 1e-6 && a_s >= amax - SEAL_GAP
    }

    /// cellules à DEUX PRESSIONS (voir l'en-tête) : grains poreux en contact et tassés (`SKEL_GAP`), hors grilles (`porous` : milieux
    /// poreux de `darcy_faces`)
    fn skeleton_of(&self, porous: &[Option<Porous>]) -> Vec<bool> {
        (0..self.nx * self.ny)
            .map(|c| {
                porous[c].is_some() && !self.is_grille(c) && !self.is_solid(c) && !self.pinned(c) && {
                    let (a_s, amax) = self.grain_packing(c);
                    a_s >= amax - SKEL_GAP
                }
            })
            .collect()
    }

    /// Pression de contact initiale des lits perméables (colonnes lithostatiques, `grain_support`) : la pression hydrostatique du
    /// MÉLANGE posée par `finalize_init` y est partagée entre le fluide des pores (p − p_s, hydrostatique du fluide) et les contacts
    /// (σ' = p_s) ; sous un lit posé sur une GRILLE, c'est elle qui porte le poids des grains, pas le fluide : la pression du fluide
    /// sous la grille en est déchargée (sans cela, 1.6 kPa de trop sous 20 cm de charbon soufflaient le lit au premier pas). Masses à
    /// recalculer ensuite. Renvoie vrai si un lit perméable existe.
    pub(crate) fn init_contact_stress(&mut self) -> bool {
        let n = self.nx * self.ny;
        self.sigma = vec![0.0; n];
        self.skel = vec![false; n];
        if !self.percolation || !self.has_granular() {
            return false;
        }
        let (ps, contact) = self.grain_contact();
        let por: Vec<Option<Porous>> = (0..n).map(|c| self.porous_cell_with(c, &contact)).collect();
        let skel = self.skeleton_of(&por);
        let mut any = false;
        for c in 0..n {
            if skel[c] {
                self.sigma[c] = ps[c].max(0.0);
                self.p[c] -= self.sigma[c];
                any = true;
            }
        }
        // poids des lits portés par une grille, retiré de la pression du fluide au-dessous (colonnes du haut vers le bas, comme
        // finalize_init : gravité vers −y ; un solide interrompt la colonne)
        if any && self.gravity[1] < 0.0 && self.gravity[1].abs() >= self.gravity[0].abs() {
            let gn = (self.gravity[0].powi(2) + self.gravity[1].powi(2)).sqrt();
            for i in 0..self.nx {
                let (mut offset, mut carried) = (0.0f64, 0.0f64);
                for j in (0..self.ny).rev() {
                    let c = self.idx(i, j);
                    if self.is_solid(c) {
                        offset = 0.0;
                        carried = 0.0;
                        continue;
                    }
                    if !skel[c] {
                        if carried > 0.0 && self.is_grille(c) {
                            offset += carried;
                        }
                        carried = 0.0;
                    }
                    if offset > 0.0 {
                        self.p[c] = (self.p[c] - offset).max(2.0 * super::mp_pressure::P_FLOOR);
                    }
                    if skel[c] {
                        // poids transmis par la face inférieure de la cellule : σ' + ½ poids déjaugé de ses grains
                        let (mut a_s, mut ms) = (0.0, 0.0);
                        for (k, ph) in self.reg.phases.iter().enumerate() {
                            if ph.is_granular() {
                                a_s += self.alpha[k][c].max(0.0);
                                ms += self.mass[k][c].max(0.0);
                            }
                        }
                        let w = if a_s > 1e-6 { (ms / a_s - self.pore_fluid(c).0).max(0.0) * a_s * gn * self.dx } else { 0.0 };
                        carried = self.sigma[c] + 0.5 * w;
                    }
                }
            }
        }
        self.skel = skel;
        any
    }

    /// (p_s, contact) de chaque cellule (voir `grain_overburden`)
    pub(crate) fn grain_contact(&self) -> (Vec<f64>, Vec<bool>) {
        let (ps, contact, _) = self.grain_support();
        (ps, contact)
    }

    /// (p_s, contact, appui) de chaque cellule : contact = appui = chaîne de grains qui repose sur un appui (le fond, un solide, une
    /// grille, ou des grains tassés eux-mêmes sur un appui), cellule de surface partiellement remplie comprise ; p_s = poids déjaugé des
    /// grains au-dessus le long de ces colonnes (pression de contact d'un lit au repos : état initial, `init_contact_stress`)
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

    /// Conductances et termes moteurs de Darcy des faces (voir l'en-tête), avec la correction d'Ergun évaluée sur la pression (du fluide)
    /// du pas précédent ; None s'il n'y a ni lit perméable ni grille
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
        let skel = self.skeleton_of(&por);
        let sealed: Vec<bool> = (0..n).map(|c| self.sealed(c, &por)).collect();
        // cellules poreuses à UNE pression (surface d'un lit, partiellement remplie ; grains en contact non tassés) : leur pression est
        // celle du mélange, la pression du fluide de leurs pores en est p − p_s (contact lithostatique). Sans cela, le poids de leurs
        // grains poussait le fluide dans le lit à deux pressions au-dessous (2 cm/s d'air à travers la surface d'un tas de sable de
        // 0.3 mm, qui coulait à 0.5 m/s)
        let ps1: Vec<f64> = (0..n).map(|c| if por[c].is_some() && !skel[c] && !self.is_grille(c) { ps[c] } else { 0.0 }).collect();
        let rho_f: Vec<f64> = (0..n).map(|c| if self.is_solid(c) { 0.0 } else { self.pore_fluid(c).0 }).collect();
        let face = |a: usize, b: usize, g: f64| -> (f64, f64) {
            let (pa, pb) = (por[a], por[b]);
            if (pa.is_none() && pb.is_none()) || sealed[a] || sealed[b] || self.is_solid(a) || self.is_solid(b) || self.pinned(a) || self.pinned(b) {
                return (0.0, 0.0);
            }
            let rf = match (rho_f[a] > 0.0, rho_f[b] > 0.0) {
                (true, true) => 0.5 * (rho_f[a] + rho_f[b]),
                (true, false) => rho_f[a],
                _ => rho_f[b],
            };
            // moteur : poids du fluide ρ_f g_n (p : pression du fluide des pores ; p − p_s dans les cellules à une pression)
            let drive = rf * g * dx + ps1[b] - ps1[a];
            let gm = ((self.p[b] - self.p[a]) - drive).abs() / dx;
            let inv = |p: Option<Porous>| p.map_or(0.0, |q| 1.0 / q.conductance(gm));
            let c = 2.0 / (inv(pa) + inv(pb));
            (c, c * drive / dx)
        };
        let mut df = DarcyFaces { cu: vec![0.0; (nx + 1) * ny], du: vec![0.0; (nx + 1) * ny], cv: vec![0.0; nx * (ny + 1)], dv: vec![0.0; nx * (ny + 1)], skel };
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
        // (grille posée avant l'initialisation : le lit y repose dès le départ, pression de contact comprise)
        for i in 0..12 {
            let q = s.idx(i, 4);
            s.set_effect(q, Some(MpEffect::Grille { hole: 0.01, open: 0.6 }));
        }
        s.finalize_init(|_, _| 293.15);
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
        // perte de charge : pression du fluide des pores (p dans un lit perméable), rangées 8 et 20 (centres), colonnes intérieures
        let pf = |j: usize| (2..s.nx - 2).map(|i| s.p[s.idx(i, j)]).sum::<f64>() / (s.nx - 4) as f64;
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
    fn a_resting_bed_does_not_leak_and_only_fine_powder_is_impermeable() {
        // sans soufflage : au repos, aucun écoulement de percolation (p hydrostatique dans les pores), le poids des grains porté par
        // les contacts (σ' = poids déjaugé au-dessus) ; perméabilités : morceaux d'1 cm ≫ grains de 0.3 mm (perméables dans l'air,
        // K/μ ≈ 6e-6 m²/(Pa·s)) ≫ poudre de 30 µm (étanche : K/μ ≈ 6e-8 < PERC_KMU_MIN)
        for d in [0.01, 3e-4] {
            let (mut s, _, _) = bed_on_grille(d, 0.0);
            while s.time < 0.5 && !s.diverged {
                s.step();
            }
            assert!(!s.diverged);
            let qmax = s.q_u.iter().chain(s.q_v.iter()).fold(0.0f64, |a, &b| a.max(b.abs()));
            let ps = s.grain_overburden();
            let q = s.idx(6, 6);
            eprintln!("lit au repos (grains de {:.1} mm) : |q| max {qmax:.2e} m/s ; contact rangée 6 {:.0} Pa (poids au-dessus {:.0})", 1e3 * d, s.sigma[q], ps[q]);
            assert!(qmax < 2e-3, "fuite de percolation au repos : {qmax}");
            assert!(s.skel[q] && (s.sigma[q] / ps[q] - 1.0).abs() < 0.05, "contact {} pour {}", s.sigma[q], ps[q]);
        }
        let (lump, _, _) = bed_on_grille(0.01, 0.0);
        let (fine, _, _) = bed_on_grille(3e-4, 0.0);
        let (powder, _, _) = bed_on_grille(3e-5, 0.0);
        let kl = lump.porous_cell(lump.idx(6, 14)).unwrap().k;
        assert!((kl / kozeny(0.01, 0.4) - 1.0).abs() < 1e-6, "K = {kl}");
        assert!(fine.porous_cell(fine.idx(6, 14)).is_some(), "grains de 0.3 mm étanches");
        assert!(powder.porous_cell(powder.idx(6, 14)).is_none() && !powder.skel[powder.idx(6, 14)], "poudre de 30 µm perméable");
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
        blown_bed(nx, ny, h, vin, 2e-3, Phase::air())
    }

    /// cuve de `fluid` (nx × ny mailles d'1 cm) : grille au rang 2, lit de sable de grains de `d` m tassé sur h mailles posé dessus,
    /// fluide soufflé par le fond à `vin` m/s, haut ouvert ; 293 K
    pub(crate) fn blown_bed(nx: usize, ny: usize, h: usize, vin: f64, d: f64, fluid: Phase) -> (MultiphaseSim, usize) {
        let mut r = Registry::new();
        let air = r.add_phase(fluid);
        let mut sand = Phase::sand();
        sand.grain_diameter = d;
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
        for i in 0..nx {
            let q = s.idx(i, 2);
            s.set_effect(q, Some(MpEffect::Grille { hole: 0.002, open: 0.4 }));
        }
        s.finalize_init(|_, _| 293.15);
        s.inlet[2] = Some(MpInlet { vx: 0.0, vy: vin, phase: air, temp: 293.15, offset: 0 });
        s.bc[2] = Bc::Wall;
        s.bc[3] = Bc::Open { p: 101325.0 };
        s.refresh();
        (s, c)
    }

    /// (a, b) d'Ergun du lit tassé (ε = 0.4, d = 2 mm, air) : gradient a q + b q², et poids déjaugé par unité de volume (N/m³)
    fn ergun_and_weight() -> (f64, f64, f64) {
        ergun_and_weight_of(2e-3, &Phase::air())
    }

    /// (a, b, poids déjaugé) d'Ergun d'un lit de sable tassé (ε = 0.4) de grains de `d` m dans `fluid`
    fn ergun_and_weight_of(d: f64, fluid: &Phase) -> (f64, f64, f64) {
        let (rho, mu) = (fluid.rho(101325.0, 293.15), fluid.mu(293.15));
        let eps = 0.4;
        let a = mu / kozeny(d, eps);
        let b = ERGUN_B * rho * (1.0 - eps) / (eps.powi(3) * d);
        (a, b, (1.0 - eps) * (Phase::sand().rho(101325.0, 293.15) - rho) * 9.81)
    }

    /// vitesse minimale de fluidisation d'Ergun du lit de sable de 2 mm dans l'air : a U + b U² = poids déjaugé (1.08 m/s)
    pub(crate) fn ergun_umf() -> f64 {
        let (a, b, w) = ergun_and_weight();
        (-a + (a * a + 4.0 * b * w).sqrt()) / (2.0 * b)
    }

    /// vitesse minimale de fluidisation d'Ergun d'un lit de grains de `d` m dans `fluid`
    fn ergun_umf_of(d: f64, fluid: &Phase) -> f64 {
        let (a, b, w) = ergun_and_weight_of(d, fluid);
        (-a + (a * a + 4.0 * b * w).sqrt()) / (2.0 * b)
    }

    /// (perte de charge du lit entre les rangées 3 et `top` (Pa, poids du fluide retiré), pression de contact moyenne de la rangée 4,
    /// plus grande vitesse des grains) d'un lit soufflé
    fn blown_bed_state(s: &MultiphaseSim, c: usize, top: usize) -> (f64, f64, f64) {
        let nx = s.nx;
        let pf = |j: usize| (2..nx - 2).map(|i| s.p[s.idx(i, j)]).sum::<f64>() / (nx - 4) as f64;
        let rho_f = (0..s.reg.phases.len()).filter(|&k| k != c).map(|k| s.mass[k][s.idx(nx / 2, top)]).sum::<f64>()
            / (0..s.reg.phases.len()).filter(|&k| k != c).map(|k| s.alpha[k][s.idx(nx / 2, top)]).sum::<f64>().max(1e-12);
        let dpf = pf(3) - pf(top) - rho_f * 9.81 * (top - 3) as f64 * s.dx;
        let psb = (2..nx - 2).map(|i| s.sigma[s.idx(i, 4)]).sum::<f64>() / (nx - 4) as f64;
        (dpf, psb, grain_speed(s, c))
    }

    #[test]
    fn a_fine_sand_bed_breathes_below_minimum_fluidization_and_fluidizes_above() {
        // sable de 0.3 mm (étanche avant les deux pressions) sur une grille, 20 cm de lit dans l'air : U_mf d'Ergun ≈ 9 cm/s. À la
        // moitié, l'air traverse le lit à la perte de charge d'Ergun, le lit ne bouge pas et ses contacts portent le poids moins la
        // traînée ; à 1.4 U_mf, les contacts ne portent plus rien et le lit gonfle (Kozeny : ε de 0.40 à ~0.44, centre de masse ~7 mm
        // plus haut à l'équilibre, atteint en ~0.4 s — l'air en excès, 3.6 cm/s, remplit les pores qui s'ouvrent)
        let (d, air) = (3e-4, Phase::air());
        let umf = ergun_umf_of(d, &air);
        let (a, b, w) = ergun_and_weight_of(d, &air);
        assert!((umf - 0.089).abs() < 0.005, "U_mf {umf}");
        let vin = 0.5 * umf;
        let (mut s, c) = blown_bed(12, 40, 20, vin, d, air.clone());
        while s.time < 1.0 && !s.diverged {
            s.step();
        }
        assert!(!s.diverged && s.state_is_sane());
        let (dpf, psb, v) = blown_bed_state(&s, c, 34);
        let g = a * vin + b * vin * vin;
        let (expect, ps_expect) = (g * 0.195, (w - g) * 0.185);
        eprintln!("sable fin à {vin:.3} m/s (U_mf {umf:.3}) : perte de charge {dpf:.0} Pa (Ergun {expect:.0}) ; contact {psb:.0} Pa (poids − traînée {ps_expect:.0}) ; |v| grains {v:.2e} ; {} pas",
            s.step_count);
        assert!((dpf / expect - 1.0).abs() < 0.05, "perte de charge {dpf} pour {expect}");
        assert!((psb / ps_expect - 1.0).abs() < 0.15, "contact {psb} pour {ps_expect}");
        assert!(v < 1e-2, "le lit bouge : {v} m/s");
        let (mut s, c) = blown_bed(12, 40, 20, 1.4 * umf, d, air);
        let (y0, m0) = (sand_height(&s, c), s.total_mass(c));
        let mut dy_half = 0.0;
        while s.time < 1.0 && !s.diverged {
            s.step();
            if dy_half == 0.0 && s.time >= 0.5 {
                dy_half = sand_height(&s, c) - y0;
            }
        }
        assert!(!s.diverged && s.state_is_sane());
        let dy = sand_height(&s, c) - y0;
        let (_, psb, _) = blown_bed_state(&s, c, 34);
        eprintln!("sable fin à 1.4 U_mf : il monte de {:.1} mm en 0.5 s, {:.1} mm en 1 s ; contact {psb:.0} Pa (poids seul {:.0}) ; sable {:.2} %",
            1e3 * dy_half, 1e3 * dy, w * 0.185, 100.0 * s.total_mass(c) / m0);
        assert!(dy > 2.5e-3 && dy < 0.02, "lit fin au-dessus de U_mf : centre de masse monté de {dy} m");
        assert!(psb < 0.1 * w * 0.185, "contacts encore chargés au-dessus de U_mf : {psb} Pa");
        assert!(s.total_mass(c) > 0.999 * m0, "sable sorti de la cuve");
    }

    #[test]
    fn water_percolates_through_a_coarse_sand_bed_at_the_ergun_pressure_drop() {
        // sable de 2 mm dans l'eau (U_mf d'Ergun ≈ 2.4 cm/s), eau poussée par le fond à 2 cm/s (0.84 U_mf) : perte de charge d'Ergun,
        // lit immobile, contacts déchargés de la traînée (le fluide porte 77 % du poids déjaugé). L'entrée monte en 0.2 s : lancer d'un
        // coup 40 cm d'eau à 2 cm/s (impulsion de ~800 Pa au fond sur un pas de 10 ms) laissait 330 Pa de trop dans les contacts — le
        // lit rigide (frottement des grains, non-glissement aux parois) garde la contrainte du choc, comme un vrai lit en colonne étroite
        let (d, water) = (2e-3, Phase::water());
        let umf = ergun_umf_of(d, &water);
        let (a, b, w) = ergun_and_weight_of(d, &water);
        let vin = 0.02;
        let (mut s, c) = blown_bed(12, 40, 20, 0.0, d, water);
        while s.time < 1.0 && !s.diverged {
            if let Some(inl) = s.inlet[2].as_mut() {
                inl.vy = vin * (s.time / 0.2).min(1.0);
            }
            s.step();
        }
        assert!(!s.diverged && s.state_is_sane());
        let (dpf, psb, v) = blown_bed_state(&s, c, 34);
        let g = a * vin + b * vin * vin;
        let (expect, ps_expect) = (g * 0.195, (w - g) * 0.185);
        eprintln!("eau à {vin} m/s dans du sable de 2 mm (U_mf {umf:.3}) : perte de charge {dpf:.0} Pa (Ergun {expect:.0}) ; contact {psb:.0} Pa (poids − traînée {ps_expect:.0}, poids seul {:.0}) ; |v| grains {v:.2e} ; {} pas",
            w * 0.185, s.step_count);
        assert!((umf - 0.024).abs() < 0.002, "U_mf {umf}");
        assert!((dpf / expect - 1.0).abs() < 0.05, "perte de charge {dpf} pour {expect}");
        assert!((psb / ps_expect - 1.0).abs() < 0.15, "contact {psb} pour {ps_expect}");
        assert!(v < 1e-2, "le lit bouge : {v} m/s");
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
            let (nx, dx) = (s.nx, s.dx);
            let pf = |j: usize| (2..nx - 2).map(|i| s.p[s.idx(i, j)]).sum::<f64>() / (nx - 4) as f64;
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
                let pf = |j: usize| (0..nx).map(|i| s.p[s.idx(i, j)]).sum::<f64>() / nx as f64;
                let dpf = pf(3) - pf(ny - 2) - 1.2 * 9.81 * (ny - 5) as f64 * s.dx;
                eprintln!("t {:.2} ({} pas, {:.0} s) : sable {:.1} % ; Δp_f {dpf:.0} Pa (poids du lit {w0:.0} Pa) ; |u| max {:.2} ; α par 3 rangées {}",
                    s.time, s.step_count, t0.elapsed().as_secs_f64(), 100.0 * s.total_mass(c) / m0, s.max_speed(), prof.join(" "));
            }
        }
    }
}
