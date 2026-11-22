//! Solveur multiphasique 2D (CPU, f64) : N gaz/liquides compressibles (EOS réelles) + solides, grille MAC,
//! interface PLIC conservative, pression implicite (Helmholtz, acoustique non limitante), tension de surface,
//! gravité, viscosité variable, énergie, ébullition / cavitation (Hardt–Wondra + Schnerr–Sauer).
//!
//! Conventions : j croît vers +y (HAUT) ; gravité par défaut (0, −9.81). u : (nx+1)×ny, v : nx×(ny+1).
//! Variables conservées par cellule : masses partielles m_k (kg/m³ de cellule) et énergie interne e (J/m³).
//! Les fractions volumiques α_k sont DÉRIVÉES : α_k = (m_k/ρ_k(p,T)) / Σ_j (m_j/ρ_j(p,T)).
//! Une seule pression et une seule température par cellule (équilibre mécanique et thermique des phases).

use super::mp_cells::{MpEffect, MpInlet, MpThermal, PinInfo, TpMix};
use super::phases::{PhaseKind, Registry};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bc {
    Wall,
    /// Bord ouvert à pression imposée (atmosphère). L'air/fluide entrant est la phase `ambient_phase` à `t_ambient`.
    Open { p: f64 },
}

pub struct MultiphaseSim {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub reg: Registry,
    pub gravity: [f64; 2],
    /// left, right, bottom, top
    pub bc: [Bc; 4],
    pub ambient_phase: usize,
    pub t_ambient: f64,

    pub mass: Vec<Vec<f64>>,  // [phase][cell]
    pub alpha: Vec<Vec<f64>>, // [phase][cell] (dérivé)
    pub e: Vec<f64>,          // énergie interne J/m³ (solide : ρ_s c_s T)
    pub temp: Vec<f64>,
    /// groupes thermiques (= groupes miscibles du registre) et groupe de chaque phase (voir mp_groups.rs)
    pub(crate) groups: Vec<Vec<usize>>,
    pub(crate) group_of: Vec<usize>,
    /// énergie interne absolue de chaque groupe par cellule (J/m³ de cellule) ; e = Σ_g eg (cellules fluides)
    pub eg: Vec<Vec<f64>>,
    /// température de chaque groupe par cellule (K) ; groupe absent : température de la cellule
    pub tg: Vec<Vec<f64>>,
    /// facteur λ de l'échange de chaleur entre groupes d'une cellule mixte (1 = conduction à travers la demi-épaisseur de chaque
    /// phase ; ∞ = une seule température par cellule, modèle historique ; 0 = aucun échange)
    pub phase_heat_exchange: f64,
    pub p: Vec<f64>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub solid: Vec<i32>, // −1 = fluide, sinon indice dans reg.solids
    pub rho_mix: Vec<f64>,
    pub mu_cell: Vec<f64>,
    pub k_cell: Vec<f64>,

    // numérique
    pub cfl: f64,
    /// advection de vitesse : false (défaut) = semi-lagrangien bicubique borné (dissipatif, ne crée pas d'énergie) ;
    /// true = MacCormack (moins diffusif mais CRÉE de l'énergie cinétique avec un rapport de densités eau/air de 1000 : déconseillé)
    pub maccormack: bool,
    pub cap_safety: f64, // facteur sur la limite capillaire de Brackbill
    pub dt_max: f64,
    pub surface_tension: bool,
    pub viscous: bool,
    pub conduction: bool,
    pub phase_change: bool,
    pub pressure_tol: f64,
    pub pressure_max_iter: usize,
    /// préconditionneur de la pression : multigrille (défaut, parallèle) ou Cholesky incomplet IC(0) (historique)
    pub multigrid: bool,
    // changement de phase
    pub nucleation_superheat: f64, // K de surchauffe pour amorcer l'ébullition en volume
    pub nucleation_alpha: f64,     // fraction de vapeur créée à l'amorçage (germes)
    pub nuclei_density: f64,       // n0 (1/m³ de liquide) pour Schnerr–Sauer
    /// surchauffe (K) au-delà de laquelle un liquide chauffé bout en volume (en dessous : métastable, seules les interfaces s'évaporent)
    pub boil_superheat_max: f64,
    pub time: f64,
    pub last_dt: f64,
    pub last_pressure_iters: usize,
    /// cellules dont la pression a été bloquée au plancher (p_sat d'un liquide : traction → cavité) au dernier pas
    pub(crate) pressure_clamped: Vec<bool>,
    /// volume créé (> 0) / détruit (< 0) par le changement de phase pendant le pas, par groupe du LIQUIDE (fraction du volume de la
    /// cellule) : `pressure_work` rend −p·ΔV à ce groupe (chaleur latente en Δe : le p·Δv vient du travail de pression)
    pub(crate) pc_dvol: Vec<Vec<f64>>,
    /// itérations de Newton de la fermeture volumique au dernier pas
    pub last_newton_iters: usize,
    /// masse (kg par m de profondeur) entrée par les bords ouverts, par phase : M_k(t) − M_k(0) = flux_k − (changements de phase)
    pub boundary_flux: Vec<f64>,
    /// masse transformée liquide→vapeur cumulée par paire (kg/m), pour les bilans
    pub transferred: Vec<f64>,
    pub(crate) step_count: usize,
    /// flux de masse total (toutes phases) à travers chaque face de cellule pendant le dernier advect_mass_energy (kg/m³ de cellule, + = vers +x / +y)
    pub(crate) mfx: Vec<f64>,
    pub(crate) mfy: Vec<f64>,
    pub(crate) m_tot_old: Vec<f64>,
    /// advection de la quantité de mouvement conservative, cohérente avec les flux de masse du VOF (défaut) ; false = semi-lagrangien
    pub conservative_momentum: bool,
    /// Prédicteur-correcteur : le pas est d'abord calculé normalement (prédiction de u^{n+1}), puis refait depuis l'état n en
    /// advectant masses et énergie avec (u^n + u^{n+1})/2. Supprime le retard d'un pas entre la vitesse qui transporte un front et
    /// celle que produit sa source de volume (flamme, changement de phase) — sans lui, boucle thermoacoustique numérique : flamme
    /// 14 % trop lente au dt par défaut. Coût : ~2 pas (CPU et GPU, mêmes formules). Toujours utilisé quand une flamme épaissie est
    /// possible (`use_predictor_corrector`) ; ailleurs seulement si demandé.
    pub predictor_corrector: bool,
    /// Flammes épaissies actives : conduction, diffusion des espèces et chimie intercalées en `flame_thermo_substeps` sous-pas (le
    /// découpage de Lie diffusion ↔ réaction, raide, est l'erreur dominante d'une flamme : Strang n'y gagne rien).
    pub flame_thermo_substeps: usize,
    /// Flamme présente (une cellule F > 1) : dt ≤ cfl_flame·dx / flame_speed_floor, temps propre de la flamme par maille (la CFL
    /// ordinaire gère les écoulements rapides). Sous prédicteur-correcteur les vitesses sont calmes et la CFL seule laisse dt grimper
    /// bien au-delà de la résolution en temps de la flamme. `flame_speed_floor` ≈ vitesse d'expansion (σ − 1)·S_L d'une flamme
    /// d'hydrocarbure ou d'hydrogène (0.1 · 1 mm / 2.5 m/s = 4e-5 s).
    pub cfl_flame: f64,
    pub flame_speed_floor: f64,
    /// vitesses d'advection masse / énergie imposées (correcteur) et pas de temps imposé
    pub(crate) adv_vel: Option<(Vec<f64>, Vec<f64>)>,
    pub(crate) dt_force: Option<f64>,
    pub(crate) pc_predicting: bool,
    /// vitesse au-delà de laquelle un pas est jugé aberrant (m/s) — déclenche le retour arrière avec sous-pas
    pub speed_limit: f64,
    /// Fraction r d'une GROSSE erreur de fermeture volumique héritée du pas précédent (Σ m/ρ − 1 au début du pas, > 0.1) que ce
    /// pas corrige dans la partie liquide d'une cellule (voir `mp_pressure::closure_correction` ; le gaz et les petites erreurs sont
    /// corrigés entièrement). 1 = correction complète en un pas (schéma historique : neutre-stable, oscille et peut s'emballer quand
    /// un liquide très chaud rencontre de l'eau ou un gaz froid) ; 0.25 = amortissement le plus rapide (facteur 0.5 par pas). Réduite
    /// d'un facteur 4 à chaque retour arrière (la vitesse de correction reste celle du pas nominal au lieu d'être multipliée par 4).
    pub closure_relax: f64,
    pub(crate) closure_relax_scale: f64,
    /// Un pas est refait (dt/4) si les vitesses QU'IL PRODUIT dépassent ce nombre de Courant (u·dt/dx). Le pas de temps est choisi
    /// avec les vitesses du début du pas (CFL `cfl` = 0.4) ; un choc thermique (liquide très chaud au contact d'un gaz froid) ou un
    /// effondrement de poche de gaz les multiplie parfois par 50 en un pas : sans ce contrôle, l'advection suivante déborde. 2 laisse
    /// passer les accélérations ordinaires (démarrage d'un jet d'entrée : ×3).
    pub cfl_reject: f64,
    /// nombre de pas annulés puis refaits avec un dt plus petit (état invalide : NaN, T hors plage, masse négative, vitesse aberrante)
    pub rollbacks: usize,
    /// true si même le plus petit sous-pas reste invalide : l'état n'avance plus (la fenêtre se met en pause au lieu de passer en NaN)
    pub diverged: bool,

    // ---- cellules spéciales, solides thermiques, entrées (voir mp_cells.rs)
    /// effet de chaque cellule fluide (puits, émetteur, accélérateur, téléporteur)
    pub special: Vec<Option<MpEffect>>,
    /// comportement thermique de chaque cellule solide
    pub thermal: Vec<MpThermal>,
    /// entrée de bord par côté (gauche, droite, bas, haut) ; n'agit que si le bord est fermé (`Bc::Wall`)
    pub inlet: [Option<MpInlet>; 4],
    /// chaleur injectée par les solides chauffants depuis le début (J par m de profondeur)
    pub energy_input: f64,
    pub(crate) pin: Vec<Option<PinInfo>>,
    pub(crate) pf_u: Vec<f64>,
    pub(crate) pf_v: Vec<f64>,
    pub(crate) tp_mix: TpMix,

    // ---- réactions chimiques, flammes, rayonnement (voir mp_species.rs)
    /// composition du fluide ambiant (fractions VOLUMIQUES par phase, somme 1) ; vide = phase `ambient_phase` pure. Ex. air = 21 % O2 + 79 % N2
    pub ambient_mix: Vec<f64>,
    /// mélanges nommés (fractions volumiques) utilisables au pinceau (ex. « méthane–air stœchiométrique »)
    pub mixtures: Vec<(String, Vec<f64>)>,
    /// cinétique chimique active
    pub chemistry: bool,
    /// rayonnement des gaz chauds (H2O, CO2), modèle optiquement mince
    pub radiation: bool,
    /// épaisseur visée des flammes épaissies (cellules) : facteur F = flame_resolution·dx/δ_L
    pub flame_resolution: f64,
    /// variation relative de volume (à p, T de la cellule) que la chimie peut produire en un pas ; au-delà, l'intégration s'arrête et dt diminue
    pub chem_volume_limit: f64,
    /// facteur d'épaississement local F_eff (1 hors des flammes) : multiplie diffusion thermique et des espèces, divise la vitesse de réaction
    pub flame_factor: Vec<f64>,
    /// puissance chimique dégagée au dernier pas (W/m³, > 0 = exothermique)
    pub heat_release: Vec<f64>,
    /// chaleur de réaction dégagée depuis le début (J par m de profondeur) et énergie rayonnée (J/m)
    pub chem_heat: f64,
    pub radiated: f64,
    /// avancement cumulé de chaque réaction (mol par m de profondeur)
    pub reacted: Vec<f64>,
    /// pas de temps maximal imposé par la chimie au pas suivant (s)
    pub dt_chem: f64,
    /// chauffage au pinceau : température cible par cellule fluide (NaN = aucune), atteinte avec la constante de temps `heat_tau` (s de temps
    /// SIMULÉ) ; posée par l'outil « Chauffer », effacée par `clear_heat_targets` (la fenêtre le fait à chaque image)
    pub heat_target: Vec<f64>,
    pub heat_tau: f64,
    /// puissance volumique maximale du pinceau « Chauffer » (W/m³)
    pub heat_power_max: f64,
    /// fraction du volume d'une cellule que l'ébullition peut transformer en vapeur en un pas (le pas de temps suivant s'adapte)
    pub phase_volume_limit: f64,
    /// temps de relaxation de l'écart à la saturation (s), modèle de Lee : l'ébullition ramène T vers T_sat avec ce temps caractéristique
    pub phase_relaxation_time: f64,
    /// pas de temps maximal imposé par le changement de phase au pas suivant (s)
    pub dt_phase: f64,
    pub(crate) heat_active: bool,
    /// profil de temps de calcul par partie du pas (voir `mp_step::StepProfile`)
    pub profile: super::mp_step::StepProfile,
}

/// Pointeur partagé entre fils pour des écritures à des indices DISJOINTS (une cellule = un seul fil).
#[derive(Clone, Copy)]
pub(crate) struct SyncPtr<T>(pub *mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}
impl<T> SyncPtr<T> {
    /// (méthode : force la capture du wrapper entier par les fermetures, pas du seul pointeur brut)
    #[inline(always)]
    pub fn at(&self, i: usize) -> *mut T {
        unsafe { self.0.add(i) }
    }
}

/// `x` borné à [lo, hi] SANS paniquer si une borne est NaN (f64::clamp panique) : un NaN né pendant un pas doit aller jusqu'au
/// contrôle de fin de pas (retour arrière du filet de sécurité), pas arrêter le programme.
#[inline]
pub(crate) fn clampf(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// Température d'un mélange de masses `m` (kg/m³) d'énergie interne totale `e` (J/m³) : résout Σ m_k e_k(T) = e.
/// e_k(T) = c_k T + ½ s_k (T − 298.15)² + e0_k ⇒ trinôme du second degré en τ = T − 298.15 (forme stable sans annulation).
pub fn temp_from_masses(phases: &[super::phases::Phase], m: impl Iterator<Item = f64>, e: f64) -> f64 {
    let (mut cap, mut slope, mut off) = (0.0, 0.0, 0.0);
    for (k, mk) in m.enumerate() {
        let p = &phases[k];
        cap += mk * p.c_e();
        slope += mk * p.cp_slope;
        off += mk * p.e0;
    }
    temp_from_parts(cap, slope, off, e)
}

/// Même inversion à partir de C = Σ m c_e, S = Σ m s, E0 = Σ m e0 (sous-ensemble de phases : un groupe).
pub fn temp_from_parts(cap: f64, slope: f64, off: f64, e: f64) -> f64 {
    let t0 = super::phases::T_STD;
    let rhs = e - off - cap * t0; // = C τ + ½ S τ²
    if slope.abs() <= 1e-12 * cap.abs().max(1e-300) {
        return t0 + rhs / cap.max(1e-30);
    }
    let disc = cap * cap + 2.0 * slope * rhs;
    if disc <= 0.0 {
        return t0 - cap / slope; // minimum de e(T) : énergie hors du domaine du modèle (ne se produit pas en pratique)
    }
    t0 + 2.0 * rhs / (cap + disc.sqrt()).max(1e-30)
}

impl MultiphaseSim {
    pub fn new(nx: usize, ny: usize, dx: f64, reg: Registry, ambient_phase: usize) -> Self {
        let n = nx * ny;
        let np = reg.phases.len();
        let nch = reg.changes.len();
        let nre = reg.reactions.len();
        let groups = reg.miscible_groups();
        let mut group_of = vec![0; np];
        for (g, members) in groups.iter().enumerate() {
            for &k in members {
                group_of[k] = g;
            }
        }
        let ng = groups.len().max(1);
        MultiphaseSim {
            nx,
            ny,
            dx,
            groups,
            group_of,
            eg: vec![vec![0.0; n]; ng],
            tg: vec![vec![293.15; n]; ng],
            phase_heat_exchange: 1.0,
            reg,
            gravity: [0.0, -9.81],
            bc: [Bc::Wall; 4],
            ambient_phase,
            t_ambient: 293.15,
            mass: vec![vec![0.0; n]; np],
            alpha: vec![vec![0.0; n]; np],
            e: vec![0.0; n],
            temp: vec![293.15; n],
            p: vec![101325.0; n],
            u: vec![0.0; (nx + 1) * ny],
            v: vec![0.0; nx * (ny + 1)],
            solid: vec![-1; n],
            rho_mix: vec![1.0; n],
            mu_cell: vec![0.0; n],
            k_cell: vec![0.0; n],
            cfl: 0.4,
            maccormack: false,
            cap_safety: 0.3,
            dt_max: 1e-2,
            surface_tension: true,
            viscous: true,
            conduction: true,
            phase_change: true,
            pressure_tol: 1e-10,
            pressure_max_iter: 600,
            multigrid: true,
            nucleation_superheat: 4.0,
            nucleation_alpha: 1e-3,
            nuclei_density: 1e13,
            boil_superheat_max: 15.0,
            time: 0.0,
            last_dt: 0.0,
            last_pressure_iters: 0,
            last_newton_iters: 0,
            pressure_clamped: vec![false; n],
            pc_dvol: vec![vec![0.0; n]; ng],
            boundary_flux: vec![0.0; np],
            transferred: vec![0.0; nch],
            step_count: 0,
            mfx: vec![0.0; (nx + 1) * ny],
            mfy: vec![0.0; nx * (ny + 1)],
            m_tot_old: vec![0.0; n],
            conservative_momentum: true,
            predictor_corrector: false,
            flame_thermo_substeps: 4,
            cfl_flame: 0.1,
            flame_speed_floor: 2.5,
            adv_vel: None,
            dt_force: None,
            pc_predicting: false,
            speed_limit: 1000.0,
            closure_relax: 0.25,
            closure_relax_scale: 1.0,
            cfl_reject: 2.0,
            rollbacks: 0,
            diverged: false,
            special: vec![None; n],
            thermal: vec![MpThermal::Passive; n],
            inlet: [None; 4],
            energy_input: 0.0,
            pin: vec![None; n],
            pf_u: vec![f64::NAN; (nx + 1) * ny],
            pf_v: vec![f64::NAN; nx * (ny + 1)],
            tp_mix: TpMix::new(),
            ambient_mix: vec![],
            mixtures: vec![],
            chemistry: true,
            radiation: nre > 0,
            flame_resolution: 6.0,
            chem_volume_limit: 0.5,
            flame_factor: vec![1.0; n],
            heat_release: vec![0.0; n],
            chem_heat: 0.0,
            radiated: 0.0,
            reacted: vec![0.0; nre],
            dt_chem: f64::INFINITY,
            heat_target: vec![f64::NAN; n],
            heat_tau: 0.2,
            heat_power_max: 3e8,
            phase_volume_limit: 0.25,
            phase_relaxation_time: 1e-4,
            dt_phase: f64::INFINITY,
            heat_active: false,
            profile: Default::default(),
        }
    }

    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }
    #[inline]
    pub fn uidx(&self, i: usize, j: usize) -> usize {
        j * (self.nx + 1) + i
    }
    #[inline]
    pub fn vidx(&self, i: usize, j: usize) -> usize {
        j * self.nx + i
    }
    pub fn is_solid(&self, c: usize) -> bool {
        self.solid[c] >= 0
    }
    pub fn center(&self, i: usize, j: usize) -> (f64, f64) {
        ((i as f64 + 0.5) * self.dx, (j as f64 + 0.5) * self.dx)
    }

    // ---------------------------------------------------------------- initialisation
    /// Remplit les fractions volumiques par sur-échantillonnage 8×8 : `f(x,y)` donne la phase (None = laisse tel quel).
    /// À appeler sur un domaine vierge (α = 0) ou pour superposer ; puis `finalize_init`.
    pub fn paint_phase(&mut self, f: impl Fn(f64, f64) -> Option<usize>) {
        let s = 8usize;
        for j in 0..self.ny {
            for i in 0..self.nx {
                let c = self.idx(i, j);
                let mut cnt = vec![0.0; self.reg.phases.len()];
                let mut tot = 0.0;
                for b in 0..s {
                    for a in 0..s {
                        let x = (i as f64 + (a as f64 + 0.5) / s as f64) * self.dx;
                        let y = (j as f64 + (b as f64 + 0.5) / s as f64) * self.dx;
                        if let Some(k) = f(x, y) {
                            cnt[k] += 1.0;
                            tot += 1.0;
                        }
                    }
                }
                if tot > 0.0 {
                    let keep = 1.0 - tot / (s * s) as f64;
                    for k in 0..cnt.len() {
                        self.alpha[k][c] = self.alpha[k][c] * keep + cnt[k] / (s * s) as f64;
                    }
                }
            }
        }
    }
    /// Comme `paint_phase` avec un MÉLANGE : `f(x,y)` donne des fractions volumiques (somme 1) ou None.
    pub fn paint_mixture(&mut self, f: impl Fn(f64, f64) -> Option<Vec<f64>>) {
        let s = 8usize;
        let np = self.reg.phases.len();
        for j in 0..self.ny {
            for i in 0..self.nx {
                let c = self.idx(i, j);
                let mut cnt = vec![0.0; np];
                let mut tot = 0.0;
                for b in 0..s {
                    for a in 0..s {
                        let x = (i as f64 + (a as f64 + 0.5) / s as f64) * self.dx;
                        let y = (j as f64 + (b as f64 + 0.5) / s as f64) * self.dx;
                        if let Some(m) = f(x, y) {
                            for k in 0..np.min(m.len()) {
                                cnt[k] += m[k];
                            }
                            tot += 1.0;
                        }
                    }
                }
                if tot > 0.0 {
                    let keep = 1.0 - tot / (s * s) as f64;
                    for k in 0..np {
                        self.alpha[k][c] = self.alpha[k][c] * keep + cnt[k] / (s * s) as f64;
                    }
                }
            }
        }
    }
    /// Marque des cellules solides : `f(x,y)` → Some(indice solide). Les cellules solides sont étanches (u·n = 0).
    pub fn paint_solid(&mut self, f: impl Fn(f64, f64) -> Option<usize>) {
        for j in 0..self.ny {
            for i in 0..self.nx {
                let (x, y) = self.center(i, j);
                if let Some(s) = f(x, y) {
                    let c = self.idx(i, j);
                    self.solid[c] = s as i32;
                }
            }
        }
    }

    /// Fixe T (fonction de la position), pose une pression hydrostatique (colonnes depuis le bord supérieur), calcule m_k et e.
    pub fn finalize_init(&mut self, t_fn: impl Fn(f64, f64) -> f64) {
        let (nx, ny) = (self.nx, self.ny);
        for j in 0..ny {
            for i in 0..nx {
                let c = self.idx(i, j);
                let (x, y) = self.center(i, j);
                self.temp[c] = t_fn(x, y);
                // fractions → somme 1 (la phase ambiante complète le vide)
                if !self.is_solid(c) {
                    let s: f64 = (0..self.reg.phases.len()).map(|k| self.alpha[k][c]).sum();
                    if s < 1.0 - 1e-12 {
                        let amb = self.ambient_alpha();
                        for k in 0..self.reg.phases.len() {
                            self.alpha[k][c] += (1.0 - s) * amb[k];
                        }
                    } else if s > 1.0 + 1e-12 {
                        for k in 0..self.reg.phases.len() {
                            self.alpha[k][c] /= s;
                        }
                    }
                }
            }
        }
        let p_top = match self.bc[3] {
            Bc::Open { p } => p,
            _ => 101325.0,
        };
        let g = -self.gravity[1];
        for i in 0..nx {
            let mut p_above = p_top;
            let mut rho_above = 0.0;
            for j in (0..ny).rev() {
                let c = self.idx(i, j);
                if self.is_solid(c) {
                    self.p[c] = p_top;
                    p_above = p_top;
                    rho_above = 0.0;
                    continue;
                }
                let t = self.temp[c];
                let rho_of = |sim: &Self, p: f64| -> f64 { (0..sim.reg.phases.len()).map(|k| sim.alpha[k][c] * sim.reg.phases[k].rho(p, t)).sum() };
                let mut pc = p_above;
                for _ in 0..8 {
                    let rc = rho_of(self, pc);
                    pc = if j == ny - 1 {
                        p_top + g * 0.5 * self.dx * rc
                    } else {
                        p_above + g * self.dx * 0.5 * (rho_above + rc)
                    };
                }
                self.p[c] = pc;
                p_above = pc;
                rho_above = rho_of(self, pc);
            }
        }
        for c in 0..nx * ny {
            if self.is_solid(c) {
                let s = &self.reg.solids[self.solid[c] as usize];
                self.e[c] = s.rho_cp() * self.temp[c];
                for k in 0..self.reg.phases.len() {
                    self.mass[k][c] = 0.0;
                    self.alpha[k][c] = 0.0;
                }
                continue;
            }
            let (p, t) = (self.p[c], self.temp[c]);
            for k in 0..self.reg.phases.len() {
                self.mass[k][c] = self.alpha[k][c] * self.reg.phases[k].rho(p, t);
            }
            self.set_cell_temperature(c, t);
        }
        self.refresh();
    }

    // ---------------------------------------------------------------- propriétés dérivées
    /// Σ m_k c_k (J/(m³·K)) d'une cellule (solide : ρ_s c_s)
    /// ∂e/∂T à la température courante de la cellule (J/(m³·K)) (solide : ρ_s c_s)
    pub fn heat_capacity(&self, c: usize) -> f64 {
        if self.is_solid(c) {
            return self.reg.solids[self.solid[c] as usize].rho_cp();
        }
        let t = self.temp[c];
        (0..self.reg.phases.len()).map(|k| self.mass[k][c] * self.reg.phases[k].c_e_at(t)).sum()
    }
    pub fn temp_from_energy(&self, c: usize) -> f64 {
        if self.is_solid(c) {
            return self.e[c] / self.reg.solids[self.solid[c] as usize].rho_cp();
        }
        let np = self.reg.phases.len();
        temp_from_masses(&self.reg.phases, (0..np).map(|k| self.mass[k][c]), self.e[c])
    }
    /// Recalcule T (depuis e), α (depuis m, p, T), ρ_mix, μ, k de chaque cellule. Parallèle par lignes sur les grandes grilles ;
    /// les phases absentes d'une cellule (cas le plus fréquent) ne coûtent rien.
    pub fn refresh(&mut self) {
        let np = self.reg.phases.len();
        let (nx, ny) = (self.nx, self.ny);
        let n = nx * ny;
        let ng = self.groups.len();
        let temp = SyncPtr(self.temp.as_mut_ptr());
        let etot = SyncPtr(self.e.as_mut_ptr());
        let egp: Vec<SyncPtr<f64>> = self.eg.iter_mut().map(|a| SyncPtr(a.as_mut_ptr())).collect();
        let tgp: Vec<SyncPtr<f64>> = self.tg.iter_mut().map(|a| SyncPtr(a.as_mut_ptr())).collect();
        let rho_mix = SyncPtr(self.rho_mix.as_mut_ptr());
        let mu_cell = SyncPtr(self.mu_cell.as_mut_ptr());
        let k_cell = SyncPtr(self.k_cell.as_mut_ptr());
        let alpha: Vec<SyncPtr<f64>> = self.alpha.iter_mut().map(|a| SyncPtr(a.as_mut_ptr())).collect();
        let this = &*self;
        let row = |j: usize| {
            let mut vk = [0.0f64; super::phases::MAX_PHASES];
            let mut egn = [0.0f64; super::phases::MAX_PHASES];
            let mut tgn = [0.0f64; super::phases::MAX_PHASES];
            for i in 0..nx {
                let c = j * nx + i;
                // SAFETY : chaque cellule c n'est écrite que par la ligne j qui la contient (lignes réparties entre les fils)
                unsafe {
                    if this.solid[c] >= 0 {
                        let s = &this.reg.solids[this.solid[c] as usize];
                        *temp.at(c) = this.e[c] / s.rho_cp();
                        *rho_mix.at(c) = s.rho;
                        *k_cell.at(c) = s.conductivity;
                        *mu_cell.at(c) = 0.0;
                        continue;
                    }
                    // énergies et températures des groupes (traces alignées), e = Σ eg, température « de mélange »
                    let (e, tc) = this.sync_calc(c, &mut egn, &mut tgn, true);
                    *temp.at(c) = tc;
                    *etot.at(c) = e;
                    for g in 0..ng {
                        *egp[g].at(c) = egn[g];
                        *tgp[g].at(c) = tgn[g];
                    }
                    let p = this.p[c];
                    let mut vol = 0.0;
                    for k in 0..np {
                        let m = this.mass[k][c];
                        vk[k] = if m > 0.0 { m / this.reg.phases[k].rho(p, tgn[this.group_of[k]]) } else { 0.0 };
                        vol += vk[k];
                    }
                    let (mut rho, mut mu, mut kc) = (0.0, 0.0, 0.0);
                    let inv = 1.0 / vol.max(1e-300);
                    for k in 0..np {
                        let a = vk[k] * inv;
                        *alpha[k].at(c) = a;
                        rho += this.mass[k][c];
                        if a > 0.0 {
                            let tk = tgn[this.group_of[k]];
                            mu += a * this.reg.phases[k].mu(tk);
                            kc += a * this.reg.phases[k].k(tk);
                        }
                    }
                    *rho_mix.at(c) = rho * inv;
                    *mu_cell.at(c) = mu;
                    *k_cell.at(c) = kc;
                }
            }
        };
        if n >= 4096 && rayon::current_num_threads() > 1 {
            use rayon::prelude::*;
            (0..ny).into_par_iter().for_each(row);
        } else {
            (0..ny).for_each(row);
        }
    }

    // ---------------------------------------------------------------- diagnostics
    /// Masse totale par phase (kg par m de profondeur)
    pub fn total_mass(&self, k: usize) -> f64 {
        let a = self.dx * self.dx;
        (0..self.nx * self.ny).map(|c| self.mass[k][c]).sum::<f64>() * a
    }
    pub fn total_mass_all(&self) -> f64 {
        (0..self.reg.phases.len()).map(|k| self.total_mass(k)).sum()
    }
    pub fn total_internal_energy(&self) -> f64 {
        self.e.iter().sum::<f64>() * self.dx * self.dx
    }
    /// Surface occupée par la phase k (m²)
    pub fn phase_area(&self, k: usize) -> f64 {
        (0..self.nx * self.ny).map(|c| if self.is_solid(c) { 0.0 } else { self.alpha[k][c] }).sum::<f64>() * self.dx * self.dx
    }
    pub fn kinetic_energy(&self) -> f64 {
        let mut ke = 0.0;
        for j in 0..self.ny {
            for i in 0..=self.nx {
                let rho = self.face_rho_x(i, j);
                ke += 0.5 * rho * self.u[self.uidx(i, j)].powi(2);
            }
        }
        for j in 0..=self.ny {
            for i in 0..self.nx {
                let rho = self.face_rho_y(i, j);
                ke += 0.5 * rho * self.v[self.vidx(i, j)].powi(2);
            }
        }
        ke * self.dx * self.dx
    }
    /// Masse volumique à la face verticale (i−½, j) : moyenne arithmétique des deux cellules (bord : la cellule seule)
    pub fn face_rho_x(&self, i: usize, j: usize) -> f64 {
        if i == 0 {
            self.rho_mix[self.idx(0, j)]
        } else if i == self.nx {
            self.rho_mix[self.idx(self.nx - 1, j)]
        } else {
            0.5 * (self.rho_mix[self.idx(i - 1, j)] + self.rho_mix[self.idx(i, j)])
        }
    }
    pub fn face_rho_y(&self, i: usize, j: usize) -> f64 {
        if j == 0 {
            self.rho_mix[self.idx(i, 0)]
        } else if j == self.ny {
            self.rho_mix[self.idx(i, self.ny - 1)]
        } else {
            0.5 * (self.rho_mix[self.idx(i, j - 1)] + self.rho_mix[self.idx(i, j)])
        }
    }
    pub fn max_speed(&self) -> f64 {
        let mu = self.u.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let mv = self.v.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        mu.max(mv)
    }
    /// Fixe une température cible de chauffage (outil « Chauffer ») pour la cellule c.
    pub fn set_heat_target(&mut self, c: usize, t: f64) {
        if !self.is_solid(c) {
            self.heat_target[c] = t;
            self.heat_active = true;
        }
    }
    /// Supprime toutes les cibles de chauffage (fin du coup de pinceau).
    pub fn clear_heat_targets(&mut self) {
        if self.heat_active {
            self.heat_target.iter_mut().for_each(|t| *t = f64::NAN);
            self.heat_active = false;
        }
    }
    /// Relaxation vers les cibles pendant dt (implicite exacte : T → T_c + (T − T_c)·e^(−dt/τ)), énergie apportée comptée dans
    /// `energy_input`. Appelée à chaque pas par le solveur CPU ; la fenêtre l'appelle une fois par image en mode GPU.
    pub fn relax_heat_targets(&mut self, dt: f64) {
        if !self.heat_active {
            return;
        }
        let f = 1.0 - (-dt / self.heat_tau.max(1e-6)).exp();
        let a2 = self.dx * self.dx;
        for c in 0..self.nx * self.ny {
            let tt = self.heat_target[c];
            if tt.is_nan() || self.is_solid(c) {
                continue;
            }
            let lim = self.heat_power_max * dt;
            // chaque groupe tend vers la cible ; puissance totale bornée
            let ng = self.groups.len();
            let mut de = [0.0f64; super::phases::MAX_PHASES];
            let mut sum_abs = 0.0;
            for g in 0..ng {
                if self.group_mass(g, c) > 0.0 {
                    de[g] = self.group_cap_at(g, c, self.tg[g][c]) * (tt - self.tg[g][c]) * f;
                    sum_abs += de[g].abs();
                }
            }
            let scale = if sum_abs > lim { lim / sum_abs } else { 1.0 };
            for g in 0..ng {
                let d = de[g] * scale;
                if d.is_finite() {
                    self.eg[g][c] += d;
                    self.energy_input += d * a2;
                }
            }
            self.cell_sync(c);
        }
    }
    /// plus grand écart |T_cible − T| des cellules chauffées (0 sans chauffage)
    pub(crate) fn heat_gap(&self) -> f64 {
        if !self.heat_active {
            return 0.0;
        }
        // échauffement maximal par seconde : min(écart/τ, puissance max / capacité) ; ramené à un « écart équivalent » ×τ
        (0..self.nx * self.ny)
            .filter(|&c| !self.heat_target[c].is_nan())
            .map(|c| (self.heat_target[c] - self.temp[c]).abs().min(self.heat_power_max * self.heat_tau / self.heat_capacity(c).max(1e-30)))
            .fold(0.0, f64::max)
    }

    /// fractions volumiques du fluide ambiant (une seule phase si `ambient_mix` est vide)
    pub fn ambient_alpha(&self) -> Vec<f64> {
        let np = self.reg.phases.len();
        if self.ambient_mix.len() == np && self.ambient_mix.iter().sum::<f64>() > 0.0 {
            let s: f64 = self.ambient_mix.iter().sum();
            self.ambient_mix.iter().map(|a| a / s).collect()
        } else {
            (0..np).map(|k| if k == self.ambient_phase { 1.0 } else { 0.0 }).collect()
        }
    }
    pub fn liquid_phases(&self) -> Vec<usize> {
        (0..self.reg.phases.len()).filter(|&k| self.reg.phases[k].kind == PhaseKind::Liquid).collect()
    }
}
