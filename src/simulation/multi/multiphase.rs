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
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Bc {
    Wall,
    /// Bord ouvert à pression imposée (atmosphère). L'air/fluide entrant est la phase `ambient_phase` à `t_ambient`. `p` est la
    /// pression au point le plus haut du domaine (selon la gravité) : le long du bord, la pression suit l'hydrostatique du fluide
    /// ambiant (`MultiphaseSim::open_face_pressure`) — le bord du haut est à `p`.
    Open { p: f64 },
}

/// L'état complet est sérialisable (sauvegarde d'expérience, `mp_state`) : un champ ajouté ici est enregistré d'office (le format
/// bincode n'est pas auto-descriptif : ajouter, retirer ou réordonner un champ change le format — incrémenter `mp_state::VERSION`).
#[derive(Serialize, Deserialize)]
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
    /// flux de percolation (Darcy, m/s) de la dernière projection, transportés au pas suivant (fluide seul ; voir mp_percolation.rs)
    pub q_u: Vec<f64>,
    pub q_v: Vec<f64>,
    /// pression de CONTACT des grains σ' (Pa, ≥ 0) des lits perméables (cellules à deux pressions, `skel`) de la dernière projection :
    /// la pression du mélange y vaut p + σ', p étant celle du fluide des pores (voir mp_percolation.rs) ; 0 ailleurs
    pub sigma: Vec<f64>,
    /// cellules à deux pressions de la dernière projection (lit perméable posé sur un appui : squelette des grains incompressible)
    pub skel: Vec<bool>,
    pub solid: Vec<i32>, // −1 = fluide, sinon indice dans reg.solids
    /// corps rigides (solides mobiles, voir mp_bodies.rs et SOLID.md) — étape 0 : géométrie et fermeture seulement, pas encore couplés
    /// à l'écoulement ; aucun scénario n'en pose
    pub bodies: Vec<super::mp_bodies::RigidBody>,
    /// mécanismes à un degré de liberté (mp_mechanism.rs : bielle-manivelle, roues, pièces en formules) : ils mènent leurs corps
    pub mechanisms: Vec<super::mp_mechanism::Mechanism>,
    /// fraction de chaque maille occupée par les corps (0 sans corps), comptée dans la fermeture Σ m/ρ + α_corps = 1
    pub body_frac: Vec<f64>,
    /// vitesse normale du corps aux faces u et v bordées par une maille couverte (corps en mouvement ; vides sinon)
    pub body_vel_u: Vec<f64>,
    pub body_vel_v: Vec<f64>,
    /// corps qui couvre le plus chaque maille (−1 : aucun ; recalculé depuis la position des corps)
    #[serde(skip)]
    pub(crate) body_owner: Vec<i32>,
    /// force du fluide sur chaque corps à la dernière projection : (Fx, Fy) en N/m et couple autour du centre de masse en N
    #[serde(skip)]
    pub body_force: Vec<[f64; 3]>,
    /// part du frottement (contraintes visqueuses) dans `body_force`
    #[serde(skip)]
    pub body_force_visc: Vec<[f64; 3]>,
    /// force des contacts (bords, solides, autres corps ; amortisseur compris) et force d'une loi en formule sur chaque corps libre à la
    /// dernière projection (N/m, couple N) ; vides sans corps libre
    #[serde(skip)]
    pub body_contact: Vec<[f64; 3]>,
    #[serde(skip)]
    pub body_applied: Vec<[f64; 3]>,
    /// volume (fraction de maille) que les corps libres ont déplacé dans chaque maille pendant ce pas (`push_fluid`) ; vide sans
    /// corps libre
    #[serde(skip)]
    pub(crate) push_src: Vec<f64>,
    /// ressorts des contacts en cours des corps libres (mp_contacts.rs)
    #[serde(skip)]
    pub(crate) contact_springs: Vec<super::mp_contacts::ContactSpring>,
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
    /// plafond d'itérations du gradient conjugué du frottement des grains (mp_friction.rs)
    pub friction_max_iter: usize,
    /// viscosité au-delà de la borne explicite (0.2 ρ dx²/dt) traitée implicitement : un fluide très visqueux ne réduit plus le pas
    pub implicit_viscosity: bool,
    /// percolation : le fluide traverse les lits de grains tassés et les grilles (mp_percolation.rs)
    pub percolation: bool,
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
    /// itérations du gradient conjugué du frottement des grains au dernier pas
    pub last_friction_iters: usize,
    /// chaleur produite par le frottement des grains depuis le début (J par m de profondeur ; bilans)
    #[serde(skip)]
    pub friction_heat: f64,
    /// masse (kg par m de profondeur) entrée par les bords ouverts, par phase : M_k(t) − M_k(0) = flux_k − (changements de phase)
    pub boundary_flux: Vec<f64>,
    /// masse transformée liquide→vapeur cumulée par paire (kg/m), pour les bilans
    pub transferred: Vec<f64>,
    /// masse gelée (liquide → solide, négative à la fusion) cumulée par paire de fusion (kg/m), pour les bilans
    #[serde(default)]
    pub frozen: Vec<f64>,
    /// surfusion : avancée du front de dendrites dans chaque cellule, par paire de fusion (en mailles : la cellule germe à 1 ; les
    /// cellules germées gardent l'excédent, repris par leurs voisines)
    #[serde(default)]
    pub fusion_front: Vec<Vec<f64>>,
    pub(crate) step_count: usize,
    /// flux de masse total (toutes phases) à travers chaque face de cellule pendant le dernier advect_mass_energy (kg/m³ de cellule, + = vers +x / +y)
    pub(crate) mfx: Vec<f64>,
    pub(crate) mfy: Vec<f64>,
    pub(crate) m_tot_old: Vec<f64>,
    /// débits suivis (mesure « débit … à travers ligne ») : emplacement de chaque face de u et de v suivie (u32::MAX : aucune ; vides
    /// tant qu'aucune face n'est suivie) ; volume et masse de chaque phase passés par chaque emplacement depuis le début, comptés par
    /// le transport PLIC (m² et kg par m de profondeur, signés vers +x / +y) : `flux_acc[(2 · emplacement + 0 | 1) · np + k]`
    #[serde(skip)]
    pub(crate) flux_slot_u: Vec<u32>,
    #[serde(skip)]
    pub(crate) flux_slot_v: Vec<u32>,
    #[serde(skip)]
    pub flux_acc: Vec<f64>,
    /// advection de la quantité de mouvement conservative, cohérente avec les flux de masse du VOF (défaut) ; false = semi-lagrangien
    pub conservative_momentum: bool,
    /// Prédicteur-correcteur : le pas est d'abord calculé normalement (prédiction de u^{n+1}), puis refait depuis l'état n en
    /// advectant masses et énergie avec (u^n + u^{n+1})/2 (CPU et GPU, mêmes formules ; coût ~2 pas). Option, désactivée par défaut :
    /// la « boucle thermoacoustique » des flammes qu'il supprimait venait en fait de la correction de fermeture, qui recorrigeait
    /// l'erreur déjà en route dans l'advection du pas (voir solve_pressure) ; corrigée, une flamme est aussi juste sans lui
    /// (prémélange, référence ≈ 271 : 252 sans, 255 avec, en deux fois moins de temps).
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
    #[serde(skip)]
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
    if slope.abs() <= 1e-12 * cap.abs().max(1e-300) || !(cap > 1e-300) {
        return t0 + rhs / cap.max(1e-30);
    }
    // forme adimensionnée τ + ½ a τ² = b (a = S/C, b = rhs/C), miroir de `tau_from` sur le GPU : C² d'une trace (1e-165 kg de poudre
    // noire, C ~ 1e-162 J/K) sortait de la plage f64 — discriminant nul, T = 298 − c/s = −802 K et pas refusés en cascade
    let (a, b) = (slope / cap, rhs / cap);
    let disc = 1.0 + 2.0 * a * b;
    if disc <= 0.0 {
        return t0 - 1.0 / a; // minimum de e(T) : énergie hors du domaine du modèle (ne se produit pas en pratique)
    }
    t0 + 2.0 * b / (1.0 + disc.sqrt())
}

impl MultiphaseSim {
    pub fn new(nx: usize, ny: usize, dx: f64, reg: Registry, ambient_phase: usize) -> Self {
        let n = nx * ny;
        let np = reg.phases.len();
        let nch = reg.changes.len();
        let nfu = reg.fusions.len();
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
            q_u: vec![0.0; (nx + 1) * ny],
            q_v: vec![0.0; nx * (ny + 1)],
            sigma: vec![0.0; n],
            skel: vec![false; n],
            solid: vec![-1; n],
            bodies: vec![],
            body_frac: vec![0.0; n],
            body_vel_u: vec![],
            body_vel_v: vec![],
            body_owner: vec![],
            body_force: vec![],
            body_force_visc: vec![],
            body_contact: vec![],
            body_applied: vec![],
            push_src: vec![],
            contact_springs: vec![],
            mechanisms: vec![],
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
            friction_max_iter: 3000,
            implicit_viscosity: true,
            percolation: true,
            multigrid: true,
            nucleation_superheat: 4.0,
            nucleation_alpha: 1e-3,
            nuclei_density: 1e13,
            boil_superheat_max: 15.0,
            time: 0.0,
            last_dt: 0.0,
            last_pressure_iters: 0,
            last_newton_iters: 0,
            last_friction_iters: 0,
            friction_heat: 0.0,
            pressure_clamped: vec![false; n],
            pc_dvol: vec![vec![0.0; n]; ng],
            boundary_flux: vec![0.0; np],
            transferred: vec![0.0; nch],
            frozen: vec![0.0; nfu],
            fusion_front: vec![vec![0.0; n]; nfu],
            step_count: 0,
            mfx: vec![0.0; (nx + 1) * ny],
            mfy: vec![0.0; nx * (ny + 1)],
            m_tot_old: vec![0.0; n],
            flux_slot_u: vec![],
            flux_slot_v: vec![],
            flux_acc: vec![],
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
    /// Suit ce qui passe par des faces de u (`vertical`) ou de v (indices) : emplacement de chaque face dans `flux_acc`, nouveau (à
    /// zéro) si elle n'était pas encore suivie. Sur le GPU, à faire avant sa création (la place des emplacements y est fixée).
    pub fn watch_faces(&mut self, vertical: bool, faces: &[usize]) -> Vec<usize> {
        let np = self.reg.phases.len();
        if self.flux_slot_u.is_empty() {
            self.flux_slot_u = vec![u32::MAX; (self.nx + 1) * self.ny];
            self.flux_slot_v = vec![u32::MAX; self.nx * (self.ny + 1)];
        }
        let mut out = vec![];
        for &f in faces {
            let n_slots = self.flux_acc.len() / (2 * np);
            let s = if vertical { &mut self.flux_slot_u[f] } else { &mut self.flux_slot_v[f] };
            if *s == u32::MAX {
                *s = n_slots as u32;
                self.flux_acc.resize((n_slots + 1) * 2 * np, 0.0);
            }
            out.push(*s as usize);
        }
        out
    }
    /// Volume `vol` (fraction de maille) et masse `dm` (kg/m³ de maille) de la phase k passés de la maille d à sa voisine r hors du
    /// transport PLIC (sédimentation des grains, percolation) : comptés si leur face commune est suivie (débits mesurés).
    #[inline]
    pub(crate) fn count_transfer(&mut self, k: usize, d: usize, r: usize, vol: f64, dm: f64) {
        if self.flux_slot_u.is_empty() {
            return;
        }
        let (lo, hi) = (d.min(r), d.max(r));
        let (i, j) = (hi % self.nx, hi / self.nx);
        let slot = if hi == lo + 1 && i > 0 {
            self.flux_slot_u[self.uidx(i, j)]
        } else if hi == lo + self.nx {
            self.flux_slot_v[self.vidx(i, j)]
        } else {
            return;
        };
        if slot != u32::MAX {
            let (np, a2, sgn) = (self.reg.phases.len(), self.dx * self.dx, if r > d { 1.0 } else { -1.0 });
            let b = 2 * slot as usize * np;
            self.flux_acc[b + k] += sgn * vol * a2;
            self.flux_acc[b + np + k] += sgn * dm * a2;
        }
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
        // lits perméables : la pression posée est celle du mélange ; le fluide des pores est à p − σ' (σ' : poids déjaugé des grains
        // au-dessus ; sous une grille qui porte un lit, sans son poids), masses recalculées à cette pression
        if self.init_contact_stress() {
            for c in 0..nx * ny {
                if self.is_solid(c) {
                    continue;
                }
                let (p, t) = (self.p[c], self.temp[c]);
                for k in 0..self.reg.phases.len() {
                    self.mass[k][c] = self.alpha[k][c] * self.reg.phases[k].rho(p, t);
                }
                self.set_cell_temperature(c, t);
            }
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
    /// Recalcule T (depuis e), α (depuis m, p, T), ρ_mix, μ, k de chaque cellule. Parallèle par tranches de lignes (`mp_par`) ;
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
        let _ = n;
        super::mp_par::for_rows(ny, nx, super::mp_par::MIN_HEAVY, row);
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
    /// masse volumique du fluide ambiant (celui qui entre par les bords ouverts) à la pression p et à `t_ambient`
    pub(crate) fn ambient_density(&self, p: f64) -> f64 {
        self.ambient_alpha().iter().enumerate().map(|(k, a)| a * self.reg.phases[k].rho(p, self.t_ambient)).sum()
    }

    /// dρ/dp du fluide ambiant à la pression p (à la température ambiante) : sa compressibilité, pour l'hydrostatique des bords ouverts
    pub(crate) fn ambient_compressibility(&self, p: f64) -> f64 {
        let d = 1e-3 * p.abs().max(1.0);
        (self.ambient_density(p + d) - self.ambient_density(p - d)) / (2.0 * d)
    }

    /// Pression imposée à une face de bord ouvert de centre (x, y) : celle du bord, `po`, plus le poids de la colonne de fluide AMBIANT
    /// depuis le coin le plus haut du domaine selon la gravité — le bord du haut reste à `po`, les bords latéraux suivent l'hydrostatique
    /// de l'air extérieur. À pression constante, un bord latéral ouvert poussait sous la gravité un courant parasite (ρ g H ≈ 5 Pa sur
    /// 40 cm d'air : l'air sortait en bas et rentrait en haut). La colonne est COMPRESSIBLE, dp/dΦ = ρ(p) (Φ : potentiel de pesanteur
    /// depuis le haut), prise au second ordre : p = po + ρ_a Φ (1 + ½ ρ' Φ), ρ_a et ρ' = dρ/dp à `po` (`rho_a`, `drho`) — l'air
    /// intérieur au repos suit l'hydrostatique de ρ(p) (`finalize_init`) ; à masse volumique constante, l'écart au pied d'une colonne
    /// de 40 cm (1e-4 Pa) soufflait un vent qui croissait sans fin (4 mm/s après 5 s, côtés ouverts). Plus ⅛ ρ_a ρ' (g dx)² sous le
    /// haut : la demi-maille du haut de l'intérieur prend la masse volumique de sa maille, ce qui décale toute la colonne d'autant
    /// (1.7e-8 Pa dans l'air à 1 cm : 7 µm/s après 5 s sans ce terme).
    pub(crate) fn open_face_pressure(&self, po: f64, rho_a: f64, drho: f64, x: f64, y: f64) -> f64 {
        let (lx, ly) = (self.nx as f64 * self.dx, self.ny as f64 * self.dx);
        let (gx, gy) = (self.gravity[0], self.gravity[1]);
        let xr = if gx < 0.0 { lx } else { 0.0 };
        let yr = if gy < 0.0 { ly } else { 0.0 };
        let phi = gx * (x - xr) + gy * (y - yr);
        let half = if phi > 0.0 { 0.125 * rho_a * drho * (gx * gx + gy * gy) * self.dx * self.dx } else { 0.0 };
        po + rho_a * phi * (1.0 + 0.5 * drho * phi) + half
    }

    pub fn liquid_phases(&self) -> Vec<usize> {
        (0..self.reg.phases.len()).filter(|&k| self.reg.phases[k].kind == PhaseKind::Liquid).collect()
    }
}
