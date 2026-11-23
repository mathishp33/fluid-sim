//! Solveur multiphasique GPU RÉSIDENT (wgpu, f32), miroir du solveur CPU f64 (`multiphase.rs`, `mp_*.rs`).
//!
//! L'état (masses, énergie, pression, vitesses, propriétés dérivées, facteur de flamme, mélanges des téléporteurs, compteurs des bilans)
//! RESTE sur le GPU d'un pas à l'autre. Par pas, seuls ~256 octets sont relus (dt, itérations, drapeaux de contrôle) : le pas de temps
//! (CFL, gravité, capillarité, viscosité, chauffage, chimie, changement de phase), les itérations de Newton et du gradient conjugué
//! (multigrille), les sous-pas de conduction et de diffusion sont décidés SUR LE GPU (réductions), le filet de sécurité aussi
//! (instantané GPU → GPU, retour arrière avec dt/4 comme le CPU). L'image affichée est rendue sur le GPU (nx·ny·4 octets relus par
//! image) ; l'état complet n'est relu que sur demande (statistiques, outils, sauvegarde : `ensure_host`).
//!
//! Conditionnement f32 : énergie SENSIBLE relative ε = e − Σ m_k e_k(T0) (T0 = 298.15 K), pression relative p − 101325 Pa. Les tables
//! (fluides, solides, changements de phase, groupes miscibles, réactions, ln Kc(T) précalculé en f64) sont dans un buffer en lecture seule.
//! Différences avec le CPU f64 : arrondis f32 (≈ 1e-7 relatif ; Newton de la pression arrêté à max(0.1 Pa, 2e-6 p) au lieu de 1e-3 Pa,
//! gradient conjugué à 1e-6 relatif au lieu de 1e-10), préconditionneur toujours multigrille, échange de surface et ordre des cellules
//! d'un même pas évalués en parallèle (le CPU les traite en séquence).

use std::collections::HashMap;
use std::time::Instant;

use super::mp_cells::{MpEffect, MpThermal, PinInfo};
use super::mp_step::P_GPU;
use super::multiphase::{Bc, MultiphaseSim};
use super::phases::{Eos, Radiator, Thermal, MAX_PHASES, T_STD};

const SHADERS: [&str; 11] = [
    include_str!("mp_gpu/common.wgsl"),
    include_str!("mp_gpu/flow.wgsl"),
    include_str!("mp_gpu/granular.wgsl"),
    include_str!("mp_gpu/friction.wgsl"),
    include_str!("mp_gpu/percolation.wgsl"),
    include_str!("mp_gpu/mg.wgsl"),
    include_str!("mp_gpu/thermo.wgsl"),
    include_str!("mp_gpu/species.wgsl"),
    include_str!("mp_gpu/special.wgsl"),
    include_str!("mp_gpu/sl.wgsl"),
    include_str!("mp_gpu/render.wgsl"),
];

/// Nombre maximal de fluides du solveur GPU (= celui du CPU).
pub const GPU_MAX_PHASES: usize = 32;
/// nombre maximal de destinations de téléporteurs
const TP_CAP: usize = 256;
/// table ln Kc(T) : 200 K → 6000 K par pas de 5 K
const KC_T0: f64 = 200.0;
const KC_DT: f64 = 5.0;
const KC_N: usize = 1161;
/// niveaux multigrilles de moins de TAIL_CELLS cellules : traités par un seul groupe de travail
const TAIL_CELLS: usize = 8192;
const D_LEN: usize = 72;
const D_FR_IT: usize = 64;
const D_FR_UNCONV: usize = 65;
/// emplacements de paramètres de lancement (a, b) : a < SLOT_A pour b = 0, a < SLOT_A_SP pour b = groupe diffusant
const SLOT_A: usize = 2048;
const SLOT_A_SP: usize = 512;
const SLOT_BYTES: u64 = 256;
const PARAMS_WORDS: usize = 64;
/// groupes de travail des copies d'instantané du prédicteur-correcteur (boucle à pas SNAP_STRIDE = 1024 × 64 dans k_pc_save / k_pc_restore)
const PC_GROUPS: u32 = 1024;

/// passe d'un pas : simple, prédiction ou correction (prédicteur-correcteur)
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pass {
    Single,
    Pred,
    Corr,
}
/// lancements de noyaux au plus par tampon de commandes d'une série de pas (lavapipe plantait de temps en temps — non reproductible
/// en soumettant les lancements un par un — à 1200 avec la température par groupe, qui ajoute des noyaux plus longs)
const MAX_BATCH_OPS: usize = 400;

// cases du buffer D (miroir de common.wgsl)
const D_DT: usize = 0;
const D_DTCHEM: usize = 1;
const D_DTPHASE: usize = 2;
const D_DTBURN: usize = 66;
const D_SUB_C: usize = 4;
const D_NEWTON_IT: usize = 8;
const D_PCG_IT: usize = 10;
const D_BAD: usize = 19;
const D_MAXPCG_N: usize = 28;
const D_REDO: usize = 30;
const D_NSUB_S_MAX: usize = 31;
const D_PCG_UNCONV: usize = 48;
const D_B_TSUM: usize = 50;
const D_B_BAD: usize = 52;
const D_B_REDO: usize = 53;
const D_B_SUBC: usize = 54;
const D_B_NSUBS: usize = 55;
const D_B_MAXPCG: usize = 56;
const D_B_NEWTON: usize = 57;
const D_B_UNCONV: usize = 58;

// drapeaux (Params.flags)
const F_ST: u32 = 1;
const F_VISC: u32 = 2;
const F_COND: u32 = 4;
const F_PC: u32 = 8;
const F_CHEM: u32 = 16;
const F_EXTRAS: u32 = 32;
const F_CONS: u32 = 64;
const F_HEAT: u32 = 128;
const F_MISC: u32 = 256;
const F_FLAME: u32 = 512;
const F_MACC: u32 = 1024;
const F_GRAN: u32 = 2048;
const F_IMPVISC: u32 = 4096;
const F_FRIC: u32 = 8192;
const F_PERC: u32 = 16384;
const F_FUS: u32 = 32768;

const PH_STRIDE: usize = 36;
const SOL_STRIDE: usize = 8;
const PC_STRIDE: usize = 8;
/// paire de fusion : liquide, solide granulaire, température de fusion, chaleur latente, surfusion maximale, vitesse du front de dendrites
/// (a, b : v = a ΔT^b)
const FU_STRIDE: usize = 7;
const LEV_STRIDE: usize = 8;
const RX_STRIDE: usize = 56;

/// Le pas doit-il passer par le solveur de viscosité implicite ? Frottement des grains, ou un fluide dont la viscosité (jusqu'à 100 × μ_ref
/// selon sa loi en T) dépasserait la borne explicite 0.2 ρ dx²/dt au pas maximal (le gradient conjugué s'arrête sinon tout de suite)
fn implicit_viscosity_needed(sim: &MultiphaseSim, _nx: usize) -> bool {
    if !sim.viscous {
        return false;
    }
    if sim.has_friction() {
        return true;
    }
    sim.implicit_viscosity
        && sim.reg.phases.iter().any(|p| {
            let rho = p.rho(101325.0, p.t_ref).max(1e-3);
            100.0 * p.mu_ref > 0.2 * rho * sim.dx * sim.dx / sim.dt_max.max(1e-12)
        })
}

/// surfusion possible : changement de phase actif et une paire de fusion qui se surfond (front de dendrites, `fusion_step`)
fn fronts_possible(sim: &MultiphaseSim) -> bool {
    sim.phase_change && sim.reg.fusions.iter().any(|f| f.supercooling > 0.0)
}

/// La scène peut-elle percoler (miroir de `darcy_faces`) : grilles, ou phases granulaires de grains d'au moins 1 mm
fn percolation_possible(sim: &MultiphaseSim) -> bool {
    sim.percolation
        && (sim.special.iter().any(|s| matches!(s, Some(MpEffect::Grille { .. })))
            || sim.reg.phases.iter().any(|p| p.is_granular() && p.grain_diameter >= 1e-3))
}

/// Pourquoi le GPU ne peut PAS simuler cette scène (None = il le peut).
pub fn gpu_unsupported_reason(sim: &MultiphaseSim) -> Option<String> {
    if sim.reg.phases.len() > GPU_MAX_PHASES {
        return Some(format!("{} fluides (le GPU en gère {GPU_MAX_PHASES} au maximum)", sim.reg.phases.len()));
    }
    for r in &sim.reg.reactions {
        if r.reactants.len() > 6 || r.products.len() > 6 || r.modifiers.len() > 4 {
            return Some(format!("réaction « {} » : plus de 6 réactifs / 6 produits / 4 modulateurs", r.name));
        }
    }
    let dests: std::collections::HashSet<usize> = sim.special.iter().filter_map(|s| if let Some(MpEffect::Teleporter { dest, .. }) = s { Some(*dest) } else { None }).collect();
    if dests.len() > TP_CAP {
        return Some(format!("plus de {TP_CAP} destinations de téléporteurs"));
    }
    None
}

/// Disposition d'un buffer : (nom, début, longueur) en f32
#[derive(Default)]
struct Layout {
    items: Vec<(String, usize, usize)>,
    map: HashMap<String, (usize, usize)>,
    total: usize,
}
impl Layout {
    fn add(&mut self, name: &str, len: usize) {
        let len = len.max(1);
        self.items.push((name.to_string(), self.total, len));
        self.map.insert(name.to_string(), (self.total, len));
        self.total += len;
    }
    fn o(&self, name: &str) -> usize {
        self.map.get(name).unwrap_or_else(|| panic!("tampon GPU inconnu : {name}")).0
    }
}

/// une commande de calcul : noyau, groupes de travail, emplacement de paramètres de lancement (décalage dynamique)
#[derive(Clone, Copy)]
struct Op {
    k: &'static str,
    wg: u32,
    slot: u32,
}

/// Informations de pas relues sur le GPU.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuStepInfo {
    pub dt: f64,
    pub newton: usize,
    pub pcg: usize,
    pub pcg_max: usize,
    pub cond_sub: usize,
    pub species_sub: usize,
    pub retries: usize,
    pub redo: usize,
    /// lancements de noyaux du dernier essai
    pub dispatches: usize,
}

/// compteurs des bilans au moment du dernier envoi / de la dernière relecture (les accumulateurs GPU repartent de zéro)
#[derive(Clone, Default)]
struct Baselines {
    boundary_flux: Vec<f64>,
    transferred: Vec<f64>,
    energy_input: f64,
    chem_heat: f64,
    radiated: f64,
    reacted: Vec<f64>,
    frozen: Vec<f64>,
}

pub struct MultiphaseGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    s_buf: wgpu::Buffer,
    t_buf: wgpu::Buffer,
    d_buf: wgpu::Buffer,
    r_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    slot_buf: wgpu::Buffer,
    snap_s: wgpu::Buffer,
    snap_d: wgpu::Buffer,
    d_read: wgpu::Buffer,
    s_read: wgpu::Buffer,
    r_read: wgpu::Buffer,
    bg0: wgpu::BindGroup,
    bg1: wgpu::BindGroup,
    pl: wgpu::PipelineLayout,
    shader: wgpu::ShaderModule,
    pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    sl: Layout,
    tl: Layout,
    pers: (usize, usize),
    acc: (usize, usize),
    // configuration
    nx: usize,
    ny: usize,
    np: usize,
    ng: usize,
    npc: usize,
    nre: usize,
    nfus: usize,
    levels: Vec<(usize, usize)>,
    mg_tail: usize,
    slot_base: Vec<usize>,
    nwg: usize,
    diff_groups: Vec<usize>,
    surf_pairs: Vec<bool>,
    has_flames: bool,
    has_radiators: bool,
    /// adaptateur logiciel (lavapipe / llvmpipe) : contournements de pilote activés
    software: bool,
    /// nom de l'adaptateur (affichage)
    pub adapter_name: String,
    // état statique (recalculé à chaque envoi)
    has_extras: bool,
    has_accel: bool,
    has_thermal: bool,
    ntp: usize,
    /// groupes de téléporteurs (destinations)
    ntd: usize,
    slots_of_dest: Vec<usize>,
    heat_on: bool,
    // budgets adaptatifs (lancements préparés ; les noyaux en trop ne font rien)
    newton_budget: usize,
    pcg_budget: usize,
    /// itérations du gradient conjugué du frottement des grains soumises par pas (adaptées comme celles de la pression)
    fr_budget: usize,
    cond_budget: usize,
    sp_budget: usize,
    base: Baselines,
    /// l'état CPU (`sim`) est en retard sur le GPU (pas effectués depuis la dernière relecture)
    host_stale: bool,
    /// facteur appliqué à `closure_relax` pendant les retours arrière (0.25 par essai, comme `MultiphaseSim::step`)
    relax_scale: f64,
    pub last: GpuStepInfo,
}

fn wg(count: usize) -> u32 {
    count.div_ceil(64).max(1) as u32
}

impl MultiphaseGpu {
    pub fn try_new(sim: &MultiphaseSim) -> Result<Self, String> {
        if let Some(why) = gpu_unsupported_reason(sim) {
            return Err(why);
        }
        let (nx, ny) = (sim.nx, sim.ny);
        let np = sim.reg.phases.len();
        let n = nx * ny;
        let (nfu, nfv) = ((nx + 1) * ny, nx * (ny + 1));
        let nf = nfu.max(nfv);
        let groups = sim.reg.miscible_groups();
        let ng = groups.len();
        let npc = sim.reg.changes.len();
        let nre = sim.reg.reactions.len();
        let nfus = sim.reg.fusions.len();
        let npmax = np.max(1);
        // hiérarchie multigrille (mêmes règles que mp_mg::Multigrid::new)
        let mut levels = vec![(nx, ny)];
        loop {
            let &(a, b) = levels.last().unwrap();
            if !(a > 3 && b > 3 && a * b > 32) {
                break;
            }
            levels.push((a.div_ceil(2), b.div_ceil(2)));
        }
        let last = levels.len() - 1;
        let mg_tail = levels.iter().position(|&(a, b)| a * b <= TAIL_CELLS).unwrap_or(last).min(last);
        // (sommes partielles : cellules, faces d'une direction, ou faces u et v ensemble — gradient conjugué du frottement)
        let nwg = n.max(nf).max(nfu + nfv).div_ceil(64).max(1);

        // ---------------------------------------------------------------- disposition de S
        let mut sl = Layout::default();
        for name in ["solid", "pin_kind", "pin_phase", "pin_temp", "pin_tp", "grl_d", "grl_e"] {
            sl.add(name, n);
        }
        sl.add("pfu_on", nfu);
        sl.add("pfu", nfu);
        sl.add("pfv_on", nfv);
        sl.add("pfv", nfv);
        // faces imposées des effets et des entrées (copie fixe ; k_fl_hold y ajoute les faces des lits retenus par leur appui)
        sl.add("pfu_on0", nfu);
        sl.add("pfv_on0", nfv);
        for name in ["acc_on", "acc_x", "acc_y", "th_kind", "th_val", "heat_t", "ovl"] {
            sl.add(name, n);
        }
        let pers_start = sl.total;
        sl.add("mass", np * n);
        sl.add("eps", n);
        sl.add("epsg", ng.max(1) * n);
        sl.add("tg", ng.max(1) * n);
        sl.add("prel", n);
        sl.add("U", nfu);
        sl.add("V", nfv);
        // flux de percolation (Darcy) de la dernière projection, transportés au pas suivant
        sl.add("qu", nfu);
        sl.add("qv", nfv);
        sl.add("temp", n);
        sl.add("alpha", np * n);
        for name in ["rho_mix", "mu_cell", "k_cell", "clamped", "flame", "heat_rel"] {
            sl.add(name, n);
        }
        sl.add("tpmix", TP_CAP * (np + 2));
        // surfusion : avancée du front de dendrites par paire de fusion (miroir de `fusion_front`)
        sl.add("fu_phi", nfus.max(1) * n);
        let acc_start = sl.total;
        sl.add("bflux", np * n);
        sl.add("acc_ein", n);
        sl.add("acc_tr", npc.max(1) * n);
        sl.add("acc_rad", n);
        sl.add("acc_chem", n);
        sl.add("acc_xi", nre.max(1) * n);
        sl.add("acc_fu", nfus.max(1) * n);
        let pers_end = sl.total;
        // prédicteur-correcteur : u^{n+1} prédit, instantané de l'état persistant et de D
        sl.add("Up", nfu);
        sl.add("Vp", nfv);
        sl.add("pcs", pers_end - pers_start);
        sl.add("pcd", D_LEN);
        sl.add("pcf", 8);
        // téléporteurs : sommes par groupe (masses absorbées, ε, capacité, pente, p·masse) pendant k_tp_absorb
        sl.add("tpacc", TP_CAP * (np + 4));
        for name in ["Un", "Us", "Fx", "ust"] {
            sl.add(name, nfu);
        }
        for name in ["Vn", "Vs", "Fy", "vst"] {
            sl.add(name, nfv);
        }
        for name in ["w1", "w2", "w3", "w4"] {
            sl.add(name, n);
        }
        sl.add("w5", (nx + 1) * (ny + 1));
        sl.add("aw", np * n);
        sl.add("an", np * n);
        sl.add("ag", ng * n);
        // phases granulaires : fractions des groupes renormalisées hors grains (interfaces), incréments de la sédimentation
        // (masses par phase + drapeau « cellule touchée », énergies par groupe)
        sl.add("agn", ng * n);
        sl.add("sdm", (np + 1) * n);
        sl.add("sde", ng.max(1) * n);
        // frottement des grains : vecteurs de faces u et v ensemble (gradient conjugué), viscosité, pression de contact, contact
        for name in ["fr_uf", "fr_r", "fr_z", "fr_p", "fr_ap", "fr_x"] {
            sl.add(name, nfu + nfv);
        }
        for name in ["fr_mu", "fr_mv", "fr_ps", "fr_frac"] {
            sl.add(name, n);
        }
        // percolation : milieu poreux des cellules, pression de contact, conductances et termes moteurs des faces, transport
        for name in ["dc_k", "dc_e", "dc_d", "dc_rf", "dc_mu", "dc_ps", "dc_ct", "pq_s", "pq_a"] {
            sl.add(name, n);
        }
        for name in ["dcu", "ddu"] {
            sl.add(name, nfu);
        }
        for name in ["dcv", "ddv"] {
            sl.add(name, nfv);
        }
        sl.add("scale", np * n);
        sl.add("fl_vol", nf * np);
        for name in ["fdon", "frcv", "fghost", "ftf", "ftg"] {
            sl.add(name, nf);
        }
        sl.add("mfx", nfu);
        sl.add("mfy", nfv);
        for name in [
            "mtot_old", "p_start", "floor", "vol", "kappa", "err0", "chi", "rhs", "sd", "sae", "san", "sact", "cx", "cr", "cz", "cp", "cap",
            "sx_dm", "sx_tgt", "sx_e", "mgs", "newe", "omega", "mask", "mtmp",
        ] {
            sl.add(name, n);
        }
        // nœuds (cellule, groupe) : températures de la projection, conduction
        for name in ["tst", "tk", "kc2", "capc", "tc", "tn", "agc", "pc_dv"] {
            sl.add(name, ng.max(1) * n);
        }
        for name in ["ch_m", "ch_s", "ch_b", "ch_o", "ch_cons"] {
            sl.add(name, npmax * n);
        }
        for name in ["ch_xi", "ch_xs", "ch_r0"] {
            sl.add(name, nre.max(1) * n);
        }
        sl.add("dloc", npmax * n);
        // fluidisation : traînée de percolation, force de décharge des contacts, cellules retenues (k_fl_drag, k_fl_state)
        for name in ["fl_drag", "fl_fc", "fl_held"] {
            sl.add(name, n);
        }
        // surfusion : germes relevés avant le front de dendrites (k_fusion_germ)
        sl.add("fu_germ", nfus.max(1) * n);
        sl.add("ys", npmax * n);
        sl.add("newm", npmax * n);
        sl.add("part", 8 * nwg);
        for (l, &(a, b)) in levels.iter().enumerate().skip(1) {
            for q in ["d", "ae", "an", "act", "x", "b"] {
                sl.add(&format!("mg{l}_{q}"), a * b);
            }
        }

        // ---------------------------------------------------------------- disposition de T
        let nrev = sim.reg.reactions.iter().filter(|r| r.reversible).count();
        let mut tl = Layout::default();
        tl.add("PH", np * PH_STRIDE);
        tl.add("SOL", sim.reg.solids.len().max(1) * SOL_STRIDE);
        tl.add("PC", npc.max(1) * PC_STRIDE);
        tl.add("GRP", ng.max(1) * (1 + npmax));
        tl.add("GORD", ng.max(1));
        tl.add("GDISP", ng.max(1)); // groupe dispersé (uniquement des phases granulaires)
        tl.add("AMB", np);
        tl.add("INL", 16);
        tl.add("LEV", levels.len() * LEV_STRIDE);
        tl.add("RX", nre.max(1) * RX_STRIDE);
        tl.add("KC", nrev.max(1) * KC_N);
        tl.add("TP", n.max(1) * 2); // paires (source, groupe) : toutes les sources possibles
        tl.add("FU", nfus.max(1) * FU_STRIDE);

        // ---------------------------------------------------------------- source WGSL
        let mut header = String::from("// ---- constantes générées par gpu_multiphase.rs ----\n");
        for (name, off, _) in &sl.items {
            header += &format!("const O_{name}: u32 = {off}u;\n");
        }
        for (name, off, _) in &tl.items {
            header += &format!("const T_{name}: u32 = {off}u;\n");
        }
        header += &format!("const PERS_START: u32 = {pers_start}u;\nconst PERS_LEN: u32 = {}u;\n", pers_end - pers_start);
        header += &format!(
            "const NPMAX: u32 = {npmax}u;\nconst NREMAX: u32 = {}u;\nconst NWG: u32 = {nwg}u;\nconst PH_STRIDE: u32 = {PH_STRIDE}u;\nconst SOL_STRIDE: u32 = {SOL_STRIDE}u;\n\
             const PC_STRIDE: u32 = {PC_STRIDE}u;\nconst GRP_STRIDE: u32 = {}u;\nconst LEV_STRIDE: u32 = {LEV_STRIDE}u;\nconst RX_STRIDE: u32 = {RX_STRIDE}u;\n\
             const KC_N: u32 = {KC_N}u;\nconst KC_T0: f32 = {KC_T0:.1};\nconst KC_DT: f32 = {KC_DT:.1};\n",
            nre.max(1),
            1 + npmax
        );
        header += &format!(
            "const NFU: u32 = {nfus}u;\nconst FU_STRIDE: u32 = {FU_STRIDE}u;\nconst FU_SEED: f32 = {:?};\nconst HOLD_DRAG: f32 = {:?};\n",
            super::phases::FUSION_SEED_ALPHA as f32,
            super::mp_percolation::HOLD_DRAG as f32
        );
        let mut source = format!("{header}{}", SHADERS.join("\n"));
        // diagnostic : substitutions dans le source (« ancien=>nouveau@@… »)
        if let Ok(patch) = std::env::var("MP_WGSL_PATCH") {
            for pair in patch.split("@@") {
                if let Some((a, b)) = pair.split_once("=>") {
                    source = source.replace(a, b);
                }
            }
        }
        if let Ok(path) = std::env::var("MP_WGSL_DUMP") {
            let _ = std::fs::write(path, &header);
        }

        // ---------------------------------------------------------------- périphérique
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends: wgpu::Backends::PRIMARY, ..Default::default() });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or("aucun adaptateur GPU (Vulkan / Metal / DX12)")?;
        let info = adapter.get_info();
        let software = info.device_type == wgpu::DeviceType::Cpu || info.name.to_lowercase().contains("llvmpipe");
        let adapter_name = format!("{} ({:?})", info.name, info.backend);
        let s_bytes = (sl.total * 4) as u64;
        let mut limits = wgpu::Limits::default();
        let al = adapter.limits();
        if s_bytes > limits.max_storage_buffer_binding_size as u64 {
            limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
            limits.max_buffer_size = al.max_buffer_size;
        }
        if s_bytes > limits.max_storage_buffer_binding_size as u64 || s_bytes > limits.max_buffer_size {
            return Err(format!("grille trop grande pour le GPU ({:.0} Mo dans un seul buffer)", s_bytes as f64 / 1e6));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor { label: Some("multiphase GPU"), required_features: wgpu::Features::empty(), required_limits: limits },
            None,
        ))
        .map_err(|e| format!("périphérique GPU : {e}"))?;

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("multiphase resident"), source: wgpu::ShaderSource::Wgsl(source.into()) });
        if let Some(e) = pollster::block_on(device.pop_error_scope()) {
            return Err(format!("compilation des shaders GPU : {e}"));
        }
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        let mk = |label: &str, size: u64, usage: wgpu::BufferUsages| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16), usage, mapped_at_creation: false });
        let s_buf = mk("S", s_bytes, st);
        let t_buf = mk("T", (tl.total * 4) as u64, st);
        let d_buf = mk("D", (D_LEN * 4) as u64, st);
        let r_buf = mk("R", (n * 4) as u64, st);
        let params_buf = mk("params", (PARAMS_WORDS * 4) as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        // emplacements de paramètres de lancement : (a, b) en 256 octets chacun ; b = 0 : a < SLOT_A ; b = groupe diffusant : a < SLOT_A_SP
        let diff_groups: Vec<usize> = (0..ng).filter(|&g| groups[g].len() > 1 && groups[g].iter().any(|&k| sim.reg.phases[k].diffusivity > 0.0)).collect();
        let mut slot_base = vec![usize::MAX; ng.max(1)];
        slot_base[0] = 0;
        let mut nslots = SLOT_A;
        for &g in &diff_groups {
            if g > 0 {
                slot_base[g] = nslots;
                nslots += SLOT_A_SP;
            }
        }
        let words = SLOT_BYTES as usize / 4;
        let mut slots = vec![0u32; words * nslots];
        for a in 0..SLOT_A {
            slots[a * words] = a as u32;
        }
        for &g in &diff_groups {
            if g > 0 {
                for a in 0..SLOT_A_SP {
                    slots[(slot_base[g] + a) * words] = a as u32;
                    slots[(slot_base[g] + a) * words + 1] = g as u32;
                }
            }
        }
        let slot_buf = mk("slots", SLOT_BYTES * nslots as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let pers_len = pers_end - pers_start;
        let snap_s = mk("snapshot S", (pers_len * 4) as u64, wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
        let snap_d = mk("snapshot D", (D_LEN * 4) as u64, wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
        let rd = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
        let d_read = mk("D readback", (D_LEN * 4) as u64, rd);
        let s_read = mk("S readback", (pers_len * 4) as u64, rd);
        let r_read = mk("R readback", (n * 4) as u64, rd);
        queue.write_buffer(&slot_buf, 0, bytemuck::cast_slice(&slots));

        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mp bgl0"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mp bgl1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: wgpu::BufferSize::new(16) },
                count: None,
            }],
        });
        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mp bg0"),
            layout: &bgl0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: s_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: t_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: d_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: r_buf.as_entire_binding() },
            ],
        });
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mp bg1"),
            layout: &bgl1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &slot_buf, offset: 0, size: wgpu::BufferSize::new(16) }),
            }],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("mp pl"), bind_group_layouts: &[&bgl0, &bgl1], push_constant_ranges: &[] });

        let surf_pairs: Vec<bool> = sim
            .reg
            .changes
            .iter()
            .map(|pc| groups.iter().find(|g| g.contains(&pc.vapor)).map_or(false, |g| g.iter().any(|&k| k != pc.vapor && sim.reg.phases[k].is_gas())))
            .collect();
        let has_flames = sim.reg.reactions.iter().any(|r| r.flame_thickness > 0.0);
        let has_radiators = sim.reg.phases.iter().any(|p| p.radiator != Radiator::None && p.is_gas());

        let mut g = MultiphaseGpu {
            device,
            queue,
            s_buf,
            t_buf,
            d_buf,
            r_buf,
            params_buf,
            slot_buf,
            snap_s,
            snap_d,
            d_read,
            s_read,
            r_read,
            bg0,
            bg1,
            pl,
            shader,
            pipelines: HashMap::new(),
            sl,
            tl,
            pers: (pers_start, pers_len),
            acc: (acc_start, pers_end - acc_start),
            nx,
            ny,
            np,
            ng,
            npc,
            nre,
            nfus,
            levels,
            mg_tail,
            slot_base,
            nwg,
            diff_groups,
            surf_pairs,
            has_flames,
            has_radiators,
            software,
            adapter_name,
            has_extras: false,
            has_accel: false,
            has_thermal: false,
            ntp: 0,
            ntd: 0,
            slots_of_dest: vec![],
            heat_on: false,
            newton_budget: 3,
            pcg_budget: 30,
            fr_budget: 100,
            cond_budget: 8,
            sp_budget: 8,
            base: Baselines::default(),
            host_stale: false,
            relax_scale: 1.0,
            last: GpuStepInfo::default(),
        };
        let _ = (&g.r_buf, &g.slot_buf, g.ng, g.nwg);
        g.device.push_error_scope(wgpu::ErrorFilter::Validation);
        g.upload(sim);
        // les noyaux de base sont compilés tout de suite : une erreur de pilote apparaît ici et non au premier pas
        for k in ["k_refresh", "k_red_dt", "k_dt_final", "k_sanity", "k_sanity_final"] {
            g.pipeline(k);
        }
        if let Some(e) = pollster::block_on(g.device.pop_error_scope()) {
            return Err(format!("initialisation GPU : {e}"));
        }
        Ok(g)
    }

    fn pipeline(&mut self, k: &'static str) {
        if !self.pipelines.contains_key(k) {
            trace(&format!("pipeline {k}"));
            let p = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(k), layout: Some(&self.pl), module: &self.shader, entry_point: k, compilation_options: Default::default() });
            self.pipelines.insert(k, p);
        }
    }

    /// décalage dynamique de l'emplacement (Q.a = a, Q.b = b)
    fn slot(&self, a: usize, b: usize) -> u32 {
        let base = self.slot_base[b];
        let lim = if b == 0 { SLOT_A } else { SLOT_A_SP };
        debug_assert!(base != usize::MAX && a < lim);
        ((base + a.min(lim - 1)) as u64 * SLOT_BYTES) as u32
    }

    // ================================================================ envoi de l'état
    /// Envoie l'état CPU complet (et les tables) vers le GPU. À appeler après toute modification de `sim` hors du GPU (outils, scène).
    pub fn upload(&mut self, sim: &MultiphaseSim) {
        trace("upload");
        let (nx, ny, np) = (self.nx, self.ny, self.np);
        let n = nx * ny;
        let (nfu, nfv) = ((nx + 1) * ny, nx * (ny + 1));
        let ph = &sim.reg.phases;
        let h0: Vec<f64> = ph.iter().map(|p| p.e_int(T_STD)).collect();
        // ---------------- tables
        let mut t = vec![0f32; self.tl.total];
        let groups = sim.reg.miscible_groups();
        let mut gid = vec![0usize; np];
        for (g, m) in groups.iter().enumerate() {
            for &k in m {
                gid[k] = g;
            }
        }
        for (k, p) in ph.iter().enumerate() {
            let b = self.tl.o("PH") + k * PH_STRIDE;
            let mut r = [0f64; PH_STRIDE];
            r[0] = if p.is_gas() { 0.0 } else { 1.0 };
            r[1] = p.cp;
            r[2] = p.c_e();
            r[3] = p.cp_slope;
            match p.eos {
                Eos::IdealGas { r_spec } => r[4] = r_spec,
                Eos::Tait { p0, b, n, thermal } => {
                    r[4] = b;
                    r[5] = n;
                    r[31] = p0;
                    match thermal {
                        Thermal::Kell => r[6] = 1.0,
                        Thermal::Beta { rho0, t0, beta } => {
                            r[7] = rho0;
                            r[8] = t0;
                            r[9] = beta;
                        }
                    }
                }
            }
            let (mi, ma, mb) = p.mu_law.gpu_params();
            let (ki, ka, kb) = p.k_law.gpu_params();
            r[10] = p.mu_ref;
            r[11] = mi as f64;
            r[12] = ma as f64;
            r[13] = mb as f64;
            r[14] = p.conductivity;
            r[15] = ki as f64;
            r[16] = ka as f64;
            r[17] = kb as f64;
            r[18] = p.t_ref;
            r[19] = p.sigma;
            r[20] = p.molar_mass;
            r[21] = p.diffusivity;
            r[22] = h0[k];
            r[23] = p.e0;
            r[24] = p.s_standard;
            r[25] = match p.radiator {
                Radiator::None => 0.0,
                Radiator::Water => 1.0,
                Radiator::CarbonDioxide => 2.0,
            };
            r[26] = gid[k] as f64;
            // groupe réactif (une de ses phases intervient dans une réaction) : seul il participe à la chimie
            r[30] = if sim.reg.phases.iter().enumerate().any(|(j, _)| gid[j] == gid[k] && sim.reg.is_cell_reactive(j)) { 1.0 } else { 0.0 };
            r[27] = p.color[0] as f64;
            r[28] = p.color[1] as f64;
            r[29] = p.color[2] as f64;
            // grains : phase granulaire, diamètre, compaction maximale
            r[32] = if p.is_granular() { 1.0 } else { 0.0 };
            r[33] = p.grain_diameter;
            r[34] = p.alpha_max;
            r[35] = if p.is_granular() { p.friction_angle.to_radians().tan() } else { 0.0 };
            for q in 0..PH_STRIDE {
                t[b + q] = r[q] as f32;
            }
        }
        for (s, so) in sim.reg.solids.iter().enumerate() {
            let b = self.tl.o("SOL") + s * SOL_STRIDE;
            for (q, v) in [so.rho_cp(), so.conductivity, so.rho, so.cp, so.color[0] as f64, so.color[1] as f64, so.color[2] as f64].iter().enumerate() {
                t[b + q] = *v as f32;
            }
        }
        for (q, pc) in sim.reg.changes.iter().enumerate() {
            let b = self.tl.o("PC") + q * PC_STRIDE;
            for (i, v) in [pc.liquid as f64, pc.vapor as f64, pc.l_ref, pc.t_ref, pc.p_ref, pc.r_v, pc.dcp, h0[pc.vapor] - h0[pc.liquid]].iter().enumerate() {
                t[b + i] = *v as f32;
            }
        }
        for (q, fu) in sim.reg.fusions.iter().enumerate() {
            let b = self.tl.o("FU") + q * FU_STRIDE;
            for (i, v) in [fu.liquid as f64, fu.solid as f64, fu.t_melt, fu.l_fusion, fu.supercooling, fu.growth_a, fu.growth_b].iter().enumerate() {
                t[b + i] = *v as f32;
            }
        }
        let gs = 1 + np.max(1);
        for (g, m) in groups.iter().enumerate() {
            let b = self.tl.o("GRP") + g * gs;
            t[b] = m.len() as f32;
            for (q, &k) in m.iter().enumerate() {
                t[b + 1 + q] = k as f32;
            }
        }
        for (pos, g) in sim.group_order(&groups).into_iter().enumerate() {
            t[self.tl.o("GORD") + pos] = g as f32;
        }
        for (g, m) in groups.iter().enumerate() {
            t[self.tl.o("GDISP") + g] = if m.iter().all(|&k| ph[k].is_granular()) { 1.0 } else { 0.0 };
        }
        let amb = sim.ambient_alpha();
        for k in 0..np {
            t[self.tl.o("AMB") + k] = amb[k] as f32;
        }
        for side in 0..4 {
            let b = self.tl.o("INL") + 4 * side;
            if let (Some(inl), Bc::Wall) = (sim.inlet[side], sim.bc[side]) {
                let k = inl.phase.min(np - 1);
                t[b] = 1.0;
                t[b + 1] = k as f32;
                t[b + 2] = inl.temp as f32;
                t[b + 3] = if k == sim.ambient_phase { 1.0 } else { 0.0 };
            }
        }
        for (l, &(a, bb)) in self.levels.iter().enumerate() {
            let b = self.tl.o("LEV") + l * LEV_STRIDE;
            let offs = if l == 0 {
                [self.sl.o("sd"), self.sl.o("sae"), self.sl.o("san"), self.sl.o("sact"), self.sl.o("cz"), self.sl.o("cr")]
            } else {
                ["d", "ae", "an", "act", "x", "b"].map(|q| self.sl.o(&format!("mg{l}_{q}")))
            };
            t[b] = a as f32;
            t[b + 1] = bb as f32;
            for (q, o) in offs.iter().enumerate() {
                t[b + 2 + q] = *o as f32;
            }
        }
        let mut kc_next = self.tl.o("KC");
        for (j, r) in sim.reg.reactions.iter().enumerate() {
            let b = self.tl.o("RX") + j * RX_STRIDE;
            let (ma, mb) = r.mass_balance(ph);
            let mut row = [0f64; RX_STRIDE];
            row[0] = r.reactants.len() as f64;
            row[1] = r.products.len() as f64;
            row[2] = r.modifiers.len() as f64;
            row[3] = if r.reversible { 1.0 } else { 0.0 };
            row[4] = if r.flame_thickness > 0.0 { 1.0 } else { 0.0 };
            row[5] = if r.pre_exp > 0.0 { r.pre_exp.ln() } else { -1e30 };
            row[6] = r.temp_exponent;
            row[7] = r.activation_energy / super::phases::R_GAS;
            row[8] = if mb > 0.0 { ma / mb } else { 1.0 };
            for (i, &(s, nu)) in r.reactants.iter().enumerate() {
                row[10 + i] = s as f64;
                row[16 + i] = nu;
                row[22 + i] = r.order(i);
            }
            for (i, &(s, nu)) in r.products.iter().enumerate() {
                row[28 + i] = s as f64;
                row[34 + i] = nu;
            }
            for (i, &(s, o)) in r.modifiers.iter().enumerate() {
                row[40 + i] = s as f64;
                row[44 + i] = o;
            }
            // réaction de surface (réactif granulaire) : 48 drapeau, 49 réactif granulaire, 50 diamètre, 51 oxydant (−1 : aucun) ;
            // irréversible (miroir de CellChem)
            let surf = r.surface(ph);
            if let Some(sr) = surf {
                row[3] = 0.0;
                row[48] = 1.0;
                row[49] = sr.solid as f64;
                row[50] = sr.d;
                row[51] = sr.ox.map_or(-1.0, |o| o as f64);
            }
            // réaction en volume d'une phase granulaire (pyrolyse, `k_pyrolysis`) : 48 = −1, 49 réactif granulaire, 50 temps de conduction
            // interne de ses grains (k_eff = k / (1 + k τ))
            if r.bulk {
                row[48] = -1.0;
                row[49] = r.reactants.first().map_or(0.0, |&(k, _)| k as f64);
                row[50] = r.bulk_tau(ph);
            }
            // combustion d'une poudre (`Reaction::burn`) : 52 vitesse de régression à 1 atm, 53 exposant de Vieille, 54 température
            // d'allumage, 55 volume de gaz × p par kg de réactif (borne du pas)
            if let Some((a, ex, t_ign)) = r.burn() {
                row[52] = a;
                row[53] = ex;
                row[54] = t_ign;
                row[55] = r.burn_gas_volume(ph);
            }
            if r.reversible && surf.is_none() {
                row[9] = kc_next as f64;
                for i in 0..KC_N {
                    let tt = KC_T0 + KC_DT * i as f64;
                    t[kc_next + i] = r.ln_kc(ph, tt).clamp(-1e30, 1e30) as f32;
                }
                kc_next += KC_N;
            }
            for q in 0..RX_STRIDE {
                t[b + q] = row[q] as f32;
            }
        }
        // ---------------- cellules spéciales (comme prepare_special)
        let (pin, pf_u, pf_v) = sim.special_tables();
        // groupes de téléporteurs (miroir de special_tables / absorb_teleporters) : un groupe par destination, clé = `dest`
        let mut dests: Vec<usize> = vec![];
        let mut pairs: Vec<(usize, usize)> = vec![];
        for c in 0..n {
            if let Some(PinInfo { kind: 1, tp_dest: Some(d), .. }) = pin[c] {
                let s = match dests.iter().position(|&x| x == d) {
                    Some(s) => s,
                    None => {
                        dests.push(d);
                        dests.len() - 1
                    }
                };
                pairs.push((c, s));
            }
        }
        for &d in sim.tp_mix.keys() {
            let used = pin.iter().any(|p| matches!(p, Some(PinInfo { kind: 2, tp_dest: Some(k), .. }) if *k == d));
            if used && !dests.contains(&d) {
                dests.push(d);
            }
        }
        dests.truncate(TP_CAP);
        pairs.retain(|&(_, s)| s < TP_CAP);
        for (q, &(c, s)) in pairs.iter().enumerate() {
            t[self.tl.o("TP") + 2 * q] = c as f32;
            t[self.tl.o("TP") + 2 * q + 1] = s as f32;
        }
        self.ntp = pairs.len();
        self.ntd = dests.len();
        self.slots_of_dest = dests.clone();
        self.has_extras = sim.has_extras();
        self.queue.write_buffer(&self.t_buf, 0, bytemuck::cast_slice(&t));

        // ---------------- statique + persistant
        let mut s = vec![0f32; self.pers.0 + self.pers.1];
        let o = |name: &str| self.sl.o(name);
        let (o_solid, o_pk, o_pp, o_pt, o_ptp) = (o("solid"), o("pin_kind"), o("pin_phase"), o("pin_temp"), o("pin_tp"));
        let (o_acc_on, o_ax, o_ay, o_thk, o_thv, o_ht, o_ovl) = (o("acc_on"), o("acc_x"), o("acc_y"), o("th_kind"), o("th_val"), o("heat_t"), o("ovl"));
        let (o_grd, o_gre) = (o("grl_d"), o("grl_e"));
        self.has_accel = false;
        self.has_thermal = false;
        let mut ovl = vec![0u32; n];
        for c in 0..n {
            s[o_solid + c] = sim.solid[c] as f32;
            s[o_ptp + c] = -1.0;
            if let Some(MpEffect::Grille { hole, open }) = sim.special[c] {
                s[o_grd + c] = hole.max(1e-5) as f32;
                s[o_gre + c] = open as f32;
            }
            if let Some(pi) = pin[c] {
                s[o_pk + c] = pi.kind as f32;
                s[o_pp + c] = if pi.phase == sim.ambient_phase { -1.0 } else { pi.phase as f32 };
                s[o_pt + c] = pi.temp as f32;
                if let (2, Some(key)) = (pi.kind, pi.tp_dest) {
                    if let Some(sl) = dests.iter().position(|&d| d == key) {
                        s[o_ptp + c] = sl as f32;
                    }
                }
            }
            if !sim.is_solid(c) {
                if let Some(MpEffect::Accelerator { ax, ay }) = sim.special[c] {
                    s[o_acc_on + c] = 1.0;
                    s[o_ax + c] = ax as f32;
                    s[o_ay + c] = ay as f32;
                    self.has_accel = true;
                }
            } else {
                match sim.thermal[c] {
                    MpThermal::Isothermal(tt) => {
                        s[o_thk + c] = 1.0;
                        s[o_thv + c] = tt as f32;
                        self.has_thermal = true;
                    }
                    MpThermal::Heater(w) => {
                        s[o_thk + c] = 2.0;
                        s[o_thv + c] = w as f32;
                        self.has_thermal = true;
                    }
                    MpThermal::Passive => {}
                }
            }
            if sim.heat_active && !sim.is_solid(c) && !sim.heat_target[c].is_nan() {
                s[o_ht + c] = sim.heat_target[c] as f32;
            }
            let col = |r: u32, g: u32, b: u32| r | (g << 8) | (b << 16);
            match sim.special[c] {
                Some(MpEffect::Sink { .. }) => ovl[c] = col(0xff, 0, 0) + 1,
                Some(MpEffect::Emitter { .. }) => ovl[c] = col(0, 0xff, 0) + 1,
                Some(MpEffect::Accelerator { .. }) => ovl[c] = col(0xff, 0, 0xff) + 1,
                Some(MpEffect::Grille { .. }) => ovl[c] = col(0x9a, 0x9a, 0xa0) + 1,
                Some(MpEffect::Teleporter { dest, radius, .. }) => {
                    ovl[c] = col(0, 0, 0xff) + 1;
                    if dest < n {
                        for d in sim.tp_dest_cells(dest, radius) {
                            ovl[d] = col(0, 0xff, 0xff) + 1;
                        }
                    }
                }
                None => {}
            }
        }
        for c in 0..n {
            s[o_ovl + c] = ovl[c] as f32;
        }
        self.heat_on = sim.heat_active;
        let (o_pfuo, o_pfu, o_pfvo, o_pfv) = (o("pfu_on"), o("pfu"), o("pfv_on"), o("pfv"));
        let (o_pfuo0, o_pfvo0) = (o("pfu_on0"), o("pfv_on0"));
        for f in 0..nfu {
            if !pf_u[f].is_nan() {
                s[o_pfuo + f] = 1.0;
                s[o_pfuo0 + f] = 1.0;
                s[o_pfu + f] = pf_u[f] as f32;
            }
        }
        for f in 0..nfv {
            if !pf_v[f].is_nan() {
                s[o_pfvo + f] = 1.0;
                s[o_pfvo0 + f] = 1.0;
                s[o_pfv + f] = pf_v[f] as f32;
            }
        }
        // persistant
        let (o_m, o_e, o_p, o_u, o_v, o_t, o_a) = (o("mass"), o("eps"), o("prel"), o("U"), o("V"), o("temp"), o("alpha"));
        let (o_eg, o_tg) = (o("epsg"), o("tg"));
        let (o_rho, o_mu, o_k, o_cl, o_fl, o_hr, o_tpm) = (o("rho_mix"), o("mu_cell"), o("k_cell"), o("clamped"), o("flame"), o("heat_rel"), o("tpmix"));
        for c in 0..n {
            let mut ch = 0.0;
            for k in 0..np {
                let m = sim.mass[k][c];
                s[o_m + k * n + c] = m as f32;
                s[o_a + k * n + c] = sim.alpha[k][c] as f32;
                ch += m * h0[k];
            }
            s[o_e + c] = if sim.is_solid(c) { (sim.e[c] - sim.reg.solids[sim.solid[c] as usize].rho_cp() * T_STD) as f32 } else { (sim.e[c] - ch) as f32 };
            // énergies relatives et températures des groupes (ε_g = E_g − Σ_k∈g m_k h0_k), après adoption des retouches directes de
            // `e` comme le fait `refresh` sur le CPU (écart réparti au prorata des capacités)
            let ngr = sim.groups.len();
            if sim.is_solid(c) || ngr == 0 {
                for g in 0..ngr {
                    s[o_eg + g * n + c] = 0.0;
                    s[o_tg + g * n + c] = sim.temp[c] as f32;
                }
            } else {
                let (mut egn, mut tgn) = ([0.0f64; MAX_PHASES], [0.0f64; MAX_PHASES]);
                sim.sync_calc(c, &mut egn[..ngr], &mut tgn[..ngr], true);
                for (g, members) in sim.groups.iter().enumerate() {
                    let chg: f64 = members.iter().map(|&k| sim.mass[k][c] * h0[k]).sum();
                    s[o_eg + g * n + c] = (egn[g] - chg) as f32;
                    s[o_tg + g * n + c] = tgn[g] as f32;
                }
            }
            s[o_p + c] = (sim.p[c] - P0) as f32;
            s[o_t + c] = sim.temp[c] as f32;
            s[o_rho + c] = sim.rho_mix[c] as f32;
            s[o_mu + c] = sim.mu_cell[c] as f32;
            s[o_k + c] = sim.k_cell[c] as f32;
            s[o_cl + c] = if sim.pressure_clamped.get(c).copied().unwrap_or(false) { 1.0 } else { 0.0 };
            s[o_fl + c] = sim.flame_factor.get(c).copied().unwrap_or(1.0) as f32;
            s[o_hr + c] = sim.heat_release.get(c).copied().unwrap_or(0.0) as f32;
        }
        let o_fp = o("fu_phi");
        for (q, fr) in sim.fusion_front.iter().enumerate().take(self.nfus) {
            for (c, &v) in fr.iter().enumerate().take(n) {
                s[o_fp + q * n + c] = v as f32;
            }
        }
        let (o_qu, o_qv) = (o("qu"), o("qv"));
        for f in 0..nfu {
            s[o_u + f] = sim.u[f] as f32;
            s[o_qu + f] = sim.q_u.get(f).copied().unwrap_or(0.0) as f32;
        }
        for f in 0..nfv {
            s[o_qv + f] = sim.q_v.get(f).copied().unwrap_or(0.0) as f32;
            s[o_v + f] = sim.v[f] as f32;
        }
        for (slot, &d) in dests.iter().enumerate() {
            if let Some((a, tt)) = sim.tp_mix.get(&d) {
                let b = o_tpm + slot * (np + 2);
                s[b] = 1.0;
                s[b + 1] = *tt as f32;
                for k in 0..np.min(a.len()) {
                    s[b + 2 + k] = a[k] as f32;
                }
            }
        }
        self.queue.write_buffer(&self.s_buf, 0, bytemuck::cast_slice(&s));
        // scalaires dynamiques
        let mut d = vec![0f32; D_LEN];
        // pas précédent (croissance du pas bornée, voir choose_dt)
        d[D_DT] = sim.last_dt as f32;
        d[D_DTCHEM] = if sim.dt_chem.is_finite() { sim.dt_chem as f32 } else { 3e38 };
        d[D_DTPHASE] = if sim.dt_phase.is_finite() { sim.dt_phase as f32 } else { 3e38 };
        // borne de pas des poudres et des fronts de dendrites : recalculée au début de chaque pas (`k_burn_dt`) quand une poudre peut
        // brûler ou une eau se surfondre ; 0 = aucune
        d[D_DTBURN] = 0.0;
        self.queue.write_buffer(&self.d_buf, 0, bytemuck::cast_slice(&d));
        self.base = Baselines {
            boundary_flux: sim.boundary_flux.clone(),
            transferred: sim.transferred.clone(),
            energy_input: sim.energy_input,
            chem_heat: sim.chem_heat,
            radiated: sim.radiated,
            reacted: sim.reacted.clone(),
            frozen: sim.frozen.clone(),
        };
        self.host_stale = false;
    }

    /// Envoie seulement les cibles du pinceau « Chauffer » (sans toucher à l'état).
    pub fn sync_heat_targets(&mut self, sim: &MultiphaseSim) {
        if !sim.heat_active && !self.heat_on {
            return;
        }
        trace("heat targets");
        let n = self.nx * self.ny;
        let ht: Vec<f32> = (0..n).map(|c| if sim.heat_active && !sim.is_solid(c) && !sim.heat_target[c].is_nan() { sim.heat_target[c] as f32 } else { 0.0 }).collect();
        self.queue.write_buffer(&self.s_buf, (self.sl.o("heat_t") * 4) as u64, bytemuck::cast_slice(&ht));
        self.heat_on = sim.heat_active;
    }

    // ================================================================ relecture de l'état
    /// Relit l'état complet si des pas ont été faits depuis la dernière relecture.
    pub fn ensure_host(&mut self, sim: &mut MultiphaseSim) {
        if self.host_stale {
            self.download(sim);
        }
    }

    /// Relit l'état complet du GPU dans `sim` (masses, énergie, pression, vitesses, propriétés, flammes, bilans).
    pub fn download(&mut self, sim: &mut MultiphaseSim) {
        trace("download");
        let (nx, ny, np) = (self.nx, self.ny, self.np);
        let n = nx * ny;
        let (nfu, nfv) = ((nx + 1) * ny, nx * (ny + 1));
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("download") });
        enc.copy_buffer_to_buffer(&self.s_buf, (self.pers.0 * 4) as u64, &self.s_read, 0, (self.pers.1 * 4) as u64);
        enc.clear_buffer(&self.s_buf, (self.acc.0 * 4) as u64, Some((self.acc.1 * 4) as u64));
        self.queue.submit(Some(enc.finish()));
        let data = read_buffer(&self.device, &self.s_read);
        let base = self.pers.0;
        let o = |name: &str| self.sl.o(name) - base;
        let h0: Vec<f64> = sim.reg.phases.iter().map(|p| p.e_int(T_STD)).collect();
        let (o_m, o_e, o_p, o_u, o_v, o_t, o_a) = (o("mass"), o("eps"), o("prel"), o("U"), o("V"), o("temp"), o("alpha"));
        let (o_eg, o_tg) = (o("epsg"), o("tg"));
        let (o_rho, o_mu, o_k, o_cl, o_fl, o_hr) = (o("rho_mix"), o("mu_cell"), o("k_cell"), o("clamped"), o("flame"), o("heat_rel"));
        if sim.pressure_clamped.len() != n {
            sim.pressure_clamped = vec![false; n];
        }
        if sim.flame_factor.len() != n {
            sim.flame_factor = vec![1.0; n];
        }
        if sim.heat_release.len() != n {
            sim.heat_release = vec![0.0; n];
        }
        for c in 0..n {
            let mut ch = 0.0;
            for k in 0..np {
                let m = data[o_m + k * n + c] as f64;
                sim.mass[k][c] = m;
                sim.alpha[k][c] = data[o_a + k * n + c] as f64;
                ch += m * h0[k];
            }
            let eps = data[o_e + c] as f64;
            sim.e[c] = if sim.is_solid(c) { eps + sim.reg.solids[sim.solid[c] as usize].rho_cp() * T_STD } else { eps + ch };
            let mut esum = 0.0;
            for (g, members) in sim.groups.iter().enumerate() {
                let chg: f64 = members.iter().map(|&k| sim.mass[k][c] * h0[k]).sum();
                sim.eg[g][c] = if sim.is_solid(c) { 0.0 } else { data[o_eg + g * n + c] as f64 + chg };
                sim.tg[g][c] = data[o_tg + g * n + c] as f64;
                esum += sim.eg[g][c];
            }
            if !sim.is_solid(c) {
                sim.e[c] = esum; // référence : Σ E_g (l'agrégat ε relu en f32 n'en diffère que de l'arrondi)
            }
            sim.p[c] = data[o_p + c] as f64 + P0;
            sim.temp[c] = data[o_t + c] as f64;
            sim.rho_mix[c] = data[o_rho + c] as f64;
            sim.mu_cell[c] = data[o_mu + c] as f64;
            sim.k_cell[c] = data[o_k + c] as f64;
            sim.pressure_clamped[c] = data[o_cl + c] > 0.5;
            sim.flame_factor[c] = data[o_fl + c] as f64;
            sim.heat_release[c] = data[o_hr + c] as f64;
        }
        let o_fp = o("fu_phi");
        sim.fusion_front = (0..self.nfus).map(|q| (0..n).map(|c| data[o_fp + q * n + c] as f64).collect()).collect();
        let (o_qu, o_qv) = (o("qu"), o("qv"));
        if sim.q_u.len() != nfu || sim.q_v.len() != nfv {
            sim.q_u = vec![0.0; nfu];
            sim.q_v = vec![0.0; nfv];
        }
        for f in 0..nfu {
            sim.u[f] = data[o_u + f] as f64;
            sim.q_u[f] = data[o_qu + f] as f64;
        }
        for f in 0..nfv {
            sim.q_v[f] = data[o_qv + f] as f64;
            sim.v[f] = data[o_v + f] as f64;
        }
        let o_tpm = o("tpmix");
        for (slot, &d) in self.slots_of_dest.iter().enumerate() {
            let b = o_tpm + slot * (np + 2);
            if data[b] > 0.5 {
                sim.tp_mix.insert(d, ((0..np).map(|k| data[b + 2 + k] as f64).collect(), data[b + 1] as f64));
            }
        }
        // bilans : base + Σ accumulateurs par cellule (repartis de zéro sur le GPU)
        let sum = |name: &str, q: usize| -> f64 { let b = o(name) + q * n; data[b..b + n].iter().map(|&x| x as f64).sum() };
        for k in 0..np {
            self.base.boundary_flux[k] += sum("bflux", k);
        }
        for q in 0..self.npc {
            self.base.transferred[q] += sum("acc_tr", q);
        }
        self.base.energy_input += sum("acc_ein", 0);
        self.base.radiated += sum("acc_rad", 0);
        self.base.chem_heat += sum("acc_chem", 0);
        if self.base.reacted.len() != self.nre {
            self.base.reacted = vec![0.0; self.nre];
        }
        for j in 0..self.nre {
            self.base.reacted[j] += sum("acc_xi", j);
        }
        if self.base.frozen.len() != self.nfus {
            self.base.frozen = vec![0.0; self.nfus];
        }
        for q in 0..self.nfus {
            self.base.frozen[q] += sum("acc_fu", q);
        }
        sim.frozen.clone_from(&self.base.frozen);
        sim.boundary_flux.clone_from(&self.base.boundary_flux);
        sim.transferred.clone_from(&self.base.transferred);
        sim.energy_input = self.base.energy_input;
        sim.radiated = self.base.radiated;
        sim.chem_heat = self.base.chem_heat;
        sim.reacted.clone_from(&self.base.reacted);
        self.host_stale = false;
    }

    // ================================================================ paramètres
    fn params(&self, sim: &MultiphaseSim, dt_cap: f64) -> [u32; PARAMS_WORDS] {
        let mut w = [0u32; PARAMS_WORDS];
        let f = |x: f64| (x as f32).to_bits();
        let bc = |b: Bc| match b {
            Bc::Wall => (0u32, 0.0),
            Bc::Open { p } => (1u32, p - P0),
        };
        let fmax = sim
            .reg
            .reactions
            .iter()
            .filter(|r| r.flame_thickness > 0.0)
            .map(|r| sim.flame_resolution * sim.dx / r.flame_thickness)
            .fold(1.0f64, f64::max);
        let mut flags = 0;
        if sim.surface_tension {
            flags |= F_ST;
        }
        if sim.viscous {
            flags |= F_VISC;
        }
        if sim.conduction {
            flags |= F_COND;
        }
        if sim.phase_change && self.npc > 0 {
            flags |= F_PC;
        }
        if sim.chemistry && self.nre > 0 {
            flags |= F_CHEM;
        }
        if self.has_extras {
            flags |= F_EXTRAS;
        }
        if sim.conservative_momentum {
            flags |= F_CONS;
        }
        if sim.heat_active {
            flags |= F_HEAT;
        }
        if !self.diff_groups.is_empty() {
            flags |= F_MISC;
        }
        if self.flames_active(sim) {
            flags |= F_FLAME;
        }
        if sim.maccormack {
            flags |= F_MACC;
        }
        if sim.has_granular() {
            flags |= F_GRAN;
        }
        if sim.viscous && sim.implicit_viscosity {
            flags |= F_IMPVISC;
        }
        if sim.has_friction() {
            flags |= F_FRIC;
        }
        if sim.phase_change && self.nfus > 0 {
            flags |= F_FUS;
        }
        if percolation_possible(sim) {
            flags |= F_PERC;
        }
        w[0] = self.nx as u32;
        w[1] = self.ny as u32;
        w[2] = self.np as u32;
        w[3] = sim.reg.solids.len() as u32;
        w[4] = self.ng as u32;
        w[5] = self.npc as u32;
        w[6] = self.nre as u32;
        w[7] = self.levels.len() as u32;
        w[8] = flags;
        w[9] = sim.ambient_phase as u32;
        w[10] = self.ntp as u32;
        w[11] = (sim.step_count % 2) as u32;
        for s in 0..4 {
            w[12 + s] = bc(sim.bc[s]).0;
            w[24 + s] = f(bc(sim.bc[s]).1);
        }
        w[16] = f(sim.dx);
        w[17] = f(sim.gravity[0]);
        w[18] = f(sim.gravity[1]);
        w[19] = f(P0);
        w[20] = f(T_STD);
        w[21] = f(sim.t_ambient);
        w[22] = f(sim.cfl);
        w[23] = f(sim.dt_max);
        w[28] = f(sim.cap_safety);
        w[29] = f(sim.speed_limit);
        w[30] = f(sim.nucleation_superheat);
        w[31] = f(sim.nucleation_alpha);
        w[32] = f(sim.nuclei_density);
        w[33] = f(sim.boil_superheat_max);
        w[34] = f(sim.phase_relaxation_time);
        w[35] = f(sim.phase_volume_limit);
        w[36] = f(sim.heat_tau);
        w[37] = f(sim.heat_power_max);
        w[38] = f(sim.flame_resolution);
        w[39] = f(sim.chem_volume_limit);
        w[40] = f(sim.pressure_tol.max(1e-6));
        w[41] = f(dt_cap);
        w[42] = self.cond_budget as u32;
        w[43] = self.sp_budget as u32;
        w[44] = sim.reg.phases.iter().filter(|p| p.is_gas()).count() as u32;
        w[45] = (!sim.ambient_mix.is_empty()) as u32;
        w[46] = (sim.radiation && self.has_radiators) as u32;
        w[47] = self.mg_tail as u32;
        w[48] = f(fmax);
        w[49] = sim.flame_resolution.ceil() as u32 + 1;
        w[50] = 4000;
        w[55] = self.software as u32;
        w[56] = f(sim.closure_relax * self.relax_scale);
        w[57] = f(sim.cfl_reject);
        w[58] = f(self.relax_scale);
        w[59] = f(if sim.phase_heat_exchange.is_finite() { sim.phase_heat_exchange } else { 1e38 });
        w[60] = f(sim.cfl_flame);
        w[61] = f(sim.flame_speed_floor);
        w[62] = self.thermo_k(sim) as u32;
        w[63] = self.ntd as u32;
        w
    }

    /// sous-pas du bloc thermique (conduction, diffusion, chimie) du correcteur / d'un pas simple : `flame_thermo_substeps` si une flamme
    /// épaissie est possible (miroir de step_raw)
    fn thermo_k(&self, sim: &MultiphaseSim) -> usize {
        if self.flames_active(sim) { sim.flame_thermo_substeps.max(1) } else { 1 }
    }

    fn flames_active(&self, sim: &MultiphaseSim) -> bool {
        let fmax = sim
            .reg
            .reactions
            .iter()
            .filter(|r| r.flame_thickness > 0.0)
            .map(|r| sim.flame_resolution * sim.dx / r.flame_thickness)
            .fold(1.0f64, f64::max);
        sim.chemistry && self.has_flames && fmax > 1.0
    }

    // ================================================================ plan d'un pas
    fn vcycle(&self, ops: &mut Vec<Op>) {
        let t = self.mg_tail;
        // V-cycle : Q.a = 2·niveau + couleur
        let s = |l: usize, c: usize| self.slot(2 * l + c, 0);
        let half = |l: usize| {
            let (a, b) = self.levels[l];
            wg(a.div_ceil(2) * b)
        };
        let cells = |l: usize| {
            let (a, b) = self.levels[l];
            wg(a * b)
        };
        for l in 0..t {
            ops.push(Op { k: "k_mg_first", wg: cells(l), slot: s(l, 0) });
            for c in [1, 0, 1] {
                ops.push(Op { k: "k_mg_smooth", wg: half(l), slot: s(l, c) });
            }
            ops.push(Op { k: "k_mg_restrict", wg: cells(l + 1), slot: s(l, 0) });
        }
        ops.push(Op { k: "k_mg_tail", wg: 1, slot: 0 });
        for l in (0..t).rev() {
            ops.push(Op { k: "k_mg_post1", wg: half(l), slot: s(l, 1) });
            for c in [0, 1, 0] {
                ops.push(Op { k: "k_mg_smooth", wg: half(l), slot: s(l, c) });
            }
        }
    }

    fn plan_step(&self, sim: &MultiphaseSim) -> Vec<Op> {
        self.plan_step_at(sim, sim.step_count)
    }

    /// plan d'un pas dont le numéro est `step_index` (ordre des balayages x/y alterné comme sur le CPU) ; prédicteur-correcteur (miroir
    /// de `MultiphaseSim::step_pc`) : instantané, prédiction (un sous-pas thermique), u^{n+1} gardé, restauration, correcteur
    fn plan_step_at(&self, sim: &MultiphaseSim, step_index: usize) -> Vec<Op> {
        if !sim.use_predictor_corrector() {
            return self.plan_pass(sim, step_index, Pass::Single);
        }
        let n_pc = PC_GROUPS;
        let mut v = vec![Op { k: "k_pc_save", wg: n_pc, slot: 0 }];
        v.extend(self.plan_pass(sim, step_index, Pass::Pred));
        let wf = wg(((self.nx + 1) * self.ny).max(self.nx * (self.ny + 1)));
        v.push(Op { k: "k_pc_keep", wg: wf, slot: 0 });
        v.push(Op { k: "k_pc_restore", wg: n_pc, slot: 0 });
        v.extend(self.plan_pass(sim, step_index, Pass::Corr));
        v.push(Op { k: "k_pc_merge", wg: 1, slot: 0 });
        v
    }

    fn plan_pass(&self, sim: &MultiphaseSim, step_index: usize, pass: Pass) -> Vec<Op> {
        let (nx, ny) = (self.nx, self.ny);
        let n = nx * ny;
        let (nfu, nfv) = ((nx + 1) * ny, nx * (ny + 1));
        let nf = nfu.max(nfv);
        let (wn, wf, wm) = (wg(n), wg(nf), wg(n.max(nf)));
        let one = |k: &'static str| Op { k, wg: 1, slot: 0 };
        let cell = |k: &'static str| Op { k, wg: wn, slot: 0 };
        let face = |k: &'static str| Op { k, wg: wf, slot: 0 };
        let mut v: Vec<Op> = Vec::with_capacity(2048);
        let extras = self.has_extras;
        v.push(cell("k_refresh"));
        // fluidisation (miroir de prepare_special / hold_faces) : traînée de percolation, état des colonnes, faces des lits retenus
        let fluid_bed = percolation_possible(sim) && sim.has_granular();
        if fluid_bed {
            let vert = sim.gravity[1].abs() >= sim.gravity[0].abs();
            v.push(cell("k_fl_drag"));
            v.push(Op { k: "k_fl_state", wg: wg(if vert { nx } else { ny }), slot: 0 });
            v.push(Op { k: "k_fl_hold", wg: wm, slot: 0 });
        }
        if extras {
            v.push(face("k_apply_pf"));
            v.push(cell("k_pin_reset"));
            v.push(cell("k_refresh"));
        }
        // combustion des poudres, fronts de dendrites (surfusion) : borne du pas sur l'état du début du pas (avant k_red_dt, qui
        // réutilise les sommes partielles)
        if (sim.chemistry && sim.reg.reactions.iter().any(|r| r.burn().is_some())) || fronts_possible(sim) {
            v.push(cell("k_burn_dt"));
            v.push(one("k_burn_dt_final"));
        }
        v.push(Op { k: "k_red_dt", wg: wm, slot: 0 });
        v.push(one("k_dt_final"));
        v.push(cell("k_err0"));
        v.push(face("k_copy_un"));
        // forces sur l'état n
        v.push(face("k_force_init"));
        if fluid_bed {
            v.push(face("k_fl_force"));
        }
        if self.has_accel {
            v.push(face("k_accel"));
        }
        if sim.surface_tension {
            v.push(cell("k_marker"));
            v.push(cell("k_smooth"));
            v.push(cell("k_curv"));
            v.push(face("k_force_st"));
        }
        if sim.viscous {
            v.push(cell("k_visc_diag"));
            v.push(Op { k: "k_visc_xy", wg: wg((nx + 1) * (ny + 1)), slot: 0 });
            v.push(face("k_visc_apply"));
        }
        if !sim.conservative_momentum {
            v.push(face("k_sl_pass1"));
            if sim.maccormack {
                v.push(face("k_sl_pass2"));
            }
        }
        // masses / énergie : balayages PLIC alternés
        if pass == Pass::Corr {
            v.push(face("k_pc_adv"));
        }
        v.push(Op { k: "k_adv_init", wg: wm, slot: 0 });
        let first = step_index % 2;
        for q in 0..2 {
            let dir = (first + q) % 2;
            let wd = wg(if dir == 0 { nfu } else { nfv });
            v.push(cell("k_an"));
            v.push(Op { k: "k_sweep_flux", wg: wd, slot: self.slot(dir, 0) });
            v.push(Op { k: "k_sweep_scale", wg: wn, slot: self.slot(dir, 0) });
            // mise à jour séquentielle par ligne (x) / colonne (y), dans l'ordre des faces du CPU
            v.push(Op { k: "k_sweep_seq", wg: wg(if dir == 0 { ny } else { nx }), slot: self.slot(dir, 0) });
            v.push(cell("k_temp_update"));
        }
        if pass == Pass::Corr {
            v.push(face("k_pc_back"));
        }
        if sim.conservative_momentum {
            v.push(face("k_mom"));
        }
        // percolation : transport du fluide par les flux de Darcy de la dernière projection (miroir de percolate)
        let perc = percolation_possible(sim);
        if perc {
            v.push(cell("k_perc_out"));
            v.push(cell("k_perc_delta"));
            v.push(cell("k_settle_apply"));
        }
        if extras {
            if self.ntp > 0 {
                v.push(one("k_tp_absorb"));
            }
            v.push(cell("k_pin_reset"));
        }
        // thermique, espèces, chimie, changement de phase sur l'état advecté
        v.push(cell("k_refresh"));
        // grains : sédimentation sur l'état advecté, une phase granulaire après l'autre (miroir de settle_grains), puis refresh
        if sim.has_granular() {
            for s in (0..self.np).filter(|&k| sim.reg.phases[k].is_granular()) {
                v.push(Op { k: "k_settle_delta", wg: wn, slot: self.slot(s, 0) });
                v.push(Op { k: "k_settle_apply", wg: wn, slot: self.slot(s, 0) });
            }
            if sim.chemistry && sim.reg.reactions.iter().any(|r| r.bulk) {
                v.push(cell("k_pyrolysis"));
            }
            v.push(cell("k_refresh"));
        }
        if sim.heat_active || self.has_thermal {
            v.push(cell("k_thermal_sources"));
        }
        if self.flames_active(sim) {
            v.push(cell("k_flame_omega"));
            v.push(one("k_flame_wmax"));
            v.push(cell("k_flame_mask"));
            v.push(cell("k_flame_dil_x"));
            v.push(cell("k_flame_dil_y"));
        } else if self.nre > 0 {
            v.push(cell("k_flame_one"));
        }
        // flammes : conduction, diffusion et chimie intercalées en sous-pas (chimie des seules cellules de flamme), puis chimie hors
        // flamme sur dt (miroir de step_raw) ; la prédiction se contente d'un sous-pas
        let kth = if pass == Pass::Pred { 1 } else { self.thermo_k(sim) };
        let chem = sim.chemistry && self.nre > 0;
        if kth > 1 {
            v.push(one("k_dtf_sub"));
        }
        for _ in 0..kth {
            if sim.conduction {
                v.push(cell("k_cond_prep"));
                v.push(one("k_cond_nsub"));
                for s in 0..self.cond_budget {
                    v.push(Op { k: "k_cond_step", wg: wn, slot: self.slot(s, 0) });
                }
                v.push(cell("k_cond_finish"));
            }
            for &g in &self.diff_groups {
                v.push(Op { k: "k_sp_dloc", wg: wn, slot: self.slot(0, g) });
                v.push(one("k_sp_nsub"));
                v.push(Op { k: "k_sp_prep", wg: wn, slot: self.slot(0, g) });
                for s in 0..self.sp_budget {
                    v.push(Op { k: "k_sp_step", wg: wn, slot: self.slot(s, g) });
                    v.push(Op { k: "k_sp_commit", wg: wn, slot: self.slot(s, g) });
                }
            }
            if chem {
                v.push(Op { k: "k_chem", wg: wn, slot: self.slot(if kth > 1 { 1 } else { 0 }, 0) });
                v.push(one("k_chem_final"));
            }
        }
        if kth > 1 {
            v.push(one("k_dtf_one"));
            if chem {
                v.push(Op { k: "k_chem", wg: wn, slot: self.slot(2, 0) });
                v.push(one("k_chem_final"));
            }
        }
        if !chem && self.nre > 0 {
            v.push(cell("k_heat_zero"));
        }
        if sim.radiation && self.has_radiators {
            v.push(cell("k_radiate"));
        }
        if sim.phase_change {
            for q in 0..self.npc {
                v.push(Op { k: "k_phase_change", wg: wn, slot: self.slot(q, 0) });
                v.push(one("k_phase_final"));
                if self.surf_pairs[q] {
                    v.push(Op { k: "k_surf_req", wg: wn, slot: self.slot(q, 0) });
                    v.push(Op { k: "k_surf_apply", wg: wn, slot: self.slot(q, 0) });
                }
            }
        }
        // échange entre groupes d'une cellule mixte, en dernier avant la projection (comme le CPU)
        if self.ng > 1 {
            v.push(cell("k_exchange"));
            if sim.has_granular() {
                v.push(cell("k_exch_grains"));
            }
        }
        // gel / fusion (eau ↔ glace) sur les températures de groupe après les échanges (miroir de fusion_step) ; avant, le front de
        // dendrites de l'eau surfondue (advance_fusion_front)
        if sim.phase_change && self.nfus > 0 {
            if fronts_possible(sim) {
                v.push(cell("k_fusion_germ"));
                v.push(cell("k_fusion_front"));
            }
            v.push(cell("k_fusion"));
        }
        // pression : Newton + MG-PCG
        v.push(cell("k_refresh"));
        // viscosité implicite (frottement des grains tassés, fluides très visqueux) sur u*, avant la projection (miroir de
        // apply_implicit_viscosity)
        if implicit_viscosity_needed(sim, self.nx) {
            let (vert, wuv) = (sim.gravity[1].abs() >= sim.gravity[0].abs(), wg(nfu + nfv));
            if sim.has_friction() {
                v.push(Op { k: "k_fr_ps", wg: wg(if vert { nx } else { ny }), slot: 0 });
            }
            v.push(cell("k_fr_mu"));
            v.push(Op { k: "k_fr_uf", wg: wuv, slot: 0 });
            v.push(Op { k: "k_fr_r0", wg: wuv, slot: 0 });
            v.push(one("k_fr_start"));
            for it in 0..self.fr_budget {
                let sl = self.slot(it % 2, 0);
                v.push(Op { k: "k_fr_ap", wg: wuv, slot: sl });
                v.push(Op { k: "k_fr_upd", wg: wuv, slot: sl });
                v.push(Op { k: "k_fr_dir", wg: wuv, slot: sl });
            }
            if sim.has_friction() {
                v.push(Op { k: "k_fr_w", wg: wuv, slot: 0 });
                v.push(cell("k_fr_heat"));
            }
            v.push(Op { k: "k_fr_apply", wg: wuv, slot: 0 });
        }
        // percolation : milieu poreux, pression de contact, conductances de Darcy (Ergun sur la pression du pas précédent)
        if perc {
            let vert = sim.gravity[1].abs() >= sim.gravity[0].abs();
            v.push(cell("k_dc_cell"));
            v.push(Op { k: "k_dc_ps", wg: wg(if vert { nx } else { ny }), slot: 0 });
            v.push(face("k_dc_faces"));
        }
        v.push(cell("k_save_p"));
        let last = self.levels.len() - 1;
        for _ in 0..self.newton_budget {
            v.push(cell("k_closure"));
            v.push(cell("k_stencil"));
            for l in 0..last {
                let (a, b) = self.levels[l + 1];
                v.push(Op { k: "k_mg_coarsen_links", wg: wg(a * b), slot: self.slot(l, 0) });
                v.push(Op { k: "k_mg_coarsen_diag", wg: wg(a * b), slot: self.slot(l + 1, 0) });
            }
            v.push(face("k_face_tmp"));
            v.push(cell("k_rhs"));
            v.push(cell("k_pcg_init"));
            v.push(one("k_pcg_bb"));
            self.vcycle(&mut v);
            v.push(cell("k_pcg_rz"));
            v.push(cell("k_pcg_p0"));
            for it in 0..self.pcg_budget {
                let sl = self.slot(it % 2, 0);
                v.push(cell("k_pcg_apply"));
                v.push(Op { k: "k_pcg_update", wg: wn, slot: sl });
                v.push(one("k_pcg_rr"));
                self.vcycle(&mut v);
                v.push(cell("k_pcg_rz"));
                v.push(Op { k: "k_pcg_dir", wg: wn, slot: sl });
            }
            v.push(cell("k_p_update"));
            v.push(one("k_newton_check"));
        }
        v.push(face("k_face_final"));
        if perc {
            v.push(face("k_dc_final"));
        }
        v.push(cell("k_pressure_work"));
        v.push(cell("k_fill_cav"));
        v.push(cell("k_refresh"));
        v.push(Op { k: "k_sanity", wg: wm, slot: 0 });
        v.push(one("k_sanity_final"));
        v
    }

    fn encode(&mut self, enc: &mut wgpu::CommandEncoder, ops: &[Op]) {
        for op in ops {
            self.pipeline(op.k);
        }
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("multiphase"), timestamp_writes: None });
        pass.set_bind_group(0, &self.bg0, &[]);
        let mut cur: &str = "";
        let mut cur_slot = u32::MAX;
        for op in ops {
            if op.k != cur {
                pass.set_pipeline(&self.pipelines[op.k]);
                cur = op.k;
            }
            if op.slot != cur_slot {
                pass.set_bind_group(1, &self.bg1, &[op.slot]);
                cur_slot = op.slot;
            }
            pass.dispatch_workgroups(op.wg, 1, 1);
        }
    }

    // ================================================================ pas de temps protégé
    /// Un pas complet sur le GPU, protégé comme `MultiphaseSim::step` (retour arrière + dt/4, jusqu'à 5 fois). Met à jour le temps,
    /// les compteurs et les diagnostics scalaires de `sim` ; les champs de `sim` ne sont relus que par `ensure_host` / `download`.
    pub fn step(&mut self, sim: &mut MultiphaseSim) -> f64 {
        trace("step");
        if sim.diverged {
            return 0.0;
        }
        let clk = Instant::now();
        let mut dt_cap = sim.dt_max;
        let mut dt_try = f64::NAN;
        let (mut tries, mut redos) = (0usize, 0usize);
        let mut restore = false;
        loop {
            self.relax_scale = 0.25f64.powi(tries as i32);
            let p = self.params(sim, dt_cap);
            self.relax_scale = 1.0;
            self.queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&p));
            let ops = self.plan_step(sim);
            let n_ops = ops.len();
            let (ps, pl) = ((self.pers.0 * 4) as u64, (self.pers.1 * 4) as u64);
            // un pas très long (budgets élevés) est soumis en plusieurs tampons de commandes de ≤ MAX_BATCH_OPS lancements
            // (diagnostic MP_GPU_SYNC : un lancement par tampon, attente après chacun, nom affiché avant)
            let chunks: Vec<&[Op]> = ops.chunks(if sync_debug() { 1 } else { max_batch_ops() }).collect();
            let last_chunk = chunks.len() - 1;
            for (ci, ch) in chunks.iter().enumerate() {
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("multiphase step") });
                if ci == 0 {
                    if restore {
                        enc.copy_buffer_to_buffer(&self.snap_s, 0, &self.s_buf, ps, pl);
                        enc.copy_buffer_to_buffer(&self.snap_d, 0, &self.d_buf, 0, (D_LEN * 4) as u64);
                    } else {
                        enc.copy_buffer_to_buffer(&self.s_buf, ps, &self.snap_s, 0, pl);
                        enc.copy_buffer_to_buffer(&self.d_buf, 0, &self.snap_d, 0, (D_LEN * 4) as u64);
                    }
                }
                self.encode(&mut enc, ch);
                if ci == last_chunk {
                    enc.copy_buffer_to_buffer(&self.d_buf, 0, &self.d_read, 0, (D_LEN * 4) as u64);
                }
                if sync_debug() {
                    eprintln!("[gpu sync] {}", ch[0].k);
                }
                self.queue.submit(Some(enc.finish()));
                if sync_debug() {
                    self.device.poll(wgpu::Maintain::Wait);
                }
            }
            let d = read_buffer(&self.device, &self.d_read);
            self.host_stale = true;
            let dt = d[D_DT] as f64;
            if dt_try.is_nan() {
                dt_try = dt;
            }
            // budgets de sous-pas insuffisants : on refait le même pas avec un budget suffisant (pas un retour arrière physique)
            if d[D_REDO] > 0.5 && redos < 3 {
                redos += 1;
                self.cond_budget = self.cond_budget.max((d[D_SUB_C] as f64 * 1.25).ceil() as usize + 2).min(2000);
                self.sp_budget = self.sp_budget.max((d[D_NSUB_S_MAX] as f64 * 1.25).ceil() as usize + 2).min(500);
                restore = true;
                continue;
            }
            if d[D_BAD] > 0.5 || !dt.is_finite() {
                if tries < 5 {
                    tries += 1;
                    sim.rollbacks += 1;
                    dt_try *= 0.25;
                    dt_cap = sim.dt_max.min(dt_try);
                    restore = true;
                    continue;
                }
                // rien ne passe : état d'avant le pas restauré, simulation arrêtée
                let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("restore") });
                enc.copy_buffer_to_buffer(&self.snap_s, 0, &self.s_buf, ps, pl);
                enc.copy_buffer_to_buffer(&self.snap_d, 0, &self.d_buf, 0, (D_LEN * 4) as u64);
                self.queue.submit(Some(enc.finish()));
                sim.diverged = true;
                sim.profile.secs[P_GPU] += clk.elapsed().as_secs_f64();
                return 0.0;
            }
            // pas accepté
            let newton = d[D_NEWTON_IT] as usize;
            let pcg_max = d[D_MAXPCG_N] as usize;
            sim.time += dt;
            sim.last_dt = dt;
            sim.step_count += 1;
            sim.last_pressure_iters = d[D_PCG_IT] as usize;
            sim.last_newton_iters = newton;
            sim.dt_chem = if d[D_DTCHEM] < 1e37 { d[D_DTCHEM] as f64 } else { f64::INFINITY };
            sim.dt_phase = if d[D_DTPHASE] < 1e37 { d[D_DTPHASE] as f64 } else { f64::INFINITY };
            let (need_c, need_s) = (d[D_SUB_C] as usize, d[D_NSUB_S_MAX] as usize);
            self.update_budgets(sim, newton, pcg_max, d[D_PCG_UNCONV] > 0.5, need_c, need_s);
            if implicit_viscosity_needed(sim, self.nx) {
                // viscosité implicite : budget = itérations du pas × 1.3 (doublé si le gradient conjugué s'est arrêté sur le budget)
                let it = d[D_FR_IT] as usize;
                sim.last_friction_iters = it;
                let mut fb = (it as f64 * 1.3).ceil() as usize + 4;
                if d[D_FR_UNCONV] > 0.5 {
                    fb = fb.max(2 * self.fr_budget);
                }
                self.fr_budget = fb.clamp(8, sim.friction_max_iter.max(8));
            }
            self.last = GpuStepInfo { dt, newton, pcg: d[D_PCG_IT] as usize, pcg_max, cond_sub: need_c, species_sub: need_s, retries: tries, redo: redos, dispatches: n_ops };
            sim.profile.secs[P_GPU] += clk.elapsed().as_secs_f64();
            sim.profile.steps += 1;
            return if tries > 0 { sim.last_dt } else { dt };
        }
    }

    /// Diagnostic : exécute le plan d'un pas jusqu'à la `occ`-ième occurrence (1 = première) du noyau `stop` incluse, sans
    /// instantané ni contrôle ; l'état GPU est alors celui du milieu du pas (à relire par `download`).
    #[cfg(test)]
    pub(crate) fn debug_partial_step(&mut self, sim: &MultiphaseSim, stop: &str, occ: usize) {
        let p = self.params(sim, sim.dt_max);
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&p));
        let ops = self.plan_step(sim);
        let mut seen = 0;
        let mut end = ops.len();
        for (i, op) in ops.iter().enumerate() {
            if op.k == stop {
                seen += 1;
                if seen == occ {
                    end = i + 1;
                    break;
                }
            }
        }
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("partial") });
        self.encode(&mut enc, &ops[..end]);
        self.queue.submit(Some(enc.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        self.host_stale = true;
    }

    /// Diagnostic : relit la tranche `name` du tampon d'état S (tous les champs, persistants ou de travail)
    #[cfg(test)]
    pub(crate) fn debug_read_slice(&self, name: &str) -> Vec<f32> {
        let (off, len) = self.sl.map[name];
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug slice"),
            size: ((len * 4) as u64).max(16),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("debug slice") });
        enc.copy_buffer_to_buffer(&self.s_buf, (off * 4) as u64, &buf, 0, (len * 4) as u64);
        self.queue.submit(Some(enc.finish()));
        read_buffer(&self.device, &buf)
    }

    /// Diagnostic : un pas BRUT (sans retour arrière) avec dt ≤ `dt_cap` et la fermeture héritée à `relax_scale` ; l'état obtenu est
    /// relu dans `sim`, puis l'état d'avant le pas est rétabli sur le GPU (`download` le relit). Retourne (pas refusé ?, dt).
    #[doc(hidden)]
    #[allow(dead_code)]
    pub fn debug_raw_step(&mut self, sim: &mut MultiphaseSim, dt_cap: f64, relax_scale: f64) -> (bool, f64) {
        self.relax_scale = relax_scale;
        let p = self.params(sim, dt_cap);
        self.relax_scale = 1.0;
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&p));
        let ops = self.plan_step(sim);
        let (ps, pl) = ((self.pers.0 * 4) as u64, (self.pers.1 * 4) as u64);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("raw snapshot") });
        enc.copy_buffer_to_buffer(&self.s_buf, ps, &self.snap_s, 0, pl);
        enc.copy_buffer_to_buffer(&self.d_buf, 0, &self.snap_d, 0, (D_LEN * 4) as u64);
        self.queue.submit(Some(enc.finish()));
        for ch in ops.chunks(MAX_BATCH_OPS) {
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("raw step") });
            self.encode(&mut enc, ch);
            self.queue.submit(Some(enc.finish()));
        }
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("raw read") });
        enc.copy_buffer_to_buffer(&self.d_buf, 0, &self.d_read, 0, (D_LEN * 4) as u64);
        self.queue.submit(Some(enc.finish()));
        let d = read_buffer(&self.device, &self.d_read);
        self.download(sim);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("raw restore") });
        enc.copy_buffer_to_buffer(&self.snap_s, 0, &self.s_buf, ps, pl);
        enc.copy_buffer_to_buffer(&self.snap_d, 0, &self.d_buf, 0, (D_LEN * 4) as u64);
        self.queue.submit(Some(enc.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        self.host_stale = true;
        (d[D_BAD] > 0.5, d[D_DT] as f64)
    }

    /// Diagnostic : exécute le plan d'un pas noyau par noyau en attendant la fin de chacun (repère un noyau fautif).
    #[cfg(test)]
    pub(crate) fn debug_step_one_by_one(&mut self, sim: &MultiphaseSim) {
        let p = self.params(sim, sim.dt_max);
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&p));
        let ops = self.plan_step(sim);
        for (i, op) in ops.iter().enumerate() {
            eprintln!("  [{i}] {}", op.k);
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("one") });
            self.encode(&mut enc, &ops[i..i + 1]);
            self.queue.submit(Some(enc.finish()));
            self.device.poll(wgpu::Maintain::Wait);
        }
        self.host_stale = true;
    }

    /// `k` pas soumis d'un bloc (une seule relecture de contrôle pour la série). Instantané au début de la série ; si un pas est
    /// invalide ou manque de budget, la série est annulée puis refaite pas à pas avec `step` (retours arrière compris). Retourne le
    /// temps simulé avancé (0 si la simulation a divergé).
    pub fn step_batch(&mut self, sim: &mut MultiphaseSim, k: usize) -> f64 {
        trace(&format!("step_batch {k}"));
        if k <= 1 || sim.diverged {
            return self.step(sim);
        }
        let clk = Instant::now();
        let p = self.params(sim, sim.dt_max);
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&p));
        let mut ops: Vec<Op> = vec![];
        for i in 0..k {
            ops.extend(self.plan_step_at(sim, sim.step_count + i));
            ops.push(Op { k: "k_batch_acc", wg: 1, slot: 0 });
        }
        let n_ops = ops.len();
        let (ps, pl) = ((self.pers.0 * 4) as u64, (self.pers.1 * 4) as u64);
        // une seule relecture de contrôle pour toute la série, mais au plus MAX_BATCH_OPS lancements par tampon de commandes
        // (lavapipe plante sur des tampons de plusieurs milliers de lancements ; aucun coût notable ailleurs)
        let chunks: Vec<&[Op]> = ops.chunks(if sync_debug() { 1 } else { max_batch_ops() }).collect();
        let last_chunk = chunks.len() - 1;
        for (ci, ch) in chunks.iter().enumerate() {
            if sync_debug() {
                eprintln!("[gpu sync] {}", ch[0].k);
            }
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("multiphase batch") });
            if ci == 0 {
                enc.copy_buffer_to_buffer(&self.s_buf, ps, &self.snap_s, 0, pl);
                enc.copy_buffer_to_buffer(&self.d_buf, 0, &self.snap_d, 0, (D_LEN * 4) as u64);
                enc.clear_buffer(&self.d_buf, (D_B_TSUM * 4) as u64, Some(((D_LEN - D_B_TSUM) * 4) as u64));
            }
            self.encode(&mut enc, ch);
            if ci == last_chunk {
                enc.copy_buffer_to_buffer(&self.d_buf, 0, &self.d_read, 0, (D_LEN * 4) as u64);
            }
            self.queue.submit(Some(enc.finish()));
            if sync_debug() {
                self.device.poll(wgpu::Maintain::Wait);
            }
        }
        let d = read_buffer(&self.device, &self.d_read);
        self.host_stale = true;
        let tsum = d[D_B_TSUM] as f64;
        if d[D_B_BAD] > 0.5 || d[D_B_REDO] > 0.5 || !tsum.is_finite() {
            // série refusée : retour à l'instantané, puis pas à pas (budgets relevés / retours arrière par `step`)
            if d[D_B_REDO] > 0.5 {
                self.cond_budget = self.cond_budget.max((d[D_B_SUBC] as f64 * 1.25).ceil() as usize + 2).min(2000);
                self.sp_budget = self.sp_budget.max((d[D_B_NSUBS] as f64 * 1.25).ceil() as usize + 2).min(500);
            }
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("batch restore") });
            enc.copy_buffer_to_buffer(&self.snap_s, 0, &self.s_buf, ps, pl);
            enc.copy_buffer_to_buffer(&self.snap_d, 0, &self.d_buf, 0, (D_LEN * 4) as u64);
            self.queue.submit(Some(enc.finish()));
            sim.profile.secs[P_GPU] += clk.elapsed().as_secs_f64();
            let t0 = sim.time;
            for _ in 0..k {
                if self.step(sim) == 0.0 {
                    break;
                }
            }
            return sim.time - t0;
        }
        let newton = d[D_B_NEWTON] as usize;
        let pcg_max = d[D_B_MAXPCG] as usize;
        sim.time += tsum;
        sim.last_dt = d[D_DT] as f64;
        sim.step_count += k;
        sim.last_pressure_iters = d[D_PCG_IT] as usize;
        sim.last_newton_iters = d[D_NEWTON_IT] as usize;
        sim.dt_chem = if d[D_DTCHEM] < 1e37 { d[D_DTCHEM] as f64 } else { f64::INFINITY };
        sim.dt_phase = if d[D_DTPHASE] < 1e37 { d[D_DTPHASE] as f64 } else { f64::INFINITY };
        self.update_budgets(sim, newton, pcg_max, d[D_B_UNCONV] > 0.5, d[D_B_SUBC] as usize, d[D_B_NSUBS] as usize);
        self.last = GpuStepInfo {
            dt: sim.last_dt,
            newton,
            pcg: sim.last_pressure_iters,
            pcg_max,
            cond_sub: d[D_B_SUBC] as usize,
            species_sub: d[D_B_NSUBS] as usize,
            retries: 0,
            redo: 0,
            dispatches: n_ops / k,
        };
        sim.profile.secs[P_GPU] += clk.elapsed().as_secs_f64();
        sim.profile.steps += k;
        tsum
    }

    /// budgets du pas suivant (marge : besoin constaté + 30 %)
    fn update_budgets(&mut self, sim: &MultiphaseSim, newton: usize, pcg_max: usize, unconverged: bool, need_c: usize, need_s: usize) {
        self.newton_budget = (newton + 1).clamp(2, 4);
        let mut pb = (pcg_max as f64 * 1.3).ceil() as usize + 2;
        if unconverged {
            pb = pb.max(2 * self.pcg_budget);
        }
        self.pcg_budget = pb.clamp(6, sim.pressure_max_iter.max(6));
        self.cond_budget = ((need_c as f64 * 1.3).ceil() as usize + 1).clamp(2, 2000);
        self.sp_budget = ((need_s as f64 * 1.3).ceil() as usize + 1).clamp(2, 500);
    }

    // ================================================================ rendu
    /// Rend l'image (mêmes couleurs que `mp_view::render_rgba`) sur le GPU et la relit dans `out` (nx·ny·4 octets, ligne 0 = haut).
    pub fn render(&mut self, sim: &MultiphaseSim, o: &super::mp_view::RenderOpts, out: &mut Vec<u8>) {
        trace(&format!("render {:?}", o.display));
        use super::mp_view::MpDisplay as M;
        let n = self.nx * self.ny;
        let mut p = self.params(sim, sim.dt_max);
        p[51] = match o.display {
            M::Phases => 0,
            M::Pressure => 1,
            M::Temperature => 2,
            M::Velocity => 3,
            M::Density => 4,
            M::Vapor => 5,
            M::Divergence => 6,
            M::Flame => 7,
            M::HeatRelease => 8,
        };
        p[52] = (o.show_effects as u32) | ((o.solids_white as u32) << 1);
        p[53] = (o.t_scale as f32).to_bits();
        p[54] = (o.p_scale as f32).to_bits();
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&p));
        let ops = [Op { k: "k_rd_red", wg: wg(n), slot: 0 }, Op { k: "k_rd_final", wg: 1, slot: 0 }, Op { k: "k_render", wg: wg(n), slot: 0 }];
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("render") });
        self.encode(&mut enc, &ops);
        enc.copy_buffer_to_buffer(&self.r_buf, 0, &self.r_read, 0, (n * 4) as u64);
        self.queue.submit(Some(enc.finish()));
        let slice = self.r_read.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        out.resize(n * 4, 255);
        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            out.copy_from_slice(&data[..n * 4]);
            drop(data);
        }
        self.r_read.unmap();
        trace("render terminé");
    }
}

impl Drop for MultiphaseGpu {
    /// attend la fin du travail soumis avant de détruire le périphérique (certains pilotes, dont lavapipe, plantent sinon)
    fn drop(&mut self) {
        trace("drop");
        self.device.poll(wgpu::Maintain::Wait);
        trace("drop: GPU inactif");
    }
}

const P0: f64 = 101325.0;

/// diagnostic : MP_GPU_TRACE=1 affiche chaque appel de haut niveau (envoi, pas, relecture, rendu)
fn trace(what: &str) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var("MP_GPU_TRACE").is_ok()) {
        eprintln!("[gpu] {what}");
    }
}

/// diagnostic : MP_GPU_SYNC=1 soumet chaque lancement de `step` seul et attend sa fin (localise un plantage de pilote)
fn sync_debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MP_GPU_SYNC").is_ok())
}

/// Relit un buffer MAP_READ entier en f32 (attente bloquante).
fn read_buffer(device: &wgpu::Device, buf: &wgpu::Buffer) -> Vec<f32> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    let v = match rx.recv() {
        Ok(Ok(())) => {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, f32>(&data).to_vec()
        }
        _ => vec![f32::NAN; (buf.size() / 4) as usize],
    };
    buf.unmap();
    v
}

/// taille max d'un tampon de commandes (contournement lavapipe) ; diagnostic : MP_GPU_BATCH_OPS=n pour mesurer son coût sur une vraie carte
fn max_batch_ops() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("MP_GPU_BATCH_OPS").ok().and_then(|v| v.parse().ok()).filter(|&n| n > 0).unwrap_or(MAX_BATCH_OPS))
}
