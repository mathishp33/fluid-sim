//! Sauvegarde de l'ÉTAT COMPLET d'une simulation multiphasique (expérience reproductible) : tous les champs de `MultiphaseSim`
//! (masses, énergies et températures par groupe, pression, vitesses, solides, comportements thermiques, cellules spéciales, bords,
//! entrées de bord, chauffage au pinceau, réglages numériques et physiques, compteurs, bilans), le registre des matériaux tel qu'il
//! est utilisé (indices compris), plus le contexte de la fenêtre (bibliothèque de matériaux, scénario, bords de l'interface).
//!
//! Format : en-tête `MAGIC` (8 octets), version (u32 petit-boutiste), puis `StateFile` en bincode — binaire, compact et EXACT au bit
//! près : une simulation rechargée continue exactement comme l'originale. bincode n'est pas auto-descriptif : toute modification des
//! champs de `MultiphaseSim` (ou des types qu'il contient) change le format → incrémenter `VERSION` (un fichier d'une autre version
//! est refusé avec un message, pas mal relu).
//!
//! Les scènes `.mps` antérieures (JSON, géométrie seule : `mp_view::scene_to_json`) restent lisibles : `is_state_file` les distingue.

use serde::{Deserialize, Serialize};

use super::fluid::Boundary;
use super::mp_materials::MaterialLibrary;
use super::multiphase::MultiphaseSim;

/// en-tête d'un fichier d'état complet
pub const MAGIC: &[u8; 8] = b"FSIMMPST";
/// version du format (voir l'en-tête du module)
pub const VERSION: u32 = 14; // 2 : rayon de la destination des téléporteurs ; 3 : phases granulaires ; 4 : frottement des grains ; 5 : viscosité implicite ; 6 : percolation, grilles ; 7 : réactions en volume (pyrolyse) ; 8 : combustion des poudres (loi de Vieille) ; 9 : gel / fusion (eau ↔ glace) ; 10 : surfusion (front de dendrites) ; 11 : lits perméables à deux pressions (pression de contact des grains) ; 12 : corps rigides (solides mobiles, étape 0) ; 13 : mouvement imposé des corps (vitesse du corps aux faces) ; 14 : corps libres (masse volumique propre d'un corps)

/// Contenu d'un fichier d'état complet.
#[derive(Serialize, Deserialize)]
pub struct StateFile {
    /// bibliothèque de matériaux de la fenêtre (celle des réglages, d'où le registre a été construit) : rétablie au chargement pour
    /// que « Recommencer » et l'éditeur de matériaux repartent des mêmes matériaux
    pub library: Option<MaterialLibrary>,
    /// scénario de la fenêtre (indice de `MpScenario::ALL`)
    pub scenario: Option<usize>,
    /// bords tels que réglés dans la fenêtre (le solveur n'en garde que la traduction : `bc`, `inlet`)
    pub boundaries: Option<[Boundary; 4]>,
    /// l'état complet du solveur, registre compris
    pub sim: MultiphaseSim,
}

/// Fichier d'état complet (et non scène JSON de la géométrie seule) ?
pub fn is_state_file(bytes: &[u8]) -> bool {
    bytes.len() >= MAGIC.len() && &bytes[..MAGIC.len()] == MAGIC
}

/// Sérialise l'état complet. Sur le GPU, relire d'abord l'état (`MultiphaseGpu::ensure_host`) : seuls les champs de `sim` sont écrits.
pub fn state_to_bytes(
    sim: &MultiphaseSim,
    library: Option<&MaterialLibrary>,
    scenario: Option<usize>,
    boundaries: Option<[Boundary; 4]>,
) -> Result<Vec<u8>, String> {
    #[derive(Serialize)]
    struct Ref<'a> {
        library: Option<&'a MaterialLibrary>,
        scenario: Option<usize>,
        boundaries: Option<[Boundary; 4]>,
        sim: &'a MultiphaseSim,
    }
    let body = bincode::serialize(&Ref { library, scenario, boundaries, sim }).map_err(|e| format!("sérialisation : {e}"))?;
    let mut out = Vec::with_capacity(MAGIC.len() + 4 + body.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Relit un état complet. Erreur claire si ce n'en est pas un, si la version diffère ou si le contenu est incohérent.
pub fn state_from_bytes(bytes: &[u8]) -> Result<StateFile, String> {
    if !is_state_file(bytes) {
        return Err("ce n'est pas un fichier d'état complet".into());
    }
    let h = MAGIC.len();
    if bytes.len() < h + 4 {
        return Err("fichier d'état tronqué".into());
    }
    let version = u32::from_le_bytes([bytes[h], bytes[h + 1], bytes[h + 2], bytes[h + 3]]);
    if version != VERSION {
        return Err(format!("état enregistré au format {version}, ce programme lit le format {VERSION}"));
    }
    let mut f: StateFile = bincode::deserialize(&bytes[h + 4..]).map_err(|e| format!("fichier d'état illisible : {e}"))?;
    check(&f.sim)?;
    // champs non enregistrés : profil de temps (repart de zéro)
    f.sim.profile = Default::default();
    Ok(f)
}

/// Cohérence des dimensions (un fichier corrompu ou forgé ne doit pas faire paniquer le solveur sur un indice hors bornes).
fn check(s: &MultiphaseSim) -> Result<(), String> {
    let (nx, ny) = (s.nx, s.ny);
    let n = nx.checked_mul(ny).ok_or("grille trop grande")?;
    let (np, ng) = (s.reg.phases.len(), s.groups.len().max(1));
    let bad = |what: &str| Err(format!("fichier d'état incohérent ({what})"));
    if nx == 0 || ny == 0 || !(s.dx > 0.0) {
        return bad("grille");
    }
    if s.mass.len() != np || s.alpha.len() != np || s.mass.iter().chain(s.alpha.iter()).any(|v| v.len() != n) {
        return bad("masses");
    }
    if s.eg.len() != ng || s.tg.len() != ng || s.eg.iter().chain(s.tg.iter()).any(|v| v.len() != n) {
        return bad("groupes");
    }
    if s.group_of.len() != np || s.group_of.iter().any(|&g| g >= ng) {
        return bad("groupes des phases");
    }
    let cells = [
        s.e.len(), s.temp.len(), s.p.len(), s.solid.len(), s.rho_mix.len(), s.mu_cell.len(), s.k_cell.len(), s.special.len(), s.thermal.len(),
        s.sigma.len(), s.skel.len(), s.body_frac.len(),
    ];
    if cells.iter().any(|&l| l != n) {
        return bad("champs par cellule");
    }
    if s.bodies.iter().any(|b| b.material >= s.reg.solids.len()) {
        return bad("matériau d'un corps rigide");
    }
    if (!s.body_vel_u.is_empty() && s.body_vel_u.len() != (nx + 1) * ny) || (!s.body_vel_v.is_empty() && s.body_vel_v.len() != nx * (ny + 1)) {
        return bad("faces des corps rigides");
    }
    if s.u.len() != (nx + 1) * ny || s.v.len() != nx * (ny + 1) {
        return bad("vitesses");
    }
    if s.ambient_phase >= np.max(1) || s.solid.iter().any(|&k| k >= 0 && k as usize >= s.reg.solids.len()) {
        return bad("indices de matériaux");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::mp_view::{apply_tool, build_scenario, MpScenario, MpTool};

    fn same(a: &MultiphaseSim, b: &MultiphaseSim) -> bool {
        a.time == b.time
            && a.step_count == b.step_count
            && a.mass == b.mass
            && a.e == b.e
            && a.eg == b.eg
            && a.p == b.p
            && a.u == b.u
            && a.v == b.v
            && a.chem_heat == b.chem_heat
            && a.boundary_flux == b.boundary_flux
    }

    /// avance `a`, l'enregistre, le relit dans `b`, avance les deux : résultats identiques au bit près
    fn roundtrip(mut a: MultiphaseSim, warm: usize, more: usize) {
        for _ in 0..warm {
            a.step();
        }
        let bytes = state_to_bytes(&a, None, Some(3), None).unwrap();
        assert!(is_state_file(&bytes));
        let f = state_from_bytes(&bytes).unwrap();
        assert_eq!(f.scenario, Some(3));
        let mut b = f.sim;
        assert!(same(&a, &b), "état relu différent");
        for _ in 0..more {
            a.step();
            b.step();
        }
        assert!(same(&a, &b), "les deux simulations divergent après relecture");
    }

    #[test]
    fn full_state_roundtrip_continues_bit_for_bit() {
        // barrage : interface, vapeur possible, bords
        roundtrip(build_scenario(MpScenario::DamBreak, 40, 30, 0.01, 9.81), 60, 40);
        // combustion : registre avec réactions, mélanges, facteur de flamme, chaleur dégagée
        roundtrip(build_scenario(MpScenario::PremixedFlame, 48, 8, 0.001, 0.0), 80, 30);
        // ébullition : solides chauffants, changement de phase
        let mut s = build_scenario(MpScenario::Boiling, 32, 24, 0.005, 9.81);
        let cells = [(5usize, 5usize), (6, 5), (7, 5)];
        apply_tool(&mut s, MpTool::Heat, &cells, 600.0, (0.0, 0.0), 1.0 / 60.0);
        roundtrip(s, 50, 30);
    }

    #[test]
    fn foreign_or_damaged_files_are_rejected_cleanly() {
        let s = build_scenario(MpScenario::DamBreak, 16, 16, 0.01, 9.81);
        let mut bytes = state_to_bytes(&s, None, None, None).unwrap();
        // ancienne scène JSON
        assert!(!is_state_file(b"{\"version\":1}"));
        assert!(state_from_bytes(b"{\"version\":1}").is_err());
        // autre version
        let mut v = bytes.clone();
        v[MAGIC.len()] = 99;
        assert!(state_from_bytes(&v).err().unwrap().contains("format 99"));
        // tronqué
        bytes.truncate(bytes.len() / 2);
        assert!(state_from_bytes(&bytes).is_err());
    }
}
