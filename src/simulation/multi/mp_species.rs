//! Espèces chimiques : diffusion moléculaire, cinétique chimique, flammes épaissies et rayonnement des gaz chauds.
//!
//! * Diffusion (groupes MISCIBLES) : loi de Fick par espèce avec vitesse de correction (approximation « mélange » de Hirschfelder–Curtiss) :
//!   J_k = −ρ D_k ∇Y_k + Y_k Σ_j ρ D_j ∇Y_j, donc Σ_k J_k = 0 (la masse du mélange n'est pas transportée par la diffusion) et chaque espèce
//!   a SA diffusivité (l'hydrogène diffuse 3 à 4 fois plus vite que l'oxygène : instabilités thermo-diffusives des flammes pauvres).
//!   D_k(T, p) : gaz ∝ T^1.75/p, liquides ∝ T/μ. L'énergie interne accompagne chaque espèce : l'énergie totale est conservée exactement.
//!
//! * Cinétique (`react_step`) : loi d'Arrhenius modifiée + action de masse, réactions réversibles par l'équilibre thermodynamique
//!   (voir `phases::Reaction`). Intégration cellule par cellule (en parallèle) à volume et énergie interne constants (découplage
//!   d'opérateurs : l'expansion est ensuite produite par la projection de pression) avec sous-pas adaptatifs — point milieu (ordre 2),
//!   échauffement ≤ 60 K et consommation ≤ 40 % par sous-pas pour les réactions irréversibles, Euler IMPLICITE (Newton) pour les réactions réversibles,
//!   raides près de l'équilibre (dissociation de CO2 / H2O). Masse exactement conservée réaction par réaction, énergie interne inchangée :
//!   la chaleur de réaction vient des enthalpies de formation portées par les décalages e0.
//!   Couplage avec l'écoulement : la chimie d'un pas ne peut pas augmenter le volume d'une cellule (à p, T) de plus de `chem_volume_limit` ;
//!   le pas de temps suivant est réduit pour suivre les dégagements de chaleur violents (allumage, explosion).
//!
//! * Flammes de prémélange (`update_flame_factor`) : modèle de flamme épaissie dynamique (TFM, Colin et al. 2000 ; Légier et al. 2000).
//!   Une flamme réelle a une épaisseur δ_L ≈ 0.3–0.5 mm, bien plus petite que la maille : on multiplie les diffusivités (thermique et des
//!   espèces) par F et on divise la vitesse de réaction par F dans la zone de flamme. La vitesse de flamme S_L ∝ √(D·ω) est conservée et
//!   la flamme est résolue sur ≈ `flame_resolution` mailles. Hors de la flamme (senseur : dégagement de chaleur, dilaté sur l'épaisseur),
//!   F = 1 : l'auto-allumage et le mélange ne sont pas modifiés.
//!
//! * Flammes de DIFFUSION (`diffusion_mixing`) : sur ces mailles, la cinétique globale évaluée à la température moyenne d'une maille où
//!   la flamme se mêle à l'air froid est trop lente — le combustible et l'air s'y côtoyaient sans brûler (une flamme de méthane seule
//!   décrochait et s'éteignait). Dans une maille chaude (≥ `DIFF_T_LIT`) où le combustible et le comburant arrivent de part et d'autre
//!   (indice de flamme de Takeno négatif), les réactions de flamme irréversibles avancent au moins au rythme du mélange τ = √(2Δ/g),
//!   comme dans les codes d'incendie ; fronts de prémélange et mélanges froids gardent leur cinétique.
//!
//! * Rayonnement (`radiate`) : modèle optiquement mince (TNF Workshop) q = 4σ Σ_k p_k a_P,k(T) (T⁴ − T_amb⁴), coefficients moyens de
//!   Planck de H2O et CO2. Pertes des gaz brûlés (refroidissement des fumées) ; surestime les pertes des très grandes flammes (opaques).

use super::multiphase::{clampf, temp_from_masses, MultiphaseSim};
use super::phases::{Phase, Radiator, Reaction, SurfaceRx, SHERWOOD};

/// cellules traitées par un appel de la chimie
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ChemSel {
    All,
    Flame,
    NonFlame,
}

/// constante de Stefan–Boltzmann (W/(m²·K⁴))
const SIGMA_SB: f64 = 5.670374419e-8;
/// plancher de concentration pour les ordres négatifs ou les exposants (o − ν) < 0 de la vitesse inverse (mol/m³)
const C_FLOOR: f64 = 1e-3;
/// échauffement (K) en dessous duquel l'avancement restant d'une réaction irréversible ne borne pas le sous-pas chimique (trace)
const TRACE_DT: f64 = 5.0;
/// nombre maximal de sous-pas chimiques par cellule et par pas
const MAX_CHEM_SUBSTEPS: usize = 4000;
/// température (K) au-dessus de laquelle une maille de flamme de diffusion brûle au rythme du mélange (voir `diffusion_mixing`) :
/// l'auto-inflammation du méthane (~810 K). Plus haut (1000 K), la couche de mélange à la base de la flamme des matières volatiles d'un
/// braséro, entre 700 et 1000 K, ne se rallumait pas depuis les braises du bord : la flamme décollait et s'éteignait en 2 à 4 s
pub(crate) const DIFF_T_LIT: f64 = 800.0;

/// résultat de l'intégration chimique d'une cellule
struct ChemOut {
    c: usize,
    m: Vec<f64>,
    temp: f64,
    q: f64,
    xi: Vec<f64>,
    dt_cell: f64,
}

/// état local d'une cellule pendant l'intégration chimique
struct CellChem<'a> {
    phases: &'a [Phase],
    rx: &'a [Reaction],
    /// masses partielles (kg/m³)
    m: Vec<f64>,
    /// énergie interne (J/m³), constante
    e: f64,
    /// facteur d'épaississement local
    f_eff: f64,
    /// rapport (masse des réactifs / masse des produits) stœchiométrique par réaction : impose la conservation exacte de la masse
    scale: Vec<f64>,
    /// réactions de surface (réactif granulaire) ; pression de la cellule (masse volumique des grains, diffusivité de l'oxydant)
    surf: &'a [Option<SurfaceRx>],
    p: f64,
    /// flamme de DIFFUSION allumée dans la cellule : 1/τ_mél (1/s) — les réactions de flamme irréversibles y avancent au moins de
    /// leur avancement possible en τ_mél (voir `diffusion_mixing`) ; 0 ailleurs
    mix: f64,
}

impl CellChem<'_> {
    fn temp(&self) -> f64 {
        temp_from_masses(self.phases, self.m.iter().copied(), self.e)
    }
    fn cap(&self, t: f64) -> f64 {
        self.m.iter().enumerate().map(|(k, &mk)| mk.max(0.0) * self.phases[k].c_e_at(t)).sum()
    }
    fn conc(&self, k: usize) -> f64 {
        self.m[k].max(0.0) / self.phases[k].molar_mass
    }
    /// vitesse NETTE de la réaction j (mol/m³/s) à la température t
    fn rate(&self, j: usize, t: f64) -> f64 {
        let (rf, rr) = self.rate_parts(j, t);
        rf - rr
    }
    /// Débit du capteur de flamme : débit net pondéré par l'écart à l'équilibre (1 − r_inv/r_dir) — le débit net dans un mélange
    /// frais, ~0 près de l'équilibre. Sans cette pondération, l'équilibre de dissociation de H2O des gaz brûlés (H2 + ½O2 ⇌ H2O,
    /// δ_L > 0), déplacé à chaque oscillation de p / T, franchissait le seuil de 2 % : le masque d'épaississement clignotait sur
    /// 10 mailles de gaz brûlés et entretenait une oscillation thermoacoustique numérique (flamme 14 % trop lente au dt par défaut).
    fn sensor_rate(&self, j: usize, t: f64) -> f64 {
        let (rf, rr) = self.rate_parts(j, t);
        if rf <= 0.0 {
            return 0.0;
        }
        (rf - rr).max(0.0) * (1.0 - rr / rf).clamp(0.0, 1.0)
    }
    /// (débit direct, débit inverse) en mol/(m³·s) ; inverse nul pour une réaction irréversible
    fn rate_parts(&self, j: usize, t: f64) -> (f64, f64) {
        if let Some(sr) = self.surf[j] {
            return (self.surface_rate(j, sr, t), 0.0);
        }
        let r = &self.rx[j];
        // réaction en volume d'une phase granulaire (pyrolyse) : opérateur à part, à la température du grain (`pyrolysis_step`)
        if r.bulk {
            return (0.0, 0.0);
        }
        let k = r.k_forward(t) / if r.flame_thickness > 0.0 { self.f_eff } else { 1.0 };
        if k <= 0.0 {
            return (0.0, 0.0);
        }
        let mut modif = 1.0;
        for &(s, o) in &r.modifiers {
            let c = self.conc(s);
            modif *= if o < 0.0 { c.max(C_FLOOR).powf(o) } else if o == 0.0 { 1.0 } else { c.powf(o) };
        }
        let mut rf = k * modif;
        for (i, &(s, _)) in r.reactants.iter().enumerate() {
            let o = r.order(i);
            let c = self.conc(s);
            // réactif absent : pas de réaction directe, quel que soit l'ordre (un ordre NÉGATIF — CH4^−0.3 du mécanisme global — donnait,
            // via le plancher C_FLOOR, un débit direct énorme sans méthane : aucune réaction possible (borne nulle) mais toutes les
            // cellules « actives » et des sous-pas chimiques de ~1e-6 s bridés par une consommation d'O2 fictive — explosion H2–air
            // confinée ~20 fois plus lente)
            rf *= if c <= 0.0 { 0.0 } else if o < 0.0 { c.max(C_FLOOR).powf(o) } else if o == 0.0 { 1.0 } else { c.powf(o) };
        }
        if !r.reversible {
            // flamme de diffusion allumée : au moins le débit du mélange (voir `diffusion_mixing`)
            if self.mix > 0.0 && r.flame_thickness > 0.0 {
                rf = rf.max(self.bounds(j).1 * self.mix);
            }
            return (rf, 0.0);
        }
        // vitesse inverse k/Kc · Π c_prod^ν · Π c_réac^(o−ν) : l'équilibre Q = Kc est exact quels que soient les ordres
        let ln_kc = r.ln_kc(self.phases, t);
        let mut ln_rr = (k * modif).ln() - ln_kc;
        for &(s, nu) in &r.products {
            let c = self.conc(s);
            if c <= 0.0 {
                return (rf, 0.0);
            }
            ln_rr += nu * c.ln();
        }
        for (i, &(s, nu)) in r.reactants.iter().enumerate() {
            let d = r.order(i) - nu;
            if d != 0.0 {
                ln_rr += d * self.conc(s).max(C_FLOOR).ln();
            }
        }
        (rf, ln_rr.min(700.0).exp())
    }
    /// surface des grains de la réaction de surface (m²/m³ de cellule) : 6 α_s / d
    fn surface_area(&self, sr: SurfaceRx, t: f64) -> f64 {
        let ms = self.m[sr.solid];
        if ms <= 0.0 { 0.0 } else { 6.0 * ms / self.phases[sr.solid].rho(self.p, t) / sr.d }
    }
    /// vitesse CINÉTIQUE d'une réaction de surface : S_v · k_s · Π c^o (réactifs non granulaires) · modulateurs
    fn surface_kinetic(&self, j: usize, sr: SurfaceRx, t: f64) -> f64 {
        let r = &self.rx[j];
        let ks = r.k_forward(t);
        let sv = self.surface_area(sr, t);
        if ks <= 0.0 || sv <= 0.0 {
            return 0.0;
        }
        let mut rk = sv * ks;
        for &(s, o) in &r.modifiers {
            let c = self.conc(s);
            rk *= if o < 0.0 { c.max(C_FLOOR).powf(o) } else if o == 0.0 { 1.0 } else { c.powf(o) };
        }
        for (i, &(s, _)) in r.reactants.iter().enumerate() {
            if s == sr.solid {
                continue;
            }
            let o = r.order(i);
            let c = self.conc(s);
            rk *= if c <= 0.0 { 0.0 } else if o < 0.0 { c.max(C_FLOOR).powf(o) } else if o == 0.0 { 1.0 } else { c.powf(o) };
        }
        rk
    }
    /// vitesse d'une réaction de surface (voir `phases::Reaction`) : la cinétique, limitée par l'apport d'oxydant jusqu'aux grains
    /// S_v · k_d · C_ox, PARTAGÉ entre toutes les réactions de surface du même couple (grain, oxydant) — C + ½O2 → CO et C + O2 → CO2
    /// puisent dans le même flux d'oxygène : consommation d'oxydant K = Σ ν_ox r_cin, facteur commun D / (K + D) (pour une seule
    /// réaction d'ordre 1 : somme harmonique 1/(1/k_s + 1/k_d))
    fn surface_rate(&self, j: usize, sr: SurfaceRx, t: f64) -> f64 {
        let rk = self.surface_kinetic(j, sr, t);
        let Some(ox) = sr.ox else { return rk };
        if rk <= 0.0 {
            return 0.0;
        }
        let nu_ox = |i: usize| self.rx[i].reactants.iter().find(|&&(k, _)| k == ox).map_or(1.0, |&(_, nu)| nu);
        let mut k_ox = 0.0;
        for (i, s2) in self.surf.iter().enumerate() {
            if let Some(s2) = *s2 {
                if s2.solid == sr.solid && s2.ox == Some(ox) {
                    k_ox += nu_ox(i) * if i == j { rk } else { self.surface_kinetic(i, s2, t) };
                }
            }
        }
        let d_sup = self.surface_area(sr, t) * SHERWOOD * self.phases[ox].diffusivity_at(t, self.p) / sr.d * self.conc(ox);
        if k_ox + d_sup > 0.0 { rk * d_sup / (k_ox + d_sup) } else { 0.0 }
    }
    /// avancement maximal (mol/m³) dans le sens direct (> 0) et inverse (< 0) permis par les masses disponibles
    fn bounds(&self, j: usize) -> (f64, f64) {
        let r = &self.rx[j];
        let fwd = r.reactants.iter().map(|&(s, nu)| self.m[s].max(0.0) / (nu * self.phases[s].molar_mass)).fold(f64::INFINITY, f64::min);
        let rev = if r.reversible && self.surf[j].is_none() {
            r.products.iter().map(|&(s, nu)| self.m[s].max(0.0) / (nu * self.phases[s].molar_mass * self.scale[j])).fold(f64::INFINITY, f64::min)
        } else {
            0.0
        };
        (-rev, fwd)
    }
    /// applique l'avancement xi (mol/m³) de la réaction j (masse totale exactement conservée)
    fn apply(&mut self, j: usize, xi: f64) {
        let r = &self.rx[j];
        for &(s, nu) in &r.reactants {
            self.m[s] = (self.m[s] - nu * self.phases[s].molar_mass * xi).max(0.0);
        }
        for &(s, nu) in &r.products {
            self.m[s] = (self.m[s] + nu * self.phases[s].molar_mass * xi * self.scale[j]).max(0.0);
        }
    }
    /// −Δu_j(T) (J/mol) : chaleur dégagée par mole d'avancement à volume constant
    fn heat(&self, j: usize, t: f64) -> f64 {
        let r = &self.rx[j];
        let s = |v: &[(usize, f64)], f: f64| v.iter().map(|&(k, nu)| nu * f * self.phases[k].molar_mass * self.phases[k].e_int(t)).sum::<f64>();
        s(&r.reactants, 1.0) - s(&r.products, self.scale[j])
    }
    /// volume occupé par les masses à (p, T) rapporté à la cellule (≈ 1 à l'équilibre mécanique)
    fn volume(&self, p: f64, t: f64) -> f64 {
        self.m.iter().enumerate().map(|(k, &mk)| mk.max(0.0) / self.phases[k].rho(p, t)).sum()
    }
}

/// Composition (fractions volumiques = molaires, gaz parfaits) après combustion COMPLÈTE d'un mélange par les réactions de flamme
/// (δ_L > 0) puis les autres réactions irréversibles, chacune jusqu'à épuisement de son réactif limitant. Sert à initialiser des noyaux
/// de gaz brûlés (allumage sans explosion parasite du noyau).
pub fn complete_combustion(reg: &super::phases::Registry, alpha: &[f64]) -> Vec<f64> {
    let mut n = alpha.to_vec();
    let mut order: Vec<usize> = (0..reg.reactions.len()).filter(|&j| reg.reactions[j].flame_thickness > 0.0).collect();
    order.extend((0..reg.reactions.len()).filter(|&j| reg.reactions[j].flame_thickness <= 0.0 && !reg.reactions[j].reversible));
    for _ in 0..3 {
        for &j in &order {
            let r = &reg.reactions[j];
            let xi = r.reactants.iter().map(|&(k, nu)| n[k].max(0.0) / nu).fold(f64::INFINITY, f64::min);
            if !(xi > 0.0) || !xi.is_finite() {
                continue;
            }
            for &(k, nu) in &r.reactants {
                n[k] = (n[k] - nu * xi).max(0.0);
            }
            for &(k, nu) in &r.products {
                n[k] += nu * xi;
            }
        }
    }
    let tot: f64 = n.iter().sum();
    n.iter().map(|x| x / tot.max(1e-300)).collect()
}

impl MultiphaseSim {
    // ================================================================ diffusion des espèces
    /// Diffusion de Fick des espèces dans les groupes miscibles (diffusivité propre à chaque espèce + vitesse de correction).
    /// Sans groupe miscible (cas par défaut) : ne fait rien.
    pub(crate) fn diffuse_species(&mut self, dt: f64) {
        let groups: Vec<Vec<usize>> = self.reg.miscible_groups().into_iter().filter(|g| g.len() > 1).collect();
        if groups.is_empty() {
            return;
        }
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let n = nx * ny;
        for g in &groups {
            if g.iter().all(|&k| !(self.reg.phases[k].diffusivity > 0.0)) {
                continue;
            }
            // groupe thermique de ce groupe miscible : la diffusion déplace masse ET énergie à sa température
            let gt = self.group_of[g[0]];
            // diffusivités locales D_k(T, p) × F_eff (flamme épaissie)
            let dloc: Vec<Vec<f64>> = g
                .iter()
                .map(|&k| {
                    super::mp_par::collect(n, super::mp_par::MIN_HEAVY, |c| {
                        if self.is_solid(c) { 0.0 } else { self.reg.phases[k].diffusivity_at(self.tg[gt][c], self.p[c]).max(0.0) * self.flame_factor[c].max(1.0) }
                    })
                })
                .collect();
            let dmax = dloc.iter().flat_map(|v| v.iter()).fold(0.0f64, |a, &b| a.max(b));
            if dmax <= 0.0 {
                continue;
            }
            let nsub = ((dt * dmax / (0.2 * dx * dx)).ceil() as usize).clamp(1, 500);
            let ds = dt / nsub as f64;
            let ng = g.len();
            // (choix de la forme selon la taille seulement : le résultat ne dépend pas du nombre de fils)
            let par = n >= 2048;
            let mut mg = vec![0.0; n];
            let mut y: Vec<Vec<f64>> = vec![vec![0.0; n]; ng];
            let mut new_m: Vec<Vec<f64>> = vec![vec![0.0; n]; ng];
            let mut new_e = vec![0.0; n];
            for _ in 0..nsub {
                // masse du groupe et fractions massiques
                {
                    let mp = super::multiphase::SyncPtr(mg.as_mut_ptr());
                    let yp: Vec<super::multiphase::SyncPtr<f64>> = y.iter_mut().map(|v| super::multiphase::SyncPtr(v.as_mut_ptr())).collect();
                    let this = &*self;
                    super::mp_par::for_ranges(n, super::mp_par::MIN_HEAVY, |lo, hi| {
                        for c in lo..hi {
                            let m: f64 = if this.is_solid(c) { 0.0 } else { g.iter().map(|&k| this.mass[k][c].max(0.0)).sum() };
                            // SAFETY : la cellule c n'est écrite que par sa tranche
                            unsafe {
                                *mp.at(c) = m;
                                for (q, &k) in g.iter().enumerate() {
                                    *yp[q].at(c) = if m > 0.0 { this.mass[k][c].max(0.0) / m } else { 0.0 };
                                }
                            }
                        }
                    });
                }
                // flux à travers la face (a, b), a < b, de a vers b : même calcul vu des deux cellules ⇒ conservation exacte
                let this = &*self;
                let (mg_, y_, dl_) = (&mg, &y, &dloc);
                let bodies = this.has_bodies();
                let face = |a: usize, b: usize, out: &mut [f64; super::phases::MAX_PHASES], ef: &mut f64| -> bool {
                    if this.is_solid(a) || this.is_solid(b) || mg_[a] <= 1e-12 || mg_[b] <= 1e-12 {
                        return false;
                    }
                    // corps rigides : diffusion par la part ouverte de la face (b voisine de droite ou du haut de a) ; sans corps : 1
                    let op = if !bodies {
                        1.0
                    } else if b == a + 1 {
                        this.u_open(b % nx, b / nx)
                    } else {
                        this.v_open(b % nx, b / nx)
                    };
                    if op == 0.0 {
                        return false;
                    }
                    let mbar = 0.5 * (mg_[a] + mg_[b]);
                    let mut jsum = 0.0;
                    for q in 0..ng {
                        let (da, db) = (dl_[q][a], dl_[q][b]);
                        let dk = 2.0 * da * db / (da + db).max(1e-300);
                        out[q] = op * (dk * mbar * (y_[q][a] - y_[q][b]) / (dx * dx) * ds);
                        jsum += out[q];
                    }
                    let mut scale = 1.0f64;
                    for (q, &k) in g.iter().enumerate() {
                        out[q] -= 0.5 * (y_[q][a] + y_[q][b]) * jsum;
                        let avail = 0.25 * if out[q] > 0.0 { this.mass[k][a].max(0.0) } else { this.mass[k][b].max(0.0) };
                        if out[q].abs() > avail {
                            scale = scale.min(avail / out[q].abs());
                        }
                    }
                    *ef = 0.0;
                    for (q, &k) in g.iter().enumerate() {
                        out[q] *= scale;
                        let t_don = if out[q] > 0.0 { this.tg[gt][a] } else { this.tg[gt][b] };
                        *ef += out[q] * this.reg.phases[k].e_int(t_don);
                    }
                    true
                };
                let nm: Vec<super::multiphase::SyncPtr<f64>> = new_m.iter_mut().map(|v| super::multiphase::SyncPtr(v.as_mut_ptr())).collect();
                let ne = super::multiphase::SyncPtr(new_e.as_mut_ptr());
                let row = |j: usize| {
                    let mut out = [0.0f64; super::phases::MAX_PHASES];
                    let mut ef = 0.0;
                    for i in 0..nx {
                        let c = j * nx + i;
                        let mut dm = [0.0f64; super::phases::MAX_PHASES];
                        let mut de = 0.0;
                        let nbs = [(i > 0).then(|| c - 1), (i + 1 < nx).then(|| c + 1), (j > 0).then(|| c - nx), (j + 1 < ny).then(|| c + nx)];
                        for b in nbs.into_iter().flatten() {
                            let (lo, hi) = if c < b { (c, b) } else { (b, c) };
                            if face(lo, hi, &mut out, &mut ef) {
                                let sgn = if c == lo { -1.0 } else { 1.0 };
                                for q in 0..ng {
                                    dm[q] += sgn * out[q];
                                }
                                de += sgn * ef;
                            }
                        }
                        // SAFETY : la cellule c n'est écrite que par sa ligne
                        unsafe {
                            for (q, &k) in g.iter().enumerate() {
                                *nm[q].at(c) = (this.mass[k][c] + dm[q]).max(0.0);
                            }
                            *ne.at(c) = this.eg[gt][c] + de;
                        }
                    }
                };
                if par {
                    // grandes grilles : chaque cellule rassemble les flux de ses 4 faces (chaque face calculée deux fois, identiquement),
                    // par tranches de lignes en parallèle
                    super::mp_par::for_rows(ny, nx, super::mp_par::MIN_HEAVY, row);
                } else {
                    // séquentiel : chaque face une seule fois, répartie sur ses deux cellules
                    for (q, &k) in g.iter().enumerate() {
                        new_m[q].copy_from_slice(&self.mass[k]);
                    }
                    new_e.copy_from_slice(&self.eg[gt]);
                    let mut out = [0.0f64; super::phases::MAX_PHASES];
                    let mut ef = 0.0;
                    for j in 0..ny {
                        for i in 0..nx {
                            let a = j * nx + i;
                            for b in [(i + 1 < nx).then(|| a + 1), (j + 1 < ny).then(|| a + nx)].into_iter().flatten() {
                                if face(a, b, &mut out, &mut ef) {
                                    for q in 0..ng {
                                        new_m[q][a] -= out[q];
                                        new_m[q][b] += out[q];
                                    }
                                    new_e[a] -= ef;
                                    new_e[b] += ef;
                                }
                            }
                        }
                    }
                    for q in 0..ng {
                        for v in new_m[q].iter_mut() {
                            *v = v.max(0.0);
                        }
                    }
                }
                for (q, &k) in g.iter().enumerate() {
                    std::mem::swap(&mut self.mass[k], &mut new_m[q]);
                }
                std::mem::swap(&mut self.eg[gt], &mut new_e);
                self.sync_cells(|_| true);
            }
        }
    }

    // ================================================================ flamme épaissie
    /// Facteur d'épaississement F_eff de chaque cellule pour ce pas (TFM dynamique). Senseur : puissance dégagée par les réactions « de
    /// flamme » (δ_L > 0) au-dessus de 2 % du maximum du domaine (et d'un échauffement de 50 K/s), dilatée sur `flame_resolution` + 1 mailles
    /// pour couvrir la zone de préchauffage.
    pub(crate) fn update_flame_factor(&mut self, _dt: f64) {
        let n = self.nx * self.ny;
        let flames: Vec<usize> = (0..self.reg.reactions.len()).filter(|&j| self.reg.reactions[j].flame_thickness > 0.0).collect();
        let f_max = flames
            .iter()
            .map(|&j| self.flame_resolution * self.dx / self.reg.reactions[j].flame_thickness)
            .fold(1.0f64, f64::max);
        if !self.chemistry || flames.is_empty() || f_max <= 1.0 {
            self.flame_factor.iter_mut().for_each(|f| *f = 1.0);
            return;
        }
        let rx = &self.reg.reactions;
        let phases = &self.reg.phases;
        let np = phases.len();
        let mut omega = vec![0.0f64; n];
        let surf: Vec<Option<SurfaceRx>> = rx.iter().map(|r| r.surface(phases)).collect();
        for c in 0..n {
            if self.is_solid(c) {
                continue;
            }
            let cc = CellChem {
                phases,
                rx,
                m: (0..np).map(|k| self.mass[k][c]).collect(),
                e: self.e[c],
                f_eff: 1.0,
                scale: vec![1.0; rx.len()],
                surf: &surf,
                p: self.p[c],
                mix: 0.0,
            };
            let mut w = 0.0;
            for &j in &flames {
                if rx[j].reactants.iter().any(|&(s, _)| self.mass[s][c] <= 1e-12) {
                    continue;
                }
                // température du groupe des réactifs
                let t = self.phase_temp(rx[j].reactants[0].0, c);
                w += (cc.sensor_rate(j, t) * cc.heat(j, t)).abs();
            }
            omega[c] = w;
        }
        let wmax = omega.iter().fold(0.0f64, |a, &b| a.max(b));
        let mut mask: Vec<f64> = (0..n)
            .map(|c| if wmax > 0.0 && omega[c] > 0.02 * wmax && omega[c] > 50.0 * self.heat_capacity(c) { 1.0 } else { 0.0 })
            .collect();
        // dilatation (filtre max séparable) sur r mailles
        let r = self.flame_resolution.ceil() as usize + 1;
        let (nx, ny) = (self.nx, self.ny);
        let mut tmp = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                let (a, b) = (i.saturating_sub(r), (i + r).min(nx - 1));
                tmp[j * nx + i] = (a..=b).map(|ii| mask[j * nx + ii]).fold(0.0, f64::max);
            }
        }
        for j in 0..ny {
            for i in 0..nx {
                let (a, b) = (j.saturating_sub(r), (j + r).min(ny - 1));
                mask[j * nx + i] = (a..=b).map(|jj| tmp[jj * nx + i]).fold(0.0, f64::max);
            }
        }
        for c in 0..n {
            self.flame_factor[c] = if self.is_solid(c) { 1.0 } else { 1.0 + (f_max - 1.0) * mask[c] };
        }
    }

    /// Mailles où brûle une flamme de DIFFUSION : 1/τ_mél, 0 ailleurs. Sur ces mailles (4 mm à 1 cm), la cinétique globale évaluée à la
    /// température MOYENNE d'une maille, où la flamme se mêle à l'air froid, est trop lente : le combustible et l'air s'y côtoyaient sans
    /// brûler — une flamme de méthane seule décrochait et s'éteignait (jet en 0.3 à 0.9 s, matières volatiles d'un braséro en 1.3 s),
    /// et les poches ainsi mêlées explosaient d'un coup. Comme dans les codes d'incendie (FDS), le combustible et le comburant réunis
    /// dans une telle maille brûlent au moins au rythme de leur mélange, τ_mél = √(2 Δ / g) (≈ 45 ms pour Δ = 1 cm). Critères : maille
    /// chaude (≥ `DIFF_T_LIT`), combustible et comburant présents, et flamme de diffusion — indice de Takeno ξ = ∇Y_c·∇Y_ox /
    /// (|∇Y_c| |∇Y_ox|) < −0.2 (ils arrivent de part et d'autre) : les fronts de PRÉMÉLANGE (ξ > 0, vitesse de flamme) et les mélanges
    /// froids (explosions, allumage) gardent leur cinétique.
    pub(crate) fn diffusion_mixing(&self) -> Vec<f64> {
        let (nx, ny) = (self.nx, self.ny);
        let n = nx * ny;
        let rx = &self.reg.reactions;
        let flames: Vec<usize> = (0..rx.len()).filter(|&j| rx[j].flame_thickness > 0.0 && !rx[j].reversible && rx[j].reactants.len() >= 2).collect();
        if !self.chemistry || flames.is_empty() {
            return vec![0.0; n];
        }
        let ox = rx[flames[0]].reactants[1].0;
        let mut fuels: Vec<usize> = flames.iter().map(|&j| rx[j].reactants[0].0).filter(|&k| k != ox).collect();
        fuels.sort_unstable();
        fuels.dedup();
        let np = self.reg.phases.len();
        let frac = |c: usize, ks: &[usize]| -> f64 {
            let tot: f64 = (0..np).map(|k| self.mass[k][c].max(0.0)).sum();
            if tot > 0.0 { ks.iter().map(|&k| self.mass[k][c].max(0.0)).sum::<f64>() / tot } else { 0.0 }
        };
        let yf: Vec<f64> = (0..n).map(|c| if self.is_solid(c) { 0.0 } else { frac(c, &fuels) }).collect();
        let yo: Vec<f64> = (0..n).map(|c| if self.is_solid(c) { 0.0 } else { frac(c, &[ox]) }).collect();
        let grad = |y: &[f64], i: usize, j: usize| -> (f64, f64) {
            let c = j * nx + i;
            let at = |q: usize| if self.is_solid(q) { y[c] } else { y[q] };
            let gx = match (i > 0, i + 1 < nx) {
                (true, true) => 0.5 * (at(c + 1) - at(c - 1)),
                (false, true) => at(c + 1) - y[c],
                (true, false) => y[c] - at(c - 1),
                _ => 0.0,
            };
            let gy = match (j > 0, j + 1 < ny) {
                (true, true) => 0.5 * (at(c + nx) - at(c - nx)),
                (false, true) => at(c + nx) - y[c],
                (true, false) => y[c] - at(c - nx),
                _ => 0.0,
            };
            (gx, gy)
        };
        let rate = 1.0 / (2.0 * self.dx / 9.81).sqrt();
        let mut mix = vec![0.0; n];
        for j in 0..ny {
            for i in 0..nx {
                let c = j * nx + i;
                if self.is_solid(c) || self.pinned(c) || yf[c] <= 1e-9 || yo[c] <= 1e-9 || self.phase_temp(ox, c) < DIFF_T_LIT {
                    continue;
                }
                let (fx, fy) = grad(&yf, i, j);
                let (ox_, oy) = grad(&yo, i, j);
                let (nf, no) = (fx.hypot(fy), ox_.hypot(oy));
                if nf < 1e-6 || no < 1e-6 {
                    continue;
                }
                if (fx * ox_ + fy * oy) / (nf * no) < -0.2 {
                    mix[c] = rate;
                }
            }
        }
        mix
    }

    // ================================================================ cinétique
    /// Réactions chimiques (APRÈS conduction et diffusion, AVANT le changement de phase). Voir l'en-tête du module.
    pub(crate) fn react_step(&mut self, dt: f64) {
        self.react_step_sel(dt, ChemSel::All, true);
    }

    /// Chimie des seules cellules `sel` (flamme : F > 1 ; hors flamme : F = 1) — sous-cyclage des flammes (voir
    /// `flame_thermo_substeps`). `first` : premier appel du pas (dt_chem repart de cet appel, sinon minimum avec les précédents).
    pub(crate) fn react_step_sel(&mut self, dt: f64, sel: ChemSel, first: bool) {
        if self.reg.reactions.is_empty() || !self.chemistry {
            self.dt_chem = f64::INFINITY;
            self.heat_release.iter_mut().for_each(|h| *h = 0.0);
            return;
        }
        let n = self.nx * self.ny;
        let np = self.reg.phases.len();
        let nre = self.reg.reactions.len();
        let a2 = self.dx * self.dx;
        let phases = self.reg.phases.clone();
        let rx = self.reg.reactions.clone();
        let scale: Vec<f64> = rx
            .iter()
            .map(|r| {
                let (a, b) = r.mass_balance(&phases);
                if b > 0.0 { a / b } else { 1.0 }
            })
            .collect();
        if self.reacted.len() != nre {
            self.reacted = vec![0.0; nre];
        }
        let surf: Vec<Option<SurfaceRx>> = rx.iter().map(|r| r.surface(&phases)).collect();
        let target = 0.5 * self.chem_volume_limit;
        let selected = |s: &Self, c: usize| match sel {
            ChemSel::All => true,
            ChemSel::Flame => s.flame_factor[c] > 1.0,
            ChemSel::NonFlame => s.flame_factor[c] <= 1.0,
        };
        for c in 0..n {
            if selected(self, c) {
                self.heat_release[c] = 0.0;
            }
        }
        let reactive = self.reactive_groups();
        let mix = self.diffusion_mixing();
        // intégration indépendante cellule par cellule : en parallèle, puis application des résultats dans l'ordre des cellules (sommes
        // globales inchangées) ; températures recalculées ensuite, en parallèle
        let this = &*self;
        let cell = |c: usize| if selected(this, c) { this.integrate_cell(c, dt, target, &phases, &rx, &scale, &surf, &reactive, &mix) } else { None };
        let out: Vec<ChemOut> =
            super::mp_par::map_ranges(n, super::mp_par::MIN_HEAVY, |lo, hi| (lo..hi).filter_map(cell).collect::<Vec<ChemOut>>()).into_iter().flatten().collect();
        let mut touched = vec![false; n];
        let mut dt_next = f64::INFINITY;
        for o in out {
            let c = o.c;
            // seuls les groupes réactifs ont participé : leurs masses, et une température commune (énergie absolue conservée)
            for k in 0..np {
                if reactive[self.group_of[k]] {
                    self.mass[k][c] = o.m[k];
                }
            }
            // énergie absolue des groupes réactifs inchangée (la chimie la conserve) : exacte au bit près pour un seul groupe ;
            // plusieurs groupes réactifs dans la cellule prennent la température commune, le dernier reçoit le reste exact
            let rg: Vec<usize> = (0..self.groups.len()).filter(|&g| reactive[g] && self.group_mass(g, c) > 0.0).collect();
            let e_r: f64 = (0..self.groups.len()).filter(|&g| reactive[g]).map(|g| self.eg[g][c]).sum();
            for g in 0..self.groups.len() {
                if reactive[g] {
                    self.eg[g][c] = 0.0;
                }
            }
            // le reste exact va au groupe de plus grande capacité calorifique (une trace — grain de charbon dans le gaz — qui le recevait
            // prenait l'arrondi de toute l'énergie du sous-système : 7000 K sur le GPU f32)
            if let Some(&main) = rg.iter().max_by(|&&a, &&b| self.group_cap_at(a, c, o.temp).total_cmp(&self.group_cap_at(b, c, o.temp))) {
                let mut acc = 0.0;
                for &g in rg.iter().filter(|&&g| g != main) {
                    self.eg[g][c] = self.group_energy_at(g, c, o.temp);
                    acc += self.eg[g][c];
                }
                self.eg[main][c] = e_r - acc;
            }
            touched[c] = true;
            self.heat_release[c] = o.q / dt;
            self.chem_heat += o.q * a2;
            for j in 0..nre {
                self.reacted[j] += o.xi[j] * a2;
            }
            dt_next = dt_next.min(o.dt_cell);
        }
        self.sync_cells(|c| touched[c]);
        let v = if dt_next.is_finite() { dt_next.max(1e-8) } else { f64::INFINITY };
        self.dt_chem = if first { v } else { self.dt_chem.min(v) };
    }

    // ================================================================ pyrolyse (réactions en volume des phases granulaires)
    /// Réactions EN VOLUME des phases granulaires (pyrolyse du bois) pendant dt, hors de la chimie des cellules : vitesse k_eff(T_s) c_s à
    /// la température du GRAIN (k_eff = k / (1 + k τ), τ : temps de conduction interne des morceaux, voir `Reaction::bulk_tau`), intégrée
    /// exactement (premier ordre : avancement c_s (1 − e^(−ν k_eff dt)) / ν). Les produits naissent à la température du grain (gaz dans leur
    /// groupe, résidu granulaire — charbon de bois — dans le sien) et le groupe du grain leur cède exactement leur énergie (chaleur de
    /// réaction comprise) : énergie totale conservée. Dans la chimie des cellules (température commune des groupes qui réagissent), le bois
    /// froid se mêlait aux braises de la même cellule : le gaz y passait d'un coup à ~900 K, se dilatait, et le calcul divergeait.
    /// En parallèle (chaque cellule ne dépend que d'elle-même) ; avancements sommés ensuite dans l'ordre des cellules.
    ///
    /// COMBUSTION d'une poudre (`Reaction::burn` : poudre noire, qui porte son oxydant) : dès que ses grains atteignent leur température
    /// d'allumage — ou que le fluide qui les baigne la dépasse (gaz de combustion qui traversent le tas : leur surface s'allume en ~1 ms,
    /// bien avant que la chaleur gagne le cœur d'un grain de 2 mm, ~3 s ; `fluid_contact_temp`) —, leur surface régresse à
    /// r = a (p / 1 atm)ⁿ (loi de Vieille) ; ils sont consommés à ρ_s r S_v = m_s · 6 r / d (premier
    /// ordre exact) et les produits naissent à la température adiabatique T_f, Σ dm_p e_p(T_f) = dm_s e_s(T_s) (la flamme est collée au
    /// grain : les grains restants gardent leur température, les gaz et la fumée emportent la chaleur). Borne du pas : `burn_dt_bound`.
    pub(crate) fn pyrolysis_step(&mut self, dt: f64) {
        use super::mp_par::{for_ranges, MIN_HEAVY};
        use super::multiphase::SyncPtr;
        use super::phases::{adiabatic_t, burn_rate_constant};
        if !self.chemistry {
            return;
        }
        /// réaction, réactif granulaire, ν, τ, rapport de masses réactifs / produits, combustion (a, n, T d'allumage), volume de gaz × p
        /// par kg de réactif (J/kg)
        struct Bulk {
            j: usize,
            s: usize,
            nu: f64,
            tau: f64,
            scale: f64,
            burn: Option<(f64, f64, f64)>,
            vg: f64,
        }
        let phases = &self.reg.phases;
        let bulk: Vec<Bulk> = self
            .reg
            .reactions
            .iter()
            .enumerate()
            .filter(|(_, r)| r.bulk)
            .filter_map(|(j, r)| {
                let &(s, nu) = r.reactants.first().filter(|&&(k, _)| phases[k].is_granular())?;
                let (a, b) = r.mass_balance(phases);
                let scale = if b > 0.0 { a / b } else { 1.0 };
                Some(Bulk { j, s, nu, tau: r.bulk_tau(phases), scale, burn: r.burn(), vg: r.burn_gas_volume(phases) })
            })
            .collect();
        if bulk.is_empty() {
            return;
        }
        let n = self.nx * self.ny;
        let nb = bulk.len();
        let mut xi_cell = vec![0.0f64; n * nb];
        let mut touched = vec![false; n];
        {
            let mp: Vec<SyncPtr<f64>> = self.mass.iter_mut().map(|v| SyncPtr(v.as_mut_ptr())).collect();
            let egp: Vec<SyncPtr<f64>> = self.eg.iter_mut().map(|v| SyncPtr(v.as_mut_ptr())).collect();
            let (xp, tp) = (SyncPtr(xi_cell.as_mut_ptr()), SyncPtr(touched.as_mut_ptr()));
            let this = &*self;
            let bulk = &bulk;
            for_ranges(n, MIN_HEAVY, |lo, hi| {
                for c in lo..hi {
                    if this.is_solid(c) || this.pinned(c) {
                        continue;
                    }
                    for (q, b) in bulk.iter().enumerate() {
                        let (j, s, nu, tau, scale) = (b.j, b.s, b.nu, b.tau, b.scale);
                        let r = &this.reg.reactions[j];
                        // SAFETY (tout le bloc) : la cellule c n'est lue et écrite que par sa tranche
                        unsafe {
                            let ms = *mp[s].at(c);
                            if ms <= 1e-12 {
                                continue;
                            }
                            let gs = this.group_of[s];
                            let ts = this.tg[gs][c];
                            if let Some((a, ex, t_ign)) = b.burn {
                                if ts < t_ign && this.fluid_contact_temp(c) < t_ign {
                                    continue;
                                }
                                let ph = &this.reg.phases;
                                let kb = burn_rate_constant(a, ex, this.p[c], ph[s].grain_diameter);
                                let xi = ms / ph[s].molar_mass * (1.0 - (-kb * dt).exp()) / nu;
                                let dms = (nu * xi * ph[s].molar_mass).min(ms);
                                let e_in = dms * ph[s].e_int(ts);
                                let (mut ce, mut sl, mut e0) = (0.0, 0.0, 0.0);
                                for &(p, nup) in &r.products {
                                    let dm = xi * nup * ph[p].molar_mass * scale;
                                    ce += dm * ph[p].c_e();
                                    sl += dm * ph[p].cp_slope;
                                    e0 += dm * ph[p].e0;
                                }
                                let tf = adiabatic_t(ce, sl, e0, e_in);
                                let mut e_out = 0.0;
                                for &(p, nup) in &r.products {
                                    let dm = xi * nup * ph[p].molar_mass * scale;
                                    let ep = dm * ph[p].e_int(tf);
                                    *mp[p].at(c) += dm;
                                    *egp[this.group_of[p]].at(c) += ep;
                                    e_out += ep;
                                }
                                // (arrondi de T_f : le reste au groupe du premier produit, énergie exactement conservée)
                                if let Some(&(p0, _)) = r.products.first() {
                                    *egp[this.group_of[p0]].at(c) += e_in - e_out;
                                }
                                *mp[s].at(c) -= dms;
                                *egp[gs].at(c) -= e_in;
                                *xp.at(c * nb + q) = xi;
                                *tp.at(c) = true;
                                continue;
                            }
                            let k = r.k_forward(ts);
                            if !(k > 0.0) {
                                continue;
                            }
                            let keff = if tau > 0.0 { k / (1.0 + k * tau) } else { k };
                            let ph = &this.reg.phases;
                            let xi = ms / ph[s].molar_mass * (1.0 - (-nu * keff * dt).exp()) / nu;
                            let dms = (nu * xi * ph[s].molar_mass).min(ms);
                            let mut e_out = 0.0;
                            for &(p, nup) in &r.products {
                                let dm = xi * nup * ph[p].molar_mass * scale;
                                let ep = dm * ph[p].e_int(ts);
                                *mp[p].at(c) += dm;
                                *egp[this.group_of[p]].at(c) += ep;
                                e_out += ep;
                            }
                            *mp[s].at(c) -= dms;
                            *egp[gs].at(c) -= e_out;
                            *xp.at(c * nb + q) = xi;
                            *tp.at(c) = true;
                        }
                    }
                }
            });
        }
        let a2 = self.dx * self.dx;
        if self.reacted.len() != self.reg.reactions.len() {
            self.reacted = vec![0.0; self.reg.reactions.len()];
        }
        for c in 0..n {
            for (q, b) in bulk.iter().enumerate() {
                let x = xi_cell[c * nb + q];
                if x != 0.0 {
                    self.reacted[b.j] += x * a2;
                }
            }
        }
        self.sync_cells(|c| touched[c]);
    }

    /// Borne du pas imposée par la combustion des poudres (`Reaction::burn`) : au plus le volume d'une cellule de gaz produit par pas, là
    /// où la poudre brûle ou est à moins de 50 K de s'allumer — une maille de poudre tassée en grains de 2 mm dégage ~8e4 fois son volume
    /// de gaz par seconde (sur l'état du début du pas ; miroir de `k_burn_dt` sur le GPU). ∞ sans poudre.
    pub(crate) fn burn_dt_bound(&self) -> f64 {
        use super::phases::burn_rate_constant;
        if !self.chemistry || !self.reg.reactions.iter().any(|r| r.burn().is_some()) {
            return f64::INFINITY;
        }
        let phases = &self.reg.phases;
        // (réactif granulaire, (a, n, T d'allumage), volume de gaz × p par kg de réactif)
        let burns: Vec<(usize, (f64, f64, f64), f64)> = self
            .reg
            .reactions
            .iter()
            .filter_map(|r| {
                let b = r.burn()?;
                let &(s, _) = r.reactants.first().filter(|&&(k, _)| phases[k].is_granular())?;
                Some((s, b, r.burn_gas_volume(phases)))
            })
            .collect();
        let n = self.nx * self.ny;
        super::mp_par::map_ranges(n, super::mp_par::MIN_HEAVY, |lo, hi| {
            let mut m = f64::INFINITY;
            for c in lo..hi {
                if self.is_solid(c) || self.pinned(c) {
                    continue;
                }
                for &(s, (a, ex, t_ign), vg) in &burns {
                    let ms = self.mass[s][c];
                    if ms <= 1e-12 || self.tg[self.group_of[s]][c].max(self.fluid_contact_temp(c)) < t_ign - 50.0 {
                        continue;
                    }
                    let p = self.p[c].max(1.0);
                    let rate = ms * burn_rate_constant(a, ex, p, phases[s].grain_diameter) * vg / p;
                    if rate > 0.0 {
                        m = m.min(1.0 / rate);
                    }
                }
            }
            m
        })
        .into_iter()
        .fold(f64::INFINITY, f64::min)
    }

    /// Intègre la chimie d'une cellule pendant dt (lecture seule de l'état) ; None si rien ne peut réagir.
    /// groupes contenant au moins une phase qui intervient dans une réaction (réactif, produit, modulateur)
    pub(crate) fn reactive_groups(&self) -> Vec<bool> {
        let mut r = vec![false; self.groups.len()];
        for k in 0..self.reg.phases.len() {
            if self.reg.is_cell_reactive(k) {
                r[self.group_of[k]] = true;
            }
        }
        r
    }

    /// Intègre la chimie d'une cellule pendant dt (lecture seule de l'état) ; None si rien ne peut réagir. Seuls les groupes
    /// réactifs participent (masses et énergie) : de l'eau liquide voisine ne sert pas de puits de chaleur instantané à une flamme.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn integrate_cell(
        &self, c: usize, dt: f64, target: f64, phases: &[Phase], rx: &[Reaction], scale: &[f64], surf: &[Option<SurfaceRx>], reactive: &[bool],
        mix: &[f64],
    ) -> Option<ChemOut> {
        let np = phases.len();
        let nre = rx.len();
        {

            if self.is_solid(c) || self.pinned(c) {
                return None;
            }
            // tri rapide : une réaction peut-elle avoir lieu ?
            let possible = rx.iter().any(|r| {
                r.reactants.iter().all(|&(s, _)| self.mass[s][c] > 1e-12) || (r.reversible && r.products.iter().all(|&(s, _)| self.mass[s][c] > 1e-12))
            });
            if !possible {
                return None;
            }
            let m_r: Vec<f64> = (0..np).map(|k| if reactive[self.group_of[k]] { self.mass[k][c] } else { 0.0 }).collect();
            let e_r: f64 = (0..self.groups.len()).filter(|&g| reactive[g]).map(|g| self.eg[g][c]).sum();
            let mut cc = CellChem { phases, rx, m: m_r, e: e_r, f_eff: self.flame_factor[c].max(1.0), scale: scale.to_vec(), surf, p: self.p[c], mix: mix[c] };
            let t0 = cc.temp();
            // cellule « gelée » ou déjà à l'équilibre ? (aucune réaction n'avancerait de 1e-7 de sa masse disponible pendant dt)
            let active = (0..nre).any(|j| {
                let (lo, hi) = cc.bounds(j);
                let r = cc.rate(j, t0);
                (r > 0.0 && r * dt > 1e-7 * hi) || (r < 0.0 && -r * dt > 1e-7 * (-lo))
            });
            if !active {
                return None;
            }
            let p = self.p[c];
            let v0 = cc.volume(p, t0);
            let m_old = cc.m.clone();
            let mut xi_tot = vec![0.0; nre];
            let mut tl = 0.0;
            let mut nsub = 0;
            let mut stopped = false;
            let mut max_h = f64::INFINITY;
            while tl < dt * (1.0 - 1e-12) && nsub < MAX_CHEM_SUBSTEPS {
                nsub += 1;
                let t = cc.temp();
                let cap = cc.cap(t).max(1e-30);
                let r0: Vec<f64> = (0..nre).map(|j| cc.rate(j, t)).collect();
                // pas : échauffement ≤ 25 K et consommation ≤ 30 % (réactions irréversibles ; les réversibles sont implicites)
                let mut h = (dt - tl).min(max_h);
                let mut dtdt = 0.0;
                let mut cons = vec![0.0; np];
                for j in 0..nre {
                    if rx[j].reversible && surf[j].is_none() {
                        continue;
                    }
                    // réaction dont tout l'avancement possible chaufferait de moins de TRACE_DT : trace (réactif résiduel, mélange
                    // diffusé), consommée sur le sous-pas sans le borner — sinon une trace de CH4 (ordre négatif plafonné à C_FLOOR,
                    // débit « normal ») découpait le pas en milliers de sous-pas de consommation ≤ 40 %
                    if cc.bounds(j).1 * cc.heat(j, t).abs() / cap < TRACE_DT {
                        continue;
                    }
                    dtdt += r0[j] * cc.heat(j, t) / cap;
                    for &(s, nu) in &rx[j].reactants {
                        cons[s] += nu * phases[s].molar_mass * r0[j].max(0.0);
                    }
                }
                if dtdt.abs() > 0.0 {
                    h = h.min(60.0 / dtdt.abs());
                }
                for s in 0..np {
                    if cons[s] > 0.0 && cc.m[s] > 1e-12 {
                        h = h.min(0.4 * cc.m[s] / cons[s]);
                    }
                }
                h = h.max(dt * 1e-9);
                let m_sub = cc.m.clone();
                let xi_sub = xi_tot.clone();
                for j in 0..nre {
                    let (lo, hi) = cc.bounds(j);
                    if hi <= 0.0 && lo >= 0.0 {
                        continue;
                    }
                    let xi = if rx[j].reversible && surf[j].is_none() {
                        // Euler implicite : ξ = h·r(ξ), Newton avec dérivée numérique (T recalculée : rétroaction thermique comprise)
                        let base = cc.m.clone();
                        let eval = |cc: &mut CellChem, x: f64| -> f64 {
                            cc.m.copy_from_slice(&base);
                            cc.apply(j, x);
                            let tt = cc.temp();
                            cc.rate(j, tt)
                        };
                        let mut x = clampf(h * r0[j], 0.999 * lo, 0.999 * hi);
                        for _ in 0..6 {
                            let rx_ = eval(&mut cc, x);
                            let g = x - h * rx_;
                            let d = 1e-7 * (hi - lo).abs().max(1e-12) + 1e-14;
                            let xd = if x + d < hi { x + d } else { x - d };
                            let rd = eval(&mut cc, xd);
                            let dr = (rd - rx_) / (xd - x);
                            let jac = 1.0 - h * dr;
                            let xn = if jac > 0.05 { x - g / jac } else { h * rx_ };
                            let xn = clampf(xn, 0.999 * lo, 0.999 * hi);
                            let conv = (xn - x).abs() <= 1e-9 * (hi - lo).abs().max(1e-30);
                            x = xn;
                            if conv {
                                break;
                            }
                        }
                        cc.m.copy_from_slice(&base);
                        x
                    } else {
                        // point milieu (Runge–Kutta 2) : vitesse évaluée à mi-pas (composition et T) → erreur O(h³) par sous-pas
                        let tt = cc.temp();
                        let r1 = cc.rate(j, tt);
                        let half = clampf(0.5 * h * r1, 0.0, 0.999 * hi);
                        if half > 0.0 {
                            let base = cc.m.clone();
                            cc.apply(j, half);
                            let th = cc.temp();
                            let rh = cc.rate(j, th);
                            cc.m.copy_from_slice(&base);
                            clampf(h * rh, 0.0, 0.999 * hi)
                        } else {
                            0.0
                        }
                    };
                    if xi != 0.0 {
                        cc.apply(j, xi);
                        xi_tot[j] += xi;
                    }
                }
                // sous-pas rejeté s'il échauffe de plus de 120 K (réactions réversibles implicites très exothermiques loin de l'équilibre)
                let dtemp = (cc.temp() - t).abs();
                if dtemp > 120.0 && h > dt * 1e-7 && nsub < MAX_CHEM_SUBSTEPS {
                    cc.m.copy_from_slice(&m_sub);
                    xi_tot.copy_from_slice(&xi_sub);
                    let h2 = h * (0.5 * 120.0 / dtemp).min(0.5);
                    // refait le sous-pas avec h2 : on borne la durée restante par h2 au prochain tour
                    max_h = h2;
                    continue;
                }
                max_h = f64::INFINITY;
                tl += h;
                // couplage avec l'écoulement : expansion maximale permise pendant le pas
                let t1 = cc.temp();
                if cc.volume(p, t1) - v0 > self.chem_volume_limit * v0 {
                    stopped = true;
                    break;
                }
            }
            // bilan : chaleur dégagée (= baisse de l'énergie chimique Σ m e0, e inchangée)
            let q: f64 = (0..np).map(|k| -(cc.m[k] - m_old[k]) * phases[k].e0).sum();
            let t_new = cc.temp();
            // pas de temps suivant : expansion visée `target` par pas
            let dv = (cc.volume(p, t_new) - v0).abs() / v0.max(1e-300);
            let dt_cell = if dv > 1e-3 { target * if stopped { tl } else { dt } / dv } else { f64::INFINITY };
            Some(ChemOut { c, m: cc.m, temp: t_new, q, xi: xi_tot, dt_cell })
        }
    }

    // ================================================================ rayonnement
    /// Pertes radiatives des gaz chauds (optiquement mince) : e −= dt · 4σ κ_P (T⁴ − T_amb⁴) · α_gaz, κ_P = Σ p_k[atm] a_P,k(T).
    pub(crate) fn radiate(&mut self, dt: f64) {
        if !self.radiation {
            return;
        }
        let np = self.reg.phases.len();
        let rads: Vec<usize> = (0..np).filter(|&k| self.reg.phases[k].radiator != Radiator::None && self.reg.phases[k].is_gas()).collect();
        if rads.is_empty() {
            return;
        }
        let ta = self.t_ambient;
        let a2 = self.dx * self.dx;
        // chaque groupe gazeux contenant un émetteur rayonne à SA température (fraction molaire des émetteurs dans ce groupe)
        let rad_groups: Vec<usize> = (0..self.groups.len()).filter(|&g| rads.iter().any(|&k| self.group_of[k] == g)).collect();
        for c in 0..self.nx * self.ny {
            if self.is_solid(c) || self.pinned(c) {
                continue;
            }
            let mut touched = false;
            for &g in &rad_groups {
                let t = self.tg[g][c];
                if (t - ta).abs() < 50.0 || self.group_mass(g, c) <= 0.0 {
                    continue;
                }
                let members = &self.groups[g];
                let ngas: f64 = members.iter().filter(|&&k| self.reg.phases[k].is_gas()).map(|&k| self.mass[k][c].max(0.0) / self.reg.phases[k].molar_mass).sum();
                if ngas <= 0.0 {
                    continue;
                }
                let a_gas: f64 = members.iter().filter(|&&k| self.reg.phases[k].is_gas()).map(|&k| self.alpha[k][c]).sum();
                let p_atm = self.p[c] / 101325.0;
                let kappa: f64 = rads
                    .iter()
                    .filter(|&&k| self.group_of[k] == g)
                    .map(|&k| (self.mass[k][c].max(0.0) / self.reg.phases[k].molar_mass / ngas) * p_atm * self.reg.phases[k].radiator.planck_mean(t))
                    .sum();
                let q = 4.0 * SIGMA_SB * kappa * (t.powi(4) - ta.powi(4)) * a_gas;
                // jamais plus de la moitié de l'écart à l'ambiante en un pas (stabilité)
                let lim = 0.5 * self.group_cap_at(g, c, t) * (t - ta).abs();
                let de = clampf(q * dt, -lim, lim);
                self.eg[g][c] -= de;
                self.radiated += de * a2;
                touched = true;
            }
            if touched {
                self.cell_sync(c);
            }
        }
    }
}
