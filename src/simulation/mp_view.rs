//! Couche « application » du solveur multiphasique, SANS dépendance egui (testable en `cargo test`) :
//! scénarios prédéfinis, outils de pinceau (ajout de phase / solide / chauffage / poussée), rendu RGBA, statistiques, infos de cellule.
//! La fenêtre egui (`window/multiphase_window.rs`) n'est qu'une fine couche par-dessus.
//!
//! Registre commun à tous les scénarios : phases 0 = air, 1 = eau, 2 = vapeur d'eau, 3 = huile ; solides 0 = acier, 1 = verre.

use serde::{Deserialize, Serialize};

use super::mp_cells::{MpEffect, MpInlet, MpThermal};
use super::mp_materials::{MaterialLibrary, Roles, AIR_SEC, CH4, H2O_G, HOUILLE, MIX_CH4_AIR, MIX_H2_AIR};
use super::multiphase::{Bc, MultiphaseSim};
use super::phases::{Phase, Registry};

pub const AIR: usize = 0;
pub const EAU: usize = 1;
pub const VAPEUR: usize = 2;
pub const HUILE: usize = 3;
pub const ACIER: usize = 0;
pub const VERRE: usize = 1;
const P_ATM: f64 = 101325.0;
/// température conseillée pour l'outil « Allumer » (K) : celle d'un noyau de gaz brûlés, proche de la température de flamme
pub const T_IGNITION: f64 = 2200.0;
/// noyau d'allumage des scénarios : gaz BRÛLÉS (combustion complète) à cette température, proche de la température de flamme —
/// la flamme démarre sans l'explosion parasite d'un noyau de gaz frais surchauffé
pub const T_KERNEL: f64 = 2200.0;

/// Registre des matériaux PAR DÉFAUT (air, eau, vapeur, huile / acier, verre) : les constantes AIR, EAU… ci-dessus en sont les indices.
/// Les matériaux de l'utilisateur viennent de `MaterialLibrary` (voir `build_scenario_with`).
pub fn registry() -> Registry {
    MaterialLibrary::default().build().expect("bibliothèque par défaut valide").registry
}

// ------------------------------------------------------------------------------------------------ scénarios
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MpScenario {
    DamBreak,
    DropSplash,
    Sloshing,
    Boiling,
    OilWaterAir,
    Empty,
    PremixedFlame,
    ConfinedExplosion,
    VentedExplosion,
    DiffusionFlame,
    /// nuage de sable dans l'air qui tombe dans de l'eau et s'y dépose (phase granulaire)
    Sedimentation,
    /// colonne de sable sec tassé qui s'effondre en un tas (frottement des grains)
    SandColumn,
    /// braséro : charbon en morceaux sur une grille, allumé par le bas, tirage naturel à travers le lit, flamme de CO au-dessus
    CoalFire,
    /// feu de bois : bûchettes sur une grille allumées par des braises ; pyrolyse, flammes des gaz du bois, charbon de bois qui brûle
    WoodFire,
    /// tas de poudre noire allumé sur un flanc : combustion sans air, gaz chauds qui traversent la poudre voisine, fumée blanche
    Gunpowder,
    /// nuage de glace pilée dans l'eau tiède : les cristaux remontent en fondant (chaleur latente), l'eau de fonte froide coule
    IceMelting,
    /// eau très pure surfondue à −8 °C et un germe de glace : le front de dendrites traverse l'eau, qui gèle d'un coup en neige fondante
    SupercooledWater,
    /// lit de sable grossier soufflé par une grille : au-delà de la vitesse minimale de fluidisation, il bouillonne
    FluidizedBed,
}

impl Default for MpScenario {
    fn default() -> Self {
        MpScenario::DamBreak
    }
}

impl MpScenario {
    pub const ALL: [MpScenario; 18] = [
        Self::DamBreak,
        Self::DropSplash,
        Self::Sloshing,
        Self::Boiling,
        Self::OilWaterAir,
        Self::Empty,
        Self::PremixedFlame,
        Self::ConfinedExplosion,
        Self::VentedExplosion,
        Self::DiffusionFlame,
        Self::Sedimentation,
        Self::SandColumn,
        Self::CoalFire,
        Self::WoodFire,
        Self::Gunpowder,
        Self::IceMelting,
        Self::SupercooledWater,
        Self::FluidizedBed,
    ];
    /// scénario de combustion : construit avec le préréglage combustion (O2, N2, CH4, H2, CO, CO2) et l'air sec comme ambiant
    /// scénario à grains : la bibliothèque reçoit une phase granulaire (le sable) si elle n'en a pas
    pub fn is_granular(self) -> bool {
        matches!(
            self,
            Self::Sedimentation
                | Self::SandColumn
                | Self::CoalFire
                | Self::WoodFire
                | Self::Gunpowder
                | Self::IceMelting
                | Self::SupercooledWater
                | Self::FluidizedBed
        )
    }
    pub fn is_combustion(self) -> bool {
        matches!(
            self,
            Self::PremixedFlame | Self::ConfinedExplosion | Self::VentedExplosion | Self::DiffusionFlame | Self::CoalFire | Self::WoodFire | Self::Gunpowder
        )
    }
    pub fn from_index(i: usize) -> Self {
        Self::ALL[i.min(Self::ALL.len() - 1)]
    }
    pub fn index(self) -> usize {
        Self::ALL.iter().position(|&m| m == self).unwrap()
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::DamBreak => "Rupture de barrage",
            Self::DropSplash => "Goutte et éclaboussure",
            Self::Sloshing => "Ballottement (vagues)",
            Self::Boiling => "Ébullition sur plaque chauffante",
            Self::OilWaterAir => "Huile / eau / air (goutte d'huile qui monte)",
            Self::Empty => "Cuve vide (à peindre)",
            Self::PremixedFlame => "Flamme de prémélange dans un tube (méthane–air)",
            Self::ConfinedExplosion => "Explosion confinée (hydrogène–air)",
            Self::VentedExplosion => "Explosion de gaz avec évent (méthane–air)",
            Self::DiffusionFlame => "Flamme de diffusion (jet de méthane)",
            Self::Sedimentation => "Sable qui tombe dans l'eau (sédimentation)",
            Self::SandColumn => "Effondrement d'une colonne de sable (frottement)",
            Self::CoalFire => "Feu de charbon sur grille (braséro : houille sur braises, flamme des matières volatiles)",
            Self::WoodFire => "Feu de bois sur grille (pyrolyse : flammes des gaz du bois)",
            Self::Gunpowder => "Tas de poudre noire (déflagration sans air, fumée blanche)",
            Self::IceMelting => "Glace pilée dans l'eau tiède (fusion, chaleur latente)",
            Self::SupercooledWater => "Eau surfondue qui gèle d'un coup (germe de glace, front de dendrites)",
            Self::FluidizedBed => "Lit fluidisé (sable grossier soufflé par une grille)",
        }
    }
}

/// Construit un scénario avec les matériaux PAR DÉFAUT (voir `build_scenario_with` pour une bibliothèque personnalisée).
pub fn build_scenario(sc: MpScenario, nx: usize, ny: usize, dx: f64, gravity: f64) -> MultiphaseSim {
    build_scenario_with(&MaterialLibrary::default(), sc, nx, ny, dx, gravity).expect("scénario avec les matériaux par défaut")
}

/// Construit un scénario sur une grille nx×ny de maille dx avec la bibliothèque de matériaux de l'utilisateur. Les formes sont exprimées en
/// fractions du domaine (valable à toute résolution). Les rôles (eau, huile, plaque…) sont retrouvés par nom puis par propriété ; si un
/// matériau nécessaire manque, l'erreur dit lequel.
pub fn build_scenario_with(lib: &MaterialLibrary, sc: MpScenario, nx: usize, ny: usize, dx: f64, gravity: f64) -> Result<MultiphaseSim, String> {
    let lib_c;
    let lib = if sc == MpScenario::CoalFire {
        lib_c = lib.with_coal();
        &lib_c
    } else if sc == MpScenario::WoodFire {
        lib_c = lib.with_wood();
        &lib_c
    } else if sc == MpScenario::Gunpowder {
        lib_c = lib.with_gunpowder();
        &lib_c
    } else if sc == MpScenario::IceMelting {
        lib_c = lib.with_ice();
        &lib_c
    } else if sc == MpScenario::SupercooledWater {
        lib_c = lib.with_supercooled_water();
        &lib_c
    } else if sc == MpScenario::FluidizedBed {
        lib_c = lib.with_coarse_sand();
        &lib_c

    } else if sc.is_combustion() {
        lib_c = lib.with_combustion();
        &lib_c
    } else if sc.is_granular() {
        lib_c = lib.with_granular();
        &lib_c
    } else {
        lib
    };
    let built = lib.build()?;
    let roles = Roles::resolve(&built.registry, built.ambient);
    let need = |o: Option<usize>, what: &str| -> Result<usize, String> { o.ok_or_else(|| format!("le scénario « {} » nécessite {what} (ajoutez-le dans les matériaux)", sc.name())) };
    let mix = |name: &str| -> Result<Vec<f64>, String> {
        built.mixtures.iter().find(|(n, _)| n == name).map(|(_, a)| a.clone()).ok_or_else(|| format!("le scénario « {} » nécessite le mélange « {name} »", sc.name()))
    };
    let (lx, ly) = (nx as f64 * dx, ny as f64 * dx);
    let mixtures = built.mixtures.clone();
    let ambient_mix = built.ambient_mix.clone();
    let mut s = MultiphaseSim::new(nx, ny, dx, built.registry, built.ambient);
    s.ambient_mix = ambient_mix;
    s.mixtures = mixtures;
    s.gravity = [0.0, -gravity];
    // noyau d'allumage (gaz brûlés à T_KERNEL) : ≈ 2 épaisseurs de flamme épaissie, sinon la diffusion l'éteint
    let r_kernel = (2.0 * (s.flame_resolution + 2.0) * dx).min(0.12 * lx.min(ly)).max((s.flame_resolution + 2.0) * dx);
    let burnt = |a: &Vec<f64>| super::mp_species::complete_combustion(&s.reg, a);
    let mut t_fn: Box<dyn Fn(f64, f64) -> f64> = Box::new(|_, _| 293.15);
    match sc {
        MpScenario::Sedimentation => {
            // eau en bas ; au-dessus, un nuage de sable dans l'air (30 % de grains) qui tombe, traverse la surface et se dépose au fond
            let eau = need(roles.water, "un liquide")?;
            let air = need(roles.air, "un gaz")?;
            let sable = need(roles.grains, "une phase granulaire (sable)")?;
            let mut cloud = vec![0.0; s.reg.phases.len()];
            cloud[air] = 0.7;
            cloud[sable] = 0.3;
            let (x0, x1, y0, y1) = (0.3 * lx, 0.7 * lx, 0.6 * ly, 0.85 * ly);
            s.paint_phase(move |_, y| if y < 0.4 * ly { Some(eau) } else { None });
            s.paint_mixture(move |x, y| if x > x0 && x < x1 && y > y0 && y < y1 { Some(cloud.clone()) } else { None });
        }
        MpScenario::IceMelting => {
            // eau tiède (27 °C) dans les 60 % du bas d'une cuve fermée, air au-dessus ; au fond, au milieu, un nuage de glace pilée
            // (cristaux de 2 mm à −10 °C, 20 % du volume) : plus légers que l'eau, les cristaux remontent en fondant — la chaleur de l'eau
            // arrive par leur surface (Nu = 2) : réchauffés à 0 °C en ~1 s, fondus en ~6 s (chaleur latente 333.6 kJ/kg) ; l'eau de fonte, à
            // 0 °C et plus dense, coule vers le fond
            let eau = need(roles.water, "un liquide (eau)")?;
            let air = need(roles.air, "un gaz")?;
            let glace = need(roles.ice, "une phase granulaire qui fond (glace) et sa paire de fusion")?;
            let mut cloud = vec![0.0; s.reg.phases.len()];
            cloud[eau] = 0.8;
            cloud[glace] = 0.2;
            let (x0, x1, y0, y1) = (0.35 * lx, 0.65 * lx, 0.08 * ly, 0.3 * ly);
            let h_w = 0.6 * ly;
            let in_cloud = move |x: f64, y: f64| x > x0 && x < x1 && y > y0 && y < y1;
            s.paint_phase(move |_, y| Some(if y < h_w { eau } else { air }));
            s.paint_mixture(move |x, y| if in_cloud(x, y) { Some(cloud.clone()) } else { None });
            t_fn = Box::new(|_, _| 300.15);
            s.finalize_init(&*t_fn);
            let gi = s.group_of[glace];
            for j in 0..ny {
                for i in 0..nx {
                    let (x, y) = s.center(i, j);
                    if in_cloud(x, y) {
                        let c = s.idx(i, j);
                        s.set_group_temperature(c, gi, 263.15);
                    }
                }
            }
            s.refresh();
            return Ok(s);
        }
        MpScenario::FluidizedBed => {
            // lit de sable grossier (grains de 2 mm, tassé sur le quart de la hauteur) posé sur une grille (trous de 2 mm) au fond de la
            // cuve ; air soufflé par le fond à 1.4 m/s, 1.3 fois la vitesse minimale de fluidisation (Ergun : 1.08 m/s) ; haut ouvert. Sous
            // ~1 m/s, le lit reste posé : l'air traverse ses pores et porte une part de son poids (contacts déchargés). Au-delà, ils ne portent
            // plus rien : le lit se soulève et bouillonne (vitesse réglable : entrée du bord du bas)
            let air = need(roles.air, "un gaz")?;
            let sg = need(built_index(&s, super::mp_materials::SABLE_GROSSIER), "du sable grossier (grains de 2 mm)")?;
            let amax = s.reg.phases[sg].alpha_max;
            let mut packed = vec![0.0; s.reg.phases.len()];
            packed[air] = 1.0 - amax;
            packed[sg] = amax;
            let (y0, y1) = (3.0 * dx, 3.0 * dx + 0.25 * ly);
            s.paint_mixture(move |_, y| if y > y0 && y < y1 { Some(packed.clone()) } else { None });
            for i in 0..nx {
                let c = s.idx(i, 2);
                s.set_effect(c, Some(MpEffect::Grille { hole: 0.002, open: 0.4 }));
            }
            s.inlet[2] = Some(MpInlet { vx: 0.0, vy: 1.4, phase: air, temp: 293.15, offset: 0 });
            s.bc[2] = Bc::Wall;
            s.bc[3] = Bc::Open { p: P_ATM };
        }
        MpScenario::SupercooledWater => {
            // eau très pure surfondue à −8 °C (elle ne gèlerait seule que sous −20 °C) dans les 70 % du bas d'une cuve fermée, air à −8 °C
            // au-dessus ; à la surface, au milieu, quelques cristaux de glace (5 % : le germe qu'on y laisse tomber). Le front de
            // dendrites part du germe et traverse l'eau à v(8 K) ≈ 5 cm/s ; derrière lui, l'eau gèle d'un coup jusqu'à 0 °C en laissant
            // 10 % de glace (c ΔT / L_f) : une neige fondante, plus légère que l'eau surfondue, qui remonte. Peindre de la glace ailleurs
            // y lance un autre front.
            let eau = need(roles.water, "un liquide (eau)")?;
            let air = need(roles.air, "un gaz")?;
            let glace = need(roles.ice, "une phase granulaire qui gèle (glace) et sa paire de fusion")?;
            let h_w = 0.7 * ly;
            let r = (0.02 * lx).max(1.5 * dx);
            let (xs, ys) = (0.5 * lx, h_w - r);
            let mut germ = vec![0.0; s.reg.phases.len()];
            germ[eau] = 0.95;
            germ[glace] = 0.05;
            s.paint_phase(move |_, y| Some(if y < h_w { eau } else { air }));
            s.paint_mixture(move |x, y| if (x - xs).hypot(y - ys) < r { Some(germ.clone()) } else { None });
            t_fn = Box::new(|_, _| 265.15);
            s.finalize_init(&*t_fn);
            s.refresh();
            return Ok(s);
        }
        MpScenario::SandColumn => {
            // colonne de sable sec tassé (α_max de grains, le reste d'air) au milieu : elle s'effondre en un tas dont les flancs
            // tiennent à l'angle de frottement (≈ 25° pour un effondrement, ≈ φ pour un tas qui s'affaisse lentement)
            let air = need(roles.air, "un gaz")?;
            let sable = need(roles.grains, "une phase granulaire (sable)")?;
            let amax = s.reg.phases[sable].alpha_max;
            let mut packed = vec![0.0; s.reg.phases.len()];
            packed[air] = 1.0 - amax;
            packed[sable] = amax;
            let (x0, x1, h) = (0.42 * lx, 0.58 * lx, 0.65 * ly);
            s.paint_mixture(move |x, y| if x > x0 && x < x1 && y < h { Some(packed.clone()) } else { None });
        }
        MpScenario::CoalFire => {
            // braséro : grille d'acier (trous 1 cm, 60 % ouverte) entre deux parois, chargée d'un lit de charbon en MORCEAUX (1.2 cm :
            // le lit respire) de 0.25 ly, entièrement en braises (1450 K, feu établi de forge). L'air frais entre par le haut ouvert, descend le
            // long des parois, passe sous la grille et traverse le lit par percolation (tirage naturel : le gaz chaud du lit est léger) ;
            // il brûle à la surface des morceaux (C + ½O2 → CO surtout, C + O2 → CO2 ; zone d'oxydation au-dessus de la grille), le CO
            // brûle dans les pores tant qu'il y reste de l'oxygène ; plus haut, sans oxygène, le CO2 est réduit en CO (Boudouard), qui
            // brûle au-dessus des braises au contact de l'air (petite flamme bleue ; l'humidité de l'air, 1 %, catalyse son oxydation).
            // Sur les braises (60 % du bas du lit), une charge de houille en noix (3 cm, 40 % du haut) déjà chaude (700 K, feu établi) :
            // elle dégage ses matières volatiles (surtout du méthane, ~0.8 %/s de sa masse : la chaleur met ~2 min à gagner le cœur d'un
            // morceau), qui sortent du lit sans oxygène et brûlent en grande flamme (~2200 K) au contact de l'air, au-dessus du lit
            // (allumée par une flamme d'allumage : gaz à 1600 K sur 4 cm au-dessus du lit) ; elle laisse son char, qui devient braise à son
            // tour. Une charge FROIDE fume : ses gaz, refroidis en la traversant, sortent sous le seuil d'allumage et étouffent les braises.
            // Sans houille dans la bibliothèque : tout le lit en braises
            let coal = need(roles.coal, "une phase granulaire combustible (charbon) et sa réaction de surface")?;
            let fresh = built_index(&s, HOUILLE);
            let plate = need(roles.plate, "un solide (parois et grille du braséro)")?;
            s.reg.phases[coal].grain_diameter = s.reg.phases[coal].grain_diameter.max(0.012);
            if let Some(h) = fresh {
                s.reg.phases[h].grain_diameter = s.reg.phases[h].grain_diameter.max(0.03);
                // flamme de diffusion des matières volatiles : épaississement modéré, comme le feu de bois
                s.flame_resolution = 0.25;
            }
            let dry = mix(AIR_SEC)?;
            let h2o = built_index(&s, H2O_G);
            let mut air = dry.iter().map(|a| a * if h2o.is_some() { 0.99 } else { 1.0 }).collect::<Vec<f64>>();
            if let Some(w) = h2o {
                air[w] += 0.01;
            }
            s.ambient_mix = air.clone();
            let amax = s.reg.phases[coal].alpha_max;
            let mut bed = air.iter().map(|a| a * (1.0 - amax)).collect::<Vec<f64>>();
            bed[coal] = amax;
            let load = fresh.map(|h| {
                let ah = s.reg.phases[h].alpha_max;
                let mut m = air.iter().map(|a| a * (1.0 - ah)).collect::<Vec<f64>>();
                m[h] = ah;
                m
            });
            let wall = (0.025 * lx).max(dx);
            let (x0, x1) = (0.3 * lx, 0.7 * lx);
            let yg = 0.15 * ly;
            let bed_h = 0.25;
            let (yb0, yb1) = (yg + dx, yg + dx + bed_h * ly);
            let ytop = yg + (0.3f64).max(bed_h + 0.1) * ly;
            let grate = move |x: f64, y: f64| x > x0 && x < x1 && y >= yg && y < yg + dx;
            let in_bed = move |x: f64, y: f64| x > x0 && x < x1 && y >= yb0 && y < yb1;
            let walls = move |x: f64, y: f64| y >= yg && y < ytop && ((x > x0 - wall && x <= x0) || (x >= x1 && x < x1 + wall));
            let y_load = if load.is_some() { yb0 + 0.6 * (yb1 - yb0) } else { yb1 };
            let on_top = move |x: f64, y: f64| in_bed(x, y) && y >= y_load;
            s.paint_mixture(move |x, y| {
                Some(match (&load, on_top(x, y)) {
                    (Some(l), true) => l.clone(),
                    _ if in_bed(x, y) => bed.clone(),
                    _ => air.clone(),
                })
            });
            s.paint_solid(move |x, y| if walls(x, y) { Some(plate) } else { None });
            s.bc[3] = Bc::Open { p: P_ATM };
            s.dt_max = 2e-3;
            // feu établi : braises à 1450 K, houille à 700 K, flamme d'allumage entre les parois au-dessus du lit
            let lit = fresh.is_some();
            let pilot = move |x: f64, y: f64| lit && x > x0 && x < x1 && y >= yb1 && y < yb1 + 0.04;
            t_fn = Box::new(move |x, y| {
                if on_top(x, y) { 700.0 } else if in_bed(x, y) { 1450.0 } else if pilot(x, y) { 1600.0 } else { 293.15 }
            });
            s.finalize_init(&*t_fn);
            for j in 0..ny {
                for i in 0..nx {
                    let (x, y) = s.center(i, j);
                    if grate(x, y) {
                        let c = s.idx(i, j);
                        s.set_effect(c, Some(MpEffect::Grille { hole: 0.01, open: 0.6 }));
                    }
                }
            }
            s.refresh();
            return Ok(s);
        }
        MpScenario::WoodFire => {
            // feu de bois : bûchettes de 2 cm sur un lit de braises de charbon de bois (le tiers du bas du tas, 1450 K), sur la grille du
            // braséro. Le bois chauffé par les braises se décompose (pyrolyse, ~1 s vers 900 K) : ses gaz (CO, CH4, H2, et la
            // vapeur et le CO2 qu'il libère) montent à travers le tas et brûlent en flamme dans l'air au-dessus ; le charbon de bois qu'il
            // laisse brûle ensuite à la surface des morceaux, comme les braises. L'air entre par le haut ouvert, descend le long des parois
            // et passe sous la grille (tirage naturel à travers le tas, percolation)
            let wood = need(roles.wood, "une phase granulaire qui se décompose en chauffant (bois) et sa pyrolyse")?;
            let charcoal = need(roles.charcoal, "le résidu granulaire de la pyrolyse (charbon de bois)")?;
            let plate = need(roles.plate, "un solide (parois et grille)")?;
            let dry = mix(AIR_SEC)?;
            let h2o = built_index(&s, H2O_G);
            let mut air = dry.iter().map(|a| a * if h2o.is_some() { 0.99 } else { 1.0 }).collect::<Vec<f64>>();
            if let Some(w) = h2o {
                air[w] += 0.01;
            }
            s.ambient_mix = air.clone();
            let (aw, ac) = (s.reg.phases[wood].alpha_max, s.reg.phases[charcoal].alpha_max);
            let mut logs = air.iter().map(|a| a * (1.0 - aw)).collect::<Vec<f64>>();
            logs[wood] = aw;
            let mut embers = air.iter().map(|a| a * (1.0 - ac)).collect::<Vec<f64>>();
            embers[charcoal] = ac;
            let wall = (0.025 * lx).max(dx);
            let (x0, x1) = (0.3 * lx, 0.7 * lx);
            let yg = 0.15 * ly;
            let bed_h = 0.25;
            let (yb0, yb1) = (yg + dx, yg + dx + bed_h * ly);
            let y_emb = yb0 + (yb1 - yb0) / 3.0;
            // parois basses (elles retiennent le tas) : au-dessus, le panache aspire l'air par les côtés et la flamme se forme sur ses bords
            let ytop = yb1;
            let grate = move |x: f64, y: f64| x > x0 && x < x1 && y >= yg && y < yg + dx;
            let in_bed = move |x: f64, y: f64| x > x0 && x < x1 && y >= yb0 && y < yb1;
            let walls = move |x: f64, y: f64| y >= yg && y < ytop && ((x > x0 - wall && x <= x0) || (x >= x1 && x < x1 + wall));
            s.paint_mixture(move |x, y| {
                Some(if in_bed(x, y) && y < y_emb { embers.clone() } else if in_bed(x, y) { logs.clone() } else { air.clone() })
            });
            s.paint_solid(move |x, y| if walls(x, y) { Some(plate) } else { None });
            s.bc[3] = Bc::Open { p: P_ATM };
            s.dt_max = 2e-3;
            // flammes de DIFFUSION des gaz du bois (le mélange règle la combustion) : épaississement MODÉRÉ (facteur ~5 à 1 cm de maille).
            // Complet (~100, fait pour la vitesse d'une flamme de prémélange), il divise les vitesses de réaction au point d'éteindre la
            // flamme ; nul, une poche de gaz du bois et d'air brûle d'un coup dans une maille, qui « pompe » (volume ±50 % d'un pas à
            // l'autre) et dont le reste de gaz, comprimé, monte à 2600–3800 K
            s.flame_resolution = 0.25;
            // feu établi : braises à 1450 K, bûches déjà chaudes (750 K) qui dégagent leurs gaz, et une flamme d'allumage (gaz à 1600 K
            // sur 4 cm au-dessus du tas) qui les enflamme au contact de l'air. Du bois FROID posé sur les braises chauffe vite (rayonnement
            // dans le tas : 1000 K au contact en 4 s), mais une telle charge (deux tiers du tas, sans cheminée : tirage faible) étouffe les
            // braises avant que ses gaz ne s'enflamment — O2 du tas à 1 %, braises à 1250 K au bout de 14 s, comme un feu trop chargé
            let pilot = move |x: f64, y: f64| x > x0 - 0.02 && x < x1 + 0.02 && y >= yb1 && y < yb1 + 0.04;
            t_fn = Box::new(move |x, y| {
                if in_bed(x, y) && y < y_emb { 1450.0 } else if in_bed(x, y) { 750.0 } else if pilot(x, y) { 1600.0 } else { 293.15 }
            });
            s.finalize_init(&*t_fn);
            for j in 0..ny {
                for i in 0..nx {
                    let (x, y) = s.center(i, j);
                    if grate(x, y) {
                        let c = s.idx(i, j);
                        s.set_effect(c, Some(MpEffect::Grille { hole: 0.01, open: 0.6 }));
                    }
                }
            }
            s.refresh();
            return Ok(s);
        }
        MpScenario::DamBreak => {
            let eau = need(roles.water, "un liquide")?;
            s.paint_phase(move |x, y| if x < 0.25 * lx && y < 0.5 * ly { Some(eau) } else { None });
        }
        MpScenario::DropSplash => {
            let eau = need(roles.water, "un liquide")?;
            let (cx, cy, r) = (0.5 * lx, 0.62 * ly, 0.09 * lx.min(ly));
            s.paint_phase(move |x, y| {
                if y < 0.22 * ly || (x - cx).powi(2) + (y - cy).powi(2) < r * r { Some(eau) } else { None }
            });
        }
        MpScenario::Sloshing => {
            let eau = need(roles.water, "un liquide")?;
            s.paint_phase(move |x, y| if y < 0.4 * ly + 0.04 * ly * (std::f64::consts::PI * x / lx).cos() { Some(eau) } else { None });
        }
        MpScenario::Boiling => {
            // plaque d'acier chaude (2 cellules) sous 40 % d'eau à 90 °C ; haut ouvert (atmosphère)
            let eau = need(roles.water, "un liquide")?;
            let plate = need(roles.plate, "un solide")?;
            let h_plate = 2.0 * dx;
            s.paint_phase(move |_, y| if y < 0.45 * ly { Some(eau) } else { None });
            s.paint_solid(move |x, y| if y < h_plate && x > 0.1 * lx && x < 0.9 * lx { Some(plate) } else { None });
            s.bc[3] = Bc::Open { p: P_ATM };
            s.dt_max = 5e-4;
            t_fn = Box::new(move |x, y| if y < h_plate && x > 0.1 * lx && x < 0.9 * lx { 600.0 } else if y < 0.45 * ly { 363.15 } else { 293.15 });
        }
        MpScenario::OilWaterAir => {
            let eau = need(roles.water, "un premier liquide (eau)")?;
            let huile = need(roles.oil, "un second liquide (huile)")?;
            let (cx, cy, r) = (0.5 * lx, 0.15 * ly, 0.07 * lx.min(ly));
            s.paint_phase(move |x, y| {
                if (x - cx).powi(2) + (y - cy).powi(2) < r * r { Some(huile) } else if y < 0.4 * ly { Some(eau) } else { None }
            });
        }
        MpScenario::Empty => {
            s.bc[3] = Bc::Open { p: P_ATM };
        }
        MpScenario::PremixedFlame => {
            // tube rempli de méthane–air stœchiométrique, fermé à gauche, ouvert à droite ; allumage plan contre le fond fermé.
            // La flamme avance à σ·S_L ≈ 7.5 × 0.38 m/s ; les gaz brûlés, bloqués par le fond, chassent les gaz frais vers l'ouverture.
            let m = mix(MIX_CH4_AIR)?;
            let b = burnt(&m);
            let xk = r_kernel.max(0.03 * lx);
            s.paint_mixture(move |x, _| Some(if x < xk { b.clone() } else { m.clone() }));
            s.bc[1] = Bc::Open { p: P_ATM };
            s.dt_max = 2e-3;
            t_fn = Box::new(move |x, _| if x < xk { T_KERNEL } else { 293.15 });
        }
        MpScenario::ConfinedExplosion => {
            // boîte fermée d'hydrogène–air stœchiométrique, étincelle au centre : la pression monte vers ≈ 8 bar (combustion isochore)
            let m = mix(MIX_H2_AIR)?;
            let b = burnt(&m);
            let (cx, cy, r) = (0.5 * lx, 0.5 * ly, r_kernel.max(0.04 * lx.min(ly)));
            let ink = move |x: f64, y: f64| (x - cx).powi(2) + (y - cy).powi(2) < r * r;
            s.paint_mixture(move |x, y| Some(if ink(x, y) { b.clone() } else { m.clone() }));
            s.dt_max = 1e-3;
            t_fn = Box::new(move |x, y| if ink(x, y) { T_KERNEL } else { 293.15 });
        }
        MpScenario::VentedExplosion => {
            // pièce aux murs d'acier remplie de méthane–air, évent (ouverture) sur le mur de droite, allumage contre le mur du fond :
            // jet de gaz frais puis boule de feu à la sortie de l'évent. Domaine ouvert en haut et à droite (atmosphère).
            let plate = need(roles.plate, "un solide (murs de la pièce)")?;
            let m = mix(MIX_CH4_AIR)?;
            let w = 2.0 * dx;
            let (x0, x1, y0, y1) = (0.04 * lx, 0.55 * lx, 0.04 * ly, 0.55 * ly);
            let (v0, v1) = (y0 + 0.30 * (y1 - y0), y0 + 0.70 * (y1 - y0));
            let inside = move |x: f64, y: f64| x > x0 + w && x < x1 - w && y > y0 + w && y < y1 - w;
            let wall = move |x: f64, y: f64| {
                let in_box = x >= x0 && x <= x1 && y >= y0 && y <= y1;
                in_box && !inside(x, y) && !(x > x1 - w && y > v0 && y < v1)
            };
            let b = burnt(&m);
            let (cx, cy, r) = (x0 + w + r_kernel, 0.5 * (y0 + y1), r_kernel.max(0.03 * lx.min(ly)));
            let ink = move |x: f64, y: f64| (x - cx).powi(2) + (y - cy).powi(2) < r * r && inside(x, y);
            s.paint_mixture(move |x, y| if ink(x, y) { Some(b.clone()) } else if inside(x, y) { Some(m.clone()) } else { None });
            s.paint_solid(move |x, y| if wall(x, y) { Some(plate) } else { None });
            s.bc[1] = Bc::Open { p: P_ATM };
            s.bc[3] = Bc::Open { p: P_ATM };
            s.dt_max = 2e-3;
            t_fn = Box::new(move |x, y| if ink(x, y) { T_KERNEL } else { 293.15 });
        }
        MpScenario::Gunpowder => {
            // petit tas de poudre noire (grains de 2 mm en vrac, α = 0.55 ; flancs à 34°, l'angle de talus ; ~1 kg par mètre de
            // profondeur à 0.8 m × 0.6 m) posé au milieu du sol, allumé sur son flanc gauche (mailles à 700 K, au-dessus des 580 K
            // d'allumage). Les grains allumés brûlent sans air, en surface (loi de Vieille) ; leurs gaz chauds (CO2, CO, N2, eau, H2 : ~3000
            // fois le volume de la poudre) traversent le tas et allument les grains qu'ils baignent : le tas fuse en quelques dizaines de ms
            // et laisse un nuage de fumée blanche (les résidus solides, en particules de 2 µm) ; le CO et le H2 des produits finissent de
            // brûler dans l'air. Haut ouvert
            let powder = need(roles.powder, "une phase granulaire qui brûle sans air (poudre noire) et sa combustion")?;
            let dry = mix(AIR_SEC)?;
            let h2o = built_index(&s, H2O_G);
            let mut air = dry.iter().map(|a| a * if h2o.is_some() { 0.99 } else { 1.0 }).collect::<Vec<f64>>();
            if let Some(w) = h2o {
                air[w] += 0.01;
            }
            s.ambient_mix = air.clone();
            let ap = s.reg.phases[powder].alpha_max;
            let mut heap = air.iter().map(|a| a * (1.0 - ap)).collect::<Vec<f64>>();
            heap[powder] = ap;
            // (au moins 3 mailles de demi-largeur et 2 de haut : sur une grille grossière le tas garde quelques mailles, dont une allumée)
            let (xc, b) = (0.5 * lx, (0.05 * lx).max(3.0 * dx));
            let hh = (b * 34f64.to_radians().tan()).max(2.0 * dx);
            let in_heap = move |x: f64, y: f64| y < hh * (1.0 - (x - xc).abs() / b);
            let fuse = move |x: f64, y: f64| in_heap(x, y) && x < xc - 0.2 * b;
            s.paint_mixture(move |x, y| Some(if in_heap(x, y) { heap.clone() } else { air.clone() }));
            s.bc[3] = Bc::Open { p: P_ATM };
            s.dt_max = 1e-3;
            // combustion du CO et du H2 des produits dans l'air : flamme de diffusion, épaississement modéré (voir le feu de bois)
            s.flame_resolution = 0.25;
            t_fn = Box::new(move |x, y| if fuse(x, y) { 700.0 } else { 293.15 });
        }
        MpScenario::DiffusionFlame => {
            // buse de méthane au centre du fond (émetteurs, 0.25 m/s), air au repos, haut ouvert ; un noyau chaud de prémélange au-dessus
            // de la buse allume le jet : flamme de diffusion laminaire, panache de fumées chaudes porté par la flottabilité.
            let ch4 = built_index(&s, CH4).ok_or("le scénario nécessite le méthane")?;
            let m = mix(MIX_CH4_AIR)?;
            s.bc[3] = Bc::Open { p: P_ATM };
            s.dt_max = 2e-3;
            let half = ((0.025 * nx as f64).round() as usize).max(2);
            let ic = nx / 2;
            // flamme pilote : gaz brûlés chauds au-dessus de la buse
            let b = burnt(&m);
            let (cx, cy, r) = ((ic as f64) * dx, 2.0 * dx + r_kernel, r_kernel);
            let ink = move |x: f64, y: f64| (x - cx).powi(2) + (y - cy).powi(2) < r * r;
            s.paint_mixture(move |x, y| if ink(x, y) { Some(b.clone()) } else { None });
            t_fn = Box::new(move |x, y| if ink(x, y) { T_KERNEL } else { 293.15 });
            s.finalize_init(&*t_fn);
            for i in ic.saturating_sub(half)..(ic + half).min(nx) {
                let c = s.idx(i, 0);
                s.set_effect(c, Some(MpEffect::Emitter { vx: 0.0, vy: 0.25, phase: ch4, temp: 293.15 }));
            }
            return Ok(s);
        }
    }
    s.finalize_init(&*t_fn);
    Ok(s)
}

fn built_index(s: &MultiphaseSim, name: &str) -> Option<usize> {
    s.reg.phases.iter().position(|p| p.name == name)
}

// ------------------------------------------------------------------------------------------------ pinceaux
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MpTool {
    Phase(usize),
    Solid(usize),
    EraseSolid,
    Heat,
    Push,
    /// pose une cellule spéciale (puits, émetteur, accélérateur, téléporteur)
    Effect(MpEffect),
    /// efface solide, effet spécial et comportement thermique (la cellule redevient du fluide ambiant)
    ClearCell,
    /// remplit d'un mélange nommé (`sim.mixtures[i]`), ex. méthane–air stœchiométrique
    Mixture(usize),
    /// allumage : crée un noyau de flamme (produits de combustion complète du mélange local à la T du pinceau, à pression constante)
    Ignite,
}

pub fn tool_name(reg: &Registry, t: MpTool) -> String {
    match t {
        MpTool::Phase(k) => format!("Ajouter : {}", reg.phases[k].name),
        MpTool::Solid(s) => format!("Solide : {}", reg.solids[s].name),
        MpTool::EraseSolid => "Effacer le solide".into(),
        MpTool::Heat => "Chauffer / refroidir".into(),
        MpTool::Push => "Pousser le fluide".into(),
        MpTool::Effect(e) => format!("Cellule spéciale : {}", e.kind_name()),
        MpTool::ClearCell => "Effacer solide + effet".into(),
        MpTool::Mixture(i) => format!("Mélange : {}", mixtures_name(reg, i)),
        MpTool::Ignite => "Allumer (étincelle)".into(),
    }
}

fn mixtures_name(_reg: &Registry, i: usize) -> String {
    format!("n° {}", i + 1)
}

/// Nom affichable d'un outil, mélanges nommés compris (préférer à `tool_name` quand la simulation est disponible).
pub fn tool_label(sim: &MultiphaseSim, t: MpTool) -> String {
    match t {
        MpTool::Mixture(i) => format!("Mélange : {}", sim.mixtures.get(i).map_or("?", |m| m.0.as_str())),
        _ => tool_name(&sim.reg, t),
    }
}

/// Outils de pinceau d'une simulation : phases pures, mélanges nommés, solides, effacement, chauffage, poussée, allumage si réactions.
pub fn sim_tools(sim: &MultiphaseSim) -> Vec<MpTool> {
    let mut v: Vec<MpTool> = (0..sim.reg.phases.len()).map(MpTool::Phase).collect();
    v.extend((0..sim.mixtures.len()).map(MpTool::Mixture));
    v.extend((0..sim.reg.solids.len()).map(MpTool::Solid));
    v.push(MpTool::EraseSolid);
    v.push(MpTool::Heat);
    v.push(MpTool::Push);
    if !sim.reg.reactions.is_empty() {
        v.push(MpTool::Ignite);
    }
    v
}

pub fn all_tools(reg: &Registry) -> Vec<MpTool> {
    let mut v: Vec<MpTool> = (0..reg.phases.len()).map(MpTool::Phase).collect();
    v.extend((0..reg.solids.len()).map(MpTool::Solid));
    v.push(MpTool::EraseSolid);
    v.push(MpTool::Heat);
    v.push(MpTool::Push);
    v
}

/// Remplit la cellule c d'une seule phase pure à (p, T) en gardant la cohérence masse/énergie. Les faces voisines ne sont pas modifiées.
fn set_pure(sim: &mut MultiphaseSim, c: usize, k: usize, t: f64) {
    let p = sim.p[c];
    sim.solid[c] = -1;
    for j in 0..sim.reg.phases.len() {
        sim.mass[j][c] = if j == k { sim.reg.phases[j].rho(p, t) } else { 0.0 };
    }
    sim.set_cell_temperature(c, t);
}

/// Paramètres d'un coup de pinceau.
#[derive(Clone, Copy, Debug)]
pub struct ToolOpts {
    /// température de pinceau (K) : phases ajoutées, solides, cible du chauffage
    pub temp: f64,
    /// vitesse du curseur (m/s, y vers le haut) pour la poussée
    pub cursor: (f64, f64),
    /// durée de la trame (s) pour les outils à taux (chauffage, poussée)
    pub dt: f64,
    /// comportement thermique donné aux solides posés
    pub thermal: MpThermal,
}

/// Applique l'outil sur les cellules `cells` (i, j) ; `temp` = température de pinceau (K) ; `cursor` = vitesse du curseur (m/s, y vers le haut) ;
/// `dt` = durée de la trame (s) pour les outils à taux (chauffage, poussée). Solides posés : passifs (voir `apply_tool_with`).
pub fn apply_tool(sim: &mut MultiphaseSim, tool: MpTool, cells: &[(usize, usize)], temp: f64, cursor: (f64, f64), dt: f64) {
    apply_tool_with(sim, tool, cells, &ToolOpts { temp, cursor, dt, thermal: MpThermal::Passive });
}

pub fn apply_tool_with(sim: &mut MultiphaseSim, tool: MpTool, cells: &[(usize, usize)], o: &ToolOpts) {
    let (temp, cursor, dt) = (o.temp, o.cursor, o.dt);
    let (nx, ny) = (sim.nx, sim.ny);
    for &(i, j) in cells {
        if i >= nx || j >= ny {
            continue;
        }
        let c = sim.idx(i, j);
        match tool {
            MpTool::Phase(k) => {
                if !sim.is_solid(c) {
                    // un liquide au-dessus de sa température d'ébullition (à la pression locale) n'existe pas : on pose sa vapeur
                    let k = match sim.reg.change_for_liquid(k) {
                        Some(pc) if temp > pc.t_sat(sim.p[c]) => pc.vapor,
                        _ => k,
                    };
                    set_pure(sim, c, k, temp);
                }
            }
            MpTool::Solid(s) => {
                sim.solid[c] = s as i32;
                sim.special[c] = None;
                for k in 0..sim.reg.phases.len() {
                    sim.mass[k][c] = 0.0;
                    sim.alpha[k][c] = 0.0;
                }
                sim.set_cell_temperature(c, temp);
                sim.set_thermal_cell(c, o.thermal);
                let (a, b, cc, d) = (sim.uidx(i, j), sim.uidx(i + 1, j), sim.vidx(i, j), sim.vidx(i, j + 1));
                sim.u[a] = 0.0;
                sim.u[b] = 0.0;
                sim.v[cc] = 0.0;
                sim.v[d] = 0.0;
            }
            MpTool::EraseSolid => {
                if sim.is_solid(c) {
                    sim.thermal[c] = MpThermal::Passive;
                    let t = sim.t_ambient;
                    sim.solid[c] = -1;
                    sim.fill_ambient(c, t);
                }
            }
            MpTool::ClearCell => sim.clear_cell(c),
            MpTool::Effect(e) => {
                if let MpEffect::Teleporter { dest, .. } = e {
                    if dest == c {
                        continue;
                    }
                }
                sim.set_effect(c, Some(e));
            }
            MpTool::Mixture(m) => {
                if !sim.is_solid(c) {
                    if let Some((_, a)) = sim.mixtures.get(m).cloned() {
                        sim.fill_mixture(c, &a, temp);
                    }
                }
            }
            MpTool::Ignite => {
                // noyau de flamme : le mélange de la cellule est remplacé par ses produits de combustion complète à `temp`, À LA PRESSION
                // de la cellule (un chauffage à volume constant d'une boule de plusieurs cm se comporterait comme une charge explosive).
                // L'énergie apportée est comptée dans `energy_input`.
                if !sim.is_solid(c) && sim.temp[c] < temp {
                    let np = sim.reg.phases.len();
                    let a: Vec<f64> = (0..np).map(|k| sim.alpha[k][c]).collect();
                    let burnt = super::mp_species::complete_combustion(&sim.reg, &a);
                    let e_old = sim.e[c];
                    sim.fill_mixture(c, &burnt, temp);
                    sim.energy_input += (sim.e[c] - e_old) * sim.dx * sim.dx;
                }
            }
            MpTool::Heat => {
                // cible de chauffage : la simulation y tend pendant le temps SIMULÉ (constante de temps `heat_tau`), voir
                // `MultiphaseSim::relax_heat_targets` ; les solides sont chauffés directement (pas de dilatation)
                if sim.is_solid(c) {
                    let rate = 1.0 - (-dt / sim.heat_tau).exp();
                    let cap = sim.heat_capacity(c);
                    let t = sim.temp[c];
                    let de = (temp - t) * cap * rate;
                    sim.e[c] += de;
                    sim.energy_input += de * sim.dx * sim.dx;
                } else {
                    sim.set_heat_target(c, temp);
                }
            }
            MpTool::Push => {
                let rate = 1.0 - (-20.0 * dt).exp();
                let (a, b, cc, d) = (sim.uidx(i, j), sim.uidx(i + 1, j), sim.vidx(i, j), sim.vidx(i, j + 1));
                let open_u = |s: &MultiphaseSim, i2: usize, j2: usize| !s.is_solid(s.idx(i2.min(nx - 1), j2));
                if i > 0 && open_u(sim, i - 1, j) && !sim.is_solid(c) {
                    sim.u[a] += (cursor.0 - sim.u[a]) * rate;
                }
                if i + 1 < nx && !sim.is_solid(sim.idx(i + 1, j)) && !sim.is_solid(c) {
                    sim.u[b] += (cursor.0 - sim.u[b]) * rate;
                }
                if j > 0 && !sim.is_solid(sim.idx(i, j - 1)) && !sim.is_solid(c) {
                    sim.v[cc] += (cursor.1 - sim.v[cc]) * rate;
                }
                if j + 1 < ny && !sim.is_solid(sim.idx(i, j + 1)) && !sim.is_solid(c) {
                    sim.v[d] += (cursor.1 - sim.v[d]) * rate;
                }
            }
        }
    }
    sim.refresh();
}

/// Cellules d'un pinceau (rond ou carré) de rayon `r` (en cellules) centré en (ci, cj), bornées à la grille.
pub fn brush_cells(nx: usize, ny: usize, ci: usize, cj: usize, r: usize, circle: bool) -> Vec<(usize, usize)> {
    let mut v = vec![];
    let r = r as isize;
    for dj in -r..=r {
        for di in -r..=r {
            if circle && di * di + dj * dj > r * r {
                continue;
            }
            let (i, j) = (ci as isize + di, cj as isize + dj);
            if i >= 0 && j >= 0 && (i as usize) < nx && (j as usize) < ny {
                v.push((i as usize, j as usize));
            }
        }
    }
    v
}

// ------------------------------------------------------------------------------------------------ rendu
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MpDisplay {
    Phases,
    Pressure,
    Temperature,
    Velocity,
    Density,
    Vapor,
    Divergence,
    Flame,
    HeatRelease,
}

impl MpDisplay {
    pub const ALL: [MpDisplay; 9] =
        [Self::Phases, Self::Pressure, Self::Temperature, Self::Velocity, Self::Density, Self::Vapor, Self::Divergence, Self::Flame, Self::HeatRelease];
    pub fn name(self) -> &'static str {
        match self {
            Self::Phases => "PHASES",
            Self::Pressure => "PRESSION (relative à 1 atm)",
            Self::Temperature => "TEMPÉRATURE",
            Self::Velocity => "VITESSE",
            Self::Density => "MASSE VOLUMIQUE",
            Self::Vapor => "VAPEUR (fraction)",
            Self::Divergence => "DIVERGENCE (∇·u)",
            Self::Flame => "FLAMME (rayonnement des gaz chauds + front de réaction)",
            Self::HeatRelease => "DÉGAGEMENT DE CHALEUR (échelle log)",
        }
    }
    pub fn next(self) -> Self {
        let i = Self::ALL.iter().position(|&m| m == self).unwrap();
        Self::ALL[(i + 1) % Self::ALL.len()]
    }
}

#[derive(Clone, Copy)]
pub struct RenderOpts {
    pub display: MpDisplay,
    /// échelle d'affichage de T − T_ambiant (K)
    pub t_scale: f64,
    /// échelle de pression (Pa), 0 = automatique (max |p − 1 atm|)
    pub p_scale: f64,
    /// surcouche des cellules spéciales (puits rouge, émetteur vert, accélérateur violet, téléporteur bleu / destination cyan) et des
    /// solides thermiques (isotherme bleu, chauffage orange). Sans elle, les cellules spéciales s'affichent comme du fluide.
    pub show_effects: bool,
    /// solides tous blancs au lieu de la couleur de leur matériau (touche I)
    pub solids_white: bool,
}

impl RenderOpts {
    pub fn new(display: MpDisplay, t_scale: f64, p_scale: f64) -> Self {
        RenderOpts { display, t_scale, p_scale, show_effects: false, solids_white: false }
    }
}

fn phase_rgb(ph: &Phase) -> [f32; 3] {
    ph.color
}

fn hsv(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = (h % 360.0 + 360.0) % 360.0;
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + m, g + m, b + m]
}

/// lueur d'un gaz chaud : noir sous 800 K, rouge sombre → orange → jaune pâle vers 2400 K (intensité et teinte approchées du corps noir)
fn blackbody_rgb(t: f64) -> [f32; 3] {
    let x = ((t - 800.0) / 1600.0).clamp(0.0, 1.0) as f32;
    [(1.6 * x).min(1.0), (1.6 * x - 0.45).clamp(0.0, 0.9), (1.6 * x - 1.0).clamp(0.0, 0.6)]
}

/// bleu/cyan (froid) → noir → rouge → jaune (chaud)
fn temp_rgb(x: f64) -> [f32; 3] {
    let x = x.clamp(-1.0, 1.0) as f32;
    if x >= 0.0 {
        [(2.0 * x).min(1.0), (2.0 * x - 1.0).max(0.0), 0.0]
    } else {
        let c = -x;
        [0.0, (2.0 * c - 1.0).max(0.0), (2.0 * c).min(1.0)]
    }
}

fn signed_rgb(v: f64, max_abs: f64) -> [f32; 3] {
    if max_abs <= 1e-12 {
        return [0.0; 3];
    }
    let x = (v / max_abs).clamp(-1.0, 1.0) as f32;
    if x < 0.0 { [0.0, -x, -x] } else { [x, 0.0, 0.0] }
}

/// Rend la simulation en RGBA (nx×ny, ligne 0 = HAUT du domaine : l'axe y est inversé par rapport aux indices j).
pub fn render_rgba(sim: &MultiphaseSim, o: RenderOpts, out: &mut Vec<u8>) {
    let (nx, ny) = (sim.nx, sim.ny);
    out.resize(nx * ny * 4, 255);
    let vapor_ids: Vec<usize> = sim.reg.changes.iter().map(|c| c.vapor).collect();
    // échelles automatiques
    let mut pmax = o.p_scale;
    if pmax <= 0.0 {
        pmax = (0..nx * ny).filter(|&c| !sim.is_solid(c)).map(|c| (sim.p[c] - P_ATM).abs()).fold(1.0, f64::max);
    }
    let (mut rmin, mut rmax, mut smax) = (f64::MAX, 0.0f64, 1e-9f64);
    for c in 0..nx * ny {
        if !sim.is_solid(c) {
            rmin = rmin.min(sim.rho_mix[c]);
            rmax = rmax.max(sim.rho_mix[c]);
        }
    }
    let qmax = sim.heat_release.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let mut dmax = 1e-9f64;
    for j in 0..ny {
        for i in 0..nx {
            let (vx, vy) = cell_velocity(sim, i, j);
            smax = smax.max((vx * vx + vy * vy).sqrt());
            if !sim.is_solid(sim.idx(i, j)) {
                dmax = dmax.max(sim.divergence_of(&sim.u, &sim.v, i, j).abs());
            }
        }
    }
    for j in 0..ny {
        for i in 0..nx {
            let c = sim.idx(i, j);
            let rgb: [f32; 3] = if sim.is_solid(c) {
                if o.display == MpDisplay::Temperature {
                    temp_rgb((sim.temp[c] - sim.t_ambient) / o.t_scale.max(1e-3))
                } else if o.solids_white {
                    [1.0, 1.0, 1.0]
                } else {
                    match (o.show_effects, sim.thermal[c]) {
                        (true, MpThermal::Isothermal(_)) => [0x20 as f32 / 255.0, 0x60 as f32 / 255.0, 0xd0 as f32 / 255.0],
                        (true, MpThermal::Heater(_)) => [0xd0 as f32 / 255.0, 0x80 as f32 / 255.0, 0x20 as f32 / 255.0],
                        _ => sim.reg.solids[sim.solid[c] as usize].color,
                    }
                }
            } else {
                match o.display {
                    MpDisplay::Phases => {
                        let mut col = [0.0f32; 3];
                        for k in 0..sim.reg.phases.len() {
                            let a = sim.alpha[k][c].clamp(0.0, 1.0) as f32;
                            let pc = phase_rgb(&sim.reg.phases[k]);
                            for q in 0..3 {
                                col[q] += a * pc[q];
                            }
                        }
                        col
                    }
                    MpDisplay::Pressure => signed_rgb(sim.p[c] - P_ATM, pmax),
                    MpDisplay::Temperature => temp_rgb((sim.temp[c] - sim.t_ambient) / o.t_scale.max(1e-3)),
                    MpDisplay::Velocity => {
                        let (vx, vy) = cell_velocity(sim, i, j);
                        let s = (vx * vx + vy * vy).sqrt();
                        hsv((vy.atan2(vx).to_degrees() + 180.0) as f32, 1.0, (s / smax).clamp(0.0, 1.0) as f32)
                    }
                    MpDisplay::Density => {
                        let x = if rmax > rmin { ((sim.rho_mix[c].ln() - rmin.ln()) / (rmax.ln() - rmin.ln())).clamp(0.0, 1.0) as f32 } else { 0.0 };
                        [x, x, x]
                    }
                    MpDisplay::Divergence => signed_rgb(sim.divergence_of(&sim.u, &sim.v, i, j), dmax),
                    MpDisplay::HeatRelease => {
                        // 6 décades sous le maximum ; exothermique jaune→blanc, endothermique (dissociation) bleu
                        let q = sim.heat_release[c];
                        let x = if qmax > 0.0 && q.abs() > 0.0 { ((q.abs() / qmax).log10() / 6.0 + 1.0).clamp(0.0, 1.0) as f32 } else { 0.0 };
                        if q >= 0.0 { [x, x * x, x * x * x * 0.5] } else { [0.0, 0.3 * x, x] }
                    }
                    MpDisplay::Flame => {
                        // fond : couleur des espèces assombrie ; gaz chauds : incandescence (corps noir approché) ∝ (T − 800 K) ;
                        // zone de réaction : lueur bleue (chimiluminescence CH*, C2*) ∝ log du dégagement de chaleur
                        let mut col = [0.0f32; 3];
                        for k in 0..sim.reg.phases.len() {
                            let a = sim.alpha[k][c].clamp(0.0, 1.0) as f32;
                            let pc = phase_rgb(&sim.reg.phases[k]);
                            for q in 0..3 {
                                col[q] += 0.35 * a * pc[q];
                            }
                        }
                        let b = blackbody_rgb(sim.temp[c]);
                        let q = sim.heat_release[c];
                        let blue = if qmax > 0.0 && q > 0.0 { ((q / qmax).log10() / 4.0 + 1.0).clamp(0.0, 1.0) as f32 } else { 0.0 };
                        [col[0] + b[0] + 0.25 * blue, col[1] + b[1] + 0.45 * blue, col[2] + b[2] + 1.0 * blue]
                    }
                    MpDisplay::Vapor => {
                        let a: f64 = vapor_ids.iter().map(|&k| sim.alpha[k][c]).sum::<f64>().clamp(0.0, 1.0);
                        let a = a as f32;
                        [a, a, a]
                    }
                }
            };
            let row = ny - 1 - j; // j croît vers le haut
            let p = (row * nx + i) * 4;
            out[p] = (rgb[0].clamp(0.0, 1.0) * 255.0) as u8;
            out[p + 1] = (rgb[1].clamp(0.0, 1.0) * 255.0) as u8;
            out[p + 2] = (rgb[2].clamp(0.0, 1.0) * 255.0) as u8;
            out[p + 3] = 255;
        }
    }
    if o.show_effects {
        let mut put = |c: usize, rgb: [u8; 3]| {
            let (i, j) = (c % nx, c / nx);
            let p = ((ny - 1 - j) * nx + i) * 4;
            out[p] = rgb[0];
            out[p + 1] = rgb[1];
            out[p + 2] = rgb[2];
        };
        for c in 0..nx * ny {
            match sim.special[c] {
                Some(MpEffect::Sink { .. }) => put(c, [0xff, 0x00, 0x00]),
                Some(MpEffect::Emitter { .. }) => put(c, [0x00, 0xff, 0x00]),
                Some(MpEffect::Accelerator { .. }) => put(c, [0xff, 0x00, 0xff]),
                Some(MpEffect::Grille { .. }) => put(c, [0x9a, 0x9a, 0xa0]),
                Some(MpEffect::Teleporter { dest, radius, .. }) => {
                    put(c, [0x00, 0x00, 0xff]);
                    if dest < nx * ny {
                        for d in sim.tp_dest_cells(dest, radius) {
                            put(d, [0x00, 0xff, 0xff]);
                        }
                    }
                }
                None => {}
            }
        }
    }
}

/// vitesse au centre de la cellule (m/s) : moyenne des deux faces
pub fn cell_velocity(sim: &MultiphaseSim, i: usize, j: usize) -> (f64, f64) {
    let vx = 0.5 * (sim.u[sim.uidx(i, j)] + sim.u[sim.uidx(i + 1, j)]);
    let vy = 0.5 * (sim.v[sim.vidx(i, j)] + sim.v[sim.vidx(i, j + 1)]);
    (vx, vy)
}

// ------------------------------------------------------------------------------------------------ statistiques
#[derive(Clone, Debug, Default)]
pub struct MpStats {
    pub time: f64,
    pub dt: f64,
    pub steps: usize,
    pub pressure_iters: usize,
    /// (nom, masse kg/m, aire m²) par phase
    pub phases: Vec<(String, f64, f64)>,
    pub kinetic: f64,
    pub internal: f64,
    pub t_min: f64,
    pub t_max: f64,
    pub p_min: f64,
    pub p_max: f64,
    pub speed_max: f64,
    pub finite: bool,
    /// chaleur injectée par les solides chauffants (J/m)
    pub heat_input: f64,
    /// nombre de cellules spéciales
    pub effects: usize,
    /// pas annulés puis refaits (filet de sécurité numérique)
    pub rollbacks: usize,
    /// chaleur de réaction dégagée (J/m), énergie rayonnée (J/m), puissance chimique instantanée (W/m)
    pub chem_heat: f64,
    pub radiated: f64,
    pub chem_power: f64,
    /// facteur d'épaississement maximal des flammes (TFM)
    pub flame_factor: f64,
    /// (nom, avancement cumulé mol/m) par réaction
    pub reactions: Vec<(String, f64)>,
}

pub fn compute_stats(sim: &MultiphaseSim) -> MpStats {
    let mut st = MpStats {
        time: sim.time,
        dt: sim.last_dt,
        steps: sim.steps_done(),
        pressure_iters: sim.last_pressure_iters,
        t_min: f64::MAX,
        t_max: f64::MIN,
        p_min: f64::MAX,
        p_max: f64::MIN,
        ..Default::default()
    };
    for k in 0..sim.reg.phases.len() {
        st.phases.push((sim.reg.phases[k].name.clone(), sim.total_mass(k), sim.phase_area(k)));
    }
    st.kinetic = sim.kinetic_energy();
    st.internal = sim.total_internal_energy();
    for c in 0..sim.nx * sim.ny {
        st.t_min = st.t_min.min(sim.temp[c]);
        st.t_max = st.t_max.max(sim.temp[c]);
        if !sim.is_solid(c) {
            st.p_min = st.p_min.min(sim.p[c]);
            st.p_max = st.p_max.max(sim.p[c]);
        }
    }
    st.speed_max = sim.max_speed();
    st.heat_input = sim.energy_input;
    st.effects = sim.special_count();
    st.rollbacks = sim.rollbacks;
    st.chem_heat = sim.chem_heat;
    st.radiated = sim.radiated;
    st.chem_power = sim.heat_release.iter().sum::<f64>() * sim.dx * sim.dx;
    st.flame_factor = sim.flame_factor.iter().fold(1.0f64, |a, &b| a.max(b));
    st.reactions = sim.reg.reactions.iter().zip(sim.reacted.iter().chain(std::iter::repeat(&0.0))).map(|(r, &x)| (r.name.clone(), x)).collect();
    st.finite = st.speed_max.is_finite() && st.p_min.is_finite() && st.p_max.is_finite() && st.t_max.is_finite() && st.kinetic.is_finite();
    st
}

/// (libellé, valeur) pour la cellule (i, j)
pub fn cell_info(sim: &MultiphaseSim, i: usize, j: usize) -> Vec<(String, String)> {
    let c = sim.idx(i, j);
    let mut v = vec![("Position".to_string(), format!("{i}, {j}")), ("Température (K)".into(), format!("{:.2}", sim.temp[c]))];
    if sim.is_solid(c) {
        v.push(("Solide".into(), sim.reg.solids[sim.solid[c] as usize].name.clone()));
        match sim.thermal[c] {
            MpThermal::Passive => {}
            MpThermal::Isothermal(t) => v.push(("Thermique".into(), format!("isotherme {t:.1} K"))),
            MpThermal::Heater(w) => v.push(("Thermique".into(), format!("chauffage {w:.3e} W/m³"))),
        }
        return v;
    }
    // cellule mixte : température de chaque groupe présent (non-équilibre thermique)
    let present: Vec<usize> = (0..sim.ngroups()).filter(|&g| sim.groups[g].iter().any(|&k| sim.mass[k][c] > 1e-9)).collect();
    if present.len() > 1 {
        for g in present {
            let name = sim.groups[g].iter().map(|&k| sim.reg.phases[k].name.as_str()).collect::<Vec<_>>().join(" + ");
            v.push((format!("T {name} (K)"), format!("{:.2}", sim.tg[g][c])));
        }
    }
    match sim.special[c] {
        Some(MpEffect::Sink { rate }) => v.push(("Cellule spéciale".into(), format!("puits, aspiration {rate:.3} m/s"))),
        Some(MpEffect::Emitter { vx, vy, phase, temp }) => v.push((
            "Cellule spéciale".into(),
            format!("émetteur {} à {temp:.1} K, v = ({vx:.2}, {vy:.2}) m/s", sim.reg.phases.get(phase).map_or("?", |p| p.name.as_str())),
        )),
        Some(MpEffect::Accelerator { ax, ay }) => v.push(("Cellule spéciale".into(), format!("accélérateur ({ax:.2}, {ay:.2}) m/s²"))),
        Some(MpEffect::Grille { hole, open }) => v.push(("Cellule spéciale".into(), format!("grille, trous de {:.1} mm, {:.0} % ouverte", hole * 1e3, open * 100.0))),
        Some(MpEffect::Teleporter { dest, speed, radius }) => v.push((
            "Cellule spéciale".into(),
            format!("téléporteur → ({}, {}), rayon {radius}, {speed:.2} m/s", dest % sim.nx, dest / sim.nx),
        )),
        None => {}
    }
    let (vx, vy) = cell_velocity(sim, i, j);
    v.push(("Pression (Pa)".into(), format!("{:.1}", sim.p[c])));
    v.push(("ρ mélange (kg/m³)".into(), format!("{:.3}", sim.rho_mix[c])));
    v.push(("Vitesse (m/s)".into(), format!("{:.3}, {:.3}", vx, vy)));
    for k in 0..sim.reg.phases.len() {
        if sim.alpha[k][c] > 1e-6 {
            v.push((format!("α {}", sim.reg.phases[k].name), format!("{:.4}", sim.alpha[k][c])));
        }
    }
    for pc in &sim.reg.changes {
        v.push((format!("T_sat à p ({})", sim.reg.phases[pc.liquid].name), format!("{:.2} K", pc.t_sat(sim.p[c]))));
    }
    if !sim.reg.reactions.is_empty() {
        v.push(("Dégagement de chaleur (W/m³)".into(), format!("{:.3e}", sim.heat_release[c])));
        if sim.flame_factor[c] > 1.0 {
            v.push(("Épaississement de flamme F".into(), format!("{:.1}", sim.flame_factor[c])));
        }
        // fractions molaires des gaz
        let np = sim.reg.phases.len();
        let n: Vec<f64> = (0..np).map(|k| if sim.reg.phases[k].is_gas() { sim.mass[k][c].max(0.0) / sim.reg.phases[k].molar_mass } else { 0.0 }).collect();
        let tot: f64 = n.iter().sum();
        if tot > 0.0 {
            for k in 0..np {
                if n[k] / tot > 1e-4 {
                    v.push((format!("x {}", sim.reg.phases[k].name), format!("{:.4}", n[k] / tot)));
                }
            }
        }
    }
    v
}

/// « Reset flow » : annule toutes les vitesses (les faces imposées par les puits / émetteurs / entrées sont rétablies au pas suivant)
/// et lève le drapeau de divergence (l'état restauré par le filet de sécurité est sain).
pub fn reset_flow(sim: &mut MultiphaseSim) {
    sim.u.iter_mut().for_each(|x| *x = 0.0);
    sim.v.iter_mut().for_each(|x| *x = 0.0);
    sim.diverged = false;
}

/// « Reset temperature » : toutes les cellules (fluide et solide) à `t` K, masses inchangées ; les solides isothermes gardent leur T imposée.
pub fn reset_temperature(sim: &mut MultiphaseSim, t: f64) {
    for c in 0..sim.nx * sim.ny {
        if sim.is_solid(c) {
            let tc = if let MpThermal::Isothermal(ti) = sim.thermal[c] { ti } else { t };
            sim.e[c] = sim.reg.solids[sim.solid[c] as usize].rho_cp() * tc;
            sim.temp[c] = tc;
        } else {
            sim.set_cell_temperature(c, t);
        }
    }
    sim.refresh();
}

// ------------------------------------------------------------------------------------------------ sauvegarde de scène
/// Effet de cellule tel qu'enregistré dans un fichier : la phase est désignée par son NOM (les indices changent si on modifie les matériaux).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EffectFile {
    Sink { rate: f64 },
    Emitter { vx: f64, vy: f64, phase: String, temp: f64 },
    Accelerator { ax: f64, ay: f64 },
    Grille { hole: f64, open: f64 },
    Teleporter {
        dest: usize,
        speed: f64,
        /// rayon de la zone de destination (absent des scènes enregistrées avant son ajout : 0, une cellule)
        #[serde(default)]
        radius: usize,
    },
}

/// Scène = solides (par nom de matériau) + comportements thermiques + cellules spéciales. Équivalent du fichier `.fsf` du mono-fluide
/// (qui ne contient que les cellules, pas l'état du fluide).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneFile {
    pub version: u32,
    pub nx: usize,
    pub ny: usize,
    pub solids: Vec<(usize, String)>,
    pub thermal: Vec<(usize, MpThermal)>,
    pub effects: Vec<(usize, EffectFile)>,
}

pub fn scene_to_json(sim: &MultiphaseSim) -> Result<String, String> {
    let mut f = SceneFile { version: 1, nx: sim.nx, ny: sim.ny, solids: vec![], thermal: vec![], effects: vec![] };
    for c in 0..sim.nx * sim.ny {
        if sim.is_solid(c) {
            f.solids.push((c, sim.reg.solids[sim.solid[c] as usize].name.clone()));
            if sim.thermal[c] != MpThermal::Passive {
                f.thermal.push((c, sim.thermal[c]));
            }
        }
        let name = |k: usize| sim.reg.phases.get(k).map(|p| p.name.clone()).unwrap_or_default();
        match sim.special[c] {
            Some(MpEffect::Sink { rate }) => f.effects.push((c, EffectFile::Sink { rate })),
            Some(MpEffect::Emitter { vx, vy, phase, temp }) => f.effects.push((c, EffectFile::Emitter { vx, vy, phase: name(phase), temp })),
            Some(MpEffect::Accelerator { ax, ay }) => f.effects.push((c, EffectFile::Accelerator { ax, ay })),
            Some(MpEffect::Grille { hole, open }) => f.effects.push((c, EffectFile::Grille { hole, open })),
            Some(MpEffect::Teleporter { dest, speed, radius }) => f.effects.push((c, EffectFile::Teleporter { dest, speed, radius })),
            None => {}
        }
    }
    serde_json::to_string(&f).map_err(|e| e.to_string())
}

/// Charge une scène : remplace TOUS les solides / effets / comportements thermiques (comme le chargement des solides du mono-fluide).
/// Retourne les avertissements (matériau inconnu ignoré, …). Erreur si la grille a une autre taille.
pub fn scene_from_json(sim: &mut MultiphaseSim, json: &str) -> Result<Vec<String>, String> {
    let f: SceneFile = serde_json::from_str(json).map_err(|e| format!("fichier illisible : {e}"))?;
    if f.nx != sim.nx || f.ny != sim.ny {
        return Err(format!("grille différente : le fichier fait {}×{}, la simulation {}×{}", f.nx, f.ny, sim.nx, sim.ny));
    }
    let n = sim.nx * sim.ny;
    let mut warn = vec![];
    for c in 0..n {
        sim.clear_cell(c);
    }
    for (c, name) in &f.solids {
        match (sim.reg.solids.iter().position(|s| &s.name == name), *c < n) {
            (Some(s), true) => {
                let (i, j) = (c % sim.nx, c / sim.nx);
                apply_tool(sim, MpTool::Solid(s), &[(i, j)], sim.t_ambient, (0.0, 0.0), 0.0);
            }
            (None, _) => warn.push(format!("solide « {name} » inconnu : cellule {c} ignorée")),
            _ => {}
        }
    }
    for (c, t) in &f.thermal {
        if *c < n {
            sim.set_thermal_cell(*c, *t);
        }
    }
    for (c, e) in &f.effects {
        if *c >= n {
            continue;
        }
        let eff = match e {
            EffectFile::Sink { rate } => Some(MpEffect::Sink { rate: *rate }),
            EffectFile::Accelerator { ax, ay } => Some(MpEffect::Accelerator { ax: *ax, ay: *ay }),
            EffectFile::Grille { hole, open } => Some(MpEffect::Grille { hole: *hole, open: *open }),
            EffectFile::Teleporter { dest, speed, radius } => Some(MpEffect::Teleporter { dest: *dest, speed: *speed, radius: *radius }),
            EffectFile::Emitter { vx, vy, phase, temp } => match sim.reg.phases.iter().position(|p| &p.name == phase) {
                Some(k) => Some(MpEffect::Emitter { vx: *vx, vy: *vy, phase: k, temp: *temp }),
                None => {
                    warn.push(format!("fluide « {phase} » inconnu : émetteur de la cellule {c} ignoré"));
                    None
                }
            },
        };
        if let Some(eff) = eff {
            sim.set_effect(*c, Some(eff));
        }
    }
    sim.refresh();
    Ok(warn)
}

impl MultiphaseSim {
    /// nombre de pas effectués
    pub fn steps_done(&self) -> usize {
        self.step_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(sim: &MultiphaseSim, o: RenderOpts, path: &str) {
        let mut buf = vec![];
        render_rgba(sim, o, &mut buf);
        if let Ok(dir) = std::env::var("MP_VIEW_OUT") {
            let mut f = String::new();
            f.push_str(&format!("P6\n{} {}\n255\n", sim.nx, sim.ny));
            let mut bytes = f.into_bytes();
            for px in buf.chunks(4) {
                bytes.extend_from_slice(&px[..3]);
            }
            std::fs::write(format!("{dir}/{path}.ppm"), bytes).unwrap();
        }
    }

    #[test]
    fn scenarios_build_and_run_and_render() {
        for sc in MpScenario::ALL {
            let mut s = build_scenario(sc, 48, 36, 0.01, 9.81);
            let m0 = s.total_mass_all();
            for _ in 0..15 {
                s.step();
            }
            let st = compute_stats(&s);
            assert!(st.finite, "{sc:?} non fini");
            let o = RenderOpts::new(MpDisplay::Phases, 100.0, 0.0);
            png(&s, o, &format!("{sc:?}"));
            let mut buf = vec![];
            for d in MpDisplay::ALL {
                render_rgba(&s, RenderOpts { display: d, ..o }, &mut buf);
                assert_eq!(buf.len(), 48 * 36 * 4);
            }
            // masse conservée (scénarios entièrement fermés, sans émetteur : les autres échangent de la masse avec l'extérieur)
            if s.bc.iter().all(|b| *b == Bc::Wall) && s.special_count() == 0 {
                assert!(((s.total_mass_all() - m0) / m0).abs() < 1e-9, "{sc:?}: masse");
            }
        }
    }

    #[test]
    fn brush_tools_keep_state_consistent() {
        let mut s = build_scenario(MpScenario::Empty, 30, 30, 0.01, 9.81);
        let cells = brush_cells(30, 30, 15, 10, 3, true);
        apply_tool(&mut s, MpTool::Phase(EAU), &cells, 293.15, (0.0, 0.0), 0.016);
        assert!(s.alpha[EAU][s.idx(15, 10)] > 0.999);
        apply_tool(&mut s, MpTool::Solid(ACIER), &brush_cells(30, 30, 5, 5, 2, false), 400.0, (0.0, 0.0), 0.016);
        assert!(s.is_solid(s.idx(5, 5)) && (s.temp[s.idx(5, 5)] - 400.0).abs() < 1e-6);
        apply_tool(&mut s, MpTool::Heat, &cells, 350.0, (0.0, 0.0), 0.1);
        let t0 = s.time;
        while s.time - t0 < 0.05 {
            s.step();
        }
        s.clear_heat_targets();
        assert!(s.temp[s.idx(15, 10)] > 293.5, "T = {}", s.temp[s.idx(15, 10)]);
        apply_tool(&mut s, MpTool::EraseSolid, &brush_cells(30, 30, 5, 5, 2, false), 293.15, (0.0, 0.0), 0.016);
        assert!(!s.is_solid(s.idx(5, 5)) && s.alpha[AIR][s.idx(5, 5)] > 0.999);
        apply_tool(&mut s, MpTool::Push, &cells, 293.15, (1.0, 0.0), 0.05);
        for _ in 0..30 {
            s.step();
        }
        assert!(compute_stats(&s).finite);
    }

    #[test]
    fn scene_roundtrip_keeps_solids_effects_and_thermal_and_rejects_other_grid() {
        let mut s = build_scenario(MpScenario::Empty, 30, 20, 0.01, 9.81);
        apply_tool_with(&mut s, MpTool::Solid(ACIER), &brush_cells(30, 20, 5, 5, 2, false), &ToolOpts { temp: 293.15, cursor: (0.0, 0.0), dt: 0.0, thermal: MpThermal::Heater(1e6) });
        apply_tool(&mut s, MpTool::Effect(MpEffect::Emitter { vx: 0.0, vy: 1.0, phase: EAU, temp: 300.0 }), &[(20, 3)], 293.15, (0.0, 0.0), 0.0);
        let dest = s.idx(25, 15);
        apply_tool(&mut s, MpTool::Effect(MpEffect::Teleporter { dest, speed: 0.4, radius: 2 }), &[(10, 10)], 293.15, (0.0, 0.0), 0.0);
        let json = scene_to_json(&s).unwrap();
        let mut t = build_scenario(MpScenario::Empty, 30, 20, 0.01, 9.81);
        let w = scene_from_json(&mut t, &json).unwrap();
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(s.solid, t.solid);
        assert_eq!(s.special, t.special);
        assert_eq!(s.thermal, t.thermal);
        // grille différente : refusée sans rien modifier
        let mut other = build_scenario(MpScenario::Empty, 31, 20, 0.01, 9.81);
        assert!(scene_from_json(&mut other, &json).is_err());
        assert!(other.solid.iter().all(|&x| x < 0));
        // un matériau supprimé entre-temps : avertissement, pas de plantage
        let mut lib = MaterialLibrary::default();
        let acier = lib.solid_index("acier").unwrap();
        lib.remove_solid(acier);
        let mut u = build_scenario_with(&lib, MpScenario::Empty, 30, 20, 0.01, 9.81).unwrap();
        let w = scene_from_json(&mut u, &json).unwrap();
        assert!(w.iter().any(|m| m.contains("acier")), "{w:?}");
    }

    #[test]
    fn scenarios_use_the_user_library_and_report_missing_materials() {
        // renommer : les rôles sont retrouvés par propriété
        let mut lib = MaterialLibrary::default();
        lib.rename_fluid(1, "H2O");
        lib.rename_fluid(0, "azote");
        let mut s = build_scenario_with(&lib, MpScenario::DamBreak, 32, 24, 0.01, 9.81).unwrap();
        assert!(s.total_mass(1) > 0.0);
        for _ in 0..10 {
            s.step();
        }
        assert!(compute_stats(&s).finite);
        // plus de liquide : message clair
        let mut lib = MaterialLibrary::default();
        lib.remove_fluid(3);
        lib.remove_fluid(1);
        let e = build_scenario_with(&lib, MpScenario::DamBreak, 32, 24, 0.01, 9.81).err().unwrap();
        assert!(e.contains("liquide"), "{e}");
        // pas d'huile : l'huile / eau / air échoue, le barrage fonctionne
        let mut lib = MaterialLibrary::default();
        lib.remove_fluid(3);
        assert!(build_scenario_with(&lib, MpScenario::OilWaterAir, 32, 24, 0.01, 9.81).is_err());
        assert!(build_scenario_with(&lib, MpScenario::DamBreak, 32, 24, 0.01, 9.81).is_ok());
        // un fluide en plus (3 liquides) : l'huile / eau / air tourne toujours
        let mut lib = MaterialLibrary::default();
        let mut extra = lib.template_fluid(crate::simulation::phases::PhaseKind::Liquid);
        extra.name = "glycérine".into();
        lib.fluids.push(extra);
        let mut s = build_scenario_with(&lib, MpScenario::OilWaterAir, 32, 24, 0.01, 9.81).unwrap();
        for _ in 0..10 {
            s.step();
        }
        assert!(compute_stats(&s).finite && s.reg.phases.len() == 5);
    }

    #[test]
    fn reset_helpers_and_effect_overlay() {
        let mut s = build_scenario(MpScenario::Boiling, 40, 30, 0.01, 9.81);
        for _ in 0..5 {
            s.step();
        }
        assert!(s.max_speed() > 0.0 || s.temp.iter().any(|&t| t > 400.0));
        reset_flow(&mut s);
        assert_eq!(s.max_speed(), 0.0);
        reset_temperature(&mut s, 293.15);
        assert!(s.temp.iter().all(|&t| (t - 293.15).abs() < 1e-6));
        // surcouche : un puits rouge, un émetteur vert ; sans surcouche rien ne change
        apply_tool(&mut s, MpTool::Effect(MpEffect::Sink { rate: 0.1 }), &[(5, 20)], 293.15, (0.0, 0.0), 0.0);
        apply_tool(&mut s, MpTool::Effect(MpEffect::Emitter { vx: 0.0, vy: 0.5, phase: EAU, temp: 293.15 }), &[(7, 20)], 293.15, (0.0, 0.0), 0.0);
        let (mut a, mut b) = (vec![], vec![]);
        let base = RenderOpts::new(MpDisplay::Phases, 100.0, 0.0);
        render_rgba(&s, base, &mut a);
        render_rgba(&s, RenderOpts { show_effects: true, ..base }, &mut b);
        let px = |buf: &Vec<u8>, i: usize, j: usize| -> [u8; 3] {
            let p = ((30 - 1 - j) * 40 + i) * 4;
            [buf[p], buf[p + 1], buf[p + 2]]
        };
        assert_ne!(px(&a, 5, 20), [255, 0, 0]);
        assert_eq!(px(&b, 5, 20), [255, 0, 0]);
        assert_eq!(px(&b, 7, 20), [0, 255, 0]);
        // le mode divergence rend sans erreur
        render_rgba(&s, RenderOpts { display: MpDisplay::Divergence, ..base }, &mut a);
        assert_eq!(a.len(), 40 * 30 * 4);
        // le GPU résident gère les extras (cellules spéciales, solides thermiques, entrées)
        assert!(crate::simulation::gpu_multiphase::gpu_unsupported_reason(&s).is_none());
    }
}
