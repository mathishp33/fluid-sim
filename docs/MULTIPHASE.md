# Solveur multiphasique (CPU, f64) — notes d'intégration

## Fichiers (`src/simulation/multi/`)
- Cœur : `multiphase.rs` (état `MultiphaseSim`), `phases.rs` (matériaux : phases, réactions, registre), `vof.rs` (géométrie PLIC),
  `mp_step.rs` (pas de temps, filet de sécurité), `mp_advect.rs` (transport), `mp_pressure.rs` et `mp_mg.rs` (pression implicite,
  multigrille), `mp_force.rs` (forces), `mp_thermo.rs` (conduction, changements de phase, travail de pression), `mp_groups.rs` (une
  température par groupe de phases), `mp_species.rs` (diffusion, chimie), `mp_cells.rs` (cellules spéciales), `mp_par.rs` (fils).
- Grains (GRANULAIRE.md) : `mp_granular.rs`, `mp_friction.rs`, `mp_percolation.rs`. Corps rigides (SOLID.md) : `mp_bodies.rs`,
  `mp_contacts.rs`, `mp_body_heat.rs`.
- Expériences (EXPERIENCES.md) : `mp_experiment.rs`, `mp_script.rs`, `mp_fields.rs`. Scénarios, pinceaux et rendu sans egui :
  `mp_view.rs` ; état complet : `mp_state.rs` ; bibliothèque de matériaux : `mp_materials.rs`.
- GPU : `gpu_multiphase.rs` (hôte wgpu) et `mp_gpu/*.wgsl` (noyaux, assemblés à la compilation).
- Essais : `mp_tests.rs`, `mp_chem_tests.rs`, `mp_gpu_tests.rs`, `mp_experiment_tests.rs`, `mp_perf_tests.rs`, et les `mod tests` des
  modules.
- Les lois de propriétés (`PropertyLaw`, `materials.rs`) sont communes avec le solveur monophasique (`src/simulation/mono/`), dont le
  multiphasique est indépendant.

## Utilisation
```rust
let (reg, air, eau, vapeur) = Registry::water_air_steam();   // ou Registry::new() + add_phase / add_solid / add_phase_change
let mut sim = MultiphaseSim::new(128, 96, 0.005, reg, air);  // dx = 5 mm
sim.bc[3] = Bc::Open { p: 101325.0 };                        // haut ouvert (gauche, droite, bas, haut)
sim.paint_phase(|x, y| if y < 0.2 { Some(eau) } else { Some(air) });
sim.paint_solid(|x, y| if (x - 0.3).abs() < 0.02 && y < 0.1 { Some(0) } else { None });   // si reg.solids non vide
sim.finalize_init(|_, _| 293.15);                            // T, pression hydrostatique, masses, énergie
loop { sim.step(); /* lire sim.alpha[eau], sim.temp, sim.p, sim.u, sim.v */ }
```
Convention : j croît vers le **haut** (gravité (0, −9.81)) ; l'affichage doit inverser l'axe y si votre UI a y vers le bas.

## Modèles physiques
| Élément | Modèle |
|---|---|
| Gaz | gaz parfait, cv = cp − R, μ(T) Sutherland, k(T) |
| Liquide | équation de Tait (eau : B = 3,07e8 Pa, n = 7,15 → c = 1482 m/s, K = 2,2 GPa), ρ(T) de Kell (maximum à 4 °C), μ(T) Vogel |
| Interface | VOF géométrique PLIC (formules analytiques), flux conservatifs, N phases (priorité liquides puis gaz) |
| Pression | faiblement compressible **implicite** (Helmholtz, κ_eff du mélange à entropie constante, Newton + gradient conjugué préconditionné par multigrille) : dt non limité par le son |
| Forces | gravité, viscosité (tenseur symétrique, μ variable), tension de surface CSF + courbure par fonctions de hauteur |
| Énergie | énergie interne conservative, conduction (k harmonique, fluides + solides), travail −p∇·u |
| Changement de phase | p_sat(T) Clausius–Clapeyron avec chaleur latente de Kirchhoff cohérente avec les enthalpies (h_v − h_l = L(T) exactement) ; interface résolue : ancrage de Hardt–Wondra à T_sat ; germes : Schnerr–Sauer ; limité par la masse, l'énergie et le retour à p = p_sat |
| Solides | cellules étanches avec ρ, cp, k propres (conduction fluide↔solide) |
| Bords | parois (glissement libre) ou ouverts : `Bc::Open { p }`, p au point le plus haut du domaine, la pression du bord suivant l'hydrostatique du fluide AMBIANT (`open_face_pressure`, colonne compressible au second ordre) ; ce qui entre est l'ambiant à `t_ambient`. À pression constante, une boîte d'air ouverte sur les côtés sous la gravité se mettait à 9 m/s (ρ g H d'écart entre le haut et le bas d'un côté) ; à masse volumique constante, un vent parasite croissait sans fin (4 mm/s après 5 s). Entrées de bord à vitesse imposée |

## Validation (tests, `cargo test --release`)
hydrostatique (courants 5e-11 m/s, p_fond exact) · Δp = β/κ·ΔT de l'eau confinée (0,25 %) · ballottement ω² = gk·tanh(kh) (+1 %) ·
Laplace Δp = σ/R (0,1 %) · barrage (front 1,3·√(gH), masse exacte) · ébullition en boîte fermée (masse 1e-14, énergie à 0,4 % de L·Δm) ·
cavitation (p relâchée à p_sat, volume de vapeur = volume retiré) · eau/huile/air statique + goutte d'huile qui monte · éclaboussure (volume exact).
Les 20 tests de `FluidSim` passent toujours.

## Advection conservative de la quantité de mouvement (défaut : `conservative_momentum = true`)
ρ_f·u est advectée sur des volumes de contrôle de faces avec les MÊMES flux de masse que le VOF (moyennés aux faces des VC),
valeur transportée reconstruite par minmod, u = q/ρ_new borné par les voisins. Éclaboussure : énergie +0,4 % (MacCormack : +10…78 %).
`maccormack` reste désactivé (non conservatif avec ρ_eau/ρ_air = 1000).

## Portage GPU (wgpu / WGSL, f32)
* GPU RÉSIDENT : l'état reste sur le GPU ; ~256 octets relus par série de pas (dt, itérations, drapeaux), l'image rendue sur le GPU,
  l'état complet relu seulement pour les statistiques, les outils et la sauvegarde. Un seul buffer de stockage `S` (tranches aux
  décalages `O_*` générés par l'hôte) et un uniforme de paramètres : pas de limite de nombre de buffers.
* Mêmes formules que le CPU, en f32 : jusqu'à 32 fluides, groupes de phases, chimie et flammes épaissies, rayonnement, Dalton,
  ébullition et cavitation, cellules spéciales, entrées de bord, grains (frottement, percolation, solidification). Pas de temps,
  Newton + gradient conjugué multigrille, sous-pas et filet de sécurité décidés sur le GPU. Les scènes avec des corps rigides se
  calculent sur le CPU (`gpu_unsupported_reason`).
* Accord avec le CPU vérifié par `mp_gpu_tests.rs` : un pas parti du même état, puis des durées entières (MESURES.md § 6). Un seul
  périphérique wgpu par processus (`shared_gpu`), accès un fil à la fois (`GPU_IO`).
* Limites f32 connues : boîte fermée entièrement remplie de liquide, fermeture à ~1e-7 près (BUGFIXES.md).

## Interface (egui, `src/window/`)
`starting_window.rs` (réglages, scénario, matériaux), `multiphase_window.rs` (scénarios, pinceaux de matériaux, cellules spéciales,
bords en direct, options physiques, bascule CPU / GPU, fils CPU, profil du pas, état complet), `material_editor.rs` (matériaux,
mélanges, réactions, paires de fusion), `experiment_window.rs` (protocoles d'expérience). Le rendu et les scénarios sont dans
`mp_view.rs`, sans egui (testés). Convention : j croît vers le haut ; l'affichage inverse l'axe y.

## Limites connues (honnêtes)
* CPU : f64, référence. GPU : voir ci-dessus.
* Parois en glissement libre ; pas d'angle de contact ; tension liquide–liquide non modélisée (seulement liquide–gaz).
* Vapeur et air ne se mélangent pas (pas de pression partielle / évaporation dans l'air humide) : l'ébullition utilise la pression totale.
* Une seule T et une seule p par cellule (équilibre instantané entre phases d'une cellule mixte). cv(liquide) ≈ cp. Pas de chauffage visqueux. B(T) de Tait constant (c(80 °C) ≈ 1500 m/s au lieu de 1555).
* p_sat de Clausius–Clapeyron/Kirchhoff : ±4 % par rapport aux tables (20–150 °C).
* L'hydrostatique initiale suppose la gravité selon −y, colonnes depuis le bord haut.
* Pas d'entrée dans `benchmarks.rs`.
