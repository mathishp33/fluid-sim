# Solveur multiphasique (CPU, f64) — notes d'intégration

## Fichiers (dans le même dossier que `materials.rs`)
`phases.rs`, `vof.rs`, `multiphase.rs`, `mp_advect.rs`, `mp_pressure.rs`, `mp_force.rs`, `mp_thermo.rs`, `mp_step.rs`, `mp_tests.rs`
GPU : `mp_gpu_params.rs`, `gpu_multiphase.rs`, `multiphase.template.wgsl` (généré par `python3 build_wgsl.py` à partir de `mpgpu/*.wgsl`), banc de test `gpu_run.js`

```rust
// simulation/mod.rs
pub mod phases;  pub mod vof;  pub mod multiphase;
pub mod mp_advect; pub mod mp_pressure; pub mod mp_force; pub mod mp_thermo; pub mod mp_step;
pub mod mp_gpu_params; pub mod gpu_multiphase;   // GPU (wgpu + pollster + bytemuck)
#[cfg(test)] mod mp_tests;
```
Le module est **indépendant** de `FluidSim` (qui reste inchangé) : même grille MAC, mais variables propres
(masses partielles par phase, énergie interne). Aucune modification de `cpu.rs/gpu.rs/fluid.rs` n'est nécessaire.

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
| Pression | faiblement compressible **implicite** (Helmholtz, κ_eff du mélange à entropie constante, Newton + PCG/IC(0)) : dt non limité par le son |
| Forces | gravité, viscosité (tenseur symétrique, μ variable), tension de surface CSF + courbure par fonctions de hauteur |
| Énergie | énergie interne conservative, conduction (k harmonique, fluides + solides), travail −p∇·u |
| Changement de phase | p_sat(T) Clausius–Clapeyron avec chaleur latente de Kirchhoff cohérente avec les enthalpies (h_v − h_l = L(T) exactement) ; interface résolue : ancrage de Hardt–Wondra à T_sat ; germes : Schnerr–Sauer ; limité par la masse, l'énergie et le retour à p = p_sat |
| Solides | cellules étanches avec ρ, cp, k propres (conduction fluide↔solide) |
| Bords | parois (glissement libre) ou ouverts : `Bc::Open { p }`, p au point le plus haut du domaine, la pression du bord suivant l'hydrostatique du fluide AMBIANT (`open_face_pressure`) ; ce qui entre est l'ambiant à `t_ambient`. À pression constante, une boîte d'air ouverte sur les côtés sous la gravité se mettait à 9 m/s (ρ g H d'écart entre le haut et le bas d'un côté) |

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
* Un seul buffer de stockage `S` + un uniforme de 512 octets : pas de limite de nombre de buffers ; taille limitée par `max_buffer_size` (~128 Mo ⇒ ≲ 380²–512²).
* Les noyaux WGSL ont été EXÉCUTÉS (Chromium headless/WebGPU SwiftShader) et comparés au CPU : hydrostatique, barrage (12 et 60 pas), Laplace, ébullition (≈4 % d'écart de vitesse après 15 pas : sensibilité chaotique f32).
* `gpu_multiphase.rs` (hôte wgpu) : **NON compilé ni exécuté** (pas de crate wgpu hors ligne). Il reproduit l'orchestration de `gpu_run.js`. Attendez-vous à des ajustements d'API mineurs selon votre version de wgpu.
* Pression : PCG-Jacobi à nombre d'itérations FIXE (`pcg_iter`, défaut 6·max(nx,ny)) ; converge mal sur grandes grilles (un multigrille serait préférable).
* Limites GPU : ≤ 4 phases, ≤ 4 solides, changement de phase sur la première paire seulement, pas de diagnostic de flux aux frontières, relecture complète de S à chaque pas (lent ; à remplacer par une relecture partielle/périodique pour l'affichage), dt choisi côté CPU.

## Intégration egui
Nouveaux fichiers : `simulation/mp_view.rs` (scénarios, pinceaux, rendu RGBA, stats — SANS egui, compilé et testé), `window/multiphase_window.rs`
(fenêtre egui + enum `SimWindow`), `window/starting_window.rs` (remplacement complet : case « Multiphase solver », choix du scénario).
Branchement (3 lignes) :
```rust
// simulation/mod.rs
pub mod mp_view; pub mod mp_gpu_params; pub mod gpu_multiphase;
// window/mod.rs
pub mod multiphase_window;
// là où vous créez la fenêtre de simulation (main.rs / app) : FluidWindow → SimWindow
use crate::window::multiphase_window::SimWindow;
let win = SimWindow::new(&settings);        // choisit FluidWindow ou MultiphaseWindow selon settings.multiphase
match win.ui(ctx) { WindowAction::BackToSettings => ..., _ => {} }   // même WindowAction qu'avant
```
`main_window.rs` n'est pas modifié. `config.json` : deux champs ajoutés (`multiphase`, `mp_scenario`), absents ⇒ valeurs par défaut.
Interface : scénarios (barrage, goutte, ballottement, ébullition sur plaque, huile/eau/air, cuve vide), affichages (phases, pression, T, vitesse, ρ, vapeur),
pinceaux (ajouter air/eau/vapeur/huile à T choisie, solides acier/verre, effacer, chauffer, pousser), bords ouverts/fermés en direct, options physiques,
vitesse visée ×temps réel (le pas est adaptatif, la fenêtre calcule autant de pas que le budget de 14 ms le permet), bascule CPU/GPU, stats par phase, infos de cellule.
Statut : `mp_view.rs` testé (6 scénarios exécutés et rendus, outils de pinceau) ; `multiphase_window.rs`, `starting_window.rs` et `gpu_multiphase.rs`
NON compilés ici (egui/wgpu absents) — ils n'utilisent que des appels egui déjà présents dans `main_window.rs`, sauf `ComboBox::from_label`/`Window::open` identiques.
Limites : bord « Inlet » non géré en multiphasique ; gravité 0 de l'écran de démarrage ⇒ 9,81 ; mode GPU : relecture complète à chaque pas.

## Limites connues (honnêtes)
* CPU : f64, référence. GPU : voir ci-dessus.
* Parois en glissement libre ; pas d'angle de contact ; tension liquide–liquide non modélisée (seulement liquide–gaz).
* Vapeur et air ne se mélangent pas (pas de pression partielle / évaporation dans l'air humide) : l'ébullition utilise la pression totale.
* Une seule T et une seule p par cellule (équilibre instantané entre phases d'une cellule mixte). cv(liquide) ≈ cp. Pas de chauffage visqueux. B(T) de Tait constant (c(80 °C) ≈ 1500 m/s au lieu de 1555).
* p_sat de Clausius–Clapeyron/Kirchhoff : ±4 % par rapport aux tables (20–150 °C).
* L'hydrostatique initiale suppose la gravité selon −y, colonnes depuis le bord haut.
* Pas d'entrée dans `benchmarks.rs`.
