# Solveur multiphasique (CPU, f64) — notes d'intégration

## Fichiers (dans le même dossier que `materials.rs`)
`phases.rs`, `vof.rs`, `multiphase.rs`, `mp_advect.rs`, `mp_pressure.rs`, `mp_force.rs`, `mp_thermo.rs`, `mp_step.rs`, `mp_tests.rs`

```rust
// simulation/mod.rs
pub mod phases;  pub mod vof;  pub mod multiphase;
pub mod mp_advect; pub mod mp_pressure; pub mod mp_force; pub mod mp_thermo; pub mod mp_step;
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
| Élément             | Modèle                                                                                                                                                                                                                                                             |
|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Gaz                 | gaz parfait, cv = cp − R, μ(T) Sutherland, k(T)                                                                                                                                                                                                                    |
| Liquide             | équation de Tait (eau : B = 3,07e8 Pa, n = 7,15 → c = 1482 m/s, K = 2,2 GPa), ρ(T) de Kell (maximum à 4 °C), μ(T) Vogel                                                                                                                                            |
| Interface           | VOF géométrique PLIC (formules analytiques), flux conservatifs, N phases (priorité liquides puis gaz)                                                                                                                                                              |
| Pression            | faiblement compressible **implicite** (Helmholtz, κ_eff du mélange à entropie constante, Newton + PCG/IC(0)) : dt non limité par le son                                                                                                                            |
| Forces              | gravité, viscosité (tenseur symétrique, μ variable), tension de surface CSF + courbure par fonctions de hauteur                                                                                                                                                    |
| Énergie             | énergie interne conservative, conduction (k harmonique, fluides + solides), travail −p∇·u                                                                                                                                                                          |
| Changement de phase | p_sat(T) Clausius–Clapeyron avec chaleur latente de Kirchhoff cohérente avec les enthalpies (h_v − h_l = L(T) exactement) ; interface résolue : ancrage de Hardt–Wondra à T_sat ; germes : Schnerr–Sauer ; limité par la masse, l'énergie et le retour à p = p_sat |
| Solides             | cellules étanches avec ρ, cp, k propres (conduction fluide↔solide)                                                                                                                                                                                                 |

## Validation (tests, `cargo test --release`)
hydrostatique (courants 5e-11 m/s, p_fond exact) · Δp = β/κ·ΔT de l'eau confinée (0,25 %) · ballottement ω² = gk·tanh(kh) (+1 %) ·
Laplace Δp = σ/R (0,1 %) · barrage (front 1,3·√(gH), masse exacte) · ébullition en boîte fermée (masse 1e-14, énergie à 0,4 % de L·Δm) ·
cavitation (p relâchée à p_sat, volume de vapeur = volume retiré) · eau/huile/air statique + goutte d'huile qui monte · éclaboussure (volume exact).
Les 20 tests de `FluidSim` passent toujours.

## Limites connues (honnêtes)
* **CPU seulement**, f64 ; pas de portage GPU/WGSL (étape suivante).
* Advection de vitesse semi-lagrangienne bicubique **non conservative en quantité de mouvement** (MacCormack testé : crée de l'énergie avec ρ_eau/ρ_air = 1000, donc désactivé). Une advection conservative (Rudman / Raessi–Pitsch) améliorerait les éclaboussures.
* Parois en glissement libre ; pas d'angle de contact ; tension liquide–liquide non modélisée (seulement liquide–gaz).
* Vapeur et air ne se mélangent pas (pas de pression partielle / évaporation dans l'air humide) : l'ébullition utilise la pression totale.
* Une seule T et une seule p par cellule (équilibre instantané entre phases d'une cellule mixte). cv(liquide) ≈ cp. Pas de chauffage visqueux. B(T) de Tait constant (c(80 °C) ≈ 1500 m/s au lieu de 1555).
* p_sat de Clausius–Clapeyron/Kirchhoff : ±4 % par rapport aux tables (20–150 °C).
* L'hydrostatique initiale suppose la gravité selon −y, colonnes depuis le bord haut.
* Pas d'intégration UI (egui) ni d'entrée dans `benchmarks.rs`.
