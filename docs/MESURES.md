# Mesures — la simulation face aux valeurs de référence

Ce que vaut le simulateur, chiffres à l'appui : chaque ligne compare une grandeur calculée à une valeur de référence (solution
exacte, corrélation, expérience publiée, bilan). Les chiffres viennent des essais de `cargo test --release` (solveur multiphasique
CPU f64, sauf mention), dont le nom est donné quand il aide à les refaire ; le détail est dans MULTIPHASE.md, GRANULAIRE.md et
SOLID.md. Mis à jour le 2026-11-26.

Légende de la colonne « Écart » : **exact** = au chiffre imprimé ou à l'arrondi près ; un pourcentage = écart relatif à la référence ;
« sous-résolu » = résultat connu pour dépendre de la maille (voir la colonne « Conditions »).

## 1. Fluide au repos, ondes, interfaces

| Grandeur | Simulé | Référence | Écart | Conditions |
|---|---|---|---|---|
| Courants parasites d'une cuve d'eau au repos | 5e-11 m/s | 0 | — | hydrostatique ; pression au fond exacte |
| Montée de pression d'une eau confinée chauffée | Δp = β/κ ΔT | β/κ ΔT (dilatation / compressibilité) | 0.25 % | boîte fermée pleine d'eau |
| Pulsation du ballottement d'une surface libre | ω² | g k tanh(k h) | +1 % | mode fondamental |
| Saut de pression d'une goutte (Laplace) | Δp | σ / R | 0.1 % | goutte au repos |
| Pression de l'eau d'un barrage qui frappe la paroi | 1.010–1.132 bar | 1.1–1.4 bar (mesures de cette taille : 0.1–0.4 bar de surpression) | dans la plage | barrage 74 × 120, jusqu'à 0.8 s ; aucune vapeur (avant la correction de fermeture : jusqu'à 93 bar) |
| Erreur de remplissage de l'eau pure (barrage) | ≤ 0.28 % | 0 | — | même calcul |
| Masse d'eau d'un barrage, d'une éclaboussure | conservée | — | exact | |

## 2. Chaleur et changements d'état

| Grandeur | Simulé | Référence | Écart | Conditions |
|---|---|---|---|---|
| Énergie d'une ébullition en boîte fermée | bilan | L Δm | 0.4 % | masse conservée à 1e-14 |
| Cavitation | p relâchée à p_sat, volume de vapeur = volume retiré | — | exact | |
| Pression de vapeur saturante | Clausius–Clapeyron / Kirchhoff | tables (20–150 °C) | ±4 % | |
| Fraction gelée d'une eau à −10 °C | 12.5498 % | c ΔT / L_f | exact | méthode enthalpique |
| Croissance de la glace sur une paroi à −10 °C | épaisseur(t) | solution exacte de Stefan | 0.2 % | GPU : 0.007 % sur 5 s |
| Eau surfondue à −8 °C qui gèle sur un germe | 10.04 % gelés | c ΔT / L_f | exact | |
| Front de dendrites dans un canal à −10 °C | 7.82 cm/s | 7.94 cm/s (loi imposée v = 1 mm/s × ΔT^1.9 ; mesures : 7.7 cm/s) | −1.5 % | la dilatation du gel pousse un peu d'eau à 0 °C devant le front |
| Couche d'huile chaude dans l'air | équilibre en ~4 s | — | énergie à 1e-6 | une température par groupe de phases |

## 3. Combustion

| Grandeur | Simulé | Référence | Écart | Conditions |
|---|---|---|---|---|
| Vitesse de flamme CH4–air stœchiométrique | 0.384 m/s | 0.36–0.40 m/s (mesures ; cible du calage 0.38) | dans la plage | flamme épaissie, quelle que soit la maille |
| Température des gaz brûlés de cette flamme | 2231 K | 2226 K (adiabatique) | +0.2 % | sans radicaux, les flammes réelles sont ~50–90 K plus froides |
| Chaleur dégagée par la flamme de prémélange au dt par défaut | 252 | 270.5 (à dt ≤ 5e-6) | −7 % | 64 × 12, dx = 1 mm, 0.02 s (convergence en temps du 1er ordre) |
| Jet de méthane à l'air libre | 2.2e5 W/m | 2.0e5 W/m (combustion complète du méthane émis) | +10 % | 3 s, mailles de 6 mm (de même à 4 mm) |
| Chaleur d'explosion de la poudre noire | 3.02 MJ/kg | 3.04 MJ/kg (Noble & Abel) | −0.7 % | 75/15/10, 42 % de gaz |
| Poudre noire en vase clos | 2846.6 K, 32.49 bar | état isochore attendu | exact (énergie à 1e-9) | |
| Vitesse de combustion des grains | r = 1.1 cm/s (p / 1 atm)^0.15 | loi de Vieille | exact | sous 1 et 10 atm |
| Fumée laissée par la poudre | 0.5804 de la masse brûlée | 0.5804 (composition) | exact | GPU à 0.2 % |
| Carbone d'un feu de charbon, de bois, d'un braséro | conservé | — | 1e-6 | |
| Dérive d'énergie pendant une explosion | jusqu'à ~5 % de la chaleur dégagée | 0 | défaut connu | schéma en énergie interne (BUGFIXES.md) |

## 4. Grains, lits, tas

| Grandeur | Simulé | Référence | Écart | Conditions |
|---|---|---|---|---|
| Front d'eau claire d'une suspension à 10 % | 1.00 cm descendu | 0.97 cm (Richardson–Zaki) | +3 % | |
| Couche de sable tassée | α = 0.6 sur 10 mailles, rien au-dessus | α_max = 0.6 | exact | masses exactes, énergie à 1e-6 |
| Air soufflé à travers un lit de morceaux de 1 cm | 0.3000 m/s, 223.7 Pa/m | Ergun : 223.7 Pa/m | exact | percolation (Darcy dans la projection) |
| Sable de 0.3 mm dans l'air à 0.5 U_mf (U_mf = 8.9 cm/s) | 1507 Pa ; contact 1454 Pa | Ergun 1495 Pa ; poids − traînée 1466 Pa | +0.8 % ; −0.8 % | deux pressions (fluide des pores, contacts) |
| Même sable à 1.4 U_mf | +7.5 mm en 1 s, contacts à 0 | Kozeny : ~7 mm à l'équilibre | +7 % | |
| Sable de 2 mm dans l'eau à 0.84 U_mf (U_mf = 2.4 cm/s) | contact 419 Pa | 411 Pa | +2 % | perte de charge à 0.5 % |
| Sable de 2 mm dans l'air à 0.5 et 0.9 m/s | 2073 et 732 Pa | 2074 et 768 Pa | 0 et −5 % | U_mf = 1.08 m/s ; à 0.85 U_mf immobile, à 1.11 U_mf il monte de 4 cm en 0.5 s |
| Lit au repos | contact 1523 Pa | poids au-dessus : 1523 Pa | exact | |
| Lit fluidisé au-delà de U_mf | gonfle ×2, 4–6 m/s | ~×1.2, agitation modérée | défaut connu | une seule vitesse par maille (BUGFIXES.md) |
| Effondrement d'une colonne de sable 4 × 8 cm (φ = 32°) | flancs à 17°, 3.0 cm de haut | Lube et al. (2005) : ~17°, 3.5 cm | 0° ; −14 % | l'élan réel étale davantage le dépôt |
| Tas trop raide (50°) | avalanche jusqu'à 28.9° | sous l'angle de frottement (32°) | — | tas à 20° stable (|u| < 1e-4 m/s) |
| Conductivité de rayonnement à travers un tas de braises | 3.18 W/(m·K) | k_rad = 3.10 W/(m·K) (Damköhler) | +2.6 % | GPU à 0.2–0.3 % |
| Miel (viscosité implicite) | décroissance d'un tourbillon | loi visqueuse exacte | 1e-4 | 30 pas au lieu de 70 |

## 5. Corps rigides (solides mobiles, SOLID.md)

| Grandeur | Simulé | Référence | Écart | Conditions |
|---|---|---|---|---|
| Aire couverte par un corps, quels que soient position et angle | Σ α dx² | aire exacte | 1e-12 | fractions exactes de maille |
| Eau au repos autour de deux corps fixes | 1.8e-7 m/s | 0 | — | pression à 3 mPa près |
| Débit dans les passages d'un canal autour d'un disque | 0.284 m/s | 0.273 m/s | +4 % | débit identique entrée / disque / sortie |
| Piston qui comprime un tube d'air | γ = 1.394 (pression), 1.392 (température) | 1.40 (adiabatique) | −0.4 % ; −0.6 % | le piston d'acier prend un peu de la chaleur de compression (1.401 / 1.399 sans échange) |
| Corps de la densité de l'eau | 2 µm en 1 s ; force du fluide = poids | immobile | 6 chiffres | |
| Chute libre dans l'air | vitesse | g t | 0.1 % | |
| Masse ajoutée d'un disque lâché ou accéléré dans l'eau | C_a = 0.94–1.04 | 1 (cylindre) | ±6 % | |
| Tirant d'eau d'un flotteur (bois 0.6, liège 0.24) | | Archimède | ±0.7 mm | oscillation qui s'amortit |
| Période d'un pendule dans l'air | | pendule pesant | 0.03 % | |
| Traînée d'un disque FIXE, Re = 20 et 40 | C_D 1.83 et 1.38 | Dennis & Chang (1970) : 2.05 et 1.52 | −11 % ; −9 % | 10 mailles par diamètre |
| Traînée d'un disque TIRÉ, Re = 20 | C_D 2.68 (10 mailles/D), 1.61 (20) | 2.05 | +31 % ; −21 % | converge en maillage ; pression à 20 mailles 1.26 pour ≈ 1.24 |
| Vitesse limite d'un disque qui sédimente dans l'huile | 0.109 m/s | 0.142 m/s (Re visé 20) | −23 % | sous-résolu |
| Rebond d'un disque d'acier dans l'air | e = 0.803 | e = 0.8 (consigne) | +0.4 % | contacts souples |
| Choc frontal de deux disques | restitution 0.777 ; quantité de mouvement × 0.999 | 0.8 ; 1 | −3 % ; 0.1 % | |
| Disque posé au fond, blocs empilés, bloc sur une étagère | 2.6e-6 m/s ; 1e-5 m/s ; 0.00° | immobiles, à plat | — | aucun pas refait |
| Corps chaud et même carré en mailles solides d'acier | refroidissements | identiques | 0.38 % | 10 s, conduction seule ; énergie à 2.7e-4 |
| Échange d'un disque d'acier à 80 °C dans l'eau | h ≈ 250 W/(m²·K) | Churchill–Chu (cylindre, régime établi) : ≈ 1400 | sous-résolu | 10 mailles par diamètre : couche limite thermique (≈ 0.5 mm) plus fine que la maille |

## 6. GPU (f32) face au CPU (f64)

Le GPU refait les mêmes formules en f32 ; on compare un pas parti du même état, puis des durées entières.

| Scène | Accord d'un pas | Sur la durée |
|---|---|---|
| Nuage de sable qui tombe dans l'eau | 1e-5 | 1 s : 92 % du sable sous la surface des deux côtés |
| Suspension, poudre chaude | 5e-8 ; 1e-7 | 3 s : mêmes colonnes ; écart grains − air à t = τ : 80.108 K des deux côtés |
| Lit fluidisé, lit à deux pressions | 1.5e-7 ; 5e-3 Pa sur p et σ' | même perte de charge |
| Glace (Stefan), glace pilée, eau surfondue | 1e-7 | 0.007 % sur 5 s ; 0.15 % ; front à 1.4 % |
| Poudre noire, braséro, feu de bois | au chiffre imprimé près | 0.2 % ; 0.04–0.3 % ; 0.1–3 % |
| Explosion H2 confinée | 2e-2 (une cellule qui s'enflamme franchit un seuil chimique d'un côté de l'arrondi) | ~1e-6 ailleurs |
| Chauffage à 2000 °C, ébullition sur plaque | 5e-3 ; 5e-4 (bulles sous-maille à un seuil d'arrondi) | ordre de grandeur (le CPU contre lui-même, perturbé de 1e-9, varie autant) |

## 7. Vitesse de calcul

Machine de développement : Ryzen 7 (8 cœurs, 16 fils), Radeon intégrée (Vulkan).

| Scène | CPU f64 (ms/pas) | GPU f32 (ms/pas) |
|---|---|---|
| Barrage 80 × 60 / 320 × 240 | 40 / 469 | 5.2 / 24.6 (4.5 / 19.9 par séries de 8 pas) |
| Ébullition 80 × 60 / 320 × 240 | 41 / 597 | 9.3 / 31.3 |

Le CPU va 1.5 à 2.3 fois plus vite sur 8 fils quand la machine ne bride pas les cœurs (barrage 160 × 120 : 48 ms sur 1 fil, 33 sur 8),
avec un résultat identique au bit près quel que soit le nombre de fils.

## À mesurer

- Front du barrage face aux expériences de Martin & Moyce (1952).
- Rebond d'une sphère dans un liquide selon le nombre de Stokes (Gondret et al. 2002).
- Convection naturelle autour d'un corps chaud à 20 mailles par diamètre (Churchill–Chu).
- Lit fluidisé : expansion de Richardson–Zaki au-delà de U_mf.
