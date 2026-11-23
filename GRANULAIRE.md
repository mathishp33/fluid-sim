# Phase granulaire (poudres, grains) — conception

Objectif : sable, poudre à canon, charbon, sciure, cendres, cristaux de glace… qui se déplacent avec l'écoulement, tombent, se
déposent, s'empilent, chauffent et brûlent. Choix retenu (2026-11-22) : une **phase granulaire eulérienne** dans le solveur
multiphasique, plutôt que des corps rigides (objets nets : plus tard, si besoin) ou des grains suivis un par un (trop coûteux).

## 1. Principe : une phase « dispersée » de plus

- Nouveau genre de phase `PhaseKind::Granular`, à côté de `Gas` et `Liquid`. Propriétés : masse volumique du grain ρ_s, c_p,
  conductivité, **diamètre des grains d**, **compaction maximale α_max** (≈ 0.60, empilement aléatoire), angle de frottement φ
  (étape 3), couleur.
- Les grains ne se compriment pas : équation d'état de liquide très raide (Tait). La fermeture Σ m/ρ = 1 s'applique telle quelle :
  une cellule de poudre tassée contient α_s ≈ 0.6 de grains et 0.4 du fluide des pores.
- Une seule vitesse par cellule (modèle de mélange, comme aujourd'hui) : les grains suivent l'écoulement, **plus une vitesse de
  dérive** (sédimentation). Limite assumée : pas d'inertie propre des grains (un grain lourd projeté par un jet ne « file » pas
  plus loin que le gaz qui le porte) — un modèle à deux vitesses serait bien plus lourd.
- Chaque phase granulaire forme **son propre groupe** (sa propre température), marqué **dispersé** : jamais d'interface reconstruite.

## 2. Transport sans interface

Une poudre dans l'air n'a pas de surface nette : c'est un mélange. La reconstruction PLIC chercherait à la séparer du gaz.

Dans le balayage PLIC (`plic_sweep`, et son miroir GPU), les groupes dispersés sont servis **en premier**, répartis uniformément
dans la donneuse : flux de volume f_g = ξ · α_g (reconstruction MUSCL de α_g, comme les compositions miscibles). Le reste,
ξ · (1 − Σ α_dispersés), est réparti entre les groupes non dispersés par la géométrie PLIC habituelle (renormalisée sur la partie
non dispersée de la cellule). Conséquence : le sable se déplace dans l'air comme dans l'eau et traverse l'interface eau / air.

## 3. Sédimentation (dérive)

Opérateur séparé après l'advection (comme la diffusion des espèces), explicite, conservatif :
- sur chaque face, dans le sens de la gravité, un volume de grains passe de la cellule du haut à celle du bas et **le même volume**
  de fluide (celui de la cellule du bas, sa composition et son énergie) remonte : masse, énergie et fermeture conservées ;
- vitesse terminale d'un grain isolé, Stokes → Newton en une formule (Ferguson & Church 2004) :
  w_t = R g d² / (18 ν + √(0.75 · 0.4 · R g d³)), R = (ρ_s − ρ_f)/ρ_f, fluide = contenu non granulaire de la cellule ;
  signe : un grain plus léger que le fluide (glace dans l'eau) remonte ;
- entravement (Richardson–Zaki) : w = w_t (1 − α_s)^n, n ≈ 4.65 (bas Reynolds) → 2.4 (haut) ; flux borné par la place restante
  dans la cellule receveuse (α_max − α_s) : la poudre s'arrête en couche tassée ;
- pas de temps : w·dt ≤ 0.5·dx (ajouté à `choose_dt`).

## 4. Thermique

Groupe propre, échange avec les autres groupes de la cellule par la surface des grains (au lieu de l'interface plane actuelle) :
aire volumique A = 6 α_s / d, h = Nu·k_f/d avec Nu = 2 (Ranz–Marshall, petit Reynolds) → G = 12 α_s k_f / d² (W/(m³·K)).
Fine poudre : équilibre quasi immédiat avec le gaz ; gros grains : retard thermique réel. Conduction entre cellules comme aujourd'hui.

## 5. Friction et tas (étape 3, faite : `mp_friction.rs`, `mp_gpu/friction.wgsl`)

Sans friction, un tas s'étale comme un liquide lourd (sable tassé + air : ρ ≈ 1600 kg/m³, μ ≈ 6e-4 Pa·s). Modèle de Coulomb continu
(Drucker–Prager régularisé, comme les simulations d'effondrement de colonnes de Lagrée, Staron & Popinet 2011) :
μ_g = tan φ · p_s / (2 √(|D|² + δ²)), |D| = √(½ D:D), δ = 1e-3 s⁻¹, plafonnée à 1000 ρ dx²/dt (borne des itérations).
- cellules concernées : tassées (α_s ≥ α_max − 0.1), ou partiellement remplies posées sur une cellule tassée (surface d'un tas), au
  prorata α_s / α_max — sans cela la surface d'un tas s'écoule comme un liquide et le tas finit à plat ;
- **pression de contact p_s = pression du mélange (contrainte moyenne) − pression du fluide des pores** (hydrostatique depuis le haut
  du massif ; dans l'eau : poids déjaugé). Le poids des grains au-dessus (contrainte verticale) avait été essayé d'abord : il rend le
  sable cohésif — une falaise verticale tient dès que tan φ > ½ (colonne figée à 68°) ;
- implicite : (ρ_f/dt)(u − u_f) = ∇·[μ(∇u + ∇uᵀ)], gradient conjugué préconditionné (Jacobi), 40–150 itérations ; u_f = u* − dt∇pⁿ/ρ
  (le frottement équilibre la force NETTE, sinon un tas au repos glisse d'un dt·g par pas), faces fixes (parois, solides) mises à 0 —
  u* y porte encore dt·g, annulé par la projection : ce défaut cisaillait à tort la couche du fond et faisait fluer les tas ;
- non-glissement du frottement contre les parois ET les obstacles solides (nœuds « paroi » : deux cellules bloquées alignées) ;
- chaleur de frottement Φ = τ : D rendue aux grains (54 % de l'énergie mécanique perdue par une colonne qui s'effondre ; le reste part
  dans la dissipation numérique du transport) ;
- viscosité implicite des fluides (`implicit_viscosity`, case « Viscosité implicite ») : la part de μ au-delà de 0.2 ρ dx²/dt passe dans
  le même solveur (glissement libre aux parois, comme l'explicite), la borne de pas visqueuse disparaît — tourbillon de Taylor–Green à
  μ = 2 Pa·s : 30 pas au lieu de 70, énergie cinétique à 1e-4 de la loi e^(−4νπ²t/L²) ; sans fluide si visqueux, rien ne change.

Validation (φ = 32°) : colonne 4 × 8 cm → tas aux flancs de 21.4° (inertie : un dépôt d'effondrement est sous l'angle de talus),
hauteur 0.42 H₀ ; sans frottement : nappe à 5°. Tas à 20° : tient (|u| < 1e-4 m/s). Tas à 50° : **avalanche, 29.8° en 0.35 s**, puis
immobile. Étagère d'acier : 90 % du sable d'une colonne y reste. GPU : mêmes formules, accord d'un pas 3e-7 (masses), tas 21.7°,
étagère 90.3 % des deux côtés, tourbillon visqueux identique.

## 6. Réactions hétérogènes et percolation (étape 4, en partie faite : charbon, feu sur grille)

Une réaction dont un réactif est granulaire est une **réaction de surface** (`phases::Reaction`, `CellChem::surface_rate`,
`cc_surface_rate` sur le GPU) : vitesse cinétique S_v k_s Π c^o, S_v = 6 α_s/d, k_s = A T^b e^(−Ea/RT) (m/s), limitée par l'apport du
réactif gazeux jusqu'aux grains S_v k_d C_ox (k_d = Sh D_ox/d, Sh = 2) **partagé** entre les réactions du même couple (grain, oxydant) :
facteur commun D/(K + D), K = Σ ν_ox r_cin. Avant, une phase granulaire réagissait comme un gaz dissous (action de masse sur m/M).
Validé : vitesse = loi à 1e-5 près ; diamètre ÷ 2 → ×2.01 à 1000 K (cinétique), ×3.6 à 1800 K (diffusion, 4 à la limite).
- **charbon** (`Phase::coal`, char de houille ; cinétique de Field 1969, A = 596 m/(s·K), Ea = 149 kJ/mol) : C + ½O2 → CO et
  C + O2 → CO2 dans le rapport d'Arthur (1951) CO/CO2 = 2500 e^(−6240/T) (4.9 à 1000 K, 78 à 1800 K, vérifié à 1 % près) ; Boudouard
  C + CO2 → 2 CO (endothermique, Ea = 250 kJ/mol, 1e5 fois plus lent que C–O2 à 1073 K) dans la zone chaude privée d'oxygène ; le CO
  brûle dans le gaz (CO + ½O2 ⇌ CO2, catalysé par l'humidité de l'air).
- **percolation** (`mp_percolation.rs`, `mp_gpu/percolation.wgsl`) : le fluide traverse les pores des lits de grains EN CONTACT (tassés
  ou posés sur un appui) de 1 mm et plus et les grilles, loi d'Ergun (Kozeny–Carman K = d² ε³ / (150 (1 − ε)²) + terme inertiel),
  moteur −∇(p − p_s) + ρ_f g avec p_s = poids lithostatique des grains (au repos : rien ne coule). Résolue DANS la projection (conductance
  C/dx² ajoutée au système de pression, ∇·q au second membre ; un flux explicite serait instable, D ~ 500 m²/s), puis transport du fluide
  seul au pas suivant ; fermeture et travail de pression comptent ∇·(u + q). Validé : air soufflé à 0.3 m/s sous un lit de morceaux de
  1 cm → 0.298 m/s à travers le lit, perte de charge 222.1 Pa/m (Ergun 222.0), lit immobile ; lit au repos : |q| < 2e-5 m/s ; GPU
  identique (0.2978 m/s, 1016.07 Pa). Les lits de grains de moins de 1 mm (sable, poudres : K < 1e-9 m², rien ne passe) sont traités comme
  étanches : la percolation y alignerait la pression sur le poids vertical des grains, ce qui raidit le frottement (un tas de sable à 50°
  s'arrêtait à 42° au lieu de 30°).
- **grille** (cellule spéciale `MpEffect::Grille`, « Grille » dans les cellules) : arrête les grains (le tas y repose, comme sur un
  solide : non-glissement, faces fermées au mélange), laisse passer le fluide par percolation (trous, fraction ouverte).
- scénario « Feu de charbon sur grille (braséro) » : charbon en morceaux de 1.2 cm sur une grille entre deux parois, lit de 0.25 ly
  entièrement en braises (1450 K), air humide (1 %), haut ouvert. Tirage naturel à travers la grille ≈ 4e-3 m²/s (≈ 2–3 cm/s), oxygène
  consommé du bas vers le haut du lit (10 % → 0), CO de Boudouard au sommet qui brûle au-dessus des braises (≈ 2 % de CO, petite flamme
  bleue vers 1000 K) ; carbone conservé à 1e-6 (charbon + CO + CO2 + sorties). GPU : carbone brûlé à 3 % du CPU. Depuis la houille,
  le haut du lit est une charge de houille (voir plus bas).
- corrigés en route : une cellule de surface partiellement remplie, rigide (frottement) mais non poreuse, scellait le lit (plus aucun
  tirage) ; la chimie donnait le reste d'énergie du sous-système réactif au DERNIER groupe — une trace de charbon dans le gaz prenait
  l'arrondi f32 de toute l'énergie (7000 K, pas refaits en boucle sur le GPU) : il va au groupe de plus grande capacité.
- **pyrolyse** (réaction EN VOLUME, `Reaction::bulk` ; `pyrolysis_step`, `k_pyrolysis` sur le GPU) : le bois (`Phase::wood`, CH1.44O0.66,
  650 kg/m³, PCS 19.5 MJ/kg) se décompose en chauffant en charbon de bois (`Phase::charcoal`, 350 kg/m³ : 22 % de la masse, ~40 % du
  volume) et en gaz — CO 35 %, CO2 15 %, eau 15 %, CH4 12 %, H2 1.3 % (goudrons comptés en CH4 et H2). Vitesse : les trois voies de
  Shafizadeh–Chin (Thurner & Mann, chêne) en une loi, A = 3.7e6 s⁻¹, Ea = 108.8 kJ/mol, bornée par la conduction interne des morceaux :
  k_eff = k / (1 + k τ), τ = (d/2)² ρ c_p / (π² k) ≈ 66 s pour 2 cm. Opérateur à part, à la température du GRAIN : les produits naissent
  à la température du bois, qui leur cède exactement leur énergie (chaleur de pyrolyse +0.18 MJ/kg) ; premier ordre intégré exactement.
  Validé : vitesse = loi à 1e-6 près, 22.00 % de charbon de bois, CH4/CO = 0.6 ; GPU : débits à 1 % près. Deux défauts évités : dans la
  chimie commune (température commune des groupes qui réagissent), le bois froid se mêlait aux braises de sa cellule (gaz porté d'un
  coup à ~900 K, divergence) ; sans la borne de conduction, une maille de bois au contact des braises se décomposait en ~3 ms (mille fois
  son volume de gaz). Le charbon de bois brûle comme le charbon (mêmes réactions de surface).
- **houille** (`Phase::bituminous_coal`, CH0.8O0.08, 1300 kg/m³, PCS 34 MJ/kg, morceaux de 1.2 cm par défaut) : même pyrolyse en volume
  que le bois, vitesse de Badzioch & Hawksley (charbons bitumineux : A = 3.12e5 s⁻¹, Ea = 74 kJ/mol, 0.1 s⁻¹ à 600 K, 1 s⁻¹ à 700 K)
  bornée par la conduction interne (τ = 19 s pour 1.2 cm, 2 min pour 3 cm) → 65 % de char (`coal`, qui brûle ensuite en braises) et
  35 % de matières volatiles (CH4 21 %, CO 8 %, CO2 3 %, eau 2.6 %, H2 0.1 % ; goudrons comptés en CH4). Validé : vitesse = loi à 2e-6
  près, 65.01 % de char, CH4/CO = 4.675, PCS 34.0 MJ/kg. Froide, elle se conserve (loi extrapolée : 4e-8 s⁻¹ à 300 K).
- braséro avec houille (feu établi) : braises sur les 60 % du bas du lit, houille en noix (3 cm) déjà chaude (700 K) sur les 40 % du
  haut, flamme d'allumage au-dessus du lit (gaz à 1600 K sur 4 cm), épaississement de flamme modéré comme le feu de bois. La houille
  dégaze à 0.83 %/s (la chaleur met ~2 min à gagner le cœur d'un morceau de 3 cm) ; ses matières volatiles sortent du lit sans oxygène
  et brûlent en grande flamme au contact de l'air (2230 K à 40 × 40, sur 3 s) ; braises à 1450 K, tirage moyen 2.6e-3 m²/s ; carbone (charbon +
  houille + CO + CO2 + CH4 + sorties) conservé à 1e-6. GPU : premier pas à 1e-7 près (masses) ; sur 1 s, houille dégazée à 0.04 %,
  carbone passé dans le gaz à 0.2 %, CO2 formé à 0.3 %, flamme 2087 K (CPU 2102 K). Une charge FROIDE (293 K) fume : ses gaz,
  refroidis en traversant la charge, sortent sous le seuil d'allumage, l'oxygène des pores tombe à 0 en 1.3 s, le tirage s'annule.
  Corrigé au passage (GPU) : `k_pyrolysis` calculait 1 − e^(−ν k dt) en f32, qui n'a que 2 ou 3 chiffres justes pour k dt ~ 1e-5
  (houille : pyrolyse à 0.17 % du CPU en un pas) → forme sans annulation (`one_minus_exp`, à 1e-7 près).
- **rayonnement dans le tas** (`mp_granular::bed_rad_k`, dans `conduct` et `exchange_grains` ; `k_cond_prep`, `k_cond_step`,
  `k_exch_grains` sur le GPU) : à haute température, les morceaux d'un tas échangent surtout par rayonnement, qui y agit comme une
  conduction de conductivité k_rad = 4 σ ε/(2 − ε) d T_m³ (Damköhler, ε = 0.9), au prorata des parts tassées f = α/α_max des deux
  nœuds granulaires — 8.5 W/(m·K) pour des braises de 1.5 cm à 1450 K, 30× la conduction du charbon de bois. Entre cellules voisines,
  T_m³ = (T_a² + T_b²)(T_a + T_b)/4, linéarisation exacte de T_a⁴ − T_b⁴ (une braise chauffe un bois froid même s'il rayonne peu) ;
  entre deux populations mêlées d'une cellule, h = 4 σ ε/(2 − ε) T_m³ sur l'aire d'une population tournée vers l'autre
  (min(6 α_1/d_1 · α_2/α_s, 6 α_2/d_2 · α_1/α_s)), implicite à deux corps, énergie exactement conservée ; sous-pas de conduction bornés
  par 2 k_rad(T_nœud) f/C. Validé : rangée de bois à 293 K sur des braises à 1450 K → 3.18 W/(m·K) à travers la face (k_rad = 3.10, plus
  la conduction) ; braises et bois mêlés → 3.24e4 W/(m³·K) pour 3.20e4 attendus (dans la cellule, et vers les voisines : de même ordre à
  1 cm de maille pour des morceaux de 1.5–2 cm) ; GPU : échauffement du bois à 0.2–0.3 % du CPU. Effet : du bois froid posé sur les
  braises atteint 566 K au contact en 1 s, 1000 K en 4 s (346 K en 6 s sans rayonnement) ; mais une charge de deux tiers du tas, dans ce
  braséro sans cheminée, étouffe les braises (O2 du tas à 1 %, braises à 1250 K en 14 s, tirage nul) avant que ses gaz ne s'enflamment —
  comme un feu trop chargé : les deux scénarios partent d'un feu établi.
- scénario « Feu de bois sur grille » : braises de charbon de bois (1450 K), bûchettes de 2 cm déjà chaudes (750 K) au-dessus, flamme
  d'allumage ; parois basses (le panache aspire l'air par les côtés : entre de hautes parois il se remplit de gaz sans oxygène et rien ne
  brûle). Le bois se décompose (~1 %/s), ses gaz brûlent en flamme de diffusion vers 2000 K au-dessus du tas (sans suie ni rayonnement :
  un vrai feu de bois est vers 1300–1500 K) ; carbone conservé à 1e-6 ; GPU : bois décomposé à 0.1 %, CO2 formé à 3 %, flamme à 15 K.
  Épaississement de flamme MODÉRÉ (`flame_resolution` 0.25, facteur ~5 à 1 cm) : complet (~100) il éteint cette flamme de diffusion ;
  nul, une poche de gaz du bois et d'air brûle d'un coup dans une maille, qui « pompe » (volume ±50 % d'un pas à l'autre) et dont le
  reste de gaz, comprimé, monte à 2600–3800 K.
- corrigé au passage (tout le solveur, CPU et GPU) : croissance du pas de temps bornée à ×2 d'un pas au suivant (`DT_GROWTH` ; ×1.25 enchaînait les retours arrière du braséro). Après
  un pas court imposé par la chimie, les vitesses de la projection — calculées pour corriger l'erreur de volume en CE pas — servaient à
  l'advection d'un pas 4× plus long et vidaient la maille à 92 % (pic à 3786 K sous la grille du feu de bois).
- **poudre noire** (`Phase::gunpowder` : 75 % KNO3, 15 % charbon de bois, 10 % S, grains de 2 mm, 1800 kg/m³, α_max 0.55) : COMBUSTION
  d'une phase granulaire qui porte son oxydant, sans air (réaction en volume avec `burn_rate`, `burn_exponent`, `ignition_temp` ;
  `pyrolysis_step`, `k_pyrolysis` sur le GPU). La surface des grains régresse à r = 1.1 cm/s (p / 1 atm)^0.15 (loi de Vieille : 2.3 s
  par pouce à l'air libre, exposant 0.13–0.16 mesuré au-dessus de 0.3 MPa — la poudre noire est peu sensible à la pression) dès que
  les grains, ou le fluide qui les baigne (`fluid_contact_temp`), atteignent 580 K (auto-allumage vers 580 °F ; la surface d'un grain
  baigné de gaz de combustion s'allume en ~1 ms, la chaleur met ~3 s à gagner le cœur d'un grain de 2 mm) ; grains consommés à 6 r / d
  (33 s⁻¹ à 1 atm, premier ordre exact). Produits nés à la température adiabatique (énergie interne quadratique : racine exacte,
  `adiabatic_t`) : par 100 g, 0.60 CO2 + 0.15 CO + 0.371 N2 + 0.05 H2O + 0.03 H2 (42 % de gaz, 12.0 mol/kg) et 58 % de fumée blanche
  (`Phase::powder_smoke` : K2CO3, K2SO4, K2S en particules de 2 µm) — Noble & Abel : 43 % de gaz, 274 cm³/g, 725.7 cal/g. Chaleur
  d'explosion 3.02 MJ/kg, flamme à volume constant 2671 K, « force » n R T_v = 267 kJ/kg. Borne du pas (`burn_dt_bound`, `k_burn_dt`) :
  au plus le volume d'une cellule de gaz produit par pas — une maille de poudre tassée en dégage ~8e4 fois son volume par seconde.
  Validé : vitesse = 6 r / d à 1e-6 près sous 1 et 10 atm (×10^0.15), énergie conservée à 1e-12 ; vase clos (10 kg/m³ dans l'azote à
  700 K) : T finale 2846.6 K et p 32.49 bar = état isochore attendu, énergie à 1e-9 ; GPU : premier pas à 1e-7 près, même débit de
  combustion et même pression (30.93 bar) au chiffre imprimé près. Limite : sous 30 bar, le rayonnement optiquement mince des gaz chauds
  viderait la boîte de sa chaleur (le test le coupe).
- scénario « Tas de poudre noire » : petit tas (flancs à 34°, ~1 kg par mètre de profondeur à 0.8 m × 0.6 m) allumé sur son flanc
  gauche (700 K), haut ouvert. Les gaz chauds jaillissent des grains tassés à plusieurs centaines de m/s, traversent le tas et allument
  les grains qu'ils baignent : le tas fuse en boule de feu (gaz vers 2800 K, le CO et le H2 des produits finissent de brûler dans l'air)
  et laisse un nuage de fumée blanche. À 40 × 30 : 37 % de la poudre brûlée en 30 ms, fumée = 0.5804 de la masse brûlée (stœchiométrie
  exacte), masse conservée sorties comprises ; GPU : poudre brûlée et fumée à 0.2 % sur 10 ms. Limite (une seule vitesse par cellule,
  les grains n'ont pas d'inertie propre) : le souffle emporte avec les gaz la moitié du tas hors du domaine avant qu'elle ait brûlé.
- corrigé au passage (CPU) : l'inversion e → T d'un groupe (`temp_from_parts`) élevait au carré sa capacité ; pour une trace de 1e-165
  kg (poudre dispersée par le souffle jusqu'au bord du domaine), C² sortait de la plage f64, le discriminant s'annulait et T valait
  298 − c/s = −802 K : pas refusés en cascade. Forme adimensionnée, celle de `tau_from` sur le GPU (qui avait le même défaut en f32).
- reste : fluidisation (la pression de contact lithostatique ne baisse pas sous un soufflage vers le haut), séchage du bois, allumage
  d'une charge froide (il faudrait une cheminée, ou moins de bois, pour que le tirage nourrisse les braises), inertie propre des grains
  (le souffle d'une poudre emporte ses grains), délai d'allumage d'un grain dans un gaz chaud.

## 7. Solidification (étape 5, faite : eau ↔ glace)

Changement d'état liquide ↔ phase granulaire à la température de fusion, avec chaleur latente (`Registry::fusions`, paires `FusionDef`
de la bibliothèque ; `fusion_step`, `k_fusion` sur le GPU). Méthode enthalpique à la température des groupes, sans paramètre de vitesse :
un liquide plus froid que T_f gèle juste assez pour que sa chaleur latente le ramène à T_f (la glace naît à T_f), un solide plus chaud
fond juste assez pour revenir à T_f ; tout gèle (fond) si la chaleur latente de tout le liquide (solide) n'y suffit pas. La vitesse est
celle de l'arrivée de la chaleur (conduction, échange par la surface des grains). La chaleur latente est portée par l'énergie de
référence du solide (e_l(T_f) − e_s(T_f) = L_f, comme celle de la vapeur pour l'ébullition) ; masse et énergie exactement conservées ;
bilan de la masse gelée par paire (`frozen`). Opérateur à part plutôt que la mécanique liquide ↔ vapeur : ni pression partielle, ni
germination, ni volume de vapeur à borner.
- **glace** (`Phase::ice` : cristaux de 2 mm, 917 kg/m³ — ils remontent dans l'eau —, c_p 2.11 kJ/(kg·K) à 0 °C croissant de
  6.8 J/(kg·K²), k = 2.22 W/(m·K)) et paire eau ↔ glace (T_f = 273.15 K, L_f = 333.6 kJ/kg ; préréglage `with_ice`).
- validé : eau à −10 °C → 12.5498 % gelés (c ΔT / L_f), eau et glace à 273.150000 K ; glace à +10 °C → 6.4278 % fondus ; à −93 °C,
  tout gèle (glace à 245 K) ; énergie à 1e-12. Problème de Stefan (eau à 0 °C contre une paroi à −10 °C, mailles de 0.25 mm) :
  croissance de la glace entre 30 et 120 s = solution exacte à 0.2 % près (2.069 mm contre 2.064), épaisseur à une maille près (retard
  pris au départ : erreur en dx de la méthode enthalpique). GPU : premier pas à 1e-7 près (gel et fusion), même croissance à 0.007 %
  (pas de 2 ms ; à 20 ms, la pression de cette colonne d'eau presque incompressible est mal conditionnée en f32 et le GPU refait des
  pas).
- scénario « Glace pilée dans l'eau tiède » : un nuage de cristaux à −10 °C au fond d'une eau à 27 °C remonte vers la surface en
  fondant (la moitié en 5 s), l'eau de fonte froide coule ; énergie à 2e-9 ; GPU : fonte à 0.15 % sur 2 s.
- limites : le solide formé est un tas de grains (une couche de glace s'écroule à son angle de talus, pas de bloc rigide → solides
  mobiles) ; ni surfusion ni germination (l'eau gèle dès qu'elle passe sous T_f) ; eau sans maximum de densité à 4 °C ; cire, lave :
  matériaux à définir (une paire de fusion de plus dans la bibliothèque) ; à la surface, des cristaux dans une cellule mêlée d'air y
  « tombent » à leur vitesse dans l'air (~7 m/s : pas de 1.5 ms).

## 8. Étapes et validation

1. **FAIT (2026-11-22) — Phase granulaire, transport dispersé, sédimentation, thermique** — CPU (`mp_granular.rs`, `plic_sweep`,
   `PhaseKind::Granular`, matériau « sable », scénario « Sable qui tombe dans l'eau (sédimentation) », bouton « + Grains » de
   l'éditeur). Tests : vitesse de chute (Stokes à 20 µm, Newton à 3 mm, grain léger qui remonte), front d'eau claire d'une suspension
   à 10 % (1.00 cm descendu pour 0.97 prédit), couche tassée exactement à α_max = 0.6 (10 mailles, rien au-dessus), masses exactes et
   énergie à 1e-6, poudre chaude → τ de l'échange par la surface des grains, suspension uniforme qui le reste sous un tourbillon,
   nuage de sable : 99.4 % sous la surface de l'eau en 1.5 s.
2. **FAIT (2026-11-22) — Portage GPU de l'étape 1** (`mp_gpu/granular.wgsl`, `k_sweep_flux`, `k_an`, `k_exchange`, `k_red_dt`) :
   sédimentation en deux passes (incréments puis application, sans conflit d'écriture), mêmes formules et mêmes plafonds que le CPU.
   Accord d'un pas : 1e-5 (nuage de sable), 5e-8 (suspension), 1e-7 (poudre chaude) ; 1 s de nuage : 92 % du sable sous la surface
   des deux côtés ; 3 s de suspension : mêmes colonnes. Corrigé en route (aussi utile sans grains) : aire PLIC f32 stable,
   relaxation thermique d'une trace sans débordement. Limite f32 connue (pas propre aux grains) : boîte fermée sans aucun gaz.
3. **FAIT (2026-11-22) — Frottement des grains (tas)**, CPU et GPU (voir § 5), puis : avalanche rapide (un défaut des faces de paroi
   faisait fluer les tas), non-glissement contre les obstacles solides, chaleur de frottement, viscosité implicite des fluides (miel).
4. **EN PARTIE (2026-11-23) — Réactions hétérogènes** : réactions de surface, charbon (CO, CO2, Boudouard), flamme de CO, percolation
   et grilles, braséro, pyrolyse du bois et feu de bois, houille, rayonnement dans le tas, poudre noire (§ 6). Reste : fluidisation.
5. **FAIT (2026-11-23) — Solidification** (eau ↔ glace, § 7) : méthode enthalpique, Stefan, glace pilée qui fond, CPU et GPU.

## 9. Choix à valider

- Groupe thermique propre pour chaque phase granulaire (recommandé) plutôt que partagé avec le fluide porteur.
- Friction (étape 3) : modèle de Coulomb régularisé (angle de talus à quelques degrés près), plutôt que la rhéologie μ(I) complète.
- Chaque étape d'abord sur le CPU, puis portée sur le GPU une fois validée (recommandé), plutôt que les deux à la fois.
