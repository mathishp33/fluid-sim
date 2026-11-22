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
  bleue vers 1000 K) ; carbone conservé à 1e-6 (charbon + CO + CO2 + sorties). GPU : carbone brûlé à 3 % du CPU.
- corrigés en route : une cellule de surface partiellement remplie, rigide (frottement) mais non poreuse, scellait le lit (plus aucun
  tirage) ; la chimie donnait le reste d'énergie du sous-système réactif au DERNIER groupe — une trace de charbon dans le gaz prenait
  l'arrondi f32 de toute l'énergie (7000 K, pas refaits en boucle sur le GPU) : il va au groupe de plus grande capacité.
- reste : fluidisation (la pression de contact lithostatique ne baisse pas sous un soufflage vers le haut), feu « vif » : la flamme
  visible d'un feu de charbon vient surtout des matières volatiles (pyrolyse : bois, houille), poudre à canon (KNO3/S/C → gaz chauds,
  oxydant intégré : brûle sans air), pyrolyse du bois (bois → gaz volatils + charbon).

## 7. Solidification (étape 5)

Changement d'état liquide ↔ phase granulaire (eau ↔ cristaux de glace, cire, lave) à la température de fusion, avec chaleur latente :
réutilise la mécanique du changement de phase (T_fusion au lieu de T_sat(p)). La glace (917 kg/m³) remonte d'elle-même par la dérive.

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
4. **EN PARTIE (2026-11-22) — Réactions hétérogènes** : réactions de surface, charbon (CO, CO2, Boudouard), flamme de CO, percolation
   et grilles, braséro (§ 6). Reste : fluidisation, matières volatiles (pyrolyse), poudre à canon, bois.
5. Solidification.

## 9. Choix à valider

- Groupe thermique propre pour chaque phase granulaire (recommandé) plutôt que partagé avec le fluide porteur.
- Friction (étape 3) : modèle de Coulomb régularisé (angle de talus à quelques degrés près), plutôt que la rhéologie μ(I) complète.
- Chaque étape d'abord sur le CPU, puis portée sur le GPU une fois validée (recommandé), plutôt que les deux à la fois.
