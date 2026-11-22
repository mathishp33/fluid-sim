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
- implicite : (ρ_f/dt)(u − u_f) = ∇·[μ_g(∇u + ∇uᵀ)], gradient conjugué préconditionné (Jacobi), 40–150 itérations ; u_f = u* − dt∇pⁿ/ρ
  (le frottement équilibre la force NETTE, sinon un tas au repos glisse d'un dt·g par pas) ; parois : non-glissement pour le frottement ;
- la viscosité des fluides reste explicite (inchangée au bit près sans grains). La viscosité implicite générale (miel) reste à faire.

Validation (φ = 32°) : colonne 4 × 8 cm → tas aux flancs de 24.7° (inertie : un dépôt d'effondrement est sous l'angle de talus),
hauteur 0.45 H₀, étalement ×4 ; sans frottement : nappe à 5°. Tas à 20° : tient (Δh < 1 mm en 1 s). Tas à 50° : 42° en 0.4 s, puis
flue jusqu'à 32.6° en 4 s (la fin est plus lente qu'une avalanche réelle : viscosité régularisée retardée d'un pas). Indépendant de δ
(×10) et du plafond (÷3) à 1° près. GPU : mêmes formules, accord d'un pas 2e-7, tas final 24.8° / 3.58 cm (CPU 24.7° / 3.59 cm).

## 6. Réactions hétérogènes (étape 4)

Phases granulaires admises dans les réactions, vitesse proportionnelle à la surface des grains (6 α_s/d) : combustion du charbon
(C + O2 → CO2), poudre à canon (décomposition KNO3/S/C → gaz chauds, oxydant intégré : brûle sans air), pyrolyse du bois
(bois → gaz volatils + charbon). Réutilise la cinétique, la thermochimie et les flammes existantes.

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
3. **FAIT (2026-11-22) — Frottement des grains (tas)**, CPU et GPU (voir § 5). Reste : viscosité implicite générale (miel),
   frottement contre les obstacles solides (glissement libre aujourd'hui), chaleur de frottement.
4. Réactions hétérogènes et matériaux (charbon, poudre à canon, bois).
5. Solidification.

## 9. Choix à valider

- Groupe thermique propre pour chaque phase granulaire (recommandé) plutôt que partagé avec le fluide porteur.
- Friction (étape 3) : modèle de Coulomb régularisé (angle de talus à quelques degrés près), plutôt que la rhéologie μ(I) complète.
- Chaque étape d'abord sur le CPU, puis portée sur le GPU une fois validée (recommandé), plutôt que les deux à la fois.
