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

## 5. Friction et tas (étape 3)

Sans friction, un tas s'étale comme un liquide lourd (le mélange grains + air a ρ ≈ 900 kg/m³). Modèle viscoplastique régularisé
dans les cellules tassées (α_s > α_c ≈ 0.5) : μ_eff = μ_f + p_s · sin φ / (2|γ̇| + ε), p_s = pression de contact des grains.
Ces viscosités énormes imposent une **viscosité implicite** (aujourd'hui explicite : dt ≤ 0.2 ρ dx²/μ) — utile aussi au miel et aux
huiles très visqueuses. Validation : angle de talus d'un tas versé.

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
   nuage de sable : 99.4 % sous la surface de l'eau en 1.5 s. Le GPU refuse proprement ces scènes (repli CPU).
2. Portage GPU de l'étape 1 (accord pas à pas avec le CPU).
3. Viscosité implicite + friction (tas).
4. Réactions hétérogènes et matériaux (charbon, poudre à canon, bois).
5. Solidification.

## 9. Choix à valider

- Groupe thermique propre pour chaque phase granulaire (recommandé) plutôt que partagé avec le fluide porteur.
- Friction (étape 3) : modèle de Coulomb régularisé (angle de talus à quelques degrés près), plutôt que la rhéologie μ(I) complète.
- Chaque étape d'abord sur le CPU, puis portée sur le GPU une fois validée (recommandé), plutôt que les deux à la fois.
