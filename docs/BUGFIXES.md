# Défauts — connus et corrigés

Les défauts du simulateur : d'abord ceux qui restent (symptôme, cause connue ou supposée, parade), puis ceux qui ont été corrigés,
du plus récent au plus ancien (symptôme → cause → correction, avec le commit). Les chiffres de validation sont dans MESURES.md, les
tâches à venir dans TODO.md. Mis à jour le 2026-11-26.

## 1. Défauts connus

### Fluide et changements d'état
- **Ébullition en film absente (Leidenfrost)** : huile à 2000 K sur de l'eau, lave qui entre dans la mer — la vapeur n'isole pas le
  liquide chaud de l'eau, les explosions de vapeur sont trop violentes (lave : pointes à ~200 m/s, 4 % projetés hors du domaine en
  1.5 s). Scénario « coulée de lave dans la mer » écarté pour cela.
- **Couplage du pas χ heuristique** : une petite phase non négligeable (eau à α ~ 1e-4 dans du gaz) fait traiter toute la cellule à
  une température (fermeture, travail de pression, chaleur latente).
- **Vapeur « verrouillée » sur son liquide** au-delà de 50 % de la masse de son groupe : seuil arbitraire.
- **Poches de gaz très détendues** (quelques centaines de Pa) refroidies jusqu'au plancher de 60 K ; la conduction des voisines
  devrait les réchauffer plus vite.
- **Peu de types de bords** : ouvert à pression imposée (hydrostatique), entrée à vitesse imposée, paroi ; ni sortie à vitesse
  imposée, ni bords périodiques, ni sortie convective.
- **Cuve vide ouverte en haut** (pas un défaut, un piège) : de l'eau qui l'emplit jusqu'au bord en sort en se dilatant, avec son
  énergie comptée depuis 0 K ; les bilans d'énergie le voient (laisser une couche d'air en haut).

### Combustion
- **Dérive d'énergie pendant une explosion** : jusqu'à ~5 % de la chaleur dégagée (schéma en énergie interne, travail de pression
  discret).
- **Flammes ~50–90 K trop chaudes** : pas de radicaux (OH, H, O) ; **pas de suie** (flammes de diffusion sans couleur ni rayonnement).
- **Flamme épaissie au dt par défaut** : ~7 % d'écart en temps (découpage de Lie diffusion ↔ réaction, couplage avec l'écoulement).
- **Poches de combustible et d'air qui s'enflamment d'un coup** dans une maille (braséro, feu de bois : pointes de 3000–3600 K).
- **Charge FROIDE posée sur des braises** : dans le braséro sans cheminée, le tirage est trop faible, elle étouffe les braises ; les
  scénarios partent d'un feu établi.
- **Jet de méthane** : le pied de la flamme s'étale sur le fond à ~4 cm de la buse au lieu de s'accrocher à son bord.
- **Poudre noire** : allumage immédiat au contact du gaz chaud (au lieu de ~1 à quelques dizaines de ms) ; le souffle d'un tas emporte
  ses grains avec les gaz (une seule vitesse par maille), la moitié du tas sort du domaine avant d'avoir brûlé.

### Grains
- **Lit fluidisé trop agité** au-delà de U_mf : il gonfle ×2 au lieu de ~×1.2, s'agite à 4–6 m/s, sa perte de charge dépasse son poids
  de 2 à 3× ; dans une colonne étroite il monte en bouchons. Une seule vitesse par maille.
- **Lits perméables** : seuil K/μ ≥ 5e-7 m²/(Pa·s) (sable de 0.1 mm et plus dans l'air, de 0.6 mm et plus dans l'eau) ; en dessous, le
  lit est étanche.

### Corps rigides
- **Convection naturelle sous-résolue autour d'un corps** à 10 mailles par diamètre : échange thermique 3 à 6 fois trop faible
  (h ≈ 250 W/(m²·K) pour ≈ 1400).
- **Traînée d'un corps en mouvement** approchée à 10 mailles par diamètre (+30 % à Re 20), frottement trop faible à 20 mailles.
- **Bruit de pression des corps en mouvement dans l'eau** : le fluide balayé va à une seule voisine.
- **Niveau de pression d'une cuve fermée pleine de liquide** qui oscille de quelques kPa sous un corps libre.
- **Masse ajoutée apparente jusqu'à 1.4 au démarrage**, quand le pas double d'un pas à l'autre.
- **Choc de deux corps libres** : la vitesse de l'autre corps est celle du pas précédent (restitution 0.777 pour 0.8).
- **Formes convexes seulement** ; un corps posé reste à ~0.3 maille de ce qui le porte (peau de contact, voulu).

### GPU (f32)
- **Boîte FERMÉE entièrement remplie de liquide, sans gaz** : diverge en ~10 pas (Newton au plafond, p jusqu'à 1e9 Pa) ; le CPU
  f64 tient, la moindre poche de gaz suffit (`gpu_closed_water_box_trace`).
- **Fermeture connue à ~1e-7 près** (masses en f32) : vapeur bruitée de ~1 % dans une cellule presque pleine de liquide.
- **Corps rigides** : pas encore sur le GPU (les scènes avec corps se calculent sur le CPU).

### Numérique
- **Advection PLIC** : flux appliqués face par face dans l'ordre des indices ; la forme « flux pure » diverge sur un jet d'eau dans
  l'air.
- **Suite de tests longue** (~65 min sur la machine de développement) : essais GPU un par un, mémoire juste (8 Go pour les essais).

## 2. Défauts corrigés

### Bords ouverts (2026-11-26)
- **Vent parasite des bords ouverts** (multiphasique) : de l'air au repos dans une cuve aux côtés ouverts se mettait en mouvement,
  de plus en plus vite (4 mm/s après 5 s, sans fin). Le bord suivait l'hydrostatique d'une colonne d'air à masse volumique
  constante, l'air intérieur celle de ρ(p), compressible : 1e-4 Pa d'écart au pied d'une colonne de 40 cm, qui ne s'annule jamais →
  colonne compressible au second ordre, p = po + ρ_a Φ (1 + ½ ρ' Φ), plus le décalage de la demi-maille du haut de l'intérieur
  (⅛ ρ_a ρ' (g dx)²), sur le CPU et le GPU : 7.6e-8 m/s après 5 s, au niveau des arrondis.

### Solides mobiles (2026-11-24 → 26)
- **Raideur de contact qui suivait le pas** → le ressort ne rendait pas à la détente l'énergie de la compression (rebond à 0.29 pour
  e = 0.8) → ressort figé pendant le contact, puis un ressort par contact réglé sur la vitesse d'approche (367bbc6).
- **Amortisseur de contact explicite** : au-delà de ζ ≈ 0.5, il renvoyait le corps plus vite qu'il n'arrivait (1.9 → 11 m/s) →
  implicite (367bbc6).
- **Calibrage de l'amortisseur hors de l'ordre du calcul** : le corps entrait d'un pas entier dans l'obstacle avant toute force ; à
  8 pas par période, cette entrée seule le renvoyait à e ≈ 0.85, ζ partait en butée et le rebond dépendait de l'instant d'entrée
  (0.01 à 0.82 pour 0.8) → calibrage dans l'ordre du calcul, ressort qui part de zéro là où le contact est vu (367bbc6).
- **Seize ressorts en parallèle sous un bloc posé sur des mailles solides** (tous les coins de mailles) → coins saillants seulement ;
  **contact en coin trop rapide** (masse effective sans la rotation) → masse effective au point de contact (367bbc6).
- **Eau qui fuyait à travers les bords du domaine** par le débit collé à un corps sur les faces de bord → faces intérieures (367bbc6).
- **Deux blocs empilés qui s'agitaient** : le contact entre les blocs prenait la raideur du contact bloc-sol (une raideur par corps
  au lieu d'une par contact) ; signe de l'amortisseur inversé sur le second corps d'une paire ; détendre un ressort en gardant sa force
  rendait de l'énergie (+20 % par détente au fond d'un rebond) (367bbc6).
- **Film de fluide de quelques microns sous un bloc posé à plat** : à chaque frémissement du bloc, son volume variait de plusieurs fois
  le sien en un pas, la détente y rendait la température négative (pas refaits sans fin, bloc relancé à 0.8 m/s, deux blocs empilés
  non calculés en 25 minutes) → peau de contact de 0.4 maille (367bbc6).
- **Traînée d'un corps en mouvement 2× trop forte et non convergente** (C_D 4.3 → 6.7 en raffinant) : le fluide collé au corps
  comptait deux fois (déplacé et transporté) → débits relatifs, vitesse du mélange transportée (08a746e).
- **Masse ajoutée comptée deux fois** (C_a 1.4–1.9) → frottement calculé sur la vitesse du mélange, linéarisé implicitement ;
  **viscosité implicite qui explosait avec des corps en mouvement** (6000 m/s) → explicite près des corps (8b381c9).
- **Cuve fermée d'huile où coule un disque** : le volume déplacé, anticipé par la projection, était pris pour une erreur de fermeture
  (niveau à ±5 kPa) → `push_src` ; la pression des mailles entièrement couvertes tombait au plancher, la cavité partait dans le sillage
  → `body_trace` (8b381c9).

### Grains, lits, feux (2026-11-22 → 24)
- **Jet de méthane qui s'éteignait en moins d'une seconde** (cuve fermée sur les côtés, la buse baignait dans les fumées) → côtés
  ouverts, combustion au rythme du mélange (c440fef).
- **Braséro dont la flamme s'éteignait en 1.3 s** : flamme décollée du lit, cinétique globale du méthane à la température moyenne
  d'une maille de 1 cm (15 % brûlé) ; une boîte d'air ouverte sur les côtés à pression constante se mettait à 9 m/s → recharge au
  centre, flammes de diffusion au rythme du mélange (`diffusion_mixing`), bords ouverts à pression HYDROSTATIQUE (d0b8017).
- **Lits à deux pressions** : cellules quasi vides (σ' de 1e7 Pa sous une flamme), cellules de surface dont le poids poussait l'air
  dans le tas, dilatation d'un squelette surtassé accumulée dans σ', frottement qui figeait σ', grilles posées après l'initialisation,
  banc GPU qui ne copiait pas σ' ; pas fixé par des traces de grains (braséro en 14 600 pas au lieu de 6 500) (b4c64e4).
- **Suite de tests figée** (dix essais GPU, ou les deux essais de fenêtre, bloqués pour toujours) : plusieurs fils sur le périphérique
  partagé → verrou `GPU_IO` (b4c64e4).
- **Lit poussé par la traînée sans que son poids s'y oppose** (la pression du mélange portait tout le poids) ; une cellule tassée
  décollée de son appui restait poreuse (le mélange s'y resserrait sans limite) (b7dd08f).
- **Grain plus léger que l'eau qui remontait dans l'air** et y retombait (va-et-vient à 0.4 m/s, pas borné à 1.5 ms) ; **viscosité
  explicite d'un liquide très visqueux contre l'air** (celle de la lave appliquée à des faces d'air : 100 m/s en quelques pas) (7240465).
- **Inversion e → T d'une trace de 1e-165 kg à −802 K** (capacité au carré hors de la plage f64, CPU) → forme adimensionnée (162bda5).
- **Pyrolyse 1 − e^(−k dt) en f32 avec annulation** (0.17 % d'écart en un pas pour la houille, GPU) (a08b55d).
- **Pic à 3786 K après un pas court imposé par la chimie** : les vitesses de la projection vidaient une maille au pas suivant →
  croissance du pas bornée à ×2 (a62145b).
- **Couche de surface d'un lit qui le scellait** ; **reste d'énergie de la chimie donné à une trace** (7000 K sur le GPU) (46ede4f).
- **Tas qui fluaient sans fin** (u* portait dt·g sur les faces de paroi, le frottement cisaillait la couche du fond) ; `tau_from` d'une
  trace hors plage (T = −490 K, GPU) (310507f).
- **Aire PLIC f32 instable** (~1 % perdu près d'un axe) ; **échange d'une trace qui débordait** (T = ∞, pas refait) (d51a6d4).

### Solveur, flammes, GPU (2026-11-21 → 23)
- **Fuite de mémoire GPU** : chaque solveur créait son instance et son périphérique, que le pilote ne rendait pas (30 solveurs :
  82 → 440 Mo réservés) ; la suite s'arrêtait net (0xc0000409, « Not enough memory left ») → un périphérique par processus (5bfd344).
- **CPU plus lent sur 16 fils que sur un seul**, résultats non reproductibles au-delà de 16 384 cellules (sommes de rayon) → groupe
  de fils propre, sommes par blocs fixes : identique au bit près quel que soit le nombre de fils (a1d0c3e).
- **Erreur de remplissage corrigée deux fois** (oscillateur neutre de 6 pas) : barrage de 1.9 kPa à 93 bar, vapeur à l'impact ; flammes
  trop rapides (0.47–0.49 m/s) et boucle thermoacoustique → correction sur l'erreur qui restera après l'advection (de2a84a).
- **Débits chimiques fantômes** (réactif absent avec un ordre négatif, traces qui bornaient les sous-pas) : coût ÷5 à ÷70 (8693448).
- **Flammes non convergées en temps** : capteur de flamme dans les gaz brûlés, bloc thermique en sous-pas, borne de pas des flammes,
  prédicteur-correcteur en option (bb33cf3, 4aaa624).
- **Huile à 2000 K sur l'eau chaude** (670 pas refaits, 300 m/s, pause du GPU) : la cavité de vapeur comblait tout le déficit de volume
  quel que soit dt ; borne de chaleur latente sur le seul liquide ; le p·ΔV d'une vapeur qui se condense allait à l'air voisin (porté à
  5000 K) (7cd112f).
- **Une température par cellule mixte** : air des cellules mixtes porté à ~1900 K par de l'huile à 2000 K, vaporisation éclair
  d'huile à 600 K sur l'eau → une température par groupe de phases ; en route, trace de liquide portée à 560 K par la condensation,
  trace de vapeur de 1e-17 kg qui imposait χ = 1 à toute la cellule.
- **Huile chaude au pinceau dans une cuve fermée** : Newton de pression linéarisé (poche recomprimée devenue puits, puis p < 0),
  vitesses produites jamais contrôlées, cavités vides dans l'huile, T < 0 à la détente, chaleur latente extrapolée négative au-delà de
  1450 K, vaporisation d'un bloc quel que soit dt, grosse erreur de fermeture corrigée en un pas.
- **Téléporteur** : seule la première source d'un bloc était servie, les autres détruisaient le fluide ; composition de la dernière
  source ; GPU plafonné à 256 sources (1132f69).
- **NaN du scénario « barrage »** → filet de sécurité (pas refait à dt/4, pause au lieu de NaN).
