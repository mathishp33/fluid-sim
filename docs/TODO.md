# TODO — Simulation de fluide en Rust

Légende : **[M]** multiphasique · **[1]** monophasique · **[M+1]** les deux. Mis à jour le 2026-11-26. Le solveur monophasique
n'est plus suivi depuis le 2026-11-26 : il reste tel quel, ses points [1] pour mémoire seulement.
À lire avec : PHYSICS.md (les lois de la simulation), MESURES.md (la simulation face aux références), BUGFIXES.md (défauts connus
et corrigés), et les documents de conception SOLID.md (solides mobiles), GRANULAIRE.md (grains), MULTIPHASE.md (solveur),
EXPERIENCES.md (protocoles).

---

## 1. Prioritaire

### Organisation et interface

### Bords du domaine
- [M] Plus de types de bords : périodiques, sortie convective. Existent : bord ouvert à pression imposée (hydrostatique de
  l'ambiant, colonne compressible : le vent parasite des côtés ouverts est corrigé, 7.6e-8 m/s après 5 s au lieu de 4 mm/s), entrées
  et sorties à vitesse imposée.

### Solides mobiles (voir SOLID.md ; étapes 0 à 4 et chaleur faites)
- [M] GPU : les scènes avec des corps se calculent sur le CPU. Porter fractions et ouvertures des faces, couplage monolithique
  pression–corps, fluide balayé, contacts et chaleur des corps (fin de l'étape 5) ; accord CPU / GPU.
- [M] Convection naturelle autour d'un corps sous-résolue à 10 mailles par diamètre (h ≈ 250 W/(m²·K) pour ≈ 1400 selon
  Churchill–Chu) : mesurer à 20 mailles, revoir les ouvertures prudentes des faces pour le fluide chauffé au contact.
- [M] Rebond dans un liquide selon le nombre de Stokes (Gondret et al. 2002) : à mesurer ; le film entre deux surfaces n'est pas
  résolu (pas de lubrification).
- [M] Bruit de pression des corps en mouvement dans l'eau (fluide balayé routé vers une seule voisine) ; niveau de pression d'une
  cuve fermée pleine de liquide qui oscille de quelques kPa sous un corps libre ; masse ajoutée apparente jusqu'à 1.4 quand le pas
  double au démarrage.
- [M] Grains : leur sédimentation ignore les corps (la fermeture les pousse hors du corps, mais ils n'y voient pas une paroi).
- [M] Plus tard : bloc de glace qui fond, bûche qui brûle, liaisons, butées, ressorts, formes non convexes, température non uniforme
  dans un corps (nombre de Biot élevé).

### Poudres, grains et feux (voir GRANULAIRE.md)
- [M] Lit fluidisé trop agité : au-delà de la vitesse minimale de fluidisation (juste, voir MESURES.md), le lit gonfle trop (×2 au
  lieu de ~×1.2), s'agite trop (4–6 m/s, même à 1.11 U_mf) et sa perte de charge dépasse son poids de 2 à 3× ; dans une colonne
  étroite il monte en bouchons qui sortent par le haut. Une seule vitesse par cellule : les paquets denses décollés de leur appui,
  étanches (perméables, la percolation y laissait le mélange se resserrer sans limite), sont poussés comme des pistons. Essayés sans
  effet notable : place des receveuses de la sédimentation comptée sur le bilan des cellules (un paquet décollé « pleut » d'un bloc),
  viscosité de la phase dense fluidisée jusqu'à 2 Pa·s. Il faut un modèle à deux vitesses (gaz et grains, traînée entre les deux),
  gros chantier.
- [M] Solidification : une seule température de fusion (pas d'intervalle liquidus–solidus : lave, cire) ; lave qui entre dans l'eau :
  explosions de vapeur trop violentes (pointes à ~200 m/s, 4 % de la lave projetée hors du domaine en 1.5 s : pas de film de vapeur
  de Leidenfrost, voir « Robustesse ») — scénario « coulée de lave dans la mer » écarté pour cela ; la neige fondante d'une eau
  surfondue est faite de cristaux libres (pas de réseau de dendrites soudé : elle flotte et s'étale en courant de gravité, le front
  court deux fois plus vite le long de la surface) ; les liquides ne rayonnent pas (une lave à l'air refroidit bien trop lentement).
- [M] Allumer une charge FROIDE : le rayonnement du tas chauffe vite du bois (ou de la houille) froid posé sur les braises (1000 K au
  contact en 4 s), mais dans le braséro ouvert, sans cheminée, le tirage est trop faible : une grosse charge étouffe les braises (O2
  du tas à 1 %, braises à 1250 K en 14 s) avant que ses gaz ne s'enflamment ; les scénarios partent donc d'un feu établi. Pistes :
  cheminée au-dessus du foyer (tirage), charge plus petite ou petit bois (morceaux de 5 mm : τ = 4 s au lieu de 66), séchage du bois.
- [M] Poches de combustible et d'air qui s'enflamment d'un coup dans une maille (braséro, feu de bois : pointes de 3000–3600 K,
  ponctuelles) : le mélange froid hors de la flamme garde la cinétique globale ; il faudrait l'allumer progressivement (flamme
  épaissie dans la poche) plutôt que d'un bloc.
- [M] Poudre noire : délai d'allumage d'un grain baigné de gaz chaud (aujourd'hui immédiat dès que le fluide dépasse 580 K ; vraie
  surface chauffée en ~1 ms à quelques dizaines de ms selon l'écoulement, nombre de Nusselt fonction de la vitesse) ; poudre dans un
  canon (charge confinée, jet à la bouche).
- [M] Lits perméables à deux pressions : σ' résolue deux fois par pas (avant le frottement, puis après la première itération de
  Newton) ; seuil de perméabilité K/μ ≥ 5e-7 m²/(Pa·s) (sable de 0.1 mm et plus dans l'air, de 0.6 mm et plus dans l'eau) ; plus de
  dilatation d'un squelette tassé au-delà de α_max (le frottement l'empêchait et σ' s'accumulait) — à surveiller sur de longues
  durées.
- [M] Réactions avec les solides : un solide (paroi, obstacle) qui brûle, fond ou se consume (bois, métal) ; aujourd'hui seules les
  phases granulaires réagissent.

### Robustesse : contacts très chauds ↔ froids
- [M] Huile à 2000 K qui atteint l'eau chaude (cuve fermée, 80×60, `hot_oil_on_hot_water_violence`) : depuis la correction de
  fermeture, contact calme — CPU p max 1.9 bar, |u| max 15 m/s, 2 pas refaits, 1 720 pas en 35 s (avant : 56 bar, 60 m/s, 10, 9 220
  pas en 370 s) ; GPU 1.5 bar, 14 m/s, 0 pas refait, 13 s (avant 45 bar, 93 m/s). Reste la physique : ébullition en FILM
  (Leidenfrost) — la vapeur isole l'huile de l'eau, la conductance d'échange ne le voit pas (corrélation de Bromley, ou couche de
  vapeur sous-maille).
- [M] Échange de chaleur entre groupes d'une cellule mixte : conduction de part et d'autre d'une interface supposée PLANE dans la
  cellule (G = λ·2k/(α·dx²)). Pour des gouttes / bulles dispersées sous la maille, l'aire d'échange est sous-estimée : modèle d'aire
  interfaciale + corrélations (Ranz–Marshall ; c'est déjà le cas des grains : échange par leur surface, Nu = 2). λ est réglable dans
  la fenêtre (curseur « Échange de chaleur entre phases », case « Une seule température par cellule » = ancien modèle).
- [M] Couplage du pas χ = max_g a_g/(1 + a_g) (heuristique) : une petite phase non négligeable (eau à α ~ 1e-4 dans du gaz) fait
  traiter toute la cellule « à une température » (fermeture, travail de pression, chaleur latente) ; seules les traces (< 1e-9 de la
  capacité de la cellule) sont exclues. À remplacer par un couplage par paire de groupes.
- [M] Vapeur « verrouillée » sur son liquide (couplage parfait) dès qu'elle fait plus de 50 % de la masse de son groupe : seuil
  arbitraire, à remplacer par une conductance d'interface vapeur ↔ liquide (évaporation / condensation à l'interface).
- [M] Contact liquide ↔ liquide (huile / eau, liquides non miscibles) maintenant hors équilibre aussi : à valider sur une expérience
  (temps de mise en équilibre d'une couche d'huile chaude sur l'eau).
- [M] Les poches de gaz très détendues (quelques centaines de Pa) refroidissent par détente jusqu'au plancher de 60 K : physique dans
  le modèle, mais c'est la conduction des voisines qui devrait les réchauffer plus vite.

### Performances (GPU et CPU)
- **[M] GPU : fusionner des noyaux** — mesuré sur une vraie carte (AMD Radeon intégrée, Vulkan ; `cargo test --release
  gpu_speed_benchmark -- --ignored --nocapture`) : le GPU est 8 à 25× plus rapide que le CPU f64 (voir MESURES.md § 7). Mais les
  petites grilles sont limitées par le COÛT DE LANCEMENT (~15–20 µs par noyau, 285 à 1446 noyaux par pas) : ×16 cellules de 80×60 à
  320×240 ne donne que ×4–7 en temps. La relecture de contrôle par pas coûte peu (séries de 8 ≈ pas à pas, sauf l'explosion H₂).
  Gains à chercher : moins de noyaux par pas (fusion, moins d'itérations Newton / PCG / sous-pas lancées une à une), pas des noyaux
  plus rapides. À remesurer sur une carte dédiée (NVIDIA / AMD), et avec des grains (le frottement implicite et la percolation
  ajoutent des noyaux à chaque pas).
- [M] Suite de tests : ~9.5 min sur la machine de développement libre (194 essais le 2026-11-26, le monophasique écarté :
  `--skip simulation::mono` ; jusqu'à 65 min la nuit, mémoire juste). Les comparaisons GPU calculent leur référence CPU sans tenir
  de jeton GPU (avant : ~12 min). Le plus long : feu de charbon (référence CPU 84 s + GPU 49 s sur 1 s simulée), puis feu de bois,
  gel et fonte, colonne de sable, sable sur étagère (35 à 55 s chacun). La raccourcir encore : durées ou grilles plus courtes pour
  ceux-là, ou série rapide séparée des longues.
- [M] GPU : contournements de bogues de lavapipe — tampons de commandes de ≤ 400 lancements : **coût nul mesuré** sur la Radeon
  (`MP_GPU_BATCH_OPS=1000000` : mêmes temps à ±10 % près), à garder. Restent non mesurés (en dur dans le WGSL) : dans les noyaux par
  groupe, boucles à borne uniforme (P.np / P.ng, appartenance au groupe par sélection) plutôt que sur les membres lus dans la table,
  pas de `continue` ni de tableau privé dans une branche divergente, pas de sortie anticipée avant une boucle (sinon plantage du code
  JIT de lavapipe, selon les données). La chimie (au moins une itération par fil) ne s'active que sur adaptateur logiciel.
  Diagnostics : `MP_GPU_SYNC=1` soumet chaque lancement seul et affiche son nom (repère le noyau qui plante) ; `MP_GPU_BATCH_OPS=n`
  change la taille max des tampons.
- [M] GPU f32 : une boîte FERMÉE entièrement remplie de liquide (eau pure 6×60, aucun gaz) diverge en ~10 pas (Newton au plafond de 4
  itérations, p jusqu'à 1e9 Pa, cavités comblées → masse d'eau qui croît), grains ou non (`gpu_closed_water_box_trace`, ignoré) ; le
  CPU f64 tient. La moindre poche de gaz suffit à stabiliser. Piste : même que ci-dessous (masses en écart à une référence), ou
  critère d'arrêt de Newton adapté au f32.
- [M] GPU f32 : limites de précision connues — la fermeture volumique 1 − Σ m/ρ n'est connue qu'à ~1e-7 près (masses stockées en
  f32) : dans une cellule presque pleine de liquide, la vapeur produite par pas est bruitée de ~1 % et la pression du liquide de
  quelques Pa. Piste : stocker les masses des liquides en écart à une référence (m − ρ₀α).
- [M] Advection PLIC : les flux sont appliqués face par face dans l'ordre des indices (la donneuse est lue après ses flux entrants
  pour un écoulement vers +x, avant pour −x). La forme « flux pure » (tout évalué sur l'état d'avant le balayage, indépendante de
  l'ordre) a été essayée : elle diverge sur un jet d'eau entrant dans l'air (écarts de remplissage m/α ≠ ρ transportés sans
  amortissement). À creuser, puis à reporter sur le GPU (qui reproduit aujourd'hui exactement l'ordre du CPU).
- [M] CPU : parties encore séquentielles (forces, quantité de mouvement, pas de temps, niveaux grossiers du multigrille,
  sédimentation, percolation, cavités) ; sur un portable qui bride les charges sur plusieurs cœurs, elles ralentissent quand les
  autres cœurs travaillent. Le multigrille sur 160×120 et moins reste limité par la taille des régions (~10–20 µs).

### Cellules spéciales
- [M] Existent : puits, émetteur, accélérateur, téléporteur, grille, vanne à guillotine (fermée, ouverte, ouverture en fraction ou
  en %), pompe / compresseur ; toutes posables par protocole et au pinceau dans la fenêtre d'expérience (vanne et pompe aussi dans
  la fenêtre multiphasique). Une porte sur charnière se fait avec un corps libre sur pivot (clapet, SOLID.md) ; une soupape de
  sécurité ou un thermostat tout ou rien, avec deux règles `quand` aux seuils écartés (`experiences/soupape.exp`). À ajouter :
- [M] Régulateur proportionnel (PI) : puissance de chauffe ou ouverture de vanne qui suit l'écart d'une sonde à sa consigne, au lieu
  du tout ou rien des règles `quand` — bain-marie, couveuse, débit régulé.
- [M] Clapet anti-retour en mailles : faces qui ne laissent passer l'écoulement que dans un sens (le clapet sur pivot existe, mais
  c'est un corps : CPU seulement, et cher). Dans la projection : fermer les faces où le flux s'inverserait, puis résoudre à nouveau.
- [M] Paroi mobile : solide dont la surface glisse à vitesse imposée (tapis roulant, cylindre qui tourne en mailles) ; donnerait la
  cavité entraînée par son couvercle, essai de référence de la viscosité (Ghia, Ghia et Shin 1982).
- [M] Ventilateur à courbe Δp(Q) : saut de pression imposé à travers la zone, qui baisse quand le débit monte (Δp = Δp₀ (1 −
  Q/Q_max)), au lieu du débit fixé de la pompe : le débit s'ajuste aux pertes du circuit.
- [M] Membrane sélective : faces ouvertes à une seule phase — évent qui laisse passer le gaz et retient le liquide, filtre qui
  retient une poudre.
- [M] Couche éponge (bord absorbant) : mailles qui ramènent doucement la vitesse au repos, pour que vagues et ondes de pression ne
  se réfléchissent pas sur les bords (bac à vagues, explosion à l'air libre).
- [M] Chauffe-fluide à puissance imposée (W/m³ dans le fluide : résistance plongée, effet Joule, micro-ondes) : « chauffer » vise
  une température ; une puissance connue sert aux essais de convection à flux imposé (Rayleigh–Bénard).
- [M] Trémie : émetteur de grains à débit massique et à fraction choisie (sablier, charbon qui alimente un feu) ; l'émetteur
  remplit sa maille d'une seule phase, ce qui ne convient pas à une poudre (à vérifier).
- [M] Évent intérieur : maille à pression imposée au milieu du domaine (cheminée, prise d'air), comme un bord ouvert.

---

## 2. Moins prioritaire

### Physique multiphasique
- [M] Tension de surface **liquide / liquide** (aujourd'hui σ seulement liquide ↔ gaz).
- [M] Dissolution d'un gaz dans un liquide (loi de Henry). C'est l'usage naturel du cas « gaz + liquide miscibles », aujourd'hui
  seulement signalé par un avertissement.
- [M] Diffusivité dépendant de la **paire** de fluides (au lieu d'une valeur par fluide).
- [M] Enthalpie de mélange.
- [M] Advection de Raessi–Pitsch (l'advection de quantité de mouvement cohérente avec les flux de masse VOF, idée de Rudman, est déjà
  en place).

### Combustion (limites connues du modèle actuel)
- [M] Flamme épaissie : écart en temps restant ~7 % au dt par défaut (prémélange du test GPU, 64×12, dx = 1 mm, 0.02 s : chaleur 252
  contre 270.5 à dt ≤ 5e-6 ; 269 à dt ≤ 2e-5). Deux parts du 1er ordre : découpage de Lie diffusion ↔ réaction (raide : Strang perd
  son ordre ; réduit par les sous-pas `flame_thermo_substeps`) et couplage avec l'écoulement. Leviers : `cfl_flame` (borne de pas des
  flammes) et le prédicteur-correcteur.
- [M] Ondes de choc / détonations : le solveur de pression ne les gère pas, les explosions restent des déflagrations.
- [M] Dérive d'énergie pendant une explosion (jusqu'à ~5 % de la chaleur dégagée) : schéma en énergie interne, travail de pression
  discret.
- [M] Radicaux (OH, H, O) : sans eux les flammes sont ~50–90 K trop chaudes.
- [M] Suie : flammes de diffusion sans leur couleur jaune ni leur rayonnement.
- [M] Auto-allumage seulement approximatif (mécanismes globaux à une étape, facteurs A recalés sur la vitesse de flamme).
- [M] Jet de méthane : le pied de la flamme s'étale sur le fond à ~4 cm de la buse au lieu de s'accrocher à son bord.

### Fidélité du solveur monophasique (plus suivi : laissé tel quel)
- [1] Tenseur des contraintes complet : les termes ∇·(μ∇uᵀ) et −⅔μ∇·u sont omis. Ils comptent quand μ varie fortement sur une courte
  distance (flamme, jet à 1000 K). Demande de réécrire le traitement des parois.
- [1] cp et diffusion du traceur constants (l'air : cp varie d'environ 3 % entre 300 et 600 K).
- [1] Retard d'un pas : propriétés évaluées sur la température du début de pas.
- [1] Benchmarks à relancer pour confirmer (effet négligeable dans la cavité à ΔT = 10 K), en particulier le test de conduction à
  600 K.

### Matériaux
- [M] Plus d'exemples de matériaux : miel (il existe dans le monophasique, pas dans la bibliothèque multiphasique), cuivre (solide :
  corps et parois très conducteurs), …
- [1] Créateur / éditeur de matériaux personnalisés pour le monophasique (le multiphasique a déjà son éditeur).

### Mesures à compléter (voir MESURES.md, « À mesurer »)
- [M] Front du barrage face aux expériences de Martin & Moyce (1952) ; lit fluidisé face à l'expansion de Richardson–Zaki.

### Autre
- [1] Optimiser la simulation de fluide.
- [M] FLIP/PIC ?

---

## 3. Pas prioritaire (limites assumées du modèle)
- Option 3D. En 2D, la turbulence diffère (cascade inverse) et il n'y a pas de modèle de sous-maille.
- [1] Le low-Mach filtre les ondes acoustiques et les chocs : c'est voulu, mais le son n'existe pas. Le multiphasique n'est pas
  low-Mach (fermeture avec ρ(p local, T)) : la projection garde les ondes de pression, amorties par Euler implicite ; les chocs n'y
  sont pas captés non plus (voir « Combustion »).
- [1] Pression thermodynamique uniforme : l'effet de la stratification hydrostatique sur ρ et sur le travail de pression est négligé
  (acceptable sur quelques mètres ; le multiphasique en tient compte).
- [M] Une seule vitesse par cellule : les grains n'ont pas d'inertie propre (un grain lourd projeté par un jet ne file pas plus loin
  que le gaz qui le porte ; à l'inverse, le souffle d'un tas de poudre noire emporte ses grains avec les gaz, à plusieurs centaines de
  m/s : la moitié du tas sort du domaine avant d'avoir brûlé) ; un modèle à deux vitesses serait bien plus lourd.

---

## 4. Fait

Résumé : une ligne par chantier. Les validations chiffrées sont dans MESURES.md, les défauts corrigés en route dans BUGFIXES.md, le
détail dans les documents de conception et l'historique git.

### Documentation et organisation (2026-11-26)
- TODO trié et élagué ; MESURES.md (la simulation face aux références) et BUGFIXES.md (défauts connus et corrigés) ; documents
  rangés dans `docs/` ; README refait.
- Code rangé en deux dossiers : `src/simulation/mono/` (monophasique) et `src/simulation/multi/` (multiphasique), lois de propriétés
  communes dans `src/simulation/materials.rs` ; anciens fichiers WGSL et `build_wgsl.py`, inutilisés depuis le 2026-11-19, retirés.
- Fenêtre monophasique retirée (elle ne sert plus) : l'écran de départ lance le multiphasique ; le solveur monophasique reste pour
  ses essais.

### Solides mobiles (2026-11-24 → 26, voir SOLID.md)
- Étude et architecture (domaine fictif à une vitesse, repli monolithique), choix validés le 2026-11-24.
- Étape 0 : corps rigides préparés (géométrie exacte, fractions de maille, contraintes, rendu), aucun résultat changé.
- Étape 1 : corps FIXES, l'eau les contourne sans y entrer ; commande `corps` et outil de la fenêtre d'expérience.
- Étape 2 : mouvement IMPOSÉ (vitesse, rotation, oscillation, pivot) : piston, palette, batteur à vagues.
- Étape 3 : corps LIBRES (axes bloqués, pivot, masse volumique propre), pression et corps résolus ensemble, force du fluide mesurable
  ; traînée des corps en mouvement corrigée (le fluide collé au corps comptait deux fois).
- Étape 4 : contacts (bords, solides fixes, autres corps ; restitution et frottement ; peau de contact).
- Étape 5, chaleur : une température par corps, échangée avec le fluide qui le borde (option `à T`, mesure `corps n température`).
- Outil « Corps » de la fenêtre d'expérience : corps fixes, à mouvement imposé (vitesse, rotation, oscillation) ou libres
  (masse volumique propre, rebond, frottement, axes bloqués), pivot au milieu d'un côté, température.
- Protocoles : `simuler jusqu'à ce que <relevé> < <seuil> [et | ou …] [au plus d]` (condition d'arrêt sur une ou plusieurs
  mesures reliées par « et » ou par « ou », CPU et GPU).
- Protocoles : sondes interpolées (`T en (x, y)` interpole entre les centres des mailles voisines, les vitesses entre les faces ; au
  centre d'une maille, sa valeur).
- Protocoles : `débit [massique] ["fluide"] à travers ligne (x1, y1) (x2, y2)` ; le solveur compte, par phase, ce qui passe par les
  faces suivies (transport PLIC, chute des grains, percolation, fluide chassé par un corps ; CPU et GPU) : le débit intégré redonne
  exactement la masse passée.
- Bords : `bord … sortie (vx, vy)`, sortie à vitesse imposée (pompe, vidange ; débit ρ v L exact, CPU et GPU).
- Corps : lois en formules (variables t, x, y, vx, vy, ax, ay, angle, rotation, x0, y0) — `position | vitesse | accélération
  formule`, `angle | rotation formule` pour un mouvement imposé (Runge–Kutta 4), `force | couple formule` ajoutés à un corps libre
  (ressort, amortisseur, moteur).
- Protocoles : `toutes les T [à partir de t0] [jusqu'à t1] : <action>`, action répétée en parallèle du déroulement, à l'heure
  pile (vanne, impulsions de chauffage) ; un bord qui redevient un mur arrête tout de suite l'écoulement de ses faces.
- Cellules spéciales : `vanne <zone> fermée | ouverte [f | n %]`, vanne à guillotine (faces fermées sur la part fermée, fluide
  gardé ; CPU et GPU), à ouvrir et fermer par des actions répétées.
- Cellules spéciales : `pompe <zone> (vx, vy)` (ou `compresseur`), débit volumique fixé à travers la zone (vitesse imposée sur ses
  faces) ; en aval fermé, le gaz se comprime (CPU et GPU).
- Protocoles et fenêtre : action datée `à t : <action>` ; un geste fait pendant une expérience (en cours ou en pause) est fait tout
  de suite et inscrit ainsi, rejoué à l'heure pile.
- Affichage « Vorticité » (rotationnel ∂v/∂x − ∂u/∂y rapporté à son maximum), dans les fenêtres, les images et les séries d'images
  des protocoles ; CPU et GPU.
- Les solides immobiles subissent des forces : mesure `force x | y sur <zone>` (force du fluide sur les mailles solides ; un corps
  fixe la mesurait déjà).
- Protocoles : `sauver image "nom" [affichage]` (image PNG d'un champ dans le dossier des résultats) et `images toutes les d
  [affichage]` (série d'images pour une animation).
- Corps : mesures `corps n fx | fy | couple contact | loi | total` (force des contacts, d'une loi en formule, ou bilan avec le
  poids : M a au pas près, vérifié à 2e-14) ; disque fixe immergé : Archimède à 4e-6 près ; disque posé au fond : le fond porte
  son poids moins la poussée à 0.3 % près.
- Cellules spéciales : `téléporteur <zone> vers (x, y) [v] [rayon r]` en protocole ; vanne, pompe, accélérateur et téléporteur
  (destination choisie d'un clic) posables dans la fenêtre d'expérience, vanne et pompe dans la fenêtre multiphasique.
- Protocoles : règles `quand <condition> : <action>` (faite chaque fois que la condition devient vraie, relevée après chaque pas ;
  aussi `pause`) : soupape de sécurité (`experiences/soupape.exp` : pression tenue entre 102.2 et 104.1 kPa, au lieu de monter de
  49 kPa/s), thermostat tout ou rien. CPU et GPU au même pas : le GPU relève pas à pas près d'un seuil (tendance prise aussi depuis
  le relevé le plus éloigné, qu'une sonde qui oscille ne trompe pas) et après chaque action ; les conditions d'arrêt en profitent.
- Protocoles : boucles `tant que <condition> [au plus d]` … `fin` (imbriquées ; le corps doit faire avancer le temps, une boucle
  qui tourne sans que le temps avance arrête l'expérience), `attendre <durée> | jusqu'à <t> | [que] <condition> [au plus d]`,
  relevé `temps` ; une règle ou une répétition écrite dans une boucle n'est posée qu'une fois ; exemple `experiences/bachees.exp`
  (vidange par bâchées).
- Protocoles : `répéter n fois … fin`, `si <condition> … sinon si … sinon … fin`, règles `quand <condition> … fin` en bloc (ses
  lignes, attentes comprises, se déroulent en parallèle du protocole chaque fois que la condition devient vraie ; une suite à la
  fois par règle) ; exemple `experiences/chasse.exp` (chasse d'eau automatique) ; journal muet au-delà du troisième tour.
- Protocoles : conditions avec parenthèses (« et » et « ou » mêlés : `(A et B) ou C`) ; variables `soit <nom> = <relevé |
  nombre | "formule">` (relevés, seuils recalculés à chaque relevé, colonnes de mesure, repères `{nom}`) ; exemple
  `experiences/compteur.exp` (rebonds comptés par une règle).
- Protocoles : valeurs calculées `{formule}` (en t et en variables) à la place d'un nombre — dans les actions (relues chaque fois
  qu'elles sont faites, aussi dans une règle ou une répétition), les durées, le nombre de tours de `répéter`, les seuils, `soit` et
  les repères ; exemple `experiences/rampe.exp` (chauffe par paliers calculés).
- Corps : système bielle-manivelle `vilebrequin (x, y) rayon r bielle l …` (manivelle, bielle et piston menés par l'angle du
  vilebrequin, cylindre en option ; mené à vitesse imposée, ou libre sous les forces du fluide, les poids, un volant et un couple
  appliqué), mesures `vilebrequin n angle | tours | rotation | couple | puissance` ; exemple `experiences/moteur.exp` (moteur à
  air comprimé).

### Expériences et interface
- Fenêtre d'expérience, protocoles reproductibles `.exp`, ligne de commande `fluid-sim experience` (2026-11-24, voir
  EXPERIENCES.md) : actions à l'heure pile, mesures sans effet sur le résultat, CPU reproductible au bit près, balayages, champs de
  valeurs (CSV, PNG, formules), exemples dans `experiences/`.
- Sauvegarde de l'ÉTAT COMPLET (`mp_state`) : exacte au bit près, versionnée ; « Géométrie seule » garde l'ancien format JSON.
- Téléporteur réparé, destination de taille réglable, sources groupées (CPU et GPU).
- Fenêtre multiphasique : toutes les fonctionnalités de `main_window.rs` (cellules spéciales, solides chauffants / isothermes, bords
  en direct, raccourcis, scènes), panneaux rabattables. [M]
- Matériaux solides, liquides et gazeux ajoutables et personnalisables (défauts supprimables et restaurables), miscibilité par paire,
  export/import JSON ; mélanges nommés (air sec, prémélanges) comme pinceaux et fluide ambiant. [M]

### Phase granulaire, feux (2026-11-22 → 24, voir GRANULAIRE.md)
- Grains : transport sans interface, chute (Ferguson–Church, Richardson–Zaki), échange thermique par leur surface ; portage GPU.
- Frottement (Drucker–Prager implicite) : le sable forme des tas ; avalanches, étagères, chaleur de frottement ; viscosité implicite
  (miel).
- Réactions de surface, charbon (CO, CO2, Boudouard), flamme de CO ; percolation (Ergun dans la projection) et grilles ; feu de
  charbon sur grille.
- Pyrolyse du bois et feu de bois ; houille ; rayonnement dans le tas ; braséro à grande flamme qui tient.
- Poudre noire : combustion sans air, tas qui fuse en fumée blanche.
- Solidification eau ↔ glace (méthode enthalpique, Stefan) ; surfusion et front de dendrites ; cire et lave.
- Fluidisation des lits de 1 mm et plus ; lits de grains fins perméables à deux pressions (fluide des pores, contacts).
- Jet de méthane dont la flamme tient à l'air libre (bords ouverts à pression hydrostatique, combustion au rythme du mélange).

### Changements d'état et réactions chimiques [M]
- Ébullition, condensation, cavitation ; évaporation à la pression partielle (Dalton), diffusion propre à chaque espèce.
- Cinétique chimique : Arrhenius, ordres partiels, réactions réversibles vers l'équilibre (dissociation de CO2 / H2O) ;
  thermochimie (enthalpies de formation, entropies, cp(T) calé sur JANAF).
- Flammes et explosions : flamme épaissie (vitesse juste quelle que soit la maille), rayonnement des gaz chauds, outil « Allumer »,
  4 scénarios ; flammes justes au dt par défaut (S_L = 0.384 m/s).
- Une température par groupe de phases dans les cellules mixtes (échange implicite entre groupes, énergie conservée).

### Solveur, numérique, performances
- Solveur Red-Black Gauss-Seidel, SOR ; sous-pas adaptatifs CFL ; advection semi-lagrangienne, upwind et MacCormack ; viscosité
  (faible). [1]
- Grille MAC ; solides immobiles ; température fluide & solide ; équation low-Mach pour les gaz, Tait pour les liquides ;
  diagnostics. [M+1]
- Multiphasique : interface PLIC conservative, pression implicite (Newton + PCG multigrille), tension de surface, advection MUSCL de
  la composition et de la température, quantité de mouvement conservative. [M]
- Filet de sécurité : pas refait à dt/4 (NaN, T aberrante, vitesses), pause au lieu de NaN ; chauffage au pinceau progressif. [M]
- Fermeture : l'erreur de remplissage n'est plus corrigée deux fois (barrage réaliste, flammes justes). [M]
- CPU parallèle (`mp_par.rs`) : résultat identique au bit près quel que soit le nombre de fils, réglage automatique du nombre de
  fils. [M]
- GPU résident (f32) : tout ce que fait le CPU (sauf les corps rigides), accord pas à pas à l'arrondi f32 près ; un seul
  périphérique par processus (fuite de mémoire corrigée). [M]
