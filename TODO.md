pas d’humidité ?
Le low-Mach filtre les ondes acoustiques et les chocs. C’est voulu, mais le son n’existe pas.
Pression thermodynamique uniforme : l’effet de la stratification hydrostatique sur ρ et sur le travail de pression est négligé. C’est acceptable pour quelques mètres.
2D uniquement. La turbulence 2D (cascade inverse) diffère de la 3D, et il n’y a pas de modèle de sous-maille.


(POUR LE BUG (réglé) :
CONFIGURATION : 
 - Rupture de barrage 
 - CPU f64
 - CFL : 0.40
 - dt_max : 0.010
 - gravité : 9.5 m²/s
 - Tension de surface
 - Viscosité
 - Conduction thermique
 - Changement de phase
 - Quantité de mouvement conservative
 - Cell size : 1cm
 - Precision : 5
 - Resolution : 800x600
DESCRIPTION : 
De la pression énorme se créée ce qui transforme l'eau en vapeur, 
   la simulation ne veut pas se remettre en route grâce au garde-fou anti NaN.
Ce bug arrive plus souvent avec le GPU en f32.
MON AVIS:
 - la precision de 5 met trop de stress sur la simulation.
   -> Il faut trouver le point faible qui cause ce crash et le fix.
   -> Peut-être qu'il faut revoir l'architecture/la codebase entière.)





# TODO — Simulation de fluide en Rust

Légende : **[M]** multiphasique · **[1]** monophasique · **[M+1]** les deux

---

## 1. Prioritaire

### Performances et architecture GPU
- **[M+1] Performances : ~5 fps alors que le CPU n'est utilisé qu'à 11 %.** Le calcul tourne sur un seul cœur : paralléliser (rayon) les boucles par cellule (pression/PCG, advection, conduction, chimie), profiler `step()`.
- **[M] GPU en f64 et résident** : ne plus recopier tout l'état à chaque pas.
- **[M] Décision d'architecture à prendre avant d'écrire du code GPU :**
    - GPU seul, ou CPU **et** GPU maintenus en parallèle ?
    - Revoir l'architecture côté GPU ?
    - Aujourd'hui, le GPU (f32, 4 fluides max) est refusé dès qu'il y a cellules spéciales, entrées de bord, miscibilité, chimie ou cp(T) : tout ce qui est récent tourne uniquement sur CPU.
- [M] PCG (Ax = b) : à revoir ? (préconditionneur, convergence) — pas sûr que ce soit le goulot.

### Expériences reproductibles
- **[M] Sauvegarder tout le contexte**, pas seulement les solides : masses, énergie, pression, vitesses, cellules spéciales, bords, matériaux utilisés. Aujourd'hui une scène `.mps` contient solides + thermique + cellules spéciales.
- **[M] Encodeur de conditions initiales / fenêtre d'expérience séparée** pour des expériences précises pas à pas, avec sauvegarde. Exemple : « de la température ici, puis j'accélère là, puis je simule 5 s, j'arrête et je sauvegarde des statistiques ». → trouver l'interface (script d'actions ? timeline ?).
- **[M] Simulation pré-calculée** : on lance une simulation avec la fenêtre d'expérience séparée (pas encore implémentée), puis on fait une sorte de "vidéo" (un peu comme blender ?) mais avec beaucoup d'informations
### Bugs / manques des cellules spéciales
- [M] Téléporteur partiellement cassé : la taille de la destination n'est pas réglable (une seule cellule de destination).
- [M] Nouvelles cellules spéciales : porte / vanne (ouverture réglable), Compresseur, Sensor (pression, température, ...), autres…
- [M] Montée en température : en ajoutant une grosse montée en température très rapide sur un liquide, ça se met sur pause (filet de sécurité ?).
### Simulation monophasique et multiphasique
- **[M + 1] Séparer la simulation monophasique et multiphasique**, avec des dossiers ?

---

## 2. Moins prioritaire

### Physique multiphasique
- [M] Tension de surface **liquide / liquide** (aujourd'hui σ seulement liquide ↔ gaz).
- [M] Dissolution d'un gaz dans un liquide (loi de Henry). C'est l'usage naturel du cas « gaz + liquide miscibles », aujourd'hui seulement signalé par un avertissement.
- [M] Diffusivité dépendant de la **paire** de fluides (au lieu d'une valeur par fluide).
- [M] Enthalpie de mélange.
- [M] Advection de Raessi–Pitsch (l'advection de quantité de mouvement cohérente avec les flux de masse VOF, idée de Rudman, est déjà en place).

### Combustion (limites connues du modèle actuel)
- [M] Ondes de choc / détonations : le solveur de pression ne les gère pas, les explosions restent des déflagrations.
- [M] Dérive d'énergie pendant une explosion (jusqu'à ~5 % de la chaleur dégagée) : schéma en énergie interne, travail de pression discret.
- [M] Radicaux (OH, H, O) : sans eux les flammes sont ~50–90 K trop chaudes.
- [M] Suie : flammes de diffusion sans leur couleur jaune ni leur rayonnement.
- [M] Auto-allumage seulement approximatif (mécanismes globaux à une étape, facteurs A recalés sur la vitesse de flamme).

### Fidélité du solveur monophasique
- [1] Tenseur des contraintes complet : les termes ∇·(μ∇uᵀ) et −⅔μ∇·u sont omis. Ils comptent quand μ varie fortement sur une courte distance (flamme, jet à 1000 K). Demande de réécrire le traitement des parois.
- [1] cp et diffusion du traceur constants (l'air : cp varie d'environ 3 % entre 300 et 600 K).
- [1] Retard d'un pas : propriétés évaluées sur la température du début de pas.
- [1] Benchmarks à relancer pour confirmer (effet négligeable dans la cavité à ΔT = 10 K), en particulier le test de conduction à 600 K.

### Matériaux
- [M] Plus d'exemples de matériaux (miel, …).
- [1] Créateur / éditeur de matériaux personnalisés pour le monophasique (le multiphasique a déjà son éditeur).

### Ancien / autre (toujours à faire)
- [1] Optimiser la simulation de fluide.
- [M+1] Meilleures conditions aux limites : périodique, ouvert, sortie à pression imposée, sortie à vitesse imposée.
- [M+1] Forces externes : poussée d'Archimède sur objets, frottements / sans frottement.
- [M+1] Solides mobiles.
- [M+1] FLIP/PIC ?
- [M+1] Indicateur de vorticité (affichage).

---

## 3. Pas prioritaire
- Option 3D.

---

## 4. Fait

### Solveur et numérique
- Solveur Red-Black Gauss-Seidel, SOR. [1]
- Sous-pas adaptatifs CFL. [1]
- Advection semi-lagrangienne, upwind et MacCormack (MUSCL ordre 2). [1]
- Grille MAC. [M+1]
- Solides immobiles. [M+1]
- Diagnostics : divergence, pression, énergie, masse, CFL, … [M+1]
- Température fluide & solide. [M+1]
- Viscosité (faible). [1]
- Équation low-Mach pour les gaz, équation de Tait pour les liquides. [M+1]
- Multiphasique : interface PLIC conservative, pression implicite (Newton + PCG), tension de surface, changement de phase (ébullition, condensation, cavitation). [M]
- Bug NaN du scénario « barrage » corrigé + filet de sécurité (retour arrière, sous-pas, pause au lieu de NaN). [M]
- Advection MUSCL de la composition et de la température dans les mélanges (diffusion numérique réduite). [M]
- Pas de temps capillaire seulement pour les liquides présents. [M]

### Fonctionnalités
- Toutes les fonctionnalités de `main_window.rs` portées dans `multiphase_window.rs` : cellules spéciales (puits, émetteur, accélérateur, téléporteur), solides chauffants / isothermes, bords modifiables en direct, raccourcis, sauvegarde/chargement de scène. [M]
- Matériaux solides, liquides et gazeux ajoutables et personnalisables (défauts supprimables et restaurables), miscibilité par paire, export/import JSON. [M]
- Mélanges nommés (air sec O2 + N2, prémélanges) comme pinceaux et fluide ambiant. [M]

### Changements d'état et réactions chimiques
- Cinétique chimique : Arrhenius, ordres partiels, réactions réversibles vers l'équilibre thermodynamique (dissociation de CO2 / H2O). [M]
- Thermochimie : enthalpies de formation, entropies, cp(T) calé sur JANAF. [M]
- Flammes et explosions : modèle de flamme épaissie (vitesse de flamme juste quelle que soit la maille), rayonnement des gaz chauds, outil « Allumer », 4 scénarios (tube, explosion confinée, explosion avec évent, flamme de diffusion), préréglage combustion. [M]
- Évaporation / condensation à la pression partielle (Dalton), diffusion propre à chaque espèce. [M]