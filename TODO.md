pas d’humidité ?
Le low-Mach filtre les ondes acoustiques et les chocs. C’est voulu, mais le son n’existe pas.
Pression thermodynamique uniforme : l’effet de la stratification hydrostatique sur ρ et sur le travail de pression est négligé. C’est acceptable pour quelques mètres.
2D uniquement. La turbulence 2D (cascade inverse) diffère de la 3D, et il n’y a pas de modèle de sous-maille.


REACTION AVEC SOLIDES & POUDRES (carbones, bois)
POUDRES (poudre a canon, carbone)

En scénario combustion confinée hydrogène : 
lorsque la combustion est fine/presque finie, lorsque je passe en vue PHASE, ça crash.
# TODO — Simulation de fluide en Rust

Légende : **[M]** multiphasique · **[1]** monophasique · **[M+1]** les deux

---

## 1. Prioritaire

### Performances et architecture GPU
- **[M] Mesurer le GPU résident sur une vraie carte graphique** (NVIDIA / AMD / Intel) : il n'a pu être validé que sur lavapipe (pilote Vulkan logiciel, qui exécute les noyaux sur le CPU : aucune mesure de vitesse n'y a de sens). Regarder le nombre de noyaux par pas affiché dans le panneau (≈ 300–500) et le temps « GPU : pas complet » du profil.
- [M] GPU : profil par partie du pas (requêtes d'horodatage wgpu) au lieu d'un seul temps « pas complet ».
- [P] GPU : contournements de bogues de lavapipe à revérifier sur une vraie carte — dans la chimie, chaque fil fait au moins une itération de boucle (seulement sur adaptateur logiciel) ; partout, une série de pas est soumise en tampons de commandes de ≤ 1200 lancements (lavapipe plantait au-delà de ~2500). Coût attendu nul, à confirmer.
- [M] GPU f32 : limites de précision connues — la fermeture volumique 1 − Σ m/ρ n'est connue qu'à ~1e-7 près (masses stockées en f32) : dans une cellule presque pleine de liquide, la vapeur produite par pas est bruitée de ~1 % et la pression du liquide de quelques Pa. Piste : stocker les masses des liquides en écart à une référence (m − ρ₀α).
- [M] Advection PLIC : les flux sont appliqués face par face dans l'ordre des indices (la donneuse est lue après ses flux entrants pour un écoulement vers +x, avant pour −x). La forme « flux pure » (tout évalué sur l'état d'avant le balayage, indépendante de l'ordre) a été essayée : elle diverge sur un jet d'eau entrant dans l'air (écarts de remplissage m/α ≠ ρ transportés sans amortissement). À creuser, puis à reporter sur le GPU (qui reproduit aujourd'hui exactement l'ordre du CPU).

### Expériences reproductibles
- **[M] Sauvegarder tout le contexte**, pas seulement les solides : masses, énergie, pression, vitesses, cellules spéciales, bords, matériaux utilisés. Aujourd'hui une scène `.mps` contient solides + thermique + cellules spéciales.
- **[M] Encodeur de conditions initiales / fenêtre d'expérience séparée** pour des expériences précises pas à pas, avec sauvegarde. Exemple : « de la température ici, puis j'accélère là, puis je simule 5 s, j'arrête et je sauvegarde des statistiques ». → trouver l'interface (script d'actions ? timeline ?).

### Bugs / manques des cellules spéciales
- [M] Téléporteur partiellement cassé : la taille de la destination n'est pas réglable (une seule cellule de destination).
- [M] Nouvelles cellules spéciales : porte / vanne (ouverture réglable), compresseur, autres…

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

### Performances, robustesse, GPU
- Performances CPU : préconditionneur multigrille du gradient conjugué (≈ 20 itérations au lieu de ≈ 270 avec IC(0)), boucles par cellule parallélisées (rayon), profil de `step()` par partie. [M]
- Montée en température brutale (2000 °C au pinceau sur un liquide) sans mise en pause : chauffage progressif en temps simulé (puissance plafonnée), ébullition en volume au-delà de 15 K de surchauffe, cavité de vapeur (DVCM) au plancher de pression. [M]
- Filet de sécurité : un NaN né pendant un pas n'arrête plus le programme (bornes « clamp » tolérantes aux NaN) ; il est détecté en fin de pas et le pas est refait avec dt/4. [M]
- **GPU résident (f32)** : l'état reste sur le GPU ; ~256 octets relus par série de pas (dt, itérations, drapeaux), l'image rendue sur le GPU, l'état complet relu seulement pour les statistiques (~2 fois/s), les outils et la sauvegarde. Gère TOUT ce que gère le CPU : jusqu'à 32 fluides, miscibilité et diffusion des espèces, cp(T), chimie (réactions réversibles comprises), flammes épaissies, rayonnement, évaporation de Dalton, ébullition / cavitation, cellules spéciales, entrées de bord, solides isothermes / chauffants, chauffage au pinceau, QDM conservative ou semi-lagrangienne / MacCormack. Pas de temps, Newton + gradient conjugué multigrille, sous-pas de conduction / diffusion et filet de sécurité (retour arrière dt/4) décidés sur le GPU. Validé contre le CPU (16 tests : mêmes résultats à l'arrondi f32 près pas à pas). Décision : CPU f64 (référence) et GPU f32 maintenus en parallèle ; le GPU f64 ne vaut pas le coup (2 à 64× plus lent sur les cartes grand public). [M]