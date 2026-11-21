## Pas trié

pas d’humidité ?
Le low-Mach filtre les ondes acoustiques et les chocs. C’est voulu, mais le son n’existe pas.
Pression thermodynamique uniforme : l’effet de la stratification hydrostatique sur ρ et sur le travail de pression est négligé. C’est acceptable pour quelques mètres.
2D uniquement. La turbulence 2D (cascade inverse) diffère de la 3D, et il n’y a pas de modèle de sous-maille.


- REACTION AVEC SOLIDES (mobile) & POUDRES (carbones, bois)
- AJOUTER POUDRES (poudre a canon, carbone)
- TRANSFORMATION LIQUIDE -> SOLIDE (mobile)



# TODO — Simulation de fluide en Rust

Légende : **[M]** multiphasique · **[1]** monophasique · **[M+1]** les deux

---

## 1. Prioritaire

### Performances et architecture GPU
- **[M] GPU : fusionner des noyaux** — mesuré sur une vraie carte (AMD Radeon intégrée, Vulkan ; `cargo test --release gpu_speed_benchmark -- --ignored --nocapture`) : le GPU est 8 à 25× plus rapide que le CPU f64 (ms par pas, CPU / GPU pas à pas / GPU séries de 8 : barrage 80×60 40 / 5.2 / 4.5, 320×240 469 / 24.6 / 19.9 ; ébullition 80×60 41 / 9.3 / 9.2, 320×240 597 / 31.3 / 31.0 ; explosion H₂ 160×120 521 / 38.7 / 18.0, 320×240 1268 / 64.3 / 50.1). Mais les petites grilles sont limitées par le COÛT DE LANCEMENT (~15–20 µs par noyau, 285 à 1446 noyaux par pas) : ×16 cellules de 80×60 à 320×240 ne donne que ×4–7 en temps. La relecture de contrôle par pas coûte peu (séries de 8 ≈ pas à pas, sauf l'explosion H₂). Gains à chercher : moins de noyaux par pas (fusion, moins d'itérations Newton / PCG / sous-pas lancées une à une), pas des noyaux plus rapides. À remesurer sur une carte dédiée (NVIDIA / AMD).
- **[M] Flamme prémélangée NON CONVERGÉE EN TEMPS** (trouvé via le test GPU) : scène du test (64×12, dx = 1 mm, 0.02 s), CPU seul, dt_max défaut / 2e-5 / 1e-5 / 5e-6 → chaleur dégagée 234 / 256 / 268 / 271, front 0.0515 / 0.0545 / 0.0575 / 0.0585 m, 455 / 1001 / 2001 / 4000 pas. Au dt par défaut la flamme brûle ~14 % trop lentement : un terme dépend du dt (découpage chimie / diffusion / flamme épaissie, sous-pas d'espèces, relaxation « par pas » ?). À trouver ; la vitesse de flamme de 0.49 m/s (ci-dessous) et les facteurs A ont été recalés à CE dt. Peut-être lié au terme « par pas et non par seconde » de l'huile à 2000 K. Le test GPU de la flamme échouait sur la Radeon pour cette raison (GPU : autre suite de dt, chaleur 259, front +4 mailles ; accord pas à pas à 1e-3 K près, le CPU perturbé de 1e-4 sur T ne bouge pas) : tolérance à long terme élargie (20 %, 8 mailles) + accord pas à pas ajouté. Les 16 tests GPU passent sur la Radeon.
- [P] GPU : contournements de bogues de lavapipe — tampons de commandes de ≤ 400 lancements : **coût nul mesuré** sur la Radeon (`MP_GPU_BATCH_OPS=1000000` : mêmes temps à ±10 % près), à garder. Restent non mesurés (en dur dans le WGSL) : dans les noyaux par groupe, boucles à borne uniforme (P.np / P.ng, appartenance au groupe par sélection) plutôt que sur les membres lus dans la table, pas de `continue` ni de tableau privé dans une branche divergente, pas de sortie anticipée avant une boucle (sinon plantage du code JIT de lavapipe, selon les données). La chimie (au moins une itération par fil) ne s'active que sur adaptateur logiciel. Diagnostics : `MP_GPU_SYNC=1` soumet chaque lancement seul et affiche son nom (repère le noyau qui plante) ; `MP_GPU_BATCH_OPS=n` change la taille max des tampons.
- [M] GPU f32 : limites de précision connues — la fermeture volumique 1 − Σ m/ρ n'est connue qu'à ~1e-7 près (masses stockées en f32) : dans une cellule presque pleine de liquide, la vapeur produite par pas est bruitée de ~1 % et la pression du liquide de quelques Pa. Piste : stocker les masses des liquides en écart à une référence (m − ρ₀α).
- [M] Advection PLIC : les flux sont appliqués face par face dans l'ordre des indices (la donneuse est lue après ses flux entrants pour un écoulement vers +x, avant pour −x). La forme « flux pure » (tout évalué sur l'état d'avant le balayage, indépendante de l'ordre) a été essayée : elle diverge sur un jet d'eau entrant dans l'air (écarts de remplissage m/α ≠ ρ transportés sans amortissement). À creuser, puis à reporter sur le GPU (qui reproduit aujourd'hui exactement l'ordre du CPU).

### Robustesse : contacts très chauds ↔ froids
- **[M] Huile à 2000 K qui ATTEINT l'eau chaude (cuve fermée, 80×60) — le GPU se met encore en pause** (A VERFIFIER JE CROIS QUE CE BUG EST REGLE) (vers la trame 18–19 sur 40 ; ~4 avant la température par groupe). Le CPU passe les 40 trames (3 essais, T de l'huile perturbée de ±0.1 K), mais le contact déclenche une explosion de vapeur : p max 1–5 MPa, vitesses jusqu'à ~100 m/s, 100 à 580 pas refaits en 40 trames. Diagnostic de la pause GPU (`debug_raw_step`, exemple `hot_oil` du crate de test) : une cellule eau à 521 K + vapeur 743 K + huile 1920 K + air, déficit de volume 1 − Σ m/ρ = 0.6 ; depuis cet état, le pas échoue à TOUS les dt (dt/4 … dt/1024) : la température du groupe eau descend sous 0 K quel que soit dt (une partie du changement de phase / de la fermeture agit « par pas » et non « par seconde »), et la correction de fermeture donne des vitesses ∝ 1/dt — le CPU f64, repartant de cet état GPU, échoue aussi (vitesse 5e8 m/s). Le CPU évite simplement cet état sur sa propre trajectoire. Pistes : (1) trouver quel terme vide le groupe eau indépendamment de dt (cavité DVCM, borne « fermeture » de la vaporisation, travail de pression borné à ±50 % du volume par pas) ; (2) borner la correction de fermeture par une vitesse (∝ dt) plutôt que par une fraction de volume par pas ; (3) ébullition en FILM (Leidenfrost) : la vapeur isole l'huile de l'eau, la conductance d'échange ne le voit pas (corrélation de Bromley, ou couche de vapeur sous-maille).
- [M] Échange de chaleur entre groupes d'une cellule mixte : conduction de part et d'autre d'une interface supposée PLANE dans la cellule (G = λ·2k/(α·dx²)). Pour des gouttes / bulles dispersées sous la maille, l'aire d'échange est sous-estimée : modèle d'aire interfaciale + corrélations (Ranz–Marshall). λ est réglable dans la fenêtre (curseur « Échange de chaleur entre phases », case « Une seule température par cellule » = ancien modèle).
- [M] Couplage du pas χ = max_g a_g/(1 + a_g) (heuristique) : une petite phase non négligeable (eau à α ~ 1e-4 dans du gaz) fait traiter toute la cellule « à une température » (fermeture, travail de pression, chaleur latente) ; seules les traces (< 1e-9 de la capacité de la cellule) sont exclues. À remplacer par un couplage par paire de groupes.
- [M] Vapeur « verrouillée » sur son liquide (couplage parfait) dès qu'elle fait plus de 50 % de la masse de son groupe : seuil arbitraire, à remplacer par une conductance d'interface vapeur ↔ liquide (évaporation / condensation à l'interface).
- [M] Contact liquide ↔ liquide (huile / eau, liquides non miscibles) maintenant hors équilibre aussi : à valider sur une expérience (temps de mise en équilibre d'une couche d'huile chaude sur l'eau).
- [M] Flamme prémélangée : la vitesse de flamme calculée passe de 0.479 à 0.491 m/s (conduction des gouttes de brouillard d'eau dans les gaz brûlés, maintenant à leur propre température) ; borne du test élargie à 0.50 — à recaler si besoin (facteurs A).
- [M] Les poches de gaz très détendues (quelques centaines de Pa) refroidissent par détente jusqu'au plancher de 60 K : physique dans le modèle, mais c'est la conduction des voisines qui devrait les réchauffer plus vite.

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
- **Une température par groupe de phases dans les cellules mixtes** (CPU et GPU, mêmes formules ; groupe = phases miscibles entre elles : un liquide, un mélange de gaz…). Énergie et température par groupe ; échange implicite entre groupes à énergie exactement conservée (conductance de conduction, λ réglable, λ = ∞ = ancien modèle) ; conduction entre nœuds (groupes) des cellules voisines ; fermeture de la projection et travail de pression répartis par groupe avec un couplage χ du pas ; changement de phase piloté par la température du LIQUIDE, chaleur latente partagée comme la capacité qui a servi à estimer la masse changée ; chimie restreinte aux groupes réactifs ; rayonnement par groupe gazeux. Résultats : huile à 2000 K dans l'air froid → l'air des cellules mixtes reste ≤ 530 K (au lieu de ~1900 K aussitôt) ; huile à 600 K sur l'eau → plus de vaporisation éclair (5.7e-11 kg de vapeur au lieu de 2.1e-7) ; boîte d'air et d'huile → équilibre en ~4 s, énergie conservée à 1e-6 ; cuve fermée + huile à 2000 K au pinceau : 40 trames sans pause sur le CPU, plus calme qu'à une température (sur les 30 premières trames : p max 5 MPa au lieu de 12, vitesse max 100 m/s au lieu de 200 ; mais 580 pas refaits au lieu de 26), le GPU tient ~18 trames au lieu de ~4 (voir « Huile à 2000 K qui atteint l'eau »). Corrigé en route : trace de liquide portée à 560 K par la chaleur latente de condensation (ou sous 50 K à l'évaporation) → pas refusés en cascade ; une trace de 1e-17 kg de vapeur imposait χ = 1 à toute la cellule (et une différence CPU/GPU selon l'arrondi). 51 tests CPU (dont 3 nouveaux hors équilibre) et 16 tests GPU au vert. Tests GPU revus : l'accord « pas à pas » repart de l'état CPU à chaque pas (comme documenté ; les écarts accumulés étaient amplifiés ×10–100 par pas par la nucléation, ancienne version comprise) ; tolérance eau de l'ébullition sur plaque 5e-4 (l'ancienne version passait à 9.7e-5 pour 1e-4) ; chauffage à 2000 °C : comparaison à long terme ramenée à l'ordre de grandeur (le CPU contre lui-même, T initiale perturbée de 1e-9, donne une vapeur nette de 2e-7 à 5e-5 kg/m ; perturbée de 1e-6 à 1e-4, un apport d'énergie qui varie de 26 %). [M]
- **Huile chaude au pinceau dans une cuve fermée (eau + vapeur chauffée) : divergences numériques corrigées** (CPU et GPU, mêmes formules). Vérifié sans pause : huile à 2000 K dans l'air, huile à 400–500 K sur l'eau chaude ; huile à 2000 K sur l'eau chaude : le CPU passait, le GPU s'arrêtait après ~4 trames sur la limite de 1000 m/s (température unique) — ~18 trames avec la température par groupe (ci-dessus), pas encore réglé. Faiblesses corrigées : (1) Newton de pression linéarisé qui ne faisait que doubler la pression d'une poche de gaz recomprimée (la cellule devenait un puits qui aspirait ses voisines) et la rendait négative à la détente → mise à jour polytropique quand |δp| > 50 % de p (≤ 4× la tangente) ; (2) vitesses produites par un pas jamais contrôlées → pas refait (dt/4) si Courant > 2 ; (3) cavités vides dans un liquide non volatil (huile) → complétées de gaz ambiant à 10 Pa ; (4) détente et advection qui menaient T sous 0 dans les cellules presque vides → planchers à 60 K ; (5) chaleur latente extrapolée NÉGATIVE au-delà de 1450 K (masse de vapeur < 0) → bornée, et vaporisation bornée par la chaleur disponible ; (6) vaporisation d'un bloc quel que soit dt (vitesse ∝ 1/dt, le filet de sécurité ne pouvait rien) → débit volumique borné (phase_volume_limit / phase_relaxation_time) ; (7) grosse erreur de fermeture (> 0.1) héritée dans un liquide, corrigée en un pas par un schéma neutre-stable → corrigée à 25 % par pas. Message de pause plus utile (cause probable, « Reset flow » pour repartir). Tests CPU et GPU de non-régression. [M]
- Performances CPU : préconditionneur multigrille du gradient conjugué (≈ 20 itérations au lieu de ≈ 270 avec IC(0)), boucles par cellule parallélisées (rayon), profil de `step()` par partie. [M]
- Montée en température brutale (2000 °C au pinceau sur un liquide) sans mise en pause : chauffage progressif en temps simulé (puissance plafonnée), ébullition en volume au-delà de 15 K de surchauffe, cavité de vapeur (DVCM) au plancher de pression. [M]
- Filet de sécurité : un NaN né pendant un pas n'arrête plus le programme (bornes « clamp » tolérantes aux NaN) ; il est détecté en fin de pas et le pas est refait avec dt/4. [M]
- **GPU résident (f32)** : l'état reste sur le GPU ; ~256 octets relus par série de pas (dt, itérations, drapeaux), l'image rendue sur le GPU, l'état complet relu seulement pour les statistiques (~2 fois/s), les outils et la sauvegarde. Gère TOUT ce que gère le CPU : jusqu'à 32 fluides, miscibilité et diffusion des espèces, cp(T), chimie (réactions réversibles comprises), flammes épaissies, rayonnement, évaporation de Dalton, ébullition / cavitation, cellules spéciales, entrées de bord, solides isothermes / chauffants, chauffage au pinceau, QDM conservative ou semi-lagrangienne / MacCormack. Pas de temps, Newton + gradient conjugué multigrille, sous-pas de conduction / diffusion et filet de sécurité (retour arrière dt/4) décidés sur le GPU. Validé contre le CPU (16 tests : mêmes résultats à l'arrondi f32 près pas à pas). Décision : CPU f64 (référence) et GPU f32 maintenus en parallèle ; le GPU f64 ne vaut pas le coup (2 à 64× plus lent sur les cartes grand public). [M]