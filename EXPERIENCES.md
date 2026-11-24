# Expériences reproductibles — protocoles, fenêtre d'expérience, ligne de commande

Une expérience est un **protocole** : un fichier texte (`.exp`) écrit en français qui dit d'où l'on part, ce que l'on pose, ce que
l'on fait à quel moment et ce que l'on mesure. Par exemple, « de la température ici, puis j'accélère là, puis je simule 5 s,
j'arrête et je sauvegarde des statistiques » s'écrit :

```
départ scénario "Rupture de barrage" grille 120 x 80 maille 5 mm
mesurer toutes les 10 ms : stats, T en (0.1, 0.05), max vitesse dans partout
chauffer disque (0.1, 0.05) rayon 2 cm à 350 K pendant 1 s
simuler 1 s
accélérer rect (0.3, 0) (0.4, 0.1) de (8, 0) m/s² pendant 0.5 s
simuler 5 s
sauver état "fin"
```

Le même protocole s'exécute dans la **fenêtre d'expérience** (écran de démarrage → « Fenêtre d'expérience »), pour le regarder et
le mettre au point, ou en **ligne de commande**, sans affichage, pour enchaîner des séries. Les résultats vont dans
`résultats/<nom du protocole>/`, à côté du protocole :

- `mesures.csv` : une ligne par relevé ; séparateur `;`, point décimal, nombres en notation scientifique ;
- `journal.txt` : chaque action, avec son temps simulé, et les avertissements ;
- les états complets demandés (`.mps`), relisibles par les fenêtres et par `départ état` ;
- `protocole.exp` : la copie exacte du protocole exécuté.

Exemples : `experiences/barrage.exp`, `experiences/chauffage.exp` (balayage de trois températures), `experiences/champs.exp`
(conditions initiales par formules).

## 1. Ce qui est garanti

- **Les actions tombent pile à leur heure.** Le pas qui précède une échéance est raccourci (`dt_max`). Quand l'échéance tombe entre
  un et deux pas, ce pas est partagé en deux, pour éviter un pas minuscule. Une échéance est atteinte à un millionième de pas près.
- **Mesurer ne change pas le résultat.** Un relevé est fait au premier pas qui atteint son heure, et le temps réel de ce pas est
  noté ; les mesures ne raccourcissent jamais un pas. Un test vérifie que la même expérience, avec et sans mesures, donne un état
  final identique au bit près.
- **Sur le CPU (f64), rejouer redonne les mêmes nombres au bit près**, quel que soit le nombre de fils (testé avec 1 et 3 fils).
  Le GPU (f32) suit le même protocole avec les mêmes échéances, plus vite, mais sans cette garantie : ses résultats restent
  proches de ceux du CPU.
- **Le protocole se lit en mètres.** L'origine est en bas à gauche et y vers le haut : le même protocole tourne à n'importe
  quelle résolution.

## 2. Écriture

- Une commande par ligne ; `#` commence un commentaire.
- Les mots-clés s'écrivent avec ou sans accents ni majuscules (`départ` = `depart`).
- Les noms (fluides, mélanges, matériaux, fichiers, scénarios) vont entre guillemets. Un début de nom suffit s'il est unique :
  `"barrage"` pour « Rupture de barrage », `"méthane"` pour « méthane–air ».
- Les nombres s'écrivent avec un **point** décimal (`2.5`) ; la virgule sépare les coordonnées : `(0.2, 0.1)`.
- Les unités sont facultatives ; sans unité, c'est le SI :
  - longueurs : m, cm, mm ;
  - durées : s, ms, min, h ;
  - températures : K, °C ;
  - vitesses : m/s, cm/s, km/h ;
  - accélérations : m/s² (ou m/s2) ;
  - pressions : Pa, kPa, bar, atm ;
  - puissances : W/m³.
- Dans une paire, chaque composante peut avoir son unité, ou une seule unité suit la parenthèse : `(5, 0) m/s²`.
- Une erreur donne sa ligne : « ligne 4 : unité « m » : il faut ici une température (K, °C) ». Les noms inconnus sont signalés
  avec la liste des noms disponibles.

## 3. Commandes

### Départ (avant toute action)

| Commande | Effet |
|---|---|
| `départ scénario "nom" [grille NX x NY] [maille DX] [gravité G]` | un scénario préréglé ; par défaut grille 80 x 60, maille 1 cm, gravité 9.81 m/s² |
| `départ état "fichier.mps"` | un état complet enregistré : fenêtre multiphasique (« Sauver l'état complet »), ou `sauver état` d'une expérience |
| `matériaux "fichier.json"` | une bibliothèque de matériaux exportée par l'éditeur (sinon les matériaux par défaut) |
| `calcul gpu` ou `calcul cpu` | le calcul demandé (la fenêtre et `--gpu` / `--cpu` peuvent l'imposer) |

Les fichiers sont lus par rapport au dossier du protocole.

### Zones

| Zone | Mailles |
|---|---|
| `partout` | tout le domaine |
| `disque (x, y) rayon r` | centre à moins de r |
| `rect (x1, y1) (x2, y2)` | centre dans le rectangle |
| `point (x, y)` (ou `en (x, y)`) | la maille qui contient le point |
| `trait (x1, y1) (x2, y2) … rayon r` | centre à moins de r de la ligne brisée (trait de pinceau) |

Une maille fait partie de la zone si son centre y est. Une zone trop petite pour contenir un centre prend la maille de son centre.

### Conditions initiales et actions instantanées

Toutes les commandes de ce tableau peuvent servir de conditions initiales (avant le premier `simuler`), ou d'actions en cours
d'expérience.

| Commande | Effet |
|---|---|
| `remplir <zone> avec "fluide" [à T]` | fluide pur ou mélange nommé, à T (par défaut la température ambiante) ; un liquide au-dessus de son ébullition est posé en vapeur |
| `température <zone> T` | change la température, composition gardée (les masses suivent : la fermeture reste juste) |
| `solide <zone> "matériau" [isotherme T \| chauffant P] [à T]` | solide passif, à température imposée, ou chauffant (W/m³) |
| `corps disque (x, y) rayon r "matériau"`, `corps rect (x1, y1) (x2, y2) "matériau" [angle a]`, `corps polygone (x1, y1) (x2, y2) (x3, y3) … "matériau"` | corps rigide FIXE aux bords lissés (fraction exacte de chaque maille, voir SOLID.md) : le fluide de son volume est chassé, l'écoulement le contourne ; angle en degrés (ou `rad`) ; polygone convexe. Les corps mobiles viendront |
| `effacer <zone>` | retire solides et cellules spéciales (fluide ambiant) |
| `allumer <zone> [à T]` | noyau de gaz brûlés (2200 K par défaut) |
| `vitesse <zone> (vx, vy)` | impose la vitesse des faces intérieures de la zone ; la projection du pas suivant la rend admissible |
| `puits <zone> [v]` | cellules qui aspirent le fluide (0.2 m/s par défaut) |
| `émetteur <zone> "fluide" (vx, vy) [à T]` | buses qui créent et expulsent un fluide |
| `grille <zone> [trous d] [ouverte f]` | grille qui arrête les grains (trous de 1 cm, 60 % ouverte par défaut) |
| `bord gauche\|droite\|bas\|haut mur` | paroi |
| `bord … ouvert [p]` | ouvert à la pression p (1 atm par défaut) au point le plus haut du bord |
| `bord … entrée (vx, vy) [à T] ["fluide"]` | entrée à vitesse imposée (premier liquide par défaut) |
| `champ …` | champ de valeurs, voir ci-dessous |
| `gravité (gx, gy)` ou `gravité g` | gravité ; un nombre seul est dirigé vers le bas |
| `pas max d` | pas de temps maximal |

### Actions à durée

| Commande | Effet |
|---|---|
| `chauffer <zone> à T [pendant d]` | cible de température du FLUIDE de la zone, comme le pinceau « Chauffer » : elle est atteinte avec la constante de temps du chauffage (0.2 s), à la puissance maximale près. Sans durée, jusqu'à la fin. Pour un solide : `solide … isotherme` ou `chauffant`. |
| `accélérer <zone> de (ax, ay) [pendant d]` | accélération de volume : des cellules « accélérateur » sont posées, puis retirées au bout de la durée. Sans durée, elles restent. Les solides et les cellules spéciales déjà posées sont laissés. |

La fin d'une action à durée est une échéance comme une autre : elle tombe pile.

### Champs de valeurs

```
champ T fichier "carte.csv" [origine haut|bas] [dans <zone>]
champ fraction "eau" fichier "niveaux.png" [valeurs a b] [dans <zone>] [à T]
champ solide "acier" fichier "murs.png" valeurs 1 0
champ vx formule "0.2*y/ly" dans rect (0, 0.2) (0.4, 0.3)
champ T formule "293.15 + 60*exp(-((x - 0.2)^2 + (y - 0.06)^2)/0.03^2)"
```

- **Grandeurs :**
  - `T` (K). Composition gardée, les masses suivent ; dans un solide, sa température.
  - `vx`, `vy` (m/s). Faces intérieures, moyenne des deux mailles voisines ; les parois restent fermées.
  - `fraction "fluide"` (0 à 1, fluide pur ou mélange). Le reste de la maille garde sa composition, sans le fluide posé ; s'il n'y
    a rien d'autre, c'est l'ambiant. Avec `à T`, la maille est mise à cette température.
  - `solide "matériau"`. Un solide est posé là où la valeur vaut au moins 0.5 ; ailleurs, rien ne change.
- **Fichier CSV** (ou texte) :
  - une rangée de valeurs par ligne, séparées par `;`, `,`, des espaces ou des tabulations ;
  - case vide, `nan` ou `-` : la maille reste telle quelle ;
  - lignes vides et lignes qui commencent par `#` ignorées ;
  - la première rangée est en haut (`origine bas` pour l'inverse).
- **Image PNG** :
  - la valeur est la luminance, de 0 (noir) à 1 (blanc) ; `valeurs a b` donne a au noir et b au blanc ;
  - pixel transparent : la maille reste telle quelle ;
  - pour dessiner des murs en noir sur fond blanc : `champ solide "acier" fichier "murs.png" valeurs 1 0`.
- **Étalement** : la grille du fichier couvre la boîte de la zone (le domaine entier par défaut), quelle que soit la résolution.
  Elle est lue au centre de chaque maille, en interpolation bilinéaire (au plus proche pour les solides) ; les cases sans valeur
  sont écartées.
- **Formules** :
  - variables : `x`, `y` (m), `lx`, `ly` (taille du domaine) ; opérations : `+ - * / ^` et parenthèses ; constantes : `pi`, `e` ;
  - fonctions : `exp`, `ln`, `log10`, `sqrt`, `abs`, `sin`, `cos`, `tan`, `tanh`, `atan2`, `min`, `max`, `pow`, `hypot`,
    `clamp(v, a, b)`, `step(v)` (0 si v < 0, sinon 1) et `si(c, a, b)` (a si c > 0, sinon b) ;
  - valeurs en SI (K, m/s), sans unités.

### Temps

| Commande | Effet |
|---|---|
| `simuler d` | avance de la durée d |
| `simuler jusqu'à t` | avance jusqu'au temps simulé t |
| `pause` | met l'expérience en pause dans la fenêtre (ignorée en ligne de commande) |
| `marque "texte"` | repère dans le journal |
| `sauver état "nom"` | écrit `nom.mps` dans le dossier des résultats |

### Mesures

```
mesurer toutes les 10 ms : stats, T en (0.2, 0.1), moyenne p dans rect (0, 0) (0.1, 0.05), masse "eau" dans disque (0.3, 0.1) rayon 5 cm comme "eau du bassin"
```

- **Une seule période par protocole.** Plusieurs lignes `mesurer` ajoutent des colonnes.
- **Grandeurs** : `T`, `p`, `vx`, `vy`, `vitesse`, `rho`, `fraction "fluide pur"`, `chaleur` (puissance chimique, W/m³).
- **Relevés** :
  - `<grandeur> en (x, y)` : valeur dans la maille du point ;
  - `moyenne | max | min | somme <grandeur> dans <zone>` : la somme est une intégrale (× dx², par mètre de profondeur) ;
    les grandeurs du fluide ignorent les mailles solides, T les compte ;
  - `masse "fluide" [dans <zone>]` : en kg par mètre de profondeur ;
  - `stats` : dt, itérations de pression, pas refaits, masse de chaque fluide, énergies cinétique et interne, T et p extrêmes,
    vitesse maximale, chaleur injectée, et, s'il y a de la chimie, chaleur de réaction, puissance chimique et énergie rayonnée.
- **Libellés** : `comme "nom"` donne le titre de la colonne.
- **Moments des relevés** : au premier `simuler`, puis à chaque période, puis à la fin.

### Balayage

```
balayage T = 330 K, 350 K, 370 K
chauffer rect (0.1, 0) (0.2, 0.01) à $T pendant 3 s
```

- Le protocole est rejoué pour chaque valeur, `$T` remplacé dans le texte. Chaque variante a son dossier de résultats
  (`résultats/chauffage/T=330_K/`…).
- Il y a un seul balayage par protocole. Si les valeurs contiennent des virgules, séparez-les par `;`.

## 4. La fenêtre d'expérience

- **Onglet « Protocole »** :
  - éditeur du texte ;
  - « Vérifier » : relit le texte et reconstruit l'aperçu ; les erreurs s'affichent avec leur ligne ;
  - « Insérer… » : modèles de commandes ;
  - résumé : nombre de conditions initiales et d'actions, mesures, fin prévue ;
  - dossier des résultats.
- **Onglet « Conditions initiales »**, comme les pinceaux de la fenêtre multiphasique :
  - remplir, solide (passif, isotherme, chauffant), corps rigide fixe (disque au clic, rectangle tiré : ligne `corps`), température,
    effacer, allumer, puits, émetteur, grille ;
  - au pinceau rond (rayon en mailles, affiché en mètres) ou au rectangle tiré à la souris.
- **Chaque geste devient une ligne du protocole**, insérée avant le premier `simuler`, par exemple
  `remplir trait (0.055, 0.055) (0.155, 0.085) rayon 0.02 avec "eau" à 293.15 K`.
  - Le geste se voit tout de suite.
  - À son relâchement, l'aperçu est reconstruit depuis le texte : ce que l'on voit est ce que la ligne de commande rejouera (testé
    au bit près).
  - « Annuler le dernier geste » retire la ligne.
- **« Charger un champ de valeurs »** : grandeur, formule ou fichier CSV / PNG, sens des rangées, niveaux de l'image, dans le
  dernier rectangle tracé ou partout. Le bouton ajoute la ligne `champ …` correspondante.
- **Exécution** :
  - « ▶ Lancer » exécute le protocole depuis son départ, avec toutes les variantes d'un balayage ;
  - Pause et Reprendre (aussi la touche Espace), « Arrêter » (mesures et journal écrits) ;
  - « Conditions initiales » revient à l'aperçu ;
  - case « GPU ».
- **Panneau de droite** :
  - état, progression, ligne en cours ;
  - dernières mesures et courbe d'une colonne en fonction du temps ;
  - maille sous le curseur, avec ses coordonnées en mètres ; un **clic droit les copie** ;
  - journal et bilan.
- **Sur l'image**, les zones du protocole sont tracées : actions en blanc, actions à durée en orange, sondes de mesure en cyan.

## 5. Ligne de commande

```
fluid-sim experience protocole.exp [--gpu | --cpu] [--sortie dossier] [--fils n]
cargo run --release -- experience experiences/chauffage.exp
```

La commande affiche le journal et la progression toutes les deux secondes. Le code de sortie est 0 si toutes les variantes sont
allées au bout. `--sortie` change la racine des résultats ; `--fils` fixe le nombre de fils du CPU, sans effet sur les résultats.

## 6. Comment c'est fait

| Fichier | Rôle |
|---|---|
| `simulation/mp_experiment.rs` | modèle du protocole, zones, application des actions, déroulement (`Runner`), mesures, sorties, `run_cli` |
| `simulation/mp_script.rs` | lecture du texte (unités, erreurs avec leur ligne), balayage, écriture des zones (`zone_text`), insertion des lignes de conditions initiales |
| `simulation/mp_fields.rs` | champs de valeurs : CSV, PNG (crate `png`), formules, étalement sur la zone |
| `window/experiment_window.rs` | la fenêtre egui |

- **Les conditions initiales** sont le départ et les actions instantanées d'avant le premier `simuler` (`initial_state`). La fenêtre
  les montre et les complète. L'expérience, elle, repart toujours du texte : départ reconstruit, puis toutes les lignes, comme la
  ligne de commande.
- **Le déroulement** (`Runner::advance`) traite le travail par tranches de temps réel, ce qui laisse la fenêtre fluide.
  - Les actions consécutives sont appliquées d'un bloc : sur le GPU, une relecture et un envoi de l'état par bloc.
  - Le prochain pas vise la plus proche échéance : fin du `simuler` ou fin d'une action à durée. Sur le CPU, il s'appuie sur le pas
    prévu par `choose_dt`.
  - Sur le GPU, des séries de pas sont soumises, bornées pour s'arrêter avant l'échéance.
  - Le chauffage passe par les cibles du pinceau « Chauffer », l'accélération par des cellules « accélérateur » : rien de nouveau
    dans le solveur, ni sur le CPU ni sur le GPU.
- **Tests** (`mp_experiment_tests.rs`, et ceux de la fenêtre) :
  - lecture et erreurs ;
  - balayage ;
  - zones (un disque redonne exactement le pinceau) ;
  - champs CSV, PNG et formules, la fermeture restant juste ;
  - échéances exactes ; mesurer sans effet sur le résultat ; même résultat avec 1 et 3 fils ;
  - chauffage tenu puis relâché ;
  - sorties ; reprise depuis un état enregistré au bit près ;
  - ligne de commande avec balayage ; protocoles d'exemple ;
  - GPU, mêmes échéances et résultats proches ;
  - fenêtre : trait de pinceau, ligne et aperçu identiques, rectangle et champ, expérience menée au bout avec une pause.

## 7. Limites et suites

- Pas de conditions ni de boucles, hormis le balayage. « simuler jusqu'à ce que T max < 400 K » serait une suite naturelle.
- Une sonde lit la valeur de sa maille, sans interpolation. Il n'y a pas de flux à travers une ligne, ni d'images des champs en
  sortie (PNG).
- Les gestes faits pendant une expérience ne sont pas enregistrés : la fenêtre ne peint qu'en mode « Conditions initiales ».
  Enregistrer en actions datées (« à t = 1.2 s, chauffer … ») ce que l'on fait à la souris pendant une expérience serait la suite.
- La commande `corps` pose des corps FIXES ; le mouvement imposé puis libre viendra (options `imposé`, `libre`, `pivot`… : voir
  SOLID.md § 5 et § 7).
