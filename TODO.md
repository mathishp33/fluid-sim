Il n’y a pas de rayonnement, alors qu’il domine à 1000 K ou plus. Pas de combustion, ni d’humidité.
Le low-Mach filtre les ondes acoustiques et les chocs. C’est voulu, mais le son n’existe pas.
Pression thermodynamique uniforme : l’effet de la stratification hydrostatique sur ρ et sur le travail de pression est négligé. C’est acceptable pour quelques mètres.
2D uniquement. La turbulence 2D (cascade inverse) diffère de la 3D, et il n’y a pas de modèle de sous-maille.


POUR LE BUG :
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
 -> Peut-être qu'il faut revoir l'architecture/la codebase entière.

## Prioritaire

# PROCHAIN
Fait la Cinétique chimique avec ce que tu as anoncé. 
Puis ajoute la possibilité d'avoir des flammes/explosions.



Faire dépendre la diffusivité de la paire de fluides, au lieu d’une valeur par fluide.
Ajouter une enthalpie de mélange.
Ajouter la dissolution d’un gaz dans un liquide (loi de Henry). 
C’est l’usage naturel du cas gaz + liquide miscibles, aujourd’hui seulement signalé par un avertissement.

METTRE GPU EN F64, TOUT EN F64

 - Door / Valve CellEffect !!!!!!!!
 - Compressor !!!!
 - plus de CellEffect !!!!

 - réactions chimiques (ajoutées par l'utilisateur)

advection Rudman / Raessi–Pitsch

IMPLEMENTER TENSION LIQUIDE/LIQUIDE
GPU résident : ne plus recopier tout l’état à chaque pas.

TENSEUR DE CONTRAINTES. Je n’ai pas implémenté le tenseur complet : les termes ∇·(μ∇uᵀ) et −⅔μ∇·u sont omis. 
Ils comptent surtout quand μ varie fortement sur une courte distance, par exemple une flamme ou un jet à 1000 K. 
Les inclure demande de réécrire le traitement des parois, et je peux le faire si tu veux ce niveau de fidélité.
cp et D. cp et la diffusion du traceur restent constants. Pour l’air, cp varie d’environ 3 % de 300 à 600 K.
Retard d’un pas. Les propriétés sont évaluées sur la température du début de pas.
Benchmarks. L’effet est négligeable dans la cavité à ΔT = 10 K, mais lance-le pour confirmer, en particulier le test de conduction à 600 K.

 - plus d'exemples de materiaux (miel)
 - créateur personnalisé de matériaux

1. encodeur de conditions initiales, fenetre séparé pour faire des experiences précises, pas à pas, en tout sauvegardant, exemple : 
   - je veux de la température là, puis j'accélère ici, puis je lance la simulation sur 5s puis j'arrête et je sauvegarde des statistiques.
   -> trouver une "interface" ou un moyen d'implémenter ça ?
3. save tout le contexte (densité, special_cells, pressions, ...) PAS QUE SOLIDES
3. PCG (Ax = b) ? pas sur
5. Teleporter() kind of broken, destination size is not customizable.

CHANGEMENT D ETATS, REACTIONS CHIMIQUES ?

## Moins prioritaire

## Pas prioritaire
1. faire une option 3D

## Ancien / Autre (Quand même à faire)

1. Optimiser la simulation de fluide (Conjugate Gradient ?, PCG ?, Multi-grid ?)
2. Meilleures conditions aux limites (Periodic, Open, PressureOutlet, VelocityOutlet)
3. Forces externes (archimède, frottements/pas frottements)
4. solides mobiles Solid(position, accéleration, masse, angle, vitesse angulaire, inertie, dynamique, forme)

FLIP/PIC ?
Indicateur vorticité
Écoulement compressible


# Fait (Fonctionnalités)
 - solveur Red-Black Gauss-Siedel
 - SOR
 - Sous-pas adaptifs CFL
 - Advection semi-Lagrangienne, upwind et MacCormack (MUSCL ordre 2)
 - grille MAC
 - solides immobiles
 - cellules spéciales
 - diagnostiques de divergence, pression, énérgie, masse, CFL, ...
 - température fluide & solide
 - viscosité (faible)
 - equation Low-Mach pour gaz
 - equation de Tait pour liquides
 - 