Il n’y a pas de rayonnement, alors qu’il domine à 1000 K ou plus. Pas de combustion, ni d’humidité.
Le low-Mach filtre les ondes acoustiques et les chocs. C’est voulu, mais le son n’existe pas.
Pression thermodynamique uniforme : l’effet de la stratification hydrostatique sur ρ et sur le travail de pression est négligé. C’est acceptable pour quelques mètres.
2D uniquement. La turbulence 2D (cascade inverse) diffère de la 3D, et il n’y a pas de modèle de sous-maille.

## Prioritaire

 - intégration egui (plusieurs solides, fluides (eau, miel, huile) en meme temps (miscibilité, ...))
 - plusieurs matériaux de fluides en meme temps
 - réactions chimiques (ajoutées par l'utilisateur)

advection Rudman / Raessi–Pitsch

GPU résident : ne plus recopier tout l’état à chaque pas.

Tenseur de contraintes. Je n’ai pas implémenté le tenseur complet : les termes ∇·(μ∇uᵀ) et −⅔μ∇·u sont omis. 
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