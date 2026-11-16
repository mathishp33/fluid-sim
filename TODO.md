Propriétés du fluide
ν, k, cp sont constants. Pour un gaz, μ ∝ T^0,7 environ et k ∝ T^0,8, donc ν varie fortement avec T et P. Pour le miel, ν(T) est énorme.
Le liquide n’a ni compressibilité, ni ébullition, ni cavitation. L’eau est monophasique : pas de surface libre, de vagues ni d’éclaboussures.
Le traceur est passif : il ne change ni la densité, ni la masse molaire, ni la chaleur spécifique. Un « panache de fumée chaude » ne se comporte donc pas comme un mélange réel.
Il n’y a pas de rayonnement, alors qu’il domine à 1000 K ou plus. Pas de combustion, ni de changement de phase, ni d’humidité.

Physique manquante
Le low-Mach filtre les ondes acoustiques et les chocs. C’est voulu, mais le son n’existe pas.
Pression thermodynamique uniforme : l’effet de la stratification hydrostatique sur ρ et sur le travail de pression est négligé. C’est acceptable pour quelques mètres.
2D uniquement. La turbulence 2D (cascade inverse) diffère de la 3D, et il n’y a pas de modèle de sous-maille.

L’énergie cinétique diminue : où part-elle ?
Dans ton code, elle part dans cinq puits, dont aucun n’alimente la température :
Dissipation numérique de l’advection semi-lagrangienne. C’est de loin le puits principal à cette résolution, et il n’a pas de contrepartie physique.
Projection. Ton pinceau clic droit injecte une vitesse non solénoïdale, et la projection retire la partie compressible. Dans un vrai gaz, cette énergie deviendrait des ondes sonores, puis de la chaleur.
Parois no-slip, avec le même mécanisme que la viscosité et la dissipation numérique.
Limiteur vmax et mesure : ton KE est calculé avec la vitesse moyennée aux centres de cellules, ce qui sous-estime l’énergie des modes à petite échelle.

## Prioritaire

Advection moins diffusive (MacCormack ou BFECC) : elle réduirait les +8 % de vitesse de la cavité à Ra = 1e5.
GPU résident : ne plus recopier tout l’état à chaque pas.

Viscosité variable : μ constant, ν = μ/ρ
Propriétés dépendant de T : k(T), μ(T) avec une loi de Sutherland pour l’air.
Advection moins diffusive : la semi-lagrangienne bilinéaire lisse beaucoup les panaches. MacCormack ou BFECC améliorent nettement le rendu visuel pour un coût modéré.

 - plus d'exemples de materiaux (miel)
 - créateur personnalisé de matériaux
 - viscosité considérée constante mais varie en fonction de la température

1. encodeur de conditions initiales, fenetre séparé pour faire des experiences précises, pas à pas, en tout sauvegardant, exemple : 
   - je veux de la température là, puis j'accélère ici, puis je lance la simulation sur 5s puis j'arrête et je sauvegarde des statistiques.
2. save tout le contexte (densité, special_cells, pressions, ...) PAS QUE SOLIDES
3. PCG (Ax = b) ? pas sur
5. Teleporter() kind of broken, destination size is not customizable.

CHANGEMENT D ETATS, REACTIONS CHIMIQUES ?

## Moins prioritaire

## Pas prioritaire
1. faire une option 3D

## Ancien / Autre (Quand même à faire)

3. Optimiser la simulation de fluide (Red/Black Gauss-Siedel ?, SOR ?, Conjugate Gradient ?, PCG ?, Multi-grid ?)
4. Viscosité de vitesse
5. Température
6. Différents fluides FluidMaterial(density, viscosity, diffusion, temperature, color)
7. Meilleures conditions aux limites (Periodic, Open, PressureOutlet, VelocityOutlet)
8. Plusieurs fluides (changer le système de "density")
9. Forces externes (archimède, frottements/pas frottements)
10. Forces sur les solides mobiles Solid(position, accéleration, masse, angle, vitesse angulaire, inertie, dynamique, forme)


Surface libre (liquide, air, interface) -> séparation liquide | air
Tension de surface
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
 - equation de Boussinesq pour liquides