Noter que pour le moment on ne supporte qu'un seul type de fluide en même temps.
 - faire une disjection de cas: 
   - gaz : formule de gaz parfait (ou autre), compressible
   - liquide : formule (choisir un modèle réaliste) , incompressible 
rho peut varier de 0 à +inf
P peut varier de -inf à +inf
T peut varier de 0 à +inf 
Si jamais c'est trop d'un coup, n'implémente pas (complétement) les gaz d'un coup

AJOUTER LA MASSE DANS CHAQUE CELLULE (LA DENSITE ACTUELLE NE LE REPRESENTE PAS) ? ????????????

## Prioritaire

Viscosité variable : μ constant, ν = μ/ρ. Ça compte pour un gaz chaud, où ν varie beaucoup.
Propriétés dépendant de T : k(T), μ(T) avec une loi de Sutherland pour l’air.
Advection moins diffusive : la semi-lagrangienne bilinéaire lisse beaucoup les panaches. MacCormack ou BFECC améliorent nettement le rendu visuel pour un coût modéré.

 - plus d'exemples de materiaux (créateur de matériaux)
 - créateur personnalisé de matériaux
 - viscosité considéré constante mais varie en fonction de la température

1. encodeur de conditions initiales (ex: je veux créer une condition initiale particulière de vitesses
 donc j'entre une formule f(x,y)=... dans une fenetre spéciale puis j'importe ces CI. cette fenetre sera 
 accessible depuis la barre haute de main_window.rs) pour faciliter les conditions initiales.
2. save tout le contexte (densité, special_cells, pressions, ...) PAS QUE SOLIDES
3. PCG (Ax = b) ? pas sur
4. force externes
5. Teleporter() kind of broken, destination size is not customizable.
6. Ajouter une cellule spéciale Heater()

## Moins prioritaire
1. backtrace RK2 ou MacCormack

## Pas prioritaire
1. faire une option 3D

## Ancien / Autre (Quand même à faire)

3. Optimiser la simulation de fluide (Red/Black Gauss-Siedel ?, SOR ?, Conjugate Gradient ?, PCG ?, Multi-grid ?)
4. Viscosité de vitesse
5. Température
6. Différents fluides FluidMaterial(density, viscosity, diffusion, temperature, color)
7. Meilleures conditions aux limites (Periodic, Open, PressureOutlet, VelocityOutlet)
8. Plusieurs fluides (changer le système de "density")
9. Forces externes (gravité, archimède, frottements/pas frottements)
10. Forces sur les solides mobiles Solid(position, accéleration, masse, angle, vitesse angulaire, inertie, dynamique, forme)


Flottabilité
Surface libre (liquide, air, interface) -> séparation liquide | air
Tension de surface
FLIP/PIC ?
Indicateur vorticité
Écoulement compressible


# Fait (Fonctionnalités)
 - solveur Red-Black Gauss-Siedel
 - SOR
 - Sous-pas adaptifs CFL
 - Advection semi-Lagrangienne
 - grille MAC
 - solides immobiles
 - cellules spéciales
 - diagnostiques de divergence, pression, énérgie, masse, CFL, ...
 - température fluide & solide
 - viscosité (faible)
 - equations Low-Mach pour gaz et Boussinesq pour liquides