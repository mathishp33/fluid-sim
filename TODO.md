

## Prioritaire
1. diagnostique des résidus de pressions, CFL, divergence RMS et moyenne, ... 
``` rust
    pub pressure_residual: f32,
    pub divergence_rms: f32,
    pub velocity_rms: f32,
    pub cfl: f32,
```
2. encodeur de conditions initiales (ex: je veux créer une condition initiale particulière de vitesses
 donc j'entre une formule f(x,y)=... dans une fenetre spéciale puis j'importe ces CI. cette fenetre sera 
 accessible depuis la barre haute de main_window.rs )
3. save tout le contexte (densité, special_cells, pressions, ...) PAS QUE SOLIDES
4. PCG (Ax = b)
5. viscosité de vitesse
6. force externes
7. thermodynamique (température)
9. Teleporter() kind of broken, destination size is not customizable.


## Moins prioritaire
1. backtrace RK2 ou MacCormack

## Pas prioritaire
1. faire une option 3D

## Ancien / Autre (Quand même à faire)

2. Conservation & Diagnostics (CFL, pressure_residuals)
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
 - diagnostiques de divergence, pression, énérgie, masse