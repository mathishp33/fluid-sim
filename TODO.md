

## Prioritaire
1. MAC
2. bug: régler diffuse_density non conservatif
3. bug: régler advect_density (et advect_velocity) 
4. Faire implémentation des SpecialCell
5. 
6. remplacer solver Poisson par solver Red-Black Gauss-Siedel

## Moins prioritaire

## Pas prioritaire
1. faire une option 3D

## Ancien / Autre

Optimiser la simulation de fluide (Red/Black Gauss-Siedel ?, SOR ?, Conjugate Gradient ?, PCG ?, Multi-grid ?)
Forces externes (gravité, archimède, frottements/pas frottements)
Forces sur les solides mobiles Solid(position, accéleration, masse, angle, vitesse angulaire, inertie, statique/dynamique, forme)
Viscosité de vitesse
Différents fluides FluidMaterial(density, viscosity, diffusion, temperature, color)
Température
Flottabilité
Plusieurs fluides (changer le système de "density")
Surface libre (liquide, air, interface) -> séparation liquide | air
Tension de surface
FLIP/PIC ?
Indicateur vorticité
Écoulement compressible
Meilleures conditions aux limites (Periodic, Open, PressureOutlet, VelocityOutlet)
Adaptation automatique du dt (condition CFL)
Conservation & Diagnostics (CFL, pressure_residuals)