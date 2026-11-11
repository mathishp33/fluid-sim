

## Prioritaire
1. scinder fluid.rs en fluid.rs et cpu.rs
2. basculer tout step() sur le GPU en mode GPU
3. faire implémentation des SpecialCell

## Moins prioritaire
1. backtrace RK2 ou MacCormack

## Pas prioritaire
1. faire une option 3D

## Ancien / Autre (Quand même à faire)

1. Adaptation automatique du dt (condition CFL)
2. Conservation & Diagnostics (CFL, pressure_residuals)
3. Optimiser la simulation de fluide (Red/Black Gauss-Siedel ?, SOR ?, Conjugate Gradient ?, PCG ?, Multi-grid ?)
4. Viscosité de vitesse
5. Température
6. Différents fluides FluidMaterial(density, viscosity, diffusion, temperature, color)
7. Meilleures conditions aux limites (Periodic, Open, PressureOutlet, VelocityOutlet)
8. Plusieurs fluides (changer le système de "density")
9. Forces externes (gravité, archimède, frottements/pas frottements)
10. Forces sur les solides mobiles Solid(position, accéleration, masse, angle, vitesse angulaire, inertie, statique/dynamique, forme)


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