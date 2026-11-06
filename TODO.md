

## Prioritaire
1. rélger le bug: taille de souris pas assez petite
2. régler le bug: quand on ajoute à la souris de la densité sur des solides, la densité se duplique
3. rajouter des "classes" de solides, Door(), Sink(), Emitter(), Accelerator(), Source()
4. remplacer solver Poisson par solver Red-Black Gauss-Siedel

## Moins prioritaire
1. changer minifb pour un système plus performant (Vulkan ? OpenGL ?) → on doit garder le cross platform

## Pas prioritaire
1. faire une option 3D

## Ancien / Autre

Optimiser la simulation de fluide (Red/Black Gauss-Siedel ?, SOR ?, Conjugate Gradient ?, PCG ?, Multi-grid ?)
Forces externes (gravité, archimède, frottements/pas frottements)
Forces sur les solides Solid(position, accéleration, masse, angle, vitesse angulaire, inertie, statique/dynamique, forme)
Viscosité de vitesse
Différents fluides FluidMaterial(density, viscosity, diffusion, temperature, color)
Température
Flottabilité
Plusieurs fluides (changer le système de "density")
Surface libre (liquide, air, interface) -> séparation liquide | air
Tension de surface
Système de particules ?
FLIP/PIC ?
Indicateur vorticité
Écoulement compressible
Meilleures conditions aux limites (Periodic, Open, PressureOutlet, VelocityOutlet)
Sources / puits Sink(), Emitter(), Accelerator(), Deccelerator()
Adaptation automatique du dt (condition CFL)
Conservation & Diagnostics (div_max, tot_mass, Ec, CFL, pressure_residuals)