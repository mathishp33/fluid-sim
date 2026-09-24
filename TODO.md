1. Ajouter un système de vélocité aux bornes pour créer un flux avec vitesse initiale
3. sauvegarder automatiquement au lancement de la simulation la configuration
4. optimiser pour avoir des belles perfs (ORDRE IMPORTANT): 
 - mesurer delta t simulation, rendu et FPS
 - paralléliser le renderer
 - mesurer le nombre de threads Rayon (j'en ai 16)
 - profiler step() (diffusion, advection, divergence, pressure, correction)
 - optimiser le solveur de pression

J'ai mesuré 