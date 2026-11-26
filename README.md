# fluid-sim — simulateur de fluides 2D en Rust

Deux solveurs sur une grille MAC 2D, avec une interface egui :

- **monophasique** : un fluide (gaz low-Mach ou liquide), température, solides immobiles, sur CPU ou GPU (wgpu) ;
- **multiphasique** : plusieurs fluides à interfaces nettes (PLIC), pression implicite (Newton + gradient conjugué multigrille),
  tension de surface, ébullition, condensation, cavitation, évaporation, gel et fusion, chimie et flammes, poudres et grains (tas,
  lits, feux de charbon et de bois), solides mobiles (corps qui flottent, coulent, rebondissent et échangent leur chaleur). Le CPU
  calcule en f64 (référence), le GPU en f32 (wgpu / WGSL), avec les mêmes formules.

## Lancer

```
cargo run --release
```

La fenêtre de départ propose le simulateur monophasique ou le multiphasique (scénarios préréglés, pinceaux de matériaux, cellules
spéciales, éditeur de matériaux), et la fenêtre d'expérience.

Une expérience reproductible (protocole `.exp` en français, voir docs/EXPERIENCES.md) se rejoue sans fenêtre :

```
cargo run --release -- experience experiences/rebonds.exp [--gpu] [--sortie dossier]
```

Les résultats (mesures en CSV, journal, états) vont dans `experiences/résultats/`.

## Commandes de la fenêtre de simulation

| Touche | Effet |
|---|---|
| clic gauche maintenu | peint le matériau, l'outil choisi |
| espace | pause / reprise |
| flèche droite / flèche haut | en pause : avance d'un pas / de dix pas |
| V | change l'affichage (matériaux, température, pression, vitesse…) |
| C (Maj + C) | rayon du pinceau plus grand (plus petit) |
| M | pinceau rond / carré |
| I | solides en blanc |
| H | rabat ou rouvre les panneaux |
| S / L | sauve / charge la scène |
| R (Maj + R) | remet l'écoulement à zéro (toute la scène) |
| Échap | retour aux réglages |

## Tests

```
cargo test --release
```

Plus de 200 essais : bilans exacts, solutions analytiques, corrélations et expériences publiées, accord GPU / CPU. Comptez environ
une heure sur la machine de développement (les essais GPU passent un par un). Les diagnostics longs sont ignorés par défaut
(`cargo test --release -- --ignored nom_du_diagnostic`).

## Documentation (`docs/`)

- **MESURES.md** — la simulation face aux valeurs de référence, chiffres à l'appui ;
- **BUGFIXES.md** — défauts connus et défauts corrigés ;
- **TODO.md** — ce qui reste à faire, par priorité, et ce qui est fait ;
- **MULTIPHASE.md** — le solveur multiphasique (modèles, validation, portage GPU) ;
- **GRANULAIRE.md** — poudres et grains (transport, chute, frottement, réactions, solidification) ;
- **SOLID.md** — solides mobiles (corps rigides : couplage, contacts, chaleur) ;
- **EXPERIENCES.md** — format des protocoles d'expérience et ligne de commande.
