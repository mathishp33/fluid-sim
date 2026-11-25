# Solides mobiles — problèmes, méthodes, architecture proposée

Objectif : des objets NETS qui bougent avec l'écoulement et le font bouger — un bloc de glace ou un bouchon qui flotte, une pierre
qui coule, un projectile, un piston, une palette qui fait des vagues, un clapet ou une porte sur charnière (la « porte / vanne » des
cellules spéciales du TODO), un agitateur qui tourne. Ce ne sont pas des poudres : la phase granulaire (GRANULAIRE.md) ne rend ni
leur forme ni leur inertie propre.

Document de conception (2026-11-24). Les choix du § 9 sont validés. Les étapes 0 (préparation, aucun résultat changé), 1 (corps
FIXES : l'eau les contourne sans y entrer), 2 (mouvement IMPOSÉ : piston, palette, batteur à vagues) et 3 (corps LIBRES : flotteur,
clapet, pendule, corps qui coule) sont faites (§ 8, `mp_bodies.rs`). La traînée d'un corps EN MOUVEMENT converge en maillage, mais
reste approchée à 10 mailles par diamètre (§ 8). Viennent ensuite les contacts (§ 7).

**En bref — recommandation**

- Corps **rigides** seulement (pas de déformation), en 2D : trois degrés de liberté (DDL) x, y, θ.
- Mouvement **libre, bloqué axe par axe, autour d'un pivot, ou imposé** : un masque de DDL et une loi de mouvement par corps, pas une
  architecture à part (§ 5). On commence par le mouvement imposé.
- Formes simples : disque, rectangle, polygone convexe. À chaque pas, la position exacte donne la FRACTION de chaque maille occupée
  par le corps : pas de marches d'escalier, pas de saut de pression quand une maille change de camp.
- Couplage par **domaine fictif à une vitesse** (Patankar ; « Rigid fluid » de Carlson et al. en image de synthèse) : le corps est une
  « phase » rigide du mélange ; la projection actuelle pousse le fluide hors du volume qu'il balaie, puis la vitesse dans le corps
  est ramenée à un mouvement rigide. C'est le cousin direct du modèle actuel (une vitesse par maille, fermeture Σ α = 1). Repli si
  les corps plus légers que l'eau posent problème : couplage monolithique dans la projection (Batty et al. 2007, Gibou & Min 2012).
- Contacts souples (ressort-amortisseur) contre les parois et entre corps ; une température par corps au début ; corps inertes (ni
  fusion ni combustion) au début.
- Les solides fixes actuels (mailles pleines) restent tels quels.

## 1. Ce qu'il faut simuler

| Cas | Mouvement | Ce qui compte |
|---|---|---|
| bloc de glace, bouchon, bûche qui flotte | libre | poussée d'Archimède ; corps plus léger que l'eau (masse ajoutée) ; surface libre autour |
| pierre ou bille qui coule, projectile | libre | traînée ; arrivée au fond (contact, rebond) |
| piston, seringue | translation sur un axe, souvent imposée | volume balayé (compression d'un gaz) ; étanchéité |
| palette à vagues, agitateur, hélice | imposé (translation ou rotation) | entraînement du fluide |
| clapet, porte, vanne sur charnière | rotation autour d'un pivot, libre ou imposée | couple du fluide ; butées ; ouverture réglable = angle imposé |
| plusieurs objets qui s'entrechoquent | libre | contacts entre corps |

Hors champ : corps déformables (élasticité, tissus), ruptures, corps plus minces que ~2 mailles (une feuille), 3D.

## 2. Les solides aujourd'hui

- Un solide est une maille PLEINE : `solid[c]` = indice du matériau dans `reg.solids` (−1 = fluide), avec ρ, cp, k propres, une
  énergie et une température (conduction fluide ↔ solide), et un comportement thermique `thermal[c]` (passif, isotherme, chauffant).
- Les faces d'une maille solide sont fermées : `face_velocities` (mp_pressure.rs) met la vitesse à 0 dès qu'une des deux mailles est
  solide ; glissement libre pour la viscosité, non-glissement pour le frottement des grains (mp_friction.rs).
- `is_solid(c)` est testé partout : ~130 endroits côté CPU (transport PLIC, pression, forces, thermique, espèces, grains, percolation,
  rendu) et ~80 dans le WGSL. Une maille ne change de camp que par un outil (pinceau), entre deux pas.
- Pour des solides mobiles, cela pose quatre problèmes :
  1. En tout ou rien, un corps qui avance d'une maille fait disparaître d'un coup le fluide de la maille couverte. Dans un liquide
     raide (Tait), c'est un pic de pression.
  2. La vitesse de paroi vaut partout 0.
  3. Rien ne calcule la force du fluide sur un solide.
  4. La chaleur d'un solide est attachée à ses mailles, pas à l'objet.

## 3. Les problèmes à résoudre

1. **Représentation** : où est le corps à l'intérieur d'une maille ? En tout ou rien (escalier), ou en fraction de volume (et
   d'aire des faces).
2. **Mailles qui changent de camp.** Elles sont « fraîches » quand le corps les découvre, « mortes » quand il les couvre. D'où vient
   le fluide d'une maille fraîche (masse, énergie, composition) ? Où va celui d'une maille morte ? C'est la première source
   d'oscillations de pression parasites des méthodes à interface nette (Lee et al. 2011). Seo & Mittal (2011) les réduisent d'un ordre
   de grandeur en rétablissant la conservation locale de la masse.
3. **Condition à la paroi** : le fluide colle au corps (vitesse V + ω × r), et le volume balayé par le corps doit sortir de la
   maille. La projection doit le savoir.
4. **Force et couple du fluide sur le corps** : on peut intégrer la pression et le frottement sur sa surface, ou les déduire de
   l'échange de quantité de mouvement.
5. **Intégration du mouvement et masse ajoutée.** Un corps léger dans un fluide dense (bouchon dans l'eau) accélère avec le fluide
   qu'il déplace. Un couplage explicite (forces du pas d'avant) devient instable quand ρ_corps / ρ_fluide descend sous ~1 (Causin,
   Gerbeau & Nobile 2005). La limite est d'environ 1.2 pour le forçage direct d'Uhlmann (2005), et d'environ 0.3 avec les améliorations
   de Kempe & Fröhlich (2012). Dans l'air, tout solide est des centaines de fois plus dense, sans souci ; dans l'eau, le bois (~0.6),
   la glace (0.92) et le liège (~0.24) sont sous la barre.
6. **Contraintes de mouvement** : axes bloqués, pivot, mouvement imposé, butées, ressorts.
7. **Contacts** : corps ↔ parois et solides fixes, corps ↔ corps. Le film de fluide qui se vide entre deux surfaces qui se
   rapprochent (lubrification) n'est pas résolu à l'échelle de la maille.
8. **Chaleur** : le corps emporte sa chaleur et échange avec le fluide qu'il traverse.
9. **Le reste du simulateur.** Il faut aussi traiter :
   - les interfaces PLIC (surface libre autour d'un flotteur) ;
   - les grains (un corps pousse le sable, le sable frotte sur lui) ;
   - la percolation et les grilles ;
   - la chimie et les changements de phase (corps qui fond ou brûle : plus tard) ;
   - les cellules spéciales ;
   - le GPU (tout doit avoir son miroir WGSL) ;
   - la sauvegarde de l'état, l'interface, les protocoles d'expérience.

## 4. Comment font les autres

**Maillage qui suit le corps.** C'est l'approche d'OpenFOAM (maillage déformable, maillages superposés « overset »), de Fluent et
de STAR-CCM+. Le maillage épouse le corps et se déforme ou glisse avec lui, et un solveur à 6 DDL avec contraintes intègre le corps.
C'est très précis à la paroi, mais hors de portée ici (grille cartésienne fixe, VOF, GPU).

**Frontières immergées (forçage).** La grille est fixe ; le corps agit par une force de volume qui impose sa vitesse au fluide dans
une couche de mailles autour de sa surface (Peskin 2002 ; forçage direct : Fadlun et al. 2000, Uhlmann 2005 ; revue de Mittal &
Iaccarino 2005).
- C'est le standard des simulations de suspensions résolues (des milliers de sphères).
- Le couplage est explicite, d'où la limite de densité du § 3, repoussée par Kempe & Fröhlich (2012) et par Breugem (2012).
- Une variante pondère le forçage par la fraction de solide de la maille (Kajishima et al. 2001) : simple et lisse.

**Interface nette, mailles coupées.** Les mailles traversées par la paroi sont découpées (volumes et aires partiels) ou complétées
par des mailles fantômes (Udaykumar et al. 2001 ; Gerris et Basilisk de Popinet, pour des parois fixes).
- La précision est d'ordre 2 à la paroi.
- C'est délicat : les petites mailles coupées bornent le pas de temps, et les mailles fraîches ou mortes font osciller la pression
  (§ 3).

**Porosité FAVOR, objets mobiles généraux (FLOW-3D).** Chaque maille porte la fraction de volume ouverte au fluide, chaque face sa
fraction d'aire ouverte, et les équations sont écrites avec ces fractions (Hirt & Sicilian 1985).
- Pour un objet mobile (modèle GMO de FLOW-3D), les fractions sont recalculées à chaque pas depuis la position de l'objet.
- Le mouvement est imposé ou couplé à l'écoulement : libre à 6 DDL, autour d'un point fixe ou d'un axe fixe ; plusieurs objets.
- C'est l'approche industrielle la plus proche de ce simulateur (grille fixe, VOF).

**Domaine fictif : le corps est du fluide rendu rigide.** Tout le domaine, corps compris (avec sa masse volumique), est traité comme
un fluide ; la vitesse dans le corps est ensuite contrainte à un mouvement rigide.
- La contrainte passe par un multiplicateur de Lagrange distribué (Glowinski, Pan, Hesla & Joseph 1999), ou directement : on projette
  la vitesse sur le mouvement rigide de même quantité de mouvement et de même moment cinétique (Patankar et al. 2000 ; Sharma &
  Patankar 2005 ; Apte, Martin & Patankar 2009).
- L'image de synthèse fait la même chose : « Rigid fluid » (Carlson, Mucha & Turk 2004) simule le corps comme du fluide puis le rend
  rigide, pour un surcoût proportionnel au nombre de corps.
- La force du fluide n'a pas à être intégrée sur la surface : elle est dans la quantité de mouvement.
- Apte et al. signalent des difficultés quand la masse volumique saute fortement sur une maille ; ce simulateur traite déjà des sauts
  eau / air de 1000 sur une maille.

**Pénalisation.** Le corps devient un milieu poreux de perméabilité quasi nulle (terme de Brinkman ; Angot, Bruneau & Fabrie 1999).
C'est très simple, mais raide, et la précision dépend du paramètre de pénalisation. La percolation d'Ergun existe déjà ici : un corps
« grille très fermée » serait une pénalisation.

**Couplage monolithique dans la projection.** Les vitesses des corps deviennent des inconnues de la projection.
- C'est l'approche de Batty, Bertails & Bridson (2007 : cadre variationnel avec fractions d'aire, écoulement sous la maille), de
  Robinson-Mosher, Shinar & Fedkiw (2008) et de Gibou & Min (2012).
- Gibou & Min obtiennent un système symétrique défini positif, résolu par gradient conjugué préconditionné, inconditionnellement
  stable et comparé à l'expérience.
- C'est stable quelle que soit la densité du corps.
- En revanche, chaque corps ajoute à la matrice un bloc dense (de rang 3 en 2D), qui ralentit le gradient conjugué quand les corps
  sont nombreux (Batty).

**Méthodes particulaires.**
- SPH : DualSPHysics calcule les forces du fluide sur les objets flottants et confie leur mouvement, leurs liaisons (charnières,
  glissières, ressorts) et leurs collisions à Project Chrono, une bibliothèque de dynamique multicorps, avec un couplage dans les deux
  sens (Martínez-Estévez et al.).
- Lattice Boltzmann : rebond sur les parois mobiles et échange de quantité de mouvement (Ladd 1994).

**Moteurs de corps rigides.** Box2D (2D) et Rapier (Rust, 2D et 3D) gèrent les collisions, les liaisons et le blocage des axes un
par un (`LockedAxes` : translation en x, en y, rotation ; en 2D seuls x, y et la rotation comptent). OpenFOAM fait de même pour ses
corps à 6 DDL : contraintes `axis`, `line`, `orientation`, `plane` et `point`.

| Méthode | Exemples | Paroi | Corps légers | Ce qu'il faudrait changer ici |
|---|---|---|---|---|
| maillage mobile | OpenFOAM, Fluent | exacte | stable (couplage implicite) | tout : hors de portée |
| frontières immergées (forçage) | Uhlmann, Breugem | diffuse, ordre 1–2 | ρ/ρ_f ≳ 0.3 à 1.2 | une force par maille + forces de surface |
| mailles coupées | Basilisk, Udaykumar | nette, ordre 2 | selon le couplage | transport, pression, thermique, GPU : lourd |
| porosité FAVOR mobile | FLOW-3D GMO | lisse, ordre 1 | selon le couplage | fractions d'aire et de volume partout |
| domaine fictif (une vitesse) | Patankar, Carlson | lisse, ordre 1 | inertie vue par la projection (à vérifier) | une « phase » rigide dans le mélange + projection de rigidité |
| monolithique | Batty, Gibou & Min | ordre 1–2 | inconditionnel | matrice de pression avec les blocs des corps |

## 5. Faut-il restreindre les mouvements ?

**Pas dans l'architecture.** Tous les codes cités offrent le mouvement général, avec des contraintes par-dessus :
- OpenFOAM : `axis`, `line`, `plane`, `point`, `orientation` ;
- FLOW-3D : 6 DDL, point fixe, axe fixe ;
- Rapier : axes verrouillés un par un ;
- Chrono : charnières, glissières, ressorts.

Une contrainte ne coûte presque rien. En 2D, le mouvement du corps est (V_x, V_y, ω) :
- bloquer un axe, c'est mettre sa composante à zéro et ne pas l'intégrer ;
- un pivot, c'est une rotation pure autour d'un point fixe, avec le moment d'inertie I_G + M d² ;
- un mouvement imposé, c'est une loi x(t), y(t), θ(t) donnée.

**La restriction ne simplifie pas** les vrais problèmes du § 3 : géométrie, mailles qui changent de camp, couplage avec la pression,
contacts. Ce sont les mêmes pour un piston et pour un bouchon libre.

**Elle apporte** des cas utiles (piston, clapet, flotteur qui ne fait que monter et descendre) et des validations plus propres (un
seul DDL à comparer à une solution connue).

**Proposition.** Chaque corps a :
- un masque de DDL (x, y, rotation : libres ou bloqués) ;
- un pivot optionnel (le point fixe autour duquel il tourne) ;
- un mode : **libre** (couplé à l'écoulement), **imposé** (loi de mouvement : vitesse constante, oscillation, table temps →
  position) ou **fixe** ;
- plus tard : des butées (angle mini et maxi d'un clapet), des ressorts et des amortisseurs (amarrage).

Pour garder la simplicité, c'est l'ORDRE de mise en œuvre qui compte, plus que la restriction : d'abord le mouvement imposé (le corps
pousse le fluide, en sens unique : piston, palette, agitateur), puis le mouvement libre avec le masque.

## 6. Ce que je propose pour ce simulateur

### 6.1 Corps et géométrie

```rust
pub struct RigidBody {
    pub shape: Shape,                      // Disque { r } | Rectangle { w, h } | Polygone { sommets dans le repère du corps }
    pub material: usize,                   // indice dans reg.solids : ρ, cp, k, couleur
    pub pos: [f64; 2], pub angle: f64,     // centre de masse, orientation
    pub vel: [f64; 2], pub omega: f64,
    pub dof: Dof,                          // x, y, rotation : libres ou bloqués
    pub pivot: Option<[f64; 2]>,
    pub motion: Motion,                    // Libre | Fixe | Imposé(loi)
    pub temp: f64,                         // une température par corps au début
}
```

- La masse et l'inertie se déduisent de la forme et de ρ.
- À chaque pas, la position donne pour chaque maille la fraction **α_c** occupée par le corps, calculée exactement : polygone découpé
  par le carré de la maille ; pour le disque, polygone fin ou formule exacte. On a Σ α_c dx² = aire du corps, sans escalier.
- Aucun champ « corps » n'est advecté : la forme est recalculée depuis la position, sans diffusion numérique.

### 6.2 Couplage : une phase rigide dans le mélange (domaine fictif)

Le modèle actuel a déjà une vitesse par maille et la fermeture Σ m_k/ρ_k = 1. Le corps y entre comme une phase de plus, rigide et
incompressible. À chaque pas :

1. **Fermeture** : Σ_k m_k/ρ_k(p, T) + α_c = 1.
   - Quand le corps avance, α_c croît dans les mailles devant lui. L'écart de fermeture entre dans le second membre de la projection,
     comme aujourd'hui l'erreur de fermeture héritée, et la projection pousse le fluide hors de la maille. Derrière le corps, le fluide
     revient.
   - Il n'y a pas de maille « fraîche » ni « morte » : la fraction change continûment, et le transport conserve la masse et l'énergie
     du fluide.
2. **Quantité de mouvement** : sur les faces du corps, la masse volumique est celle du mélange, corps compris. La prédiction part de
   la vitesse rigide du pas précédent, plus la gravité.
3. **Projection** : inchangée. Le corps y est un « fluide » dense (ou léger) : la pression voit son inertie, donc la masse ajoutée
   est prise en compte sans itérer.
4. **Rigidification**, sur les faces couvertes par le corps (fraction α_f) :
   - quantité de mouvement P = Σ ρ_c α_f u_f dA et moment cinétique L = Σ ρ_c α_f (r × u)_f dA ;
   - V = P/M et ω = L/I, en mettant à zéro les axes bloqués (pivot : rotation pure) ;
   - puis u_f ← u_f + α_f (V + ω × r − u_f), une pondération par la fraction comme chez Kajishima et al. (2001) ;
   - en mouvement imposé, V et ω sont donnés par la loi.
5. **Mouvement** : la position et l'angle avancent avec (V, ω), plus les forces de contact (§ 6.3). La force du fluide, à afficher
   et à mesurer, vaut M dV/dt − M g − contacts.
6. **Pas de temps** : la vitesse de surface du corps, |V| + |ω| r_max, entre dans le CFL.

**Pourquoi cette méthode.**
- C'est celle qui change le moins le solveur : pas de nouvelle condition à la limite dans la projection, pas d'intégrale de surface,
  pas de terme dans la matrice.
- Elle est lisse (fractions).
- Elle traite la masse ajoutée par la projection.

**Ses limites.**
- Le non-glissement est étalé sur une maille, et la traînée est d'ordre 1 : il faudra une dizaine de mailles ou plus dans le
  diamètre pour une traînée juste à quelques pour cent (à mesurer).
- Les corps minces sont mal rendus.
- La rigidification après la projection laisse une petite erreur de volume à la paroi, que la fermeture corrige au pas suivant.

**À vérifier tôt** : un corps nettement plus léger que l'eau (liège). Si ce n'est pas stable, on se replie sur le couplage
monolithique (Gibou & Min 2012) : les vitesses des corps deviennent des inconnues de la projection (trois par corps).

**Ce que l'étape 1 a changé à ce schéma : les ouvertures des faces.** La limite ci-dessus (rigidifier APRÈS une projection qui ne
voit pas le corps laisse une erreur de volume à la paroi) est grave ici. Dans l'eau, très raide (Tait), une erreur de volume de 1e-6
fausse la pression de 2 kPa. La projection sait donc où est le corps :
- chaque face est ouverte au fluide à A = 1 − α (α de la plus couverte de ses deux mailles ; 0 si l'une est entièrement couverte) ;
- la vitesse corrigée de la face est A (u* − dt/ρ ∇p) + (1 − A) V_corps (avec V_corps = 0 pour un corps fixe), et la matrice de
  pression a les mêmes coefficients A dt/ρ ;
- la fermeture Σ m/ρ + α = 1 est donc exacte autour des corps, comme en porosité FAVOR. Le « une seule place où la face est fermée
  et à quelle vitesse » écarté à l'étape 0 revient sous cette forme fractionnaire.

La vitesse de face est alors la vitesse débitante : le flux de volume par unité de face. Le transport (PLIC), la percolation, la
conduction et la diffusion passent par les mêmes ouvertures. Comme une maille couverte à 1 − φ a toutes ses faces ouvertes au plus à
φ, conductance et capacité y diminuent ensemble, et les schémas explicites gardent le pas du fluide pur (sans cela, la conduction
s'emballait sur la trace d'une maille couverte).

**Ce que l'étape 2 a ajouté : déplacer le fluide balayé.** Pour un corps qui avance, l'ouverture 1 − α a un défaut : la face par
laquelle le fluide d'une maille presque couverte doit sortir se ferme à mesure que la maille se couvre. Le dernier fluide passait
donc par une fente de plus en plus étroite, et la pression nécessaire tendait vers l'infini : Δp ≈ ρ V dx / (A dt), déjà 6 kPa
dans l'air pour une fente A = 1e-5, un piston à 5 cm/s, des mailles de 1 cm et un pas de 10 ms (estimation).
- **Ouvertures exactes, essayées et écartées.** On a d'abord essayé la part exacte de chaque face hors du corps (segment découpé par
  le corps). Mais une petite maille dont une face reste grande ouverte voit passer plus de fluide qu'elle n'en contient : erreurs de
  fermeture jusqu'à 6 % autour d'un disque fixe. On garde donc 1 − α.
- **Déplacement direct du fluide balayé.** Le fluide qu'un corps balaie dans une maille passe aussitôt (masses et énergies, comme en
  redistribution des « petites mailles » des méthodes à mailles coupées) à la maille voisine du côté opposé au corps (−∇α). Une maille
  qu'il découvre prend le sien à cette voisine. La projection n'a plus qu'à évacuer, par des faces bien ouvertes, l'excédent ou le
  manque de cette voisine : au plus le volume balayé en un pas.
- **Travail du corps.** Le déplacement ne porte pas de travail : on ajoute p·Δα à l'énergie de la maille voisine, réparti entre ses
  groupes comme le travail de pression. Sans cela, le gaz qu'un piston comprime ne chauffait que de 5 % au lieu de 15 %.
- **Entraînement.** Sur une face bordée par une maille couverte, le fluide est entraîné à la vitesse du corps :
  vitesse débitante A (u* − dt/ρ ∇p) + A (1 − A) V_corps (non-glissement étalé sur une maille).
- **Pas de temps.** La vitesse de surface des corps entre dans le CFL.

**Ce que l'étape 3 a ajouté : les corps libres, en couplage MONOLITHIQUE.** La rigidification après coup (domaine fictif) ne
convenait plus : la projection ne voit pas l'intérieur des corps (ouvertures 1 − α). La vitesse d'un corps libre est donc une inconnue
de la projection, comme chez Batty et al. (2007) et Gibou & Min (2012).
- **Volume balayé anticipé.** Le volume que le corps balaiera au pas suivant, dt J·V, entre dans le bilan des mailles. J est la
  sensibilité du volume de chaque maille au mouvement du corps, routée vers la maille qui recevra le fluide déplacé (comme
  `push_fluid`).
- **Force de pression.** −Σ p_m J_m (travail virtuel, symétrique du bilan), plus l'écart hydrostatique ρ_m g·(x − x_m) entre le
  centre de la maille m et le bord, intégré exactement le long du bord : la poussée d'Archimède est exacte où que soit le corps dans
  ses mailles.
- **Système résolu.** Matrice A + (dt/dx²) J W Jᵀ (W : mobilité du corps dans ses mouvements permis), symétrique définie positive,
  de rang 3 par corps en plus ; le gradient conjugué la résout avec quelques itérations de plus. La masse ajoutée est implicite : un
  corps quatre fois plus léger que l'eau (liège) est stable, alors que le couplage explicite, essayé en comparaison, diverge.
- **Frottement.** Les contraintes visqueuses sont calculées sur la vitesse du MÉLANGE fluide + corps des faces, u + (1 − A) V_corps
  (l'intérieur d'un corps qui bouge n'est pas un fluide au repos), et intégrées sur la part du corps des faces qu'il porte ; leur
  dépendance linéaire en V est implicite. Un premier essai faisait payer au corps la quantité de mouvement que la projection donne au
  fluide qu'il entraîne : c'était compter deux fois l'inertie de ce fluide, déjà dans la masse ajoutée (C_a mesuré 1.4 à 1.9).
- **Conditions de pas.** La surface d'un corps libre parcourt au plus 0.15 maille par pas. Près d'un corps qui bouge, la viscosité
  reste explicite : la viscosité implicite ne voit pas sa vitesse, et un disque dans l'huile faisait exploser les pas plus longs.
- **Fluide collé au corps (tout corps en mouvement, imposé ou libre).** Aux faces qu'un corps en mouvement porte, le débit A V_corps
  du fluide qui avance avec lui est porté par le déplacement géométrique (`push_fluid`). Le transport des masses et les bilans de
  volume (divergence de la projection, travail de pression) ne voient que le débit relatif u − A V_corps (`body_attached_flux`).
  Avant, ce fluide sortait deux fois de la maille qu'il quitte, une fois transporté et une fois déplacé, et la projection le freinait
  devant le corps. Révélateur : un disque qui avance à la vitesse du courant, où toute force est un défaut, avait le fluide qui le
  borde à 0.8 V et une traînée de 2.4 ½ρU²d.
- **Quantité de mouvement du mélange.** Le transport de la quantité de mouvement porte la vitesse du mélange w = u + (1 − A) V_corps,
  régulière (rigide dans les corps) ; le débit s'en déduit ensuite avec la position des corps après leur pas (`mixture_to_flux`). Le
  débit, nul dans le corps et réduit à A V au bord, ne se transporte pas comme une vitesse.

### 6.3 Contacts

- Une force de ressort-amortisseur s'applique, comme dans la méthode des éléments discrets, quand le corps pénètre une paroi, un
  solide fixe (maille carrée) ou un autre corps : F_n = k δ − c v_n, plus un frottement de Coulomb tangent.
- Formes convexes (disque, rectangle, polygone convexe) : test des axes séparateurs.
- La raideur est liée au pas : si besoin, le mouvement des corps est intégré en sous-pas dans un pas fluide.
- Plus tard, si les contacts se compliquent (empilements, liaisons entre corps) : Rapier (`rapier2d`, en Rust) pour les corps, le
  fluide restant ici.

### 6.4 Chaleur

- Au début, une température par corps (capacité M cp). Elle échange avec le fluide des mailles qu'il couvre, par une conductance
  proportionnelle à la longueur de paroi dans la maille (k du fluide sur une demi-maille). C'est valable pour un corps métallique,
  dont le nombre de Biot est faible.
- Plus tard : un champ de température attaché au corps (une grille dans son repère) et la conduction interne.

### 6.5 Avec le reste

- **Surface libre** : les fractions des fluides sont renormalisées sur la partie ouverte de la maille, comme pour les grains (`agn`).
- **Grains** : la fermeture les pousse hors du corps comme le fluide. Pour le frottement, le corps est une paroi mobile
  (non-glissement à sa vitesse).
- **Cellules spéciales, percolation, chimie** : pas dans les mailles d'un corps au début ; les corps sont inertes (ni fusion ni
  combustion).
- **GPU** : la même logique que le CPU, comme pour le reste. Un noyau recalcule les fractions (formes simples), des réductions (une
  par corps) font la rigidification, et le mouvement des corps tient en quelques nombres.
- **État, interface, protocoles** : l'état complet passe en version 12 ; il faudra des scénarios, un outil « Corps » dans la fenêtre,
  et des commandes dans le protocole d'expérience (EXPERIENCES.md), par exemple :
  - `corps disque (0.2, 0.3) rayon 2 cm "Glace" libre`
  - `corps rect … "Acier" pivot (x, y)`
  - `corps … imposé vitesse (0.1, 0) m/s`

## 7. Étapes et validation

| Étape | Contenu | Validation |
|---|---|---|
| 0 (faite) | préparer (§ 8) : géométrie exacte, fractions, contraintes, fermeture, état, rendu ; aucun résultat changé | Σ α dx² = aire (1e-12) quels que soient la position et l'angle ; empreintes physiques des 18 scénarios identiques au bit près |
| 1 (faite) | corps fixes : poser un corps chasse le fluide (trace dans les mailles couvertes) ; ouvertures des faces dans la projection, le transport, la percolation, la conduction et la diffusion ; commande `corps`, outil de la fenêtre d'expérience | fermeture à 1e-9 après la pose ; eau au repos autour de deux corps : 1.8e-7 m/s, pression à 3 mPa près ; canal autour d'un disque : débit identique entrée / disque / sortie, passages à 0.284 m/s (0.273 attendus), rien n'entre dans le corps ; barrage contre un muret et un disque : masse d'eau à 4e-16, aucun pas refait |
| 2 (faite) | mouvement imposé : vitesse et rotation constantes, oscillation sinusoïdale, pivot ; fluide balayé déplacé, travail p·dV, entraînement | piston dans un tube d'air fermé : γ = 1.4014 par la pression et 1.3990 par la température (attendu : 1.40), masse de gaz à 2e-7 ; disque tiré dans l'eau : la pression d'une sonde saute de 1.5 Pa au plus quand il traverse les mailles, masse d'eau à 2e-16 ; palette sur pivot : eau brassée à 0.11 m/s près des bouts (ω r = 0.11) ; lois intégrées exactement (une oscillation revient pile à son départ) |
| 3 (faite) | mouvement libre + masque de DDL et pivot ; couplage monolithique pression–corps ; force du fluide sur tout corps (mesurable) | corps de la densité de l'eau immobile (2 µm en 1 s, force du fluide = poids à 6 chiffres) ; chute dans l'air : vitesse à 0.1 % ; masse ajoutée d'un disque lâché ou accéléré dans l'eau : C_a = 0.94 à 1.04 (1 attendu) ; flotteurs : tirant d'eau à 0.7 mm près (bois 0.6, liège 0.24), oscillation qui s'amortit ; pendule dans l'air : période à 0.03 % ; clapet flottant redressé à la verticale ; traînée d'un disque FIXE à Re 20 / 40 : −11 % / −9 % de Dennis & Chang (1970) ; disque TIRÉ à Re 20 : C_D 2.68 à 10 mailles par diamètre, 1.61 à 20 (Dennis & Chang : 2.05), pression à 20 mailles 1.26 (≈ 1.24) ; disque qui sédimente dans l'huile : vitesse limite à −23 % |
| 4 | contacts | disque posé sur le fond (repos sans tremblement), deux disques ; rebond dans un liquide qui faiblit avec le nombre de Stokes (Gondret et al. 2002, sphères) |
| 5 | chaleur, GPU, protocole | bille chaude dans l'eau : refroidissement quasi exponentiel ; accord CPU / GPU |
| plus tard | fonte d'un bloc de glace, combustion d'une bûche, liaisons, butées, ressorts, formes non convexes | |

## 8. Ce qui est fait : étapes 0 (préparer l'architecture sans changer un résultat), 1 (corps fixes), 2 (mouvement imposé) et 3 (corps libres)

1. **`mp_bodies.rs`**, testé à part (6 tests) :
   - corps rigide `RigidBody` (forme, matériau, position, angle, vitesses, axes libres `Dof`, pivot, mode `Motion` : libre, fixe,
     imposé ; température) ;
   - formes convexes : disque (polygone régulier de 96 côtés, de même aire : écart au cercle < 0.04 % du rayon), rectangle,
     polygone convexe (vérifié, recentré sur son centre de masse) ;
   - aire et moment d'inertie exacts ;
   - fraction de chaque maille couverte, par découpage exact du polygone par le carré de la maille (Sutherland–Hodgman) :
     Σ α dx² = aire à 1e-12 près, quels que soient la position et l'angle ;
   - vitesse d'un point du corps, mouvement permis (`constrained_velocity` : axes bloqués, pivot avec le théorème de Huygens,
     imposé, fixe), avance (rotation exacte autour d'un pivot).
2. **`MultiphaseSim`** : `bodies` (vide), `body_frac` (nul), recalculée par `update_body_fractions` ; état complet en version 12
   (corps et fractions enregistrés et vérifiés au chargement) ; instantané du pas (retour arrière du filet de sécurité).
3. **Point d'entrée du domaine fictif dans le solveur** : la fraction de corps compte dans la fermeture, Σ m/ρ + α_c = 1. Elle
   entre dans `cell_volume` (erreur héritée, changement de phase, cavités) et dans `cell_closure_g` (itérations de Newton),
   seulement s'il y a des corps.
4. **Rendu** : la part de chaque maille couverte par un corps prend la couleur de son matériau (ou de sa température), bords
   lissés par la fraction. Les infos de maille donnent la fraction.
5. **GPU** : `gpu_unsupported_reason` refuse les scènes avec des corps (calcul sur le CPU, la raison est affichée) jusqu'à
   l'étape 5.

**Écart au plan d'origine.** Le point « un seul endroit qui dit si une face est fermée et à quelle vitesse » n'a pas été fait : il
ne sert pas le domaine fictif retenu au § 9. Dans un corps, les faces restent ouvertes et portent sa vitesse rigide ; c'est la
rigidification (étape 1) qui l'impose. Ce point ne reviendrait qu'avec le repli monolithique.

**Non-régression.** Les empreintes de l'état PHYSIQUE (masses, énergies, températures, pression, vitesses, grains, flammes, bilans)
des 18 scénarios, en 80 × 60 et 160 × 120 après 20 pas (`cpu_physics_fingerprints`), sont identiques avant et après au bit près.
Les empreintes de l'état complet (`cpu_state_fingerprints`), elles, changent forcément : l'état a deux champs de plus.

**Étape 1 (corps fixes, faite le 2026-11-24).**
- `add_body` chasse le fluide du volume du corps : masses et énergies ramenées à 1 − α, composition et température gardées, une
  trace de 1e-6 dans les mailles entièrement couvertes. `clear_bodies` rend le fluide.
- Les ouvertures des faces (§ 6.2) entrent dans la projection, le transport, la percolation, la conduction et la diffusion des
  espèces.
- La commande `corps disque | rect | polygone … "matériau" [angle a]` des protocoles (EXPERIENCES.md) et l'outil « Corps rigide
  fixe » de la fenêtre d'expérience (disque au clic, rectangle tiré) posent les corps.
- La « masse du corps dans ρ » prévue au départ n'est pas utile pour un corps fixe : la part couverte ne bouge pas, seule l'inertie
  du fluide compte. Elle reviendra avec les corps libres.
- Le corps n'échange pas encore de chaleur avec le fluide : la trace des mailles couvertes est isolée et le corps garde sa
  température (étape 5). Les grains ne le voient pas encore : leur sédimentation ignore les corps.
- Tests (`mp_bodies.rs`) : fermeture après la pose, eau au repos, canal autour d'un disque, barrage contre deux corps ; commande et
  outil ; empreintes physiques des 18 scénarios toujours identiques au bit près.

**Étape 2 (mouvement imposé, faite le 2026-11-24).**
- Loi `Motion::Imposed` : vitesse et rotation constantes plus une oscillation sinusoïdale de période donnée, éventuellement autour
  d'un pivot. `move_imposed` l'intègre exactement : l'oscillation revient pile à son point de départ, quel que soit le pas.
- À chaque pas, après le transport, les corps avancent (`move_bodies`), le fluide balayé est déplacé avec le travail du corps
  (`push_fluid`, § 6.2), et la vitesse du corps aux faces bordées par une maille couverte sert à l'entraînement dans la projection.
  Le retour arrière du filet de sécurité remet les corps à leur place.
- Commande `corps … vitesse (vx, vy) | rotation ω | oscillation (vx, vy) période T | oscillation rotation ω période T | pivot (x, y)`
  (EXPERIENCES.md). Exemples : `experiences/piston.exp` (adiabatique), `experiences/vagues.exp` (batteur à vagues). État complet en
  version 13.
- Depuis l'étape 3, la force du fluide sur un corps imposé est calculée et mesurable, et le mouvement imposé profite des corrections
  du fluide collé au corps (§ 6.2). Effets : disque tiré dans l'eau, saut de pression d'un pas 9.9 Pa au lieu de 1.5 (le routage du
  fluide balayé vers une seule voisine fait encore du bruit, § 8) ; palette 0.119 m/s au lieu de 0.114 ; piston identique au bit
  près.

**Étape 3 (corps libres, faite le 2026-11-25).**
- Corps `Motion::Free` : axes bloqués un par un (`Dof`), pivot, masse volumique propre (`rho`, état complet en version 14).
  `move_bodies` les avance à la vitesse de la projection précédente. Un corps libre s'arrête net contre un bord du domaine ou un
  solide : il n'y a pas encore de contacts.
- Projection (`mp_pressure.rs`, `BodyCoupling`, `BodyLowRank`) : voir § 6.2. La force du fluide sur chaque corps est dans
  `body_force` (le frottement à part dans `body_force_visc`). Mesures : `corps n x | y | angle | vx | vy | rotation | fx | fy | couple`.
- Commande `corps … libre [bloqué x | y | rotation] [pivot (x, y)] [vitesse …] [masse volumique ρ]` (EXPERIENCES.md). Exemples :
  `experiences/flotteurs.exp` (bois et liège), `experiences/clapet.exp`.
- Deux défauts trouvés en route, sur une cuve FERMÉE pleine d'huile où coule un disque (son niveau de pression ne tient qu'à la
  compressibilité) :
  - le volume que le corps vient de déplacer, anticipé par la projection précédente, était pris pour une erreur de fermeture
    héritée et seulement en partie corrigé au-delà de 10 % : le niveau oscillait de ±5 kPa ;
  - la pression des mailles ENTIÈREMENT couvertes, qui ne tient qu'à la trace de fluide, bondissait au moindre écart de volume et
    tombait au plancher ; la cavité, comblée de gaz, partait dans le sillage et le calcul divergeait.
  Corrigés : `push_src`, et `body_trace` (pression non résolue, reprise de la voisine quand le corps découvre la maille). Il reste
  un niveau qui oscille de quelques kPa dans une cuve fermée pleine de liquide, sans effet sur les forces.
- Validations : § 7. Diagnostics (`--ignored`) : `added_mass_of_a_disc_diag`, `imposed_acceleration_force_diag`,
  `drag_of_a_fixed_disc_diag`, `towed_drag_resolution_diag`, `co_moving_disc_diag`, `settling_disc_diag`.
- **Traînée d'un corps EN MOUVEMENT (corrigée le 2026-11-25).** Dans la première version, même disque, Re = 20, 10 mailles par
  diamètre : fixe dans un courant C_D = 1.83 (frottement 0.57, pression 1.25 ; Dennis & Chang : 2.05), mais tiré dans le fluide au
  repos C_D = 4.3, et l'écart croissait quand on raffinait (6.7 à 20 mailles) et quand le pas diminuait. Cause : le fluide collé au
  corps comptait deux fois (§ 6.2). Après correction :
  - disque tiré à Re = 20 : C_D = 2.68 à 10 mailles par diamètre (frottement 0.80, pression 1.88), 1.61 à 20 mailles (frottement
    0.35, pression 1.26) — l'écart décroît maintenant avec la maille, la pression rejoint Dennis & Chang (≈ 1.24) ;
  - disque tiré, domaine de 45 × 15 diamètres : C_D = 2.47 à Re = 20 et 2.36 à Re = 40 (fixe : 1.83 et 1.38) ;
  - disque qui sédimente dans l'huile (Re visé 20) : 0.109 m/s pour 0.142 (−23 %, Re 15), au lieu de 0.065.
  Restent : le frottement, trop faible à 20 mailles (non-glissement étalé sur une maille), et un bruit de pression sur les corps en
  mouvement dans l'eau (force d'un disque qui suit le courant qui varie de plusieurs ½ρU²d d'un pas à l'autre : le fluide balayé va
  à une seule voisine, alors que le fluide collé quitte la maille par deux faces sur un bord oblique). Répartir le volume balayé
  entre les voisines, au prorata du débit collé qui sort par chaque face, améliorait la traînée (2.36) mais cassait un flotteur à la
  ligne d'eau (mailles mêlées d'eau et d'air) : écarté. Diagnostic : `co_moving_disc_diag` (disque qui suit le courant).
- Pas encore : contacts (étape 4), chaleur des corps (étape 5), grains, GPU. Au démarrage, quand le pas double d'un pas à l'autre,
  la masse ajoutée apparente monte jusqu'à 1.4 (la correction de fermeture était prévue pour l'ancien pas).

## 9. Choix (validés le 2026-11-24)

- Corps rigides seulement, en 2D, formes convexes (disque, rectangle, polygone).
- Couplage par domaine fictif (phase rigide dans le mélange), avec repli monolithique si les corps légers posent problème.
- Ordre : corps fixes, puis mouvement imposé, puis mouvement libre, puis contacts ; une température par corps au début.
- Pas de corps de forme quelconque au début : plus tard, par décomposition en polygones convexes.

## Références

- P. Angot, C.-H. Bruneau, P. Fabrie (1999), A penalization method to take into account obstacles in incompressible viscous flows,
  Numer. Math. 81, 497–520.
- S. V. Apte, M. Martin, N. A. Patankar (2009), A numerical method for fully resolved simulation (FRS) of rigid particle–flow
  interactions in complex flows, J. Comput. Phys. 228, 2712–2738.
- C. Batty, F. Bertails, R. Bridson (2007), A fast variational framework for accurate solid-fluid coupling, ACM Trans. Graph. 26(3)
  (SIGGRAPH) — [PDF](https://cs.uwaterloo.ca/~c2batty/papers/Batty07.pdf).
- W.-P. Breugem (2012), A second-order accurate immersed boundary method for fully resolved simulations of particle-laden flows,
  J. Comput. Phys. 231, 4469–4498.
- M. Carlson, P. J. Mucha, G. Turk (2004), Rigid fluid: animating the interplay between rigid bodies and fluid, ACM Trans. Graph.
  23(3), 377–384 — [PDF](https://faculty.cc.gatech.edu/~turk/my_papers/rigid_fluid.pdf).
- P. Causin, J.-F. Gerbeau, F. Nobile (2005), Added-mass effect in the design of partitioned algorithms for fluid–structure
  problems, Comput. Methods Appl. Mech. Engrg. 194, 4506–4527.
- S. C. R. Dennis, G.-Z. Chang (1970), Numerical solutions for steady flow past a circular cylinder at Reynolds numbers up to 100,
  J. Fluid Mech. 42, 471–489 (C_D = 1.522 à Re = 40, ≈ 2.05 à Re = 20).
- E. A. Fadlun, R. Verzicco, P. Orlandi, J. Mohd-Yusof (2000), Combined immersed-boundary finite-difference methods for
  three-dimensional complex flow simulations, J. Comput. Phys. 161, 35–60.
- Flow Science, A fixed mesh method for general moving objects in fluid flow (note technique TN73, modèle GMO de FLOW-3D) —
  [PDF](https://flow3d.co.kr/wp-content/uploads/A-Fixed-Mesh-Method-for-General-Moving-Objects-in-Fluid-Flow.pdf).
- F. Gibou, C. Min (2012), Efficient symmetric positive definite second-order accurate monolithic solver for fluid/solid
  interactions, J. Comput. Phys. 231, 3246–3263 — [PDF](https://me.ucsb.edu/sites/me.ucsb.edu/files/publications/Gibou/FG_Pub38.pdf).
- R. Glowinski, T.-W. Pan, T. I. Hesla, D. D. Joseph (1999), A distributed Lagrange multiplier/fictitious domain method for
  particulate flows, Int. J. Multiphase Flow 25, 755–794 —
  [PDF](https://dept.aem.umn.edu/people/faculty/joseph/archive/docs/IJMF-1999.25-DLM-FDM.pdf).
- R. Glowinski, T.-W. Pan, T. I. Hesla, D. D. Joseph, J. Périaux (2001), A fictitious domain approach to the direct numerical
  simulation of incompressible viscous flow past moving rigid bodies: application to particulate flow, J. Comput. Phys. 169, 363–426.
- P. Gondret, M. Lance, L. Petit (2002), Bouncing motion of spherical particles in fluids, Phys. Fluids 14, 643–652.
- C. W. Hirt, J. M. Sicilian (1985), A porosity technique for the definition of obstacles in rectangular cell meshes, 4th Int. Conf.
  on Numerical Ship Hydrodynamics, Washington (méthode FAVOR).
- T. Kajishima, S. Takiguchi, H. Hamasaki, Y. Miyake (2001), Turbulence structure of particle-laden flow in a vertical plane channel
  due to vortex shedding, JSME Int. J. Ser. B 44, 526–535.
- T. Kempe, J. Fröhlich (2012), An improved immersed boundary method with direct forcing for the simulation of particle laden flows,
  J. Comput. Phys. 231, 3663–3684.
- A. J. C. Ladd (1994), Numerical simulations of particulate suspensions via a discretized Boltzmann equation. Part 1, J. Fluid Mech.
  271, 285–309.
- J. Lee, J. Kim, H. Choi, K.-S. Yang (2011), Sources of spurious force oscillations from an immersed boundary method for moving-body
  problems, J. Comput. Phys. 230, 2677–2695.
- I. Martínez-Estévez et al., Coupling of an SPH-based solver with a multiphysics library (DualSPHysics + Project Chrono) —
  [arXiv:2104.00537](https://arxiv.org/abs/2104.00537).
- R. Mittal, G. Iaccarino (2005), Immersed boundary methods, Annu. Rev. Fluid Mech. 37, 239–261.
- OpenFOAM, contraintes des corps à 6 DDL —
  [sixDoFRigidBodyMotionConstraints](https://cpp.openfoam.org/v7/namespaceFoam_1_1sixDoFRigidBodyMotionConstraints.html).
- N. A. Patankar, P. Singh, D. D. Joseph, R. Glowinski, T.-W. Pan (2000), A new formulation of the distributed Lagrange
  multiplier/fictitious domain method for particulate flows, Int. J. Multiphase Flow 26, 1509–1524.
- C. S. Peskin (2002), The immersed boundary method, Acta Numerica 11, 479–517.
- Rapier (moteur physique en Rust), blocage des translations et rotations —
  [guide](https://rapier.rs/docs/user_guides/rust/rigid_bodies) ; Box2D : [box2d.org](https://box2d.org).
- A. Robinson-Mosher, T. Shinar, R. Fedkiw (2008), Two-way coupling of fluids to rigid and deformable solids and shells, ACM Trans.
  Graph. 27(3) (SIGGRAPH).
- J. H. Seo, R. Mittal (2011), A sharp-interface immersed boundary method with improved mass conservation and reduced spurious
  pressure oscillations, J. Comput. Phys. 230, 7347–7363 —
  [PDF](https://engineering.jhu.edu/fsag/wp-content/uploads/2014/06/JCP_Seo_Mittal_2011_2.pdf).
- N. Sharma, N. A. Patankar (2005), A fast computation technique for the direct numerical simulation of rigid particulate flows,
  J. Comput. Phys. 205, 439–457.
- D. J. Tritton (1959), Experiments on the flow past a circular cylinder at low Reynolds numbers, J. Fluid Mech. 6, 547–567.
- H. S. Udaykumar, R. Mittal, P. Rampunggoon, A. Khanna (2001), A sharp interface Cartesian grid method for simulating flows with
  complex moving boundaries, J. Comput. Phys. 174, 345–380.
- M. Uhlmann (2005), An immersed boundary method with direct forcing for the simulation of particulate flows, J. Comput. Phys. 209,
  448–476.
- D. Wan, S. Turek (2006), Direct numerical simulation of particulate flow via multigrid FEM techniques and the fictitious boundary
  method, Int. J. Numer. Meth. Fluids 51, 531–566.
