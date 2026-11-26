# Lois physiques, mathématiques et chimiques de la simulation

Ce document recense les lois qu'emploie le solveur MULTIPHASIQUE (`src/simulation/multi/`, celui de l'application) et dit
brièvement à quoi chacune sert. Le CPU calcule en f64 et le GPU (`mp_gpu/*.wgsl`) applique les mêmes formules en f32. Le solveur
monophasique (`src/simulation/mono/`) n'est plus suivi : il n'est pas décrit ici.

Pour aller plus loin : MULTIPHASE.md (solveur), GRANULAIRE.md (grains), SOLID.md (corps rigides), EXPERIENCES.md (protocoles et
mesures) ; la simulation comparée aux références : MESURES.md.

Notations : m_k masse partielle de la phase k (kg par m³ de maille), α_k fraction volumique, ρ masse volumique, p pression,
T température, u = (u, v) vitesse, e énergie interne (J/m³), g pesanteur, dx taille de maille, dt pas de temps.
Constantes : R = 8.314 J/(mol·K) ; σ_SB = 5.670374e-8 W/(m²·K⁴) ; g = 9.81 m/s² par défaut ; P° = 1 atm.

---

## 1. Équations de conservation (le cadre)

| Loi | Forme | À quoi elle sert |
|---|---|---|
| Masse de chaque phase | ∂m_k/∂t + ∇·(m_k u) = sources (changement d'état, réactions, cellules spéciales) | transport de chaque fluide et des grains ; les masses sont les variables conservées, à l'arrondi près |
| Quantité de mouvement du mélange (Navier–Stokes compressible) | ρ (∂u/∂t + u·∇u) = −∇p + ∇·[μ (∇u + ∇uᵀ)] + ρ g + f_σ + accélérateurs | une seule vitesse par maille pour toutes les phases qui s'y trouvent |
| Énergie interne, par groupe de phases | ∂e_g/∂t + ∇·(e_g u) = ∇·(k ∇T) − p ∇·u + échanges + sources | une température par groupe miscible (air, eau, huile… ont chacun la leur) |
| Fermeture volumique | Σ_k m_k / ρ_k(p, T_k) + α_corps = 1 dans chaque maille | c'est elle qui fixe la pression : le fluide remplit exactement la maille (`mp_pressure.rs`) |
| Fractions volumiques | α_k = (m_k/ρ_k) / Σ_j (m_j/ρ_j) | dérivées des masses, jamais transportées seules |

Chaque maille a une seule pression (équilibre mécanique) et, par groupe de phases, une seule température.

## 2. Matériaux : équations d'état et propriétés (`phases.rs`, `materials.rs`)

| Loi | Forme | À quoi elle sert |
|---|---|---|
| Gaz parfait | ρ = p / (R_s T) ; c_v = c_p − R_s ; c² = γ R_s T | masse volumique et compressibilité des gaz (dilatation, compression d'un piston, explosions) |
| Équation de Tait (liquides) | (p + B)/(p₀ + B) = (ρ/ρ_T(T))ⁿ — eau : B = 3.07e8 Pa, n = 7.15 (c = 1482 m/s) | compressibilité des liquides : coup de bélier, eau confinée chauffée, cavitation |
| Masse volumique de l'eau selon Kell | polynôme de Kell (1975), maximum à 4 °C, valable de −30 à 150 °C | convection naturelle de l'eau, eau surfondue |
| Dilatation à β constant (autres liquides) | ρ_T(T) = ρ₀ / (1 + β (T − T₀)) | convection des huiles, de la cire fondue, de la lave |
| Capacité thermique | c_p(T) = c_p + pente (T − T_réf) ; e(T) = e₀ + ∫ c_v dT | énergie interne ; les décalages e₀ portent chaleurs latentes et enthalpies de formation |
| Viscosité de Sutherland (gaz) | μ = μ_réf (T/T_réf)^{3/2} (T_réf + S)/(T + S) — air : S = 110.4 K | viscosité et conductivité des gaz qui montent avec T |
| Viscosité de Vogel–Fulcher–Tammann (Andrade) | μ = μ_réf 10^{b [1/(T − c) − 1/(T_réf − c)]} — eau : b = 247.8 K, c = 140 K ; lave (Giordano 2008) | viscosité de l'eau et de la lave, qui chutent avec T |
| Viscosité d'Arrhenius | μ = μ_réf exp[E/R (1/T − 1/T_réf)] | huiles, cire fondue, liquides très visqueux |
| Conductivité linéaire | k = k_réf [1 + pente (T − T_réf)] | conductivité de l'eau (+0.35 %/K) |
| Diffusivité moléculaire | gaz : D ∝ T^1.75 / p (Fuller, Chapman–Enskog) ; liquides : D ∝ T / μ(T) (Stokes–Einstein) | mélange des gaz et des liquides miscibles, flammes |
| Compressibilité isentropique d'un mélange | κ_eff = Σ V_i κ_T,i − T (Σ V_i β_i)² / Σ m_i c_p,i | réponse de la pression d'une maille mixte : gaz → 1/(γ p), bulle dans l'eau → quasi isotherme |
| Solides | ρ, c_p, k constants | parois, obstacles, corps rigides |

## 3. Écoulement

| Loi | Forme | À quoi elle sert |
|---|---|---|
| Pesanteur et hydrostatique | ∇p = ρ g au repos ; la pression et les forces utilisent le même ρ aux faces | un liquide au repos reste au repos (courants résiduels 5e-11 m/s) ; poussée d'Archimède |
| Viscosité (tenseur des contraintes) | τ = μ (∇u + ∇uᵀ), μ variable ; explicite sous μ dt < 0.2 ρ dx², implicite au-delà (`mp_friction.rs`) | traînée, couches limites ; le miel ne réduit plus le pas de temps |
| Tension de surface, CSF (Brackbill, Kothe et Zemach 1992) | f_σ = σ κ ∇α, courbure κ par fonctions de hauteur | gouttes, bulles, ménisques ; saut de Laplace Δp = σ/R (vérifié à 0.1 %) |
| Pression faiblement compressible implicite (Helmholtz) | −∇·(dt/ρ_f ∇δp) + (κ_eff/dt) δp = (V − 1)/dt − ∇·(Ũ − uⁿ), uⁿ⁺¹ = Ũ − dt/ρ_f ∇δp | projection qui rend la vitesse compatible avec la fermeture ; l'acoustique implicite ne limite pas dt par la vitesse du son |
| Parois | glissement libre : vitesse normale nulle, pas de frottement tangentiel | bords du domaine et mailles solides |
| Bord ouvert, colonne compressible | p = p_o + ρ_a Φ (1 + ½ ρ′ Φ) + ⅛ ρ_a ρ′ (g dx)², Φ potentiel de pesanteur depuis le haut, ρ′ = dρ/dp | un côté ouvert suit l'hydrostatique de l'air extérieur, sans vent parasite (7.6e-8 m/s après 5 s) |

## 4. Chaleur

| Loi | Forme | À quoi elle sert |
|---|---|---|
| Conduction (Fourier) | q = −k ∇T, k harmonique entre deux mailles, sous-pas explicites stables | conduction dans les fluides, les solides et de l'un à l'autre |
| Travail de pression | −p ∇·u, après la projection | refroidissement d'un gaz qui se détend, chauffage d'un gaz comprimé ; restitue la chaleur latente exacte |
| Échange entre groupes d'une maille mixte | G_g = λ 2 k_g / (α_g dx²) ; T_i = Σ G_g T_g/(1 + a_g) / Σ G_g/(1 + a_g), a_g = dt G_g / C_g, implicite exact | équilibre thermique progressif entre phases non miscibles (huile à 2000 °C au contact de l'eau) |
| Échange grains ↔ fluide (Ranz–Marshall) | aire volumique 6 α_s/d, Nu = 2 : G = 12 α_s k_f / d² | une poudre se met en quelques millisecondes à la température du gaz qui la baigne |
| Rayonnement des gaz chauds (optiquement mince, TNF Workshop) | q = 4 σ_SB Σ_k p_k a_P,k(T) (T⁴ − T_amb⁴), coefficients moyens de Planck de H₂O et CO₂ (RADCAL) | refroidissement des fumées et des gaz brûlés |
| Rayonnement dans un tas de grains (Damköhler, Schotte) | k_rad = 4 σ_SB ε_r d T_m³, ε_r = ε/(2 − ε), T_m³ = (T_a² + T_b²)(T_a + T_b)/4 | des braises chauffent le bois voisin à travers un lit chaud |
| Solides thermiques | isotherme (T imposée) ou chauffant (puissance volumique, W/m³) | plaques chauffantes, parois froides |
| Chauffage d'une zone | T tend vers la cible avec τ = 0.2 s, à la puissance maximale près | pinceau « Chauffer », action `chauffer` des protocoles |
| Chaleur d'un corps rigide (capacité localisée) | une température par corps, capacité M c_p (nombre de Biot faible) ; conductance L k_h / dx avec k_h harmonique du corps et du fluide | une bille chaude qui réchauffe l'eau qui l'entoure |
| Chaleur de frottement des grains | Φ = τ : D, rendue aux grains | énergie dissipée par un tas qui s'écoule |

## 5. Changements d'état (`mp_thermo.rs`)

| Loi | Forme | À quoi elle sert |
|---|---|---|
| Clausius–Clapeyron, chaleur latente de Kirchhoff | L(T) = L_réf + (c_p,v − c_p,l)(T − T_réf) ; p_sat(T) intégrée analytiquement ; h_v − h_l = L(T) exactement | courbe de vapeur saturante (±4 % des tables de 20 à 150 °C), cohérente avec le bilan d'énergie |
| Ébullition et condensation à l'interface (Hardt–Wondra) | la maille d'interface est ancrée à T_sat(p) : dm = C (T − T_sat) / L(T), limité par la masse disponible | ébullition sur une plaque chaude, condensation sur une paroi froide |
| Germes dans le liquide (Schnerr–Sauer) | taux de création de vapeur des bulles sous-maille, plafonné par l'énergie disponible | nucléation dans un liquide surchauffé |
| Cavitation | vapeur créée là où p < p_sat(T), la pression étant relâchée à p_sat | un liquide tiré se vaporise au lieu de supporter une traction |
| Évaporation dans un gaz mélangé (Dalton) | équilibre à la PRESSION PARTIELLE de la vapeur : x_v p = p_sat(T) | évaporation dans l'air humide, condensation dans les fumées |
| Gel et fusion, méthode enthalpique | à T_f, la chaleur latente L_f ramène le liquide (ou le solide) à T_f : il gèle (ou fond) juste assez | glace qui fond, eau qui gèle sur une paroi froide (front de Stefan à 0.2 %) |
| Surfusion et front de dendrites | gel seulement sur un germe ou au-delà de ΔT_n de surfusion ; front v = a (ΔT / 1 K)^b (revue de Libbrecht 2014) | une eau surfondue qui gèle d'un coup en neige fondante (10 % de glace à −8 °C) |

## 6. Chimie et combustion (`mp_species.rs`, `phases.rs`)

| Loi | Forme | À quoi elle sert |
|---|---|---|
| Arrhenius modifiée et action de masse | k(T) = A Tᵇ exp(−E_a/(R T)) ; r = k Π c_i^{o_i} (1 − Q/K_c si réversible), c = m/M | vitesse de toutes les réactions en phase gazeuse |
| Équilibre thermodynamique | K_c = exp(−ΔG°/(R T)) (P°/(R T))^Δn, ΔG° tiré des h_f°, s° et c_p(T) des espèces | réactions réversibles : dissociation de CO₂ et de H₂O dans les flammes |
| Chaleur de réaction | portée par les enthalpies de formation (décalages e₀) : e inchangée par la réaction, T monte d'elle-même | dégagement de chaleur cohérent avec la thermochimie |
| Mécanismes globaux | Westbrook–Dryer (CH₄, CO), Marinov (H₂) | flammes de méthane, d'hydrogène, explosions |
| Diffusion des espèces (Fick, correction de Hirschfelder–Curtiss) | J_k = −ρ D_k ∇Y_k + Y_k Σ_j ρ D_j ∇Y_j (Σ J_k = 0) | mélange des gaz ; chaque espèce a sa diffusivité (H₂ diffuse 3 à 4 fois plus vite que O₂) |
| Flamme épaissie dynamique (TFM, Colin et al. 2000 ; Légier et al. 2000) | diffusivités × F, vitesse de réaction ÷ F dans la flamme : S_L ∝ √(D ω) conservée | une flamme de 0.4 mm résolue sur quelques mailles (vitesse de flamme du méthane : 0.384 m/s) |
| Flammes de diffusion | dans une maille chaude où combustible et comburant arrivent de part et d'autre (indice de Takeno négatif), avancement au moins au rythme du mélange τ = √(2Δ/g) | un jet de méthane brûle à l'air libre sans décrocher |
| Réactions de surface des grains | r = S_v C_ox / (1/k_s + 1/k_d), S_v = 6 α_s/d, k_s Arrhenius, k_d = Sh D_ox/d avec Sh = 2 | combustion du charbon (C + O₂ → CO₂, C + ½ O₂ → CO) et réaction de Boudouard (C + CO₂ → 2 CO), limitées par la cinétique puis par la diffusion de l'oxygène |
| Réactions en volume des grains (pyrolyse) | premier ordre en quantité de grains, Arrhenius, à la température du grain | bois → charbon de bois + gaz ; houille → charbon + méthane |
| Loi de Vieille (poudre noire) | la surface des grains régresse à r = a (p / 1 atm)ⁿ dès l'allumage ; produits à la température adiabatique | déflagration d'un tas de poudre, plus vive sous pression |
| Allumage | noyau de gaz brûlés à 2200 K | allumer un mélange (pinceau « Allumer », action `allumer`) |

## 7. Grains (`mp_granular.rs`, `mp_friction.rs`, `mp_percolation.rs`)

| Loi | Forme | À quoi elle sert |
|---|---|---|
| Vitesse terminale d'un grain (Ferguson et Church 2004) | w_t = R g d² / (18 ν + √(0,75 C₂ R g d³)), R = (ρ_s − ρ_f)/ρ_f, C₂ = 0.4 | sédimentation, du régime de Stokes à celui de Newton ; un grain plus léger remonte |
| Entravement (Richardson–Zaki, exposant de Rowe 1987) | w = w_t (1 − α_s)ⁿ, n = (4.7 + 0.1175 Re^0,75)/(1 + 0.0175 Re^0,75) | un nuage dense tombe moins vite qu'un grain seul |
| Tassement | α_s ≤ α_max (0.6 environ) | un tas se forme au fond |
| Frottement de Coulomb continu (Drucker–Prager régularisé) | μ_g = tan φ · p_s / (2 √(\|D\|² + δ²)), p_s pression de contact | un tas tient sous son angle de talus et s'effondre au-delà |
| Loi d'Ergun (Darcy–Forchheimer) | −∂p/∂n + ρ_f g_n = (μ/K) q + b \|q\| q, K = d² ε³ / (150 (1 − ε)²) (Kozeny–Carman), b = 1.75 ρ_f (1 − ε)/(ε³ d) | un fluide qui traverse un lit de grains ou une grille (tirage d'un feu, infiltration) |
| Pression de contact (Terzaghi), lits à deux pressions | p fluide des pores, σ′ ≥ 0 multiplicateur du squelette incompressible ; σ′ = 0 : le lit se soulève | lit fluidisé : vitesse minimale de fluidisation d'Ergun |

## 8. Corps rigides (`mp_bodies.rs`, `mp_contacts.rs`, `mp_body_heat.rs`)

| Loi | Forme | À quoi elle sert |
|---|---|---|
| Domaine fictif | fraction exacte de chaque maille couverte (découpage de Sutherland–Hodgman), Σ m/ρ + α_corps = 1, faces ouvertes à 1 − α | un corps aux bords lisses dans une grille carrée, sans marches d'escalier |
| Mouvement d'un solide rigide (Newton–Euler 2D) | v(p) = V + ω × (p − G) ; M dV/dt = F, I dω/dt = Γ ; axes bloqués, pivot (Huygens : I_q = I + M \|G − q\|²) | corps libres, clapets, palettes |
| Couplage monolithique pression–corps (Causin, Gerbeau et Nobile 2005) | pression et vitesse du corps résolues ensemble : masse ajoutée implicite | un corps plus léger que l'eau reste stable (bouchon, flotteur) |
| Poussée d'Archimède exacte | p(x) ≈ p_m + ρ_m g·(x − x_m) intégrée le long du bord ; force exacte (linéaire), couple par Simpson | un corps flotte à sa juste hauteur, où qu'il soit dans ses mailles |
| Contact souple (éléments discrets) | F_n = k (δ − δ₀) − c v_n, k = m ω², ω = max(√(g/δ_max), v_n/δ_max) ; frottement de Coulomb \|F_t\| ≤ μ F_n | rebonds et frottements contre les bords, les solides et les autres corps |
| Coefficient de restitution | amortissement c réglé (dichotomie sur le schéma réel) pour que la vitesse de rebond vaille e fois celle d'arrivée | rebond réglable de 0 (choc mou) à 1 |
| Mouvements imposés | vitesse constante + oscillation sinusoïdale intégrée exactement ; ou lois en formules (position, vitesse, accélération, angle, rotation) | piston, batteur à vagues, agitateur, bielle–manivelle |
| Forces en formules | force et couple ajoutés à un corps libre, explicites | ressort F = −k (x − x₀), amortisseur, moteur |
| Bilan des forces d'un corps | M (Vⁿ⁺¹ − Vⁿ)/dt = F_fluide + F_contacts + F_loi + M g, au pas près (amortisseur des contacts à la vitesse obtenue) | mesures `corps n fx \| fy \| couple [contact \| loi \| total]` : ce que subit un corps, poste par poste |

## 9. Bords et cellules spéciales (`mp_cells.rs`)

| Élément | Loi | À quoi il sert |
|---|---|---|
| Faces à vitesse imposée | vitesse normale fixée, condition de Neumann avec flux dans la pression | entrées et sorties de bord, émetteurs, puits, pompes |
| Cellules épinglées | composition remise à une cible à chaque pas, δp = 0 (Dirichlet) | puits (aspirent), émetteurs (créent un fluide), téléporteurs |
| Accélérateur | accélération de volume ajoutée aux faces, comme la pesanteur | ventilateur, agitation |
| Grille | faces fermées au mélange, le fluide passe par percolation (Ergun, trous de taille d) | sole d'un foyer, tamis |
| Vanne | faces fermées sur la part fermée, le fluide reste | vanne à guillotine, réglable de 0 à 100 % |
| Pompe (compresseur) | vitesse imposée sur les faces de la zone : débit volumique fixé | pompe, compresseur (en aval fermé, la pression monte) |

## 10. Méthodes mathématiques et numériques

| Méthode | Forme | À quoi elle sert |
|---|---|---|
| Grille décalée (MAC) | p, T, m_k aux centres ; u sur les faces verticales, v sur les horizontales | couplage pression–vitesse sans oscillations en damier |
| Volumes finis conservatifs | flux échangés face par face | masse et énergie conservées à l'arrondi près |
| Projection (Chorin), implicite en pression | vitesse prédite, puis corrigée par ∇p | incompressibilité approchée et compressibilité ensemble |
| Méthode de Newton | 2 à 4 itérations sur la fermeture V(p) ≈ V(p_k) − κ_eff (p − p_k) | pression d'une maille dont les fluides ont des équations d'état non linéaires |
| Gradient conjugué préconditionné par multigrille (MGPCG, McAdams et al. 2010) | V-cycle : agrégation 2×2, opérateur grossier rediscrétisé, Gauss–Seidel rouge-noir ; ou Cholesky incomplet IC(0) | système de pression symétrique défini positif, en 10 à 30 itérations quelle que soit la grille |
| VOF géométrique PLIC | normale de Youngs, formules analytiques de Scardovelli et Zaleski (2000), balayages directionnels, limiteur de sortie | interface nette entre fluides non miscibles, volumes exacts |
| Quantité de mouvement conservative | ρ u transportée avec les mêmes flux de masse que le VOF, reconstruction minmod | stabilité des grands rapports de densités (eau/air = 1000) |
| Semi-lagrangien MacCormack limité | remontée des caractéristiques, correction d'erreur bornée | transport de la vitesse (option) |
| MUSCL, limiteur de van Leer | température reconstruite aux faces | transport de la chaleur peu diffusif sans oscillations |
| Viscosité implicite | gradient conjugué préconditionné de Jacobi sur l'opérateur symétrique (ρ/dt) − ∇·μ(∇ + ∇ᵀ) | fluides très visqueux et frottement des grains sans réduire le pas |
| Intégration de la chimie | sous-pas adaptatifs : point milieu pour les réactions irréversibles, Euler implicite (Newton) pour les réversibles | chimie raide près de l'équilibre |
| Runge–Kutta d'ordre 4 | intégration sur un pas des lois de vitesse et d'accélération des corps | mouvements imposés en formules |
| Bornes du pas de temps | CFL dt ≤ cfl dx/\|u\| ; pesanteur dt ≤ ½ √(dx/g) ; capillarité dt ≤ √((ρ_min + ρ_max) dx³ / (4π σ)) ; viscosité explicite ; chimie, flammes, grains (½ maille), corps (0.15 maille), contacts (8 pas par période) ; croissance ≤ ×2 | stabilité sans limite acoustique |
| Filet de sécurité | un pas qui donne un état invalide est refait avec dt divisé, jusqu'à dt/1024 | robustesse face aux chocs thermiques violents |
| Interpolation bilinéaire | entre les centres de mailles (vitesses : entre les faces) | sondes « T en (x, y) » des protocoles |
| Comptage des flux | volume et masse de chaque phase passés par les faces suivies, comptés par le transport lui-même | débits exacts à travers une ligne |
| Calcul parallèle déterministe | tranches de lignes, ordre des sommes fixé | même résultat au bit près quel que soit le nombre de fils |
