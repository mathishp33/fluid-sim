//! Cellules spéciales, solides thermiques et entrées de bord du solveur multiphasique (équivalents de `CellEffect`, `ThermalCell` et
//! `BoundaryType::Inlet` du solveur mono-fluide, `fluid.rs`).
//!
//! Principe : « faces à vitesse imposée ». Un puits, un émetteur, un téléporteur ou une entrée imposent la composante normale de
//! la vitesse sur certaines faces (`pf_u`, `pf_v`, NaN = face libre). Ces faces :
//!  * sont ré-écrites dans u, v avant l'advection (les flux de masse du PLIC les utilisent donc),
//!  * ne sont PAS corrigées par la pression (Neumann avec flux : aucun couplage dans la matrice, la divergence imposée entre dans le
//!    second membre, donc le fluide voisin se comprime / se détend de façon cohérente).
//! Les cellules « épinglées » (`pin`) — puits, émetteurs, téléporteurs — sont remises à une composition cible à chaque pas (air ambiant
//! pour un puits ; phase choisie à T pour un émetteur ; mélange absorbé par le puits source pour la destination d'un téléporteur). La masse
//! créée / détruite est inscrite dans `boundary_flux` : le bilan M_k(t) − M_k(0) = flux_k reste exact. Dans la matrice de pression
//! elles sont des conditions de Dirichlet (δp = 0) : elles ne perturbent pas le champ de pression voisin.
//!
//! Accélérateur : accélération de volume ajoutée aux faces (comme la gravité). Solide thermique : `ThermalCell`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::multiphase::{Bc, MultiphaseSim};

/// Effet d'une cellule FLUIDE spéciale. Vitesses en m/s, accélérations en m/s².
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum MpEffect {
    /// aspire le fluide voisin à `rate` m/s sur chaque face ouverte et le fait disparaître
    Sink { rate: f64 },
    /// crée en continu la phase `phase` à `temp` K et l'expulse à (vx, vy) m/s (composantes sortantes de chaque face)
    Emitter { vx: f64, vy: f64, phase: usize, temp: f64 },
    /// accélération de volume
    Accelerator { ax: f64, ay: f64 },
    /// aspire à `speed` m/s ; le fluide absorbé (composition, T) réapparaît autour de `dest` (indice de cellule), sur un disque de
    /// `radius` mailles (0 = la seule cellule `dest`), avec le même débit volumique. Toutes les sources qui visent la même `dest`
    /// forment un groupe : leurs débits s'additionnent et leurs fluides se mélangent (rayon du groupe : le plus grand).
    Teleporter { dest: usize, speed: f64, radius: usize },
    /// grille : arrête les grains (un tas y repose, comme sur un solide) et laisse passer le fluide par percolation (trous de `hole` m,
    /// fraction ouverte `open` ; loi d'Ergun, voir mp_percolation.rs). Ses faces sont fermées à l'écoulement du mélange.
    Grille { hole: f64, open: f64 },
    /// maille FERMÉE d'une vanne : ses faces sont fermées à l'écoulement, son fluide reste (la part ouverte d'une vanne est faite de
    /// mailles ordinaires)
    Valve,
}

impl MpEffect {
    pub const KINDS: [&'static str; 6] = ["Puits", "Émetteur", "Accélérateur", "Téléporteur", "Grille", "Vanne"];
    pub fn kind_name(&self) -> &'static str {
        match self {
            MpEffect::Sink { .. } => "Puits",
            MpEffect::Emitter { .. } => "Émetteur",
            MpEffect::Accelerator { .. } => "Accélérateur",
            MpEffect::Teleporter { .. } => "Téléporteur",
            MpEffect::Grille { .. } => "Grille",
            MpEffect::Valve => "Vanne",
        }
    }
}

/// Comportement thermique d'une cellule SOLIDE.
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub enum MpThermal {
    #[default]
    Passive,
    /// température imposée (K)
    Isothermal(f64),
    /// source volumique de chaleur (W/m³)
    Heater(f64),
}

/// Entrée de bord (équivalent de `BoundaryType::Inlet`) : vitesse (vx, vy) en m/s imposée sur la face normale, fluide `phase` à `temp` K.
/// `offset` : rampe (en cellules) depuis les coins, comme le mono-fluide. La composante tangentielle est ignorée (glissement libre).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MpInlet {
    pub vx: f64,
    pub vy: f64,
    pub phase: usize,
    pub temp: f64,
    pub offset: usize,
}

/// Cellule épinglée : 1 = puits (ou source de téléporteur), 2 = émetteur (ou destination de téléporteur)
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) struct PinInfo {
    pub kind: u8,
    pub phase: usize,
    pub temp: f64,
    /// indice de la destination si cette cellule est la source d'un téléporteur
    pub tp_dest: Option<usize>,
}

/// Facteur de rampe depuis les coins (identique à `cpu.rs::offset_factor`).
pub fn offset_factor(x: usize, size: usize, offset: usize) -> f64 {
    if offset == 0 {
        return 1.0;
    }
    if size <= 2 || offset >= size {
        return 0.0;
    }
    (x.min(size - 1 - x) as f64 / offset as f64).clamp(0.0, 1.0)
}

impl MultiphaseSim {
    // ---------------------------------------------------------------- édition
    /// Pose (ou retire) un effet sur la cellule c. Une cellule solide devient fluide (ambiant) ; un effet pur ne change pas le fluide.
    pub fn set_effect(&mut self, c: usize, effect: Option<MpEffect>) {
        if effect.is_some() && self.is_solid(c) {
            self.solid[c] = -1;
            self.thermal[c] = MpThermal::Passive;
            let t = self.t_ambient;
            self.fill_ambient(c, t);
        }
        self.special[c] = effect;
    }

    /// Fixe le comportement thermique d'une cellule solide (sans effet sur une cellule fluide).
    pub fn set_thermal_cell(&mut self, c: usize, t: MpThermal) {
        if self.is_solid(c) {
            self.thermal[c] = t;
            if let MpThermal::Isothermal(temp) = t {
                let s = &self.reg.solids[self.solid[c] as usize];
                self.e[c] = s.rho_cp() * temp;
                self.temp[c] = temp;
            }
        } else {
            self.thermal[c] = MpThermal::Passive;
        }
    }

    /// Supprime solide, effet spécial et comportement thermique de la cellule (elle redevient du fluide ambiant).
    pub fn clear_cell(&mut self, c: usize) {
        if self.is_solid(c) {
            self.solid[c] = -1;
            let t = self.t_ambient;
            self.fill_ambient(c, t);
        }
        self.special[c] = None;
        self.thermal[c] = MpThermal::Passive;
    }

    /// Remplit la cellule d'une phase pure à (p courant, T) ; les faces ne sont pas modifiées.
    pub fn fill_pure(&mut self, c: usize, k: usize, t: f64) {
        let p = self.p[c];
        for j in 0..self.reg.phases.len() {
            self.mass[j][c] = if j == k { self.reg.phases[j].rho(p, t) } else { 0.0 };
        }
        self.set_cell_temperature(c, t);
    }

    /// Remplit la cellule du fluide ambiant (pur ou mélange, ex. air O2/N2) à (p courant, T).
    pub fn fill_ambient(&mut self, c: usize, t: f64) {
        let a = self.ambient_alpha();
        self.fill_mixture(c, &a, t);
    }

    /// Remplit la cellule d'un mélange (fractions volumiques `alpha`, somme 1) à (p courant, T).
    pub fn fill_mixture(&mut self, c: usize, alpha: &[f64], t: f64) {
        let p = self.p[c];
        for j in 0..self.reg.phases.len() {
            self.mass[j][c] = alpha[j] * self.reg.phases[j].rho(p, t);
        }
        self.set_cell_temperature(c, t);
    }

    /// Un des extras (effet spécial, solide thermique, entrée) est-il actif ? (le GPU ne les gère pas)
    pub fn has_extras(&self) -> bool {
        self.special.iter().any(|s| s.is_some()) || self.thermal.iter().any(|t| *t != MpThermal::Passive) || self.inlet.iter().any(|i| i.is_some())
    }
    pub fn special_count(&self) -> usize {
        self.special.iter().filter(|s| s.is_some()).count()
    }
    #[inline]
    pub(crate) fn pinned(&self, c: usize) -> bool {
        self.pin[c].is_some()
    }
    #[inline]
    pub(crate) fn face_u_fixed(&self, f: usize) -> bool {
        !self.pf_u[f].is_nan()
    }
    #[inline]
    pub(crate) fn face_v_fixed(&self, f: usize) -> bool {
        !self.pf_v[f].is_nan()
    }

    /// destination de téléporteur valide pour la source c ?
    fn tp_dest_ok(&self, c: usize, dest: usize) -> bool {
        dest < self.nx * self.ny && dest != c && !self.is_solid(dest) && self.special[dest].is_none()
    }

    /// cellules de la zone de destination d'un téléporteur : disque de `radius` mailles autour de `dest`, fluides et sans effet
    pub(crate) fn tp_dest_cells(&self, dest: usize, radius: usize) -> Vec<usize> {
        let (nx, ny) = (self.nx, self.ny);
        let (ci, cj, r) = ((dest % nx) as isize, (dest / nx) as isize, radius as isize);
        let mut v = vec![];
        for dj in -r..=r {
            for di in -r..=r {
                let (i, j) = (ci + di, cj + dj);
                if di * di + dj * dj > r * r || i < 0 || j < 0 || i >= nx as isize || j >= ny as isize {
                    continue;
                }
                let d = j as usize * nx + i as usize;
                if !self.is_solid(d) && self.special[d].is_none() {
                    v.push(d);
                }
            }
        }
        v
    }

    // ---------------------------------------------------------------- préparation d'un pas
    /// Reconstruit `pin`, `pf_u`, `pf_v` depuis les effets et les entrées de bord. Appelé au début de chaque pas.
    pub(crate) fn prepare_special(&mut self) {
        let (pin, pf_u, pf_v) = self.special_tables();
        self.pin = pin;
        self.pf_u = pf_u;
        self.pf_v = pf_v;
    }

    /// (`pin`, `pf_u`, `pf_v`) des effets et entrées de bord courants, sans modifier l'état (aussi utilisé par le solveur GPU).
    pub(crate) fn special_tables(&self) -> (Vec<Option<PinInfo>>, Vec<f64>, Vec<f64>) {
        let (nx, ny) = (self.nx, self.ny);
        let n = nx * ny;
        let mut pin: Vec<Option<PinInfo>> = vec![None; n];
        let mut fx = FixedFaces { nx, ny, solid: &self.solid, pf_u: vec![f64::NAN; (nx + 1) * ny], pf_v: vec![f64::NAN; nx * (ny + 1)] };
        if !self.has_extras() {
            return (pin, fx.pf_u, fx.pf_v);
        }
        // 1. cellules épinglées + vitesses de leurs faces (téléporteurs : faces fixées une fois TOUTES les cellules épinglées connues)
        let mut tp_groups: Vec<(usize, usize, Vec<(usize, f64)>)> = vec![]; // (dest, rayon, [(source, vitesse)])
        for c in 0..n {
            if self.is_solid(c) {
                continue;
            }
            let (i, j) = (c % nx, c / nx);
            match self.special[c] {
                Some(MpEffect::Sink { rate }) => {
                    pin[c] = Some(PinInfo { kind: 1, phase: self.ambient_phase, temp: self.t_ambient, tp_dest: None });
                    fx.fix_faces(i, j, -rate.abs());
                }
                Some(MpEffect::Emitter { vx, vy, phase, temp }) => {
                    pin[c] = Some(PinInfo { kind: 2, phase: phase.min(self.reg.phases.len() - 1), temp, tp_dest: None });
                    fx.fix_nozzle(i, j, vx, vy);
                }
                // grille : faces fermées au mélange (le fluide passe par percolation, les grains ne passent pas)
                Some(MpEffect::Grille { .. }) => fx.fix_faces(i, j, 0.0),
                // maille fermée d'une vanne : faces fermées, fluide gardé (ni épinglée, ni vidée)
                Some(MpEffect::Valve) => fx.fix_faces(i, j, 0.0),
                Some(MpEffect::Teleporter { dest, speed, radius }) if self.tp_dest_ok(c, dest) => {
                    pin[c] = Some(PinInfo { kind: 1, phase: self.ambient_phase, temp: self.t_ambient, tp_dest: Some(dest) });
                    match tp_groups.iter_mut().find(|g| g.0 == dest) {
                        Some(g) => {
                            g.1 = g.1.max(radius);
                            g.2.push((c, speed.abs()));
                        }
                        None => tp_groups.push((dest, radius, vec![(c, speed.abs())])),
                    }
                }
                _ => {}
            }
        }
        // zones de destination (cellules encore libres ; une cellule déjà prise par un autre groupe ou un autre effet est sautée)
        let mut tp_zones: Vec<Vec<usize>> = vec![];
        for (dest, radius, _) in &tp_groups {
            let zone: Vec<usize> = self.tp_dest_cells(*dest, *radius).into_iter().filter(|&d| pin[d].is_none()).collect();
            for &d in &zone {
                pin[d] = Some(PinInfo { kind: 2, phase: self.ambient_phase, temp: self.t_ambient, tp_dest: Some(*dest) });
            }
            tp_zones.push(zone);
        }
        // débits : seules les faces donnant sur du fluide NON épinglé comptent (les faces entre deux cellules épinglées sont fermées
        // à l'étape 3) — volume aspiré par le groupe = volume réémis par sa zone
        let free_faces = |c: usize| {
            let (i, j) = ((c % nx) as isize, (c / nx) as isize);
            [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                .iter()
                .filter(|&&(a, b)| a >= 0 && b >= 0 && (a as usize) < nx && (b as usize) < ny)
                .map(|&(a, b)| b as usize * nx + a as usize)
                .filter(|&d| !self.is_solid(d) && pin[d].is_none())
                .count()
        };
        for ((_, _, sources), zone) in tp_groups.iter().zip(&tp_zones) {
            let mut q = 0.0;
            for &(src, speed) in sources {
                fx.fix_faces(src % nx, src / nx, -speed);
                q += speed * free_faces(src) as f64;
            }
            let n_out: usize = zone.iter().map(|&d| free_faces(d)).sum();
            if n_out > 0 {
                for &d in zone {
                    fx.fix_faces(d % nx, d / nx, q / n_out as f64);
                }
            }
        }
        // 2. entrées de bord (uniquement sur un bord fermé)
        for side in 0..4 {
            let Some(inl) = self.inlet[side] else { continue };
            if self.bc[side] != Bc::Wall {
                continue;
            }
            match side {
                0 | 1 => {
                    let i = if side == 0 { 0 } else { nx };
                    for j in 0..ny {
                        let cell = if side == 0 { self.idx(0, j) } else { self.idx(nx - 1, j) };
                        if !self.is_solid(cell) {
                            let f = self.uidx(i, j);
                            fx.pf_u[f] = inl.vx * offset_factor(j, ny, inl.offset);
                        }
                    }
                }
                _ => {
                    let j = if side == 2 { 0 } else { ny };
                    for i in 0..nx {
                        let cell = if side == 2 { self.idx(i, 0) } else { self.idx(i, ny - 1) };
                        if !self.is_solid(cell) {
                            let f = self.vidx(i, j);
                            fx.pf_v[f] = inl.vy * offset_factor(i, nx, inl.offset);
                        }
                    }
                }
            }
        }
        // 3. faces entre deux cellules épinglées : fermées
        for j in 0..ny {
            for i in 1..nx {
                if pin[self.idx(i - 1, j)].is_some() && pin[self.idx(i, j)].is_some() {
                    let f = self.uidx(i, j);
                    fx.pf_u[f] = 0.0;
                }
            }
        }
        for j in 1..ny {
            for i in 0..nx {
                if pin[self.idx(i, j - 1)].is_some() && pin[self.idx(i, j)].is_some() {
                    let f = self.vidx(i, j);
                    fx.pf_v[f] = 0.0;
                }
            }
        }
        (pin, fx.pf_u, fx.pf_v)
    }

    /// nombre de faces de la cellule (i, j) donnant sur du fluide intérieur (ni solide, ni bord)
    fn open_faces(&self, i: usize, j: usize) -> usize {
        let (nx, ny) = (self.nx, self.ny);
        let ok = |a: isize, b: isize| a >= 0 && b >= 0 && (a as usize) < nx && (b as usize) < ny && !self.is_solid(self.idx(a as usize, b as usize));
        let (ii, jj) = (i as isize, j as isize);
        [(ii - 1, jj), (ii + 1, jj), (ii, jj - 1), (ii, jj + 1)].iter().filter(|&&(a, b)| ok(a, b)).count()
    }

    /// Ré-écrit les vitesses imposées dans u, v.
    pub(crate) fn apply_prescribed_faces(&mut self) {
        for (u, p) in self.u.iter_mut().zip(&self.pf_u) {
            if !p.is_nan() {
                *u = *p;
            }
        }
        for (v, p) in self.v.iter_mut().zip(&self.pf_v) {
            if !p.is_nan() {
                *v = *p;
            }
        }
    }

    /// Remet les cellules épinglées à leur composition cible ; la masse créée / détruite est comptée dans `boundary_flux`.
    pub(crate) fn pin_reset(&mut self) {
        let np = self.reg.phases.len();
        let a2 = self.dx * self.dx;
        for c in 0..self.nx * self.ny {
            let Some(pin) = self.pin[c] else { continue };
            let before: Vec<f64> = (0..np).map(|k| self.mass[k][c]).collect();
            match (pin.kind, pin.tp_dest.and_then(|k| self.tp_mix.get(&k))) {
                (2, Some((alpha, t))) => {
                    let (alpha, t) = (alpha.clone(), *t);
                    self.fill_mixture(c, &alpha, t);
                }
                _ if pin.phase == self.ambient_phase => self.fill_ambient(c, pin.temp),
                _ => self.fill_pure(c, pin.phase, pin.temp),
            }
            for k in 0..np {
                self.boundary_flux[k] += (self.mass[k][c] - before[k]) * a2;
            }
        }
    }

    /// Après l'advection : enregistre la composition absorbée par chaque GROUPE de téléporteurs (toutes les sources d'une même
    /// destination : masses et énergies additionnées, puis composition et température du mélange), réémise dans sa zone.
    pub(crate) fn absorb_teleporters(&mut self) {
        let np = self.reg.phases.len();
        let amb = self.ambient_alpha();
        let mut acc: Vec<(usize, Vec<f64>, f64, f64)> = vec![]; // (dest, Σ masses absorbées, Σ énergie absorbée, p moyenne pondérée)
        for c in 0..self.nx * self.ny {
            let Some(PinInfo { kind: 1, tp_dest: Some(dest), .. }) = self.pin[c] else { continue };
            let (p, ta) = (self.p[c], self.t_ambient);
            // contenu de remplissage (fluide ambiant) à soustraire
            let base: Vec<f64> = (0..np).map(|k| amb[k] * self.reg.phases[k].rho(p, ta)).collect();
            let e_base: f64 = (0..np).map(|k| base[k] * self.reg.phases[k].e_int(ta)).sum();
            let ma: Vec<f64> = (0..np).map(|k| (self.mass[k][c] - base[k]).max(0.0)).collect();
            let tot: f64 = ma.iter().sum();
            if tot < 1e-9 {
                continue;
            }
            let e_abs = self.e[c] - e_base;
            let g = match acc.iter().position(|g| g.0 == dest) {
                Some(g) => g,
                None => {
                    acc.push((dest, vec![0.0; np], 0.0, 0.0));
                    acc.len() - 1
                }
            };
            for k in 0..np {
                acc[g].1[k] += ma[k];
            }
            acc[g].2 += e_abs;
            acc[g].3 += p * tot;
        }
        for (dest, ma, e_abs, pw) in acc {
            let tot: f64 = ma.iter().sum();
            let p = pw / tot;
            let t = super::multiphase::temp_from_masses(&self.reg.phases, ma.iter().copied(), e_abs).clamp(100.0, 3000.0);
            let vols: Vec<f64> = (0..np).map(|k| ma[k] / self.reg.phases[k].rho(p, t)).collect();
            let vsum: f64 = vols.iter().sum();
            if vsum > 0.0 {
                self.tp_mix.insert(dest, (vols.iter().map(|v| v / vsum).collect::<Vec<f64>>(), t));
            }
        }
    }

    /// (fractions volumiques, T) du fluide qui ENTRE par la face de bord (dir, fi, fj) : celui de l'entrée du côté concerné, sinon l'ambiant.
    pub(crate) fn ghost_source(&self, dir: usize, fi: usize, fj: usize) -> (Vec<f64>, f64) {
        let side = if dir == 0 {
            if fi == 0 { 0 } else { 1 }
        } else if fj == 0 {
            2
        } else {
            3
        };
        match self.inlet[side] {
            Some(inl) if self.bc[side] == Bc::Wall => {
                let k = inl.phase.min(self.reg.phases.len() - 1);
                if k == self.ambient_phase {
                    (self.ambient_alpha(), inl.temp)
                } else {
                    ((0..self.reg.phases.len()).map(|j| if j == k { 1.0 } else { 0.0 }).collect(), inl.temp)
                }
            }
            _ => (self.ambient_alpha(), self.t_ambient),
        }
    }

    // ---------------------------------------------------------------- thermique des solides
    /// Chauffages (e += P·dt) et solides isothermes (T imposée), avant la conduction.
    pub(crate) fn apply_thermal_sources(&mut self, dt: f64) {
        // chauffage au pinceau : apport progressif pendant le temps simulé (un dépôt instantané de chaleur dans un liquide est un chauffage
        // à volume constant : ~1 MPa/K dans l'eau, l'équivalent d'une explosion)
        self.relax_heat_targets(dt);
        for c in 0..self.nx * self.ny {
            if !self.is_solid(c) {
                continue;
            }
            match self.thermal[c] {
                MpThermal::Heater(w) => {
                    self.e[c] += w * dt;
                    self.energy_input += w * dt * self.dx * self.dx;
                }
                MpThermal::Isothermal(t) => {
                    let s = &self.reg.solids[self.solid[c] as usize];
                    self.e[c] = s.rho_cp() * t;
                    self.temp[c] = t;
                }
                MpThermal::Passive => {}
            }
        }
    }

    // ---------------------------------------------------------------- accélérateurs
    /// Ajoute dt × a aux faces adjacentes à un accélérateur (moyenne des accélérateurs adjacents, pas des cellules voisines).
    pub(crate) fn add_accelerators(&self, dt: f64, us: &mut [f64], vs: &mut [f64]) {
        let (nx, ny) = (self.nx, self.ny);
        if !self.special.iter().any(|s| matches!(s, Some(MpEffect::Accelerator { .. }))) {
            return;
        }
        let acc = |c: usize| -> Option<(f64, f64)> {
            match self.special[c] {
                Some(MpEffect::Accelerator { ax, ay }) if !self.is_solid(c) => Some((ax, ay)),
                _ => None,
            }
        };
        for j in 0..ny {
            for i in 1..nx {
                let (mut sum, mut cnt) = (0.0, 0.0);
                for c in [self.idx(i - 1, j), self.idx(i, j)] {
                    if let Some((ax, _)) = acc(c) {
                        sum += ax;
                        cnt += 1.0;
                    }
                }
                if cnt > 0.0 && !self.is_solid(self.idx(i - 1, j)) && !self.is_solid(self.idx(i, j)) {
                    us[j * (nx + 1) + i] += dt * sum / cnt;
                }
            }
        }
        for j in 1..ny {
            for i in 0..nx {
                let (mut sum, mut cnt) = (0.0, 0.0);
                for c in [self.idx(i, j - 1), self.idx(i, j)] {
                    if let Some((_, ay)) = acc(c) {
                        sum += ay;
                        cnt += 1.0;
                    }
                }
                if cnt > 0.0 && !self.is_solid(self.idx(i, j - 1)) && !self.is_solid(self.idx(i, j)) {
                    vs[j * nx + i] += dt * sum / cnt;
                }
            }
        }
    }
}

pub(crate) type TpMix = HashMap<usize, (Vec<f64>, f64)>;

/// Faces à vitesse imposée en construction (voir `special_tables`).
struct FixedFaces<'a> {
    nx: usize,
    ny: usize,
    solid: &'a [i32],
    pf_u: Vec<f64>,
    pf_v: Vec<f64>,
}

impl FixedFaces<'_> {
    fn free(&self, a: isize, b: isize) -> bool {
        a >= 0 && b >= 0 && (a as usize) < self.nx && (b as usize) < self.ny && self.solid[b as usize * self.nx + a as usize] < 0
    }
    /// vitesse normale `s` ALGÉBRIQUE VERS L'EXTÉRIEUR de la cellule (s < 0 : aspiration) sur ses faces ouvertes
    fn fix_faces(&mut self, i: usize, j: usize, s: f64) {
        let (nx, ii, jj) = (self.nx, i as isize, j as isize);
        if self.free(ii - 1, jj) {
            self.pf_u[j * (nx + 1) + i] = -s;
        }
        if self.free(ii + 1, jj) {
            self.pf_u[j * (nx + 1) + i + 1] = s;
        }
        if self.free(ii, jj - 1) {
            self.pf_v[j * nx + i] = -s;
        }
        if self.free(ii, jj + 1) {
            self.pf_v[(j + 1) * nx + i] = s;
        }
    }
    /// Émetteur : buse. La composante sortante (vx, vy) est imposée sur les faces concernées et TOUTES les autres faces ouvertes sont
    /// fermées (vitesse nulle) : le débit émis est exactement ρ·|v|·dx, sans fuite latérale ni aspiration par l'amont. Un émetteur de
    /// vitesse nulle laisse ses faces libres (réservoir à la pression du voisinage).
    fn fix_nozzle(&mut self, i: usize, j: usize, vx: f64, vy: f64) {
        if vx == 0.0 && vy == 0.0 {
            return;
        }
        let (nx, ii, jj) = (self.nx, i as isize, j as isize);
        if self.free(ii - 1, jj) {
            self.pf_u[j * (nx + 1) + i] = vx.min(0.0);
        }
        if self.free(ii + 1, jj) {
            self.pf_u[j * (nx + 1) + i + 1] = vx.max(0.0);
        }
        if self.free(ii, jj - 1) {
            self.pf_v[j * nx + i] = vy.min(0.0);
        }
        if self.free(ii, jj + 1) {
            self.pf_v[(j + 1) * nx + i] = vy.max(0.0);
        }
    }
}
