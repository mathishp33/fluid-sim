//! Couche « application » du solveur multiphasique, SANS dépendance egui (testable en `cargo test`) :
//! scénarios prédéfinis, outils de pinceau (ajout de phase / solide / chauffage / poussée), rendu RGBA, statistiques, infos de cellule.
//! La fenêtre egui (`window/multiphase_window.rs`) n'est qu'une fine couche par-dessus.
//!
//! Registre commun à tous les scénarios : phases 0 = air, 1 = eau, 2 = vapeur d'eau, 3 = huile ; solides 0 = acier, 1 = verre.

use super::multiphase::{Bc, MultiphaseSim};
use super::phases::{Phase, PhaseKind, Registry, Solid};

pub const AIR: usize = 0;
pub const EAU: usize = 1;
pub const VAPEUR: usize = 2;
pub const HUILE: usize = 3;
pub const ACIER: usize = 0;
pub const VERRE: usize = 1;
const P_ATM: f64 = 101325.0;

pub fn registry() -> Registry {
    let (mut r, air, eau, vap) = Registry::water_air_steam();
    debug_assert_eq!((air, eau, vap), (AIR, EAU, VAPEUR));
    let huile = r.add_phase(Phase::oil());
    debug_assert_eq!(huile, HUILE);
    r.add_solid(Solid::steel());
    r.add_solid(Solid::glass());
    r
}

// ------------------------------------------------------------------------------------------------ scénarios
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MpScenario {
    DamBreak,
    DropSplash,
    Sloshing,
    Boiling,
    OilWaterAir,
    Empty,
}

impl Default for MpScenario {
    fn default() -> Self {
        MpScenario::DamBreak
    }
}

impl MpScenario {
    pub const ALL: [MpScenario; 6] = [Self::DamBreak, Self::DropSplash, Self::Sloshing, Self::Boiling, Self::OilWaterAir, Self::Empty];
    pub fn from_index(i: usize) -> Self {
        Self::ALL[i.min(Self::ALL.len() - 1)]
    }
    pub fn index(self) -> usize {
        Self::ALL.iter().position(|&m| m == self).unwrap()
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::DamBreak => "Rupture de barrage",
            Self::DropSplash => "Goutte et éclaboussure",
            Self::Sloshing => "Ballottement (vagues)",
            Self::Boiling => "Ébullition sur plaque chauffante",
            Self::OilWaterAir => "Huile / eau / air (goutte d'huile qui monte)",
            Self::Empty => "Cuve vide (à peindre)",
        }
    }
}

/// Construit un scénario sur une grille nx×ny de maille dx. Les formes sont exprimées en fractions du domaine (valable à toute résolution).
pub fn build_scenario(sc: MpScenario, nx: usize, ny: usize, dx: f64, gravity: f64) -> MultiphaseSim {
    let (lx, ly) = (nx as f64 * dx, ny as f64 * dx);
    let mut s = MultiphaseSim::new(nx, ny, dx, registry(), AIR);
    s.gravity = [0.0, -gravity];
    let mut t_fn: Box<dyn Fn(f64, f64) -> f64> = Box::new(|_, _| 293.15);
    match sc {
        MpScenario::DamBreak => {
            s.paint_phase(move |x, y| if x < 0.25 * lx && y < 0.5 * ly { Some(EAU) } else { Some(AIR) });
        }
        MpScenario::DropSplash => {
            let (cx, cy, r) = (0.5 * lx, 0.62 * ly, 0.09 * lx.min(ly));
            s.paint_phase(move |x, y| {
                if y < 0.22 * ly || (x - cx).powi(2) + (y - cy).powi(2) < r * r { Some(EAU) } else { Some(AIR) }
            });
        }
        MpScenario::Sloshing => {
            s.paint_phase(move |x, y| if y < 0.4 * ly + 0.04 * ly * (std::f64::consts::PI * x / lx).cos() { Some(EAU) } else { Some(AIR) });
        }
        MpScenario::Boiling => {
            // plaque d'acier chaude (2 cellules) sous 40 % d'eau à 90 °C ; haut ouvert (atmosphère)
            let h_plate = 2.0 * dx;
            s.paint_phase(move |_, y| if y < 0.45 * ly { Some(EAU) } else { Some(AIR) });
            s.paint_solid(move |x, y| if y < h_plate && x > 0.1 * lx && x < 0.9 * lx { Some(ACIER) } else { None });
            s.bc[3] = Bc::Open { p: P_ATM };
            s.dt_max = 5e-4;
            t_fn = Box::new(move |x, y| if y < h_plate && x > 0.1 * lx && x < 0.9 * lx { 600.0 } else if y < 0.45 * ly { 363.15 } else { 293.15 });
        }
        MpScenario::OilWaterAir => {
            let (cx, cy, r) = (0.5 * lx, 0.15 * ly, 0.07 * lx.min(ly));
            s.paint_phase(move |x, y| {
                if (x - cx).powi(2) + (y - cy).powi(2) < r * r { Some(HUILE) } else if y < 0.4 * ly { Some(EAU) } else { Some(AIR) }
            });
        }
        MpScenario::Empty => {
            s.bc[3] = Bc::Open { p: P_ATM };
            s.paint_phase(|_, _| Some(AIR));
        }
    }
    s.finalize_init(&*t_fn);
    s
}

// ------------------------------------------------------------------------------------------------ pinceaux
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MpTool {
    Phase(usize),
    Solid(usize),
    EraseSolid,
    Heat,
    Push,
}

pub fn tool_name(reg: &Registry, t: MpTool) -> String {
    match t {
        MpTool::Phase(k) => format!("Ajouter : {}", reg.phases[k].name),
        MpTool::Solid(s) => format!("Solide : {}", reg.solids[s].name),
        MpTool::EraseSolid => "Effacer le solide".into(),
        MpTool::Heat => "Chauffer / refroidir".into(),
        MpTool::Push => "Pousser le fluide".into(),
    }
}

pub fn all_tools(reg: &Registry) -> Vec<MpTool> {
    let mut v: Vec<MpTool> = (0..reg.phases.len()).map(MpTool::Phase).collect();
    v.extend((0..reg.solids.len()).map(MpTool::Solid));
    v.push(MpTool::EraseSolid);
    v.push(MpTool::Heat);
    v.push(MpTool::Push);
    v
}

/// Remplit la cellule c d'une seule phase pure à (p, T) en gardant la cohérence masse/énergie. Les faces voisines ne sont pas modifiées.
fn set_pure(sim: &mut MultiphaseSim, c: usize, k: usize, t: f64) {
    let p = sim.p[c];
    sim.solid[c] = -1;
    let mut e = 0.0;
    for j in 0..sim.reg.phases.len() {
        let m = if j == k { sim.reg.phases[j].rho(p, t) } else { 0.0 };
        sim.mass[j][c] = m;
        e += m * sim.reg.phases[j].e_int(t);
    }
    sim.e[c] = e;
    sim.temp[c] = t;
}

/// Applique l'outil sur les cellules `cells` (i, j) ; `temp` = température de pinceau (K) ; `cursor` = vitesse du curseur (m/s, y vers le haut) ;
/// `dt` = durée de la trame (s) pour les outils à taux (chauffage, poussée).
pub fn apply_tool(sim: &mut MultiphaseSim, tool: MpTool, cells: &[(usize, usize)], temp: f64, cursor: (f64, f64), dt: f64) {
    let (nx, ny) = (sim.nx, sim.ny);
    for &(i, j) in cells {
        if i >= nx || j >= ny {
            continue;
        }
        let c = sim.idx(i, j);
        match tool {
            MpTool::Phase(k) => {
                if !sim.is_solid(c) {
                    set_pure(sim, c, k, temp);
                }
            }
            MpTool::Solid(s) => {
                sim.solid[c] = s as i32;
                for k in 0..sim.reg.phases.len() {
                    sim.mass[k][c] = 0.0;
                    sim.alpha[k][c] = 0.0;
                }
                sim.e[c] = sim.reg.solids[s].rho_cp() * temp;
                sim.temp[c] = temp;
                let (a, b, cc, d) = (sim.uidx(i, j), sim.uidx(i + 1, j), sim.vidx(i, j), sim.vidx(i, j + 1));
                sim.u[a] = 0.0;
                sim.u[b] = 0.0;
                sim.v[cc] = 0.0;
                sim.v[d] = 0.0;
            }
            MpTool::EraseSolid => {
                if sim.is_solid(c) {
                    let t = sim.t_ambient;
                    set_pure(sim, c, sim.ambient_phase, t);
                }
            }
            MpTool::Heat => {
                let rate = 1.0 - (-5.0 * dt).exp();
                let cap = sim.heat_capacity(c);
                let t = sim.temp[c];
                sim.e[c] += (temp - t) * cap * rate;
            }
            MpTool::Push => {
                let rate = 1.0 - (-20.0 * dt).exp();
                let (a, b, cc, d) = (sim.uidx(i, j), sim.uidx(i + 1, j), sim.vidx(i, j), sim.vidx(i, j + 1));
                let open_u = |s: &MultiphaseSim, i2: usize, j2: usize| !s.is_solid(s.idx(i2.min(nx - 1), j2));
                if i > 0 && open_u(sim, i - 1, j) && !sim.is_solid(c) {
                    sim.u[a] += (cursor.0 - sim.u[a]) * rate;
                }
                if i + 1 < nx && !sim.is_solid(sim.idx(i + 1, j)) && !sim.is_solid(c) {
                    sim.u[b] += (cursor.0 - sim.u[b]) * rate;
                }
                if j > 0 && !sim.is_solid(sim.idx(i, j - 1)) && !sim.is_solid(c) {
                    sim.v[cc] += (cursor.1 - sim.v[cc]) * rate;
                }
                if j + 1 < ny && !sim.is_solid(sim.idx(i, j + 1)) && !sim.is_solid(c) {
                    sim.v[d] += (cursor.1 - sim.v[d]) * rate;
                }
            }
        }
    }
    sim.refresh();
}

/// Cellules d'un pinceau (rond ou carré) de rayon `r` (en cellules) centré en (ci, cj), bornées à la grille.
pub fn brush_cells(nx: usize, ny: usize, ci: usize, cj: usize, r: usize, circle: bool) -> Vec<(usize, usize)> {
    let mut v = vec![];
    let r = r as isize;
    for dj in -r..=r {
        for di in -r..=r {
            if circle && di * di + dj * dj > r * r {
                continue;
            }
            let (i, j) = (ci as isize + di, cj as isize + dj);
            if i >= 0 && j >= 0 && (i as usize) < nx && (j as usize) < ny {
                v.push((i as usize, j as usize));
            }
        }
    }
    v
}

// ------------------------------------------------------------------------------------------------ rendu
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MpDisplay {
    Phases,
    Pressure,
    Temperature,
    Velocity,
    Density,
    Vapor,
}

impl MpDisplay {
    pub const ALL: [MpDisplay; 6] = [Self::Phases, Self::Pressure, Self::Temperature, Self::Velocity, Self::Density, Self::Vapor];
    pub fn name(self) -> &'static str {
        match self {
            Self::Phases => "PHASES",
            Self::Pressure => "PRESSION (relative à 1 atm)",
            Self::Temperature => "TEMPÉRATURE",
            Self::Velocity => "VITESSE",
            Self::Density => "MASSE VOLUMIQUE",
            Self::Vapor => "VAPEUR (fraction)",
        }
    }
    pub fn next(self) -> Self {
        let i = Self::ALL.iter().position(|&m| m == self).unwrap();
        Self::ALL[(i + 1) % Self::ALL.len()]
    }
}

#[derive(Clone, Copy)]
pub struct RenderOpts {
    pub display: MpDisplay,
    /// échelle d'affichage de T − T_ambiant (K)
    pub t_scale: f64,
    /// échelle de pression (Pa), 0 = automatique (max |p − 1 atm|)
    pub p_scale: f64,
}

fn phase_rgb(ph: &Phase, k: usize) -> [f32; 3] {
    match ph.name.as_str() {
        "air" => [0.05, 0.07, 0.11],
        "eau" => [0.10, 0.42, 0.88],
        "vapeur" => [0.90, 0.93, 0.96],
        "huile" => [0.85, 0.62, 0.10],
        _ => {
            let h = (k as f32 * 67.0) % 360.0;
            let c = hsv(h, 0.7, if ph.kind == PhaseKind::Gas { 0.5 } else { 0.9 });
            [c[0], c[1], c[2]]
        }
    }
}

fn solid_rgb(name: &str) -> [f32; 3] {
    match name {
        "acier" => [0.55, 0.57, 0.60],
        "verre" => [0.62, 0.82, 0.80],
        _ => [0.6, 0.6, 0.6],
    }
}

fn hsv(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = (h % 360.0 + 360.0) % 360.0;
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r + m, g + m, b + m]
}

/// bleu/cyan (froid) → noir → rouge → jaune (chaud)
fn temp_rgb(x: f64) -> [f32; 3] {
    let x = x.clamp(-1.0, 1.0) as f32;
    if x >= 0.0 {
        [(2.0 * x).min(1.0), (2.0 * x - 1.0).max(0.0), 0.0]
    } else {
        let c = -x;
        [0.0, (2.0 * c - 1.0).max(0.0), (2.0 * c).min(1.0)]
    }
}

fn signed_rgb(v: f64, max_abs: f64) -> [f32; 3] {
    if max_abs <= 1e-12 {
        return [0.0; 3];
    }
    let x = (v / max_abs).clamp(-1.0, 1.0) as f32;
    if x < 0.0 { [0.0, -x, -x] } else { [x, 0.0, 0.0] }
}

/// Rend la simulation en RGBA (nx×ny, ligne 0 = HAUT du domaine : l'axe y est inversé par rapport aux indices j).
pub fn render_rgba(sim: &MultiphaseSim, o: RenderOpts, out: &mut Vec<u8>) {
    let (nx, ny) = (sim.nx, sim.ny);
    out.resize(nx * ny * 4, 255);
    let vapor_ids: Vec<usize> = sim.reg.changes.iter().map(|c| c.vapor).collect();
    // échelles automatiques
    let mut pmax = o.p_scale;
    if pmax <= 0.0 {
        pmax = (0..nx * ny).filter(|&c| !sim.is_solid(c)).map(|c| (sim.p[c] - P_ATM).abs()).fold(1.0, f64::max);
    }
    let (mut rmin, mut rmax, mut smax) = (f64::MAX, 0.0f64, 1e-9f64);
    for c in 0..nx * ny {
        if !sim.is_solid(c) {
            rmin = rmin.min(sim.rho_mix[c]);
            rmax = rmax.max(sim.rho_mix[c]);
        }
    }
    for j in 0..ny {
        for i in 0..nx {
            let (vx, vy) = cell_velocity(sim, i, j);
            smax = smax.max((vx * vx + vy * vy).sqrt());
        }
    }
    for j in 0..ny {
        for i in 0..nx {
            let c = sim.idx(i, j);
            let rgb: [f32; 3] = if sim.is_solid(c) {
                if o.display == MpDisplay::Temperature {
                    temp_rgb((sim.temp[c] - sim.t_ambient) / o.t_scale.max(1e-3))
                } else {
                    solid_rgb(&sim.reg.solids[sim.solid[c] as usize].name)
                }
            } else {
                match o.display {
                    MpDisplay::Phases => {
                        let mut col = [0.0f32; 3];
                        for k in 0..sim.reg.phases.len() {
                            let a = sim.alpha[k][c].clamp(0.0, 1.0) as f32;
                            let pc = phase_rgb(&sim.reg.phases[k], k);
                            for q in 0..3 {
                                col[q] += a * pc[q];
                            }
                        }
                        col
                    }
                    MpDisplay::Pressure => signed_rgb(sim.p[c] - P_ATM, pmax),
                    MpDisplay::Temperature => temp_rgb((sim.temp[c] - sim.t_ambient) / o.t_scale.max(1e-3)),
                    MpDisplay::Velocity => {
                        let (vx, vy) = cell_velocity(sim, i, j);
                        let s = (vx * vx + vy * vy).sqrt();
                        hsv((vy.atan2(vx).to_degrees() + 180.0) as f32, 1.0, (s / smax).clamp(0.0, 1.0) as f32)
                    }
                    MpDisplay::Density => {
                        let x = if rmax > rmin { ((sim.rho_mix[c].ln() - rmin.ln()) / (rmax.ln() - rmin.ln())).clamp(0.0, 1.0) as f32 } else { 0.0 };
                        [x, x, x]
                    }
                    MpDisplay::Vapor => {
                        let a: f64 = vapor_ids.iter().map(|&k| sim.alpha[k][c]).sum::<f64>().clamp(0.0, 1.0);
                        let a = a as f32;
                        [a, a, a]
                    }
                }
            };
            let row = ny - 1 - j; // j croît vers le haut
            let p = (row * nx + i) * 4;
            out[p] = (rgb[0].clamp(0.0, 1.0) * 255.0) as u8;
            out[p + 1] = (rgb[1].clamp(0.0, 1.0) * 255.0) as u8;
            out[p + 2] = (rgb[2].clamp(0.0, 1.0) * 255.0) as u8;
            out[p + 3] = 255;
        }
    }
}

/// vitesse au centre de la cellule (m/s) : moyenne des deux faces
pub fn cell_velocity(sim: &MultiphaseSim, i: usize, j: usize) -> (f64, f64) {
    let vx = 0.5 * (sim.u[sim.uidx(i, j)] + sim.u[sim.uidx(i + 1, j)]);
    let vy = 0.5 * (sim.v[sim.vidx(i, j)] + sim.v[sim.vidx(i, j + 1)]);
    (vx, vy)
}

// ------------------------------------------------------------------------------------------------ statistiques
#[derive(Clone, Debug, Default)]
pub struct MpStats {
    pub time: f64,
    pub dt: f64,
    pub steps: usize,
    pub pressure_iters: usize,
    /// (nom, masse kg/m, aire m²) par phase
    pub phases: Vec<(String, f64, f64)>,
    pub kinetic: f64,
    pub internal: f64,
    pub t_min: f64,
    pub t_max: f64,
    pub p_min: f64,
    pub p_max: f64,
    pub speed_max: f64,
    pub finite: bool,
}

pub fn compute_stats(sim: &MultiphaseSim) -> MpStats {
    let mut st = MpStats {
        time: sim.time,
        dt: sim.last_dt,
        steps: sim.steps_done(),
        pressure_iters: sim.last_pressure_iters,
        t_min: f64::MAX,
        t_max: f64::MIN,
        p_min: f64::MAX,
        p_max: f64::MIN,
        ..Default::default()
    };
    for k in 0..sim.reg.phases.len() {
        st.phases.push((sim.reg.phases[k].name.clone(), sim.total_mass(k), sim.phase_area(k)));
    }
    st.kinetic = sim.kinetic_energy();
    st.internal = sim.total_internal_energy();
    for c in 0..sim.nx * sim.ny {
        st.t_min = st.t_min.min(sim.temp[c]);
        st.t_max = st.t_max.max(sim.temp[c]);
        if !sim.is_solid(c) {
            st.p_min = st.p_min.min(sim.p[c]);
            st.p_max = st.p_max.max(sim.p[c]);
        }
    }
    st.speed_max = sim.max_speed();
    st.finite = st.speed_max.is_finite() && st.p_min.is_finite() && st.p_max.is_finite() && st.t_max.is_finite() && st.kinetic.is_finite();
    st
}

/// (libellé, valeur) pour la cellule (i, j)
pub fn cell_info(sim: &MultiphaseSim, i: usize, j: usize) -> Vec<(String, String)> {
    let c = sim.idx(i, j);
    let mut v = vec![("Position".to_string(), format!("{i}, {j}")), ("Température (K)".into(), format!("{:.2}", sim.temp[c]))];
    if sim.is_solid(c) {
        v.push(("Solide".into(), sim.reg.solids[sim.solid[c] as usize].name.clone()));
        return v;
    }
    let (vx, vy) = cell_velocity(sim, i, j);
    v.push(("Pression (Pa)".into(), format!("{:.1}", sim.p[c])));
    v.push(("ρ mélange (kg/m³)".into(), format!("{:.3}", sim.rho_mix[c])));
    v.push(("Vitesse (m/s)".into(), format!("{:.3}, {:.3}", vx, vy)));
    for k in 0..sim.reg.phases.len() {
        if sim.alpha[k][c] > 1e-6 {
            v.push((format!("α {}", sim.reg.phases[k].name), format!("{:.4}", sim.alpha[k][c])));
        }
    }
    for pc in &sim.reg.changes {
        v.push((format!("T_sat à p ({})", sim.reg.phases[pc.liquid].name), format!("{:.2} K", pc.t_sat(sim.p[c]))));
    }
    v
}

impl MultiphaseSim {
    /// nombre de pas effectués
    pub fn steps_done(&self) -> usize {
        self.step_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(sim: &MultiphaseSim, o: RenderOpts, path: &str) {
        let mut buf = vec![];
        render_rgba(sim, o, &mut buf);
        if let Ok(dir) = std::env::var("MP_VIEW_OUT") {
            let mut f = String::new();
            f.push_str(&format!("P6\n{} {}\n255\n", sim.nx, sim.ny));
            let mut bytes = f.into_bytes();
            for px in buf.chunks(4) {
                bytes.extend_from_slice(&px[..3]);
            }
            std::fs::write(format!("{dir}/{path}.ppm"), bytes).unwrap();
        }
    }

    #[test]
    fn scenarios_build_and_run_and_render() {
        for sc in MpScenario::ALL {
            let mut s = build_scenario(sc, 48, 36, 0.01, 9.81);
            let m0 = s.total_mass_all();
            for _ in 0..15 {
                s.step();
            }
            let st = compute_stats(&s);
            assert!(st.finite, "{sc:?} non fini");
            let o = RenderOpts { display: MpDisplay::Phases, t_scale: 100.0, p_scale: 0.0 };
            png(&s, o, &format!("{sc:?}"));
            let mut buf = vec![];
            for d in MpDisplay::ALL {
                render_rgba(&s, RenderOpts { display: d, ..o }, &mut buf);
                assert_eq!(buf.len(), 48 * 36 * 4);
            }
            // masse conservée (bords fermés ; Boiling/Empty ont le haut ouvert : on ne teste que les fermés)
            if !matches!(sc, MpScenario::Boiling | MpScenario::Empty) {
                assert!(((s.total_mass_all() - m0) / m0).abs() < 1e-9, "{sc:?}: masse");
            }
        }
    }

    #[test]
    fn brush_tools_keep_state_consistent() {
        let mut s = build_scenario(MpScenario::Empty, 30, 30, 0.01, 9.81);
        let cells = brush_cells(30, 30, 15, 10, 3, true);
        apply_tool(&mut s, MpTool::Phase(EAU), &cells, 293.15, (0.0, 0.0), 0.016);
        assert!(s.alpha[EAU][s.idx(15, 10)] > 0.999);
        apply_tool(&mut s, MpTool::Solid(ACIER), &brush_cells(30, 30, 5, 5, 2, false), 400.0, (0.0, 0.0), 0.016);
        assert!(s.is_solid(s.idx(5, 5)) && (s.temp[s.idx(5, 5)] - 400.0).abs() < 1e-6);
        apply_tool(&mut s, MpTool::Heat, &cells, 350.0, (0.0, 0.0), 0.1);
        assert!(s.temp[s.idx(15, 10)] > 293.5);
        apply_tool(&mut s, MpTool::EraseSolid, &brush_cells(30, 30, 5, 5, 2, false), 293.15, (0.0, 0.0), 0.016);
        assert!(!s.is_solid(s.idx(5, 5)) && s.alpha[AIR][s.idx(5, 5)] > 0.999);
        apply_tool(&mut s, MpTool::Push, &cells, 293.15, (1.0, 0.0), 0.05);
        for _ in 0..30 {
            s.step();
        }
        assert!(compute_stats(&s).finite);
    }
}
