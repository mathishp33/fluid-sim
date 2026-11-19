//! Pont CPU → GPU du solveur multiphasique : construction du buffer uniforme `Params` (512 octets, EXACTEMENT la disposition de
//! `struct Params` de multiphase.wgsl), conversion de l'état (m_k, ε, p relatif, u, v, solides) en tableaux f32, et export JSON
//! (utilisé par le banc de comparaison CPU/GPU en WebGPU headless).

use super::materials::PropertyLaw;
use super::multiphase::{Bc, MultiphaseSim};
use super::phases::{Eos, Thermal};

pub const P_REF: f64 = 101325.0;
pub const T_REF: f64 = 293.15;
pub const PARAMS_BYTES: usize = 512;
pub const NP_MAX: usize = 4;

pub const FLAG_ST: u32 = 1;
pub const FLAG_VISC: u32 = 2;
pub const FLAG_COND: u32 = 4;
pub const FLAG_PC: u32 = 8;

fn law_ids(l: &PropertyLaw) -> (f32, f32, f32) {
    let (id, a, b) = l.gpu_params();
    (id as f32, a, b)
}

pub struct ParamsWriter(pub Vec<u8>);
impl ParamsWriter {
    fn put_u32(&mut self, off: usize, v: u32) {
        self.0[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    fn put_f32(&mut self, off: usize, v: f32) {
        self.0[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
}

impl MultiphaseSim {
    /// Nombre de sous-pas de conduction nécessaires (borne conservative sur k/(ρ cp))
    pub fn conduction_substeps(&self, dt: f64) -> u32 {
        let mut amax = 0.0f64;
        for (c, &k) in self.k_cell.iter().enumerate() {
            let cap = self.heat_capacity(c);
            if cap > 0.0 {
                amax = amax.max(k / cap);
            }
        }
        ((dt * amax / (0.2 * self.dx * self.dx)).ceil() as u32).clamp(1, 2000)
    }

    /// Buffer uniforme pour un pas de durée `dt`.
    pub fn gpu_params_bytes(&self, dt: f64, nsub: u32, newton_iter: u32, pcg_iter: u32) -> Vec<u8> {
        assert!(self.reg.phases.len() <= NP_MAX, "au plus {NP_MAX} phases sur GPU");
        let mut w = ParamsWriter(vec![0u8; PARAMS_BYTES]);
        let np = self.reg.phases.len() as u32;
        w.put_u32(0, self.nx as u32);
        w.put_u32(4, self.ny as u32);
        w.put_u32(8, np);
        w.put_u32(12, (self.step_count % 2) as u32);
        w.put_f32(16, self.dx as f32);
        w.put_f32(20, dt as f32);
        w.put_f32(24, self.gravity[0] as f32);
        w.put_f32(28, self.gravity[1] as f32);
        w.put_f32(32, P_REF as f32);
        w.put_f32(36, T_REF as f32);
        w.put_f32(40, self.t_ambient as f32);
        w.put_u32(44, self.ambient_phase as u32);
        for (q, b) in self.bc.iter().enumerate() {
            let (open, p) = match *b {
                Bc::Wall => (0u32, 0.0),
                Bc::Open { p } => (1u32, p - P_REF),
            };
            w.put_u32(48 + 4 * q, open);
            w.put_f32(64 + 4 * q, p as f32);
        }
        let mut flags = 0;
        if self.surface_tension { flags |= FLAG_ST; }
        if self.viscous { flags |= FLAG_VISC; }
        if self.conduction { flags |= FLAG_COND; }
        if self.phase_change { flags |= FLAG_PC; }
        w.put_u32(80, flags);
        w.put_u32(84, nsub);
        w.put_u32(88, newton_iter);
        w.put_u32(92, pcg_iter);
        if let (true, Some(pc)) = (self.phase_change, self.reg.changes.first()) {
            let (l, v) = (&self.reg.phases[pc.liquid], &self.reg.phases[pc.vapor]);
            w.put_u32(96, pc.liquid as u32);
            w.put_u32(100, pc.vapor as u32);
            w.put_u32(104, 1);
            w.put_f32(112, pc.l_ref as f32);
            w.put_f32(116, pc.t_ref as f32);
            w.put_f32(120, pc.p_ref as f32);
            w.put_f32(124, pc.r_v as f32);
            w.put_f32(128, pc.dcp as f32);
            // h0 = e_v(T0) − e_l(T0) : variation de ε quand dm passe de liquide à vapeur est −dm·h0
            w.put_f32(132, (v.e_int(T_REF) - l.e_int(T_REF)) as f32);
            w.put_f32(136, self.nuclei_density as f32);
            w.put_f32(140, self.nucleation_superheat as f32);
            w.put_f32(144, self.nucleation_alpha as f32);
        }
        for (s, sol) in self.reg.solids.iter().take(4).enumerate() {
            w.put_f32(160 + 4 * s, sol.rho_cp() as f32);
            w.put_f32(176 + 4 * s, sol.conductivity as f32);
        }
        for (k, ph) in self.reg.phases.iter().enumerate() {
            let o = 192 + 80 * k;
            let kind = if ph.is_gas() { 0.0 } else { 1.0 };
            w.put_f32(o, kind);
            w.put_f32(o + 4, ph.cp as f32);
            w.put_f32(o + 8, ph.c_e() as f32);
            w.put_f32(o + 12, ph.e0 as f32);
            match ph.eos {
                Eos::IdealGas { r_spec } => w.put_f32(o + 16, r_spec as f32),
                Eos::Tait { b, n, thermal, .. } => {
                    w.put_f32(o + 16, b as f32);
                    w.put_f32(o + 20, n as f32);
                    match thermal {
                        Thermal::Kell => w.put_f32(o + 32, 1.0),
                        Thermal::Beta { rho0, t0, beta } => {
                            w.put_f32(o + 32, 0.0);
                            w.put_f32(o + 36, rho0 as f32);
                            w.put_f32(o + 40, t0 as f32);
                            w.put_f32(o + 44, beta as f32);
                        }
                    }
                }
            }
            w.put_f32(o + 24, ph.conductivity as f32);
            w.put_f32(o + 28, ph.mu_ref as f32);
            let (mid, ma, mb) = law_ids(&ph.mu_law);
            w.put_f32(o + 48, mid);
            w.put_f32(o + 52, ma);
            w.put_f32(o + 56, mb);
            w.put_f32(o + 60, ph.sigma as f32);
            let (kid, ka, kb) = law_ids(&ph.k_law);
            w.put_f32(o + 64, kid);
            w.put_f32(o + 68, ka);
            w.put_f32(o + 72, kb);
            w.put_f32(o + 76, ph.t_ref as f32);
        }
        w.0
    }

    /// ε = Σ m_k c_k (T − T0) (solide : ρ c (T − T0))
    pub fn eps_field(&self) -> Vec<f64> {
        (0..self.nx * self.ny)
            .map(|c| {
                let t = self.temp_from_energy(c);
                self.heat_capacity(c) * (t - T_REF)
            })
            .collect()
    }

    /// Tableaux f32 [masses k-major, ε, p relatif, u, v, solide] pour l'envoi au GPU
    pub fn gpu_state(&self) -> GpuState {
        let n = self.nx * self.ny;
        let np = self.reg.phases.len();
        let mut mass = vec![0f32; np * n];
        for k in 0..np {
            for c in 0..n {
                mass[k * n + c] = self.mass[k][c] as f32;
            }
        }
        GpuState {
            mass,
            eps: self.eps_field().iter().map(|&v| v as f32).collect(),
            prel: self.p.iter().map(|&p| (p - P_REF) as f32).collect(),
            u: self.u.iter().map(|&v| v as f32).collect(),
            v: self.v.iter().map(|&v| v as f32).collect(),
            solid: self.solid.clone(),
        }
    }

    /// Relit un état GPU (après readback) dans le solveur CPU pour comparaison / affichage : met à jour m, ε→e, p, u, v.
    pub fn load_gpu_state(&mut self, s: &GpuState) {
        let n = self.nx * self.ny;
        let np = self.reg.phases.len();
        for k in 0..np {
            for c in 0..n {
                self.mass[k][c] = s.mass[k * n + c] as f64;
            }
        }
        for c in 0..n {
            self.p[c] = s.prel[c] as f64 + P_REF;
            let t = T_REF + s.eps[c] as f64 / self.heat_capacity(c).max(1e-30);
            // e = Σ m (c T + e0) ; solide : ρ c T
            self.e[c] = if self.is_solid(c) {
                self.reg.solids[self.solid[c] as usize].rho_cp() * t
            } else {
                (0..np).map(|k| self.mass[k][c] * self.reg.phases[k].e_int(t)).sum()
            };
        }
        self.u = s.u.iter().map(|&v| v as f64).collect();
        self.v = s.v.iter().map(|&v| v as f64).collect();
        self.refresh();
    }
}

pub struct GpuState {
    pub mass: Vec<f32>,
    pub eps: Vec<f32>,
    pub prel: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub solid: Vec<i32>,
}

fn arr<T: std::fmt::Display>(v: &[T]) -> String {
    let mut s = String::from("[");
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&x.to_string());
    }
    s.push(']');
    s
}

impl GpuState {
    pub fn to_json(&self) -> String {
        format!(
            "{{\"mass\":{},\"eps\":{},\"prel\":{},\"u\":{},\"v\":{},\"solid\":{}}}",
            arr(&self.mass), arr(&self.eps), arr(&self.prel), arr(&self.u), arr(&self.v), arr(&self.solid)
        )
    }
}

/// JSON de comparaison : paramètres (octets), état initial, dt de chaque pas CPU, état final CPU
pub fn export_case(sim: &mut MultiphaseSim, steps: usize, newton: u32, pcg: u32, path: &str) {
    let init = sim.gpu_state();
    let mut dts = vec![];
    let mut nsubs = vec![];
    let mut parities = vec![];
    for _ in 0..steps {
        sim.refresh();
        let dt = sim.choose_dt();
        nsubs.push(sim.conduction_substeps(dt));
        parities.push(sim.step_count % 2);
        let used = sim.step();
        dts.push(used);
    }
    let fin = sim.gpu_state();
    let bytes = sim.gpu_params_bytes(dts[0], 1, newton, pcg);
    let json = format!(
        "{{\"header\":{:?},\"total\":{},\"offsets\":{{{}}},\"nx\":{},\"ny\":{},\"np\":{},\"parity0\":0,\"params\":{},\"dts\":{},\"nsubs\":{},\"newton\":{},\"pcg\":{},\"init\":{},\"final\":{}}}",
        gpu_layout_header(sim.nx, sim.ny, sim.reg.phases.len()).0,
        gpu_layout_header(sim.nx, sim.ny, sim.reg.phases.len()).1,
        gpu_layout_header(sim.nx, sim.ny, sim.reg.phases.len()).2.iter().map(|(n, o, l)| format!("\"{n}\":[{o},{l}]")).collect::<Vec<_>>().join(","),
        sim.nx, sim.ny, sim.reg.phases.len(), arr(&bytes), arr(&dts), arr(&nsubs), newton, pcg, init.to_json(), fin.to_json()
    );
    std::fs::write(path, json).unwrap();
}

/// Disposition du buffer de stockage unique `S` (en f32) : (nom, nombre d'éléments). Doit rester synchronisée avec `names` de
/// build_wgsl.py (même ordre n'est pas requis : l'en-tête généré donne les offsets).
pub fn gpu_layout(nx: usize, ny: usize, np: usize) -> Vec<(&'static str, usize)> {
    let n = nx * ny;
    let nu = (nx + 1) * ny;
    let nv = nx * (ny + 1);
    let nf = nu.max(nv);
    let wmax = (np * n).max((nx + 1) * (ny + 1));
    vec![
        ("mass", np * n), ("alpha", np * n), ("eps", n), ("temp", n), ("prel", n), ("rho_mix", n), ("mu_cell", n), ("k_cell", n),
        ("solid", n), ("U", nu), ("V", nv), ("Us", nu), ("Vs", nv), ("Fx", nu), ("Fy", nv),
        ("w1", wmax), ("w2", wmax), ("w3", wmax), ("w4", wmax), ("mfx", nu), ("mfy", nv), ("mtot_old", n),
        ("fl_phase", nf * np), ("fl_mass", nf * np), ("fl_en", nf), ("tst", n), ("p_start", n), ("bnd_flux", np),
        ("sd", n), ("sae", n), ("san", n), ("cx", n), ("cr", n), ("cz", n), ("cp_", n), ("cap_", n),
        ("scal", 16), ("part", n / 64 + 2),
    ]
}

/// (en-tête WGSL `const O_x: u32 = …u;`, taille totale en f32, offsets)
pub fn gpu_layout_header(nx: usize, ny: usize, np: usize) -> (String, usize, Vec<(&'static str, usize, usize)>) {
    let mut off = 0usize;
    let mut h = String::new();
    let mut v = vec![];
    for (name, len) in gpu_layout(nx, ny, np) {
        h.push_str(&format!("const O_{name}: u32 = {off}u;\n"));
        v.push((name, off, len));
        off += len;
    }
    (h, off, v)
}

impl GpuState {
    /// Écrit l'état dans une image complète du buffer `S`
    pub fn write_into(&self, sim: &MultiphaseSim, s: &mut [f32]) {
        let (_, _, offs) = gpu_layout_header(sim.nx, sim.ny, sim.reg.phases.len());
        let o = |name: &str| offs.iter().find(|x| x.0 == name).unwrap().1;
        s[o("mass")..o("mass") + self.mass.len()].copy_from_slice(&self.mass);
        s[o("eps")..o("eps") + self.eps.len()].copy_from_slice(&self.eps);
        s[o("prel")..o("prel") + self.prel.len()].copy_from_slice(&self.prel);
        s[o("U")..o("U") + self.u.len()].copy_from_slice(&self.u);
        s[o("V")..o("V") + self.v.len()].copy_from_slice(&self.v);
        for (i, &sv) in self.solid.iter().enumerate() {
            s[o("solid") + i] = sv as f32;
        }
    }
}
