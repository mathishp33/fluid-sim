// ===== Solveur multiphasique GPU (f32) — miroir du solveur CPU f64 (multiphase.rs et mp_*.rs) =====
// Variables : masses partielles m_k (kg/m³), énergie RELATIVE ε = Σ m_k c_k (T − T0) (conditionnement f32), pression RELATIVE
// prel = p − P0 (P0 = 101325 Pa). ε se transforme sous changement de phase : Δε = −dm·h0 (h0 = e_v(T0) − e_l(T0)).
const NPMAX: u32 = 4u;

struct PhaseP {
    a: vec4<f32>, // kind (0 gaz, 1 liquide), cp, c_e, e0
    b: vec4<f32>, // gaz : R_spec, -, k_ref, mu_ref ; liquide : B, n, k_ref, mu_ref
    c: vec4<f32>, // thermique liquide : id (0 beta, 1 Kell), rho0, t0, beta
    d: vec4<f32>, // mu_law id, mu_a, mu_b, sigma
    e: vec4<f32>, // k_law id, k_a, k_b, t_ref
};

struct Params {
    nx: u32, ny: u32, np: u32, step_parity: u32,
    dx: f32, dt: f32, gx: f32, gy: f32,
    p0: f32, t0: f32, t_amb: f32, ambient: u32,
    bc_l: u32, bc_r: u32, bc_b: u32, bc_t: u32,       // 0 paroi, 1 ouvert
    pb_l: f32, pb_r: f32, pb_b: f32, pb_t: f32,       // pression relative imposée (ouvert)
    flags: u32, nsub: u32, newton_iter: u32, pcg_iter: u32, // flags : bit0 tension, bit1 viscosité, bit2 conduction, bit3 chgt de phase
    pc_liq: u32, pc_vap: u32, pc_on: u32, _p0: u32,
    pc_lref: f32, pc_tref: f32, pc_pref: f32, pc_rv: f32,
    pc_dcp: f32, pc_h0: f32, nuclei: f32, nuc_dT: f32,
    nuc_alpha: f32, _q0: f32, _q1: f32, _q2: f32,
    solid_rhocp: vec4<f32>, solid_k: vec4<f32>,
    ph: array<PhaseP, 4>,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> mass: array<f32>;    // [k*n + c]
@group(0) @binding(2) var<storage, read_write> alpha: array<f32>;   // [k*n + c]
@group(0) @binding(3) var<storage, read_write> eps: array<f32>;
@group(0) @binding(4) var<storage, read_write> temp: array<f32>;
@group(0) @binding(5) var<storage, read_write> prel: array<f32>;
@group(0) @binding(6) var<storage, read_write> rho_mix: array<f32>;
@group(0) @binding(7) var<storage, read_write> mu_cell: array<f32>;
@group(0) @binding(8) var<storage, read_write> k_cell: array<f32>;
@group(0) @binding(9) var<storage, read_write> solid: array<i32>;
@group(0) @binding(10) var<storage, read_write> U: array<f32>;      // (nx+1)*ny
@group(0) @binding(11) var<storage, read_write> V: array<f32>;      // nx*(ny+1)
@group(0) @binding(12) var<storage, read_write> Us: array<f32>;     // vitesse prédite u*
@group(0) @binding(13) var<storage, read_write> Vs: array<f32>;
@group(0) @binding(14) var<storage, read_write> Fx: array<f32>;     // accélération × dt (forces)
@group(0) @binding(15) var<storage, read_write> Fy: array<f32>;
@group(0) @binding(16) var<storage, read_write> w1: array<f32>;     // tampons de travail (cellules, k*n + c)
@group(0) @binding(17) var<storage, read_write> w2: array<f32>;
@group(0) @binding(18) var<storage, read_write> w3: array<f32>;
@group(0) @binding(19) var<storage, read_write> w4: array<f32>;
@group(0) @binding(20) var<storage, read_write> mfx: array<f32>;
@group(0) @binding(21) var<storage, read_write> mfy: array<f32>;
@group(0) @binding(22) var<storage, read_write> mtot_old: array<f32>;
@group(0) @binding(23) var<storage, read_write> fl_phase: array<f32>; // flux de volume par phase et par face [f*np + k]
@group(0) @binding(24) var<storage, read_write> fl_mass: array<f32>;  // flux de masse par phase et par face
@group(0) @binding(25) var<storage, read_write> fl_en: array<f32>;    // flux d'énergie relative par face
@group(0) @binding(26) var<storage, read_write> tst: array<f32>;      // T* (début de la résolution de pression)
@group(0) @binding(27) var<storage, read_write> p_start: array<f32>;
@group(0) @binding(28) var<storage, read_write> bnd_flux: array<f32>; // masse entrée par bord et par phase (diagnostic)

fn ncell() -> u32 { return P.nx * P.ny; }
fn cidx(i: u32, j: u32) -> u32 { return j * P.nx + i; }
fn uidx(i: u32, j: u32) -> u32 { return j * (P.nx + 1u) + i; }
fn vidx(i: u32, j: u32) -> u32 { return j * P.nx + i; }
fn solid_id(c: u32) -> i32 { return i32(solid[c]); }
fn is_solid(c: u32) -> bool { return solid_id(c) >= 0; }
fn ci(i: i32, lo: i32, hi: i32) -> i32 { return clamp(i, lo, hi); }

fn law_factor(id: u32, a: f32, b: f32, t_in: f32, t_ref: f32) -> f32 {
    let t = clamp(t_in, 100.0, 4000.0);
    var f = 1.0;
    if (id == 1u) { let r = t / t_ref; f = r * sqrt(r) * (t_ref + a) / (t + a); }
    else if (id == 2u) { let d = max(t - b, 20.0); let dr = max(t_ref - b, 20.0); f = exp(2.302585093 * a * (1.0 / d - 1.0 / dr)); }
    else if (id == 3u) { f = exp(a * (1.0 / t - 1.0 / t_ref)); }
    else if (id == 4u) { f = 1.0 + a * (t - t_ref); }
    return clamp(f, 0.01, 100.0);
}

fn kell_rho(t: f32) -> f32 {
    let c = clamp(t - 273.15, 0.0, 150.0);
    let num = 999.83952 + 16.945176 * c - 7.9870401e-3 * c * c - 46.170461e-6 * c * c * c + 105.56302e-9 * c * c * c * c - 280.54253e-12 * c * c * c * c * c;
    return num / (1.0 + 16.87985e-3 * c);
}
fn liq_rho_p0(k: u32, t: f32) -> f32 {
    if (P.ph[k].c.x > 0.5) { return kell_rho(t); }
    return P.ph[k].c.y / (1.0 + P.ph[k].c.w * (t - P.ph[k].c.z));
}
fn is_gas(k: u32) -> bool { return P.ph[k].a.x < 0.5; }

// masse volumique de la phase k à (pression relative pr, T)
fn rho_k(k: u32, pr: f32, t: f32) -> f32 {
    if (is_gas(k)) { return max(pr + P.p0, 1.0) / (P.ph[k].b.x * max(t, 1.0)); }
    let x = max(pr / (P.p0 + P.ph[k].b.x), -0.99);
    return liq_rho_p0(k, t) * exp(log(1.0 + x) / P.ph[k].b.y);
}
fn kappa_k(k: u32, pr: f32) -> f32 {
    if (is_gas(k)) { return 1.0 / max(pr + P.p0, 1.0); }
    return 1.0 / (P.ph[k].b.y * max(pr + P.p0 + P.ph[k].b.x, 1.0));
}
fn beta_k(k: u32, t: f32) -> f32 {
    if (is_gas(k)) { return 1.0 / max(t, 1.0); }
    if (P.ph[k].c.x > 0.5) { return -(kell_rho(t + 1.0) - kell_rho(t - 1.0)) / (2.0 * kell_rho(t)); }
    return P.ph[k].c.w / (1.0 + P.ph[k].c.w * (t - P.ph[k].c.z));
}
fn mu_k(k: u32, t: f32) -> f32 { return P.ph[k].b.w * law_factor(u32(P.ph[k].d.x), P.ph[k].d.y, P.ph[k].d.z, t, P.ph[k].e.w); }
fn cond_k(k: u32, t: f32) -> f32 { return P.ph[k].b.z * law_factor(u32(P.ph[k].e.x), P.ph[k].e.y, P.ph[k].e.z, t, P.ph[k].e.w); }

fn heat_cap(c: u32) -> f32 {
    let s = solid_id(c);
    if (s >= 0) { return P.solid_rhocp[u32(s)]; }
    var h = 0.0;
    for (var k = 0u; k < P.np; k++) { h += mass[k * ncell() + c] * P.ph[k].a.z; }
    return h;
}

// --- face density (moyenne arithmétique des cellules, bord : la cellule seule) ---
fn frho_x(i: u32, j: u32) -> f32 {
    if (i == 0u) { return rho_mix[cidx(0u, j)]; }
    if (i == P.nx) { return rho_mix[cidx(P.nx - 1u, j)]; }
    return 0.5 * (rho_mix[cidx(i - 1u, j)] + rho_mix[cidx(i, j)]);
}
fn frho_y(i: u32, j: u32) -> f32 {
    if (j == 0u) { return rho_mix[cidx(i, 0u)]; }
    if (j == P.ny) { return rho_mix[cidx(i, P.ny - 1u)]; }
    return 0.5 * (rho_mix[cidx(i, j - 1u)] + rho_mix[cidx(i, j)]);
}

// ===== refresh : T, α, ρ_mix, μ, k depuis (m, ε, p) =====
@compute @workgroup_size(64)
fn k_refresh(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    let n = ncell();
    if (c >= n) { return; }
    let s = solid_id(c);
    if (s >= 0) {
        temp[c] = P.t0 + eps[c] / P.solid_rhocp[u32(s)];
        rho_mix[c] = 1000.0; // sans effet : les faces solides sont des parois
        k_cell[c] = P.solid_k[u32(s)];
        mu_cell[c] = 0.0;
        return;
    }
    let cap = max(heat_cap(c), 1e-30);
    let t = P.t0 + eps[c] / cap;
    temp[c] = t;
    let pr = prel[c];
    var vol = 0.0;
    var vk: array<f32, 4>;
    var mt = 0.0;
    for (var k = 0u; k < P.np; k++) {
        let m = max(mass[k * n + c], 0.0);
        vk[k] = m / rho_k(k, pr, t);
        vol += vk[k];
        mt += mass[k * n + c];
    }
    var mu = 0.0; var kc = 0.0;
    for (var k = 0u; k < P.np; k++) {
        let a = vk[k] / max(vol, 1e-30);
        alpha[k * n + c] = a;
        mu += a * mu_k(k, t);
        kc += a * cond_k(k, t);
    }
    rho_mix[c] = mt / max(vol, 1e-30);
    mu_cell[c] = mu;
    k_cell[c] = kc;
}

@compute @workgroup_size(64)
fn k_save_mtot(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n) { return; }
    var mt = 0.0;
    for (var k = 0u; k < P.np; k++) { mt += mass[k * n + c]; }
    mtot_old[c] = mt;
}

@compute @workgroup_size(64)
fn k_zero_flux(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < (P.nx + 1u) * P.ny) { mfx[f] = 0.0; }
    if (f < P.nx * (P.ny + 1u)) { mfy[f] = 0.0; }
}
