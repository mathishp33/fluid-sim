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
@group(0) @binding(1) var<storage, read_write> S: array<f32>;

fn ncell() -> u32 { return P.nx * P.ny; }
fn cidx(i: u32, j: u32) -> u32 { return j * P.nx + i; }
fn uidx(i: u32, j: u32) -> u32 { return j * (P.nx + 1u) + i; }
fn vidx(i: u32, j: u32) -> u32 { return j * P.nx + i; }
fn solid_id(c: u32) -> i32 { return i32(S[O_solid + (c)]); }
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
    for (var k = 0u; k < P.np; k++) { h += S[O_mass + (k * ncell() + c)] * P.ph[k].a.z; }
    return h;
}

// --- face density (moyenne arithmétique des cellules, bord : la cellule seule) ---
fn frho_x(i: u32, j: u32) -> f32 {
    if (i == 0u) { return S[O_rho_mix + (cidx(0u, j))]; }
    if (i == P.nx) { return S[O_rho_mix + (cidx(P.nx - 1u, j))]; }
    return 0.5 * (S[O_rho_mix + (cidx(i - 1u, j))] + S[O_rho_mix + (cidx(i, j))]);
}
fn frho_y(i: u32, j: u32) -> f32 {
    if (j == 0u) { return S[O_rho_mix + (cidx(i, 0u))]; }
    if (j == P.ny) { return S[O_rho_mix + (cidx(i, P.ny - 1u))]; }
    return 0.5 * (S[O_rho_mix + (cidx(i, j - 1u))] + S[O_rho_mix + (cidx(i, j))]);
}

// ===== refresh : T, α, ρ_mix, μ, k depuis (m, ε, p) =====
@compute @workgroup_size(64)
fn k_refresh(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    let n = ncell();
    if (c >= n) { return; }
    let s = solid_id(c);
    if (s >= 0) {
        S[O_temp + (c)] = P.t0 + S[O_eps + (c)] / P.solid_rhocp[u32(s)];
        S[O_rho_mix + (c)] = 1000.0; // sans effet : les faces solides sont des parois
        S[O_k_cell + (c)] = P.solid_k[u32(s)];
        S[O_mu_cell + (c)] = 0.0;
        return;
    }
    let cap = max(heat_cap(c), 1e-30);
    let t = P.t0 + S[O_eps + (c)] / cap;
    S[O_temp + (c)] = t;
    let pr = S[O_prel + (c)];
    var vol = 0.0;
    var vk: array<f32, 4>;
    var mt = 0.0;
    for (var k = 0u; k < P.np; k++) {
        let m = max(S[O_mass + (k * n + c)], 0.0);
        vk[k] = m / rho_k(k, pr, t);
        vol += vk[k];
        mt += S[O_mass + (k * n + c)];
    }
    var mu = 0.0; var kc = 0.0;
    for (var k = 0u; k < P.np; k++) {
        let a = vk[k] / max(vol, 1e-30);
        S[O_alpha + (k * n + c)] = a;
        mu += a * mu_k(k, t);
        kc += a * cond_k(k, t);
    }
    S[O_rho_mix + (c)] = mt / max(vol, 1e-30);
    S[O_mu_cell + (c)] = mu;
    S[O_k_cell + (c)] = kc;
}

@compute @workgroup_size(64)
fn k_save_mtot(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n) { return; }
    var mt = 0.0;
    for (var k = 0u; k < P.np; k++) { mt += S[O_mass + (k * n + c)]; }
    S[O_mtot_old + (c)] = mt;
}

@compute @workgroup_size(64)
fn k_zero_flux(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < (P.nx + 1u) * P.ny) { S[O_mfx + (f)] = 0.0; }
    if (f < P.nx * (P.ny + 1u)) { S[O_mfy + (f)] = 0.0; }
}

// ===== PLIC conservatif : géométrie, flux de volume par phase, limiteur, application, mise à jour =====
// w1 = α normalisé (an), w2 = α de travail (aw), w3 = facteur de limitation par (cellule, phase)

fn a_from_vol(m1: f32, m2: f32, v: f32) -> f32 {
    if (v <= 0.0) { return 0.0; }
    if (v >= 1.0) { return 1.0; }
    if (m1 < 1e-6) { return v; }
    if (v <= 0.5 * m1 / m2) { return sqrt(2.0 * m1 * m2 * v); }
    if (v <= 1.0 - 0.5 * m1 / m2) { return v * m2 + 0.5 * m1; }
    return 1.0 - sqrt(2.0 * m1 * m2 * (1.0 - v));
}

// plan : (nx, ny, c) avec région {nx x + ny y <= c}
fn plic_new(a: f32, gx: f32, gy: f32) -> vec3<f32> {
    var nx = -gx; var ny = -gy;
    let s = abs(nx) + abs(ny);
    if (s < 1e-7) { nx = 0.0; ny = 1.0; } else { nx = nx / s; ny = ny / s; }
    let ax = abs(nx); let ay = abs(ny);
    let aa = a_from_vol(min(ax, ay), max(ax, ay), a);
    return vec3<f32>(nx, ny, aa + min(nx, 0.0) + min(ny, 0.0));
}

fn plic_area(pl: vec3<f32>, x0i: f32, x1i: f32, y0i: f32, y1i: f32) -> f32 {
    var x0 = x0i; var x1 = x1i; var y0 = y0i; var y1 = y1i;
    let neg = min(pl.x, 0.0) + min(pl.y, 0.0);
    if (pl.x < 0.0) { let t = x0; x0 = 1.0 - x1; x1 = 1.0 - t; }
    if (pl.y < 0.0) { let t = y0; y0 = 1.0 - y1; y1 = 1.0 - t; }
    let a = pl.z - neg;
    let mx = abs(pl.x); let my = abs(pl.y);
    if (mx < 1e-4) { return (x1 - x0) * (clamp(a / my, y0, y1) - y0); }
    if (my < 1e-4) { return (y1 - y0) * (clamp(a / mx, x0, x1) - x0); }
    let d = 2.0 * mx * my;
    // F(X,Y) = T(a) − T(a−mx X) − T(a−my Y) + T(a−mx X−my Y), T(s)=s²/(2 mx my) pour s>0
    let F00 = tri(a, d) - tri(a - mx * x0, d) - tri(a - my * y0, d) + tri(a - mx * x0 - my * y0, d);
    let F10 = tri(a, d) - tri(a - mx * x1, d) - tri(a - my * y0, d) + tri(a - mx * x1 - my * y0, d);
    let F01 = tri(a, d) - tri(a - mx * x0, d) - tri(a - my * y1, d) + tri(a - mx * x0 - my * y1, d);
    let F11 = tri(a, d) - tri(a - mx * x1, d) - tri(a - my * y1, d) + tri(a - mx * x1 - my * y1, d);
    return max(F11 - F01 - F10 + F00, 0.0);
}
fn tri(s: f32, d: f32) -> f32 { if (s > 0.0) { return s * s / d; } return 0.0; }

fn an_at(k: u32, i: i32, j: i32, ci_: u32, cj_: u32) -> f32 {
    let ii = u32(clamp(i, 0, i32(P.nx) - 1));
    let jj = u32(clamp(j, 0, i32(P.ny) - 1));
    let c = cidx(ii, jj);
    if (is_solid(c)) { return S[O_w1 + (k * ncell() + cidx(ci_, cj_))]; }
    return S[O_w1 + (k * ncell() + c)];
}
fn youngs(k: u32, i: u32, j: u32) -> vec2<f32> {
    let I = i32(i); let J = i32(j);
    let gx = ((an_at(k, I + 1, J + 1, i, j) + 2.0 * an_at(k, I + 1, J, i, j) + an_at(k, I + 1, J - 1, i, j))
            - (an_at(k, I - 1, J + 1, i, j) + 2.0 * an_at(k, I - 1, J, i, j) + an_at(k, I - 1, J - 1, i, j))) / 8.0;
    let gy = ((an_at(k, I + 1, J + 1, i, j) + 2.0 * an_at(k, I, J + 1, i, j) + an_at(k, I - 1, J + 1, i, j))
            - (an_at(k, I + 1, J - 1, i, j) + 2.0 * an_at(k, I, J - 1, i, j) + an_at(k, I - 1, J - 1, i, j))) / 8.0;
    return vec2<f32>(gx, gy);
}

@compute @workgroup_size(64)
fn k_aw_init(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i < ncell() * P.np) { S[O_w2 + (i)] = S[O_alpha + (i)]; }
}

@compute @workgroup_size(64)
fn k_an(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n) { return; }
    var s = 0.0;
    for (var k = 0u; k < P.np; k++) { s += clamp(S[O_w2 + (k * n + c)], 0.0, 1.0); }
    s = max(s, 1e-30);
    for (var k = 0u; k < P.np; k++) { S[O_w1 + (k * n + c)] = clamp(S[O_w2 + (k * n + c)], 0.0, 1.0) / s; }
}

struct FaceInfo { uf: f32, lo: i32, hi: i32, fi: u32, fj: u32, valid: bool };
fn face_info(f: u32, dir: u32) -> FaceInfo {
    var r: FaceInfo;
    r.valid = false; r.lo = -1; r.hi = -1; r.uf = 0.0;
    var fiw = P.nx + 1u; var fjh = P.ny;
    if (dir == 1u) { fiw = P.nx; fjh = P.ny + 1u; }
    r.fi = f % fiw; r.fj = f / fiw;
    if (r.fj >= fjh) { return r; }
    if (dir == 0u) {
        r.uf = S[O_U + (uidx(r.fi, r.fj))];
        if (r.fi > 0u) { r.lo = i32(cidx(r.fi - 1u, r.fj)); }
        if (r.fi < P.nx) { r.hi = i32(cidx(r.fi, r.fj)); }
    } else {
        r.uf = S[O_V + (vidx(r.fi, r.fj))];
        if (r.fj > 0u) { r.lo = i32(cidx(r.fi, r.fj - 1u)); }
        if (r.fj < P.ny) { r.hi = i32(cidx(r.fi, r.fj)); }
    }
    if (r.uf == 0.0) { return r; }
    if (r.lo >= 0 && is_solid(u32(r.lo))) { return r; }
    if (r.hi >= 0 && is_solid(u32(r.hi))) { return r; }
    r.valid = true;
    return r;
}

fn phase_order(pos: u32) -> u32 {
    // liquides d'abord (indices croissants), puis gaz
    var cnt = 0u;
    for (var k = 0u; k < P.np; k++) { if (!is_gas(k)) { if (cnt == pos) { return k; } cnt++; } }
    for (var k = 0u; k < P.np; k++) { if (is_gas(k)) { if (cnt == pos) { return k; } cnt++; } }
    return 0u;
}

fn sweep_flux(f: u32, dir: u32) {
    let np = P.np; let n = ncell();
    let info = face_info(f, dir);
    for (var k = 0u; k < np; k++) { S[O_fl_phase + (f * np + k)] = 0.0; }
    if (!info.valid) { return; }
    let xi = min(abs(info.uf) * P.dt / P.dx, 1.0);
    var d = info.lo; if (info.uf < 0.0) { d = info.hi; }
    if (d < 0) { S[O_fl_phase + (f * np + P.ambient)] = xi; return; }
    let dc = u32(d);
    var s0 = 0.0; var s1 = xi; if (info.uf > 0.0) { s0 = 1.0 - xi; s1 = 1.0; }
    // phase pure ?
    for (var k = 0u; k < np; k++) {
        if (S[O_w1 + (k * n + dc)] >= 1.0 - 1e-6) { S[O_fl_phase + (f * np + k)] = xi; return; }
    }
    let ci_ = dc % P.nx; let cj_ = dc / P.nx;
    var rem = xi;
    var spare: array<f32, 4>;
    for (var pos = 0u; pos < np; pos++) {
        let k = phase_order(pos);
        let a = S[O_w1 + (k * n + dc)];
        var want = 0.0;
        if (pos == np - 1u) { want = rem; }
        else if (a > 1e-6 && a < 1.0 - 1e-6) {
            let gr = youngs(k, ci_, cj_);
            let pl = plic_new(a, gr.x, gr.y);
            if (dir == 0u) { want = plic_area(pl, s0, s1, 0.0, 1.0); } else { want = plic_area(pl, 0.0, 1.0, s0, s1); }
        }
        let fk = max(min(min(want, rem), a), 0.0);
        S[O_fl_phase + (f * np + k)] = fk;
        rem -= fk;
        spare[k] = a - fk;
    }
    if (rem > 1e-7) {
        var tot = 0.0;
        for (var k = 0u; k < np; k++) { tot += max(spare[k], 0.0); }
        if (tot > 0.0) { for (var k = 0u; k < np; k++) { S[O_fl_phase + (f * np + k)] += rem * max(spare[k], 0.0) / tot; } }
    }
}
@compute @workgroup_size(64) fn k_sweep_flux_x(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < (P.nx + 1u) * P.ny) { sweep_flux(g.x, 0u); } }
@compute @workgroup_size(64) fn k_sweep_flux_y(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < P.nx * (P.ny + 1u)) { sweep_flux(g.x, 1u); } }

// facteur de limitation : sortie totale de chaque phase ≤ 0.999 α_travail
fn sweep_scale(c: u32, dir: u32) {
    let np = P.np; let n = ncell();
    let i = c % P.nx; let j = c / P.nx;
    var f_lo = uidx(i, j); var f_hi = uidx(i + 1u, j);
    var u_lo = S[O_U + (f_lo)]; var u_hi = S[O_U + (f_hi)];
    if (dir == 1u) { f_lo = vidx(i, j); f_hi = vidx(i, j + 1u); u_lo = S[O_V + (f_lo)]; u_hi = S[O_V + (f_hi)]; }
    for (var k = 0u; k < np; k++) {
        var out = 0.0;
        if (u_lo < 0.0) { out += S[O_fl_phase + (f_lo * np + k)]; }
        if (u_hi > 0.0) { out += S[O_fl_phase + (f_hi * np + k)]; }
        let cap = max(S[O_w2 + (k * n + c)], 0.0) * 0.999;
        var s = 1.0;
        if (out > cap && out > 0.0) { s = cap / out; }
        S[O_w3 + (k * n + c)] = s;
    }
}
@compute @workgroup_size(64) fn k_sweep_scale_x(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < ncell()) { sweep_scale(g.x, 0u); } }
@compute @workgroup_size(64) fn k_sweep_scale_y(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < ncell()) { sweep_scale(g.x, 1u); } }

fn sweep_apply(f: u32, dir: u32) {
    let np = P.np; let n = ncell();
    let info = face_info(f, dir);
    var en = 0.0; var mt = 0.0;
    var d = info.lo; var r = info.hi;
    if (info.uf < 0.0) { d = info.hi; r = info.lo; }
    for (var k = 0u; k < np; k++) {
        var fk = S[O_fl_phase + (f * np + k)];
        var mf = 0.0;
        if (info.valid && fk != 0.0) {
            var rho_d = 0.0; var t_d = P.t_amb;
            if (d < 0) {
                var pr = 0.0; if (r >= 0) { pr = S[O_prel + (u32(r))]; }
                rho_d = rho_k(k, pr, P.t_amb);
            } else {
                let dc = u32(d);
                fk = fk * S[O_w3 + (k * n + dc)];
                t_d = S[O_temp + (dc)];
                let aw = S[O_w2 + (k * n + dc)];
                let rho_eos = rho_k(k, S[O_prel + (dc)], t_d);
                rho_d = rho_eos;
                if (aw > 1e-5) { let q = S[O_mass + (k * n + dc)] / aw; if (q > 0.0) { rho_d = q; } }
            }
            S[O_fl_phase + (f * np + k)] = fk;
            mf = fk * rho_d;
            en += mf * P.ph[k].a.z * (t_d - P.t0);
            mt += mf;
        }
        S[O_fl_mass + (f * np + k)] = mf;
    }
    S[O_fl_en + (f)] = en;
    if (info.valid) {
        let sg = sign(info.uf);
        if (dir == 0u) { S[O_mfx + (uidx(info.fi, info.fj))] += sg * mt; } else { S[O_mfy + (vidx(info.fi, info.fj))] += sg * mt; }
    }
}
@compute @workgroup_size(64) fn k_sweep_apply_x(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < (P.nx + 1u) * P.ny) { sweep_apply(g.x, 0u); } }
@compute @workgroup_size(64) fn k_sweep_apply_y(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < P.nx * (P.ny + 1u)) { sweep_apply(g.x, 1u); } }

fn sweep_update(c: u32, dir: u32) {
    if (is_solid(c)) { return; }
    let np = P.np; let n = ncell();
    let i = c % P.nx; let j = c / P.nx;
    var f_lo = uidx(i, j); var f_hi = uidx(i + 1u, j);
    var u_lo = S[O_U + (f_lo)]; var u_hi = S[O_U + (f_hi)];
    if (dir == 1u) { f_lo = vidx(i, j); f_hi = vidx(i, j + 1u); u_lo = S[O_V + (f_lo)]; u_hi = S[O_V + (f_hi)]; }
    var de = 0.0;
    for (var k = 0u; k < np; k++) {
        var dv = 0.0; var dm = 0.0;
        // face basse : U>0 entre dans c ; U<0 sort de c
        if (u_lo > 0.0) { dv += S[O_fl_phase + (f_lo * np + k)]; dm += S[O_fl_mass + (f_lo * np + k)]; }
        if (u_lo < 0.0) { dv -= S[O_fl_phase + (f_lo * np + k)]; dm -= S[O_fl_mass + (f_lo * np + k)]; }
        if (u_hi < 0.0) { dv += S[O_fl_phase + (f_hi * np + k)]; dm += S[O_fl_mass + (f_hi * np + k)]; }
        if (u_hi > 0.0) { dv -= S[O_fl_phase + (f_hi * np + k)]; dm -= S[O_fl_mass + (f_hi * np + k)]; }
        S[O_w2 + (k * n + c)] += dv;
        S[O_mass + (k * n + c)] += dm;
    }
    if (u_lo > 0.0) { de += S[O_fl_en + (f_lo)]; }
    if (u_lo < 0.0) { de -= S[O_fl_en + (f_lo)]; }
    if (u_hi < 0.0) { de += S[O_fl_en + (f_hi)]; }
    if (u_hi > 0.0) { de -= S[O_fl_en + (f_hi)]; }
    S[O_eps + (c)] += de;
    S[O_temp + (c)] = P.t0 + S[O_eps + (c)] / max(heat_cap(c), 1e-30);
}
@compute @workgroup_size(64) fn k_sweep_update_x(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < ncell()) { sweep_update(g.x, 0u); } }
@compute @workgroup_size(64) fn k_sweep_update_y(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < ncell()) { sweep_update(g.x, 1u); } }

// ===== Forces : gravité, tension de surface (CSF + fonctions de hauteur), viscosité ; QDM conservative =====
// Étage forces : w1 = marqueur liquide ℓ, w2 = courbure (1e30 = indéfinie), w3 = ℓ lissé, w4 = σ(cellule).
// Étage viscosité : w2 = τxx, w3 = τyy, w1 = τxy (nœuds (nx+1)(ny+1)).

fn mk(i: i32, j: i32, ci_: u32, cj_: u32) -> f32 {
    let ii = u32(clamp(i, 0, i32(P.nx) - 1)); let jj = u32(clamp(j, 0, i32(P.ny) - 1));
    let c = cidx(ii, jj);
    if (is_solid(c)) { return S[O_w1 + (cidx(ci_, cj_))]; }
    return S[O_w1 + (c)];
}
fn smk(i: i32, j: i32, ci_: u32, cj_: u32) -> f32 {
    let ii = u32(clamp(i, 0, i32(P.nx) - 1)); let jj = u32(clamp(j, 0, i32(P.ny) - 1));
    let c = cidx(ii, jj);
    if (is_solid(c)) { return S[O_w3 + (cidx(ci_, cj_))]; }
    return S[O_w3 + (c)];
}

@compute @workgroup_size(64)
fn k_force_init(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < (P.nx + 1u) * P.ny) { S[O_Fx + (f)] = P.dt * P.gx; }
    if (f < P.nx * (P.ny + 1u)) { S[O_Fy + (f)] = P.dt * P.gy; }
}

@compute @workgroup_size(64)
fn k_marker(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n) { return; }
    var l = 0.0; var sg = 0.0;
    if (!is_solid(c)) {
        for (var k = 0u; k < P.np; k++) { if (!is_gas(k)) { l += S[O_alpha + (k * n + c)]; sg += S[O_alpha + (k * n + c)] * P.ph[k].d.w; } }
    }
    S[O_w1 + (c)] = clamp(l, 0.0, 1.0);
    if (l < 1e-12) { S[O_w4 + (c)] = 0.0; } else { S[O_w4 + (c)] = sg / l; }
}

@compute @workgroup_size(64)
fn k_smooth(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    let i = i32(c % P.nx); let j = i32(c / P.nx);
    var s = 0.0;
    for (var dj = -1; dj <= 1; dj++) { for (var di = -1; di <= 1; di++) {
        s += f32((2 - abs(di)) * (2 - abs(dj))) * mk(i + di, j + dj, u32(i), u32(j));
    } }
    S[O_w3 + (c)] = s / 16.0;
}

fn nrm(i: i32, j: i32, ci_: u32, cj_: u32) -> vec2<f32> {
    let gx = (smk(i + 1, j, ci_, cj_) - smk(i - 1, j, ci_, cj_)) * 0.5;
    let gy = (smk(i, j + 1, ci_, cj_) - smk(i, j - 1, ci_, cj_)) * 0.5;
    let s = max(sqrt(gx * gx + gy * gy), 1e-9);
    return vec2<f32>(-gx / s, -gy / s);
}

@compute @workgroup_size(64)
fn k_curv(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    let iu = c % P.nx; let ju = c / P.nx;
    let i = i32(iu); let j = i32(ju);
    let a = S[O_w1 + (c)];
    var kap = 1e30;
    if (a > 1e-4 && a < 1.0 - 1e-4) {
        // gradient de Youngs du marqueur
        let gx = ((mk(i + 1, j + 1, iu, ju) + 2.0 * mk(i + 1, j, iu, ju) + mk(i + 1, j - 1, iu, ju)) - (mk(i - 1, j + 1, iu, ju) + 2.0 * mk(i - 1, j, iu, ju) + mk(i - 1, j - 1, iu, ju))) / 8.0;
        let gy = ((mk(i + 1, j + 1, iu, ju) + 2.0 * mk(i, j + 1, iu, ju) + mk(i - 1, j + 1, iu, ju)) - (mk(i + 1, j - 1, iu, ju) + 2.0 * mk(i, j - 1, iu, ju) + mk(i - 1, j - 1, iu, ju))) / 8.0;
        let vertical = abs(gy) >= abs(gx);
        var hs: array<f32, 3>;
        var valid = true;
        for (var q = 0; q < 3; q++) {
            let off = q - 1;
            var s = 0.0; var first = 0.0; var last = 0.0;
            for (var t = -3; t <= 3; t++) {
                var v = 0.0;
                if (vertical) { v = mk(i + off, j + t, iu, ju); } else { v = mk(i + t, j + off, iu, ju); }
                if (t == -3) { first = v; }
                if (t == 3) { last = v; }
                s += v;
            }
            if (abs(first - last) < 0.5) { valid = false; }
            hs[q] = s;
        }
        if (valid) {
            let h1 = 0.5 * (hs[2] - hs[0]);
            let h2 = hs[2] - 2.0 * hs[1] + hs[0];
            kap = -h2 / pow(1.0 + h1 * h1, 1.5) / P.dx;
        } else {
            let dv = (nrm(i + 1, j, iu, ju).x - nrm(i - 1, j, iu, ju).x + nrm(i, j + 1, iu, ju).y - nrm(i, j - 1, iu, ju).y) * 0.5 / P.dx;
            kap = dv;
        }
    }
    S[O_w2 + (c)] = kap;
}

fn st_face(a: u32, b: u32) -> f32 {
    let la = S[O_w1 + (a)]; let lb = S[O_w1 + (b)];
    let wa = la * (1.0 - la); let wb = lb * (1.0 - lb);
    let w = wa + wb;
    if (w < 1e-9) { return 0.0; }
    var ka = S[O_w2 + (a)]; if (ka > 1e29) { ka = 0.0; }
    var kb = S[O_w2 + (b)]; if (kb > 1e29) { kb = 0.0; }
    let kf = (ka * wa + kb * wb) / w;
    let sg = (S[O_w4 + (a)] * wa + S[O_w4 + (b)] * wb) / w;
    return sg * kf * (lb - la) / P.dx;
}

@compute @workgroup_size(64)
fn k_force_st_u(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f >= (P.nx + 1u) * P.ny) { return; }
    let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
    if (i == 0u || i >= P.nx) { return; }
    let a = cidx(i - 1u, j); let b = cidx(i, j);
    if (is_solid(a) || is_solid(b)) { return; }
    S[O_Fx + (f)] += P.dt * st_face(a, b) / frho_x(i, j);
}
@compute @workgroup_size(64)
fn k_force_st_v(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f >= P.nx * (P.ny + 1u)) { return; }
    let i = f % P.nx; let j = f / P.nx;
    if (j == 0u || j >= P.ny) { return; }
    let a = cidx(i, j - 1u); let b = cidx(i, j);
    if (is_solid(a) || is_solid(b)) { return; }
    S[O_Fy + (f)] += P.dt * st_face(a, b) / frho_y(i, j);
}

// ----- viscosité -----
@compute @workgroup_size(64)
fn k_visc_diag(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    let i = c % P.nx; let j = c / P.nx;
    S[O_w2 + (c)] = 2.0 * S[O_mu_cell + (c)] * (S[O_U + (uidx(i + 1u, j))] - S[O_U + (uidx(i, j))]) / P.dx;
    S[O_w3 + (c)] = 2.0 * S[O_mu_cell + (c)] * (S[O_V + (vidx(i, j + 1u))] - S[O_V + (vidx(i, j))]) / P.dx;
}
@compute @workgroup_size(64)
fn k_visc_xy(@builtin(global_invocation_id) g: vec3<u32>) {
    let q = g.x;
    let W = P.nx + 1u;
    if (q >= W * (P.ny + 1u)) { return; }
    let i = q % W; let j = q / W;
    var t = 0.0;
    if (i >= 1u && i < P.nx && j >= 1u && j < P.ny) {
        let c0 = cidx(i - 1u, j - 1u); let c1 = cidx(i, j - 1u); let c2 = cidx(i - 1u, j); let c3 = cidx(i, j);
        if (!(is_solid(c0) || is_solid(c1) || is_solid(c2) || is_solid(c3))) {
            let mu = 0.25 * (S[O_mu_cell + (c0)] + S[O_mu_cell + (c1)] + S[O_mu_cell + (c2)] + S[O_mu_cell + (c3)]);
            t = mu * ((S[O_U + (uidx(i, j))] - S[O_U + (uidx(i, j - 1u))]) / P.dx + (S[O_V + (vidx(i, j))] - S[O_V + (vidx(i - 1u, j))]) / P.dx);
        }
    }
    S[O_w1 + (q)] = t;
}
@compute @workgroup_size(64)
fn k_visc_u(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f >= (P.nx + 1u) * P.ny) { return; }
    let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
    if (i == 0u || i >= P.nx) { return; }
    let a = cidx(i - 1u, j); let b = cidx(i, j);
    if (is_solid(a) || is_solid(b)) { return; }
    let W = P.nx + 1u;
    let fx = (S[O_w2 + (b)] - S[O_w2 + (a)]) / P.dx + (S[O_w1 + ((j + 1u) * W + i)] - S[O_w1 + (j * W + i)]) / P.dx;
    S[O_Fx + (f)] += P.dt * fx / frho_x(i, j);
}
@compute @workgroup_size(64)
fn k_visc_v(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f >= P.nx * (P.ny + 1u)) { return; }
    let i = f % P.nx; let j = f / P.nx;
    if (j == 0u || j >= P.ny) { return; }
    let a = cidx(i, j - 1u); let b = cidx(i, j);
    if (is_solid(a) || is_solid(b)) { return; }
    let W = P.nx + 1u;
    let fy = (S[O_w3 + (b)] - S[O_w3 + (a)]) / P.dx + (S[O_w1 + (j * W + i + 1u)] - S[O_w1 + (j * W + i)]) / P.dx;
    S[O_Fy + (f)] += P.dt * fy / frho_y(i, j);
}

// ===== quantité de mouvement conservative (miroir de advect_momentum) =====
fn ux_(i: i32, j: i32) -> f32 {
    return S[O_U + (u32(clamp(j, 0, i32(P.ny) - 1)) * (P.nx + 1u) + u32(clamp(i, 0, i32(P.nx))))];
}
fn vy_(i: i32, j: i32) -> f32 {
    return S[O_V + (u32(clamp(j, 0, i32(P.ny))) * P.nx + u32(clamp(i, 0, i32(P.nx) - 1)))];
}
fn minmod(a: f32, b: f32) -> f32 {
    if (a * b <= 0.0) { return 0.0; }
    if (abs(a) < abs(b)) { return a; }
    return b;
}
fn mnew(c: u32) -> f32 {
    var s = 0.0;
    for (var k = 0u; k < P.np; k++) { s += S[O_mass + (k * ncell() + c)]; }
    return s;
}
fn carry_ux(f: f32, lo: i32, j: i32) -> f32 {
    if (f >= 0.0) { return ux_(lo, j) + 0.5 * minmod(ux_(lo, j) - ux_(lo - 1, j), ux_(lo + 1, j) - ux_(lo, j)); }
    return ux_(lo + 1, j) - 0.5 * minmod(ux_(lo + 1, j) - ux_(lo, j), ux_(lo + 2, j) - ux_(lo + 1, j));
}
fn carry_uy(f: f32, i: i32, lo: i32) -> f32 {
    if (f >= 0.0) { return ux_(i, lo) + 0.5 * minmod(ux_(i, lo) - ux_(i, lo - 1), ux_(i, lo + 1) - ux_(i, lo)); }
    return ux_(i, lo + 1) - 0.5 * minmod(ux_(i, lo + 1) - ux_(i, lo), ux_(i, lo + 2) - ux_(i, lo + 1));
}
fn carry_vy(f: f32, i: i32, lo: i32) -> f32 {
    if (f >= 0.0) { return vy_(i, lo) + 0.5 * minmod(vy_(i, lo) - vy_(i, lo - 1), vy_(i, lo + 1) - vy_(i, lo)); }
    return vy_(i, lo + 1) - 0.5 * minmod(vy_(i, lo + 1) - vy_(i, lo), vy_(i, lo + 2) - vy_(i, lo + 1));
}
fn carry_vx(f: f32, lo: i32, j: i32) -> f32 {
    if (f >= 0.0) { return vy_(lo, j) + 0.5 * minmod(vy_(lo, j) - vy_(lo - 1, j), vy_(lo + 1, j) - vy_(lo, j)); }
    return vy_(lo + 1, j) - 0.5 * minmod(vy_(lo + 1, j) - vy_(lo, j), vy_(lo + 2, j) - vy_(lo + 1, j));
}

@compute @workgroup_size(64)
fn k_mom_u(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f >= (P.nx + 1u) * P.ny) { return; }
    let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
    S[O_Us + (f)] = S[O_U + (f)];
    if (i == 0u || i >= P.nx) { return; }
    let a = cidx(i - 1u, j); let b = cidx(i, j);
    if (is_solid(a) || is_solid(b)) { return; }
    let ii = i32(i); let jj = i32(j);
    let W = P.nx + 1u;
    let rho_old = 0.5 * (S[O_mtot_old + (a)] + S[O_mtot_old + (b)]);
    let rho_new = 0.5 * (mnew(a) + mnew(b));
    let fr = 0.5 * (S[O_mfx + (j * W + i)] + S[O_mfx + (j * W + i + 1u)]);
    let fl = 0.5 * (S[O_mfx + (j * W + i - 1u)] + S[O_mfx + (j * W + i)]);
    let fb = 0.5 * (S[O_mfy + (j * P.nx + i - 1u)] + S[O_mfy + (j * P.nx + i)]);
    let ft = 0.5 * (S[O_mfy + ((j + 1u) * P.nx + i - 1u)] + S[O_mfy + ((j + 1u) * P.nx + i)]);
    let cr = carry_ux(fr, ii, jj); let cl = carry_ux(fl, ii - 1, jj);
    var ct = ux_(ii, jj); var cb = ct; var ft_ = 0.0; var fb_ = 0.0;
    if (j + 1u < P.ny) { ct = carry_uy(ft, ii, jj); ft_ = ft; }
    if (j > 0u) { cb = carry_uy(fb, ii, jj - 1); fb_ = fb; }
    let q = rho_old * S[O_U + (f)] - (fr * cr - fl * cl) - (ft_ * ct - fb_ * cb);
    let lo = min(min(min(min(ux_(ii, jj), ux_(ii - 1, jj)), ux_(ii + 1, jj)), ux_(ii, jj - 1)), ux_(ii, jj + 1));
    let hi = max(max(max(max(ux_(ii, jj), ux_(ii - 1, jj)), ux_(ii + 1, jj)), ux_(ii, jj - 1)), ux_(ii, jj + 1));
    if (rho_new > 1e-6) { S[O_Us + (f)] = clamp(q / rho_new, lo, hi); }
}

@compute @workgroup_size(64)
fn k_mom_v(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f >= P.nx * (P.ny + 1u)) { return; }
    let i = f % P.nx; let j = f / P.nx;
    S[O_Vs + (f)] = S[O_V + (f)];
    if (j == 0u || j >= P.ny) { return; }
    let a = cidx(i, j - 1u); let b = cidx(i, j);
    if (is_solid(a) || is_solid(b)) { return; }
    let ii = i32(i); let jj = i32(j);
    let W = P.nx + 1u;
    let rho_old = 0.5 * (S[O_mtot_old + (a)] + S[O_mtot_old + (b)]);
    let rho_new = 0.5 * (mnew(a) + mnew(b));
    let ft = 0.5 * (S[O_mfy + (j * P.nx + i)] + S[O_mfy + ((j + 1u) * P.nx + i)]);
    let fb = 0.5 * (S[O_mfy + ((j - 1u) * P.nx + i)] + S[O_mfy + (j * P.nx + i)]);
    let fr = 0.5 * (S[O_mfx + (j * W + i + 1u)] + S[O_mfx + ((j - 1u) * W + i + 1u)]);
    let fl = 0.5 * (S[O_mfx + (j * W + i)] + S[O_mfx + ((j - 1u) * W + i)]);
    let ct = carry_vy(ft, ii, jj); let cb = carry_vy(fb, ii, jj - 1);
    var cr = vy_(ii, jj); var cl = cr; var fr_ = 0.0; var fl_ = 0.0;
    if (i + 1u < P.nx) { cr = carry_vx(fr, ii, jj); fr_ = fr; }
    if (i > 0u) { cl = carry_vx(fl, ii - 1, jj); fl_ = fl; }
    let q = rho_old * S[O_V + (f)] - (ft * ct - fb * cb) - (fr_ * cr - fl_ * cl);
    let lo = min(min(min(min(vy_(ii, jj), vy_(ii, jj - 1)), vy_(ii, jj + 1)), vy_(ii - 1, jj)), vy_(ii + 1, jj));
    let hi = max(max(max(max(vy_(ii, jj), vy_(ii, jj - 1)), vy_(ii, jj + 1)), vy_(ii - 1, jj)), vy_(ii + 1, jj));
    if (rho_new > 1e-6) { S[O_Vs + (f)] = clamp(q / rho_new, lo, hi); }
}

@compute @workgroup_size(64)
fn k_add_forces(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < (P.nx + 1u) * P.ny) { S[O_Us + (f)] += S[O_Fx + (f)]; }
    if (f < P.nx * (P.ny + 1u)) { S[O_Vs + (f)] += S[O_Fy + (f)]; }
}

// ===== Conduction, changement de phase, travail de pression =====

fn psat_abs(t_in: f32) -> f32 {
    let t = clamp(t_in, 200.0, 640.0);
    let a = (P.pc_lref - P.pc_dcp * P.pc_tref) / P.pc_rv;
    return P.pc_pref * exp(a * (1.0 / P.pc_tref - 1.0 / t) + P.pc_dcp / P.pc_rv * log(t / P.pc_tref));
}
fn tsat_of(p_abs: f32) -> f32 {
    var lo = 200.0; var hi = 640.0;
    for (var it = 0; it < 40; it++) {
        let mid = 0.5 * (lo + hi);
        if (psat_abs(mid) < p_abs) { lo = mid; } else { hi = mid; }
    }
    return 0.5 * (lo + hi);
}
fn latent_of(t: f32) -> f32 { return P.pc_lref + P.pc_dcp * (t - P.pc_tref); }

// volume du mélange, κ_eff (entropie constante) et γ_T = dT/dp adiabatique, pour (pression relative pr, T)
fn closure(c: u32, pr: f32, t: f32) -> vec3<f32> {
    let n = ncell();
    var v = 0.0; var kap = 0.0; var vb = 0.0; var mcp = 0.0;
    for (var k = 0u; k < P.np; k++) {
        let m = S[O_mass + (k * n + c)];
        if (m <= 0.0) { continue; }
        let vk = m / rho_k(k, pr, t);
        v += vk; kap += vk * kappa_k(k, pr); vb += vk * beta_k(k, t); mcp += m * P.ph[k].a.y;
    }
    mcp = max(mcp, 1e-30);
    return vec3<f32>(v, kap - t * vb * vb / mcp, t * vb / mcp);
}

@compute @workgroup_size(64)
fn k_conduct_step(@builtin(global_invocation_id) g: vec3<u32>) {
    // un sous-pas explicite : w1 = ε nouveau (gather : chaque cellule lit ses 4 voisins)
    let c = g.x;
    if (c >= ncell()) { return; }
    let i = c % P.nx; let j = c / P.nx;
    let ds = P.dt / f32(max(P.nsub, 1u));
    let cap_c = max(heat_cap(c), 1e-30);
    let tc = P.t0 + S[O_eps + (c)] / cap_c;
    var de = 0.0;
    for (var q = 0u; q < 4u; q++) {
        var ii = i32(i); var jj = i32(j);
        if (q == 0u) { ii += 1; } else if (q == 1u) { ii -= 1; } else if (q == 2u) { jj += 1; } else { jj -= 1; }
        if (ii < 0 || jj < 0 || ii >= i32(P.nx) || jj >= i32(P.ny)) { continue; }
        let b = cidx(u32(ii), u32(jj));
        let tb = P.t0 + S[O_eps + (b)] / max(heat_cap(b), 1e-30);
        let ka = S[O_k_cell + (c)]; let kb = S[O_k_cell + (b)];
        var kl = 0.0; if (ka + kb > 0.0) { kl = 2.0 * ka * kb / (ka + kb); }
        de += kl * (tb - tc) / (P.dx * P.dx);
    }
    S[O_w1 + (c)] = S[O_eps + (c)] + ds * de;
}
@compute @workgroup_size(64)
fn k_conduct_commit(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    S[O_eps + (c)] = S[O_w1 + (c)];
    S[O_temp + (c)] = P.t0 + S[O_eps + (c)] / max(heat_cap(c), 1e-30);
}

@compute @workgroup_size(64)
fn k_phase_change(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n || P.pc_on == 0u || is_solid(c)) { return; }
    let liq = P.pc_liq; let vap = P.pc_vap;
    let ml = S[O_mass + (liq * n + c)]; let mv = S[O_mass + (vap * n + c)];
    if (ml <= 1e-9 && mv <= 1e-9) { return; }
    let t = S[O_temp + (c)]; let pr = S[O_prel + (c)]; let p = pr + P.p0;
    let al = S[O_alpha + (liq * n + c)]; let av = S[O_alpha + (vap * n + c)];
    var ao = 0.0;
    for (var k = 0u; k < P.np; k++) { if (k != liq && k != vap && is_gas(k)) { ao += S[O_alpha + (k * n + c)]; } }
    let tsat = tsat_of(p);
    let cap = heat_cap(c);
    let lat = latent_of(t);
    let rho_v = rho_k(vap, pr, t);
    let rho_l = rho_k(liq, pr, t);
    var dm = 0.0;
    let PI = 3.14159265;
    if (t > tsat && ml > 1e-9) {
        let dm_e = min(cap * (t - tsat) / lat, ml);
        let gas = av + ao;
        let resolved = gas > 0.02 || (al < 0.98 && gas > 0.0 && ao > 0.0);
        if (resolved && al > 1e-6) {
            dm = dm_e;
        } else if (av >= 1e-7 && ao < 1e-9) {
            let rb = pow(av / max(1.0 - av, 1e-9) * 3.0 / (4.0 * PI * P.nuclei), 1.0 / 3.0);
            let ps = psat_abs(t);
            let rate = rho_v * rho_l / S[O_rho_mix + (c)] * av * (1.0 - av) * 3.0 / max(rb, 1e-9) * sqrt(max(2.0 / 3.0 * max(ps - p, 0.0) / rho_l, 0.0));
            var thermal = 0.0; if (ps - p <= 0.0) { thermal = dm_e; }
            dm = min(max(rate * P.dt, thermal), dm_e);
        } else if (av < 1e-7 && ao < 1e-9 && t - tsat > P.nuc_dT) {
            dm = min(P.nuc_alpha * rho_v, dm_e);
        }
    } else if (t < tsat && mv > 1e-9 && al > 1e-6) {
        let dm_e = min(cap * (tsat - t) / lat, mv);
        let gas = av + ao;
        if (gas > 0.02 || ao > 0.0) {
            dm = -dm_e;
        } else if (av > 1e-7) {
            let rb = pow(av / max(1.0 - av, 1e-9) * 3.0 / (4.0 * PI * P.nuclei), 1.0 / 3.0);
            let ps = psat_abs(t);
            let rate = rho_v * rho_l / S[O_rho_mix + (c)] * av * (1.0 - av) * 3.0 / max(rb, 1e-9) * sqrt(max(2.0 / 3.0 * max(p - ps, 0.0) / rho_l, 0.0));
            dm = -min(min(rate * P.dt, dm_e), mv);
        }
    }
    if (dm != 0.0) {
        // couplage implicite local : retour à p = p_sat(T), jamais au-delà
        let ps = psat_abs(t);
        let cl = closure(c, ps - P.p0, t);
        let denom = 1.0 / rho_k(vap, ps - P.p0, t) - 1.0 / rho_k(liq, ps - P.p0, t);
        if (dm > 0.0) { dm = min(dm, max((1.0 - cl.x) / denom, 0.0)); }
        else { dm = -min(-dm, max((cl.x - 1.0) / denom, 0.0)); }
    }
    if (dm != 0.0) {
        S[O_mass + (liq * n + c)] -= dm;
        S[O_mass + (vap * n + c)] += dm;
        S[O_eps + (c)] -= dm * P.pc_h0;
    }
}

@compute @workgroup_size(64)
fn k_pressure_work(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell() || is_solid(c)) { return; }
    let i = c % P.nx; let j = c / P.nx;
    let d = (S[O_U + (uidx(i + 1u, j))] - S[O_U + (uidx(i, j))] + S[O_V + (vidx(i, j + 1u))] - S[O_V + (vidx(i, j))]) / P.dx;
    S[O_eps + (c)] -= P.dt * (S[O_prel + (c)] + P.p0) * d;
}

@compute @workgroup_size(64)
fn k_save_p(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    S[O_p_start + (c)] = S[O_prel + (c)];
    S[O_tst + (c)] = S[O_temp + (c)];
}

// ===== Pression faiblement compressible implicite : Newton + PCG (préconditionneur de Jacobi) =====
// w1 = V (volume du mélange), w2 = κ_eff, w3 = second membre b. Fx,Fy = Ũ,Ṽ (vitesses corrigées par p_k).

@compute @workgroup_size(64)
fn k_closure(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    if (is_solid(c)) { S[O_w1 + (c)] = 1.0; S[O_w2 + (c)] = 1e-30; return; }
    let g0 = closure(c, S[O_p_start + (c)], S[O_tst + (c)]).z;
    let tk = S[O_tst + (c)] + g0 * (S[O_prel + (c)] - S[O_p_start + (c)]);
    let cl = closure(c, S[O_prel + (c)], tk);
    S[O_w1 + (c)] = cl.x;
    S[O_w2 + (c)] = max(cl.y, 1e-30);
}

fn face_u(f: u32, final_: bool) -> f32 {
    let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
    let us = S[O_Us + (f)];
    if (i > 0u && i < P.nx) {
        let a = cidx(i - 1u, j); let b = cidx(i, j);
        if (is_solid(a) || is_solid(b)) { return 0.0; }
        return us - P.dt / frho_x(i, j) * (S[O_prel + (b)] - S[O_prel + (a)]) / P.dx;
    }
    if (i == 0u) {
        let b = cidx(0u, j);
        if (P.bc_l == 1u && !is_solid(b)) { return us - P.dt / S[O_rho_mix + (b)] * (S[O_prel + (b)] - P.pb_l) / (0.5 * P.dx); }
        return 0.0;
    }
    let a = cidx(P.nx - 1u, j);
    if (P.bc_r == 1u && !is_solid(a)) { return us - P.dt / S[O_rho_mix + (a)] * (P.pb_r - S[O_prel + (a)]) / (0.5 * P.dx); }
    return 0.0;
}
fn face_v(f: u32) -> f32 {
    let i = f % P.nx; let j = f / P.nx;
    let vs = S[O_Vs + (f)];
    if (j > 0u && j < P.ny) {
        let a = cidx(i, j - 1u); let b = cidx(i, j);
        if (is_solid(a) || is_solid(b)) { return 0.0; }
        return vs - P.dt / frho_y(i, j) * (S[O_prel + (b)] - S[O_prel + (a)]) / P.dx;
    }
    if (j == 0u) {
        let b = cidx(i, 0u);
        if (P.bc_b == 1u && !is_solid(b)) { return vs - P.dt / S[O_rho_mix + (b)] * (S[O_prel + (b)] - P.pb_b) / (0.5 * P.dx); }
        return 0.0;
    }
    let a = cidx(i, P.ny - 1u);
    if (P.bc_t == 1u && !is_solid(a)) { return vs - P.dt / S[O_rho_mix + (a)] * (P.pb_t - S[O_prel + (a)]) / (0.5 * P.dx); }
    return 0.0;
}
@compute @workgroup_size(64)
fn k_face_tmp(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < (P.nx + 1u) * P.ny) { S[O_Fx + (f)] = face_u(f, false); }
    if (f < P.nx * (P.ny + 1u)) { S[O_Fy + (f)] = face_v(f); }
}
@compute @workgroup_size(64)
fn k_face_final(@builtin(global_invocation_id) g: vec3<u32>) {
    // écrit u^{n+1} dans U, V (U ancien n'est plus nécessaire)
    let f = g.x;
    if (f < (P.nx + 1u) * P.ny) { S[O_U + (f)] = face_u(f, true); }
    if (f < P.nx * (P.ny + 1u)) { S[O_V + (f)] = face_v(f); }
}

@compute @workgroup_size(64)
fn k_rhs(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    if (is_solid(c)) { S[O_w3 + (c)] = 0.0; return; }
    let i = c % P.nx; let j = c / P.nx;
    let dn = (S[O_Fx + (uidx(i + 1u, j))] - S[O_Fx + (uidx(i, j))] + S[O_Fy + (vidx(i, j + 1u))] - S[O_Fy + (vidx(i, j))]) / P.dx;
    let do_ = (S[O_U + (uidx(i + 1u, j))] - S[O_U + (uidx(i, j))] + S[O_V + (vidx(i, j + 1u))] - S[O_V + (vidx(i, j))]) / P.dx;
    S[O_w3 + (c)] = (S[O_w1 + (c)] - 1.0) / P.dt - (dn - do_);
}

@compute @workgroup_size(64)
fn k_stencil(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    S[O_sae + (c)] = 0.0; S[O_san + (c)] = 0.0; S[O_sd + (c)] = 1.0;
    if (is_solid(c)) { return; }
    let i = c % P.nx; let j = c / P.nx;
    let dx2 = P.dx * P.dx;
    var d = S[O_w2 + (c)] / P.dt;
    if (i + 1u < P.nx) { if (!is_solid(c + 1u)) { let t = P.dt / (frho_x(i + 1u, j) * dx2); S[O_sae + (c)] = t; d += t; } }
    else if (P.bc_r == 1u) { d += 2.0 * P.dt / (S[O_rho_mix + (c)] * dx2); }
    if (i > 0u) { if (!is_solid(c - 1u)) { d += P.dt / (frho_x(i, j) * dx2); } }
    else if (P.bc_l == 1u) { d += 2.0 * P.dt / (S[O_rho_mix + (c)] * dx2); }
    if (j + 1u < P.ny) { if (!is_solid(c + P.nx)) { let t = P.dt / (frho_y(i, j + 1u) * dx2); S[O_san + (c)] = t; d += t; } }
    else if (P.bc_t == 1u) { d += 2.0 * P.dt / (S[O_rho_mix + (c)] * dx2); }
    if (j > 0u) { if (!is_solid(c - P.nx)) { d += P.dt / (frho_y(i, j) * dx2); } }
    else if (P.bc_b == 1u) { d += 2.0 * P.dt / (S[O_rho_mix + (c)] * dx2); }
    S[O_sd + (c)] = d;
}

// --- PCG ---
@compute @workgroup_size(64)
fn k_pcg_init(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    S[O_cx + (c)] = 0.0;
    S[O_cr + (c)] = S[O_w3 + (c)];
    S[O_cz + (c)] = S[O_w3 + (c)] / S[O_sd + (c)];
    S[O_cp_ + (c)] = S[O_cz + (c)];
}
@compute @workgroup_size(64)
fn k_apply(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    let i = c % P.nx;
    var s = S[O_sd + (c)] * S[O_cp_ + (c)];
    if (S[O_sae + (c)] != 0.0) { s -= S[O_sae + (c)] * S[O_cp_ + (c + 1u)]; }
    if (i > 0u && S[O_sae + (c - 1u)] != 0.0) { s -= S[O_sae + (c - 1u)] * S[O_cp_ + (c - 1u)]; }
    if (S[O_san + (c)] != 0.0) { s -= S[O_san + (c)] * S[O_cp_ + (c + P.nx)]; }
    if (c >= P.nx && S[O_san + (c - P.nx)] != 0.0) { s -= S[O_san + (c - P.nx)] * S[O_cp_ + (c - P.nx)]; }
    S[O_cap_ + (c)] = s;
}

var<workgroup> sh: array<f32, 64>;
fn reduce_wg(v: f32, lid: u32, wg: u32) {
    sh[lid] = v;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (lid < s) { sh[lid] += sh[lid + s]; }
        workgroupBarrier();
    }
    if (lid == 0u) { S[O_part + (wg)] = sh[0]; }
}
// mode 0: r·z → part ; 1: p·Ap ; 2: r·r ; 3: b·b (w3)
@compute @workgroup_size(64) fn k_dot_rz(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    var v = 0.0; if (g.x < ncell()) { v = S[O_cr + (g.x)] * S[O_cz + (g.x)]; } reduce_wg(v, l.x, w.x);
}
@compute @workgroup_size(64) fn k_dot_pap(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    var v = 0.0; if (g.x < ncell()) { v = S[O_cp_ + (g.x)] * S[O_cap_ + (g.x)]; } reduce_wg(v, l.x, w.x);
}
@compute @workgroup_size(64) fn k_dot_rr(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    var v = 0.0; if (g.x < ncell()) { v = S[O_cr + (g.x)] * S[O_cr + (g.x)]; } reduce_wg(v, l.x, w.x);
}
@compute @workgroup_size(64) fn k_dot_bb(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    var v = 0.0; if (g.x < ncell()) { v = S[O_w3 + (g.x)] * S[O_w3 + (g.x)]; } reduce_wg(v, l.x, w.x);
}
// somme finale des sommes partielles dans S[O_scal + (dst)] (1 seul thread, nb groupes = ceil(n/64))
fn sum_part(dst: u32) {
    var s = 0.0;
    let ng = (ncell() + 63u) / 64u;
    for (var q = 0u; q < ng; q++) { s += S[O_part + (q)]; }
    S[O_scal + (dst)] = s;
}
@compute @workgroup_size(1) fn k_fin_rz0() { sum_part(0u); }
@compute @workgroup_size(1) fn k_fin_pap() { sum_part(1u); S[O_scal + (2)] = select(0.0, S[O_scal + (0)] / S[O_scal + (1)], abs(S[O_scal + (1)]) > 1e-30); }
@compute @workgroup_size(1) fn k_fin_rznew() {
    sum_part(4u);
    S[O_scal + (3)] = select(0.0, S[O_scal + (4)] / S[O_scal + (0)], abs(S[O_scal + (0)]) > 1e-30);
    S[O_scal + (0)] = S[O_scal + (4)];
}
@compute @workgroup_size(1) fn k_fin_rr() { sum_part(6u); }
@compute @workgroup_size(1) fn k_fin_bb() { sum_part(5u); }

@compute @workgroup_size(64)
fn k_cg_update(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    let a = S[O_scal + (2)];
    S[O_cx + (c)] += a * S[O_cp_ + (c)];
    S[O_cr + (c)] -= a * S[O_cap_ + (c)];
    S[O_cz + (c)] = S[O_cr + (c)] / S[O_sd + (c)];
}
@compute @workgroup_size(64)
fn k_cg_dir(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    S[O_cp_ + (c)] = S[O_cz + (c)] + S[O_scal + (3)] * S[O_cp_ + (c)];
}

// pk += δp (garde-fou : un gaz reste à p ≥ 10 Pa absolu)
@compute @workgroup_size(64)
fn k_p_update(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n || is_solid(c)) { return; }
    var pr = S[O_prel + (c)] + S[O_cx + (c)];
    var gas = 0.0;
    for (var k = 0u; k < P.np; k++) { if (is_gas(k)) { gas += S[O_alpha + (k * n + c)]; } }
    if (gas > 1e-6 && pr + P.p0 < 10.0) { pr = 10.0 - P.p0; }
    S[O_prel + (c)] = pr;
}

