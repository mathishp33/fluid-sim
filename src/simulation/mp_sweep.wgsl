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
    if (is_solid(c)) { return w1[k * ncell() + cidx(ci_, cj_)]; }
    return w1[k * ncell() + c];
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
    if (i < ncell() * P.np) { w2[i] = alpha[i]; }
}

@compute @workgroup_size(64)
fn k_an(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n) { return; }
    var s = 0.0;
    for (var k = 0u; k < P.np; k++) { s += clamp(w2[k * n + c], 0.0, 1.0); }
    s = max(s, 1e-30);
    for (var k = 0u; k < P.np; k++) { w1[k * n + c] = clamp(w2[k * n + c], 0.0, 1.0) / s; }
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
        r.uf = U[uidx(r.fi, r.fj)];
        if (r.fi > 0u) { r.lo = i32(cidx(r.fi - 1u, r.fj)); }
        if (r.fi < P.nx) { r.hi = i32(cidx(r.fi, r.fj)); }
    } else {
        r.uf = V[vidx(r.fi, r.fj)];
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
    for (var k = 0u; k < np; k++) { fl_phase[f * np + k] = 0.0; }
    if (!info.valid) { return; }
    let xi = min(abs(info.uf) * P.dt / P.dx, 1.0);
    var d = info.lo; if (info.uf < 0.0) { d = info.hi; }
    if (d < 0) { fl_phase[f * np + P.ambient] = xi; return; }
    let dc = u32(d);
    var s0 = 0.0; var s1 = xi; if (info.uf > 0.0) { s0 = 1.0 - xi; s1 = 1.0; }
    // phase pure ?
    for (var k = 0u; k < np; k++) {
        if (w1[k * n + dc] >= 1.0 - 1e-6) { fl_phase[f * np + k] = xi; return; }
    }
    let ci_ = dc % P.nx; let cj_ = dc / P.nx;
    var rem = xi;
    var spare: array<f32, 4>;
    for (var pos = 0u; pos < np; pos++) {
        let k = phase_order(pos);
        let a = w1[k * n + dc];
        var want = 0.0;
        if (pos == np - 1u) { want = rem; }
        else if (a > 1e-6 && a < 1.0 - 1e-6) {
            let gr = youngs(k, ci_, cj_);
            let pl = plic_new(a, gr.x, gr.y);
            if (dir == 0u) { want = plic_area(pl, s0, s1, 0.0, 1.0); } else { want = plic_area(pl, 0.0, 1.0, s0, s1); }
        }
        let fk = max(min(min(want, rem), a), 0.0);
        fl_phase[f * np + k] = fk;
        rem -= fk;
        spare[k] = a - fk;
    }
    if (rem > 1e-7) {
        var tot = 0.0;
        for (var k = 0u; k < np; k++) { tot += max(spare[k], 0.0); }
        if (tot > 0.0) { for (var k = 0u; k < np; k++) { fl_phase[f * np + k] += rem * max(spare[k], 0.0) / tot; } }
    }
}
@compute @workgroup_size(64) fn k_sweep_flux_x(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < (P.nx + 1u) * P.ny) { sweep_flux(g.x, 0u); } }
@compute @workgroup_size(64) fn k_sweep_flux_y(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < P.nx * (P.ny + 1u)) { sweep_flux(g.x, 1u); } }

// facteur de limitation : sortie totale de chaque phase ≤ 0.999 α_travail
fn sweep_scale(c: u32, dir: u32) {
    let np = P.np; let n = ncell();
    let i = c % P.nx; let j = c / P.nx;
    var f_lo = uidx(i, j); var f_hi = uidx(i + 1u, j);
    var u_lo = U[f_lo]; var u_hi = U[f_hi];
    if (dir == 1u) { f_lo = vidx(i, j); f_hi = vidx(i, j + 1u); u_lo = V[f_lo]; u_hi = V[f_hi]; }
    for (var k = 0u; k < np; k++) {
        var out = 0.0;
        if (u_lo < 0.0) { out += fl_phase[f_lo * np + k]; }
        if (u_hi > 0.0) { out += fl_phase[f_hi * np + k]; }
        let cap = max(w2[k * n + c], 0.0) * 0.999;
        var s = 1.0;
        if (out > cap && out > 0.0) { s = cap / out; }
        w3[k * n + c] = s;
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
        var fk = fl_phase[f * np + k];
        var mf = 0.0;
        if (info.valid && fk != 0.0) {
            var rho_d = 0.0; var t_d = P.t_amb;
            if (d < 0) {
                var pr = 0.0; if (r >= 0) { pr = prel[u32(r)]; }
                rho_d = rho_k(k, pr, P.t_amb);
            } else {
                let dc = u32(d);
                fk = fk * w3[k * n + dc];
                t_d = temp[dc];
                let aw = w2[k * n + dc];
                let rho_eos = rho_k(k, prel[dc], t_d);
                rho_d = rho_eos;
                if (aw > 1e-5) { let q = mass[k * n + dc] / aw; if (q > 0.0) { rho_d = q; } }
            }
            fl_phase[f * np + k] = fk;
            mf = fk * rho_d;
            en += mf * P.ph[k].a.z * (t_d - P.t0);
            mt += mf;
        }
        fl_mass[f * np + k] = mf;
    }
    fl_en[f] = en;
    if (info.valid) {
        let sg = sign(info.uf);
        if (dir == 0u) { mfx[uidx(info.fi, info.fj)] += sg * mt; } else { mfy[vidx(info.fi, info.fj)] += sg * mt; }
    }
}
@compute @workgroup_size(64) fn k_sweep_apply_x(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < (P.nx + 1u) * P.ny) { sweep_apply(g.x, 0u); } }
@compute @workgroup_size(64) fn k_sweep_apply_y(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < P.nx * (P.ny + 1u)) { sweep_apply(g.x, 1u); } }

fn sweep_update(c: u32, dir: u32) {
    if (is_solid(c)) { return; }
    let np = P.np; let n = ncell();
    let i = c % P.nx; let j = c / P.nx;
    var f_lo = uidx(i, j); var f_hi = uidx(i + 1u, j);
    var u_lo = U[f_lo]; var u_hi = U[f_hi];
    if (dir == 1u) { f_lo = vidx(i, j); f_hi = vidx(i, j + 1u); u_lo = V[f_lo]; u_hi = V[f_hi]; }
    var de = 0.0;
    for (var k = 0u; k < np; k++) {
        var dv = 0.0; var dm = 0.0;
        // face basse : U>0 entre dans c ; U<0 sort de c
        if (u_lo > 0.0) { dv += fl_phase[f_lo * np + k]; dm += fl_mass[f_lo * np + k]; }
        if (u_lo < 0.0) { dv -= fl_phase[f_lo * np + k]; dm -= fl_mass[f_lo * np + k]; }
        if (u_hi < 0.0) { dv += fl_phase[f_hi * np + k]; dm += fl_mass[f_hi * np + k]; }
        if (u_hi > 0.0) { dv -= fl_phase[f_hi * np + k]; dm -= fl_mass[f_hi * np + k]; }
        w2[k * n + c] += dv;
        mass[k * n + c] += dm;
    }
    if (u_lo > 0.0) { de += fl_en[f_lo]; }
    if (u_lo < 0.0) { de -= fl_en[f_lo]; }
    if (u_hi < 0.0) { de += fl_en[f_hi]; }
    if (u_hi > 0.0) { de -= fl_en[f_hi]; }
    eps[c] += de;
    temp[c] = P.t0 + eps[c] / max(heat_cap(c), 1e-30);
}
@compute @workgroup_size(64) fn k_sweep_update_x(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < ncell()) { sweep_update(g.x, 0u); } }
@compute @workgroup_size(64) fn k_sweep_update_y(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < ncell()) { sweep_update(g.x, 1u); } }
