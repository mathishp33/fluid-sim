// ===== Forces : gravité, tension de surface (CSF + fonctions de hauteur), viscosité ; QDM conservative =====
// Étage forces : w1 = marqueur liquide ℓ, w2 = courbure (1e30 = indéfinie), w3 = ℓ lissé, w4 = σ(cellule).
// Étage viscosité : w2 = τxx, w3 = τyy, w1 = τxy (nœuds (nx+1)(ny+1)).

fn mk(i: i32, j: i32, ci_: u32, cj_: u32) -> f32 {
    let ii = u32(clamp(i, 0, i32(P.nx) - 1)); let jj = u32(clamp(j, 0, i32(P.ny) - 1));
    let c = cidx(ii, jj);
    if (is_solid(c)) { return w1[cidx(ci_, cj_)]; }
    return w1[c];
}
fn smk(i: i32, j: i32, ci_: u32, cj_: u32) -> f32 {
    let ii = u32(clamp(i, 0, i32(P.nx) - 1)); let jj = u32(clamp(j, 0, i32(P.ny) - 1));
    let c = cidx(ii, jj);
    if (is_solid(c)) { return w3[cidx(ci_, cj_)]; }
    return w3[c];
}

@compute @workgroup_size(64)
fn k_force_init(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < (P.nx + 1u) * P.ny) { Fx[f] = P.dt * P.gx; }
    if (f < P.nx * (P.ny + 1u)) { Fy[f] = P.dt * P.gy; }
}

@compute @workgroup_size(64)
fn k_marker(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n) { return; }
    var l = 0.0; var sg = 0.0;
    if (!is_solid(c)) {
        for (var k = 0u; k < P.np; k++) { if (!is_gas(k)) { l += alpha[k * n + c]; sg += alpha[k * n + c] * P.ph[k].d.w; } }
    }
    w1[c] = clamp(l, 0.0, 1.0);
    if (l < 1e-12) { w4[c] = 0.0; } else { w4[c] = sg / l; }
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
    w3[c] = s / 16.0;
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
    let a = w1[c];
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
    w2[c] = kap;
}

fn st_face(a: u32, b: u32) -> f32 {
    let la = w1[a]; let lb = w1[b];
    let wa = la * (1.0 - la); let wb = lb * (1.0 - lb);
    let w = wa + wb;
    if (w < 1e-9) { return 0.0; }
    var ka = w2[a]; if (ka > 1e29) { ka = 0.0; }
    var kb = w2[b]; if (kb > 1e29) { kb = 0.0; }
    let kf = (ka * wa + kb * wb) / w;
    let sg = (w4[a] * wa + w4[b] * wb) / w;
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
    Fx[f] += P.dt * st_face(a, b) / frho_x(i, j);
}
@compute @workgroup_size(64)
fn k_force_st_v(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f >= P.nx * (P.ny + 1u)) { return; }
    let i = f % P.nx; let j = f / P.nx;
    if (j == 0u || j >= P.ny) { return; }
    let a = cidx(i, j - 1u); let b = cidx(i, j);
    if (is_solid(a) || is_solid(b)) { return; }
    Fy[f] += P.dt * st_face(a, b) / frho_y(i, j);
}

// ----- viscosité -----
@compute @workgroup_size(64)
fn k_visc_diag(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    let i = c % P.nx; let j = c / P.nx;
    w2[c] = 2.0 * mu_cell[c] * (U[uidx(i + 1u, j)] - U[uidx(i, j)]) / P.dx;
    w3[c] = 2.0 * mu_cell[c] * (V[vidx(i, j + 1u)] - V[vidx(i, j)]) / P.dx;
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
            let mu = 0.25 * (mu_cell[c0] + mu_cell[c1] + mu_cell[c2] + mu_cell[c3]);
            t = mu * ((U[uidx(i, j)] - U[uidx(i, j - 1u)]) / P.dx + (V[vidx(i, j)] - V[vidx(i - 1u, j)]) / P.dx);
        }
    }
    w1[q] = t;
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
    let fx = (w2[b] - w2[a]) / P.dx + (w1[(j + 1u) * W + i] - w1[j * W + i]) / P.dx;
    Fx[f] += P.dt * fx / frho_x(i, j);
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
    let fy = (w3[b] - w3[a]) / P.dx + (w1[j * W + i + 1u] - w1[j * W + i]) / P.dx;
    Fy[f] += P.dt * fy / frho_y(i, j);
}

// ===== quantité de mouvement conservative (miroir de advect_momentum) =====
fn ux_(i: i32, j: i32) -> f32 {
    return U[u32(clamp(j, 0, i32(P.ny) - 1)) * (P.nx + 1u) + u32(clamp(i, 0, i32(P.nx)))];
}
fn vy_(i: i32, j: i32) -> f32 {
    return V[u32(clamp(j, 0, i32(P.ny))) * P.nx + u32(clamp(i, 0, i32(P.nx) - 1))];
}
fn minmod(a: f32, b: f32) -> f32 {
    if (a * b <= 0.0) { return 0.0; }
    if (abs(a) < abs(b)) { return a; }
    return b;
}
fn mnew(c: u32) -> f32 {
    var s = 0.0;
    for (var k = 0u; k < P.np; k++) { s += mass[k * ncell() + c]; }
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
    Us[f] = U[f];
    if (i == 0u || i >= P.nx) { return; }
    let a = cidx(i - 1u, j); let b = cidx(i, j);
    if (is_solid(a) || is_solid(b)) { return; }
    let ii = i32(i); let jj = i32(j);
    let W = P.nx + 1u;
    let rho_old = 0.5 * (mtot_old[a] + mtot_old[b]);
    let rho_new = 0.5 * (mnew(a) + mnew(b));
    let fr = 0.5 * (mfx[j * W + i] + mfx[j * W + i + 1u]);
    let fl = 0.5 * (mfx[j * W + i - 1u] + mfx[j * W + i]);
    let fb = 0.5 * (mfy[j * P.nx + i - 1u] + mfy[j * P.nx + i]);
    let ft = 0.5 * (mfy[(j + 1u) * P.nx + i - 1u] + mfy[(j + 1u) * P.nx + i]);
    let cr = carry_ux(fr, ii, jj); let cl = carry_ux(fl, ii - 1, jj);
    var ct = ux_(ii, jj); var cb = ct; var ft_ = 0.0; var fb_ = 0.0;
    if (j + 1u < P.ny) { ct = carry_uy(ft, ii, jj); ft_ = ft; }
    if (j > 0u) { cb = carry_uy(fb, ii, jj - 1); fb_ = fb; }
    let q = rho_old * U[f] - (fr * cr - fl * cl) - (ft_ * ct - fb_ * cb);
    let lo = min(min(min(min(ux_(ii, jj), ux_(ii - 1, jj)), ux_(ii + 1, jj)), ux_(ii, jj - 1)), ux_(ii, jj + 1));
    let hi = max(max(max(max(ux_(ii, jj), ux_(ii - 1, jj)), ux_(ii + 1, jj)), ux_(ii, jj - 1)), ux_(ii, jj + 1));
    if (rho_new > 1e-6) { Us[f] = clamp(q / rho_new, lo, hi); }
}

@compute @workgroup_size(64)
fn k_mom_v(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f >= P.nx * (P.ny + 1u)) { return; }
    let i = f % P.nx; let j = f / P.nx;
    Vs[f] = V[f];
    if (j == 0u || j >= P.ny) { return; }
    let a = cidx(i, j - 1u); let b = cidx(i, j);
    if (is_solid(a) || is_solid(b)) { return; }
    let ii = i32(i); let jj = i32(j);
    let W = P.nx + 1u;
    let rho_old = 0.5 * (mtot_old[a] + mtot_old[b]);
    let rho_new = 0.5 * (mnew(a) + mnew(b));
    let ft = 0.5 * (mfy[j * P.nx + i] + mfy[(j + 1u) * P.nx + i]);
    let fb = 0.5 * (mfy[(j - 1u) * P.nx + i] + mfy[j * P.nx + i]);
    let fr = 0.5 * (mfx[j * W + i + 1u] + mfx[(j - 1u) * W + i + 1u]);
    let fl = 0.5 * (mfx[j * W + i] + mfx[(j - 1u) * W + i]);
    let ct = carry_vy(ft, ii, jj); let cb = carry_vy(fb, ii, jj - 1);
    var cr = vy_(ii, jj); var cl = cr; var fr_ = 0.0; var fl_ = 0.0;
    if (i + 1u < P.nx) { cr = carry_vx(fr, ii, jj); fr_ = fr; }
    if (i > 0u) { cl = carry_vx(fl, ii - 1, jj); fl_ = fl; }
    let q = rho_old * V[f] - (ft * ct - fb * cb) - (fr_ * cr - fl_ * cl);
    let lo = min(min(min(min(vy_(ii, jj), vy_(ii, jj - 1)), vy_(ii, jj + 1)), vy_(ii - 1, jj)), vy_(ii + 1, jj));
    let hi = max(max(max(max(vy_(ii, jj), vy_(ii, jj - 1)), vy_(ii, jj + 1)), vy_(ii - 1, jj)), vy_(ii + 1, jj));
    if (rho_new > 1e-6) { Vs[f] = clamp(q / rho_new, lo, hi); }
}

@compute @workgroup_size(64)
fn k_add_forces(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < (P.nx + 1u) * P.ny) { Us[f] += Fx[f]; }
    if (f < P.nx * (P.ny + 1u)) { Vs[f] += Fy[f]; }
}
