// ===== Pression faiblement compressible implicite : Newton + PCG (préconditionneur de Jacobi) =====
// w1 = V (volume du mélange), w2 = κ_eff, w3 = second membre b. Fx,Fy = Ũ,Ṽ (vitesses corrigées par p_k).
@group(1) @binding(0) var<storage, read_write> sd: array<f32>;   // diagonale
@group(1) @binding(1) var<storage, read_write> sae: array<f32>;  // couplage est (c ↔ c+1)
@group(1) @binding(2) var<storage, read_write> san: array<f32>;  // couplage nord (c ↔ c+nx)
@group(1) @binding(3) var<storage, read_write> cx: array<f32>;   // x = δp
@group(1) @binding(4) var<storage, read_write> cr: array<f32>;   // résidu
@group(1) @binding(5) var<storage, read_write> cz: array<f32>;   // préconditionné
@group(1) @binding(6) var<storage, read_write> cp_: array<f32>;  // direction
@group(1) @binding(7) var<storage, read_write> cap_: array<f32>; // A p
@group(1) @binding(8) var<storage, read_write> scal: array<f32>; // 0 rz, 1 pap, 2 alpha, 3 beta, 4 rz_new, 5 |b|², 6 |r|², 7 dmax
@group(1) @binding(9) var<storage, read_write> part: array<f32>; // sommes partielles par groupe

@compute @workgroup_size(64)
fn k_closure(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    if (is_solid(c)) { w1[c] = 1.0; w2[c] = 1e-30; return; }
    let g0 = closure(c, p_start[c], tst[c]).z;
    let tk = tst[c] + g0 * (prel[c] - p_start[c]);
    let cl = closure(c, prel[c], tk);
    w1[c] = cl.x;
    w2[c] = max(cl.y, 1e-30);
}

fn face_u(f: u32, final_: bool) -> f32 {
    let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
    let us = Us[f];
    if (i > 0u && i < P.nx) {
        let a = cidx(i - 1u, j); let b = cidx(i, j);
        if (is_solid(a) || is_solid(b)) { return 0.0; }
        return us - P.dt / frho_x(i, j) * (prel[b] - prel[a]) / P.dx;
    }
    if (i == 0u) {
        let b = cidx(0u, j);
        if (P.bc_l == 1u && !is_solid(b)) { return us - P.dt / rho_mix[b] * (prel[b] - P.pb_l) / (0.5 * P.dx); }
        return 0.0;
    }
    let a = cidx(P.nx - 1u, j);
    if (P.bc_r == 1u && !is_solid(a)) { return us - P.dt / rho_mix[a] * (P.pb_r - prel[a]) / (0.5 * P.dx); }
    return 0.0;
}
fn face_v(f: u32) -> f32 {
    let i = f % P.nx; let j = f / P.nx;
    let vs = Vs[f];
    if (j > 0u && j < P.ny) {
        let a = cidx(i, j - 1u); let b = cidx(i, j);
        if (is_solid(a) || is_solid(b)) { return 0.0; }
        return vs - P.dt / frho_y(i, j) * (prel[b] - prel[a]) / P.dx;
    }
    if (j == 0u) {
        let b = cidx(i, 0u);
        if (P.bc_b == 1u && !is_solid(b)) { return vs - P.dt / rho_mix[b] * (prel[b] - P.pb_b) / (0.5 * P.dx); }
        return 0.0;
    }
    let a = cidx(i, P.ny - 1u);
    if (P.bc_t == 1u && !is_solid(a)) { return vs - P.dt / rho_mix[a] * (P.pb_t - prel[a]) / (0.5 * P.dx); }
    return 0.0;
}
@compute @workgroup_size(64)
fn k_face_tmp(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < (P.nx + 1u) * P.ny) { Fx[f] = face_u(f, false); }
    if (f < P.nx * (P.ny + 1u)) { Fy[f] = face_v(f); }
}
@compute @workgroup_size(64)
fn k_face_final(@builtin(global_invocation_id) g: vec3<u32>) {
    // écrit u^{n+1} dans U, V (U ancien n'est plus nécessaire)
    let f = g.x;
    if (f < (P.nx + 1u) * P.ny) { U[f] = face_u(f, true); }
    if (f < P.nx * (P.ny + 1u)) { V[f] = face_v(f); }
}

@compute @workgroup_size(64)
fn k_rhs(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    if (is_solid(c)) { w3[c] = 0.0; return; }
    let i = c % P.nx; let j = c / P.nx;
    let dn = (Fx[uidx(i + 1u, j)] - Fx[uidx(i, j)] + Fy[vidx(i, j + 1u)] - Fy[vidx(i, j)]) / P.dx;
    let do_ = (U[uidx(i + 1u, j)] - U[uidx(i, j)] + V[vidx(i, j + 1u)] - V[vidx(i, j)]) / P.dx;
    w3[c] = (w1[c] - 1.0) / P.dt - (dn - do_);
}

@compute @workgroup_size(64)
fn k_stencil(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    sae[c] = 0.0; san[c] = 0.0; sd[c] = 1.0;
    if (is_solid(c)) { return; }
    let i = c % P.nx; let j = c / P.nx;
    let dx2 = P.dx * P.dx;
    var d = w2[c] / P.dt;
    if (i + 1u < P.nx) { if (!is_solid(c + 1u)) { let t = P.dt / (frho_x(i + 1u, j) * dx2); sae[c] = t; d += t; } }
    else if (P.bc_r == 1u) { d += 2.0 * P.dt / (rho_mix[c] * dx2); }
    if (i > 0u) { if (!is_solid(c - 1u)) { d += P.dt / (frho_x(i, j) * dx2); } }
    else if (P.bc_l == 1u) { d += 2.0 * P.dt / (rho_mix[c] * dx2); }
    if (j + 1u < P.ny) { if (!is_solid(c + P.nx)) { let t = P.dt / (frho_y(i, j + 1u) * dx2); san[c] = t; d += t; } }
    else if (P.bc_t == 1u) { d += 2.0 * P.dt / (rho_mix[c] * dx2); }
    if (j > 0u) { if (!is_solid(c - P.nx)) { d += P.dt / (frho_y(i, j) * dx2); } }
    else if (P.bc_b == 1u) { d += 2.0 * P.dt / (rho_mix[c] * dx2); }
    sd[c] = d;
}

// --- PCG ---
@compute @workgroup_size(64)
fn k_pcg_init(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    cx[c] = 0.0;
    cr[c] = w3[c];
    cz[c] = w3[c] / sd[c];
    cp_[c] = cz[c];
}
@compute @workgroup_size(64)
fn k_apply(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    let i = c % P.nx;
    var s = sd[c] * cp_[c];
    if (sae[c] != 0.0) { s -= sae[c] * cp_[c + 1u]; }
    if (i > 0u && sae[c - 1u] != 0.0) { s -= sae[c - 1u] * cp_[c - 1u]; }
    if (san[c] != 0.0) { s -= san[c] * cp_[c + P.nx]; }
    if (c >= P.nx && san[c - P.nx] != 0.0) { s -= san[c - P.nx] * cp_[c - P.nx]; }
    cap_[c] = s;
}

var<workgroup> sh: array<f32, 64>;
fn reduce_wg(v: f32, lid: u32, wg: u32) {
    sh[lid] = v;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (lid < s) { sh[lid] += sh[lid + s]; }
        workgroupBarrier();
    }
    if (lid == 0u) { part[wg] = sh[0]; }
}
// mode 0: r·z → part ; 1: p·Ap ; 2: r·r ; 3: b·b (w3)
@compute @workgroup_size(64) fn k_dot_rz(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    var v = 0.0; if (g.x < ncell()) { v = cr[g.x] * cz[g.x]; } reduce_wg(v, l.x, w.x);
}
@compute @workgroup_size(64) fn k_dot_pap(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    var v = 0.0; if (g.x < ncell()) { v = cp_[g.x] * cap_[g.x]; } reduce_wg(v, l.x, w.x);
}
@compute @workgroup_size(64) fn k_dot_rr(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    var v = 0.0; if (g.x < ncell()) { v = cr[g.x] * cr[g.x]; } reduce_wg(v, l.x, w.x);
}
@compute @workgroup_size(64) fn k_dot_bb(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    var v = 0.0; if (g.x < ncell()) { v = w3[g.x] * w3[g.x]; } reduce_wg(v, l.x, w.x);
}
// somme finale des sommes partielles dans scal[dst] (1 seul thread, nb groupes = ceil(n/64))
fn sum_part(dst: u32) {
    var s = 0.0;
    let ng = (ncell() + 63u) / 64u;
    for (var q = 0u; q < ng; q++) { s += part[q]; }
    scal[dst] = s;
}
@compute @workgroup_size(1) fn k_fin_rz0() { sum_part(0u); }
@compute @workgroup_size(1) fn k_fin_pap() { sum_part(1u); scal[2] = select(0.0, scal[0] / scal[1], abs(scal[1]) > 1e-30); }
@compute @workgroup_size(1) fn k_fin_rznew() {
    sum_part(4u);
    scal[3] = select(0.0, scal[4] / scal[0], abs(scal[0]) > 1e-30);
    scal[0] = scal[4];
}
@compute @workgroup_size(1) fn k_fin_rr() { sum_part(6u); }
@compute @workgroup_size(1) fn k_fin_bb() { sum_part(5u); }

@compute @workgroup_size(64)
fn k_cg_update(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    let a = scal[2];
    cx[c] += a * cp_[c];
    cr[c] -= a * cap_[c];
    cz[c] = cr[c] / sd[c];
}
@compute @workgroup_size(64)
fn k_cg_dir(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    cp_[c] = cz[c] + scal[3] * cp_[c];
}

// pk += δp (garde-fou : un gaz reste à p ≥ 10 Pa absolu)
@compute @workgroup_size(64)
fn k_p_update(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n || is_solid(c)) { return; }
    var pr = prel[c] + cx[c];
    var gas = 0.0;
    for (var k = 0u; k < P.np; k++) { if (is_gas(k)) { gas += alpha[k * n + c]; } }
    if (gas > 1e-6 && pr + P.p0 < 10.0) { pr = 10.0 - P.p0; }
    prel[c] = pr;
}
