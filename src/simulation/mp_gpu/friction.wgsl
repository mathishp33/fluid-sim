// ===== Frottement des grains tassés (miroir de mp_friction.rs) : viscosité de Drucker–Prager régularisée, implicite, gradient
// conjugué préconditionné (Jacobi) sur les faces. Vecteurs de faces « combinés » : u dans [0, nfu), v dans [nfu, nfu + nfv).

const FR_DELTA: f32 = 1e-3;     // taux de déformation de régularisation (1/s)
const FR_KMAX: f32 = 1000.0;    // plafond de μ_g dt / (ρ dx²)
const FR_TOL: f32 = 1e-6;       // tolérance relative du gradient conjugué

fn nfuv() -> u32 { return nfu() + nfv(); }
fn tanphi_k(k: u32) -> f32 { return ph(k, 35u); }

// propriétés granulaires d'une cellule : fraction de grains, poids déjaugé (Pa), tan φ moyen, montée de compaction, α_max moyen,
// masse volumique du fluide des pores
struct FrCell { as_: f32, w: f32, tp: f32, ramp: f32, amx: f32, rhof: f32 };
fn fr_cell(c: u32) -> FrCell {
    var r: FrCell;
    r.as_ = 0.0; r.w = 0.0; r.tp = 0.0; r.ramp = 0.0; r.amx = 1.0; r.rhof = 0.0;
    if (is_solid(c)) { return r; }
    var as_ = 0.0; var ms = 0.0; var af = 0.0; var mf = 0.0; var tp = 0.0; var am = 0.0;
    for (var k = 0u; k < P.np; k++) {
        let a = max(alpha(k, c), 0.0);
        let m = max(mass(k, c), 0.0);
        let gr = gran_k(k);
        as_ += select(0.0, a, gr); ms += select(0.0, m, gr);
        tp += select(0.0, a * tanphi_k(k), gr); am += select(0.0, a * grain_amax(k), gr);
        af += select(a, 0.0, gr); mf += select(m, 0.0, gr);
    }
    r.rhof = select(0.0, mf / max(af, 1e-30), af > 1e-12);
    if (as_ <= 1e-9) { return r; }
    let rho_s = ms / as_;
    r.as_ = as_;
    r.w = max(rho_s - r.rhof, 0.0) * as_ * grav_norm() * P.dx;
    r.tp = tp / as_;
    r.amx = am / as_;
    let ac = r.amx - 0.1;
    r.ramp = clamp((as_ - ac) / (r.amx - ac), 0.0, 1.0);
    return r;
}
fn fr_vertical() -> bool { return abs(P.gy) >= abs(P.gx); }
fn fr_down() -> bool { if (fr_vertical()) { return P.gy < 0.0; } return P.gx < 0.0; }
// voisine « en dessous » (aval de la gravité, axe dominant) ; -1 hors domaine
fn fr_below(c: u32) -> i32 {
    let i = c % P.nx; let j = c / P.nx;
    if (fr_vertical()) {
        if (fr_down()) { return select(-1, i32(c - P.nx), j > 0u); }
        return select(-1, i32(c + P.nx), j + 1u < P.ny);
    }
    if (fr_down()) { return select(-1, i32(c - 1u), i > 0u); }
    return select(-1, i32(c + 1u), i + 1u < P.nx);
}
// part de la cellule en contact (tassée, ou posée sur une cellule tassée) : α_s / α_max ; 0 sinon
fn fr_frac(c: u32, fc: FrCell) -> f32 {
    if (fc.as_ <= 1e-9 || is_solid(c)) { return 0.0; }
    let b = fr_below(c);
    var rest = true;
    if (b >= 0) { let bb = u32(b); rest = is_solid(bb) || fr_cell(bb).ramp > 0.0; }
    if (fc.ramp > 0.0 || rest) { return clamp(fc.as_ / fc.amx, 0.0, 1.0); }
    return 0.0;
}

// pression de contact p_s le long de chaque ligne de gravité (un fil par ligne) : pression du mélange moins celle du fluide des pores
@compute @workgroup_size(64)
fn k_fr_ps(@builtin(global_invocation_id) g: vec3<u32>) {
    let vert = fr_vertical();
    var lines = P.ny; var len = P.nx;
    if (vert) { lines = P.nx; len = P.ny; }
    let l = g.x;
    if (l >= lines) { return; }
    let down = fr_down();
    let gn = grav_norm();
    var pf = 0.0; var have_pf = false;
    var prev = -1;
    var prev_rhof = 0.0;
    for (var t = 0u; t < len; t++) {
        var s_ = t; if (down) { s_ = len - 1u - t; }
        var c = cidx(s_, l); if (vert) { c = cidx(l, s_); }
        let fc = fr_cell(c);
        let fr = fr_frac(c, fc);
        S[O_fr_frac + c] = fr;
        S[O_fr_ps + c] = 0.0;
        if (is_solid(c) || fr <= 0.0) {
            have_pf = false;
            prev = select(i32(c), -1, is_solid(c));
            prev_rhof = fc.rhof;
        } else {
            let pc = S[O_prel + c];
            if (!have_pf) {
                // haut du massif : fluide de la cellule du dessus, ou demi-poids de la cellule au bord du domaine
                if (prev >= 0) { pf = S[O_prel + u32(prev)] + 0.5 * (prev_rhof + fc.rhof) * gn * P.dx; } else { pf = pc - 0.5 * fc.w; }
                have_pf = true;
            } else if (prev >= 0) {
                pf += 0.5 * (prev_rhof + fc.rhof) * gn * P.dx;
            }
            S[O_fr_ps + c] = max(pc - pf, 0.0);
            prev = i32(c);
            prev_rhof = fc.rhof;
        }
    }
}
// viscosité de frottement μ_g par cellule (taux de déformation des vitesses du début du pas)
@compute @workgroup_size(64)
fn k_fr_mu(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    S[O_fr_mu + c] = 0.0;
    let fr = S[O_fr_frac + c];
    if (fr <= 0.0 || is_solid(c)) { return; }
    let fc = fr_cell(c);
    let nx = P.nx; let ny = P.ny; let dx = P.dx;
    let i = c % nx; let j = c / nx;
    let dxx = (S[O_Un + uidx(i + 1u, j)] - S[O_Un + uidx(i, j)]) / dx;
    let dyy = (S[O_Vn + vidx(i, j + 1u)] - S[O_Vn + vidx(i, j)]) / dx;
    let jm = select(j, j - 1u, j > 0u); let jp = min(j + 1u, ny - 1u);
    let im = select(i, i - 1u, i > 0u); let ip = min(i + 1u, nx - 1u);
    let dudy = 0.5 * ((S[O_Un + uidx(i, jp)] - S[O_Un + uidx(i, jm)]) + (S[O_Un + uidx(i + 1u, jp)] - S[O_Un + uidx(i + 1u, jm)])) / (f32(max(jp - jm, 1u)) * dx);
    let dvdx = 0.5 * ((S[O_Vn + vidx(ip, j)] - S[O_Vn + vidx(im, j)]) + (S[O_Vn + vidx(ip, j + 1u)] - S[O_Vn + vidx(im, j + 1u)])) / (f32(max(ip - im, 1u)) * dx);
    let dxy = 0.5 * (dudy + dvdx);
    let d2 = 0.5 * (dxx * dxx + dyy * dyy + 2.0 * dxy * dxy);
    let m = fr * fc.tp * S[O_fr_ps + c] / (2.0 * sqrt(d2 + FR_DELTA * FR_DELTA));
    S[O_fr_mu + c] = min(m, FR_KMAX * S[O_rho_mix + c] * dx * dx / D[D_DT]);
}

// viscosité au nœud (i, j) : moyenne des cellules voisines ; 0 contre un solide, aux coins et sur un bord ouvert
fn fr_mn(i: u32, j: u32) -> f32 {
    let nx = P.nx; let ny = P.ny;
    let on_l = i == 0u; let on_r = i == nx; let on_b = j == 0u; let on_t = j == ny;
    if ((on_l || on_r) && (on_b || on_t)) { return 0.0; }
    if ((on_l && P.bc_l != 0u) || (on_r && P.bc_r != 0u) || (on_b && P.bc_b != 0u) || (on_t && P.bc_t != 0u)) { return 0.0; }
    var s = 0.0; var cnt = 0.0; var solid = false;
    for (var q = 0u; q < 4u; q++) {
        let ci = i32(i) - 1 + i32(q & 1u); let cj = i32(j) - 1 + i32(q >> 1u);
        let inside = ci >= 0 && cj >= 0 && ci < i32(nx) && cj < i32(ny);
        let c = cidx(u32(clamp(ci, 0, i32(nx) - 1)), u32(clamp(cj, 0, i32(ny) - 1)));
        solid = solid || (inside && is_solid(c));
        s += select(0.0, S[O_fr_mu + c], inside);
        cnt += select(0.0, 1.0, inside);
    }
    if (solid) { return 0.0; }
    return s / max(cnt, 1.0);
}
fn fr_wu(o: u32, i: u32, j: u32) -> f32 { return S[o + uidx(i, j)]; }
fn fr_wv(o: u32, i: u32, j: u32) -> f32 { return S[o + nfu() + vidx(i, j)]; }
fn fr_txx(o: u32, i: u32, j: u32) -> f32 { let c = cidx(i, j); return 2.0 * S[O_fr_mu + c] * (fr_wu(o, i + 1u, j) - fr_wu(o, i, j)) / P.dx; }
fn fr_tyy(o: u32, i: u32, j: u32) -> f32 { let c = cidx(i, j); return 2.0 * S[O_fr_mu + c] * (fr_wv(o, i, j + 1u) - fr_wv(o, i, j)) / P.dx; }
fn fr_txy(o: u32, i: u32, j: u32) -> f32 {
    let m = fr_mn(i, j);
    if (m <= 0.0) { return 0.0; }
    let nx = P.nx; let ny = P.ny; let dx = P.dx;
    // vitesse tangentielle nulle à la paroi : ∂u/∂y = ±2u/dx ; ∂v/∂x nul le long d'une paroi horizontale (et inversement)
    var dudy = 0.0; var dvdx = 0.0;
    if (j == 0u) { dudy = 2.0 * fr_wu(o, i, 0u) / dx; }
    else if (j == ny) { dudy = -2.0 * fr_wu(o, i, ny - 1u) / dx; }
    else if (i == 0u) { dvdx = 2.0 * fr_wv(o, 0u, j) / dx; }
    else if (i == nx) { dvdx = -2.0 * fr_wv(o, nx - 1u, j) / dx; }
    else { dudy = (fr_wu(o, i, j) - fr_wu(o, i, j - 1u)) / dx; dvdx = (fr_wv(o, i, j) - fr_wv(o, i - 1u, j)) / dx; }
    return m * (dudy + dvdx);
}
// face inconnue (intérieure, entre deux cellules fluides) ; masse volumique de la face
fn fr_unknown(f: u32) -> bool {
    if (f < nfu()) {
        let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
        return i > 0u && i < P.nx && !is_solid(cidx(i - 1u, j)) && !is_solid(cidx(i, j));
    }
    let fv = f - nfu();
    let i = fv % P.nx; let j = fv / P.nx;
    return j > 0u && j < P.ny && !is_solid(cidx(i, j - 1u)) && !is_solid(cidx(i, j));
}
fn fr_rho(f: u32) -> f32 {
    if (f < nfu()) { return frho_x(f % (P.nx + 1u), f / (P.nx + 1u)); }
    let fv = f - nfu();
    return frho_y(fv % P.nx, fv / P.nx);
}
// force de frottement F(w) sur la face f (0 hors des inconnues)
fn fr_force(o: u32, f: u32) -> f32 {
    if (!fr_unknown(f)) { return 0.0; }
    if (f < nfu()) {
        let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
        return (fr_txx(o, i, j) - fr_txx(o, i - 1u, j)) / P.dx + (fr_txy(o, i, j + 1u) - fr_txy(o, i, j)) / P.dx;
    }
    let fv = f - nfu();
    let i = fv % P.nx; let j = fv / P.nx;
    return (fr_tyy(o, i, j) - fr_tyy(o, i, j - 1u)) / P.dx + (fr_txy(o, i + 1u, j) - fr_txy(o, i, j)) / P.dx;
}
// diagonale de A = ρ/dt − F (préconditionneur de Jacobi)
fn fr_diag(f: u32) -> f32 {
    if (!fr_unknown(f)) { return 1.0; }
    let dx2 = P.dx * P.dx;
    let r = fr_rho(f) / D[D_DT];
    if (f < nfu()) {
        let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
        let wt = select(1.0, 2.0, j + 1u == P.ny); let wb = select(1.0, 2.0, j == 0u);
        return r + 2.0 * (S[O_fr_mu + cidx(i - 1u, j)] + S[O_fr_mu + cidx(i, j)]) / dx2 + (wt * fr_mn(i, j + 1u) + wb * fr_mn(i, j)) / dx2;
    }
    let fv = f - nfu();
    let i = fv % P.nx; let j = fv / P.nx;
    let wr = select(1.0, 2.0, i + 1u == P.nx); let wl = select(1.0, 2.0, i == 0u);
    return r + 2.0 * (S[O_fr_mu + cidx(i, j - 1u)] + S[O_fr_mu + cidx(i, j)]) / dx2 + (wr * fr_mn(i + 1u, j) + wl * fr_mn(i, j)) / dx2;
}
// A w = ρ/dt w − F(w) sur les inconnues
fn fr_apply(o: u32, f: u32) -> f32 {
    if (!fr_unknown(f)) { return 0.0; }
    return fr_rho(f) / D[D_DT] * S[o + f] - fr_force(o, f);
}
fn fr_wg_total(q: u32, lid: u32) -> f32 {
    let nw = (nfuv() + 63u) / 64u;
    var acc = 0.0;
    for (var w = lid; w < nw; w += 64u) { acc += S[O_part + q * NWG + w]; }
    shg[lid] = acc;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (lid < s) { shg[lid] = shg[lid] + shg[lid + s]; }
        workgroupBarrier();
    }
    let r = shg[0];
    workgroupBarrier();
    return r;
}
fn fr_off() -> bool { return D[D_FR_DONE] > 0.5; }

// u_f = u* − dt ∇pⁿ / ρ_f sur les inconnues (vitesse qui équilibre déjà la pression du pas précédent)
@compute @workgroup_size(64)
fn k_fr_uf(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f >= nfuv()) { return; }
    var us = 0.0; var dp = 0.0;
    if (f < nfu()) {
        us = S[O_Us + f];
        let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
        if (fr_unknown(f)) { dp = S[O_prel + cidx(i, j)] - S[O_prel + cidx(i - 1u, j)]; }
    } else {
        let fv = f - nfu();
        us = S[O_Vs + fv];
        let i = fv % P.nx; let j = fv / P.nx;
        if (fr_unknown(f)) { dp = S[O_prel + cidx(i, j)] - S[O_prel + cidx(i, j - 1u)]; }
    }
    S[O_fr_uf + f] = select(us, us - D[D_DT] / fr_rho(f) * dp / P.dx, fr_unknown(f));
}
// r = F(u_f), z = r / diag, p = z, x = 0 ; sommes partielles r·z (quantité 0) et r·r (quantité 1)
@compute @workgroup_size(64)
fn k_fr_r0(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let f = g.x;
    var v: array<f32, 8>;
    v[0] = 0.0; v[1] = 0.0;
    if (f < nfuv()) {
        let r = fr_force(O_fr_uf, f);
        let z = r / fr_diag(f);
        S[O_fr_r + f] = r; S[O_fr_z + f] = z; S[O_fr_p + f] = z; S[O_fr_x + f] = 0.0;
        v[0] = r * z; v[1] = r * r;
    }
    wg_reduce(v, 2u, 0u, l.x, w.x, 0u);
}
@compute @workgroup_size(256)
fn k_fr_start(@builtin(local_invocation_id) l: vec3<u32>) {
    let nw = (nfuv() + 63u) / 64u;
    let rz = final_reduce(0u, nw, 0u, l.x);
    let rr = final_reduce(1u, nw, 0u, l.x);
    if (l.x == 0u) {
        D[D_FR_RZ] = rz; D[D_FR_RZ + 1u] = rz;
        D[D_FR_R0] = sqrt(rr);
        D[D_FR_IT] = 0.0;
        D[D_FR_DONE] = select(0.0, 1.0, rr <= 0.0);
    }
}
// itération (emplacement Q.a = parité) : A p, sommes partielles p·A p (quantité 0)
@compute @workgroup_size(64)
fn k_fr_ap(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let f = g.x;
    var v: array<f32, 8>;
    v[0] = 0.0;
    if (f < nfuv() && !fr_off()) {
        let a = fr_apply(O_fr_p, f);
        S[O_fr_ap + f] = a;
        v[0] = S[O_fr_p + f] * a;
    }
    wg_reduce(v, 1u, 0u, l.x, w.x, 0u);
}
// α = rz / pAp ; x += α p, r −= α A p, z = r / diag ; sommes partielles r·z (quantité 1) et r·r (quantité 2) — la quantité 0
// (p·A p) est encore lue par les autres groupes de travail pendant ce lancement
@compute @workgroup_size(64)
fn k_fr_upd(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let pap = fr_wg_total(0u, l.x);
    let f = g.x;
    var v: array<f32, 8>;
    v[0] = 0.0; v[1] = 0.0;
    let off = fr_off() || !(pap > 0.0);
    if (f < nfuv() && !off) {
        let a = D[D_FR_RZ + ((Q.a + 1u) & 1u)] / pap;
        S[O_fr_x + f] += a * S[O_fr_p + f];
        let r = S[O_fr_r + f] - a * S[O_fr_ap + f];
        let z = r / fr_diag(f);
        S[O_fr_r + f] = r; S[O_fr_z + f] = z;
        v[0] = r * z; v[1] = r * r;
    }
    if (f == 0u && !fr_off() && !(pap > 0.0)) { D[D_FR_DONE] = 1.0; }
    wg_reduce(v, 2u, 0u, l.x, w.x, 1u);
}
// β = rz_nouveau / rz, p = z + β p ; convergence ‖r‖ ≤ tol ‖r0‖ ; rz_nouveau dans D_FR_RZ + (parité)
@compute @workgroup_size(64)
fn k_fr_dir(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let rz2 = fr_wg_total(1u, l.x);
    let rr = fr_wg_total(2u, l.x);
    let f = g.x;
    if (fr_off()) { return; }
    let conv = sqrt(rr) <= FR_TOL * D[D_FR_R0];
    if (f == 0u) {
        D[D_FR_RZ + (Q.a & 1u)] = rz2;
        D[D_FR_IT] += 1.0;
        if (conv) { D[D_FR_DONE] = 1.0; }
    }
    if (conv) { return; }
    let rz = D[D_FR_RZ + ((Q.a + 1u) & 1u)];
    let be = rz2 / max(rz, 1e-30);
    if (f < nfuv()) { S[O_fr_p + f] = S[O_fr_z + f] + be * S[O_fr_p + f]; }
}
// u* += δ ; budget insuffisant signalé (l'hôte l'augmente au pas suivant)
@compute @workgroup_size(64)
fn k_fr_apply(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f == 0u && D[D_FR_DONE] < 0.5) { D[D_FR_UNCONV] = 1.0; }
    if (f >= nfuv()) { return; }
    let x = S[O_fr_x + f];
    if (f < nfu()) { S[O_Us + f] += x; } else { S[O_Vs + f - nfu()] += x; }
}
