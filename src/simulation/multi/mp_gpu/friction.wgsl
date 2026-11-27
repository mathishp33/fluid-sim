// ===== Viscosité implicite (miroir de mp_friction.rs) : frottement des grains tassés (Drucker–Prager régularisé, non-glissement aux
// parois et aux solides, chaleur de frottement) et part des fluides au-delà de la borne explicite ; gradient conjugué préconditionné
// (Jacobi) sur les faces. Vecteurs de faces « combinés » : u dans [0, nfu), v dans [nfu, nfu + nfv).

const FR_DELTA: f32 = 1e-3;     // taux de déformation de régularisation (1/s)
const FR_KMAX: f32 = 1000.0;    // plafond de μ_g dt / (ρ dx²)
const FR_TOL: f32 = 1e-6;       // tolérance relative du gradient conjugué

fn nfuv() -> u32 { return nfu() + nfv(); }
fn tanphi_k(k: u32) -> f32 { return ph(k, 35u); }

// propriétés granulaires d'une cellule : fraction de grains, poids déjaugé (Pa), tan φ moyen, montée de compaction, α_max moyen,
// masse volumique du fluide des pores
struct FrCell { as_: f32, w: f32, tp: f32, ramp: f32, amx: f32, rhof: f32 };
// cellule qui bloque les grains pour le frottement : solide ou grille (miroir de fr_blocked)
fn fr_blocked(c: u32) -> bool { return is_solid(c) || is_grille(c); }
fn fr_cell(c: u32) -> FrCell {
    var r: FrCell;
    r.as_ = 0.0; r.w = 0.0; r.tp = 0.0; r.ramp = 0.0; r.amx = 1.0; r.rhof = 0.0;
    if (fr_blocked(c)) { return r; }
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
    if (fc.as_ <= 1e-9 || fr_blocked(c)) { return 0.0; }
    let b = fr_below(c);
    var rest = true;
    if (b >= 0) { let bb = u32(b); rest = fr_blocked(bb) || fr_cell(bb).ramp > 0.0; }
    if (fc.ramp > 0.0 || rest) { return clamp(fc.as_ / fc.amx, 0.0, 1.0); }
    return 0.0;
}

// pression de contact p_s le long de chaque ligne de gravité (un fil par ligne) : lits perméables, σ' (sig) ; ailleurs, pression du
// mélange moins celle du fluide des pores (hydrostatique depuis le haut du massif)
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
        if (fr_blocked(c) || fr <= 0.0) {
            have_pf = false;
            prev = select(i32(c), -1, fr_blocked(c));
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
            S[O_fr_ps + c] = max(select(pc - pf, S[O_sig + c], S[O_skel + c] > 0.5), 0.0);
            prev = i32(c);
            prev_rhof = fc.rhof;
        }
    }
}
// viscosités implicites par cellule : frottement μ_g (taux de déformation des vitesses du début du pas) et part des fluides au-delà de
// la borne explicite μ_v = μ − 0.2 ρ dx²/dt (viscosité implicite)
@compute @workgroup_size(64)
fn k_fr_mu(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    S[O_fr_mu + c] = 0.0;
    S[O_fr_mv + c] = 0.0;
    if (fr_blocked(c)) { return; }
    let dx = P.dx;
    if (has(F_IMPVISC)) { S[O_fr_mv + c] = max(S[O_mu_cell + c] - mu_exp_cap(c), 0.0); }
    if (!has(F_FRIC)) { return; }
    let fr = S[O_fr_frac + c];
    if (fr <= 0.0) { return; }
    let fc = fr_cell(c);
    let nx = P.nx; let ny = P.ny;
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

// genre du nœud (i, j) (miroir de NodeKind) : 0 intérieur, 1 paroi dessous, 2 dessus, 3 gauche, 4 droite, 5 aucun (coin, bord ouvert,
// une ou trois cellules bloquées) ; cellule bloquée = solide, ou hors d'un bord fermé
const NK_IN: u32 = 0u;
const NK_FLOOR: u32 = 1u;
const NK_CEIL: u32 = 2u;
const NK_LEFT: u32 = 3u;
const NK_RIGHT: u32 = 4u;
const NK_NONE: u32 = 5u;
fn fr_kind(i: u32, j: u32) -> u32 {
    let nx = i32(P.nx); let ny = i32(P.ny);
    var bits = 0u;
    var open = false;
    for (var q = 0u; q < 4u; q++) {
        let ci = i32(i) - 1 + i32(q & 1u); let cj = i32(j) - 1 + i32(q >> 1u);
        let out = ci < 0 || cj < 0 || ci >= nx || cj >= ny;
        open = open || (ci < 0 && P.bc_l != 0u) || (ci >= nx && P.bc_r != 0u) || (cj < 0 && P.bc_b != 0u) || (cj >= ny && P.bc_t != 0u);
        let c = cidx(u32(clamp(ci, 0, nx - 1)), u32(clamp(cj, 0, ny - 1)));
        let b = out || fr_blocked(c);
        bits = bits | select(0u, 1u << q, b);
    }
    if (open) { return NK_NONE; }
    // bits : 1 = SO, 2 = SE, 4 = NO, 8 = NE
    if (bits == 0u) { return NK_IN; }
    if (bits == 3u) { return NK_FLOOR; }
    if (bits == 12u) { return NK_CEIL; }
    if (bits == 5u) { return NK_LEFT; }
    if (bits == 10u) { return NK_RIGHT; }
    return NK_NONE;
}
// nombre de cellules non bloquées autour du nœud, et moyenne d'une viscosité par cellule (tranche o) sur elles
fn fr_node_cnt(i: u32, j: u32) -> f32 {
    let nx = i32(P.nx); let ny = i32(P.ny);
    var cnt = 0.0;
    for (var q = 0u; q < 4u; q++) {
        let ci = i32(i) - 1 + i32(q & 1u); let cj = i32(j) - 1 + i32(q >> 1u);
        let inside = ci >= 0 && cj >= 0 && ci < nx && cj < ny;
        let c = cidx(u32(clamp(ci, 0, nx - 1)), u32(clamp(cj, 0, ny - 1)));
        cnt += select(0.0, 1.0, inside && !fr_blocked(c));
    }
    return cnt;
}
fn fr_node_avg(o: u32, i: u32, j: u32) -> f32 {
    let nx = i32(P.nx); let ny = i32(P.ny);
    var s = 0.0; var cnt = 0.0;
    for (var q = 0u; q < 4u; q++) {
        let ci = i32(i) - 1 + i32(q & 1u); let cj = i32(j) - 1 + i32(q >> 1u);
        let inside = ci >= 0 && cj >= 0 && ci < nx && cj < ny;
        let c = cidx(u32(clamp(ci, 0, nx - 1)), u32(clamp(cj, 0, ny - 1)));
        let ok = inside && !fr_blocked(c);
        s += select(0.0, S[o + c], ok);
        cnt += select(0.0, 1.0, ok);
    }
    return s / max(cnt, 1.0);
}
// nœud (i, j) d'un bord du domaine (sans les coins) contre une paroi adhérente : les fluides y adhèrent aussi (miroir de wall_node_vel)
fn fr_wall(i: u32, j: u32) -> bool {
    if ((j == 0u || j == P.ny) && i >= 1u && i < P.nx) { return select(P.ns_b, P.ns_t, j == P.ny) != 0u; }
    if ((i == 0u || i == P.nx) && j >= 1u && j < P.ny) { return select(P.ns_l, P.ns_r, i == P.nx) != 0u; }
    return false;
}
// vitesse de la paroi au nœud, pour le second membre et la chaleur ; 0 dans l'opérateur A (linéaire), appliqué au tampon O_fr_p
fn fr_uw(o: u32, i: u32, j: u32) -> f32 {
    if (o == O_fr_p || !fr_wall(i, j)) { return 0.0; }
    if (j == 0u) { return P.uw_b; }
    if (j == P.ny) { return P.uw_t; }
    if (i == 0u) { return P.uw_l; }
    return P.uw_r;
}
fn fr_wu(o: u32, i: u32, j: u32) -> f32 { return S[o + uidx(i, j)]; }
fn fr_wv(o: u32, i: u32, j: u32) -> f32 { return S[o + nfu() + vidx(i, j)]; }
fn fr_mu_c(c: u32) -> f32 { return S[O_fr_mu + c] + S[O_fr_mv + c]; }
fn fr_txx(o: u32, i: u32, j: u32) -> f32 { let c = cidx(i, j); return 2.0 * fr_mu_c(c) * (fr_wu(o, i + 1u, j) - fr_wu(o, i, j)) / P.dx; }
fn fr_tyy(o: u32, i: u32, j: u32) -> f32 { let c = cidx(i, j); return 2.0 * fr_mu_c(c) * (fr_wv(o, i, j + 1u) - fr_wv(o, i, j)) / P.dx; }
// cisaillement γ au nœud (genre k) : non-glissement aux parois (∂u/∂y = ±2u/dx, ∂v/∂x = ±2v/dx)
fn fr_gamma(o: u32, i: u32, j: u32, k: u32) -> f32 {
    let dx = P.dx;
    if (k == NK_IN) { return ((fr_wu(o, i, j) - fr_wu(o, i, j - 1u)) + (fr_wv(o, i, j) - fr_wv(o, i - 1u, j))) / dx; }
    if (k == NK_FLOOR) { return 2.0 * (fr_wu(o, i, j) - fr_uw(o, i, j)) / dx; }
    if (k == NK_CEIL) { return -2.0 * (fr_wu(o, i, j - 1u) - fr_uw(o, i, j)) / dx; }
    if (k == NK_LEFT) { return 2.0 * (fr_wv(o, i, j) - fr_uw(o, i, j)) / dx; }
    if (k == NK_RIGHT) { return -2.0 * (fr_wv(o, i - 1u, j) - fr_uw(o, i, j)) / dx; }
    return 0.0;
}
// contrainte de cisaillement au nœud : μ_g (parois comprises) + μ_v des fluides (nœuds intérieurs et parois adhérentes, comme
// l'explicite)
fn fr_txy(o: u32, i: u32, j: u32) -> f32 {
    let k = fr_kind(i, j);
    if (k == NK_NONE) { return 0.0; }
    var m = fr_node_avg(O_fr_mu, i, j);
    if (k == NK_IN || fr_wall(i, j)) { m += fr_node_avg(O_fr_mv, i, j); }
    if (m <= 0.0) { return 0.0; }
    return m * fr_gamma(o, i, j, k);
}
// contribution d'un nœud à la diagonale de la face qu'il cisaille : intérieur μ_g + μ_v, paroi `wall` 2 μ_g (+ 2 μ_v contre une paroi
// adhérente)
fn fr_nodec(i: u32, j: u32, wall: u32) -> f32 {
    let k = fr_kind(i, j);
    if (k == NK_IN) { return fr_node_avg(O_fr_mu, i, j) + fr_node_avg(O_fr_mv, i, j); }
    if (k == wall) { return 2.0 * (fr_node_avg(O_fr_mu, i, j) + select(0.0, fr_node_avg(O_fr_mv, i, j), fr_wall(i, j))); }
    return 0.0;
}
// face inconnue (intérieure, entre deux cellules fluides) ; masse volumique de la face
fn fr_unknown(f: u32) -> bool {
    if (f < nfu()) {
        let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
        return i > 0u && i < P.nx && !fr_blocked(cidx(i - 1u, j)) && !fr_blocked(cidx(i, j));
    }
    let fv = f - nfu();
    let i = fv % P.nx; let j = fv / P.nx;
    return j > 0u && j < P.ny && !fr_blocked(cidx(i, j - 1u)) && !fr_blocked(cidx(i, j));
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
        return r + 2.0 * (fr_mu_c(cidx(i - 1u, j)) + fr_mu_c(cidx(i, j))) / dx2 + (fr_nodec(i, j + 1u, NK_CEIL) + fr_nodec(i, j, NK_FLOOR)) / dx2;
    }
    let fv = f - nfu();
    let i = fv % P.nx; let j = fv / P.nx;
    return r + 2.0 * (fr_mu_c(cidx(i, j - 1u)) + fr_mu_c(cidx(i, j))) / dx2 + (fr_nodec(i + 1u, j, NK_RIGHT) + fr_nodec(i, j, NK_LEFT)) / dx2;
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

// u_f = u* − dt ∇pⁿ / ρ_f sur les inconnues (vitesse qui équilibre déjà la pression du pas précédent) ; faces fixes (parois, solides)
// à 0 : u* y porte encore dt·g (la projection les annule), qui fausserait le cisaillement de la couche voisine
@compute @workgroup_size(64)
fn k_fr_uf(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f >= nfuv()) { return; }
    var us = 0.0; var dp = 0.0; var fixed = false;
    if (f < nfu()) {
        us = S[O_Us + f];
        let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
        if (fr_unknown(f)) { dp = ptot(cidx(i, j)) - ptot(cidx(i - 1u, j)); }
        let solid_side = (i > 0u && fr_blocked(cidx(i - 1u, j))) || (i < P.nx && fr_blocked(cidx(i, j)));
        fixed = (i == 0u && P.bc_l == 0u) || (i == P.nx && P.bc_r == 0u) || (i > 0u && i < P.nx && !fr_unknown(f)) || solid_side;
    } else {
        let fv = f - nfu();
        us = S[O_Vs + fv];
        let i = fv % P.nx; let j = fv / P.nx;
        if (fr_unknown(f)) { dp = ptot(cidx(i, j)) - ptot(cidx(i, j - 1u)); }
        let solid_side = (j > 0u && fr_blocked(cidx(i, j - 1u))) || (j < P.ny && fr_blocked(cidx(i, j)));
        fixed = (j == 0u && P.bc_b == 0u) || (j == P.ny && P.bc_t == 0u) || (j > 0u && j < P.ny && !fr_unknown(f)) || solid_side;
    }
    let w = select(us, us - D[D_DT] / fr_rho(f) * dp / P.dx, fr_unknown(f));
    S[O_fr_uf + f] = select(w, 0.0, fixed);
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
// vitesse résolue w = u_f + δ (dans fr_ap, libre après le gradient conjugué) : chaleur de frottement
@compute @workgroup_size(64)
fn k_fr_w(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f >= nfuv()) { return; }
    S[O_fr_ap + f] = S[O_fr_uf + f] + S[O_fr_x + f];
}
// chaleur de frottement Φ dt (Φ = τ : D, part des grains ; miroir de friction_dissipation) rendue aux groupes granulaires : 2 μ_g (Dxx² +
// Dyy²) dans la cellule, plus sa part des nœuds voisins (μ_n γ², poids ½ sur une paroi) au prorata de μ_g des cellules du nœud
@compute @workgroup_size(64)
fn k_fr_heat(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell() || fr_blocked(c)) { return; }
    let mg = S[O_fr_mu + c];
    if (mg <= 0.0) { return; }
    let o = O_fr_ap;
    let i = c % P.nx; let j = c / P.nx;
    let dxx = (fr_wu(o, i + 1u, j) - fr_wu(o, i, j)) / P.dx;
    let dyy = (fr_wv(o, i, j + 1u) - fr_wv(o, i, j)) / P.dx;
    var phi = 2.0 * mg * (dxx * dxx + dyy * dyy);
    for (var q = 0u; q < 4u; q++) {
        let ni = i + (q & 1u); let nj = j + (q >> 1u);
        let k = fr_kind(ni, nj);
        let mn = fr_node_avg(O_fr_mu, ni, nj);
        let gam = fr_gamma(o, ni, nj, k);
        let wgt = select(0.5, 1.0, k == NK_IN);
        // Σ μ_g des cellules non bloquées du nœud = moyenne × nombre
        let tot = mn * fr_node_cnt(ni, nj);
        phi += select(0.0, wgt * mn * gam * gam * mg / max(tot, 1e-30), k != NK_NONE && tot > 0.0);
    }
    var as_ = 0.0;
    for (var k = 0u; k < P.np; k++) { as_ += select(0.0, max(alpha(k, c), 0.0), gran_k(k)); }
    if (as_ <= 0.0) { return; }
    let n = ncell();
    let q = D[D_DT] * phi;
    for (var k = 0u; k < P.np; k++) {
        let a = select(0.0, max(alpha(k, c), 0.0), gran_k(k));
        S[O_epsg + group_k(k) * n + c] += q * a / as_;
    }
    cell_sync(c);
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
