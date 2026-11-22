// ===== Espèces : diffusion de Fick (groupes miscibles), flamme épaissie (TFM), cinétique chimique (miroir de mp_species.rs) =====

// ---------------------------------------------------------------- diffusion des espèces d'un groupe (Q.b = groupe, Q.a = sous-pas)
@compute @workgroup_size(64)
fn k_sp_dloc(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let c = g.x; let n = ncell();
    let gi = Q.b;
    var v: array<f32, 8>;
    v[0] = 0.0;
    if (c < n) {
        let cnt = gcount(gi);
        let pa = S[O_prel + c] + P.p0;
        for (var q = 0u; q < cnt; q++) {
            let k = gmember(gi, q);
            var d = 0.0;
            if (!is_solid(c)) { d = max(diff_at(k, ptemp(k, c), pa), 0.0) * max(S[O_flame + c], 1.0); }
            S[O_dloc + q * n + c] = d;
            v[0] = max(v[0], d);
        }
    }
    wg_reduce(v, 1u, 1u, l.x, w.x, 0u);
}
@compute @workgroup_size(256)
fn k_sp_nsub(@builtin(local_invocation_id) l: vec3<u32>) {
    let dmax = final_reduce(0u, nwg_cells(), 1u, l.x);
    if (l.x == 0u) {
        var ns = 0.0;
        if (dmax > 0.0) { ns = clamp(ceil(dtv() * dmax / (0.2 * P.dx * P.dx)), 1.0, 500.0); }
        D[D_NSUB_S_MAX] = max(D[D_NSUB_S_MAX], ns);
        if (ns > f32(P.nsub_sp_max)) { D[D_REDO] = 1.0; ns = 0.0; }
        D[D_NSUB_S] = ns;
    }
}
fn sp_off(s: u32) -> bool { return f32(s) >= D[D_NSUB_S]; }
// masse du groupe et fractions massiques
fn sp_prep_cell(c: u32, gi: u32) {
    let n = ncell();
    let cnt = gcount(gi);
    var mg = 0.0;
    if (!is_solid(c)) { for (var q = 0u; q < cnt; q++) { mg += max(mass(gmember(gi, q), c), 0.0); } }
    S[O_mgs + c] = mg;
    for (var q = 0u; q < cnt; q++) {
        var y = 0.0;
        if (mg > 0.0) { y = max(mass(gmember(gi, q), c), 0.0) / mg; }
        S[O_ys + q * n + c] = y;
    }
}
@compute @workgroup_size(64)
fn k_sp_prep(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell() || sp_off(0u)) { return; }
    sp_prep_cell(c, Q.b);
}
// flux à travers la face (a < b) de a vers b ; out[q] = masse transférée, retour : énergie sensible transportée (−1e38 = pas de flux)
fn sp_face(a: u32, b: u32, gi: u32, ds: f32, out: ptr<function, array<f32, NPMAX>>) -> f32 {
    let n = ncell();
    let mga = S[O_mgs + a]; let mgb = S[O_mgs + b];
    if (is_solid(a) || is_solid(b) || mga <= 1e-12 || mgb <= 1e-12) { return -1.0e38; }
    let cnt = gcount(gi);
    let mbar = 0.5 * (mga + mgb);
    var jsum = 0.0;
    for (var q = 0u; q < cnt; q++) {
        let da = S[O_dloc + q * n + a]; let db = S[O_dloc + q * n + b];
        let dk = 2.0 * da * db / max(da + db, 1e-30);
        (*out)[q] = dk * mbar * (S[O_ys + q * n + a] - S[O_ys + q * n + b]) / (P.dx * P.dx) * ds;
        jsum += (*out)[q];
    }
    var sc = 1.0;
    for (var q = 0u; q < cnt; q++) {
        let k = gmember(gi, q);
        (*out)[q] -= 0.5 * (S[O_ys + q * n + a] + S[O_ys + q * n + b]) * jsum;
        var avail = 0.0;
        if ((*out)[q] > 0.0) { avail = 0.25 * max(mass(k, a), 0.0); } else { avail = 0.25 * max(mass(k, b), 0.0); }
        if (abs((*out)[q]) > avail) { sc = min(sc, avail / abs((*out)[q])); }
    }
    var ef = 0.0;
    for (var q = 0u; q < cnt; q++) {
        let k = gmember(gi, q);
        (*out)[q] *= sc;
        let gq = group_k(gmember(gi, q));
        var td = tgrp(gq, b); if ((*out)[q] > 0.0) { td = tgrp(gq, a); }
        ef += (*out)[q] * erel_k(k, td);
    }
    return ef;
}
@compute @workgroup_size(64)
fn k_sp_step(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n || sp_off(Q.a)) { return; }
    let gi = Q.b;
    let cnt = gcount(gi);
    let ds = dtv() / D[D_NSUB_S];
    let i = c % P.nx; let j = c / P.nx;
    var dm: array<f32, NPMAX>;
    for (var q = 0u; q < NPMAX; q++) { dm[q] = 0.0; }
    var de = 0.0;
    var out: array<f32, NPMAX>;
    for (var s = 0u; s < 4u; s++) {
        var b = -1;
        if (s == 0u && i > 0u) { b = i32(c - 1u); }
        if (s == 1u && i + 1u < P.nx) { b = i32(c + 1u); }
        if (s == 2u && j > 0u) { b = i32(c - P.nx); }
        if (s == 3u && j + 1u < P.ny) { b = i32(c + P.nx); }
        if (b < 0) { continue; }
        let bu = u32(b);
        var lo = c; var hi = bu; var sg = -1.0;
        if (bu < c) { lo = bu; hi = c; sg = 1.0; }
        let ef = sp_face(lo, hi, gi, ds, &out);
        if (ef > -1.0e37) {
            for (var q = 0u; q < cnt; q++) { dm[q] += sg * out[q]; }
            de += sg * ef;
        }
    }
    for (var q = 0u; q < cnt; q++) { S[O_newm + q * n + c] = max(mass(gmember(gi, q), c) + dm[q], 0.0); }
    // énergie du groupe thermique de ce groupe miscible (mêmes indices : groupes miscibles = groupes thermiques)
    S[O_newe + c] = epsg(gi, c) + de;
}
@compute @workgroup_size(64)
fn k_sp_commit(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n || sp_off(Q.a)) { return; }
    let gi = Q.b;
    let cnt = gcount(gi);
    for (var q = 0u; q < cnt; q++) { S[O_mass + gmember(gi, q) * n + c] = S[O_newm + q * n + c]; }
    if (!is_solid(c)) { S[O_epsg + gi * n + c] = S[O_newe + c]; cell_sync(c); }
    sp_prep_cell(c, gi);
}

// ---------------------------------------------------------------- réactions : tables
// T_RX + j·RX_STRIDE : 0 nréactifs, 1 nproduits, 2 nmodulateurs, 3 réversible, 4 flamme, 5 ln A (−1e30 : A ≤ 0), 6 b, 7 Ea/R,
//   8 échelle de masse (réactifs/produits), 9 début de la table ln Kc dans T, 10–15 réactifs, 16–21 ν, 22–27 ordres,
//   28–33 produits, 34–39 ν, 40–43 modulateurs, 44–47 ordres, 48 réaction de surface, 49 réactif granulaire, 50 diamètre des grains,
//   51 oxydant (−1 : aucun).
fn rxv(j: u32, q: u32) -> f32 { return T[T_RX + j * RX_STRIDE + q]; }
fn rx_nr(j: u32) -> u32 { return u32(rxv(j, 0u)); }
fn rx_np(j: u32) -> u32 { return u32(rxv(j, 1u)); }
fn rx_nm(j: u32) -> u32 { return u32(rxv(j, 2u)); }
fn rx_rev(j: u32) -> bool { return rxv(j, 3u) > 0.5; }
fn rx_flame(j: u32) -> bool { return rxv(j, 4u) > 0.5; }
fn rx_rs(j: u32, i: u32) -> u32 { return u32(rxv(j, 10u + i)); }
fn rx_rnu(j: u32, i: u32) -> f32 { return rxv(j, 16u + i); }
fn rx_rord(j: u32, i: u32) -> f32 { return rxv(j, 22u + i); }
fn rx_ps(j: u32, i: u32) -> u32 { return u32(rxv(j, 28u + i)); }
fn rx_pnu(j: u32, i: u32) -> f32 { return rxv(j, 34u + i); }
fn rx_ms(j: u32, i: u32) -> u32 { return u32(rxv(j, 40u + i)); }
fn rx_mord(j: u32, i: u32) -> f32 { return rxv(j, 44u + i); }
fn ln_kf(j: u32, t_in: f32) -> f32 {
    let t = max(t_in, 1.0);
    return rxv(j, 5u) + rxv(j, 6u) * log(t) - rxv(j, 7u) / t;
}
// ln Kc(T) interpolé (table de l'hôte, f64 → f32, pas KC_DT à partir de KC_T0)
fn ln_kc(j: u32, t: f32) -> f32 {
    let base = u32(rxv(j, 9u));
    var x = clamp((t - KC_T0) / KC_DT, 0.0, f32(KC_N - 1u) - 1e-3);
    if (!(x == x)) { x = 0.0; } // T invalide (pas rejeté ensuite) : jamais d'indice hors table
    let i = u32(floor(x));
    let f = x - f32(i);
    return T[base + i] * (1.0 - f) + T[base + i + 1u] * f;
}
const C_FLOOR: f32 = 1e-3;
fn powc(c: f32, o: f32) -> f32 {
    // c^o pour c > 0 ; c = 0 : 0 (o > 0)
    if (o == 0.0) { return 1.0; }
    if (c <= 0.0) { if (o > 0.0) { return 0.0; } return pow(C_FLOOR, o); }
    return exp(o * log(c));
}

// état chimique de la cellule en cours : masses de travail dans des tampons GLOBAUX par cellule (O_ch_m, copies O_ch_s / O_ch_b /
// O_ch_o) plutôt que dans des tableaux privés indexés dynamiquement (mémoire « locale » de toute façon sur GPU ; lavapipe plante
// sur ces tableaux privés dans des branches divergentes). Scalaires privés : cellule, ε, facteur de flamme.
var<private> cc_c: u32;
var<private> ceps: f32;
var<private> cfe: f32;
fn chm(k: u32) -> f32 { return S[O_ch_m + k * ncell() + cc_c]; }
fn ch_set(k: u32, v: f32) { S[O_ch_m + k * ncell() + cc_c] = v; }
// copie des masses de travail : tampon src → dst (décalages O_ch_*)
fn ch_copy(dst: u32, src: u32) {
    let n = ncell();
    for (var k = 0u; k < P.np; k++) { S[dst + k * n + cc_c] = S[src + k * n + cc_c]; }
}
fn cc_conc(k: u32) -> f32 { return max(chm(k), 0.0) / molar_k(k); }
fn cc_temp() -> f32 {
    var cap = 0.0; var sl = 0.0;
    for (var k = 0u; k < P.np; k++) { let m = chm(k); cap += m * ce_k(k); sl += m * slope_k(k); }
    return P.t0 + tau_from(cap, sl, ceps);
}
fn cc_cap(t: f32) -> f32 {
    var s = 0.0;
    for (var k = 0u; k < P.np; k++) { s += max(chm(k), 0.0) * cet_k(k, t); }
    return s;
}
fn cc_rate(j: u32, t: f32) -> f32 { let r = cc_rate_parts(j, t); return r.x - r.y; }
// capteur de flamme : débit net pondéré par l'écart à l'équilibre (1 − r_inv/r_dir) (miroir de CellChem::sensor_rate)
fn cc_sensor_rate(j: u32, t: f32) -> f32 {
    let r = cc_rate_parts(j, t);
    if (r.x <= 0.0) { return 0.0; }
    return max(r.x - r.y, 0.0) * clamp(1.0 - r.y / r.x, 0.0, 1.0);
}
// réaction de surface (miroir de CellChem::surface_rate) : S_v k_s Π c^o (réactifs non granulaires), limitée par la diffusion de
// l'oxydant jusqu'aux grains (somme harmonique avec S_v k_d C_ox, k_d = Sh D_ox / d, Sh = 2)
fn cc_surface_rate(j: u32, t: f32) -> f32 {
    let s = u32(rxv(j, 49u)); let d = rxv(j, 50u); let ox = rxv(j, 51u);
    let ms = chm(s);
    if (ms <= 0.0 || rxv(j, 5u) < -1e29) { return 0.0; }
    let pr = S[O_prel + cc_c];
    let sv = 6.0 * ms / rho_k(s, pr, t) / d;
    var lrk = ln_kf(j, t) + log(sv);
    var zero = false;
    for (var i = 0u; i < rx_nm(j); i++) {
        let o = rx_mord(j, i);
        let c = cc_conc(rx_ms(j, i));
        if (o < 0.0) { lrk += o * log(max(c, C_FLOOR)); }
        else if (o > 0.0) { if (c <= 0.0) { zero = true; } else { lrk += o * log(c); } }
    }
    for (var i = 0u; i < rx_nr(j); i++) {
        let k = rx_rs(j, i);
        if (k != s) {
            let o = rx_rord(j, i);
            let c = cc_conc(k);
            if (c <= 0.0) { zero = true; }
            else if (o < 0.0) { lrk += o * log(max(c, C_FLOOR)); }
            else if (o > 0.0) { lrk += o * log(c); }
        }
    }
    if (zero) { return 0.0; }
    let rk = exp(min(lrk, 80.0));
    if (ox < 0.0) { return rk; }
    let oi = u32(ox);
    let rd = sv * 2.0 * diff_at(oi, t, pr + P.p0) / d * cc_conc(oi);
    return select(0.0, rk * rd / (rk + rd), rk + rd > 0.0);
}
// (débit direct, débit inverse) (miroir de CellChem::rate_parts)
fn cc_rate_parts(j: u32, t: f32) -> vec2<f32> {
    if (rxv(j, 48u) > 0.5) { return vec2<f32>(cc_surface_rate(j, t), 0.0); }
    if (rxv(j, 5u) < -1e29) { return vec2<f32>(0.0, 0.0); }
    var lk = ln_kf(j, t);
    if (rx_flame(j)) { lk -= log(cfe); }
    var lmod = 0.0;
    var zero = false;
    for (var i = 0u; i < rx_nm(j); i++) {
        let o = rx_mord(j, i);
        let c = cc_conc(rx_ms(j, i));
        if (o < 0.0) { lmod += o * log(max(c, C_FLOOR)); }
        else if (o > 0.0) { if (c <= 0.0) { zero = true; } else { lmod += o * log(c); } }
    }
    var lrf = lk + lmod;
    var rzero = zero;
    for (var i = 0u; i < rx_nr(j); i++) {
        let o = rx_rord(j, i);
        let c = cc_conc(rx_rs(j, i));
        // réactif absent : pas de réaction directe, quel que soit l'ordre (miroir de CellChem::rate_parts)
        if (c <= 0.0) { rzero = true; }
        else if (o < 0.0) { lrf += o * log(max(c, C_FLOOR)); }
        else if (o > 0.0) { lrf += o * log(c); }
    }
    var rf = 0.0;
    if (!rzero) { rf = exp(min(lrf, 80.0)); }
    var lrr = lk + lmod - ln_kc(j, t);
    var pzero = zero || !rx_rev(j);
    for (var i = 0u; i < rx_np(j); i++) {
        let c = cc_conc(rx_ps(j, i));
        if (c <= 0.0) { pzero = true; } else { lrr += rx_pnu(j, i) * log(c); }
    }
    for (var i = 0u; i < rx_nr(j); i++) {
        let d = rx_rord(j, i) - rx_rnu(j, i);
        if (d != 0.0) { lrr += d * log(max(cc_conc(rx_rs(j, i)), C_FLOOR)); }
    }
    var rr = 0.0;
    if (!pzero) { rr = exp(min(lrr, 80.0)); }
    return vec2<f32>(rf, rr);
}
fn cc_bounds(j: u32) -> vec2<f32> {
    var fwd = 3.0e38;
    for (var i = 0u; i < rx_nr(j); i++) { let s = rx_rs(j, i); fwd = min(fwd, max(chm(s), 0.0) / (rx_rnu(j, i) * molar_k(s))); }
    var rev = 3.0e38;
    let sc = rxv(j, 8u);
    for (var i = 0u; i < rx_np(j); i++) { let s = rx_ps(j, i); rev = min(rev, max(chm(s), 0.0) / (rx_pnu(j, i) * molar_k(s) * sc)); }
    if (!rx_rev(j)) { rev = 0.0; }
    return vec2<f32>(-rev, fwd);
}
// avancement xi : masse exactement conservée (échelle), énergie interne absolue inchangée ⇒ ε −= Σ Δm h0
fn cc_apply(j: u32, xi: f32) {
    for (var i = 0u; i < rx_nr(j); i++) {
        let s = rx_rs(j, i);
        let o = chm(s);
        let nw = max(o - rx_rnu(j, i) * molar_k(s) * xi, 0.0);
        ch_set(s, nw);
        ceps -= (nw - o) * h0_k(s);
    }
    let sc = rxv(j, 8u);
    for (var i = 0u; i < rx_np(j); i++) {
        let s = rx_ps(j, i);
        let o = chm(s);
        let nw = max(o + rx_pnu(j, i) * molar_k(s) * xi * sc, 0.0);
        ch_set(s, nw);
        ceps -= (nw - o) * h0_k(s);
    }
}
// −Δu_j(T) (J/mol)
fn cc_heat(j: u32, t: f32) -> f32 {
    var s = 0.0;
    for (var i = 0u; i < rx_nr(j); i++) { let k = rx_rs(j, i); s += rx_rnu(j, i) * molar_k(k) * eabs_k(k, t); }
    let sc = rxv(j, 8u);
    for (var i = 0u; i < rx_np(j); i++) { let k = rx_ps(j, i); s -= rx_pnu(j, i) * sc * molar_k(k) * eabs_k(k, t); }
    return s;
}
fn cc_volume(pr: f32, t: f32) -> f32 {
    var v = 0.0;
    for (var k = 0u; k < P.np; k++) { v += max(chm(k), 0.0) / rho_k(k, pr, t); }
    return v;
}
fn reactive_k(k: u32) -> bool { return ph(k, 30u) > 0.5; }
// sous-système réactif de la cellule : masses et énergie des seuls groupes réactifs (miroir de integrate_cell)
fn cc_load(c: u32, feff: f32) {
    cc_c = c;
    for (var k = 0u; k < P.np; k++) { var m = 0.0; if (reactive_k(k)) { m = mass(k, c); } ch_set(k, m); }
    var e = 0.0;
    for (var g = 0u; g < P.ng; g++) { e += select(0.0, epsg(g, c), reactive_k(gmember(g, 0u))); }
    ceps = e;
    cfe = feff;
}
fn chx(o: u32, j: u32) -> f32 { return S[o + j * ncell() + cc_c]; }
fn chx_set(o: u32, j: u32, v: f32) { S[o + j * ncell() + cc_c] = v; }

// ---------------------------------------------------------------- flamme épaissie dynamique
@compute @workgroup_size(64)
fn k_flame_omega(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let c = g.x;
    var v: array<f32, 8>;
    v[0] = 0.0;
    if (c < ncell()) {
        var wsum = 0.0;
        if (!is_solid(c)) {
            cc_load(c, 1.0);
            for (var j = 0u; j < P.nre; j++) {
                if (!rx_flame(j)) { continue; }
                var skip = false;
                for (var i = 0u; i < rx_nr(j); i++) { if (mass(rx_rs(j, i), c) <= 1e-12) { skip = true; } }
                if (skip) { continue; }
                let t = ptemp(rx_rs(j, 0u), c);
                wsum += abs(cc_sensor_rate(j, t) * cc_heat(j, t));
            }
        }
        S[O_omega + c] = wsum;
        v[0] = wsum;
    }
    wg_reduce(v, 1u, 1u, l.x, w.x, 0u);
}
@compute @workgroup_size(256)
fn k_flame_wmax(@builtin(local_invocation_id) l: vec3<u32>) {
    let m = final_reduce(0u, nwg_cells(), 1u, l.x);
    if (l.x == 0u) { D[D_WMAX] = m; }
}
@compute @workgroup_size(64)
fn k_flame_mask(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    let wmax = D[D_WMAX];
    let o = S[O_omega + c];
    var m = 0.0;
    if (wmax > 0.0 && o > 0.02 * wmax && o > 50.0 * heat_cap(c)) { m = 1.0; }
    S[O_mask + c] = m;
}
@compute @workgroup_size(64)
fn k_flame_dil_x(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    let i = c % P.nx; let j = c / P.nx;
    let r = P.dil_r;
    let a = select(0u, i - r, i >= r);
    let b = min(i + r, P.nx - 1u);
    var m = 0.0;
    for (var ii = a; ii <= b; ii++) { m = max(m, S[O_mask + j * P.nx + ii]); }
    S[O_mtmp + c] = m;
}
@compute @workgroup_size(64)
fn k_flame_dil_y(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    let i = c % P.nx; let j = c / P.nx;
    let r = P.dil_r;
    let a = select(0u, j - r, j >= r);
    let b = min(j + r, P.ny - 1u);
    var m = 0.0;
    for (var jj = a; jj <= b; jj++) { m = max(m, S[O_mtmp + jj * P.nx + i]); }
    var f = 1.0 + (P.fmax - 1.0) * m;
    if (is_solid(c)) { f = 1.0; }
    S[O_flame + c] = f;
}
@compute @workgroup_size(64)
fn k_flame_one(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c < ncell()) { S[O_flame + c] = 1.0; }
}
@compute @workgroup_size(64)
fn k_heat_zero(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c < ncell()) { S[O_heat_rel + c] = 0.0; }
}

// ---------------------------------------------------------------- cinétique : intégration cellule par cellule
@compute @workgroup_size(64)
fn k_chem(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let c = min(g.x, ncell() - 1u);
    let inside = g.x < ncell();
    let n = ncell();
    let dt = dtv();
    var dt_cell = 3.0e38;
    // tri rapide
    // sélection (Q.a) : 0 toutes les cellules, 1 cellules de flamme (F > 1), 2 hors flamme (miroir de react_step_sel)
    let fl = S[O_flame + c] > 1.0;
    let sel_ok = Q.a == 0u || (Q.a == 1u && fl) || (Q.a == 2u && !fl);
    var go = inside && sel_ok && !(is_solid(c) || pinned(c));
    var possible = false;
    for (var j = 0u; j < P.nre; j++) {
        var all_r = true;
        for (var i = 0u; i < rx_nr(j); i++) { if (mass(rx_rs(j, i), c) <= 1e-12) { all_r = false; } }
        var all_p = rx_rev(j);
        for (var i = 0u; i < rx_np(j); i++) { if (mass(rx_ps(j, i), c) <= 1e-12) { all_p = false; } }
        if (all_r || all_p) { possible = true; }
    }
    go = go && possible;
    // NB : flot de contrôle uniforme autour de toutes les boucles (fils inactifs : calculs prédiqués, résultats jetés)
    cc_load(c, max(S[O_flame + c], 1.0));
    let t0 = cc_temp();
    var act = false;
    for (var j = 0u; j < P.nre; j++) {
        let bd = cc_bounds(j);
        let r = cc_rate(j, t0);
        if ((r > 0.0 && r * dt > 1e-7 * bd.y) || (r < 0.0 && -r * dt > 1e-7 * (-bd.x))) { act = true; }
    }
    go = go && act;
    let pr = S[O_prel + c];
    let v0 = cc_volume(pr, t0);
    ch_copy(O_ch_o, O_ch_m);
    for (var j = 0u; j < P.nre; j++) { chx_set(O_ch_xi, j, 0.0); }
    var tl = 0.0;
    var nsub = 0u;
    var stopped = false;
    var max_h = 3.0e38;
    loop {
        // lavapipe : chaque fil fait au moins un tour (résultats jetés si !go) — une boucle qu'aucun fil d'un groupe ne parcourt y fait
        // planter le code JIT (pointeur nul) ; inutile (et coûteux) sur les vrais pilotes
        if ((nsub > 0u || P.lvp_safe == 0u) && (!go || !(tl < dt * (1.0 - 1e-6)) || nsub >= P.chem_max_sub)) { break; }
        nsub += 1u;
        let t = cc_temp();
        let cap = max(cc_cap(t), 1e-30);
        for (var j = 0u; j < P.nre; j++) { chx_set(O_ch_r0, j, cc_rate(j, t)); }
        var h = min(dt - tl, max_h);
        var dtdt = 0.0;
        for (var k = 0u; k < P.np; k++) { chx_set(O_ch_cons, k, 0.0); }
        for (var j = 0u; j < P.nre; j++) {
            let r0 = chx(O_ch_r0, j);
            // trace : tout l'avancement possible chaufferait de moins de 5 K → ne borne pas le sous-pas (miroir de TRACE_DT)
            let sig = cc_bounds(j).y * abs(cc_heat(j, t)) / cap >= 5.0;
            var irr = 1.0; if (rx_rev(j) || !sig) { irr = 0.0; }
            dtdt += irr * r0 * cc_heat(j, t) / cap;
            for (var i = 0u; i < rx_nr(j); i++) {
                let s = rx_rs(j, i);
                chx_set(O_ch_cons, s, chx(O_ch_cons, s) + irr * rx_rnu(j, i) * molar_k(s) * max(r0, 0.0));
            }
        }
        if (abs(dtdt) > 0.0) { h = min(h, 60.0 / abs(dtdt)); }
        for (var s = 0u; s < P.np; s++) { let cs = chx(O_ch_cons, s); if (cs > 0.0 && chm(s) > 1e-12) { h = min(h, 0.4 * chm(s) / cs); } }
        h = max(h, dt * 1e-9);
        ch_copy(O_ch_s, O_ch_m);
        let sub_e = ceps;
        for (var j = 0u; j < P.nre; j++) { chx_set(O_ch_xs, j, chx(O_ch_xi, j)); }
        for (var j = 0u; j < P.nre; j++) {
            let bd = cc_bounds(j);
            let lo = bd.x; let hi = bd.y;
            let can = !(hi <= 0.0 && lo >= 0.0);
            let isrev = can && rx_rev(j);
            let ismid = can && !rx_rev(j);
            ch_copy(O_ch_b, O_ch_m);
            let base_e = ceps;
            var xi = 0.0;
            // réversible : Euler implicite (Newton, dérivée numérique ; T recalculée)
            var x = clamp(h * chx(O_ch_r0, j), 0.999 * lo, 0.999 * hi);
            var conv = !isrev;
            for (var it = 0; it < 6; it++) {
                if (!conv) {
                    ch_copy(O_ch_m, O_ch_b); ceps = base_e; cc_apply(j, x);
                    let rx_ = cc_rate(j, cc_temp());
                    let gx = x - h * rx_;
                    // pas de la dérivée numérique ~√ε_f32 relatif à x (le CPU f64 prend 1e-7 de l'intervalle)
                    let d = 3e-4 * max(abs(x), 1e-3 * abs(hi - lo)) + 1e-14;
                    var xd = x - d; if (x + d < hi) { xd = x + d; }
                    ch_copy(O_ch_m, O_ch_b); ceps = base_e; cc_apply(j, xd);
                    let rd = cc_rate(j, cc_temp());
                    let dr = (rd - rx_) / (xd - x);
                    let jac = 1.0 - h * dr;
                    var xn = h * rx_;
                    if (jac > 0.05) { xn = x - gx / jac; }
                    xn = clamp(xn, 0.999 * lo, 0.999 * hi);
                    conv = abs(xn - x) <= 1e-5 * max(abs(xn), 1e-4 * abs(hi - lo)) + 1e-30;
                    x = xn;
                }
            }
            ch_copy(O_ch_m, O_ch_b); ceps = base_e;
            if (isrev) { xi = x; }
            // irréversible : point milieu (RK2)
            let r1 = cc_rate(j, cc_temp());
            var half = clamp(0.5 * h * r1, 0.0, 0.999 * hi);
            if (!ismid) { half = 0.0; }
            cc_apply(j, half);
            let rh = cc_rate(j, cc_temp());
            ch_copy(O_ch_m, O_ch_b); ceps = base_e;
            if (ismid && half > 0.0) { xi = clamp(h * rh, 0.0, 0.999 * hi); }
            cc_apply(j, xi);
            chx_set(O_ch_xi, j, chx(O_ch_xi, j) + xi);
        }
        let dtemp = abs(cc_temp() - t);
        if (dtemp > 120.0 && h > dt * 1e-7 && nsub < P.chem_max_sub) {
            // sous-pas rejeté (échauffement > 120 K) : refait avec un pas plus court
            ch_copy(O_ch_m, O_ch_s); ceps = sub_e;
            for (var j = 0u; j < P.nre; j++) { chx_set(O_ch_xi, j, chx(O_ch_xs, j)); }
            max_h = h * min(0.5 * 120.0 / dtemp, 0.5);
        } else {
            max_h = 3.0e38;
            tl += h;
            let t1 = cc_temp();
            if (cc_volume(pr, t1) - v0 > P.chem_vol_lim * v0) { stopped = true; break; }
        }
    }
    var q = 0.0;
    for (var k = 0u; k < P.np; k++) { q -= (chm(k) - S[O_ch_o + k * n + c]) * e0_k(k); }
    let tn = cc_temp();
    let dv = abs(cc_volume(pr, tn) - v0) / max(v0, 1e-30);
    if (inside && sel_ok) { S[O_heat_rel + c] = 0.0; }
    if (go) {
        if (dv > 1e-3) {
            var span = dt; if (stopped) { span = tl; }
            dt_cell = 0.5 * P.chem_vol_lim * span / dv;
        }
        // groupes réactifs : nouvelles masses, énergie relative du sous-système répartie (un seul groupe : tout ; plusieurs :
        // température commune tn, le dernier reçoit le reste exact)
        for (var k = 0u; k < P.np; k++) { if (reactive_k(k)) { S[O_mass + k * n + c] = chm(k); } }
        // (sélections, sans « continue » : lavapipe)
        var last = 9999u;
        for (var gq = 0u; gq < P.ng; gq++) { last = select(last, gq, reactive_k(gmember(gq, 0u)) && group_mass(gq, c) > 0.0); }
        var acc = 0.0;
        for (var gq = 0u; gq < P.ng; gq++) {
            let r = reactive_k(gmember(gq, 0u));
            let use_t = r && gq != last && group_mass(gq, c) > 0.0;
            let eg = select(0.0, group_erel_at(gq, c, tn), use_t);
            acc += eg;
            S[O_epsg + gq * n + c] = select(epsg(gq, c), eg, r);
        }
        if (last < 9999u) { S[O_epsg + last * n + c] = ceps - acc; }
        cell_sync(c);
        S[O_heat_rel + c] = q / dt;
        let a2 = P.dx * P.dx;
        S[O_acc_chem + c] += q * a2;
        for (var j = 0u; j < P.nre; j++) { S[O_acc_xi + j * n + c] += chx(O_ch_xi, j) * a2; }
    }
    var v: array<f32, 8>;
    v[0] = -dt_cell;
    wg_reduce(v, 1u, 1u, l.x, w.x, 0u);
}
@compute @workgroup_size(256)
fn k_chem_final(@builtin(local_invocation_id) l: vec3<u32>) {
    let m = -final_reduce(0u, nwg_cells(), 1u, l.x);
    if (l.x == 0u) {
        var d = 3.0e38;
        if (m < 1e37) { d = max(m, 1e-8); }
        D[D_DTCHEM_NEW] = min(D[D_DTCHEM_NEW], d);   // plusieurs appels par pas (sous-pas des flammes) : minimum
    }
}
