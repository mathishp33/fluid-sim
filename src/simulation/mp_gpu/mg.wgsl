// ===== Gradient conjugué préconditionné par multigrille (miroir de mp_mg.rs / mp_pressure.rs::pcg_mg) =====
// Niveaux : T[T_LEV + l·LEV_STRIDE] = nx, ny, diag, ae, an, act, x, b (décalages dans S). Niveau 0 = matrice de pression
// (sd, sae, san, sact), x = z (cz), b = r (cr). V-cycle : pré-lissage 2 × (rouge, noir), restriction du résidu (somme sur l'agrégat 2×2),
// correction grossière, prolongation constante, post-lissage 2 × (noir, rouge) ; grille la plus grossière : 20 balayages symétriques.
// Les niveaux l ≥ P.mg_tail (petites grilles) sont traités par UN groupe de travail (k_mg_tail) : un seul lancement au lieu de ~10 par niveau.
// Tous les noyaux s'arrêtent si le gradient conjugué ou le Newton ont convergé (drapeaux dans D).

fn lv(l: u32, q: u32) -> u32 { return u32(T[T_LEV + l * LEV_STRIDE + q]); }
fn pcg_off() -> bool { return D[D_PCG_DONE] > 0.5 || D[D_NEWTON_DONE] > 0.5; }
fn lnx(l: u32) -> u32 { return lv(l, 0u); }
fn lny(l: u32) -> u32 { return lv(l, 1u); }

// Σ couplages × x des voisins (niveau l)
fn mg_off(l: u32, i: u32, j: u32, c: u32, xo: u32) -> f32 {
    let nx = lnx(l); let ny = lny(l); let ae = lv(l, 3u); let an = lv(l, 4u);
    var s = 0.0;
    if (i + 1u < nx) { s += S[ae + c] * S[xo + c + 1u]; }
    if (i > 0u) { s += S[ae + c - 1u] * S[xo + c - 1u]; }
    if (j + 1u < ny) { s += S[an + c] * S[xo + c + nx]; }
    if (j > 0u) { s += S[an + c - nx] * S[xo + c - nx]; }
    return s;
}
// mise à jour de Gauss–Seidel de la cellule c
fn gs_cell(l: u32, c: u32) {
    let nx = lnx(l);
    let i = c % nx; let j = c / nx;
    let d = lv(l, 2u); let xo = lv(l, 6u); let bo = lv(l, 7u);
    if (S[lv(l, 5u) + c] > 0.5) { S[xo + c] = (S[bo + c] + mg_off(l, i, j, c, xo)) / S[d + c]; }
    else { S[xo + c] = S[bo + c] / S[d + c]; }
}
// premier demi-balayage depuis x = 0 : rouge = b/d, noir = 0
fn first_cell(l: u32, c: u32) {
    let nx = lnx(l);
    let i = c % nx; let j = c / nx;
    let xo = lv(l, 6u);
    if ((i + j) % 2u == 0u) { S[xo + c] = S[lv(l, 7u) + c] / S[lv(l, 2u) + c]; } else { S[xo + c] = 0.0; }
}
// résidu de la cellule fine k (niveau l)
fn resid_cell(l: u32, c: u32) -> f32 {
    let nx = lnx(l);
    let i = c % nx; let j = c / nx;
    let xo = lv(l, 6u);
    return S[lv(l, 7u) + c] - (S[lv(l, 2u) + c] * S[xo + c] - mg_off(l, i, j, c, xo));
}
// second membre grossier = Σ des résidus des cellules fines actives de l'agrégat
fn restrict_cell(l: u32, gc: u32) {
    let fnx = lnx(l); let fny = lny(l); let cnx = lnx(l + 1u);
    let gi = gc % cnx; let gj = gc / cnx;
    let fac = lv(l, 5u);
    var s = 0.0;
    for (var dj = 0u; dj < 2u; dj++) { for (var di = 0u; di < 2u; di++) {
        let i = 2u * gi + di; let j = 2u * gj + dj;
        if (i >= fnx || j >= fny) { continue; }
        let c = j * fnx + i;
        if (S[fac + c] > 0.5) { s += resid_cell(l, c); }
    } }
    S[lv(l + 1u, 7u) + gc] = s;
}
// premier demi-balayage noir du post-lissage, prolongation incluse : voisins (rouges) lus avec leur correction grossière
fn post1_cell(l: u32, c: u32) {
    let nx = lnx(l); let ny = lny(l);
    let i = c % nx; let j = c / nx;
    let d = lv(l, 2u); let ae = lv(l, 3u); let an = lv(l, 4u); let ac = lv(l, 5u); let xo = lv(l, 6u); let bo = lv(l, 7u);
    let cnx = lnx(l + 1u); let xc = lv(l + 1u, 6u);
    if (S[ac + c] < 0.5) { S[xo + c] = S[bo + c] / S[d + c]; return; }
    var s = 0.0;
    if (i + 1u < nx) { let b = c + 1u; s += S[ae + c] * (S[xo + b] + S[ac + b] * S[xc + (j / 2u) * cnx + (i + 1u) / 2u]); }
    if (i > 0u) { let b = c - 1u; s += S[ae + b] * (S[xo + b] + S[ac + b] * S[xc + (j / 2u) * cnx + (i - 1u) / 2u]); }
    if (j + 1u < ny) { let b = c + nx; s += S[an + c] * (S[xo + b] + S[ac + b] * S[xc + ((j + 1u) / 2u) * cnx + i / 2u]); }
    if (j > 0u) { let b = c - nx; s += S[an + b] * (S[xo + b] + S[ac + b] * S[xc + ((j - 1u) / 2u) * cnx + i / 2u]); }
    S[xo + c] = (S[bo + c] + s) / S[d + c];
}
// index de la t-ième cellule de la couleur `color` (une demi-grille par lancement)
fn color_cell(l: u32, t: u32, color: u32) -> i32 {
    let nx = lnx(l); let ny = lny(l);
    let half = (nx + 1u) / 2u;
    let j = t / half;
    if (j >= ny) { return -1; }
    let i = 2u * (t % half) + ((color + j) % 2u);
    if (i >= nx) { return -1; }
    return i32(j * nx + i);
}

// ---------------------------------------------------------------- hiérarchie (à chaque itération de Newton)
@compute @workgroup_size(64)
fn k_mg_coarsen_links(@builtin(global_invocation_id) g: vec3<u32>) {
    if (D[D_NEWTON_DONE] > 0.5) { return; }
    let l = Q.a;
    let fnx = lnx(l); let fny = lny(l);
    let cnx = lnx(l + 1u); let cny = lny(l + 1u);
    let gc = g.x;
    if (gc >= cnx * cny) { return; }
    let gi = gc % cnx; let gj = gc / cnx;
    let fd = lv(l, 2u); let fae = lv(l, 3u); let fan = lv(l, 4u); let fac = lv(l, 5u);
    var own = 0.0; var ae = 0.0; var an = 0.0; var act = 0.0;
    for (var dj = 0u; dj < 2u; dj++) { for (var di = 0u; di < 2u; di++) {
        let i = 2u * gi + di; let j = 2u * gj + dj;
        if (i >= fnx || j >= fny) { continue; }
        let c = j * fnx + i;
        if (S[fac + c] < 0.5) { continue; }
        act = 1.0;
        var o = S[fd + c];
        if (i + 1u < fnx) { o -= S[fae + c]; }
        if (i > 0u) { o -= S[fae + c - 1u]; }
        if (j + 1u < fny) { o -= S[fan + c]; }
        if (j > 0u) { o -= S[fan + c - fnx]; }
        own += o;
        if (di == 1u && i + 1u < fnx && S[fac + c + 1u] > 0.5) { ae += 0.5 * S[fae + c]; }
        if (dj == 1u && j + 1u < fny && S[fac + c + fnx] > 0.5) { an += 0.5 * S[fan + c]; }
    } }
    S[lv(l + 1u, 2u) + gc] = own;
    S[lv(l + 1u, 3u) + gc] = ae;
    S[lv(l + 1u, 4u) + gc] = an;
    S[lv(l + 1u, 5u) + gc] = act;
}
@compute @workgroup_size(64)
fn k_mg_coarsen_diag(@builtin(global_invocation_id) g: vec3<u32>) {
    if (D[D_NEWTON_DONE] > 0.5) { return; }
    let l = Q.a; // niveau grossier
    let nx = lnx(l); let ny = lny(l);
    let c = g.x;
    if (c >= nx * ny) { return; }
    let d = lv(l, 2u); let ae = lv(l, 3u); let an = lv(l, 4u); let ac = lv(l, 5u);
    if (S[ac + c] < 0.5) { S[d + c] = 1.0; return; }
    let i = c % nx; let j = c / nx;
    var s = max(S[d + c], 0.0);
    if (i + 1u < nx) { s += S[ae + c]; }
    if (i > 0u) { s += S[ae + c - 1u]; }
    if (j + 1u < ny) { s += S[an + c]; }
    if (j > 0u) { s += S[an + c - nx]; }
    S[d + c] = max(s, 1e-30);
}

// ---------------------------------------------------------------- V-cycle, niveaux fins (un lancement par opération ; Q.a = 2·niveau + couleur)
@compute @workgroup_size(64)
fn k_mg_first(@builtin(global_invocation_id) g: vec3<u32>) {
    if (pcg_off()) { return; }
    let l = Q.a >> 1u;
    if (g.x >= lnx(l) * lny(l)) { return; }
    first_cell(l, g.x);
}
@compute @workgroup_size(64)
fn k_mg_smooth(@builtin(global_invocation_id) g: vec3<u32>) {
    if (pcg_off()) { return; }
    let l = Q.a >> 1u;
    let c = color_cell(l, g.x, Q.a & 1u);
    if (c < 0) { return; }
    gs_cell(l, u32(c));
}
@compute @workgroup_size(64)
fn k_mg_restrict(@builtin(global_invocation_id) g: vec3<u32>) {
    if (pcg_off()) { return; }
    let l = Q.a >> 1u;
    if (g.x >= lnx(l + 1u) * lny(l + 1u)) { return; }
    restrict_cell(l, g.x);
}
@compute @workgroup_size(64)
fn k_mg_post1(@builtin(global_invocation_id) g: vec3<u32>) {
    if (pcg_off()) { return; }
    let l = Q.a >> 1u;
    let c = color_cell(l, g.x, 1u);
    if (c < 0) { return; }
    post1_cell(l, u32(c));
}

// ---------------------------------------------------------------- V-cycle, petits niveaux : un seul groupe de travail
fn sync() { storageBarrier(); workgroupBarrier(); }
@compute @workgroup_size(256)
fn k_mg_tail(@builtin(local_invocation_id) lid: vec3<u32>) {
    let off = pcg_off();
    let t0 = P.mg_tail;
    let last = P.nlev - 1u;
    // descente
    for (var l = t0; l < last; l++) {
        let n = lnx(l) * lny(l);
        if (!off) { for (var c = lid.x; c < n; c += 256u) { first_cell(l, c); } }
        sync();
        for (var h = 0u; h < 3u; h++) {
            let color = (h + 1u) % 2u;
            if (!off) { for (var c = lid.x; c < n; c += 256u) { let i = c % lnx(l); let j = c / lnx(l); if ((i + j) % 2u == color) { gs_cell(l, c); } } }
            sync();
        }
        let nc = lnx(l + 1u) * lny(l + 1u);
        if (!off) { for (var c = lid.x; c < nc; c += 256u) { restrict_cell(l, c); } }
        sync();
    }
    // grille la plus grossière : 20 × (rouge, noir, noir, rouge) depuis x = 0
    let n = lnx(last) * lny(last);
    for (var it = 0u; it < 20u; it++) {
        for (var h = 0u; h < 4u; h++) {
            var color = 0u; if (h == 1u || h == 2u) { color = 1u; }
            if (!off) {
                for (var c = lid.x; c < n; c += 256u) {
                    let i = c % lnx(last); let j = c / lnx(last);
                    if ((i + j) % 2u == color) {
                        if (it == 0u && h == 0u) { first_cell(last, c); } else { gs_cell(last, c); }
                    } else if (it == 0u && h == 0u) {
                        S[lv(last, 6u) + c] = 0.0;
                    }
                }
            }
            sync();
        }
    }
    // remontée
    for (var q = 0u; q < last - t0; q++) {
        let l = last - 1u - q;
        let nf = lnx(l) * lny(l);
        if (!off) { for (var c = lid.x; c < nf; c += 256u) { let i = c % lnx(l); let j = c / lnx(l); if ((i + j) % 2u == 1u) { post1_cell(l, c); } } }
        sync();
        // puis rouge, noir, rouge
        for (var h = 0u; h < 3u; h++) {
            if (!off) { for (var c = lid.x; c < nf; c += 256u) { let i = c % lnx(l); let j = c / lnx(l); if ((i + j) % 2u == h % 2u) { gs_cell(l, c); } } }
            sync();
        }
    }
}

// ---------------------------------------------------------------- gradient conjugué (niveau 0)
// D_RZ + (it mod 2) : r·z de l'itération it (double tampon : lu et écrit dans des lancements différents)
fn apply_a(c: u32, xo: u32) -> f32 {
    let i = c % P.nx; let j = c / P.nx;
    return S[O_sd + c] * S[xo + c] - mg_off(0u, i, j, c, xo);
}
// réduit dans CHAQUE groupe de travail les sommes partielles de la quantité 0 (64 fils) — résultat identique partout
var<workgroup> shg: array<f32, 64>;
fn wg_total(lid: u32) -> f32 {
    var acc = 0.0;
    for (var w = lid; w < nwg_cells(); w += 64u) { acc += S[O_part + w]; }
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
@compute @workgroup_size(64)
fn k_pcg_init(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let c = g.x;
    var v: array<f32, 8>;
    v[0] = 0.0;
    if (c < ncell() && D[D_NEWTON_DONE] < 0.5) {
        S[O_cx + c] = 0.0;
        let b = S[O_rhs + c];
        S[O_cr + c] = b;
        v[0] = b * b;
    }
    wg_reduce(v, 1u, 0u, l.x, w.x, 0u);
}
@compute @workgroup_size(256)
fn k_pcg_bb(@builtin(local_invocation_id) l: vec3<u32>) {
    let bb = final_reduce(0u, nwg_cells(), 0u, l.x);
    if (l.x == 0u) {
        D[D_BB] = bb;
        D[D_PCG_ITN] = 0.0;
        if (bb <= 0.0) { D[D_PCG_DONE] = 1.0; } else { D[D_PCG_DONE] = 0.0; }
    }
}
// sommes partielles de r·z
@compute @workgroup_size(64)
fn k_pcg_rz(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let c = g.x;
    var v: array<f32, 8>;
    v[0] = 0.0;
    if (c < ncell() && !pcg_off()) { v[0] = S[O_cr + c] * S[O_cz + c]; }
    wg_reduce(v, 1u, 0u, l.x, w.x, 0u);
}
// direction initiale p = z, r·z initial dans D_RZ + 1 (lu par l'itération 0)
@compute @workgroup_size(64)
fn k_pcg_p0(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let rz = wg_total(l.x);
    let c = g.x;
    if (pcg_off()) { return; }
    if (c == 0u) { D[D_RZ + 1u] = rz; }
    if (c < ncell()) { S[O_cp + c] = S[O_cz + c]; }
}
@compute @workgroup_size(64)
fn k_pcg_apply(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let c = g.x;
    var v: array<f32, 8>;
    v[0] = 0.0;
    if (c < ncell() && !pcg_off()) {
        let a = apply_a(c, O_cp);
        S[O_cap + c] = a;
        v[0] = S[O_cp + c] * a;
    }
    wg_reduce(v, 1u, 0u, l.x, w.x, 0u);
}
// α = rz / pAp (réduit localement), x += α p, r −= α A p, sommes partielles de r·r (quantité 1)
@compute @workgroup_size(64)
fn k_pcg_update(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let pap = wg_total(l.x);
    let c = g.x;
    var v: array<f32, 8>;
    v[0] = 0.0;
    let off = pcg_off() || abs(pap) < 1e-30;
    if (c < ncell() && !off) {
        let a = D[D_RZ + ((Q.a + 1u) & 1u)] / pap;
        S[O_cx + c] += a * S[O_cp + c];
        let r = S[O_cr + c] - a * S[O_cap + c];
        S[O_cr + c] = r;
        v[0] = r * r;
    }
    if (c == 0u && !pcg_off() && abs(pap) < 1e-30) { D[D_PCG_DONE] = 1.0; }
    wg_reduce(v, 1u, 0u, l.x, w.x, 1u);
}
@compute @workgroup_size(256)
fn k_pcg_rr(@builtin(local_invocation_id) l: vec3<u32>) {
    let rr = final_reduce(1u, nwg_cells(), 0u, l.x);
    if (l.x == 0u && !pcg_off()) {
        D[D_RR] = rr;
        D[D_PCG_ITN] += 1.0;
        if (sqrt(rr) <= P.ptol * sqrt(D[D_BB])) { D[D_PCG_DONE] = 1.0; }
    }
}
// β = rz_nouveau / rz, p = z + β p ; écrit rz_nouveau dans D_RZ + (it mod 2)
@compute @workgroup_size(64)
fn k_pcg_dir(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let rz2 = wg_total(l.x);
    let c = g.x;
    if (pcg_off()) { return; }
    let rz = D[D_RZ + ((Q.a + 1u) & 1u)];
    let be = rz2 / max(rz, 1e-30);
    if (c == 0u) { D[D_RZ + (Q.a & 1u)] = rz2; }
    if (c < ncell()) { S[O_cp + c] = S[O_cz + c] + be * S[O_cp + c]; }
}
