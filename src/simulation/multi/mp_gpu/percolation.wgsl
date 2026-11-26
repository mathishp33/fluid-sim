// ===== Percolation (miroir de mp_percolation.rs) : le fluide traverse les lits de grains perméables posés sur un appui et les grilles,
// loi d'Ergun résolue dans la projection (conductance C/dx² dans k_stencil, divergence de q dans k_rhs), transport du fluide seul au pas
// suivant (k_perc_*). Faces : q = −C (p_b − p_a)/dx + d, de a (gauche / bas) vers b. Lits perméables à deux pressions : p (prel) y est
// la pression du fluide des pores, σ' (sig) celle des contacts, résolue une fois par pas (k_sk_*) ; le mélange est poussé par p + σ'.

const ERGUN_A: f32 = 150.0;
const ERGUN_B: f32 = 1.75;
// cellule de grains non poreuse d'au moins α_max − SEAL_GAP de grains : close au fluide des pores (miroir de mp_percolation::SEAL_GAP)
const SEAL_GAP: f32 = 0.2;

fn is_grille(c: u32) -> bool { return S[O_grl_d + c] > 0.0; }
fn perc_on() -> bool { return has(F_PERC); }

// fluide (phases non granulaires) de la cellule : (ρ, μ, fraction) ; (0, 0, 0) sans fluide
fn pore_fluid(c: u32) -> vec3<f32> {
    var a = 0.0; var m = 0.0; var mu = 0.0;
    for (var k = 0u; k < P.np; k++) {
        let ak = max(alpha(k, c), 0.0);
        let ok = !gran_k(k) && ak > 0.0;
        a += select(0.0, ak, ok);
        m += select(0.0, max(mass(k, c), 0.0), ok);
        mu += select(0.0, ak * mu_k(k, ptemp(k, c)), ok);
    }
    if (a <= 1e-12) { return vec3<f32>(0.0, 0.0, 0.0); }
    return vec3<f32>(m / a, mu / a, a);
}
fn kozeny(d: f32, eps: f32) -> f32 { return d * d * eps * eps * eps / (ERGUN_A * (1.0 - eps) * (1.0 - eps)); }

// milieu poreux de chaque cellule (k_dc_cell) : K (0 : pas poreux), ε, d, ρ_f, μ_f dans dc_k, dc_e, dc_d, dc_rf, dc_mu
@compute @workgroup_size(64)
fn k_dc_cell(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    S[O_dc_k + c] = 0.0; S[O_dc_rf + c] = 0.0;
    if (is_solid(c)) { return; }
    let pf = pore_fluid(c);
    S[O_dc_rf + c] = pf.x;
    if (pinned(c) || pf.y <= 0.0) { return; }
    var eps = 0.0; var d = 0.0;
    if (is_grille(c)) {
        d = max(S[O_grl_d + c], 1e-5); eps = clamp(S[O_grl_e + c], 0.05, 0.95);
    } else {
        var as_ = 0.0; var ad = 0.0; var am = 0.0;
        for (var k = 0u; k < P.np; k++) {
            let a = select(0.0, max(alpha(k, c), 0.0), gran_k(k));
            as_ += a; ad += a * max(grain_d(k), 1e-6); am += a * grain_amax(k);
        }
        // (le contact — chaîne posée sur un appui — est vérifié dans dc_porous, après k_dc_ps ; lit trop peu perméable : étanche)
        if (as_ <= 1e-6) { return; }
        d = ad / as_; eps = clamp(1.0 - as_, 0.05, 0.95);
        if (kozeny(d, eps) / pf.y < PERC_KMU_MIN) { return; }
    }
    S[O_dc_k + c] = kozeny(d, eps);
    S[O_dc_e + c] = eps; S[O_dc_d + c] = d; S[O_dc_mu + c] = pf.y;
}
// cellule poreuse : grille, ou grains en contact (miroir de porous_cell_with)
fn dc_porous(c: u32) -> bool { return S[O_dc_k + c] > 0.0 && (is_grille(c) || S[O_dc_ct + c] > 0.5); }
// conductance K_eff/μ d'une cellule poreuse pour un gradient moteur |G| (racine de b q² + (μ/K) q = G) ; 1/conductance, 0 si non poreuse
fn dc_inv(c: u32, gm: f32) -> f32 {
    let k = S[O_dc_k + c];
    if (!dc_porous(c)) { return 0.0; }
    let eps = S[O_dc_e + c];
    let a = S[O_dc_mu + c] / k;
    let b = ERGUN_B * S[O_dc_rf + c] * (1.0 - eps) / (eps * eps * eps * S[O_dc_d + c]);
    return 0.5 * (a + sqrt(a * a + 4.0 * b * abs(gm)));
}

// pression de contact lithostatique p_s (un fil par ligne de gravité, miroir de grain_overburden : état d'une cellule qui entre dans un
// lit perméable, k_sk_mark) ; contact dans dc_ct
fn dc_w(c: u32) -> vec3<f32> {
    // (poids déjaugé w, tassé, a des grains)
    var as_ = 0.0; var ms = 0.0; var am = 0.0;
    for (var k = 0u; k < P.np; k++) {
        let gr = gran_k(k);
        let a = select(0.0, max(alpha(k, c), 0.0), gr);
        as_ += a; ms += select(0.0, max(mass(k, c), 0.0), gr); am += a * grain_amax(k);
    }
    if (as_ <= 1e-6 || is_solid(c) || is_grille(c)) { return vec3<f32>(0.0, 0.0, 0.0); }
    let w = max(ms / as_ - S[O_dc_rf + c], 0.0) * as_ * grav_norm() * P.dx;
    return vec3<f32>(w, select(0.0, 1.0, as_ >= am / as_ - 0.1), 1.0);
}
@compute @workgroup_size(64)
fn k_dc_ps(@builtin(global_invocation_id) g: vec3<u32>) {
    let vert = abs(P.gy) >= abs(P.gx);
    var down = P.gx < 0.0; if (vert) { down = P.gy < 0.0; }
    var lines = P.ny; var len = P.nx;
    if (vert) { lines = P.nx; len = P.ny; }
    let l = g.x;
    if (l >= lines) { return; }
    if (grav_norm() == 0.0) {
        // sans gravité : pas de poids ; contact = grains tassés
        for (var t = 0u; t < len; t++) {
            var c = cidx(t, l); if (vert) { c = cidx(l, t); }
            let wv = dc_w(c);
            S[O_dc_ct + c] = select(0.0, 1.0, wv.z > 0.5 && wv.y > 0.5);
            S[O_dc_ps + c] = 0.0;
        }
        return;
    }
    // du bas vers le haut : contact (tassé, ou posé sur un appui : fond, solide, grille, grains tassés en contact)
    var prev_ct = false; var prev_packed = false; var prev_has = false; var prev_block = true;
    for (var t = 0u; t < len; t++) {
        var s_ = t; if (!down) { s_ = len - 1u - t; }
        var c = cidx(s_, l); if (vert) { c = cidx(l, s_); }
        let wv = dc_w(c);
        let support = prev_block || (prev_has && prev_ct && prev_packed);
        // (contact = chaîne posée : une cellule tassée décollée de son appui n'est ni poreuse ni porteuse)
        let ct = wv.z > 0.5 && support;
        S[O_dc_ct + c] = select(0.0, 1.0, ct);
        prev_ct = ct; prev_packed = wv.y > 0.5; prev_has = wv.z > 0.5; prev_block = is_solid(c) || is_grille(c);
    }
    // du haut vers le bas : cumul du poids sur les colonnes en contact
    var acc = 0.0;
    for (var tt = 0u; tt < len; tt++) {
        let t = len - 1u - tt;
        var s_ = t; if (!down) { s_ = len - 1u - t; }
        var c = cidx(s_, l); if (vert) { c = cidx(l, s_); }
        let wv = dc_w(c);
        let ct = S[O_dc_ct + c] > 0.5;
        S[O_dc_ps + c] = select(0.0, acc + 0.5 * wv.x, ct);
        acc = select(0.0, acc + wv.x, ct);
    }
}
// (α_s, α_max) : fraction des grains et compacité maximale de leur mélange (1 sans grains ; miroir de grain_packing)
fn grain_pack(c: u32) -> vec2<f32> {
    var as_ = 0.0; var am = 0.0;
    for (var k = 0u; k < P.np; k++) {
        let a = select(0.0, max(alpha(k, c), 0.0), gran_k(k));
        as_ += a; am += a * grain_amax(k);
    }
    if (as_ <= 1e-6) { return vec2<f32>(as_, 1.0); }
    return vec2<f32>(as_, am / as_);
}
// cellule de grains non poreuse et dense, close au fluide des pores (miroir de sealed)
fn dc_sealed(c: u32) -> bool {
    if (dc_porous(c) || is_solid(c)) { return false; }
    let pk = grain_pack(c);
    return pk.x > 1e-6 && pk.x >= pk.y - SEAL_GAP;
}
// cellule à deux pressions (miroir de skeleton_of) : grains poreux en contact et tassés, hors grilles
fn sk_now(c: u32) -> bool {
    let pk = grain_pack(c);
    return perc_on() && !is_solid(c) && !pinned(c) && !is_grille(c) && dc_porous(c) && pk.x >= pk.y - SKEL_GAP;
}
// pression de contact lithostatique des cellules poreuses à UNE pression (la pression du fluide de leurs pores est p − p_s)
fn dc_ps1(c: u32) -> f32 {
    return select(0.0, S[O_dc_ps + c], dc_porous(c) && !is_grille(c) && !sk_now(c));
}
// conductance et terme moteur de la face entre a et b (composante de gravité gcomp le long de a → b ; moteur : poids du fluide, et
// pression de contact des cellules à une pression)
fn dc_face(a: u32, b: u32, gcomp: f32) -> vec2<f32> {
    let pa = dc_porous(a); let pb = dc_porous(b);
    if ((!pa && !pb) || dc_sealed(a) || dc_sealed(b) || is_solid(a) || is_solid(b) || pinned(a) || pinned(b)) { return vec2<f32>(0.0, 0.0); }
    let ra = S[O_dc_rf + a]; let rb = S[O_dc_rf + b];
    var rf = rb;
    if (ra > 0.0 && rb > 0.0) { rf = 0.5 * (ra + rb); } else if (ra > 0.0) { rf = ra; }
    let drive = rf * gcomp * P.dx + dc_ps1(b) - dc_ps1(a);
    let gm = abs((S[O_prel + b] - S[O_prel + a]) - drive) / P.dx;
    let c = 2.0 / (dc_inv(a, gm) + dc_inv(b, gm));
    return vec2<f32>(c, c * drive / P.dx);
}
@compute @workgroup_size(64)
fn k_dc_faces(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < nfu()) {
        let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
        var r = vec2<f32>(0.0, 0.0);
        if (i > 0u && i < P.nx) { r = dc_face(cidx(i - 1u, j), cidx(i, j), P.gx); }
        S[O_dcu + f] = r.x; S[O_ddu + f] = r.y;
    }
    if (f < nfv()) {
        let i = f % P.nx; let j = f / P.nx;
        var r = vec2<f32>(0.0, 0.0);
        if (j > 0u && j < P.ny) { r = dc_face(cidx(i, j - 1u), cidx(i, j), P.gy); }
        S[O_dcv + f] = r.x; S[O_ddv + f] = r.y;
    }
}
// flux de Darcy des faces pour la pression courante (O_prel)
fn darcy_qu(f: u32) -> f32 {
    let c = S[O_dcu + f];
    if (c <= 0.0) { return 0.0; }
    let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
    return -c * (S[O_prel + cidx(i, j)] - S[O_prel + cidx(i - 1u, j)]) / P.dx + S[O_ddu + f];
}
fn darcy_qv(f: u32) -> f32 {
    let c = S[O_dcv + f];
    if (c <= 0.0) { return 0.0; }
    let i = f % P.nx; let j = f / P.nx;
    return -c * (S[O_prel + cidx(i, j)] - S[O_prel + cidx(i, j - 1u)]) / P.dx + S[O_ddv + f];
}
// divergence du flux de Darcy à la pression courante, et du flux stocké (celui qui a servi au transport de ce pas)
fn darcy_div_now(i: u32, j: u32) -> f32 {
    return (darcy_qu(uidx(i + 1u, j)) - darcy_qu(uidx(i, j)) + darcy_qv(vidx(i, j + 1u)) - darcy_qv(vidx(i, j))) / P.dx;
}
fn darcy_div_old(i: u32, j: u32) -> f32 {
    return (S[O_qu + uidx(i + 1u, j)] - S[O_qu + uidx(i, j)] + S[O_qv + vidx(i, j + 1u)] - S[O_qv + vidx(i, j)]) / P.dx;
}
// flux de Darcy de la pression finale : persistants, transportés au pas suivant
@compute @workgroup_size(64)
fn k_dc_final(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < nfu()) { S[O_qu + f] = darcy_qu(f); }
    if (f < nfv()) { S[O_qv + f] = darcy_qv(f); }
}

// ---------------------------------------------------------------- transport du fluide (miroir de percolate)
fn fluid_frac(c: u32) -> f32 {
    var a = 0.0;
    for (var k = 0u; k < P.np; k++) { a += select(max(alpha(k, c), 0.0), 0.0, gran_k(k)); }
    return a;
}
// sortie totale de chaque cellule (volume rapporté à la cellule) et facteur de limitation (≤ 90 % de son fluide) dans pq_s (part fluide dans pq_a)
@compute @workgroup_size(64)
fn k_perc_out(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    let i = c % P.nx; let j = c / P.nx;
    let dt = dtv();
    let ql = S[O_qu + uidx(i, j)]; let qr = S[O_qu + uidx(i + 1u, j)];
    let qb = S[O_qv + vidx(i, j)]; let qt = S[O_qv + vidx(i, j + 1u)];
    let out = (max(-ql, 0.0) + max(qr, 0.0) + max(-qb, 0.0) + max(qt, 0.0)) * dt / P.dx;
    let af = select(fluid_frac(c), 0.0, is_solid(c));
    S[O_pq_s + c] = select(1.0, 0.9 * af / max(out, 1e-30), out > 0.9 * af);
    S[O_pq_a + c] = af;
}
// incréments de masse et d'énergie de la cellule c (dans sdm / sde, drapeau « touchée » en sdm[np])
@compute @workgroup_size(64)
fn k_perc_delta(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell(); let np = P.np;
    if (c >= n) { return; }
    for (var k = 0u; k <= np; k++) { S[O_sdm + k * n + c] = 0.0; }
    for (var gq = 0u; gq < P.ng; gq++) { S[O_sde + gq * n + c] = 0.0; }
    if (is_solid(c)) { return; }
    let i = c % P.nx; let j = c / P.nx;
    let dt = dtv();
    var touched = false;
    for (var q = 0u; q < 4u; q++) {
        // faces gauche, droite, bas, haut : flux sortant de c positif
        var qf = 0.0; var nb = c; var ok = false;
        if (q == 0u) { ok = i > 0u; qf = -S[O_qu + uidx(i, j)]; nb = select(c, c - 1u, ok); }
        else if (q == 1u) { ok = i + 1u < P.nx; qf = S[O_qu + uidx(i + 1u, j)]; nb = select(c, c + 1u, ok); }
        else if (q == 2u) { ok = j > 0u; qf = -S[O_qv + vidx(i, j)]; nb = select(c, c - P.nx, ok); }
        else { ok = j + 1u < P.ny; qf = S[O_qv + vidx(i, j + 1u)]; nb = select(c, c + P.nx, ok); }
        let live = ok && qf != 0.0;
        let out = qf > 0.0;
        let dn = select(nb, c, out);
        let vol = select(0.0, abs(qf) * dt / P.dx * S[O_pq_s + dn], live);
        let af = S[O_pq_a + dn];
        let use_ = vol > 0.0 && af > 1e-12;
        touched = touched || use_;
        // face gauche ou basse de c suivie par une mesure de débit (comptée une fois, par sa maille haute) : vers − si le fluide sort
        // de c (miroir de count_transfer)
        var fw = 0u;
        if (use_ && q == 0u) { fw = u32(S[O_fw_u + uidx(i, j)]); }
        if (use_ && q == 2u) { fw = u32(S[O_fw_v + vidx(i, j)]); }
        let fb = (max(fw, 1u) - 1u) * 2u * np;
        let psg = select(1.0, -1.0, out);
        let a2 = P.dx * P.dx;
        for (var k = 0u; k < np; k++) {
            let mk = mass(k, dn);
            let take = use_ && !gran_k(k) && mk > 0.0;
            let dm = select(0.0, vol * mk / max(af, 1e-30), take);
            let gk = group_k(k);
            let de = dm * erel_k(k, tgrp(gk, dn));
            S[O_sdm + k * n + c] += select(dm, -dm, out);
            S[O_sde + gk * n + c] += select(de, -de, out);
            if (fw > 0u && take) {
                S[O_flacc + fb + k] += psg * vol * alpha(k, dn) / max(af, 1e-30) * a2;
                S[O_flacc + fb + np + k] += psg * dm * a2;
            }
        }
    }
    S[O_sdm + np * n + c] = select(0.0, 1.0, touched);
}

// ===== lits perméables à deux pressions (miroir de skeleton_transitions, solve_skeleton) =====
// cellules qui entrent dans un lit perméable (σ' = poids déjaugé au-dessus, retiré de p) ou en sortent (σ' rendue à p) ; masque skel
@compute @workgroup_size(64)
fn k_sk_mark(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    let now = sk_now(c);
    let was = S[O_skel + c] > 0.5;
    if (now && !was) {
        let pa = S[O_prel + c] + P.p0;
        let s0 = clamp(S[O_dc_ps + c], 0.0, max(pa - 2.0 * P_FLOOR, 0.0));
        S[O_sig + c] = s0;
        S[O_prel + c] -= s0;
    } else if (!now && was) {
        S[O_prel + c] += S[O_sig + c];
        S[O_sig + c] = 0.0;
    } else if (!now) {
        S[O_sig + c] = 0.0;
    }
    S[O_skel + c] = select(0.0, 1.0, now);
}
// matrice de la correction δσ' (niveau 0 : sd, sae, san, sact) : faces du mélange des cellules skel, Dirichlet vers les autres
@compute @workgroup_size(64)
fn k_sk_stencil(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell() || newton_done()) { return; }
    S[O_sae + c] = 0.0; S[O_san + c] = 0.0; S[O_sd + c] = 1.0; S[O_sact + c] = 0.0;
    if (S[O_skel + c] < 0.5) { return; }
    S[O_sact + c] = 1.0;
    let i = c % P.nx; let j = c / P.nx;
    let dx2 = P.dx * P.dx;
    let dt = dtv();
    let edge = 2.0 * dt / (S[O_rho_mix + c] * dx2);
    var d = 0.0;
    if (i + 1u < P.nx) {
        if (link(c, c + 1u, S[O_pfu_on + uidx(i + 1u, j)] > 0.5) > 0u) {
            let t = dt / (frho_x(i + 1u, j) * dx2);
            d += t;
            if (S[O_skel + c + 1u] > 0.5) { S[O_sae + c] = t; }
        }
    } else if (P.bc_r == 1u) { d += edge; }
    if (i > 0u) {
        if (link(c - 1u, c, S[O_pfu_on + uidx(i, j)] > 0.5) > 0u) { d += dt / (frho_x(i, j) * dx2); }
    } else if (P.bc_l == 1u) { d += edge; }
    if (j + 1u < P.ny) {
        if (link(c, c + P.nx, S[O_pfv_on + vidx(i, j + 1u)] > 0.5) > 0u) {
            let t = dt / (frho_y(i, j + 1u) * dx2);
            d += t;
            if (S[O_skel + c + P.nx] > 0.5) { S[O_san + c] = t; }
        }
    } else if (P.bc_t == 1u) { d += edge; }
    if (j > 0u) {
        if (link(c - P.nx, c, S[O_pfv_on + vidx(i, j)] > 0.5) > 0u) { d += dt / (frho_y(i, j) * dx2); }
    } else if (P.bc_b == 1u) { d += edge; }
    S[O_sd + c] = max(d, 1e-30);
}
// second membre : ∇·u = 0 dans le lit (u : k_face_tmp à la pression courante)
@compute @workgroup_size(64)
fn k_sk_rhs(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell() || newton_done()) { return; }
    if (S[O_skel + c] < 0.5) { S[O_rhs + c] = 0.0; return; }
    let i = c % P.nx; let j = c / P.nx;
    let dv = (S[O_Fx + uidx(i + 1u, j)] - S[O_Fx + uidx(i, j)] + S[O_Fy + vidx(i, j + 1u)] - S[O_Fy + vidx(i, j)]) / P.dx;
    S[O_rhs + c] = -dv;
}
// σ' += δσ' (relâchée à 0 : le lit se décolle) ; |Δσ'| max rapporté au bruit d'arrondi (sommes partielles, quantité 0)
@compute @workgroup_size(64)
fn k_sk_update(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let c = g.x;
    var v: array<f32, 8>;
    v[0] = 0.0;
    if (c < ncell() && !newton_done() && S[O_skel + c] > 0.5) {
        let s0 = S[O_sig + c];
        let s1 = max(s0 + S[O_cx + c], 0.0);
        S[O_sig + c] = s1;
        v[0] = abs(s1 - s0) / max(0.1, 2e-6 * max(S[O_prel + c] + P.p0 + s1, 0.0));
    }
    wg_reduce(v, 1u, 1u, l.x, w.x, 0u);
}
// fin de la pré-résolution d'avant le frottement : itérations comptées, gradient conjugué remis à zéro pour la projection
@compute @workgroup_size(64)
fn k_sk_pre_final(@builtin(local_invocation_id) l: vec3<u32>) {
    if (l.x == 0u) {
        D[D_PCG_IT] += D[D_PCG_ITN];
        D[D_MAXPCG_N] = max(D[D_MAXPCG_N], D[D_PCG_ITN]);
        if (D[D_PCG_DONE] < 0.5) { D[D_PCG_UNCONV] = 1.0; }
        D[D_PCG_DONE] = 0.0; D[D_PCG_ITN] = 0.0;
    }
}
@compute @workgroup_size(256)
fn k_sk_final(@builtin(local_invocation_id) l: vec3<u32>) {
    let mx = final_reduce(0u, nwg_cells(), 1u, l.x);
    if (l.x == 0u && D[D_NEWTON_DONE] < 0.5) {
        D[D_SK_MOVED] = select(0.0, 1.0, mx >= 1.0);
        D[D_PCG_IT] += D[D_PCG_ITN];
        D[D_MAXPCG_N] = max(D[D_MAXPCG_N], D[D_PCG_ITN]);
        if (D[D_PCG_DONE] < 0.5) { D[D_PCG_UNCONV] = 1.0; }
    }
}
