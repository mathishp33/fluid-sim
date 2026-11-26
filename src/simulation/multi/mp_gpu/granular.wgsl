// ===== Phases granulaires (sable, poudre) : sédimentation et échange thermique grains ↔ fluide (miroir de mp_granular.rs) =====
// Écrit sans tableaux privés ni boucles dans des branches divergentes (sélections), comme le reste du solveur (lavapipe).

fn gran_k(k: u32) -> bool { return ph(k, 32u) > 0.5; }
fn grain_d(k: u32) -> f32 { return ph(k, 33u); }
fn grain_amax(k: u32) -> f32 { return ph(k, 34u); }
// groupe dispersé : uniquement des phases granulaires (table)
fn gdisp(g: u32) -> bool { return T[T_GDISP + g] > 0.5; }
// fraction volumique de grains (toutes phases granulaires) et de fluide (le reste) de la cellule c
fn grain_fraction(c: u32) -> f32 {
    var s = 0.0;
    for (var k = 0u; k < P.np; k++) { s += select(0.0, alpha(k, c), gran_k(k)); }
    return s;
}
fn fluid_fraction(c: u32) -> f32 {
    var s = 0.0;
    for (var k = 0u; k < P.np; k++) { s += select(alpha(k, c), 0.0, gran_k(k)); }
    return s;
}
fn grav_norm() -> f32 { return sqrt(P.gx * P.gx + P.gy * P.gy); }
// température du fluide au contact des grains de c (miroir de fluid_contact_temp) : groupes non granulaires présents, pondérés par leur
// fraction volumique ; 0 sans fluide
fn fluid_contact_temp(c: u32) -> f32 {
    var a = 0.0; var at = 0.0;
    for (var g = 0u; g < P.ng; g++) {
        let ok = !gdisp(g) && group_mass(g, c) > 0.0;
        let al = select(0.0, group_alpha(g, c), ok);
        a += al;
        at += al * select(0.0, tgrp(g, c), ok);
    }
    return select(0.0, at / max(a, 1e-30), a > 0.0);
}
// rayonnement dans un tas de grains : conductance de face 4 σ ε_r d T_m³ f_a f_b (miroir de mp_granular::bed_rad_k ; ε = 0.9)
fn bed_rad_k(d: f32, ta: f32, tb: f32, fa: f32, fb: f32) -> f32 {
    let tm3 = (ta * ta + tb * tb) * (ta + tb) * 0.25;
    return 4.0 * 5.670374e-8 * (0.9 / 1.1) * d * tm3 * fa * fb;
}

// vitesse de chute ENTRAVÉE de la phase granulaire s dans la cellule c (> 0 dans le sens de la gravité, < 0 : le grain remonte) :
// Ferguson–Church (C1 = 18, C2 = 0.4), Richardson–Zaki avec l'exposant de Rowe. 0 sans gravité, sans fluide ou sans grains.
fn settling_w(s: u32, c: u32) -> f32 {
    let gn = grav_norm();
    var a = 0.0; var m = 0.0; var mu = 0.0;
    for (var q = 0u; q < P.np; q++) {
        let aq = alpha(q, c);
        let use_ = !gran_k(q) && aq > 0.0;
        a += select(0.0, aq, use_);
        m += select(0.0, max(mass(q, c), 0.0), use_);
        mu += select(0.0, aq * mu_k(q, ptemp(q, c)), use_);
    }
    let d = grain_d(s);
    let rho_f = m / max(a, 1e-30);
    let mu_f = mu / max(a, 1e-30);
    let ok = gn > 0.0 && alpha(s, c) > 0.0 && a > 1e-12 && m > 0.0 && d > 0.0 && mu_f > 0.0;
    let rf = max(rho_f, 1e-30);
    let rho_s = rho_k(s, S[O_prel + c], ptemp(s, c));
    let r = (rho_s - rf) / rf;
    let nu = max(mu_f, 1e-30) / rf;
    let wt = r * gn * d * d / max(18.0 * nu + sqrt(0.75 * 0.4 * abs(r) * gn * d * d * d), 1e-30);
    let x = abs(wt) * d / nu;
    let re = select(0.0, pow(max(x, 1e-30), 0.75), x > 0.0);
    let nn = (4.7 + 0.1175 * re) / (1.0 + 0.0175 * re);
    let b = clamp(1.0 - grain_fraction(c), 0.0, 1.0);
    let h = select(0.0, pow(max(b, 1e-30), nn), b > 0.0);
    return select(0.0, wt * h, ok);
}

// masse volumique du fluide porteur de c (contenu non granulaire, miroir de carrier_fluid) ; 0 sans fluide
fn carrier_rho(c: u32) -> f32 {
    var a = 0.0; var m = 0.0;
    for (var q = 0u; q < P.np; q++) {
        let aq = alpha(q, c);
        let use_ = !gran_k(q) && aq > 0.0;
        a += select(0.0, aq, use_);
        m += select(0.0, max(mass(q, c), 0.0), use_);
    }
    return select(0.0, m / max(a, 1e-30), a > 1e-12 && m > 0.0);
}

// volume de grains s qui traverse la face (c0 | c1) (c1 : voisine d'indice supérieur le long de l'axe, composante de gravité ga)
// pendant dt, évalué sur l'état d'avant l'opérateur : > 0 de c0 vers c1, < 0 de c1 vers c0 (miroir de settle_grains)
fn settle_vol(s: u32, c0: u32, c1: u32, ga: f32, share: f32, dt: f32) -> f32 {
    let gn = grav_norm();
    // a : cellule « amont » au sens de la gravité, b : « aval »
    let down = ga < 0.0;
    let a = select(c0, c1, down);
    let b = select(c1, c0, down);
    let movable = !is_solid(a) && !pinned(a) && !is_grille(a) && !is_solid(b) && !pinned(b) && !is_grille(b);
    let wa = settling_w(s, a);
    let wb = settling_w(s, b);
    // grains lourds : de a vers b ; grains légers (w < 0) : de b vers a (vitesse évaluée dans la donneuse)
    let heavy = wa > 0.0;
    let light = !heavy && wb < 0.0;
    let d = select(b, a, heavy);
    let r = select(a, b, heavy);
    let w = select(-wb, wa, heavy);
    let ad = alpha(s, d);
    var vol = ad * w * (abs(ga) / max(gn, 1e-30)) * dt / P.dx;
    vol = min(vol, share * ad);
    vol = min(vol, share * max(grain_amax(s) - grain_fraction(r), 0.0));
    vol = min(vol, share * fluid_fraction(r));
    // un grain léger qui remonte (glace dans l'eau) ne passe pas dans un fluide où il serait lourd (l'air au-dessus de la surface) : il
    // flotte dans la dernière maille d'eau (miroir de settle_grains)
    let rf = carrier_rho(r);
    let rho_g = rho_k(s, S[O_prel + d], ptemp(s, d));
    let keeps = heavy || (rf > 0.0 && rho_g < rf);
    let ok = movable && (heavy || light) && ad > 1e-12 && vol > 1e-14 && keeps;
    return select(0.0, select(-vol, vol, d == c0), ok);
}

// incréments de masse (par phase) et d'énergie (par groupe) de la cellule c dus à la sédimentation de la phase s (= Q.a) : à travers
// chaque face, un volume de grains passe de la donneuse à la receveuse et le même volume de fluide de la receveuse (sa composition,
// ses températures) remonte. Lit l'état sans l'écrire : les incréments sont appliqués par k_settle_apply.
@compute @workgroup_size(64)
fn k_settle_delta(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell(); let np = P.np;
    if (c >= n) { return; }
    let s = Q.a;
    let gs = group_k(s);
    for (var k = 0u; k <= np; k++) { S[O_sdm + k * n + c] = 0.0; }
    for (var gq = 0u; gq < P.ng; gq++) { S[O_sde + gq * n + c] = 0.0; }
    if (is_solid(c)) { return; }
    // gravité oblique : deux familles de faces, chacune limitée à la moitié des stocks
    let share = select(1.0, 0.5, P.gx != 0.0 && P.gy != 0.0);
    let dt = dtv();
    let i = c % P.nx; let j = c / P.nx;
    var touched = false;
    for (var q = 0u; q < 4u; q++) {
        // faces : gauche, droite, bas, haut ; voisine hors domaine → cellule elle-même (face ignorée)
        var nb = c; var ga = P.gx; var lo = false; var ok = false;
        if (q == 0u) { ok = i > 0u; nb = select(c, c - 1u, ok); lo = true; }
        else if (q == 1u) { ok = i + 1u < P.nx; nb = select(c, c + 1u, ok); }
        else if (q == 2u) { ok = j > 0u; nb = select(c, c - P.nx, ok); lo = true; ga = P.gy; }
        else { ok = j + 1u < P.ny; nb = select(c, c + P.nx, ok); ga = P.gy; }
        ok = ok && ga != 0.0;
        let c0 = select(c, nb, lo);
        let c1 = select(nb, c, lo);
        let v = select(0.0, settle_vol(s, c0, c1, ga, share, dt), ok);
        let vol = abs(v);
        let dn = select(c1, c0, v > 0.0);   // donneuse de grains
        let rc = select(c0, c1, v > 0.0);   // receveuse
        let out = dn == c;                  // les grains quittent c (sinon ils y entrent)
        let live = vol > 0.0;
        touched = touched || live;
        // grains : masse volumique des grains de la donneuse (m/α), énergie à la température de leur groupe
        let ms = max(mass(s, dn), 0.0);
        let dm = select(0.0, min(vol * mass(s, dn) / max(alpha(s, dn), 1e-30), ms), live);
        let de = dm * erel_k(s, tgrp(gs, dn));
        S[O_sdm + s * n + c] += select(dm, -dm, out);
        S[O_sde + gs * n + c] += select(de, -de, out);
        // le même volume de fluide de la receveuse remonte (composition et températures de la receveuse)
        let fr = max(fluid_fraction(rc), 1e-30);
        for (var k = 0u; k < np; k++) {
            let use_ = live && !gran_k(k) && alpha(k, rc) > 0.0;
            let mk = max(mass(k, rc), 0.0);
            let dmk = select(0.0, min(vol * mass(k, rc) / fr, mk), use_);
            let gk = group_k(k);
            let dek = dmk * erel_k(k, tgrp(gk, rc));
            S[O_sdm + k * n + c] += select(-dmk, dmk, out);
            S[O_sde + gk * n + c] += select(-dek, dek, out);
        }
    }
    S[O_sdm + np * n + c] = select(0.0, 1.0, touched);
}
@compute @workgroup_size(64)
fn k_settle_apply(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell(); let np = P.np;
    if (c >= n || S[O_sdm + np * n + c] < 0.5) { return; }
    for (var k = 0u; k < np; k++) { S[O_mass + k * n + c] += S[O_sdm + k * n + c]; }
    for (var gq = 0u; gq < P.ng; gq++) { S[O_epsg + gq * n + c] += S[O_sde + gq * n + c]; }
    cell_sync(c);
}

// plus grande vitesse de chute dans la cellule c (borne de pas w·dt ≤ 0.5·dx, hors traces ; miroir de max_settling_speed)
fn max_settling_at(c: u32) -> f32 {
    var w = 0.0;
    for (var s = 0u; s < P.np; s++) {
        let on = gran_k(s) && alpha(s, c) > SETTLE_TRACE;
        w = max(w, select(0.0, abs(settling_w(s, c)), on));
    }
    return w;
}

// échange thermique grains ↔ fluides par la surface des grains (miroir de exchange_grains) : G = λ 12 α_s k_f / d² au prorata du
// volume de chaque fluide ; implicite à deux corps, énergie conservée. λ = ∞ : k_exchange égalise déjà tout.
@compute @workgroup_size(64)
fn k_exch_grains(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n || is_solid(c) || P.phase_exch <= 0.0 || P.phase_exch >= 1e37) { return; }
    var af = 0.0; var gr = false; var fl = false;
    for (var gq = 0u; gq < P.ng; gq++) {
        let disp = gdisp(gq);
        let present = group_mass(gq, c) > 0.0;
        af += select(group_alpha(gq, c), 0.0, disp);
        gr = gr || (disp && present);
        fl = fl || (!disp && present);
    }
    if (af <= 1e-12 || !gr || !fl) { return; }
    let dt = dtv();
    // températures et capacités d'avant l'échange (T_g n'est pas modifiée avant cell_sync)
    for (var s = 0u; s < P.ng; s++) {
        let sok = gdisp(s) && group_mass(s, c) > 0.0;
        let d = max(grain_d(gmember(s, 0u)), 1e-6);
        let as_ = group_alpha(s, c);
        let ts = tgrp(s, c);
        let cs = max(group_cap_at(s, c, ts), 1e-30);
        for (var f = 0u; f < P.ng; f++) {
            let fok = sok && !gdisp(f) && group_mass(f, c) > 0.0;
            let tf = tgrp(f, c);
            let gg = P.phase_exch * 12.0 * as_ * group_kcond(f, c) / (d * d) * group_alpha(f, c) / af;
            let cf = max(group_cap_at(f, c, tf), 1e-30);
            let q = select(0.0, gg * dt * (tf - ts) / (1.0 + gg * dt * (1.0 / cs + 1.0 / cf)), fok);
            S[O_epsg + s * n + c] += q;
            S[O_epsg + f * n + c] -= q;
        }
    }
    // rayonnement entre deux populations de grains de la cellule (miroir de exchange_grains), températures et capacités d'avant l'échange
    var a_s = 0.0;
    for (var s = 0u; s < P.ng; s++) { a_s += select(0.0, group_alpha(s, c), gdisp(s)); }
    for (var s1 = 0u; s1 < P.ng; s1++) {
        for (var s2 = 0u; s2 < P.ng; s2++) {
            let ok = s2 > s1 && gdisp(s1) && gdisp(s2) && group_mass(s1, c) > 0.0 && group_mass(s2, c) > 0.0;
            let t1 = tgrp(s1, c); let t2 = tgrp(s2, c);
            let a1 = group_alpha(s1, c); let a2 = group_alpha(s2, c);
            let d1 = max(grain_d(gmember(s1, 0u)), 1e-6); let d2 = max(grain_d(gmember(s2, 0u)), 1e-6);
            let ia = 1.0 / max(a_s, 1e-30);
            let area = min(6.0 * a1 / d1 * a2 * ia, 6.0 * a2 / d2 * a1 * ia);
            let gq = P.phase_exch * 4.0 * 5.670374e-8 * (0.9 / 1.1) * (t1 * t1 + t2 * t2) * (t1 + t2) * 0.25 * area;
            let c1 = max(group_cap_at(s1, c, t1), 1e-30); let c2 = max(group_cap_at(s2, c, t2), 1e-30);
            let q = select(0.0, gq * dt * (t2 - t1) / (1.0 + gq * dt * (1.0 / c1 + 1.0 / c2)), ok);
            S[O_epsg + s1 * n + c] += q;
            S[O_epsg + s2 * n + c] -= q;
        }
    }
    cell_sync(c);
}

// ===== gel / fusion (miroir de MultiphaseSim::fusion_step) : méthode enthalpique à la température des groupes =====
// liquide sous T_f : gèle juste assez pour que sa chaleur latente le ramène à T_f (glace née à T_f) ; solide au-dessus : fond juste assez
// pour revenir à T_f ; tout gèle / fond si la chaleur latente n'y suffit pas (le reste d'énergie du groupe vidé passe à l'autre).
// Énergies relatives à P.t0 : ε_g = E_g − Σ m h0
fn fuq(q: u32, k: u32) -> f32 { return T[T_FU + q * FU_STRIDE + k]; }
@compute @workgroup_size(64)
fn k_fusion(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n) { return; }
    let live = !is_solid(c) && !pinned(c);
    var touched = false;
    for (var q = 0u; q < NFU; q++) {
        let l = u32(fuq(q, 0u)); let s = u32(fuq(q, 1u)); let tm = fuq(q, 2u);
        let gl = group_k(l); let gs = group_k(s);
        let el_m = eabs_k(l, tm); let es_m = eabs_k(s, tm);
        let lf = max(el_m - es_m, 1.0);
        // gel : chaleur qui ramènerait le groupe du liquide à T_f (énergie relative du groupe à T_f, masses courantes)
        var ef = 0.0;
        for (var k = 0u; k < P.np; k++) { ef += select(0.0, max(mass(k, c), 0.0) * erel_k(k, tm), group_k(k) == gl); }
        let ml = mass(l, c);
        // germe (surfusion) : glace dans la cellule, front de dendrites arrivé, germination spontanée ; sans surfusion, partout
        let dtn = fuq(q, 4u);
        let seeded = dtn <= 0.0 || alpha(s, c) >= FU_SEED || S[O_fu_phi + q * n + c] >= 1.0 || tm - tgrp(gl, c) > dtn;
        let frz = live && ml > 1e-12 && tgrp(gl, c) < tm && seeded;
        let d1 = select(0.0, clamp((ef - S[O_epsg + gl * n + c]) / lf, 0.0, ml), frz);
        S[O_mass + l * n + c] -= d1;
        S[O_mass + s * n + c] += d1;
        S[O_epsg + gl * n + c] -= d1 * (es_m - h0_k(l));
        S[O_epsg + gs * n + c] += d1 * (es_m - h0_k(s));
        // fusion (groupe granulaire : cette seule phase ; glace née ci-dessus comprise)
        let ms = mass(s, c);
        let mlt = live && ms > 1e-12 && tgrp(gs, c) > tm;
        let d2 = select(0.0, clamp((S[O_epsg + gs * n + c] - ms * (es_m - h0_k(s))) / lf, 0.0, ms), mlt);
        S[O_mass + s * n + c] -= d2;
        S[O_mass + l * n + c] += d2;
        S[O_epsg + gs * n + c] -= d2 * (el_m - h0_k(s));
        S[O_epsg + gl * n + c] += d2 * (el_m - h0_k(l));
        let dm = d1 - d2;
        let act = d1 != 0.0 || d2 != 0.0;
        // groupe vidé : le reste de son énergie (la chaleur qui n'a pas suffi à tout geler / fondre) passe à l'autre
        var mgl = 0.0;
        for (var k = 0u; k < P.np; k++) { mgl += select(0.0, max(mass(k, c), 0.0), group_k(k) == gl); }
        let empty_l = act && mgl <= 0.0;
        let empty_s = act && !empty_l && mass(s, c) <= 0.0;
        let eps_l = S[O_epsg + gl * n + c];
        let eps_s = S[O_epsg + gs * n + c];
        S[O_epsg + gs * n + c] = select(select(eps_s, 0.0, empty_s), eps_s + eps_l, empty_l);
        S[O_epsg + gl * n + c] = select(select(eps_l, eps_l + eps_s, empty_s), 0.0, empty_l);
        S[O_acc_fu + q * n + c] += dm * P.dx * P.dx;
        touched = touched || act;
    }
    if (touched) { cell_sync(c); }
}

// ===== surfusion : front de dendrites (miroir de MultiphaseSim::fusion_germ, advance_fusion_front, front_seeds) =====
// germe de la paire q dans la cellule b : cristaux du solide s (fraction ≥ FU_SEED), ou du solide là où le front est passé (front ≥ 1)
fn fu_is_germ(q: u32, s: u32, b: u32) -> bool {
    return alpha(s, b) >= FU_SEED || (S[O_fu_phi + q * ncell() + b] >= 1.0 && mass(s, b) > 1e-12);
}
// germe dans la cellule (i, j) du domaine, hors solides ; flags : lu dans fu_germ (relevé par k_fusion_germ avant k_fusion_front),
// sinon évalué sur place
fn fu_germ_at(q: u32, s: u32, i: i32, j: i32, flags: bool) -> bool {
    if (i < 0 || j < 0 || i >= i32(P.nx) || j >= i32(P.ny)) { return false; }
    let b = u32(j) * P.nx + u32(i);
    if (is_solid(b)) { return false; }
    if (flags) { return S[O_fu_germ + q * ncell() + b] > 0.5; }
    return fu_is_germ(q, s, b);
}
// voisines de c qui ont un germe : x = par une face, y = par un coin (0 / 1), z = excédent du front (au-delà de 1) le plus grand parmi
// les voisines par une face
fn fu_seeds(q: u32, s: u32, c: u32, flags: bool) -> vec3<f32> {
    let i = i32(c % P.nx); let j = i32(c / P.nx);
    let o = O_fu_phi + q * ncell();
    let fl = fu_germ_at(q, s, i - 1, j, flags); let fr = fu_germ_at(q, s, i + 1, j, flags);
    let fb = fu_germ_at(q, s, i, j - 1, flags); let ft = fu_germ_at(q, s, i, j + 1, flags);
    let dg = fu_germ_at(q, s, i - 1, j - 1, flags) || fu_germ_at(q, s, i + 1, j - 1, flags) || fu_germ_at(q, s, i - 1, j + 1, flags)
        || fu_germ_at(q, s, i + 1, j + 1, flags);
    var inh = 0.0;
    if (fl) { inh = max(inh, S[o + c - 1u] - 1.0); }
    if (fr) { inh = max(inh, S[o + c + 1u] - 1.0); }
    if (fb) { inh = max(inh, S[o + c - P.nx] - 1.0); }
    if (ft) { inh = max(inh, S[o + c + P.nx] - 1.0); }
    return vec3<f32>(select(0.0, 1.0, fl || fr || fb || ft), select(0.0, 1.0, dg), inh);
}
// vitesse du front de dendrites de la paire q dans un liquide surfondu de dT (m/s)
fn fu_speed(q: u32, dT: f32) -> f32 { return fuq(q, 5u) * pow(max(dT, 1e-6), fuq(q, 6u)); }
// relevé des germes sur l'état d'avant le front (le front n'écrit que les cellules sans germe : ses lectures restent fixes)
@compute @workgroup_size(64)
fn k_fusion_germ(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n) { return; }
    for (var q = 0u; q < NFU; q++) {
        let s = u32(fuq(q, 1u));
        S[O_fu_germ + q * n + c] = select(0.0, 1.0, fu_is_germ(q, s, c));
    }
}
// le front avance de v(ΔT) dt / dx dans une cellule de liquide surfondu sans germe voisine d'un germe (dt / (√2 dx) par un coin) ; la
// cellule germe à 1 et garde l'excédent (≤ 2), repris par ses voisines par une face ; une cellule qui a un germe n'est pas écrite ; le
// front s'efface dans une cellule qui n'est plus surfondue
@compute @workgroup_size(64)
fn k_fusion_front(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n) { return; }
    let live = !is_solid(c) && !pinned(c);
    let dt = D[D_DT];
    for (var q = 0u; q < NFU; q++) {
        let l = u32(fuq(q, 0u)); let s = u32(fuq(q, 1u)); let tm = fuq(q, 2u);
        let dtn = fuq(q, 4u);
        let own = S[O_fu_germ + q * n + c] > 0.5;
        let dT = tm - tgrp(group_k(l), c);
        let nb = fu_seeds(q, s, c, true);
        let v = fu_speed(q, dT);
        let rate = select(select(0.0, v / (1.41421356 * P.dx), nb.y > 0.5), v / P.dx, nb.x > 0.5);
        let o = O_fu_phi + q * n + c;
        let phi = S[o];
        let sup = mass(l, c) > 1e-12 && dT > 0.0;
        let nv = select(0.0, select(phi, min(max(phi, nb.z) + rate * dt, 2.0), rate > 0.0), sup);
        if (live && dtn > 0.0 && !own) { S[o] = nv; }
    }
}
