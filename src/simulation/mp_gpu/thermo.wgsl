// ===== Thermique : sources, conduction, changement de phase, rayonnement (miroir de mp_cells.rs / mp_thermo.rs / mp_species.rs) =====

// ---------------------------------------------------------------- sources : cibles du pinceau, chauffages, solides isothermes
@compute @workgroup_size(64)
fn k_thermal_sources(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    let dt = dtv();
    let a2 = P.dx * P.dx;
    let s = solid_id(c);
    if (s < 0) {
        if (has(F_HEAT)) {
            let tt = S[O_heat_t + c];
            if (tt > 0.0) {
                // chaque groupe tend vers la cible ; puissance totale bornée (miroir de relax_heat_targets)
                let f = one_minus_exp(dt / max(P.heat_tau, 1e-6));
                let lim = P.heat_pmax * dt;
                let n = ncell();
                // (sans tableau privé dans cette branche divergente : apport recalculé au second passage, T_g lue avant écriture)
                var sum_abs = 0.0;
                for (var gq = 0u; gq < P.ng; gq++) {
                    let tq = tgrp(gq, c);
                    sum_abs += select(0.0, abs(group_cap_at(gq, c, tq) * (tt - tq) * f), group_mass(gq, c) > 0.0);
                }
                var sc = 1.0; if (sum_abs > lim) { sc = lim / sum_abs; }
                for (var gq = 0u; gq < P.ng; gq++) {
                    let tq = tgrp(gq, c);
                    let d = select(0.0, group_cap_at(gq, c, tq) * (tt - tq) * f, group_mass(gq, c) > 0.0) * sc;
                    S[O_epsg + gq * n + c] += d;
                    S[O_acc_ein + c] += d * a2;
                }
                cell_sync(c);
            }
        }
        return;
    }
    let kind = S[O_th_kind + c];
    if (kind > 1.5) {
        let w = S[O_th_val + c];
        S[O_eps + c] += w * dt;
        S[O_acc_ein + c] += w * dt * a2;
    } else if (kind > 0.5) {
        let tt = S[O_th_val + c];
        S[O_eps + c] = solid_rhocp(u32(s)) * (tt - P.t0);
        S[O_temp + c] = tt;
    }
}

// ---------------------------------------------------------------- conduction explicite à sous-pas entre NŒUDS (cellule solide = nœud 0 ;
// cellule fluide : un nœud par groupe présent), couplage des groupes g (cellule a) et h (cellule b) sur la fraction de face α_g,a·α_h,b
// (miroir de MultiphaseSim::conduct)
fn ndx(g: u32, c: u32) -> u32 { return g * ncell() + c; }
@compute @workgroup_size(64)
fn k_cond_prep(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let c = g.x;
    var v: array<f32, 8>;
    v[0] = 0.0;
    if (c < ncell()) {
        for (var gq = 0u; gq < P.ng; gq++) { S[O_agc + ndx(gq, c)] = 0.0; S[O_kc2 + ndx(gq, c)] = 0.0; S[O_capc + ndx(gq, c)] = 0.0; }
        let s = solid_id(c);
        if (s >= 0) {
            let kc = S[O_k_cell + c];
            let cap = heat_cap(c);
            S[O_agc + c] = 1.0; S[O_kc2 + c] = kc; S[O_capc + c] = cap; S[O_tc + c] = temp_of(c);
            if (cap > 0.0) { v[0] = kc / cap; }
        } else {
            let ff = max(S[O_flame + c], 1.0);
            for (var gq = 0u; gq < P.ng; gq++) {
                let t = tgrp(gq, c);
                S[O_tc + ndx(gq, c)] = t;
                if (group_mass(gq, c) > 0.0) {
                    let a = group_alpha(gq, c);
                    let kc = group_kcond(gq, c) * ff;
                    let cap = group_cap_at(gq, c, t);
                    S[O_agc + ndx(gq, c)] = a; S[O_kc2 + ndx(gq, c)] = kc; S[O_capc + ndx(gq, c)] = cap;
                    if (cap > 0.0 && a > 0.0) { v[0] = max(v[0], kc * a / cap); }
                }
            }
        }
    }
    wg_reduce(v, 1u, 1u, l.x, w.x, 0u);
}
@compute @workgroup_size(256)
fn k_cond_nsub(@builtin(local_invocation_id) l: vec3<u32>) {
    let amax = final_reduce(0u, nwg_cells(), 1u, l.x);
    if (l.x == 0u) {
        var ns = 0.0;
        if (amax > 0.0) { ns = clamp(ceil(dtv() * amax / (0.2 * P.dx * P.dx)), 1.0, 2000.0); }
        D[D_SUB_C] = ns;  // besoin (relu par l'hôte pour le budget du pas suivant)
        if (ns > f32(P.nsub_cond_max)) { D[D_REDO] = 1.0; ns = 0.0; }
        D[D_NSUB_C] = ns;
    }
}
// sous-pas Q.a : températures des nœuds en alternance tc → tn (pair) / tn → tc (impair)
fn cond_off() -> bool { return f32(Q.a) >= D[D_NSUB_C]; }
@compute @workgroup_size(64)
fn k_cond_step(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell() || cond_off()) { return; }
    var src = O_tc; var dst = O_tn;
    if ((Q.a & 1u) == 1u) { src = O_tn; dst = O_tc; }
    let i = c % P.nx; let j = c / P.nx;
    let ds = dtv() / D[D_NSUB_C];
    let dx2 = P.dx * P.dx;
    let s = solid_id(c);
    if (s >= 0 && S[O_th_kind + c] > 0.5 && S[O_th_kind + c] < 1.5) {
        // solide isotherme : source / puits idéal, T imposée à chaque sous-pas
        let t = S[O_th_val + c];
        S[O_eps + c] = solid_rhocp(u32(s)) * (t - P.t0);
        S[dst + c] = t;
        return;
    }
    var ngc = P.ng; if (s >= 0) { ngc = 1u; }
    // boucles à borne uniforme (P.ng) et sélections plutôt que « continue » (lavapipe) ; nœud absent : flux nul
    for (var gq = 0u; gq < P.ng; gq++) {
        let on = gq < ngc && S[O_agc + ndx(gq, c)] > 0.0;
        let aa = select(0.0, S[O_agc + ndx(gq, c)], on);
        let ka = S[O_kc2 + ndx(gq, c)];
        let tcv = S[src + ndx(gq, c)];
        var de = 0.0;
        for (var q = 0u; q < 4u; q++) {
            var ii = i32(i); var jj = i32(j);
            if (q == 0u) { ii += 1; } else if (q == 1u) { ii -= 1; } else if (q == 2u) { jj += 1; } else { jj -= 1; }
            let inside = ii >= 0 && jj >= 0 && ii < i32(P.nx) && jj < i32(P.ny);
            let b = cidx(u32(clamp(ii, 0, i32(P.nx) - 1)), u32(clamp(jj, 0, i32(P.ny) - 1)));
            var ngb = P.ng; if (is_solid(b)) { ngb = 1u; }
            for (var h = 0u; h < P.ng; h++) {
                let ab = select(0.0, S[O_agc + ndx(h, b)], inside && h < ngb && S[O_agc + ndx(h, b)] > 0.0);
                let kb = S[O_kc2 + ndx(h, b)];
                var kl = 0.0; if (ka + kb > 0.0) { kl = 2.0 * ka * kb / (ka + kb); }
                de += kl * aa * ab * (S[src + ndx(h, b)] - tcv) / dx2;
            }
        }
        if (gq < ngc) {
            if (s >= 0) { S[O_eps + c] += ds * de; } else { S[O_epsg + ndx(gq, c)] += ds * de; }
            S[dst + ndx(gq, c)] = select(tcv, tcv + ds * de / max(S[O_capc + ndx(gq, c)], 1e-30), on);
        }
    }
}
@compute @workgroup_size(64)
fn k_cond_finish(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    let ns = u32(D[D_NSUB_C]);
    if (c >= ncell() || ns == 0u) { return; }
    var src = O_tc; if ((ns & 1u) == 1u) { src = O_tn; }
    if (is_solid(c)) { S[O_temp + c] = S[src + c]; return; }
    cell_sync(c);
}

// ---------------------------------------------------------------- échange entre groupes d'une cellule mixte (miroir de mp_groups.rs)
fn lam_inf() -> bool { return P.phase_exch >= 1e37; }
fn vapor_locked(g: u32, c: u32) -> bool {
    let mg = group_mass(g, c);
    var r = false;
    for (var q = 0u; q < P.npc; q++) {
        let liq = u32(pcq(q, 0u)); let vap = u32(pcq(q, 1u));
        r = r || (mg > 0.0 && group_k(vap) == g && group_k(liq) != g && mass(vap, c) > 0.5 * mg && group_mass(group_k(liq), c) > 0.0);
    }
    return r;
}
// conductance G_g (≥ 1e30 : couplage parfait)
fn exch_g(g: u32, c: u32) -> f32 {
    if (P.phase_exch > 0.0 && has(F_PC) && vapor_locked(g, c)) { return 1e30; }
    let a = max(group_alpha(g, c), 1e-6);
    return P.phase_exch * 2.0 * group_kcond(g, c) / (a * P.dx * P.dx);
}
fn thermal_coupling(c: u32, dt: f32) -> f32 {
    if (P.ng < 2u || is_solid(c)) { return 1.0; }
    // sélections plutôt que « continue » (lavapipe) ; les traces (capacité < TRACE_CAP de la cellule) ne comptent pas (miroir CPU)
    var ctot = 0.0;
    for (var g = 0u; g < P.ng; g++) { ctot += select(0.0, group_cap_at(g, c, tgrp(g, c)), group_mass(g, c) > 0.0); }
    var present = 0u; var chi = 0.0;
    for (var g = 0u; g < P.ng; g++) {
        let pres = group_mass(g, c) > 0.0 && group_cap_at(g, c, tgrp(g, c)) >= TRACE_CAP * ctot;
        present += select(0u, 1u, pres);
        let cg = max(group_cap_at(g, c, tgrp(g, c)), 1e-30);
        let gk = exch_g(g, c);
        let a = dt * min(gk, 1e29) / cg;
        let x = select(a / (1.0 + a), 1.0, lam_inf() || gk >= 1e29 || a > 1e30);
        chi = max(chi, select(0.0, x, pres));
    }
    return select(chi, 1.0, present < 2u);
}
@compute @workgroup_size(64)
fn k_exchange(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell() || is_solid(c) || P.ng < 2u || P.phase_exch <= 0.0) { return; }
    let n = ncell();
    let dt = dtv();
    // pas de sortie anticipée « un seul groupe présent » (lavapipe plante sur les boucles qu'aucun fil d'un groupe de travail ne
    // parcourt) : avec un seul groupe, T_i = T_g et rien ne change. Pas de tableaux privés : C_g et G_g recalculés au second
    // passage (ils ne dépendent que des masses et de T_g, lue avant d'être écrite)
    var num = 0.0; var den = 0.0; var ce = 0.0; var cs = 0.0;
    for (var gq = 0u; gq < P.ng; gq++) {
        let pres = group_mass(gq, c) > 0.0;
        let t = tgrp(gq, c);
        let ck = select(0.0, max(group_cap_at(gq, c, t), 1e-30), pres);
        ce += ck * t; cs += ck;
        let gk = exch_g(gq, c);
        let bk = select(gk / (1.0 + dt * gk / max(ck, 1e-30)), ck / dt, gk >= 1e29);
        num += select(0.0, bk * t, pres && !lam_inf()); den += select(0.0, bk, pres && !lam_inf());
    }
    var ti = 0.0;
    if (lam_inf()) { ti = ce / max(cs, 1e-30); } else { ti = num / max(den, 1e-30); }
    let ok = select(den > 0.0, cs > 0.0, lam_inf());
    for (var gq = 0u; gq < P.ng; gq++) {
        let pres = group_mass(gq, c) > 0.0 && ok;
        let t = tgrp(gq, c);
        let ck = max(group_cap_at(gq, c, t), 1e-30);
        let gk = exch_g(gq, c);
        let a2 = dt * min(gk, 1e29) / ck;
        let tn = select(ti, (t + a2 * ti) / (1.0 + a2), !lam_inf() && gk < 1e29);
        S[O_epsg + gq * n + c] += select(0.0, ck * (tn - t), pres);
        S[O_tg + gq * n + c] = select(t, tn, pres);
    }
}

// ---------------------------------------------------------------- changement de phase
fn pcq(q: u32, k: u32) -> f32 { return T[T_PC + q * PC_STRIDE + k]; }
fn tsat_q(q: u32, p_abs: f32) -> f32 {
    // T_sat(p) par dichotomie sur p_sat (monotone), bornes [100, 1500] K
    var lo = 200.0; var hi = 640.0;
    for (var it = 0; it < 40; it++) {
        let mid = 0.5 * (lo + hi);
        if (psat_q(q, mid) < p_abs) { lo = mid; } else { hi = mid; }
    }
    return 0.5 * (lo + hi);
}
fn latent_q(q: u32, t: f32) -> f32 { return hmass_k(u32(pcq(q, 1u)), t) - hmass_k(u32(pcq(q, 0u)), t); }
// moles des autres gaz du groupe de la vapeur (loi de Dalton) ; -1 si la vapeur n'est miscible à aucun gaz
fn n_others(vap: u32, c: u32) -> f32 {
    let gi = group_k(vap);
    let cnt = gcount(gi);
    var any = false;
    var s = 0.0;
    for (var q = 0u; q < cnt; q++) {
        let k = gmember(gi, q);
        if (k != vap && is_gas(k)) { any = true; s += max(mass(k, c), 0.0) / molar_k(k); }
    }
    if (!any) { return -1.0; }
    return s;
}
// équilibre de pression partielle (Dalton) dans la cellule : dm > 0 évaporation
fn pp_equilibrium(c: u32, q: u32, n_o: f32, ml: f32, mv: f32, al: f32, dt: f32) -> f32 {
    let liq = u32(pcq(q, 0u)); let vap = u32(pcq(q, 1u));
    let t = ptemp(liq, c); let p = S[O_prel + c] + P.p0;
    var cap = heat_cap(c);
    if (ml > 1e-12) { cap = latent_cap(c, liq, t, dt); }
    cap = max(cap, 1e-30);
    let lat_u = max(eabs_k(vap, t) - eabs_k(liq, t), 1.0);
    let mvm = molar_k(vap);
    let f0 = pp_f(q, 0.0, t, lat_u, cap, mv, mvm, n_o, p);
    var lo = 0.0; var hi = 0.0;
    if (f0 < 0.0 && ml > 1e-12 && al > 1e-6) { lo = 0.0; hi = ml; }
    else if (f0 > 0.0 && mv > 1e-12 && (al > 1e-6 || f0 > 0.05 * psat_q(q, t))) { lo = -mv; hi = 0.0; }
    else { return 0.0; }
    if (pp_f(q, lo, t, lat_u, cap, mv, mvm, n_o, p) >= 0.0) { return lo; }
    if (pp_f(q, hi, t, lat_u, cap, mv, mvm, n_o, p) <= 0.0) { return hi; }
    var a = lo; var b = hi;
    for (var it = 0; it < 40; it++) {
        let m = 0.5 * (a + b);
        if (pp_f(q, m, t, lat_u, cap, mv, mvm, n_o, p) < 0.0) { a = m; } else { b = m; }
    }
    return 0.5 * (a + b);
}
fn pp_f(q: u32, dm: f32, t: f32, lat_u: f32, cap: f32, mv: f32, mvm: f32, n_o: f32, p: f32) -> f32 {
    let tn = max(t - lat_u * dm / cap, 150.0);
    let nv = max((mv + dm) / mvm, 0.0);
    return nv / (nv + n_o) * p - psat_q(q, tn);
}
// transfert liquide → vapeur dm (< 0 : condensation) ; entre groupes, le LIQUIDE fournit / reçoit la chaleur latente (miroir de
// MultiphaseSim::transfer). En énergies relatives : Δε_l = −dm (e_x − h0_l), Δε_v = dm (e_x − h0_v), e_x = énergie absolue massique
// de la masse déplacée (vapeur formée à t_vap ; vapeur qui se condense à T_v ; brouillard : liquide à T_v)
// chaleur latente partagée comme dans latent_cap : liquide C_l, autres groupes χ·C_g (miroir de transfer). Le partage est appliqué
// AVANT le déplacement des masses (capacités d'avant le transfert), sans tableau privé (appelé dans des branches divergentes)
fn do_transfer(c: u32, q: u32, dm: f32, t_vap: f32, chi: f32) {
    let n = ncell();
    let liq = u32(pcq(q, 0u)); let vap = u32(pcq(q, 1u));
    let gl = group_k(liq); let gv = group_k(vap);
    let ml0 = mass(liq, c);
    if (gl != gv) {
        let tv = tgrp(gv, c);
        let tl = tgrp(gl, c);
        var exl = 0.0; var exv = 0.0;   // e_x − h0_l, e_x − h0_v
        let fog = dm < 0.0 && ml0 < -dm;
        // volume créé / détruit par le transfert (miroir de transfer : pc_dvol) — rendu au groupe du liquide par k_pressure_work
        let pr = S[O_prel + c];
        let dvol = dm * (1.0 / rho_k(vap, pr, select(tv, t_vap, dm > 0.0)) - 1.0 / rho_k(liq, pr, select(tl, tv, fog)));
        S[O_pc_dv + gl * n + c] += dvol;
        if (dm > 0.0) { let r = erel_k(vap, t_vap); exl = r + pcq(q, 7u); exv = r; }
        else if (fog) { let r = erel_k(liq, tv); exl = r; exv = r - pcq(q, 7u); }
        else { let r = erel_k(vap, tv); exl = r + pcq(q, 7u); exv = r; }
        // part latente prise au (> 0) / donnée au (< 0) liquide au-delà de son énergie sensible
        let lat = select(dm * (exl - erel_k(liq, tl)), 0.0, fog || chi <= 0.0);
        var ctot = group_cap_at(gl, c, tl);
        for (var g = 0u; g < P.ng; g++) {
            ctot += select(0.0, chi * group_cap_at(g, c, tgrp(g, c)), g != gl && group_mass(g, c) > 0.0 && !latent_excluded(g, c));
        }
        var back = 0.0;
        for (var g = 0u; g < P.ng; g++) {
            let w = select(0.0, chi * group_cap_at(g, c, tgrp(g, c)) / max(ctot, 1e-30), g != gl && group_mass(g, c) > 0.0 && !latent_excluded(g, c));
            let qg = lat * w;
            S[O_epsg + g * n + c] -= qg;
            back += qg;
        }
        S[O_epsg + gl * n + c] += back - dm * exl;
        S[O_epsg + gv * n + c] += dm * exv;
    } else {
        S[O_epsg + gl * n + c] -= dm * pcq(q, 7u);   // Δε = dm (h0_l − h0_v)
        let pr = S[O_prel + c]; let tq = tgrp(gl, c);
        S[O_pc_dv + gl * n + c] += dm * (1.0 / rho_k(vap, pr, select(tq, t_vap, dm > 0.0)) - 1.0 / rho_k(liq, pr, tq));
    }
    S[O_mass + liq * n + c] -= dm;
    S[O_mass + vap * n + c] += dm;
    S[O_acc_tr + q * n + c] += dm * P.dx * P.dx;
    cell_sync(c);
}
// volume de la cellule à la pression relative pr, températures de groupe courantes
fn cell_vol_at(c: u32, pr: f32) -> f32 {
    var v = 0.0;
    for (var k = 0u; k < P.np; k++) { let m = mass(k, c); if (m > 0.0) { v += m / rho_k(k, pr, ptemp(k, c)); } }
    return v;
}
// capacité qui fournit la chaleur latente : liquide + χ × autres groupes (miroir de latent_cap)
fn latent_cap(c: u32, liq: u32, t: f32, dt: f32) -> f32 {
    let gl = group_k(liq);
    let cl = group_cap_at(gl, c, t);
    var oth = 0.0;
    for (var g = 0u; g < P.ng; g++) { oth += select(0.0, group_cap_at(g, c, tgrp(g, c)), g != gl && !latent_excluded(g, c)); }
    return cl + thermal_coupling(c, dt) * oth;
}
// vapeur verrouillée sur son liquide (λ fini) : hors du partage de la chaleur latente (miroir de latent_excluded)
fn latent_excluded(g: u32, c: u32) -> bool { return !lam_inf() && has(F_PC) && vapor_locked(g, c); }
@compute @workgroup_size(64)
fn k_phase_change(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let c = g.x; let n = ncell();
    var dtp = 3.0e38;
    let q = Q.a;
    if (c < n && !is_solid(c) && !pinned(c)) {
        let dt = dtv();
        loop {
            let liq = u32(pcq(q, 0u)); let vap = u32(pcq(q, 1u));
            let ml = mass(liq, c); let mv = mass(vap, c);
            // température du LIQUIDE (son groupe)
            let t = ptemp(liq, c); let pr = S[O_prel + c]; let p = pr + P.p0;
            let al = alpha(liq, c); let av = alpha(vap, c);
            var aog = 0.0;
            for (var k = 0u; k < P.np; k++) { if (k != liq && k != vap && is_gas(k)) { aog += alpha(k, c); } }
            let n_o = n_others(vap, c);
            if (n_o < 0.0 && mv <= 1e-12 && psat_q(q, t) <= p) { break; }
            if (n_o >= 0.0) {
                if (n_o > 1e-9 && psat_q(q, t) < 0.98 * p && (ml > 1e-12 || mv > 1e-12)) {
                    let dmp = pp_equilibrium(c, q, n_o, ml, mv, al, dt);
                    if (dmp != 0.0) { var chi = 1.0; if (ml > 1e-12) { chi = thermal_coupling(c, dt); } do_transfer(c, q, dmp, t, chi); }
                    break;
                }
            }
            if (ml <= 1e-12 && mv <= 1e-12) { break; }
            let tsat = tsat_q(q, p);
            let cap = latent_cap(c, liq, t, dt);
            // estimation de la masse à changer : L dans le domaine de p_sat (≤ 640 K), bornée par le bas (h_v − h_l < 0 vers 1450 K)
            let latent = max(latent_q(q, clamp(t, 200.0, 640.0)), 0.05 * pcq(q, 2u));
            let rho_v = rho_k(vap, pr, t);
            let rho_l = rho_k(liq, pr, t);
            var dm = 0.0;
            var bulk = false;
            let relax = one_minus_exp(dt / max(P.pc_tau, 1e-12));
            let rate_cap = P.pc_vol_lim * dt / max(P.pc_tau, 1e-12);   // débit volumique ≤ pc_vol_lim / pc_tau
            if (t > tsat && ml > 1e-12) {
                let dm_e = min(cap * (t - tsat) / latent * relax, ml);
                let gas = av + aog;
                let resolved = gas > 0.02 || (al < 0.98 && gas > 0.0 && aog > 0.0);
                if (resolved && al > 1e-6) {
                    dm = dm_e;
                } else if (av >= 1e-7 && aog < 1e-9) {
                    let rb = pow(av / max(1.0 - av, 1e-9) * 3.0 / (4.0 * PI * P.nuclei), 1.0 / 3.0);
                    let ps = psat_q(q, t);
                    let rate = rho_v * rho_l / S[O_rho_mix + c] * av * (1.0 - av) * 3.0 / max(rb, 1e-9) * sqrt(2.0 / 3.0 * max(ps - p, 0.0) / rho_l);
                    let excess = t - tsat_q(q, max(p, 101325.0)) - P.boil_dT;
                    var thermal = 0.0;
                    if (excess > 0.0) { thermal = min(cap * excess / latent * relax, ml); }
                    bulk = excess > 0.0;
                    dm = min(max(rate * dt, thermal), dm_e);
                } else if (av < 1e-7 && aog < 1e-9 && t - tsat > P.nuc_dT) {
                    dm = min(P.nuc_alpha * rho_v, dm_e);
                }
            } else if (t < tsat && mv > 1e-12 && al > 1e-6) {
                let dm_e = min(cap * (tsat - t) / latent * relax, mv);
                let gas = av + aog;
                if (gas > 0.02 || aog > 0.0) {
                    dm = -dm_e;
                } else if (av > 1e-7) {
                    let rb = pow(av / max(1.0 - av, 1e-9) * 3.0 / (4.0 * PI * P.nuclei), 1.0 / 3.0);
                    let ps = psat_q(q, t);
                    let rate = rho_v * rho_l / S[O_rho_mix + c] * av * (1.0 - av) * 3.0 / max(rb, 1e-9) * sqrt(2.0 / 3.0 * max(p - ps, 0.0) / rho_l);
                    dm = -min(min(rate * dt, dm_e), mv);
                }
            }
            // cavité discrète : la pression a été bloquée au plancher p_sat au pas précédent
            var cavity = 0.0;
            if (ml > 1e-12 && al > 1e-3 && dm >= 0.0) {
                let ps = psat_q(q, t);
                if (S[O_clamped + c] > 0.5 && p <= 1.01 * ps) {
                    let vv = cell_vol_at(c, ps - P.p0);
                    if (vv < 1.0) {
                        let den = max(1.0 / rho_k(vap, ps - P.p0, t) - 1.0 / rho_k(liq, ps - P.p0, t), 1e-12);
                        cavity = min((1.0 - vv) / den, ml - dm);
                        // même débit volumique maximal que la vaporisation (miroir du CPU)
                        cavity = min(cavity, rate_cap / den);
                        let dvc = cavity * den;
                        if (dvc > 1e-3) { dtp = min(dtp, 0.5 * P.pc_vol_lim * dt / dvc); }
                    }
                }
            }
            if (dm != 0.0) {
                let ps = psat_q(q, t);
                let vv = cell_vol_at(c, ps - P.p0);
                let denom = 1.0 / rho_k(vap, ps - P.p0, t) - 1.0 / rho_k(liq, ps - P.p0, t);
                let pe = max(p, ps);
                let dv_unit = max(1.0 / rho_k(vap, pe - P.p0, t) - 1.0 / rho_k(liq, pe - P.p0, t), 1e-12);
                if (dm > 0.0) {
                    let by_volume = P.pc_vol_lim / dv_unit;
                    let clo = max((1.0 - vv) / denom, 0.0);
                    var lim = clo;
                    if (al > 0.98 && bulk) { lim = max(clo, by_volume); }
                    dm = min(dm, lim);
                    dm = min(dm, rate_cap / dv_unit);
                    let dv = dm * dv_unit;
                    if (dv > 1e-3) { dtp = min(dtp, 0.5 * P.pc_vol_lim * dt / dv); }
                } else {
                    dm = -min(-dm, max((vv - 1.0) / denom, 0.0));
                    dm = -min(-dm, min(P.pc_vol_lim, rate_cap) / dv_unit);
                    let dv = -dm * dv_unit;
                    if (dv > 1e-3) { dtp = min(dtp, 0.5 * P.pc_vol_lim * dt / dv); }
                }
            }
            var dtot = dm + max(cavity, 0.0);
            // la chaleur latente prélevée ne fait pas tomber T de plus de la moitié de l'écart à T_PW_MIN
            // borné par le groupe participant le plus froid (miroir du CPU ; boucle à borne uniforme, appartenance par sélection)
            let chi = thermal_coupling(c, dt);
            let glq = group_k(liq);
            var head = t - T_PW_MIN;
            let share = chi > 0.0 && glq != group_k(vap);
            for (var gq = 0u; gq < P.ng; gq++) {
                let part = share && gq != glq && group_mass(gq, c) > 0.0 && !latent_excluded(gq, c);
                head = select(head, min(head, tgrp(gq, c) - T_PW_MIN), part);
            }
            if (dtot > 0.0) { dtot = min(dtot, 0.5 * cap * max(head, 0.0) / latent); }
            if (dtot != 0.0) { do_transfer(c, q, dtot, min(tsat, t), chi); }
            break;
        }
    }
    var v: array<f32, 8>;
    v[0] = -dtp;
    wg_reduce(v, 1u, 1u, l.x, w.x, 0u);
}
@compute @workgroup_size(256)
fn k_phase_final(@builtin(local_invocation_id) l: vec3<u32>) {
    let m = -final_reduce(0u, nwg_cells(), 1u, l.x);
    if (l.x == 0u) { D[D_DTPHASE_NEW] = min(D[D_DTPHASE_NEW], max(m, 1e-8)); }
}
// échange à travers une interface alignée sur la grille (Dalton) — passe A : demandes des cellules de gaz (paire Q.a)
@compute @workgroup_size(64)
fn k_surf_req(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n) { return; }
    S[O_sx_dm + c] = 0.0; S[O_sx_tgt + c] = -1.0;
    let q = Q.a;
    let liq = u32(pcq(q, 0u)); let vap = u32(pcq(q, 1u));
    if (is_solid(c) || pinned(c) || alpha(liq, c) > 1e-6) { return; }
    let n_o = n_others(vap, c);
    if (n_o <= 1e-9) { return; }
    let i = c % P.nx; let j = c / P.nx;
    var lc = -1;
    for (var s = 0u; s < 4u; s++) {
        var b = -1;
        if (s == 0u && i > 0u) { b = i32(c - 1u); }
        if (s == 1u && i + 1u < P.nx) { b = i32(c + 1u); }
        if (s == 2u && j > 0u) { b = i32(c - P.nx); }
        if (s == 3u && j + 1u < P.ny) { b = i32(c + P.nx); }
        if (b < 0) { continue; }
        let bu = u32(b);
        if (is_solid(bu) || pinned(bu) || alpha(liq, bu) <= 0.5) { continue; }
        var ag_b = 0.0;
        for (var k = 0u; k < P.np; k++) { if (is_gas(k)) { ag_b += alpha(k, bu); } }
        if (ag_b < 1e-3) { lc = b; break; }
    }
    if (lc < 0) { return; }
    let l = u32(lc);
    let p = S[O_prel + c] + P.p0;
    let gl = group_k(liq); let gv = group_k(vap);
    let tg = tgrp(gv, c); let mv = mass(vap, c); let ml = mass(liq, l);
    let ev_rel = erel_k(vap, tg);
    let ev_abs_minus_h0l = ev_rel + pcq(q, 7u);         // e_v(T_g) − h0_l
    let mvm = molar_k(vap);
    // groupe du liquide après l'échange : ε_l − dm (e_v(T_g) − h0_l), masse m_l − dm
    var cs = vec2<f32>(0.0, 0.0);
    for (var k = 0u; k < P.np; k++) { let m = select(0.0, max(mass(k, l), 0.0), group_k(k) == gl); cs.x += m * ce_k(k); cs.y += m * slope_k(k); }
    let ce_l = ce_k(liq); let sl_l = slope_k(liq);
    let el = epsg(gl, l);
    // f(dm) = x_v p − p_sat(T_liq(dm))
    let t0l = P.t0 + tau_from(cs.x, cs.y, el);
    let f0 = mv / mvm / (mv / mvm + n_o) * p - psat_q(q, t0l);
    if (abs(f0) < 1e-3 * max(psat_q(q, tg), 1.0) || psat_q(q, t0l) >= 0.98 * p) { return; }
    var lo = -mv; var hi = 0.0;
    if (f0 < 0.0) { lo = 0.0; hi = 0.2 * ml; }
    var a = lo; var b = hi;
    var dm = 0.0;
    let flo = surf_f(q, lo, cs, ce_l, sl_l, el, ev_abs_minus_h0l, mv, mvm, n_o, p);
    let fhi = surf_f(q, hi, cs, ce_l, sl_l, el, ev_abs_minus_h0l, mv, mvm, n_o, p);
    if (flo >= 0.0) { dm = lo; }
    else if (fhi <= 0.0) { dm = hi; }
    else {
        for (var it = 0; it < 40; it++) {
            let m = 0.5 * (a + b);
            if (surf_f(q, m, cs, ce_l, sl_l, el, ev_abs_minus_h0l, mv, mvm, n_o, p) < 0.0) { a = m; } else { b = m; }
        }
        dm = 0.5 * (a + b);
    }
    if (dm == 0.0) { return; }
    S[O_sx_dm + c] = dm; S[O_sx_tgt + c] = f32(l); S[O_sx_e + c] = ev_rel;
}
fn surf_f(q: u32, dm: f32, cs: vec2<f32>, ce_l: f32, sl_l: f32, el: f32, de_unit: f32, mv: f32, mvm: f32, n_o: f32, p: f32) -> f32 {
    let tl = max(P.t0 + tau_from(cs.x - dm * ce_l, cs.y - dm * sl_l, el - dm * de_unit), 150.0);
    let nv = max((mv + dm) / mvm, 0.0);
    return nv / (nv + n_o) * p - psat_q(q, tl);
}
// passe B : la cellule de gaz reçoit sa vapeur, la cellule liquide fournit masse et chaleur latente
@compute @workgroup_size(64)
fn k_surf_apply(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n) { return; }
    let q = Q.a;
    let liq = u32(pcq(q, 0u)); let vap = u32(pcq(q, 1u));
    var touched = false;
    let dm = S[O_sx_dm + c];
    let gl = group_k(liq); let gv = group_k(vap);
    if (dm != 0.0) {
        S[O_mass + vap * n + c] += dm;
        S[O_epsg + gv * n + c] += dm * S[O_sx_e + c];
        S[O_acc_tr + q * n + c] += dm * P.dx * P.dx;
        touched = true;
    }
    let i = c % P.nx; let j = c / P.nx;
    for (var s = 0u; s < 4u; s++) {
        var b = -1;
        if (s == 0u && i > 0u) { b = i32(c - 1u); }
        if (s == 1u && i + 1u < P.nx) { b = i32(c + 1u); }
        if (s == 2u && j > 0u) { b = i32(c - P.nx); }
        if (s == 3u && j + 1u < P.ny) { b = i32(c + P.nx); }
        if (b < 0) { continue; }
        let bu = u32(b);
        if (S[O_sx_tgt + bu] == f32(c) && S[O_sx_dm + bu] != 0.0) {
            let d2 = S[O_sx_dm + bu];
            S[O_mass + liq * n + c] -= d2;
            S[O_epsg + gl * n + c] -= d2 * (S[O_sx_e + bu] + pcq(q, 7u));
            touched = true;
        }
    }
    if (touched) { cell_sync(c); }
}

// ---------------------------------------------------------------- rayonnement optiquement mince (H2O, CO2)
fn planck(id: u32, t_in: f32) -> f32 {
    let t = clamp(t_in, 500.0, 2500.0);
    let x = 1000.0 / t;
    var r = 0.0;
    if (id == 1u) { r = -0.23093 + x * (-1.12390 + x * (9.41530 + x * (-2.99880 + x * (0.51382 + x * (-1.86840e-5))))); }
    else if (id == 2u) { r = 18.741 + x * (-121.310 + x * (273.500 + x * (-194.050 + x * (56.310 + x * (-5.8169))))); }
    return max(r, 0.0);
}
@compute @workgroup_size(64)
fn k_radiate(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell() || P.rad_on == 0u || is_solid(c) || pinned(c)) { return; }
    let ta = P.t_amb;
    let n = ncell();
    var touched = false;
    // chaque groupe gazeux contenant un émetteur rayonne à SA température (miroir de MultiphaseSim::radiate) ; boucles à borne
    // uniforme (P.np, appartenance par sélection) et sans « continue » (lavapipe)
    let patm = (S[O_prel + c] + P.p0) / 101325.0;
    for (var gq = 0u; gq < P.ng; gq++) {
        let t = tgrp(gq, c);
        var ngas = 0.0; var agas = 0.0; var has_rad = false; var kn = 0.0;
        for (var k = 0u; k < P.np; k++) {
            let sel = group_k(k) == gq && is_gas(k);
            let nk = select(0.0, max(mass(k, c), 0.0) / molar_k(k), sel);
            ngas += nk;
            agas += select(0.0, alpha(k, c), sel);
            let id = rad_k(k);
            has_rad = has_rad || (sel && id > 0u);
            if (sel && id > 0u) { kn += nk * planck(id, t); }
        }
        let emits = abs(t - ta) >= 50.0 && group_mass(gq, c) > 0.0 && has_rad && ngas > 0.0;
        let kap = select(0.0, kn / max(ngas, 1e-30) * patm, emits);
        if (kap > 0.0) {
            let qr = 4.0 * 5.670374419e-8 * kap * (t * t * t * t - ta * ta * ta * ta) * agas;
            let lim = 0.5 * group_cap_at(gq, c, t) * abs(t - ta);
            let de = clamp(qr * dtv(), -lim, lim);
            S[O_epsg + gq * n + c] -= de;
            S[O_acc_rad + c] += de * P.dx * P.dx;
            touched = true;
        }
    }
    if (touched) { cell_sync(c); }
}
