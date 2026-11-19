// ===== Conduction, changement de phase, travail de pression =====

fn psat_abs(t_in: f32) -> f32 {
    let t = clamp(t_in, 200.0, 640.0);
    let a = (P.pc_lref - P.pc_dcp * P.pc_tref) / P.pc_rv;
    return P.pc_pref * exp(a * (1.0 / P.pc_tref - 1.0 / t) + P.pc_dcp / P.pc_rv * log(t / P.pc_tref));
}
fn tsat_of(p_abs: f32) -> f32 {
    var lo = 200.0; var hi = 640.0;
    for (var it = 0; it < 40; it++) {
        let mid = 0.5 * (lo + hi);
        if (psat_abs(mid) < p_abs) { lo = mid; } else { hi = mid; }
    }
    return 0.5 * (lo + hi);
}
fn latent_of(t: f32) -> f32 { return P.pc_lref + P.pc_dcp * (t - P.pc_tref); }

// volume du mélange, κ_eff (entropie constante) et γ_T = dT/dp adiabatique, pour (pression relative pr, T)
fn closure(c: u32, pr: f32, t: f32) -> vec3<f32> {
    let n = ncell();
    var v = 0.0; var kap = 0.0; var vb = 0.0; var mcp = 0.0;
    for (var k = 0u; k < P.np; k++) {
        let m = mass[k * n + c];
        if (m <= 0.0) { continue; }
        let vk = m / rho_k(k, pr, t);
        v += vk; kap += vk * kappa_k(k, pr); vb += vk * beta_k(k, t); mcp += m * P.ph[k].a.y;
    }
    mcp = max(mcp, 1e-30);
    return vec3<f32>(v, kap - t * vb * vb / mcp, t * vb / mcp);
}

@compute @workgroup_size(64)
fn k_conduct_step(@builtin(global_invocation_id) g: vec3<u32>) {
    // un sous-pas explicite : w1 = ε nouveau (gather : chaque cellule lit ses 4 voisins)
    let c = g.x;
    if (c >= ncell()) { return; }
    let i = c % P.nx; let j = c / P.nx;
    let ds = P.dt / f32(max(P.nsub, 1u));
    let cap_c = max(heat_cap(c), 1e-30);
    let tc = P.t0 + eps[c] / cap_c;
    var de = 0.0;
    for (var q = 0u; q < 4u; q++) {
        var ii = i32(i); var jj = i32(j);
        if (q == 0u) { ii += 1; } else if (q == 1u) { ii -= 1; } else if (q == 2u) { jj += 1; } else { jj -= 1; }
        if (ii < 0 || jj < 0 || ii >= i32(P.nx) || jj >= i32(P.ny)) { continue; }
        let b = cidx(u32(ii), u32(jj));
        let tb = P.t0 + eps[b] / max(heat_cap(b), 1e-30);
        let ka = k_cell[c]; let kb = k_cell[b];
        var kl = 0.0; if (ka + kb > 0.0) { kl = 2.0 * ka * kb / (ka + kb); }
        de += kl * (tb - tc) / (P.dx * P.dx);
    }
    w1[c] = eps[c] + ds * de;
}
@compute @workgroup_size(64)
fn k_conduct_commit(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    eps[c] = w1[c];
    temp[c] = P.t0 + eps[c] / max(heat_cap(c), 1e-30);
}

@compute @workgroup_size(64)
fn k_phase_change(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n || P.pc_on == 0u || is_solid(c)) { return; }
    let liq = P.pc_liq; let vap = P.pc_vap;
    let ml = mass[liq * n + c]; let mv = mass[vap * n + c];
    if (ml <= 1e-9 && mv <= 1e-9) { return; }
    let t = temp[c]; let pr = prel[c]; let p = pr + P.p0;
    let al = alpha[liq * n + c]; let av = alpha[vap * n + c];
    var ao = 0.0;
    for (var k = 0u; k < P.np; k++) { if (k != liq && k != vap && is_gas(k)) { ao += alpha[k * n + c]; } }
    let tsat = tsat_of(p);
    let cap = heat_cap(c);
    let lat = latent_of(t);
    let rho_v = rho_k(vap, pr, t);
    let rho_l = rho_k(liq, pr, t);
    var dm = 0.0;
    let PI = 3.14159265;
    if (t > tsat && ml > 1e-9) {
        let dm_e = min(cap * (t - tsat) / lat, ml);
        let gas = av + ao;
        let resolved = gas > 0.02 || (al < 0.98 && gas > 0.0 && ao > 0.0);
        if (resolved && al > 1e-6) {
            dm = dm_e;
        } else if (av >= 1e-7 && ao < 1e-9) {
            let rb = pow(av / max(1.0 - av, 1e-9) * 3.0 / (4.0 * PI * P.nuclei), 1.0 / 3.0);
            let ps = psat_abs(t);
            let rate = rho_v * rho_l / rho_mix[c] * av * (1.0 - av) * 3.0 / max(rb, 1e-9) * sqrt(max(2.0 / 3.0 * max(ps - p, 0.0) / rho_l, 0.0));
            var thermal = 0.0; if (ps - p <= 0.0) { thermal = dm_e; }
            dm = min(max(rate * P.dt, thermal), dm_e);
        } else if (av < 1e-7 && ao < 1e-9 && t - tsat > P.nuc_dT) {
            dm = min(P.nuc_alpha * rho_v, dm_e);
        }
    } else if (t < tsat && mv > 1e-9 && al > 1e-6) {
        let dm_e = min(cap * (tsat - t) / lat, mv);
        let gas = av + ao;
        if (gas > 0.02 || ao > 0.0) {
            dm = -dm_e;
        } else if (av > 1e-7) {
            let rb = pow(av / max(1.0 - av, 1e-9) * 3.0 / (4.0 * PI * P.nuclei), 1.0 / 3.0);
            let ps = psat_abs(t);
            let rate = rho_v * rho_l / rho_mix[c] * av * (1.0 - av) * 3.0 / max(rb, 1e-9) * sqrt(max(2.0 / 3.0 * max(p - ps, 0.0) / rho_l, 0.0));
            dm = -min(min(rate * P.dt, dm_e), mv);
        }
    }
    if (dm != 0.0) {
        // couplage implicite local : retour à p = p_sat(T), jamais au-delà
        let ps = psat_abs(t);
        let cl = closure(c, ps - P.p0, t);
        let denom = 1.0 / rho_k(vap, ps - P.p0, t) - 1.0 / rho_k(liq, ps - P.p0, t);
        if (dm > 0.0) { dm = min(dm, max((1.0 - cl.x) / denom, 0.0)); }
        else { dm = -min(-dm, max((cl.x - 1.0) / denom, 0.0)); }
    }
    if (dm != 0.0) {
        mass[liq * n + c] -= dm;
        mass[vap * n + c] += dm;
        eps[c] -= dm * P.pc_h0;
    }
}

@compute @workgroup_size(64)
fn k_pressure_work(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell() || is_solid(c)) { return; }
    let i = c % P.nx; let j = c / P.nx;
    let d = (U[uidx(i + 1u, j)] - U[uidx(i, j)] + V[vidx(i, j + 1u)] - V[vidx(i, j)]) / P.dx;
    eps[c] -= P.dt * (prel[c] + P.p0) * d;
}

@compute @workgroup_size(64)
fn k_save_p(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    p_start[c] = prel[c];
    tst[c] = temp[c];
}
