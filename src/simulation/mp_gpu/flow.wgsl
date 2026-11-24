// ===== Écoulement : propriétés, pas de temps, forces, advection PLIC (groupes miscibles, MUSCL), QDM, pression =====

// pas de temps des noyaux : dt × fraction (sous-pas du bloc thermique ; fraction nulle = D pas encore initialisé → 1)
fn dtv() -> f32 { let fr = D[D_DTF]; return D[D_DT] * select(1.0, fr, fr > 0.0); }

// ---------------------------------------------------------------- refresh : T, α, ρ_mix, μ, k depuis (m, ε, p)
@compute @workgroup_size(64)
fn k_refresh(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    let n = ncell();
    if (c >= n) { return; }
    let s = solid_id(c);
    if (s >= 0) {
        S[O_temp + c] = P.t0 + S[O_eps + c] / solid_rhocp(u32(s));
        S[O_rho_mix + c] = solid_rho(u32(s));
        S[O_k_cell + c] = solid_k(u32(s));
        S[O_mu_cell + c] = 0.0;
        return;
    }
    // énergies / températures des groupes (traces alignées), ε = Σ ε_g, T de mélange
    cell_sync(c);
    let pr = S[O_prel + c];
    var vol = 0.0;
    var vk: array<f32, NPMAX>;
    var mt = 0.0;
    for (var k = 0u; k < P.np; k++) {
        let m = mass(k, c);
        if (m > 0.0) { vk[k] = m / rho_k(k, pr, ptemp(k, c)); } else { vk[k] = 0.0; }
        vol += vk[k];
        mt += m;
    }
    let inv = 1.0 / max(vol, 1e-30);
    var mu = 0.0; var kc = 0.0;
    for (var k = 0u; k < P.np; k++) {
        let a = vk[k] * inv;
        S[O_alpha + k * n + c] = a;
        if (a > 0.0) { let tk = ptemp(k, c); mu += a * mu_k(k, tk); kc += a * cond_k(k, tk); }
    }
    S[O_rho_mix + c] = mt * inv;
    S[O_mu_cell + c] = mu;
    S[O_k_cell + c] = kc;
}

// vitesses du début de pas (second membre de la pression)
@compute @workgroup_size(64)
fn k_copy_un(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < nfu()) { S[O_Un + f] = S[O_U + f]; }
    if (f < nfv()) { S[O_Vn + f] = S[O_V + f]; }
}

// ---------------------------------------------------------------- pas de temps (miroir de choose_dt)
@compute @workgroup_size(64)
fn k_red_dt(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    var v: array<f32, 8>;
    for (var q = 0u; q < 8u; q++) { v[q] = -3.0e38; }
    let i = g.x;
    if (i < nfu()) { v[0] = abs(S[O_U + i]); }
    if (i < nfv()) { v[0] = max(v[0], abs(S[O_V + i])); }
    if (i < ncell() && !is_solid(i)) {
        let r = S[O_rho_mix + i];
        v[1] = -r; v[2] = r;
        let mu = S[O_mu_cell + i];
        if (mu > 0.0) { v[3] = -(0.2 * r * P.dx * P.dx / mu); }
        var sg = 0.0;
        for (var k = 0u; k < P.np; k++) { if (sigma_k(k) > sg && mass(k, i) > 1e-9) { sg = sigma_k(k); } }
        v[4] = sg;
        if (has(F_HEAT)) {
            let tt = S[O_heat_t + i];
            if (tt > 0.0) {
                let cap = max(heat_cap(i), 1e-30);
                v[5] = min(abs(tt - S[O_temp + i]), P.heat_pmax * P.heat_tau / cap);
            }
        }
        if (S[O_flame + i] > 1.0) { v[6] = 1.0; }
        if (has(F_GRAN)) { v[7] = max_settling_at(i); }
        // percolation : |q| dt ≤ 0.4 du fluide de la cellule amont (faces droite et haut de la cellule ; miroir de max_darcy_speed)
        if (perc_on()) {
            let ci = i % P.nx; let cj = i / P.nx;
            if (ci + 1u < P.nx) {
                let q = S[O_qu + uidx(ci + 1u, cj)];
                if (q != 0.0) { let dn = select(i + 1u, i, q > 0.0); v[7] = max(v[7], 1.25 * abs(q) / max(fluid_frac(dn), 0.05)); }
            }
            if (cj + 1u < P.ny) {
                let q = S[O_qv + vidx(ci, cj + 1u)];
                if (q != 0.0) { let dn = select(i + P.nx, i, q > 0.0); v[7] = max(v[7], 1.25 * abs(q) / max(fluid_frac(dn), 0.05)); }
            }
        }
    }
    wg_reduce(v, 8u, 1u, l.x, w.x, 0u);
}
@compute @workgroup_size(256)
fn k_dt_final(@builtin(local_invocation_id) l: vec3<u32>) {
    let nw = (max(max(nfu(), nfv()), ncell()) + 63u) / 64u;
    let umax = final_reduce(0u, nw, 1u, l.x);
    let rmin = -final_reduce(1u, nw, 1u, l.x);
    let rmax = final_reduce(2u, nw, 1u, l.x);
    let dtvis = -final_reduce(3u, nw, 1u, l.x);
    let sig = final_reduce(4u, nw, 1u, l.x);
    let gap = final_reduce(5u, nw, 1u, l.x);
    let flame = final_reduce(6u, nw, 1u, l.x);
    let wset = final_reduce(7u, nw, 1u, l.x);
    if (l.x == 0u) {
        var dt = min(min(P.dt_max, D[D_DTCHEM]), min(D[D_DTPHASE], P.dt_cap));
        // borne des poudres (k_burn_dt, recalculée à chaque pas quand une poudre peut brûler ; 0 = aucune : une série de pas remet à
        // zéro les emplacements D_B_TSUM et suivants)
        if (D[D_DTBURN] > 0.0) { dt = min(dt, D[D_DTBURN]); }
        if (gap > 0.0) { dt = min(dt, max(P.heat_tau * 20.0 / gap, 1e-6)); }
        dt = min(dt, P.cfl * P.dx / max(umax, 1e-9));
        // grains : la dérive explicite ne traverse pas plus d'une demi-maille par pas ; percolation : |q| dt ≤ 0.4 du fluide amont
        if ((has(F_GRAN) || has(F_PERC)) && wset > 0.0) { dt = min(dt, 0.5 * P.dx / wset); }
        // flamme présente (miroir de choose_dt)
        if (has(F_FLAME) && flame > 0.5) { dt = min(dt, P.cfl_flame * P.dx / P.flame_u_floor); }
        let gg = sqrt(P.gx * P.gx + P.gy * P.gy);
        if (gg > 0.0) { dt = min(dt, 0.5 * sqrt(P.dx / gg)); }
        if (dtvis > 0.0 && dtvis < 1e37 && !has(F_IMPVISC)) { dt = min(dt, dtvis); }
        if (has(F_ST) && sig > 0.0 && rmax > 0.0) {
            dt = min(dt, P.cap_safety * sqrt((rmin + rmax) * P.dx * P.dx * P.dx / (4.0 * PI * sig)));
        }
        // croissance du pas bornée à ×2 (miroir de choose_dt, DT_GROWTH) : D[D_DT] est encore le pas précédent
        let prev = D[D_DT];
        if (prev > 0.0) { dt = min(dt, 2.0 * prev); }
        D[D_DT] = dt;
        D[D_DTF] = 1.0;
        D[D_SUB_C] = 0.0; D[D_SUB_S] = 0.0;
        D[D_NEWTON_DONE] = 0.0; D[D_NEWTON_IT] = 0.0; D[D_NEWTON_MAX] = 0.0; D[D_NEWTON_DP] = 0.0; D[D_SK_MOVED] = 0.0;
        D[D_PCG_DONE] = 0.0; D[D_PCG_IT] = 0.0; D[D_PCG_ITN] = 0.0; D[D_MAXPCG_N] = 0.0;
        D[D_DTCHEM_NEW] = 3.0e38; D[D_DTPHASE_NEW] = 3.0e38;
        D[D_REDO] = 0.0; D[D_NSUB_S_MAX] = 0.0; D[D_PCG_UNCONV] = 0.0; D[D_NSUB_C] = 0.0; D[D_NSUB_S] = 0.0;
        D[D_FR_UNCONV] = 0.0; D[D_FR_IT] = 0.0;
    }
}

// ---------------------------------------------------------------- sous-pas du bloc thermique des flammes (miroir de step_raw)
@compute @workgroup_size(64)
fn k_dtf_sub(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x == 0u) { D[D_DTF] = 1.0 / f32(max(P.thermo_k, 1u)); } }
@compute @workgroup_size(64)
fn k_dtf_one(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x == 0u) { D[D_DTF] = 1.0; } }

// ---------------------------------------------------------------- prédicteur-correcteur (miroir de step_pc)
// instantané de l'état persistant (S) et des scalaires (D) au début du pas ; boucle à pas fixe (dispatch de PC_GROUPS groupes, pas SNAP_STRIDE)
const SNAP_STRIDE: u32 = 65536u;   // = 1024 groupes × 64 fils
@compute @workgroup_size(64)
fn k_pc_save(@builtin(global_invocation_id) g: vec3<u32>) {
    for (var i = g.x; i < PERS_LEN; i += SNAP_STRIDE) { S[O_pcs + i] = S[PERS_START + i]; }
    if (g.x < D_LEN) { S[O_pcd + g.x] = D[g.x]; }
}
@compute @workgroup_size(64)
fn k_pc_restore(@builtin(global_invocation_id) g: vec3<u32>) {
    for (var i = g.x; i < PERS_LEN; i += SNAP_STRIDE) { S[PERS_START + i] = S[O_pcs + i]; }
    if (g.x < D_LEN) { D[g.x] = S[O_pcd + g.x]; }
}
// fin de la prédiction : u^{n+1} prédit ; budgets / refus de la prédiction gardés (la restauration de D et k_dt_final du correcteur
// les effaceraient : une prédiction qui dépasse un budget de sous-pas sautait sa diffusion sans que le budget ne grandisse)
@compute @workgroup_size(64)
fn k_pc_keep(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < nfu()) { S[O_Up + f] = S[O_U + f]; }
    if (f < nfv()) { S[O_Vp + f] = S[O_V + f]; }
    if (f == 0u) {
        S[O_pcf + 0u] = D[D_SUB_C]; S[O_pcf + 1u] = D[D_NSUB_S_MAX]; S[O_pcf + 2u] = D[D_REDO];
        S[O_pcf + 3u] = D[D_MAXPCG_N]; S[O_pcf + 4u] = D[D_PCG_UNCONV]; S[O_pcf + 5u] = D[D_BAD];
    }
}
// fin du correcteur : maxima des deux passes
@compute @workgroup_size(64)
fn k_pc_merge(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x == 0u) {
        D[D_SUB_C] = max(D[D_SUB_C], S[O_pcf + 0u]); D[D_NSUB_S_MAX] = max(D[D_NSUB_S_MAX], S[O_pcf + 1u]);
        D[D_REDO] = max(D[D_REDO], S[O_pcf + 2u]); D[D_MAXPCG_N] = max(D[D_MAXPCG_N], S[O_pcf + 3u]);
        D[D_PCG_UNCONV] = max(D[D_PCG_UNCONV], S[O_pcf + 4u]);
    }
}
// correcteur, avant l'advection masse / énergie : U ← (u^n + u^{n+1})/2 (et Un, second membre de la pression), u^n gardé dans Up
@compute @workgroup_size(64)
fn k_pc_adv(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < nfu()) { let un = S[O_U + f]; let ua = 0.5 * (un + S[O_Up + f]); S[O_Up + f] = un; S[O_U + f] = ua; S[O_Un + f] = ua; }
    if (f < nfv()) { let vn = S[O_V + f]; let va = 0.5 * (vn + S[O_Vp + f]); S[O_Vp + f] = vn; S[O_V + f] = va; S[O_Vn + f] = va; }
}
// après l'advection masse / énergie : U ← u^n (quantité de mouvement transportée, flux de masse de l'advection)
@compute @workgroup_size(64)
fn k_pc_back(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < nfu()) { S[O_U + f] = S[O_Up + f]; }
    if (f < nfv()) { S[O_V + f] = S[O_Vp + f]; }
}

// ---------------------------------------------------------------- forces
@compute @workgroup_size(64)
fn k_force_init(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < nfu()) { S[O_Fx + f] = dtv() * P.gx; }
    if (f < nfv()) { S[O_Fy + f] = dtv() * P.gy; }
}
// accélérateurs : moyenne des cellules accélératrices adjacentes à la face
@compute @workgroup_size(64)
fn k_accel(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < nfu()) {
        let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
        if (i >= 1u && i < P.nx) {
            let a = cidx(i - 1u, j); let b = cidx(i, j);
            if (!is_solid(a) && !is_solid(b)) {
                var s = 0.0; var cnt = 0.0;
                if (S[O_acc_on + a] > 0.5) { s += S[O_acc_x + a]; cnt += 1.0; }
                if (S[O_acc_on + b] > 0.5) { s += S[O_acc_x + b]; cnt += 1.0; }
                if (cnt > 0.0) { S[O_Fx + f] += dtv() * s / cnt; }
            }
        }
    }
    if (f < nfv()) {
        let i = f % P.nx; let j = f / P.nx;
        if (j >= 1u && j < P.ny) {
            let a = cidx(i, j - 1u); let b = cidx(i, j);
            if (!is_solid(a) && !is_solid(b)) {
                var s = 0.0; var cnt = 0.0;
                if (S[O_acc_on + a] > 0.5) { s += S[O_acc_y + a]; cnt += 1.0; }
                if (S[O_acc_on + b] > 0.5) { s += S[O_acc_y + b]; cnt += 1.0; }
                if (cnt > 0.0) { S[O_Fy + f] += dtv() * s / cnt; }
            }
        }
    }
}

// tension de surface : w1 = marqueur liquide ℓ, w2 = courbure (1e30 = indéfinie), w3 = ℓ lissé, w4 = σ(cellule)
fn mk(i: i32, j: i32, ci_: u32, cj_: u32) -> f32 {
    let ii = u32(clamp(i, 0, i32(P.nx) - 1)); let jj = u32(clamp(j, 0, i32(P.ny) - 1));
    let c = cidx(ii, jj);
    if (is_solid(c)) { return S[O_w1 + cidx(ci_, cj_)]; }
    return S[O_w1 + c];
}
fn smk(i: i32, j: i32, ci_: u32, cj_: u32) -> f32 {
    let ii = u32(clamp(i, 0, i32(P.nx) - 1)); let jj = u32(clamp(j, 0, i32(P.ny) - 1));
    let c = cidx(ii, jj);
    if (is_solid(c)) { return S[O_w3 + cidx(ci_, cj_)]; }
    return S[O_w3 + c];
}
@compute @workgroup_size(64)
fn k_marker(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n) { return; }
    var l = 0.0; var sg = 0.0;
    if (!is_solid(c)) {
        for (var k = 0u; k < P.np; k++) { if (!is_gas(k)) { l += alpha(k, c); sg += alpha(k, c) * sigma_k(k); } }
    }
    S[O_w1 + c] = clamp(l, 0.0, 1.0);
    if (l < 1e-12) { S[O_w4 + c] = 0.0; } else { S[O_w4 + c] = sg / l; }
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
    S[O_w3 + c] = s / 16.0;
}
fn nrm(i: i32, j: i32, ci_: u32, cj_: u32) -> vec2<f32> {
    let gx = (smk(i + 1, j, ci_, cj_) - smk(i - 1, j, ci_, cj_)) * 0.5;
    let gy = (smk(i, j + 1, ci_, cj_) - smk(i, j - 1, ci_, cj_)) * 0.5;
    let s = max(sqrt(gx * gx + gy * gy), 1e-12);
    return vec2<f32>(-gx / s, -gy / s);
}
@compute @workgroup_size(64)
fn k_curv(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    let iu = c % P.nx; let ju = c / P.nx;
    let i = i32(iu); let j = i32(ju);
    let a = S[O_w1 + c];
    var kap = 1e30;
    if (a > 1e-6 && a < 1.0 - 1e-6) {
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
            kap = (nrm(i + 1, j, iu, ju).x - nrm(i - 1, j, iu, ju).x + nrm(i, j + 1, iu, ju).y - nrm(i, j - 1, iu, ju).y) * 0.5 / P.dx;
        }
    }
    S[O_w2 + c] = kap;
}
fn st_face(a: u32, b: u32) -> f32 {
    let la = S[O_w1 + a]; let lb = S[O_w1 + b];
    let wa = la * (1.0 - la); let wb = lb * (1.0 - lb);
    let w = wa + wb;
    if (w < 1e-12) { return 0.0; }
    var ka = S[O_w2 + a]; if (ka > 1e29) { ka = 0.0; }
    var kb = S[O_w2 + b]; if (kb > 1e29) { kb = 0.0; }
    let kf = (ka * wa + kb * wb) / w;
    let sg = (S[O_w4 + a] * wa + S[O_w4 + b] * wb) / w;
    return sg * kf * (lb - la) / P.dx;
}
@compute @workgroup_size(64)
fn k_force_st(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < nfu()) {
        let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
        if (i >= 1u && i < P.nx) {
            let a = cidx(i - 1u, j); let b = cidx(i, j);
            if (!is_solid(a) && !is_solid(b)) { S[O_Fx + f] += dtv() * st_face(a, b) / frho_x(i, j); }
        }
    }
    if (f < nfv()) {
        let i = f % P.nx; let j = f / P.nx;
        if (j >= 1u && j < P.ny) {
            let a = cidx(i, j - 1u); let b = cidx(i, j);
            if (!is_solid(a) && !is_solid(b)) { S[O_Fy + f] += dtv() * st_face(a, b) / frho_y(i, j); }
        }
    }
}
// viscosité : w2 = τxx, w3 = τyy (centres), w5 = τxy (nœuds)
// plafond de la viscosité traitée explicitement (au-delà : implicite, friction.wgsl), miroir de explicit_mu_cap
// (bornée par la masse volumique de ses faces, ½ (ρ_c + ρ_voisine))
fn mu_exp_cap(c: u32) -> f32 {
    if (!has(F_IMPVISC)) { return 3.0e38; }
    let i = c % P.nx; let j = c / P.nx;
    let rc = S[O_rho_mix + c];
    var rn = rc;
    if (i > 0u && !is_solid(c - 1u)) { rn = min(rn, S[O_rho_mix + c - 1u]); }
    if (i + 1u < P.nx && !is_solid(c + 1u)) { rn = min(rn, S[O_rho_mix + c + 1u]); }
    if (j > 0u && !is_solid(c - P.nx)) { rn = min(rn, S[O_rho_mix + c - P.nx]); }
    if (j + 1u < P.ny && !is_solid(c + P.nx)) { rn = min(rn, S[O_rho_mix + c + P.nx]); }
    return 0.2 * 0.5 * (rc + rn) * P.dx * P.dx / D[D_DT];
}
// plafond de la viscosité explicite d'un nœud : 0.2 ρ_min dx² / dt sur ses quatre cellules (miroir de explicit_node_mu_cap)
fn mu_node_cap(c0: u32, c1: u32, c2: u32, c3: u32) -> f32 {
    if (!has(F_IMPVISC)) { return 3.0e38; }
    let r = min(min(S[O_rho_mix + c0], S[O_rho_mix + c1]), min(S[O_rho_mix + c2], S[O_rho_mix + c3]));
    return 0.2 * r * P.dx * P.dx / D[D_DT];
}
fn mu_exp(c: u32) -> f32 { return min(S[O_mu_cell + c], mu_exp_cap(c)); }
@compute @workgroup_size(64)
fn k_visc_diag(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    let i = c % P.nx; let j = c / P.nx;
    S[O_w2 + c] = 2.0 * mu_exp(c) * (S[O_U + uidx(i + 1u, j)] - S[O_U + uidx(i, j)]) / P.dx;
    S[O_w3 + c] = 2.0 * mu_exp(c) * (S[O_V + vidx(i, j + 1u)] - S[O_V + vidx(i, j)]) / P.dx;
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
            let mu = min(0.25 * (mu_exp(c0) + mu_exp(c1) + mu_exp(c2) + mu_exp(c3)), mu_node_cap(c0, c1, c2, c3));
            t = mu * ((S[O_U + uidx(i, j)] - S[O_U + uidx(i, j - 1u)]) / P.dx + (S[O_V + vidx(i, j)] - S[O_V + vidx(i - 1u, j)]) / P.dx);
        }
    }
    S[O_w5 + q] = t;
}
@compute @workgroup_size(64)
fn k_visc_apply(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    let W = P.nx + 1u;
    if (f < nfu()) {
        let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
        if (i >= 1u && i < P.nx) {
            let a = cidx(i - 1u, j); let b = cidx(i, j);
            if (!is_solid(a) && !is_solid(b)) {
                let fx = (S[O_w2 + b] - S[O_w2 + a]) / P.dx + (S[O_w5 + (j + 1u) * W + i] - S[O_w5 + j * W + i]) / P.dx;
                S[O_Fx + f] += dtv() * fx / frho_x(i, j);
            }
        }
    }
    if (f < nfv()) {
        let i = f % P.nx; let j = f / P.nx;
        if (j >= 1u && j < P.ny) {
            let a = cidx(i, j - 1u); let b = cidx(i, j);
            if (!is_solid(a) && !is_solid(b)) {
                let fy = (S[O_w3 + b] - S[O_w3 + a]) / P.dx + (S[O_w5 + j * W + i + 1u] - S[O_w5 + j * W + i]) / P.dx;
                S[O_Fy + f] += dtv() * fy / frho_y(i, j);
            }
        }
    }
}

// ---------------------------------------------------------------- advection PLIC par groupes miscibles
fn gcount(gi: u32) -> u32 { return u32(T[T_GRP + gi * GRP_STRIDE]); }
fn gmember(gi: u32, q: u32) -> u32 { return u32(T[T_GRP + gi * GRP_STRIDE + 1u + q]); }
fn gorder(pos: u32) -> u32 { return u32(T[T_GORD + pos]); }
fn ag(gi: u32, c: u32) -> f32 { return S[O_ag + gi * ncell() + c]; }

@compute @workgroup_size(64)
fn k_adv_init(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x; let n = ncell();
    if (i < n) {
        var mt = 0.0;
        for (var k = 0u; k < P.np; k++) { mt += mass(k, i); S[O_aw + k * n + i] = alpha(k, i); }
        S[O_mtot_old + i] = mt;
    }
    if (i < nfu()) { S[O_mfx + i] = 0.0; }
    if (i < nfv()) { S[O_mfy + i] = 0.0; }
}
// α normalisé (an) et α des groupes (ag) à partir de α de travail (aw)
@compute @workgroup_size(64)
fn k_an(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n) { return; }
    var s = 0.0;
    for (var k = 0u; k < P.np; k++) { s += clamp(S[O_aw + k * n + c], 0.0, 1.0); }
    s = max(s, 1e-30);
    for (var gi = 0u; gi < P.ng; gi++) { S[O_ag + gi * n + c] = 0.0; }
    for (var k = 0u; k < P.np; k++) {
        let a = clamp(S[O_aw + k * n + c], 0.0, 1.0) / s;
        S[O_an + k * n + c] = a;
        S[O_ag + group_k(k) * n + c] += a;
    }
    for (var gi = 0u; gi < P.ng; gi++) { S[O_ag + gi * n + c] = min(S[O_ag + gi * n + c], 1.0); }
    // grains : fractions des groupes non dispersés RENORMALISÉES hors grains (leurs interfaces), 0 pour les groupes dispersés
    if (has(F_GRAN)) {
        var ad = 0.0;
        for (var gi = 0u; gi < P.ng; gi++) { ad += select(0.0, S[O_ag + gi * n + c], gdisp(gi)); }
        let rr = 1.0 - min(ad, 1.0);
        for (var gi = 0u; gi < P.ng; gi++) {
            S[O_agn + gi * n + c] = select(min(S[O_ag + gi * n + c] / max(rr, 1e-30), 1.0), 0.0, gdisp(gi) || rr <= 1e-10);
        }
    }
}
// fraction du groupe vue par la reconstruction d'interface : renormalisée hors grains s'il y a des phases granulaires
fn agr(gi: u32, c: u32) -> f32 {
    if (has(F_GRAN)) { return S[O_agn + gi * ncell() + c]; }
    return ag(gi, c);
}

fn a_from_vol(m1: f32, m2: f32, v: f32) -> f32 {
    if (v <= 0.0) { return 0.0; }
    if (v >= 1.0) { return 1.0; }
    if (m1 < 1e-6) { return v; }
    if (v <= 0.5 * m1 / m2) { return sqrt(2.0 * m1 * m2 * v); }
    if (v <= 1.0 - 0.5 * m1 / m2) { return v * m2 + 0.5 * m1; }
    return 1.0 - sqrt(2.0 * m1 * m2 * (1.0 - v));
}
fn plic_new(a: f32, gx: f32, gy: f32) -> vec3<f32> {
    var nx = -gx; var ny = -gy;
    let s = abs(nx) + abs(ny);
    if (s < 1e-7) { nx = 0.0; ny = 1.0; } else { nx = nx / s; ny = ny / s; }
    let ax = abs(nx); let ay = abs(ny);
    let aa = a_from_vol(min(ax, ay), max(ax, ay), a);
    return vec3<f32>(nx, ny, aa + min(nx, 0.0) + min(ny, 0.0));
}
fn plic_area(pl: vec3<f32>, x0i: f32, x1i: f32, y0i: f32, y1i: f32) -> f32 {
    var x0 = x0i; var x1 = x1i; var y0 = y0i; var y1 = y1i;
    let neg = min(pl.x, 0.0) + min(pl.y, 0.0);
    if (pl.x < 0.0) { let t = x0; x0 = 1.0 - x1; x1 = 1.0 - t; }
    if (pl.y < 0.0) { let t = y0; y0 = 1.0 - y1; y1 = 1.0 - t; }
    let a = pl.z - neg;
    let mx = abs(pl.x); let my = abs(pl.y);
    if (mx < 1e-4) { return (x1 - x0) * (clamp(a / my, y0, y1) - y0); }
    if (my < 1e-4) { return (y1 - y0) * (clamp(a / mx, x0, x1) - x0); }
    // intégrale de la hauteur de la droite le long de l'axe de plus petite composante normale (pente ≤ 1) : exacte et stable en
    // f32. L'inclusion-exclusion des triangles (Σ ± s²/(2 m_x m_y), celle du CPU en f64) perd ~1 % sur une bande étroite quand
    // l'interface est presque parallèle à un axe (m ~ 1e-3 : termes ~ 400 pour une aire ~ 1e-3)
    if (mx <= my) { return band_area(a, mx, my, x0, x1, y0, y1); }
    return band_area(a, my, mx, y0, y1, x0, x1);
}
// aire de {mu·u + mv·v ≤ a} ∩ [u0, u1] × [v0, v1] (mu, mv > 0) = ∫ (clamp((a − mu u)/mv, v0, v1) − v0) du
fn band_area(a: f32, mu: f32, mv: f32, u0: f32, u1: f32, v0: f32, v1: f32) -> f32 {
    let l = clamp((a - mv * v1) / mu, u0, u1);   // u < l : hauteur pleine
    let r = clamp((a - mv * v0) / mu, u0, u1);   // u > r : rien
    let lin = (r - l) * ((a - mu * 0.5 * (l + r)) / mv - v0);
    return max((l - u0) * (v1 - v0) + lin, 0.0);
}
fn ag_at(gi: u32, i: i32, j: i32, ci_: u32, cj_: u32) -> f32 {
    let ii = u32(clamp(i, 0, i32(P.nx) - 1));
    let jj = u32(clamp(j, 0, i32(P.ny) - 1));
    let c = cidx(ii, jj);
    if (is_solid(c)) { return agr(gi, cidx(ci_, cj_)); }
    return agr(gi, c);
}
fn youngs_g(gi: u32, i: u32, j: u32) -> vec2<f32> {
    let I = i32(i); let J = i32(j);
    let gx = ((ag_at(gi, I + 1, J + 1, i, j) + 2.0 * ag_at(gi, I + 1, J, i, j) + ag_at(gi, I + 1, J - 1, i, j))
            - (ag_at(gi, I - 1, J + 1, i, j) + 2.0 * ag_at(gi, I - 1, J, i, j) + ag_at(gi, I - 1, J - 1, i, j))) / 8.0;
    let gy = ((ag_at(gi, I + 1, J + 1, i, j) + 2.0 * ag_at(gi, I, J + 1, i, j) + ag_at(gi, I - 1, J + 1, i, j))
            - (ag_at(gi, I + 1, J - 1, i, j) + 2.0 * ag_at(gi, I, J - 1, i, j) + ag_at(gi, I - 1, J - 1, i, j))) / 8.0;
    return vec2<f32>(gx, gy);
}
fn van_leer(a: f32, b: f32) -> f32 { if (a * b > 0.0) { return 2.0 * a * b / (a + b); } return 0.0; }
// groupe présent à 99 % dans la cellule (sinon ng = aucun) ; avec des grains : groupe non dispersé à 99 % hors grains
fn single_g(c: u32) -> u32 {
    if (has(F_GRAN)) {
        for (var gi = 0u; gi < P.ng; gi++) { if (!gdisp(gi) && agr(gi, c) > 0.99) { return gi; } }
        return 9999u;
    }
    for (var gi = 0u; gi < P.ng; gi++) { if (ag(gi, c) > 0.99) { return gi; } }
    return 9999u;
}

struct FaceInfo { uf: f32, lo: i32, hi: i32, fi: u32, fj: u32, valid: bool };
fn face_info(f: u32, dir: u32) -> FaceInfo {
    var r: FaceInfo;
    r.valid = false; r.lo = -1; r.hi = -1; r.uf = 0.0;
    var fiw = P.nx + 1u; var fjh = P.ny;
    if (dir == 1u) { fiw = P.nx; fjh = P.ny + 1u; }
    r.fi = f % fiw; r.fj = f / fiw;
    if (r.fj >= fjh) { return r; }
    if (dir == 0u) {
        r.uf = S[O_U + uidx(r.fi, r.fj)];
        if (r.fi > 0u) { r.lo = i32(cidx(r.fi - 1u, r.fj)); }
        if (r.fi < P.nx) { r.hi = i32(cidx(r.fi, r.fj)); }
    } else {
        r.uf = S[O_V + vidx(r.fi, r.fj)];
        if (r.fj > 0u) { r.lo = i32(cidx(r.fi, r.fj - 1u)); }
        if (r.fj < P.ny) { r.hi = i32(cidx(r.fi, r.fj)); }
    }
    if (r.uf == 0.0) { return r; }
    if (r.lo >= 0 && is_solid(u32(r.lo))) { return r; }
    if (r.hi >= 0 && is_solid(u32(r.hi))) { return r; }
    r.valid = true;
    return r;
}
fn nfaces_dir(dir: u32) -> u32 { if (dir == 0u) { return nfu(); } return nfv(); }

// flux de volume par phase à travers la face f (direction Q.a) ; donneuse / receveuse / fantôme / T reconstruite
@compute @workgroup_size(64)
fn k_sweep_flux(@builtin(global_invocation_id) g: vec3<u32>) {
    let dir = Q.a;
    let f = g.x;
    if (f >= nfaces_dir(dir)) { return; }
    let np = P.np; let n = ncell(); let ng = P.ng;
    for (var k = 0u; k < np; k++) { S[O_fl_vol + f * np + k] = 0.0; }
    S[O_fdon + f] = -1.0; S[O_frcv + f] = -1.0; S[O_fghost + f] = 0.0; S[O_ftf + f] = -1.0; S[O_ftg + f] = -1.0;
    let info = face_info(f, dir);
    if (!info.valid) { return; }
    let xi = min(abs(info.uf) * dtv() / P.dx, 1.0);
    var d = info.lo; var r = info.hi;
    if (info.uf < 0.0) { d = info.hi; r = info.lo; }
    S[O_fdon + f] = f32(d); S[O_frcv + f] = f32(r);
    if (d < 0) {
        // entrée par le bord : ambiant (pur ou mélange) ou fluide de l'entrée du côté
        var side = 0u;
        if (dir == 0u) { if (info.fi != 0u) { side = 1u; } } else { if (info.fj == 0u) { side = 2u; } else { side = 3u; } }
        let ib = T_INL + side * 4u;
        var bc = P.bc_l; if (side == 1u) { bc = P.bc_r; } else if (side == 2u) { bc = P.bc_b; } else if (side == 3u) { bc = P.bc_t; }
        var gt = P.t_amb;
        if (T[ib] > 0.5 && bc == 0u) {
            gt = T[ib + 2u];
            if (T[ib + 3u] > 0.5) {
                for (var k = 0u; k < np; k++) { S[O_fl_vol + f * np + k] = xi * amb_alpha(k); }
            } else {
                S[O_fl_vol + f * np + u32(T[ib + 1u])] = xi;
            }
        } else {
            for (var k = 0u; k < np; k++) { S[O_fl_vol + f * np + k] = xi * amb_alpha(k); }
        }
        S[O_fghost + f] = 1.0; S[O_ftf + f] = gt;
        return;
    }
    let dc = u32(d);
    var s0 = 0.0; var s1 = xi; if (info.uf > 0.0) { s0 = 1.0 - xi; s1 = 1.0; }
    // cellule amont de la donneuse (pentes MUSCL)
    var up = -1;
    if (dir == 0u) {
        if (info.uf > 0.0) { if (info.fi >= 2u) { up = i32(cidx(info.fi - 2u, info.fj)); } }
        else { if (info.fi + 1u < P.nx) { up = i32(cidx(info.fi + 1u, info.fj)); } }
    } else {
        if (info.uf > 0.0) { if (info.fj >= 2u) { up = i32(cidx(info.fi, info.fj - 2u)); } }
        else { if (info.fj + 1u < P.ny) { up = i32(cidx(info.fi, info.fj + 1u)); } }
    }
    var fg: array<f32, NPMAX>;
    var spare: array<f32, NPMAX>;
    for (var gi = 0u; gi < ng; gi++) { fg[gi] = 0.0; spare[gi] = 0.0; }
    if (has(F_GRAN)) {
        // groupes DISPERSÉS d'abord (miroir de plic_sweep) : part ξ·α de la bande, α reconstruit à la face (MUSCL / van Leer)
        let ci_ = dc % P.nx; let cj_ = dc / P.nx;
        var rem = xi;
        let edge = up >= 0 && r >= 0 && !is_solid(u32(max(up, 0))) && !is_solid(u32(max(r, 0)));
        for (var gi = 0u; gi < ng; gi++) {
            let a = ag(gi, dc);
            if (gdisp(gi) && a > 0.0) {
                var af = a;
                if (edge) { af = clamp(a + 0.5 * (1.0 - xi) * van_leer(a - ag(gi, u32(up)), ag(gi, u32(r)) - a), 0.0, 1.0); }
                let fk = max(min(min(xi * af, a), rem), 0.0);
                fg[gi] = fk;
                rem -= fk;
                spare[gi] = a - fk;
            }
        }
        // reste de la bande : groupes non dispersés, géométrie PLIC renormalisée hors grains
        let scale = rem / xi;
        var pure = 9999u;
        for (var gi = 0u; gi < ng; gi++) { if (!gdisp(gi) && agr(gi, dc) >= 1.0 - 1e-6) { pure = gi; break; } }
        if (pure < 9999u) {
            fg[pure] = rem;
            rem = 0.0;
        } else {
            var last = 9999u;
            for (var pos = 0u; pos < ng; pos++) { if (!gdisp(gorder(pos))) { last = pos; } }
            for (var pos = 0u; pos < ng; pos++) {
                let gi = gorder(pos);
                if (!gdisp(gi)) {
                    let a = ag(gi, dc);
                    let an_ = agr(gi, dc);
                    var want = 0.0;
                    if (pos == last) { want = rem; }
                    else if (an_ > 1e-6 && an_ < 1.0 - 1e-6) {
                        let gr = youngs_g(gi, ci_, cj_);
                        let pl = plic_new(an_, gr.x, gr.y);
                        if (dir == 0u) { want = scale * plic_area(pl, s0, s1, 0.0, 1.0); } else { want = scale * plic_area(pl, 0.0, 1.0, s0, s1); }
                    }
                    let fk = max(min(min(want, rem), a), 0.0);
                    fg[gi] = fk;
                    rem -= fk;
                    spare[gi] = a - fk;
                }
            }
        }
        if (rem > 1e-9) {
            var tot = 0.0;
            for (var gi = 0u; gi < ng; gi++) { tot += max(spare[gi], 0.0); }
            if (tot > 0.0) { for (var gi = 0u; gi < ng; gi++) { fg[gi] += rem * max(spare[gi], 0.0) / tot; } }
        }
    } else {
    var pure = 9999u;
    for (var gi = 0u; gi < ng; gi++) { if (ag(gi, dc) >= 1.0 - 1e-6) { pure = gi; break; } }
    if (pure < 9999u) {
        fg[pure] = xi;
    } else {
        let ci_ = dc % P.nx; let cj_ = dc / P.nx;
        var rem = xi;
        for (var pos = 0u; pos < ng; pos++) {
            let gi = gorder(pos);
            let a = ag(gi, dc);
            var want = 0.0;
            if (pos == ng - 1u) { want = rem; }
            else if (a > 1e-6 && a < 1.0 - 1e-6) {
                let gr = youngs_g(gi, ci_, cj_);
                let pl = plic_new(a, gr.x, gr.y);
                if (dir == 0u) { want = plic_area(pl, s0, s1, 0.0, 1.0); } else { want = plic_area(pl, 0.0, 1.0, s0, s1); }
            }
            let fk = max(min(min(want, rem), a), 0.0);
            fg[gi] = fk;
            rem -= fk;
            spare[gi] = a - fk;
        }
        if (rem > 1e-9) {
            var tot = 0.0;
            for (var gi = 0u; gi < ng; gi++) { tot += max(spare[gi], 0.0); }
            if (tot > 0.0) { for (var gi = 0u; gi < ng; gi++) { fg[gi] += rem * max(spare[gi], 0.0) / tot; } }
        }
    }
    }
    // MUSCL : cellule amont de la donneuse et receveuse, toutes trois dans un même groupe à 99 %
    var musc = false; var gs = 9999u;
    var uc = 0u; var rc = 0u;
    if (up >= 0 && r >= 0) {
        uc = u32(up); rc = u32(r);
        if (!is_solid(uc) && !is_solid(rc)) {
            let gd = single_g(dc);
            if (gd < 9999u && single_g(uc) == gd && single_g(rc) == gd) { musc = true; gs = gd; }
        }
    }
    if (musc) {
        // reconstruction de la température DU groupe présent dans les trois cellules
        let tu = tgrp(gs, uc); let td = tgrp(gs, dc); let tr = tgrp(gs, rc);
        S[O_ftf + f] = td + 0.5 * (1.0 - xi) * van_leer(td - tu, tr - td);
        S[O_ftg + f] = f32(gs);
    }
    for (var gi = 0u; gi < ng; gi++) {
        if (fg[gi] <= 0.0) { continue; }
        let tot = max(ag(gi, dc), 1e-30);
        let cnt = gcount(gi);
        if (musc && gs == gi && cnt > 1u) {
            let tu = max(ag(gi, uc), 1e-30); let tr = max(ag(gi, rc), 1e-30);
            var phi: array<f32, NPMAX>;
            var sphi = 0.0;
            for (var q = 0u; q < cnt; q++) {
                let k = gmember(gi, q);
                let pu = S[O_an + k * n + uc] / tu; let pd = S[O_an + k * n + dc] / tot; let pr_ = S[O_an + k * n + rc] / tr;
                phi[q] = max(pd + 0.5 * (1.0 - xi) * van_leer(pd - pu, pr_ - pd), 0.0);
                sphi += phi[q];
            }
            sphi = max(sphi, 1e-30);
            for (var q = 0u; q < cnt; q++) { S[O_fl_vol + f * np + gmember(gi, q)] = fg[gi] * phi[q] / sphi; }
        } else {
            for (var q = 0u; q < cnt; q++) { let k = gmember(gi, q); S[O_fl_vol + f * np + k] = fg[gi] * S[O_an + k * n + dc] / tot; }
        }
    }
}

// limiteur : sortie totale de chaque phase ≤ 0.999 α de travail
@compute @workgroup_size(64)
fn k_sweep_scale(@builtin(global_invocation_id) g: vec3<u32>) {
    let dir = Q.a;
    let c = g.x; let n = ncell(); let np = P.np;
    if (c >= n) { return; }
    let i = c % P.nx; let j = c / P.nx;
    var f_lo = uidx(i, j); var f_hi = uidx(i + 1u, j);
    if (dir == 1u) { f_lo = vidx(i, j); f_hi = vidx(i, j + 1u); }
    let fc = f32(c);
    for (var k = 0u; k < np; k++) {
        var out = 0.0;
        if (S[O_fdon + f_lo] == fc) { out += S[O_fl_vol + f_lo * np + k]; }
        if (S[O_fdon + f_hi] == fc) { out += S[O_fl_vol + f_hi * np + k]; }
        let cap = max(S[O_aw + k * n + c], 0.0) * 0.999;
        var sc = 1.0;
        if (out > cap && out > 0.0) { sc = cap / out; }
        S[O_scale + k * n + c] = sc;
    }
}
// mise à jour conservative (α de travail, m_k, ε) face par face DANS L'ORDRE des indices, comme le CPU : la masse volumique m/α de la
// donneuse est lue sur l'état courant (flux entrants des faces précédentes compris). Un fil par ligne (balayage x) ou par colonne
// (balayage y) : les lignes (colonnes) sont indépendantes, l'ordre dans une ligne (colonne) est celui du CPU.
@compute @workgroup_size(64)
fn k_sweep_seq(@builtin(global_invocation_id) g: vec3<u32>) {
    let dir = Q.a;
    let t = g.x;
    let np = P.np; let n = ncell();
    var nline = P.ny; var nface = P.nx + 1u;
    if (dir == 1u) { nline = P.nx; nface = P.ny + 1u; }
    if (t >= nline) { return; }
    let a2 = P.dx * P.dx;
    for (var q = 0u; q < nface; q++) {
        var f = 0u;
        if (dir == 0u) { f = uidx(q, t); } else { f = vidx(t, q); }
        let d = i32(S[O_fdon + f]); let r = i32(S[O_frcv + f]);
        let ghost = S[O_fghost + f] > 0.5;
        if (d < 0 && !ghost) { continue; }
        var eng: array<f32, NPMAX>;
        for (var gq = 0u; gq < P.ng; gq++) { eng[gq] = 0.0; }
        var mt = 0.0;
        for (var k = 0u; k < np; k++) {
            var fk = S[O_fl_vol + f * np + k];
            if (fk == 0.0) { continue; }
            var mf = 0.0; var t_d = 0.0;
            if (ghost) {
                var pr = 0.0; if (r >= 0) { pr = S[O_prel + u32(r)]; }
                t_d = S[O_ftf + f];
                mf = fk * rho_k(k, pr, t_d);
                if (r >= 0) { S[O_bflux + k * n + u32(r)] += mf * a2; }
            } else {
                let dc = u32(d);
                fk = fk * S[O_scale + k * n + dc];
                // température du groupe de la phase dans la donneuse (reconstruite si la face est celle de ce groupe)
                let told = ptemp(k, dc);
                t_d = told; if (S[O_ftf + f] > 0.0 && S[O_ftg + f] == f32(group_k(k))) { t_d = S[O_ftf + f]; }
                let aw = S[O_aw + k * n + dc];
                var rho_d = rho_k(k, S[O_prel + dc], told);
                if (aw > 1e-8) { let qq = mass(k, dc) / aw; if (qq > 0.0 && qq < 1e30) { rho_d = qq; } }
                mf = min(fk * rho_d, max(mass(k, dc), 0.0));
                S[O_aw + k * n + dc] -= fk;
                S[O_mass + k * n + dc] -= mf;
                if (r < 0) { S[O_bflux + k * n + dc] -= mf * a2; }
            }
            if (r >= 0) {
                S[O_aw + k * n + u32(r)] += fk;
                S[O_mass + k * n + u32(r)] += mf;
            }
            eng[group_k(k)] += mf * erel_k(k, t_d);
            mt += mf;
        }
        for (var gq = 0u; gq < P.ng; gq++) {
            let en = eng[gq];
            if (en == 0.0) { continue; }
            if (d >= 0) { S[O_epsg + gq * n + u32(d)] -= en; S[O_eps + u32(d)] -= en; }
            if (r >= 0) { S[O_epsg + gq * n + u32(r)] += en; S[O_eps + u32(r)] += en; }
        }
        if (dir == 0u) { var sg = 1.0; if (S[O_U + f] < 0.0) { sg = -1.0; } S[O_mfx + f] += sg * mt; }
        else { var sg = 1.0; if (S[O_V + f] < 0.0) { sg = -1.0; } S[O_mfy + f] += sg * mt; }
    }
}
@compute @workgroup_size(64)
fn k_temp_update(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell() || is_solid(c)) { return; }
    // plancher d'énergie par groupe (T_g ≥ T_PW_MIN) : cellule presque vide dont le flux d'énergie sortant dépasse le contenu
    let n = ncell();
    for (var gq = 0u; gq < P.ng; gq++) {
        if (group_mass(gq, c) > 0.0) {
            let emin = group_erel_at(gq, c, T_PW_MIN);
            if (epsg(gq, c) < emin) { S[O_epsg + gq * n + c] = emin; }
        }
    }
    cell_sync(c);
}
// erreur de fermeture héritée Σ m/ρ(p, T) − 1 au début du pas (sous-relaxée dans k_rhs)
@compute @workgroup_size(64)
fn k_err0(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    if (is_solid(c) || pinned(c)) { S[O_err0 + c] = 0.0; return; }
    S[O_err0 + c] = cell_vol_at(c, S[O_prel + c]) - 1.0;
}

// ---------------------------------------------------------------- quantité de mouvement conservative (Rudman)
fn ux_(i: i32, j: i32) -> f32 { return S[O_U + u32(clamp(j, 0, i32(P.ny) - 1)) * (P.nx + 1u) + u32(clamp(i, 0, i32(P.nx)))]; }
fn vy_(i: i32, j: i32) -> f32 { return S[O_V + u32(clamp(j, 0, i32(P.ny))) * P.nx + u32(clamp(i, 0, i32(P.nx) - 1))]; }
fn minmod(a: f32, b: f32) -> f32 { if (a * b <= 0.0) { return 0.0; } if (abs(a) < abs(b)) { return a; } return b; }
fn mnew(c: u32) -> f32 { var s = 0.0; for (var k = 0u; k < P.np; k++) { s += mass(k, c); } return s; }
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
fn k_mom(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    let W = P.nx + 1u;
    if (f < nfu()) {
        let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
        var out = S[O_U + f];
        if (i >= 1u && i < P.nx) {
            let a = cidx(i - 1u, j); let b = cidx(i, j);
            if (!is_solid(a) && !is_solid(b)) {
                let ii = i32(i); let jj = i32(j);
                let rho_old = 0.5 * (S[O_mtot_old + a] + S[O_mtot_old + b]);
                let rho_new = 0.5 * (mnew(a) + mnew(b));
                let fr = 0.5 * (S[O_mfx + j * W + i] + S[O_mfx + j * W + i + 1u]);
                let fl = 0.5 * (S[O_mfx + j * W + i - 1u] + S[O_mfx + j * W + i]);
                let fb = 0.5 * (S[O_mfy + j * P.nx + i - 1u] + S[O_mfy + j * P.nx + i]);
                let ft = 0.5 * (S[O_mfy + (j + 1u) * P.nx + i - 1u] + S[O_mfy + (j + 1u) * P.nx + i]);
                let cr = carry_ux(fr, ii, jj); let cl = carry_ux(fl, ii - 1, jj);
                var ct = ux_(ii, jj); var cb = ct; var ft_ = 0.0; var fb_ = 0.0;
                if (j + 1u < P.ny) { ct = carry_uy(ft, ii, jj); ft_ = ft; }
                if (j > 0u) { cb = carry_uy(fb, ii, jj - 1); fb_ = fb; }
                let q = rho_old * S[O_U + f] - (fr * cr - fl * cl) - (ft_ * ct - fb_ * cb);
                let lo = min(min(min(min(ux_(ii, jj), ux_(ii - 1, jj)), ux_(ii + 1, jj)), ux_(ii, jj - 1)), ux_(ii, jj + 1));
                let hi = max(max(max(max(ux_(ii, jj), ux_(ii - 1, jj)), ux_(ii + 1, jj)), ux_(ii, jj - 1)), ux_(ii, jj + 1));
                if (rho_new > 1e-8) { out = clamp(q / rho_new, lo, hi); }
            }
        }
        S[O_Us + f] = out + S[O_Fx + f];
    }
    if (f < nfv()) {
        let i = f % P.nx; let j = f / P.nx;
        var out = S[O_V + f];
        if (j >= 1u && j < P.ny) {
            let a = cidx(i, j - 1u); let b = cidx(i, j);
            if (!is_solid(a) && !is_solid(b)) {
                let ii = i32(i); let jj = i32(j);
                let rho_old = 0.5 * (S[O_mtot_old + a] + S[O_mtot_old + b]);
                let rho_new = 0.5 * (mnew(a) + mnew(b));
                let ft = 0.5 * (S[O_mfy + j * P.nx + i] + S[O_mfy + (j + 1u) * P.nx + i]);
                let fb = 0.5 * (S[O_mfy + (j - 1u) * P.nx + i] + S[O_mfy + j * P.nx + i]);
                let fr = 0.5 * (S[O_mfx + j * W + i + 1u] + S[O_mfx + (j - 1u) * W + i + 1u]);
                let fl = 0.5 * (S[O_mfx + j * W + i] + S[O_mfx + (j - 1u) * W + i]);
                let ct = carry_vy(ft, ii, jj); let cb = carry_vy(fb, ii, jj - 1);
                var cr = vy_(ii, jj); var cl = cr; var fr_ = 0.0; var fl_ = 0.0;
                if (i + 1u < P.nx) { cr = carry_vx(fr, ii, jj); fr_ = fr; }
                if (i > 0u) { cl = carry_vx(fl, ii - 1, jj); fl_ = fl; }
                let q = rho_old * S[O_V + f] - (ft * ct - fb * cb) - (fr_ * cr - fl_ * cl);
                let lo = min(min(min(min(vy_(ii, jj), vy_(ii, jj - 1)), vy_(ii, jj + 1)), vy_(ii - 1, jj)), vy_(ii + 1, jj));
                let hi = max(max(max(max(vy_(ii, jj), vy_(ii, jj - 1)), vy_(ii, jj + 1)), vy_(ii - 1, jj)), vy_(ii + 1, jj));
                if (rho_new > 1e-8) { out = clamp(q / rho_new, lo, hi); }
            }
        }
        S[O_Vs + f] = out + S[O_Fy + f];
    }
}

// ---------------------------------------------------------------- pression : Newton + MG-PCG
fn psat_q(q: u32, t_in: f32) -> f32 {
    let b = T_PC + q * PC_STRIDE;
    let lref = T[b + 2u]; let tref = T[b + 3u]; let pref = T[b + 4u]; let rv = T[b + 5u]; let dcp = T[b + 6u];
    let t = clamp(t_in, 200.0, 640.0);
    let a = (lref - dcp * tref) / rv;
    return pref * exp(a * (1.0 / tref - 1.0 / t) + dcp / rv * log(t / tref));
}
@compute @workgroup_size(64)
fn k_save_p(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n) { return; }
    let p0 = S[O_prel + c];
    S[O_p_start + c] = p0;
    for (var gq = 0u; gq < P.ng; gq++) { S[O_tst + gq * n + c] = tgrp(gq, c); }
    S[O_chi + c] = thermal_coupling(c, dtv());
    S[O_clamped + c] = 0.0;
    // plancher : P_FLOOR ou p_sat d'un liquide présent (cavité discrète), à la température de ce liquide
    var fl = P_FLOOR;
    if (has(F_PC) && !is_solid(c)) {
        for (var q = 0u; q < P.npc; q++) {
            let liq = u32(T[T_PC + q * PC_STRIDE]);
            if (alpha(liq, c) > 1e-3) { fl = max(fl, min(psat_q(q, ptemp(liq, c)), 0.95 * max(p0 + P.p0, P_FLOOR))); }
        }
    }
    S[O_floor + c] = fl - P.p0;
}
// Volume V et κ_eff à la pression relative pr, groupe g à la température t[g], couplage χ (miroir de cell_closure_g) ; gam[g] =
// dT_g/dp, wpw[g] = part du groupe dans le travail de pression
fn closure_g(c: u32, pr: f32, t: ptr<function, array<f32, NPMAX>>, chi_in: f32, gam: ptr<function, array<f32, NPMAX>>, wpw: ptr<function, array<f32, NPMAX>>) -> vec2<f32> {
    var v = 0.0; var kt = 0.0; var own = 0.0; var vb_all = 0.0; var mcp_all = 0.0; var mcpt = 0.0;
    var vbg: array<f32, NPMAX>; var mcpg: array<f32, NPMAX>;
    for (var g = 0u; g < P.ng; g++) {
        let tg = (*t)[g];
        var vb = 0.0; var mcp = 0.0;
        // borne uniforme (P.np) et appartenance testée, sans « continue » (lavapipe)
        for (var k = 0u; k < P.np; k++) {
            let m = mass(k, c);
            if (group_k(k) == g && m > 0.0) {
                let vk = m / rho_k(k, pr, tg);
                v += vk; kt += vk * kappa_k(k, pr); vb += vk * beta_k(k, tg); mcp += m * cp_k(k);
            }
        }
        vbg[g] = vb; mcpg[g] = mcp;
        if (mcp > 0.0) { own += tg * vb * vb / mcp; }
        vb_all += vb; mcp_all += mcp; mcpt += mcp * tg;
    }
    let chi = clamp(chi_in, 0.0, 1.0);
    var tm = 0.0; var gm = 0.0;
    if (mcp_all > 0.0) { tm = mcpt / mcp_all; gm = tm * vb_all / mcp_all; }
    let mix = gm * vb_all;
    for (var g = 0u; g < P.ng; g++) {
        var go = 0.0; if (mcpg[g] > 0.0) { go = (*t)[g] * vbg[g] / mcpg[g]; }
        (*gam)[g] = chi * gm + (1.0 - chi) * go;
        (*wpw)[g] = max(mcpg[g] * (*gam)[g], 0.0);
    }
    return vec2<f32>(v, kt - (chi * mix + (1.0 - chi) * own));
}
fn newton_done() -> bool { return D[D_NEWTON_DONE] > 0.5; }
@compute @workgroup_size(64)
fn k_closure(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell() || newton_done()) { return; }
    if (is_solid(c)) { S[O_vol + c] = 1.0; S[O_kappa + c] = 1e-30; return; }
    if (pinned(c)) { S[O_vol + c] = 1.0; S[O_kappa + c] = 1.0; return; }
    let n = ncell();
    var ts: array<f32, NPMAX>; var gam: array<f32, NPMAX>; var wp: array<f32, NPMAX>;
    for (var gq = 0u; gq < P.ng; gq++) { ts[gq] = S[O_tst + gq * n + c]; }
    let chi = S[O_chi + c];
    _ = closure_g(c, S[O_p_start + c], &ts, chi, &gam, &wp);
    var tk: array<f32, NPMAX>;
    for (var gq = 0u; gq < P.ng; gq++) {
        tk[gq] = ts[gq] + gam[gq] * (S[O_prel + c] - S[O_p_start + c]);
        S[O_tk + gq * n + c] = tk[gq];
    }
    let cl = closure_g(c, S[O_prel + c], &tk, chi, &gam, &wp);
    S[O_vol + c] = cl.x;
    S[O_kappa + c] = max(cl.y, 1e-30);
}
// pression (relative) qui pousse le mélange : p + σ' (lits perméables : fluide des pores + contacts ; σ' = 0 ailleurs)
fn ptot(c: u32) -> f32 { return S[O_prel + c] + S[O_sig + c]; }
fn face_u(f: u32) -> f32 {
    if (S[O_pfu_on + f] > 0.5) { return S[O_pfu + f]; }
    let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
    let us = S[O_Us + f];
    if (i > 0u && i < P.nx) {
        let a = cidx(i - 1u, j); let b = cidx(i, j);
        if (is_solid(a) || is_solid(b)) { return 0.0; }
        return us - dtv() / frho_x(i, j) * (ptot(b) - ptot(a)) / P.dx;
    }
    if (i == 0u) {
        let b = cidx(0u, j);
        if (P.bc_l == 1u && !is_solid(b)) { return us - dtv() / S[O_rho_mix + b] * (ptot(b) - P.pb_l) / (0.5 * P.dx); }
        return 0.0;
    }
    let a = cidx(P.nx - 1u, j);
    if (P.bc_r == 1u && !is_solid(a)) { return us - dtv() / S[O_rho_mix + a] * (P.pb_r - ptot(a)) / (0.5 * P.dx); }
    return 0.0;
}
fn face_v(f: u32) -> f32 {
    if (S[O_pfv_on + f] > 0.5) { return S[O_pfv + f]; }
    let i = f % P.nx; let j = f / P.nx;
    let vs = S[O_Vs + f];
    if (j > 0u && j < P.ny) {
        let a = cidx(i, j - 1u); let b = cidx(i, j);
        if (is_solid(a) || is_solid(b)) { return 0.0; }
        return vs - dtv() / frho_y(i, j) * (ptot(b) - ptot(a)) / P.dx;
    }
    if (j == 0u) {
        let b = cidx(i, 0u);
        if (P.bc_b == 1u && !is_solid(b)) { return vs - dtv() / S[O_rho_mix + b] * (ptot(b) - P.pb_b) / (0.5 * P.dx); }
        return 0.0;
    }
    let a = cidx(i, P.ny - 1u);
    if (P.bc_t == 1u && !is_solid(a)) { return vs - dtv() / S[O_rho_mix + a] * (P.pb_t - ptot(a)) / (0.5 * P.dx); }
    return 0.0;
}
@compute @workgroup_size(64)
fn k_face_tmp(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (newton_done()) { return; }
    if (f < nfu()) { S[O_Fx + f] = face_u(f); }
    if (f < nfv()) { S[O_Fy + f] = face_v(f); }
}
@compute @workgroup_size(64)
fn k_face_final(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < nfu()) { S[O_U + f] = face_u(f); }
    if (f < nfv()) { S[O_V + f] = face_v(f); }
}
@compute @workgroup_size(64)
fn k_rhs(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell() || newton_done()) { return; }
    if (is_solid(c) || pinned(c)) { S[O_rhs + c] = 0.0; return; }
    let i = c % P.nx; let j = c / P.nx;
    var dn = (S[O_Fx + uidx(i + 1u, j)] - S[O_Fx + uidx(i, j)] + S[O_Fy + vidx(i, j + 1u)] - S[O_Fy + vidx(i, j)]) / P.dx;
    var d0 = (S[O_Un + uidx(i + 1u, j)] - S[O_Un + uidx(i, j)] + S[O_Vn + vidx(i, j + 1u)] - S[O_Vn + vidx(i, j)]) / P.dx;
    // percolation : flux de Darcy à la pression courante, et celui qui a servi au transport de ce pas (miroir de solve_pressure)
    if (perc_on()) { dn += darcy_div_now(i, j); d0 += darcy_div_old(i, j); }
    // part corrigée de l'erreur héritée (miroir de mp_pressure::closure_correction)
    var ag = 0.0;
    for (var k = 0u; k < P.np; k++) { if (is_gas(k)) { ag += alpha(k, c); } }
    ag = clamp(ag, 0.0, 1.0);
    // erreur qui restera après l'advection du pas (miroir de solve_pressure : f = err0 − dt·∇·u^n)
    let e0 = S[O_err0 + c] - dtv() * d0;
    let r = clamp(P.closure_relax, 0.0, 1.0);
    let small = 0.1 * P.closure_small_scale;   // mp_pressure::CLOSURE_SMALL
    var liq = e0;
    if (abs(e0) > small) { liq = sign(e0) * max(small, r * abs(e0)); }
    let corr = ag * e0 + (1.0 - ag) * liq;
    S[O_rhs + c] = (S[O_vol + c] - 1.0 - (S[O_err0 + c] - corr)) / dtv() - (dn - d0);
}
// couplage : 0 aucun (solide, face imposée), 1 complet, 2 Dirichlet (voisin épinglé)
fn link(a: u32, b: u32, fixed: bool) -> u32 {
    if (fixed || is_solid(a) || is_solid(b)) { return 0u; }
    if (pinned(a) || pinned(b)) { return 2u; }
    return 1u;
}
@compute @workgroup_size(64)
fn k_stencil(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell() || newton_done()) { return; }
    S[O_sae + c] = 0.0; S[O_san + c] = 0.0; S[O_sd + c] = 1.0;
    S[O_sact + c] = 0.0;
    if (is_solid(c) || pinned(c)) { return; }
    S[O_sact + c] = 1.0;
    let i = c % P.nx; let j = c / P.nx;
    let dx2 = P.dx * P.dx;
    let dt = dtv();
    var d = S[O_kappa + c] / dt;
    // percolation : conductance de Darcy C/dx² ajoutée au couplage (y compris des faces fermées au mélange : grilles)
    let pc = perc_on();
    if (i + 1u < P.nx) {
        let k = link(c, c + 1u, S[O_pfu_on + uidx(i + 1u, j)] > 0.5);
        let td = select(0.0, S[O_dcu + uidx(i + 1u, j)] / dx2, pc);
        if (k > 0u) { let t = dt / (frho_x(i + 1u, j) * dx2) + td; if (k == 1u) { S[O_sae + c] = t; } d += t; }
        else if (td > 0.0) { S[O_sae + c] = td; d += td; }
    } else if (P.bc_r == 1u) { d += 2.0 * dt / (S[O_rho_mix + c] * dx2); }
    if (i > 0u) {
        if (link(c - 1u, c, S[O_pfu_on + uidx(i, j)] > 0.5) > 0u) { d += dt / (frho_x(i, j) * dx2); }
        d += select(0.0, S[O_dcu + uidx(i, j)] / dx2, pc);
    } else if (P.bc_l == 1u) { d += 2.0 * dt / (S[O_rho_mix + c] * dx2); }
    if (j + 1u < P.ny) {
        let k = link(c, c + P.nx, S[O_pfv_on + vidx(i, j + 1u)] > 0.5);
        let td = select(0.0, S[O_dcv + vidx(i, j + 1u)] / dx2, pc);
        if (k > 0u) { let t = dt / (frho_y(i, j + 1u) * dx2) + td; if (k == 1u) { S[O_san + c] = t; } d += t; }
        else if (td > 0.0) { S[O_san + c] = td; d += td; }
    } else if (P.bc_t == 1u) { d += 2.0 * dt / (S[O_rho_mix + c] * dx2); }
    if (j > 0u) {
        if (link(c - P.nx, c, S[O_pfv_on + vidx(i, j)] > 0.5) > 0u) { d += dt / (frho_y(i, j) * dx2); }
        d += select(0.0, S[O_dcv + vidx(i, j)] / dx2, pc);
    } else if (P.bc_b == 1u) { d += 2.0 * dt / (S[O_rho_mix + c] * dx2); }
    S[O_sd + c] = d;
}
// mise à jour de Newton d'une cellule contenant du gaz quand |δp| > 50 % de p : loi polytropique tangente au même κ
// (miroir de MultiphaseSim::gas_newton_update ; pr = pression relative)
fn gas_newton_update(c: u32, pr: f32, kap: f32, dp: f32) -> f32 {
    let pa = pr + P.p0;
    if (abs(dp) <= 0.5 * max(pa, P_FLOOR)) { return dp; }
    var vg = 0.0; var kl = 0.0;
    let n = ncell();
    for (var k = 0u; k < P.np; k++) {
        let m = mass(k, c);
        if (m <= 0.0) { continue; }
        let vk = m / rho_k(k, pr, S[O_tk + group_k(k) * n + c]);
        if (is_gas(k)) { vg += vk; } else { kl += vk * kappa_k(k, pr); }
    }
    let kg = kap - kl;
    if (vg <= 1e-9 || kg <= 0.0) { return dp; }
    let npoly = clamp(vg / (max(pa, 1.0) * kg), 1.0, 2.0);
    let vgt = max(vg - kg * dp, 0.02 * vg);
    let d = pa * (pow(vg / vgt, npoly) - 1.0);
    // au plus 4 fois la tangente à la compression (le reste du système est linéarisé autour de p_k)
    if (dp > 0.0) { return min(d, 4.0 * dp); }
    return d;
}
// pk += δp, plancher (cavité), |δp| max par groupe
@compute @workgroup_size(64)
fn k_p_update(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let c = g.x;
    var v: array<f32, 8>;
    v[0] = 0.0;
    if (c < ncell() && !newton_done() && !is_solid(c) && !pinned(c)) {
        let p_old = S[O_prel + c];
        let dp = gas_newton_update(c, p_old, S[O_kappa + c], S[O_cx + c]);
        var pk = p_old + dp;
        let fl = S[O_floor + c];
        if (!(pk == pk) || abs(pk) > 1e30) { pk = max(S[O_p_start + c], fl); }
        else if (pk < fl) { pk = fl; S[O_clamped + c] = 1.0; }
        else { S[O_clamped + c] = 0.0; }
        S[O_prel + c] = pk;
        // f32 : |δp| rapporté au bruit d'arrondi (0.1 Pa, 2e-6 p)
        v[0] = abs(pk - p_old) / max(0.1, 2e-6 * max(pk + P.p0, 0.0));
    }
    wg_reduce(v, 1u, 1u, l.x, w.x, 0u);
}
// |δp| max de l'itération (sommes partielles de k_p_update), itérations du gradient conjugué de la pression
@compute @workgroup_size(256)
fn k_newton_dp(@builtin(local_invocation_id) l: vec3<u32>) {
    let dmax = final_reduce(0u, nwg_cells(), 1u, l.x);
    if (l.x == 0u && D[D_NEWTON_DONE] < 0.5) {
        D[D_NEWTON_DP] = dmax;
        D[D_PCG_IT] += D[D_PCG_ITN];
        D[D_MAXPCG_N] = max(D[D_MAXPCG_N], D[D_PCG_ITN]);
        if (D[D_PCG_DONE] < 0.5) { D[D_PCG_UNCONV] = 1.0; }
    }
}
@compute @workgroup_size(64)
fn k_newton_check(@builtin(local_invocation_id) l: vec3<u32>) {
    if (l.x == 0u) {
        if (D[D_NEWTON_DONE] < 0.5) {
            D[D_NEWTON_IT] += 1.0;
            // f32 : |δp| < max(0.1 Pa, 2e-6 p) (le CPU f64 utilise 1e-3 Pa), et pression de contact des lits perméables stable
            if (D[D_NEWTON_DP] < 1.0 && D[D_SK_MOVED] < 0.5) { D[D_NEWTON_DONE] = 1.0; }
            D[D_SK_MOVED] = 0.0;
        }
        D[D_PCG_DONE] = 0.0; D[D_PCG_ITN] = 0.0;
    }
}
// travail de pression −p ∇·u (u^{n+1}), |dt ∇·u| ≤ 0.5
@compute @workgroup_size(64)
fn k_pressure_work(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell() || is_solid(c) || pinned(c)) { return; }
    let i = c % P.nx; let j = c / P.nx;
    let dt = dtv();
    // (+ divergence du flux de percolation : le fluide des pores se détend ou se comprime aussi)
    var dq = 0.0;
    if (perc_on()) { dq = darcy_div_old(i, j); }
    let d = clamp(dt * (div_uv(i, j) + dq), -0.5, 0.5) / dt;
    // répartition entre groupes comme dans la fermeture (C_g dT_g/dp, couplage χ) ; par groupe : au plus −50 % / +100 % de T et pas
    // de refroidissement sous T_PW_MIN (miroir de pressure_work)
    let n = ncell();
    var ts: array<f32, NPMAX>; var gam: array<f32, NPMAX>; var wp: array<f32, NPMAX>;
    for (var gq = 0u; gq < P.ng; gq++) { ts[gq] = tgrp(gq, c); }
    _ = closure_g(c, S[O_prel + c], &ts, S[O_chi + c], &gam, &wp);
    var ksum = 0.0; var asum = 0.0;
    for (var gq = 0u; gq < P.ng; gq++) { ksum += wp[gq]; asum += group_alpha(gq, c); }
    asum = max(asum, 1e-30);
    let pc = max(S[O_prel + c] + P.p0, 0.0);
    let de_tot = -dt * pc * d;
    // part due au changement de phase (−p·ΔV_pc) : au groupe du liquide, partagée comme la chaleur latente, bornée au signe et à
    // l'amplitude de −p∇·u dt (miroir de pressure_work) ; ΔV_pc remis à zéro pour le pas suivant
    var pc_sum = 0.0;
    for (var gq = 0u; gq < P.ng; gq++) { pc_sum += -pc * S[O_pc_dv + gq * n + c]; }
    var scale = 0.0;
    if (pc_sum != 0.0 && pc_sum * de_tot > 0.0) { scale = min(de_tot / pc_sum, 1.0); }
    let de_rest = de_tot - scale * pc_sum;
    let chi = S[O_chi + c];
    var de_g: array<f32, NPMAX>;
    for (var gq = 0u; gq < P.ng; gq++) {
        let w = select(group_alpha(gq, c) / asum, wp[gq] / ksum, ksum > 0.0);
        de_g[gq] = select(0.0, de_rest * w, group_mass(gq, c) > 0.0);
    }
    for (var gs = 0u; gs < P.ng; gs++) {
        let q = scale * (-pc * S[O_pc_dv + gs * n + c]);
        var ctot = 0.0;
        for (var gq = 0u; gq < P.ng; gq++) {
            let part = group_mass(gq, c) > 0.0 && (gq == gs || !latent_excluded(gq, c));
            ctot += select(0.0, group_cap_at(gq, c, max(ts[gq], 1.0)) * select(chi, 1.0, gq == gs), part);
        }
        for (var gq = 0u; gq < P.ng; gq++) {
            let part = group_mass(gq, c) > 0.0 && (gq == gs || !latent_excluded(gq, c));
            let cw = select(0.0, group_cap_at(gq, c, max(ts[gq], 1.0)) * select(chi, 1.0, gq == gs), part);
            de_g[gq] += select(select(0.0, q, gq == gs), q * cw / ctot, ctot > 0.0);
        }
        S[O_pc_dv + gs * n + c] = 0.0;
    }
    for (var gq = 0u; gq < P.ng; gq++) {
        let t = max(ts[gq], 1.0);
        let cap = group_cap_at(gq, c, t);
        let de_min = min(max(-0.5 * cap * t, cap * (T_PW_MIN - t)), 0.0);
        S[O_epsg + gq * n + c] += select(0.0, clamp(de_g[gq], de_min, cap * t), group_mass(gq, c) > 0.0);
    }
    cell_sync(c);
}
// cavités sans liquide volatil : cellule bloquée au plancher dont le contenu n'occupe plus tout le volume → gaz ambiant à cette
// pression (miroir de MultiphaseSim::fill_cavities)
@compute @workgroup_size(64)
fn k_fill_cav(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n || is_solid(c) || pinned(c) || S[O_clamped + c] < 0.5) { return; }
    if (has(F_PC)) {
        for (var q = 0u; q < P.npc; q++) {
            let liq = u32(T[T_PC + q * PC_STRIDE]);
            if (alpha(liq, c) > 1e-3) { return; }
        }
    }
    let pr = S[O_prel + c];
    var t = S[O_temp + c];
    if (t == t && abs(t) < 1e30) { t = clamp(t, 150.0, 5000.0); } else { t = P.t_amb; }
    let deficit = 1.0 - cell_vol_at(c, pr);
    if (deficit <= 1e-6) { return; }
    for (var k = 0u; k < P.np; k++) {
        let a = amb_alpha(k);
        if (a > 0.0) {
            let m = deficit * a * rho_k(k, pr, t);
            S[O_mass + k * n + c] += m;
            S[O_epsg + group_k(k) * n + c] += m * erel_k(k, t);
        }
    }
    cell_sync(c);
}

// ---------------------------------------------------------------- contrôle de fin de pas (miroir de state_is_sane) + extrema
@compute @workgroup_size(64)
fn k_sanity(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    var v: array<f32, 8>;
    for (var q = 0u; q < 8u; q++) { v[q] = -3.0e38; }
    v[0] = 0.0;
    let i = g.x;
    if (i < nfu()) { let u = S[O_U + i]; if (!(u == u) || abs(u) > P.speed_limit) { v[0] = 1.0; } v[1] = abs(u); }
    if (i < nfv()) { let u = S[O_V + i]; if (!(u == u) || abs(u) > P.speed_limit) { v[0] = 1.0; } v[1] = max(v[1], abs(u)); }
    if (i < ncell()) {
        let t = S[O_temp + i]; let e = S[O_eps + i]; let p = S[O_prel + i] + P.p0;
        if (!(t == t) || !(e == e) || !(p == p) || abs(t) > 1e30 || abs(e) > 1e37 || abs(p) > 1e37) { v[0] = 1.0; }
        if (!is_solid(i)) {
            if (t < 50.0 || t > 5000.0 || p < 0.0) { v[0] = 1.0; }
            for (var gq = 0u; gq < P.ng; gq++) {
                if (group_mass(gq, i) > 0.0) { let tq = tgrp(gq, i); if (!(tq == tq) || tq < 50.0 || tq > 5000.0) { v[0] = 1.0; } }
            }
            for (var k = 0u; k < P.np; k++) { let m = mass(k, i); if (!(m == m) || m < -1e-6 || abs(m) > 1e30) { v[0] = 1.0; } }
            v[2] = -p; v[3] = p;
        }
        v[4] = -t; v[5] = t;
    }
    wg_reduce(v, 6u, 1u, l.x, w.x, 0u);
}
@compute @workgroup_size(256)
fn k_sanity_final(@builtin(local_invocation_id) l: vec3<u32>) {
    let nw = (max(max(nfu(), nfv()), ncell()) + 63u) / 64u;
    let bad = final_reduce(0u, nw, 1u, l.x);
    let um = final_reduce(1u, nw, 1u, l.x);
    let pmin = -final_reduce(2u, nw, 1u, l.x);
    let pmax = final_reduce(3u, nw, 1u, l.x);
    let tmin = -final_reduce(4u, nw, 1u, l.x);
    let tmax = final_reduce(5u, nw, 1u, l.x);
    if (l.x == 0u) {
        // vitesses produites par le pas trop grandes pour son dt (choc thermique, effondrement de poche) : pas refait à dt/4
        var b = bad;
        if (um * dtv() > P.cfl_reject * P.dx) { b = 1.0; }
        D[D_BAD] = b; D[D_UMAX] = um; D[D_PMIN] = pmin; D[D_PMAX] = pmax; D[D_TMIN] = tmin; D[D_TMAX] = tmax;
        D[D_DTCHEM] = D[D_DTCHEM_NEW];
        D[D_DTPHASE] = D[D_DTPHASE_NEW];
    }
}

// cumul d'un pas d'une série (un fil)
@compute @workgroup_size(1)
fn k_batch_acc() {
    D[D_B_BAD] = max(D[D_B_BAD], D[D_BAD]);
    D[D_B_REDO] = max(D[D_B_REDO], D[D_REDO]);
    D[D_B_TSUM] += D[D_DT];
    D[D_B_NSTEPS] += 1.0;
    D[D_B_SUBC] = max(D[D_B_SUBC], D[D_SUB_C]);
    D[D_B_NSUBS] = max(D[D_B_NSUBS], D[D_NSUB_S_MAX]);
    D[D_B_MAXPCG] = max(D[D_B_MAXPCG], D[D_MAXPCG_N]);
    D[D_B_NEWTON] = max(D[D_B_NEWTON], D[D_NEWTON_IT]);
    D[D_B_UNCONV] = max(D[D_B_UNCONV], D[D_PCG_UNCONV]);
}
