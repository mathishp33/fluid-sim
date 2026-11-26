// ===== Cellules spéciales (miroir de mp_cells.rs) : faces imposées, cellules épinglées, téléporteurs =====
// Tables statiques (calculées par l'hôte à l'envoi, comme prepare_special) :
//   pin_kind : 0 libre, 1 puits / source de téléporteur, 2 émetteur / destination ; pin_phase : phase pure (−1 = fluide ambiant) ;
//   pin_temp : T de remplissage ; pin_tp : case du mélange absorbé (destination de téléporteur, −1 sinon) ;
//   pfu_on / pfu, pfv_on / pfv : faces à vitesse imposée.
// Mélanges absorbés (persistants) : O_tpmix + s·(np + 2) = [valide, T, α_0 … α_np−1] ; paires (source, case) dans T_TP.

@compute @workgroup_size(64)
fn k_apply_pf(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < nfu() && S[O_pfu_on + f] > 0.5) { S[O_U + f] = S[O_pfu + f]; }
    if (f < nfv() && S[O_pfv_on + f] > 0.5) { S[O_V + f] = S[O_pfv + f]; }
}

// remplissage d'une cellule par des fractions volumiques (mélange absorbé d'un téléporteur) à (p courant, T)
fn fill_mix_tp(c: u32, slot: u32, t: f32) {
    let pr = S[O_prel + c];
    let n = ncell();
    let b = O_tpmix + slot * (P.np + 2u);
    for (var k = 0u; k < P.np; k++) {
        S[O_mass + k * n + c] = S[b + 2u + k] * rho_k(k, pr, t);
    }
    set_cell_temp(c, t);
}

@compute @workgroup_size(64)
fn k_pin_reset(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x; let n = ncell();
    if (c >= n) { return; }
    let kind = S[O_pin_kind + c];
    if (kind < 0.5) { return; }
    var before: array<f32, NPMAX>;
    for (var k = 0u; k < P.np; k++) { before[k] = mass(k, c); }
    let slot = i32(S[O_pin_tp + c]);
    var done = false;
    if (kind > 1.5 && slot >= 0) {
        let b = O_tpmix + u32(slot) * (P.np + 2u);
        if (S[b] > 0.5) { fill_mix_tp(c, u32(slot), S[b + 1u]); done = true; }
    }
    if (!done) {
        let ph_ = i32(S[O_pin_phase + c]);
        let t = S[O_pin_temp + c];
        if (ph_ < 0) { fill_ambient(c, t); } else { fill_alpha(c, u32(ph_), t); }
    }
    let a2 = P.dx * P.dx;
    for (var k = 0u; k < P.np; k++) { S[O_bflux + k * n + c] += (mass(k, c) - before[k]) * a2; }
}

// composition absorbée par chaque GROUPE de téléporteurs (toutes les sources d'une même destination : masses, énergie, capacité et
// p·masse additionnées, puis composition et température du mélange ; miroir de absorb_teleporters). Un seul fil.
@compute @workgroup_size(1)
fn k_tp_absorb() {
    let np = P.np;
    let st = np + 4u;   // [m_0 … m_np−1, ε, capacité, pente, Σ p_rel·m]
    for (var s = 0u; s < P.ntd; s++) {
        for (var q = 0u; q < st; q++) { S[O_tpacc + s * st + q] = 0.0; }
    }
    for (var q = 0u; q < P.ntp; q++) {
        let c = u32(T[T_TP + 2u * q]);
        let slot = u32(T[T_TP + 2u * q + 1u]);
        let pr = S[O_prel + c];
        let ta = P.t_amb;
        var ma: array<f32, NPMAX>;
        var tot = 0.0;
        var eps = S[O_eps + c];
        var cap = 0.0; var sl = 0.0;
        for (var k = 0u; k < np; k++) {
            let base = amb_alpha(k) * rho_k(k, pr, ta);
            let d = mass(k, c) - base;
            ma[k] = max(d, 0.0);
            tot += ma[k];
            // e_abs = e − e_base ⇒ ε' = ε − Σ base·erel(T_a) + Σ min(m − base, 0)·h0
            eps -= base * erel_k(k, ta);
            eps += min(d, 0.0) * h0_k(k);
            cap += ma[k] * ce_k(k); sl += ma[k] * slope_k(k);
        }
        if (tot < 1e-9) { continue; }
        let b = O_tpacc + slot * st;
        for (var k = 0u; k < np; k++) { S[b + k] += ma[k]; }
        S[b + np] += eps; S[b + np + 1u] += cap; S[b + np + 2u] += sl; S[b + np + 3u] += pr * tot;
    }
    for (var s = 0u; s < P.ntd; s++) {
        let b = O_tpacc + s * st;
        var tot = 0.0;
        for (var k = 0u; k < np; k++) { tot += S[b + k]; }
        if (tot < 1e-9) { continue; }
        let t = clamp(P.t0 + tau_from(S[b + np + 1u], S[b + np + 2u], S[b + np]), 100.0, 3000.0);
        let pr = S[b + np + 3u] / tot;
        var vs = 0.0;
        for (var k = 0u; k < np; k++) { vs += S[b + k] / rho_k(k, pr, t); }
        if (vs > 0.0) {
            let m = O_tpmix + s * (np + 2u);
            S[m] = 1.0; S[m + 1u] = t;
            for (var k = 0u; k < np; k++) { S[m + 2u + k] = S[b + k] / rho_k(k, pr, t) / vs; }
        }
    }
}
