// ===== Rendu RGBA sur GPU (miroir de mp_view::render_rgba) : seule l'image (nx·ny·4 octets) est relue =====
// P.rd_mode : 0 phases, 1 pression, 2 température, 3 vitesse, 4 masse volumique, 5 vapeur, 6 divergence, 7 flamme, 8 dégagement de chaleur
// P.rd_flags : bit 0 = surcouche des effets, bit 1 = solides blancs

fn cell_vel(i: u32, j: u32) -> vec2<f32> {
    return vec2<f32>(0.5 * (S[O_U + uidx(i, j)] + S[O_U + uidx(i + 1u, j)]), 0.5 * (S[O_V + vidx(i, j)] + S[O_V + vidx(i, j + 1u)]));
}
@compute @workgroup_size(64)
fn k_rd_red(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>, @builtin(workgroup_id) w: vec3<u32>) {
    let c = g.x;
    var v: array<f32, 8>;
    v[0] = 1.0; v[1] = -3.0e38; v[2] = -3.0e38; v[3] = 0.0; v[4] = 1e-9; v[5] = 1e-9;
    if (c < ncell()) {
        let i = c % P.nx; let j = c / P.nx;
        let cv = cell_vel(i, j);
        v[4] = max(1e-9, length(cv));
        v[3] = abs(S[O_heat_rel + c]);
        if (!is_solid(c)) {
            v[0] = max(1.0, abs(S[O_prel + c] + P.p0 - 101325.0));
            v[1] = -S[O_rho_mix + c];
            v[2] = S[O_rho_mix + c];
            v[5] = max(1e-9, abs(div_uv(i, j)));
        }
    }
    wg_reduce(v, 6u, 1u, l.x, w.x, 0u);
}
@compute @workgroup_size(256)
fn k_rd_final(@builtin(local_invocation_id) l: vec3<u32>) {
    let nw = nwg_cells();
    var r: array<f32, 6>;
    for (var q = 0u; q < 6u; q++) { r[q] = final_reduce(q, nw, 1u, l.x); }
    if (l.x == 0u) { for (var q = 0u; q < 6u; q++) { D[D_SCALE + q] = r[q]; } }
}
fn hsv(h_in: f32, s: f32, v: f32) -> vec3<f32> {
    let h = ((h_in % 360.0) + 360.0) % 360.0;
    let c = v * s;
    let x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0));
    let m = v - c;
    let k = u32(h / 60.0);
    var r = vec3<f32>(c, 0.0, x);
    if (k == 0u) { r = vec3<f32>(c, x, 0.0); }
    else if (k == 1u) { r = vec3<f32>(x, c, 0.0); }
    else if (k == 2u) { r = vec3<f32>(0.0, c, x); }
    else if (k == 3u) { r = vec3<f32>(0.0, x, c); }
    else if (k == 4u) { r = vec3<f32>(x, 0.0, c); }
    return r + vec3<f32>(m, m, m);
}
fn blackbody(t: f32) -> vec3<f32> {
    let x = clamp((t - 800.0) / 1600.0, 0.0, 1.0);
    return vec3<f32>(min(1.6 * x, 1.0), clamp(1.6 * x - 0.45, 0.0, 0.9), clamp(1.6 * x - 1.0, 0.0, 0.6));
}
fn temp_rgb(xi: f32) -> vec3<f32> {
    let x = clamp(xi, -1.0, 1.0);
    if (x >= 0.0) { return vec3<f32>(min(2.0 * x, 1.0), max(2.0 * x - 1.0, 0.0), 0.0); }
    let c = -x;
    return vec3<f32>(0.0, max(2.0 * c - 1.0, 0.0), min(2.0 * c, 1.0));
}
fn signed_rgb(v: f32, m: f32) -> vec3<f32> {
    if (m <= 1e-12) { return vec3<f32>(0.0, 0.0, 0.0); }
    let x = clamp(v / m, -1.0, 1.0);
    if (x < 0.0) { return vec3<f32>(0.0, -x, -x); }
    return vec3<f32>(x, 0.0, 0.0);
}
fn phase_mix(c: u32, f: f32) -> vec3<f32> {
    var col = vec3<f32>(0.0, 0.0, 0.0);
    for (var k = 0u; k < P.np; k++) {
        let a = clamp(alpha(k, c), 0.0, 1.0);
        col += f * a * vec3<f32>(ph(k, 27u), ph(k, 28u), ph(k, 29u));
    }
    return col;
}
fn pack_rgb(c: vec3<f32>) -> u32 {
    let r = u32(clamp(c.x, 0.0, 1.0) * 255.0); let g = u32(clamp(c.y, 0.0, 1.0) * 255.0); let b = u32(clamp(c.z, 0.0, 1.0) * 255.0);
    return r | (g << 8u) | (b << 16u) | (255u << 24u);
}
@compute @workgroup_size(64)
fn k_render(@builtin(global_invocation_id) g: vec3<u32>) {
    let c = g.x;
    if (c >= ncell()) { return; }
    let i = c % P.nx; let j = c / P.nx;
    var pmax = P.rd_pscale;
    if (pmax <= 0.0) { pmax = D[D_SCALE + 0u]; }
    let rmin = -D[D_SCALE + 1u]; let rmax = D[D_SCALE + 2u];
    let qmax = D[D_SCALE + 3u]; let smax = D[D_SCALE + 4u]; let dmax = D[D_SCALE + 5u];
    let show = (P.rd_flags & 1u) != 0u;
    let white = (P.rd_flags & 2u) != 0u;
    let mode = P.rd_mode;
    let ts = max(P.rd_tscale, 1e-3);
    var rgb = vec3<f32>(0.0, 0.0, 0.0);
    let s = solid_id(c);
    if (s >= 0) {
        if (mode == 2u) { rgb = temp_rgb((S[O_temp + c] - P.t_amb) / ts); }
        else if (white) { rgb = vec3<f32>(1.0, 1.0, 1.0); }
        else {
            let th = S[O_th_kind + c];
            if (show && th > 0.5 && th < 1.5) { rgb = vec3<f32>(32.0, 96.0, 208.0) / 255.0; }
            else if (show && th > 1.5) { rgb = vec3<f32>(208.0, 128.0, 32.0) / 255.0; }
            else { let su = u32(s); rgb = vec3<f32>(sol(su, 4u), sol(su, 5u), sol(su, 6u)); }
        }
    } else {
        if (mode == 0u) { rgb = phase_mix(c, 1.0); }
        else if (mode == 1u) { rgb = signed_rgb(S[O_prel + c] + P.p0 - 101325.0, pmax); }
        else if (mode == 2u) { rgb = temp_rgb((S[O_temp + c] - P.t_amb) / ts); }
        else if (mode == 3u) {
            let v = cell_vel(i, j);
            let sp = length(v);
            rgb = hsv(atan2(v.y, v.x) * 57.29577951 + 180.0, 1.0, clamp(sp / smax, 0.0, 1.0));
        } else if (mode == 4u) {
            var x = 0.0;
            if (rmax > rmin) { x = clamp((log(S[O_rho_mix + c]) - log(rmin)) / (log(rmax) - log(rmin)), 0.0, 1.0); }
            rgb = vec3<f32>(x, x, x);
        } else if (mode == 5u) {
            var a = 0.0;
            for (var q = 0u; q < P.npc; q++) { a += alpha(u32(pcq(q, 1u)), c); }
            a = clamp(a, 0.0, 1.0);
            rgb = vec3<f32>(a, a, a);
        } else if (mode == 6u) { rgb = signed_rgb(div_uv(i, j), dmax); }
        else if (mode == 7u) {
            let col = phase_mix(c, 0.35);
            let b = blackbody(S[O_temp + c]);
            let q = S[O_heat_rel + c];
            var blue = 0.0;
            if (qmax > 0.0 && q > 0.0) { blue = clamp(log(q / qmax) / 2.302585093 / 4.0 + 1.0, 0.0, 1.0); }
            rgb = col + b + vec3<f32>(0.25 * blue, 0.45 * blue, blue);
        } else {
            let q = S[O_heat_rel + c];
            var x = 0.0;
            if (qmax > 0.0 && abs(q) > 0.0) { x = clamp(log(abs(q) / qmax) / 2.302585093 / 6.0 + 1.0, 0.0, 1.0); }
            if (q >= 0.0) { rgb = vec3<f32>(x, x * x, x * x * x * 0.5); } else { rgb = vec3<f32>(0.0, 0.3 * x, x); }
        }
    }
    var px = pack_rgb(rgb);
    if (show) {
        let o = u32(S[O_ovl + c]);
        if (o > 0u) { let rgbp = o - 1u; px = (rgbp & 0xffffffu) | (255u << 24u); }
    }
    R[(P.ny - 1u - j) * P.nx + i] = px;
}
