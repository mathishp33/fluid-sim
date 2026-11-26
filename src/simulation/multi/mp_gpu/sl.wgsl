// ===== Advection semi-lagrangienne de la vitesse (quantité de mouvement NON conservative) : bicubique bornée ou MacCormack =====
// Miroir de MultiphaseSim::advect_velocity. Écrit u* + forces dans Us / Vs (les forces F ont été évaluées sur l'état n).

struct Bl { v: f32, lo: f32, hi: f32 };
// champ w×h stocké à partir de `o`
fn bilerp(o: u32, w: u32, h: u32, gx_in: f32, gy_in: f32) -> Bl {
    let gx = clamp(gx_in, 0.0, f32(w - 1u));
    let gy = clamp(gy_in, 0.0, f32(h - 1u));
    let i0 = min(u32(floor(gx)), max(w, 2u) - 2u);
    let j0 = min(u32(floor(gy)), max(h, 2u) - 2u);
    let i1 = min(i0 + 1u, w - 1u); let j1 = min(j0 + 1u, h - 1u);
    let fx = gx - f32(i0); let fy = gy - f32(j0);
    let a = S[o + j0 * w + i0]; let b = S[o + j0 * w + i1]; let c = S[o + j1 * w + i0]; let d = S[o + j1 * w + i1];
    var r: Bl;
    r.v = (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (c * (1.0 - fx) + d * fx) * fy;
    r.lo = min(min(a, b), min(c, d));
    r.hi = max(max(a, b), max(c, d));
    return r;
}
fn at_(o: u32, w: u32, h: u32, i: i32, j: i32) -> f32 {
    return S[o + u32(clamp(j, 0, i32(h) - 1)) * w + u32(clamp(i, 0, i32(w) - 1))];
}
fn catrom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    return p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)));
}
fn bicubic_limited(o: u32, w: u32, h: u32, gx_in: f32, gy_in: f32) -> f32 {
    let bl = bilerp(o, w, h, gx_in, gy_in);
    let gx = clamp(gx_in, 0.0, f32(w - 1u));
    let gy = clamp(gy_in, 0.0, f32(h - 1u));
    let i0 = i32(floor(gx)); let j0 = i32(floor(gy));
    let fx = gx - f32(i0); let fy = gy - f32(j0);
    var col: array<f32, 4>;
    for (var q = 0; q < 4; q++) {
        let dj = q - 1;
        col[q] = catrom(at_(o, w, h, i0 - 1, j0 + dj), at_(o, w, h, i0, j0 + dj), at_(o, w, h, i0 + 1, j0 + dj), at_(o, w, h, i0 + 2, j0 + dj), fx);
    }
    let v = catrom(col[0], col[1], col[2], col[3], fy);
    if (!(v == v) || abs(v) > 1e30) { return bl.v; }
    return clamp(v, bl.lo, bl.hi);
}
fn samp_u(o: u32, x: f32, y: f32) -> Bl { return bilerp(o, P.nx + 1u, P.ny, x / P.dx, y / P.dx - 0.5); }
fn samp_v(o: u32, x: f32, y: f32) -> Bl { return bilerp(o, P.nx, P.ny + 1u, x / P.dx - 0.5, y / P.dx); }
fn vel_at(x: f32, y: f32) -> vec2<f32> { return vec2<f32>(samp_u(O_U, x, y).v, samp_v(O_V, x, y).v); }
fn trace(x: f32, y: f32, s: f32) -> vec2<f32> {
    let dt = dtv();
    let a = vel_at(x, y);
    let m = vec2<f32>(x - 0.5 * s * dt * a.x, y - 0.5 * s * dt * a.y);
    let b = vel_at(m.x, m.y);
    return vec2<f32>(x - s * dt * b.x, y - s * dt * b.y);
}
// passe 1 : u* (bicubique bornée, ou bilinéaire pour MacCormack)
@compute @workgroup_size(64)
fn k_sl_pass1(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    let mac = has(F_MACC);
    if (f < nfu()) {
        let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
        let p = trace(f32(i) * P.dx, (f32(j) + 0.5) * P.dx, 1.0);
        var u = 0.0;
        if (mac) { u = samp_u(O_U, p.x, p.y).v; } else { u = bicubic_limited(O_U, P.nx + 1u, P.ny, p.x / P.dx, p.y / P.dx - 0.5); }
        S[O_ust + f] = u;
        if (!mac) { S[O_Us + f] = u + S[O_Fx + f]; }
    }
    if (f < nfv()) {
        let i = f % P.nx; let j = f / P.nx;
        let p = trace((f32(i) + 0.5) * P.dx, f32(j) * P.dx, 1.0);
        var v = 0.0;
        if (mac) { v = samp_v(O_V, p.x, p.y).v; } else { v = bicubic_limited(O_V, P.nx, P.ny + 1u, p.x / P.dx - 0.5, p.y / P.dx); }
        S[O_vst + f] = v;
        if (!mac) { S[O_Vs + f] = v + S[O_Fy + f]; }
    }
}
// passe 2 (MacCormack) : correction limitée
@compute @workgroup_size(64)
fn k_sl_pass2(@builtin(global_invocation_id) g: vec3<u32>) {
    let f = g.x;
    if (f < nfu()) {
        let i = f % (P.nx + 1u); let j = f / (P.nx + 1u);
        let x = f32(i) * P.dx; let y = (f32(j) + 0.5) * P.dx;
        let pb = trace(x, y, 1.0);
        let bl = samp_u(O_U, pb.x, pb.y);
        let pf = trace(x, y, -1.0);
        let back = samp_u(O_ust, pf.x, pf.y).v;
        let c = S[O_ust + f] + 0.5 * (S[O_U + f] - back);
        S[O_Us + f] = clamp(c, bl.lo, bl.hi) + S[O_Fx + f];
    }
    if (f < nfv()) {
        let i = f % P.nx; let j = f / P.nx;
        let x = (f32(i) + 0.5) * P.dx; let y = f32(j) * P.dx;
        let pb = trace(x, y, 1.0);
        let bl = samp_v(O_V, pb.x, pb.y);
        let pf = trace(x, y, -1.0);
        let back = samp_v(O_vst, pf.x, pf.y).v;
        let c = S[O_vst + f] + 0.5 * (S[O_V + f] - back);
        S[O_Vs + f] = clamp(c, bl.lo, bl.hi) + S[O_Fy + f];
    }
}
