// Simulation complète sur grille MAC (vitesses aux faces), exécutée sur GPU.
//
//   velocity_x[i] : face droite de la cellule i (entre i et i+1)
//   velocity_y[i] : face basse  de la cellule i (entre i et i+width)
//
// Les faces ouvertes (open_u / open_v) ne sont plus stockées : elles sont
// recalculées à la volée à partir de `cells` et du type des bordures.
//
// scratch (9 * width * height floats) :
// [0, n) : velocity_x temporaire  [n, 2n) : velocity_y temporaire
// [2n, 3n) : tracer temporaire   [3n, 4n) : temperature temporaire
// [4n, 5n) : temperature (champ persistant, en K)
// [5n, 6n) : divergence cible S (1/s), low-Mach, calculée côté CPU
// [6n, 7n) : source de chaleur q (W/m³)
// [7n, 8n) : rho (kg/m³), champ PERSISTANT en low-Mach (masse), envoyé et relu par le CPU
// [8n, 9n) : rho temporaire
//
// Low-Mach (params.lowmach == 1) : ρ est un champ persistant, advecté de façon conservative avec ρT
// (même flux de masse upwind) ; T = (ρT)/ρ. Les fantômes et les murs ont ρ = p_th/(r_spec·T).
// Ce ρ persistant ne sert QU'À l'advection (masse) ; la quantité de mouvement utilise ρ = EOS(T, P_th) :
//   - projection à densité variable : poids de face = ρ_amb/ρ_face (voir cond_u / cond_v)
//   - Poisson : Σ poids·(p_voisin − p) = div − S
//   - flottabilité : g·(ρ_amb/ρ_EOS − 1)
//   - tracer : forme non conservative (flux − Y·div u)
// Équation d'énergie (temperature_step), hors low-Mach :
//   dT/dt = −u·∇T + (∇·(k∇T) + q + dP_th/dt)/(ρ·cp)
// Low-Mach, cellules de gaz ordinaires :
//   ∂ρ/∂t = −∇·(ρu)     ∂(ρT)/∂t = −∇·(ρuT) + (∇·(k∇T) + q + dP_th/dt)/cp

struct Params {
    width: u32,
    height: u32,
    omega: f32,
    dt: f32,
    dts: f32,
    diffusion: f32,
    vmax: f32,
    special_count: u32,
    visc_k: f32, // dt_sous-pas / dx² (la viscosité dynamique μ(T) et ρ sont locales)
    gravity: f32, // g / dx (cellules/s²)
    t_ref: f32,
    beta: f32,
    kind: u32, // 0 = gaz, 1 = liquide
    lowmach: u32, // 1 = gaz low-Mach
    p_th: f32, // pression thermodynamique du début de pas (Pa)
    r_spec: f32,
    rho_amb: f32, // kg/m³ de référence
    dpdt: f32, // dP_th/dt (Pa/s)
    k_fluid: f32, // W/(m·K)
    k_solid: f32,
    cp: f32, // J/(kg·K)
    cap_solid: f32, // ρ_s·cp_s, J/(m³·K)
    inv_dx2: f32, // 1/dx²
    noslip: u32, // 1 = parois à adhérence
    maccormack: u32, // 1 = advection de la vitesse en MacCormack limité (sinon semi-lagrangienne)
    muscl: u32, // 1 = advection des scalaires (tracer, T, rho) en MUSCL van Leer, 0 = upwind
    visc_law: u32, // loi μ(T) : 0 constante, 1 Sutherland, 2 Vogel, 3 Arrhenius, 4 linéaire
    visc_a: f32,
    visc_b: f32,
    cond_law: u32, // loi k(T)
    cond_a: f32,
    cond_b: f32,
    mu_ref: f32, // μ_ref = ν_ref·ρ_ambient (Pa·s)
    varprops: u32, // 1 = propriétés variables (μ(T), k(T), ν = μ/ρ) ; 0 = ν et k constantes
    pad1: u32,
    pad2: u32,
    // type de bordure : gauche, droite, haut, bas (0 = Wall, 1 = Inlet, 2 = Outlet)
    btype: vec4<u32>,
    // (velocity_x, velocity_y, tracer, offset)
    bleft: vec4<f32>,
    bright: vec4<f32>,
    btop: vec4<f32>,
    bbottom: vec4<f32>,
    btemp: vec4<f32>, // temp des bords : gauche, droite, haut, bas
};

// kind : 1 = Sink (a = rate), 2 = Emitter (a = vx, b = vy, c = tracer),
//        3 = Accelerator (a = ax, b = ay), 4 = Teleporter (dest)
struct Special {
    idx: u32,
    kind: u32,
    a: f32,
    b: f32,
    c: f32,
    dest: u32,
};

@group(0) @binding(0) var<storage, read_write> velocity_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> velocity_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(3) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(4) var<storage, read> cells: array<u32>;
@group(0) @binding(5) var<storage, read_write> tracer: array<f32>;
@group(0) @binding(6) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(7) var<storage, read> specials: array<Special>;
@group(0) @binding(8) var<uniform> params: Params;

const CELL_WALL: u32 = 1u;
const BT_WALL: u32 = 0u;
const BT_INLET: u32 = 1u;
const BT_OUTLET: u32 = 2u;

const CELL_SINK: u32 = 2u;
const CELL_EMITTER: u32 = 3u;

fn is_sink(i: u32) -> bool {
    return cells[i] == CELL_SINK;
}

fn is_solid(i: u32) -> bool {
    return cells[i] == CELL_WALL;
}

fn interior(x: u32, y: u32) -> bool {
    return x >= 1u && y >= 1u && x < params.width - 1u && y < params.height - 1u;
}

fn open_u_at(x: u32, y: u32) -> bool {
    let w = params.width;
    let h = params.height;
    if (y == 0u || y >= h - 1u || x >= w - 1u) {
        return false;
    }
    let i = x + y * w;
    if (x == 0u) {
        return params.btype.x == BT_OUTLET && !is_solid(i + 1u);
    }
    if (x == w - 2u) {
        return params.btype.y == BT_OUTLET && !is_solid(i);
    }
    return !is_solid(i) && !is_solid(i + 1u);
}

fn open_v_at(x: u32, y: u32) -> bool {
    let w = params.width;
    let h = params.height;
    if (y >= h - 1u || x == 0u || x >= w - 1u) {
        return false;
    }
    let i = x + y * w;
    if (y == 0u) {
        return params.btype.z == BT_OUTLET && !is_solid(i + w);
    }
    if (y == h - 2u) {
        return params.btype.w == BT_OUTLET && !is_solid(i);
    }
    return !is_solid(i) && !is_solid(i + w);
}

fn open_u_f(x: u32, y: u32) -> f32 {
    if (open_u_at(x, y)) { return 1.0; }
    return 0.0;
}

fn open_v_f(x: u32, y: u32) -> f32 {
    if (open_v_at(x, y)) { return 1.0; }
    return 0.0;
}

// ρ (kg/m³) d'un gaz parfait à P_th uniforme
fn rho_of_t(t: f32) -> f32 {
    return params.p_th / (params.r_spec * max(t, 1.0));
}

// ρ persistant (low-Mach)
fn rho_get(i: u32) -> f32 {
    return scratch[7u * params.width * params.height + i];
}

fn rho_set(i: u32, v: f32) {
    scratch[7u * params.width * params.height + i] = v;
}

// poids de face du solveur de pression : ρ_amb/ρ_face (ρ_face = moyenne arithmétique), 1 si incompressible
fn beta_face(i: u32, j: u32) -> f32 {
    if (params.lowmach == 0u) {
        return 1.0;
    }
    // ρ de l'EOS (T, P_th), PAS le ρ persistant : un résidu de projection ne doit pas pouvoir raidir le solveur
    return 2.0 * params.rho_amb / max(rho_of_t(temp_get(i)) + rho_of_t(temp_get(j)), 1e-9);
}

// conductance = face ouverte * poids de densité
fn cond_u(x: u32, y: u32) -> f32 {
    if (!open_u_at(x, y)) {
        return 0.0;
    }
    let i = x + y * params.width;
    return beta_face(i, i + 1u);
}

fn cond_v(x: u32, y: u32) -> f32 {
    if (!open_v_at(x, y)) {
        return 0.0;
    }
    let i = x + y * params.width;
    return beta_face(i, i + params.width);
}

// divergence cible S de la cellule i (0 si incompressible)
fn source_at(i: u32) -> f32 {
    if (params.lowmach == 0u) {
        return 0.0;
    }
    return scratch[5u * params.width * params.height + i];
}


// valeur de la face i : mode 0 = vitesse courante, mode 1 = prédiction semi-lagrangienne q̂ (scratch)
fn face_val(horiz: bool, mode: u32, i: u32) -> f32 {
    if (mode == 0u) {
        if (horiz) {
            return velocity_x[i];
        }
        return velocity_y[i];
    }
    let n = params.width * params.height;
    if (horiz) {
        return scratch[i];
    }
    return scratch[n + i];
}

fn sample_face(horiz: bool, px: f32, py: f32, fallback: f32) -> f32 {
    return sample_face_ex(horiz, 0u, px, py, fallback).x;
}

// Interpolation bilinéaire d'une composante de vitesse. Renvoie (valeur, min, max) des taps valides :
// min/max servent de limiteur au MacCormack.
fn sample_face_ex(horiz: bool, mode: u32, px: f32, py: f32, fallback: f32) -> vec3<f32> {
    let w = params.width;
    let h = params.height;

    var gx: f32;
    var gy: f32;
    var xm: u32;
    var ym: u32;
    if (horiz) {
        gx = px - 0.5;
        gy = py;
        xm = w - 2u;
        ym = h - 1u;
    } else {
        gx = px;
        gy = py - 0.5;
        xm = w - 1u;
        ym = h - 2u;
    }
    gx = clamp(gx, 0.0, f32(xm));
    gy = clamp(gy, 0.0, f32(ym));

    let x0 = u32(floor(gx));
    let y0 = u32(floor(gy));
    let x1 = min(x0 + 1u, xm);
    let y1 = min(y0 + 1u, ym);
    let sx = gx - f32(x0);
    let sy = gy - f32(y0);

    var stride = 1u;
    if (!horiz) {
        stride = w;
    }

    var sum = 0.0;
    var weight = 0.0;
    var lo = 3.0e38;
    var hi = -3.0e38;
    for (var k = 0u; k < 4u; k++) {
        var tx = x0;
        var ty = y0;
        var wt = (1.0 - sx) * (1.0 - sy);
        if (k == 1u) {
            tx = x1;
            wt = sx * (1.0 - sy);
        } else if (k == 2u) {
            ty = y1;
            wt = (1.0 - sx) * sy;
        } else if (k == 3u) {
            tx = x1;
            ty = y1;
            wt = sx * sy;
        }
        let i = tx + ty * w;
        if (is_solid(i) && is_solid(i + stride)) {
            continue;
        }
        let val = face_val(horiz, mode, i);
        sum += val * wt;
        weight += wt;
        lo = min(lo, val);
        hi = max(hi, val);
    }

    if (weight > 1e-6) {
        return vec3<f32>(sum / weight, lo, hi);
    }
    return vec3<f32>(fallback, fallback, fallback);
}

// Écrit les deux composantes dans scratch
@compute @workgroup_size(8, 8)
fn advect_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let x = gid.x;
    let y = gid.y;
    let i = x + y * w;
    let n = w * h;
    let dt = params.dt;

    var nu = velocity_x[i];
    var nv = velocity_y[i];

    if (x >= 1u && x + 2u < w && y >= 1u && y + 1u < h && open_u_at(x, y)) {
        let px = f32(x) + 0.5;
        let py = f32(y);
        let vx = velocity_x[i];
        let vy = sample_face(false, px, py, 0.0);
        nu = sample_face(true, px - vx * dt, py - vy * dt, vx);
    }

    if (x >= 1u && x + 1u < w && y >= 1u && y + 2u < h && open_v_at(x, y)) {
        let px = f32(x);
        let py = f32(y) + 0.5;
        let vy = velocity_y[i];
        let vx = sample_face(true, px, py, 0.0);
        nv = sample_face(false, px - vx * dt, py - vy * dt, vy);
    }

    scratch[i] = nu;
    scratch[n + i] = nv;
}

// MacCormack limité : q' = q̂ + ½(q − q̃), borné par le min/max des taps de la prédiction.
//   q  = vitesse courante (velocity_x / velocity_y, NON modifiée par advect_velocity)
//   q̂  = prédiction semi-lagrangienne (scratch[0, 2n), écrite par advect_velocity)
//   q̃  = prédiction ré-advectée en sens inverse (échantillonnée dans q̂ à +v·dt)
// Sortie dans scratch[2n, 4n) (régions libres pendant la phase vitesse), recopiée par commit_velocity_mc.
@compute @workgroup_size(8, 8)
fn advect_correct(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let x = gid.x;
    let y = gid.y;
    let i = x + y * w;
    let n = w * h;
    let dt = params.dt;

    var ru = velocity_x[i];
    var rv = velocity_y[i];

    if (x >= 1u && x + 2u < w && y >= 1u && y + 1u < h && open_u_at(x, y)) {
        let px = f32(x) + 0.5;
        let py = f32(y);
        let vx = velocity_x[i];
        let vy = sample_face(false, px, py, 0.0);
        let a = sample_face_ex(true, 0u, px - vx * dt, py - vy * dt, vx);
        let back = sample_face_ex(true, 1u, px + vx * dt, py + vy * dt, scratch[i]).x;
        let mc = scratch[i] + 0.5 * (velocity_x[i] - back);
        ru = min(max(mc, a.y), a.z);
    }

    if (x >= 1u && x + 1u < w && y >= 1u && y + 2u < h && open_v_at(x, y)) {
        let px = f32(x);
        let py = f32(y) + 0.5;
        let vy = velocity_y[i];
        let vx = sample_face(true, px, py, 0.0);
        let a = sample_face_ex(false, 0u, px - vx * dt, py - vy * dt, vy);
        let back = sample_face_ex(false, 1u, px + vx * dt, py + vy * dt, scratch[n + i]).x;
        let mc = scratch[n + i] + 0.5 * (velocity_y[i] - back);
        rv = min(max(mc, a.y), a.z);
    }

    scratch[2u * n + i] = ru;
    scratch[3u * n + i] = rv;
}

@compute @workgroup_size(8, 8)
fn commit_velocity_mc(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let i = gid.x + gid.y * w;
    let n = w * h;
    velocity_x[i] = scratch[2u * n + i];
    velocity_y[i] = scratch[3u * n + i];
}

@compute @workgroup_size(8, 8)
fn commit_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let i = gid.x + gid.y * w;
    let n = w * h;
    velocity_x[i] = scratch[i];
    velocity_y[i] = scratch[n + i];
}


@compute @workgroup_size(1)
fn special_velocity() {
    for (var k = 0u; k < params.special_count; k++) {
        let s = specials[k];
        if (s.kind == 1u) {
            let w = params.width;
            velocity_x[s.idx] = -s.a;
            velocity_x[s.idx - 1u] = s.a;
            velocity_y[s.idx] = -s.a;
            velocity_y[s.idx - w] = s.a;
            pressure[s.idx] = 0.0;
        } else if (s.kind == 2u) {
            velocity_x[s.idx] = s.a;
            velocity_y[s.idx] = s.b;
        } else if (s.kind == 3u) {
            velocity_x[s.idx] += s.a * params.dt;
            velocity_y[s.idx] += s.b * params.dt;
        } else if (s.kind == 4u) {
            if (s.dest != s.idx && !is_solid(s.dest)) {
                velocity_x[s.dest] = velocity_x[s.idx];
                velocity_y[s.dest] = velocity_y[s.idx];
            }
        }
    }
}

@compute @workgroup_size(1)
fn special_tracer() {
    for (var k = 0u; k < params.special_count; k++) {
        let s = specials[k];
        if (s.kind == 1u) {
            tracer[s.idx] = 0.0;
        } else if (s.kind == 2u) {
            tracer[s.idx] = s.c;
        } else if (s.kind == 4u) {
            if (s.dest != s.idx && !is_solid(s.dest)) {
                tracer[s.dest] = min(tracer[s.dest] + tracer[s.idx], 1.0);
                tracer[s.idx] = 0.0;
            }
        }
    }
}

// CL

// faces intérieures fermées par un solide -> 0
@compute @workgroup_size(8, 8)
fn close_faces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    let x = gid.x;
    let y = gid.y;
    if (x >= w || y >= h) {
        return;
    }
    let i = x + y * w;
    if (y >= 1u && y + 1u < h && x >= 1u && x + 2u < w && !open_u_at(x, y)) {
        velocity_x[i] = 0.0;
    }
    if (y >= 1u && y + 2u < h && x >= 1u && x + 1u < w && !open_v_at(x, y)) {
        velocity_y[i] = 0.0;
    }
}

fn offset_factor(x: u32, size: u32, offset: u32) -> f32 {
    if (offset == 0u) {
        return 1.0;
    }
    if (size <= 2u || offset >= size) {
        return 0.0;
    }
    let dist = min(x, size - 1u - x);
    return clamp(f32(dist) / f32(offset), 0.0, 1.0);
}

// (composante normale, composante tangentielle fantôme)
// no-slip : fantôme = 2·U_paroi − intérieur ; glissement libre : copie de l'intérieur
fn edge_values(t: u32, f: f32, in_n: f32, in_t: f32, inner_n: f32, inner_t: f32) -> vec2<f32> {
    if (t == BT_WALL) {
        if (params.noslip == 1u) {
            return vec2<f32>(0.0, -inner_t);
        }
        return vec2<f32>(0.0, inner_t);
    }
    if (t == BT_OUTLET) {
        return vec2<f32>(inner_n, inner_t); // gradient = 0
    }
    if (params.noslip == 1u) {
        return vec2<f32>(in_n * f, 2.0 * in_t * f - inner_t);
    }
    return vec2<f32>(in_n * f, in_t * f);
}

@compute @workgroup_size(64)
fn boundary_lr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    let y = gid.x;
    if (y < 1u || y >= h - 1u) {
        return;
    }
    let row = y * w;

    let bl = params.bleft;
    let fl = offset_factor(y, h, u32(bl.w));
    let el = edge_values(params.btype.x, fl, bl.x, bl.y, velocity_x[row + 1u], velocity_y[row + 1u]);
    velocity_x[row] = select(el.x, 0.0, is_solid(row + 1u));
    velocity_y[row] = el.y;

    let br = params.bright;
    let fr = offset_factor(y, h, u32(br.w));
    let er = edge_values(params.btype.y, fr, br.x, br.y, velocity_x[row + w - 3u], velocity_y[row + w - 2u]);
    velocity_x[row + w - 2u] = select(er.x, 0.0, is_solid(row + w - 2u));
    velocity_y[row + w - 1u] = er.y;
}

@compute @workgroup_size(64)
fn boundary_tb(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    let x = gid.x;
    if (x < 1u || x >= w - 1u) {
        return;
    }
    let last = (h - 2u) * w;

    let bt = params.btop;
    let ft = offset_factor(x, w, u32(bt.w));
    let et = edge_values(params.btype.z, ft, bt.y, bt.x, velocity_y[x + w], velocity_x[x + w]);
    velocity_y[x] = select(et.x, 0.0, is_solid(x + w));
    velocity_x[x] = et.y;

    let bb = params.bbottom;
    let fb = offset_factor(x, w, u32(bb.w));
    let eb = edge_values(params.btype.w, fb, bb.y, bb.x, velocity_y[x + last - w], velocity_x[x + last]);
    velocity_y[x + last] = select(eb.x, 0.0, is_solid(x + last));
    velocity_x[x + last + w] = eb.y;
}

fn sanitize(v: f32) -> f32 {
    // NaN/Inf -> 0, sinon clamp à ±vmax
    if ((bitcast<u32>(v) & 0x7f800000u) == 0x7f800000u) {
        return 0.0;
    }
    return clamp(v, -params.vmax, params.vmax);
}

@compute @workgroup_size(8, 8)
fn limit_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let i = gid.x + gid.y * w;
    velocity_x[i] = sanitize(velocity_x[i]);
    velocity_y[i] = sanitize(velocity_y[i]);
}


@compute @workgroup_size(8, 8)
fn calculate_divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!interior(gid.x, gid.y)) {
        return;
    }
    let w = params.width;
    let i = gid.x + gid.y * w;

    if (is_solid(i) || is_sink(i)) {
        divergence[i] = 0.0;
        return;
    }

    divergence[i] = velocity_x[i] - velocity_x[i - 1u] + velocity_y[i] - velocity_y[i - w];
}

fn rb_sweep(gid: vec3<u32>, color: u32) {
    if (!interior(gid.x, gid.y) || ((gid.x + gid.y) & 1u) != color) {
        return;
    }
    let w = params.width;
    let x = gid.x;
    let y = gid.y;
    let i = x + y * w;

    if (is_solid(i) || is_sink(i)) {
        return;
    }

    let a = cond_u(x, y);
    let b = cond_u(x - 1u, y);
    let c = cond_v(x, y);
    let d = cond_v(x, y - 1u);
    let n = a + b + c + d;
    if (n == 0.0) {
        return;
    }

    let s = a * pressure[i + 1u] + b * pressure[i - 1u] + c * pressure[i + w] + d * pressure[i - w];
    let p_new = (s - (divergence[i] - source_at(i))) / n;
    pressure[i] = mix(pressure[i], p_new, params.omega);
}

@compute @workgroup_size(8, 8)
fn solve_red(@builtin(global_invocation_id) gid: vec3<u32>) {
    rb_sweep(gid, 0u);
}

@compute @workgroup_size(8, 8)
fn solve_black(@builtin(global_invocation_id) gid: vec3<u32>) {
    rb_sweep(gid, 1u);
}

@compute @workgroup_size(8, 8)
fn correct_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    let x = gid.x;
    let y = gid.y;
    if (x >= w || y >= h) {
        return;
    }
    let i = x + y * w;

    let cu = cond_u(x, y);
    if (cu > 0.0) {
        velocity_x[i] = velocity_x[i] - cu * (pressure[i + 1u] - pressure[i]);
    }
    let cv = cond_v(x, y);
    if (cv > 0.0) {
        velocity_y[i] = velocity_y[i] - cv * (pressure[i + w] - pressure[i]);
    }
}

fn ghost_value(t: u32, inlet_d: f32, mode: u32, inner: f32) -> f32 {
    if (t == BT_INLET) {
        return inlet_d;
    }
    if (t == BT_OUTLET && mode == 0u) {
        return 0.0;
    }
    return inner;
}

fn ghost_tracer(t: u32, mode: u32) {
    let w = params.width;
    let h = params.height;

    if (t >= 1u && t < h - 1u) {
        let l = t * w;
        let r = l + w - 1u;
        tracer[l] = ghost_value(params.btype.x, params.bleft.z, mode, tracer[l + 1u]);
        tracer[r] = ghost_value(params.btype.y, params.bright.z, mode, tracer[r - 1u]);
    }
    if (t >= 1u && t < w - 1u) {
        let tp = t;
        let bt = t + (h - 1u) * w;
        tracer[tp] = ghost_value(params.btype.z, params.btop.z, mode, tracer[tp + w]);
        tracer[bt] = ghost_value(params.btype.w, params.bbottom.z, mode, tracer[bt - w]);
    }
}

@compute @workgroup_size(64)
fn prepare_ghost_tracer(@builtin(global_invocation_id) gid: vec3<u32>) {
    ghost_tracer(gid.x, 0u);
}

@compute @workgroup_size(64)
fn apply_tracer_boundaries(@builtin(global_invocation_id) gid: vec3<u32>) {
    ghost_tracer(gid.x, 1u);
}

// ---- Advection d'ordre 2 des scalaires (même algorithme que le CPU : recon/pick de cpu.rs) ----
// Face entre l'amont U et l'aval D, UU = amont de U (aucun au bord, contre un mur ou si muscl = 0) :
//  a = φ_U − φ_UU, b = φ_D − φ_U, φ_face = φ_U + ½(a|b| + |a|b)/(|a| + |b|)   (van Leer : TVD)
const NO_CELL: u32 = 0xffffffffu;

fn recon(fu: f32, fd: f32, fuu: f32, has_uu: bool) -> f32 {
    if (!has_uu) {
        return fu;
    }
    let a = fu - fuu;
    let b = fd - fu;
    let den = abs(a) + abs(b);
    if (den > 0.0) {
        return fu + 0.5 * (a * abs(b) + abs(a) * b) / den;
    }
    return fu;
}

// k = cellule « avant » de la face (face entre k et k+stride), along = coordonnée le long de l'axe, len = taille de l'axe
// résultat : (U, D, UU)
fn pick(k: u32, stride: u32, along: u32, len: u32, uf: f32) -> vec3<u32> {
    var r = vec3<u32>(0u, 0u, NO_CELL);
    let hi = params.muscl == 1u;
    if (uf > 0.0) {
        r.x = k;
        r.y = k + stride;
        if (hi && along >= 1u && cells[k - stride] != CELL_WALL) {
            r.z = k - stride;
        }
    } else {
        r.x = k + stride;
        r.y = k;
        if (hi && along + 2u < len && cells[k + 2u * stride] != CELL_WALL) {
            r.z = k + 2u * stride;
        }
    }
    return r;
}

fn tracer_face(k: u32, stride: u32, along: u32, len: u32, uf: f32) -> f32 {
    let p = pick(k, stride, along, len, uf);
    var fuu = 0.0;
    if (p.z != NO_CELL) {
        fuu = tracer[p.z];
    }
    return recon(tracer[p.x], tracer[p.y], fuu, p.z != NO_CELL);
}

fn flux_x(x: u32, y: u32) -> f32 {
    let w = params.width;
    let i = x + y * w;
    let uf = velocity_x[i];
    var adv = 0.0;
    if (uf != 0.0) {
        adv = uf * tracer_face(i, 1u, x, w, uf);
    }
    var dif = 0.0;
    if (x >= 1u && x + 2u < w && open_u_at(x, y)) {
        dif = -params.diffusion * (tracer[i + 1u] - tracer[i]);
    }
    return adv + dif;
}

fn flux_y(x: u32, y: u32) -> f32 {
    let w = params.width;
    let h = params.height;
    let i = x + y * w;
    let vf = velocity_y[i];
    var adv = 0.0;
    if (vf != 0.0) {
        adv = vf * tracer_face(i, w, y, h, vf);
    }
    var dif = 0.0;
    if (y >= 1u && y + 2u < h && open_v_at(x, y)) {
        dif = -params.diffusion * (tracer[i + w] - tracer[i]);
    }
    return adv + dif;
}

// Un sous-pas : lit tracer, écrit scratch[2n + i] (cellules intérieures).
@compute @workgroup_size(8, 8)
fn tracer_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (!interior(x, y)) {
        return;
    }
    let w = params.width;
    let n = w * params.height;
    let i = x + y * w;

    if (is_solid(i)) {
        scratch[2u * n + i] = 0.0;
        return;
    }
    var val = tracer[i] - params.dts * (flux_x(x, y) - flux_x(x - 1u, y) + flux_y(x, y) - flux_y(x, y - 1u));
    if (params.lowmach == 1u) {
        // fraction (pas une densité) : on retire Y·div u pour ne pas la comprimer/diluer
        let div = velocity_x[i] - velocity_x[i - 1u] + velocity_y[i] - velocity_y[i - w];
        val += params.dts * tracer[i] * div;
    }
    scratch[2u * n + i] = val;
}

@compute @workgroup_size(8, 8)
fn tracer_commit(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!interior(gid.x, gid.y)) {
        return;
    }
    let w = params.width;
    let n = w * params.height;
    let i = gid.x + gid.y * w;
    tracer[i] = scratch[2u * n + i];
}

// Facteur f(T) (f(T_ref) = 1) des lois de propriétés : mêmes formules que PropertyLaw::factor (materials.rs)
fn law_factor(id: u32, a: f32, b: f32, t_in: f32) -> f32 {
    let t = clamp(t_in, 100.0, 4000.0);
    let tr = params.t_ref;
    var f = 1.0;
    if (id == 1u) { // Sutherland : (T/Tr)^1,5 (Tr + S)/(T + S)
        let r = t / tr;
        f = r * sqrt(r) * (tr + a) / (t + a);
    } else if (id == 2u) { // Vogel : 10^{b (1/(T − c) − 1/(Tr − c))}
        let d = max(t - b, 20.0);
        let dr = max(tr - b, 20.0);
        f = exp(2.302585093 * a * (1.0 / d - 1.0 / dr));
    } else if (id == 3u) { // Arrhenius
        f = exp(a * (1.0 / t - 1.0 / tr));
    } else if (id == 4u) { // linéaire relatif
        f = 1.0 + a * (t - tr);
    }
    return clamp(f, 0.01, 100.0);
}

// viscosité dynamique de la cellule i (Pa·s)
fn mu_cell(i: u32) -> f32 {
    if (params.varprops == 1u && !is_solid(i)) {
        return params.mu_ref * law_factor(params.visc_law, params.visc_a, params.visc_b, temp_get(i));
    }
    return params.mu_ref;
}

// ρ utilisée pour ν = μ/ρ : EOS (T, P_th) en low-Mach avec propriétés variables, sinon ρ_ambient
fn rho_visc(i: u32) -> f32 {
    if (params.varprops == 1u && params.lowmach == 1u) {
        return max(rho_of_t(temp_get(i)), 1e-6);
    }
    return params.rho_amb;
}

fn mu_fu(j: u32) -> f32 { return 0.5 * (mu_cell(j) + mu_cell(j + 1u)); }
fn mu_fv(j: u32) -> f32 { return 0.5 * (mu_cell(j) + mu_cell(j + params.width)); }

// Contribution d'un voisin au laplacien (même règles que le CPU) :
//  face ouverte : val − c ; fermée : 0 (glissement libre) ; no-slip : normale ou fantôme -> val − c,
//  tangentielle contre un solide -> −2c (paroi à mi-cellule)
fn nb_term(open: bool, mu_l: f32, val: f32, c: f32, normal: bool, ghost: bool) -> f32 {
    if (open) {
        return mu_l * (val - c);
    }
    if (params.noslip == 0u) {
        return 0.0;
    }
    if (normal || ghost) {
        return mu_l * (val - c);
    }
    return -2.0 * mu_l * c;
}

// μ du lien : moyenne des μ des deux faces s'il est ouvert, μ de la face sinon (même règle que le CPU)
fn link_mu(open: bool, m0: f32, mj: f32) -> f32 {
    if (open) {
        return 0.5 * (m0 + mj);
    }
    return m0;
}

@compute @workgroup_size(8, 8)
fn diffuse_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let x = gid.x;
    let y = gid.y;
    let i = x + y * w;
    let n = w * h;
    let k = params.visc_k;

    var nu = velocity_x[i];
    var nv = velocity_y[i];

    if (x >= 1u && x + 2u < w && y >= 1u && y + 1u < h && open_u_at(x, y)) {
        let c = velocity_x[i];
        let m0 = mu_fu(i);
        let or = open_u_at(x + 1u, y);
        let ol = open_u_at(x - 1u, y);
        let od = open_u_at(x, y + 1u);
        let oup = open_u_at(x, y - 1u);
        let lap = nb_term(or, link_mu(or, m0, mu_fu(i + 1u)), velocity_x[i + 1u], c, true, false)
            + nb_term(ol, link_mu(ol, m0, mu_fu(i - 1u)), velocity_x[i - 1u], c, true, false)
            + nb_term(od, link_mu(od, m0, mu_fu(i + w)), velocity_x[i + w], c, false, y + 1u == h - 1u)
            + nb_term(oup, link_mu(oup, m0, mu_fu(i - w)), velocity_x[i - w], c, false, y == 1u);
        nu = c + k * lap / (0.5 * (rho_visc(i) + rho_visc(i + 1u)));
    }

    if (x >= 1u && x + 1u < w && y >= 1u && y + 2u < h && open_v_at(x, y)) {
        let c = velocity_y[i];
        let m0 = mu_fv(i);
        let or = open_v_at(x + 1u, y);
        let ol = open_v_at(x - 1u, y);
        let od = open_v_at(x, y + 1u);
        let oup = open_v_at(x, y - 1u);
        let lap = nb_term(or, link_mu(or, m0, mu_fv(i + 1u)), velocity_y[i + 1u], c, false, x + 1u == w - 1u)
            + nb_term(ol, link_mu(ol, m0, mu_fv(i - 1u)), velocity_y[i - 1u], c, false, x == 1u)
            + nb_term(od, link_mu(od, m0, mu_fv(i + w)), velocity_y[i + w], c, true, false)
            + nb_term(oup, link_mu(oup, m0, mu_fv(i - w)), velocity_y[i - w], c, true, false);
        nv = c + k * lap / (0.5 * (rho_visc(i) + rho_visc(i + w)));
    }

    scratch[i] = nu;
    scratch[n + i] = nv; // puis commit_velocity
}

fn temp_get(i: u32) -> f32 {
    return scratch[4u * params.width * params.height + i];
}

fn temp_set(i: u32, v: f32) {
    scratch[4u * params.width * params.height + i] = v;
}

fn buoyancy_factor(t: f32) -> f32 {
    if (params.kind == 0u) { return (t - params.t_ref) / max(t, 1.0); }
    return params.beta * (t - params.t_ref);
}

// facteur de flottabilité de la cellule i : low-Mach -> ρ = EOS(T, P_th), sinon Boussinesq via T
fn buoyancy_cell(i: u32) -> f32 {
    if (params.lowmach == 1u) {
        return params.rho_amb / rho_of_t(temp_get(i)) - 1.0;
    }
    return buoyancy_factor(temp_get(i));
}

@compute @workgroup_size(8, 8)
fn buoyancy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    let x = gid.x;
    let y = gid.y;
    if (x < 1u || y < 1u || x + 1u >= w || y + 2u >= h) {
        return;
    }
    if (!open_v_at(x, y)) {
        return;
    }
    let i = x + y * w;
    velocity_y[i] = velocity_y[i] - params.gravity * params.dt * 0.5 * (buoyancy_cell(i) + buoyancy_cell(i + w));
}

fn ghost_temp_value(t: u32, inlet_t: f32, mode: u32, inner: f32) -> f32 {
    if (t == BT_INLET) {
        return inlet_t;
    }
    if (t == BT_OUTLET && mode == 0u) {
        return params.t_ref;
    }
    return inner;
}

// fantôme : T, et ρ = EOS(T) en low-Mach (sert aux flux de masse des faces de bord)
fn set_ghost(i: u32, tv: f32) {
    temp_set(i, tv);
    if (params.lowmach == 1u) {
        rho_set(i, rho_of_t(tv));
    }
}

fn ghost_temperature(t: u32, mode: u32) {
    let w = params.width;
    let h = params.height;
    if (t >= 1u && t < h - 1u) {
        let l = t * w;
        let r = l + w - 1u;
        set_ghost(l, ghost_temp_value(params.btype.x, params.btemp.x, mode, temp_get(l + 1u)));
        set_ghost(r, ghost_temp_value(params.btype.y, params.btemp.y, mode, temp_get(r - 1u)));
    }
    if (t >= 1u && t < w - 1u) {
        let bt = t + (h - 1u) * w;
        set_ghost(t, ghost_temp_value(params.btype.z, params.btemp.z, mode, temp_get(t + w)));
        set_ghost(bt, ghost_temp_value(params.btype.w, params.btemp.w, mode, temp_get(bt - w)));
    }
}

@compute @workgroup_size(64)
fn prepare_ghost_temperature(@builtin(global_invocation_id) gid: vec3<u32>) {
    ghost_temperature(gid.x, 0u);
}

@compute @workgroup_size(64)
fn apply_temperature_boundaries(@builtin(global_invocation_id) gid: vec3<u32>) {
    ghost_temperature(gid.x, 1u);
}

fn k_of(i: u32) -> f32 {
    if (is_solid(i)) {
        return params.k_solid;
    }
    if (params.varprops == 1u) {
        return params.k_fluid * law_factor(params.cond_law, params.cond_a, params.cond_b, temp_get(i));
    }
    return params.k_fluid;
}

// conductivité de face : moyenne harmonique
fn k_face(a: u32, b: u32) -> f32 {
    let ka = k_of(a);
    let kb = k_of(b);
    let s = ka + kb;
    if (s <= 0.0) {
        return 0.0;
    }
    return 2.0 * ka * kb / s;
}

// Flux (masse, ρT) à travers la face « droite » (stride 1) / « basse » (stride w) de la cellule k, en cellules/s·kg/m³.
// Un seul flux par face, partagé par ses deux cellules voisines (masse conservée exactement).
// ρ_face et (ρT)_face : MUSCL van Leer (muscl = 1) ou upwind.
fn mt_flux(k: u32, stride: u32, along: u32, len: u32, uf: f32) -> vec2<f32> {
    if (uf == 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    let p = pick(k, stride, along, len, uf);
    var ruu = 0.0;
    var tuu = 0.0;
    if (p.z != NO_CELL) {
        ruu = rho_get(p.z);
        tuu = temp_get(p.z);
    }
    let has = p.z != NO_CELL;
    let rf = recon(rho_get(p.x), rho_get(p.y), ruu, has);
    // ρT reconstruit directement (et non ρ_face·T_face) : en low-Mach ρT = P_th/r est quasi uniforme, une interface
    // chaud/froid ne crée donc pas d'écart à l'EOS (voir cpu.rs, thermo_substep_conservative)
    let rtf = recon(rho_get(p.x) * temp_get(p.x), rho_get(p.y) * temp_get(p.y), ruu * tuu, has);
    return vec2<f32>(uf * rf, uf * rtf);
}

// T de la face (hors low-Mach conservatif), pour l'advection non conservative
fn temp_face(k: u32, stride: u32, along: u32, len: u32, uf: f32) -> f32 {
    let p = pick(k, stride, along, len, uf);
    var tuu = 0.0;
    if (p.z != NO_CELL) {
        tuu = temp_get(p.z);
    }
    return recon(temp_get(p.x), temp_get(p.y), tuu, p.z != NO_CELL);
}

// Hors low-Mach : dT/dt = −u·∇T + (∇·(k∇T) + q + dP_th/dt)/(ρ·cp)
//  - advection : upwind sur les seules faces entrantes (= flux conservatif − T·div u)
//  - conduction : Σ k_face·(T_j − T_i)/dx², divisée par la capacité thermique de LA cellule i
//    (énergie conservée aux interfaces fluide/solide) ; pas de conduction par les faces du bord
//  - dP_th/dt : travail de pression, fluide seulement (0 hors boîte fermée low-Mach)
// Low-Mach, cellules de gaz ordinaires : advection CONSERVATIVE de (ρ, ρT) avec le même flux de masse
//  upwind (masse conservée exactement), T = (ρT)/ρ ; ρ est écrit dans scratch[8n + i].
// Low-Mach, murs / Sink / Emitter : équation de T ci-dessus (capacité ρ·cp du ρ persistant), ρ = EOS(T).
@compute @workgroup_size(8, 8)
fn temperature_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (!interior(x, y)) {
        return;
    }
    let w = params.width;
    let h = params.height;
    let n = w * h;
    let i = x + y * w;
    let ti = temp_get(i);

    var cond = 0.0;
    if (x + 2u < w) { cond += k_face(i, i + 1u) * (temp_get(i + 1u) - ti); }
    if (x >= 2u) { cond += k_face(i, i - 1u) * (temp_get(i - 1u) - ti); }
    if (y + 2u < h) { cond += k_face(i, i + w) * (temp_get(i + w) - ti); }
    if (y >= 2u) { cond += k_face(i, i - w) * (temp_get(i - w) - ti); }

    let q = scratch[6u * n + i];
    let wall = is_solid(i);

    if (params.lowmach == 1u && !wall && cells[i] != CELL_SINK && cells[i] != CELL_EMITTER) {
        let rho_i = rho_get(i);
        let fr = mt_flux(i, 1u, x, w, velocity_x[i]);
        let fl = mt_flux(i - 1u, 1u, x - 1u, w, velocity_x[i - 1u]);
        let fd = mt_flux(i, w, y, h, velocity_y[i]);
        let fu = mt_flux(i - w, w, y - 1u, h, velocity_y[i - w]);
        let dm = fr.x - fl.x + fd.x - fu.x;
        let dq = fr.y - fl.y + fd.y - fu.y;
        // sous CFL (sorties <= 0,9/dts) rho_new >= 0,1·rho_i : positif
        let rho_new = max(rho_i - params.dts * dm, 1e-6);
        // partie advective de T : combinaison convexe des T amont (donc bornée)
        let t_adv = (rho_i * ti - params.dts * dq) / rho_new;
        // sources divisées par une capacité thermique bornée par l'EOS (ρ persistant ~0 = résidu de projection)
        let rho_eos = rho_of_t(ti);
        let t_new = t_adv + params.dts * (cond * params.inv_dx2 + q + params.dpdt)
            / (params.cp * max(rho_new, 0.3 * rho_eos));
        // garde-fou : ρ persistant dans [0,3 ; 3]·ρ_EOS (inactif en fonctionnement normal)
        let rho_eos_new = rho_of_t(t_new);
        scratch[8u * n + i] = clamp(rho_new, 0.3 * rho_eos_new, 3.0 * rho_eos_new);
        scratch[3u * n + i] = t_new;
        return;
    }

    // advection : Σ_faces u_sortant·(T_face − T_i) ; upwind : seules les faces entrantes ; MUSCL : T_face reconstruit
    var adv = 0.0;
    let fr = velocity_x[i];
    let fl = -velocity_x[i - 1u];
    let fd = velocity_y[i];
    let fu = -velocity_y[i - w];
    if (params.muscl == 0u) {
        if (fr < 0.0) { adv += fr * (temp_get(i + 1u) - ti); }
        if (fl < 0.0) { adv += fl * (temp_get(i - 1u) - ti); }
        if (fd < 0.0) { adv += fd * (temp_get(i + w) - ti); }
        if (fu < 0.0) { adv += fu * (temp_get(i - w) - ti); }
    } else {
        if (fr != 0.0) { adv += fr * (temp_face(i, 1u, x, w, velocity_x[i]) - ti); }
        if (fl != 0.0) { adv += fl * (temp_face(i - 1u, 1u, x - 1u, w, velocity_x[i - 1u]) - ti); }
        if (fd != 0.0) { adv += fd * (temp_face(i, w, y, h, velocity_y[i]) - ti); }
        if (fu != 0.0) { adv += fu * (temp_face(i - w, w, y - 1u, h, velocity_y[i - w]) - ti); }
    }

    var cap = params.rho_amb * params.cp;
    if (wall) {
        cap = params.cap_solid;
    } else if (params.lowmach == 1u) {
        cap = rho_of_t(ti) * params.cp;
    }
    cap = max(cap, 1.0);

    var dtdt = -adv + (cond * params.inv_dx2 + q) / cap;
    if (!wall) {
        dtdt += params.dpdt / cap;
    }
    let tn = ti + params.dts * dtdt;
    scratch[3u * n + i] = tn;
    if (params.lowmach == 1u) {
        scratch[8u * n + i] = rho_of_t(tn);
    }
}

@compute @workgroup_size(8, 8)
fn temperature_commit(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!interior(gid.x, gid.y)) {
        return;
    }
    let w = params.width;
    let n = w * params.height;
    let i = gid.x + gid.y * w;
    temp_set(i, scratch[3u * n + i]);
    if (params.lowmach == 1u) {
        rho_set(i, scratch[8u * n + i]);
    }
}
