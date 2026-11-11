// Projection sur grille MAC (vitesses aux faces).
//
//   velocity_x[i] : face droite de la cellule i (entre i et i+1)
//   velocity_y[i] : face basse  de la cellule i (entre i et i+width)
//
// flags[i] : bit 0 = face droite ouverte (correction de pression possible)
//            bit 1 = face basse  ouverte
//            bit 2 = cellule solide
//
// Les cellules fantômes (bordure) ont une pression fixée à 0 : jamais écrite,
// seules les bordures "Outlet" ont des faces ouvertes vers elles (Dirichlet).

struct Params {
    width: u32,
    height: u32,
    omega: f32,
    pad: u32,
};

@group(0) @binding(0) var<storage, read_write> velocity_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> velocity_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(3) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(4) var<storage, read> flags: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

fn open_u(i: u32) -> f32 {
    return f32(flags[i] & 1u);
}

fn open_v(i: u32) -> f32 {
    return f32((flags[i] >> 1u) & 1u);
}

fn is_solid(i: u32) -> bool {
    return (flags[i] & 4u) != 0u;
}

fn interior(x: u32, y: u32) -> bool {
    return x >= 1u && y >= 1u && x < params.width - 1u && y < params.height - 1u;
}

@compute @workgroup_size(8, 8)
fn calculate_divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!interior(gid.x, gid.y)) {
        return;
    }
    let w = params.width;
    let i = gid.x + gid.y * w;

    if (is_solid(i)) {
        divergence[i] = 0.0;
        return;
    }

    divergence[i] = velocity_x[i] - velocity_x[i - 1u] + velocity_y[i] - velocity_y[i - w];
}

// Un balayage Red-Black Gauss-Seidel + SOR, en place.
// Les cellules d'une même couleur ne se lisent jamais entre elles : pas de course.
fn rb_sweep(gid: vec3<u32>, color: u32) {
    if (!interior(gid.x, gid.y) || ((gid.x + gid.y) & 1u) != color) {
        return;
    }
    let w = params.width;
    let i = gid.x + gid.y * w;

    if (is_solid(i)) {
        return;
    }

    let a = open_u(i);
    let b = open_u(i - 1u);
    let c = open_v(i);
    let d = open_v(i - w);
    let n = a + b + c + d;
    if (n == 0.0) {
        return;
    }

    let s = a * pressure[i + 1u] + b * pressure[i - 1u] + c * pressure[i + w] + d * pressure[i - w];
    let p_new = (s - divergence[i]) / n;
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

// Correction des seules faces ouvertes : les faces fermées (murs, solides, Inlet)
// gardent exactement la vitesse imposée.
@compute @workgroup_size(8, 8)
fn correct_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.width;
    let h = params.height;
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let i = gid.x + gid.y * w;

    if (gid.y >= 1u && gid.y < h - 1u && gid.x <= w - 2u && open_u(i) != 0.0) {
        velocity_x[i] = velocity_x[i] - (pressure[i + 1u] - pressure[i]);
    }
    if (gid.x >= 1u && gid.x < w - 1u && gid.y <= h - 2u && open_v(i) != 0.0) {
        velocity_y[i] = velocity_y[i] - (pressure[i + w] - pressure[i]);
    }
}
