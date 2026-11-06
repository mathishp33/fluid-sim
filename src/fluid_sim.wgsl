struct Params {
    width: u32,
    height: u32,
    left_type: u32,
    right_type: u32,
    top_type: u32,
    bottom_type: u32,
};

@group(0) @binding(0) var<storage, read_write> velocity_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> velocity_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> pressure: array<f32>;
@group(0) @binding(3) var<storage, read_write> pressure_temp: array<f32>;
@group(0) @binding(4) var<storage, read_write> divergence: array<f32>;
@group(0) @binding(5) var<storage, read> solids: array<u32>;
@group(0) @binding(6) var<uniform> params: Params;

fn is_solid(idx: u32) -> bool {
    return solids[idx] != 0u;
}

@compute @workgroup_size(8, 8)
fn calculate_divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x == 0u || y == 0u || x >= params.width - 1u || y >= params.height - 1u) {
        return;
    }

    let idx = x + y * params.width;

    if (is_solid(idx)) {
        divergence[idx] = 0.0;
        return;
    }

    let vx_right = select(velocity_x[idx + 1u], 0.0, is_solid(idx + 1u));
    let vx_left = select(velocity_x[idx - 1u], 0.0, is_solid(idx - 1u));
    let vy_down = select(velocity_y[idx + params.width], 0.0, is_solid(idx + params.width));
    let vy_up = select(velocity_y[idx - params.width], 0.0, is_solid(idx - params.width));

    let div = 0.5 * (vx_right - vx_left + vy_down - vy_up);
    divergence[idx] = -div;
}

@compute @workgroup_size(8, 8)
fn solve_pressure_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x == 0u || y == 0u || x >= params.width - 1u || y >= params.height - 1u) {
        return;
    }

    let idx = x + y * params.width;

    if (is_solid(idx)) {
        pressure_temp[idx] = 0.0;
        return;
    }

    var sum = divergence[idx];
    let right = idx + 1u;
    let left = idx - 1u;
    let down = idx + params.width;
    let up = idx - params.width;

    sum += select(pressure[right], pressure[idx], is_solid(right));
    sum += select(pressure[left], pressure[idx], is_solid(left));
    sum += select(pressure[down], pressure[idx], is_solid(down));
    sum += select(pressure[up], pressure[idx], is_solid(up));

    pressure_temp[idx] = sum * 0.25;
}

@compute @workgroup_size(64)
fn apply_pressure_boundaries(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let w = params.width;
    let h = params.height;

    if (i < h - 2u) {
        let y = i + 1u;

        let l0 = 0u + y * w;
        if (is_solid(l0)) {
            pressure[l0] = 0.0;
        } else if (params.left_type == 2u) {
            pressure[l0] = 0.0;
        } else {
            pressure[l0] = pressure[1u + y * w];
        }

        let l1 = (w - 1u) + y * w;
        if (is_solid(l1)) {
            pressure[l1] = 0.0;
        } else if (params.right_type == 2u) {
            pressure[l1] = 0.0;
        } else {
            pressure[l1] = pressure[(w - 2u) + y * w];
        }
    }

    if (i < w - 2u) {
        let x = i + 1u;

        let l2 = x;
        if (is_solid(l2)) {
            pressure[l2] = 0.0;
        } else if (params.top_type == 2u) {
            pressure[l2] = 0.0;
        } else {
            pressure[l2] = pressure[x + 1u * w];
        }

        let l3 = x + (h - 1u) * w;
        if (is_solid(l3)) {
            pressure[l3] = 0.0;
        } else if (params.bottom_type == 2u) {
            pressure[l3] = 0.0;
        } else {
            pressure[l3] = pressure[x + (h - 2u) * w];
        }
    }
}

@compute @workgroup_size(8, 8)
fn correct_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x == 0u || y == 0u || x >= params.width - 1u || y >= params.height - 1u) {
        return;
    }

    let idx = x + y * params.width;

    if (is_solid(idx)) {
        velocity_x[idx] = 0.0;
        velocity_y[idx] = 0.0;
        return;
    }

    let right = idx + 1u;
    let left = idx - 1u;
    let down = idx + params.width;
    let up = idx - params.width;

    let p_right = select(pressure[right], pressure[idx], is_solid(right));
    let p_left = select(pressure[left], pressure[idx], is_solid(left));
    let p_down = select(pressure[down], pressure[idx], is_solid(down));
    let p_up = select(pressure[up], pressure[idx], is_solid(up));

    let grad_x = 0.5 * (p_right - p_left);
    let grad_y = 0.5 * (p_down - p_up);

    velocity_x[idx] -= grad_x;
    velocity_y[idx] -= grad_y;
}