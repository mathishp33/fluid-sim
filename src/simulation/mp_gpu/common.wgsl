// ===== Solveur multiphasique GPU RÉSIDENT (f32) — miroir du solveur CPU f64 (multiphase.rs, mp_*.rs) =====
// L'état reste sur le GPU d'un pas à l'autre. Conditionnement f32 :
//   ε  = Σ_k m_k (e_k(T) − e_k(T0)) = Σ m (c_e τ + ½ s τ²), τ = T − T0, T0 = 298.15 K (énergie SENSIBLE relative, sans les décalages e0)
//   prel = p − P0 (P0 = 101325 Pa)
// Un transfert de masse dm de k vers j à énergie interne absolue constante donne Δε = dm (h0_k − h0_j), h0 = e(T0) absolue (table).
// Les constantes O_* (décalages dans S), T_* (décalages dans T), NPMAX… sont générées par l'hôte et préfixées à ce fichier.

struct Params {
    nx: u32, ny: u32, np: u32, ns: u32,
    ng: u32, npc: u32, nre: u32, nlev: u32,
    flags: u32, ambient: u32, ntp: u32, parity: u32,
    bc_l: u32, bc_r: u32, bc_b: u32, bc_t: u32,
    dx: f32, gx: f32, gy: f32, p0: f32,
    t0: f32, t_amb: f32, cfl: f32, dt_max: f32,
    pb_l: f32, pb_r: f32, pb_b: f32, pb_t: f32,
    cap_safety: f32, speed_limit: f32, nuc_dT: f32, nuc_alpha: f32,
    nuclei: f32, boil_dT: f32, pc_tau: f32, pc_vol_lim: f32,
    heat_tau: f32, heat_pmax: f32, flame_res: f32, chem_vol_lim: f32,
    ptol: f32, dt_cap: f32, nsub_cond_max: u32, nsub_sp_max: u32,
    ngas: u32, has_mix: u32, rad_on: u32, mg_tail: u32,
    fmax: f32, dil_r: u32, chem_max_sub: u32, rd_mode: u32,
    rd_flags: u32, rd_tscale: f32, rd_pscale: f32, lvp_safe: u32,   // lvp_safe : contournements pour lavapipe (pilote logiciel)
    closure_relax: f32, cfl_reject: f32, closure_small_scale: f32, phase_exch: f32,   // fermeture héritée (r, échelle) ; Courant de refus ; λ (≥ 1e37 : ∞)
    cfl_flame: f32, flame_u_floor: f32, thermo_k: u32, ntd: u32,   // borne de pas des flammes ; sous-pas du bloc thermique (correcteur) ; groupes de téléporteurs
    rho_l: f32, rho_r: f32, rho_b: f32, rho_t: f32,   // masse volumique de l'ambiant aux bords ouverts (pression hydrostatique le long du bord)
    drho_l: f32, drho_r: f32, drho_b: f32, drho_t: f32,   // sa compressibilité dρ/dp (colonne d'ambiant compressible)
};
// drapeaux (Params.flags)
const F_ST: u32 = 1u;        // tension de surface
const F_VISC: u32 = 2u;      // viscosité
const F_COND: u32 = 4u;      // conduction
const F_PC: u32 = 8u;        // changement de phase
const F_CHEM: u32 = 16u;     // chimie
const F_EXTRAS: u32 = 32u;   // cellules spéciales / entrées / solides thermiques
const F_CONS: u32 = 64u;     // quantité de mouvement conservative
const F_HEAT: u32 = 128u;    // cibles de chauffage actives
const F_MISC: u32 = 256u;    // au moins un groupe miscible (≥ 2 membres)
const F_FLAME: u32 = 512u;   // au moins une réaction de flamme (TFM)
const F_MACC: u32 = 1024u;   // MacCormack (advection semi-lagrangienne)
const F_GRAN: u32 = 2048u;   // au moins une phase granulaire (granular.wgsl)
const F_IMPVISC: u32 = 4096u; // viscosité implicite au-delà de la borne explicite (friction.wgsl)
const F_FRIC: u32 = 8192u;   // frottement des grains (friction.wgsl)
const F_PERC: u32 = 16384u;  // percolation : lits de grains perméables (deux pressions), grilles (percolation.wgsl)
const F_FUS: u32 = 32768u;   // gel / fusion actifs (paires de fusion, changement de phase coché)

// paramètres par lancement (décalage dynamique, groupe 1)
struct Disp { a: u32, b: u32, c: u32, d: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> S: array<f32>;
@group(0) @binding(2) var<storage, read> T: array<f32>;
@group(0) @binding(3) var<storage, read_write> D: array<f32>;   // scalaires dynamiques (dt, réductions, drapeaux)
@group(0) @binding(4) var<storage, read_write> R: array<u32>;   // image RGBA
@group(1) @binding(0) var<uniform> Q: Disp;

const PI: f32 = 3.14159265;
const R_GAS: f32 = 8.314462618;
const P_STD: f32 = 1.0e5;
const P_FLOOR: f32 = 10.0;
const T_PW_MIN: f32 = 60.0;
const TRACE_CAP: f32 = 1e-9;      // mp_groups::TRACE_CAP    // la détente / l'advection ne refroidissent pas une cellule sous cette température (mp_thermo.rs)

fn ncell() -> u32 { return P.nx * P.ny; }
fn nfu() -> u32 { return (P.nx + 1u) * P.ny; }
fn nfv() -> u32 { return P.nx * (P.ny + 1u); }
fn cidx(i: u32, j: u32) -> u32 { return j * P.nx + i; }
fn uidx(i: u32, j: u32) -> u32 { return j * (P.nx + 1u) + i; }
fn vidx(i: u32, j: u32) -> u32 { return j * P.nx + i; }
fn solid_id(c: u32) -> i32 { return i32(S[O_solid + c]); }
fn is_solid(c: u32) -> bool { return S[O_solid + c] >= 0.0; }
fn pinned(c: u32) -> bool { return S[O_pin_kind + c] > 0.5; }
fn has(f: u32) -> bool { return (P.flags & f) != 0u; }
fn mass(k: u32, c: u32) -> f32 { return S[O_mass + k * ncell() + c]; }
fn alpha(k: u32, c: u32) -> f32 { return S[O_alpha + k * ncell() + c]; }

// ----- tables des phases (T_PH + k * PH_STRIDE) -----
fn ph(k: u32, q: u32) -> f32 { return T[T_PH + k * PH_STRIDE + q]; }
fn is_gas(k: u32) -> bool { return ph(k, 0u) < 0.5; }
fn cp_k(k: u32) -> f32 { return ph(k, 1u); }
fn ce_k(k: u32) -> f32 { return ph(k, 2u); }
fn slope_k(k: u32) -> f32 { return ph(k, 3u); }
fn rspec_k(k: u32) -> f32 { if (is_gas(k)) { return ph(k, 4u); } return 0.0; }
fn sigma_k(k: u32) -> f32 { return ph(k, 19u); }
fn molar_k(k: u32) -> f32 { return ph(k, 20u); }
fn diff_k(k: u32) -> f32 { return ph(k, 21u); }
fn h0_k(k: u32) -> f32 { return ph(k, 22u); }       // e_k(T0) absolue (J/kg)
fn e0_k(k: u32) -> f32 { return ph(k, 23u); }       // décalage e0 (J/kg)
fn s0_k(k: u32) -> f32 { return ph(k, 24u); }
fn rad_k(k: u32) -> u32 { return u32(ph(k, 25u)); }
fn group_k(k: u32) -> u32 { return u32(ph(k, 26u)); }
fn tref_k(k: u32) -> f32 { return ph(k, 18u); }

fn law_factor(id: u32, a: f32, b: f32, t_in: f32, t_ref: f32) -> f32 {
    let t = clamp(t_in, 100.0, 4000.0);
    var f = 1.0;
    if (id == 1u) { let r = t / t_ref; f = r * sqrt(r) * (t_ref + a) / (t + a); }
    else if (id == 2u) { let d = max(t - b, 20.0); let dr = max(t_ref - b, 20.0); f = exp(2.302585093 * a * (1.0 / d - 1.0 / dr)); }
    else if (id == 3u) { f = exp(a * (1.0 / t - 1.0 / t_ref)); }
    else if (id == 4u) { f = 1.0 + a * (t - t_ref); }
    return clamp(f, 1e-2, 1e2);
}
fn kell_rho(t: f32) -> f32 {
    let c = clamp(t - 273.15, -30.0, 150.0);   // (phases::KELL_C_MIN : eau surfondue)
    let num = 999.83952 + c * (16.945176 + c * (-7.9870401e-3 + c * (-46.170461e-6 + c * (105.56302e-9 - c * 280.54253e-12))));
    return num / (1.0 + 16.87985e-3 * c);
}
fn liq_rho_p0(k: u32, t: f32) -> f32 {
    if (ph(k, 6u) > 0.5) { return kell_rho(t); }
    return ph(k, 7u) / (1.0 + ph(k, 9u) * (t - ph(k, 8u)));
}
// log(1 + x) précis pour |x| petit (Tait : x = Δp/(p0 + B) ~ 1e-4)
fn log1p_(x: f32) -> f32 {
    if (abs(x) < 1e-3) { return x * (1.0 - x * (0.5 - x * (1.0 / 3.0 - 0.25 * x))); }
    return log(1.0 + x);
}
// masse volumique de la phase k à (pression relative pr, T)
fn rho_k(k: u32, pr: f32, t: f32) -> f32 {
    if (is_gas(k)) { return max(pr + P.p0, 1.0) / (ph(k, 4u) * max(t, 1.0)); }
    let pt0 = ph(k, 31u);            // p0 de l'équation de Tait (absolu)
    let b = ph(k, 4u);
    let x = max((pr + P.p0 - pt0) / (pt0 + b), -0.99);
    return liq_rho_p0(k, t) * exp(log1p_(x) / ph(k, 5u));
}
fn kappa_k(k: u32, pr: f32) -> f32 {
    if (is_gas(k)) { return 1.0 / max(pr + P.p0, 1.0); }
    return 1.0 / (ph(k, 5u) * max(pr + P.p0 + ph(k, 4u), 1.0));
}
fn beta_k(k: u32, t: f32) -> f32 {
    if (is_gas(k)) { return 1.0 / max(t, 1.0); }
    if (ph(k, 6u) > 0.5) { let h = 0.05; return -(kell_rho(t + h) - kell_rho(t - h)) / (2.0 * h) / kell_rho(t); }
    let b = ph(k, 9u);
    return b / (1.0 + b * (t - ph(k, 8u)));
}
fn mu_k(k: u32, t: f32) -> f32 { return ph(k, 10u) * law_factor(u32(ph(k, 11u)), ph(k, 12u), ph(k, 13u), t, tref_k(k)); }
fn cond_k(k: u32, t: f32) -> f32 { return ph(k, 14u) * law_factor(u32(ph(k, 15u)), ph(k, 16u), ph(k, 17u), t, tref_k(k)); }
// énergie interne relative e_k(T) − e_k(T0) (J/kg) et chaleur massique de/dT
fn erel_k(k: u32, t: f32) -> f32 { let tau = t - P.t0; return ce_k(k) * tau + 0.5 * slope_k(k) * tau * tau; }
fn cet_k(k: u32, t: f32) -> f32 { return ce_k(k) + slope_k(k) * (t - P.t0); }
// énergie interne ABSOLUE (pour les bilans chimiques) et enthalpie massique absolue
fn eabs_k(k: u32, t: f32) -> f32 { return h0_k(k) + erel_k(k, t); }
fn hmass_k(k: u32, t: f32) -> f32 { return eabs_k(k, t) + rspec_k(k) * t; }
fn diff_at(k: u32, t: f32, p: f32) -> f32 {
    let tr = max(tref_k(k), 1.0);
    if (is_gas(k)) { return diff_k(k) * pow(max(t, 50.0) / tr, 1.75) * (101325.0 / max(p, 1e3)); }
    return diff_k(k) * (max(t, 50.0) / tr) * (ph(k, 10u) / max(mu_k(k, t), 1e-12));
}

// ----- solides -----
fn sol(s: u32, q: u32) -> f32 { return T[T_SOL + s * SOL_STRIDE + q]; }
fn solid_rhocp(s: u32) -> f32 { return sol(s, 0u); }
fn solid_k(s: u32) -> f32 { return sol(s, 1u); }
fn solid_rho(s: u32) -> f32 { return sol(s, 2u); }

// ----- température d'un mélange : Σ m_k (c_e τ + ½ s τ²) = ε (trinôme stable) -----
// forme adimensionnée τ + ½ a τ² = b (a = s/C, b = ε/C) : C² d'une trace (C ~ 1e-21 J/K) sortait de la plage f32 (discriminant nul,
// τ = −C/s ≈ −790 K pour une trace de charbon)
fn tau_from(cap: f32, slope: f32, eps: f32) -> f32 {
    if (abs(slope) <= 1e-6 * abs(cap) + 1e-30 || cap < 1e-35) { return eps / max(cap, 1e-30); }
    let a = slope / cap;
    let b = eps / cap;
    let disc = 1.0 + 2.0 * a * b;
    if (disc <= 0.0) { return -1.0 / a; }
    return 2.0 * b / (1.0 + sqrt(disc));
}
fn cell_cap0(c: u32) -> vec2<f32> {
    var cap = 0.0; var sl = 0.0;
    for (var k = 0u; k < P.np; k++) { let m = mass(k, c); cap += m * ce_k(k); sl += m * slope_k(k); }
    return vec2<f32>(cap, sl);
}
fn temp_of(c: u32) -> f32 {
    let s = solid_id(c);
    if (s >= 0) { return P.t0 + S[O_eps + c] / solid_rhocp(u32(s)); }
    let cs = cell_cap0(c);
    return P.t0 + tau_from(cs.x, cs.y, S[O_eps + c]);
}
// ∂ε/∂T à la température courante
fn heat_cap(c: u32) -> f32 {
    let s = solid_id(c);
    if (s >= 0) { return solid_rhocp(u32(s)); }
    let t = S[O_temp + c];
    var h = 0.0;
    for (var k = 0u; k < P.np; k++) { h += mass(k, c) * cet_k(k, t); }
    return h;
}
fn heat_cap_at(c: u32, t: f32) -> f32 {
    let s = solid_id(c);
    if (s >= 0) { return solid_rhocp(u32(s)); }
    var h = 0.0;
    for (var k = 0u; k < P.np; k++) { h += mass(k, c) * cet_k(k, t); }
    return h;
}

// ----- masse volumique aux faces -----
fn frho_x(i: u32, j: u32) -> f32 {
    if (i == 0u) { return S[O_rho_mix + cidx(0u, j)]; }
    if (i == P.nx) { return S[O_rho_mix + cidx(P.nx - 1u, j)]; }
    return 0.5 * (S[O_rho_mix + cidx(i - 1u, j)] + S[O_rho_mix + cidx(i, j)]);
}
fn frho_y(i: u32, j: u32) -> f32 {
    if (j == 0u) { return S[O_rho_mix + cidx(i, 0u)]; }
    if (j == P.ny) { return S[O_rho_mix + cidx(i, P.ny - 1u)]; }
    return 0.5 * (S[O_rho_mix + cidx(i, j - 1u)] + S[O_rho_mix + cidx(i, j)]);
}
fn div_uv(i: u32, j: u32) -> f32 {
    return (S[O_U + uidx(i + 1u, j)] - S[O_U + uidx(i, j)] + S[O_V + vidx(i, j + 1u)] - S[O_V + vidx(i, j)]) / P.dx;
}

// ----- température PAR GROUPE (miroir de mp_groups.rs) : ε_g = Σ_k∈g m_k (e_k(T_g) − e_k(T0)), T_g ; ε = Σ_g ε_g -----
fn epsg(g: u32, c: u32) -> f32 { return S[O_epsg + g * ncell() + c]; }
fn tgrp(g: u32, c: u32) -> f32 { return S[O_tg + g * ncell() + c]; }
fn ptemp(k: u32, c: u32) -> f32 { return tgrp(group_k(k), c); }
// boucles sur TOUTES les phases (borne uniforme P.np) avec appartenance par sélection, plutôt que sur les membres du groupe (borne
// lue dans la table) : lavapipe plante, selon les données, sur des boucles imbriquées à borne chargée en mémoire
fn group_mass(g: u32, c: u32) -> f32 {
    var s = 0.0;
    for (var k = 0u; k < P.np; k++) { s += select(0.0, max(mass(k, c), 0.0), group_k(k) == g); }
    return s;
}
fn group_cap_at(g: u32, c: u32, t: f32) -> f32 {
    var s = 0.0;
    for (var k = 0u; k < P.np; k++) { s += select(0.0, max(mass(k, c), 0.0) * cet_k(k, t), group_k(k) == g); }
    return s;
}
fn group_erel_at(g: u32, c: u32, t: f32) -> f32 {
    var s = 0.0;
    for (var k = 0u; k < P.np; k++) { s += select(0.0, max(mass(k, c), 0.0) * erel_k(k, t), group_k(k) == g); }
    return s;
}
fn group_temp_from(g: u32, c: u32, e: f32) -> f32 {
    var cap = 0.0; var sl = 0.0;
    for (var k = 0u; k < P.np; k++) {
        let m = select(0.0, max(mass(k, c), 0.0), group_k(k) == g);
        cap += m * ce_k(k); sl += m * slope_k(k);
    }
    return P.t0 + tau_from(cap, sl, e);
}
fn group_alpha(g: u32, c: u32) -> f32 {
    var s = 0.0;
    for (var k = 0u; k < P.np; k++) { s += select(0.0, alpha(k, c), group_k(k) == g); }
    return min(s, 1.0);
}
fn group_kcond(g: u32, c: u32) -> f32 {
    var a = 0.0; var kk = 0.0;
    let t = tgrp(g, c);
    for (var k = 0u; k < P.np; k++) {
        let al = select(0.0, max(alpha(k, c), 0.0), group_k(k) == g);
        a += al; kk += al * cond_k(k, t);
    }
    if (a > 0.0) { return kk / a; }
    return 0.0;
}
// tous les groupes de la cellule fluide c à t ; ε, T, T_g cohérents
fn set_cell_temp(c: u32, t: f32) {
    let n = ncell();
    var e = 0.0;
    for (var g = 0u; g < P.ng; g++) { let eg = group_erel_at(g, c, t); S[O_epsg + g * n + c] = eg; S[O_tg + g * n + c] = t; e += eg; }
    S[O_eps + c] = e;
    S[O_temp + c] = t;
}
// ε = Σ ε_g, T (mélange), traces alignées sur T (écart d'énergie au groupe principal), groupes vides → énergie au principal, T_g
fn cell_sync(c: u32) {
    if (is_solid(c)) { return; }
    let n = ncell();
    var e = 0.0;
    for (var g = 0u; g < P.ng; g++) { e += epsg(g, c); }
    S[O_eps + c] = e;
    let tc = temp_of(c);
    S[O_temp + c] = tc;
    if (P.ng == 1u) { S[O_tg + c] = tc; return; }
    // sans « continue » ni branche divergente dans les boucles (lavapipe plante sur ces motifs, selon les données) : sélections
    // et écart du groupe principal cumulé dans un scalaire
    var main = 0u; var cmax = -1.0; var ctot = 0.0;
    for (var g = 0u; g < P.ng; g++) {
        let cg = group_cap_at(g, c, tc);
        ctot += cg;
        let better = cg > cmax;
        main = select(main, g, better);
        cmax = select(cmax, cg, better);
    }
    var dmain = 0.0;
    for (var g = 0u; g < P.ng; g++) {
        let eg0 = epsg(g, c);
        let empty = group_mass(g, c) <= 0.0;
        let trace = !empty && group_cap_at(g, c, tc) < TRACE_CAP * ctot;
        let tgt_e = select(eg0, group_erel_at(g, c, tc), trace);
        let eg1 = select(select(eg0, tgt_e, trace), 0.0, empty);
        let moved = select(eg0 - eg1, 0.0, g == main);
        dmain += moved;
        S[O_epsg + g * n + c] = eg0 - moved;
    }
    S[O_epsg + main * n + c] += dmain;
    for (var g = 0u; g < P.ng; g++) {
        let m = group_mass(g, c);
        let tf = group_temp_from(g, c, epsg(g, c));
        S[O_tg + g * n + c] = select(tc, tf, m > 0.0);
    }
}

// ----- remplissages (équilibre p, T) -----
fn fill_alpha(c: u32, kk: u32, t: f32) {
    // phase pure kk
    let pr = S[O_prel + c];
    let n = ncell();
    for (var k = 0u; k < P.np; k++) {
        var m = 0.0;
        if (k == kk) { m = rho_k(k, pr, t); }
        S[O_mass + k * n + c] = m;
    }
    set_cell_temp(c, t);
}
fn amb_alpha(k: u32) -> f32 { return T[T_AMB + k]; }
fn fill_ambient(c: u32, t: f32) {
    let pr = S[O_prel + c];
    let n = ncell();
    for (var k = 0u; k < P.np; k++) {
        S[O_mass + k * n + c] = amb_alpha(k) * rho_k(k, pr, t);
    }
    set_cell_temp(c, t);
}

// ----- réductions : sommes partielles par groupe de travail (64), 8 quantités par passe -----
var<workgroup> shr: array<f32, 512>;
// op : 0 = somme, 1 = max, 2 = min
fn wg_reduce(v_in: array<f32, 8>, nq: u32, op: u32, lid: u32, wg: u32, base: u32) {
    var v = v_in;
    for (var q = 0u; q < nq; q++) { shr[q * 64u + lid] = v[q]; }
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            for (var q = 0u; q < nq; q++) {
                let a = shr[q * 64u + lid]; let b = shr[q * 64u + lid + s];
                var r = a + b;
                if (op == 1u) { r = max(a, b); } else if (op == 2u) { r = min(a, b); }
                shr[q * 64u + lid] = r;
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) { for (var q = 0u; q < nq; q++) { S[O_part + (base + q) * NWG + wg] = shr[q * 64u]; } }
}
// réduit les sommes partielles de la quantité q (n_wg groupes) — à appeler depuis UN groupe de 256 fils
var<workgroup> shf: array<f32, 256>;
fn final_reduce(q: u32, n_wg: u32, op: u32, lid: u32) -> f32 {
    var acc = 0.0;
    if (op == 1u) { acc = -3.0e38; } else if (op == 2u) { acc = 3.0e38; }
    var w = lid;
    loop {
        if (w >= n_wg) { break; }
        let v = S[O_part + q * NWG + w];
        if (op == 0u) { acc += v; } else if (op == 1u) { acc = max(acc, v); } else { acc = min(acc, v); }
        w += 256u;
    }
    shf[lid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            let a = shf[lid]; let b = shf[lid + s];
            if (op == 0u) { shf[lid] = a + b; } else if (op == 1u) { shf[lid] = max(a, b); } else { shf[lid] = min(a, b); }
        }
        workgroupBarrier();
    }
    let r = shf[0];
    workgroupBarrier();
    return r;
}
// 1 − e^(−x) sans annulation pour x petit
fn one_minus_exp(x: f32) -> f32 {
    if (x < 1e-3) { return x * (1.0 - x * (0.5 - x / 6.0)); }
    return 1.0 - exp(-x);
}
fn nwg_cells() -> u32 { return (ncell() + 63u) / 64u; }
fn nwg_faces() -> u32 { return (max(nfu(), nfv()) + 63u) / 64u; }

// ----- emplacements des scalaires dynamiques D[...] -----
const D_DT: u32 = 0u;          // pas de temps du pas en cours
const D_DTCHEM: u32 = 1u;      // limite chimique (pas suivant)
const D_DTPHASE: u32 = 2u;     // limite du changement de phase (pas suivant)
const D_NSUB_C: u32 = 3u;      // sous-pas de conduction nécessaires
const D_SUB_C: u32 = 4u;       // compteur de sous-pas de conduction
const D_NSUB_S: u32 = 5u;      // sous-pas de diffusion des espèces nécessaires
const D_SUB_S: u32 = 6u;       // compteur de sous-pas de diffusion
const D_NEWTON_DONE: u32 = 7u; // Newton convergé (1)
const D_NEWTON_IT: u32 = 8u;   // itérations de Newton effectuées
const D_PCG_DONE: u32 = 9u;    // gradient conjugué convergé (1)
const D_PCG_IT: u32 = 10u;     // itérations PCG cumulées sur le pas
const D_PCG_ITN: u32 = 11u;    // itérations PCG de la résolution courante
const D_RZ: u32 = 12u;
const D_PAP: u32 = 13u;
const D_ALPHA: u32 = 14u;
const D_BETA: u32 = 15u;
const D_RR: u32 = 16u;
const D_BB: u32 = 17u;
const D_OMAX: u32 = 18u;       // dégagement de chaleur max (senseur de flamme)
const D_BAD: u32 = 19u;        // état invalide (> 0)
const D_UMAX: u32 = 20u;
const D_PMIN: u32 = 21u;
const D_PMAX: u32 = 22u;
const D_TMIN: u32 = 23u;
const D_TMAX: u32 = 24u;
const D_DTCHEM_NEW: u32 = 25u;
const D_DTPHASE_NEW: u32 = 26u;
const D_FMAX: u32 = 27u;       // facteur d'épaississement max (TFM)
const D_MAXPCG_N: u32 = 28u;   // max d'itérations PCG d'une résolution sur le pas
const D_NEWTON_MAX: u32 = 29u; // itérations de Newton sur le pas
const D_REDO: u32 = 30u;       // budget de sous-pas insuffisant : le pas doit être refait (l'hôte augmente le budget)
const D_NSUB_S_MAX: u32 = 31u; // sous-pas de diffusion des espèces nécessaires (max sur les groupes)
const D_SCALE: u32 = 32u;      // 32..47 : échelles du rendu
const D_PCG_UNCONV: u32 = 48u; // un gradient conjugué s'est arrêté sur le budget d'itérations
const D_WMAX: u32 = 49u;       // senseur de flamme : dégagement de chaleur max
// série de pas soumise d'un bloc (step_batch) : cumuls et maxima sur la série (remis à zéro par l'hôte au début)
const D_B_TSUM: u32 = 50u;
const D_B_NSTEPS: u32 = 51u;
const D_B_BAD: u32 = 52u;
const D_B_REDO: u32 = 53u;
const D_B_SUBC: u32 = 54u;
const D_B_NSUBS: u32 = 55u;
const D_B_MAXPCG: u32 = 56u;
const D_B_NEWTON: u32 = 57u;
const D_B_UNCONV: u32 = 58u;
const D_DTF: u32 = 59u;   // fraction de dt des noyaux (sous-pas du bloc thermique des flammes ; 1 ailleurs)
// frottement des grains (friction.wgsl) : r·z (deux emplacements alternés), ‖r0‖, convergé, itérations, budget insuffisant
const D_FR_RZ: u32 = 60u;
const D_FR_R0: u32 = 62u;
const D_FR_DONE: u32 = 63u;
const D_FR_IT: u32 = 64u;
const D_FR_UNCONV: u32 = 65u;
const D_DTBURN: u32 = 66u;   // borne du pas des poudres (k_burn_dt ; 0 ou 3e38 : aucune)
const D_NEWTON_DP: u32 = 67u; // |δp| max de l'itération de Newton (rapporté au bruit d'arrondi, k_newton_dp)
const D_SK_MOVED: u32 = 68u;  // la pression de contact des lits perméables a bougé (k_sk_final) : encore une itération
const D_LEN: u32 = 72u;
