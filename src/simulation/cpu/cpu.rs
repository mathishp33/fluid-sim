use rayon::prelude::*;
use crate::simulation::fluid::{Boundary, BoundaryType, CellEffect, CellType, FluidSim, GridBoundary};

const CFL: f32 = 0.9;
// CFL des sous-pas de scalaires en ordre 2 : le MUSCL à limiteur van Leer (facteur de pente <= 2) est TVD
// sous CFL <= 1/2 (somme des flux sortants). En ordre 1 (upwind) on garde CFL.
const CFL_SCALAR: f32 = 0.5;
const MAX_SUBSTEPS: usize = 64;
const MAX_VISC_SUBSTEPS: usize = 16;

// Relaxation de ρ vers l'EOS (fraction de l'écart résorbée par pas, via la divergence cible S).
// 0 = désactivé. 0,1–0,5 : les écarts locaux ρ ≠ P_th/(r·T) sont résorbés en quelques pas.
const EOS_RELAX: f32 = 0.2;

#[derive(Clone, Copy)]
struct SharedMut(*mut f32);
unsafe impl Send for SharedMut {}
unsafe impl Sync for SharedMut {}
impl SharedMut {
    #[inline]
    fn ptr(self) -> *mut f32 {
        self.0
    }
}

// ---- Reconstruction MUSCL (limiteur van Leer) aux faces, pour l'advection d'ordre 2 des scalaires.
//
// Pour une face entre la cellule amont U et la cellule aval D, avec UU l'amont de U :
//   a = φ_U − φ_UU,  b = φ_D − φ_U,   φ_face = φ_U + ½·(a·|b| + |a|·b) / (|a| + |b|)
// (= forme harmonique de van Leer : 0 aux extrema, φ_face reste entre φ_U et φ_D, donc pas de
// sur/sous-oscillation). Sans UU (bord, mur, ordre 1 demandé) : upwind, φ_face = φ_U.
#[inline]
fn recon(fu: f32, fd: f32, fuu: Option<f32>) -> f32 {
    match fuu {
        Some(fuu) => {
            let (a, b) = (fu - fuu, fd - fu);
            let den = a.abs() + b.abs();
            if den > 0.0 { fu + 0.5 * (a * b.abs() + a.abs() * b) / den } else { fu }
        }
        None => fu,
    }
}

/// Stencil d'une face. `k` = cellule « avant » de la face (la face est entre k et k+stride), `along` = sa
/// coordonnée le long de l'axe (x pour les faces verticales), `len` la taille de l'axe, `uf` le flux de la face.
/// Renvoie (amont U, aval D, amont-amont UU) : UU est omis au bord, contre un mur, ou si `hi` est faux.
#[inline]
fn pick(k: usize, stride: usize, along: usize, len: usize, uf: f32, hi: bool, s: &[CellType]) -> (usize, usize, Option<usize>) {
    if uf > 0.0 {
        (k, k + stride, if hi && along >= 1 && s[k - stride] != CellType::Wall { Some(k - stride) } else { None })
    } else {
        (k + stride, k, if hi && along + 2 < len && s[k + 2 * stride] != CellType::Wall { Some(k + 2 * stride) } else { None })
    }
}

// Flux du TRACEUR (advection + diffusion) aux faces de la grille MAC. Unités de grille.
struct TracerKernel<'a> {
    w: usize,
    h: usize,
    u: &'a [f32],
    v: &'a [f32],
    open_u: &'a [u8],
    open_v: &'a [u8],
    s: &'a [CellType],
    hi: bool, // advection d'ordre 2 (MUSCL) ; sinon upwind
    d: f32, // diffusion du tracer en cellules²/s
}

impl TracerKernel<'_> {
    #[inline]
    fn fx(&self, tr: &[f32], x: usize, y: usize) -> f32 {
        let i = x + y * self.w;
        let uf = self.u[i];
        let adv = if uf == 0.0 {
            0.0
        } else {
            let (a, b, c) = pick(i, 1, x, self.w, uf, self.hi, self.s);
            uf * recon(tr[a], tr[b], c.map(|k| tr[k]))
        };
        let dif = if x >= 1 && x + 2 < self.w && self.open_u[i] != 0 {
            -self.d * (tr[i + 1] - tr[i])
        } else {
            0.0
        };
        adv + dif
    }

    #[inline]
    fn fy(&self, tr: &[f32], x: usize, y: usize) -> f32 {
        let i = x + y * self.w;
        let vf = self.v[i];
        let adv = if vf == 0.0 {
            0.0
        } else {
            let (a, b, c) = pick(i, self.w, y, self.h, vf, self.hi, self.s);
            vf * recon(tr[a], tr[b], c.map(|k| tr[k]))
        };
        let dif = if y >= 1 && y + 2 < self.h && self.open_v[i] != 0 {
            -self.d * (tr[i + self.w] - tr[i])
        } else {
            0.0
        };
        adv + dif
    }
}

impl FluidSim {
    // ------------------------------------------------------------------ viscosité

    pub(crate) fn viscosity_substeps(&self, dt: f32) -> usize {
        ((4.0 * self.nu_grid() * dt / CFL).ceil() as usize).clamp(1, MAX_VISC_SUBSTEPS)
    }

    // Laplacien de la vitesse sur les faces. Contribution d'un voisin (face de même orientation) :
    //  - face ouverte : v_voisin − c
    //  - fermée, glissement libre (no_slip = false) : 0 (gradient nul)
    //  - fermée, no-slip : direction normale ou voisin = cellule fantôme du bord -> valeur stockée − c
    //    (0 pour une face fermée par un solide, valeur de CL au bord) ; direction tangentielle contre
    //    un solide -> −2c (paroi à mi-cellule, adhérence)
    pub fn diffuse_velocity(&mut self, dt: f32) {
        if self.nu_grid() <= 0.0 {
            return;
        }
        let (w, h) = (self.width, self.height);
        let n = self.viscosity_substeps(dt);
        let k = self.nu_grid() * dt / n as f32;
        let ns = self.no_slip;
        let term = move |open: bool, val: f32, c: f32, normal: bool, ghost: bool| -> f32 {
            if open { val - c } else if !ns { 0.0 } else if normal || ghost { val - c } else { -2.0 * c }
        };

        for _ in 0..n {
            self.velocity_x_temp.copy_from_slice(&self.velocity_x);
            self.velocity_y_temp.copy_from_slice(&self.velocity_y);

            {
                let (u, ou) = (&self.velocity_x, &self.open_u);
                self.velocity_x_temp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                    if y == 0 || y >= h - 1 {
                        return;
                    }
                    for x in 1..w - 2 {
                        let i = x + y * w;
                        if ou[i] == 0 {
                            continue;
                        }
                        let c = u[i];
                        let lap = term(ou[i + 1] != 0, u[i + 1], c, true, false)
                            + term(ou[i - 1] != 0, u[i - 1], c, true, false)
                            + term(ou[i + w] != 0, u[i + w], c, false, y + 1 == h - 1)
                            + term(ou[i - w] != 0, u[i - w], c, false, y == 1);
                        row[x] = c + k * lap;
                    }
                });
            }
            {
                let (v, ov) = (&self.velocity_y, &self.open_v);
                self.velocity_y_temp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                    if y == 0 || y >= h - 2 {
                        return;
                    }
                    for x in 1..w - 1 {
                        let i = x + y * w;
                        if ov[i] == 0 {
                            continue;
                        }
                        let c = v[i];
                        let lap = term(ov[i + 1] != 0, v[i + 1], c, false, x + 1 == w - 1)
                            + term(ov[i - 1] != 0, v[i - 1], c, false, x == 1)
                            + term(ov[i + w] != 0, v[i + w], c, true, false)
                            + term(ov[i - w] != 0, v[i - w], c, true, false);
                        row[x] = c + k * lap;
                    }
                });
            }

            std::mem::swap(&mut self.velocity_x, &mut self.velocity_x_temp);
            std::mem::swap(&mut self.velocity_y, &mut self.velocity_y_temp);
        }
    }

    // ------------------------------------------------------------------ chauffage visqueux
    //
    // L'opérateur ν∇² de `diffuse_velocity` retire de l'énergie cinétique ; elle doit réapparaître en chaleur
    // (sinon l'énergie totale n'est pas conservée et un fluide brassé ne s'échauffe jamais).
    // Énergie cinétique par face E = ½u² ; dE/dt = −ν·Σ_liens (Δu)² (chaque lien compté une fois), plus,
    // contre un solide en no-slip : 2ν·u² (tangentiel, paroi à mi-cellule) ou ν·u·(u − u_voisin) (normal/fantôme).
    // C'est exactement le terme que `diffuse_velocity` applique, donc le bilan discret est fermé
    // (au pas près : on évalue sur les vitesses du début de pas). Unités : u en cellules/s, ν physique en m²/s,
    // le dx² du laplacien et de l'énergie se simplifient : D[W/kg] = ν·Σ(Δu_grille)².
    // La puissance (W/m³ = ρ·D) est ajoutée à `heat_source`, moitié dans chacune des deux cellules qui
    // encadrent la face ; les cellules solides ne reçoivent rien.
    pub(crate) fn add_viscous_heating(&mut self) {
        let nu = self.material.viscosity;
        if nu <= 0.0 {
            return;
        }
        let (w, h) = (self.width, self.height);
        let ns = self.no_slip;
        let low = self.low_mach_active();
        let m = self.material;
        let p_th = self.p_th;
        let rho_amb = m.rho_ambient();
        let mut d = vec![0.0f32; w * h]; // W/kg par cellule
        let (u, v, ou, ov) = (&self.velocity_x, &self.velocity_y, &self.open_u, &self.open_v);

        // face u[i] : entre les cellules i et i+1 ; face v[i] : entre i et i+w
        let mut put_u = |i: usize, e: f32| { d[i] += 0.5 * e; d[i + 1] += 0.5 * e; };
        for y in 1..h - 1 {
            for x in 1..w - 2 {
                let i = x + y * w;
                if ou[i] == 0 { continue; }
                let c = u[i];
                // normal (x) : voisin droit (un lien ouvert n'est compté que depuis sa face de gauche)
                if ou[i + 1] != 0 { let dv = u[i + 1] - c; put_u(i, nu * dv * dv); }
                else if ns { put_u(i, (nu * c * (c - u[i + 1])).max(0.0)); }
                if ou[i - 1] == 0 && ns { put_u(i, (nu * c * (c - u[i - 1])).max(0.0)); }
                // tangentiel (y)
                if ou[i + w] != 0 { let dv = u[i + w] - c; put_u(i, nu * dv * dv); }
                else if ns { put_u(i, if y + 1 == h - 1 { (nu * c * (c - u[i + w])).max(0.0) } else { 2.0 * nu * c * c }); }
                if ou[i - w] == 0 && ns { put_u(i, if y == 1 { (nu * c * (c - u[i - w])).max(0.0) } else { 2.0 * nu * c * c }); }
            }
        }
        let mut put_v = |d: &mut Vec<f32>, i: usize, e: f32| { d[i] += 0.5 * e; d[i + w] += 0.5 * e; };
        for y in 1..h - 2 {
            for x in 1..w - 1 {
                let i = x + y * w;
                if ov[i] == 0 { continue; }
                let c = v[i];
                if ov[i + w] != 0 { let dv = v[i + w] - c; put_v(&mut d, i, nu * dv * dv); }
                else if ns { put_v(&mut d, i, (nu * c * (c - v[i + w])).max(0.0)); }
                if ov[i - w] == 0 && ns { put_v(&mut d, i, (nu * c * (c - v[i - w])).max(0.0)); }
                if ov[i + 1] != 0 { let dv = v[i + 1] - c; put_v(&mut d, i, nu * dv * dv); }
                else if ns { put_v(&mut d, i, if x + 1 == w - 1 { (nu * c * (c - v[i + 1])).max(0.0) } else { 2.0 * nu * c * c }); }
                if ov[i - 1] == 0 && ns { put_v(&mut d, i, if x == 1 { (nu * c * (c - v[i - 1])).max(0.0) } else { 2.0 * nu * c * c }); }
            }
        }
        for i in 0..w * h {
            if d[i] > 0.0 && self.solids[i] != CellType::Wall {
                let rho = if low { m.density(self.temperature[i], p_th) } else { rho_amb };
                self.heat_source[i] += rho * d[i];
            }
        }
    }

    // ------------------------------------------------------------------ flottabilité

    // y pointe vers le bas : dv/dt = -(accélération vers le haut).
    //  - low-Mach (gaz)         : a_haut = g·(ρ_ref/ρ − 1), avec ρ = EOS(T, P_th)
    //  - Boussinesq (autres)    : a_haut = g·(ρ_ref − ρ)/ρ_ref via buoyancy_factor(T)
    fn apply_buoyancy(&mut self, dt: f32) {
        let g = self.gravity_grid();
        if g == 0.0 {
            return;
        }
        let (w, h) = (self.width, self.height);
        let m = self.material;
        let low = self.low_mach_active();
        let a = g * dt;
        let p_th = self.p_th;
        let (t, ov) = (&self.temperature, &self.open_v);
        // ρ de l'EOS, PAS le ρ persistant : un résidu de projection ne doit pas pouvoir raidir la flottabilité
        let factor = |i: usize| if low { m.buoyancy_from_rho(m.density(t[i], p_th)) } else { m.buoyancy_factor(t[i]) };
        self.velocity_y.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if y == 0 || y >= h - 2 { return; }
            for x in 1..w - 1 {
                let i = x + y * w;
                if ov[i] != 0 {
                    row[x] -= a * 0.5 * (factor(i) + factor(i + w));
                }
            }
        });
    }

    // ------------------------------------------------------------------ température

    // fantômes : Inlet = valeur, Wall = adiabatique,
    // Outlet = ambiante (t_ref) avant l'advection, copie de l'intérieur après
    fn update_ghost_temperature(&mut self, before_advection: bool) {
        let (w, h) = (self.width, self.height);
        let t_ref = self.material.t_ref;
        let ghost = |b: &Boundary, inner: f32| match b.boundary_type {
            BoundaryType::Inlet => b.temperature,
            BoundaryType::Outlet if before_advection => t_ref,
            _ => inner,
        };
        for y in 1..h - 1 {
            let (l, r) = (y * w, (w - 1) + y * w);
            self.temperature[l] = ghost(&self.left_boundary, self.temperature[l + 1]);
            self.temperature[r] = ghost(&self.right_boundary, self.temperature[r - 1]);
        }
        for x in 1..w - 1 {
            let (t, b) = (x, x + (h - 1) * w);
            self.temperature[t] = ghost(&self.top_boundary, self.temperature[t + w]);
            self.temperature[b] = ghost(&self.bottom_boundary, self.temperature[b - w]);
        }
    }

    // ρ des fantômes = EOS(T fantôme, P_th) (low-Mach). Sert aux flux de masse des faces de bord.
    fn update_ghost_rho(&mut self) {
        let (w, h) = (self.width, self.height);
        let m = self.material;
        let p = self.p_th;
        for y in 1..h - 1 {
            for i in [y * w, (w - 1) + y * w] {
                self.rho[i] = m.density(self.temperature[i], p);
            }
        }
        for x in 1..w - 1 {
            for i in [x, x + (h - 1) * w] {
                self.rho[i] = m.density(self.temperature[i], p);
            }
        }
    }

    // Équation d'énergie (forme non conservative, valable avec div u ≠ 0) :
    //   dT/dt = −u·∇T + (∇·(k∇T) + q + dP_th/dt) / (ρ·cp)
    // - advection : upwind sur les seules faces entrantes (= flux conservatif − T·div u)
    // - conduction : Σ k_face·(T_j − T_i)/dx², k_face = moyenne harmonique ; chaque cellule
    //   divise par SA capacité thermique (fluide ρ·cp, solide ρ_s·cp_s) -> l'énergie est conservée
    //   aux interfaces fluide/solide ; pas de conduction à travers les faces du bord (adiabatique)
    // - q : source de chaleur (W/m³), dP_th/dt : travail de pression (fluide seulement)
    //
    // Utilisé hors low-Mach (ρ constant ou dérivé de T, la masse n'y est pas un invariant).
    fn temperature_substep(&mut self, dts: f32, hi: bool) {
        let (w, h) = (self.width, self.height);
        let m = self.material;
        let inv_dx2 = 1.0 / (self.dx * self.dx);
        let low = self.low_mach_active();
        let dpdt = self.dpdt;
        let (kf, ks, cp) = (m.conductivity, m.solid_conductivity, m.cp);
        let (cap_ref, cap_solid) = (m.heat_capacity_ref(), m.heat_capacity_solid());

        self.temperature_temp.copy_from_slice(&self.temperature); // conserve les fantômes

        let (t, u, v, s, rho, q) = (&self.temperature, &self.velocity_x, &self.velocity_y,
                                    &self.solids, &self.rho, &self.heat_source);
        let k_of = |i: usize| if s[i] == CellType::Wall { ks } else { kf };
        let k_face = |a: usize, b: usize| {
            let (ka, kb) = (k_of(a), k_of(b));
            let sum = ka + kb;
            if sum <= 0.0 { 0.0 } else { 2.0 * ka * kb / sum }
        };

        self.temperature_temp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if y == 0 || y == h - 1 {
                return;
            }
            for x in 1..w - 1 {
                let i = x + y * w;
                let ti = t[i];

                // advection : Σ_faces u_sortant·(T_face − T_i)   (= flux conservatif − T·div u)
                //  - ordre 1 : upwind, seules les faces entrantes comptent (T_face = T_i côté sortant)
                //  - ordre 2 : T_face reconstruit (MUSCL van Leer) sur toutes les faces ouvertes
                let mut adv = 0.0;
                let (fr, fl, fd, fu) = (u[i], -u[i - 1], v[i], -v[i - w]); // flux sortants
                if !hi {
                    if fr < 0.0 { adv += fr * (t[i + 1] - ti); }
                    if fl < 0.0 { adv += fl * (t[i - 1] - ti); }
                    if fd < 0.0 { adv += fd * (t[i + w] - ti); }
                    if fu < 0.0 { adv += fu * (t[i - w] - ti); }
                } else {
                    let tf = |k: usize, stride: usize, along: usize, len: usize, uf: f32| -> f32 {
                        let (a, b, c) = pick(k, stride, along, len, uf, true, s);
                        recon(t[a], t[b], c.map(|j| t[j]))
                    };
                    if fr != 0.0 { adv += fr * (tf(i, 1, x, w, u[i]) - ti); }
                    if fl != 0.0 { adv += fl * (tf(i - 1, 1, x - 1, w, u[i - 1]) - ti); }
                    if fd != 0.0 { adv += fd * (tf(i, w, y, h, v[i]) - ti); }
                    if fu != 0.0 { adv += fu * (tf(i - w, w, y - 1, h, v[i - w]) - ti); }
                }

                // conduction (faces intérieures seulement)
                let mut cond = 0.0;
                if x + 2 < w { cond += k_face(i, i + 1) * (t[i + 1] - ti); }
                if x >= 2 { cond += k_face(i, i - 1) * (t[i - 1] - ti); }
                if y + 2 < h { cond += k_face(i, i + w) * (t[i + w] - ti); }
                if y >= 2 { cond += k_face(i, i - w) * (t[i - w] - ti); }

                let wall = s[i] == CellType::Wall;
                let cap = if wall { cap_solid } else if low { rho[i] * cp } else { cap_ref }.max(1.0);
                let mut dtdt = -adv + (cond * inv_dx2 + q[i]) / cap;
                if !wall {
                    dtdt += dpdt / cap;
                }
                row[x] = ti + dts * dtdt;
            }
        });
        std::mem::swap(&mut self.temperature, &mut self.temperature_temp);
    }

    // Sous-pas CONSERVATIF du gaz low-Mach : on advecte (ρ, ρT) avec le MÊME flux de masse upwind.
    //   ∂ρ/∂t    = −∇·(ρu)
    //   ∂(ρT)/∂t = −∇·(ρuT) + (∇·(k∇T) + q + dP_th/dt)/cp        (= ρ·DT/Dt, d'après la continuité)
    //   T = (ρT)/ρ
    // - la masse est conservée exactement : les flux télescopent, et sont nuls sur les faces fermées
    //   (murs, bords Wall) ; seuls Inlet/Outlet/Sink/Emitter en créent ou en retirent ;
    // - T_new est une moyenne pondérée (poids ≥ 0 sous CFL) de T_i et des T amont : pas de sur/sous-oscillation ;
    // - murs : équation de T inchangée (capacité ρ_s·cp_s), ρ = EOS(T), sans masse ;
    // - Sink / Emitter : cellules à état imposé (elles créent/détruisent de la masse par nature) :
    //   T par l'ancienne équation, ρ = EOS(T).
    fn thermo_substep_conservative(&mut self, dts: f32, hi: bool) {
        let (w, h) = (self.width, self.height);
        let m = self.material;
        let inv_dx2 = 1.0 / (self.dx * self.dx);
        let dpdt = self.dpdt;
        let p_th = self.p_th;
        let r_spec = m.r_spec;
        let (kf, ks, cp) = (m.conductivity, m.solid_conductivity, m.cp);
        let cap_solid = m.heat_capacity_solid();

        self.temperature_temp.copy_from_slice(&self.temperature); // conserve les fantômes
        self.rho_temp.copy_from_slice(&self.rho);

        {
            let (t, u, v, s, rho, q) = (&self.temperature, &self.velocity_x, &self.velocity_y,
                                        &self.solids, &self.rho, &self.heat_source);
            let k_of = |i: usize| if s[i] == CellType::Wall { ks } else { kf };
            let k_face = |a: usize, b: usize| {
                let (ka, kb) = (k_of(a), k_of(b));
                let sum = ka + kb;
                if sum <= 0.0 { 0.0 } else { 2.0 * ka * kb / sum }
            };

            // Flux (masse, ρT) à travers la face droite (x) / basse (y) de la cellule k, en cellules/s·kg/m³.
            // Un SEUL flux par face, partagé par les deux cellules voisines (les flux télescopent : la masse
            // est conservée exactement). Le flux de ρT utilise le même flux de masse ρ_face·u que la masse.
            // ρ_face et T_face sont reconstruits (MUSCL van Leer si `hi`, sinon upwind).
            let flux = |k: usize, stride: usize, along: usize, len: usize, uf: f32| -> (f32, f32) {
                if uf == 0.0 {
                    return (0.0, 0.0);
                }
                let (a, b, c) = pick(k, stride, along, len, uf, hi, s);
                let rf = recon(rho[a], rho[b], c.map(|j| rho[j]));
                let tf = recon(t[a], t[b], c.map(|j| t[j]));
                (uf * rf, uf * rf * tf)
            };
            let fx = |k: usize| flux(k, 1, k % w, w, u[k]);
            let fy = |k: usize| flux(k, w, k / w, h, v[k]);

            self.temperature_temp.par_chunks_mut(w)
                .zip(self.rho_temp.par_chunks_mut(w))
                .enumerate()
                .for_each(|(y, (trow, rrow))| {
                    if y == 0 || y == h - 1 {
                        return;
                    }
                    for x in 1..w - 1 {
                        let i = x + y * w;
                        let ti = t[i];

                        // conduction (faces intérieures seulement)
                        let mut cond = 0.0;
                        if x + 2 < w { cond += k_face(i, i + 1) * (t[i + 1] - ti); }
                        if x >= 2 { cond += k_face(i, i - 1) * (t[i - 1] - ti); }
                        if y + 2 < h { cond += k_face(i, i + w) * (t[i + w] - ti); }
                        if y >= 2 { cond += k_face(i, i - w) * (t[i - w] - ti); }

                        let kind = s[i];
                        if kind != CellType::Wall && kind != CellType::Sink && kind != CellType::Emitter {
                            let rho_i = rho[i];
                            let (mr, er) = fx(i);
                            let (ml, el) = fx(i - 1);
                            let (md, ed) = fy(i);
                            let (mu, eu) = fy(i - w);
                            let dm = mr - ml + md - mu;
                            let dq = er - el + ed - eu;
                            // Sous CFL (sorties <= 0,9/dts) rho_new >= 0,1·rho_i : positif.
                            let rho_new = (rho_i - dts * dm).max(1e-6);
                            // partie advective de T : combinaison convexe des T amont (donc bornée)
                            let t_adv = (rho_i * ti - dts * dq) / rho_new;
                            // sources : divisées par une capacité thermique bornée par l'EOS, sinon un ρ
                            // persistant proche de 0 (résidu de projection) amplifierait q et dP_th/dt
                            let rho_eos = p_th / (r_spec * ti.max(1.0));
                            let t_new = t_adv
                                + dts * (cond * inv_dx2 + q[i] + dpdt) / (cp * rho_new.max(0.3 * rho_eos));
                            // Garde-fou : ρ persistant dans [0,3 ; 3]·ρ_EOS. Inactif en fonctionnement normal
                            // (écart ~1e-3) ; ne mord que si le solveur de pression diverge (pinceau violent
                            // avec peu d'itérations). La masse n'est alors plus exacte, mais rien n'explose.
                            let rho_eos_new = p_th / (r_spec * t_new.max(1.0));
                            rrow[x] = rho_new.clamp(0.3 * rho_eos_new, 3.0 * rho_eos_new);
                            trow[x] = t_new;
                        } else {
                            // mur, Sink, Emitter : ancienne équation de T (advection upwind non conservative)
                            let mut adv = 0.0;
                            let (fr, fl, fd, fu) = (u[i], -u[i - 1], v[i], -v[i - w]);
                            if fr < 0.0 { adv += fr * (t[i + 1] - ti); }
                            if fl < 0.0 { adv += fl * (t[i - 1] - ti); }
                            if fd < 0.0 { adv += fd * (t[i + w] - ti); }
                            if fu < 0.0 { adv += fu * (t[i - w] - ti); }

                            let wall = kind == CellType::Wall;
                            let cap = if wall { cap_solid } else { p_th / (r_spec * ti.max(1.0)) * cp }.max(1.0);
                            let mut dtdt = -adv + (cond * inv_dx2 + q[i]) / cap;
                            if !wall {
                                dtdt += dpdt / cap;
                            }
                            let tn = ti + dts * dtdt;
                            trow[x] = tn;
                            rrow[x] = p_th / (r_spec * tn.max(1.0));
                        }
                    }
                });
        }
        std::mem::swap(&mut self.temperature, &mut self.temperature_temp);
        std::mem::swap(&mut self.rho, &mut self.rho_temp);
    }

    // diffusivité thermique max en cellules²/s (pour la stabilité des sous-pas)
    fn thermal_alpha_max_grid(&self) -> f32 {
        let m = self.material;
        let rho_f = if self.low_mach_active() { self.rho_min } else { m.rho_ambient() };
        let a_f = m.conductivity / (rho_f.max(1e-6) * m.cp);
        let a_s = m.alpha_solid();
        a_f.max(a_s) / (self.dx * self.dx)
    }

    // ------------------------------------------------------------------ masques / conductances

    #[inline]
    fn offset_factor(x: usize, size: usize, offset: usize) -> f32 {
        if offset == 0 {
            return 1.0;
        }

        if size <= 2 || offset >= size {
            return 0.0;
        }

        let distance_from_edge = x.min(size - 1 - x);

        (distance_from_edge as f32 / offset as f32).clamp(0.0, 1.0)
    }

    fn rebuild_masks(&mut self) {
        let (w, h) = (self.width, self.height);
        let s = &self.solids;
        let lo = self.left_boundary.boundary_type == BoundaryType::Outlet;
        let ro = self.right_boundary.boundary_type == BoundaryType::Outlet;
        let to = self.top_boundary.boundary_type == BoundaryType::Outlet;
        let bo = self.bottom_boundary.boundary_type == BoundaryType::Outlet;

        self.open_u.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            row.fill(0);
            if y == 0 || y >= h - 1 {
                return;
            }
            for x in 0..w - 1 {
                let i = x + y * w;
                let open = if x == 0 {
                    lo && s[i + 1] != CellType::Wall
                } else if x == w - 2 {
                    ro && s[i] != CellType::Wall
                } else {
                    s[i] != CellType::Wall && s[i + 1] != CellType::Wall
                };
                row[x] = open as u8;
            }
        });

        self.open_v.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            row.fill(0);
            if y >= h - 1 {
                return;
            }
            for x in 1..w - 1 {
                let i = x + y * w;
                let open = if y == 0 {
                    to && s[i + w] != CellType::Wall
                } else if y == h - 2 {
                    bo && s[i] != CellType::Wall
                } else {
                    s[i] != CellType::Wall && s[i + w] != CellType::Wall
                };
                row[x] = open as u8;
            }
        });
    }

    // Conductance des faces pour le solveur de pression : open · (ρ_ref/ρ_face).
    // ρ_face = moyenne arithmétique des deux cellules. Incompressible / Boussinesq : 1 si ouverte.
    // C'est le « /ρ » de l'équation de quantité de mouvement (projection à densité variable).
    fn rebuild_conductances(&mut self) {
        let w = self.width;
        let varrho = self.low_mach_active();
        let rho_ref = self.material.rho_ambient();
        let (m, p_th) = (self.material, self.p_th);
        let (t, ou, ov) = (&self.temperature, &self.open_u, &self.open_v);
        // ρ de l'EOS (T, P_th), PAS le ρ persistant : le poids de la projection reste borné par T,
        // même si le solveur de pression laisse un résidu et fait dériver le ρ persistant.
        let beta = |a: usize, b: usize| 2.0 * rho_ref / (m.density(t[a], p_th) + m.density(t[b], p_th)).max(1e-9);

        self.cond_u.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let i = x + y * w;
                row[x] = if ou[i] == 0 { 0.0 } else if varrho { beta(i, i + 1) } else { 1.0 };
            }
        });
        self.cond_v.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let i = x + y * w;
                row[x] = if ov[i] == 0 { 0.0 } else if varrho { beta(i, i + w) } else { 1.0 };
            }
        });
    }

    // ------------------------------------------------------------------ advection de la vitesse

    fn sample_face(field: &[f32], solids: &[CellType], w: usize, h: usize, horiz: bool, px: f32, py: f32, fallback: f32) -> f32 {
        let (gx, gy, xm, ym) = if horiz {
            (px - 0.5, py, w - 2, h - 1)
        } else {
            (px, py - 0.5, w - 1, h - 2)
        };
        let gx = gx.clamp(0.0, xm as f32);
        let gy = gy.clamp(0.0, ym as f32);

        let x0 = gx.floor() as usize;
        let y0 = gy.floor() as usize;
        let x1 = (x0 + 1).min(xm);
        let y1 = (y0 + 1).min(ym);

        let sx = gx - x0 as f32;
        let sy = gy - y0 as f32;

        let stride = if horiz { 1 } else { w };
        let taps = [
            (x0, y0, (1.0 - sx) * (1.0 - sy)),
            (x1, y0, sx * (1.0 - sy)),
            (x0, y1, (1.0 - sx) * sy),
            (x1, y1, sx * sy),
        ];

        let mut sum = 0.0;
        let mut weight = 0.0;
        for (x, y, wt) in taps {
            let i = x + y * w;
            if solids[i] == CellType::Wall && solids[i + stride] == CellType::Wall {
                continue;
            }
            sum += field[i] * wt;
            weight += wt;
        }

        if weight > 1e-6 { sum / weight } else { fallback }
    }

    /// Comme `sample_face`, mais renvoie aussi le min et le max des taps valides (limiteur du MacCormack).
    fn sample_face_mm(field: &[f32], solids: &[CellType], w: usize, h: usize, horiz: bool, px: f32, py: f32, fallback: f32) -> (f32, f32, f32) {
        let (gx, gy, xm, ym) = if horiz {
            (px - 0.5, py, w - 2, h - 1)
        } else {
            (px, py - 0.5, w - 1, h - 2)
        };
        let gx = gx.clamp(0.0, xm as f32);
        let gy = gy.clamp(0.0, ym as f32);

        let x0 = gx.floor() as usize;
        let y0 = gy.floor() as usize;
        let x1 = (x0 + 1).min(xm);
        let y1 = (y0 + 1).min(ym);

        let sx = gx - x0 as f32;
        let sy = gy - y0 as f32;

        let stride = if horiz { 1 } else { w };
        let taps = [
            (x0, y0, (1.0 - sx) * (1.0 - sy)),
            (x1, y0, sx * (1.0 - sy)),
            (x0, y1, (1.0 - sx) * sy),
            (x1, y1, sx * sy),
        ];

        let (mut sum, mut weight) = (0.0, 0.0);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for (x, y, wt) in taps {
            let i = x + y * w;
            if solids[i] == CellType::Wall && solids[i + stride] == CellType::Wall {
                continue;
            }
            sum += field[i] * wt;
            weight += wt;
            lo = lo.min(field[i]);
            hi = hi.max(field[i]);
        }

        if weight > 1e-6 { (sum / weight, lo, hi) } else { (fallback, fallback, fallback) }
    }

    /// Advection de la vitesse : MacCormack limité (par défaut) ou semi-lagrangienne simple.
    pub fn advect_velocity(&mut self, dt: f32) {
        if self.maccormack {
            self.advect_velocity_maccormack(dt);
        } else {
            self.semi_lagrangian_pass(dt);
            std::mem::swap(&mut self.velocity_x, &mut self.velocity_x_temp);
            std::mem::swap(&mut self.velocity_y, &mut self.velocity_y_temp);
        }
    }

    // MacCormack (Selle et al. 2008) sur les faces de la grille MAC :
    //   q̂  = SL(q, −v·dt)            prédiction semi-lagrangienne (velocity_*_temp)
    //   q̃  = SL(q̂, +v·dt)            retour en arrière
    //   q' = q̂ + ½·(q − q̃)           estimation de l'erreur et correction (ordre 2)
    // puis limiteur : q' est ramené dans [min, max] des taps bilinéaires de la prédiction, ce qui
    // garantit l'absence de nouvel extremum (stable, sans oscillation de Gibbs).
    // Les faces fermées (murs, bords) ne sont pas touchées : elles sont fixées par les CL.
    fn advect_velocity_maccormack(&mut self, dt: f32) {
        let (w, h) = (self.width, self.height);

        self.semi_lagrangian_pass(dt); // q̂ dans velocity_*_temp
        self.velocity_x_mc.copy_from_slice(&self.velocity_x);
        self.velocity_y_mc.copy_from_slice(&self.velocity_y);

        {
            let (u, v, s, ou) = (&self.velocity_x, &self.velocity_y, &self.solids, &self.open_u);
            let q_hat = &self.velocity_x_temp;
            self.velocity_x_mc.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                if y == 0 || y >= h - 1 {
                    return;
                }
                for x in 1..w - 2 {
                    let i = x + y * w;
                    if ou[i] == 0 {
                        continue;
                    }
                    let (px, py) = (x as f32 + 0.5, y as f32);
                    let vx = u[i];
                    let vy = Self::sample_face(v, s, w, h, false, px, py, 0.0);
                    let (_, lo, hi) = Self::sample_face_mm(u, s, w, h, true, px - vx * dt, py - vy * dt, vx);
                    let back = Self::sample_face(q_hat, s, w, h, true, px + vx * dt, py + vy * dt, q_hat[i]);
                    let mc = q_hat[i] + 0.5 * (u[i] - back);
                    row[x] = mc.max(lo).min(hi); // pas de clamp() : il panique si un borne est NaN
                }
            });
        }

        {
            let (u, v, s, ov) = (&self.velocity_x, &self.velocity_y, &self.solids, &self.open_v);
            let q_hat = &self.velocity_y_temp;
            self.velocity_y_mc.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                if y == 0 || y >= h - 2 {
                    return;
                }
                for x in 1..w - 1 {
                    let i = x + y * w;
                    if ov[i] == 0 {
                        continue;
                    }
                    let (px, py) = (x as f32, y as f32 + 0.5);
                    let vy = v[i];
                    let vx = Self::sample_face(u, s, w, h, true, px, py, 0.0);
                    let (_, lo, hi) = Self::sample_face_mm(v, s, w, h, false, px - vx * dt, py - vy * dt, vy);
                    let back = Self::sample_face(q_hat, s, w, h, false, px + vx * dt, py + vy * dt, q_hat[i]);
                    let mc = q_hat[i] + 0.5 * (v[i] - back);
                    row[x] = mc.max(lo).min(hi);
                }
            });
        }

        std::mem::swap(&mut self.velocity_x, &mut self.velocity_x_mc);
        std::mem::swap(&mut self.velocity_y, &mut self.velocity_y_mc);
    }

    // Passe semi-lagrangienne : écrit q̂ dans velocity_x_temp / velocity_y_temp (copie des originaux
    // pour les faces non advectées), sans échanger les buffers.
    fn semi_lagrangian_pass(&mut self, dt: f32) {
        let (w, h) = (self.width, self.height);

        self.velocity_x_temp.copy_from_slice(&self.velocity_x);
        self.velocity_y_temp.copy_from_slice(&self.velocity_y);

        {
            let (u, v, s, ou) = (&self.velocity_x, &self.velocity_y, &self.solids, &self.open_u);
            self.velocity_x_temp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                if y == 0 || y >= h - 1 {
                    return;
                }
                for x in 1..w - 2 {
                    let i = x + y * w;
                    if ou[i] == 0 {
                        continue; // face fermée : fixée par les conditions aux limites
                    }
                    let (px, py) = (x as f32 + 0.5, y as f32);
                    let vx = u[i];
                    let vy = Self::sample_face(v, s, w, h, false, px, py, 0.0);
                    row[x] = Self::sample_face(u, s, w, h, true, px - vx * dt, py - vy * dt, vx);
                }
            });
        }

        {
            let (u, v, s, ov) = (&self.velocity_x, &self.velocity_y, &self.solids, &self.open_v);
            self.velocity_y_temp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                if y == 0 || y >= h - 2 {
                    return;
                }
                for x in 1..w - 1 {
                    let i = x + y * w;
                    if ov[i] == 0 {
                        continue;
                    }
                    let (px, py) = (x as f32, y as f32 + 0.5);
                    let vy = v[i];
                    let vx = Self::sample_face(u, s, w, h, true, px, py, 0.0);
                    row[x] = Self::sample_face(v, s, w, h, false, px - vx * dt, py - vy * dt, vy);
                }
            });
        }
    }

    // ------------------------------------------------------------------ conditions aux limites

    // CL des faces à appliquer avant la projection
    // (composante normale, composante tangentielle fantôme). no-slip : fantôme = 2·U_paroi − intérieur.
    fn edge_values(b: &GridBoundary, no_slip: bool, factor: f32, inlet_n: f32, inlet_t: f32, inner_n: f32, inner_t: f32) -> (f32, f32) {
        match b.boundary_type {
            BoundaryType::Wall => (0.0, if no_slip { -inner_t } else { inner_t }),
            BoundaryType::Outlet => (inner_n, inner_t), // grad = 0
            BoundaryType::Inlet => (inlet_n * factor, if no_slip { 2.0 * inlet_t * factor - inner_t } else { inlet_t * factor }),
        }
    }

    fn apply_velocity_boundaries(&mut self) {
        let (w, h) = (self.width, self.height);
        let ns = self.no_slip;
        let dx = self.dx;

        // faces intérieures fermées par un solide -> 0
        for y in 1..h - 1 {
            for x in 1..w - 2 {
                let i = x + y * w;
                if self.open_u[i] == 0 {
                    self.velocity_x[i] = 0.0;
                }
            }
        }
        for y in 1..h - 2 {
            for x in 1..w - 1 {
                let i = x + y * w;
                if self.open_v[i] == 0 {
                    self.velocity_y[i] = 0.0;
                }
            }
        }

        // gauche / droite : normale = velocity_x (face de bord), tangentielle = velocity_y (fantôme)
        for y in 1..h - 1 {
            let row = y * w;

            let b = self.left_boundary.to_grid(dx);
            let f = Self::offset_factor(y, h, b.offset);
            let (n, t) = Self::edge_values(&b, ns, f, b.vx, b.vy,
                                           self.velocity_x[row + 1], self.velocity_y[row + 1]);
            self.velocity_x[row] = if self.solids[row + 1] == CellType::Wall { 0.0 } else { n };
            self.velocity_y[row] = t;

            let b = self.right_boundary.to_grid(dx);
            let f = Self::offset_factor(y, h, b.offset);
            let (n, t) = Self::edge_values(&b, ns, f, b.vx, b.vy,
                                           self.velocity_x[row + w - 3], self.velocity_y[row + w - 2]);
            self.velocity_x[row + w - 2] = if self.solids[row + w - 2] == CellType::Wall { 0.0 } else { n };
            self.velocity_y[row + w - 1] = t;
        }

        // haut / bas : normale = velocity_y (face de bord), tangentielle = velocity_x (fantôme)
        for x in 1..w - 1 {
            let last = (h - 2) * w;

            let b = self.top_boundary.to_grid(dx);
            let f = Self::offset_factor(x, w, b.offset);
            let (n, t) = Self::edge_values(&b, ns, f, b.vy, b.vx,
                                           self.velocity_y[x + w], self.velocity_x[x + w]);
            self.velocity_y[x] = if self.solids[x + w] == CellType::Wall { 0.0 } else { n };
            self.velocity_x[x] = t;

            let b = self.bottom_boundary.to_grid(dx);
            let f = Self::offset_factor(x, w, b.offset);
            let (n, t) = Self::edge_values(&b, ns, f, b.vy, b.vx,
                                           self.velocity_y[x + last - w], self.velocity_x[x + last]);
            self.velocity_y[x + last] = if self.solids[x + last] == CellType::Wall { 0.0 } else { n };
            self.velocity_x[x + last + w] = t;
        }
    }

    /// Vitesse maximale autorisée pour que le nombre de sous-pas de scalaires reste <= MAX_SUBSTEPS.
    /// Partagée avec le GPU (limit_velocity dans le shader).
    pub(crate) fn velocity_limit(&self, dt: f32) -> f32 {
        CFL * MAX_SUBSTEPS as f32 / (4.0 * dt.max(1e-6))
    }

    fn limit_velocity(&mut self, dt: f32) {
        let vmax = self.velocity_limit(dt);
        let fix = |v: &mut f32| {
            *v = if v.is_finite() { v.clamp(-vmax, vmax) } else { 0.0 };
        };
        self.velocity_x.par_iter_mut().for_each(fix);
        self.velocity_y.par_iter_mut().for_each(fix);
    }

    fn apply_special_velocity(&mut self, dt: f32) {
        let dx = self.dx;
        for &(idx, effect) in self.special_cells.iter() {
            match effect.to_grid(dx) {
                CellEffect::Emitter { vx, vy, .. } => {
                    self.velocity_x[idx] = vx;
                    self.velocity_y[idx] = vy;
                }
                CellEffect::Accelerator { ax, ay } => {
                    self.velocity_x[idx] += ax * dt;
                    self.velocity_y[idx] += ay * dt;
                }
                CellEffect::Teleporter { dest_idx } => {
                    if dest_idx != idx && self.solids[dest_idx] != CellType::Wall {
                        self.velocity_x[dest_idx] = self.velocity_x[idx];
                        self.velocity_y[dest_idx] = self.velocity_y[idx];
                    }
                }
                CellEffect::Sink { rate } => {
                    self.velocity_x[idx] = -rate;
                    self.velocity_x[idx - 1] = rate;
                    self.velocity_y[idx] = -rate;
                    self.velocity_y[idx - self.width] = rate;
                    self.pressure[idx] = 0.0;
                }
            }
        }
    }

    // ------------------------------------------------------------------ low-Mach : divergence cible

    /// Flux volumique sortant par les faces du bord (somme sur les faces de bord, cellules/s·cellule).
    /// Les faces fermées par un solide valent déjà 0 (apply_velocity_boundaries).
    fn boundary_outflow(&self) -> f64 {
        let (w, h) = (self.width, self.height);
        let (u, v) = (&self.velocity_x, &self.velocity_y);
        let mut q = 0.0f64;
        for y in 1..h - 1 {
            q += (-u[y * w] + u[(w - 2) + y * w]) as f64;
        }
        for x in 1..w - 1 {
            q += (-v[x] + v[x + (h - 2) * w]) as f64;
        }
        q
    }

    /// Même flux sortant que `boundary_outflow`, mais déduit des seules conditions aux limites
    /// (valeurs de bord après apply_velocity_boundaries + limit_velocity), donc calculable AVANT
    /// d'avoir les vitesses du pas. Valable en boîte fermée (aucun Outlet) : bord = 0 (Wall)
    /// ou facteur·vitesse (Inlet), 0 si la cellule voisine est solide.
    pub(crate) fn boundary_outflow_analytic(&self, dt: f32) -> f64 {
        let (w, h) = (self.width, self.height);
        let vmax = self.velocity_limit(dt);
        let s = &self.solids;
        let dx = self.dx;
        let face = |b: &GridBoundary, f: f32, comp: f32, solid_neighbor: bool| -> f64 {
            if solid_neighbor || b.boundary_type != BoundaryType::Inlet {
                return 0.0;
            }
            let v = comp * f;
            (if v.is_finite() { v.clamp(-vmax, vmax) } else { 0.0 }) as f64
        };
        let mut q = 0.0f64;
        for y in 1..h - 1 {
            let row = y * w;
            let (bl, br) = (self.left_boundary.to_grid(dx), self.right_boundary.to_grid(dx));
            q -= face(&bl, Self::offset_factor(y, h, bl.offset), bl.vx, s[row + 1] == CellType::Wall);
            q += face(&br, Self::offset_factor(y, h, br.offset), br.vx, s[row + w - 2] == CellType::Wall);
        }
        for x in 1..w - 1 {
            let last = (h - 2) * w;
            let (bt, bb) = (self.top_boundary.to_grid(dx), self.bottom_boundary.to_grid(dx));
            q -= face(&bt, Self::offset_factor(x, w, bt.offset), bt.vy, s[x + w] == CellType::Wall);
            q += face(&bb, Self::offset_factor(x, w, bb.offset), bb.vy, s[x + last] == CellType::Wall);
        }
        q
    }

    // Gaz low-Mach : on impose  div u = S  avec
    //   S = (∇·(k∇T) + q)/(ρ·cp·T)  −  (1/(γ·P_th))·dP_th/dt   +   S_eos
    // - (ρ·T = P_th/r par l'EOS, donc le premier terme vaut (∇·(k∇T) + q)·r/(cp·P_th) : il ne dépend
    //   pas du ρ persistant, et dP_th/dt reste purement énergétique)
    // - S_eos = EOS_RELAX/dt · (ρ − ρ_eos)/ρ, avec ρ_eos = P_th/(r·T) : si le ρ persistant dérive de l'EOS
    //   (erreurs de schéma, résidu du SOR), le gaz se dilate ou se comprime légèrement pour le résorber.
    //   En boîte fermée, on en retire la moyenne (la compatibilité de Neumann ΣS = flux de bord est
    //   respectée et P_th ne dépend que de l'énergie). Un écart UNIFORME ρ/ρ_eos n'est donc pas
    //   corrigé : il vaut la différence entre la masse suivie et P_th suivie, deux grandeurs exactes.
    // - domaine ouvert (Outlet ou Sink) : P_th = P_ref, dP_th/dt = 0
    // - boîte fermée : dP_th/dt est choisi pour que le problème de Poisson (Neumann) soit
    //   résoluble, i.e. Σ S = flux sortant par les faces de bord (Inlets).
    // Incompressible / Boussinesq : S = 0.
    //
    // `q_out` : flux de bord déjà connu (chemin GPU, où les vitesses ne sont pas encore calculées) ;
    // None = on le lit dans les tableaux de vitesse (chemin CPU).
    pub(crate) fn compute_source(&mut self, dt: f32, q_out: Option<f64>) {
        if !self.low_mach_active() {
            self.source.fill(0.0);
            self.dpdt = 0.0;
            self.p_th = self.material.p_ref;
            return;
        }
        let (w, h) = (self.width, self.height);
        let m = self.material;
        let inv_dx2 = 1.0 / (self.dx * self.dx);
        let (kf, ks, cp) = (m.conductivity, m.solid_conductivity, m.cp);
        let closed = self.is_closed();
        // pression de l'EOS : P_th en boîte fermée, P_ref en domaine ouvert
        let p_eos = if closed { self.p_th.max(1.0) } else { m.p_ref };
        let eos_k = m.r_spec / (cp * p_eos); // 1/(ρ·cp·T) = r/(cp·P)
        let r_spec = m.r_spec;
        let relax = EOS_RELAX / dt.max(1e-6);

        let (sum_a, sum_c, n_active) = {
            let (t, s, rho, q) = (&self.temperature, &self.solids, &self.rho, &self.heat_source);
            let k_of = |i: usize| if s[i] == CellType::Wall { ks } else { kf };
            let k_face = |a: usize, b: usize| {
                let (ka, kb) = (k_of(a), k_of(b));
                let sum = ka + kb;
                if sum <= 0.0 { 0.0 } else { 2.0 * ka * kb / sum }
            };

            self.source.par_chunks_mut(w).enumerate().map(|(y, row)| {
                row.fill(0.0);
                if y == 0 || y >= h - 1 {
                    return (0.0f64, 0.0f64, 0u64);
                }
                let (mut sum, mut sum_c, mut n) = (0.0f64, 0.0f64, 0u64);
                for x in 1..w - 1 {
                    let i = x + y * w;
                    if s[i] == CellType::Wall || s[i] == CellType::Sink {
                        continue;
                    }
                    let ti = t[i];
                    let mut cond = 0.0;
                    if x + 2 < w { cond += k_face(i, i + 1) * (t[i + 1] - ti); }
                    if x >= 2 { cond += k_face(i, i - 1) * (t[i - 1] - ti); }
                    if y + 2 < h { cond += k_face(i, i + w) * (t[i + w] - ti); }
                    if y >= 2 { cond += k_face(i, i - w) * (t[i - w] - ti); }

                    let a = (cond * inv_dx2 + q[i]) * eos_k;

                    let rho_eos = p_eos / (r_spec * ti.max(1.0));
                    let ratio = ((rho[i] - rho_eos) / rho[i].max(1e-6)).clamp(-0.5, 0.5);
                    let c = relax * ratio;

                    row[x] = a + c;
                    sum += a as f64;
                    sum_c += c as f64;
                    n += 1;
                }
                (sum, sum_c, n)
            }).reduce(|| (0.0, 0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
        };

        if closed && n_active > 0 {
            let q_out = match q_out {
                Some(q) => q,
                None => {
                    let q = self.boundary_outflow();
                    debug_assert!(
                        (q - self.boundary_outflow_analytic(dt)).abs() <= 1e-3 * (1.0 + q.abs()),
                        "flux de bord analytique != flux lu dans les vitesses"
                    );
                    q
                }
            };
            let delta = ((sum_a - q_out) / n_active as f64) as f32; // = dP_th/dt / (γ·P_th)
            let mean_c = (sum_c / n_active as f64) as f32;
            let gamma = m.gamma();
            self.dpdt = gamma * self.p_th * delta;
            // cumul en f64 : à ~1e5 Pa, l'ulp f32 vaut ~0,008 Pa, un chauffage lent (incrément < ½ ulp par pas)
            // ne changerait jamais p_th. Si p_th a été modifié de l'extérieur (UI), on se resynchronise dessus.
            if self.p_th != self.p_th_acc as f32 {
                self.p_th_acc = self.p_th as f64;
            }
            self.p_th_acc = (self.p_th_acc + self.dpdt as f64 * dt as f64).max(1.0);
            self.p_th = self.p_th_acc as f32;

            let shift = delta + mean_c;
            let s = &self.solids;
            self.source.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                if y == 0 || y >= h - 1 {
                    return;
                }
                for x in 1..w - 1 {
                    let i = x + y * w;
                    if s[i] != CellType::Wall && s[i] != CellType::Sink {
                        row[x] -= shift;
                    }
                }
            });
        } else {
            self.dpdt = 0.0;
            self.p_th = m.p_ref;
        }
    }

    // ------------------------------------------------------------------ projection

    fn calculate_divergence(&mut self) {
        let (w, h) = (self.width, self.height);
        let (u, v, s) = (&self.velocity_x, &self.velocity_y, &self.solids);

        self.divergence.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if y == 0 || y == h - 1 {
                return;
            }
            for x in 1..w - 1 {
                let i = x + y * w;
                row[x] = if s[i] == CellType::Wall || s[i] == CellType::Sink {
                    0.0
                } else {
                    u[i] - u[i - 1] + v[i] - v[i - w]
                };
            }
        });
    }

    // Poisson à coefficients variables : Σ_faces c_f·(p_voisin − p_i) = div_i − S_i
    // (c_f = conductance de la face, voir rebuild_conductances), Red-Black SOR.
    fn solve_pressure(&mut self, iterations: usize) {
        let (w, h) = (self.width, self.height);
        let omega = self.sor_omega;
        let shared = SharedMut(self.pressure.as_mut_ptr());
        let (div, src, cu, cv, s) = (&self.divergence, &self.source, &self.cond_u, &self.cond_v, &self.solids);

        for _ in 0..iterations {
            for color in 0..2usize {
                (1..h - 1).into_par_iter().for_each(|y| {
                    let p = shared.ptr();
                    let mut x = if ((1 + y) & 1) == color { 1 } else { 2 };
                    while x < w - 1 {
                        let i = x + y * w;
                        if s[i] != CellType::Wall && s[i] != CellType::Sink {
                            let a = cu[i];
                            let b = cu[i - 1];
                            let c = cv[i];
                            let d = cv[i - w];
                            let n = a + b + c + d;
                            if n > 0.0 {
                                unsafe {
                                    let sum = a * *p.add(i + 1) + b * *p.add(i - 1)
                                        + c * *p.add(i + w) + d * *p.add(i - w);
                                    let p_new = (sum - (div[i] - src[i])) / n;
                                    let old = *p.add(i);
                                    *p.add(i) = old + omega * (p_new - old);
                                }
                            }
                        }
                        x += 2;
                    }
                });
            }
        }
    }

    // Problème de Neumann pur (boîte fermée) : la pression n'est définie qu'à une constante près.
    // On retire la moyenne pour qu'elle ne dérive pas.
    pub(crate) fn remove_pressure_mean(&mut self) {
        let (w, h) = (self.width, self.height);
        let s = &self.solids;
        let p = &self.pressure;
        let (sum, n) = (1..h - 1).into_par_iter().map(|y| {
            let (mut sum, mut n) = (0.0f64, 0u64);
            for x in 1..w - 1 {
                let i = x + y * w;
                if s[i] != CellType::Wall && s[i] != CellType::Sink {
                    sum += p[i] as f64;
                    n += 1;
                }
            }
            (sum, n)
        }).reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
        if n == 0 {
            return;
        }
        let mean = (sum / n as f64) as f32;
        self.pressure.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if y == 0 || y >= h - 1 {
                return;
            }
            for x in 1..w - 1 {
                let i = x + y * w;
                if s[i] != CellType::Wall && s[i] != CellType::Sink {
                    row[x] -= mean;
                }
            }
        });
    }

    fn correct_velocity(&mut self) {
        let (w, h) = (self.width, self.height);
        let (p, cu, cv) = (&self.pressure, &self.cond_u, &self.cond_v);

        self.velocity_x.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if y == 0 || y >= h - 1 {
                return;
            }
            for x in 0..w - 1 {
                let i = x + y * w;
                if cu[i] > 0.0 {
                    row[x] -= cu[i] * (p[i + 1] - p[i]);
                }
            }
        });

        self.velocity_y.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if y >= h - 1 {
                return;
            }
            for x in 1..w - 1 {
                let i = x + y * w;
                if cv[i] > 0.0 {
                    row[x] -= cv[i] * (p[i + w] - p[i]);
                }
            }
        });
    }

    pub fn enforce_incompressibility_cpu(&mut self, pressure_iterations: usize) {
        self.calculate_divergence();
        self.solve_pressure(pressure_iterations);
        if self.is_closed() {
            self.remove_pressure_mean();
        }
        self.correct_velocity();
        self.calculate_divergence(); // divergence résiduelle : doit valoir S
    }

    // ------------------------------------------------------------------ tracer (colorant)

    // fantômes du tracer : Inlet = valeur, Wall = copie, Outlet = 0 avant l'advection, copie après
    fn ghost_tracer(&mut self, before_advection: bool) {
        let (w, h) = (self.width, self.height);
        let ghost = |b: &Boundary, inner: f32| match b.boundary_type {
            BoundaryType::Inlet => b.tracer,
            BoundaryType::Outlet if before_advection => 0.0,
            _ => inner,
        };
        for y in 1..h - 1 {
            let (l, r) = (y * w, (w - 1) + y * w);
            self.tracer[l] = ghost(&self.left_boundary, self.tracer[l + 1]);
            self.tracer[r] = ghost(&self.right_boundary, self.tracer[r - 1]);
        }
        for x in 1..w - 1 {
            let (t, b) = (x, x + (h - 1) * w);
            self.tracer[t] = ghost(&self.top_boundary, self.tracer[t + w]);
            self.tracer[b] = ghost(&self.bottom_boundary, self.tracer[b - w]);
        }
    }

    /// Nombre de sous-pas des scalaires (tracer + température) et ordre de l'advection.
    /// Garantit dt_sub · (flux sortant + 4·diffusion max) <= CFL_SCALAR en ordre 2 (MUSCL), CFL en ordre 1.
    /// Si l'ordre 2 est demandé mais demanderait plus de MAX_SUBSTEPS sous-pas (vitesses extrêmes), on
    /// retombe en upwind pour ce pas : plus diffusif mais sûr. Renvoie (sous-pas, ordre 2 actif).
    pub(crate) fn scalar_plan(&self, dt: f32) -> (usize, bool) {
        let (w, h) = (self.width, self.height);
        let (u, v, s) = (&self.velocity_x, &self.velocity_y, &self.solids);

        let rate = (1..h - 1)
            .into_par_iter()
            .map(|y| {
                let mut m = 0.0f32;
                for x in 1..w - 1 {
                    let i = x + y * w;
                    if s[i] == CellType::Wall {
                        continue;
                    }
                    let out = u[i].max(0.0) + (-u[i - 1]).max(0.0) + v[i].max(0.0) + (-v[i - w]).max(0.0);
                    m = m.max(out);
                }
                m
            })
            .reduce(|| 0.0, f32::max);

        let diffusion = self.thermal_alpha_max_grid().max(self.tracer_diff_grid());
        let load = dt * (rate + 4.0 * diffusion);
        if self.high_order_scalars {
            let n = (load / CFL_SCALAR).ceil() as usize;
            if n <= MAX_SUBSTEPS {
                return (n.max(1), true);
            }
        }
        (((load / CFL).ceil() as usize).clamp(1, MAX_SUBSTEPS), false)
    }

    // Tracer + température en sous-pas. Le tracer est une FRACTION (pas une densité) :
    // avec div u ≠ 0 (low-Mach) on advecte sa forme non conservative (flux − Y·div u),
    // sinon il serait comprimé/dilué par l'expansion du gaz.
    // Low-Mach : (ρ, ρT) sont advectés de façon conservative (thermo_substep_conservative).
    pub fn advect_diffuse_scalars(&mut self, dt: f32) {
        let (w, h) = (self.width, self.height);

        self.ghost_tracer(true);
        self.update_ghost_temperature(true);
        let expansion = self.low_mach_active();
        if expansion {
            self.update_ghost_rho();
        }
        let (n, hi) = self.scalar_plan(dt);
        let dts = dt / n as f32;
        let d = self.tracer_diff_grid();

        for _ in 0..n {
            self.tracer_temp.copy_from_slice(&self.tracer); // conserve les cellules fantômes

            let kernel = TracerKernel { w, h, u: &self.velocity_x, v: &self.velocity_y,
                open_u: &self.open_u, open_v: &self.open_v, s: &self.solids, hi, d };
            let (tr, s) = (&self.tracer, &self.solids);
            let (u, v) = (&self.velocity_x, &self.velocity_y);

            self.tracer_temp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                if y == 0 || y == h - 1 {
                    return;
                }
                for x in 1..w - 1 {
                    let i = x + y * w;
                    row[x] = if s[i] == CellType::Wall {
                        0.0
                    } else {
                        let mut val = tr[i] - dts * (kernel.fx(tr, x, y) - kernel.fx(tr, x - 1, y)
                            + kernel.fy(tr, x, y) - kernel.fy(tr, x, y - 1));
                        if expansion {
                            val += dts * tr[i] * (u[i] - u[i - 1] + v[i] - v[i - w]);
                        }
                        val
                    };
                }
            });

            std::mem::swap(&mut self.tracer, &mut self.tracer_temp);
            if expansion {
                self.thermo_substep_conservative(dts, hi);
            } else {
                self.temperature_substep(dts, hi);
            }
        }
        self.heat_source.fill(0.0); // la source de chaleur du pinceau est consommée
    }

    fn apply_special_tracer(&mut self, _dt: f32) {
        for &(idx, effect) in self.special_cells.iter() {
            match effect {
                CellEffect::Emitter { tracer, .. } => {
                    self.tracer[idx] = tracer;
                }
                CellEffect::Sink { .. } => {
                    self.tracer[idx] = 0.0;
                }
                CellEffect::Teleporter { dest_idx } => {
                    if dest_idx != idx && self.solids[dest_idx] != CellType::Wall {
                        self.tracer[dest_idx] = (self.tracer[dest_idx] + self.tracer[idx]).min(1.0); //todo attention au surplus
                        self.tracer[idx] = 0.0;
                    }
                }
                CellEffect::Accelerator { .. } => {}
            }
        }
    }

    // ------------------------------------------------------------------ pas complet

    pub(crate) fn step_cpu(&mut self, dt: f32, pressure_iterations: usize) {
        self.refresh_rho(); // low-Mach : ρ persistant (masse) ; sinon ρ(T, P_th)
        self.rebuild_masks(); // solides | types de bordure -> faces ouvertes
        self.rebuild_conductances(); // faces ouvertes * ρ_ref/ρ_face

        self.advect_velocity(dt); // advection semi-lagrangienne des faces
        self.diffuse_velocity(dt);
        self.apply_buoyancy(dt);
        self.apply_special_velocity(dt);

        self.apply_velocity_boundaries(); // CL + faces fermées = 0
        self.limit_velocity(dt);

        self.compute_source(dt, None); // low-Mach : S et dP_th/dt (0 sinon)
        self.enforce_incompressibility_cpu(pressure_iterations); // div u = S

        self.advect_diffuse_scalars(dt); // tracer + (ρ, ρT) / T, sous-pas CFL
        self.apply_special_tracer(dt);
        self.ghost_tracer(false); // fantômes, affichage
        self.update_ghost_temperature(false);
        if self.low_mach_active() {
            self.update_ghost_rho();
        }

        self.update_pressure_pa(dt);
    }
}