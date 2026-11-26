//! Pression faiblement compressible IMPLICITE (Helmholtz).
//!
//! Fermeture volumique : à la fin du pas, Σ_k m_k/ρ_k(p, T) = 1 dans chaque cellule, avec les masses m_k déjà advectées.
//! Linéarisée (Newton, 2–4 itérations) autour de p_k :  V(p) ≈ V(p_k, T_k) − κ_eff (p − p_k),
//! κ_eff = Σ V_i κ_T,i − T (Σ V_i β_i)² / Σ m_i cp_i   (compressibilité à entropie constante du mélange : gaz → 1/(γp),
//! bulle dans l'eau → quasi isotherme).
//! Équation : −∇·(dt/ρ_f ∇δp) + (κ_eff/dt) δp = (V−1)/dt − ∇·(Ũ − uⁿ),   Ũ = u* − dt/ρ_f ∇p_k,
//! uⁿ⁺¹ = Ũ − dt/ρ_f ∇δp. Le terme acoustique κ/dt est implicite : le pas de temps n'est PAS limité par c ≈ 1480 m/s.
//! Matrice SPD résolue par gradients conjugués préconditionnés par un V-cycle multigrille (`mp_mg`, défaut) ou par Cholesky incomplet
//! IC(0) (`multigrid = false`, séquentiel, ~10× plus d'itérations).
//! Lits perméables (mp_percolation.rs) : p y est la pression du fluide des pores, le mélange est poussé par p + σ' ; la pression de
//! contact σ' (squelette des grains incompressible) est résolue après la première itération de Newton (`solve_skeleton`), les
//! itérations suivantes rétablissant le volume (blocs de Gauss–Seidel p / σ'), et une première fois avant le frottement, sur la vitesse
//! prédite (`presolve_contact_stress`).

use super::mp_bodies::{BodyCoupling, BodyLowRank};
use super::mp_par::{for_ranges, for_rows, map_ranges, MIN_HEAVY, MIN_LIGHT};
use super::multiphase::{Bc, MultiphaseSim, SyncPtr};
use super::phases::MAX_PHASES;

/// pression absolue minimale d'une cellule fluide (Pa)
pub(crate) const P_FLOOR: f64 = 10.0;
/// erreur de fermeture héritée corrigée entièrement en un pas tant qu'elle reste sous ce seuil (usage normal : < 1e-2 ; bulle de
/// vapeur qui grossit dans l'eau : jusqu'à ~0.1, qu'il faut corriger entièrement sous peine de courants parasites)
pub(crate) const CLOSURE_SMALL: f64 = 0.1;

/// Part de l'erreur de fermeture `e0` que ce pas corrige — appelée avec l'erreur qui restera après l'advection du pas,
/// f = (Σ m/ρ − 1 au début du pas) − dt·∇·u^n (voir solve_pressure). Dans le gaz (fraction `ag`) : tout,
/// par compression. Dans le liquide : tout si |e0| ≤ `small` (une erreur, même petite, laissée dans un liquide raide fausse sa pression
/// de |e0|/κ), sinon max(small, r·|e0|) — corrigée en un pas, une grosse erreur dans un liquide impose une vitesse ∝ |e0|/dt
/// (schéma neutre-stable qui s'emballe : huile très chaude au contact d'eau ou d'air froid). Même formule sur le GPU (k_rhs).
pub(crate) fn closure_correction(e0: f64, ag: f64, r: f64, small: f64) -> f64 {
    let liq = if e0.abs() <= small { e0 } else { e0.signum() * small.max(r * e0.abs()) };
    let ag = ag.clamp(0.0, 1.0);
    ag * e0 + (1.0 - ag) * liq
}

struct Stencil {
    diag: Vec<f64>,
    ae: Vec<f64>,
    an: Vec<f64>,
}

impl MultiphaseSim {
    /// Volume V et κ_eff d'une cellule à la pression p, le groupe g étant à la température t[g], avec le couplage thermique `chi`
    /// des groupes pendant le pas (`thermal_coupling`). `gam[g]` reçoit dT_g/dp et `wpw[g]` ∝ C_g·dT_g/dp, la part du groupe dans
    /// le travail de pression (même modèle, sinon la fin du pas contredit la fermeture). χ = 0 : chaque groupe se comprime
    /// adiabatiquement pour son compte (une bulle d'air dans de l'huile chaude a κ = V/(γp)) ; χ = 1 : température commune
    /// (κ_eff = Σκ_T − T(ΣVβ)²/Σm c_p, modèle à une température : une bulle de vapeur dans l'eau se comprime presque isotherme).
    pub(crate) fn cell_closure_g(&self, c: usize, p: f64, t: &[f64], chi: f64, gam: &mut [f64], wpw: &mut [f64]) -> (f64, f64) {
        let ng = self.groups.len();
        let (mut v, mut kt_all) = (0.0, 0.0);
        let (mut own, mut vb_all, mut mcp_all, mut mcpt) = (0.0, 0.0, 0.0, 0.0);
        let mut vbg = [0.0f64; super::phases::MAX_PHASES];
        let mut mcpg = [0.0f64; super::phases::MAX_PHASES];
        for (g, members) in self.groups.iter().enumerate() {
            let tg = t[g];
            let (mut vb, mut mcp) = (0.0, 0.0);
            for &k in members {
                let m = self.mass[k][c];
                if m <= 0.0 {
                    continue;
                }
                let ph = &self.reg.phases[k];
                let vk = m / ph.rho(p, tg);
                v += vk;
                kt_all += vk * ph.kappa_t(p, tg);
                vb += vk * ph.beta(p, tg);
                mcp += m * ph.cp;
            }
            vbg[g] = vb;
            mcpg[g] = mcp;
            if mcp > 0.0 {
                own += tg * vb * vb / mcp;
            }
            vb_all += vb;
            mcp_all += mcp;
            mcpt += mcp * tg;
        }
        let chi = chi.clamp(0.0, 1.0);
        let tm = if mcp_all > 0.0 { mcpt / mcp_all } else { 0.0 };
        let gm = if mcp_all > 0.0 { tm * vb_all / mcp_all } else { 0.0 };
        let mix = gm * vb_all;
        for g in 0..ng {
            let go = if mcpg[g] > 0.0 { t[g] * vbg[g] / mcpg[g] } else { 0.0 };
            gam[g] = chi * gm + (1.0 - chi) * go;
            wpw[g] = (mcpg[g] * gam[g]).max(0.0);
        }
        // corps rigides : volume incompressible ajouté à la fermeture (voir `cell_volume`)
        let v = if self.has_bodies() { v + self.body_volume(c) } else { v };
        (v, kt_all - (chi * mix + (1.0 - chi) * own))
    }

    /// Volume de la cellule à la pression p avec les températures de groupe courantes (tg).
    pub(crate) fn cell_volume(&self, c: usize, p: f64) -> f64 {
        let v: f64 = (0..self.reg.phases.len())
            .filter(|&k| self.mass[k][c] > 0.0)
            .map(|k| self.mass[k][c] / self.reg.phases[k].rho(p, self.phase_temp(k, c)))
            .sum();
        // corps rigides (mp_bodies.rs) : leur volume compte dans la fermeture, Σ m/ρ + α_corps = 1 (sans corps : inchangé)
        if self.has_bodies() { v + self.body_volume(c) } else { v }
    }

    /// Mise à jour de Newton δ d'une cellule CONTENANT DU GAZ quand |δp| > 50 % de p : la tangente V(p_k) − κ δp sous-estime
    /// énormément la pression nécessaire pour recomprimer une poche de gaz basse pression (p ne fait que doubler par itération, la
    /// cellule reste un puits qui aspire ses voisines) et passe sous zéro à la détente (plancher → déficit de volume). On impose au gaz
    /// la variation de volume que le système linéaire lui attribue, ΔV_g = (κ − κ_l) δp, avec la loi polytropique V_g ∝ p^(−1/n)
    /// tangente au même κ (n = V_g / (p (κ − κ_l)) ∈ [1, 2] : adiabatique dans un gaz pur, isotherme au contact d'un liquide) :
    /// p' = p (V_g / (V_g − ΔV_g))ⁿ, ≡ p + δp au premier ordre. Mêmes formules sur le GPU (`k_p_update`).
    pub(crate) fn gas_newton_update(&self, c: usize, pk: f64, t: &[f64], kappa: f64, dp: f64) -> f64 {
        if dp.abs() <= 0.5 * pk.max(P_FLOOR) {
            return dp;
        }
        let (mut vg, mut kl) = (0.0, 0.0);
        for (k, ph) in self.reg.phases.iter().enumerate() {
            let m = self.mass[k][c];
            if m <= 0.0 {
                continue;
            }
            let tk = t[self.group_of[k]];
            let vk = m / ph.rho(pk, tk);
            if ph.is_gas() {
                vg += vk;
            } else {
                kl += vk * ph.kappa_t(pk, tk);
            }
        }
        let kg = kappa - kl;
        if vg <= 1e-9 || kg <= 0.0 {
            return dp;
        }
        let n = (vg / (pk.max(1.0) * kg)).clamp(1.0, 2.0);
        let vgt = (vg - kg * dp).max(0.02 * vg);
        // au plus 4 fois la tangente à la compression : le reste du système (vitesses des faces voisines) est linéarisé autour de p_k
        let d = pk * ((vg / vgt).powf(n) - 1.0);
        if dp > 0.0 {
            d.min(4.0 * dp)
        } else {
            d
        }
    }

    /// vitesse de face corrigée Ũ = u* − dt/ρ_f ∇p (0 sur parois/solides). Corps rigides (mp_bodies.rs) : Ũ est la vitesse
    /// DÉBITANTE du fluide, A (u* − dt/ρ_f ∇p) avec A l'ouverture d'écoulement de la face (la part couverte appartient au corps) ; la
    /// matrice (`build_stencil`) a les mêmes coefficients A, d'où une fermeture exacte autour des corps. Corps en mouvement imposé :
    /// + A (1 − A) V_corps, le fluide de la face partiellement couverte est entraîné à la vitesse du corps (non-glissement étalé sur
    /// une maille). Sans corps : facteur 1.
    fn face_velocities(&self, us: &[f64], vs: &[f64], p: &[f64], dt: f64) -> (Vec<f64>, Vec<f64>) {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let bodies = self.has_bodies();
        let moving = self.has_moving_bodies();
        let ou = |i: usize, j: usize| if bodies { self.u_open(i, j) } else { 1.0 };
        let ov = |i: usize, j: usize| if bodies { self.v_open(i, j) } else { 1.0 };
        let bf = |a: f64, x: f64, ub: f64| if moving { a * x + a * (1.0 - a) * ub } else { a * x };
        let mut un = vec![0.0; (nx + 1) * ny];
        let mut vn = vec![0.0; nx * (ny + 1)];
        // bords ouverts : pression hydrostatique du fluide ambiant le long du bord (voir `open_face_pressure`)
        let rho_open: [f64; 4] = std::array::from_fn(|s| match self.bc[s] {
            Bc::Open { p } => self.ambient_density(p),
            Bc::Wall => 0.0,
        });
        let drho_open: [f64; 4] = std::array::from_fn(|s| match self.bc[s] {
            Bc::Open { p } => self.ambient_compressibility(p),
            Bc::Wall => 0.0,
        });
        let (lx, ly) = (nx as f64 * dx, ny as f64 * dx);
        let up = SyncPtr(un.as_mut_ptr());
        for_rows(ny, nx + 1, MIN_LIGHT, |j| {
            for i in 0..=nx {
                let f = j * (nx + 1) + i;
                let (l, r) = (if i > 0 { Some(j * nx + i - 1) } else { None }, if i < nx { Some(j * nx + i) } else { None });
                let val = match (l, r) {
                    (Some(a), Some(b)) => {
                        if self.is_solid(a) || self.is_solid(b) {
                            0.0
                        } else {
                            bf(ou(i, j), us[f] - dt / self.face_rho_x(i, j) * (p[b] - p[a]) / dx, self.u_body_vel(f))
                        }
                    }
                    (None, Some(b)) => match (self.bc[0], self.is_solid(b)) {
                        (Bc::Open { p: po }, false) => {
                            let pb = self.open_face_pressure(po, rho_open[0], drho_open[0], 0.0, (j as f64 + 0.5) * dx);
                            bf(ou(i, j), us[f] - dt / self.rho_mix[b] * (p[b] - pb) / (0.5 * dx), self.u_body_vel(f))
                        }
                        _ => 0.0,
                    },
                    (Some(a), None) => match (self.bc[1], self.is_solid(a)) {
                        (Bc::Open { p: po }, false) => {
                            let pb = self.open_face_pressure(po, rho_open[1], drho_open[1], lx, (j as f64 + 0.5) * dx);
                            bf(ou(i, j), us[f] - dt / self.rho_mix[a] * (pb - p[a]) / (0.5 * dx), self.u_body_vel(f))
                        }
                        _ => 0.0,
                    },
                    _ => 0.0,
                };
                // SAFETY : la face f n'est écrite que par sa ligne
                unsafe { *up.at(f) = val }
            }
        });
        let vp = SyncPtr(vn.as_mut_ptr());
        for_rows(ny + 1, nx, MIN_LIGHT, |j| {
            for i in 0..nx {
                let f = j * nx + i;
                let (l, r) = (if j > 0 { Some((j - 1) * nx + i) } else { None }, if j < ny { Some(j * nx + i) } else { None });
                let val = match (l, r) {
                    (Some(a), Some(b)) => {
                        if self.is_solid(a) || self.is_solid(b) {
                            0.0
                        } else {
                            bf(ov(i, j), vs[f] - dt / self.face_rho_y(i, j) * (p[b] - p[a]) / dx, self.v_body_vel(f))
                        }
                    }
                    (None, Some(b)) => match (self.bc[2], self.is_solid(b)) {
                        (Bc::Open { p: po }, false) => {
                            let pb = self.open_face_pressure(po, rho_open[2], drho_open[2], (i as f64 + 0.5) * dx, 0.0);
                            bf(ov(i, j), vs[f] - dt / self.rho_mix[b] * (p[b] - pb) / (0.5 * dx), self.v_body_vel(f))
                        }
                        _ => 0.0,
                    },
                    (Some(a), None) => match (self.bc[3], self.is_solid(a)) {
                        (Bc::Open { p: po }, false) => {
                            let pb = self.open_face_pressure(po, rho_open[3], drho_open[3], (i as f64 + 0.5) * dx, ly);
                            bf(ov(i, j), vs[f] - dt / self.rho_mix[a] * (pb - p[a]) / (0.5 * dx), self.v_body_vel(f))
                        }
                        _ => 0.0,
                    },
                    _ => 0.0,
                };
                // SAFETY : la face f n'est écrite que par sa ligne
                unsafe { *vp.at(f) = val }
            }
        });
        for (f, p) in self.pf_u.iter().enumerate() {
            if !p.is_nan() {
                un[f] = *p;
            }
        }
        for (f, p) in self.pf_v.iter().enumerate() {
            if !p.is_nan() {
                vn[f] = *p;
            }
        }
        (un, vn)
    }

    pub(crate) fn divergence_of(&self, u: &[f64], v: &[f64], i: usize, j: usize) -> f64 {
        let (nx, dx) = (self.nx, self.dx);
        (u[j * (nx + 1) + i + 1] - u[j * (nx + 1) + i] + v[(j + 1) * nx + i] - v[j * nx + i]) / dx
    }

    /// Couplage de pression entre deux cellules voisines a, b à travers une face : 0 = aucun (solide, face à vitesse imposée),
    /// 1 = couplage complet, 2 = Dirichlet (une des deux est épinglée : δp = 0 de son côté, terme de diagonale seulement)
    fn link(&self, a: usize, b: usize, fixed: bool) -> u8 {
        if fixed || self.is_solid(a) || self.is_solid(b) {
            0
        } else if self.pinned(a) || self.pinned(b) {
            2
        } else {
            1
        }
    }

    /// matrice de la correction de pression ; `darcy` : conductances de percolation (C/dx² ajouté au couplage des faces des milieux
    /// poreux, y compris des faces fermées au mélange — grilles)
    fn build_stencil(&self, kappa: &[f64], dt: f64, darcy: Option<&super::mp_percolation::DarcyFaces>) -> Stencil {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let n = nx * ny;
        let mut s = Stencil { diag: vec![1.0; n], ae: vec![0.0; n], an: vec![0.0; n] };
        let dcu = |f: usize| darcy.map_or(0.0, |d| d.cu[f] / (dx * dx));
        let dcv = |f: usize| darcy.map_or(0.0, |d| d.cv[f] / (dx * dx));
        // corps fixes : couplage des faces multiplié par leur ouverture au fluide (voir `face_velocities`) ; sans corps : 1
        let bodies = self.has_bodies();
        let ou = |i: usize, j: usize| if bodies { self.u_open(i, j) } else { 1.0 };
        let ov = |i: usize, j: usize| if bodies { self.v_open(i, j) } else { 1.0 };
        let (dgp, aep, anp) = (SyncPtr(s.diag.as_mut_ptr()), SyncPtr(s.ae.as_mut_ptr()), SyncPtr(s.an.as_mut_ptr()));
        // SAFETY (écritures ci-dessous) : la cellule c n'est écrite que par sa ligne
        for_rows(ny, nx, MIN_LIGHT, |j| {
            for i in 0..nx {
                let c = j * nx + i;
                if self.is_solid(c) || self.pinned(c) || self.body_trace(c) {
                    continue; // ligne identité (δp = 0)
                }
                let mut d = kappa[c] / dt;
                // est
                if i + 1 < nx {
                    let k = self.link(c, c + 1, self.face_u_fixed(self.uidx(i + 1, j)));
                    let td = dcu(self.uidx(i + 1, j));
                    if k > 0 {
                        let t = ou(i + 1, j) * (dt / (self.face_rho_x(i + 1, j) * dx * dx)) + td;
                        if k == 1 {
                            unsafe { *aep.at(c) = t }
                        }
                        d += t;
                    } else if td > 0.0 {
                        unsafe { *aep.at(c) = td }
                        d += td;
                    }
                } else if let Bc::Open { .. } = self.bc[1] {
                    d += ou(nx, j) * (2.0 * dt / (self.rho_mix[c] * dx * dx));
                }
                // ouest
                if i > 0 {
                    if self.link(c - 1, c, self.face_u_fixed(self.uidx(i, j))) > 0 {
                        d += ou(i, j) * (dt / (self.face_rho_x(i, j) * dx * dx));
                    }
                    d += dcu(self.uidx(i, j));
                } else if let Bc::Open { .. } = self.bc[0] {
                    d += ou(0, j) * (2.0 * dt / (self.rho_mix[c] * dx * dx));
                }
                // nord
                if j + 1 < ny {
                    let k = self.link(c, c + nx, self.face_v_fixed(self.vidx(i, j + 1)));
                    let td = dcv(self.vidx(i, j + 1));
                    if k > 0 {
                        let t = ov(i, j + 1) * (dt / (self.face_rho_y(i, j + 1) * dx * dx)) + td;
                        if k == 1 {
                            unsafe { *anp.at(c) = t }
                        }
                        d += t;
                    } else if td > 0.0 {
                        unsafe { *anp.at(c) = td }
                        d += td;
                    }
                } else if let Bc::Open { .. } = self.bc[3] {
                    d += ov(i, ny) * (2.0 * dt / (self.rho_mix[c] * dx * dx));
                }
                // sud
                if j > 0 {
                    if self.link(c - nx, c, self.face_v_fixed(self.vidx(i, j))) > 0 {
                        d += ov(i, j) * (dt / (self.face_rho_y(i, j) * dx * dx));
                    }
                    d += dcv(self.vidx(i, j));
                } else if let Bc::Open { .. } = self.bc[2] {
                    d += ov(i, 0) * (2.0 * dt / (self.rho_mix[c] * dx * dx));
                }
                unsafe { *dgp.at(c) = d }
            }
        });
        s
    }

    /// Résout (A + L) x = b, L : terme de rang faible des corps libres (`BodyLowRank`, absent sans eux). Retourne le nombre
    /// d'itérations.
    fn pcg(&self, st: &Stencil, b: &[f64], x: &mut [f64], extra: Option<&BodyLowRank>) -> usize {
        if self.multigrid {
            self.pcg_mg(st, b, x, None, extra)
        } else {
            self.pcg_ic0(st, b, x, extra)
        }
    }

    /// Gradient conjugué préconditionné par multigrille (vecteurs et opérateur parallèles). `mask` : lignes actives (les autres sont
    /// des lignes identité) ; défaut : cellules ni solides ni épinglées. `extra` : terme des corps libres ajouté à l'opérateur (le
    /// préconditionneur l'ignore : de rang 3 par corps, il coûte au plus quelques itérations).
    fn pcg_mg(&self, st: &Stencil, b: &[f64], x: &mut [f64], mask: Option<&[bool]>, extra: Option<&BodyLowRank>) -> usize {
        use super::mp_mg::{Level, Multigrid};
        use super::mp_par::{collect, dot, for_ranges, MIN_LIGHT};
        use super::multiphase::SyncPtr;
        let n = b.len();
        let active: Vec<bool> = match mask {
            Some(m) => m.to_vec(),
            None => collect(n, MIN_LIGHT, |c| !(self.is_solid(c) || self.pinned(c))),
        };
        let fine = Level { nx: self.nx, ny: self.ny, diag: st.diag.clone(), ae: st.ae.clone(), an: st.an.clone(), active };
        let mut mg = Multigrid::new(fine);
        let mut r = vec![0.0; n];
        let mut ax = vec![0.0; n];
        mg.levels[0].apply(x, &mut ax);
        if let Some(l) = extra {
            l.apply_add(x, &mut ax);
        }
        super::mp_par::fill(&mut r, MIN_LIGHT, |c| b[c] - ax[c]);
        let bn = dot(b, b).sqrt().max(1e-300);
        let mut z = vec![0.0; n];
        mg.precondition(&r, &mut z);
        let mut pdir = z.clone();
        let mut rz = dot(&r, &z);
        let mut it = 0;
        while it < self.pressure_max_iter {
            if dot(&r, &r).sqrt() <= self.pressure_tol * bn {
                break;
            }
            mg.levels[0].apply(&pdir, &mut ax);
            if let Some(l) = extra {
                l.apply_add(&pdir, &mut ax);
            }
            let pap = dot(&pdir, &ax);
            if pap.abs() < 1e-300 {
                break;
            }
            let al = rz / pap;
            {
                let (xp, rp) = (SyncPtr(x.as_mut_ptr()), SyncPtr(r.as_mut_ptr()));
                let (pd, axr) = (&pdir, &ax);
                for_ranges(n, MIN_LIGHT, |lo, hi| {
                    for c in lo..hi {
                        // SAFETY : chaque c dans une seule tranche
                        unsafe {
                            *xp.at(c) += al * pd[c];
                            *rp.at(c) -= al * axr[c];
                        }
                    }
                });
            }
            mg.precondition(&r, &mut z);
            let rz2 = dot(&r, &z);
            let be = rz2 / rz.max(1e-300);
            rz = rz2;
            {
                let pp = SyncPtr(pdir.as_mut_ptr());
                let zr = &z;
                for_ranges(n, MIN_LIGHT, |lo, hi| {
                    for c in lo..hi {
                        // SAFETY : chaque c dans une seule tranche
                        unsafe { *pp.at(c) = zr[c] + be * *pp.at(c) }
                    }
                });
            }
            it += 1;
        }
        it
    }

    /// Gradient conjugué préconditionné par Cholesky incomplet IC(0) (historique, séquentiel).
    fn pcg_ic0(&self, st: &Stencil, b: &[f64], x: &mut [f64], extra: Option<&BodyLowRank>) -> usize {
        let nx = self.nx;
        let n = b.len();
        let mut l = vec![1.0; n];
        let (mut lw, mut ls) = (vec![0.0; n], vec![0.0; n]);
        for c in 0..n {
            let mut d = st.diag[c];
            if c % nx > 0 {
                lw[c] = -st.ae[c - 1] / l[c - 1];
                d -= lw[c] * lw[c];
            }
            if c >= nx {
                ls[c] = -st.an[c - nx] / l[c - nx];
                d -= ls[c] * ls[c];
            }
            l[c] = d.max(1e-300).sqrt();
        }
        let precond = |r: &[f64], z: &mut [f64], y: &mut [f64]| {
            for c in 0..n {
                let mut s = r[c];
                if lw[c] != 0.0 {
                    s -= lw[c] * y[c - 1];
                }
                if ls[c] != 0.0 {
                    s -= ls[c] * y[c - nx];
                }
                y[c] = s / l[c];
            }
            for c in (0..n).rev() {
                let mut s = y[c];
                if c + 1 < n && lw[c + 1] != 0.0 {
                    s -= lw[c + 1] * z[c + 1];
                }
                if c + nx < n && ls[c + nx] != 0.0 {
                    s -= ls[c + nx] * z[c + nx];
                }
                z[c] = s / l[c];
            }
        };
        let apply = |x: &[f64], out: &mut [f64]| {
            for c in 0..n {
                let mut s = st.diag[c] * x[c];
                if st.ae[c] != 0.0 {
                    s -= st.ae[c] * x[c + 1];
                }
                if c % nx > 0 && st.ae[c - 1] != 0.0 {
                    s -= st.ae[c - 1] * x[c - 1];
                }
                if st.an[c] != 0.0 {
                    s -= st.an[c] * x[c + nx];
                }
                if c >= nx && st.an[c - nx] != 0.0 {
                    s -= st.an[c - nx] * x[c - nx];
                }
                out[c] = s;
            }
            if let Some(l) = extra {
                l.apply_add(x, out);
            }
        };
        let mut r = vec![0.0; n];
        let mut ax = vec![0.0; n];
        apply(x, &mut ax);
        for c in 0..n {
            r[c] = b[c] - ax[c];
        }
        let bn = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-300);
        let (mut z, mut y) = (vec![0.0; n], vec![0.0; n]);
        precond(&r, &mut z, &mut y);
        let mut pdir = z.clone();
        let mut rz: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let mut it = 0;
        while it < self.pressure_max_iter {
            let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if rn <= self.pressure_tol * bn {
                break;
            }
            apply(&pdir, &mut ax);
            let pap: f64 = pdir.iter().zip(&ax).map(|(a, b)| a * b).sum();
            if pap.abs() < 1e-300 {
                break;
            }
            let al = rz / pap;
            for c in 0..n {
                x[c] += al * pdir[c];
                r[c] -= al * ax[c];
            }
            precond(&r, &mut z, &mut y);
            let rz2: f64 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
            let be = rz2 / rz.max(1e-300);
            rz = rz2;
            for c in 0..n {
                pdir[c] = z[c] + be * pdir[c];
            }
            it += 1;
        }
        it
    }

    /// Cellules qui entrent dans un lit perméable ou en sortent depuis la projection précédente (lit posé sur une grille, grains qui se
    /// déposent, fluide des pores qui change…) : la pression de la cellule est partagée au mieux, la pression totale du mélange restant
    /// la même. Entrée : σ' = poids déjaugé des grains au-dessus (colonne lithostatique), retiré de p (sans cela le poids passait aux
    /// contacts au fil des pas, le lit tassé se posant entre-temps sur les parois : 75 % du poids porté par les contacts après 25 pas de
    /// 10 ms dans un lit de grains de 0.3 mm) ; sortie : σ' rendue à p.
    fn skeleton_transitions(&mut self, skel: &[bool]) {
        let n = self.nx * self.ny;
        if self.sigma.len() != n || self.skel.len() != n {
            self.sigma = vec![0.0; n];
            self.skel = vec![false; n];
        }
        let entering = (0..n).any(|c| skel[c] && !self.skel[c]);
        let ps = if entering { self.grain_overburden() } else { vec![] };
        for c in 0..n {
            if skel[c] && !self.skel[c] {
                self.sigma[c] = ps[c].max(0.0).min((self.p[c] - 2.0 * P_FLOOR).max(0.0));
                self.p[c] -= self.sigma[c];
            } else if !skel[c] && self.skel[c] {
                self.p[c] += self.sigma[c];
                self.sigma[c] = 0.0;
            } else if !skel[c] {
                self.sigma[c] = 0.0;
            }
        }
        self.skel.copy_from_slice(skel);
    }

    /// Pression de contact du pas courant, AVANT le frottement : squelette des lits perméables résolu sur la vitesse prédite (us, vs),
    /// avec les cellules à deux pressions de la projection précédente. Le frottement (`apply_implicit_viscosity`) voit ainsi le σ' qui
    /// équilibre les charges du moment : avec celui du pas précédent, sa viscosité retirait l'écart de σ' avant que la projection ne le
    /// corrige, et le lit restait figé dans son état passé — un lit de sable dans l'eau portait encore 40 % de contact de trop 2 s après
    /// la mise en route du courant, un lit de braises sous une flamme accumulait σ'. Renvoie les itérations du gradient conjugué.
    pub(crate) fn presolve_contact_stress(&mut self, us: &[f64], vs: &[f64], dt: f64) -> usize {
        let n = self.nx * self.ny;
        if self.skel.len() != n || self.sigma.len() != n || !self.skel.iter().any(|&s| s) {
            return 0;
        }
        let skel = self.skel.clone();
        let pt: Vec<f64> = self.p.iter().zip(&self.sigma).map(|(a, b)| a + b).collect();
        let (uf, vf) = self.face_velocities(us, vs, &pt, dt);
        let (ds, it) = self.solve_skeleton(&skel, &uf, &vf, dt);
        for c in 0..n {
            if skel[c] {
                self.sigma[c] = (self.sigma[c] + ds[c]).max(0.0);
            }
        }
        it
    }

    /// Correction δσ' de la pression de CONTACT des lits perméables (cellules `skel`, voir mp_percolation.rs) qui rend leur squelette
    /// incompressible : ∇·u = 0 dans chaque cellule du lit, u = (uf, vf) − dt/ρ_f ∇δσ' sur les faces du mélange (δσ' = 0 hors du lit :
    /// Dirichlet vers le fluide libre et les lits étanches, dont la pression est celle du mélange ; rien à travers les parois et les faces
    /// à vitesse imposée). Matrice SPD, mêmes coefficients que la projection (sans compressibilité ni percolation : le fluide des pores
    /// est déjà passé). Renvoie (δσ', itérations du gradient conjugué).
    fn solve_skeleton(&self, skel: &[bool], uf: &[f64], vf: &[f64], dt: f64) -> (Vec<f64>, usize) {
        let (nx, ny, dx) = (self.nx, self.ny, self.dx);
        let n = nx * ny;
        let mut st = Stencil { diag: vec![1.0; n], ae: vec![0.0; n], an: vec![0.0; n] };
        let mut b = vec![0.0; n];
        let open = |side: usize| matches!(self.bc[side], Bc::Open { .. });
        for j in 0..ny {
            for i in 0..nx {
                let c = j * nx + i;
                if !skel[c] {
                    continue;
                }
                let mut d = 0.0;
                let edge = 2.0 * dt / (self.rho_mix[c] * dx * dx);
                if i + 1 < nx {
                    if self.link(c, c + 1, self.face_u_fixed(self.uidx(i + 1, j))) > 0 {
                        let t = dt / (self.face_rho_x(i + 1, j) * dx * dx);
                        d += t;
                        if skel[c + 1] {
                            st.ae[c] = t;
                        }
                    }
                } else if open(1) {
                    d += edge;
                }
                if i > 0 {
                    if self.link(c - 1, c, self.face_u_fixed(self.uidx(i, j))) > 0 {
                        d += dt / (self.face_rho_x(i, j) * dx * dx);
                    }
                } else if open(0) {
                    d += edge;
                }
                if j + 1 < ny {
                    if self.link(c, c + nx, self.face_v_fixed(self.vidx(i, j + 1))) > 0 {
                        let t = dt / (self.face_rho_y(i, j + 1) * dx * dx);
                        d += t;
                        if skel[c + nx] {
                            st.an[c] = t;
                        }
                    }
                } else if open(3) {
                    d += edge;
                }
                if j > 0 {
                    if self.link(c - nx, c, self.face_v_fixed(self.vidx(i, j))) > 0 {
                        d += dt / (self.face_rho_y(i, j) * dx * dx);
                    }
                } else if open(2) {
                    d += edge;
                }
                // (lit entièrement clos, sans voisin de Dirichlet : matrice singulière — très légère compressibilité du squelette)
                st.diag[c] = (d * (1.0 + 1e-9)).max(1e-300);
                // (pas de dilatation d'un squelette tassé au-delà de α_max : le frottement l'empêche, et σ' accumulait d'un pas à
                // l'autre l'impulsion qui la relançait — 2e4 Pa au fond d'un lit de braises de 750 Pa de poids après 110 pas)
                b[c] = -self.divergence_of(uf, vf, i, j);
            }
        }
        let mut x = vec![0.0; n];
        let it = if self.multigrid { self.pcg_mg(&st, &b, &mut x, Some(skel), None) } else { self.pcg_ic0(&st, &b, &mut x, None) };
        (x, it)
    }

    /// Projette (u*, v*) : met à jour self.p et self.u/self.v (u^{n+1}). `un`, `vn` = vitesses du début de pas (celles qui ont
    /// servi à advecter les masses). Retourne le nombre total d'itérations PCG.
    pub(crate) fn solve_pressure(&mut self, dt: f64, us: &[f64], vs: &[f64], un: &[f64], vn: &[f64], err0: &[f64]) -> usize {
        let relax = (self.closure_relax * self.closure_relax_scale).clamp(0.0, 1.0);
        let small = CLOSURE_SMALL * self.closure_relax_scale;
        let gas_ids: Vec<usize> = (0..self.reg.phases.len()).filter(|&k| self.reg.phases[k].is_gas()).collect();
        let (nx, ny) = (self.nx, self.ny);
        let n = nx * ny;
        // percolation : conductances de Darcy (Ergun évalué sur p du pas précédent) ; cellules à deux pressions (lits perméables)
        let darcy = self.darcy_faces();
        let skel: Vec<bool> = darcy.as_ref().map_or_else(|| vec![false; n], |d| d.skel.clone());
        self.skeleton_transitions(&skel);
        let p_start = self.p.clone();
        let ng = self.groups.len();
        // températures de groupe du début de la projection et dT_g/dp adiabatique (fixes pendant les itérations de Newton)
        let t_star = self.tg.clone();
        let mut gam0 = vec![vec![0.0; n]; ng];
        let chi: Vec<f64> = super::mp_par::collect(n, MIN_HEAVY, |c| self.thermal_coupling(c, dt));
        {
            let gp: Vec<SyncPtr<f64>> = gam0.iter_mut().map(|v| SyncPtr(v.as_mut_ptr())).collect();
            let (ts, ps, chi) = (&t_star, &p_start, &chi);
            let this = &*self;
            for_ranges(n, MIN_HEAVY, |lo, hi| {
                let (mut tc, mut ga, mut ks) = ([0.0f64; MAX_PHASES], [0.0f64; MAX_PHASES], [0.0f64; MAX_PHASES]);
                for c in lo..hi {
                    if this.is_solid(c) || this.pinned(c) {
                        continue;
                    }
                    for g in 0..ng {
                        tc[g] = ts[g][c];
                    }
                    this.cell_closure_g(c, ps[c], &tc[..ng], chi[c], &mut ga[..ng], &mut ks[..ng]);
                    for g in 0..ng {
                        // SAFETY : la cellule c n'est écrite que par sa tranche
                        unsafe { *gp[g].at(c) = ga[g] }
                    }
                }
            });
        }
        let mut pk = self.p.clone();
        let mut total_it = 0;
        let mut kappa = vec![0.0; n];
        // plancher de pression par cellule : P_FLOOR, ou la pression de vapeur saturante d'un liquide présent (modèle de cavité discrète,
        // DVCM) — un liquide ne supporte pas de traction : sous p_sat il se vaporise, la cavité de vapeur comble le déficit de volume
        // (voir `phase_change_step`). Sans cela l'eau reste à 10 Pa, l'erreur de volume s'accumule et finit en coup de bélier.
        let floor: Vec<f64> = super::mp_par::collect(n, MIN_HEAVY, |c| {
                let mut f = P_FLOOR;
                if self.phase_change && !self.is_solid(c) {
                    for pc in &self.reg.changes {
                        if self.alpha[pc.liquid][c] > 1e-3 {
                            f = f.max(pc.p_sat(self.phase_temp(pc.liquid, c)).min(0.95 * p_start[c].max(P_FLOOR)));
                        }
                    }
                }
                f
            });
        let mut clamped = vec![false; n];
        let mut tkv = t_star.clone();
        let mut newton = 0;
        // divergence du flux de Darcy qui a servi à ce pas
        let (q_old_u, q_old_v) = (self.q_u.clone(), self.q_v.clone());
        // lits perméables : pression de contact σ' (celle du pas précédent pour commencer ; nulle hors des cellules à deux pressions) ;
        // le mélange est poussé par la pression totale p + σ'
        let mut sig: Vec<f64> = (0..n).map(|c| if skel[c] { self.sigma[c].max(0.0) } else { 0.0 }).collect();
        let mut skel_pending = skel.iter().any(|&s| s);
        let total = |pk: &[f64], sig: &[f64]| -> Vec<f64> { pk.iter().zip(sig).map(|(a, b)| a + b).collect() };
        // corps rigides (mp_bodies.rs) : force du fluide sur chacun ; corps libres résolus avec la pression (terme de rang faible
        // (dt/dx²) J W Jᵀ dans la matrice, volume qu'ils balaieront au pas suivant dans le second membre)
        if self.has_bodies() && self.body_owner.len() != n {
            self.update_body_fractions();
        }
        let couplings: Vec<BodyCoupling> = if self.has_bodies() { self.body_couplings(dt) } else { vec![] };
        let psrc: Vec<f64> = if self.push_src.len() == n { self.push_src.clone() } else { vec![] };
        // débit du fluide collé aux corps en mouvement, position après leur pas : la divergence qui compte pour le pas suivant est celle
        // du débit relatif (voir `body_attached_flux`)
        let attached = self.body_attached_flux();
        let dx2 = self.dx * self.dx;
        let act: Vec<bool> = (0..n).map(|m| !self.is_solid(m) && !self.pinned(m) && !self.body_trace(m)).collect();
        let lowrank = if couplings.iter().any(|c| c.free) {
            Some(BodyLowRank {
                blocks: couplings
                    .iter()
                    .filter(|c| c.free)
                    .map(|c| {
                        let cells: Vec<(usize, [f64; 3])> = c.cells.iter().filter(|(m, _)| act[*m]).cloned().collect();
                        let s: [[f64; 3]; 3] = std::array::from_fn(|r| std::array::from_fn(|q| dt / dx2 * c.w[r][q]));
                        (cells, s)
                    })
                    .collect(),
            })
        } else {
            None
        };
        for _newton in 0..4 {
            newton += 1;
            let mut vol = vec![1.0; n];
            {
                let (volp, kp) = (SyncPtr(vol.as_mut_ptr()), SyncPtr(kappa.as_mut_ptr()));
                let tkp: Vec<SyncPtr<f64>> = tkv.iter_mut().map(|v| SyncPtr(v.as_mut_ptr())).collect();
                let (ts, g0, pkr, ps, chi) = (&t_star, &gam0, &pk, &p_start, &chi);
                let this = &*self;
                for_ranges(n, MIN_HEAVY, |lo, hi| {
                    let (mut tc, mut ga, mut ks) = ([0.0f64; MAX_PHASES], [0.0f64; MAX_PHASES], [0.0f64; MAX_PHASES]);
                    for c in lo..hi {
                        if this.is_solid(c) {
                            continue;
                        }
                        // SAFETY (écritures) : la cellule c n'est écrite que par sa tranche
                        if this.pinned(c) || this.body_trace(c) {
                            unsafe { *kp.at(c) = 1.0 };
                            continue; // cellule épinglée : δp = 0
                        }
                        for g in 0..ng {
                            tc[g] = ts[g][c] + g0[g][c] * (pkr[c] - ps[c]);
                            unsafe { *tkp[g].at(c) = tc[g] };
                        }
                        let (v, kap) = this.cell_closure_g(c, pkr[c], &tc[..ng], chi[c], &mut ga[..ng], &mut ks[..ng]);
                        unsafe {
                            *volp.at(c) = v;
                            *kp.at(c) = kap.max(1e-30);
                        }
                    }
                });
            }
            let st = self.build_stencil(&kappa, dt, darcy.as_ref());
            // corps libres : le fluide qu'ils entraînent va à la vitesse que leur donne la pression p_k (pas celle du pas précédent : la
            // couche entraînée en retard sur un corps qui accélère agissait comme une inertie de plus)
            if lowrank.is_some() {
                let pt = total(&pk, &sig);
                for c in couplings.iter().filter(|c| c.free) {
                    let v = c.velocity(&pt, dt);
                    let b = &mut self.bodies[c.body];
                    b.vel = [v[0], v[1]];
                    b.omega = v[2];
                }
                self.update_body_face_velocities();
            }
            let (ut, vt) = self.face_velocities(us, vs, &total(&pk, &sig), dt);
            let qt = darcy.as_ref().map(|df| self.darcy_flux(df, &pk));
            let mut b = vec![0.0; n];
            let bp = SyncPtr(b.as_mut_ptr());
            for_rows(ny, nx, MIN_HEAVY, |j| {
                for i in 0..nx {
                    let c = j * nx + i;
                    if self.is_solid(c) || self.pinned(c) || self.body_trace(c) {
                        continue;
                    }
                    let mut d_new = self.divergence_of(&ut, &vt, i, j);
                    if let Some((eu, ev)) = &attached {
                        d_new -= self.divergence_of(eu, ev, i, j);
                    }
                    let mut d_old = self.divergence_of(un, vn, i, j);
                    if let Some((qu, qv)) = &qt {
                        d_new += self.divergence_of(qu, qv, i, j);
                    }
                    if self.percolation {
                        d_old += self.divergence_of(&q_old_u, &q_old_v, i, j);
                    }
                    // erreur qui restera APRÈS l'advection de ce pas (u^n, dont la divergence est la correction décidée au pas
                    // précédent) : corriger err0 lui-même recorrigeait cette part déjà en route — e_{n+2} = e_{n+1} − e_n, oscillateur
                    // neutre de période 6 pas (barrage : ε jusqu'à 45 % dans l'eau pure, pression de 1.9 kPa à 93 bar, vapeur à
                    // l'impact). Sur f = err0 − dt·∇·u^n : f_{n+1} = f_n − corr, sans oscillation.
                    // corps libres : le volume qu'ils viennent de déplacer dans la maille était anticipé par la projection précédente (ses
                    // flux lui ont fait place) ; ce n'est pas une erreur héritée — sans cela, au-delà de CLOSURE_SMALL, sa correction partielle
                    // se reportait sur la compression : niveau de pression d'une cuve fermée d'huile qui oscillait de ±5 kPa puis s'effondrait
                    let e0 = err0[c] + psrc.get(c).copied().unwrap_or(0.0);
                    let f = e0 - dt * d_old;
let corr = closure_correction(f, gas_ids.iter().map(|&k| self.alpha[k][c]).sum(), relax, small);
                    // SAFETY : la cellule c n'est écrite que par sa ligne
                    unsafe { *bp.at(c) = (vol[c] - 1.0 - (e0 - corr)) / dt - (d_new - d_old) }
                }
            });
            // corps libres : le volume que chacun balaiera au pas suivant, à la vitesse de la pression p_k (la correction δp y entre par
            // le terme de rang faible)
            if lowrank.is_some() {
                let pt = total(&pk, &sig);
                for c in couplings.iter().filter(|c| c.free) {
                    let vk = c.velocity(&pt, dt);
                    for (m, j) in &c.cells {
                        if act[*m] {
                            b[*m] += (j[0] * vk[0] + j[1] * vk[1] + j[2] * vk[2]) / dx2;
                        }
                    }
                }
            }
            let mut dp = vec![0.0; n];
            total_it += self.pcg(&st, &b, &mut dp, lowrank.as_ref());
            let dmax = {
                let (pkp, clp) = (SyncPtr(pk.as_mut_ptr()), SyncPtr(clamped.as_mut_ptr()));
                let (tkv, kap, dpr, fl, ps) = (&tkv, &kappa, &dp, &floor, &p_start);
                let this = &*self;
                map_ranges(n, MIN_HEAVY, |lo, hi| {
                    let mut tc = [0.0f64; MAX_PHASES];
                    let mut dmax = 0.0f64;
                    for c in lo..hi {
                        if !this.is_solid(c) && !this.pinned(c) && !this.body_trace(c) {
                            // très grands écarts (> 50 % de p) dans une cellule contenant du gaz : mise à jour polytropique (voir
                            // `gas_newton_update`)
                            // SAFETY (tout le bloc) : la cellule c n'est lue et écrite que par sa tranche
                            unsafe {
                                let p_old = *pkp.at(c);
                                for g in 0..ng {
                                    tc[g] = tkv[g][c];
                                }
                                let mut pn = p_old + this.gas_newton_update(c, p_old, &tc[..ng], kap[c], dpr[c]);
                                // Plancher de pression : aucun fluide ne descend sous P_FLOOR (vide / cavitation). Avant, seules les cellules
                                // contenant du gaz étaient protégées : une cellule quasi vide d'eau seule pouvait atteindre −8e7 Pa, puis le
                                // travail de pression −dt·p·∇·u sur sa capacité calorifique minuscule donnait T = −1.6e5 K → NaN.
                                if !pn.is_finite() {
                                    pn = ps[c].max(fl[c]);
                                } else if pn < fl[c] {
                                    pn = fl[c];
                                    *clp.at(c) = true;
                                } else {
                                    *clp.at(c) = false;
                                }
                                *pkp.at(c) = pn;
                                dmax = dmax.max((pn - p_old).abs());
                            }
                        }
                    }
                    dmax
                })
                .into_iter()
                .fold(0.0f64, f64::max)
            };
            // squelette des lits perméables (une fois par pas, après la première itération) : σ' rend ∇·u nul dans le lit, relâchée à
            // 0 où elle deviendrait négative (le lit se décolle) ; les itérations suivantes rétablissent le volume des cellules
            let mut sig_moved = false;
            if skel_pending {
                skel_pending = false;
                let (uf, vf) = self.face_velocities(us, vs, &total(&pk, &sig), dt);
                let (ds, it) = self.solve_skeleton(&skel, &uf, &vf, dt);
                total_it += it;
                for c in 0..n {
                    if skel[c] {
                        let s1 = (sig[c] + ds[c]).max(0.0);
                        sig_moved |= (s1 - sig[c]).abs() >= 1e-3;
                        sig[c] = s1;
                    }
                }
            }
            if dmax < 1e-3 && !sig_moved {
                break;
            }
        }
        self.last_newton_iters = newton;
        self.pressure_clamped = clamped;
        // corps : vitesse des corps libres pour le pas suivant (le fluide qu'ils entraînent la prend aussi), force du fluide sur chacun
        if !couplings.is_empty() {
            let pt = total(&pk, &sig);
            let mut forces = vec![[0.0; 3]; self.bodies.len()];
            let mut visc = vec![[0.0; 3]; self.bodies.len()];
            for c in &couplings {
                let b = &self.bodies[c.body];
                let v = if c.free { c.velocity(&pt, dt) } else { [b.vel[0], b.vel[1], b.omega] };
                forces[c.body] = c.force(&pt, v);
                visc[c.body] = std::array::from_fn(|r| c.fu[r] - (0..3).map(|q| c.d[r][q] * v[q]).sum::<f64>());
                if c.free {
                    let b = &mut self.bodies[c.body];
                    b.vel = [v[0], v[1]];
                    b.omega = v[2];
                }
            }
            self.body_force = forces;
            self.body_force_visc = visc;
            if lowrank.is_some() {
                self.update_body_face_velocities();
            }
        }
        let (uf, vf) = self.face_velocities(us, vs, &total(&pk, &sig), dt);
        self.u = uf;
        self.v = vf;
        self.sigma = sig;
        self.skel = skel;
        // flux de Darcy de la pression finale : transportés au pas suivant (`percolate`)
        match &darcy {
            Some(df) => (self.q_u, self.q_v) = self.darcy_flux(df, &pk),
            None => {
                if self.q_u.iter().chain(self.q_v.iter()).any(|&q| q != 0.0) {
                    self.q_u.iter_mut().for_each(|q| *q = 0.0);
                    self.q_v.iter_mut().for_each(|q| *q = 0.0);
                }
            }
        }
        self.p = pk;
        total_it
    }
}
