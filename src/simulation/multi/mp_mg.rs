//! Multigrille géométrique (V-cycle) utilisé comme PRÉCONDITIONNEUR du gradient conjugué de la pression (MGPCG, cf. McAdams et al. 2010).
//!
//! Le Cholesky incomplet IC(0) demandait ~O(N^½) itérations (270 sur 80×60, bien plus sur des grilles fines) et ses deux remontées
//! triangulaires sont intrinsèquement séquentielles. Ici :
//!  * hiérarchie de grilles par agrégation 2×2 ; opérateur grossier REDISCRÉTISÉ : couplage grossier = ½ Σ des deux couplages fins qui
//!    traversent la face grossière (moyenne des transmissivités dt/(ρ dx²) avec dx doublé et une aire ×4), terme diagonal restant
//!    (Helmholtz κ/dt, bords ouverts, voisins épinglés) sommé sur l'agrégat ; lignes identité (solides, cellules épinglées) exclues ;
//!  * lisseur Gauss–Seidel rouge-noir (rouge puis noir avant la correction grossière, noir puis rouge après) : le V-cycle est un opérateur
//!    linéaire SYMÉTRIQUE défini positif, condition pour préconditionner un gradient conjugué ;
//!  * lissage, résidus, restriction et prolongation parallélisés par tranches de lignes (`mp_par`) sur les grilles assez grandes ;
//!    chaque cellule est calculée exactement comme en séquentiel (résultat identique quel que soit le nombre de fils).
//! Nombre d'itérations du gradient conjugué : ~10–30, presque indépendant de la taille de grille et du rapport de densités eau/air.

use super::mp_par::{for_rows, MIN_LIGHT};
use super::multiphase::SyncPtr;

/// Matrice 5 points symétrique : (A x)_c = diag_c x_c − ae_c x_{c+1} − ae_{c−1} x_{c−1} − an_c x_{c+nx} − an_{c−nx} x_{c−nx}
#[derive(Clone)]
pub(crate) struct Level {
    pub nx: usize,
    pub ny: usize,
    pub diag: Vec<f64>,
    pub ae: Vec<f64>,
    pub an: Vec<f64>,
    /// ligne active (sinon identité, découplée)
    pub active: Vec<bool>,
}

impl Level {
    /// Σ couplages × x des voisins de la cellule (i, j) (sans modulo : i, j connus)
    #[inline(always)]
    fn off(&self, i: usize, j: usize, c: usize, x: &[f64]) -> f64 {
        let nx = self.nx;
        let mut s = 0.0;
        if i + 1 < nx {
            s += self.ae[c] * x[c + 1];
        }
        if i > 0 {
            s += self.ae[c - 1] * x[c - 1];
        }
        if j + 1 < self.ny {
            s += self.an[c] * x[c + nx];
        }
        if j > 0 {
            s += self.an[c - nx] * x[c - nx];
        }
        s
    }
    /// out = A x
    pub fn apply(&self, x: &[f64], out: &mut [f64]) {
        let nx = self.nx;
        let o = SyncPtr(out.as_mut_ptr());
        for_rows(self.ny, nx, MIN_LIGHT, |j| {
            let base = j * nx;
            for i in 0..nx {
                let c = base + i;
                // SAFETY : la ligne j n'écrit que ses cellules
                unsafe { *o.at(c) = self.diag[c] * x[c] - self.off(i, j, c, x) }
            }
        });
    }
    /// une demi-passe de Gauss–Seidel sur la couleur `color` ((i + j) % 2)
    fn gs_color(&self, b: &[f64], x: &mut [f64], color: usize) {
        let (nx, ny) = (self.nx, self.ny);
        let p = SyncPtr(x.as_mut_ptr());
        for_rows(ny, nx, MIN_LIGHT, |j| {
            let mut i = (color + j) % 2;
            let base = j * nx;
            while i < nx {
                let c = base + i;
                // SAFETY : chaque ligne j n'écrit que ses cellules de la couleur `color` ; les voisins lus sont de l'autre couleur (non
                // écrits pendant cette demi-passe) : aucune course de données. Mêmes opérations, dans le même ordre, que `off`.
                unsafe {
                    let xr = |q: usize| *p.at(q);
                    let mut s = 0.0;
                    if i + 1 < nx {
                        s += self.ae[c] * xr(c + 1);
                    }
                    if i > 0 {
                        s += self.ae[c - 1] * xr(c - 1);
                    }
                    if j + 1 < ny {
                        s += self.an[c] * xr(c + nx);
                    }
                    if j > 0 {
                        s += self.an[c - nx] * xr(c - nx);
                    }
                    *p.at(c) = if self.active[c] { (b[c] + s) / self.diag[c] } else { b[c] / self.diag[c] };
                }
                i += 2;
            }
        });
    }
    /// r = b − A x
    fn residual(&self, b: &[f64], x: &[f64], r: &mut [f64]) {
        let nx = self.nx;
        let o = SyncPtr(r.as_mut_ptr());
        for_rows(self.ny, nx, MIN_LIGHT, |j| {
            let base = j * nx;
            for i in 0..nx {
                let c = base + i;
                let ax = self.diag[c] * x[c] - self.off(i, j, c, x);
                // SAFETY : la ligne j n'écrit que ses cellules
                unsafe { *o.at(c) = b[c] - ax }
            }
        });
    }
    /// grille grossière (agrégats 2×2)
    fn coarsen(&self) -> Level {
        let (nx, ny) = (self.nx, self.ny);
        let (cx, cy) = (nx.div_ceil(2), ny.div_ceil(2));
        let nc = cx * cy;
        let mut diag = vec![0.0; nc];
        let mut ae = vec![0.0; nc];
        let mut an = vec![0.0; nc];
        let mut active = vec![false; nc];
        for j in 0..ny {
            for i in 0..nx {
                let c = j * nx + i;
                if !self.active[c] {
                    continue;
                }
                let g = (j / 2) * cx + i / 2;
                active[g] = true;
                // partie diagonale « propre » de la ligne : diag − Σ couplages (Helmholtz, bords ouverts, voisins de Dirichlet)
                let mut own = self.diag[c];
                if i + 1 < nx {
                    own -= self.ae[c];
                }
                if i > 0 {
                    own -= self.ae[c - 1];
                }
                if j + 1 < ny {
                    own -= self.an[c];
                }
                if j > 0 {
                    own -= self.an[c - nx];
                }
                diag[g] += own;
                // couplages traversant une face grossière (est / nord seulement : chaque lien une fois)
                if i + 1 < nx && (i / 2) != ((i + 1) / 2) && self.active[c + 1] {
                    ae[g] += 0.5 * self.ae[c];
                }
                if j + 1 < ny && (j / 2) != ((j + 1) / 2) && self.active[c + nx] {
                    an[g] += 0.5 * self.an[c];
                }
            }
        }
        // diagonale = partie propre + Σ couplages grossiers
        for gj in 0..cy {
            for gi in 0..cx {
                let g = gj * cx + gi;
                if !active[g] {
                    diag[g] = 1.0;
                    ae[g] = 0.0;
                    an[g] = 0.0;
                    continue;
                }
                let mut d = diag[g].max(0.0);
                if gi + 1 < cx {
                    d += ae[g];
                }
                if gi > 0 {
                    d += ae[g - 1];
                }
                if gj + 1 < cy {
                    d += an[g];
                }
                if gj > 0 {
                    d += an[g - cx];
                }
                diag[g] = d.max(1e-300);
            }
        }
        // un lien vers une cellule grossière inactive n'existe pas
        for g in 0..nc {
            if !active[g] {
                if g % cx > 0 {
                    ae[g - 1] = 0.0;
                }
                if g >= cx {
                    an[g - cx] = 0.0;
                }
            }
        }
        Level { nx: cx, ny: cy, diag, ae, an, active }
    }
}

pub(crate) struct Multigrid {
    pub levels: Vec<Level>,
    /// tampons par niveau : (résidu, second membre, solution)
    work: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>,
    pub pre: usize,
    pub post: usize,
}

impl Multigrid {
    pub fn new(fine: Level) -> Self {
        let mut levels = vec![fine];
        while {
            let l = levels.last().unwrap();
            l.nx > 3 && l.ny > 3 && l.nx * l.ny > 32
        } {
            let c = levels.last().unwrap().coarsen();
            levels.push(c);
        }
        let work = levels.iter().map(|l| (vec![0.0; l.nx * l.ny], vec![0.0; l.nx * l.ny], vec![0.0; l.nx * l.ny])).collect();
        Multigrid { levels, work, pre: 2, post: 2 }
    }

    /// z = M⁻¹ r (un V-cycle, z initialisé à 0) : opérateur symétrique défini positif
    pub fn precondition(&mut self, r: &[f64], z: &mut [f64]) {
        z.iter_mut().for_each(|v| *v = 0.0);
        self.vcycle(0, r, z);
    }

    fn vcycle(&mut self, l: usize, b: &[f64], x: &mut [f64]) {
        let last = self.levels.len() - 1;
        if l == last {
            // grille la plus grossière : balayages symétriques en nombre fixe (opérateur linéaire symétrique)
            let lev = &self.levels[l];
            for _ in 0..20 {
                lev.gs_color(b, x, 0);
                lev.gs_color(b, x, 1);
                lev.gs_color(b, x, 1);
                lev.gs_color(b, x, 0);
            }
            return;
        }
        {
            let lev = &self.levels[l];
            for _ in 0..self.pre {
                lev.gs_color(b, x, 0);
                lev.gs_color(b, x, 1);
            }
        }
        // résidu fin puis restriction (somme sur l'agrégat)
        let mut r = std::mem::take(&mut self.work[l].0);
        self.levels[l].residual(b, x, &mut r);
        let mut bc = std::mem::take(&mut self.work[l + 1].1);
        let mut xc = std::mem::take(&mut self.work[l + 1].2);
        xc.iter_mut().for_each(|v| *v = 0.0);
        {
            // somme sur l'agrégat 2×2, dans l'ordre des lignes fines (celui de l'ancienne accumulation)
            let (f, c) = (&self.levels[l], &self.levels[l + 1]);
            let o = SyncPtr(bc.as_mut_ptr());
            let r = &r;
            for_rows(c.ny, c.nx, MIN_LIGHT / 4, |gj| {
                for gi in 0..c.nx {
                    let mut s = 0.0;
                    for j in 2 * gj..(2 * gj + 2).min(f.ny) {
                        for i in 2 * gi..(2 * gi + 2).min(f.nx) {
                            let k = j * f.nx + i;
                            if f.active[k] {
                                s += r[k];
                            }
                        }
                    }
                    // SAFETY : une cellule grossière par (gi, gj)
                    unsafe { *o.at(gj * c.nx + gi) = s }
                }
            });
        }
        self.vcycle(l + 1, &bc, &mut xc);
        // prolongation constante par agrégat
        {
            let (f, c) = (&self.levels[l], &self.levels[l + 1]);
            let nxf = f.nx;
            let cnx = c.nx;
            let o = SyncPtr(x.as_mut_ptr());
            let xc = &xc;
            for_rows(f.ny, nxf, MIN_LIGHT, |j| {
                for i in 0..nxf {
                    let k = j * nxf + i;
                    if f.active[k] {
                        // SAFETY : la ligne j n'écrit que ses cellules
                        unsafe { *o.at(k) += xc[(j / 2) * cnx + i / 2] }
                    }
                }
            });
        }
        self.work[l].0 = r;
        self.work[l + 1].1 = bc;
        self.work[l + 1].2 = xc;
        let lev = &self.levels[l];
        for _ in 0..self.post {
            lev.gs_color(b, x, 1);
            lev.gs_color(b, x, 0);
        }
    }
}

