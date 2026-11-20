//! Multigrille géométrique (V-cycle) utilisé comme PRÉCONDITIONNEUR du gradient conjugué de la pression (MGPCG, cf. McAdams et al. 2010).
//!
//! Le Cholesky incomplet IC(0) demandait ~O(N^½) itérations (270 sur 80×60, bien plus sur des grilles fines) et ses deux remontées
//! triangulaires sont intrinsèquement séquentielles. Ici :
//!  * hiérarchie de grilles par agrégation 2×2 ; opérateur grossier REDISCRÉTISÉ : couplage grossier = ½ Σ des deux couplages fins qui
//!    traversent la face grossière (moyenne des transmissivités dt/(ρ dx²) avec dx doublé et une aire ×4), terme diagonal restant
//!    (Helmholtz κ/dt, bords ouverts, voisins épinglés) sommé sur l'agrégat ; lignes identité (solides, cellules épinglées) exclues ;
//!  * lisseur Gauss–Seidel rouge-noir (rouge puis noir avant la correction grossière, noir puis rouge après) : le V-cycle est un opérateur
//!    linéaire SYMÉTRIQUE défini positif, condition pour préconditionner un gradient conjugué ;
//!  * lissage, résidus, restriction et prolongation parallélisés (rayon) par lignes sur les grilles assez grandes.
//! Nombre d'itérations du gradient conjugué : ~10–30, presque indépendant de la taille de grille et du rapport de densités eau/air.

use rayon::prelude::*;

/// en dessous de ce nombre de cellules, une grille est traitée séquentiellement (le coût de la répartition dépasse le gain)
pub(crate) const PAR_MIN: usize = PAR_MIN_CELLS;
/// seuil de parallélisation (cellules)
pub const PAR_MIN_CELLS: usize = 16384;

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

#[derive(Clone, Copy)]
struct Ptr(*mut f64);
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}

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
        let f = |(j, row): (usize, &mut [f64])| {
            let base = j * nx;
            for i in 0..nx {
                let c = base + i;
                row[i] = self.diag[c] * x[c] - self.off(i, j, c, x);
            }
        };
        if x.len() >= PAR_MIN {
            out.par_chunks_mut(nx).enumerate().for_each(f);
        } else {
            out.chunks_mut(nx).enumerate().for_each(f);
        }
    }
    /// une demi-passe de Gauss–Seidel sur la couleur `color` ((i + j) % 2)
    fn gs_color(&self, b: &[f64], x: &mut [f64], color: usize) {
        let (nx, ny) = (self.nx, self.ny);
        let n = x.len();
        let row = |xs: &mut [f64], j: usize| {
            let mut i = (color + j) % 2;
            let base = j * nx;
            while i < nx {
                let c = base + i;
                xs[c] = if self.active[c] { (b[c] + self.off(i, j, c, xs)) / self.diag[c] } else { b[c] / self.diag[c] };
                i += 2;
            }
        };
        if n >= PAR_MIN {
            let ptr = Ptr(x.as_mut_ptr());
            (0..ny).into_par_iter().for_each(|j| {
                let p = ptr;
                // SAFETY : chaque ligne j n'écrit que ses cellules de la couleur `color` ; les voisins lus sont de l'autre couleur (non
                // écrits pendant cette demi-passe) : aucune course de données.
                let xs = unsafe { std::slice::from_raw_parts_mut(p.0, n) };
                row(xs, j);
            });
        } else {
            for j in 0..ny {
                row(x, j);
            }
        }
    }
    fn residual(&self, b: &[f64], x: &[f64], r: &mut [f64]) {
        self.apply(x, r);
        let f = |(rr, bb): (&mut f64, &f64)| *rr = bb - *rr;
        if r.len() >= PAR_MIN {
            r.par_iter_mut().zip(b.par_iter()).for_each(f);
        } else {
            r.iter_mut().zip(b.iter()).for_each(f);
        }
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
        bc.iter_mut().for_each(|v| *v = 0.0);
        xc.iter_mut().for_each(|v| *v = 0.0);
        {
            let (f, c) = (&self.levels[l], &self.levels[l + 1]);
            for j in 0..f.ny {
                for i in 0..f.nx {
                    let k = j * f.nx + i;
                    if f.active[k] {
                        bc[(j / 2) * c.nx + i / 2] += r[k];
                    }
                }
            }
        }
        self.vcycle(l + 1, &bc, &mut xc);
        // prolongation constante par agrégat
        {
            let (f, c) = (&self.levels[l], &self.levels[l + 1]);
            let nxf = f.nx;
            let cnx = c.nx;
            let add = |(j, row): (usize, &mut [f64])| {
                for i in 0..nxf {
                    if f.active[j * nxf + i] {
                        row[i] += xc[(j / 2) * cnx + i / 2];
                    }
                }
            };
            if x.len() >= PAR_MIN {
                x.par_chunks_mut(nxf).enumerate().for_each(add);
            } else {
                x.chunks_mut(nxf).enumerate().for_each(add);
            }
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

/// produit scalaire (parallèle sur les grands vecteurs)
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    if a.len() >= PAR_MIN {
        a.par_iter().zip(b.par_iter()).map(|(x, y)| x * y).sum()
    } else {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}
