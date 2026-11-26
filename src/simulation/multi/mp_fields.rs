//! Champs de valeurs des conditions initiales (commande « champ » des protocoles, voir EXPERIENCES.md) : température, vitesses,
//! fraction d'un fluide ou solides, lus dans un fichier (CSV : une ligne de valeurs par rangée ; PNG : niveaux de gris) ou donnés
//! par une formule en x, y. La grille du fichier est étalée sur la boîte de la zone (le domaine entier par défaut) et lue au centre
//! de chaque maille, en interpolation bilinéaire (au plus proche pour les solides). Une valeur absente (case vide du CSV, pixel
//! transparent) laisse la maille telle quelle.

use std::path::Path;

use super::mp_cells::MpThermal;
use super::mp_experiment::{find_fluid, find_solid, FieldQty, FieldSource, FluidRef, Zone};
use super::mp_view::{apply_tool_with, MpTool, ToolOpts};
use super::multiphase::MultiphaseSim;

/// Grille de valeurs lue dans un fichier : `h` rangées de `w` valeurs, rangée 0 = première du fichier (le haut d'une image).
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub w: usize,
    pub h: usize,
    pub v: Vec<f64>,
}

/// CSV (ou texte) : valeurs séparées par des virgules, points-virgules, espaces ou tabulations ; case vide, « nan » ou « - » =
/// pas de valeur ; lignes vides et commençant par # ignorées. Toutes les rangées doivent avoir le même nombre de valeurs.
pub fn read_csv(text: &str) -> Result<Grid, String> {
    let mut rows: Vec<Vec<f64>> = vec![];
    for (k, l) in text.lines().enumerate() {
        let t = l.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        // séparateur : point-virgule ou virgule s'il y en a, sinon blancs (une case vide entre deux séparateurs compte)
        let cells: Vec<&str> = if t.contains(';') {
            t.split(';').collect()
        } else if t.contains(',') {
            t.split(',').collect()
        } else {
            t.split_whitespace().collect()
        };
        let mut row = vec![];
        for c in cells {
            let c = c.trim();
            if c.is_empty() || c == "-" || c.eq_ignore_ascii_case("nan") {
                row.push(f64::NAN);
            } else {
                row.push(c.parse::<f64>().map_err(|_| format!("ligne {} : valeur « {c} » illisible", k + 1))?);
            }
        }
        rows.push(row);
    }
    if rows.is_empty() {
        return Err("fichier sans valeur".into());
    }
    let w = rows[0].len();
    if let Some((k, r)) = rows.iter().enumerate().find(|(_, r)| r.len() != w) {
        return Err(format!("rangée {} : {} valeurs au lieu de {w}", k + 1, r.len()));
    }
    let h = rows.len();
    Ok(Grid { w, h, v: rows.into_iter().flatten().collect() })
}

/// PNG : luminance de 0 (noir) à 1 (blanc) ; pixel transparent (alpha < ½) = pas de valeur.
pub fn read_png(bytes: &[u8]) -> Result<Grid, String> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().map_err(|e| format!("image PNG illisible : {e}"))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("image PNG illisible : {e}"))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let ch = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return Err("image PNG à palette non développée".into()),
    };
    if info.bit_depth != png::BitDepth::Eight {
        return Err("image PNG : profondeur de couleur non gérée".into());
    }
    let mut v = Vec::with_capacity(w * h);
    for y in 0..h {
        let row = &buf[y * info.line_size..];
        for x in 0..w {
            let px = &row[x * ch..x * ch + ch];
            let f = |b: u8| b as f64 / 255.0;
            let (lum, alpha) = match ch {
                1 => (f(px[0]), 1.0),
                2 => (f(px[0]), f(px[1])),
                3 => (0.2126 * f(px[0]) + 0.7152 * f(px[1]) + 0.0722 * f(px[2]), 1.0),
                _ => (0.2126 * f(px[0]) + 0.7152 * f(px[1]) + 0.0722 * f(px[2]), f(px[3])),
            };
            v.push(if alpha < 0.5 { f64::NAN } else { lum });
        }
    }
    Ok(Grid { w, h, v })
}

/// Lit un fichier de champ selon son extension (.png, sinon texte CSV).
pub fn load_grid(path: &Path) -> Result<Grid, String> {
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    if ext == "png" {
        let bytes = std::fs::read(path).map_err(|e| format!("« {} » : {e}", path.display()))?;
        read_png(&bytes)
    } else {
        let text = std::fs::read_to_string(path).map_err(|e| format!("« {} » : {e}", path.display()))?;
        read_csv(&text).map_err(|e| format!("« {} » : {e}", path.display()))
    }
}

impl Grid {
    fn at(&self, i: usize, j: usize) -> f64 {
        self.v[j * self.w + i]
    }
    /// Valeur au point (u, v) ∈ [0, 1]² de la boîte (u vers la droite, v vers le BAS : rangée 0 en v = 0). Chaque valeur occupe une
    /// case ; bilinéaire entre les centres des cases, les cases sans valeur écartées (poids des autres renormalisés), au plus proche
    /// si `nearest`. NaN si aucune case utile n'a de valeur.
    pub fn sample(&self, u: f64, v: f64, nearest: bool) -> f64 {
        let fx = (u.clamp(0.0, 1.0) * self.w as f64 - 0.5).clamp(0.0, (self.w - 1) as f64);
        let fy = (v.clamp(0.0, 1.0) * self.h as f64 - 0.5).clamp(0.0, (self.h - 1) as f64);
        let near = || self.at(fx.round() as usize, fy.round() as usize);
        if nearest {
            return near();
        }
        let (i0, j0) = (fx.floor() as usize, fy.floor() as usize);
        let (i1, j1) = ((i0 + 1).min(self.w - 1), (j0 + 1).min(self.h - 1));
        let (tx, ty) = (fx - i0 as f64, fy - j0 as f64);
        let q = [self.at(i0, j0), self.at(i1, j0), self.at(i0, j1), self.at(i1, j1)];
        let w = [(1.0 - tx) * (1.0 - ty), tx * (1.0 - ty), (1.0 - tx) * ty, tx * ty];
        let (mut s, mut ws) = (0.0, 0.0);
        for k in 0..4 {
            if q[k].is_finite() && w[k] > 0.0 {
                s += w[k] * q[k];
                ws += w[k];
            }
        }
        if ws > 1e-9 { s / ws } else { near() }
    }
}

// ------------------------------------------------------------------------------------------------ formules

#[derive(Clone, Debug)]
enum Expr {
    Num(f64),
    /// 0 = x, 1 = y, 2 = lx, 3 = ly
    Var(usize),
    Neg(Box<Expr>),
    Bin(char, Box<Expr>, Box<Expr>),
    Call(Func, Vec<Expr>),
}

#[derive(Clone, Copy, Debug)]
enum Func {
    Exp,
    Ln,
    Log10,
    Sqrt,
    Abs,
    Sin,
    Cos,
    Tan,
    Tanh,
    Atan2,
    Min,
    Max,
    Pow,
    Hypot,
    Clamp,
    Step,
    Si,
}

/// Formule en x, y (m), lx, ly (taille du domaine, m) ; + − × / ^, parenthèses, pi, e ; fonctions exp, ln, log10, sqrt, abs, sin,
/// cos, tan, tanh, atan2, min, max, pow, hypot, clamp(v, a, b), step(v) (0 si v < 0, sinon 1), si(c, a, b) (a si c > 0, sinon b).
#[derive(Clone, Debug)]
pub struct Formula {
    e: Expr,
}

struct FP<'a> {
    s: &'a [char],
    i: usize,
}

impl<'a> FP<'a> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_whitespace() {
            self.i += 1;
        }
    }
    fn peek(&mut self) -> Option<char> {
        self.ws();
        self.s.get(self.i).copied()
    }
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn err<T>(&self, m: &str) -> Result<T, String> {
        Err(format!("formule, caractère {} : {m}", self.i + 1))
    }
    fn expr(&mut self) -> Result<Expr, String> {
        let mut a = self.term()?;
        loop {
            if self.eat('+') {
                a = Expr::Bin('+', Box::new(a), Box::new(self.term()?));
            } else if self.eat('-') {
                a = Expr::Bin('-', Box::new(a), Box::new(self.term()?));
            } else {
                return Ok(a);
            }
        }
    }
    fn term(&mut self) -> Result<Expr, String> {
        let mut a = self.unary()?;
        loop {
            if self.eat('*') || self.eat('×') {
                a = Expr::Bin('*', Box::new(a), Box::new(self.unary()?));
            } else if self.eat('/') {
                a = Expr::Bin('/', Box::new(a), Box::new(self.unary()?));
            } else {
                return Ok(a);
            }
        }
    }
    fn unary(&mut self) -> Result<Expr, String> {
        if self.eat('-') {
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        if self.eat('+') {
            return self.unary();
        }
        self.power()
    }
    fn power(&mut self) -> Result<Expr, String> {
        let b = self.primary()?;
        if self.eat('^') {
            // associatif à droite : 2^3^2 = 2^9 ; -x^2 = -(x^2)
            let e = self.unary()?;
            return Ok(Expr::Bin('^', Box::new(b), Box::new(e)));
        }
        Ok(b)
    }
    fn primary(&mut self) -> Result<Expr, String> {
        match self.peek() {
            Some('(') => {
                self.i += 1;
                let e = self.expr()?;
                if !self.eat(')') {
                    return self.err("il manque « ) »");
                }
                Ok(e)
            }
            Some(c) if c.is_ascii_digit() || c == '.' => {
                let st = self.i;
                while self.i < self.s.len() && (self.s[self.i].is_ascii_digit() || self.s[self.i] == '.') {
                    self.i += 1;
                }
                if self.i < self.s.len() && (self.s[self.i] == 'e' || self.s[self.i] == 'E') {
                    let mut j = self.i + 1;
                    if j < self.s.len() && (self.s[j] == '-' || self.s[j] == '+') {
                        j += 1;
                    }
                    if j < self.s.len() && self.s[j].is_ascii_digit() {
                        while j < self.s.len() && self.s[j].is_ascii_digit() {
                            j += 1;
                        }
                        self.i = j;
                    }
                }
                let t: String = self.s[st..self.i].iter().collect();
                t.parse::<f64>().map(Expr::Num).map_err(|_| format!("formule : nombre « {t} » illisible"))
            }
            Some(c) if c.is_alphabetic() || c == '_' => {
                let st = self.i;
                while self.i < self.s.len() && (self.s[self.i].is_alphanumeric() || self.s[self.i] == '_') {
                    self.i += 1;
                }
                let name: String = self.s[st..self.i].iter().collect::<String>().to_lowercase();
                if self.peek() == Some('(') {
                    self.i += 1;
                    let f = match name.as_str() {
                        "exp" => Func::Exp,
                        "ln" | "log" => Func::Ln,
                        "log10" => Func::Log10,
                        "sqrt" | "racine" => Func::Sqrt,
                        "abs" => Func::Abs,
                        "sin" => Func::Sin,
                        "cos" => Func::Cos,
                        "tan" => Func::Tan,
                        "tanh" => Func::Tanh,
                        "atan2" => Func::Atan2,
                        "min" => Func::Min,
                        "max" => Func::Max,
                        "pow" => Func::Pow,
                        "hypot" => Func::Hypot,
                        "clamp" => Func::Clamp,
                        "step" | "marche" => Func::Step,
                        "si" | "if" => Func::Si,
                        _ => return Err(format!("formule : fonction « {name} » inconnue")),
                    };
                    let mut args = vec![];
                    if !self.eat(')') {
                        loop {
                            args.push(self.expr()?);
                            if self.eat(')') {
                                break;
                            }
                            if !self.eat(',') {
                                return self.err("il faut « , » ou « ) » entre les arguments");
                            }
                        }
                    }
                    let n = match f {
                        Func::Atan2 | Func::Pow | Func::Hypot => 2,
                        Func::Clamp | Func::Si => 3,
                        Func::Min | Func::Max => args.len().max(1),
                        _ => 1,
                    };
                    if args.len() != n || args.is_empty() {
                        return Err(format!("formule : « {name} » attend {n} argument(s), {} donné(s)", args.len()));
                    }
                    return Ok(Expr::Call(f, args));
                }
                match name.as_str() {
                    "x" => Ok(Expr::Var(0)),
                    "y" => Ok(Expr::Var(1)),
                    "lx" => Ok(Expr::Var(2)),
                    "ly" => Ok(Expr::Var(3)),
                    "pi" => Ok(Expr::Num(std::f64::consts::PI)),
                    "e" => Ok(Expr::Num(std::f64::consts::E)),
                    _ => Err(format!("formule : nom « {name} » inconnu (variables : x, y, lx, ly ; constantes : pi, e)")),
                }
            }
            Some(c) => self.err(&format!("« {c} » inattendu")),
            None => self.err("formule incomplète"),
        }
    }
}

impl Formula {
    pub fn parse(s: &str) -> Result<Formula, String> {
        let cs: Vec<char> = s.chars().collect();
        let mut p = FP { s: &cs, i: 0 };
        let e = p.expr()?;
        if p.peek().is_some() {
            return p.err("texte en trop");
        }
        Ok(Formula { e })
    }
    pub fn eval(&self, x: f64, y: f64, lx: f64, ly: f64) -> f64 {
        ev(&self.e, &[x, y, lx, ly])
    }
}

fn ev(e: &Expr, v: &[f64; 4]) -> f64 {
    match e {
        Expr::Num(a) => *a,
        Expr::Var(k) => v[*k],
        Expr::Neg(a) => -ev(a, v),
        Expr::Bin(op, a, b) => {
            let (a, b) = (ev(a, v), ev(b, v));
            match op {
                '+' => a + b,
                '-' => a - b,
                '*' => a * b,
                '/' => a / b,
                _ => a.powf(b),
            }
        }
        Expr::Call(f, args) => {
            let a: Vec<f64> = args.iter().map(|x| ev(x, v)).collect();
            match f {
                Func::Exp => a[0].exp(),
                Func::Ln => a[0].ln(),
                Func::Log10 => a[0].log10(),
                Func::Sqrt => a[0].sqrt(),
                Func::Abs => a[0].abs(),
                Func::Sin => a[0].sin(),
                Func::Cos => a[0].cos(),
                Func::Tan => a[0].tan(),
                Func::Tanh => a[0].tanh(),
                Func::Atan2 => a[0].atan2(a[1]),
                Func::Min => a.iter().copied().fold(f64::INFINITY, f64::min),
                Func::Max => a.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                Func::Pow => a[0].powf(a[1]),
                Func::Hypot => a[0].hypot(a[1]),
                Func::Clamp => a[0].clamp(a[1].min(a[2]), a[1].max(a[2])),
                Func::Step => {
                    if a[0] < 0.0 {
                        0.0
                    } else {
                        1.0
                    }
                }
                Func::Si => {
                    if a[0] > 0.0 {
                        a[1]
                    } else {
                        a[2]
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ application

/// Source prête à être lue en (x, y)
enum Src {
    Grid { g: Grid, top: bool, range: Option<(f64, f64)>, b0: [f64; 2], b1: [f64; 2] },
    Formula { f: Formula, lx: f64, ly: f64 },
}

impl Src {
    fn value(&self, x: f64, y: f64, nearest: bool) -> f64 {
        match self {
            Src::Grid { g, top, range, b0, b1 } => {
                let u = (x - b0[0]) / (b1[0] - b0[0]).max(1e-300);
                let w = (y - b0[1]) / (b1[1] - b0[1]).max(1e-300);
                let v = if *top { 1.0 - w } else { w };
                let s = g.sample(u, v, nearest);
                match range {
                    Some((a, b)) if s.is_finite() => a + (b - a) * s,
                    _ => s,
                }
            }
            Src::Formula { f, lx, ly } => f.eval(x, y, *lx, *ly),
        }
    }
}

/// Applique un champ de valeurs à la zone ; retourne le nombre de mailles modifiées.
pub fn apply_field(sim: &mut MultiphaseSim, qty: &FieldQty, source: &FieldSource, zone: &Zone, temp: Option<f64>, base: &Path) -> Result<usize, String> {
    let (nx, ny, dx) = (sim.nx, sim.ny, sim.dx);
    let (lx, ly) = (nx as f64 * dx, ny as f64 * dx);
    let src = match source {
        FieldSource::File { path, top_origin, range } => {
            let (b0, b1) = zone.bbox(lx, ly, dx);
            Src::Grid { g: load_grid(&base.join(path))?, top: *top_origin, range: *range, b0, b1 }
        }
        FieldSource::Formula(f) => Src::Formula { f: Formula::parse(f)?, lx, ly },
    };
    let cells = zone.cells(nx, ny, dx);
    let nearest = matches!(qty, FieldQty::Solid(_));
    let center = |i: usize, j: usize| ((i as f64 + 0.5) * dx, (j as f64 + 0.5) * dx);
    let np = sim.reg.phases.len();
    let mut changed = 0usize;
    match qty {
        FieldQty::T => {
            for &(i, j) in &cells {
                let (x, y) = center(i, j);
                let t = src.value(x, y, false);
                if !t.is_finite() {
                    continue;
                }
                if !(t > 0.0) {
                    return Err(format!("champ T : température {t} K en ({x:.4}, {y:.4}) — elle doit être positive (en kelvins)"));
                }
                let c = sim.idx(i, j);
                if sim.is_solid(c) {
                    sim.set_cell_temperature(c, t);
                } else {
                    let a: Vec<f64> = (0..np).map(|k| sim.alpha[k][c]).collect();
                    sim.fill_mixture(c, &a, t);
                }
                changed += 1;
            }
        }
        FieldQty::Vx | FieldQty::Vy => {
            // valeur au centre des mailles de la zone, puis faces : moyenne des deux mailles (ou la seule qui en a une)
            let mut val = vec![f64::NAN; nx * ny];
            for &(i, j) in &cells {
                let (x, y) = center(i, j);
                let c = sim.idx(i, j);
                if !sim.is_solid(c) {
                    val[c] = src.value(x, y, false);
                }
            }
            let pick = |a: f64, b: f64| match (a.is_finite(), b.is_finite()) {
                (true, true) => Some(0.5 * (a + b)),
                (true, false) => Some(a),
                (false, true) => Some(b),
                _ => None,
            };
            if matches!(qty, FieldQty::Vx) {
                for j in 0..ny {
                    for i in 1..nx {
                        let (a, b) = (sim.idx(i - 1, j), sim.idx(i, j));
                        if sim.is_solid(a) || sim.is_solid(b) {
                            continue;
                        }
                        if let Some(v) = pick(val[a], val[b]) {
                            let f = sim.uidx(i, j);
                            sim.u[f] = v;
                            changed += 1;
                        }
                    }
                }
            } else {
                for j in 1..ny {
                    for i in 0..nx {
                        let (a, b) = (sim.idx(i, j - 1), sim.idx(i, j));
                        if sim.is_solid(a) || sim.is_solid(b) {
                            continue;
                        }
                        if let Some(v) = pick(val[a], val[b]) {
                            let f = sim.vidx(i, j);
                            sim.v[f] = v;
                            changed += 1;
                        }
                    }
                }
            }
        }
        FieldQty::Fraction(name) => {
            let target = find_fluid(sim, name)?;
            let m: Vec<f64> = match target {
                FluidRef::Phase(k) => (0..np).map(|j| if j == k { 1.0 } else { 0.0 }).collect(),
                FluidRef::Mixture(mi) => {
                    let a = &sim.mixtures[mi].1;
                    let s: f64 = a.iter().sum::<f64>().max(1e-300);
                    (0..np).map(|j| a.get(j).copied().unwrap_or(0.0) / s).collect()
                }
            };
            let amb = sim.ambient_alpha();
            for &(i, j) in &cells {
                let (x, y) = center(i, j);
                let f = src.value(x, y, false);
                let c = sim.idx(i, j);
                if !f.is_finite() || sim.is_solid(c) {
                    continue;
                }
                let f = f.clamp(0.0, 1.0);
                // reste de la maille : sa composition sans le fluide posé ; à défaut l'ambiant (sans lui), à défaut l'ambiant
                let strip = |a: &[f64]| -> Option<Vec<f64>> {
                    let r: Vec<f64> = (0..np).map(|k| if m[k] > 0.0 { 0.0 } else { a[k].max(0.0) }).collect();
                    let s: f64 = r.iter().sum();
                    (s > 1e-12).then(|| r.iter().map(|x| x / s).collect())
                };
                let cur: Vec<f64> = (0..np).map(|k| sim.alpha[k][c]).collect();
                let rest = strip(&cur).or_else(|| strip(&amb)).unwrap_or_else(|| amb.clone());
                let a: Vec<f64> = (0..np).map(|k| f * m[k] + (1.0 - f) * rest[k]).collect();
                let t = temp.unwrap_or(sim.temp[c]);
                sim.fill_mixture(c, &a, t);
                changed += 1;
            }
        }
        FieldQty::Solid(name) => {
            let s = find_solid(sim, name)?;
            let mut put = vec![];
            for &(i, j) in &cells {
                let (x, y) = center(i, j);
                let v = src.value(x, y, nearest);
                if v.is_finite() && v >= 0.5 {
                    put.push((i, j));
                }
            }
            let t = temp.unwrap_or(sim.t_ambient);
            apply_tool_with(sim, MpTool::Solid(s), &put, &ToolOpts { temp: t, cursor: (0.0, 0.0), dt: 0.0, thermal: MpThermal::Passive });
            changed = put.len();
        }
    }
    sim.refresh();
    Ok(changed)
}
