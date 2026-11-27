//! Lecture des protocoles d'expérience écrits en français (format complet : EXPERIENCES.md) → `mp_experiment::Protocol`.
//!
//! Une commande par ligne, `#` commence un commentaire ; mots-clés sans casse ni accents obligatoires (« départ » = « depart ») ;
//! noms (fluides, matériaux, fichiers) entre guillemets ; nombres avec un POINT décimal ; unités facultatives (SI par défaut :
//! m, s, K, m/s, m/s², Pa, W/m³). Un `balayage NOM = v1, v2, …` rejoue le protocole pour chaque valeur (`$NOM` dans le texte).

use super::mp_experiment::{
    norm, Action, BodyMotion, BodyQty, BodySpec, BoundarySpec, Clause, Cmp, FieldQty, FieldSource, Measure, Protocol, Quantity, Reduce, Start, Step, ThermalSpec, Zone,
};
use super::mp_view::T_IGNITION;

const P_ATM: f64 = 101325.0;

/// Erreur de lecture, avec sa ligne (à partir de 1).
#[derive(Clone, Debug, PartialEq)]
pub struct ScriptError {
    pub line: usize,
    pub msg: String,
}

impl std::fmt::Display for ScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ligne {} : {}", self.line, self.msg)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    /// mot (normalisé, tel qu'écrit)
    W(String, String),
    N(f64),
    S(String),
    LP,
    RP,
    Comma,
    Colon,
    Eq,
    /// comparaison (<, ≤, >, ≥) d'une condition d'arrêt
    Cmp(Cmp),
}

fn show(t: &Tok) -> String {
    match t {
        Tok::Cmp(c) => format!("« {} »", c.symbol()),
        Tok::W(_, s) => format!("« {s} »"),
        Tok::N(v) => format!("le nombre {v}"),
        Tok::S(s) => format!("« \"{s}\" »"),
        Tok::LP => "« ( »".into(),
        Tok::RP => "« ) »".into(),
        Tok::Comma => "une virgule".into(),
        Tok::Colon => "« : »".into(),
        Tok::Eq => "« = »".into(),
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Dim {
    Length,
    Time,
    Temp,
    Speed,
    Accel,
    Pressure,
    Power,
    /// angle : DEGRÉS sans unité (°, deg ou rad écrits)
    Angle,
    /// vitesse angulaire : DEGRÉS PAR SECONDE sans unité (°/s, rad/s, tr/min, tr/s écrits)
    AngSpeed,
    /// masse volumique : kg/m³ sans unité
    Density,
    Plain,
}

fn dim_name(d: Dim) -> &'static str {
    match d {
        Dim::Length => "une longueur (m, cm, mm)",
        Dim::Time => "une durée (s, ms, min)",
        Dim::Temp => "une température (K, °C)",
        Dim::Speed => "une vitesse (m/s, cm/s, km/h)",
        Dim::Accel => "une accélération (m/s²)",
        Dim::Pressure => "une pression (Pa, kPa, bar, atm)",
        Dim::Power => "une puissance volumique (W/m³)",
        Dim::Angle => "un angle (°, rad)",
        Dim::AngSpeed => "une vitesse de rotation (°/s, rad/s, tr/min)",
        Dim::Density => "une masse volumique (kg/m³, g/cm³)",
        Dim::Plain => "un nombre sans unité",
    }
}

/// (dimension, facteur, décalage) d'une unité écrite (normalisée : minuscules, ² → 2)
fn unit_of(w: &str) -> Option<(Dim, f64, f64)> {
    Some(match w {
        "m" => (Dim::Length, 1.0, 0.0),
        "cm" => (Dim::Length, 1e-2, 0.0),
        "mm" => (Dim::Length, 1e-3, 0.0),
        "µm" | "um" => (Dim::Length, 1e-6, 0.0),
        "km" => (Dim::Length, 1e3, 0.0),
        "s" => (Dim::Time, 1.0, 0.0),
        "ms" => (Dim::Time, 1e-3, 0.0),
        "µs" | "us" => (Dim::Time, 1e-6, 0.0),
        "min" => (Dim::Time, 60.0, 0.0),
        "h" => (Dim::Time, 3600.0, 0.0),
        "k" => (Dim::Temp, 1.0, 0.0),
        "°c" | "degc" | "c" => (Dim::Temp, 1.0, 273.15),
        "m/s" => (Dim::Speed, 1.0, 0.0),
        "cm/s" => (Dim::Speed, 1e-2, 0.0),
        "mm/s" => (Dim::Speed, 1e-3, 0.0),
        "km/h" => (Dim::Speed, 1.0 / 3.6, 0.0),
        "m/s2" => (Dim::Accel, 1.0, 0.0),
        "cm/s2" => (Dim::Accel, 1e-2, 0.0),
        "pa" => (Dim::Pressure, 1.0, 0.0),
        "hpa" | "mbar" => (Dim::Pressure, 100.0, 0.0),
        "kpa" => (Dim::Pressure, 1e3, 0.0),
        "mpa" => (Dim::Pressure, 1e6, 0.0),
        "bar" => (Dim::Pressure, 1e5, 0.0),
        "atm" => (Dim::Pressure, P_ATM, 0.0),
        "w/m3" => (Dim::Power, 1.0, 0.0),
        "kw/m3" => (Dim::Power, 1e3, 0.0),
        "mw/m3" => (Dim::Power, 1e6, 0.0),
        "°" | "deg" => (Dim::Angle, std::f64::consts::PI / 180.0, 0.0),
        "rad" => (Dim::Angle, 1.0, 0.0),
        "°/s" | "deg/s" => (Dim::AngSpeed, std::f64::consts::PI / 180.0, 0.0),
        "rad/s" => (Dim::AngSpeed, 1.0, 0.0),
        "tr/min" => (Dim::AngSpeed, 2.0 * std::f64::consts::PI / 60.0, 0.0),
        "tr/s" => (Dim::AngSpeed, 2.0 * std::f64::consts::PI, 0.0),
        "kg/m3" | "g/l" => (Dim::Density, 1.0, 0.0),
        "g/cm3" | "kg/l" | "kg/dm3" => (Dim::Density, 1e3, 0.0),
        _ => return None,
    })
}

fn word_start(c: char) -> bool {
    c.is_alphabetic() || matches!(c, '_' | '°' | 'µ' | '%')
}

fn word_char(c: char) -> bool {
    // (tout caractère qui commence un mot doit en faire partie : sinon la lecture n'avance plus)
    c.is_alphanumeric() || matches!(c, '_' | '/' | '²' | '³' | '°' | 'µ' | '%' | '\'' | '’')
}

fn tokenize(line: &str, ln: usize) -> Result<Vec<Tok>, ScriptError> {
    let err = |m: String| ScriptError { line: ln, msg: m };
    let cs: Vec<char> = line.chars().collect();
    let n = cs.len();
    let digit = |k: usize| k < n && cs[k].is_ascii_digit();
    let mut out: Vec<Tok> = vec![];
    let mut i = 0;
    while i < n {
        let c = cs[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '#' {
            break;
        }
        let simple = match c {
            '(' => Some(Tok::LP),
            ')' => Some(Tok::RP),
            ',' | ';' => Some(Tok::Comma),
            ':' => Some(Tok::Colon),
            '=' => Some(Tok::Eq),
            '×' => Some(Tok::W("x".into(), "×".into())),
            '≤' => Some(Tok::Cmp(Cmp::Le)),
            '≥' => Some(Tok::Cmp(Cmp::Ge)),
            _ => None,
        };
        if let Some(t) = simple {
            out.push(t);
            i += 1;
            continue;
        }
        if c == '<' || c == '>' {
            let eq = i + 1 < n && cs[i + 1] == '=';
            out.push(Tok::Cmp(match (c, eq) {
                ('<', false) => Cmp::Lt,
                ('<', true) => Cmp::Le,
                ('>', false) => Cmp::Gt,
                _ => Cmp::Ge,
            }));
            i += if eq { 2 } else { 1 };
            continue;
        }
        if matches!(c, '"' | '«' | '“') {
            let close = match c {
                '«' => '»',
                '“' => '”',
                _ => '"',
            };
            let mut j = i + 1;
            while j < n && cs[j] != close {
                j += 1;
            }
            if j >= n {
                return Err(err("guillemet non refermé".into()));
            }
            out.push(Tok::S(cs[i + 1..j].iter().collect::<String>().trim().to_string()));
            i = j + 1;
            continue;
        }
        if c == '$' {
            let name: String = cs[i + 1..].iter().take_while(|c| c.is_alphanumeric() || **c == '_').collect();
            return Err(err(format!("variable « ${name} » : il faut une ligne « balayage {name} = … »")));
        }
        let prev_value = matches!(out.last(), Some(Tok::N(_)) | Some(Tok::RP));
        let num_start = c.is_ascii_digit() || (c == '.' && digit(i + 1)) || ((c == '-' || c == '+') && !prev_value && (digit(i + 1) || (i + 1 < n && cs[i + 1] == '.' && digit(i + 2))));
        if num_start {
            let start = i;
            if c == '-' || c == '+' {
                i += 1;
            }
            while digit(i) {
                i += 1;
            }
            if i < n && cs[i] == '.' {
                i += 1;
                while digit(i) {
                    i += 1;
                }
            }
            if i < n && (cs[i] == 'e' || cs[i] == 'E') {
                let mut j = i + 1;
                if j < n && (cs[j] == '-' || cs[j] == '+') {
                    j += 1;
                }
                if digit(j) {
                    while digit(j) {
                        j += 1;
                    }
                    i = j;
                }
            }
            let s: String = cs[start..i].iter().collect();
            let v: f64 = s.parse().map_err(|_| err(format!("nombre illisible « {s} »")))?;
            out.push(Tok::N(v));
            continue;
        }
        if word_start(c) {
            // « 160x120 » : x collé entre deux nombres
            if (c == 'x' || c == 'X') && matches!(out.last(), Some(Tok::N(_))) && digit(i + 1) {
                out.push(Tok::W("x".into(), "x".into()));
                i += 1;
                continue;
            }
            let start = i;
            while i < n && word_char(cs[i]) {
                i += 1;
            }
            let s: String = cs[start..i].iter().collect();
            out.push(Tok::W(norm(&s), s));
            continue;
        }
        return Err(err(format!("caractère inattendu « {c} »")));
    }
    Ok(out)
}

type R<T> = Result<T, ScriptError>;

struct P<'a> {
    t: &'a [Tok],
    i: usize,
    ln: usize,
}

impl<'a> P<'a> {
    fn err<T>(&self, m: impl Into<String>) -> R<T> {
        Err(ScriptError { line: self.ln, msg: m.into() })
    }
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }
    fn peek_w(&self) -> Option<&str> {
        match self.peek() {
            Some(Tok::W(w, _)) => Some(w.as_str()),
            _ => None,
        }
    }
    fn done(&self) -> bool {
        self.i >= self.t.len()
    }
    fn eat_w(&mut self, w: &str) -> bool {
        if self.peek_w() == Some(w) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == Some(t) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn here(&self) -> String {
        match self.peek() {
            None => " (fin de ligne)".into(),
            Some(t) => format!(" (trouvé {})", show(t)),
        }
    }
    fn word(&mut self, what: &str) -> R<String> {
        match self.peek() {
            Some(Tok::W(w, _)) => {
                let w = w.clone();
                self.i += 1;
                Ok(w)
            }
            _ => self.err(format!("il faut {what}{}", self.here())),
        }
    }
    fn string(&mut self, what: &str) -> R<String> {
        match self.peek() {
            Some(Tok::S(s)) => {
                let s = s.clone();
                self.i += 1;
                Ok(s)
            }
            _ => self.err(format!("il faut {what} entre guillemets{}", self.here())),
        }
    }
    fn number(&mut self, what: &str) -> R<f64> {
        match self.peek() {
            Some(Tok::N(v)) => {
                let v = *v;
                self.i += 1;
                Ok(v)
            }
            _ => self.err(format!("il faut {what}{}", self.here())),
        }
    }
    fn count(&mut self, what: &str) -> R<usize> {
        let v = self.number(what)?;
        if v < 1.0 || v.fract() != 0.0 {
            return self.err(format!("{what} : il faut un entier positif (trouvé {v})"));
        }
        Ok(v as usize)
    }
    /// unité facultative après une valeur : (valeur SI, unité écrite ?)
    fn opt_unit(&mut self, dim: Dim, v: f64) -> R<(f64, bool)> {
        if let Some(Tok::W(w, orig)) = self.peek() {
            if let Some((d, f, off)) = unit_of(w) {
                if d == dim {
                    self.i += 1;
                    return Ok((v * f + off, true));
                }
                if dim != Dim::Plain {
                    return self.err(format!("unité « {orig} » : il faut ici {}", dim_name(dim)));
                }
            }
        }
        Ok((v, false))
    }
    fn quantity(&mut self, dim: Dim, what: &str) -> R<f64> {
        let v = self.number(what)?;
        Ok(self.opt_unit(dim, v)?.0)
    }
    /// affichage d'une image (phases par défaut)
    fn display(&mut self) -> R<super::mp_view::MpDisplay> {
        use super::mp_view::MpDisplay as D;
        let Some(w) = self.peek_w() else { return Ok(D::Phases) };
        let d = match w {
            "phases" => D::Phases,
            "temperature" | "t" => D::Temperature,
            "pression" | "p" => D::Pressure,
            "vitesse" => D::Velocity,
            "densite" | "rho" | "masse" => D::Density,
            "vapeur" => D::Vapor,
            "divergence" => D::Divergence,
            "vorticite" | "rotationnel" => D::Vorticity,
            "flamme" => D::Flame,
            "chaleur" => D::HeatRelease,
            o => {
                return self.err(format!(
                    "affichage « {o} » inconnu : phases, température, pression, vitesse, masse volumique, vapeur, divergence, vorticité, flamme, chaleur"
                ))
            }
        };
        self.i += 1;
        if d == D::Density {
            self.eat_w("volumique");
        }
        Ok(d)
    }
    /// seuil d'une condition, dans l'unité où le relevé est mesuré (K, Pa, m/s, m ; angles en degrés, rotations en °/s)
    fn threshold(&mut self, m: &Measure) -> R<f64> {
        let dim = match m {
            Measure::Probe { reduce: Reduce::Sum, .. } | Measure::Mass { .. } | Measure::SolidForce { .. } | Measure::Flux { .. } | Measure::Stats => {
                Dim::Plain
            }
            Measure::Time { .. } => Dim::Time,
            Measure::Probe { q, .. } => match q {
                Quantity::T => Dim::Temp,
                Quantity::P => Dim::Pressure,
                Quantity::Vx | Quantity::Vy | Quantity::Speed => Dim::Speed,
                Quantity::Rho => Dim::Density,
                Quantity::Fraction(_) => Dim::Plain,
                Quantity::HeatRelease => Dim::Power,
            },
            Measure::Body { q, .. } => match q {
                BodyQty::X | BodyQty::Y => Dim::Length,
                BodyQty::Angle => Dim::Angle,
                BodyQty::Vx | BodyQty::Vy => Dim::Speed,
                BodyQty::Rotation => Dim::AngSpeed,
                BodyQty::Fx(_) | BodyQty::Fy(_) | BodyQty::Torque(_) => Dim::Plain,
                BodyQty::Temp => Dim::Temp,
            },
        };
        let v = self.number("le seuil de la condition")?;
        let (v, unit) = self.opt_unit(dim, v)?;
        // angle et rotation écrits avec une unité : convertis en radians, à remettre en degrés comme la mesure
        Ok(if unit && matches!(dim, Dim::Angle | Dim::AngSpeed) { v.to_degrees() } else { v })
    }
    fn point(&mut self) -> R<[f64; 2]> {
        if !self.eat(&Tok::LP) {
            return self.err(format!("il faut un point (x, y){}", self.here()));
        }
        let x = self.quantity(Dim::Length, "l'abscisse x")?;
        if !self.eat(&Tok::Comma) {
            return self.err(format!("il faut une virgule entre x et y{}", self.here()));
        }
        let y = self.quantity(Dim::Length, "l'ordonnée y")?;
        if !self.eat(&Tok::RP) {
            return self.err(format!("il faut « ) » après (x, y){}", self.here()));
        }
        Ok([x, y])
    }
    /// (a, b) [unité] : chaque composante peut porter son unité, sinon celle écrite après la parenthèse
    fn vector(&mut self, dim: Dim, what: &str) -> R<[f64; 2]> {
        if !self.eat(&Tok::LP) {
            return self.err(format!("il faut {what} sous la forme (x, y){}", self.here()));
        }
        let a = self.number(&format!("la composante x de {what}"))?;
        let (a, ua) = self.opt_unit(dim, a)?;
        if !self.eat(&Tok::Comma) {
            return self.err(format!("il faut une virgule entre les deux composantes{}", self.here()));
        }
        let b = self.number(&format!("la composante y de {what}"))?;
        let (b, ub) = self.opt_unit(dim, b)?;
        if !self.eat(&Tok::RP) {
            return self.err(format!("il faut « ) »{}", self.here()));
        }
        if let Some(Tok::W(w, orig)) = self.peek() {
            if let Some((d, f, _)) = unit_of(w) {
                if d != dim {
                    return self.err(format!("unité « {orig} » : il faut ici {}", dim_name(dim)));
                }
                self.i += 1;
                return Ok([if ua { a } else { a * f }, if ub { b } else { b * f }]);
            }
        }
        Ok([a, b])
    }
    fn zone(&mut self) -> R<Zone> {
        let w = self.word("une zone (partout, disque, rect, point, trait)")?;
        match w.as_str() {
            "partout" | "tout" => Ok(Zone::Partout),
            "disque" | "cercle" => {
                let c = self.point()?;
                if !(self.eat_w("rayon") || self.eat_w("r")) {
                    return self.err(format!("il faut « rayon <r> » après le centre du disque{}", self.here()));
                }
                let r = self.quantity(Dim::Length, "le rayon")?;
                if !(r >= 0.0) {
                    return self.err("le rayon doit être positif (0 : la seule maille du centre)");
                }
                Ok(Zone::Disque { c, r })
            }
            "rect" | "rectangle" => {
                let a = self.point()?;
                self.eat_w("a");
                let b = self.point()?;
                Ok(Zone::Rect { a, b })
            }
            "point" | "en" => Ok(Zone::Point(self.point()?)),
            "trait" => {
                let mut pts = vec![self.point()?];
                while self.peek() == Some(&Tok::LP) {
                    pts.push(self.point()?);
                }
                if !(self.eat_w("rayon") || self.eat_w("r")) {
                    return self.err(format!("il faut « rayon <r> » après les points du trait{}", self.here()));
                }
                let r = self.quantity(Dim::Length, "le rayon")?;
                Ok(Zone::Trait { pts, r })
            }
            o => self.err(format!(
                "zone « {o} » inconnue : partout, disque (x, y) rayon r, rect (x1, y1) (x2, y2), point (x, y), trait (x1, y1) (x2, y2)… rayon r"
            )),
        }
    }
    /// température facultative : « à 20 °C » ou directement « 20 °C »
    fn opt_temp(&mut self) -> R<Option<f64>> {
        if self.eat_w("a") || matches!(self.peek(), Some(Tok::N(_))) {
            return Ok(Some(self.quantity(Dim::Temp, "la température")?));
        }
        Ok(None)
    }
    fn opt_duration(&mut self) -> R<Option<f64>> {
        if self.eat_w("pendant") {
            let d = self.quantity(Dim::Time, "la durée")?;
            if !(d > 0.0) {
                return self.err("la durée doit être positive");
            }
            return Ok(Some(d));
        }
        Ok(None)
    }
    fn end(&self) -> R<()> {
        if self.done() {
            return Ok(());
        }
        let rest: Vec<String> = self.t[self.i..].iter().map(show).collect();
        let hint = if self.peek() == Some(&Tok::Comma) && matches!(self.t.get(self.i + 1), Some(Tok::N(_))) {
            " — virgule décimale ? les nombres s'écrivent avec un point (2.5)"
        } else {
            ""
        };
        self.err(format!("texte en trop : {}{hint}", rest.join(" ")))
    }
    fn quantity_name(&mut self) -> R<Quantity> {
        let w = self.word("une grandeur (T, p, vx, vy, vitesse, rho, fraction \"fluide\", chaleur)")?;
        Ok(match w.as_str() {
            "t" | "temperature" => Quantity::T,
            "p" | "pression" => Quantity::P,
            "vx" => Quantity::Vx,
            "vy" => Quantity::Vy,
            "vitesse" | "v" => Quantity::Speed,
            "rho" | "densite" => Quantity::Rho,
            "fraction" => Quantity::Fraction(self.string("le fluide")?),
            "chaleur" | "puissance" => Quantity::HeatRelease,
            o => return self.err(format!("grandeur « {o} » inconnue : T, p, vx, vy, vitesse, rho, fraction \"fluide\", chaleur")),
        })
    }
    fn measure(&mut self) -> R<Measure> {
        let w = self.word("une mesure (stats, T en (x, y), moyenne T dans <zone>, masse \"fluide\", corps 1 y, force x sur <zone>, débit à travers ligne …)")?;
        let reduce = match w.as_str() {
            "moyenne" => Some(Reduce::Mean),
            "max" | "maximum" => Some(Reduce::Max),
            "min" | "minimum" => Some(Reduce::Min),
            "somme" | "integrale" => Some(Reduce::Sum),
            _ => None,
        };
        let mut m = match (w.as_str(), reduce) {
            ("stats" | "statistiques", _) => Measure::Stats,
            ("temps", _) => Measure::Time { label: None },
            ("masse", _) => {
                let fluid = self.string("le fluide")?;
                let zone = if self.eat_w("dans") { self.zone()? } else { Zone::Partout };
                Measure::Mass { fluid, zone, label: None }
            }
            ("force", _) => {
                let axis = match self.word("la composante de la force (x ou y)")?.as_str() {
                    "x" => 0,
                    "y" => 1,
                    o => return self.err(format!("composante « {o} » de la force : x ou y")),
                };
                if !self.eat_w("sur") {
                    return self.err(format!("il faut « force x | y sur <zone> »{}", self.here()));
                }
                Measure::SolidForce { axis, zone: self.zone()?, label: None }
            }
            ("debit", _) => {
                // débit [massique] ["fluide"] à travers ligne (x1, y1) (x2, y2)
                let mass = self.eat_w("massique");
                let fluid = if matches!(self.peek(), Some(Tok::S(_))) { Some(self.string("le fluide")?) } else { None };
                if !(self.eat_w("a") && self.eat_w("travers") && self.eat_w("ligne")) {
                    return self.err(format!("il faut « débit [massique] [\"fluide\"] à travers ligne (x1, y1) (x2, y2) »{}", self.here()));
                }
                let (a, b) = (self.point()?, self.point()?);
                if a[0] != b[0] && a[1] != b[1] {
                    return self.err("une ligne de débit est horizontale ou verticale : elle suit les faces des mailles (même x ou même y aux deux bouts)");
                }
                Measure::Flux { fluid, mass, a, b, label: None }
            }
            ("corps", _) => {
                let n = self.count("le numéro du corps (1 pour le premier posé)")?;
                let w = self.word("la grandeur du corps (x, y, angle, vx, vy, rotation, fx, fy, couple [contact | loi | total], température)")?;
                let q = match w.as_str() {
                    "x" => BodyQty::X,
                    "y" => BodyQty::Y,
                    "angle" => BodyQty::Angle,
                    "vx" => BodyQty::Vx,
                    "vy" => BodyQty::Vy,
                    "rotation" => BodyQty::Rotation,
                    "fx" | "fy" | "couple" => {
                        // force du fluide (par défaut), des contacts, d'une loi en formule, ou totale (avec le poids)
                        use super::mp_experiment::ForcePart as F;
                        let part = if self.eat_w("contact") {
                            F::Contact
                        } else if self.eat_w("loi") {
                            F::Law
                        } else if self.eat_w("total") || self.eat_w("totale") {
                            F::Total
                        } else {
                            self.eat_w("fluide");
                            F::Fluid
                        };
                        match w.as_str() {
                            "fx" => BodyQty::Fx(part),
                            "fy" => BodyQty::Fy(part),
                            _ => BodyQty::Torque(part),
                        }
                    }
                    "t" | "temperature" => BodyQty::Temp,
                    o => return self.err(format!("grandeur « {o} » d'un corps : x, y, angle, vx, vy, rotation, fx, fy, couple, température")),
                };
                Measure::Body { index: n - 1, q, label: None }
            }
            (_, Some(reduce)) => {
                let q = self.quantity_name()?;
                if !self.eat_w("dans") {
                    return self.err(format!("il faut « dans <zone> »{}", self.here()));
                }
                let zone = self.zone()?;
                Measure::Probe { q, reduce, zone, label: None }
            }
            _ => {
                self.i -= 1;
                let q = self.quantity_name()?;
                if !self.eat_w("en") {
                    return self.err(format!("il faut « en (x, y) » (ou « moyenne … dans <zone> »){}", self.here()));
                }
                Measure::Probe { q, reduce: Reduce::At, zone: Zone::Point(self.point()?), label: None }
            }
        };
        if self.eat_w("comme") {
            let l = self.string("le nom de la colonne")?;
            match &mut m {
                Measure::Probe { label, .. }
                | Measure::Mass { label, .. }
                | Measure::Body { label, .. }
                | Measure::SolidForce { label, .. }
                | Measure::Flux { label, .. }
                | Measure::Time { label } => *label = Some(l),
                Measure::Stats => return self.err("« comme » ne s'applique pas à « stats »"),
            }
        }
        Ok(m)
    }
}

/// « : <action> » d'une action datée ou répétée : le reste de la ligne, lu comme une commande d'une seule action
/// <relevé> < | ≤ | > | ≥ <seuil> [et | ou <relevé> … <seuil>]… : les comparaisons, et leur lien (« ou » : vrai si l'une l'est)
fn condition(p: &mut P) -> R<(Vec<Clause>, bool)> {
    let mut clauses = vec![];
    let mut any: Option<bool> = None;
    loop {
        let measure = p.measure()?;
        if measure == Measure::Stats {
            return p.err("une condition porte sur un relevé (T en (x, y), max T dans <zone>, corps 1 y…), pas sur « stats »");
        }
        let cmp = match p.peek() {
            Some(Tok::Cmp(c)) => *c,
            _ => return p.err(format!("il faut une comparaison après le relevé : <, <=, >, >= (ou ≤, ≥){}", p.here())),
        };
        p.i += 1;
        let value = p.threshold(&measure)?;
        clauses.push(Clause { measure, cmp, value });
        let link = if p.eat_w("et") {
            false
        } else if p.eat_w("ou") {
            true
        } else {
            break;
        };
        if any.is_some_and(|a| a != link) {
            return p.err("« et » et « ou » ne se mélangent pas dans une même condition (pas de parenthèses)");
        }
        any = Some(link);
    }
    Ok((clauses, any.unwrap_or(false)))
}

/// [au plus <durée>] après une condition
fn opt_limit(p: &mut P) -> R<Option<f64>> {
    if !p.eat_w("au") {
        return Ok(None);
    }
    if !p.eat_w("plus") {
        return p.err(format!("il faut « au plus <durée> »{}", p.here()));
    }
    let d = p.quantity(Dim::Time, "la durée maximale")?;
    if !(d > 0.0) {
        return p.err("la durée maximale doit être positive");
    }
    Ok(Some(d))
}

/// action après « : » d'une action datée, répétée ou d'une règle « quand » (`pause_ok` : une pause est permise, pour « quand »)
fn timed_action(p: &mut P, ln: usize, pause_ok: bool) -> R<Action> {
    if !p.eat(&Tok::Colon) {
        return p.err(format!("il faut « : » puis l'action{}", p.here()));
    }
    if p.done() {
        return p.err("il manque l'action après « : »");
    }
    let mut inner = Protocol::default();
    let mut q = P { t: &p.t[p.i..], i: 0, ln };
    parse_line(&mut q, &mut inner)?;
    p.i = p.t.len();
    let only = inner.measures.is_empty() && inner.period.is_none() && inner.images.is_empty() && inner.start.is_none();
    match inner.steps.pop() {
        Some((Step::Do(a), _))
            if only && inner.steps.is_empty() && !matches!(a, Action::Periodic { .. } | Action::When { .. }) && (pause_ok || a != Action::Pause) =>
        {
            Ok(a)
        }
        _ => p.err(if pause_ok {
            "une règle « quand » porte sur une seule action (remplir, chauffer, vanne, bord, corps, marque, pause…), pas sur simuler, mesurer ni une action datée, répétée ou une autre règle"
        } else {
            "une action datée ou répétée porte sur une seule action (remplir, chauffer, vanne, bord, corps…), pas sur simuler, mesurer, pause ni une autre action datée, répétée ou une règle « quand »"
        }),
    }
}

fn push(proto: &mut Protocol, a: Action, ln: usize) {
    proto.steps.push((Step::Do(a), ln));
}

fn parse_line(p: &mut P, proto: &mut Protocol) -> R<()> {
    let ln = p.ln;
    let cmd = p.word("une commande")?;
    match cmd.as_str() {
        "depart" => {
            if proto.start.is_some() {
                return p.err("un seul « départ » par protocole");
            }
            if !proto.steps.is_empty() {
                return p.err("« départ » doit précéder les actions");
            }
            let kind = p.word("« scénario » ou « état »")?;
            let start = match kind.as_str() {
                "scenario" => {
                    let name = match p.peek() {
                        Some(Tok::N(_)) => format!("{}", p.count("le numéro du scénario")? as usize),
                        _ => p.string("le nom du scénario")?,
                    };
                    let (mut nx, mut ny, mut dx, mut g) = (80usize, 60usize, 0.01f64, 9.81f64);
                    while !p.done() {
                        let opt = p.word("une option (grille, maille, gravité)")?;
                        match opt.as_str() {
                            "grille" => {
                                nx = p.count("le nombre de mailles en x")?;
                                if !p.eat_w("x") {
                                    return p.err(format!("il faut « x » entre les deux nombres de mailles (grille 160 x 120){}", p.here()));
                                }
                                ny = p.count("le nombre de mailles en y")?;
                            }
                            "maille" => dx = p.quantity(Dim::Length, "la taille de maille")?,
                            "gravite" => g = p.quantity(Dim::Accel, "la gravité")?,
                            o => return p.err(format!("option « {o} » inconnue après « départ scénario » : grille, maille, gravité")),
                        }
                    }
                    if !(dx > 0.0) {
                        return p.err("la taille de maille doit être positive");
                    }
                    Start::Scenario { name, nx, ny, dx, g }
                }
                "etat" => Start::State(p.string("le fichier d'état complet (.mps)")?),
                o => return p.err(format!("départ « {o} » : il faut « départ scénario \"…\" » ou « départ état \"fichier.mps\" »")),
            };
            p.end()?;
            proto.start = Some((start, ln));
        }
        "materiaux" => {
            proto.materials = Some(p.string("le fichier de matériaux (.json, exporté par l'éditeur)")?);
            p.end()?;
        }
        "toutes" | "chaque" => {
            // toutes les <T> [à partir de <t0>] [jusqu'à <t1>] : <action> — répétée en parallèle du déroulement
            if cmd == "toutes" && !p.eat_w("les") {
                return p.err(format!("il faut « toutes les <période> : <action> »{}", p.here()));
            }
            let period = p.quantity(Dim::Time, "la période de répétition")?;
            if !(period > 0.0) {
                return p.err("la période de répétition doit être positive");
            }
            let (mut start, mut end) = (None, None);
            loop {
                if p.eat_w("a") {
                    if !(p.eat_w("partir") && p.eat_w("de")) {
                        return p.err(format!("il faut « à partir de <instant> »{}", p.here()));
                    }
                    start = Some(p.quantity(Dim::Time, "l'instant de la première fois")?);
                } else if p.eat_w("jusqua") || p.eat_w("jusque") {
                    end = Some(p.quantity(Dim::Time, "l'instant de la dernière fois")?);
                } else {
                    break;
                }
            }
            let action = timed_action(p, ln, false)?;
            push(proto, Action::Periodic { period, start, end, action: Box::new(action) }, ln);
        }
        "quand" => {
            // quand <condition> : <action> — faite chaque fois que la condition devient vraie (relevée après chaque pas)
            let (clauses, any) = condition(p)?;
            let action = timed_action(p, ln, true)?;
            push(proto, Action::When { clauses, any, action: Box::new(action) }, ln);
        }
        "a" => {
            // à <t> : <action> — faite une fois, à l'instant t pile, en parallèle du déroulement (geste enregistré par la fenêtre…)
            let t = p.quantity(Dim::Time, "l'instant de l'action")?;
            let action = timed_action(p, ln, false)?;
            push(proto, Action::Periodic { period: f64::INFINITY, start: Some(t), end: Some(t), action: Box::new(action) }, ln);
        }
        "calcul" => {
            let w = p.word("« cpu » ou « gpu »")?;
            proto.gpu = Some(match w.as_str() {
                "gpu" => true,
                "cpu" => false,
                o => return p.err(format!("calcul « {o} » : « cpu » ou « gpu »")),
            });
            p.end()?;
        }
        "images" | "image" => {
            // images toutes les <durée> [affichage] : une série d'images PNG (plusieurs séries possibles)
            if p.eat_w("toutes") {
                if !p.eat_w("les") {
                    return p.err("il faut « images toutes les <durée> [affichage] »");
                }
            } else if !p.eat_w("chaque") {
                return p.err(format!("il faut « images toutes les <durée> [affichage] »{}", p.here()));
            }
            let period = p.quantity(Dim::Time, "la période des images")?;
            if !(period > 0.0) {
                return p.err("la période des images doit être positive");
            }
            let display = p.display()?;
            p.end()?;
            if proto.images.iter().any(|(_, d, _)| *d == display) {
                return p.err("une seule série d'images par affichage");
            }
            proto.images.push((period, display, ln));
        }
        "mesurer" | "mesure" => {
            if p.eat_w("toutes") {
                if !p.eat_w("les") {
                    return p.err("il faut « toutes les <durée> »");
                }
            } else if !p.eat_w("chaque") {
                return p.err(format!("il faut « toutes les <durée> : … »{}", p.here()));
            }
            let period = p.quantity(Dim::Time, "la période des mesures")?;
            if !(period > 0.0) {
                return p.err("la période des mesures doit être positive");
            }
            if let Some(old) = proto.period {
                if (old - period).abs() > 1e-12 * old.max(1.0) {
                    return p.err(format!("une seule période de mesure par protocole (déjà {old} s)"));
                }
            }
            proto.period = Some(period);
            p.eat(&Tok::Colon);
            loop {
                let m = p.measure()?;
                proto.measures.push((m, ln));
                if !p.eat(&Tok::Comma) {
                    break;
                }
            }
            p.end()?;
        }
        "simuler" | "avancer" => {
            let step = if (p.eat_w("jusqua") || p.eat_w("jusque")) && p.eat_w("ce") {
                // simuler jusqu'à ce que <relevé> < | ≤ | > | ≥ <seuil> [et | ou <relevé> … <seuil>]… [au plus <durée>]
                if !p.eat_w("que") {
                    return p.err(format!("il faut « jusqu'à ce que <relevé> < <valeur> »{}", p.here()));
                }
                let (clauses, any) = condition(p)?;
                Step::UntilCond { clauses, any, limit: opt_limit(p)? }
            } else if matches!(p.t.get(p.i.wrapping_sub(1)), Some(Tok::W(w, _)) if w == "jusqua" || w == "jusque") {
                Step::Until(p.quantity(Dim::Time, "le temps à atteindre")?)
            } else {
                let d = p.quantity(Dim::Time, "la durée à simuler")?;
                if !(d >= 0.0) {
                    return p.err("la durée doit être positive");
                }
                Step::Simulate(d)
            };
            p.end()?;
            proto.steps.push((step, ln));
        }
        "attendre" => {
            // attendre <durée> | jusqu'à <instant> | [que | jusqu'à ce que] <condition> [au plus <durée>] — comme « simuler »
            let step = if matches!(p.peek(), Some(Tok::N(_))) {
                let d = p.quantity(Dim::Time, "la durée à attendre")?;
                if !(d >= 0.0) {
                    return p.err("la durée doit être positive");
                }
                Step::Simulate(d)
            } else if (p.eat_w("jusqua") || p.eat_w("jusque")) && !p.eat_w("ce") {
                Step::Until(p.quantity(Dim::Time, "l'instant à atteindre")?)
            } else {
                p.eat_w("que");
                let (clauses, any) = condition(p)?;
                Step::UntilCond { clauses, any, limit: opt_limit(p)? }
            };
            p.end()?;
            proto.steps.push((step, ln));
        }
        "tant" => {
            // tant que <condition> [au plus <durée>] … fin — boucle (les liens entre « tant que » et « fin » sont posés par `parse`)
            if !p.eat_w("que") {
                return p.err(format!("il faut « tant que <condition> »{}", p.here()));
            }
            let (clauses, any) = condition(p)?;
            let limit = opt_limit(p)?;
            p.end()?;
            proto.steps.push((Step::While { clauses, any, limit, end: usize::MAX }, ln));
        }
        "fin" => {
            // fin [tant que] : ferme la boucle ouverte la plus récente
            if p.eat_w("tant") && !p.eat_w("que") {
                return p.err("il faut « fin » ou « fin tant que »");
            }
            p.end()?;
            proto.steps.push((Step::EndWhile { start: usize::MAX }, ln));
        }
        "remplir" => {
            let zone = p.zone()?;
            if !p.eat_w("avec") {
                return p.err(format!("il faut « avec \"fluide\" »{}", p.here()));
            }
            let fluid = p.string("le fluide ou le mélange")?;
            let temp = p.opt_temp()?;
            p.end()?;
            push(proto, Action::Fill { zone, fluid, temp }, ln);
        }
        "temperature" => {
            let zone = p.zone()?;
            p.eat_w("a");
            let temp = p.quantity(Dim::Temp, "la température")?;
            p.end()?;
            push(proto, Action::SetTemp { zone, temp }, ln);
        }
        "solide" => {
            let zone = p.zone()?;
            let material = p.string("le matériau solide")?;
            let (mut thermal, mut temp) = (ThermalSpec::Passive, None);
            while !p.done() {
                let w = p.word("une option (isotherme, chauffant, à)")?;
                match w.as_str() {
                    "isotherme" => {
                        p.eat_w("a");
                        thermal = ThermalSpec::Isothermal(p.quantity(Dim::Temp, "la température de l'isotherme")?);
                    }
                    "chauffant" => thermal = ThermalSpec::Heater(p.quantity(Dim::Power, "la puissance de chauffe")?),
                    "a" => temp = Some(p.quantity(Dim::Temp, "la température")?),
                    o => return p.err(format!("option « {o} » inconnue après « solide » : isotherme <T>, chauffant <W/m³>, à <T>")),
                }
            }
            push(proto, Action::Solid { zone, material, thermal, temp }, ln);
        }
        "effacer" => {
            let zone = p.zone()?;
            p.end()?;
            push(proto, Action::Clear { zone }, ln);
        }
        "corps" => {
            let shape = match p.word("la forme du corps (disque, rect, polygone)")?.as_str() {
                "disque" | "cercle" => {
                    let c = p.point()?;
                    if !(p.eat_w("rayon") || p.eat_w("r")) {
                        return p.err(format!("il faut « rayon <r> » après le centre du disque{}", p.here()));
                    }
                    let r = p.quantity(Dim::Length, "le rayon")?;
                    if !(r > 0.0) {
                        return p.err("le rayon d'un corps doit être positif");
                    }
                    BodySpec::Disc { c, r }
                }
                "rect" | "rectangle" => {
                    let a = p.point()?;
                    p.eat_w("a");
                    let b = p.point()?;
                    BodySpec::Rect { a, b }
                }
                "polygone" => {
                    let mut pts = vec![p.point()?];
                    while p.peek() == Some(&Tok::LP) {
                        pts.push(p.point()?);
                    }
                    if pts.len() < 3 {
                        return p.err("un polygone a au moins 3 sommets");
                    }
                    BodySpec::Polygon(pts)
                }
                o => return p.err(format!("forme « {o} » : disque (x, y) rayon r, rect (x1, y1) (x2, y2) ou polygone (x1, y1) (x2, y2) (x3, y3)…")),
            };
            let material = p.string("le matériau solide")?;
            let mut angle = 0.0;
            let mut motion = BodyMotion::default();
            let mut osc = false;
            let mut fixed_kw = false;
            let mut temp = None;
            // formule(s) d'une loi de mouvement ou de force, vérifiées sur les variables de `LAW_VARS`
            let law_formulas = |p: &mut P, n: usize, what: &str| -> R<Vec<String>> {
                if !p.eat_w("formule") {
                    return p.err(format!("il faut « {what} formule \"…\"{} »{}", if n == 2 { " \"…\"" } else { "" }, p.here()));
                }
                let mut out = vec![];
                for k in 0..n {
                    let f = p.string(&format!("la formule {} de {what}", ["x", "y"][k.min(1)]))?;
                    if let Err(e) = super::mp_fields::Formula::parse_vars(&f, &super::mp_bodies::LAW_VARS) {
                        return p.err(format!("{what} : {e}"));
                    }
                    out.push(f);
                }
                Ok(out)
            };
            let set_lin = |p: &mut P, motion: &mut BodyMotion, l: super::mp_bodies::LinLaw| -> R<()> {
                if motion.lin_law.is_some() {
                    return p.err("une seule loi de translation par corps : position, vitesse ou accélération en formule");
                }
                motion.lin_law = Some(l);
                Ok(())
            };
            let set_rot = |p: &mut P, motion: &mut BodyMotion, l: super::mp_bodies::RotLaw| -> R<()> {
                if motion.rot_law.is_some() {
                    return p.err("une seule loi de rotation par corps : angle ou rotation en formule");
                }
                motion.rot_law = Some(l);
                Ok(())
            };
            // vitesse angulaire : degrés par seconde sans unité
            let ang_speed = |p: &mut P| -> R<f64> {
                let v = p.number("la vitesse de rotation")?;
                let (v, unit) = p.opt_unit(Dim::AngSpeed, v)?;
                Ok(if unit { v } else { v.to_radians() })
            };
            while !p.done() {
                match p.word("une option (angle, fixe, libre, bloqué, vitesse, rotation, oscillation, période, pivot, masse volumique, rebond, frottement, à T)")?.as_str() {
                    "a" => temp = Some(p.quantity(Dim::Temp, "la température du corps")?),
                    "angle" if p.peek_w() == Some("formule") => {
                        let f = law_formulas(p, 1, "angle")?;
                        set_rot(p, &mut motion, super::mp_bodies::RotLaw::Angle(f[0].clone()))?;
                    }
                    "angle" => {
                        let v = p.number("l'angle")?;
                        let (v, unit) = p.opt_unit(Dim::Angle, v)?;
                        angle = if unit { v } else { v.to_radians() };
                    }
                    "position" => {
                        let f = law_formulas(p, 2, "position")?;
                        set_lin(p, &mut motion, super::mp_bodies::LinLaw::Position([f[0].clone(), f[1].clone()]))?;
                    }
                    "acceleration" => {
                        let f = law_formulas(p, 2, "accélération")?;
                        set_lin(p, &mut motion, super::mp_bodies::LinLaw::Acceleration([f[0].clone(), f[1].clone()]))?;
                    }
                    "force" => {
                        let f = law_formulas(p, 2, "force")?;
                        motion.force = Some([f[0].clone(), f[1].clone()]);
                    }
                    "couple" => {
                        let f = law_formulas(p, 1, "couple")?;
                        motion.torque = Some(f[0].clone());
                    }
                    "fixe" => fixed_kw = true,
                    "libre" => motion.free = true,
                    "bloque" => {
                        let mut any = false;
                        loop {
                            // (« rotation » suivi d'un nombre est l'option de vitesse de rotation, pas un axe bloqué)
                            let rotation_value = matches!(p.t.get(p.i + 1), Some(Tok::N(_)));
                            match p.peek_w() {
                                Some("x") => motion.locked[0] = true,
                                Some("y") => motion.locked[1] = true,
                                Some("rotation") if !rotation_value => motion.locked[2] = true,
                                Some("et") if any => {}
                                _ => break,
                            }
                            p.i += 1;
                            any = true;
                        }
                        if !any {
                            return p.err(format!("il faut l'axe bloqué après « bloqué » : x, y ou rotation{}", p.here()));
                        }
                    }
                    "masse" => {
                        if !p.eat_w("volumique") {
                            return p.err(format!("il faut « masse volumique <ρ> »{}", p.here()));
                        }
                        let r = p.quantity(Dim::Density, "la masse volumique")?;
                        if !(r > 0.0) {
                            return p.err("la masse volumique doit être positive");
                        }
                        motion.rho = Some(r);
                    }
                    "rebond" => {
                        let e = p.quantity(Dim::Plain, "le coefficient de restitution")?;
                        if !(0.0..=1.0).contains(&e) {
                            return p.err("le coefficient de restitution est entre 0 (choc mou) et 1 (rebond parfait)");
                        }
                        motion.restitution = Some(e);
                    }
                    "frottement" => {
                        let m = p.quantity(Dim::Plain, "le coefficient de frottement")?;
                        if !(m >= 0.0) {
                            return p.err("le coefficient de frottement doit être positif ou nul");
                        }
                        motion.friction = Some(m);
                    }
                    "vitesse" if p.peek_w() == Some("formule") => {
                        let f = law_formulas(p, 2, "vitesse")?;
                        set_lin(p, &mut motion, super::mp_bodies::LinLaw::Velocity([f[0].clone(), f[1].clone()]))?;
                    }
                    "rotation" if p.peek_w() == Some("formule") => {
                        let f = law_formulas(p, 1, "rotation")?;
                        set_rot(p, &mut motion, super::mp_bodies::RotLaw::Rate(f[0].clone()))?;
                    }
                    "vitesse" => motion.v = p.vector(Dim::Speed, "la vitesse du corps")?,
                    "rotation" => motion.omega = ang_speed(p)?,
                    "oscillation" => {
                        if p.eat_w("rotation") {
                            motion.osc_omega = ang_speed(p)?;
                        } else {
                            motion.osc_v = p.vector(Dim::Speed, "l'amplitude de la vitesse")?;
                        }
                        osc = true;
                    }
                    "periode" => {
                        motion.period = p.quantity(Dim::Time, "la période")?;
                        if !(motion.period > 0.0) {
                            return p.err("la période doit être positive");
                        }
                    }
                    "pivot" => motion.pivot = Some(p.point()?),
                    o => {
                        return p.err(format!(
                            "option « {o} » inconnue après « corps » : angle <a>, fixe, libre, bloqué x | y | rotation, vitesse (vx, vy), \
                             rotation <ω>, oscillation (vx, vy) | oscillation rotation <ω> avec période <T>, pivot (x, y), masse volumique <ρ>, \
                             rebond <e>, frottement <μ>, à <T> ; lois en formules : position | vitesse | accélération formule \"…\" \"…\", \
                             angle | rotation formule \"…\", force formule \"…\" \"…\", couple formule \"…\""
                        ))
                    }
                }
            }
            if motion.free && fixed_kw {
                return p.err("un corps est « fixe » ou « libre », pas les deux");
            }
            let laws = motion.lin_law.is_some() || motion.rot_law.is_some();
            if motion.free && laws {
                return p.err("un corps libre suit l'écoulement : pas de loi de mouvement imposée (« force formule » et « couple formule » s'ajoutent à un corps libre)");
            }
            if (motion.force.is_some() || motion.torque.is_some()) && !motion.free {
                return p.err("« force formule » et « couple formule » s'appliquent à un corps « libre »");
            }
            if laws && osc {
                return p.err("une loi en formule ne se combine pas avec une oscillation (la formule peut osciller)");
            }
            use super::mp_bodies::LinLaw;
            if matches!(motion.lin_law, Some(LinLaw::Position(_)) | Some(LinLaw::Velocity(_))) && motion.v != [0.0, 0.0] {
                return p.err("deux translations : « vitesse (vx, vy) » et la loi de position ou de vitesse (avec « accélération formule », elle est la vitesse de départ)");
            }
            if motion.rot_law.is_some() && motion.omega != 0.0 {
                return p.err("deux rotations : « rotation ω » et la loi d'angle ou de rotation");
            }
            if motion.pivot.is_some() && motion.lin_law.is_some() {
                return p.err("sur un pivot, seule la rotation compte : pas de loi de translation");
            }
            if motion.free && osc {
                return p.err("un corps libre suit l'écoulement : pas d'oscillation imposée (« vitesse » et « rotation » donnent sa vitesse de départ)");
            }
            if motion.locked.iter().any(|&l| l) && !motion.free {
                return p.err("« bloqué » ne s'applique qu'à un corps « libre »");
            }
            if fixed_kw && !motion.is_fixed() {
                return p.err("un corps « fixe » n'a ni vitesse, ni rotation, ni oscillation");
            }
            if osc && !(motion.period > 0.0) {
                return p.err("une oscillation demande sa « période <T> »");
            }
            if motion.period > 0.0 && !osc {
                return p.err("« période » sans « oscillation »");
            }
            if motion.pivot.is_some() && (motion.v != [0.0, 0.0] || motion.osc_v != [0.0, 0.0]) {
                return p.err("sur un pivot, seul le mouvement de rotation compte (rotation, oscillation rotation)");
            }
            push(proto, Action::Body { shape, material, angle, motion, temp }, ln);
        }
        "chauffer" | "refroidir" => {
            let zone = p.zone()?;
            p.eat_w("a");
            let temp = p.quantity(Dim::Temp, "la température visée")?;
            let duration = p.opt_duration()?;
            p.end()?;
            push(proto, Action::Heat { zone, temp, duration }, ln);
        }
        "accelerer" => {
            let zone = p.zone()?;
            p.eat_w("de");
            let a = p.vector(Dim::Accel, "l'accélération")?;
            let duration = p.opt_duration()?;
            p.end()?;
            push(proto, Action::Accelerate { zone, a, duration }, ln);
        }
        "vitesse" => {
            let zone = p.zone()?;
            p.eat_w("a");
            let v = p.vector(Dim::Speed, "la vitesse")?;
            p.end()?;
            push(proto, Action::Velocity { zone, v }, ln);
        }
        "allumer" => {
            let zone = p.zone()?;
            let temp = p.opt_temp()?.unwrap_or(T_IGNITION);
            p.end()?;
            push(proto, Action::Ignite { zone, temp }, ln);
        }
        "puits" => {
            let zone = p.zone()?;
            let rate = if p.done() { 0.2 } else { p.quantity(Dim::Speed, "la vitesse d'aspiration")? };
            p.end()?;
            push(proto, Action::Sink { zone, rate }, ln);
        }
        "emetteur" => {
            let zone = p.zone()?;
            let fluid = p.string("le fluide émis")?;
            let v = p.vector(Dim::Speed, "la vitesse d'éjection")?;
            let temp = p.opt_temp()?;
            p.end()?;
            push(proto, Action::Emitter { zone, fluid, v, temp }, ln);
        }
        "pompe" | "compresseur" => {
            // pompe <zone> (vx, vy) : vitesse imposée sur les faces de la zone ; (0, 0) retire la pompe
            let zone = p.zone()?;
            let v = p.vector(Dim::Speed, "la vitesse de la pompe")?;
            p.end()?;
            push(proto, Action::Pump { zone, v }, ln);
        }
        "teleporteur" => {
            // téléporteur <zone> vers (x, y) [v m/s] [rayon r] : aspire à v (0.5 m/s par défaut), rejette le même débit autour de (x, y)
            let zone = p.zone()?;
            if !p.eat_w("vers") {
                return p.err(format!("il faut « vers (x, y) », la destination du téléporteur{}", p.here()));
            }
            let dest = p.point()?;
            let (mut speed, mut radius) = (0.5, 0.0);
            while !p.done() {
                if p.eat_w("rayon") {
                    radius = p.quantity(Dim::Length, "le rayon de la destination")?;
                    if !(radius >= 0.0) {
                        return p.err("le rayon de la destination est positif (0 : la seule maille de destination)");
                    }
                } else {
                    speed = p.quantity(Dim::Speed, "la vitesse d'aspiration (ou « rayon r »)")?;
                }
            }
            push(proto, Action::Teleporter { zone, dest, speed, radius }, ln);
        }
        "vanne" => {
            // vanne <zone> [fermée | ouverte [fraction | n %]] — vanne à guillotine, fermée par défaut
            let zone = p.zone()?;
            let open = if p.eat_w("fermee") || p.done() {
                0.0
            } else if p.eat_w("ouverte") {
                if p.done() {
                    1.0
                } else {
                    let v = p.number("l'ouverture (fraction de 0 à 1, ou pourcentage suivi de %)")?;
                    let v = if p.eat_w("%") { v / 100.0 } else { v };
                    if !(0.0..=1.0).contains(&v) {
                        return p.err("l'ouverture d'une vanne est entre 0 (fermée) et 1 (ouverte), ou entre 0 et 100 %");
                    }
                    v
                }
            } else {
                return p.err(format!("après la zone de la vanne : « fermée » ou « ouverte [fraction] »{}", p.here()));
            };
            p.end()?;
            push(proto, Action::Valve { zone, open }, ln);
        }
        "grille" => {
            let zone = p.zone()?;
            let (mut hole, mut open) = (0.01, 0.6);
            while !p.done() {
                let w = p.word("une option (trous, ouverte)")?;
                match w.as_str() {
                    "trous" => hole = p.quantity(Dim::Length, "la taille des trous")?,
                    "ouverte" => {
                        open = p.number("la fraction ouverte")?;
                        if !(open > 0.0 && open < 1.0) {
                            return p.err("la fraction ouverte est entre 0 et 1");
                        }
                    }
                    o => return p.err(format!("option « {o} » inconnue après « grille » : trous <d>, ouverte <f>")),
                }
            }
            push(proto, Action::Grille { zone, hole, open }, ln);
        }
        "bord" => {
            let side = match p.word("le côté (gauche, droite, bas, haut)")?.as_str() {
                "gauche" => 0,
                "droite" => 1,
                "bas" => 2,
                "haut" => 3,
                o => return p.err(format!("bord « {o} » : gauche, droite, bas ou haut")),
            };
            let spec = match p.word("mur, ouvert ou entrée")?.as_str() {
                "mur" | "paroi" => BoundarySpec::Wall,
                // sortie (vx, vy) : vitesse imposée vers l'extérieur ; « sortie » seule ou suivie d'une pression : bord ouvert
                "sortie" if matches!(p.peek(), Some(Tok::LP)) => {
                    let v = p.vector(Dim::Speed, "la vitesse de sortie")?;
                    let out = match side {
                        0 => -v[0],
                        1 => v[0],
                        2 => -v[1],
                        _ => v[1],
                    };
                    if !(out > 0.0) {
                        return p.err("la vitesse de sortie est dirigée vers l'extérieur du bord (vers la gauche pour le bord gauche, vers le bas pour le bas…)");
                    }
                    BoundarySpec::Outlet { v }
                }
                "ouvert" | "sortie" => BoundarySpec::Open(if p.done() { P_ATM } else { p.quantity(Dim::Pressure, "la pression du bord")? }),
                "entree" => {
                    let v = p.vector(Dim::Speed, "la vitesse d'entrée")?;
                    let (mut temp, mut fluid) = (None, None);
                    while !p.done() {
                        if matches!(p.peek(), Some(Tok::S(_))) {
                            fluid = Some(p.string("le fluide")?);
                        } else {
                            temp = p.opt_temp()?;
                            if temp.is_none() {
                                return p.err(format!("après « entrée (vx, vy) » : à <T> et \"fluide\"{}", p.here()));
                            }
                        }
                    }
                    BoundarySpec::Inlet { v, temp, fluid }
                }
                o => return p.err(format!("bord « {o} » : mur, ouvert [pression], entrée (vx, vy) [à T] [\"fluide\"] ou sortie (vx, vy)")),
            };
            p.end()?;
            push(proto, Action::Boundary { side, spec }, ln);
        }
        "champ" => {
            let qty = match p.word("la grandeur du champ (T, vx, vy, fraction, solide)")?.as_str() {
                "t" | "temperature" => FieldQty::T,
                "vx" => FieldQty::Vx,
                "vy" => FieldQty::Vy,
                "fraction" => FieldQty::Fraction(p.string("le fluide")?),
                "solide" => FieldQty::Solid(p.string("le matériau solide")?),
                o => return p.err(format!("champ « {o} » : T, vx, vy, fraction \"fluide\" ou solide \"matériau\"")),
            };
            let dim = match qty {
                FieldQty::T => Dim::Temp,
                FieldQty::Vx | FieldQty::Vy => Dim::Speed,
                _ => Dim::Plain,
            };
            let mut source = match p.word("« fichier \"…\" » ou « formule \"…\" »")?.as_str() {
                "fichier" => FieldSource::File { path: p.string("le fichier (CSV ou PNG)")?, top_origin: true, range: None },
                "formule" => FieldSource::Formula(p.string("la formule en x, y")?),
                o => return p.err(format!("source « {o} » : fichier \"…\" ou formule \"…\"")),
            };
            let (mut zone, mut temp) = (Zone::Partout, None);
            while !p.done() {
                let w = p.word("une option (dans, origine, valeurs, à)")?;
                match w.as_str() {
                    "dans" => zone = p.zone()?,
                    "origine" => {
                        let top = match p.word("haut ou bas")?.as_str() {
                            "haut" => true,
                            "bas" => false,
                            o => return p.err(format!("origine « {o} » : haut ou bas")),
                        };
                        match &mut source {
                            FieldSource::File { top_origin, .. } => *top_origin = top,
                            _ => return p.err("« origine » ne vaut que pour un fichier"),
                        }
                    }
                    "valeurs" => {
                        let a = p.quantity(dim, "la valeur du niveau 0")?;
                        let b = p.quantity(dim, "la valeur du niveau 1")?;
                        match &mut source {
                            FieldSource::File { range, .. } => *range = Some((a, b)),
                            _ => return p.err("« valeurs » ne vaut que pour un fichier"),
                        }
                    }
                    "a" => temp = Some(p.quantity(Dim::Temp, "la température")?),
                    o => return p.err(format!("option « {o} » inconnue après « champ » : dans <zone>, origine haut|bas, valeurs a b, à <T>")),
                }
            }
            push(proto, Action::Field { qty, source, zone, temp }, ln);
        }
        "gravite" => {
            let g = if p.peek() == Some(&Tok::LP) {
                p.vector(Dim::Accel, "la gravité")?
            } else {
                [0.0, -p.quantity(Dim::Accel, "la gravité")?]
            };
            p.end()?;
            push(proto, Action::Gravity(g), ln);
        }
        "pas" => {
            if !p.eat_w("max") {
                return p.err("il faut « pas max <durée> »");
            }
            let d = p.quantity(Dim::Time, "le pas de temps maximal")?;
            if !(d > 0.0) {
                return p.err("le pas maximal doit être positif");
            }
            p.end()?;
            push(proto, Action::DtMax(d), ln);
        }
        "sauver" | "enregistrer" => {
            if p.eat_w("image") {
                // sauver image "nom" [phases | température | pression | vitesse | masse volumique | vapeur | divergence | flamme | chaleur]
                let name = p.string("le nom de l'image")?;
                if name.is_empty() || name.contains(['/', '\\', ':']) || name.starts_with('.') {
                    return p.err(format!("nom d'image « {name} » : un simple nom de fichier, sans dossier"));
                }
                let display = p.display()?;
                p.end()?;
                push(proto, Action::SaveImage { name, display }, ln);
                return Ok(());
            }
            if !p.eat_w("etat") {
                return p.err("il faut « sauver état \"nom\" » ou « sauver image \"nom\" [affichage] »");
            }
            let name = p.string("le nom de l'état")?;
            if name.is_empty() || name.contains(['/', '\\', ':']) || name.starts_with('.') {
                return p.err(format!("nom d'état « {name} » : un simple nom de fichier, sans dossier"));
            }
            p.end()?;
            push(proto, Action::SaveState(name), ln);
        }
        "marque" | "repere" => {
            let t = p.string("le texte du repère")?;
            p.end()?;
            push(proto, Action::Mark(t), ln);
        }
        "pause" => {
            p.end()?;
            push(proto, Action::Pause, ln);
        }
        "balayage" => return p.err("un seul « balayage » par protocole"),
        o => {
            return p.err(format!(
                "commande « {o} » inconnue (départ, mesurer, simuler, remplir, température, solide, corps, effacer, chauffer, accélérer, vitesse, \
                 allumer, puits, émetteur, grille, vanne, pompe, téléporteur, bord, champ, gravité, pas max, sauver état, marque, pause, \
                 calcul, matériaux, balayage, toutes les, à, quand, attendre, tant que … fin)"
            ))
        }
    }
    Ok(())
}

/// Lit un protocole (sans balayage : voir `parse_all`).
pub fn parse(text: &str, variant: Option<(String, String)>) -> Result<Protocol, Vec<ScriptError>> {
    let mut proto = Protocol { source: text.to_string(), variant, ..Default::default() };
    let mut errs = vec![];
    // boucles « tant que » ouvertes : (indice de leur étape, ligne) ; usize::MAX pour une ligne « tant que » mal écrite (son « fin »
    // la ferme sans autre erreur)
    let mut open: Vec<(usize, usize)> = vec![];
    for (k, l) in text.lines().enumerate() {
        let ln = k + 1;
        let toks = match tokenize(l, ln) {
            Ok(t) => t,
            Err(e) => {
                errs.push(e);
                continue;
            }
        };
        if toks.is_empty() || matches!(toks.first(), Some(Tok::W(w, _)) if w == "balayage") {
            continue;
        }
        let mut p = P { t: &toks, i: 0, ln };
        let n0 = proto.steps.len();
        if let Err(e) = parse_line(&mut p, &mut proto) {
            if matches!(toks.first(), Some(Tok::W(w, _)) if w == "tant") {
                open.push((usize::MAX, ln));
            }
            errs.push(e);
        }
        for s in n0..proto.steps.len() {
            match proto.steps[s].0 {
                Step::While { .. } => open.push((s, ln)),
                Step::EndWhile { .. } => match open.pop() {
                    Some((usize::MAX, _)) => {}
                    Some((w, wl)) => {
                        let advances = proto.steps[w + 1..s].iter().any(|(st, _)| matches!(st, Step::Simulate(_) | Step::Until(_) | Step::UntilCond { .. }));
                        if !advances {
                            errs.push(ScriptError { line: wl, msg: "le corps d'une boucle « tant que » doit faire avancer le temps (simuler, attendre)".into() });
                        }
                        if let Step::While { end, .. } = &mut proto.steps[w].0 {
                            *end = s;
                        }
                        if let Step::EndWhile { start } = &mut proto.steps[s].0 {
                            *start = w;
                        }
                    }
                    None => errs.push(ScriptError { line: ln, msg: "« fin » sans « tant que » ouvert".into() }),
                },
                _ => {}
            }
        }
    }
    for (w, wl) in open {
        if w != usize::MAX {
            errs.push(ScriptError { line: wl, msg: "« tant que » sans son « fin »".into() });
        }
    }
    if proto.start.is_none() && errs.is_empty() {
        errs.push(ScriptError { line: 1, msg: "il manque la ligne « départ scénario \"…\" » (ou « départ état \"fichier.mps\" »)".into() });
    }
    if errs.is_empty() { Ok(proto) } else { Err(errs) }
}

fn strip_comment(l: &str) -> &str {
    let mut quoted = false;
    for (i, c) in l.char_indices() {
        match c {
            '"' => quoted = !quoted,
            '#' if !quoted => return &l[..i],
            _ => {}
        }
    }
    l
}

/// Ligne de balayage : (nom, valeurs) ; None si la ligne n'en est pas une.
fn sweep_line(l: &str, ln: usize) -> Option<Result<(String, Vec<String>), ScriptError>> {
    let t = strip_comment(l).trim_start();
    let first: String = t.chars().take_while(|c| c.is_alphabetic()).collect();
    if norm(&first) != "balayage" {
        return None;
    }
    let err = |m: &str| Some(Err(ScriptError { line: ln, msg: m.into() }));
    let rest = &t[first.len()..];
    let Some((name, vals)) = rest.split_once('=') else {
        return err("il faut « balayage NOM = valeur1, valeur2, … »");
    };
    let name = name.trim().trim_start_matches('$').to_string();
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return err("nom de balayage : lettres, chiffres et _ seulement");
    }
    let sep = if vals.contains(';') { ';' } else { ',' };
    let values: Vec<String> = vals.split(sep).map(|v| v.trim().to_string()).filter(|v| !v.is_empty()).collect();
    if values.is_empty() {
        return err("balayage sans valeur");
    }
    Some(Ok((name, values)))
}

/// Lit un protocole et ses variantes : une par valeur du balayage (`balayage NOM = v1, v2, …` ; « ; » entre les valeurs si elles
/// contiennent des virgules), `$NOM` remplacé dans le texte. Les numéros de ligne sont ceux du texte d'origine.
pub fn parse_all(text: &str) -> Result<Vec<Protocol>, Vec<ScriptError>> {
    let lines: Vec<&str> = text.lines().collect();
    let mut sweep = None;
    let mut errs = vec![];
    for (k, l) in lines.iter().enumerate() {
        match sweep_line(l, k + 1) {
            None => {}
            Some(Err(e)) => errs.push(e),
            Some(Ok(s)) => {
                if sweep.is_some() {
                    errs.push(ScriptError { line: k + 1, msg: "un seul « balayage » par protocole".into() });
                } else {
                    sweep = Some((k, s.0, s.1));
                }
            }
        }
    }
    if !errs.is_empty() {
        return Err(errs);
    }
    let Some((k, name, values)) = sweep else {
        return parse(text, None).map(|p| vec![p]);
    };
    let var = format!("${name}");
    let mut out = vec![];
    for v in &values {
        let src: Vec<String> = lines.iter().enumerate().map(|(i, l)| if i == k { format!("# balayage {name} = {v}") } else { l.replace(&var, v) }).collect();
        out.push(parse(&src.join("\n"), Some((name.clone(), v.clone())))?);
    }
    Ok(out)
}

/// Écriture d'une zone dans la syntaxe du script (lue à l'identique par `parse`).
pub fn zone_text(z: &Zone) -> String {
    use super::mp_experiment::{fmt_num, fmt_point};
    match z {
        Zone::Partout => "partout".into(),
        Zone::Point(p) => format!("point {}", fmt_point(*p)),
        Zone::Disque { c, r } => format!("disque {} rayon {}", fmt_point(*c), fmt_num(*r)),
        Zone::Rect { a, b } => format!("rect {} {}", fmt_point(*a), fmt_point(*b)),
        Zone::Trait { pts, r } => {
            let p: Vec<String> = pts.iter().map(|p| fmt_point(*p)).collect();
            format!("trait {} rayon {}", p.join(" "), fmt_num(*r))
        }
    }
}

/// Insère une ligne de conditions initiales dans le texte d'un protocole : juste avant la première ligne qui fait avancer le temps
/// (« simuler »), sinon à la fin. Retourne le nouveau texte et le numéro (à partir de 1) de la ligne insérée.
pub fn insert_initial_line(text: &str, line: &str) -> (String, usize) {
    let lines: Vec<&str> = text.lines().collect();
    let pos = lines
        .iter()
        .position(|l| {
            let first: String = strip_comment(l).trim_start().chars().take_while(|c| c.is_alphabetic()).collect();
            matches!(norm(&first).as_str(), "simuler" | "avancer" | "attendre" | "tant")
        })
        .unwrap_or(lines.len());
    // pas de ligne vide en trop à la fin
    let mut out: Vec<String> = lines[..pos].iter().map(|s| s.to_string()).collect();
    let mut at = pos;
    if pos == lines.len() {
        while out.last().is_some_and(|l| l.trim().is_empty()) {
            out.pop();
        }
        at = out.len();
    }
    out.push(line.to_string());
    out.extend(lines[pos..].iter().map(|s| s.to_string()));
    let mut s = out.join("\n");
    if text.ends_with('\n') || pos == lines.len() {
        s.push('\n');
    }
    (s, at + 1)
}
