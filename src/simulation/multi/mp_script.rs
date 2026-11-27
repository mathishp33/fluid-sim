//! Lecture des protocoles d'expérience écrits en français (format complet : EXPERIENCES.md) → `mp_experiment::Protocol`.
//!
//! Une commande par ligne, `#` commence un commentaire ; mots-clés sans casse ni accents obligatoires (« départ » = « depart ») ;
//! noms (fluides, matériaux, fichiers) entre guillemets ; nombres avec un POINT décimal ; unités facultatives (SI par défaut :
//! m, s, K, m/s, m/s², Pa, W/m³). Un `balayage NOM = v1, v2, …` rejoue le protocole pour chaque valeur (`$NOM` dans le texte).
//! Composants (`inclure "fichier.exp" avec NOM = valeur`, `paramètre NOM = valeur`, `$NOM`, `$(expression)`) : `mp_compose`, avant la
//! lecture ; zones nommées (`zone "nom" = <zone>`, puis `zone "nom"` à la place d'une zone) : ici, à la lecture de chaque ligne.

use super::mp_experiment::{
    norm, Action, BodyMotion, BodyQty, BodySpec, BoundarySpec, Calc, Clause, Cmp, CrankQty, CrankSpec, FieldQty, FieldSource, Includes, LetValue, Logic, Measure,
    MechSpec, PieceSpec, Protocol, Quantity, Reduce, Start, Step, ThermalSpec, Threshold, WheelOn, WheelSpec, Who, Zone,
};
use std::path::Path;
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
pub(crate) enum Tok {
    /// mot (normalisé, tel qu'écrit)
    W(String, String),
    N(f64),
    S(String),
    /// valeur calculée « {…} » : formule en t (temps simulé) et en variables
    Calc(String),
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
        Tok::Calc(s) => format!("« {{{s}}} »"),
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

/// valeur SI d'un nombre suivi ou non de son unité (« 5 cm » : 0.05 ; « 20 °C » : 293.15 ; angles et vitesses de rotation en degrés,
/// comme leurs relevés : « 600 tr/min » : 3600) ; None si le texte n'est pas un nombre
pub(crate) fn si_number(text: &str) -> Option<f64> {
    match tokenize(text, 0).ok()?.as_slice() {
        [Tok::N(v)] => Some(*v),
        [Tok::N(v), Tok::W(w, _)] => {
            let (d, f, off) = unit_of(w)?;
            let x = v * f + off;
            Some(if matches!(d, Dim::Angle | Dim::AngSpeed) { x.to_degrees() } else { x })
        }
        _ => None,
    }
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
        if c == '{' {
            let mut j = i + 1;
            while j < n && cs[j] != '}' {
                j += 1;
            }
            if j >= n {
                return Err(err("accolade non refermée".into()));
            }
            out.push(Tok::Calc(cs[i + 1..j].iter().collect::<String>().trim().to_string()));
            i = j + 1;
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
            return Err(err(format!("valeur « ${name} » inconnue : il faut une ligne « paramètre {name} = … » (ou « balayage {name} = … »)")));
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

/// genre d'un objet nommé (vérification des noms en fin de lecture)
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Body,
    /// vilebrequin (un mécanisme à bielle-manivelle)
    Crank,
    /// mécanisme quelconque (`mécanisme "X"` ; un vilebrequin nommé en est un)
    Mech,
    Wheel,
}

struct P<'a> {
    t: &'a [Tok],
    i: usize,
    ln: usize,
    /// variables affectées par les lignes d'avant (`soit`) : seuils, relevés et formules peuvent les nommer
    vars: Vec<String>,
    /// noms lus sur la ligne : objets nommés désignés (relevés, roues et pièces de mécanismes) et posés — vérifiés en fin de lecture
    refs: Vec<(Kind, String)>,
    decls: Vec<(Kind, String)>,
}

/// mots qui ne peuvent pas nommer une variable (relevés, grandeurs, liaisons, fonctions des formules)
const RESERVED: [&str; 51] = [
    "t", "temps", "stats", "statistiques", "masse", "force", "debit", "corps", "moyenne", "max", "maximum", "min", "minimum", "somme",
    "integrale", "temperature", "p", "pression", "vx", "vy", "vitesse", "v", "rho", "densite", "fraction", "chaleur", "puissance", "et",
    "ou", "que", "au", "plus", "si", "sinon", "fin", "comme", "dans", "en", "pi", "e", "exp", "ln", "sqrt", "abs", "sin", "cos", "tan",
    "pow", "hypot", "clamp", "step",
];

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
            Some(Tok::Calc(_)) => self.err(format!(
                "une valeur calculée {{…}} ne va pas ici (il faut {what}) : elle va dans une action, une durée de simuler ou attendre, un nombre de tours de répéter, un seuil ou la valeur de « soit »"
            )),
            _ => self.err(format!("il faut {what}{}", self.here())),
        }
    }
    /// corps ou vilebrequin d'un relevé : son numéro (1 pour le premier posé) ou son nom entre guillemets (noté, pour la vérification
    /// de fin de lecture)
    fn who(&mut self, what: &str, kind: Kind) -> R<Who> {
        if let Some(Tok::S(s)) = self.peek() {
            let s = s.clone();
            self.i += 1;
            if s.trim().is_empty() {
                return self.err("nom vide");
            }
            self.refs.push((kind, s.clone()));
            return Ok(Who::Name(s));
        }
        Ok(Who::Rank(self.count(what)? - 1))
    }
    /// nom (entre guillemets) d'un objet que la ligne désigne, noté pour la vérification de fin de lecture
    fn named(&mut self, what: &str, kind: Kind) -> R<String> {
        let s = self.string(what)?;
        if s.trim().is_empty() {
            return self.err("nom vide");
        }
        self.refs.push((kind, s.clone()));
        Ok(s)
    }
    /// angle en degrés sans unité (ou avec la sienne)
    fn degrees(&mut self, what: &str) -> R<f64> {
        let v = self.number(what)?;
        let (v, unit) = self.opt_unit(Dim::Angle, v)?;
        Ok(if unit { v.to_degrees() } else { v })
    }
    /// vitesse de rotation en degrés par seconde sans unité (ou avec la sienne) : rad/s
    fn rotation_speed(&mut self, what: &str) -> R<f64> {
        let v = self.number(what)?;
        let (v, unit) = self.opt_unit(Dim::AngSpeed, v)?;
        Ok(if unit { v } else { v.to_radians() })
    }
    /// grandeur d'un vilebrequin ou d'un mécanisme
    fn crank_qty(&mut self) -> R<CrankQty> {
        Ok(match self.word("la grandeur du mécanisme (angle, tours, rotation, couple, puissance)")?.as_str() {
            "angle" => CrankQty::Angle,
            "tours" | "tour" => CrankQty::Turns,
            "rotation" => CrankQty::Rotation,
            "couple" => CrankQty::Torque,
            "puissance" => CrankQty::Power,
            o => return self.err(format!("grandeur « {o} » d'un mécanisme : angle, tours, rotation, couple, puissance")),
        })
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
    /// formule en t (temps simulé) et en les variables déjà affectées
    fn calc(&self, text: &str) -> R<Calc> {
        let names: Vec<&str> = std::iter::once("t").chain(self.vars.iter().map(|s| s.as_str())).collect();
        match Calc::new(text, &names) {
            Ok(c) => Ok(c),
            Err(e) => self.err(format!("formule « {text} » : {e}")),
        }
    }
    /// unité facultative quelconque après un nombre : sa valeur SI (angles et vitesses de rotation en degrés, comme leurs relevés)
    fn any_unit(&mut self, v: f64) -> f64 {
        if let Some(Tok::W(w, _)) = self.peek() {
            if let Some((d, f, off)) = unit_of(w) {
                self.i += 1;
                let x = v * f + off;
                return if matches!(d, Dim::Angle | Dim::AngSpeed) { x.to_degrees() } else { x };
            }
        }
        v
    }
    fn threshold(&mut self, m: &Measure) -> R<Threshold> {
        // variable seule, ou formule en t et en variables : calculées à chaque relevé, dans l'unité du relevé
        match self.peek() {
            Some(Tok::S(s) | Tok::Calc(s)) => {
                let s = s.clone();
                self.i += 1;
                return Ok(Threshold::Calc(self.calc(&s)?));
            }
            Some(Tok::W(w, _)) if self.vars.contains(w) => {
                let w = w.clone();
                self.i += 1;
                return Ok(Threshold::Calc(self.calc(&w)?));
            }
            _ => {}
        }
        let dim = match m {
            Measure::Probe { reduce: Reduce::Sum, .. } | Measure::Mass { .. } | Measure::SolidForce { .. } | Measure::Flux { .. } | Measure::Stats => {
                Dim::Plain
            }
            Measure::Time { .. } => Dim::Time,
            Measure::Var { .. } => Dim::Plain,
            Measure::Crank { q, .. } => match q {
                CrankQty::Angle => Dim::Angle,
                CrankQty::Rotation => Dim::AngSpeed,
                CrankQty::Turns | CrankQty::Torque | CrankQty::Power => Dim::Plain,
            },
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
        Ok(Threshold::Num(if unit && matches!(dim, Dim::Angle | Dim::AngSpeed) { v.to_degrees() } else { v }))
    }
    fn point(&mut self) -> R<[f64; 2]> {
        // « point (x, y) » aussi : ce qu'écrit une zone nommée qui est un point (`zone "capteur" = point (x, y)`, puis `p en zone "capteur"`)
        if self.peek_w() == Some("point") && self.t.get(self.i + 1) == Some(&Tok::LP) {
            self.i += 1;
        }
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
        let var = self.vars.iter().position(|v| *v == w);
        let mut m = match (w.as_str(), reduce) {
            _ if var.is_some() => Measure::Var { var: var.unwrap(), label: None },
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
            ("vilebrequin", _) => {
                let who = self.who("le vilebrequin : son numéro (1 pour le premier posé) ou son nom entre guillemets", Kind::Crank)?;
                Measure::Crank { who, q: self.crank_qty()?, label: None, any: false }
            }
            ("mecanisme", _) => {
                let who = self.who("le mécanisme : son numéro (1 pour le premier posé, vilebrequins compris) ou son nom entre guillemets", Kind::Mech)?;
                Measure::Crank { who, q: self.crank_qty()?, label: None, any: true }
            }
            ("corps", _) => {
                let who = self.who("le corps : son numéro (1 pour le premier posé) ou son nom entre guillemets", Kind::Body)?;
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
                Measure::Body { who, q, label: None }
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
                | Measure::Time { label }
                | Measure::Var { label, .. }
                | Measure::Crank { label, .. } => *label = Some(l),
                Measure::Stats => return self.err("« comme » ne s'applique pas à « stats »"),
            }
        }
        Ok(m)
    }
}

/// « : <action> » d'une action datée ou répétée : le reste de la ligne, lu comme une commande d'une seule action
/// <comparaison> [et | ou <comparaison>]… — une comparaison est « <relevé> < | ≤ | > | ≥ <seuil> » ou un groupe entre parenthèses :
/// les comparaisons, et leur liaison
fn condition(p: &mut P) -> R<(Vec<Clause>, Logic)> {
    let mut clauses = vec![];
    let logic = cond_group(p, &mut clauses)?;
    Ok((clauses, logic))
}

/// suite de comparaisons (ou de groupes) liées par un seul mot : « et » ou « ou » (mêlés : parenthèses)
fn cond_group(p: &mut P, clauses: &mut Vec<Clause>) -> R<Logic> {
    let mut items = vec![cond_item(p, clauses)?];
    let mut link: Option<bool> = None;
    loop {
        let l = if p.eat_w("et") {
            false
        } else if p.eat_w("ou") {
            true
        } else {
            break;
        };
        if link.is_some_and(|a| a != l) {
            return p.err("« et » et « ou » mêlés : mettez des parenthèses, par exemple « (A et B) ou C »");
        }
        link = Some(l);
        items.push(cond_item(p, clauses)?);
    }
    Ok(match (items.len(), link) {
        (1, _) => items.pop().unwrap(),
        (_, Some(true)) => Logic::Any(items),
        _ => Logic::All(items),
    })
}

/// une comparaison, ou un groupe « ( … ) »
fn cond_item(p: &mut P, clauses: &mut Vec<Clause>) -> R<Logic> {
    if p.eat(&Tok::LP) {
        let g = cond_group(p, clauses)?;
        if !p.eat(&Tok::RP) {
            return p.err(format!("il manque « ) » à la fin du groupe{}", p.here()));
        }
        return Ok(g);
    }
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
    Ok(Logic::Leaf(clauses.len() - 1))
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
    let mut q = P { t: &p.t[p.i..], i: 0, ln, vars: p.vars.clone(), refs: vec![], decls: vec![] };
    parse_line(&mut q, &mut inner)?;
    p.refs.append(&mut q.refs);
    p.decls.append(&mut q.decls);
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

/// forme d'un corps : disque (x, y) rayon r, rect (x1, y1) (x2, y2), polygone (x1, y1) (x2, y2) (x3, y3)…
fn body_shape(p: &mut P) -> R<BodySpec> {
    Ok(match p.word("la forme du corps (disque, rect, polygone)")?.as_str() {
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
    })
}

/// noms que pose une roue : la roue, son disque, ses dents ou ses pales
fn wheel_names(s: &WheelSpec) -> Vec<(Kind, String)> {
    let Some(n) = &s.name else { return vec![] };
    let mut v = vec![(Kind::Wheel, n.clone()), (Kind::Body, n.clone())];
    if let Some(k) = s.teeth {
        v.extend((1..=k).map(|j| (Kind::Body, format!("{n}.dent{j}"))));
    }
    if let Some((k, _, _)) = s.blades {
        v.extend((1..=k).map(|j| (Kind::Body, format!("{n}.pale{j}"))));
    }
    v
}

/// commandes qui ne sont pas des actions : leurs valeurs calculées {…} (seuils, durées, tours, « soit ») sont lues par elles-mêmes
const NOT_ACTIONS: [&str; 19] = [
    "depart", "materiaux", "mesurer", "simuler", "avancer", "attendre", "tant", "repeter", "si", "sinon", "fin", "quand", "soit", "toutes",
    "chaque", "a", "calcul", "images", "image",
];

/// Action dont des valeurs sont calculées (`{…}`) : vérifiée ici, ces valeurs posées à 1 ; relue avec les valeurs de l'instant chaque
/// fois qu'elle est faite
fn deferred(p: &mut P, ln: usize) -> R<Action> {
    let toks: Vec<Tok> = p.t[p.i - 1..].to_vec();
    let mut calcs = vec![];
    for t in &toks {
        if let Tok::Calc(f) = t {
            calcs.push(p.calc(f)?);
        }
    }
    let probe: Vec<Tok> = toks.iter().map(|t| if matches!(t, Tok::Calc(_)) { Tok::N(1.0) } else { t.clone() }).collect();
    match action_from_tokens(&probe, ln, &p.vars) {
        Ok(a) => {
            p.i = p.t.len();
            match &a {
                Action::Crank(CrankSpec { name: Some(n), .. }) => p.decls.push((Kind::Crank, n.clone())),
                Action::Body { name: Some(n), .. } | Action::Piece(PieceSpec { name: Some(n), .. }) => p.decls.push((Kind::Body, n.clone())),
                Action::Mechanism(MechSpec { name, .. }) => p.decls.push((Kind::Mech, name.clone())),
                Action::Wheel(s) => p.decls.extend(wheel_names(s)),
                _ => {}
            }
            Ok(Action::Deferred { toks, calcs, check: Box::new(a) })
        }
        Err(e) => p.err(e),
    }
}

/// action d'une ligne déjà découpée (valeurs calculées remplacées par des nombres) ; `vars` : les variables du protocole
pub(crate) fn action_from_tokens(toks: &[Tok], ln: usize, vars: &[String]) -> Result<Action, String> {
    let mut scratch = Protocol { vars: vars.to_vec(), ..Default::default() };
    let mut q = P { t: toks, i: 0, ln, vars: vars.to_vec(), refs: vec![], decls: vec![] };
    parse_line(&mut q, &mut scratch).map_err(|e| e.msg)?;
    match scratch.steps.pop() {
        Some((Step::Do(a), _)) if scratch.steps.is_empty() && scratch.measures.is_empty() => Ok(a),
        _ => Err("des valeurs calculées {…} vont dans une action, une durée (simuler, attendre {…} s), un nombre de tours (répéter {…} fois), un seuil ou la valeur de « soit »".into()),
    }
}

fn parse_line(p: &mut P, proto: &mut Protocol) -> R<()> {
    let ln = p.ln;
    let cmd = p.word("une commande")?;
    // action aux valeurs calculées {…} : relue avec les valeurs de l'instant chaque fois qu'elle est faite (aussi après « : » d'une
    // règle ou d'une répétition, lue par cette même fonction)
    if !NOT_ACTIONS.contains(&cmd.as_str()) && p.t[p.i..].iter().any(|t| matches!(t, Tok::Calc(_))) {
        let a = deferred(p, ln)?;
        push(proto, a, ln);
        return Ok(());
    }
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
            // quand <condition> : <action> — faite chaque fois que la condition devient vraie (relevée après chaque pas) ; sans « : »,
            // un bloc jusqu'au « fin », déroulé en parallèle (attentes comprises)
            let (clauses, logic) = condition(p)?;
            if p.done() {
                proto.steps.push((Step::WhenBlock { clauses, logic, end: usize::MAX }, ln));
            } else {
                let action = timed_action(p, ln, true)?;
                push(proto, Action::When { clauses, logic, action: Box::new(action) }, ln);
            }
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
            // « mesurer toutes les <durée> [: <mesures>] » ; « mesurer : <mesures> » ajoute des colonnes sans fixer la période (celles
            // d'un composant : la période est celle de l'expérience)
            if !p.eat(&Tok::Colon) {
                if p.eat_w("toutes") {
                    if !p.eat_w("les") {
                        return p.err("il faut « toutes les <durée> »");
                    }
                } else if !p.eat_w("chaque") {
                    return p.err(format!("il faut « toutes les <durée> : … » (ou « mesurer : … » sans la période){}", p.here()));
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
                if p.done() {
                    return Ok(());
                }
                p.eat(&Tok::Colon);
            }
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
                let (clauses, logic) = condition(p)?;
                Step::UntilCond { clauses, logic, limit: opt_limit(p)? }
            } else if matches!(p.t.get(p.i.wrapping_sub(1)), Some(Tok::W(w, _)) if w == "jusqua" || w == "jusque") {
                Step::Until(p.quantity(Dim::Time, "le temps à atteindre")?)
            } else if let Some(Tok::Calc(f)) = p.peek() {
                // durée calculée {…} [unité], quand la ligne est atteinte
                let f = f.clone();
                p.i += 1;
                let calc = p.calc(&f)?;
                let factor = p.opt_unit(Dim::Time, 1.0)?.0;
                Step::SimulateCalc { calc, factor }
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
        "soit" => {
            // soit <nom> = <relevé> | <nombre> [unité] | "<formule>" — variable du protocole (0 avant sa première affectation)
            let name = p.word("le nom de la variable")?;
            if !name.chars().next().is_some_and(|c| c.is_alphabetic()) || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                return p.err(format!("« {name} » : un nom de variable est fait de lettres, de chiffres et de « _ », et commence par une lettre"));
            }
            if RESERVED.contains(&name.as_str()) {
                return p.err(format!("« {name} » est un mot du protocole (relevé, liaison, fonction) : choisissez un autre nom de variable"));
            }
            if !p.eat(&Tok::Eq) {
                return p.err(format!("il faut « soit {name} = <valeur> »{}", p.here()));
            }
            let var = match proto.vars.iter().position(|v| *v == name) {
                Some(k) => k,
                None => {
                    proto.vars.push(name.clone());
                    p.vars.push(name.clone());
                    proto.vars.len() - 1
                }
            };
            let value = match p.peek() {
                Some(Tok::S(f) | Tok::Calc(f)) => {
                    let f = f.clone();
                    p.i += 1;
                    LetValue::Calc(p.calc(&f)?)
                }
                Some(Tok::N(_)) => {
                    let v = p.number("la valeur")?;
                    LetValue::Num(p.any_unit(v))
                }
                _ => {
                    let m = p.measure()?;
                    if m == Measure::Stats {
                        return p.err("« stats » n'est pas une valeur : un relevé (T en (x, y), masse \"eau\", corps 1 y…), un nombre ou une formule");
                    }
                    LetValue::Measure(m)
                }
            };
            p.end()?;
            push(proto, Action::Let { var, value }, ln);
        }
        "attendre" => {
            // attendre <durée> | jusqu'à <instant> | [que | jusqu'à ce que] <condition> [au plus <durée>] — comme « simuler »
            let step = if let Some(Tok::Calc(f)) = p.peek() {
                // durée calculée {…} [unité], quand la ligne est atteinte
                let f = f.clone();
                p.i += 1;
                let calc = p.calc(&f)?;
                let factor = p.opt_unit(Dim::Time, 1.0)?.0;
                Step::SimulateCalc { calc, factor }
            } else if matches!(p.peek(), Some(Tok::N(_))) {
                let d = p.quantity(Dim::Time, "la durée à attendre")?;
                if !(d >= 0.0) {
                    return p.err("la durée doit être positive");
                }
                Step::Simulate(d)
            } else if (p.eat_w("jusqua") || p.eat_w("jusque")) && !p.eat_w("ce") {
                Step::Until(p.quantity(Dim::Time, "l'instant à atteindre")?)
            } else {
                p.eat_w("que");
                let (clauses, logic) = condition(p)?;
                Step::UntilCond { clauses, logic, limit: opt_limit(p)? }
            };
            p.end()?;
            proto.steps.push((step, ln));
        }
        "tant" => {
            // tant que <condition> [au plus <durée>] … fin — boucle (les liens entre « tant que » et « fin » sont posés par `parse`)
            if !p.eat_w("que") {
                return p.err(format!("il faut « tant que <condition> »{}", p.here()));
            }
            let (clauses, logic) = condition(p)?;
            let limit = opt_limit(p)?;
            p.end()?;
            proto.steps.push((Step::While { clauses, logic, limit, end: usize::MAX }, ln));
        }
        "fin" => {
            // fin [tant que | répéter | si | quand] : ferme le bloc ouvert le plus récent
            if p.eat_w("tant") && !p.eat_w("que") {
                return p.err("il faut « fin » ou « fin tant que »");
            }
            let _ = p.eat_w("repeter") || p.eat_w("si") || p.eat_w("quand");
            p.end()?;
            proto.steps.push((Step::End { start: usize::MAX }, ln));
        }
        "repeter" => {
            // répéter <n> fois … fin — n peut être calculé ({…}, à l'entrée dans la boucle)
            let (n, count) = if let Some(Tok::Calc(f)) = p.peek() {
                let f = f.clone();
                p.i += 1;
                (0, Some(p.calc(&f)?))
            } else {
                (p.count("le nombre de tours")?, None)
            };
            if !p.eat_w("fois") {
                return p.err(format!("il faut « répéter <n> fois »{}", p.here()));
            }
            p.end()?;
            proto.steps.push((Step::Repeat { n, count, end: usize::MAX }, ln));
        }
        "si" => {
            // si <condition> … [sinon si <condition> …] [sinon …] fin
            let (clauses, logic) = condition(p)?;
            p.end()?;
            proto.steps.push((Step::If { clauses, logic, next: usize::MAX, end: usize::MAX }, ln));
        }
        "sinon" => {
            let cond = if p.eat_w("si") { Some(condition(p)?) } else { None };
            p.end()?;
            proto.steps.push((Step::Else { cond, next: usize::MAX, end: usize::MAX }, ln));
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
        "vilebrequin" => {
            // vilebrequin (x, y) rayon r bielle l ["matériau"] [axe a] [angle a0] [alésage d] [piston h] [cylindre] [chambre c] [paroi e]
            //   [rotation ω] [libre] [inertie I] [couple "formule"] [à T]
            let pivot = p.point()?;
            if !p.eat_w("rayon") {
                return p.err(format!("il faut « rayon r », le rayon de manivelle (la moitié de la course){}", p.here()));
            }
            let r = p.quantity(Dim::Length, "le rayon de manivelle")?;
            if !p.eat_w("bielle") {
                return p.err(format!("il faut « bielle l », la longueur de la bielle entre ses axes{}", p.here()));
            }
            let l = p.quantity(Dim::Length, "la longueur de la bielle")?;
            let material = if matches!(p.peek(), Some(Tok::S(_))) { p.string("le matériau")? } else { "acier".to_string() };
            let mut spec = CrankSpec {
                pivot,
                r,
                l,
                axis_deg: 90.0,
                bore: None,
                piston_h: None,
                angle_deg: 0.0,
                cylinder: None,
                material,
                free: false,
                omega: 0.0,
                inertia: 0.0,
                load: None,
                temp: None,
                name: None,
            };
            let deg = |p: &mut P, what: &str| -> R<f64> {
                let v = p.number(what)?;
                let (v, unit) = p.opt_unit(Dim::Angle, v)?;
                Ok(if unit { v.to_degrees() } else { v })
            };
            while !p.done() {
                match p.word("une option (axe, angle, alésage, piston, cylindre, chambre, paroi, rotation, libre, inertie, couple, nom, à T)")?.as_str() {
                    "axe" => spec.axis_deg = deg(p, "la direction de l'axe du cylindre")?,
                    "angle" => spec.angle_deg = deg(p, "l'angle de départ depuis le point mort haut")?,
                    "alesage" => spec.bore = Some(p.quantity(Dim::Length, "l'alésage")?),
                    "piston" => spec.piston_h = Some(p.quantity(Dim::Length, "la hauteur du piston")?),
                    "cylindre" => spec.cylinder = Some(spec.cylinder.unwrap_or((None, None))),
                    "chambre" => {
                        let c = p.quantity(Dim::Length, "la longueur de la chambre au point mort haut")?;
                        spec.cylinder = Some((Some(c), spec.cylinder.and_then(|x| x.1)));
                    }
                    "paroi" => {
                        let e = p.quantity(Dim::Length, "l'épaisseur des parois du cylindre")?;
                        spec.cylinder = Some((spec.cylinder.and_then(|x| x.0), Some(e)));
                    }
                    "rotation" => {
                        let v = p.number("la vitesse de rotation")?;
                        let (v, unit) = p.opt_unit(Dim::AngSpeed, v)?;
                        spec.omega = if unit { v } else { v.to_radians() };
                    }
                    "libre" => spec.free = true,
                    "inertie" => {
                        spec.inertia = p.number("l'inertie du volant (kg·m par mètre de profondeur)")?;
                        if !(spec.inertia >= 0.0) {
                            return p.err("l'inertie du volant est positive");
                        }
                    }
                    "couple" => {
                        p.eat_w("formule");
                        let f = p.string("la formule du couple appliqué (N)")?;
                        if let Err(e) = super::mp_fields::Formula::parse_vars(&f, &super::mp_mechanism::CRANK_VARS) {
                            return p.err(format!("couple « {f} » : {e} (variables : t, angle, tours, rotation)"));
                        }
                        spec.load = Some(f);
                    }
                    "nom" => {
                        let n = p.string("le nom du vilebrequin")?;
                        if n.trim().is_empty() {
                            return p.err("nom vide");
                        }
                        p.decls.push((Kind::Crank, n.clone()));
                        spec.name = Some(n);
                    }
                    "a" => spec.temp = Some(p.quantity(Dim::Temp, "la température des pièces")?),
                    o => return p.err(format!("option « {o} » inconnue après « vilebrequin »")),
                }
            }
            if !spec.free && (spec.inertia > 0.0 || spec.load.is_some()) {
                return p.err("inertie et couple : pour un vilebrequin « libre » (un vilebrequin mené tourne à sa rotation)");
            }
            if !(spec.r > 0.0 && spec.l > 1.05 * spec.r) {
                return p.err("la bielle doit être plus longue que le rayon de manivelle (au moins 1.05 fois)");
            }
            push(proto, Action::Crank(spec), ln);
        }
        "mecanisme" => {
            // mécanisme "nom" [libre] [rotation ω] [inertie I] [couple "formule"] [angle a0]
            let name = p.string("le nom du mécanisme (ses roues et ses pièces le désignent : « roue … sur \"nom\" »)")?;
            if name.trim().is_empty() {
                return p.err("nom vide");
            }
            let mut spec = MechSpec { name: name.clone(), free: false, omega: 0.0, inertia: 0.0, load: None, angle_deg: 0.0 };
            while !p.done() {
                match p.word("une option (libre, rotation, inertie, couple, angle)")?.as_str() {
                    "libre" => spec.free = true,
                    "rotation" => spec.omega = p.rotation_speed("la vitesse de rotation")?,
                    "inertie" => {
                        spec.inertia = p.number("l'inertie du volant (kg·m par mètre de profondeur)")?;
                        if !(spec.inertia >= 0.0) {
                            return p.err("l'inertie du volant est positive");
                        }
                    }
                    "couple" => {
                        p.eat_w("formule");
                        let f = p.string("la formule du couple appliqué (N)")?;
                        if let Err(e) = super::mp_fields::Formula::parse_vars(&f, &super::mp_mechanism::CRANK_VARS) {
                            return p.err(format!("couple « {f} » : {e} (variables : t, angle, tours, rotation)"));
                        }
                        spec.load = Some(f);
                    }
                    "angle" => spec.angle_deg = p.degrees("l'angle de départ")?,
                    o => return p.err(format!("option « {o} » inconnue après « mécanisme » : libre, rotation ω, inertie I, couple \"formule\", angle a")),
                }
            }
            if !spec.free && (spec.inertia > 0.0 || spec.load.is_some()) {
                return p.err("inertie et couple : pour un mécanisme « libre » (un mécanisme mené tourne à sa rotation)");
            }
            p.decls.push((Kind::Mech, name));
            push(proto, Action::Mechanism(spec), ln);
        }
        "roue" => {
            // roue (x, y) rayon r ["matériau"] sur "M" [rapport k] | engrène "R" ; [dents n | pales n longueur L [épaisseur e]]
            //   [angle a] [nom "R"] [à T]
            let center = p.point()?;
            if !(p.eat_w("rayon") || p.eat_w("r")) {
                return p.err(format!("il faut « rayon r » après le centre de la roue{}", p.here()));
            }
            let r = p.quantity(Dim::Length, "le rayon de la roue")?;
            if !(r > 0.0) {
                return p.err("le rayon d'une roue doit être positif");
            }
            let mut spec =
                WheelSpec { center, r, material: "acier".into(), on: WheelOn::Mesh(String::new()), teeth: None, blades: None, angle_deg: 0.0, name: None, temp: None };
            let (mut on, mut ratio) = (None, None);
            while !p.done() {
                // (le matériau, entre guillemets, n'importe où parmi les options)
                if let Some(Tok::S(m)) = p.peek() {
                    spec.material = m.clone();
                    p.i += 1;
                    continue;
                }
                match p.word("une option (sur, engrène, rapport, dents, pales, angle, nom, à T) ou le matériau entre guillemets")?.as_str() {
                    "sur" => on = Some(WheelOn::Shaft { mech: p.named("le mécanisme qui porte la roue", Kind::Mech)?, ratio: 1.0 }),
                    "engrene" => on = Some(WheelOn::Mesh(p.named("la roue qu'elle engrène", Kind::Wheel)?)),
                    "rapport" => {
                        let k = p.number("le rapport (tours de la roue pour un tour de l'arbre ; négatif : sens inverse)")?;
                        if !(k != 0.0 && k.is_finite()) {
                            return p.err("le rapport d'une roue n'est pas nul");
                        }
                        ratio = Some(k);
                    }
                    "dents" => {
                        let n = p.count("le nombre de dents")?;
                        if n < 6 {
                            return p.err("une roue dentée a au moins 6 dents");
                        }
                        spec.teeth = Some(n);
                    }
                    "pales" => {
                        let n = p.count("le nombre de pales")?;
                        if !p.eat_w("longueur") {
                            return p.err(format!("il faut « pales n longueur L [épaisseur e] »{}", p.here()));
                        }
                        let l = p.quantity(Dim::Length, "la longueur des pales")?;
                        let e = if p.eat_w("epaisseur") { Some(p.quantity(Dim::Length, "l'épaisseur des pales")?) } else { None };
                        if !(l > 0.0 && e.map_or(true, |e| e > 0.0)) {
                            return p.err("longueur et épaisseur des pales positives");
                        }
                        spec.blades = Some((n, l, e));
                    }
                    "angle" => spec.angle_deg = p.degrees("l'angle de départ de la roue")?,
                    "nom" => {
                        let n = p.string("le nom de la roue")?;
                        if n.trim().is_empty() {
                            return p.err("nom vide");
                        }
                        spec.name = Some(n);
                    }
                    "a" => spec.temp = Some(p.quantity(Dim::Temp, "la température de la roue")?),
                    o => return p.err(format!("option « {o} » inconnue après « roue » : sur \"mécanisme\" [rapport k] ou engrène \"roue\", dents n, pales n longueur L [épaisseur e], angle a, nom \"R\", à T")),
                }
            }
            spec.on = match (on, ratio) {
                (None, _) => return p.err("il faut « sur \"mécanisme\" » (la roue tourne avec son arbre) ou « engrène \"roue\" »"),
                (Some(WheelOn::Shaft { mech, .. }), Some(k)) => WheelOn::Shaft { mech, ratio: k },
                (Some(WheelOn::Mesh(_)), Some(_)) => {
                    return p.err("« rapport » : pour une roue « sur » un mécanisme (une roue qui en engrène une autre tourne au rapport de leurs rayons)")
                }
                (Some(o), None) => o,
            };
            if spec.teeth.is_some() && spec.blades.is_some() {
                return p.err("une roue a des dents ou des pales, pas les deux");
            }
            p.decls.extend(wheel_names(&spec));
            push(proto, Action::Wheel(spec), ln);
        }
        "piece" => {
            // pièce <forme> ["matériau"] [angle a] sur "M" [position "x(θ)" "y(θ)"] [angle "a(θ)"] [nom "P"] [à T]
            let shape = body_shape(p)?;
            let mut spec = PieceSpec { shape, material: "acier".into(), angle: 0.0, mech: String::new(), x: None, y: None, a: None, name: None, temp: None };
            let formula = |p: &mut P, what: &str| -> R<String> {
                let f = p.string(what)?;
                if let Err(e) = super::mp_fields::Formula::parse_vars(&norm(&f), &super::mp_mechanism::PIECE_VARS) {
                    return p.err(format!("{what} « {f} » : {e} (variables : theta, angle, tours, x0, y0, a0)"));
                }
                Ok(f)
            };
            while !p.done() {
                if let Some(Tok::S(m)) = p.peek() {
                    spec.material = m.clone();
                    p.i += 1;
                    continue;
                }
                match p.word("une option (sur, position, angle, nom, à T) ou le matériau entre guillemets")?.as_str() {
                    "sur" => spec.mech = p.named("le mécanisme qui mène la pièce", Kind::Mech)?,
                    "position" => {
                        p.eat_w("formule");
                        spec.x = Some(formula(p, "la position x (m)")?);
                        spec.y = Some(formula(p, "la position y (m)")?);
                    }
                    "angle" if matches!(p.peek(), Some(Tok::S(_))) => spec.a = Some(formula(p, "l'angle (°)")?),
                    "angle" if p.peek_w() == Some("formule") => {
                        p.i += 1;
                        spec.a = Some(formula(p, "l'angle (°)")?);
                    }
                    "angle" => spec.angle = p.degrees("l'angle de la pièce")?.to_radians(),
                    "nom" => {
                        let n = p.string("le nom de la pièce")?;
                        if n.trim().is_empty() {
                            return p.err("nom vide");
                        }
                        p.decls.push((Kind::Body, n.clone()));
                        spec.name = Some(n);
                    }
                    "a" => spec.temp = Some(p.quantity(Dim::Temp, "la température de la pièce")?),
                    o => return p.err(format!("option « {o} » inconnue après « pièce » : sur \"mécanisme\", position \"x\" \"y\", angle \"a\" (formules), angle a, nom \"P\", à T")),
                }
            }
            if spec.mech.is_empty() {
                return p.err("il faut « sur \"mécanisme\" » : le mécanisme qui mène la pièce");
            }
            push(proto, Action::Piece(spec), ln);
        }
        "corps" => {
            let shape = body_shape(p)?;
            let material = p.string("le matériau solide")?;
            let mut angle = 0.0;
            let mut motion = BodyMotion::default();
            let mut osc = false;
            let mut fixed_kw = false;
            let mut temp = None;
            let mut name: Option<String> = None;
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
                match p.word("une option (angle, fixe, libre, bloqué, vitesse, rotation, oscillation, période, pivot, masse volumique, rebond, frottement, nom, à T)")?.as_str() {
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
                    "nom" => {
                        let n = p.string("le nom du corps")?;
                        if n.trim().is_empty() {
                            return p.err("nom vide");
                        }
                        p.decls.push((Kind::Body, n.clone()));
                        name = Some(n);
                    }
                    o => {
                        return p.err(format!(
                            "option « {o} » inconnue après « corps » : angle <a>, fixe, libre, bloqué x | y | rotation, vitesse (vx, vy), \
                             rotation <ω>, oscillation (vx, vy) | oscillation rotation <ω> avec période <T>, pivot (x, y), masse volumique <ρ>, \
                             rebond <e>, frottement <μ>, nom \"X\", à <T> ; lois en formules : position | vitesse | accélération formule \"…\" \"…\", \
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
            push(proto, Action::Body { shape, material, angle, motion, temp, name }, ln);
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
        "eponge" => {
            // éponge <zone> [temps τ] [vers (vx, vy)] [uniforme] : couche éponge (bord absorbant), taux 1/τ au fond (τ : 0.1 s par défaut)
            let zone = p.zone()?;
            let (mut tau, mut target, mut uniform) = (0.1, [0.0, 0.0], false);
            while !p.done() {
                let w = p.word("une option (temps, vers, uniforme)")?;
                match w.as_str() {
                    "temps" => {
                        tau = p.quantity(Dim::Time, "le temps d'amortissement")?;
                        if !(tau > 0.0) {
                            return p.err("le temps d'amortissement d'une éponge est positif");
                        }
                    }
                    "vers" => target = p.vector(Dim::Speed, "la vitesse vers laquelle l'éponge ramène l'écoulement")?,
                    "uniforme" => uniform = true,
                    o => return p.err(format!("option « {o} » inconnue après « éponge » : temps <τ>, vers (vx, vy), uniforme")),
                }
            }
            push(proto, Action::Sponge { zone, tau, target, uniform }, ln);
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
            // « {…} » : valeur d'une formule en t et en variables (un nom de variable seul, par exemple), quand le repère est posé
            let mut rest = t.as_str();
            while let Some(i) = rest.find('{') {
                let Some(j) = rest[i..].find('}') else { break };
                p.calc(rest[i + 1..i + j].trim())?;
                rest = &rest[i + j + 1..];
            }
            p.end()?;
            push(proto, Action::Mark(t), ln);
        }
        "pause" => {
            p.end()?;
            push(proto, Action::Pause, ln);
        }
        "balayage" => return p.err("un seul « balayage » par protocole"),
        "zone" => return p.err("il faut « zone \"nom\" = <zone> » (puis « zone \"nom\" » à la place d'une zone)"),
        o => {
            return p.err(format!(
                "commande « {o} » inconnue (départ, mesurer, simuler, remplir, température, solide, corps, effacer, chauffer, accélérer, vitesse, \
                 allumer, puits, émetteur, grille, vanne, pompe, éponge, téléporteur, bord, champ, gravité, pas max, sauver état, marque, pause, \
                 calcul, matériaux, balayage, toutes les, à, quand, attendre, tant que … fin, répéter n fois … fin, si … sinon … fin, soit, \
                 vilebrequin, mécanisme, roue, pièce, zone \"nom\" = …, inclure, paramètre)"
            ))
        }
    }
    Ok(())
}

/// nom d'un bloc pour les messages
fn block_name(s: &Step) -> &'static str {
    match s {
        Step::While { .. } => "tant que",
        Step::Repeat { .. } => "répéter",
        Step::If { .. } => "si",
        Step::WhenBlock { .. } => "quand",
        _ => "bloc",
    }
}

/// Liens des blocs pour l'étape `s` (ligne `ln`) qui vient d'être lue : une ouverture (tant que, répéter, si, quand) est empilée ;
/// un « sinon » s'accroche au « si » ouvert ; un « fin » ferme le bloc le plus récent (sauts posés de part et d'autre)
fn close_blocks(proto: &mut Protocol, open: &mut Vec<(usize, usize, usize)>, errs: &mut Vec<ScriptError>, s: usize, ln: usize) {
    let set_next = |st: &mut Step, to: usize| {
        if let Step::If { next, .. } | Step::Else { next, .. } = st {
            *next = to;
        }
    };
    match &proto.steps[s].0 {
        Step::While { .. } | Step::Repeat { .. } | Step::If { .. } | Step::WhenBlock { .. } => open.push((s, ln, s)),
        Step::Else { .. } => match open.last_mut() {
            Some((usize::MAX, _, _)) => {}
            Some((w, _, branch)) if matches!(proto.steps[*w].0, Step::If { .. }) => {
                if matches!(proto.steps[*branch].0, Step::Else { cond: None, .. }) {
                    errs.push(ScriptError { line: ln, msg: "une branche après « sinon » : le « sinon » sans condition est la dernière branche du « si »".into() });
                }
                set_next(&mut proto.steps[*branch].0, s);
                *branch = s;
            }
            _ => errs.push(ScriptError { line: ln, msg: "« sinon » hors d'un « si »".into() }),
        },
        Step::End { .. } => match open.pop() {
            Some((usize::MAX, _, _)) => {}
            Some((w, wl, branch)) => {
                if matches!(proto.steps[w].0, Step::While { .. }) {
                    let advances = proto.steps[w + 1..s]
                        .iter()
                        .any(|(st, _)| matches!(st, Step::Simulate(_) | Step::SimulateCalc { .. } | Step::Until(_) | Step::UntilCond { .. }));
                    if !advances {
                        errs.push(ScriptError { line: wl, msg: "le corps d'une boucle « tant que » doit faire avancer le temps (simuler, attendre)".into() });
                    }
                }
                if let Step::While { end, .. } | Step::Repeat { end, .. } | Step::If { end, .. } | Step::WhenBlock { end, .. } = &mut proto.steps[w].0 {
                    *end = s;
                }
                if matches!(proto.steps[w].0, Step::If { .. }) {
                    // la dernière branche mène au « fin » ; chaque « sinon » du bloc en sort après sa branche
                    set_next(&mut proto.steps[branch].0, s);
                    for b in w + 1..s {
                        if let Step::Else { end, .. } = &mut proto.steps[b].0 {
                            if *end == usize::MAX {
                                *end = s;
                            }
                        }
                    }
                }
                if let Step::End { start } = &mut proto.steps[s].0 {
                    *start = w;
                }
            }
            None => errs.push(ScriptError { line: ln, msg: "« fin » sans bloc ouvert (tant que, répéter, si, quand)".into() }),
        },
        _ => {}
    }
}

/// Lit un protocole (sans balayage : voir `parse_all`) ; ses composants (`inclure`) se cherchent depuis le dossier courant.
pub fn parse(text: &str, variant: Option<(String, String)>) -> Result<Protocol, Vec<ScriptError>> {
    parse_in(text, Path::new("."), variant)
}

/// Lit un protocole dont les composants se cherchent depuis le dossier `base` ; `variant` : (nom, valeur) d'une variante de balayage.
fn parse_in(text: &str, base: &Path, variant: Option<(String, String)>) -> Result<Protocol, Vec<ScriptError>> {
    let ex = super::mp_compose::expand(text, base, variant.as_ref().map(|(n, v)| (n.as_str(), v.as_str())))?;
    let includes = ex.includes.clone();
    parse_lines(&ex.lines, ex.text, ex.includes, variant).map_err(|errs| errs.into_iter().map(|e| locate(&includes, e)).collect())
}

/// erreur à une ligne venue d'un composant : à la ligne de son « inclure », située dans le composant
fn locate(inc: &Includes, e: ScriptError) -> ScriptError {
    match inc.get(e.line) {
        Some(o) => ScriptError { line: o.line, msg: format!("{}« {} » ligne {} : {}", o.via, o.file, o.at, e.msg) },
        None => e,
    }
}

/// zone nommée : (nom sans casse ni accents, nom écrit, mots de la zone, ligne de sa définition)
type NamedZone = (String, String, Vec<Tok>, usize);

/// `zone "nom"` remplacé par les mots de la zone nommée (la définition `zone "nom" = …` garde son nom)
fn named_zones(toks: &[Tok], zones: &[NamedZone], ln: usize) -> R<Vec<Tok>> {
    if !toks.iter().any(|t| matches!(t, Tok::W(w, _) if w == "zone")) {
        return Ok(toks.to_vec());
    }
    let mut out = vec![];
    let mut i = 0;
    if matches!(toks, [Tok::W(w, _), Tok::S(_), Tok::Eq, ..] if w == "zone") {
        out.extend_from_slice(&toks[..3]);
        i = 3;
    }
    while i < toks.len() {
        if let (Tok::W(w, _), Some(Tok::S(name))) = (&toks[i], toks.get(i + 1)) {
            if w == "zone" {
                let key = norm(name);
                let Some(z) = zones.iter().find(|z| z.0 == key) else {
                    let known: Vec<&str> = zones.iter().map(|z| z.1.as_str()).collect();
                    let hint = if known.is_empty() { String::new() } else { format!(" (zones nommées : {})", known.join(", ")) };
                    return Err(ScriptError { line: ln, msg: format!("zone « {name} » inconnue : il faut d'abord une ligne « zone \"{name}\" = … »{hint}") });
                };
                out.extend(z.2.iter().cloned());
                i += 2;
                continue;
            }
        }
        out.push(toks[i].clone());
        i += 1;
    }
    Ok(out)
}

/// `zone "nom" = <zone>` : la zone, vérifiée, retenue par ses mots
fn define_zone(name: &str, rest: &[Tok], ln: usize, zones: &mut Vec<NamedZone>, inc: &Includes) -> R<()> {
    let err = |m: String| Err(ScriptError { line: ln, msg: m });
    if name.trim().is_empty() {
        return err("nom de zone vide".into());
    }
    if rest.iter().any(|t| matches!(t, Tok::Calc(_))) {
        return err("une zone nommée est fixée à la lecture : pour la calculer, $(…) avec des paramètres (pas {…}, relu pendant l'expérience)".into());
    }
    let key = norm(name);
    if let Some(z) = zones.iter().find(|z| z.0 == key) {
        return err(format!("zone « {name} » déjà nommée ({})", inc.place(z.3)));
    }
    let mut q = P { t: rest, i: 0, ln, vars: vec![], refs: vec![], decls: vec![] };
    q.zone()?;
    q.end()?;
    zones.push((key, name.to_string(), rest.to_vec(), ln));
    Ok(())
}

/// d'où vient une ligne : du protocole (None), ou de quel composant inclus où
fn origin_key(inc: &Includes, l: usize) -> Option<(usize, String, String)> {
    inc.get(l).map(|o| (o.line, o.via.clone(), o.file.clone()))
}

/// Lit les lignes (numéro, texte) d'un protocole développé (`mp_compose::expand`).
fn parse_lines(lines: &[(usize, String)], source: String, includes: Includes, variant: Option<(String, String)>) -> Result<Protocol, Vec<ScriptError>> {
    let mut proto = Protocol { source, variant, includes, ..Default::default() };
    let mut errs = vec![];
    // blocs ouverts (tant que, répéter, si, quand … fin) : (étape d'ouverture, ligne, dernière branche d'un « si ») ; usize::MAX pour
    // une ligne d'ouverture mal écrite (son « fin » la ferme sans autre erreur)
    let mut open: Vec<(usize, usize, usize)> = vec![];
    let mut zones: Vec<NamedZone> = vec![];
    // objets nommés posés (genre, nom) ; lignes qui en désignent un (genre, nom, ligne)
    let mut decls: Vec<(Kind, String)> = vec![];
    let mut refs: Vec<(Kind, String, usize)> = vec![];
    for (ln, l) in lines {
        let ln = *ln;
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
        let toks = match named_zones(&toks, &zones, ln) {
            Ok(t) => t,
            Err(e) => {
                errs.push(e);
                continue;
            }
        };
        if let [Tok::W(w, _), Tok::S(name), Tok::Eq, rest @ ..] = toks.as_slice() {
            if w == "zone" {
                if let Err(e) = define_zone(name, rest, ln, &mut zones, &proto.includes) {
                    errs.push(e);
                }
                continue;
            }
        }
        let mut p = P { t: &toks, i: 0, ln, vars: proto.vars.clone(), refs: vec![], decls: vec![] };
        let n0 = proto.steps.len();
        if let Err(e) = parse_line(&mut p, &mut proto) {
            let opens = match toks.first() {
                Some(Tok::W(w, _)) => matches!(w.as_str(), "tant" | "repeter" | "si") || (w == "quand" && !toks.contains(&Tok::Colon)),
                _ => false,
            };
            if opens {
                open.push((usize::MAX, ln, usize::MAX));
            }
            errs.push(e);
        }
        refs.extend(p.refs.drain(..).map(|(c, n)| (c, n, ln)));
        decls.append(&mut p.decls);
        for s in n0..proto.steps.len() {
            close_blocks(&mut proto, &mut open, &mut errs, s, ln);
        }
    }
    for (w, wl, _) in open {
        if w != usize::MAX {
            errs.push(ScriptError { line: wl, msg: format!("« {} » sans son « fin »", block_name(&proto.steps[w].0)) });
        }
    }
    // un bloc s'ouvre et se ferme dans le même fichier
    for (st, ln) in &proto.steps {
        if let Step::End { start } = st {
            if *start < proto.steps.len() && origin_key(&proto.includes, proto.steps[*start].1) != origin_key(&proto.includes, *ln) {
                errs.push(ScriptError {
                    line: *ln,
                    msg: "ce « fin » ferme un bloc ouvert dans un autre fichier : un bloc s'ouvre et se ferme dans le même fichier (protocole ou composant)".into(),
                });
            }
        }
    }
    // objets nommés que des lignes désignent : posés par le protocole (un état de départ enregistré peut en apporter d'autres) ; une
    // fois les autres erreurs corrigées (une ligne refusée ne pose pas ses noms), une erreur par nom et par ligne
    refs.dedup();
    if errs.is_empty() && matches!(proto.start, Some((Start::Scenario { .. }, _))) {
        let of = |k: Kind| -> Vec<String> { decls.iter().filter(|d| d.0 == k).map(|d| d.1.clone()).collect() };
        let cranks = of(Kind::Crank);
        let mut mechs = of(Kind::Mech);
        mechs.extend(cranks.iter().cloned());
        let wheels = of(Kind::Wheel);
        let mut bodies = of(Kind::Body);
        for c in &cranks {
            bodies.extend(super::mp_mechanism::CRANK_PARTS.iter().map(|part| format!("{c}.{part}")));
        }
        for (kind, name, ln) in &refs {
            let (known, what, line) = match kind {
                Kind::Crank => (&cranks, "aucun vilebrequin", "il faut « nom \"X\" » sur sa ligne « vilebrequin … »"),
                Kind::Mech => (&mechs, "aucun mécanisme", "il faut d'abord sa ligne « mécanisme \"X\" » (ou « vilebrequin … nom \"X\" »)"),
                Kind::Wheel => (&wheels, "aucune roue", "il faut « nom \"X\" » sur sa ligne « roue … »"),
                Kind::Body => (&bodies, "aucun corps", "il faut « nom \"X\" » sur sa ligne (« corps », « pièce », « roue », ou son vilebrequin)"),
            };
            if !known.iter().any(|k| norm(k) == norm(name)) {
                let hint = if known.is_empty() { String::new() } else { format!(" ; noms posés : {}", known.join(", ")) };
                errs.push(ScriptError { line: *ln, msg: format!("{what} ne s'appelle « {name} » : {}{hint}", line.replace('X', name)) });
            }
        }
    }
    if let (None, Some((_, l))) = (proto.period, proto.measures.first()) {
        errs.push(ScriptError {
            line: *l,
            msg: "des colonnes de mesure sans période : il faut une ligne « mesurer toutes les <durée> » (seule, ou suivie de ses mesures)".into(),
        });
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
/// contiennent des virgules), `$NOM` remplacé dans le texte. Les numéros de ligne sont ceux du texte d'origine. Composants cherchés
/// depuis le dossier courant : voir `parse_all_in`.
pub fn parse_all(text: &str) -> Result<Vec<Protocol>, Vec<ScriptError>> {
    parse_all_in(text, Path::new("."))
}

/// Comme `parse_all`, les composants (`inclure "fichier.exp"`) cherchés depuis le dossier `base` (celui du protocole).
pub fn parse_all_in(text: &str, base: &Path) -> Result<Vec<Protocol>, Vec<ScriptError>> {
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
    let Some((_, name, values)) = sweep else {
        return parse_in(text, base, None).map(|p| vec![p]);
    };
    let mut out = vec![];
    for v in &values {
        out.push(parse_in(text, base, Some((name.clone(), v.clone())))?);
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
            matches!(norm(&first).as_str(), "simuler" | "avancer" | "attendre" | "tant" | "repeter" | "si")
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
