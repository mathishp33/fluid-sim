//! Composants des protocoles d'expérience (EXPERIENCES.md, « Composants ») : paramètres `paramètre NOM = valeur`, valeurs `$NOM` et
//! `$(expression)` remplacées à la lecture, fichiers `inclure "fichier.exp" avec NOM = valeur, …` développés à leur place — un même
//! composant (un moteur, un réservoir, une soupape) sert dans plusieurs expériences, ou plusieurs fois dans une seule.
//!
//! Le résultat est ce que `mp_script` lit, ligne à ligne : les lignes du protocole gardent leur numéro, celles des composants sont
//! numérotées à leur suite (`Includes`), pour que les erreurs et le journal disent d'où elles viennent.

use super::mp_experiment::{norm, IncludedLine, Includes};
use super::mp_script::{si_number, ScriptError};
use std::path::{Path, PathBuf};

/// niveaux d'inclusions imbriquées, au plus
const MAX_DEPTH: usize = 16;

/// Protocole développé.
pub(crate) struct Expanded {
    /// (numéro, texte) de chaque ligne à lire, dans l'ordre
    pub lines: Vec<(usize, String)>,
    pub includes: Includes,
    /// texte complet, valeurs remplacées et composants développés : une copie du protocole qui se relance seule
    pub text: String,
}

/// paramètres d'un fichier : (nom, valeur telle qu'écrite), dans l'ordre de leur déclaration
type Scope = Vec<(String, String)>;

/// fichier en cours de lecture
struct Ctx {
    /// ligne du texte du protocole : l'« inclure » de premier niveau (0 dans le protocole lui-même)
    top: usize,
    /// fichier tel qu'écrit dans son « inclure » (vide pour le protocole) et inclusions qui y mènent
    file: String,
    via: String,
    /// dossier où chercher les fichiers qu'il inclut
    dir: PathBuf,
    depth: usize,
    /// fichiers ouverts, pour une inclusion circulaire
    stack: Vec<PathBuf>,
}

impl Ctx {
    /// erreur à la ligne `ln` de ce fichier : à cette ligne dans le protocole, sinon à la ligne de son « inclure », située
    fn err(&self, ln: usize, msg: String) -> ScriptError {
        if self.file.is_empty() {
            ScriptError { line: ln, msg }
        } else {
            ScriptError { line: self.top, msg: format!("{}« {} » ligne {ln} : {msg}", self.via, self.file) }
        }
    }
}

/// Développe le protocole `text` (dossier `base`) ; `sweep` : (nom, valeur) de la variante d'un balayage.
pub(crate) fn expand(text: &str, base: &Path, sweep: Option<(&str, &str)>) -> Result<Expanded, Vec<ScriptError>> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Expanded { lines: vec![], includes: Includes { first: lines.len() + 1, lines: vec![] }, text: String::new() };
    let mut errs = vec![];
    let args: Vec<(String, String)> = sweep.map(|(n, v)| vec![(n.to_string(), v.to_string())]).unwrap_or_default();
    let ctx = Ctx { top: 0, file: String::new(), via: String::new(), dir: base.to_path_buf(), depth: 0, stack: vec![] };
    expand_lines(&lines, &args, &ctx, &mut out, &mut errs);
    if errs.is_empty() { Ok(out) } else { Err(errs) }
}

/// ligne `ln` du fichier en cours : `text` à lire ; `copy` : à écrire aussi dans la copie du protocole
fn emit(out: &mut Expanded, ctx: &Ctx, ln: usize, text: String, copy: Option<&str>) {
    if let Some(c) = copy {
        out.text.push_str(c);
        out.text.push('\n');
    }
    if ctx.file.is_empty() {
        out.lines.push((ln, text));
    } else {
        let n = out.includes.first + out.includes.lines.len();
        out.includes.lines.push(IncludedLine { line: ctx.top, file: ctx.file.clone(), at: ln, via: ctx.via.clone(), text: text.clone() });
        out.lines.push((n, text));
    }
}

/// Lignes d'un fichier : ses paramètres (valeurs de `args`, sinon celles par défaut), puis chaque ligne, valeurs remplacées et
/// composants inclus développés.
fn expand_lines(lines: &[&str], args: &[(String, String)], ctx: &Ctx, out: &mut Expanded, errs: &mut Vec<ScriptError>) {
    let main = ctx.file.is_empty();
    let decls: Vec<(usize, Result<(String, Option<String>), String>)> =
        lines.iter().enumerate().filter_map(|(k, l)| param_line(l).map(|r| (k + 1, r))).collect();
    let names: Vec<String> = decls.iter().filter_map(|(_, r)| r.as_ref().ok().map(|(n, _)| n.clone())).collect();
    let mut scope: Scope = vec![];
    for (n, v) in args {
        if names.contains(n) {
            continue;
        }
        if main {
            // variable d'un balayage qui n'est pas un paramètre déclaré
            scope.push((n.clone(), v.clone()));
        } else {
            let known = if names.is_empty() { "aucun".to_string() } else { names.join(", ") };
            errs.push(ScriptError { line: ctx.top, msg: format!("{}« {} » n'a pas de paramètre « {n} » (paramètres : {known})", ctx.via, ctx.file) });
        }
    }
    let mut declared: Vec<String> = vec![];
    for (ln, r) in decls {
        let (name, default) = match r {
            Err(m) => {
                errs.push(ctx.err(ln, m));
                continue;
            }
            Ok(p) => p,
        };
        if declared.contains(&name) {
            errs.push(ctx.err(ln, format!("paramètre « {name} » déjà déclaré")));
            continue;
        }
        declared.push(name.clone());
        let value = match (args.iter().find(|(n, _)| *n == name), default) {
            (Some((_, v)), _) => v.clone(),
            (None, Some(d)) => match substitute(&d, &scope) {
                Ok(v) => v,
                Err(m) => {
                    errs.push(ctx.err(ln, m));
                    continue;
                }
            },
            (None, None) => {
                errs.push(ctx.err(
                    ln,
                    if main {
                        format!("paramètre « {name} » sans valeur : un protocole lancé seul donne une valeur à chacun (« paramètre {name} = … »)")
                    } else {
                        format!("paramètre « {name} » sans valeur : il faut « avec {name} = … » sur la ligne « inclure »")
                    },
                ));
                continue;
            }
        };
        scope.push((name, value));
    }
    for (k, l) in lines.iter().enumerate() {
        let ln = k + 1;
        let first = first_word(l);
        match first.as_str() {
            "parametre" => {
                // (dans la copie : sa valeur)
                let copy = match param_line(l) {
                    Some(Ok((n, _))) => match scope.iter().find(|(m, _)| *m == n) {
                        Some((_, v)) => format!("# paramètre {n} = {v}"),
                        None => format!("# {}", l.trim()),
                    },
                    _ => format!("# {}", l.trim()),
                };
                emit(out, ctx, ln, String::new(), Some(&copy));
                continue;
            }
            "balayage" => {
                if !main {
                    errs.push(ctx.err(ln, "« balayage » : dans le protocole seulement ; un composant reçoit ses valeurs par « avec » sur la ligne « inclure »".into()));
                }
                // la variante en cours : sa valeur, en commentaire (le balayage est lu par `parse_all_in`)
                let shown = match args.first() {
                    Some((n, v)) if main => format!("# balayage {n} = {v}"),
                    _ => l.to_string(),
                };
                emit(out, ctx, ln, shown.clone(), Some(&shown));
                continue;
            }
            _ => {}
        }
        let s = match substitute(l, &scope) {
            Ok(s) => s,
            Err(m) => {
                errs.push(ctx.err(ln, m));
                emit(out, ctx, ln, String::new(), None);
                continue;
            }
        };
        if first == "inclure" {
            emit(out, ctx, ln, String::new(), None);
            match include_line(&s) {
                Ok((file, a)) => include(&file, &a, ctx, ln, out, errs),
                Err(m) => errs.push(ctx.err(ln, m)),
            }
            continue;
        }
        emit(out, ctx, ln, s.clone(), Some(&s));
    }
}

/// Composant `file` inclus à la ligne `ln` du fichier `parent`, avec les valeurs `args`.
fn include(file: &str, args: &[(String, String)], parent: &Ctx, ln: usize, out: &mut Expanded, errs: &mut Vec<ScriptError>) {
    if parent.depth >= MAX_DEPTH {
        errs.push(parent.err(ln, format!("plus de {MAX_DEPTH} niveaux d'inclusions imbriquées")));
        return;
    }
    let path = parent.dir.join(file);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            errs.push(parent.err(ln, format!("composant « {file} » : {e} (cherché depuis « {} »)", parent.dir.display())));
            return;
        }
    };
    let key = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
    if parent.stack.contains(&key) {
        errs.push(parent.err(ln, format!("inclusion circulaire : « {file} » s'inclut lui-même, directement ou par un autre composant")));
        return;
    }
    let mut stack = parent.stack.clone();
    stack.push(key);
    let ctx = Ctx {
        top: if parent.file.is_empty() { ln } else { parent.top },
        file: file.to_string(),
        via: if parent.file.is_empty() { String::new() } else { format!("{}« {} » ligne {ln} → ", parent.via, parent.file) },
        dir: path.parent().map(Path::to_path_buf).unwrap_or_default(),
        depth: parent.depth + 1,
        stack,
    };
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    let lines: Vec<&str> = text.lines().collect();
    let given: Vec<String> = args.iter().map(|(n, v)| format!("{n} = {v}")).collect();
    let with = if given.is_empty() { String::new() } else { format!(" avec {}", given.join(", ")) };
    out.text.push_str(&format!("# ─── composant « {file} »{with}\n"));
    expand_lines(&lines, args, &ctx, out, errs);
    out.text.push_str(&format!("# ─── fin de « {file} »\n"));
}

/// code d'une ligne, sans son commentaire (un « # » hors des guillemets)
fn code_of(l: &str) -> &str {
    let mut close = None;
    for (i, c) in l.char_indices() {
        match close {
            Some(q) if c == q => close = None,
            Some(_) => {}
            None => match c {
                '"' => close = Some('"'),
                '«' => close = Some('»'),
                '“' => close = Some('”'),
                '#' => return &l[..i],
                _ => {}
            },
        }
    }
    l
}

/// premier mot d'une ligne, sans casse ni accents
fn first_word(l: &str) -> String {
    let w: String = code_of(l).trim_start().chars().take_while(|c| c.is_alphabetic()).collect();
    norm(&w)
}

fn is_name(n: &str) -> bool {
    !n.is_empty() && n.chars().all(|c| c.is_alphanumeric() || c == '_') && !n.starts_with(|c: char| c.is_ascii_digit())
}

/// `paramètre NOM [= valeur]` : (nom, valeur par défaut) ; None si la ligne n'en est pas une
fn param_line(l: &str) -> Option<Result<(String, Option<String>), String>> {
    if first_word(l) != "parametre" {
        return None;
    }
    let code = code_of(l).trim_start();
    let kw: String = code.chars().take_while(|c| c.is_alphabetic()).collect();
    let rest = code[kw.len()..].trim();
    let (name, value) = match rest.split_once('=') {
        Some((n, v)) => (n.trim(), Some(v.trim().to_string())),
        None => (rest, None),
    };
    let name = name.trim_start_matches('$');
    if !is_name(name) {
        return Some(Err("il faut « paramètre NOM = valeur » (NOM : lettres, chiffres et _, sans espace)".into()));
    }
    if value.as_deref() == Some("") {
        return Some(Err(format!("paramètre « {name} » : il manque la valeur après « = »")));
    }
    Some(Ok((name.to_string(), value)))
}

/// éléments séparés par des virgules (ou des points-virgules) hors des parenthèses et des guillemets
fn split_top(s: &str) -> Vec<String> {
    let (mut out, mut cur, mut depth, mut close) = (vec![], String::new(), 0i32, None);
    for c in s.chars() {
        match close {
            Some(q) if c == q => close = None,
            Some(_) => {}
            None => match c {
                '"' => close = Some('"'),
                '«' => close = Some('»'),
                '“' => close = Some('”'),
                '(' => depth += 1,
                ')' => depth -= 1,
                ',' | ';' if depth == 0 => {
                    out.push(std::mem::take(&mut cur));
                    continue;
                }
                _ => {}
            },
        }
        cur.push(c);
    }
    out.push(cur);
    out.into_iter().map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect()
}

/// `inclure "fichier" [avec NOM = valeur, …]` (valeurs déjà remplacées) : (fichier, valeurs)
fn include_line(s: &str) -> Result<(String, Vec<(String, String)>), String> {
    let usage = "il faut « inclure \"fichier.exp\" [avec NOM = valeur, NOM2 = valeur2…] »";
    let code = code_of(s).trim_start();
    let kw: String = code.chars().take_while(|c| c.is_alphabetic()).collect();
    let rest = code[kw.len()..].trim_start();
    let Some(open) = rest.chars().next() else { return Err(usage.into()) };
    let close = match open {
        '"' => '"',
        '«' => '»',
        '“' => '”',
        _ => return Err(usage.into()),
    };
    let body = &rest[open.len_utf8()..];
    let Some(end) = body.find(close) else { return Err("guillemet non refermé".into()) };
    let file = body[..end].trim().to_string();
    if file.is_empty() {
        return Err("nom de fichier vide".into());
    }
    let tail = body[end + close.len_utf8()..].trim();
    let mut args: Vec<(String, String)> = vec![];
    if tail.is_empty() {
        return Ok((file, args));
    }
    let kw: String = tail.chars().take_while(|c| c.is_alphabetic()).collect();
    if norm(&kw) != "avec" {
        return Err(format!("après le fichier, il faut « avec NOM = valeur, … » (trouvé « {tail} »)"));
    }
    for item in split_top(&tail[kw.len()..]) {
        let Some((n, v)) = item.split_once('=') else { return Err(format!("« {item} » : il faut NOM = valeur")) };
        let (n, v) = (n.trim().trim_start_matches('$'), v.trim());
        if !is_name(n) {
            return Err(format!("nom de paramètre « {n} » : lettres, chiffres et _, sans espace"));
        }
        if v.is_empty() {
            return Err(format!("« {n} » : il manque la valeur après « = »"));
        }
        if args.iter().any(|(m, _)| m == n) {
            return Err(format!("« {n} » donné deux fois"));
        }
        args.push((n.to_string(), v.to_string()));
    }
    if args.is_empty() {
        return Err("« avec » sans valeur".into());
    }
    Ok((file, args))
}

/// valeur sans ses guillemets si elle n'est qu'un texte entre guillemets
fn unquote(v: &str) -> &str {
    let t = v.trim();
    for (a, b) in [('"', '"'), ('«', '»'), ('“', '”')] {
        if t.len() >= a.len_utf8() + b.len_utf8() && t.starts_with(a) && t.ends_with(b) {
            let inner = &t[a.len_utf8()..t.len() - b.len_utf8()];
            if !inner.contains(b) {
                return inner.trim();
            }
        }
    }
    t
}

fn unknown(name: &str, scope: &Scope) -> String {
    let known = if scope.is_empty() { String::new() } else { format!(" ; paramètres : {}", scope.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", ")) };
    format!("valeur « ${name} » inconnue : il faut une ligne « paramètre {name} = … » (ou un « balayage {name} = … » dans le protocole){known}")
}

/// `$NOM` et `$(expression)` remplacés dans le code d'une ligne (pas dans son commentaire) : `$NOM` par sa valeur telle qu'écrite
/// (entre guillemets, sans ses guillemets), `$(…)` par la valeur de l'expression
pub(crate) fn substitute(line: &str, scope: &Scope) -> Result<String, String> {
    if !line.contains('$') {
        return Ok(line.to_string());
    }
    let cs: Vec<char> = line.chars().collect();
    let mut out = String::new();
    let mut close: Option<char> = None;
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        match close {
            Some(q) if c == q => close = None,
            None if c == '"' => close = Some('"'),
            None if c == '«' => close = Some('»'),
            None if c == '“' => close = Some('”'),
            None if c == '#' => {
                out.extend(&cs[i..]);
                break;
            }
            _ => {}
        }
        if c != '$' {
            out.push(c);
            i += 1;
            continue;
        }
        if cs.get(i + 1) == Some(&'(') {
            let mut depth = 0;
            let mut j = i + 1;
            while j < cs.len() {
                match cs[j] {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    _ => {}
                }
                j += 1;
            }
            if j >= cs.len() {
                return Err("« $( » sans sa parenthèse fermante".into());
            }
            let expr: String = cs[i + 2..j].iter().collect();
            out.push_str(&fmt_value(eval(&expr, scope)?));
            i = j + 1;
            continue;
        }
        let name: String = cs[i + 1..].iter().take_while(|c| c.is_alphanumeric() || **c == '_').collect();
        if name.is_empty() {
            out.push(c);
            i += 1;
            continue;
        }
        let Some((_, v)) = scope.iter().find(|(n, _)| *n == name) else { return Err(unknown(&name, scope)) };
        out.push_str(if close.is_some() { unquote(v) } else { v.trim() });
        i += 1 + name.chars().count();
    }
    Ok(out)
}

/// valeur d'une expression `$(…)` : formule (+ − × / ^, parenthèses, fonctions des formules) en nombres SI et en paramètres (leur
/// valeur SI : « 5 cm » vaut 0.05 ; angles et vitesses de rotation en degrés, comme leurs relevés)
fn eval(expr: &str, scope: &Scope) -> Result<f64, String> {
    let cs: Vec<char> = expr.chars().collect();
    let mut s = String::new();
    let mut i = 0;
    while i < cs.len() {
        let c = cs[i];
        if c == '$' {
            i += 1;
            continue;
        }
        // un nom (pas la fin d'un nombre : « 1e-3 »)
        let in_number = s.chars().last().is_some_and(|p| p.is_ascii_digit() || p == '.');
        if (c.is_alphabetic() || c == '_') && !in_number {
            let w: String = cs[i..].iter().take_while(|c| c.is_alphanumeric() || **c == '_').collect();
            i += w.chars().count();
            match scope.iter().find(|(n, _)| *n == w) {
                Some((_, v)) => {
                    let x = si_number(unquote(v)).ok_or_else(|| {
                        format!("$({expr}) : « {w} » vaut « {} », pas un nombre (un nombre, et son unité s'il en a une)", v.trim())
                    })?;
                    s.push_str(&format!("({x})"));
                }
                None => s.push_str(&w),
            }
            continue;
        }
        s.push(c);
        i += 1;
    }
    let f = super::mp_fields::Formula::parse_vars(&norm(&s), &[]).map_err(|e| format!("$({expr}) : {e} (nombres en unités SI, sans unité écrite)"))?;
    let v = f.eval_vars(&[]);
    if !v.is_finite() {
        return Err(format!("$({expr}) vaut {v} : il faut un nombre fini"));
    }
    Ok(v)
}

/// nombre écrit à la place d'une expression : 12 chiffres significatifs (0.1 + 0.2 s'écrit 0.3)
fn fmt_value(v: f64) -> String {
    let r: f64 = format!("{v:.11e}").parse().unwrap_or(v);
    format!("{}", if r == 0.0 { 0.0 } else { r })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope(v: &[(&str, &str)]) -> Scope {
        v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }

    #[test]
    fn values_and_expressions() {
        let s = scope(&[("X", "5 cm"), ("NOM", "\"gauche\""), ("N", "3"), ("V", "600 tr/min")]);
        assert_eq!(substitute("disque ($X, 0.1) rayon 1 cm", &s).unwrap(), "disque (5 cm, 0.1) rayon 1 cm");
        assert_eq!(substitute("vilebrequin … nom $NOM", &s).unwrap(), "vilebrequin … nom \"gauche\"");
        assert_eq!(substitute("corps \"$NOM.piston\" fy", &s).unwrap(), "corps \"gauche.piston\" fy");
        assert_eq!(substitute("rect ($(X - 0.01), 0) ($(X + 0.01), $(N * 0.1))", &s).unwrap(), "rect (0.04, 0) (0.06, 0.3)");
        assert_eq!(substitute("couple \"-0.5*max(rotation - $(V), 0)\"", &s).unwrap(), "couple \"-0.5*max(rotation - 3600, 0)\"");
        assert_eq!(substitute("x $(2*$N + 1e-3) # $INCONNU en commentaire", &s).unwrap(), "x 6.001 # $INCONNU en commentaire");
        assert_eq!(substitute("prix en $ : 3", &s).unwrap(), "prix en $ : 3");
        assert!(substitute("disque ($Y, 0)", &s).unwrap_err().contains("« $Y » inconnue"));
        assert!(substitute("$(NOM + 1)", &s).unwrap_err().contains("pas un nombre"));
        assert!(substitute("$(X + 1 cm)", &s).unwrap_err().contains("unités SI"));
        assert_eq!(fmt_value(0.1 + 0.2), "0.3");
        assert_eq!(fmt_value(-0.0), "0");
    }

    #[test]
    fn include_lines() {
        let (f, a) = include_line("inclure \"composants/moteur.exp\" avec NOM = \"g\", P = (0.1, 0.2), Z = rect (0, 0) (1, 1) # c").unwrap();
        assert_eq!(f, "composants/moteur.exp");
        assert_eq!(a, vec![("NOM".into(), "\"g\"".into()), ("P".into(), "(0.1, 0.2)".into()), ("Z".into(), "rect (0, 0) (1, 1)".into())]);
        assert_eq!(include_line("inclure « a.exp »").unwrap(), ("a.exp".to_string(), vec![]));
        assert!(include_line("inclure a.exp").is_err());
        assert!(include_line("inclure \"a.exp\" avec X = 1, X = 2").unwrap_err().contains("deux fois"));
        assert!(include_line("inclure \"a.exp\" pour X = 1").is_err());
        assert_eq!(param_line("paramètre X0 = 0.05 # défaut"), Some(Ok(("X0".into(), Some("0.05".into())))));
        assert_eq!(param_line("parametre NOM"), Some(Ok(("NOM".into(), None))));
        assert!(matches!(param_line("paramètre 2X = 1"), Some(Err(_))));
        assert_eq!(param_line("remplir partout"), None);
    }
}
