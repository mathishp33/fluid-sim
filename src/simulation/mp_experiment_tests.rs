//! Tests des expériences reproductibles : lecture du script, zones, champs de valeurs, déroulement exact et reproductible, sorties,
//! ligne de commande.

use std::path::{Path, PathBuf};
use std::time::Duration;

use super::mp_experiment::{
    apply_action, fmt_num, initial_state, Action, BoundarySpec, FieldQty, FieldSource, Measure, Quantity, Reduce, RunState, Runner, Start, Step,
    Zone,
};
use super::mp_fields::{read_csv, Formula};
use super::mp_script::{insert_initial_line, parse, parse_all, zone_text};
use super::mp_view::{brush_cells, build_scenario, MpScenario};
use super::multiphase::{Bc, MultiphaseSim};

fn proto(text: &str) -> super::mp_experiment::Protocol {
    match parse_all(text) {
        Ok(mut v) => v.remove(0),
        Err(e) => panic!("{}", e.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n")),
    }
}

/// dossier temporaire propre à un test
fn tmp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("fluid-sim-exp-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// exécute un protocole jusqu'au bout (les pauses sont reprises)
fn run_to_end(r: &mut Runner) {
    loop {
        match r.advance(Duration::from_secs(3600)).clone() {
            RunState::Running => {}
            RunState::Paused(_) => r.resume(),
            RunState::Finished => return,
            RunState::Failed(e) => panic!("échec : {e}\n{}", r.log.join("\n")),
        }
    }
}

fn run(text: &str, out: Option<PathBuf>) -> Runner {
    let mut r = Runner::new(proto(text), Path::new("."), out, Some(false)).unwrap_or_else(|e| panic!("{e}"));
    run_to_end(&mut r);
    r
}

#[test]
fn script_reads_commands_units_and_reports_errors_with_their_line() {
    let text = "\
# essai complet
départ scénario \"barrage\" grille 40 x 30 maille 1 cm gravité 9.81 m/s²
mesurer toutes les 10 ms : stats, T en (20 cm, 10 cm), moyenne p dans rect (0, 0) (0.1, 0.05), masse \"eau\" comme \"eau totale\"
remplir disque (0.2, 0.1) rayon 2 cm avec \"eau\" à 20 °C
température rect (0, 0) (0.4, 0.1) 350 K
solide rect (0.3, 0) (0.32, 0.05) \"acier\" isotherme 400 K
chauffer disque (0.1, 0.2) rayon 1 cm à 600 K pendant 0.5 s
accélérer rect (0, 0) (0.1, 0.05) de (5, 0) m/s² pendant 200 ms
vitesse point (0.2, 0.2) (0.5, -0.5) m/s
bord haut ouvert
bord gauche entrée (0.3, 0) m/s à 15 °C \"eau\"
champ T formule \"300 + 50*x/lx\" dans rect (0, 0) (0.4, 0.3)
champ fraction \"eau\" fichier \"f.csv\" origine bas
gravité (0, -9.81)
pas max 2 ms
simuler 1 s
marque \"fin de la chauffe\"
sauver état \"fin\"
simuler jusqu'à 2.5 s
";
    let p = proto(text);
    assert_eq!(p.start, Some((Start::Scenario { name: "barrage".into(), nx: 40, ny: 30, dx: 0.01, g: 9.81 }, 2)));
    assert_eq!(p.period, Some(0.01));
    assert_eq!(p.measures.len(), 4);
    assert_eq!(p.measures[1].0, Measure::Probe { q: Quantity::T, reduce: Reduce::At, zone: Zone::Point([0.2, 0.1]), label: None });
    assert!(matches!(&p.measures[2].0, Measure::Probe { q: Quantity::P, reduce: Reduce::Mean, .. }));
    assert!(matches!(&p.measures[3].0, Measure::Mass { label: Some(l), .. } if l == "eau totale"));
    let acts: Vec<&Action> = p.steps.iter().filter_map(|(s, _)| if let Step::Do(a) = s { Some(a) } else { None }).collect();
    match acts[0] {
        Action::Fill { zone: Zone::Disque { c, r }, fluid, temp } => {
            assert_eq!((*c, *r, fluid.as_str()), ([0.2, 0.1], 0.02, "eau"));
            assert!((temp.unwrap() - 293.15).abs() < 1e-9);
        }
        a => panic!("{a:?}"),
    }
    assert!(matches!(acts[3], Action::Heat { temp, duration: Some(d), .. } if *temp == 600.0 && *d == 0.5));
    assert!(matches!(acts[4], Action::Accelerate { a, duration: Some(d), .. } if *a == [5.0, 0.0] && (*d - 0.2).abs() < 1e-12));
    assert!(matches!(acts[5], Action::Velocity { v, .. } if *v == [0.5, -0.5]));
    assert!(matches!(acts[6], Action::Boundary { side: 3, spec: BoundarySpec::Open(p) } if *p == 101325.0));
    assert!(matches!(acts[7], Action::Boundary { side: 0, spec: BoundarySpec::Inlet { v, temp: Some(t), fluid: Some(f) } } if *v == [0.3, 0.0] && (*t - 288.15).abs() < 1e-9 && f == "eau"));
    assert!(matches!(acts[8], Action::Field { qty: FieldQty::T, source: FieldSource::Formula(_), .. }));
    assert!(matches!(acts[9], Action::Field { qty: FieldQty::Fraction(_), source: FieldSource::File { top_origin: false, .. }, zone: Zone::Partout, .. }));
    assert!(matches!(acts[11], Action::DtMax(d) if (*d - 0.002).abs() < 1e-15));
    assert_eq!(p.steps.iter().filter(|(s, _)| matches!(s, Step::Simulate(d) if *d == 1.0)).count(), 1);
    assert!(matches!(p.steps.last().unwrap(), (Step::Until(t), 19) if *t == 2.5));
    assert_eq!(p.planned_end(0.0), 2.5);
    assert_eq!(p.first_advance(), 12);

    // erreurs : toutes, chacune avec sa ligne
    let bad = "départ scénario \"barrage\"\nremplir disque (0.2, 0.1) rayon 2,5 cm avec \"eau\"\nvoler partout\nchauffer partout à 300 m\nremplir partout avec \"eau";
    let errs = parse(bad, None).unwrap_err();
    let lines: Vec<usize> = errs.iter().map(|e| e.line).collect();
    assert_eq!(lines, vec![2, 3, 4, 5], "{errs:?}");
    assert!(errs[0].msg.contains("rayon") || errs[0].msg.contains("virgule"), "{}", errs[0]);
    assert!(errs[1].msg.contains("inconnue"));
    assert!(errs[2].msg.contains("température"));
    assert!(errs[3].msg.contains("guillemet"));
    // pas de départ
    assert!(parse("simuler 1 s", None).unwrap_err()[0].msg.contains("départ"));
    // nom inconnu : signalé au lancement, avec sa ligne et la liste des noms
    let e = Runner::new(proto("départ scénario \"barrage\" grille 16 x 16\nremplir partout avec \"mercure\"\nsimuler 1 ms"), Path::new("."), None, Some(false)).err().unwrap();
    assert!(e.starts_with("ligne 2") && e.contains("mercure") && e.contains("eau"), "{e}");
}

#[test]
fn a_sweep_gives_one_protocol_per_value_with_the_same_line_numbers() {
    let text = "départ scénario \"cuve vide\" grille 16 x 16\nbalayage T0 = 300 K, 350 K, 40 °C\nremplir partout avec \"eau\" à $T0\nsimuler 1 ms\n";
    let ps = parse_all(text).unwrap();
    assert_eq!(ps.len(), 3);
    let temps: Vec<f64> = ps
        .iter()
        .map(|p| match &p.steps[0] {
            (Step::Do(Action::Fill { temp: Some(t), .. }), 3) => *t,
            s => panic!("{s:?}"),
        })
        .collect();
    assert_eq!(temps[..2], [300.0, 350.0]);
    assert!((temps[2] - 313.15).abs() < 1e-9);
    assert_eq!(ps[1].variant, Some(("T0".into(), "350 K".into())));
    assert!(parse_all("départ scénario \"cuve vide\"\nremplir partout avec \"eau\" à $T1").is_err());
}

#[test]
fn zones_take_the_cells_whose_centre_is_inside_and_a_disc_reproduces_the_brush() {
    let (nx, ny, dx) = (40usize, 30usize, 0.01f64);
    let set = |v: Vec<(usize, usize)>| {
        let mut v = v;
        v.sort();
        v
    };
    for (ci, cj, r) in [(10usize, 12usize, 0usize), (5, 5, 1), (20, 15, 3), (0, 29, 4), (39, 0, 6), (7, 22, 12)] {
        let c = [(ci as f64 + 0.5) * dx, (cj as f64 + 0.5) * dx];
        let z = Zone::Disque { c, r: r as f64 * dx };
        assert_eq!(set(z.cells(nx, ny, dx)), set(brush_cells(nx, ny, ci, cj, r, true)), "disque r = {r} en ({ci}, {cj})");
        // relu depuis son écriture dans le script : mêmes mailles
        let text = format!("départ scénario \"cuve vide\"\nremplir {} avec \"eau\"", zone_text(&z));
        let p = proto(&text);
        let Step::Do(Action::Fill { zone, .. }) = &p.steps[0].0 else { panic!() };
        assert_eq!(set(zone.cells(nx, ny, dx)), set(z.cells(nx, ny, dx)));
    }
    assert_eq!(Zone::Partout.cells(nx, ny, dx).len(), nx * ny);
    assert_eq!(Zone::Rect { a: [0.0, 0.0], b: [0.1, 0.05] }.cells(nx, ny, dx).len(), 10 * 5);
    assert_eq!(Zone::Point([0.215, 0.299]).cells(nx, ny, dx), vec![(21, 29)]);
    // trop petit pour un centre de maille : la maille du centre
    assert_eq!(Zone::Disque { c: [0.103, 0.204], r: 0.001 }.cells(nx, ny, dx), vec![(10, 20)]);
    // hors du domaine : rien
    assert!(Zone::Rect { a: [1.0, 1.0], b: [2.0, 2.0] }.cells(nx, ny, dx).is_empty());
    // trait : les mailles à moins de r de la ligne brisée, sans trou entre deux points éloignés
    let t = Zone::Trait { pts: vec![[0.055, 0.155], [0.255, 0.155]], r: 0.01 };
    let cells = t.cells(nx, ny, dx);
    for i in 5..=25 {
        assert!(cells.contains(&(i, 15)) && cells.contains(&(i, 14)) && cells.contains(&(i, 16)), "trou en {i}");
    }
    assert!(!cells.contains(&(15, 17)));
    assert_eq!(fmt_num(0.155), "0.155");
    assert_eq!(fmt_num(2.0), "2");
}

#[test]
fn value_fields_from_csv_png_and_formulas_set_a_consistent_initial_state() {
    let dir = tmp_dir("champs");
    let mut s = build_scenario(MpScenario::Empty, 8, 8, 0.01, 9.81);
    let air = s.ambient_phase;
    let eau = s.reg.phases.iter().position(|p| p.name == "eau").unwrap();
    // CSV 2 × 2 sur tout le domaine : 300 | 400 (haut), 500 | case vide (bas) ; au centre des cases, valeurs exactes
    std::fs::write(dir.join("t.csv"), "300; 400\n500;\n").unwrap();
    let g = read_csv("300; 400\n500;\n").unwrap();
    assert_eq!((g.w, g.h), (2, 2));
    assert!(g.v[3].is_nan());
    let t0 = s.temp.clone();
    let field = |q: FieldQty, src: FieldSource| Action::Field { qty: q, source: src, zone: Zone::Partout, temp: None };
    let n = apply_action(&mut s, &field(FieldQty::T, FieldSource::File { path: "t.csv".into(), top_origin: true, range: None }), &dir).unwrap();
    assert!(n > 0);
    let t_at = |s: &MultiphaseSim, i: usize, j: usize| s.temp[s.idx(i, j)];
    // maille (1, 6) (en haut à gauche) : centre de la case 300 → 300 ; maille (5, 6) : entre 300 et 400 (bilinéaire)
    assert!((t_at(&s, 1, 6) - 300.0).abs() < 1e-9, "{}", t_at(&s, 1, 6));
    assert!(t_at(&s, 4, 6) > 300.0 && t_at(&s, 4, 6) < 400.0);
    assert!((t_at(&s, 1, 1) - 500.0).abs() < 1e-9);
    // la case vide du bas à droite : la maille la plus proche n'a pas de valeur → inchangée
    assert_eq!(t_at(&s, 7, 0), t0[s.idx(7, 0)]);
    // la fermeture reste juste : Σ α = 1 (les masses suivent la nouvelle température)
    for c in 0..64 {
        let sa: f64 = (0..s.reg.phases.len()).map(|k| s.alpha[k][c]).sum();
        assert!((sa - 1.0).abs() < 1e-9, "Σα = {sa} en {c}");
    }
    // formule : T linéaire en x
    apply_action(&mut s, &field(FieldQty::T, FieldSource::Formula("300 + 100*x/lx".into())), &dir).unwrap();
    assert!((t_at(&s, 0, 3) - 306.25).abs() < 1e-9 && (t_at(&s, 7, 3) - 393.75).abs() < 1e-9);
    // fraction d'eau 0.5 : la moitié de la maille, le reste garde sa composition (air)
    apply_action(
        &mut s,
        &Action::Field { qty: FieldQty::Fraction("eau".into()), source: FieldSource::Formula("si(y - 0.04, 0, 0.5)".into()), zone: Zone::Partout, temp: Some(293.15) },
        &dir,
    )
    .unwrap();
    let c = s.idx(3, 1);
    assert!((s.alpha[eau][c] - 0.5).abs() < 1e-9 && (s.alpha[air][c] - 0.5).abs() < 1e-9);
    assert!(s.alpha[eau][s.idx(3, 6)] < 1e-12);
    // vitesse : faces intérieures seulement (les parois restent fermées)
    apply_action(&mut s, &field(FieldQty::Vx, FieldSource::Formula("0.25".into())), &dir).unwrap();
    assert_eq!(s.u[s.uidx(4, 4)], 0.25);
    assert_eq!(s.u[s.uidx(0, 4)], 0.0);
    // PNG : murs dessinés en noir sur blanc (valeurs 1 0 : noir = 1 = solide), pixels transparents ignorés
    {
        let f = std::fs::File::create(dir.join("murs.png")).unwrap();
        let mut enc = png::Encoder::new(std::io::BufWriter::new(f), 4, 4);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut w = enc.write_header().unwrap();
        let mut px = vec![];
        for y in 0..4 {
            for x in 0..4 {
                let (v, a) = if y == 3 { (0u8, 255u8) } else if x == 3 { (0, 0) } else { (255, 255) };
                px.extend([v, v, v, a]);
            }
        }
        w.write_image_data(&px).unwrap();
    }
    let solid = Action::Field {
        qty: FieldQty::Solid("acier".into()),
        source: FieldSource::File { path: "murs.png".into(), top_origin: true, range: Some((1.0, 0.0)) },
        zone: Zone::Partout,
        temp: None,
    };
    apply_action(&mut s, &solid, &dir).unwrap();
    // la rangée du bas de l'image (y = 3) couvre les deux rangées du bas du domaine 8 × 8
    for i in 0..8 {
        assert!(s.is_solid(s.idx(i, 0)) && s.is_solid(s.idx(i, 1)), "maille ({i}, 0)");
        assert!(!s.is_solid(s.idx(i, 4)));
    }
    assert!(Formula::parse("exp(-x^2) + max(1, 2, 3) * pi").is_ok());
    assert!(Formula::parse("foo(x)").is_err() && Formula::parse("1 +").is_err());
    assert_eq!(Formula::parse("-2^2").unwrap().eval(0.0, 0.0, 1.0, 1.0), -4.0);
}

const TIMING: &str = "\
départ scénario \"barrage\" grille 40 x 30 maille 1 cm
pas max 1 ms
mesurer toutes les 4 ms : stats, T en (0.05, 0.05), max vitesse dans partout
simuler 12.3 ms
chauffer disque (0.05, 0.05) rayon 2 cm à 330 K pendant 7 ms
accélérer rect (0.2, 0) (0.3, 0.05) de (0, 20) pendant 3 ms
simuler 20 ms
";

#[test]
fn actions_land_exactly_on_their_time_and_measuring_does_not_change_the_result() {
    let r = run(TIMING, None);
    assert!((r.sim.time - 0.0323).abs() < 1e-8, "t final {}", r.sim.time);
    let joined = r.log.join("\n");
    assert!(joined.contains("[t = 0.012300 s] ligne 5 : chauffer"), "{joined}");
    assert!(joined.contains("[t = 0.015300 s] fin d'une accélération"), "{joined}");
    assert!(joined.contains("[t = 0.019300 s] fin d'un chauffage"), "{joined}");
    assert!(!r.sim.heat_active && r.sim.special_count() == 0, "chauffage ou accélérateurs restés en place");
    // mesures : au départ, puis au premier pas qui atteint chaque multiple de 4 ms (pas ≤ 1 ms), et à la fin ; temps croissants
    let t: Vec<f64> = r.rows.iter().map(|row| row[0]).collect();
    assert_eq!(t[0], 0.0);
    assert!(t.windows(2).all(|w| w[1] > w[0]) && (t.last().unwrap() - 0.0323).abs() < 1e-8);
    // 0, 4, …, 32 ms, plus la fin (32.3 ms) si le relevé de 32 ms n'est pas déjà tombé sur le dernier pas
    assert!(t.len() == 9 || t.len() == 10, "{t:?}");
    for (k, &tk) in t.iter().enumerate().skip(1).take(8) {
        let due = 0.004 * k as f64;
        assert!(tk >= due - 1e-9 && tk < due + 1e-3 + 1e-9, "relevé {k} à {tk} s pour {due} s : {t:?}");
    }
    assert_eq!(r.columns.len(), r.rows[0].len());
    assert!(r.columns[2].starts_with("T en (0.05, 0.05)"), "{:?}", r.columns);
    // la même expérience sans mesures : état final identique au bit près
    let quiet = run(&TIMING.replace("mesurer toutes les 4 ms : stats, T en (0.05, 0.05), max vitesse dans partout\n", ""), None);
    assert_eq!(bincode::serialize(&quiet.sim).unwrap(), bincode::serialize(&r.sim).unwrap(), "mesurer a changé le résultat");
    // rejouée : mêmes nombres
    let again = run(TIMING, None);
    assert_eq!(again.rows, r.rows);
}

#[test]
fn the_gpu_follows_the_same_protocol_with_the_same_deadlines() {
    let cpu = run(TIMING, None);
    // jeton GPU des tests (au plus deux à la fois) ; sans adaptateur, test ignoré
    let probe = build_scenario(MpScenario::DamBreak, 40, 30, 0.01, 9.81);
    if super::mp_gpu_tests::gpu_or_skip(&probe).is_none() {
        return;
    }
    let mut r = Runner::new(proto(TIMING), Path::new("."), None, Some(true)).unwrap();
    assert!(r.gpu.is_some(), "{}", r.log.join("\n"));
    run_to_end(&mut r);
    assert!((r.sim.time - 0.0323).abs() < 1e-8, "t final {}", r.sim.time);
    let joined = r.log.join("\n");
    for s in ["[t = 0.012300 s] ligne 5 : chauffer", "[t = 0.015300 s] fin d'une accélération", "[t = 0.019300 s] fin d'un chauffage"] {
        assert!(joined.contains(s), "{s}\n{joined}");
    }
    r.sync_host();
    assert!(!r.sim.heat_active && r.sim.special_count() == 0);
    // f32 contre f64 : la température sondée et la vitesse maximale finales restent proches
    let (a, b) = (cpu.rows.last().unwrap(), r.rows.last().unwrap());
    assert!((a[2] - b[2]).abs() < 0.5, "T sondée : CPU {} / GPU {}", a[2], b[2]);
    assert!((a[3] - b[3]).abs() < 0.05 * a[3].abs().max(0.1), "vitesse max : CPU {} / GPU {}", a[3], b[3]);
}

#[test]
fn a_protocol_gives_the_same_numbers_whatever_the_number_of_cpu_threads() {
    super::mp_par::set_threads(1);
    let a = run(TIMING, None);
    super::mp_par::set_threads(3);
    let b = run(TIMING, None);
    super::mp_par::set_threads(0);
    assert_eq!(a.rows, b.rows);
    assert_eq!(bincode::serialize(&a.sim).unwrap(), bincode::serialize(&b.sim).unwrap());
}

#[test]
fn heating_holds_its_target_for_its_duration_then_lets_go() {
    let text = "\
départ scénario \"cuve vide\" grille 12 x 12 maille 1 cm
mesurer toutes les 20 ms : moyenne T dans rect (0.03, 0.03) (0.09, 0.09)
chauffer rect (0.03, 0.03) (0.09, 0.09) à 400 K pendant 0.3 s
simuler 0.3 s
simuler 0.2 s
";
    let p = proto(text);
    let mut r = Runner::new(p, Path::new("."), None, Some(false)).unwrap();
    // jusqu'à la fin du chauffage
    while r.sim.time < 0.299 {
        assert_eq!(r.advance(Duration::from_millis(1)).clone(), RunState::Running);
        if r.sim.time > 0.0 && r.sim.time < 0.299 {
            assert!(r.sim.heat_active, "cible de chauffage perdue à t = {}", r.sim.time);
        }
    }
    run_to_end(&mut r);
    assert!(!r.sim.heat_active);
    let mean: Vec<f64> = r.rows.iter().map(|row| row[2]).collect();
    let at = |t: f64| r.rows.iter().position(|row| row[0] >= t - 1e-9).map(|k| mean[k]).unwrap();
    // l'air chauffé tend vers 400 K (constante de temps 0.2 s) puis se refroidit / se mélange après l'arrêt
    assert!(at(0.3) > 330.0 && at(0.3) < 400.0, "T moyenne à 0.3 s : {}", at(0.3));
    assert!(mean[0] < 300.0);
    assert!(r.sim.energy_input > 0.0);
}

#[test]
fn results_are_written_and_a_saved_state_starts_a_new_experiment() {
    let dir = tmp_dir("sorties");
    let out = dir.join("résultats");
    let text = "\
départ scénario \"barrage\" grille 24 x 18 maille 1 cm
mesurer toutes les 5 ms : T en (0.05, 0.05), masse \"eau\"
simuler 10 ms
marque \"milieu\"
sauver état \"milieu\"
simuler 10 ms
";
    let r = run(text, Some(out.clone()));
    for f in ["protocole.exp", "mesures.csv", "journal.txt", "milieu.mps"] {
        assert!(out.join(f).exists(), "{f} manquant");
    }
    let csv = std::fs::read_to_string(out.join("mesures.csv")).unwrap();
    let lines: Vec<&str> = csv.lines().collect();
    assert!(lines[0].starts_with("t (s);pas;T en (0.05, 0.05) (K);masse eau partout (kg/m)"), "{}", lines[0]);
    assert_eq!(lines.len(), r.rows.len() + 1);
    assert!(std::fs::read_to_string(out.join("journal.txt")).unwrap().contains("repère « milieu »"));
    // nouvelle expérience depuis l'état enregistré : elle repart de t = 10 ms et rejoint la première au bit près
    let p2 = proto("départ état \"milieu.mps\"\nsimuler 10 ms\n");
    let mut r2 = Runner::new(p2, &out, None, Some(false)).unwrap();
    assert!((r2.sim.time - 0.01).abs() < 1e-8);
    run_to_end(&mut r2);
    assert_eq!(r2.sim.mass, r.sim.mass);
    assert_eq!(r2.sim.u, r.sim.u);
}

#[test]
fn the_initial_state_is_the_start_plus_the_actions_before_the_first_simulate() {
    let text = "départ scénario \"cuve vide\" grille 20 x 20 maille 1 cm\nremplir rect (0, 0) (0.2, 0.05) avec \"eau\" à 20 °C\nsimuler 1 ms\nremplir partout avec \"huile\"\n";
    let p = proto(text);
    let (s, lib, sc) = initial_state(&p, Path::new(".")).unwrap();
    let eau = s.reg.phases.iter().position(|p| p.name == "eau").unwrap();
    let huile = s.reg.phases.iter().position(|p| p.name == "huile").unwrap();
    assert!(s.mass[eau][s.idx(10, 2)] > 900.0 && s.mass[huile].iter().all(|&m| m == 0.0));
    assert!(lib.is_some() && sc == Some(MpScenario::Empty.index()));
    // une ligne de conditions initiales s'insère avant le premier « simuler »
    let (t2, line) = insert_initial_line(text, "solide rect (0, 0.1) (0.2, 0.12) \"acier\"");
    assert_eq!(line, 3);
    assert_eq!(t2.lines().nth(2).unwrap(), "solide rect (0, 0.1) (0.2, 0.12) \"acier\"");
    assert_eq!(t2.lines().nth(3).unwrap(), "simuler 1 ms");
    let (t3, line3) = insert_initial_line("départ scénario \"cuve vide\"\n\n", "effacer partout");
    assert_eq!((t3.as_str(), line3), ("départ scénario \"cuve vide\"\neffacer partout\n", 2));
    // les bords du scénario sont gardés tant qu'aucune ligne « bord » ne les change
    assert!(matches!(s.bc[3], Bc::Open { .. }));
}

#[test]
fn the_example_protocols_read_and_start() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiences");
    let mut n = 0;
    for e in std::fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        if p.extension().map_or(true, |x| x != "exp") {
            continue;
        }
        let text = std::fs::read_to_string(&p).unwrap();
        let protos = parse_all(&text).unwrap_or_else(|e| panic!("{} : {}", p.display(), e.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n")));
        for pr in protos {
            let mut r = Runner::new(pr, &dir, None, Some(false)).unwrap_or_else(|e| panic!("{} : {e}", p.display()));
            for _ in 0..3 {
                r.advance(Duration::from_millis(1));
            }
            assert!(matches!(r.state, RunState::Running | RunState::Finished), "{} : {:?}", p.display(), r.state);
            assert!(r.sim.state_is_sane());
        }
        n += 1;
    }
    assert!(n >= 3, "{n} protocoles d'exemple");
}

#[test]
fn the_command_line_runs_every_variant_of_a_protocol_file() {
    let dir = tmp_dir("cli");
    let f = dir.join("essai.exp");
    std::fs::write(
        &f,
        "départ scénario \"cuve vide\" grille 12 x 12 maille 1 cm\nbalayage T = 300 K, 320 K\nremplir rect (0, 0) (0.12, 0.03) avec \"eau\" à $T\nmesurer toutes les 2 ms : T en (0.06, 0.01)\nsimuler 4 ms\n",
    )
    .unwrap();
    let out = dir.join("sorties");
    let code = super::mp_experiment::run_cli(&[f.display().to_string(), "--cpu".into(), "--sortie".into(), out.display().to_string()]);
    assert_eq!(code, 0);
    for v in ["T=300_K", "T=320_K"] {
        let csv = std::fs::read_to_string(out.join("essai").join(v).join("mesures.csv")).unwrap_or_else(|e| panic!("{v} : {e}"));
        assert!(csv.lines().count() >= 3, "{csv}");
    }
    // fichier absent, protocole faux : code d'erreur
    assert_ne!(super::mp_experiment::run_cli(&[dir.join("absent.exp").display().to_string()]), 0);
    std::fs::write(dir.join("faux.exp"), "départ scénario \"cuve vide\"\nvoler\n").unwrap();
    assert_ne!(super::mp_experiment::run_cli(&[dir.join("faux.exp").display().to_string()]), 0);
}
