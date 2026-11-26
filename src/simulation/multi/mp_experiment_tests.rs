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
mesurer toutes les 4 ms : stats, T en (0.05, 0.05), max vitesse dans partout, débit massique \"eau\" à travers ligne (0.1, 0) (0.1, 0.3)
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
    // la même expérience sans mesures (le débit fait compter le transport sur ses faces) : état final identique au bit près
    let quiet = run(&TIMING.replace("mesurer toutes les 4 ms : stats, T en (0.05, 0.05), max vitesse dans partout, débit massique \"eau\" à travers ligne (0.1, 0) (0.1, 0.3)\n", ""), None);
    assert!(r.rows.iter().any(|row| row[4] > 0.1), "l'eau passe x = 10 cm : {:?}", r.rows.iter().map(|row| row[4]).collect::<Vec<_>>());
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
fn the_corps_command_places_fixed_rigid_bodies() {
    let text = "\
départ scénario \"cuve vide\" grille 30 x 30 maille 1 cm
corps disque (0.1, 0.1) rayon 2 cm \"acier\"
corps rect (0.15, 0.05) (0.25, 0.08) \"verre\" angle 45
corps polygone (0.05, 0.2) (0.12, 0.2) (0.08, 0.27) \"acier\" fixe
simuler 5 ms
";
    let p = proto(text);
    let (s, _, _) = initial_state(&p, Path::new(".")).unwrap();
    assert_eq!(s.bodies.len(), 3);
    assert!((s.bodies[1].angle - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    let area: f64 = s.body_frac.iter().sum::<f64>() * s.dx * s.dx;
    let expect = std::f64::consts::PI * 4e-4 + 0.1 * 0.03 + 0.5 * 0.07 * 0.07;
    assert!((area - expect).abs() < 1e-9, "aire {area} pour {expect}");
    // le protocole tourne (le GPU refuse les scènes avec des corps : voir mp_bodies)
    let mut r = Runner::new(p, Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    assert!((r.sim.time - 0.005).abs() < 1e-9 && r.sim.bodies.len() == 3);
    assert!(r.log.iter().any(|l| l.contains("corps fixe « verre »") && l.contains("tourné de 45°")), "{}", r.log.join("\n"));
    // mouvements imposés : vitesse, rotation (°/s par défaut), oscillation sur une période, pivot
    let m = proto(
        "départ scénario \"cuve vide\" grille 30 x 30 maille 1 cm\n\
         corps rect (0.02, 0.05) (0.04, 0.25) \"acier\" oscillation (0.1, 0) m/s période 1 s\n\
         corps disque (0.2, 0.2) rayon 2 cm \"acier\" vitesse (5 cm/s, 0)\n\
         corps rect (0.1, 0.19) (0.2, 0.21) \"verre\" pivot (0.15, 0.2) rotation 30 tr/min\n",
    );
    let (s, _, _) = initial_state(&m, Path::new(".")).unwrap();
    use super::mp_bodies::Motion;
    assert!(matches!(s.bodies[0].motion, Motion::Imposed { osc_v, period, .. } if osc_v == [0.1, 0.0] && period == 1.0));
    assert!(matches!(s.bodies[1].motion, Motion::Imposed { v, .. } if (v[0] - 0.05).abs() < 1e-15));
    assert!(matches!(s.bodies[2].motion, Motion::Imposed { omega, .. } if (omega - std::f64::consts::PI).abs() < 1e-12));
    assert_eq!(s.bodies[2].pivot, Some([0.15, 0.2]));
    assert!(s.has_moving_bodies());
    // corps libres : axes bloqués, masse volumique propre, vitesse de départ ramenée aux mouvements permis, pivot ; mesures
    let f = proto(
        "départ scénario \"cuve vide\" grille 30 x 30 maille 1 cm\n\
         corps disque (0.2, 0.2) rayon 2 cm \"acier\" libre bloqué x et rotation masse volumique 0.6 g/cm³ vitesse (0.3, 0.1)\n\
         corps rect (0.1, 0.19) (0.2, 0.21) \"verre\" libre pivot (0.1, 0.2) rotation 90\n\
         mesurer toutes les 1 ms : corps 1 y comme \"hauteur\", corps 2 rotation, corps 1 fy\n\
         simuler 3 ms\n",
    );
    let (s, _, _) = initial_state(&f, Path::new(".")).unwrap();
    let (a, b) = (&s.bodies[0], &s.bodies[1]);
    assert!(a.is_free() && !a.dof.x && a.dof.y && !a.dof.rot && a.rho == Some(600.0), "{a:?}");
    assert!(a.vel == [0.0, 0.1] && a.omega == 0.0, "vitesse de départ {:?} {}", a.vel, a.omega);
    // pivot : rotation de 90 °/s autour de (0.1, 0.2), le centre (0.15, 0.2) part vers le haut à ω r
    assert!(b.is_free() && b.pivot == Some([0.1, 0.2]) && (b.omega - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    assert!(b.vel[0].abs() < 1e-15 && (b.vel[1] - 0.05 * std::f64::consts::FRAC_PI_2).abs() < 1e-12, "{:?}", b.vel);
    let mut r = Runner::new(f, Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    assert!(r.columns.iter().any(|c| c == "hauteur (m)") && r.columns.iter().any(|c| c == "corps 2 rotation (°/s)"), "{:?}", r.columns);
    assert!(r.log.iter().any(|l| l.contains("corps libre « acier »") && l.contains("bloqué en x, rotation") && l.contains("600 kg/m³")), "{}", r.log.join("\n"));
    let last = r.rows.last().unwrap();
    let yc = r.columns.iter().position(|c| c == "hauteur (m)").unwrap();
    assert!(last[yc] < 0.2 + 0.1 * 0.004 && last[yc] > 0.2, "le disque monte à 0.1 m/s : y = {}", last[yc]);
    for (bad, what) in [
        ("corps disque (0.1, 0.1) rayon 2 cm \"acier\" libre oscillation (0.1, 0) période 1 s", "oscillation"),
        ("corps disque (0.1, 0.1) rayon 2 cm \"acier\" bloqué x", "libre"),
        ("corps disque (0.1, 0.1) rayon 2 cm \"acier\" fixe libre", "pas les deux"),
        ("corps disque (0.1, 0.1) rayon 2 cm \"acier\" libre bloqué", "axe"),
        ("corps disque (0.1, 0.1) rayon 2 cm \"acier\" libre masse volumique 0", "positive"),
        ("corps disque (0.1, 0.1) rayon 2 cm \"acier\" oscillation (0.1, 0)", "période"),
        ("corps disque (0.1, 0.1) rayon 2 cm \"acier\" période 1 s", "oscillation"),
        ("corps rect (0, 0) (0.1, 0.02) \"acier\" pivot (0, 0) vitesse (1, 0)", "pivot"),
    ] {
        let e = parse(&format!("départ scénario \"cuve vide\"\n{bad}"), None).unwrap_err();
        assert!(e[0].line == 2 && e[0].msg.contains(what), "{bad} : {e:?}");
    }
    let e = Runner::new(proto("départ scénario \"cuve vide\" grille 16 x 16\ncorps disque (0.05, 0.05) rayon 1 cm \"bois flotté\"\n"), Path::new("."), None, Some(false)).err().unwrap();
    assert!(e.starts_with("ligne 2") && e.contains("acier"), "{e}");
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

#[test]
fn a_simulation_can_run_until_a_reading_crosses_a_threshold() {
    use super::mp_experiment::{BodyQty, Clause, Cmp};
    // lecture : relevé comme une mesure, comparaison, seuil avec son unité, durée maximale facultative
    let p = proto(
        "départ scénario \"cuve vide\"
         simuler jusqu'à ce que max T dans partout < 400 K au plus 5 s
         simuler jusqu'à ce que corps 1 angle ≥ 0.5 rad
         simuler jusqu'à ce que T en (0.1, 0.1) > 20 °C
         simuler jusqu'à 2 s
",
    );
    // une seule comparaison : la clause et le seuil
    let one = |k: usize| match &p.steps[k].0 {
        Step::UntilCond { clauses, any: false, limit } if clauses.len() == 1 => (clauses[0].clone(), *limit),
        s => panic!("{s:?}"),
    };
    match one(0) {
        (Clause { measure: Measure::Probe { q: Quantity::T, reduce: Reduce::Max, zone: Zone::Partout, .. }, cmp: Cmp::Lt, value }, Some(l)) => {
            assert!(value == 400.0 && l == 5.0)
        }
        s => panic!("{s:?}"),
    }
    match one(1) {
        (Clause { measure: Measure::Body { index: 0, q: BodyQty::Angle, .. }, cmp: Cmp::Ge, value }, None) => {
            assert!((value - 0.5f64.to_degrees()).abs() < 1e-9, "angle en degrés comme la mesure : {value}")
        }
        s => panic!("{s:?}"),
    }
    assert!(matches!(one(2).0, Clause { cmp: Cmp::Gt, value, .. } if (value - 293.15).abs() < 1e-9));
    assert!(matches!(&p.steps[3].0, Step::Until(t) if *t == 2.0));
    assert_eq!(p.planned_end(0.0), 5.0);
    for (bad, what) in [
        ("simuler jusqu'à ce que max T dans partout 400 K", "comparaison"),
        ("simuler jusqu'à ce que stats < 3", "stats"),
        ("simuler jusqu'à ce que corps 1 y < 0.1 au plus", "durée maximale"),
        ("simuler jusqu'à ce que corps 1 y < 0.1 au plus -1 s", "positive"),
    ] {
        let e = parse_all(&format!("départ scénario \"cuve vide\"\n{bad}\n")).unwrap_err();
        assert!(e[0].line == 2 && e[0].msg.contains(what), "{bad} : {e:?}");
    }

    // déroulement : un disque d'acier tombe dans l'air ; l'expérience s'arrête au pas où il passe sous y = 0.1 m (chute libre :
    // t = √(2 × 0.05 / g) = 0.101 s), puis continue normalement
    let mut r = Runner::new(
        proto(
            "départ scénario \"cuve vide\" grille 20 x 40 maille 5 mm
             corps disque (0.05, 0.15) rayon 1 cm \"acier\" libre
             simuler jusqu'à ce que corps 1 y < 0.1 au plus 1 s
             mesurer toutes les 1 ms : corps 1 y
             simuler 10 ms
             simuler jusqu'à ce que corps 1 y < -1 au plus 20 ms
             simuler jusqu'à ce que corps 1 y < 1
",
        ),
        Path::new("."),
        None,
        Some(false),
    )
    .unwrap();
    run_to_end(&mut r);
    let log = r.log.join("\n");
    // [t = … s] ligne 3 : corps 1 y < 0.1 m atteint (y m)
    let hit = r.log.iter().find(|l| l.contains("corps 1 y < 0.1 m atteint")).unwrap_or_else(|| panic!("{log}"));
    let num = |s: &str| s.trim().parse::<f64>().unwrap();
    let t1 = num(&hit[hit.find("t = ").unwrap() + 4..hit.find(" s]").unwrap()]);
    let y1 = num(&hit[hit.rfind('(').unwrap() + 1..hit.rfind(" m)").unwrap()]);
    let t_free = (2.0 * 0.05 / 9.81f64).sqrt();
    eprintln!("condition atteinte à t = {t1:.5} s (chute libre : {t_free:.5}), y = {y1:.5} m");
    // arrêt au premier pas sous le seuil : moins d'une maille de chute au-delà
    assert!(y1 < 0.1 && y1 > 0.1 - r.sim.dx, "arrêt au pas où y passe sous 0.1 : y = {y1}");
    // le corps avance à la vitesse du pas précédent : retard de ½ g t dt (2.5 % à des pas de 3 ms)
    assert!((t1 / t_free - 1.0).abs() < 0.05, "temps de chute {t1} pour {t_free}");
    assert!(log.contains("corps 1 y < -1 m non atteint au bout du temps permis"), "{log}");
    assert!(log.contains("corps 1 y < 1 m déjà vrai"), "{log}");
    assert!((r.sim.time - (t1 + 0.01 + 0.02)).abs() < 2e-6, "durées suivantes : t = {} (attendu {})", r.sim.time, t1 + 0.03);
}

#[test]
fn the_flow_rate_through_a_line_is_measured_along_the_faces() {
    // lecture
    let p = proto("départ scénario \"cuve vide\"\nmesurer toutes les 1 ms : débit massique \"eau\" à travers ligne (0.1, 0) (0.1, 0.2) comme \"q\", débit à travers ligne (0, 0.1) (0.3, 0.1)\n");
    assert_eq!(p.measures[0].0, Measure::Flux { fluid: Some("eau".into()), mass: true, a: [0.1, 0.0], b: [0.1, 0.2], label: Some("q".into()) });
    assert_eq!(p.measures[1].0, Measure::Flux { fluid: None, mass: false, a: [0.0, 0.1], b: [0.3, 0.1], label: None });
    for (bad, what) in [
        ("débit à travers ligne (0.1, 0) (0.2, 0.1)", "horizontale ou verticale"),
        ("débit \"eau\" ligne (0.1, 0) (0.1, 0.1)", "à travers ligne"),
        ("débit à travers (0.1, 0) (0.1, 0.1)", "à travers ligne"),
    ] {
        let e = parse_all(&format!("départ scénario \"cuve vide\"\nmesurer toutes les 1 ms : {bad}\n")).unwrap_err();
        assert!(e[0].line == 2 && e[0].msg.contains(what), "{bad} : {e:?}");
    }
    let e = Runner::new(proto("départ scénario \"cuve vide\" grille 10 x 10 maille 1 cm\nmesurer toutes les 1 ms : débit à travers ligne (0.5, 0) (0.5, 0.1)\nsimuler 1 ms\n"), Path::new("."), None, Some(false)).err().unwrap();
    assert!(e.contains("hors du domaine"), "{e}");

    // champ de vitesse uniforme au départ, eau au fond (5 rangées), air au-dessus : débits exacts
    let text = "départ scénario \"cuve vide\" grille 10 x 10 maille 1 cm
remplir rect (0, 0) (0.1, 0.05) avec \"eau\"
champ vx formule \"0.2\"
champ vy formule \"-0.1\"
mesurer toutes les 1 ms : débit à travers ligne (0.05, 0) (0.05, 0.1), débit \"eau\" à travers ligne (0.05, 0) (0.05, 0.1), débit massique \"eau\" à travers ligne (0.05, 0) (0.05, 0.1), débit à travers ligne (0, 0.03) (0.1, 0.03), débit \"air\" à travers ligne (0, 0.03) (0.1, 0.03), débit à travers ligne (0.02, 0.06) (0.08, 0.06) comme \"q\"
simuler 1 ms
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    assert_eq!(r.columns[3], "débit eau à travers ligne (0.05, 0) (0.05, 0.1) (m²/s)", "{:?}", r.columns);
    assert_eq!(r.columns[4], "débit massique eau à travers ligne (0.05, 0) (0.05, 0.1) (kg/(m·s))");
    assert_eq!(r.columns[7], "q (m²/s)");
    let row = &r.rows[0];
    assert_eq!(row[0], 0.0);
    // tout le fluide : 0.2 m/s sur 10 cm ; l'eau : sur 5 cm ; à travers y = 3 cm, l'eau descend (maille amont : celle du dessus) ; à
    // travers y = 6 cm, de x = 2 à 8 cm, l'air descend
    for (k, w) in [(2, 0.02), (3, 0.01), (5, -0.01), (6, 0.0), (7, -0.006)] {
        assert!((row[k] - w).abs() < 1e-12, "{} : {} au lieu de {w}", r.columns[k], row[k]);
    }
    assert!(row[4] / row[3] > 990.0 && row[4] / row[3] < 1005.0, "masse volumique de l'eau : {}", row[4] / row[3]);

    // le débit relevé est le débit moyen depuis le relevé précédent, compté par le transport : débit × durée, sommé sur les intervalles,
    // donne exactement la masse passée. Bilan (masse passée au plus, plus grand écart) ; `sign` : −1 pour une zone du côté −
    let balance = |r: &Runner, sign: f64| -> (f64, f64) {
        let (m0, mut acc, mut worst, mut m_max) = (r.rows[0][2], 0.0f64, 0.0f64, 0.0f64);
        for w in r.rows.windows(2) {
            acc += sign * w[1][3] * (w[1][0] - w[0][0]);
            worst = worst.max((acc - (w[1][2] - m0)).abs());
            m_max = m_max.max((w[1][2] - m0).abs());
        }
        (m_max, worst)
    };
    // rupture de barrage : l'eau qui passe x = 20 cm est celle qui s'accumule à droite
    let dam = "départ scénario \"Rupture de barrage\" grille 40 x 30 maille 1 cm
mesurer toutes les 2 ms : masse \"eau\" dans rect (0.2, 0) (0.4, 0.3), débit massique \"eau\" à travers ligne (0.2, 0) (0.2, 0.3)
simuler 0.3 s
";
    let mut r = Runner::new(proto(dam), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let (m_max, worst) = balance(&r, 1.0);
    eprintln!("barrage (CPU) : {m_max:.4} kg/m passés à droite au plus, écart au débit intégré {worst:.2e} kg/m");
    assert!(m_max > 2.0, "l'eau n'est pas arrivée à droite : {m_max}");
    assert!(worst < 1e-9 * m_max, "débit intégré et masse accumulée : écart {worst} pour {m_max}");
    // sable qui tombe dans l'eau : le débit des grains compte aussi leur chute à travers le fluide (sédimentation)
    let sand = "départ scénario \"sédimentation\" grille 15 x 20 maille 2 cm
mesurer toutes les 10 ms : masse \"sable\" dans rect (0, 0) (0.3, 0.12), débit massique \"sable\" à travers ligne (0, 0.12) (0.3, 0.12)
simuler 1.5 s
";
    let mut s = Runner::new(proto(sand), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut s);
    let (s_max, s_worst) = balance(&s, -1.0);
    eprintln!("sédimentation (CPU) : {s_max:.4} kg/m de sable passés sous 12 cm au plus, écart {s_worst:.2e} kg/m");
    assert!(s_max > 0.5, "le sable n'est pas descendu sous 12 cm : {s_max}\n{}", s.log.join("\n"));
    assert!(s_worst < 1e-9 * s_max, "sable : écart {s_worst} pour {s_max}");
    // disque poussé à travers la ligne : l'eau qu'il chasse de maille en maille est comptée aussi
    let body = "départ scénario \"cuve vide\" grille 30 x 20 maille 1 cm
remplir rect (0, 0) (0.3, 0.12) avec \"eau\"
corps disque (0.08, 0.06) rayon 2 cm \"acier\" vitesse (0.3, 0)
mesurer toutes les 5 ms : masse \"eau\" dans rect (0.15, 0) (0.3, 0.2), débit massique \"eau\" à travers ligne (0.15, 0) (0.15, 0.2)
simuler 0.3 s
";
    let mut b = Runner::new(proto(body), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut b);
    let (b_max, b_worst) = balance(&b, 1.0);
    eprintln!("corps poussé (CPU) : {b_max:.4} kg/m d'écart de masse à droite au plus, écart au débit intégré {b_worst:.2e} kg/m");
    assert!(b.sim.bodies[0].pos[0] > 0.16 && b_max > 0.1, "le disque n'a pas passé la ligne : x = {}, {b_max}", b.sim.bodies[0].pos[0]);
    assert!(b_worst < 1e-9 * b_max, "corps : écart {b_worst} pour {b_max}");
    // lit fluidisé : l'air traverse le lit tassé par ses pores (percolation) ; entre deux lignes, la masse d'air et de sable change de
    // ce qui entre par l'une moins ce qui sort par l'autre
    let bed = "départ scénario \"lit fluidisé\" grille 20 x 40 maille 1 cm
mesurer toutes les 5 ms : masse \"air\" dans rect (0, 0.05) (0.2, 0.1), débit massique \"air\" à travers ligne (0, 0.05) (0.2, 0.05), débit massique \"air\" à travers ligne (0, 0.1) (0.2, 0.1), masse \"sable grossier\" dans rect (0, 0.05) (0.2, 0.1), débit massique \"sable grossier\" à travers ligne (0, 0.05) (0.2, 0.05), débit massique \"sable grossier\" à travers ligne (0, 0.1) (0.2, 0.1)
simuler 0.2 s
";
    let between = |r: &Runner, c: usize| -> (f64, f64, f64) {
        let (m0, mut acc, mut worst, mut m_max, mut through) = (r.rows[0][c], 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for w in r.rows.windows(2) {
            let dt = w[1][0] - w[0][0];
            acc += (w[1][c + 1] - w[1][c + 2]) * dt;
            through += w[1][c + 1].abs() * dt;
            worst = worst.max((acc - (w[1][c] - m0)).abs());
            m_max = m_max.max((w[1][c] - m0).abs());
        }
        (m_max, through, worst)
    };
    let mut f = Runner::new(proto(bed), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut f);
    for (c, what) in [(2, "air"), (5, "sable")] {
        let (m_max, through, worst) = between(&f, c);
        eprintln!("lit fluidisé (CPU), {what} : passé {through:.4e} kg/m, variation {m_max:.4e} kg/m, écart {worst:.2e} kg/m");
        assert!(through > 1e-3 && worst < 1e-9 * through.max(m_max), "lit, {what} : écart {worst} pour {through}");
    }

    // GPU : mêmes comptes (cumuls en f32 sur le GPU, repris en f64 à chaque relecture)
    let probe = build_scenario(MpScenario::DamBreak, 40, 30, 0.01, 9.81);
    if super::mp_gpu_tests::gpu_or_skip(&probe).is_none() {
        return;
    }
    for (text, sign, cpu_max, what) in [(dam, 1.0, m_max, "barrage"), (sand, -1.0, s_max, "sédimentation")] {
        let mut g = Runner::new(proto(text), Path::new("."), None, Some(true)).unwrap();
        assert!(g.gpu.is_some(), "{}", g.log.join("\n"));
        run_to_end(&mut g);
        let (g_max, g_worst) = balance(&g, sign);
        eprintln!("{what} (GPU) : {g_max:.4} kg/m au plus (CPU {cpu_max:.4}), écart au débit intégré {g_worst:.2e} kg/m");
        assert!(g_worst < 1e-3 * g_max, "{what} (GPU) : écart {g_worst} pour {g_max}");
        assert!((g_max / cpu_max - 1.0).abs() < 0.1, "{what} : GPU {g_max} / CPU {cpu_max}");
    }
    let mut g = Runner::new(proto(bed), Path::new("."), None, Some(true)).unwrap();
    assert!(g.gpu.is_some(), "{}", g.log.join("\n"));
    run_to_end(&mut g);
    for (c, what) in [(2, "air"), (5, "sable")] {
        let (m_max, through, worst) = between(&g, c);
        eprintln!("lit fluidisé (GPU), {what} : passé {through:.4e} kg/m, variation {m_max:.4e} kg/m, écart {worst:.2e} kg/m");
        assert!(through > 1e-3 && worst < 1e-3 * through.max(m_max), "lit (GPU), {what} : écart {worst} pour {through}");
    }
}

#[test]
fn a_point_probe_interpolates_between_the_neighbouring_cells() {
    // champs linéaires : l'interpolation bilinéaire les rend exactement, entre les centres des mailles pour T, entre les faces pour
    // les vitesses ; au centre d'une maille, sa valeur ; à moins d'une demi-maille du bord, la rangée de mailles la plus proche
    let text = "départ scénario \"cuve vide\" grille 10 x 10 maille 1 cm
champ T formule \"300 + 1000*x + 2000*y\"
champ vx formule \"0.1 + 2*y\"
champ vy formule \"0.05 + 3*x\"
mesurer toutes les 1 ms : T en (0.033, 0.047), T en (0.035, 0.045), T en (0.002, 0.047), vx en (0.033, 0.047), vy en (0.033, 0.047), vitesse en (0.033, 0.047)
simuler 1 ms
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let row = &r.rows[0];
    assert_eq!(row[0], 0.0, "{:?}", r.columns);
    let want = [427.0, 425.0, 399.0, 0.194, 0.149, 0.194f64.hypot(0.149)];
    for (k, w) in want.iter().enumerate() {
        assert!((row[2 + k] - w).abs() < 1e-9 * w.max(1.0), "{} : {} au lieu de {w}", r.columns[2 + k], row[2 + k]);
    }

    // maille solide voisine : écartée pour p (poids des autres renormalisés), comptée pour T ; point dans le solide : pas de pression
    let text = "départ scénario \"cuve vide\" grille 10 x 10 maille 1 cm
remplir rect (0, 0) (0.1, 0.05) avec \"eau\"
solide rect (0.04, 0) (0.06, 0.03) \"acier\" à 350 K
mesurer toutes les 1 ms : p en (0.038, 0.012), p en (0.045, 0.015), T en (0.038, 0.012)
simuler 1 ms
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let row = r.rows.last().unwrap();
    let s = &r.sim;
    assert!(s.is_solid(s.idx(4, 1)) && !s.is_solid(s.idx(3, 1)), "solide mal placé");
    // (0.038, 0.012) : colonnes 3 (0.7) et 4 (0.3, solide), rangées 0 (0.3) et 1 (0.7)
    let at = |f: &[f64], i: usize, j: usize| f[s.idx(i, j)];
    let p = 0.3 * at(&s.p, 3, 0) + 0.7 * at(&s.p, 3, 1);
    assert!((row[2] - p).abs() < 1e-9 * p.abs(), "p : {} au lieu de {p}", row[2]);
    assert!(row[3].is_nan(), "p dans le solide : {}", row[3]);
    let t = 0.7 * (0.3 * at(&s.temp, 3, 0) + 0.7 * at(&s.temp, 3, 1)) + 0.3 * (0.3 * at(&s.temp, 4, 0) + 0.7 * at(&s.temp, 4, 1));
    assert!((row[4] - t).abs() < 1e-9 * t, "T : {} au lieu de {t}", row[4]);
    assert!(at(&s.temp, 4, 1) > at(&s.temp, 3, 1) + 10.0, "le solide est plus chaud que l'eau : {} / {}", at(&s.temp, 4, 1), at(&s.temp, 3, 1));
}

#[test]
fn a_condition_can_combine_comparisons_with_and_or() {
    use super::mp_experiment::{BodyQty, Clause, Cmp};
    // lecture : comparaisons reliées par « et » ou par « ou » (sans mélange), durée maximale à la fin
    let p = proto(
        "départ scénario \"cuve vide\"
         simuler jusqu'à ce que max T dans partout > 400 K ou corps 1 y < 5 cm ou corps 1 angle > 1.5 rad au plus 2 s
         simuler jusqu'à ce que corps 1 vy > -0.01 et corps 1 y < 0.05
",
    );
    match &p.steps[0].0 {
        Step::UntilCond { clauses, any: true, limit: Some(l) } if clauses.len() == 3 && *l == 2.0 => {
            assert!(matches!(clauses[0], Clause { measure: Measure::Probe { q: Quantity::T, reduce: Reduce::Max, .. }, cmp: Cmp::Gt, value } if value == 400.0));
            assert!(matches!(clauses[1], Clause { measure: Measure::Body { index: 0, q: BodyQty::Y, .. }, cmp: Cmp::Lt, value } if (value - 0.05).abs() < 1e-12));
            assert!(matches!(clauses[2], Clause { measure: Measure::Body { q: BodyQty::Angle, .. }, cmp: Cmp::Gt, value } if (value - 1.5f64.to_degrees()).abs() < 1e-9));
        }
        s => panic!("{s:?}"),
    }
    assert!(matches!(&p.steps[1].0, Step::UntilCond { clauses, any: false, limit: None } if clauses.len() == 2));
    for (bad, what) in [
        ("simuler jusqu'à ce que corps 1 y < 0.1 et corps 1 x > 0 ou corps 1 vy < 0", "ne se mélangent pas"),
        ("simuler jusqu'à ce que corps 1 y < 0.1 et stats < 3", "stats"),
        ("simuler jusqu'à ce que corps 1 y < 0.1 ou corps 1 x 0.2", "comparaison"),
        ("simuler jusqu'à ce que corps 1 y < 0.1 et", ""),
    ] {
        let e = parse_all(&format!("départ scénario \"cuve vide\"\n{bad}\n")).unwrap_err();
        assert!(e[0].line == 2 && e[0].msg.contains(what), "{bad} : {e:?}");
    }

    // déroulement : le disque d'acier qui tombe dans l'air ; « ou » s'arrête à la première comparaison vraie, « et » attend toutes
    let mut r = Runner::new(
        proto(
            "départ scénario \"cuve vide\" grille 20 x 40 maille 5 mm
             corps disque (0.05, 0.15) rayon 1 cm \"acier\" libre
             simuler jusqu'à ce que corps 1 y < -1 ou corps 1 y < 0.1 au plus 1 s
             simuler jusqu'à ce que corps 1 y < 0.08 et corps 1 x > 1 au plus 20 ms
             simuler jusqu'à ce que corps 1 y < 0.09 et corps 1 x > 0.04
",
        ),
        Path::new("."),
        None,
        Some(false),
    )
    .unwrap();
    run_to_end(&mut r);
    let log = r.log.join("\n");
    let num = |s: &str| s.trim().parse::<f64>().unwrap_or_else(|e| panic!("« {s} » : {e}"));
    // [t = … s] ligne 3 : corps 1 y < -1 m ou corps 1 y < 0.1 m atteint (y m ; y m)
    let hit = r.log.iter().find(|l| l.contains("corps 1 y < -1 m ou corps 1 y < 0.1 m atteint (")).unwrap_or_else(|| panic!("{log}"));
    let t1 = num(&hit[hit.find("t = ").unwrap() + 4..hit.find(" s]").unwrap()]);
    let t_free = (2.0 * 0.05 / 9.81f64).sqrt();
    assert!((t1 / t_free - 1.0).abs() < 0.05, "temps de chute {t1} pour {t_free}");
    // « et » : le disque passe sous 8 cm, mais x reste à 5 cm : la condition court jusqu'au bout des 20 ms
    let miss = r
        .log
        .iter()
        .find(|l| l.contains("corps 1 y < 0.08 m et corps 1 x > 1 m non atteint au bout du temps permis ("))
        .unwrap_or_else(|| panic!("{log}"));
    let vals: Vec<f64> = miss[miss.rfind('(').unwrap() + 1..miss.rfind(')').unwrap()].split(" ; ").map(|v| num(v.trim_end_matches(" m"))).collect();
    assert!(vals.len() == 2 && vals[0] < 0.08 && (vals[1] - 0.05).abs() < 1e-3, "{miss}");
    assert!(log.contains("corps 1 y < 0.09 m et corps 1 x > 0.04 m déjà vrai"), "{log}");
    assert!((r.sim.time - (t1 + 0.02)).abs() < 2e-6, "t = {} (attendu {})", r.sim.time, t1 + 0.02);
}

#[test]
fn a_condition_stops_the_gpu_run_like_the_cpu_run() {
    // eau chauffée par le fond : la condition sur la température d'une sonde arrête le calcul au même moment sur le CPU et le GPU
    // (« et » avec une comparaison toujours vraie : le GPU repasse au pas unique à l'approche du seuil de l'autre)
    let text = "départ scénario \"cuve vide\" grille 12 x 12 maille 1 cm
remplir rect (0, 0) (0.12, 0.08) avec \"eau\" à 20 °C
chauffer rect (0.03, 0) (0.09, 0.02) à 360 K
simuler jusqu'à ce que T en (0.06, 0.015) > 320 K et max T dans partout < 1000 K au plus 5 s
";
    let stop = |r: &Runner| -> f64 {
        let l = r.log.iter().find(|l| l.contains("> 320 K et ") && l.contains("< 1000 K atteint")).unwrap_or_else(|| panic!("{}", r.log.join("\n")));
        l[l.find("t = ").unwrap() + 4..l.find(" s]").unwrap()].trim().parse().unwrap()
    };
    let mut cpu = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut cpu);
    let t_cpu = stop(&cpu);
    assert!(t_cpu > 0.0 && t_cpu < 5.0 && (cpu.sim.time - t_cpu).abs() < 1e-6, "CPU : arrêt à {t_cpu}, t final {}", cpu.sim.time);
    let probe = build_scenario(MpScenario::DamBreak, 40, 30, 0.01, 9.81);
    if super::mp_gpu_tests::gpu_or_skip(&probe).is_none() {
        return;
    }
    let mut gpu = Runner::new(proto(text), Path::new("."), None, Some(true)).unwrap();
    assert!(gpu.gpu.is_some(), "{}", gpu.log.join("\n"));
    run_to_end(&mut gpu);
    let t_gpu = stop(&gpu);
    eprintln!("condition : CPU à {t_cpu:.4} s, GPU à {t_gpu:.4} s");
    // le GPU vérifie après chaque série de pas (au plus 8) : même moment à quelques pas près
    assert!((t_gpu - t_cpu).abs() < 0.05 * t_cpu, "arrêt : CPU {t_cpu} / GPU {t_gpu}");
}

#[test]
fn the_fluid_force_on_solid_cells_is_archimedes_and_the_hydrostatic_thrust() {
    // bloc de mailles solides entouré d'eau au repos : poussée d'Archimède ρ g A vers le haut, rien de côté ; muret qui retient
    // 10 cm d'eau : poussée horizontale ½ (ρ_eau − ρ_air) g H² (la pression de l'air agit des deux côtés)
    let text = "départ scénario \"cuve vide\" grille 20 x 20 maille 1 cm
remplir rect (0, 0) (0.2, 0.18) avec \"eau\" à 20 °C
solide rect (0.07, 0.06) (0.13, 0.1) \"acier\"
mesurer toutes les 10 ms : force x sur rect (0.07, 0.06) (0.13, 0.1), force y sur rect (0.07, 0.06) (0.13, 0.1) comme \"poussée\"
simuler 50 ms
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    assert_eq!(r.columns[3], "poussée (N/m)");
    let row = r.rows.last().unwrap();
    let eau = r.sim.reg.phases.iter().position(|p| p.name == "eau").unwrap();
    let rho = r.sim.reg.phases[eau].rho(101325.0, 293.15);
    let expect = rho * 9.81 * 0.06 * 0.04;
    eprintln!("bloc immergé : force ({:.4}, {:.4}) N/m, Archimède {expect:.4}", row[2], row[3]);
    assert!(row[2].abs() < 1e-3 * expect, "force de côté {}", row[2]);
    assert!((row[3] / expect - 1.0).abs() < 2e-3, "poussée {} pour {expect}", row[3]);

    let text = "départ scénario \"cuve vide\" grille 20 x 20 maille 1 cm
solide rect (0.1, 0) (0.12, 0.15) \"acier\"
remplir rect (0, 0) (0.1, 0.1) avec \"eau\" à 20 °C
mesurer toutes les 10 ms : force x sur rect (0.1, 0) (0.12, 0.15)
simuler 20 ms
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let fx = r.rows.last().unwrap()[2];
    let air = r.sim.reg.phases.iter().position(|p| p.name == "air").unwrap();
    let expect = 0.5 * (rho - r.sim.reg.phases[air].rho(101325.0, 293.15)) * 9.81 * 0.1 * 0.1;
    eprintln!("muret : force x {fx:.4} N/m, ½ Δρ g H² = {expect:.4}");
    assert!((fx / expect - 1.0).abs() < 1e-2, "poussée sur le muret {fx} pour {expect}");
    // erreur : zone hors du domaine
    let e = Runner::new(proto("départ scénario \"cuve vide\" grille 10 x 10\nmesurer toutes les 1 ms : force y sur rect (2, 2) (3, 3)\nsimuler 1 ms\n"), Path::new("."), None, Some(false)).err().unwrap();
    assert!(e.contains("hors du domaine"), "{e}");
}

#[test]
fn a_protocol_saves_png_images_of_its_fields() {
    use super::mp_view::MpDisplay;
    let p = proto("départ scénario \"cuve vide\"\nsauver image \"a\"\nsauver image \"b\" masse volumique\nsauver image \"c\" température\n");
    let disp: Vec<MpDisplay> = p
        .steps
        .iter()
        .filter_map(|(s, _)| match s {
            Step::Do(Action::SaveImage { display, .. }) => Some(*display),
            _ => None,
        })
        .collect();
    assert_eq!(disp, [MpDisplay::Phases, MpDisplay::Density, MpDisplay::Temperature]);
    for (bad, what) in [("sauver image \"x\" couleur", "affichage"), ("sauver image \"../x\"", "sans dossier"), ("sauver image", "nom de l'image")] {
        let e = parse_all(&format!("départ scénario \"cuve vide\"\n{bad}\n")).unwrap_err();
        assert!(e[0].line == 2 && e[0].msg.contains(what), "{bad} : {e:?}");
    }
    // barrage : image des phases au départ, de la vitesse après 0.1 s ; 40 × 30 mailles agrandies 16 fois (au plus)
    let dir = tmp_dir("images");
    let text = "départ scénario \"Rupture de barrage\" grille 40 x 30 maille 1 cm
sauver image \"départ\"
simuler 0.1 s
sauver image \"vitesse\" vitesse
";
    let mut r = Runner::new(proto(text), Path::new("."), Some(dir.clone()), Some(false)).unwrap();
    run_to_end(&mut r);
    for (name, expect_motion) in [("départ", false), ("vitesse", true)] {
        let path = dir.join(format!("{name}.png"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{} : {e}\n{}", path.display(), r.log.join("\n")));
        let dec = png::Decoder::new(std::io::Cursor::new(bytes));
        let mut rd = dec.read_info().unwrap();
        let mut buf = vec![0u8; rd.output_buffer_size()];
        let info = rd.next_frame(&mut buf).unwrap();
        assert_eq!((info.width, info.height), (640, 480), "{name}");
        // pixels de couleurs différentes : l'image n'est pas uniforme
        let first = &buf[0..4];
        let distinct = buf[..info.buffer_size()].chunks(4).any(|px| px != first);
        assert!(distinct || !expect_motion, "{name} : image uniforme");
    }
    assert!(r.log.iter().any(|l| l.contains("image vitesse (640 × 480)")), "{}", r.log.join("\n"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_protocol_saves_an_image_series_at_regular_intervals() {
    use super::mp_view::MpDisplay;
    let p = proto("départ scénario \"cuve vide\"\nimages toutes les 50 ms température\nimages chaque 0.1 s\nsimuler 1 s\n");
    assert_eq!(p.images.iter().map(|&(t, d, _)| (t, d)).collect::<Vec<_>>(), [(0.05, MpDisplay::Temperature), (0.1, MpDisplay::Phases)]);
    for (bad, what) in [
        ("images toutes les 0 s", "positive"),
        ("images toutes les 1 s couleur", "affichage"),
        ("images 1 s", "toutes les"),
        ("images toutes les 1 s\nimages toutes les 2 s phases", "une seule série"),
    ] {
        let e = parse_all(&format!("départ scénario \"cuve vide\"\n{bad}\n")).unwrap_err();
        assert!(e[0].msg.contains(what), "{bad} : {e:?}");
    }
    // barrage : une image de la vitesse toutes les 20 ms pendant 0.1 s, au départ compris
    let dir = tmp_dir("serie");
    let text = "départ scénario \"Rupture de barrage\" grille 40 x 30 maille 1 cm
images toutes les 20 ms vitesse
simuler 0.1 s
";
    let mut r = Runner::new(proto(text), Path::new("."), Some(dir.clone()), Some(false)).unwrap();
    run_to_end(&mut r);
    let sub = dir.join("images-vitesse");
    let mut names: Vec<String> = std::fs::read_dir(&sub)
        .unwrap_or_else(|e| panic!("{} : {e}\n{}", sub.display(), r.log.join("\n")))
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, ["0000.png", "0001.png", "0002.png", "0003.png", "0004.png", "0005.png"], "{}", r.log.join("\n"));
    let bytes = std::fs::read(sub.join("0003.png")).unwrap();
    let rd = png::Decoder::new(std::io::Cursor::new(bytes)).read_info().unwrap();
    assert_eq!((rd.info().width, rd.info().height), (640, 480));
    assert!(r.log.iter().any(|l| l.contains("6 images (vitesse) toutes les 0.02 s dans images-vitesse/")), "{}", r.log.join("\n"));
    let _ = std::fs::remove_dir_all(&dir);
}
