//! Tests des expériences reproductibles : lecture du script, zones, champs de valeurs, déroulement exact et reproductible, sorties,
//! ligne de commande.

use std::path::{Path, PathBuf};
use std::time::Duration;

use super::mp_experiment::{
    apply_action, fmt_num, initial_state, Action, BoundarySpec, FieldQty, FieldSource, Measure, Quantity, Reduce, RunState, Runner, Start, Step,
    Zone,
};
use super::mp_fields::{read_csv, Formula};
use super::mp_script::{insert_initial_line, parse, parse_all, parse_all_in, zone_text};
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
        let protos = parse_all_in(&text, &dir).unwrap_or_else(|e| panic!("{} : {}", p.display(), e.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n")));
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
        Step::UntilCond { clauses, logic: super::mp_experiment::Logic::Leaf(0), limit } if clauses.len() == 1 => (clauses[0].clone(), *limit),
        s => panic!("{s:?}"),
    };
    match one(0) {
        (Clause { measure: Measure::Probe { q: Quantity::T, reduce: Reduce::Max, zone: Zone::Partout, .. }, cmp: Cmp::Lt, value: super::mp_experiment::Threshold::Num(value) }, Some(l)) => {
            assert!(value == 400.0 && l == 5.0)
        }
        s => panic!("{s:?}"),
    }
    match one(1) {
        (Clause { measure: Measure::Body { who: super::mp_experiment::Who::Rank(0), q: BodyQty::Angle, .. }, cmp: Cmp::Ge, value: super::mp_experiment::Threshold::Num(value) }, None) => {
            assert!((value - 0.5f64.to_degrees()).abs() < 1e-9, "angle en degrés comme la mesure : {value}")
        }
        s => panic!("{s:?}"),
    }
    assert!(matches!(one(2).0, Clause { cmp: Cmp::Gt, value: super::mp_experiment::Threshold::Num(value), .. } if (value - 293.15).abs() < 1e-9));
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
fn a_pump_moves_a_fixed_volume_flow_and_compresses_the_gas_downstream() {
    let p = proto("départ scénario \"cuve vide\"\npompe rect (0.1, 0.04) (0.11, 0.06) (0.5, 0)\ncompresseur partout (0, 0)\n");
    let acts: Vec<&Action> = p.steps.iter().filter_map(|(s, _)| if let Step::Do(a) = s { Some(a) } else { None }).collect();
    assert!(matches!(acts[0], Action::Pump { v, .. } if *v == [0.5, 0.0]) && matches!(acts[1], Action::Pump { v, .. } if *v == [0.0, 0.0]));
    // boîte d'air fermée coupée par un mur, sauf une ouverture de 2 cm où souffle la pompe, de gauche à droite à 0.5 m/s : le débit
    // volumique est v A, quoi qu'il arrive ; l'air se comprime à droite et se détend à gauche
    let text = "départ scénario \"cuve vide\" grille 20 x 10 maille 1 cm
bord haut mur
solide rect (0.1, 0) (0.11, 0.1) \"acier\"
effacer rect (0.1, 0.04) (0.11, 0.06)
pompe rect (0.1, 0.04) (0.11, 0.06) (0.5, 0)
mesurer toutes les 10 ms : moyenne p dans rect (0, 0) (0.1, 0.1), moyenne p dans rect (0.11, 0) (0.2, 0.1), débit à travers ligne (0.1, 0.04) (0.1, 0.06)
simuler 0.1 s
";
    let check = |r: &Runner, what: &str| -> f64 {
        let log = r.log.join("\n");
        assert!(log.contains("pompe rect (0.1, 0.04) (0.11, 0.06) à (0.5, 0) m/s — 2 mailles"), "{what}\n{log}");
        for row in &r.rows[1..] {
            assert!((row[4] / (0.5 * 0.02) - 1.0).abs() < 1e-4, "{what} : débit volumique {} pour v A = 0.01 m²/s à t = {}", row[4], row[0]);
        }
        let last = r.rows.last().unwrap();
        let dp = last[3] - last[2];
        eprintln!("{what} : pompe, Δp droite − gauche = {:.0} Pa après {:.2} s", dp, last[0]);
        assert!(dp > 1e4, "{what} : la pompe ne comprime pas l'air de droite (Δp = {dp} Pa)");
        dp
    };
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let cpu = check(&r, "CPU");
    let probe = build_scenario(MpScenario::DamBreak, 40, 30, 0.01, 9.81);
    if super::mp_gpu_tests::gpu_or_skip(&probe).is_none() {
        return;
    }
    let mut g = Runner::new(proto(text), Path::new("."), None, Some(true)).unwrap();
    assert!(g.gpu.is_some(), "{}", g.log.join("\n"));
    run_to_end(&mut g);
    let gpu = check(&g, "GPU");
    assert!((gpu / cpu - 1.0).abs() < 0.05, "Δp : GPU {gpu} / CPU {cpu}");
}

#[test]
fn a_valve_closes_its_cells_to_the_flow_and_opens_like_a_gate() {
    // lecture : fermée (par défaut), ouverte, ouverte en fraction ou en pourcentage
    let p = proto(
        "départ scénario \"cuve vide\"
vanne rect (0.15, 0) (0.16, 0.3) fermée
vanne rect (0.15, 0) (0.16, 0.3) ouverte
vanne rect (0.15, 0) (0.16, 0.3) ouverte 0.4
vanne rect (0.15, 0) (0.16, 0.3) ouverte 40 %
vanne rect (0.15, 0) (0.16, 0.3)
",
    );
    let opens: Vec<f64> = p.steps.iter().filter_map(|(s, _)| if let Step::Do(Action::Valve { open, .. }) = s { Some(*open) } else { None }).collect();
    assert_eq!(opens, [0.0, 1.0, 0.4, 0.4, 0.0]);
    for (bad, what) in [("vanne partout ouverte 1.5", "entre 0"), ("vanne partout entrouverte", "« fermée » ou « ouverte")] {
        let e = parse_all(&format!("départ scénario \"cuve vide\"\n{bad}\n")).unwrap_err();
        assert!(e[0].line == 2 && e[0].msg.contains(what), "{bad} : {e:?}");
    }

    // rupture de barrage (eau à gauche, 15 cm de haut), vanne en travers à x = 15 cm, sur toute la hauteur (30 mailles) : fermée,
    // rien ne passe ; ouverte à moitié (les 15 mailles du bas), l'eau passe par le bas seulement
    let text = |state: &str| {
        format!(
            "départ scénario \"Rupture de barrage\" grille 40 x 30 maille 1 cm
vanne rect (0.15, 0) (0.16, 0.3) {state}
mesurer toutes les 10 ms : masse \"eau\" dans rect (0.16, 0) (0.4, 0.3), débit massique \"eau\" à travers ligne (0.15, 0) (0.15, 0.15), débit massique \"eau\" à travers ligne (0.15, 0.15) (0.15, 0.3)
simuler 0.4 s
"
        )
    };
    let run = |state: &str, gpu: bool| -> Runner {
        let mut r = Runner::new(proto(&text(state)), Path::new("."), None, Some(gpu)).unwrap();
        assert!(r.gpu.is_some() == gpu, "{}", r.log.join("\n"));
        run_to_end(&mut r);
        r
    };
    let closed = run("fermée", false);
    let log = closed.log.join("\n");
    assert!(log.contains("vanne rect (0.15, 0) (0.16, 0.3) fermée — 30 mailles fermées"), "{log}");
    let last = closed.rows.last().unwrap();
    assert!(last[2].abs() < 1e-9 && closed.rows.iter().all(|r| r[3] == 0.0 && r[4] == 0.0), "fermée : {} kg/m passés", last[2]);
    let half = |r: &Runner, what: &str| -> f64 {
        let log = r.log.join("\n");
        assert!(log.contains("ouverte à 50 % — 15 mailles fermées"), "{what}\n{log}");
        let passed = r.rows.last().unwrap()[2];
        let low: f64 = r.rows.iter().map(|row| row[3]).fold(0.0, f64::max);
        assert!(r.rows.iter().all(|row| row[4] == 0.0), "{what} : la moitié fermée laisse passer");
        eprintln!("{what} : vanne ouverte à moitié, {passed:.4} kg/m passés (débit du bas jusqu'à {low:.3} kg/(m·s))");
        assert!(passed > 1.0 && low > 1.0, "{what} : {passed} kg/m passés");
        passed
    };
    let cpu = half(&run("ouverte 0.5", false), "CPU");
    let probe = build_scenario(MpScenario::DamBreak, 40, 30, 0.01, 9.81);
    if super::mp_gpu_tests::gpu_or_skip(&probe).is_none() {
        return;
    }
    let gpu = half(&run("ouverte 0.5", true), "GPU");
    assert!((gpu / cpu - 1.0).abs() < 0.1, "GPU {gpu} / CPU {cpu}");
}

#[test]
fn a_dated_action_happens_once_on_time() {
    let p = proto("départ scénario \"cuve vide\"\nà 0.05 s : marque \"x\"\nà 20 ms : remplir rect (0, 0) (0.1, 0.02) avec \"eau\"\n");
    let acts: Vec<&Action> = p.steps.iter().filter_map(|(s, _)| if let Step::Do(a) = s { Some(a) } else { None }).collect();
    assert!(
        matches!(acts[0], Action::Periodic { period, start: Some(s), end: Some(e), action } if period.is_infinite() && *s == 0.05 && *e == 0.05 && matches!(**action, Action::Mark(_))),
        "{:?}",
        acts[0]
    );
    assert!(matches!(acts[1], Action::Periodic { start: Some(s), action, .. } if (*s - 0.02).abs() < 1e-15 && matches!(**action, Action::Fill { .. })));
    for (bad, what) in [("à 1 s marque \"x\"", "« : »"), ("à 1 s : simuler 1 s", "une seule action"), ("à : marque \"x\"", "instant")] {
        let e = parse_all(&format!("départ scénario \"cuve vide\"\n{bad}\n")).unwrap_err();
        assert!(e[0].line == 2 && e[0].msg.contains(what), "{bad} : {e:?}");
    }
    // à l'heure pile, une seule fois ; un instant déjà passé quand la ligne est atteinte : tout de suite, noté
    let text = "départ scénario \"cuve vide\" grille 20 x 20 maille 1 cm
à 0.05 s : marque \"top\"
simuler 0.1 s
à 0.05 s : marque \"tard\"
simuler 0.05 s
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let log = r.log.join("\n");
    assert!(log.contains("[t = 0.050000 s] ligne 2 : repère « top »") && log.matches("repère « top »").count() == 1, "{log}");
    assert!(log.contains("[t = 0.100000 s] ligne 4 : à 0.05 s — instant déjà passé") && log.contains("[t = 0.100000 s] ligne 4 : repère « tard »"), "{log}");
    assert!(log.contains("ligne 2 : à 0.05 s — faite"), "{log}");
}

#[test]
fn periodic_actions_repeat_on_time_alongside_the_protocol() {
    // lecture : période, début et fin facultatifs, puis l'action après « : »
    let p = proto(
        "départ scénario \"cuve vide\"
toutes les 0.5 s à partir de 1 s jusqu'à 3 s : chauffer rect (0, 0) (0.1, 0.02) à 600 K pendant 0.1 s
chaque 20 ms : bord haut ouvert
",
    );
    let acts: Vec<&Action> = p.steps.iter().filter_map(|(s, _)| if let Step::Do(a) = s { Some(a) } else { None }).collect();
    assert!(
        matches!(acts[0], Action::Periodic { period, start: Some(s), end: Some(e), action } if *period == 0.5 && *s == 1.0 && *e == 3.0 && matches!(**action, Action::Heat { duration: Some(d), .. } if d == 0.1)),
        "{:?}",
        acts[0]
    );
    assert!(matches!(acts[1], Action::Periodic { period, start: None, end: None, action } if (*period - 0.02).abs() < 1e-15 && matches!(**action, Action::Boundary { side: 3, .. })));
    for (bad, what) in [
        ("toutes les 0 s : effacer partout", "positive"),
        ("toutes les 1 s effacer partout", "« : »"),
        ("toutes les 1 s :", "il manque l'action"),
        ("toutes les 1 s : simuler 1 s", "une seule action"),
        ("toutes les 1 s : toutes les 2 s : effacer partout", "une seule action"),
        ("toutes les 1 s : mesurer toutes les 1 ms : stats", "une seule action"),
        ("toutes 1 s : effacer partout", "toutes les"),
        ("toutes les 1 s : voler", "commande"),
    ] {
        let e = parse_all(&format!("départ scénario \"cuve vide\"\n{bad}\n")).unwrap_err();
        assert!(e[0].line == 2 && e[0].msg.contains(what), "{bad} : {e:?}");
    }

    // à l'heure pile : repères à 0.05, 0.15, 0.25 et 0.35 s (le pas s'arrête sur chaque échéance) ; trois notés, puis le compte
    let text = "départ scénario \"cuve vide\" grille 20 x 20 maille 1 cm
toutes les 0.1 s à partir de 0.05 s jusqu'à 0.35 s : marque \"top\"
simuler 0.5 s
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let log = r.log.join("\n");
    for t in ["0.050000", "0.150000", "0.250000"] {
        assert!(log.contains(&format!("[t = {t} s] ligne 2 : repère « top »")), "{t}\n{log}");
    }
    assert!(!log.contains("[t = 0.350000 s] ligne 2 : repère") && log.contains("répétitions suivantes non notées"), "{log}");
    assert!(log.contains("ligne 2 : toutes les 0.1 s — 4 fois"), "{log}");

    // vanne : la sortie du fond est ouverte 0.1 s sur 0.2 s — la cuve perd la moitié de ce qu'elle perdrait toujours ouverte
    let text = "départ scénario \"cuve vide\" grille 20 x 30 maille 1 cm
remplir rect (0, 0) (0.2, 0.2) avec \"eau\"
toutes les 0.2 s : bord bas sortie (0, -0.05)
toutes les 0.2 s à partir de 0.1 s : bord bas mur
mesurer toutes les 0.1 s : masse \"eau\"
simuler 1 s
";
    let check = |r: &Runner, what: &str| {
        let log = r.log.join("\n");
        assert!(log.contains("ligne 3 : toutes les 0.2 s — 6 fois") && log.contains("ligne 4 : toutes les 0.2 s — 5 fois"), "{what}\n{log}");
        assert!(log.contains("[t = 0.100000 s] ligne 4 : bord bas : mur") && log.contains("[t = 0.200000 s] ligne 3 : bord bas : sortie"), "{what}\n{log}");
        let eau = r.sim.reg.phases.iter().position(|p| p.name == "eau").unwrap();
        let full = r.sim.reg.phases[eau].rho(101325.0, 293.15) * 0.05 * 0.2 * 1.0;
        let lost = r.rows[0][2] - r.rows.last().unwrap()[2];
        eprintln!("{what} : vanne ouverte une demi-période sur deux, {lost:.4} kg/m sortis (moitié du débit continu : {:.4})", 0.5 * full);
        assert!((lost / (0.5 * full) - 1.0).abs() < 0.02, "{what} : {lost} kg/m sortis pour {}", 0.5 * full);
    };
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    check(&r, "CPU");
    let probe = build_scenario(MpScenario::DamBreak, 40, 30, 0.01, 9.81);
    if super::mp_gpu_tests::gpu_or_skip(&probe).is_none() {
        return;
    }
    let mut g = Runner::new(proto(text), Path::new("."), None, Some(true)).unwrap();
    assert!(g.gpu.is_some(), "{}", g.log.join("\n"));
    run_to_end(&mut g);
    check(&g, "GPU");
}

#[test]
fn a_teleporter_carries_what_it_draws_in_to_its_destination() {
    use super::mp_cells::MpEffect;
    // téléporteur au fond de l'eau, destination dans l'air en haut à droite : ses mailles visent la maille de destination, et l'eau
    // aspirée y réapparaît
    let text = "départ scénario \"cuve vide\" grille 24 x 24 maille 1 cm
remplir rect (0, 0) (0.24, 0.12) avec \"eau\" à 20 °C
téléporteur rect (0.04, 0.03) (0.06, 0.05) vers (0.185, 0.185) 0.3 m/s rayon 1 cm
mesurer toutes les 0.1 s : masse \"eau\" dans disque (0.185, 0.185) rayon 4 cm comme \"arrivée\"
simuler 0.3 s
";
    let p = proto(text);
    let (s, _, _) = initial_state(&p, Path::new(".")).unwrap();
    let dst = s.idx(18, 18);
    let src: Vec<Option<MpEffect>> = s.special.iter().copied().filter(|e| matches!(e, Some(MpEffect::Teleporter { .. }))).collect();
    assert_eq!(src.len(), 4);
    assert!(src.iter().all(|e| *e == Some(MpEffect::Teleporter { dest: dst, speed: 0.3, radius: 1 })), "{src:?}");
    let mut r = Runner::new(p, Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let c = r.columns.iter().position(|c| c.starts_with("arrivée")).unwrap();
    let (first, last) = (r.rows[0][c], r.rows.last().unwrap()[c]);
    eprintln!("téléporteur : eau autour de la destination {first:.2e} → {last:.4} kg/m");
    assert!(first < 1e-6 && last > 1e-2, "rien n'est arrivé à destination : {first} → {last}");
    assert!(r.log.iter().any(|l| l.contains("téléporteur") && l.contains("vers (0.185, 0.185)")), "{}", r.log.join("\n"));
    // erreurs de lecture, puis de pose (destination hors du domaine, dans un solide, sous la zone)
    for (bad, what) in [
        ("téléporteur rect (0, 0) (0.02, 0.02)", "vers (x, y)"),
        ("téléporteur rect (0, 0) (0.02, 0.02) vers (0.1, 0.1) rayon -1 cm", "positif"),
        ("téléporteur rect (0, 0) (0.02, 0.02) vers (0.1, 0.1) vite", "vitesse d'aspiration"),
    ] {
        let e = parse(&format!("départ scénario \"cuve vide\"\n{bad}"), None).unwrap_err();
        assert!(e[0].line == 2 && e[0].msg.contains(what), "{bad} : {e:?}");
    }
    for (bad, what) in [
        ("téléporteur rect (0, 0) (0.02, 0.02) vers (0.5, 0.1)", "hors du domaine"),
        ("solide rect (0.1, 0.1) (0.12, 0.12) \"acier\"\ntéléporteur rect (0, 0) (0.02, 0.02) vers (0.11, 0.11)", "dans un solide"),
        ("puits rect (0.1, 0.1) (0.12, 0.12)
téléporteur rect (0, 0) (0.02, 0.02) vers (0.11, 0.11)", "autre cellule spéciale"),
    ] {
        let p = proto(&format!("départ scénario \"cuve vide\" grille 24 x 24 maille 1 cm\n{bad}\nsimuler 1 ms\n"));
        let e = initial_state(&p, Path::new(".")).err().unwrap_or_else(|| panic!("{bad} : pas d'erreur"));
        assert!(e.contains(what), "{bad} : {e}");
    }
}

#[test]
fn a_body_follows_motion_and_force_laws_written_as_formulas() {
    use super::mp_bodies::{LinLaw, RotLaw};
    // lecture : lois de translation, de rotation, force et couple d'un corps libre ; formules vérifiées à la lecture
    let p = proto(
        "départ scénario \"cuve vide\"
corps disque (0.1, 0.1) rayon 1 cm \"acier\" position formule \"x0 + 0.02*sin(2*pi*t/0.4)\" \"y0\" angle formule \"90*t\"
corps disque (0.2, 0.1) rayon 1 cm \"acier\" accélération formule \"-2*vx\" \"-9.81\" vitesse (0.3, 0) rotation formule \"360\"
corps disque (0.3, 0.1) rayon 1 cm \"acier\" libre force formule \"-100*(x - x0)\" \"-0.5*ay\" couple formule \"0\"
",
    );
    let motions: Vec<&super::mp_experiment::BodyMotion> =
        p.steps.iter().filter_map(|(s, _)| if let Step::Do(Action::Body { motion, .. }) = s { Some(motion) } else { None }).collect();
    assert!(matches!(&motions[0].lin_law, Some(LinLaw::Position(f)) if f[1] == "y0") && matches!(&motions[0].rot_law, Some(RotLaw::Angle(_))));
    assert!(matches!(&motions[1].lin_law, Some(LinLaw::Acceleration(_))) && motions[1].v == [0.3, 0.0] && matches!(&motions[1].rot_law, Some(RotLaw::Rate(_))));
    assert!(motions[2].free && motions[2].force.is_some() && motions[2].torque.as_deref() == Some("0"));
    let body = "corps disque (0.1, 0.1) rayon 1 cm \"acier\"";
    for (bad, what) in [
        ("position formule \"x0 + z\" \"y0\"", "« z » inconnu"),
        ("position \"x0\" \"y0\"", "il faut « position formule"),
        ("position formule \"x0\" \"y0\" vitesse formule \"1\" \"0\"", "une seule loi de translation"),
        ("angle formule \"t\" rotation formule \"1\"", "une seule loi de rotation"),
        ("libre position formule \"x0\" \"y0\"", "un corps libre suit l'écoulement"),
        ("force formule \"0\" \"0\"", "corps « libre »"),
        ("position formule \"x0\" \"y0\" vitesse (1, 0)", "deux translations"),
        ("rotation 30 angle formule \"t\"", "deux rotations"),
        ("position formule \"x0\" \"y0\" oscillation rotation 10 période 1 s", "oscillation"),
        ("pivot (0.1, 0.11) vitesse formule \"1\" \"0\"", "sur un pivot"),
    ] {
        let e = parse_all(&format!("départ scénario \"cuve vide\"\n{body} {bad}\n")).unwrap_err();
        assert!(e[0].line == 2 && e[0].msg.contains(what), "{bad} : {e:?}");
    }

    // mouvements imposés, comparés à leur solution exacte au bout de 0.1 s : position et angle posés à leur valeur, vitesse qui
    // dépend de la position et accélération qui dépend de la vitesse intégrées (Runge–Kutta 4)
    let text = "départ scénario \"cuve vide\" grille 40 x 20 maille 1 cm
corps disque (0.1, 0.1) rayon 1 cm \"acier\" position formule \"x0 + 0.02*sin(2*pi*t/0.4)\" \"y0\" angle formule \"90*t\"
corps disque (0.25, 0.15) rayon 1 cm \"acier\" accélération formule \"-2*vx\" \"-9.81\" vitesse (0.3, 0)
corps disque (0.35, 0.05) rayon 1 cm \"acier\" vitesse formule \"-5*(x - 0.32)\" \"0\" rotation formule \"360\"
simuler 0.1 s
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let b = &r.sim.bodies;
    let close = |a: f64, b: f64, tol: f64, what: &str| assert!((a - b).abs() < tol, "{what} : {a} au lieu de {b}");
    close(b[0].pos[0], 0.12, 1e-12, "position imposée");
    close(b[0].vel[0], 0.0, 1e-6, "vitesse d'une position imposée (dérivée)");
    close(b[0].angle, 9f64.to_radians(), 1e-12, "angle imposé");
    close(b[1].pos[0], 0.25 + 0.15 * (1.0 - (-0.2f64).exp()), 1e-9, "x d'une accélération -2 vx");
    close(b[1].pos[1], 0.15 - 0.5 * 9.81 * 0.01, 1e-9, "chute à g");
    close(b[1].vel[0], 0.3 * (-0.2f64).exp(), 1e-9, "vitesse amortie");
    close(b[1].vel[1], -0.981, 1e-9, "vitesse de chute");
    close(b[1].acc[1], -9.81, 1e-6, "accélération mesurée sur le pas");
    close(b[2].pos[0], 0.32 + 0.03 * (-0.5f64).exp(), 1e-9, "vitesse rappelée vers x = 0.32");
    close(b[2].angle, 36f64.to_radians(), 1e-12, "rotation imposée");

    // corps libre rappelé par un ressort (force en formule), lancé à 0.2 m/s dans l'air : il s'arrête au quart de période
    // T/4 = (π/2) √(m/k), à A = v₀ √(m/k) de son départ
    let text = "départ scénario \"cuve vide\" grille 40 x 20 maille 1 cm
corps disque (0.2, 0.1) rayon 1 cm \"acier\" libre bloqué y et rotation vitesse (0.2, 0) force formule \"-100*(x - x0)\" \"0\"
simuler jusqu'à ce que corps 1 vx < 0 au plus 1 s
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let b = &r.sim.bodies[0];
    let m = b.mass(b.density(r.sim.reg.solids[b.material].rho));
    let w = (100.0 / m).sqrt();
    let hit = r.log.iter().find(|l| l.contains("corps 1 vx < 0 m/s atteint")).unwrap_or_else(|| panic!("{}", r.log.join("\n")));
    let t = hit[hit.find("t = ").unwrap() + 4..hit.find(" s]").unwrap()].trim().parse::<f64>().unwrap();
    eprintln!("ressort : arrêt à t = {t:.4} s (T/4 = {:.4}), x − x0 = {:.5} m (A = {:.5})", 0.5 * std::f64::consts::PI / w, b.pos[0] - 0.2, 0.2 / w);
    assert!((t * w / (0.5 * std::f64::consts::PI) - 1.0).abs() < 0.02, "quart de période {t} pour {}", 0.5 * std::f64::consts::PI / w);
    assert!(((b.pos[0] - 0.2) * w / 0.2 - 1.0).abs() < 0.02, "amplitude {} pour {}", b.pos[0] - 0.2, 0.2 / w);
}

#[test]
fn an_outlet_draws_the_fluid_out_at_its_imposed_speed() {
    // lecture : vitesse dirigée vers l'extérieur du bord ; « sortie » seule reste un bord ouvert
    let p = proto("départ scénario \"cuve vide\"\nbord bas sortie (0, -0.05)\nbord droite sortie\nbord haut sortie (0.1, 2 cm/s)\n");
    let acts: Vec<&Action> = p.steps.iter().filter_map(|(s, _)| if let Step::Do(a) = s { Some(a) } else { None }).collect();
    assert!(matches!(acts[0], Action::Boundary { side: 2, spec: BoundarySpec::Outlet { v } } if *v == [0.0, -0.05]), "{:?}", acts[0]);
    assert!(matches!(acts[1], Action::Boundary { side: 1, spec: BoundarySpec::Open(p) } if *p == 101325.0), "{:?}", acts[1]);
    assert!(matches!(acts[2], Action::Boundary { side: 3, spec: BoundarySpec::Outlet { v } } if (v[1] - 0.02).abs() < 1e-12), "{:?}", acts[2]);
    for bad in ["bord bas sortie (0, 0.05)", "bord gauche sortie (0.1, 0)", "bord droite sortie (0, 0.3)"] {
        let e = parse_all(&format!("départ scénario \"cuve vide\"\n{bad}\n")).unwrap_err();
        assert!(e[0].line == 2 && e[0].msg.contains("vers l'extérieur"), "{bad} : {e:?}");
    }
    // vidange par le fond, haut ouvert : l'eau sort à ρ v L (le débit mesuré au bord du bas), quelle que soit la hauteur
    let text = "départ scénario \"cuve vide\" grille 20 x 30 maille 1 cm
remplir rect (0, 0) (0.2, 0.2) avec \"eau\"
bord bas sortie (0, -0.05)
mesurer toutes les 50 ms : masse \"eau\", débit massique \"eau\" à travers ligne (0, 0) (0.2, 0)
simuler 1 s
";
    let check = |r: &Runner, what: &str| {
        let log = r.log.join("\n");
        assert!(log.contains("bord bas : sortie à (0, -0.05) m/s"), "{log}");
        let eau = r.sim.reg.phases.iter().position(|p| p.name == "eau").unwrap();
        let rho = r.sim.reg.phases[eau].rho(101325.0, 293.15);
        let q = -rho * 0.05 * 0.2;
        // (premier intervalle : la mise en route de l'écoulement, sa vitesse partant de zéro)
        assert!((r.rows[1][3] / q - 1.0).abs() < 2e-2, "{what} : débit {} pour {q} au premier relevé", r.rows[1][3]);
        for row in &r.rows[2..] {
            assert!((row[3] / q - 1.0).abs() < 2e-3, "{what} : débit {} pour {q} à t = {}", row[3], row[0]);
        }
        let (first, last) = (&r.rows[0], r.rows.last().unwrap());
        let lost = first[2] - last[2];
        eprintln!("{what} : {lost:.5} kg/m sortis en {:.3} s (ρ v L t = {:.5})", last[0], -q * last[0]);
        assert!((lost / (-q * last[0]) - 1.0).abs() < 2e-3, "{what} : {lost} kg/m sortis pour {}", -q * last[0]);
    };
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    check(&r, "CPU");
    let probe = build_scenario(MpScenario::DamBreak, 40, 30, 0.01, 9.81);
    if super::mp_gpu_tests::gpu_or_skip(&probe).is_none() {
        return;
    }
    let mut g = Runner::new(proto(text), Path::new("."), None, Some(true)).unwrap();
    assert!(g.gpu.is_some(), "{}", g.log.join("\n"));
    run_to_end(&mut g);
    check(&g, "GPU");
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
        Step::UntilCond { clauses, logic: super::mp_experiment::Logic::Any(_), limit: Some(l) } if clauses.len() == 3 && *l == 2.0 => {
            assert!(matches!(clauses[0], Clause { measure: Measure::Probe { q: Quantity::T, reduce: Reduce::Max, .. }, cmp: Cmp::Gt, value: super::mp_experiment::Threshold::Num(value) } if value == 400.0));
            assert!(matches!(clauses[1], Clause { measure: Measure::Body { who: super::mp_experiment::Who::Rank(0), q: BodyQty::Y, .. }, cmp: Cmp::Lt, value: super::mp_experiment::Threshold::Num(value) } if (value - 0.05).abs() < 1e-12));
            assert!(matches!(clauses[2], Clause { measure: Measure::Body { q: BodyQty::Angle, .. }, cmp: Cmp::Gt, value: super::mp_experiment::Threshold::Num(value) } if (value - 1.5f64.to_degrees()).abs() < 1e-9));
        }
        s => panic!("{s:?}"),
    }
    assert!(matches!(&p.steps[1].0, Step::UntilCond { clauses, logic: super::mp_experiment::Logic::All(_), limit: None } if clauses.len() == 2));
    for (bad, what) in [
        ("simuler jusqu'à ce que corps 1 y < 0.1 et corps 1 x > 0 ou corps 1 vy < 0", "mettez des parenthèses"),
        ("simuler jusqu'à ce que (corps 1 y < 0.1 ou corps 1 x > 0", "il manque « ) »"),
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

/// instant d'une ligne du journal « [t = … s] … »
fn log_time(l: &str) -> f64 {
    l[l.find("t = ").unwrap() + 4..l.find(" s]").unwrap()].trim().parse().unwrap()
}

#[test]
fn a_when_rule_acts_each_time_its_condition_becomes_true() {
    // disque lâché dans l'air : « y < 15 cm » devient vrai une fois, à t = √(2 × 0.05 m / g) ; « y < 25 cm » l'est dès le départ
    // (faite tout de suite, une seule fois) ; lecture et erreurs
    let text = "départ scénario \"cuve vide\" grille 30 x 30 maille 1 cm
corps disque (0.15, 0.2) rayon 1 cm \"acier\" libre
quand corps 1 y < 25 cm : marque \"déjà\"
quand corps 1 y < 15 cm : marque \"passé\"
simuler 0.15 s
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let marks = |what: &str| -> Vec<f64> { r.log.iter().filter(|l| l.contains(&format!("repère « {what} »"))).map(|l| log_time(l)).collect() };
    let (deja, passe) = (marks("déjà"), marks("passé"));
    let expect = (2.0 * 0.05 / 9.81f64).sqrt();
    eprintln!("quand : « déjà » à {deja:?}, « passé » à {passe:?} (chute de 5 cm : {expect:.4} s)");
    assert!(deja == [0.0], "{deja:?}\n{}", r.log.join("\n"));
    // (au premier pas qui franchit le seuil, jamais avant)
    assert!(passe.len() == 1 && passe[0] >= expect - 2e-4 && passe[0] < expect + 5e-3, "{passe:?}\n{}", r.log.join("\n"));
    assert!(r.log.iter().any(|l| l.contains("ligne 4 : quand corps 1 y < 0.15 m — 1 fois")), "{}", r.log.join("\n"));
    for (bad, what) in [
        ("quand corps 1 y < 0.1 : simuler 1 s", "une seule action"),
        ("quand corps 1 y < 0.1", "« quand » sans son « fin »"),
        ("quand stats < 3 : marque \"x\"", "relevé"),
        ("toutes les 1 s : quand corps 1 y < 0.1 : marque \"x\"", "règle « quand »"),
        ("quand corps 1 y < 0.1 : quand corps 1 y > 0.2 : pause", "autre règle"),
    ] {
        let e = parse(&format!("départ scénario \"cuve vide\"\n{bad}"), None).unwrap_err();
        assert!(e[0].line == 2 && e[0].msg.contains(what), "{bad} : {e:?}");
    }
    // une pause sur un événement est permise
    assert!(parse("départ scénario \"cuve vide\"\nquand max T dans partout > 400 K : pause\n", None).is_ok());
}

/// cuve d'air fermée (18 × 16 mailles) gonflée par un compresseur dans sa paroi gauche (0.5 m/s sur 2 cm : la pression monte
/// d'environ γ p Q / V ≈ 49 kPa/s) ; soupape de 2 cm dans le couvercle, ouverte au-dessus de 104 kPa, refermée sous 102 kPa
const SAFETY_VALVE: &str = "départ scénario \"cuve vide\" grille 30 x 24 maille 1 cm
bord haut ouvert
pas max 2 ms
solide rect (0.05, 0.02) (0.25, 0.2) \"acier\"
effacer rect (0.06, 0.03) (0.24, 0.19)
effacer rect (0.05, 0.1) (0.06, 0.12)
pompe rect (0.05, 0.1) (0.06, 0.12) (0.5, 0) m/s
effacer rect (0.14, 0.19) (0.16, 0.2)
vanne rect (0.14, 0.19) (0.16, 0.2) fermée
quand p en (0.15, 0.1) > 104 kPa : vanne rect (0.14, 0.19) (0.16, 0.2) ouverte
quand p en (0.15, 0.1) < 102 kPa : vanne rect (0.14, 0.19) (0.16, 0.2) fermée
mesurer toutes les 2 ms : p en (0.15, 0.1) comme \"p cuve\"
simuler 0.3 s
";

/// (nombre d'ouvertures de la soupape, pression la plus haute, pression la plus basse après la première ouverture)
fn safety_valve_run(gpu: bool) -> (usize, f64, f64) {
    let mut r = Runner::new(proto(SAFETY_VALVE), Path::new("."), None, Some(gpu)).unwrap();
    assert!(r.gpu.is_some() == gpu, "{}", r.log.join("\n"));
    run_to_end(&mut r);
    let opened = r.log.iter().find(|l| l.contains("ligne 10 : quand") && l.ends_with(" fois")).unwrap_or_else(|| panic!("{}", r.log.join("\n")));
    let n: usize = opened[opened.find("— ").unwrap() + "— ".len()..opened.len() - " fois".len()].parse().unwrap();
    let c = r.columns.iter().position(|c| c.starts_with("p cuve")).unwrap();
    let first = r.log.iter().find(|l| l.contains("ligne 10 : condition vraie")).map(|l| log_time(l)).unwrap_or(f64::INFINITY);
    let pmax = r.rows.iter().map(|row| row[c]).fold(f64::MIN, f64::max);
    let pmin = r.rows.iter().filter(|row| row[0] > first).map(|row| row[c]).fold(f64::MAX, f64::min);
    eprintln!("soupape ({}) : {n} ouvertures, la première à {first:.4} s ; p entre {pmin:.0} et {pmax:.0} Pa", if gpu { "GPU" } else { "CPU" });
    (n, pmax, pmin)
}

#[test]
fn when_rules_make_a_safety_valve_that_holds_the_pressure_of_a_pumped_vessel() {
    // sans soupape la pression monterait à ≈ 116 kPa en 0.3 s ; avec, elle reste dans la fourchette, la soupape s'ouvrant et se
    // refermant plusieurs fois
    let (n, pmax, pmin) = safety_valve_run(false);
    assert!(n >= 5, "la soupape ne s'est ouverte que {n} fois");
    assert!(pmax < 104.5e3 && pmin > 101.5e3, "pression entre {pmin} et {pmax} Pa");
    // le GPU suit les mêmes règles (relevées après chaque série de pas, pas à pas près du seuil et après chaque action)
    let probe = build_scenario(MpScenario::DamBreak, 40, 30, 0.01, 9.81);
    if super::mp_gpu_tests::gpu_or_skip(&probe).is_none() {
        return;
    }
    let (ng, pmax_g, pmin_g) = safety_valve_run(true);
    assert!(ng + 1 >= n && ng <= n + 1, "ouvertures : CPU {n}, GPU {ng}");
    assert!(pmax_g < 104.5e3 && pmin_g > 101.5e3, "GPU : pression entre {pmin_g} et {pmax_g} Pa");
}

/// déroule un protocole jusqu'au bout, ou jusqu'à son échec (sans paniquer) : le message d'échec
fn run_until_failure(r: &mut Runner) -> Option<String> {
    loop {
        match r.advance(Duration::from_secs(3600)).clone() {
            RunState::Running => {}
            RunState::Paused(_) => r.resume(),
            RunState::Finished => return None,
            RunState::Failed(e) => return Some(e),
        }
    }
}

#[test]
fn a_while_loop_repeats_its_body_and_wait_waits_for_a_time_or_a_condition() {
    // disque lâché à 25 cm : « tant que y > 10 cm » attend 20 ms par tour — y(0.16 s) = 12.4 cm, y(0.18 s) = 9.1 cm : 9 tours,
    // sortie à 0.18 s pile ; puis « attendre » une condition (y < 5 cm à t = √(0.4 m / g) ≈ 0.2019 s), un instant, une durée
    let text = "départ scénario \"cuve vide\" grille 30 x 30 maille 1 cm
corps disque (0.15, 0.25) rayon 1 cm \"acier\" libre
mesurer toutes les 0.1 s : temps comme \"horloge\", corps 1 y
tant que corps 1 y > 10 cm
    attendre 20 ms
    marque \"tour\"
fin
marque \"sorti\"
attendre que corps 1 y < 5 cm
marque \"bas\"
attendre jusqu'à 0.3 s
attendre 10 ms
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let marks = |r: &Runner, what: &str| -> Vec<f64> { r.log.iter().filter(|l| l.contains(&format!("repère « {what} »"))).map(|l| log_time(l)).collect() };
    let (tours, sorti, bas) = (marks(&r, "tour"), marks(&r, "sorti"), marks(&r, "bas"));
    eprintln!("tant que : tours à {tours:?}, sortie à {sorti:?}, « bas » à {bas:?}");
    // (trois tours notés, puis le corps se tait au journal, comme une répétition ; le compte des tours est à la sortie)
    assert!(tours.len() == 3 && tours.iter().enumerate().all(|(k, &t)| (t - 0.02 * (k + 1) as f64).abs() < 1e-9), "{tours:?}\n{}", r.log.join("\n"));
    assert!(r.log.iter().any(|l| l.contains("ligne 4 : tours suivants non notés au journal")), "{}", r.log.join("\n"));
    assert!(sorti.len() == 1 && (sorti[0] - 0.18).abs() < 1e-9, "{sorti:?}");
    let expect = (0.4 / 9.81f64).sqrt();
    assert!(bas.len() == 1 && bas[0] >= expect - 2e-4 && bas[0] < expect + 5e-3, "{bas:?} pour {expect}");
    assert!((r.sim.time - 0.31).abs() < 1e-9, "fin à {}", r.sim.time);
    assert!(r.log.iter().any(|l| l.contains("ligne 4 : tant que corps 1 y > 0.1 m") && l.contains("fini après 9 tours")), "{}", r.log.join("\n"));
    // le relevé « temps » redonne l'instant du relevé
    let c = r.columns.iter().position(|c| c == "horloge (s)").unwrap();
    assert!(r.rows.len() >= 3 && r.rows.iter().all(|row| row[c] == row[0]), "{:?}", r.rows);

    // boucles imbriquées, « au plus » ; une règle écrite dans une boucle n'est posée qu'une fois
    let text = "départ scénario \"cuve vide\" grille 20 x 20 maille 1 cm
tant que temps < 0.1 s
    quand temps > 0.02 s : marque \"règle\"
    tant que temps < 0.05 s
        attendre 10 ms
    fin
    attendre 20 ms
fin
tant que temps > 0 au plus 50 ms
    attendre 20 ms
fin tant que
";
    // fin prévue (barre de progression) : corps d'une boucle sans borne compté une fois (10 + 20 ms), boucle bornée pour sa durée
    // permise (50 ms)
    assert!((proto(text).planned_end(0.0) - 0.08).abs() < 1e-12, "{}", proto(text).planned_end(0.0));
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let log = r.log.join("\n");
    let count = |pat: &[&str]| r.log.iter().filter(|l| pat.iter().all(|p| l.contains(p))).count();
    assert!((r.sim.time - 0.17).abs() < 1e-9, "fin à {}\n{log}", r.sim.time);
    assert_eq!(count(&["ligne 2 : tant que temps < 0.1 s", "fini après 3 tours"]), 1, "{log}");
    assert_eq!(count(&["ligne 4 : tant que temps < 0.05 s", "fini après 5 tours"]), 1, "{log}");
    assert_eq!(count(&["ligne 4 : tant que temps < 0.05 s", "aucun tour"]), 2, "{log}");
    assert_eq!(count(&["ligne 9 : tant que temps > 0 s", "fini après 3 tours", "au bout des 0.05 s permises"]), 1, "{log}");
    assert_eq!(count(&["ligne 3 : quand temps > 0.02 s"]), 2, "règle posée une fois (plus son bilan) :\n{log}");
    assert_eq!(marks(&r, "règle").len(), 1, "{log}");

    // lecture : erreurs
    for (bad, line, what) in [
        ("tant que temps < 1 s\nmarque \"x\"\nfin", 2, "faire avancer le temps"),
        ("fin", 2, "sans bloc ouvert"),
        ("tant que temps < 1 s\nattendre 1 ms", 2, "sans son « fin »"),
        ("tant temps < 1 s\nattendre 1 ms\nfin", 2, "il faut « tant que"),
        ("toutes les 1 s : tant que temps < 1 s", 2, "une seule action"),
        ("quand temps > 1 s : attendre 1 ms", 2, "une seule action"),
        ("attendre", 2, "mesure"),
        ("attendre que temps < 1 s au plus 0 s", 2, "positive"),
    ] {
        let e = parse(&format!("départ scénario \"cuve vide\"\n{bad}"), None).unwrap_err();
        assert!(e.len() == 1 && e[0].line == line && e[0].msg.contains(what), "{bad} : {e:?}");
    }
    // une boucle dont les attentes sont déjà atteintes arrête l'expérience au lieu de tourner sans fin
    let p = proto("départ scénario \"cuve vide\" grille 10 x 10 maille 1 cm\ntant que temps < 1 s\n    attendre jusqu'à 0.02 s\nfin\n");
    let mut r = Runner::new(p, Path::new("."), None, Some(false)).unwrap();
    let e = run_until_failure(&mut r).expect("la boucle aurait dû arrêter l'expérience");
    assert!(e.contains("sans que le temps avance") && (r.sim.time - 0.02).abs() < 1e-9, "{e}");
    // un geste de la fenêtre s'inscrit avant la première ligne qui fait avancer le temps : simuler, attendre ou tant que
    for first in ["tant que temps < 1 s\n    attendre 0.1 s\nfin", "attendre 0.1 s", "simuler 0.1 s"] {
        let (t, n) = insert_initial_line(&format!("départ scénario \"cuve vide\"\nremplir partout avec \"air\"\n{first}\n"), "solide rect (0, 0) (0.1, 0.1) \"acier\"");
        assert!(n == 3 && t.lines().nth(2) == Some("solide rect (0, 0) (0.1, 0.1) \"acier\""), "{t}");
    }
}

#[test]
fn if_else_and_repeat_blocks_choose_and_count_their_steps() {
    // « répéter 4 fois » : à 10, 20, 30 et 40 ms, « si temps < 15 ms » (a), « sinon si temps < 35 ms » (b), « sinon » (c) ; un « si »
    // faux sans « sinon » saute son bloc ; blocs imbriqués
    let text = "départ scénario \"cuve vide\" grille 10 x 10 maille 1 cm
répéter 4 fois
    attendre 10 ms
    si temps < 15 ms
        marque \"a\"
    sinon si temps < 35 ms
        marque \"b\"
    sinon
        marque \"c\"
    fin si
    si temps > 1 s
        marque \"jamais\"
    fin
fin répéter
marque \"après\"
";
    // fin prévue : 4 tours de 10 ms
    assert!((proto(text).planned_end(0.0) - 0.04).abs() < 1e-12, "{}", proto(text).planned_end(0.0));
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let marks: Vec<(String, f64)> = r
        .log
        .iter()
        .filter_map(|l| l.find("repère « ").map(|i| (l[i + "repère « ".len()..l.rfind(" »").unwrap()].to_string(), log_time(l))))
        .collect();
    eprintln!("si / sinon : {marks:?}");
    let names: Vec<&str> = marks.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["a", "b", "b", "après"], "{}", r.log.join("\n"));
    for ((_, t), want) in marks.iter().zip([0.01, 0.02, 0.03, 0.04]) {
        assert!((t - want).abs() < 1e-9, "{marks:?}");
    }
    let log = r.log.join("\n");
    // (le quatrième tour se tait au journal : « c » n'est pas noté, mais il a été fait — seul le compte des tours le dit)
    assert!(log.contains("ligne 2 : tours suivants non notés") && log.contains("ligne 2 : répéter 4 fois — fini après 4 tours"), "{log}");
    assert!(log.contains("ligne 4 : si temps < 0.015 s (0.01 s) — vrai") && log.contains("ligne 6 : sinon si temps < 0.035 s (0.02 s) — vrai"), "{log}");
    assert!((r.sim.time - 0.04).abs() < 1e-9);
    // lecture : erreurs
    for (bad, line, what) in [
        ("sinon", 2, "hors d'un « si »"),
        ("si temps < 1 s\nattendre 1 ms\nsinon\nattendre 1 ms\nsinon si temps < 2 s\nfin", 6, "dernière branche"),
        ("répéter 0 fois\nattendre 1 ms\nfin", 2, "entier positif"),
        ("répéter 2\nattendre 1 ms\nfin", 2, "répéter <n> fois"),
        ("si temps < 1 s\nattendre 1 ms", 2, "« si » sans son « fin »"),
        ("répéter 2 fois\nattendre 1 ms", 2, "« répéter » sans son « fin »"),
        ("quand temps > 1 s\nmarque \"x\"", 2, "« quand » sans son « fin »"),
        ("fin", 2, "sans bloc ouvert"),
    ] {
        let e = parse(&format!("départ scénario \"cuve vide\"\n{bad}"), None).unwrap_err();
        assert!(e.len() == 1 && e[0].line == line && e[0].msg.contains(what), "{bad} : {e:?}");
    }
}

#[test]
fn a_when_block_runs_in_parallel_each_time_and_skips_while_still_running() {
    // eau versée puis retirée toutes les 20 ms (trois fois) ; deux règles en bloc sur la masse d'eau : une suite courte (15 ms)
    // reprend à chaque versement ; une longue (25 ms) tourne encore au deuxième : sautée, puis relancée au troisième, que la fin du
    // protocole interrompt. Les attentes des suites tombent à l'heure pile, en parallèle du protocole
    let text = "départ scénario \"cuve vide\" grille 10 x 10 maille 1 cm
pas max 1 ms
quand masse \"eau\" > 0.2
    marque \"court\"
    attendre 15 ms
    marque \"court fini\"
fin quand
quand masse \"eau\" > 0.2
    marque \"long\"
    attendre 25 ms
    marque \"long fini\"
fin
répéter 3 fois
    remplir rect (0, 0) (0.02, 0.02) avec \"eau\"
    attendre 10 ms
    remplir rect (0, 0) (0.02, 0.02) avec \"air\"
    attendre 10 ms
fin
";
    // les conditions initiales sautent le corps des blocs : seul « pas max » (sans ce saut, la marque du premier bloc en serait)
    let p = proto(text);
    assert_eq!(p.initial_actions().0.len(), 1);
    assert!((p.planned_end(0.0) - 0.06).abs() < 1e-12, "{}", p.planned_end(0.0));
    let mut r = Runner::new(p, Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let log = r.log.join("\n");
    let marks = |what: &str| -> Vec<f64> { r.log.iter().filter(|l| l.contains(&format!("repère « {what} »"))).map(|l| log_time(l)).collect() };
    eprintln!("quand en bloc : court {:?} / {:?}, long {:?} / {:?}", marks("court"), marks("court fini"), marks("long"), marks("long fini"));
    let near = |a: &[f64], b: &[f64]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9);
    assert!(near(&marks("court"), &[0.001, 0.021, 0.041]) && near(&marks("court fini"), &[0.016, 0.036, 0.056]), "{log}");
    assert!(near(&marks("long"), &[0.001, 0.041]) && near(&marks("long fini"), &[0.026]), "{log}");
    assert!(log.contains("ligne 3 : quand masse eau partout > 0.2 kg/m — 3 fois"), "{log}");
    assert!(log.contains("ligne 8 : quand masse eau partout > 0.2 kg/m — 2 fois, 1 fois sautée"), "{log}");
    assert!(log.contains("ligne 8 : suite de la règle « quand » arrêtée par la fin du protocole (à la ligne 11)"), "{log}");
    assert!((r.sim.time - 0.06).abs() < 1e-9, "fin à {}", r.sim.time);
    // un bloc dont la condition est vraie dès sa ligne démarre tout de suite ; ses lignes ne sont pas des conditions initiales
    let text = "départ scénario \"cuve vide\" grille 10 x 10 maille 1 cm
quand temps >= 0 s
    remplir partout avec \"eau\"
fin
remplir rect (0, 0) (0.02, 0.02) avec \"eau\"
simuler 1 ms
";
    let (s, _, _) = initial_state(&proto(text), Path::new(".")).unwrap();
    let eau = s.reg.phases.iter().position(|p| p.name == "eau").unwrap();
    let cells = s.mass[eau].iter().filter(|&&m| m > 0.0).count();
    assert_eq!(cells, 4, "conditions initiales : seul le remplissage hors du bloc");
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    assert!(r.sim.mass[eau].iter().all(|&m| m > 0.0), "le bloc a rempli tout le domaine au départ");
}

#[test]
fn conditions_take_parentheses_and_variables_count_store_and_compare() {
    use super::mp_experiment::Logic;
    let marks = |r: &Runner| -> Vec<(String, f64)> {
        r.log.iter().filter_map(|l| l.find("repère « ").map(|i| (l[i + "repère « ".len()..l.rfind(" »").unwrap()].to_string(), log_time(l)))).collect()
    };
    // parenthèses : (A et B) ou C, relevé à 20, 30 et 50 ms
    let text = "départ scénario \"cuve vide\" grille 10 x 10 maille 1 cm
attendre 20 ms
si (temps < 0.025 s et temps > 0.015 s) ou temps > 0.045 s
    marque \"a\"
fin
attendre 10 ms
si (temps < 0.025 s et temps > 0.015 s) ou temps > 0.045 s
    marque \"b\"
fin
attendre 20 ms
si (temps < 0.025 s et temps > 0.015 s) ou temps > 0.045 s
    marque \"c\"
fin
";
    let p = proto(text);
    assert!(
        matches!(&p.steps[1].0, Step::If { logic: Logic::Any(v), .. } if v.len() == 2 && v[0] == Logic::All(vec![Logic::Leaf(0), Logic::Leaf(1)]) && v[1] == Logic::Leaf(2)),
        "{:?}",
        p.steps[1].0
    );
    let mut r = Runner::new(p, Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let m = marks(&r);
    assert!(m.len() == 2 && m[0].0 == "a" && (m[0].1 - 0.02).abs() < 1e-9 && m[1].0 == "c" && (m[1].1 - 0.05).abs() < 1e-9, "{m:?}");
    assert!(r.log.iter().any(|l| l.contains("si (temps < 0.025 s et temps > 0.015 s) ou temps > 0.045 s")), "{}", r.log.join("\n"));

    // variables : un compteur qu'une règle incrémente, un instant relevé repris dans des seuils calculés, une colonne de mesure, des
    // repères qui montrent leurs valeurs
    let text = "départ scénario \"cuve vide\" grille 10 x 10 maille 1 cm
pas max 1 ms
soit n = 0
soit t0 = temps
soit fin_attente = \"t0 + 0.02\"
mesurer toutes les 10 ms : n, temps
quand temps > \"fin_attente\" : soit n = \"n + 1\"
répéter 3 fois
    attendre 10 ms
fin
si n >= 1 et n < 2
    marque \"n vaut {n}\"
fin
soit seuil = 104 kPa
soit a = 30 °
attendre que temps > \"t0 + 0.05\"
marque \"fin : n = {n}, seuil = {seuil}, a = {a}\"
";
    let p = proto(text);
    assert_eq!(p.vars, ["n", "t0", "fin_attente", "seuil", "a"]);
    let mut r = Runner::new(p, Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let log = r.log.join("\n");
    let m = marks(&r);
    eprintln!("variables : {m:?}");
    assert!(m.len() == 2 && m[0].0 == "n vaut 1" && (m[0].1 - 0.03).abs() < 1e-9, "{m:?}\n{log}");
    assert_eq!(m[1].0, "fin : n = 1, seuil = 104000, a = 30", "{log}");
    // (premier pas où temps > 0.05 s : celui de 50 ms, à l'arrondi des sommes de pas près)
    assert!(m[1].1 >= 0.05 - 1e-9 && m[1].1 < 0.0515, "{m:?}");
    assert!(log.contains("ligne 7 : soit n = 1") && log.contains("ligne 7 : quand temps > « fin_attente » (0.02 s) — 1 fois"), "{log}");
    // colonne de la variable : 0, puis 1 dès que la règle a joué (juste après 20 ms)
    let c = r.columns.iter().position(|c| c == "n").unwrap_or_else(|| panic!("{:?}", r.columns));
    let at = |t: f64| r.rows.iter().find(|row| (row[0] - t).abs() < 1e-9).map(|row| row[c]);
    assert!(at(0.01) == Some(0.0) && at(0.02) == Some(0.0) && at(0.03) == Some(1.0) && at(0.05) == Some(1.0), "{:?}", r.rows);

    // lecture : erreurs
    for (bad, what) in [
        ("soit t = 3", "mot du protocole"),
        ("soit n 3", "soit n = <valeur>"),
        ("soit n = \"m + 1\"", "formule « m + 1 »"),
        ("soit n = stats", "pas une valeur"),
        ("marque \"{x}\"", "« x » inconnu"),
        ("si n > 0\nattendre 1 ms\nfin", "inconnue"),
    ] {
        let e = parse(&format!("départ scénario \"cuve vide\"\n{bad}"), None).unwrap_err();
        assert!(e[0].line == 2 && e[0].msg.contains(what), "{bad} : {e:?}");
    }
}

#[test]
fn computed_values_in_braces_are_read_when_the_action_is_done() {
    // « répéter {2 + 1} fois » : au tour k, gravité (0, −2k) et attente de k ms — à la fin, (0, −6) au bout de 6 ms
    let text = "départ scénario \"cuve vide\" grille 10 x 10 maille 1 cm
soit k = 0
répéter {2 + 1} fois
    soit k = {k + 1}
    gravité (0, {-2*k})
    attendre {k} ms
fin
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let log = r.log.join("\n");
    assert!((r.sim.time - 0.006).abs() < 1e-9 && r.sim.gravity == [0.0, -6.0], "t = {}, g = {:?}\n{log}", r.sim.time, r.sim.gravity);
    assert!(log.contains("ligne 3 : répéter 3 fois — fini après 3 tours") && log.contains("ligne 5 : gravité"), "{log}");
    // dans une règle, les valeurs sont celles de l'instant où elle joue (vent = 5 posé après elle)
    let text = "départ scénario \"cuve vide\" grille 10 x 10 maille 1 cm
pas max 1 ms
soit vent = 1
quand temps > 0.0025 s : gravité ({vent}, 0)
soit vent = 5
attendre 5 ms
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    assert_eq!(r.sim.gravity, [5.0, 0.0], "{}", r.log.join("\n"));
    // dans une répétition : relue à chaque fois (t en ms), seuil et nombre de tours calculés
    let text = "départ scénario \"cuve vide\" grille 10 x 10 maille 1 cm
toutes les 1 ms : gravité ({1000*t}, 0)
attendre que temps > {0.0025}
répéter {0} fois
    attendre 1 ms
fin
attendre jusqu'à 3.5 ms
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let log = r.log.join("\n");
    // (fois à 0, 1, 2 et 3 ms : la dernière donne 3)
    assert!((r.sim.gravity[0] - 3.0).abs() < 1e-9 && log.contains("ligne 4 : répéter 0 fois — aucun tour"), "{:?}\n{log}", r.sim.gravity);
    // lecture : erreurs
    for (bad, what) in [
        ("toutes les {1} s : marque \"x\"", "ne va pas ici"),
        ("gravité (0, {m})", "formule « m »"),
        ("gravité (0, {-9.81)", "accolade non refermée"),
        ("chauffer {1}", "zone"),
        ("simuler jusqu'à {2} s", "ne va pas ici"),
    ] {
        let e = parse(&format!("départ scénario \"cuve vide\"\n{bad}"), None).unwrap_err();
        assert!(e[0].line == 2 && e[0].msg.contains(what), "{bad} : {e:?}");
    }
    // une valeur qui n'est pas un nombre arrête l'expérience
    let p = proto("départ scénario \"cuve vide\" grille 10 x 10 maille 1 cm\nsoit z = 0\ngravité (0, {1/z})\nattendre 1 ms\n");
    let mut r = Runner::new(p, Path::new("."), None, Some(false)).unwrap();
    let e = run_until_failure(&mut r).expect("la valeur infinie aurait dû arrêter l'expérience");
    assert!(e.contains("« 1/z »") && e.contains("il faut un nombre"), "{e}");
}

/// énergie cinétique (J/m) des pièces du vilebrequin k et de son volant
fn crank_energy(s: &MultiphaseSim, k: usize) -> f64 {
    let ck = &s.mechanisms[k];
    let mut e = 0.5 * ck.inertia * ck.omega * ck.omega;
    for i in ck.bodies() {
        let b = &s.bodies[i];
        let rho = b.density(s.reg.solids[b.material].rho);
        e += 0.5 * rho * b.shape.area() * (b.vel[0].powi(2) + b.vel[1].powi(2)) + 0.5 * rho * b.shape.polar_moment() * b.omega.powi(2);
    }
    e
}

fn column(r: &Runner, name: &str) -> usize {
    r.columns.iter().position(|c| c == name).unwrap_or_else(|| panic!("colonne « {name} » absente : {:?}", r.columns))
}

#[test]
fn a_driven_crank_moves_its_rod_and_piston_along_the_slider_crank_law() {
    // vilebrequin mené à 300 tr/min (1800 °/s) : l'axe du piston à s(θ) = r cos θ + √(l² − r² sin² θ) au-dessus de l'axe du
    // vilebrequin, le centre du piston une demi-hauteur plus haut ; à 0.1 s, point mort bas (θ = 180°)
    let text = "départ scénario \"cuve vide\" grille 30 x 40 maille 1 cm
vilebrequin (0.15, 0.08) rayon 2 cm bielle 6 cm \"acier\" rotation 300 tr/min
mesurer toutes les 10 ms : vilebrequin 1 angle, vilebrequin 1 tours, corps 3 y comme \"piston\"
simuler 0.1 s
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let (rr, l, h) = (0.02, 0.06, 0.02);
    let s = |th: f64| rr * th.cos() + (l * l - (rr * th.sin()).powi(2)).sqrt();
    let (ca, ct, cp) = (column(&r, "vilebrequin 1 angle (°)"), column(&r, "vilebrequin 1 tours (tr)"), column(&r, "piston (m)"));
    for row in &r.rows {
        let th = 1800f64.to_radians() * row[0];
        assert!((row[cp] - (0.08 + s(th) + h / 2.0)).abs() < 1e-9, "t = {} : piston à {} pour {}", row[0], row[cp], 0.08 + s(th) + h / 2.0);
        assert!((row[ca] - th.to_degrees().rem_euclid(360.0)).abs() < 1e-6, "{row:?}");
    }
    let last = r.rows.last().unwrap();
    assert!((last[0] - 0.1).abs() < 1e-9 && (last[ct] - 0.5).abs() < 1e-9 && (last[cp] - 0.13).abs() < 1e-9, "{last:?}");
    // la bielle relie l'axe de manivelle à l'axe du piston ; les trois pièces sont menées
    let ck = &r.sim.mechanisms[0];
    let (a, b) = ck.crank.as_ref().unwrap().pins(ck.theta);
    assert!(((b[0] - a[0]).hypot(b[1] - a[1]) - l).abs() < 1e-12);
    assert!(r.sim.bodies.len() == 3 && r.sim.bodies.iter().all(|b| b.motion == super::mp_bodies::Motion::Driven));
    // lecture : erreurs
    for (bad, what) in [
        ("vilebrequin (0.1, 0.1) rayon 2 cm bielle 2 cm", "plus longue"),
        ("vilebrequin (0.1, 0.1) rayon 2 cm", "bielle"),
        ("vilebrequin (0.1, 0.1) rayon 2 cm bielle 6 cm inertie 0.1", "libre"),
        ("vilebrequin (0.1, 0.1) rayon 2 cm bielle 6 cm libre couple \"-0.1*vitesse\"", "couple"),
        ("vilebrequin (0.1, 0.1) rayon 2 cm bielle 6 cm volant", "option"),
    ] {
        let e = parse(&format!("départ scénario \"cuve vide\"\n{bad}"), None).unwrap_err();
        assert!(e[0].line == 2 && e[0].msg.contains(what), "{bad} : {e:?}");
    }
    let p = proto("départ scénario \"cuve vide\" grille 30 x 30 maille 1 cm\nvilebrequin (0.15, 0.1) rayon 2 cm bielle 6 cm axe 45 cylindre\nsimuler 1 ms\n");
    let e = initial_state(&p, Path::new(".")).err().unwrap();
    assert!(e.contains("0, 90, 180 ou 270"), "{e}");
}

#[test]
fn a_free_crank_keeps_its_kinetic_energy_and_its_speed_follows_its_inertia() {
    // vilebrequin libre lancé à 300 tr/min, sans pesanteur ni cylindre : ½ M(θ) ω² se conserve (l'air ne freine presque pas) ;
    // la vitesse varie au cours du tour, M(θ) avec elle (plus vite aux points morts, où le piston est arrêté)
    let text = "départ scénario \"cuve vide\" grille 30 x 40 maille 1 cm
gravité (0, 0)
vilebrequin (0.15, 0.08) rayon 2 cm bielle 6 cm \"acier\" libre rotation 300 tr/min inertie 0.002
mesurer toutes les 2 ms : vilebrequin 1 rotation, vilebrequin 1 angle
simuler 0.2 s
";
    let p = proto(text);
    let (s0, _, _) = initial_state(&p, Path::new(".")).unwrap();
    let e0 = crank_energy(&s0, 0);
    let mut r = Runner::new(p, Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let e1 = crank_energy(&r.sim, 0);
    let c = column(&r, "vilebrequin 1 rotation (°/s)");
    let (wmin, wmax) = r.rows.iter().map(|row| row[c]).fold((f64::MAX, f64::MIN), |(a, b), w| (a.min(w), b.max(w)));
    eprintln!("vilebrequin libre : énergie {e0:.5} → {e1:.5} J/m, rotation entre {wmin:.0} et {wmax:.0} °/s, {:.3} tour", r.sim.mechanisms[0].theta / std::f64::consts::TAU);
    assert!((e1 / e0 - 1.0).abs() < 1e-2, "énergie {e0} → {e1}");
    assert!(wmax / wmin > 1.1 && r.sim.mechanisms[0].theta > 1.5 * std::f64::consts::PI, "rotation {wmin} … {wmax}");
}

#[test]
fn a_piston_compresses_the_gas_of_its_cylinder_without_leaking() {
    // du point mort bas au point mort haut à 600 tr/min (0.05 s), le volume de la chambre est divisé par ε = (c + 2r)/c = 5 : sans
    // fuite autour du piston, la masse d'air reste, et p monte vers ε^γ ≈ 9.5 fois (adiabatique ; un peu moins, la chaleur passant
    // dans les parois d'acier)
    let text = "départ scénario \"cuve vide\" grille 30 x 40 maille 5 mm
vilebrequin (0.075, 0.05) rayon 2 cm bielle 6 cm \"acier\" alésage 4 cm piston 1 cm cylindre chambre 1 cm angle 180 rotation 600 tr/min
mesurer toutes les 2 ms : p en (0.075, 0.145) comme \"chambre\", masse \"air\" dans rect (0.055, 0.1) (0.095, 0.15) comme \"air haut\", masse \"air\" dans rect (0.055, 0.14) (0.095, 0.15) comme \"air PMH\"
simuler 0.05 s
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let (cp, cm, ct) = (column(&r, "chambre (Pa)"), column(&r, "air haut (kg/m)"), column(&r, "air PMH (kg/m)"));
    let (first, last) = (&r.rows[0], r.rows.last().unwrap());
    let ratio = last[cp] / first[cp];
    let lost = 1.0 - last[ct] / first[cm];
    let t_gas = r.sim.temp[r.sim.idx(15, 28)];
    eprintln!("compression : p {:.0} → {:.0} Pa (× {ratio:.2} ; adiabatique × 9.52), air {:.3e} → {:.3e} kg/m ({:.2} % perdu), T = {t_gas:.0} K", first[cp], last[cp], first[cm], last[ct], 100.0 * lost);
    assert!(ratio > 7.5 && ratio < 9.7, "rapport de pression {ratio}");
    assert!(lost.abs() < 1e-2, "fuite : {:.2} % de l'air", 100.0 * lost);
}

#[test]
fn compressed_air_drives_a_free_crank_and_its_work_becomes_kinetic_energy() {
    // moteur à air comprimé : vilebrequin libre à l'arrêt 20° après le point mort haut ; un compresseur dans la culasse pousse l'air
    // du dehors dans la chambre, la pression monte et pousse le piston, le vilebrequin accélère. L'énergie cinétique gagnée vaut le
    // travail de la force du fluide sur le piston ∫ F dy, et celui de la pression de la chambre ∫ (p − p_atm) A |dy| (relevée en un
    // point)
    let text = "départ scénario \"cuve vide\" grille 30 x 40 maille 5 mm
gravité (0, 0)
pas max 0.2 ms
vilebrequin (0.075, 0.05) rayon 2 cm bielle 6 cm \"acier\" alésage 4 cm piston 1 cm cylindre chambre 1 cm angle 20 libre inertie 0.002
effacer rect (0.07, 0.15) (0.08, 0.155)
pompe rect (0.07, 0.15) (0.08, 0.155) (0, -4) m/s
mesurer toutes les 0.2 ms : p en (0.075, 0.145) comme \"chambre\", corps 3 y comme \"piston\", corps 3 fy comme \"force\", vilebrequin 1 angle
simuler 15 ms
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let (cp, cy, cf, ca) = (column(&r, "chambre (Pa)"), column(&r, "piston (m)"), column(&r, "force (N/m)"), column(&r, "vilebrequin 1 angle (°)"));
    let (mut wf, mut wp) = (0.0, 0.0);
    // (pas de force au premier relevé : aucune projection encore)
    for w in r.rows.windows(2).filter(|w| w[0][cf].is_finite()) {
        let dy = w[1][cy] - w[0][cy];
        wf += 0.5 * (w[0][cf] + w[1][cf]) * dy;
        wp += 0.5 * (w[0][cp] + w[1][cp] - 2.0 * 101325.0) * 0.04 * (-dy);
    }
    let ke = crank_energy(&r.sim, 0);
    let last = r.rows.last().unwrap();
    eprintln!("moteur : angle 20 → {:.1}°, énergie cinétique {ke:.3} J/m ; travail de la force {wf:.3}, de la pression {wp:.3} J/m", last[ca]);
    assert!(last[ca] > 40.0 && ke > 0.1, "le vilebrequin n'a pas tourné : {last:?}");
    assert!((wf / ke - 1.0).abs() < 0.03, "travail de la force {wf} pour {ke}");
    assert!((wp / ke - 1.0).abs() < 0.1, "travail de la pression {wp} pour {ke}");
}

#[test]
fn filling_a_cell_under_a_body_leaves_the_body_its_place() {
    // disque fixe posé, puis « remplir partout avec eau » : chaque maille reçoit le fluide de la place que le corps lui laisse
    // (Σ m/ρ = 1 − α_corps) ; avant, une maille à demi couverte était remplie en entier (pic de pression au pas suivant)
    let text = "départ scénario \"cuve vide\" grille 20 x 20 maille 1 cm
corps disque (0.1, 0.1) rayon 3 cm \"acier\"
remplir partout avec \"eau\" à 20 °C
simuler 1 ms
";
    let (s, _, _) = initial_state(&proto(text), Path::new(".")).unwrap();
    let mut worst = 0.0f64;
    let mut partial = 0;
    for c in 0..s.nx * s.ny {
        let a = s.body_frac[c];
        if a > 0.0 && a < 1.0 {
            partial += 1;
        }
        let vol: f64 = (0..s.reg.phases.len()).map(|k| s.mass[k][c] / s.reg.phases[k].rho(s.p[c], s.temp[c])).sum();
        worst = worst.max((vol - (1.0 - a.min(1.0 - super::mp_bodies::FLUID_TRACE))).abs());
    }
    assert!(partial > 10 && worst < 1e-9, "{partial} mailles à demi couvertes, écart au volume laissé {worst}");
}

#[test]
fn a_spark_in_the_compressed_charge_drives_the_crank() {
    // temps moteur : mélange hydrogène–air dans le cylindre, vilebrequin libre lancé au point mort bas, compression (≈ 9 atm),
    // étincelle à 355° (un noyau de 4 mm : `allumer` remplace sa charge par des gaz brûlés, un gros noyau la gaspille) ; la
    // combustion fait monter la pression bien au-dessus de la compression, et la détente relance le vilebrequin
    let text = "départ scénario \"Explosion confinée (hydrogène–air)\" grille 30 x 40 maille 5 mm
remplir partout avec \"air sec (O2 + N2)\" à 20 °C
vilebrequin (0.075, 0.05) rayon 2 cm bielle 6 cm \"acier\" alésage 4 cm piston 1 cm cylindre chambre 1 cm angle 180 libre rotation 600 tr/min inertie 0.5
remplir rect (0.055, 0.1) (0.095, 0.15) avec \"hydrogène–air stœchiométrique\" à 20 °C
quand vilebrequin 1 angle > 355 : allumer disque (0.075, 0.145) rayon 4 mm
mesurer toutes les 1 ms : p en (0.075, 0.145) comme \"chambre\", vilebrequin 1 rotation, vilebrequin 1 angle
attendre que vilebrequin 1 angle > 60 et vilebrequin 1 angle < 180 au plus 0.15 s
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let (cp, cw, ca) = (column(&r, "chambre (Pa)"), column(&r, "vilebrequin 1 rotation (°/s)"), column(&r, "vilebrequin 1 angle (°)"));
    let spark = r.rows.iter().position(|row| row[ca] > 355.0 && row[ca] < 359.9).unwrap_or_else(|| panic!("pas d'allumage\n{}", r.log.join("\n")));
    let (p_spark, w_spark) = (r.rows[spark][cp], r.rows[spark][cw]);
    let p_max = r.rows[spark..].iter().map(|row| row[cp]).fold(0.0, f64::max);
    let w_end = r.rows.last().unwrap()[cw];
    eprintln!("temps moteur : p {p_spark:.0} Pa à l'étincelle, {p_max:.0} Pa au plus ; rotation {w_spark:.0} → {w_end:.0} °/s");
    assert!(p_max > 2.5 * p_spark, "la charge n'a pas brûlé : p {p_spark} → {p_max}");
    assert!(w_end > 1.2 * w_spark, "la détente n'a pas relancé le vilebrequin : {w_spark} → {w_end}");
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
fn the_forces_on_a_body_split_into_fluid_contacts_law_and_weight() {
    use super::mp_experiment::{BodyQty, ForcePart};
    let col = |r: &Runner, name: &str| -> usize {
        r.columns.iter().position(|c| c == name).unwrap_or_else(|| panic!("colonne « {name} » absente : {:?}", r.columns))
    };
    // moyenne des n derniers relevés d'une colonne
    let tail = |r: &Runner, c: usize, n: usize| -> f64 { r.rows[r.rows.len() - n..].iter().map(|row| row[c]).sum::<f64>() / n as f64 };
    let water = |s: &MultiphaseSim| -> f64 {
        let k = s.reg.phases.iter().position(|p| p.name == "eau").unwrap();
        s.reg.phases[k].rho(101325.0, 293.15)
    };
    let mass = |s: &MultiphaseSim, k: usize| -> f64 {
        let b = &s.bodies[k];
        b.mass(b.density(s.reg.solids[b.material].rho))
    };

    // 1. disque d'acier FIXE dans l'eau au repos : le fluide le pousse vers le haut de ρ g π r² (Archimède), sans force de côté ni
    //    couple ; ni contact ni loi ; avec le poids, la force totale est l'opposé de ce que son support retient
    let text = "départ scénario \"cuve vide\" grille 30 x 30 maille 1 cm
remplir rect (0, 0) (0.3, 0.27) avec \"eau\" à 20 °C
corps disque (0.15, 0.12) rayon 3 cm \"acier\"
mesurer toutes les 10 ms : corps 1 fx, corps 1 fy, corps 1 couple, corps 1 fy contact, corps 1 fy loi, corps 1 fy total comme \"bilan\"
simuler 50 ms
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let (fx, fy, tq) = (col(&r, "corps 1 fx (N/m)"), col(&r, "corps 1 fy (N/m)"), col(&r, "corps 1 couple (N)"));
    let (fc, fl, ft) = (col(&r, "corps 1 fy contact (N/m)"), col(&r, "corps 1 fy loi (N/m)"), col(&r, "bilan (N/m)"));
    let row = r.rows.last().unwrap().clone();
    let g = -r.sim.gravity[1];
    let archi = water(&r.sim) * g * std::f64::consts::PI * 9e-4;
    let w = mass(&r.sim, 0) * g;
    eprintln!(
        "disque fixe : fluide ({:.4}, {:.4}) N/m, couple {:.2e} N, Archimède {archi:.4} ; total {:.4} = fluide − poids ({w:.4})",
        row[fx], row[fy], row[tq], row[ft]
    );
    assert!((row[fy] / archi - 1.0).abs() < 1e-3, "poussée {} pour {archi}", row[fy]);
    assert!(row[fx].abs() < 1e-3 * archi && row[tq].abs() < 1e-3 * archi * 0.03, "force de côté {} ou couple {}", row[fx], row[tq]);
    assert!(row[fc] == 0.0 && row[fl] == 0.0, "contact {} et loi {} sur un corps fixe", row[fc], row[fl]);
    assert!((row[ft] - (row[fy] - w)).abs() < 1e-9 * w, "total {} pour {}", row[ft], row[fy] - w);

    // 2. disque d'acier LIBRE posé au fond de l'eau : le fond porte son poids moins la poussée d'Archimède, et la somme des forces
    //    (M a) s'annule au repos
    let text = "départ scénario \"cuve vide\" grille 40 x 30 maille 5 mm
remplir rect (0, 0) (0.2, 0.12) avec \"eau\" à 20 °C
corps disque (0.1, 0.02) rayon 2 cm \"acier\" libre
mesurer toutes les 10 ms : corps 1 fy, corps 1 fy contact, corps 1 fy total, corps 1 vy
simuler 0.3 s
";
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(false)).unwrap();
    run_to_end(&mut r);
    let (fy, fc, ft) = (col(&r, "corps 1 fy (N/m)"), col(&r, "corps 1 fy contact (N/m)"), col(&r, "corps 1 fy total (N/m)"));
    let g = -r.sim.gravity[1];
    let archi = water(&r.sim) * g * std::f64::consts::PI * 4e-4;
    let w = mass(&r.sim, 0) * g;
    let (f, c, t) = (tail(&r, fy, 10), tail(&r, fc, 10), tail(&r, ft, 10));
    eprintln!("disque posé : fluide {f:.4} (Archimède {archi:.4}), contact {c:.4} (poids − Archimède {:.4}), total {t:.2e} N/m", w - archi);
    assert!((f / archi - 1.0).abs() < 1e-2, "poussée {f} pour {archi}");
    assert!((c / (w - archi) - 1.0).abs() < 1e-2, "le fond porte {c} pour {}", w - archi);
    assert!(t.abs() < 1e-2 * w, "somme des forces {t} sur un corps au repos");

    // 3. bilan exact à chaque pas, M (Vⁿ⁺¹ − Vⁿ) / dt = fluide + contacts + loi + poids, en translation comme en rotation : disque
    //    lancé de biais dans l'eau, tiré par un ressort et un couple en formules, qui frappe le fond
    let p = proto(
        "départ scénario \"cuve vide\" grille 40 x 30 maille 5 mm
remplir rect (0, 0) (0.2, 0.12) avec \"eau\" à 20 °C
corps disque (0.1, 0.035) rayon 2 cm \"acier\" libre vitesse (0.1, -0.3) force formule \"-20*(x - x0)\" \"3\" couple formule \"0.002\"
",
    );
    let (mut s, _, _) = initial_state(&p, Path::new(".")).unwrap();
    let m = mass(&s, 0);
    let ib = s.bodies[0].density(s.reg.solids[s.bodies[0].material].rho) * s.bodies[0].shape.polar_moment();
    let scale = [m * 9.81, m * 9.81, m * 9.81 * 0.02];
    let (mut touched, mut worst, mut steps) = (false, 0.0f64, 0);
    while s.time < 0.15 {
        let (v0, w0) = (s.bodies[0].vel, s.bodies[0].omega);
        let dt = s.step();
        assert!(dt > 0.0, "pas refusé à t = {}", s.time);
        steps += 1;
        let b = &s.bodies[0];
        let total = [BodyQty::Fx(ForcePart::Total), BodyQty::Fy(ForcePart::Total), BodyQty::Torque(ForcePart::Total)].map(|q| q.value(&s, 0));
        let ma = [m * (b.vel[0] - v0[0]) / dt, m * (b.vel[1] - v0[1]) / dt, ib * (b.omega - w0) / dt];
        for r in 0..3 {
            worst = worst.max((total[r] - ma[r]).abs() / scale[r]);
        }
        let law = [BodyQty::Fy(ForcePart::Law), BodyQty::Torque(ForcePart::Law)].map(|q| q.value(&s, 0));
        assert!(law == [3.0, 0.002], "loi {law:?}");
        touched |= BodyQty::Fy(ForcePart::Contact).value(&s, 0) > 0.0;
    }
    eprintln!("bilan sur {steps} pas : écart relatif au plus {worst:.2e} ; contact avec le fond : {touched}");
    assert!(touched, "le disque n'a pas touché le fond");
    assert!(worst < 1e-9, "fluide + contacts + loi + poids s'écartent de M a de {worst} (relatif)");
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


// ---------------------------------------------------------------------------------------------------------------- composants

/// protocole lu depuis le dossier `dir` (ses composants y sont cherchés)
fn proto_in(text: &str, dir: &Path) -> super::mp_experiment::Protocol {
    match parse_all_in(text, dir) {
        Ok(mut v) => v.remove(0),
        Err(e) => panic!("{}", e.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n")),
    }
}

fn errors_in(text: &str, dir: &Path) -> Vec<super::mp_script::ScriptError> {
    parse_all_in(text, dir).err().unwrap_or_else(|| panic!("le protocole devait être refusé :\n{text}"))
}

fn steps_only(p: &super::mp_experiment::Protocol) -> Vec<Step> {
    p.steps.iter().map(|(s, _)| s.clone()).collect()
}

#[test]
fn a_component_is_included_with_its_parameters_and_its_lines_say_where_they_come_from() {
    let dir = tmp_dir("composants");
    std::fs::create_dir_all(dir.join("comp")).unwrap();
    // un bac d'eau au fond, posé deux fois : ses zones et sa colonne nommées d'après NOM
    std::fs::write(
        dir.join("comp/bac.exp"),
        "# bac d'eau au fond
paramètre NOM = \"bac\"
paramètre X = 0.02          # bord gauche
paramètre L = 4 cm
paramètre FLUIDE = \"eau\"
paramètre FIN = $(X + L)
zone \"$NOM.fond\" = rect ($X, 0) ($FIN, 0.02)
remplir zone \"$NOM.fond\" avec $FLUIDE à 20 °C
mesurer : masse $FLUIDE dans zone \"$NOM.fond\" comme \"$NOM\"
quand temps > 1 ms : marque \"$NOM plein\"
",
    )
    .unwrap();
    let text = "départ scénario \"cuve vide\" grille 12 x 12 maille 1 cm
inclure \"comp/bac.exp\" avec NOM = \"a\"
inclure \"comp/bac.exp\" avec NOM = \"b\", X = 6 cm, L = $(0.01 * 3)
mesurer toutes les 1 ms : masse \"eau\" dans zone \"a.fond\" comme \"a encore\"
simuler 2 ms
";
    let p = proto_in(text, &dir);
    let fills: Vec<(&Zone, usize)> =
        p.steps.iter().filter_map(|(s, l)| if let Step::Do(Action::Fill { zone, .. }) = s { Some((zone, *l)) } else { None }).collect();
    assert_eq!(fills.len(), 2);
    assert_eq!(*fills[0].0, Zone::Rect { a: [0.02, 0.0], b: [0.06, 0.02] });
    assert_eq!(*fills[1].0, Zone::Rect { a: [0.06, 0.0], b: [0.09, 0.02] }, "FIN = X + L, L calculé dans le protocole");
    // lignes d'un composant : numérotées après le texte (5 lignes), situées dans leur fichier et à la ligne de leur « inclure »
    assert!(fills[0].1 > 5 && fills[1].1 > fills[0].1);
    assert_eq!(p.includes.place(fills[0].1), "ligne 2 (« comp/bac.exp » ligne 8)");
    assert_eq!(p.includes.place(fills[1].1), "ligne 3 (« comp/bac.exp » ligne 8)");
    assert_eq!(p.includes.place(4), "ligne 4");
    // la copie du protocole se relit seule, aux mêmes étapes
    assert!(p.source.contains("# ─── composant « comp/bac.exp » avec NOM = \"b\", X = 6 cm, L = 0.03"), "{}", p.source);
    assert!(p.source.contains("# paramètre FIN = 0.09"), "{}", p.source);
    assert!(!p.source.contains('$') && !p.source.lines().any(|l| l.starts_with("inclure")), "{}", p.source);
    assert_eq!(steps_only(&proto(&p.source)), steps_only(&p));
    // déroulé : les deux bacs pesés, le journal situe les règles des composants
    let mut r = Runner::new(p, &dir, None, Some(false)).unwrap();
    run_to_end(&mut r);
    let first = &r.rows[0];
    let (a, b, a2) = (first[column(&r, "a (kg/m)")], first[column(&r, "b (kg/m)")], first[column(&r, "a encore (kg/m)")]);
    assert!(a > 0.7 && a < 0.9, "{a}");
    assert_eq!(a.to_bits(), a2.to_bits());
    assert!((b / a - 0.75).abs() < 1e-9, "{b} {a}");
    assert!(r.log.iter().any(|l| l.contains("ligne 3 (« comp/bac.exp » ligne 10) : quand temps")), "{}", r.log.join("\n"));
    assert!(r.log.iter().any(|l| l.contains("b plein")), "{}", r.log.join("\n"));

    // erreurs : à la ligne de l'« inclure », situées dans le composant
    let start = "départ scénario \"cuve vide\" grille 12 x 12 maille 1 cm\n";
    let e = errors_in(&format!("{start}inclure \"comp/bac.exp\" avec NOMM = \"x\"\n"), &dir);
    assert_eq!(e[0].line, 2);
    assert!(e[0].msg.contains("n'a pas de paramètre « NOMM » (paramètres : NOM, X, L, FLUIDE, FIN)"), "{}", e[0].msg);
    let e = errors_in(&format!("{start}inclure \"comp/absent.exp\"\n"), &dir);
    assert!(e[0].line == 2 && e[0].msg.contains("composant « comp/absent.exp »"), "{e:?}");
    std::fs::write(dir.join("comp/faux.exp"), "paramètre N = 1\nremplir rect (0, 0) (0.1, 0.1) avec \"eau\" à $N K\nvoler\n").unwrap();
    let e = errors_in(&format!("{start}\ninclure \"comp/faux.exp\"\n"), &dir);
    assert_eq!(e.len(), 1, "{e:?}");
    assert!(e[0].line == 3 && e[0].msg.starts_with("« comp/faux.exp » ligne 3 : commande « voler » inconnue"), "{e:?}");
    // inclusions imbriquées : chemin relatif au composant, chaîne des inclusions dans le message
    std::fs::write(dir.join("comp/double.exp"), "inclure \"bac.exp\" avec NOM = \"d\"\ninclure \"faux.exp\"\n").unwrap();
    let e = errors_in(&format!("{start}inclure \"comp/double.exp\"\n"), &dir);
    assert!(e[0].line == 2 && e[0].msg.starts_with("« comp/double.exp » ligne 2 → « faux.exp » ligne 3 : commande « voler »"), "{e:?}");
    // valeur inconnue, paramètre sans valeur, inclusion circulaire, balayage dans un composant
    let e = errors_in(&format!("{start}remplir rect (0, 0) ($Y, 0.01) avec \"eau\"\n"), &dir);
    assert!(e[0].line == 2 && e[0].msg.contains("valeur « $Y » inconnue"), "{e:?}");
    std::fs::write(dir.join("comp/requis.exp"), "paramètre Z\nremplir rect (0, 0) ($Z, 0.01) avec \"eau\"\n").unwrap();
    let e = errors_in(&format!("{start}inclure \"comp/requis.exp\"\n"), &dir);
    assert!(e[0].msg.contains("paramètre « Z » sans valeur : il faut « avec Z = … »"), "{e:?}");
    assert!(proto_in(&format!("{start}inclure \"comp/requis.exp\" avec Z = 3 cm\n"), &dir).steps.len() == 1);
    std::fs::write(dir.join("comp/boucle.exp"), "inclure \"boucle.exp\"\n").unwrap();
    let e = errors_in(&format!("{start}inclure \"comp/boucle.exp\"\n"), &dir);
    assert!(e[0].msg.contains("inclusion circulaire"), "{e:?}");
    std::fs::write(dir.join("comp/balaye.exp"), "balayage T = 1, 2\n").unwrap();
    let e = errors_in(&format!("{start}inclure \"comp/balaye.exp\"\n"), &dir);
    assert!(e[0].msg.contains("« balayage » : dans le protocole seulement"), "{e:?}");
    // un bloc s'ouvre et se ferme dans le même fichier
    std::fs::write(dir.join("comp/ouvre.exp"), "si temps > 0 s\nmarque \"x\"\n").unwrap();
    let e = errors_in(&format!("{start}inclure \"comp/ouvre.exp\"\nfin\n"), &dir);
    assert!(e.iter().any(|e| e.line == 3 && e.msg.contains("ferme un bloc ouvert dans un autre fichier")), "{e:?}");
    // des colonnes sans période
    let e = errors_in(&format!("{start}mesurer : temps\nsimuler 1 ms\n"), &dir);
    assert!(e[0].line == 2 && e[0].msg.contains("sans période"), "{e:?}");
    // un paramètre du protocole lui-même, et le balayage qui le remplace
    let ps = parse_all_in(&format!("{start}paramètre T0 = 300 K\nbalayage T0 = 310 K, 320 K\nremplir partout avec \"eau\" à $(T0 + 5)\n"), &dir).unwrap();
    let temps: Vec<Option<f64>> = ps.iter().map(|p| match &p.steps[0].0 { Step::Do(Action::Fill { temp, .. }) => *temp, _ => None }).collect();
    assert_eq!(temps, vec![Some(315.0), Some(325.0)]);
    let p = proto(&format!("{start}paramètre T0 = 300 K\nremplir partout avec \"eau\" à $(T0 + 5)\n"));
    assert!(matches!(p.steps[0].0, Step::Do(Action::Fill { temp: Some(t), .. }) if t == 305.0));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_named_zone_stands_for_its_zone_or_its_point() {
    let text = "départ scénario \"cuve vide\" grille 12 x 12 maille 1 cm
zone \"bas\" = rect (0, 0) (0.12, 3 cm)
zone \"capteur\" = point (0.06, 0.01)
zone \"bas bis\" = zone \"Bas\"
remplir zone \"bas\" avec \"eau\" à 20 °C
chauffer disque zone \"capteur\" rayon 1 cm à 30 °C pendant 1 ms
mesurer toutes les 1 ms : T en zone \"capteur\" comme \"T\", moyenne T dans zone \"bas bis\"
simuler 1 ms
";
    let p = proto(text);
    assert!(matches!(&p.steps[0].0, Step::Do(Action::Fill { zone: Zone::Rect { a, b }, .. }) if *a == [0.0, 0.0] && *b == [0.12, 0.03]));
    assert!(matches!(&p.steps[1].0, Step::Do(Action::Heat { zone: Zone::Disque { c, r }, .. }) if *c == [0.06, 0.01] && *r == 0.01));
    assert!(matches!(&p.measures[0].0, Measure::Probe { zone: Zone::Point(c), reduce: Reduce::At, .. } if *c == [0.06, 0.01]));
    assert!(matches!(&p.measures[1].0, Measure::Probe { zone: Zone::Rect { b, .. }, reduce: Reduce::Mean, .. } if *b == [0.12, 0.03]));
    let start = "départ scénario \"cuve vide\" grille 12 x 12 maille 1 cm\n";
    let bad = |rest: &str| parse(&format!("{start}zone \"bas\" = rect (0, 0) (0.12, 0.03)\n{rest}"), None).unwrap_err();
    let e = bad("remplir zone \"haut\" avec \"eau\"\n");
    assert!(e[0].line == 3 && e[0].msg.contains("zone « haut » inconnue") && e[0].msg.contains("(zones nommées : bas)"), "{e:?}");
    let e = bad("zone \"BAS\" = point (0, 0)\n");
    assert!(e[0].msg.contains("zone « BAS » déjà nommée (ligne 2)"), "{e:?}");
    let e = bad("zone \"t\" = rect ({t}, 0) (1, 1)\n");
    assert!(e[0].msg.contains("fixée à la lecture"), "{e:?}");
    let e = bad("zone \"d\" = disque (0, 0)\n");
    assert!(e[0].line == 3 && e[0].msg.contains("rayon"), "{e:?}");
}

#[test]
fn named_bodies_and_cranks_are_found_by_their_name() {
    let text = "départ scénario \"cuve vide\" grille 30 x 30 maille 1 cm
corps disque (0.05, 0.2) rayon 1 cm \"acier\" libre nom \"bille\"
corps disque (0.15, 0.2) rayon 1 cm \"acier\" nom \"Borne\"
vilebrequin (0.1, 0.05) rayon 1 cm bielle 4 cm rotation 600 tr/min nom \"a\"
vilebrequin (0.22, 0.05) rayon 1 cm bielle 4 cm rotation 300 tr/min angle 90 nom \"b\"
mesurer toutes les 2 ms : corps \"bille\" y, corps \"borne\" y, corps 1 y, vilebrequin \"b\" angle, vilebrequin 2 angle, corps \"b.piston\" y, corps 8 y
simuler 10 ms
";
    let r = run(text, None);
    assert_eq!(r.sim.bodies.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(), [
        "bille", "Borne", "a.manivelle", "a.bielle", "a.piston", "b.manivelle", "b.bielle", "b.piston"
    ]);
    assert_eq!(r.sim.mechanisms.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["a", "b"]);
    let last = r.rows.last().unwrap();
    let col = |n: &str| last[column(&r, n)];
    assert!(col("corps bille y (m)") < 0.2, "la bille tombe");
    assert_eq!(col("corps borne y (m)"), 0.2);
    assert_eq!(col("corps bille y (m)").to_bits(), col("corps 1 y (m)").to_bits());
    assert_eq!(col("vilebrequin b angle (°)").to_bits(), col("vilebrequin 2 angle (°)").to_bits());
    assert!((col("vilebrequin b angle (°)") - 108.0).abs() < 1e-6, "90° + 1800 °/s × 10 ms");
    assert_eq!(col("corps b.piston y (m)").to_bits(), col("corps 8 y (m)").to_bits());
    // un nom que rien ne pose : refusé à la lecture, avec les noms posés
    let e = parse(&text.replace("corps \"bille\" y,", "corps \"bile\" y,"), None).unwrap_err();
    assert!(e[0].line == 6 && e[0].msg.contains("aucun corps ne s'appelle « bile »") && e[0].msg.contains("noms posés : bille, Borne, a.manivelle, a.bielle"), "{e:?}");
    let e = parse(&text.replace("vilebrequin \"b\" angle", "vilebrequin \"c\" angle"), None).unwrap_err();
    assert!(e[0].msg.contains("aucun vilebrequin ne s'appelle « c »"), "{e:?}");
    // deux objets du même nom : refusés à la pose
    let twice = "départ scénario \"cuve vide\" grille 30 x 30 maille 1 cm
corps disque (0.05, 0.2) rayon 1 cm \"acier\" nom \"x\"
corps disque (0.15, 0.2) rayon 1 cm \"acier\" nom \"X\"
";
    let e = initial_state(&proto(twice), Path::new(".")).err().unwrap();
    assert!(e.starts_with("ligne 3 : un corps s'appelle déjà « X »"), "{e}");
    let twice = "départ scénario \"cuve vide\" grille 30 x 30 maille 1 cm
vilebrequin (0.1, 0.05) rayon 1 cm bielle 4 cm nom \"m\"
vilebrequin (0.22, 0.05) rayon 1 cm bielle 4 cm nom \"m\"
";
    let e = initial_state(&proto(twice), Path::new(".")).err().unwrap();
    assert!(e.contains("un vilebrequin s'appelle déjà « m »"), "{e}");
}

/// texte d'un exemple d'avant ses composants, et l'exemple d'aujourd'hui : mêmes relevés, au bit près, sur un début d'expérience
fn same_rows(old: &str, file: &str, short: &[(&str, &str)]) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiences");
    let new = std::fs::read_to_string(dir.join(file)).unwrap();
    let (mut old, mut new) = (old.to_string(), new);
    for (a, b) in short {
        assert!(old.contains(a) && new.contains(a), "{file} : « {a} »");
        old = old.replace(a, b);
        new = new.replace(a, b);
    }
    let mut ro = Runner::new(proto(&old), &dir, None, Some(false)).unwrap();
    run_to_end(&mut ro);
    let mut rn = Runner::new(proto_in(&new, &dir), &dir, None, Some(false)).unwrap();
    run_to_end(&mut rn);
    assert_eq!(ro.columns.len(), rn.columns.len(), "{file} : {:?} / {:?}", ro.columns, rn.columns);
    assert!(ro.rows.len() >= 3, "{file} : {} relevés", ro.rows.len());
    assert_eq!(ro.rows.len(), rn.rows.len(), "{file}");
    for (a, b) in ro.rows.iter().zip(&rn.rows) {
        let bits = |r: &Vec<f64>| r.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(a), bits(b), "{file} : {a:?} / {b:?}");
    }
}

#[test]
fn the_examples_built_from_components_give_the_same_experiments_as_before() {
    same_rows(
        "départ scénario \"Explosion confinée (hydrogène–air)\" grille 30 x 40 maille 5 mm
remplir partout avec \"air sec (O2 + N2)\" à 20 °C
vilebrequin (0.075, 0.05) rayon 2 cm bielle 6 cm \"acier\" alésage 4 cm piston 1 cm cylindre chambre 1 cm angle 180 libre rotation 600 tr/min inertie 0.5 couple \"-0.5*max(rotation - 3600, 0)\"
effacer rect (0.07, 0.15) (0.08, 0.155)
vanne rect (0.07, 0.15) (0.08, 0.155) fermée
remplir rect (0.055, 0.1) (0.095, 0.15) avec \"hydrogène–air stœchiométrique\" à 20 °C
quand vilebrequin 1 angle > 140 et vilebrequin 1 angle < 180 : vanne rect (0.07, 0.15) (0.08, 0.155) ouverte
quand vilebrequin 1 angle > 180 et vilebrequin 1 angle < 270
    remplir rect (0.055, 0.1) (0.095, 0.15) avec \"hydrogène–air stœchiométrique\" à 20 °C
    vanne rect (0.07, 0.15) (0.08, 0.155) fermée
fin
quand vilebrequin 1 angle > 355 : allumer disque (0.075, 0.145) rayon 4 mm
mesurer toutes les 2 ms : vilebrequin 1 rotation, vilebrequin 1 tours, vilebrequin 1 puissance, p en (0.075, 0.145) comme \"chambre\", max T dans rect (0.055, 0.1) (0.095, 0.15) comme \"T chambre\"
simuler 1 s
",
        "moteur_explosion.exp",
        &[("simuler 1 s", "simuler 6 ms")],
    );
    same_rows(
        "départ scénario \"Cuve vide\" grille 30 x 40 maille 5 mm
vilebrequin (0.075, 0.05) rayon 2 cm bielle 6 cm \"acier\" alésage 4 cm piston 1 cm cylindre chambre 1 cm angle 10 libre inertie 0.005 couple \"-0.0002*rotation\"
effacer rect (0.07, 0.15) (0.08, 0.155)
vanne rect (0.07, 0.15) (0.08, 0.155) fermée
quand vilebrequin 1 angle > 5 et vilebrequin 1 angle < 150
    vanne rect (0.07, 0.15) (0.08, 0.155) ouverte
    pompe rect (0.07, 0.15) (0.08, 0.155) (0, -8) m/s
fin
quand vilebrequin 1 angle > 150 et vilebrequin 1 angle < 355
    pompe rect (0.07, 0.15) (0.08, 0.155) (0, 0) m/s
fin
quand vilebrequin 1 angle > 355
    vanne rect (0.07, 0.15) (0.08, 0.155) fermée
fin
mesurer toutes les 5 ms : vilebrequin 1 rotation, vilebrequin 1 tours, vilebrequin 1 puissance, p en (0.075, 0.145) comme \"chambre\"
simuler 1 s
",
        "moteur.exp",
        &[("simuler 1 s", "simuler 15 ms")],
    );
    same_rows(
        "départ scénario \"Cuve vide\" grille 40 x 40 maille 1 cm
solide rect (0.11, 0.21) (0.29, 0.37) \"acier\"
effacer rect (0.12, 0.22) (0.28, 0.37)
effacer rect (0.19, 0.21) (0.21, 0.22)
vanne rect (0.19, 0.21) (0.21, 0.22) fermée
remplir rect (0.12, 0.22) (0.28, 0.34) avec \"eau\" à 20 °C
mesurer toutes les 20 ms : masse \"eau\" dans rect (0.12, 0.22) (0.28, 0.37) comme \"réservoir\", masse \"eau\" dans rect (0, 0) (0.4, 0.21) comme \"bassin\"
tant que masse \"eau\" dans rect (0.12, 0.22) (0.28, 0.37) > 4 au plus 4 s
    vanne rect (0.19, 0.21) (0.21, 0.22) ouverte
    attendre 0.2 s
    vanne rect (0.19, 0.21) (0.21, 0.22) fermée
    attendre 0.3 s
fin
marque \"réservoir presque vide\"
attendre 0.5 s
",
        "bachees.exp",
        &[("au plus 4 s", "au plus 0.25 s"), ("attendre 0.5 s", "attendre 0.02 s")],
    );
    same_rows(
        "départ scénario \"Cuve vide\" grille 40 x 40 maille 1 cm
solide rect (0.11, 0.21) (0.29, 0.37) \"acier\"
effacer rect (0.12, 0.22) (0.28, 0.37)
effacer rect (0.18, 0.21) (0.22, 0.22)
vanne rect (0.18, 0.21) (0.22, 0.22) fermée
émetteur rect (0.13, 0.35) (0.15, 0.36) \"eau\" (0, -0.3) m/s à 20 °C
mesurer toutes les 20 ms : masse \"eau\" dans rect (0.12, 0.22) (0.28, 0.37) comme \"réservoir\", masse \"eau\" dans rect (0, 0) (0.4, 0.21) comme \"bassin\"
quand masse \"eau\" dans rect (0.12, 0.22) (0.28, 0.37) > 10
    marque \"chasse\"
    vanne rect (0.18, 0.21) (0.22, 0.22) ouverte
    attendre que masse \"eau\" dans rect (0.12, 0.22) (0.28, 0.37) < 3 au plus 2 s
    vanne rect (0.18, 0.21) (0.22, 0.22) fermée
fin
simuler 8 s
",
        "chasse.exp",
        &[("simuler 8 s", "simuler 0.1 s")],
    );
    same_rows(
        "départ scénario \"Cuve vide\" grille 30 x 24 maille 1 cm
bord haut ouvert
pas max 2 ms
solide rect (0.05, 0.02) (0.25, 0.2) \"acier\"
effacer rect (0.06, 0.03) (0.24, 0.19)
effacer rect (0.05, 0.1) (0.06, 0.12)
pompe rect (0.05, 0.1) (0.06, 0.12) (0.5, 0) m/s
effacer rect (0.14, 0.19) (0.16, 0.2)
vanne rect (0.14, 0.19) (0.16, 0.2) fermée
quand p en (0.15, 0.1) > 104 kPa : vanne rect (0.14, 0.19) (0.16, 0.2) ouverte
quand p en (0.15, 0.1) < 102 kPa : vanne rect (0.14, 0.19) (0.16, 0.2) fermée
mesurer toutes les 2 ms : p en (0.15, 0.1) comme \"pression dans la cuve\"
simuler 1 s
",
        "soupape.exp",
        &[("simuler 1 s", "simuler 0.12 s")],
    );
    same_rows(
        "départ scénario \"Cuve vide\" grille 30 x 30 maille 1 cm
soit k = 0
mesurer toutes les 20 ms : T en (0.15, 0.08) comme \"air au-dessus\", k comme \"palier\"
répéter 5 fois
    soit k = {k + 1}
    chauffer rect (0.1, 0.01) (0.2, 0.03) à {300 + 20*k} K pendant 0.5 s
    attendre 0.5 s
fin
marque \"dernière consigne : {300 + 20*k} K\"
",
        "rampe.exp",
        &[("0.5 s", "0.04 s")],
    );
}

#[test]
fn one_engine_component_placed_twice_runs_two_engines_on_their_own_rules() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("experiences");
    let text = std::fs::read_to_string(dir.join("deux_moteurs.exp")).unwrap().replace("simuler 1 s", "simuler 4 ms");
    let p = proto_in(&text, &dir);
    // les règles de chaque moteur ne nomment que son vilebrequin ; « droite » s'allume plus tôt
    let rules: Vec<String> = p.steps.iter().filter_map(|(s, _)| if let Step::Do(a @ Action::When { .. }) = s { Some(format!("{a:?}")) } else { None }).collect();
    assert_eq!(rules.len(), 4, "{rules:?}");
    assert!(rules[..2].iter().all(|r| r.contains("Name(\"gauche\")") && !r.contains("droite")), "{rules:?}");
    assert!(rules[2..].iter().all(|r| r.contains("Name(\"droite\")") && !r.contains("gauche")), "{rules:?}");
    assert!(rules[3].contains("Num(330.0)") && rules[1].contains("Num(355.0)"), "{rules:?}");
    let mut r = Runner::new(p, &dir, None, Some(false)).unwrap();
    run_to_end(&mut r);
    assert_eq!(r.sim.mechanisms.len(), 2);
    assert_eq!((r.sim.mechanisms[0].name.as_str(), r.sim.mechanisms[1].name.as_str()), ("gauche", "droite"));
    let pivot = r.sim.mechanisms[1].crank.as_ref().unwrap().pivot;
    assert!((pivot[0] - 0.225).abs() < 1e-12, "{pivot:?}");
    let last = r.rows.last().unwrap();
    for side in ["gauche", "droite"] {
        let w = last[column(&r, &format!("{side} rotation (°/s)"))];
        assert!(w > 3000.0 && w < 4000.0, "{side} : {w} °/s");
        assert!(last[column(&r, &format!("{side} chambre (Pa)"))] > 1e5);
    }
}


// ------------------------------------------------------------------------------------------------- mécanismes : roues et pièces

/// aire commune de deux polygones convexes en sens trigonométrique (Sutherland–Hodgman)
fn overlap_area(a: &[[f64; 2]], b: &[[f64; 2]]) -> f64 {
    let cross = |p: [f64; 2], q: [f64; 2], x: [f64; 2]| (q[0] - p[0]) * (x[1] - p[1]) - (q[1] - p[1]) * (x[0] - p[0]);
    let mut out = a.to_vec();
    for k in 0..b.len() {
        let (p, q) = (b[k], b[(k + 1) % b.len()]);
        let input = std::mem::take(&mut out);
        for i in 0..input.len() {
            let (s, e) = (input[(i + input.len() - 1) % input.len()], input[i]);
            let (ds, de) = (cross(p, q, s), cross(p, q, e));
            let cut = || {
                let t = ds / (ds - de);
                [s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])]
            };
            match (ds >= 0.0, de >= 0.0) {
                (true, true) => out.push(e),
                (true, false) => out.push(cut()),
                (false, true) => {
                    out.push(cut());
                    out.push(e);
                }
                _ => {}
            }
        }
        if out.is_empty() {
            return 0.0;
        }
    }
    let n = out.len();
    (0..n).map(|i| out[i][0] * out[(i + 1) % n][1] - out[(i + 1) % n][0] * out[i][1]).sum::<f64>().abs() / 2.0
}

#[test]
fn gears_mesh_at_the_ratio_of_their_teeth_without_overlapping() {
    // pignon de 12 dents mené à 300 tr/min (1800 °/s), roue de 24 dents qui l'engrène : 900 °/s en sens inverse
    let text = "départ scénario \"cuve vide\" grille 130 x 90 maille 2 mm
mécanisme \"train\" rotation 300 tr/min
roue (0.045, 0.09) rayon 3.6 cm dents 12 \"acier\" sur \"train\" nom \"pignon\"
roue (0.153, 0.09) rayon 7.2 cm dents 24 engrène \"pignon\" nom \"roue\"
mesurer toutes les 2 ms : mécanisme \"train\" angle, corps \"pignon\" angle, corps \"roue\" angle, corps \"roue\" rotation
simuler 12 ms
";
    let r = run(text, None);
    let s = &r.sim;
    assert_eq!(s.bodies.len(), 38);
    assert_eq!((s.bodies[0].name.as_str(), s.bodies[12].name.as_str(), s.bodies[13].name.as_str(), s.bodies[37].name.as_str()), ("pignon", "pignon.dent12", "roue", "roue.dent24"));
    let (ca, cp, cr, cw) = (column(&r, "mécanisme train angle (°)"), column(&r, "corps pignon angle (°)"), column(&r, "corps roue angle (°)"), column(&r, "corps roue rotation (°/s)"));
    for w in r.rows.windows(2) {
        let (d_meca, d_roue) = (w[1][ca] - w[0][ca], w[1][cr] - w[0][cr]);
        assert!((w[1][cp] - w[0][cp] - d_meca).abs() < 1e-9 && (d_roue + 0.5 * d_meca).abs() < 1e-9, "{w:?}");
        assert!((w[1][cw] + 900.0).abs() < 1e-9, "{w:?}");
    }
    // sur une dent du pignon (30°) : aucune dent dans une dent d'en face, aucune dans le disque de l'autre
    let m = &s.mechanisms[0];
    let mut worst: f64 = 0.0;
    for step in 0..=60 {
        let th = m.theta + (step as f64 / 60.0) * 30f64.to_radians();
        let mut bodies = s.bodies.clone();
        for (p, (c, a)) in m.pieces.iter().zip(m.body_poses(th)) {
            bodies[p.body].pos = c;
            bodies[p.body].angle = a;
        }
        let v: Vec<Vec<[f64; 2]>> = bodies.iter().map(|b| b.world_vertices()).collect();
        for i in 0..13 {
            for j in 13..38 {
                worst = worst.max(overlap_area(&v[i], &v[j]));
            }
        }
    }
    assert!(worst < 1e-12, "dents qui se chevauchent : {worst} m²");
    assert!(r.sim.state_is_sane());
}

#[test]
fn a_free_wheel_spins_up_as_its_inertia_says_and_a_pendulum_piece_swings_with_its_period() {
    // couple constant de 0.5 N sur un volant et une roue d'acier : ω = Γ t / I
    let text = "départ scénario \"cuve vide\" grille 40 x 40 maille 5 mm
mécanisme \"m\" libre inertie 0.01 couple \"0.5\"
roue (0.1, 0.1) rayon 4 cm \"acier\" sur \"m\" nom \"volant\"
mesurer toutes les 10 ms : mécanisme \"m\" rotation
simuler 0.2 s
";
    let r = run(text, None);
    let b = &r.sim.bodies[0];
    let inertia = 0.01 + b.density(r.sim.reg.solids[b.material].rho) * b.shape.polar_moment();
    let w = r.rows.last().unwrap()[column(&r, "mécanisme m rotation (°/s)")];
    let expect = (0.5 * 0.2 / inertia).to_degrees();
    assert!((w / expect - 1.0).abs() < 5e-3, "{w} °/s pour {expect}");
    // pendule : une bille de 1 cm au bout d'un bras de 10 cm (pièce en formules, sans rotation propre), lâchée à 0.2 rad ; période
    // 2π √(L/g) (1 + θ0²/16) aux petits angles
    let text = "départ scénario \"cuve vide\" grille 40 x 40 maille 5 mm
mécanisme \"p\" libre
pièce disque (0.1, 0.05) rayon 1 cm \"acier\" sur \"p\" position \"0.1 + 0.1*sin(theta + 0.2)\" \"0.15 - 0.1*cos(theta + 0.2)\" nom \"bille\"
mesurer toutes les 2 ms : mécanisme \"p\" rotation, corps \"bille\" x
simuler 1.4 s
";
    let r = run(text, None);
    let cw = column(&r, "mécanisme p rotation (°/s)");
    // instants où la rotation change de signe (extrêmes du balancement)
    let mut zeros = vec![];
    for w in r.rows.windows(2) {
        let (a, b) = (w[0][cw], w[1][cw]);
        if a != 0.0 && (a > 0.0) != (b > 0.0) {
            zeros.push(w[0][0] + (w[1][0] - w[0][0]) * a / (a - b));
        }
    }
    assert!(zeros.len() >= 4, "{zeros:?}");
    let period = 2.0 * (zeros[zeros.len() - 1] - zeros[0]) / (zeros.len() - 1) as f64;
    let expect = 2.0 * std::f64::consts::PI * (0.1f64 / 9.81).sqrt() * (1.0 + 0.04 / 16.0);
    eprintln!("pendule : période {period:.5} s pour {expect:.5} s");
    assert!((period / expect - 1.0).abs() < 1e-2, "{period} s pour {expect} s");
    // la bille reste à 10 cm de l'axe
    let x = r.rows.iter().map(|row| row[column(&r, "corps bille x (m)")]).fold(f64::MIN, f64::max);
    assert!((x - (0.1 + 0.1 * 0.2f64.sin())).abs() < 2e-3, "{x}");
}

#[test]
fn pieces_ride_a_crank_shaft_and_mechanism_lines_report_their_errors() {
    // une poulie sur l'arbre d'un vilebrequin (rapport −0.5 : une courroie croisée) et une crémaillère en formule
    let text = "départ scénario \"cuve vide\" grille 40 x 30 maille 5 mm
vilebrequin (0.06, 0.05) rayon 1 cm bielle 4 cm rotation 300 tr/min nom \"v\"
roue (0.15, 0.05) rayon 2 cm sur \"v\" rapport -0.5 nom \"poulie\"
pièce rect (0.1, 0.12) (0.14, 0.13) \"acier\" sur \"v\" position \"x0 + 0.01*theta\" \"y0\" nom \"crémaillère\"
mesurer toutes les 5 ms : corps \"crémaillère\" x, corps \"poulie\" rotation, vilebrequin \"v\" tours, mécanisme 1 tours
simuler 20 ms
";
    let r = run(text, None);
    let last = r.rows.last().unwrap();
    let turns = last[column(&r, "vilebrequin v tours (tr)")];
    assert!((turns - 0.1).abs() < 1e-9, "{turns}");
    assert_eq!(turns.to_bits(), last[column(&r, "mécanisme 1 tours (tr)")].to_bits());
    assert!((last[column(&r, "corps crémaillère x (m)")] - (0.12 + 0.01 * turns * std::f64::consts::TAU)).abs() < 1e-12);
    assert!((last[column(&r, "corps poulie rotation (°/s)")] + 900.0).abs() < 1e-9);
    assert_eq!(r.sim.mechanisms[0].pieces.len(), 5);
    // lecture : erreurs
    let start = "départ scénario \"cuve vide\" grille 40 x 30 maille 5 mm\nmécanisme \"m\" rotation 60 tr/min\n";
    for (bad, what) in [
        ("roue (0.1, 0.1) rayon 2 cm", "il faut « sur"),
        ("roue (0.1, 0.1) rayon 2 cm sur \"m\" dents 12 pales 4 longueur 1 cm", "dents ou des pales"),
        ("roue (0.1, 0.1) rayon 2 cm sur \"m\" dents 4", "au moins 6 dents"),
        ("roue (0.1, 0.1) rayon 2 cm sur \"m\" nom \"a\"\nroue (0.14, 0.1) rayon 2 cm engrène \"a\" rapport 2", "« rapport » : pour une roue « sur »"),
        ("roue (0.1, 0.1) rayon 2 cm sur \"zz\"", "aucun mécanisme ne s'appelle « zz »"),
        ("roue (0.1, 0.1) rayon 2 cm engrène \"r\"", "aucune roue ne s'appelle « r »"),
        ("pièce disque (0.1, 0.1) rayon 1 cm \"acier\"", "il faut « sur"),
        ("pièce disque (0.1, 0.1) rayon 1 cm sur \"m\" position \"x0 + vitesse\" \"y0\"", "variables : theta"),
        ("mécanisme \"n\" inertie 1", "pour un mécanisme « libre »"),
        ("mesurer toutes les 1 ms : mécanisme \"q\" angle", "aucun mécanisme ne s'appelle « q »"),
    ] {
        let e = parse(&format!("{start}{bad}\n"), None).unwrap_err();
        assert!(e[0].msg.contains(what), "« {bad} » : {e:?}");
    }
    // pose : des roues qui ne se touchent pas, des dents de tailles différentes, des dents trop fines pour la maille
    for (bad, what) in [
        ("roue (0.1, 0.1) rayon 2 cm sur \"m\" nom \"a\"\nroue (0.16, 0.1) rayon 2 cm engrène \"a\"", "ne se touchent pas"),
        ("roue (0.06, 0.08) rayon 4.5 cm dents 6 sur \"m\" nom \"a\"\nroue (0.15, 0.08) rayon 4.5 cm dents 8 engrène \"a\"", "tailles différentes"),
        ("roue (0.06, 0.08) rayon 2 cm dents 12 sur \"m\"", "trop fines pour la maille"),
        ("roue (0.06, 0.08) rayon 4.5 cm dents 6 sur \"m\" nom \"a\"\nroue (0.15, 0.08) rayon 4.5 cm engrène \"a\"", "une roue dentée engrène une roue dentée"),
    ] {
        let p = proto(&format!("{}{bad}\n", start.replace("grille 40 x 30", "grille 60 x 40")));
        let e = initial_state(&p, Path::new(".")).err().unwrap_or_else(|| panic!("« {bad} » aurait dû être refusé"));
        assert!(e.contains(what), "« {bad} » : {e}");
    }
}

// ------------------------------------------------------------------------------------------------- bords ouverts, pas effondré

#[test]
fn air_drawn_in_through_an_open_side_has_the_total_pressure_of_the_outside_and_settles() {
    // un puits tire 1 m/s par le fond d'une cuve ouverte en haut (sans pesanteur) : l'air entre à 1 m/s, sa pression est celle du
    // dehors moins ½ ρ v² (Bernoulli depuis le repos), dans toute la cuve
    let r = run(
        "départ scénario \"Cuve vide\" grille 20 x 20 maille 1 cm
gravité (0, 0)
puits rect (0, 0) (0.2, 0.01) 1 m/s
mesurer toutes les 50 ms : p en (0.1, 0.195), vy en (0.1, 0.195), p en (0.05, 0.1)
simuler 0.5 s
",
        None,
    );
    let last = r.rows.last().unwrap();
    let (p_top, v_top, p_mid) = (last[column(&r, "p en (0.1, 0.195) (Pa)")], last[column(&r, "vy en (0.1, 0.195) (m/s)")], last[column(&r, "p en (0.05, 0.1) (Pa)")]);
    let expect = 101325.0 - 0.5 * r.sim.ambient_density(101325.0) * v_top * v_top;
    assert!((v_top + 1.0).abs() < 1e-6, "{v_top}");
    assert!((p_top - expect).abs() < 1e-3 && (p_mid - p_top).abs() < 1e-6, "p {p_top} (milieu {p_mid}) pour {expect}");
    // derrière un obstacle, l'écoulement s'établit ; avec la pression statique du dehors, le filet le plus rapide recevait plus
    // d'énergie que ses voisins et accélérait sans fin (vitesse la plus grande de 3.4 à 4.7 m/s en 2 s ; sous un puits fort, une
    // face du bord à 1000 m/s)
    let r = run(
        "départ scénario \"Cuve vide\" grille 30 x 30 maille 1 cm
puits rect (0, 0) (0.3, 0.02) 2 m/s
solide rect (0.1, 0.12) (0.2, 0.15) \"acier\"
mesurer toutes les 0.5 s : max vitesse dans partout
simuler 1.5 s
",
        None,
    );
    let v: Vec<f64> = r.rows.iter().map(|row| row[column(&r, "max vitesse partout (m/s)")]).collect();
    eprintln!("vitesse la plus grande derrière l'obstacle : {v:?}");
    assert!(v.len() == 4 && v[3] < 4.0 && (v[3] / v[1] - 1.0).abs() < 1e-3, "{v:?}");
}

#[test]
fn a_collapsing_time_step_stops_the_run_with_its_reason() {
    // une vitesse qui monte vers la limite (ici abaissée à 0.3 m/s : de l'eau qui tombe) : les pas acceptés raccourcissaient sans fin,
    // sans état invalide, et le temps n'avançait plus
    let p = proto(
        "départ scénario \"Cuve vide\" grille 10 x 20 maille 1 cm
remplir rect (0.03, 0.12) (0.07, 0.18) avec \"eau\" à 20 °C
simuler 1 s
",
    );
    let (mut sim, _, _) = initial_state(&p, Path::new(".")).unwrap();
    sim.speed_limit = 0.3;
    let mut steps = 0;
    while !sim.diverged && steps < 20_000 {
        sim.step();
        steps += 1;
    }
    assert!(sim.diverged && sim.last_dt < super::mp_step::DT_COLLAPSE, "{steps} pas, dt {:e}, t {}", sim.last_dt, sim.time);
    assert!(sim.time < 0.5 && steps < 20_000, "{steps} pas, t {}", sim.time);
    // le protocole s'arrête avec la raison
    let mut r = Runner::new(p, Path::new("."), None, Some(false)).unwrap();
    r.sim.speed_limit = 0.3;
    let e = run_until_failure(&mut r).expect("le pas effondré aurait dû arrêter l'expérience");
    assert!(e.contains("le pas de temps s'effondre") && e.contains("une vitesse s'emballe"), "{e}");
}

#[test]
fn a_sponge_layer_reads_its_options_and_ramps_its_rate_up_from_its_entrance() {
    use super::mp_cells::MpEffect;
    let p = proto("départ scénario \"cuve vide\"\néponge partout\néponge rect (0.1, 0) (0.2, 0.1) temps 50 ms vers (0.1, 0) m/s uniforme\n");
    let acts: Vec<&Action> = p.steps.iter().filter_map(|(s, _)| if let Step::Do(a) = s { Some(a) } else { None }).collect();
    assert!(matches!(acts[0], Action::Sponge { tau, target, uniform: false, .. } if *tau == 0.1 && *target == [0.0, 0.0]), "{:?}", acts[0]);
    assert!(
        matches!(acts[1], Action::Sponge { tau, target, uniform: true, .. } if (*tau - 0.05).abs() < 1e-15 && *target == [0.1, 0.0]),
        "{:?}",
        acts[1]
    );
    for (bad, want) in [("éponge partout temps 0 s", "positif"), ("éponge partout épaisse", "inconnue après « éponge »")] {
        let e = errors_in(&format!("départ scénario \"cuve vide\"\n{bad}\n"), Path::new("."));
        let e = e.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n");
        assert!(e.contains(want), "{bad} : {e}");
    }
    // bande de 10 mailles le long du bord droit : l'entrée est sa colonne de gauche, le taux monte en ((d + 1) / 10)² jusqu'à 1/τ
    // contre le bord ; un solide dans la zone est laissé, ses voisines ne sont pas une entrée
    let text = "départ scénario \"cuve vide\" grille 40 x 20 maille 1 cm
solide rect (0.39, 0.1) (0.4, 0.11) \"acier\"
éponge rect (0.3, 0) (0.4, 0.2) temps 0.1 s
simuler 1 ms
";
    let r = run(text, None);
    let log = r.log.join("\n");
    assert!(log.contains("éponge rect (0.3, 0) (0.4, 0.2) temps 0.1 s — 199 mailles"), "{log}");
    let sim = &r.sim;
    let rate = |i: usize, j: usize| match sim.special[sim.idx(i, j)] {
        Some(MpEffect::Sponge { rate, ux, uy }) if ux == 0.0 && uy == 0.0 => rate,
        e => panic!("maille ({i}, {j}) : {e:?}"),
    };
    for j in [0, 9, 10, 11, 19] {
        for k in (0..10).filter(|&k| (k, j) != (9, 10)) {
            let x = (k + 1) as f64 / 10.0;
            assert!((rate(30 + k, j) - 10.0 * x * x).abs() < 1e-12, "({}, {j}) : {} pour {}", 30 + k, rate(30 + k, j), 10.0 * x * x);
        }
    }
    assert!(sim.is_solid(sim.idx(39, 10)) && sim.special[sim.idx(39, 10)].is_none() && sim.special[sim.idx(29, 5)].is_none());
    // zone sans entrée (tout le fluide) : uniforme
    let (sim, _, _) = initial_state(&proto("départ scénario \"cuve vide\" grille 10 x 10 maille 1 cm\néponge partout temps 0.5 s\n"), Path::new(".")).unwrap();
    assert!(sim.special.iter().all(|e| *e == Some(MpEffect::Sponge { rate: 2.0, ux: 0.0, uy: 0.0 })));
}

/// tourbillon de Taylor–Green dans une boîte d'eau fermée, sans gravité, avec ou sans une éponge uniforme : (temps, vitesse max)
fn sponge_vortex(sponge: &str, gpu: bool) -> Option<Vec<(f64, f64)>> {
    let text = format!(
        "départ scénario \"cuve vide\" grille 20 x 20 maille 5 mm
gravité (0, 0)
bord haut mur
remplir partout avec \"eau\" à 20 °C
champ vx formule \"0.01*sin(pi*x/0.1)*cos(pi*y/0.1)\"
champ vy formule \"-0.01*cos(pi*x/0.1)*sin(pi*y/0.1)\"
{sponge}
pas max 1 ms
mesurer toutes les 20 ms : max vitesse dans partout comme \"v\"
simuler 0.2 s
"
    );
    if gpu {
        let probe = build_scenario(MpScenario::DamBreak, 40, 30, 0.01, 9.81);
        super::mp_gpu_tests::gpu_or_skip(&probe)?;
    }
    let mut r = Runner::new(proto(&text), Path::new("."), None, Some(gpu)).unwrap();
    assert_eq!(r.gpu.is_some(), gpu, "{}", r.log.join("\n"));
    run_to_end(&mut r);
    let k = column(&r, "v (m/s)");
    Some(r.rows.iter().map(|row| (row[0], row[k])).collect())
}

#[test]
fn a_uniform_sponge_damps_the_flow_as_exp_minus_t_over_tau() {
    // l'éponge (τ = 0.1 s) amortit toutes les vitesses d'un même facteur exp(−t/τ) — rapporté au même tourbillon sans éponge, que le
    // schéma amortit un peu de lui-même ; le pas de 1 ms (implicite) donne 1 % de plus au bout de 0.2 s
    for gpu in [false, true] {
        let (Some(free), Some(damped)) = (sponge_vortex("", gpu), sponge_vortex("éponge partout temps 0.1 s", gpu)) else {
            return;
        };
        let what = if gpu { "GPU" } else { "CPU" };
        assert_eq!(free.len(), damped.len());
        for (f, d) in free.iter().zip(&damped).skip(1) {
            let got = (d.1 / damped[0].1) / (f.1 / free[0].1);
            let want = (-f.0 / 0.1).exp();
            eprintln!("{what} : t = {:.2} s, rapport {got:.4} pour exp(−t/τ) = {want:.4}", f.0);
            assert!((got / want - 1.0).abs() < 0.03, "{what} : t = {} s, vitesse ramenée à {got} au lieu de {want}", f.0);
        }
    }
}

/// bassin de 1.2 m, 10 cm d'eau : (temps, eau au-dessus de 10 cm entre 0.3 et 0.4 m) après une bosse d'eau lâchée contre le bord gauche
fn sponge_basin(sponge: &str, hump: bool) -> Vec<(f64, f64, f64)> {
    let text = format!(
        "départ scénario \"Cuve vide\" grille 120 x 25 maille 1 cm
remplir rect (0, 0) (1.2, 0.1) avec \"eau\" à 20 °C
{}
{sponge}
mesurer toutes les 20 ms : masse \"eau\" dans rect (0.3, 0.1) (0.4, 0.25) comme \"sonde\", max vitesse dans partout comme \"v\"
simuler {}
",
        if hump { "remplir rect (0, 0.1) (0.15, 0.13) avec \"eau\" à 20 °C" } else { "" },
        if hump { "3 s" } else { "0.5 s" }
    );
    let r = run(&text, None);
    let (k, v) = (column(&r, "sonde (kg/m)"), column(&r, "v (m/s)"));
    r.rows.iter().map(|row| (row[0], row[k], row[v])).collect()
}

#[test]
fn a_sponge_leaves_water_at_rest_alone_and_absorbs_the_waves_that_reach_it() {
    let sponge = "éponge rect (0.7, 0) (1.2, 0.25) temps 0.1 s";
    // au repos : l'éponge n'amortit que la vitesse, l'équilibre hydrostatique reste tel quel — à quelques 1e-8 m/s près (1e-11 sans
    // éponge ; ce reste croît un peu avec le taux, 3e-9 m/s à τ = 1 s, 5e-8 à 0.01 s, et ne grandit pas avec le temps)
    let (still, still_sp) = (sponge_basin("", false), sponge_basin(sponge, false));
    let vmax = |r: &[(f64, f64, f64)]| r.iter().map(|x| x.2).fold(0.0, f64::max);
    eprintln!("eau au repos : vitesse max {:.3e} m/s sans éponge, {:.3e} avec", vmax(&still), vmax(&still_sp));
    assert!(vmax(&still_sp) < 1e-6 && vmax(&still) < 1e-6, "eau au repos : {} m/s avec l'éponge, {} sans", vmax(&still_sp), vmax(&still));
    // une bosse d'eau lâchée au bord gauche : ses vagues courent vers la droite ; contre un mur elles reviennent, dans l'éponge
    // (0.5 m, la moitié de la longueur d'onde des plus longues) elles s'amortissent — l'eau qui oscille à la sonde après leur
    // retour (de 1.5 à 3 s) : écart-type 0.36 kg/m avec un mur, 0.13 avec l'éponge (0.06 dans un bassin trois fois plus long)
    let spread = |r: &[(f64, f64, f64)]| {
        let s: Vec<f64> = r.iter().filter(|x| x.0 >= 1.5).map(|x| x.1).collect();
        let m = s.iter().sum::<f64>() / s.len() as f64;
        (s.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / s.len() as f64).sqrt()
    };
    let (wall, sp) = (spread(&sponge_basin("", true)), spread(&sponge_basin(sponge, true)));
    eprintln!("vagues réfléchies à la sonde : écart-type {wall:.3} kg/m contre un mur, {sp:.3} avec l'éponge");
    assert!(sp < 0.5 * wall, "l'éponge réfléchit les vagues : {sp} contre {wall} pour un mur");
}

/// vitesses des faces u de la colonne x = i dx (rangées j = 0..ny), et des faces v de la rangée y = j dx (colonnes i = 0..nx)
fn u_column(s: &MultiphaseSim, i: usize) -> Vec<f64> {
    (0..s.ny).map(|j| s.u[s.uidx(i, j)]).collect()
}
fn v_row(s: &MultiphaseSim, j: usize) -> Vec<f64> {
    (0..s.nx).map(|i| s.v[s.vidx(i, j)]).collect()
}

/// extremum d'une suite de valeurs régulièrement espacées, affiné par la parabole des trois valeurs qui l'entourent : (rang, valeur)
fn parabolic_peak(v: &[f64], min: bool) -> (f64, f64) {
    let k = (0..v.len()).min_by(|&a, &b| if min { v[a].total_cmp(&v[b]) } else { v[b].total_cmp(&v[a]) }).unwrap();
    if k == 0 || k + 1 == v.len() {
        return (k as f64, v[k]);
    }
    let (a, b, c) = (v[k - 1], v[k], v[k + 1]);
    let den = a - 2.0 * b + c;
    if den == 0.0 {
        return (k as f64, b);
    }
    let d = 0.5 * (a - c) / den;
    (k as f64 + d, b - 0.25 * (a - c) * d)
}

#[test]
fn walls_can_be_made_adhering_or_moving_along_their_side() {
    let p = proto("départ scénario \"cuve vide\"\nbord bas mur adhérent\nbord haut mur mobile (0.1, 0) m/s\nbord gauche paroi mobile (0, -0.2) m/s\nbord droite mur\n");
    let acts: Vec<&Action> = p.steps.iter().filter_map(|(s, _)| if let Step::Do(a) = s { Some(a) } else { None }).collect();
    assert!(matches!(acts[0], Action::Boundary { side: 2, spec: BoundarySpec::NoSlip { v } } if *v == 0.0), "{:?}", acts[0]);
    assert!(matches!(acts[1], Action::Boundary { side: 3, spec: BoundarySpec::NoSlip { v } } if *v == 0.1), "{:?}", acts[1]);
    assert!(matches!(acts[2], Action::Boundary { side: 0, spec: BoundarySpec::NoSlip { v } } if *v == -0.2), "{:?}", acts[2]);
    assert!(matches!(acts[3], Action::Boundary { side: 1, spec: BoundarySpec::Wall }), "{:?}", acts[3]);
    // une paroi mobile glisse le long de son bord
    let e = errors_in("départ scénario \"cuve vide\"\nbord haut mur mobile (0, 0.1) m/s\n", Path::new("."));
    let e = e.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n");
    assert!(e.contains("vitesse normale est nulle"), "{e}");
    // posée, puis rendue au glissement libre par un autre bord ; un bord ouvert n'adhère pas
    let r = run("départ scénario \"cuve vide\" grille 10 x 10 maille 1 cm\nbord bas mur adhérent\nbord haut mur mobile (0.1, 0) m/s\nbord gauche mur mobile (0, 0.3) m/s\nsimuler 1 ms\nbord gauche ouvert\nsimuler 1 ms\n", None);
    let log = r.log.join("\n");
    assert!(log.contains("bord bas : mur adhérent") && log.contains("bord haut : mur mobile à 0.1 m/s"), "{log}");
    assert_eq!(r.sim.no_slip, [None, None, Some(0.0), Some(0.1)]);
    assert_eq!([r.sim.wall_slide(0), r.sim.wall_slide(2), r.sim.wall_slide(3)], [None, Some(0.0), Some(0.1)]);
}

#[test]
fn a_moving_lid_drags_the_fluid_of_a_closed_channel_into_the_exact_profile() {
    // canal fermé de 5 cm × 1 cm plein d'air, fond adhérent, couvercle qui glisse à U = 5 cm/s : loin des bouts, le débit est nul et
    // u(y) = U (3η² − 2η), η = y / H (un contre-courant de U/3 au tiers de la hauteur) ; à 10 mailles dans la hauteur, à 0.7 % de U
    // (0.15 % à 20 mailles)
    let text = "départ scénario \"cuve vide\" grille 50 x 10 maille 1 mm
gravité (0, 0)
remplir partout avec \"air\" à 20 °C
bord bas mur adhérent
bord haut mur mobile (0.05, 0) m/s
simuler 6 s
";
    let r = run(text, None);
    let u = u_column(&r.sim, 25);
    let mut worst = 0.0f64;
    for (j, &uj) in u.iter().enumerate() {
        let eta = (j as f64 + 0.5) / 10.0;
        let want = 0.05 * (3.0 * eta * eta - 2.0 * eta);
        eprintln!("η = {eta:.3} : u = {uj:+.5} m/s (exact {want:+.5})");
        worst = worst.max((uj - want).abs() / 0.05);
    }
    assert!(worst < 0.01, "écart au profil exact : {worst} U");
}

#[test]
fn air_blown_between_two_adhering_walls_takes_the_poiseuille_profile() {
    // canal de 1 cm entre deux parois adhérentes, air entré à 5 cm/s à gauche (Re = 33), sorti à droite : à 3.5 cm de l'entrée, le
    // profil est établi, parabolique — u = 6 U η (1 − η), 1.5 U au milieu — à 0.5 % de U ; la pression baisse de 12 μ U / H² par m
    let text = "départ scénario \"cuve vide\" grille 120 x 20 maille 0.5 mm
gravité (0, 0)
remplir partout avec \"air\" à 20 °C
bord bas mur adhérent
bord haut mur adhérent
bord gauche entrée (0.05, 0) m/s à 20 °C \"air\"
bord droite sortie (0.05, 0) m/s
simuler 2 s
";
    let r = run(text, None);
    let u = u_column(&r.sim, 70);
    let mut worst = 0.0f64;
    for (j, &uj) in u.iter().enumerate() {
        let eta = (j as f64 + 0.5) / 20.0;
        worst = worst.max((uj - 0.05 * 6.0 * eta * (1.0 - eta)).abs() / 0.05);
    }
    let (_, top) = parabolic_peak(&u, false);
    eprintln!("Poiseuille : vitesse au milieu {:.4} U (1.5 U), écart au profil {worst:.4} U", top / 0.05);
    assert!(worst < 0.005, "écart au profil de Poiseuille : {worst} U");
}

/// cavité carrée de 2 cm pleine d'air, trois parois adhérentes, couvercle à 7.5 cm/s (Re = U L / ν = 100), 40 × 40 mailles, 3 s
fn lid_driven_cavity(gpu: bool) -> Option<MultiphaseSim> {
    let text = "départ scénario \"cuve vide\" grille 40 x 40 maille 0.5 mm
gravité (0, 0)
remplir partout avec \"air\" à 20 °C
bord gauche mur adhérent
bord droite mur adhérent
bord bas mur adhérent
bord haut mur mobile (0.075, 0) m/s
simuler 3 s
";
    if gpu {
        let probe = build_scenario(MpScenario::DamBreak, 40, 30, 0.01, 9.81);
        super::mp_gpu_tests::gpu_or_skip(&probe)?;
    }
    let mut r = Runner::new(proto(text), Path::new("."), None, Some(gpu)).unwrap();
    assert_eq!(r.gpu.is_some(), gpu, "{}", r.log.join("\n"));
    run_to_end(&mut r);
    Some(r.sim)
}

#[test]
fn a_lid_driven_cavity_turns_like_the_reference_solutions_at_re_100() {
    // extrema des vitesses sur les axes de la cavité, rapportés à U : solutions fines u_min −0.2140, v_max 0.1796, v_min −0.2538
    // (Ghia, Ghia et Shin 1982 : −0.2109, 0.1753, −0.2453). À 40 × 40 mailles, à 2.5 % près (mesuré 1.7 à 2.4 % ; 80 × 80 : 0.5 à
    // 0.7 %, ordre 2), et à leur place ; le GPU comme le CPU
    let (u_lid, n) = (0.075, 40.0);
    let mut seen = vec![];
    for gpu in [false, true] {
        let Some(s) = lid_driven_cavity(gpu) else { break };
        let what = if gpu { "GPU" } else { "CPU" };
        // u sur l'axe vertical (faces de la colonne x = L/2, à y = (j + ½) dx), v sur l'axe horizontal (faces de la rangée y = L/2)
        let (yu, umin) = parabolic_peak(&u_column(&s, 20), true);
        let v = v_row(&s, 20);
        let ((xa, vmax), (xb, vmin)) = (parabolic_peak(&v, false), parabolic_peak(&v, true));
        let got = [umin / u_lid, vmax / u_lid, vmin / u_lid];
        let at = [(yu + 0.5) / n, (xa + 0.5) / n, (xb + 0.5) / n];
        eprintln!("{what} : u_min {:+.4} en y = {:.3}, v_max {:+.4} en x = {:.3}, v_min {:+.4} en x = {:.3}", got[0], at[0], got[1], at[1], got[2], at[2]);
        for (k, (want, place)) in [(-0.2140, 0.458), (0.1796, 0.237), (-0.2538, 0.810)].into_iter().enumerate() {
            assert!((got[k] / want - 1.0).abs() < 0.025, "{what} : extremum {k} {} au lieu de {want}", got[k]);
            assert!((at[k] - place).abs() < 0.02, "{what} : extremum {k} en {} au lieu de {place}", at[k]);
        }
        seen.push(got);
    }
    if let [c, g] = &seen[..] {
        for k in 0..3 {
            assert!((g[k] / c[k] - 1.0).abs() < 2e-3, "extremum {k} : GPU {} CPU {}", g[k], c[k]);
        }
    }
}
