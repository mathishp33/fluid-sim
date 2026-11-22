//! Parallélisme du solveur CPU.
//!
//! Deux règles :
//!  1. Le résultat ne dépend ni du nombre de fils ni du découpage : chaque cellule (ou face, ou ligne) est calculée par un seul fil,
//!     avec les opérations et l'ordre du calcul séquentiel ; les sommes sur la grille se font par blocs de taille FIXE (`dsum`). Une
//!     simulation rechargée continue donc identiquement, sur n'importe quelle machine (avant, les sommes de rayon du gradient conjugué
//!     rendaient les grilles de plus de 16 384 cellules non reproductibles d'une exécution à l'autre).
//!  2. Des fils qui ATTENDENT EN TOURNANT. Mesuré sur un portable (Ryzen 7, 8 cœurs / 16 fils, Windows en mode équilibré) : un calcul
//!     continu sur 8 fils va 7.4× plus vite que sur un seul, mais des rafales de travail (régions parallèles de quelques ms séparées de
//!     calcul séquentiel, le cas d'un pas de simulation) ne vont pas plus vite qu'en séquentiel, avec rayon comme avec des fils système :
//!     Windows laisse en veille (« parking ») les cœurs d'une charge intermittente et y entasse les fils réveillés. Avec rayon, le pas
//!     était même 2 à 3× PLUS LENT sur 16 fils que sur un seul. Ici, les fils du solveur attendent le travail suivant (quelques µs en
//!     tournant, puis en cédant le processeur à chaque tour : si les cœurs manquent, un fil qui tournerait sans céder priverait de
//!     processeur celui qui exécute une tranche) tant que la simulation avance ; ils s'endorment au bout de `spin_time` sans travail.
//!     Même ainsi, ce portable bride par moments les charges sur plusieurs cœurs (alimentation, chaleur) : `refresh` en 160×120 va 3.6×
//!     plus vite sur 4 fils, 4.8× sur 6… puis, quelques secondes plus tard, 2× PLUS LENTEMENT. D'où le réglage automatique du nombre de
//!     fils (`set_auto`, utilisé par la fenêtre), et des valeurs `ligne de cache` isolées (`Padded`).
//!
//! `for_ranges` (et `for_rows`, `fill`, `collect`, `dsum`, `dmax`) découpe une boucle en tranches contiguës, exécutées par les fils du
//! groupe et par le fil appelant. Une boucle trop petite (argument `min`), un appel imbriqué (depuis une tranche) ou concurrent (un
//! autre fil utilise déjà le groupe : tests en parallèle, deux simulations) s'exécute séquentiellement — même résultat (règle 1).

use super::multiphase::SyncPtr;
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::{Duration, Instant};

/// travail minimal d'une tranche, en éléments, pour les noyaux LÉGERS (quelques opérations par cellule : produit matrice–vecteur,
/// Gauss–Seidel, combinaisons de vecteurs)
pub(crate) const MIN_LIGHT: usize = 2048;
/// … pour les noyaux LOURDS (équations d'état, températures depuis les énergies, flux PLIC, chimie : 0.1–1 µs par cellule)
pub(crate) const MIN_HEAVY: usize = 256;

/// durée pendant laquelle un fil sans travail attend en tournant avant de s'endormir (µs ; `MP_SPIN_US` pour l'essayer)
fn spin_time() -> Duration {
    static SPIN_US: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    Duration::from_micros(*SPIN_US.get_or_init(|| std::env::var("MP_SPIN_US").ok().and_then(|v| v.parse().ok()).unwrap_or(5000)))
}

/// nombre de fils par défaut (fil appelant compris) : `MP_CPU_THREADS`, sinon `RAYON_NUM_THREADS`, sinon la moitié des processeurs
/// logiques (un fil par cœur physique sur les processeurs à deux fils par cœur : les deux fils d'un même cœur se gênent ici) ; un seul
/// sous `cargo test` (les tests tournent déjà en parallèle, des fils qui attendent en tournant les ralentiraient)
pub fn default_threads() -> usize {
    for var in ["MP_CPU_THREADS", "RAYON_NUM_THREADS"] {
        if let Some(n) = std::env::var(var).ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0) {
            return n;
        }
    }
    if cfg!(test) {
        return 1;
    }
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    (logical / 2).max(1)
}

// ---------------------------------------------------------------------------------------------------------------- groupe de fils

/// `next` : génération (24 bits) | nombre de tranches k (20 bits) | prochaine tranche i (20 bits)
const KB: u64 = 20;
const MASK: u64 = (1 << KB) - 1;
fn pack(g: u64, k: u64, i: u64) -> u64 {
    (g << (2 * KB)) | (k << KB) | i
}
fn unpack(x: u64) -> (u64, u64, u64) {
    (x >> (2 * KB), (x >> KB) & MASK, x & MASK)
}

/// travail en cours : la fermeture, sur la pile de l'appelant (vivante tant qu'une tranche n'est pas terminée)
struct Job<'a> {
    f: &'a (dyn Fn(usize) + Sync),
}

/// valeur seule sur sa ligne de cache (128 octets : deux lignes, préchargement par paires) : les fils en attente relisent `next` sans
/// arrêt ; partagée avec une autre donnée écrite par le fil principal (le tas range les petits objets côte à côte), chaque écriture
/// forcerait sept cœurs à recharger la ligne — mesuré : un calcul séquentiel 7 à 9× plus lent juste après une région parallèle
#[repr(align(128))]
struct Padded<T>(T);
impl<T> std::ops::Deref for Padded<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0
    }
}

struct Shared {
    next: Padded<AtomicU64>,
    remaining: Padded<AtomicUsize>,
    job: AtomicPtr<()>,
    panicked: AtomicBool,
    shutdown: AtomicBool,
    sleepers: Mutex<usize>,
    wake: Condvar,
    _pad: Padded<()>,
}

impl Shared {
    /// prend une tranche du travail en cours et l'exécute ; false s'il n'y en a plus
    fn run_one(&self) -> bool {
        let mut cur = self.next.load(Ordering::Acquire);
        loop {
            let (g, k, i) = unpack(cur);
            if i >= k {
                return false;
            }
            match self.next.compare_exchange_weak(cur, pack(g, k, i + 1), Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => {
                    // la tranche i est à nous : le travail ne peut pas changer avant qu'elle soit finie (remaining > 0)
                    let job = self.job.load(Ordering::Acquire) as *const Job;
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        IN_JOB.with(|c| c.set(true));
                        // SAFETY : voir ci-dessus, l'appelant attend remaining == 0 avant de libérer `Job`
                        unsafe { ((*job).f)(i as usize) };
                        IN_JOB.with(|c| c.set(false));
                    }));
                    if r.is_err() {
                        IN_JOB.with(|c| c.set(false));
                        self.panicked.store(true, Ordering::Relaxed);
                    }
                    self.remaining.fetch_sub(1, Ordering::Release);
                    return true;
                }
                Err(now) => cur = now,
            }
        }
    }
    fn has_work(&self) -> bool {
        let (_, k, i) = unpack(self.next.load(Ordering::Acquire));
        i < k
    }
}

fn worker(sh: Arc<Shared>) {
    loop {
        // attente active (les cœurs restent éveillés entre les régions d'un pas), puis sommeil
        let t0 = Instant::now();
        let spin = spin_time();
        let mut n = 0u32;
        while !sh.has_work() && !sh.shutdown.load(Ordering::Relaxed) {
            // d'abord quelques µs d'attente active pure, puis en CÉDANT le processeur à chaque tour : si Windows ne laisse au processus
            // qu'une partie des cœurs (cœurs en veille), un fil qui tournerait sans céder priverait de processeur le fil qui exécute
            // une tranche, et toute la région l'attendrait un quantum entier (mesuré : au-delà de 4 fils, tranches 7 à 13× plus longues)
            n = n.wrapping_add(1);
            if n < 2048 {
                std::hint::spin_loop();
                continue;
            }
            std::thread::yield_now();
            if n % 64 == 0 && t0.elapsed() > spin {
                let mut s = sh.sleepers.lock().unwrap_or_else(|e| e.into_inner());
                *s += 1;
                while !sh.has_work() && !sh.shutdown.load(Ordering::Relaxed) {
                    s = sh.wake.wait(s).unwrap_or_else(|e| e.into_inner());
                }
                *s -= 1;
                break;
            }
        }
        if sh.shutdown.load(Ordering::Relaxed) {
            return;
        }
        while sh.run_one() {}
    }
}

struct Pool {
    threads: usize,
    shared: Arc<Shared>,
    /// un seul appelant à la fois (les autres calculent en séquentiel)
    busy: Mutex<()>,
    gen: AtomicU64,
}

impl Pool {
    fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared {
            next: Padded(AtomicU64::new(0)),
            remaining: Padded(AtomicUsize::new(0)),
            job: AtomicPtr::new(std::ptr::null_mut()),
            panicked: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            sleepers: Mutex::new(0),
            wake: Condvar::new(),
            _pad: Padded(()),
        });
        for w in 1..threads {
            let sh = shared.clone();
            std::thread::Builder::new()
                .name(format!("solveur-cpu-{w}"))
                .stack_size(8 << 20)
                .spawn(move || worker(sh))
                .expect("fil du solveur CPU");
        }
        Pool { threads, shared, busy: Mutex::new(()), gen: AtomicU64::new(0) }
    }

    /// f(0), …, f(k − 1), en parallèle ; false si le groupe est occupé (l'appelant fait alors le travail lui-même)
    fn run(&self, k: usize, f: &(dyn Fn(usize) + Sync)) -> bool {
        let Ok(_guard) = self.busy.try_lock() else { return false };
        let sh = &*self.shared;
        let job = Job { f };
        sh.job.store(&job as *const Job as *mut (), Ordering::Release);
        sh.remaining.store(k, Ordering::Relaxed);
        sh.panicked.store(false, Ordering::Relaxed);
        let g = (self.gen.fetch_add(1, Ordering::Relaxed) + 1) & ((1 << 24) - 1);
        sh.next.store(pack(g, k as u64, 0), Ordering::Release);
        if *sh.sleepers.lock().unwrap_or_else(|e| e.into_inner()) > 0 {
            sh.wake.notify_all();
        }
        while sh.run_one() {}
        let mut n = 0u32;
        while sh.remaining.load(Ordering::Acquire) > 0 {
            n = n.wrapping_add(1);
            if n < 2048 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        if sh.panicked.load(Ordering::Relaxed) {
            panic!("une tranche parallèle du solveur CPU a paniqué");
        }
        true
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Relaxed);
        let _s = self.shared.sleepers.lock();
        self.shared.wake.notify_all();
    }
}

static POOL: RwLock<Option<Arc<Pool>>> = RwLock::new(None);

thread_local! {
    /// vrai pendant l'exécution d'une tranche (un appel imbriqué est fait en séquentiel)
    static IN_JOB: Cell<bool> = const { Cell::new(false) };
}

fn pool() -> Arc<Pool> {
    if let Some(p) = POOL.read().ok().and_then(|g| g.clone()) {
        return p;
    }
    let mut g = POOL.write().unwrap_or_else(|e| e.into_inner());
    g.get_or_insert_with(|| Arc::new(Pool::new(default_threads()))).clone()
}

/// nombre de fils du solveur CPU (fil appelant compris)
pub fn threads() -> usize {
    pool().threads
}

/// change le nombre de fils du solveur CPU (0 = valeur par défaut). Sans effet sur les résultats (règle 1), seulement sur la vitesse.
pub fn set_threads(n: usize) {
    let n = if n == 0 { default_threads() } else { n };
    let mut g = POOL.write().unwrap_or_else(|e| e.into_inner());
    if g.as_ref().map(|p| p.threads) != Some(n) {
        *g = Some(Arc::new(Pool::new(n)));
    }
}

/// nombre maximal de fils proposé (processeurs logiques)
pub fn max_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

// ---------------------------------------------------------------------------------------------------------------- réglage automatique

/// Réglage automatique du nombre de fils (activé par la fenêtre ; les tests et bancs d'essai gardent un nombre fixe). Le bon nombre
/// dépend de la grille (une petite grille ne nourrit pas 8 fils) et de l'état de la machine : sur un portable mesuré, 8 fils allaient
/// 1.5 à 2.3× plus vite qu'un seul sur 160×120, puis, machine chaude et alimentation limitée, 2× PLUS LENTEMENT. Le temps de chaque
/// pas est mesuré ; après `TUNE_MEASURE` pas, on essaie la moitié ou le double (en alternance) pendant `TUNE_MEASURE` pas et l'on garde
/// le plus rapide (5 % d'avance exigés) ; nouvel essai après `TUNE_PAUSE` pas. Sans effet sur les résultats (règle 1).
#[derive(Clone, Copy)]
enum TunePhase {
    Base,
    Trial { prev: usize, base: f64 },
    Wait { n: usize, until: usize },
}
struct Tuner {
    on: bool,
    /// sens du prochain essai (au départ : moins de fils)
    up: bool,
    phase: TunePhase,
    /// durées des derniers pas de la mesure en cours (médiane : un pas exceptionnel — retour arrière, sous-pas — ne décide pas)
    samples: [f64; TUNE_MEASURE],
    n: usize,
}
static TUNER: Mutex<Tuner> = Mutex::new(Tuner { on: false, up: false, phase: TunePhase::Base, samples: [0.0; TUNE_MEASURE], n: 0 });
const TUNE_MEASURE: usize = 15;
const TUNE_PAUSE: usize = 400;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// active / désactive le réglage automatique du nombre de fils
pub fn set_auto(on: bool) {
    let mut t = TUNER.lock().unwrap_or_else(|e| e.into_inner());
    if t.on != on {
        t.on = on;
        t.phase = TunePhase::Base;
        t.n = 0;
    }
}

/// réglage automatique actif ?
pub fn auto_threads() -> bool {
    TUNER.lock().map(|t| t.on).unwrap_or(false)
}

/// durée d'un pas du solveur CPU (appelé par `MultiphaseSim::step`) : nourrit le réglage automatique
pub(crate) fn record_step(secs: f64) {
    let mut t = TUNER.lock().unwrap_or_else(|e| e.into_inner());
    if !t.on || !secs.is_finite() {
        return;
    }
    if let TunePhase::Wait { n, until } = t.phase {
        t.phase = if n + 1 >= until { TunePhase::Base } else { TunePhase::Wait { n: n + 1, until } };
        t.n = 0;
        return;
    }
    let k = t.n;
    t.samples[k] = secs;
    t.n += 1;
    if t.n < TUNE_MEASURE {
        return;
    }
    t.n = 0;
    let med = median(&mut t.samples);
    t.phase = match t.phase {
        TunePhase::Base => {
            let cur = threads();
            let (lo, hi) = ((cur / 2).max(1), (cur * 2).min(max_threads()));
            let cand = if t.up { if hi != cur { hi } else { lo } } else if lo != cur { lo } else { hi };
            if cand == cur {
                TunePhase::Wait { n: 0, until: TUNE_PAUSE }
            } else {
                set_threads(cand);
                TunePhase::Trial { prev: cur, base: med }
            }
        }
        TunePhase::Trial { prev, base } => {
            if med < 0.95 * base {
                // plus rapide : on garde, et l'on poursuit bientôt dans le même sens
                TunePhase::Wait { n: 0, until: TUNE_MEASURE }
            } else {
                set_threads(prev);
                t.up = !t.up;
                TunePhase::Wait { n: 0, until: TUNE_PAUSE }
            }
        }
        TunePhase::Wait { .. } => TunePhase::Base,
    };
}

// ---------------------------------------------------------------------------------------------------------------- boucles

/// nombre de tranches pour `len` éléments d'au moins `min` chacun (au plus deux par fil ; 1 = séquentiel)
fn chunks(threads: usize, len: usize, min: usize) -> usize {
    if threads <= 1 || IN_JOB.with(|c| c.get()) {
        1
    } else {
        (len / min.max(1)).clamp(1, 2 * threads).min(MASK as usize)
    }
}

/// Appelle f(début, fin) sur des tranches contiguës de 0..len d'au moins `min` éléments, en parallèle s'il y en a au moins deux.
/// f ne doit écrire que dans ce que couvre sa tranche (ses cellules, ses faces, ses lignes).
pub(crate) fn for_ranges(len: usize, min: usize, f: impl Fn(usize, usize) + Sync) {
    let p = pool();
    let k = chunks(p.threads, len, min);
    if k > 1 {
        let g = |q: usize| {
            let (lo, hi) = (q * len / k, (q + 1) * len / k);
            if lo < hi {
                f(lo, hi)
            }
        };
        if p.run(k, &g) {
            return;
        }
    }
    f(0, len);
}

/// [f(lo₀, hi₀), f(lo₁, hi₁), …] : un résultat par tranche contiguë de 0..len (au moins `min` éléments), dans l'ordre des tranches
pub(crate) fn map_ranges<T: Send>(len: usize, min: usize, f: impl Fn(usize, usize) -> T + Sync) -> Vec<T> {
    let p = pool();
    let k = chunks(p.threads, len, min);
    if k > 1 {
        let slots: Vec<Mutex<Option<T>>> = (0..k).map(|_| Mutex::new(None)).collect();
        let g = |q: usize| {
            let r = f(q * len / k, (q + 1) * len / k);
            *slots[q].lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
        };
        if p.run(k, &g) {
            return slots.into_iter().map(|m| m.into_inner().unwrap_or_else(|e| e.into_inner()).expect("tranche calculée")).collect();
        }
    }
    vec![f(0, len)]
}

/// f(j) pour chaque ligne j de 0..ny (lignes de `nx` cellules), par tranches de lignes d'au moins `min` cellules
pub(crate) fn for_rows(ny: usize, nx: usize, min: usize, f: impl Fn(usize) + Sync) {
    for_ranges(ny, min.div_ceil(nx.max(1)), |j0, j1| {
        for j in j0..j1 {
            f(j)
        }
    });
}

/// out[i] = f(i) pour tout i, par tranches d'au moins `min` éléments
pub(crate) fn fill<T: Copy + Send>(out: &mut [T], min: usize, f: impl Fn(usize) -> T + Sync) {
    let len = out.len();
    let p = SyncPtr(out.as_mut_ptr());
    for_ranges(len, min, |lo, hi| {
        for i in lo..hi {
            // SAFETY : chaque i appartient à une seule tranche
            unsafe { *p.at(i) = f(i) }
        }
    });
}

/// vecteur [f(0), …, f(len − 1)], calculé par tranches d'au moins `min` éléments
pub(crate) fn collect<T: Copy + Send + Default>(len: usize, min: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let mut v = vec![T::default(); len];
    fill(&mut v, min, f);
    v
}

/// en dessous de cette longueur, `dsum` est la somme séquentielle simple (celle d'avant : résultats inchangés sur les petites grilles)
const SUM_SEQ_MAX: usize = 16384;
/// taille FIXE des blocs de `dsum` au-delà (le résultat ne dépend donc pas du nombre de fils)
const SUM_BLOCK: usize = 4096;

/// Σ f(i), i ∈ 0..len, DÉTERMINISTE : séquentielle sous `SUM_SEQ_MAX`, sinon par blocs fixes de `SUM_BLOCK` (chacun sommé dans
/// l'ordre, puis les blocs dans l'ordre), calculés en parallèle
pub(crate) fn dsum(len: usize, f: impl Fn(usize) -> f64 + Sync) -> f64 {
    if len < SUM_SEQ_MAX {
        return (0..len).map(f).sum();
    }
    let nb = len.div_ceil(SUM_BLOCK);
    let mut part = vec![0.0f64; nb];
    let pp = SyncPtr(part.as_mut_ptr());
    for_ranges(nb, MIN_LIGHT.div_ceil(SUM_BLOCK), |b0, b1| {
        for b in b0..b1 {
            let s: f64 = (b * SUM_BLOCK..((b + 1) * SUM_BLOCK).min(len)).map(&f).sum();
            // SAFETY : un bloc par indice
            unsafe { *pp.at(b) = s }
        }
    });
    part.iter().sum()
}

/// produit scalaire déterministe (voir `dsum`)
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    dsum(a.len(), |i| a[i] * b[i])
}

/// max(init, f(0), …, f(len − 1)) — exact quel que soit le découpage (NaN ignorés, comme `f64::max`)
#[allow(dead_code)]
pub(crate) fn dmax(len: usize, min: usize, init: f64, f: impl Fn(usize) -> f64 + Sync) -> f64 {
    let p = pool();
    let k = chunks(p.threads, len, min);
    if k <= 1 {
        return (0..len).map(f).fold(init, f64::max);
    }
    let mut part = vec![init; k];
    let pp = SyncPtr(part.as_mut_ptr());
    for_ranges(k, 1, |q0, q1| {
        for q in q0..q1 {
            let m = (q * len / k..(q + 1) * len / k).map(&f).fold(init, f64::max);
            // SAFETY : une tranche par indice
            unsafe { *pp.at(q) = m }
        }
    });
    part.into_iter().fold(init, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_sum_and_loops_do_not_depend_on_the_thread_count() {
        let n = 100_003;
        let v: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.37).sin() * 1e3 + 1e-7 * i as f64).collect();
        let s1 = dsum(n, |i| v[i]);
        let m1 = dmax(n, 1000, f64::NEG_INFINITY, |i| v[i]);
        let bits = |w: Vec<f64>| w.into_iter().map(f64::to_bits).collect::<Vec<u64>>();
        let c1 = bits(collect(n, 100, |i| v[i].abs().sqrt()));
        // petites longueurs : somme séquentielle exacte d'avant
        assert_eq!(dsum(1000, |i| v[i]).to_bits(), v[..1000].iter().sum::<f64>().to_bits());
        for _ in 0..50 {
            assert_eq!(dsum(n, |i| v[i]).to_bits(), s1.to_bits());
            assert_eq!(dmax(n, 1000, f64::NEG_INFINITY, |i| v[i]).to_bits(), m1.to_bits());
            assert_eq!(bits(collect(n, 100, |i| v[i].abs().sqrt())), c1);
        }
        set_threads(4);
        for _ in 0..20 {
            assert_eq!(dsum(n, |i| v[i]).to_bits(), s1.to_bits());
            assert_eq!(bits(collect(n, 100, |i| v[i].abs().sqrt())), c1);
        }
        set_threads(0);
        // imbrication : séquentiel à l'intérieur d'une tranche
        let tot = AtomicUsize::new(0);
        for_ranges(64, 1, |a, b| {
            for_ranges(100, 1, |c, d| {
                tot.fetch_add((b - a) * (d - c), Ordering::Relaxed);
            })
        });
        assert_eq!(tot.load(Ordering::Relaxed), 6400);
    }
}
