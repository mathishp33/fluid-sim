//! Géométrie PLIC 2D exacte : reconstruction d'un plan à partir de (α, normale) et aire de la phase dans un rectangle.
//! La phase occupe {x : nx·x + ny·y ≤ c} dans la cellule unité [0,1]² (la normale (nx,ny) pointe HORS de la phase).
//! Formules analytiques (Scardovelli & Zaleski 2000) : pas d'itération, exactes à l'arrondi près.

#[derive(Clone, Copy, Debug)]
pub struct Plic {
    pub nx: f64,
    pub ny: f64,
    pub c: f64,
}

/// Volume de {m1 x + m2 y ≤ a} dans [0,1]², avec m1 ≤ m2, m1 + m2 = 1.
fn vol_from_a(m1: f64, m2: f64, a: f64) -> f64 {
    if a <= 0.0 {
        0.0
    } else if a >= 1.0 {
        1.0
    } else if m1 < 1e-12 {
        a // plan horizontal : m2 = 1
    } else if a <= m1 {
        a * a / (2.0 * m1 * m2)
    } else if a <= m2 {
        (a - 0.5 * m1) / m2
    } else {
        1.0 - (1.0 - a) * (1.0 - a) / (2.0 * m1 * m2)
    }
}

/// Inverse : constante `a` du plan pour une fraction volumique donnée.
fn a_from_vol(m1: f64, m2: f64, v: f64) -> f64 {
    if v <= 0.0 {
        0.0
    } else if v >= 1.0 {
        1.0
    } else if m1 < 1e-12 {
        v
    } else if v <= 0.5 * m1 / m2 {
        (2.0 * m1 * m2 * v).sqrt()
    } else if v <= 1.0 - 0.5 * m1 / m2 {
        v * m2 + 0.5 * m1
    } else {
        1.0 - (2.0 * m1 * m2 * (1.0 - v)).sqrt()
    }
}

impl Plic {
    /// `alpha` fraction de la phase, `(gx, gy)` gradient de α (la normale sortante est −∇α).
    pub fn new(alpha: f64, gx: f64, gy: f64) -> Plic {
        let (mut nx, mut ny) = (-gx, -gy);
        let s = nx.abs() + ny.abs();
        if s < 1e-14 {
            nx = 0.0;
            ny = 1.0;
        } else {
            nx /= s;
            ny /= s;
        }
        let (ax, ay) = (nx.abs(), ny.abs());
        let a = a_from_vol(ax.min(ay), ax.max(ay), alpha);
        let neg = nx.min(0.0) + ny.min(0.0);
        Plic { nx, ny, c: a + neg }
    }

    /// Aire de la phase dans le rectangle [x0,x1]×[y0,y1] ⊂ [0,1]²
    pub fn area(&self, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
        let (mut x0, mut x1, mut y0, mut y1) = (x0, x1, y0, y1);
        let neg = self.nx.min(0.0) + self.ny.min(0.0);
        if self.nx < 0.0 {
            (x0, x1) = (1.0 - x1, 1.0 - x0);
        }
        if self.ny < 0.0 {
            (y0, y1) = (1.0 - y1, 1.0 - y0);
        }
        let a = self.c - neg;
        let (mx, my) = (self.nx.abs(), self.ny.abs());
        if mx < 1e-9 {
            return (x1 - x0) * ((a / my).clamp(y0, y1) - y0);
        }
        if my < 1e-9 {
            return (y1 - y0) * ((a / mx).clamp(x0, x1) - x0);
        }
        let t = |s: f64| if s > 0.0 { s * s / (2.0 * mx * my) } else { 0.0 };
        let f = |xx: f64, yy: f64| t(a) - t(a - mx * xx) - t(a - my * yy) + t(a - mx * xx - my * yy);
        (f(x1, y1) - f(x0, y1) - f(x1, y0) + f(x0, y0)).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_recovers_volume_fraction() {
        for k in 0..64 {
            let th = k as f64 * 0.1;
            for &al in &[0.01, 0.2, 0.5, 0.77, 0.999] {
                let p = Plic::new(al, th.cos(), th.sin());
                let a = p.area(0.0, 1.0, 0.0, 1.0);
                assert!((a - al).abs() < 1e-12, "θ={th} α={al} → {a}");
            }
        }
    }

    #[test]
    fn strips_add_up() {
        // Aire(strip1) + Aire(strip2) = Aire(strip1 ∪ strip2)
        let p = Plic::new(0.37, 0.8, -0.6);
        let a = p.area(0.0, 0.3, 0.0, 1.0) + p.area(0.3, 1.0, 0.0, 1.0);
        assert!((a - 0.37).abs() < 1e-12);
        let b = p.area(0.2, 0.9, 0.0, 0.4) + p.area(0.2, 0.9, 0.4, 1.0);
        assert!((b - p.area(0.2, 0.9, 0.0, 1.0)).abs() < 1e-12);
    }

    #[test]
    fn flat_interface_exact() {
        // eau en bas : α = 0.4, ∇α vers le bas (gy < 0) → normale sortante vers le haut ; phase = y ≤ 0.4
        let p = Plic::new(0.4, 0.0, -1.0);
        assert!((p.area(0.0, 1.0, 0.0, 1.0) - 0.4).abs() < 1e-14);
        assert!((p.area(0.5, 1.0, 0.0, 1.0) - 0.2).abs() < 1e-14);
        assert!((p.area(0.0, 1.0, 0.3, 1.0) - 0.1).abs() < 1e-14);
        // vertical : phase x ≤ 0.25
        let q = Plic::new(0.25, -1.0, 0.0);
        assert!((q.area(0.0, 0.1, 0.0, 1.0) - 0.1).abs() < 1e-14);
        assert!((q.area(0.2, 1.0, 0.0, 1.0) - 0.05).abs() < 1e-14);
    }

    #[test]
    fn mirror_symmetry() {
        let p = Plic::new(0.3, 0.5, 0.2);
        let q = Plic::new(0.3, -0.5, 0.2); // miroir en x
        let (a, b) = (p.area(0.6, 1.0, 0.0, 1.0), q.area(0.0, 0.4, 0.0, 1.0));
        assert!((a - b).abs() < 1e-12);
    }
}
