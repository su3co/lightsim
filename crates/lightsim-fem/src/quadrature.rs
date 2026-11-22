//! Quadrature rules on the reference triangle and on the unit interval.
//!
//! Triangle rules are the symmetric rules of D. A. Dunavant, "High degree
//! efficient symmetrical Gaussian quadrature rules for the triangle",
//! Int. J. Numer. Meth. Eng. 21, 1129-1148 (1985). Points are barycentric
//! coordinates; weights sum to 1 (multiply by the triangle area).

/// A rule on the reference triangle: barycentric points and weights (sum 1).
#[derive(Clone, Debug)]
pub struct TriRule {
    pub points: Vec<[f64; 3]>,
    pub weights: Vec<f64>,
}

fn push_s3(r: &mut TriRule, w: f64) {
    r.points.push([1.0 / 3.0; 3]);
    r.weights.push(w);
}
fn push_s21(r: &mut TriRule, w: f64, a: f64) {
    let b = 1.0 - 2.0 * a;
    for p in [[a, a, b], [a, b, a], [b, a, a]] {
        r.points.push(p);
        r.weights.push(w);
    }
}
fn push_s111(r: &mut TriRule, w: f64, a: f64, b: f64) {
    let c = 1.0 - a - b;
    for p in [
        [a, b, c],
        [a, c, b],
        [b, a, c],
        [b, c, a],
        [c, a, b],
        [c, b, a],
    ] {
        r.points.push(p);
        r.weights.push(w);
    }
}

/// The Dunavant rule exact for polynomials of total degree `degree`
/// (1, 2, 4, 5, 6 or 8; other degrees round up; above 8 the degree-8 rule).
pub fn triangle(degree: usize) -> TriRule {
    let mut r = TriRule {
        points: vec![],
        weights: vec![],
    };
    match degree {
        0 | 1 => push_s3(&mut r, 1.0),
        2 => push_s21(&mut r, 1.0 / 3.0, 1.0 / 6.0),
        3 | 4 => {
            push_s21(&mut r, 0.223381589678011, 0.445948490915965);
            push_s21(&mut r, 0.109951743655322, 0.091576213509771);
        }
        5 => {
            push_s3(&mut r, 0.225);
            push_s21(&mut r, 0.132394152788506, 0.470142064105115);
            push_s21(&mut r, 0.125939180544827, 0.101286507323456);
        }
        6 => {
            push_s21(&mut r, 0.116786275726379, 0.249286745170910);
            push_s21(&mut r, 0.050844906370207, 0.063089014491502);
            push_s111(
                &mut r,
                0.082851075618374,
                0.053145049844817,
                0.310352451033784,
            );
        }
        _ => {
            push_s3(&mut r, 0.144315607677787);
            push_s21(&mut r, 0.095091634267285, 0.459292588292723);
            push_s21(&mut r, 0.103217370534718, 0.170569307751760);
            push_s21(&mut r, 0.032458497623198, 0.050547228317031);
            push_s111(
                &mut r,
                0.027230314174435,
                0.008394777409958,
                0.263112829634638,
            );
        }
    }
    r
}

/// Gauss-Legendre rule on [0, 1] with `n` points (1 to 5): (points, weights summing to 1).
pub fn line(n: usize) -> (Vec<f64>, Vec<f64>) {
    let (x, w): (Vec<f64>, Vec<f64>) = match n {
        0 | 1 => (vec![0.0], vec![2.0]),
        2 => {
            let a = 1.0 / 3f64.sqrt();
            (vec![-a, a], vec![1.0, 1.0])
        }
        3 => {
            let a = (0.6f64).sqrt();
            (vec![-a, 0.0, a], vec![5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        }
        4 => {
            let a = (3.0 / 7.0 - 2.0 / 7.0 * (1.2f64).sqrt()).sqrt();
            let b = (3.0 / 7.0 + 2.0 / 7.0 * (1.2f64).sqrt()).sqrt();
            let wa = (18.0 + 30f64.sqrt()) / 36.0;
            let wb = (18.0 - 30f64.sqrt()) / 36.0;
            (vec![-b, -a, a, b], vec![wb, wa, wa, wb])
        }
        _ => {
            let a = (5.0 - 2.0 * (10.0f64 / 7.0).sqrt()).sqrt() / 3.0;
            let b = (5.0 + 2.0 * (10.0f64 / 7.0).sqrt()).sqrt() / 3.0;
            let wa = (322.0 + 13.0 * 70f64.sqrt()) / 900.0;
            let wb = (322.0 - 13.0 * 70f64.sqrt()) / 900.0;
            (vec![-b, -a, 0.0, a, b], vec![wb, wa, 128.0 / 225.0, wa, wb])
        }
    };
    (
        x.iter().map(|t| 0.5 * (t + 1.0)).collect(),
        w.iter().map(|t| 0.5 * t).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fact(n: u32) -> f64 {
        (1..=n).map(|k| k as f64).product()
    }

    #[test]
    fn triangle_rules_integrate_monomials() {
        // int over the reference triangle (area 1/2) of L1^a L2^b L3^c = a! b! c! / (a+b+c+2)!
        for (deg, _) in [(1, 0), (2, 0), (4, 0), (5, 0), (6, 0), (8, 0)] {
            let r = triangle(deg);
            let s: f64 = r.weights.iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "weights of degree {deg}: {s}");
            for a in 0..=deg as u32 {
                for b in 0..=(deg as u32 - a) {
                    let c = deg as u32 - a - b;
                    let exact = fact(a) * fact(b) * fact(c) / fact(a + b + c + 2);
                    let q: f64 = r
                        .points
                        .iter()
                        .zip(&r.weights)
                        .map(|(p, w)| {
                            w * p[0].powi(a as i32) * p[1].powi(b as i32) * p[2].powi(c as i32)
                        })
                        .sum::<f64>()
                        * 0.5;
                    assert!(
                        (q - exact).abs() < 1e-11 * exact.max(1e-3),
                        "deg {deg} a{a} b{b} c{c}: {q} vs {exact}"
                    );
                }
            }
        }
    }

    #[test]
    fn line_rules() {
        for n in 1..=5 {
            let (x, w) = line(n);
            for k in 0..(2 * n) {
                let q: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(k as i32)).sum();
                assert!((q - 1.0 / (k as f64 + 1.0)).abs() < 1e-13);
            }
        }
    }
}
