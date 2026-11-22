//! Per-cell material updates, their inverses (time reversal) and their
//! hand-written adjoints.
//!
//! These are the scalar kernels of the dispersive (pole) and nonlinear
//! (chi2 / chi3) updates of [`crate::engine::cpu`] and of the GPU kernels,
//! written as free functions so that the reverse pass, the tests and, with
//! the `enzyme` feature, `#[autodiff_reverse]` all work on the same code.
//! The GPU kernels of the reverse pass repeat these formulas.

use crate::Real;
/// ADE pole step `P(n+1) = c0 P(n) + c1 P(n-1) + c2 E(n)`.
#[inline(always)]
pub fn pole_step(p: Real, pprev: Real, e: Real, c: [Real; 3]) -> Real {
    c[0] * p + c[1] * pprev + c[2] * e
}

/// Time reversal of a dispersive cell. After forward step n a pole holds
/// `P(n+1)` (`p_new`) and `P(n)` (`p_cur`, used by second-order poles only).
/// With `E(n+1) = ca E(n) + cb curl - scale sum_q (P_q(n+1) - P_q(n))` and,
/// for first-order poles (`c1 = 0`), `P(n) = (P(n+1) - c2 E(n)) / c0`:
///
/// ```text
/// E(n) = g / (1 - scale/ca sum_first c2/c0),
/// g = (E(n+1) + scale D - cb curl) / ca,
/// D = sum_second (P(n+1) - P(n)) + sum_first (P(n+1) - P(n+1)/c0)
/// ```
///
/// where `g` is what the linear reversal returns for `E(n+1) + scale D`.
/// This is the pole's contribution to `D`.
#[inline(always)]
pub fn pole_reverse_d(p_new: Real, p_cur: Real, c: [Real; 3]) -> Real {
    if c[1] != 0.0 { p_new - p_cur } else { p_new - p_new / c[0] }
}

/// The pole's contribution to `sum_first c2/c0` (see [`pole_reverse_d`]).
#[inline(always)]
pub fn pole_reverse_kappa(c: [Real; 3]) -> Real {
    if c[1] != 0.0 { 0.0 } else { c[2] / c[0] }
}

/// Pole state `(P(n), P(n-1))` before forward step n, from the state after
/// it and `E(n)` (first-order poles do not use `P(n-1)`: zero).
#[inline(always)]
pub fn pole_reverse_state(p_new: Real, p_cur: Real, e: Real, c: [Real; 3]) -> (Real, Real) {
    if c[1] != 0.0 { (p_cur, (p_new - c[0] * p_cur - c[2] * e) / c[1]) } else { ((p_new - c[2] * e) / c[0], 0.0) }
}

/// One step of a dispersive cell with two poles: from `E(n)` (`e`), the
/// linear update `lin = ca E(n) + cb curl H`, and the pole states `p`, `pp`
/// to `E(n+1)` and the new pole states (written to `out = [e', p0', p1',
/// pp0', pp1']`). `c` holds the six ADE coefficients.
pub fn disp_step2(x: &[Real; 6], c: &[Real; 6], scale: Real, out: &mut [Real; 5]) {
    let (e, lin) = (x[0], x[1]);
    let mut dp = 0.0 as Real;
    for q in 0..2 {
        let (p, pp) = (x[2 + q], x[4 + q]);
        let pn = pole_step(p, pp, e, [c[3 * q], c[3 * q + 1], c[3 * q + 2]]);
        dp += pn - p;
        out[1 + q] = pn;
        out[3 + q] = p;
    }
    out[0] = lin - scale * dp;
}

/// Hand-written vector-Jacobian product of [`disp_step2`]: the cotangent of
/// `x = [e, lin, p0, p1, pp0, pp1]` for the cotangent `g` of `out`.
pub fn disp_step2_vjp(c: &[Real; 6], scale: Real, g: &[Real; 5]) -> [Real; 6] {
    let mut d = [0 as Real; 6];
    d[1] = g[0];
    for q in 0..2 {
        // cotangent of P(n+1): its own, and through E' = lin - scale (P' - P)
        let w = g[1 + q] - scale * g[0];
        d[0] += c[3 * q + 2] * w;
        d[2 + q] = c[3 * q] * w + scale * g[0] + g[3 + q];
        d[4 + q] = c[3 * q + 1] * w;
    }
    d
}

/// One reverse step of the pole-coefficient gradient of a dispersive cell.
/// `a` is the (true) cotangent of E(n+1), `s` the cell's polarisation scale
/// (`E(n+1) = lin - s sum (P(n+1) - P(n))`), `(p, pp, e)` = P(n), P(n-1),
/// E(n) and `(gp, gpp)` the carried cotangents of the pole's outputs P(n+1)
/// and P(n) (the copy kept as P(n-1)), updated in place to those of P(n) and
/// P(n-1). Returns `dL/d(c0, c1, c2)` of this step. The E cotangent itself is
/// propagated by the adjoint field (see [`disp_step2_vjp`], of which this is
/// the pole part).
#[inline(always)]
pub fn pole_coef_step(a: Real, s: Real, p: Real, pp: Real, e: Real, c: [Real; 3], gp: &mut Real, gpp: &mut Real) -> [Real; 3] {
    let w = *gp - s * a;
    let g = c[0] * w + s * a + *gpp;
    *gpp = c[1] * w;
    *gp = g;
    [w * p, w * pp, w * e]
}

/// True for the padding entries of the coefficient table (media with fewer
/// poles than the table width): `[1, 0, 0]`.
#[inline(always)]
pub fn pole_is_padding(c: [Real; 3]) -> bool {
    c[0] == 1.0 && c[1] == 0.0 && c[2] == 0.0
}

/// "Displacement" of a chi2/chi3 cell: `D(x) = x + a x^2 + b x^3`
/// (`a = chi2 / eps`, `b = chi3 / eps`). The nonlinear step solves
/// `D(E(n+1)) = D(E(n)) + E_lin - E(n)`.
#[inline(always)]
pub fn nl_d(x: Real, a: Real, b: Real) -> Real {
    x + a * x * x + b * x * x * x
}

#[inline(always)]
pub fn nl_dprime(x: Real, a: Real, b: Real) -> Real {
    1.0 + 2.0 * a * x + 3.0 * b * x * x
}

/// The nonlinear step, as in the engines (four Newton iterations).
#[inline(always)]
pub fn nl_step(e: Real, elin: Real, a: Real, b: Real) -> Real {
    crate::engine::cpu::nl_solve(e, elin, a, b)
}

/// Inverse of the nonlinear step: `E(n)` from `E(n+1)` (`e1`) and the output
/// `g = (E(n+1) - cb curl H) / ca` of the linear reversal applied to
/// `E(n+1)`: solves `D(e) + (ca - 1) e = D(e1) - e1 + ca g`.
pub fn nl_unsolve(e1: Real, g: Real, ca: Real, a: Real, b: Real) -> Real {
    let rhs = nl_d(e1, a, b) - e1 + ca * g;
    let mut e = g;
    for _ in 0..6 {
        let f = nl_d(e, a, b) + (ca - 1.0) * e - rhs;
        let fp = nl_dprime(e, a, b) + ca - 1.0;
        e -= f / fp;
    }
    e
}

/// Partial derivatives `(dF/de, dF/delin)` of the nonlinear step
/// `e1 = F(e, elin)` by the implicit function theorem on
/// `D(e1) = D(e) + elin - e`.
#[inline(always)]
pub fn nl_partials(e: Real, e1: Real, a: Real, b: Real) -> (Real, Real) {
    let d1 = nl_dprime(e1, a, b);
    ((nl_dprime(e, a, b) - 1.0) / d1, 1.0 / d1)
}

/// Partial derivatives `(dF/da, dF/db)` of the nonlinear step with respect
/// to its coefficients.
#[inline(always)]
pub fn nl_partials_coef(e: Real, e1: Real, a: Real, b: Real) -> (Real, Real) {
    let d1 = nl_dprime(e1, a, b);
    ((e * e - e1 * e1) / d1, (e * e * e - e1 * e1 * e1) / d1)
}

/// `(dF/de, dF/delin)` as used by the reverse pass (the implicit-function
/// form; `crate::autodiff::enzyme` checks it against the derivative Enzyme
/// generates from [`nl_step`]).
#[inline(always)]
pub fn nl_factors(e: Real, e1: Real, a: Real, b: Real) -> (Real, Real) {
    nl_partials(e, e1, a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fd(f: impl Fn(Real) -> Real, x: Real, h: Real) -> Real {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    #[test]
    fn nonlinear_step_inverse_and_partials() {
        for &(e, elin, a, b) in &[(0.3 as Real, 0.42 as Real, 0.2 as Real, 0.5 as Real), (-0.7, -0.55, 0.1, 0.3), (0.05, 0.4, -0.2, 0.8), (1.1, 0.9, 0.0, 0.25)] {
            let e1 = nl_step(e, elin, a, b);
            // the step satisfies D(e1) = D(e) + elin - e
            assert!((nl_d(e1, a, b) - nl_d(e, a, b) - (elin - e)).abs() < 2e-6);
            // inverse (lossless and lossy): g = (e1 - cb curl)/ca with cb curl = elin - ca e
            for ca in [1.0 as Real, 0.97] {
                let elin_c = elin + (ca - 1.0) * e;
                let e1c = nl_step(e, elin_c, a, b);
                let g = (e1c - (elin_c - ca * e)) / ca;
                let back = nl_unsolve(e1c, g, ca, a, b);
                assert!((back - e).abs() < 5e-6, "{back} vs {e}");
            }
            let (fe, fl) = nl_partials(e, e1, a, b);
            assert!((fe - fd(|x| nl_step(x, elin, a, b), e, 1e-2)).abs() < 2e-3, "dF/de");
            assert!((fl - fd(|x| nl_step(e, x, a, b), elin, 1e-2)).abs() < 2e-3, "dF/delin");
            let (fa, fb) = nl_partials_coef(e, e1, a, b);
            assert!((fa - fd(|x| nl_step(e, elin, x, b), a, 1e-2)).abs() < 2e-3, "dF/da");
            assert!((fb - fd(|x| nl_step(e, elin, a, x), b, 1e-2)).abs() < 2e-3, "dF/db");
        }
    }

    /// Forward steps of a cell with a Lorentz, a Drude-like and a Debye
    /// pole, then the reversal: fields and pole states come back.
    #[test]
    fn dispersive_cell_reversal() {
        let cs = [[1.93 as Real, -0.96, 0.031], [0.82, 0.0, 0.11], [1.7, -0.8, 0.05]];
        let (ca, cb, scale) = (0.98 as Real, 0.3 as Real, 0.4 as Real);
        let curl = |n: usize| (0.37 * n as Real).sin();
        let mut e = 0.2 as Real;
        let (mut p, mut pp) = ([0 as Real; 3], [0 as Real; 3]);
        let mut hist = vec![(e, p, pp)];
        for n in 0..25 {
            let mut dp = 0.0;
            for q in 0..3 {
                let pn = pole_step(p[q], pp[q], e, cs[q]);
                dp += pn - p[q];
                pp[q] = p[q];
                p[q] = pn;
            }
            e = ca * e + cb * curl(n) - scale * dp;
            hist.push((e, p, pp));
        }
        for n in (0..25).rev() {
            let (mut d, mut kappa) = (0.0, 0.0);
            for q in 0..3 {
                d += pole_reverse_d(p[q], pp[q], cs[q]);
                kappa += pole_reverse_kappa(cs[q]);
            }
            let g = (e + scale * d - cb * curl(n)) / ca;
            e = g / (1.0 - scale / ca * kappa);
            for q in 0..3 {
                (p[q], pp[q]) = pole_reverse_state(p[q], pp[q], e, cs[q]);
            }
            let (e0, p0, pp0) = hist[n];
            assert!((e - e0).abs() < 2e-4, "E at {n}: {e} vs {e0}");
            for q in 0..3 {
                assert!((p[q] - p0[q]).abs() < 2e-4, "P{q} at {n}");
                if cs[q][1] != 0.0 {
                    assert!((pp[q] - pp0[q]).abs() < 2e-4, "Pprev{q} at {n}");
                }
            }
        }
    }

    /// The hand-written VJP of the dispersive cell is the transpose of its
    /// (linear) step, and the transposed recursion run backwards in time is
    /// the forward recursion itself: the adjoint of E driven through the
    /// poles obeys `a(n) = lin-part + sum_m k_m a(n+1+m)` with the same
    /// kernel `k` that maps E(n-m) to E(n+1).
    #[test]
    fn dispersive_cell_adjoint() {
        let c = [1.93 as Real, -0.96, 0.031, 1.7, -0.8, 0.05];
        let scale = 0.4 as Real;
        // transpose: <g, J x> = <J^T g, x> on the canonical basis
        let mut jac = [[0 as Real; 6]; 5];
        for i in 0..6 {
            let mut x = [0 as Real; 6];
            x[i] = 1.0;
            let mut out = [0 as Real; 5];
            disp_step2(&x, &c, scale, &mut out);
            for o in 0..5 {
                jac[o][i] = out[o];
            }
        }
        for o in 0..5 {
            let mut g = [0 as Real; 5];
            g[o] = 1.0;
            let d = disp_step2_vjp(&c, scale, &g);
            for i in 0..6 {
                assert!((d[i] - jac[o][i]).abs() < 1e-6, "J^T[{i}][{o}]");
            }
        }
        // forward impulse response: E(0) = 1, lin = 0 afterwards -> E(n+1) = -scale dP(n+1)
        let steps = 40;
        let mut k = vec![0 as Real; steps];
        let (mut p, mut pp) = ([0 as Real; 2], [0 as Real; 2]);
        for (n, kn) in k.iter_mut().enumerate() {
            let x = [if n == 0 { 1.0 } else { 0.0 }, 0.0, p[0], p[1], pp[0], pp[1]];
            let mut out = [0 as Real; 5];
            disp_step2(&x, &c, scale, &mut out);
            *kn = out[0];
            p = [out[1], out[2]];
            pp = [out[3], out[4]];
        }
        // adjoint impulse response: cotangent 1 on E at the last step, zero
        // linear part, stepped backwards with the VJP
        let mut ka = vec![0 as Real; steps];
        let mut g = [1 as Real, 0.0, 0.0, 0.0, 0.0];
        for kn in ka.iter_mut() {
            let d = disp_step2_vjp(&c, scale, &g);
            *kn = d[0];
            // the cotangent of E through `lin` is not followed (zero linear part)
            g = [0.0, d[2], d[3], d[4], d[5]];
        }
        for n in 0..steps {
            assert!((k[n] - ka[n]).abs() < 2e-5 * k[0].abs().max(1.0), "step {n}: {} vs {}", k[n], ka[n]);
        }
    }
}
