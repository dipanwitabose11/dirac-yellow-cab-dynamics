"""
convergence_report.py -- the convergence test of Section 5 for the run from the uniform
initial generator (written by run_uniform.py). Works on a finished or a still-running run.

Writes output_c<c>/:
    report.txt                 all numbers (checks, ascent, step decay, fits, residuals, estimate)
    04_generator.png           the last iterate on the 86 states
    05_ascent_and_bounds.png   l_lambda/K, l/K (with l_sat) and R (with rho_sat) against k
    06_penalized_likelihood.png  l_lambda/K on the plane through Q^(0), Q^(8), Q^(last)
    07_convergence.png         step length, path length, increments with tail fits

    python3 convergence_report.py
    python3 convergence_report.py --c 10            # for a run started with --c 10
    python3 convergence_report.py --no-contour      # skip figure 06 (900 likelihood evaluations)
"""
import argparse
from pathlib import Path

import numpy as np

import em_tools as et

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c", type=float, default=None, help="the c used in run_uniform.py (default 74.79)")
    ap.add_argument("--ckpt", default=None, help="checkpoint file (default runs/uniform_c<c>.npz)")
    ap.add_argument("--tol", type=float, default=1e-6, help="the tolerance used in run_uniform.py")
    ap.add_argument("--no-contour", action="store_true")
    ap.add_argument("--project", default=None)
    a = ap.parse_args()

    rt, be = et.load(et.find_project(HERE, a.project))
    M, data, K, nv = rt.M, rt.data, rt.nobs, rt.nv
    c = et.uniform_rate_default(rt) if a.c is None else a.c
    ckpt = Path(a.ckpt) if a.ckpt else HERE / "runs" / f"uniform_c{c:.4g}.npz"
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} not found: run  python3 run_uniform.py" + ("" if a.c is None else f" --c {a.c:g}"))
    H = et.history(rt, be, ckpt)
    V, ll, R, gain, lpen, lam, eps = H["V"], H["ll"], H["R"], H["gain"], H["lpen"], H["lam"], H["eps"]
    n = len(V) - 1
    if n < 2:
        raise SystemExit("need at least 2 iterations")
    out = HERE / f"output_c{c:.4g}"
    out.mkdir(exist_ok=True)
    L = []
    p = L.append

    steps, vn = et.step_stats(V)
    inc = np.diff(lpen)
    Qn = np.array([np.linalg.norm(M.Q(v)) for v in V])
    lsat = be.saturated_loglik(data)
    rho_sat = R[0] + (lsat - ll[0]) / lam
    rel = steps[-1] / vn[-1]
    if H["stalled_at"] is not None:
        status = (f"STALLED: at k = {H['stalled_at']} the M-step could not improve the surrogate and kept "
                  f"the current iterate. That step is left out; the analysis uses k = 0..{n}. "
                  f"The run had NOT converged (last real step/|v| = {rel:.2e}, tol {a.tol:g}).")
    elif rel < a.tol:
        status = f"STOPPED by the tolerance (last step/|v| = {rel:.2e} < tol {a.tol:g})"
    else:
        status = f"NOT yet stopped (last step/|v| = {rel:.2e} > tol {a.tol:g})"

    p(f"Uniform initial generator: c = {V[0][0]:.4f} per hour on all {len(M.I)} transitions of S0")
    p(f"trips K = {K}, zones {nv}, links {len(rt.edges)}, states {M.n}; lambda = {lam:.4f}, floor = {eps}")
    p(f"iterations used {n}. {status}")
    p("Rates and steps are per hour; steps are Euclidean norms of the change of the 204 rates.")

    # 1. numerical checks at the last iterate ------------------------------------
    N, A, _ = be.estep(M, V[-1], data)
    T_total = sum(tau * C.sum() for tau, C in data)
    g = be.gradient(M, V[-1], data)
    rng = np.random.default_rng(0)
    fd = []
    for j in rng.choice(len(V[-1]), 5, replace=False):
        h = 1e-5 * V[-1][j]
        e = np.zeros_like(V[-1]); e[j] = h
        fd.append(abs((be.loglik(M, V[-1] + e, data) - be.loglik(M, V[-1] - e, data)) / (2 * h) - g[j]) / abs(g[j]))
    p("\n1. Numerical checks at the last iterate")
    p(f"  E-step: expected occupation times sum to the total observed time, relative error {A.sum() / T_total - 1:.1e}")
    p(f"  gradient identity vs central finite differences (5 coordinates): max relative error {max(fd):.1e}")

    # 2. results of Section 3 ---------------------------------------------------
    p("\n2. Predictions of the theory (Table 1 of the paper)")
    p(f"  ascent: l_lambda decreased in {int((inc < 0).sum())} of {n} steps; "
      f"min (increment - surrogate gain)/K = {(inc - gain).min() / K:.2e} (must be >= 0)")
    p(f"  M-step improved the surrogate in {int((gain > 0).sum())} of {n} steps")
    p(f"  saturated bound: max_k l(Q_k)/K = {ll.max() / K:.4f} <= l_sat/K = {lsat / K:.4f}")
    p(f"  boundedness: max_k R(Q_k) = {R.max():.1f} <= rho_sat = {rho_sat:.1f}")
    p(f"  lambda R at the last iterate = {lam * R[-1] / K:.4f} per trip <= lambda rho_sat = {lam * rho_sat / K:.4f}")
    p(f"  floor: eta = smallest rate over all iterates = {V.min():.3f} > floor {eps}  "
      f"(at iterate k = {int(np.unravel_index(V.argmin(), V.shape)[0])})")

    # 3. ascent ------------------------------------------------------------------
    p("\n3. Ascent")
    p(f"  l_lambda/K: {lpen[0] / K:.6f} -> {lpen[-1] / K:.6f}  (90% of the gain after {et.k90(lpen)} iterations)")
    p(f"  l/K:        {ll[0] / K:.6f} -> {ll[-1] / K:.6f}  (90% after {et.k90(ll)})")
    p(f"  R:          {R[0]:.1f} -> {R[-1]:.1f}  (90% after {et.k90(R)})")

    # 4. convergence of the iterates ---------------------------------------------
    p("\n4. Convergence of the iterates")
    p(f"  {'k':>5} {'step':>10} {'step/|v|':>10} {'step/||Q||F':>12} {'incr/K':>10} {'l_lambda/K':>12}")
    for k in sorted({1, 5, 10, 25, 50, 75, 100, 125, 150, *range(200, n, 100), n}):
        if k <= n:
            p(f"  {k:5d} {steps[k - 1]:10.4f} {steps[k - 1] / vn[k]:10.2e} {steps[k - 1] / Qn[k]:12.2e} "
              f"{inc[k - 1] / K:10.2e} {lpen[k] / K:12.6f}")
    p(f"  |v| at the end {vn[-1]:.1f}, ||Q||_F at the end {Qn[-1]:.1f}")
    p(f"  path length sum_k step = {steps.sum():.1f};  |v_end - v_0| = {np.linalg.norm(V[-1] - V[0]):.1f}")
    w = 25
    p(f"  step ratio r in windows of {w} iterations (constant r = geometric decay):")
    rs = []
    for a0 in range(max(0, n % w), n - w + 1, w):
        rs.append(et.geo_ratio(steps, a0, a0 + w))
        p(f"    k = {a0 + 1:4d}..{a0 + w:4d}:  r = {rs[-1]:.4f}")
    if rs:
        r = rs[-1]
        p(f"  remaining path if the last ratio holds = step*r/(1-r) = {steps[-1] * r / (1 - r):.3g} "
          f"({steps[-1] * r / (1 - r) / vn[-1]:.1e} of |v|)")
        p("  (a ratio that creeps up towards 1 means slower than geometric decay; then this is an underestimate)")

    # 5. increments: geometric vs power law --------------------------------------
    k0 = max(1, n // 2)
    f = et.tail_fits(inc, K, k0)
    p(f"\n5. Increments of l_lambda on k >= {k0}")
    p(f"  geometric fit: ratio {f['ratio']:.4f} (rms residual of log increments {f['rms_geo']:.3f})")
    p(f"  power-law fit: k^{f['power']:.2f} (rms residual {f['rms_pow']:.3f})")
    if f["ratio"] < 1:
        p(f"  remaining gain of l_lambda/K if the geometric fit holds: {inc[-1] / K * f['ratio'] / (1 - f['ratio']):.2e}")

    # 6. first-order condition ---------------------------------------------------
    p("\n6. First-order condition at the last iterate: min_s ||grad l - lambda s|| / ||grad l||")
    for tol in (1e-6, 1e-9):
        p(f"  frame coefficients below {tol:g} x largest treated as zero: {et.foc_relative(be, rt, V[-1], lam, tol):.3f}")

    # 7. the estimate ------------------------------------------------------------
    p(f"\n7. The last iterate Q^({n})")
    for k_, x in et.rates_summary(rt, V[-1]).items():
        p(f"  {k_:30s} [{x[0]:.2f}, {x[1]:.2f}]")

    txt = "\n".join(L)
    print(txt)
    (out / "report.txt").write_text(txt + "\n")
    figures(rt, be, H, out, lsat, rho_sat, steps, inc, f, not a.no_contour)
    print(f"\nsaved report.txt and figures in {out}")


def figures(rt, be, H, out, lsat, rho_sat, steps, inc, f, contour):
    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.bbox": "tight"})
    M, data, K = rt.M, rt.data, rt.nobs
    V, lam, eps = H["V"], H["lam"], H["eps"]
    n = len(V) - 1

    Q = M.Q(V[-1]); Am = Q.copy(); Am[Am == 0] = np.nan
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("g", ["0.0", "0.82"])
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(3.2, 3))
    im = ax.imshow(Am, cmap=cmap, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.7, label="rate (per hour)")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(out / "04_generator.png", dpi=200); plt.close(fig)

    k = np.arange(n + 1)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    ax[0].plot(k, H["lpen"] / K, c="k", lw=0.9); ax[0].set_ylabel(r"$\ell_\lambda/K$")
    ax[1].plot(k, H["ll"] / K, c="k", lw=0.9); ax[1].axhline(lsat / K, c="k", ls="--", lw=0.8)
    ax[1].set_ylabel(r"$\ell/K$")
    ax[2].plot(k, H["R"], c="k", lw=0.9); ax[2].axhline(rho_sat, c="k", ls="--", lw=0.8)
    ax[2].set_yscale("log"); ax[2].set_ylabel(r"$R_{Q^{(k)}}$")
    for x in ax:
        x.set_xlabel("iteration $k$")
    fig.tight_layout(); fig.savefig(out / "05_ascent_and_bounds.png", dpi=200); plt.close(fig)

    if contour:
        print("computing the contour plot (900 likelihood evaluations) ...", flush=True)
        e1 = V[-1] - V[0]; L1 = np.linalg.norm(e1); e1 = e1 / L1
        w8 = V[min(8, n)] - V[0]; e2 = w8 - (w8 @ e1) * e1; e2 /= np.linalg.norm(e2)
        s = np.linspace(-0.25 * L1, 1.25 * L1, 30)
        t = np.linspace(-1.5, 1.5, 30) * max(abs((V - V[0]) @ e2).max(), 1e-3)
        Z = np.full((len(t), len(s)), np.nan)
        for i, ti in enumerate(t):
            for j, sj in enumerate(s):
                v = V[0] + sj * e1 + ti * e2
                if v.min() >= eps:
                    Z[i, j] = (be.loglik(M, v, data) - lam * M.R(v)) / K
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        ax.contour(s, t, Z, 15, colors="0.5", linewidths=0.5, linestyles="solid")
        ax.plot((V - V[0]) @ e1, (V - V[0]) @ e2, ".-", c="k", ms=2, lw=0.5)
        ax.set_xlabel(rf"along $v^{{({n})}}-v^{{(0)}}$ (per hour)")
        ax.set_ylabel(r"orthogonal, through $v^{(8)}$")
        fig.savefig(out / "06_penalized_likelihood.png", dpi=200); plt.close(fig)

    kk = np.arange(1, n + 1)
    m = f["mask"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    ax[0].semilogy(kk, steps, c="k", lw=0.9); ax[0].set_ylabel(r"$\|v^{(k)}-v^{(k-1)}\|$")
    ax[1].plot(kk, np.cumsum(steps), c="k", lw=0.9); ax[1].set_ylabel("path length")
    pos = inc > 0
    ax[2].loglog(kk[pos], inc[pos] / K, ".", c="k", ms=2)
    g0, g1 = f["geo"]; p0, p1 = f["pow"]
    ax[2].loglog(kk[m], np.exp(g0 + g1 * kk[m]), "--", c="0.4", lw=0.8, label=rf"${f['ratio']:.3f}^k$")
    ax[2].loglog(kk[m], np.exp(p0) * kk[m] ** p1, ":", c="k", lw=0.8, label=rf"$k^{{{p1:.2f}}}$")
    ax[2].set_ylabel(r"$\Delta\ell_\lambda/K$"); ax[2].legend(frameon=False, fontsize=7)
    for x in ax:
        x.set_xlabel("iteration $k$")
    fig.tight_layout(); fig.savefig(out / "07_convergence.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
