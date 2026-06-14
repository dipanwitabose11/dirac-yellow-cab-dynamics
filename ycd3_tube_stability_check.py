"""
Stability verification for the theorem-respecting 95% bridge tube on YCD.

Identical procedure to pp_tube_stability_check.py, adapted for YCD's CSV
generators and zone-id labels. Three checks:

  (A) Pointwise bound containment.
        |hat_p(tau, c) - p_ref(tau, c)|  <=  eps_marg_c(tau)  for every (tau, c).
  (B) Set-mass coverage under p_ref.
        sum_{c in A_tau}  p_ref(tau, c)  >=  level (default 0.95) at every tau.
  (C) Hypothesis margin.
        hat_P_T(a,b)  >  T * rho.

We treat Q_emp as the "true" reference Q (since the true generator on real
taxi data is unknown) and check that the theorem's bound contains the
empirical disagreement |hat_p - p_ref|. If it does, the rho is large
enough that the theorem genuinely guarantees the 95% tube; if not, rho
must be increased.

Outputs (under <out>/petal_3d/stability/):
    figures/
        stability_pointwise_a<a>_b<b>_T<T>.png    gap & bound heatmaps
        stability_set_a<a>_b<b>_T<T>.png          per-tau coverage curve
        stability_summary_a<a>_b<b>_T<T>.png      compact one-pager
    stability_report_a<a>_b<b>_T<T>.txt           plain-text verdict
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.linalg import expm, norm

EPS = 1e-14


def bridge_marginal(Q, a, b, T, tau, PT=None):
    if PT is None:
        PT = expm(T * Q)
    Pt = expm(tau * Q)
    Prem = expm((T - tau) * Q)
    p = Pt[a, :] * Prem[:, b] / max(PT[a, b], EPS)
    p = np.maximum(p, 0.0)
    return p / max(p.sum(), EPS)


def joint_eps_per_state(Qhat, a, b, T, tau, rho):
    Pt = expm(tau * Qhat); Prem = expm((T - tau) * Qhat)
    return tau * rho * Prem[:, b] + (T - tau) * rho * Pt[a, :] + tau * (T - tau) * rho * rho


def conditional_bridge_bound(Qhat, a, b, T, tau, rho, eps_c, PT=None):
    if PT is None:
        PT = expm(T * Qhat)
    PT_ab = float(PT[a, b])
    E_T = T * rho
    denom = PT_ab - E_T
    if denom <= 0:
        return np.full_like(eps_c, np.inf)
    Pt = expm(tau * Qhat); Prem = expm((T - tau) * Qhat)
    Nhat_c = Pt[a, :] * Prem[:, b]
    return eps_c / denom + Nhat_c * E_T / (PT_ab * denom)


def hdr_with_theorem(p, eps_marg, level=0.95):
    order = np.argsort(p)[::-1]
    cum_lower = 0.0
    keep = np.zeros_like(p, dtype=bool)
    for idx in order:
        keep[idx] = True
        cum_lower += max(float(p[idx] - eps_marg[idx]), 0.0)
        if cum_lower >= level:
            break
    return keep


def hdr_no_theorem(p, level=0.95):
    order = np.argsort(p)[::-1]
    cum = 0.0
    keep = np.zeros_like(p, dtype=bool)
    for idx in order:
        keep[idx] = True
        cum += float(p[idx])
        if cum >= level:
            break
    return keep


def compute_stability(Qhat, Qref, a, b, T, n_tau, rho, level):
    n = Qhat.shape[0]
    PT_h = expm(T * Qhat); PT_r = expm(T * Qref)
    PT_ab_hat = float(PT_h[a, b]); PT_ab_ref = float(PT_r[a, b])
    E_T = T * rho

    taus = np.linspace(0.0, T, n_tau)
    P_hat = np.zeros((n_tau, n))
    P_ref = np.zeros((n_tau, n))
    Eps = np.zeros((n_tau, n))
    A_thm = np.zeros((n_tau, n), dtype=bool)
    A_naive = np.zeros((n_tau, n), dtype=bool)

    for ti, tau in enumerate(taus):
        if tau <= 0.0 or tau >= T:
            P_hat[ti] = 0.0; P_ref[ti] = 0.0
            P_hat[ti, a] = 1.0 if tau <= 0 else 0.0
            P_hat[ti, b] = 1.0 if tau >= T else 0.0
            P_ref[ti] = P_hat[ti].copy()
            A_thm[ti, a] = tau <= 0; A_thm[ti, b] = tau >= T
            A_naive[ti] = A_thm[ti]
            continue
        P_hat[ti] = bridge_marginal(Qhat, a, b, T, float(tau), PT=PT_h)
        P_ref[ti] = bridge_marginal(Qref, a, b, T, float(tau), PT=PT_r)
        eps_c = joint_eps_per_state(Qhat, a, b, T, float(tau), rho)
        eps_marg = conditional_bridge_bound(Qhat, a, b, T, float(tau), rho, eps_c, PT=PT_h)
        Eps[ti] = eps_marg
        A_thm[ti] = hdr_with_theorem(P_hat[ti], eps_marg, level=level)
        A_naive[ti] = hdr_no_theorem(P_hat[ti], level=level)

    Gap = np.abs(P_hat - P_ref)
    in_band = Gap <= Eps
    set_mass_ref_thm = np.array([P_ref[ti, A_thm[ti]].sum() for ti in range(n_tau)])
    set_mass_hat_thm = np.array([P_hat[ti, A_thm[ti]].sum() for ti in range(n_tau)])
    set_mass_ref_naive = np.array([P_ref[ti, A_naive[ti]].sum() for ti in range(n_tau)])
    set_mass_hat_naive = np.array([P_hat[ti, A_naive[ti]].sum() for ti in range(n_tau)])

    return {
        "taus": taus,
        "P_hat": P_hat, "P_ref": P_ref, "Eps": Eps,
        "Gap": Gap, "in_band": in_band,
        "A_thm": A_thm, "A_naive": A_naive,
        "set_mass_ref_thm": set_mass_ref_thm,
        "set_mass_hat_thm": set_mass_hat_thm,
        "set_mass_ref_naive": set_mass_ref_naive,
        "set_mass_hat_naive": set_mass_hat_naive,
        "PT_ab_hat": PT_ab_hat, "PT_ab_ref": PT_ab_ref, "E_T": E_T,
        "rho": rho, "T": T, "n": n, "level": level,
        "n_in_band": int(in_band.sum()),
        "n_total": int(in_band.size),
        "fraction_in_band": float(in_band.mean()),
        "max_gap": float(Gap.max()),
        "max_eps": float(Eps[np.isfinite(Eps)].max() if np.isfinite(Eps).any() else np.inf),
    }


def figure_pointwise(res, a, b, zones, outpath):
    P_hat, P_ref, Gap, Eps = res["P_hat"], res["P_ref"], res["Gap"], res["Eps"]
    taus = res["taus"]; n = res["n"]
    finite_eps = np.where(np.isfinite(Eps), Eps, 0.0)
    vmax = max(float(Gap.max()), float(finite_eps.max()), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                                gridspec_kw={"width_ratios": [1, 1, 1.05]})
    GRAY = LinearSegmentedColormap.from_list("g", ["#ffffff", "#1a1a1a"])
    DIVERGING = LinearSegmentedColormap.from_list(
        "d", ["#138a3f", "#cdcdcd", "#c7271a"])

    im0 = axes[0].imshow(Gap.T, aspect="auto", origin="lower",
                            extent=[taus[0], taus[-1], -0.5, n - 0.5],
                            cmap=GRAY, vmin=0, vmax=vmax, interpolation="nearest")
    axes[0].set_title(r"empirical gap $|\hat p - p_{\rm ref}|$", fontsize=10)
    axes[0].set_xlabel(r"time $\tau$"); axes[0].set_ylabel("state index $c$")
    plt.colorbar(im0, ax=axes[0], fraction=0.04, pad=0.02)

    im1 = axes[1].imshow(finite_eps.T, aspect="auto", origin="lower",
                            extent=[taus[0], taus[-1], -0.5, n - 0.5],
                            cmap="YlOrBr", vmin=0, vmax=vmax, interpolation="nearest")
    axes[1].set_title(r"theorem bound $\varepsilon_{\rm marg}(\tau, c)$", fontsize=10)
    axes[1].set_xlabel(r"time $\tau$"); axes[1].set_yticks([])
    plt.colorbar(im1, ax=axes[1], fraction=0.04, pad=0.02)

    diff = (Gap - finite_eps).T
    vrange = max(float(np.abs(diff).max()), 1e-6)
    norm_div = TwoSlopeNorm(vmin=-vrange, vcenter=0.0, vmax=vrange)
    im2 = axes[2].imshow(diff, aspect="auto", origin="lower",
                            extent=[taus[0], taus[-1], -0.5, n - 0.5],
                            cmap=DIVERGING, norm=norm_div, interpolation="nearest")
    axes[2].set_title(r"violation map $|\hat p - p_{\rm ref}| - \varepsilon$"
                       " (green = within bound, red = violates)", fontsize=10)
    axes[2].set_xlabel(r"time $\tau$"); axes[2].set_yticks([])
    plt.colorbar(im2, ax=axes[2], fraction=0.04, pad=0.02)

    z_a = f"z{zones[a]}" if zones else f"a={a}"
    z_b = f"z{zones[b]}" if zones else f"b={b}"
    fig.suptitle(
        f"YCD pointwise stability: empirical gap vs theorem bound  "
        f"({z_a} → {z_b},  T = {res['T']},  ρ = {res['rho']:.4g})\n"
        f"in-band fraction = {res['fraction_in_band']:.4f}  "
        f"({res['n_in_band']} of {res['n_total']} grid cells)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outpath, dpi=200); plt.close(fig)


def figure_set(res, a, b, zones, outpath):
    taus = res["taus"]
    z_a = f"z{zones[a]}" if zones else f"a={a}"
    z_b = f"z{zones[b]}" if zones else f"b={b}"
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(taus, res["set_mass_ref_thm"], color="#7a1410", lw=1.8,
             marker="o", ms=4, label=r"$p_{\rm ref}$-mass on $A_\tau$ (theorem-respecting)")
    ax.plot(taus, res["set_mass_hat_thm"], color="#138a3f", lw=1.4,
             marker="s", ms=3, label=r"$\hat p$-mass on $A_\tau$  (must $\geq 0.95$)")
    ax.plot(taus, res["set_mass_ref_naive"], color="#7a1410", lw=1.0, ls=":",
             label=r"$p_{\rm ref}$-mass on $A_\tau^{\rm naive}$ (no-error HDR)")
    ax.plot(taus, res["set_mass_hat_naive"], color="#138a3f", lw=1.0, ls=":",
             label=r"$\hat p$-mass on $A_\tau^{\rm naive}$ (no-error HDR)")
    ax.axhline(res["level"], color="0.0", lw=1.0, ls="--",
                label=f"target = {res['level']}")
    ax.set_xlabel(r"time $\tau$"); ax.set_ylabel("mass on the chosen set")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        f"YCD set-level stability: does $A_\\tau$ carry "
        f"$\\geq {res['level']}$ mass under $p_{{\\rm ref}}$?  "
        f"({z_a} → {z_b},  T = {res['T']},  ρ = {res['rho']:.4g})",
        fontsize=10)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200); plt.close(fig)


def figure_summary(res, a, b, zones, outpath):
    z_a = f"z{zones[a]}" if zones else f"a={a}"
    z_b = f"z{zones[b]}" if zones else f"b={b}"
    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.30)

    ax1 = fig.add_subplot(gs[0, 0])
    margin = (res["PT_ab_hat"] - res["E_T"]) / max(res["PT_ab_hat"], EPS)
    cats = [r"$\hat P_T(a,b)$", r"$E_T = T\rho$"]
    vals = [res["PT_ab_hat"], res["E_T"]]
    ax1.bar(cats, vals, color=["#138a3f", "#cdc4b1"], edgecolor="0.2")
    for ci, v in zip(cats, vals):
        ax1.text(ci, v, f"  {v:.4g}", ha="center", va="bottom", fontsize=9)
    ax1.set_title(f"(C) hypothesis margin = {margin:.2%}\n"
                   r"need $\hat P_T(a,b) > E_T$",
                   fontsize=10)
    ax1.set_ylabel("value")

    ax2 = fig.add_subplot(gs[0, 1])
    in_band_per_tau = res["in_band"].mean(axis=1)
    ax2.plot(res["taus"], in_band_per_tau, color="#138a3f", lw=1.6)
    ax2.axhline(1.0, color="0.5", lw=0.8, ls=":")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel(r"$\tau$"); ax2.set_ylabel("fraction of states in band")
    ax2.set_title(f"(A) pointwise in-band fraction per τ\n"
                   f"global = {res['fraction_in_band']:.3f}", fontsize=10)
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(res["taus"], res["set_mass_ref_thm"], color="#7a1410", lw=1.8, marker="o", ms=3,
              label=r"$p_{\rm ref}$-mass on $A_\tau$ (theorem-respecting)")
    ax3.plot(res["taus"], res["set_mass_ref_naive"], color="#0b1d2a", lw=1.4, marker="s", ms=3,
              label=r"$p_{\rm ref}$-mass on $A_\tau^{\rm naive}$ (no-error HDR)")
    ax3.axhline(res["level"], color="0.0", lw=1.0, ls="--", label=f"target = {res['level']}")
    ax3.set_xlabel(r"$\tau$"); ax3.set_ylabel("mass on chosen set")
    ax3.set_ylim(0, 1.05)
    ax3.set_title(r"(B) set-mass containment under $p_{\rm ref}$  "
                   "(red curve must stay above target line)", fontsize=10)
    ax3.legend(loc="lower right", fontsize=9, frameon=False)
    ax3.grid(True, alpha=0.25)

    fig.suptitle(
        f"YCD stability checks for theorem-respecting 95% bridge tube\n"
        f"{z_a} → {z_b},  T = {res['T']},  ρ = {res['rho']:.4g}",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=200); plt.close(fig)


def write_report(res, a, b, zones, outpath):
    margin = (res["PT_ab_hat"] - res["E_T"]) / max(res["PT_ab_hat"], EPS)
    set_min_thm = float(res["set_mass_ref_thm"].min())
    set_min_naive = float(res["set_mass_ref_naive"].min())
    z_a = f"z{zones[a]}" if zones else str(a)
    z_b = f"z{zones[b]}" if zones else str(b)
    txt = f"""# YCD stability report — theorem-respecting 95% bridge tube

Pair                   : state {a} ({z_a}) -> state {b} ({z_b})
Horizon T              : {res['T']}
rho                    : {res['rho']:.6g}
target level           : {res['level']}
state space size       : {res['n']}

==============================================================
(C) HYPOTHESIS MARGIN
==============================================================
  hat_P_T(a,b)         : {res['PT_ab_hat']:.6g}
  E_T = T * rho        : {res['E_T']:.6g}
  margin (relative)    : {margin:.2%}
  hypothesis OK?       : {'YES' if res['PT_ab_hat'] > res['E_T'] else 'NO -> bound is +inf'}

==============================================================
(A) POINTWISE BOUND CONTAINMENT
==============================================================
  in-band fraction     : {res['fraction_in_band']:.4f}
                         ({res['n_in_band']} of {res['n_total']} grid cells)
  max empirical gap    : {res['max_gap']:.6g}
  max theorem bound    : {res['max_eps']:.6g}
  bound dominates gap? : {'YES (everywhere)' if res['fraction_in_band'] >= 0.999 else
                          f'PARTIALLY ({(1-res["fraction_in_band"])*100:.1f}% violations)'}

==============================================================
(B) SET-MASS COVERAGE UNDER p_ref
==============================================================
  min p_ref-mass on A_tau (theorem-respecting) : {set_min_thm:.4f}
  min p_ref-mass on A_tau (no-error HDR)       : {set_min_naive:.4f}
  theorem set covers >= {res['level']} everywhere? :
        {'YES' if set_min_thm >= res['level'] else f'NO (worst tau drops to {set_min_thm:.3f})'}
  no-error set covers >= {res['level']} everywhere? :
        {'YES' if set_min_naive >= res['level'] else f'NO (worst tau drops to {set_min_naive:.3f})'}
"""
    outpath.write_text(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ycd3_output")
    ap.add_argument("--generator", default="Q_em.csv")
    ap.add_argument("--reference_generator", default="Q_emp.csv")
    ap.add_argument("--zones", default="zone_ids.csv")
    ap.add_argument("--a", type=int, default=9)
    ap.add_argument("--b", type=int, default=7)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--n_tau", type=int, default=121)
    ap.add_argument("--rho", type=float, default=0.001)
    ap.add_argument("--level", type=float, default=0.95)
    args = ap.parse_args()

    out = Path(args.out)
    Qhat = pd.read_csv(out / args.generator).to_numpy()
    Qref = pd.read_csv(out / args.reference_generator).to_numpy()
    zones = pd.read_csv(out / args.zones)["zone_id"].tolist() if (out / args.zones).exists() else None

    rho_HS = float(norm(Qref - Qhat, "fro"))
    rho_2 = float(norm(Qref - Qhat, 2))
    rho_inf = float(np.max(np.sum(np.abs(Qref - Qhat), axis=1)))
    print(f"=== YCD stability check ===")
    print(f"State space n = {Qhat.shape[0]}")
    print(f"Data norms: ||..||_HS={rho_HS:.4f}  ||..||_2={rho_2:.4f}  ||..||_inf={rho_inf:.4f}")
    print(f"rho used = {args.rho:.6f}    level = {args.level}")
    z_a = f"z{zones[args.a]}" if zones else str(args.a)
    z_b = f"z{zones[args.b]}" if zones else str(args.b)
    print(f"Pair: a={args.a} ({z_a}) -> b={args.b} ({z_b}), T={args.T}")

    res = compute_stability(Qhat, Qref, args.a, args.b, args.T, args.n_tau,
                              args.rho, args.level)

    bout = out / "petal_3d" / "stability"
    figdir = bout / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    p1 = figdir / f"stability_pointwise_a{args.a}_b{args.b}_T{args.T}.png"
    p2 = figdir / f"stability_set_a{args.a}_b{args.b}_T{args.T}.png"
    p3 = figdir / f"stability_summary_a{args.a}_b{args.b}_T{args.T}.png"
    figure_pointwise(res, args.a, args.b, zones, p1)
    figure_set(res, args.a, args.b, zones, p2)
    figure_summary(res, args.a, args.b, zones, p3)
    write_report(res, args.a, args.b, zones,
                  bout / f"stability_report_a{args.a}_b{args.b}_T{args.T}.txt")

    margin = (res["PT_ab_hat"] - res["E_T"]) / max(res["PT_ab_hat"], EPS)
    print(f"\n--- Verdict ---")
    print(f"(C) hypothesis margin     : {margin:.2%}  "
          f"({'OK' if margin > 0 else 'VIOLATED'})")
    print(f"(A) in-band fraction      : {res['fraction_in_band']:.4f} "
          f"({res['n_in_band']}/{res['n_total']} (tau, c) cells)")
    print(f"(A) max gap : max bound   : {res['max_gap']:.5g} : {res['max_eps']:.5g}")
    print(f"(B) min p_ref-mass on A_τ : {res['set_mass_ref_thm'].min():.4f}  "
          f"(target >= {args.level})")
    print(f"\nFigures -> {figdir}")
    print(f"Report  -> {bout / f'stability_report_a{args.a}_b{args.b}_T{args.T}.txt'}")


if __name__ == "__main__":
    main()
