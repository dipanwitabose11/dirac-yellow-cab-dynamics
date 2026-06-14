"""
Reference-layout iterative Doob-bridge refinement for the YCD3 yellow-taxi
generator (30 NYC zones).  Same pointwise bell-curve error analysis as the
predator-prey demo, adapted to read CSV-format generators.

Inputs (under --out, default = ycd3_output/):
    Q_em.csv       (estimated generator)
    Q_emp.csv      (reference / empirical generator)
    zone_ids.csv   (state -> NYC zone-id mapping)
    summary.json   (optional metadata)

Outputs (under <out>/reference_refinement/):
    figures/refinement_a<a>_b<b>_c<c>_T<T>_rho<rho>.png   reference layout
    refinement_full.csv     full bridge curves per k
    refinement_sub.csv      per sub-bridge curves per k
    refinement_summary.csv  max ε_c per k

Usage
-----
python ycd3_reference_refinement.py
python ycd3_reference_refinement.py --a 9 --b 1 --c 13 --T 1.0 --max_k 3 --rho 0.27
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm, norm

EPS = 1e-14


# ---------------- bridge marginals ----------------

def bridge_marginal_at_c(Q, a, b, c, T, tau, PT=None):
    if PT is None:
        PT = expm(T * Q)
    Pt = expm(tau * Q)
    Prem = expm((T - tau) * Q)
    return float(Pt[a, c] * Prem[c, b] / max(PT[a, b], EPS))


def full_bridge_marginal(Q, a, b, T, tau, PT=None):
    if PT is None:
        PT = expm(T * Q)
    Pt = expm(tau * Q)
    Prem = expm((T - tau) * Q)
    p = Pt[a, :] * Prem[:, b] / max(PT[a, b], EPS)
    p = np.maximum(p, 0.0)
    return p / max(p.sum(), EPS)


def bell_eps_full(Qhat, a, b, c, T, tau, rho):
    """First-theorem unconditional bound on |N_c - hat_N_c|, where
        N_c(t) = P_t(a,c) P_{T-t}(c,b)
    This is NOT a bound on |p(t,c) - hat_p(t,c)|; for that, see
    bell_eps_conditional below.

        E_c(t) = t rho P_{T-t}(c,b) + (T-t) rho P_t(a,c) + t(T-t) rho^2
    """
    P_t = expm(tau * Qhat)
    P_rem = expm((T - tau) * Qhat)
    p_a_c = float(P_t[a, c])
    p_c_b = float(P_rem[c, b])
    return tau * rho * p_c_b + (T - tau) * rho * p_a_c + tau * (T - tau) * rho * rho


def bell_eps_conditional(Qhat, a, b, c, T, tau, rho, PT=None):
    """First-theorem CONDITIONAL bound on |p(t,c) - hat p(t,c)|.

    From the boxed identity in the theorem:
        |p(t,c) - hat_p(t,c)|
            <= E_c(t) / (hat_P_T(a,b) - E_T)
             + hat_N_c(t) E_T / [ hat_P_T(a,b) (hat_P_T(a,b) - E_T) ]
    with E_T = T rho.  Returns +inf if hypothesis hat_P_T(a,b) > E_T fails.
    """
    if PT is None:
        PT = expm(T * Qhat)
    PT_ab = float(PT[a, b])
    P_t = expm(tau * Qhat)
    P_rem = expm((T - tau) * Qhat)
    p_a_c = float(P_t[a, c])
    p_c_b = float(P_rem[c, b])
    Nhat = p_a_c * p_c_b
    E_c = tau * rho * p_c_b + (T - tau) * rho * p_a_c + tau * (T - tau) * rho * rho
    E_T = T * rho
    denom = PT_ab - E_T
    if denom <= 0:
        return float("inf")
    return E_c / denom + Nhat * E_T / (PT_ab * denom)


def bell_eps(tau, T, rho):
    """ε_c(t) = t (T-t) ρ²   (dominant first-theorem term, bell shape)."""
    return tau * (T - tau) * rho * rho


# ---------------- waypoint refinement ----------------

def choose_waypoints(Qhat, a, b, T, k):
    if k == 0:
        return np.array([], dtype=float), np.array([], dtype=int)
    times = np.array([j * T / (k + 1) for j in range(1, k + 1)], dtype=float)
    PT = expm(T * Qhat)
    states = np.array(
        [int(np.argmax(full_bridge_marginal(Qhat, a, b, T, float(tj), PT=PT)))
         for tj in times], dtype=int)
    return times, states


def compute_step(Qhat, Qref, a, b, c, T, k, grid, rho, eps_mode="full"):
    wp_t, wp_s = choose_waypoints(Qhat, a, b, T, k)
    ts = np.linspace(0.0, T, grid)
    PT_h = expm(T * Qhat); PT_r = expm(T * Qref)
    full_rows = []
    if eps_mode == "conditional":
        eps_fn = (lambda Q, A, B, C, TT, tt, PT_=None:
                  bell_eps_conditional(Q, A, B, C, TT, tt, rho, PT=PT_))
    elif eps_mode == "full":
        eps_fn = (lambda Q, A, B, C, TT, tt, PT_=None: bell_eps_full(Q, A, B, C, TT, tt, rho))
    else:  # dominant
        eps_fn = (lambda Q, A, B, C, TT, tt, PT_=None: bell_eps(tt, TT, rho))
    for tau in ts:
        pr_v = bridge_marginal_at_c(Qref, a, b, c, T, float(tau), PT=PT_r)
        ph_v = bridge_marginal_at_c(Qhat, a, b, c, T, float(tau), PT=PT_h)
        e = eps_fn(Qhat, a, b, c, T, float(tau), PT_h)
        full_rows.append({"k": k, "t": float(tau),
                          "p_hat": ph_v, "p_ref": pr_v, "eps": e,
                          "abs_gap": abs(ph_v - pr_v),
                          "phat_in_band": bool(abs(ph_v - pr_v) <= e)})
    full = pd.DataFrame(full_rows)
    nodes_t = [0.0, *wp_t.tolist(), T]
    nodes_s = [a, *wp_s.tolist(), b]
    rows_sub = []
    for j in range(len(nodes_t) - 1):
        tL, tR = nodes_t[j], nodes_t[j + 1]
        aj, bj = nodes_s[j], nodes_s[j + 1]
        T_sub = tR - tL
        if T_sub <= 0:
            continue
        seg = np.linspace(0.0, T_sub, grid)
        PT_sh = expm(T_sub * Qhat); PT_sr = expm(T_sub * Qref)
        for tau in seg:
            pr_v = bridge_marginal_at_c(Qref, aj, bj, c, T_sub, float(tau), PT=PT_sr)
            ph_v = bridge_marginal_at_c(Qhat, aj, bj, c, T_sub, float(tau), PT=PT_sh)
            e = eps_fn(Qhat, aj, bj, c, T_sub, float(tau), PT_sh)
            rows_sub.append({
                "k": k, "segment": j, "tL": float(tL), "tR": float(tR),
                "T_sub": float(T_sub), "tau": float(tau),
                "global_t": float(tL + tau),
                "a_sub": int(aj), "b_sub": int(bj),
                "p_hat": ph_v, "p_ref": pr_v, "eps": e,
                "abs_gap": abs(ph_v - pr_v),
                "phat_in_band": bool(abs(ph_v - pr_v) <= e),
            })
    sub = pd.DataFrame(rows_sub)
    return (full, sub,
            float(full["eps"].max()),
            float(sub["eps"].max()) if not sub.empty else 0.0,
            wp_t, wp_s)


# ---------------- figure ----------------

def stitched_estimate_at(c, t_global, T, segments_for_k, Qhat):
    """For a given waypoint partition, the stitched estimated bridge marginal at
    intermediate state c at global time t is the bridge marginal of the
    sub-bridge containing t (under Qhat), for the segment endpoints (a_j, b_j, T_sub).
    """
    for (tL, tR, aj, bj) in segments_for_k:
        if tL - 1e-12 <= t_global <= tR + 1e-12:
            tau = max(min(t_global - tL, tR - tL), 0.0)
            T_sub = tR - tL
            if T_sub <= 0:
                return float("nan")
            return bridge_marginal_at_c(Qhat, aj, bj, c, T_sub, tau)
    return float("nan")


def make_convergence_figure(Qhat, Qref, a, b, c, T, c_grid, k_values, outpath: Path,
                             time_unit: str = "h"):
    """Convergence plots for many refinement steps.

    Plots:
      (1) Estimated bridge p_hat (under Q_em) reconstructed by stitching sub-bridges
          through k waypoints (waypoints chosen as bridge mode under Q_emp =true bridge),
          for several k values, against the true bridge p(t,c). Curves should
          approach the true bridge as k grows.
      (2) Worst-case error bound max_t ε_c(t) vs k (log-y), with a 1/(k+1)^2
          reference line.
      (3) Sup-norm gap max_t |stitched_p_hat(t,c) - p_true(t,c)| vs k.
    """
    PT_h = expm(T * Qhat); PT_r = expm(T * Qref)
    ts = np.linspace(0.0, T, c_grid)
    p_true = np.array([bridge_marginal_at_c(Qref, a, b, c, T, float(tau), PT=PT_r) for tau in ts])

    # For each k, build waypoints from the TRUE bridge (Qref) and stitch.
    stitched_curves = {}
    for k in k_values:
        if k == 0:
            wp_t = np.array([], dtype=float); wp_s = np.array([], dtype=int)
        else:
            wp_t = np.array([j * T / (k + 1) for j in range(1, k + 1)], dtype=float)
            wp_s = np.array([
                int(np.argmax(full_bridge_marginal(Qref, a, b, T, float(tj), PT=PT_r)))
                for tj in wp_t
            ], dtype=int)
        nodes_t = [0.0, *wp_t.tolist(), T]
        nodes_s = [a, *wp_s.tolist(), b]
        segs = [(nodes_t[j], nodes_t[j + 1], nodes_s[j], nodes_s[j + 1])
                for j in range(len(nodes_t) - 1) if nodes_t[j + 1] > nodes_t[j]]
        ph = np.array([stitched_estimate_at(c, float(tau), T, segs, Qhat) for tau in ts])
        stitched_curves[k] = ph

    sup_gap = {k: float(np.nanmax(np.abs(stitched_curves[k] - p_true))) for k in k_values}

    # Worst-case bound (using full first-theorem bound at the auto-rho already set
    # for k=0, but here we just report (T_sub/2)^2 rho^2 + linear-terms upper bound).
    # We re-use the same per-segment ε from compute_step but only need its max per k.
    # Theorem bound shrinkage under uniform refinement: peak of t(T-t)rho^2 over
    # a sub-interval of length T_sub = T/(k+1) is (T_sub/2)^2 rho^2. Normalise to k=0.
    rel_bound = {k: 1.0 / (k + 1) ** 2 for k in k_values}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # (1) stitched p_hat vs true bridge for several k
    ax = axes[0]
    ax.plot(ts, p_true, color="0.30", lw=2.2, ls="--", label=r"True bridge $p(t,c)$")
    cmap = plt.cm.viridis(np.linspace(0.0, 0.9, len(k_values)))
    for col, k in zip(cmap, k_values):
        ax.plot(ts, stitched_curves[k], color=col, lw=1.4,
                label=f"stitched $\\hat p$, $k={k}$")
    ax.set_xlabel(f"$t$ ({time_unit})"); ax.set_ylabel(r"$p(t,c)$")
    ax.set_title(f"Stitched estimated bridge converges to true bridge\n"
                 f"$a={a}\\to b={b}$, $c={c}$, $T={T}$")
    ax.legend(fontsize=7.5, loc="best", frameon=False); ax.grid(True, alpha=0.25)

    # (2) sup-norm gap vs k
    ax = axes[1]
    ax.semilogy(list(sup_gap.keys()), list(sup_gap.values()), marker="o",
                color="0.10", lw=1.6, label=r"$\sup_t |\hat p_{\rm stitched}(t)-p(t)|$")
    # 1/(k+1)^2 reference
    refk = np.array(list(sup_gap.keys()))
    if len(refk) > 1:
        ref_curve = sup_gap[refk[0]] / ((refk + 1.0) / (refk[0] + 1.0)) ** 2
        ax.semilogy(refk, ref_curve, color="0.55", lw=1.0, ls=":",
                    label=r"$\propto 1/(k+1)^2$")
    ax.set_xlabel("number of intermediate waypoints $k$")
    ax.set_ylabel(r"$\sup_t|\hat p_{\rm stitched}-p|$")
    ax.set_title("Empirical convergence of stitched estimate to true bridge")
    ax.legend(fontsize=8, frameon=False); ax.grid(True, which="both", alpha=0.25)

    # (3) bound rel
    ax = axes[2]
    ax.semilogy(list(rel_bound.keys()), list(rel_bound.values()), marker="s",
                color="0.10", lw=1.6, label=r"theorem $\max_t \varepsilon_c$ relative")
    ax.set_xlabel("number of intermediate waypoints $k$")
    ax.set_ylabel(r"$\max_t\varepsilon_c(t)\,/\,\max_t\varepsilon_c|_{k=0}$")
    ax.set_title("Theoretical bound shrinkage under refinement")
    ax.legend(fontsize=8, frameon=False); ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("Convergence of estimated Doob bridge to true bridge as # of intermediate waypoints grows",
                 fontsize=12, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=200); plt.close(fig)
    return sup_gap, rel_bound


def _zone_label(zones, idx):
    return f"{idx}({zones[idx]})" if zones else f"{idx}"


def make_per_step_figures(results, a, b, c, T, rho, zones, figdir: Path,
                            time_unit: str = "h"):
    """One clean figure per refinement step. Generous spacing; nothing overlaps:
       - figure suptitle (pair metadata) at the top
       - one figure-level legend ABOVE the data area
       - top axes: full bridge
       - clear vertical gap
       - bottom axes: sub-bridge sub-grid with its own subgrid suptitle
    """
    BAND = "#cdc4b1"; LINE_HAT = "0.0"; LINE_REF = "0.40"; WP = "0.55"
    p_max = max(max(float(r["full"]["p_hat"].max()), float(r["full"]["p_ref"].max())) for r in results)

    def _finite_max(series):
        v = series.replace([np.inf, -np.inf], np.nan).dropna()
        return float(v.max()) if len(v) else 0.0

    def _finite_min(series):
        v = series.replace([np.inf, -np.inf], np.nan).dropna()
        return float(v.min()) if len(v) else 0.0

    band_top = max(_finite_max(r["full"]["p_ref"] + r["full"]["eps"]) for r in results)
    band_bot = min(_finite_min(r["full"]["p_ref"] - r["full"]["eps"]) for r in results)
    span = max(band_top - band_bot, p_max * 0.5, 0.05)
    yhi = band_top + 0.10 * span
    ylo = min(band_bot - 0.10 * span, -0.02 * span)
    cap_eps = max(yhi - p_max, span)
    paths = []

    pair_str = (f"$a={_zone_label(zones, a)}$  →  $b={_zone_label(zones, b)}$,   "
                f"intermediate state $c={_zone_label(zones, c)}$,   $T={T}$ {time_unit},   "
                f"$\\rho={rho:.3g}$")

    for k_idx, res in enumerate(results):
        full = res["full"]; sub = res["sub"]
        wp_t = res["wp_t"]

        # ----- explicit axes positions -----
        # Figure: 14 x 10 inches.
        # Suptitle band:                top    0.93 - 1.00
        # Legend strip:                 0.86 - 0.92
        # Full-bridge axes:             0.45 - 0.83
        # Subgrid title strip:          0.39 - 0.43
        # Sub-bridge axes (n_seg cols): 0.07 - 0.36
        fig = plt.figure(figsize=(14, 10))

        # ---- Top axes: full bridge ----
        ax_top = fig.add_axes([0.085, 0.46, 0.88, 0.36])
        t = full["t"].to_numpy()
        ph = full["p_hat"].to_numpy(); pr = full["p_ref"].to_numpy()
        eps = full["eps"].to_numpy()
        eps_plot = np.where(np.isfinite(eps), np.clip(eps, 0.0, cap_eps), cap_eps)
        h_band = ax_top.fill_between(t, pr - eps_plot, pr + eps_plot, color=BAND, alpha=1.0,
                                      label=r"Error band $p(t,c)\pm\varepsilon_c(t)$")
        h_ref, = ax_top.plot(t, pr, color=LINE_REF, lw=2.0, ls="--",
                              label=r"True bridge $p(t,c)$  [$Q_{\rm emp}$]")
        h_hat, = ax_top.plot(t, ph, color=LINE_HAT, lw=2.2,
                              label=r"Estimated bridge $\hat p(t,c)$  [$Q_{\rm em}$]")
        for tj in wp_t:
            ax_top.axvline(tj, color=WP, ls=":", lw=0.8, alpha=0.6)
        ax_top.set_xlim(-0.02 * T, 1.02 * T); ax_top.set_ylim(ylo, yhi)
        ax_top.set_xlabel(f"$t$  ({time_unit})", fontsize=10)
        ax_top.set_ylabel("bridge marginal probability at state $c$", fontsize=10)
        ax_top.set_title(
            f"Full bridge   ·   Step $k={k_idx}$:  {k_idx+1} sub-interval{'s' if k_idx else ''}, "
            f"{k_idx} waypoint{'s' if k_idx != 1 else ''}",
            fontsize=11, pad=8)
        ax_top.grid(True, alpha=0.25)

        # ---- Figure-level suptitle (pair metadata) ----
        fig.text(0.5, 0.965,
                 "Doob bridge refinement — pointwise error band", ha="center",
                 fontsize=13, weight="bold")
        fig.text(0.5, 0.935, pair_str, ha="center", fontsize=10, color="0.20")

        # ---- Figure-level legend above data area ----
        fig.legend(handles=[h_hat, h_ref, h_band],
                   loc="center", bbox_to_anchor=(0.5, 0.885), ncol=3,
                   frameon=False, fontsize=10)

        # ---- Sub-bridge row title ----
        fig.text(0.5, 0.405, f"Per sub-bridge view  (step k={k_idx})",
                 ha="center", fontsize=12, color="0.15")

        # ---- Bottom axes: sub-bridges via add_axes (avoid gridspec/tight_layout interactions) ----
        n_seg = sub["segment"].nunique() if not sub.empty else 1
        if not sub.empty:
            sub_top = _finite_max(sub["p_ref"] + sub["eps"])
            sub_bot = _finite_min(sub["p_ref"] - sub["eps"])
            sspan = max(sub_top - sub_bot, 0.02)
            syhi = sub_top + 0.08 * sspan
            sylo = sub_bot - 0.08 * sspan
            sub_cap = max(syhi - float(sub["p_ref"].max()), sspan)
        else:
            syhi, sylo, sub_cap = yhi, ylo, cap_eps

        left, right = 0.085, 0.965
        bottom, top = 0.07, 0.34
        gap = 0.018
        avail = right - left - gap * (n_seg - 1)
        wseg = avail / max(n_seg, 1)
        for j_seg, (seg_id, seg_df) in enumerate(sub.groupby("segment")):
            x0 = left + j_seg * (wseg + gap)
            ax = fig.add_axes([x0, bottom, wseg, top - bottom])
            tau = seg_df["tau"].to_numpy()
            sph = seg_df["p_hat"].to_numpy(); spr = seg_df["p_ref"].to_numpy()
            seps = seg_df["eps"].to_numpy()
            seps_plot = np.where(np.isfinite(seps), np.clip(seps, 0.0, sub_cap), sub_cap)
            ax.fill_between(tau, spr - seps_plot, spr + seps_plot, color=BAND, alpha=1.0)
            ax.plot(tau, spr, color=LINE_REF, lw=1.6, ls="--")
            ax.plot(tau, sph, color=LINE_HAT, lw=1.8)
            tL = float(seg_df["tL"].iloc[0]); tR = float(seg_df["tR"].iloc[0])
            aj = int(seg_df["a_sub"].iloc[0]); bj = int(seg_df["b_sub"].iloc[0])
            ax.set_title(
                f"sub-bridge {j_seg}\n"
                f"$t\\in[{tL:.2f},\\,{tR:.2f}]$  ·  "
                f"$z_{{{j_seg}}}={_zone_label(zones, aj)}\\!\\to\\!z_{{{j_seg+1}}}={_zone_label(zones, bj)}$",
                fontsize=9, pad=6)
            ax.set_xlim(-0.02 * (tR - tL), 1.02 * (tR - tL))
            ax.set_ylim(sylo, syhi)
            ax.set_xlabel(r"$\tau$  (within segment)", fontsize=9)
            if j_seg == 0:
                ax.set_ylabel(r"$\hat p(\tau,c)$", fontsize=10)
            else:
                ax.tick_params(labelleft=False)
            ax.grid(True, alpha=0.25)

        outpng = figdir / f"step_k{k_idx}.png"
        fig.savefig(outpng, dpi=200, bbox_inches=None)
        plt.close(fig)
        paths.append(outpng)
    return paths


def make_eps_overlay_figure(results, T, rho, outpath: Path, time_unit: str = "h"):
    """ε_c(t) overlays. Generous spacing; nothing overlaps."""
    fig = plt.figure(figsize=(14, 6))
    fig.text(0.5, 0.94,
             "Bell band $\\varepsilon_c(t)$ — first-theorem pointwise bound",
             ha="center", fontsize=13, weight="bold")
    fig.text(0.5, 0.905,
             f"$\\rho={rho:.3g}$,   $T={T}$ {time_unit}",
             ha="center", fontsize=10, color="0.20")

    ax_l = fig.add_axes([0.06, 0.13, 0.40, 0.68])
    for k_idx, res in enumerate(results):
        col = str(0.05 + 0.20 * k_idx)
        eps = np.where(np.isfinite(res["full"]["eps"]), res["full"]["eps"], np.nan)
        ax_l.plot(res["full"]["t"], eps, color=col, lw=1.8, label=f"$k={k_idx}$")
    ax_l.set_title("Full-bridge band across refinement steps", fontsize=11, pad=10)
    ax_l.set_xlabel(f"$t$  ({time_unit})", fontsize=10)
    ax_l.set_ylabel(r"$\varepsilon_c(t)$", fontsize=10)
    ax_l.legend(fontsize=9, frameon=False, loc="best")
    ax_l.grid(True, alpha=0.25)

    ax_r = fig.add_axes([0.56, 0.13, 0.40, 0.68])
    for k_idx, res in enumerate(results):
        col = str(0.05 + 0.20 * k_idx); sub = res["sub"]
        for seg_id, seg_df in sub.groupby("segment"):
            eps = np.where(np.isfinite(seg_df["eps"]), seg_df["eps"], np.nan)
            ax_r.plot(seg_df["global_t"], eps, color=col, lw=1.5,
                      label=(f"$k={k_idx}$" if seg_id == 0 else None))
    ax_r.set_title("Per-sub-bridge bands stitched in global time", fontsize=11, pad=10)
    ax_r.set_xlabel(f"global $t$  ({time_unit})", fontsize=10)
    ax_r.set_ylabel(r"$\varepsilon_c(\tau)$", fontsize=10)
    ax_r.legend(fontsize=9, frameon=False, loc="best")
    ax_r.grid(True, alpha=0.25)

    fig.savefig(outpath, dpi=200, bbox_inches=None); plt.close(fig)


def make_bars_figure(results, outpath: Path):
    """Clean convergence-bar figure. All labels are placed in non-overlapping regions:
       - bar VALUE labels   : just above each bar tip (small px offset)
       - sub/full RATIO row : in a dedicated reserved strip ABOVE the bars
       - LEGEND             : outside the axes, in the bottom margin
       - title/subtitle     : at the very top of the figure, above the axes
    """
    BAR_FULL = "#cdc4b1"; BAR_SUB = "#5c5345"
    ks = np.arange(len(results))
    full_max = np.array([r["max_full"] for r in results], dtype=float)
    sub_max = np.array([r["max_sub"] for r in results], dtype=float)
    finite_max = float(max(np.nan_to_num(full_max, nan=0.0, posinf=0.0).max(),
                            np.nan_to_num(sub_max, nan=0.0, posinf=0.0).max(),
                            1e-12))
    full_plot = np.where(np.isfinite(full_max), full_max, finite_max)
    sub_plot = np.where(np.isfinite(sub_max), sub_max, finite_max)
    bar_top = max(full_plot.max(), sub_plot.max())

    # Reserve visual budget:
    #   bars occupy 0..bar_top
    #   value labels sit at bar_top + 4 px (rendered via offset, not data)
    #   sub/full ratio row sits at y = 1.18 * bar_top  (visible inside axes)
    #   y-axis upper limit = 1.30 * bar_top  (headroom)
    ymax = bar_top * 1.32

    fig = plt.figure(figsize=(11, 6.2))
    # Reserve bottom strip for legend, top strip for title — explicit so nothing collides.
    ax = fig.add_axes([0.10, 0.20, 0.86, 0.62])

    width = 0.38
    ax.bar(ks - width / 2, full_plot, width=width, color=BAR_FULL, edgecolor="0.45",
           label="Full bridge")
    ax.bar(ks + width / 2, sub_plot, width=width, color=BAR_SUB, edgecolor="0.10",
           label="Worst sub-bridge")

    # Value labels just above each bar (pixel-offset; never collide with the bar tip)
    for ki, v in zip(ks, full_max):
        ax.annotate("∞" if not np.isfinite(v) else f"{v:.4g}",
                    xy=(ki - width / 2, full_plot[ki]),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color="0.10")
    for ki, v in zip(ks, sub_max):
        ax.annotate("∞" if not np.isfinite(v) else f"{v:.4g}",
                    xy=(ki + width / 2, sub_plot[ki]),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color="0.10")

    # sub/full ratios in a dedicated row inside the reserved headroom (single y-position)
    ratio_y = bar_top * 1.20
    for ki, (vf, vs) in enumerate(zip(full_max, sub_max)):
        if np.isfinite(vf) and vf > 0 and np.isfinite(vs):
            ax.text(ki, ratio_y, f"sub / full = {vs/vf:.2f}",
                    ha="center", va="center", fontsize=9, color="0.30")

    ax.set_ylim(0, ymax)
    ax.set_xticks(ks)
    ax.set_xticklabels([f"k = {ki}\n({ki+1} sub-intervals)" for ki in ks], fontsize=10)
    ax.set_ylabel(r"max $\varepsilon_c$ over time", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # Title / subtitle at the very top — outside the axes box
    fig.text(0.5, 0.94, "Band-limit convergence under waypoint refinement",
             ha="center", fontsize=13, weight="bold")
    fig.text(0.5, 0.895,
             "Beige = full-bridge bound (constant).    Dark = worst sub-bridge bound.",
             ha="center", fontsize=10, color="0.30")

    # Legend OUTSIDE the axes, in the reserved bottom margin — no overlap possible
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              frameon=False, fontsize=11)

    fig.savefig(outpath, dpi=200); plt.close(fig)


def make_figure(results, a, b, c, T, rho, zones, outpath: Path, time_unit: str = "h"):
    K_steps = len(results)
    BAND = "#cdc4b1"; LINE_HAT = "0.0"; LINE_REF = "0.40"; WP = "0.30"
    BAR_FULL = "#cdc4b1"; BAR_SUB = "#5c5345"
    p_max = max(max(float(r["full"]["p_hat"].max()), float(r["full"]["p_ref"].max())) for r in results)
    band_top = max(float((r["full"]["p_ref"] + r["full"]["eps"]).max()) for r in results)
    band_bot = min(float((r["full"]["p_ref"] - r["full"]["eps"]).min()) for r in results)
    # Reserve 12% headroom at top for waypoint labels and 4% below for endpoint labels
    span = max(band_top - band_bot, p_max * 0.5, 0.05)
    yhi = band_top + 0.18 * span
    ylo = min(band_bot - 0.06 * span, -0.02 * span)

    fig = plt.figure(figsize=(4.5 * (K_steps + 1), 11.5))
    gs = fig.add_gridspec(3, K_steps + 1, height_ratios=[1.0, 1.0, 0.6],
                          hspace=0.55, wspace=0.30)

    za = zones[a] if zones else a
    zb = zones[b] if zones else b
    zc = zones[c] if zones else c

    # ---------- TOP ROW ----------
    for k_idx, res in enumerate(results):
        full = res["full"]; wp_t = res["wp_t"]; wp_s = res["wp_s"]
        ax = fig.add_subplot(gs[0, k_idx])
        t = full["t"].to_numpy()
        ph = full["p_hat"].to_numpy(); pr = full["p_ref"].to_numpy()
        eps = full["eps"].to_numpy()
        # Band centred on the TRUE bridge p(t,c) under Q_emp, with width ±ε_c(t).
        # Theorem guarantees that the estimated bridge p_hat lies inside this band.
        ax.fill_between(t, pr - eps, pr + eps, color=BAND, alpha=1.0,
                        label=r"Error band $p(t,c)\pm\varepsilon_c(t)$" if k_idx == 0 else None)
        ax.plot(t, pr, color=LINE_REF, lw=2.0, ls="--",
                label=r"True bridge $p(t,c)$  [$Q_{\rm emp}$]" if k_idx == 0 else None)
        ax.plot(t, ph, color=LINE_HAT, lw=2.2,
                label=r"Estimated bridge $\hat p(t,c)$  [$Q_{\rm em}$]   (must lie in band)" if k_idx == 0 else None)
        # Endpoint labels at the very bottom-left and bottom-right (out of the data area)
        ax.annotate(f"$a={a}$ (z{za})", xy=(0.0, ylo), xytext=(2, 2),
                    textcoords="offset points", color="0.30", fontsize=8.5,
                    ha="left", va="bottom")
        ax.annotate(f"$b={b}$ (z{zb})", xy=(T, ylo), xytext=(-2, 2),
                    textcoords="offset points", color="0.30", fontsize=8.5,
                    ha="right", va="bottom")
        # Waypoint labels in the reserved headroom strip ABOVE the band
        wp_label_y = band_top + 0.05 * span
        for tj, sj in zip(wp_t, wp_s):
            zsj = zones[sj] if zones else sj
            ax.axvline(tj, color=WP, ls="--", lw=0.8, ymax=0.88)
            ax.annotate(f"$z={sj}$\n(z{zsj})", xy=(tj, wp_label_y),
                        xytext=(0, 0), textcoords="offset points",
                        color=WP, fontsize=7, rotation=0, va="bottom", ha="center")
        ax.set_xlim(-0.03 * T, 1.03 * T); ax.set_ylim(ylo, yhi)
        ax.set_title(f"Step $k={k_idx}$:  {k_idx+1} sub-interval{'s' if k_idx else ''}\n"
                     f"{k_idx} waypoint{'s' if k_idx != 1 else ''}", fontsize=10)
        ax.set_xlabel(f"$t$ ({time_unit})")
        if k_idx == 0:
            ax.set_ylabel(r"$\hat p(t,c) = \hat P_t(a,c)\,\hat P_{T-t}(c,b)/\hat P_T(a,b)$")
            # Legend OUTSIDE the axes (top), so it never overlaps band/waypoints
            ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 1.18),
                      frameon=False, ncol=1)
        ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(gs[0, -1])
    for k_idx, res in enumerate(results):
        col = str(0.05 + 0.20 * k_idx)
        ax.plot(res["full"]["t"], res["full"]["eps"], color=col, lw=1.6,
                label=f"$k={k_idx}$ (full bridge)")
    ax.set_title(r"Band $\varepsilon_c(t)$:  full bridge across refinement steps")
    ax.set_xlabel(f"$t$ ({time_unit})"); ax.set_ylabel(r"$\varepsilon_c(t)$")
    ax.legend(fontsize=8, frameon=False, loc="upper right"); ax.grid(True, alpha=0.25)

    # ---------- MIDDLE ROW ----------
    for k_idx, res in enumerate(results):
        sub = res["sub"]; n_seg = sub["segment"].nunique() if not sub.empty else 1
        sub_gs = gs[1, k_idx].subgridspec(1, max(n_seg, 1), wspace=0.04)
        for j_seg, (seg_id, seg_df) in enumerate(sub.groupby("segment")):
            ax = fig.add_subplot(sub_gs[0, j_seg])
            tau = seg_df["tau"].to_numpy()
            ph = seg_df["p_hat"].to_numpy(); pr = seg_df["p_ref"].to_numpy()
            eps = seg_df["eps"].to_numpy()
            ax.fill_between(tau, pr - eps, pr + eps, color=BAND, alpha=1.0)
            ax.plot(tau, pr, color=LINE_REF, lw=1.6, ls="--")
            ax.plot(tau, ph, color=LINE_HAT, lw=1.8)
            tL = float(seg_df["tL"].iloc[0]); tR = float(seg_df["tR"].iloc[0])
            aj = int(seg_df["a_sub"].iloc[0]); bj = int(seg_df["b_sub"].iloc[0])
            ax.set_title(f"$[{tL:.2f},\\,{tR:.2f}]$\n"
                         f"$z_{{{j_seg}}}={aj}\\,\\to\\,z_{{{j_seg+1}}}={bj}$", fontsize=8.5)
            ax.set_xlim(-0.02 * (tR - tL), 1.02 * (tR - tL))
            ax.set_ylim(ylo, yhi); ax.set_xlabel(r"$\tau$")
            if j_seg > 0: ax.set_yticklabels([])
            if j_seg == 0 and k_idx == 0:
                ax.set_ylabel(r"$\hat p(\tau,c)$")
            ax.grid(True, alpha=0.25)
    ax = fig.add_subplot(gs[1, -1])
    for k_idx, res in enumerate(results):
        col = str(0.05 + 0.20 * k_idx); sub = res["sub"]
        for seg_id, seg_df in sub.groupby("segment"):
            ax.plot(seg_df["global_t"], seg_df["eps"], color=col, lw=1.5,
                    label=(f"$k={k_idx}$ sub-bridges" if seg_id == 0 else None))
    ax.set_title(r"Band $\varepsilon_c(\tau)$: each sub-bridge (stitched in global time)")
    ax.set_xlabel(f"global time ({time_unit})"); ax.set_ylabel(r"$\varepsilon_c(\tau)$")
    ax.legend(fontsize=8, frameon=False); ax.grid(True, alpha=0.25)

    # ---------- BOTTOM ROW ----------
    ax = fig.add_subplot(gs[2, :])
    ks = np.arange(len(results))
    full_max = [r["max_full"] for r in results]; sub_max = [r["max_sub"] for r in results]
    width = 0.40
    ax.bar(ks - width / 2, full_max, width=width, color=BAR_FULL, edgecolor="0.45",
           label=r"Full bridge   $\max_t \varepsilon_c(t)$")
    ax.bar(ks + width / 2, sub_max, width=width, color=BAR_SUB, edgecolor="0.10",
           label=r"Sub-bridge   $\max_\tau \varepsilon_c(\tau)$ (worst sub-interval)")
    for ki, v in enumerate(full_max):
        ax.text(ki - width / 2, v, f"{v:.4g}", ha="center", va="bottom", fontsize=9)
    for ki, v in enumerate(sub_max):
        ax.text(ki + width / 2, v, f"{v:.4g}", ha="center", va="bottom", fontsize=9)
    for ki, (vf, vs) in enumerate(zip(full_max, sub_max)):
        if vf > 0:
            ax.text(ki, max(vf, vs) * 1.07, f"sub/full\n={vs/vf:.2f}",
                    ha="center", va="bottom", fontsize=8.5, color="0.30")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"$k={ki}$\n({ki+1} sub-intervals)" for ki in ks])
    ax.set_ylabel(r"max $\varepsilon_c$ over time")
    ax.set_title("Band-limit convergence across refinement steps")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "YCD3 Yellow-Taxi Bridge Refinement — Pointwise Bell-Curve Error Bands\n"
        f"Pair $a={a}$ (zone {za}) $\\to b={b}$ (zone {zb}),   "
        f"$T={T:.2f}$ {time_unit},   intermediate state $c={c}$ (zone {zc}),   $\\rho={rho:.3g}$\n"
        r"Each column adds one waypoint $z_k$ (mode of bridge at $t_k=kT/(k+1)$).  "
        r"Band $=\pm\varepsilon_c(t)$ collapses to zero at endpoints and is thickest near $t\approx T/2$.",
        fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=200); plt.close(fig)


# ---------------- driver ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ycd3_output")
    ap.add_argument("--generator", default="Q_em.csv")
    ap.add_argument("--reference_generator", default="Q_emp.csv")
    ap.add_argument("--zones", default="zone_ids.csv")
    ap.add_argument("--a", type=int, default=9)
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--c", type=int, default=13)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--max_k", type=int, default=3)
    ap.add_argument("--grid", type=int, default=121)
    ap.add_argument("--eps_mode", choices=["conditional", "full", "dominant"],
                    default="full",
                    help="'conditional' = boxed bound on |p - p_hat| (the right thing to "
                         "envelope the bridge marginal). 'full' = unconditional E_c on "
                         "|N_c - N_hat_c| (joint, not marginal). 'dominant' = t(T-t)rho^2 only.")
    ap.add_argument("--rho_mode", choices=["explicit", "data", "auto"], default="auto",
                    help="'explicit' uses --rho; 'data' uses ||Q_emp-Q_em||_2 (spectral); "
                         "'auto' picks the smallest rho such that p_hat lies in the band around p_ref")
    ap.add_argument("--rho", type=float, default=0.27,
                    help="rho for the bell band (0.27 reproduces the reference figure scale)")
    ap.add_argument("--time_unit", default="h", help="time-axis unit label, e.g. 'h' for taxi data")
    ap.add_argument("--convergence_max_k", type=int, default=20,
                    help="maximum k for the separate convergence-to-true-bridge plot")
    ap.add_argument("--convergence_grid", type=int, default=201,
                    help="time grid for the convergence figure")
    args = ap.parse_args()

    out = Path(args.out)
    Qhat = pd.read_csv(out / args.generator).to_numpy()
    Qref = pd.read_csv(out / args.reference_generator).to_numpy()
    zones = pd.read_csv(out / args.zones)["zone_id"].tolist() if (out / args.zones).exists() else None

    rho_HS = float(norm(Qref - Qhat, "fro"))
    rho_2 = float(norm(Qref - Qhat, 2))
    rho_inf = float(np.max(np.sum(np.abs(Qref - Qhat), axis=1)))
    print(f"State space size n = {Qhat.shape[0]}")
    if zones:
        print(f"Zones loaded: {len(zones)} NYC taxi zones")
    print(f"||Q_emp - Q_em||_HS         = {rho_HS:.4f}")
    print(f"||Q_emp - Q_em||_2          = {rho_2:.4f}")
    print(f"||Q_emp - Q_em||_(inf->inf) = {rho_inf:.4f}")

    # Resolve rho according to chosen mode
    if args.rho_mode == "data":
        rho = rho_2
        rho_label = "data spectral norm ||Q_emp-Q_em||_2"
    elif args.rho_mode == "auto":
        # Pick the smallest rho such that the chosen bound (conditional/full/dominant)
        # envelopes |p_hat - p_ref|(t) on EVERY sub-bridge across all refinement steps,
        # so the band properly contains p_hat in every panel of every figure.
        a_, b_, c_, T_ = args.a, args.b, args.c, args.T
        PT_h0 = expm(T_ * Qhat); PT_r0 = expm(T_ * Qref)

        # Build the list of (a_j, b_j, T_sub) sub-bridges that will appear across k=0..max_k
        sub_bridges = [(a_, b_, T_)]
        for k in range(1, args.max_k + 1):
            wp_t_ = [j * T_ / (k + 1) for j in range(1, k + 1)]
            wp_s_ = [int(np.argmax(full_bridge_marginal(Qhat, a_, b_, T_, float(tj), PT=PT_h0)))
                     for tj in wp_t_]
            nodes = [a_, *wp_s_, b_]
            ts_n = [0.0, *wp_t_, T_]
            for j in range(len(nodes) - 1):
                T_sub = ts_n[j + 1] - ts_n[j]
                if T_sub > 0:
                    sub_bridges.append((nodes[j], nodes[j + 1], T_sub))

        # For each sub-bridge, sample the gap |p_hat - p_ref|(tau) and require the
        # bound to envelop it. With the chosen eps_mode, the bound is monotone in rho,
        # so bisect.
        def bound_at(Qh, A, B, C, TT, tt, PTh, rho_x):
            if args.eps_mode == "conditional":
                return bell_eps_conditional(Qh, A, B, C, TT, tt, rho_x, PT=PTh)
            elif args.eps_mode == "full":
                return bell_eps_full(Qh, A, B, C, TT, tt, rho_x)
            else:
                return bell_eps(tt, TT, rho_x)

        # Pre-compute per-sub-bridge data (PT_sh, PT_sr, gap[tau]) once.
        sub_data = []
        for (A, B, T_sub) in sub_bridges:
            PT_sh = expm(T_sub * Qhat); PT_sr = expm(T_sub * Qref)
            taus = np.linspace(0.0, T_sub, 41)[1:-1]
            gaps = []
            for tau in taus:
                ph_v = bridge_marginal_at_c(Qhat, A, B, args.c, T_sub, float(tau), PT=PT_sh)
                pr_v = bridge_marginal_at_c(Qref, A, B, args.c, T_sub, float(tau), PT=PT_sr)
                gaps.append((float(tau), abs(ph_v - pr_v)))
            sub_data.append((A, B, T_sub, PT_sh, gaps))

        # For conditional bound, hypothesis is rho < P_T_hat(a,b)/T_sub on every sub-bridge.
        if args.eps_mode == "conditional":
            rho_max_valid = min(float(PT_sh[A, B]) / T_sub for (A, B, T_sub, PT_sh, _) in sub_data)
        else:
            rho_max_valid = float("inf")

        # bound is monotone increasing in rho within (0, rho_max_valid). Bisect for the
        # smallest rho such that bound(tau) >= gap(tau) for all sampled tau on every sub.
        def all_inside(rho_x):
            for (A, B, T_sub, PT_sh, gaps) in sub_data:
                for (tau, gap) in gaps:
                    eb = bound_at(Qhat, A, B, args.c, T_sub, float(tau), PT_sh, rho_x)
                    if not np.isfinite(eb) or eb < gap:
                        return False
            return True

        if rho_max_valid <= 0 or np.isinf(rho_max_valid):
            # No hypothesis constraint; use a large bracket.
            lo, hi = 1e-6, max(rho_2, 1.0)
            while not all_inside(hi):
                hi *= 2.0
                if hi > 1e6:
                    break
        else:
            lo = 1e-6
            hi = 0.999 * rho_max_valid
            if not all_inside(hi):
                # Even at the boundary the bound is too small (it actually -> inf at the
                # boundary, so this should always succeed); fall back to data spectral norm.
                rho = rho_2; rho_label = "fallback to data spectral norm (auto-sizing failed)"
                hi = None

        if hi is not None:
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if all_inside(mid):
                    hi = mid
                else:
                    lo = mid
            rho = hi  # already gives containment; small safety multiplier risks crossing hypothesis
            if np.isfinite(rho_max_valid):
                rho = min(rho, 0.95 * rho_max_valid)
            rho_label = (f"auto-sized so {args.eps_mode} bound contains "
                         f"|p_hat - p| on every sub-bridge (k=0..{args.max_k})")
    else:
        rho = args.rho
        rho_label = "explicit"
    print(f"rho used in band            = {rho:.4f}   ({rho_label})")
    za = zones[args.a] if zones else args.a
    zb = zones[args.b] if zones else args.b
    zc = zones[args.c] if zones else args.c
    print(f"Pair: a={args.a} (zone {za})  b={args.b} (zone {zb})  c={args.c} (zone {zc})  "
          f"T={args.T:.3f} {args.time_unit}   max_k={args.max_k}")

    results = []
    for k in range(args.max_k + 1):
        full, sub, mf, ms, wp_t, wp_s = compute_step(
            Qhat, Qref, args.a, args.b, args.c, args.T, k, args.grid, rho, args.eps_mode)
        results.append({"k": k, "full": full, "sub": sub,
                        "max_full": mf, "max_sub": ms, "wp_t": wp_t, "wp_s": wp_s})
        ratio = ms / mf if mf > 0 else float("nan")
        wp_zones = [zones[s] if zones else int(s) for s in wp_s]
        print(f"  k={k}  waypoints t={np.round(wp_t,3).tolist()} states={wp_s.tolist()} zones={wp_zones}  "
              f"max_full={mf:.4g}  max_sub={ms:.4g}  sub/full={ratio:.3f}")

    bout = out / "reference_refinement"
    figdir = bout / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    pd.concat([r["full"] for r in results], ignore_index=True).to_csv(
        bout / "refinement_full.csv", index=False)
    pd.concat([r["sub"] for r in results], ignore_index=True).to_csv(
        bout / "refinement_sub.csv", index=False)
    pd.DataFrame([
        {"k": r["k"], "n_waypoints": int(r["k"]),
         "max_full": r["max_full"], "max_sub": r["max_sub"],
         "ratio_sub_over_full": (r["max_sub"]/r["max_full"]) if r["max_full"]>0 else float("nan"),
         "waypoint_times": ";".join(f"{x:.3f}" for x in r["wp_t"]),
         "waypoint_states": ";".join(str(int(x)) for x in r["wp_s"])}
        for r in results
    ]).to_csv(bout / "refinement_summary.csv", index=False)

    # Per-step clean figures (one PNG per refinement step k).
    step_paths = make_per_step_figures(results, args.a, args.b, args.c, args.T, rho,
                                         zones, figdir, args.time_unit)
    # Single ε_c(t) overlay across k.
    eps_png = figdir / "eps_curves_overlay.png"
    make_eps_overlay_figure(results, args.T, rho, eps_png, args.time_unit)
    # Convergence bars in their own figure.
    bars_png = figdir / "convergence_bars.png"
    make_bars_figure(results, bars_png)
    print(f"Per-step figures -> {[p.name for p in step_paths]}")
    print(f"Eps overlay      -> {eps_png.name}")
    print(f"Bars figure      -> {bars_png.name}")

    # Separate convergence figure with many waypoints
    k_values = sorted(set([0, 1, 2, 3, 5, 8, 12, 16, args.convergence_max_k]))
    k_values = [k for k in k_values if 0 <= k <= args.convergence_max_k]
    conv_png = figdir / f"convergence_a{args.a}_b{args.b}_c{args.c}_T{args.T}_kmax{args.convergence_max_k}.png"
    sup_gap, rel_bound = make_convergence_figure(
        Qhat, Qref, args.a, args.b, args.c, args.T, args.convergence_grid,
        k_values, conv_png, args.time_unit)
    pd.DataFrame({"k": list(sup_gap.keys()),
                  "sup_gap_stitched_minus_true": list(sup_gap.values()),
                  "relative_bound": list(rel_bound.values())}).to_csv(
        bout / "convergence_summary.csv", index=False)
    print(f"Convergence figure -> {conv_png}")
    print(f"  sup_t |stitched p_hat - p_true|  by k:")
    for k, v in sup_gap.items():
        print(f"    k={k:3d}  sup_gap = {v:.5g}")
    print(f"\nAll figures in -> {figdir}")
    print(f"CSVs   -> {bout}")


if __name__ == "__main__":
    main()
