"""
Petal-style bridge visualization for the YCD3 yellow-taxi generator.

For an endpoint pair (a, b) with horizon T:
  - x-axis : state index, ORDERED by the time tau* at which each state has peak
              bridge marginal probability (so that the "petal" naturally
              pinches at endpoints, with mid-bridge states in the middle).
  - y-axis : time tau in [0, T].
  - grayscale heatmap : bridge marginal p(tau, c) under Qhat (= Q_em).
  - 95% HDR outline   : contour of the smallest set of states capturing 95%
                         mass at each tau.  Union of these sets across time
                         is the joint 95%-containment "petal core".
  - beige overlay     : the same construction enlarged by the first-theorem
                         pointwise band p +/- eps_c(tau) so that the union
                         95% region with theorem error allowance is shown.
                         If you read off the petal at any tau and read the
                         beige region's union over tau, you have a band that
                         contains the bridge with 95% probability under the
                         theorem-respecting error allowance.

Inputs (from --out, default = ycd3_output/):
    Q_em.csv, Q_emp.csv, zone_ids.csv

Outputs (under <out>/petal_visualization/):
    figures/petal_a<a>_b<b>_T<T>.png

Run
---
python ycd3_petal_visualization.py --a 9 --b 1 --T 1.0
python ycd3_petal_visualization.py --a 9 --b 1 --T 1.0 --grid 401
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
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


def bell_eps_vector(Qhat, a, b, T, tau, rho):
    Pt = expm(tau * Qhat)
    Prem = expm((T - tau) * Qhat)
    return tau * rho * Prem[:, b] + (T - tau) * rho * Pt[a, :] + tau * (T - tau) * rho * rho


def auto_rho_for_pair(Qhat, Qref, a, b, T, sample_taus, rho_max=10.0):
    """Pick smallest rho such that bell_eps_vector(rho) >= |hat_p - p| at every state, every sampled tau."""
    PT_h = expm(T * Qhat); PT_r = expm(T * Qref)
    rho_lower = []
    for tau in sample_taus:
        p_hat = bridge_marginal(Qhat, a, b, T, float(tau), PT=PT_h)
        p_ref = bridge_marginal(Qref, a, b, T, float(tau), PT=PT_r)
        gaps = np.abs(p_hat - p_ref)
        Pt = expm(tau * Qhat); Prem = expm((T - tau) * Qhat)
        A1 = tau * Prem[:, b] + (T - tau) * Pt[a, :]
        A2 = tau * (T - tau)
        for c in range(len(gaps)):
            g = float(gaps[c]); a1 = float(A1[c])
            if g <= 0:
                continue
            if A2 > 0:
                disc = a1 * a1 + 4 * A2 * g
                rho_c = (-a1 + np.sqrt(disc)) / (2 * A2)
            else:
                rho_c = g / max(a1, 1e-12)
            rho_lower.append(rho_c)
    return min(max(rho_lower) * 1.05 if rho_lower else 1e-3, rho_max)


def hdr_mask(p, level=0.95):
    """Return boolean mask over states marking the smallest set whose mass is >= level."""
    order = np.argsort(p)[::-1]
    cum = np.cumsum(p[order])
    keep_until = int(np.searchsorted(cum, level)) + 1
    keep_until = min(keep_until, len(p))
    keep = np.zeros_like(p, dtype=bool)
    keep[order[:keep_until]] = True
    return keep


def hdr_mask_with_band(p, eps, level=0.95):
    """HDR with theorem error allowance: smallest set C such that the
    *worst-case* total mass on C is still at least `level`.

    Theorem says |p(c) - p_true(c)| <= eps(c). To guarantee that the TRUE
    distribution puts >= `level` mass on C (no matter which valid Q),
    we need
            sum_{c in C} (p(c) - eps(c)) >= level.
    Greedily include states in decreasing order of p(c) until that lower
    bound is met. This always grows the set vs the no-band HDR (or stays
    the same when eps = 0).
    """
    order = np.argsort(p)[::-1]
    cum_lower = 0.0
    keep = np.zeros_like(p, dtype=bool)
    for idx in order:
        keep[idx] = True
        cum_lower += max(float(p[idx] - eps[idx]), 0.0)
        if cum_lower >= level:
            break
    return keep


def order_states_by_peak_time(prob_matrix):
    """Order states by argmax_tau p(tau, c). States with early peaks appear first."""
    n_tau, n = prob_matrix.shape
    peak_t = np.argmax(prob_matrix, axis=0)
    return np.argsort(peak_t)


def _effective_N(prob_matrix, eps_matrix):
    """Raw inverse-Simpson effective-N spread per tau, for core and halo."""
    p_safe = np.maximum(prob_matrix, EPS)
    eff_N_core = 1.0 / np.sum(p_safe ** 2, axis=1)
    p_plus = np.clip(prob_matrix + eps_matrix, 0.0, 1.0)
    p_plus_norm = p_plus / np.maximum(p_plus.sum(axis=1, keepdims=True), EPS)
    eff_N_halo = 1.0 / np.sum(np.maximum(p_plus_norm, EPS) ** 2, axis=1)
    peak = prob_matrix.max(axis=1)
    return eff_N_core, eff_N_halo, peak


def _petal_widths(prob_matrix, eps_matrix, scale=None):
    """Returns (half_core, half_halo, peak_norm). If `scale` is given, use that
    as the half-width-per-effective-N factor (ensures consistent scaling
    across multiple sub-bridges)."""
    eff_N_core, eff_N_halo, peak = _effective_N(prob_matrix, eps_matrix)
    if scale is None:
        scale = 0.5 / max(float(np.max(eff_N_halo)), 1.0)
    peak_norm = peak / max(peak.max(), EPS)
    return eff_N_core * scale, eff_N_halo * scale, peak_norm


def _draw_petal(ax, taus, half_core, half_halo, peak_norm, t_offset=0.0):
    """Render one petal segment on the given axis, anchored at t_offset on y-axis."""
    y = taus + t_offset
    ax.fill_betweenx(y, -half_halo, +half_halo,
                       color=to_rgba("#dccab2", 0.75),
                       edgecolor="#a3946d", lw=0.9, ls="--", zorder=1)
    nx = 301
    x = np.linspace(-0.6, 0.6, nx)
    XX, _ = np.meshgrid(x, taus)
    half_core_grid = np.repeat(half_core[:, None], nx, axis=1)
    peak_grid = np.repeat(peak_norm[:, None], nx, axis=1)
    ramp = np.clip(1.0 - (np.abs(XX) / np.maximum(half_core_grid, EPS)), 0.0, 1.0)
    intensity = (ramp ** 1.4) * (peak_grid ** 0.5)
    GRAY = LinearSegmentedColormap.from_list(
        "pg", [(1, 1, 1, 0), (0.45, 0.45, 0.45, 0.55), (0.05, 0.05, 0.05, 0.95)])
    ax.imshow(intensity, aspect="auto", origin="lower",
                extent=[x[0], x[-1], y[0], y[-1]], cmap=GRAY,
                interpolation="bilinear", zorder=2)
    ax.plot(half_core, y, color="#0b1d2a", lw=0.9, zorder=3)
    ax.plot(-half_core, y, color="#0b1d2a", lw=0.9, zorder=3)


def make_centered_petal_figure(taus, prob_matrix, eps_matrix, hdr_core, hdr_with_band,
                                  a, b, T, zones, rho, outpath: Path):
    """Centred-spine petal + adjacent peak-mass strip.  Minimal in-plot text:
    only the start/end zone labels on the dots themselves.  All other narrative
    in PETAL_LEGEND.txt."""
    half_core, half_halo, peak_norm = _petal_widths(prob_matrix, eps_matrix)

    fig = plt.figure(figsize=(8.5, 10.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[5.5, 1.0], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    axs = fig.add_subplot(gs[0, 1], sharey=ax)

    # ---- petal ----
    _draw_petal(ax, taus, half_core, half_halo, peak_norm, t_offset=0.0)
    ax.axvline(0.0, color="0.55", lw=0.6, ls=":", zorder=2)
    ax.scatter([0.0], [0.0], s=180, color="#138a3f", zorder=10,
                edgecolor="white", linewidth=1.6)
    ax.scatter([0.0], [T], s=180, color="#c7271a", zorder=10,
                edgecolor="white", linewidth=1.6)
    z_a = f"z{zones[a]}" if zones else f"a={a}"
    z_b = f"z{zones[b]}" if zones else f"b={b}"
    ax.annotate(f"start  {z_a}", xy=(0.0, 0.0), xytext=(12, -2),
                  textcoords="offset points", fontsize=9.5, color="#13602e",
                  ha="left", va="top")
    ax.annotate(f"end  {z_b}",   xy=(0.0, T),   xytext=(12, 2),
                  textcoords="offset points", fontsize=9.5, color="#7a1410",
                  ha="left", va="bottom")
    ax.set_xlim(-0.62, 0.62); ax.set_ylim(-0.04 * T, 1.04 * T)
    ax.set_xticks([]); ax.set_ylabel(r"time  $\tau$")
    ax.set_title(f"Doob bridge petal:  {z_a} → {z_b},   $T={T}$,   $\\rho={rho:.4g}$",
                 fontsize=10)

    # ---- adjacent peak-mass-zone strip ----
    peak_state = np.argmax(prob_matrix, axis=1)
    cmap = plt.cm.viridis
    axs.scatter(np.zeros_like(taus), taus, c=peak_state, cmap=cmap, s=18, marker="s")
    runs = []
    cur = int(peak_state[0]); start = float(taus[0])
    for ti in range(1, len(taus)):
        s = int(peak_state[ti])
        if s != cur:
            runs.append((start, float(taus[ti - 1]), cur))
            cur = s; start = float(taus[ti])
    runs.append((start, float(taus[-1]), cur))
    for s_t, e_t, s_idx in runs:
        if e_t - s_t < 0.02 * T:
            continue
        mid = 0.5 * (s_t + e_t)
        lbl = f"z{zones[s_idx]}" if zones else f"c={s_idx}"
        axs.annotate(lbl, xy=(0, mid), xytext=(8, 0),
                       textcoords="offset points", fontsize=8.5, color="0.20",
                       va="center", ha="left")
    axs.set_xlim(-0.5, 1.4); axs.set_xticks([]); axs.set_yticklabels([])
    for sp in ("right", "top", "bottom"):
        axs.spines[sp].set_visible(False)
    axs.set_title("most-likely\nzone", fontsize=8.5, color="0.30")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200); plt.close(fig)


def make_multi_k_petals(Qhat, Qref, a, b, T, c_grid, rho, k_values, zones,
                          outpath: Path, eps_func):
    """Stitched petals across refinement: for each k, insert k waypoints (modes
    of the full bridge under Qhat at evenly-spaced times) and draw the
    *concatenated* sub-bridge petals into a single column. Each k gets its own
    panel; reading left to right shows refinement.

    eps_func(Qhat, a_sub, b_sub, c, T_sub, tau, PT_sub) -> bound at one (state, time).
    """
    PT_h_full = expm(T * Qhat)

    fig, axes = plt.subplots(1, len(k_values), figsize=(2.6 * len(k_values), 9),
                                sharey=True)
    if len(k_values) == 1:
        axes = [axes]

    n_states = Qhat.shape[0]
    for col_i, k in enumerate(k_values):
        ax = axes[col_i]
        # waypoints from Qhat full bridge
        if k == 0:
            wp_t = np.array([], dtype=float); wp_s = np.array([], dtype=int)
        else:
            wp_t = np.array([j * T / (k + 1) for j in range(1, k + 1)], dtype=float)
            wp_s = np.array(
                [int(np.argmax(bridge_marginal(Qhat, a, b, T, float(tj), PT=PT_h_full)))
                 for tj in wp_t], dtype=int)
        nodes_t = [0.0, *wp_t.tolist(), T]
        nodes_s = [a, *wp_s.tolist(), b]

        for j in range(len(nodes_t) - 1):
            tL, tR = nodes_t[j], nodes_t[j + 1]
            aj, bj = nodes_s[j], nodes_s[j + 1]
            T_sub = tR - tL
            if T_sub <= 0:
                continue
            taus_sub = np.linspace(0.0, T_sub, c_grid)
            PT_sh = expm(T_sub * Qhat)
            P_sub = np.zeros((c_grid, n_states))
            E_sub = np.zeros((c_grid, n_states))
            for ti, tau in enumerate(taus_sub):
                P_sub[ti] = bridge_marginal(Qhat, aj, bj, T_sub, float(tau), PT=PT_sh)
                Pt = expm(tau * Qhat); Prem = expm((T_sub - tau) * Qhat)
                E_sub[ti] = (tau * rho * Prem[:, bj] + (T_sub - tau) * rho * Pt[aj, :]
                              + tau * (T_sub - tau) * rho * rho)
            half_core, half_halo, peak_norm = _petal_widths(P_sub, E_sub)
            _draw_petal(ax, taus_sub, half_core, half_halo, peak_norm, t_offset=tL)

        # Spine + waypoints
        ax.axvline(0.0, color="0.55", lw=0.6, ls=":", zorder=2)
        for tj in wp_t:
            ax.axhline(tj, color="0.30", lw=0.5, ls=":", zorder=2)
        # Endpoints
        ax.scatter([0.0], [0.0], s=110, color="#138a3f", zorder=10,
                    edgecolor="white", linewidth=1.2)
        ax.scatter([0.0], [T], s=110, color="#c7271a", zorder=10,
                    edgecolor="white", linewidth=1.2)
        # Waypoint dots on spine
        if k > 0:
            ax.scatter(np.zeros_like(wp_t), wp_t, s=28,
                        color="#1a1a1a", edgecolor="white", linewidth=0.6,
                        zorder=10)
        ax.set_xlim(-0.62, 0.62); ax.set_ylim(-0.04 * T, 1.04 * T)
        ax.set_xticks([])
        ax.set_title(f"$k={k}$  ({k+1} sub-bridge{'s' if k else ''})", fontsize=10)
        if col_i == 0:
            ax.set_ylabel(r"time  $\tau$")

    fig.suptitle(
        f"Petal under refinement — Manhattan zones,  "
        f"a={('z'+str(zones[a])) if zones else a} → b={('z'+str(zones[b])) if zones else b},   "
        f"T={T},   ρ={rho:.4g}",
        fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=200); plt.close(fig)


def make_peak_state_strip(taus, prob_matrix, a, b, T, zones, outpath: Path):
    """Minimal companion: peak-mass zone over time as a coloured vertical strip."""
    peak_state = np.argmax(prob_matrix, axis=1)
    fig, ax = plt.subplots(figsize=(2.5, 10))
    cmap = plt.cm.viridis
    ax.scatter(np.zeros_like(taus), taus, c=peak_state, cmap=cmap,
                 s=20, marker="s")
    # Mark unique runs of peak state with zone-id labels at the segment midpoints
    runs = []
    cur = int(peak_state[0]); start = float(taus[0])
    for ti in range(1, len(taus)):
        s = int(peak_state[ti])
        if s != cur:
            runs.append((start, float(taus[ti - 1]), cur))
            cur = s; start = float(taus[ti])
    runs.append((start, float(taus[-1]), cur))
    for s_t, e_t, s_idx in runs:
        mid = 0.5 * (s_t + e_t)
        lbl = f"z{zones[s_idx]}" if zones else f"c={s_idx}"
        ax.annotate(lbl, xy=(0, mid), xytext=(10, 0),
                     textcoords="offset points", fontsize=9, color="0.20",
                     va="center", ha="left")
    ax.scatter([0], [0], s=130, color="#138a3f", zorder=10,
                edgecolor="white", linewidth=1.4)
    ax.scatter([0], [T], s=130, color="#c7271a", zorder=10,
                edgecolor="white", linewidth=1.4)
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.04 * T, 1.04 * T)
    ax.set_xticks([]); ax.set_yticklabels([])
    for spine in ("right", "top", "bottom"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200); plt.close(fig)


def make_petal_convergence_figure(Qhat, Qref, a, b, T, c_grid, rho, k_values,
                                      zones, outpath: Path):
    """Convergence-to-the-path visualisation.

    Top row     : one petal per k in `k_values`, all sharing the same y-axis.
    Bottom row  : log-scale convergence curves of three metrics vs k:
                    - max petal half-width (over tau)        [theorem-side]
                    - mean petal half-width (over tau)
                    - sup-gap |stitched p_hat - p_true| at c (mid-tau, mid-state)
                  with a 1/(k+1)^2 reference line.
    """
    PT_h_full = expm(T * Qhat); PT_r_full = expm(T * Qref)
    n_states = Qhat.shape[0]

    # ---- compute a GLOBAL width scale from the k=0 full-bridge effective N
    taus_full = np.linspace(0.0, T, c_grid)
    P_full = np.zeros((c_grid, n_states))
    E_full = np.zeros((c_grid, n_states))
    for ti, tau in enumerate(taus_full):
        P_full[ti] = bridge_marginal(Qhat, a, b, T, float(tau), PT=PT_h_full)
        Pt = expm(tau * Qhat); Prem = expm((T - tau) * Qhat)
        E_full[ti] = (tau * rho * Prem[:, b] + (T - tau) * rho * Pt[a, :]
                       + tau * (T - tau) * rho * rho)
    _, eff_N_halo_full, _ = _effective_N(P_full, E_full)
    global_scale = 0.5 / max(float(np.max(eff_N_halo_full)), 1.0)

    K = len(k_values)
    fig = plt.figure(figsize=(2.5 * K + 1.5, 11))
    gs = fig.add_gridspec(2, K, height_ratios=[6.0, 2.4], hspace=0.35, wspace=0.20)

    metrics = []

    for col, k in enumerate(k_values):
        ax = fig.add_subplot(gs[0, col])

        # Waypoints from Qhat full bridge
        if k == 0:
            wp_t = np.array([], dtype=float); wp_s = np.array([], dtype=int)
        else:
            wp_t = np.array([j * T / (k + 1) for j in range(1, k + 1)], dtype=float)
            wp_s = np.array(
                [int(np.argmax(bridge_marginal(Qhat, a, b, T, float(tj), PT=PT_h_full)))
                 for tj in wp_t], dtype=int)
        nodes_t = [0.0, *wp_t.tolist(), T]
        nodes_s = [a, *wp_s.tolist(), b]

        max_half_per_k = 0.0
        sum_half_per_k = 0.0
        n_pts_per_k = 0
        # Full-bridge stitched p_hat at midpoint of full bridge (for sup-gap)
        # — sample the stitched bridge at a fixed evaluation grid.
        eval_taus = np.linspace(0.0, T, c_grid)
        stitched = np.zeros(c_grid)
        for tg_i, tg in enumerate(eval_taus):
            # find segment containing tg
            for j in range(len(nodes_t) - 1):
                if nodes_t[j] - 1e-12 <= tg <= nodes_t[j + 1] + 1e-12:
                    aj, bj = nodes_s[j], nodes_s[j + 1]
                    T_sub = nodes_t[j + 1] - nodes_t[j]
                    if T_sub <= 0:
                        continue
                    tau = max(min(tg - nodes_t[j], T_sub), 0.0)
                    PT_sh = expm(T_sub * Qhat)
                    p = bridge_marginal(Qhat, aj, bj, T_sub, tau, PT=PT_sh)
                    stitched[tg_i] = float(p[a])  # sample at start state for a metric
                    break
        # True bridge marginal at a (under Q_emp) on same eval grid
        truth = np.array([bridge_marginal(Qref, a, b, T, float(tg), PT=PT_r_full)[a]
                           for tg in eval_taus])
        sup_gap = float(np.max(np.abs(stitched - truth)))

        for j in range(len(nodes_t) - 1):
            tL, tR = nodes_t[j], nodes_t[j + 1]
            aj, bj = nodes_s[j], nodes_s[j + 1]
            T_sub = tR - tL
            if T_sub <= 0:
                continue
            taus_sub = np.linspace(0.0, T_sub, c_grid)
            PT_sh = expm(T_sub * Qhat)
            P_sub = np.zeros((c_grid, n_states))
            E_sub = np.zeros((c_grid, n_states))
            for ti, tau in enumerate(taus_sub):
                P_sub[ti] = bridge_marginal(Qhat, aj, bj, T_sub, float(tau), PT=PT_sh)
                Pt = expm(tau * Qhat); Prem = expm((T_sub - tau) * Qhat)
                E_sub[ti] = (tau * rho * Prem[:, bj] + (T_sub - tau) * rho * Pt[aj, :]
                              + tau * (T_sub - tau) * rho * rho)
            half_core, half_halo, peak_norm = _petal_widths(P_sub, E_sub,
                                                              scale=global_scale)
            max_half_per_k = max(max_half_per_k, float(half_halo.max()))
            sum_half_per_k += float(half_halo.sum())
            n_pts_per_k += len(half_halo)
            _draw_petal(ax, taus_sub, half_core, half_halo, peak_norm, t_offset=tL)

        mean_half_per_k = sum_half_per_k / max(n_pts_per_k, 1)

        # Spine + waypoints + endpoints
        ax.axvline(0.0, color="0.55", lw=0.6, ls=":", zorder=2)
        for tj in wp_t:
            ax.axhline(tj, color="0.30", lw=0.4, ls=":", zorder=2)
        ax.scatter([0.0], [0.0], s=80, color="#138a3f", zorder=10,
                    edgecolor="white", linewidth=1.0)
        ax.scatter([0.0], [T], s=80, color="#c7271a", zorder=10,
                    edgecolor="white", linewidth=1.0)
        if k > 0:
            ax.scatter(np.zeros_like(wp_t), wp_t, s=22,
                        color="#1a1a1a", edgecolor="white", linewidth=0.5, zorder=10)

        ax.set_xlim(-0.62, 0.62); ax.set_ylim(-0.04 * T, 1.04 * T)
        ax.set_xticks([])
        if col == 0:
            ax.set_ylabel(r"time  $\tau$")
        else:
            ax.set_yticklabels([])
        ax.set_title(f"$k={k}$", fontsize=11)

        metrics.append({"k": k, "max_half": max_half_per_k,
                         "mean_half": mean_half_per_k, "sup_gap": sup_gap})

    # Convergence curves (bottom row spans all columns)
    ax_curve = fig.add_subplot(gs[1, :])
    ks = np.array([m["k"] for m in metrics])
    max_h = np.array([m["max_half"] for m in metrics])
    mean_h = np.array([m["mean_half"] for m in metrics])
    sup_g = np.array([m["sup_gap"] for m in metrics])

    ax_curve.semilogy(ks, max_h, marker="o", color="#0b1d2a", lw=1.6,
                        label="max petal half-width over $\\tau$")
    ax_curve.semilogy(ks, mean_h, marker="s", color="#7a6b4a", lw=1.4,
                        label="mean petal half-width")
    ax_curve.semilogy(ks, sup_g, marker="^", color="#c7271a", lw=1.4,
                        label=r"$\sup_\tau|\hat p_{\rm stitched} - p|$ at start state")
    if len(ks) > 1 and max_h[0] > 0:
        ref = max_h[0] / (ks + 1) ** 2
        ax_curve.semilogy(ks, ref, ls=":", color="0.45", lw=1.0,
                            label=r"$\propto 1/(k+1)^2$ reference")
    ax_curve.set_xlabel("number of intermediate waypoints  $k$")
    ax_curve.set_ylabel("metric  (log scale)")
    ax_curve.set_xticks(ks)
    ax_curve.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax_curve.grid(True, which="both", alpha=0.25)

    fig.suptitle(
        f"Petal convergence to the actual path under refinement\n"
        f"a={'z'+str(zones[a]) if zones else a}  →  b={'z'+str(zones[b]) if zones else b},   "
        f"T={T},   ρ={rho:.4g}",
        fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=200); plt.close(fig)
    return metrics


def make_legend_text(a, b, T, rho, zones, n_states, hdr_core, hdr_with_band, outpath: Path):
    """Companion text describing what the petal figure encodes."""
    union_core = int(hdr_core.any(axis=0).sum())
    union_band = int(hdr_with_band.any(axis=0).sum())
    z_a = zones[a] if zones else a
    z_b = zones[b] if zones else b
    txt = f"""# Doob-bridge petal — what each visual element encodes

## The pair
- start a = state {a} (zone {z_a})         [green dot at bottom of figure]
- end   b = state {b} (zone {z_b})         [red dot at top of figure]
- horizon T = {T} d
- generator perturbation rho = {rho:.4g}
- state space size n = {n_states}

## How to read the petal

The figure is an *abstract* shape, not a per-state heatmap. Time runs UP on
the y-axis. The petal is centred on a vertical spine (x = 0). The petal's
WIDTH at height tau encodes how spread out the bridge marginal p(tau, .)
is at that instant.

Width measure: petal half-width = (1/2) / sum_c p(tau, c)^2  (Inverse Simpson;
"effective number of states with mass"). At tau = 0 the marginal is a Dirac
on a, so effective N = 1 -> the petal pinches to a single point at the
spine. Same at tau = T. In between, mass spreads across many states ->
the petal opens up.

## What each colour means

- BLACK ribbon edge .................. boundary of the no-error 95% mass spread
                                       (effective-N measure of where mass is
                                       under Q_em alone).
- INTERNAL GRAYSCALE GRADIENT ........ darker where peak probability is high
                                       (concentration). Lighter where the
                                       distribution is flatter at that tau.
- BEIGE FILL between core and halo ... theorem allowance: at this tau the
                                       true bridge marginal could spread
                                       further out by up to eps_c(tau) per
                                       state, expanding the effective-N to
                                       the halo width.
- BEIGE DASHED edge .................. outer boundary of the theorem-respecting
                                       95% spread (effective N evaluated under
                                       worst-case p + eps).
- GREEN dot at (0, 0) ................ start state a, where the petal pinches.
- RED dot at (0, T) .................. end state b, where the petal pinches.
- DOTTED vertical line ............... the spine x = 0 (purely a visual aid).

## Companion files

- petal_a{a}_b{b}_T{T}.png ........... the petal itself.
- peak_state_strip_a{a}_b{b}_T{T}.png  vertical strip showing which zone has
                                       the highest bridge probability at each
                                       time (the "most likely intermediate
                                       zone" sequence).
- union_a{a}_b{b}_T{T}.png ........... per-state residence time in the 95% HDR.
- petal_summary.csv .................. per-tau HDR sizes and core mass.

## Summary numbers

- {union_core} of {n_states} states are *ever* in the 95% HDR core
  (no error allowance) over tau in (0, T).
- {union_band} of {n_states} states must be included if the theorem error
  allowance is also respected.
"""
    outpath.write_text(txt)


def make_petal_figure(taus, prob_matrix, eps_matrix, hdr_core, hdr_with_band,
                       state_order, a, b, T, zones, rho, outpath: Path):
    n_tau, n = prob_matrix.shape
    Pord = prob_matrix[:, state_order]
    Eord = eps_matrix[:, state_order]
    HDR_core = hdr_core[:, state_order]
    HDR_band = hdr_with_band[:, state_order]
    a_pos = int(np.where(state_order == a)[0][0])
    b_pos = int(np.where(state_order == b)[0][0])

    # Custom colormaps:
    #   GRAY  : white -> black for the petal core
    #   BEIGE : transparent -> beige for the error halo
    GRAY = LinearSegmentedColormap.from_list(
        "petal_gray", ["#ffffff", "#dcdcdc", "#7a7a7a", "#1a1a1a"])
    BEIGE = LinearSegmentedColormap.from_list(
        "petal_beige", [(0, 0, 0, 0), to_rgba("#dcd2bb", 0.45),
                         to_rgba("#a3946d", 0.85)])

    fig, ax = plt.subplots(figsize=(13, 8.5))

    # --- 1. Beige halo: eps_c(tau) where bridge mass is LOW -----------------
    # Highlight uncertainty in the "tail" region of the bridge marginal.
    p_max_per_row = np.maximum(Pord.max(axis=1, keepdims=True), EPS)
    p_rownorm = Pord / p_max_per_row
    halo = (Eord / max(Eord.max(), EPS)) * (1.0 - p_rownorm)
    ax.imshow(halo, aspect="auto", origin="lower",
              extent=[-0.5, n - 0.5, 0.0, T],
              cmap=BEIGE, interpolation="bilinear", zorder=1)

    # --- 2. Grayscale petal: PER-ROW normalised so the petal is visible -----
    # Each tau-row gets its own [0, 1] scale, which keeps the petal's shape
    # legible even at intermediate tau where absolute mass is small.
    P_show = p_rownorm ** 0.7
    im = ax.imshow(P_show, aspect="auto", origin="lower",
                    extent=[-0.5, n - 0.5, 0.0, T],
                    cmap=GRAY, interpolation="bilinear", zorder=2)

    # --- 3. Single semi-transparent fill for the union 95% HDR core --------
    # Build a mask of (tau, c) cells in the HDR_core, render with a single
    # contour-like fill (gives a clean "ridge" outline rather than 30 boxes).
    Y = np.linspace(0.0, T, n_tau)
    X = np.arange(n)
    contour_data = HDR_core.astype(float)
    ax.contour(X, Y, contour_data, levels=[0.5], colors=["#0b1d2a"],
               linewidths=1.6, zorder=4)

    # --- 4. Beige dashed contour for HDR-with-band (theorem error allowance)
    contour_band = HDR_band.astype(float)
    ax.contour(X, Y, contour_band, levels=[0.5], colors=["#8a7c5a"],
               linewidths=1.0, linestyles="--", zorder=4)

    # --- 5. Conditional-mean ridge line --------------------------------------
    # mu_pos(tau) = sum_c c * p(tau, c) over the *ordered* x-axis.
    mu_pos = np.array([float(np.arange(n) @ Pord[i]) for i in range(n_tau)])
    ax.plot(mu_pos, Y, color="#9a1d1d", lw=2.0, alpha=0.9, zorder=5,
            label=r"conditional-mean state $E[c\,|\,\tau]$")

    # --- 6. Endpoint markers -------------------------------------------------
    ax.scatter([a_pos], [0.0], s=180, color="#138a3f", zorder=6,
                edgecolor="white", linewidth=1.5, label=f"start  $a={a}$")
    ax.scatter([b_pos], [T], s=180, color="#c7271a", zorder=6,
                edgecolor="white", linewidth=1.5, label=f"end    $b={b}$")

    # x-tick labels: show original state indices (and zone labels if present, sparse)
    n_show = min(n, 24)
    tick_idx = np.linspace(0, n - 1, n_show).round().astype(int)
    if zones is not None:
        tick_lbls = [f"{state_order[i]}\nz{zones[state_order[i]]}" for i in tick_idx]
    else:
        tick_lbls = [str(state_order[i]) for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_lbls, fontsize=7.5)
    ax.set_xlabel("state index (ordered by peak-bridge-marginal time)\n"
                  "natural pinch at endpoints (a at left of timeline, b at right of timeline)")
    ax.set_ylabel("time τ (d)")
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(0.0, T)

    # Colorbar for grayscale prob (per-row normalised)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(r"$p(\tau,c)\,/\,\max_c p(\tau,c)$  (per-time-slice)")

    fig.suptitle("Bridge petal — Doob fwd/bwd marginal with theorem error halo",
                 fontsize=13, y=0.995)
    ax.set_title(
        f"$a={a}\\to b={b}$,   $T={T}$ d,   $\\rho={rho:.4g}$    "
        f"grayscale = bridge marginal   |   beige halo = $\\varepsilon_c(\\tau)$ in low-mass region\n"
        r"navy ridge contour = 95% HDR per $\tau$   |   beige dashed contour = with error allowance"
        r"   |   red curve = $E[c\,|\,\tau]$",
        fontsize=9.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.07),
              ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(outpath, dpi=200); plt.close(fig)


def make_union_strip(taus, hdr_core, hdr_with_band, state_order, a, b, T,
                       prob_matrix, outpath: Path):
    """Compact companion: per-state TIME-FRACTION-IN-HDR strip + endpoint dots."""
    union_band = hdr_with_band.any(axis=0)[state_order]
    n = len(state_order)
    # Fraction of time each state is in the no-band HDR core (intensity).
    frac_core = hdr_core[:, state_order].mean(axis=0)        # in [0,1]
    band_only = hdr_with_band & (~hdr_core)
    frac_band = band_only[:, state_order].mean(axis=0)

    fig, ax = plt.subplots(figsize=(13, 2.6))
    # Beige (added by error allowance) drawn under
    ax.bar(np.arange(n), frac_band, width=0.86, bottom=0.0,
            color="#cdc4b1", edgecolor="#a3946d", lw=0.6,
            label="extra states forced in by theorem error")
    # Dark (always in HDR core) drawn over
    ax.bar(np.arange(n), frac_core, width=0.86, bottom=0.0,
            color="#1a1a1a", edgecolor="0.0", lw=0.6,
            label="fraction of time in 95% HDR core")
    a_pos = int(np.where(state_order == a)[0][0])
    b_pos = int(np.where(state_order == b)[0][0])
    ax.scatter([a_pos], [1.05], s=120, color="#138a3f", zorder=5,
                edgecolor="white", linewidth=1.2, clip_on=False, label=f"start a={a}")
    ax.scatter([b_pos], [1.05], s=120, color="#c7271a", zorder=5,
                edgecolor="white", linewidth=1.2, clip_on=False, label=f"end b={b}")

    n_show = min(n, 30)
    tick_idx = np.linspace(0, n - 1, n_show).round().astype(int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([str(state_order[i]) for i in tick_idx], fontsize=8)
    ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(0, 1.15)
    ax.set_xlabel("state index (same ordering as petal figure)")
    ax.set_ylabel(r"fraction of $\tau \in [0,T]$")
    ax.set_title("Per-state residence in 95% HDR — taller bar = state visited longer",
                  fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4,
              fontsize=9, frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ycd3_output")
    ap.add_argument("--generator", default="Q_em.csv")
    ap.add_argument("--reference_generator", default="Q_emp.csv")
    ap.add_argument("--zones", default="zone_ids.csv")
    ap.add_argument("--a", type=int, default=9)
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--grid", type=int, default=201, help="time-grid resolution")
    ap.add_argument("--level", type=float, default=0.95, help="HDR level (default 0.95)")
    ap.add_argument("--rho_mode", choices=["auto", "explicit", "data"], default="auto")
    ap.add_argument("--rho", type=float, default=0.27)
    ap.add_argument("--multi_k", default="0,1,2,3,4,5",
                    help="comma-separated list of k values for the multi-k refinement panel "
                         "(k = number of intermediate waypoints; sub-bridges = k+1).")
    ap.add_argument("--convergence_k", default="0,1,2,4,8,16",
                    help="comma-separated list of k for the convergence figure")
    args = ap.parse_args()

    out = Path(args.out)
    Qhat = pd.read_csv(out / args.generator).to_numpy()
    Qref = pd.read_csv(out / args.reference_generator).to_numpy()
    zones = pd.read_csv(out / args.zones)["zone_id"].tolist() if (out / args.zones).exists() else None
    n = Qhat.shape[0]
    rho_2 = float(norm(Qref - Qhat, 2))

    print(f"=== petal visualization for a={args.a} -> b={args.b}, T={args.T} ===")
    print(f"State space n = {n}   ||Q_emp-Q_em||_2 = {rho_2:.4f}")

    taus_grid = np.linspace(0.0, args.T, args.grid)
    sample_taus = taus_grid[1:-1][::max(1, len(taus_grid) // 41)]

    if args.rho_mode == "explicit":
        rho = args.rho
    elif args.rho_mode == "data":
        rho = rho_2
    else:
        rho = auto_rho_for_pair(Qhat, Qref, args.a, args.b, args.T, sample_taus)
    print(f"rho used for band = {rho:.4f}")

    # Compute bridge marginals + eps vectors on the time grid (use Qhat for the heatmap)
    PT_h = expm(args.T * Qhat)
    prob_matrix = np.zeros((args.grid, n))
    eps_matrix = np.zeros((args.grid, n))
    hdr_core = np.zeros((args.grid, n), dtype=bool)
    hdr_band = np.zeros((args.grid, n), dtype=bool)
    for i, tau in enumerate(taus_grid):
        if i == 0 or i == args.grid - 1:
            # Endpoints: deterministic.
            p = np.zeros(n)
            p[args.a if i == 0 else args.b] = 1.0
            eb = np.zeros(n)
        else:
            p = bridge_marginal(Qhat, args.a, args.b, args.T, float(tau), PT=PT_h)
            eb = bell_eps_vector(Qhat, args.a, args.b, args.T, float(tau), rho)
            eb = np.clip(eb, 0.0, 1.0)
        prob_matrix[i] = p; eps_matrix[i] = eb
        hdr_core[i] = hdr_mask(p, args.level)
        hdr_band[i] = hdr_mask_with_band(p, eb, args.level)

    state_order = order_states_by_peak_time(prob_matrix)

    bout = out / "petal_visualization"
    figdir = bout / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    petal_png = figdir / f"petal_a{args.a}_b{args.b}_T{args.T}.png"
    peak_png = figdir / f"peak_state_strip_a{args.a}_b{args.b}_T{args.T}.png"
    union_png = figdir / f"union_a{args.a}_b{args.b}_T{args.T}.png"
    legend_txt = bout / "PETAL_LEGEND.txt"

    make_centered_petal_figure(taus_grid, prob_matrix, eps_matrix, hdr_core, hdr_band,
                                  args.a, args.b, args.T, zones, rho, petal_png)
    make_peak_state_strip(taus_grid, prob_matrix, args.a, args.b, args.T, zones, peak_png)
    make_union_strip(taus_grid, hdr_core, hdr_band, state_order, args.a, args.b,
                      args.T, prob_matrix, union_png)
    make_legend_text(args.a, args.b, args.T, rho, zones, prob_matrix.shape[1],
                       hdr_core, hdr_band, legend_txt)

    # Multi-k refinement petals — one panel per number of intermediate waypoints.
    multi_png = figdir / f"petal_multi_k_a{args.a}_b{args.b}_T{args.T}.png"
    k_values = [int(k) for k in args.multi_k.split(",") if k.strip()]
    make_multi_k_petals(Qhat, Qref, args.a, args.b, args.T, args.grid // 2, rho,
                          k_values, zones, multi_png, eps_func=None)
    print(f"Multi-k petals -> {multi_png.name}")

    # Convergence figure: petals shrinking + log-scale convergence curves
    conv_k = [int(k) for k in args.convergence_k.split(",") if k.strip()]
    conv_png = figdir / f"petal_convergence_a{args.a}_b{args.b}_T{args.T}.png"
    metrics = make_petal_convergence_figure(
        Qhat, Qref, args.a, args.b, args.T, args.grid // 2, rho, conv_k, zones, conv_png)
    pd.DataFrame(metrics).to_csv(bout / "petal_convergence_summary.csv", index=False)
    print(f"Convergence    -> {conv_png.name}")
    for m in metrics:
        print(f"  k={m['k']:2d}  max_half={m['max_half']:.4f}  "
              f"mean_half={m['mean_half']:.4f}  sup_gap={m['sup_gap']:.4f}")

    # Summary CSV: per-tau number of states in HDR core / band
    summary = pd.DataFrame({
        "tau": taus_grid,
        "core_size": hdr_core.sum(axis=1),
        "band_size": hdr_band.sum(axis=1),
        "core_mass": [float(prob_matrix[i, hdr_core[i]].sum()) for i in range(args.grid)],
    })
    summary.to_csv(bout / "petal_summary.csv", index=False)

    n_union_core = int(hdr_core.any(axis=0).sum())
    n_union_band = int(hdr_band.any(axis=0).sum())
    print(f"States in union 95% HDR core (no error band)     : {n_union_core} of {n}")
    print(f"States in union 95% HDR with theorem error band  : {n_union_band} of {n}")
    print(f"Petal figure -> {petal_png}")
    print(f"Union strip  -> {union_png}")
    print(f"Summary CSV  -> {bout / 'petal_summary.csv'}")


if __name__ == "__main__":
    main()
