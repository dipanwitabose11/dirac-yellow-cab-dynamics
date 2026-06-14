"""
Theorem-respecting 95% bridge tube over the Manhattan zone graph (3D).

PROCEDURE (matches the user's theorem statement):

  For each intermediate time tau_i in (0, T):

    1. Compute the Doob-bridge marginal under the estimated generator:
           hat_p(tau, c) = hat_P_t(a, c) * hat_P_{T-t}(c, b) / hat_P_T(a, b).

    2. Compute the per-state pointwise bound on |N_c(tau) - hat_N_c(tau)|
       from the L2-tight version of the theorem (with no e^{KT}, since
       e^{sQ} is sub-stochastic):
           eps_c(tau) = t * rho * hat_P_{T-t}(c, b)
                      + (T - t) * rho * hat_P_t(a, c)
                      + t * (T - t) * rho^2.

    3. From the boxed conditional bound,
           |B_A(t) - hat_B_A(t)|
              <=  E_A(t) / (hat_P_T(a,b) - E_T)
                + hat_N_A(t) * E_T / [ hat_P_T(a,b) (hat_P_T(a,b) - E_T) ]
       with E_T = T * rho. We evaluate this per-singleton-state
       eps_marg_c(tau) (set A = {c}).

    4. The "theorem-respecting 95% HDR" at tau is the smallest set A_tau
       such that
           sum_{c in A_tau}  max( hat_p(tau, c) - eps_marg_c(tau), 0 )  >=  0.95.
       This guarantees: under any admissible Q with || Q - hat_Q || <= rho,
       the true bridge has mass >= 0.95 on A_tau at time tau.

    5. UNION = union over tau of A_tau. By a union bound, the bridge lies
       inside UNION at *every* tau with marginal probability >= 0.95.

VISUAL:

  3D plot with zone-graph layout on (x, y) and time on z. UNION states
  are drawn as beige vertical columns from z = 0 to z = T. At each
  rendered tau slice, dark dots mark the states currently in A_tau, sized
  by hat_p(tau, c). A 3D polyline traces the most-likely zone over time.

Outputs (under <out>/petal_visualization/figures/):
    petal_3d_tube_a<a>_b<b>_T<T>.png        main 3D figure
    petal_3d_slices_a<a>_b<b>_T<T>.png      per-tau zone-graph cross-sections
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.linalg import expm, norm
import networkx as nx

EPS = 1e-14


# ------------------ bridge ------------------

def bridge_marginal(Q, a, b, T, tau, PT=None):
    if PT is None:
        PT = expm(T * Q)
    Pt = expm(tau * Q)
    Prem = expm((T - tau) * Q)
    p = Pt[a, :] * Prem[:, b] / max(PT[a, b], EPS)
    p = np.maximum(p, 0.0)
    return p / max(p.sum(), EPS)


# ------------------ theorem bounds ------------------

def joint_eps_per_state(Qhat, a, b, T, tau, rho):
    """Per-state pointwise bound  |N_c - hat_N_c|  for A = {c}.
        eps_c = t rho hat_P_{T-t}(c,b) + (T-t) rho hat_P_t(a,c) + t(T-t) rho^2.
    """
    Pt = expm(tau * Qhat)
    Prem = expm((T - tau) * Qhat)
    return tau * rho * Prem[:, b] + (T - tau) * rho * Pt[a, :] + tau * (T - tau) * rho * rho


def conditional_bridge_bound(Qhat, a, b, T, tau, rho, eps_c, PT=None):
    """Per-state conditional bound on |B_c(tau) - hat_B_c(tau)| (set A={c})
    using the boxed identity:
        |B - hat_B| <= E_A / (hat_P_T - E_T) + hat_N_A E_T / [hat_P_T (hat_P_T - E_T)]
    Returns +inf for every state if hypothesis hat_P_T(a,b) > E_T fails.
    """
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
    """Smallest set such that worst-case lower-bound mass >= level."""
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
    """Smallest set such that mass >= level under p alone."""
    order = np.argsort(p)[::-1]
    cum = 0.0
    keep = np.zeros_like(p, dtype=bool)
    for idx in order:
        keep[idx] = True
        cum += float(p[idx])
        if cum >= level:
            break
    return keep


# ------------------ layout ------------------

def build_layout(Qhat, edges_df, zones_list, layout_alg="kamada_kawai", seed=42):
    n = Qhat.shape[0]
    zone_to_state = {z: i for i, z in enumerate(zones_list)}
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for _, r in edges_df.iterrows():
        u_id = int(r["u"]); v_id = int(r["v"])
        if u_id in zone_to_state and v_id in zone_to_state:
            G.add_edge(zone_to_state[u_id], zone_to_state[v_id])
    if layout_alg == "kamada_kawai":
        layout = nx.kamada_kawai_layout(G)
    elif layout_alg == "spectral":
        layout = nx.spectral_layout(G)
    else:
        layout = nx.spring_layout(G, seed=seed, iterations=300, k=1.4 / np.sqrt(n))
    xs = np.array([layout[i][0] for i in range(n)])
    ys = np.array([layout[i][1] for i in range(n)])
    return xs, ys, G


# ------------------ 3D figure ------------------

def make_3d_tube(Qhat, Qref, a, b, T, n_slices, rho, xs, ys, G,
                   zones, outpath: Path, azim, elev, level=0.95):
    n = Qhat.shape[0]
    PT_h = expm(T * Qhat)
    PT_ab = float(PT_h[a, b])
    E_T = T * rho

    # Per-tau A_τ (theorem-respecting 95% HDR) and bridge mass
    taus = np.linspace(0.0, T, n_slices)
    A_at_tau = []
    A_naive_at_tau = []
    p_at_tau = []
    for tau in taus:
        if tau <= 0.0 or tau >= T:
            A_at_tau.append(np.zeros(n, dtype=bool))
            A_naive_at_tau.append(np.zeros(n, dtype=bool))
            p_at_tau.append(np.zeros(n))
            continue
        p_hat = bridge_marginal(Qhat, a, b, T, float(tau), PT=PT_h)
        eps_c = joint_eps_per_state(Qhat, a, b, T, float(tau), rho)
        eps_marg = conditional_bridge_bound(Qhat, a, b, T, float(tau), rho, eps_c, PT=PT_h)
        A_at_tau.append(hdr_with_theorem(p_hat, eps_marg, level=level))
        A_naive_at_tau.append(hdr_no_theorem(p_hat, level=level))
        p_at_tau.append(p_hat)

    union_states = np.any(np.stack(A_at_tau), axis=0)
    union_naive = np.any(np.stack(A_naive_at_tau), axis=0)
    print(f"  hypothesis hat_P_T(a,b) > T*rho:  "
          f"{PT_ab:.4g} > {E_T:.4g}  ->  {'OK' if PT_ab > E_T else 'VIOLATED -> bound = inf'}")
    print(f"  union 95% HDR (no error)           : {int(union_naive.sum())} of {n}")
    print(f"  union 95% HDR (theorem-respecting) : {int(union_states.sum())} of {n}")

    fig = plt.figure(figsize=(13, 10.5))
    ax = fig.add_subplot(111, projection="3d")

    # Floor: zone graph
    for u, v in G.edges():
        ax.plot([xs[u], xs[v]], [ys[u], ys[v]], [0, 0],
                color="0.85", lw=0.5, zorder=1)
    ax.scatter(xs, ys, np.zeros(n), c="0.7", s=18, depthshade=False, zorder=2)

    # Beige columns for theorem-respecting union states (the 95% bridge tube)
    BEIGE = to_rgba("#cdc4b1", 0.55)
    for c in range(n):
        if union_states[c]:
            ax.plot([xs[c], xs[c]], [ys[c], ys[c]], [0, T],
                    color=BEIGE, lw=6.0, zorder=3, solid_capstyle="round")

    # Naive (no-error) union: thin solid dark column ON TOP of the beige
    for c in range(n):
        if union_naive[c]:
            ax.plot([xs[c], xs[c]], [ys[c], ys[c]], [0, T],
                    color="0.20", lw=1.5, zorder=4, alpha=0.6)

    # ---- per-slice dots: SIZE *and* COLOUR both encode p(tau, c) ----
    # Reference scale: probability p ∈ [0, 1].
    # Marker AREA: s = S_MAX * p   (linear in p; radius scales as sqrt(p))
    # Marker COLOUR: plasma-like ramp keyed to p (same colorbar for endpoints)
    S_MAX = 1500.0
    P_CMAP = plt.cm.plasma   # 0 -> dark purple, 1 -> bright yellow

    # Collect ALL (xs, ys, taus, p, color) for one combined scatter (so the
    # colorbar is shared and depthshade works consistently).
    Xs_all, Ys_all, Zs_all, Ss_all, Cs_all = [], [], [], [], []
    for ti, tau in enumerate(taus):
        if not (0 < tau < T):
            continue
        in_A = A_at_tau[ti]
        if not in_A.any():
            continue
        idx = np.where(in_A)[0]
        p_vals = p_at_tau[ti][idx]
        Xs_all.append(xs[idx]); Ys_all.append(ys[idx])
        Zs_all.append(np.full(len(idx), tau))
        Ss_all.append(S_MAX * p_vals)
        Cs_all.append(p_vals)
    if Xs_all:
        Xs_all = np.concatenate(Xs_all); Ys_all = np.concatenate(Ys_all)
        Zs_all = np.concatenate(Zs_all); Ss_all = np.concatenate(Ss_all)
        Cs_all = np.concatenate(Cs_all)
        sc = ax.scatter(Xs_all, Ys_all, Zs_all, s=Ss_all, c=Cs_all,
                          cmap=P_CMAP, vmin=0.0, vmax=1.0,
                          edgecolor="white", linewidth=0.3,
                          depthshade=True, alpha=0.92, zorder=5)
    else:
        sc = None

    # ---- endpoints: same encoding (p = 1 at both, so they get S_MAX size and top of cmap) ----
    end_color_a = P_CMAP(1.0)
    end_color_b = P_CMAP(1.0)
    ax.scatter([xs[a]], [ys[a]], [0], s=S_MAX, c=[end_color_a], marker="o",
                edgecolor="#138a3f", linewidth=2.4, zorder=20,
                label=f"start  z{zones[a]}  ($\\tau$=0,  p=1)")
    ax.scatter([xs[b]], [ys[b]], [T], s=S_MAX, c=[end_color_b], marker="o",
                edgecolor="#c7271a", linewidth=2.4, zorder=20,
                label=f"end    z{zones[b]}  ($\\tau$=T,  p=1)")

    # ---- most-likely zone path ----
    fine = np.linspace(0.0, T, 200)
    peak_state = np.array(
        [int(np.argmax(bridge_marginal(Qhat, a, b, T, float(tau), PT=PT_h)))
         for tau in fine])
    ax.plot(xs[peak_state], ys[peak_state], fine,
             color="#7a1410", lw=2.0, zorder=8, alpha=0.85,
             label="most-likely zone path")

    # ---- shared colorbar ----
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.05, shrink=0.7)
        cb.set_label(r"bridge marginal  $p(\tau, c)$  (size also linear in $p$)")
        cb.ax.tick_params(labelsize=8)

    # Zone labels for union states only (so the figure isn't cluttered)
    for c in range(n):
        if union_states[c]:
            ax.text(xs[c], ys[c], -0.04, f"z{zones[c]}",
                     color="0.30", fontsize=7.5)

    ax.set_xlabel("zone graph layout x"); ax.set_ylabel("zone graph layout y")
    ax.set_zlabel("time τ")
    ax.set_zlim(0, T)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(
        f"Theorem-respecting 95% bridge tube on Manhattan zone graph\n"
        f"z{zones[a]} → z{zones[b]},  T = {T},  ρ = {rho:.4g}  "
        f"(union: {int(union_states.sum())} of {n} zones)\n"
        r"ball area & colour both $\propto p(\tau, c)$"
        "  |  endpoints (p = 1) are the largest, brightest balls\n"
        "beige column = theorem-respecting 95% tube  |  dark line = no-error 95% tube  |  "
        "red line = most-likely zone",
        fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.97),
               fontsize=9, frameon=False,
               labelspacing=1.4,        # vertical gap between entries
               handletextpad=1.1,        # gap between marker and text
               borderpad=0.6,
               markerscale=0.30)         # shrink the giant scatter markers in the legend
    fig.tight_layout()
    fig.savefig(outpath, dpi=200); plt.close(fig)
    return union_states, union_naive, A_at_tau, p_at_tau


def make_slice_grid(Qhat, Qref, a, b, T, slice_taus, rho, xs, ys, G,
                       zones, outpath: Path, level=0.95):
    """Per-tau snapshot: highlight A_τ on the zone graph."""
    cols = min(4, len(slice_taus))
    rows = int(np.ceil(len(slice_taus) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows),
                                squeeze=False)
    PT_h = expm(T * Qhat)
    BEIGE = to_rgba("#cdc4b1", 0.65)
    n = Qhat.shape[0]
    for ti, tau in enumerate(slice_taus):
        ax = axes[ti // cols][ti % cols]
        for u, v in G.edges():
            ax.plot([xs[u], xs[v]], [ys[u], ys[v]], color="0.86", lw=0.5, zorder=1)
        ax.scatter(xs, ys, c="0.65", s=10, zorder=2)
        if 0 < tau < T:
            p_hat = bridge_marginal(Qhat, a, b, T, float(tau), PT=PT_h)
            eps_c = joint_eps_per_state(Qhat, a, b, T, float(tau), rho)
            eps_marg = conditional_bridge_bound(Qhat, a, b, T, float(tau), rho, eps_c, PT=PT_h)
            A = hdr_with_theorem(p_hat, eps_marg, level=level)
            # Beige halo (extra states forced in by theorem)
            A_naive = hdr_no_theorem(p_hat, level=level)
            extra = A & (~A_naive)
            if extra.any():
                idx = np.where(extra)[0]
                ax.scatter(xs[idx], ys[idx], s=400, c=[BEIGE], zorder=3)
            # Dark core: states in the no-error HDR, sized by mass
            if A_naive.any():
                idx = np.where(A_naive)[0]
                ax.scatter(xs[idx], ys[idx], s=40 + 1500 * p_hat[idx],
                            c=["#1a1a1a"], zorder=4, edgecolor="white", lw=0.4)
        ax.scatter([xs[a]], [ys[a]], s=120, c="#138a3f", zorder=10,
                    edgecolor="white", lw=1.3)
        ax.scatter([xs[b]], [ys[b]], s=120, c="#c7271a", zorder=10,
                    edgecolor="white", lw=1.3)
        ax.set_title(f"τ = {tau:.3f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(len(slice_taus), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(
        f"Per-slice 95% HDR sets on zone graph,  "
        f"z{zones[a]} → z{zones[b]},  T = {T},  ρ = {rho:.4g}\n"
        "dark = states in no-error 95% HDR (sized by mass)  |  "
        "beige = states forced in by theorem error allowance",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ycd3_output")
    ap.add_argument("--generator", default="Q_em.csv")
    ap.add_argument("--reference_generator", default="Q_emp.csv")
    ap.add_argument("--zones", default="zone_ids.csv")
    ap.add_argument("--edges", default="zone_graph_edges.csv")
    ap.add_argument("--a", type=int, default=9)
    ap.add_argument("--b", type=int, default=7)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--n_slices", type=int, default=15)
    ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--azim", type=float, default=-55)
    ap.add_argument("--elev", type=float, default=22)
    ap.add_argument("--layout", default="kamada_kawai",
                    choices=["kamada_kawai", "spring", "spectral"])
    args = ap.parse_args()

    out = Path(args.out)
    Qhat = pd.read_csv(out / args.generator).to_numpy()
    Qref = pd.read_csv(out / args.reference_generator).to_numpy()
    zones = pd.read_csv(out / args.zones)["zone_id"].tolist()
    edges_df = pd.read_csv(out / args.edges)

    rho_HS = float(norm(Qref - Qhat, "fro"))
    rho_2 = float(norm(Qref - Qhat, 2))
    rho_inf = float(np.max(np.sum(np.abs(Qref - Qhat), axis=1)))
    print("=== theorem-respecting 95% bridge tube ===")
    print(f"State space n = {Qhat.shape[0]}")
    print(f"||Q_emp-Q_em||_HS={rho_HS:.4f}, _2={rho_2:.4f}, _(inf->inf)={rho_inf:.4f}")
    print(f"rho used = {args.rho:.4f}    level = {args.level}")
    print(f"Pair: a={args.a} (z{zones[args.a]}) -> b={args.b} (z{zones[args.b]}), T={args.T}")

    xs, ys, G = build_layout(Qhat, edges_df, zones, layout_alg=args.layout)

    bout = out / "petal_visualization"
    figdir = bout / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    main_png = figdir / f"petal_3d_tube_a{args.a}_b{args.b}_T{args.T}.png"
    union_states, union_naive, A_at_tau, p_at_tau = make_3d_tube(
        Qhat, Qref, args.a, args.b, args.T, args.n_slices, args.rho,
        xs, ys, G, zones, main_png, args.azim, args.elev, args.level)
    print(f"3D tube    -> {main_png.name}")

    slice_taus = np.linspace(0.0, args.T, 9)
    slices_png = figdir / f"petal_3d_slices_a{args.a}_b{args.b}_T{args.T}.png"
    make_slice_grid(Qhat, Qref, args.a, args.b, args.T, slice_taus, args.rho,
                       xs, ys, G, zones, slices_png, args.level)
    print(f"Slice grid -> {slices_png.name}")

    # Save union summary
    df = pd.DataFrame({
        "state": np.arange(Qhat.shape[0]),
        "zone_id": zones,
        "in_no_error_union": union_naive.astype(int),
        "in_theorem_union": union_states.astype(int),
        "max_p_over_tau": np.array([max((p_at_tau[ti][c] for ti in range(len(p_at_tau))),
                                          default=0.0) for c in range(Qhat.shape[0])]),
    })
    df.to_csv(bout / f"tube_summary_a{args.a}_b{args.b}_T{args.T}.csv", index=False)


if __name__ == "__main__":
    main()
