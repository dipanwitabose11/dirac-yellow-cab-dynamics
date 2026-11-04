"""
em_tools.py -- shared helpers for the long EM runs on the NYC taxi example.

The data, the Dirac operator, the regularizer, lambda and the floor are taken unchanged
from run_taxi.py and bridge_em.py in NYC_code_oct16; this file only adds
  * a resumable EM loop that saves a checkpoint after every iteration,
  * a stopping rule based on the size of the last step,
  * the convergence diagnostics used in Section 5 of the paper.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------- setup
def find_project(start=None, given=None):
    """Folder that contains run_taxi.py and bridge_em.py (NYC_code_oct16)."""
    if given:
        p = Path(given).expanduser().resolve()
        if not (p / "run_taxi.py").exists():
            sys.exit(f"run_taxi.py not found in {p}")
        return p
    p = Path(start or __file__).resolve()
    for q in [p] + list(p.parents):
        if (q / "run_taxi.py").exists() and (q / "bridge_em.py").exists():
            return q
    sys.exit("Could not find NYC_code_oct16 (the folder with run_taxi.py). Use --project PATH.")


def load(project):
    """Import run_taxi (data, model, lambda) and bridge_em from the project folder."""
    here = os.getcwd()
    os.chdir(project)                        # run_taxi.py reads its files by relative path
    sys.path.insert(0, str(project))
    import matplotlib
    matplotlib.use("Agg")
    print("loading the trip data (this takes a little while) ...", flush=True)
    import bridge_em as be
    import run_taxi as rt
    os.chdir(here)
    return rt, be


def uniform_rate_default(rt):
    """c = R(Q*)/R(all ones): the uniform generator with the same R as Q* (74.79 per hour)."""
    ref = np.load(Path(rt.__file__).parent / "em_run_v2.npz")
    ones = np.ones(len(rt.M.I))
    return float(rt.M.R(ref["vstar"]) / rt.M.R(ones))


# ----------------------------------------------------------------------------- EM loop
def _save(path, **arrays):
    tmp = Path(str(path) + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)                    # never leaves a half-written checkpoint


def run(rt, be, ckpt, v0=None, seed=None, lam=None, eps=None, max_iters=2000,
        tol=1e-6, label=""):
    """
    Algorithm 1 (penalized EM), exactly as be.bridge_em, but resumable.

    ckpt   checkpoint file (.npz). If it exists the run continues from it.
    v0     initial rates on S0 (used only when there is no checkpoint and no seed).
    seed   dict with V, ll, R, gain of earlier iterations to continue from.
    Stops when ||v_k - v_{k-1}|| / ||v_k|| < tol, when the M-step cannot improve
    the surrogate (step exactly 0), or after max_iters iterations in total.
    The checkpoint is written after every iteration, so Ctrl-C loses at most one step.
    """
    M, data, K = rt.M, rt.data, rt.nobs
    lam = rt.lam if lam is None else lam
    eps = rt.EPS if eps is None else eps
    ckpt = Path(ckpt)
    if ckpt.exists():
        d = np.load(ckpt)
        if not (np.isclose(float(d["lam"]), lam) and np.isclose(float(d["eps"]), eps)):
            sys.exit(f"{ckpt} was made with lambda={float(d['lam'])}, eps={float(d['eps'])}; "
                     f"delete it or use the same values.")
        V, LL, RR, G = list(d["V"]), list(d["ll"]), list(d["R"]), list(d["gain"])
        print(f"resuming {ckpt.name} at k = {len(V) - 1}")
    elif seed is not None:
        V, LL, RR, G = list(seed["V"]), list(seed["ll"]), list(seed["R"]), list(seed["gain"])
        print(f"continuing given iterates from k = {len(V) - 1}")
    else:
        V, LL, RR, G = [np.maximum(np.asarray(v0, float), eps)], [], [], []
    meta = dict(lam=lam, eps=eps, label=label)

    v = V[-1].copy()
    reason = f"reached max_iters = {max_iters}"
    if len(V) > 1:
        s = np.linalg.norm(V[-1] - V[-2])
        if s == 0.0:
            print(f"this run already stalled at k = {len(V) - 1}: the M-step could not improve the "
                  f"surrogate there, and running again from the same iterate gives the same result.")
            return ckpt
        if s / np.linalg.norm(V[-1]) < tol:
            print("already below the tolerance; nothing to do")
            return ckpt
    print(f"{'k':>5} {'step':>10} {'step/|v|':>10} {'l_lambda/K':>12} {'incr/K':>10} "
          f"{'min rate':>9} {'s/iter':>7}", flush=True)
    while len(V) - 1 < max_iters:
        t0 = time.time()
        N, A, ll = be.estep(M, v, data)
        LL.append(ll)
        RR.append(M.R(v))
        vn = be.mstep(M, N, A, lam, eps, v)
        G.append(be.surrogate(M, N, A, vn, lam) - be.surrogate(M, N, A, v, lam))
        step = np.linalg.norm(vn - v)
        v = vn
        V.append(v.copy())
        _save(ckpt, V=np.array(V), ll=np.array(LL), R=np.array(RR), gain=np.array(G), **meta)
        k = len(V) - 1
        lp_old = (ll - lam * RR[-1]) / K                   # l_lambda(Q^(k-1)) / K
        lp_new = (be.loglik(M, v, data) - lam * M.R(v)) / K  # l_lambda(Q^(k)) / K
        print(f"{k:5d} {step:10.4g} {step / np.linalg.norm(v):10.3e} {lp_new:12.6f} "
              f"{lp_new - lp_old:10.3e} {v.min():9.3f} {time.time() - t0:7.1f}", flush=True)
        if step == 0.0:
            reason = "the M-step could not improve the surrogate (step = 0)"
            break
        if step / np.linalg.norm(v) < tol:
            reason = f"step/|v| < tol = {tol:g}"
            break
    print(f"stopped at k = {len(V) - 1}: {reason}")
    return ckpt


def history(rt, be, ckpt):
    """Load a checkpoint and add l and R at the last iterate (so ll, R, lpen have k+1 entries)."""
    d = np.load(ckpt)
    V = d["V"]
    ll, R, gain = list(d["ll"]), list(d["R"]), list(d["gain"])
    # A step of exactly 0 means the M-step could not improve the surrogate and kept the
    # current iterate (see bridge_em.mstep). Such steps are not convergence; drop them and
    # remember where the run stalled.
    stalled_at = None
    while len(V) > 1 and len(gain) == len(V) - 1 and np.array_equal(V[-1], V[-2]):
        stalled_at = len(V) - 1              # ends at the first zero step
        V, gain = V[:-1], gain[:-1]
        ll, R = ll[:len(V)], R[:len(V)]
    if len(ll) < len(V):
        ll.append(be.loglik(rt.M, V[-1], rt.data))
        R.append(rt.M.R(V[-1]))
    lam = float(d["lam"])
    H = dict(V=V, ll=np.array(ll), R=np.array(R), gain=np.array(gain), lam=lam,
             eps=float(d["eps"]), stalled_at=stalled_at)
    H["lpen"] = H["ll"] - lam * H["R"]
    return H


# ----------------------------------------------------------------------------- diagnostics
def step_stats(V):
    steps = np.linalg.norm(np.diff(V, axis=0), axis=1)
    return steps, np.linalg.norm(V, axis=1)


def geo_ratio(x, a, b):
    """Least-squares fit log x_k = c + k log r over k = a+1..b (x[k-1] is the k-th value)."""
    kk = np.arange(a + 1, b + 1)
    y = x[a:b]
    m = y > 0
    return float(np.exp(np.polyfit(kk[m], np.log(y[m]), 1)[0]))


def tail_fits(inc, K, k0):
    """Geometric and power-law fits of the increments on k >= k0, with rms residuals of log."""
    kk = np.arange(1, len(inc) + 1)
    m = (kk >= k0) & (inc > 0)
    y = np.log(inc[m] / K)
    g1, g0 = np.polyfit(kk[m], y, 1)
    p1, p0 = np.polyfit(np.log(kk[m]), y, 1)
    rg = float(np.sqrt(np.mean((y - (g0 + g1 * kk[m])) ** 2)))
    rp = float(np.sqrt(np.mean((y - (p0 + p1 * np.log(kk[m]))) ** 2)))
    return dict(k0=k0, ratio=float(np.exp(g1)), rms_geo=rg, power=float(p1), rms_pow=rp,
                geo=(g0, g1), pow=(p0, p1), mask=m)


def foc_relative(be, rt, v, lam, tol):
    r, g = be.foc_residual(rt.M, v, rt.data, lam, tol=tol)
    return r / g


def k90(x):
    return int(np.argmax(x - x[0] >= 0.9 * (x[-1] - x[0])))


def rates_summary(rt, v):
    M, nv = rt.M, rt.nv
    q = -np.diag(M.Q(v))
    zl = M.I < nv
    return {
        "zone -> link rates": (v[zl].min(), v[zl].max()),
        "link -> zone rates": (v[~zl].min(), v[~zl].max()),
        "zone exit rates": (q[:nv].min(), q[:nv].max()),
        "link exit rates": (q[nv:].min(), q[nv:].max()),
        "zone mean holding time (min)": (60 / q[:nv].max(), 60 / q[:nv].min()),
        "link mean holding time (min)": (60 / q[nv:].max(), 60 / q[nv:].min()),
    }
