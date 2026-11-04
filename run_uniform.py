"""
run_uniform.py -- Algorithm 1 on the NYC taxi data from the UNIFORM initial generator,
run until it stops moving.

Start: the same rate c on every allowed transition of S0 (zone -> link and link -> zone).
Default c = 74.79 per hour = R(Q*)/R(all ones), i.e. the uniform generator with the same
value of the regularizer as the companion-paper estimator; set another value with --c.
lambda (0.413) and the floor (0.01) are those of the paper (run_taxi.py).

Saves runs/uniform_c<c>.npz after every iteration. Stop with Ctrl-C at any time and run
the same command again to continue.

Stopping rule: ||v_k - v_{k-1}|| / ||v_k|| < --tol (default 1e-6), or the M-step can no
longer improve the surrogate, or --max-iters iterations in total (default 2000).

    python3 run_uniform.py
    python3 run_uniform.py --c 10
    python3 run_uniform.py --max-iters 3          # quick test
"""
import argparse
from pathlib import Path

import numpy as np

import em_tools as et

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c", type=float, default=None, help="uniform rate per hour (default 74.79)")
    ap.add_argument("--lam", type=float, default=None, help="penalty weight (default: the paper's 0.413)")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--max-iters", type=int, default=2000)
    ap.add_argument("--project", default=None, help="path of NYC_code_oct16 (found automatically)")
    a = ap.parse_args()

    rt, be = et.load(et.find_project(HERE, a.project))
    c = et.uniform_rate_default(rt) if a.c is None else a.c
    lam = rt.lam if a.lam is None else a.lam
    (HERE / "runs").mkdir(exist_ok=True)
    ckpt = HERE / "runs" / f"uniform_c{c:.4g}.npz"
    print(f"trips {rt.nobs}, |S0| {len(rt.M.I)}, lambda {lam:.4f}, floor {rt.EPS}, "
          f"uniform start c = {c:.4f} per hour, tol {a.tol:g}")
    print(f"checkpoint: {ckpt}")
    et.run(rt, be, ckpt, v0=np.full(len(rt.M.I), c), lam=lam, max_iters=a.max_iters, tol=a.tol,
           label=f"uniform c={c:.6g}")
    print("next:  python3 convergence_report.py" + ("" if a.c is None else f" --c {a.c:g}"))


if __name__ == "__main__":
    main()
