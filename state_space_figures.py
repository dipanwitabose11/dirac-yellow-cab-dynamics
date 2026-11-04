"""
state_space_figures.py -- figures 01-03 of the state space (written to figures/):
    01_graph.png     zones and links drawn over the Manhattan street centerline
    02_dirac.png     nonzero entries of the incidence Dirac operator D
    03_spectrum.png  sorted eigenvalues of D

    python3 state_space_figures.py
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

import matplotlib.collections  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import run_taxi as rt  # noqa: E402  (loads the data and builds D; does not run the EM)

S = np.load('manhattan_streets.npz')
sx, sy = rt.rotate(S['xy'][:, 0], S['xy'][:, 1])
cuts = np.cumsum(S['lens'])[:-1]
streets = [np.c_[a, b] for a, b in zip(np.split(sx, cuts), np.split(sy, cuts))]
lo, hi = rt.zxy.min(0) - 0.6, rt.zxy.max(0) + 0.6

fig, ax = plt.subplots(figsize=(3.4, 5.2))
ax.add_collection(matplotlib.collections.LineCollection(streets, colors='0.6', linewidths=0.45))
for e, (u, w) in enumerate(rt.edges):
    ax.plot(*np.c_[rt.zxy[u], rt.exy[e], rt.zxy[w]], c='k', lw=1.0, zorder=2)
ax.scatter(*rt.zxy.T, s=26, c='k', edgecolor='w', lw=0.6, zorder=3)
ax.scatter(*rt.exy.T, s=16, marker='D', facecolor='w', edgecolor='k', lw=0.8, zorder=3)
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
ax.set_aspect('equal'); ax.set_xlabel('km'); ax.set_ylabel('km')
rt.save(fig, '01_graph.png')

fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(np.abs(rt.D), cmap='Greys', interpolation='nearest')
ax.axhline(rt.nv - .5, c='0.5', lw=0.5); ax.axvline(rt.nv - .5, c='0.5', lw=0.5)
ax.set_xticks([]); ax.set_yticks([])
rt.save(fig, '02_dirac.png')

fig, ax = plt.subplots(figsize=(3.2, 2.2))
ax.plot(np.sort(rt.M.lam), '.', c='k', ms=3)
ax.axhline(0, c='0.7', lw=0.5)
ax.set_xlabel('index'); ax.set_ylabel('eigenvalue')
rt.save(fig, '03_spectrum.png')
print('saved figures/01_graph.png, 02_dirac.png, 03_spectrum.png')
