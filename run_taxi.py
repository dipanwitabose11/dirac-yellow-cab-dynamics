from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections
import bridge_em as be

# the trip extract is kept once, in the repository root
CSV = next(p for p in ('yellow_tripdata_2016-01.csv', '../yellow_tripdata_2016-01.csv') if Path(p).exists())
OUT = Path('figures'); OUT.mkdir(exist_ok=True)
NZONES, NBINS, EPS, ITERS = 35, 16, 0.01, 150
ROT = np.deg2rad(29.0)
CELL = (0.7, 0.9)

plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 150, 'savefig.bbox': 'tight'})


def rotate(lon, lat):
    x, y = (lon + 73.97) * 84.2, (lat - 40.75) * 111.2
    return x * np.cos(ROT) - y * np.sin(ROT), x * np.sin(ROT) + y * np.cos(ROT)


d = pd.read_csv(CSV, parse_dates=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
d['dur'] = (d.tpep_dropoff_datetime - d.tpep_pickup_datetime).dt.total_seconds() / 60
d = d[(d.dur >= 1) & (d.dur <= 90)]
for c in ('pickup', 'dropoff'):
    d = d[d[f'{c}_longitude'].between(-74.03, -73.90) & d[f'{c}_latitude'].between(40.70, 40.88)]
d['px'], d['py'] = rotate(d.pickup_longitude.values, d.pickup_latitude.values)
d['dx'], d['dy'] = rotate(d.dropoff_longitude.values, d.dropoff_latitude.values)
d['pc'] = list(zip((d.px // CELL[0]).astype(int), (d.py // CELL[1]).astype(int)))
d['dc'] = list(zip((d.dx // CELL[0]).astype(int), (d.dy // CELL[1]).astype(int)))

busy = pd.concat([d.pc, d.dc]).value_counts().index[:NZONES]
cells = set(busy)
adj = {c: [o for o in cells if abs(o[0] - c[0]) + abs(o[1] - c[1]) == 1] for c in cells}
comp, stack = set(), [busy[0]]
while stack:
    c = stack.pop()
    if c not in comp:
        comp.add(c); stack += adj[c]
zones = [c for c in busy if c in comp]
zid = {c: i for i, c in enumerate(zones)}
edges = sorted({tuple(sorted((zid[c], zid[o]))) for c in zones for o in adj[c] if o in comp})
nv = len(zones)
zxy = np.array([((c[0] + .5) * CELL[0], (c[1] + .5) * CELL[1]) for c in zones])
exy = np.array([(zxy[u] + zxy[w]) / 2 for u, w in edges])
xy = np.vstack([zxy, exy])

d = d[d.pc.isin(zid) & d.dc.isin(zid)].copy()
d['x'], d['y'] = d.pc.map(zid), d.dc.map(zid)
d['bin'] = pd.qcut(d.dur, NBINS, labels=False, duplicates='drop')
D = be.dirac_operator(nv, edges)
M = be.Model(D)
n, nobs = M.n, len(d)
data = []
for b, g in d.groupby('bin'):
    C = np.zeros((n, n))
    np.add.at(C, (g.x.values, g.y.values), 1)
    data.append((g.dur.median() / 60, C))

if Path('em_run_v2.npz').exists():
    z = np.load('em_run_v2.npz')
    vstar, logp, lamstar = z['vstar'], float(z['logp']), float(z['lamstar'])
    H = {k: z[k] for k in ('ll', 'R', 'gain', 'lpen')}
    H['v'] = list(z['V'])
    vhat = H['v'][-1]
else:
    v1 = np.full(len(M.I), 10.0)
    p1log = {}
    vstar, logp, lamstar, Rpath = be.paper1_estimator(M, data, nobs, v1, log=p1log)
    import json
    with open('initial_estimator_log_v2.json', 'w') as f:
        json.dump(p1log, f, indent=1, default=str)
    vhat, H = be.bridge_em(M, data, vstar, 0.1 * lamstar, EPS, ITERS)
    np.savez('em_run_v2.npz', vstar=vstar, logp=logp, lamstar=lamstar, V=np.array(H['v']),
             ll=H['ll'], R=H['R'], gain=H['gain'], lpen=H['lpen'])
lam = 0.1 * lamstar
Qstar, Qhat = M.Q(vstar), M.Q(vhat)

hop = np.full((nv, nv), np.inf); np.fill_diagonal(hop, 0)
for u, w in edges:
    hop[u, w] = hop[w, u] = 1
for m_ in range(nv):
    hop = np.minimum(hop, hop[:, [m_]] + hop[[m_], :])


def save(fig, name):
    fig.savefig(OUT / name); plt.close(fig)


def checks():
    V = np.array(H['v'])
    lsat = be.saturated_loglik(data)
    l0, R0 = H['ll'][0], H['R'][0]
    rho_sat = R0 + (lsat - l0) / lam
    inc = np.diff(H['lpen'])
    steps = np.linalg.norm(np.diff(V, axis=0), axis=1)
    eta = V.min()
    N, A, _ = be.estep(M, vhat, data)
    T_total = sum(tau * C.sum() for tau, C in data)
    g = be.gradient(M, vhat, data)
    rng = np.random.default_rng(0)
    fd = []
    for c in rng.choice(len(vhat), 5, replace=False):
        h = 1e-5 * vhat[c]; e = np.zeros_like(vhat); e[c] = h
        fd.append(abs((be.loglik(M, vhat + e, data) - be.loglik(M, vhat - e, data)) / (2 * h) - g[c]) / abs(g[c]))
    res, gnorm = be.foc_residual(M, vhat, data, lam)
    print(f'trips {nobs}  zones {nv}  links {len(edges)}  n {n}  |S0| {len(M.I)}')
    print(f'lambda* {lamstar:.3f}  lambda {lam:.3f}  floor {EPS}  iterations {len(V) - 1}')
    print(f'E-step         sum A / sum T - 1 = {A.sum() / T_total - 1:.2e}')
    print(f'gradient ident max relative gradient error (finite differences) = {max(fd):.2e}')
    print(f'ascent         decreasing steps = {(inc < 0).sum()}   min(increment - surrogate gain)/R = {(inc - H["gain"]).min() / nobs:.2e}')
    print(f'saturated bnd  max l(Q_k)/R = {H["ll"].max() / nobs:.4f}  <=  l_sat/R = {lsat / nobs:.4f}')
    print(f'boundedness    max R(Q_k) = {H["R"].max():.1f}  <=  rho_sat = {rho_sat:.1f}')
    print(f'vanishing (i)  l(Q)-l_lambda(Q) at end = lambda R = {lam * H["R"][-1] / nobs:.4f} per trip  <=  lambda rho_sat = {lam * rho_sat / nobs:.4f}')
    print(f'full sequence  path length = {steps.sum():.1f}   last step = {steps[-1]:.2e}   last increment/R = {inc[-1] / nobs:.2e}')
    print(f'floor indep.   eta = min rate over run = {eta:.3f}  >  floor {EPS}: {eta > EPS}')
    print(f'floor indep.   first-order residual |grad l - lambda s| / |grad l| = {res / gnorm:.2e}')
    return lsat, rho_sat, steps, inc


def figures(lsat, rho_sat, steps, inc):
    S = np.load('manhattan_streets.npz')
    sx, sy = rotate(S['xy'][:, 0], S['xy'][:, 1])
    cuts = np.cumsum(S['lens'])[:-1]
    streets = [np.c_[a, b] for a, b in zip(np.split(sx, cuts), np.split(sy, cuts))]
    lo, hi = zxy.min(0) - 0.6, zxy.max(0) + 0.6
    fig, ax = plt.subplots(figsize=(3.4, 5.2))
    ax.add_collection(matplotlib.collections.LineCollection(streets, colors='0.6', linewidths=0.45))
    for e, (u, w) in enumerate(edges):
        ax.plot(*np.c_[zxy[u], exy[e], zxy[w]], c='k', lw=1.0, zorder=2)
    ax.scatter(*zxy.T, s=26, c='k', edgecolor='w', lw=0.6, zorder=3)
    ax.scatter(*exy.T, s=16, marker='D', facecolor='w', edgecolor='k', lw=0.8, zorder=3)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect('equal'); ax.set_xlabel('km'); ax.set_ylabel('km')
    save(fig, '01_graph.png')

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(np.abs(D), cmap='Greys', interpolation='nearest')
    ax.axhline(nv - .5, c='0.5', lw=0.5); ax.axvline(nv - .5, c='0.5', lw=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, '02_dirac.png')

    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.plot(np.sort(M.lam), '.', c='k', ms=3)
    ax.axhline(0, c='0.7', lw=0.5)
    ax.set_xlabel('index'); ax.set_ylabel('eigenvalue')
    save(fig, '03_spectrum.png')

    A = Qhat.copy(); A[A == 0] = np.nan
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('g', ['0.0', '0.82'])
    cmap.set_bad('white')
    fig, ax = plt.subplots(figsize=(3.2, 3))
    im = ax.imshow(A, cmap=cmap, interpolation='nearest')
    fig.colorbar(im, ax=ax, shrink=0.7, label=r'rate (h$^{-1}$)')
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, '04_generator.png')

    k = np.arange(len(H['ll']))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    ax[0].plot(k, H['lpen'] / nobs, c='k', lw=0.9)
    ax[0].set_ylabel(r'$\ell_\lambda/K$')
    ax[1].plot(k, H['ll'] / nobs, c='k', lw=0.9)
    ax[1].axhline(lsat / nobs, c='k', ls='--', lw=0.8)
    ax[1].set_ylabel(r'$\ell/K$')
    ax[2].plot(k, H['R'], c='k', lw=0.9)
    ax[2].axhline(rho_sat, c='k', ls='--', lw=0.8)
    ax[2].set_yscale('log'); ax[2].set_ylabel(r'$R_{Q^{(k)}}$')
    for a in ax:
        a.set_xlabel('iteration $k$')
    fig.tight_layout()
    save(fig, '05_ascent_and_bounds.png')

    V = np.array(H['v'])
    e1 = V[-1] - V[0]; L1 = np.linalg.norm(e1); e1 /= L1
    w8 = V[min(8, len(V) - 1)] - V[0]; e2 = w8 - (w8 @ e1) * e1; e2 /= np.linalg.norm(e2)
    s = np.linspace(-0.25 * L1, 1.25 * L1, 30)
    t = np.linspace(-1.5, 1.5, 30) * max(abs((V - V[0]) @ e2).max(), 1e-3)
    Z = np.full((len(t), len(s)), np.nan)
    for i, ti in enumerate(t):
        for j, sj in enumerate(s):
            v = V[0] + sj * e1 + ti * e2
            if v.min() >= EPS:
                Z[i, j] = (be.loglik(M, v, data) - lam * M.R(v)) / nobs
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.contour(s, t, Z, 15, colors='0.5', linewidths=0.5)
    ax.plot((V - V[0]) @ e1, (V - V[0]) @ e2, '.-', c='k', ms=2, lw=0.5)
    ax.set_xlabel(rf'along $v^{{({len(V) - 1})}}-v^{{(0)}}$ (h$^{{-1}}$)'); ax.set_ylabel(r'orthogonal, through $v^{(8)}$ (h$^{-1}$)')
    save(fig, '06_penalized_likelihood.png')

    kk = np.arange(1, len(inc) + 1)
    m = (kk >= len(kk) // 2) & (inc > 0)
    q, c0 = np.polyfit(np.log(kk[m]), np.log(inc[m] / nobs), 1)
    gr, c1 = np.polyfit(kk[m], np.log(inc[m] / nobs), 1)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    ax[0].semilogy(kk, steps, c='k', lw=0.9)
    ax[0].set_ylabel(r'$\|v^{(k)}-v^{(k-1)}\|$')
    ax[1].plot(kk, np.cumsum(steps), c='k', lw=0.9)
    ax[1].set_ylabel('path length')
    ax[2].loglog(kk[inc > 0], inc[inc > 0] / nobs, '.', c='k', ms=2)
    ax[2].loglog(kk[m], np.exp(c1 + gr * kk[m]), '--', c='0.4', lw=0.8, label=rf'${np.exp(gr):.3f}^k$')
    ax[2].loglog(kk[m], np.exp(c0) * kk[m] ** q, ':', c='k', lw=0.8, label=rf'$k^{{{q:.2f}}}$')
    ax[2].set_ylabel(r'$\Delta\ell_\lambda/K$'); ax[2].legend(frameon=False, fontsize=7)
    for a in ax:
        a.set_xlabel('iteration $k$')
    fig.tight_layout()
    save(fig, '07_convergence.png')
    print(f'tail fits on k >= {len(kk) // 2}: geometric ratio {np.exp(gr):.4f}, power k^{q:.2f}')


if __name__ == '__main__':
    figures(*checks())
