import warnings
import numpy as np
import scipy.linalg as sl
from scipy.signal import lfilter
import cvxpy as cp

warnings.filterwarnings('ignore', message='.*encountered in matmul')

TOL = dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)


def dirac_operator(nv, edges):
    ne = len(edges)
    B = np.zeros((nv, ne))
    for e, (u, w) in enumerate(edges):
        B[u, e], B[w, e] = -1.0, 1.0
    D = np.zeros((nv + ne, nv + ne))
    D[:nv, nv:], D[nv:, :nv] = B, B.T
    return D


def eigenbasis(D, tol=1e-9):
    lam, U = np.linalg.eigh(D)
    ker = np.abs(lam) < tol
    if ker.any():
        P = U[:, ker] @ U[:, ker].T
        basis = []
        for e in np.eye(len(D)):
            v = P @ e
            for b in basis:
                v -= (b @ v) * b
            if np.linalg.norm(v) > 1e-8:
                basis.append(v / np.linalg.norm(v))
            if len(basis) == ker.sum():
                break
        U[:, ker] = np.array(basis).T
    if not np.all(np.isfinite(U)) or not np.allclose(U.T @ U, np.eye(len(D)), atol=1e-8):
        raise FloatingPointError('eigenbasis is not orthonormal; check the numpy/BLAS installation')
    return lam, U


class Model:
    def __init__(self, D):
        self.D = D
        self.n = len(D)
        self.S0 = np.array(np.nonzero(D)).T
        self.I, self.J = self.S0[:, 0], self.S0[:, 1]
        self.lam, self.U = eigenbasis(D)
        a = np.abs(D).sum(1).max()
        w = np.abs(self.U).max(0)
        U, I, J = self.U, self.I, self.J
        Ac = np.einsum('kn,km,k->mnk', U[I], U[J], D[J, I]).reshape(self.n ** 2, -1)
        Ad = -U[I].T
        A = np.vstack([Ac, Ad])
        om = np.concatenate([(a * np.outer(w, w)).ravel(), w])
        keep = np.abs(A).max(1) > 1e-12
        self.Amat, self.om = A[keep], om[keep]

    def Q(self, v):
        Q = np.zeros((self.n, self.n))
        Q[self.I, self.J] = v
        Q[np.diag_indices(self.n)] = -Q.sum(1)
        return Q

    def R(self, v):
        return self.om @ np.abs(self.Amat @ v)


def loglik(M, v, data):
    Q = M.Q(v)
    return sum((C[C > 0] * np.log(sl.expm(tau * Q)[C > 0])).sum() for tau, C in data)


def estep(M, v, data):
    Q, n = M.Q(v), M.n
    J = np.zeros((n, n))
    ll = 0.0
    for tau, C in data:
        P = sl.expm(tau * Q)
        W = np.where(C > 0, C / np.where(C > 0, P, 1.0), 0.0)
        ll += (C[C > 0] * np.log(P[C > 0])).sum()
        big = np.block([[Q.T, W], [np.zeros((n, n)), Q.T]])
        J += sl.expm(tau * big)[:n, n:]
    return v * J[M.I, M.J], np.diag(J).copy(), ll


def mstep(M, N, A, lam, eps, v_cur):
    sc = N.sum()
    v = cp.Variable(len(N))
    obj = (N / sc) @ cp.log(v) - (A[M.I] / sc) @ v
    if lam > 0:
        obj -= (lam / sc) * (M.om @ cp.abs(M.Amat @ v))
    prob = cp.Problem(cp.Maximize(obj), [v >= eps])
    base = surrogate(M, N, A, v_cur, lam)
    for opts in (dict(solver=cp.CLARABEL, **TOL), dict(solver=cp.CLARABEL),
                 dict(solver=cp.SCS, eps=1e-9, max_iters=200000)):
        try:
            prob.solve(**opts)
        except cp.error.SolverError:
            continue
        if v.value is not None:
            vn = np.maximum(v.value, eps)
            if surrogate(M, N, A, vn, lam) >= base:
                return vn
    return v_cur.copy()


def surrogate(M, N, A, v, lam):
    return N @ np.log(v) - A[M.I] @ v - lam * M.R(v)


def paper1_estimator(M, data, n_obs, v_init, unpen_iter=300, sca_iter=40, log=None, max_halvings=12):
    log = {} if log is None else log
    v = v_init.copy()
    path = []
    log['unpenalized_stop'] = 'reached unpen_iter'
    for _ in range(unpen_iter):
        N, A, ll = estep(M, v, data)
        if not np.isfinite(ll):
            log['unpenalized_stop'] = 'non-finite likelihood'
            break
        if path and ll - path[-1][1] < 1e-6 * n_obs:
            log['unpenalized_stop'] = 'gain below 1e-6 per trip'
            break
        path.append((v.copy(), ll))
        v = N / A[M.I]
    if not path:
        raise FloatingPointError('endpoint likelihood is not finite at the starting generator')
    ll_top = path[-1][1]
    logp = ll_top - n_obs
    start = next(i for i, (p, l) in enumerate(path) if l >= logp)
    v = path[start][0]
    log.update(unpenalized_steps=len(path), unpenalized_cap=unpen_iter,
               unpenalized_ll_per_trip=[l / n_obs for _, l in path],
               ll_top_per_trip=ll_top / n_obs, logp_per_trip=logp / n_obs,
               sca_start_index=start, sca=[])
    Rs = [M.R(v)]
    mult, mult_from = None, None
    log['sca_stop'] = 'reached sca_iter'
    for k in range(sca_iter):
        N, A, ll = estep(M, v, data)
        x = cp.Variable(len(v))
        rhs = (logp - ll + N @ np.log(v) - A[M.I] @ v) / n_obs
        con = (N / n_obs) @ cp.log(x) - (A[M.I] / n_obs) @ x >= rhs
        prob = cp.Problem(cp.Minimize(M.om @ cp.abs(M.Amat @ x)), [con, x >= 1e-6])
        try:
            prob.solve(solver=cp.CLARABEL, **TOL)
        except cp.error.SolverError:
            pass
        rec = dict(step=k + 1, status=prob.status,
                   solver_iterations=getattr(prob.solver_stats, 'num_iters', None),
                   solve_time=getattr(prob.solver_stats, 'solve_time', None))
        t, vn = 0.0, None
        if x.value is not None and np.all(np.isfinite(x.value)):
            xv = np.maximum(x.value, 1e-6)
            rec['minorant_slack_per_trip'] = float((N / n_obs) @ np.log(xv) - (A[M.I] / n_obs) @ xv - rhs)
            # l is continuous and R is convex, so if l(v) > log p and R(x) < R(v), then for t
            # small enough the point v + t (x - v) keeps l >= log p and lowers R. If l(v) = log p
            # (up to rounding), this gives no usable step length, and the halving may fail.
            t = 1.0
            for _ in range(max_halvings + 1):
                u = v + t * (xv - v)
                if loglik(M, u, data) >= logp and M.R(u) < Rs[-1]:
                    vn = u
                    break
                t /= 2
        if vn is None:
            rec.update(accepted=False, step_size=0.0)
            log['sca'].append(rec)
            log['sca_stop'] = 'no feasible decrease of R'
            break
        v = vn
        Rs.append(M.R(v))
        if t == 1.0 and prob.status == cp.OPTIMAL:
            mult, mult_from = float(con.dual_value) / n_obs, k + 1
        rec.update(accepted=True, step_size=t, R=Rs[-1],
                   likelihood_minus_level_per_trip=(loglik(M, v, data) - logp) / n_obs,
                   multiplier=float(con.dual_value) / n_obs, min_rate=float(v.min()))
        log['sca'].append(rec)
        if abs(Rs[-2] - Rs[-1]) < 1e-6 * Rs[-1]:
            log['sca_stop'] = 'R converged'
            break
    if mult is None:
        raise RuntimeError('no convex step was solved to full accuracy; lambda* is undefined')
    log['sca_steps'] = len(log['sca'])
    log['multiplier_from_step'] = mult_from
    return v, logp, 1.0 / mult, np.array(Rs)


def bridge_em(M, data, v0, lam, eps, iters):
    v = np.maximum(v0, eps)
    hist = {'v': [v.copy()], 'll': [], 'R': [], 'gain': []}
    for _ in range(iters):
        N, A, ll = estep(M, v, data)
        hist['ll'].append(ll)
        hist['R'].append(M.R(v))
        vn = mstep(M, N, A, lam, eps, v)
        hist['gain'].append(surrogate(M, N, A, vn, lam) - surrogate(M, N, A, v, lam))
        v = vn
        hist['v'].append(v.copy())
    hist['ll'].append(loglik(M, v, data))
    hist['R'].append(M.R(v))
    for k in ('ll', 'R', 'gain'):
        hist[k] = np.array(hist[k])
    hist['lpen'] = hist['ll'] - lam * hist['R']
    return v, hist


def doob_generator(Q, y, T, t):
    h = sl.expm((T - t) * Q)[:, y]
    Qh = Q * h[None, :] / h[:, None]
    np.fill_diagonal(Qh, 0.0)
    np.fill_diagonal(Qh, -Qh.sum(1))
    return Qh


def bridge_occupation(Q, x, y, T):
    n = len(Q)
    W = np.zeros((n, n))
    W[x, y] = 1.0 / sl.expm(T * Q)[x, y]
    big = np.block([[Q.T, W], [np.zeros((n, n)), Q.T]])
    return np.diag(sl.expm(T * big)[:n, n:]).copy()


def bridge_marginals(Q, x, y, T, times):
    PT = sl.expm(T * Q)[x, y]
    return np.array([sl.expm(t * Q)[x] * sl.expm((T - t) * Q)[:, y] / PT for t in times])


def bridge_jumps(Q, x, y, T):
    n = len(Q)
    W = np.zeros((n, n))
    W[x, y] = 1.0 / sl.expm(T * Q)[x, y]
    big = np.block([[Q.T, W], [np.zeros((n, n)), Q.T]])
    J = sl.expm(T * big)[:n, n:]
    Nexp = Q * J
    np.fill_diagonal(Nexp, 0.0)
    return Nexp


def saturated_loglik(data):
    s = 0.0
    for _, C in data:
        rows = C.sum(1, keepdims=True)
        W = np.divide(C, rows, out=np.zeros_like(C), where=rows > 0)
        s += (C[C > 0] * np.log(W[C > 0])).sum()
    return s


def skeleton_matrix(Q, sig):
    m = len(sig) - 1
    B = np.diag(np.diag(Q)[list(sig)])
    B[np.arange(m), np.arange(1, m + 1)] = 1.0
    return B


def skeleton_prob(Q, sig, T, PT):
    m = len(sig) - 1
    rates = np.prod([Q[sig[k], sig[k + 1]] for k in range(m)]) if m else 1.0
    return rates * sl.expm(T * skeleton_matrix(Q, sig))[0, m] / PT


def most_likely_skeleton(Q, x, y, T, top=1, grid=500, max_nodes=5000000, margin=1e-3):
    import heapq
    import networkx as nx
    from math import lgamma, log
    n = len(Q)
    q = -np.diag(Q)
    h = T / grid
    u = np.linspace(0.0, T, grid + 1)
    wts = np.full(grid + 1, h); wts[[0, -1]] = h / 2
    PT = sl.expm(T * Q)[x, y]
    zs = np.concatenate([[0.0], np.geomspace(1e-2, 20.0, 40) * q.max()])
    K = wts[None, :] * np.exp(-zs[:, None] * u[None, :])
    dist = np.full((len(zs), n), np.inf)
    for a_, z in enumerate(zs):
        G = nx.DiGraph()
        for i, j in zip(*np.nonzero(Q)):
            if i != j:
                G.add_edge(int(j), int(i), w=-log(Q[i, j] / (q[i] + z)))
        for s, dd in nx.single_source_dijkstra_path_length(G, y, weight='w').items():
            dist[a_, s] = dd
    logPT = log(PT)

    def bound(a, s):
        v = zs * T + np.log(np.maximum(K @ a, 1e-300)) - dist[:, s]
        return float(np.exp(v.min() - logPT)) * (1 + margin)

    def advance(a, s):
        decay = np.exp(-q[s] * h)
        b = np.zeros(grid + 1)
        b[1:] = lfilter([1.0], [1.0, -decay], (h / 2) * (a[:-1] * decay + a[1:]))
        return b

    found, kept = {}, []

    def record(sg, val):
        if len(kept) < top:
            heapq.heappush(kept, val); found[sg] = val
        elif val > kept[0]:
            heapq.heapreplace(kept, val); found[sg] = val

    best = lambda: kept[0] if len(kept) >= top else 0.0
    b0 = np.exp(-q[x] * u)
    if x == y:
        record((x,), b0[-1] / PT)
    root = float(np.exp((zs * T - dist[:, x]).min() - logPT))
    def profile_of(sig):
        b = b0
        for s0, s1 in zip(sig[:-1], sig[1:]):
            b = advance(Q[s0, s1] * b, s1)
        return b

    heap = [(-root, (x,))]
    nodes = 0
    while heap and nodes < max_nodes:
        nb, sig = heapq.heappop(heap)
        if -nb <= best():
            break
        nodes += 1
        s = sig[-1]
        b = profile_of(sig)
        for j in np.nonzero(Q[s])[0]:
            if j == s:
                continue
            a = Q[s, j] * b
            bd = bound(a, j)
            if bd <= best():
                continue
            child = sig + (int(j),)
            if j == y:
                record(child, advance(a, j)[-1] / PT)
            heapq.heappush(heap, (-bd, child))
    gap = -heap[0][0] if heap else 0.0
    ranked = sorted(((sg, skeleton_prob(Q, sg, T, PT)) for sg in found), key=lambda t: -t[1])[:top]
    from scipy.stats import poisson
    M = 0
    while poisson.sf(M, q.max() * T) >= PT * ranked[0][1]:
        M += 1
    return ranked, M, nodes, gap


def jump_time_profile(Q, sig, T, nodes=80):
    m = len(sig) - 1
    if m == 0:
        return np.array([])
    B = skeleton_matrix(Q, sig)
    g, w = np.polynomial.legendre.leggauss(nodes)
    u, w = T * (g + 1) / 2, T * w / 2
    E = np.array([sl.expm(t * B) for t in u])
    Er = E[::-1]
    num = np.array([(w * u * E[:, 0, j - 1] * Er[:, j, m]).sum() for j in range(1, m + 1)])
    den = sl.expm(T * B)[0, m]
    return num / den


def gradient(M, v, data):
    N, A, _ = estep(M, v, data)
    return N / v - A[M.I]


def foc_residual(M, v, data, lam, tol=1e-6):
    g = gradient(M, v, data)
    lin = M.Amat @ v
    fixed = np.abs(lin) > tol * np.abs(lin).max()
    sig = cp.Variable(len(lin))
    cons = [sig[fixed] == np.sign(lin[fixed])] if fixed.any() else []
    if (~fixed).any():
        cons += [cp.abs(sig[~fixed]) <= 1]
    r = g - lam * (M.Amat.T @ cp.multiply(M.om, sig))
    cp.Problem(cp.Minimize(cp.norm(r)), cons).solve(solver=cp.CLARABEL)
    return float(np.linalg.norm(g - lam * M.Amat.T @ (M.om * sig.value))), float(np.linalg.norm(g))
