"""Shared models: the Pair-only low-rank predictor with its Laplace posterior, the GP on single effects,
the training-pair graph, and small helpers."""
from collections import deque

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import rankdata

LAMS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


class PairGraph:
    """Graph of training pairs; classifies a test pair by how its genes are connected."""

    def __init__(self, gene_pairs):
        self.adj = {}
        for a, b in gene_pairs:
            self.adj.setdefault(a, set()).add(b)
            self.adj.setdefault(b, set()).add(a)
        self.comp, self.side, self.bip = {}, {}, {}
        cid = 0
        for s in self.adj:
            if s in self.comp:
                continue
            ok, self.comp[s], self.side[s] = True, cid, 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in self.adj[u]:
                    if v not in self.comp:
                        self.comp[v], self.side[v] = cid, 1 - self.side[u]
                        q.append(v)
                    elif self.side[v] == self.side[u]:
                        ok = False
            self.bip[cid] = ok
            cid += 1

    def classify(self, a, b):
        if a not in self.comp or b not in self.comp:
            return "isolated_gene"
        if self.comp[a] != self.comp[b]:
            return "cross_component"
        if self.bip[self.comp[a]] and self.side[a] == self.side[b]:
            return "bipartite_same_side"
        return "identified"


def lowrank(n, a, b, y, rank, lam, sweeps=25, seed=0):
    """Symmetric z_ab = c + beta_a + beta_b + u_a . u_b fitted by alternating ridge on observed pairs."""
    rng = np.random.default_rng(seed)
    U, beta, c = 0.1 * rng.standard_normal((n, rank)), np.zeros(n), float(np.mean(y))
    nb = [[] for _ in range(n)]
    for k, (i, j) in enumerate(zip(a, b)):
        nb[i].append((j, k))
        nb[j].append((i, k))
    nb = [(np.array([h for h, _ in L], int), np.array([k for _, k in L], int)) for L in nb]
    U[[len(h) == 0 for h, _ in nb]] = 0.0
    reg = lam * np.eye(rank + 1)
    for _ in range(sweeps):
        for g in range(n):
            h, k = nb[g]
            if len(h) == 0:
                continue
            X = np.hstack([np.ones((len(h), 1)), U[h]])
            w = np.linalg.solve(X.T @ X + reg, X.T @ (y[k] - c - beta[h]))
            beta[g], U[g] = w[0], w[1:]
        c = float(np.mean(y - beta[a] - beta[b] - np.sum(U[a] * U[b], 1)))
    f = lambda p, q: c + beta[p] + beta[q] + np.sum(U[p] * U[q], 1)
    f.state = (c, beta, U)
    return f


def lowrank_laplace(n, a, b, ta, tb, sigma, lam, state):
    """Laplace posterior of f = c + beta_a + beta_b + u_a . u_b at test pairs.
    Returns additive part m, interaction part u, sd at the mode and sd at the sign flipped mode."""
    c, beta, U = state
    r = U.shape[1]
    P = 1 + n + n * r

    def jac(p, q, s=1.0):
        k = len(p)
        rows = [np.arange(k)] * 3
        cols = [np.zeros(k, int), 1 + p, 1 + q]
        vals = [np.ones(k)] * 3
        for j in range(r):
            rows += [np.arange(k)] * 2
            cols += [1 + n + p * r + j, 1 + n + q * r + j]
            vals += [s * U[q, j], s * U[p, j]]
        return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(k, P))

    J = jac(a, b)
    cho = cho_factor((J.T @ J).toarray() + np.diag(np.r_[1e-8, np.full(P - 1, lam)]))
    sd = []
    for s in (1.0, -1.0):
        G = jac(ta, tb, s).toarray()
        sd.append(sigma * np.sqrt(np.maximum(np.sum(G * cho_solve(cho, G.T).T, 1), 1e-12)))
    return c + beta[ta] + beta[tb], np.sum(U[ta] * U[tb], 1), sd[0], sd[1]


def auroc(score, y):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def pair_features(Xg, a, b):
    return np.hstack([Xg[a] + Xg[b], Xg[a] * Xg[b]])


def onehot(n, a, b):
    M = np.zeros((len(a), n))
    M[np.arange(len(a)), a] += 1
    M[np.arange(len(a)), b] += 1
    return M


def cv_lambda(fit_predict, a, b, y, lams=LAMS, folds=5, seed=0):
    perm = np.random.default_rng(seed).permutation(len(y))
    best = None
    for lam in lams:
        err = 0.0
        for f in range(folds):
            te = perm[f::folds]
            tr = np.setdiff1d(perm, te)
            if len(te) == 0 or len(tr) < 3:
                continue
            err += float(np.sum((fit_predict(a[tr], b[tr], y[tr], lam, a[te], b[te]) - y[te]) ** 2))
        if best is None or err < best[0]:
            best = (err, lam)
    return best[1]


def gp_fit(Ftr, ytr, Fte, sigma, s2=None):
    d2 = lambda P, Q: np.sum(P ** 2, 1)[:, None] + np.sum(Q ** 2, 1)[None] - 2 * P @ Q.T
    D = d2(Ftr, Ftr)
    gk = 1.0 / max(np.median(D[D > 0]), 1e-12)
    K, Ks = np.exp(-gk * D), np.exp(-gk * d2(Fte, Ftr))
    y0 = ytr.mean()
    yc = ytr - y0
    best = None
    for s in ([s2] if s2 is not None else sigma ** 2 * np.logspace(-2, 3, 26)):
        L = np.linalg.cholesky(s * K + sigma ** 2 * np.eye(len(K)))
        al = np.linalg.solve(L.T, np.linalg.solve(L, yc))
        ll = -0.5 * yc @ al - np.sum(np.log(np.diag(L)))
        if best is None or ll > best[0]:
            best = (ll, s, L, al)
    _, s, L, al = best
    W = np.linalg.solve(L, s * Ks.T)
    return y0 + s * Ks @ al, np.sqrt(np.maximum(s - np.sum(W ** 2, 0), 1e-12)), s
