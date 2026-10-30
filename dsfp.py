"""DSFP: Dyadic Sign-Flip Process, reliability of sign predictions for unseen gene pairs.

Fixed predictor: Pair-only rank-2 low-rank model (its signs are never changed).
Error model on the 4-fold out-of-fold training errors e:
    e_x = g(x) + eta_x + s_x xi_x,   g ~ GP(0, tau_S^2 K_S + tau_g^2 K_g),   s_x = sign(mu_x)
    K_S: single-effect pair kernel,  K_g: shared genes,  eta ~ N(0, tau_eta^2),  xi_x ~ N(beta |mu_x|, omega^2)
Reliability: R(x) = P(s_x z_x > 0) = Phi(((1 + beta)|mu_x| + s_x m_g(x)) / sqrt(V_x + omega^2)).
Fitting: (tau_S, ell, tau_g, tau_eta) by Gaussian marginal likelihood; (beta, omega) by leave-one-out predictive
likelihood; the two groups are updated alternately.  Training pairs only.

Modes
  --confirm   fit on each training split and score its untouched test pairs (paper: seeds 101..105)
  default     development: nested CV inside each training split, test pairs unused (paper: seeds 42,43,44)

Usage:  python dsfp.py --confirm [--datasets norman,directcapture,wessels2023,gse164996]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import json
import time

import numpy as np
from scipy.special import owens_t as _owens_t
from scipy.stats import norm, rankdata

import data as D
import models as M
from data import DATASETS
from reliability import prepare


def _solve(A, B):
    try:
        return np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, B, rcond=None)[0]


# ---------------------------------------------------------------- channels
def fit_pair(n, a, b, y, rank):
    lam = M.cv_lambda(lambda a_, b_, y_, l, p, q: M.lowrank(n, a_, b_, y_, rank, l)(p, q), a, b, y)
    return M.lowrank(n, a, b, y, rank, lam), lam


def pair_predict(n, a, b, y, p, q, sigma, rank=2):
    f, lam = fit_pair(n, a, b, y, rank)
    m, u, sp, _ = M.lowrank_laplace(n, a, b, p, q, sigma, lam, f.state)
    return np.asarray(m + u, float), np.asarray(sp, float)


def ridge_predict(n, a, b, y, p, q):
    f, _ = fit_pair(n, a, b, y, 0)
    return np.asarray(f(p, q), float).reshape(-1)


def single_predict(Xg, a, b, y, p, q, sigma, return_sd=False):
    out = M.gp_fit(M.pair_features(Xg, a, b), y, M.pair_features(Xg, p, q), sigma)
    mu = np.asarray(out[0], float).reshape(-1)
    if not return_sd:
        return mu
    for o in out[1:]:
        v = np.asarray(o, float).reshape(-1)
        if v.shape == mu.shape and np.all(np.isfinite(v)) and np.all(v >= 0):
            return mu, v
    raise RuntimeError("could not identify the GP posterior sd among gp_fit outputs")


# ---------------------------------------------------------------- skew-normal (alpha = 0 gives the Gaussian DSFP)
def sn_lambda(alpha):
    return alpha / np.sqrt(1.0 + alpha ** 2)


def sn_moments(beta, omega, alpha, absmu):
    lam = sn_lambda(alpha)
    return beta * absmu + omega * lam * np.sqrt(2 / np.pi), omega ** 2 * (1 - 2 * lam ** 2 / np.pi)


def flip_prob(sM, V, absmu, beta, omega, alpha):
    """P(y > -|mu|), y ~ SN(sM + beta|mu|, sqrt(V + omega^2), lambda* = omega lambda / Omega)."""
    Om = np.sqrt(np.maximum(V, 1e-12) + omega ** 2)
    lam_s = np.clip(omega * sn_lambda(alpha) / Om, -0.999999, 0.999999)
    a_s = lam_s / np.sqrt(1 - lam_s ** 2)
    t = (-absmu - (sM + beta * absmu)) / Om
    F = norm.cdf(t) - 2 * _owens_t(t, a_s)
    return np.clip(1.0 - F, 1e-9, 1 - 1e-9)


def sn_logpdf(y, sM, V, absmu, beta, omega, alpha):
    Om = np.sqrt(np.maximum(V, 1e-12) + omega ** 2)
    lam_s = np.clip(omega * sn_lambda(alpha) / Om, -0.999999, 0.999999)
    a_s = lam_s / np.sqrt(1 - lam_s ** 2)
    t = (y - (sM + beta * absmu)) / Om
    return np.log(2.0) - np.log(Om) + norm.logpdf(t) + norm.logcdf(a_s * t)


# ---------------------------------------------------------------- kernels
def pair_kernel(Xg, pa, pb, ell):
    def kg(i, j):
        d2 = ((Xg[i][:, None, :] - Xg[j][None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * ell ** 2))
    return 0.5 * (kg(pa, pa) * kg(pb, pb) + kg(pa, pb) * kg(pb, pa))


def gene_kernel(n, pa, pb):
    Z = np.zeros((len(pa), n))
    Z[np.arange(len(pa)), pa] += 1.0
    Z[np.arange(len(pa)), pb] += 1.0
    return Z @ Z.T


def gauss_loglik(C, e):
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        return -np.inf
    al = np.linalg.solve(L, e)
    return -0.5 * al @ al - np.log(np.diag(L)).sum() - 0.5 * len(e) * np.log(2 * np.pi)


# ---------------------------------------------------------------- temperature scaling
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def fit_temperature(p, correct):
    """T > 0 minimising the NLL of sigmoid(logit(p) / T) for the event `correct` (log-grid, then golden refinement)."""
    L, c = _logit(np.asarray(p, float)), np.asarray(correct, bool)
    def nll(logT):
        q = np.clip(1.0 / (1.0 + np.exp(-L / np.exp(logT))), 1e-6, 1 - 1e-6)
        return -np.mean(np.where(c, np.log(q), np.log(1 - q)))
    grid = np.linspace(np.log(0.02), np.log(1000.0), 161)
    vals = [nll(g) for g in grid]
    i = int(np.argmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    g = (np.sqrt(5) - 1) / 2
    for _ in range(60):
        x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
        if nll(x1) < nll(x2):
            hi = x2
        else:
            lo = x1
    return float(np.exp((lo + hi) / 2))


def apply_temperature(p, T):
    return 1.0 / (1.0 + np.exp(-_logit(np.asarray(p, float)) / T))


# ---------------------------------------------------------------- DSFP
class DSFP:
    """One fit on a training fold. Kp[ell], Kg: kernels over the joint index (training first, then targets)."""

    BETAS = (-1.0, -0.5, 0.0, 0.5)
    OMEGAS = (0.0, 0.3, 0.6, 1.0, 1.5)          # multiples of sd of the OOF residuals
    ALPHAS = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)

    def __init__(self, Kp, Kg, e, sgn, absmu, correct, s2, var_e, use_gene=True, use_alpha=False, use_beta=True,
                 dec_objective="density", rounds=2, betas=None, use_single=True, use_omega=True):
        self.betas = tuple(betas) if betas is not None else self.BETAS
        self.use_single, self.use_omega = use_single, use_omega
        self.Kp, self.Kg, self.e, self.sgn, self.absmu, self.correct = Kp, Kg, e, sgn, absmu, correct
        self.m, self.s2, self.var_e = len(e), s2, var_e
        self.use_gene, self.use_alpha, self.use_beta, self.obj = use_gene, use_alpha, use_beta, dec_objective
        self.dec = (0.0, 0.0, 0.0)                   # start at A2
        for _ in range(rounds):
            self.rep = self._fit_rep()
            self.dec = self._fit_dec()
        self.rep = self._fit_rep()

    def _resid(self, dec):
        beta, om, al = dec
        mean_xi, var_xi = sn_moments(beta, om, al, self.absmu)
        return self.e - self.sgn * mean_xi, var_xi

    def _C(self, rep, var_xi, idx=slice(None)):
        t2, ell, tg, teta = rep
        m = self.m
        K = t2 * self.Kp[ell][:m, :m] + tg * self.Kg[:m, :m]
        return K + (teta + var_xi + 1e-9 * self.var_e) * np.eye(m)

    def _fit_rep(self):
        r, var_xi = self._resid(self.dec)
        best = (-np.inf, None)
        tgs = (0.0, 0.1, 0.3, 1.0) if self.use_gene else (0.0,)
        ells = list(self.Kp) if self.use_single else list(self.Kp)[:1]       # ell is irrelevant when tau_S = 0
        t2s = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0) if self.use_single else (0.0,)
        for ell in ells:
            for t2 in self.var_e * np.array(t2s):
                for tg in self.var_e * np.array(tgs):
                    for teta in self.s2 * np.array([0.5, 1.0, 2.0, 4.0]) + self.var_e * 0.0:
                        for t0 in self.var_e * np.array([0.0, 0.1, 0.3, 1.0]):
                            rep = (t2, ell, tg, teta + t0)
                            ll = gauss_loglik(self._C(rep, var_xi), r)
                            if ll > best[0]:
                                best = (ll, rep)
        return best[1]

    def _loo(self, dec):
        r, var_xi = self._resid(dec)
        Ci = np.linalg.inv(self._C(self.rep, var_xi))
        d = np.diag(Ci)
        M = r - (Ci @ r) / d
        V = np.maximum(1.0 / d - var_xi, 1e-12)
        return M, V

    def _fit_dec(self):
        sd = np.sqrt(self.var_e)
        betas = self.betas if self.use_beta else (0.0,)
        alphas = self.ALPHAS if self.use_alpha else (0.0,)
        best = (-np.inf, self.dec)
        y = self.sgn * self.e
        for be in betas:
            for om in sd * np.array(self.OMEGAS if self.use_omega else (0.0,)):
                for al in alphas:
                    if om == 0.0 and al != 0.0:
                        continue
                    dec = (be, om, al)
                    M, V = self._loo(dec)
                    if self.obj == "event":
                        p = flip_prob(self.sgn * M, V, self.absmu, *dec)
                        sc = float(np.mean(np.where(self.correct, np.log(p), np.log(1 - p))))
                    else:                                # skew-normal residual density (ablation)
                        sc = float(np.mean(sn_logpdf(y, self.sgn * M, V, self.absmu, *dec)))
                    if sc > best[0]:
                        best = (sc, dec)
        return best[1]

    def predict(self, sgn_X, absmu_X):
        m = self.m
        t2, ell, tg, teta = self.rep
        r, var_xi = self._resid(self.dec)
        Kfull = t2 * self.Kp[ell] + tg * self.Kg
        C = self._C(self.rep, var_xi)
        KXT = Kfull[m:, :m]
        W = _solve(C, KXT.T).T
        M = W @ r
        V = np.diag(Kfull)[m:] + teta - np.sum(W * KXT, 1)
        return flip_prob(sgn_X * M, V, absmu_X, *self.dec)


# ---------------------------------------------------------------- one outer fold
N_PERM = 5          # negative control: random reassignments of single-effect features among genes


def outer_fold(n, Xg, a_T, b_T, y_T, a_X, b_X, sigma, n_inner, rng, exploratory=False):
    m, s2 = len(y_T), sigma ** 2
    muP_X, sp_X = pair_predict(n, a_T, b_T, y_T, a_X, b_X, sigma)
    muS_X, sdS_X = single_predict(Xg, a_T, b_T, y_T, a_X, b_X, sigma, return_sd=True)
    muR_X = ridge_predict(n, a_T, b_T, y_T, a_X, b_X)

    muP_O, sP_O = np.zeros(m), np.zeros(m)
    muS_O, sdS_O = np.zeros(m), np.zeros(m)
    for F in np.array_split(rng.permutation(m), n_inner):
        k = np.setdiff1d(np.arange(m), F)
        muP_O[F], sP_O[F] = pair_predict(n, a_T[k], b_T[k], y_T[k], a_T[F], b_T[F], sigma)
        muS_O[F], sdS_O[F] = single_predict(Xg, a_T[k], b_T[k], y_T[k], a_T[F], b_T[F], sigma, return_sd=True)
    e_O = y_T - muP_O
    sgn = lambda v: np.where(v >= 0, 1.0, -1.0)
    sO, sX = sgn(muP_O), sgn(muP_X)
    absO, absX = np.abs(muP_O), np.abs(muP_X)
    correct_O = np.sign(y_T) == sO

    pa, pb = np.r_[a_T, a_X], np.r_[b_T, b_X]
    dd = np.sqrt(((Xg[:, None, :] - Xg[None, :, :]) ** 2).sum(-1))
    med_d = float(np.median(dd[dd > 0])) if np.any(dd > 0) else 1.0
    Kp = {ell: pair_kernel(Xg, pa, pb, ell) for ell in med_d * np.array([0.3, 1.0, 3.0])}
    Kg = gene_kernel(n, pa, pb)
    Kg = Kg / max(float(np.mean(np.diag(Kg))), 1e-12)          # unit mean prior variance
    var_e = float(np.var(e_O)) + 1e-12
    common = (Kp, Kg, e_O, sO, absO, correct_O, s2, var_e)

    # paper rows: Residual GP, the three single-component ablations and DSFP
    variants = {
        "A2 residual GP": dict(rounds=0),
        "DSFP w/o gene kernel": dict(use_gene=False),
        "DSFP w/o direction (beta=0)": dict(use_beta=False),
        "DSFP w/o single-effect kernel": dict(use_single=False),
        "DSFP (ours)": dict(),
    }
    if exploratory:                                             # further variants, not reported in the paper
        variants.update({"DSFP + skew (alpha)": dict(use_alpha=True),
                         "DSFP event-risk decision": dict(dec_objective="event"),
                         "DSFP w/o directional variance": dict(use_omega=False)})
    scores, diag = {}, {}
    d2 = max(0.0, float(np.mean(e_O ** 2) - s2 - np.mean(sP_O ** 2)))
    scores["Noise+Error Laplace"] = norm.cdf(absX / np.sqrt(sp_X ** 2 + s2 + d2))
    scores["Model Agreement"] = (1 + (sgn(muS_X) == sX) + (sgn(muR_X) == sX)) / 3.0
    # probabilistic agreement: posterior probability, under the GP on single effects, that z has the Pair-only sign
    scores["GP-singles sign posterior"] = norm.cdf(sX * muS_X / np.sqrt(sdS_X ** 2 + s2))
    # same posterior, temperature fitted by NLL on the out-of-fold training predictions (test pairs untouched)
    p_O = norm.cdf(sO * muS_O / np.sqrt(sdS_O ** 2 + s2))
    T_gp = fit_temperature(p_O, correct_O)
    scores["GP-singles sign posterior + temperature"] = apply_temperature(scores["GP-singles sign posterior"], T_gp)
    for name, kw in variants.items():
        if kw.get("rounds", 2) == 0:                            # A2: representation layer only, Gaussian
            mdl = DSFP(*common, use_gene=False, use_alpha=False, use_beta=False, rounds=0)
            mdl.dec = (0.0, 0.0, 0.0)
            mdl.rep = mdl._fit_rep()
        else:
            mdl = DSFP(*common, **kw)
        scores[name] = mdl.predict(sX, absX)
        if name == "DSFP (ours)":
            t2, ell, tg, teta = mdl.rep
            diag = {"tau_g2": float(tg / var_e), "tau_S2": float(t2 / var_e), "beta": float(mdl.dec[0]),
                    "omega": float(mdl.dec[1] / np.sqrt(var_e)), "alpha": float(mdl.dec[2])}

    # negative control (paper): the single-effect features phi_g used by K_S are randomly reassigned among
    # genes (the same permutation for training and test pairs); predictor, OOF errors, labels, splits and K_g unchanged
    rng_perm = np.random.default_rng(7919 + int(1e6 * float(np.abs(y_T).sum())) % (2 ** 31))   # deterministic, main rng untouched
    for r_ in range(N_PERM):
        Xp = Xg[rng_perm.permutation(Xg.shape[0])]
        Kp_perm = {ell: pair_kernel(Xp, pa, pb, ell) for ell in Kp}            # same length-scale grid
        mdl = DSFP(Kp_perm, Kg, e_O, sO, absO, correct_O, s2, var_e)
        scores[f"DSFP shuffled single-gene features #{r_}"] = mdl.predict(sX, absX)

    beta_ols = float(muP_O @ e_O / max(muP_O @ muP_O, 1e-12))
    diag["beta_ols"] = beta_ols
    diag["T_gp"] = T_gp
    if not exploratory:
        return muP_X, scores, diag

    # in-sample residuals instead of out-of-fold residuals: errors of the full-training-set fit on its own pairs
    muP_in, _ = pair_predict(n, a_T, b_T, y_T, a_T, b_T, sigma)
    e_in = y_T - muP_in
    s_in, abs_in = sgn(muP_in), np.abs(muP_in)
    mdl = DSFP(Kp, Kg, e_in, s_in, abs_in, np.sign(y_T) == s_in, s2, float(np.var(e_in)) + 1e-12)
    scores["DSFP in-sample residuals"] = mdl.predict(sX, absX)

    # two-stage controls: calibration slope by OLS of the OOF residual on mu, then a Gaussian residual GP
    # on the recalibrated residual; differs from DSFP only in how beta is fitted (separately vs jointly)
    for name, gene in (("Slope calib. + residual GP", False), ("Slope calib. + residual GP + gene", True)):
        mdl = DSFP(*common, use_gene=gene, use_alpha=False, use_beta=False, rounds=0)
        mdl.dec = (beta_ols, 0.0, 0.0)
        mdl.rep = mdl._fit_rep()
        scores[name] = mdl.predict(sX, absX)
    return muP_X, scores, diag


# ---------------------------------------------------------------- metrics
def selective_metrics(p, correct):
    p, correct = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6), np.asarray(correct, bool)
    n, npos = len(correct), int(correct.sum())
    au = np.nan
    if 0 < npos < n:
        r = rankdata(p)
        au = (r[correct].sum() - npos * (npos + 1) / 2) / (npos * (n - npos))
    order = np.argsort(-p, kind="mergesort")
    risk = np.cumsum(~correct[order]) / np.arange(1, n + 1)
    sr = lambda c: float(risk[max(1, int(np.ceil(c * n))) - 1])
    return {"AUROC": float(au), "AURC": float(risk.mean()), "SR@50%": sr(0.5), "SR@80%": sr(0.8),
            "Brier": float(np.mean((p - correct) ** 2)),
            "NLL": float(-np.mean(np.where(correct, np.log(p), np.log(1 - p))))}


def paired(per_block, k, ref, mt, n_boot=4000, seed=0):
    """Paired difference k - ref over blocks, bootstrap CI over blocks."""
    rng = np.random.default_rng(seed)
    d = np.array([pb["metrics"][k][mt] - pb["metrics"][ref][mt] for pb in per_block], float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan, np.nan, np.nan, 0, 0
    boots = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    better = int(np.sum(d > 1e-9)) if mt == "AUROC" else int(np.sum(d < -1e-9))
    return float(d.mean()), float(lo), float(hi), better, len(d)


def cluster_paired(per_block, k, ref, mt, n_boot=4000, seed=0):
    """Paired difference k - ref; the two programs of one split seed are averaged first, then seeds are resampled."""
    by_seed = {}
    for pb in per_block:
        d = pb["metrics"][k][mt] - pb["metrics"][ref][mt]
        if np.isfinite(d):
            by_seed.setdefault(pb["seed"], []).append(d)
    u = np.array([np.mean(v) for v in by_seed.values()], float)
    if len(u) == 0:
        return np.nan, np.nan, np.nan, 0, 0
    rng = np.random.default_rng(seed)
    bs = [u[rng.integers(0, len(u), len(u))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    better = int(np.sum(u > 1e-9)) if mt == "AUROC" else int(np.sum(u < -1e-9))
    return float(u.mean()), float(lo), float(hi), better, len(u)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--confirm", action="store_true",
                    help="confirmatory evaluation: fit on each training split, score its untouched test pairs")
    ap.add_argument("--seeds", default=None, help="split seeds (default 101..105 with --confirm, else 42,43,44)")
    ap.add_argument("--outer", type=int, default=5, help="development mode: outer folds inside the training split")
    ap.add_argument("--inner", type=int, default=4, help="out-of-fold folds for the training errors")
    ap.add_argument("--exploratory", action="store_true", help="also fit variants that the paper does not report")
    ap.add_argument("--programs", type=int, default=2)
    ap.add_argument("--features", type=int, default=10)
    ap.add_argument("--splits", type=int, default=5, help="random cell-half replicates for sigma")
    D.add_data_args(ap)
    ap.add_argument("--out", default=None)
    cli = D.resolve_data_args(ap.parse_args())
    cli.seeds = cli.seeds or ("101,102,103,104,105" if cli.confirm else "42,43,44")
    cli.out = cli.out or ("results/dsfp_confirm.json" if cli.confirm else "results/dsfp_dev.json")
    results = {}
    for d in D.parse_list(cli.datasets):
        if cli.confirm:
            print(f"\n===== {d} CONFIRMATORY (seeds {cli.seeds}; fit on training split, scored on its test pairs) =====",
                  flush=True)
        else:
            print(f"\n===== {d} (nested CV inside training splits; w, mean, sigma refit per outer fold; test unused) =====",
                  flush=True)
        args = argparse.Namespace(**vars(cli))
        data, genes, Xg, blocks, reps = prepare(d, args)
        Xg = np.asarray(Xg, float)
        parse = data["pair_genes_fn"]
        gi = {g: i for i, g in enumerate(genes)}
        built, per_block = {}, []
        for b in blocks:
            t0 = time.time()
            n, k_prog, key = b["n"], int(b["program"]), int(b["seed"])
            if key not in built:
                tr_c, base_, y_, _ = D.build_pairs(data, b["train"])
                built[key] = (list(tr_c), np.asarray(y_, float) - np.asarray(base_, float))
            tr_c, r_full = built[key]
            ai = np.array([gi[parse(p)[0]] for p in tr_c]); bi = np.array([gi[parse(p)[1]] for p in tr_c])
            ref = (r_full - b["mean_tr"]) @ b["w"]
            if not (np.array_equal(ai, np.asarray(b["tr"][0])) and np.allclose(ref, b["tr"][2], atol=1e-6 * (np.std(ref) + 1e-12))):
                raise RuntimeError("recomputed training targets do not match prepare(); pair order differs")
            rng = np.random.default_rng(1000 * key + k_prog)
            S, C, diags = {}, [], []
            if cli.confirm:                              # w, mean_tr, sigma from prepare() use training pairs only
                ta, tb, tz = (np.asarray(v) for v in b["test"])
                muP_X, sc, dg = outer_fold(n, Xg, ai, bi, np.asarray(b["tr"][2], float), ta.astype(int),
                                           tb.astype(int), float(b["sigma"]), cli.inner, rng, cli.exploratory)
                C.append(np.sign(muP_X) == np.sign(np.asarray(tz, float)))
                for k_, v in sc.items():
                    S.setdefault(k_, []).append(v)
                diags.append(dg)
            for X in ([] if cli.confirm else np.array_split(rng.permutation(len(tr_c)), cli.outer)):
                T = np.setdiff1d(np.arange(len(tr_c)), X)
                mean_T = r_full[T].mean(0)
                wT = np.linalg.svd(r_full[T] - mean_T, full_matrices=False)[2][k_prog]
                if wT @ b["w"] < 0:
                    wT = -wT
                y_T, y_X = (r_full[T] - mean_T) @ wT, (r_full[X] - mean_T) @ wT
                pT = [tr_c[i] for i in T]
                sigma_T = float(np.mean([np.sqrt(np.mean([(((h[p][0] - h[p][1]) / 2) @ wT) ** 2 for p in pT])) for h in reps]))
                muP_X, sc, dg = outer_fold(n, Xg, ai[T], bi[T], y_T, ai[X], bi[X], sigma_T, cli.inner, rng,
                                           cli.exploratory)
                C.append(np.sign(muP_X) == np.sign(y_X))
                for k_, v in sc.items():
                    S.setdefault(k_, []).append(v)
                diags.append(dg)
            correct = np.concatenate(C)
            mets = {k_: selective_metrics(np.concatenate(v), correct) for k_, v in S.items()}
            per_block.append({"seed": key, "program": k_prog, "n": int(len(correct)), "acc": float(correct.mean()),
                              "metrics": mets, "diag": diags,
                              # per-pair outputs, kept so that any later metric needs no rerun
                              "correct": [bool(x) for x in correct],
                              "scores": {k_: [float(x) for x in np.concatenate(v)] for k_, v in S.items()}})
            avg = lambda f: np.mean([x[f] for x in diags])
            print(f"  seed {key} program {k_prog}: n={len(correct)} acc={correct.mean():.2f} ({time.time() - t0:.0f}s) | "
                  f"tau_g2 {avg('tau_g2'):.2f}  tau_S2 {avg('tau_S2'):.2f} | beta {avg('beta'):+.2f} (OLS {avg('beta_ols'):+.2f})  "
                  f"omega {avg('omega'):.2f}  T_gp {avg('T_gp'):.2f}", flush=True)
            for k_ in ("A2 residual GP", "DSFP (ours)"):
                v = mets[k_]
                print(f"      {k_:<28} AUROC {v['AUROC']:.3f}  AURC {v['AURC']:.3f}  Brier {v['Brier']:.3f}  NLL {v['NLL']:.3f}")

        methods = list(per_block[0]["metrics"].keys())
        mts = ("AUROC", "AURC", "SR@50%", "SR@80%", "Brier", "NLL")
        print(f"\n  ===== {d}: mean over {len(per_block)} blocks{' (CONFIRMATORY, test pairs)' if cli.confirm else ''} =====")
        print(f"  {'method':<28}" + "".join(f"{x:>8}" for x in mts))
        summary = {}
        for k_ in methods:
            row = {mt: float(np.nanmean([pb["metrics"][k_][mt] for pb in per_block])) for mt in mts}
            summary[k_] = row
            print(f"  {k_:<28}" + "".join(f"{row[x]:>8.3f}" for x in mts))
        print("\n  paired difference vs A2 residual GP: mean [95% bootstrap CI over blocks] (#blocks better)")
        for k_ in methods:
            if k_ == "A2 residual GP":
                continue
            parts = []
            for mt in ("AUROC", "AURC", "NLL"):
                mean_, lo, hi, bt, nb = paired(per_block, k_, "A2 residual GP", mt)
                parts.append(f"{mt} {mean_:+.3f} [{lo:+.3f}, {hi:+.3f}] ({bt}/{nb})")
                summary[k_][f"d{mt}_vs_A2"] = [mean_, lo, hi]
            print(f"  {k_:<28} " + "   ".join(parts))
        checks = [("DSFP (ours)", "GP-singles sign posterior + temperature"),
                  ("GP-singles sign posterior + temperature", "GP-singles sign posterior")]
        print("\n  seed-clustered paired differences (programs of a seed averaged; seeds resampled) (#seeds better)")
        for k_, ref_ in checks:
            parts = []
            for mt in ("AUROC", "Brier", "NLL"):
                mean_, lo, hi, bt, nb = cluster_paired(per_block, k_, ref_, mt)
                parts.append(f"{mt} {mean_:+.3f} [{lo:+.3f}, {hi:+.3f}] ({bt}/{nb})")
                summary.setdefault(f"{k_} - {ref_}", {})[mt] = [mean_, lo, hi, bt, nb]
            print(f"  {k_} - {ref_}\n      " + "   ".join(parts))
        Ts = np.array([x["T_gp"] for pb in per_block for x in pb["diag"]], float)
        print(f"  GP-singles temperature T: median {np.median(Ts):.2f}  range [{Ts.min():.2f}, {Ts.max():.2f}]")
        results[d] = {"summary": summary, "blocks": per_block}
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    with open(cli.out, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nsaved {cli.out}")


if __name__ == "__main__":
    main()