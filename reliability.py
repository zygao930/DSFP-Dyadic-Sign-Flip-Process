"""Reliability baselines for sign predictions on unseen gene pairs (paper Table 1, baseline rows).

Every method scores the same fixed Pair-only predictions (rank-2 low-rank model, lambda chosen by 5-fold CV on
the training pairs). The calibration split is used only by the methods that need calibration.
Development seeds: 42,43,44.  Confirmatory seeds (paper): 101,102,103,104,105.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import json
import math

import numpy as np
from scipy.stats import norm

import data as D
import models as M
from data import DATASETS
from models import cv_lambda, gp_fit, onehot, pair_features

COL = {"norman": "Norman", "directcapture": "DirectCapture", "wessels2023": "Wessels", "gse164996": "GSE164996"}
ROWS = [("Pair-only", "magnitude", "Magnitude"),
        ("Pair-only", "bootstrap", "Bootstrap Stability"),
        ("Pair-only", "agreement", "Model Agreement"),
        ("Pair-only", "laplace", "Vanilla Laplace"),
        ("Pair-only", "noise_error", "Noise + Error Laplace"),
        ("Pair-only", "conformal", "Split Conformal"),
        ("Pair-only", "epicscore", "EPICSCORE (UAI 2025)"),
        ("Pair-only", "abstain", "\u0394-KNN-RLog (ICLR 2026)"),
        ("GP on Singles", "gp_magnitude", "Magnitude"),
        ("GP on Singles", "gp_posterior", "GP Posterior Confidence")]
# the other selectors of Heng et al. (ICLR 2026), and the in-sample reference used before the audit
ABSTAIN_ROWS = [("Pair-only", "abstain_knn", "\u0394-KNN (ICLR 2026)"),
                ("Pair-only", "abstain_mds", "\u0394-MDS (ICLR 2026)"),
                ("Pair-only", "abstain_mds_rlog", "\u0394-MDS-RLog (ICLR 2026)"),
                ("Pair-only", "abstain_insample", "\u0394-KNN-RLog, in-sample reference (ICLR 2026)")]
_i = [k for _, k, _ in ROWS].index("abstain") + 1
ALL_ROWS = ROWS[:_i] + ABSTAIN_ROWS + ROWS[_i:]      # Table 2 order: all selectors together
PROB = {"bootstrap", "agreement", "laplace", "noise_error", "gp_posterior"}
INTERVALS = ("conformal", "epicscore", "noise_error")


# ---------- shared data preparation (also used by prediction.py and dsfp.py) ----------

def prepare(name, args):
    """Per seed and program: indices, targets (interaction residual projected on a training program),
    replicate noise sigma, graph categories, single effect features. Halves give sigma and the ceiling.
    Returns data, genes, Xg (single-effect features per gene), blocks, reps (cell-half replicates)."""
    args.dataset = name
    data = D.load(args)
    D.validate_dataset(data, name)
    parse = data["pair_genes_fn"]
    reps = D.dataset_halves(args, data, list(data["comp_eligible"]), args.splits)
    genes = sorted({g for p in data["comp_eligible"] for g in parse(p)})
    gi = {g: i for i, g in enumerate(genes)}
    E = np.stack([data["single_effects"][g] for g in genes])
    u_, sv, _ = np.linalg.svd(E - E.mean(0), full_matrices=False)
    Xg = u_[:, : args.features] * sv[: args.features]
    Xg /= np.sqrt(np.mean(Xg ** 2))
    idx = lambda cs: (np.array([gi[parse(p)[0]] for p in cs]), np.array([gi[parse(p)[1]] for p in cs]))
    blocks = []
    for seed in D.parse_seeds(args.seeds):
        train, cal, test = D.split_pairs_60_20_20(list(data["comp_eligible"]), seed)
        tr_c, tr_base, tr_y, _ = D.build_pairs(data, train)
        r_tr = tr_y - tr_base
        mean_tr = r_tr.mean(0)
        graph = M.PairGraph([parse(p) for p in tr_c])
        sets = {}
        for split, pairs in (("cal", cal), ("test", test)):
            c_, base_, y_, _ = D.build_pairs(data, pairs)
            sets[split] = (list(c_), y_ - base_)
        for k, w in enumerate(np.linalg.svd(r_tr - mean_tr, full_matrices=False)[2][: args.programs]):
            sigma = float(np.mean([np.sqrt(np.mean([(((h[p][0] - h[p][1]) / 2) @ w) ** 2 for p in tr_c])) for h in reps]))
            b = {"seed": seed, "program": k, "w": w, "mean_tr": mean_tr, "sigma": sigma, "n": len(genes),
                 "tr": idx(tr_c) + ((r_tr - mean_tr) @ w,), "tr_pairs": list(tr_c), "train": train,
                 "cal_pairs_all": cal, "test_pairs_all": test}
            for split, (c_, r_) in sets.items():
                ta, tb = idx(c_)
                b[split] = (ta, tb, (r_ - mean_tr) @ w)
                b[split + "_pairs"] = c_
                b[split + "_cats"] = np.array([graph.classify(*parse(p)) for p in c_])
            te = sets["test"][0]
            b["ceiling"] = [float(np.mean([np.sign((h[p][0] - mean_tr) @ w) == np.sign((h[p][1] - mean_tr) @ w)
                                           for h in reps])) for p in te]
            blocks.append(b)
    return data, genes, Xg, blocks, reps










# ---------- reliability methods ----------

def conformal_q(res, alpha):
    n = len(res)
    k = math.ceil((n + 1) * (1 - alpha))
    return np.inf if k > n else float(np.sort(res)[k - 1])


def epicscore(Xc, res_c, Xt, alpha, rng):
    """Cruz-Cabezas et al. (UAI 2025), regression: score s = |y - g(x)|; a Bayesian model of s given x
    (GP, Normal predictive) is fitted on cal part 1; scores of cal part 2 are mapped through the predictive
    CDF; cutoff = conformal quantile of the mapped scores; half width = predictive quantile at the cutoff."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    n = len(res_c)
    n2 = max(1, int(round(0.3 * n)))
    perm = rng.permutation(n)
    i2, i1 = perm[:n2], perm[n2:]
    if len(i1) < 3:
        return np.full(len(Xt), np.inf)
    gp = GaussianProcessRegressor(ConstantKernel() * RBF() + WhiteKernel(), normalize_y=True,
                                  random_state=0).fit(Xc[i1], res_c[i1])
    m2, s2 = gp.predict(Xc[i2], return_std=True)
    t = conformal_q(norm.cdf((res_c[i2] - m2) / np.maximum(s2, 1e-9)), alpha)
    if not np.isfinite(t) or t >= 1:
        # too few calibration pairs for the conformal cutoff (needs >= ceil(1/alpha) - 1 in part 2):
        # fall back to the nominal predictive quantile; widths stay input adaptive, coverage is no longer guaranteed
        print(f"      EPICSCORE: {len(i2)} pairs in cal part 2 is too few for the conformal cutoff, "
              f"using the nominal {1 - alpha:.2f} predictive quantile", flush=True)
        t = 1.0 - alpha
    mt, st = gp.predict(Xt, return_std=True)
    return np.maximum(mt + st * norm.ppf(t), 0.0)


def abstain_scores(phi_c, correct_c, phi_t, rlog_c, rlog_t, k=25, shrink=0.1):
    """Heng et al. (ICLR 2026) selectors. Reference points phi_c with correct / wrong labels correct_c.
    Delta-KNN: negative mean log distance to the k nearest correct minus wrong reference points.
    Delta-MDS: Mahalanobis score (negative squared distance, class mean and shrunk class covariance)
    to the correct minus the wrong reference points.  X-RLog: X + lambda * RLog, with RLog replaced by
    the sign margin |mu| / s (regression has no logits) and lambda = sd(X) / sd(RLog) on the reference set.
    If a reference class has too few points the corresponding term is 0."""
    def dknn(A, X):
        if len(A) == 0:
            return np.zeros(len(X))
        d = np.sqrt(np.maximum(np.sum(X ** 2, 1)[:, None] + np.sum(A ** 2, 1)[None] - 2 * X @ A.T, 1e-12))
        kk = min(k, len(A))
        return -np.mean(np.log(np.sort(d, 1)[:, :kk]), 1)

    def dmds(A, X):
        if len(A) < 2:
            return np.zeros(len(X))
        m = A.mean(0)
        C = np.cov(A, rowvar=False).reshape(A.shape[1], A.shape[1])
        C = (1 - shrink) * C + shrink * (np.trace(C) / len(C) + 1e-6) * np.eye(len(C))
        D = X - m
        return -np.sum(D * np.linalg.solve(C, D.T).T, 1)

    mu, sd = phi_c.mean(0), phi_c.std(0) + 1e-9
    Pc, Pt = (phi_c - mu) / sd, (phi_t - mu) / sd
    cc = np.asarray(correct_c, bool)
    out = {}
    for name, fn in (("knn", dknn), ("mds", dmds)):
        s_t = fn(Pc[cc], Pt) - fn(Pc[~cc], Pt)
        s_c = fn(Pc[cc], Pc) - fn(Pc[~cc], Pc)
        lam = np.std(s_c) / (np.std(rlog_c) + 1e-9) if np.std(s_c) > 0 else 1.0
        out[name] = s_t
        out[name + "_rlog"] = s_t + lam * rlog_t
    return out


def abstain_block(b, args, f, lam, mu_t, sp_t):
    """All Delta-KNN / Delta-MDS scores for one block.
    Main reference (fair to the other methods and to DSFP): out-of-fold predictions on the training pairs,
    5 folds, same rank and lambda as the fitted predictor; features from the fitted model, as in the paper.
    Also returns the in-sample reference (the model's own training fits) for the appendix."""
    n, sigma = b["n"], b["sigma"]
    a, bb, y = b["tr"]
    ta, tb, _ = b["test"]
    sgn = lambda v: np.where(v >= 0, 1, -1)
    U, beta = f.state[2], f.state[1]
    phi = lambda p, r: np.hstack([U[p] + U[r], U[p] * U[r], (beta[p] + beta[r])[:, None]])
    rlog_t = np.abs(mu_t) / sp_t

    mu_o, sp_o = np.zeros(len(y)), np.ones(len(y))
    perm = np.random.default_rng(7 + 1000 * b["seed"] + b["program"]).permutation(len(y))
    for F in np.array_split(perm, 5):
        keep = np.setdiff1d(np.arange(len(y)), F)
        fk = M.lowrank(n, a[keep], bb[keep], y[keep], args.rank, lam)
        m_, u_, s_, _ = M.lowrank_laplace(n, a[keep], bb[keep], a[F], bb[F], sigma, lam, fk.state)
        mu_o[F], sp_o[F] = m_ + u_, s_
    oof = abstain_scores(phi(a, bb), sgn(mu_o) == sgn(y), phi(ta, tb), np.abs(mu_o) / sp_o, rlog_t)

    mu_r = f(a, bb)
    _, _, sp_r, _ = M.lowrank_laplace(n, a, bb, a, bb, sigma, lam, f.state)
    ins = abstain_scores(phi(a, bb), sgn(mu_r) == sgn(y), phi(ta, tb), np.abs(mu_r) / sp_r, rlog_t)
    return {"abstain": oof["knn_rlog"], "abstain_knn": oof["knn"], "abstain_mds": oof["mds"],
            "abstain_mds_rlog": oof["mds_rlog"], "abstain_insample": ins["knn_rlog"],
            "abstain_ref_wrong": int(np.sum(sgn(mu_o) != sgn(y))),
            "abstain_insample_ref_wrong": int(np.sum(sgn(mu_r) != sgn(y)))}


def run_block(b, Xg, args):
    sgn = lambda v: np.where(v >= 0, 1, -1)
    n, sigma = b["n"], b["sigma"]
    a, bb, y = b["tr"]
    ca, cb, cz = b["cal"]
    ta, tb, tz = b["test"]
    rng = np.random.default_rng(1000 * b["seed"] + b["program"])

    lr_fp = lambda a_, b_, y_, lam, p, q: M.lowrank(n, a_, b_, y_, args.rank, lam)(p, q)
    lam = cv_lambda(lr_fp, a, bb, y)
    f = M.lowrank(n, a, bb, y, args.rank, lam)
    m_t, u_t, sp_t, sm_t = M.lowrank_laplace(n, a, bb, ta, tb, sigma, lam, f.state)
    m_c, u_c, sp_c, _ = M.lowrank_laplace(n, a, bb, ca, cb, sigma, lam, f.state)
    mu_t, mu_c = m_t + u_t, m_c + u_c
    correct = sgn(mu_t) == sgn(tz)

    gp_t, gsd_t, s2 = gp_fit(pair_features(Xg, a, bb), y, pair_features(Xg, ta, tb), sigma)
    gp_c, _, _ = gp_fit(pair_features(Xg, a, bb), y, pair_features(Xg, ca, cb), sigma, s2=s2)
    ridge_fp = lambda a_, b_, y_, lam_, p, q: M.lowrank(n, a_, b_, y_, 0, lam_)(p, q)
    lam_r = cv_lambda(ridge_fp, a, bb, y)
    ridge_t = M.lowrank(n, a, bb, y, 0, lam_r)(ta, tb)

    boot = np.zeros(len(tz))
    for _ in range(args.boot):
        i = rng.integers(0, len(y), len(y))
        boot += sgn(M.lowrank(n, a[i], bb[i], y[i], args.rank, lam)(ta, tb)) == sgn(mu_t)
    boot /= args.boot

    agree = (1 + (sgn(gp_t) == sgn(mu_t)) + (sgn(ridge_t) == sgn(mu_t))) / 3.0
    lap = norm.cdf(np.abs(mu_t) / sp_t)
    d2 = max(0.0, float(np.mean((cz - mu_c) ** 2) - sigma ** 2 - np.mean(sp_c ** 2)))
    tot = lambda s: np.sqrt(s ** 2 + d2 + sigma ** 2)
    ne = norm.cdf(np.abs(mu_t) / tot(sp_t))

    res_c = np.abs(cz - mu_c)
    q = conformal_q(res_c, args.alpha)
    conf = np.searchsorted(np.sort(res_c), np.abs(mu_t), side="left") / (len(res_c) + 1)

    X1h_c, X1h_t = onehot(n, ca, cb), onehot(n, ta, tb)
    half_epi = epicscore(X1h_c, res_c, X1h_t, args.alpha, rng)
    with np.errstate(divide="ignore", invalid="ignore"):
        epi = np.where(np.isfinite(half_epi), np.abs(mu_t) / np.maximum(half_epi, 1e-12), 0.0)

    abst = abstain_block(b, args, f, lam, mu_t, sp_t)

    gp_post = norm.cdf(np.abs(gp_t) / np.sqrt(gsd_t ** 2 + sigma ** 2))

    z90 = norm.ppf(1 - args.alpha / 2)
    intervals = {"conformal": np.full(len(tz), q), "epicscore": half_epi, "noise_error": z90 * tot(sp_t)}
    recs = []
    for i in range(len(tz)):
        r = {"seed": b["seed"], "program": b["program"], "pair": str(b["test_pairs"][i]),
             "category": str(b["test_cats"][i]), "truth": float(tz[i]), "mu": float(mu_t[i]), "gp": float(gp_t[i]),
             "correct": bool(correct[i]), "gp_correct": bool(sgn(gp_t[i]) == sgn(tz[i])),
             "magnitude": float(abs(mu_t[i])), "bootstrap": float(boot[i]), "agreement": float(agree[i]),
             "laplace": float(lap[i]), "noise_error": float(ne[i]), "conformal": float(conf[i]),
             "epicscore": float(epi[i]),
             "gp_magnitude": float(abs(gp_t[i])), "gp_posterior": float(gp_post[i]), "ceiling": b["ceiling"][i]}
        for kname in ABSTAIN_KEYS:
            r[kname] = float(abst[kname][i])
        for kname, h in intervals.items():
            hw = float(h[i])
            r[f"{kname}_cover"] = bool(abs(tz[i] - mu_t[i]) <= hw)
            r[f"{kname}_decide"] = bool(abs(mu_t[i]) > hw)
        recs.append(r)
    print(f"  seed {b['seed']} program {b['program']}: n_test={len(tz)} n_cal={len(cz)} sigma={sigma:.3f} "
          f"lam={lam} acc pair-only={correct.mean():.2f} gp={np.mean(sgn(gp_t) == sgn(tz)):.2f} "
          f"conformal q={q:.3f} | abstain reference wrong: OOF {abst['abstain_ref_wrong']}, "
          f"in-sample {abst['abstain_insample_ref_wrong']} of {len(y)}", flush=True)
    return recs


ABSTAIN_KEYS = ("abstain", "abstain_knn", "abstain_mds", "abstain_mds_rlog", "abstain_insample")




# ---------- metrics and tables ----------

def score_metrics(v, y):
    order = np.argsort(-(v + 1e-12 * np.random.default_rng(0).random(len(v))))
    err = 1 - y[order]
    return {"auroc": M.auroc(v, y.astype(bool)),
            "aurc": float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1))),
            "top25": float(y[order[: max(1, len(y) // 4)]].mean())}


def ece(p, y):
    bins = np.minimum(((p - 0.5) * 20).astype(int), 9)
    return float(sum(np.mean(bins == q) * abs(p[bins == q].mean() - y[bins == q].mean()) for q in set(bins)))


def summarise(recs):
    out = {}
    for pred, key, _ in ALL_ROWS:
        if key not in recs[0]:
            continue
        ck = "gp_correct" if pred.startswith("GP") else "correct"
        y = np.array([r[ck] for r in recs], float)
        v = np.array([r[key] for r in recs], float)
        m = score_metrics(v, y)
        m["acc"] = float(y.mean())
        if key in PROB:
            m["ece"] = ece(v, y)
            m["brier"] = float(np.mean((v - y) ** 2))
            m["mean_conf"] = float(v.mean())
        out[f"{pred}|{key}"] = m
    for kname in INTERVALS:
        if f"{kname}_cover" not in recs[0]:
            continue
        cov = np.mean([r[f"{kname}_cover"] for r in recs])
        dec = np.array([r[f"{kname}_decide"] for r in recs])
        y = np.array([r["correct"] for r in recs])
        out[f"interval|{kname}"] = {"coverage": float(cov), "decision_rate": float(dec.mean()),
                                    "conditional_error": float(1 - y[dec].mean()) if dec.any() else float("nan")}
    out["n"] = len(recs)
    out["ceiling"] = float(np.mean([r["ceiling"] for r in recs]))
    return out


def report(results, alpha):
    ds = [d for d in DATASETS if d in results]
    W = 17
    head = f"  {'Predictor':<15}{'Reliability Method':<36}" + "".join(f"{COL[d]:>{W}}" for d in ds)

    def table(title, key, fmt):
        print("\n" + "=" * len(head))
        print(f"  {title}")
        print("=" * len(head))
        print(head)
        for pred, k, name in ALL_ROWS:
            if not all(f"{pred}|{k}" in results[d] for d in ds):
                continue
            cells = []
            for d in ds:
                v = results[d][f"{pred}|{k}"].get(key)
                cells.append(f"{'':>{W}}" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:>{W}{fmt}}")
            print(f"  {pred:<15}{name:<36}" + "".join(cells))

    table("Table 2. Reliability of Unseen Gene-Pair Predictions: Correctness AUROC (higher is better)", "auroc", ".2f")
    print(f"  {'':<51}" + "".join(f"{'n=' + str(results[d]['n']):>{W}}" for d in ds))
    print("  Accuracy of the scored predictions: Pair-only " + ", ".join(
        f"{COL[d]} {results[d]['Pair-only|magnitude']['acc']:.2f}" for d in ds)
          + "; GP " + ", ".join(f"{COL[d]} {results[d]['GP on Singles|gp_magnitude']['acc']:.2f}" for d in ds))
    print("  Ceiling (independent cell halves agree on sign): " + ", ".join(f"{COL[d]} {results[d]['ceiling']:.2f}" for d in ds))
    table("Supplementary: AURC (lower is better)", "aurc", ".3f")
    table("Supplementary: Brier score of stated P(correct) (lower is better; only methods that output a probability)",
          "brier", ".3f")
    table("Supplementary: ECE of stated P(correct) (lower is better; only methods that output a probability)", "ece", ".3f")
    table("Supplementary: accuracy of the top 25% most trusted", "top25", ".2f")
    print("\n" + "=" * len(head))
    print(f"  Supplementary: interval methods at alpha={alpha} (coverage / sign decision rate / conditional error)")
    print("=" * len(head))
    for kname, name in (("conformal", "Split Conformal"), ("epicscore", "EPICSCORE"),
                        ("noise_error", "Noise + Error Laplace")):
        if not all(f"interval|{kname}" in results[d] for d in ds):
            continue
        cells = []
        for d in ds:
            m = results[d][f"interval|{kname}"]
            ce = "-" if np.isnan(m["conditional_error"]) else f"{m['conditional_error']:.2f}"
            cells.append(f"{m['coverage']:.2f}/{m['decision_rate']:.2f}/{ce}".rjust(W))
        print(f"  {'Pair-only':<15}{name:<36}" + "".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default="42,43,44", help="split seeds; the paper reports 101,102,103,104,105")
    ap.add_argument("--programs", type=int, default=2, help="number of interaction programs w_k")
    ap.add_argument("--features", type=int, default=10, help="single-effect SVD features per gene")
    ap.add_argument("--rank", type=int, default=2, help="rank of the Pair-only model")
    ap.add_argument("--splits", type=int, default=5, help="random cell-half replicates for sigma and the ceiling")
    ap.add_argument("--boot", type=int, default=20, help="bootstrap refits for Bootstrap Stability")
    ap.add_argument("--alpha", type=float, default=0.1, help="miscoverage level of the interval methods")
    D.add_data_args(ap)
    ap.add_argument("--outdir", default="results/reliability")
    ap.add_argument("--report", action="store_true", help="print tables from saved results only")
    args = D.resolve_data_args(ap.parse_args())
    os.makedirs(args.outdir, exist_ok=True)
    for d in D.parse_list(args.datasets):
        if args.report:
            continue
        print(f"\n=== {COL[d]} ===", flush=True)
        _, _, Xg, blocks, _ = prepare(d, args)
        recs = [r for b in blocks for r in run_block(b, Xg, args)]
        path = os.path.join(args.outdir, f"{d}.json")
        with open(path, "w") as f:
            json.dump({"args": vars(args), "summary": summarise(recs), "records": recs}, f,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))
        print(f"  saved {path}", flush=True)
    results = {}
    for d in DATASETS:
        p = os.path.join(args.outdir, f"{d}.json")
        if os.path.exists(p):
            with open(p) as f:
                results[d] = json.load(f)["summary"]
    report(results, args.alpha)


if __name__ == "__main__":
    main()