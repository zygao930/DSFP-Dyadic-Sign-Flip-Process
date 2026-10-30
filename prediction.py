"""Prediction quality on unseen gene pairs (paper Table 2): sign accuracy and MAE of the program score z.

Target: z = w_k^T (r - mean_tr), r = pair effect - single a - single b, projected on a training interaction
program; same splits, programs and test pairs as reliability.py.  Paper: seeds 42,43,44.
GEARS uses the official cell-gears package (gears_adapter.py); NAIAD the official package with its defaults.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import json
import pickle

import numpy as np

import data as D
import models as M
from data import DATASETS
from reliability import COL, prepare

ROWS = [("ridge", "Ridge (pair-only)"), ("bilinear", "Bilinear (rank-2)"), ("kernel", "Kernel on Single Effects"),
        ("pairlaplace", "Pair-only + Laplace"), ("gp", "GP on Single Effects"), ("gears", "GEARS (2024)"),
        ("naiad", "NAIAD (2025)")]
REFERENCE = ("train_mean", "additive")       # z_hat = 0  and  r_hat = 0 (z_hat = -mean_tr @ w), MAE only
KRR_LAMS = (0.01, 0.03, 0.1, 0.3, 1.0)


def bilinear(n, a, b, y, rank, lam, sweeps=40, seed=0):
    """f_ab = v_a . v_b, alternating ridge; genes without training pairs stay at 0."""
    V = 0.1 * np.random.default_rng(seed).standard_normal((n, rank))
    nb = [[] for _ in range(n)]
    for k, (i, j) in enumerate(zip(a, b)):
        nb[i].append((j, k))
        nb[j].append((i, k))
    nb = [(np.array([h for h, _ in L], int), np.array([k for _, k in L], int)) for L in nb]
    V[[len(h) == 0 for h, _ in nb]] = 0.0
    for _ in range(sweeps):
        for g in range(n):
            h, k = nb[g]
            if len(h):
                V[g] = np.linalg.solve(V[h].T @ V[h] + lam * np.eye(rank), V[h].T @ y[k])
    return lambda p, q: np.sum(V[p] * V[q], 1)


def krr(Ftr, ytr, Fte, lam):
    d2 = lambda P, Q: np.sum(P ** 2, 1)[:, None] + np.sum(Q ** 2, 1)[None] - 2 * P @ Q.T
    D = d2(Ftr, Ftr)
    gk = 1.0 / max(np.median(D[D > 0]), 1e-12)
    al = np.linalg.solve(np.exp(-gk * D) + lam * np.eye(len(D)), ytr - ytr.mean())
    return ytr.mean() + np.exp(-gk * d2(Fte, Ftr)) @ al


def gears_predictions(d, data, blocks, args):
    """seed -> {pair: predicted pair effect vector}; trained once per seed and cached on disk."""
    import gears_adapter
    cache_dir = os.path.join(args.checkpoint_root, d, "gears_preds")
    os.makedirs(cache_dir, exist_ok=True)
    adapter, out = None, {}
    for seed in sorted({b["seed"] for b in blocks}):
        b = next(x for x in blocks if x["seed"] == seed)
        test = list(b["test_pairs_all"])
        meta = {"train": sorted(map(str, b["train"])), "cal": sorted(map(str, b["cal_pairs_all"])),
                "test": sorted(map(str, test)), "n_genes": int(data["n_genes_out"])}
        path = os.path.join(cache_dir, f"seed{seed}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                saved = pickle.load(fh)
            if saved.get("meta") == meta:
                out[seed] = saved["preds"]
                print(f"  GEARS seed {seed}: loaded saved predictions ({len(out[seed])}/{len(test)})", flush=True)
                continue
            print(f"  GEARS seed {seed}: saved predictions are for a different split or data, retraining", flush=True)
        if adapter is None:
            adapter = gears_adapter.prepare_gears_adapter(d, data, args)
        out[seed] = gears_adapter.run_gears(d, data, adapter, b["train"], b["cal_pairs_all"], test, seed,
                                            args.device, args.checkpoint_root)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump({"meta": meta, "preds": out[seed]}, fh)
        os.replace(tmp, path)
        print(f"  GEARS seed {seed}: {len(out[seed])}/{len(test)} test pairs predicted (saved)", flush=True)
    return out


def naiad_predict(data, genes, b, args):
    """Official NAIAD ('both': single phenotype MLP + adaptive gene embedding MLP), defaults from the
    tutorial (lr 1e-2, 100 epochs, batch 1024, best model on validation = our calibration split)."""
    import pandas as pd
    import torch
    from naiad import NAIAD
    parse = data["pair_genes_fn"]
    w = b["w"]
    single = {g: float(np.asarray(data["single_effects"][g]) @ w) for g in genes}

    def frame(pairs):
        rows = []
        for p in pairs:
            x, y = parse(p)
            rows.append({"id1": x, "id2": y, "comb_score": float(np.asarray(data["pair_effects"][p]) @ w),
                         "id1_score": single[x], "id2_score": single[y]})
        return pd.DataFrame(rows).astype({"comb_score": "float32", "id1_score": "float32", "id2_score": "float32"})

    parts = {"train": frame(b["tr_pairs"]), "val": frame(b["cal_pairs"]), "test": frame(b["test_pairs"])}
    allp = pd.concat(list(parts.values()), ignore_index=True)
    m = NAIAD(allp, n_train=len(parts["train"]), n_val=len(parts["val"]), n_test=len(parts["test"]), batch_size=1024)
    m.set_seed(int(b["seed"]))
    m.data = parts
    m.prepare_dataloaders()
    m.initialize_model(device=args.device)
    m.setup_trainer(n_epoch=args.naiad_epochs)
    m.train_model()
    m.generate_preds(use_best=True)
    comb = np.asarray(m.preds["test"]["preds"].values, float)
    base = np.array([single[parse(p)[0]] + single[parse(p)[1]] for p in b["test_pairs"]])
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return comb - base - float(b["mean_tr"] @ w)


def block_predictions(b, data, genes, Xg, gears, args):
    """Predictions of every method for the test pairs of one seed x program block, all on the
    target z = (r - mean_tr) @ w (r = pair effect - single a - single b)."""
    n, sigma = b["n"], b["sigma"]
    a, bb, y = b["tr"]
    ta, tb, _ = b["test"]
    pred = {}
    lam_r = M.cv_lambda(lambda a_, b_, y_, l, p, q: M.lowrank(n, a_, b_, y_, 0, l)(p, q), a, bb, y)
    pred["ridge"] = M.lowrank(n, a, bb, y, 0, lam_r)(ta, tb)
    lam_b = M.cv_lambda(lambda a_, b_, y_, l, p, q: bilinear(n, a_, b_, y_, args.rank, l)(p, q), a, bb, y)
    pred["bilinear"] = bilinear(n, a, bb, y, args.rank, lam_b)(ta, tb)
    F = lambda p, q: M.pair_features(Xg, p, q)
    lam_k = M.cv_lambda(lambda a_, b_, y_, l, p, q: krr(F(a_, b_), y_, F(p, q), l), a, bb, y, lams=KRR_LAMS)
    pred["kernel"] = krr(F(a, bb), y, F(ta, tb), lam_k)
    lam = M.cv_lambda(lambda a_, b_, y_, l, p, q: M.lowrank(n, a_, b_, y_, args.rank, l)(p, q), a, bb, y)
    f = M.lowrank(n, a, bb, y, args.rank, lam)
    m_t, u_t, _, _ = M.lowrank_laplace(n, a, bb, ta, tb, sigma, lam, f.state)
    pred["pairlaplace"] = m_t + u_t                          # the fixed predictions scored by DSFP
    pred["gp"], _, _ = M.gp_fit(F(a, bb), y, F(ta, tb), sigma)
    if "naiad" in args.methods:
        try:
            pred["naiad"] = naiad_predict(data, genes, b, args)
        except Exception as e:
            print(f"  NAIAD failed: {e}", flush=True)
    g = gears.get(b["seed"], {})
    if g:
        base = {p: np.asarray(data["single_effects"][data["pair_genes_fn"](p)[0]], float)
                + np.asarray(data["single_effects"][data["pair_genes_fn"](p)[1]], float) for p in b["test_pairs"]}
        pred["gears"] = np.array([((g[p] - base[p] - b["mean_tr"]) @ b["w"]) if p in g else np.nan
                                  for p in b["test_pairs"]])
    return pred


def run_dataset(d, args):
    data, genes, Xg, blocks, _ = prepare(d, args)
    gears = gears_predictions(d, data, blocks, args) if "gears" in args.methods else {}
    sgn = lambda v: np.where(v >= 0, 1, -1)
    recs, out_blocks = [], []
    for b in blocks:
        tz = np.asarray(b["test"][2], float)
        truth = sgn(tz)
        pred = block_predictions(b, data, genes, Xg, gears, args)
        for i in range(len(tz)):
            r = {"seed": b["seed"], "program": b["program"], "pair": str(b["test_pairs"][i]),
                 "category": str(b["test_cats"][i]), "ceiling": b["ceiling"][i]}
            for k, v in pred.items():
                if np.isfinite(v[i]):
                    r[k] = bool(sgn(v[i]) == truth[i])
            recs.append(r)
        pred["train_mean"] = np.zeros(len(tz))
        pred["additive"] = np.full(len(tz), -float(b["mean_tr"] @ b["w"]))
        m = {}
        for k, v in pred.items():
            ok = np.isfinite(v)
            if not ok.any():
                continue
            e = np.abs(tz[ok] - v[ok])
            m[k] = {"mae": float(e.mean()), "rmse": float(np.sqrt(np.mean(e ** 2))),
                    "nmae": float(e.mean() / max(np.mean(np.abs(tz[ok])), 1e-12)),
                    "n": int(ok.sum()), "n_total": int(len(tz))}
        out_blocks.append({"seed": int(b["seed"]), "program": int(b["program"]), "metrics": m,
                           "pairs": [str(p) for p in b["test_pairs"]], "truth": [float(x) for x in tz],
                           "pred": {k: [float(x) if np.isfinite(x) else None for x in v] for k, v in pred.items()}})
        print(f"  seed {b['seed']} program {b['program']}: " + "  ".join(
            f"{k} acc={np.nanmean(np.where(np.isfinite(v), sgn(v) == truth, np.nan)):.2f}"
            f" mae={m[k]['mae']:.3f}" for k, v in pred.items() if k in m and k not in REFERENCE), flush=True)
    return recs, out_blocks


def summarise(recs, blocks):
    """Sign accuracy and MAE: per seed (pooled over its programs), then mean and sd over seeds."""
    out = {"ceiling": float(np.mean([r["ceiling"] for r in recs])), "n": len(recs)}
    seeds = sorted({r["seed"] for r in recs})
    for k, _ in ROWS:
        per_seed = [np.mean([r[k] for r in recs if r["seed"] == s and k in r]) for s in seeds
                    if any(k in r for r in recs if r["seed"] == s)]
        out[k] = None if not per_seed else {"mean": float(np.mean(per_seed)),
                                            "std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                                            "n": sum(k in r for r in recs)}
    for k in [k for k, _ in ROWS] + list(REFERENCE):
        per_seed = [np.mean([b["metrics"][k]["mae"] for b in blocks if b["seed"] == s and k in b["metrics"]])
                    for s in seeds if any(k in b["metrics"] for b in blocks if b["seed"] == s)]
        if per_seed:
            out.setdefault("mae", {})[k] = {"mean": float(np.mean(per_seed)),
                                            "std": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0}
    return out


def report(results):
    ds = [d for d in DATASETS if d in results]
    W = 16
    head = f"  {'Prediction Method':<28}" + "".join(f"{COL[d]:>{W}}" for d in ds)
    for title, get in (("Sign accuracy (higher is better)", lambda r, k: r.get(k)),
                       ("MAE of z (lower is better)", lambda r, k: r.get("mae", {}).get(k))):
        print("\n" + "=" * len(head))
        print(f"  Prediction on unseen gene pairs: {title}, mean +/- sd over seeds")
        print("=" * len(head))
        print(head)
        for k, name in ROWS:
            cells = []
            for d in ds:
                v = get(results[d], k)
                cells.append(f"{'-':>{W}}" if v is None else f"{v['mean']:.3f} +/- {v['std']:.3f}".rjust(W))
            print(f"  {name:<28}" + "".join(cells))
    print(f"  {'n test predictions':<28}" + "".join(f"{results[d]['n']:>{W}}" for d in ds))
    print(f"  {'Ceiling (cell halves agree)':<28}" + "".join(f"{results[d]['ceiling']:>{W}.2f}" for d in ds))
    for k in ("gears", "naiad"):
        part = [f"{COL[d]} {results[d][k]['n']}/{results[d]['n']}" for d in ds
                if results[d].get(k) and results[d][k]["n"] < results[d]["n"]]
        if part:
            print(f"  {k.upper()} covered only part of the test pairs (GO graph filter): " + ", ".join(part))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--methods", default="gears,naiad", help="external methods to run (the others always run)")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--programs", type=int, default=2)
    ap.add_argument("--features", type=int, default=10)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--naiad-epochs", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--checkpoint-root", default="results/checkpoints")
    D.add_data_args(ap)
    ap.add_argument("--outdir", default="results/prediction")
    ap.add_argument("--report", action="store_true", help="print tables from saved results only")
    args = D.resolve_data_args(ap.parse_args())
    args.methods = D.parse_list(args.methods)
    os.makedirs(args.outdir, exist_ok=True)
    for d in ([] if args.report else D.parse_list(args.datasets)):
        print(f"\n=== {COL[d]} ===", flush=True)
        recs, blocks = run_dataset(d, args)
        path = os.path.join(args.outdir, f"{d}.json")
        with open(path, "w") as f:
            json.dump({"args": vars(args), "summary": summarise(recs, blocks), "records": recs, "blocks": blocks},
                      f, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        print(f"  saved {path}", flush=True)
    results = {}
    for d in DATASETS:
        p = os.path.join(args.outdir, f"{d}.json")
        if os.path.exists(p):
            with open(p) as f:
                results[d] = json.load(f)["summary"]
    report(results)


if __name__ == "__main__":
    main()
