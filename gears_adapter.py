"""Adapters that run the official GEARS package (cell-gears) on the same pairs and splits as prediction.py.

Each adapter writes a GEARS-ready AnnData on the same expression scale as the corresponding loader in data.py,
so that predicted pair effects can be compared with the measured ones directly.
"""
import gzip
import os
import pickle
import re

import numpy as np

GEARS_EPOCHS, GEARS_BATCH_SIZE, GEARS_HIDDEN, GEARS_LR = 15, 64, 64, 1e-3


def pair_genes(data, pair):
    genes = list(data["pair_genes_fn"](pair))
    assert len(genes) == 2, f"Invalid pair {pair!r}: {genes}"
    return genes


def canonical_pair_condition(a, b):
    a, b = sorted((str(a), str(b)))
    return f"{a}+{b}"


def canonical_single_condition(g):
    return f"{g}+ctrl"


def go_genes(data_root):
    """Genes of the GEARS GO graph (downloaded by GEARS into <data_root>/go_essential_all)."""
    import pandas as pd
    go = pd.read_csv(os.path.join(data_root, "go_essential_all", "go_essential_all.csv"))
    return set(go["source"].unique()) | set(go["target"].unique())


def strip_anndata_for_gears(adata):
    ac = adata.copy()
    for key in list(ac.obsm.keys()): del ac.obsm[key]
    if hasattr(ac, "obsp"):
        for key in list(ac.obsp.keys()): del ac.obsp[key]
    ac.uns = {}
    if "gene_name" not in ac.var.columns:
        ac.var["gene_name"] = [str(x) for x in ac.var_names]
    ac.obs["condition"] = ac.obs["condition"].astype(str)
    ac.obs["condition_name"] = ac.obs["condition"].astype(str)
    ac.obs["cell_type"] = ac.obs.get("cell_type", "baseline")
    return ac

def build_gears_norman(data_root, data):
    from gears import PertData
    pdata = PertData(data_root); pdata.load(data_name="norman")
    adata = pdata.adata
    labels = list(adata.obs["condition"].astype(str).unique())

    single_map = {}
    for cond in labels:
        genes = [g for g in cond.split("+") if g != "ctrl"]
        if len(genes) == 1: single_map[genes[0]] = cond

    pair_map = {}
    for pair in data["comp_eligible"]:
        genes = pair_genes(data, pair)
        candidates = [str(pair), f"{genes[0]}+{genes[1]}", f"{genes[1]}+{genes[0]}"]
        found = next((x for x in candidates if x in labels), None)
        if found is None: raise RuntimeError(f"Norman GEARS condition missing for {pair}")
        pair_map[pair] = found

    ctrl_mask = np.asarray(adata.obs["condition"].astype(str) == "ctrl")
    ctrl_mean = np.asarray(adata[ctrl_mask].X.mean(axis=0)).reshape(-1)
    return {"pdata": pdata, "single_map": single_map, "pair_map": pair_map,
            "transform": lambda pred: np.asarray(pred, float) - ctrl_mean}


def build_gears_wessels(data_root, h5ad_path, data, cache_name="gears_wessels2023"):
    import anndata as ad
    from gears import PertData

    ds_dir = os.path.abspath(os.path.join(data_root, cache_name))
    processed = os.path.join(ds_dir, "perturb_processed.h5ad")

    if os.path.exists(processed):
        pdata = PertData(data_root); pdata.load(data_path=ds_dir)
    else:
        source = ad.read_h5ad(h5ad_path)
        gc = source.obs["Guide.Class"].astype(str)
        cond = source.obs["condition"].astype(str)
        allowed_genes = set(data["single_effects"])
        allowed_pairs = {tuple(sorted(pair_genes(data, p))) for p in data["comp_eligible"]}

        # keep only genes in the GEARS GO graph
        go = go_genes(data_root)
        n_before = len(allowed_genes)
        allowed_genes = {g for g in allowed_genes if g in go}
        allowed_pairs = {p for p in allowed_pairs if all(g in go for g in p)}
        print(f"      GO filter: {n_before} -> {len(allowed_genes)} singles, {len(allowed_pairs)} pairs")

        labels, keep = [], np.zeros(source.n_obs, dtype=bool)
        for i, (cls, c) in enumerate(zip(gc, cond)):
            if cls == "NT": labels.append("ctrl"); keep[i] = True; continue
            genes = [g for g in str(c).split("+") if g and g != "ctrl"]
            if cls == "Single" and len(genes) == 1 and genes[0] in allowed_genes:
                labels.append(canonical_single_condition(genes[0])); keep[i] = True
            elif cls == "Dual" and len(genes) == 2 and tuple(sorted(genes)) in allowed_pairs:
                labels.append(canonical_pair_condition(genes[0], genes[1])); keep[i] = True
            else: labels.append("__drop__")

        ac = source[keep].copy()
        ac.obs["condition"] = np.asarray(labels, dtype=object)[keep]
        ac = strip_anndata_for_gears(ac)
        os.makedirs(ds_dir, exist_ok=True)
        pdata = PertData(data_root)
        pdata.new_data_process(dataset_name=cache_name, adata=ac, skip_calc_de=False)
        del source, ac

    labels = set(pdata.adata.obs["condition"].astype(str).unique())
    single_map = {g: canonical_single_condition(g) for g in data["single_effects"]}
    pair_map = {p: canonical_pair_condition(*pair_genes(data, p)) for p in data["comp_eligible"]}
    missing = [c for c in list(single_map.values()) + list(pair_map.values()) + ["ctrl"] if c not in labels]
    if missing:
        print(f"      Wessels GEARS: {len(missing)} conditions not in GO graph, filtering")
        single_map = {g: c for g, c in single_map.items() if c in labels}
        pair_map = {p: c for p, c in pair_map.items() if c in labels}

    ctrl_mask = np.asarray(pdata.adata.obs["condition"].astype(str) == "ctrl")
    ctrl_mean = np.asarray(pdata.adata[ctrl_mask].X.mean(axis=0)).reshape(-1)
    return {"pdata": pdata, "single_map": single_map, "pair_map": pair_map,
            "transform": lambda pred: np.asarray(pred, float) - ctrl_mean}


def build_gears_directcapture(data_root, raw_dir, data, cache_name="gears_directcapture_log1p"):
    """Same cells, genes and CP10K + log1p normalisation as data.load_directcapture."""
    import anndata as ad
    import pandas as pd
    from scipy import sparse
    from scipy.io import mmread
    from gears import PertData

    d = os.path.abspath(raw_dir) + "/"
    feat = pd.read_csv(d + "GSM4367984_exp6.features.tsv.gz", sep="\t", header=None)
    ens2sym = dict(zip(feat[0].astype(str), feat[1].astype(str)))

    ds_dir = os.path.abspath(os.path.join(data_root, cache_name))
    if os.path.exists(os.path.join(ds_dir, "perturb_processed.h5ad")):
        pdata = PertData(data_root)
        pdata.load(data_path=ds_dir)
    else:
        meta = pd.read_csv(d + "GSM4367984_exp6.cell_identities.csv.gz")
        bars = pd.read_csv(d + "GSM4367984_exp6.barcodes.tsv.gz", sep="\t", header=None)[0].astype(str)
        with gzip.open(d + "GSM4367984_exp6.matrix.mtx.gz", "rb") as fh:
            Xg = mmread(fh).tocsr()
        ge = (feat.iloc[:, 2].astype(str) == "Gene Expression").to_numpy()
        b2i = {b: i for i, b in enumerate(bars)}
        X = Xg[ge][:, np.array([b2i[b] for b in meta["cell_barcode"].astype(str)])].T.tocsr().astype(np.float64)
        del Xg
        lib = np.asarray(X.sum(1)).ravel()
        keep_gene = np.asarray(X.mean(0)).ravel() >= 1.0
        Xn = (sparse.diags(1e4 / lib) @ X).tocsr()[:, keep_gene]
        Xn.data = np.log1p(Xn.data)
        del X
        features = feat.loc[ge].reset_index(drop=True).loc[keep_gene].reset_index(drop=True)

        A = meta["target_A"].astype(str).to_numpy()
        B = meta["target_B"].astype(str).to_numpy()
        NT = "Non-Targeting"
        allowed_genes = set(data["single_effects"])
        allowed_pairs = {tuple(sorted(pair_genes(data, p))) for p in data["comp_eligible"]}
        labels, keep = [], np.zeros(len(meta), dtype=bool)
        for i, (a, b) in enumerate(zip(A, B)):
            sa, sb = ens2sym.get(a, a), ens2sym.get(b, b)
            if a == NT and b == NT:
                labels.append("ctrl"); keep[i] = True
            elif a == NT and b in allowed_genes:
                labels.append(canonical_single_condition(sb)); keep[i] = True
            elif b == NT and a in allowed_genes:
                labels.append(canonical_single_condition(sa)); keep[i] = True
            elif a != NT and b != NT and a != b and tuple(sorted((a, b))) in allowed_pairs:
                labels.append(canonical_pair_condition(sa, sb)); keep[i] = True
            else:
                labels.append("__drop__")
        Xk = Xn[keep]
        cond = np.asarray(labels, dtype=object)[keep]

        # the GEARS input must reproduce the effects of data.load_directcapture
        mu0 = np.asarray(Xk[cond == "ctrl"].mean(0)).ravel()
        worst = 0.0
        for g, eff in data["single_effects"].items():
            m = cond == canonical_single_condition(ens2sym.get(g, g))
            worst = max(worst, float(np.max(np.abs(np.asarray(Xk[m].mean(0)).ravel() - mu0 - eff))))
        for p in data["comp_eligible"]:
            m = cond == canonical_pair_condition(*[ens2sym.get(x, x) for x in pair_genes(data, p)])
            worst = max(worst, float(np.max(np.abs(np.asarray(Xk[m].mean(0)).ravel() - mu0
                                                   - data["pair_effects"][p]))))
        print(f"      DirectCapture scale check: max|condition effect - loader effect|={worst:.2e}")
        if worst > 1e-3:
            raise RuntimeError("DirectCapture GEARS input is not on the loader scale")

        obs = meta.loc[keep].copy()
        obs["condition"] = cond
        obs.index = [f"dc_{i}" for i in range(len(obs))]
        var = features.copy()
        var.columns = [str(c) for c in var.columns]
        var.index = var.iloc[:, 0].astype(str)
        var["gene_name"] = var.iloc[:, 1].astype(str).to_numpy()
        ac = strip_anndata_for_gears(ad.AnnData(X=Xk.astype(np.float32), obs=obs, var=var))
        os.makedirs(ds_dir, exist_ok=True)
        pdata = PertData(data_root)
        pdata.new_data_process(dataset_name=cache_name, adata=ac, skip_calc_de=False)
        del ac, Xk, Xn

    labels = set(pdata.adata.obs["condition"].astype(str).unique())
    single_map = {g: canonical_single_condition(ens2sym.get(g, g)) for g in data["single_effects"]}
    pair_map = {p: canonical_pair_condition(*[ens2sym.get(g, g) for g in pair_genes(data, p)])
                for p in data["comp_eligible"]}
    missing = [c for c in list(single_map.values()) + list(pair_map.values()) + ["ctrl"] if c not in labels]
    if missing:
        raise RuntimeError(f"DirectCapture GEARS missing: {missing[:10]}")
    if pdata.adata.n_vars != data["n_genes_out"]:
        raise RuntimeError(f"DirectCapture GEARS genes {pdata.adata.n_vars} != loader {data['n_genes_out']}")
    ctrl_mask = np.asarray(pdata.adata.obs["condition"].astype(str) == "ctrl")
    ctrl_mean = np.asarray(pdata.adata[ctrl_mask].X.mean(axis=0)).reshape(-1)
    return {"pdata": pdata, "single_map": single_map, "pair_map": pair_map,
            "transform": lambda pred: np.nan_to_num(np.asarray(pred, float) - ctrl_mean), "ens2sym": ens2sym}


def build_gears_gse164996(data_root, raw_dir, data, cache_name="gears_gse164996"):
    import anndata as ad
    import pandas as pd
    from scipy.io import mmread
    from scipy import sparse
    from gears import PertData

    ds_dir = os.path.abspath(os.path.join(data_root, cache_name))
    processed = os.path.join(ds_dir, "perturb_processed.h5ad")

    if os.path.exists(processed):
        pdata = PertData(data_root); pdata.load(data_path=ds_dir)
    else:
        matrix_f = os.path.join(raw_dir, "GSE164996_S2_filtered_matrix.mtx.gz")
        features_f = os.path.join(raw_dir, "GSE164996_S2_filtered_features.tsv.gz")
        barcodes_f = os.path.join(raw_dir, "GSE164996_S2_filtered_barcodes.tsv.gz")
        calls_f = os.path.join(raw_dir, "GSE164996_S2_protospacer_calls_per_cell.csv.gz")

        features = pd.read_csv(features_f, sep="\t", header=None, names=["ensembl", "gene", "type"])
        barcodes = pd.read_csv(barcodes_f, sep="\t", header=None, names=["barcode"])
        calls = pd.read_csv(calls_f)

        with gzip.open(matrix_f, "rb") as fh: Xg = mmread(fh).tocsr()
        Xraw = Xg.T.tocsr().astype(np.float64); del Xg
        ge_mask = (features["type"].astype(str) == "Gene Expression").to_numpy()
        Xraw = Xraw[:, ge_mask].tocsr()
        features = features.loc[ge_mask].reset_index(drop=True)

        lib = np.asarray(Xraw.sum(axis=1)).reshape(-1)
        X = sparse.diags(1e4 / lib) @ Xraw; X.data = np.log1p(X.data)
        b2i = {str(b): i for i, b in enumerate(barcodes["barcode"].astype(str))}

        def guide_to_gene(g):
            g = str(g)
            if re.match(r"^CTRL\d+$", g): return g
            if g.startswith("hRosa26"): return "hRosa26"
            return re.sub(r"_\d+$", "", g)

        ctrls = {"CTRL0001", "CTRL0002"}
        allowed_genes = set(data["single_effects"])
        allowed_pairs = {tuple(sorted(pair_genes(data, p))) for p in data["comp_eligible"]}

        # keep only genes in the GEARS GO graph
        go = go_genes(data_root)
        n_before = len(allowed_genes)
        allowed_genes = {g for g in allowed_genes if g in go}
        allowed_pairs = {p for p in allowed_pairs if all(g in go for g in p)}
        print(f"      GO filter: {n_before} -> {len(allowed_genes)} singles, {len(allowed_pairs)} pairs")

        selected_idx, labels = [], []
        for row in calls.itertuples(index=False):
            bc = str(row.cell_barcode)
            if bc not in b2i: continue
            guides = str(row.feature_call).split("|")
            if len(guides) != 2: continue
            a, b = [guide_to_gene(x) for x in guides]
            if a == b: continue

            a_ctrl, b_ctrl = a in ctrls, b in ctrls
            condition = None
            if a_ctrl and b_ctrl: condition = "ctrl"
            elif a_ctrl ^ b_ctrl:
                gene = b if a_ctrl else a
                if gene in allowed_genes: condition = canonical_single_condition(gene)
            elif a != "hRosa26" and b != "hRosa26" and tuple(sorted((a, b))) in allowed_pairs:
                condition = canonical_pair_condition(a, b)

            if condition is not None:
                selected_idx.append(b2i[bc]); labels.append(condition)

        obs = pd.DataFrame({"condition": labels}, index=[f"g164996_{i}" for i in range(len(labels))])
        var = pd.DataFrame({"gene_name": features["gene"].astype(str).to_numpy()},
                           index=features["ensembl"].astype(str).to_numpy())
        ac = ad.AnnData(X=X[np.asarray(selected_idx, np.int64)].astype(np.float32), obs=obs, var=var)
        ac = strip_anndata_for_gears(ac)
        os.makedirs(ds_dir, exist_ok=True)
        pdata = PertData(data_root)
        pdata.new_data_process(dataset_name=cache_name, adata=ac, skip_calc_de=False)
        del ac, Xraw, X

    labels = set(pdata.adata.obs["condition"].astype(str).unique())
    single_map = {g: canonical_single_condition(g) for g in data["single_effects"]}
    pair_map = {p: canonical_pair_condition(*pair_genes(data, p)) for p in data["comp_eligible"]}
    missing = [c for c in list(single_map.values()) + list(pair_map.values()) + ["ctrl"] if c not in labels]
    if missing:
        print(f"      GSE164996 GEARS: {len(missing)} conditions not in GO graph, filtering")
        single_map = {g: c for g, c in single_map.items() if c in labels}
        pair_map = {p: c for p, c in pair_map.items() if c in labels}

    ctrl_mask = np.asarray(pdata.adata.obs["condition"].astype(str) == "ctrl")
    ctrl_mean = np.asarray(pdata.adata[ctrl_mask].X.mean(axis=0)).reshape(-1)
    return {"pdata": pdata, "single_map": single_map, "pair_map": pair_map,
            "transform": lambda pred: np.asarray(pred, float) - ctrl_mean}


def prepare_gears_adapter(dataset_name, data, args):
    name = dataset_name.lower()
    if name == "norman":
        return build_gears_norman(args.data_dir, data)
    if name == "wessels2023":
        return build_gears_wessels(args.data_dir, args.wessels_path, data)
    if name == "directcapture":
        return build_gears_directcapture(args.data_dir, args.directcapture_dir, data)
    if name == "gse164996":
        return build_gears_gse164996(args.data_dir, args.gse164996_dir, data)
    raise KeyError(dataset_name)


def run_gears(dataset_name, data, adapter, train_pairs, val_pairs, test_pairs, seed, device, checkpoint_root):
    """Train GEARS on singles + training pairs (validation = calibration pairs); return {test pair: effect}."""
    import torch
    from gears import GEARS

    pdata = adapter["pdata"]
    single_map, pair_map = adapter["single_map"], adapter["pair_map"]
    split_dict = {
        "train": ["ctrl"] + [single_map[g] for g in sorted(data["single_effects"]) if g in single_map]
                 + [pair_map[p] for p in train_pairs if p in pair_map],
        "val": [pair_map[p] for p in val_pairs if p in pair_map],
        "test": [pair_map[p] for p in test_pairs if p in pair_map],
    }
    split_dir = os.path.join(checkpoint_root, dataset_name, "gears_splits")
    os.makedirs(split_dir, exist_ok=True)
    split_path = os.path.join(split_dir, f"seed{seed}.pkl")
    with open(split_path, "wb") as f:
        pickle.dump(split_dict, f)
    pdata.prepare_split(split="custom", seed=int(seed), split_dict_path=split_path)
    pdata.get_dataloader(batch_size=GEARS_BATCH_SIZE, test_batch_size=256)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = GEARS(pdata, device=device, weight_bias_track=False)
    model.model_initialize(hidden_size=GEARS_HIDDEN)
    model.train(epochs=GEARS_EPOCHS, lr=GEARS_LR)

    preds, ens2sym = {}, adapter.get("ens2sym", {})
    for pair in test_pairs:
        genes = [ens2sym.get(g, g) for g in pair_genes(data, pair)]
        result = None
        for attempt in range(3):
            try:
                result = model.predict([genes])
                break
            except Exception as exc:
                msg = str(exc)
                if "not in the perturbation graph" in msg:
                    print(f"      GEARS skipped {pair}: gene not in GEARS GO graph")
                    break
                if "out of memory" in msg.lower() and attempt < 2:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"      GEARS OOM on {pair}, retrying after freeing cache", flush=True)
                    continue
                raise RuntimeError(f"GEARS prediction failed for {pair}: {msg}") from exc
        if result is None:
            continue
        pred = adapter["transform"](np.asarray(next(iter(result.values())), float).reshape(-1))
        assert pred.shape == (data["n_genes_out"],)
        preds[pair] = pred
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds
