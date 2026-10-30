"""Dataset loading, the 60/20/20 pair split, and independent cell-half replicates.

Every dataset is returned as a dict with
    single_effects  gene -> mean expression change of the single perturbation (vs control)
    pair_effects    pair label -> mean expression change of the double perturbation
    comp_eligible   pair labels whose two singles are both observed
    pert_gene2idx   gene -> row index
    n_genes_out     number of measured output genes
    pair_genes_fn   pair label -> [gene_a, gene_b]
    combine_mode    always "sum"
"""
import gzip
import os
import re
from collections import defaultdict

import numpy as np

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.60, 0.20, 0.20
DATASETS = ["norman", "directcapture", "wessels2023", "gse164996"]
DC_MIN_CELLS = 30           # DirectCapture: minimum cells per single / pair condition
DC_CACHE = "directcapture_exp6_condition_effects_ge.npz"


def add_data_args(ap):
    """Command line options shared by every entry script."""
    ap.add_argument("--data-dir", default="./data", help="GEARS data root (Norman is downloaded here)")
    ap.add_argument("--wessels-path", default=None, help="default: <data-dir>/wessels23/Wessels2023.h5ad")
    ap.add_argument("--directcapture-dir", default=None, help="default: <data-dir>/directcapture")
    ap.add_argument("--gse164996-dir", default=None, help="default: <data-dir>/gse164996")


def resolve_data_args(args):
    args.wessels_path = args.wessels_path or os.path.join(args.data_dir, "wessels23", "Wessels2023.h5ad")
    args.directcapture_dir = args.directcapture_dir or os.path.join(args.data_dir, "directcapture")
    args.gse164996_dir = args.gse164996_dir or os.path.join(args.data_dir, "gse164996")
    return args


def parse_seeds(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_list(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def build_pairs(data, pairs):
    se, pe, pg, p2i = (
        data["single_effects"], data["pair_effects"],
        data["pair_genes_fn"], data["pert_gene2idx"],
    )
    conds, base, y, idx = [], [], [], []
    for p in pairs:
        if p not in pe:
            continue
        genes = pg(p)
        if len(genes) != 2 or any(g not in se or g not in p2i for g in genes):
            continue
        a, b = genes
        conds.append(p)
        base.append(se[a] + se[b])
        y.append(pe[p])
        idx.append((p2i[a], p2i[b]))
    return conds, np.asarray(base), np.asarray(y), idx


# ---------------------------------------------------------------- loaders


def load_norman(data_dir):
    from gears import PertData

    pd = PertData(data_dir)
    pd.load(data_name="norman")
    adata = pd.adata

    def parse(c):
        return [g for g in str(c).split("+") if g != "ctrl"]

    labels = list(adata.obs["condition"].astype(str).unique())
    singles = [c for c in labels if c != "ctrl" and len(parse(c)) == 1]
    pairs = [c for c in labels if len(parse(c)) == 2]
    single_names = {parse(c)[0] for c in singles}
    eligible = [p for p in pairs if all(g in single_names for g in parse(p))]

    ctrl = adata[np.asarray(adata.obs["condition"].astype(str) == "ctrl")].X
    ctrl_mean = np.asarray(
        ctrl.toarray().mean(0) if hasattr(ctrl, "toarray") else ctrl.mean(0)
    ).reshape(-1)

    def profile(c):
        x = adata[np.asarray(adata.obs["condition"].astype(str) == str(c))].X
        v = np.asarray(
            x.toarray().mean(0) if hasattr(x, "toarray") else x.mean(0)
        ).reshape(-1)
        return np.nan_to_num(v - ctrl_mean).astype(float)

    se = {parse(c)[0]: profile(c) for c in singles}
    genes = sorted(se)
    return {
        "single_effects": se,
        "pair_effects": {p: profile(p) for p in pairs},
        "comp_eligible": eligible,
        "pert_gene2idx": {g: i for i, g in enumerate(genes)},
        "n_genes_out": adata.n_vars,
        "pair_genes_fn": parse,
        "combine_mode": "sum",
    }


def load_wessels(path):
    import anndata as ad

    adata = ad.read_h5ad(path)
    required = {"Guide.Class", "condition"}
    missing = required.difference(adata.obs.columns)
    if missing:
        raise KeyError(f"Wessels2023 missing obs columns: {sorted(missing)}")

    guide_class = adata.obs["Guide.Class"].astype(str)
    condition = adata.obs["condition"].astype(str)
    ctrl_mask = np.asarray(guide_class == "NT")
    if not np.any(ctrl_mask):
        ctrl_mask = np.asarray(condition == "ctrl")
    if not np.any(ctrl_mask):
        raise RuntimeError("Wessels2023 has no NT/control cells")

    def mean_profile(mask):
        x = adata[np.asarray(mask)].X
        mean = x.mean(axis=0)
        if hasattr(mean, "A1"):
            mean = mean.A1
        else:
            mean = np.asarray(mean).reshape(-1)
        return np.nan_to_num(mean).astype(float)

    ctrl_mean = mean_profile(ctrl_mask)
    single_labels = sorted(condition[guide_class == "Single"].unique())
    double_labels = sorted(condition[guide_class == "Dual"].unique())

    def parse_single(label):
        genes = [g for g in str(label).split("+") if g and g != "ctrl"]
        return genes[0] if len(genes) == 1 else None

    def parse_pair(label):
        genes = [g for g in str(label).split("+") if g and g != "ctrl"]
        return genes if len(genes) == 2 else []

    single_effects = {}
    for label in single_labels:
        gene = parse_single(label)
        if gene is not None:
            single_effects[gene] = (
                mean_profile(np.asarray(condition == label)) - ctrl_mean
            )

    pair_effects, pair_map = {}, {}
    for label in double_labels:
        genes = parse_pair(label)
        if len(genes) == 2 and all(g in single_effects for g in genes):
            pair_effects[label] = (
                mean_profile(np.asarray(condition == label)) - ctrl_mean
            )
            pair_map[label] = tuple(genes)

    genes = sorted(single_effects)
    eligible = sorted(pair_effects)
    print(
        f"  Wessels2023: {len(single_effects)} singles, "
        f"{len(eligible)} eligible doubles, {adata.n_vars} genes, "
        f"{int(np.sum(ctrl_mask))} controls"
    )
    return {
        "single_effects": single_effects,
        "pair_effects": pair_effects,
        "comp_eligible": eligible,
        "pert_gene2idx": {g: i for i, g in enumerate(genes)},
        "n_genes_out": adata.n_vars,
        "pair_genes_fn": lambda c: list(pair_map[str(c)]),
        "combine_mode": "sum",
    }


def load_directcapture(data_dir):
    """Replogle et al. direct-capture Perturb-seq exp6 (GSE146194 / GSM4367984).

    1) keep cells with published perturbation identities, gene-expression features only
    2) keep genes with mean raw expression >= 1 UMI per cell
    3) CP10K (library size over all genes) followed by log1p
    4) effects are condition means minus the mean of non-targeting + non-targeting cells
    5) singles and unordered cross-gene doubles with at least DC_MIN_CELLS cells; both singles observed
    The result is cached as an .npz next to the raw files.
    """
    import pandas as pd
    from scipy import sparse
    from scipy.io import mmread

    data_dir = os.path.abspath(data_dir)
    cache_path = os.path.join(data_dir, DC_CACHE)
    if not os.path.exists(cache_path):
        build_directcapture_cache(data_dir, cache_path, pd, sparse, mmread)

    z = np.load(cache_path, allow_pickle=False)
    single_names = [str(x) for x in z["single_names"]]
    pair_names = [str(x) for x in z["pair_names"]]
    pair_a = [str(x) for x in z["pair_a"]]
    pair_b = [str(x) for x in z["pair_b"]]
    se_mat = np.asarray(z["single_effects"], dtype=float)
    pe_mat = np.asarray(z["pair_effects"], dtype=float)
    n_genes = int(z["n_genes"])
    single_effects = {g: se_mat[i] for i, g in enumerate(single_names)}
    pair_effects = {p: pe_mat[i] for i, p in enumerate(pair_names)}
    pair_map = {p: (pair_a[i], pair_b[i]) for i, p in enumerate(pair_names)}
    genes = sorted(single_effects)
    print(f"  DirectCapture: {len(single_effects)} singles, {len(pair_names)} eligible doubles, "
          f"{n_genes} retained genes, {int(z['n_controls'])} controls, {int(z['n_cells'])} identity-mapped cells "
          f"/ {int(z['n_rna_cells'])} RNA cells [cache]")
    return {
        "single_effects": single_effects,
        "pair_effects": pair_effects,
        "comp_eligible": pair_names,
        "pert_gene2idx": {g: i for i, g in enumerate(genes)},
        "n_genes_out": n_genes,
        "pair_genes_fn": lambda c: list(pair_map[str(c)]),
        "combine_mode": "sum",
    }


def build_directcapture_cache(data_dir, out, pd, sparse, mmread):
    d = data_dir + "/"
    for f in ("matrix.mtx.gz", "features.tsv.gz", "barcodes.tsv.gz", "cell_identities.csv.gz"):
        if not os.path.exists(d + "GSM4367984_exp6." + f):
            raise FileNotFoundError(f"DirectCapture file missing: {d}GSM4367984_exp6.{f}")
    meta = pd.read_csv(d + "GSM4367984_exp6.cell_identities.csv.gz")
    feat = pd.read_csv(d + "GSM4367984_exp6.features.tsv.gz", sep="\t", header=None)
    bars = pd.read_csv(d + "GSM4367984_exp6.barcodes.tsv.gz", sep="\t", header=None)[0].astype(str)
    with gzip.open(d + "GSM4367984_exp6.matrix.mtx.gz", "rb") as fh:
        Xg = mmread(fh).tocsr()
    ge = (feat.iloc[:, 2].astype(str) == "Gene Expression").to_numpy()
    b2i = {b: i for i, b in enumerate(bars)}
    idx = np.array([b2i[b] for b in meta["cell_barcode"].astype(str)])
    X = Xg[ge][:, idx].T.tocsr().astype(np.float64)
    del Xg

    lib = np.asarray(X.sum(1)).ravel()                      # total gene expression UMI per cell
    keep = np.asarray(X.mean(0)).ravel() >= 1.0             # mean raw expression >= 1 UMI per cell
    Xn = (sparse.diags(1e4 / lib) @ X).tocsr()[:, keep]     # CP10K on all genes, then keep
    Xn.data = np.log1p(Xn.data)

    A = meta["target_A"].astype(str).to_numpy()
    B = meta["target_B"].astype(str).to_numpy()
    NT = "Non-Targeting"

    def mean(m):
        return np.asarray(Xn[m].mean(0)).ravel()

    ctrl = (A == NT) & (B == NT)
    mu0 = mean(ctrl)

    singles = {}
    for g in sorted(set(A[(A != NT) & (B == NT)]) | set(B[(A == NT) & (B != NT)])):
        m = ((A == g) & (B == NT)) | ((A == NT) & (B == g))
        if m.sum() >= DC_MIN_CELLS:
            singles[g] = mean(m) - mu0

    pairs, pa, pb, cnt = {}, [], [], []
    for a, b in sorted({tuple(sorted(x)) for x in zip(A, B) if NT not in x and x[0] != x[1]}):
        m = ((A == a) & (B == b)) | ((A == b) & (B == a))
        if a in singles and b in singles and m.sum() >= DC_MIN_CELLS:
            pairs[f"{a}|{b}"] = mean(m) - mu0
            pa.append(a)
            pb.append(b)
            cnt.append(int(m.sum()))

    names, pn = sorted(singles), list(pairs)
    np.savez_compressed(
        out,
        single_names=np.array(names), pair_names=np.array(pn), pair_a=np.array(pa), pair_b=np.array(pb),
        single_effects=np.stack([singles[g] for g in names]).astype(np.float32),
        pair_effects=np.stack([pairs[p] for p in pn]).astype(np.float32),
        n_genes=np.array(int(keep.sum())), n_controls=np.array(int(ctrl.sum())),
        n_cells=np.array(len(idx)), n_rna_cells=np.array(len(bars)), pair_cell_counts=np.array(cnt),
        retained_feature_ids=feat.loc[ge].iloc[keep.nonzero()[0], 0].astype(str).to_numpy(),
    )
    print(f"  DirectCapture cache built: {len(singles)} singles, {len(pairs)} pairs, {int(keep.sum())} genes -> {out}")


def load_gse164996(data_dir):
    """Load GSE164996 S2 using a matched exact-two-guide factorial design.

    Formal HNR inputs are constructed only from exact two-guide cells:
      delta_a  = mu(a + CTRL) - mu(CTRL + CTRL)
      delta_ab = mu(a + b)    - mu(CTRL + CTRL)

    Therefore the interaction residual used by HNR is exactly:
      I_ab = delta_ab - delta_a - delta_b
           = mu(a+b) - mu(a+CTRL) - mu(b+CTRL) + mu(CTRL+CTRL)

    This keeps single and double effects in the same two-guide context.
    Cells with 3+ assigned guides, same-gene doubles, and hRosa26 pairs are excluded.
    """
    import pandas as pd
    from scipy import sparse
    from scipy.io import mmread

    data_dir = os.path.abspath(data_dir)
    matrix_f = os.path.join(data_dir, "GSE164996_S2_filtered_matrix.mtx.gz")
    features_f = os.path.join(data_dir, "GSE164996_S2_filtered_features.tsv.gz")
    barcodes_f = os.path.join(data_dir, "GSE164996_S2_filtered_barcodes.tsv.gz")
    calls_f = os.path.join(data_dir, "GSE164996_S2_protospacer_calls_per_cell.csv.gz")
    required = [matrix_f, features_f, barcodes_f, calls_f]
    missing = [x for x in required if not os.path.exists(x)]
    if missing:
        raise FileNotFoundError("GSE164996 missing files: " + ", ".join(missing))

    features = pd.read_csv(
        features_f, sep="\t", header=None,
        names=["ensembl", "gene", "type"],
    )
    barcodes = pd.read_csv(
        barcodes_f, sep="\t", header=None,
        names=["barcode"],
    )
    calls = pd.read_csv(calls_f)

    if calls["cell_barcode"].astype(str).duplicated().any():
        raise RuntimeError("GSE164996 protospacer-call table contains duplicate cell barcodes")
    if barcodes["barcode"].astype(str).duplicated().any():
        raise RuntimeError("GSE164996 RNA barcode file contains duplicate barcodes")

    with gzip.open(matrix_f, "rb") as fh:
        Xg = mmread(fh).tocsr()
    if Xg.shape != (len(features), len(barcodes)):
        raise RuntimeError(
            f"GSE164996 unexpected RNA matrix shape {Xg.shape}; expected "
            f"({len(features)}, {len(barcodes)}) genes x cells"
        )
    Xraw = Xg.T.tocsr().astype(np.float64)
    del Xg
    # Keep Gene Expression features only (drop CRISPR Guide Capture columns).
    ge_mask = (features["type"].astype(str) == "Gene Expression").to_numpy()
    Xraw = Xraw[:, ge_mask].tocsr()
    features = features.loc[ge_mask].reset_index(drop=True)

    barcode_to_idx = {
        str(b): i for i, b in enumerate(barcodes["barcode"].astype(str))
    }

    def guide_to_gene(g):
        g = str(g)
        if re.match(r"^CTRL\d+$", g):
            return g
        if g.startswith("hRosa26"):
            return "hRosa26"
        return re.sub(r"_\d+$", "", g)

    canonical_ctrl = {"CTRL0001", "CTRL0002"}
    ctrl_ctrl_cells = []
    gene_ctrl_cells = defaultdict(list)
    pair_cells = defaultdict(list)

    for row in calls.itertuples(index=False):
        bc = str(row.cell_barcode)
        if bc not in barcode_to_idx:
            continue

        guides = str(row.feature_call).split("|")
        if len(guides) != 2:
            continue

        a, b = [guide_to_gene(x) for x in guides]
        if a == b:
            continue

        idx = barcode_to_idx[bc]
        a_ctrl = a in canonical_ctrl
        b_ctrl = b in canonical_ctrl

        if a_ctrl and b_ctrl:
            ctrl_ctrl_cells.append(idx)
        elif a_ctrl ^ b_ctrl:
            gene = b if a_ctrl else a
            if gene != "hRosa26":
                gene_ctrl_cells[gene].append(idx)
        elif a != "hRosa26" and b != "hRosa26":
            pair_cells[tuple(sorted((a, b)))].append(idx)

    if not ctrl_ctrl_cells:
        raise RuntimeError("GSE164996 has no canonical CTRL+CTRL cells")

    genes = sorted(gene_ctrl_cells)
    gene_set = set(genes)
    eligible_keys = sorted([
        p for p in pair_cells
        if p[0] in gene_set and p[1] in gene_set
    ])
    if not eligible_keys:
        raise RuntimeError("GSE164996 has no eligible matched two-guide gene pairs")

    # Normalize every RNA cell once: counts per 10k followed by log1p.
    lib = np.asarray(Xraw.sum(axis=1)).reshape(-1)
    if np.any(lib <= 0):
        raise RuntimeError("GSE164996 contains zero-library RNA cells")
    X = sparse.diags(1e4 / lib) @ Xraw
    X.data = np.log1p(X.data)

    def mean_profile(indices):
        x = X[np.asarray(indices, dtype=np.int64)]
        mean = x.mean(axis=0)
        if hasattr(mean, "A1"):
            mean = mean.A1
        else:
            mean = np.asarray(mean).reshape(-1)
        return np.nan_to_num(mean).astype(float)

    mu_cc = mean_profile(ctrl_ctrl_cells)

    single_effects = {
        g: mean_profile(gene_ctrl_cells[g]) - mu_cc
        for g in genes
    }

    pair_effects, pair_map, pair_counts = {}, {}, {}
    for a, b in eligible_keys:
        name = f"{a}|{b}"
        pair_effects[name] = mean_profile(pair_cells[(a, b)]) - mu_cc
        pair_map[name] = (a, b)
        pair_counts[name] = len(pair_cells[(a, b)])

    pair_names = sorted(pair_effects)
    counts = list(pair_counts.values())

    print(
        f"  GSE164996: {len(single_effects)} matched gene+CTRL singles, "
        f"{len(pair_names)} eligible doubles, {X.shape[1]} genes, "
        f"{len(ctrl_ctrl_cells)} CTRL+CTRL controls"
    )
    print(
        "  GSE164996 preprocessing: exact two-guide matched design; "
        "CP10K + log1p; gene+CTRL singles and gene+gene doubles"
    )
    print(
        f"  GSE164996 pair cells: min={min(counts)}, "
        f"median={float(np.median(counts)):.1f}, max={max(counts)}; "
        f"observed={len(pair_names)}/{len(genes)*(len(genes)-1)//2} possible pairs"
    )

    return {
        "single_effects": single_effects,
        "pair_effects": pair_effects,
        "comp_eligible": pair_names,
        "pert_gene2idx": {g: i for i, g in enumerate(genes)},
        "n_genes_out": X.shape[1],
        "pair_genes_fn": lambda c: list(pair_map[str(c)]),
        "combine_mode": "sum",
    }


def validate_dataset(data, dataset_name):
    required = {
        "single_effects", "pair_effects", "comp_eligible", "pert_gene2idx",
        "n_genes_out", "pair_genes_fn", "combine_mode",
    }
    missing = required.difference(data)
    if missing:
        raise KeyError(f"{dataset_name}: dataset loader missing keys {sorted(missing)}")

    if data["combine_mode"] != "sum":
        raise RuntimeError(
            f"{dataset_name}: unsupported combine_mode {data['combine_mode']!r}"
        )

    eligible = list(data["comp_eligible"])
    if len(eligible) != len(set(eligible)):
        raise RuntimeError(f"{dataset_name}: duplicate eligible condition labels detected")

    n_genes_out = int(data["n_genes_out"])
    single_effects = data["single_effects"]
    pair_effects = data["pair_effects"]
    parse_pair = data["pair_genes_fn"]

    for gene, effect in single_effects.items():
        x = np.asarray(effect)
        if x.shape != (n_genes_out,):
            raise RuntimeError(
                f"{dataset_name}: single {gene!r} has shape {x.shape}, "
                f"expected {(n_genes_out,)}"
            )
        if not np.all(np.isfinite(x)):
            raise RuntimeError(f"{dataset_name}: single {gene!r} contains non-finite values")

    canonical = {}
    for pair in eligible:
        if pair not in pair_effects:
            raise RuntimeError(
                f"{dataset_name}: eligible pair {pair!r} missing from pair_effects"
            )
        genes = list(parse_pair(pair))
        if len(genes) != 2 or genes[0] == genes[1]:
            raise RuntimeError(f"{dataset_name}: invalid pair {pair!r} -> {genes}")
        if any(g not in single_effects for g in genes):
            raise RuntimeError(
                f"{dataset_name}: pair {pair!r} lacks an observed single perturbation"
            )

        key = tuple(sorted(genes))
        if key in canonical:
            raise RuntimeError(
                f"{dataset_name}: duplicate biological pair {key} appears as both "
                f"{canonical[key]!r} and {pair!r}"
            )
        canonical[key] = pair

        x = np.asarray(pair_effects[pair])
        if x.shape != (n_genes_out,):
            raise RuntimeError(
                f"{dataset_name}: pair {pair!r} has shape {x.shape}, "
                f"expected {(n_genes_out,)}"
            )
        if not np.all(np.isfinite(x)):
            raise RuntimeError(f"{dataset_name}: pair {pair!r} contains non-finite values")

    if set(data["pert_gene2idx"]) != set(single_effects):
        raise RuntimeError(
            f"{dataset_name}: pert_gene2idx and single_effects contain different gene sets"
        )
    return True


def split_pairs_60_20_20(pairs, seed):
    """Split eligible double perturbations only; all measured singles stay observed."""
    pairs = list(pairs)
    n = len(pairs)
    if n < 5:
        raise ValueError(f"Need at least 5 eligible pairs for 60/20/20 split; got {n}")
    if not np.isclose(TRAIN_FRAC + VAL_FRAC + TEST_FRAC, 1.0):
        raise ValueError("TRAIN_FRAC + VAL_FRAC + TEST_FRAC must equal 1")

    rng = np.random.RandomState(int(seed))
    order = rng.permutation(n)
    shuffled = [pairs[i] for i in order]

    n_val = max(1, int(round(VAL_FRAC * n)))
    n_test = max(1, int(round(TEST_FRAC * n)))
    if n_val + n_test >= n:
        raise ValueError(f"Not enough pairs for non-empty train/val/test partitions: n={n}")

    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]

    train_set, val_set, test_set = set(train), set(val), set(test)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Pair split overlap detected")
    if len(train_set | val_set | test_set) != n:
        raise RuntimeError("Pair split does not cover all eligible pairs exactly once")
    return train, val, test


def load(args):
    """Load one of the four datasets named by args.dataset."""
    paths = {"norman": os.path.join(args.data_dir, "norman"), "wessels2023": args.wessels_path,
             "directcapture": args.directcapture_dir, "gse164996": args.gse164996_dir}
    if not os.path.exists(paths[args.dataset]):
        raise FileNotFoundError(f"local data not found: {os.path.abspath(paths[args.dataset])}")
    if args.dataset == "norman":
        return load_norman(args.data_dir)
    if args.dataset == "wessels2023":
        return load_wessels(args.wessels_path)
    if args.dataset == "directcapture":
        return load_directcapture(args.directcapture_dir)
    return load_gse164996(args.gse164996_dir)


# ---------------------------------------------------------------- independent cell halves
# Used for the measurement noise sigma of each program and for the ceiling (do two halves agree on the sign).


def _split_halves(Xc, rows_of, ctrl_key, single, parse, pairs, seed, n_rep):
    """rows_of: condition key -> cell rows. Per rep, r_ab computed separately from two cell halves."""
    def mean_rows(r):
        return np.asarray(Xc[r].mean(axis=0)).reshape(-1)

    reps = []
    for rep_i in range(n_rep):
        rng = np.random.default_rng(seed + rep_i)
        half = {}
        for c, rows in rows_of.items():
            rows = rng.permutation(rows)
            h = len(rows) // 2
            half[c] = [mean_rows(rows[:h]), mean_rows(rows[h:2 * h])]
        out = {}
        for p in pairs:
            a, b = parse(p)
            out[p] = [half[p][k] - half[ctrl_key][k] - (half[single[a]][k] - half[ctrl_key][k])
                      - (half[single[b]][k] - half[ctrl_key][k]) for k in (0, 1)]
        reps.append(out)
    return reps if n_rep > 1 else reps[0]


def norman_halves(data_dir, data, pairs, seed=0, n_rep=1):
    """Same conditions and control as load_norman, cells split in half."""
    from gears import PertData
    pdata = PertData(data_dir)
    pdata.load(data_name="norman")
    ad = pdata.adata
    cond = ad.obs["condition"].astype(str).to_numpy()
    Xc = ad.X.tocsr() if hasattr(ad.X, "tocsr") else np.asarray(ad.X)
    parse = data["pair_genes_fn"]
    single = {}
    for c in dict.fromkeys(cond):
        if c != "ctrl" and len(parse(c)) == 1:
            single[parse(c)[0]] = c                  # last wins, as in load_norman
    need = sorted(set(pairs) | {single[g] for p in pairs for g in parse(p)} | {"ctrl"})
    rows_of = {c: np.nonzero(cond == c)[0] for c in need}
    return _split_halves(Xc, rows_of, "ctrl", single, parse, pairs, seed, n_rep)


def wessels_halves(path, data, pairs, seed=0, n_rep=1):
    """Same conditions and control as load_wessels (NT guide class, raw X means), cells split in half."""
    import anndata
    ad = anndata.read_h5ad(path)
    gc = ad.obs["Guide.Class"].astype(str).to_numpy()
    cond = ad.obs["condition"].astype(str).to_numpy()
    ctrl = np.nonzero(gc == "NT")[0]
    if len(ctrl) == 0:
        ctrl = np.nonzero(cond == "ctrl")[0]
    Xc = ad.X.tocsr() if hasattr(ad.X, "tocsr") else np.asarray(ad.X)
    single = {}
    for label in sorted(set(cond[gc == "Single"])):
        genes = [g for g in label.split("+") if g and g != "ctrl"]
        if len(genes) == 1:
            single[genes[0]] = label                 # last in sorted order wins, as in load_wessels
    parse = data["pair_genes_fn"]
    rows_of = {"__ctrl__": ctrl}
    for p in pairs:
        rows_of[p] = np.nonzero(cond == str(p))[0]
        for g in parse(p):
            rows_of[single[g]] = np.nonzero(cond == single[g])[0]
    return _split_halves(Xc, rows_of, "__ctrl__", single, parse, pairs, seed, n_rep)


def gse164996_halves(data_dir, data, pairs, seed=0, n_rep=1):
    """Same cells and normalisation as load_gse164996 (exact two guide cells, CP10K + log1p,
    CTRL+CTRL reference, gene+CTRL singles, gene+gene pairs), cells split in half."""
    import pandas as pd
    from scipy import sparse
    from scipy.io import mmread

    d = os.path.abspath(data_dir)
    features = pd.read_csv(os.path.join(d, "GSE164996_S2_filtered_features.tsv.gz"), sep="\t",
                           header=None, names=["ensembl", "gene", "type"])
    barcodes = pd.read_csv(os.path.join(d, "GSE164996_S2_filtered_barcodes.tsv.gz"), sep="\t",
                           header=None, names=["barcode"])
    calls = pd.read_csv(os.path.join(d, "GSE164996_S2_protospacer_calls_per_cell.csv.gz"))
    with gzip.open(os.path.join(d, "GSE164996_S2_filtered_matrix.mtx.gz"), "rb") as fh:
        Xraw = mmread(fh).tocsr().T.tocsr().astype(np.float64)
    ge = (features["type"].astype(str) == "Gene Expression").to_numpy()
    Xraw = Xraw[:, ge].tocsr()
    lib = np.asarray(Xraw.sum(axis=1)).reshape(-1)
    X = (sparse.diags(1e4 / lib) @ Xraw).tocsr()
    X.data = np.log1p(X.data)

    bidx = {str(b): i for i, b in enumerate(barcodes["barcode"].astype(str))}

    def g2g(g):
        g = str(g)
        if re.match(r"^CTRL\d+$", g):
            return g
        if g.startswith("hRosa26"):
            return "hRosa26"
        return re.sub(r"_\d+$", "", g)

    ctrl_set = {"CTRL0001", "CTRL0002"}
    cc, gc, pc = [], defaultdict(list), defaultdict(list)
    for row in calls.itertuples(index=False):
        bc = str(row.cell_barcode)
        if bc not in bidx:
            continue
        guides = str(row.feature_call).split("|")
        if len(guides) != 2:
            continue
        a, b = [g2g(x) for x in guides]
        if a == b:
            continue
        i = bidx[bc]
        ac, bcx = a in ctrl_set, b in ctrl_set
        if ac and bcx:
            cc.append(i)
        elif ac ^ bcx:
            gene = b if ac else a
            if gene != "hRosa26":
                gc[gene].append(i)
        elif a != "hRosa26" and b != "hRosa26":
            pc[tuple(sorted((a, b)))].append(i)

    parse = data["pair_genes_fn"]
    single = {}
    rows_of = {"__ctrl__": np.array(cc)}
    for p in pairs:
        a, b = parse(p)
        rows_of[p] = np.array(pc[tuple(sorted((a, b)))])
        for g in (a, b):
            single[g] = "single:" + g
            rows_of[single[g]] = np.array(gc[g])
    return _split_halves(X, rows_of, "__ctrl__", single, parse, pairs, seed, n_rep)


def directcapture_halves(data_dir, data, pairs, seed=0, n_rep=1):
    """Same cells and normalisation as load_directcapture (CP10K + log1p, NT+NT control), cells split in half."""
    import pandas as pd
    from scipy import sparse
    from scipy.io import mmread

    d = os.path.abspath(data_dir) + "/"
    meta = pd.read_csv(d + "GSM4367984_exp6.cell_identities.csv.gz")
    feat = pd.read_csv(d + "GSM4367984_exp6.features.tsv.gz", sep="\t", header=None)
    bars = pd.read_csv(d + "GSM4367984_exp6.barcodes.tsv.gz", sep="\t", header=None)[0].astype(str)
    with gzip.open(d + "GSM4367984_exp6.matrix.mtx.gz", "rb") as fh:
        Xg = mmread(fh).tocsr()
    ge = (feat.iloc[:, 2].astype(str) == "Gene Expression").to_numpy()
    b2i = {b: i for i, b in enumerate(bars)}
    X = Xg[ge][:, np.array([b2i[b] for b in meta["cell_barcode"].astype(str)])].T.tocsr().astype(np.float64)
    del Xg
    lib = np.asarray(X.sum(1)).ravel()
    keep = np.asarray(X.mean(0)).ravel() >= 1.0
    Xn = (sparse.diags(1e4 / lib) @ X).tocsr()[:, keep]
    Xn.data = np.log1p(Xn.data)
    A = meta["target_A"].astype(str).to_numpy()
    B = meta["target_B"].astype(str).to_numpy()
    NT = "Non-Targeting"
    parse = data["pair_genes_fn"]
    single = {}
    rows_of = {"__ctrl__": np.nonzero((A == NT) & (B == NT))[0]}
    for p in pairs:
        a, b = parse(p)
        rows_of[p] = np.nonzero(((A == a) & (B == b)) | ((A == b) & (B == a)))[0]
        for g in (a, b):
            single[g] = "single:" + g
            rows_of[single[g]] = np.nonzero(((A == g) & (B == NT)) | ((A == NT) & (B == g)))[0]
    return _split_halves(Xn, rows_of, "__ctrl__", single, parse, pairs, seed, n_rep)


def dataset_halves(args, data, pairs, n_rep):
    """List of n_rep replicates; each maps pair -> [r_ab from half 1, r_ab from half 2]."""
    fn = {"norman": lambda: norman_halves(args.data_dir, data, pairs, n_rep=n_rep),
          "wessels2023": lambda: wessels_halves(args.wessels_path, data, pairs, n_rep=n_rep),
          "gse164996": lambda: gse164996_halves(args.gse164996_dir, data, pairs, n_rep=n_rep),
          "directcapture": lambda: directcapture_halves(args.directcapture_dir, data, pairs, n_rep=n_rep)}
    reps = fn[args.dataset]()
    return reps if isinstance(reps, list) else [reps]
