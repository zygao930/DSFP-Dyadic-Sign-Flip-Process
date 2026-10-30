# DSFP: Dyadic Sign-Flip Process

Code for estimating how reliable the **sign** of a predicted genetic interaction is for gene pairs that were never
measured together. DSFP keeps the predictions of a fixed predictor unchanged and returns, for each unseen pair,
the probability that the predicted sign is correct.

For a pair `x = (a, b)` the target is the interaction residual `r_x = δ_ab − δ_a − δ_b` projected on a training
interaction program, `z_x = w_kᵀ (r_x − r̄_train)`. DSFP models the out-of-fold errors of the Pair-only predictor
with a Gaussian process over gene pairs (single-effect kernel `K_S` plus shared-gene kernel `K_g`) and a
direction-aware error term, and reports

```
R(x) = Φ( ((1 + β)|μ_x| + s_x m_g(x)) / sqrt(V_x + ω²) )
```

## Repository layout

| File | Contents |
|:--|:--|
| `dsfp.py` | DSFP, Residual GP, the component ablations and the shuffled-feature negative control |
| `reliability.py` | Reliability baselines on the same fixed predictions (Magnitude, Bootstrap Stability, Model Agreement, Laplace variants, Split Conformal, EPICSCORE, Δ-KNN / Δ-MDS selectors, GP on Singles) |
| `prediction.py` | Prediction quality of the predictors (Ridge, Bilinear, kernel and GP on single effects, Pair-only + Laplace, GEARS, NAIAD): sign accuracy and MAE |
| `models.py` | Pair-only low-rank model and its Laplace posterior, GP on single effects, training-pair graph |
| `data.py` | Loaders for the four datasets, the 60/20/20 pair split and cell-half replicates |
| `gears_adapter.py` | Runs the official GEARS package on the same pairs and splits |

## Installation

Tested with Python 3.10.

```bash
conda create -n dsfp python=3.10 -y
conda activate dsfp
pip install -r requirements.txt
```

`requirements.txt` covers DSFP and all reliability baselines. The two external predictors in `prediction.py`
need extra packages:

```bash
# GEARS (also needed to download the Norman data)
pip install torch==2.0.0 torch_geometric==2.8.0 cell-gears==0.1.2
# NAIAD: clone the official repository and install it in editable mode
git clone <NAIAD repository URL> NAIAD
pip install -e NAIAD
```

## Data

All loaders read from `./data` by default (`--data-dir` changes the root; each dataset path can also be set with
`--wessels-path`, `--directcapture-dir` and `--gse164996-dir`).

```
data/
├── norman/                 downloaded automatically by GEARS (PertData.load("norman"))
├── go_essential_all/       downloaded automatically by GEARS
├── directcapture/          GEO GSM4367984 (GSE146194, exp6)
│   ├── GSM4367984_exp6.matrix.mtx.gz
│   ├── GSM4367984_exp6.features.tsv.gz
│   ├── GSM4367984_exp6.barcodes.tsv.gz
│   └── GSM4367984_exp6.cell_identities.csv.gz
├── gse164996/              GEO GSE164996, sample S2
│   ├── GSE164996_S2_filtered_matrix.mtx.gz
│   ├── GSE164996_S2_filtered_features.tsv.gz
│   ├── GSE164996_S2_filtered_barcodes.tsv.gz
│   └── GSE164996_S2_protospacer_calls_per_cell.csv.gz
└── wessels23/
    └── Wessels2023.h5ad
```

Preprocessing (fixed, independent of the split):

* **Norman**: GEARS processed data; effects are condition means minus the control mean.
* **DirectCapture**: genes with mean raw expression ≥ 1 UMI per cell, CP10K + log1p, effects relative to
  non-targeting + non-targeting cells, conditions with at least 30 cells. The result is cached as
  `directcapture_exp6_condition_effects_ge.npz` on first use.
* **GSE164996**: exact two-guide cells only, CP10K + log1p; singles are `gene + CTRL`, the reference is
  `CTRL + CTRL`; hRosa26 and same-gene doubles are excluded.
* **Wessels 2023**: effects relative to non-targeting cells.

## Protocol

Gene pairs whose two singles are observed are split 60% train / 20% calibration / 20% test; every single
perturbation stays observed. DSFP uses the training split only (4-fold out-of-fold predictions for the errors).
The calibration split is used only by baselines that need calibration. Seeds 42, 43, 44 were used for
development and seeds 101 to 105 for the confirmatory evaluation.

## Reproducing the paper

```bash
# Reliability: DSFP, Residual GP, ablations and negative control (confirmatory seeds 101..105)
python dsfp.py --confirm

# Reliability baselines on the same splits
python reliability.py --seeds 101,102,103,104,105

# Prediction quality (development seeds 42, 43, 44), including GEARS and NAIAD
python prediction.py --seeds 42,43,44
```

Useful options:

* `--datasets norman,directcapture,wessels2023,gse164996` selects datasets.
* `python dsfp.py` without `--confirm` runs the development protocol (nested CV inside each training split; test
  pairs are never used).
* `python dsfp.py --confirm --exploratory` also fits variants that the paper does not report (skew-normal
  decision layer, event-likelihood decision layer, no directional variance, in-sample residuals, two-stage slope
  calibration).
* `python prediction.py --methods ""` skips GEARS and NAIAD; `--device cpu` runs them on CPU.
* `python reliability.py --report` and `python prediction.py --report` reprint the tables from saved results.

Every script saves per-pair records (scores, predictions and correctness) as JSON under `results/`, so any metric
can be recomputed without rerunning.

## License

MIT, see `LICENSE`.
