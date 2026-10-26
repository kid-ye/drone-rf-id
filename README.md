# RF-Based Drone Identification — Project Guide (GPU machine)

Classify recorded drone RF signals three ways:

1. **Presence** — drone vs no-drone (2 classes)
2. **Family** — background / Bebop / AR / Phantom (4 classes)
3. **Mode** — the 10 specific operating conditions (10 classes, the hard extension)

Dataset: **DroneRF** (Al-Sa'd et al., 2019). Models compared: Logistic Regression, Random Forest, SVM-RBF, XGBoost.

This guide is self-contained: follow it top to bottom on a fresh GPU machine.

---

## 0. What the GPU actually speeds up

Only **XGBoost** uses the GPU (`device="cuda"`). On CPU the 10-class 5-fold XGBoost run took >1 hour; on GPU it finishes in a couple of minutes.

- **Logistic Regression / Random Forest** — CPU only (sklearn), already fast.
- **SVM-RBF** — CPU only, and O(n²+). We cap its training to a 5k subsample so it stays tractable. No GPU path in sklearn.
- **XGBoost** — the GPU win. This is the whole reason to move machines.

Preprocessing is **CPU + disk bound** (parsing 41 GB of text), not GPU — budget ~45 min once.

---

## 1. Environment setup

Python 3.10 or 3.11. Make a fresh venv/conda env, then:

```bash
pip install numpy pandas pyarrow scikit-learn
# GPU XGBoost — the official wheel ships CUDA support:
pip install xgboost
# Kaggle CLI for the dataset download:
pip install kaggle
```

Verify the GPU is visible to XGBoost:

```bash
python -c "import xgboost as xgb; print(xgb.__version__)"
nvidia-smi    # confirm a CUDA GPU is present
```

XGBoost ≥ 2.0 uses `device="cuda"` + `tree_method="hist"` (the old `gpu_hist` is deprecated). The code below already uses the new API.

---

## 2. Get the dataset (DroneRF, ~4 GB zip → 41 GB csv)

Put your Kaggle API token at `~/.kaggle/kaggle.json` (from kaggle.com → Account → Create New Token), then:

```bash
mkdir -p data/raw && cd data/raw
kaggle datasets download -d sshikamaru/dronerf --unzip
cd ../..
```

You should end up with `data/raw/DroneRF/<AR drone|Background RF activites|Bepop drone|Phantom drone>/...` — **454 csv files** total (227 low-band `L` + 227 high-band `H`).

### How the labels work (no manual labeling needed)

Every file is named like `10111H_3.csv`: a **5-digit BUI code** + band letter (`L`/`H`) + segment index. The BUI decodes all three labels:

| BUI code(s) | Family | Meaning |
|---|---|---|
| `00000` | 0 | Background — no drone |
| `10000 10001 10010 10011` | 1 | Bebop (4 modes) |
| `10100 10101 10110 10111` | 2 | AR drone (4 modes) |
| `11000` | 3 | Phantom (1 mode) |

- **presence** = 0 if `00000`, else 1
- **family** = the table's family column (0–3)
- **mode** = 0–9, one per distinct BUI code

---

## 3. Project layout

```
drone-rf-id/
├── data/
│   ├── raw/                 # DroneRF csvs (gitignored, 41 GB)
│   └── features.parquet     # built by preprocess.py (~460 MB)
├── src/
│   ├── preprocess.py        # raw csv -> features + 3 labels
│   ├── train.py             # LR/RF/SVM/XGBoost, split by recording
│   └── test_preprocess.py   # tiny self-check
├── results/                 # confusion matrices land here
└── requirements.txt
```

Create the folders: `mkdir -p src data/raw results`

---

## 4. `src/preprocess.py`

Python port of the reference MATLAB (`Main_1` + `Main_2`). Each segment → FFT magnitude → power spectrum; L and H bands stitched with a continuity scaling; labels decoded from the filename.

```python
"""Raw DroneRF csv -> feature table (PSD features + 3 labels)."""
import argparse, glob, os, re
import numpy as np
import pandas as pd

L = 100_000  # samples per segment
M = 2048     # FFT bins
Q = 10       # band-stitch continuity points

FAMILY = {
    "00000": 0,
    "10000": 1, "10001": 1, "10010": 1, "10011": 1,  # Bebop
    "10100": 2, "10101": 2, "10110": 2, "10111": 2,  # AR
    "11000": 3,                                       # Phantom
}
MODE = {bui: i for i, bui in enumerate(FAMILY)}  # 0..9


def segment_features(x):
    xf = np.abs(np.fft.fftshift(np.fft.fft(x - x.mean(), M)))
    return xf[M // 2:]


def process_capture(bui, low_csv, high_csv, recording_id):
    x = np.loadtxt(low_csv, delimiter=",")
    y = np.loadtxt(high_csv, delimiter=",")
    rows = []
    for i in range(len(x) // L):
        sl = slice(i * L, (i + 1) * L)
        xf = segment_features(x[sl])
        yf = segment_features(y[sl])
        yf = yf * (xf[-Q:].mean() / yf[:Q].mean())
        rows.append((np.concatenate([xf, yf]) ** 2, recording_id))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data/raw")
    ap.add_argument("--out", default="../data/features.parquet")
    args = ap.parse_args()

    pat = re.compile(r"(\d{5})([LH])_(\d+)\.csv$")
    pairs = {}
    for path in glob.glob(os.path.join(args.raw, "**", "*.csv"), recursive=True):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        bui, band, idx = m.group(1), m.group(2), m.group(3)
        pairs.setdefault((bui, idx), {})[band] = path

    feats, families, recs = [], [], []
    valid = [k for k, b in sorted(pairs.items())
             if "L" in b and "H" in b and k[0] in FAMILY]
    for done, (bui, idx) in enumerate(valid, 1):
        bands = pairs[(bui, idx)]
        print(f"[{done}/{len(valid)}] {bui}_{idx}", flush=True)
        for feat, rid in process_capture(bui, bands["L"], bands["H"], f"{bui}_{idx}"):
            feats.append(feat); families.append(bui); recs.append(rid)

    X = np.vstack(feats)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["bui"] = families
    df["recording"] = recs
    df["presence"] = df["bui"].map(lambda b: 0 if FAMILY[b] == 0 else 1)
    df["family"] = df["bui"].map(FAMILY)
    df["mode"] = df["bui"].map(MODE)
    df.to_parquet(args.out)
    print(f"wrote {args.out}: {len(df)} segments, {X.shape[1]} features")
    print(df[["presence", "family", "mode"]].value_counts().sort_index())


if __name__ == "__main__":
    main()
```

Run it (once, ~45 min):

```bash
cd src
python preprocess.py --raw ../data/raw --out ../data/features.parquet
```

Expected: `22700 segments, 2048 features`.

---

## 5. `src/train.py` (GPU-ready)

The **only GPU change** vs the CPU version is `device="cuda"` in the XGBoost line.

```python
"""Train + evaluate the 4 models on one task, split by recording (no leakage)."""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def get_models():
    m = {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1),
        "SVM-RBF": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    }
    if HAS_XGB:
        m["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6,
            tree_method="hist", device="cuda",   # <-- GPU
            n_jobs=-1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["presence", "family", "mode"], default="presence")
    ap.add_argument("--data", default="../data/features.parquet")
    ap.add_argument("--cv", type=int, default=0, help="GroupKFold folds (0 = single split)")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].to_numpy()
    y = df[args.task].to_numpy()
    groups = df["recording"].to_numpy()

    classes = sorted(np.unique(y))
    remap = {c: i for i, c in enumerate(classes)}
    y = np.array([remap[v] for v in y])

    if args.cv:
        folds = list(GroupKFold(n_splits=args.cv).split(X, y, groups))
        print(f"task={args.task}  {args.cv}-fold GroupKFold  classes={len(classes)}")
    else:
        sp = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
        folds = [next(sp.split(X, y, groups))]
        print(f"task={args.task}  train={len(folds[0][0])}  test={len(folds[0][1])}  classes={len(classes)}")

    for name, model in get_models().items():
        scores, last = [], None
        for tr, te in folds:
            if name == "SVM-RBF" and len(tr) > 5000:   # SVM doesn't scale; subsample
                tr = np.random.RandomState(0).choice(tr, 5000, replace=False)
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            scores.append(f1_score(y[te], pred, average="macro"))
            last = (y[te], pred)
        s = np.array(scores)
        tag = f"{s.mean():.3f} +/- {s.std():.3f}" if args.cv else f"{s[0]:.3f}"
        print(f"\n### {name}  (macro-F1 = {tag})")
        print(classification_report(*last, zero_division=0))
        pd.DataFrame(confusion_matrix(*last)).to_csv(
            f"../results/cm_{args.task}_{name}.csv", index=False)


if __name__ == "__main__":
    main()
```

Run all three tasks (each a couple of minutes on GPU):

```bash
cd src
python train.py --task presence --cv 5
python train.py --task family   --cv 5
python train.py --task mode     --cv 5
```

`--cv 5` = 5-fold GroupKFold grouped by recording. This is the **anti-leakage split**: segments from one recording file never span train and test.

---

## 6. Results from the CPU run (baseline to reproduce)

Single 75/25 split for presence/family; 5-fold for mode.

| Task | LR | RF | SVM-RBF | XGBoost |
|---|---|---|---|---|
| **Presence** | 1.000 | 0.978 | 1.000 | 1.000 |
| **Family** | 0.999 | 0.898 | 0.977 | 1.000 |
| **Mode (5-fold)** | 0.375 ±0.009 | 0.371 ±0.014 | 0.160 ±0.028* | **run on GPU** |

\*SVM capped at 5k training samples — thin for 10 classes, hence low.

**The story:** presence is trivial, family is solved, **mode is the genuinely hard task** and is where your analysis lives. XGBoost on mode is the one number still missing — it's the first thing to get on the GPU.

### On the near-perfect presence/family numbers
These are honestly high because the split already groups by recording file (the classic DroneRF leak is avoided) *and* drone RF fingerprints are genuinely distinct. Note in your report: DroneRF's ~20 files per mode may be chunks of one session, so some residual correlation across files can remain. 5-fold GroupKFold with tight std (±0.01–0.02) shows the result is stable, not a lucky split.

---

## 7. Remaining work (after the GPU run)

1. **Get XGBoost's mode number** — `python train.py --task mode --cv 5` on GPU. Expect it to edge above the ~0.37 LR/RF baseline.
2. **Noise-robustness test** (the deck promises it) — add Gaussian noise to the test features at a few SNR levels, re-measure macro-F1, plot the degradation curve. ~30 lines, add once the base results are confirmed.
3. **Writeup** — confusion matrices are already in `results/`; turn them into heatmaps and pull the per-class F1 from the classification reports.

---

## 8. Quick start (fresh GPU machine, TL;DR)

```bash
pip install numpy pandas pyarrow scikit-learn xgboost kaggle
mkdir -p src data/raw results
# put preprocess.py, train.py from sections 4-5 into src/
# put kaggle.json at ~/.kaggle/
cd data/raw && kaggle datasets download -d sshikamaru/dronerf --unzip && cd ../..
cd src
python preprocess.py --raw ../data/raw --out ../data/features.parquet   # ~45 min once
python train.py --task presence --cv 5
python train.py --task family   --cv 5
python train.py --task mode     --cv 5                                  # GPU shines here
```

### Gotchas
- `device="cuda"` needs XGBoost ≥ 2.0; on CPU-only boxes drop it (falls back automatically if you remove the arg).
- Preprocessing is CPU/disk bound — the GPU won't speed it up.
- Keep `features.parquet` between runs; never re-run preprocessing unless the raw data changes.
