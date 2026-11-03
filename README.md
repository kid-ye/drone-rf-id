# RF-Based Drone Identification Using Machine Learning

Classify recorded drone radio-frequency (RF) signals three ways, using classical
machine learning on the public **DroneRF** dataset:

1. **Presence** — drone vs no-drone (2 classes)
2. **Family** — background / Parrot Bebop / Parrot AR / DJI Phantom (4 classes)
3. **Operating mode** — the 10 recorded operating conditions (10 classes)

Four models are compared on each task: Logistic Regression, Random Forest,
Support Vector Machine (RBF), and XGBoost — under a **leakage-aware** split.

Course project, Department of Electronics and Telecommunication Engineering,
BMS College of Engineering. Guide: Dr. C Gururaj.

---

## Results (macro-F1)

All scores are 5-fold GroupKFold macro-F1 (recording-grouped, leakage-free).

| Model | Presence | Family | Mode |
|---|---|---|---|
| Logistic Regression | 0.450 | 0.453 | 0.401 |
| Random Forest | 0.995 | 0.874 | 0.711 |
| Support Vector Machine | 0.450* | 0.430* | 0.262* |
| **XGBoost** | **0.997** | **0.906** | **0.806** |

\*SVM trained on a 5,000-sample subset per fold (it does not scale).

- **XGBoost wins every task, decisively.** Difficulty rises as expected:
  presence (0.997) → family (0.906) → mode (0.806).
- The linear models (Logistic Regression, SVM) sit near chance (~0.4), while the
  tree ensembles soar — which shows the problem is strongly **non-linear** and,
  crucially, confirms there is **no label leakage** (a leak would lift the linear
  models too).
- Features use **Welch averaging** over the whole 100,000-sample segment (96
  overlapping Hann windows), not just the first 2,048 samples — this is what
  pushes the hard mode task from 0.33 to **0.81**.
- **Noise-robustness test** (family task): XGBoost is the most accurate at every
  SNR and degrades gracefully (0.900 clean → 0.300 at −5 dB); Random Forest is
  second. See `results/noise_all.png`.

---

## Repository structure

```
drone-rf-id/
├── src/
│   ├── preprocess.py        # raw CSV -> 2048-D power-spectrum features + 3 labels
│   ├── train.py             # 4 models x 3 tasks, recording-grouped split
│   ├── save_models.py       # train + persist XGBoost models for the demo
│   ├── demo.py              # Streamlit app: live predictions + RF spectrum
│   ├── noise_test.py        # noise robustness (Random Forest, family)
│   ├── noise_all.py         # noise robustness, all 4 models
│   ├── plots.py            # confusion-matrix heatmaps + noise curves
│   └── test_preprocess.py   # self-check
├── results/                 # confusion matrices (CSV + PNG), noise curves
├── paper/                   # IEEE-format paper (LaTeX source + PDF)
├── ppt/                     # Beamer presentation (LaTeX source + PDF)
├── Viva_Preparation_Guide.docx
├── requirements.txt
└── README.md
```

Not in the repo (regenerated locally): `data/raw/` (41 GB, Kaggle download),
`data/features.parquet`, and `models/` (run `save_models.py`).

---

## How to run

```bash
pip install -r requirements.txt

# 1. Download the dataset (~41 GB) with the Kaggle API
mkdir -p data/raw && cd data/raw
kaggle datasets download -d sshikamaru/dronerf --unzip && cd ../..

# 2. Build the feature table (~45 min, one-time)
cd src
python preprocess.py --raw ../data/raw --out ../data/features.parquet

# 3. Train and evaluate (per task)
python train.py --task presence --cv 5
python train.py --task family   --cv 5
python train.py --task mode     --cv 5

# 4. Noise robustness and plots
python noise_all.py
python plots.py
```

### Live demo (Streamlit)

```bash
cd src
python save_models.py          # trains + saves models once
streamlit run demo.py          # opens at localhost:8501
```

The demo picks a random held-out recording, plots its RF power spectrum, and
shows the predicted presence / family / mode against the true labels.

---

## Method (summary)

- **Features:** each recording is cut into 100,000-sample windows. Each window is
  processed by **Welch averaging** — ~96 overlapping (50%) 2048-sample Hann
  windows, FFT, and averaged power spectra — so the *whole* segment contributes,
  not just its first 2,048 samples. The upper 1024 bins of each band are kept,
  low and high concatenated and squared to a power spectrum. Result: a 2048-D
  vector per segment, 22,700 segments total.
- **Labels:** decoded from the 5-digit Binary Unique Identifier (BUI) in each
  filename — no manual labelling.
- **Leakage control:** all segments of one recording stay on the same side of the
  split (`GroupKFold` / `GroupShuffleSplit`), so scores are honest.

---

## GPU note

Only XGBoost benefits from a GPU. On a CUDA machine, set
`device="cuda"` in `train.py`'s XGBoost and raise `n_estimators` to 300.
Everything else is CPU-bound; preprocessing is disk/CPU-bound and unaffected.

---

## Deliverables

- `paper/main.pdf` — IEEE-format project paper
- `ppt/ml_ppt.pdf` — presentation (18 slides)
- `Viva_Preparation_Guide.docx` — detailed Q&A viva prep
