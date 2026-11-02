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

| Model | Presence | Family | Mode (5-fold) |
|---|---|---|---|
| Logistic Regression | 1.000 | 0.999 | 0.375 |
| Random Forest | 0.978 | 0.898 | 0.378 |
| Support Vector Machine | 1.000 | 0.977 | 0.160* |
| **XGBoost** | **1.000** | **1.000** | **0.492** |

\*SVM trained on a 5,000-sample subset per fold (it does not scale).

- Presence and family are essentially solved; the 10-class **mode** task is hard,
  and **XGBoost leads** it at 0.492 (±0.009 over 5 folds).
- **Noise-robustness test:** family macro-F1 falls from 0.90 (clean) to 0.29
  (−5 dB SNR). Notably, **Random Forest is the most noise-robust**, while the
  high clean-accuracy models collapse under noise (see `results/noise_all.png`).

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

- **Features:** each recording is cut into 100,000-sample windows; a 2048-point
  FFT magnitude is taken, the upper 1024 bins kept, the low and high sub-bands
  concatenated and squared to a power spectrum. Result: a 2048-D vector per
  segment, 22,700 segments total.
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
