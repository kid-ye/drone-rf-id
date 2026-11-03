"""Train all four models per task and save them for the demo app.

Also saves a small set of held-out test samples so the demo can pick a
random real recording the models never trained on, and compute live
confusion matrices for every model.

Run:  python save_models.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

DATA = "../data/features.parquet"
OUT = "../models"
TASKS = ["presence", "family", "mode"]
MODELS = ["LogisticRegression", "RandomForest", "SVM-RBF", "XGBoost"]


def build(name):
    if name == "LogisticRegression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1)
    if name == "SVM-RBF":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf"))
    return XGBClassifier(n_estimators=150, max_depth=5, tree_method="hist", n_jobs=-1)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(DATA)
    feat = [f"f{i}" for i in range(2048)]
    assert len(feat) == 2048 and "family" not in feat
    X = df[feat].to_numpy()
    groups = df["recording"].to_numpy()

    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0)
                  .split(X, df["presence"], groups))

    for task in TASKS:
        y = df[task].to_numpy()
        for name in MODELS:
            tri = tr
            if name == "SVM-RBF" and len(tr) > 5000:        # RBF SVM does not scale
                tri = np.random.RandomState(0).choice(tr, 5000, replace=False)
            m = build(name).fit(X[tri], y[tri])
            joblib.dump(m, f"{OUT}/{name}_{task}.pkl")
            if name == "XGBoost":                           # keep Streamlit demo working
                joblib.dump(m, f"{OUT}/xgb_{task}.pkl")
        print(f"saved 4 models for {task}")

    # small held-out sample pool for the demo (test recordings only)
    sample = df.iloc[te].sample(min(600, len(te)), random_state=0)
    sample.to_parquet(f"{OUT}/demo_samples.parquet")
    print(f"saved {len(sample)} demo samples")


if __name__ == "__main__":
    main()
