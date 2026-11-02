"""Train one XGBoost model per task and save them for the demo app.

Also saves a small set of held-out test samples so the demo can pick a
random real recording the models never trained on.

Run:  python save_models.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

DATA = "../data/features.parquet"
OUT = "../models"
TASKS = ["presence", "family", "mode"]


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(DATA)
    feat = [c for c in df.columns if c.startswith("f")]
    X = df[feat].to_numpy()
    groups = df["recording"].to_numpy()

    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0)
                  .split(X, df["presence"], groups))

    for task in TASKS:
        y = df[task].to_numpy()
        model = XGBClassifier(n_estimators=150, max_depth=5,
                              tree_method="hist", n_jobs=-1)
        model.fit(X[tr], y[tr])
        joblib.dump(model, f"{OUT}/xgb_{task}.pkl")
        print(f"saved {task} model")

    # small held-out sample pool for the demo (test recordings only)
    sample = df.iloc[te].sample(min(600, len(te)), random_state=0)
    sample.to_parquet(f"{OUT}/demo_samples.parquet")
    print(f"saved {len(sample)} demo samples")


if __name__ == "__main__":
    main()
