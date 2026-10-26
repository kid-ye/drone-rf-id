"""Train + evaluate the 4 models on one task, split by recording (no leakage).

Tasks:  presence (2-class) | family (4-class) | mode (10-class)

Run:  python train.py --task presence
      python train.py --task family
"""
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
            n_estimators=300, max_depth=6, tree_method="hist", n_jobs=-1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["presence", "family", "mode"],
                    default="presence")
    ap.add_argument("--data", default="../data/features.parquet")
    ap.add_argument("--cv", type=int, default=0,
                    help="GroupKFold folds (0 = single split)")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].to_numpy()
    y = df[args.task].to_numpy()
    groups = df["recording"].to_numpy()  # split by recording, never by segment

    # XGBoost needs contiguous 0..K-1 labels
    classes = sorted(np.unique(y))
    remap = {c: i for i, c in enumerate(classes)}
    y = np.array([remap[v] for v in y])

    if args.cv:
        folds = list(GroupKFold(n_splits=args.cv).split(X, y, groups))
        print(f"task={args.task}  {args.cv}-fold GroupKFold  "
              f"classes={len(classes)}  (grouped by recording)")
    else:
        sp = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
        folds = [next(sp.split(X, y, groups))]
        print(f"task={args.task}  train={len(folds[0][0])}  "
              f"test={len(folds[0][1])}  classes={len(classes)}")

    for name, model in get_models().items():
        scores, last = [], None
        for tr, te in folds:
            # ponytail: SVM-RBF is O(n^2+); cap its training so it stays
            # tractable at this scale. Others train on the full fold.
            if name == "SVM-RBF" and len(tr) > 5000:
                rng = np.random.RandomState(0)
                tr = rng.choice(tr, 5000, replace=False)
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            scores.append(f1_score(y[te], pred, average="macro"))
            last = (y[te], pred)
        s = np.array(scores)
        tag = (f"{s.mean():.3f} +/- {s.std():.3f}" if args.cv
               else f"{s[0]:.3f}")
        print(f"\n### {name}  (macro-F1 = {tag})")
        print(classification_report(*last, zero_division=0))
        pd.DataFrame(confusion_matrix(*last)).to_csv(
            f"../results/cm_{args.task}_{name}.csv", index=False)


if __name__ == "__main__":
    main()
