"""Noise-robustness curves for all four models on the family task.

Trains each model on clean data, then measures macro-F1 as Gaussian noise is
added to the test set at decreasing SNR. Produces one figure with four curves.

Run:  python noise_all.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

SNRS = ["clean", 20, 10, 5, 0, -5]


def add_noise(X, snr_db, rng):
    p = (X ** 2).mean(axis=1, keepdims=True)
    npow = p / (10 ** (snr_db / 10))
    return X + rng.normal(0, np.sqrt(npow), X.shape)


def main():
    df = pd.read_parquet("../data/features.parquet")
    feat = [f"f{i}" for i in range(2048)]
    assert len(feat) == 2048 and "family" not in feat
    X, y = df[feat].to_numpy(), df["family"].to_numpy()
    g = df["recording"].to_numpy()
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(X, y, g))

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1),
        "Support Vector Machine": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=5, tree_method="hist", n_jobs=-1),
    }

    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    table = {}
    for name, m in models.items():
        tri = tr
        if name == "Support Vector Machine":
            tri = rng.choice(tr, 5000, replace=False)
        m.fit(X[tri], y[tri])
        scores = []
        for snr in SNRS:
            Xte = X[te] if snr == "clean" else add_noise(X[te], snr, rng)
            scores.append(f1_score(y[te], m.predict(Xte), average="macro"))
        table[name] = [round(s, 3) for s in scores]
        ax.plot([str(s) for s in SNRS], scores, "o-", label=name)
        print(name, table[name])

    ax.set_xlabel("SNR (dB), left = clean")
    ax.set_ylabel("macro-F1")
    ax.set_title("Noise robustness by model (family task)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("../results/noise_all.png", dpi=130)
    pd.DataFrame(table, index=[str(s) for s in SNRS]).to_csv("../results/noise_all.csv")
    print("saved ../results/noise_all.png")


if __name__ == "__main__":
    main()
