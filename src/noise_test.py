"""Noise-robustness test: train on clean signals, test under added Gaussian
noise at several SNR levels, and record how macro-F1 degrades.

This is the deck's "noise and interference" robustness check. The 2.4 GHz band
is shared with Wi-Fi/Bluetooth, so we want to know the model still works when
the test signal is noisier than training.

Run:  python noise_test.py --task family
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit


def add_noise(X, snr_db, rng):
    """Add white Gaussian noise to each sample at the given SNR (dB)."""
    sig_power = (X ** 2).mean(axis=1, keepdims=True)
    noise_power = sig_power / (10 ** (snr_db / 10))
    return X + rng.normal(0, np.sqrt(noise_power), X.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["presence", "family", "mode"],
                    default="family")
    ap.add_argument("--data", default="../data/features.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    feat = [f"f{i}" for i in range(2048)]
    assert len(feat) == 2048 and "family" not in feat
    X, y = df[feat].to_numpy(), df[args.task].to_numpy()
    groups = df["recording"].to_numpy()

    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=0)
                  .split(X, y, groups))
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    model.fit(X[tr], y[tr])  # trained once on clean data

    rng = np.random.RandomState(0)
    rows = []
    for snr in ["clean", 20, 10, 5, 0, -5]:
        Xte = X[te] if snr == "clean" else add_noise(X[te], snr, rng)
        f1 = f1_score(y[te], model.predict(Xte), average="macro")
        rows.append({"snr_db": snr, "macro_f1": round(f1, 3)})
        print(f"SNR={snr!s:>5}  macro-F1={f1:.3f}")

    out = f"../results/noise_{args.task}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
