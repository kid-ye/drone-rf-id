"""CLI demo for RF-based drone identification.

Three commands, run one at a time:

    python cli_demo.py predict            # classify a held-out recording
    python cli_demo.py matrix             # confusion matrices, all 4 models
    python cli_demo.py noise              # noise stress test

Options:
    predict / noise : --i N   pick recording 0..5 (default random)
    matrix          : --task {presence,family,mode}   --png
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

M = "../models"
NBINS, Q = 2048, 10
PRESENCE = {0: "No drone", 1: "Drone present"}
FAMILY = {0: "Background", 1: "Parrot Bebop", 2: "Parrot AR", 3: "DJI Phantom"}
MODE = {0: "Background", 1: "Bebop mode 1", 2: "Bebop mode 2", 3: "Bebop mode 3",
        4: "Bebop mode 4", 5: "AR mode 1", 6: "AR mode 2", 7: "AR mode 3",
        8: "AR mode 4", 9: "Phantom"}
NAMES = {"presence": PRESENCE, "family": FAMILY, "mode": MODE}
SHORT = {"presence": ["NoDrone", "Drone"],
         "family": ["Bgnd", "Bebop", "AR", "Phantom"],
         "mode": [f"m{i}" for i in range(10)]}
MODELS4 = ["LogisticRegression", "RandomForest", "SVM-RBF", "XGBoost"]

_WIN = np.hanning(NBINS)
_WIN_RMS = np.sqrt((_WIN ** 2).mean())


def welch_mag(x):
    w = sliding_window_view(x - x.mean(), NBINS)[:: NBINS // 2]
    spec = np.fft.fftshift(np.fft.fft(w * _WIN, axis=1), axes=1)
    return np.sqrt((np.abs(spec) ** 2).mean(axis=0))[NBINS // 2:] / _WIN_RMS


def pipeline(L, H):
    xf, yf = welch_mag(L), welch_mag(H)
    yf = yf * (xf[-Q:].mean() / yf[:Q].mean())
    return (np.concatenate([xf, yf]) ** 2).reshape(1, -1)


def add_noise(x, snr_db, rng):
    npow = (x ** 2).mean() / (10 ** (snr_db / 10))
    return x + rng.normal(0, np.sqrt(npow), x.shape)


def rule(t):
    print("\n" + "=" * 64 + f"\n  {t}\n" + "=" * 64)


def print_cm(cm, labels):
    w = max(7, max(len(l) for l in labels) + 1)
    print("    " + " " * w + "predicted ->")
    print("    " + " " * w + "".join(f"{l:>{w}}" for l in labels))
    for i, l in enumerate(labels):
        row = "".join(f"{cm[i, j]:>{w}}" for j in range(len(labels)))
        acc = cm[i, i] / cm[i].sum() * 100 if cm[i].sum() else 0
        print(f"    {l:>{w}}{row}   ({acc:4.0f}% recall)")
    print(f"\n    Overall accuracy {np.trace(cm) / cm.sum() * 100:.1f}%   "
          f"(n = {cm.sum()} held-out segments)")


def spotlight(i):
    raw = joblib.load(f"{M}/raw_demo.pkl")
    i = i if i is not None else int(np.random.randint(len(raw)))
    s = raw[i]
    x = pipeline(np.asarray(s["L"], float), np.asarray(s["H"], float))
    return s, x


# ---------------------------------------------------------------- predict
def cmd_predict(args):
    s, x = spotlight(args.i)
    models = {t: joblib.load(f"{M}/xgb_{t}.pkl") for t in NAMES}
    allm = {n: joblib.load(f"{M}/{n}_family.pkl") for n in MODELS4}

    rule("Classification of a held-out recording")
    print(f"    File : {s['filename']}   (never seen in training)\n")
    print(f"    XGBoost on all three tasks:")
    for task in ["presence", "family", "mode"]:
        pred, true = int(models[task].predict(x)[0]), int(s[task])
        nm = NAMES[task]
        mark = "OK " if pred == true else "XX "
        print(f"      [{mark}] {task:9s} predicted {nm[pred]:<14s}  actual {nm[true]}")

    true = int(s["family"])
    print(f"\n    All four models on the family task "
          f"(actual: {FAMILY[true]}):")
    for n in MODELS4:
        pred = int(allm[n].predict(x)[0])
        mark = "OK " if pred == true else "XX "
        print(f"      [{mark}] {n:<20s} {FAMILY[pred]}")
    print()


# ---------------------------------------------------------------- matrix
def cmd_matrix(args):
    from sklearn.metrics import confusion_matrix
    hd = pd.read_parquet(f"{M}/demo_samples.parquet")
    Xh = hd[[f"f{j}" for j in range(2048)]].to_numpy()
    yh = hd[args.task].to_numpy()
    labels = [SHORT[args.task][k] for k in sorted(np.unique(yh))]

    rule(f"Confusion matrix per model ({args.task})")
    cms = {}
    for n in MODELS4:
        yp = joblib.load(f"{M}/{n}_{args.task}.pkl").predict(Xh)
        cms[n] = confusion_matrix(yh, yp)
        print(f"\n  -- {n} --")
        print_cm(cms[n], labels)

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(9, 8))
        for ax, n in zip(axes.ravel(), MODELS4):
            cm = cms[n]
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
            for a in range(len(labels)):
                for b in range(len(labels)):
                    ax.text(b, a, cm[a, b], ha="center", va="center", fontsize=8,
                            color="white" if cm[a, b] > cm.max() / 2 else "black")
            ax.set_title(n, fontsize=10)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        fig.suptitle(f"Confusion matrices - {args.task} (600 held-out segments)")
        fig.tight_layout()
        out = f"../results/demo_cm_{args.task}.png"
        fig.savefig(out, dpi=130)
        print(f"\n    saved {out}")


# ---------------------------------------------------------------- noise
def cmd_noise(args):
    s, x = spotlight(args.i)
    fam = joblib.load(f"{M}/xgb_family.pkl")
    rng = np.random.RandomState(0)
    true = int(s["family"])

    rule("Noise stress test (family task)")
    print(f"    File   : {s['filename']}   (never seen in training)")
    print(f"    Actual : {FAMILY[true]}\n")
    print(f"    {'SNR':>7} | {'predicted':<14} | result")
    print(f"    {'-'*7}-+-{'-'*14}-+-------")
    for snr in ["clean", 20, 10, 5, 0, -5]:
        if snr == "clean":
            proba = fam.predict_proba(x)[0]
        else:
            draws = np.vstack([add_noise(x, snr, rng) for _ in range(25)])
            proba = fam.predict_proba(draws).mean(axis=0)
        pred = int(proba.argmax())
        tag = "clean" if snr == "clean" else f"{snr} dB"
        mark = "correct" if pred == true else "misclassified"
        print(f"    {tag:>7} | {FAMILY[pred]:<14} | {mark}")
    print()


ap = argparse.ArgumentParser(description="RF drone ID demo")
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("predict", help="classify a held-out recording")
p.add_argument("--i", type=int, default=None)
p.set_defaults(func=cmd_predict)
p = sub.add_parser("matrix", help="confusion matrices, all 4 models")
p.add_argument("--task", default="family", choices=list(NAMES))
p.add_argument("--png", action="store_true")
p.set_defaults(func=cmd_matrix)
p = sub.add_parser("noise", help="noise stress test")
p.add_argument("--i", type=int, default=None)
p.set_defaults(func=cmd_noise)

args = ap.parse_args()
args.func(args)
