"""Turn the saved confusion-matrix CSVs into heatmap PNGs for the writeup.

Run:  python plots.py
Reads  ../results/cm_<task>_<model>.csv  ->  writes matching .png next to them.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {
    "presence": ["no drone", "drone"],
    "family": ["background", "Bebop", "AR", "Phantom"],
    "mode": [str(i) for i in range(10)],
}


def main():
    for csv in sorted(glob.glob("../results/cm_*.csv")):
        name = os.path.basename(csv)[3:-4]        # cm_family_XGBoost -> family_XGBoost
        task = name.split("_")[0]
        cm = pd.read_csv(csv).to_numpy()
        labels = LABELS.get(task, [str(i) for i in range(len(cm))])

        fig, ax = plt.subplots(figsize=(1 + len(cm), 1 + len(cm)))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(cm)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(cm)), labels)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(name)
        thresh = cm.max() / 2
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        out = csv[:-4] + ".png"
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"wrote {out}")


def noise_curves():
    for csv in sorted(glob.glob("../results/noise_*.csv")):
        task = os.path.basename(csv)[6:-4]
        d = pd.read_csv(csv)
        if "snr_db" not in d.columns:
            continue  # noise_all.csv is wide-format and self-plots; skip it
        x = [str(v) for v in d["snr_db"]]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, d["macro_f1"], "o-")
        ax.set_xlabel("SNR (dB), left=clean"); ax.set_ylabel("macro-F1")
        ax.set_title(f"noise robustness — {task}"); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = csv[:-4] + ".png"
        fig.savefig(out, dpi=120); plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
    noise_curves()
