"""CLI demo: pick a held-out recording, run the pipeline, print the numbers.

Run:  python cli_demo.py            # random recording
      python cli_demo.py --i 2      # specific one
      python cli_demo.py --full     # print all 2048 feature values
"""
import argparse
import joblib
import numpy as np

M = "../models"
NBINS, Q = 2048, 10
PRESENCE = {0: "No drone", 1: "Drone present"}
FAMILY = {0: "Background", 1: "Parrot Bebop", 2: "Parrot AR", 3: "DJI Phantom"}
MODE = {0: "Background", 1: "Bebop mode 1", 2: "Bebop mode 2", 3: "Bebop mode 3",
        4: "Bebop mode 4", 5: "AR mode 1", 6: "AR mode 2", 7: "AR mode 3",
        8: "AR mode 4", 9: "Phantom"}
NAMES = {"presence": PRESENCE, "family": FAMILY, "mode": MODE}


_WIN = np.hanning(NBINS)
_WIN_RMS = np.sqrt((_WIN ** 2).mean())


def seg_feat(x):
    """Welch averaging over the whole segment (matches preprocess.py)."""
    from numpy.lib.stride_tricks import sliding_window_view
    x = x - x.mean()
    w = sliding_window_view(x, NBINS)[:: NBINS // 2]
    spec = np.fft.fftshift(np.fft.fft(w * _WIN, axis=1), axes=1)
    psd = (np.abs(spec) ** 2).mean(axis=0)
    return np.sqrt(psd)[NBINS // 2:] / _WIN_RMS


def show(label, a, n=12):
    head = ", ".join(f"{v:.4f}" for v in a[:n])
    print(f"  {label}: [{head}, ...]  (len {len(a)}, min {a.min():.3f}, max {a.max():.3f})")


def rule(t):
    print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)


ap = argparse.ArgumentParser()
ap.add_argument("--i", type=int, default=None)
ap.add_argument("--full", action="store_true")
args = ap.parse_args()

models = {t: joblib.load(f"{M}/xgb_{t}.pkl") for t in NAMES}
raw = joblib.load(f"{M}/raw_demo.pkl")

i = args.i if args.i is not None else int(np.random.randint(len(raw)))
s = raw[i]
L, H = np.asarray(s["L"], float), np.asarray(s["H"], float)

rule("STEP 1  Selected recording")
print(f"  File : {s['filename']}")
print(f"  BUI  : {s['bui']}  (ground-truth code in the filename)")
print(f"  Raw samples per band: {len(L)} (low) + {len(H)} (high)")
show("Raw low-band  (time domain)", L)
show("Raw high-band (time domain)", H)

rule("STEP 2  FFT magnitude (upper 1024 bins per band)")
xf, yf = seg_feat(L), seg_feat(H)
show("FFT low  |X_f|", xf)
show("FFT high |Y_f|", yf)

rule("STEP 3  Band fusion + power spectrum (model input)")
yf = yf * (xf[-Q:].mean() / yf[:Q].mean())      # continuity scaling
feat = np.concatenate([xf, yf]) ** 2
show("2048-D power spectrum", feat)
if args.full:
    print("\n  Full feature vector:")
    print("  " + np.array2string(feat, threshold=np.inf, precision=4,
                                  max_line_width=100).replace("\n", "\n  "))

rule("STEP 4  Predictions")
x = feat.reshape(1, -1)
for task in ["presence", "family", "mode"]:
    pred = int(models[task].predict(x)[0])
    true = int(s[task])
    nm = NAMES[task]
    mark = "OK " if pred == true else "XX "
    print(f"  [{mark}] {task:9s} predicted {nm[pred]:<14s}  actual {nm[true]}")

conf = models["mode"].predict_proba(x)[0]
top = conf.argsort()[::-1][:3]
print("\n  Mode confidence (top 3): " +
      ", ".join(f"{MODE[k]} {conf[k]*100:.1f}%" for k in top))
print()
