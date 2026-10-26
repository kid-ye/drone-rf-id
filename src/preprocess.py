"""Raw DroneRF csv -> feature table (PSD features + 3 labels).

Python port of the reference MATLAB (Al-Sad/DroneRF, Main_1 + Main_2).
Each capture has an L (low-band) and H (high-band) csv. We segment,
FFT->power, stitch the bands, and decode all 3 labels from the BUI filename.

Run:  python preprocess.py --raw ../data/raw --out ../data/features.parquet
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

L = 100_000  # samples per segment  (MATLAB: L = 1e5)
M = 2048     # FFT bins             (MATLAB: M = 2048)
Q = 10       # band-stitch continuity points

# BUI -> family index. Presence and mode are derived from these too.
FAMILY = {
    "00000": 0,  # background / no drone
    "10000": 1, "10001": 1, "10010": 1, "10011": 1,  # Bebop
    "10100": 2, "10101": 2, "10110": 2, "10111": 2,  # AR
    "11000": 3,  # Phantom
}
MODE = {bui: i for i, bui in enumerate(FAMILY)}  # 0..9, one per BUI


def segment_features(x):
    """One L-length segment -> M/2 power-spectrum bins (mirrors MATLAB)."""
    xf = np.abs(np.fft.fftshift(np.fft.fft(x - x.mean(), M)))
    return xf[M // 2:]


def process_capture(bui, low_csv, high_csv, recording_id):
    x = np.loadtxt(low_csv, delimiter=",")
    y = np.loadtxt(high_csv, delimiter=",")
    n_seg = len(x) // L
    rows = []
    for i in range(n_seg):
        sl = slice(i * L, (i + 1) * L)
        xf = segment_features(x[sl])
        yf = segment_features(y[sl])
        # scale H onto L for spectral continuity, then square -> power
        yf = yf * (xf[-Q:].mean() / yf[:Q].mean())
        feat = np.concatenate([xf, yf]) ** 2
        rows.append((feat, recording_id))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data/raw")
    ap.add_argument("--out", default="../data/features.parquet")
    args = ap.parse_args()

    # group files by BUI + segment index: e.g. 10111L_3.csv / 10111H_3.csv
    pat = re.compile(r"(\d{5})([LH])_(\d+)\.csv$")
    pairs = {}
    for path in glob.glob(os.path.join(args.raw, "**", "*.csv"), recursive=True):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        bui, band, idx = m.group(1), m.group(2), m.group(3)
        pairs.setdefault((bui, idx), {})[band] = path

    feats, families, recs = [], [], []
    valid = [k for k, b in sorted(pairs.items())
             if "L" in b and "H" in b and k[0] in FAMILY]
    for done, (bui, idx) in enumerate(valid, 1):
        bands = pairs[(bui, idx)]
        print(f"[{done}/{len(valid)}] {bui}_{idx}", flush=True)
        rec_id = f"{bui}_{idx}"  # split by this to avoid leakage
        for feat, rid in process_capture(bui, bands["L"], bands["H"], rec_id):
            feats.append(feat)
            families.append(bui)
            recs.append(rid)

    X = np.vstack(feats)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["bui"] = families
    df["recording"] = recs
    df["presence"] = df["bui"].map(lambda b: 0 if FAMILY[b] == 0 else 1)
    df["family"] = df["bui"].map(FAMILY)
    df["mode"] = df["bui"].map(MODE)
    df.to_parquet(args.out)
    print(f"wrote {args.out}: {len(df)} segments, {X.shape[1]} features")
    print(df[["presence", "family", "mode"]].value_counts().sort_index())


if __name__ == "__main__":
    main()
