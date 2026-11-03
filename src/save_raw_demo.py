"""Save a few raw RF segments (with their filenames) so the demo can run the
full pipeline live: raw samples -> FFT -> power spectrum -> prediction.

Run:  python save_raw_demo.py
"""
import glob
import os

import joblib
import numpy as np

RAW = "../data/raw"
OUT = "../models/raw_demo.pkl"
L = 100_000

FAMILY = {"00000": 0, "10000": 1, "10001": 1, "10010": 1, "10011": 1,
          "10100": 2, "10101": 2, "10110": 2, "10111": 2, "11000": 3}
MODE = {b: i for i, b in enumerate(FAMILY)}
# one representative recording per operating mode
PICK = ["00000", "10000", "10011", "10100", "10111", "11000"]


def read_head(path, n=L):
    """Read the first n comma-separated floats from a one-line CSV, fast."""
    buf = ""
    with open(path) as f:
        while buf.count(",") < n + 1:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            buf += chunk
    vals = [float(p) for p in buf.split(",")[:n] if p.strip()]
    return np.array(vals[:n])


def find(bui, band):
    hits = glob.glob(f"{RAW}/**/{bui}{band}_0.csv", recursive=True)
    return hits[0] if hits else None


def main():
    samples = []
    for bui in PICK:
        lp, hp = find(bui, "L"), find(bui, "H")
        if not lp or not hp:
            print(f"skip {bui} (missing)")
            continue
        xl, xh = read_head(lp), read_head(hp)
        if len(xl) < L or len(xh) < L:
            print(f"skip {bui} (short)")
            continue
        samples.append({
            "filename": os.path.basename(lp).replace("L_", "{L,H}_"),
            "bui": bui,
            "L": xl.astype(np.float32),
            "H": xh.astype(np.float32),
            "presence": 0 if FAMILY[bui] == 0 else 1,
            "family": FAMILY[bui],
            "mode": MODE[bui],
        })
        print(f"saved {bui}")
    os.makedirs("../models", exist_ok=True)
    joblib.dump(samples, OUT)
    print(f"wrote {OUT} with {len(samples)} raw segments")


if __name__ == "__main__":
    main()
