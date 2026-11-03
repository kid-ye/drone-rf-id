"""Streamlit demo for RF-based drone identification.

Picks a real held-out recording, runs the full pipeline live
(raw samples -> FFT -> power spectrum), classifies it with the trained models,
and shows the prediction against the true label decoded from the filename.

Run:  streamlit run demo.py
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

M = "../models"
NBINS, Q = 2048, 10

PRESENCE = {0: "No drone", 1: "Drone present"}
FAMILY = {0: "Background", 1: "Parrot Bebop", 2: "Parrot AR", 3: "DJI Phantom"}
MODE = {0: "Background", 1: "Bebop mode 1", 2: "Bebop mode 2", 3: "Bebop mode 3",
        4: "Bebop mode 4", 5: "AR mode 1", 6: "AR mode 2", 7: "AR mode 3",
        8: "AR mode 4", 9: "Phantom"}
NAMES = {"presence": PRESENCE, "family": FAMILY, "mode": MODE}
CHECK, CROSS = "✅", "❌"
ACCENT = "#0A84FF"

# ---- clean, Apple-inspired matplotlib style ----
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#C7C7CC", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E5E5EA", "grid.linewidth": 0.6,
    "xtick.color": "#8E8E93", "ytick.color": "#8E8E93",
    "axes.labelcolor": "#3A3A3C", "text.color": "#1C1C1E",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


@st.cache_resource
def load():
    models = {t: joblib.load(f"{M}/xgb_{t}.pkl") for t in NAMES}
    raw = joblib.load(f"{M}/raw_demo.pkl")
    return models, raw


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


def pipeline(L, H):
    """Reproduce preprocess.py: raw L/H -> 2048-D power spectrum."""
    xf, yf = seg_feat(L), seg_feat(H)
    yf = yf * (xf[-Q:].mean() / yf[:Q].mean())
    feat = np.concatenate([xf, yf]) ** 2
    return xf, yf, feat


def line(y, xlabel, ylabel, color=ACCENT, mark=None):
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.plot(y, lw=0.7, color=color)
    if mark is not None:
        ax.axvline(mark, color="#8E8E93", ls="--", lw=0.8)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.margins(x=0)
    fig.tight_layout()
    return fig


# ================= page =================
st.set_page_config(page_title="RF Drone ID", page_icon="\U0001F4E1", layout="centered")
st.markdown("""
<style>
html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont,
  "SF Pro Text", "Segoe UI", system-ui, sans-serif; }
h1 { letter-spacing: -0.02em; font-weight: 700; }
h2, h3 { letter-spacing: -0.01em; font-weight: 600; }
.block-container { padding-top: 2.5rem; max-width: 820px; }
div.stButton > button { border-radius: 980px; border: none; background: #0A84FF;
  color: white; font-weight: 600; padding: 0.55rem 1.4rem; transition: transform .1s ease; }
div.stButton > button:hover { background: #0071e3; color: white; }
div.stButton > button:active { transform: scale(0.97); }
.card { background: #FFFFFF; border: 1px solid #E5E5EA; border-radius: 16px;
  padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
.mono { font-family: "SF Mono", ui-monospace, Menlo, monospace; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

st.title("RF-Based Drone Identification")
st.caption("From a recorded radio-frequency segment, the models predict drone "
           "presence, family and operating mode. The full pipeline runs live.")

models, raw = load()

if st.button("Analyse a random RF segment", type="primary"):
    st.session_state.i = int(np.random.randint(len(raw)))

if "i" not in st.session_state:
    st.info("Press the button to pick a held-out recording and run it end to end.")
    st.stop()

s = raw[st.session_state.i]
L, H = np.asarray(s["L"], float), np.asarray(s["H"], float)
xf, yf, feat = pipeline(L, H)
x = feat.reshape(1, -1)

# ---- source file ----
st.markdown(f"<div class='card'><b>Source file:</b> "
            f"<span class='mono'>{s['filename']}</span><br>"
            f"<span style='color:#8E8E93'>The 5-digit code "
            f"<span class='mono'>{s['bui']}</span> is the ground-truth "
            f"Binary Unique Identifier.</span></div>", unsafe_allow_html=True)
st.write("")

# ---- predictions ----
st.subheader("What the models found")
cols = st.columns(3)
for col, task in zip(cols, ["presence", "family", "mode"]):
    pred = int(models[task].predict(x)[0])
    true = int(s[task])
    nm = NAMES[task]
    col.markdown(f"**{task.capitalize()}**")
    col.metric("Predicted", nm.get(pred, pred))
    col.markdown(f"Actual: **{nm.get(true, true)}** {CHECK if pred == true else CROSS}")

# ---- live pipeline stages ----
st.subheader("Signal pipeline, stage by stage")
st.markdown("**Stage 1 - Raw signal (time domain, low band)**")
st.pyplot(line(L[:2000], "sample index (first 2000)", "amplitude", color="#1C1C1E"))
st.caption("The raw recording looks like noise in time; the drone's signature is "
           "hidden in its frequency content.")

st.markdown("**Stage 2 - Frequency spectrum via FFT (low band)**")
st.pyplot(line(xf, "frequency bin (1024)", "magnitude"))
st.caption("A 2048-point Fast Fourier Transform exposes the frequency pattern; "
           "the upper 1024 bins are kept.")

st.markdown("**Stage 3 - Low + high bands, power spectrum (model input)**")
st.pyplot(line(feat, "frequency bin (2048)", "power", mark=1024))
st.caption("Low and high sub-bands are joined (dashed line) and squared to a "
           "power spectrum: the 2048 features the models classify.")

# ---- mode confidence ----
st.subheader("Model confidence (operating mode)")
proba = models["mode"].predict_proba(x)[0]
import pandas as pd
st.bar_chart(pd.DataFrame({"probability": proba},
             index=[MODE[i] for i in range(len(proba))]))

# ---- confusion matrices, any model ----
st.subheader("Confusion matrices")
c1, c2 = st.columns(2)
task_sel = c1.selectbox("Task", ["presence", "family", "mode"], index=1)
model_sel = c2.selectbox("Model", ["XGBoost", "RandomForest", "LogisticRegression", "SVM-RBF"])
tmap = {"presence": "presence", "family": "family", "mode": "mode"}
st.image(f"../results/cm_{tmap[task_sel]}_{model_sel}.png",
         caption=f"{model_sel} on the {task_sel} task")

# ---- results table ----
with st.expander("Overall results (macro-F1)"):
    st.table(pd.DataFrame({
        "Presence": [0.450, 0.995, 0.450, 0.997],
        "Family": [0.453, 0.874, 0.430, 0.906],
        "Mode": [0.401, 0.711, 0.262, 0.806],
    }, index=["Logistic Regression", "Random Forest", "Support Vector Machine", "XGBoost"]))
