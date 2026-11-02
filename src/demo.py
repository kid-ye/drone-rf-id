"""Streamlit demo for RF-based drone identification.

Pick a random held-out recording; the trained XGBoost models predict presence,
family and operating mode, shown against the true answer.

Run:  streamlit run demo.py
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

M = "../models"

PRESENCE = {0: "No drone", 1: "Drone present"}
FAMILY = {0: "Background", 1: "Parrot Bebop", 2: "Parrot AR", 3: "DJI Phantom"}
MODE = {
    0: "Background", 1: "Bebop mode 1", 2: "Bebop mode 2", 3: "Bebop mode 3",
    4: "Bebop mode 4", 5: "AR mode 1", 6: "AR mode 2", 7: "AR mode 3",
    8: "AR mode 4", 9: "Phantom",
}
NAMES = {"presence": PRESENCE, "family": FAMILY, "mode": MODE}
CHECK, CROSS = "✅", "❌"


@st.cache_resource
def load():
    models = {t: joblib.load(f"{M}/xgb_{t}.pkl")
              for t in ["presence", "family", "mode"]}
    samples = pd.read_parquet(f"{M}/demo_samples.parquet")
    return models, samples


st.set_page_config(page_title="RF Drone ID Demo", page_icon="\U0001F4E1")
st.title("\U0001F4E1 RF-Based Drone Identification")
st.caption("Classical ML on the DroneRF dataset. XGBoost predicts presence, "
           "family and operating mode from a recorded RF segment.")

models, samples = load()
feat = [c for c in samples.columns if c.startswith("f")]

if st.button("Pick a random RF segment", type="primary"):
    st.session_state.row = samples.sample(1).iloc[0]

if "row" not in st.session_state:
    st.info("Click the button to classify a held-out recording the models "
            "never trained on.")
    st.stop()

row = st.session_state.row
x = row[feat].to_numpy(dtype=float).reshape(1, -1)

st.subheader("Recorded RF signal (power spectrum)")
vals = row[feat].to_numpy(dtype=float)
db = 10 * np.log10(vals + 1e-12)
fig, ax = plt.subplots(figsize=(9, 2.6))
ax.plot(db, lw=0.6, color="#284878")
ax.axvline(1024, color="gray", ls="--", lw=0.9)
ax.text(300, db.max(), "low band", color="gray", fontsize=8)
ax.text(1300, db.max(), "high band", color="gray", fontsize=8)
ax.set_xlabel("frequency bin")
ax.set_ylabel("power (dB)")
ax.margins(x=0)
st.pyplot(fig)
st.caption("This frequency fingerprint is the 2048-number feature vector the "
           "models classify. Different drones leave different patterns here.")

st.subheader("Predictions vs. ground truth")
cols = st.columns(3)
for col, task in zip(cols, ["presence", "family", "mode"]):
    pred = int(models[task].predict(x)[0])
    true = int(row[task])
    names = NAMES[task]
    ok = pred == true
    col.markdown(f"**{task.capitalize()}**")
    col.metric(label="Predicted", value=names.get(pred, pred))
    col.markdown(f"True: **{names.get(true, true)}**  "
                 f"{CHECK if ok else CROSS}")

st.divider()
st.subheader("Model confidence (operating mode)")
proba = models["mode"].predict_proba(x)[0]
conf = pd.DataFrame({"mode": [MODE[i] for i in range(len(proba))],
                     "probability": proba}).set_index("mode")
st.bar_chart(conf)

with st.expander("Overall results and confusion matrices"):
    st.markdown("**Macro-F1 by model and task**")
    st.table(pd.DataFrame({
        "Task": ["Presence", "Family", "Mode (5-fold)"],
        "LR": [1.000, 0.999, 0.375],
        "RF": [0.978, 0.898, 0.378],
        "SVM": [1.000, 0.977, 0.160],
        "XGBoost": [1.000, 1.000, 0.492],
    }).set_index("Task"))
    c1, c2 = st.columns(2)
    c1.image("../results/cm_family_XGBoost.png", caption="Family (XGBoost)")
    c2.image("../results/cm_mode_XGBoost.png", caption="Mode (XGBoost)")
