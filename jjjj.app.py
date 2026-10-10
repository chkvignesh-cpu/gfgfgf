import logging

import numpy as np
import pandas as pd
import streamlit as st

import model as mdl
from alerts import AlertEngine
from news2 import LEVEL_COLORS, score_vitals
from simulator import NAMES, VITALS, Patient

st.set_page_config(
    page_title="PulseGuard AI",
    page_icon="🫀",
    layout="wide",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_ROWS = 6000

LABELS = {
    "hr": "Heart rate (bpm)",
    "spo2": "SpO2 (%)",
    "sbp": "Systolic BP (mmHg)",
    "temp": "Temperature (°C)",
    "rr": "Respiratory rate (/min)",
    "score": "NEWS2 score",
}


@st.cache_resource(show_spinner="Loading risk prediction model...")
def get_model():
    return mdl.load_or_train()


def advance():
    ss = st.session_state
    ss.tick += 1

    for patient in ss.patients:
        try:
            vitals = patient.step()
            result = score_vitals(vitals)

            ss.rows.append({
                "patient": patient.pid,
                "t": ss.tick,
                "name": patient.name,
                "bed": patient.bed,
                **vitals,
                "score": result["total"],
                "level": result["level"],
                "red": result["red"],
            })
        except Exception:
            logger.exception(
                "Could not generate patient reading."
            )

    if len(ss.rows) > MAX_ROWS:
        ss.rows = ss.rows[-MAX_ROWS:]


def init_state(number_of_patients, seed=7):
    ss = st.session_state
    rng = np.random.default_rng(seed)

    number_of_patients = int(number_of_patients)

    if not NAMES:
        raise ValueError("simulator.NAMES cannot be empty.")

    at_risk = set(
        rng.choice(
            number_of_patients,
            size=min(3, number_of_patients),
            replace=False,
        ).tolist()
    )

    ss.patients = []

    for i in range(number_of_patients):
        onset = (
            int(rng.integers(18, 40))
            if i in at_risk
            else None
        )

        ss.patients.append(
            Patient(
                i,
                NAMES[i % len(NAMES)],
                f"B{i + 1:02d}",
                rng,
                onset=onset,
            )
        )

    ss.rows = []
    ss.tick = 0
    ss.engine = AlertEngine()

    for _ in range(12):
        advance()


def predict_risk(pipe, history):
    if history.empty:
        return {}

    try:
        predictions = mdl.predict_latest(pipe, history)

        if predictions is None:
            return {}

        result = {}

        for patient_id, probability in predictions.items():
            try:
                probability = float(probability)

                if np.isfinite(probability):
                    result[patient_id] = float(
                        np.clip(probability, 0.0, 1.0)
                    )
            except (TypeError, ValueError):
                continue

        return result

    except Exception:
        logger.exception("Risk prediction failed.")
        return {}


# ---------------- SIDEBAR ----------------

st.sidebar.title("🫀 PulseGuard AI")

number_of_patients = st.sidebar.slider(
    "Patients on ward", 6, 20, 12
)

running = st.sidebar.toggle(
    "Live simulation", value=True
)

interval = st.sidebar.slider(
    "Seconds per reading", 1, 10, 2
)

st.sidebar.subheader("Alerts")

alerts_enabled = st.sidebar.toggle(
    "Send alerts", value=True
)

channels = st.sidebar.multiselect(
    "Channels",
    ["SMS", "Telegram"],
    default=["SMS"],
)

cooldown = st.sidebar.slider(
    "Cooldown per patient (seconds)",
    30, 600, 120
)

if (
    "patients" not in st.session_state
    or st.session_state.get("n") != number_of_patients
):
    st.session_state.n = number_of_patients
    init_state(number_of_patients, seed=7)

st.session_state.engine.cooldown_s = cooldown

st.sidebar.subheader("Demo controls")

patient_options = [
    f"{p.bed} - {p.name}"
    for p in st.session_state.patients
]

selected_patient = st.sidebar.selectbox(
    "Patient", patient_options, key="demo_patient"
)

patient_index = patient_options.index(selected_patient)

col1, col2 = st.sidebar.columns(2)

if col1.button("Deteriorate", use_container_width=True):
    st.session_state.patients[
        patient_index
    ].deteriorating = True

if col2.button("Recover", use_container_width=True):
    st.session_state.patients[
        patient_index
    ].deteriorating = False

if st.sidebar.button("Reset ward", use_container_width=True):
    seed = int(np.random.default_rng().integers(1, 10000))
    init_state(number_of_patients, seed=seed)

# ---------------- MODEL ----------------

try:
    pipe, metrics = get_model()
except Exception:
    st.error(
        "Could not load the model. Check model.py "
        "and install the required dependencies."
    )
    logger.exception("Model initialization failed.")
    st.stop()

with st.sidebar.expander("Model performance"):
    if isinstance(metrics, dict):
        st.write("ROC-AUC:", metrics.get("roc_auc", "N/A"))
        st.write(
            "Recall:",
            metrics.get("recall_escalation", "N/A"),
        )
        st.write(
            "Precision:",
            metrics.get("precision_escalation", "N/A"),
        )

        st.caption(
            "Prediction target: NEWS2 >= 5 within "
            f"{metrics.get('horizon_readings', 'N/A')} readings."
        )

        weights = metrics.get("feature_weights", {})

        if isinstance(weights, dict) and weights:
            st.bar_chart(
                pd.Series(weights, name="Feature weight")
            )


# ---------------- LIVE DASHBOARD ----------------

def live_panel():
    ss = st.session_state

    if running:
        advance()

    df = pd.DataFrame(ss.rows)

    if df.empty:
        st.warning("Waiting for patient readings.")
        return

    required = [
        "patient", "t", "name", "bed",
        "hr", "spo2", "sbp", "temp", "rr",
        "score", "level", "red",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        return

    recent = df[df["t"] >= ss.tick - 6].copy()
    risk = predict_risk(pipe, recent)

    now = df[df["t"] == ss.tick].copy()

    if now.empty:
        st.warning("No readings available for this tick.")
        return

    now["risk"] = now["patient"].map(risk)

    now = now.sort_values(
        ["score", "risk"],
        ascending=[False, False],
        na_position="last",
    )

    # ---------------- ALERTS ----------------

    if alerts_enabled and channels:
        for row in now.itertuples(index=False):
            vitals_text = (
                f"HR {row.hr:.0f}, "
                f"SpO2 {row.spo2:.0f}%, "
                f"BP {row.sbp:.0f}, "
                f"Temp {row.temp:.1f}, "
                f"RR {row.rr:.0f}"
            )

            try:
                if row.level == "HIGH":
                    ss.engine.notify(
                        f"high-{row.patient}",
                        row.name,
                        "HIGH RISK",
                        (
                            f"PulseGuard ALERT: {row.name} "
                            f"({row.bed}) NEWS2={row.score} HIGH. "
                            f"{vitals_text}"
                        ),
                        channels,
                    )

                elif (
                    pd.notna(row.risk)
                    and row.risk >= 0.75
                    and row.level != "HIGH"
                ):
                    ss.engine.notify(
                        f"early-{row.patient}",
                        row.name,
                        "EARLY WARNING",
                        (
                            f"PulseGuard EARLY WARNING: "
                            f"{row.name} ({row.bed}), "
                            f"risk {row.risk:.0%}, "
                            f"NEWS2={row.score}. {vitals_text}"
                        ),
                        channels,
                    )

            except Exception:
                logger.exception(
                    "Alert failed for patient %s",
                    row.patient,
                )

    # ---------------- SUMMARY ----------------

    st.title("PulseGuard AI - Ward Early-Warning Desk")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("HIGH", int((now["level"] == "HIGH").sum()))
    c2.metric("MEDIUM", int((now["level"] == "MEDIUM").sum()))
    c3.metric("LOW", int((now["level"] == "LOW").sum()))
    c4.metric("Reading #", int(ss.tick))

    # ---------------- PATIENT TABLE ----------------

    st.subheader("Active tracking desk")

    table = now[
        [
            "bed", "name", "hr", "spo2", "sbp",
            "temp", "rr", "score", "level", "risk",
        ]
    ].rename(columns={
        "bed": "Bed",
        "name": "Patient",
        "hr": "HR",
        "spo2": "SpO2",
        "sbp": "SBP",
        "temp": "Temp",
        "rr": "RR",
        "score": "NEWS2",
        "level": "Level",
        "risk": "Escalation risk",
    })

    def style_row(row):
        color = LEVEL_COLORS.get(
            str(row["Level"]), "#FFFFFF"
        )
        return [
            f"background-color: {color}; color: #111111"
            for _ in row
        ]

    styled = (
        table.style
        .apply(style_row, axis=1)
        .format(
            {
                "HR": "{:.0f}",
                "SpO2": "{:.0f}",
                "SBP": "{:.0f}",
                "Temp": "{:.1f}",
                "RR": "{:.0f}",
                "NEWS2": "{:.0f}",
                "Escalation risk": (
                    lambda value: (
                        f"{value:.0%}"
                        if pd.notna(value)
                        else "Unavailable"
                    )
                ),
            },
            na_rep="—",
        )
    )

    st.dataframe(
        styled,
        hide_index=True,
        height=min(60 + 36 * len(table), 520),
        use_container_width=True,
    )

    st.caption(
        "Demonstration bands: LOW 0–4, MEDIUM 5–6 or "
        "a single vital scoring 3, HIGH 7+. "
        "Predictions are estimates, not diagnoses."
    )

    # ---------------- PATIENT DETAILS ----------------

    st.subheader("Patient detail")

    detail_options = [
        f"{row.bed} - {row.name}"
        for row in now.itertuples(index=False)
    ]

    selected = st.selectbox(
        "Select patient",
        detail_options,
        key="detail_pick",
    )

    selected_bed = selected.split(" - ", 1)[0]
    selected_rows = now[now["bed"] == selected_bed]

    if not selected_rows.empty:
        pid = selected_rows.iloc[0]["patient"]

        history = (
            df[df["patient"] == pid]
            .sort_values("t")
            .tail(60)
            .set_index("t")
        )

        chart_columns = list(VITALS) + ["score"]
        chart_columns = list(dict.fromkeys(
            col for col in chart_columns
            if col in history.columns
        ))

        cols = st.columns(3)

        for i, vital in enumerate(chart_columns):
            with cols[i % 3]:
                st.caption(LABELS.get(vital, vital))
                st.line_chart(history[[vital]], height=180)

    # ---------------- ALERT LOG ----------------

    st.subheader("Alert log")

    if ss.engine.log:
        st.dataframe(
            pd.DataFrame(ss.engine.log[::-1]),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info(
            "No alerts yet. Use Deteriorate to test the simulation."
        )


st.fragment(
    run_every=interval if running else None
)(live_panel)()

st.caption(
    "Synthetic data only. This prototype is not a medical device "
    "and must not be used to make real patient-care decisions."
)
