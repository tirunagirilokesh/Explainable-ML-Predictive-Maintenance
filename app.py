from pathlib import Path
from datetime import datetime
import os
import io
import wave
import base64
import smtplib
from email.message import EmailMessage

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, matthews_corrcoef,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
HISTORY_PATH = REPORTS_DIR / "dashboard_prediction_history.csv"

MODEL_PATHS = {
    "Logistic Regression": MODELS_DIR / "logistic_regression.joblib",
    "Random Forest": MODELS_DIR / "random_forest.joblib",
    "Decision Tree": MODELS_DIR / "decision_tree.joblib",
    "Gradient Boosting": MODELS_DIR / "gradient_boosting.joblib",
}

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_L",
    "Type_M",
]

st.set_page_config(
    page_title="Predictive Maintenance | Explainable AI",
    page_icon="⚙️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main {background-color: #f7f9fc;}
      div[data-testid="stMetric"] {
        background: white; padding: 14px; border-radius: 12px;
        border: 1px solid #e5e7eb;
      }
      .small-note {font-size: 0.88rem; color: #596579;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
@st.cache_resource
def load_model(path_string: str):
    path = Path(path_string)
    if not path.exists():
        return None
    return joblib.load(path)


def find_project_file(filename: str):
    search_folders = [
        REPORTS_DIR, PROJECT_DIR, PROJECT_DIR / "data",
        PROJECT_DIR / "data" / "processed", PROJECT_DIR / "datasets",
        PROJECT_DIR / "processed", PROJECT_DIR / "outputs",
    ]
    for folder in search_folders:
        candidate = folder / filename
        if candidate.is_file():
            return candidate
    for root in [
        PROJECT_DIR / "data", REPORTS_DIR, PROJECT_DIR / "datasets",
        PROJECT_DIR / "processed", PROJECT_DIR / "outputs",
    ]:
        if root.is_dir():
            try:
                matches = sorted(root.rglob(filename), key=lambda p: len(p.parts))
                if matches:
                    return matches[0]
            except (OSError, PermissionError):
                pass
    return None


def read_csv_if_present(filename: str):
    path = find_project_file(filename)
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read `{filename}`: {exc}")
        return None


def make_sensor_reading():
    """Generate a demonstration reading; this is not a physical sensor feed."""
    air_temp = float(np.random.normal(300.0, 1.5))
    process_temp = float(air_temp + np.random.normal(10.0, 1.0))
    rpm = int(np.clip(np.random.normal(1500, 180), 1000, 2200))
    torque = float(np.clip(np.random.normal(40, 9), 10, 75))
    tool_wear = int(np.random.randint(0, 251))

    # A small share of readings use more challenging combinations for demos.
    if np.random.random() < 0.18:
        rpm = int(np.random.randint(1100, 1450))
        torque = float(np.random.uniform(48, 70))
        tool_wear = int(np.random.randint(180, 251))
        process_temp = float(air_temp + np.random.uniform(8.0, 13.0))

    machine_type = str(np.random.choice(["L", "M", "H"], p=[0.5, 0.3, 0.2]))
    row = {
        "Air temperature [K]": round(air_temp, 2),
        "Process temperature [K]": round(process_temp, 2),
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": round(torque, 2),
        "Tool wear [min]": tool_wear,
        "Type_L": int(machine_type == "L"),
        "Type_M": int(machine_type == "M"),
    }
    return row, machine_type


def predict_one(model, features_df: pd.DataFrame, threshold: float):
    """Return prediction using the selected threshold when probabilities exist."""
    prediction = int(model.predict(features_df)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_df)[0]
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            probability = float(probabilities[classes.index(1)])
        elif len(probabilities) > 1:
            probability = float(probabilities[-1])
        if probability is not None:
            prediction = int(probability >= threshold)
    return prediction, probability


def append_history(record: dict):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([record])
    if HISTORY_PATH.exists():
        new_row.to_csv(HISTORY_PATH, mode="a", header=False, index=False)
    else:
        new_row.to_csv(HISTORY_PATH, index=False)


def make_beep_data_uri():
    sample_rate, duration, frequency, amplitude = 22050, 0.45, 880, 9000
    n_samples = int(sample_rate * duration)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        samples = bytearray()
        for i in range(n_samples):
            fade = min(1.0, i / 500, (n_samples - i) / 500)
            value = int(
                amplitude * max(0, fade)
                * np.sin(2 * np.pi * frequency * i / sample_rate)
            )
            samples.extend(int(value).to_bytes(2, "little", signed=True))
        wav.writeframes(samples)
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def send_email_alert(sensor_row, probability, sender=None, password=None,
                     recipient=None, smtp_host=None, smtp_port=None):
    sender = sender or os.getenv("PM_EMAIL_SENDER")
    password = password or os.getenv("PM_EMAIL_APP_PASSWORD")
    recipient = recipient or os.getenv("PM_EMAIL_RECIPIENT")
    smtp_host = smtp_host or os.getenv("PM_SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(smtp_port or os.getenv("PM_SMTP_PORT", "465"))

    if not (sender and password and recipient):
        return False, "Email settings are incomplete. Enter sender, app password, and recipient."
    msg = EmailMessage()
    msg["Subject"] = "Predictive Maintenance Alert: Potential Failure"
    msg["From"] = sender
    msg["To"] = recipient
    score_text = f"{probability:.1%}" if probability is not None else "not available"
    msg.set_content(
        "A potential machine failure was predicted by the dashboard.\n\n"
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Model-estimated failure score: {score_text}\n\n"
        "Machine readings:\n"
        + "\n".join(f"{key}: {value}" for key, value in sensor_row.items())
        + "\n\nThis demonstration uses simulated readings, not a physical sensor feed."
    )
    try:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=15) as server:
            server.login(sender, password)
            server.send_message(msg)
        return True, "Email alert sent successfully."
    except Exception as exc:
        return False, f"Email could not be sent: {exc}"


def display_report_csv(filename: str, title: str):
    data = read_csv_if_present(filename)
    st.subheader(title)
    if data is None:
        st.info(f"`reports/{filename}` was not found.")
        return
    st.dataframe(data, use_container_width=True)
    st.download_button(
        f"Download {filename}", data.to_csv(index=False).encode("utf-8"),
        filename, "text/csv", key=f"download_{filename}",
    )


def get_failure_scores(model, X):
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        return None
    return probabilities[:, classes.index(1)]


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("⚙️ Predictive Maintenance")
st.sidebar.caption("Explainable AI • AI4I 2020 dataset")

page = st.sidebar.radio(
    "Navigation",
    ["Live Dashboard", "Model Evaluation", "Explainable AI",
     "Reports & History", "About Project"],
)
st.sidebar.divider()

auto_refresh = st.sidebar.toggle("Auto-refresh simulated sensors", value=True)
refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 3, 30, 5)
threshold = st.sidebar.slider(
    "Failure alert threshold", 0.10, 0.90, 0.40, 0.05,
    help="Demonstration setting; this is not a validated industrial threshold.",
)

st.sidebar.subheader("Primary prediction model")
available_model_names = [
    name for name, path in MODEL_PATHS.items() if path.exists()
]
if available_model_names:
    # Gradient Boosting is the default based on the previously reported failure F1.
    default_model = (
        "Gradient Boosting" if "Gradient Boosting" in available_model_names
        else available_model_names[0]
    )
    selected_model_name = st.sidebar.selectbox(
        "Model used for the main prediction",
        available_model_names,
        index=available_model_names.index(default_model),
        help="The default is Gradient Boosting based on your previously reported test F1-score. You can change it.",
    )
else:
    selected_model_name = None

st.sidebar.divider()
st.sidebar.subheader("📧 Email Alert Settings")
email_alerts_enabled = st.sidebar.toggle("Enable email alerts", value=False)
email_sender = st.sidebar.text_input("Sender Gmail address", value=os.getenv("PM_EMAIL_SENDER", ""))
email_password = st.sidebar.text_input(
    "Gmail app password", value=os.getenv("PM_EMAIL_APP_PASSWORD", ""), type="password",
    help="Use a Google App Password, not your normal Gmail password.",
)
email_recipient = st.sidebar.text_input("Recipient email address", value=os.getenv("PM_EMAIL_RECIPIENT", ""))
smtp_host = st.sidebar.text_input("SMTP host", value=os.getenv("PM_SMTP_HOST", "smtp.gmail.com"))
smtp_port = st.sidebar.number_input(
    "SMTP SSL port", min_value=1, max_value=65535,
    value=int(os.getenv("PM_SMTP_PORT", "465")), step=1,
)

if email_alerts_enabled:
    st.sidebar.info("Alerts are sent on a transition from Normal to Potential Failure.")
if st.sidebar.button("Send test email"):
    ok, message = send_email_alert(
        {"Test": "Dashboard email configuration test"}, None,
        sender=email_sender, password=email_password, recipient=email_recipient,
        smtp_host=smtp_host, smtp_port=smtp_port,
    )
    (st.sidebar.success if ok else st.sidebar.error)(message)

st.sidebar.caption("Demo readings are simulated. No physical sensors are connected.")

if auto_refresh and st_autorefresh is not None and page == "Live Dashboard":
    refresh_count = st_autorefresh(interval=refresh_seconds * 1000, key="sensor_refresh")
else:
    refresh_count = None

# ---------------------------------------------------------
# Load models
# ---------------------------------------------------------
models = {name: load_model(str(path)) for name, path in MODEL_PATHS.items()}
models = {name: model for name, model in models.items() if model is not None}

st.title("🏭 Explainable AI for Predictive Maintenance")
st.caption("AI4I 2020 dataset • Simulated sensor stream • Model-based failure prediction")

if not models:
    st.error("No model files were found. Expected saved `.joblib` files in the `models` folder.")
    st.stop()

# ---------------------------------------------------------
# Live Dashboard
# ---------------------------------------------------------
if page == "Live Dashboard":
    st.warning(
        "DEMO MODE: Values are simulated by software, not collected from physical sensors. "
        "Do not use these predictions as the sole basis for real maintenance or safety decisions."
    )

    if selected_model_name is None:
        st.error("No primary model is available.")
        st.stop()
    primary_model = models[selected_model_name]

    # Keep the same reading during ordinary widget reruns. Generate a new one only
    # on the first run, an auto-refresh tick, or a manual button click.
    if "reading_version" not in st.session_state:
        st.session_state.reading_version = 0
        st.session_state.sensor_row, st.session_state.machine_type = make_sensor_reading()
        st.session_state.logged_reading_version = -1
        st.session_state.last_prediction = None
        st.session_state.alert_log = []

    if st.button("Generate new simulated reading"):
        st.session_state.reading_version += 1
        st.session_state.sensor_row, st.session_state.machine_type = make_sensor_reading()

    if refresh_count is not None:
        if "last_refresh_count" not in st.session_state:
            st.session_state.last_refresh_count = refresh_count
        elif refresh_count != st.session_state.last_refresh_count:
            st.session_state.last_refresh_count = refresh_count
            st.session_state.reading_version += 1
            st.session_state.sensor_row, st.session_state.machine_type = make_sensor_reading()

    sensor_row = st.session_state.sensor_row
    machine_type = st.session_state.machine_type
    input_df = pd.DataFrame([sensor_row], columns=FEATURES)
    prediction, probability = predict_one(primary_model, input_df, threshold)
    now = datetime.now()

    st.session_state.latest_sensor_row = sensor_row
    st.session_state.latest_prediction = prediction
    st.session_state.latest_probability = probability

    # Save each generated reading once, rather than on every Streamlit rerun.
    if st.session_state.logged_reading_version != st.session_state.reading_version:
        history_record = {
            "Timestamp": now.isoformat(timespec="seconds"),
            **sensor_row,
            "Machine Type": machine_type,
            "Selected Model": selected_model_name,
            "Predicted Failure": prediction,
            "Failure Score": probability,
            "Threshold": threshold,
            "Data Source": "SIMULATED",
        }
        append_history(history_record)
        st.session_state.logged_reading_version = st.session_state.reading_version

    previous_prediction = st.session_state.get("last_prediction")
    if prediction == 1 and previous_prediction == 0 and email_alerts_enabled:
        success, message = send_email_alert(
            sensor_row, probability, sender=email_sender, password=email_password,
            recipient=email_recipient, smtp_host=smtp_host, smtp_port=smtp_port,
        )
        st.session_state.alert_log.insert(0, {
            "Timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "Event": "Potential failure detected",
            "Email result": message,
        })
        if success:
            st.toast("Email alert sent.", icon="📧")
        else:
            st.toast("Potential failure detected, but email could not be sent.", icon="⚠️")
    elif prediction == 1 and previous_prediction == 0:
        st.session_state.alert_log.insert(0, {
            "Timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "Event": "Potential failure detected",
            "Email result": "Email alerts disabled",
        })
    st.session_state.last_prediction = prediction

    probability_text = f"{probability:.1%}" if probability is not None else "Not available"
    if prediction == 1:
        st.error("🚨 POTENTIAL FAILURE DETECTED")
        st.markdown(f"**Model-estimated failure score:** {probability_text}")
        st.audio(base64.b64decode(make_beep_data_uri().split(",", 1)[1]), format="audio/wav")
        st.caption("Use the audio player's play button if sound does not play automatically.")
    else:
        st.success("✅ Machine status: NORMAL")
        st.markdown(f"**Model-estimated failure score:** {probability_text}")

    cols = st.columns(5)
    sensor_cards = [
        ("Air temperature", f"{sensor_row['Air temperature [K]']:.2f} K"),
        ("Process temperature", f"{sensor_row['Process temperature [K]']:.2f} K"),
        ("Rotational speed", f"{sensor_row['Rotational speed [rpm]']} rpm"),
        ("Torque", f"{sensor_row['Torque [Nm]']:.2f} Nm"),
        ("Tool wear", f"{sensor_row['Tool wear [min]']} min"),
    ]
    for col, (label, value) in zip(cols, sensor_cards):
        col.metric(label, value)

    c1, c2, c3 = st.columns(3)
    c1.metric("Machine type", machine_type)
    c2.metric("Primary model", selected_model_name)
    c3.metric("Failure threshold", f"{threshold:.2f}")

    st.subheader("Model input vector")
    st.dataframe(input_df, use_container_width=True, hide_index=True)
    st.caption("The model receives the same seven engineered features used during training.")

    st.subheader("Four-model prediction comparison")
    comparison_rows = []
    for model_name, model in models.items():
        try:
            pred, score = predict_one(model, input_df, threshold)
            comparison_rows.append({
                "Model": model_name,
                "Predicted status": "Potential Failure" if pred else "Normal",
                "Model-estimated failure score": f"{score:.2%}" if score is not None else "Not available",
            })
        except Exception as exc:
            comparison_rows.append({"Model": model_name, "Error": str(exc)})
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
    votes = sum(row.get("Predicted status") == "Potential Failure" for row in comparison_rows)
    st.metric("Models predicting potential failure", f"{votes}/{len(comparison_rows)}")
    st.caption(
        "Models may disagree. Scores are model estimates and may not be calibrated real-world probabilities."
    )

    st.subheader("Recent dashboard history")
    if HISTORY_PATH.exists():
        try:
            st.dataframe(pd.read_csv(HISTORY_PATH).tail(15).iloc[::-1], use_container_width=True, hide_index=True)
        except Exception as exc:
            st.info(f"History could not be displayed: {exc}")
    if st.session_state.alert_log:
        st.subheader("Recent alert events")
        st.dataframe(pd.DataFrame(st.session_state.alert_log), use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Model Evaluation
# ---------------------------------------------------------
elif page == "Model Evaluation":
    st.header("📊 Model Evaluation")
    st.write(
        "Metrics below are calculated from the held-out test set. Simulated dashboard readings "
        "are not used to calculate these metrics."
    )
    x_path = find_project_file("X_test.csv")
    y_path = find_project_file("y_test.csv")

    if not x_path or not y_path:
        st.warning("X_test.csv and/or y_test.csv were not found. Test metrics cannot be recalculated.")
    else:
        try:
            X_test = pd.read_csv(x_path)
            y_df = pd.read_csv(y_path)
            X_test = X_test.loc[:, ~X_test.columns.astype(str).str.startswith("Unnamed")]
            y_df = y_df.loc[:, ~y_df.columns.astype(str).str.startswith("Unnamed")]
            if X_test.empty or y_df.empty:
                raise ValueError("X_test.csv or y_test.csv is empty.")
            target_col = next(
                (c for c in y_df.columns if str(c).strip().lower()
                 in {"machine failure", "machine_failure", "target", "label", "y"}),
                y_df.columns[0],
            )
            y_true = pd.to_numeric(y_df[target_col], errors="raise").astype(int).to_numpy()
            missing = [feature for feature in FEATURES if feature not in X_test.columns]
            if missing:
                raise ValueError("X_test.csv is missing features: " + ", ".join(missing))
            X_test = X_test[FEATURES].copy()
            if len(X_test) != len(y_true):
                raise ValueError(f"X_test has {len(X_test)} rows but y_test has {len(y_true)}.")
            st.success(f"Loaded {len(X_test):,} held-out test rows.")

            metric_rows, confusion_data, report_data = [], {}, {}
            threshold_rows = []
            for name, model in models.items():
                y_pred_default = np.asarray(model.predict(X_test)).astype(int)
                y_scores = get_failure_scores(model, X_test)
                y_pred = (y_scores >= threshold).astype(int) if y_scores is not None else y_pred_default
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                metric_rows.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
                    "Precision (Failure)": precision_score(y_true, y_pred, zero_division=0),
                    "Recall (Failure)": recall_score(y_true, y_pred, zero_division=0),
                    "F1-score (Failure)": f1_score(y_true, y_pred, zero_division=0),
                    "Specificity": tn / (tn + fp) if tn + fp else 0.0,
                    "MCC": matthews_corrcoef(y_true, y_pred),
                    "ROC-AUC": roc_auc_score(y_true, y_scores) if y_scores is not None and len(np.unique(y_true)) == 2 else np.nan,
                    "PR-AUC (Average Precision)": average_precision_score(y_true, y_scores) if y_scores is not None and len(np.unique(y_true)) == 2 else np.nan,
                    "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                })
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                confusion_data[name] = pd.DataFrame(
                    cm, index=["Actual Normal (0)", "Actual Failure (1)"],
                    columns=["Predicted Normal (0)", "Predicted Failure (1)"],
                )
                report_data[name] = pd.DataFrame(classification_report(
                    y_true, y_pred, labels=[0, 1],
                    target_names=["Normal (0)", "Failure (1)"],
                    output_dict=True, zero_division=0,
                )).T
                if y_scores is not None:
                    tn_t, fp_t, fn_t, tp_t = confusion_matrix(
                        y_true, (y_scores >= threshold).astype(int), labels=[0, 1]
                    ).ravel()
                    threshold_rows.append({
                        "Model": name, "Threshold": threshold,
                        "Precision (Failure)": precision_score(y_true, (y_scores >= threshold).astype(int), zero_division=0),
                        "Recall (Failure)": recall_score(y_true, (y_scores >= threshold).astype(int), zero_division=0),
                        "F1-score (Failure)": f1_score(y_true, (y_scores >= threshold).astype(int), zero_division=0),
                        "False Positives": int(fp_t), "False Negatives": int(fn_t),
                    })

            metrics_df = pd.DataFrame(metric_rows).sort_values("F1-score (Failure)", ascending=False)
            st.subheader(f"Test-set metrics at threshold {threshold:.2f}")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download full evaluation metrics", metrics_df.to_csv(index=False).encode("utf-8"),
                "complete_model_evaluation.csv", "text/csv",
            )
            st.caption(
                "Because failures are uncommon in this dataset, accuracy alone can be misleading. "
                "Review failure recall, F1, PR-AUC, and false negatives."
            )
            st.subheader("Confusion matrices")
            for name, cm_df in confusion_data.items():
                with st.expander(f"{name} — confusion matrix"):
                    st.dataframe(cm_df, use_container_width=True)
                    st.write("Rows are actual classes; columns are predicted classes.")
            st.subheader("Classification reports")
            for name, report_df in report_data.items():
                with st.expander(f"{name} — classification report"):
                    st.dataframe(report_df, use_container_width=True)
            if threshold_rows:
                st.subheader("Threshold comparison at current setting")
                threshold_df = pd.DataFrame(threshold_rows)
                st.dataframe(threshold_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download threshold comparison", threshold_df.to_csv(index=False).encode("utf-8"),
                    "threshold_analysis_current_setting.csv", "text/csv",
                )
        except Exception as exc:
            st.error(f"Could not evaluate saved models: {exc}")

    st.divider()
    st.subheader("Saved reports")
    if REPORTS_DIR.exists():
        import re
        report_files = sorted(
            [p for p in REPORTS_DIR.rglob("*") if p.is_file()
             and p.name != HISTORY_PATH.name
             and p.suffix.lower() in {".csv", ".txt", ".json", ".xlsx"}],
            key=lambda p: p.name.lower(),
        )
        grouped = {}
        for path in report_files:
            match = re.search(r"day[\s_-]*0*(\d+)", str(path).lower())
            group = f"Day {int(match.group(1))}" if match else "Other reports"
            grouped.setdefault(group, []).append(path)
        def group_sort(label):
            match = re.search(r"Day (\d+)", label)
            return (0, int(match.group(1))) if match else (1, label.lower())
        for group in sorted(grouped, key=group_sort):
            with st.expander(f"{group} ({len(grouped[group])} files)"):
                for path in grouped[group]:
                    st.markdown(f"**{path.name}**")
                    try:
                        if path.suffix.lower() == ".csv":
                            st.dataframe(pd.read_csv(path), use_container_width=True, hide_index=True)
                        elif path.suffix.lower() == ".txt":
                            st.text(path.read_text(encoding="utf-8", errors="replace")[:10000])
                        elif path.suffix.lower() == ".json":
                            st.code(path.read_text(encoding="utf-8", errors="replace")[:10000], language="json")
                        else:
                            st.caption("Excel report available to download.")
                        st.download_button(
                            f"Download {path.name}", path.read_bytes(), path.name,
                            key=f"eval_report_{path.name}",
                        )
                    except Exception as exc:
                        st.warning(f"Could not display {path.name}: {exc}")

# ---------------------------------------------------------
# Explainable AI
# ---------------------------------------------------------
elif page == "Explainable AI":
    st.header("🔍 Explainable AI")
    st.write(
        "This page displays saved explainability artifacts from your project. "
        "The images are not recalculated for the current live prediction."
    )
    feature_imp = read_csv_if_present("day5_feature_importance.csv")
    if feature_imp is not None:
        st.subheader("Feature importance data")
        st.dataframe(feature_imp, use_container_width=True)
        if len(feature_imp.columns) >= 2:
            cols_lower = {c: c.lower() for c in feature_imp.columns}
            feature_col = next(
                (c for c in feature_imp.columns if any(k in cols_lower[c] for k in ["feature", "variable", "name"])),
                feature_imp.columns[0],
            )
            importance_col = next(
                (c for c in feature_imp.columns if any(k in cols_lower[c] for k in ["importance", "shap", "score"])),
                feature_imp.columns[1],
            )
            chart_data = feature_imp[[feature_col, importance_col]].copy()
            chart_data[importance_col] = pd.to_numeric(chart_data[importance_col], errors="coerce")
            chart_data = chart_data.dropna().set_index(feature_col)
            if not chart_data.empty:
                st.bar_chart(chart_data.sort_values(importance_col, ascending=True))

    image_items = [
        ("day5_feature_importance.png", "Random Forest Feature Importance"),
        ("day5_shap_importance.png", "SHAP Feature Importance"),
        ("day5_shap_summary.png", "SHAP Summary Plot"),
    ]
    for filename, title in image_items:
        path = REPORTS_DIR / filename
        if path.exists():
            st.subheader(title)
            st.image(str(path), use_container_width=True)
        else:
            st.info(f"`reports/{filename}` was not found.")

    st.info(
        "Global feature-importance and SHAP plots describe model behaviour across the data used to create them. "
        "They do not prove causation and do not guarantee that a real machine will fail."
    )

# ---------------------------------------------------------
# Reports & History
# ---------------------------------------------------------
elif page == "Reports & History":
    st.header("📁 Reports and Prediction History")
    st.write("Browse and download saved reports and the dashboard's simulated prediction history.")
    import re
    report_files = []
    if REPORTS_DIR.exists():
        report_files = sorted(
            [p for p in REPORTS_DIR.rglob("*") if p.is_file()
             and p.name != HISTORY_PATH.name
             and p.suffix.lower() in {".csv", ".txt", ".json", ".xlsx", ".png", ".jpg", ".jpeg"}],
            key=lambda p: p.name.lower(),
        )
    grouped = {}
    for path in report_files:
        match = re.search(r"day[\s_-]*0*(\d+)", str(path).lower())
        group = f"Day {int(match.group(1))} Reports" if match else "Other Reports"
        grouped.setdefault(group, []).append(path)
    def report_group_sort(label):
        match = re.search(r"Day (\d+)", label)
        return (0, int(match.group(1))) if match else (1, label.lower())
    if not report_files:
        st.info("No report files were found in the reports folder.")
    for group in sorted(grouped, key=report_group_sort):
        with st.expander(f"{group} ({len(grouped[group])} files)"):
            for path in grouped[group]:
                st.markdown(f"**{path.name}**")
                try:
                    suffix = path.suffix.lower()
                    if suffix == ".csv":
                        st.dataframe(pd.read_csv(path), use_container_width=True, hide_index=True)
                    elif suffix in {".png", ".jpg", ".jpeg"}:
                        st.image(str(path), use_container_width=True)
                    elif suffix == ".txt":
                        st.text(path.read_text(encoding="utf-8", errors="replace")[:12000])
                    elif suffix == ".json":
                        st.code(path.read_text(encoding="utf-8", errors="replace")[:12000], language="json")
                    else:
                        st.caption("Excel report file.")
                    st.download_button(
                        f"Download {path.name}", path.read_bytes(), path.name,
                        key=f"history_report_{path.name}",
                    )
                except Exception as exc:
                    st.warning(f"Could not display {path.name}: {exc}")

    st.subheader("Dashboard prediction history")
    if HISTORY_PATH.exists():
        try:
            history = pd.read_csv(HISTORY_PATH)
            st.dataframe(history.tail(500).iloc[::-1], use_container_width=True)
            st.download_button(
                "Download complete dashboard history",
                history.to_csv(index=False).encode("utf-8"),
                "dashboard_prediction_history.csv", "text/csv",
            )
        except Exception as exc:
            st.error(f"Could not load dashboard history: {exc}")
    else:
        st.info("History will appear after the Live Dashboard generates its first reading.")

# ---------------------------------------------------------
# About
# ---------------------------------------------------------
elif page == "About Project":
    st.header("ℹ️ About This Project")
    st.markdown(
        """
        **Project:** Explainable Machine Learning for Predictive Maintenance in Industry 4.0

        **Dataset:** AI4I 2020 Predictive Maintenance Dataset

        **Dashboard mode:** Simulated sensor readings

        **Models:** Logistic Regression, Random Forest, Decision Tree, and Gradient Boosting

        **Functions:** Model-based predictions, test-set evaluation, saved feature-importance/SHAP visuals,
        prediction history, browser sound alert, and optional email alert.

        **Limitations**
        - No physical sensor hardware is connected.
        - The models have not been validated on a specific industrial machine.
        - Simulated readings demonstrate the software workflow only.
        - A model-estimated failure score is not necessarily a calibrated real-world probability.
        - Email alerts require valid SMTP configuration and successful delivery testing.
        - Saved SHAP plots are global/offline artifacts, not explanations recalculated for every live prediction.
        """
    )
    st.subheader("Expected project structure")
    st.code(
        "predictive-maintenance-xai/\n"
        "├── app.py\n"
        "├── models/\n"
        "│   ├── random_forest.joblib\n"
        "│   ├── logistic_regression.joblib\n"
        "│   ├── decision_tree.joblib\n"
        "│   └── gradient_boosting.joblib\n"
        "├── data/processed/\n"
        "│   ├── X_test.csv\n"
        "│   └── y_test.csv\n"
        "└── reports/\n"
        "    ├── day5_shap_summary.png\n"
        "    └── ...",
        language="text",
    )

st.sidebar.divider()
st.sidebar.caption("Simulated sensor stream • Optional email alerts • No physical sensor integration")
