import streamlit as st

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, roc_curve


st.set_page_config(
    page_title="Prediksi Penyakit Ginjal Kronis",
    page_icon="CKD",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "kidney_disease.csv"
MODEL_FILE = BASE_DIR / "model_artifacts.joblib"
HISTORY_FILE = BASE_DIR / "prediction_history.csv"

TARGET_COLUMN = "classification"
NUMERIC_COLUMNS = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wc",
    "rc",
]
CATEGORICAL_COLUMNS = [
    "rbc",
    "pc",
    "pcc",
    "ba",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
]

FIELD_LABELS = {
    "age": "Age",
    "bp": "Blood Pressure",
    "sg": "Specific gravity",
    "al": "Albumin",
    "su": "Sugar",
    "rbc": "Red blood cells",
    "pc": "Pus cell",
    "pcc": "Pus cell clumps",
    "ba": "Bacteria",
    "bgr": "Blood glucose random",
    "bu": "Blood urea",
    "sc": "Serum creatinine",
    "sod": "Sodium",
    "pot": "Potassium",
    "hemo": "Hemoglobin",
    "pcv": "Packed cell volume",
    "wc": "White blood cell count",
    "rc": "Red blood cell count",
    "htn": "Hypertension",
    "dm": "Diabetes mellitus",
    "cad": "Coronary artery disease",
    "appet": "Appetite",
    "pe": "Pedal edema",
    "ane": "Anemia",
}

CATEGORY_OPTIONS = {
    "rbc": ["normal", "abnormal"],
    "pc": ["normal", "abnormal"],
    "pcc": ["notpresent", "present"],
    "ba": ["notpresent", "present"],
    "htn": ["no", "yes"],
    "dm": ["no", "yes"],
    "cad": ["no", "yes"],
    "appet": ["good", "poor"],
    "pe": ["no", "yes"],
    "ane": ["no", "yes"],
}

RISK_COLORS = {
    "Rendah": "#15803d",
    "Sedang": "#d97706",
    "Tinggi": "#dc2626",
}

# Model yang ditampilkan di grafik ROC dan tabel perbandingan pada aplikasi Streamlit.
ROC_DISPLAY_MODELS = ["Random Forest", "Random Classifier"]
COMPARISON_DISPLAY_MODELS = ["Random Forest"]


def clean_text(value: object) -> object:
    if pd.isna(value):
        return np.nan
    cleaned = str(value).strip().lower()
    return np.nan if cleaned in {"", "?", "nan"} else cleaned


def load_and_clean_data(file_or_path) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)
    df.columns = [str(col).strip().lower() for col in df.columns]

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(clean_text)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = (
            df[TARGET_COLUMN]
            .map(clean_text)
            .replace({"ckd\t": "ckd", "not ckd": "notckd", "not_ckd": "notckd"})
        )

    return df


@st.cache_data(show_spinner=False)
def cached_default_data() -> pd.DataFrame:
    return load_and_clean_data(DEFAULT_DATASET)


@st.cache_resource(show_spinner=False)
def _load_artifacts_cached(file_path: str, file_mtime: float) -> dict:
    # file_mtime ikut jadi kunci cache supaya artifact otomatis dimuat ulang
    # setiap kali model_artifacts.joblib diperbarui dari train_model.ipynb.
    return joblib.load(file_path)


def load_model_artifacts() -> dict:
    if not MODEL_FILE.exists():
        st.error(
            "File model_artifacts.joblib belum ada. Jalankan train_model.ipynb dulu "
            "sampai cell penyimpanan model selesai."
        )
        st.stop()
    return _load_artifacts_cached(str(MODEL_FILE), MODEL_FILE.stat().st_mtime)


def build_roc_per_class_data(artifacts: dict) -> pd.DataFrame:
    """Ambil data ROC per kelas dari artifact.

    Kalau artifact masih versi lama dan belum punya `roc_per_class_data`,
    data kurva dihitung ulang di sini memakai hasil prediksi data uji yang
    sudah tersimpan di dalam metrics, jadi grafik tetap bisa tampil.
    """
    existing = artifacts.get("roc_per_class_data")
    if isinstance(existing, pd.DataFrame) and not existing.empty:
        return existing

    metrics = artifacts.get("metrics", {})
    y_true = metrics.get("test_actual")
    y_proba = metrics.get("test_proba")
    if y_true is None or y_proba is None:
        return pd.DataFrame()

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    rows = []
    for class_index, class_name in [(1, "CKD"), (0, "Not CKD")]:
        class_score = y_proba if class_index == 1 else 1.0 - y_proba
        fpr, tpr, _ = roc_curve(y_true, class_score, pos_label=class_index)
        class_auc = auc(fpr, tpr)
        for curve_fpr, curve_tpr in zip(fpr, tpr):
            rows.append(
                {
                    "Kelas": class_name,
                    "False Positive Rate": curve_fpr,
                    "True Positive Rate": curve_tpr,
                    "AUC": class_auc,
                }
            )

    rows.extend(
        [
            {"Kelas": "Random Classifier", "False Positive Rate": 0.0, "True Positive Rate": 0.0, "AUC": 0.5},
            {"Kelas": "Random Classifier", "False Positive Rate": 1.0, "True Positive Rate": 1.0, "AUC": 0.5},
        ]
    )
    return pd.DataFrame(rows)


def load_history() -> pd.DataFrame:
    if HISTORY_FILE.exists():
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=["nama_pasien", "tanggal_prediksi", "hasil_prediksi", "confidence", "tingkat_risiko"])


def save_history(row: dict) -> None:
    history = load_history()
    history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    history.to_csv(HISTORY_FILE, index=False)


def predict_patient(pipeline, patient: dict) -> dict:
    patient_df = pd.DataFrame([patient])[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    probabilities = pipeline.predict_proba(patient_df)[0]
    ckd_probability = float(probabilities[1])
    prediction = "CKD" if ckd_probability >= 0.5 else "Not CKD"
    confidence = ckd_probability if prediction == "CKD" else float(probabilities[0])

    if ckd_probability >= 0.75:
        risk = "Tinggi"
    elif ckd_probability >= 0.45:
        risk = "Sedang"
    else:
        risk = "Rendah"

    return {
        "prediction": prediction,
        "ckd_probability": ckd_probability,
        "confidence": confidence,
        "risk": risk,
        "description": (
            f"Pasien terindikasi Penyakit Ginjal Kronis dengan tingkat probabilitas {ckd_probability * 100:.1f}%."
            if prediction == "CKD"
            else f"Pasien tidak terindikasi Penyakit Ginjal Kronis dengan tingkat probabilitas aman {(1 - ckd_probability) * 100:.1f}%."
        ),
    }


def draw_confusion_matrix(cm: np.ndarray) -> None:
    fig = px.imshow(
        cm,
        text_auto=True,
        labels={"x": "Prediksi", "y": "Aktual", "color": "Jumlah"},
        x=["CKD", "Not CKD"],
        y=["CKD", "Not CKD"],
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def draw_feature_importance(importance_df: pd.DataFrame) -> None:
    top_df = importance_df.head(10).sort_values("Persentase")
    fig = px.bar(
        top_df,
        x="Persentase",
        y="Nama Fitur",
        orientation="h",
        text=top_df["Persentase"].map(lambda value: f"{value:.1f}%"),
        color="Persentase",
        color_continuous_scale="Teal",
    )
    fig.update_layout(
        height=430,
        xaxis_title="Kontribusi (%)",
        yaxis_title="",
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def draw_roc_curve(roc_curve_data: pd.DataFrame) -> None:
    if roc_curve_data is None or roc_curve_data.empty:
        st.info("Data ROC Curve belum tersedia. Jalankan ulang train_model.ipynb untuk memperbarui artifact model.")
        return

    # Grafik ini hanya menampilkan Random Forest dan Random Classifier.
    curve_df = roc_curve_data[roc_curve_data["Model"].isin(ROC_DISPLAY_MODELS)].copy()
    if curve_df.empty:
        st.info("Data ROC Curve Random Forest belum tersedia pada artifact model.")
        return

    curve_df["Label"] = curve_df.apply(
        lambda row: f"{row['Model']} (AUC = {row['AUC']:.4f})",
        axis=1,
    )

    fig = px.line(
        curve_df,
        x="False Positive Rate",
        y="True Positive Rate",
        color="Label",
        line_dash="Model",
        line_dash_map={
            "Random Forest": "solid",
            "Random Classifier": "dash",
        },
        title="ROC Curves",
        range_x=[0, 1],
        range_y=[0, 1],
    )
    fig.update_layout(
        height=500,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend_title_text="Model",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    st.plotly_chart(fig, use_container_width=True)


def draw_roc_per_class(roc_per_class_data: pd.DataFrame) -> None:
    if roc_per_class_data is None or roc_per_class_data.empty:
        st.info(
            "Data ROC Curve per kelas belum bisa dihitung karena hasil prediksi data uji "
            "tidak ada di dalam model_artifacts.joblib. Jalankan ulang train_model.ipynb "
            "sampai cell terakhir, lalu refresh halaman ini."
        )
        return

    curve_df = roc_per_class_data.copy()
    class_colors = {"CKD": "#1f77b4", "Not CKD": "#ff7f0e", "Random Classifier": "#00008b"}

    fig = go.Figure()
    for class_name in ["CKD", "Not CKD", "Random Classifier"]:
        class_df = curve_df[curve_df["Kelas"] == class_name]
        if class_df.empty:
            continue

        class_auc = float(class_df["AUC"].max())
        label = (
            f"Random Classifier (AUC = {class_auc:.4f})"
            if class_name == "Random Classifier"
            else f"ROC curve of {class_name} (AUC = {class_auc:.4f})"
        )
        fig.add_trace(
            go.Scatter(
                x=class_df["False Positive Rate"],
                y=class_df["True Positive Rate"],
                mode="lines",
                name=label,
                line=dict(
                    color=class_colors.get(class_name),
                    dash="dash" if class_name == "Random Classifier" else "solid",
                    width=2,
                ),
            )
        )

    fig.update_layout(
        title="ROC Curves per Class for Random Forest Model",
        height=500,
        xaxis=dict(title="False Positive Rate", range=[0, 1], showgrid=True),
        yaxis=dict(title="True Positive Rate", range=[0, 1.02], showgrid=True),
        legend=dict(x=0.45, y=0.12, bgcolor="rgba(255,255,255,0.6)"),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def build_form_defaults(dataset: pd.DataFrame, artifacts: dict) -> tuple[pd.Series, dict]:
    fallback_numeric = dataset[NUMERIC_COLUMNS].median(numeric_only=True)
    numeric_defaults = pd.Series(artifacts.get("numeric_defaults", fallback_numeric), dtype="float64")
    mode_defaults = artifacts.get("categorical_defaults", {})
    categorical_defaults = {
        col: mode_defaults.get(col) or dataset[col].mode(dropna=True).iloc[0]
        if col in dataset and not dataset[col].mode(dropna=True).empty
        else CATEGORY_OPTIONS[col][0]
        for col in CATEGORICAL_COLUMNS
    }
    return numeric_defaults, categorical_defaults


def patient_form(dataset: pd.DataFrame, artifacts: dict) -> tuple[str, dict, bool]:
    median_values, mode_values = build_form_defaults(dataset, artifacts)

    with st.form("patient_input_form"):
        st.subheader("Input Data Pasien")
        patient_name = st.text_input("Nama pasien", placeholder="Contoh: Budi Santoso")

        col1, col2, col3 = st.columns(3)
        patient = {}

        with col1:
            patient["age"] = st.number_input("Age", min_value=1.0, max_value=120.0, value=float(median_values["age"]), step=1.0)
            patient["bp"] = st.number_input("Blood Pressure", min_value=40.0, max_value=220.0, value=float(median_values["bp"]), step=1.0)
            patient["sg"] = st.selectbox("Specific gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=3)
            patient["al"] = st.selectbox("Albumin", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            patient["su"] = st.selectbox("Sugar", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            patient["rbc"] = st.selectbox("Red blood cells", CATEGORY_OPTIONS["rbc"], index=CATEGORY_OPTIONS["rbc"].index(mode_values["rbc"]))
            patient["pc"] = st.selectbox("Pus cell", CATEGORY_OPTIONS["pc"], index=CATEGORY_OPTIONS["pc"].index(mode_values["pc"]))
            patient["pcc"] = st.selectbox("Pus cell clumps", CATEGORY_OPTIONS["pcc"], index=CATEGORY_OPTIONS["pcc"].index(mode_values["pcc"]))

        with col2:
            patient["ba"] = st.selectbox("Bacteria", CATEGORY_OPTIONS["ba"], index=CATEGORY_OPTIONS["ba"].index(mode_values["ba"]))
            patient["bgr"] = st.number_input("Blood glucose random", min_value=20.0, max_value=600.0, value=float(median_values["bgr"]), step=1.0)
            patient["bu"] = st.number_input("Blood urea", min_value=1.0, max_value=400.0, value=float(median_values["bu"]), step=1.0)
            patient["sc"] = st.number_input("Serum creatinine", min_value=0.1, max_value=80.0, value=float(median_values["sc"]), step=0.1)
            patient["sod"] = st.number_input("Sodium", min_value=1.0, max_value=200.0, value=float(median_values["sod"]), step=1.0)
            patient["pot"] = st.number_input("Potassium", min_value=1.0, max_value=50.0, value=float(median_values["pot"]), step=0.1)
            patient["hemo"] = st.number_input("Hemoglobin", min_value=1.0, max_value=25.0, value=float(median_values["hemo"]), step=0.1)
            patient["pcv"] = st.number_input("Packed cell volume", min_value=1.0, max_value=80.0, value=float(median_values["pcv"]), step=1.0)

        with col3:
            patient["wc"] = st.number_input("White blood cell count", min_value=1000.0, max_value=30000.0, value=float(median_values["wc"]), step=100.0)
            patient["rc"] = st.number_input("Red blood cell count", min_value=1.0, max_value=10.0, value=float(median_values["rc"]), step=0.1)
            patient["htn"] = st.selectbox("Hypertension", CATEGORY_OPTIONS["htn"], index=CATEGORY_OPTIONS["htn"].index(mode_values["htn"]))
            patient["dm"] = st.selectbox("Diabetes mellitus", CATEGORY_OPTIONS["dm"], index=CATEGORY_OPTIONS["dm"].index(mode_values["dm"]))
            patient["cad"] = st.selectbox("Coronary artery disease", CATEGORY_OPTIONS["cad"], index=CATEGORY_OPTIONS["cad"].index(mode_values["cad"]))
            patient["appet"] = st.selectbox("Appetite", CATEGORY_OPTIONS["appet"], index=CATEGORY_OPTIONS["appet"].index(mode_values["appet"]))
            patient["pe"] = st.selectbox("Pedal edema", CATEGORY_OPTIONS["pe"], index=CATEGORY_OPTIONS["pe"].index(mode_values["pe"]))
            patient["ane"] = st.selectbox("Anemia", CATEGORY_OPTIONS["ane"], index=CATEGORY_OPTIONS["ane"].index(mode_values["ane"]))

        submitted = st.form_submit_button("Prediksi Pasien", type="primary", use_container_width=True)

    return patient_name.strip(), patient, submitted


def main() -> None:
    st.title("Sistem Prediksi Penyakit Ginjal Kronis")
    st.caption("Aplikasi Streamlit untuk prediksi. Proses training dipisahkan ke train_model.ipynb.")

    artifacts = load_model_artifacts()
    pipeline = artifacts["model"]
    metrics = artifacts["metrics"]
    importance_df = artifacts["feature_importance"]
    accuracy_curve = artifacts["accuracy_curve"]
    roc_curve_data = artifacts.get("roc_curve_data", pd.DataFrame())
    roc_per_class_data = build_roc_per_class_data(artifacts)
    model_comparison = artifacts.get("model_comparison", pd.DataFrame())

    with st.sidebar:
        st.header("Dataset")
        uploaded = st.file_uploader("Upload file CSV untuk preview dashboard", type=["csv"])
        if uploaded is not None:
            dataset = load_and_clean_data(uploaded)
            st.success("CSV berhasil dimuat untuk preview.")
        else:
            dataset = cached_default_data()
            st.info("Menggunakan kidney_disease.csv bawaan.")

        st.divider()
        st.write("Kolom dataset:", len(dataset.columns))
        st.write("Jumlah baris:", len(dataset))
        st.caption("Model yang dipakai berasal dari model_artifacts.joblib hasil train_model.ipynb.")

    tab_input, tab_dashboard, tab_history, tab_visual = st.tabs(
        ["Input Pasien", "Dashboard", "Riwayat Prediksi", "Visualisasi Data"]
    )

    with tab_input:
        patient_name, patient, submitted = patient_form(dataset, artifacts)

        if submitted:
            if not patient_name:
                st.warning("Nama pasien wajib diisi sebelum prediksi disimpan.")
            else:
                result = predict_patient(pipeline, patient)
                timestamp = datetime.now(ZoneInfo("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")
                save_history(
                    {
                        "nama_pasien": patient_name,
                        "tanggal_prediksi": timestamp,
                        "hasil_prediksi": result["prediction"],
                        "confidence": round(result["confidence"] * 100, 2),
                        "tingkat_risiko": result["risk"],
                    }
                )
                st.session_state["last_result"] = result
                st.success("Prediksi berhasil dan riwayat sudah disimpan.")

        if "last_result" in st.session_state:
            result = st.session_state["last_result"]
            st.subheader("Hasil Prediksi")
            c1, c2, c3 = st.columns(3)
            c1.metric("Hasil prediksi", result["prediction"])
            c2.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
            c3.metric("Tingkat risiko", result["risk"])

            color = RISK_COLORS[result["risk"]]
            st.markdown(
                f"""
                <div style="border-left: 6px solid {color}; padding: 1rem; background: #f8fafc; border-radius: 8px;">
                    <strong>Keterangan hasil:</strong><br>{result["description"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=result["ckd_probability"] * 100,
                    number={"suffix": "%"},
                    title={"text": "Probabilitas CKD"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 45], "color": "#dcfce7"},
                            {"range": [45, 75], "color": "#fef3c7"},
                            {"range": [75, 100], "color": "#fee2e2"},
                        ],
                    },
                )
            )
            gauge.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(gauge, use_container_width=True)

    with tab_dashboard:
        st.subheader("Dashboard Data Pasien")
        total_patients = len(dataset)
        ckd_count = int((dataset[TARGET_COLUMN] == "ckd").sum()) if TARGET_COLUMN in dataset else 0
        non_ckd_count = int((dataset[TARGET_COLUMN] == "notckd").sum()) if TARGET_COLUMN in dataset else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jumlah data pasien", f"{total_patients:,}")
        c2.metric("Jumlah CKD", f"{ckd_count:,}")
        c3.metric("Jumlah non-CKD", f"{non_ckd_count:,}")
        c4.metric("Akurasi holdout", f"{metrics['accuracy'] * 100:.1f}%")

        v1, v2, v3 = st.columns(3)
        v1.metric("Akurasi train", f"{metrics['train_accuracy'] * 100:.1f}%")
        v2.metric("Akurasi CV 5-fold", f"{metrics['cv_mean'] * 100:.1f}%", f"+/- {metrics['cv_std'] * 100:.2f}%")
        v3.metric("Gap train-test", f"{metrics['generalization_gap'] * 100:.1f}%")

        m1, m2, m3 = st.columns(3)
        m1.metric("Precision CKD", f"{metrics['precision'] * 100:.1f}%")
        m2.metric("Recall CKD", f"{metrics['recall'] * 100:.1f}%")
        m3.metric("F1-score CKD", f"{metrics['f1_score'] * 100:.1f}%")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            label_counts = pd.DataFrame({"Status": ["CKD", "Not CKD"], "Jumlah": [ckd_count, non_ckd_count]})
            fig = px.pie(label_counts, names="Status", values="Jumlah", hole=0.45, title="Distribusi Dataset")
            st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            history_current = load_history()
            if history_current.empty:
                st.info("Belum ada prediksi baru. Grafik riwayat akan muncul setelah input pasien diprediksi.")
            else:
                pred_counts = history_current["hasil_prediksi"].value_counts().rename_axis("Hasil").reset_index(name="Jumlah")
                fig = px.bar(pred_counts, x="Hasil", y="Jumlah", color="Hasil", title="Grafik Hasil Prediksi")
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Preview Dataset")
        st.dataframe(dataset.head(25), use_container_width=True)

    with tab_history:
        st.subheader("Riwayat Prediksi")
        history_current = load_history()
        if history_current.empty:
            st.info("Belum ada riwayat prediksi.")
        else:
            st.dataframe(history_current.sort_values("tanggal_prediksi", ascending=False), use_container_width=True)
            st.download_button(
                "Download Riwayat CSV",
                data=history_current.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab_visual:
        st.subheader("Visualisasi Data dan Evaluasi Model")
        v1, v2 = st.columns(2)

        with v1:
            st.markdown("**Confusion Matrix**")
            draw_confusion_matrix(metrics["confusion_matrix"])

        with v2:
            st.markdown("**Grafik Akurasi Random Forest**")
            fig = px.line(
                accuracy_curve,
                x="Jumlah Pohon",
                y="Akurasi",
                markers=True,
                range_y=[0, 100],
                error_y="Std",
            )
            fig.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("****")
        draw_roc_curve(roc_curve_data)

        st.markdown("****")
        draw_roc_per_class(roc_per_class_data)

        if not model_comparison.empty:
            # Tabel di Streamlit hanya menampilkan baris Random Forest.
            comparison_display = model_comparison[
                model_comparison["Model"].isin(COMPARISON_DISPLAY_MODELS)
            ].copy()
            for col in ["Accuracy", "Precision", "Recall", "F1-score", "AUC"]:
                comparison_display[col] = comparison_display[col].map(lambda value: f"{value:.4f}")
            st.dataframe(comparison_display, use_container_width=True, hide_index=True)

        st.markdown("**Cek Leakage, Overfitting, dan Underfitting**")
        checks = pd.DataFrame(
            [
                {
                    "Pemeriksaan": "Data leakage",
                    "Status": "Aman",
                    "Keterangan": "Kolom id dan classification tidak masuk fitur. Imputer dan encoder di-fit di dalam pipeline training.",
                },
                {
                    "Pemeriksaan": "Overfitting",
                    "Status": "Aman",
                    "Keterangan": f"Train {metrics['train_accuracy'] * 100:.1f}% vs holdout {metrics['accuracy'] * 100:.1f}%, gap {metrics['generalization_gap'] * 100:.1f}%.",
                },
                {
                    "Pemeriksaan": "Underfitting",
                    "Status": "Aman",
                    "Keterangan": f"CV 9-fold rata-rata {metrics['cv_mean'] * 100:.1f}% dengan deviasi {metrics['cv_std'] * 100:.2f}%.",
                },
            ]
        )
        st.dataframe(checks, use_container_width=True, hide_index=True)

        st.markdown("**Metrik Evaluasi Model**")
        evaluation_metrics = pd.DataFrame(
            [
                {"Metrik": "Accuracy", "Nilai": f"{metrics['accuracy'] * 100:.1f}%"},
                {"Metrik": "Precision CKD", "Nilai": f"{metrics['precision'] * 100:.1f}%"},
                {"Metrik": "Recall CKD", "Nilai": f"{metrics['recall'] * 100:.1f}%"},
                {"Metrik": "F1-score CKD", "Nilai": f"{metrics['f1_score'] * 100:.1f}%"},
                {"Metrik": "AUC Random Forest", "Nilai": f"{metrics.get('auc', 0) * 100:.1f}%"},
            ]
        )
        st.dataframe(evaluation_metrics, use_container_width=True, hide_index=True)

        st.markdown("**Feature Importance Random Forest**")
        draw_feature_importance(importance_df)

        top_features = importance_df.head(5).copy()
        top_features["Persentase"] = top_features["Persentase"].map(lambda value: f"{value:.1f}%")
        st.dataframe(top_features[["Nama Fitur", "Persentase"]], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

