"""Streamlit frontend for heart disease risk prediction.

This file contains only UI and UX logic. The ML training and inference logic
is imported from ai_pipeline.py.
"""

# Postpones evaluation of type hints to runtime import time.
from __future__ import annotations

# Used only for a short spinner delay to improve user feedback.
import time

# Provides the lightweight data container for form inputs.
from dataclasses import dataclass

# Type used for the model-ready encoded feature dictionary.
from typing import Dict

# Streamlit powers the complete frontend.
import streamlit as st

# AI pipeline imports: dataset path, model metadata class, builder, predictor.
from ai_pipeline import DATASET_PATH, ModelBundle, build_optimized_model, predict_disease


# Encodes sex values exactly as expected by the trained dataset schema.
SEX_MAP = {"Female": 0, "Male": 1}

# Encodes chest pain categories to their numeric model representation.
CHEST_PAIN_MAP = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3,
}

# Encodes fasting blood sugar yes/no into numeric feature values.
FBS_MAP = {"No": 0, "Yes": 1}

# Encodes resting ECG labels to model-compatible integers.
REST_ECG_MAP = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2,
}

# Encodes exercise-induced angina yes/no into numeric feature values.
EXANG_MAP = {"No": 0, "Yes": 1}

# Encodes slope categories to model-compatible integers.
ST_SLOPE_MAP = {
    "Up-sloping": 0,
    "Flat": 1,
    "Down-sloping": 2,
}

# Friendly labels for thal numeric codes shown in the advanced section.
THAL_LABELS = {
    0: "Unknown / 0",
    1: "Normal",
    2: "Fixed Defect",
    3: "Reversible Defect",
}


@dataclass
class PatientInput:
    """Represents raw user selections from the UI widgets."""

    # Core demographic and clinical fields.
    age: int
    sex: str
    chest_pain_type: str
    resting_bp: int
    cholesterol: int
    fasting_blood_sugar: str
    resting_ecg: str
    max_heart_rate: int
    exercise_induced_angina: str
    oldpeak: float
    st_slope: str

    # Advanced optional fields that match additional dataset columns.
    ca: int
    thal: int

    def to_model_features(self) -> Dict[str, float]:
        """Convert UI selections into numeric features consumed by the model."""
        # Return a dictionary keyed by exact training column names.
        return {
            # Age is already numeric; cast to float for model consistency.
            "age": float(self.age),
            # Sex uses fixed lookup encoding.
            "sex": float(SEX_MAP[self.sex]),
            # Chest pain type uses fixed lookup encoding.
            "cp": float(CHEST_PAIN_MAP[self.chest_pain_type]),
            # Resting blood pressure in mm Hg.
            "trestbps": float(self.resting_bp),
            # Serum cholesterol in mg/dl.
            "chol": float(self.cholesterol),
            # Fasting blood sugar encoded to 0/1.
            "fbs": float(FBS_MAP[self.fasting_blood_sugar]),
            # Resting ECG encoded to 0/1/2.
            "restecg": float(REST_ECG_MAP[self.resting_ecg]),
            # Maximum heart rate achieved.
            "thalach": float(self.max_heart_rate),
            # Exercise angina encoded to 0/1.
            "exang": float(EXANG_MAP[self.exercise_induced_angina]),
            # ST depression value.
            "oldpeak": float(self.oldpeak),
            # ST slope encoded to 0/1/2.
            "slope": float(ST_SLOPE_MAP[self.st_slope]),
            # Number of major vessels from advanced section.
            "ca": float(self.ca),
            # Thal code from advanced section.
            "thal": float(self.thal),
        }


@st.cache_resource(show_spinner=True)
def get_model_bundle(csv_path: str) -> ModelBundle:
    """Cache model training results for fast subsequent predictions."""
    # Delegate all model construction logic to the pipeline module.
    return build_optimized_model(csv_path)


def configure_page() -> None:
    """Set page-level settings and custom CSS for polished UI."""
    # Configure title, icon, and layout before rendering any widgets.
    st.set_page_config(
        page_title="CardioCare AI",
        page_icon="🫀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS for branding, clean chrome, and visual consistency.
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Fraunces:opsz,wght@9..144,600&display=swap');

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {visibility: hidden;}

        :root {
            --bg-card: #ffffff;
            --bg-soft: #f5fbfb;
            --text-main: #102a30;
            --text-muted: #4f6b70;
            --brand: #0f8a8f;
            --brand-dark: #0b5e62;
            --accent: #e76f51;
            --border-soft: #d3e8ea;
            --shadow-soft: 0 14px 34px rgba(10, 53, 61, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 8%, rgba(15,138,143,0.15) 0%, transparent 35%),
                radial-gradient(circle at 92% 10%, rgba(231,111,81,0.13) 0%, transparent 33%),
                linear-gradient(180deg, #fcfeff 0%, #eef7f8 100%);
            color: var(--text-main);
        }

        html, body, [class*="css"] {
            font-family: "Manrope", sans-serif;
        }

        .hero-banner {
            background: linear-gradient(130deg, #0f8a8f 0%, #0b5e62 75%);
            color: #f6ffff;
            border-radius: 18px;
            padding: 18px 22px;
            margin-bottom: 8px;
            box-shadow: var(--shadow-soft);
        }

        .hero-tag {
            display: inline-block;
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.78rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-soft);
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: var(--shadow-soft);
            margin-bottom: 12px;
        }

        .card h3 {
            margin: 0;
            color: var(--brand-dark);
            font-family: "Fraunces", serif;
            font-size: 1.2rem;
        }

        .card p {
            margin: 6px 0 0 0;
            color: var(--text-muted);
            line-height: 1.45;
        }

        .stButton > button {
            background: linear-gradient(120deg, #0f8a8f 0%, #0b5e62 100%);
            color: #ffffff;
            border: none;
            border-radius: 12px;
            padding: 0.7rem 1.2rem;
            font-size: 1rem;
            font-weight: 700;
            box-shadow: 0 8px 20px rgba(11, 94, 98, 0.25);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(11, 94, 98, 0.32);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fcfd 0%, #edf6f7 100%);
            border-right: 1px solid var(--border-soft);
        }

        [data-testid="stMetricValue"] {
            color: var(--brand-dark);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_header() -> None:
    """Render hero section, title, subtitle, and medical disclaimer."""
    # Render a branded hero banner at the top of the page.
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-tag">AI-Assisted Cardiac Screening</div>
            <div>Fast, structured intake for heart disease risk estimation.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main page title.
    st.title("CardioCare AI")

    # Short explanation of the app purpose.
    st.subheader(
        "An intelligent pre-screening assistant to estimate heart disease risk "
        "from patient demographics, symptoms, and clinical indicators."
    )

    # Medical disclaimer to avoid clinical misuse.
    st.warning(
        "Medical Disclaimer: This tool is for educational and decision-support "
        "purposes only. It is not a substitute for professional medical advice, "
        "diagnosis, or treatment."
    )


def get_user_inputs(bundle: ModelBundle) -> PatientInput:
    """Collect patient data from grouped tabs and return a structured object."""
    # Introduce the form section in a styled card.
    st.markdown(
        """
        <div class="card">
            <h3>Patient Intake Form</h3>
            <p>Complete all sections to generate a risk estimate.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Organize inputs into three logical tabs.
    tab_demo, tab_symptoms, tab_labs = st.tabs(
        [
            "Patient Demographics",
            "Clinical Symptoms",
            "Lab Results & Vitals",
        ]
    )

    # Demographics tab for age and sex.
    with tab_demo:
        col_age, col_sex = st.columns(2)

        # Age input with medically plausible bounds.
        with col_age:
            age = st.slider(
                "Age",
                min_value=18,
                max_value=100,
                value=45,
                step=1,
                help="Patient age in years.",
            )

        # Sex categorical selector.
        with col_sex:
            sex = st.radio(
                "Sex",
                options=["Male", "Female"],
                index=0,
                horizontal=True,
                help="Encoding follows the notebook dataset mapping.",
            )

    # Symptoms tab for chest pain and exercise-induced angina.
    with tab_symptoms:
        col_cp, col_exang = st.columns(2)

        # Chest pain type selector.
        with col_cp:
            chest_pain_type = st.selectbox(
                "Chest Pain Type",
                options=[
                    "Typical Angina",
                    "Atypical Angina",
                    "Non-anginal Pain",
                    "Asymptomatic",
                ],
                index=2,
            )

        # Exercise-induced angina selector.
        with col_exang:
            exercise_induced_angina = st.radio(
                "Exercise-Induced Angina",
                options=["Yes", "No"],
                index=1,
                horizontal=True,
            )

    # Lab/vitals tab for numeric and encoded measurements.
    with tab_labs:
        col_left, col_mid, col_right = st.columns(3)

        # Left column: blood pressure, cholesterol, fasting sugar.
        with col_left:
            resting_bp = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=80,
                max_value=250,
                value=120,
                step=1,
            )
            cholesterol = st.number_input(
                "Cholesterol (mg/dl)",
                min_value=100,
                max_value=700,
                value=200,
                step=1,
            )
            fasting_blood_sugar = st.radio(
                "Fasting Blood Sugar > 120 mg/dl",
                options=["No", "Yes"],
                index=0,
                horizontal=True,
            )

        # Middle column: resting ECG and max heart rate.
        with col_mid:
            resting_ecg = st.selectbox(
                "Resting ECG",
                options=[
                    "Normal",
                    "ST-T Wave Abnormality",
                    "Left Ventricular Hypertrophy",
                ],
                index=0,
            )
            max_heart_rate = st.slider(
                "Max Heart Rate Achieved",
                min_value=60,
                max_value=220,
                value=150,
                step=1,
            )

        # Right column: oldpeak and ST slope.
        with col_right:
            oldpeak = st.number_input(
                "Oldpeak (ST depression)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                format="%.1f",
            )
            st_slope = st.selectbox(
                "ST Slope",
                options=["Up-sloping", "Flat", "Down-sloping"],
                index=1,
            )

        # Advanced fields improve feature completeness for the trained model.
        with st.expander("Advanced Inputs (Recommended for higher accuracy)"):
            # Calculate current default positions from allowed values.
            ca_default_index = bundle.ca_values.index(bundle.ca_default)
            thal_default_index = bundle.thal_values.index(bundle.thal_default)

            # Number of major vessels.
            ca = st.selectbox(
                "Number of Major Vessels (ca)",
                options=bundle.ca_values,
                index=ca_default_index,
                help="Integer count encoded in the original dataset.",
            )

            # Thalassemia code with friendly labels.
            thal = st.selectbox(
                "Thalassemia (thal)",
                options=bundle.thal_values,
                index=thal_default_index,
                format_func=lambda v: THAL_LABELS.get(v, f"Code {v}"),
                help="Encoded thal value used during model training.",
            )

    # Return all collected form values as a typed object.
    return PatientInput(
        age=age,
        sex=sex,
        chest_pain_type=chest_pain_type,
        resting_bp=resting_bp,
        cholesterol=cholesterol,
        fasting_blood_sugar=fasting_blood_sugar,
        resting_ecg=resting_ecg,
        max_heart_rate=max_heart_rate,
        exercise_induced_angina=exercise_induced_angina,
        oldpeak=oldpeak,
        st_slope=st_slope,
        ca=ca,
        thal=thal,
    )


def render_output(
    predicted_label: int,
    probability: float,
    threshold: float,
) -> None:
    """Render professional output messages based on model prediction."""
    # Output section title.
    st.markdown("### Assessment Result")

    # Visual risk bar with percent label.
    st.progress(float(probability), text=f"Estimated risk score: {probability * 100:.1f}%")

    # High-risk branch with clear caution.
    if predicted_label == 1:
        st.error(
            "High Risk Detected: The current profile indicates an elevated risk of "
            "heart disease. Please consult a cardiologist or healthcare provider "
            "as soon as possible for full clinical evaluation."
        )
        st.info(
            "Recommended next steps: Schedule a medical appointment, share complete "
            "clinical history, and consider diagnostic follow-up tests guided by a "
            "qualified professional."
        )
    # Low-risk branch with preventive guidance.
    else:
        st.success(
            "Low Risk Indicated: The current profile suggests a lower likelihood of "
            "heart disease at this time."
        )
        st.info(
            "Continue preventive care: maintain healthy habits, monitor symptoms, "
            "and keep regular checkups with your healthcare provider."
        )

    # Show calibrated threshold for transparency.
    st.caption(
        "Prediction uses optimized model selection and threshold tuning. "
        f"Current risk threshold: {threshold:.2f}."
    )


def main() -> None:
    """Application entry point."""
    # Apply page-level style and configuration first.
    configure_page()

    # Build and cache model bundle (training occurs once per session/runtime).
    try:
        model_bundle = get_model_bundle(str(DATASET_PATH))
    except Exception as exc:
        # Fail gracefully if model initialization cannot be completed.
        st.error("Unable to initialize optimized model from heart-disease.csv.")
        st.exception(exc)
        st.stop()

    # Render static intro content.
    create_header()

    # Collect raw user inputs.
    patient_input = get_user_inputs(model_bundle)

    # Encode UI input into model-ready numeric features.
    model_input = patient_input.to_model_features()

    # Visual divider before action section.
    st.markdown("---")

    # Main prediction action button.
    predict_clicked = st.button("Predict Heart Disease Risk", use_container_width=True)

    # Execute prediction flow only when user clicks the button.
    if predict_clicked:
        # Show spinner while running model inference.
        with st.spinner("Analyzing patient data..."):
            # Small delay to provide visible processing feedback.
            time.sleep(1.2)
            # Call pipeline inference function.
            predicted_label, probability = predict_disease(
                model_input,
                model_bundle,
            )

        # Render user-facing risk interpretation.
        render_output(predicted_label, probability, model_bundle.decision_threshold)


# Standard Python module entrypoint guard.
if __name__ == "__main__":
    main()
