"""Interactive Streamlit interface for the water potability classifier."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "water_potability.csv"
MODEL_PATH = ROOT / "artifacts" / "water_potability_model.joblib"
FEATURE_INFO = [
    ("ph", "pH", "pH scale"),
    ("Hardness", "Hardness", "mg/L"),
    ("Solids", "Solids (TDS)", "ppm"),
    ("Chloramines", "Chloramines", "ppm"),
    ("Sulfate", "Sulfate", "mg/L"),
    ("Conductivity", "Conductivity", "µS/cm"),
    ("Organic_carbon", "Organic carbon", "ppm"),
    ("Trihalomethanes", "Trihalomethanes", "µg/L"),
    ("Turbidity", "Turbidity", "NTU"),
]

st.set_page_config(page_title="AI Water Quality Prediction", page_icon="💧", layout="centered")
st.title("AI-Based Water Quality Prediction")
st.markdown("Enter the nine water measurements below to get the model's dataset classification.")

if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error("The model or dataset is missing. Open the VS Code terminal in this project folder and run `python train_model.py`.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_defaults():
    return pd.read_csv(DATA_PATH)

model = load_model()
data = load_defaults()

st.subheader("Water measurements")
st.caption("You can use the example median values as a starting point, or enter your own lab readings.")

with st.form("water_quality_form"):
    values = {}
    grid = st.columns(3)
    for index, (key, label, unit) in enumerate(FEATURE_INFO):
        default = float(data[key].median())
        values[key] = grid[index % 3].number_input(
            f"{label} ({unit})",
            value=default,
            format="%.3f",
            help=f"Example value is the training dataset median: {default:.3f} {unit}.",
        )
    submitted = st.form_submit_button("Predict water class", type="primary", use_container_width=True)

if submitted:
    input_data = pd.DataFrame([values], columns=[item[0] for item in FEATURE_INFO])
    predicted_class = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    class_index = list(model.classes_).index(predicted_class)
    model_probability = float(probabilities[class_index]) * 100

    st.subheader("Prediction")
    if predicted_class == 1:
        st.success("Model classification: Potable class (1)")
    else:
        st.warning("Model classification: Non-potable class (0)")
    st.metric("Model's estimated probability for this class", f"{model_probability:.1f}%")
    st.caption("This percentage reflects the model output; it is not a laboratory-confirmed safety probability.")

st.divider()
with st.expander("About the model and dataset"):
    st.write(
        "The app uses a Random Forest trained on the Water Potability dataset. Missing training "
        "measurements are filled with training-set medians. The example input values above are "
        "medians from that dataset."
    )
    st.write(f"Dataset rows: {len(data):,} · Input features: {len(FEATURE_INFO)}")
    st.warning(
        "This is an educational preliminary screening tool. A model prediction cannot establish "
        "that water is safe to drink. Use appropriate accredited laboratory testing and applicable standards."
    )
