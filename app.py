import os
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Wellness Tourism Package Predictor")
st.write(
    "Upload customer records to predict which customers are likely "
    "to purchase a wellness tourism package."
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "best_model.pkl"
)

FEATURE_COLUMNS = [
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()


st.subheader("1. Upload Customer Data")

st.write(
    "Accepted formats: Excel (.xlsx) and CSV (.csv). "
    "Each row should represent one customer."
)

template = pd.DataFrame(columns=FEATURE_COLUMNS)

template_buffer = BytesIO()
with pd.ExcelWriter(template_buffer, engine="openpyxl") as writer:
    template.to_excel(writer, index=False, sheet_name="Customers")

st.download_button(
    label="Download Excel Template",
    data=template_buffer.getvalue(),
    file_name="tourism_customer_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

uploaded_file = st.file_uploader(
    "Choose a customer data file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Clean column names
        df.columns = df.columns.str.strip()

        st.write(f"**Uploaded records:** {len(df)}")

        if df.empty:
            st.error("The uploaded file contains no customer records.")
            st.stop()

        # Retain original columns for the output, but do not use
        # identifiers or the target as model input.
        original_df = df.copy()

        ignored_columns = [
            col for col in ["Unnamed: 0", "CustomerID", "ProdTaken"]
            if col in df.columns
        ]

        model_input = df.drop(columns=ignored_columns, errors="ignore")

        missing_columns = [
            col for col in FEATURE_COLUMNS
            if col not in model_input.columns
        ]

        if missing_columns:
            st.error(
                "Your file is missing these required columns: "
                + ", ".join(missing_columns)
            )
            st.info(
                "Please use the Excel template above and provide all "
                "18 customer features."
            )
            st.stop()

        # Select columns in the exact order used during training.
        model_input = model_input[FEATURE_COLUMNS].copy()

        # Apply the same category cleanup used during training.
        model_input["Gender"] = model_input["Gender"].replace(
            {"Fe Male": "Female"}
        )
        model_input["MaritalStatus"] = model_input["MaritalStatus"].replace(
            {"Unmarried": "Single"}
        )
        model_input["Occupation"] = model_input["Occupation"].replace(
            {"Free Lancer": "Freelancer"}
        )

        # Validate and convert numeric fields.
        numeric_columns = [
            col for col in FEATURE_COLUMNS
            if col not in [
                "TypeofContact",
                "Occupation",
                "Gender",
                "ProductPitched",
                "MaritalStatus",
                "Designation"
            ]
        ]

        for col in numeric_columns:
            model_input[col] = pd.to_numeric(
                model_input[col], errors="coerce"
            )

        invalid_columns = [
            col for col in numeric_columns
            if model_input[col].isna().any()
        ]

        if invalid_columns:
            st.error(
                "Missing or non-numeric values were found in: "
                + ", ".join(invalid_columns)
                + ". Please correct the file and upload it again."
            )
            st.stop()

        if st.button("Predict Customer Purchases", type="primary"):
            predictions = model.predict(model_input)
            probabilities = model.predict_proba(model_input)[:, 1]

            results = original_df.copy()
            results["Predicted_ProdTaken"] = predictions.astype(int)
            results["Purchase_Probability"] = probabilities
            results["Prediction"] = np.where(
                predictions == 1,
                "Likely to Purchase",
                "Unlikely to Purchase"
            )

            st.subheader("2. Prediction Summary")

            col1, col2, col3 = st.columns(3)

            likely_count = int((predictions == 1).sum())
            unlikely_count = int((predictions == 0).sum())

            col1.metric("Total Customers", len(results))
            col2.metric("Likely to Purchase", likely_count)
            col3.metric("Unlikely to Purchase", unlikely_count)

            st.write("**Customer-level predictions**")

            display_results = results.copy()
            display_results["Purchase_Probability"] = (
                display_results["Purchase_Probability"] * 100
            ).round(2)

            st.dataframe(
                display_results,
                use_container_width=True
            )

            # Generate downloadable Excel results.
            output = BytesIO()

            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                results.to_excel(
                    writer,
                    index=False,
                    sheet_name="Predictions"
                )

            st.download_button(
                label="Download Prediction Results (Excel)",
                data=output.getvalue(),
                file_name="tourism_prediction_results.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                )
            )

            st.caption(
                "Purchase probabilities are model estimates, "
                "not guarantees of customer behavior."
            )

    except Exception as e:
        st.error(f"Unable to process the uploaded file: {e}")
