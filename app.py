import streamlit as st
import pandas as pd
import joblib  # For loading the trained model

# Load the trained model
def load_model():
    model = joblib.load('breast_cancer_model.pkl')
    return model

# App title and instructions
st.title("Breast Cancer Prediction App")

st.markdown("""
### Instructions:
1. Upload a CSV file containing the feature columns required for prediction.
2. Ensure that the dataset does not include the `target` column during prediction.
3. The model will process the data and display predictions for each row in the dataset.
""")

# File uploader for dataset
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset:")
    st.write(df.head())

    try:
        # Load the trained model
        model = load_model()

        # Separate features (X) and predict outcomes
        X = df.drop(columns=['target'])  # Ensure 'target' is not in features
        y_pred = model.predict(X)

        # Add predictions to the dataset
        df['Prediction'] = y_pred

        # Display a summary of predictions
        st.write("### Prediction Summary:")
        st.write(df['Prediction'].value_counts())

        # Display predictions
        st.write("### Predictions:")
        st.write(df)

    except KeyError as e:
        st.error(f"Missing column: {e}. Please ensure your dataset includes all necessary feature columns.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.write("Please upload a dataset to see predictions.")
