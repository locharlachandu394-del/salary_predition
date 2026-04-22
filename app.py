import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(layout="wide")

# --- Load Model and Preprocessors ---
@st.cache_resource
def load_model():
    with open('RandomForest.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_data_and_encoders():
    # Load the original CSV to get true categorical values and re-create encoders
    # Make sure 'Salary_Data.csv' is available in the deployment environment or adjust path
    original_df = pd.read_csv("Salary_Data.csv")

    # Ensure consistency with how NaNs were handled before encoding in the notebook
    categorical_cols = ['Gender', 'Education Level', 'Job Title']
    for col in categorical_cols:
        original_df[col] = original_df[col].fillna('Unknown')

    label_encoders = {}
    for col in categorical_cols:
        le_temp = LabelEncoder()
        # Fit on unique values after NaN imputation
        # Use .astype(str) to handle potential mixed types safely
        le_temp.fit(original_df[col].astype(str).unique())
        label_encoders[col] = le_temp

    # Get the unique original categories for display in selectbox
    gender_options = original_df['Gender'].unique().tolist()
    education_options = original_df['Education Level'].unique().tolist()
    job_options = original_df['Job Title'].unique().tolist()

    return label_encoders, gender_options, education_options, job_options

# Load assets
model = load_model()
label_encoders, gender_options, education_options, job_options = load_data_and_encoders()

# Get feature columns in the order the model expects
# This order is crucial. Based on df.drop('Salary', axis=1) in notebook, it was:
feature_columns = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']

# --- Streamlit App ---
st.title("Salary Prediction Dashboard")
st.write("Enter the details below to predict the salary.")

with st.sidebar:
    st.header("Input Features")
    age = st.slider("Age", min_value=18, max_value=70, value=30)
    gender_raw = st.selectbox("Gender", options=gender_options)
    education_raw = st.selectbox("Education Level", options=education_options)
    job_title_raw = st.selectbox("Job Title", options=job_options)
    years_of_experience = st.slider("Years of Experience", min_value=0, max_value=40, value=5)

    # Display selected categorical values
    st.markdown("--- Chosen Categorical Values ---")
    st.write(f"**Gender**: {gender_raw}")
    st.write(f"**Education Level**: {education_raw}")
    st.write(f"**Job Title**: {job_title_raw}")

    predict_button = st.button("Predict Salary")

if predict_button:
    # Preprocess inputs
    processed_gender = label_encoders['Gender'].transform([gender_raw])[0]
    processed_education = label_encoders['Education Level'].transform([education_raw])[0]
    processed_job_title = label_encoders['Job Title'].transform([job_title_raw])[0]

    # Create a DataFrame for prediction, ensuring column order matches training data
    input_data = pd.DataFrame([[age, processed_gender, processed_education, processed_job_title, years_of_experience]],
                                columns=feature_columns)

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f"The predicted salary is: ${prediction:,.2f}")
else:
    st.info("Adjust the input features in the sidebar and click 'Predict Salary'.")


%%writefile requirements.txt
streamlit
pandas
scikit-learn
numpy
matplotlib
seaborn
