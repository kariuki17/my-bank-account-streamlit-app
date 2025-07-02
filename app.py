
import streamlit as st
import pandas as pd
import joblib

# Load your trained model and encoder
model = joblib.load('bank_account_model.pkl')
encoder = joblib.load('encoder.pkl')  # If you used one-hot or label encoder

st.set_page_config(page_title="Financial Inclusion Prediction", layout="centered")

st.title("🌍 Financial Inclusion in Africa")
st.markdown("Predict whether a person has a **bank account** based on their demographics.")

# User input form
with st.form("prediction_form"):
    country = st.selectbox("Country", ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
    year = st.selectbox("Year", [2016, 2017, 2018])
    location_type = st.selectbox("Location Type", ['Urban', 'Rural'])
    cellphone_access = st.selectbox("Cellphone Access", ['Yes', 'No'])
    household_size = st.slider("Household Size", 1, 20, 3)
    age_of_respondent = st.slider("Age", 16, 100, 30)
    gender_of_respondent = st.selectbox("Gender", ['Male', 'Female'])
    relationship_with_head = st.selectbox("Relationship with Head", ['Head of Household', 'Spouse', 'Child', 'Other relative', 'Other non-relatives'])
    education_level = st.selectbox("Education Level", ['No formal education', 'Primary education', 'Secondary education', 'Tertiary education', 'Vocational/Specialised training'])
    job_type = st.selectbox("Job Type", ['Self employed', 'Government Dependent', 'Formally employed Government', 'Formally employed Private',
                                         'Informally employed', 'Farming and Fishing', 'Remittance Dependent', 'Other Income', 'No Income'])

    submit = st.form_submit_button("Predict")

# Make prediction
if submit:
    try:
        input_dict = {
            'country': country,
            'year': year,
            'location_type': location_type,
            'cellphone_access': cellphone_access,
            'household_size': household_size,
            'age_of_respondent': age_of_respondent,
            'gender_of_respondent': gender_of_respondent,
            'relationship_with_head': relationship_with_head,
            'education_level': education_level,
            'job_type': job_type
        }

        input_df = pd.DataFrame([input_dict])
        
        # Apply encoder if needed
        input_df_encoded = encoder.transform(input_df)

        prediction = model.predict(input_df_encoded)

        st.success(f"✅ Prediction: The person is likely to **{'have' if prediction[0] == 'Yes' else 'not have'} a bank account.**")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    