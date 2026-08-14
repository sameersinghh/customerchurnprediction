import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Load the trained model
model = tf.keras.models.load_model('churn_model.h5')

# Load the encoders and scaler
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit app
st.title('Customer Churn Prediction')

# User inputs
geography = st.selectbox(
    'Geography',
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    'Gender',
    label_encoder_gender.classes_
)

age = st.number_input(
    'Age',
    min_value=18,
    max_value=100,
    value=30
)

balance = st.number_input(
    'Balance',
    min_value=0.0,
    value=1000.0
)

credit_score = st.number_input(
    'Credit Score',
    min_value=300,
    max_value=850,
    value=600
)

estimated_salary = st.number_input(
    'Estimated Salary',
    min_value=0.0,
    value=50000.0
)

tenure = st.number_input(
    'Tenure',
    min_value=0,
    max_value=10,
    value=5
)

num_of_products = st.number_input(
    'Number of Products',
    min_value=1,
    max_value=4,
    value=1
)

has_cr_card = st.selectbox(
    'Has Credit Card',
    ['Yes', 'No']
)

is_active_member = st.selectbox(
    'Is Active Member',
    ['Yes', 'No']
)


# --------------------------------------------------
# Prepare input data
# --------------------------------------------------

input_data = pd.DataFrame({
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Balance': [balance],
    'CreditScore': [credit_score],
    'EstimatedSalary': [estimated_salary],
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
    'IsActiveMember': [1 if is_active_member == 'Yes' else 0]
})


# --------------------------------------------------
# Encode Geography
# --------------------------------------------------

geo_encoded = onehot_encoder_geo.transform(
    input_data[['Geography']]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)


# --------------------------------------------------
# Encode Gender
# --------------------------------------------------

input_data['Gender'] = label_encoder_gender.transform(
    input_data['Gender']
)


# --------------------------------------------------
# Combine features
# --------------------------------------------------

input_data = pd.concat(
    [
        input_data.drop('Geography', axis=1),
        geo_encoded_df
    ],
    axis=1
)


# --------------------------------------------------
# Match training feature names and order
# --------------------------------------------------

input_data = input_data.reindex(
    columns=scaler.feature_names_in_,
    fill_value=0
)


# --------------------------------------------------
# Scale input
# --------------------------------------------------

input_data_scaled = scaler.transform(input_data)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button('Predict Churn'):

    prediction = model.predict(input_data_scaled)

    prediction_result = prediction[0][0]

    if prediction_result > 0.5:
        st.error(
            f"The customer is likely to churn. "
            f"Probability: {prediction_result:.2%}"
        )
    else:
        st.success(
            f"The customer is not likely to churn. "
            f"Probability of churn: {prediction_result:.2%}"
        )