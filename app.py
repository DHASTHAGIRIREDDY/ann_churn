import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle


# =========================
# Load trained model
# =========================

model = tf.keras.models.load_model("churn_model.h5")


# =========================
# Load encoders and scaler
# =========================

# Geography -> OneHotEncoder
with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

# Gender -> LabelEncoder
with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

# StandardScaler
with open("sscaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# =========================
# Streamlit UI
# =========================

st.title("Customer Churn Prediction")

st.write("Enter the customer details below:")


# =========================
# User Input
# =========================

age = st.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

geo = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

credit_score = st.slider(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=600
)

tenure = st.slider(
    "Tenure",
    min_value=0,
    max_value=10,
    value=5
)

num_of_products = st.slider(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=2
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)


# =========================
# Encoding
# =========================

# Gender -> Label Encoding
gender_encoded = label_encoder_gender.transform([gender])[0]


# Geography -> One Hot Encoding
geo_encoded = onehot_encoder_geo.transform([[geo]]).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)


# =========================
# Create input DataFrame
# =========================

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [gender_encoded],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})


# Add Geography one-hot columns
input_data = pd.concat(
    [input_data, geo_encoded_df],
    axis=1
)


# =========================
# Arrange columns in the
# same order used during training
# =========================

if hasattr(scaler, "feature_names_in_"):
    input_data = input_data[scaler.feature_names_in_]


# =========================
# Scale input
# =========================

input_scaled = scaler.transform(input_data)


# =========================
# Prediction
# =========================

prediction = model.predict(
    input_scaled,
    verbose=0
)

prediction_probability = prediction[0][0]


# =========================
# Display Result
# =========================

st.subheader("Prediction")

if prediction_probability > 0.5:

    st.error(
        f"The customer is likely to churn "
        f"with a probability of {prediction_probability:.2%}"
    )

else:

    st.success(
        f"The customer is not likely to churn "
        f"with a probability of {(1 - prediction_probability):.2%}"
    )