import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px


st.title("Customer Churn Prediction Project")

@st.cache_data
def load_data():
    data = pd.read_csv("Telco-Customer-Churn.csv")
    return data

data = load_data()
st.subheader("Raw Data")
st.dataframe(data.head())


data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.fillna(0, inplace=True)

# Encode categorical variables
categorical_cols = ['gender','Partner','Dependents','PhoneService','MultipleLines',
                    'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
                    'TechSupport','StreamingTV','StreamingMovies','Contract',
                    'PaperlessBilling','PaymentMethod']

data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Encode target variable
data['Churn'] = data['Churn'].map({'Yes':1,'No':0})


X = data.drop(['customerID','Churn'], axis=1)
y = data['Churn']

# train - test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]


st.subheader("Model Evaluation")
st.text(classification_report(y_test, y_pred))
st.text(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.3f}")

# Feature Importance Plot
st.subheader("Top 10 Important Features")
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = feat_importances.nlargest(10)
fig, ax = plt.subplots()
top_features.plot(kind='barh', ax=ax)
st.pyplot(fig)


st.subheader("Churn Visualization")

# Reconstruct ContractType and PaymentMethodType for plotting
data['ContractType'] = np.where(data.get('Contract_One year', 0)==1, 'One year',
                        np.where(data.get('Contract_Two year', 0)==1, 'Two year', 'Month-to-month'))

data['PaymentMethodType'] = np.where(data.get('PaymentMethod_Credit card (automatic)', 0)==1, 'Credit Card',
                            np.where(data.get('PaymentMethod_Electronic check', 0)==1, 'Electronic Check',
                            np.where(data.get('PaymentMethod_Mailed check', 0)==1, 'Mailed Check', 'Bank Transfer')))

fig2 = px.bar(data, x='ContractType', y='Churn', color='PaymentMethodType', barmode='group')
st.plotly_chart(fig2)

# -------------------------------
st.subheader("Predict Churn for a New Customer")

# Input fields for key features
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=840.0)

# Contract input
contract_input = st.selectbox("Contract Type", ['Month-to-month','One year','Two year'])
# Paperless billing
paperless_input = st.selectbox("Paperless Billing?", ['No','Yes'])
# Internet Service
internet_input = st.selectbox("Internet Service", ['DSL','Fiber optic','No'])

if st.button("Predict Churn"):
    # Create feature array with zeros
    input_data = np.zeros(X.shape[1])

    # Map numeric inputs
    input_data[X.columns.get_loc('tenure')] = tenure
    input_data[X.columns.get_loc('MonthlyCharges')] = monthly_charges
    input_data[X.columns.get_loc('TotalCharges')] = total_charges

    # Map contract type
    if contract_input == 'One year' and 'Contract_One year' in X.columns:
        input_data[X.columns.get_loc('Contract_One year')] = 1
    if contract_input == 'Two year' and 'Contract_Two year' in X.columns:
        input_data[X.columns.get_loc('Contract_Two year')] = 1
    # Month-to-month is implied when both are 0

    # Map paperless billing
    if paperless_input=='Yes' and 'PaperlessBilling_Yes' in X.columns:
        input_data[X.columns.get_loc('PaperlessBilling_Yes')] = 1

    # Map internet service
    if internet_input=='Fiber optic' and 'InternetService_Fiber optic' in X.columns:
        input_data[X.columns.get_loc('InternetService_Fiber optic')] = 1
    if internet_input=='No' and 'InternetService_No' in X.columns:
        input_data[X.columns.get_loc('InternetService_No')] = 1
    # DSL is implied when both are 0

    # Predict
    prediction = model.predict([input_data])[0]
    probability = model.predict_proba([input_data])[0][1]

    st.write(f"Predicted Churn: {'Yes' if prediction==1 else 'No'}")
    st.write(f"Probability of Churn: {probability:.2f}")
