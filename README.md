# Customer Churn Prediction System

An end-to-end machine learning project to predict customer churn using the [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn).  
Built with **Python, scikit-learn, pandas, Streamlit, and Plotly**, the project includes data cleaning, model training, evaluation, and an interactive dashboard for visualization and real-time predictions.  

---

##  Features
- **Data Preprocessing**: Missing value handling, categorical encoding, feature engineering.  
- **Machine Learning**: Random Forest Classifier with ROC-AUC > 0.80.  
- **Insights**: Feature importance analysis to identify churn drivers.  
- **Interactive Dashboard**: Built with Streamlit + Plotly for churn visualization.  
- **Real-time Prediction Tool**: Input customer details and get churn probability instantly.  

---

## Demo
- **Live App**: [Streamlit Demo](https://rv99-cust-churn-proj.streamlit.app)  
- **Code Repo**: [GitHub Repo](https://github.com/rohithvadla99/customer-churn-prediction)  


---

##  Tech Stack
- **Languages & Libraries**: Python, pandas, numpy, scikit-learn, matplotlib, plotly  
- **Modeling**: Random Forest, Feature Engineering, ROC-AUC evaluation  
- **Visualization & UI**: Streamlit, Plotly  
- **Deployment**: Streamlit Cloud  

---

##  Project Structure
```text
customer-churn-prediction/
│-- churn_project.py # Main Streamlit app
│-- Telco-Customer-Churn.csv # Dataset
│-- requirements.txt # Dependencies
│-- README.md # Project documentation

```

##  Quick Start

### 1. Clone Repository
- git clone https://github.com/rohithvadla99/customer-churn-prediction.git
- cd customer-churn-prediction

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run App
streamlit run churn_project.py

The dashboard will launch at http://localhost:8501/
