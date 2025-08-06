import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Title
st.title("🧠 ML Classifier Evaluation App")

# GitHub raw CSV URL
github_url = "https://raw.githubusercontent.com/bloxxastro1/Epsilon-Grad2/main/dirty_cafe_sales.csv"

@st.cache_data
def load_data(url):
    return pd.read_csv(url, on_bad_lines='skip')

# Load data
df = load_data(github_url)

# Basic cleaning
df.drop_duplicates(inplace=True)
df.replace(["UNKNOWN", "ERROR", "nan"], np.nan, inplace=True)
df.drop(["Transaction ID", "Total Spent"], axis=1, inplace=True)
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
df['Year'] = df['Transaction Date'].dt.year
df['Month'] = df['Transaction Date'].dt.month
df['Day'] = df['Transaction Date'].dt.day
df.dropna(subset=['Item'], inplace=True)

# Convert columns to float
df['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')

# Fill in known prices
df.loc[df["Item"] == "coffee", "Price Per Unit"] = 2
df.loc[df["Item"] == "Tea", "Price Per Unit"] = 1.5
df.loc[df["Item"] == "Sandwich", "Price Per Unit"] = 4
df.loc[df["Item"] == "Cookie", "Price Per Unit"] = 1
df.loc[df["Item"] == "Juice", "Price Per Unit"] = 3
df.loc[df["Item"] == "Smoothie", "Price Per Unit"] = 4
df.loc[df["Item"] == "Salad ", "Price Per Unit"] = 4
df.loc[df["Item"] == "cake", "Price Per Unit"] = 3

# Total spent
df["Total Spent"] = df["Quantity"] * df["Price Per Unit"]

# Drop date
df.drop(columns="Transaction Date", inplace=True)

# Label encode target
target_col = "Item"
le_target = LabelEncoder()
df[target_col] = le_target.fit_transform(df[target_col].astype(str))

# Features & target
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode other object columns
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Scaling
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Imputation
imputer = KNNImputer(n_neighbors=3)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classifier selection
classifier_name = st.selectbox("Choose a classifier", (
    "Logistic Regression", "K-Nearest Neighbors", "Decision Tree",
    "Random Forest", "SVM", "Naive Bayes", "XGBoost"
))

def get_classifier(name):
    if name == "Logistic Regression":
        return LogisticRegression()
    elif name == "K-Nearest Neighbors":
        return KNeighborsClassifier()
    elif name == "Decision Tree":
        return DecisionTreeClassifier()
    elif name == "Random Forest":
        return RandomForestClassifier()
    elif name == "SVM":
        return SVC()
    elif name == "Naive Bayes":
        return GaussianNB()
    elif name == "XGBoost":
        return XGBClassifier()

clf = get_classifier(classifier_name)

# Fit and predict
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
st.markdown("### 📊 Evaluation Metrics:")
st.write(f"**Accuracy**: {accuracy_score(y_test, y_pred):.4f}")
st.write(f"**Precision**: {precision_score(y_test, y_pred, average='weighted'):.4f}")
st.write(f"**Recall**: {recall_score(y_test, y_pred, average='weighted'):.4f}")
st.write(f"**F1 Score**: {f1_score(y_test, y_pred, average='weighted'):.4f}")

st.markdown("### 🧾 Classification Report:")
st.text(classification_report(y_test, y_pred, zero_division=0))

