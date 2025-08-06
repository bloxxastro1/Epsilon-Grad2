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
st.title("ML Classifier Evaluation App")

# Correct GitHub raw CSV URL
github_url = "https://raw.githubusercontent.com/bloxxastro1/Epsilon-Grad2/main/dirty_cafe_sales.csv"

@st.cache_data
def load_data(url):
    return pd.read_csv(url, on_bad_lines='skip')

df = load_data(github_url)

# Cleaning and preprocessing
df.drop_duplicates(inplace=True)
for col in df.columns:
    df[col] = df[col].replace(["UNKNOWN", "ERROR", "nan"], np.nan)
df.drop(["Transaction ID", "Total Spent"], axis=1, inplace=True)
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], errors='coerce')
df['Year'] = df['Transaction Date'].dt.year
df['Month'] = df['Transaction Date'].dt.month
df['Day'] = df['Transaction Date'].dt.day
df = df.dropna(subset=['Item'])
df['Price Per Unit'] = df['Price Per Unit'].astype(float)
df['Quantity'] = df['Quantity'].astype(float)

# Standardize known item prices
df.loc[df["Item"] == "coffee", "Price Per Unit"] = 2
df.loc[df["Item"] == "Tea", "Price Per Unit"] = 1.5
df.loc[df["Item"] == "Sandwich", "Price Per Unit"] = 4
df.loc[df["Item"] == "Cookie", "Price Per Unit"] = 1
df.loc[df["Item"] == "Juice", "Price Per Unit"] = 3
df.loc[df["Item"] == "Smoothie", "Price Per Unit"] = 4
df.loc[df["Item"] == "Salad ", "Price Per Unit"] = 4
df.loc[df["Item"] == "cake", "Price Per Unit"] = 3

df["Total Spent"] = df["Quantity"] * df["Price Per Unit"]
df = df.drop(columns="Transaction Date")
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])
# Impute missing values
imputer = KNNImputer(n_neighbors=3)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
st.write("### Dataset Preview:")
st.dataframe(df.head())
# Select target column
# Only allow object or integer-type columns (assumed categorical) as targets
possible_targets = df.select_dtypes(include=['object', 'category', 'int']).columns
if len(possible_targets) == 0:
    st.error("No valid classification targets found in this dataset.")
    st.stop()
target_col = st.selectbox("Select the target column (must be categorical)", possible_targets)
# Features and Target
X = df.drop(target_col, axis=1)
# Encode target if it's categorical
if df[target_col].dtype == 'object':
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[target_col])
else:
    y = df[target_col]

# Train-test split
test_size = st.slider("Test set size (%)", 10, 50, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

# Apply SMOTE

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Model selection
classifier_name = st.selectbox("Choose a classifier", (
    "Logistic Regression", "K-Nearest Neighbors", "Decision Tree",
    "Random Forest", "SVM", "Naive Bayes", "XGBoost"
))

# Initialize classifier
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

# Train and predict
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluation
st.write("### Evaluation Metrics:")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
st.write(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.4f}")
st.write(f"Recall: {recall_score(y_test, y_pred, average='weighted'):.4f}")
st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))








