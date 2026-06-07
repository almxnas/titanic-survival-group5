import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")
st.title("🚢 Titanic Survival Prediction - Group 5")
st.markdown("**Binary Classification Problem** - Predicting passenger survival (0 = Died, 1 = Survived)")

# Load Data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

df = load_data()

st.header("1. Dataset")
st.dataframe(df.head(10))

st.subheader("Key Features")
feature_table = pd.DataFrame({
    "Feature": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
    "Description": ["Ticket class", "Gender", "Age", "Siblings/Spouse", "Parents/Children", "Ticket Fare", "Embarkation Port"]
})
st.table(feature_table)

# Preprocessing
def preprocess(df):
    df = df.copy()
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    le_sex = LabelEncoder()
    le_emb = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['Embarked'] = le_emb.fit_transform(df['Embarked'])
    return df, le_sex, le_emb

processed_df, le_sex, le_emb = preprocess(df)

# Training
X = processed_df.drop('Survived', axis=1)
y = processed_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg = LogisticRegression(max_iter=200)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Evaluation
st.header("4. Model Evaluation")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Logistic Regression")
    y_pred = logreg.predict(X_test)
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
    st.metric("Precision", f"{precision_score(y_test, y_pred):.1%}")
    st.metric("Recall", f"{recall_score(y_test, y_pred):.1%}")
    st.metric("F1-Score", f"{f1_score(y_test, y_pred):.1%}")

with col2:
    st.subheader("Random Forest")
    y_pred_rf = rf.predict(X_test)
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_rf):.1%}")
    st.metric("Precision", f"{precision_score(y_test, y_pred_rf):.1%}")
    st.metric("Recall", f"{recall_score(y_test, y_pred_rf):.1%}")
    st.metric("F1-Score", f"{f1_score(y_test, y_pred_rf):.1%}")

# Live Prediction
st.header("5. Live Prediction")
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Pclass", [1,2,3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 30)
with col2:
    sibsp = st.slider("SibSp", 0, 8, 0)
    parch = st.slider("Parch", 0, 6, 0)
    fare = st.slider("Fare", 0.0, 520.0, 50.0)
    embarked = st.selectbox("Embarked", ["S", "C", "Q"])

if st.button("Predict Survival"):
    input_data = pd.DataFrame({
        'Pclass': [pclass], 'Sex': le_sex.transform([sex]), 'Age': [age],
        'SibSp': [sibsp], 'Parch': [parch], 'Fare': [fare],
        'Embarked': le_emb.transform([embarked])
    })
    
    pred_log = logreg.predict(input_data)[0]
    prob_log = logreg.predict_proba(input_data)[0][1]
    pred_rf = rf.predict(input_data)[0]
    prob_rf = rf.predict_proba(input_data)[0][1]
    
    st.success(f"**Logistic Regression**: {'Survived' if pred_log == 1 else 'Did not survive'} ({prob_log:.1%})")
    st.success(f"**Random Forest**: {'Survived' if pred_rf == 1 else 'Did not survive'} ({prob_rf:.1%})")
