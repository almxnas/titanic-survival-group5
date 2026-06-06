import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")
st.title("🚢 Titanic Survival Prediction - Group 5")
st.markdown("**Binary Classification Problem** - Predicting passenger survival (0 = Died, 1 = Survived)")

# ==================== 1. DATASET ====================
st.header("1. Dataset")
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

df = load_data()
st.dataframe(df.head(10))   # ← Nice table as teacher wants

st.subheader("Key Features")
feature_table = pd.DataFrame({
    "Feature": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
    "Description": ["Ticket class (1/2/3)", "Gender", "Age in years", "Siblings/Spouse", 
                    "Parents/Children", "Ticket fare", "Port of Embarkation"]
})
st.table(feature_table)

# ==================== 2. PREPROCESSING ====================
st.header("2. Data Preprocessing")
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

# ==================== 3. MODEL TRAINING ====================
st.header("3. Model Training")
X = processed_df.drop('Survived', axis=1)
y = processed_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg = LogisticRegression(max_iter=200)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

logreg.fit(X_train, y_train)
rf.fit(X_train, y_train)

# ==================== 4. EVALUATION ====================
st.header("4. Model Evaluation")
y_pred_log = logreg.predict(X_test)
y_pred_rf = rf.predict(X_test)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Logistic Regression")
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_log):.1%}")
    st.metric("Precision", f"{precision_score(y_test, y_pred_log):.1%}")
    st.metric("Recall", f"{recall_score(y_test, y_pred_log):.1%}")
    st.metric("F1-Score", f"{f1_score(y_test, y_pred_log):.1%}")

with col2:
    st.subheader("Random Forest")
    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_rf):.1%}")
    st.metric("Precision", f"{precision_score(y_test, y_pred_rf):.1%}")
    st.metric("Recall", f"{recall_score(y_test, y_pred_rf):.1%}")
    st.metric("F1-Score", f"{f1_score(y_test, y_pred_rf):.1%}")

# Confusion Matrix
st.subheader("Confusion Matrix (Random Forest)")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# ==================== 5. LIVE PREDICTION ====================
st.header("5. Live Prediction (Interactive Demo)")
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Pclass", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 30)

with col2:
    sibsp = st.slider("SibSp", 0, 8, 0)
    parch = st.slider("Parch", 0, 6, 0)
    fare = st.slider("Fare", 0.0, 520.0, 50.0)
    embarked = st.selectbox("Embarked", ["S", "C", "Q"])

if st.button("🚀 Predict Survival"):
    input_data = pd.DataFrame({
        'Pclass': [pclass], 'Sex': le_sex.transform([sex]), 'Age': [age],
        'SibSp': [sibsp], 'Parch': [parch], 'Fare': [fare],
        'Embarked': le_emb.transform([embarked])
    })
    
    pred_log = logreg.predict(input_data)[0]
    prob_log = logreg.predict_proba(input_data)[0][1]
    pred_rf = rf.predict(input_data)[0]
    prob_rf = rf.predict_proba(input_data)[0][1]
    
    st.success(f"**Logistic Regression**: {'✅ Survived' if pred_log == 1 else '❌ Did not survive'} ({prob_log:.1%})")
    st.success(f"**Random Forest**: {'✅ Survived' if pred_rf == 1 else '❌ Did not survive'} ({prob_rf:.1%})")