import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

st.title("🚢 Titanic Survival Prediction - Group 5")
st.markdown("**Binary Classification:** Predict passenger survival (0 = Died, 1 = Survived)")

# ==================== LOAD & TRAIN MODEL ====================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

@st.cache_resource
def train_models():
    df = load_data()
    
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    le_sex = LabelEncoder()
    le_emb = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['Embarked'] = le_emb.fit_transform(df['Embarked'])
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr = LogisticRegression(max_iter=200)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    
    avg_fare = df.groupby('Pclass')['Fare'].mean().to_dict()
    
    return lr, rf, le_sex, le_emb, avg_fare

lr, rf, le_sex, le_emb, avg_fare = train_models()

# ==================== LIVE PREDICTION ====================
st.header("Live Survival Prediction")

st.markdown("Enter passenger details below and click **Predict Survival** to see the result.")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3],
        format_func=lambda x: {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}[x]
    )

with col2:
    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)

if st.button("Predict Survival", type="primary", use_container_width=True):
    sex_encoded = 1 if sex == "Female" else 0
    realistic_fare = avg_fare[pclass]
    
    # Age is set to 30 (default) - NOT shown to user
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [30],        # Fixed default value
        'SibSp': [0],
        'Parch': [0],
        'Fare': [realistic_fare],
        'Embarked': [0]
    })
    
    pred_lr = lr.predict(input_data)[0]
    prob_lr = lr.predict_proba(input_data)[0][1]
    pred_rf = rf.predict(input_data)[0]
    prob_rf = rf.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    st.subheader("Prediction Results")
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.markdown("### Logistic Regression")
        if pred_lr == 1:
            st.success("SURVIVED")
        else:
            st.error("DID NOT SURVIVE")
        st.caption(f"Confidence: {prob_lr:.1%}")
    
    with col_res2:
        st.markdown("### Random Forest")
        if pred_rf == 1:
            st.success("SURVIVED")
        else:
            st.error("DID NOT SURVIVE")
        st.caption(f"Confidence: {prob_rf:.1%}")
    
    # Show why
    st.markdown("---")
    st.markdown("### Why this prediction?")
    
    if pclass == 1:
        st.markdown("✓ First-class passengers had priority access to lifeboats")
    elif pclass == 3:
        st.markdown("✗ Third-class passengers had limited lifeboat access")
    
    if sex == "Female":
        st.markdown("✓ Women were evacuated first")
    else:
        st.markdown("✗ Men had lower priority during evacuation")
    
    # Survival chance
    survival_chance = prob_rf * 100
    st.markdown(f"**Survival Probability:** {survival_chance:.1f}%")
    st.progress(int(survival_chance))

