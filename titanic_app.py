import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

# ==================== TITLE ====================
st.title("Titanic Survival Prediction - Group 5")
st.markdown("**Binary Classification:** Predict passenger survival (0 = Died, 1 = Survived)")

# ==================== LOAD & TRAIN MODEL (Background) ====================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

@st.cache_resource
def train_models():
    df = load_data()
    
    # Preprocessing
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
    
    return lr, rf, le_sex, le_emb, X_test, y_test

# Train models once
lr, rf, le_sex, le_emb, X_test, y_test = train_models()

# ==================== LIVE PREDICTION DEMO (ONLY SECTION) ====================
st.header("Live Survival Prediction")

st.markdown("Enter passenger details below and click **Predict Survival** to see the result.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Passenger Details")
    
    pclass = st.selectbox(
        "Passenger Class", 
        [1, 2, 3],
        format_func=lambda x: {1: "1st Class (Upper)", 2: "2nd Class (Middle)", 3: "3rd Class (Lower)"}[x],
        help="1st class had better lifeboat access"
    )
    
    sex = st.radio(
        "Sex",
        ["Female", "Male"],
        horizontal=True,
        help="Women and children were prioritized"
    )
    
    age = st.slider(
        "Age (years)",
        min_value=0,
        max_value=100,
        value=30,
        step=1,
        help="Children (under 15) had higher survival rates"
    )

with col2:
    st.subheader("Ticket Information")
    
    fare = st.number_input(
        "Fare ($)",
        min_value=0.0,
        max_value=500.0,
        value=50.0,
        step=10.0,
        help="Higher fare generally means higher class"
    )
    
    st.markdown("---")
    st.caption("Tip: First-class women and children have the highest survival chance.")
    st.caption("Tip: Third-class men have the lowest survival chance.")

st.markdown("---")

# Predict button
if st.button("Predict Survival", type="primary", use_container_width=True):
    # Convert inputs
    sex_encoded = 1 if sex == "Female" else 0
    
    # Create input dataframe with default values for removed features
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'SibSp': [0],      # Default - traveling alone
        'Parch': [0],      # Default - no parents/children
        'Fare': [fare],
        'Embarked': [0]    # Default - Southampton (most common)
    })
    
    # Get predictions
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
    
    # Show passenger summary
    st.markdown("---")
    st.markdown("### Passenger Summary")
    
    survival_chance_rf = prob_rf * 100
    
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    with col_sum1:
        st.metric("Class", f"{pclass}{'st' if pclass==1 else 'nd' if pclass==2 else 'rd'} Class")
    with col_sum2:
        st.metric("Sex", sex)
    with col_sum3:
        st.metric("Age", f"{age} years")
    
    # Survival chance bar
    st.markdown("**Survival Probability (Random Forest):**")
    st.progress(int(survival_chance_rf))
    st.caption(f"{survival_chance_rf:.1f}% chance of survival")

