import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

st.title("Titanic Survival Prediction - Group 5")
st.markdown("**Binary Classification:** Predict passenger survival (0 = Died, 1 = Survived)")

# ==================== LOAD & TRAIN MODEL ====================
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
    
    # Note: Fare is kept for model training but not shown in UI
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr = LogisticRegression(max_iter=200)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    
    # Get average fare by class for default values
    avg_fare = df.groupby('Pclass')['Fare'].mean().to_dict()
    
    return lr, rf, le_sex, le_emb, avg_fare

lr, rf, le_sex, le_emb, avg_fare = train_models()

# ==================== LIVE PREDICTION DEMO ====================
st.header("Live Survival Prediction")

st.markdown("Enter passenger details below and click **Predict Survival** to see the result.")

st.markdown("---")

# Three simple inputs - all that matters!
col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class", 
        [1, 2, 3],
        format_func=lambda x: {1: "1st Class (Upper)", 2: "2nd Class (Middle)", 3: "3rd Class (Lower)"}[x],
        help="1st class had priority access to lifeboats"
    )

with col2:
    sex = st.radio(
        "Sex",
        ["Female", "Male"],
        horizontal=True,
        help="Women and children were evacuated first"
    )

with col3:
    age = st.slider(
        "Age (years)",
        min_value=0,
        max_value=100,
        value=30,
        step=1,
        help="Children under 15 were given priority"
    )

st.markdown("---")

# Show survival factors explanation
with st.expander("What affects survival?"):
    st.markdown("""
    | Factor | Why It Matters |
    |--------|----------------|
    | **Passenger Class** | 1st class had better lifeboat access (62% survived vs 26% in 3rd class) |
    | **Sex** | Women were evacuated first (74% survived vs 19% of men) |
    | **Age** | Children were prioritized (54% of children survived vs 38% of adults) |
    
    *These three factors were the strongest predictors of survival on the Titanic.*
    """)

# Predict button
if st.button("Predict Survival", type="primary", use_container_width=True):
    # Convert inputs
    sex_encoded = 1 if sex == "Female" else 0
    
    # Use realistic default values based on class
    # Fare is determined by class (not user input)
    realistic_fare = avg_fare[pclass]
    
    # Default values for other features (set to most common/normal values)
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'SibSp': [0],      # Default: traveling alone (most common)
        'Parch': [0],      # Default: no children (most common)  
        'Fare': [realistic_fare],  # Automatically set by class
        'Embarked': [0]    # Default: Southampton (most common)
    })
    
    # Get predictions
    pred_rf = rf.predict(input_data)[0]
    prob_rf = rf.predict_proba(input_data)[0][1]
    
    # Display result prominently
    st.markdown("---")
    st.subheader("Prediction Result")
    
    # Big result box
    result_col1, result_col2 = st.columns([2, 1])
    
    with result_col1:
        if pred_rf == 1:
            st.success("## SURVIVED")
            st.caption(f"Confidence: {prob_rf:.1%}")
        else:
            st.error("## DID NOT SURVIVE")
            st.caption(f"Confidence: {prob_rf:.1%}")
    
    with result_col2:
        # Survival chance meter
        survival_chance = prob_rf * 100
        st.metric("Survival Probability", f"{survival_chance:.1f}%")
        st.progress(int(survival_chance))
    
    # Why this prediction?
    st.markdown("---")
    st.markdown("### Why this prediction?")
    
    reasons = []
    if pclass == 1:
        reasons.append("✓ First-class passengers had priority access to lifeboats")
    elif pclass == 3:
        reasons.append("✗ Third-class passengers had limited lifeboat access")
    
    if sex == "Female":
        reasons.append("✓ Women were evacuated first")
    else:
        reasons.append("✗ Men had lower priority during evacuation")
    
    if age < 15:
        reasons.append("✓ Children were given priority")
    elif age > 60:
        reasons.append("✗ Elderly passengers had lower survival rates")
    
    for reason in reasons:
        st.markdown(reason)
