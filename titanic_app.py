import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

# ==================== TITLE ====================
st.title("🚢 Titanic Survival Prediction - Group 5")
st.markdown("**🎯 Binary Classification:** Predict passenger survival (0 = Died, 1 = Survived)")

st.markdown("---")

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

# ==================== LIVE PREDICTION DEMO ====================
st.header("🎯 Live Survival Prediction")

st.markdown("📝 Enter passenger details below and click **Predict Survival** to see the result.")

st.markdown("---")

# TWO columns only (Class and Sex) - NO AGE
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🎫 Passenger Class")
    pclass = st.selectbox(
        "Select Class", 
        [1, 2, 3],
        format_func=lambda x: {1: "👑 1st Class (Upper)", 2: "📘 2nd Class (Middle)", 3: "⚓ 3rd Class (Lower)"}[x],
        help="1st class had priority access to lifeboats",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("#### 👤 Sex")
    sex = st.radio(
        "Select Sex",
        ["👩 Female", "👨 Male"],
        horizontal=True,
        help="Women were evacuated first",
        label_visibility="collapsed"
    )
    # Clean the sex value for processing
    sex_clean = "Female" if "Female" in sex else "Male"

st.markdown("---")

# Expandable info section
with st.expander("📊 What affects survival? (Click to expand)"):
    st.markdown("""
    | 👑 Factor | 📈 Why It Matters |
    |-----------|-------------------|
    | **🎫 Passenger Class** | 1st class: 62% survived &nbsp;&nbsp;&nbsp; 3rd class: 26% survived |
    | **👤 Sex** | Women: 74% survived &nbsp;&nbsp;&nbsp; Men: 19% survived |
    
    ---
    
    💡 **Class and Sex were the TWO strongest predictors of survival on the Titanic.**
    """)

# Predict button
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_clicked = st.button("🚀 PREDICT SURVIVAL", type="primary", use_container_width=True)

if predict_clicked:
    # Convert inputs
    sex_encoded = 1 if sex_clean == "Female" else 0
    realistic_fare = avg_fare[pclass]
    
    # Use default values for other features
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [30],           # Default age (doesn't affect much)
        'SibSp': [0],
        'Parch': [0],
        'Fare': [realistic_fare],
        'Embarked': [0]
    })
    
    # Get predictions from BOTH models
    pred_lr = lr.predict(input_data)[0]
    prob_lr = lr.predict_proba(input_data)[0][1]
    pred_rf = rf.predict(input_data)[0]
    prob_rf = rf.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Two columns for both models
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.markdown("#### 🤖 Logistic Regression")
        if pred_lr == 1:
            st.success("✅ **SURVIVED**")
        else:
            st.error("❌ **DID NOT SURVIVE**")
        st.caption(f"📈 Confidence: {prob_lr:.1%}")
    
    with col_res2:
        st.markdown("#### 🌲 Random Forest")
        if pred_rf == 1:
            st.success("✅ **SURVIVED**")
        else:
            st.error("❌ **DID NOT SURVIVE**")
        st.caption(f"📈 Confidence: {prob_rf:.1%}")
    
    # Why this prediction?
    st.markdown("---")
    st.markdown("### 🤔 Why this prediction?")
    
    reasons_positive = []
    reasons_negative = []
    
    # Class reasons
    if pclass == 1:
        reasons_positive.append("👑 First-class passengers had priority access to lifeboats")
    elif pclass == 2:
        reasons_positive.append("📘 Second-class passengers had decent lifeboat access")
    else:
        reasons_negative.append("⚓ Third-class passengers had limited lifeboat access")
    
    # Sex reasons
    if sex_clean == "Female":
        reasons_positive.append("👩 Women were evacuated first")
    else:
        reasons_negative.append("👨 Men had lower priority during evacuation")
    
    col_reason1, col_reason2 = st.columns(2)
    
    with col_reason1:
        if reasons_positive:
            st.markdown("**✅ Factors that helped:**")
            for r in reasons_positive:
                st.markdown(f"- {r}")
    
    with col_reason2:
        if reasons_negative:
            st.markdown("**❌ Factors that reduced chances:**")
            for r in reasons_negative:
                st.markdown(f"- {r}")
    
    # Passenger Summary
    st.markdown("---")
    st.markdown("### 📋 Passenger Summary")
    
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    
    class_emoji = {1: "👑", 2: "📘", 3: "⚓"}
    sex_emoji = "👩" if sex_clean == "Female" else "👨"
    survival_chance = prob_rf * 100
    
    with col_sum1:
        st.metric("🎫 Class", f"{class_emoji[pclass]} {pclass}{'st' if pclass==1 else 'nd' if pclass==2 else 'rd'} Class")
    with col_sum2:
        st.metric("👤 Sex", f"{sex_emoji} {sex_clean}")
    with col_sum3:
        st.metric("📊 Survival Chance", f"{survival_chance:.1f}%")
    
    # Survival probability bar
    st.markdown("**📈 Survival Probability Meter:**")
    st.progress(int(survival_chance))
    
    # Final verdict
    if pred_rf == 1:
        st.balloons()
        st.success("🎉 **VERDICT: The passenger would likely SURVIVE the Titanic disaster!** 🎉")
    else:
        st.error("💔 **VERDICT: The passenger would likely NOT SURVIVE the Titanic disaster.** 💔")

