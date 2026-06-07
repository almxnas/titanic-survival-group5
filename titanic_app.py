import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

# ==================== TITLE ====================
st.title("🚢 Titanic Survival Prediction - Group 5")
st.markdown("**🎯 Binary Classification:** Predict passenger survival (0 = Died, 1 = Survived)")
st.markdown("---")

# ==================== LOAD & TRAIN MODELS ====================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

@st.cache_resource
def train_models():
    df = load_data()
    
    # Preprocessing (same as your report)
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
    
    return lr, rf, le_sex, le_emb, avg_fare, df.head(10)

lr, rf, le_sex, le_emb, avg_fare, sample_data = train_models()

# ==================== DATASET ====================
st.header("1. Dataset")
st.dataframe(sample_data)

# ==================== LIVE PREDICTION ====================
st.header("🎯 Live Survival Prediction")
st.markdown("Enter passenger details below and click **Predict Survival**")

col1, col2, col3 = st.columns(3)
with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3], 
                         format_func=lambda x: f"{x} - {'1st (Upper)' if x==1 else '2nd (Middle)' if x==2 else '3rd (Lower)'}")

with col2:
    sex = st.radio("Sex", ["Female", "Male"], horizontal=True)

with col3:
    age = st.slider("Age", 0, 80, 30)

if st.button("🚀 PREDICT SURVIVAL", type="primary", use_container_width=True):
    sex_encoded = 0 if sex == "Male" else 1
    fare = avg_fare[pclass]
    
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'SibSp': [0],
        'Parch': [0],
        'Fare': [fare],
        'Embarked': [0]
    })
    
    pred_lr = lr.predict(input_data)[0]
    prob_lr = lr.predict_proba(input_data)[0][1]
    pred_rf = rf.predict(input_data)[0]
    prob_rf = rf.predict_proba(input_data)[0][1]
    
    st.markdown("---")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.subheader("Logistic Regression")
        if pred_lr == 1:
            st.success(f"✅ SURVIVED ({prob_lr:.1%})")
        else:
            st.error(f"❌ DID NOT SURVIVE ({prob_lr:.1%})")
    
    with col_res2:
        st.subheader("Random Forest")
        if pred_rf == 1:
            st.success(f"✅ SURVIVED ({prob_rf:.1%})")
        else:
            st.error(f"❌ DID NOT SURVIVE ({prob_rf:.1%})")

st.caption("Note: This model matches our group report (Logistic Regression + Random Forest)")
