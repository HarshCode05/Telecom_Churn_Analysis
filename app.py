import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Churn Dashboard Pro", layout="wide")

# =========================
# LOAD FILES
# =========================
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
df = pd.read_csv("telecom_churn.csv")

# =========================
# TITLE
# =========================
st.title("🚀 Telecom Churn Analytics Dashboard")
st.markdown("### Advanced Customer Insights & Prediction")

# =========================
# FILTERS
# =========================
st.sidebar.header("🔍 Filters")

selected_state = st.sidebar.selectbox("Select State", ["All"] + sorted(df['state'].unique()))
selected_city = st.sidebar.selectbox("Select City", ["All"] + sorted(df['city'].unique()))
selected_gender = st.sidebar.selectbox("Select Gender", ["All"] + sorted(df['gender'].unique()))

filtered_df = df.copy()

if selected_state != "All":
    filtered_df = filtered_df[filtered_df['state'] == selected_state]

if selected_city != "All":
    filtered_df = filtered_df[filtered_df['city'] == selected_city]

if selected_gender != "All":
    filtered_df = filtered_df[filtered_df['gender'] == selected_gender]

# =========================
# KPI CARDS
# =========================
st.markdown("## 📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Total Users", len(filtered_df))
col2.metric("📉 Churn Rate", f"{round(filtered_df['churn'].mean()*100,2)}%")
col3.metric("🎂 Avg Age", round(filtered_df['age'].mean(),1))
col4.metric("📡 Avg Data Usage", round(filtered_df['data_used'].mean(),1))

# =========================
# PREDICTION
# =========================
st.markdown("## 🔮 Churn Prediction")

with st.expander("Enter Customer Details"):

    age = st.slider("Age", 18, 70, 30)

    # ✅ FIXED UI (names only)
    gender = st.selectbox("Gender", sorted(df['gender'].unique()))
    city = st.selectbox("City", sorted(df['city'].unique()))
    state = st.selectbox("State", sorted(df['state'].unique()))

    calls = st.number_input("Calls Made", 0, 500, 50)
    sms = st.number_input("SMS Sent", 0, 500, 20)
    data_used = st.number_input("Data Used", 0.0, 50.0, 5.0)
    salary = st.number_input("Salary", 10000, 200000, 50000)

    if st.button("Predict"):

        input_data = pd.DataFrame({
            'age':[age],
            'gender':[encoders['gender'].transform([gender])[0]],
            'city':[encoders['city'].transform([city])[0]],
            'state':[encoders['state'].transform([state])[0]],
            'calls_made':[calls],
            'sms_sent':[sms],
            'data_used':[data_used],
            'estimated_salary':[salary]
        })

        result = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if result == 1:
            st.error(f"❌ High Risk Customer (Churn: {round(prob*100,2)}%)")
        else:
            st.success(f"✅ Safe Customer (Retention: {round((1-prob)*100,2)}%)")

# =========================
# DASHBOARD
# =========================
st.markdown("## 📈 Insights")

col5, col6 = st.columns(2)

with col5:
    st.markdown("### Churn Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='churn', data=filtered_df, ax=ax1)
    st.pyplot(fig1)

with col6:
    st.markdown("### Age vs Churn")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='churn', y='age', data=filtered_df, ax=ax2)
    st.pyplot(fig2)

col7, col8 = st.columns(2)

with col7:
    st.markdown("### Gender Analysis")
    fig3, ax3 = plt.subplots()
    sns.countplot(x='gender', hue='churn', data=filtered_df, ax=ax3)
    st.pyplot(fig3)

with col8:
    st.markdown("### Data Usage")
    fig4, ax4 = plt.subplots()
    sns.histplot(filtered_df['data_used'], kde=True, ax=ax4)
    st.pyplot(fig4)

# =========================
# GEO ANALYSIS
# =========================
st.markdown("## 🌍 Geographic Insights")

col9, col10 = st.columns(2)

with col9:
    st.markdown("### Top Cities")
    top_cities = filtered_df['city'].value_counts().head(10).index
    city_df = filtered_df[filtered_df['city'].isin(top_cities)]

    fig5, ax5 = plt.subplots()
    sns.countplot(y='city', hue='churn', data=city_df, ax=ax5)
    st.pyplot(fig5)

with col10:
    st.markdown("### Top States")
    top_states = filtered_df['state'].value_counts().head(10).index
    state_df = filtered_df[filtered_df['state'].isin(top_states)]

    fig6, ax6 = plt.subplots()
    sns.countplot(y='state', hue='churn', data=state_df, ax=ax6)
    st.pyplot(fig6)