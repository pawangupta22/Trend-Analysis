import streamlit as st
import pandas as pd
import joblib

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("trend_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Trend Analyzer", layout="wide")

st.title("🔥 Social Media Trend Prediction Dashboard")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # ===============================
    # CLEAN DATA
    # ===============================
    df['Post_Date'] = pd.to_datetime(df['Post_Date'], errors='coerce')
    df = df.dropna()

    df['Total_Engagement'] = df['Likes'] + df['Shares'] + df['Comments']

    # ===============================
    # ENCODING
    # ===============================
    from sklearn.preprocessing import LabelEncoder

    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()

    df['Platform'] = le1.fit_transform(df['Platform'])
    df['Content_Type'] = le2.fit_transform(df['Content_Type'])
    df['Region'] = le3.fit_transform(df['Region'])

    # ===============================
    # PREDICTION
    # ===============================
    X = df[['Platform', 'Content_Type', 'Region',
            'Views', 'Likes', 'Shares', 'Comments']]

    X_scaled = scaler.transform(X)

    df['Prediction'] = model.predict(X)

    # ===============================
    # RESULTS
    # ===============================
    st.subheader("🔮 Prediction Results")
    st.dataframe(df[['Prediction']].head())

    # ===============================
    # TRENDING DATA
    # ===============================
    trending_df = df[df['Prediction'] == 1]

    st.subheader("🔥 Predicted Trending Hashtags")
    st.write(trending_df['Hashtag'].value_counts().head(10))

    st.subheader("📱 Best Platform")
    st.write(trending_df['Platform'].value_counts().head())

    st.subheader("🎬 Best Content Type")
    st.write(trending_df['Content_Type'].value_counts().head())

else:
    st.info("Upload a dataset to start")
