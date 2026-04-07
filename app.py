import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

st.title("🔥 Trend Prediction Dashboard")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # CLEAN
    df['Post_Date'] = pd.to_datetime(df['Post_Date'], errors='coerce')
    df = df.dropna()

    df['Total_Engagement'] = df['Likes'] + df['Shares'] + df['Comments']

    # ENCODE
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()

    df['Platform'] = le1.fit_transform(df['Platform'])
    df['Content_Type'] = le2.fit_transform(df['Content_Type'])
    df['Region'] = le3.fit_transform(df['Region'])

    # ===============================
    # K-MEANS
    # ===============================
    features = df[['Platform','Content_Type','Region',
                   'Views','Likes','Shares','Comments','Total_Engagement']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Create label
    cluster_eng = df.groupby('Cluster')['Total_Engagement'].mean()
    trending_cluster = cluster_eng.idxmax()

    df['Trending'] = df['Cluster'].apply(lambda x: 1 if x == trending_cluster else 0)

    # ===============================
    # RANDOM FOREST
    # ===============================
    X = df[['Platform','Content_Type','Region',
            'Views','Likes','Shares','Comments']]

    y = df['Trending']

    model = RandomForestClassifier()
    model.fit(X, y)

    # Predict
    df['Prediction'] = model.predict(X)

    # ===============================
    # OUTPUT
    # ===============================
    st.subheader("🔥 Trending Hashtags")
    st.write(df[df['Prediction']==1]['Hashtag'].value_counts().head(10))

    st.subheader("📱 Best Platform")
    st.write(df[df['Prediction']==1]['Platform'].value_counts())

    st.subheader("🎬 Best Content Type")
    st.write(df[df['Prediction']==1]['Content_Type'].value_counts())

else:
    st.info("Upload dataset to start")
