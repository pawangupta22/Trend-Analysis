import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Trend Dashboard", layout="wide")

st.title("Social Media Trend Analysis Dashboard")

uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:

    # ===============================
    # 📊 LOAD DATA
    # ===============================
    df = pd.read_csv(uploaded_file)
    df_original = df.copy()

    st.subheader("Raw Data")
    st.dataframe(df.head())

    # ===============================
    # 🧹 CLEAN DATA
    # ===============================
    df['Post_Date'] = pd.to_datetime(df['Post_Date'], errors='coerce')
    df = df.dropna()

    # ===============================
    # 📌 FEATURE ENGINEERING
    # ===============================
    df['Total_Engagement'] = df['Likes'] + df['Shares'] + df['Comments']

    # ===============================
    # 🔥 TRENDING FILTER (TOP 25%)
    # ===============================
    threshold = df['Total_Engagement'].quantile(0.75)
    trending_df = df[df['Total_Engagement'] >= threshold]

    # ===============================
    # 🔥 TOP TRENDING HASHTAGS
    # ===============================
    st.subheader("🔥 Top Trending Hashtags")

    top_hashtags = (
        trending_df.groupby('Hashtag')['Total_Engagement']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    st.dataframe(top_hashtags.head(10))

    # ===============================
    # 📈 TREND SCORE
    # ===============================
    st.subheader("Trend Score Chart")

    score = top_hashtags.copy()
    score['Score'] = 100 * (score['Total_Engagement'] - score['Total_Engagement'].min()) / (
        score['Total_Engagement'].max() - score['Total_Engagement'].min()
    )

    plt.figure()
    plt.bar(score['Hashtag'].head(10), score['Score'].head(10))
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # ===============================
    # 📱 PLATFORM-WISE TRENDS
    # ===============================
    st.subheader("📱 Platform-wise Trends")

    platform_trend = (
        trending_df.groupby(['Platform', 'Hashtag'])['Total_Engagement']
        .sum()
        .reset_index()
    )

    for platform in platform_trend['Platform'].unique():
        st.write(f"### {platform}")
        st.dataframe(
            platform_trend[platform_trend['Platform'] == platform]
            .sort_values(by='Total_Engagement', ascending=False)
            .head(5)
        )

    # ===============================
    # 🎬 BEST CONTENT TYPE
    # ===============================
    st.subheader("Best Content Type per Hashtag")

    content_trend = (
        trending_df.groupby(['Hashtag', 'Content_Type'])['Total_Engagement']
        .sum()
        .reset_index()
    )

    best_content = content_trend.sort_values(
        ['Hashtag', 'Total_Engagement'], ascending=[True, False]
    )

    for tag in best_content['Hashtag'].unique():
        top = best_content[best_content['Hashtag'] == tag].head(1)
        st.write(f"{tag} → {top['Content_Type'].values[0]}")

    # ===============================
    # 📉 TREND DIRECTION
    # ===============================
    st.subheader("Trend Direction")

    df['Month'] = df['Post_Date'].dt.month

    growth = (
        df.groupby(['Hashtag', 'Month'])['Total_Engagement']
        .sum()
        .reset_index()
    )

    growth['Change'] = growth.groupby('Hashtag')['Total_Engagement'].diff()

    rising = growth[growth['Change'] > 0]['Hashtag'].unique()
    falling = growth[growth['Change'] < 0]['Hashtag'].unique()

    st.write("Rising Hashtags:", list(rising[:5]))
    st.write("Falling Hashtags:", list(falling[:5]))

    # ===============================
    # 🚀 SMART INSIGHTS
    # ===============================
    st.subheader("Smart Insights")

    for tag in top_hashtags['Hashtag'].head(5):

        best_platform = (
            platform_trend[platform_trend['Hashtag'] == tag]
            .sort_values(by='Total_Engagement', ascending=False)
            .iloc[0]['Platform']
        )

        best_content = (
            content_trend[content_trend['Hashtag'] == tag]
            .sort_values(by='Total_Engagement', ascending=False)
            .iloc[0]['Content_Type']
        )

        st.success(f"Use {tag} on {best_platform} with {best_content}")

else:
    st.info("Upload dataset to start analysis")
