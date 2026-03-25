import streamlit as st
import pickle
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from genai_utils import analyze_with_genai

model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

st.title("Sentiment Analysis with GenAI")

text = st.text_area("Enter text")

option = st.selectbox("Choose Model", ["ML Model", "GenAI"])

if st.button("Analyze"):
    if option == "ML Model":
        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        st.write(f"Prediction: {pred}")

    else:
        result = analyze_with_genai(text)
        st.write(result)


st.subheader("📊 Dataset Insights")

df = pd.read_csv("data/reviews.csv")

sentiment_counts = df['sentiment'].value_counts()

st.bar_chart(sentiment_counts)

# WordCloud
# text_data = " ".join(df['review'])

# wc = WordCloud(width=800, height=400).generate(text_data)

# fig, ax = plt.subplots()
# ax.imshow(wc)
# ax.axis("off")

# st.pyplot(fig)