# app.py

import streamlit as st
from newspaper import Article
from transformers import pipeline
import nltk

nltk.download('punkt')

# Page Config
st.set_page_config(page_title="🧠 News Summarizer", layout="wide")
st.title("🗞️ Hugging Face News Summarizer")

# Summarizer pipeline (only load once)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# Option: URL or Raw Text
option = st.radio("Choose input method:", ["🔗 Enter URL", "📝 Paste Article Text"])

article_text = ""

# URL Method
if option == "🔗 Enter URL":
    url = st.text_input("Enter the News Article URL:")
    if url and st.button("Fetch & Summarize"):
        try:
            article = Article(url)
            article.download()
            article.parse()
            article_text = article.text

            st.subheader("📝 Original Article")
            st.write(article_text[:1000] + "...")

        except Exception as e:
            st.error(f"⚠️ Failed to load article: {str(e)}")

# Text Method
elif option == "📝 Paste Article Text":
    article_text = st.text_area("Paste the full article text below:")
    if article_text and st.button("Summarize"):
        st.subheader("📝 Original Article")
        st.write(article_text[:1000] + "...")

# Summarization (if article_text exists)
if article_text and st.button("🔍 Get Summary"):
    try:
        with st.spinner("Summarizing..."):
            summary = summarizer(article_text[:1024], max_length=150, min_length=40, do_sample=False)
            st.subheader("📌 Summary")
            st.success(summary[0]['summary_text'])
    except Exception as e:
        st.error(f"⚠️ Summarization failed: {str(e)}")

