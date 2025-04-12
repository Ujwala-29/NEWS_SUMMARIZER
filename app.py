import streamlit as st
from newspaper import Article
from transformers import pipeline
import nltk

# Download NLTK punkt for text processing
nltk.download('punkt')

# Page Config
st.set_page_config(page_title="🧠 News Summarizer", layout="wide")
st.title("🗞️ Hugging Face News Summarizer")

# Load summarizer once
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# Input method
option = st.radio("Choose input method:", ["🔗 Enter URL", "📝 Paste Article Text"])
article_text = ""
summary = ""

# URL input
if option == "🔗 Enter URL":
    url = st.text_input("Enter the News Article URL:")
    if st.button("Fetch Article"):
        try:
            article = Article(url)
            article.download()
            article.parse()
            article_text = article.text

            st.subheader("📝 Original Article")
            st.write(article_text[:1000] + "..." if len(article_text) > 1000 else article_text)
        except Exception as e:
            st.error(f"⚠️ Failed to load article: {str(e)}")

# Text input
elif option == "📝 Paste Article Text":
    article_text = st.text_area("Paste the full article text below:")

# Length sliders for summary
max_length = st.slider("Max Length of Summary", 50, 300, 150)
min_length = st.slider("Min Length of Summary", 30, 100, 40)

# Summarize
if article_text and st.button("🔍 Get Summary"):
    try:
        with st.spinner("Summarizing..."):
            summary = summarizer(article_text[:1024], max_length=max_length, min_length=min_length, do_sample=False)
        st.subheader("📌 Summary")
        st.success(summary[0]['summary_text'])
    except Exception as e:
        st.error(f"⚠️ Summarization failed: {str(e)}")
