import streamlit as st
import feedparser
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from newspaper import Article
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Market Intelligence NLP Tool", layout="centered")

st.title("📊 Market Intelligence NLP Tool")

st.markdown("""
### 👋 Welcome!
Analyze real-world articles and extract:

- 🧠 TF-IDF Themes  
- 🔗 Common phrases (2–4 words)  
- 📉 Sentiment analysis  
- ☁️ Word clouds (1 → 5 word patterns)

👉 Try:
- healthy snacking trends India  
- electric vehicles future  
- AI in healthcare  
""")

# -----------------------
# NLTK SETUP
# -----------------------
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)

custom_stopwords = {
    "india","indian","market","markets","report","said","also",
    "new","like","would","could","global","year","time"
}

stop_words = set(stopwords.words('english')).union(custom_stopwords)
sia = SentimentIntensityAnalyzer()

# -----------------------
# FETCH ARTICLES (CACHED)
# -----------------------
@st.cache_data(show_spinner=False)
def fetch_articles(query, max_articles=8):  # reduced from 12 → 8
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}"
    feed = feedparser.parse(url)

    texts, titles, links = [], [], []
    count = 0

    for entry in feed.entries:
        if count >= max_articles:
            break

        try:
            titles.append(entry.title)
            links.append(entry.link)

            texts.append(entry.title + " " + entry.summary)

            article = Article(entry.link)
            article.download()
            article.parse()

            if len(article.text) > 300:
                texts.append(article.text)

            count += 1
        except:
            continue

    return pd.DataFrame({"text": texts}), titles, links

# -----------------------
# CLEAN TEXT
# -----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 3]
    return " ".join(words)

# -----------------------
# SENTIMENT
# -----------------------
def apply_sentiment(df):
    df["score"] = df["clean_text"].apply(lambda x: sia.polarity_scores(x)["compound"])

    def label(x):
        if x > 0.05:
            return "Positive"
        elif x < -0.05:
            return "Negative"
        else:
            return "Neutral"

    df["sentiment"] = df["score"].apply(label)
    return df

def sentiment_summary(df):
    dist = df["sentiment"].value_counts(normalize=True) * 100
    return {k: round(v,2) for k,v in dist.items()}

# -----------------------
# TF-IDF THEMES
# -----------------------
def get_tfidf_themes(df):
    vec = TfidfVectorizer(max_features=10)
    X = vec.fit_transform(df["clean_text"])
    return vec.get_feature_names_out()

# -----------------------
# NGRAMS (SAFE)
# -----------------------
def get_ngrams(df, n):
    try:
        vec = CountVectorizer(
            ngram_range=(n,n),
            max_features=10,
            min_df=2
        )
        X = vec.fit_transform(df["clean_text"])
        return vec.get_feature_names_out()

    except:
        vec = CountVectorizer(
            ngram_range=(n,n),
            max_features=10,
            min_df=1
        )
        X = vec.fit_transform(df["clean_text"])
        return vec.get_feature_names_out()

# -----------------------
# WORD CLOUD TEXT (SAFE)
# -----------------------
def generate_ngram_text(df, n):
    try:
        vec = CountVectorizer(
            ngram_range=(n,n),
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        X = vec.fit_transform(df["clean_text"])

    except:
        vec = CountVectorizer(
            ngram_range=(n,n),
            stop_words='english',
            min_df=1
        )
        X = vec.fit_transform(df["clean_text"])

    phrases = vec.get_feature_names_out()
    counts = X.toarray().sum(axis=0)

    words = []
    for p, c in zip(phrases, counts):
        phrase = p.replace(" ", "_")
        words.extend([phrase] * int(c))

    return " ".join(words)

# -----------------------
# WORD CLOUD
# -----------------------
def plot_wc(text, title):
    if not text.strip():
        return None

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        collocations=False
    ).generate(text)

    wc.words_ = {k.replace("_", " "): v for k, v in wc.words_.items()}

    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    ax.set_title(title)
    return fig

# -----------------------
# INPUT
# -----------------------
query = st.text_input("🔍 Enter your topic:")

if st.button("Run Analysis"):

    if not query:
        st.warning("Please enter a topic")
    else:
        with st.spinner("⏳ Fetching articles and running NLP analysis..."):

            df, titles, links = fetch_articles(query)

            if df.empty:
                st.error("No data found.")
            else:
                df["clean_text"] = df["text"].apply(clean_text)
                df = apply_sentiment(df)

                themes = get_tfidf_themes(df)
                sentiment = sentiment_summary(df)

                short = get_ngrams(df,2)
                medium = get_ngrams(df,3)
                long = get_ngrams(df,4)

        st.success("✅ Analysis Complete")

        st.subheader("📰 Source Articles")
        for t,l in zip(titles,links):
            st.markdown(f"- [{t}]({l})")

        st.subheader("🎯 Key Themes (TF-IDF)")
        st.write(list(themes))

        st.subheader("📉 Sentiment Distribution")
        st.write(sentiment)

        st.subheader("🔗 Common Phrases")
        st.write("Short (2-word):", list(short))
        st.write("Medium (3-word):", list(medium))
        st.write("Long (4-word):", list(long))

        st.subheader("☁️ Word Cloud Analysis (1 → 5 grams)")

        for n in range(1,6):
            text = generate_ngram_text(df, n)
            fig = plot_wc(text, f"{n}-gram")

            if fig:
                st.pyplot(fig)
            else:
                st.write(f"No meaningful {n}-gram data found.")
