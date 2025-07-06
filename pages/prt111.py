import streamlit as st
import pandas as pd
import os
from PIL import Image
import pytesseract
import speech_recognition as sr
import re
from collections import Counter
from wordcloud import WordCloud
from transformers import pipeline
import plotly.graph_objs as go
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from groq import Groq
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import networkx as nx
from sklearn.manifold import TSNE
from dotenv import load_dotenv


load_dotenv()  # load .env file

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- CONFIG ---
GROQ_MODEL = "llama3-70b-8192"
groq_client = Groq(api_key=GROQ_API_KEY)
PRODUCT_CONTEXT = (
    "You are analyzing customer reviews for a chocolate-flavoured whey protein powder. "
    "The product is aimed at fitness enthusiasts and helps with muscle growth and recovery."
)
RAW_CSV_PATH = "/Users/kushagraaatre/Downloads/Texpedition/data.csv"
REVIEW_FOLDER = "/Users/kushagraaatre/Downloads/Texpedition/review_files"
DEFAULT_CSV_PATH = "/Users/kushagraaatre/Downloads/Texpedition/data_with_text.csv"

# Neon colors for blocks
neon_blue = "#00fff7"
neon_green = "#7CFC00"
neon_pink = "#F72585"
neon_yellow = "#FFF600"
neon_bg = "#181830"
neon_orange = "#FFB347"

# --- UTILS ---
def clean_name(name):
    return (
        str(name)
        .strip()
        .replace('\ufeff', '')
        .replace('\n', '')
        .replace('\r', '')
        .replace('\t', '')
        .lower()
    )

def extract_review_text(df, review_file_dict):
    review_texts = []
    for i, row in df.iterrows():
        fname = clean_name(row['review_file'])
        file = review_file_dict.get(fname)
        text = ""
        if file is None:
            text = "(missing file)"
        elif fname.endswith(".txt"):
            try:
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()
                if not text:
                    text = "(text unreadable)"
            except Exception:
                text = "(text unreadable)"
        elif fname.endswith(".png"):
            try:
                img = Image.open(file)
                text = pytesseract.image_to_string(img).strip()
                if not text:
                    text = "(image unreadable)"
            except Exception:
                text = "(image unreadable)"
        elif fname.endswith(".wav"):
            r = sr.Recognizer()
            try:
                with sr.AudioFile(file) as source:
                    audio = r.record(source)
                text = r.recognize_google(audio)
                if not text:
                    text = "(audio unreadable)"
            except Exception:
                text = "(audio unreadable)"
        else:
            text = "(unsupported file)"
        review_texts.append(text)
    return review_texts

@st.cache_resource(show_spinner=True)
def get_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def hf_sentiment(text):
    try:
        result = sentiment_pipeline(text[:512])[0]
        label = result['label']
        score = result['score']
        if score <= 0.6:
            return ("Neutral", 0.0)
        if label == "POSITIVE" and score > 0.8:
            return ("Strongly Positive", score)
        elif label == "POSITIVE":
            return ("Positive", score)
        elif label == "NEGATIVE" and score > 0.8:
            return ("Strongly Negative", -score)
        else:
            return ("Negative", -score)
    except Exception:
        return ("Neutral", 0.0)

def groq_bullets(chart_desc, chart_data_text):
    user_prompt = (
        f"Summarize as exactly two bullet points the main insights for a chocolate whey protein product, from this chart: {chart_desc}. "
        f"Here is the relevant data or result: {chart_data_text}. "
        "Do not use the words 'says', 'shows', 'suggests', 'tells', 'reveals', 'indicates', or any similar phrases. Just facts."
    )
    try:
        chat_completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": PRODUCT_CONTEXT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=80,
            temperature=0.6,
        )
        bullets = chat_completion.choices[0].message.content.strip()
        points = [line for line in bullets.splitlines() if line.strip().startswith(("-", "•"))]
        points = [pt.strip() for pt in points if pt.strip() and not pt.lower().startswith("summary")]
        return "\n".join(points[:2]) if len(points) >= 2 else "- " + bullets
    except Exception:
        return "- Summary not available.\n- (LLM error)"

def block_markdown(text, color):
    text = text.replace('\n', '<br>')
    return (
        f'<div style="background:linear-gradient(90deg,{color}22,#181830 90%);'
        f'padding:16px 22px;border-radius:14px;margin:10px 0 24px 0;'
        f'font-weight:600;color:#fff;font-size:1.04em;line-height:1.6">'
        f'{text}</div>'
    )

def groq_summary_block(prompt):
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": PRODUCT_CONTEXT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "(Summary not available.)"

def groq_top_sentiments(all_text, pos_or_neg="positive"):
    prompt = (
        f"Summarize the top 3 {pos_or_neg} sentiments from these customer reviews about a chocolate whey protein powder. "
        f"Give each sentiment as a short, specific bullet point (not quotes)."
        f"Reviews: {all_text[:4000]}"
    )
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": PRODUCT_CONTEXT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=80,
            temperature=0.5,
        )
        lines = [line for line in resp.choices[0].message.content.strip().split('\n') if line.strip().startswith(("-", "•"))]
        return "\n".join(lines[:3])
    except Exception:
        return "- Not available.\n- (LLM error)"

def top_n_reviews(df, sentiment, n=3):
    if sentiment.lower().startswith("pos"):
        filt = df["sentiment_label"].str.contains("Positive", case=False)
        top = df.loc[filt].sort_values("polarity", ascending=False)
    elif sentiment.lower().startswith("neg"):
        filt = df["sentiment_label"].str.contains("Negative", case=False)
        # Filter for .txt reviews only
        if 'review_file' in df.columns:
            txt_mask = df["review_file"].astype(str).str.endswith('.txt')
            top = df.loc[filt & txt_mask].sort_values("polarity")
        else:
            top = df.loc[filt].sort_values("polarity")
    else:
        return []
    return top["review_text"].head(n).tolist()


# --- LAYOUT ---
st.set_page_config(page_title="🌐 Insight Engine", layout="wide", initial_sidebar_state="collapsed")

# --- AGE TITLE ---
st.markdown(
    "<h1 style='color:#00fff7;font-size:2.65rem;font-weight:900;letter-spacing:0.01em;margin-bottom:5px;'>🌐 Insight Engine</h1>", 
    unsafe_allow_html=True
)

# --- CHEEKY INTRO (PURPLE, Multimodal, Text to Insights) ---
st.markdown("""
<div style="font-size:1.22rem; color:#AC7CFF; font-weight:600; margin-top:-12px; margin-bottom:11px; line-height:1.56;">
    🚀 Welcome to your all-in-one playground for market insight magic—supercharged with <b>multimodal skills</b>!  
    Drop in text, images, or even audio—we'll crunch it all and transform bland data into beautiful, actionable insights.  
    Curious what customers really think? Need to turn a wall of reviews into dazzling graphs, smart summaries, and aha-moments?  
</div>
""", unsafe_allow_html=True)

# --- EXPLANATION FOR THE DEMO DATASET (YELLOW/ORANGE) ---
st.markdown("""
<div style="font-size:1.12rem; color:#FFB347; font-weight:700; margin-bottom:14px; line-height:1.49;">
    For this demo, we’ve loaded up a dataset of chocolate protein powder reviews—so you can see all features in action, no setup needed.  
    But hey, The magic works for everything from cookies to kettlebells.
</div>
""", unsafe_allow_html=True)


# Add custom CSS for neon buttons
st.markdown("""
    <style>
    .neon-btn {
        display:inline-block;
        font-weight:bold;
        padding:14px 32px;
        border:none;
        border-radius:12px;
        font-size:1.1em;
        margin-right:18px;
        cursor:pointer;
        box-shadow:0 0 14px #00fff777;
        color:#222 !important;
        background:linear-gradient(90deg,#7CFC00,#00fff7);
        text-decoration:none !important;
        transition: transform 0.08s;
    }
    .neon-btn-pink {
        background:linear-gradient(90deg,#F72585,#00fff7);
        color:#fff !important;
        box-shadow:0 0 14px #F7258577;
    }
    .neon-btn:hover {
        transform:scale(1.04);
        box-shadow:0 0 24px #00fff799;
    }
    .neon-btn-pink:hover {
        box-shadow:0 0 24px #F7258599;
    }
    </style>
""", unsafe_allow_html=True)

# Place the links side by side
st.markdown("""
<div style="display:flex;gap:2em;">
    <a href="/persona" class="neon-btn"target="_self">👤 Persona Analysis</a>
    <a href="/newprod" class="neon-btn neon-btn-pink"target="_self">🚀 New Product Launch</a>
</div>
<br>
""", unsafe_allow_html=True)

# --- LOAD DATA & PREPROCESS ---
csv_path = DEFAULT_CSV_PATH
if not os.path.exists(csv_path):
    st.warning(f"Preprocessed CSV not found at {csv_path}. Starting file extraction & text recognition...")
    if not os.path.exists(RAW_CSV_PATH):
        st.error(f"Raw CSV file not found at {RAW_CSV_PATH}")
        st.stop()
    df = pd.read_csv(RAW_CSV_PATH)
    review_file_dict = {}
    if not os.path.exists(REVIEW_FOLDER):
        st.error(f"Review folder not found at {REVIEW_FOLDER}")
        st.stop()
    for fname in os.listdir(REVIEW_FOLDER):
        key = clean_name(fname)
        full_path = os.path.join(REVIEW_FOLDER, fname)
        if os.path.isfile(full_path):
            review_file_dict[key] = full_path
    df["review_text"] = extract_review_text(df, review_file_dict)
    df.to_csv(csv_path, index=False)
    st.success("Preprocessing complete! Continuing with analysis...")
else:
    df = pd.read_csv(csv_path)

df["review_text"] = df["review_text"].fillna("")
sentiment_pipeline = get_sentiment_pipeline()

with st.spinner("Running HuggingFace sentiment analysis on reviews... (first time may take a minute)"):
    df[["sentiment_label", "polarity"]] = df["review_text"].apply(
        lambda x: hf_sentiment(x) if x and "unreadable" not in x and "missing" not in x else ("Neutral", 0)
    ).apply(pd.Series)

df["review_length"] = df["review_text"].apply(lambda x: len(str(x).split()))
df_valid = df[
    ~df["review_text"].str.contains("unreadable|missing|unsupported", case=False, na=False)
    & df["review_text"].str.strip().astype(bool)
]
all_reviews = " ".join(df_valid["review_text"])

# ----------------- MAIN GRAPHS (numbered, with summaries in blocks) -------------------

# --- 1 & 2. Sentiment Distribution + Top Themes ---
c1, c2 = st.columns(2)
with c1:
    st.subheader("1. Sentiment Distribution")
    sentiment_counts = df["sentiment_label"].value_counts()
    color_dict = {
        "Strongly Positive": neon_green,
        "Positive": neon_blue,
        "Neutral": neon_yellow,
        "Negative": neon_pink,
        "Strongly Negative": "#c1121f"
    }
    colors = [color_dict.get(lbl, "#a67b5b") for lbl in sentiment_counts.index]
    fig_pie = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=0.4,
        marker=dict(colors=colors),
    )])
    fig_pie.update_traces(textinfo='percent+label')
    fig_pie.update_layout(showlegend=True, legend=dict(orientation="h"), font=dict(size=16))
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown(block_markdown(groq_bullets("Sentiment distribution pie chart", f"Counts: {sentiment_counts.to_dict()}"), neon_blue), unsafe_allow_html=True)

with c2:
    st.subheader("2. Top Themes")
    if len(df_valid) > 0:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=10)
        X = vectorizer.fit_transform(df_valid["review_text"].fillna(""))
        keywords = [w for w in vectorizer.get_feature_names_out() if len(w) > 2 and w.lower() not in ["says", "tells", "said", "like", "really"]]
        counts = X.sum(axis=0).A1
        theme_counts = sorted(zip(keywords, counts), key=lambda x: -x[1])
        fig_theme = go.Figure(data=[
            go.Bar(
                x=[k for k, _ in theme_counts], y=[int(c) for _, c in theme_counts],
                marker=dict(color=[neon_green, neon_pink, neon_blue, neon_yellow, neon_orange]*2)
            )
        ])
        fig_theme.update_layout(xaxis_title='Theme/Keyword', yaxis_title='Frequency', font=dict(size=16))
        st.plotly_chart(fig_theme, use_container_width=True)
        st.markdown(block_markdown(
            groq_bullets("Bar chart of frequency of top review themes", 
                         ', '.join([k for k,_ in theme_counts])), neon_orange), unsafe_allow_html=True)
    else:
        st.write("No valid reviews for theme extraction.")
        st.markdown(block_markdown("- No data.\n- No chart.", neon_orange), unsafe_allow_html=True)

st.markdown("---")

# --- 3 & 4. Sentiment Trend Over Time + Aspect-Based Sentiment ---
c3, c4 = st.columns(2)
with c3:
    st.subheader("3. Sentiment Trend Over Time")
    df_valid = df_valid.reset_index()
    df_valid["review_idx"] = df_valid.index + 1
    df_valid_trend = df_valid.groupby("review_idx").agg({"polarity": "mean"}).reset_index()
    fig_line = go.Figure(data=[
        go.Scatter(
            x=df_valid_trend["review_idx"], y=df_valid_trend["polarity"],
            mode="lines+markers+text",
            line=dict(color=neon_pink, width=4, dash='dash'),
            marker=dict(size=8, color=neon_green, symbol="diamond"),
        )
    ])
    fig_line.update_layout(
        xaxis_title="Review Index (chronological)",
        yaxis_title="Avg Sentiment",
        font=dict(size=16, color=neon_pink),
        plot_bgcolor=neon_bg
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown(block_markdown(
        groq_bullets("Sentiment trend line over time (reviews in chronological order)",
        f"Polarity: {list(df_valid_trend['polarity'][:30])}"), neon_pink), unsafe_allow_html=True)

with c4:
    st.subheader("4. Aspect-Based Sentiment")
    aspects = ["price", "quality", "delivery", "taste", "mixability"]
    aspect_scores = []
    for aspect in aspects:
        mask = df_valid["review_text"].str.contains(aspect, case=False, na=False)
        pols = df_valid.loc[mask, "polarity"]
        aspect_scores.append(pols.mean() if not pols.empty else 0)
    fig_aspect = go.Figure(data=[
        go.Bar(
            x=aspects, y=aspect_scores,
            marker=dict(color=[neon_blue, neon_green, neon_pink, neon_yellow, neon_orange])
        )
    ])
    fig_aspect.update_layout(xaxis_title="Aspect", yaxis_title="Avg Sentiment", font=dict(size=16))
    st.plotly_chart(fig_aspect, use_container_width=True)
    st.markdown(block_markdown(
        groq_bullets(
            "Bar chart of sentiment for product aspects (price, quality, delivery, taste, mixability)",
            str(dict(zip(aspects, [f"{x:.2f}" for x in aspect_scores])))
        ), neon_green), unsafe_allow_html=True)


st.markdown("---")

# --- 5 & 6. Word Cloud + Review Length Trend ---
# Add this above your word cloud and co-occurrence logic
stopwords = set("""
the and for with you that this are have from all has can will just get out too its on an is in it of to a i my says said tell tells also would could should not as if be do does did was were been being by he she they them their our we us his her its so or at more most some such only may might like one two first second every much well still own even many go goes gone didn't don't isn't aren't wasn't weren't doesn't haven't hadn't can't won't won't wouldn't mustn't protein powder review
""".split())

def filter_tokens(words):
    return [w for w in words if w not in stopwords and len(w) > 2 and not w.isnumeric()]
c5, c6 = st.columns(2)
with c5:
    st.subheader("5. Word Cloud")
    if all_reviews.strip():
        words = re.findall(r'\w+', all_reviews.lower())
        filtered_words = filter_tokens(words)
        filtered_text = " ".join(filtered_words)
        wc = WordCloud(
            width=900, height=400, background_color=neon_bg, colormap='winter',
            max_words=80, random_state=42
        ).generate(filtered_text)
        st.image(wc.to_array(), use_column_width=True)
        top_words = ", ".join([w for w, _ in Counter(filtered_words).most_common(12)])
        st.markdown(block_markdown(groq_bullets("Word cloud of frequent review words", top_words), neon_yellow), unsafe_allow_html=True)
    else:
        st.write("No review text available.")
        st.markdown(block_markdown("- No text for word cloud.", neon_yellow), unsafe_allow_html=True)

with c6:
    st.subheader("6. Review Length Trend")
    if len(df_valid) > 0:
        review_lengths = df_valid["review_length"].reset_index(drop=True)
        fig_line_length = go.Figure(data=[
            go.Scatter(
                x=review_lengths.index + 1, y=review_lengths,
                mode="lines+markers",
                line=dict(color=neon_orange, width=3)
            )
        ])
        fig_line_length.update_layout(
            xaxis_title="Review (chronological order)",
            yaxis_title="Review Length (words)",
            font=dict(size=16), plot_bgcolor=neon_bg
        )
        st.plotly_chart(fig_line_length, use_container_width=True)
        st.markdown(block_markdown(
            groq_bullets("Line chart showing trend of review lengths (number of words) in chronological order",
            f"Lengths: {list(review_lengths[:50])}"), neon_orange), unsafe_allow_html=True)
    else:
        st.write("No valid reviews for length trend.")
        st.markdown(block_markdown("- No data.\n- No chart.", neon_orange), unsafe_allow_html=True)

st.markdown("---")

# --- 7 & 8. Sentiment Polarity Histogram + Emotion Analysis ---
c7, c8 = st.columns(2)
with c7:
    st.subheader("7. Sentiment Polarity Histogram")
    # Make histogram visually full by using kde line (density)
    polarity_values = df_valid["polarity"].values
    fig_hist, ax = plt.subplots(figsize=(7,3))
    ax.hist(polarity_values, bins=8, color=neon_blue, alpha=0.88, edgecolor="#222", density=True)
    ax.set_xlabel("Sentiment Polarity Score")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Sentiment Scores")
    # KDE line
    if len(polarity_values) > 1:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(polarity_values)
        x_range = np.linspace(-1, 1, 200)
        ax.plot(x_range, kde(x_range), color=neon_green, lw=2)
    st.pyplot(fig_hist)
    st.markdown(block_markdown(
        groq_bullets("Histogram of sentiment scores", list(polarity_values[:50])), neon_blue
    ), unsafe_allow_html=True)

with c8:
    st.subheader("8. Emotion Analysis Bar Chart")
    @st.cache_resource(show_spinner=True)
    def get_emotion_pipeline():
        return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    emotion_pipeline = get_emotion_pipeline()
    emotion_counts = {}
    for review in df_valid["review_text"]:
        try:
            emotions = emotion_pipeline(review[:512])
            for e in emotions:
                for d in e:
                    emotion = d['label']
                    if d['score'] > 0.2:
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        except Exception:
            continue
    if emotion_counts:
        fig_emotion = go.Figure(data=[
            go.Bar(
                x=list(emotion_counts.keys()),
                y=list(emotion_counts.values()),
                marker=dict(color=[neon_pink, neon_green, neon_blue, neon_yellow, neon_orange])
            )
        ])
        fig_emotion.update_layout(xaxis_title="Emotion", yaxis_title="Count", font=dict(size=16))
        st.plotly_chart(fig_emotion, use_container_width=True)
        st.markdown(block_markdown(
            groq_bullets("Bar chart of detected emotions in reviews", str(emotion_counts)), neon_pink
        ), unsafe_allow_html=True)
    else:
        st.write("No emotion results (try more reviews).")


st.markdown("---")

# --- 9 & 10. Bigram/Trigram Frequency + Co-occurrence Network ---
c9, c10 = st.columns(2)
with c9:
    st.subheader("9. Bigram/Trigram Frequency")
    # Use only meaningful ngrams (exclude numbers, names)
    corpus = df_valid["review_text"].tolist()
    vect = CountVectorizer(ngram_range=(2,3), stop_words='english', max_features=20, token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b')
    X_ngram = vect.fit_transform(corpus)
    ngram_counts = X_ngram.sum(axis=0).A1
    ngrams = vect.get_feature_names_out()
    ngram_freq = sorted(zip(ngrams, ngram_counts), key=lambda x: -x[1])
    fig_ngram = go.Figure(data=[
        go.Bar(
            y=[ng for ng,_ in ngram_freq],
            x=[int(c) for _,c in ngram_freq],
            orientation='h',
            marker=dict(color=neon_blue)
        )
    ])
    fig_ngram.update_layout(yaxis_title='Phrase', xaxis_title='Count', font=dict(size=15))
    st.plotly_chart(fig_ngram, use_container_width=True)
    st.markdown(block_markdown(
        groq_bullets("Bar chart of most common bigrams/trigrams", ', '.join([f"{ng}: {c}" for ng,c in ngram_freq])), neon_blue
    ), unsafe_allow_html=True)

with c10:
    st.subheader("10. Co-occurrence Network Graph")

    def get_top_cooc_words(texts, top_n=12):
        words = [filter_tokens(re.findall(r'\w+', t.lower())) for t in texts]
        all_pairs = []
        for wlist in words:
            all_pairs.extend(list(combinations(set(wlist), 2)))
        counter = Counter(all_pairs)
        return counter.most_common(top_n)

    top_pairs = get_top_cooc_words(df_valid["review_text"])
    G = nx.Graph()
    for (a, b), w in top_pairs:
        G.add_edge(a, b, weight=w)

    # Use Kamada-Kawai layout for more even node spacing
    pos = nx.kamada_kawai_layout(G)

    # Adjust node and font size for clarity
    node_count = G.number_of_nodes()
    base_node_size = 620 if node_count <= 10 else max(390, 1400 // (node_count + 1))
    font_size = 15 if node_count <= 10 else max(9, 20 - node_count // 2)

    plt.figure(figsize=(7.4, 6.1))
    nx.draw_networkx_nodes(
        G, pos, node_color=neon_orange, edgecolors="#fff", linewidths=2,
        node_size=base_node_size, alpha=0.96
    )
    nx.draw_networkx_edges(
        G, pos,
        width=[2.2 + G[u][v]['weight'] / 2.4 for u, v in G.edges()],
        edge_color=neon_blue, alpha=0.76
    )
    nx.draw_networkx_labels(
        G, pos, font_size=font_size, font_color="#212121", font_weight="bold"
    )
    plt.axis('off')
    plt.tight_layout(pad=0.3)
    st.pyplot(plt.gcf())
    plt.clf()

    # --- GROQ SUMMARY (2 lines, info box style) ---
    def groq_summary_graph(prompt):
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": PRODUCT_CONTEXT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=90,
                temperature=0.55,
            )
            # Remove asterisks, intro, etc
            lines = [
                line.strip(" *-•1234567890.").replace("**", "")
                for line in resp.choices[0].message.content.strip().split("\n")
                if line.strip()
            ]
            # Only first 2 lines (you may get 1-3 lines, but only keep 2)
            return "<br>".join(lines[:2])
        except Exception:
            return "Summary not available."

    cooc_pairs_str = "; ".join([f"{a}-{b} ({w})" for (a, b), w in top_pairs])
    graph_summary = groq_summary_graph(
        f"Summarize the key relationships or surprising findings in exactly two punchy, non-repetitive lines from this co-occurrence network of customer review words. "
        f"No generic intro, only crisp insights. Pairs: {cooc_pairs_str}"
    )

    st.markdown(
        f"""
        <div style='background:linear-gradient(90deg,{neon_blue}22,{neon_orange}22);border-radius:14px;padding:18px 22px 12px 22px;margin-top:14px;margin-bottom:14px;box-shadow:0 2px 18px {neon_blue}19;'>
            <span style='color:{neon_orange};font-size:1.15em;font-weight:800;'>Quick Network Insights:</span><br>
            <span style='color:#fff;font-size:1.09em;'>{graph_summary}</span>
        </div>
        """, unsafe_allow_html=True
    )
    



st.markdown("---")

# --- 11. Review Cluster Visualization (t-SNE) ---
st.subheader("11. Review Cluster Visualization (t-SNE)")
vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
X = vectorizer.fit_transform(df_valid["review_text"].fillna("")).toarray()
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(df_valid)//2)))
X_tsne = tsne.fit_transform(X)
fig_tsne = go.Figure(data=[
    go.Scatter(
        x=X_tsne[:,0], y=X_tsne[:,1], mode="markers",
        marker=dict(color=df_valid["polarity"], colorscale="RdYlGn", size=12, showscale=True),
        text=df_valid["sentiment_label"]
    )
])
fig_tsne.update_layout(xaxis_title="t-SNE 1", yaxis_title="t-SNE 2", font=dict(size=16))
st.plotly_chart(fig_tsne, use_container_width=True)
st.markdown(block_markdown(
    groq_bullets("2D scatterplot of review clusters by t-SNE", "points colored by sentiment"), neon_blue
), unsafe_allow_html=True)

st.markdown("---")

# ----------- Final Neon Blocks: Top Quotes and Summaries -----------
st.markdown("---")
cl1, cl2 = st.columns(2)
with cl1:
    st.markdown(block_markdown(
        "<b>Top 3 Enthusiastic Positive Reviews:</b><br>" + "<br><br>".join(
            [f'<span style="color:{neon_green}">“{r}”</span>' for r in top_n_reviews(df_valid, "Positive", 3)]
        ),
        neon_green), unsafe_allow_html=True)
with cl2:
    st.markdown(block_markdown(
        "<b>Top 3 Most Critical Negative Reviews:</b><br>" + "<br><br>".join(
            [f'<span style="color:{neon_pink}">“{r}”</span>' for r in top_n_reviews(df_valid, "Negative", 3)]
        ),
        neon_pink), unsafe_allow_html=True)

cl3, cl4 = st.columns(2)
with cl3:
    all_pos_text = " ".join(df_valid[df_valid["polarity"] > 0]["review_text"])
    st.markdown(block_markdown(
        "<b>Top 3 Positive Sentiments:</b><br>" + groq_top_sentiments(all_pos_text, "positive"),
        neon_green), unsafe_allow_html=True)
with cl4:
    all_neg_text = " ".join(df_valid[df_valid["polarity"] < 0]["review_text"])
    st.markdown(block_markdown(
        "<b>Top 3 Negative Sentiments:</b><br>" + groq_top_sentiments(all_neg_text, "negative"),
        neon_pink), unsafe_allow_html=True)

cl5, cl6 = st.columns(2)
with cl5:
    sentiment_texts = groq_summary_block(
        "List the top 3 overall customer sentiments about the chocolate whey protein product as short phrases (not sentences, not quotes, just phrases)."
    )
    st.markdown(block_markdown(
        "<b>Top 3 Overall Sentiments:</b><br>" + sentiment_texts.replace('\n', '<br>'),
        neon_yellow), unsafe_allow_html=True)
with cl6:
    trend_summary = groq_summary_block(
        "Summarize trends in one short sentence for chocolate protein reviews. "
        "What do people like most, and what do they dislike most?"
    )
    st.markdown(block_markdown(
        "<b>Summary of Trends:</b><br>" + trend_summary,
        neon_blue), unsafe_allow_html=True)



st.markdown("---\n<small style='color:#7CFC00'>Bugs Fring — End of Report</small>", unsafe_allow_html=True)
