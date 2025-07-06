import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from groq import Groq
import plotly.graph_objs as go
from collections import defaultdict
from itertools import cycle
import json
from dotenv import load_dotenv
PERSONA_PATH = "personas.json"

# --- THEME COLORS ---
neon_blue = "#00fff7"
neon_green = "#7CFC00"
neon_pink = "#F72585"
neon_yellow = "#FFF600"
neon_bg = "#181830"
neon_orange = "#FFB347"
neon_dark = "#202037"
load_dotenv()  # load .env file

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- CONFIG ---
GROQ_MODEL = "llama3-70b-8192"
groq_client = Groq(api_key=GROQ_API_KEY)
PRODUCT_CONTEXT = (
    "You are an AI market research expert analyzing customer reviews for a chocolate-flavoured whey protein powder. "
    "Generate user personas based on patterns and diversity in the reviews."
)
CSV_PATH = "/Users/kushagraaatre/Downloads/Texpedition/data_with_text.csv"

st.set_page_config(page_title="Persona Lab", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    "<h1 style='color:#00fff7;font-size:2.6rem;font-weight:900;letter-spacing:0.01em;margin-bottom:5px;'>🎭 Persona Lab</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div style="font-size:1.21rem; color:#AC7CFF; font-weight:600; margin-top:-13px; margin-bottom:14px; line-height:1.5;">
        Ready to peek inside the minds of your customers?  
        This is your sandbox for uncovering who buys, why they rave, and what they crave—powered by real reviews and sharp AI.  
        Dive in, explore the personas that drive your market, and see your brand through their eyes (and taste buds)!
    </div>
    """,
    unsafe_allow_html=True
)

# --- NAVIGATION BUTTONS ---
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

st.markdown("""
<div style="display:flex;gap:2em;justify-content:flex-start;">
    <a href="/prt111" class="neon-btn"target="_self">🏠 Home</a>
    <a href="/newprod" class="neon-btn neon-btn-pink"target="_self">🚀 New Product Launch</a>
</div>
<br>
""", unsafe_allow_html=True)


def block_markdown(text, color):
    text = text.replace('\n', '<br>')
    return (
        f'<div style="background:linear-gradient(90deg,{color}22,#181830 90%);'
        f'padding:16px 22px;border-radius:16px;margin:10px 0 24px 0;'
        f'font-weight:600;color:#fff;font-size:1.04em;line-height:1.6;box-shadow:0 2px 24px {color}19;">'
        f'{text}</div>'
    )

@st.cache_data(show_spinner=True)
def load_reviews(csv_path):
    if not os.path.exists(csv_path):
        st.error(f"CSV file not found: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if "polarity" not in df.columns:
        try:
            from transformers import pipeline
            sa = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            df["polarity"] = df["review_text"].apply(lambda x: 1 if sa(x)[0]["label"] == "POSITIVE" else -1)
        except Exception as e:
            st.warning("Could not compute sentiment scores. All reviews set to neutral (0).")
            df["polarity"] = 0

    if "review_length" not in df.columns:
        df["review_length"] = df["review_text"].apply(lambda x: len(str(x).split()))
    return df

def generate_personas(review_texts, n_personas=4):
    prompt = (
        f"Read the following customer reviews for a chocolate-flavored whey protein powder. "
        f"Based on the language, interests, and context, segment these users into {n_personas} distinct personas. "
        "For each persona, provide:\n"
        "1. Persona Name starting with emoji\n"
        "2. A one-line summary\n"
        "3. Five detailed bullet points describing their characteristics, needs, goals, or behaviors (each bullet should be specific and insightful, not generic).\n"
        "Give the answer as a numbered list, one for each persona. Format:\n"
        "1. [Emoji] Persona Name\nSummary: ...\n- ...\n- ...\n- ...\n- ...\n- ...\n"
        "\nREVIEWS:\n" +
        "\n".join(review_texts[:120])[:3600]
    )
    try:
        chat_completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": PRODUCT_CONTEXT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=900,
            temperature=0.6,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating personas: {e}"

def parse_personas_bulletproof(llm_output, n=4):
    lines = llm_output.splitlines()
    persona_headers = []
    for i, line in enumerate(lines):
        if re.match(r"^([0-9]{1,2}[.)-]?\s*)?[\U0001F300-\U0001FAFF]", line.strip()):
            persona_headers.append(i)
    persona_blocks = []
    for idx, start in enumerate(persona_headers):
        end = persona_headers[idx+1] if idx+1 < len(persona_headers) else len(lines)
        persona_blocks.append(lines[start:end])

    personas = []
    for block in persona_blocks[:n]:
        name_line = re.sub(r"^([0-9]{1,2}[.)-]?\s*)?", "", block[0]).strip().replace("**", "")
        summary = ""
        bullets = []
        for l in block[1:]:
            l = l.strip()
            if not l: continue
            if not summary and ("summary" in l.lower() or not l.startswith(("-", "•", "*", "+"))):
                summary = re.sub(r"^summary[:\- ]*", "", l, flags=re.I)
            elif l.startswith(("-", "•", "*", "+")) or re.match(r"^[0-9]{1,2}[.)-]", l):
                b = re.sub(r"^[-•*+0-9. ]+", "", l)
                if b: bullets.append(b)
        personas.append({
            "name": name_line,
            "summary": summary,
            "bullets": bullets[:5]
        })
    return personas

def assign_review_to_persona_tfidf(df, persona_defs):
    # Use TF-IDF cosine similarity for assignment (faster than LLM for large data)
    from sklearn.feature_extraction.text import TfidfVectorizer
    persona_texts = [p["summary"] + " " + " ".join(p["bullets"]) for p in persona_defs]
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(df["review_text"].tolist() + persona_texts)
    review_vecs = X[:-len(persona_texts)]
    persona_vecs = X[-len(persona_texts):]
    assignments = []
    for i in range(review_vecs.shape[0]):
        sims = review_vecs[i].dot(persona_vecs.T).toarray().flatten()
        idx = np.argmax(sims)
        assignments.append(persona_defs[idx]["name"])
    return assignments

def groq_bullets_persona(chart_desc, chart_data_text):
    user_prompt = (
        f"Summarize as exactly two bullet points the main insights for this chart: {chart_desc}. "
        f"Here is the data: {chart_data_text}. "
        "Provide a percentage if applicable. Just facts."
    )
    try:
        chat_completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": PRODUCT_CONTEXT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=80,
            temperature=0.5,
        )
        bullets = chat_completion.choices[0].message.content.strip()
        points = [line for line in bullets.splitlines() if line.strip().startswith(("-", "•"))]
        return "\n".join(points[:2]) if len(points) >= 2 else "- " + bullets
    except Exception:
        return "- Summary not available.\n- (LLM error)"

# --- EMOTION PIPELINE (optional) ---
def emotion_pipeline(df):
    try:
        from transformers import pipeline
        emo = pipeline(
            "text-classification",
            model="finiteautomata/bertweet-base-emotion-analysis",  # much smaller than roberta-base!
            top_k=None,
            device=-1  # always use CPU, avoid meta-tensor bug
        )
    except Exception as e:
        st.warning(f"Could not load emotion model, skipping emotion analysis: {e}")
        df["main_emotion"] = "neutral"
        return df
    all_emotions = []
    for t in df["review_text"]:
        try:
            emotions = emo(t[:512])
            if isinstance(emotions, list) and len(emotions) and isinstance(emotions[0], list):
                # Sometimes returns list of lists
                emotions = emotions[0]
            main_emo = sorted(emotions, key=lambda x: -x["score"])[0]["label"]
        except Exception:
            main_emo = "neutral"
        all_emotions.append(main_emo)
    df["main_emotion"] = all_emotions
    return df


# ========== MAIN PIPELINE ========== #

with st.spinner("🔎 Analyzing your data... Please wait a few moments."):
    df = load_reviews(CSV_PATH)
    reviews = df["review_text"].dropna().tolist() if not df.empty else []
    reviews = [t for t in reviews if "unreadable" not in t and "missing" not in t and t.strip()]
    if reviews:
        personas_raw = generate_personas(reviews, 4)
        personas = parse_personas_bulletproof(personas_raw, 4)
        if personas:
            with open(PERSONA_PATH, "w", encoding="utf-8") as f:
                json.dump(personas, f, ensure_ascii=False, indent=2)
            st.session_state['personas'] = personas
            st.success(f"{len(personas)} personas saved for next use.")
    else:
        personas = []

    persona_colors = [neon_green, neon_blue, neon_pink, neon_orange]
    persona_cycler = cycle(persona_colors)
    persona_blocks = []
    persona_names = []

    # Persona grid (left-right)
    if personas:
        st.markdown("<br>", unsafe_allow_html=True)
        grid_cols = st.columns(2)
        for i, p in enumerate(personas):
            c = next(persona_cycler)
            col = grid_cols[i%2]
            with col:
                st.markdown(
                    f"<div style='background:linear-gradient(90deg,{c}18,#181830 95%);"
                    "padding:24px 26px 16px 26px;border-radius:18px;margin-bottom:24px;"
                    f"box-shadow:0 2px 22px {c}22;'>"
                    f"<h2 style='color:{c};margin-bottom:0.18em'>{p['name']}</h2>"
                    f"<div style='color:#fff;font-size:1.15em;font-weight:500;margin-bottom:10px'>Summary: {p['summary']}</div>"
                    f"<div style='color:{neon_pink};font-weight:700;font-size:1.08em;margin-bottom:2px'>Characteristics</div>"
                    f"<ul style='font-size:1.02em;margin-top:3px'>{''.join([f'<li>{b}</li>' for b in p['bullets']])}</ul>"
                    "</div>", unsafe_allow_html=True
                )
            persona_names.append(p["name"])
        st.markdown("<hr>", unsafe_allow_html=True)

    if personas and len(reviews) > 0:
        # Assign reviews to persona via TF-IDF (fast)
        persona_for_review = assign_review_to_persona_tfidf(df, personas)
        df_reviews = df.copy()
        df_reviews = df_reviews.iloc[:len(persona_for_review)].copy()
        df_reviews["persona"] = persona_for_review

        # --- Generate all summary stats for new graphs
        # 1. Persona Review Share
        persona_counts = df_reviews["persona"].value_counts()
        # 2. Persona Sentiment
        avg_sentiment = df_reviews.groupby("persona")["polarity"].mean()
        # 3. Persona Review Length
        avg_length = df_reviews.groupby("persona")["review_length"].mean()
        # 4. Persona Emotion (optional)
        if "main_emotion" not in df_reviews.columns:
            df_reviews = emotion_pipeline(df_reviews)
        emo_dist = df_reviews.groupby("persona")["main_emotion"].value_counts().unstack().fillna(0)

        # --- Row 1: Pie and Sentiment Bar
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<h3 style='color:#fff;font-size:2rem;font-weight:700;'>Sales/Review Share by Persona</h3>", unsafe_allow_html=True)
            fig = go.Figure(data=[go.Pie(labels=persona_counts.index, values=persona_counts.values, hole=0.45)])
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(block_markdown(
                groq_bullets_persona("Sales/Review Share by Persona", persona_counts.to_dict()), neon_green
            ), unsafe_allow_html=True)

        with c2:
            st.markdown("<h3 style='color:#fff;font-size:2rem;font-weight:700;'>Average Sentiment by Persona</h3>", unsafe_allow_html=True)
            fig2 = go.Figure(data=[go.Bar(x=avg_sentiment.index, y=avg_sentiment.values, marker=dict(color=[neon_green, neon_blue, neon_pink, neon_orange]))])
            fig2.update_layout(xaxis_title="Persona", yaxis_title="Avg Sentiment", font=dict(size=15))
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown(block_markdown(
                groq_bullets_persona("Average Sentiment by Persona", avg_sentiment.to_dict()), neon_blue
            ), unsafe_allow_html=True)

        # --- Row 2: Review Length and Emotion Distribution
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("<h3 style='color:#fff;font-size:2rem;font-weight:700;'>Persona vs. Review Length Distribution</h3>", unsafe_allow_html=True)
            fig3 = go.Figure(data=[go.Bar(x=avg_length.index, y=avg_length.values, marker=dict(color=[neon_green, neon_blue, neon_pink, neon_orange]))])
            fig3.update_layout(xaxis_title="Persona", yaxis_title="Avg Review Length", font=dict(size=15))
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown(block_markdown(
                groq_bullets_persona("Average review length (words) by persona", avg_length.to_dict()), neon_orange
            ), unsafe_allow_html=True)

        with c4:
            st.markdown("<h3 style='color:#fff;font-size:2rem;font-weight:700;'>Persona vs. Emotion Distribution</h3>", unsafe_allow_html=True)
            fig4 = go.Figure()
            for idx, em in enumerate(emo_dist.columns):
                fig4.add_trace(go.Bar(name=em, x=emo_dist.index, y=emo_dist[em].values))
            fig4.update_layout(barmode='stack', xaxis_title="Persona", yaxis_title="Emotion Count", font=dict(size=15))
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown(block_markdown(
                groq_bullets_persona("Distribution of primary emotions per persona", emo_dist.to_dict()), neon_pink
            ), unsafe_allow_html=True)

        # --- Persona-wise Highlights, grouped by persona with headings ---
st.markdown("<hr><h2 style='color:#fff'>Persona-wise Sentiment Highlights & Recommendations</h2>", unsafe_allow_html=True)
persona_grid = st.columns(2)

for idx, p in enumerate(personas):
    persona_df = df_reviews[df_reviews["persona"] == p["name"]]
    top_pos = persona_df[persona_df["polarity"] > 0]["review_text"].head(2).tolist()
    top_neg = persona_df[persona_df["polarity"] < 0]["review_text"].head(2).tolist()
    pos_summary = groq_bullets_persona(
        f"Summarize two main positive sentiment points, with percentage, for persona '{p['name']}'.",
        " ".join(top_pos)
    ) if top_pos else "No positive reviews."
    neg_summary = groq_bullets_persona(
        f"Summarize two main negative sentiment points, with percentage, for persona '{p['name']}'.",
        " ".join(top_neg)
    ) if top_neg else "No negative reviews."

    rec_prompt = (
    f"You are a product marketing strategist. "
    f"Based on the review highlights and persona details for '{p['name']}' "
    f"(do not repeat the characteristics), write one concise or mention name of user, actionable product or marketing recommendation. Dont put * anywhere "
    f"for the company to better engage this persona. "
    f"Focus on practical actions the business can take (such as messaging, offers, features, or campaigns). "
    f"Reply with 1-2 sentences, avoid restating the persona’s traits."
    )

    try:
        rec_out = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": PRODUCT_CONTEXT},
                {"role": "user", "content": rec_prompt}
            ],
            max_tokens=80, temperature=0.5
        ).choices[0].message.content.strip()
    except:
        rec_out = "No recommendation available."

    with persona_grid[idx % 2]:
        st.markdown(
            f"<div style='margin-bottom:38px;padding:18px 20px 8px 20px;border-radius:18px;"
            f"background:linear-gradient(90deg,{persona_colors[idx%4]}22,#181830 100%);box-shadow:0 2px 22px {persona_colors[idx%4]}18;'>"
            f"<h2 style='color:{persona_colors[idx%4]};font-size:1.35em;margin-bottom:0.3em'>{p['name']}</h2>"
            f"<div style='color:#fff;font-size:1.13em;font-weight:400;margin-bottom:14px;'>{p['summary']}</div>"
            "<div style='margin-bottom:16px'>"
            f"<b style='color:{neon_green};font-size:1.1em;'>Top Positive Sentiments:</b><br>{block_markdown(pos_summary, neon_green)}"
            "</div>"
            "<div style='margin-bottom:16px'>"
            f"<b style='color:{neon_pink};font-size:1.1em;'>Top Negative Sentiments:</b><br>{block_markdown(neg_summary, neon_pink)}"
            "</div>"
            "<div>"
            f"<b style='color:{neon_yellow};font-size:1.1em;'>Recommendation:</b><br>{block_markdown(rec_out, neon_yellow)}"
            "</div>"
            "</div>", unsafe_allow_html=True
        )

   
st.markdown("---")
st.markdown(
    f"<small style='color:{neon_yellow}'>Powered By Bugs Fring</small>",
    unsafe_allow_html=True
)
