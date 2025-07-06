import streamlit as st
import json
import os
import numpy as np
import plotly.graph_objs as go
from groq import Groq
from dotenv import load_dotenv
load_dotenv()  # load .env file

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- CONFIG ---
GROQ_MODEL = "llama3-70b-8192"
groq_client = Groq(api_key=GROQ_API_KEY)
PERSONA_PATH = "personas.json"

# --- THEME COLORS ---
neon_blue = "#00fff7"
neon_green = "#7CFC00"
neon_pink = "#F72585"
neon_cyan = "#0ffcff"
neon_bg = "#181830"
neon_orange = "#FFB347"
neon_shadow = "#2dfdff44"

font_main = "Inter, Segoe UI, Arial, sans-serif"

st.set_page_config(page_title="🚀 New Launch Studio", layout="wide", initial_sidebar_state="collapsed")

# --- STYLE ---
st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        background-color: {neon_bg} !important;
        font-family: {font_main} !important;
    }}
    /* HEADERS */
    .neon-title {{
        font-size:2.8rem; font-weight:900; color:{neon_blue};
        letter-spacing:0.02em; margin-bottom:7px; margin-top:6px;
        text-shadow:0 2px 24px {neon_blue}33;
    }}
    .neon-sub {{
        font-size:1.25rem;font-weight:600;color:#fff;
        margin-bottom:2px;margin-top:0px;
    }}
    .neon-heads-up {{
        font-size:1.08rem;color:{neon_pink};font-weight:700;margin-bottom:32px;
        margin-top:7px;
    }}

    /* BUTTONS */
    .neon-btn {{
        display:inline-block;
        font-weight:bold;
        padding:13px 32px;
        border:none;
        border-radius:13px;
        font-size:1.10em;
        margin-right:16px;
        cursor:pointer;
        box-shadow:0 0 13px {neon_blue}55;
        color:#1d1d1d !important;
        background:linear-gradient(90deg,{neon_green}, {neon_blue});
        text-decoration:none !important;
        transition:transform 0.10s;
    }}
    .neon-btn-pink {{
        background:linear-gradient(90deg,{neon_pink}, {neon_blue});
        color:#fff !important;
        box-shadow:0 0 16px {neon_pink}88;
    }}
    .neon-btn:hover {{ transform:scale(1.06); }}

    /* PERSONA NAME BOX */
    .persona-name-box {{
        background: linear-gradient(90deg, {neon_blue}, {neon_pink} 80%);
        color: #15192A;
        font-size:2.2rem;
        font-weight:900;
        border-radius:28px;
        padding: 12px 40px 10px 25px;
        margin-bottom:15px;
        display: inline-block;
        box-shadow: 0 2px 26px {neon_cyan}99;
        letter-spacing:0.01em;
        margin-top:18px;
    }}

    /* PERSONA CARD CONTENTS */
    .persona-section-row {{
        display: flex;
        gap: 2.5em;
        margin-bottom: 0;
    }}

    .persona-section-col {{
        flex: 1;
        min-width: 340px;
    }}

    /* LABELS */
    .block-label {{
        font-weight:900;
        font-size:1.15em;
        margin-bottom:8px;
        margin-top:8px;
        letter-spacing:0.01em;
        display:flex;
        align-items:center;
        gap:0.6em;
    }}
    .label-blue {{ color:{neon_blue}; }}
    .label-green {{ color:{neon_green}; }}
    .label-pink {{ color:{neon_pink}; }}
    .label-orange {{ color:{neon_orange}; }}
    .label-cyan {{ color:{neon_cyan}; }}

    /* BULLET LISTS */
    ul.insight-list {{
        margin-top:7px; margin-bottom:16px;
        padding-left:22px;
    }}
    ul.insight-list li {{
        font-size:1.11em; font-weight:500; color:#fff;
        margin-bottom:5px; line-height:1.53;
    }}

    /* INTEREST & NOTIF */
    .interest-badge {{
        display:inline-block;
        background:linear-gradient(90deg, {neon_green}, {neon_blue} 90%);
        color:#15192A; font-size:1.09em; font-weight:900;
        border-radius:15px; padding:8px 30px 7px 18px;
        margin-right:14px;
        box-shadow:0 0 17px {neon_green}2c;
        margin-top:10px;
    }}
    .notification-block {{
        background:linear-gradient(90deg,{neon_cyan}44,#232344 96%);
        border-left:5px solid {neon_blue};
        padding:17px 23px 17px 23px;
        border-radius:14px;
        font-weight:700;
        color:{neon_blue};
        font-size:1.06em;
        line-height:1.45;
        box-shadow:0 2px 18px {neon_cyan}1a;
        margin-bottom:8px;
        margin-top:10px;
        letter-spacing:0.01em;
        max-width:430px;
        min-width: 240px;
        display: inline-block;
    }}

    /* CHART/INSIGHT CARDS */
    .section-card {{
        background:rgba(23,28,49,0.97);
        border-radius: 17px;
        box-shadow:0 0 22px {neon_cyan}32;
        padding: 34px 42px 22px 42px;
        margin-bottom:36px;
        margin-top:16px;
    }}

    /* COMBINED INSIGHTS */
    .insight-box {{
        background:rgba(23,28,49,0.98);
        border-radius: 18px;
        box-shadow:0 0 26px {neon_blue}45;
        padding: 32px 34px 18px 34px;
        margin-bottom:33px;
        margin-top:20px;
    }}

    /* SUMMARY BOX */
    .summary-box {{
        background:rgba(23,28,49,0.97);
        border-radius: 15px;
        box-shadow:0 0 22px {neon_green}32;
        padding: 32px 38px 22px 38px;
        margin-bottom:36px;
        margin-top:14px;
        color:#fff;
        font-size:1.17em;
    }}

    /* RESPONSIVE */
    @media (max-width: 1000px) {{
      .persona-section-row {{ flex-direction: column; }}
      .persona-section-col {{ min-width: 100%; }}
    }}
    </style>
""", unsafe_allow_html=True)

# --- TITLE & DESCRIPTION ---
st.markdown(f"<div class='neon-title'>🚀 New Launch Studio</div>", unsafe_allow_html=True)
st.markdown(f"<div class='neon-sub'>Will your next product idea actually vibe with your audience? Pop your concept below and instantly see what your customer personas think—no fluff, just punchy, actionable feedback and a reality check on your launch.</div>", unsafe_allow_html=True)
st.markdown(f"<div class='neon-heads-up'>⚡ Heads up: Our demo and market data is based on protein powder reviews—so for best results, enter a health, nutrition, or supplement product!</div>", unsafe_allow_html=True)

# --- NAVIGATION BUTTONS ---
st.markdown(f"""
<div style="display:flex;gap:2em;justify-content:flex-start;margin-bottom:6px;">
    <a href="/prt111" class="neon-btn" target="_self">🏠 Home</a>
    <a href="/persona" class="neon-btn neon-btn-pink" target="_self">👤 Persona Analysis</a>
</div>
""", unsafe_allow_html=True)

# --- PRODUCT DESCRIPTION INPUT ---
st.markdown(f"<h2 style='color:{neon_blue};font-size:2.04rem;font-weight:900;margin-top:30px;margin-bottom:7px;'>1. Describe Your New Product</h2>", unsafe_allow_html=True)
product_desc = st.text_area(
    "",
    height=110,
    placeholder="E.g. Introducing VanillaWhey: zero sugar, 25g protein, added digestive enzymes, eco-packaging, smooth vanilla flavor, perfect for fitness and daily wellness."
)

# --- LOAD PERSONAS ---
if os.path.exists(PERSONA_PATH):
    with open(PERSONA_PATH, "r", encoding="utf-8") as f:
        personas = json.load(f)
else:
    personas = []
    st.warning("No personas found. Please generate personas first in the Persona Analysis page.")

def clean_points(text, max_points=2):
    lines = [l for l in text.replace('\r', '\n').split('\n') if l.strip() and not l.strip().lower().startswith(
        ('here is', 'here are', 'persona:', 'this is', 'for this persona', 'concerns:', 'the following', 'alignment:', '*', 'point'))]
    points = []
    for l in lines:
        l = l.lstrip('-•1234567890. ').strip()
        if l and len(points) < max_points:
            points.append(l)
    return points if points else [text.strip()]

def ai_points(prompt, max_points=2, max_tokens=120):
    try:
        chat_completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system",
                 "content": f"You are a market research strategist. Reply with ONLY exactly {max_points} very brief, but fully written bullet points—no intros, no repetition, no generic phrases. Each point should be a full, clear sentence. Never add 'Here are' or any extra intro. Dont mention any names."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens, temperature=0.7, stop=None
        )
        return clean_points(chat_completion.choices[0].message.content.strip(), max_points)
    except Exception as e:
        return [f"Error: {e}"]

def ai_notification(prompt, max_tokens=44):
    try:
        chat_completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are a copywriter. Write a single, short, energetic notification or email (max 30 words, no names, no symbols), ending with a call-to-action. Make it stand out and complete."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens, temperature=0.72, stop=None
        )
        return chat_completion.choices[0].message.content.strip().replace("**", "")
    except Exception as e:
        return f"Error: {e}"

def ai_percent(prompt):
    try:
        chat_completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": "You are a market research strategist."}, {"role": "user", "content": prompt}],
            max_tokens=8, temperature=0.25
        )
        s = chat_completion.choices[0].message.content.strip()
        percent = ''.join([c for c in s if c.isdigit()])
        return percent + "%" if percent else s
    except Exception as e:
        return "?"

def ai_graph_insights(prompt, max_tokens=160):
    try:
        chat_completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are a market analyst. Give only 4 numbered, very concise but meaningful insights in separate sentences, no intro line or extra formatting, no 'Here are', no asterisks or stars, just the facts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens, temperature=0.7, stop=None
        )
        # Always keep only 4, no prefix text
        lines = [l.lstrip('-•1234567890. ').strip().replace("**", "") for l in chat_completion.choices[0].message.content.strip().split('\n') if l.strip()]
        return lines[:4]
    except Exception as e:
        return [f"Error: {e}"]

def ai_summary(prompt, max_tokens=90):
    try:
        chat_completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system",
                 "content": "Write a concise, professional executive summary in 3 sentences. No intro lines, no 'Here is', no asterisks. Be direct and to the point."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens, temperature=0.7, stop=None
        )
        return chat_completion.choices[0].message.content.strip().replace("**", "")
    except Exception as e:
        return f"Error: {e}"

st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

# --- GENERATE BUTTON ---
test_btn = st.button(
    "🚦 Run Persona–Product Fit Check",
    help="Instantly see AI-powered feedback from every persona's perspective!",
    use_container_width=True
)
st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

if test_btn and product_desc and personas:
    st.markdown(f"<h2 style='color:{neon_blue};font-size:2.23rem;font-weight:900;margin-bottom:12px;margin-top:17px;'>2. Persona-by-Persona Results</h2>", unsafe_allow_html=True)
    persona_colors = [neon_blue, neon_green, neon_pink, neon_orange, neon_cyan]
    persona_cycle = iter(persona_colors)
    section_icons = {
        "Probable Reaction": "💡",
        "Alignment with Persona": "✅",
        "Potential Mismatches or Concerns": "⚠️",
        "Marketing Strategy": "📢",
        "Personalized Notification": "🔔",
    }

    def persona_block(persona, color):
        return st.container()

    # Pair personas 2 per row
    for i in range(0, len(personas), 2):
        cols = st.columns(2, gap="large")
        for j, col in enumerate(cols):
            if i + j < len(personas):
                persona = personas[i + j]
                color = next(persona_cycle, neon_blue)
                with col:
                    st.markdown(f"<div class='persona-name-box' style='background:linear-gradient(90deg,{neon_blue},{neon_pink} 80%);margin-bottom:16px;'><span>{persona.get('icon','')} {persona['name']}</span></div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='height:4px;'></div>", unsafe_allow_html=True)
                    st.markdown("<div class='persona-section-row'>", unsafe_allow_html=True)
                    st.markdown("<div class='persona-section-col'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='block-label label-blue'>{section_icons['Probable Reaction']} Probable Reaction</div>", unsafe_allow_html=True)
                    reactions = ai_points(
                        f"Summarize two brief but complete points for this persona's likely reaction to the product: {product_desc}. Use clear, direct language.",
                        max_points=2, max_tokens=90
                    )
                    st.markdown(f"<ul class='insight-list'>" + "".join([f"<li>{r}</li>" for r in reactions]) + "</ul>", unsafe_allow_html=True)
                    st.markdown(f"<div class='block-label label-green'>{section_icons['Alignment with Persona']} Alignment with Persona</div>", unsafe_allow_html=True)
                    aligns = ai_points(
                        f"List two specific ways this persona's characteristics or needs will match with the features or benefits of the product: {product_desc}. "
                        f"Be explicit: mention which part of the persona is satisfied by which product feature. Use clear, direct language.",
                        max_points=2, max_tokens=100
                    )
                    st.markdown(f"<ul class='insight-list'>" + "".join([f"<li>{a}</li>" for a in aligns]) + "</ul>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("<div class='persona-section-col'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='block-label label-pink'>{section_icons['Potential Mismatches or Concerns']} Potential Mismatches or Concerns</div>", unsafe_allow_html=True)
                    mismatches = ai_points(
                        f"List two precise concerns or mismatches: Which features or aspects of the {product_desc} may NOT align with this persona's preferences or needs? "
                        f"Be explicit: mention which product feature is likely to be a turn-off or ignored by this persona.",
                        max_points=2, max_tokens=100
                    )

                    st.markdown(f"<ul class='insight-list'>" + "".join([f"<li>{m}</li>" for m in mismatches]) + "</ul>", unsafe_allow_html=True)
                    st.markdown(f"<div class='block-label label-orange'>{section_icons['Marketing Strategy']} Marketing Strategy</div>", unsafe_allow_html=True)
                    strategy = ai_points(
                        f"Suggest two creative, product-specific marketing strategies targeted at this persona for this product: {product_desc}. "
                        f"Each point must clearly connect a product feature with a unique marketing approach for this persona.",
                        max_points=2, max_tokens=100
                    )

                    st.markdown(f"<ul class='insight-list'>" + "".join([f"<li>{s}</li>" for s in strategy]) + "</ul>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div style='display:flex;align-items:center;gap:22px;margin-top:12px;margin-bottom:22px;'>
                            <span class='interest-badge'>Interest Likelihood: {ai_percent('Estimate the likelihood (percent) that '+persona['name']+' would be interested in this product. Just the number and % sign, nothing else.')}</span>
                            <div>
                                <div class='block-label label-cyan' style='margin-bottom:3px;'>{section_icons['Personalized Notification']} Personalized Notification</div>
                                <div class='notification-block'>{ai_notification(
                                    f"Write a concise, energetic notification or email about this product: {product_desc} aimed specifically at the persona {persona['name']}. "
                                    f"Address their top motivations and finish with a strong call-to-action. No names, no symbols."
                                )}</div>

                            
                        </div>
                        """, unsafe_allow_html=True
                    )
                    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)

    # --- CHARTS (Demo) ---
    st.markdown(f"<h2 style='color:{neon_cyan};font-size:2.1rem;font-weight:800;margin-top:32px;'>3. Projected Market Impact</h2>", unsafe_allow_html=True)
    persona_names = [p['name'] for p in personas]
    np.random.seed(42)
    projected_market_share = np.random.dirichlet(np.ones(len(persona_names)), size=1)[0]
    projected_sentiment = projected_market_share * 0.6 + np.random.rand(len(persona_names)) * 0.4  # correlation

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div style='font-size:1.17em;color:{neon_blue};font-weight:700;margin-bottom:6px;'>Projected Market Share by Persona</div>", unsafe_allow_html=True)
        fig1 = go.Figure(data=[go.Pie(labels=persona_names, values=projected_market_share, hole=0.45)])
        fig1.update_traces(textinfo='percent+label')
        fig1.update_layout(margin=dict(l=14, r=14, b=14, t=14), showlegend=True)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.markdown(f"<div style='font-size:1.17em;color:{neon_orange};font-weight:700;margin-bottom:6px;'>Projected Sentiment by Persona</div>", unsafe_allow_html=True)
        fig2 = go.Figure(data=[go.Bar(x=persona_names, y=projected_sentiment,
                                      marker=dict(color=[neon_green, neon_blue, neon_pink, neon_orange, neon_cyan][:len(persona_names)]))])
        fig2.update_layout(xaxis_title="Persona", yaxis_title="Projected Sentiment", font=dict(size=15))
        st.plotly_chart(fig2, use_container_width=True)

    # --- Combined Chart Insights ---
    combined_prompt = (
        f"Given the projected market share {list(np.round(projected_market_share*100,1))} percent and projected sentiment {list(np.round(projected_sentiment*100,1))} for these personas: {', '.join(persona_names)}, "
        "summarize 4 concise points that correlate the two charts and reveal the most important market insights. Each point should be in a new line and fully written."
    )
    insights = ai_graph_insights(combined_prompt, max_tokens=200)
    st.markdown(
        f"<div class='insight-box'><div style='font-size:1.18em;color:{neon_blue};font-weight:700;margin-bottom:10px;'>Key Combined Insights</div>"
        f"<ul class='insight-list'>" + "".join([f"<li>{bp}</li>" for bp in insights]) + "</ul></div>", unsafe_allow_html=True
    )

    # --- OVERALL SUMMARY ---
    st.markdown(f"<h2 style='color:{neon_green};font-size:2rem;font-weight:900;margin-top:18px;'>4. Overall Summary</h2>", unsafe_allow_html=True)
    overall_prompt = (
        f"Given these personas: {', '.join([p['name'] for p in personas])}, and the new product: {product_desc}, "
        "write a concise executive summary (3 sentences, no intro, no asterisks), focusing on overall fit, the main challenge, and the best next move for launch."
    )
    summary_text = ai_summary(overall_prompt, max_tokens=1000)
    st.markdown(
        f"<div class='summary-box'>{summary_text}</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

elif test_btn:
    st.warning("Please enter your product description to see the results.")

# --- FOOTER ---
st.markdown(
    f"<small style='color:{neon_pink};font-size:1.09em;'>Powered by Bugs Fring</small>",
    unsafe_allow_html=True
)