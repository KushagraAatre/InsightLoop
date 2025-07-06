import streamlit as st
import base64
import os

st.set_page_config(
    page_title="InsightLoop",
    page_icon="🔁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOGO ---
def get_logo_base64(logo_path):
    if not os.path.exists(logo_path):
        return None
    with open(logo_path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode("utf-8")
    return b64

logo_path = "logo.png"
logo_b64 = get_logo_base64(logo_path)

if logo_b64:
    st.markdown(
        f"""
        <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;margin-bottom:0.2em;'>
            <img src="data:image/png;base64,{logo_b64}" alt="Logo" style="width:260px;height:260px;object-fit:contain;filter:drop-shadow(0 0 26px #00fff7cc);"/>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;margin-bottom:0.2em;'>
            <span style='font-size:2.9rem;font-weight:900;color:#00fff7;letter-spacing:-2px;margin-top:0;'>InsightLoop</span>
        </div>
        """, unsafe_allow_html=True
    )

# --- GLASSY INTRO BOX ---
st.markdown("""
<div style="
    background: rgba(32,40,52,0.96); 
    border: 2.3px solid #00fff7cc;
    border-radius: 24px;
    margin: 1.3em auto 2.1em auto;
    max-width: 1060px;
    min-width: 420px;
    padding: 38px 36px 20px 36px;
    box-shadow:0 14px 68px #00fff799;
    color: #fff;
    font-size: 1.03rem;
    line-height:1.63;
    font-family: 'Montserrat', sans-serif;">
    <span style="font-size:1.38rem;font-weight:900;background:linear-gradient(90deg,#00fff7,#F72585 92%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Stop guessing. Start knowing—every persona, every opinion, every launch.
    </span>
    <br><br>
    <b>InsightLoop</b> isn’t just another dashboard. It’s your on-demand product research team—powered by AI, fueled by every kind of customer feedback.
    <br><br>
    <div style="display:grid;grid-template-columns:38px 1fr;gap:10px 13px;align-items:start;">
      <div style="font-size:1.6rem;">🔀</div>
      <div>
        <b style="color:#00fff7;">Multimodal Magic</b><br>
        Bring your reviews, screenshots, or even audio rants—InsightLoop reads, listens, and <i>sees</i> to give you the whole customer picture. 
        No more missing context, no more “one channel” blind spots.
      </div>
      <div style="font-size:1.6rem;">🧬</div>
      <div>
        <b style="color:#7CFC00;">Persona Discovery</b><br>
        Our AI auto-sorts your users into real-world groups—think loyalists, price hawks, fitness geeks, skeptics, and more.
        It finds them by their words, their tone, their buying patterns—even if you’ve never tagged or labeled a thing.
      </div>
      <div style="font-size:1.6rem;">⚡</div>
      <div>
        <b style="color:#F72585;">Instant, Actionable Insights</b><br>
        Get beautiful, interactive dashboards for <i>every</i> persona: see what excites them, what drives them nuts, and what’ll keep them coming back (or send them running). All charted, all explained—no stats degree needed.
      </div>
      <div style="font-size:1.6rem;">🚀</div>
      <div>
        <b style="color:#FFB347;">Launch Simulation Superpower</b><br>
        Curious about a new product or feature? Instantly predict how each persona will react—with auto-generated marketing hooks and push notifications written <i>for them</i>, by AI. 
        Turn “blind launch” into “bullseye launch.”
      </div>
    </div>
    <br>
    <span style="color:#AC7CFF;font-weight:700;">
        Fully automated. Zero spreadsheets.<br>
        Just drop your data, and let InsightLoop surface the real market signals you’ve been missing.
    </span>
    <br><br>
    <span style="color:#fff;font-weight:500;font-size:1.01em;">
        <b>Why does this matter?</b> Most brands guess. The best brands <i>know</i>. InsightLoop makes it effortless to know.
    </span>
</div>
""", unsafe_allow_html=True)

# --- GLOW BUTTON (Streamlit native for navigation) ---
st.markdown("""
    <style>
    div.stButton > button.insight-glow-btn {
        font-family: 'Montserrat',sans-serif;
        font-size: 1.19em;
        font-weight: 900;
        color: #181830;
        background: linear-gradient(90deg,#00fff7 10%,#7CFC00 90%);
        padding: 15px 46px 13px 46px;
        border: 2px solid #00fff7;
        border-radius: 16px;
        box-shadow: 0 0 18px #00fff799, 0 0 36px #7CFC0055;
        cursor: pointer;
        transition: all 0.11s cubic-bezier(.31,1.3,.7,1);
        letter-spacing: 0.01em;
        margin: 0 auto;
        outline: none;
        display: block;
    }
    div.stButton > button.insight-glow-btn:hover, 
    div.stButton > button.insight-glow-btn:focus {
        transform: scale(1.045);
        background: linear-gradient(90deg,#7CFC00 10%,#00fff7 90%);
        color: #222;
        box-shadow: 0 0 32px #00fff7cc, 0 0 36px #7CFC00cc;
    }
    </style>
""", unsafe_allow_html=True)

center = """
    <div style='display: flex; justify-content: center; align-items: center; margin: 1.7em auto 0.7em auto;'>
        {content}
    </div>
"""

# Place this where you want your big button (centered and as wide as your boxes):

st.markdown("""
<style>
.custom-glow-btn-main {
    display: block;
    width: 100%;
    max-width: 1060px;
    min-width: 420px;
    margin: 2em auto 1.1em auto;
    font-family: 'Montserrat',sans-serif;
    font-size: 1.26em;
    font-weight: 900;
    color: #181830 !important;
    background: linear-gradient(90deg,#00fff7 12%,#7CFC00 88%);
    padding: 21px 0 18px 0;
    border: 2.5px solid #00fff7cc;
    border-radius: 17px;
    box-shadow: 0 0 24px #00fff7b8, 0 0 46px #7CFC0090;
    text-align: center;
    text-decoration: none !important;
    transition: all 0.13s cubic-bezier(.31,1.3,.7,1);
    cursor: pointer;
    outline: none;
}
.custom-glow-btn-main:hover, .custom-glow-btn-main:focus {
    background: linear-gradient(90deg,#7CFC00 14%,#00fff7 86%);
    color: #19191f !important;
    box-shadow: 0 0 38px #00fff7cc, 0 0 56px #7CFC00cc;
    transform: scale(1.018);
}
</style>
<div style="display:flex;justify-content:center;">
    <a href="/prt111" class="custom-glow-btn-main" target="_self">💡 Let’s Start</a>
</div>
""", unsafe_allow_html=True)

# --- HEADS UP & DEMO BOXES (smaller) ---
st.markdown("""
<style>
.insight-centerbox, .insight-demo {
    margin: 1.5em auto 0.3em auto;
    background: rgba(22,26,38,0.97);
    border-radius: 19px;
    box-shadow: 0 5px 30px #00fff73c;
    font-family: 'Montserrat', sans-serif;
    text-align: center;
    max-width: 1060px;
    min-width: 420px;
    padding: 20px 30px 17px 30px;
    border: none;
}
.insight-centerbox h4 {
    color: #7CFC00;
    font-size: 1.06em;
    margin-bottom: 0.45em;
    font-weight: 900;
    letter-spacing: 0.02em;
}
.insight-centerbox p {
    color: #7CFC00;
    font-size: 0.99em;
    font-weight: 600;
    margin-bottom: 0;
}
.insight-centerbox i {
    color: #A0FFB2;
    font-size: 0.97em;
}
.insight-demo {
    background: rgba(0,30,40,0.94);
    color: #00fff7;
    box-shadow: 0 4px 18px #00fff75c;
    border-radius: 19px;
    margin-top: 1.2em;
    max-width: 1060px;
    min-width: 420px;
    padding: 18px 32px 13px 32px;
    text-align: center;
    font-size: 0.97em;
}
.insight-demo b {
    color: #7CFC00;
    font-weight: 900;
    font-size: 1.06em;
}
.insight-demo .demo-product {
    color: #FFB347;
    font-weight: 700;
}
</style>
<div class="insight-centerbox">
    <h4>🤖 Heads up: InsightLoop is working hard behind the scenes!</h4>
    <p>
    Analyzing images and audio takes a few seconds—so just sit back, relax, and let the AI do the heavy lifting.<br>
    <i>You’ll have deeper insights in less time than it takes to reheat your coffee. ☕🦾</i>
    </p>
</div>
<div class="insight-demo">
    <b>🚀 Demo mode:</b> <span class="demo-product">This dashboard is running on real chocolate protein powder reviews—just to show what’s possible.</span><br>
    InsightLoop adapts to <b>any</b> product or dataset. When you’re ready, upload your own and watch the magic happen.
</div>
""", unsafe_allow_html=True)

# --- POWERED BY SIGN-OFF ---
st.markdown(
    """
    <div style="margin-top:1.7em;margin-bottom:0.5em;text-align:center;">
        <span style="font-size:1.09em;color:#F72585;font-weight:700;">Powered by Bugs Fring</span>
    </div>
    """,
    unsafe_allow_html=True
)
