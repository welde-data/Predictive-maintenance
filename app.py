"""
Predictive Maintenance Dashboard · AI4I 2020
Stack : Databricks · Delta Lake · PySpark ML · Gemini 2.0 Flash · Streamlit · Whisper
Run   : streamlit run app.py
Secrets: GEMINI_API_KEY = "AIza..."
Install: pip install openai-whisper audio-recorder-streamlit reportlab
         Windows: C:\ffmpeg\bin on PATH
"""
import os, io, json, warnings, tempfile
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, date
warnings.filterwarnings("ignore")

# ── Optional dependencies ─────────────────────────────────────────
try:
    import google.generativeai as genai
    _KEY = os.getenv("GEMINI_API_KEY", "")
    if _KEY:
        genai.configure(api_key=_KEY)
        _GEMINI = genai.GenerativeModel("gemini-2.0-flash")
        GEMINI_OK = True
    else:
        GEMINI_OK = False
except:
    GEMINI_OK = False

try:
    import whisper as _wlib
    WHISPER_OK = True
except:
    WHISPER_OK = False

try:
    from audio_recorder_streamlit import audio_recorder
    RECORDER_OK = True
except:
    RECORDER_OK = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable)
    REPORTLAB_OK = True
except:
    REPORTLAB_OK = False

@st.cache_resource(show_spinner="Loading Whisper model…")
def load_whisper():
    import whisper
    return whisper.load_model("base")

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="PredMaint · AI4I 2020",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens — light theme ──────────────────────────────────
BG      = "#F8F9FB"   # page background
CARD    = "#FFFFFF"   # card / surface
SURFACE = "#F0F3F7"   # input / elevated surface
BORDER  = "#D8E0EA"   # borders
BORDER2 = "#B0BEC8"   # stronger border

INK     = "#1A1F2E"   # primary text — near black
BODY    = "#2E3A4A"   # body text
SUB     = "#4A5C6A"   # secondary text
DIM     = "#8096A8"   # muted labels
WHITE   = "#FFFFFF"
LIGHT   = "#1A1F2E"   # reused as primary text in HTML templates

AMBER   = "#D4860A"   # accent — darker amber readable on white
GREEN   = "#007A50"   # success — darker green readable on white
RED     = "#C0392B"   # danger — darker red readable on white
BLUE    = "#1A6FA8"   # info
PURPLE  = "#6B4FBB"   # AI box
TEAL    = "#007A6E"   # voice / teal
MED     = SUB         # alias — some templates still use MED

# ── CSS — light theme ─────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500&display=swap');

html,body,.stApp,.main,.main .block-container{{
    background:{BG}!important;font-family:'DM Sans',sans-serif!important;color:{INK}!important}}
.main .block-container{{padding:0 2.5rem 3rem!important;max-width:100%!important}}

/* ── Sidebar ── */
section[data-testid="stSidebar"]>div:first-child{{
    background:{CARD}!important;border-right:1px solid {BORDER}!important}}
section[data-testid="stSidebar"] *{{
    background:transparent!important;color:{INK}!important;font-family:'DM Sans',sans-serif!important}}
section[data-testid="stSidebar"] [data-testid="stSlider"] label,
section[data-testid="stSidebar"] [data-testid="stSlider"] label p{{
    color:{BODY}!important;font-size:0.82rem!important;font-weight:500!important}}
section[data-testid="stSidebar"] [data-testid="stSlider"] output{{
    color:{AMBER}!important;font-weight:700!important;font-family:'DM Mono',monospace!important}}
section[data-testid="stSidebar"] button{{
    background:{SURFACE}!important;color:{INK}!important;border:1px solid {BORDER2}!important;
    border-radius:6px!important;font-family:'DM Mono',monospace!important;font-size:0.7rem!important;
    letter-spacing:0.06em!important;text-transform:uppercase!important;width:100%!important;padding:10px!important}}
section[data-testid="stSidebar"] button:hover{{
    background:{AMBER}!important;color:{WHITE}!important;border-color:{AMBER}!important}}
section[data-testid="stSidebar"] hr{{border-color:{BORDER}!important;margin:14px 0!important}}
section[data-testid="stSidebar"] [data-testid="stFileUploader"],
section[data-testid="stSidebar"] [data-testid="stFileUploader"]>div,
section[data-testid="stSidebar"] [data-testid="stFileUploader"]>div>div{{
    background:{SURFACE}!important;border:1px dashed {BORDER2}!important;border-radius:8px!important}}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] p,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] span{{
    color:{SUB}!important;font-size:0.72rem!important;font-family:'DM Mono',monospace!important}}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button{{
    background:{SURFACE}!important;color:{AMBER}!important;border:1px solid {AMBER}!important;
    border-radius:4px!important;font-size:0.68rem!important;padding:6px 10px!important;width:auto!important}}

/* ── Tabs — high contrast, clearly distinct ── */
[data-testid="stTabs"] [role="tablist"]{{
    border-bottom:2px solid {BORDER}!important;
    background:{SURFACE}!important;
    gap:0!important;
    border-radius:8px 8px 0 0!important;
    padding:4px 4px 0!important}}
[data-testid="stTabs"] [role="tab"]{{
    font-family:'DM Mono',monospace!important;
    font-size:0.78rem!important;
    letter-spacing:0.08em!important;
    text-transform:uppercase!important;
    color:{SUB}!important;
    background:transparent!important;
    border:none!important;
    border-radius:6px 6px 0 0!important;
    padding:12px 20px!important;
    font-weight:600!important;
    transition:all 0.15s!important}}
[data-testid="stTabs"] [role="tab"]:hover{{
    color:{INK}!important;
    background:{BORDER}!important}}
[data-testid="stTabs"] [aria-selected="true"]{{
    color:{WHITE}!important;
    background:{AMBER}!important;
    border-bottom:none!important;
    font-weight:800!important;
    font-size:0.82rem!important}}
[data-testid="stTabs"] [role="tabpanel"]{{
    background:transparent!important;
    padding-top:22px!important}}

/* ── Form controls ── */
[data-testid="stSelectbox"] label p,[data-testid="stTextInput"] label p,
[data-testid="stNumberInput"] label p,[data-testid="stTextArea"] label p,
[data-testid="stDateInput"] label p{{
    color:{BODY}!important;font-family:'DM Mono',monospace!important;font-size:0.7rem!important;
    letter-spacing:0.08em!important;text-transform:uppercase!important;font-weight:600!important}}
[data-testid="stSelectbox"]>div,[data-testid="stTextInput"]>div>div,
[data-testid="stNumberInput"]>div,[data-testid="stTextArea"]>div>div,
[data-testid="stDateInput"]>div{{
    background:{WHITE}!important;border:1px solid {BORDER}!important;
    color:{INK}!important;border-radius:8px!important}}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea{{
    color:{INK}!important;font-family:'DM Mono',monospace!important;background:transparent!important}}
[data-testid="stFileUploader"]{{
    background:{SURFACE}!important;border:1px dashed {BORDER2}!important;
    border-radius:10px!important;padding:8px!important}}
audio{{width:100%;border-radius:6px;margin:8px 0}}
[data-testid="stDataFrame"]{{border:1px solid {BORDER}!important;border-radius:8px!important}}

/* ── Buttons ── */
.stButton>button{{
    background:{WHITE}!important;color:{INK}!important;border:1px solid {BORDER2}!important;
    border-radius:6px!important;font-family:'DM Mono',monospace!important;font-size:0.72rem!important;
    letter-spacing:0.06em!important;text-transform:uppercase!important;transition:all 0.15s!important}}
.stButton>button:hover{{background:{AMBER}!important;color:{WHITE}!important;border-color:{AMBER}!important}}
[data-testid="stDownloadButton"] button{{
    background:transparent!important;color:{GREEN}!important;border:1px solid {GREEN}!important;
    border-radius:6px!important;font-family:'DM Mono',monospace!important;font-size:0.72rem!important}}
[data-testid="stDownloadButton"] button:hover{{background:{GREEN}!important;color:{WHITE}!important}}
[data-testid="stInfo"]{{background:rgba(26,111,168,0.08)!important;border:1px solid {BLUE}!important;
    border-radius:8px!important;color:{INK}!important}}
[data-testid="stSpinner"] p{{color:{SUB}!important;font-family:'DM Mono',monospace!important}}
.audio-recorder{{background:{SURFACE}!important;border:2px solid {TEAL}!important;
    border-radius:50%!important;box-shadow:0 0 14px rgba(0,122,110,0.2)!important}}
.audio-recorder:hover{{box-shadow:0 0 24px rgba(0,122,110,0.4)!important}}
</style>""", unsafe_allow_html=True)

# ── Plotly theme — light ──────────────────────────────────────────
BASE_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=WHITE,
    font=dict(family="DM Mono, monospace", color=INK, size=11),
    margin=dict(l=16, r=24, t=44, b=16),
)

def ax(**kw):
    return dict(gridcolor=BORDER, zerolinecolor=BORDER, color=SUB,
                tickfont=dict(color=SUB, family="DM Mono, monospace", size=10), **kw)

def ax_t(t, **kw):
    return dict(gridcolor=BORDER, zerolinecolor=BORDER, color=SUB,
                tickfont=dict(color=SUB, family="DM Mono, monospace", size=10),
                title=dict(text=t, font=dict(color=SUB, size=10, family="DM Mono, monospace")), **kw)

def ply(fig, **kw):
    fig.update_layout(**BASE_LAYOUT, **kw)
    return fig

# ── HTML helpers ──────────────────────────────────────────────────
def kpi(label, value, color=INK, accent=None, bg=None):
    b   = f"border-left:4px solid {accent};" if accent else ""
    bg_ = bg or CARD
    bc  = accent or BORDER
    return (f"<div style='background:{bg_};border:1px solid {bc};"
            f"border-radius:10px;padding:16px 18px;{b}'>"
            f"<p style='margin:0 0 8px;font-family:DM Mono,monospace;font-size:0.54rem;"
            f"letter-spacing:0.2em;text-transform:uppercase;color:{DIM};font-weight:600'>{label}</p>"
            f"<p style='margin:0;font-family:Syne,sans-serif;font-size:2rem;"
            f"font-weight:700;color:{color};line-height:1;letter-spacing:-0.02em'>{value}</p></div>")

def mini(label, value, color=INK):
    return (f"<div style='background:{CARD};border:1px solid {BORDER};"
            f"border-radius:8px;padding:16px 18px'>"
            f"<p style='margin:0 0 8px;font-family:DM Mono,monospace;font-size:0.54rem;"
            f"letter-spacing:0.2em;text-transform:uppercase;color:{DIM};font-weight:600'>{label}</p>"
            f"<p style='margin:0;font-family:Syne,sans-serif;font-size:1.8rem;"
            f"font-weight:700;color:{color};line-height:1'>{value}</p></div>")

def section(text):
    st.markdown(
        f"<p style='font-family:DM Mono,monospace;font-size:0.6rem;letter-spacing:0.2em;"
        f"text-transform:uppercase;color:{SUB};border-bottom:1px solid {BORDER};"
        f"padding-bottom:10px;margin-bottom:18px;font-weight:700'>{text}</p>",
        unsafe_allow_html=True,
    )

def banner(msg, color, bg):
    st.markdown(
        f"<div style='background:{bg};border:1px solid {color};border-radius:8px;"
        f"padding:14px 20px;margin-bottom:22px'>"
        f"<p style='margin:0;font-family:DM Sans,sans-serif;font-size:0.85rem;"
        f"color:{color};font-weight:600'>{msg}</p></div>",
        unsafe_allow_html=True,
    )

def ai_box(text):
    clean = str(text).replace("**","").replace("##","").replace("```","")
    st.markdown(
        f"<div style='background:rgba(107,79,187,0.06);border:1px solid {PURPLE};"
        f"border-radius:10px;padding:16px 20px;margin-top:12px'>"
        f"<p style='margin:0 0 6px;font-family:DM Mono,monospace;font-size:0.56rem;"
        f"letter-spacing:0.18em;text-transform:uppercase;color:{PURPLE};font-weight:700'>"
        f"Gemini 2.0 Flash</p>"
        f"<p style='margin:0;font-family:DM Sans,sans-serif;font-size:0.88rem;"
        f"color:{INK};line-height:1.7;white-space:pre-wrap'>{clean}</p></div>",
        unsafe_allow_html=True,
    )

def transcript_box(text, lang=""):
    ls = f" · {lang}" if lang else ""
    st.markdown(
        f"<div style='background:rgba(0,122,110,0.06);border:1px solid {TEAL};"
        f"border-radius:10px;padding:16px 20px;margin:12px 0'>"
        f"<p style='margin:0 0 6px;font-family:DM Mono,monospace;font-size:0.56rem;"
        f"letter-spacing:0.18em;text-transform:uppercase;color:{TEAL};font-weight:700'>Whisper{ls}</p>"
        f"<p style='margin:0;font-family:DM Sans,sans-serif;font-size:0.92rem;"
        f"color:{INK};line-height:1.75'>{text}</p></div>",
        unsafe_allow_html=True,
    )

def log_row_box(entry):
    rows = "".join([
        f"<tr><td style='padding:7px 14px;font-family:DM Mono,monospace;font-size:0.62rem;"
        f"letter-spacing:0.1em;text-transform:uppercase;color:{DIM};white-space:nowrap;"
        f"border-bottom:1px solid {BORDER};background:{SURFACE}'>{k}</td>"
        f"<td style='padding:7px 14px;font-family:DM Sans,sans-serif;font-size:0.85rem;"
        f"color:{INK};border-bottom:1px solid {BORDER}'>{v}</td></tr>"
        for k, v in entry.items()
    ])
    st.markdown(
        f"<div style='background:{CARD};border:1px solid {BORDER};border-radius:10px;"
        f"overflow:hidden;margin-top:10px'>"
        f"<table style='width:100%;border-collapse:collapse'>{rows}</table></div>",
        unsafe_allow_html=True,
    )

def hint(text):
    st.markdown(
        f"<p style='font-family:DM Mono,monospace;font-size:0.65rem;"
        f"color:{DIM};margin:0 0 10px;line-height:1.7'>{text}</p>",
        unsafe_allow_html=True,
    )

def field_label(text):
    st.markdown(
        f"<p style='font-family:DM Mono,monospace;font-size:0.65rem;letter-spacing:0.12em;"
        f"text-transform:uppercase;color:{SUB};margin:0 0 4px;font-weight:700'>{text}</p>",
        unsafe_allow_html=True,
    )

# ── Audio helpers ─────────────────────────────────────────────────
def transcribe_audio(audio_bytes, suffix=".wav"):
    if not WHISPER_OK:
        return None, "unknown"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        m = load_whisper()
        r = m.transcribe(tmp_path)
        return r["text"].strip(), r.get("language", "unknown")
    except Exception as e:
        st.error(f"Whisper error: {e}")
        return None, "unknown"
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

def mic_button(key):
    """Render a mic button and return transcribed text or None. Never calls st.rerun()."""
    if not RECORDER_OK or not WHISPER_OK:
        return None
    ab = audio_recorder(
        text="", recording_color=RED, neutral_color=TEAL,
        icon_name="microphone", icon_size="1x",
        pause_threshold=3.0, sample_rate=16000, key=key,
    )
    if ab:
        with st.spinner("Transcribing…"):
            text, _ = transcribe_audio(ab)
        if text:
            st.caption(f"Heard: {text}")
            return text
    return None

def voice_select(key, options):
    """Mic button that matches spoken text to a list of options."""
    if not RECORDER_OK or not WHISPER_OK:
        return None
    ab = audio_recorder(
        text="", recording_color=RED, neutral_color=TEAL,
        icon_name="microphone", icon_size="1x",
        pause_threshold=3.0, sample_rate=16000, key=key,
    )
    if ab:
        with st.spinner("Matching…"):
            text, _ = transcribe_audio(ab)
        if text:
            st.caption(f"Heard: {text}")
            spoken = text.lower().replace("-","").replace("udi","").replace("m","").strip()
            for opt in options:
                opt_c = opt.lower().replace("-","").replace("udi","").replace("m","").strip()
                if spoken in opt_c or opt_c in spoken:
                    return opt
            if GEMINI_OK:
                pick = gemini(
                    f"User said: '{text}'. Pick the best match from: {options}. "
                    f"Return only the exact matching option or 'none'."
                ).strip().strip('"').strip("'")
                if pick in options:
                    return pick
    return None

# ── AI helpers ────────────────────────────────────────────────────
def gemini(prompt):
    if not GEMINI_OK:
        return "Gemini API key not configured."
    try:
        return _GEMINI.generate_content(prompt).text
    except Exception as e:
        return f"Gemini error: {e}"

def explain_machine(row, n_total, n_at_risk, avg_prob):
    vctx = ""
    if os.path.exists("data/voice_maintenance_log.csv"):
        try:
            vl = pd.read_csv("data/voice_maintenance_log.csv")
            ml = vl[vl["machine_id"].str.contains(str(int(row["udi"])), na=False)]
            if not ml.empty:
                vctx = "\n\nField reports on file:\n" + ml[["timestamp","urgency","transcript"]].to_string(index=False)
        except:
            pass
    return gemini(
        f"You are a senior predictive maintenance engineer. "
        f"Write a concise risk assessment in plain English. No markdown symbols.\n\n"
        f"Machine ID: M-{int(row['udi'])}\n"
        f"Failure probability: {row['prob_failure']:.1%}\n"
        f"Risk tier: {row['risk_tier']}\n"
        f"Actual failure recorded: {'YES' if row['machine_failure']==1 else 'No'}\n"
        f"Fleet average: {avg_prob:.1%}\n"
        f"{vctx}\n\n"
        f"Write exactly 3 paragraphs:\n"
        f"Risk Assessment: [why flagged]\n"
        f"Recommended Action: [what to do now]\n"
        f"Urgency: [how time-critical]\n"
        f"Max 120 words total."
    )

def ask_data(question, df, threshold):
    high = int((df["risk_tier"]=="High").sum())
    med  = int((df["risk_tier"]=="Medium").sum())
    low  = int((df["risk_tier"]=="Low").sum())
    top5 = df.nlargest(5,"prob_failure")[["udi","prob_failure","machine_failure"]].to_string(index=False)
    tp   = int(((df["prediction"]==1)&(df["machine_failure"]==1)).sum())
    fp   = int(((df["prediction"]==1)&(df["machine_failure"]==0)).sum())
    fn   = int(((df["prediction"]==0)&(df["machine_failure"]==1)).sum())
    tn   = int(((df["prediction"]==0)&(df["machine_failure"]==0)).sum())
    acc  = (df["prediction"].astype(int)==df["machine_failure"].astype(int)).mean()
    # Compute threshold counts so Gemini can answer any "above X%" question
    above_80  = int((df["prob_failure"] >= 0.80).sum())
    above_90  = int((df["prob_failure"] >= 0.90).sum())
    above_95  = int((df["prob_failure"] >= 0.95).sum())
    above_50  = int((df["prob_failure"] >= 0.50).sum())
    above_30  = int((df["prob_failure"] >= 0.30).sum())
    vctx = ""
    if os.path.exists("data/voice_maintenance_log.csv"):
        try:
            vl = pd.read_csv("data/voice_maintenance_log.csv").tail(5)
            if not vl.empty:
                vctx = "\n\nRecent field reports:\n" + vl[["timestamp","machine_id","urgency","transcript"]].to_string(index=False)
        except:
            pass
    return gemini(
        f"You are a data analyst for a predictive maintenance dashboard. "
        f"Answer concisely in plain English. No markdown symbols. "
        f"Use the exact numbers from the data below — do not say 'data not provided'.\n\n"
        f"Dataset summary:\n"
        f"- Total machines scored: {len(df)}\n"
        f"- Risk threshold in use: P >= {threshold:.2f}\n"
        f"- High risk (P >= 0.55): {high} machines\n"
        f"- Medium risk (0.30-0.55): {med} machines\n"
        f"- Low risk (P < 0.30): {low} machines\n"
        f"- Machines with P >= 0.80: {above_80}\n"
        f"- Machines with P >= 0.90: {above_90}\n"
        f"- Machines with P >= 0.95: {above_95}\n"
        f"- Machines with P >= 0.50: {above_50}\n"
        f"- Machines with P >= 0.30: {above_30}\n"
        f"- Confirmed failures in dataset: {int(df['machine_failure'].sum())}\n"
        f"- Average failure probability: {df['prob_failure'].mean():.1%}\n"
        f"- Max failure probability: {df['prob_failure'].max():.1%}\n"
        f"- Model accuracy: {acc:.1%} | TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}\n"
        f"- Precision: {tp/(tp+fp) if (tp+fp)>0 else 0:.1%} | Recall: {tp/(tp+fn) if (tp+fn)>0 else 0:.1%}\n"
        f"Top 5 machines by risk:\n{top5}{vctx}\n\n"
        f"Question: {question}"
    )

def shift_report(df, top_k, threshold):
    top = df.nlargest(top_k,"prob_failure")[["udi","prob_failure","risk_tier","machine_failure"]].to_string(index=False)
    tp  = int(((df["prediction"]==1)&(df["machine_failure"]==1)).sum())
    fn  = int(((df["prediction"]==0)&(df["machine_failure"]==1)).sum())
    vctx = ""
    if os.path.exists("data/voice_maintenance_log.csv"):
        try:
            vl = pd.read_csv("data/voice_maintenance_log.csv").tail(10)
            if not vl.empty:
                vctx = "\n\nField reports:\n" + vl[["timestamp","machine_id","urgency","transcript"]].to_string(index=False)
        except:
            pass
    return gemini(
        f"You are a maintenance shift supervisor writing a handover briefing. "
        f"Plain English only. No markdown symbols. Use plain section labels.\n\n"
        f"Results: {len(df)} machines | threshold P>={threshold:.2f} | "
        f"High:{int((df['risk_tier']=='High').sum())} | "
        f"Failures:{int(df['machine_failure'].sum())} | TP:{tp} Missed:{fn}\n"
        f"Top {top_k}: {top}{vctx}\n\n"
        f"Write these sections:\n"
        f"EXECUTIVE SUMMARY\n"
        f"PRIORITY DISPATCH LIST (top 3 with specific action)\n"
        f"WATCH LIST (medium risk)\n"
        f"MODEL CONFIDENCE NOTE\n"
        f"RECOMMENDED STAFFING\n"
        f"Max 300 words. Use machine IDs."
    )

def analyse_voice_note(transcript, machine_id, urgency):
    return gemini(
        f"You are a senior maintenance engineer reviewing a voice report. "
        f"Plain English only. No markdown symbols.\n\n"
        f"Machine: {machine_id} | Urgency: {urgency}\n"
        f"Technician said: \"{transcript}\"\n\n"
        f"Write 3 paragraphs:\n"
        f"Fault Assessment: [fault type]\n"
        f"Recommended Action: [specific steps]\n"
        f"Urgency Validation: [is declared urgency appropriate?]\n"
        f"Max 100 words."
    )

def parse_voice_to_fields(transcript):
    raw = gemini(
        f"Extract maintenance report fields from this voice note.\n"
        f"Return ONLY valid JSON with keys: fault_description, root_cause, "
        f"work_performed, parts_used, downtime_hours, recommendations, next_maintenance_date\n"
        f"Rules: next_maintenance_date=ISO YYYY-MM-DD or \"\"; "
        f"downtime_hours=number or 0; others=plain text or \"\"; no markdown.\n"
        f"Voice note: \"{transcript}\""
    )
    try:
        clean = raw.strip().replace("```json","").replace("```","").strip()
        return json.loads(clean)
    except:
        return {"fault_description":transcript,"root_cause":"","work_performed":"",
                "parts_used":"","downtime_hours":0,"recommendations":"","next_maintenance_date":""}

def polish_field(field_name, raw_text):
    return gemini(
        f"Rewrite this maintenance report field professionally. "
        f"Plain text only. No markdown. Max 3 sentences.\n"
        f"Field: {field_name}\nRaw: \"{raw_text}\""
    )

# ── PDF generator ─────────────────────────────────────────────────
def generate_pdf(fields, machine_data, shift_link=""):
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=20*mm, rightMargin=20*mm,
                             topMargin=20*mm, bottomMargin=20*mm)
    NAVY   = colors.HexColor("#1A3A5C")
    AMBER_ = colors.HexColor("#F5A623")
    LGRAY  = colors.HexColor("#F0F4F8")
    MGRAY  = colors.HexColor("#B8C4D0")
    DKTEXT = colors.HexColor("#1A1A2E")
    RED_   = colors.HexColor("#FF4B4B")
    GREEN_ = colors.HexColor("#00C875")
    styles = getSampleStyleSheet()
    def sty(n, base=None, **kw): return ParagraphStyle(n, parent=base or styles["Normal"], **kw)
    T = sty("T", fontSize=20, textColor=colors.white, fontName="Helvetica-Bold", spaceAfter=4)
    S = sty("S", fontSize=10, textColor=MGRAY, fontName="Helvetica", spaceAfter=2)
    H = sty("H", fontSize=9,  textColor=NAVY, fontName="Helvetica-Bold",
            spaceBefore=14, spaceAfter=4, textTransform="uppercase", letterSpacing=1.5)
    B = sty("B", fontSize=10, textColor=DKTEXT, fontName="Helvetica", leading=15, spaceAfter=4)
    SM= sty("SM",fontSize=8,  textColor=MGRAY, fontName="Helvetica", spaceAfter=2)
    rc = (RED_ if machine_data.get("risk_tier")=="High"
          else colors.HexColor("#F5A623") if machine_data.get("risk_tier")=="Medium"
          else GREEN_)
    story = []
    ht = Table([[Paragraph("MAINTENANCE REPORT",T),
                 Paragraph(f"<b>{fields.get('report_date','')}</b><br/>{fields.get('technician','Unknown')}",S)]],
               colWidths=[110*mm,60*mm])
    ht.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,-1),NAVY),("TEXTCOLOR",(0,0),(-1,-1),colors.white),
        ("ALIGN",(1,0),(1,0),"RIGHT"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),10),("BOTTOMPADDING",(0,0),(-1,-1),10),
        ("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),
    ]))
    story.append(ht); story.append(Spacer(1,6*mm))
    prob_str = f"{machine_data.get('prob_failure',0):.1%}" if machine_data.get("prob_failure") else "N/A"
    tier_str = machine_data.get("risk_tier","N/A")
    mt = Table([
        [Paragraph("MACHINE ID",SM), Paragraph("RISK TIER",SM),
         Paragraph("FAILURE PROB",SM), Paragraph("LOCATION",SM)],
        [Paragraph(f"<b>{fields.get('machine_id','')}</b>",B),
         Paragraph(f"<b>{tier_str}</b>", sty("RT", base=B, textColor=rc)),
         Paragraph(f"<b>{prob_str}</b>",B),
         Paragraph(fields.get("location","—"),B)],
    ], colWidths=[42*mm,35*mm,35*mm,58*mm])
    mt.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),LGRAY),("BACKGROUND",(0,1),(-1,1),colors.white),
        ("BOX",(0,0),(-1,-1),0.5,MGRAY),("INNERGRID",(0,0),(-1,-1),0.5,MGRAY),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
    ]))
    story.append(mt); story.append(Spacer(1,6*mm))
    def fb(label, content):
        if not content or not str(content).strip(): return
        story.append(Paragraph(label,H))
        story.append(HRFlowable(width="100%",thickness=0.5,color=MGRAY,spaceAfter=4))
        story.append(Paragraph(str(content),B)); story.append(Spacer(1,2*mm))
    fb("Fault Description",    fields.get("fault_description",""))
    fb("Root Cause Analysis",  fields.get("root_cause",""))
    fb("Work Performed",       fields.get("work_performed",""))
    fb("Parts Used",           fields.get("parts_used",""))
    if fields.get("downtime_hours"): fb("Downtime",f"{fields['downtime_hours']} hours")
    fb("Recommendations",      fields.get("recommendations",""))
    if fields.get("next_maintenance_date"): fb("Next Maintenance Date",fields["next_maintenance_date"])
    if shift_link:
        story.append(Spacer(1,4*mm))
        story.append(Paragraph("LINKED SHIFT REPORT",H))
        story.append(HRFlowable(width="100%",thickness=0.5,color=MGRAY,spaceAfter=4))
        story.append(Paragraph(shift_link[:300]+"…" if len(shift_link)>300 else shift_link,B))
    story.append(Spacer(1,8*mm))
    story.append(HRFlowable(width="100%",thickness=1,color=AMBER_,spaceAfter=4))
    story.append(Paragraph(
        f"Generated by PredMaint · AI4I 2020 · {fields.get('report_date','')} · "
        f"Technician: {fields.get('technician','—')}", SM))
    doc.build(story)
    return buf.getvalue()

# ── Data loading ──────────────────────────────────────────────────
LOCAL_CSV     = "data/predictions.csv"
VOICE_LOG_CSV = "data/voice_maintenance_log.csv"

def parse_df(source):
    df = pd.read_csv(source)
    if "prob_failure" not in df.columns and "probability" in df.columns:
        df["prob_failure"] = df["probability"].apply(
            lambda v: float(str(v).strip("[]").split()[-1])
        )
    return df

@st.cache_data(ttl=300, show_spinner="Loading predictions…")
def load_local():
    if os.path.exists(LOCAL_CSV):
        return parse_df(LOCAL_CSV)
    host  = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    if host and token:
        try:
            from databricks.connect import DatabricksSession
            spark = DatabricksSession.builder.getOrCreate()
            df = spark.table("ai4i2020_demo_db.ai4i_predictions").toPandas()
            df["prob_failure"] = df["probability"].apply(lambda v: float(list(v)[1]))
            df.drop(columns=["probability"], inplace=True)
            return df
        except Exception as e:
            st.sidebar.warning(f"Databricks: {e}")
    import numpy as np
    rng = np.random.default_rng(42); n = 500; prob = rng.beta(1.5,8,size=n)
    return pd.DataFrame({"udi":rng.integers(1000,9999,size=n),
                          "machine_failure":(prob>0.55).astype(int),
                          "prediction":(prob>0.45).astype(float),
                          "prob_failure":prob.round(4)})

def load_voice_log():
    if os.path.exists(VOICE_LOG_CSV):
        try: return pd.read_csv(VOICE_LOG_CSV)
        except: pass
    return pd.DataFrame(columns=["timestamp","machine_id","technician","urgency",
                                  "transcript","ai_response","detected_language",
                                  "source","audio_filename"])

def save_voice_log(entry):
    os.makedirs("data", exist_ok=True)
    pd.DataFrame([entry]).to_csv(
        VOICE_LOG_CSV, mode="a",
        header=not os.path.exists(VOICE_LOG_CSV), index=False,
    )

# ── SESSION STATE — initialise ALL widget-backed keys FIRST ───────
# This must happen before ANY widget is created to avoid the
# "cannot be modified after instantiation" error.
_field_defaults = {
    "r_fault": "", "r_cause": "", "r_work": "",
    "r_parts": "", "r_recs":  "",
}
for _k, _v in _field_defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Apply any pending mic updates BEFORE widgets render
for _field in ["r_fault","r_cause","r_recs"]:
    _pending = f"{_field}_pending"
    if st.session_state.get(_pending):
        st.session_state[_field] = st.session_state.pop(_pending)

# ── Sidebar ───────────────────────────────────────────────────────
def sb(html): st.sidebar.markdown(html, unsafe_allow_html=True)
def sb_hr():  sb(f"<hr style='border:none;border-top:1px solid {BORDER};margin:14px 0'>")

sb(f"<h1 style='font-family:Syne,sans-serif;font-size:1.25rem;font-weight:800;"
   f"color:{AMBER};margin:0 0 2px'>PredMaint</h1>"
   f"<p style='font-family:DM Mono,monospace;font-size:0.7rem;color:{DIM};margin:0'>"
   f"AI4I 2020 · Gemini · Whisper</p>")
sb_hr()
risk_threshold = st.sidebar.slider("Risk threshold (P ≥)", 0.10, 0.90, 0.45, 0.05, format="%.2f")
sb_hr()
sb(f"<p style='font-family:DM Mono,monospace;font-size:0.58rem;letter-spacing:0.2em;"
   f"text-transform:uppercase;color:{DIM};margin:0 0 8px;font-weight:600'>Upload Predictions</p>")
uploaded_file = st.sidebar.file_uploader(
    "predictions.csv", type=["csv"], label_visibility="collapsed",
)
sb_hr()
if st.sidebar.button("Refresh data"):
    st.cache_data.clear(); st.rerun()
sb_hr()
ai_dot  = f"<span style='color:{GREEN}'>connected</span>" if GEMINI_OK else f"<span style='color:{RED}'>not configured</span>"
src_lbl = "uploaded file" if uploaded_file else LOCAL_CSV
sb(f"<p style='font-family:DM Mono,monospace;font-size:0.68rem;color:{DIM};margin:0 0 3px'>"
   f"AI · {ai_dot}</p>"
   f"<p style='font-family:DM Mono,monospace;font-size:0.68rem;color:{DIM};margin:0'>"
   f"Data · <span style='color:{MED}'>{src_lbl}</span></p>")

# ── Load data ─────────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        df = parse_df(uploaded_file)
        st.sidebar.markdown(
            f"<p style='font-family:DM Mono,monospace;font-size:0.68rem;"
            f"color:{GREEN};margin:6px 0 0'>{len(df):,} rows loaded</p>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.sidebar.error(f"Upload error: {e}"); df = load_local()
else:
    df = load_local()

# ── Derive columns ────────────────────────────────────────────────
df["at_risk"]   = df["prob_failure"] >= risk_threshold
df["correct"]   = df["prediction"].astype(int) == df["machine_failure"].astype(int)
df["risk_tier"] = pd.cut(df["prob_failure"], bins=[0,0.30,0.55,1.01],
                          labels=["Low","Medium","High"], right=False)
n_total   = len(df); n_at_risk = int(df["at_risk"].sum()); accuracy = df["correct"].mean()
tp = int(((df["prediction"]==1)&(df["machine_failure"]==1)).sum())
fp = int(((df["prediction"]==1)&(df["machine_failure"]==0)).sum())
fn = int(((df["prediction"]==0)&(df["machine_failure"]==1)).sum())
tn = int(((df["prediction"]==0)&(df["machine_failure"]==0)).sum())
precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
recall    = tp/(tp+fn) if (tp+fn)>0 else 0.0
f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

# Machine options used across tabs
machine_opts = [f"M-{int(u)}" for u in df.sort_values("prob_failure",ascending=False).head(20)["udi"].tolist()]

# ── Header ────────────────────────────────────────────────────────
st.markdown(
    f"<div style='padding:24px 0 18px;border-bottom:1px solid {BORDER};margin-bottom:24px'>"
    f"<h1 style='font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;"
    f"color:{INK};margin:0;line-height:1.1;letter-spacing:-0.02em'>"
    f"Predictive Maintenance <span style='color:{AMBER}'>Dashboard</span></h1></div>",
    unsafe_allow_html=True,
)

# ── KPIs ──────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
risk_c = RED if n_at_risk>20 else AMBER if n_at_risk>5 else GREEN
acc_c  = GREEN if accuracy>=0.90 else AMBER if accuracy>=0.75 else RED
rec_c  = GREEN if recall>=0.80  else AMBER if recall>=0.60  else RED
fn_c   = GREEN if fn==0 else RED
k1.markdown(kpi("Machines Scored",  f"{n_total:,}",   INK,    BLUE), unsafe_allow_html=True)
k2.markdown(kpi("Flagged for Alert",f"{n_at_risk:,}", risk_c, risk_c, "#FFF3E0" if n_at_risk>20 else CARD), unsafe_allow_html=True)
k3.markdown(kpi("Missed Failures",  f"{fn:,}",        fn_c,   fn_c,  "#FDECEA" if fn>0 else CARD), unsafe_allow_html=True)
k4.markdown(kpi("Model Recall",     f"{recall:.1%}",  rec_c,  rec_c), unsafe_allow_html=True)
k5.markdown(kpi("Accuracy",         f"{accuracy:.1%}",acc_c,  acc_c), unsafe_allow_html=True)
st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "Alerts","Analysis","Ask AI","Voice Logger","Maintenance Report",
])

# ══════════════════════════════════════════════
# TAB 1 — Alerts
# ══════════════════════════════════════════════
with tab1:
    high_n = int((df["risk_tier"]=="High").sum())
    if high_n>0:
        banner(f"⚠  {high_n} machine{'s' if high_n!=1 else ''} in HIGH risk — immediate maintenance required.", RED, "#FDECEA")
    else:
        banner("No machines in HIGH risk tier at the current threshold.", GREEN, "#E8F5EE")

    # ── Controls row ─────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([1, 4, 1])
    with ctrl1:
        top_k = st.selectbox("Show top", options=[5,10,15,20], index=1, key="top_k_sel")
    with ctrl3:
        gen_report = st.button("Generate Shift Report", key="btn_report")

    top_df = df.sort_values("prob_failure",ascending=False).head(top_k).reset_index(drop=True)
    top_df.index += 1
    top_machine_opts = [f"M-{int(u)}" for u in top_df["udi"].tolist()]

    # ── Main row: compact table LEFT, chart RIGHT ─────────────────
    lc, rc = st.columns([1.05, 1], gap="large")

    with lc:
        section(f"Top {top_k} machines · ranked by failure probability")
        rows_html = ""
        for rank, row in top_df.iterrows():
            tier   = str(row["risk_tier"]); prob = row["prob_failure"]
            actual = "FAILED" if row["machine_failure"]==1 else "—"
            ac     = RED if row["machine_failure"]==1 else DIM
            bc     = RED if tier=="High" else AMBER if tier=="Medium" else GREEN
            bw     = int(prob*100)
            rows_html += (
                f"<tr style='border-bottom:1px solid {BORDER}'>"
                f"<td style='padding:7px 12px;font-family:DM Mono,monospace;font-size:0.68rem;color:{DIM}'>{rank:02d}</td>"
                f"<td style='padding:7px 12px;font-family:Syne,sans-serif;font-size:0.9rem;color:{INK};font-weight:700'>M-{int(row['udi'])}</td>"
                f"<td style='padding:7px 12px'>"
                f"<span style='color:{bc};font-family:DM Mono,monospace;font-weight:700;font-size:0.76rem'>{tier}</span>"
                f"<span style='font-family:DM Mono,monospace;font-size:0.7rem;color:{MED};margin-left:6px'>{prob:.1%}</span>"
                f"<div style='margin-top:3px;width:80px;height:2px;background:{BORDER};border-radius:2px'>"
                f"<div style='width:{bw}%;height:2px;background:{bc};border-radius:2px'></div></div></td>"
                f"<td style='padding:7px 12px;font-family:DM Mono,monospace;font-size:0.72rem;color:{ac};font-weight:600'>{actual}</td>"
                f"</tr>"
            )
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse;background:{CARD};"
            f"border:1px solid {BORDER};border-radius:10px;overflow:hidden'>"
            f"<thead><tr style='background:{SURFACE};border-bottom:2px solid {BORDER2}'>"
            f"<th style='padding:7px 12px;text-align:left;font-family:DM Mono,monospace;font-size:0.5rem;letter-spacing:0.2em;color:{SUB};font-weight:700'>#</th>"
            f"<th style='padding:7px 12px;text-align:left;font-family:DM Mono,monospace;font-size:0.5rem;letter-spacing:0.2em;color:{SUB};font-weight:700'>MACHINE ID</th>"
            f"<th style='padding:7px 12px;text-align:left;font-family:DM Mono,monospace;font-size:0.5rem;letter-spacing:0.2em;color:{SUB};font-weight:700'>RISK</th>"
            f"<th style='padding:7px 12px;text-align:left;font-family:DM Mono,monospace;font-size:0.5rem;letter-spacing:0.2em;color:{SUB};font-weight:700'>ACTUAL</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )

        # Machine explainer — compact
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        ex1, ex2, ex3 = st.columns([4, 1, 1])
        with ex1:
            sel_mid = st.selectbox("Machine", top_machine_opts,
                                   key="explainer_uid", label_visibility="collapsed")
        with ex2:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            explain_clicked = st.button("Explain", key="btn_explain")
        with ex3:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            v_sel = voice_select("mic_explainer_sel", top_machine_opts)
            if v_sel:
                st.session_state["explainer_uid"] = top_machine_opts.index(v_sel)
                st.rerun()

        if explain_clicked:
            uid_num = int(sel_mid.replace("M-",""))
            sel_row = top_df[top_df["udi"]==uid_num].iloc[0].to_dict()
            with st.spinner("Analysing…"):
                ai_box(explain_machine(sel_row, n_total, n_at_risk, df["prob_failure"].mean()))

    with rc:
        cdf   = top_df.head(min(top_k, len(top_df)))
        probs = cdf["prob_failure"].tolist()
        def prob_color(p):
            return RED if p>=0.75 else AMBER if p>=0.45 else GREEN
        fig = go.Figure(go.Bar(
            x=probs,
            y=[f"M-{int(u)}" for u in cdf["udi"]],
            orientation="h",
            marker=dict(color=[prob_color(p) for p in probs],
                        opacity=[0.4+0.6*p for p in probs],
                        line=dict(width=0)),
            text=[f"{p:.1%}" for p in probs],
            textposition="outside",
            textfont=dict(family="DM Mono, monospace", size=10, color=INK),
        ))
        fig.add_vline(x=risk_threshold, line_dash="dot", line_color=AMBER, line_width=1.5,
                      annotation_text=f"P≥{risk_threshold:.2f}",
                      annotation_font=dict(size=9, color=AMBER, family="DM Mono, monospace"))
        ply(fig,
            title=dict(text="Failure probability by machine ID",
                       font=dict(size=11, color=SUB, family="DM Mono, monospace")),
            xaxis=dict(tickformat=".0%", range=[0,1.2], **ax()),
            yaxis=dict(autorange="reversed", type="category",
                       tickfont=dict(size=11, color=INK, family="DM Mono, monospace"),
                       gridcolor=BORDER, zerolinecolor=BORDER, color=MED),
            height=max(180, len(cdf)*34),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Shift report — full width below ──────────────────────────
    if gen_report:
        with st.spinner("Gemini writing shift report…"):
            report = shift_report(df, int(top_k), risk_threshold)
        st.session_state["last_shift_report"] = report
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        section("Shift report")
        ai_box(report)
        st.download_button("Download report", data=report.encode(),
                           file_name="shift_report.txt", mime="text/plain")
    elif st.session_state.get("last_shift_report"):
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        section("Shift report")
        ai_box(st.session_state["last_shift_report"])
        st.download_button("Download report",
                           data=st.session_state["last_shift_report"].encode(),
                           file_name="shift_report.txt", mime="text/plain")

# ══════════════════════════════════════════════
# TAB 2 — Analysis
# ══════════════════════════════════════════════
with tab2:
    c1,c2 = st.columns(2, gap="large")
    with c1:
        section("Failure probability distribution")
        fh = go.Figure(go.Histogram(x=df["prob_failure"], nbinsx=40,
                                    marker=dict(color=df["prob_failure"],
                                                colorscale=[[0,GREEN],[0.45,AMBER],[0.8,RED]],
                                                line=dict(width=0))))
        fh.add_vline(x=risk_threshold, line_dash="dot", line_color=AMBER, line_width=1.5,
                     annotation_text=f"threshold {risk_threshold:.2f}",
                     annotation_font=dict(size=9, color=AMBER, family="DM Mono, monospace"))
        ply(fh, title=dict(text="P(failure) distribution",
                           font=dict(size=12,color=INK,family="DM Mono, monospace")),
            xaxis=ax_t("P(failure)",tickformat=".0%"), yaxis=ax_t("Machines"), height=240)
        st.plotly_chart(fh, use_container_width=True)

    with c2:
        section("Risk tier breakdown")
        tc = df["risk_tier"].value_counts().reindex(["Low","Medium","High"]).fillna(0)
        fd = go.Figure(go.Pie(
            labels=tc.index.tolist(),
            values=tc.values.tolist(),
            hole=0.65,
            marker=dict(colors=[GREEN, AMBER, RED], line=dict(width=0)),
            textinfo="percent",
            textposition="outside",
            textfont=dict(family="DM Mono, monospace", size=11, color=INK),
            pull=[0, 0, 0.04],
            showlegend=True,
        ))
        ply(fd,
            title=dict(text="Risk tier split",
                       font=dict(size=12, color=INK, family="DM Mono, monospace")),
            legend=dict(
                orientation="v", x=0.85, y=0.5,
                font=dict(family="DM Mono, monospace", size=11, color=INK),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=240, showlegend=True,
        )
        st.plotly_chart(fd, use_container_width=True)
        # Summary counts below chart so small slices are always visible
        low_n  = int(tc.get("Low",  0))
        med_n  = int(tc.get("Medium",0))
        high_n_= int(tc.get("High", 0))
        st.markdown(
            f"<div style='display:flex;gap:16px;margin-top:4px'>"
            f"<span style='font-family:DM Mono,monospace;font-size:0.72rem;"
            f"color:{GREEN}'>Low {low_n:,}</span>"
            f"<span style='font-family:DM Mono,monospace;font-size:0.72rem;"
            f"color:{AMBER}'>Medium {med_n:,}</span>"
            f"<span style='font-family:DM Mono,monospace;font-size:0.72rem;"
            f"color:{RED}'>High {high_n_:,}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    cm1,cm2 = st.columns(2, gap="large")
    with cm1:
        section("Confusion matrix")
        fcm = go.Figure(go.Heatmap(
            z=[[tn,fp],[fn,tp]],
            x=["Pred: No Failure","Pred: Failure"],
            y=["Actual: No Failure","Actual: Failure"],
            colorscale=[[0,WHITE],[0.4,"#C8ECD8"],[1,GREEN]], showscale=False,
            text=[[f"{tn}",f"{fp}"],[f"{fn}",f"{tp}"]], texttemplate="%{text}",
            textfont=dict(family="DM Mono, monospace",size=24,color=INK)))
        ply(fcm, xaxis=dict(side="bottom",**ax()), yaxis=ax(), height=240)
        st.plotly_chart(fcm, use_container_width=True)

    with cm2:
        section("Classification metrics")
        fm = go.Figure()
        for name,val in [("Accuracy",accuracy),("Precision",precision),("Recall",recall),("F1",f1)]:
            c = GREEN if val>=0.80 else AMBER if val>=0.60 else RED
            fm.add_trace(go.Bar(x=[val],y=[name],orientation="h",
                                marker=dict(color=c,line=dict(width=0),opacity=0.88),
                                text=[f"{val:.3f}"],textposition="outside",
                                textfont=dict(family="DM Mono, monospace",size=12,color=INK),
                                showlegend=False))
        ply(fm, xaxis=dict(range=[0,1.2],tickformat=".0%",**ax()),
            yaxis=ax(), height=240, barmode="overlay")
        st.plotly_chart(fm, use_container_width=True)

    section("Prediction breakdown")
    b1,b2,b3,b4 = st.columns(4)
    b1.markdown(mini("True Positives",  f"{tp:,}", GREEN), unsafe_allow_html=True)
    b2.markdown(mini("True Negatives",  f"{tn:,}", GREEN), unsafe_allow_html=True)
    b3.markdown(mini("False Positives", f"{fp:,}", AMBER), unsafe_allow_html=True)
    b4.markdown(mini("False Negatives", f"{fn:,}", RED),   unsafe_allow_html=True)
    if fn>0:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        banner(f"⚠  {fn} missed failure(s). Consider lowering the risk threshold.", RED, "#FDECEA")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    section("Actual failure rate per tier")
    ts = (df.groupby("risk_tier",observed=True)
            .agg(count=("machine_failure","count"),failures=("machine_failure","sum"))
            .assign(rate=lambda x: x["failures"]/x["count"]).reset_index())
    cmap = {"Low":GREEN,"Medium":AMBER,"High":RED}
    ft = go.Figure()
    for _,r in ts.iterrows():
        ft.add_trace(go.Bar(x=[str(r["risk_tier"])],y=[r["rate"]],
                            marker=dict(color=cmap.get(str(r["risk_tier"]),DIM),line=dict(width=0),opacity=0.88),
                            text=[f"{r['rate']:.1%}"],textposition="outside",
                            textfont=dict(family="DM Mono, monospace",size=12,color=INK),
                            showlegend=False))
    ply(ft, title=dict(text="Failure rate per risk tier",
                       font=dict(size=12,color=INK,family="DM Mono, monospace")),
        xaxis=ax(), yaxis=ax_t("Failure rate",tickformat=".0%"), height=260)
    st.plotly_chart(ft, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — Ask AI
# ══════════════════════════════════════════════
with tab3:
    section("Ask the data — Gemini 2.0 Flash")

    example_questions = [
        "Choose a suggested question or write your own below…",
        "How many machines are in the high risk tier?",
        "What percentage of machines have above 80% failure probability?",
        "How reliable is the model — precision vs recall trade-off?",
        "Are there more false negatives or false positives?",
        "If I only have 2 technicians this shift, which 3 machines should I prioritize and why?",
        "What would happen if we used time-based replacement instead of this predictive model?",
    ]

    ai_q1, ai_q2 = st.columns([3, 1], gap="medium")
    with ai_q1:
        selected_q = st.selectbox(
            "Suggested questions",
            options=example_questions,
            key="suggested_q",
            label_visibility="collapsed",
        )
    with ai_q2:
        use_suggestion = st.button(
            "Ask →", key="btn_suggestion",
            disabled=(selected_q == example_questions[0]),
        )

    st.markdown(
        f"<p style='font-family:DM Mono,monospace;font-size:0.62rem;"
        f"color:{DIM};margin:8px 0;letter-spacing:0.1em'>— OR WRITE YOUR OWN —</p>",
        unsafe_allow_html=True,
    )

    cq1, cq2 = st.columns([5, 1], gap="small")
    with cq1:
        custom_q = st.text_input(
            "Your question",
            placeholder="e.g. Which failure mode is most common in high risk machines?",
            key="custom_q",
            label_visibility="collapsed",
        )
    with cq2:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        ask_custom = st.button("Ask →", key="btn_ask_custom")

    # Determine what to ask
    final_q = None
    if use_suggestion and selected_q != example_questions[0]:
        final_q = selected_q
    elif ask_custom and custom_q.strip():
        final_q = custom_q.strip()
    elif ask_custom:
        st.warning("Please type a question first.")

    if final_q:
        with st.spinner("Gemini is thinking…"):
            result = ask_data(final_q, df, risk_threshold)
        ai_box(f"{final_q}\n\n{result}")

    if not GEMINI_OK:
        banner("GEMINI_API_KEY not set. Add it to Streamlit Cloud Settings → Secrets.", AMBER, "#FFF8E7")

# ══════════════════════════════════════════════
# TAB 4 — Voice Logger
# ══════════════════════════════════════════════
with tab4:
    miss = [p for p,ok in [("openai-whisper",WHISPER_OK),("audio-recorder-streamlit",RECORDER_OK)] if not ok]
    if miss:
        banner(f"Missing: {', '.join(miss)} — pip install {' '.join(miss)} then restart.", AMBER, "#FFF8E7")
    else:
        vl_col,vr_col = st.columns([1.1,1], gap="large")
        with vl_col:
            section("Voice fault logger")
            st.markdown(f"<p style='font-family:DM Sans,sans-serif;font-size:0.88rem;color:{MED};"
                        f"margin:0 0 20px;line-height:1.7'>Record a voice fault note. "
                        f"Whisper transcribes locally. Gemini classifies and recommends.</p>",
                        unsafe_allow_html=True)

            mopts_vl = ["Manual entry"] + machine_opts
            mid1,mid2 = st.columns([6,1])
            with mid1:
                sm_vl = st.selectbox("Machine ID", mopts_vl, key="v_machine")
            with mid2:
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                v_mid = voice_select("mic_vlog_mid", machine_opts)
                if v_mid and v_mid in mopts_vl:
                    st.session_state["v_machine"] = mopts_vl.index(v_mid)
                    st.rerun()

            vlog_machine_id = (st.text_input("Enter machine ID", placeholder="AGP-LINE-1",
                                              key="v_mid_manual", label_visibility="collapsed")
                               if sm_vl=="Manual entry" else sm_vl)
            vlog_technician = st.text_input("Technician name", placeholder="Your name", key="v_tech")

            urgency_opts = ["High — stop immediately","Medium — fix within shift",
                            "Low — schedule next maintenance","Unknown"]
            urg1,urg2 = st.columns([6,1])
            with urg1:
                vlog_urgency = st.selectbox("Declared urgency", urgency_opts, key="v_urgency")
            with urg2:
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                v_urg = voice_select("mic_vlog_urg", urgency_opts)
                if v_urg and v_urg in urgency_opts:
                    st.session_state["v_urgency"] = urgency_opts.index(v_urg)
                    st.rerun()

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            section("Record voice note")
            hint("Click mic to start. Click again or wait 3 s silence to stop.")
            ab = audio_recorder(text="", recording_color=RED, neutral_color=TEAL,
                                icon_name="microphone", icon_size="2x",
                                pause_threshold=3.0, sample_rate=16000, key="v_recorder")
            if ab: st.audio(ab, format="audio/wav"); hint("Captured. Click Transcribe below.")
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            section("Or upload audio")
            af = st.file_uploader("Upload audio", type=["mp3","wav","m4a","ogg","flac"],
                                   key="v_upload", label_visibility="collapsed")
            if af: st.audio(af, format="audio/wav")

            if ab:   adata=ab;       aname="mic_recording.wav"; asuffix=".wav";  asrc="microphone"
            elif af: adata=af.read();aname=af.name;              asuffix="."+af.name.split(".")[-1]; asrc="file upload"
            else:    adata=None;     aname=None;                 asuffix=".wav"; asrc=None

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            if adata is None: hint("Record or upload audio to activate the button.")
            if st.button("Transcribe & Analyse", key="btn_transcribe", disabled=(adata is None)):
                if not vlog_machine_id or not vlog_machine_id.strip():
                    banner("Please enter a machine ID.", AMBER, "#FFF8E7")
                else:
                    with st.spinner("Transcribing with Whisper…"):
                        transcript,lang = transcribe_audio(adata, asuffix)
                    if transcript:
                        transcript_box(transcript, lang)
                        with st.spinner("Gemini analysing…"):
                            air = analyse_voice_note(transcript, vlog_machine_id, vlog_urgency)
                        ai_box(air)
                        entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                 "machine_id": vlog_machine_id.strip(),
                                 "technician": vlog_technician.strip() or "Unknown",
                                 "urgency": vlog_urgency, "transcript": transcript,
                                 "ai_response": air, "detected_language": lang,
                                 "source": asrc, "audio_filename": aname}
                        save_voice_log(entry)
                        section("Saved")
                        log_row_box({"Timestamp":entry["timestamp"],"Machine":entry["machine_id"],
                                     "Technician":entry["technician"],"Urgency":entry["urgency"],"Source":asrc})
                        banner(f"Saved to {VOICE_LOG_CSV}", GREEN, "#E8F5EE")

        with vr_col:
            section("Voice log history")
            vlog = load_voice_log()
            if vlog.empty:
                hint("No voice logs yet. Record a fault note on the left.")
            else:
                mc1,mc2,mc3 = st.columns(3)
                mc1.markdown(mini("Total",       f"{len(vlog):,}",              TEAL),   unsafe_allow_html=True)
                mc2.markdown(mini("Machines",    f"{vlog['machine_id'].nunique():,}", BLUE),   unsafe_allow_html=True)
                mc3.markdown(mini("Technicians", f"{vlog['technician'].nunique():,}",PURPLE), unsafe_allow_html=True)
                st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
                dcols = [c for c in ["timestamp","machine_id","technician","urgency","transcript"] if c in vlog.columns]
                st.dataframe(vlog[dcols].sort_values("timestamp",ascending=False).head(20).reset_index(drop=True),
                             use_container_width=True, hide_index=True)
                st.download_button("Download voice log (CSV)", data=vlog.to_csv(index=False).encode(),
                                   file_name="voice_maintenance_log.csv", mime="text/csv", key="dl_vlog")
                if "urgency" in vlog.columns and len(vlog)>1:
                    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
                    section("Urgency breakdown")
                    urg = vlog["urgency"].value_counts()
                    uc = [RED if "High" in str(l) else AMBER if "Medium" in str(l) else GREEN if "Low" in str(l) else DIM for l in urg.index]
                    fu = go.Figure(go.Bar(x=urg.values.tolist(),y=urg.index.tolist(),orientation="h",
                                         marker=dict(color=uc,line=dict(width=0),opacity=0.88),
                                         text=[str(v) for v in urg.values],textposition="outside",
                                         textfont=dict(family="DM Mono, monospace",size=11,color=INK)))
                    ply(fu, xaxis=ax_t("Reports"), yaxis=ax(), height=200, showlegend=False)
                    st.plotly_chart(fu, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 5 — Maintenance Report
# ══════════════════════════════════════════════
with tab5:
    if not REPORTLAB_OK:
        banner("ReportLab not installed. Run: pip install reportlab then restart.", AMBER, "#FFF8E7")
    else:
        r_left,r_right = st.columns([1.2,1], gap="large")
        with r_left:
            section("Maintenance report generator")
            st.markdown(f"<p style='font-family:DM Sans,sans-serif;font-size:0.88rem;color:{MED};"
                        f"margin:0 0 20px;line-height:1.7'>Fill fields manually, use mic buttons, "
                        f"or use Speak Once to let AI fill the form.</p>", unsafe_allow_html=True)

            # ── Speak once ────────────────────────────────────────
            section("Option A — Speak once, AI fills the form")
            hint("Describe fault, root cause, and recommendations in one voice note.")
            if RECORDER_OK and WHISPER_OK:
                so_bytes = audio_recorder(text="", recording_color=RED, neutral_color=TEAL,
                                          icon_name="microphone", icon_size="2x",
                                          pause_threshold=4.0, sample_rate=16000, key="r_speak_once")
                if so_bytes:
                    st.audio(so_bytes, format="audio/wav")
                    if st.button("Auto-fill form from voice", key="btn_autofill"):
                        with st.spinner("Transcribing…"):
                            so_text,so_lang = transcribe_audio(so_bytes)
                        if so_text:
                            transcript_box(so_text, so_lang)
                            with st.spinner("Gemini extracting fields…"):
                                extracted = parse_voice_to_fields(so_text)
                            # Write to pending keys — applied at top of next run
                            for sk,ek in [("r_fault","fault_description"),("r_cause","root_cause"),
                                          ("r_work","work_performed"),("r_parts","parts_used"),
                                          ("r_recs","recommendations")]:
                                st.session_state[f"{sk}_pending"] = extracted.get(ek,"")
                            nd = extracted.get("next_maintenance_date","")
                            if nd:
                                try: st.session_state["r_next_date"] = date.fromisoformat(nd)
                                except: pass
                            banner("Form filled from voice. Review and edit below.", TEAL, "#E8F5EE")
                            st.rerun()
            else:
                hint("Install openai-whisper and audio-recorder-streamlit to enable voice input.")

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # ── Manual form ───────────────────────────────────────
            section("Option B — Fill fields manually or speak field by field")
            mopts_r = ["Manual entry"] + machine_opts
            rm1,rm2 = st.columns([6,1])
            with rm1:
                r_machine_sel = st.selectbox("Machine ID", mopts_r, key="r_machine_sel")
            with rm2:
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                v_rmid = voice_select("mic_report_mid", machine_opts)
                if v_rmid and v_rmid in mopts_r:
                    st.session_state["r_machine_sel"] = mopts_r.index(v_rmid)
                    st.rerun()

            r_machine_id = (st.text_input("Enter machine ID", placeholder="AGP-LINE-3",
                                           key="r_machine_manual", label_visibility="collapsed")
                            if r_machine_sel=="Manual entry" else r_machine_sel)
            r_machine_data = {}
            if r_machine_id and r_machine_id.startswith("M-"):
                try:
                    uid = int(r_machine_id.replace("M-",""))
                    row_ = df[df["udi"]==uid]
                    if not row_.empty:
                        r_machine_data = {"risk_tier":str(row_.iloc[0]["risk_tier"]),
                                          "prob_failure":float(row_.iloc[0]["prob_failure"])}
                        tc_ = RED if r_machine_data["risk_tier"]=="High" else AMBER if r_machine_data["risk_tier"]=="Medium" else GREEN
                        st.markdown(f"<p style='font-family:DM Mono,monospace;font-size:0.72rem;"
                                    f"color:{tc_};margin:4px 0 12px'>Risk tier: {r_machine_data['risk_tier']} "
                                    f"· Failure probability: {r_machine_data['prob_failure']:.1%}</p>",
                                    unsafe_allow_html=True)
                except: pass

            r_location    = st.text_input("Location / line", placeholder="e.g. Production line 3, Gent", key="r_loc")
            r_technician  = st.text_input("Technician name", placeholder="Your name", key="r_tech_name")
            r_report_date = st.date_input("Report date", value=date.today(), key="r_date")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # ── Text fields with inline mic ───────────────────────
            # The pending pattern: mic writes to X_pending, which is consumed
            # at the top of the NEXT run before widgets render.
            field_label("Fault description")
            f1c,f2c = st.columns([8,1])
            with f1c:
                r_fault = st.text_area("Fault description", height=90, key="r_fault",
                                        label_visibility="collapsed",
                                        placeholder="Describe the fault observed…")
            with f2c:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                mf = mic_button("mic_r_fault")
                if mf:
                    st.session_state["r_fault_pending"] = mf
                    st.rerun()

            field_label("Root cause")
            rc1,rc2 = st.columns([8,1])
            with rc1:
                r_cause = st.text_area("Root cause", height=90, key="r_cause",
                                        label_visibility="collapsed",
                                        placeholder="What caused the fault?…")
            with rc2:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                mc = mic_button("mic_r_cause")
                if mc:
                    st.session_state["r_cause_pending"] = mc
                    st.rerun()

            field_label("Recommendations")
            rr1,rr2 = st.columns([8,1])
            with rr1:
                r_recs = st.text_area("Recommendations", height=90, key="r_recs",
                                       label_visibility="collapsed",
                                       placeholder="Preventive actions, follow-up steps…")
            with rr2:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                mr = mic_button("mic_r_recs")
                if mr:
                    st.session_state["r_recs_pending"] = mr
                    st.rerun()

            r_next_date   = st.date_input("Next maintenance date",
                                           value=st.session_state.get("r_next_date", None),
                                           key="r_next_date_input")
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            polish_toggle = st.checkbox("Polish report fields with Gemini before generating PDF",
                                         value=True, key="r_polish")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            if st.button("Generate PDF Report", key="btn_gen_pdf"):
                if not r_machine_id or not r_machine_id.strip():
                    banner("Please select or enter a machine ID.", AMBER, "#FFF8E7")
                elif not r_fault and not r_cause and not r_recs:
                    banner("Please fill in at least one field.", AMBER, "#FFF8E7")
                else:
                    fields_ = {
                        "machine_id":           r_machine_id.strip(),
                        "location":             r_location,
                        "technician":           r_technician or "Unknown",
                        "report_date":          str(r_report_date),
                        "fault_description":    r_fault,
                        "root_cause":           r_cause,
                        "work_performed":       st.session_state.get("r_work",""),
                        "parts_used":           st.session_state.get("r_parts",""),
                        "downtime_hours":       st.session_state.get("r_downtime",""),
                        "recommendations":      r_recs,
                        "next_maintenance_date": str(r_next_date) if r_next_date else "",
                    }
                    if polish_toggle and GEMINI_OK:
                        with st.spinner("Gemini polishing report…"):
                            for fk in ["fault_description","root_cause","recommendations"]:
                                if fields_.get(fk,"").strip():
                                    fields_[fk] = polish_field(fk.replace("_"," ").title(), fields_[fk])
                    shift_link = st.session_state.get("last_shift_report","")
                    with st.spinner("Generating PDF…"):
                        pdf_bytes = generate_pdf(fields_, r_machine_data, shift_link)
                    st.session_state["last_pdf"]        = pdf_bytes
                    st.session_state["last_pdf_fields"] = fields_
                    st.session_state["last_pdf_machine"]= r_machine_data
                    banner("PDF generated. Download on the right.", GREEN, "#E8F5EE")

        with r_right:
            section("Report preview & download")
            if "last_pdf_fields" in st.session_state:
                f_  = st.session_state["last_pdf_fields"]
                md_ = st.session_state.get("last_pdf_machine",{})
                tier_str = md_.get("risk_tier","—")
                prob_str = f"{md_['prob_failure']:.1%}" if md_.get("prob_failure") else "—"
                log_row_box({"Machine ID":  f_.get("machine_id","—"),
                             "Location":    f_.get("location","—") or "—",
                             "Technician":  f_.get("technician","—"),
                             "Report date": f_.get("report_date","—"),
                             "Risk tier":   tier_str,
                             "Failure prob":prob_str,
                             "Next maint.": f_.get("next_maintenance_date","—") or "—"})
                st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
                for lbl,key in [("Fault description","fault_description"),
                                 ("Root cause","root_cause"),
                                 ("Recommendations","recommendations")]:
                    if f_.get(key):
                        section(lbl)
                        st.markdown(f"<p style='font-family:DM Sans,sans-serif;font-size:0.9rem;"
                                    f"color:{INK};line-height:1.7'>{f_[key]}</p>",
                                    unsafe_allow_html=True)
                if f_.get("next_maintenance_date"):
                    section("Next maintenance date")
                    st.markdown(f"<p style='font-family:Syne,sans-serif;font-size:1.4rem;"
                                f"font-weight:700;color:{AMBER}'>{f_['next_maintenance_date']}</p>",
                                unsafe_allow_html=True)
                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                fname = f"maint_report_{f_.get('machine_id','').replace(' ','_')}_{f_.get('report_date','')}.pdf"
                st.download_button("Download PDF Report", data=st.session_state["last_pdf"],
                                   file_name=fname, mime="application/pdf", key="dl_pdf")
                if st.session_state.get("last_shift_report"):
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    banner("This report includes the latest shift report from Tab 1.", TEAL, "#E8F5EE")
            else:
                hint("Fill in the form on the left and click Generate PDF Report.")
                hint("Machine risk tier and failure probability are pulled automatically from the dashboard.")
                hint("If you generated a shift report in Tab 1, it will be linked in the PDF.")