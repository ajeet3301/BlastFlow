"""
BLAST BioSuite Pro — Streamlit Bioinformatics App
Run:  streamlit run app.py
Requires: GROQ_API_KEY in .streamlit/secrets.toml
"""

import io, os, re, zipfile, warnings, textwrap
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from Bio.Blast import NCBIXML, NCBIWWW
    from Bio import SeqIO, Phylo
    from Bio.Seq import Seq
    from Bio.SeqUtils import gc_fraction
    from Bio.SeqUtils.MeltingTemp import Tm_NN, Tm_GC, DNA_NN4
    from Bio.SeqUtils import molecular_weight
    BIO = True
except ImportError:
    BIO = False

try:
    from groq import Groq
    GROQ = True
except ImportError:
    GROQ = False

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BLAST BioSuite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state router ───────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"

def go(page_id: str):
    st.session_state.page = page_id
    st.rerun()

# ── GROQ key from secrets ──────────────────────────────────────────────────────
try:
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

# ══════════════════════════════════════════════════════════════════════════════
# LIGHT GLASSMORPHISM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:          #f0f4ff;
  --bg2:         #e8eeff;
  --glass:       rgba(255,255,255,0.72);
  --glass-hover: rgba(255,255,255,0.88);
  --border:      rgba(150,160,210,0.25);
  --border-h:    rgba(120,130,200,0.45);
  --shadow:      rgba(100,110,180,0.12);
  --shadow-h:    rgba(100,110,180,0.22);
  --text:        #1e1f3a;
  --muted:       #7879a0;
  --font:        'Inter', sans-serif;
  --mono:        'JetBrains Mono', monospace;

  /* Accents */
  --violet: #7c3aed; --violet-l: #ede9fe; --violet-t: rgba(124,58,237,0.12);
  --sky:    #0284c7; --sky-l:    #e0f2fe; --sky-t:    rgba(2,132,199,0.10);
  --teal:   #0d9488; --teal-l:   #ccfbf1; --teal-t:   rgba(13,148,136,0.10);
  --amber:  #d97706; --amber-l:  #fef3c7; --amber-t:  rgba(217,119,6,0.10);
  --pink:   #be185d; --pink-l:   #fce7f3; --pink-t:   rgba(190,24,93,0.10);
  --green:  #15803d; --green-l:  #dcfce7; --green-t:  rgba(21,128,61,0.10);
  --orange: #c2410c; --orange-l: #ffedd5; --orange-t: rgba(194,65,12,0.10);
  --blue:   #1d4ed8; --blue-l:   #dbeafe; --blue-t:   rgba(29,78,216,0.10);
}

/* ── Hide sidebar completely ── */
[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }
.main .block-container { max-width: 1200px; padding: 2rem 2.5rem 4rem; }

/* ── Global ── */
html, body, [class*="css"], .stApp {
    font-family: var(--font) !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* Soft gradient mesh background */
.stApp {
    background: linear-gradient(135deg,
        #f0f4ff 0%, #f5f0ff 35%, #f0f7ff 65%, #f0fffa 100%) !important;
}
.stApp::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background:
        radial-gradient(ellipse 60% 50% at 15% 20%, rgba(124,58,237,0.06) 0%, transparent 100%),
        radial-gradient(ellipse 50% 40% at 85% 70%, rgba(2,132,199,0.07) 0%, transparent 100%),
        radial-gradient(ellipse 40% 35% at 50% 90%, rgba(13,148,136,0.05) 0%, transparent 100%);
}
.main .block-container { position: relative; z-index: 1; }

/* ── Glass card ── */
.gc {
    background: var(--glass);
    backdrop-filter: blur(18px) saturate(160%);
    -webkit-backdrop-filter: blur(18px) saturate(160%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 24px 26px;
    margin-bottom: 14px;
    box-shadow: 0 2px 16px var(--shadow), inset 0 1px 0 rgba(255,255,255,0.8);
    transition: all 0.22s ease;
}
.gc:hover {
    background: var(--glass-hover);
    border-color: var(--border-h);
    box-shadow: 0 6px 28px var(--shadow-h), inset 0 1px 0 rgba(255,255,255,0.9);
    transform: translateY(-2px);
}
.gc-sm {
    background: var(--glass);
    backdrop-filter: blur(14px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 12px var(--shadow);
}

/* ── Nav card (home) ── */
.nc {
    background: var(--glass);
    backdrop-filter: blur(20px) saturate(180%);
    border: 1.5px solid var(--border);
    border-radius: 20px;
    padding: 28px 22px;
    text-align: center;
    transition: all 0.2s ease;
    box-shadow: 0 2px 14px var(--shadow), inset 0 1px 0 rgba(255,255,255,0.85);
    cursor: pointer;
    height: 100%;
}
.nc:hover {
    background: var(--glass-hover);
    box-shadow: 0 8px 32px var(--shadow-h);
    border-color: var(--border-h);
    transform: translateY(-4px);
}

/* ── Typography ── */
h1 {
    font-family: var(--font) !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.6px !important;
    line-height: 1.2 !important;
}
h2 { color: var(--violet) !important; font-weight: 600 !important; font-size: 1.2rem !important; }
h3 { color: var(--sky) !important; font-weight: 600 !important; font-size: 1rem !important; }
p, li { color: var(--text) !important; }
code, pre {
    font-family: var(--mono) !important;
    background: var(--violet-l) !important;
    border: 1px solid rgba(124,58,237,0.18) !important;
    color: var(--violet) !important;
    border-radius: 5px; font-size: 0.82rem !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--glass) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 12px var(--shadow) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important; color: var(--violet) !important; font-size: 1.5rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.78rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--violet) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important;
    font-family: var(--font) !important; font-weight: 600 !important;
    font-size: 0.88rem !important; padding: 0.5rem 1.3rem !important;
    box-shadow: 0 2px 12px rgba(124,58,237,0.3) !important;
    transition: all 0.18s !important;
}
.stButton > button:hover {
    background: #6d28d9 !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.45) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stDownloadButton"] > button {
    background: var(--teal) !important;
    box-shadow: 0 2px 12px rgba(13,148,136,0.25) !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #0f766e !important;
    box-shadow: 0 4px 20px rgba(13,148,136,0.4) !important;
}

/* ── Inputs ── */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.85) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important; color: var(--text) !important;
    font-family: var(--font) !important;
    box-shadow: inset 0 2px 4px rgba(100,110,180,0.06) !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--violet) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.85) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.6) !important;
    border: 2px dashed rgba(124,58,237,0.3) !important;
    border-radius: 14px !important;
    transition: all 0.2s;
}
[data-testid="stFileUploader"]:hover {
    background: rgba(255,255,255,0.82) !important;
    border-color: var(--violet) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.55) !important;
    border-radius: 12px 12px 0 0 !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 2px !important; padding: 5px 5px 0 !important;
    backdrop-filter: blur(10px) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: 8px 8px 0 0 !important;
    color: var(--muted) !important; font-family: var(--font) !important;
    font-weight: 500 !important; font-size: 0.85rem !important;
    transition: all 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--violet) !important; }
.stTabs [aria-selected="true"] {
    background: rgba(124,58,237,0.1) !important; color: var(--violet) !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: rgba(255,255,255,0.55) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid var(--border) !important; border-top: none !important;
    border-radius: 0 0 12px 12px !important; padding: 20px !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.7) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important; overflow: hidden !important;
    box-shadow: 0 2px 10px var(--shadow) !important;
}

/* ── Alerts ── */
.stAlert { border-radius: 12px !important; border-left-width: 3px !important; }

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.6) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important; overflow: hidden;
}

/* ── Progress ── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--violet), var(--sky)) !important;
    border-radius: 99px !important;
}
.stProgress { background: rgba(124,58,237,0.1) !important; border-radius: 99px !important; }

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.2rem 0 !important; }

/* ── Chat ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.75) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important; margin-bottom: 8px !important;
    box-shadow: 0 2px 10px var(--shadow) !important;
}

/* ── Seq display ── */
.seq-block {
    font-family: var(--mono); font-size: 0.82rem;
    background: rgba(255,255,255,0.85);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 18px; line-height: 1.9;
    overflow-x: auto; letter-spacing: 0.05em; word-break: break-all;
    box-shadow: inset 0 2px 6px rgba(100,110,180,0.07);
}
.nuc-A { color: #dc2626; font-weight: 600; }
.nuc-T { color: #2563eb; font-weight: 600; }
.nuc-G { color: #16a34a; font-weight: 600; }
.nuc-C { color: #d97706; font-weight: 600; }
.nuc-U { color: #ea580c; font-weight: 600; }

/* ── Badges ── */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 99px;
    font-size: 0.69rem; font-weight: 600; letter-spacing: 0.3px; text-transform: uppercase;
}
.badge-violet { background: var(--violet-l); color: var(--violet); border: 1px solid rgba(124,58,237,0.2); }
.badge-sky    { background: var(--sky-l);    color: var(--sky);    border: 1px solid rgba(2,132,199,0.2); }
.badge-teal   { background: var(--teal-l);   color: var(--teal);   border: 1px solid rgba(13,148,136,0.2); }
.badge-amber  { background: var(--amber-l);  color: var(--amber);  border: 1px solid rgba(217,119,6,0.2); }
.badge-pink   { background: var(--pink-l);   color: var(--pink);   border: 1px solid rgba(190,24,93,0.2); }
.badge-green  { background: var(--green-l);  color: var(--green);  border: 1px solid rgba(21,128,61,0.2); }
.badge-orange { background: var(--orange-l); color: var(--orange); border: 1px solid rgba(194,65,12,0.2); }
.badge-blue   { background: var(--blue-l);   color: var(--blue);   border: 1px solid rgba(29,78,216,0.2); }

/* ── Back button pill ── */
.back-btn { display: inline-flex; align-items: center; gap: 6px; }

/* ── Slider track ── */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { color: var(--muted) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME (light)
# ══════════════════════════════════════════════════════════════════════════════
PLY = dict(
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(248,249,255,0.5)",
    font=dict(family="Inter, sans-serif", color="#1e1f3a", size=12),
    title_font=dict(family="Inter, sans-serif", color="#7c3aed", size=14, weight="bold"),
    legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(150,160,210,0.25)", borderwidth=1),
    coloraxis_colorbar=dict(
        bgcolor="rgba(255,255,255,0.8)", bordercolor="rgba(150,160,210,0.25)",
        tickfont=dict(color="#7879a0"), title_font=dict(color="#7879a0"),
    ),
)
GRID = dict(gridcolor="rgba(150,160,210,0.12)", zerolinecolor="rgba(150,160,210,0.25)")
G_VIOLET = ["#ede9fe","#c4b5fd","#a78bfa","#7c3aed","#5b21b6"]
G_MULTI  = ["#7c3aed","#0284c7","#0d9488","#d97706","#be185d"]

def th(fig):
    fig.update_layout(**PLY)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CORE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def need_bio():
    if not BIO:
        st.error("⚠️ Run `pip install biopython` to use this feature.")
        st.stop()

def csv_bytes(df): return df.to_csv(index=False).encode()

def colorize(seq):
    m = {"A":"nuc-A","T":"nuc-T","G":"nuc-G","C":"nuc-C","U":"nuc-U"}
    return "".join(f'<span class="{m[c]}">{c}</span>' if c in m else c for c in seq.upper())

def seq_block(seq, label, badge="badge-violet"):
    return f'<div style="margin-bottom:12px;"><span class="badge {badge}">{label}</span><div class="seq-block" style="margin-top:6px;">{colorize(seq)}</div></div>'

def back_btn(label="← Back to Home"):
    if st.button(label, key="back_home"):
        go("home")

# ══════════════════════════════════════════════════════════════════════════════
# BLAST CORE
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def parse_xml(xml_bytes: bytes) -> pd.DataFrame:
    rows, handle = [], io.StringIO(xml_bytes.decode("utf-8", errors="replace"))
    for rec in NCBIXML.parse(handle):
        qid, qlen = rec.query.split()[0], rec.query_length
        for aln in rec.alignments:
            for hsp in aln.hsps:
                rows.append({
                    "Query ID": qid, "Query Length": qlen,
                    "Hit ID": aln.hit_id, "Hit Description": aln.hit_def[:90],
                    "Hit Length": aln.length, "Score": hsp.score, "Bit Score": hsp.bits,
                    "E-Value": hsp.expect,
                    "Identity (%)": round(hsp.identities / hsp.align_length * 100, 2),
                    "Alignment Length": hsp.align_length, "Gaps": hsp.gaps,
                    "Query Start": hsp.query_start, "Query End": hsp.query_end,
                    "Subject Start": hsp.sbjct_start, "Subject End": hsp.sbjct_end,
                })
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("E-Value").reset_index(drop=True)

@st.cache_data(show_spinner=False, ttl=3600)
def run_blast(seq, prog, db): 
    h = NCBIWWW.qblast(prog, db, seq)
    return parse_xml(h.read().encode())

def blast_charts(df):
    if df.empty: return
    top10 = df.nlargest(10,"Identity (%)").copy()
    top10["Lbl"] = top10["Hit ID"] + " · " + top10["Hit Description"].str[:35]
    f1 = px.bar(top10, x="Identity (%)", y="Lbl", orientation="h",
                color="Identity (%)", color_continuous_scale=G_VIOLET,
                title="Top 10 Hits by Identity %", height=380)
    f1.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    f1.update_xaxes(range=[0,105])
    st.plotly_chart(th(f1), use_container_width=True)

    sdf = df.copy(); sdf["ev"] = sdf["E-Value"].apply(lambda e: max(e,1e-200))
    f2 = px.scatter(sdf, x="Score", y="ev", color="Identity (%)",
                    size="Alignment Length",
                    hover_data=["Hit ID","Hit Description","Identity (%)"],
                    color_continuous_scale=G_MULTI, log_y=True,
                    title="E-Value vs Score (bubble = alignment length)",
                    labels={"ev":"E-Value (log)"}, height=380)
    st.plotly_chart(th(f2), use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        f3 = px.histogram(sdf[sdf["E-Value"]>0], x="E-Value", nbins=35, log_x=True,
                          color_discrete_sequence=["#7c3aed"], title="E-Value Distribution", height=300)
        f3.update_layout(bargap=0.06)
        st.plotly_chart(th(f3), use_container_width=True)
    with c2:
        f4 = px.box(df, x="Query ID", y="Bit Score", color="Query ID",
                    title="Bit Score by Query", height=300)
        f4.update_layout(showlegend=False); f4.update_xaxes(tickangle=25)
        st.plotly_chart(th(f4), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# AI EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
def get_client():
    if not GROQ: return None
    try: return Groq(api_key=GROQ_KEY)
    except: return None

def explain(df, question=""):
    client = get_client()
    if not client: return "❌ AI unavailable. Check secrets.toml."
    rows = df.head(5)[["Hit ID","Hit Description","Identity (%)","E-Value","Bit Score","Alignment Length"]].to_dict("records")
    ctx = "\n".join(
        f"  Hit {i}: {r['Hit ID']} — {r['Hit Description']}\n"
        f"    Identity: {r['Identity (%)']}%  E-Value: {r['E-Value']:.2e}  Bit Score: {r['Bit Score']:.1f}"
        for i,r in enumerate(rows,1)
    )
    sys = textwrap.dedent("""
        You are an expert bioinformatician. Explain BLAST results clearly to a graduate-level biologist.
        Structure: (1) One-paragraph summary. (2) Bullet points for top 3 hits. (3) Any flags (contamination, paralogs). (4) Next step.
        Under 350 words.
    """)
    prompt = f"BLAST results:\n{ctx}" + (f"\n\nUser question: {question}" if question.strip() else "") + "\n\nExplain."
    try:
        r = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role":"system","content":sys},{"role":"user","content":prompt}],
            temperature=0.3, max_tokens=600,
        )
        return r.choices[0].message.content
    except Exception as e: return f"❌ AI error: {e}"

def ai_widget(df, key):
    if df.empty: return
    st.markdown("---")
    st.markdown("### 🤖 AI Explainer")
    q = st.text_input("Follow-up question (optional)", placeholder="e.g. Is this contamination?", key=f"q_{key}")
    if st.button("✨ Explain results", key=f"btn_{key}"):
        with st.spinner("Thinking…"):
            ans = explain(df, q)
        st.session_state[f"ai_{key}"] = ans
    if f"ai_{key}" in st.session_state:
        with st.chat_message("assistant", avatar="🧬"):
            st.markdown(st.session_state[f"ai_{key}"])


# ══════════════════════════════════════════════════════════════════════════════
# CENTRAL DOGMA
# ══════════════════════════════════════════════════════════════════════════════
TABLES = {"Standard (1)":1,"Mitochondrial (2)":2,"Bacterial (11)":11}

def central_dogma(raw, table_id=1):
    clean = re.sub(r">.*\n?","",raw); clean = re.sub(r"\s","",clean).upper()
    bad = set(clean) - set("ATGCNRYSWKMBDHV")
    if bad: raise ValueError(f"Invalid chars: {', '.join(sorted(bad))}")
    s = Seq(clean)
    return {"DNA 5'→3'": str(s), "Complement 3'→5'": str(s.complement()),
            "Reverse Complement": str(s.reverse_complement()),
            "mRNA": str(s.transcribe()), "Protein": str(s.translate(table=table_id))}


# ══════════════════════════════════════════════════════════════════════════════
# GC PROFILER
# ══════════════════════════════════════════════════════════════════════════════
def profile(seq_str, sid="seq"):
    seq_str = seq_str.upper(); n = len(seq_str)
    if not n: return {}
    cnt = {b: seq_str.count(b) for b in "ATGCN"}
    gc = cnt["G"]+cnt["C"]; at = cnt["A"]+cnt["T"]
    try: mw = molecular_weight(Seq(seq_str), seq_type="DNA")
    except: mw = None
    return {"id":sid,"length":n,"gc":round(gc/n*100,2),"at":round(at/n*100,2),
            "cnt":cnt,"mw":mw,"tm":4*gc+2*at if n<30 else None,"raw":seq_str}

def gc_charts(p):
    pie = pd.DataFrame({"Base":list("ATGCN"),"Count":[p["cnt"][b] for b in "ATGCN"]})
    pie = pie[pie["Count"]>0]
    f = px.pie(pie, names="Base", values="Count", hole=0.42,
               color="Base",
               color_discrete_map={"A":"#ef4444","T":"#3b82f6","G":"#22c55e","C":"#f59e0b","N":"#94a3b8"},
               title="Base Composition", height=340)
    f.update_traces(textinfo="label+percent", textfont=dict(size=13),
                    marker=dict(line=dict(color="rgba(255,255,255,0.7)",width=2)))
    c1,c2 = st.columns([1,1.4])
    with c1: st.plotly_chart(th(f), use_container_width=True)
    with c2:
        raw = p.get("raw","")
        if raw:
            w = max(50,len(raw)//50)
            skews,pos = [],[]
            for i in range(0,len(raw)-w,w//2):
                ch=raw[i:i+w]; g,c=ch.count("G"),ch.count("C"); d=g+c
                skews.append((g-c)/d if d else 0); pos.append(i+w//2)
            fs = px.line(x=pos,y=skews,title="GC Skew [(G-C)/(G+C)]",
                         labels={"x":"Position (bp)","y":"GC Skew"},
                         color_discrete_sequence=["#7c3aed"],height=340)
            fs.add_hline(y=0,line_dash="dot",line_color="rgba(124,58,237,0.3)")
            fs.update_traces(line=dict(width=1.8))
            st.plotly_chart(th(fs), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PRIMER DESIGNER
# ══════════════════════════════════════════════════════════════════════════════
def design_primers(seq_raw, length=20):
    s = re.sub(r"\s|>.*","",seq_raw.upper()); s = re.sub(r"[^ATGCN]","",s)
    if len(s) < length*2+20:
        raise ValueError(f"Sequence too short (need ≥{length*2+20} bp).")
    fwd = s[:length]; rev = str(Seq(s[-length:]).reverse_complement())
    def stats(seq, name):
        gc = gc_fraction(Seq(seq))*100
        try: tm_nn = Tm_NN(seq, nn_table=DNA_NN4)
        except: tm_nn = None
        try: tm_gc = Tm_GC(seq)
        except: tm_gc = None
        rc = str(Seq(seq).reverse_complement())
        hp = any(seq[i:i+4]==rc[j:j+4] for i in range(len(seq)-3) for j in range(len(rc)-3))
        km = [seq[i:i+4] for i in range(len(seq)-3)]
        return {"name":name,"seq":seq,"len":len(seq),"gc":round(gc,1),
                "tm_nn":round(tm_nn,1) if tm_nn else None,
                "tm_gc":round(tm_gc,1) if tm_gc else None,
                "hairpin":hp,"dimer":len(km)!=len(set(km))}
    return {"fwd":stats(fwd,"Forward"),"rev":stats(rev,"Reverse"),"product":len(s)}

def primer_card(p):
    ok = not p["hairpin"] and not p["dimer"]
    issues = ("✅ No issues" if ok else
              " · ".join(filter(None,["⚠️ Hairpin" if p["hairpin"] else "",
                                       "⚠️ Self-dimer" if p["dimer"] else ""])))
    color = "#15803d" if ok else "#be185d"
    st.markdown(f"""
    <div class="gc">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
        <b style="color:#1e1f3a;">{p['name']} Primer</b>
        <span style="font-size:0.75rem;font-weight:600;color:{color};">{issues}</span>
      </div>
      <div class="seq-block" style="margin-bottom:12px;">{colorize(p['seq'])}</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;font-size:0.82rem;">
        <div><span style="color:#7879a0;">Length</span><br><b>{p['len']} bp</b></div>
        <div><span style="color:#7879a0;">GC</span><br><b>{p['gc']}%</b></div>
        <div><span style="color:#7879a0;">Tm (NN)</span><br><b>{"N/A" if not p['tm_nn'] else f"{p['tm_nn']} °C"}</b></div>
        <div><span style="color:#7879a0;">Tm (GC)</span><br><b>{"N/A" if not p['tm_gc'] else f"{p['tm_gc']} °C"}</b></div>
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE — clickable nav cards
# ══════════════════════════════════════════════════════════════════════════════
TOOLS = [
    ("xml",     "📄", "XML Parser",     "violet", "Upload BLAST XML → filterable table + CSV"),
    ("blast",   "🌐", "Online BLAST",   "sky",    "Submit FASTA → live NCBI qblast"),
    ("batch",   "📦", "Batch Processor","teal",   "Multi-file FASTA → combined results ZIP"),
    ("ai",      "🤖", "AI Explainer",   "amber",  "LLaMA 3 explains top hits plainly"),
    ("dogma",   "🔀", "Central Dogma",  "pink",   "DNA → mRNA → Protein translation"),
    ("gc",      "📊", "GC Dashboard",   "green",  "GC%, composition, skew profiling"),
    ("phylo",   "🌿", "Phylo Viewer",   "orange", "Render Newick / NEXUS phylo trees"),
    ("primer",  "🔬", "Primer Designer","blue",   "Design primers with Tm & hairpin check"),
]

ACCENT_COLORS = {
    "violet": ("#7c3aed","#ede9fe","rgba(124,58,237,0.12)","rgba(124,58,237,0.25)"),
    "sky":    ("#0284c7","#e0f2fe","rgba(2,132,199,0.12)","rgba(2,132,199,0.25)"),
    "teal":   ("#0d9488","#ccfbf1","rgba(13,148,136,0.12)","rgba(13,148,136,0.25)"),
    "amber":  ("#d97706","#fef3c7","rgba(217,119,6,0.12)","rgba(217,119,6,0.25)"),
    "pink":   ("#be185d","#fce7f3","rgba(190,24,93,0.12)","rgba(190,24,93,0.25)"),
    "green":  ("#15803d","#dcfce7","rgba(21,128,61,0.12)","rgba(21,128,61,0.25)"),
    "orange": ("#c2410c","#ffedd5","rgba(194,65,12,0.12)","rgba(194,65,12,0.25)"),
    "blue":   ("#1d4ed8","#dbeafe","rgba(29,78,216,0.12)","rgba(29,78,216,0.25)"),
}

def home():
    st.markdown("""
    <div style="text-align:center;padding:20px 0 32px;">
      <div style="font-size:2.8rem;margin-bottom:8px;">🧬</div>
      <h1 style="margin:0 0 6px;font-size:2.2rem!important;">BLAST BioSuite</h1>
      <p style="color:#7879a0;font-size:1rem;margin:0;">Select a tool to get started</p>
    </div>
    """, unsafe_allow_html=True)

    for row_start in range(0, len(TOOLS), 4):
        row = TOOLS[row_start:row_start+4]
        cols = st.columns(len(row))
        for col, (pid, icon, title, accent, desc) in zip(cols, row):
            c_text, c_bg, c_border, c_border_h = ACCENT_COLORS[accent]
            with col:
                # Show visual card + button stacked
                st.markdown(f"""
                <div style="
                  background:rgba(255,255,255,0.72);
                  backdrop-filter:blur(18px) saturate(160%);
                  border:1.5px solid {c_border};
                  border-radius:20px;
                  padding:24px 20px 14px;
                  margin-bottom:0px;
                  box-shadow:0 2px 16px rgba(100,110,180,0.1),inset 0 1px 0 rgba(255,255,255,0.85);
                ">
                  <div style="
                    width:44px;height:44px;border-radius:12px;
                    background:{c_bg};border:1px solid {c_border_h};
                    display:flex;align-items:center;justify-content:center;
                    font-size:1.4rem;margin-bottom:12px;
                  ">{icon}</div>
                  <div style="font-weight:700;color:#1e1f3a;font-size:0.95rem;margin-bottom:4px;">{title}</div>
                  <div style="font-size:0.78rem;color:#7879a0;line-height:1.45;margin-bottom:14px;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Open {title}", key=f"nav_{pid}", use_container_width=True):
                    go(pid)
        st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER HELPER
# ══════════════════════════════════════════════════════════════════════════════
def page_header(icon, title, accent="violet"):
    c_text, c_bg, c_border, _ = ACCENT_COLORS[accent]
    back = st.button("← Home", key="back_top")
    if back: go("home")
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;margin:8px 0 20px;">
      <div style="width:48px;height:48px;border-radius:14px;background:{c_bg};
                  border:1px solid {c_border};display:flex;align-items:center;
                  justify-content:center;font-size:1.5rem;flex-shrink:0;">{icon}</div>
      <h1 style="margin:0;font-size:1.75rem!important;">{title}</h1>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: XML PARSER
# ══════════════════════════════════════════════════════════════════════════════
def page_xml():
    need_bio()
    page_header("📄","BLAST XML Parser","violet")

    up = st.file_uploader("Upload BLAST XML (outfmt 5)", type=["xml"])
    if not up: return

    with st.spinner("Parsing…"):
        df = parse_xml(up.read())

    if df.empty:
        st.error("No hits found. Check the file is valid BLAST XML (outfmt 5)."); return

    st.success(f"✅ {len(df):,} HSPs · {df['Query ID'].nunique()} queries")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("HSPs", f"{len(df):,}")
    c2.metric("Queries", df["Query ID"].nunique())
    c3.metric("Unique Hits", df["Hit ID"].nunique())
    c4.metric("Best E-Value", f"{df['E-Value'].min():.2e}")

    t1,t2,t3 = st.tabs(["📋 Table","📊 Charts","🤖 AI"])
    with t1:
        with st.expander("Filter", expanded=False):
            ev = st.slider("Max E-Value",0.0,1.0,1.0,0.001,format="%.3f")
            mi = st.slider("Min Identity %",0.0,100.0,0.0,1.0)
            cols = st.multiselect("Columns",df.columns.tolist(),default=df.columns.tolist())
        fdf = df[(df["E-Value"]<=ev)&(df["Identity (%)"]>=mi)][cols]
        st.dataframe(fdf, use_container_width=True, height=400)
        st.download_button("⬇️ Download CSV", csv_bytes(fdf),
                           f"blast_{Path(up.name).stem}.csv","text/csv")
    with t2: blast_charts(df)
    with t3: ai_widget(df, "xml")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ONLINE BLAST
# ══════════════════════════════════════════════════════════════════════════════
def page_blast():
    need_bio()
    page_header("🌐","Online NCBI BLAST","sky")
    st.warning("⏳ Queries take 30–120 seconds on NCBI servers.", icon="⚠️")

    c1,c2 = st.columns(2)
    with c1: prog = st.selectbox("Program",["blastn","blastp","blastx","tblastn","tblastx"])
    with c2: db   = st.selectbox("Database",["nt","nr","swissprot","refseq_rna","refseq_protein"])

    seq = st.text_area("FASTA sequence", height=160, placeholder=">seq\nATGCGT…")
    fu  = st.file_uploader("…or upload FASTA", type=["fasta","fa","fna","faa","txt"])
    if fu: seq = fu.read().decode()

    if st.button("🚀 Submit to NCBI", disabled=not seq.strip()):
        with st.spinner(f"Running {prog} on NCBI…"):
            try: df = run_blast(seq.strip(), prog, db)
            except Exception as e: st.error(f"Failed: {e}"); return
        if df.empty: st.warning("No significant hits returned."); return
        st.success(f"✅ {len(df):,} HSPs")
        c1,c2,c3 = st.columns(3)
        c1.metric("Unique Hits", df["Hit ID"].nunique())
        c2.metric("Best E-Value",f"{df['E-Value'].min():.2e}")
        c3.metric("Max Identity",f"{df['Identity (%)'].max():.1f}%")
        t1,t2,t3 = st.tabs(["📋 Table","📊 Charts","🤖 AI"])
        with t1:
            st.dataframe(df, use_container_width=True, height=380)
            st.download_button("⬇️ Download CSV", csv_bytes(df),"online_blast.csv","text/csv")
        with t2: blast_charts(df)
        with t3: ai_widget(df,"online")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════
def page_batch():
    need_bio()
    page_header("📦","Batch FASTA Processor","teal")
    st.warning("One NCBI query per file — 5 files ≈ 5–10 min.", icon="⚠️")

    c1,c2 = st.columns(2)
    with c1: bp = st.selectbox("Program",["blastn","blastp","blastx"],key="bp")
    with c2: bd = st.selectbox("Database",["nt","nr","swissprot"],key="bd")

    files = st.file_uploader("Upload FASTA files",type=["fasta","fa","fna","faa"],accept_multiple_files=True)
    if not files: return
    st.info(f"📁 {len(files)} file(s) queued.")

    if st.button(f"▶️ Run Batch ({len(files)} files)"):
        all_dfs, fdfs = [], {}
        prog = st.progress(0,"Starting…"); status = st.empty()
        for i,f in enumerate(files):
            seq = f.read().decode("utf-8","replace")
            status.markdown(f"⏳ **{f.name}** ({i+1}/{len(files)})…")
            try:
                dfi = run_blast(seq,bp,bd)
                dfi.insert(0,"Source File",f.name)
                fdfs[f.name] = dfi; all_dfs.append(dfi)
                status.success(f"✅ `{f.name}` → {len(dfi):,} HSPs")
            except Exception as e: status.error(f"❌ `{f.name}`: {e}")
            prog.progress((i+1)/len(files),f"{i+1}/{len(files)}")
        prog.empty(); status.empty()
        if not all_dfs: st.error("All queries failed."); return

        combined = pd.concat(all_dfs, ignore_index=True)
        st.success(f"🎉 {len(combined):,} total HSPs from {len(all_dfs)} queries.")

        smry = combined.groupby("Source File").agg(
            HSPs=("Score","count"), Hits=("Hit ID","nunique"),
            BestE=("E-Value","min"), MaxIdent=("Identity (%)","max")
        ).reset_index().round(4)
        st.dataframe(smry, use_container_width=True)

        d1,d2 = st.columns(2)
        with d1:
            st.download_button("⬇️ Combined CSV", csv_bytes(combined),"batch_blast.csv","text/csv")
        with d2:
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED) as zf:
                for fn,dfi in fdfs.items(): zf.writestr(Path(fn).stem+"_blast.csv",dfi.to_csv(index=False))
            zbuf.seek(0)
            st.download_button("📦 Download ZIP",zbuf.getvalue(),"batch_blast.zip","application/zip")

        st.markdown("---"); blast_charts(combined); ai_widget(combined,"batch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
def page_ai():
    page_header("🤖","AI Explainer","amber")
    if not GROQ: st.error("Run `pip install groq` to enable AI."); return
    if not GROQ_KEY: st.warning("⚠️ GROQ_API_KEY not found in secrets.toml.")

    up = st.file_uploader("Upload BLAST XML", type=["xml"])
    if up:
        need_bio()
        with st.spinner("Parsing…"): df = parse_xml(up.read())
        if df.empty: st.error("No hits found."); return
        st.success(f"✅ {len(df):,} HSPs loaded.")
        st.dataframe(df.head(5)[["Hit ID","Hit Description","Identity (%)","E-Value","Bit Score"]],
                     use_container_width=True)
        ai_widget(df,"standalone")

    st.markdown("---")
    st.markdown("### 💬 Ask a bioinformatics question")
    if "chat" not in st.session_state: st.session_state.chat = []
    for m in st.session_state.chat:
        with st.chat_message(m["role"], avatar="🧬" if m["role"]=="assistant" else "🧑‍🔬"):
            st.markdown(m["content"])
    if prompt := st.chat_input("Ask anything about BLAST, sequences, biology…"):
        st.session_state.chat.append({"role":"user","content":prompt})
        with st.chat_message("user",avatar="🧑‍🔬"): st.markdown(prompt)
        client = get_client()
        if client:
            with st.chat_message("assistant",avatar="🧬"):
                with st.spinner("…"):
                    try:
                        msgs = [{"role":"system","content":"You are an expert bioinformatician. Answer clearly and concisely."}] + st.session_state.chat
                        r = client.chat.completions.create(model="llama3-70b-8192",messages=msgs,max_tokens=500,temperature=0.3)
                        ans = r.choices[0].message.content
                    except Exception as e: ans = f"❌ {e}"
                    st.markdown(ans)
                    st.session_state.chat.append({"role":"assistant","content":ans})
        else:
            with st.chat_message("assistant",avatar="🧬"): st.markdown("❌ AI unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CENTRAL DOGMA
# ══════════════════════════════════════════════════════════════════════════════
def page_dogma():
    need_bio()
    page_header("🔀","Central Dogma Tool","pink")
    st.markdown("DNA → Complement → mRNA → Protein. Each base is colour-coded.")

    c1,c2 = st.columns([2,1])
    with c1: dna = st.text_area("DNA sequence (FASTA or raw)",height=150,placeholder=">gene\nATGCGT…")
    with c2:
        tbl = st.selectbox("Genetic code table", list(TABLES.keys()))
        show_rc  = st.checkbox("Show reverse complement", True)
        show_rna = st.checkbox("Show mRNA", True)
        show_aa  = st.checkbox("Show protein", True)

    if st.button("🔀 Translate", disabled=not dna.strip()):
        try:
            res = central_dogma(dna, TABLES[tbl])
            gc  = round(gc_fraction(Seq(res["DNA 5'→3'"]))*100,1)
            st.markdown(f"""
            <div style="display:flex;gap:8px;margin:10px 0 16px;flex-wrap:wrap;">
              <span class="badge badge-violet">{len(res["DNA 5'→3'"])} bp</span>
              <span class="badge badge-green">GC {gc}%</span>
              <span class="badge badge-sky">{tbl}</span>
            </div>""", unsafe_allow_html=True)
            st.markdown(seq_block(res["DNA 5'→3'"],"DNA 5'→3'","badge-violet"), unsafe_allow_html=True)
            st.markdown(seq_block(res["Complement 3'→5'"],"Complement 3'→5'","badge-sky"), unsafe_allow_html=True)
            if show_rc:
                st.markdown(seq_block(res["Reverse Complement"],"Reverse Complement","badge-teal"), unsafe_allow_html=True)
            if show_rna:
                st.markdown(seq_block(res["mRNA"],"mRNA","badge-amber"), unsafe_allow_html=True)
            if show_aa:
                prot = res["Protein"]
                st.markdown(f'<span class="badge badge-pink">Protein</span><div class="seq-block" style="margin-top:6px;color:#be185d;letter-spacing:0.15em">{prot}</div>', unsafe_allow_html=True)
                aa_df = pd.Series(list(prot.replace("*",""))).value_counts().reset_index()
                aa_df.columns = ["AA","Count"]
                f = px.bar(aa_df.head(15),x="AA",y="Count",color="Count",
                           color_continuous_scale=G_VIOLET,title="Amino Acid Frequency",height=280)
                f.update_layout(coloraxis_showscale=False)
                st.plotly_chart(th(f), use_container_width=True)
        except ValueError as e: st.error(f"Error: {e}")

    with st.expander("💡 Example sequence"):
        st.code(">TP53_partial\nATGGAGGAGCCGCAGTCAGATCCTAGCGTTGAATGAGAGGAATTTGAGGGAGCCGTGGGTGGG", language="text")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GC DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_gc():
    need_bio()
    page_header("📊","GC Content Dashboard","green")

    up = st.file_uploader("Upload FASTA file", type=["fasta","fa","fna","txt"])
    manual = st.text_area("…or paste raw DNA", height=100, placeholder="ATGCGATCGATCG…")

    seqs = []
    if up:
        txt = up.read().decode("utf-8","replace")
        recs = list(SeqIO.parse(io.StringIO(txt),"fasta"))
        if recs: seqs = [(r.id,str(r.seq)) for r in recs[:20]]
        else:
            raw = re.sub(r"\s","",txt.upper())
            if re.match(r"^[ATGCN]+$",raw): seqs = [("uploaded",raw)]
    elif manual.strip():
        raw = re.sub(r"\s","",manual.upper())
        seqs = [("manual",raw)]

    if not seqs: return

    profiles = [profile(s,sid) for sid,s in seqs]
    st.success(f"✅ {len(profiles)} sequence(s) loaded.")

    sel = st.selectbox("Sequence",options=[p["id"] for p in profiles]) if len(profiles)>1 else profiles[0]["id"]
    p = next(x for x in profiles if x["id"]==sel)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Length", f"{p['length']:,} bp")
    c2.metric("GC Content", f"{p['gc']}%")
    c3.metric("AT Content", f"{p['at']}%")
    c4.metric("Mol. Weight", f"{p['mw']/1000:.1f} kDa" if p['mw'] else "N/A")

    gc_charts(p)

    if len(profiles)>1:
        st.markdown("### All Sequences")
        sdf = pd.DataFrame([{"ID":x["id"],"Length":x["length"],"GC%":x["gc"],"AT%":x["at"],
                              "MW (kDa)":round(x["mw"]/1000,1) if x["mw"] else None} for x in profiles])
        st.dataframe(sdf, use_container_width=True)
        st.download_button("⬇️ Download CSV", csv_bytes(sdf),"gc_summary.csv","text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PHYLO VIEWER
# ══════════════════════════════════════════════════════════════════════════════
def page_phylo():
    need_bio()
    page_header("🌿","Phylogenetic Tree Viewer","orange")

    c1,c2 = st.columns([2,1])
    with c1: up  = st.file_uploader("Upload tree file",type=["nwk","tree","dnd","nex","nexus","txt"])
    with c2: fmt = st.selectbox("Format",["newick","nexus","nexml","phyloxml"])

    if not up:
        st.markdown("""
        <div class="gc" style="text-align:center;padding:40px 20px;">
          <div style="font-size:2.5rem;margin-bottom:10px;">🌿</div>
          <div style="color:#7879a0;">Upload a Newick or NEXUS tree file</div>
        </div>""", unsafe_allow_html=True)
        with st.expander("Example Newick"):
            st.code("((Homo_sapiens:0.12,Pan_troglodytes:0.08):0.05,(Mus_musculus:0.25,Rattus_norvegicus:0.22):0.10,Danio_rerio:0.45);")
        return

    try: tree = Phylo.read(io.StringIO(up.read().decode("utf-8","replace")), fmt)
    except Exception as e: st.error(f"Could not parse tree: {e}"); return

    terms = tree.get_terminals()
    fig,ax = plt.subplots(figsize=(12, max(5,len(terms)*0.38)))
    fig.patch.set_facecolor("#f8f9ff"); ax.set_facecolor("#f8f9ff")
    Phylo.draw(tree, axes=ax, do_show=False, show_confidence=True)
    ax.tick_params(colors="#7879a0"); ax.xaxis.label.set_color("#7879a0")
    for spine in ax.spines.values(): spine.set_edgecolor("rgba(150,160,210,0.3)")
    for txt in ax.texts: txt.set_color("#1e1f3a"); txt.set_fontsize(9)
    for ln in ax.get_lines(): ln.set_color("#7c3aed"); ln.set_alpha(0.7); ln.set_linewidth(1.3)
    ax.set_title("Phylogenetic Tree", color="#7c3aed", fontsize=14, pad=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close(fig)

    depths = [tree.distance(t) for t in terms]
    c1,c2,c3 = st.columns(3)
    c1.metric("Terminal Taxa", len(terms))
    c2.metric("Internal Nodes", len(tree.get_nonterminals()))
    c3.metric("Max Branch Depth", f"{max(depths):.4f}" if depths else "N/A")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRIMER DESIGNER
# ══════════════════════════════════════════════════════════════════════════════
def page_primer():
    need_bio()
    page_header("🔬","PCR Primer Designer","blue")

    c1,c2 = st.columns([2,1])
    with c1:
        seq = st.text_area("Target DNA sequence",height=180,placeholder=">target\nATGCGT…")
        fu  = st.file_uploader("…or upload FASTA",type=["fasta","fa","fna","txt"],key="pfu")
        if fu: seq = fu.read().decode()
    with c2:
        plen = st.slider("Primer length (bp)",15,30,20)
        st.markdown("""
        <div class="gc-sm" style="margin-top:8px;">
          <div style="font-size:0.78rem;color:#7879a0;line-height:1.8;">
            <b style="color:#1e1f3a;">Ideal specs</b><br>
            GC: 40–60% · Tm: 55–65 °C<br>
            ΔTm &lt; 5 °C · No 3′ hairpins
          </div>
        </div>""", unsafe_allow_html=True)

    if st.button("🔬 Design Primers", disabled=not seq.strip()):
        try:
            res = design_primers(seq, plen)
            c1,c2 = st.columns(2)
            with c1: primer_card(res["fwd"])
            with c2: primer_card(res["rev"])

            st.markdown(f"""
            <div class="gc-sm" style="text-align:center;margin-top:4px;">
              <span style="color:#7879a0;font-size:0.82rem;">Expected PCR Product</span><br>
              <b style="font-size:1.5rem;color:#1d4ed8;">{res['product']:,} bp</b>
            </div>""", unsafe_allow_html=True)

            fwd_tm = res["fwd"]["tm_nn"] or res["fwd"]["tm_gc"] or 0
            rev_tm = res["rev"]["tm_nn"] or res["rev"]["tm_gc"] or 0
            dtm = abs(fwd_tm - rev_tm)
            if dtm > 5: st.warning(f"ΔTm = {dtm:.1f} °C (>5 °C may reduce PCR efficiency)")
            else: st.success(f"✅ ΔTm = {dtm:.1f} °C — primers well matched")

            fig = go.Figure()
            for p,col in [(res["fwd"],"#7c3aed"),(res["rev"],"#0284c7")]:
                fig.add_trace(go.Bar(name=p["name"],x=["Tm (NN)","GC (%)"],
                                     y=[p["tm_nn"] or 0, p["gc"]],
                                     marker_color=col,opacity=0.85))
            fig.update_layout(barmode="group",title="Primer Comparison",height=260)
            st.plotly_chart(th(fig), use_container_width=True)

        except ValueError as e: st.error(f"Design error: {e}")

    with st.expander("💡 Demo — GAPDH fragment"):
        st.code(">GAPDH_demo\nATGGGGAAGGTGAAGGTCGGAGTCAACGGATTTGGTCGTATTGGGCGCCTGGTCACCAGGGCTGCTTTTAACTCTGGTAAAGTGGATATTGTTGCCATCAATGACCCCTTCATTGACCTCAACTACATGGTCTACATGTTCCAGTATGACTCCACTCACGGCAAATTC",language="text")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
PAGE = st.session_state.page
if   PAGE == "home":   home()
elif PAGE == "xml":    page_xml()
elif PAGE == "blast":  page_blast()
elif PAGE == "batch":  page_batch()
elif PAGE == "ai":     page_ai()
elif PAGE == "dogma":  page_dogma()
elif PAGE == "gc":     page_gc()
elif PAGE == "phylo":  page_phylo()
elif PAGE == "primer": page_primer()
else: go("home")
