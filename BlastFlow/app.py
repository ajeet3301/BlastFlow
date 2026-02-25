"""
BLAST BioSuite Pro — Streamlit Bioinformatics App
Run:  streamlit run app.py
Requires: GROQ_API_KEY in .streamlit/secrets.toml
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import io
import os
import re
import math
import zipfile
import warnings
import textwrap
from pathlib import Path

# ─── Third-party ─────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Biopython ────────────────────────────────────────────────────────────────
try:
    from Bio.Blast import NCBIXML, NCBIWWW
    from Bio import SeqIO, Phylo
    from Bio.Seq import Seq
    from Bio.SeqUtils import gc_fraction
    from Bio.SeqUtils.MeltingTemp import Tm_NN, Tm_GC, salt_correction
    from Bio.SeqUtils.MeltingTemp import DNA_NN4
    from Bio.SeqUtils import molecular_weight
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

# ─── Groq LLM client ─────────────────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BLAST BioSuite Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLASSMORPHISM CSS — Full Design System
# ══════════════════════════════════════════════════════════════════════════════

GLASS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Playfair+Display:wght@700&display=swap');

/* ── Reset & Root Variables ────────────────────────────────────────────── */
:root {
  --bg-deep:        #020b14;
  --bg-mid:         #041626;
  --glass-bg:       rgba(10, 30, 55, 0.55);
  --glass-bg-light: rgba(20, 50, 85, 0.40);
  --glass-border:   rgba(100, 200, 255, 0.18);
  --glass-border-h: rgba(100, 200, 255, 0.40);
  --glow-blue:      rgba(56, 189, 248, 0.35);
  --glow-purple:    rgba(167, 139, 250, 0.25);
  --glow-teal:      rgba(45, 212, 191, 0.25);
  --text-primary:   #e0f2fe;
  --text-secondary: #7dd3fc;
  --text-muted:     #4a8aaa;
  --accent-blue:    #38bdf8;
  --accent-purple:  #a78bfa;
  --accent-teal:    #2dd4bf;
  --accent-pink:    #f472b6;
  --accent-green:   #4ade80;
  --font-body:      'Space Grotesk', sans-serif;
  --font-mono:      'JetBrains Mono', monospace;
  --font-display:   'Playfair Display', serif;
}

/* ── Global Body ───────────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: var(--font-body) !important;
    background: var(--bg-deep) !important;
    color: var(--text-primary) !important;
}

/* Animated aurora background */
.stApp::before {
    content: '';
    position: fixed;
    top: -40%;
    left: -20%;
    width: 80vw;
    height: 80vh;
    background: radial-gradient(ellipse, rgba(56,189,248,0.07) 0%, transparent 65%);
    animation: drift1 18s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: -30%;
    right: -10%;
    width: 60vw;
    height: 60vh;
    background: radial-gradient(ellipse, rgba(167,139,250,0.06) 0%, transparent 65%);
    animation: drift2 22s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
@keyframes drift1 {
    from { transform: translate(0, 0) rotate(0deg); }
    to   { transform: translate(5vw, 5vh) rotate(8deg); }
}
@keyframes drift2 {
    from { transform: translate(0, 0) rotate(0deg); }
    to   { transform: translate(-4vw, -4vh) rotate(-6deg); }
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,
        rgba(2, 15, 30, 0.95) 0%,
        rgba(4, 22, 44, 0.95) 100%) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid var(--glass-border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px;
    border-radius: 8px;
    transition: all 0.2s;
    display: block;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(56,189,248,0.1) !important;
}

/* ── Main Content Area ─────────────────────────────────────────────────── */
.main .block-container {
    padding-top: 2rem;
    position: relative;
    z-index: 1;
}

/* ── Glass Card Component ──────────────────────────────────────────────── */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 16px;
    box-shadow:
        0 8px 32px rgba(0,0,0,0.4),
        inset 0 1px 0 rgba(255,255,255,0.05);
    transition: border-color 0.3s, box-shadow 0.3s;
}
.glass-card:hover {
    border-color: var(--glass-border-h);
    box-shadow:
        0 12px 40px rgba(0,0,0,0.5),
        0 0 20px var(--glow-blue),
        inset 0 1px 0 rgba(255,255,255,0.07);
}
.glass-card-sm {
    background: var(--glass-bg-light);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

/* ── Typography ────────────────────────────────────────────────────────── */
h1 {
    font-family: var(--font-display) !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #2dd4bf 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem !important;
}
h2 {
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    color: var(--accent-blue) !important;
    font-size: 1.4rem !important;
}
h3 {
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    font-size: 1.1rem !important;
}
p, li, span, label { color: var(--text-primary) !important; }
code, pre {
    font-family: var(--font-mono) !important;
    background: rgba(56,189,248,0.08) !important;
    border: 1px solid rgba(56,189,248,0.15) !important;
    color: var(--accent-teal) !important;
    border-radius: 6px;
}

/* ── Metric Cards ──────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
    transition: all 0.25s ease;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(56,189,248,0.35) !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.4), 0 0 14px var(--glow-blue);
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    color: var(--accent-blue) !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }

/* ── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(99,102,241,0.15)) !important;
    backdrop-filter: blur(8px) !important;
    color: var(--accent-blue) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.3px !important;
    padding: 0.5rem 1.4rem !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(56,189,248,0.28), rgba(99,102,241,0.28)) !important;
    border-color: var(--accent-blue) !important;
    color: white !important;
    box-shadow: 0 0 22px var(--glow-blue) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, rgba(45,212,191,0.15), rgba(74,222,128,0.15)) !important;
    color: var(--accent-teal) !important;
    border-color: rgba(45,212,191,0.3) !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, rgba(45,212,191,0.3), rgba(74,222,128,0.3)) !important;
    box-shadow: 0 0 22px var(--glow-teal) !important;
}

/* ── Inputs & Selects ──────────────────────────────────────────────────── */
.stTextArea textarea, .stTextInput input, .stSelectbox > div > div {
    background: rgba(10,30,55,0.7) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 14px var(--glow-blue) !important;
}

/* ── File Uploader ─────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(10,30,55,0.5) !important;
    border: 2px dashed var(--glass-border) !important;
    border-radius: 14px !important;
    transition: border-color 0.25s, background 0.25s;
}
[data-testid="stFileUploader"]:hover {
    background: rgba(56,189,248,0.06) !important;
    border-color: rgba(56,189,248,0.4) !important;
}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(10,30,55,0.5) !important;
    backdrop-filter: blur(8px) !important;
    border-radius: 12px 12px 0 0 !important;
    border-bottom: 1px solid var(--glass-border) !important;
    gap: 4px !important;
    padding: 6px 6px 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    color: var(--text-muted) !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    border: 1px solid transparent !important;
    transition: all 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--accent-blue) !important;
    background: rgba(56,189,248,0.08) !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(56,189,248,0.15) !important;
    color: var(--accent-blue) !important;
    border-color: var(--glass-border) !important;
    border-bottom-color: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: rgba(10,30,55,0.35) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid var(--glass-border) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 20px !important;
}

/* ── DataFrames ────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    background: rgba(10,30,55,0.5) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(8px) !important;
    overflow: hidden !important;
}

/* ── Alerts ────────────────────────────────────────────────────────────── */
.stAlert {
    border-radius: 12px !important;
    backdrop-filter: blur(8px) !important;
    border-left-width: 3px !important;
}

/* ── Expanders ─────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: var(--glass-bg-light) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── Progress bar ──────────────────────────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #2dd4bf) !important;
    border-radius: 99px !important;
}
.stProgress {
    background: rgba(56,189,248,0.1) !important;
    border-radius: 99px !important;
}

/* ── Divider ───────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid var(--glass-border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Chat messages (LLM Explainer) ────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 14px !important;
    margin-bottom: 10px !important;
}

/* ── Slider ────────────────────────────────────────────────────────────── */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { color: var(--text-muted) !important; }

/* ── Sequence display ──────────────────────────────────────────────────── */
.seq-block {
    font-family: var(--font-mono);
    font-size: 0.82rem;
    background: rgba(5, 20, 40, 0.8);
    border: 1px solid var(--glass-border);
    border-radius: 10px;
    padding: 16px 20px;
    line-height: 1.9;
    overflow-x: auto;
    letter-spacing: 0.05em;
    word-break: break-all;
}
.nuc-A { color: #f87171; font-weight: 600; }
.nuc-T { color: #60a5fa; font-weight: 600; }
.nuc-G { color: #4ade80; font-weight: 600; }
.nuc-C { color: #facc15; font-weight: 600; }
.nuc-U { color: #fb923c; font-weight: 600; }

/* ── Badge / Chip ──────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}
.badge-blue   { background: rgba(56,189,248,0.15); color: #38bdf8; border: 1px solid rgba(56,189,248,0.3); }
.badge-green  { background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
.badge-purple { background: rgba(167,139,250,0.15); color: #a78bfa; border: 1px solid rgba(167,139,250,0.3); }
.badge-pink   { background: rgba(244,114,182,0.15); color: #f472b6; border: 1px solid rgba(244,114,182,0.3); }
</style>
"""

st.markdown(GLASS_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(4,22,44,0.0)",
    plot_bgcolor="rgba(4,22,44,0.0)",
    font=dict(family="Space Grotesk, sans-serif", color="#e0f2fe", size=12),
    title_font=dict(family="Space Grotesk, sans-serif", color="#7dd3fc", size=15),
    legend=dict(
        bgcolor="rgba(10,30,55,0.6)",
        bordercolor="rgba(100,200,255,0.18)",
        borderwidth=1,
    ),
    coloraxis_colorbar=dict(
        bgcolor="rgba(10,30,55,0.6)",
        bordercolor="rgba(100,200,255,0.18)",
        tickfont=dict(color="#7dd3fc"),
        title_font=dict(color="#7dd3fc"),
    ),
)
GRID_STYLE = dict(gridcolor="rgba(56,189,248,0.08)", zerolinecolor="rgba(56,189,248,0.15)")
GRAD_SEQ   = ["#0c4a6e", "#0284c7", "#38bdf8", "#7dd3fc", "#2dd4bf"]
GRAD_PURP  = ["#2e1065", "#6d28d9", "#a78bfa", "#f472b6"]


def apply_theme(fig):
    """Apply shared glassmorphism theme to any Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(**GRID_STYLE)
    fig.update_yaxes(**GRID_STYLE)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def check_biopython():
    if not BIOPYTHON_AVAILABLE:
        st.error("⚠️ **Biopython not installed.** Run `pip install biopython`.")
        st.stop()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def colorize_sequence(seq: str, seq_type: str = "DNA") -> str:
    """Wrap each nucleotide/amino acid in a colored HTML span."""
    color_map = {
        "A": "nuc-A", "T": "nuc-T", "G": "nuc-G",
        "C": "nuc-C", "U": "nuc-U",
    }
    html = []
    for ch in seq.upper():
        cls = color_map.get(ch, "")
        if cls:
            html.append(f'<span class="{cls}">{ch}</span>')
        else:
            html.append(ch)
    # Break into chunks of 60 for readability
    flat = "".join(html)
    # Insert line break every 60 raw characters (approx — HTML safe wrap)
    return flat


def seq_html_block(seq: str, label: str, badge_class: str = "badge-blue") -> str:
    return f"""
    <div style="margin-bottom:14px;">
      <div style="margin-bottom:6px;">
        <span class="badge {badge_class}">{label}</span>
      </div>
      <div class="seq-block">{colorize_sequence(seq)}</div>
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
# BLAST CORE FUNCTIONS  (unchanged logic, shared across pages)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def parse_blast_xml(xml_bytes: bytes) -> pd.DataFrame:
    """
    Parse BLAST XML bytes → tidy DataFrame.
    @st.cache_data: memoized by file bytes — same upload = zero re-parse.
    """
    records = []
    handle  = io.StringIO(xml_bytes.decode("utf-8", errors="replace"))
    for blast_record in NCBIXML.parse(handle):
        query_id  = blast_record.query.split()[0]
        query_len = blast_record.query_length
        for alignment in blast_record.alignments:
            for hsp in alignment.hsps:
                ident_pct = round((hsp.identities / hsp.align_length) * 100, 2)
                records.append({
                    "Query ID":         query_id,
                    "Query Length":     query_len,
                    "Hit ID":           alignment.hit_id,
                    "Hit Description":  alignment.hit_def[:90],
                    "Hit Length":       alignment.length,
                    "Score":            hsp.score,
                    "Bit Score":        hsp.bits,
                    "E-Value":          hsp.expect,
                    "Identity (%)":     ident_pct,
                    "Alignment Length": hsp.align_length,
                    "Gaps":             hsp.gaps,
                    "Query Start":      hsp.query_start,
                    "Query End":        hsp.query_end,
                    "Subject Start":    hsp.sbjct_start,
                    "Subject End":      hsp.sbjct_end,
                })
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).sort_values("E-Value").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def run_online_blast(sequence: str, program: str, database: str) -> pd.DataFrame:
    """
    Live NCBI BLAST via qblast.
    ttl=3600 → cached 1 hr; identical sequences never re-query NCBI.
    """
    result_handle = NCBIWWW.qblast(program, database, sequence)
    xml_bytes     = result_handle.read().encode("utf-8")
    return parse_blast_xml(xml_bytes)


def build_blast_charts(df: pd.DataFrame):
    """Four interactive Plotly charts from a BLAST DataFrame."""
    if df.empty:
        st.warning("No data to visualize.")
        return

    # 1. Top-10 Identity bar
    top10 = df.nlargest(10, "Identity (%)").copy()
    top10["Label"] = top10["Hit ID"] + " · " + top10["Hit Description"].str[:35]
    fig1 = px.bar(
        top10, x="Identity (%)", y="Label", orientation="h",
        color="Identity (%)",
        color_continuous_scale=GRAD_SEQ,
        title="🏆 Top 10 Hits by Percentage Identity",
        height=420,
    )
    fig1.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    fig1.update_xaxes(range=[0, 105])
    st.plotly_chart(apply_theme(fig1), use_container_width=True)

    # 2. E-Value vs Score scatter
    sdf = df.copy()
    sdf["E_safe"] = sdf["E-Value"].apply(lambda e: max(e, 1e-200))
    fig2 = px.scatter(
        sdf, x="Score", y="E_safe", color="Identity (%)",
        size="Alignment Length",
        hover_data=["Hit ID", "Hit Description", "Identity (%)"],
        color_continuous_scale=GRAD_PURP,
        log_y=True,
        title="🔬 E-Value vs Score  (bubble = alignment length)",
        labels={"E_safe": "E-Value (log scale)"},
        height=420,
    )
    st.plotly_chart(apply_theme(fig2), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        hdf = sdf[sdf["E-Value"] > 0]
        fig3 = px.histogram(
            hdf, x="E-Value", nbins=40, log_x=True,
            color_discrete_sequence=["#38bdf8"],
            title="📊 E-Value Distribution", height=340,
        )
        fig3.update_layout(bargap=0.05)
        st.plotly_chart(apply_theme(fig3), use_container_width=True)
    with col2:
        fig4 = px.box(
            df, x="Query ID", y="Bit Score", color="Query ID",
            title="📦 Bit Score per Query", height=340,
        )
        fig4.update_layout(showlegend=False)
        fig4.update_xaxes(tickangle=25)
        st.plotly_chart(apply_theme(fig4), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# LLM EXPLAINER  (Groq)
# ══════════════════════════════════════════════════════════════════════════════

def get_groq_client(api_key: str):
    """Return a Groq client, or None on failure."""
    if not GROQ_AVAILABLE:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


def explain_blast_results(df: pd.DataFrame, api_key: str,
                           user_question: str = "") -> str:
    """
    Pass top-5 BLAST hits as structured context to Groq LLM and get
    a plain-English expert explanation back.
    """
    client = get_groq_client(api_key)
    if client is None:
        return "❌ AI client unavailable. Check secrets.toml."

    top5 = df.head(5)[
        ["Hit ID", "Hit Description", "Identity (%)", "E-Value",
         "Bit Score", "Alignment Length"]
    ].to_dict(orient="records")

    context_lines = []
    for i, hit in enumerate(top5, 1):
        context_lines.append(
            f"  Hit {i}: {hit['Hit ID']} — {hit['Hit Description']}\n"
            f"    Identity: {hit['Identity (%)']}%  |  E-Value: {hit['E-Value']:.2e}"
            f"  |  Bit Score: {hit['Bit Score']:.1f}  |  Align Len: {hit['Alignment Length']}"
        )
    context = "\n".join(context_lines)

    question_part = (
        f"\n\nThe user also asks: {user_question}" if user_question.strip() else ""
    )

    system_prompt = textwrap.dedent("""
        You are an expert bioinformatician and molecular biologist.
        When given BLAST search results, you explain them in clear, accessible
        language for biologists at a graduate level.
        Structure your response with:
        1. A one-paragraph plain-English summary of what the results mean overall.
        2. Brief bullet-point interpretation of the top 3 hits (species, significance,
           what the E-value and identity tell us).
        3. Any notable flags: contamination, paralogs, low complexity, etc.
        4. One concise next-step recommendation.
        Keep your total response under 350 words.
    """)

    user_prompt = (
        f"Here are the top BLAST results for a query sequence:\n\n{context}"
        f"{question_part}\n\nPlease explain these results."
    )

    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.35,
            max_tokens=600,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ AI error: {e}"


def render_llm_explainer(df: pd.DataFrame, groq_key: str, section_key: str = ""):
    """Render the LLM explainer widget beneath any results DataFrame."""
    if df.empty:
        return

    st.markdown("---")
    st.markdown("### 🤖 AI Result Explainer")

    user_q = st.text_input(
        "Ask a follow-up question (optional)",
        placeholder="e.g. Could this be contamination? What organism is this?",
        key=f"llm_question_{section_key}",
    )

    if st.button("✨  Explain these BLAST results", key=f"llm_btn_{section_key}"):
        if not groq_key.strip():
            st.error("AI key not found. Check your secrets.toml file.")
        else:
            with st.spinner("🧠 Thinking…"):
                explanation = explain_blast_results(df, groq_key, user_q)

            with st.chat_message("assistant", avatar="🧬"):
                st.markdown(explanation)

            # Store in session so it persists on rerun
            st.session_state[f"llm_result_{section_key}"] = explanation

    # Show previous result if available
    elif f"llm_result_{section_key}" in st.session_state:
        with st.chat_message("assistant", avatar="🧬"):
            st.markdown(st.session_state[f"llm_result_{section_key}"])


# ══════════════════════════════════════════════════════════════════════════════
# CENTRAL DOGMA TOOL
# ══════════════════════════════════════════════════════════════════════════════

CODON_TABLE_DISPLAY = {
    "Standard (1)": 1, "Vertebrate Mitochondrial (2)": 2,
    "Bacterial (11)": 11, "Alternative Yeast (12)": 12,
}

def run_central_dogma(dna_raw: str, table_id: int = 1):
    """
    Given a raw DNA string, return a dict with:
    complement, reverse_complement, mRNA, protein.
    Raises ValueError for invalid sequence.
    """
    # Clean: upper, strip FASTA header, remove whitespace
    clean = re.sub(r">.*\n?", "", dna_raw)
    clean = re.sub(r"\s", "", clean).upper()
    valid = set("ATGCNRYSWKMBDHVN")
    invalid = set(clean) - valid
    if invalid:
        raise ValueError(f"Invalid characters in DNA: {', '.join(sorted(invalid))}")

    seq  = Seq(clean)
    comp = seq.complement()
    rc   = seq.reverse_complement()
    mrna = seq.transcribe()
    # Translate with to_stop=False to show full ORF including stop *
    protein = seq.translate(table=table_id, to_stop=False)

    return {
        "DNA (5'→3')":            str(seq),
        "Complement (3'→5')":     str(comp),
        "Reverse Complement":     str(rc),
        "mRNA (Transcription)":   str(mrna),
        "Protein (Translation)":  str(protein),
    }


# ══════════════════════════════════════════════════════════════════════════════
# GC CONTENT & SEQUENCE PROFILING
# ══════════════════════════════════════════════════════════════════════════════

def profile_sequence(seq_str: str, seq_id: str = "Sequence"):
    """
    Calculate comprehensive sequence statistics.
    Returns dict of metrics + base counts.
    """
    seq_str = seq_str.upper()
    length  = len(seq_str)
    if length == 0:
        return {}

    counts = {b: seq_str.count(b) for b in "ATGCN"}
    gc     = counts["G"] + counts["C"]
    at     = counts["A"] + counts["T"]
    gc_pct = round(gc / length * 100, 2)
    at_pct = round(at / length * 100, 2)

    try:
        mw = molecular_weight(Seq(seq_str), seq_type="DNA")
    except Exception:
        mw = None

    # Simple Tm estimate (Wallace rule for short seqs)
    tm_wallace = 4 * gc + 2 * at if length < 30 else None

    return {
        "id":       seq_id,
        "length":   length,
        "gc_pct":   gc_pct,
        "at_pct":   at_pct,
        "counts":   counts,
        "mw":       mw,
        "tm":       tm_wallace,
        "purine":   round((counts["A"] + counts["G"]) / length * 100, 2),
        "pyrimidine": round((counts["T"] + counts["C"]) / length * 100, 2),
    }


def render_gc_dashboard(profiles: list[dict]):
    """Render GC content dashboard cards + charts for a list of profiles."""
    if not profiles:
        return

    # Metrics row for first sequence (or only one)
    for profile in profiles:
        st.markdown(f"""
        <div class="glass-card">
          <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">
            <span style="font-size:1.3rem;">🧬</span>
            <span style="font-weight:700; color:#38bdf8; font-size:1.05rem;">
              {profile['id']}
            </span>
            <span class="badge badge-green">{profile['length']:,} bp</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GC Content",      f"{profile['gc_pct']}%")
        c2.metric("AT Content",      f"{profile['at_pct']}%")
        c3.metric("Mol. Weight",
                  f"{profile['mw']/1000:.1f} kDa" if profile['mw'] else "N/A")
        c4.metric("Tm (Wallace)",
                  f"{profile['tm']} °C" if profile['tm'] else "N/A (>30 bp)")

    # ── Nucleotide Pie Chart ───────────────────────────────────────────────
    p = profiles[0]
    pie_df = pd.DataFrame({
        "Base":  ["A", "T", "G", "C", "N"],
        "Count": [p["counts"][b] for b in "ATGCN"],
    })
    pie_df = pie_df[pie_df["Count"] > 0]

    fig_pie = px.pie(
        pie_df, names="Base", values="Count",
        color="Base",
        color_discrete_map={
            "A": "#f87171", "T": "#60a5fa",
            "G": "#4ade80", "C": "#facc15", "N": "#94a3b8",
        },
        title="🍕 Nucleotide Base Composition",
        hole=0.45,
        height=380,
    )
    fig_pie.update_traces(
        textinfo="label+percent",
        textfont=dict(family="Space Grotesk", size=13, color="white"),
        marker=dict(line=dict(color="rgba(4,22,44,0.8)", width=2)),
    )

    # ── GC Skew over sliding window ────────────────────────────────────────
    seq_str = "".join(
        b * p["counts"].get(b, 0) for b in "ATGCN"
    )  # reconstruct approximate sequence for skew
    # Better: if we have the actual raw string we use it directly
    # We'll pass it through profile to avoid re-computation
    if "raw" in p:
        raw = p["raw"]
        window = max(50, len(raw) // 50)
        skews, positions = [], []
        for i in range(0, len(raw) - window, window // 2):
            chunk = raw[i : i + window]
            g, c  = chunk.count("G"), chunk.count("C")
            denom = g + c
            skews.append((g - c) / denom if denom > 0 else 0)
            positions.append(i + window // 2)

        fig_skew = px.line(
            x=positions, y=skews,
            title="📈 GC Skew  [(G–C)/(G+C)]  sliding window",
            labels={"x": "Position (bp)", "y": "GC Skew"},
            color_discrete_sequence=["#38bdf8"],
            height=280,
        )
        fig_skew.add_hline(y=0, line_dash="dot",
                           line_color="rgba(255,255,255,0.2)")
        fig_skew.update_traces(line=dict(width=1.5))

        col_a, col_b = st.columns([1, 1.4])
        with col_a:
            st.plotly_chart(apply_theme(fig_pie), use_container_width=True)
        with col_b:
            st.plotly_chart(apply_theme(fig_skew), use_container_width=True)
    else:
        st.plotly_chart(apply_theme(fig_pie), use_container_width=True)

    # Multi-sequence GC bar (batch mode)
    if len(profiles) > 1:
        cmp_df = pd.DataFrame({
            "Sequence": [p["id"] for p in profiles],
            "GC (%)":   [p["gc_pct"] for p in profiles],
            "Length":   [p["length"] for p in profiles],
        })
        fig_bar = px.bar(
            cmp_df, x="Sequence", y="GC (%)",
            color="GC (%)", color_continuous_scale=GRAD_SEQ,
            title="🔬 GC Content Comparison", height=320,
        )
        st.plotly_chart(apply_theme(fig_bar), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PHYLOGENETIC TREE VIEWER
# ══════════════════════════════════════════════════════════════════════════════

def render_phylo_tree(tree_bytes: bytes, fmt: str = "newick"):
    """Parse and draw a phylogenetic tree using Biopython + Matplotlib."""
    handle = io.StringIO(tree_bytes.decode("utf-8", errors="replace"))
    try:
        tree = Phylo.read(handle, fmt)
    except Exception as e:
        st.error(f"Could not parse tree file: {e}")
        return

    # Count terminals to set figure height
    n_terminals = len(tree.get_terminals())
    fig_h = max(5, n_terminals * 0.35)

    fig, ax = plt.subplots(figsize=(12, fig_h))
    fig.patch.set_facecolor("#020b14")
    ax.set_facecolor("#041626")

    # Draw using Biopython's matplotlib integration
    Phylo.draw(
        tree,
        axes=ax,
        do_show=False,
        show_confidence=True,
    )

    # Style overrides
    ax.tick_params(colors="#7dd3fc")
    ax.xaxis.label.set_color("#7dd3fc")
    ax.yaxis.label.set_color("#7dd3fc")
    ax.title.set_color("#38bdf8")
    for spine in ax.spines.values():
        spine.set_edgecolor("rgba(56,189,248,0.2)")
    for text in ax.texts:
        text.set_color("#e0f2fe")
        text.set_fontfamily("Space Grotesk")
        text.set_fontsize(9)
    for line in ax.get_lines():
        line.set_color("#38bdf8")
        line.set_alpha(0.7)
        line.set_linewidth(1.2)
    ax.set_title("Phylogenetic Tree", color="#38bdf8",
                 fontfamily="Space Grotesk", fontsize=14, pad=12)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Stats
    depths  = [tree.distance(t) for t in tree.get_terminals()]
    col1, col2, col3 = st.columns(3)
    col1.metric("Terminal Taxa",  n_terminals)
    col2.metric("Internal Nodes", len(tree.get_nonterminals()))
    col3.metric("Max Branch Depth", f"{max(depths):.4f}" if depths else "N/A")


# ══════════════════════════════════════════════════════════════════════════════
# PRIMER DESIGN ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════

def design_primer(sequence: str, primer_len: int = 20) -> dict:
    """
    Design a simple primer pair from the first and last `primer_len` bp of
    a target sequence. Returns dict with both primers' properties.
    """
    seq_clean = re.sub(r"\s", "", sequence.upper())
    seq_clean = re.sub(r">.*", "", seq_clean)
    seq_clean = re.sub(r"[^ATGCN]", "", seq_clean)

    if len(seq_clean) < primer_len * 2 + 20:
        raise ValueError(
            f"Sequence too short (< {primer_len*2 + 20} bp) for primer design."
        )

    fwd_seq = seq_clean[:primer_len]
    rev_seq = str(Seq(seq_clean[-primer_len:]).reverse_complement())
    product_size = len(seq_clean)

    def primer_stats(seq_str: str, name: str) -> dict:
        s   = Seq(seq_str)
        gc  = gc_fraction(s) * 100
        try:
            tm_nn = Tm_NN(seq_str, nn_table=DNA_NN4)
        except Exception:
            tm_nn = None
        try:
            tm_gc = Tm_GC(seq_str)
        except Exception:
            tm_gc = None

        # Basic hairpin check: find longest self-complement run ≥ 4
        rc_str    = str(s.reverse_complement())
        hairpin   = False
        for i in range(len(seq_str) - 3):
            for j in range(len(rc_str) - 3):
                window = 4
                if seq_str[i:i+window] == rc_str[j:j+window]:
                    hairpin = True
                    break

        # Self-dimer: any 4-mer repeat
        kmers     = [seq_str[i:i+4] for i in range(len(seq_str)-3)]
        self_dim  = len(kmers) != len(set(kmers))

        return {
            "name":       name,
            "sequence":   seq_str,
            "length":     len(seq_str),
            "gc_pct":     round(gc, 1),
            "tm_nn":      round(tm_nn, 1) if tm_nn else None,
            "tm_gc":      round(tm_gc, 1) if tm_gc else None,
            "hairpin":    hairpin,
            "self_dimer": self_dim,
        }

    fwd = primer_stats(fwd_seq, "Forward Primer")
    rev = primer_stats(rev_seq, "Reverse Primer")
    return {"forward": fwd, "reverse": rev, "product_size": product_size}


def render_primer_results(result: dict):
    """Render primer design results as a glass-styled summary."""
    for primer in [result["forward"], result["reverse"]]:
        badge_color = "badge-green" if not primer["hairpin"] and not primer["self_dimer"] \
                      else "badge-pink"
        issues = []
        if primer["hairpin"]:    issues.append("⚠️ Potential hairpin")
        if primer["self_dimer"]: issues.append("⚠️ Possible self-dimer")
        issues_html = " &nbsp;·&nbsp; ".join(issues) if issues else \
                      '<span style="color:#4ade80;">✅ No issues detected</span>'

        st.markdown(f"""
        <div class="glass-card">
          <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px;">
            <span style="font-weight:700; color:#38bdf8; font-size:1rem;">
              {primer['name']}
            </span>
            <span class="badge {badge_color}">
              {'OK' if not issues else 'Check Required'}
            </span>
          </div>
          <div class="seq-block" style="margin-bottom:14px; font-size:0.95rem; letter-spacing:0.1em;">
            {primer['sequence']}
          </div>
          <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:10px;
                      font-size:0.82rem; color:#7dd3fc; margin-bottom:10px;">
            <div><span style="color:#4a8aaa;">Length</span><br>
                 <b style="color:#e0f2fe;">{primer['length']} bp</b></div>
            <div><span style="color:#4a8aaa;">GC Content</span><br>
                 <b style="color:#e0f2fe;">{primer['gc_pct']}%</b></div>
            <div><span style="color:#4a8aaa;">Tm (NN)</span><br>
                 <b style="color:#e0f2fe;">
                   {f"{primer['tm_nn']} °C" if primer['tm_nn'] else "N/A"}
                 </b></div>
            <div><span style="color:#4a8aaa;">Tm (GC)</span><br>
                 <b style="color:#e0f2fe;">
                   {f"{primer['tm_gc']} °C" if primer['tm_gc'] else "N/A"}
                 </b></div>
          </div>
          <div style="font-size:0.8rem;">{issues_html}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="glass-card-sm" style="text-align:center;">
      <span style="color:#4a8aaa; font-size:0.85rem;">Expected PCR Product Size</span><br>
      <span style="font-family:var(--font-mono); font-size:1.5rem;
                  color:#38bdf8; font-weight:600;">{result['product_size']:,} bp</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:18px 0 20px;">
      <div style="font-size:2.5rem; margin-bottom:6px;">🧬</div>
      <div style="font-family:'Playfair Display',serif; font-size:1.3rem;
                  font-weight:700; background:linear-gradient(135deg,#38bdf8,#a78bfa);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                  background-clip:text;">
        BLAST BioSuite Pro
      </div>
      <div style="font-size:0.68rem; color:#4a8aaa; margin-top:3px;
                  font-family:'Space Grotesk',sans-serif;">
        Biopython · AI · Bioinformatics Suite
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        options=[
            "🏠  Home",
            "── BLAST ──",
            "📄  XML Parser",
            "🌐  Online BLAST",
            "📦  Batch Processor",
            "── Analysis ──",
            "🤖  AI Explainer",
            "🔀  Central Dogma",
            "📊  GC Dashboard",
            "🌿  Phylo Viewer",
            "🔬  Primer Designer",
        ],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style="font-size:0.72rem; color:#2a5a70; padding:10px 0; line-height:1.7;">
      <b style="color:#3a7a90;">Databases</b><br>
      <code>nt</code> · <code>nr</code> · <code>swissprot</code><br><br>
      <b style="color:#3a7a90;">Programs</b><br>
      <code>blastn</code> · <code>blastp</code> · <code>blastx</code>
    </div>
    """, unsafe_allow_html=True)


# Resolve page key (strip separators)
_page = page.strip().lstrip("─ ").strip()
# Read key from .streamlit/secrets.toml → [GROQ_API_KEY]
try:
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════

if _page == "🏠  Home":
    st.title("BLAST BioSuite Pro")
    st.markdown(
        "A premium bioinformatics workbench. Parse, run, and visualize BLAST "
        "results — with AI-powered explanations and built-in molecular tools."
    )

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tools Available", "8")
    c2.metric("BLAST Programs",  "5")
    c3.metric("LLM Model",       "LLaMA 3 70B")
    c4.metric("Chart Types",     "8+")

    st.markdown("---")

    CARDS = [
        ("📄", "XML Parser",      "badge-blue",
         "Upload BLAST XML (outfmt 5) → instant tidy table + CSV export."),
        ("🌐", "Online BLAST",    "badge-blue",
         "Live qblast against NCBI nt / nr with cached results."),
        ("📦", "Batch Processor", "badge-blue",
         "Multi-file FASTA → sequential NCBI BLAST → combined ZIP download."),
        ("🤖", "AI Explainer",    "badge-purple",
         "LLaMA 3 explains top hits in plain English with next-step advice."),
        ("🔀", "Central Dogma",   "badge-green",
         "DNA → Complement → mRNA → Protein with colored sequence display."),
        ("📊", "GC Dashboard",    "badge-green",
         "GC%, molecular weight, nucleotide pie, GC skew profile."),
        ("🌿", "Phylo Viewer",    "badge-pink",
         "Upload Newick / NEXUS tree → rendered phylogram."),
        ("🔬", "Primer Designer", "badge-pink",
         "Auto-design forward/reverse primers with Tm, GC, hairpin checks."),
    ]

    rows = [CARDS[:4], CARDS[4:]]
    for row in rows:
        cols = st.columns(4)
        for col, (icon, title, badge, desc) in zip(cols, row):
            col.markdown(f"""
            <div class="glass-card" style="height:100%;">
              <div style="font-size:1.8rem; margin-bottom:8px;">{icon}</div>
              <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                <span style="font-weight:700; color:#e0f2fe; font-size:0.95rem;">{title}</span>
                <span class="badge {badge}" style="font-size:0.65rem;">new</span>
              </div>
              <div style="font-size:0.82rem; color:#4a8aaa; line-height:1.55;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("")

    st.info("🤖 AI Explainer is ready — key loaded from secrets.toml.")


# ══════════════════════════════════════════════════════════════════════════════
# SEPARATOR PAGE
# ══════════════════════════════════════════════════════════════════════════════

elif _page in ("── BLAST ──", "── Analysis ──"):
    st.info("👈 Select a tool from the sidebar.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: XML PARSER
# ══════════════════════════════════════════════════════════════════════════════

elif _page == "📄  XML Parser":
    check_biopython()

    st.title("BLAST XML Parser")
    st.markdown(
        "Upload a BLAST XML file (generated with `-outfmt 5`). "
        "Every HSP is extracted into a tidy, filterable table."
    )

    uploaded_xml = st.file_uploader(
        "Drop BLAST XML file here",
        type=["xml"],
    )

    if uploaded_xml:
        with st.spinner("🔍 Parsing with Biopython NCBIXML…"):
            xml_bytes = uploaded_xml.read()
            df = parse_blast_xml(xml_bytes)

        if df.empty:
            st.error("No hits found. Verify this is a valid BLAST XML (outfmt 5) file.")
        else:
            st.success(
                f"✅ Parsed **{len(df):,}** HSPs across "
                f"**{df['Query ID'].nunique()}** queries."
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total HSPs",     f"{len(df):,}")
            c2.metric("Unique Queries", df["Query ID"].nunique())
            c3.metric("Unique Hits",    df["Hit ID"].nunique())
            c4.metric("Best E-Value",   f"{df['E-Value'].min():.2e}")

            st.markdown("---")
            tab_table, tab_charts, tab_ai = st.tabs(
                ["📋 Data Table", "📊 Visualizations", "🤖 AI Explainer"]
            )

            with tab_table:
                with st.expander("🔧 Filter Options", expanded=False):
                    max_ev = st.slider(
                        "Max E-Value", 0.0, 1.0, 1.0, 0.001, format="%.3f"
                    )
                    min_id = st.slider("Min Identity %", 0.0, 100.0, 0.0, 1.0)
                    show_cols = st.multiselect(
                        "Columns", df.columns.tolist(), default=df.columns.tolist()
                    )

                fdf = df[(df["E-Value"] <= max_ev) & (df["Identity (%)"] >= min_id)
                         ][show_cols]
                st.dataframe(fdf, use_container_width=True, height=420)
                st.download_button(
                    "⬇️  Download CSV",
                    data=df_to_csv_bytes(fdf),
                    file_name=f"blast_{Path(uploaded_xml.name).stem}.csv",
                    mime="text/csv",
                )

            with tab_charts:
                build_blast_charts(df)

            with tab_ai:
                render_llm_explainer(df, GROQ_KEY, section_key="xml")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ONLINE BLAST
# ══════════════════════════════════════════════════════════════════════════════

elif _page == "🌐  Online BLAST":
    check_biopython()

    st.title("Online NCBI BLAST Runner")
    st.markdown(
        "Submit a FASTA sequence directly to NCBI via `Bio.Blast.NCBIWWW.qblast`. "
        "Results are cached for 1 hour."
    )
    st.warning("⏳ Typical query time: **30–120 seconds** depending on NCBI load.", icon="⚠️")

    col_p, col_d = st.columns(2)
    with col_p:
        program  = st.selectbox("Program",  ["blastn","blastp","blastx","tblastn","tblastx"])
    with col_d:
        database = st.selectbox("Database", ["nt","nr","swissprot","refseq_rna","refseq_protein"])

    seq_input = st.text_area(
        "Paste FASTA sequence",
        height=180,
        placeholder=">my_seq\nATGCGTACGTAGCTAGCTAGCT…",
    )
    fasta_up = st.file_uploader("…or upload FASTA", type=["fasta","fa","fna","faa","txt"])
    if fasta_up:
        seq_input = fasta_up.read().decode("utf-8")

    if st.button("🚀  Submit to NCBI BLAST", disabled=not seq_input.strip()):
        with st.spinner(f"Running {program.upper()} on NCBI…"):
            try:
                df = run_online_blast(seq_input.strip(), program, database)
            except Exception as e:
                st.error(f"BLAST failed: {e}")
                df = pd.DataFrame()

        if df.empty:
            st.warning("No significant hits returned.")
        else:
            st.success(f"✅ **{len(df):,}** HSPs returned.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Unique Hits",  df["Hit ID"].nunique())
            c2.metric("Best E-Value", f"{df['E-Value'].min():.2e}")
            c3.metric("Max Identity", f"{df['Identity (%)'].max():.1f}%")

            tab_t, tab_c, tab_ai = st.tabs(
                ["📋 Results", "📊 Charts", "🤖 AI Explainer"]
            )
            with tab_t:
                st.dataframe(df, use_container_width=True, height=400)
                st.download_button("⬇️  CSV", df_to_csv_bytes(df),
                                   "online_blast.csv", "text/csv")
            with tab_c:
                build_blast_charts(df)
            with tab_ai:
                render_llm_explainer(df, GROQ_KEY, section_key="online")

    with st.expander("💡 Demo sequence"):
        demo = (">NM_001301717_BRCA1_partial\n"
                "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAA\n"
                "AATCTTAGAGTGTCCCATCTGTCTGGAGTTGATCAAGGAACCTGTCTCCACAAAGTGTG")
        st.code(demo, language="text")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

elif _page == "📦  Batch Processor":
    check_biopython()

    st.title("Batch FASTA Processor")
    st.markdown("Upload multiple FASTA files → BLAST each → merged CSV + ZIP.")
    st.warning("One NCBI query per file. 5 files ≈ 5–10 minutes.", icon="⚠️")

    c1, c2 = st.columns(2)
    with c1:
        bp = st.selectbox("Program",  ["blastn","blastp","blastx"], key="bp")
    with c2:
        bd = st.selectbox("Database", ["nt","nr","swissprot"], key="bd")

    fastas = st.file_uploader("Upload FASTA files", type=["fasta","fa","fna","faa"],
                               accept_multiple_files=True)

    if fastas:
        st.info(f"📁 **{len(fastas)}** file(s) queued.")
        if st.button(f"▶️  Run Batch ({len(fastas)} files)"):
            all_dfs, file_dfs = [], {}
            prog  = st.progress(0, "Starting…")
            status = st.empty()
            for i, f in enumerate(fastas):
                seq = f.read().decode("utf-8", errors="replace")
                status.markdown(f"⏳ Processing **{f.name}** ({i+1}/{len(fastas)})…")
                try:
                    dfi = run_online_blast(seq, bp, bd)
                    dfi.insert(0, "Source File", f.name)
                    file_dfs[f.name] = dfi
                    all_dfs.append(dfi)
                    status.success(f"✅ `{f.name}` → {len(dfi):,} HSPs")
                except Exception as e:
                    status.error(f"❌ `{f.name}`: {e}")
                prog.progress((i+1)/len(fastas), f"Completed {i+1}/{len(fastas)}")

            prog.empty(); status.empty()
            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True)
                st.success(f"🎉 **{len(combined):,}** total HSPs from **{len(all_dfs)}** queries.")

                summary = combined.groupby("Source File").agg(
                    HSPs=("Score","count"),
                    Unique_Hits=("Hit ID","nunique"),
                    Best_EValue=("E-Value","min"),
                    Max_Identity=("Identity (%)","max"),
                ).reset_index().round(4)
                st.dataframe(summary, use_container_width=True)

                dl1, dl2 = st.columns(2)
                with dl1:
                    st.download_button("⬇️  Combined CSV", df_to_csv_bytes(combined),
                                       "blast_batch.csv", "text/csv")
                with dl2:
                    zbuf = io.BytesIO()
                    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for fn, dfi in file_dfs.items():
                            zf.writestr(Path(fn).stem + "_blast.csv", dfi.to_csv(index=False))
                    zbuf.seek(0)
                    st.download_button("📦  Download ZIP", zbuf.getvalue(),
                                       "blast_batch.zip", "application/zip")

                st.markdown("---")
                build_blast_charts(combined)
                render_llm_explainer(combined, GROQ_KEY, section_key="batch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LLM EXPLAINER (standalone)
# ══════════════════════════════════════════════════════════════════════════════

elif _page == "🤖  AI Explainer":
    st.title("AI BLAST Explainer")
    st.markdown("Upload a BLAST XML — the AI reads the top hits and explains them in plain English.")

    if not GROQ_AVAILABLE:
        st.error("Install the AI SDK: `pip install groq`")
        st.stop()

    if not GROQ_KEY.strip():
        st.warning("⚠️ AI key not found. Add `GROQ_API_KEY` to `.streamlit/secrets.toml`.")

    st.markdown("---")

    uploaded = st.file_uploader("Upload BLAST XML", type=["xml"])
    if uploaded:
        check_biopython()
        with st.spinner("Parsing XML…"):
            df = parse_blast_xml(uploaded.read())

        if df.empty:
            st.error("No hits found in the uploaded file.")
        else:
            st.success(f"✅ **{len(df):,}** HSPs loaded.")

            st.markdown("#### Top 5 Hits")
            st.dataframe(
                df.head(5)[["Hit ID","Hit Description","Identity (%)","E-Value","Bit Score"]],
                use_container_width=True,
            )

            render_llm_explainer(df, GROQ_KEY, section_key="standalone")

    # Chat interface
    st.markdown("---")
    st.markdown("### 💬 Ask a Question")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"], avatar="🧬" if msg["role"]=="assistant" else "🧑‍🔬"):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about BLAST, sequence analysis, bioinformatics…"):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍🔬"):
            st.markdown(prompt)

        if not GROQ_KEY.strip():
            with st.chat_message("assistant", avatar="🧬"):
                st.warning("AI key not found. Check secrets.toml.")
        else:
            with st.chat_message("assistant", avatar="🧬"):
                with st.spinner("Thinking…"):
                    client = get_groq_client(GROQ_KEY)
                    if client:
                        messages = [
                            {"role": "system", "content":
                             "You are an expert bioinformatician. Answer clearly and concisely."},
                        ] + st.session_state["chat_history"]
                        try:
                            resp = client.chat.completions.create(
                                model="llama3-70b-8192",
                                messages=messages,
                                max_tokens=500,
                                temperature=0.3,
                            )
                            answer = resp.choices[0].message.content
                        except Exception as e:
                            answer = f"❌ AI error: {e}"
                    else:
                        answer = "❌ AI unavailable."

                    st.markdown(answer)
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": answer}
                    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CENTRAL DOGMA TOOL
# ══════════════════════════════════════════════════════════════════════════════

elif _page == "🔀  Central Dogma":
    check_biopython()

    st.title("Central Dogma Tool")
    st.markdown(
        "Convert **DNA → Complement → mRNA → Protein** instantly. "
        "Each nucleotide is color-coded: "
        '<span class="nuc-A">A</span> '
        '<span class="nuc-T">T</span> '
        '<span class="nuc-G">G</span> '
        '<span class="nuc-C">C</span> '
        '<span class="nuc-U">U</span>',
        unsafe_allow_html=True,
    )

    col_in, col_cfg = st.columns([2, 1])
    with col_in:
        dna_input = st.text_area(
            "Enter DNA sequence (FASTA or raw)",
            height=160,
            placeholder=">my_gene\nATGCGTACGTAGCATGCATCGATCGATCGATCGATCGAT",
        )
    with col_cfg:
        table_name = st.selectbox("Genetic Code Table", list(CODON_TABLE_DISPLAY.keys()))
        table_id   = CODON_TABLE_DISPLAY[table_name]
        include_rc = st.checkbox("Show Reverse Complement", value=True)
        include_rna = st.checkbox("Show mRNA", value=True)
        include_aa  = st.checkbox("Show Protein", value=True)

    if st.button("🔀  Run Translation", disabled=not dna_input.strip()):
        try:
            results = run_central_dogma(dna_input, table_id)

            length = len(results["DNA (5'→3')"])
            st.markdown(f"""
            <div style="display:flex; gap:12px; margin:16px 0; flex-wrap:wrap;">
              <span class="badge badge-blue">Length: {length:,} bp</span>
              <span class="badge badge-green">
                GC: {round(gc_fraction(Seq(results["DNA (5'→3')"]))*100,1)}%
              </span>
              <span class="badge badge-purple">Table: {table_name}</span>
            </div>
            """, unsafe_allow_html=True)

            # DNA always shown
            st.markdown(
                seq_html_block(results["DNA (5'→3')"], "DNA  5'→3'", "badge-blue"),
                unsafe_allow_html=True,
            )
            st.markdown(
                seq_html_block(results["Complement (3'→5')"], "Complement  3'→5'", "badge-blue"),
                unsafe_allow_html=True,
            )
            if include_rc:
                st.markdown(
                    seq_html_block(results["Reverse Complement"], "Reverse Complement", "badge-purple"),
                    unsafe_allow_html=True,
                )
            if include_rna:
                st.markdown(
                    seq_html_block(results["mRNA (Transcription)"], "mRNA (Transcription)", "badge-green"),
                    unsafe_allow_html=True,
                )
            if include_aa:
                protein = results["Protein (Translation)"]
                st.markdown(
                    f'<div style="margin-bottom:14px;">'
                    f'<span class="badge badge-pink">Protein (Translation)</span>'
                    f'<div class="seq-block" style="margin-top:6px; color:#f9a8d4; '
                    f'letter-spacing:0.15em; font-size:0.88rem;">{protein}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Amino acid counts
                aa_df = (
                    pd.Series(list(protein.replace("*","")))
                    .value_counts()
                    .reset_index()
                )
                aa_df.columns = ["Amino Acid", "Count"]
                fig_aa = px.bar(
                    aa_df.head(15), x="Amino Acid", y="Count",
                    color="Count", color_continuous_scale=GRAD_PURP,
                    title="Amino Acid Frequency (top 15)",
                    height=300,
                )
                fig_aa.update_layout(coloraxis_showscale=False)
                st.plotly_chart(apply_theme(fig_aa), use_container_width=True)

        except ValueError as e:
            st.error(f"Sequence error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

    with st.expander("💡 Example — TP53 coding sequence (partial)"):
        st.code(""">TP53_CDS_partial
ATGGAGGAGCCGCAGTCAGATCCTAGCGTTGAATGAGAGGAATTTGAGGGAGCCGTGGG
TGGGAGTATTTGGGAAGGGAGCAGGCTGTGGGCGTGGCAGCAGCCTGTGGTGGCGTGCC
CGGCTCGCAGCACCTGTGTTATGGGGTGGAGGAGTCTGTGCTGTGTTTGGGAAGTTTCC
TGGGTTCATGCCAGCCTTCCACTTCTTCTTCCTTACCAGGGCAGCTACGGTTTCCGTCT
GGGCTTCTTGCATTCTGGG""", language="text")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GC CONTENT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

elif _page == "📊  GC Dashboard":
    check_biopython()

    st.title("GC Content & Sequence Profiler")
    st.markdown(
        "Upload a FASTA file to generate a complete sequence health dashboard: "
        "GC%, molecular weight, nucleotide composition, and GC skew profile."
    )

    uploaded_fasta = st.file_uploader(
        "Upload FASTA file",
        type=["fasta","fa","fna","txt"],
        accept_multiple_files=False,
    )

    if uploaded_fasta:
        raw_text = uploaded_fasta.read().decode("utf-8", errors="replace")
        records  = list(SeqIO.parse(io.StringIO(raw_text), "fasta"))

        if not records:
            # Try treating as raw sequence
            raw_seq = re.sub(r"\s", "", raw_text.upper())
            if re.match(r"^[ATGCN]+$", raw_seq):
                from Bio.SeqRecord import SeqRecord
                records = [SeqRecord(Seq(raw_seq), id="raw_sequence")]
            else:
                st.error("Could not parse FASTA. Check your file format.")
                st.stop()

        st.success(f"✅ Loaded **{len(records)}** sequence(s).")

        profiles = []
        for rec in records[:20]:  # cap at 20 for perf
            p = profile_sequence(str(rec.seq), seq_id=rec.id)
            p["raw"] = str(rec.seq).upper()
            profiles.append(p)

        if len(records) > 20:
            st.info("Showing first 20 sequences for performance.")

        # Sequence selector
        if len(profiles) > 1:
            seq_names = [p["id"] for p in profiles]
            selected  = st.selectbox("Select sequence to profile:", seq_names)
            display_profiles = [p for p in profiles if p["id"] == selected]
        else:
            display_profiles = profiles

        render_gc_dashboard(display_profiles)

        # Table of all sequences
        if len(profiles) > 1:
            st.markdown("### 📋 All Sequences Summary")
            summary_df = pd.DataFrame([{
                "ID":        p["id"],
                "Length":    p["length"],
                "GC (%)":    p["gc_pct"],
                "AT (%)":    p["at_pct"],
                "Purine (%)": p["purine"],
                "MW (kDa)":  round(p["mw"]/1000, 1) if p["mw"] else None,
            } for p in profiles])
            st.dataframe(summary_df, use_container_width=True)
            st.download_button(
                "⬇️  Download Summary CSV",
                data=df_to_csv_bytes(summary_df),
                file_name="gc_profile_summary.csv",
                mime="text/csv",
            )

    else:
        # Quick manual analysis
        st.markdown("---")
        st.markdown("### Quick Sequence Analysis")
        manual_seq = st.text_area(
            "Or paste a raw DNA sequence directly:",
            height=120,
            placeholder="ATGCGATCGATCGATCGTAGCTAGCTAGCTAGC…",
        )
        if manual_seq.strip():
            clean = re.sub(r"\s|>.*", "", manual_seq).upper()
            if clean:
                p = profile_sequence(clean, "manual_input")
                p["raw"] = clean
                render_gc_dashboard([p])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PHYLOGENETIC TREE VIEWER
# ══════════════════════════════════════════════════════════════════════════════

elif _page == "🌿  Phylo Viewer":
    check_biopython()

    st.title("Phylogenetic Tree Viewer")
    st.markdown(
        "Upload a **Newick** (`.nwk`, `.tree`, `.dnd`) or **NEXUS** (`.nex`, `.nexus`) "
        "tree file. The app draws a phylogram using Biopython + Matplotlib."
    )

    col_file, col_fmt = st.columns([2, 1])
    with col_file:
        tree_file = st.file_uploader(
            "Upload tree file",
            type=["nwk","tree","dnd","nex","nexus","txt"],
        )
    with col_fmt:
        tree_fmt = st.selectbox(
            "Format",
            ["newick","nexus","nexml","phyloxml"],
            index=0,
        )

    if tree_file:
        tree_bytes = tree_file.read()
        with st.spinner("Rendering phylogenetic tree…"):
            render_phylo_tree(tree_bytes, fmt=tree_fmt)

    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:40px;">
          <div style="font-size:3rem; margin-bottom:10px;">🌿</div>
          <div style="color:#4a8aaa; font-size:0.9rem;">
            Upload a Newick / NEXUS tree file to render the phylogram
          </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("💡 Generate a Newick tree from BLAST results"):
            st.markdown("""
            After running BLAST, you can create a simple NJ tree from your hits
            using **MEGA**, **ClustalW**, or the NCBI Tree Viewer.
            Example Newick format:
            """)
            st.code(
                "((Homo_sapiens:0.12, Pan_troglodytes:0.08):0.05,"
                " (Mus_musculus:0.25, Rattus_norvegicus:0.22):0.10,"
                " Danio_rerio:0.45);",
                language="text",
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRIMER DESIGNER
# ══════════════════════════════════════════════════════════════════════════════

elif _page == "🔬  Primer Designer":
    check_biopython()

    st.title("PCR Primer Design Assistant")
    st.markdown(
        "Input a target DNA sequence and get automatically designed **forward & "
        "reverse primers** with melting temperature (Tm), GC content, and "
        "hairpin / self-dimer checks."
    )

    col_seq, col_cfg = st.columns([2, 1])
    with col_seq:
        target_seq = st.text_area(
            "Target DNA sequence (FASTA or raw)",
            height=200,
            placeholder=">my_target\nATGCGTACGTAGCATGCATCGATCGATCGATCGATCGATCGTAGCTAGCTAGCTAGCTA…",
        )
        fasta_up = st.file_uploader("…or upload FASTA", type=["fasta","fa","fna","txt"],
                                     key="primer_fasta")
        if fasta_up:
            target_seq = fasta_up.read().decode("utf-8", errors="replace")

    with col_cfg:
        primer_len  = st.slider("Primer length (bp)", 15, 30, 20)
        st.markdown("---")
        st.markdown("""
        <div class="glass-card-sm">
          <div style="font-size:0.78rem; color:#4a8aaa; line-height:1.8;">
            <b style="color:#7dd3fc;">Ideal primer specs</b><br>
            GC content: 40–60%<br>
            Tm: 55–65 °C<br>
            ΔTm (fwd/rev): &lt; 5 °C<br>
            No 3′ hairpins<br>
            No self-complementarity
          </div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔬  Design Primers", disabled=not target_seq.strip()):
        try:
            result = design_primer(target_seq, primer_len)
            render_primer_results(result)

            # Tm comparison chart
            fwd_tm = result["forward"]["tm_nn"] or result["forward"]["tm_gc"] or 0
            rev_tm = result["reverse"]["tm_nn"] or result["reverse"]["tm_gc"] or 0

            fig_tm = go.Figure()
            for primer_info, color in [
                (result["forward"], "#38bdf8"),
                (result["reverse"], "#a78bfa"),
            ]:
                fig_tm.add_trace(go.Bar(
                    name=primer_info["name"],
                    x=["Tm (NN)", "GC (%)"],
                    y=[primer_info["tm_nn"] or 0, primer_info["gc_pct"]],
                    marker_color=color,
                    opacity=0.85,
                ))
            fig_tm.update_layout(
                barmode="group",
                title="Primer Property Comparison",
                height=300,
            )
            st.plotly_chart(apply_theme(fig_tm), use_container_width=True)

            # ΔTm warning
            dtm = abs(fwd_tm - rev_tm)
            if dtm > 5:
                st.warning(
                    f"⚠️ ΔTm between primers is **{dtm:.1f} °C** "
                    "(> 5 °C may reduce PCR efficiency). "
                    "Consider adjusting primer length."
                )
            else:
                st.success(f"✅ ΔTm = {dtm:.1f} °C — primers are well-matched.")

        except ValueError as e:
            st.error(f"Design error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

    with st.expander("💡 Demo — GAPDH fragment"):
        demo_gapdh = """>GAPDH_amplicon_demo
ATGGGGAAGGTGAAGGTCGGAGTCAACGGATTTGGTCGTATTGGGCGCCTGGTCACCAGGGCTGCTTTTAAC
TCTGGTAAAGTGGATATTGTTGCCATCAATGACCCCTTCATTGACCTCAACTACATGGTCTACATGTTCCA
GTATGACTCCACTCACGGCAAATTCAACGGCACAGTCAAGGCTGAGAACGGGAAGCTTGTCATCAATGGAAA
TCCCATCACCATCTTCCAGGAGCGAGATCCCTCCAAAATCAAGTGGGGCGATGCTGGCGCTGAGTACGTCGT
GGAGTCCACTGGCGTCTTCACCACCATGGAGAAGGCTGGGGCTCATTTGCAGGGGGGAGCCAAAAGGGTCAT
CATCTCTGCCCCCTCTGCTGATGCCCCCATGTTCGTCATGGGTGTGAACCATGAGAAGTATGACAACAGCCT
CAAGATCATCAGCAATGCCTCCTGCACCACCAACTGCTTAGCACCCCTGGCCAAGGTCATCCATGACAACTT
TGGCATTGTGGAAGGGCTCATGACCACAGTCCATGCCATCACTGCCACCCAGAAGACTGTGGATGGCCCCTC
CGGGAAACTGTGGCGTGATGGCCGCGGGGCTCTCCAGAACATCATCCCTGCCTCTACTGGCGCTGCCAAGGC
TGTGGGCAAGGTCATCCCTGAGCTGAACGGGAAGCTCACTGGCATGGCCTTCCGTGTTCCTACCCCCAATGT
GTCAGTGGTGGACCTGACCTGCCGTCTAGAAAAACCTGCCAAATATGATGACATCAAGAAGGTGGTGAAGCA"""
        st.code(demo_gapdh[:200] + "…", language="text")
        if st.button("Use GAPDH demo"):
            st.session_state["primer_demo"] = demo_gapdh
            st.rerun()
