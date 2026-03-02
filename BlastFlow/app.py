"""
BLAST BioSuite Pro ─ Professional Bioinformatics Suite
Run:   streamlit run app.py
Requires: pip install streamlit biopython pandas numpy plotly matplotlib groq openpyxl
          GROQ_API_KEY in .streamlit/secrets.toml  (optional – enables AI features)
"""

import io, os, re, zipfile, warnings, textwrap, time, datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit.components.v1 as stc

try:
    from Bio.Blast import NCBIXML, NCBIWWW
    from Bio import SeqIO, Phylo, pairwise2
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

try:
    import openpyxl
    XLSX = True
except ImportError:
    XLSX = False

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BLAST BioSuite Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Session init
for k,v in [("page","blast"),("blast_results",None),("chat_history",[]),
             ("history",[]),("bookmarks",[])]:
    if k not in st.session_state:
        st.session_state[k] = v

def go(p):
    st.session_state.page = p
    st.rerun()

try:    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except: GROQ_KEY = os.environ.get("GROQ_API_KEY","")

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
BG   = "#0b0f19"
BG2  = "#121826"
PA   = "#7c9cff"   # primary accent
SA   = "#00e0ff"   # secondary accent
TM   = "#e6ecff"   # text main
TF   = "#9aa4c7"   # text faded
BD   = "rgba(255,255,255,0.08)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root{{
  --bg:{BG};--bg2:{BG2};--pa:{PA};--sa:{SA};--tm:{TM};--tf:{TF};
  --bd:{BD};--bh:rgba(255,255,255,0.15);
  --glass:rgba(18,24,38,0.75);--glass2:rgba(18,24,38,0.92);
  --gp:rgba(124,156,255,0.15);--gs:rgba(0,224,255,0.12);
  --font:'Inter',sans-serif;--mono:'JetBrains Mono',monospace;
}}

/* ─ Reset & base ─ */
[data-testid="stSidebar"],[data-testid="collapsedControl"]{{display:none!important;}}
.main .block-container{{max-width:1280px;padding:0 2rem 5rem;}}
html,body,[class*="css"],.stApp{{font-family:var(--font)!important;color:var(--tm)!important;}}
.stApp{{background:var(--bg)!important;}}
.stApp::before{{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(ellipse 60% 50% at 8% 10%,rgba(124,156,255,.055) 0%,transparent 65%),
    radial-gradient(ellipse 50% 40% at 92% 80%,rgba(0,224,255,.05) 0%,transparent 65%),
    radial-gradient(ellipse 30% 30% at 50% 55%,rgba(124,156,255,.025) 0%,transparent 65%);}}
.main .block-container{{position:relative;z-index:1;}}

/* ─ Nav bar wrapper ─ */
.nav-wrap{{
  position:sticky;top:0;z-index:999;
  background:rgba(11,15,25,0.94);
  backdrop-filter:blur(24px) saturate(160%);
  border-bottom:1px solid {BD};
  margin:0 -2rem 1.8rem;
  padding:10px 2rem 10px;
  display:flex;align-items:center;gap:16px;
  box-shadow:0 2px 24px rgba(0,0,0,.55);
}}
.nav-logo{{
  font-weight:700;font-size:1rem;letter-spacing:-.2px;
  white-space:nowrap;
  background:linear-gradient(90deg,{PA},{SA});
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
/* Style all nav buttons to look like tabs */
.stHorizontalBlock [data-testid="column"] > div > div > div > div > button{{
  background:transparent!important;
  border:1px solid transparent!important;
  border-radius:8px!important;
  color:{TF}!important;
  font-size:.82rem!important;
  font-weight:500!important;
  padding:6px 10px!important;
  width:100%!important;
  transition:all .15s!important;
  box-shadow:none!important;
  white-space:nowrap!important;
}}
.stHorizontalBlock [data-testid="column"] > div > div > div > div > button:hover{{
  background:rgba(124,156,255,.1)!important;
  border-color:rgba(124,156,255,.2)!important;
  color:{TM}!important;
  transform:none!important;
}}
/* Active nav button - applied via class on parent */
[data-nav-active="true"] button{{
  background:rgba(124,156,255,.15)!important;
  border-color:rgba(124,156,255,.3)!important;
  color:{PA}!important;
  font-weight:600!important;
}}

/* ─ Section header ─ */
.sec-header{{
  display:flex;align-items:center;gap:14px;margin-bottom:24px;padding-top:4px;
}}
.sec-icon{{
  width:46px;height:46px;border-radius:13px;
  display:flex;align-items:center;justify-content:center;font-size:1.4rem;
  box-shadow:0 0 20px var(--ic,rgba(124,156,255,.25));
}}
.sec-title{{font-size:1.65rem;font-weight:700;color:{TM};letter-spacing:-.3px;margin:0;}}
.sec-sub{{font-size:.82rem;color:{TF};margin:2px 0 0;}}

/* ─ Glass card ─ */
.gc{{
  background:var(--glass);backdrop-filter:blur(20px) saturate(140%);
  border:1px solid var(--bd);border-radius:16px;padding:22px 24px;margin-bottom:14px;
  box-shadow:0 4px 24px rgba(0,0,0,.4),inset 0 1px 0 rgba(255,255,255,.04);
  transition:border-color .2s,box-shadow .2s,transform .2s;}}
.gc:hover{{border-color:var(--bh);box-shadow:0 8px 36px rgba(0,0,0,.5),0 0 18px var(--gp);transform:translateY(-1px);}}
.gc-sm{{background:rgba(18,24,38,.6);backdrop-filter:blur(14px);border:1px solid var(--bd);
  border-radius:12px;padding:14px 18px;box-shadow:0 2px 12px rgba(0,0,0,.3);}}
.gc-inset{{background:rgba(11,15,25,.7);border:1px solid var(--bd);border-radius:12px;padding:16px 20px;}}

/* ─ Info/stat cards ─ */
.stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0;}}
.stat{{background:rgba(18,24,38,.8);border:1px solid var(--bd);border-radius:12px;
  padding:16px 18px;text-align:center;}}
.stat-val{{font-family:var(--mono);font-size:1.5rem;font-weight:700;color:{PA};margin-bottom:2px;}}
.stat-lbl{{font-size:.73rem;color:{TF};text-transform:uppercase;letter-spacing:.5px;}}

/* ─ Typography ─ */
h1,h2,h3,h4{{font-family:var(--font)!important;}}
h1{{font-size:1.9rem!important;font-weight:700!important;color:{TM}!important;letter-spacing:-.4px!important;}}
h2{{font-size:1.15rem!important;font-weight:600!important;color:{PA}!important;}}
h3{{font-size:1rem!important;font-weight:600!important;color:{SA}!important;}}
p,li{{color:{TM}!important;line-height:1.65!important;}}
a{{color:{PA}!important;}}
code,pre{{font-family:var(--mono)!important;background:rgba(124,156,255,.1)!important;
  border:1px solid rgba(124,156,255,.2)!important;color:{PA}!important;border-radius:6px;font-size:.82rem!important;}}

/* ─ Streamlit metric ─ */
[data-testid="metric-container"]{{background:rgba(18,24,38,.8)!important;border:1px solid var(--bd)!important;
  border-radius:14px!important;padding:14px 18px!important;box-shadow:0 2px 16px rgba(0,0,0,.3)!important;}}
[data-testid="metric-container"]:hover{{border-color:var(--bh)!important;}}
[data-testid="stMetricValue"]{{font-family:var(--mono)!important;color:{PA}!important;font-size:1.45rem!important;}}
[data-testid="stMetricLabel"]{{color:{TF}!important;font-size:.78rem!important;}}

/* ─ Buttons ─ */
.stButton>button{{
  background:linear-gradient(135deg,rgba(124,156,255,.15),rgba(0,224,255,.1))!important;
  color:{PA}!important;border:1px solid rgba(124,156,255,.3)!important;
  border-radius:10px!important;font-family:var(--font)!important;font-weight:600!important;
  font-size:.87rem!important;padding:.47rem 1.2rem!important;
  backdrop-filter:blur(8px)!important;transition:all .18s!important;}}
.stButton>button:hover{{
  background:linear-gradient(135deg,rgba(124,156,255,.28),rgba(0,224,255,.18))!important;
  border-color:{PA}!important;color:#fff!important;
  box-shadow:0 0 22px rgba(124,156,255,.3)!important;transform:translateY(-1px)!important;}}
[data-testid="stDownloadButton"]>button{{
  background:linear-gradient(135deg,rgba(0,224,255,.12),rgba(124,156,255,.08))!important;
  color:{SA}!important;border-color:rgba(0,224,255,.28)!important;}}
[data-testid="stDownloadButton"]>button:hover{{
  background:linear-gradient(135deg,rgba(0,224,255,.25),rgba(124,156,255,.15))!important;
  border-color:{SA}!important;color:#fff!important;
  box-shadow:0 0 22px rgba(0,224,255,.28)!important;}}

/* ─ Inputs ─ */
.stTextArea textarea,.stTextInput input{{
  background:rgba(18,24,38,.88)!important;border:1px solid var(--bd)!important;
  border-radius:10px!important;color:{TM}!important;font-family:var(--font)!important;
  transition:border-color .18s,box-shadow .18s!important;}}
.stTextArea textarea:focus,.stTextInput input:focus{{
  border-color:rgba(124,156,255,.5)!important;box-shadow:0 0 0 3px rgba(124,156,255,.1)!important;}}
.stSelectbox>div>div,.stMultiSelect>div>div{{
  background:rgba(18,24,38,.88)!important;border:1px solid var(--bd)!important;
  border-radius:10px!important;color:{TM}!important;}}
.stCheckbox label,[data-testid="stCheckbox"] label{{color:{TM}!important;}}
.stRadio label{{color:{TM}!important;}}

/* ─ File uploader ─ */
[data-testid="stFileUploader"]{{background:rgba(18,24,38,.5)!important;
  border:2px dashed rgba(124,156,255,.2)!important;border-radius:14px!important;}}
[data-testid="stFileUploader"]:hover{{background:rgba(124,156,255,.05)!important;
  border-color:rgba(124,156,255,.4)!important;}}

/* ─ Tabs ─ */
.stTabs [data-baseweb="tab-list"]{{background:rgba(18,24,38,.7)!important;
  border-radius:12px 12px 0 0!important;border-bottom:1px solid var(--bd)!important;
  gap:2px!important;padding:4px 4px 0!important;backdrop-filter:blur(12px)!important;}}
.stTabs [data-baseweb="tab"]{{background:transparent!important;border-radius:8px 8px 0 0!important;
  color:{TF}!important;font-family:var(--font)!important;font-weight:500!important;font-size:.83rem!important;}}
.stTabs [data-baseweb="tab"]:hover{{color:{PA}!important;}}
.stTabs [aria-selected="true"]{{background:rgba(124,156,255,.12)!important;color:{PA}!important;font-weight:600!important;}}
.stTabs [data-baseweb="tab-panel"]{{background:rgba(18,24,38,.55)!important;backdrop-filter:blur(12px)!important;
  border:1px solid var(--bd)!important;border-top:none!important;
  border-radius:0 0 12px 12px!important;padding:20px!important;}}

/* ─ DataFrame ─ */
[data-testid="stDataFrame"]{{background:rgba(18,24,38,.7)!important;border:1px solid var(--bd)!important;
  border-radius:12px!important;overflow:hidden!important;box-shadow:0 2px 16px rgba(0,0,0,.4)!important;}}

/* ─ Misc ─ */
.stAlert{{border-radius:12px!important;border-left-width:3px!important;}}
[data-testid="stExpander"]{{background:rgba(18,24,38,.55)!important;border:1px solid var(--bd)!important;
  border-radius:12px!important;overflow:hidden;}}
.stProgress>div>div{{background:linear-gradient(90deg,{PA},{SA})!important;border-radius:99px!important;}}
.stProgress{{background:rgba(124,156,255,.1)!important;border-radius:99px!important;}}
hr{{border:none!important;border-top:1px solid var(--bd)!important;margin:1.2rem 0!important;}}
[data-testid="stChatMessage"]{{background:var(--glass)!important;border:1px solid var(--bd)!important;
  border-radius:14px!important;margin-bottom:8px!important;}}
.stSlider [data-testid="stTickBarMin"],.stSlider [data-testid="stTickBarMax"]{{color:{TF}!important;}}

/* ─ Sequence display ─ */
.seq-block{{font-family:var(--mono);font-size:.81rem;background:rgba(11,15,25,.9);
  border:1px solid var(--bd);border-radius:10px;padding:14px 18px;
  line-height:1.9;overflow-x:auto;letter-spacing:.04em;word-break:break-all;}}
.nuc-A{{color:#ff6b6b;font-weight:600;}} .nuc-T{{color:{PA};font-weight:600;}}
.nuc-G{{color:{SA};font-weight:600;}}   .nuc-C{{color:#ffd166;font-weight:600;}}
.nuc-U{{color:#ff9f43;font-weight:600;}}

/* ─ Badges ─ */
.badge{{display:inline-block;padding:2px 10px;border-radius:99px;
  font-size:.67rem;font-weight:600;letter-spacing:.4px;text-transform:uppercase;}}
.bv{{background:rgba(124,156,255,.12);color:{PA};border:1px solid rgba(124,156,255,.25);}}
.bs{{background:rgba(0,224,255,.1);color:{SA};border:1px solid rgba(0,224,255,.22);}}
.bt{{background:rgba(103,232,249,.1);color:#67e8f9;border:1px solid rgba(103,232,249,.22);}}
.ba{{background:rgba(255,209,102,.1);color:#ffd166;border:1px solid rgba(255,209,102,.22);}}
.bp{{background:rgba(255,107,107,.1);color:#ff6b6b;border:1px solid rgba(255,107,107,.22);}}
.bg{{background:rgba(52,211,153,.1);color:#34d399;border:1px solid rgba(52,211,153,.22);}}
.bo{{background:rgba(255,159,67,.1);color:#ff9f43;border:1px solid rgba(255,159,67,.22);}}
.bb{{background:rgba(147,197,253,.1);color:#93c5fd;border:1px solid rgba(147,197,253,.22);}}
.br{{background:rgba(252,165,165,.1);color:#fca5a5;border:1px solid rgba(252,165,165,.22);}}

/* ─ Feature pill tags ─ */
.pill{{display:inline-flex;align-items:center;gap:5px;padding:4px 12px;border-radius:99px;
  font-size:.75rem;font-weight:500;background:rgba(124,156,255,.08);
  border:1px solid rgba(124,156,255,.18);color:{TF};margin:2px;}}

/* ─ Divider with label ─ */
.divider{{display:flex;align-items:center;gap:12px;margin:20px 0;}}
.divider::before,.divider::after{{content:'';flex:1;height:1px;background:var(--bd);}}
.divider span{{font-size:.75rem;color:{TF};font-weight:500;text-transform:uppercase;letter-spacing:.5px;}}

/* ─ Scrollbar ─ */
::-webkit-scrollbar{{width:5px;height:5px;}}
::-webkit-scrollbar-track{{background:transparent;}}
::-webkit-scrollbar-thumb{{background:rgba(124,156,255,.2);border-radius:99px;}}
::-webkit-scrollbar-thumb:hover{{background:rgba(124,156,255,.4);}}

/* ══ CRT SCANLINES ══ */
.crt-overlay{{
  position:fixed;inset:0;pointer-events:none;z-index:9998;
  background:repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.06) 2px,
    rgba(0,0,0,0.06) 4px
  );
  animation:scanroll 8s linear infinite;
}}
@keyframes scanroll{{
  0%{{background-position:0 0;}}
  100%{{background-position:0 400px;}}
}}

/* ══ VIGNETTE ══ */
.vignette-overlay{{
  position:fixed;inset:0;pointer-events:none;z-index:9997;
  background:radial-gradient(
    ellipse 85% 80% at 50% 50%,
    transparent 55%,
    rgba(0,0,0,0.55) 100%
  );
}}

/* ══ PARTICLE CANVAS ══ */
#particle-canvas{{
  position:fixed;inset:0;z-index:0;pointer-events:none;
}}

/* ══ LIGHTNING ══ */
#lightning-canvas{{
  position:fixed;inset:0;z-index:1;pointer-events:none;opacity:.65;
}}

/* ══ GLITCH TEXT ══ */
.glitch{{
  position:relative;
  display:inline-block;
  animation:glitch-base 5s infinite;
}}
.glitch::before,.glitch::after{{
  content:attr(data-text);
  position:absolute;inset:0;
}}
.glitch::before{{
  color:#00e0ff;
  clip-path:polygon(0 30%,100% 30%,100% 50%,0 50%);
  animation:glitch-top 5s infinite;
  left:2px;
}}
.glitch::after{{
  color:#ff6b6b;
  clip-path:polygon(0 55%,100% 55%,100% 75%,0 75%);
  animation:glitch-bot 5s infinite;
  left:-2px;
}}
@keyframes glitch-base{{
  0%,90%,100%{{transform:none;opacity:1;}}
  91%{{transform:skewX(-1deg);}}
  93%{{transform:skewX(1deg) skewY(.5deg);}}
  95%{{transform:none;}}
  97%{{transform:skewX(2deg);opacity:.9;}}
}}
@keyframes glitch-top{{
  0%,88%,100%{{transform:none;opacity:0;}}
  89%{{transform:translate(-3px,0);opacity:.8;}}
  91%{{transform:translate(3px,0);opacity:.6;}}
  93%{{transform:none;opacity:0;}}
}}
@keyframes glitch-bot{{
  0%,88%,100%{{transform:none;opacity:0;}}
  90%{{transform:translate(3px,0);opacity:.7;}}
  92%{{transform:translate(-3px,0);opacity:.5;}}
  94%{{transform:none;opacity:0;}}
}}

/* ══ NAV — animated underline + status dot ══ */
.nav-outer{{
  position:sticky;top:0;z-index:9995;
  background:rgba(11,15,25,0.94);
  backdrop-filter:blur(28px) saturate(180%);
  border-bottom:1px solid rgba(124,156,255,.12);
  margin:0 -2rem 1.6rem;
  box-shadow:0 1px 0 rgba(124,156,255,.08),0 4px 30px rgba(0,0,0,.6);
}}
.nav-inner{{
  max-width:1280px;margin:0 auto;
  padding:0 2rem;
  display:flex;align-items:center;gap:0;
}}
.nav-brand{{
  display:flex;align-items:center;gap:9px;
  padding:14px 24px 14px 0;
  border-right:1px solid rgba(255,255,255,.07);
  margin-right:12px;flex-shrink:0;
}}
.nav-brand-text{{
  font-weight:700;font-size:.95rem;letter-spacing:-.2px;
  background:linear-gradient(90deg,{PA} 0%,{SA} 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
.status-dot{{
  width:7px;height:7px;border-radius:50%;
  background:#34d399;
  box-shadow:0 0 6px #34d399,0 0 12px rgba(52,211,153,.4);
  animation:pulse-dot 2.4s ease-in-out infinite;flex-shrink:0;
}}
@keyframes pulse-dot{{
  0%,100%{{transform:scale(1);opacity:1;}}
  50%{{transform:scale(1.35);opacity:.7;}}
}}
.nav-links{{display:flex;align-items:stretch;flex:1;}}
.nav-link{{
  position:relative;
  display:flex;align-items:center;gap:5px;
  padding:14px 11px;
  font-size:.81rem;font-weight:500;color:{TF};
  cursor:pointer;white-space:nowrap;
  border:none;background:none;
  transition:color .15s;
  text-decoration:none;
}}
.nav-link::after{{
  content:'';
  position:absolute;bottom:0;left:50%;right:50%;
  height:2px;
  background:linear-gradient(90deg,{PA},{SA});
  border-radius:2px 2px 0 0;
  transition:left .22s ease,right .22s ease;
  box-shadow:0 0 8px {PA};
}}
.nav-link:hover{{color:{TM};}}
.nav-link:hover::after{{left:8%;right:8%;}}
.nav-link.active{{color:{PA};font-weight:700;}}
.nav-link.active::after{{left:0;right:0;}}

/* ══ FOOTER ══ */
.site-footer{{
  margin-top:60px;padding:28px 0 16px;
  border-top:1px solid rgba(255,255,255,.06);
  text-align:center;
}}
.site-footer p{{
  font-family:'JetBrains Mono',monospace!important;
  font-size:.72rem!important;color:rgba(124,156,255,.5)!important;
  letter-spacing:.5px!important;margin:3px 0!important;
}}
.site-footer a{{color:rgba(124,156,255,.6)!important;text-decoration:none!important;}}
.site-footer a:hover{{color:{PA}!important;}}

/* ══ DIFFICULTY BADGES ══ */
.diff-beginner{{
  display:inline-block;padding:2px 9px;border-radius:4px;
  font-size:.65rem;font-weight:700;letter-spacing:.6px;text-transform:uppercase;
  background:rgba(52,211,153,.12);color:#34d399;border:1px solid rgba(52,211,153,.3);
}}
.diff-intermediate{{
  display:inline-block;padding:2px 9px;border-radius:4px;
  font-size:.65rem;font-weight:700;letter-spacing:.6px;text-transform:uppercase;
  background:rgba(255,209,102,.12);color:#ffd166;border:1px solid rgba(255,209,102,.3);
}}
.diff-advanced{{
  display:inline-block;padding:2px 9px;border-radius:4px;
  font-size:.65rem;font-weight:700;letter-spacing:.6px;text-transform:uppercase;
  background:rgba(255,107,107,.12);color:#ff6b6b;border:1px solid rgba(255,107,107,.3);
}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
NAV_ITEMS = [
    ("blast",  "🌐", "NCBI BLAST"),
    ("seqana", "🔬", "Seq Analyzer"),
    ("dogma",  "🔀", "Central Dogma"),
    ("gc",     "📊", "GC Dashboard"),
    ("primer", "⚗️", "Primer Design"),
    ("phylo",  "🌿", "Phylo Viewer"),
    ("prot3d", "🧊", "3D Protein"),
    ("ai",     "🤖", "AI Assistant"),
    ("history","📋", "History"),
]

PROGRAMS = {
    "blastn":  ("Nucleotide → Nucleotide", "DNA/RNA query vs nucleotide database"),
    "blastp":  ("Protein → Protein",       "Protein query vs protein database"),
    "blastx":  ("Nucleotide → Protein",    "Translated DNA query vs protein database"),
    "tblastn": ("Protein → Nucleotide",    "Protein query vs translated nucleotide db"),
    "tblastx": ("Nucleotide (6-frame)",    "Translated DNA query vs translated DNA db"),
    "megablast":("MegaBLAST",              "Optimised for highly similar sequences"),
    "dc-megablast":("Discontinuous MegaBLAST","For cross-species comparisons"),
}

DATABASES = {
    "Nucleotide": ["nt","refseq_rna","16S_ribosomal_RNA","ITS_RefSeq_Fungi",
                   "env_nt","patnt","vector","mito","human_genomic","mouse_genomic"],
    "Protein":    ["nr","swissprot","refseq_protein","refseq_select_prot",
                   "pdb","env_nr","pat","tsa_nr"],
}

MATRICES  = ["BLOSUM62","BLOSUM45","BLOSUM80","PAM30","PAM70","PAM250"]
WORD_SIZES = {"blastn":[7,11,15,20,28],"blastp":[2,3,5,6],"blastx":[2,3,5,6],
              "tblastn":[2,3,5,6],"tblastx":[2,3],"megablast":[16,20,28,32,64],
              "dc-megablast":[11,12]}

TABLES   = {"Standard (1)":1,"Vertebrate Mitochondrial (2)":2,"Bacterial/Archaea (11)":11,
            "Ciliate Nuclear (6)":6,"Echinoderm Mito (9)":9,"Euplotid Nuclear (10)":10}

ACCENT = {
    "violet": (PA, "rgba(124,156,255,.12)", "rgba(124,156,255,.2)", "rgba(124,156,255,.4)"),
    "sky":    (SA, "rgba(0,224,255,.1)",    "rgba(0,224,255,.18)",  "rgba(0,224,255,.38)"),
    "teal":   ("#67e8f9","rgba(103,232,249,.1)","rgba(103,232,249,.18)","rgba(103,232,249,.35)"),
    "amber":  ("#ffd166","rgba(255,209,102,.1)","rgba(255,209,102,.18)","rgba(255,209,102,.35)"),
    "pink":   ("#ff6b6b","rgba(255,107,107,.1)","rgba(255,107,107,.18)","rgba(255,107,107,.35)"),
    "green":  ("#34d399","rgba(52,211,153,.1)","rgba(52,211,153,.18)","rgba(52,211,153,.35)"),
    "orange": ("#ff9f43","rgba(255,159,67,.1)","rgba(255,159,67,.18)","rgba(255,159,67,.35)"),
    "blue":   ("#93c5fd","rgba(147,197,253,.1)","rgba(147,197,253,.18)","rgba(147,197,253,.35)"),
    "rose":   ("#fca5a5","rgba(252,165,165,.1)","rgba(252,165,165,.18)","rgba(252,165,165,.35)"),
}

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY DARK THEME
# ══════════════════════════════════════════════════════════════════════════════
PLY = dict(
    paper_bgcolor="rgba(11,15,25,0)", plot_bgcolor="rgba(18,24,38,.45)",
    font=dict(family="Inter,sans-serif", color=TM, size=12),
    title_font=dict(family="Inter,sans-serif", color=PA, size=13, weight="bold"),
    legend=dict(bgcolor="rgba(18,24,38,.85)", bordercolor=BD, borderwidth=1),
    coloraxis_colorbar=dict(bgcolor="rgba(18,24,38,.85)", bordercolor=BD,
        tickfont=dict(color=TF), title_font=dict(color=TF)),
    hoverlabel=dict(bgcolor="rgba(11,15,25,.95)", bordercolor=BD, font_color=TM),
    margin=dict(t=40,b=20,l=10,r=10),
)
GRID = dict(gridcolor="rgba(255,255,255,.045)", zerolinecolor="rgba(255,255,255,.08)")
GV = ["#1a1f3a","#2d3a7a","#5472d4",PA,SA]
GM = [PA,SA,"#34d399","#ffd166","#ff6b6b","#ff9f43"]

def th(fig, height=None):
    kw = dict(**PLY)
    if height: kw["height"] = height
    fig.update_layout(**kw); fig.update_xaxes(**GRID); fig.update_yaxes(**GRID)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def need_bio():
    if not BIO: st.error("Run `pip install biopython`"); st.stop()

def csv_bytes(df): return df.to_csv(index=False).encode()

def excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w:
        df.to_excel(w,index=False,sheet_name="BLAST")
    return buf.getvalue()

def colorize(seq):
    m={"A":"nuc-A","T":"nuc-T","G":"nuc-G","C":"nuc-C","U":"nuc-U"}
    return "".join(f'<span class="{m[c]}">{c}</span>' if c in m else c for c in seq.upper())

def seq_block(seq,label="",badge="bv"):
    pre = f'<span class="badge {badge}">{label}</span>' if label else ""
    return f'<div style="margin-bottom:10px;">{pre}<div class="seq-block" style="margin-top:{6 if label else 0}px;">{colorize(seq)}</div></div>'

def section_header(icon, title, subtitle="", accent="violet"):
    ct,cbg,cb,_ = ACCENT[accent]
    sub_html = f'<div class="sec-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="sec-header">
      <div class="sec-icon" style="background:{cbg};border:1px solid {cb};--ic:{cb};">{icon}</div>
      <div>
        <div class="sec-title glitch" data-text="{title}">{title}</div>
        {sub_html}
      </div>
    </div>""", unsafe_allow_html=True)

def divider(label=""):
    if label:
        st.markdown(f'<div class="divider"><span>{label}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown("<hr>", unsafe_allow_html=True)

def pill(text):
    return f'<span class="pill">{text}</span>'

def save_history(query, prog, db, n_hits):
    st.session_state.history.append({
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "query": query[:60] + ("…" if len(query)>60 else ""),
        "program": prog, "database": db, "hits": n_hits,
    })

# ══════════════════════════════════════════════════════════════════════════════
# NAV BAR  (rendered every run)
# ══════════════════════════════════════════════════════════════════════════════
def render_fx():
    """Inject particle system + lightning canvas + CRT/vignette overlays via HTML component."""
    stc.html("""
<!DOCTYPE html><html><head><style>
*{margin:0;padding:0;box-sizing:border-box;}
body{background:transparent;overflow:hidden;}
canvas{position:fixed;inset:0;pointer-events:none;}
#ptc{z-index:0;}
#ltc{z-index:1;opacity:.55;}
#crt{z-index:9;pointer-events:none;position:fixed;inset:0;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.055) 2px,rgba(0,0,0,.055) 4px);}
#vig{z-index:8;pointer-events:none;position:fixed;inset:0;
  background:radial-gradient(ellipse 88% 82% at 50% 50%,transparent 50%,rgba(0,0,0,.62) 100%);}
</style></head><body>
<canvas id="ptc"></canvas>
<canvas id="ltc"></canvas>
<div id="crt"></div>
<div id="vig"></div>
<script>
// ── PARTICLES ──────────────────────────────────────────────────────
const pc = document.getElementById('ptc');
const px = pc.getContext('2d');
let W = pc.width  = window.innerWidth;
let H = pc.height = window.innerHeight;
let mx = W/2, my = H/2;

const COLORS = [
  'rgba(124,156,255,', 'rgba(0,224,255,',
  'rgba(180,120,255,', 'rgba(255,107,200,',
  'rgba(100,200,255,',
];

class Particle {
  constructor() { this.reset(true); }
  reset(init) {
    this.x  = Math.random() * W;
    this.y  = init ? Math.random() * H : H + 10;
    this.vx = (Math.random() - .5) * .4;
    this.vy = -(Math.random() * .8 + .3);
    this.r  = Math.random() * 2.2 + .4;
    this.a  = Math.random() * .6 + .15;
    this.da = (Math.random() * .005 + .002) * (Math.random()<.5?1:-1);
    this.col= COLORS[Math.floor(Math.random()*COLORS.length)];
    this.pulse = Math.random() * Math.PI * 2;
  }
  update(parallaxX, parallaxY) {
    this.pulse += .02;
    this.x  += this.vx + parallaxX * .012;
    this.y  += this.vy + parallaxY * .008;
    this.a  += this.da;
    if (this.a > .75 || this.a < .08) this.da *= -1;
    if (this.y < -8 || this.x < -8 || this.x > W+8) this.reset(false);
  }
  draw() {
    const glow = Math.sin(this.pulse) * .3 + .7;
    px.beginPath();
    px.arc(this.x, this.y, this.r * glow, 0, Math.PI*2);
    px.fillStyle = this.col + this.a + ')';
    px.fill();
    // glow halo
    const g = px.createRadialGradient(this.x,this.y,0,this.x,this.y,this.r*4*glow);
    g.addColorStop(0, this.col + (this.a*.4) + ')');
    g.addColorStop(1, this.col + '0)');
    px.beginPath();
    px.arc(this.x,this.y,this.r*4*glow,0,Math.PI*2);
    px.fillStyle = g;
    px.fill();
  }
}

const particles = Array.from({length:120}, ()=>new Particle());
let prlx = 0, prly = 0;
document.addEventListener('mousemove', e => {
  prlx = (e.clientX - W/2) / W;
  prly = (e.clientY - H/2) / H;
});

function animParticles() {
  px.clearRect(0,0,W,H);
  particles.forEach(p => { p.update(prlx,prly); p.draw(); });
  requestAnimationFrame(animParticles);
}
animParticles();

// ── LIGHTNING ──────────────────────────────────────────────────────
const lc = document.getElementById('ltc');
const lx = lc.getContext('2d');
lc.width  = W; lc.height = H;

function bolt(x1,y1,x2,y2,rough,depth,ctx) {
  if (depth <= 0) {
    ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
    return;
  }
  const mx2 = (x1+x2)/2 + (Math.random()-.5)*rough;
  const my2 = (y1+y2)/2 + (Math.random()-.5)*rough;
  bolt(x1,y1,mx2,my2,rough*.6,depth-1,ctx);
  bolt(mx2,my2,x2,y2,rough*.6,depth-1,ctx);
  // random branch
  if (depth > 1 && Math.random() < .28) {
    const bx = mx2 + (Math.random()-.5)*rough*2;
    const by = my2 + rough*(Math.random()*.8+.2);
    bolt(mx2,my2,bx,by,rough*.4,depth-2,ctx);
  }
}

function flashLightning() {
  lx.clearRect(0,0,W,H);
  const strikes = Math.floor(Math.random()*2)+1;
  for (let s=0;s<strikes;s++) {
    const sx = Math.random()*W;
    const ex = sx + (Math.random()-.5)*300;
    // glow pass
    lx.save();
    lx.strokeStyle = 'rgba(180,140,255,.25)';
    lx.lineWidth   = 6;
    lx.shadowBlur  = 30;
    lx.shadowColor = '#7c9cff';
    bolt(sx,0,ex,H*.7+Math.random()*H*.3,220,7,lx);
    // core pass
    lx.strokeStyle = 'rgba(220,200,255,.9)';
    lx.lineWidth   = 1;
    lx.shadowBlur  = 12;
    lx.shadowColor = '#00e0ff';
    bolt(sx,0,ex,H*.7+Math.random()*H*.3,220,7,lx);
    lx.restore();
  }
  // fade out
  let fade = 1;
  const fadeOut = setInterval(()=>{
    fade -= .08;
    lx.globalAlpha = Math.max(0,fade);
    if (fade <= 0) { clearInterval(fadeOut); lx.clearRect(0,0,W,H); lx.globalAlpha=1; }
  }, 30);
}

// trigger lightning every 4-10 seconds
function scheduleLightning() {
  flashLightning();
  setTimeout(scheduleLightning, 4000 + Math.random()*6000);
}
setTimeout(scheduleLightning, 1500);

// ── RESIZE ────────────────────────────────────────────────────────
window.addEventListener('resize', ()=>{
  W = pc.width  = lc.width  = window.innerWidth;
  H = pc.height = lc.height = window.innerHeight;
});
</script></body></html>
""", height=0, scrolling=False)


def render_nav():
    cur = st.session_state.page
    # Build nav link items HTML
    links_html = ""
    for pid, icon, label in NAV_ITEMS:
        active_cls = " active" if pid == cur else ""
        links_html += f'<span class="nav-link{active_cls}" data-pid="{pid}">{icon} {label}</span>'

    st.markdown(f"""
    <div class="nav-outer">
      <div class="nav-inner">
        <div class="nav-brand">
          <div class="status-dot" title="NCBI Connected"></div>
          <span class="nav-brand-text">🧬 BLAST BioSuite Pro</span>
        </div>
        <nav class="nav-links" id="nav-links">{links_html}</nav>
      </div>
    </div>
    <style>
    /* ensure Streamlit block-container sits below nav */
    .main .block-container{{padding-top:.5rem!important;}}
    </style>
    """, unsafe_allow_html=True)

    # Real routing buttons — styled to match nav visually but kept visible
    # They sit right under the nav bar inside a tight row
    st.markdown("""
    <style>
    /* Nav routing row — make buttons visually identical to nav links */
    div.nav-btn-row > div[data-testid="column"] > div > div > div > div > button {
        background:transparent!important;
        border:none!important;
        border-radius:0!important;
        color:transparent!important;
        font-size:0!important;
        padding:0!important;height:2px!important;
        width:100%!important;
        box-shadow:none!important;
        cursor:pointer!important;
        position:absolute!important;
        opacity:0!important;
        pointer-events:all!important;
    }
    div.nav-btn-row{position:relative;height:0;overflow:visible;}
    /* Stretch invisible buttons to cover the visual nav links above */
    div.nav-btn-row > div[data-testid="column"]{
        position:relative;
    }
    div.nav-btn-row > div[data-testid="column"] > div > div > div > div > button{
        position:absolute!important;
        top:-56px!important;
        left:0!important;right:0!important;
        height:56px!important;
        opacity:0!important;
    }
    </style>
    <div class="nav-btn-row">
    """, unsafe_allow_html=True)

    cols = st.columns(len(NAV_ITEMS))
    for col, (pid, icon, label) in zip(cols, NAV_ITEMS):
        with col:
            if st.button(f"{icon} {label}", key=f"nb_{pid}", use_container_width=True):
                go(pid)

    st.markdown("</div>", unsafe_allow_html=True)


def render_footer():
    st.markdown("""
    <div class="site-footer">
      <p>BLAST BioSuite Pro &nbsp;·&nbsp; Powered by NCBI qBLAST &nbsp;·&nbsp; Biopython &nbsp;·&nbsp; Groq LLaMA 3</p>
      <p style="margin-top:6px!important;">
        <a href="https://blast.ncbi.nlm.nih.gov" target="_blank">NCBI BLAST</a>
        &nbsp;·&nbsp;
        <a href="https://biopython.org" target="_blank">Biopython</a>
        &nbsp;·&nbsp;
        <a href="https://www.rcsb.org" target="_blank">RCSB PDB</a>
      </p>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BLAST CORE
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def parse_xml(xml_bytes:bytes) -> pd.DataFrame:
    rows = []
    handle = io.StringIO(xml_bytes.decode("utf-8", errors="replace"))
    for rec in NCBIXML.parse(handle):
        qid  = rec.query.split()[0]
        qlen = rec.query_length
        for aln in rec.alignments:
            for hsp in aln.hsps:
                cov = round((hsp.query_end - hsp.query_start + 1) / qlen * 100, 1)
                rows.append({
                    "Accession":         aln.hit_id,
                    "Description":       aln.hit_def[:100],
                    "Query ID":          qid,
                    "Query Length":      qlen,
                    "Hit Length":        aln.length,
                    "Max Score":         hsp.score,
                    "Bit Score":         round(hsp.bits, 1),
                    "E-Value":           hsp.expect,
                    "Identity (%)":      round(hsp.identities / hsp.align_length * 100, 2),
                    "Query Coverage (%)":cov,
                    "Gaps":              hsp.gaps,
                    "Alignment Length":  hsp.align_length,
                    "Positives":         hsp.positives,
                    "Query Start":       hsp.query_start,
                    "Query End":         hsp.query_end,
                    "Sbjct Start":       hsp.sbjct_start,
                    "Sbjct End":         hsp.sbjct_end,
                    "Query Seq":         hsp.query,
                    "Match Line":        hsp.match,
                    "Sbjct Seq":         hsp.sbjct,
                })
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("E-Value").reset_index(drop=True)

@st.cache_data(show_spinner=False, ttl=3600)
def run_blast_ncbi(seq:str, prog:str, db:str,
                   hitlist:int=50, expect:float=10.0,
                   word_size:int=0, matrix:str="BLOSUM62",
                   filter_low:bool=True, entrez_query:str="") -> pd.DataFrame:
    kwargs = dict(hitlist_size=hitlist, expect=expect, filter="L" if filter_low else "F")
    if word_size>0: kwargs["word_size"] = word_size
    if prog in("blastp","blastx","tblastn","tblastx"): kwargs["matrix_name"] = matrix
    if entrez_query.strip(): kwargs["entrez_query"] = entrez_query.strip()
    h = NCBIWWW.qblast(prog, db, seq, **kwargs)
    return parse_xml(h.read().encode())

# ══════════════════════════════════════════════════════════════════════════════
# BLAST RESULTS – TABLE
# ══════════════════════════════════════════════════════════════════════════════
DISP_COLS = ["Accession","Description","Max Score","Bit Score","E-Value",
             "Identity (%)","Query Coverage (%)","Alignment Length","Gaps"]

def blast_table(df:pd.DataFrame, key:str="tbl"):
    if df.empty: return
    with st.expander("⚙️ Filter & Columns", expanded=False):
        c1,c2,c3,c4 = st.columns(4)
        with c1: ev  = st.slider("Max E-Value",     0.0,  1.0, 1.0, .001, format="%.3f", key=f"ev_{key}")
        with c2: mi  = st.slider("Min Identity %",  0.0,100.0, 0.0,  1.0, key=f"mi_{key}")
        with c3: mc  = st.slider("Min Coverage %",  0.0,100.0, 0.0,  1.0, key=f"mc_{key}")
        with c4: mb  = st.slider("Min Bit Score",   0.0,500.0, 0.0, 10.0, key=f"mb_{key}")
        avail = [c for c in DISP_COLS if c in df.columns]
        show  = st.multiselect("Visible columns", avail, default=avail, key=f"cs_{key}")

    fdf = df[
        (df["E-Value"]           <= ev) &
        (df["Identity (%)"]      >= mi) &
        (df["Query Coverage (%)"]>= mc) &
        (df["Bit Score"]         >= mb)
    ]
    cols = [c for c in (show or DISP_COLS) if c in fdf.columns]
    disp = fdf[cols]
    st.caption(f"**{len(disp):,}** hits shown • Click column header to sort")
    st.dataframe(disp, use_container_width=True, height=440,
        column_config={
            "E-Value":           st.column_config.NumberColumn(format="%.2e"),
            "Identity (%)":      st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
            "Query Coverage (%)":st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
            "Bit Score":         st.column_config.NumberColumn(format="%.1f"),
        })

    d1,d2,d3,d4 = st.columns(4)
    with d1:
        st.download_button("⬇️ CSV", csv_bytes(disp), f"blast_{key}.csv","text/csv",
                           key=f"dcsv_{key}", use_container_width=True)
    with d2:
        if XLSX:
            st.download_button("⬇️ Excel", excel_bytes(disp), f"blast_{key}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"dxl_{key}", use_container_width=True)
    with d3:
        fasta_lines = []
        for _,r in fdf.head(25).iterrows():
            if r.get("Sbjct Seq"):
                fasta_lines += [f">{r['Accession']} {r['Description']}", r["Sbjct Seq"]]
        if fasta_lines:
            st.download_button("⬇️ FASTA (top 25)","\n".join(fasta_lines).encode(),
                               f"hits_{key}.fasta","text/plain",key=f"dfa_{key}",use_container_width=True)
    with d4:
        report_lines = [
            "BLAST Analysis Report", "="*50,
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Total Hits: {len(disp):,}", f"Best E-Value: {fdf['E-Value'].min():.2e}",
            f"Max Identity: {fdf['Identity (%)'].max():.1f}%",
            f"Max Coverage: {fdf['Query Coverage (%)'].max():.1f}%",
            "", "Top 10 Hits:", "-"*40,
        ]
        for i,(_,r) in enumerate(fdf.head(10).iterrows(),1):
            report_lines.append(
                f"{i:2}. {r['Accession']:20} E={r['E-Value']:.2e}  Ident={r['Identity (%)']:.1f}%  {r['Description'][:50]}"
            )
        st.download_button("📄 Report TXT","\n".join(report_lines).encode(),
                           f"report_{key}.txt","text/plain",key=f"drp_{key}",use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# BLAST CHARTS – all visualisations
# ══════════════════════════════════════════════════════════════════════════════
def render_dot_plot(s1,s2,k=6):
    s1,s2 = s1.upper().strip(),s2.upper().strip()
    kmers = {}
    for i in range(len(s1)-k+1):
        mer = s1[i:i+k]
        if mer not in kmers: kmers[mer]=[]
        kmers[mer].append(i)
    xs,ys=[],[]
    for j in range(len(s2)-k+1):
        mer=s2[j:j+k]
        if mer in kmers:
            for x in kmers[mer]: xs.append(x); ys.append(j)
    if not xs: st.warning("No matching k-mers. Try a smaller word size."); return
    fig = px.scatter(x=xs,y=ys,opacity=.35,color_discrete_sequence=[PA],
        labels={"x":f"Seq 1 (len={len(s1)})","y":f"Seq 2 (len={len(s2)})"},
        title=f"Dot Plot — k={k}  ·  {len(xs):,} matches")
    fig.update_traces(marker_size=3)
    st.plotly_chart(th(fig,400), use_container_width=True)

def blast_charts(df:pd.DataFrame):
    if df.empty: return
    t1,t2,t3,t4,t5 = st.tabs(
        ["📊 Identity & Score","🗺 Alignment Map","🔵 Dot Plot","📈 Distributions","🏷 Species Breakdown"])

    with t1:
        top = df.nlargest(min(15,len(df)),"Identity (%)").copy()
        top["Lbl"] = top["Accession"].str[:15] + " · " + top["Description"].str[:35]
        f1 = px.bar(top,x="Identity (%)",y="Lbl",orientation="h",
                    color="Identity (%)",color_continuous_scale=GV,
                    title="Top Hits — Identity %", height=420)
        f1.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        f1.update_xaxes(range=[0,105])
        st.plotly_chart(th(f1), use_container_width=True)

        sdf = df.copy(); sdf["_ev"] = sdf["E-Value"].apply(lambda e:max(e,1e-200))
        f2 = px.scatter(sdf,x="Bit Score",y="_ev",color="Identity (%)",
                        size="Alignment Length",size_max=18,
                        hover_data=["Accession","Description","Identity (%)","E-Value","Query Coverage (%)"],
                        color_continuous_scale=GM,log_y=True,
                        title="E-Value vs Bit Score  (bubble = alignment length)",
                        labels={"_ev":"E-Value (log)"},height=400)
        st.plotly_chart(th(f2), use_container_width=True)

        # Coverage vs Identity scatter
        f3 = px.scatter(df,x="Query Coverage (%)",y="Identity (%)",
                        color="Bit Score",size="Alignment Length",size_max=14,
                        color_continuous_scale=["#1a1f3a",PA,SA],
                        hover_data=["Accession","Description"],
                        title="Query Coverage vs Identity %",height=380)
        st.plotly_chart(th(f3), use_container_width=True)

    with t2:
        adf = df.nlargest(min(20,len(df)),"Identity (%)").reset_index(drop=True)
        qlen = int(adf["Query Length"].iloc[0]) if "Query Length" in adf.columns else 1000
        fig = go.Figure()
        for _,row in adf.iterrows():
            pct = row["Identity (%)"]
            r = int(max(0,min(255, 255-pct*2.5)))
            g = int(max(0,min(255, pct*2.5)))
            b = 180
            fig.add_trace(go.Bar(
                y=[f"{row['Accession'][:18]}"],
                x=[row["Query End"]-row["Query Start"]],
                base=[row["Query Start"]],
                orientation="h",
                marker_color=f"rgba({r},{g},{b},.8)",
                hovertemplate=(
                    f"<b>{row['Description'][:55]}</b><br>"
                    f"Identity: {row['Identity (%)']}%<br>"
                    f"E-Value: {row['E-Value']:.2e}<br>"
                    f"Pos: {row['Query Start']}–{row['Query End']}<extra></extra>"
                ),
                showlegend=False,
            ))
        fig.update_layout(title="Sequence Alignment Map — Query Coverage",
                          xaxis_title="Query Position (bp)",
                          yaxis=dict(autorange="reversed",tickfont_size=10),
                          height=max(300,len(adf)*26+80))
        fig.add_vline(x=qlen,line_dash="dash",line_color="#ff6b6b",opacity=.6,
                      annotation_text="Query end",annotation_position="top right")
        st.plotly_chart(th(fig), use_container_width=True)
        st.caption("Colour: green = high identity, red = low. Width = aligned region on query.")

    with t3:
        dc1,dc2 = st.columns(2)
        with dc1: s1=st.text_area("Sequence 1",height=90,placeholder="ATGCGT…",key="dp1")
        with dc2: s2=st.text_area("Sequence 2",height=90,placeholder="ATGCGT…",key="dp2")
        wk = st.slider("Word size (k-mer)",3,15,6,key="dpw")
        if st.button("Generate Dot Plot",key="dpbtn"):
            if s1.strip() and s2.strip():
                render_dot_plot(re.sub(r"\s|>.*","",s1), re.sub(r"\s|>.*","",s2), wk)
            else: st.warning("Paste both sequences first.")

    with t4:
        c1,c2 = st.columns(2)
        with c1:
            fe = px.histogram(df[df["E-Value"]>0],x="E-Value",nbins=40,log_x=True,
                              color_discrete_sequence=[PA],title="E-Value Distribution")
            fe.update_layout(bargap=.05)
            st.plotly_chart(th(fe,300), use_container_width=True)
        with c2:
            fi = px.histogram(df,x="Identity (%)",nbins=30,
                              color_discrete_sequence=[SA],title="Identity % Distribution")
            fi.update_layout(bargap=.05)
            st.plotly_chart(th(fi,300), use_container_width=True)

        c3,c4 = st.columns(2)
        with c3:
            fb = px.box(df,y="Bit Score",color_discrete_sequence=[PA],
                        title="Bit Score Box Plot")
            st.plotly_chart(th(fb,280), use_container_width=True)
        with c4:
            fc = px.histogram(df,x="Query Coverage (%)",nbins=25,
                              color_discrete_sequence=["#34d399"],
                              title="Query Coverage Distribution")
            fc.update_layout(bargap=.05)
            st.plotly_chart(th(fc,280), use_container_width=True)

    with t5:
        # Extract organism names heuristically from description
        def extract_org(desc):
            m = re.search(r"\[([^\]]+)\]",str(desc))
            return m.group(1) if m else "Unknown"
        sdf = df.copy(); sdf["Organism"] = sdf["Description"].apply(extract_org)
        org_counts = sdf["Organism"].value_counts().head(15).reset_index()
        org_counts.columns = ["Organism","Count"]
        fo = px.bar(org_counts,x="Count",y="Organism",orientation="h",
                    color="Count",color_continuous_scale=GV,
                    title="Top 15 Organisms in Results")
        fo.update_layout(yaxis=dict(autorange="reversed"),coloraxis_showscale=False)
        st.plotly_chart(th(fo,400), use_container_width=True)

        # Best hit per organism
        best = sdf.sort_values("Identity (%)").drop_duplicates("Organism",keep="last")
        best = best[["Organism","Accession","Identity (%)","E-Value","Bit Score"]].head(20)
        st.caption("Best hit per organism (by Identity %)")
        st.dataframe(best, use_container_width=True, height=320,
            column_config={"E-Value":st.column_config.NumberColumn(format="%.2e"),
                           "Identity (%)":st.column_config.ProgressColumn(format="%.1f%%",min_value=0,max_value=100)})

# ══════════════════════════════════════════════════════════════════════════════
# AI WIDGET
# ══════════════════════════════════════════════════════════════════════════════
def get_groq():
    if not GROQ or not GROQ_KEY: return None
    try: return Groq(api_key=GROQ_KEY)
    except: return None

def ai_explain_blast(df:pd.DataFrame, q:str="") -> str:
    c = get_groq()
    if not c: return "❌ AI unavailable — add GROQ_API_KEY to .streamlit/secrets.toml"
    rows = df.head(5)[["Accession","Description","Identity (%)","E-Value","Bit Score"]].to_dict("records")
    ctx  = "\n".join(
        f"  {i}. {r['Accession']} | {r['Description'][:70]}\n"
        f"     Identity {r['Identity (%)']}%  E-Value {r['E-Value']:.2e}  Bit {r['Bit Score']:.1f}"
        for i,r in enumerate(rows,1)
    )
    sys = ("You are an expert bioinformatician. Given BLAST results, provide:\n"
           "1. Concise biological interpretation (2-3 sentences)\n"
           "2. Top 3 hit significance (bullet points)\n"
           "3. Red flags (contamination, paralogs, low coverage)\n"
           "4. Recommended next step\n"
           "Limit 400 words. Use markdown formatting.")
    prompt = f"BLAST top hits:\n{ctx}" + (f"\n\nUser question: {q}" if q.strip() else "") + "\n\nExplain."
    try:
        r = c.chat.completions.create(model="llama3-70b-8192",
            messages=[{"role":"system","content":sys},{"role":"user","content":prompt}],
            temperature=.3, max_tokens=700)
        return r.choices[0].message.content
    except Exception as e: return f"❌ {e}"

def ai_widget(df:pd.DataFrame, key:str):
    if df.empty: return
    divider("AI Analysis")
    q = st.text_input("Ask a follow-up question (optional)",
                      placeholder="e.g. Is this contamination? What do these E-values mean?",
                      key=f"q_{key}")
    if st.button("✨ AI Interpret Results", key=f"ab_{key}"):
        with st.spinner("Analysing with LLaMA 3…"):
            ans = ai_explain_blast(df, q)
        st.session_state[f"ai_{key}"] = ans
    if f"ai_{key}" in st.session_state:
        with st.chat_message("assistant", avatar="🧬"):
            st.markdown(st.session_state[f"ai_{key}"])

# ══════════════════════════════════════════════════════════════════════════════
# 3D PROTEIN VIEWER
# ══════════════════════════════════════════════════════════════════════════════
def protein_3d_viewer(pdb_id:str, height:int=500):
    pid = pdb_id.strip().upper()
    if not pid or len(pid) != 4:
        st.warning("Enter a valid 4-character PDB ID"); return
    html = f"""<!DOCTYPE html><html><head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
<script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
<style>
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{background:#0b0f19;font-family:Inter,sans-serif;}}
  #v{{width:100%;height:{height}px;border-radius:14px;overflow:hidden;
      box-shadow:0 4px 32px rgba(0,0,0,.6),0 0 40px rgba(124,156,255,.06);}}
  .ctrl{{position:absolute;top:10px;right:10px;z-index:10;display:flex;gap:5px;flex-wrap:wrap;}}
  .b{{background:rgba(18,24,38,.9);border:1px solid rgba(124,156,255,.22);border-radius:7px;
      padding:4px 11px;font-size:11px;font-weight:600;color:#7c9cff;cursor:pointer;
      backdrop-filter:blur(10px);transition:all .15s;}}
  .b:hover{{background:rgba(124,156,255,.18);color:#e6ecff;border-color:#7c9cff;}}
  .lbl{{position:absolute;bottom:10px;left:10px;background:rgba(18,24,38,.88);
        backdrop-filter:blur(10px);border-radius:9px;padding:5px 12px;
        font-size:12px;font-weight:600;color:#e6ecff;border:1px solid rgba(255,255,255,.08);}}
  .info{{position:absolute;bottom:10px;right:10px;background:rgba(18,24,38,.88);
         backdrop-filter:blur(10px);border-radius:9px;padding:5px 12px;
         font-size:11px;color:#9aa4c7;border:1px solid rgba(255,255,255,.06);}}
</style></head><body>
<div style="position:relative;">
  <div id="v"></div>
  <div class="ctrl">
    <button class="b" onclick="ss('cartoon')">Cartoon</button>
    <button class="b" onclick="ss('stick')">Stick</button>
    <button class="b" onclick="ss('sphere')">Sphere</button>
    <button class="b" onclick="ss('line')">Line</button>
    <button class="b" onclick="addSurf()">Surface</button>
    <button class="b" onclick="v.spin(true)">⟳ Spin</button>
    <button class="b" onclick="v.spin(false)">■ Stop</button>
    <button class="b" onclick="v.zoomTo()">⊙ Reset</button>
  </div>
  <div class="lbl">PDB: {pid}  ·  drag rotate  ·  scroll zoom  ·  right-drag translate</div>
  <div class="info" id="info">Loading…</div>
</div>
<script>
let v=$3Dmol.createViewer(document.getElementById("v"),{{backgroundColor:"#0b0f19",antialias:true}});
let loaded=false;
$3Dmol.download("pdb:{pid}",v,{{}},function(){{
  v.setStyle({{}},{{cartoon:{{colorscheme:"ssJmol"}}}});
  v.zoomTo();v.render();loaded=true;
  document.getElementById("info").textContent="PDB {pid} loaded";
  setTimeout(()=>document.getElementById("info").style.opacity="0",3000);
}});
function ss(s){{
  if(!loaded)return;
  v.setStyle({{}},{{}});
  if(s==="cartoon")v.setStyle({{}},{{cartoon:{{colorscheme:"ssJmol"}}}});
  else if(s==="stick")v.setStyle({{}},{{stick:{{colorscheme:"rasmol"}}}});
  else if(s==="sphere")v.setStyle({{}},{{sphere:{{colorscheme:"rasmol",radius:.6}}}});
  else if(s==="line")v.setStyle({{}},{{line:{{colorscheme:"ssJmol"}}}});
  v.render();
}}
function addSurf(){{
  if(!loaded)return;
  v.addSurface($3Dmol.SurfaceType.VDW,{{opacity:.5,colorscheme:"whiteCarbon"}});
  v.render();
}}
</script></body></html>"""
    stc.html(html, height=height+10)

# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def detect_seq_type(seq):
    s = seq.upper().replace(" ","")
    if not s: return "Unknown"
    if set(s) <= set("ATGCN"): return "DNA"
    if set(s) <= set("AUGCN"): return "RNA"
    if set(s) <= set("ACDEFGHIKLMNPQRSTVWY*X"): return "Protein"
    return "DNA"

def seq_analysis(raw):
    clean = re.sub(r">.*\n?","",raw); clean = re.sub(r"\s","",clean).upper()
    if not clean: return {}
    st_ = detect_seq_type(clean); n = len(clean)
    counts = {c:clean.count(c) for c in set(clean)}
    try:
        mw = molecular_weight(Seq(clean), seq_type="DNA" if st_ in("DNA","RNA") else "protein")
    except: mw = None
    r = {"seq":clean,"type":st_,"length":n,"counts":counts,"mw":mw,"gc":None,"at":None,"tm":None}
    if st_ == "DNA":
        gc = clean.count("G") + clean.count("C")
        at = clean.count("A") + clean.count("T")
        r["gc"] = round(gc/n*100,2); r["at"] = round(at/n*100,2)
        r["tm"] = round(4*gc+2*at,1) if n < 30 else None
    return r

def render_seq_analysis(info):
    if not info: return
    st_ = info["type"]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Type",st_)
    c2.metric("Length",f"{info['length']:,}" + (" bp" if st_!="Protein" else " aa"))
    c3.metric("GC Content", f"{info['gc']}%" if info.get("gc") is not None else "—")
    c4.metric("Mol. Weight", f"{info['mw']/1000:.2f} kDa" if info.get("mw") else "—")
    if st_=="DNA":
        cc1,cc2 = st.columns(2)
        cc1.metric("AT Content",f"{info['at']}%")
        cc2.metric("Tm (Wallace)",f"{info['tm']} °C" if info.get("tm") else "—")

    divider()
    cnt = info["counts"]
    freq_df = pd.DataFrame({"Base":list(cnt.keys()),"Count":list(cnt.values())}).sort_values("Count",ascending=False)
    CMAP = {"A":"#ff6b6b","T":PA,"G":SA,"C":"#ffd166","N":"#6b7280","U":"#ff9f43"}

    fa,fb = st.columns(2)
    with fa:
        if st_ in("DNA","RNA"):
            f = px.bar(freq_df,x="Base",y="Count",color="Base",
                       color_discrete_map=CMAP,title="Nucleotide Frequency")
        else:
            f = px.bar(freq_df.head(20),x="Base",y="Count",
                       color="Count",color_continuous_scale=GV,title="Amino Acid Frequency")
        f.update_layout(showlegend=False,coloraxis_showscale=False)
        st.plotly_chart(th(f,300), use_container_width=True)
    with fb:
        if st_ in("DNA","RNA") and len(freq_df)<=8:
            fp = px.pie(freq_df,names="Base",values="Count",hole=.45,
                        color="Base",color_discrete_map=CMAP,title="Base Composition")
            fp.update_traces(textinfo="label+percent",
                             marker=dict(line=dict(color=BG,width=2)))
            st.plotly_chart(th(fp,300), use_container_width=True)
        else:
            f2 = px.bar(freq_df.head(20),y="Base",x="Count",orientation="h",
                        color="Count",color_continuous_scale=GV,title="Frequency (horizontal)")
            f2.update_layout(yaxis=dict(autorange="reversed"),coloraxis_showscale=False)
            st.plotly_chart(th(f2,300), use_container_width=True)

    if st_=="DNA" and info["length"]>100:
        raw = info["seq"]; w = max(50,info["length"]//60)
        sk,po=[],[]
        for i in range(0,len(raw)-w,w//2):
            ch=raw[i:i+w]; g,c=ch.count("G"),ch.count("C"); d=g+c
            sk.append((g-c)/d if d else 0); po.append(i+w//2)
        fg = px.area(x=po,y=sk,title="GC Skew  [(G−C)/(G+C)]",
                     labels={"x":"Position (bp)","y":"GC Skew"},
                     color_discrete_sequence=[PA])
        fg.add_hline(y=0,line_dash="dot",line_color=SA,opacity=.4)
        fg.update_traces(line_width=1.6,fillcolor="rgba(124,156,255,.08)")
        st.plotly_chart(th(fg,260), use_container_width=True)

    divider("Colourised Sequence Preview")
    st.markdown(
        f'<div class="seq-block">{colorize(info["seq"][:400])}'
        f'{"<span style=\'color:#9aa4c7\'>  …</span>" if info["length"]>400 else ""}</div>',
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# GC PROFILE (for GC Dashboard)
# ══════════════════════════════════════════════════════════════════════════════
def profile_seq(seq_str, sid="seq"):
    s=seq_str.upper(); n=len(s)
    if not n: return {}
    cnt={b:s.count(b) for b in "ATGCN"}
    gc=cnt["G"]+cnt["C"]; at=cnt["A"]+cnt["T"]
    try: mw=molecular_weight(Seq(s),seq_type="DNA")
    except: mw=None
    return {"id":sid,"length":n,"gc":round(gc/n*100,2),"at":round(at/n*100,2),
            "cnt":cnt,"mw":mw,"tm":4*gc+2*at if n<30 else None,"raw":s}

# ══════════════════════════════════════════════════════════════════════════════
# PRIMER DESIGN
# ══════════════════════════════════════════════════════════════════════════════
def design_primers(raw, length=20):
    s = re.sub(r"\s|>.*","",raw.upper()); s = re.sub(r"[^ATGCN]","",s)
    if len(s) < length*2+20: raise ValueError(f"Need ≥{length*2+20} bp.")
    def stats(seq,name):
        gc_pct = gc_fraction(Seq(seq))*100
        try:    tm_nn = Tm_NN(seq,nn_table=DNA_NN4)
        except: tm_nn = None
        try:    tm_gc = Tm_GC(seq)
        except: tm_gc = None
        rc  = str(Seq(seq).reverse_complement())
        hp  = any(seq[i:i+4]==rc[j:j+4] for i in range(len(seq)-3) for j in range(len(rc)-3))
        km  = [seq[i:i+4] for i in range(len(seq)-3)]
        return {"name":name,"seq":seq,"len":len(seq),"gc":round(gc_pct,1),
                "tm_nn":round(tm_nn,1) if tm_nn else None,
                "tm_gc":round(tm_gc,1) if tm_gc else None,
                "hairpin":hp,"dimer":len(km)!=len(set(km))}
    return {
        "fwd":  stats(s[:length],"Forward"),
        "rev":  stats(str(Seq(s[-length:]).reverse_complement()),"Reverse"),
        "product": len(s),
    }

def primer_card(p):
    ok   = not p["hairpin"] and not p["dimer"]
    flag = "✅ No issues" if ok else " · ".join(
        filter(None,["⚠️ Hairpin" if p["hairpin"] else "",
                     "⚠️ Self-dimer" if p["dimer"] else ""]))
    col  = "#34d399" if ok else "#ff6b6b"
    st.markdown(f"""
    <div class="gc">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
        <b style="color:{TM};">{p['name']} Primer</b>
        <span style="font-size:.75rem;font-weight:600;color:{col};">{flag}</span>
      </div>
      <div class="seq-block" style="margin-bottom:14px;">{colorize(p['seq'])}</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;font-size:.82rem;">
        <div><span style="color:{TF};">Length</span><br><b>{p['len']} bp</b></div>
        <div><span style="color:{TF};">GC%</span><br><b>{p['gc']}%</b></div>
        <div><span style="color:{TF};">Tm (NN)</span><br><b>{"—" if not p['tm_nn'] else f"{p['tm_nn']} °C"}</b></div>
        <div><span style="color:{TF};">Tm (GC)</span><br><b>{"—" if not p['tm_gc'] else f"{p['tm_gc']} °C"}</b></div>
      </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

# ─── NCBI BLAST ───────────────────────────────────────────────────────────────
def page_blast():
    need_bio()
    section_header("🌐","NCBI BLAST","Full NCBI qblast with all programs & databases","violet")

    # ── INPUT ──────────────────────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="gc">', unsafe_allow_html=True)

        # Sequence input
        seq_col, opt_col = st.columns([3,2])
        with seq_col:
            st.markdown("#### 1 · Sequence Input")
            raw_seq = st.text_area(
                "Paste FASTA or raw sequence",
                height=180,
                placeholder=">my_sequence\nATGCGTACGTAGCTAGCTATATATGCGATCGATCGATCGAGCTAGC…",
                key="blast_seq_input",
            )
            up = st.file_uploader("…or upload FASTA file",
                                  type=["fasta","fa","fna","faa","txt"],key="blast_up")
            if up: raw_seq = up.read().decode()

            # Auto-detect + preview
            if raw_seq.strip():
                info = seq_analysis(raw_seq)
                if info:
                    badges = f'<span class="badge bv">{info["type"]}</span> '
                    badges += f'<span class="badge bs">{info["length"]:,} {"bp" if info["type"]!="Protein" else "aa"}</span> '
                    if info.get("gc") is not None:
                        badges += f'<span class="badge bg">GC {info["gc"]}%</span> '
                    st.markdown(badges, unsafe_allow_html=True)

        with opt_col:
            st.markdown("#### 2 · Program & Database")
            prog = st.selectbox("BLAST Program",
                list(PROGRAMS.keys()),
                format_func=lambda k: f"{k}  —  {PROGRAMS[k][0]}",
                key="blast_prog")
            st.caption(PROGRAMS[prog][1])

            db_group = "Protein" if prog in("blastp","tblastn") else "Nucleotide"
            db_choices = DATABASES[db_group]
            db = st.selectbox("Database", db_choices, key="blast_db")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── ADVANCED PARAMETERS ──────────────────────────────────────────────
    with st.expander("⚙️ Advanced Parameters", expanded=False):
        ac1,ac2,ac3,ac4 = st.columns(4)
        with ac1: hitlist = st.slider("Max hits",     5,500,50,5,key="blast_hits")
        with ac2: expect  = st.select_slider("E-value threshold",
                                [1e-100,1e-50,1e-20,1e-10,1e-5,.001,.01,.1,1,10,100],
                                value=10.0,key="blast_ev")
        with ac3:
            ws_opts = WORD_SIZES.get(prog,[11])
            word_sz = st.selectbox("Word size",ws_opts,index=len(ws_opts)//2,key="blast_ws")
        with ac4:
            matrix = "BLOSUM62"
            if prog not in("blastn","megablast","dc-megablast"):
                matrix = st.selectbox("Scoring matrix",MATRICES,key="blast_mat")

        bc1,bc2,bc3 = st.columns(3)
        with bc1: filter_lc = st.checkbox("Filter low-complexity", True, key="blast_fl")
        with bc2: mask_lower = st.checkbox("Mask lowercase",       False,key="blast_ml")
        with bc3: entrez_q  = st.text_input("Entrez query (filter)",
                                placeholder="e.g. Homo sapiens[Organism]",key="blast_eq")

        sc1,sc2 = st.columns(2)
        with sc1: gap_open  = st.slider("Gap open cost",  0,25,11,1,key="blast_go")
        with sc2: gap_ext   = st.slider("Gap extend cost",0,10, 1,1,key="blast_ge")

    # ── QUICK SEQ ANALYSIS ────────────────────────────────────────────────
    if raw_seq.strip():
        with st.expander("🔬 Quick Sequence Profile", expanded=False):
            render_seq_analysis(seq_analysis(raw_seq))

    # ── SUBMIT ────────────────────────────────────────────────────────────
    btn_cols = st.columns([2,1,1])
    with btn_cols[0]:
        submit = st.button("🚀  Run BLAST on NCBI",
                           disabled=not raw_seq.strip(),
                           use_container_width=True,
                           key="blast_submit")
    with btn_cols[1]:
        if st.button("🗑 Clear Results",use_container_width=True,key="blast_clear"):
            st.session_state.blast_results = None; st.rerun()
    with btn_cols[2]:
        st.caption("⏳ Typical: 30–120 sec on NCBI")

    if submit:
        seq_clean = raw_seq.strip()
        with st.spinner(f"Running **{prog}** vs **{db}** on NCBI servers…"):
            try:
                df = run_blast_ncbi(seq_clean, prog, db, hitlist, expect,
                                    word_sz if word_sz else 0, matrix, filter_lc, entrez_q)
                st.session_state.blast_results = {"df":df,"prog":prog,"db":db,"seq":seq_clean[:80]}
                save_history(seq_clean, prog, db, len(df))
            except Exception as e:
                st.error(f"BLAST failed: {e}"); return

    # ── RESULTS ───────────────────────────────────────────────────────────
    res = st.session_state.blast_results
    if res is None: return
    df = res["df"]

    if df.empty:
        st.warning("No significant hits returned. Try increasing E-value threshold or switching database.")
        return

    # Summary bar
    st.markdown(f"""
    <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;
                background:rgba(18,24,38,.7);border:1px solid {BD};border-radius:12px;
                padding:14px 20px;margin:8px 0 16px;">
      <span style="color:{TF};font-size:.8rem;margin-right:4px;">Results for
        <code style="color:{PA};">{res['prog']}</code> vs
        <code style="color:{SA};">{res['db']}</code></span>
      <span class="badge bv">{len(df):,} HSPs</span>
      <span class="badge bs">{df['Accession'].nunique()} unique hits</span>
      <span class="badge bg">Best E: {df['E-Value'].min():.2e}</span>
      <span class="badge ba">Max Identity: {df['Identity (%)'].max():.1f}%</span>
      <span class="badge bt">Max Coverage: {df['Query Coverage (%)'].max():.1f}%</span>
    </div>""", unsafe_allow_html=True)

    # Metric row
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Total HSPs",         f"{len(df):,}")
    m2.metric("Unique Hits",        df["Accession"].nunique())
    m3.metric("Best E-Value",       f"{df['E-Value'].min():.2e}")
    m4.metric("Max Identity",       f"{df['Identity (%)'].max():.1f}%")
    m5.metric("Max Coverage",       f"{df['Query Coverage (%)'].max():.1f}%")

    # Tabs
    tabs_labels = ["📋 Results Table","📊 Visualizations","🤖 AI Analysis"]
    if res["prog"] == "blastp":
        tabs_labels.append("🧊 3D Structure")
    tabs = st.tabs(tabs_labels)

    with tabs[0]: blast_table(df,"blast")
    with tabs[1]: blast_charts(df)
    with tabs[2]: ai_widget(df,"blast")
    if res["prog"] == "blastp" and len(tabs) > 3:
        with tabs[3]:
            st.markdown("#### Interactive 3D Protein Structure")
            st.caption("Enter a PDB ID from the results above to visualise the 3D structure.")
            pdb_id = st.text_input("PDB ID (4 characters)",
                                   placeholder="e.g. 1HHO",
                                   max_chars=4, key="blast_pdb")
            if pdb_id.strip(): protein_3d_viewer(pdb_id.strip())

    # ── XML PARSER ─────────────────────────────────────────────────────────
    divider("Upload Saved XML")
    with st.expander("📄 Parse previously saved BLAST XML (outfmt 5)", expanded=False):
        up_xml = st.file_uploader("Upload BLAST XML", type=["xml"], key="xml_up")
        if up_xml:
            with st.spinner("Parsing…"):
                df_xml = parse_xml(up_xml.read())
            if df_xml.empty:
                st.error("No hits found.")
            else:
                st.success(f"✅ {len(df_xml):,} HSPs loaded from XML")
                t1,t2,t3 = st.tabs(["📋 Table","📊 Charts","🤖 AI"])
                with t1: blast_table(df_xml,"xml")
                with t2: blast_charts(df_xml)
                with t3: ai_widget(df_xml,"xml")


# ─── SEQUENCE ANALYSER ────────────────────────────────────────────────────────
def page_seqana():
    need_bio()
    section_header("🔬","Sequence Analyzer","Length · GC · Frequency · Skew — before you BLAST","rose")
    c1,c2 = st.columns([3,1])
    with c1:
        raw = st.text_area("Paste FASTA or raw sequence",height=180,
                           placeholder=">gene\nATGCGTACGT…", key="sa_input")
        fu  = st.file_uploader("…or upload FASTA",
                               type=["fasta","fa","fna","faa","txt"],key="sa_up")
        if fu: raw = fu.read().decode()
    with c2:
        st.markdown("""
        <div class="gc-sm">
        <b style="font-size:.85rem;color:#e6ecff;">This tool shows</b>
        <ul style="font-size:.77rem;color:#9aa4c7;margin:8px 0 0;padding-left:15px;line-height:1.85;">
          <li>Auto-detects DNA / RNA / Protein</li>
          <li>Length &amp; molecular weight</li>
          <li>GC%, AT%, Melting temp (Tm)</li>
          <li>Nucleotide/AA frequency chart</li>
          <li>Base composition pie</li>
          <li>GC skew sliding window</li>
          <li>Colour-coded sequence preview</li>
        </ul></div>""", unsafe_allow_html=True)

        with st.expander("Multi-FASTA"):
            st.caption("Upload a multi-FASTA to get a summary table for all sequences.")
            mup = st.file_uploader("Multi-FASTA",type=["fasta","fa","fna"],key="sa_multi")
            if mup:
                txt   = mup.read().decode()
                recs  = list(SeqIO.parse(io.StringIO(txt),"fasta"))
                if recs:
                    rows = []
                    for r in recs[:50]:
                        p = seq_analysis(f">{r.id}\n{str(r.seq)}")
                        rows.append({"ID":r.id,"Length":p["length"],
                                     "GC%":p["gc"],"AT%":p["at"],
                                     "MW(kDa)":round(p["mw"]/1000,2) if p.get("mw") else None})
                    mdf = pd.DataFrame(rows)
                    st.dataframe(mdf,use_container_width=True,height=280)
                    st.download_button("⬇️ CSV",csv_bytes(mdf),"multifasta_stats.csv",
                                       "text/csv",use_container_width=True)

    if raw.strip():
        info = seq_analysis(raw)
        if info: render_seq_analysis(info)
        else: st.error("Could not parse sequence.")

    with st.expander("💡 Example: BRCA1 partial"):
        st.code(">BRCA1_exon2_partial\n"
                "ATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAG"
                "AGTGTCCCATCTGTCTGGAGTTGATCAAGGAACCTGTCTCC", language="text")


# ─── CENTRAL DOGMA ────────────────────────────────────────────────────────────
def page_dogma():
    need_bio()
    section_header("🔀","Central Dogma","DNA → Complement → mRNA → Protein","pink")
    c1,c2 = st.columns([3,1])
    with c1:
        dna = st.text_area("DNA sequence (FASTA or raw)",height=150,placeholder=">gene\nATGCGT…")
    with c2:
        tbl    = st.selectbox("Genetic code",list(TABLES.keys()))
        show_rc  = st.checkbox("Reverse Complement",True)
        show_rna = st.checkbox("mRNA",True)
        show_prot= st.checkbox("Protein",True)
        show_aa  = st.checkbox("AA frequency chart",True)

    if st.button("🔀 Translate & Transcribe",disabled=not dna.strip()):
        try:
            clean = re.sub(r">.*\n?","",dna); clean=re.sub(r"\s","",clean).upper()
            bad = set(clean)-set("ATGCNRYSWKMBDHV")
            if bad: raise ValueError(f"Invalid characters: {', '.join(sorted(bad))}")
            s = Seq(clean); table_id = TABLES[tbl]
            res={
                "DNA 5'→3'":           str(s),
                "Complement 3'→5'":    str(s.complement()),
                "Reverse Complement":  str(s.reverse_complement()),
                "mRNA":                str(s.transcribe()),
                "Protein":             str(s.translate(table=table_id)),
            }
            gc_ = round(gc_fraction(s)*100,1)
            st.markdown(
                f'<div style="display:flex;gap:8px;margin:10px 0 16px;flex-wrap:wrap;">'
                f'<span class="badge bv">{len(clean)} bp</span>'
                f'<span class="badge bg">GC {gc_}%</span>'
                f'<span class="badge bs">{tbl}</span></div>',
                unsafe_allow_html=True)
            st.markdown(seq_block(res["DNA 5'→3'"],"DNA 5'→3'","bv"),unsafe_allow_html=True)
            st.markdown(seq_block(res["Complement 3'→5'"],"Complement 3'→5'","bs"),unsafe_allow_html=True)
            if show_rc:   st.markdown(seq_block(res["Reverse Complement"],"Reverse Complement","bt"),unsafe_allow_html=True)
            if show_rna:  st.markdown(seq_block(res["mRNA"],"mRNA","ba"),unsafe_allow_html=True)
            if show_prot:
                prot = res["Protein"]
                stops = prot.count("*")
                st.markdown(
                    f'<span class="badge bp">Protein</span>'
                    f'<span style="font-size:.75rem;color:{TF};margin-left:8px;">{len(prot)-stops} aa · {stops} stop codons</span>'
                    f'<div class="seq-block" style="margin-top:6px;color:#ff6b6b;letter-spacing:.15em">{prot}</div>',
                    unsafe_allow_html=True)
            if show_aa and show_prot:
                adf = pd.Series(list(res["Protein"].replace("*",""))).value_counts().reset_index()
                adf.columns=["AA","Count"]
                f   = px.bar(adf.head(20),x="AA",y="Count",color="Count",
                             color_continuous_scale=GV,title="Amino Acid Composition")
                f.update_layout(coloraxis_showscale=False)
                st.plotly_chart(th(f,280),use_container_width=True)

            # Downloads
            fasta_out = ">DNA\n" + res["DNA 5'\u21923'"] + "\n>mRNA\n" + res["mRNA"] + "\n>protein\n" + res["Protein"]
            st.download_button("⬇️ Download FASTA (DNA + mRNA + Protein)",
                fasta_out.encode(), "central_dogma.fasta","text/plain",use_container_width=False)
        except ValueError as e: st.error(f"Error: {e}")

    with st.expander("💡 Demo — TP53 fragment"):
        st.code(">TP53_partial\nATGGAGGAGCCGCAGTCAGATCCTAGCGTTGAATGAGAGGAATTTGAGGGAGCCGTGGGTGGG",language="text")


# ─── GC DASHBOARD ─────────────────────────────────────────────────────────────
def page_gc():
    need_bio()
    section_header("📊","GC Content Dashboard","Composition · Skew · Multi-sequence comparison","green")
    up = st.file_uploader("Upload FASTA (single or multi)",type=["fasta","fa","fna","txt"])
    manual = st.text_area("…or paste raw DNA",height=90,placeholder="ATGCGATCGATCG…")

    seqs = []
    if up:
        txt  = up.read().decode("utf-8","replace")
        recs = list(SeqIO.parse(io.StringIO(txt),"fasta"))
        if recs: seqs = [(r.id,str(r.seq)) for r in recs[:30]]
        else:
            raw2 = re.sub(r"\s","",txt.upper())
            if re.match(r"^[ATGCN]+$",raw2): seqs=[("uploaded",raw2)]
    elif manual.strip():
        seqs=[("manual",re.sub(r"\s","",manual).upper())]
    if not seqs: return

    profiles = [profile_seq(s,sid) for sid,s in seqs]
    st.success(f"✅ {len(profiles)} sequence(s) loaded")

    sel = st.selectbox("Select sequence",[p["id"] for p in profiles]) if len(profiles)>1 else profiles[0]["id"]
    p   = next(x for x in profiles if x["id"]==sel)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Length",   f"{p['length']:,} bp")
    c2.metric("GC%",      f"{p['gc']}%")
    c3.metric("AT%",      f"{p['at']}%")
    c4.metric("MW",       f"{p['mw']/1000:.1f} kDa" if p['mw'] else "—")
    c5.metric("Tm",       f"{p['tm']} °C" if p['tm'] else "—")

    CMAP={"A":"#ff6b6b","T":PA,"G":SA,"C":"#ffd166","N":"#6b7280"}
    pie=pd.DataFrame({"Base":list("ATGCN"),"Count":[p["cnt"][b] for b in "ATGCN"]})
    pie=pie[pie["Count"]>0]

    ca,cb2 = st.columns(2)
    with ca:
        fp=px.pie(pie,names="Base",values="Count",hole=.45,color="Base",
                  color_discrete_map=CMAP,title="Base Composition")
        fp.update_traces(textinfo="label+percent",marker=dict(line=dict(color=BG,width=2)))
        st.plotly_chart(th(fp,340),use_container_width=True)
    with cb2:
        raw3=p.get("raw",""); w=max(50,len(raw3)//60); sk,po=[],[]
        for i in range(0,len(raw3)-w,w//2):
            ch=raw3[i:i+w]; g,c=ch.count("G"),ch.count("C"); d=g+c
            sk.append((g-c)/d if d else 0); po.append(i+w//2)
        fs=px.area(x=po,y=sk,title="GC Skew",labels={"x":"Position (bp)","y":"(G-C)/(G+C)"},
                   color_discrete_sequence=[PA])
        fs.add_hline(y=0,line_dash="dot",line_color=SA,opacity=.4)
        fs.update_traces(fillcolor="rgba(124,156,255,.08)",line_width=1.6)
        st.plotly_chart(th(fs,340),use_container_width=True)

    if len(profiles)>1:
        divider("All Sequences")
        sdf=pd.DataFrame([{"ID":x["id"],"Len(bp)":x["length"],"GC%":x["gc"],"AT%":x["at"],
                           "MW(kDa)":round(x["mw"]/1000,1) if x["mw"] else None} for x in profiles])
        st.dataframe(sdf,use_container_width=True)
        fb=px.bar(sdf,x="ID",y="GC%",color="GC%",color_continuous_scale=GV,title="GC% Comparison")
        fb.update_layout(coloraxis_showscale=False,xaxis_tickangle=25)
        st.plotly_chart(th(fb,280),use_container_width=True)
        st.download_button("⬇️ Summary CSV",csv_bytes(sdf),"gc_summary.csv","text/csv")


# ─── PRIMER DESIGNER ──────────────────────────────────────────────────────────
def page_primer():
    need_bio()
    section_header("⚗️","PCR Primer Designer","Forward + Reverse · Tm · GC · Hairpin · Dimer","blue")
    c1,c2 = st.columns([3,1])
    with c1:
        seq = st.text_area("Target DNA",height=180,placeholder=">target\nATGCGT…")
        fu  = st.file_uploader("…or upload FASTA",type=["fasta","fa","fna","txt"],key="pfu")
        if fu: seq = fu.read().decode()
    with c2:
        plen   = st.slider("Primer length (bp)",15,30,20)
        st.markdown(f"""
        <div class="gc-sm">
        <div style="font-size:.78rem;color:{TF};line-height:1.85;">
          <b style="color:{TM};">Ideal primer criteria</b><br>
          GC: 40–60%<br>
          Tm: 55–65 °C<br>
          ΔTm &lt; 5 °C between pair<br>
          No 3' hairpins / self-dimers<br>
          No runs of ≥4 same base
        </div></div>""",unsafe_allow_html=True)

    if st.button("⚗️ Design Primers",disabled=not seq.strip(),use_container_width=False):
        try:
            res = design_primers(seq,plen)
            pc1,pc2 = st.columns(2)
            with pc1: primer_card(res["fwd"])
            with pc2: primer_card(res["rev"])

            fwd_tm = res["fwd"]["tm_nn"] or res["fwd"]["tm_gc"] or 0
            rev_tm = res["rev"]["tm_nn"] or res["rev"]["tm_gc"] or 0
            dtm    = abs(fwd_tm-rev_tm)

            st.markdown(
                f'<div class="gc-sm" style="text-align:center;margin:4px 0 12px;">'
                f'<span style="color:{TF};font-size:.8rem;">Expected product</span><br>'
                f'<b style="font-size:1.6rem;color:{PA};">{res["product"]:,} bp</b>'
                f'</div>',unsafe_allow_html=True)

            if dtm > 5: st.warning(f"ΔTm = {dtm:.1f} °C (>5 °C may reduce PCR efficiency)")
            else:       st.success(f"✅ ΔTm = {dtm:.1f} °C — primer pair well matched")

            # Comparison chart
            fc = go.Figure()
            for pp,col in [(res["fwd"],PA),(res["rev"],SA)]:
                fc.add_trace(go.Bar(
                    name=pp["name"],
                    x=["Tm (NN)","GC (%)","Length/2"],
                    y=[pp["tm_nn"] or 0, pp["gc"], pp["len"]/2],
                    marker_color=col,opacity=.85))
            fc.update_layout(barmode="group",title="Primer Property Comparison",height=270)
            st.plotly_chart(th(fc),use_container_width=True)

            # Download primers
            primer_txt = (
                f">Forward_primer\n{res['fwd']['seq']}\n"
                f">Reverse_primer\n{res['rev']['seq']}\n"
                f"\nProduct size: {res['product']} bp\n"
                f"Fwd Tm (NN): {res['fwd']['tm_nn']} °C\n"
                f"Rev Tm (NN): {res['rev']['tm_nn']} °C\n"
            )
            st.download_button("⬇️ Download Primers FASTA",
                               primer_txt.encode(),"primers.fasta","text/plain")

        except ValueError as e: st.error(f"Design error: {e}")

    with st.expander("💡 Demo — GAPDH"):
        st.code(">GAPDH_demo\nATGGGGAAGGTGAAGGTCGGAGTCAACGGATTTGGTCGTATTGGGCGCCTGGTCACCAGGGCTGCTTTTAACTCTGGTAAAGTGGATATTGTTGCCATCAATGACCCCTTCATTGACCTCAACTACATGGTCTACATGTTCCAGTATGACTCCACTCACGGCAAATTC",language="text")


# ─── PHYLO VIEWER ─────────────────────────────────────────────────────────────
def page_phylo():
    need_bio()
    section_header("🌿","Phylogenetic Tree Viewer","Newick · NEXUS · Phyloxml · Radial or Cladogram","orange")
    uc1,uc2 = st.columns([3,1])
    with uc1: up  = st.file_uploader("Upload tree file",type=["nwk","tree","dnd","nex","nexus","txt","xml"])
    with uc2: fmt = st.selectbox("Format",["newick","nexus","nexml","phyloxml"])

    if not up:
        st.markdown(f'<div class="gc" style="text-align:center;padding:50px 20px;">'
                    f'<div style="font-size:2.5rem;filter:drop-shadow(0 0 16px rgba(255,159,67,.3))">🌿</div>'
                    f'<div style="color:{TF};margin-top:10px;">Upload a tree file to render</div>'
                    f'</div>',unsafe_allow_html=True)
        with st.expander("💡 Example Newick"):
            st.code("((Homo_sapiens:0.12,Pan_troglodytes:0.08):0.05,(Mus_musculus:0.25,Rattus_norvegicus:0.22):0.10,(Danio_rerio:0.45,Xenopus_laevis:0.38):0.15);")
        return

    try:
        tree = Phylo.read(io.StringIO(up.read().decode("utf-8","replace")),fmt)
    except Exception as e: st.error(f"Cannot parse tree: {e}"); return

    terms = tree.get_terminals()
    fig,ax = plt.subplots(figsize=(14,max(6,len(terms)*.42)))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG2)
    Phylo.draw(tree,axes=ax,do_show=False)
    ax.tick_params(colors=TF)
    for sp in ax.spines.values(): sp.set_edgecolor(BD)
    for ln in ax.get_lines(): ln.set_color(PA); ln.set_alpha(.85); ln.set_linewidth(1.4)
    for txt in ax.texts: txt.set_color(TM); txt.set_fontsize(9)
    ax.set_title(f"Phylogenetic Tree — {len(terms)} taxa",color=PA,fontsize=13,pad=10)
    ax.xaxis.label.set_color(TF); ax.yaxis.label.set_color(TF)
    plt.tight_layout()
    st.pyplot(fig,use_container_width=True); plt.close(fig)

    depths=[tree.distance(t) for t in terms]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Terminal Taxa",   len(terms))
    c2.metric("Internal Nodes",  len(tree.get_nonterminals()))
    c3.metric("Max Depth",       f"{max(depths):.4f}" if depths else "—")
    c4.metric("Total Branches",  len(list(tree.find_clades())))


# ─── 3D PROTEIN ────────────────────────────────────────────────────────────────
def page_prot3d():
    section_header("🧊","3D Protein Structure","Interactive py3Dmol · RCSB PDB","rose")
    pc1,pc2 = st.columns([1,2])
    with pc1:
        pdb = st.text_input("PDB ID",placeholder="e.g. 1HHO",max_chars=4).strip().upper()
        st.markdown(f"""
        <div class="gc">
          <b style="font-size:.82rem;color:{TM};">Popular PDB IDs</b>
          <table style="width:100%;font-size:.76rem;margin-top:8px;border-collapse:collapse;">
            {''.join(f"<tr><td><code>{i}</code></td><td style='color:{TF};padding-left:8px;'>{n}</td></tr>"
                     for i,n in [("1HHO","Oxy-haemoglobin"),("4HHB","Deoxy-haemoglobin"),
                                  ("6LU7","SARS-CoV-2 Mpro"),("1TUP","p53–DNA complex"),
                                  ("2LYZ","Lysozyme"),("1CRN","Crambin"),
                                  ("3NIR","Insulin"),("4EK3","GFP variant"),
                                  ("1ATP","cAMP kinase"),("1AON","GroEL chaperonin")])}
          </table>
        </div>""",unsafe_allow_html=True)
        style_opt = st.radio("Default style",["Cartoon","Stick","Sphere","Line"],horizontal=True)
    with pc2:
        if pdb and len(pdb)==4:
            protein_3d_viewer(pdb,560)
        else:
            st.markdown(f'<div class="gc" style="text-align:center;padding:80px 20px;">'
                        f'<div style="font-size:3rem;filter:drop-shadow(0 0 20px rgba(252,165,165,.3))">🧊</div>'
                        f'<div style="color:{TF};margin-top:12px;font-size:.9rem;">'
                        f'Enter any 4-character PDB ID on the left<br>'
                        f'<span style="font-size:.78rem;">Source: RCSB Protein Data Bank</span>'
                        f'</div></div>',unsafe_allow_html=True)


# ─── AI ASSISTANT ──────────────────────────────────────────────────────────────
def page_ai():
    section_header("🤖","AI Bioinformatics Assistant","LLaMA 3 70B via Groq · BLAST interpretation · Q&A","amber")

    if not GROQ:
        st.error("Install Groq: `pip install groq`"); return
    if not GROQ_KEY:
        st.warning("⚠️ Add `GROQ_API_KEY` to `.streamlit/secrets.toml` to enable AI features.")
        return

    # Upload XML for AI explanation
    with st.expander("📊 Interpret a BLAST XML file", expanded=False):
        up_xml = st.file_uploader("Upload BLAST XML",type=["xml"],key="ai_xml")
        if up_xml:
            need_bio()
            with st.spinner("Parsing…"): df_ai = parse_xml(up_xml.read())
            if not df_ai.empty:
                st.success(f"✅ {len(df_ai):,} HSPs loaded")
                st.dataframe(df_ai[["Accession","Description","Identity (%)","E-Value","Bit Score"]].head(10),
                             use_container_width=True,height=260)
                ai_widget(df_ai,"ai_upload")

    divider("Chat")
    # Chat interface
    if not st.session_state.chat_history:
        st.session_state.chat_history.append({
            "role":"assistant",
            "content":"Hello! I'm your AI bioinformatics assistant. Ask me anything about BLAST results, sequence analysis, molecular biology, or bioinformatics workflows.",
        })

    for m in st.session_state.chat_history:
        with st.chat_message(m["role"], avatar="🧬" if m["role"]=="assistant" else "👤"):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask about BLAST, sequences, PCR, molecular biology…"):
        st.session_state.chat_history.append({"role":"user","content":prompt})
        with st.chat_message("user",avatar="👤"): st.markdown(prompt)
        client = get_groq()
        if client:
            with st.chat_message("assistant",avatar="🧬"):
                with st.spinner("Thinking…"):
                    try:
                        sys = ("You are an expert bioinformatician and molecular biologist. "
                               "Be precise, cite evidence, use markdown. If asked about BLAST results, "
                               "interpret them biologically. Keep answers focused and ≤400 words unless more is needed.")
                        msgs = [{"role":"system","content":sys}] + st.session_state.chat_history
                        r = client.chat.completions.create(
                            model="llama3-70b-8192",messages=msgs,max_tokens=800,temperature=.3)
                        ans = r.choices[0].message.content
                    except Exception as e: ans = f"❌ {e}"
                    st.markdown(ans)
                    st.session_state.chat_history.append({"role":"assistant","content":ans})

    # Clear chat
    if len(st.session_state.chat_history) > 2:
        if st.button("🗑 Clear conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


# ─── HISTORY ──────────────────────────────────────────────────────────────────
def page_history():
    section_header("📋","Search History","Your recent BLAST searches this session","teal")
    hist = st.session_state.history

    if not hist:
        st.markdown(f'<div class="gc" style="text-align:center;padding:40px 20px;">'
                    f'<div style="font-size:2rem">📋</div>'
                    f'<div style="color:{TF};margin-top:8px;">No searches yet this session.</div>'
                    f'</div>',unsafe_allow_html=True)
        return

    hdf = pd.DataFrame(hist[::-1])  # newest first
    st.dataframe(hdf, use_container_width=True, height=420,
        column_config={
            "hits": st.column_config.NumberColumn("Hits"),
            "time": st.column_config.TextColumn("Timestamp"),
        })
    st.download_button("⬇️ Export History CSV", csv_bytes(hdf),
                       "blast_history.csv","text/csv")
    if st.button("🗑 Clear History"):
        st.session_state.history = []; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════════════════════════════
render_fx()
render_nav()

P = st.session_state.page
if   P == "blast":   page_blast()
elif P == "seqana":  page_seqana()
elif P == "dogma":   page_dogma()
elif P == "gc":      page_gc()
elif P == "primer":  page_primer()
elif P == "phylo":   page_phylo()
elif P == "prot3d":  page_prot3d()
elif P == "ai":      page_ai()
elif P == "history": page_history()
else: go("blast")

render_footer()
