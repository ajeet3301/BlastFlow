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
import streamlit.components.v1 as stc

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

try:
    import openpyxl
    XLSX = True
except ImportError:
    XLSX = False

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BLAST BioSuite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "page" not in st.session_state:
    st.session_state.page = "home"

def go(page_id: str):
    st.session_state.page = page_id
    st.rerun()

try:
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

# ══════════════════════════════════════════════════════════════════════════════
# CSS — Light Glassmorphism
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --glass:  rgba(255,255,255,0.72);
  --glass2: rgba(255,255,255,0.88);
  --border: rgba(150,160,210,0.25);
  --border-h:rgba(120,130,200,0.45);
  --shadow: rgba(100,110,180,0.12);
  --sh2:    rgba(100,110,180,0.22);
  --text:   #1e1f3a;
  --muted:  #7879a0;
  --font:   'Inter', sans-serif;
  --mono:   'JetBrains Mono', monospace;

  --violet:#7c3aed; --vl:#ede9fe; --vt:rgba(124,58,237,0.12); --vh:rgba(124,58,237,0.25);
  --sky:   #0284c7; --sl:#e0f2fe; --st:rgba(2,132,199,0.12);  --sh:rgba(2,132,199,0.25);
  --teal:  #0d9488; --tl:#ccfbf1; --tt:rgba(13,148,136,0.12); --th:rgba(13,148,136,0.25);
  --amber: #d97706; --al:#fef3c7; --at:rgba(217,119,6,0.12);  --ah:rgba(217,119,6,0.25);
  --pink:  #be185d; --pl:#fce7f3; --pt:rgba(190,24,93,0.12);  --ph:rgba(190,24,93,0.25);
  --green: #15803d; --gl:#dcfce7; --gt:rgba(21,128,61,0.12);  --gh:rgba(21,128,61,0.25);
  --orange:#c2410c; --ol:#ffedd5; --ot:rgba(194,65,12,0.12);  --oh:rgba(194,65,12,0.25);
  --blue:  #1d4ed8; --bl:#dbeafe; --bt:rgba(29,78,216,0.12);  --bh:rgba(29,78,216,0.25);
  --rose:  #e11d48; --rl:#ffe4e6; --rt:rgba(225,29,72,0.12);  --rh:rgba(225,29,72,0.25);
}

[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important;}
.main .block-container{max-width:1200px;padding:2rem 2.5rem 4rem;position:relative;z-index:1;}

html,body,[class*="css"],.stApp{font-family:var(--font)!important;color:var(--text)!important;}
.stApp{background:linear-gradient(135deg,#f0f4ff 0%,#f5f0ff 35%,#f0f7ff 65%,#f0fffa 100%)!important;}
.stApp::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:radial-gradient(ellipse 60% 50% at 15% 20%,rgba(124,58,237,.06) 0%,transparent 100%),
             radial-gradient(ellipse 50% 40% at 85% 70%,rgba(2,132,199,.07) 0%,transparent 100%),
             radial-gradient(ellipse 40% 35% at 50% 90%,rgba(13,148,136,.05) 0%,transparent 100%);}

.gc{background:var(--glass);backdrop-filter:blur(18px) saturate(160%);border:1px solid var(--border);
    border-radius:18px;padding:22px 24px;margin-bottom:14px;
    box-shadow:0 2px 16px var(--shadow),inset 0 1px 0 rgba(255,255,255,.8);transition:all .22s ease;}
.gc:hover{background:var(--glass2);border-color:var(--border-h);
    box-shadow:0 6px 28px var(--sh2),inset 0 1px 0 rgba(255,255,255,.9);transform:translateY(-2px);}
.gc-sm{background:var(--glass);backdrop-filter:blur(14px);border:1px solid var(--border);
    border-radius:12px;padding:14px 18px;box-shadow:0 2px 10px var(--shadow);}

h1{font-family:var(--font)!important;font-size:2rem!important;font-weight:700!important;
   color:var(--text)!important;letter-spacing:-.5px!important;}
h2{color:var(--violet)!important;font-weight:600!important;font-size:1.2rem!important;}
h3{color:var(--sky)!important;font-weight:600!important;font-size:1rem!important;}
code,pre{font-family:var(--mono)!important;background:var(--vl)!important;
    border:1px solid var(--vh)!important;color:var(--violet)!important;border-radius:5px;font-size:.82rem!important;}

[data-testid="metric-container"]{background:var(--glass)!important;border:1px solid var(--border)!important;
    border-radius:14px!important;padding:14px 18px!important;box-shadow:0 2px 12px var(--shadow)!important;}
[data-testid="stMetricValue"]{font-family:var(--mono)!important;color:var(--violet)!important;font-size:1.45rem!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:.78rem!important;}

.stButton>button{background:var(--violet)!important;color:#fff!important;border:none!important;
    border-radius:10px!important;font-family:var(--font)!important;font-weight:600!important;
    font-size:.88rem!important;padding:.5rem 1.3rem!important;
    box-shadow:0 2px 12px rgba(124,58,237,.3)!important;transition:all .18s!important;}
.stButton>button:hover{background:#6d28d9!important;box-shadow:0 4px 20px rgba(124,58,237,.45)!important;transform:translateY(-1px)!important;}
[data-testid="stDownloadButton"]>button{background:var(--teal)!important;box-shadow:0 2px 12px rgba(13,148,136,.25)!important;}
[data-testid="stDownloadButton"]>button:hover{background:#0f766e!important;box-shadow:0 4px 20px rgba(13,148,136,.4)!important;}

.stTextArea textarea,.stTextInput input{background:rgba(255,255,255,.85)!important;
    border:1.5px solid var(--border)!important;border-radius:10px!important;
    color:var(--text)!important;font-family:var(--font)!important;
    box-shadow:inset 0 2px 4px rgba(100,110,180,.06)!important;transition:border-color .18s,box-shadow .18s!important;}
.stTextArea textarea:focus,.stTextInput input:focus{border-color:var(--violet)!important;
    box-shadow:0 0 0 3px rgba(124,58,237,.12)!important;}
.stSelectbox>div>div{background:rgba(255,255,255,.85)!important;
    border:1.5px solid var(--border)!important;border-radius:10px!important;color:var(--text)!important;}

[data-testid="stFileUploader"]{background:rgba(255,255,255,.6)!important;
    border:2px dashed rgba(124,58,237,.3)!important;border-radius:14px!important;transition:all .2s;}
[data-testid="stFileUploader"]:hover{background:rgba(255,255,255,.82)!important;border-color:var(--violet)!important;}

.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,.55)!important;
    border-radius:12px 12px 0 0!important;border-bottom:1px solid var(--border)!important;
    gap:2px!important;padding:5px 5px 0!important;backdrop-filter:blur(10px)!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;border-radius:8px 8px 0 0!important;
    color:var(--muted)!important;font-family:var(--font)!important;font-weight:500!important;font-size:.85rem!important;}
.stTabs [data-baseweb="tab"]:hover{color:var(--violet)!important;}
.stTabs [aria-selected="true"]{background:rgba(124,58,237,.1)!important;color:var(--violet)!important;font-weight:600!important;}
.stTabs [data-baseweb="tab-panel"]{background:rgba(255,255,255,.55)!important;
    backdrop-filter:blur(10px)!important;border:1px solid var(--border)!important;
    border-top:none!important;border-radius:0 0 12px 12px!important;padding:20px!important;}

[data-testid="stDataFrame"]{background:rgba(255,255,255,.7)!important;
    border:1px solid var(--border)!important;border-radius:12px!important;
    overflow:hidden!important;box-shadow:0 2px 10px var(--shadow)!important;}
.stAlert{border-radius:12px!important;border-left-width:3px!important;}
[data-testid="stExpander"]{background:rgba(255,255,255,.6)!important;
    border:1px solid var(--border)!important;border-radius:12px!important;overflow:hidden;}
.stProgress>div>div{background:linear-gradient(90deg,var(--violet),var(--sky))!important;border-radius:99px!important;}
.stProgress{background:rgba(124,58,237,.1)!important;border-radius:99px!important;}
hr{border:none!important;border-top:1px solid var(--border)!important;margin:1.2rem 0!important;}
[data-testid="stChatMessage"]{background:rgba(255,255,255,.75)!important;
    border:1px solid var(--border)!important;border-radius:14px!important;
    margin-bottom:8px!important;box-shadow:0 2px 10px var(--shadow)!important;}
.stSlider [data-testid="stTickBarMin"],.stSlider [data-testid="stTickBarMax"]{color:var(--muted)!important;}

.seq-block{font-family:var(--mono);font-size:.82rem;background:rgba(255,255,255,.9);
    border:1px solid var(--border);border-radius:10px;padding:14px 18px;
    line-height:1.9;overflow-x:auto;letter-spacing:.05em;word-break:break-all;
    box-shadow:inset 0 2px 6px rgba(100,110,180,.07);}
.nuc-A{color:#dc2626;font-weight:600;} .nuc-T{color:#2563eb;font-weight:600;}
.nuc-G{color:#16a34a;font-weight:600;} .nuc-C{color:#d97706;font-weight:600;}
.nuc-U{color:#ea580c;font-weight:600;}

.badge{display:inline-block;padding:2px 10px;border-radius:99px;
    font-size:.69rem;font-weight:600;letter-spacing:.3px;text-transform:uppercase;}
.bv{background:var(--vl);color:var(--violet);border:1px solid var(--vh);}
.bs{background:var(--sl);color:var(--sky);border:1px solid var(--sh);}
.bt{background:var(--tl);color:var(--teal);border:1px solid var(--th);}
.ba{background:var(--al);color:var(--amber);border:1px solid var(--ah);}
.bp{background:var(--pl);color:var(--pink);border:1px solid var(--ph);}
.bg{background:var(--gl);color:var(--green);border:1px solid var(--gh);}
.bo{background:var(--ol);color:var(--orange);border:1px solid var(--oh);}
.bb{background:var(--bl);color:var(--blue);border:1px solid var(--bh);}
.br{background:var(--rl);color:var(--rose);border:1px solid var(--rh);}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY LIGHT THEME
# ══════════════════════════════════════════════════════════════════════════════
PLY = dict(
    paper_bgcolor="rgba(255,255,255,0)",plot_bgcolor="rgba(248,249,255,.5)",
    font=dict(family="Inter,sans-serif",color="#1e1f3a",size=12),
    title_font=dict(family="Inter,sans-serif",color="#7c3aed",size=14),
    legend=dict(bgcolor="rgba(255,255,255,.8)",bordercolor="rgba(150,160,210,.25)",borderwidth=1),
    coloraxis_colorbar=dict(bgcolor="rgba(255,255,255,.8)",bordercolor="rgba(150,160,210,.25)",
        tickfont=dict(color="#7879a0"),title_font=dict(color="#7879a0")),
)
GRID = dict(gridcolor="rgba(150,160,210,.12)",zerolinecolor="rgba(150,160,210,.25)")
GV = ["#ede9fe","#c4b5fd","#a78bfa","#7c3aed","#5b21b6"]
GM = ["#7c3aed","#0284c7","#0d9488","#d97706","#be185d","#e11d48"]

def th(fig):
    fig.update_layout(**PLY); fig.update_xaxes(**GRID); fig.update_yaxes(**GRID)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# CORE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def need_bio():
    if not BIO: st.error("Run `pip install biopython`"); st.stop()

def csv_bytes(df): return df.to_csv(index=False).encode()

def excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="BLAST Results")
    return buf.getvalue()

def colorize(seq):
    m={"A":"nuc-A","T":"nuc-T","G":"nuc-G","C":"nuc-C","U":"nuc-U"}
    return "".join(f'<span class="{m[c]}">{c}</span>' if c in m else c for c in seq.upper())

def seq_block(seq,label,badge="bv"):
    return f'<div style="margin-bottom:12px;"><span class="badge {badge}">{label}</span><div class="seq-block" style="margin-top:6px;">{colorize(seq)}</div></div>'

TABLES = {"Standard (1)":1,"Mitochondrial (2)":2,"Bacterial (11)":11}

ACCENT = {
    "violet":("#7c3aed","#ede9fe","rgba(124,58,237,.12)","rgba(124,58,237,.25)"),
    "sky":   ("#0284c7","#e0f2fe","rgba(2,132,199,.12)","rgba(2,132,199,.25)"),
    "teal":  ("#0d9488","#ccfbf1","rgba(13,148,136,.12)","rgba(13,148,136,.25)"),
    "amber": ("#d97706","#fef3c7","rgba(217,119,6,.12)","rgba(217,119,6,.25)"),
    "pink":  ("#be185d","#fce7f3","rgba(190,24,93,.12)","rgba(190,24,93,.25)"),
    "green": ("#15803d","#dcfce7","rgba(21,128,61,.12)","rgba(21,128,61,.25)"),
    "orange":("#c2410c","#ffedd5","rgba(194,65,12,.12)","rgba(194,65,12,.25)"),
    "blue":  ("#1d4ed8","#dbeafe","rgba(29,78,216,.12)","rgba(29,78,216,.25)"),
    "rose":  ("#e11d48","#ffe4e6","rgba(225,29,72,.12)","rgba(225,29,72,.25)"),
}

def page_header(icon,title,accent="violet"):
    ct,cbg,cb,_ = ACCENT[accent]
    if st.button("← Home",key="bk"): go("home")
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;margin:8px 0 20px;">
      <div style="width:48px;height:48px;border-radius:14px;background:{cbg};border:1px solid {cb};
                  display:flex;align-items:center;justify-content:center;font-size:1.5rem;">{icon}</div>
      <h1 style="margin:0;font-size:1.75rem!important;">{title}</h1>
    </div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BLAST PARSE & QUERY
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def parse_xml(xml_bytes:bytes)->pd.DataFrame:
    rows,handle=[],io.StringIO(xml_bytes.decode("utf-8",errors="replace"))
    for rec in NCBIXML.parse(handle):
        qid,qlen=rec.query.split()[0],rec.query_length
        for aln in rec.alignments:
            for hsp in aln.hsps:
                cov=round((hsp.query_end-hsp.query_start+1)/qlen*100,1)
                rows.append({
                    "Accession":aln.hit_id,"Description":aln.hit_def[:90],
                    "Query ID":qid,"Query Length":qlen,
                    "Hit Length":aln.length,"Max Score":hsp.score,"Bit Score":hsp.bits,
                    "E-Value":hsp.expect,
                    "Identity (%)":round(hsp.identities/hsp.align_length*100,2),
                    "Query Coverage (%)":cov,
                    "Alignment Length":hsp.align_length,"Gaps":hsp.gaps,
                    "Query Start":hsp.query_start,"Query End":hsp.query_end,
                    "Sbjct Start":hsp.sbjct_start,"Sbjct End":hsp.sbjct_end,
                    "Query Seq":hsp.query,"Match Line":hsp.match,"Sbjct Seq":hsp.sbjct,
                })
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("E-Value").reset_index(drop=True)

@st.cache_data(show_spinner=False,ttl=3600)
def run_blast(seq,prog,db):
    h=NCBIWWW.qblast(prog,db,seq); return parse_xml(h.read().encode())

# ══════════════════════════════════════════════════════════════════════════════
# SEQ ANALYZER  (pre-BLAST)
# ══════════════════════════════════════════════════════════════════════════════
def detect_seq_type(seq:str)->str:
    seq=seq.upper().replace(" ","")
    dna_chars=set("ATGCN"); rna_chars=set("AUGCN")
    prot_chars=set("ACDEFGHIKLMNPQRSTVWY*X")
    if set(seq)<=dna_chars: return "DNA"
    if set(seq)<=rna_chars: return "RNA"
    if set(seq)<=prot_chars: return "Protein"
    return "DNA"  # fallback

def seq_analysis(raw:str)->dict:
    clean=re.sub(r">.*\n?","",raw); clean=re.sub(r"\s","",clean).upper()
    if not clean: return {}
    stype=detect_seq_type(clean)
    length=len(clean)
    counts={c:clean.count(c) for c in set(clean)}
    try:
        mw=molecular_weight(Seq(clean),seq_type="DNA" if stype in("DNA","RNA") else "protein")
    except: mw=None

    result={"seq":clean,"type":stype,"length":length,"counts":counts,"mw":mw}

    if stype=="DNA":
        result["gc"]=round(gc_fraction(Seq(clean))*100,2)
        result["at"]=round(100-result["gc"],2)
        result["tm"]=round(4*(clean.count("G")+clean.count("C"))+2*(clean.count("A")+clean.count("T")),1) if length<30 else None
    elif stype=="Protein":
        result["gc"]=None
    return result

def render_seq_analysis(info:dict):
    if not info: return
    stype=info["type"]
    ct,cbg,cb,_=ACCENT["violet"] if stype=="DNA" else (ACCENT["teal"] if stype=="Protein" else ACCENT["sky"])

    # Metric row
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Sequence Type",stype)
    c2.metric("Length",f"{info['length']:,}" + (" bp" if stype!="Protein" else " aa"))
    c3.metric("GC Content",f"{info['gc']}%" if info.get('gc') is not None else "—")
    c4.metric("Mol. Weight",f"{info['mw']/1000:.2f} kDa" if info["mw"] else "—")

    if stype=="DNA":
        c5,c6=st.columns(2)
        with c5: st.metric("AT Content",f"{info['at']}%")
        with c6: st.metric("Tm (Wallace)",f"{info['tm']} °C" if info['tm'] else ">30 bp formula N/A")

    st.markdown("---")
    # ── Frequency charts ──────────────────────────────────────────────────
    cnt=info["counts"]
    freq_df=pd.DataFrame({"Base/AA":list(cnt.keys()),"Count":list(cnt.values())})
    freq_df=freq_df.sort_values("Count",ascending=False)

    COLOR_MAP_DNA={"A":"#ef4444","T":"#3b82f6","G":"#16a34a","C":"#f59e0b","N":"#94a3b8","U":"#ea580c"}

    ca,cb2=st.columns(2)
    with ca:
        if stype in("DNA","RNA"):
            f=px.bar(freq_df,x="Base/AA",y="Count",color="Base/AA",
                     color_discrete_map=COLOR_MAP_DNA,
                     title="Nucleotide Frequency",height=320)
        else:
            f=px.bar(freq_df.head(20),x="Base/AA",y="Count",
                     color="Count",color_continuous_scale=GV,
                     title="Amino Acid Frequency (Top 20)",height=320)
        f.update_layout(showlegend=False,coloraxis_showscale=False)
        st.plotly_chart(th(f),use_container_width=True)

    with cb2:
        if stype in("DNA","RNA") and len(freq_df)<=8:
            fp=px.pie(freq_df,names="Base/AA",values="Count",hole=0.42,
                      color="Base/AA",color_discrete_map=COLOR_MAP_DNA,
                      title="Base Composition",height=320)
            fp.update_traces(textinfo="label+percent",
                             marker=dict(line=dict(color="white",width=2)))
            st.plotly_chart(th(fp),use_container_width=True)
        else:
            f2=px.bar(freq_df.head(20),y="Base/AA",x="Count",orientation="h",
                      color="Count",color_continuous_scale=GV,
                      title="Frequency (horizontal)",height=320)
            f2.update_layout(yaxis=dict(autorange="reversed"),coloraxis_showscale=False)
            st.plotly_chart(th(f2),use_container_width=True)

    # GC skew for DNA
    if stype=="DNA" and info["length"]>100:
        raw=info["seq"]; w=max(50,info["length"]//50)
        skews,pos=[],[]
        for i in range(0,len(raw)-w,w//2):
            ch=raw[i:i+w]; g,c=ch.count("G"),ch.count("C"); d=g+c
            skews.append((g-c)/d if d else 0); pos.append(i+w//2)
        fs=px.area(x=pos,y=skews,title="GC Skew [(G−C)/(G+C)]",
                   labels={"x":"Position (bp)","y":"GC Skew"},height=260,
                   color_discrete_sequence=["#7c3aed"])
        fs.add_hline(y=0,line_dash="dot",line_color="rgba(124,58,237,.4)")
        fs.update_traces(line=dict(width=1.8),fillcolor="rgba(124,58,237,.08)")
        st.plotly_chart(th(fs),use_container_width=True)

    # Coloured sequence preview
    st.markdown("**Sequence Preview (first 300 characters)**")
    st.markdown(f'<div class="seq-block">{colorize(info["seq"][:300])}{"…" if info["length"]>300 else ""}</div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BLAST CHARTS — comprehensive
# ══════════════════════════════════════════════════════════════════════════════
def blast_charts(df:pd.DataFrame):
    if df.empty: return

    t1,t2,t3,t4=st.tabs(["📊 Identity & E-Value","🗺 Alignment Map","🔵 Dot Plot","📈 Distribution"])

    with t1:
        top10=df.nlargest(10,"Identity (%)").copy()
        top10["Lbl"]=top10["Accession"]+" · "+top10["Description"].str[:35]
        f1=px.bar(top10,x="Identity (%)",y="Lbl",orientation="h",
                  color="Identity (%)",color_continuous_scale=GV,
                  title="Top 10 Hits — Identity %",height=380)
        f1.update_layout(yaxis=dict(autorange="reversed"),coloraxis_showscale=False)
        f1.update_xaxes(range=[0,105])
        st.plotly_chart(th(f1),use_container_width=True)

        sdf=df.copy(); sdf["ev"]=sdf["E-Value"].apply(lambda e:max(e,1e-200))
        f2=px.scatter(sdf,x="Bit Score",y="ev",color="Identity (%)",
                      size="Alignment Length",
                      hover_data=["Accession","Description","Identity (%)","E-Value"],
                      color_continuous_scale=GM,log_y=True,
                      title="E-Value vs Bit Score  (bubble = alignment length)",
                      labels={"ev":"E-Value (log scale)"},height=380)
        st.plotly_chart(th(f2),use_container_width=True)

    with t2:
        # Sequence Alignment Map — horizontal bars per hit showing query coverage
        adf=df.nlargest(20,"Identity (%)").copy()
        qlen=adf["Query Length"].iloc[0] if "Query Length" in adf.columns else 100
        fig=go.Figure()
        for i,row in adf.reset_index(drop=True).iterrows():
            pct_id=row["Identity (%)"]
            col=f"rgba({int(124+pct_id)},{int(58+pct_id*.5)},237,0.85)"
            fig.add_trace(go.Bar(
                y=[f"{row['Accession']}"],x=[row["Query End"]-row["Query Start"]],
                base=[row["Query Start"]],orientation="h",
                marker_color=f"rgba({max(0,int(200-pct_id))},{int(pct_id*2)},{int(pct_id*1.5)},0.8)",
                hovertemplate=(f"<b>{row['Description'][:60]}</b><br>"
                               f"Identity: {row['Identity (%)']}%<br>"
                               f"E-Value: {row['E-Value']:.2e}<br>"
                               f"Query: {row['Query Start']}–{row['Query End']}<extra></extra>"),
                showlegend=False,
            ))
        fig.update_layout(
            title="Sequence Alignment Map — Query Coverage per Hit",
            xaxis_title="Query Position (bp)",
            yaxis=dict(autorange="reversed",tickfont=dict(size=10)),
            height=max(300,len(adf)*28+80),barmode="overlay",
        )
        fig.add_vline(x=qlen,line_dash="dash",line_color="#dc2626",
                      annotation_text="Query End",annotation_position="top right")
        st.plotly_chart(th(fig),use_container_width=True)
        st.caption("Each bar = one hit. Colour: green = high identity, red = low. Width = aligned region on query.")

    with t3:
        st.markdown("#### Dot Plot — Two Sequence Comparison")
        st.info("Paste two sequences below to compare them. Diagonal lines = matching regions.")
        dc1,dc2=st.columns(2)
        with dc1: s1=st.text_area("Sequence 1",height=100,placeholder="ATGCGTACGT…",key="dp1")
        with dc2: s2=st.text_area("Sequence 2",height=100,placeholder="ATGCGTACGT…",key="dp2")
        wsize=st.slider("Word size (k-mer)",3,12,6,key="dpw")
        if st.button("Generate Dot Plot",key="dpbtn"):
            if s1.strip() and s2.strip():
                render_dot_plot(re.sub(r"\s|>.*","",s1).upper(),
                                re.sub(r"\s|>.*","",s2).upper(), wsize)
            else: st.warning("Enter both sequences.")

    with t4:
        c1,c2=st.columns(2)
        with c1:
            f3=px.histogram(df[df["E-Value"]>0],x="E-Value",nbins=35,log_x=True,
                            color_discrete_sequence=["#7c3aed"],
                            title="E-Value Distribution",height=300)
            f3.update_layout(bargap=.06)
            st.plotly_chart(th(f3),use_container_width=True)
        with c2:
            f4=px.box(df,x="Query ID",y="Bit Score",color="Query ID",
                      title="Bit Score by Query",height=300)
            f4.update_layout(showlegend=False); f4.update_xaxes(tickangle=25)
            st.plotly_chart(th(f4),use_container_width=True)

        f5=px.scatter(df,x="Query Coverage (%)",y="Identity (%)",
                      color="E-Value",size="Bit Score",
                      color_continuous_scale="RdYlGn",
                      hover_data=["Accession","Description"],
                      title="Query Coverage vs Identity %",height=360,
                      labels={"E-Value":"E-Value"})
        st.plotly_chart(th(f5),use_container_width=True)


def render_dot_plot(s1:str,s2:str,k:int=6):
    xs,ys=[],[]
    kmers={s1[i:i+k]:i for i in range(len(s1)-k+1)}
    for j in range(len(s2)-k+1):
        mer=s2[j:j+k]
        if mer in kmers:
            xs.append(kmers[mer]); ys.append(j)
    if not xs: st.warning("No matching k-mers found. Try a smaller word size."); return
    fig=px.scatter(x=xs,y=ys,opacity=.4,
                   color_discrete_sequence=["#7c3aed"],
                   labels={"x":"Sequence 1 position","y":"Sequence 2 position"},
                   title=f"Dot Plot (k={k})",height=420)
    fig.update_traces(marker=dict(size=3))
    st.plotly_chart(th(fig),use_container_width=True)
    st.caption(f"{len(xs):,} matching {k}-mers found.")

# ══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE BLAST TABLE  (with all key metrics + exports)
# ══════════════════════════════════════════════════════════════════════════════
DISPLAY_COLS=["Accession","Description","Max Score","Bit Score","E-Value",
              "Identity (%)","Query Coverage (%)","Alignment Length","Gaps"]

def blast_table(df:pd.DataFrame,key:str="tbl"):
    if df.empty: return

    with st.expander("⚙️ Filter & Columns",expanded=False):
        c1,c2,c3=st.columns(3)
        with c1: ev=st.slider("Max E-Value",0.0,1.0,1.0,.001,format="%.3f",key=f"ev_{key}")
        with c2: mi=st.slider("Min Identity %",0.0,100.0,0.0,1.0,key=f"mi_{key}")
        with c3: mc=st.slider("Min Query Coverage %",0.0,100.0,0.0,1.0,key=f"mc_{key}")
        show=st.multiselect("Columns to display",
                            options=[c for c in DISPLAY_COLS if c in df.columns],
                            default=[c for c in DISPLAY_COLS if c in df.columns],
                            key=f"cols_{key}")

    fdf=df[(df["E-Value"]<=ev)&(df["Identity (%)"]>=mi)&(df["Query Coverage (%)"]>=mc)]
    show=[c for c in show if c in fdf.columns]
    disp=fdf[show] if show else fdf

    st.markdown(f"**{len(disp):,}** hits shown · click column header to sort")
    st.dataframe(disp,use_container_width=True,height=420,
                 column_config={
                     "E-Value":st.column_config.NumberColumn(format="%.2e"),
                     "Identity (%)":st.column_config.ProgressColumn(format="%.1f%%",min_value=0,max_value=100),
                     "Query Coverage (%)":st.column_config.ProgressColumn(format="%.1f%%",min_value=0,max_value=100),
                 })

    # ── Download row ──────────────────────────────────────────────────────
    d1,d2,d3=st.columns(3)
    with d1:
        st.download_button("⬇️ CSV",csv_bytes(disp),f"blast_{key}.csv","text/csv",
                           key=f"dcsv_{key}",use_container_width=True)
    with d2:
        if XLSX:
            st.download_button("⬇️ Excel",excel_bytes(disp),f"blast_{key}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key=f"dxlsx_{key}",use_container_width=True)
        else:
            st.button("⬇️ Excel (install openpyxl)",disabled=True,
                      use_container_width=True,key=f"dxlsx_dis_{key}")
    with d3:
        # Download top hits as FASTA
        fasta_lines=[]
        for _,row in fdf.head(20).iterrows():
            if "Sbjct Seq" in row and row["Sbjct Seq"]:
                fasta_lines.append(f">{row['Accession']} {row['Description']}")
                fasta_lines.append(row["Sbjct Seq"])
        if fasta_lines:
            st.download_button("⬇️ Top Hits FASTA","\n".join(fasta_lines).encode(),
                               f"hits_{key}.fasta","text/plain",
                               key=f"dfasta_{key}",use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3D PROTEIN VIEWER  (py3Dmol via HTML component)
# ══════════════════════════════════════════════════════════════════════════════
def protein_3d_viewer(pdb_id:str):
    pid=pdb_id.strip().upper()
    if not pid or len(pid)!=4:
        st.warning("Enter a valid 4-character PDB ID (e.g. 1HHO, 6LU7, 4HHB)"); return

    html=f"""
    <html><head>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js"></script>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <style>
      body{{margin:0;background:linear-gradient(135deg,#f0f4ff,#f5f0ff);}}
      #viewer{{width:100%;height:480px;position:relative;border-radius:16px;overflow:hidden;
               box-shadow:0 4px 24px rgba(124,58,237,.18);}}
      .ctrl{{position:absolute;top:12px;right:12px;z-index:10;display:flex;gap:6px;flex-wrap:wrap;}}
      .btn{{background:rgba(255,255,255,.85);border:1px solid rgba(124,58,237,.25);
            border-radius:8px;padding:5px 12px;font-family:Inter,sans-serif;font-size:12px;
            font-weight:600;color:#7c3aed;cursor:pointer;backdrop-filter:blur(8px);}}
      .btn:hover{{background:rgba(124,58,237,.1);}}
      .label{{position:absolute;bottom:12px;left:12px;background:rgba(255,255,255,.82);
              backdrop-filter:blur(8px);border-radius:10px;padding:6px 14px;
              font-family:Inter,sans-serif;font-size:13px;font-weight:600;color:#1e1f3a;
              border:1px solid rgba(124,58,237,.2);}}
    </style></head><body>
    <div style="position:relative;">
      <div id="viewer"></div>
      <div class="ctrl">
        <button class="btn" onclick="setStyle('cartoon')">Cartoon</button>
        <button class="btn" onclick="setStyle('stick')">Stick</button>
        <button class="btn" onclick="setStyle('sphere')">Sphere</button>
        <button class="btn" onclick="setStyle('surface')">Surface</button>
        <button class="btn" onclick="viewer.spin(true)">Spin</button>
        <button class="btn" onclick="viewer.spin(false)">Stop</button>
      </div>
      <div class="label">PDB: {pid} · drag to rotate · scroll to zoom</div>
    </div>
    <script>
    let viewer=$3Dmol.createViewer(document.getElementById("viewer"),
        {{backgroundColor:"rgba(240,244,255,0.6)"}});
    $3Dmol.download("pdb:{pid}",viewer,{{}},function(){{
        viewer.setStyle({{}},{{cartoon:{{colorscheme:"ssJmol"}}}});
        viewer.zoomTo(); viewer.render();
    }});
    function setStyle(s){{
        viewer.setStyle({{}},{{}});
        if(s==="cartoon") viewer.setStyle({{}},{{cartoon:{{colorscheme:"ssJmol"}}}});
        else if(s==="stick") viewer.setStyle({{}},{{stick:{{colorscheme:"rasmol"}}}});
        else if(s==="sphere") viewer.setStyle({{}},{{sphere:{{colorscheme:"rasmol",radius:.8}}}});
        else if(s==="surface"){{
            viewer.setStyle({{}},{{cartoon:{{colorscheme:"ssJmol"}}}});
            viewer.addSurface($3Dmol.SurfaceType.VDW,{{opacity:.6,colorscheme:"whiteCarbon"}});
        }}
        viewer.render();
    }}
    </script></body></html>
    """
    stc.html(html,height=510)

# ══════════════════════════════════════════════════════════════════════════════
# AI EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
def get_client():
    if not GROQ: return None
    try: return Groq(api_key=GROQ_KEY)
    except: return None

def explain(df,question=""):
    client=get_client()
    if not client: return "❌ AI unavailable."
    rows=df.head(5)[["Accession","Description","Identity (%)","E-Value","Bit Score"]].to_dict("records")
    ctx="\n".join(f"  Hit {i}: {r['Accession']} — {r['Description']}\n"
                  f"    Identity:{r['Identity (%)']}% E-Value:{r['E-Value']:.2e} Bit:{r['Bit Score']:.1f}"
                  for i,r in enumerate(rows,1))
    sys=textwrap.dedent("""You are an expert bioinformatician. Explain BLAST results to a graduate biologist.
        (1) One-paragraph summary. (2) Bullet interpretation of top 3 hits. (3) Flags (contamination,paralogs).
        (4) One next-step recommendation. Under 350 words.""")
    prompt=f"BLAST results:\n{ctx}"+( f"\n\nUser question: {question}" if question.strip() else "")+"\n\nExplain."
    try:
        r=get_client().chat.completions.create(model="llama3-70b-8192",
            messages=[{"role":"system","content":sys},{"role":"user","content":prompt}],
            temperature=.3,max_tokens=600)
        return r.choices[0].message.content
    except Exception as e: return f"❌ {e}"

def ai_widget(df,key):
    if df.empty: return
    st.markdown("---"); st.markdown("### 🤖 AI Explainer")
    q=st.text_input("Follow-up question",placeholder="Is this contamination?",key=f"q_{key}")
    if st.button("✨ Explain results",key=f"ab_{key}"):
        with st.spinner("Thinking…"): ans=explain(df,q)
        st.session_state[f"ai_{key}"]=ans
    if f"ai_{key}" in st.session_state:
        with st.chat_message("assistant",avatar="🧬"):
            st.markdown(st.session_state[f"ai_{key}"])

# ══════════════════════════════════════════════════════════════════════════════
# GC / SEQ PROFILE  (used in GC Dashboard)
# ══════════════════════════════════════════════════════════════════════════════
def profile(seq_str,sid="seq"):
    s=seq_str.upper(); n=len(s)
    if not n: return {}
    cnt={b:s.count(b) for b in "ATGCN"}
    gc=cnt["G"]+cnt["C"]; at=cnt["A"]+cnt["T"]
    try: mw=molecular_weight(Seq(s),seq_type="DNA")
    except: mw=None
    return {"id":sid,"length":n,"gc":round(gc/n*100,2),"at":round(at/n*100,2),
            "cnt":cnt,"mw":mw,"tm":4*gc+2*at if n<30 else None,"raw":s}

def gc_charts(p):
    pie=pd.DataFrame({"Base":list("ATGCN"),"Count":[p["cnt"][b] for b in "ATGCN"]})
    pie=pie[pie["Count"]>0]
    f=px.pie(pie,names="Base",values="Count",hole=.42,color="Base",
             color_discrete_map={"A":"#ef4444","T":"#3b82f6","G":"#22c55e","C":"#f59e0b","N":"#94a3b8"},
             title="Base Composition",height=340)
    f.update_traces(textinfo="label+percent",marker=dict(line=dict(color="white",width=2)))
    c1,c2=st.columns([1,1.4])
    with c1: st.plotly_chart(th(f),use_container_width=True)
    with c2:
        raw=p.get("raw","")
        if raw:
            w=max(50,len(raw)//50); skews,pos=[],[]
            for i in range(0,len(raw)-w,w//2):
                ch=raw[i:i+w]; g,c=ch.count("G"),ch.count("C"); d=g+c
                skews.append((g-c)/d if d else 0); pos.append(i+w//2)
            fs=px.area(x=pos,y=skews,title="GC Skew",
                       labels={"x":"Position","y":"(G-C)/(G+C)"},height=340,
                       color_discrete_sequence=["#7c3aed"])
            fs.add_hline(y=0,line_dash="dot",line_color="rgba(124,58,237,.4)")
            fs.update_traces(line=dict(width=1.8),fillcolor="rgba(124,58,237,.08)")
            st.plotly_chart(th(fs),use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# CENTRAL DOGMA
# ══════════════════════════════════════════════════════════════════════════════
def central_dogma(raw,table_id=1):
    clean=re.sub(r">.*\n?","",raw); clean=re.sub(r"\s","",clean).upper()
    bad=set(clean)-set("ATGCNRYSWKMBDHV")
    if bad: raise ValueError(f"Invalid chars: {', '.join(sorted(bad))}")
    s=Seq(clean)
    return {"DNA 5'→3'":str(s),"Complement 3'→5'":str(s.complement()),
            "Reverse Complement":str(s.reverse_complement()),
            "mRNA":str(s.transcribe()),"Protein":str(s.translate(table=table_id))}

# ══════════════════════════════════════════════════════════════════════════════
# PRIMER DESIGNER
# ══════════════════════════════════════════════════════════════════════════════
def design_primers(raw,length=20):
    s=re.sub(r"\s|>.*","",raw.upper()); s=re.sub(r"[^ATGCN]","",s)
    if len(s)<length*2+20: raise ValueError(f"Need ≥{length*2+20} bp.")
    def stats(seq,name):
        gc=gc_fraction(Seq(seq))*100
        try: tm_nn=Tm_NN(seq,nn_table=DNA_NN4)
        except: tm_nn=None
        try: tm_gc=Tm_GC(seq)
        except: tm_gc=None
        rc=str(Seq(seq).reverse_complement())
        hp=any(seq[i:i+4]==rc[j:j+4] for i in range(len(seq)-3) for j in range(len(rc)-3))
        km=[seq[i:i+4] for i in range(len(seq)-3)]
        return{"name":name,"seq":seq,"len":len(seq),"gc":round(gc,1),
               "tm_nn":round(tm_nn,1) if tm_nn else None,
               "tm_gc":round(tm_gc,1) if tm_gc else None,"hairpin":hp,"dimer":len(km)!=len(set(km))}
    return{"fwd":stats(s[:length],"Forward"),"rev":stats(str(Seq(s[-length:]).reverse_complement()),"Reverse"),"product":len(s)}

def primer_card(p):
    ok=not p["hairpin"] and not p["dimer"]
    issues="✅ No issues" if ok else " · ".join(filter(None,["⚠️ Hairpin" if p["hairpin"] else "","⚠️ Self-dimer" if p["dimer"] else ""]))
    col="#15803d" if ok else "#be185d"
    st.markdown(f"""<div class="gc">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
        <b>{p['name']} Primer</b>
        <span style="font-size:.75rem;font-weight:600;color:{col};">{issues}</span></div>
      <div class="seq-block" style="margin-bottom:12px;">{colorize(p['seq'])}</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;font-size:.82rem;">
        <div><span style="color:#7879a0;">Length</span><br><b>{p['len']} bp</b></div>
        <div><span style="color:#7879a0;">GC</span><br><b>{p['gc']}%</b></div>
        <div><span style="color:#7879a0;">Tm (NN)</span><br><b>{"N/A" if not p['tm_nn'] else f"{p['tm_nn']} °C"}</b></div>
        <div><span style="color:#7879a0;">Tm (GC)</span><br><b>{"N/A" if not p['tm_gc'] else f"{p['tm_gc']} °C"}</b></div>
      </div></div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HOME — clickable card grid
# ══════════════════════════════════════════════════════════════════════════════
TOOLS=[
    ("seqana", "🔬","Seq Analyzer",  "rose",  "Length, MW, GC%, frequency charts — pre-BLAST"),
    ("xml",    "📄","XML Parser",    "violet","Upload BLAST XML → sortable table + CSV/Excel/FASTA"),
    ("blast",  "🌐","Online BLAST",  "sky",   "Submit FASTA → live NCBI qblast"),
    ("batch",  "📦","Batch",         "teal",  "Multi-file FASTA → combined ZIP results"),
    ("ai",     "🤖","AI Explainer",  "amber", "LLaMA 3 explains top hits plainly"),
    ("dogma",  "🔀","Central Dogma", "pink",  "DNA → mRNA → Protein translation"),
    ("gc",     "📊","GC Dashboard",  "green", "GC%, composition, skew profiling"),
    ("phylo",  "🌿","Phylo Viewer",  "orange","Render Newick / NEXUS phylo trees"),
    ("primer", "🔬","Primer Design", "blue",  "Auto-design primers with Tm & hairpin check"),
    ("prot3d", "🧊","3D Protein",    "rose",  "Interactive 3D protein structure viewer"),
]

def home():
    st.markdown("""
    <div style="text-align:center;padding:20px 0 30px;">
      <div style="font-size:2.6rem;margin-bottom:8px;">🧬</div>
      <h1 style="margin:0 0 6px;font-size:2rem!important;">BLAST BioSuite</h1>
      <p style="color:#7879a0;margin:0;">Select a tool to get started</p>
    </div>""",unsafe_allow_html=True)

    for row_start in range(0,len(TOOLS),5):
        row=TOOLS[row_start:row_start+5]
        cols=st.columns(len(row))
        for col,(pid,icon,title,acc,desc) in zip(cols,row):
            ct,cbg,cb,cbh=ACCENT[acc]
            with col:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,.72);backdrop-filter:blur(18px) saturate(160%);
                  border:1.5px solid {cb};border-radius:18px;padding:22px 18px 12px;
                  box-shadow:0 2px 16px rgba(100,110,180,.1),inset 0 1px 0 rgba(255,255,255,.85);
                  margin-bottom:0;">
                  <div style="width:42px;height:42px;border-radius:11px;background:{cbg};
                    border:1px solid {cbh};display:flex;align-items:center;justify-content:center;
                    font-size:1.3rem;margin-bottom:10px;">{icon}</div>
                  <div style="font-weight:700;color:#1e1f3a;font-size:.9rem;margin-bottom:4px;">{title}</div>
                  <div style="font-size:.75rem;color:#7879a0;line-height:1.4;margin-bottom:12px;">{desc}</div>
                </div>""",unsafe_allow_html=True)
                if st.button(f"Open",key=f"nav_{pid}",use_container_width=True): go(pid)
        st.markdown("<div style='margin-bottom:6px'></div>",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SEQUENCE ANALYZER (pre-BLAST)
# ══════════════════════════════════════════════════════════════════════════════
def page_seqana():
    need_bio()
    page_header("🔬","Sequence Analyzer","rose")
    st.markdown("Paste or upload your sequence for a quick profile before running BLAST.")

    c1,c2=st.columns([2,1])
    with c1:
        raw=st.text_area("Paste sequence (FASTA or raw)",height=160,
                          placeholder=">my_seq\nATGCGTACGTAGCTAGCTAGCT…")
        fu=st.file_uploader("…or upload FASTA",type=["fasta","fa","fna","faa","txt"],key="sa_up")
        if fu: raw=fu.read().decode()
    with c2:
        st.markdown('<div class="gc-sm" style="margin-top:0"><b style="font-size:.85rem;">What this shows</b><br>'
                    '<ul style="font-size:.78rem;color:#7879a0;margin:6px 0 0;padding-left:16px;line-height:1.8">'
                    '<li>Sequence length &amp; molecular weight</li>'
                    '<li>GC / AT content</li>'
                    '<li>Melting temperature (short seqs)</li>'
                    '<li>Nucleotide / AA frequency bar chart</li>'
                    '<li>Base composition pie chart</li>'
                    '<li>GC skew profile (DNA)</li>'
                    '</ul></div>',unsafe_allow_html=True)

    if raw.strip():
        info=seq_analysis(raw)
        if info: render_seq_analysis(info)
        else: st.error("Could not parse sequence.")

    with st.expander("💡 Example sequences"):
        st.code(">BRCA1_partial\nATGGATTTATCTGCTCTTCGCGTTGAAGAAGTACAAAATGTCATTAATGCTATGCAGAAAATCTTAGAGTGTCCCATCTGTCTGGAGTTGATCAAGGAACCTGTCTCC",language="text")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: XML PARSER
# ══════════════════════════════════════════════════════════════════════════════
def page_xml():
    need_bio()
    page_header("📄","BLAST XML Parser","violet")
    up=st.file_uploader("Upload BLAST XML (outfmt 5)",type=["xml"])
    if not up: return
    with st.spinner("Parsing…"): df=parse_xml(up.read())
    if df.empty: st.error("No hits found."); return
    st.success(f"✅ {len(df):,} HSPs · {df['Query ID'].nunique()} queries")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("HSPs",f"{len(df):,}"); c2.metric("Queries",df["Query ID"].nunique())
    c3.metric("Unique Hits",df["Accession"].nunique()); c4.metric("Best E-Value",f"{df['E-Value'].min():.2e}")
    t1,t2,t3=st.tabs(["📋 Table","📊 Charts","🤖 AI"])
    with t1: blast_table(df,"xml")
    with t2: blast_charts(df)
    with t3: ai_widget(df,"xml")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ONLINE BLAST
# ══════════════════════════════════════════════════════════════════════════════
def page_blast():
    need_bio()
    page_header("🌐","Online NCBI BLAST","sky")
    st.warning("⏳ 30–120 seconds on NCBI servers.",icon="⚠️")
    c1,c2=st.columns(2)
    with c1: prog=st.selectbox("Program",["blastn","blastp","blastx","tblastn","tblastx"])
    with c2: db=st.selectbox("Database",["nt","nr","swissprot","refseq_rna","refseq_protein"])
    seq=st.text_area("FASTA",height=140,placeholder=">seq\nATGCGT…")
    fu=st.file_uploader("…or upload FASTA",type=["fasta","fa","fna","faa","txt"])
    if fu: seq=fu.read().decode()

    # Pre-BLAST analysis inline
    if seq.strip():
        with st.expander("🔬 Quick Sequence Profile",expanded=False):
            info=seq_analysis(seq)
            if info: render_seq_analysis(info)

    if st.button("🚀 Submit to NCBI",disabled=not seq.strip()):
        with st.spinner(f"Running {prog}…"):
            try: df=run_blast(seq.strip(),prog,db)
            except Exception as e: st.error(f"Failed: {e}"); return
        if df.empty: st.warning("No hits returned."); return
        st.success(f"✅ {len(df):,} HSPs")
        c1,c2,c3=st.columns(3)
        c1.metric("Unique Hits",df["Accession"].nunique())
        c2.metric("Best E-Value",f"{df['E-Value'].min():.2e}")
        c3.metric("Max Identity",f"{df['Identity (%)'].max():.1f}%")
        t1,t2,t3,t4=st.tabs(["📋 Table","📊 Charts","🤖 AI","🧊 3D Structure"])
        with t1: blast_table(df,"ob")
        with t2: blast_charts(df)
        with t3: ai_widget(df,"online")
        with t4:
            if prog=="blastp":
                st.markdown("#### Interactive 3D Protein Viewer")
                pdb=st.text_input("PDB ID from results",placeholder="e.g. 1HHO",max_chars=4)
                if pdb: protein_3d_viewer(pdb)
            else:
                st.info("3D structure viewer is available for blastp results only.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BATCH PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════
def page_batch():
    need_bio()
    page_header("📦","Batch FASTA Processor","teal")
    st.warning("One NCBI query per file — 5 files ≈ 5–10 min.",icon="⚠️")
    c1,c2=st.columns(2)
    with c1: bp=st.selectbox("Program",["blastn","blastp","blastx"],key="bp")
    with c2: bd=st.selectbox("Database",["nt","nr","swissprot"],key="bd")
    files=st.file_uploader("Upload FASTA files",type=["fasta","fa","fna","faa"],accept_multiple_files=True)
    if not files: return
    st.info(f"📁 {len(files)} file(s) queued.")
    if st.button(f"▶️ Run Batch ({len(files)} files)"):
        all_dfs,fdfs=[],{}; prg=st.progress(0,"Starting…"); st_=st.empty()
        for i,f in enumerate(files):
            seq=f.read().decode("utf-8","replace")
            st_.markdown(f"⏳ **{f.name}** ({i+1}/{len(files)})…")
            try:
                dfi=run_blast(seq,bp,bd); dfi.insert(0,"Source File",f.name)
                fdfs[f.name]=dfi; all_dfs.append(dfi)
                st_.success(f"✅ `{f.name}` → {len(dfi):,} HSPs")
            except Exception as e: st_.error(f"❌ {f.name}: {e}")
            prg.progress((i+1)/len(files),f"{i+1}/{len(files)}")
        prg.empty(); st_.empty()
        if not all_dfs: st.error("All queries failed."); return
        combined=pd.concat(all_dfs,ignore_index=True)
        st.success(f"🎉 {len(combined):,} HSPs from {len(all_dfs)} queries.")
        smry=combined.groupby("Source File").agg(
            HSPs=("Max Score","count"),Hits=("Accession","nunique"),
            BestE=("E-Value","min"),MaxIdent=("Identity (%)","max")).reset_index().round(4)
        st.dataframe(smry,use_container_width=True)
        d1,d2,d3=st.columns(3)
        with d1: st.download_button("⬇️ Combined CSV",csv_bytes(combined),"batch.csv","text/csv")
        with d2:
            if XLSX: st.download_button("⬇️ Excel",excel_bytes(combined),"batch.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with d3:
            zbuf=io.BytesIO()
            with zipfile.ZipFile(zbuf,"w",zipfile.ZIP_DEFLATED) as zf:
                for fn,dfi in fdfs.items(): zf.writestr(Path(fn).stem+"_blast.csv",dfi.to_csv(index=False))
            zbuf.seek(0)
            st.download_button("📦 Download ZIP",zbuf.getvalue(),"batch.zip","application/zip")
        st.markdown("---"); blast_charts(combined); ai_widget(combined,"batch")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI EXPLAINER (standalone)
# ══════════════════════════════════════════════════════════════════════════════
def page_ai():
    page_header("🤖","AI Explainer","amber")
    if not GROQ: st.error("Run `pip install groq`"); return
    if not GROQ_KEY: st.warning("⚠️ GROQ_API_KEY not found.")
    up=st.file_uploader("Upload BLAST XML",type=["xml"])
    if up:
        need_bio()
        with st.spinner("Parsing…"): df=parse_xml(up.read())
        if df.empty: st.error("No hits found."); return
        st.success(f"✅ {len(df):,} HSPs loaded.")
        st.dataframe(df.head(5)[[c for c in DISPLAY_COLS if c in df.columns]],use_container_width=True)
        ai_widget(df,"standalone")
    st.markdown("---"); st.markdown("### 💬 Ask a question")
    if "chat" not in st.session_state: st.session_state.chat=[]
    for m in st.session_state.chat:
        with st.chat_message(m["role"],avatar="🧬" if m["role"]=="assistant" else "🧑‍🔬"):
            st.markdown(m["content"])
    if prompt:=st.chat_input("Ask about BLAST, sequences, biology…"):
        st.session_state.chat.append({"role":"user","content":prompt})
        with st.chat_message("user",avatar="🧑‍🔬"): st.markdown(prompt)
        c=get_client()
        if c:
            with st.chat_message("assistant",avatar="🧬"):
                with st.spinner("…"):
                    try:
                        msgs=[{"role":"system","content":"You are an expert bioinformatician."}]+st.session_state.chat
                        r=c.chat.completions.create(model="llama3-70b-8192",messages=msgs,max_tokens=500,temperature=.3)
                        ans=r.choices[0].message.content
                    except Exception as e: ans=f"❌ {e}"
                    st.markdown(ans)
                    st.session_state.chat.append({"role":"assistant","content":ans})

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CENTRAL DOGMA
# ══════════════════════════════════════════════════════════════════════════════
def page_dogma():
    need_bio()
    page_header("🔀","Central Dogma Tool","pink")
    c1,c2=st.columns([2,1])
    with c1: dna=st.text_area("DNA sequence",height=150,placeholder=">gene\nATGCGT…")
    with c2:
        tbl=st.selectbox("Genetic code",list(TABLES.keys()))
        show_rc=st.checkbox("Reverse Complement",True)
        show_rna=st.checkbox("mRNA",True); show_aa=st.checkbox("Protein",True)
    if st.button("🔀 Translate",disabled=not dna.strip()):
        try:
            res=central_dogma(dna,TABLES[tbl])
            gc_=round(gc_fraction(Seq(res["DNA 5'→3'"]))*100,1)
            st.markdown(f'<div style="display:flex;gap:8px;margin:10px 0 16px;flex-wrap:wrap;">'
                        f'<span class="badge bv">{len(res["DNA 5\'→3\'"])} bp</span>'
                        f'<span class="badge bg">GC {gc_}%</span>'
                        f'<span class="badge bs">{tbl}</span></div>',unsafe_allow_html=True)
            st.markdown(seq_block(res["DNA 5'→3'"],"DNA 5'→3'","bv"),unsafe_allow_html=True)
            st.markdown(seq_block(res["Complement 3'→5'"],"Complement 3'→5'","bs"),unsafe_allow_html=True)
            if show_rc: st.markdown(seq_block(res["Reverse Complement"],"Reverse Complement","bt"),unsafe_allow_html=True)
            if show_rna: st.markdown(seq_block(res["mRNA"],"mRNA","ba"),unsafe_allow_html=True)
            if show_aa:
                prot=res["Protein"]
                st.markdown(f'<span class="badge bp">Protein</span><div class="seq-block" style="margin-top:6px;color:#be185d;letter-spacing:.15em">{prot}</div>',unsafe_allow_html=True)
                adf=pd.Series(list(prot.replace("*",""))).value_counts().reset_index(); adf.columns=["AA","Count"]
                f=px.bar(adf.head(15),x="AA",y="Count",color="Count",color_continuous_scale=GV,
                         title="Amino Acid Frequency",height=260); f.update_layout(coloraxis_showscale=False)
                st.plotly_chart(th(f),use_container_width=True)
        except ValueError as e: st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: GC DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_gc():
    need_bio()
    page_header("📊","GC Content Dashboard","green")
    up=st.file_uploader("Upload FASTA",type=["fasta","fa","fna","txt"])
    manual=st.text_area("…or paste raw DNA",height=90,placeholder="ATGCGATCGATCG…")
    seqs=[]
    if up:
        txt=up.read().decode("utf-8","replace")
        recs=list(SeqIO.parse(io.StringIO(txt),"fasta"))
        if recs: seqs=[(r.id,str(r.seq)) for r in recs[:20]]
        else:
            raw=re.sub(r"\s","",txt.upper())
            if re.match(r"^[ATGCN]+$",raw): seqs=[("uploaded",raw)]
    elif manual.strip():
        seqs=[("manual",re.sub(r"\s","",manual).upper())]
    if not seqs: return
    profiles=[profile(s,sid) for sid,s in seqs]
    st.success(f"✅ {len(profiles)} sequence(s) loaded.")
    sel=st.selectbox("Sequence",[p["id"] for p in profiles]) if len(profiles)>1 else profiles[0]["id"]
    p=next(x for x in profiles if x["id"]==sel)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Length",f"{p['length']:,} bp"); c2.metric("GC%",f"{p['gc']}%")
    c3.metric("AT%",f"{p['at']}%"); c4.metric("MW",f"{p['mw']/1000:.1f} kDa" if p['mw'] else "N/A")
    gc_charts(p)
    if len(profiles)>1:
        st.markdown("### Summary")
        sdf=pd.DataFrame([{"ID":x["id"],"Len":x["length"],"GC%":x["gc"],"AT%":x["at"],"MW(kDa)":round(x["mw"]/1000,1) if x["mw"] else None} for x in profiles])
        st.dataframe(sdf,use_container_width=True)
        st.download_button("⬇️ CSV",csv_bytes(sdf),"gc_summary.csv","text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PHYLO VIEWER
# ══════════════════════════════════════════════════════════════════════════════
def page_phylo():
    need_bio()
    page_header("🌿","Phylogenetic Tree Viewer","orange")
    c1,c2=st.columns([2,1])
    with c1: up=st.file_uploader("Upload tree file",type=["nwk","tree","dnd","nex","nexus","txt"])
    with c2: fmt=st.selectbox("Format",["newick","nexus","nexml","phyloxml"])
    if not up:
        st.markdown('<div class="gc" style="text-align:center;padding:36px"><div style="font-size:2.5rem">🌿</div><div style="color:#7879a0">Upload a Newick or NEXUS tree file</div></div>',unsafe_allow_html=True)
        with st.expander("Example Newick"):
            st.code("((Homo_sapiens:0.12,Pan_troglodytes:0.08):0.05,(Mus_musculus:0.25,Rattus_norvegicus:0.22):0.10,Danio_rerio:0.45);")
        return
    try: tree=Phylo.read(io.StringIO(up.read().decode("utf-8","replace")),fmt)
    except Exception as e: st.error(f"Cannot parse tree: {e}"); return
    terms=tree.get_terminals()
    fig,ax=plt.subplots(figsize=(12,max(5,len(terms)*.38))); fig.patch.set_facecolor("#f8f9ff"); ax.set_facecolor("#f8f9ff")
    Phylo.draw(tree,axes=ax,do_show=False)
    ax.tick_params(colors="#7879a0")
    for sp in ax.spines.values(): sp.set_edgecolor("rgba(150,160,210,.3)")
    for ln in ax.get_lines(): ln.set_color("#7c3aed"); ln.set_alpha(.7); ln.set_linewidth(1.3)
    for txt in ax.texts: txt.set_color("#1e1f3a"); txt.set_fontsize(9)
    ax.set_title("Phylogenetic Tree",color="#7c3aed",fontsize=14,pad=10); plt.tight_layout()
    st.pyplot(fig,use_container_width=True); plt.close(fig)
    depths=[tree.distance(t) for t in terms]
    c1,c2,c3=st.columns(3)
    c1.metric("Terminal Taxa",len(terms)); c2.metric("Internal Nodes",len(tree.get_nonterminals()))
    c3.metric("Max Depth",f"{max(depths):.4f}" if depths else "N/A")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRIMER DESIGNER
# ══════════════════════════════════════════════════════════════════════════════
def page_primer():
    need_bio()
    page_header("🔬","PCR Primer Designer","blue")
    c1,c2=st.columns([2,1])
    with c1:
        seq=st.text_area("Target DNA",height=170,placeholder=">target\nATGCGT…")
        fu=st.file_uploader("…or upload FASTA",type=["fasta","fa","fna","txt"],key="pfu")
        if fu: seq=fu.read().decode()
    with c2:
        plen=st.slider("Primer length (bp)",15,30,20)
        st.markdown('<div class="gc-sm"><div style="font-size:.78rem;color:#7879a0;line-height:1.8"><b style="color:#1e1f3a;">Ideal specs</b><br>GC: 40–60% · Tm: 55–65 °C<br>ΔTm &lt; 5 °C · No hairpins</div></div>',unsafe_allow_html=True)
    if st.button("🔬 Design Primers",disabled=not seq.strip()):
        try:
            res=design_primers(seq,plen); c1,c2=st.columns(2)
            with c1: primer_card(res["fwd"])
            with c2: primer_card(res["rev"])
            st.markdown(f'<div class="gc-sm" style="text-align:center"><span style="color:#7879a0;font-size:.82rem;">Product Size</span><br><b style="font-size:1.5rem;color:#1d4ed8;">{res["product"]:,} bp</b></div>',unsafe_allow_html=True)
            ft=res["fwd"]["tm_nn"] or res["fwd"]["tm_gc"] or 0
            rt=res["rev"]["tm_nn"] or res["rev"]["tm_gc"] or 0
            dtm=abs(ft-rt)
            if dtm>5: st.warning(f"ΔTm = {dtm:.1f} °C (>5 °C may reduce efficiency)")
            else: st.success(f"✅ ΔTm = {dtm:.1f} °C")
        except ValueError as e: st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: 3D PROTEIN VIEWER
# ══════════════════════════════════════════════════════════════════════════════
def page_prot3d():
    page_header("🧊","3D Protein Structure","rose")
    st.markdown("Enter a **PDB ID** to load the 3D structure. Drag to rotate, scroll to zoom.")
    c1,c2=st.columns([1,2])
    with c1:
        pdb=st.text_input("PDB ID",placeholder="e.g. 1HHO",max_chars=4).strip().upper()
        st.markdown("""
        <div class="gc-sm" style="margin-top:8px">
          <div style="font-size:.78rem;color:#7879a0;line-height:1.8">
            <b style="color:#1e1f3a">Common PDB IDs</b><br>
            <code>1HHO</code> — Hemoglobin<br>
            <code>4HHB</code> — Deoxy-hemoglobin<br>
            <code>6LU7</code> — SARS-CoV-2 protease<br>
            <code>1TUP</code> — p53 tumor suppressor<br>
            <code>1CRN</code> — Crambin (small)<br>
            <code>2LYZ</code> — Lysozyme
          </div>
        </div>""",unsafe_allow_html=True)
        style=st.selectbox("Default style",["Cartoon","Stick","Sphere"])
    with c2:
        if pdb and len(pdb)==4:
            protein_3d_viewer(pdb)
        else:
            st.markdown('<div class="gc" style="text-align:center;padding:60px 20px"><div style="font-size:3rem">🧊</div><div style="color:#7879a0;margin-top:8px">Enter a 4-character PDB ID to load the structure</div></div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
P=st.session_state.page
if   P=="home":   home()
elif P=="seqana": page_seqana()
elif P=="xml":    page_xml()
elif P=="blast":  page_blast()
elif P=="batch":  page_batch()
elif P=="ai":     page_ai()
elif P=="dogma":  page_dogma()
elif P=="gc":     page_gc()
elif P=="phylo":  page_phylo()
elif P=="primer": page_primer()
elif P=="prot3d": page_prot3d()
else: go("home")
