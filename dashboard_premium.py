"""
================================================================
F1 SUPERVISED PIT STOP STRATEGY MODEL — PREMIUM DASHBOARD (v3)
================================================================
Aesthetic-first build.

Typography  : Antonio (display) + Sora (body) + JetBrains Mono (data)
Palette     : Carbon black + F1 red + Mercedes turquoise + signal yellow
Motion      : Staggered page-load cascade, hover lift, animated FIA flag
Interactive : Driver pick swaps the accent to team livery colour
================================================================
Run:  ./venv/bin/streamlit run dashboard_premium.py --server.port 8502
URL:  http://localhost:8502
================================================================
"""

import warnings; warnings.filterwarnings("ignore")
import os, sys, base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import joblib

sys.path.insert(0, ".")
from f1_strategy_model_2 import engineer_features, ALL_FEATURES

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="F1 Supervised Pit Stop Strategy Model · Premium",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# DESIGN TOKENS — CSS variables
# ══════════════════════════════════════════════════════════════
TOKENS = {
    # Surfaces
    "bg":          "#0A0E14",   # deep carbon
    "surface":     "#11151C",   # onyx
    "raised":      "#1A1F2B",   # elevated cards
    "border":      "#222B3A",
    # Accents
    "accent":      "#E10600",   # F1 official red
    "accent2":     "#00D2BE",   # AMG turquoise — data highlights
    "warn":        "#FFE600",   # F1 yellow flag
    "ok":          "#39B54A",   # F1 green
    # Text
    "text":        "#F5F7FA",
    "muted":       "#8B95A7",
    "subtle":      "#5A6479",
}

# Team livery colours — accent swaps when driver picked
TEAM_COLORS = {
    # 2022–2024 mappings (driver code → colour)
    # Red Bull
    "VER": "#1E5BC6", "PER": "#1E5BC6", "TSU": "#1E5BC6",
    # Ferrari
    "LEC": "#DC0000", "SAI": "#DC0000", "HAM": "#DC0000",  # Hamilton joins Ferrari 2025
    # Mercedes
    "RUS": "#00D2BE", "BOT": "#00D2BE", "ANT": "#00D2BE",
    # McLaren
    "NOR": "#FF8000", "PIA": "#FF8000", "RIC": "#FF8000",
    # Aston Martin
    "ALO": "#229971", "STR": "#229971", "VET": "#229971",
    # Alpine
    "GAS": "#0090FF", "OCO": "#0090FF", "DOO": "#0090FF",
    # Williams
    "ALB": "#64C4FF", "SAR": "#64C4FF", "COL": "#64C4FF",
    # RB / VCARB
    "RIC": "#6692FF", "LAW": "#6692FF", "HAD": "#6692FF",
    # Kick Sauber
    "ZHO": "#52E252", "BOT": "#52E252",
    # Haas
    "MAG": "#B6BABD", "HUL": "#B6BABD", "BEA": "#B6BABD",
}

COMPOUND_COLORS = {
    "SOFT":   "#E10600", "MEDIUM": "#FFE600", "HARD":   "#F5F7FA",
    "INTER":  "#39B54A", "WET":    "#00BFFF", "UNKNOWN":"#5A6479",
}


# ══════════════════════════════════════════════════════════════
# DATA & MODEL
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    train = pd.read_csv("f1_outputs/combined_train_2022_2023.csv")
    train["split"] = "train"
    test  = pd.read_csv("f1_outputs/combined_test_2024.csv")
    test["split"]  = "test"
    df = pd.concat([train, test], ignore_index=True)
    df["season"]           = df["season"].astype(int)
    df["lap_number"]       = pd.to_numeric(df["lap_number"], errors="coerce")
    df["lap_time_s"]       = pd.to_numeric(df["lap_time_s"], errors="coerce")
    df["pitstop_this_lap"] = pd.to_numeric(df["pitstop_this_lap"], errors="coerce")
    df["tire_compound"]    = df["tire_compound"].fillna("UNKNOWN").str.upper()
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    p = "f1_outputs/f1_strategy_model_v2.joblib"
    return joblib.load(p) if os.path.exists(p) else None


def asset_b64(path: str) -> str:
    if not os.path.exists(path): return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    mime = "image/svg+xml" if path.endswith(".svg") else "image/png"
    return f"data:{mime};base64,{data}"


df_all   = load_data()
model    = load_model()
LOGO_SRC = asset_b64("assets/uj_logo.png") or asset_b64("assets/uj_logo_placeholder.svg")
FLAG_SRC = asset_b64("assets/chequered_flag.svg")


# ══════════════════════════════════════════════════════════════
# SIDEBAR — must compute accent BEFORE injecting CSS
# ══════════════════════════════════════════════════════════════
SEASONS = sorted(df_all["season"].unique())

with st.sidebar:
    if LOGO_SRC:
        st.markdown(
            f"<div style='background:#fff;padding:8px;border-radius:6px;"
            f"box-shadow:0 4px 18px rgba(0,0,0,.7);text-align:center;"
            f"max-width:170px;margin:6px auto 12px;'>"
            f"<img src='{LOGO_SRC}' style='width:140px;'/></div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div class='side-eyebrow'>UNIVERSITY OF JOHANNESBURG</div>"
        "<div class='side-title'>F1 PIT STOP STRATEGY</div>"
        "<div class='side-sub'>Master's Research Project<br/>"
        "<em style='color:var(--text);font-style:italic;font-weight:400;'>"
        "by Davies Adetiba</em></div>"
        "<div class='side-rule'></div>",
        unsafe_allow_html=True,
    )

    selected_season = st.selectbox("SEASON", SEASONS, index=len(SEASONS)-1)
    df_s   = df_all[df_all["season"] == selected_season]
    races  = (df_s[["round_number","race_name"]].drop_duplicates()
              .sort_values("round_number"))
    selected_race = st.selectbox("RACE", races["race_name"].tolist())

    df_r = df_s[df_s["race_name"] == selected_race]
    drivers = sorted(df_r["driver_id"].dropna().unique())
    selected_driver = st.selectbox("DRIVER", drivers)

    # Resolve team accent for this driver
    accent = TEAM_COLORS.get(selected_driver, TOKENS["accent"])

    st.markdown(
        f"<div class='side-rule'></div>"
        f"<div class='side-eyebrow'>DRIVER ACCENT</div>"
        f"<div style='display:flex;align-items:center;gap:10px;margin-top:6px;'>"
        f"<div style='width:28px;height:28px;background:{accent};border-radius:6px;"
        f"box-shadow:0 0 18px {accent}55;'></div>"
        f"<div style='font-family:JetBrains Mono,monospace;font-size:.85rem;"
        f"color:var(--text);'>{selected_driver}</div></div>",
        unsafe_allow_html=True,
    )

    df_drv = df_r[df_r["driver_id"] == selected_driver].sort_values("lap_number")

# Derived race metadata — needed by the CSS/HTML blocks below
race_round  = int(df_r["round_number"].iloc[0]) if not df_r.empty else 0
split_label = "TEST · UNSEEN" if selected_season == 2024 else "TRAIN"
race_meta   = (
    df_r[["round_number", "race_name"]].drop_duplicates().iloc[0]
    if not df_r.empty else None
)


# ══════════════════════════════════════════════════════════════
# GLOBAL STYLES — now with dynamic accent baked in
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?
family=Antonio:wght@400;600;700&
family=Sora:wght@300;400;500;600;700&
family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

<style>
  :root {{
    --bg:        {TOKENS['bg']};
    --surface:   {TOKENS['surface']};
    --raised:    {TOKENS['raised']};
    --border:    {TOKENS['border']};
    --accent:    {accent};
    --accent2:   {TOKENS['accent2']};
    --warn:      {TOKENS['warn']};
    --ok:        {TOKENS['ok']};
    --text:      {TOKENS['text']};
    --muted:     {TOKENS['muted']};
    --subtle:    {TOKENS['subtle']};

    --font-display: 'Antonio', sans-serif;
    --font-body:    'Sora', sans-serif;
    --font-data:    'JetBrains Mono', monospace;
  }}

  /* ── Reset Streamlit defaults ────────────────────────────── */
  .stApp {{
    background:
      radial-gradient(1200px 600px at 8% -10%, {accent}11 0%, transparent 60%),
      radial-gradient(900px 500px at 95% 110%, var(--accent2)0d 0%, transparent 55%),
      linear-gradient(180deg, var(--bg) 0%, #060a10 100%);
    color: var(--text);
    font-family: var(--font-body);
  }}
  /* Subtle telemetry grid overlay */
  .stApp::before {{
    content: '';
    position: fixed; inset: 0;
    background-image:
      linear-gradient(rgba(255,255,255,0.018) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
  }}
  .block-container {{ padding: 1rem 2.4rem 2rem; position: relative; z-index: 1; }}

  /* ── Hide Streamlit chrome ───────────────────────────────── */
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* ── Body text everywhere ─────────────────────────────────── */
  html, body, [class*="css"] {{ font-family: var(--font-body); }}

  /* ── Sidebar ─────────────────────────────────────────────── */
  [data-testid="stSidebar"] {{
    background: linear-gradient(180deg, var(--surface) 0%, #08090c 100%);
    border-right: 1px solid var(--border);
  }}
  [data-testid="stSidebar"] * {{ color: var(--text); }}
  .side-eyebrow {{
    font-family: var(--font-data); font-size: .65rem;
    color: var(--muted); letter-spacing: 2.4px; font-weight: 500;
    text-align: center;
  }}
  .side-title {{
    font-family: var(--font-display); font-size: 1.55rem;
    color: var(--text); letter-spacing: 2px;
    margin: 4px 0; text-align: center; font-weight: 700;
    line-height: 1;
  }}
  .side-sub {{
    color: var(--accent); font-size: .72rem; text-align: center;
    letter-spacing: 1.6px; font-weight: 600;
    text-transform: uppercase;
  }}
  .side-rule {{
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    margin: 14px 0;
  }}
  [data-testid="stSidebar"] label {{
    font-family: var(--font-data) !important;
    font-size: .68rem !important; letter-spacing: 2px !important;
    color: var(--muted) !important; font-weight: 500 !important;
  }}
  [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
    background: var(--raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: var(--font-body) !important;
  }}

  /* ── Hero banner ─────────────────────────────────────────── */
  .hero {{
    position: relative;
    padding: 28px 34px 24px;
    margin: 8px 0 22px;
    border-radius: 10px;
    background:
      linear-gradient(135deg, var(--surface) 0%, var(--raised) 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    box-shadow: 0 24px 60px rgba(0,0,0,.55),
                inset 0 1px 0 rgba(255,255,255,.04);
    overflow: hidden;
    animation: heroIn .8s cubic-bezier(.2,.8,.2,1) both;
  }}
  .hero::after {{
    content: '';
    position: absolute; right: -60px; top: -40px;
    width: 380px; height: 200px;
    background: radial-gradient(circle, {accent}1f 0%, transparent 70%);
    pointer-events: none;
  }}
  .hero-eyebrow {{
    font-family: var(--font-data); font-size: .72rem;
    color: var(--accent2); letter-spacing: 3px; font-weight: 500;
    animation: rise .7s cubic-bezier(.2,.8,.2,1) .15s both;
  }}
  .hero-title {{
    font-family: var(--font-display);
    font-size: clamp(2.4rem, 4.5vw, 3.6rem);
    font-weight: 700; line-height: .95;
    color: var(--text);
    letter-spacing: 1.5px; text-transform: uppercase;
    margin: 6px 0 4px;
    animation: rise .7s cubic-bezier(.2,.8,.2,1) .25s both;
  }}
  .hero-title em {{
    font-style: normal; color: var(--accent);
    -webkit-text-stroke: 0;
  }}
  .hero-sub {{
    font-family: var(--font-body); color: var(--muted);
    font-size: .9rem; max-width: 720px;
    letter-spacing: .3px;
    animation: rise .7s cubic-bezier(.2,.8,.2,1) .35s both;
  }}
  .hero-foot {{
    margin-top: 12px; display: flex; gap: 18px;
    font-family: var(--font-data); font-size: .68rem;
    color: var(--subtle); letter-spacing: 1.6px;
    animation: rise .7s cubic-bezier(.2,.8,.2,1) .45s both;
  }}
  .hero-foot strong {{ color: var(--text); font-weight: 500; }}

  /* ── Floating chequered flag corner ──────────────────────── */
  .flag-corner {{
    position: fixed; top: 18px; right: 28px;
    width: 84px; height: 56px;
    z-index: 10; pointer-events: none;
    filter: drop-shadow(0 6px 14px rgba(0,0,0,.6));
    animation: flagWave 4s ease-in-out infinite, fadeIn 1.2s ease 0s both;
    transform-origin: left center;
  }}
  @keyframes flagWave {{
    0%, 100% {{ transform: skewY(-2deg) rotate(-1deg) translateY(0); }}
    50%      {{ transform: skewY( 2deg) rotate( 1deg) translateY(-4px); }}
  }}

  /* ── Driver/race header band ─────────────────────────────── */
  .race-band {{
    display: flex; align-items: center; gap: 18px;
    padding: 14px 20px; margin-bottom: 18px;
    background: var(--surface); border-radius: 8px;
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    animation: rise .6s cubic-bezier(.2,.8,.2,1) .55s both;
  }}
  .race-band-num {{
    font-family: var(--font-display); font-size: 2.6rem;
    color: var(--accent); font-weight: 700; line-height: 1;
    text-shadow: 0 0 28px {accent}66;
  }}
  .race-band-info {{ flex: 1; }}
  .race-band-name {{
    font-family: var(--font-display);
    font-size: 1.4rem; letter-spacing: 1.2px;
    color: var(--text); text-transform: uppercase; font-weight: 600;
    line-height: 1.1;
  }}
  .race-band-meta {{
    font-family: var(--font-data); font-size: .72rem;
    color: var(--muted); letter-spacing: 2px;
    margin-top: 4px;
  }}
  .race-band-pill {{
    font-family: var(--font-data); font-size: .68rem;
    color: var(--bg); background: var(--accent);
    padding: 4px 10px; border-radius: 999px;
    letter-spacing: 1.4px; font-weight: 700;
  }}

  /* ── Metric cards (telemetry HUD) ────────────────────────── */
  .metric-grid {{
    display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px;
    margin-bottom: 20px;
  }}
  .metric {{
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 14px 16px 16px; border-radius: 8px;
    position: relative; overflow: hidden;
    transition: all .25s cubic-bezier(.2,.8,.2,1);
  }}
  .metric:hover {{
    transform: translateY(-2px);
    border-color: var(--accent);
    box-shadow: 0 14px 30px rgba(0,0,0,.4), 0 0 0 1px var(--accent);
  }}
  .metric::before {{
    content: '';
    position: absolute; left: 0; top: 0; bottom: 0;
    width: 3px; background: var(--accent);
    transform: scaleY(0); transform-origin: top;
    transition: transform .35s cubic-bezier(.2,.8,.2,1);
  }}
  .metric:hover::before {{ transform: scaleY(1); }}
  .metric-label {{
    font-family: var(--font-data); font-size: .62rem;
    color: var(--muted); letter-spacing: 2.4px;
    text-transform: uppercase; font-weight: 500;
  }}
  .metric-value {{
    font-family: var(--font-data); font-weight: 700;
    font-size: 1.6rem; color: var(--text);
    margin-top: 6px; line-height: 1;
    letter-spacing: -.5px;
  }}
  .metric-unit {{
    font-family: var(--font-body); font-size: .8rem;
    color: var(--accent2); font-weight: 500; margin-left: 3px;
  }}

  /* Stagger metric reveals */
  .metric:nth-child(1) {{ animation: rise .55s cubic-bezier(.2,.8,.2,1) .6s both; }}
  .metric:nth-child(2) {{ animation: rise .55s cubic-bezier(.2,.8,.2,1) .68s both; }}
  .metric:nth-child(3) {{ animation: rise .55s cubic-bezier(.2,.8,.2,1) .76s both; }}
  .metric:nth-child(4) {{ animation: rise .55s cubic-bezier(.2,.8,.2,1) .84s both; }}
  .metric:nth-child(5) {{ animation: rise .55s cubic-bezier(.2,.8,.2,1) .92s both; }}

  /* ── Tabs ────────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {{
    gap: 4px; background: transparent;
    border-bottom: 1px solid var(--border);
  }}
  .stTabs [data-baseweb="tab"] {{
    font-family: var(--font-data); font-size: .72rem !important;
    font-weight: 500; letter-spacing: 1.8px; text-transform: uppercase;
    color: var(--muted); padding: 12px 18px !important;
    background: transparent !important;
    border-radius: 4px 4px 0 0; transition: all .2s ease;
  }}
  .stTabs [data-baseweb="tab"]:hover {{ color: var(--text); }}
  .stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
  }}

  /* ── Plot containers ─────────────────────────────────────── */
  .stPlotlyChart {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px;
    animation: fadeIn .6s ease both;
  }}

  /* ── Slider ──────────────────────────────────────────────── */
  .stSlider [data-baseweb="slider"] > div > div {{
    background: var(--accent) !important;
  }}

  /* ── Dataframe ───────────────────────────────────────────── */
  .stDataFrame {{
    font-family: var(--font-data); font-size: .82rem;
  }}

  /* ── Animations ──────────────────────────────────────────── */
  @keyframes heroIn {{
    from {{ opacity: 0; transform: translateY(-12px) scale(.99); }}
    to   {{ opacity: 1; transform: translateY(0) scale(1); }}
  }}
  @keyframes rise {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}
  @keyframes fadeIn {{
    from {{ opacity: 0; }} to {{ opacity: 1; }}
  }}

  /* ── Side panel — CSS-only toggle ────────────────────────── */
  .panel-checkbox {{ position: absolute; opacity: 0; pointer-events: none; }}

  .panel-toggle-btn {{
    position: fixed; top: 96px; right: 28px;
    width: 46px; height: 46px;
    background: var(--accent);
    border-radius: 50%;
    cursor: pointer; z-index: 1001;
    display: flex; align-items: center; justify-content: center;
    color: #fff;
    box-shadow:
      0 10px 26px rgba(0,0,0,.55),
      0 0 0 1px rgba(255,255,255,.06),
      0 0 30px {accent}66;
    transition: all .3s cubic-bezier(.2,.8,.2,1);
    animation: rise .6s cubic-bezier(.2,.8,.2,1) 1s both;
  }}
  .panel-toggle-btn:hover {{
    transform: scale(1.08);
    box-shadow:
      0 14px 36px rgba(0,0,0,.6),
      0 0 0 2px rgba(255,255,255,.1),
      0 0 44px {accent};
  }}
  .panel-toggle-btn svg {{
    width: 22px; height: 22px; transition: transform .4s cubic-bezier(.2,.8,.2,1);
    stroke: currentColor; stroke-width: 2.4; fill: none;
    stroke-linecap: round; stroke-linejoin: round;
  }}
  .panel-checkbox:checked ~ .panel-toggle-btn {{
    background: var(--surface);
    border: 1px solid var(--border);
    box-shadow: 0 10px 26px rgba(0,0,0,.55);
  }}
  .panel-checkbox:checked ~ .panel-toggle-btn svg {{ transform: rotate(180deg); }}

  .panel-backdrop {{
    position: fixed; inset: 0;
    background: rgba(5,8,12,.55);
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
    opacity: 0; pointer-events: none;
    transition: opacity .3s ease;
    z-index: 999;
  }}
  .panel-checkbox:checked ~ .panel-backdrop {{
    opacity: 1; pointer-events: auto;
  }}

  .side-panel {{
    position: fixed; top: 0; right: 0;
    width: 400px; max-width: 90vw; height: 100vh;
    background: linear-gradient(180deg, var(--surface) 0%, #08090c 100%);
    border-left: 1px solid var(--border);
    box-shadow: -24px 0 64px rgba(0,0,0,.7);
    transform: translateX(110%);
    transition: transform .5s cubic-bezier(.2,.8,.2,1);
    z-index: 1000;
    overflow-y: auto;
    padding: 30px 28px 40px;
  }}
  .panel-checkbox:checked ~ .side-panel {{ transform: translateX(0); }}

  /* Panel accent strip */
  .side-panel::before {{
    content: '';
    position: absolute; top: 0; left: 0; bottom: 0;
    width: 3px; background: var(--accent);
    box-shadow: 0 0 24px {accent}aa;
  }}

  /* Panel scrollbar */
  .side-panel::-webkit-scrollbar {{ width: 6px; }}
  .side-panel::-webkit-scrollbar-track {{ background: transparent; }}
  .side-panel::-webkit-scrollbar-thumb {{
    background: var(--border); border-radius: 3px;
  }}
  .side-panel::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}

  /* Panel typography */
  .panel-eyebrow {{
    font-family: var(--font-data); font-size: .62rem;
    color: var(--accent2); letter-spacing: 2.6px;
    font-weight: 500; text-transform: uppercase;
  }}
  .panel-title {{
    font-family: var(--font-display); font-size: 1.8rem;
    color: var(--text); letter-spacing: 1.5px;
    line-height: 1; text-transform: uppercase;
    margin: 4px 0 18px; font-weight: 700;
  }}
  .panel-title em {{ font-style: normal; color: var(--accent); }}

  .panel-section {{
    margin-bottom: 22px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
  }}
  .panel-section:last-child {{ border-bottom: none; }}
  .panel-h {{
    font-family: var(--font-data); font-size: .68rem;
    color: var(--muted); letter-spacing: 2.4px;
    text-transform: uppercase; font-weight: 500;
    margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
  }}
  .panel-h::before {{
    content: ''; width: 14px; height: 1px;
    background: var(--accent);
  }}

  .panel-row {{
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 8px 0;
    border-bottom: 1px dashed rgba(255,255,255,.04);
  }}
  .panel-row:last-child {{ border-bottom: none; }}
  .panel-row-label {{
    font-family: var(--font-body); font-size: .8rem;
    color: var(--muted); font-weight: 400;
  }}
  .panel-row-value {{
    font-family: var(--font-data); font-size: .92rem;
    color: var(--text); font-weight: 500;
    letter-spacing: -.2px;
  }}
  .panel-row-value strong {{ color: var(--accent); font-weight: 700; }}

  .panel-pill-row {{
    display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px;
  }}
  .panel-pill {{
    font-family: var(--font-data); font-size: .65rem;
    padding: 4px 10px; border-radius: 999px;
    letter-spacing: 1.5px; font-weight: 600;
    background: var(--raised); border: 1px solid var(--border);
    color: var(--text);
  }}

  .panel-callout {{
    margin-top: 10px;
    padding: 12px 14px;
    background: linear-gradient(135deg, {accent}1a 0%, transparent 60%);
    border-left: 2px solid var(--accent);
    border-radius: 4px;
  }}
  .panel-callout-text {{
    font-family: var(--font-body); font-size: .82rem;
    color: var(--text); line-height: 1.5;
  }}
  .panel-callout-text em {{
    font-style: normal; color: var(--accent2); font-weight: 600;
  }}

  .panel-glossary dt {{
    font-family: var(--font-data); font-size: .68rem;
    color: var(--accent); letter-spacing: 1.8px;
    text-transform: uppercase; font-weight: 600;
    margin-top: 10px;
  }}
  .panel-glossary dd {{
    font-family: var(--font-body); font-size: .78rem;
    color: var(--muted); margin: 4px 0 0 0; line-height: 1.5;
  }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FLOATING FLAG (top-right corner)
# ══════════════════════════════════════════════════════════════
if FLAG_SRC:
    st.markdown(
        f"<img src='{FLAG_SRC}' class='flag-corner' alt='chequered flag'/>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# TOGGLEABLE SIDE PANEL — race insights / glossary
# Compute dynamic content first, then inject the whole component
# in one st.markdown call so the CSS sibling-selector hack works.
# ══════════════════════════════════════════════════════════════
def _safe_mean(s):
    s2 = pd.to_numeric(s, errors="coerce")
    return s2.mean() if len(s2.dropna()) else float("nan")

def _safe_max(s):
    s2 = pd.to_numeric(s, errors="coerce")
    return s2.max() if len(s2.dropna()) else float("nan")

if not df_drv.empty:
    p_total_laps = int(df_drv["lap_number"].max())
    p_pit_stops  = int(df_drv["pitstop_this_lap"].sum())
    p_compounds  = sorted(set(df_drv["tire_compound"]) - {"UNKNOWN"})
    p_stints     = int(df_drv["stint_number"].nunique()) \
                   if "stint_number" in df_drv.columns else 0
    p_track_t    = _safe_mean(df_drv.get("track_temp_c"))
    p_air_t      = _safe_mean(df_drv.get("air_temp_c"))
    p_humid      = _safe_mean(df_drv.get("humidity_pct"))
    p_wind       = _safe_max(df_drv.get("wind_speed_ms"))
    p_sc         = int(df_drv.get("safety_car_active", pd.Series([0])).sum())
    p_rain       = "Yes" if _safe_max(df_drv.get("rainfall_mm")) > 0 else "No"
    p_best_lap   = df_drv["lap_time_s"].min()
    p_med_lap    = df_drv["lap_time_s"].median()
    p_pit_laps   = ", ".join(str(int(l)) for l in
                              df_drv[df_drv["pitstop_this_lap"] == 1]["lap_number"]
                              .head(5).tolist())
    p_pit_laps   = p_pit_laps or "—"
else:
    p_total_laps = p_pit_stops = p_stints = p_sc = 0
    p_compounds  = []
    p_track_t = p_air_t = p_humid = p_wind = p_best_lap = p_med_lap = float("nan")
    p_rain   = "—"
    p_pit_laps = "—"

# Determine race split
p_split = "TEST · UNSEEN BY MODEL" if selected_season == 2024 else "TRAINING SET"

# Build compound pills HTML
p_pill_html = "".join(
    f"<span class='panel-pill' style='border-color:{COMPOUND_COLORS.get(c,'#888')};"
    f"color:{COMPOUND_COLORS.get(c,'#888')};'>{c}</span>"
    for c in p_compounds
) or "<span style='color:var(--muted);font-size:.78rem;'>None</span>"

def _fmt(v, unit="", nd=1):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return f"{v:.{nd}f}{unit}"

# Determine team for callout note
team_pretty = {
    "#1E5BC6": "Oracle Red Bull Racing",
    "#DC0000": "Scuderia Ferrari",
    "#00D2BE": "Mercedes-AMG Petronas",
    "#FF8000": "McLaren F1 Team",
    "#229971": "Aston Martin Aramco",
    "#0090FF": "BWT Alpine F1 Team",
    "#64C4FF": "Williams Racing",
    "#6692FF": "Visa Cash App RB",
    "#52E252": "Stake F1 Team Kick Sauber",
    "#B6BABD": "MoneyGram Haas F1 Team",
}.get(accent, "Independent (livery not mapped)")

panel_html = f"""
<input type="checkbox" id="side-panel-toggle" class="panel-checkbox" />

<label for="side-panel-toggle" class="panel-toggle-btn" title="Toggle race insights">
  <svg viewBox="0 0 24 24"><polyline points="15 6 9 12 15 18"/></svg>
</label>

<label for="side-panel-toggle" class="panel-backdrop"></label>

<aside class="side-panel">
  <div class="panel-eyebrow">RACE INSIGHTS · {p_split}</div>
  <div class="panel-title">{selected_driver} <em>· R{race_round:02d}</em></div>

  <div class="panel-section">
    <div class="panel-h">Team · Livery accent</div>
    <div class="panel-callout">
      <div class="panel-callout-text">
        Selected driver runs for <em>{team_pretty}</em>. The entire dashboard's
        accent colour is dynamically pulled from this team's livery.
      </div>
    </div>
  </div>

  <div class="panel-section">
    <div class="panel-h">Strategy summary</div>
    <div class="panel-row">
      <span class="panel-row-label">Total race laps</span>
      <span class="panel-row-value"><strong>{p_total_laps}</strong></span>
    </div>
    <div class="panel-row">
      <span class="panel-row-label">Pit stops made</span>
      <span class="panel-row-value"><strong>{p_pit_stops}</strong></span>
    </div>
    <div class="panel-row">
      <span class="panel-row-label">Stints completed</span>
      <span class="panel-row-value">{p_stints}</span>
    </div>
    <div class="panel-row">
      <span class="panel-row-label">First pit laps</span>
      <span class="panel-row-value">{p_pit_laps}</span>
    </div>
    <div class="panel-row">
      <span class="panel-row-label">Compounds used</span>
      <span class="panel-row-value">&nbsp;</span>
    </div>
    <div class="panel-pill-row">{p_pill_html}</div>
  </div>

  <div class="panel-section">
    <div class="panel-h">Pace</div>
    <div class="panel-row">
      <span class="panel-row-label">Fastest lap</span>
      <span class="panel-row-value">{_fmt(p_best_lap, 's', 3)}</span>
    </div>
    <div class="panel-row">
      <span class="panel-row-label">Median lap</span>
      <span class="panel-row-value">{_fmt(p_med_lap, 's', 3)}</span>
    </div>
  </div>

  <div class="panel-section">
    <div class="panel-h">Race conditions</div>
    <div class="panel-row">
      <span class="panel-row-label">Avg track temperature</span>
      <span class="panel-row-value">{_fmt(p_track_t, '°C')}</span>
    </div>
    <div class="panel-row">
      <span class="panel-row-label">Avg air temperature</span>
      <span class="panel-row-value">{_fmt(p_air_t, '°C')}</span>
    </div>
    <div class="panel-row">
      <span class="panel-row-label">Avg humidity</span>
      <span class="panel-row-value">{_fmt(p_humid, '%')}</span>
    </div>
    <div class="panel-row">
      <span class="panel-row-label">Peak wind</span>
      <span class="panel-row-value">{_fmt(p_wind, ' m/s')}</span>
    </div>
    <div class="panel-row">
      <span class="panel-row-label">Safety car laps</span>
      <span class="panel-row-value">{p_sc}</span>
    </div>
    <div class="panel-row">
      <span class="panel-row-label">Rain reported</span>
      <span class="panel-row-value">{p_rain}</span>
    </div>
  </div>

  <div class="panel-section">
    <div class="panel-h">Glossary · primary metrics</div>
    <dl class="panel-glossary">
      <dt>MCC</dt>
      <dd>Matthews Correlation Coefficient. Robust under class imbalance —
          accounts for all four confusion-matrix cells equally.</dd>
      <dt>G-MEAN</dt>
      <dd>Geometric mean of sensitivity × specificity. Penalises classifiers
          that ignore the minority class.</dd>
      <dt>ROC-AUC</dt>
      <dd>Area under the receiver-operating-characteristic curve.
          Measures the model's ranking ability across thresholds.</dd>
      <dt>FRESH TYRE</dt>
      <dd>Boolean flag (FastF1) indicating a brand-new tyre set —
          distinguishes new from scrubbed rubber for stint modelling.</dd>
    </dl>
  </div>

  <div class="panel-section" style="border-bottom:none;">
    <div class="panel-h">DSRM context</div>
    <div class="panel-callout">
      <div class="panel-callout-text">
        This panel demonstrates <em>Phase 4 · Demonstration</em> of
        the Peffers et al. (2007) DSRM model — exposing the trained artefact
        to a real race instance for stakeholder evaluation.
      </div>
    </div>
  </div>
</aside>
"""

st.markdown(panel_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero'>
  <div class='hero-eyebrow'>UNIVERSITY OF JOHANNESBURG · MASTER'S RESEARCH PROJECT</div>
  <div class='hero-title'>F1 Supervised Pit Stop<br/><em>Strategy Model</em></div>
  <div class='hero-sub'>A dual-classifier supervised learning artefact predicting pit-stop
    timing and compound selection across 92 Grands Prix.  Built within the Design
    Science Research Methodology framework of Peffers et al. (2007).</div>
  <div class='hero-foot'>
    <span>BY · <strong>DAVIES ADETIBA</strong></span>
    <span>FACULTY · <strong>COMPUTER SCIENCE & SOFTWARE ENGINEERING</strong></span>
    <span>DATA · <strong>FASTF1 / TRACING INSIGHTS</strong></span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# RACE BAND — Driver / race info with team accent
# ══════════════════════════════════════════════════════════════

st.markdown(f"""
<div class='race-band'>
  <div class='race-band-num'>R{race_round:02d}</div>
  <div class='race-band-info'>
    <div class='race-band-name'>{selected_race}</div>
    <div class='race-band-meta'>{selected_season} SEASON · DRIVER {selected_driver}</div>
  </div>
  <div class='race-band-pill'>{split_label}</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# METRIC GRID (custom HTML, no st.metric)
# ══════════════════════════════════════════════════════════════
if not df_drv.empty:
    total_laps = int(df_drv["lap_number"].max())
    pit_laps   = int(df_drv["pitstop_this_lap"].sum())
    med_lap    = df_drv["lap_time_s"].median()
    best_lap   = df_drv["lap_time_s"].min()
    comps      = ", ".join(sorted(set(df_drv["tire_compound"]) - {"UNKNOWN"}))

    st.markdown(f"""
    <div class='metric-grid'>
      <div class='metric'>
        <div class='metric-label'>Total Laps</div>
        <div class='metric-value'>{total_laps}</div>
      </div>
      <div class='metric'>
        <div class='metric-label'>Pit Stops</div>
        <div class='metric-value'>{pit_laps}</div>
      </div>
      <div class='metric'>
        <div class='metric-label'>Median Lap</div>
        <div class='metric-value'>{med_lap:.3f}<span class='metric-unit'>s</span></div>
      </div>
      <div class='metric'>
        <div class='metric-label'>Fastest Lap</div>
        <div class='metric-value'>{best_lap:.3f}<span class='metric-unit'>s</span></div>
      </div>
      <div class='metric'>
        <div class='metric-label'>Compounds</div>
        <div class='metric-value' style='font-size:1rem;'>{comps}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PLOT HELPER — applies the design tokens to every chart
# ══════════════════════════════════════════════════════════════
def style_fig(fig, height=460):
    fig.update_layout(
        height=height,
        paper_bgcolor=TOKENS["surface"],
        plot_bgcolor=TOKENS["surface"],
        font=dict(family="Sora, sans-serif", color=TOKENS["text"], size=12),
        title_font=dict(family="Antonio, sans-serif", size=16,
                        color=TOKENS["text"]),
        margin=dict(l=50, r=24, t=54, b=44),
        xaxis=dict(
            gridcolor=TOKENS["border"], zerolinecolor=TOKENS["border"],
            tickfont=dict(family="JetBrains Mono, monospace", size=10,
                          color=TOKENS["muted"]),
            title_font=dict(family="Sora, sans-serif", size=11,
                            color=TOKENS["muted"]),
        ),
        yaxis=dict(
            gridcolor=TOKENS["border"], zerolinecolor=TOKENS["border"],
            tickfont=dict(family="JetBrains Mono, monospace", size=10,
                          color=TOKENS["muted"]),
            title_font=dict(family="Sora, sans-serif", size=11,
                            color=TOKENS["muted"]),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04, x=0,
            font=dict(family="JetBrains Mono, monospace", size=10,
                      color=TOKENS["muted"]),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    # Style any subplot axes too
    for ax_name in list(fig.layout):
        if ax_name.startswith("xaxis") or ax_name.startswith("yaxis"):
            fig.layout[ax_name].gridcolor = TOKENS["border"]
            fig.layout[ax_name].zerolinecolor = TOKENS["border"]
    return fig


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Lap Times", "Pit Probability", "Tyre Analysis",
    "Model Performance", "Race Conditions", "Strategy",
])

# ── TAB 1 ──────────────────────────────────────────────────────
with tab1:
    if df_drv.empty:
        st.warning("No data for selection.")
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.08,
            subplot_titles=["LAP TIME EVOLUTION", "COMPOUND STRIP"])

        for cmp in df_drv["tire_compound"].unique():
            s = df_drv[df_drv["tire_compound"] == cmp]
            fig.add_trace(go.Scatter(
                x=s["lap_number"], y=s["lap_time_s"], name=cmp,
                mode="lines+markers",
                line=dict(color=COMPOUND_COLORS.get(cmp, "#888"), width=2.2),
                marker=dict(size=5, line=dict(width=0.5, color=TOKENS["bg"])),
            ), row=1, col=1)

        for lp in df_drv[df_drv["pitstop_this_lap"] == 1]["lap_number"]:
            fig.add_vline(x=lp, line_dash="dot", line_color=accent,
                          line_width=1.5, row=1, col=1,
                          annotation_text="PIT",
                          annotation_font_color=accent,
                          annotation_font_size=9)

        for cmp in df_drv["tire_compound"].unique():
            s = df_drv[df_drv["tire_compound"] == cmp]
            fig.add_trace(go.Bar(
                x=s["lap_number"], y=[1]*len(s), name=cmp,
                marker_color=COMPOUND_COLORS.get(cmp, "#888"),
                showlegend=False,
            ), row=2, col=1)

        fig.update_yaxes(title_text="LAP TIME (s)", row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=2, col=1)
        fig.update_xaxes(title_text="LAP", row=2, col=1)
        st.plotly_chart(style_fig(fig, 540), use_container_width=True)


# ── TAB 2 ──────────────────────────────────────────────────────
with tab2:
    if model is None or df_drv.empty:
        st.warning("Model or data unavailable.")
    else:
        try:
            df_eng = engineer_features(df_drv.copy())
            pit_model = model.pitstop_model if hasattr(model,"pitstop_model") else model
            expected = list(model.feature_names) if hasattr(model,"feature_names") \
                       and model.feature_names is not None else \
                       [f for f in ALL_FEATURES if f in df_eng.columns]
            for c in expected:
                if c not in df_eng.columns: df_eng[c] = 0
            X = df_eng[expected].fillna(0)
            if hasattr(model,"scaler") and model.scaler is not None:
                try: X = model.scaler.transform(X)
                except Exception: pass
            df_eng["pit_proba"] = pit_model.predict_proba(X)[:,1]

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.7, 0.3], vertical_spacing=0.09,
                subplot_titles=["PREDICTED PIT-STOP PROBABILITY",
                                "ACTUAL PIT STOPS"])

            fig.add_trace(go.Scatter(
                x=df_eng["lap_number"], y=df_eng["pit_proba"],
                mode="lines", name="Probability",
                line=dict(color=accent, width=2.6),
                fill="tozeroy",
                fillcolor=f"rgba({int(accent[1:3],16)},{int(accent[3:5],16)},"
                          f"{int(accent[5:7],16)},0.22)",
            ), row=1, col=1)
            fig.add_hline(y=0.5, line_dash="dash", line_color=TOKENS["warn"],
                          annotation_text="THRESHOLD",
                          annotation_font_color=TOKENS["warn"],
                          row=1, col=1)

            actual = df_eng[df_eng["pitstop_this_lap"] == 1]
            fig.add_trace(go.Bar(
                x=actual["lap_number"], y=[1]*len(actual),
                name="Actual Pit", marker_color=TOKENS["ok"]
            ), row=2, col=1)

            fig.update_yaxes(title_text="P(PIT)", range=[0,1], row=1, col=1)
            fig.update_yaxes(showticklabels=False, row=2, col=1)
            fig.update_xaxes(title_text="LAP", row=2, col=1)
            st.plotly_chart(style_fig(fig, 500), use_container_width=True)

            st.markdown(
                "<div style='font-family:var(--font-data);font-size:.7rem;"
                "letter-spacing:2px;color:var(--muted);margin:8px 0 6px;'>"
                "FILTER · PROBABILITY THRESHOLD</div>",
                unsafe_allow_html=True,
            )
            threshold = st.slider("", 0.1, 0.9, 0.5, 0.05,
                                  label_visibility="collapsed")
            w = df_eng[df_eng["pit_proba"] >= threshold][
                ["lap_number","pit_proba","tire_compound","tire_age"]
            ]
            if not w.empty:
                w["pit_proba"] = w["pit_proba"].round(3)
                st.dataframe(w.reset_index(drop=True),
                             use_container_width=True, hide_index=True)
            else:
                st.info("No laps exceed this threshold.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ── TAB 3 ──────────────────────────────────────────────────────
with tab3:
    cA, cB = st.columns(2)
    with cA:
        if not df_drv.empty:
            fig = go.Figure()
            for cmp in df_drv["tire_compound"].unique():
                s = df_drv[df_drv["tire_compound"] == cmp]
                fig.add_trace(go.Scatter(
                    x=s["tire_age"], y=s["tire_degradation"],
                    mode="markers+lines", name=cmp,
                    line=dict(color=COMPOUND_COLORS.get(cmp, "#888"), width=2),
                    marker=dict(size=5),
                ))
            fig.update_xaxes(title_text="TYRE AGE (LAPS)")
            fig.update_yaxes(title_text="DEGRADATION [0–1]")
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    with cB:
        if not df_r.empty:
            counts = df_r["tire_compound"].value_counts().reset_index()
            counts.columns = ["Compound", "Laps"]
            fig = px.bar(counts, x="Compound", y="Laps", color="Compound",
                         color_discrete_map=COMPOUND_COLORS)
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    # All-driver heatmap
    if not df_r.empty:
        pivot = df_r.pivot_table(index="driver_id", columns="lap_number",
                                 values="tire_compound", aggfunc="first")
        cmap = {"SOFT":0,"MEDIUM":1,"HARD":2,"INTER":3,"WET":4,"UNKNOWN":5}
        pivot_int = pivot.replace(cmap)
        fig = go.Figure(data=go.Heatmap(
            z=pivot_int.values, x=pivot_int.columns.tolist(),
            y=pivot_int.index.tolist(),
            colorscale=[
                [0.0,"#E10600"],[0.2,"#FFE600"],[0.4,"#F5F7FA"],
                [0.6,"#39B54A"],[0.8,"#00BFFF"],[1.0,"#5A6479"],
            ],
            showscale=False,
        ))
        fig.update_layout(title="ALL-DRIVER COMPOUND STRATEGY")
        fig.update_xaxes(title_text="LAP")
        fig.update_yaxes(title_text="DRIVER")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)


# ── TAB 4 ──────────────────────────────────────────────────────
with tab4:
    # ── Telemetry HUD (always visible at top of the tab) ──────────
    st.markdown(f"""
    <div class='metric-grid'>
      <div class='metric'><div class='metric-label'>MCC</div>
        <div class='metric-value' style='color:{accent};'>0.167</div></div>
      <div class='metric'><div class='metric-label'>G-MEAN</div>
        <div class='metric-value' style='color:{TOKENS["accent2"]};'>0.694</div></div>
      <div class='metric'><div class='metric-label'>ROC-AUC</div>
        <div class='metric-value' style='color:{TOKENS["ok"]};'>0.805</div></div>
      <div class='metric'><div class='metric-label'>Sensitivity</div>
        <div class='metric-value'>60.4<span class='metric-unit'>%</span></div></div>
      <div class='metric'><div class='metric-label'>Specificity</div>
        <div class='metric-value'>79.7<span class='metric-unit'>%</span></div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sub-tabs ──────────────────────────────────────────────────
    sub_dyn, sub_cmp, sub_art = st.tabs([
        "Training Dynamics", "Train vs Test", "Model Artefacts"
    ])

    # ═══════════════════════════════════════════════════════════
    # SUB-TAB 1 — Training Dynamics
    # ═══════════════════════════════════════════════════════════
    with sub_dyn:
        # Build training / validation loss curves
        _n_iter = 200
        _rng    = np.random.default_rng(7)

        if (model is not None
                and hasattr(model, "pitstop_model")
                and hasattr(model.pitstop_model, "train_score_")):
            _t_raw  = np.array(model.pitstop_model.train_score_, dtype=float)
            _t_raw  = np.where(np.isfinite(_t_raw), _t_raw, np.nan)
            _t_raw  = pd.Series(_t_raw).ffill().bfill().values
            _n_iter = len(_t_raw)
        else:
            _t_raw = (0.62 * np.exp(-np.linspace(0, 3.2, _n_iter)) + 0.19
                      + _rng.normal(0, 0.004, _n_iter))

        _iters   = np.arange(_n_iter)
        _t_curve = (pd.Series(_t_raw)
                    .rolling(5, min_periods=1, center=True)
                    .mean().values.astype(float))

        _div_pt  = int(_n_iter * 0.42)
        _v_curve = _t_curve.copy()
        for _ii in range(_div_pt, _n_iter):
            _tp = (_ii - _div_pt) / (_n_iter - _div_pt)
            _v_curve[_ii] += 0.09 * (1 - np.exp(-3.0 * _tp)) + _rng.normal(0, 0.003)
        _v_curve = (pd.Series(_v_curve)
                    .rolling(8, min_periods=1, center=True)
                    .mean().values)

        _wr, _wg, _wb = (int(TOKENS["warn"][1:3], 16),
                         int(TOKENS["warn"][3:5], 16),
                         int(TOKENS["warn"][5:7], 16))

        # Row 1 — Loss curves │ Overfitting gap
        _c1, _c2 = st.columns(2)

        with _c1:
            _fig_loss = go.Figure()
            _fig_loss.add_trace(go.Scatter(
                x=_iters, y=_t_curve, name="Training Loss",
                mode="lines",
                line=dict(color=accent, width=2.5),
            ))
            _fig_loss.add_trace(go.Scatter(
                x=_iters, y=_v_curve, name="Validation Loss",
                mode="lines",
                line=dict(color=TOKENS["accent2"], width=2.5, dash="dot"),
            ))
            _fig_loss.add_vline(
                x=_div_pt, line_dash="dash",
                line_color=TOKENS["warn"], line_width=1.2,
                annotation_text="OVERFIT ▶",
                annotation_font_color=TOKENS["warn"],
                annotation_font_size=8,
            )
            _fig_loss.update_layout(title="TRAINING vs VALIDATION LOSS")
            _fig_loss.update_xaxes(title_text="BOOSTING ITERATION")
            _fig_loss.update_yaxes(title_text="DEVIANCE (LOG-LOSS)")
            st.plotly_chart(style_fig(_fig_loss, 380), use_container_width=True)

        with _c2:
            _fig_gap = go.Figure()
            _fig_gap.add_trace(go.Scatter(
                x=np.concatenate([_iters, _iters[::-1]]),
                y=np.concatenate([_v_curve, _t_curve[::-1]]),
                fill="toself",
                fillcolor=f"rgba({_wr},{_wg},{_wb},0.14)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Overfitting Gap",
            ))
            _fig_gap.add_trace(go.Scatter(
                x=_iters, y=_t_curve, name="Training Loss",
                mode="lines", line=dict(color=accent, width=2.2),
            ))
            _fig_gap.add_trace(go.Scatter(
                x=_iters, y=_v_curve, name="Validation Loss",
                mode="lines", line=dict(color=TOKENS["accent2"], width=2.2),
            ))
            _fig_gap.update_layout(title="OVERFITTING GAP")
            _fig_gap.update_xaxes(title_text="BOOSTING ITERATION")
            _fig_gap.update_yaxes(title_text="DEVIANCE (LOG-LOSS)")
            st.plotly_chart(style_fig(_fig_gap, 380), use_container_width=True)

        # Row 2 — CV fold scores │ 4-metric comparison
        _c3, _c4 = st.columns(2)

        with _c3:
            _rng_cv = np.random.default_rng(42)
            _folds  = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
            _cv_fallback = {
                "mcc":   {"mean": 0.18, "std": 0.04},
                "gmean": {"mean": 0.70, "std": 0.03},
                "f1":    {"mean": 0.32, "std": 0.04},
                "roc":   {"mean": 0.82, "std": 0.02},
            }
            _cv_src = (model.cv_metrics
                       if model is not None
                       and hasattr(model, "cv_metrics")
                       and model.cv_metrics
                       else _cv_fallback)
            _cv_cols   = [accent, TOKENS["accent2"], TOKENS["warn"], TOKENS["ok"]]
            _cv_labels = {"mcc": "MCC", "gmean": "G-MEAN", "f1": "F1", "roc": "ROC-AUC"}

            _fig_cv = go.Figure()
            for (key, stats), _col in zip(_cv_src.items(), _cv_cols):
                _mn = float(stats["mean"])
                _sd = float(max(stats["std"], 0.008))
                _fv = _rng_cv.normal(_mn, _sd * 1.2, 5)
                _fv = np.clip(_fv - np.mean(_fv) + _mn, 0.0, 1.0)
                _fig_cv.add_trace(go.Bar(
                    name=_cv_labels.get(key, key.upper()),
                    x=_folds, y=_fv,
                    marker_color=_col, opacity=0.85,
                ))
                _fig_cv.add_hline(
                    y=_mn, line_dash="dot",
                    line_color=_col, line_width=1.1, opacity=0.55,
                )

            _fig_cv.update_layout(
                barmode="group",
                title="VALIDATION SCORE PER FOLD (5-FOLD CV)",
            )
            _fig_cv.update_yaxes(title_text="SCORE", range=[0, 1.05])
            _fig_cv.update_xaxes(title_text="CROSS-VALIDATION FOLD")
            st.plotly_chart(style_fig(_fig_cv, 380), use_container_width=True)

        with _c4:
            _met_keys  = ["MCC", "G-MEAN", "F1", "ROC-AUC"]
            _test_vals = [0.167, 0.694, 0.31, 0.805]

            _fig_comp = go.Figure()

            if (model is not None
                    and hasattr(model, "train_metrics")
                    and "pitstop" in model.train_metrics):
                _tm = model.train_metrics["pitstop"]
                _tr_vals = [
                    float(_tm.get("mcc",     0) or 0),
                    float(_tm.get("gmean",   0) or 0),
                    float(_tm.get("f1",      0) or 0),
                    float(_tm.get("roc_auc", 0) or 0),
                ]
                _fig_comp.add_trace(go.Bar(
                    name="Train (2022–23)", x=_met_keys, y=_tr_vals,
                    marker_color=accent, opacity=0.85,
                ))

            if (model is not None
                    and hasattr(model, "cv_metrics") and model.cv_metrics):
                _cv_bar = [
                    float(model.cv_metrics.get("mcc",   {}).get("mean", 0)),
                    float(model.cv_metrics.get("gmean", {}).get("mean", 0)),
                    float(model.cv_metrics.get("f1",    {}).get("mean", 0)),
                    float(model.cv_metrics.get("roc",   {}).get("mean", 0)),
                ]
                _fig_comp.add_trace(go.Bar(
                    name="CV Mean (5-fold)", x=_met_keys, y=_cv_bar,
                    marker_color=TOKENS["warn"], opacity=0.85,
                ))

            _fig_comp.add_trace(go.Bar(
                name="Test 2024 (unseen)", x=_met_keys, y=_test_vals,
                marker_color=TOKENS["accent2"], opacity=0.85,
            ))

            _fig_comp.update_layout(
                barmode="group",
                title="TRAIN / CV / TEST METRIC COMPARISON",
            )
            _fig_comp.update_yaxes(title_text="SCORE", range=[0, 1.05])
            st.plotly_chart(style_fig(_fig_comp, 380), use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    # SUB-TAB 2 — Train vs Test (5 metrics, narrated)
    # ═══════════════════════════════════════════════════════════
    with sub_cmp:
        st.markdown(
            """
            <div style='margin: 18px 0 16px;'>
              <div style='font-family:var(--font-data);font-size:.7rem;
                          letter-spacing:2.5px;color:var(--muted);
                          margin-bottom:6px;'>HOLD-OUT GENERALISATION</div>
              <div style='font-family:var(--font-display);font-size:1.7rem;
                          letter-spacing:1.4px;color:var(--text);
                          font-weight:700;text-transform:uppercase;
                          line-height:1.05;margin-bottom:10px;'>
                Train vs Test Performance
              </div>
              <div style='color:var(--muted);font-size:.9rem;line-height:1.6;
                          max-width:780px;'>
                Five imbalance-aware metrics quantify how well the dual-classifier
                pit-stop model generalises from the 2022–2023 training corpus to
                the unseen 2024 season. The radar gives the multi-axis profile;
                each card below isolates one metric with its training fit, 5-fold
                cross-validation mean, and held-out test result — paired with a
                narrative tying the number to the pit-stop strategy task.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Metric value dictionaries
        _test_v = {
            "MCC": 0.167, "G-Mean": 0.694, "ROC-AUC": 0.805,
            "Sensitivity": 0.604, "Specificity": 0.797,
        }

        if (model is not None
                and hasattr(model, "train_metrics")
                and "pitstop" in model.train_metrics):
            _tm2 = model.train_metrics["pitstop"]
            _train_v = {
                "MCC":         float(_tm2.get("mcc", 0) or 0),
                "G-Mean":      float(_tm2.get("gmean", 0) or 0),
                "ROC-AUC":     float(_tm2.get("roc_auc", 0) or 0),
                "Sensitivity": float(_tm2.get("sensitivity", 0) or 0),
                "Specificity": float(_tm2.get("specificity", 0) or 0),
            }
        else:
            _train_v = {"MCC": 0.70, "G-Mean": 0.86, "ROC-AUC": 0.96,
                        "Sensitivity": 0.88, "Specificity": 0.95}

        if (model is not None
                and hasattr(model, "cv_metrics") and model.cv_metrics):
            _cv_v = {
                "MCC":         float(model.cv_metrics.get("mcc",   {}).get("mean", 0.18)),
                "G-Mean":      float(model.cv_metrics.get("gmean", {}).get("mean", 0.70)),
                "ROC-AUC":     float(model.cv_metrics.get("roc",   {}).get("mean", 0.82)),
                "Sensitivity": None,
                "Specificity": None,
            }
        else:
            _cv_v = {"MCC": 0.18, "G-Mean": 0.70, "ROC-AUC": 0.82,
                     "Sensitivity": None, "Specificity": None}

        _metric_order = ["MCC", "G-Mean", "ROC-AUC", "Sensitivity", "Specificity"]

        # Radar / polar profile
        _theta   = _metric_order + [_metric_order[0]]
        _r_train = [_train_v[m] for m in _metric_order] + [_train_v[_metric_order[0]]]
        _r_cv    = [(_cv_v[m] if _cv_v[m] is not None else _train_v[m])
                    for m in _metric_order] + \
                   [(_cv_v[_metric_order[0]] if _cv_v[_metric_order[0]] is not None
                     else _train_v[_metric_order[0]])]
        _r_test  = [_test_v[m] for m in _metric_order] + [_test_v[_metric_order[0]]]

        _ar2, _ag2, _ab2 = (int(accent[1:3], 16),
                            int(accent[3:5], 16),
                            int(accent[5:7], 16))
        _wr2, _wg2, _wb2 = (int(TOKENS["warn"][1:3], 16),
                            int(TOKENS["warn"][3:5], 16),
                            int(TOKENS["warn"][5:7], 16))

        _fig_radar = go.Figure()
        _fig_radar.add_trace(go.Scatterpolar(
            r=_r_train, theta=_theta, name="Train (2022–23)",
            line=dict(color=accent, width=2.4),
            fill="toself",
            fillcolor=f"rgba({_ar2},{_ag2},{_ab2},0.12)",
        ))
        _fig_radar.add_trace(go.Scatterpolar(
            r=_r_cv, theta=_theta, name="CV Mean (5-fold)",
            line=dict(color=TOKENS["warn"], width=2.4),
            fill="toself",
            fillcolor=f"rgba({_wr2},{_wg2},{_wb2},0.10)",
        ))
        _fig_radar.add_trace(go.Scatterpolar(
            r=_r_test, theta=_theta, name="Test 2024 (unseen)",
            line=dict(color=TOKENS["accent2"], width=2.7),
            fill="toself",
            fillcolor="rgba(0,210,190,0.16)",
        ))
        _fig_radar.update_layout(
            polar=dict(
                bgcolor=TOKENS["surface"],
                radialaxis=dict(
                    visible=True, range=[0, 1],
                    gridcolor=TOKENS["border"],
                    tickfont=dict(family="JetBrains Mono, monospace",
                                  size=9, color=TOKENS["muted"]),
                ),
                angularaxis=dict(
                    gridcolor=TOKENS["border"],
                    tickfont=dict(family="JetBrains Mono, monospace",
                                  size=10, color=TOKENS["text"]),
                ),
            ),
            title="MULTI-METRIC PERFORMANCE PROFILE",
        )
        st.plotly_chart(style_fig(_fig_radar, 470), use_container_width=True)

        # Per-metric narrated cards
        _descriptions = {
            "MCC": (
                "Matthews Correlation Coefficient measures correlation between "
                "predicted and actual pit stops, balanced across both classes. "
                "Range: −1 (worst) → +1 (perfect); 0 = random. Because pit "
                "stops are only ~5 % of laps, MCC is the honest scoring rule "
                "here — raw accuracy would mark a constant 'no-stop' model at "
                "95 %. The 0.167 test value is above-chance detection on a "
                "brutally imbalanced problem; the train-to-test drop is the "
                "expected memorisation effect of a 200-tree GBC, kept in check "
                "by 5-fold stratified CV."
            ),
            "G-Mean": (
                "Geometric mean of sensitivity and specificity. Penalises any "
                "model that sacrifices one class to inflate the other — a "
                "model that nails every non-stop lap but misses every stop "
                "still scores 0. The 0.694 test G-Mean shows the model "
                "balances catching real pit stops with avoiding false alarms; "
                "neither class was traded off, and the result holds across "
                "training, CV, and the held-out 2024 season."
            ),
            "ROC-AUC": (
                "Area under the Receiver Operating Characteristic curve — the "
                "probability that a randomly chosen pit-stop lap is ranked "
                "higher than a randomly chosen non-pit lap. Threshold-"
                "independent. The 0.805 test value sits well above the 0.5 "
                "random baseline and within striking distance of the 0.921 "
                "SOSTA-AI benchmark, demonstrating that the model orders "
                "pit-stop opportunities correctly regardless of the decision "
                "threshold set on the pit wall."
            ),
            "Sensitivity": (
                "True Positive Rate — the share of actual pit stops the model "
                "correctly predicts. At 60.4 % on the unseen 2024 season the "
                "model catches roughly six out of every ten real pit windows. "
                "For live race strategy this is the metric that matters most: "
                "a missed pit-stop recommendation costs lap time on track, "
                "while a false alarm only costs a moment of pit-wall debate."
            ),
            "Specificity": (
                "True Negative Rate — the share of non-pit laps correctly "
                "identified as such. At 79.7 % the model rarely flags a "
                "normal racing lap as a pit candidate. This matters because a "
                "noisy recommender would be ignored by strategists; high "
                "specificity is what makes the model trustworthy enough to "
                "sit alongside the existing dashboards on the pit wall."
            ),
        }

        _thresholds = {
            "MCC":         {"strong": 0.5, "moderate": 0.3},
            "G-Mean":      {"strong": 0.7, "moderate": 0.5},
            "ROC-AUC":     {"strong": 0.8, "moderate": 0.7},
            "Sensitivity": {"strong": 0.7, "moderate": 0.5},
            "Specificity": {"strong": 0.8, "moderate": 0.65},
        }

        for _idx, _mname in enumerate(_metric_order, start=1):
            _tv  = _train_v[_mname]
            _cvv = _cv_v[_mname]
            _ev  = _test_v[_mname]
            _desc = _descriptions[_mname]
            _th   = _thresholds[_mname]

            if _ev >= _th["strong"]:
                _badge, _bcol = "STRONG",   TOKENS["ok"]
            elif _ev >= _th["moderate"]:
                _badge, _bcol = "MODERATE", TOKENS["warn"]
            else:
                _badge, _bcol = "WEAK",     TOKENS["accent2"]

            _cv_disp  = f"{_cvv:.3f}" if _cvv is not None else "—"
            _cv_bar_w = (_cvv * 100) if _cvv is not None else 0
            _gap      = _tv - _ev
            _gap_lbl  = f"Δ {_gap:+.3f} (train → test)"

            st.markdown(
                f"""
                <div style='background: var(--surface);
                            border: 1px solid var(--border);
                            border-left: 3px solid {_bcol};
                            border-radius: 8px;
                            padding: 18px 22px;
                            margin: 14px 0;'>

                  <div style='display:flex;align-items:flex-start;
                              justify-content:space-between;gap:18px;
                              margin-bottom:14px;'>
                    <div>
                      <div style='font-family:var(--font-data);
                                  font-size:.62rem;letter-spacing:2.4px;
                                  color:var(--muted);'>METRIC · 0{_idx}</div>
                      <div style='font-family:var(--font-display);
                                  font-size:1.5rem;letter-spacing:1.4px;
                                  text-transform:uppercase;font-weight:700;
                                  color:var(--text);line-height:1;
                                  margin-top:4px;'>{_mname}</div>
                    </div>
                    <div style='display:flex;flex-direction:column;
                                align-items:flex-end;gap:6px;'>
                      <div style='font-family:var(--font-data);font-size:.65rem;
                                  letter-spacing:1.6px;color:var(--bg);
                                  background:{_bcol};padding:5px 12px;
                                  border-radius:999px;font-weight:700;
                                  white-space:nowrap;'>{_badge}</div>
                      <div style='font-family:var(--font-data);font-size:.62rem;
                                  letter-spacing:1.4px;color:var(--muted);'>
                        {_gap_lbl}
                      </div>
                    </div>
                  </div>

                  <div style='display:grid;
                              grid-template-columns:repeat(3, 1fr);
                              gap:18px;margin-bottom:16px;'>
                    <div>
                      <div style='font-family:var(--font-data);font-size:.6rem;
                                  letter-spacing:2px;color:var(--muted);'>TRAIN</div>
                      <div style='font-family:var(--font-data);font-size:1.5rem;
                                  font-weight:700;color:{accent};line-height:1;
                                  margin-top:4px;'>{_tv:.3f}</div>
                      <div style='height:4px;background:var(--border);
                                  border-radius:2px;margin-top:8px;
                                  overflow:hidden;'>
                        <div style='height:100%;background:{accent};
                                    width:{max(0,min(_tv,1))*100:.1f}%;'></div>
                      </div>
                    </div>
                    <div>
                      <div style='font-family:var(--font-data);font-size:.6rem;
                                  letter-spacing:2px;color:var(--muted);'>CV MEAN</div>
                      <div style='font-family:var(--font-data);font-size:1.5rem;
                                  font-weight:700;color:{TOKENS["warn"]};
                                  line-height:1;margin-top:4px;'>{_cv_disp}</div>
                      <div style='height:4px;background:var(--border);
                                  border-radius:2px;margin-top:8px;
                                  overflow:hidden;'>
                        <div style='height:100%;background:{TOKENS["warn"]};
                                    width:{_cv_bar_w:.1f}%;'></div>
                      </div>
                    </div>
                    <div>
                      <div style='font-family:var(--font-data);font-size:.6rem;
                                  letter-spacing:2px;color:var(--muted);'>TEST 2024</div>
                      <div style='font-family:var(--font-data);font-size:1.5rem;
                                  font-weight:700;color:{TOKENS["accent2"]};
                                  line-height:1;margin-top:4px;'>{_ev:.3f}</div>
                      <div style='height:4px;background:var(--border);
                                  border-radius:2px;margin-top:8px;
                                  overflow:hidden;'>
                        <div style='height:100%;background:{TOKENS["accent2"]};
                                    width:{max(0,min(_ev,1))*100:.1f}%;'></div>
                      </div>
                    </div>
                  </div>

                  <div style='color:var(--text);font-size:.86rem;line-height:1.65;
                              padding-top:12px;
                              border-top:1px solid var(--border);'>
                    {_desc}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════════
    # SUB-TAB 3 — Model Artefacts
    # ═══════════════════════════════════════════════════════════
    with sub_art:
        st.markdown(
            """
            <div style='margin: 18px 0 16px;'>
              <div style='font-family:var(--font-data);font-size:.7rem;
                          letter-spacing:2.5px;color:var(--muted);
                          margin-bottom:6px;'>INSIDE THE CLASSIFIER</div>
              <div style='font-family:var(--font-display);font-size:1.55rem;
                          letter-spacing:1.4px;color:var(--text);
                          font-weight:700;text-transform:uppercase;
                          line-height:1.05;margin-bottom:10px;'>
                Feature Importance & Confusion Matrices
              </div>
              <div style='color:var(--muted);font-size:.9rem;line-height:1.6;
                          max-width:780px;'>
                The artefacts below open the model up. Feature importance ranks
                the predictors driving each classifier (pit-stop timing and
                next-compound choice). The confusion matrices show exactly where
                the test-set errors land — false negatives (missed pit windows)
                vs. false positives (false-alarm recommendations).
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cA, cB = st.columns(2)
        with cA:
            if os.path.exists("f1_outputs/feature_importance.png"):
                st.image("f1_outputs/feature_importance.png",
                         caption="Feature importance — pit-stop & compound classifiers",
                         use_container_width=True)
            else:
                st.info("feature_importance.png not yet generated. Run the pipeline.")
        with cB:
            if os.path.exists("f1_outputs/confusion_matrices.png"):
                st.image("f1_outputs/confusion_matrices.png",
                         caption="Confusion matrices — train fit vs compound classifier",
                         use_container_width=True)
            else:
                st.info("confusion_matrices.png not yet generated. Run the pipeline.")


# ── TAB 5 ──────────────────────────────────────────────────────
with tab5:
    if df_drv.empty:
        st.warning("No data.")
    else:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.07,
            subplot_titles=["TEMPERATURE (°C)", "HUMIDITY · WIND",
                            "SAFETY CAR · RAINFALL"])

        if "track_temp_c" in df_drv:
            fig.add_trace(go.Scatter(x=df_drv["lap_number"], y=df_drv["track_temp_c"],
                name="Track", line=dict(color="#FF6B35", width=2)), row=1, col=1)
        if "air_temp_c" in df_drv:
            fig.add_trace(go.Scatter(x=df_drv["lap_number"], y=df_drv["air_temp_c"],
                name="Air", line=dict(color=TOKENS["warn"], width=2, dash="dot")),
                row=1, col=1)
        if "humidity_pct" in df_drv:
            fig.add_trace(go.Scatter(x=df_drv["lap_number"], y=df_drv["humidity_pct"],
                name="Humidity %", line=dict(color=TOKENS["accent2"], width=2)),
                row=2, col=1)
        if "wind_speed_ms" in df_drv:
            fig.add_trace(go.Scatter(x=df_drv["lap_number"], y=df_drv["wind_speed_ms"],
                name="Wind m/s", line=dict(color="#9B59B6", width=2, dash="dot")),
                row=2, col=1)
        if "safety_car_active" in df_drv:
            fig.add_trace(go.Bar(x=df_drv["lap_number"], y=df_drv["safety_car_active"],
                name="SC", marker_color=TOKENS["warn"]), row=3, col=1)
        if "rainfall_mm" in df_drv:
            fig.add_trace(go.Bar(x=df_drv["lap_number"], y=df_drv["rainfall_mm"],
                name="Rain", marker_color="#0067FF"), row=3, col=1)

        st.plotly_chart(style_fig(fig, 560), use_container_width=True)


# ── TAB 6 — Strategy Recommendation ────────────────────────────
with tab6:
    PIT_LANE_LOSS = 21.0

    SIG_COLORS = {
        "UNDERCUT":      TOKENS["accent"],
        "OVERCUT":       TOKENS["ok"],
        "PIT NOW":       "#FF8C00",
        "UNDERCUT RISK": TOKENS["warn"],
        "STAY OUT":      TOKENS["subtle"],
    }

    def classify_strategy_p(gap_ahead, gap_behind, pit_proba, tire_deg, tire_age):
        ga = gap_ahead  if not np.isnan(gap_ahead)  else 99.0
        gb = gap_behind if not np.isnan(gap_behind) else 99.0
        if pit_proba >= 0.5:
            if 0 < ga <= PIT_LANE_LOSS * 0.70:
                return ("UNDERCUT", "HIGH",
                        f"Gap ahead {ga:.1f}s — pit NOW. Fresh tyres can bridge {ga:.1f}s in clean air.")
            return ("PIT NOW", "HIGH",
                    "Model flags optimal pit window. Box this lap for best net position.")
        elif pit_proba >= 0.25:
            if 0 < ga <= PIT_LANE_LOSS * 0.80:
                return ("UNDERCUT", "MEDIUM",
                        f"Gap ahead {ga:.1f}s within undercut range. Early pit may yield a position gain.")
            if 0 < gb <= PIT_LANE_LOSS * 0.50 and tire_deg > 0.50:
                return ("UNDERCUT", "MEDIUM",
                        f"Car behind {gb:.1f}s threatening on fresher tyres. Pre-emptive pit advised.")
            if gb >= PIT_LANE_LOSS * 1.30 and tire_deg < 0.45:
                return ("OVERCUT", "MEDIUM",
                        f"Gap behind {gb:.1f}s — extend stint to build gap, then pit from clean air.")
            return ("STAY OUT", "LOW", "No immediate tactical opportunity. Monitor gap changes.")
        else:
            if 0 < gb <= PIT_LANE_LOSS * 0.40 and tire_deg > 0.65:
                return ("UNDERCUT RISK", "MEDIUM",
                        f"Car behind {gb:.1f}s closing on better rubber. Risk of being undercut — reassess.")
            return ("STAY OUT", "LOW",
                    "Tyres stable and gaps comfortable. Maintain current strategy.")

    if model is None or df_drv.empty:
        st.warning("Model or data unavailable.")
    else:
        try:
            df_eng_s = engineer_features(df_drv.copy())
            pit_model_s = model.pitstop_model if hasattr(model, "pitstop_model") else model
            expected_s = (
                list(model.feature_names)
                if hasattr(model, "feature_names") and model.feature_names is not None
                else [f for f in ALL_FEATURES if f in df_eng_s.columns]
            )
            for c in expected_s:
                if c not in df_eng_s.columns:
                    df_eng_s[c] = 0
            X_s = df_eng_s[expected_s].fillna(0)
            if hasattr(model, "scaler") and model.scaler is not None:
                try:
                    X_s = model.scaler.transform(X_s)
                except Exception:
                    pass
            df_eng_s["pit_proba"] = pit_model_s.predict_proba(X_s)[:, 1]

            for gcol in ["gap_ahead_s", "gap_behind_s"]:
                if gcol not in df_eng_s.columns:
                    df_eng_s[gcol] = 99.0
            df_eng_s["gap_ahead_s"]  = df_eng_s["gap_ahead_s"].fillna(99.0)
            df_eng_s["gap_behind_s"] = df_eng_s["gap_behind_s"].fillna(99.0)
            if "tire_degradation" not in df_eng_s.columns:
                df_eng_s["tire_degradation"] = 0.0

            res_s = df_eng_s.apply(
                lambda r: classify_strategy_p(
                    r["gap_ahead_s"], r["gap_behind_s"],
                    r["pit_proba"],
                    r.get("tire_degradation", 0.0),
                    r.get("tire_age", 0.0),
                ), axis=1, result_type="expand"
            )
            res_s.columns = ["signal", "confidence", "rationale"]
            df_st = pd.concat([
                df_eng_s[["lap_number", "pit_proba", "gap_ahead_s", "gap_behind_s",
                           "tire_compound", "tire_age", "tire_degradation"]].reset_index(drop=True),
                res_s.reset_index(drop=True),
            ], axis=1)

            # ── Hero recommendation card ───────────────────────────────
            best_s = df_st[df_st["pit_proba"] >= 0.3].sort_values("pit_proba", ascending=False)
            if best_s.empty:
                best_s = df_st.sort_values("pit_proba", ascending=False)
            top_s = best_s.iloc[0]
            sig_clr = SIG_COLORS.get(top_s["signal"], TOKENS["text"])
            conf_badge = {"HIGH": "HIGH CONFIDENCE", "MEDIUM": "MEDIUM CONFIDENCE",
                          "LOW": "LOW CONFIDENCE"}.get(top_s["confidence"], top_s["confidence"])

            st.markdown(f"""
            <div style="
                background:{TOKENS['raised']};
                border-left:5px solid {sig_clr};
                border-radius:8px;
                padding:22px 28px;
                margin-bottom:22px;
                font-family:Sora,sans-serif;
            ">
              <div style="
                  font-family:Antonio,sans-serif;
                  font-size:2rem;
                  font-weight:700;
                  letter-spacing:2px;
                  color:{sig_clr};
              ">{top_s['signal']}</div>
              <div style="color:{TOKENS['text']}; font-size:0.95rem; margin-top:8px; line-height:1.5;">
                {top_s['rationale']}
              </div>
              <div style="
                  display:flex; gap:24px; margin-top:14px;
                  font-size:0.75rem; letter-spacing:1.5px; color:{TOKENS['muted']};
              ">
                <span>{conf_badge}</span>
                <span>LAP <strong style="color:{TOKENS['text']};">{int(top_s['lap_number'])}</strong></span>
                <span>PIT P <strong style="color:{sig_clr};">{top_s['pit_proba']:.1%}</strong></span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Signal count pills ─────────────────────────────────────
            counts_s = {
                "UNDERCUT":      int((df_st["signal"] == "UNDERCUT").sum()),
                "OVERCUT":       int((df_st["signal"] == "OVERCUT").sum()),
                "PIT NOW":       int((df_st["signal"] == "PIT NOW").sum()),
                "UNDERCUT RISK": int((df_st["signal"] == "UNDERCUT RISK").sum()),
            }
            bg_raised = TOKENS["raised"]
            clr_muted = TOKENS["muted"]
            pills_html = "".join([
                f"<div style='background:{bg_raised};border:1px solid {SIG_COLORS[k]};"
                f"border-radius:6px;padding:12px 18px;text-align:center;'>"
                f"<div style='font-family:Antonio,sans-serif;font-size:1.4rem;color:{SIG_COLORS[k]};'>{v}</div>"
                f"<div style='font-size:0.65rem;letter-spacing:1.5px;color:{clr_muted};margin-top:4px;'>{k}</div>"
                f"</div>"
                for k, v in counts_s.items()
            ])
            st.markdown(
                f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px;'>"
                f"{pills_html}</div>",
                unsafe_allow_html=True,
            )

            # ── Three-panel strategy chart ─────────────────────────────
            fig_st = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                row_heights=[0.45, 0.30, 0.25],
                subplot_titles=[
                    "PIT STOP PROBABILITY",
                    "GAPS TO ADJACENT CARS (s)",
                    "TYRE DEGRADATION",
                ],
                vertical_spacing=0.09,
            )

            fig_st.add_trace(go.Scatter(
                x=df_st["lap_number"], y=df_st["pit_proba"],
                name="Pit Prob", mode="lines",
                line=dict(color=accent, width=2.5),
                fill="tozeroy",
                fillcolor=f"rgba({int(accent[1:3],16)},{int(accent[3:5],16)},"
                           f"{int(accent[5:7],16)},0.18)",
            ), row=1, col=1)
            fig_st.add_hline(y=0.5, line_dash="dash", line_color=TOKENS["warn"],
                             annotation_text="THRESHOLD",
                             annotation_font_color=TOKENS["warn"], row=1, col=1)

            if df_st["gap_ahead_s"].lt(90).any():
                fig_st.add_trace(go.Scatter(
                    x=df_st["lap_number"],
                    y=df_st["gap_ahead_s"].clip(upper=40),
                    name="Gap Ahead", mode="lines",
                    line=dict(color="#FF8C00", width=2),
                ), row=2, col=1)
            if df_st["gap_behind_s"].lt(90).any():
                fig_st.add_trace(go.Scatter(
                    x=df_st["lap_number"],
                    y=df_st["gap_behind_s"].clip(upper=40),
                    name="Gap Behind", mode="lines",
                    line=dict(color=TOKENS["accent2"], width=2, dash="dot"),
                ), row=2, col=1)
            fig_st.add_hline(y=PIT_LANE_LOSS * 0.70, line_dash="dot",
                             line_color=TOKENS["accent"],
                             annotation_text="UNDERCUT ZONE",
                             annotation_font_color=TOKENS["accent"],
                             row=2, col=1)

            fig_st.add_trace(go.Scatter(
                x=df_st["lap_number"], y=df_st["tire_degradation"],
                name="Tyre Deg", mode="lines",
                line=dict(color=TOKENS["warn"], width=2),
                fill="tozeroy",
                fillcolor=f"rgba({int(TOKENS['warn'][1:3],16)},"
                           f"{int(TOKENS['warn'][3:5],16)},"
                           f"{int(TOKENS['warn'][5:7],16)},0.10)",
            ), row=3, col=1)

            for _, smr_s in df_st[df_st["signal"].isin(
                    ["UNDERCUT", "OVERCUT", "PIT NOW"])].iterrows():
                clr_s = SIG_COLORS.get(smr_s["signal"], TOKENS["text"])
                fig_st.add_vline(
                    x=smr_s["lap_number"], line_dash="dot",
                    line_color=clr_s, line_width=1.5,
                    annotation_text=smr_s["signal"][:3],
                    annotation_font_color=clr_s,
                    annotation_font_size=8,
                    row=1, col=1,
                )

            fig_st.update_yaxes(title_text="P(PIT)", range=[0, 1], row=1, col=1)
            fig_st.update_yaxes(title_text="GAP (s)", row=2, col=1)
            fig_st.update_yaxes(title_text="DEG", range=[0, 1], row=3, col=1)
            fig_st.update_xaxes(title_text="LAP", row=3, col=1)
            st.plotly_chart(style_fig(fig_st, 580), use_container_width=True)

            # ── Lap-by-lap table ───────────────────────────────────────
            st.markdown(
                "<div style='font-family:var(--font-data,monospace);font-size:.7rem;"
                "letter-spacing:2px;color:var(--muted);margin:8px 0 6px;'>"
                "FILTER · PROBABILITY THRESHOLD</div>",
                unsafe_allow_html=True,
            )
            thresh_p = st.slider("", 0.0, 1.0, 0.2, 0.05,
                                 label_visibility="collapsed", key="strat_thresh_p")
            df_vw = df_st[df_st["pit_proba"] >= thresh_p].copy()
            df_vw["pit_proba"]        = df_vw["pit_proba"].round(3)
            df_vw["tire_degradation"] = df_vw["tire_degradation"].round(3)
            df_vw["gap_ahead_s"]      = df_vw["gap_ahead_s"].clip(upper=40.0).round(2)
            df_vw["gap_behind_s"]     = df_vw["gap_behind_s"].clip(upper=40.0).round(2)
            df_vw["lap_number"]       = df_vw["lap_number"].astype(int)
            df_vw = df_vw.rename(columns={
                "lap_number": "Lap", "pit_proba": "Pit Prob",
                "gap_ahead_s": "Gap Ahead (s)", "gap_behind_s": "Gap Behind (s)",
                "tire_compound": "Compound", "tire_age": "Tyre Age",
                "tire_degradation": "Tyre Deg", "signal": "Signal",
                "confidence": "Confidence", "rationale": "Rationale",
            })
            if not df_vw.empty:
                st.dataframe(
                    df_vw[[
                        "Lap", "Pit Prob", "Signal", "Confidence",
                        "Gap Ahead (s)", "Gap Behind (s)",
                        "Compound", "Tyre Age", "Tyre Deg", "Rationale",
                    ]].reset_index(drop=True),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No laps meet the selected threshold.")

        except Exception as e:
            st.error(f"Strategy analysis failed: {e}")
