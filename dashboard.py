"""
============================================================
F1 RACE STRATEGY — STREAMLIT DASHBOARD
============================================================
Run:  cd /Users/daviesadetiba/f1_model
      ./venv/bin/streamlit run dashboard.py
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import joblib

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Supervised Pit Stop Strategy Model — University of Johannesburg",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── UJ branding assets (red / navy / white theme) ─────────────
UJ_LOGO_LOCAL       = "assets/uj_logo.png"   # user-saved official logo (preferred)
UJ_LOGO_PLACEHOLDER = "assets/uj_logo_placeholder.svg"

# Red / Navy / White colour palette
RED    = "#C8102E"   # primary accent (UJ red / F1 inspired)
NAVY   = "#0B1B3D"   # primary background accent
WHITE  = "#FFFFFF"
LIGHT  = "#F5F5F5"
DARK   = "#0E1117"   # base page bg
INK    = "#1A1A1A"   # text on light surfaces


def uj_logo_source() -> str:
    """Return path to local UJ logo image if saved, else SVG placeholder."""
    if os.path.exists(UJ_LOGO_LOCAL):
        return UJ_LOGO_LOCAL
    return UJ_LOGO_PLACEHOLDER

# ── Colour palette ─────────────────────────────────────────────
COMPOUND_COLORS = {
    "SOFT":    "#E8002D",
    "MEDIUM":  "#FFF200",
    "HARD":    "#EEEEEE",
    "INTER":   "#39B54A",
    "WET":     "#0067FF",
    "UNKNOWN": "#AAAAAA",
}

# ── Red / Navy / White theme overrides ─────────────────────────
st.markdown(f"""
<style>
  /* Global background */
  .stApp {{ background-color: {DARK}; }}
  .block-container {{ padding-top: 1rem; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
      background: linear-gradient(180deg, {NAVY} 0%, #050d1f 100%);
      border-right: 2px solid {RED};
  }}
  [data-testid="stSidebar"] * {{ color: {WHITE} !important; }}

  /* Header banner — NAVY backdrop with RED accent band */
  .uj-banner {{
      background: linear-gradient(135deg, {NAVY} 0%, #112352 100%);
      border-top: 4px solid {RED};
      border-bottom: 4px solid {RED};
      border-radius: 6px;
      padding: 20px 28px;
      margin-bottom: 18px;
      display: flex;
      align-items: center;
      gap: 28px;
      box-shadow: 0 4px 12px rgba(200, 16, 46, 0.18);
  }}
  .uj-banner-logo {{
      background: {WHITE};
      padding: 10px;
      border-radius: 6px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.45);
      flex: 0 0 auto;
  }}
  .uj-banner-text {{ flex: 1; color: {WHITE}; }}
  .uj-banner-eyebrow {{
      color: {WHITE};
      font-weight: 700;
      font-size: 0.78rem;
      letter-spacing: 1.8px;
      opacity: 0.85;
  }}
  .uj-banner-title {{
      font-size: 1.7rem;
      font-weight: 700;
      margin-top: 4px;
      letter-spacing: 0.2px;
  }}
  .uj-banner-sub {{
      color: {RED};
      font-size: 0.95rem;
      margin-top: 4px;
      letter-spacing: 0.5px;
      font-weight: 500;
  }}
  .uj-banner-meta {{
      color: #bfc8df;
      font-size: 0.8rem;
      margin-top: 6px;
  }}

  /* Tabs — red active underline */
  .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
      color: {RED} !important;
      border-bottom-color: {RED} !important;
  }}

  /* Metrics */
  [data-testid="stMetric"] {{
      background: {NAVY};
      padding: 14px 16px;
      border-radius: 6px;
      border-left: 3px solid {RED};
  }}
  [data-testid="stMetricLabel"] {{ color: #bfc8df !important; }}
  [data-testid="stMetricValue"] {{ color: {WHITE} !important; }}

  /* Buttons & sliders */
  .stSlider [data-baseweb="slider"] > div > div {{ background: {RED} !important; }}
  .stButton button {{
      background: {RED};
      color: {WHITE};
      border: none;
  }}
  .stButton button:hover {{ background: #a30d24; color: {WHITE}; }}

  /* Plotly chart corner radius */
  .stPlotlyChart {{
      border-radius: 6px;
      border: 1px solid rgba(255,255,255,0.06);
  }}

  /* Dataframe header */
  .stDataFrame thead tr th {{
      background-color: {NAVY} !important;
      color: {WHITE} !important;
  }}
</style>
""", unsafe_allow_html=True)

# ── UJ-branded header banner (navy base, red accents) ──────────
hdr_col1, hdr_col2 = st.columns([1, 7], gap="medium")
with hdr_col1:
    st.image(uj_logo_source(), width=130)
with hdr_col2:
    st.markdown(f"""
        <div style="padding-top:10px;">
          <div class="uj-banner-eyebrow" style="color:{WHITE};">
              UNIVERSITY OF JOHANNESBURG
          </div>
          <div class="uj-banner-title" style="color:{WHITE};">
              F1 Supervised Pit Stop Strategy Model
          </div>
          <div class="uj-banner-sub" style="color:{RED};">
              A Design Science Research Methodology Artefact
          </div>
          <div class="uj-banner-meta" style="color:#bfc8df;">
              Master's Research Project by Davies Adetiba · Faculty of Computer Science and Software Engineering
          </div>
        </div>
    """, unsafe_allow_html=True)

# Red accent stripe under header
st.markdown(
    f"<div style='height:4px; background:{RED}; margin:8px 0 16px 0; "
    f"border-radius:2px;'></div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading datasets…")
def load_data():
    output_dir = "f1_outputs"
    train_path = os.path.join(output_dir, "combined_train_2022_2023.csv")
    test_path  = os.path.join(output_dir, "combined_test_2024.csv")

    dfs = []
    for path, label in [(train_path, "train"), (test_path, "test")]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["split"] = label
            dfs.append(df)

    if not dfs:
        st.error("No data found in f1_outputs/. Run run_model.py first.")
        st.stop()

    df = pd.concat(dfs, ignore_index=True)
    df["season"]       = df["season"].astype(int)
    df["lap_number"]   = pd.to_numeric(df["lap_number"],  errors="coerce")
    df["lap_time_s"]   = pd.to_numeric(df["lap_time_s"],  errors="coerce")
    df["tire_age"]     = pd.to_numeric(df["tire_age"],     errors="coerce")
    df["position"]     = pd.to_numeric(df["position"],     errors="coerce")
    df["pitstop_this_lap"] = pd.to_numeric(df["pitstop_this_lap"], errors="coerce")
    df["tire_compound"] = df["tire_compound"].fillna("UNKNOWN").str.upper()
    return df


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    path = "f1_outputs/f1_strategy_model_v2.joblib"
    if os.path.exists(path):
        return joblib.load(path)
    return None


df_all  = load_data()
model   = load_model()


# ══════════════════════════════════════════════════════════════
# SIDEBAR — FILTERS
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        f"<div style='background:{WHITE}; padding:10px; border-radius:6px; "
        f"text-align:center; box-shadow:0 2px 6px rgba(0,0,0,0.5);'>",
        unsafe_allow_html=True,
    )
    st.image(uj_logo_source(), width=150)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
        <div style="color:{WHITE}; font-weight:700; font-size:0.78rem;
                    letter-spacing:1.4px; margin-top:14px; text-align:center;">
            UNIVERSITY OF JOHANNESBURG
        </div>
        <div style="font-size:1.1rem; font-weight:700; margin-top:8px;
                    text-align:center; color:{WHITE};">
            🏎️ F1 Pit Stop Strategy
        </div>
        <div style="color:{RED}; font-size:0.78rem; margin-top:2px;
                    letter-spacing:0.5px; text-align:center; font-weight:600;">
            Master's Research Project<br/>by Davies Adetiba
        </div>
    """, unsafe_allow_html=True)

    # Red divider line
    st.markdown(
        f"<div style='height:2px; background:{RED}; margin:14px 0;'></div>",
        unsafe_allow_html=True,
    )

    seasons_available = sorted(df_all["season"].unique())
    selected_season   = st.selectbox("Season", seasons_available, index=len(seasons_available)-1)

    df_season = df_all[df_all["season"] == selected_season]
    races_available = (
        df_season[["round_number", "race_name"]]
        .drop_duplicates()
        .sort_values("round_number")
    )
    race_labels  = races_available["race_name"].tolist()
    selected_race = st.selectbox("Race", race_labels)

    df_race = df_season[df_season["race_name"] == selected_race]
    drivers_available = sorted(df_race["driver_id"].dropna().unique())
    selected_driver   = st.selectbox("Driver", drivers_available)

    st.divider()
    st.caption(f"**Data split:** {'🟢 Test' if selected_season == 2024 else '🔵 Train'}")
    n_rows = len(df_race)
    n_pits = int(df_race["pitstop_this_lap"].sum())
    st.caption(f"Race laps in dataset: **{n_rows:,}** | Pit stops: **{n_pits}**")


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════

st.markdown(f"## {selected_season} · {selected_race}")
st.markdown(f"*Driver: **{selected_driver}***  &nbsp;|&nbsp;  Season split: {'🟢 **Test set** (unseen)' if selected_season == 2024 else '🔵 **Training set**'}")
st.divider()


# ══════════════════════════════════════════════════════════════
# TOP METRICS ROW
# ══════════════════════════════════════════════════════════════

df_driver = df_race[df_race["driver_id"] == selected_driver].sort_values("lap_number")

col1, col2, col3, col4, col5 = st.columns(5)

if not df_driver.empty:
    total_laps   = int(df_driver["lap_number"].max())
    pit_laps     = int(df_driver["pitstop_this_lap"].sum())
    avg_lap      = df_driver["lap_time_s"].median()
    best_lap     = df_driver["lap_time_s"].min()
    compounds    = df_driver["tire_compound"].unique()

    col1.metric("Total Laps",    f"{total_laps}")
    col2.metric("Pit Stops",     f"{pit_laps}")
    col3.metric("Median Lap",    f"{avg_lap:.3f}s")
    col4.metric("Fastest Lap",   f"{best_lap:.3f}s")
    col5.metric("Compounds Used", ", ".join(sorted(set(compounds) - {"UNKNOWN"})))

st.divider()


# ══════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏁 Lap Times & Strategy",
    "🔮 Pit Stop Probability",
    "🛞 Tyre Analysis",
    "📊 Model Performance",
    "🌦️ Race Conditions",
    "🎯 Strategy Recommendation",
])


# ── TAB 1: Lap Times & Strategy ────────────────────────────────
with tab1:
    if df_driver.empty:
        st.warning("No data for selected driver.")
    else:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.7, 0.3],
            subplot_titles=["Lap Time Evolution", "Tyre Compound per Lap"],
            vertical_spacing=0.08,
        )

        # Lap time line coloured by compound
        for compound in df_driver["tire_compound"].unique():
            sub = df_driver[df_driver["tire_compound"] == compound]
            fig.add_trace(go.Scatter(
                x=sub["lap_number"], y=sub["lap_time_s"],
                mode="lines+markers",
                name=compound,
                line=dict(color=COMPOUND_COLORS.get(compound, "#888"), width=2),
                marker=dict(size=5),
                hovertemplate="Lap %{x}<br>%{y:.3f}s<extra>" + compound + "</extra>",
            ), row=1, col=1)

        # Pit stop vertical lines
        pit_laps = df_driver[df_driver["pitstop_this_lap"] == 1]["lap_number"]
        for lp in pit_laps:
            fig.add_vline(
                x=lp, line_dash="dot", line_color=RED,
                line_width=1.5, row=1, col=1,
                annotation_text="PIT", annotation_font_color=RED,
                annotation_font_size=9,
            )

        # Compound strip chart (row 2)
        for compound in df_driver["tire_compound"].unique():
            sub = df_driver[df_driver["tire_compound"] == compound]
            fig.add_trace(go.Bar(
                x=sub["lap_number"],
                y=[1] * len(sub),
                name=compound,
                marker_color=COMPOUND_COLORS.get(compound, "#888"),
                showlegend=False,
                hovertemplate="Lap %{x}<br>" + compound + "<extra></extra>",
            ), row=2, col=1)

        fig.update_layout(
            height=520, template="plotly_dark",
            paper_bgcolor=NAVY, plot_bgcolor=NAVY,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            margin=dict(l=40, r=20, t=60, b=20),
        )
        fig.update_yaxes(title_text="Lap Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Compound", row=2, col=1, showticklabels=False)
        fig.update_xaxes(title_text="Lap Number", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Race stint summary table
        st.subheader("Stint Summary")
        if "stint_number" in df_driver.columns:
            stint_summary = (
                df_driver.groupby("stint_number")
                .agg(
                    Compound   =("tire_compound", "first"),
                    Start_Lap  =("lap_number", "min"),
                    End_Lap    =("lap_number", "max"),
                    Laps       =("lap_number", "count"),
                    Avg_Lap_s  =("lap_time_s", "median"),
                    Fresh      =("fresh_tyre", "first"),
                )
                .reset_index()
                .rename(columns={"stint_number": "Stint"})
            )
            stint_summary["Avg_Lap_s"] = stint_summary["Avg_Lap_s"].round(3)
            stint_summary["Fresh"] = stint_summary["Fresh"].map({1: "✅ New", 0: "♻️ Scrubbed"})
            st.dataframe(stint_summary, use_container_width=True, hide_index=True)


# ── TAB 2: Pit Stop Probability ────────────────────────────────
with tab2:
    st.subheader("Predicted Pit Stop Probability")

    if model is None:
        st.warning("Model not loaded. Run run_model.py first.")
    elif df_driver.empty:
        st.warning("No data for selected driver.")
    else:
        from f1_strategy_model_2 import engineer_features, ALL_FEATURES

        try:
            df_eng  = engineer_features(df_driver.copy())

            # Locate the binary pit-stop sklearn estimator inside the
            # F1StrategyModel wrapper (attribute is `pitstop_model`).
            pit_model = (
                model.pitstop_model
                if hasattr(model, "pitstop_model")
                else (model.pit_model if hasattr(model, "pit_model") else model)
            )

            # Use the feature order the model was trained on
            expected = (
                list(model.feature_names)
                if hasattr(model, "feature_names") and model.feature_names is not None
                else (
                    list(pit_model.feature_names_in_)
                    if hasattr(pit_model, "feature_names_in_")
                    else [f for f in ALL_FEATURES if f in df_eng.columns]
                )
            )

            # Ensure every expected column exists, then align order
            for col in expected:
                if col not in df_eng.columns:
                    df_eng[col] = 0
            X_input = df_eng[expected].fillna(0)

            # Optional scaler stored on the wrapper
            if hasattr(model, "scaler") and model.scaler is not None:
                try:
                    X_input = model.scaler.transform(X_input)
                except Exception:
                    pass

            proba = pit_model.predict_proba(X_input)[:, 1]
            df_eng["pit_proba"] = proba

            fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 row_heights=[0.65, 0.35],
                                 subplot_titles=["Pit Stop Probability", "Actual Pit Stops"],
                                 vertical_spacing=0.08)

            # Probability fill
            fig2.add_trace(go.Scatter(
                x=df_eng["lap_number"], y=df_eng["pit_proba"],
                mode="lines", name="Pit Probability",
                line=dict(color=RED, width=2),
                fill="tozeroy", fillcolor="rgba(200,16,46,0.18)",
            ), row=1, col=1)

            # 50% threshold line
            fig2.add_hline(y=0.5, line_dash="dash", line_color=WHITE,
                          annotation_text="50% threshold",
                          annotation_font_color=WHITE, row=1, col=1)

            # Actual pit stop bars
            actual_pits = df_eng[df_eng["pitstop_this_lap"] == 1]
            fig2.add_trace(go.Bar(
                x=actual_pits["lap_number"],
                y=[1] * len(actual_pits),
                name="Actual Pit Stop",
                marker_color="#39B54A",
            ), row=2, col=1)

            fig2.update_layout(
                height=450, template="plotly_dark",
                paper_bgcolor=NAVY, plot_bgcolor=NAVY,
                margin=dict(l=40, r=20, t=60, b=20),
            )
            fig2.update_yaxes(title_text="Probability", range=[0, 1], row=1, col=1)
            fig2.update_xaxes(title_text="Lap Number", row=2, col=1)
            st.plotly_chart(fig2, use_container_width=True)

            # High-probability windows
            threshold = st.slider("Probability threshold", 0.1, 0.9, 0.5, 0.05)
            windows = df_eng[df_eng["pit_proba"] >= threshold][["lap_number", "pit_proba", "tire_compound", "tire_age"]]
            if not windows.empty:
                st.success(f"**{len(windows)} laps** exceed the {threshold:.0%} threshold:")
                windows["pit_proba"] = windows["pit_proba"].round(3)
                st.dataframe(windows.reset_index(drop=True), use_container_width=True, hide_index=True)
            else:
                st.info("No laps exceed the selected threshold for this driver.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.caption("The model may need retraining with the current feature set.")


# ── TAB 3: Tyre Analysis ───────────────────────────────────────
with tab3:
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Tyre Degradation")
        if not df_driver.empty:
            fig3 = go.Figure()
            for compound in df_driver["tire_compound"].unique():
                sub = df_driver[df_driver["tire_compound"] == compound]
                fig3.add_trace(go.Scatter(
                    x=sub["tire_age"], y=sub["tire_degradation"],
                    mode="markers+lines", name=compound,
                    line=dict(color=COMPOUND_COLORS.get(compound, "#888")),
                    marker=dict(size=5),
                ))
            fig3.update_layout(
                template="plotly_dark", height=320,
                paper_bgcolor=NAVY, plot_bgcolor=NAVY,
                xaxis_title="Tyre Age (laps)", yaxis_title="Degradation [0–1]",
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig3, use_container_width=True)

    with colB:
        st.subheader("Compound Usage — Full Race")
        if not df_race.empty:
            comp_counts = df_race["tire_compound"].value_counts().reset_index()
            comp_counts.columns = ["Compound", "Laps"]
            fig4 = px.bar(
                comp_counts, x="Compound", y="Laps",
                color="Compound",
                color_discrete_map=COMPOUND_COLORS,
                template="plotly_dark",
            )
            fig4.update_layout(
                height=320, showlegend=False,
                paper_bgcolor=NAVY, plot_bgcolor=NAVY,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig4, use_container_width=True)

    # All drivers tyre strategy heatmap
    st.subheader("Tyre Strategy — All Drivers")
    if not df_race.empty:
        pivot = df_race.pivot_table(
            index="driver_id", columns="lap_number",
            values="tire_compound", aggfunc="first"
        )
        # Map compounds to integers for colouring
        comp_int = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTER": 3, "WET": 4, "UNKNOWN": 5}
        pivot_int = pivot.replace(comp_int)

        fig5 = go.Figure(data=go.Heatmap(
            z=pivot_int.values,
            x=pivot_int.columns.tolist(),
            y=pivot_int.index.tolist(),
            colorscale=[
                [0.0,  "#E8002D"],
                [0.2,  "#FFF200"],
                [0.4,  "#EEEEEE"],
                [0.6,  "#39B54A"],
                [0.8,  "#0067FF"],
                [1.0,  "#555555"],
            ],
            showscale=False,
            hovertemplate="Driver: %{y}<br>Lap: %{x}<extra></extra>",
        ))
        fig5.update_layout(
            template="plotly_dark", height=420,
            paper_bgcolor=NAVY, plot_bgcolor=NAVY,
            xaxis_title="Lap Number", yaxis_title="Driver",
            margin=dict(l=60, r=20, t=20, b=40),
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("🔴 SOFT   🟡 MEDIUM   ⬜ HARD   🟢 INTER   🔵 WET")


# ── TAB 4: Model Performance ───────────────────────────────────
with tab4:
    st.subheader("Model Evaluation Metrics — Test Set (2024)")
    st.caption("Trained on 2022+2023 · Evaluated on unseen 2024 season")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("MCC",         "0.167",  help="Matthews Correlation Coefficient — primary metric. Accounts for class imbalance.")
    m2.metric("G-mean",      "0.694",  help="Geometric mean of sensitivity and specificity. Balances both classes.")
    m3.metric("ROC-AUC",     "0.805",  help="Area under the ROC curve. Measures ranking ability.")
    m4.metric("Sensitivity", "60.4%",  help="Proportion of actual pit stops correctly predicted.")
    m5.metric("Specificity", "79.7%",  help="Proportion of non-pit laps correctly dismissed.")

    st.divider()

    col_feat, col_conf = st.columns([1, 1])

    with col_feat:
        st.subheader("Feature Importance")
        img_path = "f1_outputs/feature_importance.png"
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)

    with col_conf:
        st.subheader("Confusion Matrix")
        img_path = "f1_outputs/confusion_matrices.png"
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)

    st.divider()

    col_roc, col_mcc = st.columns([1, 1])
    with col_roc:
        st.subheader("ROC & PR Curves")
        img_path = "f1_outputs/roc_pr_curves_Test.png"
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)

    with col_mcc:
        st.subheader("MCC / G-mean Summary")
        img_path = "f1_outputs/mcc_gmean_Test.png"
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)

    st.divider()
    st.subheader("Confusion Matrix — Raw Numbers")
    conf_data = {
        "": ["**Actual: No Pit**", "**Actual: Pit**"],
        "Predicted: No Pit": ["20,388 (TN ✅)", "314 (FN ❌)"],
        "Predicted: Pit":    ["5,185 (FP ⚠️)", "478 (TP ✅)"],
    }
    st.table(pd.DataFrame(conf_data).set_index(""))
    st.caption("""
    **TN** = correctly dismissed non-pit laps · **TP** = correctly predicted pit stops
    **FN** = missed pit windows (dangerous in race strategy) · **FP** = false alarms (low cost)
    """)


# ── TAB 5: Race Conditions ─────────────────────────────────────
with tab5:
    st.subheader("Weather & Race Conditions")

    weather_cols = ["lap_number", "track_temp_c", "air_temp_c",
                    "humidity_pct", "wind_speed_ms", "rainfall_mm", "safety_car_active"]
    avail = [c for c in weather_cols if c in df_driver.columns]

    if df_driver.empty or len(avail) < 3:
        st.info("Weather data not available for this selection.")
    else:
        df_w = df_driver[avail].sort_values("lap_number")

        fig6 = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=["Temperature (°C)", "Humidity & Wind", "Safety Car / Rain"],
            vertical_spacing=0.08,
        )

        if "track_temp_c" in df_w.columns:
            fig6.add_trace(go.Scatter(
                x=df_w["lap_number"], y=df_w["track_temp_c"],
                name="Track Temp", line=dict(color="#FF6B35", width=2),
            ), row=1, col=1)
        if "air_temp_c" in df_w.columns:
            fig6.add_trace(go.Scatter(
                x=df_w["lap_number"], y=df_w["air_temp_c"],
                name="Air Temp", line=dict(color="#FFD700", width=2, dash="dot"),
            ), row=1, col=1)

        if "humidity_pct" in df_w.columns:
            fig6.add_trace(go.Scatter(
                x=df_w["lap_number"], y=df_w["humidity_pct"],
                name="Humidity %", line=dict(color="#00D2BE", width=2),
            ), row=2, col=1)
        if "wind_speed_ms" in df_w.columns:
            fig6.add_trace(go.Scatter(
                x=df_w["lap_number"], y=df_w["wind_speed_ms"],
                name="Wind (m/s)", line=dict(color="#9B59B6", width=2, dash="dot"),
            ), row=2, col=1)

        if "safety_car_active" in df_w.columns:
            fig6.add_trace(go.Bar(
                x=df_w["lap_number"], y=df_w["safety_car_active"],
                name="Safety Car", marker_color="#FFD700",
            ), row=3, col=1)
        if "rainfall_mm" in df_w.columns:
            fig6.add_trace(go.Bar(
                x=df_w["lap_number"], y=df_w["rainfall_mm"],
                name="Rainfall", marker_color="#0067FF",
            ), row=3, col=1)

        fig6.update_layout(
            height=520, template="plotly_dark",
            paper_bgcolor=NAVY, plot_bgcolor=NAVY,
            margin=dict(l=40, r=20, t=60, b=20),
        )
        st.plotly_chart(fig6, use_container_width=True)

        # Summary stats
        st.subheader("Conditions Summary")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Avg Track Temp", f"{df_w['track_temp_c'].mean():.1f}°C" if 'track_temp_c' in df_w.columns else "N/A")
        sc2.metric("Avg Humidity",   f"{df_w['humidity_pct'].mean():.1f}%"  if 'humidity_pct' in df_w.columns else "N/A")
        sc3.metric("Max Wind",       f"{df_w['wind_speed_ms'].max():.1f} m/s" if 'wind_speed_ms' in df_w.columns else "N/A")
        sc4.metric("SC Laps",        f"{int(df_w['safety_car_active'].sum())}" if 'safety_car_active' in df_w.columns else "N/A")


# ── TAB 6: Strategy Recommendation ────────────────────────────
with tab6:
    st.subheader("🎯 Undercut / Overcut Strategy Recommendation")

    PIT_LANE_LOSS = 21.0  # typical pit lane time loss (s)

    SIGNAL_COLORS = {
        "UNDERCUT":      RED,
        "OVERCUT":       "#39B54A",
        "PIT NOW":       "#FF8C00",
        "UNDERCUT RISK": "#FFD700",
        "STAY OUT":      "#5A7FA5",
    }

    def classify_strategy(gap_ahead, gap_behind, pit_proba, tire_deg, tire_age):
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

    if model is None or df_driver.empty:
        st.warning("Load model and select a driver to see strategy recommendations.")
    else:
        try:
            df_eng2 = engineer_features(df_driver.copy())
            pit_model2 = (
                model.pitstop_model if hasattr(model, "pitstop_model")
                else (model.pit_model if hasattr(model, "pit_model") else model)
            )
            expected2 = (
                list(model.feature_names)
                if hasattr(model, "feature_names") and model.feature_names is not None
                else (list(pit_model2.feature_names_in_)
                      if hasattr(pit_model2, "feature_names_in_")
                      else [f for f in ALL_FEATURES if f in df_eng2.columns])
            )
            for col in expected2:
                if col not in df_eng2.columns:
                    df_eng2[col] = 0
            X2 = df_eng2[expected2].fillna(0)
            if hasattr(model, "scaler") and model.scaler is not None:
                try:
                    X2 = model.scaler.transform(X2)
                except Exception:
                    pass
            df_eng2["pit_proba"] = pit_model2.predict_proba(X2)[:, 1]

            for gcol in ["gap_ahead_s", "gap_behind_s"]:
                if gcol not in df_eng2.columns:
                    df_eng2[gcol] = 99.0
            df_eng2["gap_ahead_s"]  = df_eng2["gap_ahead_s"].fillna(99.0)
            df_eng2["gap_behind_s"] = df_eng2["gap_behind_s"].fillna(99.0)
            if "tire_degradation" not in df_eng2.columns:
                df_eng2["tire_degradation"] = 0.0

            results = df_eng2.apply(
                lambda r: classify_strategy(
                    r["gap_ahead_s"], r["gap_behind_s"],
                    r["pit_proba"], r.get("tire_degradation", 0.0), r.get("tire_age", 0.0)
                ), axis=1, result_type="expand"
            )
            results.columns = ["signal", "confidence", "rationale"]
            df_strat = pd.concat([
                df_eng2[["lap_number", "pit_proba", "gap_ahead_s", "gap_behind_s",
                          "tire_compound", "tire_age", "tire_degradation"]].reset_index(drop=True),
                results.reset_index(drop=True),
            ], axis=1)

            # ── Top recommendation card ────────────────────────────────
            best_rows = df_strat[df_strat["pit_proba"] >= 0.3].sort_values("pit_proba", ascending=False)
            if best_rows.empty:
                best_rows = df_strat.sort_values("pit_proba", ascending=False)
            top = best_rows.iloc[0]
            sig_color  = SIGNAL_COLORS.get(top["signal"], WHITE)
            conf_label = {"HIGH": "🔴 HIGH", "MEDIUM": "🟡 MEDIUM", "LOW": "🟢 LOW"}.get(
                top["confidence"], top["confidence"])

            st.markdown(f"""
            <div style="background:{NAVY}; border-left:5px solid {sig_color};
                        border-radius:6px; padding:20px 24px; margin-bottom:18px;">
              <div style="color:{sig_color}; font-size:1.6rem; font-weight:700;
                          letter-spacing:1px;">{top['signal']}</div>
              <div style="color:{WHITE}; font-size:0.95rem; margin-top:6px;">
                {top['rationale']}
              </div>
              <div style="color:#bfc8df; font-size:0.8rem; margin-top:8px;">
                Confidence: <strong style="color:{sig_color};">{conf_label}</strong>
                &nbsp;|&nbsp; Lap <strong>{int(top['lap_number'])}</strong>
                &nbsp;|&nbsp; Pit Probability: <strong>{top['pit_proba']:.1%}</strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Three-panel strategy chart ─────────────────────────────
            fig_s = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                row_heights=[0.45, 0.30, 0.25],
                subplot_titles=[
                    "Pit Stop Probability",
                    "Gaps to Adjacent Cars (s)",
                    "Tyre Degradation",
                ],
                vertical_spacing=0.09,
            )

            fig_s.add_trace(go.Scatter(
                x=df_strat["lap_number"], y=df_strat["pit_proba"],
                name="Pit Prob", mode="lines",
                line=dict(color=RED, width=2),
                fill="tozeroy", fillcolor="rgba(200,16,46,0.15)",
            ), row=1, col=1)
            fig_s.add_hline(y=0.5, line_dash="dash", line_color=WHITE,
                            annotation_text="50%", annotation_font_color=WHITE,
                            row=1, col=1)

            if df_strat["gap_ahead_s"].lt(90).any():
                fig_s.add_trace(go.Scatter(
                    x=df_strat["lap_number"],
                    y=df_strat["gap_ahead_s"].clip(upper=40),
                    name="Gap Ahead", mode="lines",
                    line=dict(color="#FF8C00", width=2),
                ), row=2, col=1)
            if df_strat["gap_behind_s"].lt(90).any():
                fig_s.add_trace(go.Scatter(
                    x=df_strat["lap_number"],
                    y=df_strat["gap_behind_s"].clip(upper=40),
                    name="Gap Behind", mode="lines",
                    line=dict(color="#00D2BE", width=2, dash="dot"),
                ), row=2, col=1)
            fig_s.add_hline(y=PIT_LANE_LOSS * 0.70, line_dash="dot", line_color=RED,
                            annotation_text="Undercut zone",
                            annotation_font_color=RED, row=2, col=1)

            fig_s.add_trace(go.Scatter(
                x=df_strat["lap_number"], y=df_strat["tire_degradation"],
                name="Tyre Deg", mode="lines",
                line=dict(color="#FFF200", width=2),
                fill="tozeroy", fillcolor="rgba(255,242,0,0.10)",
            ), row=3, col=1)

            # Annotate signal laps on probability panel
            for _, smr in df_strat[df_strat["signal"].isin(
                    ["UNDERCUT", "OVERCUT", "PIT NOW"])].iterrows():
                clr = SIGNAL_COLORS.get(smr["signal"], WHITE)
                fig_s.add_vline(
                    x=smr["lap_number"], line_dash="dot",
                    line_color=clr, line_width=1.5,
                    annotation_text=smr["signal"][:3],
                    annotation_font_color=clr,
                    annotation_font_size=8,
                    row=1, col=1,
                )

            fig_s.update_layout(
                height=560, template="plotly_dark",
                paper_bgcolor=NAVY, plot_bgcolor=NAVY,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=40, r=20, t=60, b=20),
            )
            fig_s.update_yaxes(title_text="Probability", range=[0, 1], row=1, col=1)
            fig_s.update_yaxes(title_text="Gap (s)", row=2, col=1)
            fig_s.update_yaxes(title_text="Degradation", range=[0, 1], row=3, col=1)
            fig_s.update_xaxes(title_text="Lap Number", row=3, col=1)
            st.plotly_chart(fig_s, use_container_width=True)

            # ── Lap-by-lap table ───────────────────────────────────────
            st.subheader("Lap-by-Lap Strategy Signals")
            thresh_s = st.slider(
                "Show laps with pit probability ≥",
                0.0, 1.0, 0.2, 0.05, key="strat_thresh",
            )
            df_view = df_strat[df_strat["pit_proba"] >= thresh_s].copy()
            df_view["pit_proba"]        = df_view["pit_proba"].round(3)
            df_view["tire_degradation"] = df_view["tire_degradation"].round(3)
            df_view["gap_ahead_s"]      = df_view["gap_ahead_s"].clip(upper=40.0).round(2)
            df_view["gap_behind_s"]     = df_view["gap_behind_s"].clip(upper=40.0).round(2)
            df_view["lap_number"]       = df_view["lap_number"].astype(int)
            df_view = df_view.rename(columns={
                "lap_number": "Lap", "pit_proba": "Pit Prob",
                "gap_ahead_s": "Gap Ahead (s)", "gap_behind_s": "Gap Behind (s)",
                "tire_compound": "Compound", "tire_age": "Tyre Age",
                "tire_degradation": "Tyre Deg", "signal": "Signal",
                "confidence": "Confidence", "rationale": "Rationale",
            })
            if not df_view.empty:
                st.dataframe(
                    df_view[[
                        "Lap", "Pit Prob", "Signal", "Confidence",
                        "Gap Ahead (s)", "Gap Behind (s)",
                        "Compound", "Tyre Age", "Tyre Deg", "Rationale",
                    ]].reset_index(drop=True),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("No laps meet the selected threshold.")

            # ── Summary counts ─────────────────────────────────────────
            st.divider()
            st.subheader("Strategy Summary")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Undercut Windows",   int((df_strat["signal"] == "UNDERCUT").sum()))
            sc2.metric("Overcut Windows",    int((df_strat["signal"] == "OVERCUT").sum()))
            sc3.metric("Pit Now Windows",    int((df_strat["signal"] == "PIT NOW").sum()))
            sc4.metric("Undercut Risk Laps", int((df_strat["signal"] == "UNDERCUT RISK").sum()))

        except Exception as e:
            st.error(f"Strategy analysis failed: {e}")


# ── Footer ─────────────────────────────────────────────────────
st.divider()
st.caption(
    "F1 Supervised Pit Stop Strategy Model · University of Johannesburg · "
    "Design Science Research Methodology (Peffers et al., 2007) · "
    "Data: FastF1 / Tracing Insights · Model: GradientBoostingClassifier + RandomForestClassifier"
)
