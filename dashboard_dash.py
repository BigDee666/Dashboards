"""
============================================================
F1 RACE STRATEGY — PLOTLY DASH DASHBOARD
============================================================
Same feature set as the Streamlit version, built on Dash so
you can compare complexity, performance, and polish.

Run:  ./venv/bin/python dashboard_dash.py
URL:  http://localhost:8050
============================================================
"""

import warnings; warnings.filterwarnings("ignore")
import os, sys, base64
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import Dash, dcc, html, Input, Output, State, no_update, callback
import dash_bootstrap_components as dbc

sys.path.insert(0, ".")
from f1_strategy_model_2 import engineer_features, ALL_FEATURES

# ══════════════════════════════════════════════════════════════
# THEME — Red / Navy / White
# ══════════════════════════════════════════════════════════════
RED   = "#C8102E"
NAVY  = "#0B1B3D"
WHITE = "#FFFFFF"
DARK  = "#0E1117"
MUTED = "#bfc8df"

COMPOUND_COLORS = {
    "SOFT":    "#E8002D", "MEDIUM":  "#FFF200", "HARD":    "#EEEEEE",
    "INTER":   "#39B54A", "WET":     "#0067FF", "UNKNOWN": "#AAAAAA",
}

# ══════════════════════════════════════════════════════════════
# DATA / MODEL LOADING (one-time at startup — Dash is stateless)
# ══════════════════════════════════════════════════════════════
def load_corpus():
    out = "f1_outputs"
    train_p = os.path.join(out, "combined_train_2022_2023.csv")
    test_p  = os.path.join(out, "combined_test_2024.csv")
    dfs = []
    for p, lbl in [(train_p, "train"), (test_p, "test")]:
        if os.path.exists(p):
            d = pd.read_csv(p); d["split"] = lbl; dfs.append(d)
    if not dfs:
        raise FileNotFoundError("Run run_model.py first.")
    df = pd.concat(dfs, ignore_index=True)
    df["season"]           = df["season"].astype(int)
    df["lap_number"]       = pd.to_numeric(df["lap_number"],  errors="coerce")
    df["lap_time_s"]       = pd.to_numeric(df["lap_time_s"],  errors="coerce")
    df["pitstop_this_lap"] = pd.to_numeric(df["pitstop_this_lap"], errors="coerce")
    df["tire_compound"]    = df["tire_compound"].fillna("UNKNOWN").str.upper()
    return df


def load_model_safe():
    p = "f1_outputs/f1_strategy_model_v2.joblib"
    return joblib.load(p) if os.path.exists(p) else None


df_all = load_corpus()
model  = load_model_safe()


def encode_logo():
    """Base64-encode the UJ logo (PNG preferred, SVG fallback)."""
    for path in ("assets/uj_logo.png", "assets/uj_logo_placeholder.svg"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            mime = "image/svg+xml" if path.endswith(".svg") else "image/png"
            return f"data:{mime};base64,{data}"
    return ""

LOGO_SRC = encode_logo()


# ══════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="F1 Supervised Pit Stop Strategy Model — UJ",
    suppress_callback_exceptions=True,
)
server = app.server   # for production

# ── Global CSS injected via index template ────────────────────
app.index_string = """
<!DOCTYPE html>
<html>
<head>
  {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
  <style>
    body { background:#0E1117; color:#FFFFFF; }
    .uj-banner {
        background: linear-gradient(135deg, #0B1B3D 0%, #112352 100%);
        border-top: 4px solid #C8102E;
        border-bottom: 4px solid #C8102E;
        padding: 18px 28px; border-radius: 6px; margin: 14px 0;
        box-shadow: 0 4px 12px rgba(200,16,46,0.18);
    }
    .uj-logo-box {
        background:#fff; padding:8px; border-radius:6px;
        max-width:140px;
    }
    .uj-eyebrow { color:#fff; font-weight:700; font-size:0.78rem;
                   letter-spacing:1.8px; opacity:0.85; }
    .uj-title { font-size:1.65rem; font-weight:700; color:#fff;
                 margin-top:4px; }
    .uj-sub { color:#C8102E; font-size:0.95rem; font-weight:500;
               letter-spacing:0.5px; margin-top:4px; }
    .uj-meta { color:#bfc8df; font-size:0.8rem; margin-top:6px; }

    .sidebar {
        background: linear-gradient(180deg, #0B1B3D 0%, #050d1f 100%);
        border-right: 2px solid #C8102E;
        padding: 16px; border-radius: 6px; height: 100%;
    }
    .sidebar-logo { background:#fff; padding:8px; border-radius:6px;
                     text-align:center; box-shadow:0 2px 6px rgba(0,0,0,.5); }
    .metric-card {
        background:#0B1B3D; padding: 14px 16px; border-radius: 6px;
        border-left: 3px solid #C8102E; margin-bottom: 10px;
    }
    .metric-label { color:#bfc8df; font-size:0.78rem; letter-spacing:0.8px;
                     text-transform:uppercase; }
    .metric-value { color:#fff; font-size:1.5rem; font-weight:700;
                     margin-top:2px; }
    .Select-control, .Select-menu-outer { background:#0B1B3D !important;
                                            color:#fff !important; }
    .nav-tabs .nav-link.active { color:#C8102E !important;
                                  background:transparent !important;
                                  border-bottom: 3px solid #C8102E !important; }
    .nav-tabs .nav-link { color:#bfc8df !important; }
    .accent-stripe { height:3px; background:#C8102E; border-radius:2px;
                      margin:8px 0 14px 0; }
  </style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>
"""

# ══════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════
SEASONS = sorted(df_all["season"].unique())

def make_dropdown(id_, options, value, label):
    return html.Div([
        html.Label(label, style={"color": WHITE, "fontWeight": 600,
                                 "fontSize": "0.85rem", "marginTop": "10px"}),
        dcc.Dropdown(
            id=id_, options=options, value=value, clearable=False,
            style={"color": INK_TEXT, "marginBottom": "4px"},
        ),
    ])

INK_TEXT = "#111"

# ── Header banner ──────────────────────────────────────────────
header_banner = html.Div(className="uj-banner", children=dbc.Row([
    dbc.Col(html.Img(src=LOGO_SRC, className="uj-logo-box",
                     style={"width": "140px"}), width="auto"),
    dbc.Col(html.Div([
        html.Div("UNIVERSITY OF JOHANNESBURG", className="uj-eyebrow"),
        html.Div("F1 Supervised Pit Stop Strategy Model", className="uj-title"),
        html.Div("A Design Science Research Methodology Artefact",
                 className="uj-sub"),
        html.Div("Master's Research Project by Davies Adetiba · Faculty of Computer Science and Software Engineering",
                 className="uj-meta"),
    ])),
], align="center"))

# ── Sidebar (left column) ──────────────────────────────────────
sidebar = html.Div(className="sidebar", children=[
    html.Div(html.Img(src=LOGO_SRC, style={"width": "140px"}),
             className="sidebar-logo"),
    html.Div("UNIVERSITY OF JOHANNESBURG",
             style={"color": WHITE, "fontWeight": 700, "fontSize": "0.72rem",
                    "letterSpacing": "1.4px", "marginTop": "14px",
                    "textAlign": "center"}),
    html.Div("🏎️ F1 Pit Stop Strategy",
             style={"color": WHITE, "fontWeight": 700, "fontSize": "1.05rem",
                    "marginTop": "8px", "textAlign": "center"}),
    html.Div("Master's Research Project",
             style={"color": RED, "fontSize": "0.78rem", "letterSpacing": "0.5px",
                    "marginTop": "2px", "textAlign": "center", "fontWeight": 600}),
    html.Div("by Davies Adetiba",
             style={"color": WHITE, "fontSize": "0.72rem",
                    "marginTop": "2px", "textAlign": "center", "fontStyle": "italic"}),
    html.Hr(style={"borderColor": RED, "borderWidth": "1px"}),

    make_dropdown("season-dd", [{"label": s, "value": s} for s in SEASONS],
                  SEASONS[-1], "Season"),
    make_dropdown("race-dd",   [], None, "Race"),
    make_dropdown("driver-dd", [], None, "Driver"),

    html.Hr(style={"borderColor": RED, "borderWidth": "1px"}),
    html.Div(id="race-summary",
             style={"color": MUTED, "fontSize": "0.78rem"}),
])

# ── Tabs ───────────────────────────────────────────────────────
tabs = dbc.Tabs(id="main-tabs", active_tab="tab-strategy", children=[
    dbc.Tab(label="🏁 Lap Times & Strategy", tab_id="tab-strategy"),
    dbc.Tab(label="🔮 Pit Stop Probability", tab_id="tab-prob"),
    dbc.Tab(label="🛞 Tyre Analysis",        tab_id="tab-tyre"),
    dbc.Tab(label="📊 Model Performance",     tab_id="tab-model"),
    dbc.Tab(label="🌦️ Race Conditions",      tab_id="tab-weather"),
])

# ── Main page ─────────────────────────────────────────────────
app.layout = dbc.Container(fluid=True, children=[
    header_banner,
    html.Div(className="accent-stripe"),

    dbc.Row([
        dbc.Col(sidebar, width=3),
        dbc.Col([
            html.Div(id="metric-row"),
            html.Hr(style={"borderColor": "#222"}),
            tabs,
            html.Div(id="tab-content", style={"marginTop": "12px"}),
        ], width=9),
    ]),

    html.Hr(style={"borderColor": "#222", "marginTop": "30px"}),
    html.Div(
        "F1 Supervised Pit Stop Strategy Model · University of Johannesburg · "
        "Design Science Research Methodology (Peffers et al., 2007) · "
        "Data: FastF1 / Tracing Insights · "
        "Model: GradientBoostingClassifier + RandomForestClassifier",
        style={"color": MUTED, "fontSize": "0.75rem", "padding": "10px 0",
               "textAlign": "center"},
    ),
])


# ══════════════════════════════════════════════════════════════
# CALLBACKS — cascading filter chain
# ══════════════════════════════════════════════════════════════
@callback(
    Output("race-dd", "options"),
    Output("race-dd", "value"),
    Input("season-dd", "value"),
)
def update_races(season):
    sub = df_all[df_all["season"] == season]
    races = (sub[["round_number", "race_name"]].drop_duplicates()
                .sort_values("round_number"))
    opts = [{"label": r["race_name"], "value": r["race_name"]} for _, r in races.iterrows()]
    return opts, opts[0]["value"] if opts else None


@callback(
    Output("driver-dd", "options"),
    Output("driver-dd", "value"),
    Input("season-dd", "value"),
    Input("race-dd",   "value"),
)
def update_drivers(season, race):
    sub = df_all[(df_all["season"] == season) & (df_all["race_name"] == race)]
    drivers = sorted(sub["driver_id"].dropna().unique())
    opts = [{"label": d, "value": d} for d in drivers]
    return opts, opts[0]["value"] if opts else None


@callback(
    Output("race-summary", "children"),
    Input("season-dd", "value"),
    Input("race-dd",   "value"),
)
def race_stats(season, race):
    sub = df_all[(df_all["season"] == season) & (df_all["race_name"] == race)]
    if sub.empty: return ""
    n      = len(sub)
    n_pit  = int(sub["pitstop_this_lap"].sum())
    split  = sub["split"].iloc[0]
    badge_color = RED if split == "test" else WHITE
    badge_text  = "🟢 Test set" if split == "test" else "🔵 Train set"
    return html.Div([
        html.Div(f"Split: {badge_text}",
                 style={"color": badge_color, "fontWeight": 600}),
        html.Div(f"Race laps: {n:,}"),
        html.Div(f"Pit stops in dataset: {n_pit}"),
    ])


# ── Top metrics row ───────────────────────────────────────────
@callback(
    Output("metric-row", "children"),
    Input("season-dd", "value"),
    Input("race-dd",   "value"),
    Input("driver-dd", "value"),
)
def render_metrics(season, race, driver):
    sub = df_all[(df_all["season"] == season) &
                 (df_all["race_name"] == race) &
                 (df_all["driver_id"] == driver)].sort_values("lap_number")
    if sub.empty:
        return html.Div("No data for selection.", style={"color": MUTED})

    total_laps = int(sub["lap_number"].max())
    pit_laps   = int(sub["pitstop_this_lap"].sum())
    med_lap    = sub["lap_time_s"].median()
    best_lap   = sub["lap_time_s"].min()
    comps      = ", ".join(sorted(set(sub["tire_compound"]) - {"UNKNOWN"}))

    def card(label, value):
        return dbc.Col(html.Div([
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value"),
        ], className="metric-card"), width=12//5)

    return dbc.Row([
        card("TOTAL LAPS",    f"{total_laps}"),
        card("PIT STOPS",     f"{pit_laps}"),
        card("MEDIAN LAP",    f"{med_lap:.3f}s"),
        card("FASTEST LAP",   f"{best_lap:.3f}s"),
        card("COMPOUNDS",     comps),
    ])


# ── Tab content router ────────────────────────────────────────
@callback(
    Output("tab-content", "children"),
    Input("main-tabs",  "active_tab"),
    Input("season-dd",  "value"),
    Input("race-dd",    "value"),
    Input("driver-dd",  "value"),
)
def render_tab(tab, season, race, driver):
    sub_race = df_all[(df_all["season"] == season) & (df_all["race_name"] == race)]
    sub_drv  = sub_race[sub_race["driver_id"] == driver].sort_values("lap_number")

    if tab == "tab-strategy":
        return tab_strategy(sub_drv)
    if tab == "tab-prob":
        return tab_probability(sub_drv)
    if tab == "tab-tyre":
        return tab_tyre(sub_drv, sub_race)
    if tab == "tab-model":
        return tab_model()
    if tab == "tab-weather":
        return tab_weather(sub_drv)
    return html.Div()


# ══════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════
def _layout_dark(fig, height=480):
    fig.update_layout(
        height=height, template="plotly_dark",
        paper_bgcolor=NAVY, plot_bgcolor=NAVY,
        font=dict(color=WHITE),
        margin=dict(l=40, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


# ── TAB 1: Lap Times & Strategy ────────────────────────────────
def tab_strategy(df_drv):
    if df_drv.empty:
        return dbc.Alert("No data for selected driver.", color="warning")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=["Lap Time Evolution", "Tyre Compound per Lap"],
        vertical_spacing=0.08,
    )
    for cmp in df_drv["tire_compound"].unique():
        s = df_drv[df_drv["tire_compound"] == cmp]
        fig.add_trace(go.Scatter(
            x=s["lap_number"], y=s["lap_time_s"], name=cmp,
            mode="lines+markers",
            line=dict(color=COMPOUND_COLORS.get(cmp, "#888"), width=2),
            marker=dict(size=5),
        ), row=1, col=1)

    pit_laps = df_drv[df_drv["pitstop_this_lap"] == 1]["lap_number"]
    for lp in pit_laps:
        fig.add_vline(x=lp, line_dash="dot", line_color=RED,
                      line_width=1.5, row=1, col=1)

    for cmp in df_drv["tire_compound"].unique():
        s = df_drv[df_drv["tire_compound"] == cmp]
        fig.add_trace(go.Bar(
            x=s["lap_number"], y=[1]*len(s), name=cmp,
            marker_color=COMPOUND_COLORS.get(cmp, "#888"),
            showlegend=False,
        ), row=2, col=1)

    fig.update_yaxes(title_text="Lap Time (s)", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(title_text="Lap Number", row=2, col=1)
    fig = _layout_dark(fig, height=520)

    # Stint summary
    children = [dcc.Graph(figure=fig, config={"displayModeBar": False})]
    if "stint_number" in df_drv.columns:
        stints = (df_drv.groupby("stint_number")
                  .agg(Compound=("tire_compound", "first"),
                       Start_Lap=("lap_number", "min"),
                       End_Lap=("lap_number", "max"),
                       Laps=("lap_number", "count"),
                       Avg_Lap_s=("lap_time_s", "median"),
                       Fresh=("fresh_tyre", "first"))
                  .reset_index().rename(columns={"stint_number": "Stint"}))
        stints["Avg_Lap_s"] = stints["Avg_Lap_s"].round(3)
        stints["Fresh"]     = stints["Fresh"].map({1: "✅ New", 0: "♻️ Scrubbed"})

        children += [
            html.H5("Stint Summary",
                    style={"color": WHITE, "marginTop": "16px"}),
            dbc.Table.from_dataframe(stints, striped=True, bordered=False,
                                     hover=True, dark=True,
                                     style={"backgroundColor": NAVY}),
        ]
    return children


# ── TAB 2: Pit Stop Probability ────────────────────────────────
def tab_probability(df_drv):
    if model is None:
        return dbc.Alert("Model not loaded.", color="warning")
    if df_drv.empty:
        return dbc.Alert("No data for selected driver.", color="warning")

    try:
        df_eng = engineer_features(df_drv.copy())

        pit_model = (model.pitstop_model
                     if hasattr(model, "pitstop_model")
                     else model)
        expected = (list(model.feature_names)
                    if hasattr(model, "feature_names") and model.feature_names is not None
                    else [f for f in ALL_FEATURES if f in df_eng.columns])
        for col in expected:
            if col not in df_eng.columns: df_eng[col] = 0
        X = df_eng[expected].fillna(0)
        if hasattr(model, "scaler") and model.scaler is not None:
            try: X = model.scaler.transform(X)
            except Exception: pass
        df_eng["pit_proba"] = pit_model.predict_proba(X)[:, 1]

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.65, 0.35],
                            subplot_titles=["Pit Stop Probability",
                                            "Actual Pit Stops"],
                            vertical_spacing=0.08)
        fig.add_trace(go.Scatter(
            x=df_eng["lap_number"], y=df_eng["pit_proba"],
            mode="lines", name="Pit Probability",
            line=dict(color=RED, width=2),
            fill="tozeroy", fillcolor="rgba(200,16,46,0.18)",
        ), row=1, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color=WHITE,
                      annotation_text="50% threshold",
                      annotation_font_color=WHITE, row=1, col=1)

        actual = df_eng[df_eng["pitstop_this_lap"] == 1]
        fig.add_trace(go.Bar(
            x=actual["lap_number"], y=[1]*len(actual),
            name="Actual Pit Stop", marker_color="#39B54A",
        ), row=2, col=1)

        fig.update_yaxes(title_text="Probability", range=[0, 1], row=1, col=1)
        fig.update_xaxes(title_text="Lap Number", row=2, col=1)
        fig = _layout_dark(fig, height=460)

        # Threshold slider + table
        return [
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            html.Div([
                html.Label("Probability threshold",
                           style={"color": WHITE, "fontWeight": 600,
                                  "marginRight": "12px"}),
                dcc.Slider(id="prob-threshold", min=0.1, max=0.9,
                           step=0.05, value=0.5,
                           marks={i/10: str(i/10) for i in range(1, 10)}),
            ], style={"marginTop": "12px"}),
            dcc.Store(id="proba-store", data=df_eng[
                ["lap_number", "pit_proba", "tire_compound", "tire_age"]
            ].to_dict("records")),
            html.Div(id="proba-windows", style={"marginTop": "10px"}),
        ]
    except Exception as e:
        return dbc.Alert(f"Prediction failed: {e}", color="danger")


@callback(
    Output("proba-windows", "children"),
    Input("prob-threshold", "value"),
    State("proba-store",   "data"),
)
def filter_proba(threshold, data):
    if not data: return ""
    df = pd.DataFrame(data)
    w  = df[df["pit_proba"] >= threshold]
    if w.empty:
        return dbc.Alert(f"No laps above {threshold:.0%}.", color="info")
    w["pit_proba"] = w["pit_proba"].round(3)
    return [
        dbc.Alert(f"{len(w)} laps exceed the {threshold:.0%} threshold.",
                  color="success"),
        dbc.Table.from_dataframe(w.reset_index(drop=True),
                                 striped=True, hover=True, dark=True,
                                 bordered=False),
    ]


# ── TAB 3: Tyre Analysis ───────────────────────────────────────
def tab_tyre(df_drv, df_race):
    if df_drv.empty:
        return dbc.Alert("No data.", color="warning")

    # Degradation chart
    fig1 = go.Figure()
    for cmp in df_drv["tire_compound"].unique():
        s = df_drv[df_drv["tire_compound"] == cmp]
        fig1.add_trace(go.Scatter(
            x=s["tire_age"], y=s["tire_degradation"],
            name=cmp, mode="markers+lines",
            line=dict(color=COMPOUND_COLORS.get(cmp, "#888")),
        ))
    fig1.update_layout(title="Tyre Degradation",
                       xaxis_title="Tyre Age (laps)",
                       yaxis_title="Degradation [0–1]")
    fig1 = _layout_dark(fig1, height=340)

    # Compound usage bar
    counts = df_race["tire_compound"].value_counts().reset_index()
    counts.columns = ["Compound", "Laps"]
    fig2 = px.bar(counts, x="Compound", y="Laps", color="Compound",
                  color_discrete_map=COMPOUND_COLORS, title="Compound Usage")
    fig2 = _layout_dark(fig2, height=340)
    fig2.update_layout(showlegend=False)

    # All drivers heatmap
    pivot = df_race.pivot_table(index="driver_id", columns="lap_number",
                                values="tire_compound", aggfunc="first")
    comp_int = {"SOFT":0, "MEDIUM":1, "HARD":2, "INTER":3, "WET":4, "UNKNOWN":5}
    pivot_int = pivot.replace(comp_int)
    fig3 = go.Figure(data=go.Heatmap(
        z=pivot_int.values, x=pivot_int.columns.tolist(),
        y=pivot_int.index.tolist(),
        colorscale=[
            [0.0, "#E8002D"], [0.2, "#FFF200"], [0.4, "#EEEEEE"],
            [0.6, "#39B54A"], [0.8, "#0067FF"], [1.0, "#555"],
        ], showscale=False,
    ))
    fig3.update_layout(title="Tyre Strategy — All Drivers",
                       xaxis_title="Lap Number", yaxis_title="Driver")
    fig3 = _layout_dark(fig3, height=420)

    return [
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig1, config={"displayModeBar": False}),
                    width=6),
            dbc.Col(dcc.Graph(figure=fig2, config={"displayModeBar": False}),
                    width=6),
        ]),
        dcc.Graph(figure=fig3, config={"displayModeBar": False}),
        html.Div("🔴 SOFT  🟡 MEDIUM  ⬜ HARD  🟢 INTER  🔵 WET",
                 style={"color": MUTED, "fontSize": "0.8rem",
                        "textAlign": "center"}),
    ]


# ── TAB 4: Model Performance ───────────────────────────────────
def tab_model():
    def img(path, caption):
        if not os.path.exists(path):
            return html.Div(f"Missing: {path}", style={"color": MUTED})
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return html.Div([
            html.H6(caption, style={"color": WHITE}),
            html.Img(src=f"data:image/png;base64,{data}",
                     style={"width": "100%", "borderRadius": "4px"}),
        ])

    def metric(label, value, help_=""):
        return dbc.Col(html.Div([
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value"),
            html.Div(help_, style={"color": MUTED, "fontSize": "0.7rem"}),
        ], className="metric-card"))

    return html.Div([
        html.H5("Evaluation — 2024 Test Set",
                style={"color": WHITE, "marginBottom": "8px"}),
        html.Div("Trained on 2022 + 2023 · Evaluated on unseen 2024 season",
                 style={"color": MUTED, "fontSize": "0.8rem",
                        "marginBottom": "12px"}),
        dbc.Row([
            metric("MCC",         "0.167", "Primary — handles imbalance"),
            metric("G-MEAN",      "0.694", "Sensitivity × specificity geometric mean"),
            metric("ROC-AUC",     "0.805", "Probability ranking quality"),
            metric("SENSITIVITY", "60.4%", "Pit stops caught"),
            metric("SPECIFICITY", "79.7%", "Non-pit laps dismissed"),
        ]),
        html.Hr(style={"borderColor": "#222", "margin": "18px 0"}),
        dbc.Row([
            dbc.Col(img("f1_outputs/feature_importance.png",
                        "Feature Importance"), width=6),
            dbc.Col(img("f1_outputs/confusion_matrices.png",
                        "Confusion Matrix"), width=6),
        ]),
        dbc.Row([
            dbc.Col(img("f1_outputs/roc_pr_curves_Test.png",
                        "ROC & PR Curves"), width=6),
            dbc.Col(img("f1_outputs/mcc_gmean_Test.png",
                        "MCC / G-mean Summary"), width=6),
        ]),
    ])


# ── TAB 5: Race Conditions ─────────────────────────────────────
def tab_weather(df_drv):
    if df_drv.empty:
        return dbc.Alert("No data.", color="warning")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
        subplot_titles=["Temperature (°C)", "Humidity & Wind",
                        "Safety Car / Rain"],
        vertical_spacing=0.08)

    if "track_temp_c" in df_drv.columns:
        fig.add_trace(go.Scatter(x=df_drv["lap_number"],
            y=df_drv["track_temp_c"], name="Track Temp",
            line=dict(color="#FF6B35", width=2)), row=1, col=1)
    if "air_temp_c" in df_drv.columns:
        fig.add_trace(go.Scatter(x=df_drv["lap_number"],
            y=df_drv["air_temp_c"], name="Air Temp",
            line=dict(color="#FFD700", width=2, dash="dot")), row=1, col=1)
    if "humidity_pct" in df_drv.columns:
        fig.add_trace(go.Scatter(x=df_drv["lap_number"],
            y=df_drv["humidity_pct"], name="Humidity %",
            line=dict(color="#00D2BE", width=2)), row=2, col=1)
    if "wind_speed_ms" in df_drv.columns:
        fig.add_trace(go.Scatter(x=df_drv["lap_number"],
            y=df_drv["wind_speed_ms"], name="Wind (m/s)",
            line=dict(color="#9B59B6", width=2, dash="dot")), row=2, col=1)
    if "safety_car_active" in df_drv.columns:
        fig.add_trace(go.Bar(x=df_drv["lap_number"],
            y=df_drv["safety_car_active"], name="Safety Car",
            marker_color=RED), row=3, col=1)
    if "rainfall_mm" in df_drv.columns:
        fig.add_trace(go.Bar(x=df_drv["lap_number"],
            y=df_drv["rainfall_mm"], name="Rainfall",
            marker_color="#0067FF"), row=3, col=1)

    fig = _layout_dark(fig, height=520)

    def stat_card(label, value):
        return dbc.Col(html.Div([
            html.Div(label, className="metric-label"),
            html.Div(value, className="metric-value"),
        ], className="metric-card"), width=3)

    cards = [
        stat_card("AVG TRACK TEMP",
                  f"{df_drv['track_temp_c'].mean():.1f}°C" if 'track_temp_c' in df_drv.columns else "N/A"),
        stat_card("AVG HUMIDITY",
                  f"{df_drv['humidity_pct'].mean():.1f}%" if 'humidity_pct' in df_drv.columns else "N/A"),
        stat_card("MAX WIND",
                  f"{df_drv['wind_speed_ms'].max():.1f} m/s" if 'wind_speed_ms' in df_drv.columns else "N/A"),
        stat_card("SC LAPS",
                  f"{int(df_drv['safety_car_active'].sum())}" if 'safety_car_active' in df_drv.columns else "N/A"),
    ]

    return [
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        html.H5("Conditions Summary",
                style={"color": WHITE, "marginTop": "14px"}),
        dbc.Row(cards),
    ]


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
