"""
F1 Pit Stop Dashboard — FastAPI Backend
Serves Tracing Insights CSV data + proxies Jolpica API
Run: ./venv/bin/uvicorn api:app --reload --port 8000
"""

import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Allow importing the strategy model from the same directory
sys.path.insert(0, str(Path(__file__).parent))
try:
    import joblib
    from f1_strategy_model_2 import engineer_features, ALL_FEATURES, _extract_pitstop_windows, _identify_factors
    _MODEL_AVAILABLE = True
except ImportError:
    _MODEL_AVAILABLE = False

app = FastAPI(title="F1 Pit Stop API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TI_DIR     = Path(__file__).parent / "data" / "tracing_insights"
KAG_DIR    = Path(__file__).parent / "data" / "kaggle"
MODEL_PATH = Path(__file__).parent / "f1_outputs" / "f1_strategy_model_v2.joblib"

# ── cache loaded DataFrames and the ML model in memory ────────
_race_cache: dict[str, pd.DataFrame] = {}
_model = None   # lazy-loaded F1StrategyModel


def _load_model():
    global _model
    if _model is None:
        if not _MODEL_AVAILABLE:
            raise HTTPException(status_code=503, detail="Strategy model module not available")
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail=f"Model not found at {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def _ti_files() -> list[dict]:
    files = sorted(glob.glob(str(TI_DIR / "*.csv")))
    races = []
    for f in files:
        name = Path(f).stem
        m = re.match(r"(\d{4})_(\d{2})_(.*)", name)
        if m:
            races.append({
                "season": int(m.group(1)),
                "round": int(m.group(2)),
                "name": m.group(3).replace("_", " "),
                "file": f,
            })
    return races


def _load_ti(season: int, round_number: int) -> pd.DataFrame:
    key = f"{season}_{round_number:02d}"
    if key not in _race_cache:
        all_files = _ti_files()
        match = next((r for r in all_files if r["season"] == season and r["round"] == round_number), None)
        if not match:
            raise HTTPException(status_code=404, detail=f"Race {season} round {round_number} not found")
        df = pd.read_csv(match["file"])
        # normalise NaN → None for JSON
        df = df.where(pd.notnull(df), None)
        _race_cache[key] = df
    return _race_cache[key]


# ── Kaggle helpers ─────────────────────────────────────────────
_kag_pitstops: pd.DataFrame | None = None
_kag_races:    pd.DataFrame | None = None


def _kag_pit() -> pd.DataFrame:
    global _kag_pitstops
    if _kag_pitstops is None:
        _kag_pitstops = pd.read_csv(KAG_DIR / "pit_stops.csv")
    return _kag_pitstops


def _kag_races_df() -> pd.DataFrame:
    global _kag_races
    if _kag_races is None:
        _kag_races = pd.read_csv(KAG_DIR / "races.csv")
    return _kag_races


# ──────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────

@app.get("/races")
def list_races():
    return [{"season": r["season"], "round": r["round"], "name": r["name"]} for r in _ti_files()]


@app.get("/races/{season}")
def list_season_races(season: int):
    return [r for r in list_races() if r["season"] == season]


@app.get("/seasons")
def list_seasons():
    seasons = sorted(set(r["season"] for r in _ti_files()))
    return seasons


@app.get("/race/{season}/{round_number}/drivers")
def get_drivers(season: int, round_number: int):
    df = _load_ti(season, round_number)
    drivers = sorted(df["driver_id"].unique().tolist())
    return drivers


@app.get("/race/{season}/{round_number}/laps")
def get_laps(season: int, round_number: int):
    df = _load_ti(season, round_number)
    cols = [
        "lap_number", "driver_id", "team_id", "tire_compound", "tire_age",
        "tire_degradation", "stint_number", "lap_time_s", "position",
        "safety_car_active", "pitstop_this_lap", "gap_ahead_s", "gap_behind_s",
        "track_temp_c", "rainfall_mm",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_dict(orient="records")


@app.get("/race/{season}/{round_number}/stints")
def get_stints(season: int, round_number: int):
    df = _load_ti(season, round_number)

    stints = []
    for (driver, stint), grp in df.groupby(["driver_id", "stint_number"], sort=False):
        grp = grp.sort_values("lap_number")
        compound = grp["tire_compound"].mode().iloc[0] if not grp.empty else "UNKNOWN"
        stints.append({
            "driver":    driver,
            "team":      grp["team_id"].iloc[0] if "team_id" in grp.columns else "",
            "stint":     int(stint),
            "lap_start": int(grp["lap_number"].min()),
            "lap_end":   int(grp["lap_number"].max()),
            "laps":      int(len(grp)),
            "compound":  compound,
            "avg_lap_s": round(float(grp["lap_time_s"].mean()), 3) if "lap_time_s" in grp.columns else None,
            "best_lap_s": round(float(grp["lap_time_s"].min()), 3) if "lap_time_s" in grp.columns else None,
            "deg_end":   round(float(grp["tire_degradation"].iloc[-1]), 4) if "tire_degradation" in grp.columns else None,
        })
    return stints


@app.get("/race/{season}/{round_number}/pitstops")
def get_pitstops(season: int, round_number: int):
    df = _load_ti(season, round_number)
    ps = df[df["pitstop_this_lap"] == 1].copy()
    result = ps[["lap_number", "driver_id", "team_id", "tire_compound", "next_compound", "position"]].to_dict(orient="records")
    return result


@app.get("/race/{season}/{round_number}/summary")
def get_summary(season: int, round_number: int):
    df = _load_ti(season, round_number)

    total_laps = int(df["lap_number"].max())
    drivers    = sorted(df["driver_id"].unique().tolist())
    stops_per_driver = df[df["pitstop_this_lap"] == 1].groupby("driver_id").size().to_dict()
    sc_laps    = int(df[df["safety_car_active"] == 1]["lap_number"].nunique()) if "safety_car_active" in df.columns else 0

    # Fastest lap overall
    fl_row = df.loc[df["lap_time_s"].idxmin()] if "lap_time_s" in df.columns else None
    fastest_lap = None
    if fl_row is not None:
        fastest_lap = {
            "driver": fl_row["driver_id"],
            "lap":    int(fl_row["lap_number"]),
            "time_s": round(float(fl_row["lap_time_s"]), 3),
        }

    compounds_used = sorted(df["tire_compound"].dropna().unique().tolist()) if "tire_compound" in df.columns else []

    # Kaggle pit stop durations if available
    kaggle_durations = {}
    try:
        races_df = _kag_races_df()
        pits_df  = _kag_pit()
        match_race = races_df[(races_df["year"] == season) & (races_df["round"] == round_number)]
        if not match_race.empty:
            race_id = int(match_race.iloc[0]["raceId"])
            rps = pits_df[pits_df["raceId"] == race_id]
            if not rps.empty:
                # duration is a string like "25.021", map driverId → avg duration
                rps = rps.copy()
                rps["dur_f"] = pd.to_numeric(rps["duration"], errors="coerce")
                kaggle_durations = rps.groupby("driverId")["dur_f"].mean().round(3).to_dict()
    except Exception:
        pass

    return {
        "season":          season,
        "round":           round_number,
        "race_name":       df["race_name"].iloc[0] if "race_name" in df.columns else "",
        "total_laps":      total_laps,
        "drivers":         drivers,
        "stops_per_driver": stops_per_driver,
        "sc_laps":         sc_laps,
        "fastest_lap":     fastest_lap,
        "compounds_used":  compounds_used,
        "avg_pit_duration_s": kaggle_durations,
    }


@app.get("/race/{season}/{round_number}/positions")
def get_positions(season: int, round_number: int):
    df = _load_ti(season, round_number)
    if "position" not in df.columns:
        return []
    pos = df[["lap_number", "driver_id", "position"]].dropna()
    return pos.to_dict(orient="records")


@app.get("/race/{season}/{round_number}/conditions")
def get_conditions(season: int, round_number: int):
    df = _load_ti(season, round_number)
    cols = ["lap_number", "track_temp_c", "air_temp_c", "rainfall_mm", "humidity_pct", "wind_speed_ms", "safety_car_active"]
    cols = [c for c in cols if c in df.columns]
    cond = df[cols].drop_duplicates("lap_number").sort_values("lap_number")
    return cond.to_dict(orient="records")


@app.get("/race/{season}/{round_number}/predictions")
def get_predictions(season: int, round_number: int):
    """
    Run the trained GBC model on a race and return per-driver
    pit stop probabilities, predicted windows, compound recommendations,
    and risk/opportunity factors.
    """
    model = _load_model()
    raw   = _load_ti(season, round_number)

    # engineer_features mutates a copy — fill NaN with sensible defaults
    df = raw.copy()
    df["rainfall_mm"]      = df["rainfall_mm"].fillna(0)
    df["humidity_pct"]     = df.get("humidity_pct", pd.Series(50, index=df.index)).fillna(50)
    df["wind_speed_ms"]    = df.get("wind_speed_ms", pd.Series(0,  index=df.index)).fillna(0)
    df["safety_car_active"]= df["safety_car_active"].fillna(0)
    df["gap_ahead_s"]      = df["gap_ahead_s"].fillna(0)
    df["gap_behind_s"]     = df["gap_behind_s"].fillna(0)
    df["fresh_tyre"]       = df.get("fresh_tyre", pd.Series(1, index=df.index)).fillna(1)

    # engineer_features adds tire_compound_enc + rolling features
    df = engineer_features(df)
    df = df.dropna(subset=ALL_FEATURES)

    drivers = sorted(df["driver_id"].unique().tolist())

    # ── Per-driver inference ───────────────────────────────────
    by_driver = {}
    all_factors = []

    for driver in drivers:
        ddf = df[df["driver_id"] == driver].sort_values("lap_number").copy()
        if len(ddf) < 3:
            continue

        X = model.scaler.transform(ddf[ALL_FEATURES].values)

        pit_probs  = model.pitstop_model.predict_proba(X)[:, 1]
        pit_preds  = model.pitstop_model.predict(X)
        comp_idx   = model.compound_model.predict(X)
        comp_recs  = model.compound_le.inverse_transform(comp_idx)

        ddf = ddf.copy()
        ddf["pitstop_prob"]  = pit_probs
        ddf["pitstop_pred"]  = pit_preds
        ddf["compound_rec"]  = comp_recs

        windows = _extract_pitstop_windows(ddf, threshold=0.40)
        factors = _identify_factors(ddf)

        # Actual pit laps (ground truth)
        actual_pits = ddf[ddf["pitstop_this_lap"] == 1]["lap_number"].astype(int).tolist()

        # Per-lap probability series (compact for JSON)
        prob_series = [
            {
                "lap": int(row["lap_number"]),
                "prob": round(float(row["pitstop_prob"]), 3),
                "pred": int(row["pitstop_pred"]),
                "comp_rec": row["compound_rec"],
            }
            for _, row in ddf.iterrows()
        ]

        # Accuracy against actual
        actual_set = set(actual_pits)
        pred_set   = set(ddf[ddf["pitstop_pred"] == 1]["lap_number"].astype(int).tolist())
        tp = len(actual_set & pred_set)
        fp = len(pred_set - actual_set)
        fn = len(actual_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        by_driver[driver] = {
            "prob_series":    prob_series,
            "windows":        windows,
            "actual_pits":    actual_pits,
            "accuracy": {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(precision, 3),
                "recall":    round(recall, 3),
            },
        }

        for f in factors:
            f["driver"] = driver
            all_factors.append(f)

    # Deduplicate race-level factors (e.g. rain appears once)
    seen_factor_types = set()
    deduped_factors = []
    for f in all_factors:
        key = f["factor"]
        if key not in seen_factor_types:
            seen_factor_types.add(key)
            deduped_factors.append(f)

    # Model metadata
    meta = {
        "model_available": True,
        "feature_count": len(ALL_FEATURES),
        "threshold": 0.40,
    }
    if hasattr(model, "train_metrics") and model.train_metrics:
        m = model.train_metrics
        meta["train_mcc"]   = round(float(m.get("mcc",   0)), 4)
        meta["train_gmean"] = round(float(m.get("gmean", 0)), 4)
        meta["train_f1"]    = round(float(m.get("f1",    0)), 4)

    return {
        "season":    season,
        "round":     round_number,
        "drivers":   drivers,
        "by_driver": by_driver,
        "factors":   deduped_factors,
        "meta":      meta,
    }
