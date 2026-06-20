"""
============================================================
F1 STRATEGY MODEL — FastF1 DATA CONNECTOR  v1.0
============================================================
Pulls live race data from the official FastF1 API and
transforms it into the REQUIRED_COLUMNS schema used by
f1_strategy_model_2.py

Key responsibilities:
  - Cache setup and management
  - Single race / full season loading
  - Raw FastF1 → model schema transformation:
      LapNumber     → lap_number
      Driver/Team   → driver_id / team_id
      Compound      → tire_compound (SOFT/MEDIUM/HARD/INTER/WET)
      TyreLife      → tire_age
      LapTime       → lap_time_s
      [computed]    → fuel_load_kg
      [computed]    → tire_degradation
      weather merge → track_temp_c / air_temp_c / rainfall_mm
      TrackStatus   → safety_car_active
      Position      → position
      [computed]    → gap_ahead_s / gap_behind_s
      PitInTime     → pitstop_this_lap
      [look-ahead]  → next_compound

Usage:
  from fastf1_connector import build_corpus
  df_train, df_test = build_corpus(
      train_years=(2022,),
      test_years=(2023, 2024),
  )
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import fastf1
    import fastf1.events
except ImportError:
    raise ImportError(
        "FastF1 is not installed.\n"
        "Run:  pip install fastf1>=3.3.0\n"
        "Or:   pip install -r requirements.txt"
    )


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

# FastF1 uses 'INTERMEDIATE' — normalise to 'INTER' for our schema
COMPOUND_NORMALISE = {
    "SOFT":         "SOFT",
    "MEDIUM":       "MEDIUM",
    "HARD":         "HARD",
    "INTERMEDIATE": "INTER",
    "INTER":        "INTER",
    "WET":          "WET",
    "UNKNOWN":      "HARD",       # mapped to HARD as fallback
    "TEST_UNKNOWN": "HARD",
    "HYPERSOFT":    "SOFT",
    "ULTRASOFT":    "SOFT",
    "SUPERSOFT":    "SOFT",
    "SUPERHARD":    "HARD",
}

# Maximum expected stint life per compound (laps) — used to normalise
# tire_degradation to the [0, 1] interval.
# Based on typical 2022–2025 Pirelli data; conservative upper bounds.
MAX_STINT_LIFE = {
    "SOFT":   28,
    "MEDIUM": 38,
    "HARD":   52,
    "INTER":  35,
    "WET":    45,
}

# Starting fuel load and per-lap consumption estimate (kg)
# Official F1 rules limit 110 kg of fuel per race.
# Actual consumption ~1.8–2.1 kg/lap; we use 1.95 as a mid-range estimate.
FUEL_START_KG   = 107.0
FUEL_PER_LAP_KG = 1.95

# TrackStatus values that indicate Safety Car or Virtual Safety Car
SC_STATUSES = {"4", "6"}   # '4' = SC deployed, '6' = VSC deployed

# Lap time sanity bounds (seconds) — drop outliers / formation laps
LAP_TIME_MIN_S = 60.0
LAP_TIME_MAX_S = 250.0

# Cap gap to next/previous car at this value (avoids lapped-car artefacts)
GAP_CAP_S = 60.0


# ─────────────────────────────────────────────────────────────
# 1.  CACHE SETUP
# ─────────────────────────────────────────────────────────────

def setup_cache(cache_dir: str = "./f1_cache") -> None:
    """
    Enable FastF1's local disk cache.

    On first access FastF1 downloads each session (~5–20 MB each).
    After that, every subsequent load is instant from cache.
    The 2022 full season uses roughly 300–500 MB of disk space.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    print(f"  ✔  FastF1 cache active → {os.path.abspath(cache_dir)}")


# ─────────────────────────────────────────────────────────────
# 2.  SINGLE SESSION LOADER
# ─────────────────────────────────────────────────────────────

def load_race_session(
    year: int,
    round_number: int,
    cache_dir: str = "./f1_cache",
    verbose: bool = False,
) -> "fastf1.core.Session":
    """
    Load a single race session from the FastF1 API.

    Parameters
    ----------
    year         : Championship year (e.g. 2022)
    round_number : Round number within the season (1-based)
    cache_dir    : Path for FastF1's local cache
    verbose      : Print FastF1 internal logs

    Returns
    -------
    fastf1.core.Session — fully loaded race session
    """
    setup_cache(cache_dir)

    if not verbose:
        import logging
        logging.getLogger("fastf1").setLevel(logging.ERROR)

    session = fastf1.get_session(year, round_number, "R")
    # Load laps + weather; skip heavy telemetry (throttle/RPM/brakes)
    # to keep download size and memory usage small.
    # Set telemetry=True later if you want RPM/throttle features in v3.
    session.load(
        telemetry=False,
        weather=True,
        messages=False,
        livedata=None,
    )
    return session


# ─────────────────────────────────────────────────────────────
# 3.  SESSION → MODEL SCHEMA TRANSFORMER
# ─────────────────────────────────────────────────────────────

def _safe_total_seconds(td) -> float:
    """Convert Timedelta → float seconds; return NaN if not Timedelta."""
    try:
        return td.total_seconds()
    except AttributeError:
        return float("nan")


def _normalise_compound(raw) -> str:
    """Map FastF1 compound string to our 5-class schema."""
    if pd.isna(raw):
        return "HARD"
    return COMPOUND_NORMALISE.get(str(raw).upper().strip(), "HARD")


def _compute_gaps(group: pd.DataFrame) -> pd.DataFrame:
    """
    Within a single lap number, approximate gap_ahead_s / gap_behind_s
    using the session timestamp at which each driver completed that lap
    (FastF1 laps.Time) sorted by race position.

    This gives the real time gap between cars AT THE MOMENT of crossing
    the finish line on that lap — accurate for cars on the same lap count;
    capped at GAP_CAP_S to reduce artefacts from lapped cars.
    """
    group = group.sort_values("position").copy()
    times = group["_time_s"].values
    n = len(times)

    ahead  = np.zeros(n)
    behind = np.zeros(n)

    for i in range(n):
        if i > 0:
            diff = times[i] - times[i - 1]
            ahead[i]  = float(np.clip(diff, 0, GAP_CAP_S))
        if i < n - 1:
            diff = times[i + 1] - times[i]
            behind[i] = float(np.clip(diff, 0, GAP_CAP_S))

    group["gap_ahead_s"]  = ahead
    group["gap_behind_s"] = behind
    return group


def session_to_dataframe(session) -> pd.DataFrame:
    """
    Convert a loaded FastF1 Session into the REQUIRED_COLUMNS DataFrame.

    Steps
    -----
    1.  Extract and clean the laps DataFrame
    2.  Normalise tire compound labels
    3.  Compute tire degradation (TyreLife / MAX_STINT_LIFE)
    4.  Estimate fuel load (linear decay from race start)
    5.  Derive safety_car_active from TrackStatus
    6.  Merge nearest-in-time weather readings (temperature, rainfall)
    7.  Compute inter-car gap estimates from session timing
    8.  Derive pitstop_this_lap and next_compound labels
    9.  Select and return REQUIRED_COLUMNS only

    Parameters
    ----------
    session : fastf1.core.Session — already loaded

    Returns
    -------
    pd.DataFrame with exactly REQUIRED_COLUMNS + [year, round, gp_name]
    """

    # ── Step 1: Base laps extraction and cleaning ──────────────────────
    laps = session.laps.copy()

    # Drop rows with no lap number or NaN lap time
    laps = laps.dropna(subset=["LapNumber", "LapTime"]).copy()

    # Convert LapTime Timedelta → seconds
    laps["lap_time_s"] = laps["LapTime"].apply(_safe_total_seconds)

    # Keep only laps within sensible time bounds
    laps = laps[
        (laps["lap_time_s"] >= LAP_TIME_MIN_S) &
        (laps["lap_time_s"] <= LAP_TIME_MAX_S)
    ].copy()

    if laps.empty:
        raise ValueError("No valid lap data after cleaning.")

    laps["lap_number"] = laps["LapNumber"].astype(int)
    laps["driver_id"]  = laps["Driver"].astype(str)
    laps["team_id"]    = laps["Team"].fillna("Unknown").astype(str)

    # ── Step 2: Compound normalisation ────────────────────────────────
    laps["tire_compound"] = laps["Compound"].apply(_normalise_compound)

    # ── Step 3: Tire age and degradation ──────────────────────────────
    laps["tire_age"] = laps["TyreLife"].fillna(1).astype(int).clip(lower=1)

    laps["tire_degradation"] = laps.apply(
        lambda row: min(
            1.0,
            row["tire_age"] / MAX_STINT_LIFE.get(row["tire_compound"], 40)
        ),
        axis=1,
    )

    # ── Step 4: Estimated fuel load ───────────────────────────────────
    laps["fuel_load_kg"] = (
        FUEL_START_KG - laps["lap_number"] * FUEL_PER_LAP_KG
    ).clip(lower=1.0).round(2)

    # ── Step 5: Safety car status ─────────────────────────────────────
    def _is_sc(status) -> int:
        if pd.isna(status):
            return 0
        return int(any(s in str(status) for s in SC_STATUSES))

    laps["safety_car_active"] = laps["TrackStatus"].apply(_is_sc)

    # ── Step 6: Weather merge ─────────────────────────────────────────
    weather = getattr(session, "weather_data", pd.DataFrame())

    if not weather.empty and "Time" in weather.columns:
        weather = weather.dropna(subset=["Time"]).sort_values("Time").copy()

        # Convert both sides to float seconds for merge_asof
        laps["_session_time_s"]    = laps["Time"].apply(_safe_total_seconds)
        weather["_weather_time_s"] = weather["Time"].apply(_safe_total_seconds)

        laps = laps.sort_values("_session_time_s")
        weather_slim = weather[
            ["_weather_time_s", "TrackTemp", "AirTemp", "Rainfall"]
        ].dropna(subset=["_weather_time_s"]).sort_values("_weather_time_s")

        laps = pd.merge_asof(
            laps,
            weather_slim,
            left_on="_weather_time_s",
            right_on="_weather_time_s",
            direction="nearest",
            tolerance=180.0,   # within 3 minutes
        )

        laps["track_temp_c"] = pd.to_numeric(
            laps["TrackTemp"], errors="coerce"
        ).fillna(30.0).round(1)
        laps["air_temp_c"] = pd.to_numeric(
            laps["AirTemp"], errors="coerce"
        ).fillna(22.0).round(1)

        # Rainfall: FastF1 returns True/False boolean or 0/1
        # We convert to a representative mm value: True → 2.0 mm, False → 0.0
        laps["rainfall_mm"] = laps["Rainfall"].apply(
            lambda x: 2.0 if (x is True or (isinstance(x, (int, float)) and x > 0))
            else 0.0
        ).fillna(0.0)
    else:
        laps["track_temp_c"] = 30.0
        laps["air_temp_c"]   = 22.0
        laps["rainfall_mm"]  = 0.0

    # ── Step 7: Inter-car gap estimates ───────────────────────────────
    laps["position"] = pd.to_numeric(
        laps["Position"], errors="coerce"
    ).fillna(10).astype(int).clip(1, 20)

    laps["_time_s"] = laps["Time"].apply(_safe_total_seconds)

    laps = (
        laps.groupby("lap_number", group_keys=False)
        .apply(_compute_gaps)
    )

    # ── Step 8: Pitstop labels ─────────────────────────────────────────
    # pitstop_this_lap = 1 if the car entered the pit lane during this lap
    laps["pitstop_this_lap"] = laps["PitInTime"].notna().astype(int)

    # next_compound = compound fitted at next stint (look one lap ahead)
    laps = laps.sort_values(["driver_id", "lap_number"]).reset_index(drop=True)
    laps["next_compound"] = pd.NA

    for driver in laps["driver_id"].unique():
        d_mask   = laps["driver_id"] == driver
        d_laps   = laps[d_mask].copy()
        pit_mask = d_laps["pitstop_this_lap"] == 1

        for idx in d_laps[pit_mask].index:
            pit_lap = laps.loc[idx, "lap_number"]
            next_lap_mask = d_mask & (laps["lap_number"] == pit_lap + 1)
            if next_lap_mask.any():
                laps.loc[idx, "next_compound"] = (
                    laps.loc[next_lap_mask, "tire_compound"].values[0]
                )

    # ── Step 9: Select REQUIRED_COLUMNS ───────────────────────────────
    REQUIRED_COLUMNS = [
        "lap_number", "driver_id", "team_id",
        "tire_compound", "tire_age", "lap_time_s",
        "fuel_load_kg", "tire_degradation",
        "track_temp_c", "air_temp_c", "rainfall_mm",
        "safety_car_active", "position",
        "gap_ahead_s", "gap_behind_s",
        "pitstop_this_lap", "next_compound",
    ]

    result = laps[REQUIRED_COLUMNS].copy()

    # Drop rows with NaN in any mandatory column
    # (next_compound is allowed to be NaN for non-pit laps)
    mandatory = [c for c in REQUIRED_COLUMNS if c != "next_compound"]
    result = result.dropna(subset=mandatory).reset_index(drop=True)

    return result


# ─────────────────────────────────────────────────────────────
# 4.  SEASON BATCH LOADER
# ─────────────────────────────────────────────────────────────

def load_season(
    year: int,
    cache_dir: str = "./f1_cache",
    max_rounds: int = None,
    skip_rounds: list = None,
) -> pd.DataFrame:
    """
    Load all race sessions for a given season and concatenate them.

    Parameters
    ----------
    year        : Championship year
    cache_dir   : FastF1 cache path
    max_rounds  : If set, only load the first N rounds (useful for testing)
    skip_rounds : List of round numbers to skip (e.g. cancelled races)

    Returns
    -------
    pd.DataFrame with REQUIRED_COLUMNS + [year, round_number, gp_name]
    """
    setup_cache(cache_dir)

    schedule = fastf1.get_event_schedule(year, include_testing=False)

    # Keep only conventional race weekends (excludes testing, sprint-only)
    schedule = schedule[
        schedule["EventFormat"].isin(["conventional", "sprint", "sprint_qualifying"])
    ].copy()

    schedule = schedule.sort_values("RoundNumber").reset_index(drop=True)

    if max_rounds is not None:
        schedule = schedule.head(max_rounds)

    if skip_rounds:
        schedule = schedule[~schedule["RoundNumber"].isin(skip_rounds)]

    all_dfs   = []
    n_rounds  = len(schedule)
    n_ok      = 0
    n_skipped = 0

    print(f"\n{'─'*60}")
    print(f"  Loading {year} season  ({n_rounds} rounds)")
    print(f"{'─'*60}")

    for _, event in schedule.iterrows():
        round_num = int(event["RoundNumber"])
        gp_name   = event.get("EventName", f"Round {round_num}")

        print(f"  [{n_ok + n_skipped + 1:02d}/{n_rounds}]  "
              f"R{round_num:02d} {gp_name[:35]:<35}", end="  ", flush=True)

        try:
            session = load_race_session(year, round_num, cache_dir, verbose=False)
            df = session_to_dataframe(session)
            df["year"]         = year
            df["round_number"] = round_num
            df["gp_name"]      = gp_name
            all_dfs.append(df)
            n_ok += 1

            pit_pct = df["pitstop_this_lap"].mean() * 100
            print(f"✔  {len(df):>5,} laps  |  {pit_pct:.1f}% pit stops")

        except Exception as exc:
            n_skipped += 1
            print(f"✗  SKIPPED  [{type(exc).__name__}: {exc}]")
            continue

    if not all_dfs:
        raise ValueError(
            f"Could not load any race data for {year}. "
            "Check your internet connection and FastF1 cache."
        )

    df_season = pd.concat(all_dfs, ignore_index=True)

    print(f"\n  {'─'*40}")
    print(f"  {year} season summary")
    print(f"  Loaded  : {n_ok} / {n_rounds} rounds")
    print(f"  Total rows    : {len(df_season):,}")
    pit_total = df_season["pitstop_this_lap"].sum()
    print(f"  Pit stop rows : {pit_total:,}  "
          f"({pit_total / len(df_season) * 100:.1f}%)")
    teams = df_season["team_id"].nunique()
    print(f"  Teams         : {teams}")
    print(f"  {'─'*40}\n")

    return df_season


# ─────────────────────────────────────────────────────────────
# 5.  CORPUS BUILDER (train / test split by season)
# ─────────────────────────────────────────────────────────────

def build_corpus(
    train_years: tuple = (2022,),
    test_years:  tuple = (2023,),
    cache_dir:   str   = "./f1_cache",
    save_csv:    bool  = True,
    csv_dir:     str   = "./f1_outputs",
) -> tuple:
    """
    Build temporally separated train and test DataFrames by season.

    The train/test boundary is at the SEASON level — all training data
    precedes all test data chronologically. This enforces temporal
    integrity as required by the DSRP methodology (Chapter 3).

    Parameters
    ----------
    train_years : Tuple of years for training (e.g. (2022,))
    test_years  : Tuple of years for testing  (e.g. (2023, 2024))
    cache_dir   : FastF1 cache directory
    save_csv    : If True, save df_train and df_test to CSV for offline reuse
    csv_dir     : Directory for saved CSVs

    Returns
    -------
    (df_train, df_test) — both as pd.DataFrames
    """
    print(f"\n{'═'*60}")
    print(f"  F1 STRATEGY MODEL — BUILDING CORPUS")
    print(f"  Train years : {train_years}")
    print(f"  Test years  : {test_years}")
    print(f"{'═'*60}")

    # ── Load training seasons ──────────────────────────────────────────
    train_frames = []
    for yr in train_years:
        df = load_season(yr, cache_dir)
        train_frames.append(df)
    df_train = pd.concat(train_frames, ignore_index=True)

    # ── Load test seasons ──────────────────────────────────────────────
    test_frames = []
    for yr in test_years:
        df = load_season(yr, cache_dir)
        test_frames.append(df)
    df_test = pd.concat(test_frames, ignore_index=True)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  CORPUS SUMMARY")
    print(f"{'─'*60}")
    for label, df in [("TRAIN", df_train), ("TEST", df_test)]:
        n       = len(df)
        n_pit   = int(df["pitstop_this_lap"].sum())
        pit_pct = n_pit / n * 100
        wet_pct = (df["rainfall_mm"] > 0.5).mean() * 100
        print(f"  {label}:  {n:>7,} rows  |  "
              f"{n_pit:>5,} pit stops ({pit_pct:.1f}%)  |  "
              f"Wet laps: {wet_pct:.1f}%")
    print(f"{'═'*60}\n")

    # ── Optional CSV save ─────────────────────────────────────────────
    if save_csv:
        os.makedirs(csv_dir, exist_ok=True)
        train_path = os.path.join(csv_dir, f"f1_train_{'_'.join(map(str,train_years))}.csv")
        test_path  = os.path.join(csv_dir, f"f1_test_{'_'.join(map(str,test_years))}.csv")
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path,   index=False)
        print(f"  💾  Saved: {train_path}")
        print(f"  💾  Saved: {test_path}\n")

    return df_train, df_test


# ─────────────────────────────────────────────────────────────
# 6.  UTILITY: load a pre-saved CSV (fast offline mode)
# ─────────────────────────────────────────────────────────────

def load_from_csv(
    train_path: str = None,
    test_path:  str = None,
    csv_dir:    str = "./f1_outputs",
    train_years: tuple = (2022,),
    test_years:  tuple = (2023,),
) -> tuple:
    """
    Load previously saved train/test CSVs instead of hitting the API.

    Falls back to build_corpus() if CSVs are not found.
    """
    if train_path is None:
        train_path = os.path.join(
            csv_dir, f"f1_train_{'_'.join(map(str, train_years))}.csv"
        )
    if test_path is None:
        test_path = os.path.join(
            csv_dir, f"f1_test_{'_'.join(map(str, test_years))}.csv"
        )

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"  📂  Loading from cached CSVs:")
        print(f"      {train_path}")
        print(f"      {test_path}")
        df_train = pd.read_csv(train_path)
        df_test  = pd.read_csv(test_path)
        print(f"  ✔  Train: {len(df_train):,} rows | Test: {len(df_test):,} rows\n")
        return df_train, df_test
    else:
        print(f"  ℹ  No cached CSVs found — pulling from FastF1 API...\n")
        return build_corpus(
            train_years=train_years,
            test_years=test_years,
            csv_dir=csv_dir,
            save_csv=True,
        )


# ─────────────────────────────────────────────────────────────
# 7.  QUICK SANITY CHECK (run standalone to test connection)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick connectivity test:
    python fastf1_connector.py

    Loads Round 1 of 2022 only (Bahrain GP) to verify
    the FastF1 connection and schema transformation work.
    """
    print("\n  F1 CONNECTOR — SANITY CHECK")
    print("  Loading 2022 Round 1 (Bahrain GP)...\n")

    try:
        session = load_race_session(2022, 1, cache_dir="./f1_cache")
        df = session_to_dataframe(session)

        print(f"\n  Schema check:")
        print(df.dtypes)
        print(f"\n  First 5 rows:")
        print(df.head())
        print(f"\n  Pit stop distribution:")
        print(df["pitstop_this_lap"].value_counts())
        print(f"\n  Compound distribution:")
        print(df["tire_compound"].value_counts())
        print(f"\n  ✔  Connector working correctly.\n")

    except Exception as e:
        print(f"\n  ✗  Error: {e}")
        print("  Check: (1) internet connection, "
              "(2) fastf1 installed, (3) cache writable.\n")
