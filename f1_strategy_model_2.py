"""
============================================================
F1 RACE STRATEGY PREDICTION MODEL  v2.0
============================================================
Upgrades from v1:
  [FIX]  FEATURE_COLS global mutation bug removed
  [FIX]  Temporal train/test split — season boundary enforced
  [NEW]  sample_weight balancing on GBC (class imbalance, inverse-frequency)
  [NEW]  StratifiedKFold(n_splits=5) cross-validation
  [NEW]  MCC (Matthews Correlation Coefficient) metric
  [NEW]  G-mean (Geometric Mean) metric
  [NEW]  ROC-AUC + precision-recall AUC metrics
  [NEW]  ROC curve visualisation
  [NEW]  Precision-Recall curve visualisation
  [NEW]  MCC / G-mean summary tile
  [NEW]  Season-level cross-team evaluation
  [KEEP] All 8 original visualisations unchanged

Architecture (matches Chapter 3 DSRP methodology):
  Classifier 1 — GradientBoostingClassifier
                 Target: pitstop_this_lap (binary 0/1)
                 Imbalance: class_weight='balanced'

  Classifier 2 — RandomForestClassifier
                 Target: next_compound (SOFT/MEDIUM/HARD/INTER/WET)
                 Trained on pit-stop laps only

Evaluation metrics (Section 3.3, Phase 5):
  Primary  : MCC, G-mean
  Secondary: F1-score, ROC-AUC, PR-AUC
  Rejected : raw accuracy (class imbalance makes it misleading)

Usage:
  # With real FastF1 data (requires fastf1_connector.py):
  from fastf1_connector import build_corpus
  df_train, df_test = build_corpus(train_years=(2022,), test_years=(2023,))
  model = F1StrategyModel()
  model.fit(df_train)
  results = model.evaluate_season(df_test, season_label="2023")
  run_pipeline(df_train=df_train, df_test=df_test)

  # With synthetic demo data (no API required):
  python f1_strategy_model_2.py
============================================================
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.cm as cm

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    matthews_corrcoef,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    recall_score,
)
import joblib


# ─────────────────────────────────────────────────────────────
# CONSTANTS  (immutable — never mutated after import)
# ─────────────────────────────────────────────────────────────

COMPOUND_COLORS = {
    "SOFT":    "#E8002D",
    "MEDIUM":  "#FFF200",
    "HARD":    "#FFFFFF",
    "INTER":   "#39B54A",
    "WET":     "#0067FF",
    "UNKNOWN": "#AAAAAA",
}

REQUIRED_COLUMNS = [
    # Identification
    "lap_number", "driver_id", "team_id",
    # Tyre state
    "tire_compound", "tire_age", "tire_degradation",
    "fresh_tyre",     # 1 = new/unscubbed, 0 = scrubbed/used
    "stint_number",   # exact stint count (estimated if TI not available)
    # Performance
    "lap_time_s", "fuel_load_kg",
    # Weather — all five FastF1 channels
    "track_temp_c", "air_temp_c", "rainfall_mm",
    "humidity_pct",    # relative humidity %
    "wind_speed_ms",   # wind speed in m/s
    # Race context
    "safety_car_active", "position",
    "gap_ahead_s", "gap_behind_s",
    # Targets
    "pitstop_this_lap", "next_compound",
]

# ── Base features (raw, always available after validation)
BASE_FEATURES = [
    "lap_number", "tire_age", "lap_time_s",
    "fuel_load_kg", "tire_degradation",
    "fresh_tyre",      # direct tyre-lifecycle signal from Tracing Insights
    "stint_number",    # exact stint count; informs pit window strategy
    "track_temp_c", "air_temp_c", "rainfall_mm",
    "humidity_pct",    # full weather: humidity affects rubber & aero
    "wind_speed_ms",   # full weather: wind affects pace variance & gaps
    "safety_car_active", "position",
    "gap_ahead_s", "gap_behind_s",
    "tire_compound_enc",
]

# ── Engineered features (added by engineer_features())
ENGINEERED_FEATURES = [
    "laptime_delta",    # lap-over-lap pace loss (deg proxy)
    "deg_rate",         # tire_degradation / tire_age rate of change
    "fuel_delta",       # estimated fuel burn per lap
    "wet_conditions",   # composite flag: rainfall > 0 OR humidity > 85%
]

# ── Full feature list used by the model (BASE + ENGINEERED)
# This is a constant — engineer_features() returns a new df,
# it does NOT mutate this list.
ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

OUTPUT_DIR = "f1_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Matplotlib F1 dark theme
plt.style.use("dark_background")
F1_RED   = "#E8002D"
F1_WHITE = "#F5F5F5"
F1_GREY  = "#1C1C1E"
F1_GOLD  = "#FFD700"
ACCENT   = "#00D2BE"


# ═══════════════════════════════════════════════════════════════
# 1.  DATA INGESTION & VALIDATION
# ═══════════════════════════════════════════════════════════════

def load_and_validate(filepath: str) -> pd.DataFrame:
    """Load CSV/Excel, validate schema and types. Returns clean DataFrame."""
    print(f"\n{'='*60}")
    print("  F1 STRATEGY MODEL v2 — DATA INGESTION")
    print(f"{'='*60}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    elif ext == ".csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .csv or .xlsx")

    print(f"  ✔  Loaded: {len(df):,} rows × {len(df.columns)} columns")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [
        "lap_number", "tire_age", "lap_time_s", "fuel_load_kg",
        "tire_degradation", "fresh_tyre", "stint_number",
        "track_temp_c", "air_temp_c", "rainfall_mm",
        "humidity_pct", "wind_speed_ms",
        "safety_car_active", "position",
        "gap_ahead_s", "gap_behind_s", "pitstop_this_lap",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["tire_compound"] = df["tire_compound"].str.upper().str.strip()
    df["next_compound"] = df["next_compound"].str.upper().str.strip()

    # Drop rows missing any mandatory column (next_compound NaN is OK)
    mandatory = [c for c in REQUIRED_COLUMNS if c != "next_compound"]
    before = len(df)
    df = df.dropna(subset=mandatory).reset_index(drop=True)
    if len(df) < before:
        print(f"  ⚠  Dropped {before - len(df)} rows with missing values.")

    # Range assertions
    assert df["tire_degradation"].between(0, 1).all(), \
        "tire_degradation must be in [0, 1]"
    assert df["pitstop_this_lap"].isin([0, 1]).all(), \
        "pitstop_this_lap must be 0 or 1"

    valid_compounds = {"SOFT", "MEDIUM", "HARD", "INTER", "WET"}
    bad = set(df["tire_compound"].unique()) - valid_compounds
    if bad:
        print(f"  ⚠  Unknown compounds mapped to HARD: {bad}")
        df["tire_compound"] = df["tire_compound"].where(
            df["tire_compound"].isin(valid_compounds), "HARD"
        )

    pit_pct = df["pitstop_this_lap"].mean() * 100
    print(f"  ✔  {len(df):,} clean rows | {pit_pct:.1f}% pit-stop laps\n")
    return df


# ═══════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived predictive features to the DataFrame.

    Returns a NEW DataFrame — does not mutate globals.
    All added columns are listed in ENGINEERED_FEATURES.
    """
    df = df.copy()

    # Encode tire compound (must be done before sorting)
    le = LabelEncoder()
    df["tire_compound_enc"] = le.fit_transform(df["tire_compound"])

    # Sort for rolling computations
    df = df.sort_values(["driver_id", "lap_number"]).reset_index(drop=True)

    # Rolling lap-time trend per driver (window=3) → deviation from mean
    df["_lt_roll"] = (
        df.groupby("driver_id")["lap_time_s"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    df["laptime_delta"] = df["lap_time_s"] - df["_lt_roll"]
    df.drop(columns=["_lt_roll"], inplace=True)

    # Per-lap degradation rate (first diff per driver/compound stint)
    df["deg_rate"] = (
        df.groupby(["driver_id", "tire_compound"])["tire_degradation"]
        .transform(lambda x: x.diff().fillna(0))
    )

    # Per-lap fuel consumption (first diff per driver)
    df["fuel_delta"] = (
        df.groupby("driver_id")["fuel_load_kg"]
        .transform(lambda x: x.diff().fillna(0))
    )

    # Composite wet-conditions flag:
    #   rainfall > 0.5 mm  OR  humidity above 85%
    # The humidity branch captures damp/overcast conditions that affect
    # tyre behaviour even without active rain (important for intermediate
    # vs dry compound decisions at circuits like Spa or Suzuka).
    df["wet_conditions"] = (
        (df["rainfall_mm"] > 0.5) |
        (df.get("humidity_pct", pd.Series(0, index=df.index)) > 85)
    ).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════
# 3.  IMBALANCE-AWARE METRICS
# ═══════════════════════════════════════════════════════════════

def compute_imbalance_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Compute the full suite of imbalance-aware metrics described in
    Chapter 3 Section 3.3 Phase 5.

    Returns a dict with: mcc, gmean, f1, roc_auc, pr_auc,
    sensitivity, specificity, confusion_matrix
    """
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    gmean       = float(np.sqrt(sensitivity * specificity))
    mcc         = float(matthews_corrcoef(y_true, y_pred))
    f1          = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc = None
    pr_auc  = None
    if y_proba is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba))
            pr_auc  = float(average_precision_score(y_true, y_proba))
        except Exception:
            pass

    return {
        "mcc":          mcc,
        "gmean":        gmean,
        "f1":           f1,
        "roc_auc":      roc_auc,
        "pr_auc":       pr_auc,
        "sensitivity":  sensitivity,
        "specificity":  specificity,
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
        "confusion_matrix": cm,
    }


# ═══════════════════════════════════════════════════════════════
# 4.  MODEL CLASS
# ═══════════════════════════════════════════════════════════════

class F1StrategyModel:
    """
    Dual-classifier F1 strategy prediction model.

    Classifier 1 (pitstop_model):
        GradientBoostingClassifier — binary pitstop_this_lap prediction
        Class imbalance handled via sample_weight (inverse class frequency),
        equivalent to class_weight='balanced' but compatible with GBC

    Classifier 2 (compound_model):
        RandomForestClassifier — multi-class next_compound prediction
        Trained on pitstop laps only
    """

    def __init__(self):
        self.pitstop_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            # class_weight not supported in GBC — handled via sample_weight
        )
        self.compound_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            n_jobs=-1,        # use all M2 cores for RFC
            class_weight="balanced",
            random_state=42,
        )
        self.scaler          = StandardScaler()
        self.compound_le     = LabelEncoder()
        self.is_trained      = False
        self.feature_names   = list(ALL_FEATURES)
        self.feature_importance_pit  = None
        self.feature_importance_comp = None
        self.train_metrics   = {}
        self.cv_metrics      = {}

    # ──────────────────────────────────────────────────────────
    # FIT
    # ──────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, n_cv_folds: int = 5):
        """
        Train both classifiers on the provided DataFrame.

        Class imbalance in pitstop_model is handled via sample_weight
        (equivalent to class_weight='balanced' but compatible with GBC).

        Cross-validation uses StratifiedKFold to preserve the minority
        class proportion in each fold.
        """
        print(f"\n{'='*60}")
        print("  TRAINING  —  F1 Strategy Model v2")
        print(f"{'='*60}")

        # Ensure features have been engineered
        for feat in ALL_FEATURES:
            if feat not in df.columns:
                raise ValueError(
                    f"Feature '{feat}' not found. "
                    "Did you call engineer_features() before fit()?"
                )

        X      = df[ALL_FEATURES].values
        y_pit  = df["pitstop_this_lap"].values.astype(int)

        # ── Compute sample weights for GBC (balances classes)
        class_counts   = np.bincount(y_pit)
        class_weights  = len(y_pit) / (2 * class_counts)  # inverse freq
        sample_weights = class_weights[y_pit]

        print(f"\n  Class distribution:")
        print(f"    No pit stop : {class_counts[0]:,} ({class_counts[0]/len(y_pit)*100:.1f}%)")
        print(f"    Pit stop    : {class_counts[1]:,} ({class_counts[1]/len(y_pit)*100:.1f}%)")
        print(f"    Imbalance   : 1 : {class_counts[0]//class_counts[1]}")

        # ── Scale features
        X_scaled = self.scaler.fit_transform(X)

        # ── 5-fold stratified cross-validation on FULL dataset ──────
        print(f"\n  Running {n_cv_folds}-fold Stratified CV...")
        skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
        cv_mcc  = []
        cv_gmean = []
        cv_f1   = []
        cv_roc  = []

        for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_scaled, y_pit)):
            X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
            y_tr, y_val = y_pit[tr_idx], y_pit[val_idx]
            sw_tr       = sample_weights[tr_idx]

            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=4,
                learning_rate=0.05, subsample=0.8, random_state=42
            )
            clf.fit(X_tr, y_tr, sample_weight=sw_tr)
            y_val_pred  = clf.predict(X_val)
            y_val_proba = clf.predict_proba(X_val)[:, 1]

            m = compute_imbalance_metrics(y_val, y_val_pred, y_val_proba)
            cv_mcc.append(m["mcc"])
            cv_gmean.append(m["gmean"])
            cv_f1.append(m["f1"])
            if m["roc_auc"] is not None:
                cv_roc.append(m["roc_auc"])

            auc_str = f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "N/A"
            print(f"    Fold {fold_i+1}: MCC={m['mcc']:.3f}  "
                  f"G-mean={m['gmean']:.3f}  F1={m['f1']:.3f}  "
                  f"AUC={auc_str}")

        self.cv_metrics = {
            "mcc":   {"mean": np.mean(cv_mcc),   "std": np.std(cv_mcc)},
            "gmean": {"mean": np.mean(cv_gmean),  "std": np.std(cv_gmean)},
            "f1":    {"mean": np.mean(cv_f1),     "std": np.std(cv_f1)},
            "roc":   {"mean": np.mean(cv_roc),    "std": np.std(cv_roc)},
        }

        print(f"\n  CV Summary ({n_cv_folds}-fold):")
        print(f"  {'─'*40}")
        for k, v in self.cv_metrics.items():
            print(f"  {k.upper():<8}: {v['mean']:.3f} ± {v['std']:.3f}")

        # ── Train final pitstop model on full training data ──────────
        print(f"\n  Training final pitstop model on full dataset...")
        self.pitstop_model.fit(X_scaled, y_pit, sample_weight=sample_weights)
        self.feature_importance_pit = self.pitstop_model.feature_importances_

        # ── Train compound model (pitstop laps only) ─────────────────
        pit_mask = df["pitstop_this_lap"] == 1
        comp_df  = df[pit_mask].copy()
        comp_df["next_compound"] = comp_df["next_compound"].fillna("HARD")

        y_comp = self.compound_le.fit_transform(comp_df["next_compound"])
        X_comp = self.scaler.transform(comp_df[ALL_FEATURES].values)

        X_ctr, X_cte, y_ctr, y_cte = train_test_split(
            X_comp, y_comp, test_size=0.2, random_state=42, stratify=y_comp
        )
        self.compound_model.fit(X_ctr, y_ctr)
        comp_preds = self.compound_model.predict(X_cte)
        comp_report = classification_report(
            y_cte, comp_preds, output_dict=True, zero_division=0
        )
        self.feature_importance_comp = self.compound_model.feature_importances_

        print(f"\n  Compound Classifier (on {len(comp_df)} pit-stop laps):")
        print(f"  {'─'*40}")
        print(f"  Accuracy  : {comp_report['accuracy']:.3f}")
        print(f"  Macro F1  : {comp_report['macro avg']['f1-score']:.3f}")

        # Store training metrics for reporting
        final_preds  = self.pitstop_model.predict(X_scaled)
        final_proba  = self.pitstop_model.predict_proba(X_scaled)[:, 1]
        self.train_metrics["pitstop"] = compute_imbalance_metrics(
            y_pit, final_preds, final_proba
        )
        self.train_metrics["compound"] = {
            "report": comp_report,
            "cm": confusion_matrix(y_cte, comp_preds),
        }

        self.is_trained = True
        print(f"\n  ✔  Models trained successfully.\n")

    # ──────────────────────────────────────────────────────────
    # PREDICT
    # ──────────────────────────────────────────────────────────
    def predict_race(self, df_race: pd.DataFrame) -> dict:
        """
        Run inference on a single race (one driver or all drivers).
        Returns pitstop probability, predictions, and strategy analysis.
        """
        assert self.is_trained, "Call fit() before predict_race()."

        X = self.scaler.transform(df_race[ALL_FEATURES].values)
        pitstop_probs  = self.pitstop_model.predict_proba(X)[:, 1]
        pitstop_preds  = self.pitstop_model.predict(X)
        compound_preds_idx = self.compound_model.predict(X)
        compound_preds = self.compound_le.inverse_transform(compound_preds_idx)

        df_out = df_race.copy()
        df_out["pitstop_prob"] = pitstop_probs
        df_out["pitstop_pred"] = pitstop_preds
        df_out["compound_rec"] = compound_preds

        THRESHOLD = 0.40
        windows = _extract_pitstop_windows(df_out, threshold=THRESHOLD)
        factors = _identify_factors(df_out)

        return {
            "predictions":     df_out,
            "pitstop_windows": windows,
            "factors":         factors,
            "compound_le_classes": list(self.compound_le.classes_),
        }

    # ──────────────────────────────────────────────────────────
    # EVALUATE (held-out season)
    # ──────────────────────────────────────────────────────────
    def evaluate_season(
        self,
        df_test: pd.DataFrame,
        season_label: str = "Test",
    ) -> dict:
        """
        Evaluate trained model on a held-out season (temporal test set).
        Returns imbalance-aware metrics and confusion matrix data.
        Prints formatted report. Saves evaluation plots.
        """
        assert self.is_trained, "Call fit() first."

        print(f"\n{'='*60}")
        print(f"  EVALUATION — {season_label} Season")
        print(f"{'='*60}")

        X_test = self.scaler.transform(df_test[ALL_FEATURES].values)
        y_test = df_test["pitstop_this_lap"].values.astype(int)

        y_pred  = self.pitstop_model.predict(X_test)
        y_proba = self.pitstop_model.predict_proba(X_test)[:, 1]

        metrics = compute_imbalance_metrics(y_test, y_pred, y_proba)

        print(f"\n  {'─'*50}")
        print(f"  Pit Stop Classifier — {season_label}")
        print(f"  {'─'*50}")
        print(f"  MCC             : {metrics['mcc']:>8.4f}   ← PRIMARY METRIC")
        print(f"  G-mean          : {metrics['gmean']:>8.4f}   ← PRIMARY METRIC")
        print(f"  F1-score        : {metrics['f1']:>8.4f}")
        print(f"  ROC-AUC         : {metrics['roc_auc']:>8.4f}")
        print(f"  PR-AUC          : {metrics['pr_auc']:>8.4f}")
        print(f"  Sensitivity     : {metrics['sensitivity']:>8.4f}  (pit stop recall)")
        print(f"  Specificity     : {metrics['specificity']:>8.4f}  (non-stop recall)")
        print(f"\n  Confusion Matrix:")
        print(f"  {'─'*30}")
        print(f"                 Pred No  Pred Yes")
        print(f"  Actual No  :  {metrics['tn']:>8,}  {metrics['fp']:>8,}")
        print(f"  Actual Yes :  {metrics['fn']:>8,}  {metrics['tp']:>8,}")
        print(f"\n  False Negatives (missed pit windows) : {metrics['fn']:,}")
        print(f"  False Positives (false alarms)       : {metrics['fp']:,}")

        # MCC interpretation
        if metrics["mcc"] >= 0.5:
            interp = "STRONG predictive performance"
        elif metrics["mcc"] >= 0.3:
            interp = "MODERATE predictive performance"
        elif metrics["mcc"] > 0.0:
            interp = "WEAK but above-chance performance"
        else:
            interp = "⚠ WORSE THAN RANDOM — check for issues"
        print(f"\n  MCC interpretation: {interp}")
        print(f"  {'─'*50}")

        # Save plots
        plot_roc_pr_curves(y_test, y_proba, season_label)
        plot_mcc_gmean_summary(
            metrics, self.cv_metrics, season_label
        )

        return {"metrics": metrics, "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba}

    # ──────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ──────────────────────────────────────────────────────────
    def save(self, path: str = "f1_strategy_model.joblib"):
        joblib.dump(self, path)
        print(f"  ✔  Model saved → {path}")

    @staticmethod
    def load(path: str = "f1_strategy_model.joblib"):
        return joblib.load(path)


# ═══════════════════════════════════════════════════════════════
# 5.  HELPER ANALYTICS
# ═══════════════════════════════════════════════════════════════

def _extract_pitstop_windows(df: pd.DataFrame, threshold: float = 0.40) -> list:
    """Group consecutive high-probability laps into pitstop windows."""
    windows  = []
    in_win   = False
    start = end = peak_prob = 0
    rec_compound = "HARD"

    for _, row in df.sort_values("lap_number").iterrows():
        if row["pitstop_prob"] >= threshold:
            if not in_win:
                start = row["lap_number"]
                in_win = True
            end          = row["lap_number"]
            peak_prob    = max(peak_prob, row["pitstop_prob"])
            rec_compound = row["compound_rec"]
        else:
            if in_win:
                windows.append({
                    "start_lap":            int(start),
                    "end_lap":              int(end),
                    "peak_prob":            float(peak_prob),
                    "recommended_compound": rec_compound,
                })
                in_win = False

    if in_win:
        windows.append({
            "start_lap":            int(start),
            "end_lap":              int(end),
            "peak_prob":            float(peak_prob),
            "recommended_compound": rec_compound,
        })
    return windows


def _identify_factors(df: pd.DataFrame) -> list:
    """Generate human-readable risk/opportunity factor alerts."""
    factors = []

    high_deg = df[df["tire_degradation"] > 0.75]
    if not high_deg.empty:
        factors.append({
            "type":   "⚠ WARNING",
            "factor": "HIGH TYRE DEGRADATION",
            "detail": f"deg > 0.75 on laps: {sorted(high_deg['lap_number'].astype(int).tolist()[:8])}",
        })

    if df["rainfall_mm"].max() > 0.5:
        rain_laps = df[df["rainfall_mm"] > 0.5]["lap_number"].astype(int).tolist()
        factors.append({
            "type":   "🌧 WEATHER",
            "factor": "WET CONDITIONS",
            "detail": f"Rainfall on laps: {sorted(rain_laps[:6])}{'...' if len(rain_laps)>6 else ''}",
        })

    if df["safety_car_active"].sum() > 0:
        sc_laps = df[df["safety_car_active"] == 1]["lap_number"].astype(int).tolist()
        factors.append({
            "type":   "🚗 SAFETY CAR",
            "factor": "SAFETY CAR OPPORTUNITY",
            "detail": f"Active on laps: {sorted(sc_laps[:6])}{'...' if len(sc_laps)>6 else ''}",
        })

    late = df[df["lap_number"] > df["lap_number"].quantile(0.7)]
    if len(late) and "laptime_delta" in df.columns:
        avg_delta = late["laptime_delta"].mean()
        if avg_delta > 1.5:
            factors.append({
                "type":   "📉 PERFORMANCE",
                "factor": "LATE-RACE LAP TIME DROP",
                "detail": f"Avg delta in final 30% of race: +{avg_delta:.2f}s",
            })

    if df["fuel_load_kg"].min() < 3.0:
        low_lap = int(df.loc[df["fuel_load_kg"].idxmin(), "lap_number"])
        factors.append({
            "type":   "⛽ FUEL",
            "factor": "LOW FUEL WARNING",
            "detail": f"Fuel drops below 3 kg on lap {low_lap}",
        })

    if not factors:
        factors.append({
            "type":   "✅ CLEAR",
            "factor": "No Critical Factors",
            "detail": "All parameters within normal operating ranges.",
        })
    return factors


# ═══════════════════════════════════════════════════════════════
# 6.  VISUALISATIONS
# ═══════════════════════════════════════════════════════════════

def _apply_f1_style(ax, title: str = ""):
    ax.set_facecolor(F1_GREY)
    ax.tick_params(colors=F1_WHITE, labelsize=8)
    ax.spines[:].set_color("#444444")
    ax.xaxis.label.set_color(F1_WHITE)
    ax.yaxis.label.set_color(F1_WHITE)
    if title:
        ax.set_title(title, color=F1_WHITE, fontsize=10, fontweight="bold", pad=8)


# ── (All original v1 plots kept intact) ──

def plot_tire_degradation(df, driver_id, save=True):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=F1_GREY)
    sub = df[df["driver_id"] == driver_id].sort_values("lap_number")
    for compound, grp in sub.groupby("tire_compound"):
        color = COMPOUND_COLORS.get(compound, "#AAAAAA")
        ax.scatter(grp["lap_number"], grp["tire_degradation"], color=color, s=30, label=compound, zorder=3)
        ax.plot(grp["lap_number"], grp["tire_degradation"], color=color, alpha=0.5, linewidth=1)
    ax.set_xlabel("Lap Number"); ax.set_ylabel("Tyre Degradation (0–1)")
    _apply_f1_style(ax, f"Tyre Degradation — {driver_id}")
    ax.legend(fontsize=8, facecolor="#333333", labelcolor=F1_WHITE)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/tire_degradation_{driver_id}.png"
    if save: plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_lap_times_comparison(df, save=True):
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=F1_GREY)
    teams = df["team_id"].unique()
    cmap  = plt.colormaps.get_cmap("tab10").resampled(len(teams))
    for i, team in enumerate(teams):
        sub = df[df["team_id"] == team].sort_values("lap_number")
        for driver, grp in sub.groupby("driver_id"):
            avg = grp.groupby("lap_number")["lap_time_s"].mean()
            ax.plot(avg.index, avg.values, linewidth=1.5, color=cmap(i), label=f"{team}—{driver}", alpha=0.85)
    ax.set_xlabel("Lap"); ax.set_ylabel("Lap Time (s)")
    _apply_f1_style(ax, "Lap Times — All Drivers")
    ax.legend(fontsize=7, facecolor="#333333", labelcolor=F1_WHITE, loc="upper right", ncol=2)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/lap_times_comparison.png"
    if save: plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_fuel_consumption(df, driver_id, save=True):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, facecolor=F1_GREY)
    sub = df[df["driver_id"] == driver_id].sort_values("lap_number")
    ax1.plot(sub["lap_number"], sub["fuel_load_kg"], color=F1_GOLD, linewidth=2)
    ax1.fill_between(sub["lap_number"], sub["fuel_load_kg"], alpha=0.15, color=F1_GOLD)
    _apply_f1_style(ax1, f"Fuel Load (kg) — {driver_id}")
    if "fuel_delta" in sub.columns:
        ax2.bar(sub["lap_number"], sub["fuel_delta"].abs(), color=ACCENT, alpha=0.7)
        _apply_f1_style(ax2, "Fuel Consumption per Lap (kg)")
        ax2.set_xlabel("Lap Number")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/fuel_consumption_{driver_id}.png"
    if save: plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_pitstop_probability(result, driver_id, save=True):
    df  = result["predictions"]
    sub = df[df["driver_id"] == driver_id].sort_values("lap_number")
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=F1_GREY)
    ax.plot(sub["lap_number"], sub["pitstop_prob"], color=F1_RED, linewidth=2, zorder=4)
    ax.fill_between(sub["lap_number"], sub["pitstop_prob"], alpha=0.2, color=F1_RED)
    ax.axhline(0.40, color=F1_GOLD, linestyle="--", linewidth=1, alpha=0.7, label="Threshold (40%)")
    for win in result["pitstop_windows"]:
        ax.axvspan(win["start_lap"], win["end_lap"], alpha=0.2, color=F1_GOLD, zorder=2)
        mid = (win["start_lap"] + win["end_lap"]) / 2
        c   = win["recommended_compound"]
        cc  = COMPOUND_COLORS.get(c, F1_WHITE)
        ax.annotate(f"PIT\nL{win['start_lap']}–{win['end_lap']}\n→{c}",
                    xy=(mid, win["peak_prob"]),
                    xytext=(mid, min(win["peak_prob"] + 0.15, 0.95)),
                    fontsize=7, color=cc, ha="center", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=cc, lw=0.8))
    ax.set_ylim(0, 1); ax.set_xlabel("Lap"); ax.set_ylabel("Pit Stop Probability")
    _apply_f1_style(ax, f"Predicted Pit Stop Windows — {driver_id}")
    ax.legend(fontsize=8, facecolor="#333333", labelcolor=F1_WHITE)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/pitstop_probability_{driver_id}.png"
    if save: plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_tire_allocation(result, driver_id, save=True):
    df  = result["predictions"]
    sub = df[df["driver_id"] == driver_id].sort_values("lap_number")
    fig, ax = plt.subplots(figsize=(12, 2.5), facecolor=F1_GREY)
    prev_comp = None; seg_start = None
    for _, row in sub.iterrows():
        c = row["tire_compound"]
        if c != prev_comp:
            if prev_comp is not None:
                color = COMPOUND_COLORS.get(prev_comp, F1_WHITE)
                ax.axvspan(seg_start, row["lap_number"], alpha=0.6, color=color)
                ax.text((seg_start + row["lap_number"]) / 2, 0.5, prev_comp,
                        ha="center", va="center", color="black", fontsize=8, fontweight="bold")
            seg_start = row["lap_number"]; prev_comp = c
    if prev_comp:
        color = COMPOUND_COLORS.get(prev_comp, F1_WHITE)
        ax.axvspan(seg_start, sub["lap_number"].max(), alpha=0.6, color=color)
        ax.text((seg_start + sub["lap_number"].max()) / 2, 0.5, prev_comp,
                ha="center", va="center", color="black", fontsize=8, fontweight="bold")
    ax.set_xlabel("Lap"); ax.set_yticks([])
    _apply_f1_style(ax, f"Tyre Compound Allocation — {driver_id}")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/tire_allocation_{driver_id}.png"
    if save: plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_weather_overlay(df, driver_id, save=True):
    sub = df[df["driver_id"] == driver_id].sort_values("lap_number")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 7), sharex=True, facecolor=F1_GREY)
    ax1.plot(sub["lap_number"], sub["lap_time_s"], color=ACCENT, linewidth=1.8)
    _apply_f1_style(ax1, f"Lap Time (s) — {driver_id}")
    ax2.plot(sub["lap_number"], sub["track_temp_c"], color="#FF6B35", linewidth=1.5, label="Track °C")
    ax2.plot(sub["lap_number"], sub["air_temp_c"], color="#FFA07A", linewidth=1.5, linestyle="--", label="Air °C")
    _apply_f1_style(ax2, "Temperature (°C)"); ax2.legend(fontsize=7, facecolor="#333333", labelcolor=F1_WHITE)
    ax3.bar(sub["lap_number"], sub["rainfall_mm"], color="#4FC3F7", alpha=0.8)
    _apply_f1_style(ax3, "Rainfall (mm)"); ax3.set_xlabel("Lap")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/weather_overlay_{driver_id}.png"
    if save: plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_feature_importance(model: F1StrategyModel, save=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=F1_GREY)
    for ax, imp, title in [
        (ax1, model.feature_importance_pit,  "Pit Stop Model — Feature Importance"),
        (ax2, model.feature_importance_comp, "Compound Model — Feature Importance"),
    ]:
        feats = model.feature_names[:len(imp)]
        idx   = np.argsort(imp)[::-1]; top_n = min(12, len(feats))
        ax.barh([feats[i] for i in idx[:top_n]][::-1],
                [imp[i] for i in idx[:top_n]][::-1], color=F1_RED, alpha=0.85)
        _apply_f1_style(ax, title); ax.set_xlabel("Importance")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/feature_importance.png"
    if save: plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_confusion_matrices(model: F1StrategyModel, save=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=F1_GREY)
    ConfusionMatrixDisplay(
        model.train_metrics["pitstop"]["confusion_matrix"],
        display_labels=["No Stop", "Pit Stop"]
    ).plot(ax=ax1, colorbar=False, cmap="Reds")
    _apply_f1_style(ax1, "Pit Stop Model — Confusion Matrix (Train)")
    ConfusionMatrixDisplay(
        model.train_metrics["compound"]["cm"]
    ).plot(ax=ax2, colorbar=False, cmap="Blues")
    _apply_f1_style(ax2, "Compound Model — Confusion Matrix")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/confusion_matrices.png"
    if save: plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


# ── NEW v2 plots ──────────────────────────────────────────────

def plot_roc_pr_curves(
    y_true, y_proba, season_label: str = "Test", save: bool = True
):
    """
    Side-by-side ROC curve and Precision-Recall curve.
    Benchmarks: Dinh et al. ROC-AUC 0.9934, SOSTA AI AUC 0.921
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=F1_GREY)

    # ── ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    ax1.plot(fpr, tpr, color=F1_RED, linewidth=2.5, label=f"This model (AUC={auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1, label="Random (AUC=0.5)")
    # Benchmark reference lines
    ax1.axhline(y=0.921, xmin=0, xmax=0.5, color="#FFA500", linestyle=":", linewidth=1,
                alpha=0.7, label="SOSTA AI benchmark (0.921)")
    ax1.axhline(y=0.9934, xmin=0, xmax=0.5, color=ACCENT, linestyle=":", linewidth=1,
                alpha=0.7, label="Dinh et al. benchmark (0.9934)")
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    _apply_f1_style(ax1, f"ROC Curve — {season_label}")
    ax1.legend(fontsize=7.5, facecolor="#333333", labelcolor=F1_WHITE)
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)

    # ── Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    baseline = y_true.mean()   # random classifier baseline = prevalence
    ax2.plot(recall, precision, color=ACCENT, linewidth=2.5, label=f"This model (PR-AUC={pr_auc:.3f})")
    ax2.axhline(baseline, color="#555555", linestyle="--", linewidth=1,
                label=f"Random baseline ({baseline:.2f})")
    ax2.set_xlabel("Recall (Sensitivity)"); ax2.set_ylabel("Precision")
    _apply_f1_style(ax2, f"Precision-Recall Curve — {season_label}")
    ax2.legend(fontsize=8, facecolor="#333333", labelcolor=F1_WHITE)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/roc_pr_curves_{season_label.replace(' ','_')}.png"
    if save: plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_mcc_gmean_summary(
    test_metrics: dict,
    cv_metrics:   dict,
    season_label: str = "Test",
    save:         bool = True,
):
    """
    Summary tile showing MCC and G-mean for CV and held-out test,
    with threshold reference lines marking 'useful', 'good', 'strong'.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=F1_GREY)

    bars_data = {
        "MCC": {
            "CV mean":  cv_metrics["mcc"]["mean"],
            "CV ±":     cv_metrics["mcc"]["std"],
            f"{season_label}": test_metrics["mcc"],
        },
        "G-mean": {
            "CV mean":  cv_metrics["gmean"]["mean"],
            "CV ±":     cv_metrics["gmean"]["std"],
            f"{season_label}": test_metrics["gmean"],
        },
    }

    thresholds = {
        "MCC":    [(0.3, "Moderate", "#FFA500"), (0.5, "Strong", "#4ade80")],
        "G-mean": [(0.5, "Moderate", "#FFA500"), (0.7, "Strong", "#4ade80")],
    }

    for ax, (metric, data) in zip([ax1, ax2], bars_data.items()):
        labels = list(data.keys())
        bar_labels = [k for k in labels if "±" not in k]
        bar_vals   = [data[k] for k in bar_labels]
        colors     = [F1_RED if "CV" in k else ACCENT for k in bar_labels]

        bars = ax.bar(bar_labels, bar_vals, color=colors, alpha=0.85, width=0.5)
        for bar, val in zip(bars, bar_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    color=F1_WHITE, fontsize=10, fontweight="bold")

        for thresh_val, thresh_label, thresh_color in thresholds[metric]:
            ax.axhline(thresh_val, color=thresh_color, linestyle="--",
                       linewidth=1.2, alpha=0.8,
                       label=f"{thresh_label} ({thresh_val})")

        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric)
        _apply_f1_style(ax, f"{metric} — CV vs {season_label}")
        ax.legend(fontsize=7.5, facecolor="#333333", labelcolor=F1_WHITE)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/mcc_gmean_{season_label.replace(' ','_')}.png"
    if save: plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_team_comparison(df_train, df_compare, model, save=True):
    fig, ax = plt.subplots(figsize=(13, 5), facecolor=F1_GREY)
    colors  = [F1_RED, ACCENT, F1_GOLD, "#A78BFA", "#34D399"]
    all_ds  = {"Training": df_train, **df_compare}
    for i, (label, df) in enumerate(all_ds.items()):
        df_eng = engineer_features(df)
        res    = model.predict_race(df_eng)
        avg    = res["predictions"].groupby("lap_number")["pitstop_prob"].mean()
        ax.plot(avg.index, avg.values, color=colors[i % len(colors)],
                linewidth=2, label=label, alpha=0.9)
    ax.axhline(0.40, color="white", linestyle="--", linewidth=0.8, alpha=0.5, label="Threshold")
    ax.set_xlabel("Lap"); ax.set_ylabel("Mean Pit Stop Probability")
    _apply_f1_style(ax, "Cross-Team Strategy Comparison")
    ax.legend(fontsize=8, facecolor="#333333", labelcolor=F1_WHITE)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/team_comparison.png"
    if save: plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


# ═══════════════════════════════════════════════════════════════
# 7.  STRATEGY REPORT
# ═══════════════════════════════════════════════════════════════

def print_strategy_report(result: dict, driver_id: str):
    print(f"\n{'═'*60}")
    print(f"  🏎  F1 STRATEGY REPORT  —  {driver_id}")
    print(f"{'═'*60}")
    windows = result["pitstop_windows"]
    if windows:
        print(f"\n  📍  PREDICTED PIT WINDOWS ({len(windows)} stop{'s' if len(windows)>1 else ''}):")
        for j, w in enumerate(windows, 1):
            print(f"      Stop {j}: Laps {w['start_lap']}–{w['end_lap']}  "
                  f"(Peak: {w['peak_prob']*100:.0f}%)  → Fit: {w['recommended_compound']}")
    else:
        print("\n  📍  No clear pit window above threshold.")
    print(f"\n  🔎  RACE FACTORS:")
    for f in result["factors"]:
        print(f"      {f['type']}  {f['factor']}")
        print(f"           {f['detail']}")
    print(f"\n{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════
# 8.  SYNTHETIC DATA GENERATOR  (unchanged from v1)
# ═══════════════════════════════════════════════════════════════

def generate_synthetic_race(
    race_laps=57, drivers=None, team="Team_A", seed=42
) -> pd.DataFrame:
    """Generate plausible synthetic race data for offline testing."""
    rng = np.random.default_rng(seed)
    if drivers is None:
        drivers = [f"{team}_D{i+1}" for i in range(2)]
    rows = []
    comps = ["SOFT", "MEDIUM", "HARD"]
    for driver in drivers:
        comp_idx = 0; tire_age = 0
        fuel = rng.uniform(100, 110); base_lap = rng.uniform(88, 92); deg = 0.0
        for lap in range(1, race_laps + 1):
            compound = comps[min(comp_idx, len(comps)-1)]
            tire_age += 1; deg = min(1.0, deg + rng.uniform(0.012, 0.025))
            fuel -= rng.uniform(1.6, 2.1); fuel = max(fuel, 0)
            pitstop = 0; next_comp = float("nan")
            if (tire_age > rng.integers(15, 22)) and comp_idx < len(comps)-1:
                pitstop = 1; comp_idx += 1
                next_comp = comps[min(comp_idx, len(comps)-1)]
                tire_age = 0; deg = rng.uniform(0.0, 0.05)
            lap_t = (base_lap + deg * rng.uniform(0.5, 2.5) + rng.normal(0, 0.15)
                     + (0.8 if compound == "SOFT" else 0.0 if compound == "MEDIUM" else 0.5))
            rainfall = max(0, rng.normal(0, 0.1)) if lap > 35 and rng.random() < 0.08 else 0.0
            rows.append({
                "lap_number": lap, "driver_id": driver, "team_id": team,
                "tire_compound": compound, "tire_age": tire_age,
                "tire_degradation": round(deg, 4),
                # fresh_tyre: 1 only on the very first lap of a new stint
                "fresh_tyre": int(tire_age == 1),
                "stint_number": comp_idx + 1,
                "lap_time_s": round(lap_t, 3), "fuel_load_kg": round(fuel, 2),
                "track_temp_c": round(rng.uniform(28, 45), 1),
                "air_temp_c":   round(rng.uniform(18, 30), 1),
                "rainfall_mm":  round(rainfall, 2),
                "humidity_pct": round(rng.uniform(35, 80), 1),
                "wind_speed_ms": round(rng.uniform(0, 12), 1),
                "safety_car_active": int(lap in [20, 21, 38]),
                "position":     rng.integers(1, 21),
                "gap_ahead_s":  round(rng.uniform(0, 10), 2),
                "gap_behind_s": round(rng.uniform(0, 10), 2),
                "pitstop_this_lap": pitstop,
                "next_compound": next_comp,
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 9.  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def run_pipeline(
    train_filepath:    str  = None,
    test_filepath:     str  = None,
    df_train:          pd.DataFrame = None,
    df_test:           pd.DataFrame = None,
    driver_to_analyse: str  = None,
    save_model:        bool = True,
):
    """
    Full pipeline: load → engineer → train → evaluate → visualise → save.

    Accepts EITHER filepaths (CSV/XLSX) OR pre-loaded DataFrames.
    When both are None, runs in synthetic demo mode.

    Train/test split is SEASONAL — not random.
    """

    # ── Step 1: Data loading ──────────────────────────────────────────
    if df_train is None:
        if train_filepath:
            df_train = load_and_validate(train_filepath)
        else:
            print("\n  ℹ  No data provided — running in SYNTHETIC DEMO mode.\n")
            df_train = generate_synthetic_race(race_laps=57, team="Team_A", seed=42)
            df_test  = generate_synthetic_race(race_laps=57, team="Team_B", seed=99)

    if df_test is None and test_filepath:
        df_test = load_and_validate(test_filepath)

    # ── Step 2: Feature engineering ──────────────────────────────────
    df_train_eng = engineer_features(df_train)
    df_test_eng  = engineer_features(df_test) if df_test is not None else None

    # ── Step 3: Select driver to analyse ─────────────────────────────
    if driver_to_analyse is None:
        driver_to_analyse = df_train_eng["driver_id"].iloc[0]
    print(f"  🎯  Primary driver: {driver_to_analyse}")

    # ── Step 4: Train model ───────────────────────────────────────────
    model = F1StrategyModel()
    model.fit(df_train_eng)

    # ── Step 5: Predict on training driver ───────────────────────────
    driver_df = df_train_eng[df_train_eng["driver_id"] == driver_to_analyse]
    result    = model.predict_race(driver_df)
    print_strategy_report(result, driver_to_analyse)

    # ── Step 6: Evaluate on test set (held-out season) ────────────────
    eval_result = None
    if df_test_eng is not None:
        test_label = str(df_test_eng.get("year", pd.Series(["Test"])).iloc[0]) \
            if "year" in df_test_eng.columns else "Test"
        eval_result = model.evaluate_season(df_test_eng, season_label=test_label)

    # ── Step 7: Visualisations ────────────────────────────────────────
    print(f"\n  🖼  Generating plots → {OUTPUT_DIR}/")
    paths = []
    paths.append(plot_tire_degradation(df_train_eng, driver_to_analyse))
    paths.append(plot_lap_times_comparison(df_train_eng))
    paths.append(plot_fuel_consumption(df_train_eng, driver_to_analyse))
    paths.append(plot_pitstop_probability(result, driver_to_analyse))
    paths.append(plot_tire_allocation(result, driver_to_analyse))
    paths.append(plot_weather_overlay(df_train_eng, driver_to_analyse))
    paths.append(plot_feature_importance(model))
    paths.append(plot_confusion_matrices(model))

    # Cross-team comparison (demo: two synthetic rivals or from test data)
    if df_test_eng is not None:
        compare_dfs = {"Test Season": df_test_eng}
    else:
        compare_dfs = {
            "Team_B": engineer_features(generate_synthetic_race(team="Team_B", seed=7)),
            "Team_C": engineer_features(generate_synthetic_race(team="Team_C", seed=13)),
        }
    paths.append(plot_team_comparison(df_train_eng, compare_dfs, model))

    # ── Step 8: Save model ────────────────────────────────────────────
    model_path = f"{OUTPUT_DIR}/f1_strategy_model_v2.joblib"
    if save_model:
        model.save(model_path)

    print(f"\n{'='*60}")
    print(f"  ✅  PIPELINE COMPLETE")
    print(f"{'='*60}")
    for p in paths:
        print(f"     📊 {p}")
    print(f"     💾 {model_path}")
    print(f"{'='*60}\n")

    return model, result, eval_result


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Usage examples:

    # Synthetic demo (no API, no files):
    python f1_strategy_model_2.py

    # With real FastF1 data (after running fastf1_connector.py):
    python f1_strategy_model_2.py --train f1_outputs/f1_train_2022.csv \\
                                   --test  f1_outputs/f1_test_2023.csv

    # Full API pull + train + evaluate:
    from fastf1_connector import build_corpus
    df_train, df_test = build_corpus(train_years=(2022,), test_years=(2023,))
    run_pipeline(df_train=df_train, df_test=df_test)
    """

    import argparse
    parser = argparse.ArgumentParser(description="F1 Strategy Model v2")
    parser.add_argument("--train", type=str, default=None, help="Training CSV/XLSX")
    parser.add_argument("--test",  type=str, default=None, help="Test CSV/XLSX")
    parser.add_argument("--driver", type=str, default=None, help="Driver ID to analyse")
    args = parser.parse_args()

    run_pipeline(
        train_filepath=args.train,
        test_filepath=args.test,
        driver_to_analyse=args.driver,
    )
