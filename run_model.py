"""
============================================================
F1 STRATEGY MODEL — MAIN RUNNER
============================================================
Training data : 2022 + 2023 seasons (Tracing Insights CSVs)
Test data     : 2024 season       (Tracing Insights CSVs)

All data is loaded from local CSV files — no internet needed.

Usage:
  cd /Users/daviesadetiba/f1_model
  ./venv/bin/python run_model.py

Outputs land in ./f1_outputs/
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

from data_connectors import build_combined_corpus
from f1_strategy_model_2 import run_pipeline

# ── 1. Load data ──────────────────────────────────────────────
df_train, df_test = build_combined_corpus(
    train_seasons        = [2022, 2023],   # ~44 races as training set
    test_seasons         = [2024],         # 24 races as unseen test set
    race_filter          = None,           # all races
    use_jolpica          = True,           # fallback if TI missing (won't be needed)
    use_kaggle           = False,
    use_tracing_insights = True,           # primary source — local CSVs only
    save_csv             = True,           # saves to f1_outputs/
)

# ── 2. Train + evaluate + visualise ───────────────────────────
run_pipeline(
    df_train          = df_train,
    df_test           = df_test,
    driver_to_analyse = None,    # auto-picks most active driver
    save_model        = True,    # saves model to f1_outputs/
)
