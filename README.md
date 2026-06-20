# F1 Pit-Stop strategy model via Juypter notebook


# F1 Strategy Notebook — IDE Reference Guide
**`f1_strategy_analysis.ipynb` · Cell-by-Cell Documentation**

University of Johannesburg · Faculty of Computer Science and Software Engineering  
Master's Research Project · Davies Adetiba assisted by Claude Pro.

---

## How to open the notebook

| Method | Command |
|---|---|
| **Jupyter Lab / Notebook** | `./venv/bin/jupyter notebook f1_strategy_analysis.ipynb` |
| **VS Code** | Open folder → click `f1_strategy_analysis.ipynb` in explorer |
| **Running server** | `http://localhost:8888/?token=<your-token>` |

> **Kernel to select:** `f1_model` (the project venv Python 3.13)

---

## Quick keyboard reference (VS Code + Jupyter)

| Action | VS Code | Jupyter |
|---|---|---|
| Run cell | `Shift+Enter` | `Shift+Enter` |
| Run all cells | `Ctrl+Shift+P → Run All` | `Kernel → Restart & Run All` |
| New cell below | Click `+` icon | `B` (command mode) |
| Delete cell | `Ctrl+Shift+P → Delete Cell` | `D D` (command mode) |
| Toggle code↔markdown | `M` / `Y` | `M` / `Y` (command mode) |
| Command mode | `Esc` | `Esc` |
| Edit mode | `Enter` | `Enter` |

---

## Section-by-section guide

---

### §1 · Setup & Libraries
**File:** Cell 2 (first code cell)

```python
import warnings, sys, os, numpy, pandas, matplotlib, plotly, joblib, sklearn
```

**What it does:**  
Imports all packages, sets the Plotly renderer to `'notebook'` (inline charts),
and applies the dark background matplotlib style. Defines the colour palette:

| Token | Hex | Used for |
|---|---|---|
| `BG` | `#0E1117` | All chart backgrounds |
| `ACC` | `#00D2BE` | Teal data accent (Mercedes-inspired) |
| `RED` | `#E8002D` | Pit stop markers, F1 red |
| `GOLD` | `#FFD700` | Threshold lines, pit stop markers |
| `COMPOUND_COLORS` | dict | Per-compound Pirelli palette |

**Output you will see:**  
A version table in a box-drawing frame confirming all libraries loaded.

**Common errors:**
- `ModuleNotFoundError: No module named 'plotly'` → Run `./venv/bin/pip install plotly`
- `ModuleNotFoundError: No module named 'f1_strategy_model_2'` → Make sure you are running with the `f1_model` kernel and the CWD is `/Users/daviesadetiba/f1_model`

---

### §2 · Load Data
**Files:** Cells 3–5

```python
df_train = pd.read_csv('f1_outputs/combined_train_2022_2023.csv')
df_test  = pd.read_csv('f1_outputs/combined_test_2024.csv')
```

**What it does:**  
Reads the two pre-built CSV files produced by `run_model.py` and stores them as DataFrames.
Then validates that all 24 required columns are present and prints null counts.

**Key variables created:**

| Variable | Shape (approx.) | Contents |
|---|---|---|
| `df_train` | ~47,000 × 25 | 2022 + 2023 training laps |
| `df_test` | ~26,000 × 25 | 2024 test laps |
| `df_all` | ~73,000 × 25 | Combined (EDA only) |

**Output you will see:**
1. A summary table: rows, races, pit stops, pit %
2. Schema validation report (missing cols, null counts)
3. `display(df_train.head(5))` — first 5 rows as a formatted table
4. Descriptive statistics table

**Why we do this:**  
Schema validation is part of DSRM Phase 3 (Design) — confirming the data contract
before any modelling begins.

---

### §3 · Exploratory Data Analysis (EDA)
**Files:** Cells 6–10

Five sub-sections, each producing one or more charts:

#### §3.1 — Class Imbalance (Cell 6)
```python
ax.bar(['No Pit Stop', 'Pit Stop'], counts.values, color=[ACC, RED])
```
**What it shows:** Two bar charts (train / test) showing how rare pit stops are (~3%).  
**Why it matters:** Justifies rejecting accuracy as a metric in favour of MCC + G-mean.

#### §3.2 — Pit Rate by Race (Cell 7)
```python
px.bar(race_stats, x='pit_rate_pct', y='short_name', orientation='h')
```
**What it shows:** Horizontal bar chart ranking every race by pit stop frequency.  
**Insight:** Wet-weather races (Brazil 2022, Japan 2022) have the highest pit rates.

#### §3.3 — Compound Usage (Cell 8)
```python
px.bar(comp_by_season, x='season', y='laps', color='tire_compound', barmode='group')
```
**What it shows:** How much each compound was used in each season.  
**Insight:** SOFT and MEDIUM dominate; WET/INTER only appear in wet races.

#### §3.4 — Lap Time Violin (Cell 9)
```python
go.Violin(x=[cmp]*n, y=df['lap_time_s'], box_visible=True, meanline_visible=True)
```
**What it shows:** Distribution of lap times for each compound — box plot inside violin.  
**Insight:** SOFT is fastest but has the widest spread; HARD is slowest but most consistent.

#### §3.5 — Weather Box Plots (Cell 10)
```python
go.Box(y=df_season[weather_col])
```
**What it shows:** Track temp, air temp, humidity, wind speed, rainfall — by season.  
**Insight:** `humidity_pct` and `wind_speed_ms` are features added in this artefact iteration.

---

### §4 · Feature Engineering
**Files:** Cells 11–12

```python
df_train_eng = engineer_features(df_train.copy())
```

**What `engineer_features()` adds:**

| New column | Formula | Meaning |
|---|---|---|
| `laptime_delta` | `lap_time_s.diff()` per driver | Lap-to-lap time change |
| `deg_rate` | `tire_degradation / tire_age` | Degradation per lap |
| `fuel_delta` | `fuel_load_kg.diff()` per driver | Fuel burn per lap |
| `wet_conditions` | `(rainfall > 0.5) OR (humidity > 85)` | Binary wet flag |
| `tire_compound_enc` | `LabelEncoder` on `tire_compound` | Numeric compound ID |

**Output you will see:**
1. Numbered list of all 21 features split by type
2. A horizontal bar chart: Pearson correlation of each feature vs `pitstop_this_lap`

**Reading the correlation chart:**  
- Red bars (positive) → feature value tends to be higher when a pit stop occurs  
- Blue bars (negative) → feature value tends to be lower when a pit stop occurs  
- Small absolute values are normal for a 3% event rate

---

### §5 · Load Trained Model
**Files:** Cells 13–14

```python
model = joblib.load('f1_outputs/f1_strategy_model_v2.joblib')
```

**What is `F1StrategyModel`?**  
A wrapper class that holds:

```
model
├── .pitstop_model    ← GradientBoostingClassifier  (binary: pit / no pit)
├── .compound_model   ← RandomForestClassifier       (multiclass: next compound)
├── .scaler           ← StandardScaler               (fitted on training features)
├── .feature_names    ← list[str]                    (21 features, in order)
├── .train_metrics    ← dict                         (CV averages)
└── .cv_metrics       ← dict                         (per-fold scores)
```

**Critical:** You must call `model.scaler.transform(X)` **before** passing `X` to
`model.pitstop_model.predict_proba()`. The scaler was fitted during training — using
it ensures the feature scale matches what the GBC was trained on.

**Output you will see:**
1. Component type summary
2. All 21 feature names numbered
3. GBC + RFC hyperparameter tables
4. Training CV metrics (if stored)

---

### §6 · Evaluation
**Files:** Cells 15–17

**Prediction pipeline:**
```python
X_test_scaled = model.scaler.transform(X_test)
y_proba = model.pitstop_model.predict_proba(X_test_scaled)[:, 1]
y_pred  = (y_proba >= 0.5).astype(int)
```

**Metrics computed:**

| Variable | Metric | Notes |
|---|---|---|
| `mcc` | Matthews Correlation Coefficient | Primary — range [-1, +1] |
| `gmean` | √(Sensitivity × Specificity) | Primary — range [0, 1] |
| `roc` | ROC-AUC | > 0.8 = good |
| `pr` | Precision-Recall AUC | Low is normal for 3% events |
| `f1` | F1-score | Harmonic mean P/R |
| `sens` | Sensitivity / Recall | TP / (TP+FN) |
| `spec` | Specificity | TN / (TN+FP) |

**Cell 16 — ROC + PR curves:**  
Two-panel Plotly chart. The dashed horizontal line on the PR plot is the
"random classifier" baseline = the pit stop rate (~0.03).

**Cell 17 — Confusion matrix:**  
A 2×2 heatmap. Important:
- `FN` (False Negatives) = missed pit stops — worst outcome for a race strategist
- `FP` (False Positives) = false alarms — annoying but safer than missing a real pit

---

### §7 · Strategy Visualisation
**Files:** Cells 18–19

**Configuration (Cell 18):**
```python
RACE_NAME = 'Monaco Grand Prix'   # ← change this
SEASON    = 2024                  # ← change this (2022, 2023, or 2024)
DRIVER    = None                  # ← None = auto-select winner; or e.g. 'VER'
```

**Cell 19 — 3-panel chart:**

| Panel | Y-axis | Colour rule |
|---|---|---|
| Top | Lap Time (s) | Line colour = tyre compound |
| Middle | Pit Stop Probability | Fill area = red below curve |
| Bottom | Tyre Degradation | Fill area = gold below curve |

Gold vertical dotted lines mark **actual pit stops** on all three panels.  
The gold dashed horizontal line at `y=0.5` is the classification threshold.

**Tip:** To compare two drivers, duplicate Cell 19 and change `DRIVER`.

---

### §8 · Multi-Driver Comparison
**File:** Cell 20

```python
px.scatter(df_race, x='lap_number', y='lap_time_s', color='tire_compound',
           symbol='driver_id', hover_data=['driver_id','tire_age','position'])
```

Shows all drivers in the selected race as a scatter. Hover over any point to see
driver, tyre age, and track position. The fastest laps per driver are printed below.

---

### §9 · Feature Importance
**File:** Cell 21

```python
importances = model.pitstop_model.feature_importances_
```

Plots a horizontal bar chart sorted by importance.  
Blue = base features · Red = engineered features.

**What to look for:**
- `lap_time_s` and `tire_degradation` are typically top-ranked → lap performance
  is the strongest signal
- `laptime_delta` (engineered) usually ranks in the top 5 → rate of change matters
  more than absolute values

---

### §10 · DSRM Summary
**File:** Cells 22–23 (markdown + code)

The final code cell re-prints all evaluation metrics and maps them to the DSRM
Phase 5 (Evaluation) targets. It also prints the commands to run each dashboard.

---

## Variables available at end of notebook

After running all cells, these variables persist in memory:

| Variable | Type | Contents |
|---|---|---|
| `df_train`, `df_test`, `df_all` | DataFrame | Raw CSV data |
| `df_train_eng`, `df_test_eng` | DataFrame | + engineered features |
| `model` | `F1StrategyModel` | Full trained pipeline |
| `y_test`, `y_proba`, `y_pred` | ndarray | Ground truth + predictions |
| `mcc`, `gmean`, `roc`, `pr` | float | All evaluation metrics |
| `df_race`, `df_drv` | DataFrame | Last selected race / driver |
| `COMPOUND_COLORS`, `BG`, `ACC`, `RED`, `GOLD` | str/dict | Colour tokens |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: f1_strategy_model_2` | Wrong kernel | Switch to `f1_model` kernel |
| `FileNotFoundError: combined_train_2022_2023.csv` | Pipeline not run | Run `./venv/bin/python run_model.py` |
| `FileNotFoundError: f1_strategy_model_v2.joblib` | Model not trained | Run `./venv/bin/python run_model.py` |
| Charts don't appear | Wrong renderer | Run `pio.renderers.default = 'notebook'` |
| Charts show in new tab | Renderer is `browser` | Change to `'notebook'` in §1 |
| `AttributeError: 'F1StrategyModel' has no attribute 'pit_model'` | Old notebook code | Use `model.pitstop_model` (with `_model`) |
| `NameError: y_proba` | §6a cell failed | Re-run §6a first |
| No output from §7b | `DRIVER` not in race | Check `df_race["driver_id"].unique()` |

---

## Running the dashboards alongside the notebook

Open additional terminal tabs:

```bash
# Terminal 1 — Jupyter Notebook
./venv/bin/jupyter notebook f1_strategy_analysis.ipynb

# Terminal 2 — Streamlit Premium Dashboard
./venv/bin/streamlit run dashboard_premium.py --server.port 8502

# Terminal 3 — Plotly Dash Dashboard
./venv/bin/python dashboard_dash.py

# Terminal 4 — Basic Streamlit Dashboard
./venv/bin/streamlit run dashboard.py --server.port 8501
```

---

*Generated for University of Johannesburg — Master's Research Project*  
*Peffers et al. (2007) Design Science Research Methodology*

