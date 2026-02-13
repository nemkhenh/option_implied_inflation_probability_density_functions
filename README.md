# Implied Inflation Distributions and Moments (1Y)

This project recovers **risk-neutral 1Y inflation distributions and moments** from option prices with a focus on **identification and stability (not forecasting)**.

Implemented methods:
- **Breeden–Litzenberger (BL)**: prices → density via second derivative
- **Kitsul–Wright (KW)**: local polynomial smoothing + BL identity (operational version)
- **Bakshi–Kapadia–Madan (BKM)**: model-free moments directly from option prices
- **Maximum Entropy (MaxEnt)**: regularized density consistent with observed prices (robustness benchmark)

A key diagnostic enhancement is a **KW boundary shutdown rule**: KW estimates are suppressed when the recovered density concentrates near the information-window boundaries \([-1\%, 5\%]\), which indicates boundary-driven curvature and weak tail identification.

---

## Repository layout

- `src/options_implied_inflation_pdf.py` : main pipeline (single file)
- `results/updated_results.csv` : generated moment time series
- `report/` : LaTeX report + figures
- `data/` : **empty** (raw Bloomberg data not included)

---

## Data (Bloomberg licensing)

Raw data files are sourced from Bloomberg and are therefore **not included** in this repository.

To reproduce the pipeline, place your own licensed data files in `data/`.

---

## Installation

### 1) Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
