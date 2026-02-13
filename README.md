# Implied Inflation Distributions and Moments (1Y)

This repository implements a full pipeline to recover **risk-neutral one-year inflation distributions and moments** from option prices.

The objective is **identification and stability**, not forecasting.

We compare multiple methods and analyze when higher moments are weakly identified due to limited strike coverage and boundary effects.

---
## Methods Implemented

### 1. Breeden–Litzenberger (BL)
- Prices → Density → Moments  
- Density recovered via second derivative of call prices:
  \[
  f(k) = \frac{1}{B(0,1)} \frac{\partial^2 C(k)}{\partial k^2}
  \]
- Sensitive to numerical noise (finite differentiation)

### 2. Kitsul–Wright (KW)
- Operational version of BL
- Local polynomial smoothing before differentiation
- Improves numerical stability

### 3. Bakshi–Kapadia–Madan (BKM)
- Prices → Moments directly
- No density recovery
- Highly sensitive to truncation and tail coverage

### 4. Maximum Entropy (MaxEnt)
- Prices → Density → Moments
- Discrete probability grid
- Regularized convex optimization
- Used as a robustness benchmark

---

## KW Boundary Shutdown Rule

Information window:
pi in [-1\%, 5\%]

---

## Repository Structure

```text
implied-inflation-moments/
├─ README.md
├─ requirements.txt
├─ LICENSE
├─ .gitignore
├─ src/
│ └─ options_implied_inflation_pdf.py
├─ results/
│ └─ updated_results.csv
├─ report/
│ ├─ main.tex
│ └─ figures/
└─ data/
└─ README_DATA.md
```


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
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### Run the pipeline
```python
python src\options_implied_inflation_pdf.py
```

## Authors

Nam Khanh Nguyen,
Rodrigue Mieuzet,
Melany Gipsy Moreno,
Khrystyna Kateryna Valenia,
Katarzyna Pastuszka

Master 2 Finance Technology Data (FTD)
Université Paris 1 Panthéon-Sorbonne
