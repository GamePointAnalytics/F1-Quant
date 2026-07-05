# How to run this

## Prerequisites

You need Python 3.9+ installed. Check with:
```
python --version
```

If you don't have it: https://www.python.org/downloads/

---

## One-time setup

From the repo root (`QuantTradingSystem-rebuild/`), install the dependencies:

```
pip install -r requirements.txt
```

This takes a few minutes the first time. You only need to do it once (or when requirements.txt changes).

---

## Running the notebook

From the repo root, launch Jupyter:

```
jupyter notebook
```

A browser tab opens showing the file tree. Click into `notebooks/` and open `01_race_analysis.ipynb`.

---

## Running cells

| Action | Shortcut |
|--------|----------|
| Run the current cell and move to next | **Shift + Enter** |
| Run the current cell and stay | **Ctrl + Enter** |
| Run all cells top to bottom | **Kernel → Restart & Run All** |

Run cells in order from top to bottom — later cells import from earlier ones.

**First run only:** when the notebook hits `get_accurate_laps(2023, 'Bahrain')`, FastF1 downloads the session data from the API. This takes around 30 seconds. After that it's cached locally in `.fastf1_cache/` and every subsequent run is instant.

---

## If something errors

**`ModuleNotFoundError`** — did you run `pip install -r requirements.txt`? If yes, make sure you're running Jupyter from the repo root, not from inside `notebooks/`.

**`KeyError` on a column name** — the FastF1 API occasionally changes column names between versions. Check `laps.columns` to see what's actually there.

**HMM convergence warning** — normal for short stints. The `detect_regimes` function returns an empty DataFrame for those; the rest of the analysis is unaffected.

---

## Running individual Python modules

If you want to test a module directly without Jupyter:

```
cd QuantTradingSystem-rebuild
python -c "from data.fastf1_loader import get_accurate_laps; laps = get_accurate_laps(2023, 'Bahrain'); print(laps.head())"
```

The `analysis/` modules don't do anything on their own — they're function libraries. Use the notebook or a short script to call them.
