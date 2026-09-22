# Football Match Predictor

A ML model built to predict results of English Premier League fixtures, trained on every match from 2015-16 to the current season (~4,000 matches).

Data from https://www.football-data.co.uk/ (results, match stats, bookmaker odds) and https://understat.com/ (xG).

## Pipeline

```
pip install -e ".[dev]"
python data/standardize.py      # combine raw season CSVs into pl_matches_all.csv (keeps existing xG)
python data/fetch_xg.py         # merge xG data from understat.com
python -m match_predictor.main  # train and evaluate (runs from any directory)
pytest                          # leakage + sanity tests
```

## Features

- Rolling 5-game averages: goals scored/conceded, shots on target, corners, xG
- Form over 3, 5, and 10 game windows; split by home/away venue
- Draw rate (last 10 games)
- Elo ratings (updated pre-match to avoid leakage); promoted sides start at the relegated sides' average rating
- xG over/underperformance (rolling goals minus xG)
- Rest days since each team's last match
- Season context: points-per-game and league table position entering the match
- Optional: bookmaker-implied probabilities (market average odds, overround removed)

## Models

**XGBoost classifier** — 3-way output (home win / draw / away win) via `predict_proba`. Trained unweighted by default: balanced class weights (`balanced=True`) raise draw recall but cost ~2.5 points of accuracy and worsen log loss/Brier/RPS. Platt scaling calibration is implemented but disabled by default (collapses minority classes at current dataset size).

**Dixon-Coles Poisson model** — time-weighted attack/defense ratings per team fitted via MLE (analytic gradient, ~0.03s per fit). Outputs a full scoreline probability matrix. Home advantage and low-score correction (rho) are jointly fitted. `backtest()` refits weekly on past matches only and rates newly promoted teams as the average of the three weakest sides.

## Evaluation

All models scored on the same 3,393 out-of-sample matches (Apr 2017 – Feb 2026, the XGBoost walk-forward test folds):

| Model | Accuracy | Log loss | Brier | RPS |
|---|---|---|---|---|
| Base rates (always the historical H/D/A mix) | 44.2% | 1.069 | 0.647 | 0.234 |
| XGBoost (stats) | 52.8% | 0.990 | 0.589 | 0.205 |
| XGBoost (stats + market odds) | 53.5% | 0.974 | 0.578 | 0.200 |
| Dixon-Coles (weekly refit backtest) | 53.9% | 0.977 | 0.579 | 0.201 |
| Bookmaker market average | **55.7%** | **0.951** | **0.563** | **0.194** |

The bookmaker market is still the benchmark to beat; neither model does yet. Dixon-Coles scored on its own training data reports ~55%, which is optimistic — the backtest is the honest number.
