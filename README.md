# Football Match Predictor

A ML model built to predict results of English Premier League fixtures using data from past 5 seasons.

Data from https://www.football-data.co.uk/ and https://www.football-data.org/.

## Pipeline

```
python data/standardize.py   # combine raw season CSVs into pl_matches_all.csv
python data/fetch_xg.py      # merge xG data from understat.com
python -m match_predictor.main  # train and evaluate
```

## Features

- Rolling 5-game averages: goals scored/conceded, shots on target, corners, xG
- Form over 3, 5, and 10 game windows; split by home/away venue
- Draw rate (last 10 games)
- Elo ratings (updated pre-match to avoid leakage)
- xG over/underperformance (rolling goals minus xG)
- Rest days since each team's last match
- Season context: points-per-game and league table position entering the match

## Models

**XGBoost classifier** — 3-way output (home win / draw / away win), trained with balanced class weights. Outputs win/draw/loss probabilities per match via `predict_proba`. Platt scaling calibration is implemented but disabled by default (collapses minority classes at current dataset size).

**Dixon-Coles Poisson model** — time-weighted attack/defense ratings per team fitted on the last 3 seasons via MLE. Outputs a full scoreline probability matrix and ranked probability score (RPS). Home advantage and low-score correction (rho) are jointly fitted.

## Evaluation

- **XGBoost hold-out** (70/20 split, 10% calibration holdout): ~49-52% accuracy
- **XGBoost walk-forward CV** (5 expanding folds): ~50.3% mean (±1.1%) — the honest estimate
- **Dixon-Coles in-sample** (last 3 seasons): ~55% accuracy, RPS ≈ 0.19
- **Baseline** (always predict home win): ~43%

Brier score and log loss are reported alongside accuracy to track probability quality.
