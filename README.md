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

## Model

XGBoost classifier, 3-way output (home win / draw / away win), trained with balanced class weights. Outputs win/draw/loss probabilities per match via `predict_proba` (uncalibrated).

## Evaluation

Two evaluation modes:

- **Hold-out**: 80/20 chronological split — ~50-52% accuracy depending on the window
- **Walk-forward CV**: 5 expanding-window folds in chronological order — ~50.3% mean accuracy, which is the more honest estimate (majority-class baseline is ~43%)

Log loss is reported alongside accuracy to track probability quality, not just argmax predictions.
