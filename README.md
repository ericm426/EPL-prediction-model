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

## Model

XGBoost classifier, 3-way output (home win / draw / away win). Trained on an 80/20 chronological split with balanced class weights. ~52% accuracy on the test set (vs ~45% for a majority-class baseline).
