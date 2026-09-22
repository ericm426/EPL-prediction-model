- [x] xG over/underperformance — rolling (goals - xG) tells you if a team is
finishing above expectation (likely to regress) or below (may be better than
results show). Computable from data we already have.

- [x] Rest days — days since last match for each team. Easy to compute from
existing dates. A team on 3 days rest vs 7 days rest is meaningful, especially
late season.

- [x] Win/draw/loss probabilities per match — XGBoost predict_proba output with
confidence score per prediction.

- [x] League position + season points-per-game features

- [x] Walk-forward (expanding window) cross-validation — single 80/20 split
was optimistic (~51.7%); honest walk-forward estimate is ~50.3%

- [x] Full scoreline probability distribution — Dixon-Coles time-weighted
Poisson model fitted on last 3 seasons. Outputs NxN scoreline matrix, most
likely scorelines, and H/D/A probabilities. Also reports Ranked Probability
Score (RPS). Team attack/defense ratings match intuition (Arsenal, Man City top).

- [x] Probability calibration (Platt scaling) — implemented but does NOT help
at current dataset size. With ~400 calibration samples the logistic regression
meta-learner collapses draw predictions. Calibration needs either more data or
a different approach (temperature scaling). Keeping calibrate=False as default.

- [x] Dixon-Coles rolling backtest — refit weekly on past matches only;
out-of-sample accuracy 53.9% (in-sample was 55.2%). Per-season table in main.

- [x] Bookmaker odds as features + market baseline — odds now filled for
2015-2019 too (BbAv* columns). Market alone gets 55.7%, beating both models.

- [x] Dropped balanced class weights by default — +2.6 pts walk-forward
accuracy and better log loss, at the cost of almost never predicting draws.

- beat the market: model the residual vs bookmaker probabilities rather than
  the raw result (stats features add little on top of the odds right now)

- create frontend (Streamlit app — two team inputs, outputs XGBoost probs +
  D-C scorelines side by side)

- probability calibration with more data (temperature scaling on full dataset,
  or wait until more seasons accumulate)
