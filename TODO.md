                      
- [x] xG over/underperformance — rolling (goals - xG) tells you if a team is
finishing above expectation (likely to regress) or below (may be better than
results show). Computable from data we already have.

- [x] Rest days — days since last match for each team. Easy to compute from
existing dates. A team on 3 days rest vs 7 days rest is meaningful, especially
late season.

- [x] Win/draw/loss probabilities per match (model.predict_proba)

- [x] League position + season points-per-game features

- [x] Walk-forward (expanding window) cross-validation — single 80/20 split
was optimistic (~51.7%); honest walk-forward estimate is ~50.3%

- create probability distributions for certain results (full scoreline
distribution via Dixon-Coles poisson model)

- probability calibration (CalibratedClassifierCV) before using probabilities
against bookmaker odds

- (maybe) create frontend
