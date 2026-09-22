import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from match_predictor.model import train, predict_proba, walk_forward_cv, FEATURE_COLS, MARKET_COLS
from match_predictor.features import build_features, add_market_probs
from match_predictor.dixon_coles import DixonColesModel, backtest
from match_predictor.evaluation import score_probs

# resolved from this file so the entry point works from any directory
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "pl_matches_all.csv"

MATCH_KEY = ["date", "home_team", "away_team"]
PROB_COLS = ["p_home", "p_draw", "p_away"]


def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


def market_predictions(data):
    mkt = add_market_probs(data.copy())
    return mkt.rename(columns=dict(zip(MARKET_COLS, PROB_COLS)))


# ---------------------------------------------------------------------------
# tuning helpers
# ---------------------------------------------------------------------------

def tune_elo(data):
    k_values = [15, 20, 25, 30, 35, 40]
    ha_values = [50, 75, 100, 125, 150]

    print(f"{'K':>4} {'HA':>5} {'Accuracy':>10}")
    print("-" * 22)
    best_acc, best_k, best_ha = 0, None, None

    for k in k_values:
        for ha in ha_values:
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, acc = train(data, elo_k=k, elo_home_adv=ha, calibrate=False)
            print(f"{k:>4} {ha:>5} {acc:>10.4f}")
            if acc > best_acc:
                best_acc, best_k, best_ha = acc, k, ha

    print(f"\nBest: k={best_k}, home_adv={best_ha}, accuracy={best_acc:.4f}")


def tune_xgb(data):
    depth_values = [2, 3, 4]
    mcw_values = [1, 5, 10, 20]

    print(f"{'depth':>6} {'mcw':>5} {'Accuracy':>10}")
    print("-" * 25)
    best_acc, best_depth, best_mcw = 0, None, None

    for depth in depth_values:
        for mcw in mcw_values:
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, acc = train(data, max_depth=depth, min_child_weight=mcw, calibrate=False)
            print(f"{depth:>6} {mcw:>5} {acc:>10.4f}")
            if acc > best_acc:
                best_acc, best_depth, best_mcw = acc, depth, mcw

    print(f"\nBest: max_depth={best_depth}, min_child_weight={best_mcw}, accuracy={best_acc:.4f}")


# ---------------------------------------------------------------------------
# display helpers
# ---------------------------------------------------------------------------

def show_xgb_probabilities(data, model, le, n=10):
    features_df = build_features(data).sort_values(by="date")
    recent = features_df.tail(n).copy()
    recent[FEATURE_COLS] = recent[FEATURE_COLS].fillna(features_df[FEATURE_COLS].median())

    probs = predict_proba(model, le, recent)
    out = pd.concat(
        [recent[["date", "home_team", "away_team", "result"]].reset_index(drop=True),
         probs.reset_index(drop=True)],
        axis=1,
    )

    print(f"\n--- XGBoost win/draw/loss probabilities (last {n} matches) ---")
    with pd.option_context("display.max_columns", None, "display.width", 180,
                           "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))


def show_dixon_coles(dc_model, matches, n=10):
    print(f"\n--- Dixon-Coles: top scorelines + result probabilities (last {n} matches, in-sample fit) ---")

    for _, row in matches.tail(n).iterrows():
        ht, at = row["home_team"], row["away_team"]
        if ht not in dc_model.attack or at not in dc_model.attack:
            continue
        tops = dc_model.top_scorelines(ht, at, n=3)
        res  = dc_model.predict_result(ht, at)
        scoreline_str = "  ".join(f"{h}-{a}({p:.1%})" for h, a, p in tops)
        print(
            f"{str(row['date'].date()):>10}  {ht:<20} vs {at:<20}"
            f"  H:{res['HOME_TEAM']:.2f} D:{res['DRAW']:.2f} A:{res['AWAY_TEAM']:.2f}"
            f"  |  {scoreline_str}"
            f"  [actual: {row['result']}]"
        )


def show_team_strengths(dc_model, n=10):
    strengths = dc_model.team_strengths()
    print(f"\n--- Dixon-Coles team ratings (top {n} attack) ---")
    print(f"{'team':<22} {'attack':>8} {'defense':>9}")
    print("-" * 42)
    for team, row in strengths.head(n).iterrows():
        print(f"{team:<22} {row['attack']:>8.3f} {row['defense']:>9.3f}")
    print(f"\nhome advantage multiplier: {dc_model.home_adv:.3f}   rho: {dc_model.rho:.4f}")


def show_backtest_by_season(dc_oos, market):
    merged = dc_oos.merge(market[MATCH_KEY + PROB_COLS + ["season"]], on=MATCH_KEY,
                          suffixes=("", "_mkt"))
    print("\n--- Dixon-Coles out-of-sample backtest by season (weekly refits) ---")
    print(f"{'season':<10} {'matches':>7} {'DC acc':>7} {'DC rps':>7} {'mkt acc':>8} {'mkt rps':>8}")
    for season, g in merged.groupby("season"):
        dc = score_probs(g["result"], g[PROB_COLS].values)
        mk = score_probs(g["result"], g[[c + "_mkt" for c in PROB_COLS]].values)
        print(f"{season:<10} {len(g):>7} {dc['accuracy']:>7.4f} {dc['rps']:>7.4f}"
              f" {mk['accuracy']:>8.4f} {mk['rps']:>8.4f}")


def compare_models(data, predictions):
    """Scores every model on the same out-of-sample matches (the XGBoost walk-forward test folds)."""
    eval_set = next(iter(predictions.values()))[MATCH_KEY + ["result"]]
    start = eval_set["date"].min()

    # naive baseline: pre-evaluation-window outcome frequencies for every match
    rates = data[data["date"] < start]["result"].value_counts(normalize=True)
    base = np.tile([rates["HOME_TEAM"], rates["DRAW"], rates["AWAY_TEAM"]], (len(eval_set), 1))
    rows = {"Base rates": score_probs(eval_set["result"], base)}

    for name, frame in predictions.items():
        merged = eval_set.merge(frame[MATCH_KEY + PROB_COLS], on=MATCH_KEY, how="left")
        rows[name] = score_probs(merged["result"], merged[PROB_COLS].values)

    print(f"\n--- Out-of-sample comparison: {len(eval_set)} matches, "
          f"{start.date()} to {eval_set['date'].max().date()} (lower is better except accuracy) ---")
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(pd.DataFrame(rows).T.to_string())


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    data = load_data()

    # --- XGBoost ---
    model, le, acc = train(data)
    _, _, xgb_oos = walk_forward_cv(data, label="stats features")
    _, _, xgb_mkt_oos = walk_forward_cv(data, feature_cols=FEATURE_COLS + MARKET_COLS,
                                        label="stats + market odds")
    show_xgb_probabilities(data, model, le)

    # --- Dixon-Coles ---
    # fit on last 3 seasons only so ratings reflect current teams
    recent_seasons = sorted(data["season"].unique())[-3:]
    dc_data = data[data["season"].isin(recent_seasons)].copy()

    print(f"\n[fitting Dixon-Coles on {', '.join(recent_seasons)}...]")
    dc = DixonColesModel(xi=0.0065)
    dc.fit(dc_data)

    # in-sample (fit and scored on the same matches) -- optimistic, shown for reference
    dc_acc, dc_rps, _ = dc.evaluate(dc_data)
    print(f"Dixon-Coles in-sample accuracy: {dc_acc:.4f}  RPS: {dc_rps:.4f}")

    show_team_strengths(dc)
    show_dixon_coles(dc, dc_data)

    # honest estimate: refit weekly on past matches only, predict the week ahead
    dc_oos = backtest(data, start_date=xgb_oos["date"].min())
    market = market_predictions(data)
    show_backtest_by_season(dc_oos, market)

    compare_models(data, {
        "XGBoost (stats)":          xgb_oos,
        "XGBoost (stats + market)": xgb_mkt_oos,
        "Dixon-Coles":              dc_oos,
        "Market (bookmaker avg)":   market,
    })


if __name__ == "__main__":
    main()
