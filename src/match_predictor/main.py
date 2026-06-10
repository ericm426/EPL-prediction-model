import contextlib
import io
import pandas as pd
from match_predictor.model import train, predict, predict_proba, walk_forward_cv, FEATURE_COLS
from match_predictor.features import build_features


def load_data():
    dataframe = pd.read_csv('data/processed/pl_matches_all.csv', parse_dates=["date"])
    return dataframe


def tune_elo(data):
    k_values = [15, 20, 25, 30, 35, 40]
    ha_values = [50, 75, 100, 125, 150]

    print(f"{'K':>4} {'HA':>5} {'Accuracy':>10}")
    print("-" * 22)
    best_acc, best_k, best_ha = 0, None, None

    for k in k_values:
        for ha in ha_values:
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, acc = train(data, elo_k=k, elo_home_adv=ha)
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
                _, _, acc = train(data, max_depth=depth, min_child_weight=mcw)
            print(f"{depth:>6} {mcw:>5} {acc:>10.4f}")
            if acc > best_acc:
                best_acc, best_depth, best_mcw = acc, depth, mcw

    print(f"\nBest: max_depth={best_depth}, min_child_weight={best_mcw}, accuracy={best_acc:.4f}")


def show_sample_probabilities(data, model, le, n=10):
    # win/draw/loss probabilities for the most recent matches
    features_df = build_features(data).sort_values(by='date')
    recent = features_df.tail(n).copy()
    recent[FEATURE_COLS] = recent[FEATURE_COLS].fillna(features_df[FEATURE_COLS].median())

    probs = predict_proba(model, le, recent)
    out = pd.concat([recent[["date", "home_team", "away_team", "result"]].reset_index(drop=True),
                     probs.reset_index(drop=True)], axis=1)

    print(f"\n--- Win/Draw/Loss probabilities (last {n} matches) ---")
    with pd.option_context('display.max_columns', None, 'display.width', 160, 'display.float_format', '{:.3f}'.format):
        print(out.to_string(index=False))


def main():
    data = load_data()

    xgboost, le, acc = train(data)
    walk_forward_cv(data)
    show_sample_probabilities(data, xgboost, le)


if __name__ == "__main__":
    main()
