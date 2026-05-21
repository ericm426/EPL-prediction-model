import contextlib
import io
import pandas as pd
from match_predictor.model import train, predict


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
                _, acc = train(data, elo_k=k, elo_home_adv=ha)
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
                _, acc = train(data, max_depth=depth, min_child_weight=mcw)
            print(f"{depth:>6} {mcw:>5} {acc:>10.4f}")
            if acc > best_acc:
                best_acc, best_depth, best_mcw = acc, depth, mcw

    print(f"\nBest: max_depth={best_depth}, min_child_weight={best_mcw}, accuracy={best_acc:.4f}")


def main():
    data = load_data()
    xgboost, acc = train(data)


if __name__ == "__main__":
    main()
