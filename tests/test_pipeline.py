from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from match_predictor.dixon_coles import DixonColesModel, backtest
from match_predictor.evaluation import rps, score_probs
from match_predictor.features import build_features, add_market_probs
from match_predictor.model import FEATURE_COLS

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "pl_matches_all.csv"
KEY = ["date", "home_team", "away_team"]


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df[df["season"].isin(["2023-2024", "2024-2025"])].reset_index(drop=True)


def scramble(df, mask):
    """Flip scores and stats on the masked rows so any leak would show up."""
    df = df.copy()
    df.loc[mask, "home_goals"] = 9
    df.loc[mask, "away_goals"] = 0
    df.loc[mask, "result"] = "HOME_TEAM"
    for col in ["home_shots", "home_shots_ot", "home_corners", "home_xg"]:
        df.loc[mask, col] = 50
    return df


def test_features_use_only_past_matches(data):
    last_day = data["date"].max()
    changed = scramble(data, data["date"] == last_day)

    before = build_features(data).set_index(KEY).sort_index()
    after = build_features(changed).set_index(KEY).sort_index()

    pd.testing.assert_frame_equal(before[FEATURE_COLS], after[FEATURE_COLS])


def test_market_probs_sum_to_one(data):
    probs = add_market_probs(data.copy())[["mkt_p_home", "mkt_p_draw", "mkt_p_away"]]
    assert probs.notna().all().all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_dixon_coles_probabilities_are_valid(data):
    model = DixonColesModel().fit(data)
    matrix = model.predict_scoreline("Arsenal", "Southampton")
    assert matrix.sum() == pytest.approx(1.0)

    res = model.predict_result("Arsenal", "Southampton")
    assert sum(res.values()) == pytest.approx(1.0)
    assert res["HOME_TEAM"] > res["AWAY_TEAM"]


def test_backtest_uses_only_past_matches(data):
    start = pd.Timestamp("2025-03-01")
    weeks = (data["date"].max() - start).days // 7
    last_block_start = start + pd.Timedelta(days=7 * weeks)
    changed = scramble(data, data["date"] >= last_block_start)

    before = backtest(data, start, refit_days=7)
    after = backtest(changed, start, refit_days=7)

    # a block's own results never feed its predictions, so nothing changes
    pd.testing.assert_frame_equal(
        before[KEY + ["p_home", "p_draw", "p_away"]],
        after[KEY + ["p_home", "p_draw", "p_away"]],
    )
    assert len(before) == (data["date"] >= start).sum()

    # sanity check: scrambling the week before does move the last block's predictions
    prev_week = (data["date"] >= last_block_start - pd.Timedelta(days=7)) & (data["date"] < last_block_start)
    moved = backtest(scramble(data, prev_week), start, refit_days=7)
    last = before["date"] >= last_block_start
    assert not np.allclose(before.loc[last, "p_home"], moved.loc[last, "p_home"])


def test_backtest_rates_unseen_teams_as_promoted(data):
    # start of 2024-25: Ipswich have no Premier League history in the window
    out = backtest(data, pd.Timestamp("2024-08-16"), refit_days=7).head(10)
    assert out[["p_home", "p_draw", "p_away"]].notna().all().all()
    np.testing.assert_allclose(out[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0)


def test_rps_and_scores():
    onehot = np.eye(3)
    assert rps(onehot, onehot) == 0.0
    # predicting away when home happened is worse than predicting a draw
    assert rps(onehot[[0]], np.array([[0, 0, 1.0]])) > rps(onehot[[0]], np.array([[0, 1.0, 0]]))

    scores = score_probs(["HOME_TEAM", "DRAW", "AWAY_TEAM"],
                         [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]])
    assert scores["accuracy"] == 1.0
