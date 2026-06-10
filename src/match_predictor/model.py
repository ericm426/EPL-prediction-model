import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, log_loss

from match_predictor.features import build_features
from match_predictor.evaluation import evaluate

FEATURE_COLS = [
    "ht_avg_gs", "at_avg_gs",
    "ht_avg_gc", "at_avg_gc",
    "ht_form_3", "at_form_3",
    "ht_overall_form", "at_overall_form",
    "ht_form_10", "at_form_10",
    "ht_home_form", "ht_away_form",
    "at_home_form", "at_away_form",
    "ht_avg_sot", "ht_avg_sot_against",
    "at_avg_sot", "at_avg_sot_against",
    "ht_draw_rate", "at_draw_rate",
    "ht_avg_corners", "ht_avg_corners_against",
    "at_avg_corners", "at_avg_corners_against",
    "ht_avg_xg", "ht_avg_xg_against",
    "at_avg_xg", "at_avg_xg_against",
    "ht_elo", "at_elo",
    "ht_xg_overperf", "at_xg_overperf",
    "ht_rest_days", "at_rest_days",
    "ht_season_ppg", "at_season_ppg",
    "ht_league_pos", "at_league_pos",
    "position_gap",
]

TARGET = "result"


def make_model(max_depth=3, min_child_weight=1):
    return XGBClassifier(n_estimators=500, max_depth=max_depth, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, random_state=1, min_child_weight=min_child_weight)


def train(df, elo_k=35, elo_home_adv=125, max_depth=3, min_child_weight=1):
    le = LabelEncoder()
    features_df = build_features(df, elo_k=elo_k, elo_home_adv=elo_home_adv).sort_values(by='date')

    split_index = int(len(features_df) * 0.8)
    train_df = features_df.iloc[:split_index].copy()
    test_df = features_df.iloc[split_index:].copy()

    col_medians = train_df[FEATURE_COLS].median()
    train_df[FEATURE_COLS] = train_df[FEATURE_COLS].fillna(col_medians)
    test_df[FEATURE_COLS] = test_df[FEATURE_COLS].fillna(col_medians)

    x_train = train_df[FEATURE_COLS]
    y_train = le.fit_transform(train_df[TARGET])

    x_test = test_df[FEATURE_COLS]
    y_test = le.transform(test_df[TARGET])

    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    xgb = make_model(max_depth=max_depth, min_child_weight=min_child_weight)

    xgb.fit(x_train, y_train, sample_weight=weights)

    xgb_proba = xgb.predict_proba(x_test)
    xgb_pred = xgb_proba.argmax(axis=1)

    acc = evaluate("XGBoost", y_test, xgb_pred, le, y_proba=xgb_proba)

    return xgb, le, acc


def walk_forward_cv(df, n_splits=5, elo_k=35, elo_home_adv=125, max_depth=3, min_child_weight=1, feature_cols=None):
    # expanding-window evaluation: train on everything before each fold,
    # test on the fold, so every fold respects chronological order
    feature_cols = feature_cols or FEATURE_COLS
    features_df = build_features(df, elo_k=elo_k, elo_home_adv=elo_home_adv).sort_values(by='date').reset_index(drop=True)

    le = LabelEncoder()
    le.fit(features_df[TARGET])

    n = len(features_df)
    fold_size = n // (n_splits + 1)

    accs, losses = [], []
    print(f"\n--- Walk-forward CV ({n_splits} folds) ---")
    print(f"{'fold':>4} {'train':>6} {'test':>5} {'accuracy':>9} {'log_loss':>9}")

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = n if i == n_splits else train_end + fold_size

        train_df = features_df.iloc[:train_end].copy()
        test_df = features_df.iloc[train_end:test_end].copy()

        col_medians = train_df[feature_cols].median()
        train_df[feature_cols] = train_df[feature_cols].fillna(col_medians)
        test_df[feature_cols] = test_df[feature_cols].fillna(col_medians)

        y_train = le.transform(train_df[TARGET])
        y_test = le.transform(test_df[TARGET])

        weights = compute_sample_weight(class_weight='balanced', y=y_train)
        xgb = make_model(max_depth=max_depth, min_child_weight=min_child_weight)
        xgb.fit(train_df[feature_cols], y_train, sample_weight=weights)

        proba = xgb.predict_proba(test_df[feature_cols])
        acc = accuracy_score(y_test, proba.argmax(axis=1))
        loss = log_loss(y_test, proba, labels=range(len(le.classes_)))

        accs.append(acc)
        losses.append(loss)
        print(f"{i:>4} {len(train_df):>6} {len(test_df):>5} {acc:>9.4f} {loss:>9.4f}")

    print(f"\nMean accuracy: {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")
    print(f"Mean log loss: {np.mean(losses):.4f}")
    return np.mean(accs), np.std(accs)


def predict(model, match_features):
    return model.predict(match_features[FEATURE_COLS])


def predict_proba(model, le, match_features):
    # win/draw/loss probabilities per match, one column per outcome
    proba = model.predict_proba(match_features[FEATURE_COLS])
    out = pd.DataFrame(proba, columns=le.classes_, index=match_features.index)
    out["predicted"] = le.inverse_transform(proba.argmax(axis=1))
    out["confidence"] = proba.max(axis=1)
    return out
