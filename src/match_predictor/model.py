from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from match_predictor.features import build_features
from match_predictor.evaluation import evaluate

FEATURE_COLS = [
    "ht_avg_gs", "at_avg_gs",
    "ht_avg_gc", "at_avg_gc",
    "ht_home_form", "ht_away_form",
    "ht_overall_form", "at_away_form",
    "at_overall_form", "at_home_form",
    "ht_avg_sot", "ht_avg_sot_against",
    "at_avg_sot", "at_avg_sot_against",
    "ht_draw_rate", "at_draw_rate",
    "ht_avg_corners", "ht_avg_corners_against",
    "at_avg_corners", "at_avg_corners_against",
]

TARGET = "result"


def train(df):
    le = LabelEncoder()
    features_df = build_features(df).sort_values(by='date')

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
    xgb = XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, random_state=1)

    xgb.fit(x_train, y_train, sample_weight=weights)

    xgb_pred = xgb.predict(x_test)

    evaluate("XGBoost", y_test, xgb_pred, le)

    return xgb


def predict(model, match_features):
    return model.predict(match_features[FEATURE_COLS])
