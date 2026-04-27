import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from match_predictor.features import build_features

FEATURE_COLS = [
    "ht_avg_gs", "at_avg_gs", 
    "ht_avg_gc", "at_avg_gc",
    "ht_home_form", "ht_away_form",
    "ht_overall_form", "at_away_form",
    "at_overall_form", "at_home_form"
]

TARGET = "result" 

# train the model 
def train(df):
    features_df = build_features(df).sort_values(by='date')
    features_df[FEATURE_COLS] = features_df[FEATURE_COLS].fillna(0)
    split_index = int(len(features_df) * 0.8) 
    train = features_df.iloc[:split_index ]
    test = features_df.iloc[split_index:]

    x_train = train[FEATURE_COLS]
    y_train = train[TARGET]

    x_test = test[FEATURE_COLS]
    y_test = test[TARGET]

    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=1)
    clf.fit(x_train, y_train) 

    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(features_df.isnull().sum())

    return clf, accuracy

def predict(model, match_features):
    
    inputs = match_features[FEATURE_COLS]
    predictions = model.predict(inputs)

    return predictions