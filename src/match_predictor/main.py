import pandas as pd
from match_predictor.model import train, predict

def load_data():
    dataframe = pd.read_csv('data/processed/pl_matches_all.csv', parse_dates=["date"])
    return dataframe


def main():
    data = load_data()
    xgboost = train(data)


if __name__ == "__main__": 
    main()
