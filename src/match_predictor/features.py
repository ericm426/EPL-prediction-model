import pandas as pd 
import numpy as np

### KEY ### 
# ht = home team
# at = away team 
# gs = goals scored
# gc = goals conceded

# data shape into single df
def build_team_view(df):
    home_view = df[["date", "home_team", "home_goals", "away_goals", "result"]].rename(columns={
        "home_team": "team",
        "home_goals":"gs",
        "away_goals":"gc"
    })

    away_view = df[["date", "away_team", "away_goals", "home_goals", "result"]].rename(columns={
        "away_team": "team",
        "away_goals": "gs",
        "home_goals": "gc"
    })
    home_view["venue"] = "home"
    away_view["venue"] = "away" 

    team_view = pd.concat([home_view, away_view])

    return team_view

# rolling average of gs and gc
def build_rolling_stats(df):
    df = df.sort_values(by="date").reset_index(drop=True)

    win_con = ((df["venue"] == "home") & (df["result"] == "HOME_TEAM")) | ((df["venue"] == "away") & (df["result"] == "AWAY_TEAM"))
    draw = df["result"] == "DRAW"
    df["points"] = np.select([win_con, draw], [3, 1], default=0)

    df["avg_gs"] = df.groupby("team")["gs"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["avg_gc"] = df.groupby("team")["gc"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["overall_form"] = df.groupby("team")["points"].transform(lambda x: x.shift(1).rolling(5).sum())

    home_df = df[df["venue"] == "home"].copy()
    home_df["home_form"] = home_df.groupby("team")["points"].transform(lambda x: x.shift(1).rolling(5).sum())

    away_df = df[df["venue"] == "away"].copy()
    away_df["away_form"] = away_df.groupby("team")["points"].transform(lambda x: x.shift(1).rolling(5).sum())

    df = df.merge(home_df[["date", "team", "home_form"]], on=["date", "team"], how="left")
    df = df.merge(away_df[["date", "team", "away_form"]], on=["date", "team"], how="left")

    df = df.sort_values(["team", "date"])
    df["home_form"] = df.groupby("team")["home_form"].ffill()
    df["away_form"] = df.groupby("team")["away_form"].ffill()

    return df

# feature functions -> df for the model
def build_features(df):
    stat_cols = ["date", "team", "avg_gs", "avg_gc", "overall_form", "home_form", "away_form"]
    df_tomerge = build_rolling_stats(build_team_view(df))[stat_cols]

    df = df.merge(df_tomerge, left_on=["date", "home_team"], right_on=["date", "team"]).rename(columns={
        "avg_gs": "ht_avg_gs",
        "avg_gc": "ht_avg_gc",
        "overall_form": "ht_overall_form",
        "home_form": "ht_home_form",
        "away_form": "ht_away_form"
    }).drop(columns=["team"])

    df = df.merge(df_tomerge, left_on=["date", "away_team"], right_on=["date", "team"]).rename(columns={
        "avg_gs": "at_avg_gs",
        "avg_gc": "at_avg_gc",
        "overall_form": "at_overall_form",
        "home_form": "at_home_form",
        "away_form": "at_away_form"
    }).drop(columns=["team"])

    return df
