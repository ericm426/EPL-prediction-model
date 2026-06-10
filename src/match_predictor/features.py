import pandas as pd 
import numpy as np

### KEY ### 
# ht = home team
# at = away team 
# gs = goals scored
# gc = goals conceded

# data shape into single df
def build_team_view(df):
    home_view = df[["date", "season", "home_team", "home_goals", "away_goals", "home_shots", "away_shots", "home_shots_ot", "away_shots_ot", "home_corners", "away_corners", "home_xg", "away_xg", "result"]].rename(columns={
        "home_team": "team",
        "home_goals":"gs",
        "away_goals":"gc",
        "home_shots": "shots_attempted",
        "away_shots": "shots_conceded",
        "home_shots_ot": "sot",
        "away_shots_ot": "sot_against",
        "home_corners": "corners_given",
        "away_corners": "corners_conceded",
        "home_xg": "xg",
        "away_xg": "xg_against",
    })

    away_view = df[["date", "season", "away_team", "away_goals", "home_goals", "away_shots", "home_shots", "away_shots_ot", "home_shots_ot", "away_corners", "home_corners", "away_xg", "home_xg", "result"]].rename(columns={
        "away_team": "team",
        "away_goals": "gs",
        "home_goals": "gc",
        "away_shots": "shots_attempted",
        "home_shots": "shots_conceded",
        "away_shots_ot": "sot",
        "home_shots_ot": "sot_against",
        "away_corners": "corners_given",
        "home_corners": "corners_conceded",
        "away_xg": "xg",
        "home_xg": "xg_against",
    })

    home_view["venue"] = "home"
    away_view["venue"] = "away" 

    team_view = pd.concat([home_view, away_view])

    return team_view

# rolling average of gs and gc
def build_rolling_stats(df):
    df = df.sort_values(by="date").reset_index(drop=True)

    # np conditions
    win_con = ((df["venue"] == "home") & (df["result"] == "HOME_TEAM")) | ((df["venue"] == "away") & (df["result"] == "AWAY_TEAM"))
    draw = df["result"] == "DRAW"
    
    # points
    df["points"] = np.select([win_con, draw], [3, 1], default=0)

    # calc 5 game goal avg
    df["avg_gs"] = df.groupby("team")["gs"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["avg_gc"] = df.groupby("team")["gc"].transform(lambda x: x.shift(1).rolling(5).mean())
    
    # draws
    df["draw_rate"] = df.groupby("team")["points"].transform(lambda x: (x.shift(1) == 1).rolling(10).mean())

    # sot
    df["avg_sot"] = df.groupby("team")["sot"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["avg_sot_against"] = df.groupby("team")["sot_against"].transform(lambda x: x.shift(1).rolling(5).mean())

    # corners
    df["avg_corners"] = df.groupby("team")["corners_given"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["avg_corners_against"] = df.groupby("team")["corners_conceded"].transform(lambda x: x.shift(1).rolling(5).mean())

    # xG
    df["avg_xg"] = df.groupby("team")["xg"].transform(lambda x: x.shift(1).rolling(5).mean())
    df["avg_xg_against"] = df.groupby("team")["xg_against"].transform(lambda x: x.shift(1).rolling(5).mean())

    # xG over/underperformance: rolling (goals - xG), positive = finishing above expectation
    df["xg_diff"] = df["gs"] - df["xg"]
    df["xg_overperf"] = df.groupby("team")["xg_diff"].transform(lambda x: x.shift(1).rolling(5).mean())

    # rest days since last match (capped at 14 so off-season gaps don't distort)
    df["rest_days"] = df.groupby("team")["date"].diff().dt.days.clip(upper=14)

    # season context: pre-match points-per-game
    df["season_match_no"] = df.groupby(["season", "team"]).cumcount()
    season_points_pre = df.groupby(["season", "team"])["points"].transform(lambda x: x.shift(1).cumsum()).fillna(0)
    df["season_ppg"] = (season_points_pre / df["season_match_no"]).where(df["season_match_no"] > 0, 0.0)

    # form (3-game, 5-game, 10-game windows)
    df["form_3"] = df.groupby("team")["points"].transform(lambda x: x.shift(1).rolling(3).sum())
    df["overall_form"] = df.groupby("team")["points"].transform(lambda x: x.shift(1).rolling(5).sum())
    df["form_10"] = df.groupby("team")["points"].transform(lambda x: x.shift(1).rolling(10).sum())

    home_df = df[df["venue"] == "home"].copy()
    home_df["home_form"] = home_df.groupby("team")["points"].transform(lambda x: x.shift(1).rolling(5).sum())

    away_df = df[df["venue"] == "away"].copy()
    away_df["away_form"] = away_df.groupby("team")["points"].transform(lambda x: x.shift(1).rolling(5).sum())

    df = df.merge(home_df[["date", "team", "home_form"]], on=["date", "team"], how="left")
    df = df.merge(away_df[["date", "team", "away_form"]], on=["date", "team"], how="left")

    df = df.sort_values(["team", "date"])
    df["home_form"] = df.groupby("team")["home_form"].ffill()
    df["away_form"] = df.groupby("team")["away_form"].ffill()

    df = compute_league_positions(df)

    return df


def compute_league_positions(df):
    # table position (1-20) entering each match day; same-day results are
    # excluded so there's no leakage
    df = df.sort_values("date")
    df["cum_points"] = df.groupby(["season", "team"])["points"].cumsum()

    pos_frames = []
    for season, sdf in df.groupby("season"):
        pts = sdf.pivot_table(index="date", columns="team", values="cum_points", aggfunc="last").ffill()
        pts_pre = pts.shift(1).fillna(0)
        rank = pts_pre.rank(axis=1, ascending=False, method="min")
        season_pos = rank.stack().rename("league_pos").reset_index()
        season_pos["season"] = season
        pos_frames.append(season_pos)

    positions = pd.concat(pos_frames)
    return df.merge(positions, on=["date", "team", "season"], how="left").drop(columns=["cum_points"])

def compute_elo(df, k=35, home_advantage=125, initial=1500):
    # pre-match ratings are recorded before updating, so there's no data leakage
    df = df.sort_values("date").reset_index(drop=True)
    ratings = {}
    ht_elo, at_elo = [], []

    for _, row in df.iterrows():
        ht, at = row["home_team"], row["away_team"]
        r_h = ratings.get(ht, initial)
        r_a = ratings.get(at, initial)

        ht_elo.append(r_h)
        at_elo.append(r_a)

        # home_advantage shifts expected probability in favour of the home side
        e_h = 1 / (1 + 10 ** ((r_a - (r_h + home_advantage)) / 400))
        e_a = 1 - e_h

        # actual scores: win=1, draw=0.5, loss=0
        if row["result"] == "HOME_TEAM":
            s_h, s_a = 1.0, 0.0
        elif row["result"] == "AWAY_TEAM":
            s_h, s_a = 0.0, 1.0
        else:
            s_h, s_a = 0.5, 0.5

        # k controls how much a single result shifts the rating
        ratings[ht] = r_h + k * (s_h - e_h)
        ratings[at] = r_a + k * (s_a - e_a)

    df["ht_elo"] = ht_elo
    df["at_elo"] = at_elo
    return df


# feature functions -> df for the model
def build_features(df, elo_k=35, elo_home_adv=125):
    df = compute_elo(df, k=elo_k, home_advantage=elo_home_adv)

    stat_cols = ["date", "team", "avg_gs", "avg_gc", "form_3", "overall_form", "form_10", "avg_sot", "avg_sot_against", "avg_corners", "avg_corners_against", "avg_xg", "avg_xg_against", "draw_rate", "home_form", "away_form", "xg_overperf", "rest_days", "season_ppg", "league_pos"]
    df_tomerge = build_rolling_stats(build_team_view(df))[stat_cols]

    df = df.merge(df_tomerge, left_on=["date", "home_team"], right_on=["date", "team"]).rename(columns={
        "avg_gs": "ht_avg_gs",
        "avg_gc": "ht_avg_gc",
        "form_3": "ht_form_3",
        "overall_form": "ht_overall_form",
        "form_10": "ht_form_10",
        "avg_sot": "ht_avg_sot",
        "avg_sot_against": "ht_avg_sot_against",
        "draw_rate": "ht_draw_rate",
        "avg_corners": "ht_avg_corners",
        "avg_corners_against": "ht_avg_corners_against",
        "avg_xg": "ht_avg_xg",
        "avg_xg_against": "ht_avg_xg_against",
        "home_form": "ht_home_form",
        "away_form": "ht_away_form",
        "xg_overperf": "ht_xg_overperf",
        "rest_days": "ht_rest_days",
        "season_ppg": "ht_season_ppg",
        "league_pos": "ht_league_pos"
    }).drop(columns=["team"])

    df = df.merge(df_tomerge, left_on=["date", "away_team"], right_on=["date", "team"]).rename(columns={
        "avg_gs": "at_avg_gs",
        "avg_gc": "at_avg_gc",
        "form_3": "at_form_3",
        "overall_form": "at_overall_form",
        "form_10": "at_form_10",
        "avg_sot": "at_avg_sot",
        "avg_sot_against": "at_avg_sot_against",
        "draw_rate": "at_draw_rate",
        "avg_corners": "at_avg_corners",
        "avg_corners_against": "at_avg_corners_against",
        "avg_xg": "at_avg_xg",
        "avg_xg_against": "at_avg_xg_against",
        "home_form": "at_home_form",
        "away_form": "at_away_form",
        "xg_overperf": "at_xg_overperf",
        "rest_days": "at_rest_days",
        "season_ppg": "at_season_ppg",
        "league_pos": "at_league_pos"
    }).drop(columns=["team"])

    # table position gap: positive when the home side sits higher in the table
    df["position_gap"] = df["at_league_pos"] - df["ht_league_pos"]

    return df
