import pandas as pd 

### KEY ### 
# ht = home team
# at = away team 
# gs = goals scored
# gc = goals conceded

# average goals scored in the past 5 matches
def avg_goals_scored(team, date, df):
    result = df[(df["date"] < date) & ((df["home_team"] == team) | (df["away_team"] == team))].tail(5)
    
    # edge case 
    if result.empty:
        return 0

    home_goals = result[result["home_team"] == team]["home_goals"].sum()
    away_goals = result[result["away_team"] == team]["away_goals"].sum()

    goals_scored = home_goals + away_goals

    # if not 5 matches, however many are available
    avg = goals_scored / len(result)
    return avg
    


# average goals conceded in the past 5 matches
def avg_goals_conceded(team, date, df):
    result = df[(df["date"] < date) & ((df["home_team"] == team) | (df["away_team"] == team))].tail(5)
    
    # edge case 
    if result.empty:
        return 0

    home_goals = result[result["home_team"] == team]["away_goals"].sum()
    away_goals = result[result["away_team"] == team]["home_goals"].sum()

    goals_conceded = home_goals + away_goals
    
    # if not 5 matches, however many are available
    avg = goals_conceded / len(result)
    return avg

# Form in the last 5 matches
def calculate_form(team, date, df):
    matches = df[(df["date"] < date) & ((df["home_team"] == team) | (df["away_team"] == team))].tail(5)
    form = 0

    if matches.empty:
        return (0)
    
    result_value = {
        "win": 3, 
        "draw": 1,
        "loss": 0
    }

    for index, rows in matches.iterrows():
        if rows["home_team"] == team:
            if rows["result"] == "HOME_TEAM":
                outcome = "win"
            elif rows["result"] == "AWAY_TEAM":
                outcome = "loss"
            else:
                outcome = "draw"
        else:
            if rows["result"] == "AWAY_TEAM":
                outcome = "win"
            elif rows["result"] == "HOME_TEAM":
                outcome = "loss"
            else:
                outcome = "draw"

        form += result_value[outcome]

    return form    
    
# home form in last 5
def home_form(team, date, df):
    matches = df[(df["date"] < date) & (df["home_team"] == team)].tail(5)
    form = 0

    if matches.empty:
        return (0)
    
    result_value = {
        "win": 3, 
        "draw": 1,
        "loss": 0
    }

    for index, rows in matches.iterrows():
        if rows["result"] == "HOME_TEAM":
            outcome = "win"
        elif rows["result"] == "AWAY_TEAM":
            outcome = "loss"
        else:
            outcome = "draw"

        form += result_value[outcome]

    return form    

def away_form(team, date, df):
    matches = df[(df["date"] < date) & (df["away_team"] == team)].tail(5)
    form = 0

    if matches.empty:
        return (0)
    
    result_value = {
        "win": 3, 
        "draw": 1,
        "loss": 0
    }

    for index, rows in matches.iterrows():
        if rows["result"] == "HOME_TEAM":
            outcome = "loss"
        elif rows["result"] == "AWAY_TEAM":
            outcome = "win"
        else:
            outcome = "draw"

        form += result_value[outcome]

    return form    

# feature functions -> new df for the model
def build_features(df):
    data = []

    for index, row, in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]
        date = row["date"]

        features = {
            "date": date,
            "home_team": home_team,
            "away_team": away_team,
            "ht_avg_gs": avg_goals_scored(home_team, date, df),
            "at_avg_gs": avg_goals_scored(away_team, date, df),
            "ht_avg_gc": avg_goals_conceded(home_team, date, df),
            "at_avg_gc": avg_goals_conceded(away_team, date, df),
            "ht_home_form": home_form(home_team, date, df),
            "at_away_form": away_form(away_team, date, df),
            "ht_overall_form": calculate_form(home_team, date, df),
            "at_overall_form": calculate_form(away_team, date, df),
            "result": row["result"]
        }

        data.append(features)
        

    return pd.DataFrame(data)
