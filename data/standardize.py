import pandas as pd
import glob
import re
import os

COLUMN_MAP = {
    'Date':     'date',
    'HomeTeam': 'home_team',
    'AwayTeam': 'away_team',
    'FTHG':     'home_goals',
    'FTAG':     'away_goals',
    'FTR':      'result',
    'HTHG':     'home_goals_ht',
    'HTAG':     'away_goals_ht',
    'HS':       'home_shots',
    'AS':       'away_shots',
    'HST':      'home_shots_ot',
    'AST':      'away_shots_ot',
    'HC':       'home_corners',
    'AC':       'away_corners',
    'Referee':  'referee',
    'AvgH':     'odds_home',
    'AvgD':     'odds_draw',
    'AvgA':     'odds_away',
}

RESULT_MAP = {'H': 'HOME_TEAM', 'A': 'AWAY_TEAM', 'D': 'DRAW'}

def parse_season(filename):
    # e.g. PL_24-25.csv -> 2024-2025
    match = re.search(r'(\d{2})-(\d{2})', filename)
    if match:
        start, end = match.groups()
        return f"20{start}-20{end}"
    return None

def standardize_file(path):
    df = pd.read_csv(path)
    df = df[[c for c in COLUMN_MAP if c in df.columns]].rename(columns=COLUMN_MAP)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['result'] = df['result'].map(RESULT_MAP)
    df['season'] = parse_season(os.path.basename(path))
    return df

processed_dir = os.path.join(os.path.dirname(__file__), 'processed')
files = sorted(f for f in glob.glob(os.path.join(processed_dir, 'PL_*.csv')) if re.search(r'PL_\d{2}-\d{2}\.csv$', f))
combined = pd.concat([standardize_file(f) for f in files], ignore_index=True)
combined = combined.sort_values('date').reset_index(drop=True)

cols = ['date', 'season', 'home_team', 'away_team',
        'home_goals', 'away_goals', 'result',
        'home_goals_ht', 'away_goals_ht',
        'home_shots', 'away_shots', 'home_shots_ot', 'away_shots_ot',
        'home_corners', 'away_corners', 'referee',
        'odds_home', 'odds_draw', 'odds_away']
combined = combined[cols]

out_path = os.path.join(processed_dir, 'pl_matches_all.csv')
combined.to_csv(out_path, index=False)
print(f"Saved {len(combined)} matches to {out_path}")
