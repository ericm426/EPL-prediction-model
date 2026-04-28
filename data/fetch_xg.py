import pandas as pd
from curl_cffi import requests
from pathlib import Path

TEAM_NAME_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
}


def normalize(name):
    return TEAM_NAME_MAP.get(name, name)


session = requests.Session()
session.get("https://understat.com/", impersonate="chrome120")


def fetch_season(season):
    league_url = f"https://understat.com/league/EPL/{season}"
    session.get(league_url, impersonate="chrome120")

    url = f"https://understat.com/getLeagueData/EPL/{season}"
    headers = {
        "Referer": league_url,
        "X-Requested-With": "XMLHttpRequest",
    }
    response = session.get(url, impersonate="chrome120", headers=headers)
    print(f"    Status: {response.status_code}")
    response.raise_for_status()

    data = response.json()
    matches = data["dates"]

    rows = []
    for match in matches:
        if not match.get("isResult"):
            continue
        rows.append({
            "date": pd.to_datetime(match["datetime"]).date(),
            "home_team": normalize(match["h"]["title"]),
            "away_team": normalize(match["a"]["title"]),
            "home_xg": float(match["xG"]["h"]),
            "away_xg": float(match["xG"]["a"]),
        })
    return rows


def main():
    csv_path = Path(__file__).parent / "processed" / "pl_matches_all.csv"
    df = pd.read_csv(csv_path, parse_dates=["date"])

    seasons = sorted({int(s.split("-")[0]) for s in df["season"].dropna().unique()})
    print(f"Fetching xG for seasons: {seasons}")

    all_rows = []
    for season in seasons:
        print(f"  Fetching {season}/{season + 1}...")
        rows = fetch_season(season)
        all_rows.extend(rows)
        print(f"    {len(rows)} matches fetched")

    xg_df = pd.DataFrame(all_rows)
    xg_df["date"] = pd.to_datetime(xg_df["date"])
    df["date"] = pd.to_datetime(df["date"])

    merged = df.merge(xg_df, on=["date", "home_team", "away_team"], how="left")

    matched = merged["home_xg"].notna().sum()
    total = len(merged)
    print(f"\nMerged {matched}/{total} matches with xG data")

    if matched < total:
        unmatched = merged[merged["home_xg"].isna()][["date", "home_team", "away_team"]]
        print(f"Unmatched rows:\n{unmatched.to_string()}")

    merged.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
