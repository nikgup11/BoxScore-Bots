# file for getting each teams defensive rating, pace, offensive rating
# using nba_api instead of scraping
import os
import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams as static_teams

OUT_DIR = "./data-collection/bbref_players_games_simple/"

# get player data
# hardcoded to SGA for now, update later to be modular
data = pd.read_csv(
    "./data-collection/bbref_players_games_simple/g/gilgesh01_Shai_Gilgeous-Alexander_last3.csv"
)

def season_end_to_str(season_end: int) -> str:
    """
    Convert season end year (e.g., 2024) to nba_api format '2023-24'.
    """
    start = season_end - 1
    return f"{start}-{str(season_end)[-2:]}"

def get_team_advanced_stats_for_season(season_end: int) -> dict:
    """
    For a given season_end (e.g. 2024), return a dict:
    { TEAM_ABBREVIATION: {"OppOffRtg": ..., "OppDefRtg": ..., "OppNetRtg": ..., "OppPace": ...}, ... }
    """
    season_str = season_end_to_str(season_end)

    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season_str,
        measure_type_detailed_defense="Advanced"
    )
    df = stats.get_data_frames()[0]

    # Build mapping from full team name -> abbreviation using static teams
    team_meta = static_teams.get_teams()
    name_to_abbr = {t["full_name"]: t["abbreviation"] for t in team_meta}

    # Keep only needed cols
    needed_cols = ["TEAM_NAME", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]
    df = df[needed_cols]

    out = {}
    for _, row in df.iterrows():
        full_name = row["TEAM_NAME"]
        abbr = name_to_abbr.get(full_name)
        if not abbr:
            # skip if we can't map name -> abbreviation
            continue
        out[abbr] = {
            "OppOffRtg": row["OFF_RATING"],
            "OppDefRtg": row["DEF_RATING"],
            "OppNetRtg": row["NET_RATING"],
            "OppPace": row["PACE"],
        }
    return out

# prepare new columns
new_cols = [
    "OppOffRtg",
    "OppDefRtg",
    "OppNetRtg",
    "OppPace",
]
for c in new_cols:
    if c not in data.columns:
        data[c] = None

# cache per-season stats so we only hit nba_api once per season
season_cache: dict[int, dict] = {}

# loop through games and attach stats
for idx, game in data.iterrows():
    team = str(game["Opp"])          # e.g. 'CHI', 'OKC'
    season_end = int(game["SEASON_END"])

    if season_end not in season_cache:
        season_cache[season_end] = get_team_advanced_stats_for_season(season_end)

    season_stats = season_cache[season_end]
    team_stats = season_stats.get(team)

    if team_stats:
        for col, val in team_stats.items():
            data.at[idx, col] = val

# save enriched player data
out_path = os.path.join(
    OUT_DIR, "gilgesh01_Shai_Gilgeous-Alexander_last3_with_opp_stats.csv"
)
data.to_csv(out_path, index=False)
print(f"Saved enriched data to {out_path}")