# file for getting each team's defensive rating, pace, offensive rating
# using nba_api instead of scraping
import os
import time
import pandas as pd
import requests
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams as static_teams

OUT_DIR = "./data-collection/clean_data/"

data = pd.read_csv(
    "./data-collection/active_data/PlayerStatistics.csv",
    dtype={"gameLabel": "string", "gameSubLabel": "string"},
    low_memory=False,
)

# drop games where player did not play (0 minutes)
data = data[data["numMinutes"] > 0].copy()

def season_end_to_str(season_end: int) -> str:
    """
    Convert season end year (e.g., 2024) to nba_api format '2023-24'.
    """
    start = season_end - 1
    return f"{start}-{str(season_end)[-2:]}"

def infer_season_end_from_date(date_str: str) -> int:
    """
    Infer NBA season end year from a game date:
    - Games in Oct–Dec belong to season that ends next calendar year.
    - Games in Jan–Jun belong to season ending current calendar year.
    """
    dt = pd.to_datetime(date_str)
    return dt.year + 1 if dt.month >= 10 else dt.year

# Build mapping from various team names to abbreviation once
_team_meta = static_teams.get_teams()
name_to_abbr = {}
for t in _team_meta:
    abbr = t["abbreviation"]
    name_to_abbr[t["full_name"]] = abbr
    name_to_abbr[t["nickname"]] = abbr
    name_to_abbr[abbr] = abbr  # allow abbreviations directly

def normalize_team_to_abbr(raw_name: str) -> str | None:
    if not isinstance(raw_name, str):
        return None
    raw_name = raw_name.strip()
    return name_to_abbr.get(raw_name)

def get_team_advanced_stats_for_season(season_end: int, max_retries: int = 3) -> dict:
    """
    Fetch advanced team stats for a season with retries on timeout.
    Returns {} on failure so caller can safely continue.
    """
    season_str = season_end_to_str(season_end)
    for attempt in range(max_retries):
        try:
            stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season_str,
                measure_type_detailed_defense="Advanced",
                timeout=60,
            )
            df = stats.get_data_frames()[0]
            needed_cols = ["TEAM_NAME", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]
            df = df[needed_cols]

            out = {}
            for _, row in df.iterrows():
                full_name = row["TEAM_NAME"]
                abbr = normalize_team_to_abbr(full_name)
                if not abbr:
                    continue
                out[abbr] = {
                    "OppOffRtg": row["OFF_RATING"],
                    "OppDefRtg": row["DEF_RATING"],
                    "OppNetRtg": row["NET_RATING"],
                    "OppPace": row["PACE"],
                }
            return out
        except requests.exceptions.ReadTimeout:
            wait = 2 * (attempt + 1)
            print(f"Timeout for season {season_str}, retry {attempt + 1}/{max_retries} in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"Error fetching stats for season {season_str}: {e}")
            break

    print(f"Failed to fetch stats for season {season_str}, leaving Opp* columns empty.")
    return {}

# prepare new columns
new_cols = ["OppOffRtg", "OppDefRtg", "OppNetRtg", "OppPace"]
for c in new_cols:
    if c not in data.columns:
        data[c] = None

# cache per-season stats so we only hit nba_api once per season
season_cache: dict[int, dict] = {}

# loop through games and attach stats
for idx, game in data.iterrows():
    opp_name = game.get("opponentteamName")
    team_abbr = normalize_team_to_abbr(opp_name)
    if not team_abbr:
        print(f"Skipping unknown team: {opp_name}")
        continue

    game_date = game.get("gameDateTimeEst")
    if pd.isna(game_date):
        continue

    season_end = infer_season_end_from_date(game_date)

    if season_end not in season_cache:
        season_cache[season_end] = get_team_advanced_stats_for_season(season_end)

    season_stats = season_cache.get(season_end, {})
    team_stats = season_stats.get(team_abbr)

    if team_stats:
        for col, val in team_stats.items():
            data.at[idx, col] = val

# save enriched player data
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "PlayerStatistics.csv")
data.to_csv(out_path, index=False)
print(f"Saved enriched data to {out_path}")
