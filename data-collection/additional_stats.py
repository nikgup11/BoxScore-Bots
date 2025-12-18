# file for getting each teams defensive rating, pace, offensive rating
# add whatever else we get here as well
import os
import time
import datetime
import requests
import pandas as pd
import re
from bs4 import BeautifulSoup

BASE_INDEX = "https://www.basketball-reference.com/teams/{team}/{season}.html" 
# team ticker ex: MIA, NYK, etc.
# season i.e. 2023-2024 is 2024 season etc.
OUT_DIR = "./bbref_players_games_simple/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; simple-scraper/1.0)"}
S = requests.Session()
S.headers.update(HEADERS)

# get player data
data = pd.read_csv('./data-collection/bbref_players_games_simple/g/gilgesh01_Shai_Gilgeous-Alexander_last3.csv')
# ^ hardcoded to sga for now, update with modular for every player after

def get_team_season_stats(team: str, season: int) -> dict:
    url = BASE_INDEX.format(team=team, season=season)

    max_retries = 8
    for attempt in range(max_retries):
        resp = S.get(url)

        if resp.status_code == 429:
            wait = 30 * (attempt + 1)  # exponential-ish backoff
            print(f"429 from {url}, retry {attempt + 1}/{max_retries} ... waiting {wait}s")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        break
    else:
        print(f"Failed to fetch {url} after {max_retries} retries (still 429 or other error)")
        return {}

    soup = BeautifulSoup(resp.text, "lxml")

    meta = soup.find("div", id="meta")
    if meta is None:
        return {}

    # Flatten text and normalize "label : value" → "label: value"
    text = " ".join(meta.stripped_strings)
    text = re.sub(r"\s+:\s*", ": ", text)

    def grab(label: str):
        # allow spaces in label, and optional minus sign in value
        pattern = rf"{re.escape(label)}:\s*([+-]?\d+(?:\.\d+)?)"
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    return {
        "OppOffRtg": grab("Off Rtg"),
        "OppDefRtg": grab("Def Rtg"),
        "OppNetRtg": grab("Net Rtg"),
        "OppPace": grab("Pace"),
        "OppPPG": grab("PTS/G"),
        "OppOppPPG": grab("Opp PTS/G"),
        "OppSRS": grab("SRS"),
    }

# prepare new columns
new_cols = [
    "OppOffRtg",
    "OppDefRtg",
    "OppNetRtg",
    "OppPace",
    "OppPPG",
    "OppOppPPG",
    "OppSRS",
]
for c in new_cols:
    data[c] = None

stats_cache: dict[tuple[str, int], dict] = {}

for idx, game in data.iterrows():
    team = str(game["Opp"])
    season = int(game["SEASON_END"])

    key = (team, season)
    if key not in stats_cache:
        stats_cache[key] = get_team_season_stats(team, season)
        time.sleep(15)  # slower between *teams* to avoid 429s
    stats = stats_cache[key]
    for col, val in stats.items():
        data.at[idx, col] = val

# save enriched player data
out_path = OUT_DIR + "gilgesh01_Shai_Gilgeous-Alexander_last3_with_opp_stats.csv"
data.to_csv(out_path, index=False)
print(f"Saved enriched data to {out_path}")