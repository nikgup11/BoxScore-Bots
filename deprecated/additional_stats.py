import os
import time
import pandas as pd
import requests
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams

# =========================
# CONFIG
# =========================
ACTIVE_FILE = "./data-collection/active_data/PlayerStatistics.csv"
CLEAN_FILE  = "./data-collection/clean_data/PlayerStatistics.csv"

print(f"Reading {ACTIVE_FILE}...")
df = pd.read_csv(ACTIVE_FILE, low_memory=False)

# 1. CLEANUP: Remove any old/empty stat columns
cols_to_remove = ["OppOffRtg", "OppDefRtg", "OppNetRtg", "OppPace"]
df = df.drop(columns=[c for c in cols_to_remove if c in df.columns], errors='ignore')

# 2. DATE FIX
df['gameDateTimeEst'] = pd.to_datetime(df['gameDateTimeEst'].astype(str).str[:10])

# 3. AUTO-MAPPER
nba_teams = teams.get_teams()
team_lookup = {}
for t in nba_teams:
    team_lookup[t['full_name']] = t['abbreviation']
    team_lookup[t['nickname']] = t['abbreviation']
    team_lookup[t['abbreviation']] = t['abbreviation']
    team_lookup[t['city']] = t['abbreviation']

def get_abbr(name):
    if not isinstance(name, str): return None
    return team_lookup.get(name.strip())

df['OppTeamAbbr'] = df['opponentteamName'].apply(get_abbr)

# 4. FETCH STATS FOR 2025-26
print("Fetching 2025-26 Season Stats...")
try:
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season='2025-26',
        measure_type_detailed_defense="Advanced",
        timeout=60
    )
    api_df = stats.get_data_frames()[0]
    
    stats_map = {}
    for _, row in api_df.iterrows():
        abbr = team_lookup.get(row['TEAM_NAME'])
        if abbr:
            stats_map[abbr] = {
                'OppOffRtg': row['OFF_RATING'],
                'OppDefRtg': row['DEF_RATING'],
                'OppNetRtg': row['NET_RATING'],
                'OppPace': row['PACE']
            }
except Exception as e:
    print(f"API Error: {e}")
    exit()

# 5. APPLY STATS (THE FIX IS HERE)
print("Attaching stats to rows...")

for index, row in df.iterrows():
    game_date = row['gameDateTimeEst']
    
    # LOGIC FIX: Check for Late 2025 OR Early 2026
    is_part_of_season = (
        (game_date.year == 2025 and game_date.month >= 10) or 
        (game_date.year == 2026 and game_date.month <= 7)
    )

    if is_part_of_season:
        opp_abbr = row['OppTeamAbbr']
        if opp_abbr in stats_map:
            vals = stats_map[opp_abbr]
            df.at[index, 'OppOffRtg'] = vals['OppOffRtg']
            df.at[index, 'OppDefRtg'] = vals['OppDefRtg']
            df.at[index, 'OppNetRtg'] = vals['OppNetRtg']
            df.at[index, 'OppPace']   = vals['OppPace']

# 6. SAVE
print(f"Saving to {CLEAN_FILE}...")
os.makedirs(os.path.dirname(CLEAN_FILE), exist_ok=True)
df.to_csv(CLEAN_FILE, index=False)

# VERIFICATION
filled_jan = df[(df['gameDateTimeEst'].dt.year == 2026) & (df['gameDateTimeEst'].dt.month == 1)]['OppPace'].count()
total_jan = len(df[(df['gameDateTimeEst'].dt.year == 2026) & (df['gameDateTimeEst'].dt.month == 1)])

print("-" * 30)
print(f"Jan 2026 Total Rows: {total_jan}")
print(f"Jan 2026 Stats Filled: {filled_jan}")
print("-" * 30)