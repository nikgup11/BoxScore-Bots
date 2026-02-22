# combined data files into 1 file for easier running

import kagglehub
import os
import pandas as pd
import time

from nba_api.stats.endpoints import leaguedashteamstats, leaguedashplayerbiostats
from nba_api.stats.static import teams

print("=" * 60)
print("NBA DATA PIPELINE: DOWNLOAD → FILTER → ENRICH")
print("=" * 60)

# ======================================================
# STEP 1: DOWNLOAD & EXTRACT KAGGLE DATA
# ======================================================
print("\n[STEP 1/3] Downloading Kaggle dataset...")
kaggle_path = kagglehub.dataset_download(
    "eoinamoore/historical-nba-data-and-player-box-scores"
)
print(f"✅ Downloaded to: {kaggle_path}")

# ======================================================
# STEP 2: FILTER ACTIVE PLAYERS
# ======================================================
print("\n[STEP 2/3] Filtering active players for 2025-26 season...")

SEASON_START_DATE = pd.Timestamp("2025-10-01")
SEASON_STRING = "2025-26"
FALLBACK_SEASON = "2024-25"

ID_COLUMNS = ["personId", "PERSON_ID", "player_id", "PLAYER_ID",
              "playerID", "id", "ID"]

DATE_COLUMNS = ["GAME_DATE", "GAME_DATE_EST",
                "GAME_DATE_LOCAL", "gameDateTimeEst"]

EXCLUDED_FILES = {"Players.csv", "Teams.csv", "TeamDetails.csv"}


def get_active_ids_from_games(data_dir):
    active_ids = set()
    print("  Using game-based active detection.")

    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(".csv") or fname in EXCLUDED_FILES:
            continue

        df = pd.read_csv(os.path.join(data_dir, fname), low_memory=False)

        id_col = next((c for c in ID_COLUMNS if c in df.columns), None)
        date_col = next((c for c in DATE_COLUMNS if c in df.columns), None)

        if not id_col or not date_col:
            continue

        dates = pd.to_datetime(df[date_col], errors="coerce")
        mask = dates >= SEASON_START_DATE

        if mask.any():
            active_ids.update(
                df.loc[mask, id_col].dropna().astype(str).unique()
            )

    return active_ids


active_ids = get_active_ids_from_games(kaggle_path)
print(f"  Found {len(active_ids)} active players.")

# Filter all relevant files
filtered_dfs = {}

for fname in os.listdir(kaggle_path):
    if not fname.lower().endswith(".csv") or fname in EXCLUDED_FILES:
        continue

    df = pd.read_csv(os.path.join(kaggle_path, fname), low_memory=False)

    id_col = next((c for c in ID_COLUMNS if c in df.columns), None)

    if id_col:
        df[id_col] = df[id_col].astype(str)
        before = len(df)
        df = df[df[id_col].isin(active_ids)]
        after = len(df)
        print(f"  {fname}: {before} → {after}")

    filtered_dfs[fname] = df

print("✅ Active-player filtering complete.")

# ======================================================
# STEP 3: ENRICH WITH TEAM STATS
# ======================================================
print("\n[STEP 3/3] Enriching with opponent team stats...")

df = filtered_dfs.get('PlayerStatistics.csv')
if df is None:
    print("❌ PlayerStatistics.csv not found!")
    exit()

print(f"  Loaded {len(df)} rows")

# Remove old columns if they exist
cols_to_remove = ["OppOffRtg", "OppDefRtg", "OppNetRtg", "OppPace"]
df = df.drop(columns=[c for c in cols_to_remove if c in df.columns],
             errors='ignore')

# Fix dates
df['gameDateTimeEst'] = pd.to_datetime(
    df['gameDateTimeEst'].astype(str).str[:10],
    errors='coerce'
)

df = df.dropna(subset=['gameDateTimeEst'])
print(f"  Rows after dropping missing gameDateTimeEst: {len(df)}")

# Remove Preseason + All-Star
df = df[df['gameType'].str.upper() != 'PRESEASON']
df = df[df['gameType'].str.upper() != 'ALL-STAR GAME']
print(f"  Rows after filtering special games: {len(df)}")

# ======================================================
# TEAM MAPPING
# ======================================================
nba_teams = teams.get_teams()
team_lookup = {}

for t in nba_teams:
    team_lookup[t['full_name']] = t['abbreviation']
    team_lookup[t['nickname']] = t['abbreviation']
    team_lookup[t['abbreviation']] = t['abbreviation']
    team_lookup[t['city']] = t['abbreviation']


def get_abbr(name):
    if not isinstance(name, str):
        return None
    return team_lookup.get(name.strip())


df['OppTeamAbbr'] = df['opponentteamName'].apply(get_abbr)

# ======================================================
# FETCH TEAM STATS (WITH RETRY + FALLBACK)
# ======================================================
def fetch_team_stats(season):
    for attempt in range(3):
        try:
            print(f"  Attempting API fetch for season {season}...")
            stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense="Advanced",
                timeout=120
            )
            return stats.get_data_frames()[0]
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            time.sleep(3)

    return None


api_df = fetch_team_stats(SEASON_STRING)

if api_df is None:
    print("  ⚠️  Falling back to previous season...")
    api_df = fetch_team_stats(FALLBACK_SEASON)

if api_df is None:
    print("❌ API completely failed.")
    exit()

print("  ✅ Team stats fetched successfully.")

# ======================================================
# BUILD STATS MAP + MERGE (FAST)
# ======================================================
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

stats_df = pd.DataFrame.from_dict(stats_map, orient='index')
stats_df.index.name = 'OppTeamAbbr'

df = df.merge(
    stats_df,
    how='left',
    left_on='OppTeamAbbr',
    right_index=True
)

print("  ✅ Stats attached via merge (vectorized).")

# ======================================================
# SAVE FINAL CLEAN DATA
# ======================================================
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

clean_dir = os.path.join(
    project_root,
    "data-collection",
    "clean_data"
)

os.makedirs(clean_dir, exist_ok=True)

clean_file = os.path.join(clean_dir, "PlayerStatistics.csv")
df.to_csv(clean_file, index=False)

print(f"✅ Final cleaned data saved to: {clean_file}")

# Verification
filled_jan = df[
    (df['gameDateTimeEst'].dt.year == 2026) &
    (df['gameDateTimeEst'].dt.month == 1)
]['OppPace'].count()

total_jan = len(df[
    (df['gameDateTimeEst'].dt.year == 2026) &
    (df['gameDateTimeEst'].dt.month == 1)
])

print("\n" + "=" * 60)
print(f"Jan 2026 Total Rows: {total_jan}")
print(f"Jan 2026 Stats Filled: {filled_jan}")
print("=" * 60)
print("✅ PIPELINE COMPLETE!")