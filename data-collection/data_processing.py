# ==============================================================
# NBA DATA PIPELINE: DOWNLOAD → FILTER → ENRICH
# ==============================================================

import kagglehub
import os
import pandas as pd

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
# STEP 2: FILTER ACTIVE PLAYERS (2025-26)
# ======================================================

print("\n[STEP 2/3] Filtering active players for 2025-26 season...")

SEASON_START_DATE = pd.Timestamp("2025-10-01")

ID_COLUMNS = [
    "personId", "PERSON_ID", "player_id", "PLAYER_ID",
    "playerID", "id", "ID"
]

DATE_COLUMNS = [
    "GAME_DATE", "GAME_DATE_EST",
    "GAME_DATE_LOCAL", "gameDateTimeEst"
]

EXCLUDED_FILES = {"Players.csv", "Teams.csv", "TeamDetails.csv"}

def get_active_ids(data_dir):
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
            active_ids.update(df.loc[mask, id_col].dropna().astype(str).unique())
    return active_ids

active_ids = get_active_ids(kaggle_path)
print(f"  Found {len(active_ids)} active players.")

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
# STEP 3: ATTACH OPPONENT TEAM ADVANCED RATINGS
# ======================================================

print("\n[STEP 3/3] Attaching opponent team stats...")

player_df = filtered_dfs.get("PlayerStatistics.csv")
team_adv_df = filtered_dfs.get("TeamStatisticsAdvanced.csv")

if player_df is None:
    raise ValueError("❌ PlayerStatistics.csv not found!")
if team_adv_df is None:
    raise ValueError("❌ TeamStatisticsAdvanced.csv not found!")

print(f"  Player rows loaded: {len(player_df)}")
print(f"  Team advanced rows loaded: {len(team_adv_df)}")

# ------------------------------------------------------
# CLEAN PLAYER DATA
# ------------------------------------------------------

player_df["gameDateTimeEst"] = pd.to_datetime(
    player_df["gameDateTimeEst"].astype(str).str[:10], errors="coerce"
)
player_df = player_df.dropna(subset=["gameDateTimeEst"])
player_df = player_df[~player_df["gameType"].str.upper().isin(["PRESEASON", "ALL-STAR GAME"])]
print(f"  Rows after cleaning: {len(player_df)}")

# ------------------------------------------------------
# CLEAN TEAM DATA
# ------------------------------------------------------

# Keep only rows with non-null team names
team_adv_df = team_adv_df.dropna(subset=["teamName"])
team_adv_df["teamName"] = team_adv_df["teamName"].astype(str)
team_adv_df["TEAM_ABBR"] = team_adv_df["teamName"].str[:3].str.upper()

# ------------------------------------------------------
# CREATE TEAM ABBREVIATION MAP
# ------------------------------------------------------

team_lookup = {row["teamName"].lower(): row["TEAM_ABBR"] for _, row in team_adv_df.iterrows()}

def map_team(city, name):
    if not isinstance(name, str):
        return None
    combined = f"{city} {name}".strip().lower()
    return team_lookup.get(name.lower()) or team_lookup.get(combined)

player_df["OppTeamAbbr"] = player_df.apply(
    lambda x: map_team(x["opponentteamCity"], x["opponentteamName"]), axis=1
)

missing_abbr = player_df["OppTeamAbbr"].isna().sum()
print(f"  Opponent abbreviations missing: {missing_abbr}")

# ------------------------------------------------------
# MERGE OPPONENT RATINGS
# ------------------------------------------------------

ratings_df = (
    team_adv_df.groupby("TEAM_ABBR")[["eOffRating", "eDefRating", "eNetRating", "ePace"]]
    .mean()
    .reset_index()
)
ratings_df.columns = ["OppTeamAbbr", "OppOffRtg", "OppDefRtg", "OppNetRtg", "OppPace"]

# Use category dtype to save memory
player_df["OppTeamAbbr"] = player_df["OppTeamAbbr"].astype("category")
ratings_df["OppTeamAbbr"] = ratings_df["OppTeamAbbr"].astype("category")

player_df = player_df.merge(ratings_df, how="left", on="OppTeamAbbr", validate="m:1")

missing_after_merge = player_df["OppOffRtg"].isna().sum()
print(f"  Missing opponent ratings after merge: {missing_after_merge}")

# ------------------------------------------------------
# FILL MISSING WITH LEAGUE AVERAGES
# ------------------------------------------------------

league_avgs = {col: player_df[col].mean() for col in ["OppOffRtg", "OppDefRtg", "OppNetRtg", "OppPace"]}
for col, val in league_avgs.items():
    player_df[col] = player_df[col].fillna(val)

print("  Missing values filled with league averages.")

# ======================================================
# SAVE FINAL CLEAN DATA
# ======================================================

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
clean_dir = os.path.join(project_root, "data-collection", "clean_data")
os.makedirs(clean_dir, exist_ok=True)

clean_file = os.path.join(clean_dir, "PlayerStatistics.csv")
player_df.to_csv(clean_file, index=False)

print("\n" + "=" * 60)
print(f"✅ Final cleaned data saved to: {clean_file}")
print("=" * 60)
print("✅ PIPELINE COMPLETE!")