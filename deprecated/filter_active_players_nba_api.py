import os
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerbiostats

# ======================================================
# CONFIG
# ======================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data-collection", "raw_data")
OUT_DIR = os.path.join(PROJECT_ROOT, "data-collection", "active_data")

os.makedirs(OUT_DIR, exist_ok=True)

# 2025–26 NBA season start
SEASON_START_DATE = pd.Timestamp("2025-10-01")
SEASON_STRING = "2025-26"

ID_COLUMNS = [
    "personId", "PERSON_ID",
    "player_id", "PLAYER_ID",
    "playerID", "id", "ID"
]

DATE_COLUMNS = [
    "GAME_DATE",
    "GAME_DATE_EST",
    "GAME_DATE_LOCAL"
]

EXCLUDED_FILES = {
    "Players.csv",
    "Teams.csv",
    "TeamDetails.csv"
}

# ======================================================
# STEP 1: CHECK IF ANY 2025–26 GAMES EXIST
# ======================================================

def season_has_games(raw_dir):
    for fname in os.listdir(raw_dir):
        if not fname.lower().endswith(".csv"):
            continue

        df = pd.read_csv(os.path.join(raw_dir, fname), low_memory=False)

        date_col = next((c for c in DATE_COLUMNS if c in df.columns), None)
        if not date_col:
            continue

        dates = pd.to_datetime(df[date_col], errors="coerce")
        if (dates >= SEASON_START_DATE).any():
            return True

    return False


# ======================================================
# STEP 2A: ACTIVE PLAYERS FROM GAME DATA
# ======================================================

def get_active_ids_from_games(raw_dir):
    active_ids = set()

    print("Using game-based active detection (season started).")

    for fname in os.listdir(raw_dir):
        if not fname.lower().endswith(".csv"):
            continue
        if fname in EXCLUDED_FILES:
            continue

        df = pd.read_csv(os.path.join(raw_dir, fname), low_memory=False)

        id_col = next((c for c in ID_COLUMNS if c in df.columns), None)
        date_col = next((c for c in DATE_COLUMNS if c in df.columns), None)

        if not id_col or not date_col:
            continue

        dates = pd.to_datetime(df[date_col], errors="coerce")
        mask = dates >= SEASON_START_DATE

        if mask.any():
            active_ids.update(
                df.loc[mask, id_col]
                  .dropna()
                  .astype(str)
                  .unique()
            )

    print(f"Found {len(active_ids)} active players from games.")
    return active_ids


# ======================================================
# STEP 2B: ACTIVE PLAYERS FROM NBA API (PRE-SEASON)
# ======================================================

def get_active_ids_from_api():
    print("Using NBA API roster-based active detection (pre-season).")

    df = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
        season=SEASON_STRING
    ).get_data_frames()[0]

    return set(df["PLAYER_ID"].astype(str))


# ======================================================
# STEP 3: BUILD FINAL ACTIVE SET
# ======================================================

if season_has_games(RAW_DIR):
    final_active_ids = get_active_ids_from_games(RAW_DIR)
else:
    final_active_ids = get_active_ids_from_api()

if not final_active_ids:
    raise RuntimeError("Active player set is empty — this should not happen.")

print(f"Final active player count: {len(final_active_ids)}")

# ======================================================
# STEP 4: FILTER Players.csv
# ======================================================

players_path = os.path.join(RAW_DIR, "Players.csv")
players_df = pd.read_csv(players_path)

if "personId" not in players_df.columns:
    players_df.rename(columns={"PERSON_ID": "personId"}, inplace=True)

players_df["personId_str"] = players_df["personId"].astype(str)

active_players_df = players_df[
    players_df["personId_str"].isin(final_active_ids)
].copy()

active_players_df.drop(columns="personId_str", inplace=True)

active_players_df.to_csv(
    os.path.join(OUT_DIR, "Players.csv"),
    index=False
)

print(f"Players.csv: {len(players_df)} → {len(active_players_df)}")

# ======================================================
# STEP 5: FILTER ALL OTHER FILES
# ======================================================

print("Filtering remaining files...")

for fname in os.listdir(RAW_DIR):
    if not fname.lower().endswith(".csv") or fname == "Players.csv":
        continue

    df = pd.read_csv(os.path.join(RAW_DIR, fname), low_memory=False)

    id_col = next((c for c in ID_COLUMNS if c in df.columns), None)
    if not id_col:
        df.to_csv(os.path.join(OUT_DIR, fname), index=False)
        continue

    before = len(df)
    df[id_col] = df[id_col].astype(str)
    df = df[df[id_col].isin(final_active_ids)]
    after = len(df)

    print(f"{fname}: {before} → {after}")
    df.to_csv(os.path.join(OUT_DIR, fname), index=False)

print("\n✅ Active-player filtering complete for 2025–26 season.")
