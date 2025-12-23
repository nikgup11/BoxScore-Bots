import os
import pandas as pd
from nba_api.stats.static import players as nba_players

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data-collection", "raw_data")
OUT_DIR = os.path.join(PROJECT_ROOT, "data-collection", "active_data")

os.makedirs(OUT_DIR, exist_ok=True)

# --- 1) Get active NBA players from nba_api (use player IDs, not names) ---
active_players = nba_players.get_active_players()
active_id_set = {str(p["id"]) for p in active_players}  # nba_api personId values

# --- 2) Load Players.csv from Kaggle and filter by personId ---
players_path = os.path.join(RAW_DIR, "Players.csv")
players_df = pd.read_csv(players_path)

if "personId" not in players_df.columns:
    raise ValueError("Players.csv must contain 'personId' column")

# Keep only rows whose personId is in the active NBA IDs
players_df["personId_str"] = players_df["personId"].astype(str)
active_players_df = players_df[players_df["personId_str"].isin(active_id_set)].copy()

# Use personId as the player identifier
player_id_col = "personId"
active_ids = set(active_players_df[player_id_col].astype(str))

# Save filtered players file
active_players_df.drop(columns=["personId_str"]).to_csv(
    os.path.join(OUT_DIR, "Players.csv"),
    index=False,
)

# --- 3) Filter all other CSVs by active player IDs when possible ---
for fname in os.listdir(RAW_DIR):
    if not fname.lower().endswith(".csv"):
        continue
    if fname == "Players.csv":
        continue

    src = os.path.join(RAW_DIR, fname)
    df = pd.read_csv(src)

    # Try to find a column that references player IDs (including personId)
    candidate_cols = ["personId", "PERSON_ID", "player_id", "PLAYER_ID", "id", "ID", "playerID"]
    filter_col = None
    for col in candidate_cols:
        if col in df.columns:
            filter_col = col
            break

    if filter_col is not None:
        df[filter_col] = df[filter_col].astype(str)
        df = df[df[filter_col].isin(active_ids)]

    out_path = os.path.join(OUT_DIR, fname)
    df.to_csv(out_path, index=False)

print(f"Filtered active-player data written to: {OUT_DIR}")