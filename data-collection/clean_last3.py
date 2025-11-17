import pandas as pd
import os

player_folder = "./espn_stats_getter/bbref_players_games_simple/"

DROP_STRINGS = {"Did Not Play", "Did Not Dress", "Inactive", "Not With Team", "Did Not Play/Out"}

for file in os.listdir(player_folder):
    if not file.endswith("_last3.csv"):
        continue
    path = os.path.join(player_folder, file)
    df = pd.read_csv(path)

    # normalize string markers to NaN (so numeric coercion will fail on non-games)
    df = df.replace(list(DROP_STRINGS), pd.NA)

    # Prefer explicit Date parsing; drop rows where Date isn't a valid date (these include repeated headers)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
    else:
        # if no Date column, skip cleaning this file
        print(f"no Date column in {file}, skipping")
        continue

    # coerce common numeric stat columns (only if they exist)
    numeric_cols = [c for c in ["PTS","MP","FG","FGA","3P","3PA","2P","2PA","FT","FTA","ORB","DRB","TRB","AST","STL","BLK","TOV","PF"] if c in df.columns]
    for c in numeric_cols:
        # MP is time-like; skip numeric coercion for MP
        if c == "MP":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows that clearly didn't play: rows where PTS and other key stats are all NA
    key_stats = [c for c in ["PTS","FG","FGA","AST","TRB"] if c in df.columns]
    if key_stats:
        mask_played = df[key_stats].notna().any(axis=1)
        df = df[mask_played]

    # coerce SEASON_END to int if present
    if "SEASON_END" in df.columns:
        df["SEASON_END"] = pd.to_numeric(df["SEASON_END"], errors="coerce").fillna(0).astype(int)
        # sort by season then date (oldest -> newest)
        df = df.sort_values(["SEASON_END","Date"], ascending=[True, True]).reset_index(drop=True)
    else:
        df = df.sort_values("Date").reset_index(drop=True)

    # save cleaned file (overwrite)
    df.to_csv(path, index=False)
    print(f"cleaned {file}: {len(df)} rows remaining")