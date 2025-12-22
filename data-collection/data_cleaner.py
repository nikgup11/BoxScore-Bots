import pandas as pd
import os

input_folder = "./data-collection/raw_data/"
output_folder = "./data-collection/cleaned_data/"

DROP_STRINGS = {"Did Not Play", "Did Not Dress", "Inactive", "Not With Team", "Did Not Play/Out"}

# converts 'MM:SS' strings to float minutes
def convert_mp(mp):
    # Handle missing values from pandas (NaN)
    if mp is None:
        return None
    if isinstance(mp, str):
        mp = mp.strip()
        if not mp:
            return None
        if ':' in mp:
            parts = mp.split(':')
            try:
                if len(parts) == 2:
                    # e.g. "19:26" -> 19 + 26/60
                    mins, secs = parts
                elif len(parts) == 3:
                    # e.g. "25:36:00" -> treat as 25:36, ignore third part
                    mins, secs, _ = parts
                else:
                    return None

                return int(mins) + int(secs) / 60.0
            except ValueError:
                return None
        else:
            # plain number as string
            try:
                return float(mp)
            except ValueError:
                return None
    elif isinstance(mp, (int, float)):
        return float(mp)
    return None

for file in os.listdir(input_folder):
    if not file.endswith(".csv"):
        continue
    input_path = os.path.join(input_folder, file)
    df = pd.read_csv(input_path)

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

    # make sure games are ordered oldest -> newest
    if 'Date' in df.columns:
        df = df.sort_values('Date')
    # coerce common numeric stat columns (only if they exist)
    numeric_cols = [c for c in ["PTS","MP","FG","FGA","3P","3PA","2P","2PA","FT","FTA","ORB","DRB","TRB","AST","STL","BLK","TOV","PF"] if c in df.columns]
    for c in numeric_cols:
        # MP is time-like; skip numeric coercion for MP
        if c == "MP":
            df["MP"] = df["MP"].apply(convert_mp)
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows that clearly didn't play: rows where PTS and other key stats are all NA
    key_stats = [c for c in ["MP","PTS","FG","FGA","AST","TRB"] if c in df.columns]
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

    # is_home: 1 if home, 0 if away
    df['is_home'] = (df['Unnamed: 5'] != '@').astype(float)
    # result_win: 1 if team won, 0 if lost
    df['result_win'] = df['Result'].str.startswith('W').astype(float)

    # save cleaned file (overwrite)
    output_path = os.path.join(output_folder, file)
    df.to_csv(output_path, index=False)
    print(f"cleaned {file}: {len(df)} rows remaining")