import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import warnings
from datetime import datetime
from nbainjuries import injury

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
SEQUENCE_LENGTH = 10
MODEL_PATH = "./ml_models/rnn_model.pth"
SCALER_PATH = "./ml_models/rnn_scaler.pkl"
PLAYER_STATS_PATH = "./data-collection/clean_data/PlayerStatistics.csv"
MATCHUPS_CSV = "./data-collection/clean_data/nba-2025-UTC.csv"
OUTPUT_PATH = "./data-collection/output_data/tonights_projections_rnn.csv"

feature_cols = [
    "points",
    "reboundsTotal",
    "assists",
    "numMinutes",
    "fieldGoalsAttempted",
    "OppOffRtg",
    "OppDefRtg",
    "OppPace",
    "home",
]

# ==========================================
# GET TARGET DATE (NEXT GAME DAY)
# ==========================================
data_temp = pd.read_csv(PLAYER_STATS_PATH, low_memory=False)
data_temp["gameDateTimeEst"] = pd.to_datetime(data_temp["gameDateTimeEst"], utc=True)

last_game_date = data_temp["gameDateTimeEst"].max()
TARGET_DATE = (
    last_game_date + pd.Timedelta(days=1)
).tz_localize(None).strftime("%Y-%m-%d")

print(f"\n🎯 Target Projection Date: {TARGET_DATE}")

# ==========================================
# LOAD INJURY REPORT (TODAY)
# ==========================================
print("Loading injury report...")

today = datetime.now()
today_midnight = datetime(today.year, today.month, today.day)

try:
    injury_df = injury.get_reportdata(today_midnight, return_df=True)
except Exception as e:
    print("Injury report failed:", e)
    injury_df = pd.DataFrame()

OUT_PLAYERS = set()
QUESTIONABLE_PLAYERS = set()

if not injury_df.empty:

    def format_name(name):
        if pd.isna(name):
            return None
        parts = name.split(",")
        if len(parts) == 2:
            return parts[1].strip() + " " + parts[0].strip()
        return name

    injury_df["Name"] = injury_df["Player Name"].apply(format_name)
    injury_df = injury_df[["Name", "Current Status"]].dropna()

    OUT_PLAYERS = set(
        injury_df[injury_df["Current Status"] == "Out"]["Name"]
    )

    QUESTIONABLE_PLAYERS = set(
        injury_df[
            injury_df["Current Status"].isin(["Questionable", "Doubtful"])
        ]["Name"]
    )

    print(f"Out Players: {len(OUT_PLAYERS)}")
    print(f"Questionable/Doubtful: {len(QUESTIONABLE_PLAYERS)}")

# ==========================================
# TEAM NAME NORMALIZATION
# ==========================================
TEAM_NAME_MAP = {
    "Cleveland": "Cavaliers",
    "Golden State": "Warriors",
    "LA": "Clippers",
    "Los Angeles": "Lakers",
    "New York": "Knicks",
    "Brooklyn": "Nets",
    "Houston": "Rockets",
    "Oklahoma City": "Thunder",
    "San Antonio": "Spurs",
    "Detroit": "Pistons",
    "Memphis": "Grizzlies",
    "Sacramento": "Kings",
    "Miami": "Heat",
    "Orlando": "Magic",
    "Toronto": "Raptors",
    "Boston": "Celtics",
    "Philadelphia": "76ers",
    "Washington": "Wizards",
    "Chicago": "Bulls",
    "Indiana": "Pacers",
    "Milwaukee": "Bucks",
    "Minnesota": "Timberwolves",
    "Denver": "Nuggets",
    "Utah": "Jazz",
    "Phoenix": "Suns",
    "Portland": "Trail Blazers",
    "Dallas": "Mavericks",
    "Charlotte": "Hornets",
    "Atlanta": "Hawks",
    "New Orleans": "Pelicans",
}

# ==========================================
# RNN MODEL
# ==========================================
class NBAPlayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=0.2
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        x = self.relu(self.fc1(last_out))
        return self.fc2(x)


print("Loading model...")
model = NBAPlayerRNN(len(feature_cols))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

print("Loading scaler...")
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ==========================================
# LOAD PLAYER DATA
# ==========================================
print("Loading player statistics...")
data = pd.read_csv(PLAYER_STATS_PATH, low_memory=False)
data["gameDateTimeEst"] = pd.to_datetime(data["gameDateTimeEst"], utc=True)
data = data[data["OppPace"].notna()].copy()
data = data.sort_values(["personId", "gameDateTimeEst"])

data["playerteamNameNorm"] = data["playerteamName"].map(
    TEAM_NAME_MAP
).fillna(data["playerteamName"])

# ==========================================
# LOAD SCHEDULE
# ==========================================
matchups = pd.read_csv(MATCHUPS_CSV, parse_dates=["Date"], dayfirst=True)
matchups["Date"] = matchups["Date"].dt.tz_localize("UTC")
matchups["DateET"] = matchups["Date"].dt.tz_convert("US/Eastern")
matchups["DateOnly"] = matchups["DateET"].dt.strftime("%Y-%m-%d")
matchups["GameTimeET"] = matchups["DateET"].dt.strftime("%H:%M")


def get_schedule(target_date):
    games = matchups[matchups["DateOnly"] == target_date]
    schedule = []

    for _, row in games.iterrows():
        away = row["Away Team"].split(" ")[-1]
        home = row["Home Team"].split(" ")[-1]
        schedule.append((away, home, row["GameTimeET"]))

    return schedule


# ==========================================
# PROJECTION ENGINE
# ==========================================
def predict_slate(target_date):

    schedule = get_schedule(target_date)

    if not schedule:
        print("No games found.")
        return pd.DataFrame()

    latest_date = data["gameDateTimeEst"].max()
    active_ids = data[
        data["gameDateTimeEst"] >= latest_date - pd.Timedelta(days=30)
    ]["personId"].unique()

    latest_team_map = (
        data.sort_values("gameDateTimeEst")
        .groupby("personId")["playerteamNameNorm"]
        .last()
    )

    results = []

    for away_team, home_team, game_time in schedule:
        for team, opp, is_home in [
            (away_team, home_team, 0),
            (home_team, away_team, 1),
        ]:

            team_norm = TEAM_NAME_MAP.get(team, team)

            roster_ids = [
                pid
                for pid in active_ids
                if latest_team_map.get(pid) == team_norm
            ]

            for pid in roster_ids:

                p_data = data[data["personId"] == pid].sort_values(
                    "gameDateTimeEst"
                )

                if len(p_data) < SEQUENCE_LENGTH:
                    continue

                if p_data["numMinutes"].tail(10).mean() < 5:
                    continue

                seq = p_data[feature_cols].tail(SEQUENCE_LENGTH)
                seq_scaled = scaler.transform(seq)
                X = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    pred = model(X).numpy()[0]

                pred_full = np.concatenate(
                    [pred, np.zeros(len(feature_cols) - 3)]
                )
                pred_denorm = scaler.inverse_transform(
                    pd.DataFrame([pred_full], columns=feature_cols)
                )[0, :3]

                pts, reb, ast = [max(0, x) for x in pred_denorm]

                name = (
                    p_data["firstName"].iloc[-1]
                    + " "
                    + p_data["lastName"].iloc[-1]
                )

                # Injury handling
                if name in OUT_PLAYERS:
                    continue

                #if name in QUESTIONABLE_PLAYERS:
                #    pts *= 0.75
                #    reb *= 0.75
                #    ast *= 0.75

                results.append(
                    {
                        "Name": name,
                        "Team": team,
                        "Opp": opp,
                        "Game_Date": target_date,
                        "Game_Time": game_time,
                        "Proj_PTS": round(pts, 2),
                        "Proj_REB": round(reb, 2),
                        "Proj_AST": round(ast, 2),
                        "Total_PRA": round(pts + reb + ast, 2),
                    }
                )

    df = pd.DataFrame(results)

    if df.empty:
        return df

    # Remove duplicates properly
    df = df.sort_values(
        ["Game_Time", "Total_PRA"], ascending=[True, False]
    )
    df = df.drop_duplicates(
        subset=["Name", "Team", "Game_Date"], keep="first"
    )

    return df


# ==========================================
# RUN
# ==========================================
print("\nGenerating projections...")
projections = predict_slate(TARGET_DATE)

if not projections.empty:
    print(f"\n--- PROJECTIONS FOR {TARGET_DATE} ---")
    projections = projections[(projections["Proj_PTS"] > 0) | (projections["Proj_REB"] > 0) | (projections["Proj_AST"] > 0)]
    print(projections.to_string())
    projections.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Saved to {OUTPUT_PATH}")
else:
    print("No projections generated.")