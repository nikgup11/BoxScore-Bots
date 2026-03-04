import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
SEQUENCE_LENGTH = 10
MODEL_DIR = './ml_models/'
SCALER_PATH = MODEL_DIR + 'xgb_scaler.pkl'
MATCHUPS_CSV = './data-collection/clean_data/nba-2025-UTC.csv'

# Target = day after most recent game
data_temp = pd.read_csv('./data-collection/clean_data/PlayerStatistics.csv', low_memory=False)
data_temp['gameDateTimeEst'] = pd.to_datetime(data_temp['gameDateTimeEst'], utc=True)
last_game_date = data_temp['gameDateTimeEst'].max()
TARGET_DATE = (last_game_date + pd.Timedelta(days=1)).tz_localize(None).strftime('%Y-%m-%d')

feature_cols = [
    'points', 'reboundsTotal', 'assists',
    'numMinutes', 'fieldGoalsAttempted',
    'OppOffRtg', 'OppDefRtg', 'OppPace', 'home'
]

target_cols = ['points', 'reboundsTotal', 'assists']

# ==========================================
# TEAM NAME NORMALIZATION
# ==========================================
TEAM_NAME_MAP = {
    'Cleveland': 'Cavaliers',
    'Golden State': 'Warriors',
    'LA': 'Clippers',
    'Los Angeles': 'Lakers',
    'New York': 'Knicks',
    'Brooklyn': 'Nets',
    'Houston': 'Rockets',
    'Oklahoma City': 'Thunder',
    'San Antonio': 'Spurs',
    'Detroit': 'Pistons',
    'Memphis': 'Grizzlies',
    'Sacramento': 'Kings',
    'Miami': 'Heat',
    'Orlando': 'Magic',
    'Toronto': 'Raptors',
    'Boston': 'Celtics',
    'Philadelphia': '76ers',
    'Washington': 'Wizards',
    'Chicago': 'Bulls',
    'Indiana': 'Pacers',
    'Milwaukee': 'Bucks',
    'Minnesota': 'Timberwolves',
    'Denver': 'Nuggets',
    'Utah': 'Jazz',
    'Phoenix': 'Suns',
    'Portland': 'Trail Blazers',
    'Dallas': 'Mavericks',
    'Charlotte': 'Hornets',
    'Atlanta': 'Hawks',
    'New Orleans': 'Pelicans'
}

# ==========================================
# LOAD MODELS
# ==========================================
print("Loading XGBoost models...")

models = {}
for stat in target_cols:
    model = xgb.Booster()
    model.load_model(MODEL_DIR + f"xgb_{stat}.json")
    models[stat] = model

print("Loading scaler...")
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# ==========================================
# LOAD DATA
# ==========================================
print("Loading player data...")
data = pd.read_csv('./data-collection/clean_data/PlayerStatistics.csv', low_memory=False)
data['gameDateTimeEst'] = pd.to_datetime(data['gameDateTimeEst'], utc=True)
data = data[data['OppPace'].notna()].copy()
data = data.sort_values(['personId', 'gameDateTimeEst'])

data['playerteamNameNorm'] = data['playerteamName'].map(TEAM_NAME_MAP).fillna(data['playerteamName'])

# ==========================================
# LOAD SCHEDULE
# ==========================================
print("Loading schedule...")
matchups = pd.read_csv(MATCHUPS_CSV, parse_dates=['Date'], dayfirst=True)

matchups['Date'] = matchups['Date'].dt.tz_localize('UTC')
matchups['DateET'] = matchups['Date'].dt.tz_convert('US/Eastern')
matchups['DateOnly'] = matchups['DateET'].dt.strftime('%Y-%m-%d')
matchups['GameTimeET'] = matchups['DateET'].dt.strftime('%H:%M')

def get_schedule_from_csv(target_date_str):
    today = matchups[matchups['DateOnly'] == target_date_str]
    schedule = []
    for _, row in today.iterrows():
        away = row['Away Team'].split(' ')[-1]
        home = row['Home Team'].split(' ')[-1]
        game_time = row['GameTimeET']
        schedule.append((away, home, game_time))
    return schedule

# ==========================================
# PREDICTION ENGINE
# ==========================================
def predict_slate_xgb(target_date_str):

    schedule = get_schedule_from_csv(target_date_str)

    if not schedule:
        print("No games found.")
        return pd.DataFrame()

    results = []

    latest_date = data['gameDateTimeEst'].max()
    active_ids = data[
        data['gameDateTimeEst'] >= (latest_date - pd.Timedelta(days=30))
    ]['personId'].unique()

    latest_team_per_player = (
        data.sort_values('gameDateTimeEst')
            .groupby('personId')['playerteamNameNorm']
            .last()
    )

    print("Generating XGBoost projections...")

    for away_team, home_team, game_time in schedule:
        matchups = [(away_team, home_team, 0), (home_team, away_team, 1)]

        for current_team, opp_team, is_home in matchups:

            current_team_norm = TEAM_NAME_MAP.get(current_team, current_team)

            roster_ids = [
                pid for pid in active_ids
                if latest_team_per_player.get(pid, None) == current_team_norm
            ]

            for pid in roster_ids:

                p_recent = data[data['personId'] == pid].sort_values('gameDateTimeEst')

                if len(p_recent) < SEQUENCE_LENGTH:
                    continue

                seq = p_recent[feature_cols].tail(SEQUENCE_LENGTH).values

                # Skip low-minute players
                if p_recent['numMinutes'].tail(10).mean() < 5:
                    continue

                seq_df = pd.DataFrame(seq, columns=feature_cols)
                seq_scaled = scaler.transform(seq_df)

                # Flatten sequence for XGB
                seq_flat = seq_scaled.reshape(1, -1)

                dmatrix = xgb.DMatrix(seq_flat)

                preds_scaled = []
                for stat in target_cols:
                    pred = models[stat].predict(dmatrix)[0]
                    preds_scaled.append(pred)

                # Denormalize correctly
                pred_full = np.zeros(len(feature_cols))
                pred_full[:3] = preds_scaled
                pred_df = pd.DataFrame([pred_full], columns=feature_cols)
                pred_denorm = scaler.inverse_transform(pred_df)[0][:3]

                pts, reb, ast = [max(0, v) for v in pred_denorm]

                p_name = f"{p_recent['firstName'].iloc[-1]} {p_recent['lastName'].iloc[-1]}"

                results.append({
                    'Name': p_name,
                    'Team': current_team,
                    'Opp': opp_team,
                    'Game_Date': target_date_str,
                    'Game_Time': game_time,
                    'Proj_PTS': round(pts, 2),
                    'Proj_REB': round(reb, 2),
                    'Proj_AST': round(ast, 2),
                    'Total_PRA': round(pts + reb + ast, 2)
                })

    return pd.DataFrame(results).sort_values(
        by=['Game_Time', 'Total_PRA'],
        ascending=[True, False]
    )

# ==========================================
# EXECUTE
# ==========================================
projections = predict_slate_xgb(TARGET_DATE)

if not projections.empty:
    projections = projections.drop_duplicates(
        subset=['Name', 'Team', 'Game_Date'],
        keep='first'
    )

    print(f"\n--- XGBOOST PROJECTIONS FOR {TARGET_DATE} ---")
    print(projections.to_string())

    projections.to_csv(
        './data-collection/output_data/tonights_projections_xgb.csv',
        index=False
    )

    print("\n✅ Saved to './data-collection/output_data/tonights_projections_xgb.csv'")
else:
    print("No projections generated.")