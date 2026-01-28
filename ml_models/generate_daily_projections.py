import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams as static_teams
import pickle
import warnings
warnings.filterwarnings('ignore')  # Suppress sklearn warnings

# ==========================================
# CONFIGURATION
# ==========================================
SEQUENCE_LENGTH = 10
MODEL_PATH = './ml_models/rnn_model.pth'
SCALER_PATH = './ml_models/rnn_scaler.pkl'

# Set TARGET_DATE to the day after the last date in PlayerStatistics data
data_temp = pd.read_csv('./data-collection/clean_data/PlayerStatistics.csv', low_memory=False)
data_temp['gameDateTimeEst'] = pd.to_datetime(data_temp['gameDateTimeEst'])
last_game_date = data_temp['gameDateTimeEst'].max()
TARGET_DATE = (last_game_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

feature_cols = ['points', 'reboundsTotal', 'assists', 'numMinutes', 
                'fieldGoalsAttempted', 'OppOffRtg', 'OppDefRtg', 'OppPace', 'home']

# ==========================================
# RNN MODEL DEFINITION
# ==========================================
class NBAPlayerRNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=3):
        super(NBAPlayerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        fc_out = self.fc1(last_out)
        fc_out = self.relu(fc_out)
        output = self.fc2(fc_out)
        return output

# ==========================================
# LOAD MODEL & SCALER
# ==========================================
print("Loading pre-trained model...")
model = NBAPlayerRNN(input_size=len(feature_cols), hidden_size=64, num_layers=2, output_size=3)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

print("Loading scaler...")
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# ==========================================
# LOAD DATA
# ==========================================
print("Loading data...")
data = pd.read_csv('./data-collection/clean_data/PlayerStatistics.csv', low_memory=False)
data['gameDateTimeEst'] = pd.to_datetime(data['gameDateTimeEst'], utc=True)
data = data[data['OppPace'].notna()].copy()
data = data.sort_values(['personId', 'gameDateTimeEst'])

# ==========================================
# DYNAMIC SCHEDULE ENGINE
# ==========================================
def get_live_schedule(date_str):
    print(f"Fetching live NBA schedule for {date_str}...")
    nba_teams = static_teams.get_teams()
    id_to_name = {t['id']: t['full_name'] for t in nba_teams}

    try:
        board = scoreboardv2.ScoreboardV2(game_date=date_str)
        games = board.game_header.get_data_frame()
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return []

    schedule = []
    for _, game in games.iterrows():
        home_id = game['HOME_TEAM_ID']
        away_id = game['VISITOR_TEAM_ID']
        game_time = game['GAME_STATUS_TEXT']
        
        home_name = id_to_name.get(home_id, "Unknown")
        away_name = id_to_name.get(away_id, "Unknown")
        
        home_simple = home_name.split(' ')[-1]
        away_simple = away_name.split(' ')[-1]
        
        schedule.append((away_simple, home_simple, game_time))

    print(f"Found {len(schedule)} games.")
    return schedule

# ==========================================
# TEAM STATS ENGINE
# ==========================================
def get_current_team_stats(df):
    stats_map = {}
    df_sorted = df.sort_values('gameDateTimeEst')
    unique_teams = df_sorted['opponentteamName'].unique()
    
    for team in unique_teams:
        recent = df_sorted[df_sorted['opponentteamName'] == team].tail(10)
        if not recent.empty:
            stats_map[team] = {
                'Pace': recent['OppPace'].mean(),
                'DefRtg': recent['OppDefRtg'].mean(),
                'OffRtg': recent['OppOffRtg'].mean()
            }
    return stats_map

DYNAMIC_STATS = get_current_team_stats(data)

# ==========================================
# PREDICT DAILY SLATE
# ==========================================
def predict_slate_rnn(team_stats_map, target_date_str):
    schedule = get_live_schedule(target_date_str)
    
    if not schedule:
        print("No games found for this date.")
        return pd.DataFrame()

    results = []
    
    latest_date = data['gameDateTimeEst'].max()
    active_ids = data[data['gameDateTimeEst'] >= (latest_date - pd.Timedelta(days=30))]['personId'].unique()
    
    print(f"Active players in last 30 days: {len(active_ids)}")

    print("Generating RNN projections...")
    
    for away_team, home_team, game_time in schedule:
        matchups = [(away_team, home_team, 0), (home_team, away_team, 1)]
        
        for current_team, opp_team, is_home in matchups:
            team_mask = data['playerteamName'].str.contains(current_team, case=False, na=False)
            roster_ids = data[team_mask & data['personId'].isin(active_ids)]['personId'].unique()
            
            for pid in roster_ids:
                p_recent_all = data[data['personId'] == pid].sort_values('gameDateTimeEst')
                
                if len(p_recent_all) < SEQUENCE_LENGTH:
                    continue
                
                # Get last SEQUENCE_LENGTH games
                p_sequence = p_recent_all[feature_cols].tail(SEQUENCE_LENGTH).values
                
                # Skip players with very low recent minutes (likely DNP)
                avg_minutes = p_recent_all['numMinutes'].tail(10).mean()
                if avg_minutes < 5:
                    continue
                
                # Normalize using scaler - convert to DataFrame to preserve column names
                p_sequence_df = pd.DataFrame(p_sequence, columns=feature_cols)
                p_sequence_normalized = scaler.transform(p_sequence_df)
                
                # Create tensor
                X_pred = torch.tensor(p_sequence_normalized, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    pred = model(X_pred).numpy()[0]
                
                # Denormalize predictions - create DataFrame with correct column names
                pred_full = np.concatenate([pred, np.zeros(len(feature_cols) - 3)])
                pred_df = pd.DataFrame([pred_full], columns=feature_cols)
                pred_denorm = scaler.inverse_transform(pred_df)[0, :3]
                
                p_pts, p_reb, p_ast = [max(0, val) for val in pred_denorm]
                
                p_name = f"{p_recent_all['firstName'].iloc[-1]} {p_recent_all['lastName'].iloc[-1]}"
                
                # Debug: print very low projections
                if p_pts + p_reb + p_ast < 1:
                    print(f"DEBUG: {p_name} ({current_team}) predicted very low: PTS={p_pts:.2f}, REB={p_reb:.2f}, AST={p_ast:.2f}")
                
                results.append({
                    'Name': p_name,
                    'Team': current_team,
                    'Opp': opp_team,
                    'Game_Date': target_date_str,
                    'Game_Time': game_time,
                    'Proj_PTS': round(p_pts, 2),
                    'Proj_REB': round(p_reb, 2),
                    'Proj_AST': round(p_ast, 2),
                    'Total_PRA': round(p_pts + p_reb + p_ast, 2)
                })

    return pd.DataFrame(results).sort_values(by=['Game_Time', 'Total_PRA'], ascending=[True, False])

# ==========================================
# EXECUTE
# ==========================================
projections = predict_slate_rnn(DYNAMIC_STATS, TARGET_DATE)

if not projections.empty:
    # Remove duplicate rows (same player + team + game)
    projections = projections.drop_duplicates(subset=['Name', 'Team', 'Game_Date'], keep='first')
    
    print(f"\n--- RNN PROJECTIONS FOR {TARGET_DATE} ---")
    print(projections.to_string())
    projections.to_csv('./data-collection/output_data/tonights_projections_rnn.csv', index=False)
    print(f"\n✅ Saved to './data-collection/output_data/tonights_projections_rnn.csv'")
else:
    print("No projections generated.")
