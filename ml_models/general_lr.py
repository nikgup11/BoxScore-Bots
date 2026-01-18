import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams as static_teams

# ==========================================
# CONFIGURATION
# ==========================================
# Set TARGET_DATE to the day after the last date in PlayerStatistics data
data_temp = pd.read_csv('./data-collection/clean_data/PlayerStatistics.csv', low_memory=False)
data_temp['gameDateTimeEst'] = pd.to_datetime(data_temp['gameDateTimeEst'])
last_game_date = data_temp['gameDateTimeEst'].max()
TARGET_DATE = (last_game_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d') 

# ==========================================
# 1. DYNAMIC SCHEDULE ENGINE
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
        
        # Simplify names (e.g., "Los Angeles Lakers" -> "Lakers")
        home_simple = home_name.split(' ')[-1]
        away_simple = away_name.split(' ')[-1]
        
        schedule.append((away_simple, home_simple, game_time))

    print(f"Found {len(schedule)} games.")
    return schedule

# ==========================================
# 2. DATA LOADING
# ==========================================
print("Loading data...")
data = pd.read_csv('./data-collection/clean_data/PlayerStatistics.csv', low_memory=False)

data['gameDateTimeEst'] = pd.to_datetime(data['gameDateTimeEst'], utc=True)
data = data[data['OppPace'].notna()].copy()
data = data.sort_values(['personId', 'gameDateTimeEst'])

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
# Position Logic
player_profiles = data.groupby('personId')[['assists', 'reboundsTotal']].mean()
def classify_pos(row):
    if pd.isna(row['reboundsTotal']): return 'F'
    if row['reboundsTotal'] > 6.5: return 'C'
    if row['assists'] > 4.5: return 'G'
    return 'F'
player_profiles['pos'] = player_profiles.apply(classify_pos, axis=1)
data['position'] = data['personId'].map(player_profiles['pos'].to_dict())

# One-Hot Encoding
data = pd.get_dummies(data, columns=['position'], prefix='role')

# --- DEFINING ROLE COLS EXPLICITLY ---
role_cols = ['role_C', 'role_F', 'role_G'] 
for r in role_cols:
    if r not in data.columns: data[r] = 0

# Rolling Stats
rolling_stats = ['points', 'reboundsTotal', 'assists', 'numMinutes', 'fieldGoalsAttempted']
for stat in rolling_stats:
    data[f'{stat}_roll_5'] = data.groupby('personId')[stat].transform(
        lambda x: x.shift(1).rolling(window=5).mean()
    )
data = data.dropna(subset=[f'{s}_roll_5' for s in rolling_stats])

# ==========================================
# 4. TEAM STATS ENGINE
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
# 5. MODEL TRAINING
# ==========================================
matchup_features = ['home', 'OppOffRtg', 'OppDefRtg', 'OppNetRtg', 'OppPace']
feature_columns = [f'{s}_roll_5' for s in rolling_stats] + role_cols + matchup_features
targets = ['points', 'reboundsTotal', 'assists']

X = data[feature_columns].values.astype(np.float32)
y = data[targets].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = nn.Sequential(
    nn.Linear(len(feature_columns), 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, len(targets)) 
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
criterion = nn.MSELoss()

print("Training Model...")
for epoch in range(800):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(torch.tensor(X_train)), torch.tensor(y_train))
    loss.backward()
    optimizer.step()

# ==========================================
# 6. PREDICT DYNAMIC SLATE
# ==========================================
def predict_slate(team_stats_map, target_date_str):
    schedule = get_live_schedule(target_date_str)
    
    if not schedule:
        print("No games found for this date.")
        return pd.DataFrame()

    model.eval()
    results = []
    
    latest_date = data['gameDateTimeEst'].max()
    active_ids = data[data['gameDateTimeEst'] >= (latest_date - pd.Timedelta(days=30))]['personId'].unique()

    print("Generating projections...")
    
    for away_team, home_team, game_time in schedule:
        matchups = [(away_team, home_team, 0), (home_team, away_team, 1)]
        
        for current_team, opp_team, is_home in matchups:
            team_mask = data['playerteamName'].str.contains(current_team, case=False, na=False)
            roster_ids = data[team_mask & data['personId'].isin(active_ids)]['personId'].unique()
            
            opp_stats = team_stats_map.get(opp_team, {'Pace': 100, 'DefRtg': 112, 'OffRtg': 112})
            
            for pid in roster_ids:
                p_recent = data[data['personId'] == pid].tail(1)
                if p_recent.empty: continue
                
                feat_roll = p_recent[[f'{s}_roll_5' for s in rolling_stats]].values.flatten()
                feat_role = p_recent[role_cols].values.flatten()
                feat_match = np.array([
                    is_home,
                    opp_stats['OffRtg'],
                    opp_stats['DefRtg'],
                    opp_stats['OffRtg'] - opp_stats['DefRtg'],
                    opp_stats['Pace']
                ])
                
                X_in = torch.tensor(np.concatenate([feat_roll, feat_role, feat_match]).astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    preds = model(X_in).numpy()[0]
                
                p_pts, p_reb, p_ast = [max(0, val) for val in preds]
                
                results.append({
                    'Name': f"{p_recent['firstName'].values[0]} {p_recent['lastName'].values[0]}",
                    'Team': current_team,
                    'Opp': opp_team,
                    'Game_Date': target_date_str,  # <--- ADDED DATE HERE
                    'Game_Time': game_time,
                    'Proj_PTS': round(p_pts, 2),
                    'Proj_REB': round(p_reb, 2),
                    'Proj_AST': round(p_ast, 2),
                    'Total_PRA': round(p_pts + p_reb + p_ast, 2)
                })

    return pd.DataFrame(results).sort_values(by=['Game_Time', 'Total_PRA'], ascending=[True, False])

# EXECUTE
projections = predict_slate(DYNAMIC_STATS, TARGET_DATE)

if not projections.empty:
    print(f"\n--- PROJECTIONS FOR {TARGET_DATE} ---")
    print(projections[['Game_Date', 'Game_Time', 'Name', 'Team', 'Total_PRA']].head(15))
    
    projections.to_csv('./data-collection/output_data/tonights_projections.csv', index=False)
    print("Saved to tonights_projections.csv")