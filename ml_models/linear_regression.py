import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

# set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False

output_df = pd.read_csv('./data-collection/output_data/linear_player_prediction.csv')

def get_model_and_prediction(player_id):
    # 1. DATA IMPORT
    data = pd.read_csv(
        './data-collection/clean_data/PlayerStatistics.csv',
        dtype={"gameLabel": "string", "gameSubLabel": "string"},
        low_memory=False,
    )
    data = data[data['personId'] == player_id].copy()
    data = data.sort_values(by='gameDateTimeEst')
    data['gameDateTimeEst'] = (
        pd.to_datetime(
            data['gameDateTimeEst'],
            utc=True,
            format='mixed',
            errors='coerce',
        )
        .dt.tz_convert(None)
    )
    data = data.dropna(subset=['gameDateTimeEst'])

    if data.empty:
        print(f"No data for player {player_id}")
        return

    # 2. IDENTIFY NEXT GAME — based on *current* time, not last game played
    player_team = data['playerteamName'].iloc[-1]

    schedule = pd.read_csv('./data-collection/active_data/LeagueSchedule25_26.csv')
    schedule['gameDateTimeEst'] = (
        pd.to_datetime(
            schedule['gameDateTimeEst'],
            utc=True,
            format='mixed',
            errors='coerce',
        )
        .dt.tz_convert(None)
    )
    schedule = schedule.dropna(subset=['gameDateTimeEst'])

    # games involving this player’s team
    team_games = schedule[
        (schedule['homeTeamName'] == player_team) |
        (schedule['awayTeamName'] == player_team)
    ]

    # "now" in the same (UTC‑naive) scale as schedule times
    now_utc_naive = pd.Timestamp.utcnow()
    # ensure it's tz‑naive to match schedule['gameDateTimeEst']
    if now_utc_naive.tzinfo is not None:
        now_utc_naive = now_utc_naive.tz_convert(None)

    today = pd.Timestamp.now().normalize()

    # next game is the earliest team game at/after now
    future_games = team_games[
        team_games['gameDateTimeEst'].dt.normalize() >= today
    ].sort_values('gameDateTimeEst')

    if future_games.empty:
        print("No future games found for this player/team.")
        return

    next_game = future_games.iloc[0]
    is_home_next = 1 if next_game['homeTeamName'] == player_team else 0
    opp_team_next = next_game['awayTeamName'] if is_home_next else next_game['homeTeamName']

    # 3. FEATURE ENGINEERING
    prev_performance_features = [
        'numMinutes', 'fieldGoalsAttempted', 'threePointersAttempted', 'freeThrowsAttempted',
        'fieldGoalsPercentage', 'threePointersPercentage', 'freeThrowsPercentage',
        'reboundsTotal', 'assists', 'turnovers', 'plusMinusPoints', 'points'
    ]
    context_features = ['home', 'OppOffRtg', 'OppDefRtg', 'OppNetRtg', 'OppPace']
    target = 'points'

    df = data.copy()

    # --- NEW: 3 lags instead of 1 ---
    lagged_feature_cols = []
    for lag in [1, 2, 3]:
        for col in prev_performance_features:
            lag_col = f"{col}_lag{lag}"
            df[lag_col] = df[col].shift(lag)
            lagged_feature_cols.append(lag_col)

    all_features = lagged_feature_cols + context_features

    df = df.dropna(subset=all_features + [target])
    if df.empty:
        print("Not enough history to create 3‑game lagged features.")
        return

    # 4. PREPARE INPUT FOR NEXT GAME PREDICTION

    # Fill NaNs then take LAST 3 games for this player
    data[prev_performance_features] = data[prev_performance_features].fillna(0.0)
    if len(data) < 3:
        print("Player has fewer than 3 games of history.")
        return

    last3 = data.tail(3)  # oldest -> newest in order
    # map to lag1 (most recent), lag2, lag3
    last_game   = last3.iloc[-1]
    prev_game   = last3.iloc[-2]
    prev2_game  = last3.iloc[-3]

    # opponent context for NEXT game (unchanged)
    opp_stats_lookup = pd.read_csv(
        './data-collection/clean_data/PlayerStatistics.csv',
        dtype={"gameLabel": "string", "gameSubLabel": "string"},
        low_memory=False,
    )
    opp_stats_lookup['gameDateTimeEst'] = (
        pd.to_datetime(
            opp_stats_lookup['gameDateTimeEst'],
            utc=True,
            format='mixed',
            errors='coerce',
        )
        .dt.tz_convert(None)
    )
    opp_stats_lookup = opp_stats_lookup.dropna(subset=['gameDateTimeEst'])

    opp_rows = opp_stats_lookup[opp_stats_lookup['playerteamName'] == opp_team_next]
    if opp_rows.empty:
        print(f"No opponent stats found for team {opp_team_next}")
        return

    latest_opp_data = opp_rows.sort_values('gameDateTimeEst').iloc[-1]

    # Safely extract Opp* with fallback 0.0
    opp_off = float(latest_opp_data['OppOffRtg']) if pd.notna(latest_opp_data['OppOffRtg']) else 0.0
    opp_def = float(latest_opp_data['OppDefRtg']) if pd.notna(latest_opp_data['OppDefRtg']) else 0.0
    opp_net = float(latest_opp_data['OppNetRtg']) if pd.notna(latest_opp_data['OppNetRtg']) else 0.0
    opp_pace = float(latest_opp_data['OppPace']) if pd.notna(latest_opp_data['OppPace']) else 0.0

    next_game_context = np.array(
        [float(is_home_next), opp_off, opp_def, opp_net, opp_pace],
        dtype=np.float32,
    )

    # build feature vector in SAME order as all_features
    x_feats = []
    for lag, game_row in [(1, last_game), (2, prev_game), (3, prev2_game)]:
        for col in prev_performance_features:
            x_feats.append(float(game_row[col]))

    x_feats = np.array(x_feats, dtype=np.float32)
    x_future_np = np.concatenate([x_feats, next_game_context], axis=0).astype(np.float32).reshape(1, -1)

    if np.isnan(x_future_np).any():
        print("NaNs still present in x_future_np, aborting prediction.")
        return

    # 5. MODEL TRAINING
    X_np = df[all_features].values.astype(np.float32)
    y_np = df[[target]].values.astype(np.float32)

    # Check if we have enough samples for train/test split
    min_samples = 5
    if len(X_np) < min_samples:
        print(f"Player has only {len(X_np)} samples after feature engineering. Need at least {min_samples} for reliable training. Skipping.")
        return

    # chronological split (shuffle=False keeps time order)
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.2, shuffle=False
    )

    # ----- NEW: recency weights (more recent games get higher weight) -----
    n_train = X_train.shape[0]
    if n_train > 1:
        positions = np.arange(n_train)  # 0 = oldest, n_train-1 = newest
        min_weight = 0.3                # oldest game weight
        sample_weights = min_weight + (1.0 - min_weight) * positions / (n_train - 1)
    else:
        sample_weights = np.array([1.0], dtype=np.float32)
    sample_weights_t = torch.from_numpy(sample_weights.astype(np.float32))

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)

    model = nn.Linear(len(all_features), 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # use per‑example loss so we can weight by recency
    criterion = nn.MSELoss(reduction='none')

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)                 # (N, 1)
        loss_raw = criterion(preds, y_train_t)   # (N, 1)
        loss = (loss_raw.squeeze() * sample_weights_t).mean()
        loss.backward()
        optimizer.step()

    # 6. EVALUATE AND PREDICT
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test)
        y_test_t = torch.from_numpy(y_test)

        test_preds = model(X_test_t)
        mse = torch.mean((test_preds - y_test_t) ** 2).item()

        raw_pred = model(torch.from_numpy(x_future_np)).item()
        prediction = max(0.0, raw_pred)  # clamp to 0+

    print(f"--- Player ID: {player_id} ---")
    print(f"Test MSE: {mse:.3f}")
    print(f"Next Opponent: {opp_team_next} ({'Home' if is_home_next else 'Away'})")
    print(f"Projected Points: {prediction:.2f}")

    new_row = {
        'personId': int(player_id),
        'firstName': data['firstName'].iloc[-1],
        'lastName': data['lastName'].iloc[-1],
        'Date': next_game['gameDateTimeEst'],
        'predictedPoints': float(prediction),
        'MSE': float(mse),
        'playerteamName': player_team,
        'opponentteamName': opp_team_next,
    }

    global output_df

    if output_df.empty:
        output_df.loc[len(output_df)] = new_row
        print("created new row from empty table")
    else:
        mask = output_df['personId'] == int(player_id)
        if mask.any():
            output_df.loc[mask, ['firstName', 'lastName', 'Date', 'predictedPoints', 'MSE', 'playerteamName', 'opponentteamName']] = [
                new_row['firstName'],
                new_row['lastName'],
                new_row['Date'],
                new_row['predictedPoints'],
                new_row['MSE'],
                new_row['playerteamName'],
                new_row['opponentteamName'],
            ]
            print("existing row updated")
        else:
            output_df.loc[len(output_df)] = new_row
            print("created new row")
if __name__ == "__main__":
    # points -> 0
    # rebounds -> 1
    # assists -> 2
    players_file = pd.read_csv('./data-collection/active_data/Players.csv')
    for player_id in players_file['personId']:
        get_model_and_prediction(player_id)
    output_df.to_csv('./data-collection/output_data/linear_player_prediction.csv', index=False)