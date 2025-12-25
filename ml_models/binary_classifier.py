import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load environment variables from .env file
load_dotenv()

# set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

output_df = pd.read_csv('./data-collection/output_data/binary_player_prediction.csv') if pd.io.common.file_exists('./data-collection/output_data/binary_player_prediction.csv') else pd.DataFrame()

def fetch_player_props_from_odds_api(api_key):
    """
    Fetch NBA player props (points) from The Odds API.
    Returns a dictionary mapping player_name -> points_line
    """
    player_props = {}
    
    try:
        # First, get all NBA events (games)
        events_url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
        events_params = {'apiKey': api_key}
        
        print("Fetching NBA events...")
        events_response = requests.get(events_url, params=events_params)
        events_response.raise_for_status()
        events = events_response.json()
        
        print(f"Found {len(events)} upcoming NBA games")
        
        # For each event, get player props
        for idx, event in enumerate(events, 1):
            event_id = event['id']
            home_team = event.get('home_team', '')
            away_team = event.get('away_team', '')
            
            print(f"\n[{idx}/{len(events)}] Fetching props for {home_team} vs {away_team}...")
            
            # Get odds for this specific event with player markets
            odds_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
            odds_params = {
                'apiKey': api_key,
                'regions': 'us',
                'markets': 'player_points',  # Correct market key from API docs
                'oddsFormat': 'american',
            }
            
            try:
                odds_response = requests.get(odds_url, params=odds_params)
                odds_response.raise_for_status()
                odds_data = odds_response.json()
                
                # Parse bookmaker data
                if 'bookmakers' in odds_data:
                    for bookmaker in odds_data['bookmakers']:
                        if 'markets' in bookmaker:
                            for market in bookmaker['markets']:
                                # Check for player points markets
                                if 'player_points' in market['key']:
                                    for outcome in market['outcomes']:
                                        player_name = outcome.get('description', outcome.get('name', ''))
                                        points_line = outcome.get('point')
                                        
                                        if player_name and points_line is not None:
                                            player_name_lower = player_name.lower()
                                            
                                            # Store only if we don't have it yet (first bookmaker wins)
                                            if player_name_lower not in player_props:
                                                player_props[player_name_lower] = points_line
                                                print(f"  ✓ {player_name}: {points_line} points")
                
            except requests.exceptions.RequestException as e:
                print(f"  ✗ Error fetching odds for this game: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Total unique players with point lines: {len(player_props)}")
        print(f"{'='*60}")
        
        # Show sample
        if player_props:
            print("\nSample of fetched odds:")
            for i, (name, line) in enumerate(list(player_props.items())[:10]):
                print(f"  {name}: {line}")
        
        return player_props
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching events from Odds API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return {}
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}

def get_model_and_prediction(player_id, over_under):
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
    
    # 5. CREATE BINARY LABELS: 1 if points > over_under, 0 otherwise
    df['over_under_binary'] = (df[target] > over_under).astype(int)
    
    X = df[all_features].values.astype(np.float32)
    y = df['over_under_binary'].values.astype(np.int64)
    
    if len(X) < 10:
        print(f"Not enough data points ({len(X)}) for training. Need at least 10 games.")
        return
    
    # Check class balance
    over_count = np.sum(y == 1)
    under_count = np.sum(y == 0)
    print(f"\nClass distribution: UNDER={under_count}, OVER={over_count}")
    print(f"Over/Under Line: {over_under} points")
    
    # 6. NORMALIZE FEATURES (critical for neural networks)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 7. TRAIN/TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    # Normalize the future game input
    x_future_scaled = scaler.transform(x_future_np)
    
    # 8. DEFINE BINARY CLASSIFIER MODEL (simplified for small datasets)
    class BinaryClassifier(nn.Module):
        def __init__(self, input_size):
            super(BinaryClassifier, self).__init__()
            self.fc1 = nn.Linear(input_size, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 2)  # 2 classes: over (1) or under (0)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)  # Reduced dropout
            self.batch_norm1 = nn.BatchNorm1d(32)
            self.batch_norm2 = nn.BatchNorm1d(16)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.batch_norm1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.batch_norm2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x
    
    # 9. TRAIN THE MODEL
    input_size = X_train.shape[1]
    model = BinaryClassifier(input_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Add L2 regularization
    
    num_epochs = 200  # Reduced epochs
    batch_size = min(16, len(X_train) // 2)  # Smaller batch size for small datasets
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    best_model_state = None
    
    model.train()
    for epoch in range(num_epochs):
        # Mini-batch training
        permutation = torch.randperm(X_train_t.size()[0])
        epoch_loss = 0.0
        
        for i in range(0, X_train_t.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_t[indices], y_train_t[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation on test set for early stopping
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_t)
            val_loss = criterion(val_outputs, y_test_t).item()
        model.train()
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss/len(X_train):.4f}, Val Loss: {val_loss:.4f}")
        
        # Stop if no improvement
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 10. EVALUATE ON TEST SET
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        probabilities_test = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test_t).float().mean()
        print(f"\nTest Accuracy: {accuracy:.2%}")
        
        # Show confidence distribution on test set
        test_confidences = probabilities_test.max(dim=1)[0]
        print(f"Test set confidence range: [{test_confidences.min():.2%}, {test_confidences.max():.2%}]")
        print(f"Test set mean confidence: {test_confidences.mean():.2%}")
        
        # Class-wise accuracy
        for cls in [0, 1]:
            cls_mask = y_test_t == cls
            if cls_mask.sum() > 0:
                cls_acc = (predicted[cls_mask] == y_test_t[cls_mask]).float().mean()
                label = "UNDER" if cls == 0 else "OVER"
                print(f"{label} Class Accuracy: {cls_acc:.2%} ({cls_mask.sum()} samples)")
    
    # 11. PREDICT NEXT GAME
    x_future_tensor = torch.tensor(x_future_scaled, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        output = model(x_future_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted_class = torch.max(output, 1)
        
        under_prob = probabilities[0][0].item()
        over_prob = probabilities[0][1].item()
        
        prediction = "OVER" if predicted_class.item() == 1 else "UNDER"
        
        print(f"\n{'='*60}")
        print(f"PREDICTION FOR PLAYER {player_id}")
        print(f"Over/Under Line: {over_under} points")
        print(f"{'='*60}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {max(under_prob, over_prob):.2%}")
        print(f"  - UNDER probability: {under_prob:.2%}")
        print(f"  - OVER probability: {over_prob:.2%}")
        print(f"{'='*60}")
        print(f"\nNext game: {player_team} vs {opp_team_next}")
        print(f"Home game: {'Yes' if is_home_next else 'No'}")
        print(f"Game date: {next_game['gameDateTimeEst']}")
        
        # Show recent performance
        print(f"\nRecent Performance (Last 3 Games):")
        for i, (idx, row) in enumerate(last3.iterrows(), 1):
            print(f"  Game {i}: {row['points']:.1f} pts vs {row['opponentteamName']}")
    
    # Save results to global dataframe
    new_row = {
        'personId': int(player_id),
        'firstName': data['firstName'].iloc[-1],
        'lastName': data['lastName'].iloc[-1],
        'Date': next_game['gameDateTimeEst'],
        'prediction': prediction,
        'over_prob': float(over_prob),
        'under_prob': float(under_prob),
        'over_under_line': float(over_under),
        'confidence': float(max(under_prob, over_prob)),
        'test_accuracy': float(accuracy.item()),
        'playerteamName': player_team,
        'opponentteamName': opp_team_next,
    }
    
    global output_df
    
    if output_df.empty:
        output_df = pd.DataFrame([new_row])
        print("Created new row from empty table")
    else:
        mask = (output_df['personId'] == int(player_id)) & (output_df['over_under_line'] == float(over_under))
        if mask.any():
            for col in new_row.keys():
                output_df.loc[mask, col] = new_row[col]
            print("Existing row updated")
        else:
            output_df = pd.concat([output_df, pd.DataFrame([new_row])], ignore_index=True)
            print("Created new row")
        
    return model, prediction, over_prob, under_prob


if __name__ == "__main__":
    # Get API key from .env file
    api_key = os.getenv('ODDS_API_KEY')
    
    if not api_key:
        print("ERROR: ODDS_API_KEY not found in .env file.")
        print("Please create a .env file in the project root with: ODDS_API_KEY=your_api_key_here")
        print("Get your API key from: https://the-odds-api.com/")
        print("\nFalling back to linear regression predictions...")
        
        # Fallback to linear regression
        linear_predictions = pd.read_csv('./data-collection/output_data/linear_player_prediction.csv')
        player_lines = {}
        for _, row in linear_predictions.iterrows():
            player_lines[row['personId']] = row['predictedPoints']
    else:
        print("Fetching player props from The Odds API...")
        odds_props = fetch_player_props_from_odds_api(api_key)
        
        # Create a mapping from personId to points line
        player_lines = {}
        players_file = pd.read_csv('./data-collection/active_data/Players.csv')
        
        for _, player_row in players_file.iterrows():
            player_id = player_row['personId']
            first_name = player_row.get('firstName', '')
            last_name = player_row.get('lastName', '')
            
            if pd.notna(first_name) and pd.notna(last_name):
                full_name = f"{first_name} {last_name}".lower()
                
                # Try to find the player in odds data
                if full_name in odds_props:
                    player_lines[player_id] = odds_props[full_name]
                    print(f"Found odds for {first_name} {last_name}: {odds_props[full_name]} points")
        
        # If we didn't get enough data from API, supplement with linear regression
        if len(player_lines) < 10:
            print("\nNot enough data from Odds API, supplementing with linear regression predictions...")
            linear_predictions = pd.read_csv('./data-collection/output_data/linear_player_prediction.csv')
            for _, row in linear_predictions.iterrows():
                if row['personId'] not in player_lines:
                    player_lines[row['personId']] = row['predictedPoints']
    
    print(f"\nTotal players with over/under lines: {len(player_lines)}")
    
    players_file = pd.read_csv('./data-collection/active_data/Players.csv')
    
    for idx, player_row in players_file.iterrows():
        player_id = player_row['personId']
        
        if player_id in player_lines:
            over_under_line = player_lines[player_id]
            print(f"\n{'='*80}")
            print(f"Processing Player {idx+1}/{len(players_file)}: {player_id}")
            print(f"Over/Under Line: {over_under_line}")
            print(f"{'='*80}")
            
            try:
                get_model_and_prediction(player_id, over_under_line)
            except Exception as e:
                print(f"Error processing player {player_id}: {e}")
                continue
        else:
            print(f"Skipping player {player_id}: No over/under line available")
    
    # Save results to CSV
    output_df.to_csv('./data-collection/output_data/binary_player_prediction.csv', index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to './data-collection/output_data/binary_player_prediction.csv'")
    print(f"Total players processed: {len(output_df)}")
    print(f"{'='*80}")