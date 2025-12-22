import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

# %%
# converts 'MM:SS' strings to float minutes
def convert_mp(mp):
    if isinstance(mp, str):
        if ':' in mp:
            try:
                mins, secs = mp.split(':')
                return int(mins) + int(secs) / 60
            except:
                return None
        else:
            try:
                return float(mp)
            except:
                return None
    elif isinstance(mp, (int, float)):
        return float(mp)
    else:
        return None

# %%
# import data
data = pd.read_csv('./data-collection/bbref_players_games_simple/gilgesh01_Shai_Gilgeous-Alexander_last3_with_opp_stats.csv')

# make sure games are ordered oldest -> newest
if 'Date' in data.columns:
    data = data.sort_values('Date')

# is_home: 1 if home, 0 if away
data['is_home'] = (data['Unnamed: 5'] != '@').astype(float)
# result_win: 1 if team won, 0 if lost
data['result_win'] = data['Result'].str.startswith('W').astype(float)

# features (previous game)
features = [
    'MP', 'FGA', '3PA', 'FTA',
    'FG%', '3P%', '2P%', 'eFG%', 'FT%',
    'TRB', 'AST', 'TOV',
    'GmSc', '+/-', 'PTS', 'is_home', 'result_win',
    'OppOffRtg', 'OppDefRtg', 'OppNetRtg', 'OppPace'
]

# convert MP to float minutes
data['MP'] = data['MP'].apply(convert_mp)

# convert numeric columns
for col in features + ['PTS']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# create TARGET from *current* game points
target = ['PTS']

# Capture the stats from the very last played game.
X_today = data[features].iloc[-1].values.reshape(1, -1)

# now shift features so they refer to the *previous* game
data[features] = data[features].shift(1)

# drop first row and rows with NaNs
data = data.dropna(subset=features + target)

# numpy arrays
X_np = data[features].values
y_np = data[target].values

USE_ALL_DATA_FOR_TODAY_CHECK = True  # False = evaluation mode

# %%
if USE_ALL_DATA_FOR_TODAY_CHECK:

    # train on ALL history
    X_train_np, y_train_np = X_np, y_np
    
    # Init and train rf model
    rfr = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=1, random_state=42)

    rfr.fit(X_train_np, y_train_np.ravel())

    y_pred_pts = rfr.predict(X_today)

    print(f"--- Prediction for Next Game ---")
    print(f"Predicted Points: {y_pred_pts[0]:.1f}")

    # Enter the actual score manually
    actual_score_today = 32  # <--- Update this with the real score

    # Calculate the difference
    diff = y_pred_pts[0] - actual_score_today

    print(f"Predicted: {y_pred_pts[0]:.1f}")
    print(f"Actual:    {actual_score_today}")
    print(f"Error:     {abs(diff):.1f} points")
else:
    # chronological train/test split: last 20% of games as test
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np, y_np, test_size=0.2, shuffle=False
    )

    # Init and train rf model
    rfr = RandomForestRegressor(n_estimators=200, random_state=42)

    rfr.fit(X_train_np, y_train_np.ravel())

    y_pred_pts = rfr.predict(X_test_np)

    print("--- Prediction vs Actual (Points) ---")
    for i in range(len(y_test_np)):
        pred = y_pred_pts[i]
        actual = y_test_np[i].item()
        diff = pred - actual
        print(f"Game {i+1}: Pred: {pred:.1f} | Actual: {actual:.1f} | Diff: {diff:.1f}")

    # Calculate Error Metrics
    mae = mean_absolute_error(y_test_np, y_pred_pts)
    rmse = root_mean_squared_error(y_test_np, y_pred_pts)

    # Interpretation
    print(f"On average, the model's prediction is off by {mae:.1f} points.")


# %%



