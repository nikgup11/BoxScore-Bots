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

# if you ever use GPU, these help determinism
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

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

# data import
data = pd.read_csv('./data-collection/bbref_players_games_simple/gilgesh01_Shai_Gilgeous-Alexander_last3_with_opp_stats.csv')
# make sure games are ordered oldest -> newest
if 'Date' in data.columns:
    data = data.sort_values('Date')

# is_home: 1 if home, 0 if away
data['is_home'] = (data['Unnamed: 5'] != '@').astype(float)
# result_win: 1 if team won, 0 if lost
data['result_win'] = data['Result'].str.startswith('W').astype(float)

# features (previous game)
prev_player_features = [
    'MP', 'FGA', '3PA', 'FTA',
    'FG%', '3P%', '2P%', 'eFG%', 'FT%',
    'TRB', 'AST', 'TOV',
    'GmSc', '+/-', 'PTS', 'is_home', 'result_win',
]

prev_opp_features = ['PrevOppOffRtg', 'PrevOppDefRtg', 'PrevOppNetRtg', 'PrevOppPace']
current_features = ['OppOffRtg', 'OppDefRtg', 'OppNetRtg', 'OppPace']

features = prev_player_features + prev_opp_features + current_features

target = ['PTS']

# convert MP to float minutes
data['MP'] = data['MP'].apply(convert_mp)

# convert numeric columns
for col in prev_player_features + current_features + ['PTS']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# build previous-opponent columns from shifted current opponent
data['PrevOppOffRtg'] = data['OppOffRtg'].shift(1)
data['PrevOppDefRtg'] = data['OppDefRtg'].shift(1)
data['PrevOppNetRtg'] = data['OppNetRtg'].shift(1)
data['PrevOppPace']   = data['OppPace'].shift(1)

# shift previous-player stats to last game
data[prev_player_features] = data[prev_player_features].shift(1)

# drop rows with NaNs in features or target
data = data.dropna(subset=features + target)

# build numpy arrays
X_np = data[features].values.astype(np.float32)
y_np = data[target].values.astype(np.float32)

USE_ALL_DATA_FOR_TODAY_CHECK = True  # False = evaluation mode

if USE_ALL_DATA_FOR_TODAY_CHECK:
    # train on ALL history
    X_train_np, y_train_np = X_np, y_np

    # convert to tensors
    X_train = torch.from_numpy(X_train_np)
    y_train = torch.from_numpy(y_train_np)

    # define linear regression model
    model = nn.Linear(X_train.shape[1], 1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 1000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()

    # use the most recent row's features as today's input
    x_today_np = X_np[-1:]  # shape (1, n_features)
    x_today = torch.from_numpy(x_today_np)

    model.eval()
    with torch.no_grad():
        pred_today = model(x_today).item()
    print(f"Today's prediction (using all history): {pred_today:.2f}")
else:
    # chronological train/test split: last 20% of games as test
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np, y_np, test_size=0.2, shuffle=False
    )

    # to torch tensors
    X_train = torch.from_numpy(X_train_np)
    y_train = torch.from_numpy(y_train_np)
    X_test = torch.from_numpy(X_test_np)
    y_test = torch.from_numpy(y_test_np)

    # define linear regression model
    model = nn.Linear(X_train.shape[1], 1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 1000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()

    # evaluate on test set
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        test_loss = criterion(test_preds, y_test).item()

    print(f"Test MSE: {test_loss:.3f}")
    print("First 5 predictions vs actuals:")
    for i in range(min(5, len(test_preds))):
        print(f"Pred: {test_preds[i].item():.2f}, Actual: {y_test[i].item():.2f}")

# manual test run for today
'''
X_test_np_today = [[36.6, 23, 7, 4, .522, .143, .688, .543, 1.000, 4, 5, 5, 16.5, 2, 29, 1, 0, 114.0, 119.7, -5.7, 96.9]]
X_test_today = torch.tensor(X_test_np_today, dtype=torch.float32)
y_today = model(X_test_today)
print(y_today.item())
'''