import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
from torch.nn.utils.rnn import pad_sequence

# ==========================================
# CONFIGURATION
# ==========================================
SEQUENCE_LENGTH = 10
MODEL_DIR = './ml_models/'
SCALER_PATH = MODEL_DIR + 'xgb_scaler.pkl'

# ---------------------------
# DATA GATHERING
# ---------------------------
data = pd.read_csv('./data-collection/clean_data/PlayerStatistics.csv', low_memory=False)
data['gameDateTimeEst'] = pd.to_datetime(data['gameDateTimeEst'], utc=True)

# ---------------------------
# FILTER LAST 5 SEASONS
# ---------------------------
latest_date = data['gameDateTimeEst'].max()
five_years_ago = latest_date - pd.DateOffset(years=5)
data = data[data['gameDateTimeEst'] >= five_years_ago].copy()
print(f"Filtered data to last 5 seasons: {len(data)} rows remain.")

# Keep only rows with opponent pace info
data = data[data['OppPace'].notna()].copy()
data = data.sort_values(['personId', 'gameDateTimeEst'])

# ==========================================
# FEATURE ENGINEERING & SCALING
# ==========================================
feature_cols = [
    'points', 'reboundsTotal', 'assists',
    'numMinutes', 'fieldGoalsAttempted',
    'OppOffRtg', 'OppDefRtg', 'OppPace', 'home'
]

target_cols = ['points', 'reboundsTotal', 'assists']

data_clean = data[feature_cols + ['personId', 'gameDateTimeEst']].dropna().copy()

scaler = StandardScaler()
data_clean[feature_cols] = scaler.fit_transform(data_clean[feature_cols])

# Save scaler
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

# ==========================================
# PREPARE SEQUENCES
# ==========================================
print("Preparing sequences...")
X_sequences = []
y_values = []

for player_id in data_clean['personId'].unique():
    player_data = data_clean[data_clean['personId'] == player_id].sort_values('gameDateTimeEst')
    player_array = player_data[feature_cols].values
    
    if len(player_array) < SEQUENCE_LENGTH + 1:
        continue
    
    for i in range(len(player_array) - SEQUENCE_LENGTH):
        sequence = player_array[i:i+SEQUENCE_LENGTH]
        target = player_array[i+SEQUENCE_LENGTH, :3]  # predict pts, reb, ast
        
        X_sequences.append(sequence)
        y_values.append(target)

# Convert to numpy
X_array = np.array(X_sequences)
y_array = np.array(y_values)

# Flatten sequences for XGBoost (VERY IMPORTANT)
num_samples = X_array.shape[0]
X_array = X_array.reshape(num_samples, -1)

# Train/Test Split
split_idx = int(0.8 * num_samples)
X_train = X_array[:split_idx]
y_train = y_array[:split_idx]
X_test = X_array[split_idx:]
y_test = y_array[split_idx:]

# ==========================================
# TRAIN XGBOOST MODELS (One per target)
# ==========================================
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

models = {}
predictions = []

for i, target_name in enumerate(target_cols):
    print(f"\nTraining model for {target_name}...")
    
    train_dmatrix = xgb.DMatrix(X_train, label=y_train[:, i])
    test_dmatrix = xgb.DMatrix(X_test, label=y_test[:, i])
    
    model = xgb.train(
        params,
        train_dmatrix,
        num_boost_round=300,
        evals=[(test_dmatrix, "eval")],
        verbose_eval=50
    )
    
    model.save_model(MODEL_DIR + f"xgb_{target_name}.json")
    
    preds = model.predict(test_dmatrix)
    predictions.append(preds)
    
    mse = mean_squared_error(y_test[:, i], preds)
    print(f"{target_name} MSE: {mse:.4f}")
    
    models[target_name] = model

# Stack predictions back together
predictions = np.column_stack(predictions)

overall_mse = mean_squared_error(y_test, predictions)
print(f"\nOverall MSE (combined): {overall_mse:.4f}")