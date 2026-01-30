import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
import pickle
import os

# ==========================================
# CONFIGURATION
# ==========================================
SEQUENCE_LENGTH = 10
MODEL_PATH = './ml_models/rnn_model.pth'
SCALER_PATH = './ml_models/rnn_scaler.pkl'

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
# DATA LOADING & PREPROCESSING
# ==========================================
print("Loading data...")
data = pd.read_csv('./data-collection/clean_data/PlayerStatistics.csv', low_memory=False)

data['gameDateTimeEst'] = pd.to_datetime(data['gameDateTimeEst'], utc=True)
data = data[data['OppPace'].notna()].copy()
data = data.sort_values(['personId', 'gameDateTimeEst'])

# ==========================================
# FEATURE ENGINEERING & SCALING
# ==========================================
feature_cols = ['points', 'reboundsTotal', 'assists', 'numMinutes', 
                'fieldGoalsAttempted', 'OppOffRtg', 'OppDefRtg', 'OppPace', 'home']

data_clean = data[feature_cols + ['personId']].dropna().copy()
scaler = StandardScaler()
data_clean[feature_cols] = scaler.fit_transform(data_clean[feature_cols])

# ==========================================
# PREPARE SEQUENCES
# ==========================================
print("Preparing sequences...")
X_sequences = []
y_values = []

for player_id in data_clean['personId'].unique():
    player_data = data_clean[data_clean['personId'] == player_id].sort_values('personId')
    player_data = player_data[feature_cols].values
    
    if len(player_data) < SEQUENCE_LENGTH + 1:
        continue
    
    for i in range(len(player_data) - SEQUENCE_LENGTH):
        sequence = player_data[i:i+SEQUENCE_LENGTH]
        target = player_data[i+SEQUENCE_LENGTH, :3]
        
        X_sequences.append(torch.tensor(sequence, dtype=torch.float32))
        y_values.append(torch.tensor(target, dtype=torch.float32))

X_tensor = pad_sequence(X_sequences, batch_first=True)
y_tensor = torch.stack(y_values)

split_idx = int(0.8 * len(X_tensor))
X_train = X_tensor[:split_idx]
y_train = y_tensor[:split_idx]
X_test = X_tensor[split_idx:]
y_test = y_tensor[split_idx:]

# ==========================================
# TRAIN MODEL
# ==========================================
print(f"Training RNN with {len(X_train)} sequences...")
model = NBAPlayerRNN(input_size=len(feature_cols), hidden_size=64, num_layers=2, output_size=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 200
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# ==========================================
# EVALUATE MODEL
# ==========================================
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

# ==========================================
# SAVE MODEL & SCALER
# ==========================================
print(f"Saving model to {MODEL_PATH}...")
torch.save(model.state_dict(), MODEL_PATH)

print(f"Saving scaler to {SCALER_PATH}...")
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")
