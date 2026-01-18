import os
import pandas as pd

# Construct the absolute path dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.normpath(os.path.join(script_dir, '../data-collection/output_data/ensemble_player_prediction.csv'))

print(f"Looking for file at: {csv_path}")  # Debug print to verify path

# 2. Check if file exists before reading
if os.path.exists(csv_path):
    output_df = pd.read_csv(csv_path)
else:
    print("File not found. Creating a new empty DataFrame.")
    output_df = pd.DataFrame(columns=['personId', 'firstName', 'lastName', 'Date', 'predictedPoints', 'MSE_lr', 'MSE_rf'])


df_lr = pd.read_csv('../data-collection/output_data/linear_player_prediction.csv')
df_rf = pd.read_csv('../data-collection/output_data/randomForestReg_player_prediction.csv')

ensemble_df = pd.merge(
    df_lr, 
    df_rf, 
    on=['personId', 'firstName', 'lastName', 'Date'], 
    suffixes=('_lr', '_rf') 
)

# Store into respective LR, RFR variables (pd series) containing data for ALL players aligned by row
lr_pred = ensemble_df['predictedPoints_lr']
lr_mse  = ensemble_df['MSE_lr']

rf_pred = ensemble_df['predictedPoints_rf']
rf_mse  = ensemble_df['MSE_rf']

player_id = ensemble_df['personId']

# Iterate and Update output_df
print("--- Updating Ensemble Predictions ---")

for index, row in ensemble_df.iterrows():
    player_id = row['personId']
    
    # Calculate Weights (Inverse of MSE)
    # Handle division by zero if MSE is 0
    mse_lr = row['MSE_lr'] if row['MSE_lr'] > 0 else 0.001
    mse_rf = row['MSE_rf'] if row['MSE_rf'] > 0 else 0.001
    
    weight_lr = 1 / mse_lr
    weight_rf = 1 / mse_rf
    
    total_weight = weight_lr + weight_rf
    
    # Normalize weights
    w_lr = weight_lr / total_weight
    w_rf = weight_rf / total_weight
    
    # Calculate Weighted Average Prediction
    final_pred = (row['predictedPoints_lr'] * w_lr) + (row['predictedPoints_rf'] * w_rf)
    
    # Prepare the Row Data
    new_data = {
        'personId': player_id,
        'firstName': row['firstName'],
        'lastName': row['lastName'],
        'Date': row['Date'],
        'predictedPoints': round(final_pred, 2), 
        'MSE_lr': mse_lr, 
        'MSE_rf': mse_rf
    }
    
    # Update if exists, append if not
    # Check if this player ID already exists in output_df
    mask = output_df['personId'] == player_id
    
    if mask.any():
        # Update existing row
        output_df.loc[mask, ['firstName', 'lastName', 'Date', 'predictedPoints', 'MSE_lr', 'MSE_rf']] = [
            new_data['firstName'], 
            new_data['lastName'], 
            new_data['Date'], 
            new_data['predictedPoints'],
            new_data['MSE_lr'],
            new_data['MSE_rf']
        ]
    else:
        # APPEND a new row
        # We convert the single dict to a DataFrame and concat it
        new_row_df = pd.DataFrame([new_data])
        output_df = pd.concat([output_df, new_row_df], ignore_index=True)
        print(f"Added new player: {new_data['firstName']} {new_data['lastName']}")

# 4. Save the final updated table
output_df.to_csv(csv_path, index=False)
print(f"\nEnsemble predictions saved to: {csv_path}")
print(output_df.head())

