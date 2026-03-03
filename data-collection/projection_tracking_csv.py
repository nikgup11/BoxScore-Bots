import pandas as pd
import numpy as np

# ======================
# Load Tracking
# ======================
tracking = pd.read_csv('./data-collection/output_data/projection_tracking.csv')

tracking['Game_Date'] = pd.to_datetime(tracking['Game_Date'])
tracking.sort_values(by=['Game_Date'], ascending=True, inplace=True)

# ======================
# Only use games with actual results
# ======================
tracking_clean = tracking.dropna(subset=['Actual_PTS'])

# ======================
# CLOSE PROJECTIONS (RNN)
# ======================
close_projections = tracking_clean[
    (abs(tracking_clean['Proj_PTS'] - tracking_clean['Actual_PTS']) <= 3) &
    (tracking_clean['Actual_PTS'] != 0) &
    (tracking_clean['Proj_PTS'] != 0)
]

print(close_projections[['Name', 'Proj_PTS', 'Actual_PTS']])

# ======================
# OVERALL MODEL ERROR
# ======================

# ---- RNN ----
rnn_dist_pts = (tracking_clean['Actual_PTS'] - tracking_clean['Proj_PTS']).abs()
rnn_dist_reb = (tracking_clean['Actual_REB'] - tracking_clean['Proj_REB']).abs()
rnn_dist_ast = (tracking_clean['Actual_AST'] - tracking_clean['Proj_AST']).abs()

print("RNN Avg Error PTS:", rnn_dist_pts.mean())
print("RNN Avg Error REB:", rnn_dist_reb.mean())
print("RNN Avg Error AST:", rnn_dist_ast.mean())

# ---- XGB ----
xgb_dist_pts = (tracking_clean['Actual_PTS'] - tracking_clean['XGB_PTS']).abs()
xgb_dist_reb = (tracking_clean['Actual_REB'] - tracking_clean['XGB_REB']).abs()
xgb_dist_ast = (tracking_clean['Actual_AST'] - tracking_clean['XGB_AST']).abs()

print("XGB Avg Error PTS:", xgb_dist_pts.mean())
print("XGB Avg Error REB:", xgb_dist_reb.mean())
print("XGB Avg Error AST:", xgb_dist_ast.mean())

# ======================
# Per-Date Average Error
# ======================

output_df = tracking_clean.groupby('Game_Date').apply(
    lambda df: pd.Series({
        'RNN_Average_Dist_PTS': (df['Actual_PTS'] - df['Proj_PTS']).abs().mean(),
        'RNN_Average_Dist_REB': (df['Actual_REB'] - df['Proj_REB']).abs().mean(),
        'RNN_Average_Dist_AST': (df['Actual_AST'] - df['Proj_AST']).abs().mean(),
        'XGB_Average_Dist_PTS': (df['Actual_PTS'] - df['XGB_PTS']).abs().mean(),
        'XGB_Average_Dist_REB': (df['Actual_REB'] - df['XGB_REB']).abs().mean(),
        'XGB_Average_Dist_AST': (df['Actual_AST'] - df['XGB_AST']).abs().mean(),
    })
).reset_index()

output_df.to_csv(
    './data-collection/output_data/average_projection_dist_by_date.csv',
    index=False
)

print("Saved per-date error comparison.")