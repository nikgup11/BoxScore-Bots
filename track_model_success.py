import pandas as pd

output_file = './data-collection/output_data/projection_tracking.csv'

# ======================
# Load data
# ======================
projection_tracking = pd.read_csv(output_file)
tonights_projections = pd.read_csv('./data-collection/output_data/tonights_projections_rnn.csv')
tonights_projections_xgb = pd.read_csv('./data-collection/output_data/tonights_projections_xgb.csv')
actual_results = pd.read_csv('./data-collection/clean_data/PlayerStatistics.csv', low_memory=False)

# ======================
# Prep actual results
# ======================
actual_results['Name'] = actual_results['firstName'] + ' ' + actual_results['lastName']
actual_results['Game_Date'] = pd.to_datetime(actual_results['gameDateTimeEst']).dt.date

actual_results = actual_results.rename(columns={
    'opponentteamName': 'Opp',
    'points': 'Actual_PTS',
    'reboundsTotal': 'Actual_REB',
    'assists': 'Actual_AST'
})

actual_results = actual_results[['Name', 'Game_Date', 'Opp', 'Actual_PTS', 'Actual_REB', 'Actual_AST']]
actual_results['Actual_PRA'] = actual_results['Actual_PTS'] + actual_results['Actual_REB'] + actual_results['Actual_AST']

# ======================
# Standardize Date Columns
# ======================
projection_tracking['Game_Date'] = pd.to_datetime(projection_tracking['Game_Date']).dt.date
tonights_projections['Game_Date'] = pd.to_datetime(tonights_projections['Game_Date']).dt.date
tonights_projections_xgb['Game_Date'] = pd.to_datetime(tonights_projections_xgb['Game_Date']).dt.date

# ======================
# Rename XGB columns correctly
# ======================
xgb_rename_map = {
    'Proj_PTS': 'XGB_PTS',
    'Proj_REB': 'XGB_REB',
    'Proj_AST': 'XGB_AST',
    'Total_PRA': 'XGB_PRA'
}
tonights_projections_xgb = tonights_projections_xgb.rename(columns=xgb_rename_map)

# ======================
# 1. UPDATE EXISTING ROWS WITH ACTUALS
# ======================
projection_tracking = projection_tracking.merge(
    actual_results,
    on=['Name', 'Game_Date', 'Opp'],
    how='left',
    suffixes=('', '_new')
)

for stat in ['PTS', 'REB', 'AST', 'PRA']:
    projection_tracking[f'Actual_{stat}'] = projection_tracking[f'Actual_{stat}'].combine_first(
        projection_tracking[f'Actual_{stat}_new']
    )

projection_tracking.drop(columns=[c for c in projection_tracking.columns if c.endswith('_new')], inplace=True)

# ======================
# 2. MERGE XGB PROJECTIONS INTO EXISTING ROWS
# ======================
projection_tracking = projection_tracking.merge(
    tonights_projections_xgb[['Game_Date', 'Name', 'Team', 'Opp', 'XGB_PTS', 'XGB_REB', 'XGB_AST', 'XGB_PRA']],
    on=['Game_Date', 'Name', 'Team', 'Opp'],
    how='left',
    suffixes=('', '_new')
)

for stat in ['PTS', 'REB', 'AST', 'PRA']:
    projection_tracking[f'XGB_{stat}'] = projection_tracking[f'XGB_{stat}'].combine_first(
        projection_tracking.get(f'XGB_{stat}_new')
    )

projection_tracking.drop(columns=[c for c in projection_tracking.columns if c.endswith('_new')], inplace=True)

# ======================
# 3. PREP NEW ROWS (RNN + XGB)
# ======================
combined_projections = tonights_projections.merge(
    tonights_projections_xgb[['Game_Date', 'Name', 'Team', 'Opp', 'XGB_PTS', 'XGB_REB', 'XGB_AST', 'XGB_PRA']],
    on=['Game_Date', 'Name', 'Team', 'Opp'],
    how='left'
)

# Find projections not already in tracking
new_rows = combined_projections.merge(
    projection_tracking[['Game_Date', 'Name', 'Team', 'Opp']],
    on=['Game_Date', 'Name', 'Team', 'Opp'],
    how='left',
    indicator=True
)
new_rows = new_rows[new_rows['_merge'] == 'left_only'].drop(columns='_merge')

# Add empty actual columns
new_rows['Actual_PTS'] = pd.NA
new_rows['Actual_REB'] = pd.NA
new_rows['Actual_AST'] = pd.NA
new_rows['Actual_PRA'] = pd.NA

# ======================
# 4. COMBINE & SAVE
# ======================
final_table = pd.concat([projection_tracking, new_rows], ignore_index=True)
final_table.to_csv(output_file, index=False)

print("Projection tracking updated successfully with XGB columns.")