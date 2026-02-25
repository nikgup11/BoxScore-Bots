import pandas as pd

tracking = pd.read_csv('./data-collection/output_data/projection_tracking.csv')
tracking.sort_values(by=['Game_Date'], ascending=True, inplace=True)

tracking_clean = tracking.dropna(subset=['Actual_PTS'])
close_projections = tracking_clean[(abs(tracking_clean['Proj_PTS'] - tracking_clean['Actual_PTS']) <= 3) & (tracking_clean['Actual_PTS'] != 0) & (tracking_clean['Proj_PTS'] != 0)]
close_projections[['Name', 'Proj_PTS', 'Actual_PTS']]

# on average how far off are the projections?
# if < 1 model is doing well
average_dist_pts = tracking_clean['Actual_PTS'] - tracking_clean['Proj_PTS']
average_dist_pts.mean()
average_dist_reb = tracking_clean['Actual_REB'] - tracking_clean['Proj_REB']
average_dist_reb.mean()
average_dist_ast = tracking_clean['Actual_AST'] - tracking_clean['Proj_AST']
average_dist_ast.mean()

rnn_projections = pd.read_csv('./data-collection/output_data/tonights_projections_rnn.csv')
rnn_projections.sort_values(by=['Total_PRA'], ascending=False, inplace=True)
rnn_projections.head(20)

unique_dates = tracking['Game_Date'].unique()
dist_by_date_pts = {}
dist_by_date_reb = {}
dist_by_date_ast = {}

for date in unique_dates:
    dist_by_date_pts[date] = average_dist_pts[tracking_clean['Game_Date'] == date].mean()
    dist_by_date_reb[date] = average_dist_reb[tracking_clean['Game_Date'] == date].mean()
    dist_by_date_ast[date] = average_dist_ast[tracking_clean['Game_Date'] == date].mean()

output_df = pd.DataFrame({
    'Date': unique_dates,
    'Average_Dist_PTS': [dist_by_date_pts[date] for date in unique_dates],
    'Average_Dist_REB': [dist_by_date_reb[date] for date in unique_dates],
    'Average_Dist_AST': [dist_by_date_ast[date] for date in unique_dates]
})

for day in dist_by_date_pts:
    print(f"{day}: Avg Dist PTS: {dist_by_date_pts[day]:.2f}, Avg Dist REB: {dist_by_date_reb[day]:.2f}, Avg Dist AST: {dist_by_date_ast[day]:.2f}")

output_df.to_csv('./data-collection/output_data/average_projection_dist_by_date.csv', index=False)


