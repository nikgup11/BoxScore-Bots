import pandas as pd
from nba_api.stats.static import teams as nba_teams

def track_model_success(projection_file, statistics_file, output_file):
    projection_data = pd.read_csv(projection_file, low_memory=False)
    game_data = pd.read_csv(statistics_file, low_memory=False)

    # Remove rows with empty Name
    projection_data = projection_data.dropna(subset=['Name'])
    projection_data['Name'] = projection_data['Name'].astype(str).str.strip()

    output = projection_data.copy()
    output['Actual_PTS'] = None
    output['Actual_REB'] = None
    output['Actual_AST'] = None
    output['Actual_PRA'] = None

    game_data['gameDateTimeEst'] = pd.to_datetime(game_data['gameDateTimeEst'])
    game_data['full_name'] = game_data['firstName'] + ' ' + game_data['lastName']

    print("Dates in game_data:")
    print(game_data['gameDateTimeEst'].dt.date.value_counts().head(5))
    print()

    print("Dates in projections:")
    print(projection_data['Game_Date'].value_counts().head(5))
    print()

    for idx, row in projection_data.iterrows():
        name = row['Name'].strip()
        team = row['Team'].strip()
        game_date = pd.to_datetime(row['Game_Date'])
        
        # Match by full name, team, and date
        match = game_data[
            (game_data['full_name'] == name) &
            (game_data['playerteamName'] == team) &
            (game_data['gameDateTimeEst'].dt.date == game_date.date())
        ]
        
        if not match.empty:
            actual_row = match.iloc[0]
            output.at[idx, 'Actual_PTS'] = actual_row['points']
            output.at[idx, 'Actual_REB'] = actual_row['reboundsTotal']
            output.at[idx, 'Actual_AST'] = actual_row['assists']
            output.at[idx, 'Actual_PRA'] = actual_row['points'] + actual_row['reboundsTotal'] + actual_row['assists']

    output = output.dropna(subset=['Name'])

    # Merge with existing tracking file: keep new data, remove old duplicates
    tracking_file = './data-collection/output_data/projection_tracking.csv'
    try:
        existing_data = pd.read_csv(tracking_file, low_memory=False)
        # Only keep existing rows from dates that are in the current projection set
        existing_data = existing_data[existing_data['Game_Date'].isin(output['Game_Date'])]
        
        # Remove rows from existing data that match Name + Game_Date
        existing_data['_key'] = existing_data['Name'] + '|' + existing_data['Game_Date'].astype(str)
        output['_key'] = output['Name'] + '|' + output['Game_Date'].astype(str)
        
        # Keep only existing rows that aren't in the new output
        existing_data = existing_data[~existing_data['_key'].isin(output['_key'])]
        existing_data = existing_data.drop(columns=['_key'])
        output = output.drop(columns=['_key'])
        
        # Concatenate new data with non-duplicate existing data
        output = pd.concat([existing_data, output], ignore_index=True)
    except FileNotFoundError:
        pass

    output.to_csv(output_file, index=False)
    print(f"output saved to {output_file}")

track_model_success('./data-collection/output_data/tonights_projections.csv', './data-collection/clean_data/PlayerStatistics.csv', './data-collection/output_data/projection_tracking.csv')
track_model_success('./data-collection/output_data/tonights_projections_rnn.csv', './data-collection/clean_data/PlayerStatistics.csv', './data-collection/output_data/projection_tracking_rnn.csv')