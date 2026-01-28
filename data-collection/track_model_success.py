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

    print("Columns in game_data:", game_data.columns.tolist())
    print()
    
    print("Dates in game_data:")
    print(game_data['gameDateTimeEst'].dt.date.value_counts().head(5))
    print()

    print("Dates in projections:")
    print(projection_data['Game_Date'].value_counts().head(5))
    print()
    
    print("Sample teams in game_data:")
    print(game_data['playerteamName'].unique()[:10])
    print()

    for idx, row in projection_data.iterrows():
        name = row['Name'].strip()
        team = row['Team'].strip()
        game_date = pd.to_datetime(row['Game_Date'])
        
        # Normalize team names: handle common variations
        team_normalized = team.replace(' ', '').lower()
        
        # Match by full name, team, and date with flexible matching
        match = game_data[
            (game_data['full_name'].str.strip() == name) &
            (game_data['playerteamName'].str.replace(' ', '').str.lower() == team_normalized) &
            (game_data['gameDateTimeEst'].dt.date == game_date.date())
        ]
        
        if not match.empty:
            actual_row = match.iloc[0]
            output.at[idx, 'Actual_PTS'] = actual_row['points']
            output.at[idx, 'Actual_REB'] = actual_row['reboundsTotal']
            output.at[idx, 'Actual_AST'] = actual_row['assists']
            output.at[idx, 'Actual_PRA'] = actual_row['points'] + actual_row['reboundsTotal'] + actual_row['assists']
        else:
            # Debug: print mismatches for first few rows to diagnose
            if idx < 3:
                print(f"No match for {name} ({team}) on {game_date.date()}")
                matching_dates = game_data[game_data['gameDateTimeEst'].dt.date == game_date.date()]
                if not matching_dates.empty:
                    print(f"  Available players on that date: {matching_dates['full_name'].unique()[:5]}")

    output = output.dropna(subset=['Name'])

    # Load existing tracking file and append new data
    tracking_file = './data-collection/output_data/projection_tracking.csv'
    try:
        existing_data = pd.read_csv(tracking_file, low_memory=False)
        
        # Create a key to identify duplicate projections (same player, same date, same opponent)
        existing_data['_key'] = existing_data['Name'] + '|' + existing_data['Game_Date'].astype(str) + '|' + existing_data['Opp'].astype(str)
        output['_key'] = output['Name'] + '|' + output['Game_Date'].astype(str) + '|' + output['Opp'].astype(str)
        
        # Keep only existing rows that aren't in the new output (avoid duplicate projections for same game)
        existing_data = existing_data[~existing_data['_key'].isin(output['_key'])]
        existing_data = existing_data.drop(columns=['_key'])
        output = output.drop(columns=['_key'])
        
        # Concatenate all historical data with new projections
        output = pd.concat([existing_data, output], ignore_index=True)
        print(f"Total rows: {len(output)} (existing: {len(existing_data)}, new: {len(output) - len(existing_data)})")
    except FileNotFoundError:
        print(f"No existing tracking file found. Creating new one.")

    # Remove any duplicate rows (same player + game date)
    output = output.drop_duplicates(subset=['Name', 'Game_Date'], keep='first')

    output.to_csv(output_file, index=False)
    print(f"output saved to {output_file}")

track_model_success('./data-collection/output_data/tonights_projections.csv', './data-collection/clean_data/PlayerStatistics.csv', './data-collection/output_data/projection_tracking.csv')
track_model_success('./data-collection/output_data/tonights_projections_rnn.csv', './data-collection/clean_data/PlayerStatistics.csv', './data-collection/output_data/projection_tracking_rnn.csv')