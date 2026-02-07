import pandas as pd

projections = pd.read_csv('./data-collection/output_data/tonights_projections_rnn.csv')
sportsbook_data = pd.read_csv('./data-collection/sportsbook_data/playerprops.csv')
output_df = pd.DataFrame()
for player in sportsbook_data.itertuples():
    # sportsbook data
    player_name = player.player  # use dot notation
    player_ou = player.line      # use dot notation

    # projections data
    projections_for_player = projections[projections['Name'] == player_name]
    if not projections_for_player.empty:
        points_proj = projections_for_player['Proj_PTS'].iloc[0]
        
        if points_proj > player_ou:
            print(f"Projection for {player_name} is OVER the sportsbook line: {points_proj} vs {player_ou}")
            print(f"The Difference between the projection and the sportsbook line is: {points_proj - player_ou}")
            output_df = output_df._append({
                "Player": player_name,
                "Projection": points_proj,
                "Sportsbook Line": player_ou,
                "Difference": points_proj - player_ou
            }, ignore_index=True)
        elif points_proj < player_ou:
            if points_proj == 0:
                continue
            print(f"Projection for {player_name} is UNDER the sportsbook line: {points_proj} vs {player_ou}")
            print(f"The Difference between the projection and the sportsbook line is: {points_proj - player_ou}")
            output_df = output_df._append({
                "Player": player_name,
                "Projection": points_proj,
                "Sportsbook Line": player_ou,
                "Difference": points_proj - player_ou
            }, ignore_index=True)
    else:
        print(f"No projection found for {player_name}")
    
output_df.to_csv('./data-collection/output_data/projection_vs_sportsbook.csv', index=False)