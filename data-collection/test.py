from nba_api.stats.endpoints import leaguedashteamstats
import pandas as pd

# Fetch live 2025-26 advanced stats
# This includes OFF_RATING, DEF_RATING, and PACE
stats = leaguedashteamstats.LeagueDashTeamStats(
    season='2023-24',
    measure_type_detailed_defense='Advanced'
)

df_2026 = stats.get_data_frames()[0]

# Filter for just the columns you need
df_clean = df_2026[['TEAM_NAME', 'OFF_RATING', 'DEF_RATING', 'PACE']]
print(df_clean.head())