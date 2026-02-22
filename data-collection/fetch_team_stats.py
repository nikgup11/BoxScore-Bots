import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats

print("Fetching team stats...")

stats = leaguedashteamstats.LeagueDashTeamStats(
    season='2025-26',
    season_type_all_star='Regular Season',
    measure_type_detailed_defense="Advanced",
    timeout=120
)

df = stats.get_data_frames()[0]
df.to_csv("team_stats_2025_26.csv", index=False)

print("Saved locally.")