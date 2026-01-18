import pandas as pd

# Load regular season data
data = pd.read_csv('./data-collection/raw_data/PlayerStatistics.csv')
print(data['gameType'].value_counts())  # Check available game typess

output = data[(data['gameType'] != 'Preseason')].copy()
