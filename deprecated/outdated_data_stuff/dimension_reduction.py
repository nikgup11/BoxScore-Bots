import pandas as pd
import sklearn.decomposition as skd

def reduce_dimensions_via_pca(df, n_components=2):
    pca = skd.PCA(n_components=n_components)
    df_reduced = pca.fit_transform(df)
    return pd.DataFrame(df_reduced)

#def get_player_dfs():
    

if __name__ == "__main__": 
    player_df = pd.read_csv("./espn_stats_getter/bbref_players_games_simple/achiupr01_Precious_Achiuwa_last3.csv")
    # drop rows with NaN so PCA won't raise "Input X contains NaN"
    num_df = player_df.select_dtypes(include=['number']).dropna()
    reduced = reduce_dimensions_via_pca(num_df, n_components=2)
    print(reduced.head())