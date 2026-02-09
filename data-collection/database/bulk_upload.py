import pandas as pd
from models import Player
from database import SessionLocal

def bulk_insert_players(csv_path):
    df = pd.read_csv(csv_path)
    db = SessionLocal()

    players = [
        Player(
            name=row["Name"],
            team=row["Team"],
            opp=row["Opp"],
            game_date=row["Game_Date"],
            game_time=row["Game_Time"],
            proj_pts=row["Proj_PTS"],
            proj_reb=row["Proj_REB"],
            proj_ast=row["Proj_AST"],
            total_pra=row["Total_PRA"],
        )
        for _, row in df.iterrows()
    ]

    db.bulk_save_objects(players)
    db.commit()
    db.close()

def bulk_remove_players():
    db = SessionLocal()
    db.query(Player).delete()
    db.commit()
    db.close()

bulk_remove_players()
bulk_insert_players("data-collection/output_data/tonights_projections_rnn.csv")