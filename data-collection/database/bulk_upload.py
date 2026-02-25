import pandas as pd
from models import Player, DailyDist
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

def bulk_insert_daily_dist(csv_path: str):
    df = pd.read_csv(csv_path)
    db = SessionLocal()

    daily_dist_records = [
        DailyDist(
            date=pd.to_datetime(row["Date"]).date(),
            average_dist_pts=row["Average_Dist_PTS"] if not pd.isna(row["Average_Dist_PTS"]) else None,
            average_dist_reb=row["Average_Dist_REB"] if not pd.isna(row["Average_Dist_REB"]) else None,
            average_dist_ast=row["Average_Dist_AST"] if not pd.isna(row["Average_Dist_AST"]) else None,
        )
        for _, row in df.iterrows()
    ]

    db.bulk_save_objects(daily_dist_records)
    db.commit()
    db.close()


def bulk_remove_daily_dist():
    db = SessionLocal()
    db.query(DailyDist).delete()
    db.commit()
    db.close()

bulk_remove_players()
bulk_insert_players("data-collection/output_data/tonights_projections_rnn.csv")


bulk_remove_daily_dist()
bulk_insert_daily_dist("data-collection/output_data/average_projection_dist_by_date.csv")