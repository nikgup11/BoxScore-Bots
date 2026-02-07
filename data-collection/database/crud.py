from sqlalchemy.orm import Session
from . import models, schemas

def create_player_stats(
    db: Session,
    stats: schemas.PlayerStatsCreate,
):
    db_obj = models.PlayerStats(**stats.model_dump())
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj

def get_player_stats(db: Session, limit: int = 50):
    return db.query(models.PlayerStats).limit(limit).all()