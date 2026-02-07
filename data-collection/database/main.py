from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Player
from schemas import PlayerStatsCreate, PlayerStatsRead

app = FastAPI()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/players", response_model=PlayerStatsRead)
def create_player(player: PlayerStatsCreate, db: Session = Depends(get_db)):
    db_player = Player(**player.dict())
    db.add(db_player)
    db.commit()
    db.refresh(db_player)
    return db_player
