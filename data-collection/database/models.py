from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.sql import func
from database import Base

class Player(Base):
    __tablename__ = "projections"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    team = Column(String, nullable=False)
    opp = Column(String, nullable=False)
    game_date = Column(String, nullable=False)
    game_time = Column(String, nullable=False)
    proj_pts = Column(Float, nullable=False)
    proj_reb = Column(Float, nullable=False)
    proj_ast = Column(Float, nullable=False)
    total_pra = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())