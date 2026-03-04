from database import engine
from models import XGB_Player

XGB_Player.metadata.create_all(bind=engine)