from database import engine
from models import DailyDist

DailyDist.metadata.create_all(bind=engine)