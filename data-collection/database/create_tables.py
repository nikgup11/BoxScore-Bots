from database import engine
from models import Difference

Difference.metadata.create_all(bind=engine)