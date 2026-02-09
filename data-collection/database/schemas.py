from pydantic import BaseModel

class PlayerStatsCreate(BaseModel):
    player_id: int
    player_name: str
    team: str
    points: float | None = None
    assists: float | None = None
    rebounds: float | None = None

class PlayerStatsRead(PlayerStatsCreate):
    id: int

    class ConfigDict:
        from_attributes = True  # Pydantic v2