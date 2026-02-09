import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY")

SPORT = "basketball_nba"
REGION = "us"
MARKETS = 'player_points'#,player_rebounds,player_assists,player_points_rebounds_assists'
ODDS_FORMAT = 'american'
DATE_FORMAT = 'iso'

# get list of nba games for their ids
def get_events():
    events_response = requests.get(
        f'https://api.the-odds-api.com/v4/sports/{SPORT}/events',
        params={
            'api_key': API_KEY,
            'regions': REGION,
            'oddsFormat': ODDS_FORMAT,
            'dateFormat': DATE_FORMAT,
        }
    )
    #print(events_response.json())
    events_df = pd.DataFrame(events_response.json())
    events_df.to_csv('./data-collection/sportsbook_data/events.csv')

def get_odds(event_id):
    odds_response = requests.get(
        f'https://api.the-odds-api.com/v4/sports/{SPORT}/events/{event_id}/odds',
        params={
            'api_key': API_KEY,
            'regions': REGION,
            'markets': MARKETS,
            'oddsFormat': ODDS_FORMAT,
            'dateFormat': DATE_FORMAT,
            'bookmakers': 'draftkings'
        }
    )
    odds_json = odds_response.json()
    # 🔑 SKIP if no odds available
    if not isinstance(odds_json, dict) or not odds_json.get("bookmakers"):
        return
    odds_to_csv(odds_json)

def odds_to_csv(event_json):
    rows = []

    event_id = event_json.get("id")
    commence_time = event_json.get("commence_time")
    game = f"{event_json.get('home_team')} vs {event_json.get('away_team')}"

    for bookmaker in event_json.get("bookmakers", []):
        bookmaker_name = bookmaker.get("title")

        for market in bookmaker.get("markets", []):
            stat = market.get("key", "").replace("player_", "")

            player_map = {}

            for outcome in market.get("outcomes", []):
                player = outcome.get("description")
                side = outcome.get("name")

                if not player:
                    continue

                if player not in player_map:
                    player_map[player] = {
                        "line": outcome.get("point"),
                        "over_odds": None,
                        "under_odds": None,
                    }

                if side == "Over":
                    player_map[player]["over_odds"] = outcome.get("price")
                elif side == "Under":
                    player_map[player]["under_odds"] = outcome.get("price")

            for player, vals in player_map.items():
                rows.append({
                    "event_id": event_id,
                    "game": game,
                    "commence_time": commence_time,
                    "bookmaker": bookmaker_name,
                    "player": player,
                    "stat": stat,
                    "line": vals["line"],
                    "over_odds": vals["over_odds"],
                    "under_odds": vals["under_odds"],
                })

    if not rows:
        return

    df = pd.DataFrame(rows)

    output_path = "./data-collection/sportsbook_data/playerprops.csv"
    df.to_csv(
        output_path,
        mode="a",
        header=not os.path.exists(output_path),
        index=False
    )


if __name__ == "__main__":
    events_path = "./data-collection/sportsbook_data/events.csv"

    if not os.path.exists(events_path):
        get_events()

    event_ids = pd.read_csv(events_path)["id"]

    for game_id in event_ids:
        get_odds(game_id)
