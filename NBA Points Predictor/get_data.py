from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
import time

# List of star players to collect data for
player_names = ["LeBron James", "Stephen Curry", "Shai Gilgeous-Alexander", "Kevin Durant", "Giannis Antetokounmpo", "Luka Doncic", "Nikola Jokic", "Joel Embiid", "Kawhi Leonard", "Anthony Davis", "James Harden", "Jalen Brunson", "Karl Anthony Towns", "Josh Hart", "Mitchell Robinson", "OG Anunoby"]

all_data = []

for name in player_names:
    print(f"Fetching data for {name}...")
    player_list = players.find_players_by_full_name(name)
    if not player_list:
        continue
    player_id = player_list[0]["id"]

    log = playergamelog.PlayerGameLog(player_id=player_id, season="2025-26")
    df = log.get_data_frames()[0]
    df["PLAYER_NAME"] = name
    all_data.append(df)
    time.sleep(1)  # Be polite to the API

combined = pd.concat(all_data, ignore_index=True)
combined.to_csv("nba_data.csv", index=False)
print(f"Saved {len(combined)} games!")
