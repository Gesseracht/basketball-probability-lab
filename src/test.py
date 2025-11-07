import os
import pandas as pd
import sys
from pathlib import Path
from elo import ELO
from scraper import Scraper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parent = Path(__file__).resolve().parent.parent
path_file = os.path.join(parent, r'data\raw\available_players.csv')

elo = ELO()
df = pd.read_csv(path_file)
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

for _, row in df.iterrows(): 
    date = row['GAME_DATE']    
    home = row['TEAM_NAME_HOME']     
    away = row['TEAM_NAME_AWAY']     
    winner = row['WL_HOME'] 
    elo.update_match(date=date, home=home, away=away, winner=winner)

print(elo.get_leaderboard(top_n=30))

# python test.py
