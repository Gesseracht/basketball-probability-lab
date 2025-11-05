import numpy as np
import pandas as pd
import requests


class ELO:
    def __init__(self):
        self.teams = {}
    
    def _initialize_team(self, team_name):
        # Later, change team_name by team_id and add in the dict team_name
        if team_name not in self.teams:
            self.teams[team_name] = {
                'elo': 1000,
                'n_games':0,
                'history': []
            }

    def _calculate_win_probability(self, elo1, elo2):
        return 1 / (1 + 10 ** ((elo2 - elo1) / 400))

    def get_k(self, team_name):
        self._initialize_team(team_name)
        n_games = self.teams[team_name]['n_games']
        return np.round(16 + 16 * np.exp(-0.072 * (n_games - 1)), 0)

    def get_elo(self, team_name):
        self._initialize_team(team_name)
        return self.teams[team_name]['elo']
    
    def predict_match(self, home, away):
        elo_home = self.get_elo(home)
        elo_away = self.get_elo(away)

        p_home = self._calculate_win_probability(elo_home, elo_away)
        p_away = 1 - p_home

        return {
            'home': home, 'away': away,
            'p_home_win': np.round(p_home, decimals=4),'p_away_win': np.round(p_away, decimals=4)
        }
    

    def next_games(self, df_upcoming):
        predictions = []

        for _, row in df_upcoming.iterrows():
            team_a = row['TEAM_NAME_HOME']
            team_b = row['TEAM_NAME_AWAY']
            
            pred = self.predict_match(team_a, team_b)
            pred['date'] = row['GAME_DATE']
            pred['elo_home'] = self.get_elo(team_a)
            pred['elo_away'] = self.get_elo(team_b)

            predictions.append(pred)

        return pd.DataFrame(predictions)
    

    def home_advantage(self, home):
        pass
    
    def update_match(self, date, home, away, winner):
        r1 = self.get_elo(home)
        r2 = self.get_elo(away)

        k1 = self.get_k(home)
        k2 = self.get_k(away)

        e1 = self._calculate_win_probability(r1, r2)
        e2 = 1 - e1

        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))

        # winner stored as W for Home wins and L for Away wins
        s1 = int(winner == 'W')
        s2 = int(winner == 'L')

        r1_prime = round(r1 + k1 * (s1 - e1), 0)
        r2_prime = round(r2 + k2 * (s2 - e2), 0)

        self.teams[home]['elo'] = r1_prime
        self.teams[away]['elo'] = r2_prime

        self.teams[home]['n_games'] += 1
        self.teams[away]['n_games'] += 1

        self.teams[home]['history'].append((date, r1_prime))
        self.teams[away]['history'].append((date, r2_prime))

    def get_leaderboard(self, top_n=10):
        sorted_teams = sorted(self.teams.items(), 
                            key=lambda x: x[1]['elo'], 
                            reverse=True)[:top_n]
        
        data = [
            {
                'Rank': i + 1,
                'Team': team_name,
                'Elo': info['elo'],
                'Games': info['n_games']
            }
            for i, (team_name, info) in enumerate(sorted_teams)
        ]
        
        df = pd.DataFrame(data)
        df.set_index('Rank', inplace=True)
        return df
    
    def get_team_history(self, team_name : str):
        if team_name not in self.teams:
            raise ValueError(f"Team '{team_name}' not found")
        history = self.teams[team_name]['history']
        return pd.DataFrame(history, columns=['date', 'elo'])

    
