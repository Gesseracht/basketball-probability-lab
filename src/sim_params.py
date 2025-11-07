import numpy as np
import pandas as pd
import math
import sys
from pathlib import Path
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parent = Path(__file__).resolve().parent.parent
players_stats = os.path.join(parent, r'data\raw\players.csv')
players = os.path.join(parent, r'data\raw\available_players.csv')
params = os.path.join(parent, r'data\processed\params.csv')

df = pd.read_csv(players_stats, parse_dates=['GAME_DATE'])
players = pd.read_csv(players)

df = df[df["PLAYER_ID"].isin(players["PLAYER_ID"])]
df = df.reset_index(drop=True)

stats = ["REB","AST","TOV","STL","BLK","PTS"]

df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=False)
orig_index = df["index"].values

frac = 0.25
min_half = 3
max_half = 200 

results_mean = {s: np.full(len(df), np.nan, dtype=np.float32) for s in stats}
results_std  = {s: np.full(len(df), np.nan, dtype=np.float32) for s in stats}

grouped = df.groupby("PLAYER_ID", sort=False)

for pid, group in grouped:
    idx = group.index.to_numpy()
    n = len(idx)
    match_idx = np.arange(1, n+1, dtype=np.int64)

    half_life = np.floor(frac * match_idx).astype(np.int64)
    half_life = np.clip(half_life, min_half, max_half)

    lam = np.power(0.5, 1.0 / half_life)
    alpha = 1.0 - lam

    for s in stats:
        x = group[s].to_numpy(dtype=np.float64)
        mu = np.empty(n, dtype=np.float64)
        m2 = np.empty(n, dtype=np.float64)

        mu[0] = x[0]
        m2[0] = x[0]*x[0]

        for t in range(1, n):
            lam_t = lam[t]
            alpha_t = alpha[t]
            xt = x[t]
            mu[t] = lam_t * mu[t-1] + alpha_t * xt
            m2[t] = lam_t * m2[t-1] + alpha_t * (xt*xt)

        var = m2 - mu*mu
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)

        results_mean[s][idx] = mu
        results_std[s][idx]  = std

for s in stats:
    df[f"{s}_adap_mean"] = results_mean[s]
    df[f"{s}_adap_std"]  = results_std[s]

df_out = df.sort_values("index").drop(columns=["index"]).reset_index(drop=True)

latest = df.groupby("PLAYER_ID").last().reset_index()
final_params = latest[["PLAYER_ID"] + [f"{s}_adap_mean" for s in stats] + [f"{s}_adap_std" for s in stats]]

final_params.to_csv(params, index=False)
print(final_params.shape)
