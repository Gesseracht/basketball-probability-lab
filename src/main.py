import h5py
import os
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
DATA_DIR = project_root / "data"
SIM_DIR = DATA_DIR / "simulations"

CSV_PARAMS = os.path.join(DATA_DIR, "processed", "params.csv")
CSV_PLAYERS = os.path.join(DATA_DIR, "raw", "available_players.csv")
SIM_DIR = os.path.join(DATA_DIR, "simulations")

def load_player_samples(player_id, stat):
    file_path = os.path.join(SIM_DIR, f"{stat}.h5")
    with h5py.File(file_path, "r") as f:
        player_ids = f["PLAYER_ID"][:]
        idx = np.where(player_ids == player_id)[0]
        if len(idx) == 0:
            return None
        samples = f["samples"][idx[0], :]
    return samples

print(load_player_samples(player_id=1629029, stat='PTS'))

