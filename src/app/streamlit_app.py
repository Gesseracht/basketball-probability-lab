import streamlit as st
import pandas as pd
import numpy as np
import h5py
import os
from pathlib import Path
import plotly.express as px

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

DATA_DIR = project_root / "data"

CSV_PARAMS = os.path.join(DATA_DIR, "processed", "params.csv")
CSV_PLAYERS = os.path.join(DATA_DIR, "raw", "available_players.csv")
SIM_DIR = os.path.join(DATA_DIR, "simulations")

STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]

@st.cache_data
def load_player_mapping():
    params = pd.read_csv(CSV_PARAMS)
    players = pd.read_csv(CSV_PLAYERS)
    df = params.merge(players, on="PLAYER_ID", how="left")
    return df

players_df = load_player_mapping()

@st.cache_data
def load_player_samples(player_id, stat):
    file_path = os.path.join(SIM_DIR, f"{stat}.h5")
    with h5py.File(file_path, "r") as f:
        player_ids = f["PLAYER_ID"][:]
        idx = np.where(player_ids == player_id)[0]
        if len(idx) == 0:
            return None
        samples = f["samples"][idx[0], :]
    return samples

def compute_probabilities(samples):
    ks = np.arange(0, int(samples.max()) + 1)
    probs = [np.mean(samples >= k) for k in ks]
    return ks, probs

# --- Streamlit UI ---
st.title("🏀 NBA Probability Lab ")

player_name = st.text_input("Enter a player name:")

if player_name:
    matches = players_df[players_df["PLAYER_NAME"].str.contains(player_name, case=False, na=False)]
    if matches.empty:
        st.warning("No matching player found.")
    else:
        selected = st.selectbox("Select player:", matches["PLAYER_NAME"].unique())
        player_id = int(matches.loc[matches["PLAYER_NAME"] == selected, "PLAYER_ID"].values[0])

        st.write(f"### Monte Carlo probabilities for {selected}")

        for stat in STATS:
            samples = load_player_samples(player_id, stat)
            if samples is None:
                st.write(f"No data for {stat}")
                continue

            ks, probs = compute_probabilities(samples)

            fig = px.bar(x=ks, y=probs, labels={"x": f"{stat} ≥ k", "y": "Probability"},
                         title=f"{stat} Probability Distribution")
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
