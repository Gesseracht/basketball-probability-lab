import numpy as np
import pandas as pd
import h5py
from concurrent.futures import ProcessPoolExecutor
import sys
from pathlib import Path
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
parent = Path(__file__).resolve().parent.parent
base_dir = os.path.join(parent, r"data/simulations")


def run_monte_carlo(file_path, player_ids, means, stds, n_samples=100_000):
    """Run one Monte Carlo simulation per mean/std pair and save to HDF5."""
    n_sim = len(means)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with h5py.File(file_path, "w") as f:
        dset = f.create_dataset(
            "samples",
            shape=(n_sim, n_samples),
            dtype=np.float16,
            compression="gzip",
            compression_opts=4,
            chunks=(1, 10_000),
        )

        f.create_dataset(
            "PLAYER_ID",
            data=player_ids.astype(np.int64),
            dtype=np.int64,
            compression="gzip",
            compression_opts=1
        )

        for i in range(n_sim):
            mu, sigma = means[i], stds[i]
            sigma = max(sigma, 1e-6)
            samples = np.random.normal(mu, sigma, n_samples).astype(np.float16)
            dset[i, :] = samples

        # métadonnées
        f.attrs["n_sim"] = n_sim
        f.attrs["n_samples"] = n_samples
        print(f"{file_path} terminé ({n_sim} joueurs).")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Important for Windows

    csv_path = os.path.join(parent, r'data\processed\params.csv')
    df = pd.read_csv(csv_path)

    stats = ["REB", "AST", "TOV", "STL", "BLK", "PTS"]
    N_SAMPLES = 100_000

    jobs = []
    for stat in stats:
        means = df[f"{stat}_adap_mean"].to_numpy(dtype=np.float16)
        stds = df[f"{stat}_adap_std"].to_numpy(dtype=np.float16)
        player_ids = df["PLAYER_ID"].to_numpy(dtype=np.int64)
        file_path = os.path.join(base_dir, f"{stat}.h5")
        jobs.append((file_path, player_ids, means, stds))

    with ProcessPoolExecutor(max_workers=6) as ex:
        futures = [
            ex.submit(run_monte_carlo, path, pid, mu, sd, N_SAMPLES)
            for path, pid, mu, sd in jobs
        ]
        for fut in futures:
            fut.result()

