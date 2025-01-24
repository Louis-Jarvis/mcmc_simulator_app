"""
Generate synthetic data for this problem.
"""
import pathlib

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DATA_FOLDER = pathlib.Path("data")
ASSETS_FOLDER = pathlib.Path("assets")

RANDOM_SEED = 42
NUM_POINTS = 100
slope = 2
offset = 50
sigma = 20

np.random.seed(RANDOM_SEED)

# Set the theme for seaborn
sns.set_theme(style="darkgrid")

def main():
    x = np.arange(0, NUM_POINTS, 0.1)   
    y = slope*x + offset + np.random.normal(0, sigma, len(x))

    data = pl.DataFrame({
        "x": x,
        "y": y
    })

    data.write_csv(DATA_FOLDER / "synthetic_data.csv")
    sns.scatterplot(data=data, x="x", y="y")
    plt.savefig(ASSETS_FOLDER / "synthetic_data.png")


if __name__ == "__main__":
    main()
