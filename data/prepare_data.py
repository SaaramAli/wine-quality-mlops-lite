"""
Pulls the UCI Wine Quality (red) dataset and splits it into:
- reference.csv   : training/reference distribution
- new_batch.csv   : simulated incoming data (clean)
- bad_batch.csv   : new_batch with injected data-quality issues, for testing validation
"""
import pandas as pd
import numpy as np

RNG = np.random.default_rng(42)
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

df = pd.read_csv(URL, sep=";")

# Binary target: "good" wine if quality >= 6 (roughly balanced classes)
df["quality_label"] = (df["quality"] >= 6).astype(int)

# Shuffle, then split 80/20
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(len(df) * 0.8)
reference = df.iloc[:split_idx].reset_index(drop=True)
new_batch = df.iloc[split_idx:].reset_index(drop=True)

reference.to_csv("data/reference.csv", index=False)
new_batch.to_csv("data/new_batch.csv", index=False)

# Build a deliberately broken copy of new_batch
bad_batch = new_batch.copy()
bad_rows_ph = RNG.choice(bad_batch.index, size=5, replace=False)
bad_batch.loc[bad_rows_ph, "pH"] = -1.0  # invalid: pH can't be negative

bad_rows_alcohol = RNG.choice(bad_batch.index, size=5, replace=False)
bad_batch.loc[bad_rows_alcohol, "alcohol"] = np.nan  # missing values

bad_rows_quality = RNG.choice(bad_batch.index, size=3, replace=False)
bad_batch.loc[bad_rows_quality, "quality"] = 20  # invalid: scale is 0-10

bad_batch.to_csv("data/bad_batch.csv", index=False)

print(f"reference.csv : {len(reference)} rows, {reference['quality_label'].mean():.1%} good wine")
print(f"new_batch.csv : {len(new_batch)} rows, {new_batch['quality_label'].mean():.1%} good wine")
print(f"bad_batch.csv : {len(bad_batch)} rows (5 bad pH, 5 missing alcohol, 3 invalid quality)")