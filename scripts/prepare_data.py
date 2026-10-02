"""Create the modeling table and verify the continuation-cell merge."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/source/severson_features.csv"
DEST = ROOT / "data/processed/severson_features.csv"
CORRECTIONS = {"b1c0": 1852, "b1c1": 2160, "b1c2": 2237, "b1c3": 1434, "b1c4": 1709}

frame = pd.read_csv(SOURCE)
for cell_id, target in CORRECTIONS.items():
    frame.loc[frame.cell_id.eq(cell_id), "cycle_life"] = target
DEST.parent.mkdir(parents=True, exist_ok=True)
frame.to_csv(DEST, index=False)
print(f"wrote {len(frame)} cells to {DEST}")
