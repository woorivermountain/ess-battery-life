import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import run
from src.visualize import generate_figures

if __name__ == "__main__":
    print(run(repeats=10, bootstrap=5000))
    generate_figures()
