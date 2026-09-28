"""python hdfcbank.py  ->  calibrated next-60-minute range for HDFCBANK (thin wrapper)."""
import sys

from engine.cli import run

if __name__ == "__main__":
    sys.exit(run("HDFCBANK"))
