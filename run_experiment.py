#!/usr/bin/env python3
"""Entry point: reproduce every number, table and figure in the paper.

    python run_experiment.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dga_detect.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
