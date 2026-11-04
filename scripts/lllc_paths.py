"""Where the scripts look for trained runs and write working files.

LLLC_RUNS_ROOT  run folders <root>/<dataset>/<run_id>/<model>/hs<width>/ (default: picture/ic_gen in this repository,
                where training writes and where the run archives unpack)
LLLC_WORK_DIR   per-run JSON files and caches written by the evaluation scripts (default: work/ in this repository)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS_ROOT = os.environ.get("LLLC_RUNS_ROOT", str(REPO / "picture" / "ic_gen"))
WORK_DIR = os.environ.get("LLLC_WORK_DIR", str(REPO / "work"))
