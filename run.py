"""Regenerate the submission from scratch.

    pip install -r requirements-pipeline.txt
    python run.py download      # the open DFL training games (other matches, CC-BY 4.0)
    python run.py train         # models/possession/ from the DFL games
    python run.py predict       # submissions/CuleCode.csv for BAR-ATH
    python run.py all           # all three

Metrica's challenge repo must be cloned into challenge/ (loaders, scorer, validator), and the
provided BAR-ATH data is expected under data/tactical/ (or set MTAH_DATA_DIR).
"""

import os
import subprocess
import sys
import urllib.request
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
HF_TREE = "https://huggingface.co/api/datasets/pysport/idsse-data/tree/main"
HF_FILE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/{}"
DFL_DIR = os.path.join(ROOT, "data", "external", "dfl")


def download():
    os.makedirs(DFL_DIR, exist_ok=True)
    with urllib.request.urlopen(HF_TREE) as response:
        files = [f["path"] for f in json.load(response) if f["path"].endswith(".xml")]
    for name in files:
        target = os.path.join(DFL_DIR, name)
        if os.path.exists(target):
            continue
        print("downloading", name, flush=True)
        urllib.request.urlretrieve(HF_FILE.format(name), target)


def module(name, *args):
    subprocess.run([sys.executable, "-W", "ignore", "-m", name, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("download", "all"):
        download()
    if step in ("train", "all"):
        module("pipeline.train")
    if step in ("predict", "all"):
        module("pipeline.predict")
        subprocess.run([sys.executable, os.path.join("challenge", "evaluation", "validate_submission.py"),
                        os.path.join("submissions", "CuleCode.csv")], cwd=ROOT, check=True)
