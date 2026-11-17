"""Predict the final submission, or reproduce training on external DFL matches."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
HF_TREE = "https://huggingface.co/api/datasets/pysport/idsse-data/tree/main"
HF_FILE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/{}"


def download():
    destination = ROOT / "data" / "external" / "dfl"
    destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(HF_TREE) as response:
        names = [item["path"] for item in json.load(response)
                 if item["path"].endswith(".xml")]
    if not names:
        raise RuntimeError("The DFL dataset listing contains no XML files")
    for name in names:
        if Path(name).name != name:
            raise ValueError("Unexpected nested dataset path: " + name)
        target = destination / name
        if target.exists():
            continue
        print("downloading", name, flush=True)
        temporary = target.with_suffix(".xml.part")
        urllib.request.urlretrieve(HF_FILE.format(name), temporary)
        temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", nargs="?", default="predict",
                        choices=("download", "train", "predict", "all"))
    args = parser.parse_args()
    # Bound CPU parallelism for repeatable, practical inference on laptops and servers.
    os.environ.setdefault("OMP_NUM_THREADS", "12")
    os.environ.setdefault("MKL_NUM_THREADS", "12")
    if args.step in ("download", "all"):
        download()
    if args.step in ("train", "all"):
        subprocess.run([sys.executable, "-m", "pipeline.train"], cwd=ROOT, check=True)
    if args.step in ("predict", "all"):
        subprocess.run([sys.executable, "-m", "pipeline.predict", "--name", "CuleCode"],
                       cwd=ROOT, check=True)
        subprocess.run([sys.executable, "evaluation/validate_submission.py",
                        "submissions/CuleCode.csv"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
