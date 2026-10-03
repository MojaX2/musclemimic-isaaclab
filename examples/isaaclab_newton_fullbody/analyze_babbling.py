"""Reanalyze a saved motor-babbling trajectory without rerunning physics."""

import argparse
import json
from pathlib import Path

import numpy as np

from run_motor_babbling import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    with np.load(args.trajectory) as data:
        arrays = {key: data[key] for key in ("current", "action", "tactile", "target")}
    records = [{key: value[step] for key, value in arrays.items()}
               for step in range(len(arrays["current"]))]
    result = analyze(records)
    result["trajectory"] = str(args.trajectory.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
