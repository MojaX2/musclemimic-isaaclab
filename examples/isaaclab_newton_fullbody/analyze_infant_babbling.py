"""Reanalyze saved infant trajectories with motor-command permutation controls."""

import argparse
import json
from pathlib import Path

import numpy as np

from run_infant_babbling import shuffled_action_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    records = []
    for result_path in args.results:
        original = json.loads(result_path.read_text())
        with np.load(original["trajectory"]) as trajectory:
            per_step = [{key: trajectory[key][step] for key in ("current", "action", "target")}
                        for step in range(len(trajectory["current"]))]
        original["mse_state_shuffled_action"] = shuffled_action_error(per_step)
        record = {key: original[key] for key in (
            "seed", "samples", "posture", "mse_state_only",
            "mse_state_action", "mse_state_shuffled_action", "mse_state_action_touch",
            "mse_state_action_shuffled_touch", "solver_limit_physics_steps")}
        record["action_smoothing"] = original.get("action_smoothing", 0.85)
        records.append(record)
    report = {
        "runs": records,
        "motor_command_beats_state_only": sum(
            item["mse_state_action"] < item["mse_state_only"] for item in records),
        "motor_command_beats_shuffled_command": sum(
            item["mse_state_action"] < item["mse_state_shuffled_action"] for item in records),
        "touch_beats_shuffled_touch": sum(
            item["mse_state_action_touch"] < item["mse_state_action_shuffled_touch"]
            for item in records),
        "touch_beats_motor_command_only": sum(
            item["mse_state_action_touch"] < item["mse_state_action"] for item in records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
