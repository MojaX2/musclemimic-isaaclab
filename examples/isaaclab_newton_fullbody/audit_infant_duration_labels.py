"""Compare infant experiment filename durations against physics metadata."""

import argparse
import json
from pathlib import Path

import mujoco


def actual_duration(report, scene_timesteps):
    if "duration_s" in report:
        return float(report["duration_s"]), "reported"
    if all(name in report for name in ("control_steps", "physics_per_control", "scene")):
        scene = Path(report["scene"])
        if not scene.exists():
            return None, "scene_missing"
        if scene not in scene_timesteps:
            scene_timesteps[scene] = float(mujoco.MjModel.from_xml_path(str(scene)).opt.timestep)
        return (report["control_steps"] * report["physics_per_control"] *
                scene_timesteps[scene]), "computed"
    return None, "metadata_missing"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    scenes = {}
    checked = []
    mismatches = []
    unresolved = []
    for path in sorted(args.artifacts.glob("*16s*.json")):
        report = json.loads(path.read_text())
        duration, source = actual_duration(report, scenes)
        row = {"artifact": path.name, "actual_duration_s": duration,
               "evidence": source}
        if duration is None:
            unresolved.append(row)
        elif abs(duration - 16) > 1e-6:
            mismatches.append(row)
        else:
            checked.append(row)
    result = {"filename_label_s": 16, "checked_matching": len(checked),
              "mismatch_count": len(mismatches), "unresolved_count": len(unresolved),
              "mismatches": mismatches, "unresolved": unresolved,
              "scope": "Filename timing audit only; metrics and learning claims require separate validation"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: result[name] for name in
                      ("checked_matching", "mismatch_count", "unresolved_count")}))


if __name__ == "__main__":
    main()
