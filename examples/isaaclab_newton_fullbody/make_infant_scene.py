"""Create a self-contained MIMo infant scene without benchmark props."""

import argparse
from itertools import combinations
from pathlib import Path
import re

import mujoco


def explicit_solref_defaults(xml):
    def replace(match):
        values = match.group(1).split()
        if len(values) == 1:
            return f'solref="{values[0]} 1"'
        return match.group(0)

    return re.sub(r'solref="([^"]+)"', replace, xml)


def explicit_geom_gap_defaults(xml):
    def replace(match):
        attributes = match.group(1)
        if re.search(r'\bgap\s*=', attributes):
            return match.group(0)
        return f'<geom gap="0"{attributes}>'

    return re.sub(r'<geom\b([^>]*)>', replace, xml)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-body-pair", nargs=2, action="append", default=[],
                        metavar=("BODY1", "BODY2"))
    parser.add_argument("--exclude-interfinger-contacts", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output exists: {args.output}")
    specification = mujoco.MjSpec.from_file(str(args.source.resolve()))
    specification.texturedir = str(args.source.resolve().parent)
    for name in ("test_object1", "test_object2"):
        body = specification.body(name)
        if body is None:
            raise ValueError(f"Expected benchmark prop {name} was not found")
        specification.delete(body)
    for first, second in args.exclude_body_pair:
        if specification.body(first) is None or specification.body(second) is None:
            raise ValueError(f"Unknown MIMo body in excluded contact pair: {first}, {second}")
        specification.add_exclude(bodyname1=first, bodyname2=second)
    if args.exclude_interfinger_contacts:
        for side in ("right", "left"):
            fingers = {}
            for family in ("ff", "mf", "rf", "lf", "th"):
                fingers[family] = []
            for body in specification.bodies:
                for family in fingers:
                    if body.name.startswith(f"{side}_{family}"):
                        fingers[family].append(body.name)
            if not all(fingers.values()):
                raise ValueError(f"Missing digit bodies on {side} hand")
            for first_family, second_family in combinations(fingers, 2):
                for first in fingers[first_family]:
                    for second in fingers[second_family]:
                        specification.add_exclude(bodyname1=first, bodyname2=second)
    model = specification.compile()
    if model.ntendon != 9 or model.nu != 90 or model.nq != 100:
        raise ValueError("Unexpected MIMo infant-only model topology")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(explicit_geom_gap_defaults(
        explicit_solref_defaults(specification.to_xml())))
    print(f"Wrote infant-only scene: {args.output} (nq={model.nq}, nu={model.nu})")


if __name__ == "__main__":
    main()
