"""Rewrite multi-anchor MJCF bodies as equivalent single-joint body chains."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def split_joint_anchors(tree, mass=1e-6):
    if mass <= 0:
        raise ValueError("Intermediate body mass must be positive")
    option = tree.find("option")
    if option is not None:
        option.set("jacobian", "sparse")
    changed = []
    parents = {child: parent for parent in tree.iter() for child in parent}
    contact = tree.find("contact")
    for body in list(tree.findall(".//body")):
        joints = body.findall("joint")
        if len(joints) < 2:
            continue
        anchors = [np.fromstring(joint.get("pos", "0 0 0"), sep=" ")
                   for joint in joints]
        if all(np.allclose(anchor, anchors[0], rtol=0, atol=1e-9)
               for anchor in anchors[1:]):
            continue
        name = body.get("name")
        if not name:
            raise ValueError("A multi-anchor body requires a name")
        parent_name = parents[body].get("name")
        if not parent_name:
            raise ValueError(f"Cannot preserve parent collision exclusion for {name}")
        rest = [element for element in body if element not in joints]
        for element in list(body):
            body.remove(element)
        body.set("name", f"{name}_anchor0")
        parent = body
        for index, joint in enumerate(joints):
            if index:
                parent = ET.SubElement(parent, "body", {
                    "name": name if index == len(joints) - 1 else f"{name}_anchor{index}",
                    "pos": "0 0 0"})
            parent.append(joint)
            if index < len(joints) - 1:
                ET.SubElement(parent, "inertial", {
                    "pos": "0 0 0", "mass": f"{mass:.9g}",
                    "diaginertia": f"{mass * 1e-3:.9g} {mass * 1e-3:.9g} {mass * 1e-3:.9g}"})
        for element in rest:
            parent.append(element)
        if contact is None:
            contact = ET.SubElement(tree.getroot(), "contact")
        ET.SubElement(contact, "exclude", {"body1": parent_name, "body2": name})
        changed.append(name)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    tree = ET.parse(args.scene)
    changed = split_joint_anchors(tree)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="unicode")
    print("Split multi-anchor bodies:", ", ".join(changed))


if __name__ == "__main__":
    main()
