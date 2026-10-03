"""Build a portable infant scene and source bundle with validated assets."""

import argparse
from pathlib import Path
import re
import shutil
import tarfile
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import mujoco

from make_infant_scene import explicit_geom_gap_defaults, explicit_solref_defaults
from split_infant_joint_anchors import split_joint_anchors


def make_portable_scene(source, destination):
    xml = source.read_text()
    portable, replacements = re.subn(r'texturedir="[^"]+"', 'texturedir="."', xml,
                                      count=1)
    if replacements != 1:
        raise ValueError("Infant scene must have exactly one texture directory")
    if re.search(r'(?:texturedir|meshdir)="/', portable):
        raise ValueError("Infant scene still contains an absolute asset path")
    destination.write_text(explicit_geom_gap_defaults(explicit_solref_defaults(portable)))
    mujoco.MjModel.from_xml_path(str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--mimo-assets", type=Path, required=True)
    parser.add_argument("--mimo-license", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Bundle already exists")
    names = [artifact.name for artifact in args.artifact]
    if len(names) != len(set(names)):
        parser.error("Artifact basenames must be unique")
    with TemporaryDirectory() as directory:
        root = Path(directory) / "infant_virtual_human"
        model_dir = root / "model"
        source_dir = root / "source"
        artifact_dir = root / "artifacts"
        model_dir.mkdir(parents=True)
        source_dir.mkdir()
        artifact_dir.mkdir()
        shutil.copytree(args.mimo_assets / "tex", model_dir / "tex")
        shutil.copy2(args.mimo_license, model_dir / "LICENSE-MIMo")
        make_portable_scene(args.scene, model_dir / "infant.xml")
        anchor_tree = ET.parse(model_dir / "infant.xml")
        split_joint_anchors(anchor_tree)
        anchor_tree.write(model_dir / "infant_anchor_preserving.xml", encoding="unicode")
        mujoco.MjModel.from_xml_path(str(model_dir / "infant_anchor_preserving.xml"))
        code_dir = source_dir / "examples" / "isaaclab_newton_fullbody"
        code_dir.mkdir(parents=True)
        for code in sorted((args.source_root / "examples" / "isaaclab_newton_fullbody").glob("*.py")):
            shutil.copy2(code, code_dir / code.name)
        (source_dir / "docs").mkdir()
        (source_dir / "scripts").mkdir()
        shutil.copy2(args.source_root / "docs" / "DEVELOPMENTAL_HUMAN.md",
                     source_dir / "docs" / "DEVELOPMENTAL_HUMAN.md")
        shutil.copy2(args.source_root / "scripts" / "run.py", source_dir / "scripts" / "run.py")
        shutil.copy2(args.source_root / "LICENSE", source_dir / "LICENSE-MuscleMimic")
        for artifact in args.artifact:
            shutil.copy2(artifact, artifact_dir / artifact.name)
        shutil.copy2(args.report, root / "PROGRESS.md")
        (root / "README.md").write_text(
            "# Infant virtual human reproduction bundle\n\n"
            "Requires an Isaac Lab 3.0.0-compatible Python environment with MuJoCo, "
            "MuJoCo Warp, Warp, PyTorch, NumPy and SciPy, plus an NVIDIA GPU. "
            "The experiments used Isaac Lab v3.0.0-EA, not verified GA 3.0.0.\n\n"
            "Run from this directory, substituting your Isaac Lab Python executable:\n\n"
            "```bash\n"
            "python source/examples/isaaclab_newton_fullbody/check_infant_isaaclab.py "
            "--scene model/infant.xml --output newton_infant_smoke.json "
            "--worlds 4 --steps 200 --muscle-drive --audit-model\n"
            "python source/examples/isaaclab_newton_fullbody/check_infant_isaaclab.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--output newton_crawl_pose.json --worlds 4 --steps 1000 --muscle-drive\n"
            "python source/examples/isaaclab_newton_fullbody/check_infant_isaaclab.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_stand_pose.npz "
            "--output newton_stand_pose.json --worlds 4 --steps 1000 --muscle-drive\n"
            "python source/examples/isaaclab_newton_fullbody/evaluate_infant_crawl_ppo.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_palm_group_ppo_touch_2s_seed37.pt "
            "--output crawl_support_eval.json --worlds 8 --control-steps 40 --seed 41\n"
            "python source/examples/isaaclab_newton_fullbody/evaluate_infant_crawl_cpg.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_palm_forward_ppo_touch_seed53.pt "
            "--baseline-oscillator artifacts/infant_crawl_cpg_forward_seed71.npz "
            "--oscillator artifacts/infant_crawl_cpg_wrist_seed73.npz "
            "--output crawl_cpg_eval.json --worlds 8 --seed 79\n"
            "python source/examples/isaaclab_newton_fullbody/evaluate_infant_crawl_cpg.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_palm_forward_ppo_touch_seed53.pt "
            "--baseline-oscillator artifacts/infant_crawl_cpg_wrist_seed73.npz "
            "--oscillator artifacts/infant_crawl_cpg_gait_objective_seed97.npz "
            "--output crawl_gait_2s_eval.json --worlds 8 --seed 107\n"
            "python source/examples/isaaclab_newton_fullbody/evaluate_infant_crawl_cpg.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_palm_forward_ppo_touch_seed53.pt "
            "--baseline-oscillator artifacts/infant_crawl_cpg_gait_objective_seed97.npz "
            "--oscillator artifacts/infant_crawl_cpg_gait_4s_seed137.npz "
            "--output crawl_gait_4s_eval.json --worlds 8 --control-steps 80 --seed 149\n"
            "python source/examples/isaaclab_newton_fullbody/evaluate_infant_crawl_ppo.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_cpg_net_progress_4s_seed281.pt "
            "--output phase_ppo_eval.json --worlds 8 --control-steps 80 --seed 337\n"
            "python source/examples/isaaclab_newton_fullbody/evaluate_infant_crawl_ppo.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_cpg_fixed_limbs_4s_seed311.pt "
            "--output fixed_limbs_eval.json --worlds 8 --control-steps 80 --seed 337\n"
            "python source/examples/isaaclab_newton_fullbody/evaluate_infant_crawl_ppo.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_tactile_transfer_normalized_pretrained_seed379.pt "
            "--output tactile_transfer_eval.json --worlds 8 --control-steps 40 --seed 383\n"
            "python source/examples/isaaclab_newton_fullbody/evaluate_infant_crawl_ppo.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_tactile_crawl_aligned_matched_seed431.pt "
            "--output crawl_aligned_transfer_eval.json --worlds 8 --control-steps 40 --seed 439\n"
            "python source/examples/isaaclab_newton_fullbody/evaluate_infant_crawl_ppo.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_cpg_antagonistic_tactile_learned_seed521.pt "
            "--output antagonistic_cpg_eval.json --worlds 8 --control-steps 80 --seed 523\n"
            "python source/examples/isaaclab_newton_fullbody/evaluate_infant_crawl_cpg.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_palm_forward_ppo_touch_seed53.pt "
            "--baseline-oscillator artifacts/infant_crawl_cpg_gait_4s_seed137.npz "
            "--oscillator artifacts/infant_crawl_cpg_shoulder_lift_4s_seed563.npz "
            "--output shoulder_lift_cpg_eval.json --worlds 8 --control-steps 80 --seed 571\n"
            "python source/examples/isaaclab_newton_fullbody/evaluate_infant_crawl_cpg.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_palm_forward_ppo_touch_seed53.pt "
            "--baseline-oscillator artifacts/infant_crawl_cpg_gait_4s_seed137.npz "
            "--oscillator artifacts/infant_crawl_cpg_coupled_sustained_8s_seed631.npz "
            "--output sustained_cpg_eval.json --worlds 8 --control-steps 160 --seed 641\n"
            "python source/examples/isaaclab_newton_fullbody/evaluate_infant_crawl_cpg.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_palm_forward_ppo_touch_seed53.pt "
            "--baseline-oscillator artifacts/infant_crawl_cpg_gait_4s_seed137.npz "
            "--oscillator artifacts/infant_crawl_cpg_shoulder_lift_4s_seed563.npz "
            "--contact-guard --output contact_guard_eval.json "
            "--worlds 8 --control-steps 160 --seed 659\n"
            "python source/examples/isaaclab_newton_fullbody/sweep_infant_crawl_shoulder_lift.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_palm_forward_ppo_touch_seed53.pt "
            "--oscillator artifacts/infant_crawl_cpg_gait_4s_seed137.npz "
            "--phase-offset-rad 0 --amplitudes 0 0.3 "
            "--output phase_lift_eval.json --worlds 8 --control-steps 80 --seed 709\n"
            "python source/examples/isaaclab_newton_fullbody/sweep_infant_crawl_stepper.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_palm_forward_ppo_touch_seed53.pt "
            "--parameters-file artifacts/infant_crawl_tactile_stepper_ga_4s_seed821.npz "
            "--absolute-targets --forward-reach --strong-plant --leg-push "
            "--output tactile_stepper_eval.json --worlds 8 --control-steps 80 --seed 827\n"
            "python source/examples/isaaclab_newton_fullbody/sweep_infant_crawl_stepper.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_palm_forward_ppo_touch_seed53.pt "
            "--parameters-file artifacts/infant_crawl_tactile_stepper_ga_8s_seed911.npz "
            "--absolute-targets --forward-reach --strong-plant --leg-push "
            "--output tactile_stepper_8s_eval.json --worlds 8 --control-steps 160 --seed 929\n"
            "python source/examples/isaaclab_newton_fullbody/train_infant_crawl_stepper_ppo.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--support-checkpoint artifacts/infant_crawl_palm_forward_ppo_touch_seed53.pt "
            "--stepper-parameters artifacts/infant_crawl_tactile_stepper_ga_8s_seed911.npz "
            "--output stepper_ppo_anchor.json --updates 20 --worlds 16 "
            "--control-steps 160 --policy-anchor 1.0 --seed 1031\n"
            "python source/examples/isaaclab_newton_fullbody/sweep_infant_crawl_stepper.py "
            "--scene model/infant.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_stepper_ppo_anchor1_8s_seed1031.pt "
            "--parameters-file artifacts/infant_crawl_tactile_stepper_ga_8s_seed911.npz "
            "--absolute-targets --forward-reach --strong-plant --leg-push "
            "--stance-loss-grace-steps 2 "
            "--output stepper_ppo_anchor_eval.json --worlds 8 --control-steps 160 --seed 1073\n"
            "```\n\n"
            "The six saved infant babbling trajectories in artifacts/ reproduce the "
            "nine-region action/no-action/shuffled-action comparison. For example:\n\n"
            "```bash\n"
            "python source/examples/isaaclab_newton_fullbody/train_infant_tactile_model.py "
            "--trajectories artifacts/infant_supine_independent_300_seed7.npz "
            "artifacts/infant_supine_independent_300_seed11.npz "
            "artifacts/infant_supine_independent_300_seed23.npz "
            "artifacts/infant_prone_independent_300_seed31.npz "
            "artifacts/infant_prone_independent_300_seed37.npz "
            "--heldout artifacts/infant_prone_independent_300_seed37.npz "
            "--regions trunk right_upper_arm left_upper_arm right_forearm "
            "left_forearm right_hand left_hand right_foot left_foot "
            "--device cuda:0 --output tactile_reproduction.json\n"
            "```\n\n"
            "Compare the same saved tactile policy in direct Warp and Newton. "
            "The anchor-preserving scene adds lightweight joint bodies and "
            "explicit parent-child collision exclusions:\n\n"
            "```bash\n"
            "python source/examples/isaaclab_newton_fullbody/evaluate_infant_newton_crawl.py "
            "--scene model/infant_anchor_preserving.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_palm_forward_ppo_touch_seed53.pt "
            "--backend newton --worlds 8 --control-steps 160 --seed 1049 "
            "--output newton_transfer.json\n"
            "```\n\n"
            "The contact-fidelity correction and held-out results are in "
            "artifacts/NEWTON_FIDELITY_VALIDATION.md. Older Newton transfer "
            "reports describe the pre-correction solver settings.\n\n"
            "The masked stepper PPO follow-up is in "
            "artifacts/NEWTON_STEPPER_PPO_CREDIT_ASSIGNMENT.md; it did not "
            "achieve robust crawling.\n\n"
            "The stage-observation follow-up is in "
            "artifacts/NEWTON_STEPPER_PHASE_OBSERVATION.md. Its checkpoint "
            "requires --stepper-parameters during evaluation and still does "
            "not achieve robust crawling.\n\n"
            "The standing reset and actuation audit is in "
            "artifacts/INFANT_STAND_RESET_AND_ACTUATION_AUDIT.md; "
            "standing and walking remain unlearned.\n\n"
            "The bilateral foot-contact reset and standing PPO follow-up is in "
            "artifacts/INFANT_STAND_BILATERAL_CONTACT_CURRICULUM.md; corrected "
            "initial contact still did not yield sustained standing.\n\n"
            "The excitation and pressure-center reflex audit is in "
            "artifacts/INFANT_STAND_FALL_AND_COP_REFLEX.md. It did not "
            "produce sustained standing or walking.\n\n"
            "The evolved pressure-center reflex and four-second holdout are in "
            "artifacts/INFANT_STAND_EVOLVED_COP_REFLEX.md; neither verified "
            "sustained standing nor walking.\n\n"
            "The Newton babbling-feature transfer and shared-trajectory tactile "
            "prediction audit are in artifacts/NEWTON_BABBLING_FEATURE_TRANSFER.md. "
            "The frozen features caused negative crawl-control transfer.\n\n"
            "The Newton on-policy tactile-adaptation follow-up is in "
            "artifacts/NEWTON_ONPOLICY_TACTILE_ADAPTATION.md. It improves "
            "contact prediction and late support, not robust crawling.\n\n"
            "The babbling warm-start and nine-region predicted-contact control "
            "audit are in artifacts/NEWTON_BABBLING_WARMSTART_CONTACT9.md. "
            "An isolated crawl criterion did not reproduce or persist.\n\n"
            "The sixteen-second PPO continuation and support/forward trade-off "
            "are in artifacts/NEWTON_SUSTAINED_CRAWL_CURRICULUM.md. "
            "No held-out policy achieved robust crawling.\n\n"
            "The direct Warp versus Isaac Lab/Newton standing-pressure audit "
            "is in artifacts/NEWTON_STAND_PRESSURE_TRANSFER.md. Both backends "
            "fail sustained four-second standing.\n\n"
            "The four-gain hip/ankle pressure-reflex evolution is in "
            "artifacts/INFANT_STAND_HIP_ANKLE_EVOLUTION.md. It improves "
            "early standing but not late support or walking.\n\n"
            "The six-gain lateral pressure reflex and fall-timing audit are in "
            "artifacts/INFANT_STAND_LATERAL_REFLEX.md. Lateral support improves, "
            "but trunk collapse and walking remain unresolved.\n\n"
            "The fixed-reflex muscle-excitation range comparison is in "
            "artifacts/INFANT_STAND_REFLEX_EXCITATION_RANGE.md. Wider excitation "
            "did not solve the four-second standing failure.\n\n"
            "The trunk-tilt reflex evolved above six-gain foot feedback is in "
            "artifacts/INFANT_STAND_TRUNK_REFLEX_TRANSFER.md. Its single-seed "
            "validation gain did not transfer to held-out standing.\n\n"
            "To reproduce the bilateral standing reset comparison:\n\n"
            "```bash\n"
            "python source/examples/isaaclab_newton_fullbody/evaluate_infant_stand_reset_projection.py "
            "--scene model/infant.xml --pose artifacts/infant_stand_pose.npz "
            "--bias artifacts/infant_stand_cem_validated_seed7.npz "
            "--seeds 53 59 61 --duration-s 2 --bilateral "
            "--output bilateral_stand_eval.json\n"
            "```\n\n"
            "Re-evaluate its held-out phase-aware policy with:\n\n"
            "```bash\n"
            "python source/examples/isaaclab_newton_fullbody/evaluate_infant_newton_crawl.py "
            "--scene model/infant_anchor_preserving.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--checkpoint artifacts/infant_crawl_newton_phase_stepper_ppo_20x160_seed1297.pt "
            "--stepper-parameters artifacts/infant_crawl_tactile_stepper_ga_8s_seed911.npz "
            "--worlds 8 --control-steps 160 --seed 1193 "
            "--output newton_phase_stepper_eval.json\n"
            "```\n\n"
            "Newton support adaptation can be reproduced with:\n\n"
            "```bash\n"
            "python source/examples/isaaclab_newton_fullbody/train_infant_newton_crawl_ppo.py "
            "--scene model/infant_anchor_preserving.xml "
            "--pose artifacts/infant_crawl_refined_palm_shin_pose.npz "
            "--initialize-from artifacts/infant_crawl_palm_forward_ppo_touch_seed53.pt "
            "--updates 20 --worlds 8 --control-steps 40 --seed 1171 "
            "--output newton_support_train.json\n"
            "```\n\n"
            "See PROGRESS.md and source/docs/DEVELOPMENTAL_HUMAN.md for limitations. "
            "The skin is rigid-contact sensing, not deformable skin or taxels. "
            "Stable crawling, body-schema acquisition and walking are not verified.\n"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.output, "w:gz") as archive:
            archive.add(root, arcname=root.name)
    print(args.output)


if __name__ == "__main__":
    main()
