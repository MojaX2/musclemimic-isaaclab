"""Render matched recorded infant crawl states without reintegrating dynamics."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--before-report", type=Path, required=True)
    parser.add_argument("--after-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    recordings = [np.load(path) for path in (args.before, args.after)]
    reports = [json.loads(path.read_text()) for path in
               (args.before_report, args.after_report)]
    if any(args.world >= recording["qpos"].shape[1] or args.world < 0
           for recording in recordings):
        parser.error("World index is outside a recording")
    if not np.array_equal(recordings[0]["initial_qpos"],
                          recordings[1]["initial_qpos"]):
        raise ValueError("Compared rollouts must share initial states")
    if not np.isclose(recordings[0]["dt"], recordings[1]["dt"]):
        raise ValueError("Compared rollouts must share control intervals")
    control_interval = float(recordings[0]["dt"])
    duration = min(recording["qpos"].shape[0] for recording in recordings) * control_interval
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    model.vis.global_.offwidth = 640
    model.vis.global_.offheight = 480
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance = 1.35
    camera.azimuth = 120
    camera.elevation = -18
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    option.tendongroup[:] = 0
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 19)
    except OSError:
        font = ImageFont.load_default()
    labels = ("Before: contact-based policy", "After: reach-trained policy")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Set FFMPEG_BINARY to an ffmpeg executable")
    writer = subprocess.Popen([
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x560",
        "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output)], stdin=subprocess.PIPE)
    frames = round(duration * args.fps)
    try:
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            for frame in range(frames):
                elapsed = frame / args.fps
                index = min(round(elapsed / control_interval),
                            recordings[0]["qpos"].shape[0] - 1,
                            recordings[1]["qpos"].shape[0] - 1)
                canvas = Image.new("RGB", (1280, 560), "#17212c")
                draw = ImageDraw.Draw(canvas)
                for column, (recording, report, label) in enumerate(
                        zip(recordings, reports, labels)):
                    data.qpos[:] = recording["qpos"][index, args.world]
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), 0.22]
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    canvas.paste(Image.fromarray(renderer.render()), (column * 640, 40))
                    draw.text((column * 640 + 12, 10), label, font=font, fill="white")
                    forward = report["per_world"]["forward_displacement_m"][args.world]
                    support = report["per_world"]["minimum_support_fraction"][args.world]
                    footer = f"16 s: forward {forward:+.3f} m | supported {support:.0%}"
                    draw.text((column * 640 + 12, 526), footer, font=font,
                              fill="#bde9f7")
                draw.text((497, 496), f"t = {elapsed:04.1f} s", font=font,
                          fill="white")
                writer.stdin.write(np.asarray(canvas).tobytes())
                if frame == 0:
                    canvas.save(output.with_suffix(".preview.png"))
    finally:
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError("Video encoder failed")
    print(f"Rendered {frames} frames ({duration:.1f} s) to {output}")


if __name__ == "__main__":
    main()
