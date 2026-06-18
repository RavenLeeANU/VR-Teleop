"""Simple MuJoCo model previewer.

By default this opens the X5 UMI custom XML in MuJoCo's native viewer.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "examples"
    / "video"
    / "assets"
    / "x5"
    / "x5_umi_custom.xml"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview a MuJoCo XML scene.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to MuJoCo XML model. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--keyframe",
        default=None,
        help="Optional MuJoCo keyframe name to reset to before opening the viewer.",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="Open the viewer without stepping physics.",
    )
    return parser.parse_args()


def _configure_camera(viewer: object, model: object) -> None:
    """Set a stable initial camera view around the loaded model."""
    extent = float(getattr(model.stat, "extent", 1.0)) or 1.0
    center = getattr(model.stat, "center", None)

    cam = viewer.cam
    if center is not None:
        cam.lookat[:] = center
    cam.distance = max(0.8, extent * 2.5)
    cam.azimuth = 135.0
    cam.elevation = -25.0


def main() -> int:
    args = _parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {model_path}")

    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is required for preview. Install it with: pip install mujoco"
        ) from exc

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    if args.keyframe:
        key_id = model.key(args.keyframe).id
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    mujoco.mj_forward(model, data)
    print(f"loaded MuJoCo model: {model_path}")
    print("close the MuJoCo viewer window to exit")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _configure_camera(viewer, model)
        while viewer.is_running():
            step_start = time.time()
            if not args.paused:
                mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - step_start
            time.sleep(max(0.0, float(model.opt.timestep) - elapsed))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
