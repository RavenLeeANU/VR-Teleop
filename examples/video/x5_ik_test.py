"""Local MuJoCo viewer for testing X5 IK directions.

The IK setup intentionally mirrors ``x5_video_host.py``: same X5 scene,
same gripper sites, same mink tasks, same velocity/configuration limits,
and the same actuator write path.  Instead of live wrist telemetry, this
script applies fixed Cartesian offsets to the gripper site.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from _common import compensate_gravity
from x5_video_host import _ARM_JOINT_NAMES, _DEFAULT_MODEL


_DIRECTION_VECTORS = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}

_DEFAULT_TRANSLATE_DIRECTIONS = "+x,-x,+y,-y,+z,-z"
_YAW_DIRECTIONS = {"+yaw", "-yaw"}
_ROTATE_DIRECTIONS = {"+roll", "-roll", "+pitch", "-pitch", "+yaw", "-yaw"}
_DEFAULT_ROTATE_DIRECTIONS = "+roll,-roll,+pitch,-pitch,+yaw,-yaw"
_TARGET_COLORS = {
    "left": np.array([0.1, 0.45, 1.0, 0.85]),
    "right": np.array([1.0, 0.45, 0.1, 0.85]),
}


@dataclass(frozen=True)
class ArmState:
    prefix: str
    task: Any
    site_id: int
    base_pos: np.ndarray
    home_pos: np.ndarray
    home_mat: np.ndarray
    joint_qpos_ids: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize X5 IK responses by direction.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(_DEFAULT_MODEL),
        help=f"MuJoCo XML model. Default: {_DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--arm",
        choices=("left", "right", "both"),
        default="left",
        help="Arm to move during the test.",
    )
    parser.add_argument(
        "--directions",
        default=_DEFAULT_TRANSLATE_DIRECTIONS,
        help=(
            "Comma-separated direction list, e.g. '+x,+y,-z', '+yaw,-yaw', "
            "or '+roll,+pitch,-yaw'."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("translate", "yaw", "point", "rotate"),
        default="translate",
        help=(
            "translate offsets the gripper; yaw rotates the target around the arm base; "
            "point tracks a given XYZ target; rotate changes gripper orientation in place."
        ),
    )
    parser.add_argument(
        "--point",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Target XYZ for point mode. Defaults to the selected gripper home position.",
    )
    parser.add_argument(
        "--point-relative",
        action="store_true",
        help="Interpret --point as an offset from each selected gripper home position.",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.08,
        help="Cartesian offset length in meters.",
    )
    parser.add_argument(
        "--yaw-angle",
        type=float,
        default=0.35,
        help="Yaw test angle in radians.",
    )
    parser.add_argument(
        "--rotate-angle",
        type=float,
        default=0.35,
        help="End-effector local rotation test angle in radians.",
    )
    parser.add_argument(
        "--hold-steps",
        type=int,
        default=90,
        help="Simulation steps to hold each direction target.",
    )
    parser.add_argument(
        "--return-steps",
        type=int,
        default=45,
        help="Simulation steps used to return to neutral between directions.",
    )
    parser.add_argument(
        "--ik-iters",
        type=int,
        default=8,
        help="Maximum IK iterations per simulation step.",
    )
    parser.add_argument(
        "--ik-position-cost",
        type=float,
        default=1.0,
        help="IK end-effector position tracking cost.",
    )
    parser.add_argument(
        "--ik-orientation-cost",
        type=float,
        default=0.25,
        help="IK end-effector orientation tracking cost.",
    )
    parser.add_argument(
        "--ik-posture-cost",
        type=float,
        default=1e-3,
        help="IK posture regularization cost.",
    )
    parser.add_argument(
        "--ik-damping",
        type=float,
        default=1e-5,
        help="IK solver damping.",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="Open viewer at neutral pose and do not run the direction sequence.",
    )
    parser.add_argument(
        "--kinematic",
        action="store_true",
        help="Apply IK qpos directly instead of waiting for position actuators to track it.",
    )
    return parser.parse_args()


def _parse_directions(value: str) -> list[str]:
    directions = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = [
        item
        for item in directions
        if item not in _DIRECTION_VECTORS
        and item not in _YAW_DIRECTIONS
        and item not in _ROTATE_DIRECTIONS
    ]
    if invalid:
        valid = ", ".join([*_DIRECTION_VECTORS, *_YAW_DIRECTIONS, *_ROTATE_DIRECTIONS])
        raise ValueError(f"Invalid directions {invalid}; valid values are: {valid}")
    return directions


def _configure_camera(viewer: Any, model: Any) -> None:
    extent = float(getattr(model.stat, "extent", 1.0)) or 1.0
    viewer.cam.lookat[:] = model.stat.center
    viewer.cam.distance = max(1.2, extent * 1.8)
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -25.0


def _draw_targets(
    viewer: Any,
    mujoco: Any,
    targets: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0
    mat = np.eye(3).reshape(9)
    size = np.array([0.025, 0.0, 0.0])

    for prefix, (target_pos, _target_mat) in targets.items():
        if scene.ngeom >= scene.maxgeom:
            break
        color = _TARGET_COLORS.get(prefix, np.array([0.0, 1.0, 0.0, 0.85]))
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size,
            target_pos,
            mat,
            color,
        )
        scene.ngeom += 1


def _joint_names(prefix: str) -> list[str]:
    return [f"{prefix}/{name}" for name in _ARM_JOINT_NAMES]


def _build_arm_state(
    model: Any,
    data: Any,
    mink: Any,
    prefix: str,
    *,
    position_cost: float,
    orientation_cost: float,
) -> ArmState:
    site_id = model.site(f"{prefix}/gripper").id
    task = mink.FrameTask(
        frame_name=f"{prefix}/gripper",
        frame_type="site",
        position_cost=position_cost,
        orientation_cost=orientation_cost,
        lm_damping=1.0,
    )
    joint_qpos_ids = np.array(
        [model.jnt_qposadr[model.joint(name).id] for name in _joint_names(prefix)]
    )
    return ArmState(
        prefix=prefix,
        task=task,
        site_id=site_id,
        base_pos=data.xpos[model.body(f"{prefix}/base_link").id].copy(),
        home_pos=data.site_xpos[site_id].copy(),
        home_mat=data.site_xmat[site_id].reshape(3, 3).copy(),
        joint_qpos_ids=joint_qpos_ids,
    )


def _set_arm_target(
    mujoco: Any,
    mink: Any,
    arm: ArmState,
    target_pos: np.ndarray,
    target_mat: np.ndarray,
) -> None:
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, target_mat.flatten())
    arm.task.set_target(
        mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(wxyz=quat),
            translation=target_pos,
        )
    )


def _yaw_target(arm: ArmState, angle: float) -> tuple[np.ndarray, np.ndarray]:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    rot = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target_pos = arm.base_pos + rot @ (arm.home_pos - arm.base_pos)
    target_mat = rot @ arm.home_mat
    return target_pos, target_mat


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ]
    )


def _rotate_target(arm: ArmState, direction: str, angle: float) -> tuple[np.ndarray, np.ndarray]:
    sign = 1.0 if direction.startswith("+") else -1.0
    name = direction[1:]
    axis_index = {"roll": 0, "pitch": 1, "yaw": 2}[name]
    local_axis = arm.home_mat[:, axis_index]
    target_mat = _axis_angle(local_axis, sign * angle) @ arm.home_mat
    return arm.home_pos.copy(), target_mat


def _rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    rel = a.T @ b
    cos_angle = (np.trace(rel) - 1.0) * 0.5
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _print_direction_result(
    direction: str,
    arm: ArmState,
    configuration: Any,
    neutral_q: np.ndarray,
    model: Any,
    data: Any,
    mujoco: Any,
    target_pos: np.ndarray | None = None,
    target_mat: np.ndarray | None = None,
) -> None:
    deltas = configuration.q[arm.joint_qpos_ids] - neutral_q[arm.joint_qpos_ids]
    pieces = [
        f"{name}={delta:+.4f}"
        for name, delta in zip(_ARM_JOINT_NAMES, deltas, strict=True)
    ]
    err = arm.task.compute_error(configuration)
    ik_data = mujoco.MjData(model)
    ik_data.qpos[:] = configuration.q
    mujoco.mj_forward(model, ik_data)
    ik_pos = ik_data.site_xpos[arm.site_id].copy()
    ik_mat = ik_data.site_xmat[arm.site_id].reshape(3, 3).copy()
    sim_pos = data.site_xpos[arm.site_id].copy()
    sim_mat = data.site_xmat[arm.site_id].reshape(3, 3).copy()
    print(
        f"[{direction}] {arm.prefix}: "
        + " ".join(pieces)
        + f" | pos_err={np.linalg.norm(err[:3]):.4f}"
        + f" rot_err={np.linalg.norm(err[3:]):.4f}"
    )
    if target_pos is not None:
        home_xy = arm.home_pos[:2] - arm.base_pos[:2]
        target_xy = target_pos[:2] - arm.base_pos[:2]
        home_yaw = float(np.degrees(np.arctan2(home_xy[1], home_xy[0])))
        target_yaw = float(np.degrees(np.arctan2(target_xy[1], target_xy[0])))
        yaw_hint = target_yaw - home_yaw
        print(
            f"    target=({target_pos[0]:+.4f}, {target_pos[1]:+.4f}, {target_pos[2]:+.4f})"
            f" ik_actual=({ik_pos[0]:+.4f}, {ik_pos[1]:+.4f}, {ik_pos[2]:+.4f})"
            f" ik_xyz_err={np.linalg.norm(ik_pos - target_pos):.4f}"
        )
        print(
            f"    sim_actual=({sim_pos[0]:+.4f}, {sim_pos[1]:+.4f}, {sim_pos[2]:+.4f})"
            f" sim_xyz_err={np.linalg.norm(sim_pos - target_pos):.4f}"
        )
        print(
            f"    base_xy_yaw: home={home_yaw:+.1f}deg"
            f" target={target_yaw:+.1f}deg hint_delta={yaw_hint:+.1f}deg"
        )
    if target_mat is not None:
        print(
            f"    ik_rot_angle_err={np.degrees(_rotation_angle(target_mat, ik_mat)):.2f}deg"
            f" sim_rot_angle_err={np.degrees(_rotation_angle(target_mat, sim_mat)):.2f}deg"
        )


def main() -> int:
    args = _parse_args()
    direction_arg = args.directions
    if args.mode == "yaw" and direction_arg == _DEFAULT_TRANSLATE_DIRECTIONS:
        direction_arg = "+yaw,-yaw"
    if args.mode == "rotate" and direction_arg == _DEFAULT_TRANSLATE_DIRECTIONS:
        direction_arg = _DEFAULT_ROTATE_DIRECTIONS
    directions = ["point"] if args.mode == "point" else _parse_directions(direction_arg)
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo XML not found: {model_path}")

    try:
        import mujoco
        import mujoco.viewer
        import mink
    except ImportError as exc:
        raise RuntimeError("This test requires mujoco, mink, and daqp.") from exc

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
    mujoco.mj_forward(model, data)

    configuration = mink.Configuration(model)
    configuration.update(data.qpos)
    neutral_q = data.qpos.copy()

    arms = {
        prefix: _build_arm_state(
            model,
            data,
            mink,
            prefix,
            position_cost=args.ik_position_cost,
            orientation_cost=args.ik_orientation_cost,
        )
        for prefix in ("left", "right")
    }

    posture_task = mink.PostureTask(model, cost=args.ik_posture_cost)
    posture_task.set_target_from_configuration(configuration)
    for arm in arms.values():
        arm.task.set_target_from_configuration(configuration)

    tasks = [arms["left"].task, arms["right"].task, posture_task]
    velocity_limits = {
        name: np.pi for prefix in ("left", "right") for name in _joint_names(prefix)
    }
    limits = [
        mink.ConfigurationLimit(model=model),
        mink.VelocityLimit(model, velocity_limits),
    ]
    joint_qpos_ids = np.array(
        [
            model.jnt_qposadr[model.joint(name).id]
            for prefix in ("left", "right")
            for name in _joint_names(prefix)
        ]
    )
    actuator_ids = np.array(
        [
            model.actuator(name).id
            for prefix in ("left", "right")
            for name in _joint_names(prefix)
        ]
    )
    subtree_ids = [model.body("left/base_link").id, model.body("right/base_link").id]
    active_prefixes = ("left", "right") if args.arm == "both" else (args.arm,)

    def point_target(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        arm = arms[prefix]
        if args.point is None:
            target_pos = arm.home_pos.copy()
        elif args.point_relative:
            target_pos = arm.home_pos + np.asarray(args.point, dtype=float)
        else:
            target_pos = np.asarray(args.point, dtype=float)
        return target_pos, arm.home_mat

    def apply_ik(targets: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        configuration.update(data.qpos)
        for prefix, arm in arms.items():
            target_pos, target_mat = targets.get(prefix, (arm.home_pos, arm.home_mat))
            _set_arm_target(mujoco, mink, arm, target_pos, target_mat)

        dt = 1.0 / 30.0
        for _ in range(args.ik_iters):
            vel = mink.solve_ik(
                configuration,
                tasks,
                dt,
                "daqp",
                limits=limits,
                damping=args.ik_damping,
            )
            configuration.integrate_inplace(vel, dt)

        data.ctrl[actuator_ids] = configuration.q[joint_qpos_ids]
        if args.kinematic:
            data.qpos[:] = configuration.q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
        else:
            compensate_gravity(model, data, subtree_ids)

    def step_viewer(viewer: Any, targets: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        started = time.time()
        apply_ik(targets)
        if not args.kinematic:
            mujoco.mj_step(model, data)
        _draw_targets(viewer, mujoco, targets)
        viewer.sync()
        elapsed = time.time() - started
        time.sleep(max(0.0, float(model.opt.timestep) - elapsed))

    print(f"loaded MuJoCo model: {model_path}")
    print(
        f"testing arm={args.arm}, mode={args.mode}, directions={directions}, "
        f"offset={args.offset:.3f}m, yaw_angle={args.yaw_angle:.3f}rad, "
        f"rotate_angle={args.rotate_angle:.3f}rad"
    )
    print("close the MuJoCo viewer window to exit")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _configure_camera(viewer, model)
        if args.paused:
            while viewer.is_running():
                viewer.user_scn.ngeom = 0
                viewer.sync()
                time.sleep(float(model.opt.timestep))
            return 0

        while viewer.is_running():
            for direction in directions:
                if not viewer.is_running():
                    break

                if args.mode == "point":
                    targets = {prefix: point_target(prefix) for prefix in active_prefixes}
                elif args.mode == "rotate":
                    if direction not in _ROTATE_DIRECTIONS:
                        print(
                            f"skip {direction}: rotate mode only accepts "
                            "+roll/-roll/+pitch/-pitch/+yaw/-yaw"
                        )
                        continue
                    targets = {
                        prefix: (
                            point_target(prefix)[0],
                            _rotate_target(arms[prefix], direction, args.rotate_angle)[1],
                        )
                        for prefix in active_prefixes
                    }
                elif args.mode == "yaw":
                    if direction not in _YAW_DIRECTIONS:
                        print(f"skip {direction}: yaw mode only accepts +yaw/-yaw")
                        continue
                    yaw = args.yaw_angle if direction == "+yaw" else -args.yaw_angle
                    targets = {
                        prefix: _yaw_target(arms[prefix], yaw) for prefix in active_prefixes
                    }
                else:
                    if direction not in _DIRECTION_VECTORS:
                        print(f"skip {direction}: translate mode only accepts Cartesian directions")
                        continue
                    vector = _DIRECTION_VECTORS[direction] * args.offset
                    targets = {
                        prefix: (arms[prefix].home_pos + vector, arms[prefix].home_mat)
                        for prefix in active_prefixes
                    }
                print(f"\n=== direction {direction} ===")

                for _ in range(args.hold_steps):
                    if not viewer.is_running():
                        break
                    step_viewer(viewer, targets)

                for prefix in active_prefixes:
                    target_pos = targets[prefix][0] if prefix in targets else None
                    _print_direction_result(
                        direction,
                        arms[prefix],
                        configuration,
                        neutral_q,
                        model,
                        data,
                        mujoco,
                        target_pos,
                        targets[prefix][1] if prefix in targets else None,
                    )

                if args.mode in ("point", "rotate"):
                    continue

                for step in range(args.return_steps):
                    if not viewer.is_running():
                        break
                    alpha = 1.0 - (step + 1) / max(1, args.return_steps)
                    if args.mode == "yaw":
                        yaw = (args.yaw_angle if direction == "+yaw" else -args.yaw_angle) * alpha
                        return_targets = {
                            prefix: _yaw_target(arms[prefix], yaw)
                            for prefix in active_prefixes
                        }
                    else:
                        vector = _DIRECTION_VECTORS[direction] * args.offset * alpha
                        return_targets = {
                            prefix: (arms[prefix].home_pos + vector, arms[prefix].home_mat)
                            for prefix in active_prefixes
                        }
                    step_viewer(viewer, return_targets)

                mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
                configuration.update(data.qpos)
                mujoco.mj_forward(model, data)
                viewer.user_scn.ngeom = 0
                viewer.sync()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
