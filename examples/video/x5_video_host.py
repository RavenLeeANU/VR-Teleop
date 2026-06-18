"""Run X5 UMI bimanual video host with IK-based teleop.

Uses mink inverse kinematics to map incoming wrist poses to X5 joint angles.
Requires ``mink`` and ``daqp``.

Usage::

    uv run examples/video/x5_video_host.py --mocap-tcp-port 5555
    uv run examples/video/x5_video_host.py --preset 1080p --perf
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from _common import build_base_parser, compensate_gravity, run_mujoco_host
from _tracking import RelativeHeadCamera, RelativeWristTracker

from hand_tracking_sdk.convert import (
    unity_left_to_rfu_position,
    unity_left_to_rfu_rotation_matrix,
)
from hand_tracking_sdk.frame import HandFrame, HeadFrame
from hand_tracking_sdk.teleop import GripConfig, grip_value

_DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "assets", "x5", "scene.xml")

# X5 joint names per arm.
_ARM_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]


def _parse_args() -> argparse.Namespace:
    parser = build_base_parser(
        "Host video service (X5 UMI MuJoCo source).",
        mujoco=True,
        default_mj_model=_DEFAULT_MODEL,
        default_mj_camera="teleop_overview",
        default_preset="480p",
        default_mocap_port=5555,
    )
    parser.add_argument(
        "--left-gripper-actuator",
        default="left/gripper",
        help="MuJoCo actuator name for left gripper.",
    )
    parser.add_argument(
        "--right-gripper-actuator",
        default="right/gripper",
        help="MuJoCo actuator name for right gripper.",
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
        help="IK posture regularization cost. Lower values let joint1 move more freely.",
    )
    parser.add_argument(
        "--ik-damping",
        type=float,
        default=1e-5,
        help="IK solver damping.",
    )
    parser.add_argument(
        "--debug-quest-pose",
        action="store_true",
        help="Print Quest wrist input and X5 gripper IK target/debug pose.",
    )
    parser.add_argument(
        "--debug-quest-pose-interval",
        type=int,
        default=30,
        help="Frame interval for --debug-quest-pose logs.",
    )
    parser.add_argument(
        "--show-ik-targets",
        action="store_true",
        help="Render left/right IK target markers in the MuJoCo scene.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# IK-based pre_step wiring
# ---------------------------------------------------------------------------


def _build_pre_step(
    latest: dict[str, HandFrame | HeadFrame],
    *,
    left_gripper_actuator: str,
    right_gripper_actuator: str,
    camera_name: str = "teleop_overview",
    grip_config: GripConfig | None = None,
    ik_iters: int = 8,
    ik_position_cost: float = 1.0,
    ik_orientation_cost: float = 0.25,
    ik_posture_cost: float = 1e-3,
    ik_damping: float = 1e-5,
    debug_quest_pose: bool = False,
    debug_quest_pose_interval: int = 30,
    show_ik_targets: bool = False,
) -> Any:
    """Build a pre_step callback that applies mocap state to MuJoCo via IK.

    Uses mink inverse kinematics to map wrist poses from incoming hand
    frames to joint-position actuator commands for the X5 arms.
    Head tracking drives 3-DOF camera rotation on *camera_name*.
    """
    if grip_config is None:
        grip_config = GripConfig()

    # Mutable state populated on first call (lazy init).
    state: dict[str, Any] = {}

    def fmt_vec(vec: Any) -> str:
        return "(" + ", ".join(f"{float(v):+.4f}" for v in vec) + ")"

    def fmt_quat(quat: Any) -> str:
        return "(" + ", ".join(f"{float(v):+.4f}" for v in quat) + ")"

    def rotation_angle(a: Any, b: Any) -> float:
        import numpy as np

        rel = a.T @ b
        cos_angle = (np.trace(rel) - 1.0) * 0.5
        return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    def mocap_id(model: Any, body_name: str) -> int | None:
        body_id = model.body(body_name).id
        mocap_id_value = int(model.body_mocapid[body_id])
        return mocap_id_value if mocap_id_value >= 0 else None

    def gripper_metrics(model: Any, data: Any, prefix: str, grip_id: int) -> dict[str, float]:
        import numpy as np

        left_joint = model.joint(f"{prefix}/left_finger").id
        right_joint = model.joint(f"{prefix}/right_finger").id
        left_qpos = float(data.qpos[model.jnt_qposadr[left_joint]])
        right_qpos = float(data.qpos[model.jnt_qposadr[right_joint]])

        left_site = model.site(f"{prefix}/left_finger").id
        right_site = model.site(f"{prefix}/right_finger").id
        site_opening = float(np.linalg.norm(data.site_xpos[left_site] - data.site_xpos[right_site]))

        return {
            "ctrl": float(data.ctrl[grip_id]),
            "left_qpos": left_qpos,
            "right_qpos": right_qpos,
            "site_opening": site_opening,
        }

    def pre_step(model: Any, data: Any) -> None:
        import mujoco
        import numpy as np

        # If initialization previously failed, do nothing.
        if state.get("disabled"):
            return

        # ---- lazy initialization on first call ----
        if not state:
            try:
                import mink
            except ImportError as exc:
                print(f"[mujoco-host] mink not available: {exc}")
                state["disabled"] = True
                return

            # Build joint name lists and resolve IDs.
            joint_names: list[str] = []
            velocity_limits: dict[str, float] = {}
            for prefix in ("left", "right"):
                for jn in _ARM_JOINT_NAMES:
                    name = f"{prefix}/{jn}"
                    joint_names.append(name)
                    velocity_limits[name] = np.pi

            joint_qpos_ids = np.array(
                [model.jnt_qposadr[model.joint(n).id] for n in joint_names]
            )
            actuator_ids = np.array([model.actuator(n).id for n in joint_names])

            left_grip_id = model.actuator(left_gripper_actuator).id
            right_grip_id = model.actuator(right_gripper_actuator).id

            left_subtree = model.body("left/base_link").id
            right_subtree = model.body("right/base_link").id
            left_target_mocap = mocap_id(model, "left/ik_target")
            right_target_mocap = mocap_id(model, "right/ik_target")

            configuration = mink.Configuration(model)

            l_ee_task = mink.FrameTask(
                frame_name="left/gripper",
                frame_type="site",
                position_cost=ik_position_cost,
                orientation_cost=ik_orientation_cost,
                lm_damping=1.0,
            )
            r_ee_task = mink.FrameTask(
                frame_name="right/gripper",
                frame_type="site",
                position_cost=ik_position_cost,
                orientation_cost=ik_orientation_cost,
                lm_damping=1.0,
            )
            posture_task = mink.PostureTask(model, cost=ik_posture_cost)

            tasks = [l_ee_task, r_ee_task, posture_task]
            limits = [
                mink.ConfigurationLimit(model=model),
                mink.VelocityLimit(model, velocity_limits),
            ]

            # Reset to neutral pose and set initial targets.
            mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
            configuration.update(data.qpos)
            mujoco.mj_forward(model, data)
            posture_task.set_target_from_configuration(configuration)
            l_ee_task.set_target_from_configuration(configuration)
            r_ee_task.set_target_from_configuration(configuration)

            # Capture initial EE poses for differential teleop.
            l_site_id = model.site("left/gripper").id
            r_site_id = model.site("right/gripper").id

            head_cam = RelativeHeadCamera(
                model,
                model.camera(camera_name).id,
                position_transform=unity_left_to_rfu_position,
                rotation_matrix_transform=unity_left_to_rfu_rotation_matrix,
            )

            state.update(
                mink=mink,
                configuration=configuration,
                l_ee_task=l_ee_task,
                r_ee_task=r_ee_task,
                tasks=tasks,
                limits=limits,
                joint_qpos_ids=joint_qpos_ids,
                actuator_ids=actuator_ids,
                left_grip_id=left_grip_id,
                right_grip_id=right_grip_id,
                target_mocap_ids={
                    "left": left_target_mocap,
                    "right": right_target_mocap,
                },
                subtree_ids=[left_subtree, right_subtree],
                head_cam=head_cam,
                left_tracker=RelativeWristTracker(
                    None,
                    data.site_xpos[l_site_id].copy(),
                    data.site_xmat[l_site_id].reshape(3, 3).copy(),
                    position_transform=unity_left_to_rfu_position,
                    rotation_matrix_transform=unity_left_to_rfu_rotation_matrix,
                ),
                right_tracker=RelativeWristTracker(
                    None,
                    data.site_xpos[r_site_id].copy(),
                    data.site_xmat[r_site_id].reshape(3, 3).copy(),
                    position_transform=unity_left_to_rfu_position,
                    rotation_matrix_transform=unity_left_to_rfu_rotation_matrix,
                ),
                frame_count=0,
                last_targets={},
                last_wrist_frames={},
            )

        # ---- per-frame IK solve ----
        mink = state["mink"]
        configuration = state["configuration"]
        l_ee_task = state["l_ee_task"]
        r_ee_task = state["r_ee_task"]
        tasks = state["tasks"]
        limits = state["limits"]
        joint_qpos_ids = state["joint_qpos_ids"]
        actuator_ids = state["actuator_ids"]

        # Sync mink configuration with actual sim state after previous mj_step.
        configuration.update(data.qpos)

        dt = 1.0 / 30.0  # Match video frame rate.

        # === Head tracking → camera rotation ===
        head = latest.get("Head")
        if isinstance(head, HeadFrame):
            state["head_cam"].update(head, model)

        left = latest.get("Left")
        right = latest.get("Right")
        state["frame_count"] += 1
        if show_ik_targets:
            for marker_mocap_id in state["target_mocap_ids"].values():
                if marker_mocap_id is not None:
                    data.mocap_pos[marker_mocap_id] = np.array([0.0, 0.0, -10.0])

        # Differential teleop: compute hand delta from reference pose,
        # apply to initial EE pose, then solve IK.
        if isinstance(left, HandFrame):
            target_pos, target_rot = state["left_tracker"].update(left.wrist)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, target_rot.flatten())
            l_ee_task.set_target(
                mink.SE3.from_rotation_and_translation(
                    rotation=mink.SO3(wxyz=quat),
                    translation=target_pos,
                )
            )
            data.ctrl[state["left_grip_id"]] = grip_value(left, grip_config)
            state["last_targets"]["left"] = (target_pos.copy(), target_rot.copy())
            state["last_wrist_frames"]["left"] = left.wrist
            if show_ik_targets and state["target_mocap_ids"]["left"] is not None:
                data.mocap_pos[state["target_mocap_ids"]["left"]] = target_pos

        if isinstance(right, HandFrame):
            target_pos, target_rot = state["right_tracker"].update(right.wrist)
            quat = np.empty(4)
            mujoco.mju_mat2Quat(quat, target_rot.flatten())
            r_ee_task.set_target(
                mink.SE3.from_rotation_and_translation(
                    rotation=mink.SO3(wxyz=quat),
                    translation=target_pos,
                )
            )
            data.ctrl[state["right_grip_id"]] = grip_value(right, grip_config)
            state["last_targets"]["right"] = (target_pos.copy(), target_rot.copy())
            state["last_wrist_frames"]["right"] = right.wrist
            if show_ik_targets and state["target_mocap_ids"]["right"] is not None:
                data.mocap_pos[state["target_mocap_ids"]["right"]] = target_pos

        # X5's base yaw (joint1) needs a few iterations to participate in
        # lateral targets; keep this tunable for calibration.
        for _ in range(ik_iters):
            vel = mink.solve_ik(
                configuration,
                tasks,
                dt,
                "daqp",
                limits=limits,
                damping=ik_damping,
            )
            configuration.integrate_inplace(vel, dt)

            l_err = l_ee_task.compute_error(configuration)
            r_err = r_ee_task.compute_error(configuration)
            if (
                np.linalg.norm(l_err[:3]) <= 5e-3
                and np.linalg.norm(l_err[3:]) <= 5e-3
                and np.linalg.norm(r_err[:3]) <= 5e-3
                and np.linalg.norm(r_err[3:]) <= 5e-3
            ):
                break

        # Write joint angles and apply gravity compensation.
        data.ctrl[actuator_ids] = configuration.q[joint_qpos_ids]
        compensate_gravity(model, data, state["subtree_ids"])

        if (
            debug_quest_pose
            and debug_quest_pose_interval > 0
            and state["frame_count"] % debug_quest_pose_interval == 0
        ):
            ik_data = mujoco.MjData(model)
            ik_data.qpos[:] = configuration.q
            mujoco.mj_forward(model, ik_data)

            for prefix in ("left", "right"):
                target = state["last_targets"].get(prefix)
                wrist = state["last_wrist_frames"].get(prefix)
                if target is None or wrist is None:
                    continue

                target_pos, target_rot = target
                target_quat = np.empty(4)
                mujoco.mju_mat2Quat(target_quat, target_rot.flatten())
                site_id = model.site(f"{prefix}/gripper").id
                actual_pos = ik_data.site_xpos[site_id].copy()
                actual_rot = ik_data.site_xmat[site_id].reshape(3, 3).copy()
                actual_quat = np.empty(4)
                mujoco.mju_mat2Quat(actual_quat, actual_rot.flatten())
                joint_values = configuration.q[
                    [model.jnt_qposadr[model.joint(f"{prefix}/{jn}").id] for jn in _ARM_JOINT_NAMES]
                ]
                grip_id = state[f"{prefix}_grip_id"]
                grip = gripper_metrics(model, data, prefix, grip_id)
                print(
                    f"[x5-debug] frame={state['frame_count']} side={prefix} "
                    f"quest_pos=({wrist.x:+.4f}, {wrist.y:+.4f}, {wrist.z:+.4f}) "
                    f"quest_quat_xyzw=({wrist.qx:+.4f}, {wrist.qy:+.4f}, "
                    f"{wrist.qz:+.4f}, {wrist.qw:+.4f})"
                )
                print(
                    f"[x5-debug] side={prefix} "
                    f"target_pos={fmt_vec(target_pos)} "
                    f"target_quat_wxyz={fmt_quat(target_quat)} "
                    f"ik_pos={fmt_vec(actual_pos)} "
                    f"ik_quat_wxyz={fmt_quat(actual_quat)} "
                    f"pos_err={np.linalg.norm(actual_pos - target_pos):.4f} "
                    f"rot_err_deg={np.degrees(rotation_angle(target_rot, actual_rot)):.2f}"
                )
                print(
                    f"[x5-debug] side={prefix} joints="
                    + " ".join(
                        f"{name}={value:+.4f}"
                        for name, value in zip(_ARM_JOINT_NAMES, joint_values, strict=True)
                    )
                    + f" grip_ctrl={grip['ctrl']:+.4f}"
                    + f" grip_left_qpos={grip['left_qpos']:+.4f}"
                    + f" grip_right_qpos={grip['right_qpos']:+.4f}"
                    + f" grip_opening={grip['site_opening']:+.4f}m"
                )

    return pre_step


def _build_pre_step_from_args(
    latest: dict[str, HandFrame | HeadFrame],
    args: argparse.Namespace,
) -> Any:
    return _build_pre_step(
        latest,
        left_gripper_actuator=args.left_gripper_actuator,
        right_gripper_actuator=args.right_gripper_actuator,
        camera_name=args.mj_camera,
        ik_iters=args.ik_iters,
        ik_position_cost=args.ik_position_cost,
        ik_orientation_cost=args.ik_orientation_cost,
        ik_posture_cost=args.ik_posture_cost,
        ik_damping=args.ik_damping,
        debug_quest_pose=args.debug_quest_pose,
        debug_quest_pose_interval=args.debug_quest_pose_interval,
        show_ik_targets=args.show_ik_targets,
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_mujoco_host(_parse_args(), _build_pre_step_from_args)))
