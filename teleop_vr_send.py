from __future__ import annotations

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import click
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

try:
    from arx5_interface import (  # type: ignore  # noqa: E402
        Arx5CartesianController,
        ControllerConfigFactory,
        EEFState,
        Gain,
        LogLevel,
        RobotConfigFactory,
    )

    ARX_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:
    Arx5CartesianController = Any
    ARX_IMPORT_ERROR = exc

from hand_tracking_sdk import (  # noqa: E402
    ErrorPolicy,
    GripConfig,
    HTSClient,
    HTSClientConfig,
    HandFrame,
    HandSide,
    StreamOutput,
    TransportMode,
    grip_value,
    unity_left_to_flu_position,
    unity_left_to_flu_rotation_matrix,
    unity_left_to_rfu_position,
    unity_left_to_rfu_rotation_matrix,
)


@dataclass
class MockRobotConfig:
    joint_dof: int = 6
    gripper_width: float = 0.08


class MockRobotConfigFactory:
    @classmethod
    def get_instance(cls) -> "MockRobotConfigFactory":
        return cls()

    def get_config(self, model: str) -> MockRobotConfig:
        _ = model
        return MockRobotConfig()


@dataclass
class MockControllerConfig:
    joint_dof: int


class MockControllerConfigFactory:
    @classmethod
    def get_instance(cls) -> "MockControllerConfigFactory":
        return cls()

    def get_config(self, name: str, joint_dof: int) -> MockControllerConfig:
        _ = name
        return MockControllerConfig(joint_dof=joint_dof)


class MockEEFState:
    def __init__(self) -> None:
        self._pose_6d = np.zeros(6, dtype=float)
        self.gripper_pos = 0.0
        self.timestamp = 0.0

    def pose_6d(self) -> np.ndarray:
        return self._pose_6d

    def copy_from(self, other: Any) -> None:
        self.pose_6d()[:] = other.pose_6d()
        self.gripper_pos = float(other.gripper_pos)
        self.timestamp = float(other.timestamp)


class MockLogLevel:
    DEBUG = "DEBUG"


class MockGain:
    def __init__(self, joint_dof: int) -> None:
        self.joint_dof = joint_dof


class MockArx5CartesianController:
    def __init__(
        self,
        robot_config: MockRobotConfig,
        controller_config: MockControllerConfig,
        interface: str,
    ) -> None:
        self._robot_config = robot_config
        self._controller_config = controller_config
        self._interface = interface
        self._home_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._eef_cmd = MockEEFState()
        self._eef_state = MockEEFState()
        self._start_time = time.monotonic()
        self._send_count = 0

    def get_robot_config(self) -> MockRobotConfig:
        return self._robot_config

    def get_home_pose(self) -> np.ndarray:
        return self._home_pose.copy()

    def reset_to_home(self) -> None:
        self._eef_cmd.pose_6d()[:] = self._home_pose
        self._eef_state.pose_6d()[:] = self._home_pose
        self._eef_cmd.gripper_pos = 0.0
        self._eef_state.gripper_pos = 0.0
        print(f"[mock-arx] reset_to_home pose={self._home_pose}")

    def set_log_level(self, level: Any) -> None:
        print(f"[mock-arx] set_log_level {level}")

    def get_timestamp(self) -> float:
        return time.monotonic() - self._start_time

    def set_eef_traj(self, traj: list[Any]) -> None:
        if not traj:
            return
        self._accept_eef_cmd(traj[-1], mode="traj")

    def set_eef_cmd(self, cmd: Any) -> None:
        self._accept_eef_cmd(cmd, mode="cmd")

    def _accept_eef_cmd(self, cmd: Any, *, mode: str) -> None:
        self._eef_cmd.copy_from(cmd)
        self._eef_state.copy_from(cmd)
        self._send_count += 1
        print(
            f"[mock-arx] send#{self._send_count} mode={mode} "
            f"timestamp={self._eef_cmd.timestamp:.4f} "
            f"pose_6d={np.array2string(self._eef_cmd.pose_6d(), precision=4, suppress_small=True)} "
            f"gripper={self._eef_cmd.gripper_pos:.4f}/{self._robot_config.gripper_width:.4f}"
        )

    def get_eef_cmd(self) -> MockEEFState:
        return self._eef_cmd

    def get_eef_state(self) -> MockEEFState:
        return self._eef_state

    def set_to_damping(self) -> None:
        print("[mock-arx] set_to_damping")


if ARX_IMPORT_ERROR is not None:
    RobotConfigFactory = MockRobotConfigFactory
    ControllerConfigFactory = MockControllerConfigFactory
    EEFState = MockEEFState
    Gain = MockGain
    LogLevel = MockLogLevel


@dataclass
class VrReference:
    pos: np.ndarray
    rot: np.ndarray
    robot_home_pose: np.ndarray


@dataclass
class DampingConfig:
    enabled: bool
    alpha: float
    max_pos_step: float
    max_ori_step: float
    max_gripper_step: float


@dataclass
class DampingState:
    pose_6d: np.ndarray | None = None
    gripper_pos: float | None = None


@dataclass
class TeleopTarget:
    side: HandSide
    sequence_id: int
    vr_pos: np.ndarray
    pos_delta: np.ndarray
    rpy_delta: np.ndarray
    raw_pose_6d: np.ndarray
    raw_gripper_pos: float
    received_at: float


def _rpy_to_rotmat(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=float,
    )
    rot_y = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=float,
    )
    rot_z = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rot_z @ rot_y @ rot_x


def _rotmat_to_rpy(rot: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to roll/pitch/yaw using Rz(yaw) Ry(pitch) Rx(roll)."""
    sy = float(np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0]))
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        roll = np.arctan2(-rot[1, 2], rot[1, 1])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def _frame_pose(
    frame: HandFrame,
    *,
    basis: str,
) -> tuple[np.ndarray, np.ndarray]:
    wrist = frame.wrist
    if basis == "flu":
        pos = np.asarray(unity_left_to_flu_position(wrist.x, wrist.y, wrist.z), dtype=float)
        rot = np.asarray(
            unity_left_to_flu_rotation_matrix(wrist.qx, wrist.qy, wrist.qz, wrist.qw),
            dtype=float,
        )
    else:
        pos = np.asarray(unity_left_to_rfu_position(wrist.x, wrist.y, wrist.z), dtype=float)
        rot = np.asarray(
            unity_left_to_rfu_rotation_matrix(wrist.qx, wrist.qy, wrist.qz, wrist.qw),
            dtype=float,
        )
    return pos, rot


def _side_from_text(side: str) -> HandSide:
    normalized = side.lower()
    if normalized == "left":
        return HandSide.LEFT
    if normalized == "right":
        return HandSide.RIGHT
    raise click.BadParameter("side must be left or right")


def _wrap_angle_delta(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _make_rpy_continuous(rpy: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    if previous is None:
        return _wrap_angle_delta(rpy)
    return previous + _wrap_angle_delta(rpy - previous)


def _limit_vector_norm(delta: np.ndarray, max_norm: float) -> tuple[np.ndarray, bool]:
    norm = float(np.linalg.norm(delta))
    if norm <= max_norm or norm <= 1e-12:
        return delta, False
    return delta * (max_norm / norm), True


def _apply_damping_protection(
    raw_pose_6d: np.ndarray,
    raw_gripper_pos: float,
    state: DampingState,
    config: DampingConfig,
) -> tuple[np.ndarray, float, bool]:
    if state.pose_6d is None or state.gripper_pos is None:
        state.pose_6d = raw_pose_6d.copy()
        state.gripper_pos = float(raw_gripper_pos)
        return state.pose_6d.copy(), state.gripper_pos, False

    delta = raw_pose_6d - state.pose_6d
    delta[3:] = _wrap_angle_delta(delta[3:])

    pos_delta, pos_limited = _limit_vector_norm(delta[:3], config.max_pos_step)
    ori_delta, ori_limited = _limit_vector_norm(delta[3:], config.max_ori_step)

    gripper_delta_raw = float(raw_gripper_pos - state.gripper_pos)
    gripper_delta = float(
        np.clip(gripper_delta_raw, -config.max_gripper_step, config.max_gripper_step)
    )
    gripper_limited = abs(gripper_delta - gripper_delta_raw) > 1e-12

    stepped_pose = state.pose_6d.copy()
    stepped_pose[:3] += pos_delta
    stepped_pose[3:] += ori_delta
    stepped_gripper = state.gripper_pos + gripper_delta

    state.pose_6d = state.pose_6d + config.alpha * (stepped_pose - state.pose_6d)
    state.gripper_pos = state.gripper_pos + config.alpha * (
        stepped_gripper - state.gripper_pos
    )

    limited = pos_limited or ori_limited or gripper_limited
    return state.pose_6d.copy(), float(state.gripper_pos), limited


def _gripper_from_frame(
    frame: HandFrame,
    *,
    robot_gripper_width: float,
    grip_config: GripConfig,
) -> float:
    grip_ctrl = grip_value(frame, grip_config)
    t = (grip_ctrl - grip_config.ctrl_min) / (grip_config.ctrl_max - grip_config.ctrl_min)
    return float(np.clip(t, 0.0, 1.0) * robot_gripper_width)


def start_vr_teleop(
    controller: Arx5CartesianController,
    *,
    hand_side: HandSide,
    mocap_host: str,
    mocap_port: int,
    transport_mode: TransportMode,
    basis: str,
    pos_scale: float,
    ori_scale: float,
    cmd_dt: float,
    preview_time: float,
    update_traj: bool,
    log_interval: int,
    grip_config: GripConfig,
    damping_config: DampingConfig,
    zero_first_frame: bool,
) -> None:
    robot_config = controller.get_robot_config()
    home_pose = np.asarray(controller.get_home_pose(), dtype=float).copy()
    target_queue: queue.Queue[TeleopTarget] = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    dropped_target_count = 0
    ignored_other_hand_count = 0
    start_time = time.monotonic()

    print(
        "VR teleop ready. "
        f"hand={hand_side.value} transport={transport_mode.value} "
        f"{mocap_host}:{mocap_port} basis={basis} "
        f"zero_first_frame={zero_first_frame}"
    )
    if damping_config.enabled:
        print(
            "Damping protection enabled: "
            f"alpha={damping_config.alpha:.3f} "
            f"max_pos_step={damping_config.max_pos_step:.4f}m "
            f"max_ori_step={damping_config.max_ori_step:.4f}rad "
            f"max_gripper_step={damping_config.max_gripper_step:.4f}m"
        )
    print("Waiting for the first matching VR hand frame as reference pose.")

    def put_latest(target: TeleopTarget) -> None:
        nonlocal dropped_target_count
        try:
            target_queue.put_nowait(target)
            return
        except queue.Full:
            pass
        try:
            target_queue.get_nowait()
            dropped_target_count += 1
        except queue.Empty:
            pass
        target_queue.put_nowait(target)

    def receive_loop() -> None:
        nonlocal ignored_other_hand_count
        reference: VrReference | None = None
        last_raw_rpy: np.ndarray | None = None
        client = HTSClient(
            HTSClientConfig(
                transport_mode=transport_mode,
                host=mocap_host,
                port=mocap_port,
                output=StreamOutput.FRAMES,
                error_policy=ErrorPolicy.TOLERANT,
            )
        )

        try:
            for event in client.iter_events():
                if stop_event.is_set():
                    break
                if not isinstance(event, HandFrame):
                    continue
                if event.side != hand_side:
                    ignored_other_hand_count += 1
                    continue

                vr_pos, vr_rot = _frame_pose(event, basis=basis)
                if reference is None:
                    reference = VrReference(
                        pos=vr_pos.copy(),
                        rot=vr_rot.copy(),
                        robot_home_pose=home_pose.copy(),
                    )
                    print(
                        "Reference captured: "
                        f"vr_pos={np.array2string(reference.pos, precision=4, suppress_small=True)} "
                        f"vr_rpy={np.array2string(_rotmat_to_rpy(reference.rot), precision=4, suppress_small=True)} "
                        f"home_pose={np.array2string(reference.robot_home_pose, precision=4, suppress_small=True)}"
                    )
                    continue

                if zero_first_frame:
                    pos_delta = (vr_pos - reference.pos) * pos_scale
                    rot_delta = vr_rot @ reference.rot.T
                    base_pose = reference.robot_home_pose
                else:
                    pos_delta = vr_pos * pos_scale
                    rot_delta = vr_rot
                    base_pose = reference.robot_home_pose
                rpy_delta = _rotmat_to_rpy(rot_delta) * ori_scale
                target_rot = _rpy_to_rotmat(base_pose[3:])
                if ori_scale != 0.0:
                    target_rot = _rpy_to_rotmat(rpy_delta) @ target_rot

                raw_pose_6d = base_pose.copy()
                raw_pose_6d[:3] = base_pose[:3] + pos_delta
                raw_pose_6d[3:] = _make_rpy_continuous(
                    _rotmat_to_rpy(target_rot),
                    last_raw_rpy,
                )
                last_raw_rpy = raw_pose_6d[3:].copy()
                raw_gripper_pos = _gripper_from_frame(
                    event,
                    robot_gripper_width=robot_config.gripper_width,
                    grip_config=grip_config,
                )

                put_latest(
                    TeleopTarget(
                        side=event.side,
                        sequence_id=event.sequence_id,
                        vr_pos=vr_pos.copy(),
                        pos_delta=pos_delta.copy(),
                        rpy_delta=rpy_delta.copy(),
                        raw_pose_6d=raw_pose_6d,
                        raw_gripper_pos=raw_gripper_pos,
                        received_at=time.monotonic(),
                    )
                )
        except Exception as exc:
            stop_event.set()
            print(f"[vr-teleop] receive thread stopped by error: {exc!r}")

    def send_loop() -> None:
        eef_cmd = EEFState()
        damping_state = DampingState()
        current_target: TeleopTarget | None = None
        last_sent_rpy: np.ndarray | None = None
        avg_error = np.zeros(6)
        avg_cnt = 0
        send_cnt = 0
        latest_seq = -1
        repeated_target_count = 0

        while not stop_event.is_set():
            target_time = start_time + (send_cnt + 1) * cmd_dt
            sleep_time = target_time - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(min(sleep_time, cmd_dt))

            try:
                while True:
                    current_target = target_queue.get_nowait()
            except queue.Empty:
                pass

            if current_target is None:
                continue

            raw_target_pose_6d = current_target.raw_pose_6d.copy()
            raw_target_gripper_pos = current_target.raw_gripper_pos
            target_pose_6d = raw_target_pose_6d
            target_gripper_pos = raw_target_gripper_pos
            damping_limited = False
            if damping_config.enabled:
                target_pose_6d, target_gripper_pos, damping_limited = _apply_damping_protection(
                    raw_target_pose_6d,
                    raw_target_gripper_pos,
                    damping_state,
                    damping_config,
                )
            target_pose_6d[3:] = _make_rpy_continuous(target_pose_6d[3:], last_sent_rpy)
            last_sent_rpy = target_pose_6d[3:].copy()

            current_timestamp = controller.get_timestamp()
            eef_cmd.pose_6d()[:] = target_pose_6d
            eef_cmd.gripper_pos = target_gripper_pos
            eef_cmd.timestamp = current_timestamp + preview_time

            if update_traj:
                controller.set_eef_traj([eef_cmd])
            else:
                controller.set_eef_cmd(eef_cmd)

            output_eef_cmd = controller.get_eef_cmd()
            eef_state = controller.get_eef_state()
            error = output_eef_cmd.pose_6d() - eef_state.pose_6d()
            avg_error += error
            avg_cnt += 1

            send_cnt += 1
            if current_target.sequence_id == latest_seq:
                repeated_target_count += 1
            else:
                latest_seq = current_target.sequence_id
                repeated_target_count = 0

            if log_interval > 0 and send_cnt % log_interval == 0:
                target_age = time.monotonic() - current_target.received_at
                print(
                    f"[vr-teleop] t={time.monotonic() - start_time:.3f}s "
                    f"send={send_cnt} side={current_target.side.value} "
                    f"seq={current_target.sequence_id} "
                    f"target_age={target_age:.4f}s repeat={repeated_target_count} "
                    f"queue_size={target_queue.qsize()} dropped={dropped_target_count} "
                    f"ignored_other_hand={ignored_other_hand_count} "
                    f"vr_pos={np.array2string(current_target.vr_pos, precision=4, suppress_small=True)} "
                    f"pos_delta={np.array2string(current_target.pos_delta, precision=4, suppress_small=True)} "
                    f"rpy_delta={np.array2string(current_target.rpy_delta, precision=4, suppress_small=True)}"
                )
                print(
                    f"[vr-teleop] target_pose_6d="
                    f"{np.array2string(target_pose_6d, precision=4, suppress_small=True)} "
                    f"target_rpy="
                    f"{np.array2string(target_pose_6d[3:], precision=4, suppress_small=True)} "
                    f"gripper={target_gripper_pos:.4f}/{robot_config.gripper_width:.4f}"
                )
                if damping_config.enabled:
                    print(
                        f"[vr-teleop] damping_limited={damping_limited} "
                        f"raw_target_pose_6d="
                        f"{np.array2string(raw_target_pose_6d, precision=4, suppress_small=True)} "
                        f"raw_gripper={raw_target_gripper_pos:.4f}"
                    )
                print(
                    f"[vr-teleop] output_pose_6d="
                    f"{np.array2string(output_eef_cmd.pose_6d(), precision=4, suppress_small=True)} "
                    f"state_pose_6d="
                    f"{np.array2string(eef_state.pose_6d(), precision=4, suppress_small=True)} "
                    f"error={np.array2string(error, precision=4, suppress_small=True)} "
                    f"avg_error={np.array2string(avg_error / max(1, avg_cnt), precision=4, suppress_small=True)}"
                )

    receiver = threading.Thread(target=receive_loop, name="vr-receiver", daemon=True)
    sender = threading.Thread(target=send_loop, name="robot-sender", daemon=True)
    receiver.start()
    sender.start()

    try:
        while receiver.is_alive() and sender.is_alive():
            time.sleep(0.1)
    finally:
        stop_event.set()
        receiver.join(timeout=1.0)
        sender.join(timeout=1.0)


@click.command()
@click.argument("model")
@click.argument("interface")
@click.option("--hand", type=click.Choice(["left", "right"]), default="right", show_default=True)
@click.option("--mocap-host", "--mocap-tcp-host", default="0.0.0.0", show_default=True)
@click.option("--mocap-port", "--mocap-tcp-port", type=int, default=5555, show_default=True)
@click.option(
    "--transport",
    type=click.Choice(["tcp_server", "udp", "tcp_client"]),
    default="tcp_server",
    show_default=True,
)
@click.option("--basis", type=click.Choice(["rfu", "flu"]), default="rfu", show_default=True)
@click.option("--pos-scale", type=float, default=1.0, show_default=True)
@click.option("--ori-scale", type=float, default=1.0, show_default=True)
@click.option("--cmd-dt", type=float, default=0.01, show_default=True)
@click.option("--preview-time", type=float, default=0.05, show_default=True)
@click.option("--single-cmd", is_flag=True, help="Use set_eef_cmd instead of set_eef_traj.")
@click.option("--log-interval", type=int, default=10, show_default=True)
@click.option("--grip-open-dist", type=float, default=0.08, show_default=True)
@click.option("--grip-close-dist", type=float, default=0.02, show_default=True)
@click.option(
    "--damping-protection/--no-damping-protection",
    default=False,
    show_default=True,
    help="Limit sudden VR target jumps before sending commands.",
)
@click.option("--damping-alpha", type=float, default=0.6, show_default=True)
@click.option("--damping-max-pos-step", type=float, default=0.01, show_default=True)
@click.option("--damping-max-ori-step", type=float, default=0.10, show_default=True)
@click.option("--damping-max-gripper-step", type=float, default=0.005, show_default=True)
@click.option(
    "--zero-first-frame/--no-zero-first-frame",
    default=True,
    show_default=True,
    help="Use the first VR frame as zero pose and remove its position/orientation bias.",
)
def main(
    model: str,
    interface: str,
    hand: str,
    mocap_host: str,
    mocap_port: int,
    transport: str,
    basis: str,
    pos_scale: float,
    ori_scale: float,
    cmd_dt: float,
    preview_time: float,
    single_cmd: bool,
    log_interval: int,
    grip_open_dist: float,
    grip_close_dist: float,
    damping_protection: bool,
    damping_alpha: float,
    damping_max_pos_step: float,
    damping_max_ori_step: float,
    damping_max_gripper_step: float,
    zero_first_frame: bool,
) -> None:
    if not 0.0 < damping_alpha <= 1.0:
        raise click.BadParameter("--damping-alpha must be in (0, 1].")
    if (
        damping_max_pos_step <= 0.0
        or damping_max_ori_step <= 0.0
        or damping_max_gripper_step <= 0.0
    ):
        raise click.BadParameter("damping max step values must be positive.")

    robot_config = RobotConfigFactory.get_instance().get_config(model)
    controller_config = ControllerConfigFactory.get_instance().get_config(
        "cartesian_controller", robot_config.joint_dof
    )
    if ARX_IMPORT_ERROR is None:
        controller = Arx5CartesianController(robot_config, controller_config, interface)
    else:
        print(
            "[mock-arx] arx5_interface is unavailable; running mock sender only. "
            f"import_error={ARX_IMPORT_ERROR}"
        )
        controller = MockArx5CartesianController(robot_config, controller_config, interface)
    controller.reset_to_home()
    controller.set_log_level(LogLevel.DEBUG)
    np.set_printoptions(precision=4, suppress=True)

    gain = Gain(robot_config.joint_dof)
    _ = gain

    try:
        start_vr_teleop(
            controller,
            hand_side=_side_from_text(hand),
            mocap_host=mocap_host,
            mocap_port=mocap_port,
            transport_mode=TransportMode(transport),
            basis=basis,
            pos_scale=pos_scale,
            ori_scale=ori_scale,
            cmd_dt=cmd_dt,
            preview_time=preview_time,
            update_traj=not single_cmd,
            log_interval=log_interval,
            grip_config=GripConfig(open_dist=grip_open_dist, close_dist=grip_close_dist),
            damping_config=DampingConfig(
                enabled=damping_protection,
                alpha=damping_alpha,
                max_pos_step=damping_max_pos_step,
                max_ori_step=damping_max_ori_step,
                max_gripper_step=damping_max_gripper_step,
            ),
            zero_first_frame=zero_first_frame,
        )
    except KeyboardInterrupt:
        print("VR teleop is terminated. Resetting to home.")
        controller.reset_to_home()
        controller.set_to_damping()


if __name__ == "__main__":
    main()
