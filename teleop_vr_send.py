from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import click
import numpy as np


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for path in (SRC_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
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
    finger_curl_angles,
    grip_value,
    unity_left_to_flu_position,
    unity_left_to_flu_rotation_matrix,
    unity_left_to_rfu_position,
    unity_left_to_rfu_rotation_matrix,
)
from teleop_vr.postprocess import (  # noqa: E402
    DampingConfig,
    TrajectorySmoother,
    load_damping_config,
    make_rpy_continuous,
    replace_nonfinite_command_values,
)
from teleop_vr.recorder import RecorderConfig, TeleopRecorder  # noqa: E402


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
class TeleopTarget:
    side: HandSide
    sequence_id: int
    vr_pos: np.ndarray
    pos_delta: np.ndarray
    rpy_delta: np.ndarray
    raw_pose_6d: np.ndarray
    raw_gripper_pos: float
    received_at: float
    record_index: int | None = None


class TargetWindow:
    """保存最近几帧 VR 目标，供发送线程按固定时间点插值采样。"""

    def __init__(self, max_size: int) -> None:
        if max_size < 2:
            raise ValueError("max_size must be at least 2")
        self._targets: deque[TeleopTarget] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def append(self, target: TeleopTarget) -> None:
        with self._lock:
            self._targets.append(target)

    def clear(self) -> None:
        with self._lock:
            self._targets.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._targets)

    def sample(self, sample_time: float) -> tuple[TeleopTarget | None, bool]:
        with self._lock:
            targets = list(self._targets)
        if not targets:
            return None, False
        if len(targets) == 1 or sample_time <= targets[0].received_at:
            return targets[0], False
        if sample_time >= targets[-1].received_at:
            return targets[-1], False

        for previous, current in zip(targets, targets[1:], strict=False):
            if previous.received_at <= sample_time <= current.received_at:
                # 在相邻两帧 VR 数据之间按接收时间线性插值，降低网络抖动带来的跳变。
                span = current.received_at - previous.received_at
                ratio = 0.0 if span <= 1e-12 else (sample_time - previous.received_at) / span
                return _interpolate_target(previous, current, ratio, sample_time), True
        return targets[-1], False


def _lerp_array(start: np.ndarray, end: np.ndarray, ratio: float) -> np.ndarray:
    return start + (end - start) * ratio


def _interpolate_target(
    previous: TeleopTarget,
    current: TeleopTarget,
    ratio: float,
    sample_time: float,
) -> TeleopTarget:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    # 姿态角在 +/-pi 附近可能跳变，插值前先把 RPY 展开到连续区间。
    pose_delta = current.raw_pose_6d - previous.raw_pose_6d
    pose_delta[3:] = make_rpy_continuous(current.raw_pose_6d[3:], previous.raw_pose_6d[3:]) - previous.raw_pose_6d[3:]
    return TeleopTarget(
        side=current.side,
        sequence_id=current.sequence_id,
        vr_pos=_lerp_array(previous.vr_pos, current.vr_pos, ratio),
        pos_delta=_lerp_array(previous.pos_delta, current.pos_delta, ratio),
        rpy_delta=_lerp_array(previous.rpy_delta, current.rpy_delta, ratio),
        raw_pose_6d=previous.raw_pose_6d + pose_delta * ratio,
        raw_gripper_pos=previous.raw_gripper_pos
        + (current.raw_gripper_pos - previous.raw_gripper_pos) * ratio,
        received_at=sample_time,
        record_index=current.record_index,
    )


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
    # HTS/Unity 坐标系先转换到机器人控制使用的基坐标系。
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


def _gripper_from_frame(
    frame: HandFrame,
    *,
    robot_gripper_width: float,
    grip_config: GripConfig,
) -> float:
    # 由手指关键点距离计算 0~1 的夹爪开合量，再映射到机器人夹爪宽度。
    grip_ctrl = grip_value(frame, grip_config)
    t = (grip_ctrl - grip_config.ctrl_min) / (grip_config.ctrl_max - grip_config.ctrl_min)
    return float(np.clip(t, 0.0, 1.0) * robot_gripper_width)


def _is_fist_frame(frame: HandFrame, *, curl_threshold: float) -> bool:
    """Return True when the four non-thumb fingers are curled enough."""

    curl = finger_curl_angles(frame, fingers=["index", "middle", "ring", "little"])
    if set(curl) != {"index", "middle", "ring", "little"}:
        return False
    return all(
        bool(angles) and float(np.mean(angles)) >= curl_threshold
        for angles in curl.values()
    )


def _fist_curl_scores(frame: HandFrame) -> dict[str, float]:
    curl = finger_curl_angles(frame, fingers=["index", "middle", "ring", "little"])
    return {
        finger: float(np.mean(angles)) if angles else 0.0
        for finger, angles in curl.items()
    }


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
    interp_window_size: int,
    interp_delay: float,
    preview_time: float,
    update_traj: bool,
    log_interval: int,
    grip_config: GripConfig,
    damping_config: DampingConfig,
    zero_first_frame: bool,
    recorder: TeleopRecorder | None,
) -> None:
    robot_config = controller.get_robot_config()
    home_pose = np.asarray(controller.get_home_pose(), dtype=float).copy()
    target_window = TargetWindow(interp_window_size)
    stop_event = threading.Event()
    client_lock = threading.Lock()
    active_client: HTSClient | None = None
    ignored_other_hand_count = 0
    start_time = time.monotonic()

    print(
        "VR teleop ready. "
        f"hand={hand_side.value} transport={transport_mode.value} "
        f"{mocap_host}:{mocap_port} basis={basis} "
        f"zero_first_frame={zero_first_frame} "
        f"interp_window_size={interp_window_size} interp_delay={interp_delay:.4f}s"
    )
    if recorder is not None and recorder.enabled:
        print("Recording enabled. CSV files will be saved after keyboard interrupt.")
    if damping_config.enabled:
        print(
            "Damping protection enabled: "
            f"alpha={damping_config.alpha:.3f} "
            f"max_pos_step={damping_config.max_pos_step:.4f}m "
            f"max_ori_step={damping_config.max_ori_step:.4f}rad "
            f"max_gripper_step={damping_config.max_gripper_step:.4f}m"
        )
        print(
            "Motion limits: "
            f"pos_vel={damping_config.max_pos_velocity:.4f}m/s "
            f"ori_vel={damping_config.max_ori_velocity:.4f}rad/s "
            f"gripper_vel={damping_config.max_gripper_velocity:.4f}m/s "
            f"pos_acc={damping_config.max_pos_acceleration:.4f}m/s^2 "
            f"ori_acc={damping_config.max_ori_acceleration:.4f}rad/s^2 "
            f"gripper_acc={damping_config.max_gripper_acceleration:.4f}m/s^2 "
            f"pos_jerk={damping_config.max_pos_jerk:.4f}m/s^3 "
            f"ori_jerk={damping_config.max_ori_jerk:.4f}rad/s^3 "
            f"gripper_jerk={damping_config.max_gripper_jerk:.4f}m/s^3"
        )
        if damping_config.pose_min is not None or damping_config.pose_max is not None:
            print(
                "Pose software limits: "
                f"min={damping_config.pose_min} max={damping_config.pose_max}"
            )
        print(
            "Gripper software limits: "
            f"min={damping_config.gripper_min:.4f} "
            f"max={damping_config.gripper_max}"
        )
    print("Waiting for the first matching VR hand frame as reference pose.")

    def receive_loop() -> None:
        nonlocal active_client, ignored_other_hand_count
        reference: VrReference | None = None
        last_raw_rpy: np.ndarray | None = None
        # 接收线程只负责读取 HTS 帧、转换到机器人目标位姿，并写入滑动窗口。
        # 真正的发送频率由 send_loop 的 cmd_dt 控制，两边解耦以吸收网络抖动。
        client = HTSClient(
            HTSClientConfig(
                transport_mode=transport_mode,
                host=mocap_host,
                port=mocap_port,
                output=StreamOutput.FRAMES,
                error_policy=ErrorPolicy.TOLERANT,
            )
        )
        with client_lock:
            active_client = client

        try:
            for event in client.iter_events():
                if stop_event.is_set():
                    break
                if not isinstance(event, HandFrame):
                    continue
                if event.side != hand_side:
                    ignored_other_hand_count += 1
                    continue

                # 原始 VR 帧先记录到 raw.csv，便于事后分析网络延迟和丢帧情况。
                received_at = time.monotonic()
                record_index = (
                    recorder.add_frame(event, received_at=received_at)
                    if recorder is not None
                    else None
                )
                vr_pos, vr_rot = _frame_pose(event, basis=basis)
                if reference is None:
                    # 第一帧匹配手作为参考零点；之后的位置/姿态默认都相对它计算。
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
                    # 默认模式：VR 的相对运动叠加到机器人 home pose 上，避免绝对坐标偏置。
                    pos_delta = (vr_pos - reference.pos) * pos_scale
                    rot_delta = vr_rot @ reference.rot.T
                    base_pose = reference.robot_home_pose
                else:
                    # 调试模式：直接使用转换后的 VR 绝对坐标，不消除第一帧偏置。
                    pos_delta = vr_pos * pos_scale
                    rot_delta = vr_rot
                    base_pose = reference.robot_home_pose
                rpy_delta = _rotmat_to_rpy(rot_delta) * ori_scale
                target_rot = _rpy_to_rotmat(base_pose[3:])
                if ori_scale != 0.0:
                    target_rot = _rpy_to_rotmat(rpy_delta) @ target_rot

                raw_pose_6d = base_pose.copy()
                raw_pose_6d[:3] = base_pose[:3] + pos_delta
                raw_pose_6d[3:] = make_rpy_continuous(
                    _rotmat_to_rpy(target_rot),
                    last_raw_rpy,
                )
                last_raw_rpy = raw_pose_6d[3:].copy()
                raw_gripper_pos = _gripper_from_frame(
                    event,
                    robot_gripper_width=robot_config.gripper_width,
                    grip_config=grip_config,
                )

                target = TeleopTarget(
                    side=event.side,
                    sequence_id=event.sequence_id,
                    vr_pos=vr_pos.copy(),
                    pos_delta=pos_delta.copy(),
                    rpy_delta=rpy_delta.copy(),
                    raw_pose_6d=raw_pose_6d,
                    raw_gripper_pos=raw_gripper_pos,
                    received_at=received_at,
                    record_index=record_index,
                )
                if recorder is not None:
                    # 这里仅缓存“转换后、后处理前”的目标；真正发出去的命令在发送线程记录。
                    recorder.add_converted_target(
                        record_index=target.record_index,
                        side=target.side.value,
                        sequence_id=target.sequence_id,
                        received_at=target.received_at,
                        vr_pos=target.vr_pos,
                        pos_delta=target.pos_delta,
                        rpy_delta=target.rpy_delta,
                        raw_pose_6d=target.raw_pose_6d,
                        raw_gripper_pos=target.raw_gripper_pos,
                    )
                target_window.append(target)
        except Exception as exc:
            if not stop_event.is_set():
                stop_event.set()
                print(f"[vr-teleop] receive thread stopped by error: {exc!r}")
        finally:
            client.close()
            with client_lock:
                if active_client is client:
                    active_client = None

    def send_loop() -> None:
        eef_cmd = EEFState()
        smoother = TrajectorySmoother(damping_config, cmd_dt=cmd_dt)
        current_target: TeleopTarget | None = None
        last_sent_rpy: np.ndarray | None = None
        last_valid_sent_pose_6d: np.ndarray | None = None
        last_valid_sent_gripper_pos: float | None = None
        avg_error = np.zeros(6)
        avg_cnt = 0
        send_cnt = 0
        next_send_time = time.monotonic() + cmd_dt
        latest_seq = -1
        repeated_target_count = 0
        reused_target_count = 0
        interpolated_target_count = 0

        while not stop_event.is_set():
            # 按 cmd_dt 固定节拍发送。若线程被系统调度延迟，只跳到下一个周期，不补发历史命令。
            now = time.monotonic()
            sleep_time = next_send_time - now
            if sleep_time > 0.0:
                time.sleep(min(sleep_time, cmd_dt))
                now = time.monotonic()
            if now >= next_send_time:
                missed_periods = int((now - next_send_time) // cmd_dt)
                next_send_time += (missed_periods + 1) * cmd_dt

            # 从最近的 VR 滑动窗口中取 now - interp_delay 的目标，
            # 留出一点时间缓冲，使多数采样点可以落在两帧之间完成插值。
            sampled_target, interpolated_target = target_window.sample(now - interp_delay)
            reused_current_target = sampled_target is None and current_target is not None
            if sampled_target is not None:
                current_target = sampled_target

            if current_target is None:
                continue

            raw_target_pose_6d = current_target.raw_pose_6d.copy()
            raw_target_gripper_pos = current_target.raw_gripper_pos
            # 后处理包含 YAML 中配置的平滑、速度/加速度/jerk 限制和软件限位。
            smoothed_target = smoother.process(raw_target_pose_6d, raw_target_gripper_pos)
            target_pose_6d = smoothed_target.pose_6d
            target_gripper_pos = smoothed_target.gripper_pos
            target_pose_6d[3:] = make_rpy_continuous(target_pose_6d[3:], last_sent_rpy)
            try:
                # 如果某一帧出现 NaN/Inf，用上一帧有效发送值兜底，避免把非法值发给硬件。
                target_pose_6d, target_gripper_pos, nonfinite_replaced = (
                    replace_nonfinite_command_values(
                        target_pose_6d,
                        target_gripper_pos,
                        last_valid_sent_pose_6d,
                        last_valid_sent_gripper_pos,
                    )
                )
            except ValueError as exc:
                print(
                    "[vr-teleop] skipped non-finite command before first valid send: "
                    f"{exc}"
                )
                continue

            if nonfinite_replaced:
                print(
                    "[vr-teleop] replaced non-finite command values with previous sent command "
                    f"seq={current_target.sequence_id}"
                )
            last_sent_rpy = target_pose_6d[3:].copy()

            current_timestamp = controller.get_timestamp()
            eef_cmd.pose_6d()[:] = target_pose_6d
            eef_cmd.gripper_pos = target_gripper_pos
            eef_cmd.timestamp = current_timestamp + preview_time
            sent_at = time.monotonic()
            if recorder is not None:
                # converted.csv 记录实际发送节拍下的命令，频率等于 cmd_dt 对应的发送频率。
                recorder.mark_sent(
                    record_index=current_target.record_index,
                    sent_at=sent_at,
                    sent_pose_6d=target_pose_6d,
                    sent_gripper_pos=target_gripper_pos,
                )

            if update_traj:
                controller.set_eef_traj([eef_cmd])
            else:
                controller.set_eef_cmd(eef_cmd)
            last_valid_sent_pose_6d = target_pose_6d.copy()
            last_valid_sent_gripper_pos = float(target_gripper_pos)

            output_eef_cmd = controller.get_eef_cmd()
            eef_state = controller.get_eef_state()
            error = output_eef_cmd.pose_6d() - eef_state.pose_6d()
            if recorder is not None:
                # 只有真实 arx5_interface 可用时才会保存 robot_state.csv；mock 模式不落该文件。
                recorder.add_robot_state(
                    sent_at=sent_at,
                    state_pose_6d=eef_state.pose_6d(),
                    state_gripper_pos=eef_state.gripper_pos,
                    cmd_pose_6d=output_eef_cmd.pose_6d(),
                    cmd_gripper_pos=output_eef_cmd.gripper_pos,
                )
            avg_error += error
            avg_cnt += 1

            send_cnt += 1
            if current_target.sequence_id == latest_seq:
                repeated_target_count += 1
            else:
                latest_seq = current_target.sequence_id
                repeated_target_count = 0
            if reused_current_target:
                reused_target_count += 1
            if interpolated_target:
                interpolated_target_count += 1

            if False and log_interval > 0 and send_cnt % log_interval == 0:
                target_age = time.monotonic() - current_target.received_at
                print(
                    f"[vr-teleop] t={time.monotonic() - start_time:.3f}s "
                    f"send={send_cnt} side={current_target.side.value} "
                    f"seq={current_target.sequence_id} "
                    f"target_age={target_age:.4f}s repeat={repeated_target_count} "
                    f"window_size={target_window.size()} reused={reused_target_count} "
                    f"interpolated={interpolated_target_count} "
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
                        f"[vr-teleop] damping_limited={smoothed_target.limited} "
                        f"step_limited={smoothed_target.step_limited} "
                        f"velocity_limited={smoothed_target.velocity_limited} "
                        f"acceleration_limited={smoothed_target.acceleration_limited} "
                        f"jerk_limited={smoothed_target.jerk_limited} "
                        f"command_limited={smoothed_target.command_limited} "
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
        with client_lock:
            client = active_client
        if client is not None:
            client.close()
        receiver.join(timeout=1.0)
        sender.join(timeout=1.0)


@dataclass
class ArmRuntime:
    name: str
    side: HandSide
    controller: Arx5CartesianController
    robot_config: Any
    home_pose: np.ndarray
    target_window: TargetWindow
    damping_config: DampingConfig
    recorder: TeleopRecorder | None
    grip_config: GripConfig
    reference: VrReference | None = None
    last_raw_rpy: np.ndarray | None = None
    ignored_other_hand_count: int = 0


def _make_controller(model: str, interface: str) -> tuple[Arx5CartesianController, Any]:
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
    return controller, robot_config


def _resolve_record_dir(record_dir: str) -> str:
    if os.path.isabs(record_dir):
        return record_dir
    return os.path.join(ROOT_DIR, record_dir)


def _make_recorder(
    *,
    enabled: bool,
    output_dir: str,
    prefix: str | None,
) -> TeleopRecorder:
    return TeleopRecorder(
        RecorderConfig(
            enabled=enabled,
            output_dir=output_dir,
            prefix=prefix,
            record_robot_state=ARX_IMPORT_ERROR is None,
        )
    )


def _load_damping_for_robot(path: str | None, robot_config: Any) -> DampingConfig:
    config = load_damping_config(path)
    if config.gripper_max is None:
        config.gripper_max = robot_config.gripper_width
    return config


def _update_arm_target_from_frame(
    arm: ArmRuntime,
    frame: HandFrame,
    *,
    basis: str,
    pos_scale: float,
    ori_scale: float,
    zero_first_frame: bool,
    received_at: float,
) -> None:
    record_index = (
        arm.recorder.add_frame(frame, received_at=received_at)
        if arm.recorder is not None
        else None
    )
    vr_pos, vr_rot = _frame_pose(frame, basis=basis)
    if arm.reference is None:
        arm.reference = VrReference(
            pos=vr_pos.copy(),
            rot=vr_rot.copy(),
            robot_home_pose=arm.home_pose.copy(),
        )
        print(
            f"[{arm.name}] reference captured: "
            f"vr_pos={np.array2string(arm.reference.pos, precision=4, suppress_small=True)} "
            f"vr_rpy={np.array2string(_rotmat_to_rpy(arm.reference.rot), precision=4, suppress_small=True)} "
            f"home_pose={np.array2string(arm.reference.robot_home_pose, precision=4, suppress_small=True)}"
        )
        return

    if zero_first_frame:
        pos_delta = (vr_pos - arm.reference.pos) * pos_scale
        rot_delta = vr_rot @ arm.reference.rot.T
        base_pose = arm.reference.robot_home_pose
    else:
        pos_delta = vr_pos * pos_scale
        rot_delta = vr_rot
        base_pose = arm.reference.robot_home_pose

    rpy_delta = _rotmat_to_rpy(rot_delta) * ori_scale
    target_rot = _rpy_to_rotmat(base_pose[3:])
    if ori_scale != 0.0:
        target_rot = _rpy_to_rotmat(rpy_delta) @ target_rot

    raw_pose_6d = base_pose.copy()
    raw_pose_6d[:3] = base_pose[:3] + pos_delta
    raw_pose_6d[3:] = make_rpy_continuous(_rotmat_to_rpy(target_rot), arm.last_raw_rpy)
    arm.last_raw_rpy = raw_pose_6d[3:].copy()
    raw_gripper_pos = _gripper_from_frame(
        frame,
        robot_gripper_width=arm.robot_config.gripper_width,
        grip_config=arm.grip_config,
    )

    target = TeleopTarget(
        side=frame.side,
        sequence_id=frame.sequence_id,
        vr_pos=vr_pos.copy(),
        pos_delta=pos_delta.copy(),
        rpy_delta=rpy_delta.copy(),
        raw_pose_6d=raw_pose_6d,
        raw_gripper_pos=raw_gripper_pos,
        received_at=received_at,
        record_index=record_index,
    )
    if arm.recorder is not None:
        arm.recorder.add_converted_target(
            record_index=target.record_index,
            side=target.side.value,
            sequence_id=target.sequence_id,
            received_at=target.received_at,
            vr_pos=target.vr_pos,
            pos_delta=target.pos_delta,
            rpy_delta=target.rpy_delta,
            raw_pose_6d=target.raw_pose_6d,
            raw_gripper_pos=target.raw_gripper_pos,
        )
    arm.target_window.append(target)


def _dual_receive_loop(
    *,
    arms_by_side: dict[HandSide, ArmRuntime],
    mocap_host: str,
    mocap_port: int,
    transport_mode: TransportMode,
    basis: str,
    pos_scale: float,
    ori_scale: float,
    zero_first_frame: bool,
    fist_reset_hold_s: float,
    fist_curl_threshold: float,
    stop_event: threading.Event,
    client_lock: threading.Lock,
    active_client: list[HTSClient | None],
) -> None:
    client = HTSClient(
        HTSClientConfig(
            transport_mode=transport_mode,
            host=mocap_host,
            port=mocap_port,
            output=StreamOutput.FRAMES,
            error_policy=ErrorPolicy.TOLERANT,
        )
    )
    with client_lock:
        active_client[0] = client

    fist_by_side = {HandSide.LEFT: False, HandSide.RIGHT: False}
    both_fist_since: float | None = None
    fist_reset_armed = True
    both_fist_logged = False

    try:
        for event in client.iter_events():
            if stop_event.is_set():
                break
            if not isinstance(event, HandFrame):
                continue
            arm = arms_by_side.get(event.side)
            if arm is None:
                for runtime in arms_by_side.values():
                    runtime.ignored_other_hand_count += 1
                continue
            if fist_reset_hold_s > 0.0:
                is_fist = _is_fist_frame(
                    event,
                    curl_threshold=fist_curl_threshold,
                )
                fist_by_side[event.side] = is_fist
                curl_scores = _fist_curl_scores(event)
                now = time.monotonic()
                both_fist = fist_by_side[HandSide.LEFT] and fist_by_side[HandSide.RIGHT]
                print(
                    "[dual-vr-teleop] fist_frame "
                    f"side={event.side.value} seq={event.sequence_id} "
                    f"is_fist={is_fist} both_fist={both_fist} "
                    f"curl={curl_scores} threshold={fist_curl_threshold:.3f}"
                )
                if both_fist:
                    if both_fist_since is None:
                        both_fist_since = now
                        both_fist_logged = True
                        print(
                            "[dual-vr-teleop] both hands are fists; "
                            f"holding for {fist_reset_hold_s:.2f}s will reset VR reference."
                        )
                    elif fist_reset_armed and now - both_fist_since >= fist_reset_hold_s:
                        for runtime in arms_by_side.values():
                            runtime.reference = None
                            runtime.last_raw_rpy = None
                            runtime.target_window.clear()
                        fist_reset_armed = False
                        print(
                            "[dual-vr-teleop] both hands held as fists for "
                            f"{fist_reset_hold_s:.2f}s; VR references will be recaptured."
                        )
                        continue
                else:
                    if both_fist_logged:
                        print("[dual-vr-teleop] fist reset gesture released.")
                    both_fist_since = None
                    fist_reset_armed = True
                    both_fist_logged = False
            _update_arm_target_from_frame(
                arm,
                event,
                basis=basis,
                pos_scale=pos_scale,
                ori_scale=ori_scale,
                zero_first_frame=zero_first_frame,
                received_at=time.monotonic(),
            )
    except Exception as exc:
        if not stop_event.is_set():
            stop_event.set()
            print(f"[dual-vr-teleop] receive thread stopped by error: {exc!r}")
    finally:
        client.close()
        with client_lock:
            if active_client[0] is client:
                active_client[0] = None


def _dual_send_loop(
    arm: ArmRuntime,
    *,
    cmd_dt: float,
    interp_delay: float,
    preview_time: float,
    update_traj: bool,
    log_interval: int,
    stop_event: threading.Event,
    start_time: float,
) -> None:
    eef_cmd = EEFState()
    smoother = TrajectorySmoother(arm.damping_config, cmd_dt=cmd_dt)
    current_target: TeleopTarget | None = None
    last_sent_rpy: np.ndarray | None = None
    last_valid_sent_pose_6d: np.ndarray | None = None
    last_valid_sent_gripper_pos: float | None = None
    avg_error = np.zeros(6)
    avg_cnt = 0
    send_cnt = 0
    latest_seq = -1
    repeated_target_count = 0
    reused_target_count = 0
    interpolated_target_count = 0
    next_send_time = time.monotonic() + cmd_dt

    while not stop_event.is_set():
        now = time.monotonic()
        sleep_time = next_send_time - now
        if sleep_time > 0.0:
            time.sleep(min(sleep_time, cmd_dt))
            now = time.monotonic()
        if now >= next_send_time:
            missed_periods = int((now - next_send_time) // cmd_dt)
            next_send_time += (missed_periods + 1) * cmd_dt

        sampled_target, interpolated_target = arm.target_window.sample(now - interp_delay)
        reused_current_target = sampled_target is None and current_target is not None
        if sampled_target is not None:
            current_target = sampled_target
        if current_target is None:
            continue

        raw_target_pose_6d = current_target.raw_pose_6d.copy()
        raw_target_gripper_pos = current_target.raw_gripper_pos
        smoothed_target = smoother.process(raw_target_pose_6d, raw_target_gripper_pos)
        target_pose_6d = smoothed_target.pose_6d
        target_gripper_pos = smoothed_target.gripper_pos
        target_pose_6d[3:] = make_rpy_continuous(target_pose_6d[3:], last_sent_rpy)
        try:
            target_pose_6d, target_gripper_pos, nonfinite_replaced = (
                replace_nonfinite_command_values(
                    target_pose_6d,
                    target_gripper_pos,
                    last_valid_sent_pose_6d,
                    last_valid_sent_gripper_pos,
                )
            )
        except ValueError as exc:
            print(f"[{arm.name}] skipped non-finite command before first valid send: {exc}")
            continue

        if nonfinite_replaced:
            print(f"[{arm.name}] replaced non-finite command values with previous sent command")
        last_sent_rpy = target_pose_6d[3:].copy()

        current_timestamp = arm.controller.get_timestamp()
        eef_cmd.pose_6d()[:] = target_pose_6d
        eef_cmd.gripper_pos = target_gripper_pos
        eef_cmd.timestamp = current_timestamp + preview_time
        sent_at = time.monotonic()
        if arm.recorder is not None:
            arm.recorder.mark_sent(
                record_index=current_target.record_index,
                sent_at=sent_at,
                sent_pose_6d=target_pose_6d,
                sent_gripper_pos=target_gripper_pos,
            )

        if update_traj:
            arm.controller.set_eef_traj([eef_cmd])
        else:
            arm.controller.set_eef_cmd(eef_cmd)
        last_valid_sent_pose_6d = target_pose_6d.copy()
        last_valid_sent_gripper_pos = float(target_gripper_pos)

        output_eef_cmd = arm.controller.get_eef_cmd()
        eef_state = arm.controller.get_eef_state()
        error = output_eef_cmd.pose_6d() - eef_state.pose_6d()
        if arm.recorder is not None:
            arm.recorder.add_robot_state(
                sent_at=sent_at,
                state_pose_6d=eef_state.pose_6d(),
                state_gripper_pos=eef_state.gripper_pos,
                cmd_pose_6d=output_eef_cmd.pose_6d(),
                cmd_gripper_pos=output_eef_cmd.gripper_pos,
            )
        avg_error += error
        avg_cnt += 1

        send_cnt += 1
        if current_target.sequence_id == latest_seq:
            repeated_target_count += 1
        else:
            latest_seq = current_target.sequence_id
            repeated_target_count = 0
        if reused_current_target:
            reused_target_count += 1
        if interpolated_target:
            interpolated_target_count += 1

        if False and log_interval > 0 and send_cnt % log_interval == 0:
            target_age = time.monotonic() - current_target.received_at
            print(
                f"[{arm.name}] t={time.monotonic() - start_time:.3f}s "
                f"send={send_cnt} side={current_target.side.value} "
                f"seq={current_target.sequence_id} target_age={target_age:.4f}s "
                f"repeat={repeated_target_count} window_size={arm.target_window.size()} "
                f"reused={reused_target_count} interpolated={interpolated_target_count} "
                f"ignored_other_hand={arm.ignored_other_hand_count} "
                f"pose={np.array2string(target_pose_6d, precision=4, suppress_small=True)} "
                f"gripper={target_gripper_pos:.4f}/{arm.robot_config.gripper_width:.4f} "
                f"avg_error={np.array2string(avg_error / max(1, avg_cnt), precision=4, suppress_small=True)}"
            )


def _save_recorder(name: str, recorder: TeleopRecorder | None) -> None:
    if recorder is None:
        return
    saved_paths = recorder.save()
    if saved_paths is None:
        return
    raw_path, converted_path, robot_state_path = saved_paths
    print(f"[{name}] Saved VR raw records to {raw_path}")
    print(f"[{name}] Saved converted records to {converted_path}")
    if robot_state_path is not None:
        print(f"[{name}] Saved robot state records to {robot_state_path}")


def start_dual_vr_teleop(
    left_arm: ArmRuntime,
    right_arm: ArmRuntime,
    *,
    mocap_host: str,
    mocap_port: int,
    transport_mode: TransportMode,
    basis: str,
    pos_scale: float,
    ori_scale: float,
    cmd_dt: float,
    interp_delay: float,
    preview_time: float,
    update_traj: bool,
    log_interval: int,
    zero_first_frame: bool,
    fist_reset_hold_s: float,
    fist_curl_threshold: float,
) -> None:
    stop_event = threading.Event()
    client_lock = threading.Lock()
    active_client: list[HTSClient | None] = [None]
    start_time = time.monotonic()
    arms_by_side = {HandSide.LEFT: left_arm, HandSide.RIGHT: right_arm}

    print(
        "Dual VR teleop ready. "
        f"left={left_arm.name} right={right_arm.name} "
        f"transport={transport_mode.value} {mocap_host}:{mocap_port} basis={basis} "
        f"zero_first_frame={zero_first_frame}"
    )
    for arm in (left_arm, right_arm):
        if arm.recorder is not None and arm.recorder.enabled:
            print(f"[{arm.name}] Recording enabled. CSV files will be saved after keyboard interrupt.")

    receiver = threading.Thread(
        target=_dual_receive_loop,
        kwargs={
            "arms_by_side": arms_by_side,
            "mocap_host": mocap_host,
            "mocap_port": mocap_port,
            "transport_mode": transport_mode,
            "basis": basis,
            "pos_scale": pos_scale,
            "ori_scale": ori_scale,
            "zero_first_frame": zero_first_frame,
            "fist_reset_hold_s": fist_reset_hold_s,
            "fist_curl_threshold": fist_curl_threshold,
            "stop_event": stop_event,
            "client_lock": client_lock,
            "active_client": active_client,
        },
        name="dual-vr-receiver",
        daemon=True,
    )
    left_sender = threading.Thread(
        target=_dual_send_loop,
        kwargs={
            "arm": left_arm,
            "cmd_dt": cmd_dt,
            "interp_delay": interp_delay,
            "preview_time": preview_time,
            "update_traj": update_traj,
            "log_interval": log_interval,
            "stop_event": stop_event,
            "start_time": start_time,
        },
        name="left-robot-sender",
        daemon=True,
    )
    right_sender = threading.Thread(
        target=_dual_send_loop,
        kwargs={
            "arm": right_arm,
            "cmd_dt": cmd_dt,
            "interp_delay": interp_delay,
            "preview_time": preview_time,
            "update_traj": update_traj,
            "log_interval": log_interval,
            "stop_event": stop_event,
            "start_time": start_time,
        },
        name="right-robot-sender",
        daemon=True,
    )

    receiver.start()
    left_sender.start()
    right_sender.start()

    try:
        while receiver.is_alive() and left_sender.is_alive() and right_sender.is_alive():
            time.sleep(0.1)
    finally:
        stop_event.set()
        with client_lock:
            client = active_client[0]
        if client is not None:
            client.close()
        receiver.join(timeout=1.0)
        left_sender.join(timeout=1.0)
        right_sender.join(timeout=1.0)


@click.command()
@click.argument("robot_args", nargs=-1)
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
@click.option("--interp-window-size", type=int, default=8, show_default=True)
@click.option("--interp-delay", type=float, default=0.02, show_default=True)
@click.option("--preview-time", type=float, default=0.05, show_default=True)
@click.option("--single-cmd", is_flag=True, help="Use set_eef_cmd instead of set_eef_traj.")
@click.option("--log-interval", type=int, default=10, show_default=True)
@click.option(
    "--record/--no-record",
    default=True,
    show_default=True,
    help="Record VR raw and converted data to CSV on Ctrl+C.",
)
@click.option("--record-dir", default="./records", show_default=True)
@click.option("--record-prefix", default=None)
@click.option("--left-record-prefix", default=None, help="CSV prefix for the left arm in dual-arm mode.")
@click.option("--right-record-prefix", default=None, help="CSV prefix for the right arm in dual-arm mode.")
@click.option("--grip-open-dist", type=float, default=0.08, show_default=True)
@click.option("--grip-close-dist", type=float, default=0.02, show_default=True)
@click.option(
    "--fist-reset-hold-s",
    type=float,
    default=1.0,
    show_default=True,
    help="Dual-arm mode: recapture VR references after both hands hold fists for this many seconds; use 0 to disable.",
)
@click.option(
    "--fist-curl-threshold",
    type=float,
    default=1.0,
    show_default=True,
    help="Dual-arm mode: average finger curl angle in radians used to classify one hand as a fist.",
)
@click.option(
    "--postprocess-config",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="YAML file for trajectory smoothing/limits.",
)
@click.option(
    "--zero-first-frame/--no-zero-first-frame",
    default=True,
    show_default=True,
    help="Use the first VR frame as zero pose and remove its position/orientation bias.",
)
def main(
    robot_args: tuple[str, ...],
    hand: str,
    mocap_host: str,
    mocap_port: int,
    transport: str,
    basis: str,
    pos_scale: float,
    ori_scale: float,
    cmd_dt: float,
    interp_window_size: int,
    interp_delay: float,
    preview_time: float,
    single_cmd: bool,
    log_interval: int,
    record: bool,
    record_dir: str,
    record_prefix: str | None,
    left_record_prefix: str | None,
    right_record_prefix: str | None,
    grip_open_dist: float,
    grip_close_dist: float,
    fist_reset_hold_s: float,
    fist_curl_threshold: float,
    postprocess_config: str | None,
    zero_first_frame: bool,
) -> None:
    if len(robot_args) not in (2, 4):
        raise click.UsageError(
            "Expected MODEL INTERFACE for single-arm mode, or "
            "LEFT_MODEL LEFT_INTERFACE RIGHT_MODEL RIGHT_INTERFACE for dual-arm mode."
        )
    if cmd_dt <= 0.0:
        raise click.BadParameter("--cmd-dt must be positive.")
    if interp_window_size < 2:
        raise click.BadParameter("--interp-window-size must be at least 2.")
    if interp_delay < 0.0:
        raise click.BadParameter("--interp-delay must be >= 0.")
    if fist_reset_hold_s < 0.0:
        raise click.BadParameter("--fist-reset-hold-s must be >= 0.")
    if fist_curl_threshold < 0.0:
        raise click.BadParameter("--fist-curl-threshold must be >= 0.")
    try:
        damping_config = load_damping_config(postprocess_config)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--postprocess-config") from exc

    np.set_printoptions(precision=4, suppress=True)
    resolved_record_dir = _resolve_record_dir(record_dir)

    if len(robot_args) == 4:
        left_model, left_interface, right_model, right_interface = robot_args
        if left_interface == right_interface:
            raise click.BadParameter(
                "left_interface and right_interface must be different.",
                param_hint="robot_args",
            )

        left_controller, left_robot_config = _make_controller(left_model, left_interface)
        right_controller, right_robot_config = _make_controller(right_model, right_interface)
        left_prefix = left_record_prefix or (
            f"{record_prefix}_left" if record_prefix is not None else "dual_left"
        )
        right_prefix = right_record_prefix or (
            f"{record_prefix}_right" if record_prefix is not None else "dual_right"
        )
        grip_config = GripConfig(open_dist=grip_open_dist, close_dist=grip_close_dist)
        left_arm = ArmRuntime(
            name="left",
            side=HandSide.LEFT,
            controller=left_controller,
            robot_config=left_robot_config,
            home_pose=np.asarray(left_controller.get_home_pose(), dtype=float).copy(),
            target_window=TargetWindow(interp_window_size),
            damping_config=_load_damping_for_robot(postprocess_config, left_robot_config),
            recorder=_make_recorder(
                enabled=record,
                output_dir=resolved_record_dir,
                prefix=left_prefix,
            ),
            grip_config=grip_config,
        )
        right_arm = ArmRuntime(
            name="right",
            side=HandSide.RIGHT,
            controller=right_controller,
            robot_config=right_robot_config,
            home_pose=np.asarray(right_controller.get_home_pose(), dtype=float).copy(),
            target_window=TargetWindow(interp_window_size),
            damping_config=_load_damping_for_robot(postprocess_config, right_robot_config),
            recorder=_make_recorder(
                enabled=record,
                output_dir=resolved_record_dir,
                prefix=right_prefix,
            ),
            grip_config=grip_config,
        )
        try:
            start_dual_vr_teleop(
                left_arm,
                right_arm,
                mocap_host=mocap_host,
                mocap_port=mocap_port,
                transport_mode=TransportMode(transport),
                basis=basis,
                pos_scale=pos_scale,
                ori_scale=ori_scale,
                cmd_dt=cmd_dt,
                interp_delay=interp_delay,
                preview_time=preview_time,
                update_traj=not single_cmd,
                log_interval=log_interval,
                zero_first_frame=zero_first_frame,
                fist_reset_hold_s=fist_reset_hold_s,
                fist_curl_threshold=fist_curl_threshold,
            )
        except KeyboardInterrupt:
            print("Dual VR teleop is terminated. Resetting both arms to home.")
        finally:
            for arm in (left_arm, right_arm):
                arm.controller.reset_to_home()
                arm.controller.set_to_damping()
                _save_recorder(arm.name, arm.recorder)
        return

    model, interface = robot_args
    robot_config = RobotConfigFactory.get_instance().get_config(model)
    if damping_config.gripper_max is None:
        damping_config.gripper_max = robot_config.gripper_width
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

    gain = Gain(robot_config.joint_dof)
    _ = gain
    recorder = _make_recorder(
        enabled=record,
        output_dir=resolved_record_dir,
        prefix=record_prefix,
    )

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
            interp_window_size=interp_window_size,
            interp_delay=interp_delay,
            preview_time=preview_time,
            update_traj=not single_cmd,
            log_interval=log_interval,
            grip_config=GripConfig(open_dist=grip_open_dist, close_dist=grip_close_dist),
            damping_config=damping_config,
            zero_first_frame=zero_first_frame,
            recorder=recorder,
        )
    except KeyboardInterrupt:
        print("VR teleop is terminated. Resetting to home.")
        controller.reset_to_home()
        controller.set_to_damping()
        # Ctrl+C 后一次性落盘，避免实时发送循环里频繁磁盘 IO 影响控制周期。
        saved_paths = recorder.save()
        if saved_paths is not None:
            raw_path, converted_path, robot_state_path = saved_paths
            print(f"Saved VR raw records to {raw_path}")
            print(f"Saved converted records to {converted_path}")
            if robot_state_path is not None:
                print(f"Saved robot state records to {robot_state_path}")


if __name__ == "__main__":
    main()
