from __future__ import annotations

import csv
import os
import threading
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from hand_tracking_sdk import HandFrame


@dataclass
class RecorderConfig:
    """Runtime options for optional CSV recording."""

    enabled: bool
    output_dir: str
    prefix: str | None = None
    record_robot_state: bool = False


@dataclass
class RawVrRecord:
    """One unmodified HTS hand frame, flattened for the *_vr_raw.csv file."""

    record_index: int
    side: str
    sequence_id: int
    frame_id: str
    # Local monotonic timestamp captured by teleop_vr_send.py when the frame is received.
    received_at_monotonic_s: float
    # SDK/HTS timing metadata copied through unchanged for offline jitter analysis.
    recv_ts_ns: int
    recv_time_unix_ns: int | None
    source_ts_ns: int | None
    source_frame_seq: int | None
    wrist_recv_ts_ns: int
    landmarks_recv_ts_ns: int
    wrist_x: float
    wrist_y: float
    wrist_z: float
    wrist_qx: float
    wrist_qy: float
    wrist_qz: float
    wrist_qw: float
    landmark_values: tuple[float, ...]


@dataclass
class ConvertedRecord:
    """One command actually sent by the sender, with the source converted target."""

    record_index: int
    side: str
    sequence_id: int
    sent_at_monotonic_s: float
    vr_pos: np.ndarray
    pos_delta: np.ndarray
    rpy_delta: np.ndarray
    raw_pose_6d: np.ndarray
    raw_gripper_pos: float
    sent_pose_6d: np.ndarray
    sent_gripper_pos: float


@dataclass
class RobotStateRecord:
    """One real robot/controller feedback sample recorded after a sent command."""

    sent_at_monotonic_s: float
    state_pose_6d: np.ndarray
    state_gripper_pos: float
    cmd_pose_6d: np.ndarray
    cmd_gripper_pos: float
    error_pose_6d: np.ndarray
    error_gripper_pos: float


class TeleopRecorder:
    """Collect teleop records in memory and write paired CSV files on shutdown."""

    def __init__(self, config: RecorderConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._raw_records: list[RawVrRecord] = []
        self._converted_records: list[ConvertedRecord] = []
        self._robot_state_records: list[RobotStateRecord] = []
        # Converted targets are cached by raw record_index until the sender consumes them.
        self._converted_targets_by_index: dict[int, dict[str, object]] = {}
        self._next_index = 0

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def add_frame(self, frame: HandFrame, *, received_at: float) -> int | None:
        """Store one raw frame and return its index for downstream converted data."""

        if not self.enabled:
            return None

        wrist = frame.wrist
        # Keep all 21 landmarks in the raw CSV while preserving HTS stream order.
        landmark_values = tuple(coord for point in frame.landmarks.points for coord in point)
        with self._lock:
            record_index = self._next_index
            self._next_index += 1
            self._raw_records.append(
                RawVrRecord(
                    record_index=record_index,
                    side=frame.side.value,
                    sequence_id=frame.sequence_id,
                    frame_id=frame.frame_id,
                    received_at_monotonic_s=received_at,
                    recv_ts_ns=frame.recv_ts_ns,
                    recv_time_unix_ns=frame.recv_time_unix_ns,
                    source_ts_ns=frame.source_ts_ns,
                    source_frame_seq=frame.source_frame_seq,
                    wrist_recv_ts_ns=frame.wrist_recv_ts_ns,
                    landmarks_recv_ts_ns=frame.landmarks_recv_ts_ns,
                    wrist_x=wrist.x,
                    wrist_y=wrist.y,
                    wrist_z=wrist.z,
                    wrist_qx=wrist.qx,
                    wrist_qy=wrist.qy,
                    wrist_qz=wrist.qz,
                    wrist_qw=wrist.qw,
                    landmark_values=landmark_values,
                )
            )
        return record_index

    def add_converted_target(
        self,
        *,
        record_index: int | None,
        side: str,
        sequence_id: int,
        received_at: float,
        vr_pos: np.ndarray,
        pos_delta: np.ndarray,
        rpy_delta: np.ndarray,
        raw_pose_6d: np.ndarray,
        raw_gripper_pos: float,
    ) -> None:
        """Cache the coordinate-converted target for later send-side recording."""

        if not self.enabled or record_index is None:
            return

        # Copy arrays because the send loop may mutate/reuse target buffers later.
        target = {
            "record_index": record_index,
            "side": side,
            "sequence_id": sequence_id,
            "vr_pos": vr_pos.copy(),
            "pos_delta": pos_delta.copy(),
            "rpy_delta": rpy_delta.copy(),
            "raw_pose_6d": raw_pose_6d.copy(),
            "raw_gripper_pos": float(raw_gripper_pos),
        }
        with self._lock:
            self._converted_targets_by_index[record_index] = target

    def mark_sent(
        self,
        *,
        record_index: int | None,
        sent_at: float,
        sent_pose_6d: np.ndarray,
        sent_gripper_pos: float,
    ) -> None:
        """Append one row for the command actually sent to the controller."""

        if not self.enabled or record_index is None:
            return

        with self._lock:
            target = self._converted_targets_by_index.get(record_index)
            if target is None:
                return
            self._converted_records.append(
                ConvertedRecord(
                    record_index=int(target["record_index"]),
                    side=str(target["side"]),
                    sequence_id=int(target["sequence_id"]),
                    sent_at_monotonic_s=sent_at,
                    vr_pos=np.asarray(target["vr_pos"], dtype=float).copy(),
                    pos_delta=np.asarray(target["pos_delta"], dtype=float).copy(),
                    rpy_delta=np.asarray(target["rpy_delta"], dtype=float).copy(),
                    raw_pose_6d=np.asarray(target["raw_pose_6d"], dtype=float).copy(),
                    raw_gripper_pos=float(target["raw_gripper_pos"]),
                    sent_pose_6d=sent_pose_6d.copy(),
                    sent_gripper_pos=float(sent_gripper_pos),
                )
            )

    def add_robot_state(
        self,
        *,
        sent_at: float,
        state_pose_6d: np.ndarray,
        state_gripper_pos: float,
        cmd_pose_6d: np.ndarray,
        cmd_gripper_pos: float,
    ) -> None:
        """Store real robot/controller feedback; disabled for mock controllers."""

        if not self.enabled or not self._config.record_robot_state:
            return

        state_pose = np.asarray(state_pose_6d, dtype=float).copy()
        cmd_pose = np.asarray(cmd_pose_6d, dtype=float).copy()
        state_gripper = float(state_gripper_pos)
        cmd_gripper = float(cmd_gripper_pos)
        with self._lock:
            self._robot_state_records.append(
                RobotStateRecord(
                    sent_at_monotonic_s=sent_at,
                    state_pose_6d=state_pose,
                    state_gripper_pos=state_gripper,
                    cmd_pose_6d=cmd_pose,
                    cmd_gripper_pos=cmd_gripper,
                    error_pose_6d=cmd_pose - state_pose,
                    error_gripper_pos=cmd_gripper - state_gripper,
                )
            )

    def save(self) -> tuple[str, str, str | None] | None:
        """Write the raw and converted CSV files after the control loop stops."""

        if not self.enabled:
            return None

        with self._lock:
            # Snapshot under lock, then do disk IO outside the receive/send threads.
            raw_records = list(self._raw_records)
            converted_records = list(self._converted_records)
            robot_state_records = list(self._robot_state_records)

        os.makedirs(self._config.output_dir, exist_ok=True)
        prefix = self._config.prefix
        if prefix is None:
            prefix = datetime.now().strftime("teleop_vr_%Y%m%d_%H%M%S")

        raw_path = os.path.join(self._config.output_dir, f"{prefix}_vr_raw.csv")
        converted_path = os.path.join(self._config.output_dir, f"{prefix}_converted.csv")
        robot_state_path = (
            os.path.join(self._config.output_dir, f"{prefix}_robot_state.csv")
            if self._config.record_robot_state
            else None
        )
        self._write_raw_csv(raw_path, raw_records)
        self._write_converted_csv(converted_path, converted_records)
        if robot_state_path is not None:
            self._write_robot_state_csv(robot_state_path, robot_state_records)
        return raw_path, converted_path, robot_state_path

    def _write_raw_csv(self, path: str, records: list[RawVrRecord]) -> None:
        """Write original VR wrist, landmark, and source timestamp data."""

        landmark_columns = [
            f"landmark_{index}_{axis}"
            for index in range(21)
            for axis in ("x", "y", "z")
        ]
        fieldnames = [
            "record_index",
            "side",
            "sequence_id",
            "frame_id",
            "received_at_monotonic_s",
            "recv_ts_ns",
            "recv_time_unix_ns",
            "source_ts_ns",
            "source_frame_seq",
            "wrist_recv_ts_ns",
            "landmarks_recv_ts_ns",
            "wrist_x",
            "wrist_y",
            "wrist_z",
            "wrist_qx",
            "wrist_qy",
            "wrist_qz",
            "wrist_qw",
            *landmark_columns,
        ]
        with open(path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = {
                    "record_index": record.record_index,
                    "side": record.side,
                    "sequence_id": record.sequence_id,
                    "frame_id": record.frame_id,
                    "received_at_monotonic_s": record.received_at_monotonic_s,
                    "recv_ts_ns": record.recv_ts_ns,
                    "recv_time_unix_ns": record.recv_time_unix_ns,
                    "source_ts_ns": record.source_ts_ns,
                    "source_frame_seq": record.source_frame_seq,
                    "wrist_recv_ts_ns": record.wrist_recv_ts_ns,
                    "landmarks_recv_ts_ns": record.landmarks_recv_ts_ns,
                    "wrist_x": record.wrist_x,
                    "wrist_y": record.wrist_y,
                    "wrist_z": record.wrist_z,
                    "wrist_qx": record.wrist_qx,
                    "wrist_qy": record.wrist_qy,
                    "wrist_qz": record.wrist_qz,
                    "wrist_qw": record.wrist_qw,
                }
                row.update(zip(landmark_columns, record.landmark_values, strict=True))
                writer.writerow(row)

    def _write_converted_csv(self, path: str, records: list[ConvertedRecord]) -> None:
        """Write converted targets and any send-side command/timing data."""

        fieldnames = [
            "record_index",
            "side",
            "sequence_id",
            "sent_at_monotonic_s",
            "vr_x",
            "vr_y",
            "vr_z",
            "pos_delta_x",
            "pos_delta_y",
            "pos_delta_z",
            "rpy_delta_roll",
            "rpy_delta_pitch",
            "rpy_delta_yaw",
            "raw_target_x",
            "raw_target_y",
            "raw_target_z",
            "raw_target_roll",
            "raw_target_pitch",
            "raw_target_yaw",
            "raw_gripper_pos",
            "sent_target_x",
            "sent_target_y",
            "sent_target_z",
            "sent_target_roll",
            "sent_target_pitch",
            "sent_target_yaw",
            "sent_gripper_pos",
        ]
        with open(path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "record_index": record.record_index,
                        "side": record.side,
                        "sequence_id": record.sequence_id,
                        "sent_at_monotonic_s": record.sent_at_monotonic_s,
                        "vr_x": record.vr_pos[0],
                        "vr_y": record.vr_pos[1],
                        "vr_z": record.vr_pos[2],
                        "pos_delta_x": record.pos_delta[0],
                        "pos_delta_y": record.pos_delta[1],
                        "pos_delta_z": record.pos_delta[2],
                        "rpy_delta_roll": record.rpy_delta[0],
                        "rpy_delta_pitch": record.rpy_delta[1],
                        "rpy_delta_yaw": record.rpy_delta[2],
                        "raw_target_x": record.raw_pose_6d[0],
                        "raw_target_y": record.raw_pose_6d[1],
                        "raw_target_z": record.raw_pose_6d[2],
                        "raw_target_roll": record.raw_pose_6d[3],
                        "raw_target_pitch": record.raw_pose_6d[4],
                        "raw_target_yaw": record.raw_pose_6d[5],
                        "raw_gripper_pos": record.raw_gripper_pos,
                        "sent_target_x": record.sent_pose_6d[0],
                        "sent_target_y": record.sent_pose_6d[1],
                        "sent_target_z": record.sent_pose_6d[2],
                        "sent_target_roll": record.sent_pose_6d[3],
                        "sent_target_pitch": record.sent_pose_6d[4],
                        "sent_target_yaw": record.sent_pose_6d[5],
                        "sent_gripper_pos": record.sent_gripper_pos,
                    }
                )

    def _write_robot_state_csv(self, path: str, records: list[RobotStateRecord]) -> None:
        """Write real robot/controller state feedback samples."""

        fieldnames = [
            "sent_at_monotonic_s",
            "state_x",
            "state_y",
            "state_z",
            "state_roll",
            "state_pitch",
            "state_yaw",
            "state_gripper_pos",
            "cmd_x",
            "cmd_y",
            "cmd_z",
            "cmd_roll",
            "cmd_pitch",
            "cmd_yaw",
            "cmd_gripper_pos",
            "error_x",
            "error_y",
            "error_z",
            "error_roll",
            "error_pitch",
            "error_yaw",
            "error_gripper_pos",
        ]
        with open(path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "sent_at_monotonic_s": record.sent_at_monotonic_s,
                        "state_x": record.state_pose_6d[0],
                        "state_y": record.state_pose_6d[1],
                        "state_z": record.state_pose_6d[2],
                        "state_roll": record.state_pose_6d[3],
                        "state_pitch": record.state_pose_6d[4],
                        "state_yaw": record.state_pose_6d[5],
                        "state_gripper_pos": record.state_gripper_pos,
                        "cmd_x": record.cmd_pose_6d[0],
                        "cmd_y": record.cmd_pose_6d[1],
                        "cmd_z": record.cmd_pose_6d[2],
                        "cmd_roll": record.cmd_pose_6d[3],
                        "cmd_pitch": record.cmd_pose_6d[4],
                        "cmd_yaw": record.cmd_pose_6d[5],
                        "cmd_gripper_pos": record.cmd_gripper_pos,
                        "error_x": record.error_pose_6d[0],
                        "error_y": record.error_pose_6d[1],
                        "error_z": record.error_pose_6d[2],
                        "error_roll": record.error_pose_6d[3],
                        "error_pitch": record.error_pose_6d[4],
                        "error_yaw": record.error_pose_6d[5],
                        "error_gripper_pos": record.error_gripper_pos,
                    }
                )
