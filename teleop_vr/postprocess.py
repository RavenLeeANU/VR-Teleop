from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class DampingConfig:
    enabled: bool
    alpha: float
    max_pos_step: float
    max_ori_step: float
    max_gripper_step: float
    max_pos_velocity: float = 0.50
    max_ori_velocity: float = 1.50
    max_gripper_velocity: float = 0.08
    max_pos_acceleration: float = 3.0
    max_ori_acceleration: float = 8.0
    max_gripper_acceleration: float = 0.40
    max_pos_jerk: float = 300.0
    max_ori_jerk: float = 800.0
    max_gripper_jerk: float = 40.0
    pose_min: np.ndarray | None = None
    pose_max: np.ndarray | None = None
    gripper_min: float = 0.0
    gripper_max: float | None = None
    max_missing_frames: int = 10
    sg_position_enabled: bool = False
    sg_window_size: int = 21
    sg_poly_order: int = 2
    orientation_ema_enabled: bool = False
    orientation_ema_alpha_x: float = 0.15
    orientation_ema_alpha_y: float = 0.15
    position_deadband: float = 0.0
    orientation_deadband: float = 0.0
    gripper_deadband: float = 0.0


def _to_optional_pose_array(value: object, name: str) -> np.ndarray | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple) or len(value) != 6:
        raise ValueError(f"{name} must be a list of 6 values")
    return np.asarray([float(item) for item in value], dtype=float)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_damping_config(path: str | Path | None = None) -> DampingConfig:
    """Load postprocess/damping configuration from YAML.

    The YAML may either contain the fields directly or nest them under a
    top-level ``postprocess`` key.
    """

    values: dict[str, Any] = {
        "enabled": False,
        "alpha": 0.6,
        "max_pos_step": 0.01,
        "max_ori_step": 0.10,
        "max_gripper_step": 0.005,
        "max_pos_velocity": 0.50,
        "max_ori_velocity": 1.50,
        "max_gripper_velocity": 0.08,
        "max_pos_acceleration": 3.0,
        "max_ori_acceleration": 8.0,
        "max_gripper_acceleration": 0.40,
        "max_pos_jerk": 300.0,
        "max_ori_jerk": 800.0,
        "max_gripper_jerk": 40.0,
        "pose_min": None,
        "pose_max": None,
        "gripper_min": 0.0,
        "gripper_max": None,
        "max_missing_frames": 10,
        "sg_position_enabled": False,
        "sg_window_size": 21,
        "sg_poly_order": 2,
        "orientation_ema_enabled": False,
        "orientation_ema_alpha_x": 0.15,
        "orientation_ema_alpha_y": 0.15,
        "position_deadband": 0.0,
        "orientation_deadband": 0.0,
        "gripper_deadband": 0.0,
    }
    if path is not None:
        data = _read_yaml(Path(path))
        data = data.get("postprocess", data)
        if not isinstance(data, dict):
            raise ValueError("postprocess must be a YAML mapping")
        unknown = set(data) - set(values)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown postprocess config keys: {names}")
        values.update(data)

    config = DampingConfig(
        enabled=bool(values["enabled"]),
        alpha=float(values["alpha"]),
        max_pos_step=float(values["max_pos_step"]),
        max_ori_step=float(values["max_ori_step"]),
        max_gripper_step=float(values["max_gripper_step"]),
        max_pos_velocity=float(values["max_pos_velocity"]),
        max_ori_velocity=float(values["max_ori_velocity"]),
        max_gripper_velocity=float(values["max_gripper_velocity"]),
        max_pos_acceleration=float(values["max_pos_acceleration"]),
        max_ori_acceleration=float(values["max_ori_acceleration"]),
        max_gripper_acceleration=float(values["max_gripper_acceleration"]),
        max_pos_jerk=float(values["max_pos_jerk"]),
        max_ori_jerk=float(values["max_ori_jerk"]),
        max_gripper_jerk=float(values["max_gripper_jerk"]),
        pose_min=_to_optional_pose_array(values["pose_min"], "pose_min"),
        pose_max=_to_optional_pose_array(values["pose_max"], "pose_max"),
        gripper_min=float(values["gripper_min"]),
        gripper_max=(
            None if values["gripper_max"] is None else float(values["gripper_max"])
        ),
        max_missing_frames=int(values["max_missing_frames"]),
        sg_position_enabled=bool(values["sg_position_enabled"]),
        sg_window_size=int(values["sg_window_size"]),
        sg_poly_order=int(values["sg_poly_order"]),
        orientation_ema_enabled=bool(values["orientation_ema_enabled"]),
        orientation_ema_alpha_x=float(values["orientation_ema_alpha_x"]),
        orientation_ema_alpha_y=float(values["orientation_ema_alpha_y"]),
        position_deadband=float(values["position_deadband"]),
        orientation_deadband=float(values["orientation_deadband"]),
        gripper_deadband=float(values["gripper_deadband"]),
    )
    validate_damping_config(config)
    return config


def validate_damping_config(config: DampingConfig) -> None:
    if not 0.0 < config.alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    positive_values = {
        "max_pos_step": config.max_pos_step,
        "max_ori_step": config.max_ori_step,
        "max_gripper_step": config.max_gripper_step,
        "max_pos_velocity": config.max_pos_velocity,
        "max_ori_velocity": config.max_ori_velocity,
        "max_gripper_velocity": config.max_gripper_velocity,
        "max_pos_acceleration": config.max_pos_acceleration,
        "max_ori_acceleration": config.max_ori_acceleration,
        "max_gripper_acceleration": config.max_gripper_acceleration,
        "max_pos_jerk": config.max_pos_jerk,
        "max_ori_jerk": config.max_ori_jerk,
        "max_gripper_jerk": config.max_gripper_jerk,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0.0]
    if invalid:
        raise ValueError(f"Postprocess values must be positive: {', '.join(invalid)}")
    if (
        config.pose_min is not None
        and config.pose_max is not None
        and np.any(config.pose_min > config.pose_max)
    ):
        raise ValueError("pose_min values must be <= pose_max values")
    if config.gripper_max is not None and config.gripper_min > config.gripper_max:
        raise ValueError("gripper_min must be <= gripper_max")
    if config.max_missing_frames < 0:
        raise ValueError("max_missing_frames must be >= 0")
    if config.sg_window_size < 1:
        raise ValueError("sg_window_size must be >= 1")
    if config.sg_poly_order < 0:
        raise ValueError("sg_poly_order must be >= 0")
    if config.sg_window_size <= config.sg_poly_order:
        raise ValueError("sg_window_size must be greater than sg_poly_order")
    if not 0.0 < config.orientation_ema_alpha_x <= 1.0:
        raise ValueError("orientation_ema_alpha_x must be in (0, 1]")
    if not 0.0 < config.orientation_ema_alpha_y <= 1.0:
        raise ValueError("orientation_ema_alpha_y must be in (0, 1]")
    deadband_values = {
        "position_deadband": config.position_deadband,
        "orientation_deadband": config.orientation_deadband,
        "gripper_deadband": config.gripper_deadband,
    }
    invalid_deadband = [name for name, value in deadband_values.items() if value < 0.0]
    if invalid_deadband:
        raise ValueError(f"Deadband values must be >= 0: {', '.join(invalid_deadband)}")


@dataclass
class SmoothedTarget:
    pose_6d: np.ndarray
    gripper_pos: float
    limited: bool
    step_limited: bool = False
    velocity_limited: bool = False
    acceleration_limited: bool = False
    jerk_limited: bool = False
    command_limited: bool = False
    gap_filled: bool = False
    deadband_applied: bool = False
    position_smoothed: bool = False
    orientation_smoothed: bool = False


def wrap_angle_delta(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def make_rpy_continuous(rpy: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    if previous is None:
        return wrap_angle_delta(rpy)
    return previous + wrap_angle_delta(rpy - previous)


def rpy_to_rotmat(rpy: np.ndarray) -> np.ndarray:
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


def rotmat_to_rpy(rot: np.ndarray) -> np.ndarray:
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


def _normalize_axis(axis: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return fallback.copy()
    return axis / norm


def _orthonormalize_xy(x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    x = _normalize_axis(x_axis, np.array([1.0, 0.0, 0.0], dtype=float))
    y_raw = y_axis - x * float(np.dot(x, y_axis))
    y = _normalize_axis(y_raw, np.array([0.0, 1.0, 0.0], dtype=float))
    z = _normalize_axis(np.cross(x, y), np.array([0.0, 0.0, 1.0], dtype=float))
    y = _normalize_axis(np.cross(z, x), y)
    return np.column_stack((x, y, z))


def replace_nonfinite_command_values(
    pose_6d: np.ndarray,
    gripper_pos: float,
    previous_pose_6d: np.ndarray | None,
    previous_gripper_pos: float | None,
) -> tuple[np.ndarray, float, bool]:
    """Replace NaN/Inf command values with the last valid sent command."""

    pose = np.asarray(pose_6d, dtype=float).copy()
    gripper = float(gripper_pos)
    replaced = False

    pose_finite = np.isfinite(pose)
    if not bool(np.all(pose_finite)):
        if previous_pose_6d is None:
            raise ValueError("non-finite pose command has no previous command fallback")
        pose[~pose_finite] = previous_pose_6d[~pose_finite]
        replaced = True

    if not math.isfinite(gripper):
        if previous_gripper_pos is None:
            raise ValueError("non-finite gripper command has no previous command fallback")
        gripper = previous_gripper_pos
        replaced = True

    return pose, gripper, replaced


def _limit_vector_norm(delta: np.ndarray, max_norm: float) -> tuple[np.ndarray, bool]:
    norm = float(np.linalg.norm(delta))
    if norm <= max_norm or norm <= 1e-12:
        return delta, False
    return delta * (max_norm / norm), True


def _limit_command_groups(
    values: np.ndarray,
    *,
    pos_limit: float,
    ori_limit: float,
    gripper_limit: float,
) -> tuple[np.ndarray, bool]:
    limited_values = values.copy()
    limited_pos, pos_limited = _limit_vector_norm(limited_values[:3], pos_limit)
    limited_ori, ori_limited = _limit_vector_norm(limited_values[3:6], ori_limit)

    gripper_raw = float(limited_values[6])
    limited_gripper = float(np.clip(gripper_raw, -gripper_limit, gripper_limit))
    gripper_limited = abs(limited_gripper - gripper_raw) > 1e-12

    limited_values[:3] = limited_pos
    limited_values[3:6] = limited_ori
    limited_values[6] = limited_gripper
    return limited_values, pos_limited or ori_limited or gripper_limited


def _compose_command(pose_6d: np.ndarray, gripper_pos: float) -> np.ndarray:
    command = np.empty(7, dtype=float)
    command[:6] = pose_6d
    command[6] = float(gripper_pos)
    return command


def _split_command(command: np.ndarray) -> tuple[np.ndarray, float]:
    return command[:6].copy(), float(command[6])


def _savgol_coefficients(window_size: int, poly_order: int) -> np.ndarray:
    offsets = np.arange(-(window_size - 1), 1, dtype=float)
    vandermonde = np.vander(offsets, poly_order + 1, increasing=True)
    return np.linalg.pinv(vandermonde)[0]


class TrajectorySmoother:
    """Smooth VR teleop targets before they are sent to the robot controller."""

    def __init__(self, config: DampingConfig, *, cmd_dt: float = 0.01) -> None:
        if cmd_dt <= 0.0:
            raise ValueError("cmd_dt must be positive")
        self._config = config
        self._cmd_dt = float(cmd_dt)
        self._command: np.ndarray | None = None
        self._velocity = np.zeros(7, dtype=float)
        self._acceleration = np.zeros(7, dtype=float)
        self._position_window: deque[np.ndarray] = deque(
            maxlen=max(1, config.sg_window_size)
        )
        self._sg_coefficients = _savgol_coefficients(
            config.sg_window_size,
            config.sg_poly_order,
        )
        self._orientation_matrix: np.ndarray | None = None
        self._orientation_rpy: np.ndarray | None = None
        self._last_input_pose: np.ndarray | None = None
        self._last_input_gripper: float | None = None
        self._deadband_pose: np.ndarray | None = None
        self._deadband_gripper: float | None = None
        self._missing_count = 0

    def reset(self) -> None:
        self._command = None
        self._velocity[:] = 0.0
        self._acceleration[:] = 0.0
        self._position_window.clear()
        self._orientation_matrix = None
        self._orientation_rpy = None
        self._last_input_pose = None
        self._last_input_gripper = None
        self._deadband_pose = None
        self._deadband_gripper = None
        self._missing_count = 0

    def process(self, raw_pose_6d: np.ndarray, raw_gripper_pos: float) -> SmoothedTarget:
        raw_pose = np.asarray(raw_pose_6d, dtype=float).copy()
        if raw_pose.shape != (6,):
            raise ValueError(f"raw_pose_6d must have shape (6,), got {raw_pose.shape}")

        raw_gripper = float(raw_gripper_pos)
        raw_pose, raw_gripper, gap_filled = self._fill_short_gap(raw_pose, raw_gripper)
        raw_pose, deadband_applied = self._apply_deadband(raw_pose)
        raw_pose, position_smoothed = self._smooth_position(raw_pose)
        raw_pose, orientation_smoothed = self._smooth_orientation(raw_pose)
        raw_command = _compose_command(raw_pose, raw_gripper)
        if not self._config.enabled:
            raw_gripper, gripper_limited = self._clip_gripper(raw_gripper)
            limited = (
                gap_filled
                or deadband_applied
                or position_smoothed
                or orientation_smoothed
                or gripper_limited
            )
            return SmoothedTarget(
                raw_pose.copy(),
                raw_gripper,
                limited,
                command_limited=gripper_limited,
                gap_filled=gap_filled,
                deadband_applied=deadband_applied,
                position_smoothed=position_smoothed,
                orientation_smoothed=orientation_smoothed,
            )

        if self._command is not None:
            raw_command[3:6] = self._command[3:6] + wrap_angle_delta(
                raw_command[3:6] - self._command[3:6]
            )
            raw_command[6] = raw_gripper

        raw_command, command_limited = self._apply_command_limits(raw_command)

        if self._command is None:
            self._command = raw_command.copy()
            pose_6d, gripper_pos = _split_command(self._command)
            return SmoothedTarget(
                pose_6d,
                gripper_pos,
                command_limited
                or gap_filled
                or deadband_applied
                or position_smoothed
                or orientation_smoothed,
                command_limited=command_limited,
                gap_filled=gap_filled,
                deadband_applied=deadband_applied,
                position_smoothed=position_smoothed,
                orientation_smoothed=orientation_smoothed,
            )

        current = self._command
        delta = raw_command - current
        delta[3:6] = wrap_angle_delta(delta[3:6])
        delta[6] = 0.0

        step_delta, step_limited = _limit_command_groups(
            delta,
            pos_limit=self._config.max_pos_step,
            ori_limit=self._config.max_ori_step,
            gripper_limit=float("inf"),
        )
        desired_delta = self._config.alpha * step_delta

        desired_delta, velocity_limited = _limit_command_groups(
            desired_delta,
            pos_limit=self._config.max_pos_velocity * self._cmd_dt,
            ori_limit=self._config.max_ori_velocity * self._cmd_dt,
            gripper_limit=float("inf"),
        )

        desired_velocity = desired_delta / self._cmd_dt
        velocity_delta, acceleration_limited = _limit_command_groups(
            desired_velocity - self._velocity,
            pos_limit=self._config.max_pos_acceleration * self._cmd_dt,
            ori_limit=self._config.max_ori_acceleration * self._cmd_dt,
            gripper_limit=float("inf"),
        )
        desired_velocity = self._velocity + velocity_delta

        desired_acceleration = (desired_velocity - self._velocity) / self._cmd_dt
        acceleration_delta, jerk_limited = _limit_command_groups(
            desired_acceleration - self._acceleration,
            pos_limit=self._config.max_pos_jerk * self._cmd_dt,
            ori_limit=self._config.max_ori_jerk * self._cmd_dt,
            gripper_limit=float("inf"),
        )
        desired_acceleration = self._acceleration + acceleration_delta
        desired_velocity = self._velocity + desired_acceleration * self._cmd_dt
        desired_delta = desired_velocity * self._cmd_dt

        next_command = current + desired_delta
        next_command[6] = raw_gripper
        next_command, final_command_limited = self._apply_command_limits(next_command)
        command_limited = command_limited or final_command_limited

        actual_delta = next_command - current
        actual_delta[3:6] = wrap_angle_delta(actual_delta[3:6])
        actual_velocity = actual_delta / self._cmd_dt
        actual_acceleration = (actual_velocity - self._velocity) / self._cmd_dt

        self._command = next_command
        self._velocity = actual_velocity
        self._acceleration = actual_acceleration

        pose_6d, gripper_pos = _split_command(next_command)
        limited = (
            step_limited
            or velocity_limited
            or acceleration_limited
            or jerk_limited
            or command_limited
            or gap_filled
            or deadband_applied
            or position_smoothed
            or orientation_smoothed
        )
        return SmoothedTarget(
            pose_6d,
            gripper_pos,
            limited,
            step_limited=step_limited,
            velocity_limited=velocity_limited,
            acceleration_limited=acceleration_limited,
            jerk_limited=jerk_limited,
            command_limited=command_limited,
            gap_filled=gap_filled,
            deadband_applied=deadband_applied,
            position_smoothed=position_smoothed,
            orientation_smoothed=orientation_smoothed,
        )

    def _apply_command_limits(self, command: np.ndarray) -> tuple[np.ndarray, bool]:
        limited = command.copy()
        if self._config.pose_min is not None:
            limited[:6] = np.maximum(limited[:6], self._config.pose_min)
        if self._config.pose_max is not None:
            limited[:6] = np.minimum(limited[:6], self._config.pose_max)
        limited[6], _ = self._clip_gripper(float(limited[6]))
        return limited, bool(np.any(np.abs(limited - command) > 1e-12))

    def _clip_gripper(self, gripper_pos: float) -> tuple[float, bool]:
        if not math.isfinite(gripper_pos):
            return gripper_pos, False
        clipped = max(float(self._config.gripper_min), float(gripper_pos))
        if self._config.gripper_max is not None:
            clipped = min(float(self._config.gripper_max), clipped)
        return clipped, not math.isclose(clipped, gripper_pos)

    def _fill_short_gap(
        self,
        raw_pose: np.ndarray,
        raw_gripper: float,
    ) -> tuple[np.ndarray, float, bool]:
        pose_finite = np.isfinite(raw_pose)
        gripper_finite = math.isfinite(raw_gripper)
        if bool(np.all(pose_finite)) and gripper_finite:
            self._last_input_pose = raw_pose.copy()
            self._last_input_gripper = raw_gripper
            self._missing_count = 0
            return raw_pose, raw_gripper, False

        self._missing_count += 1
        if (
            self._last_input_pose is None
            or self._last_input_gripper is None
            or self._missing_count > self._config.max_missing_frames
        ):
            return raw_pose, raw_gripper, False

        filled_pose = raw_pose.copy()
        filled_pose[~pose_finite] = self._last_input_pose[~pose_finite]
        filled_gripper = raw_gripper if gripper_finite else self._last_input_gripper
        return filled_pose, filled_gripper, True

    def _apply_deadband(self, raw_pose: np.ndarray) -> tuple[np.ndarray, bool]:
        if (
            self._config.position_deadband <= 0.0
            and self._config.orientation_deadband <= 0.0
        ):
            return raw_pose, False
        if not bool(np.all(np.isfinite(raw_pose))):
            return raw_pose, False

        if self._deadband_pose is None:
            self._deadband_pose = raw_pose.copy()
            return raw_pose, False

        filtered_pose = raw_pose.copy()
        applied = False

        pos_delta = raw_pose[:3] - self._deadband_pose[:3]
        if (
            self._config.position_deadband > 0.0
            and float(np.linalg.norm(pos_delta)) < self._config.position_deadband
        ):
            filtered_pose[:3] = self._deadband_pose[:3]
            applied = True
        else:
            self._deadband_pose[:3] = raw_pose[:3]

        ori_delta = wrap_angle_delta(raw_pose[3:6] - self._deadband_pose[3:6])
        if (
            self._config.orientation_deadband > 0.0
            and float(np.linalg.norm(ori_delta)) < self._config.orientation_deadband
        ):
            filtered_pose[3:6] = self._deadband_pose[3:6]
            applied = True
        else:
            self._deadband_pose[3:6] = raw_pose[3:6]

        return filtered_pose, applied

    def _smooth_position(self, raw_pose: np.ndarray) -> tuple[np.ndarray, bool]:
        if not self._config.sg_position_enabled:
            return raw_pose, False
        if not bool(np.all(np.isfinite(raw_pose[:3]))):
            return raw_pose, False

        self._position_window.append(raw_pose[:3].copy())
        if len(self._position_window) < self._config.sg_window_size:
            return raw_pose, False

        window = np.asarray(self._position_window, dtype=float)
        smoothed_pose = raw_pose.copy()
        smoothed_pose[:3] = self._sg_coefficients @ window
        return smoothed_pose, True

    def _smooth_orientation(self, raw_pose: np.ndarray) -> tuple[np.ndarray, bool]:
        if not self._config.orientation_ema_enabled:
            return raw_pose, False
        if not bool(np.all(np.isfinite(raw_pose[3:6]))):
            return raw_pose, False

        current_rot = rpy_to_rotmat(raw_pose[3:6])
        if self._orientation_matrix is None:
            self._orientation_matrix = _orthonormalize_xy(
                current_rot[:, 0],
                current_rot[:, 1],
            )
            self._orientation_rpy = raw_pose[3:6].copy()
            return raw_pose, False

        x_axis = current_rot[:, 0]
        y_axis = current_rot[:, 1]
        previous_x = self._orientation_matrix[:, 0]
        previous_y = self._orientation_matrix[:, 1]
        if float(np.dot(x_axis, previous_x)) < 0.0:
            x_axis = -x_axis
        if float(np.dot(y_axis, previous_y)) < 0.0:
            y_axis = -y_axis

        x_smoothed = (
            (1.0 - self._config.orientation_ema_alpha_x) * previous_x
            + self._config.orientation_ema_alpha_x * x_axis
        )
        y_smoothed = (
            (1.0 - self._config.orientation_ema_alpha_y) * previous_y
            + self._config.orientation_ema_alpha_y * y_axis
        )
        self._orientation_matrix = _orthonormalize_xy(x_smoothed, y_smoothed)

        smoothed_pose = raw_pose.copy()
        smoothed_pose[3:6] = make_rpy_continuous(
            rotmat_to_rpy(self._orientation_matrix),
            self._orientation_rpy,
        )
        self._orientation_rpy = smoothed_pose[3:6].copy()
        return smoothed_pose, True
