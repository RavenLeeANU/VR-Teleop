from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class RuntimePostprocessConfig:
    enabled: bool = False
    alpha: float = 0.6
    max_pos_step: float = 0.01
    max_ori_step: float = 0.10
    max_gripper_step: float = 0.005
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
    gripper_closed_threshold: float | None = None
    gripper_open_threshold: float | None = None
    max_missing_frames: int = 10
    position_deadband: float = 0.0
    orientation_deadband: float = 0.0
    gripper_deadband: float = 0.0
    mpc_tracking_enabled: bool = False
    mpc_delay_frames: int = 5
    mpc_tracking_frequency: float = 12.0
    mpc_damping_ratio: float = 1.0
    mpc_reference_velocity_gain: float = 1.0
    mpc_orientation_tracking_frequency: float | None = None
    mpc_orientation_damping_ratio: float | None = None
    mpc_orientation_reference_velocity_gain: float | None = None


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
    stationary_held: bool = False
    input_spike_rejected: bool = False
    transition_active: bool = False
    mpc_tracking_active: bool = False
    manifold_spline_active: bool = False


def wrap_angle_delta(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def make_rpy_continuous(rpy: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    if previous is None:
        return wrap_angle_delta(rpy)
    return previous + wrap_angle_delta(rpy - previous)


def replace_nonfinite_command_values(
    pose_6d: np.ndarray,
    gripper_pos: float,
    previous_pose_6d: np.ndarray | None,
    previous_gripper_pos: float | None,
) -> tuple[np.ndarray, float, bool]:
    pose = np.asarray(pose_6d, dtype=float).copy()
    if pose.shape != (6,):
        raise ValueError(f"pose_6d must have shape (6,), got {pose.shape}")

    gripper = float(gripper_pos)
    pose_bad = ~np.isfinite(pose)
    gripper_bad = not math.isfinite(gripper)
    if not bool(np.any(pose_bad)) and not gripper_bad:
        return pose, gripper, False
    if previous_pose_6d is None or previous_gripper_pos is None:
        raise ValueError("non-finite command received before a valid command exists")

    previous_pose = np.asarray(previous_pose_6d, dtype=float)
    if previous_pose.shape != (6,) or not bool(np.all(np.isfinite(previous_pose))):
        raise ValueError("previous pose is invalid")
    if not math.isfinite(float(previous_gripper_pos)):
        raise ValueError("previous gripper is invalid")
    pose[pose_bad] = previous_pose[pose_bad]
    if gripper_bad:
        gripper = float(previous_gripper_pos)
    return pose, gripper, True


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _optional_pose(value: object, name: str) -> np.ndarray | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple) or len(value) != 6:
        raise ValueError(f"{name} must be a list of 6 values")
    return np.asarray([float(item) for item in value], dtype=float)


def load_runtime_postprocess_config(
    path: str | Path | None = None,
) -> RuntimePostprocessConfig:
    values: dict[str, Any] = {
        field: getattr(RuntimePostprocessConfig(), field)
        for field in RuntimePostprocessConfig.__dataclass_fields__
    }
    # Accept legacy/experiment keys from postprocess_config.yaml, but ignore
    # branches that the runtime sender no longer uses.
    ignored_keys = {
        "deadband_velocity_threshold",
        "stationary_hold_enabled",
        "stationary_window_size",
        "stationary_pos_range",
        "stationary_ori_range",
        "stationary_command_pos_threshold",
        "stationary_command_ori_threshold",
        "stationary_frames",
        "input_jump_protection_enabled",
        "max_input_pos_jump",
        "max_input_ori_jump",
        "transition_confirm_frames",
        "stationary_hold_cooldown_frames",
        "manifold_spline_enabled",
        "manifold_spline_position_tension",
        "manifold_spline_orientation_tension",
        "sg_position_enabled",
        "sg_window_size",
        "sg_poly_order",
        "orientation_ema_enabled",
        "orientation_ema_alpha_x",
        "orientation_ema_alpha_y",
    }
    if path is not None:
        data = _read_yaml(Path(path))
        data = data.get("postprocess", data)
        if not isinstance(data, dict):
            raise ValueError("postprocess must be a YAML mapping")
        unknown = set(data) - set(values) - ignored_keys
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown runtime postprocess config keys: {names}")
        values.update({key: value for key, value in data.items() if key in values})

    config = RuntimePostprocessConfig(
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
        pose_min=_optional_pose(values["pose_min"], "pose_min"),
        pose_max=_optional_pose(values["pose_max"], "pose_max"),
        gripper_min=float(values["gripper_min"]),
        gripper_max=None if values["gripper_max"] is None else float(values["gripper_max"]),
        gripper_closed_threshold=(
            None
            if values["gripper_closed_threshold"] is None
            else float(values["gripper_closed_threshold"])
        ),
        gripper_open_threshold=(
            None
            if values["gripper_open_threshold"] is None
            else float(values["gripper_open_threshold"])
        ),
        max_missing_frames=int(values["max_missing_frames"]),
        position_deadband=float(values["position_deadband"]),
        orientation_deadband=float(values["orientation_deadband"]),
        gripper_deadband=float(values["gripper_deadband"]),
        mpc_tracking_enabled=bool(values["mpc_tracking_enabled"]),
        mpc_delay_frames=int(values["mpc_delay_frames"]),
        mpc_tracking_frequency=float(values["mpc_tracking_frequency"]),
        mpc_damping_ratio=float(values["mpc_damping_ratio"]),
        mpc_reference_velocity_gain=float(values["mpc_reference_velocity_gain"]),
        mpc_orientation_tracking_frequency=(
            None
            if values["mpc_orientation_tracking_frequency"] is None
            else float(values["mpc_orientation_tracking_frequency"])
        ),
        mpc_orientation_damping_ratio=(
            None
            if values["mpc_orientation_damping_ratio"] is None
            else float(values["mpc_orientation_damping_ratio"])
        ),
        mpc_orientation_reference_velocity_gain=(
            None
            if values["mpc_orientation_reference_velocity_gain"] is None
            else float(values["mpc_orientation_reference_velocity_gain"])
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: RuntimePostprocessConfig) -> None:
    if not 0.0 < config.alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    positive = {
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
    invalid = [name for name, value in positive.items() if value <= 0.0]
    if invalid:
        raise ValueError(f"Postprocess values must be positive: {', '.join(invalid)}")
    if config.pose_min is not None and config.pose_max is not None:
        if bool(np.any(config.pose_min > config.pose_max)):
            raise ValueError("pose_min values must be <= pose_max values")
    if config.gripper_max is not None and config.gripper_min > config.gripper_max:
        raise ValueError("gripper_min must be <= gripper_max")
    if config.max_missing_frames < 0:
        raise ValueError("max_missing_frames must be >= 0")
    if config.position_deadband < 0.0 or config.orientation_deadband < 0.0:
        raise ValueError("deadband values must be >= 0")
    if config.mpc_delay_frames < 0:
        raise ValueError("mpc_delay_frames must be >= 0")
    if config.mpc_tracking_frequency <= 0.0:
        raise ValueError("mpc_tracking_frequency must be positive")
    if config.mpc_damping_ratio <= 0.0:
        raise ValueError("mpc_damping_ratio must be positive")


def _compose(pose_6d: np.ndarray, gripper: float) -> np.ndarray:
    return np.concatenate([np.asarray(pose_6d, dtype=float), [float(gripper)]])


def _split(command: np.ndarray) -> tuple[np.ndarray, float]:
    return command[:6].copy(), float(command[6])


def _limit_vector(vector: np.ndarray, limit: float) -> tuple[np.ndarray, bool]:
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm <= 1e-12:
        return vector, False
    return vector * (limit / norm), True


def _limit_groups(
    command: np.ndarray,
    *,
    pos_limit: float,
    ori_limit: float,
    gripper_limit: float,
) -> tuple[np.ndarray, bool]:
    result = command.copy()
    limited = False
    result[:3], pos_limited = _limit_vector(result[:3], pos_limit)
    result[3:6], ori_limited = _limit_vector(result[3:6], ori_limit)
    if abs(result[6]) > gripper_limit:
        result[6] = math.copysign(gripper_limit, result[6])
        limited = True
    return result, limited or pos_limited or ori_limited


class TrajectorySmoother:
    def __init__(self, config: RuntimePostprocessConfig, *, cmd_dt: float = 0.01) -> None:
        if cmd_dt <= 0.0:
            raise ValueError("cmd_dt must be positive")
        self._config = config
        self._cmd_dt = float(cmd_dt)
        self._command: np.ndarray | None = None
        self._velocity = np.zeros(7, dtype=float)
        self._acceleration = np.zeros(7, dtype=float)
        self._last_valid_pose: np.ndarray | None = None
        self._last_valid_gripper: float | None = None
        self._missing_count = 0
        self._deadband_pose: np.ndarray | None = None
        self._reference_window: deque[np.ndarray] = deque(
            maxlen=max(2, config.mpc_delay_frames + 2)
        )

    def process(self, raw_pose_6d: np.ndarray, raw_gripper_pos: float) -> SmoothedTarget:
        raw_pose = np.asarray(raw_pose_6d, dtype=float).copy()
        if raw_pose.shape != (6,):
            raise ValueError(f"raw_pose_6d must have shape (6,), got {raw_pose.shape}")
        raw_gripper = float(raw_gripper_pos)
        raw_pose, raw_gripper, gap_filled = self._fill_gap(raw_pose, raw_gripper)
        raw_gripper = self._shape_gripper(raw_gripper)
        raw_pose, deadband_applied = self._apply_deadband(raw_pose)
        raw_command = _compose(raw_pose, raw_gripper)
        raw_command, command_limited = self._apply_command_limits(raw_command)

        if not self._config.enabled:
            pose, gripper = _split(raw_command)
            return SmoothedTarget(
                pose,
                gripper,
                gap_filled or deadband_applied or command_limited,
                command_limited=command_limited,
                gap_filled=gap_filled,
                deadband_applied=deadband_applied,
            )

        if self._config.mpc_tracking_enabled:
            return self._process_delayed_tracker(
                raw_command,
                raw_gripper,
                gap_filled=gap_filled,
                deadband_applied=deadband_applied,
                command_limited=command_limited,
            )
        return self._process_limited_follow(
            raw_command,
            raw_gripper,
            gap_filled=gap_filled,
            deadband_applied=deadband_applied,
            command_limited=command_limited,
        )

    def _fill_gap(
        self,
        raw_pose: np.ndarray,
        raw_gripper: float,
    ) -> tuple[np.ndarray, float, bool]:
        finite = bool(np.all(np.isfinite(raw_pose))) and math.isfinite(raw_gripper)
        if finite:
            self._last_valid_pose = raw_pose.copy()
            self._last_valid_gripper = float(raw_gripper)
            self._missing_count = 0
            return raw_pose, raw_gripper, False
        self._missing_count += 1
        if (
            self._last_valid_pose is not None
            and self._last_valid_gripper is not None
            and self._missing_count <= self._config.max_missing_frames
        ):
            pose = np.where(np.isfinite(raw_pose), raw_pose, self._last_valid_pose)
            gripper = raw_gripper if math.isfinite(raw_gripper) else self._last_valid_gripper
            return pose, float(gripper), True
        return raw_pose, raw_gripper, False

    def _shape_gripper(self, gripper: float) -> float:
        if not math.isfinite(gripper):
            return gripper
        if (
            self._config.gripper_closed_threshold is not None
            and gripper < self._config.gripper_closed_threshold
        ):
            return float(self._config.gripper_min)
        if (
            self._config.gripper_open_threshold is not None
            and gripper > self._config.gripper_open_threshold
            and self._config.gripper_max is not None
        ):
            return float(self._config.gripper_max)
        return gripper

    def _clip_gripper(self, gripper: float) -> tuple[float, bool]:
        if not math.isfinite(gripper):
            return gripper, False
        clipped = max(float(self._config.gripper_min), float(gripper))
        if self._config.gripper_max is not None:
            clipped = min(float(self._config.gripper_max), clipped)
        return clipped, not math.isclose(clipped, gripper)

    def _apply_command_limits(self, command: np.ndarray) -> tuple[np.ndarray, bool]:
        limited = command.copy()
        if self._config.pose_min is not None:
            limited[:6] = np.maximum(limited[:6], self._config.pose_min)
        if self._config.pose_max is not None:
            limited[:6] = np.minimum(limited[:6], self._config.pose_max)
        limited[6], gripper_limited = self._clip_gripper(float(limited[6]))
        pose_limited = bool(np.any(np.abs(limited[:6] - command[:6]) > 1e-12))
        return limited, pose_limited or gripper_limited

    def _apply_deadband(self, raw_pose: np.ndarray) -> tuple[np.ndarray, bool]:
        if self._config.position_deadband <= 0.0 and self._config.orientation_deadband <= 0.0:
            return raw_pose, False
        if self._deadband_pose is None:
            self._deadband_pose = raw_pose.copy()
            return raw_pose, False

        filtered = raw_pose.copy()
        applied = False
        pos_delta = raw_pose[:3] - self._deadband_pose[:3]
        if (
            self._config.position_deadband > 0.0
            and float(np.linalg.norm(pos_delta)) < self._config.position_deadband
        ):
            filtered[:3] = self._deadband_pose[:3]
            applied = True
        else:
            self._deadband_pose[:3] = raw_pose[:3]

        ori_delta = wrap_angle_delta(raw_pose[3:6] - self._deadband_pose[3:6])
        if (
            self._config.orientation_deadband > 0.0
            and float(np.linalg.norm(ori_delta)) < self._config.orientation_deadband
        ):
            filtered[3:6] = self._deadband_pose[3:6]
            applied = True
        else:
            self._deadband_pose[3:6] = raw_pose[3:6]
        return filtered, applied

    def _process_limited_follow(
        self,
        raw_command: np.ndarray,
        raw_gripper: float,
        *,
        gap_filled: bool,
        deadband_applied: bool,
        command_limited: bool,
    ) -> SmoothedTarget:
        if self._command is None:
            self._command = raw_command.copy()
            pose, gripper = _split(self._command)
            return SmoothedTarget(
                pose,
                gripper,
                gap_filled or deadband_applied or command_limited,
                command_limited=command_limited,
                gap_filled=gap_filled,
                deadband_applied=deadband_applied,
            )

        current = self._command
        delta = raw_command - current
        delta[3:6] = wrap_angle_delta(delta[3:6])
        delta[6] = 0.0
        step_delta, step_limited = _limit_groups(
            delta,
            pos_limit=self._config.max_pos_step,
            ori_limit=self._config.max_ori_step,
            gripper_limit=float("inf"),
        )
        desired_delta = self._config.alpha * step_delta
        next_command, flags = self._integrate_limited_delta(current, desired_delta, raw_gripper)
        final_command, final_limited = self._apply_command_limits(next_command)
        command_limited = command_limited or final_limited
        self._update_state(current, final_command)
        pose, gripper = _split(final_command)
        return SmoothedTarget(
            pose,
            gripper,
            step_limited or command_limited or gap_filled or deadband_applied or any(flags),
            step_limited=step_limited,
            velocity_limited=flags[0],
            acceleration_limited=flags[1],
            jerk_limited=flags[2],
            command_limited=command_limited,
            gap_filled=gap_filled,
            deadband_applied=deadband_applied,
        )

    def _process_delayed_tracker(
        self,
        raw_command: np.ndarray,
        raw_gripper: float,
        *,
        gap_filled: bool,
        deadband_applied: bool,
        command_limited: bool,
    ) -> SmoothedTarget:
        self._reference_window.append(raw_command.copy())
        reference, reference_velocity = self._delayed_reference()
        reference[6] = raw_gripper
        if self._command is None:
            self._command = reference.copy()
            pose, gripper = _split(self._command)
            return SmoothedTarget(
                pose,
                gripper,
                gap_filled or deadband_applied or command_limited,
                command_limited=command_limited,
                gap_filled=gap_filled,
                deadband_applied=deadband_applied,
                mpc_tracking_active=True,
            )

        current = self._command
        error = reference - current
        error[3:6] = wrap_angle_delta(error[3:6])
        error[6] = 0.0
        velocity_error = self._config.mpc_reference_velocity_gain * reference_velocity - self._velocity
        ori_velocity_gain = (
            self._config.mpc_reference_velocity_gain
            if self._config.mpc_orientation_reference_velocity_gain is None
            else self._config.mpc_orientation_reference_velocity_gain
        )
        velocity_error[3:6] = ori_velocity_gain * reference_velocity[3:6] - self._velocity[3:6]
        velocity_error[6] = 0.0

        pos_omega = 2.0 * math.pi * self._config.mpc_tracking_frequency
        ori_frequency = (
            self._config.mpc_tracking_frequency
            if self._config.mpc_orientation_tracking_frequency is None
            else self._config.mpc_orientation_tracking_frequency
        )
        ori_damping = (
            self._config.mpc_damping_ratio
            if self._config.mpc_orientation_damping_ratio is None
            else self._config.mpc_orientation_damping_ratio
        )
        ori_omega = 2.0 * math.pi * ori_frequency

        desired_acceleration = np.zeros(7, dtype=float)
        desired_acceleration[:3] = (
            pos_omega * pos_omega * error[:3]
            + 2.0 * self._config.mpc_damping_ratio * pos_omega * velocity_error[:3]
        )
        desired_acceleration[3:6] = (
            ori_omega * ori_omega * error[3:6]
            + 2.0 * ori_damping * ori_omega * velocity_error[3:6]
        )

        next_command, flags = self._integrate_limited_acceleration(
            current,
            desired_acceleration,
            raw_gripper,
        )
        final_command, final_limited = self._apply_command_limits(next_command)
        command_limited = command_limited or final_limited
        self._update_state(current, final_command)
        pose, gripper = _split(final_command)
        return SmoothedTarget(
            pose,
            gripper,
            command_limited or gap_filled or deadband_applied or any(flags),
            velocity_limited=flags[0],
            acceleration_limited=flags[1],
            jerk_limited=flags[2],
            command_limited=command_limited,
            gap_filled=gap_filled,
            deadband_applied=deadband_applied,
            mpc_tracking_active=True,
        )

    def _integrate_limited_delta(
        self,
        current: np.ndarray,
        desired_delta: np.ndarray,
        raw_gripper: float,
    ) -> tuple[np.ndarray, tuple[bool, bool, bool]]:
        desired_delta, velocity_limited = _limit_groups(
            desired_delta,
            pos_limit=self._config.max_pos_velocity * self._cmd_dt,
            ori_limit=self._config.max_ori_velocity * self._cmd_dt,
            gripper_limit=float("inf"),
        )
        desired_velocity = desired_delta / self._cmd_dt
        velocity_delta, acceleration_limited = _limit_groups(
            desired_velocity - self._velocity,
            pos_limit=self._config.max_pos_acceleration * self._cmd_dt,
            ori_limit=self._config.max_ori_acceleration * self._cmd_dt,
            gripper_limit=float("inf"),
        )
        desired_velocity = self._velocity + velocity_delta
        desired_acceleration = (desired_velocity - self._velocity) / self._cmd_dt
        return self._integrate_limited_acceleration(
            current,
            desired_acceleration,
            raw_gripper,
            prelimited=(velocity_limited, acceleration_limited),
        )

    def _integrate_limited_acceleration(
        self,
        current: np.ndarray,
        desired_acceleration: np.ndarray,
        raw_gripper: float,
        *,
        prelimited: tuple[bool, bool] = (False, False),
    ) -> tuple[np.ndarray, tuple[bool, bool, bool]]:
        acceleration_delta, jerk_limited = _limit_groups(
            desired_acceleration - self._acceleration,
            pos_limit=self._config.max_pos_jerk * self._cmd_dt,
            ori_limit=self._config.max_ori_jerk * self._cmd_dt,
            gripper_limit=float("inf"),
        )
        desired_acceleration = self._acceleration + acceleration_delta
        desired_acceleration, acceleration_limited = _limit_groups(
            desired_acceleration,
            pos_limit=self._config.max_pos_acceleration,
            ori_limit=self._config.max_ori_acceleration,
            gripper_limit=float("inf"),
        )
        desired_velocity = self._velocity + desired_acceleration * self._cmd_dt
        desired_velocity, velocity_limited = _limit_groups(
            desired_velocity,
            pos_limit=self._config.max_pos_velocity,
            ori_limit=self._config.max_ori_velocity,
            gripper_limit=float("inf"),
        )
        desired_delta = desired_velocity * self._cmd_dt
        desired_delta, step_limited = _limit_groups(
            desired_delta,
            pos_limit=self._config.max_pos_step,
            ori_limit=self._config.max_ori_step,
            gripper_limit=float("inf"),
        )
        next_command = current.copy()
        next_command[:6] = current[:6] + desired_delta[:6]
        next_command[6] = raw_gripper
        return next_command, (
            velocity_limited or prelimited[0],
            acceleration_limited or prelimited[1],
            jerk_limited or step_limited,
        )

    def _update_state(self, previous: np.ndarray, current: np.ndarray) -> None:
        actual_delta = current - previous
        actual_delta[3:6] = wrap_angle_delta(actual_delta[3:6])
        actual_delta[6] = 0.0
        actual_velocity = actual_delta / self._cmd_dt
        actual_acceleration = (actual_velocity - self._velocity) / self._cmd_dt
        self._command = current.copy()
        self._velocity = actual_velocity
        self._acceleration = actual_acceleration

    def _delayed_reference(self) -> tuple[np.ndarray, np.ndarray]:
        window = list(self._reference_window)
        if not window:
            return np.zeros(7, dtype=float), np.zeros(7, dtype=float)
        index = max(0, len(window) - 1 - self._config.mpc_delay_frames)
        reference = window[index].copy()
        previous = reference.copy() if index <= 0 else window[index - 1].copy()
        delta = reference - previous
        delta[3:6] = wrap_angle_delta(delta[3:6])
        delta[6] = 0.0
        velocity = delta / self._cmd_dt
        velocity, _ = _limit_groups(
            velocity,
            pos_limit=self._config.max_pos_velocity,
            ori_limit=self._config.max_ori_velocity,
            gripper_limit=float("inf"),
        )
        return reference, velocity
