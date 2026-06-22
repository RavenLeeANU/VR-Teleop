from __future__ import annotations

import math

import numpy as np

from hand_tracking_sdk.frame import HandFrame
from hand_tracking_sdk.models import HandLandmarks, HandSide, WristPose
from teleop_vr.postprocess import (
    DampingConfig,
    TrajectorySmoother,
    replace_nonfinite_command_values,
)
from teleop_vr.recorder import RecorderConfig, TeleopRecorder
from teleop_vr_send import TargetWindow, TeleopTarget


def _config(
    *,
    enabled: bool = True,
    alpha: float = 1.0,
    max_pos_step: float = 0.1,
    max_ori_step: float = 0.2,
    max_gripper_step: float = 0.01,
    max_pos_velocity: float = 1_000.0,
    max_ori_velocity: float = 1_000.0,
    max_gripper_velocity: float = 1_000.0,
    max_pos_acceleration: float = 1_000.0,
    max_ori_acceleration: float = 1_000.0,
    max_gripper_acceleration: float = 1_000.0,
    max_pos_jerk: float = 1_000.0,
    max_ori_jerk: float = 1_000.0,
    max_gripper_jerk: float = 1_000.0,
    pose_min: np.ndarray | None = None,
    pose_max: np.ndarray | None = None,
    gripper_min: float = 0.0,
    gripper_max: float | None = None,
    max_missing_frames: int = 10,
    sg_position_enabled: bool = False,
    sg_window_size: int = 21,
    sg_poly_order: int = 2,
    orientation_ema_enabled: bool = False,
    orientation_ema_alpha_x: float = 0.15,
    orientation_ema_alpha_y: float = 0.15,
    position_deadband: float = 0.0,
    orientation_deadband: float = 0.0,
    gripper_deadband: float = 0.0,
) -> DampingConfig:
    return DampingConfig(
        enabled=enabled,
        alpha=alpha,
        max_pos_step=max_pos_step,
        max_ori_step=max_ori_step,
        max_gripper_step=max_gripper_step,
        max_pos_velocity=max_pos_velocity,
        max_ori_velocity=max_ori_velocity,
        max_gripper_velocity=max_gripper_velocity,
        max_pos_acceleration=max_pos_acceleration,
        max_ori_acceleration=max_ori_acceleration,
        max_gripper_acceleration=max_gripper_acceleration,
        max_pos_jerk=max_pos_jerk,
        max_ori_jerk=max_ori_jerk,
        max_gripper_jerk=max_gripper_jerk,
        pose_min=pose_min,
        pose_max=pose_max,
        gripper_min=gripper_min,
        gripper_max=gripper_max,
        max_missing_frames=max_missing_frames,
        sg_position_enabled=sg_position_enabled,
        sg_window_size=sg_window_size,
        sg_poly_order=sg_poly_order,
        orientation_ema_enabled=orientation_ema_enabled,
        orientation_ema_alpha_x=orientation_ema_alpha_x,
        orientation_ema_alpha_y=orientation_ema_alpha_y,
        position_deadband=position_deadband,
        orientation_deadband=orientation_deadband,
        gripper_deadband=gripper_deadband,
    )


def test_trajectory_smoother_first_enabled_sample_passes_through() -> None:
    smoother = TrajectorySmoother(_config())
    raw_pose = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])

    target = smoother.process(raw_pose, 0.04)

    assert np.allclose(target.pose_6d, raw_pose)
    assert target.gripper_pos == 0.04
    assert target.limited is False


def test_trajectory_smoother_disabled_passes_through_without_state_lag() -> None:
    smoother = TrajectorySmoother(_config(enabled=False))
    first_pose = np.zeros(6)
    second_pose = np.array([1.0, -2.0, 3.0, 0.4, -0.5, 0.6])

    smoother.process(first_pose, 0.01)
    target = smoother.process(second_pose, 0.08)

    assert np.allclose(target.pose_6d, second_pose)
    assert target.gripper_pos == 0.08
    assert target.limited is False


def test_trajectory_smoother_clips_gripper_even_when_disabled() -> None:
    smoother = TrajectorySmoother(
        _config(enabled=False, gripper_min=0.0, gripper_max=0.08)
    )

    high = smoother.process(np.zeros(6), 0.20)
    low = smoother.process(np.zeros(6), -0.05)

    assert math.isclose(high.gripper_pos, 0.08)
    assert math.isclose(low.gripper_pos, 0.0)
    assert high.command_limited is True
    assert low.command_limited is True


def test_trajectory_smoother_limits_large_position_orientation_and_gripper_jumps() -> None:
    smoother = TrajectorySmoother(
        _config(
            max_pos_acceleration=1_000_000.0,
            max_ori_acceleration=1_000_000.0,
            max_gripper_acceleration=1_000_000.0,
            max_pos_jerk=1_000_000.0,
            max_ori_jerk=1_000_000.0,
            max_gripper_jerk=1_000_000.0,
        )
    )
    smoother.process(np.zeros(6), 0.0)

    target = smoother.process(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), 0.1)

    assert np.allclose(target.pose_6d[:3], [0.1, 0.0, 0.0])
    assert np.allclose(target.pose_6d[3:], [0.0, 0.2, 0.0])
    assert math.isclose(target.gripper_pos, 0.01)
    assert target.step_limited is True
    assert target.limited is True


def test_trajectory_smoother_alpha_blends_limited_step() -> None:
    smoother = TrajectorySmoother(
        _config(
            alpha=0.5,
            max_pos_acceleration=1_000_000.0,
            max_ori_acceleration=1_000_000.0,
            max_gripper_acceleration=1_000_000.0,
            max_pos_jerk=1_000_000.0,
            max_ori_jerk=1_000_000.0,
            max_gripper_jerk=1_000_000.0,
        )
    )
    smoother.process(np.zeros(6), 0.0)

    target = smoother.process(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), 0.1)

    assert np.allclose(target.pose_6d[:3], [0.05, 0.0, 0.0])
    assert np.allclose(target.pose_6d[3:], [0.0, 0.1, 0.0])
    assert math.isclose(target.gripper_pos, 0.005)
    assert target.step_limited is True


def test_trajectory_smoother_limits_velocity() -> None:
    smoother = TrajectorySmoother(
        _config(
            max_pos_step=100.0,
            max_ori_step=100.0,
            max_gripper_step=100.0,
            max_pos_velocity=0.2,
            max_ori_velocity=0.4,
            max_gripper_velocity=0.1,
            max_pos_acceleration=1_000_000.0,
            max_ori_acceleration=1_000_000.0,
            max_gripper_acceleration=1_000_000.0,
            max_pos_jerk=1_000_000.0,
            max_ori_jerk=1_000_000.0,
            max_gripper_jerk=1_000_000.0,
        ),
        cmd_dt=0.1,
    )
    smoother.process(np.zeros(6), 0.0)

    target = smoother.process(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), 1.0)

    assert np.allclose(target.pose_6d[:3], [0.02, 0.0, 0.0])
    assert np.allclose(target.pose_6d[3:], [0.0, 0.04, 0.0])
    assert math.isclose(target.gripper_pos, 0.01)
    assert target.velocity_limited is True


def test_trajectory_smoother_limits_acceleration() -> None:
    smoother = TrajectorySmoother(
        _config(
            max_pos_step=100.0,
            max_ori_step=100.0,
            max_gripper_step=100.0,
            max_pos_velocity=100.0,
            max_ori_velocity=100.0,
            max_gripper_velocity=100.0,
            max_pos_acceleration=0.5,
            max_ori_acceleration=1.0,
            max_gripper_acceleration=0.25,
            max_pos_jerk=1_000_000.0,
            max_ori_jerk=1_000_000.0,
            max_gripper_jerk=1_000_000.0,
        ),
        cmd_dt=0.1,
    )
    smoother.process(np.zeros(6), 0.0)

    target = smoother.process(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), 1.0)

    assert np.allclose(target.pose_6d[:3], [0.005, 0.0, 0.0])
    assert np.allclose(target.pose_6d[3:], [0.0, 0.01, 0.0])
    assert math.isclose(target.gripper_pos, 0.0025)
    assert target.acceleration_limited is True


def test_trajectory_smoother_limits_jerk() -> None:
    smoother = TrajectorySmoother(
        _config(
            max_pos_step=100.0,
            max_ori_step=100.0,
            max_gripper_step=100.0,
            max_pos_velocity=100.0,
            max_ori_velocity=100.0,
            max_gripper_velocity=100.0,
            max_pos_acceleration=100.0,
            max_ori_acceleration=100.0,
            max_gripper_acceleration=100.0,
            max_pos_jerk=2.0,
            max_ori_jerk=4.0,
            max_gripper_jerk=1.0,
        ),
        cmd_dt=0.1,
    )
    smoother.process(np.zeros(6), 0.0)

    target = smoother.process(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), 1.0)

    assert np.allclose(target.pose_6d[:3], [0.002, 0.0, 0.0])
    assert np.allclose(target.pose_6d[3:], [0.0, 0.004, 0.0])
    assert math.isclose(target.gripper_pos, 0.001)
    assert target.jerk_limited is True


def test_trajectory_smoother_applies_software_limits_to_pose_and_gripper() -> None:
    smoother = TrajectorySmoother(
        _config(
            pose_min=np.array([-0.5, -0.5, -0.5, -0.2, -0.2, -0.2]),
            pose_max=np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.2]),
            gripper_min=0.01,
            gripper_max=0.06,
        )
    )

    target = smoother.process(np.array([2.0, 0.0, -2.0, 1.0, 0.0, -1.0]), 0.2)

    assert np.allclose(target.pose_6d, [0.5, 0.0, -0.5, 0.2, 0.0, -0.2])
    assert math.isclose(target.gripper_pos, 0.06)
    assert target.command_limited is True
    assert target.limited is True


def test_trajectory_smoother_fills_short_nonfinite_gap() -> None:
    smoother = TrajectorySmoother(_config(max_missing_frames=2))
    smoother.process(np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]), 0.04)

    target = smoother.process(np.array([math.nan, 4.0, 5.0, 0.2, 0.3, 0.4]), math.nan)

    assert math.isfinite(target.pose_6d[0])
    assert math.isfinite(target.gripper_pos)
    assert target.gap_filled is True


def test_trajectory_smoother_deadband_holds_small_input_changes() -> None:
    smoother = TrajectorySmoother(
        _config(
            enabled=False,
            position_deadband=0.01,
            orientation_deadband=0.05,
            gripper_deadband=0.005,
        )
    )

    first = smoother.process(np.zeros(6), 0.02)
    small = smoother.process(np.array([0.003, 0.0, 0.0, 0.0, 0.01, 0.0]), 0.022)
    large = smoother.process(np.array([0.02, 0.0, 0.0, 0.0, 0.08, 0.0]), 0.03)

    assert np.allclose(first.pose_6d, np.zeros(6))
    assert np.allclose(small.pose_6d, np.zeros(6))
    assert math.isclose(small.gripper_pos, 0.02)
    assert small.deadband_applied is True
    assert np.allclose(large.pose_6d, [0.02, 0.0, 0.0, 0.0, 0.08, 0.0])
    assert math.isclose(large.gripper_pos, 0.03)


def test_trajectory_smoother_causal_sg_reduces_position_noise() -> None:
    rng = np.random.default_rng(11)
    time_s = np.arange(0.0, 2.0, 0.01)
    clean_x = 0.2 + 0.02 * time_s
    noisy_x = clean_x + rng.normal(0.0, 0.006, size=len(time_s))
    smoother = TrajectorySmoother(
        _config(
            enabled=False,
            sg_position_enabled=True,
            sg_window_size=21,
            sg_poly_order=2,
        )
    )

    smoothed_x = []
    for x_value in noisy_x:
        target = smoother.process(np.array([x_value, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.0)
        smoothed_x.append(target.pose_6d[0])

    start = 21
    assert np.std(np.asarray(smoothed_x[start:]) - clean_x[start:]) < np.std(
        noisy_x[start:] - clean_x[start:]
    )


def test_trajectory_smoother_orientation_ema_keeps_finite_continuous_output() -> None:
    smoother = TrajectorySmoother(
        _config(
            enabled=False,
            orientation_ema_enabled=True,
            orientation_ema_alpha_x=0.15,
            orientation_ema_alpha_y=0.15,
        )
    )

    first = smoother.process(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3.10]), 0.0)
    second = smoother.process(np.array([0.0, 0.0, 0.0, 0.0, 0.0, -3.10]), 0.0)

    assert np.all(np.isfinite(second.pose_6d))
    assert abs(second.pose_6d[5] - first.pose_6d[5]) < 1.0
    assert second.orientation_smoothed is True


def test_teleop_recorder_writes_raw_and_converted_csv(tmp_path) -> None:
    recorder = TeleopRecorder(
        RecorderConfig(enabled=True, output_dir=str(tmp_path), prefix="sample")
    )
    frame = HandFrame(
        side=HandSide.LEFT,
        frame_id="left",
        wrist=WristPose(x=1.0, y=2.0, z=3.0, qx=0.1, qy=0.2, qz=0.3, qw=0.9),
        landmarks=HandLandmarks(points=tuple((float(i), 0.0, 0.0) for i in range(21))),
        sequence_id=7,
        recv_ts_ns=100,
        recv_time_unix_ns=200,
        source_ts_ns=300,
        wrist_recv_ts_ns=90,
        landmarks_recv_ts_ns=95,
        source_frame_seq=6,
    )

    record_index = recorder.add_frame(frame, received_at=1.25)
    assert record_index == 0
    recorder.add_converted_target(
        record_index=record_index,
        side=HandSide.LEFT.value,
        sequence_id=7,
        received_at=1.25,
        vr_pos=np.array([1.0, 2.0, 3.0]),
        pos_delta=np.array([0.1, 0.2, 0.3]),
        rpy_delta=np.array([0.01, 0.02, 0.03]),
        raw_pose_6d=np.array([1.1, 2.2, 3.3, 0.1, 0.2, 0.3]),
        raw_gripper_pos=0.04,
    )
    recorder.mark_sent(
        record_index=record_index,
        sent_at=1.30,
        sent_pose_6d=np.array([1.0, 2.0, 3.0, 0.0, 0.1, 0.2]),
        sent_gripper_pos=0.03,
    )
    recorder.mark_sent(
        record_index=record_index,
        sent_at=1.31,
        sent_pose_6d=np.array([1.1, 2.1, 3.1, 0.0, 0.1, 0.2]),
        sent_gripper_pos=0.031,
    )

    saved = recorder.save()

    assert saved is not None
    raw_path, converted_path, robot_state_path = saved
    raw_text = open(raw_path, encoding="utf-8").read()
    converted_text = open(converted_path, encoding="utf-8").read()
    assert "wrist_x" in raw_text
    assert "landmark_20_z" in raw_text
    assert "sent_target_yaw" in converted_text
    assert "target_age_s" not in converted_text
    assert "controller_timestamp_s" not in converted_text
    assert "command_timestamp_s" not in converted_text
    assert "smoothing_limited" not in converted_text
    assert converted_text.count("\n") == 3
    assert "1.31" in converted_text
    assert "sample" in raw_path
    assert robot_state_path is None


def test_teleop_recorder_writes_robot_state_when_enabled(tmp_path) -> None:
    recorder = TeleopRecorder(
        RecorderConfig(
            enabled=True,
            output_dir=str(tmp_path),
            prefix="sample",
            record_robot_state=True,
        )
    )

    recorder.add_robot_state(
        sent_at=2.0,
        state_pose_6d=np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
        state_gripper_pos=0.02,
        cmd_pose_6d=np.array([1.5, 2.5, 3.5, 0.2, 0.4, 0.6]),
        cmd_gripper_pos=0.03,
    )

    saved = recorder.save()

    assert saved is not None
    _, _, robot_state_path = saved
    assert robot_state_path is not None
    robot_state_text = open(robot_state_path, encoding="utf-8").read()
    assert "state_x" in robot_state_text
    assert "cmd_yaw" in robot_state_text
    assert "error_gripper_pos" in robot_state_text
    assert "0.5" in robot_state_text


def test_replace_nonfinite_command_values_uses_previous_sent_values() -> None:
    pose, gripper, replaced = replace_nonfinite_command_values(
        np.array([1.0, math.nan, 3.0, math.inf, 5.0, 6.0]),
        math.nan,
        np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        0.03,
    )

    assert replaced is True
    assert np.allclose(pose, [1.0, 20.0, 3.0, 40.0, 5.0, 6.0])
    assert math.isclose(gripper, 0.03)


def test_replace_nonfinite_command_values_requires_previous_values() -> None:
    try:
        replace_nonfinite_command_values(
            np.array([1.0, math.nan, 3.0, 4.0, 5.0, 6.0]),
            0.02,
            None,
            None,
        )
    except ValueError as exc:
        assert "no previous command fallback" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_target_window_interpolates_between_received_targets() -> None:
    window = TargetWindow(max_size=4)
    first = TeleopTarget(
        side=HandSide.LEFT,
        sequence_id=1,
        vr_pos=np.array([0.0, 0.0, 0.0]),
        pos_delta=np.array([0.0, 0.0, 0.0]),
        rpy_delta=np.array([0.0, 0.0, 0.0]),
        raw_pose_6d=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        raw_gripper_pos=0.0,
        received_at=1.0,
        record_index=10,
    )
    second = TeleopTarget(
        side=HandSide.LEFT,
        sequence_id=2,
        vr_pos=np.array([2.0, 4.0, 6.0]),
        pos_delta=np.array([2.0, 4.0, 6.0]),
        rpy_delta=np.array([0.2, 0.4, 0.6]),
        raw_pose_6d=np.array([2.0, 4.0, 6.0, 0.2, 0.4, 0.6]),
        raw_gripper_pos=0.08,
        received_at=3.0,
        record_index=11,
    )

    window.append(first)
    window.append(second)
    target, interpolated = window.sample(2.0)

    assert target is not None
    assert interpolated is True
    assert np.allclose(target.raw_pose_6d, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    assert math.isclose(target.raw_gripper_pos, 0.04)
    assert target.record_index == 11
