"""Offline smoothing experiment for recorded VR teleop trajectories.

Examples:
    python -m quest_hand.smooth_trajectory_experiment records/test02_vr_raw.csv --save records/test02_raw_smooth.png --no-show
    python -m quest_hand.smooth_trajectory_experiment records/test02_converted.csv --source converted --alpha 0.3 --max-pos-velocity 0.05
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from quest_hand.postprocess import DampingConfig, TrajectorySmoother, load_damping_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay recorded trajectories through TrajectorySmoother for offline tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path, help="Path to *_vr_raw.csv or *_converted.csv.")
    parser.add_argument(
        "--source",
        choices=["auto", "raw-wrist", "converted"],
        default="auto",
        help="Input trajectory source. raw-wrist uses wrist_x/y/z; converted uses raw_target_*.",
    )
    parser.add_argument("--save", type=Path, default=None, help="Optional image output path.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for recomputed smoothed trajectory.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not show matplotlib window.")
    parser.add_argument("--cmd-dt", type=float, default=0.01, help="Smoother command period in seconds.")
    parser.add_argument(
        "--postprocess-config",
        type=Path,
        default=None,
        help="Optional YAML file for smoothing/limit parameters. CLI smoothing values are ignored when set.",
    )
    parser.add_argument("--alpha", type=float, default=0.6, help="Smoothing alpha.")
    parser.add_argument("--max-pos-step", type=float, default=0.01)
    parser.add_argument("--max-ori-step", type=float, default=0.10)
    parser.add_argument("--max-gripper-step", type=float, default=0.005)
    parser.add_argument("--max-pos-velocity", type=float, default=0.50)
    parser.add_argument("--max-ori-velocity", type=float, default=1.50)
    parser.add_argument("--max-gripper-velocity", type=float, default=0.08)
    parser.add_argument("--max-pos-acceleration", type=float, default=3.0)
    parser.add_argument("--max-ori-acceleration", type=float, default=8.0)
    parser.add_argument("--max-gripper-acceleration", type=float, default=0.40)
    parser.add_argument("--max-pos-jerk", type=float, default=300.0)
    parser.add_argument("--max-ori-jerk", type=float, default=800.0)
    parser.add_argument("--max-gripper-jerk", type=float, default=40.0)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=None)
    parser.add_argument("--gripper-closed-threshold", type=float, default=None)
    parser.add_argument("--gripper-open-threshold", type=float, default=None)
    parser.add_argument("--max-missing-frames", type=int, default=10)
    parser.add_argument("--position-deadband", type=float, default=0.0)
    parser.add_argument("--orientation-deadband", type=float, default=0.0)
    parser.add_argument("--gripper-deadband", type=float, default=0.0)
    parser.add_argument("--deadband-velocity-threshold", type=float, default=None)
    parser.add_argument("--stationary-hold-enabled", action="store_true")
    parser.add_argument("--stationary-window-size", type=int, default=8)
    parser.add_argument("--stationary-pos-range", type=float, default=0.006)
    parser.add_argument("--stationary-ori-range", type=float, default=0.02)
    parser.add_argument("--stationary-command-pos-threshold", type=float, default=0.010)
    parser.add_argument("--stationary-command-ori-threshold", type=float, default=0.03)
    parser.add_argument("--stationary-frames", type=int, default=3)
    parser.add_argument("--input-jump-protection-enabled", action="store_true")
    parser.add_argument("--max-input-pos-jump", type=float, default=0.03)
    parser.add_argument("--max-input-ori-jump", type=float, default=0.25)
    parser.add_argument("--transition-confirm-frames", type=int, default=3)
    parser.add_argument("--stationary-hold-cooldown-frames", type=int, default=20)
    parser.add_argument("--mpc-tracking-enabled", action="store_true")
    parser.add_argument("--mpc-delay-frames", type=int, default=5)
    parser.add_argument("--mpc-tracking-frequency", type=float, default=12.0)
    parser.add_argument("--mpc-damping-ratio", type=float, default=1.0)
    parser.add_argument("--mpc-reference-velocity-gain", type=float, default=1.0)
    parser.add_argument("--mpc-orientation-tracking-frequency", type=float, default=None)
    parser.add_argument("--mpc-orientation-damping-ratio", type=float, default=None)
    parser.add_argument("--mpc-orientation-reference-velocity-gain", type=float, default=None)
    parser.add_argument("--manifold-spline-enabled", action="store_true")
    parser.add_argument("--manifold-spline-position-tension", type=float, default=0.5)
    parser.add_argument("--manifold-spline-orientation-tension", type=float, default=0.5)
    parser.add_argument("--sg-position-enabled", action="store_true")
    parser.add_argument("--sg-window-size", type=int, default=21)
    parser.add_argument("--sg-poly-order", type=int, default=2)
    parser.add_argument("--orientation-ema-enabled", action="store_true")
    parser.add_argument("--orientation-ema-alpha-x", type=float, default=0.15)
    parser.add_argument("--orientation-ema-alpha-y", type=float, default=0.15)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _to_float(value: str | None) -> float:
    if value is None:
        return math.nan
    text = value.strip()
    if text == "" or text.lower() == "none":
        return math.nan
    return float(text)


def _series(rows: Sequence[dict[str, str]], column: str) -> list[float]:
    return [_to_float(row.get(column)) for row in rows]


def _first_finite(values: Sequence[float]) -> float:
    for value in values:
        if math.isfinite(value):
            return value
    return math.nan


def _relative_time(rows: Sequence[dict[str, str]], column: str) -> list[float]:
    values = _series(rows, column)
    start = _first_finite(values)
    if not math.isfinite(start):
        return [float(index) for index, _ in enumerate(values)]
    return [value - start if math.isfinite(value) else math.nan for value in values]


def _has_columns(rows: Sequence[dict[str, str]], columns: Sequence[str]) -> bool:
    return bool(rows) and all(column in rows[0] for column in columns)


def _resolve_source(rows: Sequence[dict[str, str]], source: str) -> str:
    if source != "auto":
        return source
    if _has_columns(rows, ["raw_target_x", "raw_target_y", "raw_target_z"]):
        return "converted"
    if _has_columns(rows, ["wrist_x", "wrist_y", "wrist_z"]):
        return "raw-wrist"
    raise RuntimeError("Cannot infer source: expected raw wrist or converted raw_target columns.")


def _load_input_trajectory(
    rows: Sequence[dict[str, str]],
    source: str,
) -> tuple[list[float], np.ndarray, np.ndarray, str]:
    if source == "converted":
        time_s = _relative_time(rows, "sent_at_monotonic_s")
        pose = np.column_stack(
            [
                _series(rows, "raw_target_x"),
                _series(rows, "raw_target_y"),
                _series(rows, "raw_target_z"),
                _series(rows, "raw_target_roll"),
                _series(rows, "raw_target_pitch"),
                _series(rows, "raw_target_yaw"),
            ]
        )
        gripper = np.asarray(_series(rows, "raw_gripper_pos"), dtype=float)
        return time_s, pose, gripper, "converted raw_target"

    time_s = _relative_time(rows, "received_at_monotonic_s")
    pose = np.zeros((len(rows), 6), dtype=float)
    pose[:, 0] = _series(rows, "wrist_x")
    pose[:, 1] = _series(rows, "wrist_y")
    pose[:, 2] = _series(rows, "wrist_z")
    gripper = np.zeros(len(rows), dtype=float)
    return time_s, pose, gripper, "raw wrist"


def _build_config(args: argparse.Namespace) -> DampingConfig:
    if args.postprocess_config is not None:
        return load_damping_config(args.postprocess_config)
    return DampingConfig(
        enabled=True,
        alpha=args.alpha,
        max_pos_step=args.max_pos_step,
        max_ori_step=args.max_ori_step,
        max_gripper_step=args.max_gripper_step,
        max_pos_velocity=args.max_pos_velocity,
        max_ori_velocity=args.max_ori_velocity,
        max_gripper_velocity=args.max_gripper_velocity,
        max_pos_acceleration=args.max_pos_acceleration,
        max_ori_acceleration=args.max_ori_acceleration,
        max_gripper_acceleration=args.max_gripper_acceleration,
        max_pos_jerk=args.max_pos_jerk,
        max_ori_jerk=args.max_ori_jerk,
        max_gripper_jerk=args.max_gripper_jerk,
        gripper_min=args.gripper_min,
        gripper_max=args.gripper_max,
        gripper_closed_threshold=args.gripper_closed_threshold,
        gripper_open_threshold=args.gripper_open_threshold,
        max_missing_frames=args.max_missing_frames,
        position_deadband=args.position_deadband,
        orientation_deadband=args.orientation_deadband,
        gripper_deadband=args.gripper_deadband,
        deadband_velocity_threshold=args.deadband_velocity_threshold,
        stationary_hold_enabled=args.stationary_hold_enabled,
        stationary_window_size=args.stationary_window_size,
        stationary_pos_range=args.stationary_pos_range,
        stationary_ori_range=args.stationary_ori_range,
        stationary_command_pos_threshold=args.stationary_command_pos_threshold,
        stationary_command_ori_threshold=args.stationary_command_ori_threshold,
        stationary_frames=args.stationary_frames,
        input_jump_protection_enabled=args.input_jump_protection_enabled,
        max_input_pos_jump=args.max_input_pos_jump,
        max_input_ori_jump=args.max_input_ori_jump,
        transition_confirm_frames=args.transition_confirm_frames,
        stationary_hold_cooldown_frames=args.stationary_hold_cooldown_frames,
        mpc_tracking_enabled=args.mpc_tracking_enabled,
        mpc_delay_frames=args.mpc_delay_frames,
        mpc_tracking_frequency=args.mpc_tracking_frequency,
        mpc_damping_ratio=args.mpc_damping_ratio,
        mpc_reference_velocity_gain=args.mpc_reference_velocity_gain,
        mpc_orientation_tracking_frequency=args.mpc_orientation_tracking_frequency,
        mpc_orientation_damping_ratio=args.mpc_orientation_damping_ratio,
        mpc_orientation_reference_velocity_gain=args.mpc_orientation_reference_velocity_gain,
        manifold_spline_enabled=args.manifold_spline_enabled,
        manifold_spline_position_tension=args.manifold_spline_position_tension,
        manifold_spline_orientation_tension=args.manifold_spline_orientation_tension,
        sg_position_enabled=args.sg_position_enabled,
        sg_window_size=args.sg_window_size,
        sg_poly_order=args.sg_poly_order,
        orientation_ema_enabled=args.orientation_ema_enabled,
        orientation_ema_alpha_x=args.orientation_ema_alpha_x,
        orientation_ema_alpha_y=args.orientation_ema_alpha_y,
    )


def _run_smoother(
    pose: np.ndarray,
    gripper: np.ndarray,
    config: DampingConfig,
    *,
    cmd_dt: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    smoother = TrajectorySmoother(config, cmd_dt=cmd_dt)
    smoothed_pose = np.zeros_like(pose, dtype=float)
    smoothed_gripper = np.zeros_like(gripper, dtype=float)
    flags = {
        "limited": np.zeros(len(pose), dtype=bool),
        "step_limited": np.zeros(len(pose), dtype=bool),
        "velocity_limited": np.zeros(len(pose), dtype=bool),
        "acceleration_limited": np.zeros(len(pose), dtype=bool),
        "jerk_limited": np.zeros(len(pose), dtype=bool),
        "command_limited": np.zeros(len(pose), dtype=bool),
        "gap_filled": np.zeros(len(pose), dtype=bool),
        "deadband_applied": np.zeros(len(pose), dtype=bool),
        "position_smoothed": np.zeros(len(pose), dtype=bool),
        "orientation_smoothed": np.zeros(len(pose), dtype=bool),
        "stationary_held": np.zeros(len(pose), dtype=bool),
        "input_spike_rejected": np.zeros(len(pose), dtype=bool),
        "transition_active": np.zeros(len(pose), dtype=bool),
        "mpc_tracking_active": np.zeros(len(pose), dtype=bool),
        "manifold_spline_active": np.zeros(len(pose), dtype=bool),
    }

    last_pose: np.ndarray | None = None
    last_gripper: float | None = None
    for index, (raw_pose, raw_gripper) in enumerate(zip(pose, gripper, strict=True)):
        if not np.all(np.isfinite(raw_pose)) or not math.isfinite(float(raw_gripper)):
            if last_pose is None or last_gripper is None:
                smoothed_pose[index] = np.full(6, np.nan)
                smoothed_gripper[index] = math.nan
                continue
            raw_pose = np.where(np.isfinite(raw_pose), raw_pose, last_pose)
            raw_gripper = float(raw_gripper) if math.isfinite(float(raw_gripper)) else last_gripper

        target = smoother.process(raw_pose, float(raw_gripper))
        smoothed_pose[index] = target.pose_6d
        smoothed_gripper[index] = target.gripper_pos
        flags["limited"][index] = target.limited
        flags["step_limited"][index] = target.step_limited
        flags["velocity_limited"][index] = target.velocity_limited
        flags["acceleration_limited"][index] = target.acceleration_limited
        flags["jerk_limited"][index] = target.jerk_limited
        flags["command_limited"][index] = target.command_limited
        flags["gap_filled"][index] = target.gap_filled
        flags["deadband_applied"][index] = target.deadband_applied
        flags["position_smoothed"][index] = target.position_smoothed
        flags["orientation_smoothed"][index] = target.orientation_smoothed
        flags["stationary_held"][index] = target.stationary_held
        flags["input_spike_rejected"][index] = target.input_spike_rejected
        flags["transition_active"][index] = target.transition_active
        flags["mpc_tracking_active"][index] = target.mpc_tracking_active
        flags["manifold_spline_active"][index] = target.manifold_spline_active
        last_pose = target.pose_6d
        last_gripper = target.gripper_pos

    return smoothed_pose, smoothed_gripper, flags


def _plot_experiment(
    time_s: Sequence[float],
    raw_pose: np.ndarray,
    raw_gripper: np.ndarray,
    smoothed_pose: np.ndarray,
    smoothed_gripper: np.ndarray,
    flags: dict[str, np.ndarray],
    *,
    title: str,
) -> object:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required: pip install matplotlib") from exc

    fig, axes = plt.subplots(4, 3, figsize=(18, 12), sharex=False)
    fig.suptitle(title)

    for col, axis_name in enumerate(("x", "y", "z")):
        axes[0, col].set_title(f"Position {axis_name.upper()}")
        axes[0, col].plot(time_s, raw_pose[:, col], label="raw")
        axes[0, col].plot(time_s, smoothed_pose[:, col], label="smoothed")
        axes[0, col].set_ylabel("cm")
        axes[0, col].legend(loc="best")
        axes[0, col].grid(True, alpha=0.3)

    for col, axis_name in enumerate(("roll", "pitch", "yaw")):
        axes[1, col].set_title(axis_name.title())
        axes[1, col].plot(time_s, raw_pose[:, col + 3], label="raw")
        axes[1, col].plot(time_s, smoothed_pose[:, col + 3], label="smoothed")
        axes[1, col].set_ylabel("rad")
        axes[1, col].legend(loc="best")
        axes[1, col].grid(True, alpha=0.3)

    axes[2, 0].set_title("Gripper")
    axes[2, 0].plot(time_s, raw_gripper, label="raw")
    axes[2, 0].plot(time_s, smoothed_gripper, label="smoothed")
    axes[2, 0].set_ylabel("cm")
    axes[2, 0].legend(loc="best")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].set_title("Position Delta Norm")
    pos_delta = np.linalg.norm(smoothed_pose[:, :3] - raw_pose[:, :3], axis=1)
    axes[2, 1].plot(time_s, pos_delta, label="|smoothed - raw|")
    axes[2, 1].set_ylabel("cm")
    axes[2, 1].legend(loc="best")
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 2].set_title("Orientation Delta Norm")
    ori_delta = np.linalg.norm(smoothed_pose[:, 3:6] - raw_pose[:, 3:6], axis=1)
    axes[2, 2].plot(time_s, ori_delta, label="|smoothed - raw|")
    axes[2, 2].set_ylabel("rad")
    axes[2, 2].legend(loc="best")
    axes[2, 2].grid(True, alpha=0.3)

    flag_names = [
        "step_limited",
        "velocity_limited",
        "acceleration_limited",
        "jerk_limited",
        "command_limited",
        "deadband_applied",
        "position_smoothed",
        "orientation_smoothed",
        "stationary_held",
        "input_spike_rejected",
        "transition_active",
        "mpc_tracking_active",
        "manifold_spline_active",
        "gap_filled",
    ]
    axes[3, 0].set_title("Limiter Flags")
    for offset, name in enumerate(flag_names):
        axes[3, 0].step(time_s, flags[name].astype(float) + offset, where="post", label=name)
    axes[3, 0].set_yticks([offset + 0.5 for offset in range(len(flag_names))], flag_names)
    axes[3, 0].grid(True, alpha=0.3)

    axes[3, 1].set_title("Summary")
    axes[3, 1].axis("off")
    summary = [
        f"samples={len(time_s)}",
        f"limited={int(flags['limited'].sum())}",
        *(f"{name}={int(flags[name].sum())}" for name in flag_names),
    ]
    axes[3, 1].text(0.0, 1.0, "\n".join(summary), va="top", ha="left", family="monospace")
    axes[3, 2].axis("off")

    for row in axes:
        for ax in row:
            if ax.axison:
                ax.set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def _write_output_csv(
    path: Path,
    time_s: Sequence[float],
    raw_pose: np.ndarray,
    raw_gripper: np.ndarray,
    smoothed_pose: np.ndarray,
    smoothed_gripper: np.ndarray,
    flags: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time_s",
        "raw_x",
        "raw_y",
        "raw_z",
        "raw_roll",
        "raw_pitch",
        "raw_yaw",
        "raw_gripper",
        "smoothed_x",
        "smoothed_y",
        "smoothed_z",
        "smoothed_roll",
        "smoothed_pitch",
        "smoothed_yaw",
        "smoothed_gripper",
        *flags.keys(),
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, time_value in enumerate(time_s):
            row = {
                "time_s": time_value,
                "raw_x": raw_pose[index, 0],
                "raw_y": raw_pose[index, 1],
                "raw_z": raw_pose[index, 2],
                "raw_roll": raw_pose[index, 3],
                "raw_pitch": raw_pose[index, 4],
                "raw_yaw": raw_pose[index, 5],
                "raw_gripper": raw_gripper[index],
                "smoothed_x": smoothed_pose[index, 0],
                "smoothed_y": smoothed_pose[index, 1],
                "smoothed_z": smoothed_pose[index, 2],
                "smoothed_roll": smoothed_pose[index, 3],
                "smoothed_pitch": smoothed_pose[index, 4],
                "smoothed_yaw": smoothed_pose[index, 5],
                "smoothed_gripper": smoothed_gripper[index],
            }
            row.update({name: bool(values[index]) for name, values in flags.items()})
            writer.writerow(row)


def _finite_stats(values: np.ndarray) -> str:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return "n/a"
    return (
        f"mean={float(np.mean(finite_values)):.6g} "
        f"p95={float(np.percentile(finite_values, 95)):.6g} "
        f"max={float(np.max(finite_values)):.6g}"
    )


def _print_summary(
    raw_pose: np.ndarray,
    raw_gripper: np.ndarray,
    smoothed_pose: np.ndarray,
    smoothed_gripper: np.ndarray,
    flags: dict[str, np.ndarray],
) -> None:
    pos_delta = np.linalg.norm(smoothed_pose[:, :3] - raw_pose[:, :3], axis=1)
    ori_delta = np.linalg.norm(smoothed_pose[:, 3:6] - raw_pose[:, 3:6], axis=1)
    gripper_delta = np.abs(smoothed_gripper - raw_gripper)
    print("smoothing summary:")
    print(f"  position delta norm: {_finite_stats(pos_delta)}")
    print(f"  orientation delta norm: {_finite_stats(ori_delta)}")
    print(f"  gripper delta: {_finite_stats(gripper_delta)}")
    for name, values in flags.items():
        print(f"  {name}: {int(values.sum())}")


def main() -> int:
    args = _parse_args()
    rows = _read_csv(args.csv_path.expanduser().resolve())
    if not rows:
        raise RuntimeError(f"No rows in {args.csv_path}")
    source = _resolve_source(rows, args.source)
    time_s, raw_pose, raw_gripper, source_label = _load_input_trajectory(rows, source)
    smoothed_pose, smoothed_gripper, flags = _run_smoother(
        raw_pose,
        raw_gripper,
        _build_config(args),
        cmd_dt=args.cmd_dt,
    )
    _print_summary(raw_pose, raw_gripper, smoothed_pose, smoothed_gripper, flags)

    fig = _plot_experiment(
        time_s,
        raw_pose,
        raw_gripper,
        smoothed_pose,
        smoothed_gripper,
        flags,
        title=f"Smoothing Experiment: {source_label}",
    )
    if args.output_csv is not None:
        output_csv = args.output_csv.expanduser().resolve()
        _write_output_csv(
            output_csv,
            time_s,
            raw_pose,
            raw_gripper,
            smoothed_pose,
            smoothed_gripper,
            flags,
        )
        print(f"saved recomputed trajectory to {output_csv}")
    if args.save is not None:
        save_path = args.save.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"saved plot to {save_path}")
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
