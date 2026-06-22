"""Generate mock trajectories and compare raw vs smoothed commands.

Examples:
    python -m teleop_vr.mock_smoothing_test --no-show
    python -m teleop_vr.mock_smoothing_test --postprocess-config teleop_vr/postprocess_config.yaml --no-show
    python -m teleop_vr.mock_smoothing_test --alpha 0.35 --max-pos-velocity 0.12 --save records/tuned_mock_smoothing.png
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from teleop_vr.postprocess import DampingConfig, TrajectorySmoother, load_damping_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mock a noisy teleop trajectory and visualize TrajectorySmoother output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--duration", type=float, default=8.0, help="Mock trajectory length in seconds.")
    parser.add_argument("--cmd-dt", type=float, default=0.01, help="Smoother command period in seconds.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for repeatable jitter/spikes.")
    parser.add_argument("--pos-noise", type=float, default=0.004, help="Position jitter std in meters.")
    parser.add_argument("--ori-noise", type=float, default=0.025, help="RPY jitter std in radians.")
    parser.add_argument("--gripper-noise", type=float, default=0.002, help="Gripper jitter std in meters.")
    parser.add_argument("--spike-count", type=int, default=8, help="Number of injected jump spikes.")
    parser.add_argument(
        "--nan-count",
        type=int,
        default=4,
        help="Number of injected NaN samples, used to test fallback behavior.",
    )
    parser.add_argument(
        "--postprocess-config",
        type=Path,
        default=None,
        help="Optional YAML config. CLI smoothing values override this file.",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Override smoothing alpha.")
    parser.add_argument("--max-pos-step", type=float, default=None)
    parser.add_argument("--max-ori-step", type=float, default=None)
    parser.add_argument("--max-gripper-step", type=float, default=None)
    parser.add_argument("--max-pos-velocity", type=float, default=None)
    parser.add_argument("--max-ori-velocity", type=float, default=None)
    parser.add_argument("--max-gripper-velocity", type=float, default=None)
    parser.add_argument("--max-pos-acceleration", type=float, default=None)
    parser.add_argument("--max-ori-acceleration", type=float, default=None)
    parser.add_argument("--max-gripper-acceleration", type=float, default=None)
    parser.add_argument("--max-pos-jerk", type=float, default=None)
    parser.add_argument("--max-ori-jerk", type=float, default=None)
    parser.add_argument("--max-gripper-jerk", type=float, default=None)
    parser.add_argument("--max-missing-frames", type=int, default=None)
    parser.add_argument("--sg-position-enabled", action="store_true")
    parser.add_argument("--no-sg-position", action="store_true")
    parser.add_argument("--sg-window-size", type=int, default=None)
    parser.add_argument("--sg-poly-order", type=int, default=None)
    parser.add_argument("--orientation-ema-enabled", action="store_true")
    parser.add_argument("--no-orientation-ema", action="store_true")
    parser.add_argument("--orientation-ema-alpha-x", type=float, default=None)
    parser.add_argument("--orientation-ema-alpha-y", type=float, default=None)
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("records/mock_smoothing_compare.png"),
        help="Output plot path. Use an empty string to skip saving.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open the matplotlib window.")
    return parser.parse_args()


def _default_enabled_config() -> DampingConfig:
    return DampingConfig(
        enabled=True,
        alpha=0.45,
        max_pos_step=0.020,
        max_ori_step=0.12,
        max_gripper_step=0.006,
        max_pos_velocity=0.35,
        max_ori_velocity=1.20,
        max_gripper_velocity=0.08,
        max_pos_acceleration=1.8,
        max_ori_acceleration=5.0,
        max_gripper_acceleration=0.35,
        max_pos_jerk=80.0,
        max_ori_jerk=220.0,
        max_gripper_jerk=20.0,
        gripper_min=0.0,
        gripper_max=0.08,
        max_missing_frames=10,
        sg_position_enabled=True,
        sg_window_size=21,
        sg_poly_order=2,
        orientation_ema_enabled=True,
        orientation_ema_alpha_x=0.15,
        orientation_ema_alpha_y=0.15,
    )


def _override_config(config: DampingConfig, args: argparse.Namespace) -> DampingConfig:
    values = {
        "enabled": True,
    }
    for field_name in (
        "alpha",
        "max_pos_step",
        "max_ori_step",
        "max_gripper_step",
        "max_pos_velocity",
        "max_ori_velocity",
        "max_gripper_velocity",
        "max_pos_acceleration",
        "max_ori_acceleration",
        "max_gripper_acceleration",
        "max_pos_jerk",
        "max_ori_jerk",
        "max_gripper_jerk",
        "max_missing_frames",
        "sg_window_size",
        "sg_poly_order",
        "orientation_ema_alpha_x",
        "orientation_ema_alpha_y",
    ):
        value = getattr(args, field_name)
        if value is not None:
            values[field_name] = value
    if args.sg_position_enabled:
        values["sg_position_enabled"] = True
    if args.no_sg_position:
        values["sg_position_enabled"] = False
    if args.orientation_ema_enabled:
        values["orientation_ema_enabled"] = True
    if args.no_orientation_ema:
        values["orientation_ema_enabled"] = False
    return replace(config, **values)


def _build_config(args: argparse.Namespace) -> DampingConfig:
    if args.postprocess_config is None:
        return _override_config(_default_enabled_config(), args)
    return _override_config(load_damping_config(args.postprocess_config), args)


def _mock_clean_trajectory(time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.zeros((len(time_s), 6), dtype=float)

    # Position is expressed in meters here because TrajectorySmoother uses SI units.
    pose[:, 0] = 0.18 + 0.08 * np.sin(2.0 * np.pi * 0.25 * time_s)
    pose[:, 1] = -0.04 + 0.05 * np.sin(2.0 * np.pi * 0.42 * time_s + 0.8)
    pose[:, 2] = 0.12 + 0.04 * np.sin(2.0 * np.pi * 0.18 * time_s + 1.4)
    pose[:, 3] = 0.25 * np.sin(2.0 * np.pi * 0.30 * time_s)
    pose[:, 4] = 0.18 * np.sin(2.0 * np.pi * 0.22 * time_s + 0.6)
    pose[:, 5] = 0.35 * np.sin(2.0 * np.pi * 0.16 * time_s + 1.2)

    # Add two deliberate operator-like step motions so the limiter behavior is visible.
    pose[time_s > time_s[-1] * 0.42, 0] += 0.07
    pose[time_s > time_s[-1] * 0.68, 4] -= 0.22

    gripper = 0.04 + 0.025 * np.sin(2.0 * np.pi * 0.35 * time_s + 0.5)
    gripper[time_s > time_s[-1] * 0.55] += 0.018
    return pose, np.clip(gripper, 0.0, 0.08)


def _mock_noisy_trajectory(
    clean_pose: np.ndarray,
    clean_gripper: np.ndarray,
    *,
    rng: np.random.Generator,
    pos_noise: float,
    ori_noise: float,
    gripper_noise: float,
    spike_count: int,
    nan_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_pose = clean_pose.copy()
    raw_gripper = clean_gripper.copy()
    raw_pose[:, :3] += rng.normal(0.0, pos_noise, size=raw_pose[:, :3].shape)
    raw_pose[:, 3:] += rng.normal(0.0, ori_noise, size=raw_pose[:, 3:].shape)
    raw_gripper += rng.normal(0.0, gripper_noise, size=raw_gripper.shape)

    valid_indices = np.arange(10, len(raw_pose) - 10)
    spike_indices = rng.choice(valid_indices, size=min(spike_count, len(valid_indices)), replace=False)
    for index in spike_indices:
        raw_pose[index, :3] += rng.normal(0.0, pos_noise * 8.0, size=3)
        raw_pose[index, 3:] += rng.normal(0.0, ori_noise * 8.0, size=3)
        raw_gripper[index] += float(rng.normal(0.0, gripper_noise * 8.0))

    nan_indices = rng.choice(valid_indices, size=min(nan_count, len(valid_indices)), replace=False)
    for index in nan_indices:
        channel = int(rng.integers(0, 7))
        if channel < 6:
            raw_pose[index, channel] = math.nan
        else:
            raw_gripper[index] = math.nan

    return raw_pose, raw_gripper, np.sort(spike_indices), np.sort(nan_indices)


def _run_smoother(
    raw_pose: np.ndarray,
    raw_gripper: np.ndarray,
    config: DampingConfig,
    *,
    cmd_dt: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    smoother = TrajectorySmoother(config, cmd_dt=cmd_dt)
    smoothed_pose = np.zeros_like(raw_pose)
    smoothed_gripper = np.zeros_like(raw_gripper)
    flags = {
        "limited": np.zeros(len(raw_pose), dtype=bool),
        "step_limited": np.zeros(len(raw_pose), dtype=bool),
        "velocity_limited": np.zeros(len(raw_pose), dtype=bool),
        "acceleration_limited": np.zeros(len(raw_pose), dtype=bool),
        "jerk_limited": np.zeros(len(raw_pose), dtype=bool),
        "command_limited": np.zeros(len(raw_pose), dtype=bool),
        "nonfinite_replaced": np.zeros(len(raw_pose), dtype=bool),
    }

    last_input_pose: np.ndarray | None = None
    last_input_gripper: float | None = None
    for index, (pose, gripper) in enumerate(zip(raw_pose, raw_gripper, strict=True)):
        input_pose = pose.copy()
        input_gripper = float(gripper)

        finite_pose = np.isfinite(input_pose)
        if not bool(np.all(finite_pose)):
            flags["nonfinite_replaced"][index] = True
            if last_input_pose is None:
                input_pose[~finite_pose] = 0.0
            else:
                input_pose[~finite_pose] = last_input_pose[~finite_pose]
        if not math.isfinite(input_gripper):
            flags["nonfinite_replaced"][index] = True
            input_gripper = 0.0 if last_input_gripper is None else last_input_gripper

        target = smoother.process(input_pose, input_gripper)
        smoothed_pose[index] = target.pose_6d
        smoothed_gripper[index] = target.gripper_pos
        flags["limited"][index] = target.limited
        flags["step_limited"][index] = target.step_limited
        flags["velocity_limited"][index] = target.velocity_limited
        flags["acceleration_limited"][index] = target.acceleration_limited
        flags["jerk_limited"][index] = target.jerk_limited
        flags["command_limited"][index] = target.command_limited

        last_input_pose = input_pose
        last_input_gripper = input_gripper

    return smoothed_pose, smoothed_gripper, flags


def _plot_result(
    time_s: np.ndarray,
    clean_pose: np.ndarray,
    clean_gripper: np.ndarray,
    raw_pose: np.ndarray,
    raw_gripper: np.ndarray,
    smoothed_pose: np.ndarray,
    smoothed_gripper: np.ndarray,
    flags: dict[str, np.ndarray],
    *,
    spike_indices: np.ndarray,
    nan_indices: np.ndarray,
    config: DampingConfig,
) -> object:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required: pip install matplotlib") from exc

    fig, axes = plt.subplots(5, 3, figsize=(18, 14), sharex=True)
    fig.suptitle("Mock Trajectory Smoothing Test")

    pos_labels = ("x", "y", "z")
    ori_labels = ("roll", "pitch", "yaw")
    for col, label in enumerate(pos_labels):
        ax = axes[0, col]
        ax.set_title(f"Position {label.upper()}")
        ax.plot(time_s, clean_pose[:, col] * 100.0, color="0.35", linewidth=1.0, label="clean")
        ax.plot(time_s, raw_pose[:, col] * 100.0, color="tab:red", alpha=0.55, label="raw mock")
        ax.plot(time_s, smoothed_pose[:, col] * 100.0, color="tab:blue", linewidth=1.6, label="smoothed")
        ax.set_ylabel("cm")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    for col, label in enumerate(ori_labels):
        ax = axes[1, col]
        ax.set_title(label)
        ax.plot(time_s, clean_pose[:, col + 3], color="0.35", linewidth=1.0, label="clean")
        ax.plot(time_s, raw_pose[:, col + 3], color="tab:red", alpha=0.55, label="raw mock")
        ax.plot(time_s, smoothed_pose[:, col + 3], color="tab:blue", linewidth=1.6, label="smoothed")
        ax.set_ylabel("rad")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[2, 0].set_title("Gripper")
    axes[2, 0].plot(time_s, clean_gripper * 100.0, color="0.35", linewidth=1.0, label="clean")
    axes[2, 0].plot(time_s, raw_gripper * 100.0, color="tab:red", alpha=0.55, label="raw mock")
    axes[2, 0].plot(time_s, smoothed_gripper * 100.0, color="tab:blue", linewidth=1.6, label="smoothed")
    axes[2, 0].set_ylabel("cm")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend(loc="best")

    axes[2, 1].set_title("Position Error Norm")
    axes[2, 1].plot(time_s, np.linalg.norm(raw_pose[:, :3] - clean_pose[:, :3], axis=1) * 100.0, label="raw-clean")
    axes[2, 1].plot(
        time_s,
        np.linalg.norm(smoothed_pose[:, :3] - clean_pose[:, :3], axis=1) * 100.0,
        label="smoothed-clean",
    )
    axes[2, 1].set_ylabel("cm")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend(loc="best")

    axes[2, 2].set_title("Orientation Error Norm")
    axes[2, 2].plot(time_s, np.linalg.norm(raw_pose[:, 3:] - clean_pose[:, 3:], axis=1), label="raw-clean")
    axes[2, 2].plot(
        time_s,
        np.linalg.norm(smoothed_pose[:, 3:] - clean_pose[:, 3:], axis=1),
        label="smoothed-clean",
    )
    axes[2, 2].set_ylabel("rad")
    axes[2, 2].grid(True, alpha=0.3)
    axes[2, 2].legend(loc="best")

    axes[3, 0].set_title("Injected Events")
    axes[3, 0].eventplot(
        [time_s[spike_indices], time_s[nan_indices]],
        lineoffsets=[1.0, 0.0],
        colors=["tab:orange", "tab:red"],
    )
    axes[3, 0].set_yticks([1.0, 0.0], ["spike", "NaN"])
    axes[3, 0].grid(True, alpha=0.3)

    flag_names = (
        "step_limited",
        "velocity_limited",
        "acceleration_limited",
        "jerk_limited",
        "command_limited",
        "nonfinite_replaced",
    )
    axes[3, 1].set_title("Limiter Flags")
    for offset, name in enumerate(flag_names):
        axes[3, 1].step(time_s, flags[name].astype(float) + offset, where="post", label=name)
    axes[3, 1].set_yticks([offset + 0.5 for offset in range(len(flag_names))], flag_names)
    axes[3, 1].grid(True, alpha=0.3)

    axes[3, 2].set_title("Smoothed Velocity Norm")
    velocity = np.gradient(smoothed_pose[:, :3], time_s, axis=0)
    axes[3, 2].plot(time_s, np.linalg.norm(velocity, axis=1), label="position velocity")
    axes[3, 2].set_ylabel("m/s")
    axes[3, 2].grid(True, alpha=0.3)
    axes[3, 2].legend(loc="best")

    axes[4, 0].axis("off")
    summary = [
        f"samples={len(time_s)}",
        f"duration={time_s[-1] - time_s[0]:.2f}s",
        f"enabled={config.enabled}",
        f"alpha={config.alpha}",
        f"limited={int(flags['limited'].sum())}",
        *(f"{name}={int(flags[name].sum())}" for name in flag_names),
    ]
    axes[4, 0].text(0.0, 1.0, "\n".join(summary), va="top", ha="left", family="monospace")
    axes[4, 1].axis("off")
    axes[4, 2].axis("off")

    for row in axes:
        for ax in row:
            if ax.axison:
                ax.set_xlabel("time (s)")
    fig.tight_layout()
    return fig


def main() -> int:
    args = _parse_args()
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if args.cmd_dt <= 0.0:
        raise ValueError("--cmd-dt must be positive")

    rng = np.random.default_rng(args.seed)
    time_s = np.arange(0.0, args.duration, args.cmd_dt)
    clean_pose, clean_gripper = _mock_clean_trajectory(time_s)
    raw_pose, raw_gripper, spike_indices, nan_indices = _mock_noisy_trajectory(
        clean_pose,
        clean_gripper,
        rng=rng,
        pos_noise=args.pos_noise,
        ori_noise=args.ori_noise,
        gripper_noise=args.gripper_noise,
        spike_count=args.spike_count,
        nan_count=args.nan_count,
    )
    config = _build_config(args)
    smoothed_pose, smoothed_gripper, flags = _run_smoother(
        raw_pose,
        raw_gripper,
        config,
        cmd_dt=args.cmd_dt,
    )

    fig = _plot_result(
        time_s,
        clean_pose,
        clean_gripper,
        raw_pose,
        raw_gripper,
        smoothed_pose,
        smoothed_gripper,
        flags,
        spike_indices=spike_indices,
        nan_indices=nan_indices,
        config=config,
    )

    if args.save:
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
