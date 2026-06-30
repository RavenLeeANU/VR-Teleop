"""Plot converted hand tracking pose/gripper CSV in 3D.

Example:
    python -m quest_hand.plot_hand_tracking_3d records/hand_tracking_pose_gripper.csv --save records/hand_tracking_3d.png --no-show
    python -m quest_hand.plot_hand_tracking_3d records/hand_tracking_pose_gripper.csv --gif records/hand_tracking_3d.gif --no-show
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot left/right wrist trajectories and binary gripper states in 3D.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path, help="CSV produced by quest_hand.convert_hand_tracking_csv.")
    parser.add_argument("--save", type=Path, default=None, help="Optional image output path.")
    parser.add_argument("--gif", type=Path, default=None, help="Optional animated GIF output path.")
    parser.add_argument("--no-show", action="store_true", help="Do not open a matplotlib window.")
    parser.add_argument(
        "--side",
        choices=["left", "right", "both"],
        default="both",
        help="Which hand trajectory to draw.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Draw every Nth trajectory sample. Use larger values for dense files.",
    )
    parser.add_argument(
        "--orientation-stride",
        type=int,
        default=40,
        help="Draw one orientation triad every N samples. Use 0 to disable orientation axes.",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.025,
        help="Length of each rendered orientation axis in position units.",
    )
    parser.add_argument("--fps", type=int, default=20, help="Animated GIF frames per second.")
    parser.add_argument(
        "--animation-stride",
        type=int,
        default=5,
        help="Use every Nth CSV row for GIF frames to keep file size manageable.",
    )
    parser.add_argument(
        "--trail",
        type=int,
        default=160,
        help="Number of recent samples shown as a bright tail in the GIF; use 0 for full history only.",
    )
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
    if text == "":
        return math.nan
    return float(text)


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    return int(float(text))


def _rpy_to_rotmat(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(float(roll)), math.sin(float(roll))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=float,
    )
    rot_y = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=float,
    )
    rot_z = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return rot_z @ rot_y @ rot_x


def _side_arrays(rows: list[dict[str, str]], side: str, stride: int) -> tuple[np.ndarray, np.ndarray]:
    positions: list[list[float]] = []
    rpy_values: list[list[float]] = []
    for index, row in enumerate(rows):
        if stride > 1 and index % stride != 0:
            continue
        active = _to_int(row.get(f"{side}_active"))
        if active != 1:
            continue
        pos = [
            _to_float(row.get(f"{side}_x")),
            _to_float(row.get(f"{side}_y")),
            _to_float(row.get(f"{side}_z")),
        ]
        rpy = [
            _to_float(row.get(f"{side}_roll")),
            _to_float(row.get(f"{side}_pitch")),
            _to_float(row.get(f"{side}_yaw")),
        ]
        if not all(math.isfinite(value) for value in pos + rpy):
            continue
        positions.append(pos)
        rpy_values.append(rpy)
    return np.asarray(positions, dtype=float), np.asarray(rpy_values, dtype=float)


def _gripper_points(
    rows: list[dict[str, str]],
    side: str,
    state: int,
    stride: int,
) -> np.ndarray:
    points: list[list[float]] = []
    for index, row in enumerate(rows):
        if stride > 1 and index % stride != 0:
            continue
        if _to_int(row.get(f"{side}_active")) != 1:
            continue
        if _to_int(row.get(f"{side}_gripper_binary")) != state:
            continue
        pos = [
            _to_float(row.get(f"{side}_x")),
            _to_float(row.get(f"{side}_y")),
            _to_float(row.get(f"{side}_z")),
        ]
        if all(math.isfinite(value) for value in pos):
            points.append(pos)
    return np.asarray(points, dtype=float)


def _side_positions_from_rows(rows: list[dict[str, str]], side: str) -> np.ndarray:
    positions: list[list[float]] = []
    for row in rows:
        if _to_int(row.get(f"{side}_active")) != 1:
            positions.append([math.nan, math.nan, math.nan])
            continue
        pos = [
            _to_float(row.get(f"{side}_x")),
            _to_float(row.get(f"{side}_y")),
            _to_float(row.get(f"{side}_z")),
        ]
        positions.append(pos if all(math.isfinite(value) for value in pos) else [math.nan, math.nan, math.nan])
    return np.asarray(positions, dtype=float)


def _side_gripper_from_rows(rows: list[dict[str, str]], side: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        state = _to_int(row.get(f"{side}_gripper_binary"))
        values.append(float(state) if state is not None else math.nan)
    return np.asarray(values, dtype=float)


def _finite_path(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3)
    mask = np.all(np.isfinite(points), axis=1)
    return points[mask]


def _set_equal_aspect(ax: object, all_positions: list[np.ndarray]) -> None:
    finite_positions = [pos for pos in all_positions if pos.size > 0]
    if not finite_positions:
        return
    stacked = np.vstack(finite_positions)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    centers = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.5)
    if radius <= 0.0:
        radius = 0.05
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass


def _draw_orientation_triads(
    ax: object,
    positions: np.ndarray,
    rpy_values: np.ndarray,
    *,
    stride: int,
    axis_length: float,
) -> None:
    if stride <= 0 or positions.size == 0:
        return
    colors = ("tab:red", "tab:green", "tab:blue")
    for pos, rpy in zip(positions[::stride], rpy_values[::stride], strict=True):
        rot = _rpy_to_rotmat(rpy)
        for axis_index, color in enumerate(colors):
            vec = rot[:, axis_index] * axis_length
            ax.quiver(
                pos[0],
                pos[1],
                pos[2],
                vec[0],
                vec[1],
                vec[2],
                color=color,
                linewidth=0.8,
                alpha=0.65,
            )


def plot_3d(
    rows: list[dict[str, str]],
    *,
    side: str,
    stride: int,
    orientation_stride: int,
    axis_length: float,
) -> object:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required: pip install matplotlib") from exc

    sides = ("left", "right") if side == "both" else (side,)
    side_colors = {"left": "tab:blue", "right": "tab:orange"}
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    all_positions: list[np.ndarray] = []

    for current_side in sides:
        positions, rpy_values = _side_arrays(rows, current_side, max(1, stride))
        all_positions.append(positions)
        if positions.size == 0:
            continue

        ax.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=side_colors[current_side],
            linewidth=1.5,
            label=f"{current_side} wrist trajectory",
        )
        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            positions[0, 2],
            color=side_colors[current_side],
            marker="o",
            s=60,
            label=f"{current_side} start",
        )
        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            positions[-1, 2],
            color=side_colors[current_side],
            marker="x",
            s=70,
            label=f"{current_side} end",
        )

        closed_points = _gripper_points(rows, current_side, 0, max(1, stride))
        open_points = _gripper_points(rows, current_side, 1, max(1, stride))
        if closed_points.size > 0:
            ax.scatter(
                closed_points[:, 0],
                closed_points[:, 1],
                closed_points[:, 2],
                color="tab:red",
                s=8,
                alpha=0.35,
                label=f"{current_side} gripper closed",
            )
        if open_points.size > 0:
            ax.scatter(
                open_points[:, 0],
                open_points[:, 1],
                open_points[:, 2],
                color="tab:green",
                s=8,
                alpha=0.25,
                label=f"{current_side} gripper open",
            )

        _draw_orientation_triads(
            ax,
            positions,
            rpy_values,
            stride=orientation_stride,
            axis_length=axis_length,
        )

    _set_equal_aspect(ax, all_positions)
    ax.set_title("Hand Wrist Trajectory, Orientation, and Binary Gripper")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def save_gif(
    rows: list[dict[str, str]],
    output_path: Path,
    *,
    side: str,
    fps: int,
    animation_stride: int,
    trail: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError as exc:
        raise RuntimeError("matplotlib and pillow are required: pip install matplotlib pillow") from exc

    sides = ("left", "right") if side == "both" else (side,)
    side_colors = {"left": "tab:blue", "right": "tab:orange"}
    positions_by_side = {current_side: _side_positions_from_rows(rows, current_side) for current_side in sides}
    gripper_by_side = {current_side: _side_gripper_from_rows(rows, current_side) for current_side in sides}
    frame_indices = list(range(0, len(rows), max(1, animation_stride)))
    if not frame_indices or frame_indices[-1] != len(rows) - 1:
        frame_indices.append(len(rows) - 1)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    _set_equal_aspect(ax, [_finite_path(values) for values in positions_by_side.values()])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    history_lines = {}
    trail_lines = {}
    current_points = {}
    for current_side in sides:
        (history_lines[current_side],) = ax.plot(
            [],
            [],
            [],
            color=side_colors[current_side],
            linewidth=1.0,
            alpha=0.28,
            label=f"{current_side} history",
        )
        (trail_lines[current_side],) = ax.plot(
            [],
            [],
            [],
            color=side_colors[current_side],
            linewidth=2.4,
            alpha=0.95,
            label=f"{current_side} recent trail",
        )
        current_points[current_side] = ax.scatter([], [], [], s=70, color=side_colors[current_side])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    def _set_scatter(scatter: object, point: np.ndarray, state: float) -> None:
        if not np.all(np.isfinite(point)):
            scatter._offsets3d = ([], [], [])
            return
        color = "tab:green" if state == 1.0 else "tab:red"
        scatter._offsets3d = ([point[0]], [point[1]], [point[2]])
        scatter.set_color(color)

    def update(frame_index: int) -> list[object]:
        row = rows[frame_index]
        artists: list[object] = []
        for current_side in sides:
            positions = positions_by_side[current_side]
            gripper = gripper_by_side[current_side]
            history = _finite_path(positions[: frame_index + 1])
            if history.size:
                history_lines[current_side].set_data(history[:, 0], history[:, 1])
                history_lines[current_side].set_3d_properties(history[:, 2])
            else:
                history_lines[current_side].set_data([], [])
                history_lines[current_side].set_3d_properties([])

            trail_start = 0 if trail <= 0 else max(0, frame_index + 1 - trail)
            recent = _finite_path(positions[trail_start : frame_index + 1])
            if recent.size:
                trail_lines[current_side].set_data(recent[:, 0], recent[:, 1])
                trail_lines[current_side].set_3d_properties(recent[:, 2])
            else:
                trail_lines[current_side].set_data([], [])
                trail_lines[current_side].set_3d_properties([])

            _set_scatter(current_points[current_side], positions[frame_index], gripper[frame_index])
            artists.extend(
                [
                    history_lines[current_side],
                    trail_lines[current_side],
                    current_points[current_side],
                ]
            )

        ax.set_title(
            "Hand Tracking 3D Timeline\n"
            f"row={frame_index + 1}/{len(rows)} "
            f"frame_number={row.get('frame_number', '')} "
            f"timestamp={row.get('timestamp', '')}"
        )
        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=1000.0 / max(1, fps),
        blit=False,
        repeat=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=max(1, fps)))
    plt.close(fig)
    print(f"saved gif to {output_path}")


def main() -> int:
    args = _parse_args()
    rows = _read_csv(args.csv_path.expanduser().resolve())
    if not rows:
        raise RuntimeError(f"No rows in {args.csv_path}")

    fig = plot_3d(
        rows,
        side=args.side,
        stride=max(1, args.stride),
        orientation_stride=args.orientation_stride,
        axis_length=args.axis_length,
    )
    if args.save is not None:
        save_path = args.save.expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"saved plot to {save_path}")
    if args.gif is not None:
        save_gif(
            rows,
            args.gif.expanduser().resolve(),
            side=args.side,
            fps=args.fps,
            animation_stride=args.animation_stride,
            trail=args.trail,
        )
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
