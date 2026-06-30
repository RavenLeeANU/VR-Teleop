"""Convert exported hand_tracking.csv rows into pose and binary gripper CSV.

Example:
    python -m quest_hand.convert_hand_tracking_csv hand_tracking.csv --output records/hand_tracking_pose_gripper.csv
    python -m quest_hand.convert_hand_tracking_csv hand_tracking.csv --hand left --basis rfu --closed-dist 0.02 --open-dist 0.06
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from hand_tracking_sdk.convert import (
    unity_left_to_flu_position,
    unity_left_to_flu_rotation_matrix,
    unity_left_to_rfu_position,
    unity_left_to_rfu_rotation_matrix,
)
from quest_hand.runtime_postprocess import make_rpy_continuous


JOINT_COUNT_PER_HAND = 26
SIDE_OUTPUT_FIELDS = [
    "active",
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "qx",
    "qy",
    "qz",
    "qw",
    "pinch_distance",
    "gripper_binary",
    "gripper_pos",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a Unity/Quest hand_tracking.csv export and write wrist pose plus "
            "binary pinch gripper commands."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("csv_path", type=Path, help="Input hand_tracking.csv path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("records/hand_tracking_pose_gripper.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--hand",
        choices=["left", "right", "both"],
        default="both",
        help="Which hand(s) to export.",
    )
    parser.add_argument(
        "--basis",
        choices=["rfu", "flu"],
        default="rfu",
        help="Output robot basis. Matches teleop_vr_send.py --basis.",
    )
    parser.add_argument(
        "--closed-dist",
        type=float,
        default=0.02,
        help="Pinch distance at or below this value is treated as closed.",
    )
    parser.add_argument(
        "--open-dist",
        type=float,
        default=0.06,
        help="Pinch distance at or above this value is treated as open.",
    )
    parser.add_argument(
        "--closed-pos",
        type=float,
        default=0.0,
        help="Output gripper_pos for closed state.",
    )
    parser.add_argument(
        "--open-pos",
        type=float,
        default=0.08,
        help="Output gripper_pos for open state.",
    )
    initial_group = parser.add_mutually_exclusive_group()
    initial_group.add_argument(
        "--initial-gripper-open",
        dest="initial_gripper_open",
        action="store_true",
        help="Use open state for distances between thresholds before a prior state exists.",
    )
    initial_group.add_argument(
        "--initial-gripper-closed",
        dest="initial_gripper_open",
        action="store_false",
        help="Use closed state for distances between thresholds before a prior state exists.",
    )
    parser.set_defaults(initial_gripper_open=True)
    return parser.parse_args()


def _to_float(value: str | None, *, default: float = math.nan) -> float:
    if value is None:
        return default
    text = value.strip()
    if text == "":
        return default
    return float(text)


def _to_int(value: str | None, *, default: int = 0) -> int:
    if value is None:
        return default
    text = value.strip()
    if text == "":
        return default
    return int(float(text))


def _active(row: dict[str, str], side: str) -> bool:
    return _to_int(row.get(f"{side}_active"), default=0) != 0


def _joint_index_by_name(row: dict[str, str], side: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for index in range(JOINT_COUNT_PER_HAND):
        name = row.get(f"{side}_joint{index}_name", "").strip().upper()
        if name:
            result[name] = index
    return result


def _joint_position(row: dict[str, str], side: str, index: int) -> np.ndarray:
    return np.asarray(
        [
            _to_float(row.get(f"{side}_joint{index}_pos_x")),
            _to_float(row.get(f"{side}_joint{index}_pos_y")),
            _to_float(row.get(f"{side}_joint{index}_pos_z")),
        ],
        dtype=float,
    )


def _joint_quaternion(row: dict[str, str], side: str, index: int) -> tuple[float, float, float, float]:
    return (
        _to_float(row.get(f"{side}_joint{index}_orientation_x")),
        _to_float(row.get(f"{side}_joint{index}_orientation_y")),
        _to_float(row.get(f"{side}_joint{index}_orientation_z")),
        _to_float(row.get(f"{side}_joint{index}_orientation_w")),
    )


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


def _convert_position(pos: np.ndarray, basis: str) -> np.ndarray:
    if basis == "flu":
        return np.asarray(unity_left_to_flu_position(float(pos[0]), float(pos[1]), float(pos[2])))
    return np.asarray(unity_left_to_rfu_position(float(pos[0]), float(pos[1]), float(pos[2])))


def _convert_rotation_matrix(quat_xyzw: tuple[float, float, float, float], basis: str) -> np.ndarray:
    qx, qy, qz, qw = quat_xyzw
    if basis == "flu":
        return np.asarray(unity_left_to_flu_rotation_matrix(qx, qy, qz, qw), dtype=float)
    return np.asarray(unity_left_to_rfu_rotation_matrix(qx, qy, qz, qw), dtype=float)


def _pinch_distance(row: dict[str, str], side: str, joint_by_name: dict[str, int]) -> float:
    thumb_index = joint_by_name.get("THUMB_TIP")
    index_index = joint_by_name.get("INDEX_TIP")
    if thumb_index is None or index_index is None:
        return math.nan
    thumb = _joint_position(row, side, thumb_index)
    index = _joint_position(row, side, index_index)
    if not bool(np.all(np.isfinite(thumb))) or not bool(np.all(np.isfinite(index))):
        return math.nan
    return float(np.linalg.norm(thumb - index))


def _binary_gripper_state(
    distance: float,
    *,
    previous_state: int,
    closed_dist: float,
    open_dist: float,
) -> int:
    # Hysteresis: small pinch is closed, large pinch is open, middle keeps last state.
    if math.isfinite(distance):
        if distance <= closed_dist:
            return 0
        if distance >= open_dist:
            return 1
    return previous_state


def _row_for_side(
    row: dict[str, str],
    *,
    side: str,
    basis: str,
    previous_rpy: np.ndarray | None,
    previous_gripper_state: int,
    closed_dist: float,
    open_dist: float,
    closed_pos: float,
    open_pos: float,
) -> tuple[dict[str, Any] | None, np.ndarray | None, int]:
    if not _active(row, side):
        return None, previous_rpy, previous_gripper_state

    joint_by_name = _joint_index_by_name(row, side)
    wrist_index = joint_by_name.get("WRIST")
    if wrist_index is None:
        return None, previous_rpy, previous_gripper_state

    wrist_pos = _joint_position(row, side, wrist_index)
    wrist_quat = _joint_quaternion(row, side, wrist_index)
    if not bool(np.all(np.isfinite(wrist_pos))) or not all(math.isfinite(value) for value in wrist_quat):
        return None, previous_rpy, previous_gripper_state

    pos = _convert_position(wrist_pos, basis)
    rot = _convert_rotation_matrix(wrist_quat, basis)
    rpy = make_rpy_continuous(_rotmat_to_rpy(rot), previous_rpy)
    pinch = _pinch_distance(row, side, joint_by_name)
    gripper_binary = _binary_gripper_state(
        pinch,
        previous_state=previous_gripper_state,
        closed_dist=closed_dist,
        open_dist=open_dist,
    )
    gripper_pos = open_pos if gripper_binary else closed_pos

    output = {
        "frame_number": row.get("frame_number", ""),
        "timestamp": row.get("timestamp", ""),
        "side": side,
        "active": 1,
        "x": pos[0],
        "y": pos[1],
        "z": pos[2],
        "roll": rpy[0],
        "pitch": rpy[1],
        "yaw": rpy[2],
        "qx": wrist_quat[0],
        "qy": wrist_quat[1],
        "qz": wrist_quat[2],
        "qw": wrist_quat[3],
        "pinch_distance": pinch,
        "gripper_binary": gripper_binary,
        "gripper_pos": gripper_pos,
    }
    return output, rpy, gripper_binary


def _prefixed_side_output(side: str, output: dict[str, Any] | None) -> dict[str, Any]:
    if output is None:
        return {f"{side}_{field}": "" for field in SIDE_OUTPUT_FIELDS}
    return {f"{side}_{field}": output[field] for field in SIDE_OUTPUT_FIELDS}


def convert_csv(
    input_path: Path,
    output_path: Path,
    *,
    hand: str,
    basis: str,
    closed_dist: float,
    open_dist: float,
    closed_pos: float,
    open_pos: float,
    initial_gripper_open: bool,
) -> int:
    if closed_dist >= open_dist:
        raise ValueError("--closed-dist must be smaller than --open-dist")

    sides = ("left", "right") if hand == "both" else (hand,)
    previous_rpy_by_side: dict[str, np.ndarray | None] = {side: None for side in sides}
    previous_gripper_by_side = {
        side: 1 if initial_gripper_open else 0 for side in sides
    }
    fieldnames = ["frame_number", "timestamp"]
    for side in sides:
        fieldnames.extend(f"{side}_{field}" for field in SIDE_OUTPUT_FIELDS)
    output_rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                output_row: dict[str, Any] = {
                    "frame_number": row.get("frame_number", ""),
                    "timestamp": row.get("timestamp", ""),
                }
                for side in sides:
                    output, next_rpy, next_gripper = _row_for_side(
                        row,
                        side=side,
                        basis=basis,
                        previous_rpy=previous_rpy_by_side[side],
                        previous_gripper_state=previous_gripper_by_side[side],
                        closed_dist=closed_dist,
                        open_dist=open_dist,
                        closed_pos=closed_pos,
                        open_pos=open_pos,
                    )
                    previous_rpy_by_side[side] = next_rpy
                    previous_gripper_by_side[side] = next_gripper
                    output_row.update(_prefixed_side_output(side, output))
                writer.writerow(output_row)
                output_rows += 1
    return output_rows


def main() -> int:
    args = _parse_args()
    rows = convert_csv(
        args.csv_path.expanduser().resolve(),
        args.output.expanduser().resolve(),
        hand=args.hand,
        basis=args.basis,
        closed_dist=args.closed_dist,
        open_dist=args.open_dist,
        closed_pos=args.closed_pos,
        open_pos=args.open_pos,
        initial_gripper_open=args.initial_gripper_open,
    )
    print(f"wrote {rows} rows to {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
