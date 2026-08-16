#!/usr/bin/env python3
import sys
import pickle
import argparse
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from scipy.interpolate import interp1d

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
sys.path.insert(0, str(PROJECT_ROOT))

from utils.cv_util import get_gripper_width


@dataclass
class WidthTask:
    pkl_file: Path
    csv_file: Path
    hand: str
    aruco_ids: Tuple[int, int]
def interpolate_widths_np(frames: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """
    使用 numpy + scipy 对抓手宽度进行插值与填补。
    行为大致等价于原先的 pandas 版本：
    - 仅在 >0 且非 NaN 的宽度上拟合
    - 对缺失值做线性插值
    - 对仍为 NaN 的位置赋默认值 0.05
    """
    widths = widths.astype(float)

    # 有效值：非 NaN 且 > 0
    valid_mask = (~np.isnan(widths)) & (widths > 0)
    valid_frames = frames[valid_mask]
    valid_widths = widths[valid_mask]

    if valid_frames.size == 0:
        # 完全没有有效值，全部设为默认值
        return np.full_like(widths, 0.05, dtype=float)
    if valid_frames.size == 1:
        # 只有一个有效值，全部填成这个值
        return np.full_like(widths, float(valid_widths[0]), dtype=float)

    # 尝试线性插值
    try:
        interp = interp1d(
            valid_frames,
            valid_widths,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        # 仅对无效位置进行插值
        invalid_mask = ~valid_mask
        if np.any(invalid_mask):
            widths[invalid_mask] = interp(frames[invalid_mask])
    except Exception:
        # 插值失败时，保留原始 widths（后面统一填默认值）
        pass

    # 对仍为 NaN 的位置填默认值
    widths = np.where(np.isnan(widths), 0.05, widths)
    return widths


def process_width(task: WidthTask) -> Path:
    detections = pickle.load(task.pkl_file.open('rb'))

    widths_list = []
    valid_count = 0

    for i, det in enumerate(detections):
        width = get_gripper_width(det['tag_dict'], task.aruco_ids[0], task.aruco_ids[1])
        if width is not None and not np.isnan(width) and width > 0:
            valid_count += 1
        widths_list.append(width)

    print(f"  {task.hand}: {valid_count}/{len(detections)} valid")

    # 构造帧号与宽度数组
    frames = np.arange(len(detections), dtype=int)
    widths = np.array(
        [np.nan if w is None else float(w) for w in widths_list],
        dtype=float,
    )
    widths = interpolate_widths_np(frames, widths)

    # 写出为 CSV：两列 frame,width
    data = np.column_stack([frames, widths])
    np.savetxt(
        task.csv_file,
        data,
        delimiter=",",
        header="frame,width",
        comments="",
        fmt=["%d", "%.8f"],
    )

    return task.csv_file


def create_tasks(demos_dir: Path, task_type: str, single_hand_side: str, left_ids: Tuple, right_ids: Tuple) -> List[WidthTask]:
    tasks = []
    
    for demo_dir in demos_dir.glob('demo_*'):
        if not demo_dir.is_dir():
            continue
        
        if task_type == "single":
            pkl = demo_dir / f'tag_detection_{single_hand_side}.pkl'
            if pkl.exists():
                tasks.append(WidthTask(
                    pkl_file=pkl,
                    csv_file=demo_dir / f'gripper_width_{single_hand_side}.csv',
                    hand=single_hand_side,
                    aruco_ids=left_ids if single_hand_side == 'left' else right_ids
                ))
        else:
            for hand, ids in [('left', left_ids), ('right', right_ids)]:
                pkl = demo_dir / f'tag_detection_{hand}.pkl'
                if pkl.exists():
                    tasks.append(WidthTask(
                        pkl_file=pkl,
                        csv_file=demo_dir / f'gripper_width_{hand}.csv',
                        hand=hand,
                        aruco_ids=ids
                    ))
    
    return tasks


def run_width_calculation(task_name: str, task_type: str, single_hand_side: str,
                          left_aruco_left_id: int, left_aruco_right_id: int,
                          right_aruco_left_id: int, right_aruco_right_id: int):
    left_ids = (left_aruco_left_id, left_aruco_right_id)
    right_ids = (right_aruco_left_id, right_aruco_right_id)
    
    demos_dir = DATA_DIR / task_name / "demos"
    tasks = create_tasks(demos_dir, task_type, single_hand_side, left_ids, right_ids)
    
    print(f"[{task_type}] Processing {len(tasks)} tasks")
    
    if not tasks:
        print("[ERROR] No tasks!")
        return
    
    saved = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_width, t): t for t in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                csv_file = future.result()
                saved.append(csv_file)
            except Exception as e:
                task = futures[future]
                print(f"[ERROR] {task.pkl_file.name}: {e}")
    
    print(f"\n[DONE] {len(saved)}/{len(tasks)} successful")

    if saved:
        total_frames = 0
        total_valid = 0
        for csv in sorted(saved):
            # 读取我们自己写出的 CSV（含 header）
            arr = np.genfromtxt(csv, delimiter=",", names=True)
            if arr.size == 0:
                continue

            widths = arr["width"]
            # 统一成 1D 数组
            widths = np.atleast_1d(widths)

            mask_valid = (~np.isnan(widths)) & (widths > 0)
            n_frames = widths.size
            n_valid = int(mask_valid.sum())

            total_frames += n_frames
            total_valid += n_valid
            ratio = (n_valid / n_frames * 100) if n_frames > 0 else 0.0
            print(f"  {csv.name}: {n_frames} frames, {n_valid} valid ({ratio:.1f}%)")

        if total_frames > 0:
            total_ratio = total_valid / total_frames * 100
            print(f"\nTotal: {total_frames} frames, {total_valid} valid ({total_ratio:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--task_type', type=str, default='bimanual', choices=['single', 'bimanual'])
    parser.add_argument('--single_hand_side', type=str, default='left', choices=['left', 'right'])
    parser.add_argument('--left_aruco_left_id', type=int, default=0)
    parser.add_argument('--left_aruco_right_id', type=int, default=1)
    parser.add_argument('--right_aruco_left_id', type=int, default=2)
    parser.add_argument('--right_aruco_right_id', type=int, default=3)
    args = parser.parse_args()
    run_width_calculation(
        task_name=args.task_name,
        task_type=args.task_type,
        single_hand_side=args.single_hand_side,
        left_aruco_left_id=args.left_aruco_left_id,
        left_aruco_right_id=args.left_aruco_right_id,
        right_aruco_left_id=args.right_aruco_left_id,
        right_aruco_right_id=args.right_aruco_right_id,
    )
