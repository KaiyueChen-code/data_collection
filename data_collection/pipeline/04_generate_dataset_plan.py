#!/usr/bin/env python3
import sys
import os
import argparse
import pickle
import json
import csv
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pose_util import mat_to_pose

def compute_rel_transform(pose: np.ndarray) -> tuple:

    world_frame = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
    pose = pose.copy()

    # Unity [x, y, z] -> right-handed [x, z, y]
    # 交换y和z轴：Unity的y（上）变成右手系的z（上），Unity的z（前）变成右手系的y（前）
    world_frame[:3] = np.array([world_frame[0], world_frame[2], world_frame[1]])
    pose[:3] = np.array([pose[0], pose[2], pose[1]])

    # 旋转矩阵Q用于交换y和z轴
    # Q将Unity坐标系中的旋转矩阵转换到右手系
    Q = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0.]])

    rot_base = Rotation.from_quat(world_frame[3:]).as_matrix()
    rot = Rotation.from_quat(pose[3:]).as_matrix()
    # 将旋转矩阵从Unity坐标系转换到右手系：Q @ rot @ Q.T
    rel_rot = Rotation.from_matrix(Q @ (rot_base.T @ rot) @ Q.T)
    # 将位置向量从Unity坐标系转换到右手系
    rel_pos = Rotation.from_matrix(Q @ rot_base.T @ Q.T).apply(pose[:3] - world_frame[:3])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    return rel_pos, rel_rot.as_quat()


def detect_demo_mode(demo_dir: Path):
    """
    Detect demo mode by checking for hand image folders.
    Supports both old format ({hand}_hand_img) and new format ({hand}_hand_visual_img).
    """
    # Check both old format (_img) and new format (_visual_img)
    left_exists = (demo_dir / 'left_hand_img').exists() or (demo_dir / 'left_hand_visual_img').exists()
    right_exists = (demo_dir / 'right_hand_img').exists() or (demo_dir / 'right_hand_visual_img').exists()
    
    if left_exists and right_exists:
        return "bimanual", ['left', 'right']
    elif left_exists:
        return "single", ['left']
    elif right_exists:
        return "single", ['right']
    
    return None, []


def check_aruco_files(demo_dir: Path, mode: str, hands: list) -> bool:
    if mode == 'single':
        pkl = demo_dir / f'tag_detection_{hands[0]}.pkl'
        return pkl.exists()
    
    for hand in hands:
        if not (demo_dir / f'tag_detection_{hand}.pkl').exists():
            return False
    return True


def find_image_folders(demo_dir: Path, mode: str, hands: list, use_tactile: bool):
    folders = {}
    
    for hand in hands:
        visual_folder = demo_dir / f'{hand}_hand_visual_img'
        raw_folder = demo_dir / f'{hand}_hand_img'
        img_folder = visual_folder if visual_folder.exists() else raw_folder
        if img_folder.exists():
            folders[f'{hand}_visual'] = img_folder
            
            if use_tactile:
                for side in ['left', 'right']:
                    tac_folder = demo_dir / f'{hand}_hand_{side}_tactile_img'
                    if tac_folder.exists():
                        folders[f'{hand}_hand_{side}_tactile'] = tac_folder
    
    return folders


def parse_timestamp(ts: str) -> float:
    """
    Parse timestamp strings produced by different recorders.
    Supports formats like:
      - 20250101_123045_123456   -> "%Y%m%d_%H%M%S_%f"
      - 2025.12.27_11.33.12.429903 -> "%Y.%m.%d_%H.%M.%S.%f"
      - 2025-12-27_11-33-12-429903 -> "%Y-%m-%d_%H-%M-%S-%f"
    Raises ValueError if none of the known formats match.
    """
    fmts = ("%Y%m%d_%H%M%S_%f", "%Y.%m.%d_%H.%M.%S.%f", "%Y-%m-%d_%H-%M-%S-%f")
    last_err = None
    for fmt in fmts:
        try:
            return datetime.strptime(ts, fmt).timestamp()
        except Exception as e:
            last_err = e

    # As a fallback, try removing dots and dashes then parse as compact form
    ts_compact = ts.replace(".", "").replace("-", "")
    try:
        return datetime.strptime(ts_compact, "%Y%m%d_%H%M%S_%f").timestamp()
    except Exception:
        raise ValueError(f"Unknown timestamp format: {ts}") from last_err


def get_image_times(demo_dir: Path, hand: str, latency: float) -> np.ndarray:
    csv_file = demo_dir / f'{hand}_hand_timestamps.csv'
    if not csv_file.exists():
        raise FileNotFoundError(f"Timestamps not found: {csv_file}")

    ram_times = []
    with csv_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "ram_time" not in reader.fieldnames:
            raise ValueError(f"'ram_time' column missing in {csv_file}")
        for row in reader:
            ts = row.get("ram_time")
            if not ts:
                continue
            try:
                ram_times.append(parse_timestamp(ts))
            except Exception:
                continue

    if not ram_times:
        raise RuntimeError(f"No valid ram_time entries in {csv_file}")

    times = np.asarray(ram_times, dtype=np.float64)
    return times - latency


def ensure_split_images(demo_dir: Path, hand: str):
    """
    Check if split image folders exist. 
    Images should already be cropped by 01_crop_img.py, so we just verify they exist.
    """
    visual_dir = demo_dir / f'{hand}_hand_visual_img'
    left_dir = demo_dir / f'{hand}_hand_left_tactile_img'
    right_dir = demo_dir / f'{hand}_hand_right_tactile_img'
    
    # Check if all three folders exist
    target_dirs = [visual_dir, left_dir, right_dir]
    all_exist = all(d.exists() and d.is_dir() for d in target_dirs)
    
    if all_exist:
        # Verify folders are not empty
        all_have_images = all(sum(1 for _ in d.glob('*.jpg')) > 0 for d in target_dirs)
        if all_have_images:
            return  # Folders exist and have images, no need to process
    
    # If folders don't exist, that's okay - they might be processed later by 01_crop_img.py
    # We don't create them here anymore since 01_crop_img.py handles the cropping
    return


def _ensure_hand_trajectory_csv(demo_dir: Path, hand: str, force_regenerate: bool = False):
    """
    生成hand的轨迹CSV文件。
    
    注意：由于左手设备使用右quest追踪，右手设备使用左quest追踪，
    所以需要互换：left_hand -> right_wrist, right_hand -> left_wrist
    
    Args:
        demo_dir: demo目录
        hand: 'left' 或 'right'
        force_regenerate: 如果True，删除旧CSV并重新生成
    """
    traj_file = demo_dir / 'pose_data' / f'{hand}_hand_trajectory.csv'
    
    # 删除旧CSV文件（如果需要重新生成）
    if force_regenerate and traj_file.exists():
        traj_file.unlink()
        print(f"  [INFO] Deleted old CSV: {traj_file.name}")
    
    if traj_file.exists() and not force_regenerate:
        return traj_file
    
    # 如果需要重新生成或文件不存在，开始生成
    if force_regenerate:
        print(f"  [INFO] Regenerating CSV for {hand} hand...")
    else:
        print(f"  [INFO] Generating CSV for {hand} hand...")
    
    task_dir = demo_dir.parent.parent
    all_traj_dir = task_dir / "all_trajectory"
    if not all_traj_dir.exists():
        raise FileNotFoundError(f"all_trajectory directory not found: {all_traj_dir}")
    
    json_files = sorted(all_traj_dir.glob("quest_poses_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No quest_poses_*.json found in {all_traj_dir}")
    
    # 互换逻辑：left_hand使用right_wrist数据，right_hand使用left_wrist数据
    if hand == 'left':
        quest_wrist_key = 'right_wrist'  # 左手设备用右quest追踪
    elif hand == 'right':
        quest_wrist_key = 'left_wrist'   # 右手设备用左quest追踪
    else:
        raise ValueError(f"Unknown hand: {hand}, expected 'left' or 'right'")
    
    # 批量读取所有JSON文件
    print(f"  [INFO] Loading {len(json_files)} JSON files for {hand} hand...")
    all_entries = []
    for json_path in json_files:
        try:
            with open(json_path, "r") as f:
                pose_list = json.load(f)
                all_entries.extend(pose_list)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file {json_path}: {e}. This might be due to corrupted data or invalid JSON format.")
        except Exception as e:
            raise RuntimeError(f"Error reading JSON file {json_path}: {e}")
    
    # 批量提取数据并转换
    print(f"  [INFO] Processing {len(all_entries)} pose entries...")
    timestamps = []
    poses = []
    
    for entry in all_entries:
        if quest_wrist_key not in entry:
            continue
        wrist = entry[quest_wrist_key]
        pos = wrist.get("position", {})
        rot = wrist.get("rotation", {})
        
        # 提取位置
        if isinstance(pos, dict):
            x = float(pos.get("x", 0.0))
            y = float(pos.get("y", 0.0))
            z = float(pos.get("z", 0.0))
        else:
            try:
                x, y, z = map(float, pos[:3])
            except Exception:
                x = y = z = 0.0
        
        # 提取旋转
        if isinstance(rot, dict):
            q_x = float(rot.get("x", 0.0))
            q_y = float(rot.get("y", 0.0))
            q_z = float(rot.get("z", 0.0))
            q_w = float(rot.get("w", 1.0))
        else:
            try:
                q_x, q_y, q_z, q_w = map(float, rot[:4])
            except Exception:
                q_x = q_y = q_z = 0.0
                q_w = 1.0
        
        # 转换坐标系
        rel_pos, rel_quat = compute_rel_transform(
            np.array([x, y, z, q_x, q_y, q_z, q_w], dtype=float)
        )
        
        ts = entry.get("timestamp_unix", entry.get("timestamp", 0.0))
        timestamps.append(ts)
        poses.append([rel_pos[0], rel_pos[1], rel_pos[2], rel_quat[0], rel_quat[1], rel_quat[2], rel_quat[3]])
    
    if not timestamps:
        raise RuntimeError(f"No valid pose data for hand '{hand}' (using {quest_wrist_key} from quest data)")
    
    # 转换为numpy数组并批量写入CSV
    pose_data_dir = demo_dir / "pose_data"
    pose_data_dir.mkdir(exist_ok=True)
    
    print(f"  [INFO] Writing CSV with {len(timestamps)} rows...")
    data_array = np.column_stack([timestamps, poses])
    
    # 使用numpy.savetxt批量写入（比逐行写入快很多）
    header = "timestamp,x,y,z,q_x,q_y,q_z,q_w"
    np.savetxt(
        traj_file,
        data_array,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.9f"
    )
    
    print(f"  [INFO] Generated trajectory CSV for {hand} hand (using {quest_wrist_key} data): {traj_file}")
    return traj_file


def process_hand_trajectory(demo_dir: Path, hand: str, target_times: np.ndarray, pose_latency: float, force_regenerate: bool = False):
    # 如果需要强制重新生成或文件不存在，调用生成函数
    traj_file = _ensure_hand_trajectory_csv(demo_dir, hand, force_regenerate=force_regenerate)

    # Read trajectory CSV without pandas
    try:
        data = np.genfromtxt(traj_file, delimiter=",", names=True)
    except Exception as e:
        raise RuntimeError(f"Failed to read trajectory CSV {traj_file}: {e}. The file might be corrupted or have invalid format.")
    if data.size == 0:
        raise RuntimeError(f"No pose samples for {hand}")

    # Ensure structured array behaves consistently for single-row case
    pose_times = np.asarray(data["timestamp"], dtype=np.float64) - pose_latency
    pos = np.column_stack([
        np.asarray(data["x"], dtype=np.float64),
        np.asarray(data["y"], dtype=np.float64),
        np.asarray(data["z"], dtype=np.float64),
    ])
    quat = np.column_stack([
        np.asarray(data["q_x"], dtype=np.float64),
        np.asarray(data["q_y"], dtype=np.float64),
        np.asarray(data["q_z"], dtype=np.float64),
        np.asarray(data["q_w"], dtype=np.float64),
    ])
    
    order = np.argsort(pose_times)
    pose_times = pose_times[order]
    pos = pos[order]
    quat = quat[order]
    
    pose_times, unique_idx = np.unique(pose_times, return_index=True)
    pos = pos[unique_idx]
    quat = quat[unique_idx]
    
    if len(pose_times) == 0:
        raise RuntimeError(f"No pose samples for {hand}")

    if len(pose_times) == 1:
        interp_pos = np.repeat(pos, len(target_times), axis=0)
        interp_quat = np.repeat(quat, len(target_times), axis=0)
    else:
        # BUGFIX: Ensure position and rotation use the same timestamps
        # Clip target times to trajectory range to avoid extrapolation
        clipped_times = np.clip(target_times, pose_times[0], pose_times[-1])
        
        # Position interpolation - use clipped_times for consistency
        interp_pos = np.column_stack([
            np.interp(clipped_times, pose_times, pos[:, i], left=pos[0, i], right=pos[-1, i])
            for i in range(3)
        ])
        
        # Rotation interpolation (SLERP) - also use clipped_times
        rot = Rotation.from_quat(quat)
        slerp = Slerp(pose_times, rot)
        interp_quat = slerp(clipped_times).as_quat()
    
    rot_mats = Rotation.from_quat(interp_quat).as_matrix()
    n_frames = len(target_times)
    pose_mat = np.zeros((n_frames, 4, 4), dtype=np.float32)
    pose_mat[:, 3, 3] = 1
    pose_mat[:, :3, 3] = interp_pos
    pose_mat[:, :3, :3] = rot_mats
    
    gripper_file = demo_dir / f'gripper_width_{hand}.csv'
    if not gripper_file.exists():
        raise FileNotFoundError(f"Gripper width not found: {gripper_file}")

    # Read gripper widths; file is written either by old pandas code or new numpy-based code.
    try:
        gdata = np.genfromtxt(gripper_file, delimiter=",", names=True)
    except Exception as e:
        raise RuntimeError(f"Failed to read gripper width CSV {gripper_file}: {e}. The file might be corrupted or have invalid format.")
    if gdata.size == 0:
        raise RuntimeError(f"No gripper width samples for {hand}")

    # Prefer 'width' column, fall back to 'gripper_width'
    if "width" in gdata.dtype.names:
        widths_arr = gdata["width"]
    elif "gripper_width" in gdata.dtype.names:
        widths_arr = gdata["gripper_width"]
    else:
        raise RuntimeError(f"No 'width' or 'gripper_width' column in {gripper_file}")

    widths = np.asarray(widths_arr, dtype=np.float32)
    if len(widths) < n_frames:
        raise ValueError(f"Not enough gripper width samples for {hand}")
    widths = widths[:n_frames]

    quest_pose = mat_to_pose(pose_mat)
    
    return {
        "quest_pose": quest_pose,
        "gripper_width": widths,
        "demo_start_pose": quest_pose[0],
        "demo_end_pose": quest_pose[-1]
    }


def create_camera_entries(image_folders: dict, demo_dir: Path, n_frames: int, mode: str, hands: list):
    cameras = []
    hand_idx_map = {'left': 0, 'right': 1}
    
    for usage_name, img_folder in image_folders.items():
        rel_path = img_folder.relative_to(demo_dir.parent)
        
        entry = {
            "image_folder": str(rel_path),
            "video_start_end": (0, n_frames),
            "usage_name": usage_name,
        }
        
        if mode == "single":
            entry["hand_position_idx"] = 0
        else:
            for hand in hands:
                if hand in usage_name:
                    entry["position"] = hand
                    entry["hand_position_idx"] = hand_idx_map[hand]
                    break
        
        cameras.append(entry)
    
    return cameras


def process_demo(demo_dir: Path, min_length: int, use_tactile: bool, visual_latency: float, pose_latency: float, force_regenerate_csv: bool = False):
    try:
        mode, hands = detect_demo_mode(demo_dir)
        if not mode:
            print(f"  [SKIP] {demo_dir.name}: No image folders")
            return None
        
        if not check_aruco_files(demo_dir, mode, hands):
            print(f"  [SKIP] {demo_dir.name}: Missing ArUco files")
            return None
        
        for hand in hands:
            ensure_split_images(demo_dir, hand)
        
        image_folders = find_image_folders(demo_dir, mode, hands, use_tactile)
        if not image_folders:
            print(f"  [SKIP] {demo_dir.name}: No valid images")
            return None
        
        image_times = {}
        for hand in hands:
            image_times[hand] = get_image_times(demo_dir, hand, visual_latency)
        n_frames = min(len(times) for times in image_times.values())
        if n_frames < min_length:
            print(f"  [SKIP] {demo_dir.name}: Too short ({n_frames}<{min_length})")
            return None
        for hand in hands:
            if len(image_times[hand]) != n_frames:
                print(f"  [WARN] {demo_dir.name}: trimming {hand} from {len(image_times[hand])} to {n_frames}")
                image_times[hand] = image_times[hand][:n_frames]
            else:
                image_times[hand] = image_times[hand][:n_frames]
        
        ref_times = image_times[hands[0]]
        if len(ref_times) > 1:
            duration = ref_times[-1] - ref_times[0]
            fps = (len(ref_times) - 1) / duration if duration > 0 else 25.0
        else:
            fps = 25.0
        
        grippers = []
        for hand in hands:
            data = process_hand_trajectory(demo_dir, hand, image_times[hand], pose_latency, force_regenerate=force_regenerate_csv)
            grippers.append(data)
        
        cameras = create_camera_entries(image_folders, demo_dir, n_frames, mode, hands)
        
        timestamps = ref_times
        
        print(f"  [OK] {demo_dir.name}: {len(hands)} hand(s), {len(cameras)} cam(s), {n_frames} frames")
        
        return {
            "episode_timestamps": timestamps,
            "grippers": grippers,
            "cameras": cameras,
            "demo_mode": mode,
            "demo_name": demo_dir.name,
            "n_frames": n_frames,
            "fps": fps
        }
    
    except Exception as e:
        print(f"  [ERROR] {demo_dir.name}: {e}")
        return None


def _process_demo_wrapper(args):
    """包装函数用于并行处理单个demo"""
    demo_dir, min_length, use_tactile, visual_latency, pose_latency, force_regenerate_csv = args
    try:
        plan = process_demo(demo_dir, min_length, use_tactile, visual_latency, pose_latency, force_regenerate_csv=force_regenerate_csv)
        return demo_dir, plan, None
    except Exception as e:
        return demo_dir, None, str(e)


def generate_plan(task_name: str, min_length: int = 10, use_tactile: bool = True,
                  visual_latency: float = 0.0, pose_latency: float = 0.0,
                  force_regenerate_csv: bool = False, num_workers: int = None):
    
    demos_dir = DATA_DIR / task_name / 'demos'
    output_file = DATA_DIR / task_name / 'dataset_plan.pkl'
    
    print(f"Task: {task_name}")
    print(f"Min length: {min_length}")
    print(f"Tactile: {use_tactile}")
    if force_regenerate_csv:
        print(f"Force regenerate CSV: YES (will delete old CSV files)")
    print()
    
    # 删除所有旧的CSV文件（如果需要）
    if force_regenerate_csv:
        print("[INFO] =========================================")
        print("[INFO] Force regenerate CSV: ENABLED")
        print("[INFO] Deleting old trajectory CSV files...")
        print("[INFO] =========================================")
        demo_dirs_preview = sorted([d for d in demos_dir.glob('demo_*') if d.is_dir()])
        deleted_count = 0
        for demo_dir in demo_dirs_preview:
            pose_data_dir = demo_dir / 'pose_data'
            if pose_data_dir.exists():
                for csv_file in pose_data_dir.glob('*_hand_trajectory.csv'):
                    csv_file.unlink()
                    deleted_count += 1
        print(f"[INFO] Deleted {deleted_count} old CSV files")
        print()
    
    demo_dirs = sorted([d for d in demos_dir.glob('demo_*') if d.is_dir()])
    print(f"Found {len(demo_dirs)} demos")
    
    # 设置并行worker数量
    if num_workers is None:
        num_workers = min(cpu_count(), len(demo_dirs), 8)  # 最多8个进程，避免过多进程竞争JSON文件
    print(f"Using {num_workers} parallel workers")
    print()
    
    plans = []
    stats = {
        'total': len(demo_dirs),
        'processed': 0,
        'skipped': 0,
        'single': 0,
        'bimanual': 0,
        'frames': 0,
        'duration': 0.0
    }
    
    # 准备并行处理的参数
    process_args = [
        (demo_dir, min_length, use_tactile, visual_latency, pose_latency, force_regenerate_csv)
        for demo_dir in demo_dirs
    ]
    
    # 并行处理所有demo
    if num_workers > 1 and len(demo_dirs) > 1:
        print(f"[INFO] Processing {len(demo_dirs)} demos in parallel...")
        with Pool(processes=num_workers) as pool:
            results = pool.map(_process_demo_wrapper, process_args)
        
        # 收集结果并统一打印（避免输出混乱）
        for demo_dir, plan, error in results:
            if error:
                print(f"  [ERROR] {demo_dir.name}: {error}")
                stats['skipped'] += 1
            elif plan:
                plans.append(plan)
                stats['processed'] += 1
                stats['frames'] += plan['n_frames']
                stats['duration'] += plan['n_frames'] / plan['fps']
                
                if plan['demo_mode'] == 'single':
                    stats['single'] += 1
                else:
                    stats['bimanual'] += 1
                
                print(f"  [OK] {demo_dir.name}: {len(plan.get('grippers', []))} hand(s), {len(plan.get('cameras', []))} cam(s), {plan['n_frames']} frames")
            else:
                stats['skipped'] += 1
    else:
        # 单进程处理（兼容性）
        print(f"[INFO] Processing {len(demo_dirs)} demos sequentially...")
        for demo_dir in demo_dirs:
            plan = process_demo(demo_dir, min_length, use_tactile, visual_latency, pose_latency, force_regenerate_csv=force_regenerate_csv)
            if plan:
                plans.append(plan)
                stats['processed'] += 1
                stats['frames'] += plan['n_frames']
                stats['duration'] += plan['n_frames'] / plan['fps']
                
                if plan['demo_mode'] == 'single':
                    stats['single'] += 1
                else:
                    stats['bimanual'] += 1
            else:
                stats['skipped'] += 1
    
    if not plans:
        print("\n[ERROR] No valid demos!")
        return
    
    with open(output_file, 'wb') as f:
        pickle.dump(plans, f)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total: {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Single-hand: {stats['single']}")
    print(f"Bimanual: {stats['bimanual']}")
    print(f"Total frames: {stats['frames']:,}")
    print(f"Total duration: {stats['duration']:.1f}s ({stats['duration']/60:.1f}min)")
    print(f"Success rate: {stats['processed']/stats['total']*100:.1f}%")
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True)
    parser.add_argument('--min_episode_length', type=int, default=10)
    parser.add_argument('--use_tactile_img', action='store_true', default=False)
    parser.add_argument('--visual_cam_latency', type=float, default=0.0)
    parser.add_argument('--pose_latency', type=float, default=0.0)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    generate_plan(
        task_name=args.task_name,
        min_length=args.min_episode_length,
        use_tactile=args.use_tactile_img,
        visual_latency=args.visual_cam_latency,
        pose_latency=args.pose_latency,
        force_regenerate_csv=False,
        num_workers=args.num_workers,
    )
