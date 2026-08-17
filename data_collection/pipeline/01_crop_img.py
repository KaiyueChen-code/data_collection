#!/usr/bin/env python3

import sys
import os
import argparse
from pathlib import Path
import re
from tqdm import tqdm
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
sys.path.append(str(PROJECT_ROOT))

from utils.cv_util import resize_center_crop


def find_demos_with_images(demos_dir: Path, task_type: str, single_hand_side: str):
    """Find all demo directories that have image folders."""
    demo_dirs = []
    for demo_dir in demos_dir.glob('demo_*'):
        if not demo_dir.is_dir():
            continue
        
        if task_type == "single":
            if (demo_dir / f'{single_hand_side}_hand_img').exists():
                demo_dirs.append(demo_dir)
        else:
            if (demo_dir / 'left_hand_img').exists() or (demo_dir / 'right_hand_img').exists():
                demo_dirs.append(demo_dir)
    
    return sorted(demo_dirs)


def _crop_images_wrapper(args):
    """Wrapper function for multiprocessing."""
    demo_dir, hand, visual_out_res, tactile_out_res = args
    return crop_images_for_hand(Path(demo_dir), hand, visual_out_res, tactile_out_res)


def crop_images_for_hand(demo_dir: Path, hand: str,
                         visual_out_res=(224, 224), tactile_out_res=(224, 224)):
    """
    Crop images for a single hand and resize to target resolution.

    Args:
        demo_dir: Demo directory path
        hand: 'left' or 'right'
        visual_out_res: (width, height) target resolution for visual images
        tactile_out_res: (width, height) target resolution for tactile images

    Returns:
        tuple: (success: bool, demo_name: str, hand: str, total: int, success_count: int, message: str)
    """
    demo_dir = Path(demo_dir)  # Ensure it's a Path object
    raw_dir = demo_dir / f'{hand}_hand_img'
    if not raw_dir.exists():
        return (False, demo_dir.name, hand, 0, 0, f"{hand}_hand_img folder not found")
    
    # Create output directories
    visual_dir = demo_dir / f'{hand}_hand_visual_img'
    left_tactile_dir = demo_dir / f'{hand}_hand_left_tactile_img'
    right_tactile_dir = demo_dir / f'{hand}_hand_right_tactile_img'
    
    # Create directories if they don't exist
    visual_dir.mkdir(parents=True, exist_ok=True)
    left_tactile_dir.mkdir(parents=True, exist_ok=True)
    right_tactile_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort image files by numeric ID in filename if possible
    raw_files = sorted(
        raw_dir.glob('*.jpg'),
        key=lambda p: int(re.search(r'(\d+)(?=\.jpg$)', p.name).group(1))
        if re.search(r'(\d+)(?=\.jpg$)', p.name) else p.name
    )
    
    if not raw_files:
        return (False, demo_dir.name, hand, 0, 0, "No JPG images found")
    
    # Image dimensions
    CROP_WIDTH = 1280
    TOTAL_WIDTH = 3840
    
    success_count = 0
    error_count = 0
    for img_path in raw_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            error_count += 1
            continue
        
        h, w = img.shape[:2]
        
        # Verify image dimensions
        if w != TOTAL_WIDTH or h != 800:
            # Continue anyway, but use actual dimensions
            if w < CROP_WIDTH * 3:
                error_count += 1
                continue
        
        # Crop into three parts: left_tactile, visual, right_tactile
        left_tactile = img[:, 0:CROP_WIDTH]
        visual = img[:, CROP_WIDTH:2*CROP_WIDTH]
        right_tactile = img[:, 2*CROP_WIDTH:3*CROP_WIDTH]
        
        # Keep all cropped panels in their original camera orientation.
        left_tactile_final = left_tactile
        visual_final = visual
        right_tactile_final = right_tactile

        # Method A: preserve aspect ratio, resize, then center crop.
        visual_final = resize_center_crop(visual_final, visual_out_res)
        left_tactile_final = resize_center_crop(left_tactile_final, tactile_out_res)
        right_tactile_final = resize_center_crop(right_tactile_final, tactile_out_res)

        # Save cropped images with the same filename
        left_tactile_path = left_tactile_dir / img_path.name
        visual_path = visual_dir / img_path.name
        right_tactile_path = right_tactile_dir / img_path.name

        cv2.imwrite(str(left_tactile_path), left_tactile_final)
        cv2.imwrite(str(visual_path), visual_final)
        cv2.imwrite(str(right_tactile_path), right_tactile_final)
        
        success_count += 1
    
    message = f"{success_count}/{len(raw_files)} images processed"
    if error_count > 0:
        message += f", {error_count} errors"
    
    return (True, demo_dir.name, hand, len(raw_files), success_count, message)


def main(task_name: str, task_type: str, single_hand_side: str = "left",
         visual_out_res=(224, 224), tactile_out_res=(224, 224),
         num_workers: int = None):
    """Main function to crop images for all demos."""
    demos_dir = DATA_DIR / task_name / "demos"
    
    if not demos_dir.exists():
        print(f"[ERROR] Demos directory not found: {demos_dir}")
        return
    
    demo_dirs = find_demos_with_images(demos_dir, task_type, single_hand_side)
    
    if not demo_dirs:
        print(f"[WARN] No demos found with image folders in {demos_dir}")
        return
    
    # Generate task list: (demo_dir, hand, visual_out_res, tactile_out_res) tuples
    tasks = []
    for demo_dir in demo_dirs:
        if task_type == "single":
            tasks.append((str(demo_dir), single_hand_side, visual_out_res, tactile_out_res))
        else:
            # Check which hands exist
            if (demo_dir / 'left_hand_img').exists():
                tasks.append((str(demo_dir), 'left', visual_out_res, tactile_out_res))
            if (demo_dir / 'right_hand_img').exists():
                tasks.append((str(demo_dir), 'right', visual_out_res, tactile_out_res))

    print(f"[INFO] Found {len(demo_dirs)} demos to process")
    print(f"[INFO] Task type: {task_type}")
    if task_type == "single":
        print(f"[INFO] Processing {single_hand_side} hand only")
    else:
        print(f"[INFO] Processing both left and right hands")
    print(f"[INFO] Total tasks: {len(tasks)}")
    print(f"[INFO] Output resolution: visual={visual_out_res}, tactile={tactile_out_res}")
    print("[INFO] Resize mode: A (aspect-preserving resize + center crop)")

    # Set default number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), len(tasks))
    print(f"[INFO] Using {num_workers} worker processes")

    # Process tasks in parallel
    if num_workers == 1:
        # Single process mode (for debugging)
        results = []
        for task_args in tqdm(tasks, desc="Processing tasks"):
            result = crop_images_for_hand(Path(task_args[0]), task_args[1],
                                          task_args[2], task_args[3])
            results.append(result)
    else:
        # Multi-process mode
        print(f"[INFO] Starting parallel processing with {num_workers} workers...")
        with Pool(processes=num_workers) as pool:
            # Use imap for real-time progress tracking
            results = []
            with tqdm(total=len(tasks), desc="Processing tasks") as pbar:
                for result in pool.imap(_crop_images_wrapper, tasks):
                    results.append(result)
                    pbar.update(1)
    
    # Print results summary
    print("\n" + "="*60)
    print("Processing Summary:")
    print("="*60)
    success_count = 0
    total_images = 0
    processed_images = 0
    
    for success, demo_name, hand, total, success_count_task, message in results:
        status = "✓" if success else "✗"
        print(f"{status} {demo_name}/{hand}_hand: {message}")
        if success:
            success_count += 1
            total_images += total
            processed_images += success_count_task
    
    print("="*60)
    print(f"[SUCCESS] Completed {success_count}/{len(results)} tasks")
    print(f"[SUCCESS] Processed {processed_images}/{total_images} images")
    print(f"[SUCCESS] Image cropping completed for {len(demo_dirs)} demos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop 3840x800 images into three 1280x800 images (left_tactile, visual, right_tactile) and resize to target resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--task_name', type=str, required=True, help='Task name (matches data/raw/{task_name}/)')
    parser.add_argument('--task_type', type=str, default='bimanual', choices=['single', 'bimanual'])
    parser.add_argument('--single_hand_side', type=str, default='left', choices=['left', 'right'],
                        help='Which hand to process (only used when task_type=single)')
    parser.add_argument('--visual_out_res', type=int, nargs=2, default=[224, 224], metavar=('W', 'H'),
                        help='Target resolution for visual images (default: 224 224)')
    parser.add_argument('--tactile_out_res', type=int, nargs=2, default=[224, 224], metavar=('W', 'H'),
                        help='Target resolution for tactile images (default: 224 224)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help=f'Number of worker processes (default: min(CPU count, number of tasks), max: {cpu_count()})')
    args = parser.parse_args()

    main(
        task_name=args.task_name,
        task_type=args.task_type,
        single_hand_side=args.single_hand_side,
        visual_out_res=tuple(args.visual_out_res),
        tactile_out_res=tuple(args.tactile_out_res),
        num_workers=args.num_workers,
    )
